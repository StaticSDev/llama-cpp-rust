//! Rust port of the hybrid memory variants.
//!
//! Covers `llama_memory_hybrid`, `llama_memory_hybrid_idx` and
//! `llama_memory_hybrid_iswa` from `llama-memory-hybrid*.cpp/.h`.  A hybrid
//! memory pairs an attention KV cache with a recurrent state cache.  The
//! indexer variant adds a third cache that holds one indexer key per token,
//! plus pooling of those keys into fixed-size groups.
//!
//! The recurrent and ISWA caches are stand-ins with the surface the hybrid
//! classes use: cell bookkeeping is shared with `crate::kv_cache`, and state
//! rows are kept in plain byte buffers.  All cache operations stay in Rust,
//! with no FFI or unsafe code.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::batch::{BatchAllocr, Ubatch as BatchUbatch};
use crate::core::{Error, Result, SeqId};
use crate::kv_cache::{
    memory_status_is_fail, CellIdx, GgmlType, HParams, IoRead, IoWrite, KvCache, KvCacheParams,
    KvCells, LSeqId, MemoryStatus, Model, NonCausalType, Pos, RopeType, SlotInfo,
    SlotInfoVec, StreamCopyInfo, SwaType, Tensor, TensorContext, UBatch,
};

const LLAMA_MAX_SEQ: usize = 512;

// `layer_filter_cb` is a plain function pointer in the cache params, so the
// default filters read the layer kind through this per-thread slot while a
// hybrid memory is being constructed.
thread_local! {
    static RECR_LAYER_FILTER: Cell<fn(u32) -> bool> = const { Cell::new(|_| false) };
}

fn filter_attn_default(il: u32) -> bool {
    !RECR_LAYER_FILTER.with(|f| f.get())(il)
}

fn filter_recr_default(il: u32) -> bool {
    RECR_LAYER_FILTER.with(|f| f.get())(il)
}

/// `llama_memory_hybrid_idx::POS_CLEAN`: no sequence has been staled.
pub const POS_CLEAN: Pos = Pos::MAX;

/// `llama_memory_status_combine`.
fn memory_status_combine(s0: MemoryStatus, s1: MemoryStatus) -> MemoryStatus {
    if memory_status_is_fail(s0) {
        return s0;
    }
    if memory_status_is_fail(s1) {
        return s1;
    }
    if s0 == MemoryStatus::Success || s1 == MemoryStatus::Success {
        return MemoryStatus::Success;
    }
    MemoryStatus::NoUpdate
}

fn ggml_pad(x: u32, n: u32) -> u32 {
    ((x + n - 1) / n) * n
}

/// The last padded pool is always unused.
fn kpool_pad(n_pool: u32) -> u32 {
    64u32.max(ggml_pad(n_pool + 1, 64))
}

fn write_u32<W: IoWrite + ?Sized>(io: &mut W, v: u32) {
    io.write_bytes(&v.to_le_bytes());
}

fn write_i32<W: IoWrite + ?Sized>(io: &mut W, v: i32) {
    io.write_bytes(&v.to_le_bytes());
}

fn read_u32<R: IoRead + ?Sized>(io: &mut R) -> u32 {
    let mut b = [0u8; 4];
    if io.read_bytes(&mut b) {
        u32::from_le_bytes(b)
    } else {
        0
    }
}

fn read_i32<R: IoRead + ?Sized>(io: &mut R) -> i32 {
    let mut b = [0u8; 4];
    if io.read_bytes(&mut b) {
        i32::from_le_bytes(b)
    } else {
        0
    }
}

/// Local copy of the port-wide f32 -> f16 conversion.
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7fff_ff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (14 - exp);
        return sign | (m as u16);
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | (((exp as u16) << 10) | ((mant >> 13) as u16))
}

/// Convert the batch allocator's ubatch to the cache-facing representation.
fn to_cache_ubatch(src: &BatchUbatch) -> UBatch {
    let token = if src.token.is_empty() {
        None
    } else {
        Some(src.token.clone())
    };
    UBatch {
        n_tokens: src.n_tokens as usize,
        n_seqs: src.n_seqs as usize,
        n_seqs_unq: src.n_seqs_unq as usize,
        seq_id_unq: src.seq_id_unq.iter().map(|&s| s as LSeqId).collect(),
        n_seq_id: src.n_seq_id.iter().map(|&n| n as usize).collect(),
        seq_id: src
            .seq_id
            .iter()
            .map(|ids| ids.iter().map(|&s| s as LSeqId).collect())
            .collect(),
        pos: src.pos.clone(),
        token,
        type_embd: src.type_.iter().map(|&v| v != 0).collect(),
        n_pos_per_embd: src.n_pos,
        equal_seqs: src.equal_seqs(),
        mixed: src.is_mixed(),
    }
}

/// Split a batch the same way the hybrid classes do: the recurrent rollback
/// pattern keeps the trailing `1 + n_rs_seq` tokens of a sequence together.
fn split_ubatches(
    balloc: &mut BatchAllocr,
    n_ubatch: u32,
    embd_all: bool,
    unified: bool,
    n_rs_seq: SeqId,
) -> Vec<UBatch> {
    balloc.split_reset();

    let n_keep_tail = if n_rs_seq > 0 { n_rs_seq + 1 } else { 0 };
    let mut res = Vec::new();
    loop {
        let part = if embd_all {
            balloc.split_seq(n_ubatch)
        } else {
            balloc.split_equal(n_ubatch, !unified, n_keep_tail)
        };
        if part.n_tokens == 0 {
            break;
        }
        res.push(to_cache_ubatch(&part));
    }

    if balloc.get_n_used() < balloc.get_n_tokens() as u32 {
        return Vec::new();
    }
    res
}

/// `ggml_get_rows` as a plain tensor copy. Out-of-range rows (the sentinel)
/// read row 0, which is always a valid cell in the indexer cache.
fn gather_rows(src: &Tensor, idxs: &[i32]) -> Tensor {
    let row = src.nb[1].max(1);
    let n_rows = src.ne[1].max(src.ne[2]).max(1);
    let mut dst = Tensor::new_2d(src.ty, src.ne[0], idxs.len().max(1));
    let mut data = vec![0u8; row * idxs.len().max(1)];
    {
        let s = src.storage.borrow();
        for (i, &idx) in idxs.iter().enumerate() {
            let r = if idx >= 0 && (idx as usize) < n_rows {
                idx as usize
            } else {
                0
            };
            let off = src.offset + r * row;
            if off + row <= s.len() {
                data[i * row..(i + 1) * row].copy_from_slice(&s[off..off + row]);
            }
        }
    }
    dst.set_bytes(&data);
    dst
}

// ---------------------------------------------------------------------------
// Recurrent cache stand-in
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RecurrentParams {
    pub hparams: HParams,
    pub model: Model,
    pub type_r: GgmlType,
    pub type_s: GgmlType,
    /// Number of elements in one recurrent state row.
    pub rs_size: u32,
    pub n_seq_max: SeqId,
    pub n_rs_seq: SeqId,
    pub rs_replay: bool,
    pub offload: bool,
    pub no_alloc: bool,
    pub filter: Option<fn(u32) -> bool>,
}

impl Default for RecurrentParams {
    fn default() -> Self {
        Self {
            hparams: HParams::default(),
            model: Model {
                arch: crate::kv_cache::Arch::Other,
                attn_rot: true,
                lid_k_hdmd: false,
                dev_buft: "CPU".to_string(),
                layers: Vec::new(),
            },
            type_r: GgmlType::F32,
            type_s: GgmlType::F32,
            rs_size: 64,
            n_seq_max: 1,
            n_rs_seq: 0,
            rs_replay: false,
            offload: false,
            no_alloc: false,
            filter: None,
        }
    }
}

/// The hybrid-facing part of `llama_memory_recurrent`.
pub struct RecurrentMemory {
    params: RecurrentParams,
    n_stream: SeqId,
    n_rs_seq: SeqId,
    cells: Vec<KvCells>,
    heads: Vec<CellIdx>,
    states: Vec<Vec<u8>>,
    seq_to_stream: Vec<SeqId>,
    state_row: usize,
    pending: Vec<SlotInfo>,
}

impl RecurrentMemory {
    pub fn new(params: RecurrentParams) -> Result<Self> {
        let n_seq_max = params.n_seq_max.max(1);
        let n_stream = n_seq_max;
        let n_cells = params.rs_size.max(1);
        let state_row = params.rs_size.max(1) as usize * params.type_s.type_size().max(1);
        let mut cells = Vec::with_capacity(n_stream as usize);
        for _ in 0..n_stream {
            let mut c = KvCells::new();
            c.resize(n_cells);
            cells.push(c);
        }
        let mut seq_to_stream = vec![0u32; LLAMA_MAX_SEQ];
        if n_stream > 1 {
            seq_to_stream = (0..n_stream).collect();
        }
        Ok(Self {
            params,
            n_stream,
            n_rs_seq: 0,
            cells,
            heads: vec![0; n_stream as usize],
            states: vec![vec![0u8; state_row * n_cells as usize]; n_stream as usize],
            seq_to_stream,
            state_row,
            pending: Vec::new(),
        })
    }

    pub fn n_rs_seq(&self) -> SeqId {
        self.n_rs_seq
    }

    pub fn get_n_stream(&self) -> SeqId {
        self.n_stream
    }

    fn stream(&self, seq_id: LSeqId) -> usize {
        debug_assert!(seq_id >= 0 && (seq_id as usize) < LLAMA_MAX_SEQ);
        self.seq_to_stream[seq_id as usize] as usize
    }

    /// Prepare the recurrent states. Cell snapshots are taken before each
    /// ubatch and put back afterwards, so the caller can commit them via
    /// `init_batch` once all caches agree on the split.
    pub fn prepare(&mut self, ubatches: &[UBatch]) -> bool {
        let mut slots = Vec::with_capacity(ubatches.len());
        let mut saved: Vec<(SlotInfo, Vec<KvCells>, Vec<CellIdx>)> = Vec::with_capacity(ubatches.len());

        for ubatch in ubatches {
            let mut sinfo = SlotInfo::default();
            if self.n_stream > 1 {
                if ubatch.n_seqs_unq == 0 || ubatch.n_tokens % ubatch.n_seqs_unq != 0 {
                    return false;
                }
            }
            let n_per_stream = ubatch.n_tokens / self.n_stream as usize;
            if n_per_stream == 0 {
                continue;
            }

            let mut ok = true;
            for s in 0..self.n_stream as usize {
                let size = self.cells[s].size() as usize;
                let start = self.heads[s] as usize;
                let end = start + n_per_stream;
                if end > size {
                    ok = false;
                    break;
                }
                sinfo.strm.push(s as SeqId);
                sinfo.idxs.push((start..end).map(|i| i as CellIdx).collect());
            }
            if !ok || sinfo.strm.is_empty() {
                return false;
            }
            sinfo.s0 = sinfo.strm[0];
            sinfo.s1 = *sinfo.strm.last().unwrap();

            let old = sinfo
                .strm
                .iter()
                .zip(sinfo.idxs.iter())
                .map(|(s, idxs)| self.cells[*s as usize].cp_idxs(idxs))
                .collect();
            let heads_old = self.heads.clone();
            self.apply_ubatch(&sinfo, ubatch);
            saved.push((sinfo.clone(), old, heads_old));
            slots.push(sinfo);
        }

        for (sinfo, old, heads_old) in saved.iter().rev() {
            for (s, cells) in sinfo.strm.iter().zip(old.iter()) {
                self.cells[*s as usize].set_idxs(&sinfo.idxs[*s as usize], cells);
            }
            self.heads.clone_from_slice(heads_old);
        }
        self.pending = slots;
        true
    }

    pub fn init_batch(cache: &Rc<RefCell<Self>>, ubatches: Vec<UBatch>) -> RecurrentContext {
        let slots = std::mem::take(&mut cache.borrow_mut().pending);
        let status = if slots.len() == ubatches.len() {
            MemoryStatus::Success
        } else {
            MemoryStatus::FailedPrepare
        };
        RecurrentContext::new(cache, RecurrentKind::Batch(slots), ubatches, status)
    }

    pub fn init_full(cache: &Rc<RefCell<Self>>) -> RecurrentContext {
        RecurrentContext::new(cache, RecurrentKind::Full, Vec::new(), MemoryStatus::Success)
    }

    pub fn init_update(cache: &Rc<RefCell<Self>>, optimize: bool) -> RecurrentContext {
        let status = if optimize {
            MemoryStatus::Success
        } else {
            MemoryStatus::NoUpdate
        };
        RecurrentContext::new(cache, RecurrentKind::Update, Vec::new(), status)
    }

    pub fn apply_ubatch(&mut self, sinfo: &SlotInfo, ubatch: &UBatch) {
        debug_assert_eq!(ubatch.n_tokens, sinfo.n_stream() * sinfo.size());
        for (s, stream) in sinfo.strm.iter().enumerate() {
            let strm = *stream as usize;
            for ii in 0..sinfo.size() {
                let i = s * sinfo.size() + ii;
                let idx = sinfo.idxs[s][ii];
                if !self.cells[strm].is_empty(idx) {
                    self.cells[strm].rm(idx);
                }
                self.cells[strm].pos_set(idx, ubatch.pos[i]);
                for seq_id in &ubatch.seq_id[i] {
                    self.cells[strm].seq_add(idx, *seq_id);
                }
                let off = idx as usize * self.state_row;
                let end = (off + self.state_row).min(self.states[strm].len());
                if off < end {
                    self.states[strm][off..end].fill(0);
                }
            }
            if let Some(last) = sinfo.idxs[s].last() {
                self.heads[strm] = *last + 1;
            }
        }
    }

    pub fn clear(&mut self, data: bool) {
        for s in 0..self.n_stream as usize {
            self.cells[s].reset();
            self.heads[s] = 0;
            if data {
                self.states[s].fill(0);
            }
        }
        self.pending.clear();
    }

    pub fn seq_rm(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos) -> bool {
        // Without replay, a partial removal cannot roll the recurrent state back.
        if self.n_rs_seq > 0 && !self.params.rs_replay && p0 >= 0 && p1 >= 0 && p1 > p0 {
            return false;
        }
        let p0 = p0.max(0);
        let p1 = if p1 < 0 { Pos::MAX } else { p1 };
        if seq_id < 0 {
            for s in 0..self.n_stream as usize {
                let size = self.cells[s].size();
                for i in 0..size {
                    if self.cells[s].pos_in(i, p0, p1) {
                        self.cells[s].rm(i);
                    }
                }
            }
            return true;
        }
        let strm = self.stream(seq_id);
        let size = self.cells[strm].size();
        for i in 0..size {
            if self.cells[strm].pos_in(i, p0, p1) && self.cells[strm].seq_has(i, seq_id) {
                self.cells[strm].seq_rm_cell(i, seq_id);
            }
        }
        true
    }

    pub fn seq_cp(&mut self, src: LSeqId, dst: LSeqId, p0: Pos, p1: Pos) {
        let p0 = p0.max(0);
        let p1 = if p1 < 0 { Pos::MAX } else { p1 };
        let s0 = self.stream(src);
        let s1 = self.stream(dst);
        let size = self.cells[s0].size();
        for i in 0..size {
            if self.cells[s0].pos_in(i, p0, p1) && self.cells[s0].seq_has(i, src) {
                self.cells[s1].seq_add(i, dst);
            }
        }
    }

    pub fn seq_keep(&mut self, seq_id: LSeqId) {
        for s in 0..self.n_stream as usize {
            let size = self.cells[s].size();
            for i in 0..size {
                self.cells[s].seq_keep(i, seq_id);
            }
        }
    }

    pub fn seq_add(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, shift: Pos) {
        if shift == 0 {
            return;
        }
        let p0 = p0.max(0);
        let p1 = if p1 < 0 { Pos::MAX } else { p1 };
        let strm = self.stream(seq_id);
        let size = self.cells[strm].size();
        for i in 0..size {
            if self.cells[strm].pos_in(i, p0, p1) && self.cells[strm].seq_has(i, seq_id) {
                self.cells[strm].pos_add(i, shift);
            }
        }
    }

    pub fn seq_div(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, d: i32) {
        if d == 1 {
            return;
        }
        let p0 = p0.max(0);
        let p1 = if p1 < 0 { Pos::MAX } else { p1 };
        let strm = self.stream(seq_id);
        let size = self.cells[strm].size();
        for i in 0..size {
            if self.cells[strm].pos_in(i, p0, p1) && self.cells[strm].seq_has(i, seq_id) {
                self.cells[strm].pos_div(i, d);
            }
        }
    }

    pub fn seq_pos_min(&self, seq_id: LSeqId) -> Pos {
        self.cells[self.stream(seq_id)].seq_pos_min(seq_id)
    }

    pub fn seq_pos_max(&self, seq_id: LSeqId) -> Pos {
        self.cells[self.stream(seq_id)].seq_pos_max(seq_id)
    }

    pub fn memory_breakdown(&self) -> BTreeMap<String, usize> {
        let buft = if self.params.offload {
            self.params.model.dev_buft.clone()
        } else {
            "CPU".to_string()
        };
        let size: usize = self.states.iter().map(|s| s.len()).sum();
        let mut res = BTreeMap::new();
        res.insert(buft, size);
        res
    }

    pub fn state_write<W: IoWrite>(&self, io: &mut W, seq_id: LSeqId, _flags: u32) {
        write_u32(io, self.n_stream);
        for s in 0..self.n_stream as usize {
            let cells = &self.cells[s];
            let mut idxs = Vec::new();
            for i in 0..cells.size() {
                if !cells.is_empty(i) && (seq_id < 0 || cells.seq_has(i, seq_id)) {
                    idxs.push(i);
                }
            }
            write_u32(io, idxs.len() as u32);
            for &i in &idxs {
                write_i32(io, cells.pos_get(i));
                write_u32(io, 1);
                if seq_id < 0 {
                    let ids: Vec<LSeqId> = (0..self.params.n_seq_max as LSeqId)
                        .filter(|id| cells.seq_has(i, *id))
                        .collect();
                    write_u32(io, ids.len() as u32);
                    for id in ids {
                        write_i32(io, id);
                    }
                } else {
                    write_u32(io, 1);
                    write_i32(io, seq_id);
                }
            }
            write_u32(io, self.state_row as u32);
            for &i in &idxs {
                let off = i as usize * self.state_row;
                let end = (off + self.state_row).min(self.states[s].len());
                io.write_bytes(&self.states[s][off..end]);
            }
        }
    }

    pub fn state_read<R: IoRead>(&mut self, io: &mut R, seq_id: LSeqId, _flags: u32) -> Result<()> {
        let n_stream = read_u32(io);
        if n_stream != self.n_stream {
            return Err(Error::new("failed to restore recurrent cache: n_stream mismatch"));
        }
        if seq_id < 0 {
            self.clear(true);
        } else {
            self.seq_rm(seq_id, -1, -1);
        }

        for s in 0..self.n_stream as usize {
            let cell_count = read_u32(io) as usize;
            if cell_count > self.cells[s].size() as usize {
                return Err(Error::new("failed to restore recurrent cache: too many cells"));
            }
            let mut idxs = Vec::with_capacity(cell_count);
            for i in 0..cell_count {
                let pos = read_i32(io);
                let n_seq_id = read_u32(io);
                let target = if seq_id < 0 { i as CellIdx } else { self.heads[s] };
                self.cells[s].pos_set(target, pos);
                if seq_id < 0 {
                    for _ in 0..n_seq_id {
                        let id = read_i32(io);
                        if id < 0 || id as SeqId >= self.params.n_seq_max {
                            return Err(Error::new("failed to restore recurrent cache: bad seq id"));
                        }
                        self.cells[s].seq_add(target, id);
                    }
                } else {
                    let _ = read_i32(io);
                    self.cells[s].seq_add(target, seq_id);
                    idxs.push(target);
                    self.heads[s] = target + 1;
                }
                if seq_id < 0 {
                    idxs.push(target);
                }
            }
            let row = read_u32(io) as usize;
            if row != self.state_row {
                return Err(Error::new("failed to restore recurrent cache: row size mismatch"));
            }
            for &i in &idxs {
                let off = i as usize * row;
                let end = (off + row).min(self.states[s].len());
                if !io.read_bytes(&mut self.states[s][off..end]) {
                    return Err(Error::new("failed to restore recurrent cache"));
                }
            }
        }
        Ok(())
    }

    pub fn state_clear(&mut self, seq_id: LSeqId) {
        if seq_id < 0 {
            self.clear(true);
            return;
        }
        self.seq_rm(seq_id, -1, -1);
        let strm = self.stream(seq_id);
        let size = self.cells[strm].size();
        for i in 0..size {
            if self.cells[strm].is_empty(i) {
                let off = i as usize * self.state_row;
                let end = (off + self.state_row).min(self.states[strm].len());
                if off < end {
                    self.states[strm][off..end].fill(0);
                }
            }
        }
    }

    pub fn update(&mut self, _do_shift: bool) -> bool {
        // Shifting recurrent states is free: positions live in the cells.
        false
    }
}

/// Recurrent context, mirroring `llama_memory_recurrent_context`.
pub enum RecurrentKind {
    Batch(Vec<SlotInfo>),
    Full,
    Update,
    Failed,
}

pub struct RecurrentContext {
    cache: Rc<RefCell<RecurrentMemory>>,
    kind: RecurrentKind,
    ubatches: Vec<UBatch>,
    i_cur: usize,
    status: MemoryStatus,
}

impl RecurrentContext {
    fn new(
        cache: &Rc<RefCell<RecurrentMemory>>,
        kind: RecurrentKind,
        ubatches: Vec<UBatch>,
        status: MemoryStatus,
    ) -> Self {
        Self {
            cache: Rc::clone(cache),
            kind,
            ubatches,
            i_cur: 0,
            status,
        }
    }

    pub fn failed(cache: &Rc<RefCell<RecurrentMemory>>) -> Self {
        Self::new(cache, RecurrentKind::Failed, Vec::new(), MemoryStatus::FailedPrepare)
    }

    pub fn next(&mut self) -> bool {
        self.i_cur += 1;
        self.i_cur < self.ubatches.len()
    }

    pub fn apply(&mut self) -> bool {
        if let RecurrentKind::Batch(slots) = &self.kind {
            if let (Some(sinfo), Some(ubatch)) = (slots.get(self.i_cur), self.ubatches.get(self.i_cur)) {
                self.cache.borrow_mut().apply_ubatch(sinfo, ubatch);
            } else {
                return false;
            }
        }
        !memory_status_is_fail(self.status)
    }

    pub fn get_status(&self) -> MemoryStatus {
        self.status
    }

    pub fn get_ubatch(&self) -> Option<&UBatch> {
        self.ubatches.get(self.i_cur)
    }
}

// ---------------------------------------------------------------------------
// ISWA attention cache stand-in
// ---------------------------------------------------------------------------

/// Parameters for `llama_kv_cache_iswa`.
#[derive(Clone)]
pub struct IswaParams {
    pub base: KvCacheParams,
    pub swa: KvCacheParams,
}

/// The hybrid-facing part of `llama_kv_cache_iswa`: a full cache and an SWA
/// cache sharing the same sequence operations.
pub struct KvCacheIswa {
    pub base: Rc<RefCell<KvCache>>,
    pub swa: Rc<RefCell<KvCache>>,
}

impl KvCacheIswa {
    pub fn new(params: IswaParams) -> Result<Self> {
        Ok(Self {
            base: Rc::new(RefCell::new(KvCache::new(params.base)?)),
            swa: Rc::new(RefCell::new(KvCache::new(params.swa)?)),
        })
    }

    pub fn get_base(&self) -> Rc<RefCell<KvCache>> {
        Rc::clone(&self.base)
    }

    pub fn get_swa(&self) -> Rc<RefCell<KvCache>> {
        Rc::clone(&self.swa)
    }

    pub fn get_can_shift(&self) -> bool {
        self.base.borrow().get_can_shift()
    }

    pub fn get_stream(&self, seq_id: LSeqId) -> SeqId {
        self.base.borrow().get_stream(seq_id)
    }

    pub fn clear(&mut self, data: bool) {
        self.base.borrow_mut().clear(data);
        self.swa.borrow_mut().clear(data);
    }

    pub fn seq_rm(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos) -> bool {
        self.base.borrow_mut().seq_rm(seq_id, p0, p1);
        self.swa.borrow_mut().seq_rm(seq_id, p0, p1);
        true
    }

    pub fn seq_cp(&mut self, src: LSeqId, dst: LSeqId, p0: Pos, p1: Pos) {
        self.base.borrow_mut().seq_cp(src, dst, p0, p1);
        self.swa.borrow_mut().seq_cp(src, dst, p0, p1);
    }

    pub fn seq_keep(&mut self, seq_id: LSeqId) {
        self.base.borrow_mut().seq_keep(seq_id);
        self.swa.borrow_mut().seq_keep(seq_id);
    }

    pub fn seq_add(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, shift: Pos) {
        self.base.borrow_mut().seq_add(seq_id, p0, p1, shift);
        self.swa.borrow_mut().seq_add(seq_id, p0, p1, shift);
    }

    pub fn seq_div(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, d: i32) {
        self.base.borrow_mut().seq_div(seq_id, p0, p1, d);
        self.swa.borrow_mut().seq_div(seq_id, p0, p1, d);
    }

    pub fn seq_pos_min(&self, seq_id: LSeqId) -> Pos {
        self.base.borrow().seq_pos_min(seq_id)
    }

    pub fn seq_pos_max(&self, seq_id: LSeqId) -> Pos {
        self.base.borrow().seq_pos_max(seq_id)
    }

    pub fn memory_breakdown(&self) -> BTreeMap<String, usize> {
        let mut res = self.base.borrow().memory_breakdown();
        for (buft, size) in self.swa.borrow().memory_breakdown() {
            *res.entry(buft).or_insert(0) += size;
        }
        res
    }

    pub fn state_write<W: IoWrite>(&self, io: &mut W, seq_id: LSeqId, flags: u32) {
        self.base.borrow().state_write(io, seq_id, flags);
        self.swa.borrow().state_write(io, seq_id, flags);
    }

    pub fn state_read<R: IoRead>(&mut self, io: &mut R, seq_id: LSeqId, flags: u32) -> Result<()> {
        self.base.borrow_mut().state_read(io, seq_id, flags)?;
        self.swa.borrow_mut().state_read(io, seq_id, flags)
    }

    pub fn state_clear(&mut self, seq_id: LSeqId) {
        self.base.borrow_mut().state_clear(seq_id);
        self.swa.borrow_mut().state_clear(seq_id);
    }
}

// ---------------------------------------------------------------------------
// Attention dispatch shared by all three hybrid variants
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum AttnCache {
    Kv(Rc<RefCell<KvCache>>),
    Iswa(Rc<RefCell<KvCacheIswa>>),
}

impl AttnCache {
    pub fn prepare(&self, ubatches: &[UBatch]) -> Result<Vec<SlotInfoVec>> {
        match self {
            Self::Kv(kv) => Ok(vec![kv.borrow_mut().prepare(ubatches)?]),
            Self::Iswa(iswa) => {
                let iswa = iswa.borrow();
                let base = iswa.get_base();
                let swa = iswa.get_swa();
                let base_slots = base.borrow_mut().prepare(ubatches)?;
                let swa_slots = swa.borrow_mut().prepare(ubatches)?;
                Ok(vec![base_slots, swa_slots])
            }
        }
    }

    pub fn get_n_stream(&self) -> SeqId {
        match self {
            Self::Kv(kv) => kv.borrow().get_n_stream(),
            Self::Iswa(iswa) => iswa.borrow().base.borrow().get_n_stream(),
        }
    }

    pub fn get_can_shift(&self) -> bool {
        match self {
            Self::Kv(kv) => kv.borrow().get_can_shift(),
            Self::Iswa(iswa) => iswa.borrow().get_can_shift(),
        }
    }

    pub fn clear(&self, data: bool) {
        match self {
            Self::Kv(kv) => kv.borrow_mut().clear(data),
            Self::Iswa(iswa) => iswa.borrow_mut().clear(data),
        }
    }

    pub fn seq_rm(&self, seq_id: LSeqId, p0: Pos, p1: Pos) -> bool {
        match self {
            Self::Kv(kv) => kv.borrow_mut().seq_rm(seq_id, p0, p1),
            Self::Iswa(iswa) => iswa.borrow_mut().seq_rm(seq_id, p0, p1),
        }
    }

    pub fn seq_cp(&self, src: LSeqId, dst: LSeqId, p0: Pos, p1: Pos) {
        match self {
            Self::Kv(kv) => kv.borrow_mut().seq_cp(src, dst, p0, p1),
            Self::Iswa(iswa) => iswa.borrow_mut().seq_cp(src, dst, p0, p1),
        }
    }

    pub fn seq_keep(&self, seq_id: LSeqId) {
        match self {
            Self::Kv(kv) => kv.borrow_mut().seq_keep(seq_id),
            Self::Iswa(iswa) => iswa.borrow_mut().seq_keep(seq_id),
        }
    }

    pub fn seq_add(&self, seq_id: LSeqId, p0: Pos, p1: Pos, shift: Pos) {
        match self {
            Self::Kv(kv) => kv.borrow_mut().seq_add(seq_id, p0, p1, shift),
            Self::Iswa(iswa) => iswa.borrow_mut().seq_add(seq_id, p0, p1, shift),
        }
    }

    pub fn seq_div(&self, seq_id: LSeqId, p0: Pos, p1: Pos, d: i32) {
        match self {
            Self::Kv(kv) => kv.borrow_mut().seq_div(seq_id, p0, p1, d),
            Self::Iswa(iswa) => iswa.borrow_mut().seq_div(seq_id, p0, p1, d),
        }
    }

    pub fn seq_pos_min(&self, seq_id: LSeqId) -> Pos {
        match self {
            Self::Kv(kv) => kv.borrow().seq_pos_min(seq_id),
            Self::Iswa(iswa) => iswa.borrow().seq_pos_min(seq_id),
        }
    }

    pub fn seq_pos_max(&self, seq_id: LSeqId) -> Pos {
        match self {
            Self::Kv(kv) => kv.borrow().seq_pos_max(seq_id),
            Self::Iswa(iswa) => iswa.borrow().seq_pos_max(seq_id),
        }
    }

    pub fn memory_breakdown(&self) -> BTreeMap<String, usize> {
        match self {
            Self::Kv(kv) => kv.borrow().memory_breakdown(),
            Self::Iswa(iswa) => iswa.borrow().memory_breakdown(),
        }
    }

    pub fn state_write<W: IoWrite>(&self, io: &mut W, seq_id: LSeqId, flags: u32) {
        match self {
            Self::Kv(kv) => kv.borrow().state_write(io, seq_id, flags),
            Self::Iswa(iswa) => iswa.borrow().state_write(io, seq_id, flags),
        }
    }

    pub fn state_read<R: IoRead>(&self, io: &mut R, seq_id: LSeqId, flags: u32) -> Result<()> {
        match self {
            Self::Kv(kv) => kv.borrow_mut().state_read(io, seq_id, flags),
            Self::Iswa(iswa) => iswa.borrow_mut().state_read(io, seq_id, flags),
        }
    }

    pub fn state_read_sinfo<R: IoRead>(
        &self,
        io: &mut R,
        seq_id: LSeqId,
        flags: u32,
        sinfos_out: Option<&mut SlotInfoVec>,
        sinfos_in: Option<&SlotInfoVec>,
    ) -> Result<()> {
        match self {
            Self::Kv(kv) => kv.borrow_mut().state_read_sinfo(io, seq_id, flags, sinfos_out, sinfos_in),
            Self::Iswa(iswa) => {
                let iswa = iswa.borrow_mut();
                iswa.base.borrow_mut().state_read_sinfo(io, seq_id, flags, sinfos_out, sinfos_in)?;
                let res = iswa.swa.borrow_mut().state_read(io, seq_id, flags);
                res
            }
        }
    }

    pub fn state_clear(&self, seq_id: LSeqId) {
        match self {
            Self::Kv(kv) => kv.borrow_mut().state_clear(seq_id),
            Self::Iswa(iswa) => iswa.borrow_mut().state_clear(seq_id),
        }
    }

    fn apply_ubatch(&self, layer: usize, sinfo: &SlotInfo, ubatch: &UBatch) {
        match self {
            Self::Kv(kv) => kv.borrow_mut().apply_ubatch(sinfo, ubatch),
            Self::Iswa(iswa) => {
                let iswa = iswa.borrow();
                let cache = if layer == 0 { &iswa.base } else { &iswa.swa };
                cache.borrow_mut().apply_ubatch(sinfo, ubatch);
            }
        }
    }

    fn update(&self, do_shift: bool, sc_info: &StreamCopyInfo) -> bool {
        match self {
            Self::Kv(kv) => kv.borrow_mut().update(do_shift, sc_info),
            Self::Iswa(iswa) => {
                let iswa = iswa.borrow();
                iswa.base.borrow_mut().update(do_shift, sc_info);
                let res = iswa.swa.borrow_mut().update(do_shift, &StreamCopyInfo::default());
                res
            }
        }
    }

    fn get_size(&self) -> CellIdx {
        match self {
            Self::Kv(kv) => kv.borrow().get_size(),
            Self::Iswa(iswa) => iswa.borrow().base.borrow().get_size(),
        }
    }

    fn get_n_kv(&self, sinfo: &SlotInfo) -> u32 {
        match self {
            Self::Kv(kv) => kv.borrow().get_n_kv(sinfo),
            Self::Iswa(iswa) => iswa.borrow().base.borrow().get_n_kv(sinfo),
        }
    }

    fn get_k(&self, ctx: &mut TensorContext, il: i32, n_kv: u32, sinfo: &SlotInfo) -> Tensor {
        match self {
            Self::Kv(kv) => kv.borrow().get_k(ctx, il, n_kv, sinfo),
            Self::Iswa(iswa) => iswa.borrow().base.borrow().get_k(ctx, il, n_kv, sinfo),
        }
    }

    fn get_v(&self, ctx: &mut TensorContext, il: i32, n_kv: u32, sinfo: &SlotInfo) -> Tensor {
        match self {
            Self::Kv(kv) => kv.borrow().get_v(ctx, il, n_kv, sinfo),
            Self::Iswa(iswa) => iswa.borrow().base.borrow().get_v(ctx, il, n_kv, sinfo),
        }
    }
}

pub enum AttnKind {
    Batch { slots: Vec<SlotInfoVec> },
    Full,
    Update { do_shift: bool, sc_info: StreamCopyInfo },
    Failed,
}

/// Attention-side context, shared by `llama_kv_cache_context` and
/// `llama_kv_cache_iswa_context` as seen from the hybrid classes.
pub struct AttentionContext {
    cache: AttnCache,
    kind: AttnKind,
    i_cur: usize,
    status: MemoryStatus,
}

impl AttentionContext {
    pub fn failed(cache: &AttnCache) -> Self {
        Self { cache: cache.clone(), kind: AttnKind::Failed, i_cur: 0, status: MemoryStatus::FailedPrepare }
    }

    pub fn full(cache: &AttnCache) -> Self {
        Self { cache: cache.clone(), kind: AttnKind::Full, i_cur: 0, status: MemoryStatus::Success }
    }

    pub fn update(cache: &AttnCache, do_shift: bool, sc_info: &StreamCopyInfo) -> Self {
        let status = if do_shift || !sc_info.is_empty() {
            MemoryStatus::Success
        } else {
            MemoryStatus::NoUpdate
        };
        Self {
            cache: cache.clone(),
            kind: AttnKind::Update { do_shift, sc_info: sc_info.clone() },
            i_cur: 0,
            status,
        }
    }

    pub fn batch(cache: &AttnCache, slots: Vec<SlotInfoVec>) -> Self {
        Self { cache: cache.clone(), kind: AttnKind::Batch { slots }, i_cur: 0, status: MemoryStatus::Success }
    }

    /// Create the dummy slot info used for a full-cache graph reservation.
    pub fn dummy_sinfo(&self) -> SlotInfo {
        let n_stream = self.cache.get_n_stream();
        let mut sinfo = SlotInfo::default();
        sinfo.s0 = 0;
        sinfo.s1 = n_stream.saturating_sub(1);
        sinfo.idxs.resize_with(n_stream as usize, Vec::new);
        for s in 0..n_stream as usize {
            sinfo.strm.push(s as SeqId);
            sinfo.idxs[s].push(0);
        }
        sinfo
    }

    pub fn next(&mut self) -> bool {
        self.i_cur += 1;
        self.i_cur < match &self.kind {
            AttnKind::Batch { slots } => slots.first().map(|s| s.len()).unwrap_or(0),
            _ => 0,
        }
    }

    pub fn apply(&mut self, ubatch: Option<&UBatch>) -> bool {
        match &self.kind {
            AttnKind::Batch { slots } => {
                let ubatch = match ubatch {
                    Some(u) => u,
                    None => return false,
                };
                for (layer, per_ubatch) in slots.iter().enumerate() {
                    let sinfo = match per_ubatch.get(self.i_cur) {
                        Some(s) => s,
                        None => return false,
                    };
                    self.cache.apply_ubatch(layer, sinfo, ubatch);
                }
            }
            AttnKind::Update { do_shift, sc_info } => {
                self.cache.update(*do_shift, sc_info);
            }
            AttnKind::Full | AttnKind::Failed => {}
        }
        !memory_status_is_fail(self.status)
    }

    pub fn get_status(&self) -> MemoryStatus {
        self.status
    }

    pub fn get_n_kv(&self) -> u32 {
        match &self.kind {
            AttnKind::Batch { slots } => slots
                .first()
                .and_then(|s| s.get(self.i_cur))
                .map(|sinfo| self.cache.get_n_kv(sinfo))
                .unwrap_or(0),
            _ => self.cache.get_size(),
        }
    }

    pub fn get_k(&self, ctx: &mut TensorContext, il: i32) -> Tensor {
        let sinfo = match &self.kind {
            AttnKind::Batch { slots } => slots[self.i_cur][0].clone(),
            _ => self.dummy_sinfo(),
        };
        self.cache.get_k(ctx, il, self.get_n_kv(), &sinfo)
    }

    pub fn get_v(&self, ctx: &mut TensorContext, il: i32) -> Tensor {
        let sinfo = match &self.kind {
            AttnKind::Batch { slots } => slots[self.i_cur][0].clone(),
            _ => self.dummy_sinfo(),
        };
        self.cache.get_v(ctx, il, self.get_n_kv(), &sinfo)
    }

    pub fn get_sinfo(&self) -> Option<&SlotInfo> {
        match &self.kind {
            AttnKind::Batch { slots } => slots.first().and_then(|s| s.get(self.i_cur)),
            _ => None,
        }
    }

    pub fn get_cache(&self) -> &AttnCache {
        &self.cache
    }
}

// ---------------------------------------------------------------------------
// llama_memory_hybrid / llama_memory_hybrid_context
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct HybridParams {
    pub hparams: HParams,
    pub model: Model,
    pub no_alloc: bool,
    /* attn */
    pub type_k: GgmlType,
    pub type_v: GgmlType,
    pub v_trans: bool,
    pub kv_size: CellIdx,
    pub n_swa: u32,
    pub swa_type: SwaType,
    /* recurrent */
    pub type_r: GgmlType,
    pub type_s: GgmlType,
    pub rs_size: u32,
    /* common */
    pub n_seq_max: SeqId,
    pub n_rs_seq: SeqId,
    pub rs_replay: bool,
    pub offload: bool,
    pub unified: bool,
    /// Stand-in for `hparams.is_recr(il)`.
    pub is_recr: fn(u32) -> bool,
    pub filter_attn: Option<fn(u32) -> bool>,
    pub filter_recr: Option<fn(u32) -> bool>,
}

impl Default for HybridParams {
    fn default() -> Self {
        Self {
            hparams: HParams::default(),
            model: Model {
                arch: crate::kv_cache::Arch::Other,
                attn_rot: true,
                lid_k_hdmd: false,
                dev_buft: "CPU".to_string(),
                layers: Vec::new(),
            },
            no_alloc: false,
            type_k: GgmlType::F16,
            type_v: GgmlType::F16,
            v_trans: true,
            kv_size: 512,
            n_swa: 0,
            swa_type: SwaType::None,
            type_r: GgmlType::F32,
            type_s: GgmlType::F32,
            rs_size: 64,
            n_seq_max: 1,
            n_rs_seq: 0,
            rs_replay: false,
            offload: false,
            unified: false,
            is_recr: |_| false,
            filter_attn: None,
            filter_recr: None,
        }
    }
}

pub struct HybridMemory {
    pub hparams: HParams,
    pub mem_attn: Rc<RefCell<KvCache>>,
    pub mem_recr: Rc<RefCell<RecurrentMemory>>,
    /// Mirrors the attention cache's pending cross-stream copies, which the
    /// update context needs for its status and apply step.
    pub attn_sc_info: StreamCopyInfo,
}

impl HybridMemory {
    pub fn new(params: HybridParams) -> Result<Self> {
        let is_recr = params.is_recr;
        RECR_LAYER_FILTER.set(is_recr);
        let filter_attn = params.filter_attn.unwrap_or(filter_attn_default);
        let filter_recr = params.filter_recr.unwrap_or(filter_recr_default);

        let mem_attn = KvCache::new(KvCacheParams {
            hparams: params.hparams.clone(),
            model: params.model.clone(),
            type_k: params.type_k,
            type_v: params.type_v,
            v_trans: params.v_trans,
            offload: params.offload,
            unified: params.unified,
            kv_size: params.kv_size,
            n_seq_max: params.n_seq_max,
            n_swa: params.n_swa,
            swa_type: params.swa_type,
            no_alloc: params.no_alloc,
            filter: Some(filter_attn),
            ..KvCacheParams::default()
        })?;

        let recr_params = RecurrentParams {
            hparams: params.hparams.clone(),
            model: params.model,
            type_r: params.type_r,
            type_s: params.type_s,
            rs_size: params.rs_size,
            n_seq_max: params.n_seq_max,
            n_rs_seq: params.n_rs_seq,
            rs_replay: params.rs_replay,
            offload: params.offload,
            no_alloc: params.no_alloc,
            filter: Some(filter_recr),
        };
        // `n_rs_seq` is read from the prepared recurrent cache by the hybrid
        // classes; keep it on the stand-in as well.
        let mut mem_recr = RecurrentMemory::new(recr_params)?;
        mem_recr.n_rs_seq = params.n_rs_seq;

        Ok(Self {
            hparams: params.hparams,
            mem_attn: Rc::new(RefCell::new(mem_attn)),
            mem_recr: Rc::new(RefCell::new(mem_recr)),
            attn_sc_info: StreamCopyInfo::default(),
        })
    }

    pub fn attn(&self) -> AttnCache {
        AttnCache::Kv(Rc::clone(&self.mem_attn))
    }

    pub fn init_batch(&mut self, balloc: &mut BatchAllocr, n_ubatch: u32, embd_all: bool) -> HybridContext {
        let unified = self.mem_attn.borrow().get_n_stream() == 1;
        let n_rs_seq = self.mem_recr.borrow().n_rs_seq();
        let ubatches = split_ubatches(balloc, n_ubatch, embd_all, unified, n_rs_seq);
        if ubatches.is_empty() {
            return HybridContext::failed(&self.attn(), &self.mem_recr);
        }

        if !self.mem_recr.borrow_mut().prepare(&ubatches) {
            return HybridContext::failed(&self.attn(), &self.mem_recr);
        }
        let attn = self.attn();
        let slots = match attn.prepare(&ubatches) {
            Ok(slots) if !slots.first().map(|s| s.is_empty()).unwrap_or(true) => slots,
            _ => return HybridContext::failed(&attn, &self.mem_recr),
        };

        let ctx_attn = AttentionContext::batch(&attn, slots);
        let ctx_recr = RecurrentMemory::init_batch(&self.mem_recr, ubatches.clone());
        HybridContext::from_contexts(ubatches, ctx_attn, ctx_recr)
    }

    pub fn init_full(&mut self) -> HybridContext {
        let attn = self.attn();
        let ctx_attn = AttentionContext::full(&attn);
        let ctx_recr = RecurrentMemory::init_full(&self.mem_recr);
        HybridContext::from_contexts(Vec::new(), ctx_attn, ctx_recr)
    }

    pub fn init_update(&mut self, optimize: bool) -> HybridContext {
        let attn = self.attn();
        let ctx_attn = AttentionContext::update(&attn, optimize, &self.attn_sc_info);
        self.attn_sc_info = StreamCopyInfo::default();
        let ctx_recr = RecurrentMemory::init_update(&self.mem_recr, optimize);
        HybridContext::from_contexts(Vec::new(), ctx_attn, ctx_recr)
    }

    pub fn get_can_shift(&self) -> bool {
        // Shifting is trivially supported for recurrent layers.
        self.mem_attn.borrow().get_can_shift()
    }

    pub fn clear(&mut self, data: bool) {
        self.mem_attn.borrow_mut().clear(data);
        self.mem_recr.borrow_mut().clear(data);
    }

    pub fn seq_rm(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos) -> bool {
        // The recurrent cache may refuse. It stays untouched when it does.
        if !self.mem_recr.borrow_mut().seq_rm(seq_id, p0, p1) {
            return false;
        }
        self.mem_attn.borrow_mut().seq_rm(seq_id, p0, p1)
    }

    pub fn seq_cp(&mut self, src: LSeqId, dst: LSeqId, p0: Pos, p1: Pos) {
        if src >= 0 && dst >= 0 {
            let s0 = self.mem_attn.borrow().get_stream(src);
            let s1 = self.mem_attn.borrow().get_stream(dst);
            if s0 != s1 {
                self.attn_sc_info.ssrc.push(s0);
                self.attn_sc_info.sdst.push(s1);
            }
        }
        self.mem_attn.borrow_mut().seq_cp(src, dst, p0, p1);
        self.mem_recr.borrow_mut().seq_cp(src, dst, p0, p1);
    }

    pub fn seq_keep(&mut self, seq_id: LSeqId) {
        self.mem_attn.borrow_mut().seq_keep(seq_id);
        self.mem_recr.borrow_mut().seq_keep(seq_id);
    }

    pub fn seq_add(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, shift: Pos) {
        self.mem_attn.borrow_mut().seq_add(seq_id, p0, p1, shift);
        self.mem_recr.borrow_mut().seq_add(seq_id, p0, p1, shift);
    }

    pub fn seq_div(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, d: i32) {
        self.mem_attn.borrow_mut().seq_div(seq_id, p0, p1, d);
        self.mem_recr.borrow_mut().seq_div(seq_id, p0, p1, d);
    }

    pub fn seq_pos_min(&self, seq_id: LSeqId) -> Pos {
        let a = self.mem_attn.borrow().seq_pos_min(seq_id);
        let r = self.mem_recr.borrow().seq_pos_min(seq_id);
        a.max(r)
    }

    pub fn seq_pos_max(&self, seq_id: LSeqId) -> Pos {
        let a = self.mem_attn.borrow().seq_pos_max(seq_id);
        let r = self.mem_recr.borrow().seq_pos_max(seq_id);
        a.min(r)
    }

    pub fn memory_breakdown(&self) -> BTreeMap<String, usize> {
        let mut res = self.mem_attn.borrow().memory_breakdown();
        for (buft, size) in self.mem_recr.borrow().memory_breakdown() {
            *res.entry(buft).or_insert(0) += size;
        }
        res
    }

    pub fn state_write<W: IoWrite>(&self, io: &mut W, seq_id: LSeqId, flags: u32) {
        self.mem_attn.borrow().state_write(io, seq_id, flags);
        self.mem_recr.borrow().state_write(io, seq_id, flags);
    }

    pub fn state_read<R: IoRead>(&mut self, io: &mut R, seq_id: LSeqId, flags: u32) -> Result<()> {
        self.mem_attn.borrow_mut().state_read(io, seq_id, flags)?;
        if let Err(e) = self.mem_recr.borrow_mut().state_read(io, seq_id, flags) {
            // A failed restore leaves the sequence empty in both caches.
            self.mem_attn.borrow_mut().state_clear(seq_id);
            return Err(e);
        }
        Ok(())
    }
}

pub struct HybridContext {
    status: MemoryStatus,
    ubatches: Vec<UBatch>,
    i_next: usize,
    ctx_attn: AttentionContext,
    ctx_recr: RecurrentContext,
}

impl HybridContext {
    fn failed(attn: &AttnCache, recr: &Rc<RefCell<RecurrentMemory>>) -> Self {
        Self {
            status: MemoryStatus::FailedPrepare,
            ubatches: Vec::new(),
            i_next: 0,
            ctx_attn: AttentionContext::failed(attn),
            ctx_recr: RecurrentContext::failed(recr),
        }
    }

    fn from_contexts(ubatches: Vec<UBatch>, attn: AttentionContext, recr: RecurrentContext) -> Self {
        let status = memory_status_combine(attn.get_status(), recr.get_status());
        Self { status, ubatches, i_next: 0, ctx_attn: attn, ctx_recr: recr }
    }

    pub fn next(&mut self) -> bool {
        debug_assert_eq!(self.status, MemoryStatus::Success);
        self.ctx_attn.next();
        self.ctx_recr.next();
        self.i_next += 1;
        self.i_next < self.ubatches.len()
    }

    pub fn apply(&mut self) -> bool {
        debug_assert!(!memory_status_is_fail(self.status));
        let ubatch = self.ubatches.get(self.i_next);
        // C++ uses `res = res & ctx->apply()`, so both caches always apply.
        let a = self.ctx_attn.apply(ubatch);
        let r = self.ctx_recr.apply();
        a & r
    }

    pub fn get_status(&self) -> MemoryStatus {
        self.status
    }

    pub fn get_ubatch(&self) -> &UBatch {
        debug_assert_eq!(self.status, MemoryStatus::Success);
        &self.ubatches[self.i_next]
    }

    pub fn get_attn(&self) -> &AttentionContext {
        &self.ctx_attn
    }

    pub fn get_recr(&self) -> &RecurrentContext {
        &self.ctx_recr
    }
}

// ---------------------------------------------------------------------------
// llama_memory_hybrid_iswa
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct HybridIswaParams {
    pub base: HybridParams,
    /// SWA window size of the attention cache.
    pub n_ubatch: CellIdx,
    pub swa_full: bool,
}

impl Default for HybridIswaParams {
    fn default() -> Self {
        Self { base: HybridParams::default(), n_ubatch: 512, swa_full: false }
    }
}

pub struct HybridIswaMemory {
    pub hparams: HParams,
    pub mem_attn: Rc<RefCell<KvCacheIswa>>,
    pub mem_recr: Rc<RefCell<RecurrentMemory>>,
    pub attn_sc_info: StreamCopyInfo,
}

impl HybridIswaMemory {
    pub fn new(params: HybridIswaParams) -> Result<Self> {
        let base_params = params.base;
        let is_recr = base_params.is_recr;
        RECR_LAYER_FILTER.set(is_recr);
        let filter_attn = base_params.filter_attn.unwrap_or(filter_attn_default);
        let filter_recr = base_params.filter_recr.unwrap_or(filter_recr_default);

        let mut hparams_swa = base_params.hparams.clone();
        if params.swa_full {
            hparams_swa.non_causal_type = NonCausalType::SwaFull;
        }
        let n_swa = base_params.n_swa.max(1);
        let base_attn = KvCacheParams {
            hparams: base_params.hparams.clone(),
            model: base_params.model.clone(),
            type_k: base_params.type_k,
            type_v: base_params.type_v,
            v_trans: base_params.v_trans,
            offload: base_params.offload,
            unified: base_params.unified,
            kv_size: base_params.kv_size,
            n_seq_max: base_params.n_seq_max,
            n_swa: 0,
            swa_type: SwaType::None,
            no_alloc: base_params.no_alloc,
            filter: Some(filter_attn),
            ..KvCacheParams::default()
        };
        let swa_attn = KvCacheParams {
            hparams: hparams_swa,
            model: base_params.model.clone(),
            type_k: base_params.type_k,
            type_v: base_params.type_v,
            v_trans: base_params.v_trans,
            offload: base_params.offload,
            unified: base_params.unified,
            kv_size: params.n_ubatch.max(n_swa),
            n_seq_max: base_params.n_seq_max,
            n_swa,
            swa_type: SwaType::Standard,
            no_alloc: base_params.no_alloc,
            name_tag: "swa_".to_string(),
            filter: Some(filter_attn),
            ..KvCacheParams::default()
        };

        let mem_attn = KvCacheIswa::new(IswaParams { base: base_attn, swa: swa_attn })?;
        let recr_params = RecurrentParams {
            hparams: base_params.hparams.clone(),
            model: base_params.model,
            type_r: base_params.type_r,
            type_s: base_params.type_s,
            rs_size: base_params.rs_size,
            n_seq_max: base_params.n_seq_max,
            n_rs_seq: base_params.n_rs_seq,
            // The ISWA variant never enables replay.
            rs_replay: false,
            offload: base_params.offload,
            no_alloc: base_params.no_alloc,
            filter: Some(filter_recr),
        };
        let mut mem_recr = RecurrentMemory::new(recr_params)?;
        mem_recr.n_rs_seq = base_params.n_rs_seq;

        Ok(Self {
            hparams: base_params.hparams,
            mem_attn: Rc::new(RefCell::new(mem_attn)),
            mem_recr: Rc::new(RefCell::new(mem_recr)),
            attn_sc_info: StreamCopyInfo::default(),
        })
    }

    pub fn attn(&self) -> AttnCache {
        AttnCache::Iswa(Rc::clone(&self.mem_attn))
    }

    pub fn init_batch(&mut self, balloc: &mut BatchAllocr, n_ubatch: u32, embd_all: bool) -> HybridContext {
        let unified = self.mem_attn.borrow().base.borrow().get_n_stream() == 1;
        let n_rs_seq = self.mem_recr.borrow().n_rs_seq();
        let ubatches = split_ubatches(balloc, n_ubatch, embd_all, unified, n_rs_seq);
        if ubatches.is_empty() {
            return HybridContext::failed(&self.attn(), &self.mem_recr);
        }
        if !self.mem_recr.borrow_mut().prepare(&ubatches) {
            return HybridContext::failed(&self.attn(), &self.mem_recr);
        }
        let attn = self.attn();
        let slots = match attn.prepare(&ubatches) {
            Ok(slots) if slots.len() >= 2 && !slots[0].is_empty() && !slots[1].is_empty() => slots,
            _ => return HybridContext::failed(&attn, &self.mem_recr),
        };
        let ctx_attn = AttentionContext::batch(&attn, slots);
        let ctx_recr = RecurrentMemory::init_batch(&self.mem_recr, ubatches.clone());
        HybridContext::from_contexts(ubatches, ctx_attn, ctx_recr)
    }

    pub fn init_full(&mut self) -> HybridContext {
        let attn = self.attn();
        let ctx_attn = AttentionContext::full(&attn);
        let ctx_recr = RecurrentMemory::init_full(&self.mem_recr);
        HybridContext::from_contexts(Vec::new(), ctx_attn, ctx_recr)
    }

    pub fn init_update(&mut self, optimize: bool) -> HybridContext {
        let attn = self.attn();
        let ctx_attn = AttentionContext::update(&attn, optimize, &self.attn_sc_info);
        self.attn_sc_info = StreamCopyInfo::default();
        let ctx_recr = RecurrentMemory::init_update(&self.mem_recr, optimize);
        HybridContext::from_contexts(Vec::new(), ctx_attn, ctx_recr)
    }

    pub fn get_can_shift(&self) -> bool {
        self.mem_attn.borrow().get_can_shift()
    }

    pub fn clear(&mut self, data: bool) {
        self.mem_attn.borrow_mut().clear(data);
        self.mem_recr.borrow_mut().clear(data);
    }

    pub fn seq_rm(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos) -> bool {
        if !self.mem_recr.borrow_mut().seq_rm(seq_id, p0, p1) {
            return false;
        }
        self.mem_attn.borrow_mut().seq_rm(seq_id, p0, p1)
    }

    pub fn seq_cp(&mut self, src: LSeqId, dst: LSeqId, p0: Pos, p1: Pos) {
        if src >= 0 && dst >= 0 {
            let s0 = self.mem_attn.borrow().get_stream(src);
            let s1 = self.mem_attn.borrow().get_stream(dst);
            if s0 != s1 {
                self.attn_sc_info.ssrc.push(s0);
                self.attn_sc_info.sdst.push(s1);
            }
        }
        self.mem_attn.borrow_mut().seq_cp(src, dst, p0, p1);
        self.mem_recr.borrow_mut().seq_cp(src, dst, p0, p1);
    }

    pub fn seq_keep(&mut self, seq_id: LSeqId) {
        self.mem_attn.borrow_mut().seq_keep(seq_id);
        self.mem_recr.borrow_mut().seq_keep(seq_id);
    }

    pub fn seq_add(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, shift: Pos) {
        self.mem_attn.borrow_mut().seq_add(seq_id, p0, p1, shift);
        self.mem_recr.borrow_mut().seq_add(seq_id, p0, p1, shift);
    }

    pub fn seq_div(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, d: i32) {
        self.mem_attn.borrow_mut().seq_div(seq_id, p0, p1, d);
        self.mem_recr.borrow_mut().seq_div(seq_id, p0, p1, d);
    }

    pub fn seq_pos_min(&self, seq_id: LSeqId) -> Pos {
        let a = self.mem_attn.borrow().seq_pos_min(seq_id);
        let r = self.mem_recr.borrow().seq_pos_min(seq_id);
        a.max(r)
    }

    pub fn seq_pos_max(&self, seq_id: LSeqId) -> Pos {
        let a = self.mem_attn.borrow().seq_pos_max(seq_id);
        let r = self.mem_recr.borrow().seq_pos_max(seq_id);
        a.min(r)
    }

    pub fn memory_breakdown(&self) -> BTreeMap<String, usize> {
        let mut res = self.mem_attn.borrow().memory_breakdown();
        for (buft, size) in self.mem_recr.borrow().memory_breakdown() {
            *res.entry(buft).or_insert(0) += size;
        }
        res
    }

    pub fn state_write<W: IoWrite>(&self, io: &mut W, seq_id: LSeqId, flags: u32) {
        self.mem_attn.borrow().state_write(io, seq_id, flags);
        self.mem_recr.borrow().state_write(io, seq_id, flags);
    }

    pub fn state_read<R: IoRead>(&mut self, io: &mut R, seq_id: LSeqId, flags: u32) -> Result<()> {
        self.mem_attn.borrow_mut().state_read(io, seq_id, flags)?;
        self.mem_recr.borrow_mut().state_read(io, seq_id, flags)
    }
}

// ---------------------------------------------------------------------------
// llama_memory_hybrid_idx
// ---------------------------------------------------------------------------

/// Which cells of a sequence make up each pool of `kpool` consecutive
/// positions (or cells, in order mode).
#[derive(Clone, Debug)]
pub struct KpoolLayout {
    pub seqs: [KpoolSeq; LLAMA_MAX_SEQ],
    pub n_pool_real: u32,
}

impl Default for KpoolLayout {
    fn default() -> Self {
        Self {
            seqs: std::array::from_fn(|_| KpoolSeq::default()),
            n_pool_real: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KpoolSeq {
    pub pos_min: Pos,
    /// Stream holding this sequence's cells.
    pub strm: SeqId,
    /// Position and stream-local cell, sorted by position then cell.
    pub cells: Vec<(Pos, CellIdx)>,
    /// Start index in `cells` of each pool.
    pub pools: Vec<u32>,
    /// Where the pool scan stopped, so an append resumes.
    pub j_next: usize,
}

/// Which pools the current ubatch must re-pool.
#[derive(Clone, Debug, Default)]
pub struct KpoolState {
    pub is_new: Vec<u32>,
    /// Per global cell, the generation that last marked a pool with that rep.
    pub rep_gen: Vec<u32>,
    pub generation: u32,
    pub n_pool_real: u32,
    /// Pools of the ubatch's sequences, the only ones the indexer scores.
    pub n_pool_ub: u32,
    pub n_new: u32,
    /// Graph size of the new pool list, stable across decode steps.
    pub n_new_g: u32,
}

/// State shared by `llama_memory_hybrid_idx` and its contexts.
pub struct IdxShared {
    pub hparams_idx: HParams,
    pub mem_idx: Option<Rc<RefCell<KvCache>>>,
    pub mem_idx_stale: [Pos; LLAMA_MAX_SEQ],
    pub kpool_lay: Option<KpoolLayout>,
    pub kpool: u32,
    pub kpool_by_order: bool,
    pub idx_sc_info: StreamCopyInfo,
}

impl IdxShared {
    pub fn get_kpool(&self) -> u32 {
        self.kpool
    }

    pub fn get_kpool_by_order(&self) -> bool {
        self.kpool_by_order
    }

    pub fn mem_idx_stale_get(&self) -> [Pos; LLAMA_MAX_SEQ] {
        self.mem_idx_stale
    }

    pub fn mem_idx_stale_clear(&mut self) {
        self.mem_idx_stale.fill(POS_CLEAN);
    }

    /// A pooled key is only valid while the grouping that produced it holds.
    pub fn mem_idx_stale_set(&mut self, seq_id: LSeqId, p0: Pos) {
        let p0 = p0.max(0);
        if seq_id < 0 {
            for p in self.mem_idx_stale.iter_mut() {
                *p = (*p).min(p0);
            }
            return;
        }
        debug_assert!((seq_id as usize) < LLAMA_MAX_SEQ);
        let s = seq_id as usize;
        self.mem_idx_stale[s] = self.mem_idx_stale[s].min(p0);
    }

    pub fn mem_idx_stale_pos(&self, seq_id: LSeqId, p0: Pos) -> Pos {
        let cache = match self.mem_idx.as_ref() {
            Some(c) => c,
            None => return 0,
        };
        if seq_id < 0 || p0 <= cache.borrow().seq_pos_min(seq_id) {
            return 0;
        }
        p0
    }

    /// Update the layout for all sequences. A plain append keeps the old
    /// cells and pools; an edit at or before `pos_min` regroups everything.
    pub fn kpool_layout_update(&mut self) {
        let cache = match self.mem_idx.as_ref() {
            Some(c) => Rc::clone(c),
            None => return,
        };
        if self.kpool_lay.is_none() {
            self.kpool_lay = Some(KpoolLayout::default());
        }
        let lay = self.kpool_lay.as_mut().unwrap();
        let kpool = self.kpool.max(1);
        let n_stream_kv = cache.borrow().get_n_stream();
        let unified = n_stream_kv == 1;
        lay.n_pool_real = 0;

        for s in 0..LLAMA_MAX_SEQ {
            let sid = s as LSeqId;
            if !unified && s >= n_stream_kv as usize {
                lay.seqs[s] = KpoolSeq::default();
                continue;
            }

            let (cells_snapshot, sp, stream) = {
                let kv = cache.borrow();
                let cells = kv.get_cells(if unified { 0 } else { sid });
                (
                    cells.seq_pos_get(sid).iter().copied().collect::<Vec<_>>(),
                    cells.seq_pos_get(sid).iter().copied().collect::<BTreeSet<_>>(),
                    if unified { 0 } else { kv.get_stream(sid) },
                )
            };
            let sq = &mut lay.seqs[s];
            sq.strm = stream;

            let stale = self.mem_idx_stale[s];
            let mut keep = !sq.cells.is_empty()
                && !sp.is_empty()
                && sq.pos_min == sp.iter().next().unwrap().0
                && stale > sq.pos_min;
            if keep {
                if stale != POS_CLEAN {
                    let cut = sq.cells.partition_point(|c| *c < (stale, 0u32));
                    sq.cells.truncate(cut);
                    while sq.pools.last().map(|j| *j as usize + kpool as usize > sq.cells.len()).unwrap_or(false) {
                        sq.pools.pop();
                    }
                    sq.j_next = sq.pools.last().map(|j| *j as usize + kpool as usize).unwrap_or(0);
                }
                let start = if stale == POS_CLEAN {
                    sq.cells.last().copied().map(|c| (c.0, c.1.saturating_add(1))).unwrap_or((Pos::MIN, 0))
                } else {
                    (stale, 0u32)
                };
                for c in sp.range(start..) {
                    sq.cells.push(*c);
                }
                keep = sq.cells.len() == sp.len();
                // `cells_snapshot` is the source of `sp`; keep the borrow alive.
                let _ = &cells_snapshot;
            }

            if !keep {
                sq.cells = cells_snapshot;
                sq.pools.clear();
                sq.j_next = 0;
                sq.pos_min = sp.iter().next().map(|c| c.0).unwrap_or(0);
            }

            let mut j = sq.j_next;
            let cells_len = sq.cells.len();
            if self.kpool_by_order {
                while j + kpool as usize <= cells_len {
                    sq.pools.push(j as u32);
                    j += kpool as usize;
                }
            } else {
                while j + kpool as usize <= cells_len {
                    let p0 = sq.cells[j].0;
                    if (p0 - sq.pos_min) % kpool as Pos != 0 {
                        j += 1;
                        continue;
                    }
                    let mut ok = true;
                    for k in 1..kpool as usize {
                        if sq.cells[j + k].0 != p0 + k as Pos {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        sq.pools.push(j as u32);
                        j += kpool as usize;
                    } else {
                        j += 1;
                    }
                }
            }
            sq.j_next = j;
            lay.n_pool_real += sq.pools.len() as u32;
        }
    }
}

#[derive(Clone, Debug)]
pub struct HybridIdxParams {
    pub base: HybridParams,
    /// `filter_idx`: the indexer cache exists only when this is given.
    pub filter_idx: Option<fn(u32) -> bool>,
    pub indexer_head_size: u32,
    pub indexer_kpool: u32,
    pub indexer_kpool_row: u32,
    pub indexer_kpool_by_order: bool,
}

impl Default for HybridIdxParams {
    fn default() -> Self {
        Self {
            base: HybridParams::default(),
            filter_idx: None,
            indexer_head_size: 0,
            indexer_kpool: 0,
            indexer_kpool_row: 0,
            indexer_kpool_by_order: false,
        }
    }
}

pub struct HybridIdxMemory {
    pub base: HybridMemory,
    pub shared: Rc<RefCell<IdxShared>>,
}

impl HybridIdxMemory {
    pub fn new(params: HybridIdxParams) -> Result<Self> {
        let base = HybridMemory::new(params.base.clone())?;
        let has_idx = params.filter_idx.is_some();
        let mut hparams_idx = params.base.hparams.clone();
        let mem_idx = if has_idx {
            // MQA with a single key head of indexer_head_size.
            hparams_idx.n_head_kv = 1;
            let kpool = params.indexer_kpool;
            let rows = if kpool > 0 { params.indexer_kpool_row } else { 1 };
            hparams_idx.n_embd_head_k_full = params.indexer_head_size * rows;
            // Raw keys: rotation happens after pooling, at read time.
            hparams_idx.rope_type = RopeType::None;
            // Fool the KV cache into treating this as MLA: no V tensors.
            hparams_idx.n_embd_head_k = params.indexer_head_size;
            hparams_idx.n_embd_head_v = params.indexer_head_size;
            hparams_idx.is_mla = true;

            Some(Rc::new(RefCell::new(KvCache::new(KvCacheParams {
                hparams: hparams_idx.clone(),
                model: params.base.model.clone(),
                type_k: params.base.type_k,
                type_v: params.base.type_v,
                v_trans: params.base.v_trans,
                offload: params.base.offload,
                unified: params.base.unified,
                kv_size: params.base.kv_size,
                n_seq_max: params.base.n_seq_max,
                n_swa: params.base.n_swa,
                swa_type: params.base.swa_type,
                no_alloc: params.base.no_alloc,
                name_tag: "idx_".to_string(),
                is_indexer: true,
                filter: params.filter_idx,
                ..KvCacheParams::default()
            })?)))
        } else {
            None
        };

        Ok(Self {
            base,
            shared: Rc::new(RefCell::new(IdxShared {
                hparams_idx,
                mem_idx,
                mem_idx_stale: [POS_CLEAN; LLAMA_MAX_SEQ],
                kpool_lay: None,
                kpool: params.indexer_kpool,
                kpool_by_order: params.indexer_kpool_by_order,
                idx_sc_info: StreamCopyInfo::default(),
            })),
        })
    }

    pub fn get_mem_idx(&self) -> Option<Rc<RefCell<KvCache>>> {
        self.shared.borrow().mem_idx.clone()
    }

    pub fn get_kpool(&self) -> u32 {
        self.shared.borrow().get_kpool()
    }

    pub fn get_kpool_by_order(&self) -> bool {
        self.shared.borrow().get_kpool_by_order()
    }

    pub fn kpool_layout_get(&self) -> KpoolLayout {
        self.shared
            .borrow()
            .kpool_lay
            .clone()
            .unwrap_or_default()
    }

    pub fn kpool_layout_update(&mut self) {
        self.shared.borrow_mut().kpool_layout_update();
    }

    pub fn clear(&mut self, data: bool) {
        self.base.clear(data);
        let mut shared = self.shared.borrow_mut();
        if let Some(mem) = shared.mem_idx.as_ref() {
            mem.borrow_mut().clear(data);
            shared.mem_idx_stale_set(-1, 0);
        }
    }

    pub fn seq_rm(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos) -> bool {
        if !self.base.mem_recr.borrow_mut().seq_rm(seq_id, p0, p1) {
            return false;
        }
        {
            let mut shared = self.shared.borrow_mut();
            if let Some(mem) = shared.mem_idx.as_ref() {
                let stale = shared.mem_idx_stale_pos(seq_id, p0);
                mem.borrow_mut().seq_rm(seq_id, p0, p1);
                shared.mem_idx_stale_set(seq_id, stale);
            }
        }
        self.base.mem_attn.borrow_mut().seq_rm(seq_id, p0, p1)
    }

    pub fn seq_cp(&mut self, src: LSeqId, dst: LSeqId, p0: Pos, p1: Pos) {
        debug_assert!(p0 <= 0 && p1 < 0, "partial seq_cp is not supported");
        self.base.seq_cp(src, dst, p0, p1);
        let mem = self.shared.borrow().mem_idx.clone();
        let mut shared = self.shared.borrow_mut();
        if let Some(mem) = mem {
            if src >= 0 && dst >= 0 {
                let s0 = mem.borrow().get_stream(src);
                let s1 = mem.borrow().get_stream(dst);
                if s0 != s1 {
                    shared.idx_sc_info.ssrc.push(s0);
                    shared.idx_sc_info.sdst.push(s1);
                }
            }
            mem.borrow_mut().seq_cp(src, dst, p0, p1);
            shared.mem_idx_stale_set(dst, 0);
        }
    }

    pub fn seq_keep(&mut self, seq_id: LSeqId) {
        self.base.seq_keep(seq_id);
        let mut shared = self.shared.borrow_mut();
        if let Some(mem) = shared.mem_idx.as_ref() {
            mem.borrow_mut().seq_keep(seq_id);
            shared.mem_idx_stale_set(-1, 0);
        }
    }

    pub fn seq_add(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, shift: Pos) {
        self.base.seq_add(seq_id, p0, p1, shift);
        let mut shared = self.shared.borrow_mut();
        if let Some(mem) = shared.mem_idx.as_ref() {
            let stale = shared.mem_idx_stale_pos(seq_id, if shift < 0 { p0 + shift } else { p0 });
            mem.borrow_mut().seq_add(seq_id, p0, p1, shift);
            shared.mem_idx_stale_set(seq_id, stale);
        }
    }

    pub fn seq_div(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, d: i32) {
        self.base.seq_div(seq_id, p0, p1, d);
        let mut shared = self.shared.borrow_mut();
        if let Some(mem) = shared.mem_idx.as_ref() {
            mem.borrow_mut().seq_div(seq_id, p0, p1, d);
            shared.mem_idx_stale_set(seq_id, 0);
        }
    }

    pub fn memory_breakdown(&self) -> BTreeMap<String, usize> {
        let mut res = self.base.memory_breakdown();
        let shared = self.shared.borrow();
        if let Some(mem) = shared.mem_idx.as_ref() {
            for (buft, size) in mem.borrow().memory_breakdown() {
                *res.entry(buft).or_insert(0) += size;
            }
        }
        res
    }

    pub fn state_write<W: IoWrite>(&self, io: &mut W, seq_id: LSeqId, flags: u32) {
        self.base.state_write(io, seq_id, flags);
        let shared = self.shared.borrow();
        if let Some(mem) = shared.mem_idx.as_ref() {
            // The indexer section is a pure suffix, so old readers stop early.
            mem.borrow().state_write(io, seq_id, flags);
        }
    }

    /// Drop a sequence directly from all caches, without the recurrent
    /// cache's refusal path.
    fn state_drop(&mut self, seq_id: LSeqId) {
        if seq_id < 0 {
            self.clear(true);
            return;
        }
        self.base.mem_attn.borrow_mut().state_clear(seq_id);
        self.base.mem_recr.borrow_mut().seq_rm(seq_id, -1, -1);
        let mut shared = self.shared.borrow_mut();
        if let Some(mem) = shared.mem_idx.as_ref() {
            mem.borrow_mut().state_clear(seq_id);
            shared.mem_idx_stale_set(seq_id, 0);
        }
    }

    pub fn state_read<R: IoRead>(&mut self, io: &mut R, seq_id: LSeqId, flags: u32) -> Result<()> {
        let mut sinfos_attn = SlotInfoVec::new();
        let has_idx = self.shared.borrow().mem_idx.is_some();
        let result = (|| -> Result<()> {
            self.base
                .mem_attn
                .borrow_mut()
                .state_read_sinfo(io, seq_id, flags, if has_idx { Some(&mut sinfos_attn) } else { None }, None)?;
            self.base.mem_recr.borrow_mut().state_read(io, seq_id, flags)?;
            let mut shared = self.shared.borrow_mut();
            if let Some(mem) = shared.mem_idx.as_ref() {
                mem.borrow_mut().state_read_sinfo(io, seq_id, flags, None, Some(&sinfos_attn))?;
                if !sinfos_attn.is_empty() {
                    shared.mem_idx_stale_set(seq_id, 0);
                }
            }
            Ok(())
        })();
        if result.is_err() {
            // A half-restored context is the one state the indexer cannot
            // fix by itself, so drop it from all three caches.
            self.state_drop(seq_id);
        }
        result
    }

    pub fn init_batch(&mut self, balloc: &mut BatchAllocr, n_ubatch: u32, embd_all: bool) -> HybridIdxContext {
        let unified = self.base.mem_attn.borrow().get_n_stream() == 1;
        let n_rs_seq = self.base.mem_recr.borrow().n_rs_seq();
        let ubatches = split_ubatches(balloc, n_ubatch, embd_all, unified, n_rs_seq);
        if ubatches.is_empty() {
            return HybridIdxContext::failed(&self.base, &self.shared);
        }
        if !self.base.mem_recr.borrow_mut().prepare(&ubatches) {
            return HybridIdxContext::failed(&self.base, &self.shared);
        }
        let attn = self.base.attn();
        let slots_attn = match attn.prepare(&ubatches) {
            Ok(slots) if !slots.first().map(|s| s.is_empty()).unwrap_or(true) => slots.into_iter().next().unwrap(),
            _ => return HybridIdxContext::failed(&self.base, &self.shared),
        };

        let shared = self.shared.borrow();
        let mem_idx = shared.mem_idx.clone();
        let slots_idx = if mem_idx.is_some() {
            slots_attn.clone()
        } else {
            SlotInfoVec::new()
        };
        drop(shared);

        let ctx_attn = AttentionContext::batch(&attn, vec![slots_attn]);
        let ctx_recr = RecurrentMemory::init_batch(&self.base.mem_recr, ubatches.clone());
        let base = HybridContext::from_contexts(ubatches.clone(), ctx_attn, ctx_recr);

        let ns_ubatch = slots_idx.iter().map(|s| s.s1 - s.s0 + 1).collect::<Vec<_>>();
        let by_order = self.get_kpool_by_order();
        let has_idx = mem_idx.is_some();
        let sinfos_kpool = if has_idx && self.get_kpool() > 0 && by_order {
            slots_idx.clone()
        } else {
            SlotInfoVec::new()
        };
        let ctx_idx = mem_idx.map(|mem| AttentionContext::batch(&AttnCache::Kv(mem), vec![slots_idx]));
        let mem_idx_stale_batch = self.shared.borrow().mem_idx_stale_get();
        HybridIdxContext {
            base,
            shared: Rc::clone(&self.shared),
            ctx_idx,
            ns_ubatch,
            sinfos_kpool,
            i_cur: 0,
            kpool_st: None,
            i_kpool: usize::MAX,
            mem_idx_stale_batch,
        }
    }

    pub fn init_full(&mut self) -> HybridIdxContext {
        let base = self.base.init_full();
        let shared = self.shared.borrow();
        let has_idx = shared.mem_idx.is_some();
        let ns_ubatch = if has_idx {
            vec![shared.mem_idx.as_ref().unwrap().borrow().get_n_stream()]
        } else {
            Vec::new()
        };
        let ctx_idx = shared.mem_idx.as_ref().map(|mem| {
            AttentionContext::full(&AttnCache::Kv(Rc::clone(mem)))
        });
        let kpool = shared.kpool;
        drop(shared);

        let mut res = HybridIdxContext {
            base,
            shared: Rc::clone(&self.shared),
            ctx_idx,
            ns_ubatch,
            sinfos_kpool: SlotInfoVec::new(),
            i_cur: 0,
            kpool_st: None,
            i_kpool: usize::MAX,
            mem_idx_stale_batch: [POS_CLEAN; LLAMA_MAX_SEQ],
        };
        if res.kpool_track() {
            res.shared.borrow_mut().kpool_layout_update();
            let mut st = res.kpool_build_sizes();
            let (kv_size, n_seq_max) = {
                let mem = res.shared.borrow().mem_idx.clone().unwrap();
                let mem = mem.borrow();
                (mem.get_size(), mem.get_n_seq_max())
            };
            let n_pool_max = (kv_size as u64 / kpool.max(1) as u64) * n_seq_max as u64;
            st.n_pool_real = st.n_pool_real.max(n_pool_max.min(u32::MAX as u64 - 64) as u32);
            st.n_pool_ub = st.n_pool_real;
            st.n_new = st.n_pool_real;
            st.n_new_g = st.n_new.max(1);
            res.kpool_st = Some(st);
            res.i_kpool = 0;
        }
        res
    }

    pub fn init_update(&mut self, optimize: bool) -> HybridIdxContext {
        let base = self.base.init_update(optimize);
        let ctx_idx = {
            let shared = self.shared.borrow();
            shared.mem_idx.as_ref().map(|mem| {
                // `update()` applies a pending cross-stream seq_cp; otherwise
                // the copy would leave stale indexer keys behind.
                AttentionContext::update(&AttnCache::Kv(Rc::clone(mem)), optimize, &shared.idx_sc_info)
            })
        };
        self.shared.borrow_mut().idx_sc_info = StreamCopyInfo::default();
        HybridIdxContext {
            base,
            shared: Rc::clone(&self.shared),
            ctx_idx,
            ns_ubatch: Vec::new(),
            sinfos_kpool: SlotInfoVec::new(),
            i_cur: 0,
            kpool_st: None,
            i_kpool: usize::MAX,
            mem_idx_stale_batch: [POS_CLEAN; LLAMA_MAX_SEQ],
        }
    }
}

/// Views of the indexer K storage: per-token key/gate rows and pooled keys.
pub struct KpoolAccess {
    key_gate: Tensor,
    pooled: Tensor,
}

impl KpoolAccess {
    pub fn gather_key_gate(&self, idxs: &[i32]) -> Tensor {
        gather_rows(&self.key_gate, idxs)
    }

    pub fn scatter_pooled(&self, values: &Tensor, idxs: &[i64]) -> Tensor {
        let mut pooled = self.pooled.clone();
        pooled.set_rows(values, idxs);
        pooled
    }

    pub fn gather_pooled(&self, idxs: &[i32]) -> Tensor {
        gather_rows(&self.pooled, idxs)
    }
}

pub struct HybridIdxContext {
    base: HybridContext,
    shared: Rc<RefCell<IdxShared>>,
    ctx_idx: Option<AttentionContext>,
    ns_ubatch: Vec<u32>,
    sinfos_kpool: SlotInfoVec,
    i_cur: usize,
    kpool_st: Option<KpoolState>,
    i_kpool: usize,
    mem_idx_stale_batch: [Pos; LLAMA_MAX_SEQ],
}

impl HybridIdxContext {
    fn failed(base: &HybridMemory, shared: &Rc<RefCell<IdxShared>>) -> Self {
        let attn = AttnCache::Kv(Rc::clone(&base.mem_attn));
        Self {
            base: HybridContext::failed(&attn, &base.mem_recr),
            shared: Rc::clone(shared),
            ctx_idx: None,
            ns_ubatch: Vec::new(),
            sinfos_kpool: SlotInfoVec::new(),
            i_cur: 0,
            kpool_st: None,
            i_kpool: usize::MAX,
            mem_idx_stale_batch: [POS_CLEAN; LLAMA_MAX_SEQ],
        }
    }

    pub fn next(&mut self) -> bool {
        // Clear only after a successful ubatch.
        if self.i_cur == 0 {
            self.shared.borrow_mut().mem_idx_stale_clear();
        }
        if let Some(ctx) = self.ctx_idx.as_mut() {
            ctx.next();
        }
        self.i_cur += 1;
        self.base.next()
    }

    pub fn apply(&mut self) -> bool {
        let ubatch = self.base.ubatches.get(self.base.i_next).cloned();
        let mut res = self.base.apply();
        if let Some(ctx) = self.ctx_idx.as_mut() {
            res &= ctx.apply(ubatch.as_ref());
        }
        if res && self.kpool_track() {
            self.shared.borrow_mut().kpool_layout_update();
            if self.kpool_st.is_none() {
                self.kpool_st = Some(KpoolState::default());
            }
            let ub = match ubatch.as_ref() {
                Some(u) => u.clone(),
                None => return res,
            };
            self.kpool_build_state(&ub);
            self.i_kpool = self.i_cur;
        }
        res
    }

    pub fn get_status(&self) -> MemoryStatus {
        self.base.get_status()
    }

    pub fn get_ubatch(&self) -> &UBatch {
        self.base.get_ubatch()
    }

    pub fn get_attn(&self) -> &AttentionContext {
        self.base.get_attn()
    }

    pub fn get_recr(&self) -> &RecurrentContext {
        self.base.get_recr()
    }

    pub fn get_idx(&self) -> Option<&AttentionContext> {
        self.ctx_idx.as_ref()
    }

    pub fn get_n_stream(&self) -> u32 {
        debug_assert!(self.i_cur < self.ns_ubatch.len());
        self.ns_ubatch[self.i_cur]
    }

    pub fn kpool_track(&self) -> bool {
        let shared = self.shared.borrow();
        shared.mem_idx.is_some() && shared.kpool > 0 && !self.ns_ubatch.is_empty()
    }

    pub fn get_kpool_access(&self, il: i32, n_embd: i64) -> Result<KpoolAccess> {
        let mem = self
            .shared
            .borrow()
            .mem_idx
            .clone()
            .ok_or_else(|| Error::new("hybrid idx: missing indexer cache"))?;
        let k = mem
            .borrow()
            .get_k_storage(il)
            .ok_or_else(|| Error::new("hybrid idx: missing indexer K storage"))?;
        let n_tok = k.ne[0] as i64 - n_embd;
        debug_assert!(n_tok > 0 && n_tok % n_embd == 0);
        let n_cells = k.ne[1] * k.ne[2];
        let offset = crate::kv_cache::ggml_row_size(k.ty, n_tok as usize);
        Ok(KpoolAccess {
            key_gate: k.view_2d(n_tok as usize, n_cells, k.nb[1], 0),
            pooled: k.view_2d(n_embd as usize, n_cells, k.nb[1], offset),
        })
    }

    fn kpool_build_sizes(&self) -> KpoolState {
        let lay = self.shared.borrow().kpool_lay.clone().unwrap_or_default();
        KpoolState { n_pool_real: lay.n_pool_real, ..KpoolState::default() }
    }

    fn kpool_cur(&self) -> &KpoolState {
        debug_assert!(self.kpool_st.is_some() && self.i_kpool == self.i_cur, "k-pool state read before apply()");
        self.kpool_st.as_ref().unwrap()
    }

    pub fn get_n_kpool(&self) -> u32 {
        kpool_pad(self.kpool_cur().n_pool_ub)
    }

    pub fn get_n_kpool_new(&self) -> u32 {
        self.kpool_cur().n_new_g
    }

    /// Mark the pools this ubatch touches or completes.
    fn kpool_build_state(&mut self, ubatch: &UBatch) {
        let lay = self.shared.borrow().kpool_lay.clone().unwrap_or_default();
        let st = self.kpool_st.as_mut().unwrap();
        let (mem_idx, kpool, by_order) = {
            let shared = self.shared.borrow();
            (
                shared.mem_idx.clone().unwrap(),
                shared.kpool.max(1),
                shared.kpool_by_order,
            )
        };
        let kv_size = mem_idx.borrow().get_size();
        let n_stream = mem_idx.borrow().get_n_stream();

        st.n_pool_real = lay.n_pool_real;
        st.n_new = 0;
        st.generation = st.generation.wrapping_add(1);
        if st.generation == 0 {
            st.is_new.fill(0);
            st.rep_gen.fill(0);
            st.generation = 1;
        }
        st.is_new.resize(lay.n_pool_real as usize, 0);
        st.rep_gen.resize(kv_size as usize * n_stream as usize, 0);

        let mut pool_start = vec![0u32; LLAMA_MAX_SEQ];
        let mut ip = 0u32;
        for s in 0..LLAMA_MAX_SEQ {
            let sq = &lay.seqs[s];
            pool_start[s] = ip;
            ip += sq.pools.len() as u32;

            let stale_from = if self.i_cur == 0 { self.mem_idx_stale_batch[s] } else { POS_CLEAN };
            if stale_from == POS_CLEAN {
                continue;
            }
            for (k, &j) in sq.pools.iter().enumerate() {
                let rep_pos = sq.cells[j as usize + kpool as usize - 1].0;
                if rep_pos >= stale_from {
                    Self::kpool_mark(sq, s, k, j, kv_size, kpool, pool_start[s], st);
                }
            }
        }
        debug_assert_eq!(ip as usize, st.is_new.len());

        st.n_pool_ub = 0;
        for &sid in &ubatch.seq_id_unq {
            if sid >= 0 && (sid as usize) < LLAMA_MAX_SEQ {
                st.n_pool_ub += lay.seqs[sid as usize].pools.len() as u32;
            }
        }

        let sinfo = if by_order { self.sinfos_kpool.get(self.i_cur) } else { None };
        let n_tps = sinfo.map(|s| s.idxs.first().map(|v| v.len()).unwrap_or(0) as u32).unwrap_or(0);
        for i in 0..ubatch.n_tokens {
            let p = ubatch.pos[i];
            for &sid in &ubatch.seq_id[i] {
                if sid < 0 || sid as usize >= LLAMA_MAX_SEQ {
                    continue;
                }
                let sq = &lay.seqs[sid as usize];
                if by_order {
                    let sinfo = match sinfo {
                        Some(s) => s,
                        None => continue,
                    };
                    let cell = sinfo
                        .idxs
                        .get(i / n_tps.max(1) as usize)
                        .and_then(|v| v.get(i % n_tps.max(1) as usize))
                        .copied()
                        .unwrap_or(0);
                    let r = kpool_rank(&sq.cells, p, cell);
                    debug_assert!(r >= 0);
                    let pool = (r as usize / kpool as usize) as u32;
                    if r >= 0 && pool < sq.pools.len() as u32 {
                        Self::kpool_mark(sq, sid as usize, pool as usize, sq.pools[pool as usize], kv_size, kpool, pool_start[sid as usize], st);
                    }
                    continue;
                }
                let mut it = sq.pools.partition_point(|j| sq.cells[*j as usize].0 <= p);
                if it == 0 {
                    continue;
                }
                it -= 1;
                let j = sq.pools[it];
                if p <= sq.cells[j as usize + kpool as usize - 1].0 {
                    Self::kpool_mark(sq, sid as usize, it, j, kv_size, kpool, pool_start[sid as usize], st);
                }
            }
        }

        let n_pool_max = kv_size / kpool * mem_idx.borrow().get_n_seq_max();
        let bound = ubatch.n_tokens as u32 / kpool + ubatch.n_seqs_unq as u32;
        st.n_new_g = st
            .n_new
            .max(1)
            .max(bound.min(kpool_pad(st.n_pool_real).saturating_sub(1)).min(n_pool_max));
    }

    fn kpool_mark(
        sq: &KpoolSeq,
        s: usize,
        k: usize,
        j: u32,
        kv_size: CellIdx,
        kpool: u32,
        pool_start: u32,
        st: &mut KpoolState,
    ) {
        let rep = sq.strm as usize * kv_size as usize
            + sq.cells[j as usize + kpool as usize - 1].1 as usize;
        if rep >= st.rep_gen.len() {
            return;
        }
        if st.rep_gen[rep] != st.generation {
            st.rep_gen[rep] = st.generation;
            let idx = pool_start as usize + k;
            if idx < st.is_new.len() {
                st.is_new[idx] = st.generation;
                st.n_new += 1;
            }
        }
        let _ = s;
    }

    /// Fill the graph inputs for pooled sparse attention.
    #[allow(clippy::too_many_arguments)]
    pub fn set_input_kpool(
        &self,
        pool_cells: &mut Tensor,
        pool_idxs: &mut Tensor,
        pool_mask: &mut Tensor,
        tail_idxs: &mut Tensor,
        sel_mask: Option<&mut Tensor>,
        new_pool_idxs: &mut Tensor,
        new_pool_rep: &mut Tensor,
        ubatch: &UBatch,
        new_pool_pos: Option<&mut Tensor>,
    ) {
        let shared = self.shared.borrow();
        let lay = shared.kpool_lay.clone().unwrap_or_default();
        let kpool = shared.kpool.max(1) as usize;
        let by_order = shared.kpool_by_order;
        let mem_idx = shared.mem_idx.clone().expect("hybrid idx: missing indexer cache");
        let st = self.kpool_cur();

        let kv_size = mem_idx.borrow().get_size();
        let n_stream_kv = mem_idx.borrow().get_n_stream();
        let n_kv = self.ctx_idx.as_ref().map(|c| c.get_n_kv()).unwrap_or(0);
        let n_tokens = ubatch.n_tokens;
        let n_pool = pool_cells.ne[0];
        let n_new = st.n_new as usize;
        let n_new_g = st.n_new_g.max(1) as usize;

        debug_assert_eq!(n_pool, kpool_pad(st.n_pool_ub) as usize);
        debug_assert_eq!(st.is_new.len(), st.n_pool_real as usize);

        let mut seq_in_ub = vec![false; LLAMA_MAX_SEQ];
        for i in 0..n_tokens {
            for &sid in &ubatch.seq_id[i] {
                if sid >= 0 && (sid as usize) < LLAMA_MAX_SEQ {
                    seq_in_ub[sid as usize] = true;
                }
            }
        }

        // A cell of this ubatch, written before any read, so padded pools
        // still address a finite K row.
        let dummy_cell = {
            let s = ubatch.seq_id[0][0] as usize;
            let sq = &lay.seqs[s];
            sq.cells
                .binary_search_by_key(&ubatch.pos[0], |c| c.0)
                .ok()
                .and_then(|r| sq.cells.get(r))
                .map(|c| sq.strm as i64 * kv_size as i64 + c.1 as i64)
                .unwrap_or(0)
        };

        let rank: Vec<i64> = if by_order {
            let sinfo = self.sinfos_kpool.get(self.i_cur).expect("missing kpool slot info");
            let n_tps = sinfo.idxs.first().map(|v| v.len()).unwrap_or(0).max(1);
            (0..n_tokens)
                .map(|i| {
                    let sid = ubatch.seq_id[i][0] as usize;
                    let cell = sinfo
                        .idxs
                        .get(i / n_tps)
                        .and_then(|v| v.get(i % n_tps))
                        .copied()
                        .unwrap_or(0);
                    kpool_rank(&lay.seqs[sid].cells, ubatch.pos[i], cell)
                })
                .collect()
        } else {
            Vec::new()
        };

        let sentinel = n_kv as i32;
        let (n_sel, n_top) = match sel_mask.as_ref() {
            Some(sel) => (sel.ne[0], (sel.ne[0] / kpool) as u32),
            None => (0usize, 0u32),
        };
        let mut gm: Option<Vec<f32>> = sel_mask.as_ref().map(|t| vec![0.0f32; t.nelements()]);

        let mut seq_pool_start = vec![0u32; LLAMA_MAX_SEQ];
        let mut pool_end: Vec<Pos> = Vec::with_capacity(n_pool);
        let mut cells_data = vec![0i32; n_pool];
        let mut idxs_data = vec![0i32; n_pool * kpool];
        let mut new_idxs = vec![0i32; kpool * n_new_g];
        let mut new_rep = vec![0i64; n_new_g];
        let mut new_pos = vec![0i32; 4 * n_new_g];
        let mut is_rep = vec![false; if n_new_g > n_new { kv_size as usize * n_stream_kv as usize } else { 0 }];
        let gcell = |sq: &KpoolSeq, cell: CellIdx| sq.strm as i64 * kv_size as i64 + cell as i64;

        let mut i_new = 0usize;
        let mut ip_all = 0usize;
        for s in 0..LLAMA_MAX_SEQ {
            let sq = &lay.seqs[s];
            seq_pool_start[s] = pool_end.len() as u32;
            for &j in &sq.pools {
                let rep = sq.cells[j as usize + kpool - 1].1;
                if !is_rep.is_empty() {
                    let g = gcell(sq, rep) as usize;
                    if g < is_rep.len() {
                        is_rep[g] = true;
                    }
                }
                if st.is_new.get(ip_all).copied().unwrap_or(0) == st.generation {
                    debug_assert!(i_new < n_new);
                    for k in 0..kpool {
                        new_idxs[i_new * kpool + k] = gcell(sq, sq.cells[j as usize + k].1) as i32;
                    }
                    new_rep[i_new] = gcell(sq, rep);
                    if new_pool_pos.is_some() {
                        let c = sq.cells[j as usize].1;
                        let kv = mem_idx.borrow();
                        let ext = *kv.get_cells(s as LSeqId).ext_get(c);
                        new_pos[0 * n_new_g + i_new] = sq.cells[j as usize].0;
                        new_pos[1 * n_new_g + i_new] = ext.y;
                        new_pos[2 * n_new_g + i_new] = ext.x;
                        new_pos[3 * n_new_g + i_new] = sq.cells[j as usize].0;
                    }
                    i_new += 1;
                }

                if !seq_in_ub[s] {
                    ip_all += 1;
                    continue;
                }
                let ip = pool_end.len();
                debug_assert!(ip + 1 < n_pool);
                cells_data[ip] = gcell(sq, rep) as i32;
                for k in 0..kpool {
                    idxs_data[ip * kpool + k] = sq.cells[j as usize + k].1 as i32;
                }
                pool_end.push(sq.cells[j as usize + kpool - 1].0);
                ip_all += 1;
            }
        }

        if n_new_g > n_new {
            let mut pad_cell = 0usize;
            for i in n_new..n_new_g {
                while pad_cell < is_rep.len() && is_rep[pad_cell] {
                    pad_cell += 1;
                }
                for k in 0..kpool {
                    new_idxs[i * kpool + k] = pad_cell as i32;
                }
                new_rep[i] = pad_cell as i64;
                pad_cell += 1;
            }
        }

        let n_pool_real = pool_end.len();
        for ip in n_pool_real..n_pool {
            cells_data[ip] = dummy_cell as i32;
            for k in 0..kpool {
                idxs_data[ip * kpool + k] = sentinel;
            }
        }

        let keep = 0.0f32;
        let drop = f32::NEG_INFINITY;
        let mut mask_data = vec![drop; n_pool * n_tokens];
        for i in 0..n_tokens {
            let s = ubatch.seq_id[i][0] as usize;
            let p = ubatch.pos[i];
            let row = i * n_pool;
            let p0 = seq_pool_start[s] as usize;
            let p1 = p0 + lay.seqs[s].pools.len();
            let nv = if by_order {
                ((rank[i] + 1) as usize / kpool).min(p1 - p0)
            } else {
                pool_end[p0..p1].partition_point(|&end| end <= p)
            };
            for j in row + p0..(row + p0 + nv).min(row + n_pool) {
                mask_data[j] = keep;
            }
            if let Some(ref mut g) = gm {
                let nvc = nv.min(n_top as usize);
                let grow = i * n_sel;
                for j in grow..(grow + nvc * kpool).min(g.len()) {
                    g[j] = 0.0;
                }
                for j in (grow + nvc * kpool).min(g.len())..(grow + n_top as usize * kpool).min(g.len()) {
                    g[j] = f32::NEG_INFINITY;
                }
            }
        }
        write_mask(pool_mask, &mask_data);

        let mut tail = vec![sentinel; n_tokens * kpool.saturating_sub(1)];
        for i in 0..n_tokens {
            let s = ubatch.seq_id[i][0] as usize;
            let p = ubatch.pos[i];
            let sq = &lay.seqs[s];
            let n_tail = if by_order {
                ((rank[i] + 1) % kpool as i64) as usize
            } else {
                ((p - sq.pos_min + 1) % kpool as Pos) as usize
            };
            for k in 0..kpool - 1 {
                let mut cell = sentinel;
                let mut real = false;
                if k < n_tail && by_order {
                    let r = rank[i] as usize - k;
                    if r < sq.cells.len() {
                        cell = sq.cells[r].1 as i32;
                        real = true;
                    }
                } else if k < n_tail {
                    let pt = p - k as Pos;
                    if let Ok(r) = sq.cells.binary_search_by_key(&pt, |c| c.0) {
                        cell = sq.cells[r].1 as i32;
                        real = true;
                    }
                }
                tail[i * (kpool - 1) + k] = cell;
                if let Some(ref mut g) = gm {
                    if n_sel % kpool != 0 {
                        g[i * n_sel + n_top as usize * kpool + k] = if real { 0.0 } else { f32::NEG_INFINITY };
                    }
                }
            }
        }

        pool_cells.set_bytes(&bytes_i32(&cells_data));
        pool_idxs.set_bytes(&bytes_i32(&idxs_data));
        tail_idxs.set_bytes(&bytes_i32(&tail));
        new_pool_idxs.set_bytes(&bytes_i32(&new_idxs));
        new_pool_rep.set_bytes(&bytes_i64(&new_rep));
        if let Some(dst) = new_pool_pos {
            dst.set_bytes(&bytes_i32(&new_pos));
        }
        if let Some(dst) = sel_mask {
            if let Some(ref g) = gm {
                dst.set_bytes(&bytes_f32(g));
            }
        }
    }
}

fn kpool_rank(cells: &[(Pos, CellIdx)], pos: Pos, cell: CellIdx) -> i64 {
    match cells.binary_search_by_key(&(pos, cell), |c| (c.0, c.1)) {
        Ok(r) => r as i64,
        Err(_) => -1,
    }
}

fn write_mask(t: &mut Tensor, data: &[f32]) {
    if t.ty == GgmlType::F16 {
        let h: Vec<u16> = data.iter().map(|&f| f32_to_f16(f)).collect();
        let mut bytes = Vec::with_capacity(h.len() * 2);
        for v in h {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        t.set_bytes(&bytes);
    } else {
        t.set_bytes(&bytes_f32(data));
    }
}

fn bytes_i32(data: &[i32]) -> Vec<u8> {
    let mut res = Vec::with_capacity(data.len() * 4);
    for v in data {
        res.extend_from_slice(&v.to_le_bytes());
    }
    res
}

fn bytes_i64(data: &[i64]) -> Vec<u8> {
    let mut res = Vec::with_capacity(data.len() * 8);
    for v in data {
        res.extend_from_slice(&v.to_le_bytes());
    }
    res
}

fn bytes_f32(data: &[f32]) -> Vec<u8> {
    let mut res = Vec::with_capacity(data.len() * 4);
    for v in data {
        res.extend_from_slice(&v.to_le_bytes());
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_combine_follows_the_cpp_rules() {
        assert_eq!(
            memory_status_combine(MemoryStatus::NoUpdate, MemoryStatus::NoUpdate),
            MemoryStatus::NoUpdate
        );
        assert_eq!(
            memory_status_combine(MemoryStatus::NoUpdate, MemoryStatus::Success),
            MemoryStatus::Success
        );
        assert_eq!(
            memory_status_combine(MemoryStatus::FailedPrepare, MemoryStatus::Success),
            MemoryStatus::FailedPrepare
        );
    }

    #[test]
    fn kpool_padding_leaves_one_unused_pool() {
        assert_eq!(kpool_pad(1), 64);
        assert_eq!(kpool_pad(64), 128);
        assert_eq!(kpool_pad(127), 128);
    }

    #[test]
    fn stale_positions_are_minimized_and_cleared() {
        let mut shared = IdxShared {
            hparams_idx: HParams::default(),
            mem_idx: None,
            mem_idx_stale: [POS_CLEAN; LLAMA_MAX_SEQ],
            kpool_lay: None,
            kpool: 4,
            kpool_by_order: false,
            idx_sc_info: StreamCopyInfo::default(),
        };
        shared.mem_idx_stale_set(0, 10);
        shared.mem_idx_stale_set(0, 4);
        assert_eq!(shared.mem_idx_stale[0], 4);
        shared.mem_idx_stale_set(-1, 7);
        assert_eq!(shared.mem_idx_stale[0], 4);
        assert_eq!(shared.mem_idx_stale[1], 7);
        shared.mem_idx_stale_clear();
        assert_eq!(shared.mem_idx_stale[0], POS_CLEAN);
    }

    #[test]
    fn pools_need_consecutive_positions() {
        let cells = vec![(0i32, 0u32), (1, 1), (2, 2), (3, 3)];
        assert_eq!(kpool_rank(&cells, 2, 2), 2);
        assert_eq!(kpool_rank(&cells, 2, 3), -1);
    }
}

//! Rust port of `llama-memory-recurrent.h` / `llama-memory-recurrent.cpp`.
//!
//! Recurrent state caches keep one state row per sequence, not per token. Cells
//! hold positions, tail links, rollback metadata and the sequence set. ggml
//! tensors are represented by byte-backed descriptors so state save/restore can
//! move rows without FFI or `unsafe`.

use crate::batch::{BatchAllocr, Pos, Ubatch};
use crate::core::{Error, Result, SeqId};
use crate::hparams::Hparams;
use crate::io::{IoRead, IoWrite};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

/// Signed sequence id; negative values are sentinels (`-1` means "all").
pub type LSeqId = i32;

/// Cell and stream indices are `uint32_t` in the C++ source.
pub type CellIdx = SeqId;

pub const LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY: u32 = 1;
pub const LLAMA_STATE_SEQ_FLAGS_ON_DEVICE: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryStatus {
    Success,
    NoUpdate,
    FailedPrepare,
}

pub fn memory_status_is_fail(s: MemoryStatus) -> bool {
    s == MemoryStatus::FailedPrepare
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    I32 = 26,
    I64 = 27,
    BF16 = 30,
    F8E4M3 = 43,
}

impl GgmlType {
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
            Self::F16 | Self::BF16 => 2,
            Self::F8E4M3 => 1,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    pub fn blck_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F8E4M3 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }
}

/// `ggml_row_size(type, n)`.
pub fn ggml_row_size(ty: GgmlType, n: usize) -> usize {
    let bs = ty.blck_size();
    ((n + bs - 1) / bs) * ty.type_size()
}

/// Minimal 2D ggml tensor stand-in over a shared byte store.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub ty: GgmlType,
    pub ne: [usize; 4],
    pub nb: [usize; 4],
    pub offset: usize,
    pub storage: Rc<RefCell<Vec<u8>>>,
    pub name: String,
}

impl Tensor {
    fn new_2d_at(ty: GgmlType, ne0: usize, ne1: usize, storage: Rc<RefCell<Vec<u8>>>, offset: usize, name: String) -> Self {
        let row = ggml_row_size(ty, ne0);
        let ts = ty.type_size();
        Self {
            ty,
            ne: [ne0, ne1, 1, 1],
            nb: [ts, row, row * ne1, row * ne1],
            offset,
            storage,
            name,
        }
    }

    pub fn nbytes(&self) -> usize {
        ggml_row_size(self.ty, self.ne[0]) * self.ne[1] * self.ne[2] * self.ne[3]
    }

    fn zero_range(&self, offset: usize, size: usize) {
        let mut s = self.storage.borrow_mut();
        let begin = self.offset + offset;
        let end = (begin + size).min(s.len());
        if begin < end {
            s[begin..end].fill(0);
        }
    }

    fn bytes_range(&self, offset: usize, size: usize) -> Vec<u8> {
        let s = self.storage.borrow();
        let begin = self.offset + offset;
        let end = (begin + size).min(s.len());
        if begin < end {
            s[begin..end].to_vec()
        } else {
            Vec::new()
        }
    }

    fn set_bytes_range(&self, offset: usize, data: &[u8]) {
        let mut s = self.storage.borrow_mut();
        let begin = self.offset + offset;
        let end = (begin + data.len()).min(s.len());
        if begin < end {
            s[begin..end].copy_from_slice(&data[..end - begin]);
        }
    }
}

/// Backend buffer descriptor used for `memory_breakdown()` and buffer clears.
#[derive(Clone, Debug)]
pub struct MemoryBuf {
    pub name: String,
    pub buft: String,
    pub size: usize,
    storage: Rc<RefCell<Vec<u8>>>,
}

/// `llama_memory_recurrent::mem_cell`.
#[derive(Clone, Debug)]
pub struct MemCell {
    pub pos: Pos,
    pub src: i32,
    pub src0: i32,
    pub tail: i32,
    pub rs_max: u32,
    pub rs_pend: u32,
    pub rs_rep: u32,
    pub seq_id: BTreeSet<LSeqId>,
}

impl Default for MemCell {
    fn default() -> Self {
        Self {
            pos: -1,
            src: -1,
            src0: -1,
            tail: -1,
            rs_max: 0,
            rs_pend: 0,
            rs_rep: 0,
            seq_id: BTreeSet::new(),
        }
    }
}

impl MemCell {
    pub fn has_seq_id(&self, id: LSeqId) -> bool {
        self.seq_id.contains(&id)
    }

    pub fn is_empty(&self) -> bool {
        self.seq_id.is_empty()
    }

    pub fn is_same_seq(&self, other: &Self) -> bool {
        self.seq_id == other.seq_id
    }
}

/// Parameters of the recurrent memory constructor.
#[derive(Clone)]
pub struct RecurrentParams {
    pub hparams: Hparams,
    pub type_r: GgmlType,
    pub type_s: GgmlType,
    pub offload: bool,
    pub mem_size: CellIdx,
    pub n_seq_max: SeqId,
    pub n_rs_seq: CellIdx,
    pub rs_replay: bool,
    pub no_alloc: bool,
    /// Buffer type name used when `offload` is false is always "CPU".
    pub dev_buft: String,
    pub filter: Option<fn(u32) -> bool>,
}

impl Default for RecurrentParams {
    fn default() -> Self {
        Self {
            hparams: Hparams::default(),
            type_r: GgmlType::F16,
            type_s: GgmlType::F32,
            offload: false,
            mem_size: 512,
            n_seq_max: 1,
            n_rs_seq: 0,
            rs_replay: false,
            no_alloc: false,
            dev_buft: "CPU".to_string(),
            filter: None,
        }
    }
}

/// A 2D tensor descriptor pending allocation in a backend buffer.
struct TensorDesc {
    ty: GgmlType,
    ne0: usize,
    ne1: usize,
    name: String,
    buft: String,
}

/// `llama_memory_recurrent`.
pub struct RecurrentMemory {
    pub head: CellIdx,
    pub size: CellIdx,
    pub used: CellIdx,
    pub n_rs_seq: CellIdx,
    pub rs_idx: Vec<CellIdx>,
    pub rs_replay: bool,
    pub n: CellIdx,
    pub rs_z: i32,
    pub cells: Vec<MemCell>,

    r_l: Vec<Option<Tensor>>,
    s_l: Vec<Option<Tensor>>,
    p_l: Vec<Option<Tensor>>,
    spk_l: Vec<Option<Tensor>>,
    spv_l: Vec<Option<Tensor>>,
    spgb_l: Vec<Option<Tensor>>,

    hparams: Hparams,
    n_seq_max: SeqId,
    ctxs_bufs: Vec<MemoryBuf>,
    is_empty: bool,
}

impl RecurrentMemory {
    pub fn new(params: RecurrentParams) -> Result<Self> {
        let hparams = params.hparams;
        let n_layer = hparams.n_layer() as usize;
        let mem_size = params.mem_size as usize;
        let n_rs_seq = params.n_rs_seq;
        let rs_replay = params.rs_replay && n_rs_seq > 0;

        let mut res = Self {
            head: 0,
            size: params.mem_size,
            used: 0,
            n_rs_seq,
            rs_idx: vec![0; params.n_seq_max as usize],
            rs_replay,
            n: 0,
            rs_z: -1,
            cells: vec![MemCell::default(); mem_size],
            r_l: vec![None; n_layer],
            s_l: vec![None; n_layer],
            p_l: vec![None; n_layer],
            spk_l: vec![None; n_layer],
            spv_l: vec![None; n_layer],
            spgb_l: vec![None; n_layer],
            hparams,
            n_seq_max: params.n_seq_max,
            ctxs_bufs: Vec::new(),
            is_empty: true,
        };

        let n_embd_r = res.hparams.n_embd_r()? as usize;
        let n_embd_s = res.hparams.n_embd_s()? as usize;
        let n_head = res.hparams.n_head(0)? as usize;
        let ple_conv_state = res.hparams.ple_conv_state() as usize;
        let ssm_d_conv = res.hparams.ssm_d_conv as usize;

        let n_pend = (n_rs_seq + 1) as usize;
        let n_ch = n_embd_r / ssm_d_conv.saturating_sub(1).max(1);
        let n_embd_r_eff = if rs_replay {
            n_ch * (ssm_d_conv.saturating_sub(1) + n_pend)
        } else {
            n_embd_r
        };
        let n_rows = mem_size * if rs_replay { 1 } else { 1 + n_rs_seq as usize };

        let mut descs: Vec<TensorDesc> = Vec::new();
        let mut r_idx: Vec<Option<usize>> = vec![None; n_layer];
        let mut s_idx: Vec<Option<usize>> = vec![None; n_layer];
        let mut p_idx: Vec<Option<usize>> = vec![None; n_layer];
        let mut spk_idx: Vec<Option<usize>> = vec![None; n_layer];
        let mut spv_idx: Vec<Option<usize>> = vec![None; n_layer];
        let mut spgb_idx: Vec<Option<usize>> = vec![None; n_layer];

        for il in 0..n_layer {
            let il_u = il as u32;
            if let Some(filter) = params.filter {
                if !filter(il_u) {
                    continue;
                }
            }

            let buft = if params.offload {
                params.dev_buft.clone()
            } else {
                "CPU".to_string()
            };

            let push = |descs: &mut Vec<TensorDesc>, ty: GgmlType, ne0: usize, ne1: usize, name: String, buft: &str| -> usize {
                let idx = descs.len();
                descs.push(TensorDesc { ty, ne0, ne1, name, buft: buft.to_string() });
                idx
            };

            r_idx[il] = Some(push(&mut descs, params.type_r, n_embd_r_eff, n_rows, format!("cache_r_l{}", il), &buft));
            s_idx[il] = Some(push(&mut descs, params.type_s, n_embd_s, n_rows, format!("cache_s_l{}", il), &buft));

            if rs_replay {
                let kda = res.hparams.n_embd_head_kda != 0;
                let n_k = if kda {
                    res.hparams.n_embd_head_kda as usize * n_head
                } else {
                    res.hparams.ssm_d_state as usize * res.hparams.ssm_n_group as usize
                };
                let n_v = if kda { n_k } else { res.hparams.ssm_d_inner as usize };
                let n_gb = if kda { n_k } else { res.hparams.ssm_dt_rank as usize };

                spk_idx[il] = Some(push(&mut descs, params.type_s, n_k * n_pend, mem_size, format!("cache_spk_l{}", il), &buft));
                spv_idx[il] = Some(push(&mut descs, params.type_s, n_v * n_pend, mem_size, format!("cache_spv_l{}", il), &buft));
                spgb_idx[il] = Some(push(&mut descs, params.type_s, n_gb * 2 * n_pend, mem_size, format!("cache_spgb_l{}", il), &buft));
            }

            if ple_conv_state > 0 && res.hparams.is_ple(il_u)? {
                let n_win_p = (res.hparams.ple_conv_kernel as usize).saturating_sub(1) * res.hparams.ple_ngram_size as usize;
                let n_embd_p = if rs_replay {
                    ple_conv_state / n_win_p.max(1) * (n_win_p + n_pend)
                } else {
                    ple_conv_state
                };
                p_idx[il] = Some(push(&mut descs, params.type_r, n_embd_p, n_rows, format!("cache_ple_r_l{}", il), &buft));
            }
        }

        // One backing store per buffer type, like a ggml context per buft.
        let mut by_buft: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, d) in descs.iter().enumerate() {
            by_buft.entry(d.buft.clone()).or_default().push(i);
        }
        let mut built: Vec<Option<Tensor>> = vec![None; descs.len()];
        for (buft, ids) in by_buft {
            let size: usize = ids.iter().map(|&i| ggml_row_size(descs[i].ty, descs[i].ne0) * descs[i].ne1).sum();
            let storage = Rc::new(RefCell::new(vec![0u8; size]));
            let mut offset = 0usize;
            for &i in &ids {
                let d = &descs[i];
                let nbytes = ggml_row_size(d.ty, d.ne0) * d.ne1;
                built[i] = Some(Tensor::new_2d_at(d.ty, d.ne0, d.ne1, storage.clone(), offset, d.name.clone()));
                offset += nbytes;
            }
            res.ctxs_bufs.push(MemoryBuf { name: "rs cache".to_string(), buft, size, storage });
        }

        let take = |built: &[Option<Tensor>], idx: Option<usize>| -> Option<Tensor> {
            idx.and_then(|i| built[i].clone())
        };
        for il in 0..n_layer {
            res.r_l[il] = take(&built, r_idx[il]);
            res.s_l[il] = take(&built, s_idx[il]);
            res.p_l[il] = take(&built, p_idx[il]);
            res.spk_l[il] = take(&built, spk_idx[il]);
            res.spv_l[il] = take(&built, spv_idx[il]);
            res.spgb_l[il] = take(&built, spgb_idx[il]);
        }

        res.is_empty = res.ctxs_bufs.is_empty();
        if res.is_empty && res.n_rs_seq != 0 {
            res.n_rs_seq = 0;
            res.rs_replay = false;
        }

        // `no_alloc` only controls backend allocation in C++; the descriptors
        // still exist. The byte store is kept so state I/O is well defined.
        let _ = params.no_alloc;

        Ok(res)
    }

    pub fn get_size(&self) -> CellIdx {
        self.size
    }

    pub fn get_can_shift(&self) -> bool {
        true
    }

    fn is_empty(&self) -> bool {
        self.is_empty
    }

    pub fn total_size(&self) -> usize {
        self.ctxs_bufs.iter().map(|b| b.size).sum()
    }

    pub fn size_r_bytes(&self) -> usize {
        self.r_l.iter().filter_map(|t| t.as_ref()).map(|t| t.nbytes()).sum()
    }

    pub fn size_s_bytes(&self) -> usize {
        let mut size = 0usize;
        for t in self.s_l.iter().chain(self.spk_l.iter()).chain(self.spv_l.iter()).chain(self.spgb_l.iter()) {
            if let Some(t) = t {
                size += t.nbytes();
            }
        }
        size
    }

    pub fn size_p_bytes(&self) -> usize {
        self.p_l.iter().filter_map(|t| t.as_ref()).map(|t| t.nbytes()).sum()
    }

    pub fn memory_breakdown(&self) -> BTreeMap<String, usize> {
        self.ctxs_bufs.iter().map(|b| (b.buft.clone(), b.size)).collect()
    }

    pub fn clear(&mut self, data: bool) {
        for cell in self.cells.iter_mut() {
            cell.pos = -1;
            cell.seq_id.clear();
            cell.src = -1;
            cell.tail = -1;
            cell.rs_max = 0;
            cell.rs_pend = 0;
            cell.rs_rep = 0;
        }
        self.head = 0;
        self.used = 0;
        if data {
            for buf in &self.ctxs_bufs {
                buf.storage.borrow_mut().fill(0);
            }
        }
        self.rs_idx.fill(0);
    }

    pub fn set_rs_idx(&mut self, seq_id: LSeqId, idx: CellIdx) {
        if seq_id < 0 {
            self.rs_idx.fill(0);
            return;
        }
        if seq_id as usize >= self.rs_idx.len() || idx > self.n_rs_seq {
            return;
        }
        self.rs_idx[seq_id as usize] = idx;
    }

    pub fn seq_rm(&mut self, seq_id: LSeqId, mut p0: Pos, mut p1: Pos) -> bool {
        let mut new_head = self.size;
        if p0 < 0 {
            p0 = 0;
        }
        if p1 < 0 {
            p1 = Pos::MAX;
        }
        if (seq_id as u32) >= self.n_seq_max {
            return false;
        }

        let rm_all = p0 == 0 && p1 == Pos::MAX;
        if rm_all {
            self.set_rs_idx(seq_id, 0);
        }

        // Mamba/RWKV states cannot be erased from the end of a sequence.
        if seq_id >= self.size as LSeqId {
            return false;
        }
        if seq_id >= 0 {
            let tail_id = self.cells[seq_id as usize].tail;
            if tail_id >= 0 {
                let cell = &self.cells[tail_id as usize];
                if 0 < p0 && p0 <= cell.pos && p1 > cell.pos {
                    if cell.seq_id.len() > 1 {
                        return false;
                    }
                    if self.is_empty() {
                        self.cells[tail_id as usize].pos = p0 - 1;
                        return true;
                    }
                    let rollback = cell.pos - (p0 - 1);
                    let pending = self.rs_idx[seq_id as usize] != 0;
                    if !pending && rollback >= 1 && rollback <= cell.rs_max as Pos {
                        if self.rs_replay {
                            let c = &mut self.cells[tail_id as usize];
                            c.rs_pend -= rollback as u32;
                            c.rs_max = 0;
                        } else {
                            self.set_rs_idx(seq_id, rollback as CellIdx);
                        }
                        self.cells[tail_id as usize].pos = p0 - 1;
                        return true;
                    }
                    return false;
                }
                if p0 <= cell.pos && cell.pos < p1 {
                    self.cells[seq_id as usize].tail = -1;
                }
            }
        } else if p0 != p1 && (p0 != 0 || p1 != Pos::MAX) {
            return false;
        }

        for i in 0..self.size as usize {
            let cell = &mut self.cells[i];
            if cell.pos >= p0 && cell.pos < p1 {
                if seq_id < 0 {
                    cell.seq_id.clear();
                } else if cell.has_seq_id(seq_id) {
                    cell.seq_id.remove(&seq_id);
                } else {
                    continue;
                }
                if cell.is_empty() {
                    if cell.pos >= 0 {
                        self.used -= 1;
                    }
                    cell.pos = -1;
                    cell.src = -1;
                    if new_head == self.size {
                        new_head = i as CellIdx;
                    }
                }
            }
        }

        if new_head != self.size && new_head < self.head {
            self.head = new_head;
        }
        true
    }

    pub fn seq_cp(&mut self, seq_id_src: LSeqId, seq_id_dst: LSeqId, mut p0: Pos, mut p1: Pos) {
        if seq_id_src == seq_id_dst {
            return;
        }
        if p0 < 0 {
            p0 = 0;
        }
        if p1 < 0 {
            p1 = Pos::MAX;
        }
        let _ = p0;
        let _ = p1;

        if (seq_id_dst as u32) < self.size && (seq_id_src as u32) < self.size {
            let tail_dst = self.cells[seq_id_dst as usize].tail;
            if tail_dst >= 0 {
                self.cells[seq_id_dst as usize].tail = -1;
                let cell_dst = &mut self.cells[tail_dst as usize];
                cell_dst.seq_id.remove(&seq_id_dst);
                if cell_dst.seq_id.is_empty() {
                    cell_dst.pos = -1;
                    cell_dst.src = -1;
                    self.used -= 1;
                }
            }
            let tail_src = self.cells[seq_id_src as usize].tail;
            if tail_src >= 0 {
                self.cells[tail_src as usize].seq_id.insert(seq_id_dst);
                self.cells[seq_id_dst as usize].tail = tail_src;
            }
        }
    }

    pub fn seq_keep(&mut self, seq_id: LSeqId) {
        let mut new_head = self.size;
        for i in 0..self.size as usize {
            if i as LSeqId != seq_id {
                self.cells[i].tail = -1;
            }
            let cell = &mut self.cells[i];
            if !cell.has_seq_id(seq_id) {
                if cell.pos >= 0 {
                    self.used -= 1;
                }
                cell.pos = -1;
                cell.src = -1;
                cell.seq_id.clear();
                if new_head == self.size {
                    new_head = i as CellIdx;
                }
            } else {
                cell.seq_id.clear();
                cell.seq_id.insert(seq_id);
            }
        }
        if new_head != self.size && new_head < self.head {
            self.head = new_head;
        }
    }

    pub fn seq_add(&mut self, seq_id: LSeqId, mut p0: Pos, mut p1: Pos, shift: Pos) {
        if shift == 0 {
            return;
        }
        if p0 < 0 {
            p0 = 0;
        }
        if p1 < 0 {
            p1 = Pos::MAX;
        }
        if p0 == p1 {
            return;
        }
        if 0 <= seq_id && seq_id < self.size as LSeqId {
            let tail = self.cells[seq_id as usize].tail;
            if tail >= 0 {
                let cell = &mut self.cells[tail as usize];
                if cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1 {
                    cell.pos += shift;
                }
            }
        }
    }

    pub fn seq_div(&mut self, seq_id: LSeqId, mut p0: Pos, mut p1: Pos, d: i32) {
        if d == 1 {
            return;
        }
        if p0 < 0 {
            p0 = 0;
        }
        if p1 < 0 {
            p1 = Pos::MAX;
        }
        if p0 == p1 {
            return;
        }
        if 0 <= seq_id && seq_id < self.size as LSeqId {
            let tail = self.cells[seq_id as usize].tail;
            if tail >= 0 {
                let cell = &mut self.cells[tail as usize];
                if cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1 {
                    cell.pos /= d;
                }
            }
        }
    }

    pub fn seq_pos_min(&self, seq_id: LSeqId) -> Pos {
        let mut result = Pos::MAX;
        for cell in &self.cells {
            if cell.has_seq_id(seq_id) {
                result = result.min(cell.pos);
            }
        }
        if result == Pos::MAX {
            -1
        } else {
            result
        }
    }

    pub fn seq_pos_max(&self, seq_id: LSeqId) -> Pos {
        let mut result = -1;
        for cell in &self.cells {
            if cell.has_seq_id(seq_id) {
                result = result.max(cell.pos);
            }
        }
        result
    }

    pub fn prepare(&mut self, ubatches: &[Ubatch]) -> bool {
        let org_cells = self.cells.clone();
        let org_used = self.used;
        let org_head = self.head;

        let mut success = true;
        for ubatch in ubatches {
            if !self.find_slot(ubatch) {
                success = false;
                break;
            }
        }

        self.cells = org_cells;
        self.used = org_used;
        self.head = org_head;
        success
    }

    /// Find a contiguous slot of memory cells and emplace the ubatch there.
    pub fn find_slot(&mut self, ubatch: &Ubatch) -> bool {
        let n_seq_tokens = ubatch.n_seq_tokens as usize;
        let n_seqs = ubatch.n_seqs as usize;
        let size = self.size as usize;
        if n_seqs == 0 || n_seq_tokens == 0 || size == 0 {
            return false;
        }

        if self.head > self.used + 2 * n_seqs as CellIdx {
            self.head = 0;
        }
        debug_assert!(ubatch.equal_seqs());

        let mut min = size as i32 - 1;
        let mut max = 0i32;

        // Resolve shared cells before making new ones.
        for s in 0..n_seqs {
            let i = s * n_seq_tokens;
            let n_seq_id = ubatch.n_seq_id[i] as usize;
            for j in 0..n_seq_id {
                let seq_id = ubatch.seq_id[i][j] as LSeqId;
                if seq_id < 0 || seq_id as u32 >= self.size {
                    return false;
                }
                if j > 0 {
                    let tail = self.cells[seq_id as usize].tail;
                    if tail >= 0 {
                        self.cells[seq_id as usize].tail = -1;
                        let cell = &mut self.cells[tail as usize];
                        cell.seq_id.remove(&seq_id);
                        if cell.is_empty() {
                            cell.pos = -1;
                            cell.src = -1;
                            self.used -= 1;
                        }
                    }
                }
            }
        }

        // Debug-only tail verification mirrors the C++ NDEBUG block.
        #[cfg(debug_assertions)]
        {
            let mut tails = vec![-1i32; size];
            for i in 0..size {
                for &seq_id in &self.cells[i].seq_id {
                    if seq_id < 0 || seq_id as usize >= size {
                        return false;
                    }
                    debug_assert_eq!(tails[seq_id as usize], -1);
                    tails[seq_id as usize] = i as i32;
                }
            }
            for i in 0..size {
                debug_assert_eq!(tails[i], self.cells[i].tail);
            }
        }

        let mut next_empty_cell = self.head as usize;
        for _ in 0..size {
            if next_empty_cell >= size {
                next_empty_cell -= size;
            }
            if self.cells[next_empty_cell].is_empty() {
                break;
            }
            next_empty_cell += 1;
        }

        for s in 0..n_seqs {
            let i = s * n_seq_tokens;
            let seq_id = ubatch.seq_id[i][0] as usize;
            let mut has_cell = false;
            let tail = self.cells[seq_id].tail;
            if tail >= 0 {
                let cell = &self.cells[tail as usize];
                debug_assert!(cell.has_seq_id(seq_id as LSeqId));
                if cell.seq_id.len() == 1 {
                    has_cell = true;
                }
            }
            if !has_cell {
                if tail >= 0 {
                    let orig = tail as usize;
                    let (pos, src, rs_pend) = {
                        let c = &self.cells[orig];
                        (c.pos, c.src, c.rs_pend)
                    };
                    let empty = &mut self.cells[next_empty_cell];
                    empty.pos = pos;
                    empty.src = src;
                    empty.rs_pend = rs_pend;
                    self.cells[orig].seq_id.remove(&(seq_id as LSeqId));
                    self.cells[next_empty_cell].seq_id.insert(seq_id as LSeqId);
                    debug_assert!(!self.cells[orig].is_empty());
                }
                self.cells[seq_id].tail = next_empty_cell as i32;
                if s + 1 < n_seqs {
                    for _ in 0..size {
                        next_empty_cell += 1;
                        if next_empty_cell >= size {
                            next_empty_cell -= size;
                        }
                        if self.cells[next_empty_cell].is_empty() {
                            break;
                        }
                    }
                }
            }
            min = min.min(self.cells[seq_id].tail);
            max = max.max(self.cells[seq_id].tail);
        }

        // Gather the cells into a contiguous range `[min, max]`.
        for s in 0..n_seqs {
            let i = s * n_seq_tokens;
            let dst_id = s as i32 + min;
            let src_id = self.cells[ubatch.seq_id[i][0] as usize].tail;
            if dst_id != src_id {
                let d = dst_id as usize;
                let sr = src_id as usize;
                let dst = {
                    let c = &self.cells[d];
                    (c.pos, c.src, c.rs_max, c.rs_pend, c.seq_id.clone())
                };
                let src = {
                    let c = &self.cells[sr];
                    (c.pos, c.src, c.rs_max, c.rs_pend, c.seq_id.clone())
                };
                self.cells[d].pos = src.0;
                self.cells[d].src = src.1;
                self.cells[d].rs_max = src.2;
                self.cells[d].rs_pend = src.3;
                self.cells[d].seq_id = src.4;
                self.cells[sr].pos = dst.0;
                self.cells[sr].src = dst.1;
                self.cells[sr].rs_max = dst.2;
                self.cells[sr].rs_pend = dst.3;
                self.cells[sr].seq_id = dst.4;
                if !self.rs_replay {
                    self.cells[sr].rs_max = 0;
                }
                for j in 0..size {
                    let tail = &mut self.cells[j].tail;
                    if *tail == src_id {
                        *tail = dst_id;
                    } else if *tail == dst_id {
                        *tail = src_id;
                    }
                }
            }
        }

        for s in 0..n_seqs {
            let i = s * n_seq_tokens;
            let last_pos = ubatch.pos[i + n_seq_tokens - 1];
            let cell_id = s as i32 + min;
            {
                let cell = &mut self.cells[cell_id as usize];
                if ubatch.n_pos == 1 && cell.pos >= 0 && last_pos != cell.pos + n_seq_tokens as Pos {
                    // Non-consecutive positions cannot clear state mid-batch.
                }
                cell.pos = last_pos;
                cell.rs_max = if n_seq_tokens as CellIdx <= self.n_rs_seq + 1 {
                    n_seq_tokens as CellIdx - 1
                } else {
                    0
                };
                if self.rs_replay {
                    cell.rs_rep = if cell.src >= 0 { cell.rs_pend } else { 0 };
                    cell.rs_pend = if cell.rs_max > 0 { n_seq_tokens as CellIdx } else { 0 };
                }
                cell.seq_id.clear();
                for j in 0..ubatch.n_seq_id[i] as usize {
                    let seq_id = ubatch.seq_id[i][j] as LSeqId;
                    cell.seq_id.insert(seq_id);
                }
            }
            let ids: Vec<LSeqId> = self.cells[cell_id as usize].seq_id.iter().copied().collect();
            for seq_id in ids {
                if seq_id >= 0 && (seq_id as usize) < size {
                    self.cells[seq_id as usize].tail = cell_id;
                }
            }
        }

        let mut refcounts = vec![0i32; size];
        for cell in &self.cells {
            if cell.src >= 0 && (cell.src as usize) < size {
                refcounts[cell.src as usize] += 1;
            }
        }
        self.rs_z = -1;
        for i in min..=max {
            if refcounts[i as usize] == 0 {
                self.rs_z = i;
                break;
            }
        }
        for i in min..=max {
            let idx = i as usize;
            if self.cells[idx].src < 0 {
                debug_assert!(self.rs_z >= 0);
                self.cells[idx].src0 = self.rs_z;
            } else {
                self.cells[idx].src0 = self.cells[idx].src;
            }
            self.cells[idx].src = i;
        }

        self.head = min as CellIdx;
        self.n = (max - min + 1) as CellIdx;
        self.used = self.cells.iter().filter(|c| !c.is_empty()).count() as CellIdx;
        self.n >= n_seqs as CellIdx
    }

    pub fn init_full(&mut self) -> RecurrentContext<'_> {
        RecurrentContext::full(self)
    }

    pub fn init_update(&self) -> RecurrentContext<'_> {
        RecurrentContext::update()
    }

    pub fn init_batch(&mut self, balloc: &mut BatchAllocr, n_ubatch: CellIdx, embd_all: bool) -> RecurrentContext<'_> {
        loop {
            balloc.split_reset();
            let mut ubatches = Vec::new();
            loop {
                let ubatch = if embd_all {
                    balloc.split_seq(n_ubatch)
                } else {
                    let n_keep_tail = if self.n_rs_seq > 0 { self.n_rs_seq + 1 } else { 0 };
                    balloc.split_equal(n_ubatch, true, n_keep_tail)
                };
                if ubatch.n_tokens == 0 {
                    break;
                }
                ubatches.push(ubatch);
            }

            if balloc.get_n_used() < balloc.get_n_tokens() as CellIdx {
                break;
            }
            if !self.prepare(&ubatches) {
                break;
            }
            return RecurrentContext::batch(self, ubatches);
        }
        RecurrentContext::failed(MemoryStatus::FailedPrepare)
    }

    // -----------------------------------------------------------------------
    // state write/load
    // -----------------------------------------------------------------------

    pub fn state_write<W: IoWrite + ?Sized>(&self, io: &mut W, seq_id: LSeqId, flags: u32) -> Result<()> {
        let mut cell_ranges: Vec<(CellIdx, CellIdx)> = Vec::new();
        let mut cell_ranges_data: Vec<(CellIdx, CellIdx)> = Vec::new();
        let mut cell_count = 0u32;
        let mut range_begin = self.size;

        for (i, cell) in self.cells.iter().enumerate() {
            let i = i as CellIdx;
            if (seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id) {
                cell_count += 1;
                let mut rs_idx_cur = 0u32;
                if self.n_rs_seq != 0 {
                    if seq_id != -1 {
                        rs_idx_cur = *self.rs_idx.get(seq_id as usize).ok_or_else(|| Error::new("invalid seq_id in recurrent state"))?;
                    } else {
                        let mut has_rs_idx = false;
                        for &cell_seq_id in &cell.seq_id {
                            let seq_rs_idx = *self.rs_idx.get(cell_seq_id as usize).ok_or_else(|| Error::new("invalid seq_id in recurrent state"))?;
                            if !has_rs_idx {
                                rs_idx_cur = seq_rs_idx;
                                has_rs_idx = true;
                            } else if rs_idx_cur != seq_rs_idx {
                                return Err(Error::new("cannot write shared recurrent state with different rollback indices"));
                            }
                        }
                    }
                }
                let cell_id = rs_idx_cur * self.size + if cell.src >= 0 { cell.src as u32 } else { i };
                if cell_ranges_data.last().map(|r| r.1) != Some(cell_id) {
                    cell_ranges_data.push((cell_id, cell_id + 1));
                } else {
                    cell_ranges_data.last_mut().unwrap().1 += 1;
                }
                if range_begin == self.size {
                    range_begin = i;
                }
            } else if range_begin != self.size {
                cell_ranges.push((range_begin, i));
                range_begin = self.size;
            }
        }
        if range_begin != self.size {
            cell_ranges.push((range_begin, self.size));
        }

        if flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE != 0 && cell_ranges.len() > 1 {
            return Err(Error::new("cannot save multiple ranges of recurrent cells to device memory"));
        }

        write_u32(io, cell_count);
        self.state_write_meta(io, &cell_ranges, seq_id);
        self.state_write_data(io, &cell_ranges_data)?;
        Ok(())
    }

    fn state_write_meta<W: IoWrite + ?Sized>(&self, io: &mut W, cell_ranges: &[(CellIdx, CellIdx)], seq_id: LSeqId) {
        for &(first, second) in cell_ranges {
            for i in first..second {
                let cell = &self.cells[i as usize];
                write_i32(io, cell.pos);
                let n_seq_id = if seq_id == -1 { cell.seq_id.len() as u32 } else { 0 };
                write_u32(io, n_seq_id);
                if n_seq_id != 0 {
                    for &id in &cell.seq_id {
                        write_i32(io, id);
                    }
                }
                if self.rs_replay {
                    write_u32(io, cell.rs_pend);
                }
            }
        }
    }

    fn state_write_data<W: IoWrite + ?Sized>(&self, io: &mut W, cell_ranges: &[(CellIdx, CellIdx)]) -> Result<()> {
        let s_trans = 0u32;
        let n_layer = self.hparams.n_layer();
        write_u32(io, s_trans);
        write_u32(io, n_layer);

        for il in 0..n_layer as usize {
            let r = match self.r_l.get(il).and_then(|t| t.as_ref()) {
                Some(r) => r,
                None => continue,
            };
            write_i32(io, r.ty as i32);
            let r_size_row = r.nb[1] as u64;
            write_u64(io, r_size_row);
            for &(first, second) in cell_ranges {
                let size = (second - first) as usize * r_size_row as usize;
                io.write(&r.bytes_range(first as usize * r_size_row as usize, size));
            }
            if let Some(p) = self.p_l.get(il).and_then(|t| t.as_ref()) {
                let p_size_row = p.nb[1] as u64;
                write_u64(io, p_size_row);
                for &(first, second) in cell_ranges {
                    let size = (second - first) as usize * p_size_row as usize;
                    io.write(&p.bytes_range(first as usize * p_size_row as usize, size));
                }
            }
        }

        let n_embd_s = self.hparams.n_embd_s()? as usize;
        if s_trans == 0 {
            for il in 0..n_layer as usize {
                let s = match self.s_l.get(il).and_then(|t| t.as_ref()) {
                    Some(s) => s,
                    None => continue,
                };
                write_i32(io, s.ty as i32);
                let s_size_row = ggml_row_size(s.ty, n_embd_s) as u64;
                write_u64(io, s_size_row);
                for &(first, second) in cell_ranges {
                    let size = (second - first) as usize * s_size_row as usize;
                    io.write(&s.bytes_range(first as usize * s_size_row as usize, size));
                }
            }
        } else {
            let mem_size = self.size as usize;
            for il in 0..n_layer as usize {
                let s = match self.s_l.get(il).and_then(|t| t.as_ref()) {
                    Some(s) => s,
                    None => continue,
                };
                write_i32(io, s.ty as i32);
                write_u32(io, s.ty.type_size() as u32);
                write_u32(io, n_embd_s as u32);
                let s_size_el = s.ty.type_size();
                for j in 0..n_embd_s {
                    for &(first, second) in cell_ranges {
                        let size = (second - first) as usize * s_size_el;
                        let src_offset = (first as usize + j * mem_size) * s_size_el;
                        io.write(&s.bytes_range(src_offset, size));
                    }
                }
            }
        }

        if self.rs_replay {
            for il in 0..n_layer as usize {
                for t in [self.spk_l.get(il), self.spv_l.get(il), self.spgb_l.get(il)] {
                    let t = match t.and_then(|t| t.as_ref()) {
                        Some(t) => t,
                        None => continue,
                    };
                    let size_row = t.nb[1] as u64;
                    write_u64(io, size_row);
                    for &(first, second) in cell_ranges {
                        let size = (second - first) as usize * size_row as usize;
                        io.write(&t.bytes_range(first as usize * size_row as usize, size));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn state_read<R: IoRead + ?Sized>(&mut self, io: &mut R, seq_id: LSeqId, _flags: u32) -> Result<()> {
        let cell_count = read_u32(io);
        let meta_read = match self.state_read_meta(io, cell_count, seq_id) {
            Ok(ok) => ok,
            Err(_) => false,
        };
        let cell_head = self.head;
        let mut res = meta_read;
        match self.state_read_data(io, cell_count) {
            Ok(ok) => res = res && ok,
            Err(_) => res = false,
        }
        if !res {
            self.state_clear(seq_id, cell_head, if meta_read { cell_count } else { 0 });
            return Err(Error::new("failed to restore kv cache"));
        }
        if self.n_rs_seq != 0 {
            self.set_rs_idx(seq_id, 0);
        }
        Ok(())
    }

    fn state_read_meta<R: IoRead + ?Sized>(&mut self, io: &mut R, cell_count: CellIdx, dest_seq_id: LSeqId) -> Result<bool> {
        if cell_count > self.size {
            return Ok(false);
        }

        if dest_seq_id != -1 {
            if dest_seq_id < 0 {
                return Ok(false);
            }
            self.seq_rm(dest_seq_id, -1, -1);
            if cell_count == 0 {
                return Ok(true);
            }

            let n_pos = self.hparams.n_pos_per_embd();
            let mut balloc = BatchAllocr::new(n_pos as usize, false);
            let mut ubatch = balloc.ubatch_reserve(cell_count, 1);
            let mut rs_pend = vec![0u32; cell_count as usize];

            for i in 0..cell_count as usize {
                let pos = read_i32(io);
                let n_seq_id = read_u32(io);
                if n_seq_id != 0 {
                    return Ok(false);
                }
                if self.rs_replay {
                    rs_pend[i] = read_u32(io);
                }
                ubatch.pos[i] = pos;
            }
            ubatch.n_seq_id[0] = 1;
            ubatch.seq_id[0] = vec![dest_seq_id as SeqId];

            if !self.find_slot(&ubatch) {
                return Ok(false);
            }
            for i in 0..cell_count as usize {
                let idx = (self.head as usize) + i;
                self.cells[idx].rs_pend = rs_pend[i];
                self.cells[idx].rs_max = 0;
            }
        } else {
            self.clear(true);
            for i in 0..cell_count as usize {
                let pos = read_i32(io);
                let n_seq_id = read_u32(io);
                self.cells[i].pos = pos;
                for _ in 0..n_seq_id {
                    let seq_id = read_i32(io);
                    if seq_id < 0 || seq_id as u32 >= self.n_seq_max {
                        return Ok(false);
                    }
                    self.cells[i].seq_id.insert(seq_id);
                    let tail = &mut self.cells[seq_id as usize].tail;
                    if *tail != -1 {
                        return Ok(false);
                    }
                    *tail = i as i32;
                }
                if self.rs_replay {
                    self.cells[i].rs_pend = read_u32(io);
                }
            }
            self.head = 0;
            self.used = cell_count;
        }

        for i in 0..cell_count as usize {
            let cell_id = self.head as usize + i;
            if cell_id < self.cells.len() {
                self.cells[cell_id].src = cell_id as i32;
            }
        }
        Ok(true)
    }

    fn state_read_data<R: IoRead + ?Sized>(&mut self, io: &mut R, cell_count: CellIdx) -> Result<bool> {
        let s_trans = read_u32(io);
        let n_layer = read_u32(io);
        if n_layer != self.hparams.n_layer() {
            return Ok(false);
        }
        if cell_count > self.size {
            return Ok(false);
        }
        if s_trans != 0 {
            return Ok(false);
        }

        let head = self.head as usize;
        let n_embd_s = self.hparams.n_embd_s()? as usize;
        let mut buf = vec![0u8; cell_count as usize * 8];

        for il in 0..n_layer as usize {
            let r = match self.r_l.get(il).and_then(|t| t.as_ref()) {
                Some(r) => r,
                None => continue,
            };
            if read_i32(io) != r.ty as i32 {
                return Ok(false);
            }
            let r_size_row = read_u64(io) as usize;
            if r_size_row != r.nb[1] {
                return Ok(false);
            }
            if cell_count != 0 {
                buf.resize(cell_count as usize * r_size_row, 0);
                io.read(&mut buf);
                r.set_bytes_range(head * r_size_row, &buf);
            }
            if let Some(p) = self.p_l.get(il).and_then(|t| t.as_ref()) {
                let p_size_row = read_u64(io) as usize;
                if p_size_row != p.nb[1] {
                    return Ok(false);
                }
                if cell_count != 0 {
                    buf.resize(cell_count as usize * p_size_row, 0);
                    io.read(&mut buf);
                    p.set_bytes_range(head * p_size_row, &buf);
                }
            }
        }

        for il in 0..n_layer as usize {
            let s = match self.s_l.get(il).and_then(|t| t.as_ref()) {
                Some(s) => s,
                None => continue,
            };
            if read_i32(io) != s.ty as i32 {
                return Ok(false);
            }
            let s_size_row = read_u64(io) as usize;
            if s_size_row != ggml_row_size(s.ty, n_embd_s) {
                return Ok(false);
            }
            if cell_count != 0 {
                buf.resize(cell_count as usize * s_size_row, 0);
                io.read(&mut buf);
                s.set_bytes_range(head * s_size_row, &buf);
            }
        }

        // s_trans == 1 is rejected above, matching state_read_data's check.

        if self.rs_replay {
            for il in 0..n_layer as usize {
                for t in [self.spk_l.get(il), self.spv_l.get(il), self.spgb_l.get(il)] {
                    let t = match t.and_then(|t| t.as_ref()) {
                        Some(t) => t,
                        None => continue,
                    };
                    let size_row = read_u64(io) as usize;
                    if size_row != t.nb[1] {
                        return Ok(false);
                    }
                    if cell_count != 0 {
                        buf.resize(cell_count as usize * size_row, 0);
                        io.read(&mut buf);
                        t.set_bytes_range(head * size_row, &buf);
                    }
                }
            }
        }
        Ok(true)
    }

    fn state_clear(&mut self, seq_id: LSeqId, cell_head: CellIdx, cell_count: CellIdx) {
        if seq_id == -1 {
            self.clear(true);
            return;
        }
        self.seq_rm(seq_id, -1, -1);
        if cell_count == 0 {
            return;
        }
        let head = cell_head as usize;
        let count = cell_count as usize;
        for il in 0..self.hparams.n_layer() as usize {
            if let Some(r) = self.r_l.get(il).and_then(|t| t.as_ref()) {
                r.zero_range(head * r.nb[1], count * r.nb[1]);
            }
            if let Some(s) = self.s_l.get(il).and_then(|t| t.as_ref()) {
                let row = ggml_row_size(s.ty, self.hparams.n_embd_s().unwrap_or(0) as usize);
                s.zero_range(head * row, count * row);
            }
            if let Some(p) = self.p_l.get(il).and_then(|t| t.as_ref()) {
                p.zero_range(head * p.nb[1], count * p.nb[1]);
            }
        }
    }
}

/// `llama_memory_recurrent_context`.
pub struct RecurrentContext<'a> {
    status: MemoryStatus,
    mem: Option<&'a mut RecurrentMemory>,
    i_next: usize,
    ubatches: Vec<Ubatch>,
    is_full: bool,
}

impl<'a> RecurrentContext<'a> {
    pub fn failed(status: MemoryStatus) -> Self {
        Self { status, mem: None, i_next: 0, ubatches: Vec::new(), is_full: false }
    }

    pub fn full(mem: &'a mut RecurrentMemory) -> Self {
        Self { status: MemoryStatus::Success, mem: Some(mem), i_next: 0, ubatches: Vec::new(), is_full: true }
    }

    pub fn update() -> Self {
        Self::failed(MemoryStatus::NoUpdate)
    }

    pub fn batch(mem: &'a mut RecurrentMemory, ubatches: Vec<Ubatch>) -> Self {
        Self { status: MemoryStatus::Success, mem: Some(mem), i_next: 0, ubatches, is_full: false }
    }

    pub fn get_status(&self) -> MemoryStatus {
        self.status
    }

    pub fn get_ubatch(&self) -> &Ubatch {
        debug_assert_eq!(self.status, MemoryStatus::Success);
        &self.ubatches[self.i_next]
    }

    pub fn next(&mut self) -> bool {
        debug_assert_eq!(self.status, MemoryStatus::Success);
        self.i_next += 1;
        self.i_next < self.ubatches.len()
    }

    pub fn apply(&mut self) -> bool {
        debug_assert!(!memory_status_is_fail(self.status));
        if self.ubatches.is_empty() {
            debug_assert_eq!(self.status, MemoryStatus::NoUpdate);
            return true;
        }
        if let Some(mem) = self.mem.as_mut() {
            mem.find_slot(&self.ubatches[self.i_next]);
        }
        true
    }

    pub fn get_n_rs(&self) -> CellIdx {
        if self.is_full {
            self.mem.as_ref().map(|m| m.size).unwrap_or(0)
        } else {
            self.mem.as_ref().map(|m| m.n).unwrap_or(0)
        }
    }

    pub fn get_head(&self) -> CellIdx {
        if self.is_full {
            0
        } else {
            self.mem.as_ref().map(|m| m.head).unwrap_or(0)
        }
    }

    pub fn get_rs_z(&self) -> i32 {
        if self.is_full {
            0
        } else {
            self.mem.as_ref().map(|m| m.rs_z).unwrap_or(-1)
        }
    }

    pub fn get_size(&self) -> CellIdx {
        self.mem.as_ref().map(|m| m.size).unwrap_or(0)
    }

    pub fn get_r_l(&self, il: i32) -> Option<&Tensor> {
        self.mem.as_ref()?.r_l.get(il as usize)?.as_ref()
    }

    pub fn get_s_l(&self, il: i32) -> Option<&Tensor> {
        self.mem.as_ref()?.s_l.get(il as usize)?.as_ref()
    }

    pub fn get_p_l(&self, il: i32) -> Option<&Tensor> {
        self.mem.as_ref()?.p_l.get(il as usize)?.as_ref()
    }

    pub fn get_spk_l(&self, il: i32) -> Option<&Tensor> {
        self.mem.as_ref()?.spk_l.get(il as usize)?.as_ref()
    }

    pub fn get_spv_l(&self, il: i32) -> Option<&Tensor> {
        self.mem.as_ref()?.spv_l.get(il as usize)?.as_ref()
    }

    pub fn get_spgb_l(&self, il: i32) -> Option<&Tensor> {
        self.mem.as_ref()?.spgb_l.get(il as usize)?.as_ref()
    }

    pub fn get_rs_replay(&self) -> bool {
        self.mem.as_ref().map(|m| m.rs_replay).unwrap_or(false)
    }

    pub fn get_rs_snapshots(&self) -> bool {
        self.mem.as_ref().map(|m| m.n_rs_seq != 0 && !m.rs_replay).unwrap_or(false)
    }

    pub fn get_rs_rep(&self, i: usize) -> CellIdx {
        if self.is_full {
            return 0;
        }
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => return 0,
        };
        mem.cells.get(mem.head as usize + i).map(|c| c.rs_rep).unwrap_or(0)
    }

    pub fn get_rs_in_place(&self) -> bool {
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => return false,
        };
        if self.is_full || self.get_rs_snapshots() || mem.rs_z >= 0 || mem.n != self.get_ubatch().n_seqs {
            return false;
        }
        for i in 0..mem.n as usize {
            let idx = mem.head as usize + i;
            if mem.cells.get(idx).map(|c| c.src0) != Some(idx as i32) {
                return false;
            }
        }
        true
    }

    /// Copy source row id for state `i`, consuming a pending rollback index.
    pub fn s_copy(&mut self, i: usize) -> i32 {
        let mem = match self.mem.as_mut() {
            Some(m) => m,
            None => return -1,
        };
        let cell_idx = mem.head as usize + i;
        let src0 = mem.cells[cell_idx].src0;
        if mem.n_rs_seq == 0 {
            return src0;
        }
        let mut idx = 0u32;
        if let Some(&seq) = mem.cells[cell_idx].seq_id.iter().next() {
            if seq >= 0 && (seq as usize) < mem.rs_idx.len() {
                idx = mem.rs_idx[seq as usize];
                mem.rs_idx[seq as usize] = 0;
            }
        }
        idx as i32 * mem.size as i32 + src0
    }
}

fn write_u32<W: IoWrite + ?Sized>(io: &mut W, v: u32) {
    io.write(&v.to_le_bytes());
}

fn write_i32<W: IoWrite + ?Sized>(io: &mut W, v: i32) {
    io.write(&v.to_le_bytes());
}

fn write_u64<W: IoWrite + ?Sized>(io: &mut W, v: u64) {
    io.write(&v.to_le_bytes());
}

fn read_u32<R: IoRead + ?Sized>(io: &mut R) -> u32 {
    let mut b = [0u8; 4];
    io.read(&mut b);
    u32::from_le_bytes(b)
}

fn read_i32<R: IoRead + ?Sized>(io: &mut R) -> i32 {
    let mut b = [0u8; 4];
    io.read(&mut b);
    i32::from_le_bytes(b)
}

fn read_u64<R: IoRead + ?Sized>(io: &mut R) -> u64 {
    let mut b = [0u8; 8];
    io.read(&mut b);
    u64::from_le_bytes(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Cursor {
        data: Vec<u8>,
        pos: usize,
    }

    impl Cursor {
        fn new() -> Self {
            Self { data: Vec::new(), pos: 0 }
        }
    }

    impl IoWrite for Cursor {
        fn write(&mut self, src: &[u8]) {
            self.data.extend_from_slice(src);
        }
        fn n_bytes(&self) -> usize {
            self.data.len()
        }
    }

    impl IoRead for Cursor {
        fn read(&mut self, dst: &mut [u8]) {
            let end = (self.pos + dst.len()).min(self.data.len());
            let n = end - self.pos;
            dst[..n].copy_from_slice(&self.data[self.pos..end]);
            self.pos = end;
        }
        fn n_bytes(&self) -> usize {
            self.pos
        }
    }

    fn params() -> RecurrentParams {
        let mut p = RecurrentParams::default();
        p.hparams.n_embd = 16;
        p.hparams.n_layer_all = 2;
        p.hparams.n_layer_nextn = 0;
        p.hparams.ssm_d_conv = 4;
        p.hparams.ssm_d_state = 4;
        p.hparams.ssm_d_inner = 16;
        p.mem_size = 8;
        p.n_seq_max = 4;
        p
    }

    fn ubatch(seq: SeqId, pos: Pos) -> Ubatch {
        let mut u = Ubatch::default();
        u.b_equal_seqs = 1;
        u.n_tokens = 1;
        u.n_seq_tokens = 1;
        u.n_seqs = 1;
        u.n_seqs_unq = 1;
        u.n_pos = 1;
        u.pos = vec![pos];
        u.n_seq_id = vec![1];
        u.seq_id = vec![vec![seq]];
        u
    }

    #[test]
    fn seq_lifecycle() {
        let mut mem = RecurrentMemory::new(params()).unwrap();
        assert!(mem.find_slot(&ubatch(0, 0)));
        assert_eq!(mem.seq_pos_min(0), 0);
        assert_eq!(mem.seq_pos_max(0), 0);
        assert_eq!(mem.cells[0].tail, 0);

        assert!(mem.find_slot(&ubatch(1, 0)));
        mem.seq_cp(0, 2, -1, -1);
        assert!(mem.cells[mem.cells[0].tail as usize].has_seq_id(2));
        mem.seq_keep(0);
        assert!(!mem.cells[mem.cells[0].tail as usize].has_seq_id(2));

        mem.seq_add(0, -1, -1, 10);
        assert_eq!(mem.seq_pos_max(0), 10);
        mem.seq_div(0, -1, -1, 2);
        assert_eq!(mem.seq_pos_max(0), 5);

        assert!(mem.seq_rm(0, -1, -1));
        assert_eq!(mem.seq_pos_min(0), -1);
        mem.clear(true);
        assert_eq!(mem.used, 0);
        assert_eq!(mem.head, 0);
    }

    #[test]
    fn state_round_trip() {
        let mut mem = RecurrentMemory::new(params()).unwrap();
        assert!(mem.find_slot(&ubatch(0, 3)));
        let tail = mem.cells[0].tail;
        mem.cells[tail as usize].seq_id.insert(1);
        mem.cells[1].tail = tail;

        let mut cur = Cursor::new();
        mem.state_write(&mut cur, -1, 0).unwrap();
        cur.pos = 0;

        let mut restored = RecurrentMemory::new(params()).unwrap();
        restored.state_read(&mut cur, -1, 0).unwrap();
        assert_eq!(restored.seq_pos_min(0), 3);
        assert_eq!(restored.seq_pos_min(1), 3);
    }
}

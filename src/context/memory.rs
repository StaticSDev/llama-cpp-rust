//! KV/memory wiring, state save/load and memory accounting.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::rc::Rc;

use crate::batch::Ubatch;
use crate::core::{Error, Result, SeqId, Token};
use crate::kv_cache::{
    self, IoRead, IoWrite, KvCache, KvCacheParams, MemReader, MemWriter, MemoryStatus,
    NonCausalType as KvNonCausalType, RopeType as KvRopeType, UBatch,
};

use super::graph::MemorySnapshot;
use super::model::Model;
use super::LlamaContext;

pub const LLAMA_SESSION_MAGIC: u32 = 0x6767_6d6c;
pub const LLAMA_SESSION_VERSION: u32 = 1;
pub const LLAMA_STATE_SEQ_MAGIC: u32 = 0x6767_6d66;
pub const LLAMA_STATE_SEQ_VERSION: u32 = 2;
pub const LLAMA_STATE_SEQ_FLAGS_ON_DEVICE: u32 = 2;
const IO_MAGIC: u32 = 0xaf14_3cd8;

/// Convert the batch allocator's ubatch to the KV cache's ubatch.
pub fn to_kv_ubatch(u: &Ubatch) -> UBatch {
    let mut token = None;
    if !u.token.is_empty() {
        token = Some(u.token.clone());
    }
    let mixed = u.is_mixed();
    let type_embd = if mixed {
        u.type_.iter().map(|&t| t != 0).collect()
    } else if token.is_none() && !u.embd.is_empty() {
        vec![true; u.n_tokens as usize]
    } else {
        vec![false; u.n_tokens as usize]
    };
    UBatch {
        n_tokens: u.n_tokens as usize,
        n_seqs: u.n_seqs as usize,
        n_seqs_unq: u.n_seqs_unq as usize,
        seq_id_unq: u.seq_id_unq.iter().map(|&s| s as i32).collect(),
        n_seq_id: u.n_seq_id.iter().map(|&s| s.max(0) as usize).collect(),
        seq_id: u
            .seq_id
            .iter()
            .map(|ids| ids.iter().map(|&s| s as i32).collect())
            .collect(),
        pos: u.pos.clone(),
        token,
        type_embd,
        n_pos_per_embd: u.n_pos,
        equal_seqs: u.equal_seqs(),
        mixed,
    }
}

/// Map the hparams port to the KV cache's local hparams view.
pub fn kv_hparams(hp: &crate::hparams::Hparams) -> kv_cache::HParams {
    let non_causal = match hp.non_causal_type {
        crate::hparams::NonCausalType::All => KvNonCausalType::None,
        crate::hparams::NonCausalType::SwaOnly => KvNonCausalType::SwaOnly,
        crate::hparams::NonCausalType::SwaFull => KvNonCausalType::SwaFull,
        crate::hparams::NonCausalType::DenseOnly => KvNonCausalType::DenseOnly,
    };
    let rope = match hp.rope_type {
        crate::hparams::RopeType::None => KvRopeType::None,
        crate::hparams::RopeType::Normal => KvRopeType::NeoX,
        crate::hparams::RopeType::Mrope => KvRopeType::Mrope,
        crate::hparams::RopeType::Imrope => KvRopeType::Imrope,
    };
    kv_cache::HParams {
        n_layer_all: hp.n_layer_all,
        n_embd_head_k: hp.n_embd_head_k_full,
        n_embd_head_v: hp.n_embd_head_v_full,
        n_embd_head_k_full: hp.n_embd_head_k_full,
        indexer_head_size: hp.indexer_head_size,
        n_head_kv: hp.n_head_kv(0).unwrap_or(1),
        n_embd_k_gqa: hp.n_embd_k_gqa(0).unwrap_or(hp.n_embd_head_k_full),
        n_embd_v_gqa: hp.n_embd_v_gqa(0).unwrap_or(hp.n_embd_head_v_full),
        n_embd_v_gqa_max: hp.n_embd_v_gqa_max().unwrap_or(hp.n_embd_head_v_full),
        v_gqa_variable: hp.is_n_embd_v_gqa_variable().unwrap_or(false),
        n_pos_per_embd: hp.n_pos_per_embd(),
        is_mla: hp.is_mla(),
        ple_n_heads: hp.ple_n_heads,
        ple_embd_token: hp.ple_embd_token,
        use_alibi: hp.use_alibi,
        non_causal_type: non_causal,
        n_rel_attn_bkts: hp.n_rel_attn_bkts,
        rope_type: rope,
    }
}

/// Map the model architecture to the KV cache architecture enum.
pub fn kv_arch(model: &Model) -> kv_cache::Arch {
    if model.arch == super::model::Arch::Other {
        kv_cache::Arch::Other
    } else {
        kv_cache::Arch::Other
    }
}

/// Build `llama_memory_params` and the resulting KV cache.
pub fn create_memory(
    model: &Model,
    cparams: &crate::cparams::CParams,
    params: &super::params::ContextParams,
    other: Option<Rc<RefCell<KvCache>>>,
) -> Result<KvCache> {
    let mut mp = KvCacheParams::default();
    mp.hparams = kv_hparams(&model.hparams);
    mp.model = kv_cache::Model {
        arch: kv_arch(model),
        attn_rot: true,
        lid_k_hdmd: !model.hadamard.rot.is_empty(),
        dev_buft: model.dev_output().unwrap_or_else(|| "CPU".to_string()),
        layers: Vec::new(),
    };
    mp.type_k = params.type_k;
    mp.type_v = params.type_v;
    mp.unified = cparams.kv_unified;
    mp.offload = cparams.offload_kqv && !model.devices.is_empty();
    mp.kv_size = if cparams.kv_unified {
        cparams.n_ctx
    } else {
        cparams.n_ctx_seq
    };
    mp.n_seq_max = cparams.n_seq_max;
    mp.no_alloc = model.no_alloc();
    // shortcut: the KV port's other-cache handle is `Rc<KvCache>`, while the
    // context keeps its cache behind `Rc<RefCell<_>>`. Share cell views only
    // when that API accepts interior mutability.
    if other.is_some() {
        crate::llama_log_warn!("create_memory: shared KV contexts are allocated independently");
    }
    KvCache::new(mp)
}

/// Non-borrowing replacement for `llama_memory_context_ptr`.
///
/// The C++ context shares a raw pointer with its memory module. Here the
/// cache sits behind `Rc<RefCell<_>>`, so the context can keep the ubatches
/// and apply them one by one without holding a borrow across graph compute.
#[derive(Clone)]
pub struct MemoryContext {
    kv: Option<Rc<RefCell<KvCache>>>,
    status: MemoryStatus,
    ubatches: Vec<UBatch>,
    i_cur: usize,
    do_shift: bool,
    n_kv: u32,
    n_kv_rows: u32,
}

impl MemoryContext {
    pub fn failed() -> Self {
        Self {
            kv: None,
            status: MemoryStatus::FailedPrepare,
            ubatches: Vec::new(),
            i_cur: 0,
            do_shift: false,
            n_kv: 0,
            n_kv_rows: 0,
        }
    }

    pub fn full(kv: &Rc<RefCell<KvCache>>) -> Self {
        Self {
            kv: Some(Rc::clone(kv)),
            status: MemoryStatus::Success,
            ubatches: Vec::new(),
            i_cur: 0,
            do_shift: false,
            n_kv: kv.borrow().get_size(),
            n_kv_rows: 0,
        }
    }

    pub fn update(kv: &Rc<RefCell<KvCache>>, optimize: bool) -> Self {
        let mut kv_ref = kv.borrow_mut();
        let mctx = kv_ref.init_update(optimize);
        let status = mctx.get_status();
        Self {
            kv: Some(Rc::clone(kv)),
            status,
            ubatches: Vec::new(),
            i_cur: 0,
            do_shift: optimize,
            n_kv: 0,
            n_kv_rows: 0,
        }
    }

    pub fn batch(kv: &Rc<RefCell<KvCache>>, ubatches: Vec<UBatch>) -> Self {
        let status = if ubatches.is_empty() {
            MemoryStatus::FailedPrepare
        } else {
            MemoryStatus::Success
        };
        Self {
            kv: Some(Rc::clone(kv)),
            status,
            ubatches,
            i_cur: 0,
            do_shift: false,
            n_kv: 0,
            n_kv_rows: 0,
        }
    }

    pub fn get_status(&self) -> MemoryStatus {
        self.status
    }

    pub fn get_ubatch(&self) -> &UBatch {
        &self.ubatches[self.i_cur]
    }

    pub fn next(&mut self) -> bool {
        self.i_cur += 1;
        self.i_cur < self.ubatches.len()
    }

    /// Apply the current ubatch (or the pending shift) to the KV cache.
    pub fn apply(&mut self) -> bool {
        let Some(kv) = self.kv.as_ref() else {
            return false;
        };
        if self.ubatches.is_empty() {
            let mut cache = kv.borrow_mut();
            let mut mctx = cache.init_update(self.do_shift);
            let ok = mctx.apply();
            drop(mctx);
            return ok;
        }
        let ubatch = self.ubatches[self.i_cur].clone();
        let mut cache = kv.borrow_mut();
        let mut mctx = cache.init_batch(vec![ubatch]);
        let ok = mctx.apply();
        self.n_kv = mctx.get_n_kv();
        self.n_kv_rows = mctx.get_n_kv_rows();
        ok
    }

    pub fn snapshot(&self) -> MemorySnapshot {
        match self.kv.as_ref() {
            Some(kv) => {
                let cache = kv.borrow();
                MemorySnapshot {
                    n_kv: self.n_kv,
                    n_kv_rows: self.n_kv_rows,
                    type_k: cache.type_k(),
                    type_v: cache.type_v(),
                }
            }
            None => MemorySnapshot::default(),
        }
    }
}

/// `llama_memory_breakdown_data`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryBreakdownData {
    pub model: usize,
    pub context: usize,
    pub compute: usize,
}

/// `llama_io_write_i::n_bytes`, implemented by the host writer.
trait ByteCount {
    fn n_bytes(&self) -> usize;
}

impl ByteCount for MemWriter {
    fn n_bytes(&self) -> usize {
        self.data.len()
    }
}

fn write_u32<W: IoWrite>(io: &mut W, v: u32) {
    io.write_bytes(&v.to_le_bytes());
}
fn write_i32<W: IoWrite>(io: &mut W, v: i32) {
    io.write_bytes(&v.to_le_bytes());
}

fn read_u32<R: IoRead>(io: &mut R) -> Option<u32> {
    let mut buf = [0u8; 4];
    if io.read_bytes(&mut buf) {
        Some(u32::from_le_bytes(buf))
    } else {
        None
    }
}

fn read_i32<R: IoRead>(io: &mut R) -> Option<i32> {
    let mut buf = [0u8; 4];
    if io.read_bytes(&mut buf) {
        Some(i32::from_le_bytes(buf))
    } else {
        None
    }
}

impl LlamaContext {
    /// Handle pending shifts/copies and re-reserve the worst-case graph.
    pub fn memory_update(&mut self, optimize: bool) -> Result<bool> {
        let Some(kv) = self.memory.as_ref() else {
            return Ok(false);
        };
        let mut mctx = MemoryContext::update(kv, optimize);
        match mctx.get_status() {
            MemoryStatus::Success => {}
            MemoryStatus::NoUpdate => return Ok(false),
            MemoryStatus::FailedPrepare => {
                crate::llama_log_error!("memory_update: failed to prepare memory update");
                return Ok(false);
            }
        }
        for res in &mut self.gf_res_prev {
            res.reset();
        }
        self.gf_res_prev_active = None;
        if !mctx.apply() {
            crate::llama_log_error!("memory_update: failed to apply memory update");
        }
        drop(mctx);

        let snapshot = self.memory.as_ref().map(|kv| {
            let cache = kv.borrow();
            MemorySnapshot {
                n_kv: cache.get_size(),
                n_kv_rows: 0,
                type_k: cache.type_k(),
                type_v: cache.type_v(),
            }
        });
        let n_seqs = self.cparams.n_seq_max.max(1);
        let n_tokens = self.cparams.n_ctx.min(self.cparams.n_ubatch).max(1);
        let n_outputs = n_tokens.min(self.cparams.n_outputs_max).max(1);
        if self
            .graph_reserve(n_tokens, n_seqs, n_outputs, snapshot, false, None)
            .is_err()
        {
            crate::llama_log_error!("memory_update: failed to reserve graph after the update");
        }
        Ok(true)
    }

    /// Split the initialized batch into KV-cache ubatches.
    pub fn prepare_ubatches(&mut self, embd_all: bool) -> Result<Vec<(Ubatch, UBatch)>> {
        let n_ubatch = self.cparams.n_ubatch.max(1);
        let n_tokens = self.balloc.get_n_tokens();
        let mut res = Vec::new();
        while self.balloc.get_n_used() < n_tokens as u32 {
            let u = if embd_all {
                self.balloc.split_equal(n_ubatch, false, 0)
            } else {
                self.balloc.split_seq(n_ubatch)
            };
            if u.n_tokens == 0 {
                break;
            }
            res.push((u.clone(), to_kv_ubatch(&u)));
        }
        if res.is_empty() {
            return Err(Error::new("failed to initialize batch: no ubatches"));
        }
        Ok(res)
    }

    fn state_write_data<W: IoWrite + ByteCount>(&self, io: &mut W) -> usize {
        let arch = self.model.arch.name();
        let bytes = arch.as_bytes();
        write_u32(io, bytes.len() as u32);
        io.write_bytes(bytes);
        if let Some(kv) = self.memory.as_ref() {
            kv.borrow().state_write(io, -1, 0);
        }
        io.n_bytes()
    }

    fn state_read_data<R: IoRead>(&mut self, io: &mut R) -> Result<()> {
        let len = read_u32(io).ok_or_else(|| Error::new("state: truncated arch string"))? as usize;
        let mut bytes = vec![0u8; len];
        if !io.read_bytes(&mut bytes) {
            return Err(Error::new("state: truncated arch string"));
        }
        let arch = std::str::from_utf8(&bytes).unwrap_or("");
        if arch != self.model.arch.name() {
            return Err(Error::new(format!(
                "wrong model arch: '{}' instead of '{}'",
                arch,
                self.model.arch.name()
            )));
        }
        if let Some(kv) = self.memory.as_ref() {
            kv.borrow_mut().state_read(io, -1, 0)?;
        }
        Ok(())
    }

    fn state_seq_write_data<W: IoWrite + ByteCount>(
        &self,
        io: &mut W,
        seq_id: SeqId,
        flags: u32,
    ) -> usize {
        if let Some(kv) = self.memory.as_ref() {
            kv.borrow().state_write(io, seq_id as i32, flags);
        }
        io.n_bytes()
    }

    fn state_seq_read_data<R: IoRead>(
        &mut self,
        io: &mut R,
        seq_id: SeqId,
        flags: u32,
    ) -> Result<()> {
        if let Some(kv) = self.memory.as_ref() {
            kv.borrow_mut().state_read(io, seq_id as i32, flags)?;
        }
        Ok(())
    }

    /// `llama_state_get_size`.
    pub fn state_get_size(&self) -> usize {
        let mut io = MemWriter::default();
        let _ = self.state_write_data(&mut io);
        io.data.len()
    }

    /// `llama_state_get_data`.
    pub fn state_get_data(&self, dst: &mut [u8]) -> usize {
        let mut io = MemWriter::default();
        let _ = self.state_write_data(&mut io);
        let n = io.data.len().min(dst.len());
        dst[..n].copy_from_slice(&io.data[..n]);
        n
    }

    /// `llama_state_set_data`.
    pub fn state_set_data(&mut self, src: &[u8]) -> usize {
        let mut io = MemReader::new(src.to_vec());
        match self.state_read_data(&mut io) {
            Ok(()) => io.pos,
            Err(err) => {
                crate::llama_log_error!("state_set_data: {}", err);
                0
            }
        }
    }

    /// `llama_state_seq_get_size_ext`.
    pub fn state_seq_get_size(&self, seq_id: SeqId, flags: u32) -> usize {
        let mut io = MemWriter::default();
        write_u32(&mut io, IO_MAGIC);
        write_i32(&mut io, seq_id as i32);
        let _ = self.state_seq_write_data(&mut io, seq_id, flags);
        io.data.len()
    }

    /// `llama_state_seq_get_data_ext`.
    pub fn state_seq_get_data(&mut self, seq_id: SeqId, dst: &mut [u8], flags: u32) -> usize {
        let mut io = MemWriter::default();
        write_u32(&mut io, IO_MAGIC);
        write_i32(&mut io, seq_id as i32);
        let _ = self.state_seq_write_data(&mut io, seq_id, flags);
        let n = io.data.len().min(dst.len());
        dst[..n].copy_from_slice(&io.data[..n]);
        if flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE != 0 {
            self.mem_storage.insert(seq_id, io.data);
        }
        n
    }

    /// `llama_state_seq_set_data_ext`.
    pub fn state_seq_set_data(&mut self, seq_id: SeqId, src: &[u8], flags: u32) -> usize {
        let mut io = MemReader::new(src.to_vec());
        match read_u32(&mut io) {
            Some(magic) if magic == IO_MAGIC => {}
            _ => {
                crate::llama_log_error!("wrong sequence state magic");
                return 0;
            }
        }
        match read_i32(&mut io) {
            Some(id) if id as SeqId == seq_id => {}
            _ => {
                crate::llama_log_error!("wrong sequence state id");
                return 0;
            }
        }
        if flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE != 0 && !self.mem_storage.contains_key(&seq_id) {
            crate::llama_log_error!("no device state stored for seq_id {}", seq_id);
            return 0;
        }
        match self.state_seq_read_data(&mut io, seq_id, flags) {
            Ok(()) => io.pos,
            Err(err) => {
                crate::llama_log_error!("state_seq_set_data: {}", err);
                0
            }
        }
    }

    /// `llama_state_load_file`.
    pub fn state_load_file(&mut self, path: &str, tokens_out: &mut [Token]) -> Result<usize> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| Error::new(format!("failed to open session file: {}", e)))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| Error::new(format!("failed to read session file: {}", e)))?;
        let mut io = MemReader::new(buf);
        match read_u32(&mut io) {
            Some(LLAMA_SESSION_MAGIC) => {}
            _ => return Err(Error::new("unknown magic for session file")),
        }
        match read_u32(&mut io) {
            Some(LLAMA_SESSION_VERSION) => {}
            _ => return Err(Error::new("unknown version for session file")),
        }
        let n_tokens =
            read_u32(&mut io).ok_or_else(|| Error::new("truncated session file"))? as usize;
        if n_tokens > tokens_out.len() {
            return Err(Error::new(format!(
                "token count exceeded capacity: {} > {}",
                n_tokens,
                tokens_out.len()
            )));
        }
        for slot in tokens_out.iter_mut().take(n_tokens) {
            *slot = match read_i32(&mut io) {
                Some(v) => v,
                None => return Err(Error::new("truncated token list")),
            };
        }
        self.state_read_data(&mut io)?;
        let n_read = io.pos;
        let remaining = io.data.len() - io.pos;
        if n_read != remaining {
            return Err(Error::new(format!(
                "did not read all session data: size {}, got {}",
                remaining, n_read
            )));
        }
        Ok(n_tokens)
    }

    /// `llama_state_save_file`.
    pub fn state_save_file(&mut self, path: &str, tokens: &[Token]) -> Result<usize> {
        let mut io = MemWriter::default();
        write_u32(&mut io, LLAMA_SESSION_MAGIC);
        write_u32(&mut io, LLAMA_SESSION_VERSION);
        write_u32(&mut io, tokens.len() as u32);
        for &token in tokens {
            write_i32(&mut io, token);
        }
        let _ = self.state_write_data(&mut io);
        let mut file = std::fs::File::create(path)
            .map_err(|e| Error::new(format!("failed to create session file: {}", e)))?;
        file.write_all(&io.data)
            .map_err(|e| Error::new(format!("failed to write session file: {}", e)))?;
        Ok(tokens.len())
    }

    /// `llama_get_memory_breakdown`.
    pub fn memory_breakdown(&self) -> BTreeMap<String, MemoryBreakdownData> {
        let mut ret: BTreeMap<String, MemoryBreakdownData> = BTreeMap::new();
        for (buft, size) in self.model.memory_breakdown() {
            ret.entry(buft).or_default().model += size;
        }
        if let Some(kv) = self.memory.as_ref() {
            for (buft, size) in kv.borrow().memory_breakdown() {
                ret.entry(buft).or_default().context += size;
            }
        }
        for backend in &self.backends {
            let buft = if backend.is_cpu {
                "CPU".to_string()
            } else {
                backend.name.clone()
            };
            let size = if self.model.no_alloc {
                *self.backend_buf_exp_size.get(backend.id).unwrap_or(&0)
            } else {
                self.sched.params.max_nodes * 64
            };
            ret.entry(buft).or_default().compute += size;
        }
        ret
    }
}

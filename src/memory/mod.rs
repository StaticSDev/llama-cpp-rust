//! Rust port of llama.cpp's base memory interface.
//!
//! Covers `llama_memory_params`, `llama_memory_bufs`, `llama_memory_status`,
//! `llama_memory_context_i`, `llama_memory_i` and the helpers from
//! `llama-memory.h` / `llama-memory.cpp`. ggml contexts and backend buffers are
//! represented by safe stand-ins: a context tracks tensor byte sizes and a
//! buffer tracks `buft`, `size` and `name`. No FFI or `unsafe` is needed.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::batch::{BatchAllocr, Pos, Ubatch};
use crate::core::{Error, Result};
use crate::io::{IoRead, IoWrite};

pub use crate::batch::ContextType;
pub use crate::kv_cache::{GgmlType, LSeqId, MemoryBuf, Tensor};

pub mod hybrid;
pub mod recurrent;

pub use hybrid::*;
pub use recurrent::*;

/// `llama_seq_id` sentinel used by the state API to mean "all sequences".
pub const SEQ_ID_ALL: LSeqId = -1;

/// `llama_state_seq_flags` (`uint32_t`).
pub type StateSeqFlags = u32;
pub const STATE_SEQ_FLAGS_NONE: StateSeqFlags = 0;
pub const STATE_SEQ_FLAGS_SWA_ONLY: StateSeqFlags = 1;
pub const STATE_SEQ_FLAGS_PARTIAL_ONLY: StateSeqFlags = 1;
pub const STATE_SEQ_FLAGS_ON_DEVICE: StateSeqFlags = 2;

/// `std::map<ggml_backend_buffer_type_t, size_t>` stand-in, keyed by buft name.
pub type MemoryBreakdown = BTreeMap<String, usize>;

/// `llama_memory_status`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemoryStatus {
    #[default]
    Success,
    NoUpdate,
    FailedPrepare,
    FailedCompute,
}

/// `llama_memory_status_combine`.
///
/// A failure wins. Otherwise any update wins over no update.
pub fn memory_status_combine(s0: MemoryStatus, s1: MemoryStatus) -> MemoryStatus {
    let mut has_update = false;

    if s0 == MemoryStatus::FailedPrepare || s0 == MemoryStatus::FailedCompute {
        return s0;
    }
    if s0 == MemoryStatus::Success {
        has_update = true;
    }

    if s1 == MemoryStatus::FailedPrepare || s1 == MemoryStatus::FailedCompute {
        return s1;
    }
    if s1 == MemoryStatus::Success {
        has_update = true;
    }

    if has_update {
        MemoryStatus::Success
    } else {
        MemoryStatus::NoUpdate
    }
}

/// `llama_memory_status_is_fail`.
pub fn memory_status_is_fail(status: MemoryStatus) -> bool {
    matches!(status, MemoryStatus::FailedPrepare | MemoryStatus::FailedCompute)
}

/// `llama_memory_params`.
pub struct MemoryParams {
    pub type_k: GgmlType,
    pub type_v: GgmlType,
    pub swa_full: bool,
    pub ctx_type: ContextType,
    pub mem_other: Option<MemoryPtr>,
    pub no_alloc: bool,
}

impl Default for MemoryParams {
    fn default() -> Self {
        Self {
            type_k: GgmlType::F16,
            type_v: GgmlType::F16,
            swa_full: false,
            ctx_type: ContextType::Default,
            mem_other: None,
            no_alloc: false,
        }
    }
}

/// Stand-in for `ggml_context`: the byte size of each tensor it owns.
#[derive(Clone, Debug, Default)]
pub struct TensorContext {
    pub tensor_sizes: Vec<usize>,
}

impl TensorContext {
    pub fn new(tensor_sizes: Vec<usize>) -> Self {
        Self { tensor_sizes }
    }

    pub fn size(&self) -> usize {
        self.tensor_sizes.iter().sum()
    }
}

/// The tensor-aware half of `llama_io_write_i`.
///
/// Users only implement `IoWrite`; the default `write_tensor` writes the
/// tensor bytes through `write`. Collectors override it.
pub trait TensorIoWrite: IoWrite {
    fn write_tensor(&mut self, tensor: &Tensor, offset: usize, size: usize) {
        let storage = tensor.storage.borrow();
        let off = tensor.offset + offset;
        let end = (off + size).min(storage.len());
        if off < end {
            self.write(&storage[off..end]);
        }
    }
}

/// The tensor-aware half of `llama_io_read_i`.
pub trait TensorIoRead: IoRead {
    fn read_tensor(&mut self, tensor: &Tensor, offset: usize, size: usize) -> bool {
        let mut buf = vec![0u8; size];
        self.read(&mut buf);
        let mut storage = tensor.storage.borrow_mut();
        let off = tensor.offset + offset;
        let end = (off + size).min(storage.len());
        if off < end {
            let n = end - off;
            storage[off..end].copy_from_slice(&buf[..n]);
        }
        true
    }
}

/// `llama_memory_bufs`.
#[derive(Clone, Debug)]
pub struct MemoryBufs {
    pub bufs: Vec<(TensorContext, MemoryBuf)>,
    pub no_alloc: bool,
}

impl MemoryBufs {
    pub fn new(no_alloc: bool) -> Self {
        Self { bufs: Vec::new(), no_alloc }
    }

    /// `llama_memory_bufs::alloc_buf`.
    ///
    /// With `no_alloc` a zero-size dummy buffer is attached, but `size` keeps
    /// the total tensor size so the memory use can still be measured.
    pub fn alloc_buf(ctx: &TensorContext, buft: &str, no_alloc: bool, name: &str) -> Result<MemoryBuf> {
        if buft.is_empty() {
            return Err(Error::new(format!("failed to allocate buffer for {}", name)));
        }
        let _ = no_alloc;
        Ok(MemoryBuf {
            name: name.to_string(),
            buft: buft.to_string(),
            size: ctx.size(),
        })
    }

    /// `llama_memory_bufs::emplace`; returns the buffer for logging.
    pub fn emplace(&mut self, ctx: TensorContext, buft: &str, name: &str) -> Result<MemoryBuf> {
        let buf = Self::alloc_buf(&ctx, buft, self.no_alloc, name)?;
        self.bufs.push((ctx, buf.clone()));
        Ok(buf)
    }

    /// `llama_memory_bufs::memory_breakdown`.
    pub fn memory_breakdown(&self) -> MemoryBreakdown {
        let mut ret = MemoryBreakdown::new();
        for (ctx, buf) in &self.bufs {
            let size = if self.no_alloc { ctx.size() } else { buf.size };
            *ret.entry(buf.buft.clone()).or_insert(0) += size;
        }
        ret
    }

    pub fn empty(&self) -> bool {
        self.bufs.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &(TensorContext, MemoryBuf)> {
        self.bufs.iter()
    }
}

/// Stand-in for `llama_context`, passed to `MemoryI::init_update`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Context;

/// `llama_memory_context_i`.
pub trait MemoryContextI {
    /// Consume the current ubatch and move to the next one; false when done.
    fn next(&mut self) -> bool;

    /// Apply the memory state for the current ubatch; false on failure.
    fn apply(&mut self) -> bool;

    fn ubatch(&self) -> &Ubatch;

    fn status(&self) -> MemoryStatus;
}

/// `llama_memory_context_ptr`.
pub type MemoryContextPtr<'a> = Box<dyn MemoryContextI + 'a>;

/// `llama_memory_i::layer_filter_cb`.
pub type LayerFilterCb = fn(i32) -> bool;
/// `llama_memory_i::layer_reuse_cb`; negative means no reuse.
pub type LayerReuseCb = fn(i32) -> i32;
/// `llama_memory_i::layer_share_cb`.
pub type LayerShareCb = fn(i32) -> i32;

/// `llama_memory_i`.
pub trait MemoryI {
    fn init_batch<'a>(
        &'a mut self,
        balloc: &'a mut BatchAllocr,
        n_ubatch: u32,
        embd_all: bool,
    ) -> MemoryContextPtr<'a>;

    /// Simulate a full cache for worst-case compute buffers.
    fn init_full(&self) -> MemoryContextPtr<'_>;

    /// Prepare pending shifts/copies; status is `NoUpdate` when idle.
    fn init_update<'a>(&'a mut self, lctx: &'a mut Context, optimize: bool) -> MemoryContextPtr<'a>;

    fn get_can_shift(&self) -> bool;

    /// `data == true` also clears the data buffers, not just the metadata.
    fn clear(&mut self, data: bool);

    fn seq_rm(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos) -> bool;
    fn seq_cp(&mut self, seq_id_src: LSeqId, seq_id_dst: LSeqId, p0: Pos, p1: Pos);
    fn seq_keep(&mut self, seq_id: LSeqId);
    fn seq_add(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, shift: Pos);
    fn seq_div(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, d: i32);

    fn seq_pos_min(&self, seq_id: LSeqId) -> Pos;
    fn seq_pos_max(&self, seq_id: LSeqId) -> Pos;

    fn memory_breakdown(&self) -> MemoryBreakdown;

    fn state_write(&self, io: &mut dyn TensorIoWrite, seq_id: LSeqId, flags: StateSeqFlags);
    fn state_read(&mut self, io: &mut dyn TensorIoRead, seq_id: LSeqId, flags: StateSeqFlags);
}

/// `llama_memory_ptr`.
pub type MemoryPtr = Box<dyn MemoryI>;

/// Unique handle for a tensor's backing storage.
pub type BufferHandle = Rc<RefCell<Vec<u8>>>;

fn same_tensor(a: &Tensor, b: &Tensor) -> bool {
    Rc::as_ptr(&a.storage) == Rc::as_ptr(&b.storage) && a.offset == b.offset && a.name == b.name
}

/// `llama_memory_tensors_writer`: collects visited tensors without keeping data.
#[derive(Default)]
pub struct StateWriter {
    pub tensors: Vec<Tensor>,
    pub size: usize,
}

impl IoWrite for StateWriter {
    fn write(&mut self, src: &[u8]) {
        self.size += src.len();
    }

    fn n_bytes(&self) -> usize {
        self.size
    }
}

impl TensorIoWrite for StateWriter {
    fn write_tensor(&mut self, tensor: &Tensor, _offset: usize, size: usize) {
        if !self.tensors.iter().any(|t| same_tensor(t, tensor)) {
            self.tensors.push(tensor.clone());
        }
        self.size += size;
    }
}

/// `llama_memory_get_tensors`: the tensors a state write would visit.
pub fn llama_memory_get_tensors(mem: Option<&dyn MemoryI>) -> Vec<Tensor> {
    let mem = match mem {
        Some(mem) => mem,
        None => return Vec::new(),
    };
    let mut writer = StateWriter::default();
    mem.state_write(&mut writer, SEQ_ID_ALL, STATE_SEQ_FLAGS_NONE);
    writer.tensors
}

/// `llama_memory_get_buffers`: distinct buffers behind those tensors.
pub fn llama_memory_get_buffers(mem: Option<&dyn MemoryI>) -> Vec<BufferHandle> {
    let mut buffers: Vec<BufferHandle> = Vec::new();
    for tensor in llama_memory_get_tensors(mem) {
        if !buffers.iter().any(|b| Rc::ptr_eq(b, &tensor.storage)) {
            buffers.push(tensor.storage.clone());
        }
    }
    buffers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_combine_prefers_failures_then_updates() {
        assert_eq!(memory_status_combine(MemoryStatus::Success, MemoryStatus::NoUpdate), MemoryStatus::Success);
        assert_eq!(memory_status_combine(MemoryStatus::NoUpdate, MemoryStatus::NoUpdate), MemoryStatus::NoUpdate);
        assert_eq!(
            memory_status_combine(MemoryStatus::Success, MemoryStatus::FailedCompute),
            MemoryStatus::FailedCompute
        );
        assert!(memory_status_is_fail(MemoryStatus::FailedPrepare));
        assert!(!memory_status_is_fail(MemoryStatus::Success));
    }

    #[test]
    fn bufs_breakdown_uses_simulated_size_when_no_alloc() {
        let mut bufs = MemoryBufs::new(true);
        bufs.emplace(TensorContext::new(vec![16, 32]), "CPU", "kv").unwrap();
        bufs.emplace(TensorContext::new(vec![8]), "GPU", "rot").unwrap();
        assert!(!bufs.empty());
        let breakdown = bufs.memory_breakdown();
        assert_eq!(breakdown.get("CPU"), Some(&48));
        assert_eq!(breakdown.get("GPU"), Some(&8));

        let real = MemoryBufs::new(false);
        assert!(real.empty());
    }
}

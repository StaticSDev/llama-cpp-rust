//! Rust translation of `ggml-alloc.c` / `ggml-alloc.h`.
//!
//! Dry port: the allocator, its free-block planner and the graph allocator are
//! translated as-is. Raw pointers become `Rc<RefCell<..>>` references, so tensor
//! identity is the address of the `Rc` allocation, and "tensor data" is an
//! offset inside a backend buffer instead of a host pointer. Backend buffers do
//! not own host memory here: only sizes, alignment and address assignment are
//! modelled.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::{Error, Result};

pub const GGML_MAX_DIMS: usize = 4;
pub const GGML_MAX_SRC: usize = 10;
pub const GGML_VBUFFER_MAX_CHUNKS: usize = 16;
pub const GGML_TENSOR_FLAG_INPUT: u32 = 1;
pub const GGML_TENSOR_FLAG_OUTPUT: u32 = 2;

const MAX_FREE_BLOCKS: usize = 256;

static NEXT_BUFFER_ID: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// small ggml stand-ins
// ---------------------------------------------------------------------------

/// `GGML_PAD(x, n)`.
fn ggml_pad(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}

/// `aligned_offset`: round `offset` so that `base + offset` is aligned.
/// `base == 0` is the `NULL` case used by the dynamic allocator.
pub fn aligned_offset(base: usize, offset: usize, alignment: usize) -> usize {
    assert!(alignment != 0 && alignment.is_power_of_two());
    let align = (alignment - ((base + offset) % alignment)) % alignment;
    offset + align
}

/// Ops that may reuse an input buffer. Mirrors `ggml_op_can_inplace`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocOp {
    None,
    Fill,
    Scale,
    Add,
    AddId,
    Sub,
    Mul,
    Div,
    Sqr,
    Sqrt,
    Log,
    Unary,
    Rope,
    RopeBack,
    SiluBack,
    SigmoidBack,
    RmsNorm,
    RmsNormBack,
    Clamp,
    SoftMax,
    SoftMaxBack,
    Other,
}

pub fn ggml_op_can_inplace(op: AllocOp) -> bool {
    matches!(
        op,
        AllocOp::Fill
            | AllocOp::Scale
            | AllocOp::Add
            | AllocOp::AddId
            | AllocOp::Sub
            | AllocOp::Mul
            | AllocOp::Div
            | AllocOp::Sqr
            | AllocOp::Sqrt
            | AllocOp::Log
            | AllocOp::Unary
            | AllocOp::Rope
            | AllocOp::RopeBack
            | AllocOp::SiluBack
            | AllocOp::SigmoidBack
            | AllocOp::RmsNorm
            | AllocOp::RmsNormBack
            | AllocOp::Clamp
            | AllocOp::SoftMax
            | AllocOp::SoftMaxBack
    )
}

/// Where a tensor's data lives. `None` in `AllocTensor::data` is a null pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorData {
    External(Rc<Vec<u8>>),
    Buffer { buffer: usize, offset: usize },
}

/// Pointer equality for `TensorData`, as in `view_src->data == parent->data`.
fn same_data(a: &Option<TensorData>, b: &Option<TensorData>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(TensorData::External(x)), Some(TensorData::External(y))) => Rc::ptr_eq(x, y),
        (
            Some(TensorData::Buffer { buffer: b1, offset: o1 }),
            Some(TensorData::Buffer { buffer: b2, offset: o2 }),
        ) => b1 == b2 && o1 == o2,
        _ => false,
    }
}

/// Minimal `ggml_tensor` needed by the allocator.
#[derive(Debug)]
pub struct AllocTensor {
    pub name: String,
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
    pub type_size: usize,
    pub blck_size: i64,
    pub data: Option<TensorData>,
    pub buffer: Option<usize>,
    pub view_src: Option<TensorRef>,
    pub src: [Option<TensorRef>; GGML_MAX_SRC],
    pub op: AllocOp,
    pub flags: u32,
}

pub type TensorRef = Rc<RefCell<AllocTensor>>;

impl AllocTensor {
    pub fn new(name: impl Into<String>, ne: [i64; GGML_MAX_DIMS], type_size: usize, blck_size: i64) -> Self {
        let mut nb = [type_size; GGML_MAX_DIMS];
        for i in 1..GGML_MAX_DIMS {
            nb[i] = nb[i - 1] * ne[i - 1].max(0) as usize;
        }
        Self {
            name: name.into(),
            ne,
            nb,
            type_size,
            blck_size,
            data: None,
            buffer: None,
            view_src: None,
            src: std::array::from_fn(|_| None),
            op: AllocOp::None,
            flags: 0,
        }
    }

    /// Common case: contiguous f32 tensor.
    pub fn new_f32(name: impl Into<String>, ne: [i64; GGML_MAX_DIMS]) -> Self {
        Self::new(name, ne, std::mem::size_of::<f32>(), 1)
    }
}

/// `size_t ggml_nbytes(const struct ggml_tensor *)`.
pub fn ggml_nbytes(t: &AllocTensor) -> usize {
    let mut nelements = 1i128;
    for &ne in &t.ne {
        nelements *= ne as i128;
    }
    let blck = t.blck_size.max(1) as i128;
    let nbytes = nelements * t.type_size as i128 / blck;
    if nbytes < 0 {
        0
    } else {
        nbytes as usize
    }
}

pub fn ggml_impl_is_view(t: &AllocTensor) -> bool {
    t.view_src.is_some()
}

/// `ggml_are_same_layout`.
fn ggml_are_same_layout(a: &AllocTensor, b: &AllocTensor) -> bool {
    a.type_size == b.type_size
        && a.blck_size == b.blck_size
        && a.ne == b.ne
        && a.nb == b.nb
}

/// Minimal `ggml_cgraph`.
#[derive(Debug, Default)]
pub struct CGraph {
    pub nodes: Vec<TensorRef>,
    pub leafs: Vec<TensorRef>,
}

impl CGraph {
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
    pub fn n_leafs(&self) -> usize {
        self.leafs.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    Compute,
}

/// Stand-in for `ggml_backend_buffer_type`. `id` keeps pointer identity: two
/// buffer types with equal fields are still distinct, as in C.
#[derive(Debug, Clone)]
pub struct BackendBufferType {
    pub id: usize,
    pub alignment: usize,
    pub max_size: usize,
    pub name: String,
}

impl BackendBufferType {
    pub fn new(alignment: usize, max_size: usize, name: impl Into<String>) -> Self {
        assert!(alignment != 0 && alignment.is_power_of_two());
        Self {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            alignment,
            max_size,
            name: name.into(),
        }
    }

    pub fn same(&self, other: &Self) -> bool {
        self.id == other.id
    }

    pub fn get_alignment(&self) -> usize {
        self.alignment
    }

    pub fn get_max_size(&self) -> usize {
        self.max_size
    }

    pub fn get_alloc_size(&self, tensor: &AllocTensor) -> usize {
        ggml_nbytes(tensor)
    }

    pub fn get_alloc_size_n(&self, tensors: &[TensorRef]) -> usize {
        tensors.iter().map(|t| self.get_alloc_size(&t.borrow())).sum()
    }

    /// `ggml_backend_buft_alloc_buffer`. No host memory is reserved in this
    /// dry port; failure is only reported for impossible sizes.
    pub fn alloc_buffer(&self, size: usize, usage: BufferUsage) -> Option<Rc<RefCell<BackendBuffer>>> {
        if size > isize::MAX as usize {
            return None;
        }
        Some(Rc::new(RefCell::new(BackendBuffer {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            size,
            align: self.alignment,
            usage,
            base: 0,
        })))
    }
}

/// Stand-in for `ggml_backend_buffer`. `base` is a simulated aligned address.
#[derive(Debug)]
pub struct BackendBuffer {
    pub id: usize,
    pub size: usize,
    pub align: usize,
    pub usage: BufferUsage,
    pub base: usize,
}

impl BackendBuffer {
    pub fn get_base(&self) -> usize {
        self.base
    }
    pub fn get_size(&self) -> usize {
        self.size
    }
    pub fn get_alignment(&self) -> usize {
        self.align
    }
    pub fn get_alloc_size(&self, tensor: &AllocTensor) -> usize {
        ggml_nbytes(tensor)
    }
    pub fn set_usage(&mut self, usage: BufferUsage) {
        self.usage = usage;
    }
    pub fn reset(&mut self) {
        // No host data stored, so there is nothing to invalidate.
    }

    /// `ggml_backend_tensor_alloc`: bind a tensor to this buffer at `offset`.
    pub fn tensor_alloc(buffer: &Rc<RefCell<Self>>, tensor: &TensorRef, offset: usize) {
        let id = buffer.borrow().id;
        let mut t = tensor.borrow_mut();
        t.buffer = Some(id);
        t.data = Some(TensorData::Buffer { buffer: id, offset });
    }
}

// ---------------------------------------------------------------------------
// tallocr
// ---------------------------------------------------------------------------

/// `struct ggml_tallocr`.
#[derive(Debug)]
pub struct GgmlTallocr {
    pub buffer: Rc<RefCell<BackendBuffer>>,
    pub base: usize,
    pub alignment: usize,
    pub offset: usize,
}

pub fn ggml_tallocr_new(buffer: Rc<RefCell<BackendBuffer>>) -> GgmlTallocr {
    let (base, align) = {
        let b = buffer.borrow();
        (b.get_base(), b.get_alignment())
    };
    assert!(align != 0 && align.is_power_of_two());
    GgmlTallocr {
        buffer,
        base,
        alignment: align,
        offset: aligned_offset(base, 0, align),
    }
}

pub fn ggml_tallocr_alloc(talloc: &mut GgmlTallocr, tensor: &TensorRef) -> Result<()> {
    let size = ggml_pad(talloc.buffer.borrow().get_alloc_size(&tensor.borrow()), talloc.alignment);
    let buffer_size = talloc.buffer.borrow().get_size();
    if talloc.offset + size > buffer_size {
        return Err(Error::new(format!(
            "ggml_tallocr_alloc: not enough space in the buffer to allocate {} (needed {}, available {})",
            tensor.borrow().name,
            size,
            buffer_size - talloc.offset
        )));
    }

    let addr = talloc.base + talloc.offset;
    talloc.offset += size;
    assert!(addr % talloc.alignment == 0);

    let buffer = Rc::clone(&talloc.buffer);
    BackendBuffer::tensor_alloc(&buffer, tensor, addr - talloc.base);
    Ok(())
}

// ---------------------------------------------------------------------------
// dynamic tensor allocator
// ---------------------------------------------------------------------------

/// Relative address inside a virtual (possibly chunked) buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferAddress {
    pub chunk: isize,
    pub offset: usize,
}

pub const GGML_BUFFER_ADDRESS_INVALID: BufferAddress = BufferAddress {
    chunk: -1,
    offset: usize::MAX,
};

impl Default for BufferAddress {
    fn default() -> Self {
        Self { chunk: 0, offset: 0 }
    }
}

pub fn ggml_buffer_address_less(a: BufferAddress, b: BufferAddress) -> bool {
    if a.chunk != b.chunk {
        a.chunk < b.chunk
    } else {
        a.offset < b.offset
    }
}

#[derive(Debug, Clone, Copy)]
struct FreeBlock {
    offset: usize,
    size: usize,
}

#[derive(Debug)]
struct TallocrChunk {
    free_blocks: Vec<FreeBlock>,
    max_size: usize,
}

impl TallocrChunk {
    fn insert_block(&mut self, offset: usize, size: usize) {
        assert!(self.free_blocks.len() < MAX_FREE_BLOCKS, "out of free blocks");
        let mut insert_pos = 0;
        while insert_pos < self.free_blocks.len() && self.free_blocks[insert_pos].offset < offset {
            insert_pos += 1;
        }
        self.free_blocks.insert(insert_pos, FreeBlock { offset, size });
    }
}

/// `struct ggml_dyn_tallocr`.
#[derive(Debug)]
pub struct DynTallocr {
    pub alignment: usize,
    pub max_chunk_size: usize,
    chunks: Vec<TallocrChunk>,
}

impl DynTallocr {
    pub fn new(alignment: usize, max_buffer_size: usize) -> Self {
        assert!(alignment != 0 && alignment.is_power_of_two());
        Self {
            alignment,
            max_chunk_size: max_buffer_size.min(usize::MAX / 2),
            chunks: Vec::new(),
        }
    }

    pub fn n_chunks(&self) -> usize {
        self.chunks.len()
    }

    pub fn max_size(&self, chunk: usize) -> usize {
        if chunk < self.chunks.len() {
            self.chunks[chunk].max_size
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.chunks.clear();
    }

    fn new_chunk(alloc: &mut Self, min_size: usize) -> Option<usize> {
        if alloc.chunks.len() >= GGML_VBUFFER_MAX_CHUNKS {
            return None;
        }
        let mut size = min_size.max(alloc.max_chunk_size);
        if alloc.chunks.len() == GGML_VBUFFER_MAX_CHUNKS - 1 {
            size = usize::MAX / 2;
        }
        alloc.chunks.push(TallocrChunk {
            free_blocks: vec![FreeBlock { offset: 0, size }],
            max_size: 0,
        });
        Some(alloc.chunks.len() - 1)
    }

    /// Reserve `size` bytes; returns its address in the virtual buffer.
    pub fn alloc(&mut self, size: usize, _tensor: &TensorRef) -> Result<BufferAddress> {
        let size = aligned_offset(0, size, self.alignment);
        let mut best_fit_chunk: isize = -1;
        let mut best_fit_block: isize = -1;
        let mut max_avail = 0usize;

        // Find the best fit excluding the last block of every chunk.
        for (c, chunk) in self.chunks.iter().enumerate() {
            let mut best_fit_size = usize::MAX;
            let last = chunk.free_blocks.len().saturating_sub(1);
            for i in 0..last {
                let block = &chunk.free_blocks[i];
                max_avail = max_avail.max(block.size);
                if block.size >= size && block.size <= best_fit_size {
                    best_fit_chunk = c as isize;
                    best_fit_block = i as isize;
                    best_fit_size = block.size;
                }
            }
        }

        if best_fit_block == -1 {
            // Try the last block; this may grow the chunk's size.
            let mut best_reuse = i64::MIN;
            for (c, chunk) in self.chunks.iter().enumerate() {
                if let Some(block) = chunk.free_blocks.last() {
                    max_avail = max_avail.max(block.size);
                    let reuse_factor = chunk.max_size.wrapping_sub(block.offset).wrapping_sub(size) as i64;
                    let better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                    let better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                    if block.size >= size && (better_reuse || better_fit) {
                        best_fit_chunk = c as isize;
                        best_fit_block = (chunk.free_blocks.len() - 1) as isize;
                        best_reuse = reuse_factor;
                    }
                }
            }
        }

        if best_fit_block == -1 {
            match Self::new_chunk(self, size) {
                Some(c) => {
                    best_fit_chunk = c as isize;
                    best_fit_block = 0;
                }
                None => {
                    return Err(Error::new(format!(
                        "ggml_dyn_tallocr_alloc: not enough space to allocate {} bytes, largest block available {} bytes",
                        size, max_avail
                    )));
                }
            }
        }
        assert!(best_fit_chunk >= 0);

        let chunk = &mut self.chunks[best_fit_chunk as usize];
        let block_idx = best_fit_block as usize;
        let block = &mut chunk.free_blocks[block_idx];
        let addr = BufferAddress {
            chunk: best_fit_chunk,
            offset: block.offset,
        };
        block.offset += size;
        block.size -= size;
        if block.size == 0 {
            chunk.free_blocks.remove(block_idx);
        }

        chunk.max_size = chunk.max_size.max(addr.offset + size);
        Ok(addr)
    }

    /// Return a previously reserved range to the free-block list.
    pub fn free_bytes(&mut self, addr: BufferAddress, size: usize) {
        let size = aligned_offset(0, size, self.alignment);
        let chunk = &mut self.chunks[addr.chunk as usize];

        for i in 0..chunk.free_blocks.len() {
            let block = &chunk.free_blocks[i];
            if block.offset + block.size == addr.offset {
                chunk.free_blocks[i].size += size;
                if i + 1 < chunk.free_blocks.len() {
                    let next_size = chunk.free_blocks[i + 1].size;
                    if chunk.free_blocks[i].offset + chunk.free_blocks[i].size == chunk.free_blocks[i + 1].offset {
                        chunk.free_blocks[i].size += next_size;
                        chunk.free_blocks.remove(i + 1);
                    }
                }
                return;
            }
            if addr.offset + size == block.offset {
                chunk.free_blocks[i].offset = addr.offset;
                chunk.free_blocks[i].size += size;
                if i > 0 {
                    let prev_end = chunk.free_blocks[i - 1].offset + chunk.free_blocks[i - 1].size;
                    let cur_offset = chunk.free_blocks[i].offset;
                    let cur_size = chunk.free_blocks[i].size;
                    if prev_end == cur_offset {
                        chunk.free_blocks[i - 1].size += cur_size;
                        chunk.free_blocks.remove(i);
                    }
                }
                return;
            }
        }
        chunk.insert_block(addr.offset, size);
    }

    pub fn free(_alloc: Self) {
        // Drop handles it.
    }
}

// ---------------------------------------------------------------------------
// virtual buffer: contiguous address range split into backend buffers
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct VBuffer {
    chunks: [Option<Rc<RefCell<BackendBuffer>>>; GGML_VBUFFER_MAX_CHUNKS],
}

fn ggml_vbuffer_chunk_size(buf: &Rc<RefCell<VBuffer>>, chunk: usize) -> usize {
    match &buf.borrow().chunks[chunk] {
        Some(b) => b.borrow().get_size(),
        None => 0,
    }
}

fn ggml_vbuffer_size(buf: &Rc<RefCell<VBuffer>>) -> usize {
    let mut size = 0;
    for chunk in buf.borrow().chunks.iter() {
        if let Some(b) = chunk {
            size += b.borrow().get_size();
        }
    }
    size
}

fn ggml_vbuffer_alloc(
    buft: &BackendBufferType,
    talloc: &DynTallocr,
    usage: BufferUsage,
) -> Option<Rc<RefCell<VBuffer>>> {
    let mut buf = VBuffer {
        chunks: std::array::from_fn(|_| None),
    };
    for n in 0..talloc.n_chunks() {
        let chunk_size = talloc.max_size(n);
        let chunk = match buft.alloc_buffer(chunk_size, usage) {
            Some(b) => b,
            None => return None,
        };
        chunk.borrow_mut().set_usage(usage);
        buf.chunks[n] = Some(chunk);
    }
    Some(Rc::new(RefCell::new(buf)))
}

fn ggml_vbuffer_tensor_alloc(buf: &Rc<RefCell<VBuffer>>, tensor: &TensorRef, addr: BufferAddress) {
    let chunk = Rc::clone(buf.borrow().chunks[addr.chunk as usize].as_ref().expect("missing chunk"));
    BackendBuffer::tensor_alloc(&chunk, tensor, addr.offset);
}

fn ggml_vbuffer_reset(buf: &Rc<RefCell<VBuffer>>) {
    for chunk in buf.borrow().chunks.iter() {
        if let Some(b) = chunk {
            b.borrow_mut().reset();
        }
    }
}

// ---------------------------------------------------------------------------
// graph allocator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct HashNode {
    n_children: i32,
    n_views: i32,
    buffer_id: isize,
    addr: BufferAddress,
    allocated: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct TensorAlloc {
    pub buffer_id: isize,
    pub addr: BufferAddress,
    /// 0 means pre-allocated, unused or a view.
    pub size_max: usize,
}

impl TensorAlloc {
    pub fn invalid() -> Self {
        Self {
            buffer_id: -1,
            addr: GGML_BUFFER_ADDRESS_INVALID,
            size_max: 0,
        }
    }
}

/// C calloc initialises these to zero.
impl Default for TensorAlloc {
    fn default() -> Self {
        Self {
            buffer_id: 0,
            addr: BufferAddress::default(),
            size_max: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LeafAlloc {
    pub leaf: TensorAlloc,
}

#[derive(Debug, Clone, Copy)]
pub struct NodeAlloc {
    pub dst: TensorAlloc,
    pub src: [TensorAlloc; GGML_MAX_SRC],
}

impl Default for NodeAlloc {
    fn default() -> Self {
        Self {
            dst: TensorAlloc::invalid(),
            src: [TensorAlloc::invalid(); GGML_MAX_SRC],
        }
    }
}

/// `struct ggml_gallocr`.
pub struct Gallocr {
    pub bufts: Vec<BackendBufferType>,
    pub buffers: Vec<Option<Rc<RefCell<VBuffer>>>>,
    pub buf_tallocs: Vec<Rc<RefCell<DynTallocr>>>,
    hash_values: HashMap<usize, HashNode>,
    pub node_allocs: Vec<NodeAlloc>,
    pub n_nodes: usize,
    pub leaf_allocs: Vec<LeafAlloc>,
    pub n_leafs: usize,
}

impl Gallocr {
    pub fn n_buffers(&self) -> usize {
        self.bufts.len()
    }

    fn tensor_key(t: &TensorRef) -> usize {
        Rc::as_ptr(t) as usize
    }

    fn hash_get(&mut self, t: &TensorRef) -> HashNode {
        *self.hash_values.entry(Self::tensor_key(t)).or_insert_with(HashNode::default)
    }

    fn hash_set(&mut self, t: &TensorRef, node: HashNode) {
        self.hash_values.insert(Self::tensor_key(t), node);
    }

    fn hash_reset(&mut self) {
        self.hash_values.clear();
    }

    fn is_own(&mut self, t: &TensorRef) -> bool {
        self.hash_get(t).allocated
    }

    fn is_allocated(&mut self, t: &TensorRef) -> bool {
        let external = {
            let b = t.borrow();
            b.data.is_some() || b.buffer.is_some()
        };
        external || self.is_own(t)
    }

    /// Free the unused tail when a view/in-place output is smaller than its parent.
    fn free_extra_space(&mut self, node: &TensorRef, parent: &TensorRef) {
        let hn = self.hash_get(node);
        let p_hn = self.hash_get(parent);
        let p_bid = p_hn.buffer_id as usize;
        let n_bid = hn.buffer_id as usize;
        let mut parent_size = self.bufts[p_bid].get_alloc_size(&parent.borrow());
        let mut node_size = self.bufts[n_bid].get_alloc_size(&node.borrow());
        assert!(parent_size >= node_size);

        let alignment = self.buf_tallocs[p_bid].borrow().alignment;
        parent_size = aligned_offset(0, parent_size, alignment);
        node_size = aligned_offset(0, node_size, alignment);

        if parent_size > node_size {
            let mut p_addr = p_hn.addr;
            p_addr.offset += node_size;
            let extra_size = parent_size - node_size;
            self.buf_tallocs[p_bid].borrow_mut().free_bytes(p_addr, extra_size);
        }
    }

    fn allocate_node(&mut self, node: &TensorRef, buffer_id: usize) -> Result<()> {
        assert!(buffer_id < self.n_buffers());
        let mut hn = self.hash_get(node);
        let allocated = self.is_allocated(node);
        let is_view = ggml_impl_is_view(&node.borrow());

        if !allocated && !is_view {
            hn.allocated = true;
            assert!(hn.addr.offset == 0);

            if ggml_op_can_inplace(node.borrow().op) {
                for i in 0..GGML_MAX_SRC {
                    let parent = { node.borrow().src[i].clone() };
                    let Some(parent) = parent else { continue };
                    if !self.is_own(&parent) {
                        continue;
                    }

                    let is_output = {
                        let b = parent.borrow();
                        b.flags & GGML_TENSOR_FLAG_OUTPUT != 0
                            || b.view_src.as_ref().map_or(false, |v| {
                                v.borrow().flags & GGML_TENSOR_FLAG_OUTPUT != 0
                            })
                    };
                    if is_output {
                        continue;
                    }
                    if !ggml_are_same_layout(&node.borrow(), &parent.borrow()) {
                        continue;
                    }

                    let p_hn = self.hash_get(&parent);
                    if p_hn.n_children == 1 && p_hn.n_views == 0 {
                        let view_src = { parent.borrow().view_src.clone() };
                        if let Some(view_src) = view_src {
                            let v_hn = self.hash_get(&view_src);
                            let same = same_data(
                                &view_src.borrow().data,
                                &parent.borrow().data,
                            );
                            if v_hn.n_views == 1 && v_hn.n_children == 0 && same {
                                hn.buffer_id = p_hn.buffer_id;
                                hn.addr = p_hn.addr;
                                let mut p = p_hn;
                                p.allocated = false;
                                self.hash_set(&parent, p);
                                let mut v = v_hn;
                                v.allocated = false;
                                self.hash_set(&view_src, v);
                                self.hash_set(node, hn);
                                self.free_extra_space(node, &view_src);
                                return Ok(());
                            }
                        } else {
                            hn.buffer_id = p_hn.buffer_id;
                            hn.addr = p_hn.addr;
                            let mut p = p_hn;
                            p.allocated = false;
                            self.hash_set(&parent, p);
                            self.hash_set(node, hn);
                            self.free_extra_space(node, &parent);
                            return Ok(());
                        }
                    }
                }
            }

            let size = self.bufts[buffer_id].get_alloc_size(&node.borrow());
            hn.buffer_id = buffer_id as isize;
            hn.addr = self.buf_tallocs[buffer_id].borrow_mut().alloc(size, node)?;
        }

        self.hash_set(node, hn);
        Ok(())
    }

    fn free_node(&mut self, node: &TensorRef) {
        if node.borrow().flags & GGML_TENSOR_FLAG_OUTPUT != 0 {
            return;
        }
        let hn = self.hash_get(node);
        let buffer_id = hn.buffer_id as usize;
        let size = self.bufts[buffer_id].get_alloc_size(&node.borrow());
        self.buf_tallocs[buffer_id].borrow_mut().free_bytes(hn.addr, size);
        let mut hn = hn;
        hn.allocated = false;
        self.hash_set(node, hn);
    }

    fn alloc_graph_impl(
        &mut self,
        graph: &CGraph,
        node_buffer_ids: Option<&[usize]>,
        leaf_buffer_ids: Option<&[usize]>,
    ) -> Result<()> {
        self.hash_reset();

        // Allocate leafs first: the application may use them outside the graph.
        for (i, leaf) in graph.leafs.iter().enumerate() {
            let id = get_node_buffer_id(leaf_buffer_ids, i);
            self.allocate_node(leaf, id)?;
        }

        // Count children and views; allocate explicit inputs early.
        for (i, node) in graph.nodes.iter().enumerate() {
            if ggml_impl_is_view(&node.borrow()) && node.borrow().op != AllocOp::None {
                let view_src = { node.borrow().view_src.clone() };
                if let Some(vs) = view_src {
                    let mut v = self.hash_get(&vs);
                    v.n_views += 1;
                    self.hash_set(&vs, v);
                }
            }
            if node.borrow().flags & GGML_TENSOR_FLAG_INPUT != 0 {
                let id = get_node_buffer_id(node_buffer_ids, i);
                self.allocate_node(node, id)?;
            }
            for j in 0..GGML_MAX_SRC {
                let src = { node.borrow().src[j].clone() };
                let Some(src) = src else { continue };
                let mut s = self.hash_get(&src);
                s.n_children += 1;
                self.hash_set(&src, s);
                if src.borrow().flags & GGML_TENSOR_FLAG_INPUT != 0 {
                    let id = get_node_buffer_id(node_buffer_ids, i);
                    self.allocate_node(&src, id)?;
                }
            }
        }

        for (i, node) in graph.nodes.iter().enumerate() {
            let buffer_id = get_node_buffer_id(node_buffer_ids, i);

            for j in 0..GGML_MAX_SRC {
                let parent = { node.borrow().src[j].clone() };
                if let Some(p) = parent {
                    self.allocate_node(&p, buffer_id)?;
                }
            }
            self.allocate_node(node, buffer_id)?;

            for j in 0..GGML_MAX_SRC {
                let parent = { node.borrow().src[j].clone() };
                let Some(parent) = parent else { continue };
                let mut p_hn = self.hash_get(&parent);
                p_hn.n_children -= 1;
                let view_src = { parent.borrow().view_src.clone() };
                if p_hn.n_children == 0 && p_hn.n_views == 0 {
                    if let Some(vs) = view_src {
                        let mut v_hn = self.hash_get(&vs);
                        v_hn.n_views -= 1;
                        if v_hn.n_views == 0 && v_hn.n_children == 0 && v_hn.allocated {
                            self.hash_set(&vs, v_hn);
                            self.free_node(&vs);
                        } else {
                            self.hash_set(&vs, v_hn);
                        }
                    } else if p_hn.allocated {
                        self.hash_set(&parent, p_hn);
                        self.free_node(&parent);
                    } else {
                        self.hash_set(&parent, p_hn);
                    }
                } else {
                    self.hash_set(&parent, p_hn);
                }
            }
        }
        Ok(())
    }

    fn reserve_n_impl(
        &mut self,
        graph: &CGraph,
        node_buffer_ids: Option<&[usize]>,
        leaf_buffer_ids: Option<&[usize]>,
        no_alloc: bool,
    ) -> Result<bool> {
        let mut min_hash_size = graph.n_nodes() + graph.n_leafs();
        min_hash_size += min_hash_size / 4; // 25% margin
        if self.hash_values.capacity() < min_hash_size {
            self.hash_values = HashMap::with_capacity(min_hash_size);
        }

        for talloc in &self.buf_tallocs {
            talloc.borrow_mut().reset();
        }

        self.alloc_graph_impl(graph, node_buffer_ids, leaf_buffer_ids)?;

        if self.node_allocs.len() < graph.n_nodes() {
            self.node_allocs.resize(graph.n_nodes(), NodeAlloc::default());
        }
        self.n_nodes = graph.n_nodes();
        for (i, node) in graph.nodes.iter().enumerate() {
            let mut dst = TensorAlloc::invalid();
            if !{ let b = node.borrow(); b.view_src.is_some() || b.data.is_some() } {
                let hn = self.hash_get(node);
                let size = self.bufts[hn.buffer_id as usize].get_alloc_size(&node.borrow());
                dst = TensorAlloc {
                    buffer_id: hn.buffer_id,
                    addr: hn.addr,
                    size_max: size,
                };
            }
            let mut src = [TensorAlloc::invalid(); GGML_MAX_SRC];
            for j in 0..GGML_MAX_SRC {
                let s = { node.borrow().src[j].clone() };
                let Some(s) = s else { continue };
                if { let b = s.borrow(); b.view_src.is_some() || b.data.is_some() } {
                    continue;
                }
                let hn = self.hash_get(&s);
                let size = self.bufts[hn.buffer_id as usize].get_alloc_size(&s.borrow());
                src[j] = TensorAlloc {
                    buffer_id: hn.buffer_id,
                    addr: hn.addr,
                    size_max: size,
                };
            }
            self.node_allocs[i] = NodeAlloc { dst, src };
        }

        if self.leaf_allocs.len() < graph.n_leafs() {
            self.leaf_allocs.resize(graph.n_leafs(), LeafAlloc::default());
        }
        self.n_leafs = graph.n_leafs();
        for (i, leaf) in graph.leafs.iter().enumerate() {
            let mut alloc = TensorAlloc::invalid();
            if !{ let b = leaf.borrow(); b.view_src.is_some() || b.data.is_some() } {
                let hn = self.hash_get(leaf);
                let size = self.bufts[hn.buffer_id as usize].get_alloc_size(&leaf.borrow());
                alloc = TensorAlloc {
                    buffer_id: hn.buffer_id,
                    addr: hn.addr,
                    size_max: size,
                };
            }
            self.leaf_allocs[i] = LeafAlloc { leaf: alloc };
        }

        for i in 0..self.n_buffers() {
            for j in 0..i {
                if Rc::ptr_eq(&self.buf_tallocs[i], &self.buf_tallocs[j]) {
                    self.buffers[i] = self.buffers[j].clone();
                    break;
                }
            }

            let mut realloc = self.buffers[i].is_none();
            let n_chunks = self.buf_tallocs[i].borrow().n_chunks();
            for c in 0..n_chunks {
                let cur_chunk_size = self.buffers[i]
                    .as_ref()
                    .map_or(0, |b| ggml_vbuffer_chunk_size(b, c));
                let new_chunk_size = self.buf_tallocs[i].borrow().max_size(c);
                if new_chunk_size > cur_chunk_size {
                    realloc = true;
                }
            }

            if realloc {
                if no_alloc {
                    self.buffers[i] = None;
                } else {
                    let buft = self.bufts[i].clone();
                    let talloc = Rc::clone(&self.buf_tallocs[i]);
                    let talloc_ref = talloc.borrow();
                    match ggml_vbuffer_alloc(&buft, &talloc_ref, BufferUsage::Compute) {
                        Some(b) => self.buffers[i] = Some(b),
                        None => return Ok(false),
                    }
                }
            }
        }

        Ok(true)
    }

    fn init_tensor(&self, tensor: &TensorRef, talloc: &TensorAlloc) {
        let size = self.bufts[talloc.buffer_id as usize].get_alloc_size(&tensor.borrow());
        assert!(
            tensor.borrow().data.is_some()
                || tensor.borrow().view_src.is_some()
                || size <= talloc.size_max
        );

        let view_src = { tensor.borrow().view_src.clone() };
        if let Some(vs) = view_src {
            if tensor.borrow().buffer.is_none() {
                assert!(talloc.addr.offset == usize::MAX);
                if vs.borrow().buffer.is_none() {
                    // Allocated without ggml-backend.
                    return;
                }
                // `ggml_backend_view_init`: keep the view's offset within the
                // source buffer when it was assigned outside the allocator.
                let (buffer, offset) = match &tensor.borrow().data {
                    Some(TensorData::Buffer { buffer, offset }) => (*buffer, *offset),
                    _ => {
                        let b = vs.borrow();
                        let buffer = b.buffer.expect("view source has no buffer");
                        let offset = match &b.data {
                            Some(TensorData::Buffer { offset, .. }) => *offset,
                            _ => 0,
                        };
                        (buffer, offset)
                    }
                };
                let mut t = tensor.borrow_mut();
                t.buffer = Some(buffer);
                t.data = Some(TensorData::Buffer { buffer, offset });
            }
        } else if tensor.borrow().data.is_none() {
            assert!(talloc.addr.offset != usize::MAX);
            assert!(size <= talloc.size_max);
            let buf = self.buffers[talloc.buffer_id as usize]
                .as_ref()
                .expect("graph buffer not allocated");
            ggml_vbuffer_tensor_alloc(buf, tensor, talloc.addr);
        } else if tensor.borrow().buffer.is_none() {
            // Allocated without ggml-backend.
        }
    }

    fn node_needs_realloc(&self, node: &TensorRef, talloc: &TensorAlloc) -> bool {
        let mut node_size = 0;
        let (has_data, is_view) = {
            let b = node.borrow();
            (b.data.is_some(), b.view_src.is_some())
        };
        if !has_data && !is_view {
            if talloc.buffer_id < 0 {
                return false;
            }
            node_size = self.bufts[talloc.buffer_id as usize].get_alloc_size(&node.borrow());
        }
        talloc.size_max >= node_size
    }

    pub fn needs_realloc(&self, graph: &CGraph) -> bool {
        if self.n_nodes != graph.n_nodes() {
            return true;
        }
        if self.n_leafs != graph.n_leafs() {
            return true;
        }
        for (i, node) in graph.nodes.iter().enumerate() {
            if !self.node_needs_realloc(node, &self.node_allocs[i].dst) {
                return true;
            }
            for j in 0..GGML_MAX_SRC {
                let src = { node.borrow().src[j].clone() };
                if let Some(s) = src {
                    if !self.node_needs_realloc(&s, &self.node_allocs[i].src[j]) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

fn get_node_buffer_id(ids: Option<&[usize]>, i: usize) -> usize {
    ids.map_or(0, |v| v[i])
}

pub fn ggml_gallocr_new_n(bufts: &[BackendBufferType]) -> Gallocr {
    let n = bufts.len();
    let mut g = Gallocr {
        bufts: Vec::with_capacity(n),
        buffers: std::iter::repeat_with(|| None).take(n).collect(),
        buf_tallocs: Vec::with_capacity(n),
        hash_values: HashMap::new(),
        node_allocs: Vec::new(),
        n_nodes: 0,
        leaf_allocs: Vec::new(),
        n_leafs: 0,
    };

    for i in 0..n {
        g.bufts.push(bufts[i].clone());
        let mut shared = None;
        for j in 0..i {
            if bufts[i].same(&bufts[j]) {
                shared = Some(Rc::clone(&g.buf_tallocs[j]));
                break;
            }
        }
        let talloc = match shared {
            Some(t) => t,
            None => Rc::new(RefCell::new(DynTallocr::new(
                bufts[i].get_alignment(),
                bufts[i].get_max_size(),
            ))),
        };
        g.buf_tallocs.push(talloc);
    }

    g
}

pub fn ggml_gallocr_new(buft: &BackendBufferType) -> Gallocr {
    ggml_gallocr_new_n(std::slice::from_ref(buft))
}

pub fn ggml_gallocr_free(_galloc: Gallocr) {
    // Drop handles it.
}

impl Gallocr {
    pub fn reserve_n(
        &mut self,
        graph: &CGraph,
        node_buffer_ids: Option<&[usize]>,
        leaf_buffer_ids: Option<&[usize]>,
    ) -> Result<bool> {
        self.reserve_n_impl(graph, node_buffer_ids, leaf_buffer_ids, false)
    }

    pub fn reserve(&mut self, graph: &CGraph) -> Result<bool> {
        self.reserve_n(graph, None, None)
    }

    pub fn reserve_n_size(
        &mut self,
        graph: &CGraph,
        node_buffer_ids: Option<&[usize]>,
        leaf_buffer_ids: Option<&[usize]>,
        sizes: &mut [usize],
    ) -> Result<()> {
        assert!(self.reserve_n_impl(graph, node_buffer_ids, leaf_buffer_ids, true)?);
        for i in 0..self.n_buffers() {
            sizes[i] = 0;
            let n_chunks = self.buf_tallocs[i].borrow().n_chunks();
            for c in 0..n_chunks {
                sizes[i] += self.buf_tallocs[i].borrow().max_size(c);
            }
        }
        Ok(())
    }

    pub fn alloc_graph(&mut self, graph: &CGraph) -> Result<bool> {
        if self.needs_realloc(graph) {
            if self.n_buffers() == 1 {
                if !self.reserve(graph)? {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }

        for buf in self.buffers.iter().flatten() {
            ggml_vbuffer_reset(buf);
        }

        for (i, leaf) in graph.leafs.iter().enumerate() {
            let ta = self.leaf_allocs[i].leaf;
            self.init_tensor(leaf, &ta);
        }
        for (i, node) in graph.nodes.iter().enumerate() {
            for j in 0..GGML_MAX_SRC {
                let src = { node.borrow().src[j].clone() };
                if let Some(s) = src {
                    let ta = self.node_allocs[i].src[j];
                    self.init_tensor(&s, &ta);
                }
            }
            let ta = self.node_allocs[i].dst;
            self.init_tensor(node, &ta);
        }

        Ok(true)
    }

    pub fn get_buffer_size(&self, buffer_id: usize) -> usize {
        assert!(buffer_id < self.n_buffers());
        let buf = match &self.buffers[buffer_id] {
            Some(b) => b,
            None => return 0,
        };
        for previous in &self.buffers[..buffer_id] {
            if previous.as_ref().map_or(false, |b| Rc::ptr_eq(b, buf)) {
                return 0;
            }
        }
        ggml_vbuffer_size(buf)
    }
}

// ---------------------------------------------------------------------------
// utils: allocate every tensor in a context from one buffer type
// ---------------------------------------------------------------------------

/// Stand-in for `ggml_context` in the no-alloc utils.
#[derive(Debug, Default)]
pub struct AllocContext {
    pub tensors: Vec<TensorRef>,
    pub no_alloc: bool,
}

#[derive(Debug, Clone)]
pub struct AllocBackend {
    pub default_buffer_type: BackendBufferType,
}

fn collect_tensors(ctx: &AllocContext) -> Vec<TensorRef> {
    ctx.tensors.clone()
}

pub fn ggml_backend_alloc_ctx_tensors_from_buft_size(ctx: &AllocContext, buft: &BackendBufferType) -> usize {
    assert!(ctx.no_alloc);
    let tensors = collect_tensors(ctx);
    buft.get_alloc_size_n(&tensors)
}

pub fn ggml_backend_alloc_ctx_tensors_from_buft(
    ctx: &AllocContext,
    buft: &BackendBufferType,
) -> Option<Rc<RefCell<BackendBuffer>>> {
    assert!(ctx.no_alloc);
    let tensors = collect_tensors(ctx);
    if tensors.is_empty() {
        return None;
    }
    let size = buft.get_alloc_size_n(&tensors);
    let buffer = buft.alloc_buffer(size + buft.get_alignment(), BufferUsage::Compute)?;
    let mut talloc = ggml_tallocr_new(Rc::clone(&buffer));
    for tensor in &tensors {
        ggml_tallocr_alloc(&mut talloc, tensor).ok()?;
    }
    Some(buffer)
}

pub fn ggml_backend_alloc_ctx_tensors(
    ctx: &AllocContext,
    backend: &AllocBackend,
) -> Option<Rc<RefCell<BackendBuffer>>> {
    ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.default_buffer_type)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, n: i64) -> TensorRef {
        Rc::new(RefCell::new(AllocTensor::new_f32(name, [n, 1, 1, 1])))
    }

    #[test]
    fn aligned_offset_rounds_up() {
        assert_eq!(aligned_offset(0, 0, 32), 0);
        assert_eq!(aligned_offset(0, 1, 32), 32);
        assert_eq!(aligned_offset(0, 33, 32), 64);
        assert_eq!(aligned_offset(8, 0, 32), 24);
    }

    #[test]
    fn dyn_tallocr_reuses_and_merges_free_blocks() {
        let mut alloc = DynTallocr::new(32, 1024);
        let a = tensor("a", 4);
        let b = tensor("b", 4);
        let addr_a = alloc.alloc(64, &a).unwrap();
        let addr_b = alloc.alloc(64, &b).unwrap();
        assert_eq!((addr_a.chunk, addr_a.offset), (0, 0));
        assert_eq!((addr_b.chunk, addr_b.offset), (0, 64));

        alloc.free_bytes(addr_a, 64);
        let c = tensor("c", 4);
        let addr_c = alloc.alloc(32, &c).unwrap();
        assert_eq!((addr_c.chunk, addr_c.offset), (0, 0));
        assert_eq!(alloc.chunks[0].free_blocks.len(), 2);
    }

    #[test]
    fn gallocr_assigns_and_reuses_addresses() {
        let buft = BackendBufferType::new(32, 4096, "cpu");
        let mut galloc = ggml_gallocr_new(&buft);
        let x = tensor("x", 4);
        let y = tensor("y", 4);
        {
            let mut t = y.borrow_mut();
            t.op = AllocOp::Add;
            t.src[0] = Some(Rc::clone(&x));
        }
        let graph = CGraph {
            nodes: vec![Rc::clone(&y)],
            leafs: vec![Rc::clone(&x)],
        };

        assert!(galloc.reserve(&graph).unwrap());
        assert_eq!(galloc.get_buffer_size(0), 32);
        assert!(galloc.alloc_graph(&graph).unwrap());

        let x_offset = match &x.borrow().data {
            Some(TensorData::Buffer { offset, .. }) => *offset,
            _ => panic!("x not assigned"),
        };
        let y_offset = match &y.borrow().data {
            Some(TensorData::Buffer { offset, .. }) => *offset,
            _ => panic!("y not assigned"),
        };
        assert_eq!(x_offset, 0);
        assert_eq!(y_offset, x_offset);
    }

    #[test]
    fn tallocr_reports_out_of_space() {
        let buft = BackendBufferType::new(32, 64, "cpu");
        let buffer = buft.alloc_buffer(64, BufferUsage::Compute).unwrap();
        let mut talloc = ggml_tallocr_new(buffer);
        let big = tensor("big", 8);
        assert!(ggml_tallocr_alloc(&mut talloc, &big).is_ok());
        assert!(ggml_tallocr_alloc(&mut talloc, &big).is_ok());
        assert!(ggml_tallocr_alloc(&mut talloc, &big).is_err());
    }
}

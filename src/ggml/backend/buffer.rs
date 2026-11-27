//! Backend buffer types and buffers.
//!
//! Port of the buffer half of `ggml-backend.cpp` plus the multi-buffer
//! implementation. Buffer memory is a shared `Vec<u8>`, so tensor data writes
//! through a buffer are visible to every tensor that views the same storage.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

use crate::core::{Error, Result};

use super::stream::DeviceRef;
use super::tensor::{data_location, effective_buffer, nbytes_ref, pad, TensorRef};
use super::traits::{AsAny, BackendBufferInterface, BufferTypeInterface};
use super::types::{BufferUsage, Status};

/// `struct ggml_backend_buffer_type *`.
pub type BuftRef = Rc<BufferType>;

/// `struct ggml_backend_buffer_type` with the C `void * context` replaced by
/// state held in the trait object.
pub struct BufferType {
    pub iface: Rc<dyn BufferTypeInterface>,
    pub device: RefCell<Option<Weak<RefCell<super::stream::Device>>>>,
}

/// `struct ggml_backend_buffer *`.
pub type BufferRef = Rc<RefCell<BackendBuffer>>;

/// Identifies the special buffer implementations the C++ code detects by
/// function-pointer comparison (`ggml_backend_buffer_is_multi_buffer`,
/// `ggml_backend_buffer_is_meta`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferKind {
    Normal,
    Multi,
    Meta,
}

/// `struct ggml_backend_buffer`.
pub struct BackendBuffer {
    pub iface: Rc<dyn BackendBufferInterface>,
    pub buft: BuftRef,
    pub storage: Rc<RefCell<Vec<u8>>>,
    pub size: usize,
    pub usage: BufferUsage,
    pub kind: BufferKind,
}

impl std::fmt::Debug for BackendBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendBuffer")
            .field("size", &self.size)
            .field("usage", &self.usage)
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

/// Interface used for zero-sized dummy buffers and for tests.
pub struct EmptyBufferInterface;

impl AsAny for EmptyBufferInterface {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl BackendBufferInterface for EmptyBufferInterface {
    fn set_tensor(&self, _buffer: &BufferRef, _tensor: &TensorRef, _data: &[u8], _offset: usize, _size: usize) {}
    fn get_tensor(&self, _buffer: &BufferRef, _tensor: &TensorRef, _data: &mut [u8], _offset: usize, _size: usize) {}
}

/// Fill a buffer's byte storage. Used by the default `clear` hook.
pub fn clear_buffer_storage(buffer: &BufferRef, value: u8) {
    let b = buffer.borrow();
    b.storage.borrow_mut().fill(value);
}

/// `ggml_backend_buffer_init` with externally supplied byte storage.
pub fn buffer_init_with_storage(
    buft: BuftRef,
    iface: Rc<dyn BackendBufferInterface>,
    storage: Rc<RefCell<Vec<u8>>>,
    size: usize,
) -> BufferRef {
    Rc::new(RefCell::new(BackendBuffer {
        iface,
        buft,
        storage,
        size,
        usage: BufferUsage::Any,
        kind: BufferKind::Normal,
    }))
}

/// `ggml_backend_buffer_init`.
pub fn buffer_init(buft: BuftRef, iface: Rc<dyn BackendBufferInterface>, size: usize) -> BufferRef {
    buffer_init_with_storage(buft, iface, Rc::new(RefCell::new(vec![0u8; size])), size)
}

// ---------------------------------------------------------------------------
// buffer type
// ---------------------------------------------------------------------------

/// `ggml_backend_buft_name`.
pub fn buft_name(buft: &BuftRef) -> String {
    buft.iface.name(buft)
}

/// `ggml_backend_buft_alloc_buffer`.
pub fn buft_alloc_buffer(buft: &BuftRef, size: usize) -> Result<BufferRef> {
    if size == 0 {
        return Ok(buffer_init(
            Rc::clone(buft),
            Rc::new(EmptyBufferInterface),
            0,
        ));
    }
    buft.iface.alloc_buffer(buft, size)
}

/// A planned group of tensors for one allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocPlanItem {
    pub size: usize,
    pub first: usize,
    pub last: usize,
}

/// `ggml_backend_buft_alloc_buffer_n_plan`.
pub fn alloc_buffer_n_plan(buft: &BuftRef, tensors: &[TensorRef]) -> Vec<AllocPlanItem> {
    let alignment = buft_get_alignment(buft);
    let max_size = buft_get_max_size(buft);

    let mut plan = Vec::new();
    let mut cur_buf_size = 0usize;
    let mut first = 0usize;

    for (i, t) in tensors.iter().enumerate() {
        let t = t.borrow();
        let this_size = if t.data.is_none() && t.view_src.is_none() {
            pad(buft_get_alloc_size(buft, &tensors[i]), alignment)
        } else {
            0
        };

        if cur_buf_size > 0 && cur_buf_size.saturating_add(this_size) > max_size {
            plan.push(AllocPlanItem { size: cur_buf_size, first, last: i });
            cur_buf_size = this_size;
            first = i;
        } else {
            cur_buf_size += this_size;
        }
    }

    if cur_buf_size > 0 {
        plan.push(AllocPlanItem { size: cur_buf_size, first, last: tensors.len() });
    }

    plan
}

/// `ggml_backend_buft_alloc_buffer_n_default`.
pub fn default_alloc_buffer_n(buft: &BuftRef, tensors: &[TensorRef]) -> Result<BufferRef> {
    let plan = alloc_buffer_n_plan(buft, tensors);
    let mut buffers: Vec<BufferRef> = Vec::with_capacity(plan.len());

    let alignment = buft_get_alignment(buft);
    for item in &plan {
        let buffer = buft_alloc_buffer(buft, item.size)?;
        let mut head = 0usize;
        // Linear allocator equivalent of ggml_tallocr_alloc.
        for t in &tensors[item.first..item.last] {
            let (is_view, has_buffer, has_data) = {
                let tb = t.borrow();
                (tb.view_src.is_some(), tb.buffer.is_some(), tb.data.is_some())
            };
            if !has_data && !is_view {
                let size = buft_get_alloc_size(buft, t);
                head = pad(head, alignment);
                tensor_alloc(&buffer, t, head)?;
                head += size;
            } else if is_view && !has_buffer {
                view_init(t)?;
            } else if has_data && is_view && !has_buffer {
                view_init(t)?;
            }
        }
        buffers.push(buffer);
    }

    match buffers.len() {
        0 => Err(Error::new("ggml_backend_buft_alloc_buffer_n: no buffers allocated")),
        1 => Ok(buffers.remove(0)),
        _ => Ok(multi_buffer_alloc_buffer(&buffers)),
    }
}

/// `ggml_backend_buft_get_alloc_size_n_default`.
pub fn default_alloc_size_n(buft: &BuftRef, tensors: &[TensorRef]) -> usize {
    alloc_buffer_n_plan(buft, tensors).iter().map(|item| item.size).sum()
}

/// `ggml_backend_buft_alloc_buffer_n`.
pub fn buft_alloc_buffer_n(buft: &BuftRef, tensors: &[TensorRef]) -> Result<BufferRef> {
    buft.iface.alloc_buffer_n(buft, tensors)
}

/// `ggml_backend_buft_get_alignment`.
pub fn buft_get_alignment(buft: &BuftRef) -> usize {
    buft.iface.alignment(buft)
}

/// `ggml_backend_buft_get_max_size`.
pub fn buft_get_max_size(buft: &BuftRef) -> usize {
    buft.iface.max_size(buft)
}

/// `ggml_backend_buft_get_alloc_size`.
pub fn buft_get_alloc_size(buft: &BuftRef, tensor: &TensorRef) -> usize {
    let size = buft.iface.alloc_size(buft, tensor);
    let nbytes = nbytes_ref(tensor);
    if size < nbytes {
        return nbytes;
    }
    size
}

/// `ggml_backend_buft_get_alloc_size_n`.
pub fn buft_get_alloc_size_n(buft: &BuftRef, tensors: &[TensorRef]) -> usize {
    buft.iface.alloc_size_n(buft, tensors)
}

/// `ggml_backend_buft_is_host`.
pub fn buft_is_host(buft: &BuftRef) -> bool {
    buft.iface.is_host(buft)
}

/// `ggml_backend_buft_get_device`.
pub fn buft_get_device(buft: &BuftRef) -> Option<DeviceRef> {
    buft.device.borrow().as_ref().and_then(|w| w.upgrade())
}

// ---------------------------------------------------------------------------
// buffer
// ---------------------------------------------------------------------------

/// `ggml_backend_buffer_name`.
pub fn buffer_name(buffer: &BufferRef) -> String {
    buft_name(&buffer.borrow().buft)
}

/// `ggml_backend_buffer_free`: runs the backend hook; Rust drops the memory.
pub fn buffer_free(buffer: &BufferRef) {
    let iface = Rc::clone(&buffer.borrow().iface);
    iface.free_buffer(buffer);
}

/// `ggml_backend_buffer_get_size`.
pub fn buffer_get_size(buffer: &BufferRef) -> usize {
    buffer.borrow().size
}

/// `ggml_backend_buffer_get_base`; `None` mirrors a NULL base pointer.
pub fn buffer_get_base(buffer: &BufferRef) -> Option<Rc<RefCell<Vec<u8>>>> {
    let b = buffer.borrow();
    if b.kind != BufferKind::Meta && b.size == 0 {
        return None;
    }
    let iface = Rc::clone(&b.iface);
    iface.get_base(buffer)
}

/// `ggml_backend_buffer_init_tensor`.
pub fn buffer_init_tensor(buffer: &BufferRef, tensor: &TensorRef) -> Result<Status> {
    let iface = Rc::clone(&buffer.borrow().iface);
    iface.init_tensor(buffer, tensor)
}

/// `ggml_backend_buffer_clear`.
pub fn buffer_clear(buffer: &BufferRef, value: u8) {
    if buffer.borrow().size == 0 {
        return;
    }
    let iface = Rc::clone(&buffer.borrow().iface);
    iface.clear(buffer, value);
}

/// `ggml_backend_buffer_get_alignment`.
pub fn buffer_get_alignment(buffer: &BufferRef) -> usize {
    buft_get_alignment(&buffer.borrow().buft)
}

/// `ggml_backend_buffer_get_max_size`.
pub fn buffer_get_max_size(buffer: &BufferRef) -> usize {
    buft_get_max_size(&buffer.borrow().buft)
}

/// `ggml_backend_buffer_get_alloc_size`.
pub fn buffer_get_alloc_size(buffer: &BufferRef, tensor: &TensorRef) -> usize {
    buft_get_alloc_size(&buffer.borrow().buft, tensor)
}

/// `ggml_backend_buffer_is_host`.
pub fn buffer_is_host(buffer: &BufferRef) -> bool {
    buft_is_host(&buffer.borrow().buft)
}

/// `ggml_backend_buffer_set_usage`.
pub fn buffer_set_usage(buffer: &BufferRef, usage: BufferUsage) {
    buffer.borrow_mut().usage = usage;
    let kind = buffer.borrow().kind;
    match kind {
        BufferKind::Multi => multi_buffer_set_usage(buffer, usage),
        BufferKind::Meta => super::meta::meta_buffer_set_usage(buffer, usage),
        BufferKind::Normal => {}
    }
}

/// `ggml_backend_buffer_get_usage`.
pub fn buffer_get_usage(buffer: &BufferRef) -> BufferUsage {
    buffer.borrow().usage
}

/// `ggml_backend_buffer_get_type`.
pub fn buffer_get_type(buffer: &BufferRef) -> BuftRef {
    Rc::clone(&buffer.borrow().buft)
}

/// `ggml_backend_buffer_reset`.
pub fn buffer_reset(buffer: &BufferRef) {
    let iface = Rc::clone(&buffer.borrow().iface);
    iface.reset(buffer);
}

/// `ggml_backend_buffer_copy_tensor`.
pub fn buffer_copy_tensor(src: &TensorRef, dst: &TensorRef) -> bool {
    let dst_buf = match effective_buffer(dst) {
        Some(b) => b,
        None => return false,
    };
    let iface = Rc::clone(&dst_buf.borrow().iface);
    iface.cpy_tensor(&dst_buf, src, dst)
}

// ---------------------------------------------------------------------------
// tensor data access
// ---------------------------------------------------------------------------

fn check_range(offset: usize, size: usize, nbytes: usize, op: &str) -> Result<()> {
    if offset.checked_add(size).map_or(true, |end| end > nbytes) {
        Err(Error::new(format!("tensor {op} out of bounds")))
    } else {
        Ok(())
    }
}

fn with_tensor_data<R>(
    tensor: &TensorRef,
    f: impl FnOnce(&Rc<RefCell<Vec<u8>>>, usize) -> R,
) -> Result<R> {
    let data = data_location(tensor).ok_or_else(|| Error::new("tensor not allocated"))?;
    Ok(f(&data.storage, data.offset))
}

/// `ggml_backend_tensor_set`.
pub fn tensor_set(tensor: &TensorRef, data: &[u8], offset: usize, size: usize) -> Result<()> {
    let buf = effective_buffer(tensor).ok_or_else(|| Error::new("tensor buffer not set"))?;
    if size == 0 {
        return Ok(());
    }
    check_range(offset, size, nbytes_ref(tensor), "write")?;
    let iface = Rc::clone(&buf.borrow().iface);
    iface.set_tensor(&buf, tensor, data, offset, size);
    Ok(())
}

/// `ggml_backend_tensor_get`.
pub fn tensor_get(tensor: &TensorRef, data: &mut [u8], offset: usize, size: usize) -> Result<()> {
    let buf = effective_buffer(tensor).ok_or_else(|| Error::new("tensor buffer not set"))?;
    if size == 0 {
        return Ok(());
    }
    check_range(offset, size, nbytes_ref(tensor), "read")?;
    let iface = Rc::clone(&buf.borrow().iface);
    iface.get_tensor(&buf, tensor, data, offset, size);
    Ok(())
}

/// `ggml_backend_tensor_set_2d`.
pub fn tensor_set_2d(
    tensor: &TensorRef,
    data: &[u8],
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) -> Result<()> {
    let buf = effective_buffer(tensor).ok_or_else(|| Error::new("tensor buffer not set"))?;
    let iface = Rc::clone(&buf.borrow().iface);
    if n_copies <= 1 || !iface.set_tensor_2d_handled(&buf) {
        for i in 0..n_copies {
            let d = &data[i * stride_data..];
            tensor_set(tensor, d, offset + i * stride_tensor, size)?;
        }
        return Ok(());
    }
    if size == 0 {
        return Ok(());
    }
    check_range(
        offset + (n_copies - 1) * stride_tensor,
        size,
        nbytes_ref(tensor),
        "write",
    )?;
    iface.set_tensor_2d(&buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    Ok(())
}

/// `ggml_backend_tensor_get_2d`.
pub fn tensor_get_2d(
    tensor: &TensorRef,
    data: &mut [u8],
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) -> Result<()> {
    let buf = effective_buffer(tensor).ok_or_else(|| Error::new("tensor buffer not set"))?;
    let iface = Rc::clone(&buf.borrow().iface);
    if n_copies <= 1 || !iface.get_tensor_2d_handled(&buf) {
        for i in 0..n_copies {
            let d = &mut data[i * stride_data..];
            tensor_get(tensor, d, offset + i * stride_tensor, size)?;
        }
        return Ok(());
    }
    if size == 0 {
        return Ok(());
    }
    check_range(
        offset + (n_copies - 1) * stride_tensor,
        size,
        nbytes_ref(tensor),
        "read",
    )?;
    iface.get_tensor_2d(&buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    Ok(())
}

/// Groups tensor indices by their buffer type, preserving first-seen order.
pub fn tensor_buft_groups<F>(n_tensors: usize, get_buffer: F) -> Vec<Vec<usize>>
where
    F: Fn(usize) -> BufferRef,
{
    let mut buft_idx: HashMap<usize, usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..n_tensors {
        let buf = get_buffer(i);
        let buft_ptr = Rc::as_ptr(&buf.borrow().buft) as usize;
        let idx = match buft_idx.get(&buft_ptr) {
            Some(&idx) => idx,
            None => {
                let idx = groups.len();
                buft_idx.insert(buft_ptr, idx);
                groups.push(Vec::new());
                idx
            }
        };
        groups[idx].push(i);
    }
    groups
}

/// Detect the 2D batched-copy pattern of `ggml_backend_tensor_batch_is_2d`.
pub fn tensor_batch_is_2d(
    tensors: &[TensorRef],
    offsets: &[usize],
    sizes: &[usize],
    data_offsets: &[usize],
) -> Option<(usize, usize, usize, usize)> {
    if tensors.len() <= 1 {
        return None;
    }
    let tensor = &tensors[0];
    let size0 = sizes[0];
    if size0 == 0 || offsets[1] < offsets[0] || data_offsets[1] < data_offsets[0] {
        return None;
    }
    let stride_tensor0 = offsets[1] - offsets[0];
    let stride_data0 = data_offsets[1] - data_offsets[0];
    for i in 1..tensors.len() {
        if !Rc::ptr_eq(tensor, &tensors[i]) || sizes[i] != size0 {
            return None;
        }
        if offsets[i] != offsets[0] + i * stride_tensor0
            || data_offsets[i] != data_offsets[0] + i * stride_data0
        {
            return None;
        }
    }
    Some((offsets[0], size0, stride_tensor0, stride_data0))
}

/// `ggml_backend_tensor_set_n`.
pub fn tensor_set_n(
    tensors: &[TensorRef],
    data: &[u8],
    offsets: &[usize],
    sizes: &[usize],
    data_offsets: &[usize],
) -> Result<()> {
    if tensors.is_empty() {
        return Ok(());
    }
    let data_ptrs: Vec<&[u8]> = data_offsets
        .iter()
        .map(|&o| data.get(o..).unwrap_or(&[]))
        .collect();

    let buf = effective_buffer(&tensors[0]).ok_or_else(|| Error::new("tensor buffer not set"))?;
    let buft_ptr = Rc::as_ptr(&buf.borrow().buft);
    let mut same_buft = true;
    for t in tensors {
        match effective_buffer(t) {
            Some(b) if Rc::as_ptr(&b.borrow().buft) == buft_ptr => {}
            _ => same_buft = false,
        }
    }

    if same_buft {
        let iface = Rc::clone(&buf.borrow().iface);
        if iface.set_tensor_n_handled(&buf) {
            iface.set_tensor_n(&buf, tensors, data, offsets, sizes, data_offsets, tensors.len());
            return Ok(());
        }
    } else {
        let groups = tensor_buft_groups(tensors.len(), |i| effective_buffer(&tensors[i]).unwrap());
        for g in groups {
            let mut gt = Vec::new();
            let mut go = Vec::new();
            let mut gs = Vec::new();
            let mut gd = Vec::new();
            for i in g {
                gt.push(Rc::clone(&tensors[i]));
                go.push(offsets[i]);
                gs.push(sizes[i]);
                gd.push(data_offsets[i]);
            }
            tensor_set_n(&gt, data, &go, &gs, &gd)?;
        }
        return Ok(());
    }

    if let Some((base, size, st, sd)) = tensor_batch_is_2d(tensors, offsets, sizes, data_offsets) {
        return tensor_set_2d(&tensors[0], data_ptrs[0], base, size, tensors.len(), st, sd);
    }
    for i in 0..tensors.len() {
        tensor_set(&tensors[i], data_ptrs[i], offsets[i], sizes[i])?;
    }
    Ok(())
}

/// `ggml_backend_tensor_get_n`.
pub fn tensor_get_n(
    tensors: &[TensorRef],
    data: &mut [u8],
    offsets: &[usize],
    sizes: &[usize],
    data_offsets: &[usize],
) -> Result<()> {
    if tensors.is_empty() {
        return Ok(());
    }
    // Carve non-overlapping mutable slices in data-offset order.
    let mut data_ptrs: Vec<&mut [u8]> = Vec::with_capacity(data_offsets.len());
    let mut rest: &mut [u8] = data;
    let mut cur = 0usize;
    for (&o, &sz) in data_offsets.iter().zip(sizes.iter()) {
        let skip = o.saturating_sub(cur);
        if skip > rest.len() {
            data_ptrs.push(&mut []);
            continue;
        }
        let (_, tail) = rest.split_at_mut(skip);
        let n = sz.min(tail.len());
        let (chunk, tail2) = tail.split_at_mut(n);
        rest = tail2;
        cur = o + n;
        data_ptrs.push(chunk);
    }

    let first = effective_buffer(&tensors[0]).ok_or_else(|| Error::new("tensor buffer not set"))?;
    let first_ptr = Rc::as_ptr(&first.borrow().buft);
    let mut same_buft = true;
    for t in tensors {
        match effective_buffer(t) {
            Some(b) if Rc::as_ptr(&b.borrow().buft) == first_ptr => {}
            _ => same_buft = false,
        }
    }

    if same_buft {
        let iface = Rc::clone(&first.borrow().iface);
        if iface.get_tensor_n_handled(&first) {
            iface.get_tensor_n(&first, tensors, data, offsets, sizes, data_offsets, tensors.len());
            return Ok(());
        }
    } else {
        let groups = tensor_buft_groups(tensors.len(), |i| effective_buffer(&tensors[i]).unwrap());
        for g in groups {
            let mut gt = Vec::new();
            let mut go = Vec::new();
            let mut gs = Vec::new();
            let mut gd = Vec::new();
            for i in g {
                gt.push(Rc::clone(&tensors[i]));
                go.push(offsets[i]);
                gs.push(sizes[i]);
                gd.push(data_offsets[i]);
            }
            tensor_get_n(&gt, data, &go, &gs, &gd)?;
        }
        return Ok(());
    }

    if let Some((base, size, st, sd)) = tensor_batch_is_2d(tensors, offsets, sizes, data_offsets) {
        return tensor_get_2d(&tensors[0], data_ptrs[0], base, size, tensors.len(), st, sd);
    }
    for i in 0..tensors.len() {
        tensor_get(&tensors[i], data_ptrs[i], offsets[i], sizes[i])?;
    }
    Ok(())
}

/// `ggml_backend_tensor_memset`.
pub fn tensor_memset(tensor: &TensorRef, value: u8, offset: usize, size: usize) -> Result<()> {
    let buf = effective_buffer(tensor).ok_or_else(|| Error::new("tensor buffer not set"))?;
    if size == 0 {
        return Ok(());
    }
    check_range(offset, size, nbytes_ref(tensor), "write")?;
    let iface = Rc::clone(&buf.borrow().iface);
    if !iface.memset_handled(&buf) {
        let chunk_size = size.min(1024 * 1024);
        let chunk = vec![value; chunk_size];
        let mut chunk_offset = 0;
        while chunk_offset < size {
            let n = (size - chunk_offset).min(chunk_size);
            tensor_set(tensor, &chunk[..n], offset + chunk_offset, n)?;
            chunk_offset += n;
        }
        return Ok(());
    }
    iface.memset_tensor(&buf, tensor, value, offset, size);
    Ok(())
}

/// `ggml_backend_tensor_copy`.
pub fn tensor_copy(src: &TensorRef, dst: &TensorRef) -> Result<()> {
    if Rc::ptr_eq(src, dst) {
        return Ok(());
    }
    let src_buf = effective_buffer(src).ok_or_else(|| Error::new("src tensor buffer not set"))?;
    let dst_buf = effective_buffer(dst).ok_or_else(|| Error::new("dst tensor buffer not set"))?;
    let nbytes = nbytes_ref(src);

    if buffer_is_host(&src_buf) {
        let bytes = with_tensor_data(src, |storage, offset| {
            storage.borrow()[offset..offset + nbytes].to_vec()
        })?;
        tensor_set(dst, &bytes, 0, nbytes)?;
    } else if buffer_is_host(&dst_buf) {
        let mut bytes = vec![0u8; nbytes];
        tensor_get(src, &mut bytes, 0, nbytes)?;
        let dst_data = data_location(dst).ok_or_else(|| Error::new("dst tensor not allocated"))?;
        dst_data.storage.borrow_mut()[dst_data.offset..dst_data.offset + nbytes]
            .copy_from_slice(&bytes);
    } else if !buffer_copy_tensor(src, dst) {
        let mut bytes = vec![0u8; nbytes];
        tensor_get(src, &mut bytes, 0, nbytes)?;
        tensor_set(dst, &bytes, 0, nbytes)?;
    }
    Ok(())
}

/// `ggml_backend_tensor_copy_n`.
pub fn tensor_copy_n(srcs: &[TensorRef], dsts: &[TensorRef]) -> Result<()> {
    if srcs.is_empty() {
        return Ok(());
    }
    let first = effective_buffer(&dsts[0]).ok_or_else(|| Error::new("dst tensor buffer not set"))?;
    let first_ptr = Rc::as_ptr(&first.borrow().buft);
    let same_buft = dsts.iter().all(|d| {
        effective_buffer(d)
            .map(|b| Rc::as_ptr(&b.borrow().buft) == first_ptr)
            .unwrap_or(false)
    });

    if same_buft {
        let iface = Rc::clone(&first.borrow().iface);
        if iface.cpy_tensor_n(&first, srcs, dsts) {
            return Ok(());
        }
    } else {
        let groups = tensor_buft_groups(dsts.len(), |i| effective_buffer(&dsts[i]).unwrap());
        for g in groups {
            let gs: Vec<TensorRef> = g.iter().map(|&i| Rc::clone(&srcs[i])).collect();
            let gd: Vec<TensorRef> = g.iter().map(|&i| Rc::clone(&dsts[i])).collect();
            tensor_copy_n(&gs, &gd)?;
        }
        return Ok(());
    }
    for i in 0..srcs.len() {
        tensor_copy(&srcs[i], &dsts[i])?;
    }
    Ok(())
}

/// `ggml_backend_view_init`.
pub fn view_init(tensor: &TensorRef) -> Result<Status> {
    let (view_src, view_offs) = {
        let t = tensor.borrow();
        (t.view_src.clone(), t.view_offs)
    };
    let src = view_src.ok_or_else(|| Error::new("view_src not set"))?;
    let (buffer, src_data) = {
        let s = src.borrow();
        (s.buffer.clone(), s.data.clone())
    };
    let buffer = buffer.ok_or_else(|| Error::new("view_src buffer not set"))?;
    let data = src_data.ok_or_else(|| Error::new("view_src data not set"))?;
    {
        let mut t = tensor.borrow_mut();
        t.buffer = Some(Rc::clone(&buffer));
        t.data = Some(super::tensor::TensorData {
            storage: Rc::clone(&data.storage),
            offset: data.offset + view_offs,
        });
    }
    buffer_init_tensor(&buffer, tensor)
}

/// `ggml_backend_tensor_alloc`: `addr` becomes `offset` in the buffer storage.
pub fn tensor_alloc(buffer: &BufferRef, tensor: &TensorRef, offset: usize) -> Result<Status> {
    let storage = Rc::clone(&buffer.borrow().storage);
    {
        let mut t = tensor.borrow_mut();
        t.buffer = Some(Rc::clone(buffer));
        t.data = Some(super::tensor::TensorData { storage: Rc::clone(&storage), offset });
    }
    buffer_init_tensor(buffer, tensor)
}

// ---------------------------------------------------------------------------
// multi-buffer
// ---------------------------------------------------------------------------

/// Buffer interface that forwards `clear` to each child buffer.
pub struct MultiBufferInterface {
    pub buffers: Vec<BufferRef>,
}

impl AsAny for MultiBufferInterface {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl BackendBufferInterface for MultiBufferInterface {
    fn set_tensor(&self, _buffer: &BufferRef, tensor: &TensorRef, data: &[u8], offset: usize, size: usize) {
        if let Some(dst) = super::tensor::data_location(tensor) {
            let start = dst.offset + offset;
            dst.storage.borrow_mut()[start..start + size].copy_from_slice(&data[..size]);
        }
    }

    fn get_tensor(&self, _buffer: &BufferRef, tensor: &TensorRef, data: &mut [u8], offset: usize, size: usize) {
        if let Some(src) = super::tensor::data_location(tensor) {
            let start = src.offset + offset;
            data[..size].copy_from_slice(&src.storage.borrow()[start..start + size]);
        }
    }

    fn clear(&self, _buffer: &BufferRef, value: u8) {
        for b in &self.buffers {
            buffer_clear(b, value);
        }
    }
}

/// `ggml_backend_multi_buffer_alloc_buffer`.
pub fn multi_buffer_alloc_buffer(buffers: &[BufferRef]) -> BufferRef {
    let total_size: usize = buffers.iter().map(|b| buffer_get_size(b)).sum();
    let iface = MultiBufferInterface { buffers: buffers.to_vec() };
    let buffer = buffer_init(Rc::clone(&buffers[0].borrow().buft), Rc::new(iface), total_size);
    buffer.borrow_mut().kind = BufferKind::Multi;
    buffer
}

/// `ggml_backend_buffer_is_multi_buffer`.
pub fn buffer_is_multi(buffer: &BufferRef) -> bool {
    buffer.borrow().kind == BufferKind::Multi
}

/// `ggml_backend_multi_buffer_set_usage`.
pub fn multi_buffer_set_usage(buffer: &BufferRef, usage: BufferUsage) {
    if let Some(iface) = buffer.borrow().iface.as_any().downcast_ref::<MultiBufferInterface>() {
        for b in &iface.buffers {
            buffer_set_usage(b, usage);
        }
    }
}

/// `ggml_backend_buffer_is_meta`.
pub fn buffer_is_meta(buffer: &BufferRef) -> bool {
    buffer.borrow().kind == BufferKind::Meta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_size_buffer_is_allowed() {
        let buft = super::super::cpu::cpu_buffer_type();
        let buf = buft_alloc_buffer(&buft, 0).unwrap();
        assert_eq!(buffer_get_size(&buf), 0);
        assert!(buffer_get_base(&buf).is_none());
    }

    #[test]
    fn allocation_plan_flushes_on_max_size() {
        let buft = Rc::new(BufferType {
            iface: Rc::new(super::super::cpu::CpuBufferType {
                name: "small".into(),
                max: 64,
            }),
            device: RefCell::new(None),
        });
        let tensors: Vec<TensorRef> = (0..4)
            .map(|i| Rc::new(RefCell::new(super::super::tensor::Tensor::new(
                format!("t{i}"), [16, 1, 1, 1], crate::gguf::GgmlType::F32,
            ))))
            .collect();
        let plan = alloc_buffer_n_plan(&buft, &tensors);
        assert!(plan.len() >= 2);
        assert_eq!(default_alloc_size_n(&buft, &tensors), plan.iter().map(|p| p.size).sum());
    }
}

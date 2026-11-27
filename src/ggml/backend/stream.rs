//! Backend streams, events, devices and registry-level accessors.
//!
//! Port of the backend/device/reg function groups in `ggml-backend.cpp`.

use std::any::Any;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

use crate::core::Result;

use super::buffer::{self, BufferRef, BuftRef};
use super::graph::GraphRef;
use super::registry::BackendRegistry;
use super::tensor::{self, TensorRef};
use super::traits::{BackendInterface, DeviceInterface, GraphOptimizeParams, GraphPlan, Proc};
use super::types::{DevProps, DevType, Status};

/// `ggml_backend_t`.
pub type BackendRef = Rc<RefCell<Backend>>;

/// `struct ggml_backend`; `guid` is a numeric stand-in for `ggml_guid_t`.
pub struct Backend {
    pub guid: u64,
    pub iface: Rc<dyn BackendInterface>,
    pub device: Option<DeviceRef>,
}

/// `ggml_backend_event_t`.
pub type EventRef = Rc<RefCell<Event>>;

/// `struct ggml_backend_event`.
pub struct Event {
    pub device: Option<DeviceRef>,
    pub context: Option<Rc<dyn Any>>,
}

/// `ggml_backend_dev_t`.
pub type DeviceRef = Rc<RefCell<Device>>;

/// `struct ggml_backend_device`. The registry back-link is weak to avoid a cycle.
pub struct Device {
    pub iface: Rc<dyn DeviceInterface>,
    pub reg: Option<Weak<RefCell<BackendRegistry>>>,
}

/// `ggml_backend_reg_t`.
pub type RegRef = Rc<RefCell<BackendRegistry>>;

/// `ggml_backend_guid`.
pub fn backend_guid(backend: &BackendRef) -> u64 {
    backend.borrow().guid
}

/// `ggml_backend_name`; a missing backend is named `"NULL"`.
pub fn backend_name(backend: &BackendRef) -> String {
    let b = backend.borrow();
    let iface = Rc::clone(&b.iface);
    iface.name(backend)
}

/// `NULL` backend name helper used by debug output.
pub fn null_backend_name() -> &'static str {
    "NULL"
}

/// `ggml_backend_free`: runs the hook and drops this reference.
pub fn backend_free(backend: &BackendRef) {
    let iface = Rc::clone(&backend.borrow().iface);
    iface.free(backend);
}

/// `ggml_backend_get_default_buffer_type`.
pub fn backend_default_buft(backend: &BackendRef) -> Result<BuftRef> {
    let dev = backend
        .borrow()
        .device
        .as_ref()
        .cloned()
        .ok_or_else(|| crate::core::Error::new("backend has no device"))?;
    dev_buffer_type(&dev)
}

/// `ggml_backend_alloc_buffer`.
pub fn backend_alloc_buffer(backend: &BackendRef, size: usize) -> Result<BufferRef> {
    buffer::buft_alloc_buffer(&backend_default_buft(backend)?, size)
}

/// `ggml_backend_get_alignment`.
pub fn backend_alignment(backend: &BackendRef) -> Result<usize> {
    Ok(buffer::buft_get_alignment(&backend_default_buft(backend)?))
}

/// `ggml_backend_get_max_size`.
pub fn backend_max_size(backend: &BackendRef) -> Result<usize> {
    Ok(buffer::buft_get_max_size(&backend_default_buft(backend)?))
}

/// `ggml_backend_tensor_set_async`.
pub fn tensor_set_async(
    backend: &BackendRef,
    tensor: &TensorRef,
    data: &[u8],
    offset: usize,
    size: usize,
) -> Result<()> {
    let iface = Rc::clone(&backend.borrow().iface);
    if iface.set_tensor_async(backend) {
        iface.do_set_tensor_async(backend, tensor, data, offset, size);
    } else {
        synchronize(backend);
        buffer::tensor_set(tensor, data, offset, size)?;
    }
    Ok(())
}

/// `ggml_backend_tensor_get_async`.
pub fn tensor_get_async(
    backend: &BackendRef,
    tensor: &TensorRef,
    data: &mut [u8],
    offset: usize,
    size: usize,
) -> Result<()> {
    let iface = Rc::clone(&backend.borrow().iface);
    if iface.get_tensor_async(backend) {
        iface.do_get_tensor_async(backend, tensor, data, offset, size);
    } else {
        synchronize(backend);
        buffer::tensor_get(tensor, data, offset, size)?;
    }
    Ok(())
}

/// `ggml_backend_tensor_set_2d_async`.
pub fn tensor_set_2d_async(
    backend: &BackendRef,
    tensor: &TensorRef,
    data: &[u8],
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) -> Result<()> {
    let iface = Rc::clone(&backend.borrow().iface);
    if n_copies <= 1 || !iface.set_tensor_2d_async_handled(backend) {
        for i in 0..n_copies {
            tensor_set_async(backend, tensor, &data[i * stride_data..], offset + i * stride_tensor, size)?;
        }
        return Ok(());
    }
    if size == 0 {
        return Ok(());
    }
    iface.do_set_tensor_2d_async(
        backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data,
    );
    Ok(())
}

/// `ggml_backend_tensor_get_2d_async`.
pub fn tensor_get_2d_async(
    backend: &BackendRef,
    tensor: &TensorRef,
    data: &mut [u8],
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) -> Result<()> {
    let iface = Rc::clone(&backend.borrow().iface);
    if n_copies <= 1 || !iface.get_tensor_2d_async_handled(backend) {
        for i in 0..n_copies {
            tensor_get_async(backend, tensor, &mut data[i * stride_data..], offset + i * stride_tensor, size)?;
        }
        return Ok(());
    }
    if size == 0 {
        return Ok(());
    }
    iface.do_get_tensor_2d_async(
        backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data,
    );
    Ok(())
}

/// `ggml_backend_synchronize`.
pub fn synchronize(backend: &BackendRef) {
    let iface = Rc::clone(&backend.borrow().iface);
    if iface.synchronize(backend) {
        iface.do_synchronize(backend);
    }
}

/// `ggml_backend_graph_plan_create`.
pub fn graph_plan_create(backend: &BackendRef, graph: &GraphRef) -> Result<GraphPlan> {
    let iface = Rc::clone(&backend.borrow().iface);
    iface.graph_plan_create(backend, graph).ok_or_else(|| {
        crate::core::Error::new("backend does not support graph plans")
    })
}

/// `ggml_backend_graph_plan_free`.
pub fn graph_plan_free(backend: &BackendRef, plan: &GraphPlan) {
    let iface = Rc::clone(&backend.borrow().iface);
    iface.graph_plan_free(backend, plan);
}

/// `ggml_backend_graph_plan_update` (internal hook).
pub fn graph_plan_update(backend: &BackendRef, plan: &GraphPlan, graph: &GraphRef) {
    let iface = Rc::clone(&backend.borrow().iface);
    iface.graph_plan_update(backend, plan, graph);
}

/// `ggml_backend_graph_plan_compute`.
pub fn graph_plan_compute(backend: &BackendRef, plan: &GraphPlan) -> Status {
    let iface = Rc::clone(&backend.borrow().iface);
    iface.graph_plan_compute(backend, plan)
}

/// `ggml_backend_graph_compute`: compute then synchronize.
pub fn graph_compute(backend: &BackendRef, graph: &GraphRef) -> Status {
    let status = graph_compute_async(backend, graph);
    synchronize(backend);
    status
}

/// `ggml_backend_graph_compute_async`.
pub fn graph_compute_async(backend: &BackendRef, graph: &GraphRef) -> Status {
    let iface = Rc::clone(&backend.borrow().iface);
    iface.graph_compute(backend, graph)
}

/// `ggml_backend_supports_op`.
pub fn supports_op(backend: &BackendRef, op: &TensorRef) -> bool {
    match &backend.borrow().device {
        Some(dev) => dev_supports_op(dev, op),
        None => false,
    }
}

/// `ggml_backend_supports_buft`.
pub fn supports_buft(backend: &BackendRef, buft: &BuftRef) -> bool {
    match &backend.borrow().device {
        Some(dev) => dev_supports_buft(dev, buft),
        None => false,
    }
}

/// `ggml_backend_offload_op`.
pub fn offload_op(backend: &BackendRef, op: &TensorRef) -> bool {
    match &backend.borrow().device {
        Some(dev) => dev_offload_op(dev, op),
        None => false,
    }
}

/// `ggml_backend_get_device`.
pub fn backend_device(backend: &BackendRef) -> Option<DeviceRef> {
    backend.borrow().device.clone()
}

/// `ggml_backend_tensor_copy_async`.
pub fn tensor_copy_async(
    backend_src: &BackendRef,
    backend_dst: &BackendRef,
    src: &TensorRef,
    dst: &TensorRef,
) -> Result<()> {
    if Rc::ptr_eq(src, dst) {
        return Ok(());
    }
    let iface = Rc::clone(&backend_dst.borrow().iface);
    if iface.cpy_tensor_async(backend_src, backend_dst, src, dst) {
        return Ok(());
    }
    synchronize(backend_src);
    synchronize(backend_dst);
    buffer::tensor_copy(src, dst)
}

/// `ggml_backend_event_new`.
pub fn event_new(device: &DeviceRef) -> Option<EventRef> {
    let dev = device.borrow();
    let iface = Rc::clone(&dev.iface);
    iface.event_new(device).map(|event| {
        event.borrow_mut().device = Some(Rc::clone(device));
        event
    })
}

/// `ggml_backend_event_free`.
pub fn event_free(event: &EventRef) {
    let dev = event.borrow().device.clone();
    if let Some(dev) = dev {
        let iface = Rc::clone(&dev.borrow().iface);
        iface.event_free(&dev, event);
    }
}

/// `ggml_backend_event_record`.
pub fn event_record(event: &EventRef, backend: &BackendRef) -> bool {
    let iface = Rc::clone(&backend.borrow().iface);
    iface.event_record(backend, event)
}

/// `ggml_backend_event_synchronize`.
pub fn event_synchronize(event: &EventRef) {
    if let Some(dev) = event.borrow().device.clone() {
        let iface = Rc::clone(&dev.borrow().iface);
        iface.event_synchronize(&dev, event);
    }
}

/// `ggml_backend_event_wait`.
pub fn event_wait(backend: &BackendRef, event: &EventRef) -> bool {
    let iface = Rc::clone(&backend.borrow().iface);
    iface.event_wait(backend, event)
}

/// `ggml_backend_graph_optimize` (internal).
pub fn graph_optimize(backend: &BackendRef, graph: &GraphRef, params: &GraphOptimizeParams) {
    let iface = Rc::clone(&backend.borrow().iface);
    iface.graph_optimize(backend, graph, params);
}

// ---------------------------------------------------------------------------
// device
// ---------------------------------------------------------------------------

/// `ggml_backend_dev_name`.
pub fn dev_name(device: &DeviceRef) -> String {
    let iface = Rc::clone(&device.borrow().iface);
    iface.name(device)
}

/// `ggml_backend_dev_description`.
pub fn dev_description(device: &DeviceRef) -> String {
    let iface = Rc::clone(&device.borrow().iface);
    iface.description(device)
}

/// `ggml_backend_dev_memory` returns `(free, total)` instead of writing through pointers.
pub fn dev_memory(device: &DeviceRef) -> (usize, usize) {
    let iface = Rc::clone(&device.borrow().iface);
    iface.memory(device)
}

/// `ggml_backend_dev_type`.
pub fn dev_type(device: &DeviceRef) -> DevType {
    let iface = Rc::clone(&device.borrow().iface);
    iface.dev_type(device)
}

/// `ggml_backend_dev_get_props`.
pub fn dev_get_props(device: &DeviceRef) -> DevProps {
    let mut props = DevProps::default();
    let iface = Rc::clone(&device.borrow().iface);
    iface.props(device, &mut props);
    props
}

/// `ggml_backend_dev_backend_reg`.
pub fn dev_backend_reg(device: &DeviceRef) -> Option<RegRef> {
    device.borrow().reg.as_ref().and_then(|w| w.upgrade())
}

/// `ggml_backend_dev_init`.
pub fn dev_init(device: &DeviceRef, params: Option<&str>) -> Result<BackendRef> {
    let iface = Rc::clone(&device.borrow().iface);
    let backend = iface.init_backend(device, params)?;
    backend.borrow_mut().device = Some(Rc::clone(device));
    Ok(backend)
}

/// `ggml_backend_dev_buffer_type`.
pub fn dev_buffer_type(device: &DeviceRef) -> Result<BuftRef> {
    let iface = Rc::clone(&device.borrow().iface);
    let buft = iface.buffer_type(device);
    let mut device_slot = buft.device.borrow_mut();
    if device_slot.is_none() {
        *device_slot = Some(Rc::downgrade(device));
    }
    drop(device_slot);
    Ok(buft)
}

/// `ggml_backend_dev_host_buffer_type`.
pub fn dev_host_buffer_type(device: &DeviceRef) -> Option<BuftRef> {
    let iface = Rc::clone(&device.borrow().iface);
    iface.host_buffer_type(device)
}

/// `ggml_backend_dev_buffer_from_host_ptr`.
pub fn dev_buffer_from_host_ptr(
    device: &DeviceRef,
    storage: Rc<RefCell<Vec<u8>>>,
    size: usize,
    max_tensor_size: usize,
) -> Option<BufferRef> {
    let iface = Rc::clone(&device.borrow().iface);
    iface.buffer_from_host_ptr(device, storage, size, max_tensor_size)
}

/// `ggml_backend_dev_supports_op`.
pub fn dev_supports_op(device: &DeviceRef, op: &TensorRef) -> bool {
    let iface = Rc::clone(&device.borrow().iface);
    iface.supports_op(device, op)
}

/// `ggml_backend_dev_supports_buft`.
pub fn dev_supports_buft(device: &DeviceRef, buft: &BuftRef) -> bool {
    let iface = Rc::clone(&device.borrow().iface);
    iface.supports_buft(device, buft)
}

/// `ggml_backend_dev_offload_op`.
pub fn dev_offload_op(device: &DeviceRef, op: &TensorRef) -> bool {
    let iface = Rc::clone(&device.borrow().iface);
    iface.offload_op(device, op)
}

// ---------------------------------------------------------------------------
// backend registry (reg) accessors
// ---------------------------------------------------------------------------

/// `ggml_backend_reg_name`.
pub fn reg_name(reg: &RegRef) -> String {
    let iface = Rc::clone(&reg.borrow().iface);
    iface.name(reg)
}

/// `ggml_backend_reg_dev_count`.
pub fn reg_dev_count(reg: &RegRef) -> usize {
    let iface = Rc::clone(&reg.borrow().iface);
    iface.device_count(reg)
}

/// `ggml_backend_reg_dev_get`.
pub fn reg_dev_get(reg: &RegRef, index: usize) -> Option<DeviceRef> {
    if index >= reg_dev_count(reg) {
        return None;
    }
    let iface = Rc::clone(&reg.borrow().iface);
    let dev = iface.device(reg, index);
    dev.borrow_mut().reg = Some(Rc::downgrade(reg));
    Some(dev)
}

/// `ggml_backend_reg_get_proc_address`: looks up an optional Rust-side
/// extension proc registered under `name`.
pub fn reg_get_proc_address(reg: &RegRef, name: &str) -> Option<Proc> {
    if let Some(proc) = reg.borrow().procs.get(name).cloned() {
        return Some(proc);
    }
    let iface = Rc::clone(&reg.borrow().iface);
    iface.proc_address(reg, name)
}

// ---------------------------------------------------------------------------
// utils
// ---------------------------------------------------------------------------

/// `ggml_backend_graph_copy` result struct.
#[derive(Clone)]
pub struct GraphCopy {
    pub buffer: BufferRef,
    pub graph: GraphRef,
}

/// `ggml_backend_view_init` re-export.
pub fn view_init(tensor: &TensorRef) -> Result<Status> {
    buffer::view_init(tensor)
}

/// `ggml_backend_tensor_alloc` re-export.
pub fn tensor_alloc(buffer: &BufferRef, tensor: &TensorRef, offset: usize) -> Result<Status> {
    buffer::tensor_alloc(buffer, tensor, offset)
}

/// `ggml_backend_cpu_buffer_from_ptr`: copies the supplied bytes into a buffer.
pub fn cpu_buffer_from_ptr(storage: Rc<RefCell<Vec<u8>>>, size: usize) -> Result<BufferRef> {
    super::cpu::cpu_buffer_from_ptr(storage, size)
}

/// `ggml_backend_cpu_buffer_type`.
pub fn cpu_buffer_type() -> BuftRef {
    super::cpu::cpu_buffer_type()
}

/// `tensor::ggml_nbytes` helper re-export.
pub fn tensor_nbytes(t: &TensorRef) -> usize {
    tensor::nbytes_ref(t)
}

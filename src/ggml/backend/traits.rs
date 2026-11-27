//! Rust replacements for the C++ `*_i` virtual function tables.
//!
//! Each C++ interface struct is a trait with default implementations for the
//! optional hooks. A `None` C function pointer becomes either `false` from a
//! `*_handled` hook or the documented default value.

use std::any::Any;
use std::rc::Rc;

use crate::core::Result;

use super::buffer::{BufferRef, BuftRef};
use super::stream::{BackendRef, DeviceRef, EventRef, RegRef};
use super::tensor::{GraphRef, TensorRef};
use super::types::{DevProps, DevType, Status};

/// `ggml_backend_graph_plan_t` (opaque backend-owned object).
pub type GraphPlan = Rc<dyn Any>;
/// Generic stand-in for `void *` returned by `ggml_backend_reg_get_proc_address`.
pub type Proc = Rc<dyn Any>;

/// Downcast support for trait objects; replaces C++ interface-tag checks.
pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
}

/// `struct ggml_backend_buffer_type_i`.
pub trait BufferTypeInterface: AsAny {
    fn name(&self, buft: &BuftRef) -> String;
    fn alloc_buffer(&self, buft: &BuftRef, size: usize) -> Result<BufferRef>;
    fn alloc_buffer_n(&self, buft: &BuftRef, tensors: &[TensorRef]) -> Result<BufferRef> {
        super::buffer::default_alloc_buffer_n(buft, tensors)
    }
    fn alignment(&self, _buft: &BuftRef) -> usize;
    fn max_size(&self, _buft: &BuftRef) -> usize {
        usize::MAX
    }
    fn alloc_size(&self, _buft: &BuftRef, tensor: &TensorRef) -> usize {
        super::tensor::nbytes_ref(tensor)
    }
    fn alloc_size_n(&self, buft: &BuftRef, tensors: &[TensorRef]) -> usize {
        super::buffer::default_alloc_size_n(buft, tensors)
    }
    fn is_host(&self, _buft: &BuftRef) -> bool {
        false
    }
}

/// `struct ggml_backend_buffer_i`.
pub trait BackendBufferInterface: AsAny {
    fn free_buffer(&self, _buffer: &BufferRef) {}
    fn get_base(&self, _buffer: &BufferRef) -> Option<Rc<std::cell::RefCell<Vec<u8>>>> {
        None
    }
    fn init_tensor(&self, _buffer: &BufferRef, _tensor: &TensorRef) -> Result<Status> {
        Ok(Status::Success)
    }
    /// Returns false to use the chunked `set_tensor` fallback.
    fn memset_handled(&self, _buffer: &BufferRef) -> bool {
        false
    }
    fn memset_tensor(&self, _buffer: &BufferRef, _tensor: &TensorRef, _value: u8, _offset: usize, _size: usize) {
    }
    fn set_tensor(&self, buffer: &BufferRef, tensor: &TensorRef, data: &[u8], offset: usize, size: usize);
    fn get_tensor(&self, buffer: &BufferRef, tensor: &TensorRef, data: &mut [u8], offset: usize, size: usize);
    fn set_tensor_2d_handled(&self, _buffer: &BufferRef) -> bool {
        false
    }
    fn set_tensor_2d(
        &self,
        _buffer: &BufferRef,
        _tensor: &TensorRef,
        _data: &[u8],
        _offset: usize,
        _size: usize,
        _n_copies: usize,
        _stride_tensor: usize,
        _stride_data: usize,
    ) {
    }
    fn get_tensor_2d_handled(&self, _buffer: &BufferRef) -> bool {
        false
    }
    fn get_tensor_2d(
        &self,
        _buffer: &BufferRef,
        _tensor: &TensorRef,
        _data: &mut [u8],
        _offset: usize,
        _size: usize,
        _n_copies: usize,
        _stride_tensor: usize,
        _stride_data: usize,
    ) {
    }
    fn cpy_tensor(&self, _buffer: &BufferRef, _src: &TensorRef, _dst: &TensorRef) -> bool {
        false
    }
    fn clear(&self, buffer: &BufferRef, value: u8) {
        super::buffer::clear_buffer_storage(buffer, value);
    }
    fn reset(&self, _buffer: &BufferRef) {}
    fn set_tensor_n_handled(&self, _buffer: &BufferRef) -> bool {
        false
    }
    fn set_tensor_n(
        &self,
        _buffer: &BufferRef,
        _tensors: &[TensorRef],
        _data: &[u8],
        _offsets: &[usize],
        _sizes: &[usize],
        _data_offsets: &[usize],
        _n_tensors: usize,
    ) {
    }
    fn get_tensor_n_handled(&self, _buffer: &BufferRef) -> bool {
        false
    }
    fn get_tensor_n(
        &self,
        _buffer: &BufferRef,
        _tensors: &[TensorRef],
        _data: &mut [u8],
        _offsets: &[usize],
        _sizes: &[usize],
        _data_offsets: &[usize],
        _n_tensors: usize,
    ) {
    }
    fn cpy_tensor_n(&self, _buffer: &BufferRef, _srcs: &[TensorRef], _dsts: &[TensorRef]) -> bool {
        let _ = (_srcs, _dsts);
        false
    }
}

/// `struct ggml_backend_graph_optimize_params`.
#[derive(Clone, Default)]
pub struct GraphOptimizeParams {
    pub add_alloc_dep: Option<Rc<dyn Fn(&TensorRef, &TensorRef)>>,
    pub add_scratch: Option<Rc<dyn Fn(usize, &TensorRef, &TensorRef) -> Option<TensorRef>>>,
    pub user_data: Option<Rc<dyn Any>>,
}

/// `struct ggml_backend_i`.
pub trait BackendInterface: AsAny {
    fn name(&self, backend: &BackendRef) -> String;
    fn free(&self, _backend: &BackendRef) {}

    /// false means: synchronize, then use the blocking buffer path.
    fn set_tensor_async(&self, _backend: &BackendRef) -> bool {
        false
    }
    fn do_set_tensor_async(
        &self,
        _backend: &BackendRef,
        _tensor: &TensorRef,
        _data: &[u8],
        _offset: usize,
        _size: usize,
    ) {
    }
    fn get_tensor_async(&self, _backend: &BackendRef) -> bool {
        false
    }
    fn do_get_tensor_async(
        &self,
        _backend: &BackendRef,
        _tensor: &TensorRef,
        _data: &mut [u8],
        _offset: usize,
        _size: usize,
    ) {
    }
    fn set_tensor_2d_async_handled(&self, _backend: &BackendRef) -> bool {
        false
    }
    fn do_set_tensor_2d_async(
        &self,
        _backend: &BackendRef,
        _tensor: &TensorRef,
        _data: &[u8],
        _offset: usize,
        _size: usize,
        _n_copies: usize,
        _stride_tensor: usize,
        _stride_data: usize,
    ) {
        let _ = (_tensor, _data, _offset, _size, _n_copies, _stride_tensor, _stride_data);
    }
    fn get_tensor_2d_async_handled(&self, _backend: &BackendRef) -> bool {
        false
    }
    fn do_get_tensor_2d_async(
        &self,
        _backend: &BackendRef,
        _tensor: &TensorRef,
        _data: &mut [u8],
        _offset: usize,
        _size: usize,
        _n_copies: usize,
        _stride_tensor: usize,
        _stride_data: usize,
    ) {
        let _ = (_tensor, _data, _offset, _size, _n_copies, _stride_tensor, _stride_data);
    }
    fn cpy_tensor_async(
        &self,
        _backend_src: &BackendRef,
        _backend_dst: &BackendRef,
        _src: &TensorRef,
        _dst: &TensorRef,
    ) -> bool {
        false
    }
    fn synchronize(&self, _backend: &BackendRef) -> bool {
        false
    }
    fn do_synchronize(&self, _backend: &BackendRef) {}

    fn graph_plan_create(&self, _backend: &BackendRef, _graph: &GraphRef) -> Option<GraphPlan> {
        None
    }
    fn graph_plan_free(&self, _backend: &BackendRef, _plan: &GraphPlan) {}
    fn graph_plan_update(&self, _backend: &BackendRef, _plan: &GraphPlan, _graph: &GraphRef) {}
    fn graph_plan_compute(&self, _backend: &BackendRef, _plan: &GraphPlan) -> Status {
        Status::Failed
    }

    fn graph_compute(&self, _backend: &BackendRef, _graph: &GraphRef) -> Status {
        Status::Failed
    }

    fn event_record(&self, _backend: &BackendRef, _event: &EventRef) -> bool {
        false
    }
    fn event_wait(&self, _backend: &BackendRef, _event: &EventRef) -> bool {
        false
    }

    fn graph_optimize(&self, _backend: &BackendRef, _graph: &GraphRef, _params: &GraphOptimizeParams) -> bool {
        false
    }
}

/// `struct ggml_backend_device_i`.
pub trait DeviceInterface: AsAny {
    fn name(&self, dev: &DeviceRef) -> String;
    fn description(&self, dev: &DeviceRef) -> String {
        self.name(dev)
    }
    fn memory(&self, _dev: &DeviceRef) -> (usize, usize) {
        (0, 0)
    }
    fn dev_type(&self, _dev: &DeviceRef) -> DevType {
        DevType::Cpu
    }
    fn props(&self, dev: &DeviceRef, props: &mut DevProps) {
        let (free, total) = self.memory(dev);
        props.name = self.name(dev);
        props.description = self.description(dev);
        props.memory_free = free;
        props.memory_total = total;
        props.type_ = self.dev_type(dev);
        props.device_id = None;
        props.caps = Default::default();
    }
    fn init_backend(&self, dev: &DeviceRef, params: Option<&str>) -> Result<BackendRef>;
    fn buffer_type(&self, dev: &DeviceRef) -> BuftRef;
    fn host_buffer_type(&self, _dev: &DeviceRef) -> Option<BuftRef> {
        None
    }
    fn buffer_from_host_ptr(
        &self,
        _dev: &DeviceRef,
        _storage: Rc<std::cell::RefCell<Vec<u8>>>,
        _size: usize,
        _max_tensor_size: usize,
    ) -> Option<BufferRef> {
        None
    }
    fn supports_op(&self, _dev: &DeviceRef, _op: &TensorRef) -> bool;
    fn supports_buft(&self, _dev: &DeviceRef, _buft: &BuftRef) -> bool;
    fn offload_op(&self, _dev: &DeviceRef, _op: &TensorRef) -> bool {
        false
    }
    fn event_new(&self, _dev: &DeviceRef) -> Option<EventRef> {
        None
    }
    fn event_free(&self, _dev: &DeviceRef, _event: &EventRef) {}
    fn event_synchronize(&self, _dev: &DeviceRef, _event: &EventRef) {}
}

/// `struct ggml_backend_reg_i`.
pub trait RegistryInterface: AsAny {
    fn name(&self, reg: &RegRef) -> String;
    fn device_count(&self, reg: &RegRef) -> usize;
    fn device(&self, reg: &RegRef, index: usize) -> DeviceRef;
    fn proc_address(&self, _reg: &RegRef, _name: &str) -> Option<Proc> {
        None
    }
}

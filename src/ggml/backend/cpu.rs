//! CPU buffer type and the always-available CPU backend.
//!
//! Port of the bottom section of `ggml-backend.cpp`. The CPU buffer is a
//! host buffer: tensor data lives in a plain `Vec<u8>`.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::core::{Error, Result};

use super::buffer::{self, BufferRef, BufferType, BuftRef};
use super::graph::GraphRef;
use super::stream::{BackendRef, DeviceRef, RegRef};
use super::tensor::{data_location, TensorRef};
use super::traits::{
    AsAny, BackendBufferInterface, BackendInterface, BufferTypeInterface, DeviceInterface,
    GraphOptimizeParams, RegistryInterface,
};
use super::types::{DevProps, DevType, Status, TENSOR_ALIGNMENT};

/// CPU buffer interface: direct copies into tensor byte storage.
pub struct CpuBufferInterface;

impl AsAny for CpuBufferInterface {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl BackendBufferInterface for CpuBufferInterface {
    fn get_base(&self, buffer: &BufferRef) -> Option<Rc<RefCell<Vec<u8>>>> {
        Some(Rc::clone(&buffer.borrow().storage))
    }

    fn memset_handled(&self, _buffer: &BufferRef) -> bool {
        true
    }

    fn memset_tensor(&self, _buffer: &BufferRef, tensor: &TensorRef, value: u8, offset: usize, size: usize) {
        if let Some(data) = data_location(tensor) {
            let start = data.offset + offset;
            data.storage.borrow_mut()[start..start + size].fill(value);
        }
    }

    fn set_tensor(&self, _buffer: &BufferRef, tensor: &TensorRef, data: &[u8], offset: usize, size: usize) {
        if let Some(dst) = data_location(tensor) {
            let start = dst.offset + offset;
            dst.storage.borrow_mut()[start..start + size].copy_from_slice(&data[..size]);
        }
    }

    fn get_tensor(&self, _buffer: &BufferRef, tensor: &TensorRef, data: &mut [u8], offset: usize, size: usize) {
        if let Some(src) = data_location(tensor) {
            let start = src.offset + offset;
            data[..size].copy_from_slice(&src.storage.borrow()[start..start + size]);
        }
    }

    fn cpy_tensor(&self, _buffer: &BufferRef, src: &TensorRef, dst: &TensorRef) -> bool {
        let src_buf = match super::tensor::effective_buffer(src) {
            Some(b) => b,
            None => return false,
        };
        if buffer::buffer_is_host(&src_buf) {
            let nbytes = super::tensor::nbytes_ref(src);
            if let (Some(s), Some(d)) = (data_location(src), data_location(dst)) {
                let bytes = s.storage.borrow()[s.offset..s.offset + nbytes].to_vec();
                d.storage.borrow_mut()[d.offset..d.offset + nbytes].copy_from_slice(&bytes);
                return true;
            }
        }
        false
    }

    fn clear(&self, buffer: &BufferRef, value: u8) {
        buffer::clear_buffer_storage(buffer, value);
    }
}

/// `ggml_backend_cpu_buffer_type` iface. `max` is the `get_max_size` value.
pub struct CpuBufferType {
    pub name: String,
    pub max: usize,
}

impl AsAny for CpuBufferType {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl BufferTypeInterface for CpuBufferType {
    fn name(&self, _buft: &BuftRef) -> String {
        self.name.clone()
    }

    fn alloc_buffer(&self, buft: &BuftRef, size: usize) -> Result<BufferRef> {
        if size > self.max {
            return Err(Error::new(format!(
                "{}: buffer of size {size} exceeds max size {}",
                self.name, self.max
            )));
        }
        Ok(buffer::buffer_init(Rc::clone(buft), Rc::new(CpuBufferInterface), size))
    }

    fn alignment(&self, _buft: &BuftRef) -> usize {
        TENSOR_ALIGNMENT
    }

    fn max_size(&self, _buft: &BuftRef) -> usize {
        self.max
    }

    fn is_host(&self, _buft: &BuftRef) -> bool {
        true
    }
}

thread_local! {
    static CPU_BUFT: BuftRef = Rc::new(BufferType {
        iface: Rc::new(CpuBufferType { name: "CPU".into(), max: usize::MAX }),
        device: RefCell::new(None),
    });
    static CPU_MAPPED_BUFT: BuftRef = Rc::new(BufferType {
        iface: Rc::new(CpuBufferType { name: "CPU_Mapped".into(), max: usize::MAX }),
        device: RefCell::new(None),
    });
}

/// `ggml_backend_cpu_buffer_type`.
pub fn cpu_buffer_type() -> BuftRef {
    CPU_BUFT.with(|b| Rc::clone(b))
}

/// `ggml_backend_cpu_buffer_from_ptr_type` (internal).
pub fn cpu_buffer_from_ptr_type() -> BuftRef {
    CPU_MAPPED_BUFT.with(|b| Rc::clone(b))
}

/// `ggml_backend_cpu_buffer_from_ptr`: wraps already-owned bytes.
pub fn cpu_buffer_from_ptr(storage: Rc<RefCell<Vec<u8>>>, size: usize) -> Result<BufferRef> {
    let size = size.min(storage.borrow().len());
    Ok(buffer::buffer_init_with_storage(
        cpu_buffer_from_ptr_type(),
        Rc::new(CpuBufferInterface),
        storage,
        size,
    ))
}

/// CPU backend stream. Actual op execution lives in the ggml compute module;
/// this interface only owns buffers and synchronizes immediately.
pub struct CpuBackend;

impl AsAny for CpuBackend {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl BackendInterface for CpuBackend {
    fn name(&self, _backend: &BackendRef) -> String {
        "CPU".into()
    }

    fn synchronize(&self, _backend: &BackendRef) -> bool {
        true
    }

    fn graph_compute(&self, _backend: &BackendRef, _graph: &GraphRef) -> Status {
        Status::Success
    }

    fn graph_optimize(&self, _backend: &BackendRef, _graph: &GraphRef, _params: &GraphOptimizeParams) -> bool {
        false
    }
}

/// CPU device.
pub struct CpuDevice;

impl AsAny for CpuDevice {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl DeviceInterface for CpuDevice {
    fn name(&self, _dev: &DeviceRef) -> String {
        "CPU".into()
    }

    fn description(&self, _dev: &DeviceRef) -> String {
        "CPU".into()
    }

    fn dev_type(&self, _dev: &DeviceRef) -> DevType {
        DevType::Cpu
    }

    fn props(&self, dev: &DeviceRef, props: &mut DevProps) {
        props.name = self.name(dev);
        props.description = self.description(dev);
        props.type_ = DevType::Cpu;
        props.device_id = None;
        props.memory_free = 0;
        props.memory_total = 0;
        props.caps = Default::default();
    }

    fn init_backend(&self, dev: &DeviceRef, _params: Option<&str>) -> Result<BackendRef> {
        let backend = Rc::new(RefCell::new(super::stream::Backend {
            guid: 1,
            iface: Rc::new(CpuBackend),
            device: Some(Rc::clone(dev)),
        }));
        Ok(backend)
    }

    fn buffer_type(&self, _dev: &DeviceRef) -> BuftRef {
        cpu_buffer_type()
    }

    fn supports_op(&self, _dev: &DeviceRef, _op: &TensorRef) -> bool {
        true
    }

    fn supports_buft(&self, _dev: &DeviceRef, buft: &BuftRef) -> bool {
        super::buffer::buft_is_host(buft)
    }
}

/// Registry containing the single CPU device.
pub struct CpuRegistry;

impl AsAny for CpuRegistry {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl RegistryInterface for CpuRegistry {
    fn name(&self, _reg: &RegRef) -> String {
        "CPU".into()
    }

    fn device_count(&self, _reg: &RegRef) -> usize {
        1
    }

    fn device(&self, reg: &RegRef, index: usize) -> DeviceRef {
        assert!(index == 0, "CPU registry has one device");
        let dev = Rc::new(RefCell::new(super::stream::Device {
            iface: Rc::new(CpuDevice),
            reg: Some(Rc::downgrade(reg)),
        }));
        dev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgmlType;

    #[test]
    fn cpu_buffer_round_trip() {
        let buft = cpu_buffer_type();
        let buf = buffer::buft_alloc_buffer(&buft, 64).unwrap();
        let t = Rc::new(RefCell::new(super::super::tensor::Tensor::new(
            "t", [16, 1, 1, 1], GgmlType::F32,
        )));
        buffer::tensor_alloc(&buf, &t, 0).unwrap();
        let data = [1u8, 2, 3, 4];
        buffer::tensor_set(&t, &data, 0, 4).unwrap();
        let mut out = [0u8; 4];
        buffer::tensor_get(&t, &mut out, 0, 4).unwrap();
        assert_eq!(out, data);
    }
}

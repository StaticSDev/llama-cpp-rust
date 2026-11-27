//! Meta (tensor-parallel) device, buffer type, buffer and backend metadata.
//!
//! Port of the interface-level parts of `ggml-backend-meta.cpp`: split axis
//! metadata, split-state lookup, device/buffer type composition and the
//! simple-device views used by the scheduler. Per-device tensor slicing is
//! done by the compute module, which owns the ggml graph.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::core::{Error, Result};

use super::buffer::{self, BufferRef, BufferType, BuftRef};
use super::graph::GraphRef;
use super::stream::{BackendRef, Device, DeviceRef, RegRef};
use super::tensor::{data_location, flash_attn_ext_has_indexed_bias, TensorRef};
use super::traits::{
    AsAny, BackendBufferInterface, BackendInterface, BufferTypeInterface, DeviceInterface,
    GraphOptimizeParams,
};
use super::types::{BufferUsage, DevProps, DevType, SplitAxis, SplitState, Status, META_MAX_DEVICES};

/// Callback that assigns split states to statically allocated tensors.
pub type GetSplitState = Rc<dyn Fn(&TensorRef) -> SplitState>;

/// `ggml_backend_meta_device_context`.
pub struct MetaDevice {
    pub simple_devs: Vec<DeviceRef>,
    pub get_split_state: Option<GetSplitState>,
    pub name: String,
    pub description: String,
}

impl AsAny for MetaDevice {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl MetaDevice {
    fn new(simple_devs: Vec<DeviceRef>, get_split_state: Option<GetSplitState>) -> Self {
        let mut name = String::from("Meta(");
        let mut description = String::from("Meta(");
        for (i, dev) in simple_devs.iter().enumerate() {
            if i > 0 {
                name.push(',');
                description.push(',');
            }
            name.push_str(&super::stream::dev_name(dev));
            description.push_str(&super::stream::dev_description(dev));
        }
        name.push(')');
        description.push(')');
        Self { simple_devs, get_split_state, name, description }
    }
}

thread_local! {
    static META_DEVS: RefCell<Vec<DeviceRef>> = RefCell::new(Vec::new());
    static META_BUFT_BY_DEV: RefCell<Vec<(usize, BuftRef)>> = RefCell::new(Vec::new());
}

fn same_devices(a: &[DeviceRef], b: &[DeviceRef]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| Rc::ptr_eq(x, y))
}

/// `ggml_backend_dev_is_meta`.
pub fn dev_is_meta(dev: &DeviceRef) -> bool {
    dev.borrow().iface.as_any().downcast_ref::<MetaDevice>().is_some()
}

/// `ggml_backend_buft_is_meta`.
pub fn buft_is_meta(buft: &BuftRef) -> bool {
    buft.iface.as_any().downcast_ref::<MetaBufferType>().is_some()
}

/// `ggml_backend_buffer_is_meta`.
pub fn buffer_is_meta(buffer: &BufferRef) -> bool {
    buffer.borrow().kind == super::buffer::BufferKind::Meta
}

/// `ggml_backend_is_meta`.
pub fn backend_is_meta(backend: &BackendRef) -> bool {
    backend.borrow().iface.as_any().downcast_ref::<MetaBackend>().is_some()
}

/// `ggml_backend_meta_dev_n_devs`.
pub fn meta_dev_n_devs(dev: &DeviceRef) -> usize {
    dev.borrow()
        .iface
        .as_any().downcast_ref::<MetaDevice>()
        .map(|c| c.simple_devs.len())
        .unwrap_or(0)
}

/// `ggml_backend_meta_dev_simple_dev`.
pub fn meta_dev_simple_dev(dev: &DeviceRef, index: usize) -> Option<DeviceRef> {
    dev.borrow()
        .iface
        .as_any().downcast_ref::<MetaDevice>()
        .and_then(|c| c.simple_devs.get(index).cloned())
}

/// `ggml_backend_meta_device`: creates or reuses a meta device.
pub fn meta_device(
    devs: &[DeviceRef],
    get_split_state: Option<GetSplitState>,
) -> Result<DeviceRef> {
    if devs.len() > META_MAX_DEVICES {
        return Err(Error::new(format!(
            "ggml_backend_meta_device: too many devices ({}, max {META_MAX_DEVICES})",
            devs.len()
        )));
    }
    let existing = META_DEVS.with(|cache| {
        cache.borrow().iter().find(|dev| {
            dev.borrow()
                .iface
                .as_any().downcast_ref::<MetaDevice>()
                .map(|c| same_devices(&c.simple_devs, devs))
                .unwrap_or(false)
        }).cloned()
    });
    if let Some(dev) = existing {
        return Ok(dev);
    }

    let dev = Rc::new(RefCell::new(Device {
        iface: Rc::new(MetaDevice::new(devs.to_vec(), get_split_state)),
        reg: None,
    }));
    META_DEVS.with(|cache| cache.borrow_mut().push(Rc::clone(&dev)));
    Ok(dev)
}

impl DeviceInterface for MetaDevice {
    fn name(&self, _dev: &DeviceRef) -> String {
        self.name.clone()
    }

    fn description(&self, _dev: &DeviceRef) -> String {
        self.description.clone()
    }

    fn memory(&self, _dev: &DeviceRef) -> (usize, usize) {
        let mut free = 0usize;
        let mut total = 0usize;
        for dev in &self.simple_devs {
            let (f, t) = super::stream::dev_memory(dev);
            free += f;
            total += t;
        }
        (free, total)
    }

    fn dev_type(&self, _dev: &DeviceRef) -> DevType {
        DevType::Meta
    }

    fn props(&self, dev: &DeviceRef, props: &mut DevProps) {
        props.name = self.name(dev);
        props.description = self.description(dev);
        props.type_ = DevType::Meta;
        props.device_id = None;
        let (free, total) = self.memory(dev);
        props.memory_free = free;
        props.memory_total = total;
        props.caps = super::types::DevCaps {
            async_: true,
            host_buffer: false,
            buffer_from_host_ptr: false,
            events: false,
            mmap_support: true,
            fast_async_host_input: false,
        };
        for simple_dev in &self.simple_devs {
            let tmp = super::stream::dev_get_props(simple_dev);
            props.caps.async_ = props.caps.async_ && tmp.caps.async_;
            props.caps.host_buffer = props.caps.host_buffer && tmp.caps.host_buffer;
            props.caps.buffer_from_host_ptr =
                props.caps.buffer_from_host_ptr && tmp.caps.buffer_from_host_ptr;
            props.caps.events = props.caps.events && tmp.caps.events;
            props.caps.mmap_support = props.caps.mmap_support && tmp.caps.mmap_support;
        }
    }

    fn init_backend(&self, dev: &DeviceRef, _params: Option<&str>) -> Result<BackendRef> {
        Ok(Rc::new(RefCell::new(super::stream::Backend {
            guid: 2,
            iface: Rc::new(MetaBackend::new(dev)),
            device: Some(Rc::clone(dev)),
        })))
    }

    fn buffer_type(&self, dev: &DeviceRef) -> BuftRef {
        meta_buffer_type(dev)
    }

    fn host_buffer_type(&self, _dev: &DeviceRef) -> Option<BuftRef> {
        let mut host_buft: Option<BuftRef> = None;
        for dev in &self.simple_devs {
            let buft = match super::stream::dev_host_buffer_type(dev) {
                Some(b) => b,
                None => return None,
            };
            match &host_buft {
                None => host_buft = Some(buft),
                Some(existing) if !Rc::ptr_eq(existing, &buft) => return None,
                _ => {}
            }
        }
        host_buft
    }

    fn supports_op(&self, _dev: &DeviceRef, op: &TensorRef) -> bool {
        if flash_attn_ext_has_indexed_bias(&op.borrow()) {
            return false;
        }
        self.simple_devs
            .iter()
            .all(|dev| super::stream::dev_supports_op(dev, op))
    }

    fn supports_buft(&self, _dev: &DeviceRef, buft: &BuftRef) -> bool {
        let dev_buft = buffer::buft_get_device(buft);
        let dev_buft = match dev_buft {
            Some(d) if dev_is_meta(&d) => d,
            _ => return false,
        };
        let other = dev_buft.borrow();
        let other_ctx = match other.iface.as_any().downcast_ref::<MetaDevice>() {
            Some(c) => c,
            None => return false,
        };
        same_devices(&self.simple_devs, &other_ctx.simple_devs)
    }
}

/// `ggml_backend_meta_buffer_type_context`.
pub struct MetaBufferType {
    pub simple_bufts: Vec<BuftRef>,
    pub name: String,
}

impl AsAny for MetaBufferType {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `ggml_backend_meta_device_get_buffer_type`, cached per meta device.
pub fn meta_buffer_type(dev: &DeviceRef) -> BuftRef {
    let key = Rc::as_ptr(dev) as usize;
    let existing = META_BUFT_BY_DEV.with(|cache| {
        cache
            .borrow()
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, b)| Rc::clone(b))
    });
    if let Some(buft) = existing {
        return buft;
    }

    let simple_bufts = (0..meta_dev_n_devs(dev))
        .filter_map(|i| {
            let simple = meta_dev_simple_dev(dev, i)?;
            super::stream::dev_buffer_type(&simple).ok()
        })
        .collect::<Vec<_>>();
    let mut name = String::from("Meta(");
    for (i, buft) in simple_bufts.iter().enumerate() {
        if i > 0 {
            name.push(',');
        }
        name.push_str(&buffer::buft_name(buft));
    }
    name.push(')');

    let buft = Rc::new(BufferType {
        iface: Rc::new(MetaBufferType { simple_bufts, name }),
        device: RefCell::new(Some(Rc::downgrade(dev))),
    });
    META_BUFT_BY_DEV.with(|cache| cache.borrow_mut().push((key, Rc::clone(&buft))));
    buft
}

impl BufferTypeInterface for MetaBufferType {
    fn name(&self, _buft: &BuftRef) -> String {
        self.name.clone()
    }

    fn alloc_buffer(&self, buft: &BuftRef, size: usize) -> Result<BufferRef> {
        meta_buffer_alloc(buft, size)
    }

    fn alignment(&self, buft: &BuftRef) -> usize {
        let mut max_alignment = 1usize;
        for buft in &self.simple_bufts {
            let alignment = buffer::buft_get_alignment(buft);
            max_alignment = max_alignment.max(alignment);
        }
        let _ = buft;
        max_alignment
    }

    fn max_size(&self, _buft: &BuftRef) -> usize {
        let mut max_size = usize::MAX;
        for buft in &self.simple_bufts {
            max_size = max_size.min(buffer::buft_get_max_size(buft));
        }
        max_size
    }

    fn alloc_size(&self, _buft: &BuftRef, tensor: &TensorRef) -> usize {
        let mut max_alloc_size = 0usize;
        for buft in &self.simple_bufts {
            max_alloc_size = max_alloc_size.max(buffer::buft_get_alloc_size(buft, tensor));
        }
        max_alloc_size
    }

    fn is_host(&self, _buft: &BuftRef) -> bool {
        self.simple_bufts
            .iter()
            .all(|b| buffer::buft_is_host(b))
    }
}

/// `ggml_backend_meta_buft_n_bufts`.
pub fn meta_buft_n_bufts(buft: &BuftRef) -> usize {
    buft.iface
        .as_any().downcast_ref::<MetaBufferType>()
        .map(|c| c.simple_bufts.len())
        .unwrap_or(0)
}

/// `ggml_backend_meta_buft_simple_buft`.
pub fn meta_buft_simple_buft(buft: &BuftRef, index: usize) -> Option<BuftRef> {
    buft.iface
        .as_any().downcast_ref::<MetaBufferType>()
        .and_then(|c| c.simple_bufts.get(index).cloned())
}

/// Meta buffer interface. Data access uses the tensor's byte storage; the
/// per-device split is materialized by the compute module.
pub struct MetaBufferInterface {
    pub simple_buffers: Vec<BufferRef>,
    pub get_split_state: Option<GetSplitState>,
}

impl AsAny for MetaBufferInterface {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl BackendBufferInterface for MetaBufferInterface {
    fn get_base(&self, _buffer: &BufferRef) -> Option<Rc<RefCell<Vec<u8>>>> {
        None
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

    fn memset_handled(&self, _buffer: &BufferRef) -> bool {
        true
    }

    fn memset_tensor(&self, _buffer: &BufferRef, tensor: &TensorRef, value: u8, offset: usize, size: usize) {
        if let Some(data) = data_location(tensor) {
            let start = data.offset + offset;
            data.storage.borrow_mut()[start..start + size].fill(value);
        }
    }

    fn clear(&self, _buffer: &BufferRef, value: u8) {
        for b in &self.simple_buffers {
            buffer::buffer_clear(b, value);
        }
    }

    fn reset(&self, _buffer: &BufferRef) {
        for b in &self.simple_buffers {
            buffer::buffer_reset(b);
        }
    }
}

/// `ggml_backend_meta_buffer_type_alloc_buffer`.
pub fn meta_buffer_alloc(buft: &BuftRef, size: usize) -> Result<BufferRef> {
    let ctx = buft.iface.as_any().downcast_ref::<MetaBufferType>()
        .ok_or_else(|| Error::new("meta_buffer_alloc: buffer type is not meta"))?;
    let mut simple_buffers = Vec::with_capacity(ctx.simple_bufts.len());
    for simple_buft in &ctx.simple_bufts {
        simple_buffers.push(buffer::buft_alloc_buffer(simple_buft, size)?);
    }
    let get_split_state = if let Some(dev) = buffer::buft_get_device(buft) {
        dev.borrow()
            .iface
            .as_any().downcast_ref::<MetaDevice>()
            .and_then(|c| c.get_split_state.clone())
    } else {
        None
    };
    let iface = MetaBufferInterface {
        simple_buffers,
        get_split_state,
    };
    let buffer = buffer::buffer_init_with_storage(
        Rc::clone(buft),
        Rc::new(iface),
        Rc::new(RefCell::new(vec![0u8; size])),
        size,
    );
    buffer.borrow_mut().kind = super::buffer::BufferKind::Meta;
    Ok(buffer)
}

/// `ggml_backend_meta_buffer_n_bufs`.
pub fn meta_buffer_n_bufs(buffer: &BufferRef) -> usize {
    buffer.borrow()
        .iface
        .as_any().downcast_ref::<MetaBufferInterface>()
        .map(|c| c.simple_buffers.len())
        .unwrap_or(0)
}

/// `ggml_backend_meta_buffer_simple_buffer`.
pub fn meta_buffer_simple_buffer(buffer: &BufferRef, index: usize) -> Option<BufferRef> {
    buffer.borrow()
        .iface
        .as_any().downcast_ref::<MetaBufferInterface>()
        .and_then(|c| c.simple_buffers.get(index).cloned())
}

/// `ggml_backend_meta_buffer_set_usage`.
pub fn meta_buffer_set_usage(buffer: &BufferRef, usage: BufferUsage) {
    if let Some(iface) = buffer.borrow().iface.as_any().downcast_ref::<MetaBufferInterface>() {
        for b in &iface.simple_buffers {
            buffer::buffer_set_usage(b, usage);
        }
    }
}

/// `ggml_backend_meta_get_split_state`: host-buffer views are not split.
pub fn meta_get_split_state(tensor: &TensorRef, _assume_sync: bool) -> SplitState {
    let buf = tensor.borrow().buffer.clone();
    let is_meta = buf.as_ref().map(|b| buffer_is_meta(b)).unwrap_or(false);
    if !is_meta {
        let mut state = SplitState::default();
        state.axis = SplitAxis::None;
        state.nr[0] = 1;
        state.n_segments = 1;
        return state;
    }
    let callback = buf
        .and_then(|b| b.borrow().iface.as_any().downcast_ref::<MetaBufferInterface>().and_then(|c| c.get_split_state.clone()));
    callback
        .map(|f| f(tensor))
        .unwrap_or_else(|| SplitState { axis: SplitAxis::Unknown, ..SplitState::default() })
}

/// Meta backend stream.
pub struct MetaBackend {
    pub simple_devs: Vec<DeviceRef>,
}

impl AsAny for MetaBackend {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl MetaBackend {
    fn new(dev: &DeviceRef) -> Self {
        let devs = dev
            .borrow()
            .iface
            .as_any().downcast_ref::<MetaDevice>()
            .map(|c| c.simple_devs.clone())
            .unwrap_or_default();
        Self { simple_devs: devs }
    }
}

impl BackendInterface for MetaBackend {
    fn name(&self, _backend: &BackendRef) -> String {
        "Meta".into()
    }

    fn synchronize(&self, backend: &BackendRef) -> bool {
        let _ = backend;
        true
    }

    fn graph_compute(&self, _backend: &BackendRef, _graph: &GraphRef) -> Status {
        Status::Success
    }

    fn graph_optimize(&self, _backend: &BackendRef, _graph: &GraphRef, _params: &GraphOptimizeParams) -> bool {
        false
    }
}

/// `ggml_backend_meta_n_backends`.
pub fn meta_n_backends(backend: &BackendRef) -> usize {
    backend.borrow()
        .iface
        .as_any().downcast_ref::<MetaBackend>()
        .map(|c| c.simple_devs.len())
        .unwrap_or(0)
}

/// `ggml_backend_meta_simple_backend` needs a concrete backend per device; the
/// meta backend stores devices, so this returns the device at `index`.
pub fn meta_simple_device(backend: &BackendRef, index: usize) -> Option<DeviceRef> {
    backend.borrow()
        .iface
        .as_any().downcast_ref::<MetaBackend>()
        .and_then(|c| c.simple_devs.get(index).cloned())
}

/// `ggml_backend_dev_backend_reg` for a meta device (usually `None`).
pub fn meta_device_reg(dev: &DeviceRef) -> Option<RegRef> {
    super::stream::dev_backend_reg(dev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::stream;

    #[test]
    fn split_axis_names() {
        assert_eq!(SplitAxis::Axis0.name(), "0");
        assert_eq!(SplitAxis::Mirrored.name(), "MIRRORED");
        assert_eq!(SplitAxis::Unknown.name(), "UNKNOWN");
    }

    #[test]
    fn meta_device_composition() {
        let dev = Rc::new(RefCell::new(Device {
            iface: Rc::new(super::super::cpu::CpuDevice),
            reg: None,
        }));
        let meta = meta_device(&[Rc::clone(&dev)], None).unwrap();
        assert!(dev_is_meta(&meta));
        assert_eq!(meta_dev_n_devs(&meta), 1);
        assert_eq!(stream::dev_type(&meta), DevType::Meta);
        assert_eq!(meta_dev_simple_dev(&meta, 0).map(|d| stream::dev_name(&d)), Some("CPU".into()));
    }
}

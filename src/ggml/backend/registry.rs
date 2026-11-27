//! Global backend registry.
//!
//! Replaces the C++ `ggml_backend_registry` singleton with thread-local
//! storage, so no `unsafe` or FFI is needed. Dynamic-library loading
//! (`ggml_backend_load*`) is intentionally not part of this port.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::core::{Error, Result};

use super::stream::{self, BackendRef, DeviceRef, RegRef};
use super::traits::{Proc, RegistryInterface};
use super::types::{BACKEND_API_VERSION, DevType};

/// `struct ggml_backend_reg`.
pub struct BackendRegistry {
    pub api_version: i32,
    pub iface: Rc<dyn RegistryInterface>,
    /// Rust-side replacement for backend-specific exported functions.
    pub procs: HashMap<String, Proc>,
}

impl BackendRegistry {
    pub fn new(iface: Rc<dyn RegistryInterface>) -> RegRef {
        Rc::new(RefCell::new(Self {
            api_version: BACKEND_API_VERSION,
            iface,
            procs: HashMap::new(),
        }))
    }
}

thread_local! {
    static REGISTRIES: RefCell<Vec<RegRef>> = RefCell::new(Vec::new());
    static DEVICES: RefCell<Vec<DeviceRef>> = RefCell::new(Vec::new());
}

/// Register a named extension proc on a registry.
pub fn register_proc(reg: &RegRef, name: impl Into<String>, proc: Proc) {
    reg.borrow_mut().procs.insert(name.into(), proc);
}

fn is_registered_device(dev: &DeviceRef) -> bool {
    DEVICES.with(|devices| devices.borrow().iter().any(|d| Rc::ptr_eq(d, dev)))
}

fn add_device(dev: &DeviceRef, reg: Option<&RegRef>) {
    if is_registered_device(dev) {
        return;
    }
    if let Some(reg) = reg {
        dev.borrow_mut().reg = Some(Rc::downgrade(reg));
    }
    DEVICES.with(|devices| devices.borrow_mut().push(Rc::clone(dev)));
}

/// `ggml_backend_register`: register a registry and all of its devices.
pub fn register(reg: RegRef) {
    REGISTRIES.with(|regs| {
        if regs.borrow().iter().any(|r| Rc::ptr_eq(r, &reg)) {
            return;
        }
        regs.borrow_mut().push(Rc::clone(&reg));
    });
    let count = stream::reg_dev_count(&reg);
    for i in 0..count {
        if let Some(dev) = stream::reg_dev_get(&reg, i) {
            add_device(&dev, Some(&reg));
        }
    }
}

/// `ggml_backend_device_register`.
pub fn device_register(dev: DeviceRef) {
    add_device(&dev, None);
}

/// `ggml_backend_reg_count`.
pub fn reg_count() -> usize {
    REGISTRIES.with(|regs| regs.borrow().len())
}

/// `ggml_backend_reg_get`.
pub fn reg_get(index: usize) -> Option<RegRef> {
    REGISTRIES.with(|regs| regs.borrow().get(index).cloned())
}

/// `ggml_backend_reg_by_name`.
pub fn reg_by_name(name: &str) -> Option<RegRef> {
    REGISTRIES.with(|regs| {
        regs.borrow()
            .iter()
            .find(|reg| stream::reg_name(reg) == name)
            .cloned()
    })
}

/// `ggml_backend_dev_count`.
pub fn dev_count() -> usize {
    DEVICES.with(|devs| devs.borrow().len())
}

/// `ggml_backend_dev_get`.
pub fn dev_get(index: usize) -> Option<DeviceRef> {
    DEVICES.with(|devs| devs.borrow().get(index).cloned())
}

/// `ggml_backend_dev_by_name`.
pub fn dev_by_name(name: &str) -> Option<DeviceRef> {
    DEVICES.with(|devs| {
        devs.borrow()
            .iter()
            .find(|dev| stream::dev_name(dev) == name)
            .cloned()
    })
}

/// `ggml_backend_dev_by_type`: first device with the requested type.
pub fn dev_by_type(type_: DevType) -> Option<DeviceRef> {
    DEVICES.with(|devs| {
        devs.borrow()
            .iter()
            .find(|dev| stream::dev_type(dev) == type_)
            .cloned()
    })
}

/// `ggml_backend_init_by_name`.
pub fn init_by_name(name: &str, params: Option<&str>) -> Result<BackendRef> {
    let dev = dev_by_name(name).ok_or_else(|| Error::new(format!("backend not found: {name}")))?;
    stream::dev_init(&dev, params)
}

/// `ggml_backend_init_by_type`.
pub fn init_by_type(type_: DevType, params: Option<&str>) -> Result<BackendRef> {
    let dev = dev_by_type(type_).ok_or_else(|| Error::new(format!("no backend of type {type_:?}")))?;
    stream::dev_init(&dev, params)
}

/// `ggml_backend_init_best`: GPU first, then CPU.
pub fn init_best() -> Result<BackendRef> {
    if let Some(dev) = dev_by_type(DevType::Gpu) {
        return stream::dev_init(&dev, None);
    }
    let dev = dev_by_type(DevType::Cpu).ok_or_else(|| Error::new("no backend available"))?;
    stream::dev_init(&dev, None)
}

/// Test-only reset of the thread-local registry state.
#[cfg(test)]
pub fn clear() {
    REGISTRIES.with(|r| r.borrow_mut().clear());
    DEVICES.with(|d| d.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumeration_and_init_best() {
        clear();
        let reg = BackendRegistry::new(Rc::new(super::super::cpu::CpuRegistry));
        register(reg);
        assert_eq!(reg_count(), 1);
        assert!(reg_by_name("CPU").is_some());
        assert_eq!(dev_count(), 1);
        assert!(dev_by_type(DevType::Cpu).is_some());
        let backend = init_best().unwrap();
        assert_eq!(stream::backend_name(&backend), "CPU");
        clear();
    }
}

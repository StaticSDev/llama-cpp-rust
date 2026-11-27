//! Constants and plain data types of the ggml backend interface.
//!
//! Mirrors the enums and property structs from `ggml-backend.h` and the
//! constants from `ggml-backend-impl.h` / `ggml-impl.h` / `ggml.h`.

use crate::gguf::GGML_MAX_DIMS;

/// `GGML_BACKEND_API_VERSION` from `ggml-backend-impl.h`.
pub const BACKEND_API_VERSION: i32 = 3;

/// `GGML_BACKEND_META_MAX_DEVICES` from `ggml-backend.h`.
pub const META_MAX_DEVICES: usize = 16;

/// `GGML_SCHED_MAX_BACKENDS` from `ggml-backend.cpp`.
pub const SCHED_MAX_BACKENDS: usize = 16;
/// `GGML_SCHED_MAX_SPLIT_INPUTS` from `ggml-backend.cpp`.
pub const SCHED_MAX_SPLIT_INPUTS: usize = 30;
/// `GGML_SCHED_MAX_COPIES` from `ggml-backend.cpp`.
pub const SCHED_MAX_COPIES: usize = 4;

/// `TENSOR_ALIGNMENT` from `ggml-impl.h`.
pub const TENSOR_ALIGNMENT: usize = 32;
/// `GGML_MAX_SRC` from `ggml.h`.
pub const MAX_SRC: usize = 10;
/// `GGML_DEFAULT_GRAPH_SIZE` from `ggml.h`.
pub const DEFAULT_GRAPH_SIZE: usize = 2048;

/// `enum ggml_backend_buffer_usage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferUsage {
    Any = 0,
    Weights = 1,
    Compute = 2,
}

impl Default for BufferUsage {
    fn default() -> Self {
        Self::Any
    }
}

/// `enum ggml_status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Success = 0,
    Failed = 1,
    AllocFailed = 2,
}

/// `enum ggml_backend_dev_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DevType {
    Cpu = 0,
    Gpu = 1,
    Igpu = 2,
    Accel = 3,
    Meta = 4,
}

/// `struct ggml_backend_dev_caps`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DevCaps {
    pub async_: bool,
    pub host_buffer: bool,
    pub buffer_from_host_ptr: bool,
    pub events: bool,
    pub mmap_support: bool,
    pub fast_async_host_input: bool,
}

/// `struct ggml_backend_dev_props`.
///
/// C strings become owned `String`; a missing device id is `None`.
#[derive(Clone, Debug, Default)]
pub struct DevProps {
    pub name: String,
    pub description: String,
    pub memory_free: usize,
    pub memory_total: usize,
    pub type_: DevType,
    pub device_id: Option<String>,
    pub caps: DevCaps,
}

impl Default for DevType {
    fn default() -> Self {
        Self::Cpu
    }
}

/// `enum ggml_backend_meta_split_axis`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitAxis {
    Axis0 = 0,
    Axis1 = 1,
    Axis2 = 2,
    Axis3 = 3,
    Mirrored = 10,
    Partial = 11,
    // internal bookkeeping only
    None = 98,
    Unknown = 99,
}

impl SplitAxis {
    /// `ggml_backend_meta_split_axis_name`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Axis0 => "0",
            Self::Axis1 => "1",
            Self::Axis2 => "2",
            Self::Axis3 => "3",
            Self::Mirrored => "MIRRORED",
            Self::Partial => "PARTIAL",
            Self::None => "NONE",
            Self::Unknown => "UNKNOWN",
        }
    }
}

/// `ggml_backend_meta_split_axis_name` free-function form.
pub fn split_axis_name(axis: SplitAxis) -> &'static str {
    axis.name()
}

/// `struct ggml_backend_meta_split_state`.
///
/// `ne` has one entry per segment and device; `nr` has one entry per segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitState {
    pub axis: SplitAxis,
    pub ne: [i64; 16 * META_MAX_DEVICES],
    pub nr: [u32; 16],
    pub n_segments: u32,
}

impl Default for SplitState {
    fn default() -> Self {
        Self {
            axis: SplitAxis::None,
            ne: [0; 16 * META_MAX_DEVICES],
            nr: [0; 16],
            n_segments: 0,
        }
    }
}

impl SplitState {
    /// Number of devices a tensor is split across.
    pub fn n_devices(&self) -> usize {
        match self.axis {
            SplitAxis::Mirrored | SplitAxis::Partial | SplitAxis::None | SplitAxis::Unknown => 1,
            _ => 2,
        }
    }

    /// Size of one segment on one device, matching the C++ segment layout:
    /// `[seg0_dev0_r0, seg0_dev1_r0, seg0_dev0_r1, seg0_dev1_r1, seg1_dev0_r0, ...]`.
    pub fn ne(&self, segment: usize, device: usize, repeat: usize) -> i64 {
        let stride = self.n_devices().max(1);
        let mut offset = 0usize;
        for s in 0..segment.min(self.n_segments as usize) {
            offset += self.nr.get(s).copied().unwrap_or(0) as usize * stride;
        }
        let idx = offset + repeat.saturating_mul(stride) + device.min(stride - 1);
        self.ne.get(idx).copied().unwrap_or(0)
    }
}

/// `struct ggml_backend_feature`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Feature {
    pub name: String,
    pub value: String,
}

/// `GGML_MAX_DIMS` re-exported for callers building tensor shapes.
pub const MAX_DIMS: usize = GGML_MAX_DIMS;

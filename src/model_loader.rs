//! Rust port of `llama-model-loader.h` / `llama-model-loader.cpp`.
//!
//! The C++ loader owns GGUF metadata, the tensor index across split files,
//! buffer-type selection and the actual weight reads. Backend buffers, mmap
//! and async uploads have no counterpart in this dry port yet, so they are
//! represented by small std-only stand-ins with the same control flow.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use crate::arch::{
    tensor_info_for, Arch, Kv as ArchKv, LlmKv, LlmTnImpl, Op, Tensor as ArchTensor, TensorInfo,
    TensorLayer,
};
use crate::core::{Error, Result};
use crate::gguf::{
    ggml_blck_size, ggml_nbytes, ggml_type_size, gguf_find_key, gguf_find_tensor,
    gguf_get_data_offset, gguf_get_kv_type, gguf_get_n_kv, gguf_get_n_tensors,
    gguf_get_tensor_name, gguf_get_tensor_offset, gguf_get_tensor_type, gguf_init_from_file,
    gguf_init_from_file_ptr, gguf_type_name, gguf_type_size, GgmlContext, GgmlTensor, GgmlType,
    GgufContext, GgufInitParams, GgufKv, GgufType, GgufValue, GGML_MAX_DIMS,
    GGUF_DEFAULT_ALIGNMENT,
};
use crate::hparams::Hparams;

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;
const GIB: usize = 1024 * MIB;

/// `llama_fver`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FileVersion {
    V1 = 1,
    V2 = 2,
    V3 = 3,
}

impl FileVersion {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => FileVersion::V1,
            2 => FileVersion::V2,
            _ => FileVersion::V3,
        }
    }
}

pub fn llama_file_version_name(version: FileVersion) -> &'static str {
    match version {
        FileVersion::V1 => "GGUF V1 (support until nov 2023)",
        FileVersion::V2 => "GGUF V2",
        FileVersion::V3 => "GGUF V3 (latest)",
    }
}

/// `llama_ftype`. The guessed bit is kept in the value, as in C++.
pub type Ftype = u32;

pub mod ftype {
    pub const ALL_F32: u32 = 0;
    pub const MOSTLY_F16: u32 = 1;
    pub const MOSTLY_Q4_0: u32 = 2;
    pub const MOSTLY_Q4_1: u32 = 3;
    pub const MOSTLY_Q8_0: u32 = 7;
    pub const MOSTLY_Q5_0: u32 = 8;
    pub const MOSTLY_Q5_1: u32 = 9;
    pub const MOSTLY_Q2_K: u32 = 10;
    pub const MOSTLY_Q3_K_S: u32 = 11;
    pub const MOSTLY_Q3_K_M: u32 = 12;
    pub const MOSTLY_Q3_K_L: u32 = 13;
    pub const MOSTLY_Q4_K_S: u32 = 14;
    pub const MOSTLY_Q4_K_M: u32 = 15;
    pub const MOSTLY_Q5_K_S: u32 = 16;
    pub const MOSTLY_Q5_K_M: u32 = 17;
    pub const MOSTLY_Q6_K: u32 = 18;
    pub const MOSTLY_IQ2_XXS: u32 = 19;
    pub const MOSTLY_IQ2_XS: u32 = 20;
    pub const MOSTLY_Q2_K_S: u32 = 21;
    pub const MOSTLY_IQ3_XS: u32 = 22;
    pub const MOSTLY_IQ3_XXS: u32 = 23;
    pub const MOSTLY_IQ1_S: u32 = 24;
    pub const MOSTLY_IQ4_NL: u32 = 25;
    pub const MOSTLY_IQ3_S: u32 = 26;
    pub const MOSTLY_IQ3_M: u32 = 27;
    pub const MOSTLY_IQ2_S: u32 = 28;
    pub const MOSTLY_IQ2_M: u32 = 29;
    pub const MOSTLY_IQ4_XS: u32 = 30;
    pub const MOSTLY_IQ1_M: u32 = 31;
    pub const MOSTLY_BF16: u32 = 32;
    pub const MOSTLY_TQ1_0: u32 = 36;
    pub const MOSTLY_TQ2_0: u32 = 37;
    pub const MOSTLY_MXFP4_MOE: u32 = 38;
    pub const MOSTLY_NVFP4: u32 = 39;
    pub const MOSTLY_Q1_0: u32 = 40;
    pub const MOSTLY_Q2_0: u32 = 41;
    pub const MOSTLY_F8_E4M3: u32 = 42;
    pub const GUESSED: u32 = 1024;
}

fn ftype_base_name(bits: u32) -> &'static str {
    match bits {
        ftype::ALL_F32 => "all F32",
        ftype::MOSTLY_F16 => "F16",
        ftype::MOSTLY_BF16 => "BF16",
        ftype::MOSTLY_Q1_0 => "Q1_0",
        ftype::MOSTLY_Q2_0 => "Q2_0",
        ftype::MOSTLY_F8_E4M3 => "F8_E4M3",
        ftype::MOSTLY_Q4_0 => "Q4_0",
        ftype::MOSTLY_Q4_1 => "Q4_1",
        ftype::MOSTLY_Q5_0 => "Q5_0",
        ftype::MOSTLY_Q5_1 => "Q5_1",
        ftype::MOSTLY_Q8_0 => "Q8_0",
        ftype::MOSTLY_MXFP4_MOE => "MXFP4 MoE",
        ftype::MOSTLY_NVFP4 => "NVFP4",
        ftype::MOSTLY_Q2_K => "Q2_K - Medium",
        ftype::MOSTLY_Q2_K_S => "Q2_K - Small",
        ftype::MOSTLY_Q3_K_S => "Q3_K - Small",
        ftype::MOSTLY_Q3_K_M => "Q3_K - Medium",
        ftype::MOSTLY_Q3_K_L => "Q3_K - Large",
        ftype::MOSTLY_Q4_K_S => "Q4_K - Small",
        ftype::MOSTLY_Q4_K_M => "Q4_K - Medium",
        ftype::MOSTLY_Q5_K_S => "Q5_K - Small",
        ftype::MOSTLY_Q5_K_M => "Q5_K - Medium",
        ftype::MOSTLY_Q6_K => "Q6_K",
        ftype::MOSTLY_TQ1_0 => "TQ1_0 - 1.69 bpw ternary",
        ftype::MOSTLY_TQ2_0 => "TQ2_0 - 2.06 bpw ternary",
        ftype::MOSTLY_IQ2_XXS => "IQ2_XXS - 2.0625 bpw",
        ftype::MOSTLY_IQ2_XS => "IQ2_XS - 2.3125 bpw",
        ftype::MOSTLY_IQ2_S => "IQ2_S - 2.5 bpw",
        ftype::MOSTLY_IQ2_M => "IQ2_M - 2.7 bpw",
        ftype::MOSTLY_IQ3_XS => "IQ3_XS - 3.3 bpw",
        ftype::MOSTLY_IQ3_XXS => "IQ3_XXS - 3.0625 bpw",
        ftype::MOSTLY_IQ1_S => "IQ1_S - 1.5625 bpw",
        ftype::MOSTLY_IQ1_M => "IQ1_M - 1.75 bpw",
        ftype::MOSTLY_IQ4_NL => "IQ4_NL - 4.5 bpw",
        ftype::MOSTLY_IQ4_XS => "IQ4_XS - 4.25 bpw",
        ftype::MOSTLY_IQ3_S => "IQ3_S - 3.4375 bpw",
        ftype::MOSTLY_IQ3_M => "IQ3_S mix - 3.66 bpw",
        _ => "unknown, may not work",
    }
}

/// `llama_ftype_name`: includes the "(guessed) " prefix when the bit is set.
pub fn ftype_name(ftype: Ftype) -> String {
    let base = ftype_base_name(ftype & !ftype::GUESSED);
    if ftype & ftype::GUESSED != 0 {
        format!("(guessed) {}", base)
    } else {
        base.to_string()
    }
}

/// `llama_load_mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum LoadMode {
    Auto = -1,
    None = 0,
    Mmap = 1,
    Mlock = 2,
    MmapMlock = 3,
    DirectIo = 4,
}

/// `llama_model_kv_override_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvOverrideType {
    Bool,
    Int,
    Float,
    Str,
}

impl KvOverrideType {
    fn name(self) -> &'static str {
        match self {
            KvOverrideType::Bool => "bool",
            KvOverrideType::Int => "int",
            KvOverrideType::Float => "float",
            KvOverrideType::Str => "str",
        }
    }
}

/// `llama_model_kv_override`.
#[derive(Clone, Debug)]
pub struct KvOverride {
    pub key: String,
    pub tag: KvOverrideType,
    pub val_bool: bool,
    pub val_i64: i64,
    pub val_f64: f64,
    pub val_str: String,
}

impl KvOverride {
    pub fn bool_(key: impl Into<String>, value: bool) -> Self {
        Self {
            key: key.into(),
            tag: KvOverrideType::Bool,
            val_bool: value,
            val_i64: 0,
            val_f64: 0.0,
            val_str: String::new(),
        }
    }
    pub fn int(key: impl Into<String>, value: i64) -> Self {
        Self {
            key: key.into(),
            tag: KvOverrideType::Int,
            val_bool: false,
            val_i64: value,
            val_f64: 0.0,
            val_str: String::new(),
        }
    }
    pub fn float(key: impl Into<String>, value: f64) -> Self {
        Self {
            key: key.into(),
            tag: KvOverrideType::Float,
            val_bool: false,
            val_i64: 0,
            val_f64: value,
            val_str: String::new(),
        }
    }
    pub fn str(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            tag: KvOverrideType::Str,
            val_bool: false,
            val_i64: 0,
            val_f64: 0.0,
            val_str: value.into(),
        }
    }
}

/// Stand-ins for `ggml_backend_dev_t` / `ggml_backend_buffer_type_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Device {
    Cpu,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BufferType {
    Cpu,
    Host,
    Device,
}

impl BufferType {
    pub fn name(self) -> &'static str {
        match self {
            BufferType::Cpu => "CPU",
            BufferType::Host => "HOST",
            BufferType::Device => "DEVICE",
        }
    }

    fn device(self) -> Device {
        match self {
            BufferType::Cpu | BufferType::Host => Device::Cpu,
            BufferType::Device => Device::Other,
        }
    }

    fn supports_op(self, op: WeightOp) -> bool {
        match self {
            BufferType::Cpu | BufferType::Host => true,
            BufferType::Device => match op {
                WeightOp::AddId => true,
                WeightOp::Op(op) => !matches!(op, Op::None | Op::RwkvWkv6),
            },
        }
    }
}

/// Op used for buffer selection; `AddId` is the bias form of `MulMatId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightOp {
    Op(Op),
    AddId,
}

/// `buft_list_t`.
pub type BuftList = Vec<(Device, BufferType)>;

/// `llama_model_tensor_buft_override`.
#[derive(Clone, Debug)]
pub struct TensorBuftOverride {
    pub pattern: Option<String>,
    pub buft: BufferType,
}

/// A tensor weight: source file, data offset and metadata tensor.
#[derive(Clone, Debug)]
pub struct TensorWeight {
    pub idx: u16,
    pub offs: usize,
    pub tensor: GgmlTensor,
}

impl TensorWeight {
    pub fn new(
        file_size: usize,
        idx: u16,
        gguf_ctx: &GgufContext,
        tensor: &GgmlTensor,
    ) -> Result<Self> {
        let tensor_idx = gguf_find_tensor(gguf_ctx, &tensor.name);
        if tensor_idx < 0 {
            return Err(Error::new(format!(
                "tensor '{}' not found in the model",
                tensor.name
            )));
        }
        let offs = gguf_get_data_offset(gguf_ctx)
            .checked_add(gguf_get_tensor_offset(gguf_ctx, tensor_idx))
            .ok_or_else(|| Error::new(format!("tensor '{}' data offset overflow", tensor.name)))?;
        let nbytes = ggml_nbytes(tensor);
        let end = offs.checked_add(nbytes).ok_or_else(|| {
            Error::new(format!(
                "tensor '{}' data is not within the file bounds, model is corrupted or incomplete",
                tensor.name
            ))
        })?;
        if end > file_size {
            return Err(Error::new(format!(
                "tensor '{}' data is not within the file bounds, model is corrupted or incomplete",
                tensor.name
            )));
        }
        Ok(Self {
            idx,
            offs,
            tensor: tensor.clone(),
        })
    }
}

/// Sort key implementing `weight_name_comparer`: layer first, then name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WeightKey {
    layer: i32,
    name: String,
}

/// Equivalent of `sscanf(name, "blk.%d.", &layer)`, -1 when no match.
pub fn parse_blk_layer(name: &str) -> i32 {
    let rest = match name.strip_prefix("blk.") {
        Some(r) => r,
        None => return -1,
    };
    let mut end = 0usize;
    let bytes = rest.as_bytes();
    if end < bytes.len() && (bytes[end] == b'-' || bytes[end] == b'+') {
        end += 1;
    }
    let digits_start = end;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == digits_start || end >= bytes.len() || bytes[end] != b'.' {
        return -1;
    }
    rest[..end].parse::<i32>().unwrap_or(-1)
}

impl WeightKey {
    pub fn new(name: &str) -> Self {
        Self {
            layer: parse_blk_layer(name),
            name: name.to_string(),
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// `llama_file` stand-in backed by `std::fs::File`.
pub struct ModelFile {
    pub path: String,
    file: File,
    pub size: usize,
    pub direct_io: bool,
}

impl ModelFile {
    pub fn open(path: &str, direct_io: bool) -> Result<Self> {
        let mut file = File::open(path)
            .map_err(|e| Error::new(format!("failed to open '{}' ({})", path, e)))?;
        let size = file
            .seek(SeekFrom::End(0))
            .map_err(|e| Error::new(e.to_string()))? as usize;
        file.seek(SeekFrom::Start(0))
            .map_err(|e| Error::new(e.to_string()))?;
        Ok(Self {
            path: path.to_string(),
            file,
            size,
            direct_io,
        })
    }

    pub fn from_file(file: File, direct_io: bool) -> Result<Self> {
        let mut reader = file.try_clone().map_err(|e| Error::new(e.to_string()))?;
        let size = reader
            .seek(SeekFrom::End(0))
            .map_err(|e| Error::new(e.to_string()))? as usize;
        reader
            .seek(SeekFrom::Start(0))
            .map_err(|e| Error::new(e.to_string()))?;
        Ok(Self {
            path: String::new(),
            file: reader,
            size,
            direct_io,
        })
    }

    pub fn read_at(&mut self, offs: usize, size: usize) -> Result<Vec<u8>> {
        if offs.checked_add(size).map_or(true, |end| end > self.size) {
            return Err(Error::new(format!(
                "read of {} bytes at {} exceeds file size {}",
                size, offs, self.size
            )));
        }
        self.file
            .seek(SeekFrom::Start(offs as u64))
            .map_err(|e| Error::new(e.to_string()))?;
        let mut buf = vec![0u8; size];
        self.file
            .read_exact(&mut buf)
            .map_err(|e| Error::new(e.to_string()))?;
        Ok(buf)
    }

    fn read_all(&mut self) -> Result<Vec<u8>> {
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| Error::new(e.to_string()))?;
        let mut buf = vec![0u8; self.size];
        self.file
            .read_exact(&mut buf)
            .map_err(|e| Error::new(e.to_string()))?;
        Ok(buf)
    }

    /// Direct I/O alignment is not available through std; keep the C++ default.
    pub fn read_alignment(&self) -> usize {
        if self.direct_io {
            4096
        } else {
            1
        }
    }
}

/// Byte range within a mapped file: `[first, last)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MappingRange {
    pub first: usize,
    pub last: usize,
}

/// `llama_mmap` stand-in: file bytes held in memory.
#[derive(Debug)]
pub struct Mapping {
    data: Vec<u8>,
}

impl Mapping {
    pub fn new(file: &mut ModelFile) -> Result<Self> {
        Ok(Self {
            data: file.read_all()?,
        })
    }
    pub fn size(&self) -> usize {
        self.data.len()
    }
    pub fn addr(&self) -> &[u8] {
        &self.data
    }
    pub fn unmap_fragment(&mut self, first: usize, last: usize) {
        let first = first.min(self.data.len());
        let last = last.min(self.data.len()).max(first);
        self.data[first..last].fill(0);
    }
}

/// `llama_mlock` stand-in: records which mappings were requested to be locked.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mlock {
    pub locked: bool,
}

impl Mlock {
    pub fn init(&mut self) {
        self.locked = true;
    }
}

pub type Mlocks = Vec<Mlock>;

/// `llama_model_loader::lazy_read`.
#[derive(Debug)]
pub struct LazyRead {
    pub mode: LazyMode,
    ranges: BTreeMap<u32, Vec<MappingRange>>,
    tensors: BTreeSet<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LazyMode {
    Off,
    On,
    Auto,
}

impl Default for LazyRead {
    fn default() -> Self {
        Self {
            mode: LazyMode::Off,
            ranges: BTreeMap::new(),
            tensors: BTreeSet::new(),
        }
    }
}

/// Mmap is emulated with owned file bytes, so lazy reads are available.
pub const MMAP_SUPPORTED: bool = true;

impl LazyRead {
    pub fn add(&mut self, name: &str, t: &GgmlTensor, w: Option<&TensorWeight>) -> bool {
        if self.mode == LazyMode::Off {
            return false;
        }
        const AUTO_MIN_SIZE: usize = 4 * GIB;
        if self.mode != LazyMode::On && ggml_nbytes(t) <= AUTO_MIN_SIZE {
            return false;
        }
        if !MMAP_SUPPORTED {
            return false;
        }
        if let Some(w) = w {
            let nbytes = ggml_nbytes(t);
            self.ranges
                .entry(w.idx as u32)
                .or_default()
                .push(MappingRange {
                    first: w.offs,
                    last: w.offs.saturating_add(nbytes),
                });
            self.tensors.insert(name.to_string());
        }
        true
    }

    pub fn any(&self) -> bool {
        !self.ranges.is_empty()
    }
    pub fn has(&self, t: &GgmlTensor) -> bool {
        self.tensors.contains(&t.name)
    }
    pub fn has_name(&self, name: &str) -> bool {
        self.tensors.contains(name)
    }
    pub fn for_file(&self, idx: u32) -> &[MappingRange] {
        self.ranges.get(&idx).map_or(&[], |v| v.as_slice())
    }
    pub fn buft() -> BufferType {
        BufferType::Cpu
    }
}

/// `TENSOR_*` flags from `llama-model-loader.h`.
pub const TENSOR_NOT_REQUIRED: u32 = 1 << 0;
pub const TENSOR_DUPLICATED: u32 = 1 << 1;
pub const TENSOR_SKIP: u32 = 1 << 2;
pub const TENSOR_SKIP_IF_VIRTUAL: u32 = 1 << 3;
pub const TENSOR_ALLOW_RESHAPE: u32 = 1 << 4;
pub const TENSOR_READ_LAZY: u32 = 1 << 5;

/// One context per buffer type; lazy tensors get their own context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CtxKey {
    pub buft: BufferType,
    pub lazy: bool,
}

#[derive(Debug, Default)]
struct CtxEntry {
    tensors: Vec<GgmlTensor>,
}

pub type SetTensorData = Box<dyn FnMut(&mut GgmlTensor)>;
pub type ProgressCallback = dyn FnMut(f32) -> bool;

/// `llama_model_loader`.
pub struct ModelLoader {
    pub n_kv: i32,
    pub n_tensors: usize,
    pub n_created: usize,

    pub n_elements: u64,
    pub n_bytes: usize,

    pub use_mmap: bool,
    pub use_direct_io: bool,
    pub check_tensors: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,

    pub lazy: LazyRead,

    pub files: Vec<ModelFile>,
    pub ftype: Ftype,
    pub fver: FileVersion,

    pub mappings: Vec<Mapping>,

    pub weights_map: BTreeMap<WeightKey, TensorWeight>,
    kv_overrides: HashMap<String, KvOverride>,
    tensor_buft_overrides: Vec<TensorBuftOverride>,

    pub metadata: GgufContext,
    set_tensor_data: Option<SetTensorData>,
    pub contexts: Vec<GgmlContext>,

    pub arch_name: String,
    llm_kv: LlmKv,

    pub size_done: usize,
    pub size_data: usize,
    pub mmaps_used: Vec<(usize, usize)>,

    ctx_map: BTreeMap<CtxKey, CtxEntry>,

    pub n_tensors_moved: usize,
    pub first_tensor_moved_name: String,
    pub first_tensor_moved_type_name: String,
    pub first_moved_from_buft: Option<BufferType>,
    pub first_moved_to_buft: Option<BufferType>,

    trace: i32,
    log: RefCell<Vec<String>>,
}

fn ggml_nelements(t: &GgmlTensor) -> u64 {
    t.ne.iter().fold(1i128, |acc, &ne| acc * ne as i128).max(0) as u64
}

fn ggml_row_size(ty: GgmlType, ne0: i64) -> usize {
    ggml_type_size(ty) * (ne0.max(0) as usize / ggml_blck_size(ty).max(1) as usize)
}

fn format_shape(ne: &[i64]) -> String {
    if ne.is_empty() {
        return String::new();
    }
    let mut out = format!("{:6}", ne[0]);
    for &n in &ne[1..] {
        out.push_str(&format!(", {:6}", n));
    }
    out
}

fn format_tensor_shape(t: &GgmlTensor) -> String {
    format_shape(&t.ne)
}

fn basename(path: &str) -> &str {
    let idx = path.rfind(|c| c == '/' || c == '\\').map_or(0, |i| i + 1);
    &path[idx..]
}

/// `llama_split_path`: `"%s-%05d-of-%05d.gguf"` with 1-based split number.
pub fn split_path(prefix: &str, split_no: u32, split_count: u32) -> String {
    format!("{}-{:05}-of-{:05}.gguf", prefix, split_no + 1, split_count)
}

/// `llama_split_prefix`: strip the exact `<idx+1>-of-<count>` suffix.
pub fn split_prefix(path: &str, split_no: u32, split_count: u32) -> Option<String> {
    let postfix = format!("-{:05}-of-{:05}.gguf", split_no + 1, split_count);
    let prefix = path.strip_suffix(&postfix)?;
    if prefix.is_empty() {
        return None;
    }
    Some(prefix.to_string())
}

fn get_list_splits(path: &str, idx: u32, n_split: u32) -> Result<Vec<String>> {
    let prefix = split_prefix(path, idx, n_split)
        .ok_or_else(|| Error::new(format!("invalid split file name: {}", path)))?;
    if prefix.is_empty() {
        return Err(Error::new(format!("invalid split file: {}", path)));
    }
    Ok((0..n_split)
        .map(|i| split_path(&prefix, i, n_split))
        .collect())
}

fn is_ple_only(ctx: &GgufContext) -> bool {
    gguf_get_n_tensors(ctx) == 1 && gguf_get_tensor_name(ctx, 0) == "per_layer_token_embd.weight"
}

/// Merge sorted, non-overlapping `[first, last)` ranges.
pub fn merge_ranges(mut ranges: Vec<MappingRange>) -> Vec<MappingRange> {
    ranges.sort_by_key(|r| r.first);
    let mut res: Vec<MappingRange> = Vec::new();
    for range in ranges {
        if let Some(last) = res.last_mut() {
            if range.first <= last.last {
                last.last = last.last.max(range.last);
                continue;
            }
        }
        res.push(range);
    }
    res
}

/// Subtract sorted, non-overlapping `sub` from `a`.
pub fn sub_ranges(a: &[MappingRange], sub: &[MappingRange]) -> Vec<MappingRange> {
    let mut res = Vec::new();
    for x in a {
        let mut cur = x.first;
        for y in sub {
            if y.last <= cur || y.first >= x.last {
                continue;
            }
            if y.first > cur {
                res.push(MappingRange {
                    first: cur,
                    last: y.first,
                });
            }
            cur = cur.max(y.last);
        }
        if cur < x.last {
            res.push(MappingRange {
                first: cur,
                last: x.last,
            });
        }
    }
    res
}

/// Small regex-search subset: literals, `.`, classes, escapes, `* + ?`, `^ $`.
/// Used for `llama_model_tensor_buft_override` patterns; std has no regex.
fn regex_search(text: &str, pattern: &str) -> bool {
    #[derive(Clone, Copy)]
    enum Atom {
        Lit(char),
        Any,
        Class(usize, bool), // index into classes, negated
        Start,
        End,
    }
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Quant {
        One,
        Star,
        Plus,
        Opt,
    }

    let chars: Vec<char> = text.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    let mut classes: Vec<Vec<(char, char)>> = Vec::new();
    let mut tokens: Vec<(Atom, Quant)> = Vec::new();
    let mut i = 0usize;

    while i < p.len() {
        let (atom, consumed) = match p[i] {
            '^' => (Atom::Start, 1),
            '$' => (Atom::End, 1),
            '.' => (Atom::Any, 1),
            '[' => {
                let mut j = i + 1;
                let negated = j < p.len() && p[j] == '^';
                if negated {
                    j += 1;
                }
                let mut ranges = Vec::new();
                while j < p.len() && p[j] != ']' {
                    let lo = p[j];
                    if j + 2 < p.len() && p[j + 1] == '-' && p[j + 2] != ']' {
                        ranges.push((lo, p[j + 2]));
                        j += 3;
                    } else {
                        ranges.push((lo, lo));
                        j += 1;
                    }
                }
                if j >= p.len() {
                    return false;
                }
                classes.push(ranges);
                (Atom::Class(classes.len() - 1, negated), j + 1 - i)
            }
            '\\' if i + 1 < p.len() => {
                let c = p[i + 1];
                match c {
                    'd' => {
                        classes.push(vec![('0', '9')]);
                        (Atom::Class(classes.len() - 1, false), 2)
                    }
                    'w' => {
                        classes.push(vec![('a', 'z'), ('A', 'Z'), ('0', '9'), ('_', '_')]);
                        (Atom::Class(classes.len() - 1, false), 2)
                    }
                    's' => {
                        classes.push(vec![(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')]);
                        (Atom::Class(classes.len() - 1, false), 2)
                    }
                    _ => (Atom::Lit(c), 2),
                }
            }
            c => (Atom::Lit(c), 1),
        };
        i += consumed;
        let quant = if i < p.len() {
            match p[i] {
                '*' => {
                    i += 1;
                    Quant::Star
                }
                '+' => {
                    i += 1;
                    Quant::Plus
                }
                '?' => {
                    i += 1;
                    Quant::Opt
                }
                _ => Quant::One,
            }
        } else {
            Quant::One
        };
        tokens.push((atom, quant));
    }

    fn atom_match(atom: Atom, classes: &[Vec<(char, char)>], c: char) -> bool {
        match atom {
            Atom::Lit(x) => x == c,
            Atom::Any => true,
            Atom::Class(idx, negated) => {
                let hit = classes[idx].iter().any(|&(lo, hi)| lo <= c && c <= hi);
                hit != negated
            }
            Atom::Start | Atom::End => false,
        }
    }

    fn go(
        tokens: &[(Atom, Quant)],
        ti: usize,
        chars: &[char],
        si: usize,
        classes: &[Vec<(char, char)>],
    ) -> bool {
        if ti == tokens.len() {
            return true;
        }
        let (atom, quant) = tokens[ti];
        match atom {
            Atom::Start => si == 0 && go(tokens, ti + 1, chars, si, classes),
            Atom::End => si == chars.len() && go(tokens, ti + 1, chars, si, classes),
            _ => {
                let min = if quant == Quant::Plus { 1 } else { 0 };
                let mut count = 0usize;
                let mut pos = si;
                while pos < chars.len() && atom_match(atom, classes, chars[pos]) {
                    pos += 1;
                    count += 1;
                }
                if count < min {
                    return false;
                }
                if quant == Quant::One {
                    return count >= 1 && go(tokens, ti + 1, chars, si + 1, classes);
                }
                let max = if quant == Quant::Opt {
                    count.min(1)
                } else {
                    count
                };
                for n in (min..=max).rev() {
                    if go(tokens, ti + 1, chars, si + n, classes) {
                        return true;
                    }
                }
                false
            }
        }
    }

    (0..=chars.len()).any(|start| go(&tokens, 0, &chars, start, &classes))
}

/// `ggml_validate_row_data` stand-in: size/alignment and finite float checks.
pub fn validate_row_data(ty: GgmlType, data: &[u8], size: usize) -> bool {
    if data.len() < size {
        return false;
    }
    let blck = ggml_blck_size(ty).max(1) as usize;
    if size % blck != 0 {
        return false;
    }
    let slice = &data[..size];
    match ty {
        GgmlType::F32 => slice
            .chunks_exact(4)
            .all(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]).is_finite()),
        GgmlType::F64 => slice.chunks_exact(8).all(|b| {
            f64::from_ne_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]).is_finite()
        }),
        GgmlType::F16 => slice.chunks_exact(2).all(|b| {
            let h = u16::from_ne_bytes([b[0], b[1]]);
            (h & 0x7c00) != 0x7c00
        }),
        GgmlType::Bf16 => slice.chunks_exact(2).all(|b| {
            let h = u16::from_ne_bytes([b[0], b[1]]);
            (h & 0x7f80) != 0x7f80
        }),
        _ => true,
    }
}

/// Scalar GGUF values accepted by `get_key` / `get_arr`.
pub trait GetScalar: Copy + GgufValue {
    fn gt() -> GgufType;
    fn override_tag() -> Option<KvOverrideType>;
    fn from_kv(kv: &GgufKv, i: usize) -> Result<Self>;
    fn from_override(ov: &KvOverride) -> Self;
}

macro_rules! impl_get_scalar {
    ($t:ty, $gt:expr, $tag:expr, $field:ident) => {
        impl GetScalar for $t {
            fn gt() -> GgufType {
                $gt
            }
            fn override_tag() -> Option<KvOverrideType> {
                Some($tag)
            }
            fn from_kv(kv: &GgufKv, i: usize) -> Result<Self> {
                if kv.type_ != $gt {
                    return Err(Error::new(format!(
                        "key has wrong type {:?} but expected {:?}",
                        kv.type_, $gt
                    )));
                }
                let n = std::mem::size_of::<$t>();
                let start = i
                    .checked_mul(n)
                    .ok_or_else(|| Error::new("array index overflow"))?;
                let end = start
                    .checked_add(n)
                    .ok_or_else(|| Error::new("array index overflow"))?;
                if end > kv.data.len() {
                    return Err(Error::new("GGUF value is shorter than its declared type"));
                }
                Ok(<$t as GgufValue>::decode(&kv.data[start..end]))
            }
            fn from_override(ov: &KvOverride) -> Self {
                ov.$field as $t
            }
        }
    };
}

impl_get_scalar!(u8, GgufType::Uint8, KvOverrideType::Int, val_i64);
impl_get_scalar!(u16, GgufType::Uint16, KvOverrideType::Int, val_i64);
impl_get_scalar!(u32, GgufType::Uint32, KvOverrideType::Int, val_i64);
impl_get_scalar!(u64, GgufType::Uint64, KvOverrideType::Int, val_i64);
impl_get_scalar!(i8, GgufType::Int8, KvOverrideType::Int, val_i64);
impl_get_scalar!(i16, GgufType::Int16, KvOverrideType::Int, val_i64);
impl_get_scalar!(i32, GgufType::Int32, KvOverrideType::Int, val_i64);
impl_get_scalar!(i64, GgufType::Int64, KvOverrideType::Int, val_i64);
impl_get_scalar!(f32, GgufType::Float32, KvOverrideType::Float, val_f64);
impl_get_scalar!(f64, GgufType::Float64, KvOverrideType::Float, val_f64);

impl GetScalar for bool {
    fn gt() -> GgufType {
        GgufType::Bool
    }
    fn override_tag() -> Option<KvOverrideType> {
        Some(KvOverrideType::Bool)
    }
    fn from_kv(kv: &GgufKv, i: usize) -> Result<Self> {
        if kv.type_ != GgufType::Bool || i >= kv.data.len() {
            return Err(Error::new("GGUF value is not a bool"));
        }
        Ok(kv.data[i] != 0)
    }
    fn from_override(ov: &KvOverride) -> Self {
        ov.val_bool
    }
}

fn kv_type_name(ty: GgufType) -> &'static str {
    gguf_type_name(ty).unwrap_or("unknown")
}

fn kv_type_string(kv: &GgufKv) -> String {
    if kv.is_array {
        format!("array[{},{}]", kv_type_name(kv.type_), kv.get_ne())
    } else {
        kv_type_name(kv.type_).to_string()
    }
}

fn scalar_to_string(gt: GgufType, data: &[u8]) -> String {
    fn d<T: GgufValue + std::fmt::Display>(data: &[u8]) -> String {
        T::decode(data).to_string()
    }
    match gt {
        GgufType::Uint8 => d::<u8>(data),
        GgufType::Int8 => d::<i8>(data),
        GgufType::Uint16 => d::<u16>(data),
        GgufType::Int16 => d::<i16>(data),
        GgufType::Uint32 => d::<u32>(data),
        GgufType::Int32 => d::<i32>(data),
        GgufType::Float32 => d::<f32>(data),
        GgufType::Uint64 => d::<u64>(data),
        GgufType::Int64 => d::<i64>(data),
        GgufType::Float64 => d::<f64>(data),
        GgufType::Bool => (data[0] != 0).to_string(),
        GgufType::String | GgufType::Array | GgufType::Count => String::new(),
    }
}

fn kv_value_string(kv: &GgufKv, max_len: usize) -> String {
    let raw = if kv.is_array {
        let n = kv.get_ne().min(16);
        let size = gguf_type_size(kv.type_);
        let parts: Vec<String> = if kv.type_ == GgufType::String {
            (0..n).map(|i| kv.get_str(i).to_string()).collect()
        } else if size == 0 {
            Vec::new()
        } else {
            (0..n)
                .map(|i| scalar_to_string(kv.type_, &kv.data[i * size..(i + 1) * size]))
                .collect()
        };
        format!("[{}]", parts.join(", "))
    } else if kv.type_ == GgufType::String {
        kv.get_str(0).to_string()
    } else {
        scalar_to_string(kv.type_, &kv.data)
    };
    let mut value = raw.replace('\n', "\\n");
    if value.len() > max_len {
        let mut end = max_len.saturating_sub(3).min(value.len());
        while end > 0 && !value.is_char_boundary(end) {
            end -= 1;
        }
        value = format!("{}...", &value[..end]);
    }
    value
}

fn ftype_for_type(ty: GgmlType) -> Ftype {
    match ty {
        GgmlType::F32 => ftype::ALL_F32,
        GgmlType::F16 => ftype::MOSTLY_F16,
        GgmlType::Bf16 => ftype::MOSTLY_BF16,
        GgmlType::Q4_0 => ftype::MOSTLY_Q4_0,
        GgmlType::Q4_1 => ftype::MOSTLY_Q4_1,
        GgmlType::Q5_0 => ftype::MOSTLY_Q5_0,
        GgmlType::Q5_1 => ftype::MOSTLY_Q5_1,
        GgmlType::Q8_0 => ftype::MOSTLY_Q8_0,
        GgmlType::Q2K => ftype::MOSTLY_Q2_K,
        GgmlType::Q3K => ftype::MOSTLY_Q3_K_M,
        GgmlType::Q4K => ftype::MOSTLY_Q4_K_M,
        GgmlType::Q5K => ftype::MOSTLY_Q5_K_M,
        GgmlType::Q6K => ftype::MOSTLY_Q6_K,
        GgmlType::Tq1_0 => ftype::MOSTLY_TQ1_0,
        GgmlType::Tq2_0 => ftype::MOSTLY_TQ2_0,
        GgmlType::Iq2Xxs => ftype::MOSTLY_IQ2_XXS,
        GgmlType::Iq2Xs => ftype::MOSTLY_IQ2_XS,
        GgmlType::Iq2S => ftype::MOSTLY_IQ2_S,
        GgmlType::Iq3Xxs => ftype::MOSTLY_IQ3_XXS,
        GgmlType::Iq1S => ftype::MOSTLY_IQ1_S,
        GgmlType::Iq1M => ftype::MOSTLY_IQ1_M,
        GgmlType::Iq4Nl => ftype::MOSTLY_IQ4_NL,
        GgmlType::Iq4Xs => ftype::MOSTLY_IQ4_XS,
        GgmlType::Iq3S => ftype::MOSTLY_IQ3_S,
        _ => ftype::ALL_F32,
    }
}

/// `weight_buft_supported` stand-in: CPU/host buffers support every op.
fn weight_buft_supported(
    _hparams: &Hparams,
    _w: &GgmlTensor,
    op: WeightOp,
    buft: BufferType,
    dev: Device,
) -> bool {
    if op == WeightOp::Op(Op::None) {
        return true;
    }
    if dev == Device::Cpu {
        return true;
    }
    buft.supports_op(op)
}

fn select_weight_buft(
    hparams: &Hparams,
    w: &GgmlTensor,
    op: WeightOp,
    list: &BuftList,
) -> Option<BufferType> {
    for &(dev, buft) in list {
        if weight_buft_supported(hparams, w, op, buft, dev) {
            return Some(buft);
        }
    }
    None
}

impl ModelLoader {
    /// Constructor. `metadata` is used when neither `fname` nor `file` is given.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        metadata: Option<GgufContext>,
        set_tensor_data: Option<SetTensorData>,
        fname: &str,
        splits: &mut Vec<String>,
        file: Option<File>,
        load_mode: LoadMode,
        check_tensors: bool,
        no_alloc: bool,
        load_mtp: bool,
        overrides: &[KvOverride],
        tensor_buft_overrides: &[TensorBuftOverride],
    ) -> Result<Self> {
        let trace = std::env::var("LLAMA_TRACE")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);

        let mut this = Self {
            n_kv: 0,
            n_tensors: 0,
            n_created: 0,
            n_elements: 0,
            n_bytes: 0,
            use_mmap: matches!(
                load_mode,
                LoadMode::Mmap | LoadMode::MmapMlock | LoadMode::Auto
            ),
            use_direct_io: load_mode == LoadMode::DirectIo,
            check_tensors,
            no_alloc,
            load_mtp,
            lazy: LazyRead::default(),
            files: Vec::new(),
            ftype: ftype::ALL_F32,
            fver: FileVersion::V3,
            mappings: Vec::new(),
            weights_map: BTreeMap::new(),
            kv_overrides: HashMap::new(),
            tensor_buft_overrides: tensor_buft_overrides.to_vec(),
            metadata: metadata.unwrap_or_default(),
            set_tensor_data,
            contexts: Vec::new(),
            arch_name: String::new(),
            llm_kv: LlmKv::new(Arch::Unknown, None),
            size_done: 0,
            size_data: 0,
            mmaps_used: Vec::new(),
            ctx_map: BTreeMap::new(),
            n_tensors_moved: 0,
            first_tensor_moved_name: String::new(),
            first_tensor_moved_type_name: String::new(),
            first_moved_from_buft: None,
            first_moved_to_buft: None,
            trace,
            log: RefCell::new(Vec::new()),
        };

        for ov in overrides {
            this.kv_overrides.insert(ov.key.clone(), ov.clone());
        }

        if !fname.is_empty() {
            let mut params = GgufInitParams {
                no_alloc: true,
                ctx: Some(GgmlContext::default()),
            };
            let meta = gguf_init_from_file(fname, &mut params).map_err(|e| {
                Error::new(format!(
                    "llama_model_loader: failed to load model from {} ({})",
                    fname, e
                ))
            })?;
            let ctx = params.ctx.take().unwrap_or_default();
            this.metadata = meta;
            this.read_arch()?;

            let f = ModelFile::open(fname, this.use_direct_io)?;
            this.files.push(f);
            this.contexts.push(ctx);
            this.index_file(0)?;

            let mut paths = splits.clone();
            let mut n_split: u16 = 0;
            this.get_key_kv(ArchKv::SplitCount, &mut n_split, false)?;

            let mut n_backbone: usize = 1;
            let kv_split_no = this.llm_kv.key(ArchKv::SplitNo);
            if n_split > 1 {
                let mut idx: u16 = 0;
                this.get_key(&kv_split_no, &mut idx, true)?;
                if idx != 0 {
                    return Err(Error::new(format!("illegal split file idx: {} (file: {}), model must be loaded with the first split", idx, fname)));
                }

                let mut backbone =
                    if paths.is_empty() || split_prefix(fname, 0, n_split as u32).is_some() {
                        get_list_splits(fname, idx as u32, n_split as u32)?
                    } else {
                        Vec::new()
                    };
                if !backbone.is_empty() {
                    let mut extras = Vec::new();
                    for path in paths.drain(..) {
                        if let Some(slot) =
                            backbone.iter_mut().find(|b| basename(b) == basename(&path))
                        {
                            *slot = path; // given path wins, splits can be in different dirs
                        } else {
                            extras.push(path);
                        }
                    }
                    let mut merged = backbone;
                    merged.extend(extras);
                    paths = merged;
                } else if paths.len() != n_split as usize {
                    return Err(Error::new(format!(
                        "invalid split count, given: {} splits, but expected {}",
                        paths.len(),
                        n_split
                    )));
                }
                n_backbone = n_split as usize;
            } else if paths.is_empty() {
                paths.push(fname.to_string());
            }

            if paths.len() > u16::MAX as usize {
                return Err(Error::new("too many model files"));
            }
            let has_extras = paths.len() > n_backbone;

            // PLE-only GGUFs need not follow the exact split numbering.
            let mut skip_shard_order_validation = false;
            for path in paths.iter().skip(n_backbone) {
                let mut params = GgufInitParams {
                    no_alloc: true,
                    ctx: None,
                };
                if let Ok(probe) = gguf_init_from_file(path, &mut params) {
                    skip_shard_order_validation |= is_ple_only(&probe);
                }
            }

            let mut n_skipped = 0usize;
            let mut n_backbone_files = 1usize;
            let mut n_backbone_tensors = gguf_get_n_tensors(&this.metadata);
            for (idx, path) in paths.iter().enumerate().skip(1) {
                if skip_shard_order_validation && idx < n_backbone {
                    if File::open(path).is_err() {
                        this.log_info(format!("llama_model_loader: GGUF split {} not found, assuming it only holds the per-layer embedding table", path));
                        n_skipped += 1;
                        continue;
                    }
                }

                let mut params = GgufInitParams {
                    no_alloc: true,
                    ctx: Some(GgmlContext::default()),
                };
                let gguf = gguf_init_from_file(path, &mut params).map_err(|e| {
                    Error::new(format!(
                        "llama_model_loader: failed to load GGUF split from {} ({})",
                        path, e
                    ))
                })?;
                let ctx = params.ctx.take().unwrap_or_default();

                let mut allow_override_tensor = false;
                if idx < n_backbone {
                    let kid = gguf_find_key(&gguf, &kv_split_no);
                    if kid < 0 {
                        return Err(Error::new(format!(
                            "missing key {} in GGUF split {}",
                            kv_split_no, path
                        )));
                    }
                    if gguf_get_kv_type(&gguf, kid) != GgufType::Uint16 {
                        return Err(Error::new(format!(
                            "invalid type for {} in GGUF split {}",
                            kv_split_no, path
                        )));
                    }
                    let kv = &gguf.kv[kid as usize];
                    if kv.data.len() < 2 {
                        return Err(Error::new(format!(
                            "invalid type for {} in GGUF split {}",
                            kv_split_no, path
                        )));
                    }
                    let idx_gguf = u16::from_ne_bytes([kv.data[0], kv.data[1]]);
                    if idx_gguf as usize != idx {
                        return Err(Error::new(format!(
                            "invalid split file idx: {} (file: {}), expected {}",
                            idx_gguf, path, idx
                        )));
                    }
                    if skip_shard_order_validation && is_ple_only(&gguf) {
                        this.log_info(format!("llama_model_loader: skipping GGUF split {}, per-layer embedding table given separately", path));
                        n_skipped += 1;
                        continue;
                    }
                    n_backbone_tensors += gguf_get_n_tensors(&gguf);
                } else {
                    allow_override_tensor = is_ple_only(&gguf);
                    let numbered = gguf_find_key(&gguf, &kv_split_no) >= 0
                        || gguf_find_key(&gguf, &this.llm_kv.key(ArchKv::SplitCount)) >= 0
                        || gguf_find_key(&gguf, &this.llm_kv.key(ArchKv::SplitTensorsCount)) >= 0;
                    this.log_info(format!(
                        "llama_model_loader: appended extra GGUF shard {}{}",
                        path,
                        if numbered {
                            " (declared split numbering ignored)"
                        } else {
                            ""
                        }
                    ));
                }

                let f = ModelFile::open(path, this.use_direct_io)?;
                this.files.push(f);
                this.contexts.push(ctx.clone());
                if idx < n_backbone {
                    n_backbone_files = this.files.len();
                }
                let file_idx = (this.files.len() - 1) as u16;
                this.index_file_with_ctx(
                    file_idx,
                    &gguf,
                    &ctx,
                    allow_override_tensor,
                    n_backbone_files,
                )?;
            }

            if n_skipped > 0 {
                let mut declared_tensors = 0i32;
                this.get_key_kv(ArchKv::SplitTensorsCount, &mut declared_tensors, true)?;
                if n_skipped != 1 || n_backbone_tensors + 1 != declared_tensors as i64 {
                    return Err(Error::new(format!(
                        "incomplete model: {} split(s) skipped, {} of {} tensors found in the split files",
                        n_skipped, n_backbone_tensors, declared_tensors
                    )));
                }
            }
            if n_split > 1 || has_extras {
                let mut declared_tensors = gguf_get_n_tensors(&this.metadata) as i32;
                this.get_key_kv(
                    ArchKv::SplitTensorsCount,
                    &mut declared_tensors,
                    n_split > 1 && !has_extras,
                )?;
                if declared_tensors as usize != this.weights_map.len() {
                    if n_split > 1 && !has_extras {
                        return Err(Error::new(format!(
                            "corrupted model: {} tensors expected but {} found",
                            declared_tensors,
                            this.weights_map.len()
                        )));
                    }
                    this.log_info(format!(
                        "llama_model_loader: using {} registered tensors instead of {} declared tensors",
                        this.weights_map.len(), declared_tensors
                    ));
                }
            }
            if this.files.len() > 1 {
                this.log_info(format!(
                    "llama_model_loader: additional {} GGUFs metadata loaded.",
                    this.files.len() - 1
                ));
            }
        } else if let Some(file) = file {
            let duplicate = file.try_clone().map_err(|e| Error::new(e.to_string()))?;
            let mut params = GgufInitParams {
                no_alloc: true,
                ctx: Some(GgmlContext::default()),
            };
            let meta = gguf_init_from_file_ptr(file, &mut params).map_err(|e| {
                Error::new(format!(
                    "llama_model_loader: failed to load model from file pointer ({})",
                    e
                ))
            })?;
            let ctx = params.ctx.take().unwrap_or_default();
            let tensor_align = GGUF_DEFAULT_ALIGNMENT;
            if this.use_mmap && gguf_get_data_offset(&meta) % tensor_align != 0 {
                return Err(Error::new(format!(
                    "llama_model_loader: GGUF data section at file offset {} is not {} byte aligned, cannot mmap",
                    gguf_get_data_offset(&meta), tensor_align
                )));
            }
            this.metadata = meta;
            this.read_arch()?;
            let f = ModelFile::from_file(duplicate, this.use_direct_io)?;
            this.files.push(f);
            this.contexts.push(ctx);
            this.index_file(0)?;
        } else {
            this.read_arch()?;
        }

        this.n_kv = gguf_get_n_kv(&this.metadata) as i32;
        this.n_tensors = this.weights_map.len();
        this.fver = FileVersion::from_u32(gguf_get_version_local(&this.metadata));

        this.log_info(format!(
            "llama_model_loader: loaded meta data with {} key-value pairs and {} tensors from {} (version {})",
            this.n_kv,
            this.n_tensors,
            if fname.is_empty() { "(file*)" } else { fname },
            llama_file_version_name(this.fver)
        ));

        this.guess_ftype()?;

        if this.use_mmap && !MMAP_SUPPORTED {
            this.use_mmap = false;
        }

        Ok(this)
    }

    fn log_info(&self, msg: String) {
        if self.trace > 0 {
            eprintln!("{}", msg);
        }
        self.log.borrow_mut().push(msg);
    }

    /// Messages emitted by the loader, including warnings and metadata dump.
    pub fn log(&self) -> Vec<String> {
        self.log.borrow().clone()
    }

    fn read_arch(&mut self) -> Result<()> {
        let mut arch = String::new();
        self.get_key_str_kv(ArchKv::GeneralArchitecture, &mut arch, false)?;
        self.arch_name = arch;
        self.llm_kv = LlmKv::new(Arch::from_string(&self.arch_name), None);
        Ok(())
    }

    fn index_file(&mut self, file_idx: u16) -> Result<()> {
        let gguf = self.metadata.clone();
        let ctx = self.contexts.last().cloned().unwrap_or_default();
        self.index_file_with_ctx(file_idx, &gguf, &ctx, false, 0)
    }

    fn index_file_with_ctx(
        &mut self,
        file_idx: u16,
        gguf: &GgufContext,
        ctx: &GgmlContext,
        allow_override: bool,
        n_backbone_files: usize,
    ) -> Result<()> {
        let file_size = self
            .files
            .last()
            .map(|f| f.size)
            .ok_or_else(|| Error::new("no file to index"))?;
        for tensor in &ctx.tensors {
            let key = WeightKey::new(&tensor.name);
            if let Some(old) = self.weights_map.get(&key) {
                // Only a PLE-only extra shard may replace a backbone tensor.
                if !allow_override || old.idx as usize >= n_backbone_files {
                    return Err(Error::new(format!(
                        "invalid model: tensor '{}' is duplicated",
                        tensor.name
                    )));
                }
                self.log_info(format!(
                    "llama_model_loader: tensor '{}' overridden by {}",
                    tensor.name,
                    self.files.last().map_or("", |f| f.path.as_str())
                ));
                self.n_elements -= ggml_nelements(&old.tensor);
                self.n_bytes -= ggml_nbytes(&old.tensor);
                self.weights_map.remove(&key);
            }
            self.n_elements += ggml_nelements(tensor);
            self.n_bytes += ggml_nbytes(tensor);
            let w = TensorWeight::new(file_size, file_idx, gguf, tensor)?;
            self.weights_map.insert(key, w);
        }
        Ok(())
    }

    fn guess_ftype(&mut self) -> Result<()> {
        let mut n_type: HashMap<GgmlType, u32> = HashMap::new();
        let mut n_type_max = 0u32;
        let mut type_max = GgmlType::F32;

        for (key, w) in &self.weights_map {
            let ty = w.tensor.type_;
            let count = n_type.entry(ty).or_insert(0);
            *count += 1;
            if n_type_max < *count {
                n_type_max = *count;
                type_max = ty;
            }
            if self.trace > 0 {
                self.log_info(format!(
                    "llama_model_loader: - tensor split {:2}: {:32} {:<8} [ {} ] {:8.2} MiB",
                    w.idx,
                    key.name(),
                    ty.name(),
                    format_tensor_shape(&w.tensor),
                    ggml_nbytes(&w.tensor) as f32 / MIB as f32
                ));
            }
        }

        let mut ftype_val = ftype_for_type(type_max);
        if ftype_val == ftype::ALL_F32 && type_max != GgmlType::F32 {
            self.log_info(format!(
                "llama_model_loader: unknown type {}",
                type_max.name()
            ));
        }
        ftype_val |= ftype::GUESSED;
        let mut declared = 0u32;
        if self.get_key_kv(ArchKv::GeneralFileType, &mut declared, false)? {
            ftype_val = declared;
        }
        self.ftype = ftype_val;

        self.log_info("llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.".to_string());
        for i in 0..self.n_kv as usize {
            let kv = &self.metadata.kv[i];
            self.log_info(format!(
                "llama_model_loader: - kv {:3}: {:42} {:<16} = {}",
                i,
                kv.get_key(),
                kv_type_string(kv),
                kv_value_string(kv, 40)
            ));
        }
        for (ty, count) in &n_type {
            if *count == 0 {
                continue;
            }
            self.log_info(format!(
                "llama_model_loader: - type {:4}: {:4} tensors",
                ty.name(),
                count
            ));
        }
        Ok(())
    }

    pub fn get_arch_name(&self) -> &str {
        &self.arch_name
    }
    pub fn get_arch(&self) -> Arch {
        self.llm_kv.arch
    }

    pub fn get_weight(&self, name: &str) -> Option<&TensorWeight> {
        self.weights_map.get(&WeightKey::new(name))
    }

    pub fn require_weight(&self, name: &str) -> Result<&TensorWeight> {
        self.get_weight(name)
            .ok_or_else(|| Error::new(format!("llama_model_loader: tensor '{}' not found", name)))
    }

    pub fn get_tensor_meta(&self, name: &str) -> Option<&GgmlTensor> {
        self.get_weight(name).map(|w| &w.tensor)
    }

    pub fn require_tensor_meta(&self, name: &str) -> Result<&GgmlTensor> {
        self.get_tensor_meta(name)
            .ok_or_else(|| Error::new(format!("llama_model_loader: tensor '{}' not found", name)))
    }

    pub fn check_tensor_dims(
        &self,
        name: &str,
        ne: &[i64],
        required: bool,
        allow_reshape: bool,
    ) -> Result<Option<GgmlTensor>> {
        let cur = match self.get_tensor_meta(name) {
            Some(t) => t,
            None => {
                if !required {
                    return Ok(None);
                }
                return Err(Error::new(format!(
                    "llama_model_loader: tensor '{}' not found",
                    name
                )));
            }
        };

        let is_ok = if allow_reshape {
            let ncur = ggml_nelements(cur) as i128;
            let nexp = ne.iter().fold(1i128, |acc, &n| acc * n as i128);
            ncur == nexp
        } else {
            (0..GGML_MAX_DIMS).all(|i| {
                let expected = if i < ne.len() { ne[i] } else { 1 };
                cur.ne[i] == expected
            })
        };

        if !is_ok {
            return Err(Error::new(format!(
                "llama_model_loader: tensor '{}' has wrong shape; expected {}, got {}",
                name,
                format_shape(ne),
                format_tensor_shape(cur)
            )));
        }
        Ok(Some(cur.clone()))
    }

    /// `get_key` for scalar metadata values, including overrides.
    pub fn get_key<T: GetScalar>(&self, key: &str, result: &mut T, required: bool) -> Result<bool> {
        if let Some(ov) = self.kv_overrides.get(key) {
            if T::override_tag() == Some(ov.tag) {
                self.log_info(format!(
                    "get_key: Using metadata override ({:5}) '{}' = {}",
                    ov.tag.name(),
                    key,
                    override_value_string(ov)
                ));
                *result = T::from_override(ov);
                return Ok(true);
            }
            if let Some(expected) = T::override_tag() {
                self.log_info(format!("get_key: Warning: Bad metadata override type for key '{}', expected {} but got {}", key, expected.name(), ov.tag.name()));
            }
        }
        let kid = gguf_find_key(&self.metadata, key);
        if kid < 0 {
            if required {
                return Err(Error::new(format!("key not found in model: {}", key)));
            }
            return Ok(false);
        }
        let kv = &self.metadata.kv[kid as usize];
        let actual = gguf_get_kv_type(&self.metadata, kid);
        if actual != T::gt() {
            return Err(Error::new(format!(
                "key {} has wrong type {} but expected type {}",
                key,
                kv_type_name(actual),
                kv_type_name(T::gt())
            )));
        }
        *result = T::from_kv(kv, 0)?;
        Ok(true)
    }

    /// `get_key` for a string metadata value.
    pub fn get_key_str(&self, key: &str, result: &mut String, required: bool) -> Result<bool> {
        if let Some(ov) = self.kv_overrides.get(key) {
            if ov.tag == KvOverrideType::Str {
                self.log_info(format!(
                    "get_key: Using metadata override ({:5}) '{}' = {}",
                    ov.tag.name(),
                    key,
                    ov.val_str
                ));
                *result = ov.val_str.clone();
                return Ok(true);
            }
            self.log_info(format!("get_key: Warning: Bad metadata override type for key '{}', expected str but got {}", key, ov.tag.name()));
        }
        let kid = gguf_find_key(&self.metadata, key);
        if kid < 0 {
            if required {
                return Err(Error::new(format!("key not found in model: {}", key)));
            }
            return Ok(false);
        }
        let kv = &self.metadata.kv[kid as usize];
        if gguf_get_kv_type(&self.metadata, kid) != GgufType::String {
            return Err(Error::new(format!("key {} has wrong type", key)));
        }
        *result = kv.get_str(0).to_string();
        Ok(true)
    }

    pub fn get_key_kv<T: GetScalar>(
        &self,
        kid: ArchKv,
        result: &mut T,
        required: bool,
    ) -> Result<bool> {
        let key = self.llm_kv.key(kid);
        self.get_key(&key, result, required)
    }

    pub fn get_key_str_kv(&self, kid: ArchKv, result: &mut String, required: bool) -> Result<bool> {
        let key = self.llm_kv.key(kid);
        self.get_key_str(&key, result, required)
    }

    pub fn get_arr_n(&self, key: &str, required: bool) -> Result<Option<usize>> {
        let kid = gguf_find_key(&self.metadata, key);
        if kid < 0 {
            if required {
                return Err(Error::new(format!("key not found in model: {}", key)));
            }
            return Ok(None);
        }
        let kv = &self.metadata.kv[kid as usize];
        if gguf_get_kv_type(&self.metadata, kid) != GgufType::Array {
            return Err(Error::new(format!("{} is not an array", key)));
        }
        Ok(Some(kv.get_ne()))
    }

    pub fn get_arr_n_kv(&self, kid: ArchKv, required: bool) -> Result<Option<usize>> {
        let key = self.llm_kv.key(kid);
        self.get_arr_n(&key, required)
    }

    pub fn get_arr<T: GetScalar>(&self, key: &str, required: bool) -> Result<Option<Vec<T>>> {
        let kid = gguf_find_key(&self.metadata, key);
        if kid < 0 || gguf_get_kv_type(&self.metadata, kid) != GgufType::Array {
            if required {
                return Err(Error::new(format!("array key not found in model: {}", key)));
            }
            return Ok(None);
        }
        let kv = &self.metadata.kv[kid as usize];
        let gt = kv.type_;
        let scalar_gt = T::gt();
        let int_pair = matches!(gt, GgufType::Uint32 | GgufType::Int32)
            && matches!(scalar_gt, GgufType::Uint32 | GgufType::Int32);
        let bool_pair =
            gt == GgufType::Bool && matches!(scalar_gt, GgufType::Uint32 | GgufType::Int32);
        if gt != scalar_gt && !int_pair && !bool_pair {
            return Err(Error::new(format!(
                "{} has wrong array element type {}",
                key,
                kv_type_name(gt)
            )));
        }
        let n = kv.get_ne();
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            if gt == GgufType::Bool {
                result.push(T::from_override(&KvOverride::int("", kv.data[i] as i64)));
            } else {
                result.push(T::from_kv(kv, i)?);
            }
        }
        Ok(Some(result))
    }

    pub fn get_arr_kv<T: GetScalar>(&self, kid: ArchKv, required: bool) -> Result<Option<Vec<T>>> {
        let key = self.llm_kv.key(kid);
        self.get_arr(&key, required)
    }

    pub fn get_arr_string(&self, key: &str, required: bool) -> Result<Option<Vec<String>>> {
        let kid = gguf_find_key(&self.metadata, key);
        if kid < 0 || gguf_get_kv_type(&self.metadata, kid) != GgufType::Array {
            if required {
                return Err(Error::new(format!("array key not found in model: {}", key)));
            }
            return Ok(None);
        }
        let kv = &self.metadata.kv[kid as usize];
        if kv.type_ != GgufType::String {
            return Err(Error::new(format!(
                "{} has wrong array element type {}",
                key,
                kv_type_name(kv.type_)
            )));
        }
        Ok(Some(
            (0..kv.get_ne())
                .map(|i| kv.get_str(i).to_string())
                .collect(),
        ))
    }

    pub fn get_arr_string_kv(&self, kid: ArchKv, required: bool) -> Result<Option<Vec<String>>> {
        let key = self.llm_kv.key(kid);
        self.get_arr_string(&key, required)
    }

    /// Get an array of exactly `n` elements, or repeat a scalar `n` times.
    pub fn get_key_or_arr<T: GetScalar>(
        &self,
        key: &str,
        n: usize,
        required: bool,
    ) -> Result<Option<Vec<T>>> {
        let kid = gguf_find_key(&self.metadata, key);
        if kid < 0 {
            if required {
                return Err(Error::new(format!("key not found in model: {}", key)));
            }
            return Ok(None);
        }
        if gguf_get_kv_type(&self.metadata, kid) == GgufType::Array {
            let arr = self.get_arr::<T>(key, required)?.unwrap_or_default();
            if arr.len() != n {
                return Err(Error::new(format!(
                    "key {} has wrong array length; expected {}, got {}",
                    key,
                    n,
                    arr.len()
                )));
            }
            return Ok(Some(arr));
        }
        let mut value = T::from_override(&KvOverride::int("", 0));
        if !self.get_key(key, &mut value, required)? {
            return Ok(None);
        }
        Ok(Some(vec![value; n]))
    }

    pub fn get_key_or_arr_kv<T: GetScalar>(
        &self,
        kid: ArchKv,
        n: usize,
        required: bool,
    ) -> Result<Option<Vec<T>>> {
        let key = self.llm_kv.key(kid);
        self.get_key_or_arr(&key, n, required)
    }

    /// Scalar-only variant: throw when the key holds an array.
    pub fn get_key_or_arr_u32(&self, key: &str, required: bool) -> Result<Option<u32>> {
        let kid = gguf_find_key(&self.metadata, key);
        if kid < 0 {
            if required {
                return Err(Error::new(format!("key not found in model: {}", key)));
            }
            return Ok(None);
        }
        if gguf_get_kv_type(&self.metadata, kid) == GgufType::Array {
            if required {
                return Err(Error::new(format!(
                    "expected scalar, found array for key: {}",
                    key
                )));
            }
            return Ok(None);
        }
        let mut value = 0u32;
        if !self.get_key(key, &mut value, required)? {
            return Ok(None);
        }
        Ok(Some(value))
    }

    pub fn get_key_or_arr_u32_kv(&self, kid: ArchKv, required: bool) -> Result<Option<u32>> {
        let key = self.llm_kv.key(kid);
        self.get_key_or_arr_u32(&key, required)
    }

    /// `create_tensor`: classify a tensor, choose its buffer and register it.
    #[allow(clippy::too_many_arguments)]
    pub fn create_tensor(
        &mut self,
        hparams: &Hparams,
        buft_list_cpu: &BuftList,
        buft_list_input: &BuftList,
        buft_list_output: &BuftList,
        buft_list_layer: Option<&BuftList>,
        tn: &LlmTnImpl,
        ne: &[i64],
        flags: u32,
    ) -> Result<Option<GgmlTensor>> {
        let name = tn.str()?;
        let mut is_lazy = false;

        let info = if self.files.is_empty() {
            if flags & TENSOR_SKIP_IF_VIRTUAL != 0 {
                return Ok(None);
            }
            let tid = gguf_find_tensor(&self.metadata, &name);
            let ty = if tid >= 0 {
                gguf_get_tensor_type(&self.metadata, tid)
            } else {
                GgmlType::F32
            };
            if flags & TENSOR_NOT_REQUIRED != 0 && ne.iter().any(|&n| n <= 0) {
                return Ok(None);
            }
            let mut t_meta = GgmlTensor::default();
            t_meta.name = name.clone();
            t_meta.type_ = ty;
            for dim in 0..GGML_MAX_DIMS {
                t_meta.ne[dim] = if dim < ne.len() { ne[dim] } else { 1 };
                t_meta.nb[dim] = if dim == 0 {
                    ggml_type_size(ty)
                } else if dim == 1 {
                    ggml_row_size(ty, t_meta.ne[0])
                } else {
                    t_meta.nb[dim - 1] * t_meta.ne[dim - 1].max(0) as usize
                };
            }
            Some(t_meta)
        } else {
            let cur = self.check_tensor_dims(
                &name,
                ne,
                flags & TENSOR_NOT_REQUIRED == 0,
                flags & TENSOR_ALLOW_RESHAPE != 0,
            )?;
            match cur {
                Some(c) => {
                    if flags & TENSOR_READ_LAZY != 0 {
                        let w = if self.no_alloc {
                            None
                        } else {
                            self.get_weight(&name).cloned()
                        };
                        is_lazy = self.lazy.add(&name, &c, w.as_ref());
                    }
                    let original_nbytes = ggml_nbytes(&c);
                    let mut t_meta = c;
                    if flags & TENSOR_ALLOW_RESHAPE != 0 {
                        for dim in 0..GGML_MAX_DIMS {
                            t_meta.ne[dim] = if dim < ne.len() { ne[dim] } else { 1 };
                            t_meta.nb[dim] = if dim == 0 {
                                ggml_type_size(t_meta.type_)
                            } else if dim == 1 {
                                ggml_row_size(t_meta.type_, t_meta.ne[0])
                            } else {
                                t_meta.nb[dim - 1] * t_meta.ne[dim - 1].max(0) as usize
                            };
                        }
                    }
                    if ggml_nbytes(&t_meta) != original_nbytes {
                        return Err(Error::new(format!(
                            "tensor '{}' reshape changed its size",
                            name
                        )));
                    }
                    Some(t_meta)
                }
                None => None,
            }
        };

        let Some(t_meta) = info else { return Ok(None) };

        let buft = match self.buft_for_tensor(
            hparams,
            buft_list_cpu,
            buft_list_input,
            buft_list_output,
            buft_list_layer,
            tn,
            &t_meta,
            flags,
            is_lazy,
        )? {
            Some(b) => b,
            None => return Ok(None),
        };

        let key = CtxKey {
            buft,
            lazy: is_lazy,
        };
        let entry = self.ctx_map.entry(key).or_insert_with(CtxEntry::default);

        if flags & TENSOR_DUPLICATED != 0 {
            if let Some(existing) = entry.tensors.iter().find(|t| t.name == name) {
                return Ok(Some(existing.clone()));
            }
        }

        let nbytes = ggml_nbytes(&t_meta);
        entry.tensors.push(t_meta.clone());
        if flags & TENSOR_DUPLICATED != 0 {
            self.size_data += nbytes;
        } else {
            self.n_created += 1;
        }
        Ok(Some(t_meta))
    }

    #[allow(clippy::too_many_arguments)]
    fn buft_for_tensor(
        &mut self,
        hparams: &Hparams,
        buft_list_cpu: &BuftList,
        buft_list_input: &BuftList,
        buft_list_output: &BuftList,
        buft_list_layer: Option<&BuftList>,
        tn: &LlmTnImpl,
        t_meta: &GgmlTensor,
        flags: u32,
        is_lazy: bool,
    ) -> Result<Option<BufferType>> {
        let tn_tensor = if tn.tensor == ArchTensor::TokenEmbd && flags & TENSOR_DUPLICATED != 0 {
            ArchTensor::Output
        } else {
            tn.tensor
        };
        let info: &TensorInfo = tensor_info_for(tn_tensor).map_err(|_| {
            Error::new(format!(
                "missing tensor info mapping for {}",
                tn.str().unwrap_or_default()
            ))
        })?;

        if info.op == Op::None || flags & TENSOR_SKIP != 0 {
            self.log_info(format!(
                "model has unused tensor {} (size = {} bytes) -- ignoring",
                tn.str().unwrap_or_default(),
                ggml_nbytes(t_meta)
            ));
            self.size_data = self.size_data.saturating_sub(ggml_nbytes(t_meta));
            self.n_created += 1;
            return Ok(None);
        }

        let name = tn.str().unwrap_or_default();
        let op = if tn.suffix.as_deref() == Some("bias") {
            if info.op == Op::MulMatId {
                WeightOp::AddId
            } else {
                WeightOp::Op(Op::Add)
            }
        } else if hparams.router_layer >= 0
            && matches!(tn.suffix.as_deref(), Some("lora_a") | Some("lora_b"))
        {
            WeightOp::Op(Op::MulMatId)
        } else {
            WeightOp::Op(info.op)
        };

        if info.layer == TensorLayer::Input || info.layer == TensorLayer::Output {
            if tn.bid != -1 {
                return Err(Error::new(format!(
                    "input/output layer tensor {} used with a layer number",
                    name
                )));
            }
        } else if tn.bid == -1 {
            return Err(Error::new(format!(
                "repeating layer tensor {} used without a layer number",
                name
            )));
        }

        if is_lazy {
            return Ok(Some(LazyRead::buft()));
        }

        let buft_list: &BuftList = match info.layer {
            TensorLayer::Input => buft_list_input,
            TensorLayer::Output => buft_list_output,
            TensorLayer::Repeating => {
                buft_list_layer.ok_or_else(|| Error::new("missing layer buffer list"))?
            }
        };

        let mut buft = None;
        for ov in &self.tensor_buft_overrides {
            if let Some(pattern) = &ov.pattern {
                if regex_search(&name, pattern) {
                    buft = if ov.buft == BufferType::Cpu {
                        select_weight_buft(hparams, t_meta, op, buft_list_cpu)
                    } else {
                        Some(ov.buft)
                    };
                    break;
                }
            }
        }

        let selected = match buft {
            Some(b) => b,
            None => select_weight_buft(hparams, t_meta, op, buft_list).ok_or_else(|| {
                Error::new(format!(
                    "failed to find a compatible buffer type for tensor {}",
                    name
                ))
            })?,
        };

        let mut selected = selected;
        if self.use_mmap && selected.device() == Device::Cpu && selected == BufferType::Host {
            selected = BufferType::Cpu;
        }
        if let Some(&(_, preferred)) = buft_list.first() {
            if selected != preferred {
                if self.n_tensors_moved == 0 {
                    self.first_tensor_moved_name = t_meta.name.clone();
                    self.first_tensor_moved_type_name = t_meta.type_.name().to_string();
                    self.first_moved_from_buft = Some(preferred);
                    self.first_moved_to_buft = Some(selected);
                }
                self.n_tensors_moved += 1;
            }
        }
        Ok(Some(selected))
    }

    pub fn done_getting_tensors(&mut self, partial: bool) -> Result<()> {
        if self.n_created > self.n_tensors {
            return Err(Error::new(format!(
                "llama_model_loader: too many tensors created; expected {}, got {}",
                self.n_tensors, self.n_created
            )));
        }
        if self.n_created < self.n_tensors {
            if !partial {
                return Err(Error::new(format!(
                    "llama_model_loader: wrong number of tensors; expected {}, got {}",
                    self.n_tensors, self.n_created
                )));
            }
            self.log_info(format!(
                "llama_model_loader: partial load - used {} of {} tensors in the file",
                self.n_created, self.n_tensors
            ));
        }
        if self.n_tensors_moved > 0 {
            let from = self.first_moved_from_buft.map_or("unknown", |b| b.name());
            let to = self.first_moved_to_buft.map_or("unknown", |b| b.name());
            self.log_info(format!(
                "llama_model_loader: tensor '{}' ({}) (and {} others) cannot be used with preferred buffer type {}, using {} instead",
                self.first_tensor_moved_name,
                self.first_tensor_moved_type_name,
                self.n_tensors_moved - 1,
                from,
                to
            ));
        }
        Ok(())
    }

    /// `init_mappings`: read each file into memory when mmap or lazy reads are used.
    pub fn init_mappings(
        &mut self,
        prefetch: bool,
        mut mlock_mmaps: Option<&mut Mlocks>,
    ) -> Result<()> {
        if self.use_mmap || self.lazy.any() {
            for idx in 0..self.files.len() {
                let mapping = Mapping::new(&mut self.files[idx])?;
                self.mmaps_used.push((mapping.size(), 0));
                if let Some(locks) = mlock_mmaps.as_deref_mut() {
                    let mut lock = Mlock::default();
                    lock.init();
                    locks.push(lock);
                }
                self.mappings.push(mapping);
            }
            let _ = prefetch; // owned bytes are read in full either way
        }
        for w in self.weights_map.values() {
            self.size_data += ggml_nbytes(&w.tensor);
        }
        Ok(())
    }

    pub fn get_mapping_ranges(&self, idx: u32, ctx_key: CtxKey) -> Vec<MappingRange> {
        let mut own: HashSet<WeightKey> = HashSet::new();
        let mut own_ranges = Vec::new();
        let mut other = Vec::new();

        if let Some(entry) = self.ctx_map.get(&ctx_key) {
            for tensor in &entry.tensors {
                if let Some(w) = self.get_weight(&tensor.name) {
                    if w.idx as u32 == idx {
                        own.insert(WeightKey::new(&tensor.name));
                        own_ranges.push(MappingRange {
                            first: w.offs,
                            last: w.offs.saturating_add(ggml_nbytes(tensor)),
                        });
                    }
                }
            }
        }
        for (key, entry) in &self.ctx_map {
            if *key == ctx_key {
                continue;
            }
            for tensor in &entry.tensors {
                let wkey = WeightKey::new(&tensor.name);
                if let Some(w) = self.weights_map.get(&wkey) {
                    if w.idx as u32 == idx && !own.contains(&wkey) {
                        other.push(MappingRange {
                            first: w.offs,
                            last: w.offs.saturating_add(ggml_nbytes(tensor)),
                        });
                    }
                }
            }
        }
        sub_ranges(&merge_ranges(own_ranges), &merge_ranges(other))
    }

    pub fn unmap_weight(&mut self, w: &TensorWeight) -> Result<()> {
        if !self.use_mmap {
            return Ok(());
        }
        let idx = w.idx as usize;
        let mapping = self
            .mappings
            .get_mut(idx)
            .ok_or_else(|| Error::new("missing mapping for weight"))?;
        mapping.unmap_fragment(w.offs, w.offs + ggml_nbytes(&w.tensor));
        Ok(())
    }

    /// Read a byte range of a weight; with mmap this borrows from the mapping.
    pub fn load_data_range(
        &mut self,
        w: &TensorWeight,
        offs: usize,
        size: usize,
        buf: &mut [u8],
    ) -> Result<Vec<u8>> {
        let nbytes = ggml_nbytes(&w.tensor);
        if offs.checked_add(size).map_or(true, |end| end > nbytes) {
            return Err(Error::new("load_data_range: range exceeds tensor"));
        }
        let data = if self.use_mmap {
            let mapping = self
                .mappings
                .get(w.idx as usize)
                .ok_or_else(|| Error::new("missing mapping for weight"))?;
            let start = w.offs + offs;
            mapping
                .addr()
                .get(start..start + size)
                .ok_or_else(|| Error::new("mapping range out of bounds"))?
                .to_vec()
        } else {
            if buf.len() < size {
                return Err(Error::new("load_data_range: buffer too small"));
            }
            let file = self
                .files
                .get_mut(w.idx as usize)
                .ok_or_else(|| Error::new("missing file for weight"))?;
            file.read_at(w.offs + offs, size)?
        };
        if self.check_tensors && !validate_row_data(w.tensor.type_, &data, size) {
            return Err(Error::new(format!(
                "tensor '{}' has invalid data",
                w.tensor.name
            )));
        }
        Ok(data)
    }

    /// `load_all_data`: fill tensor data from mappings or files.
    pub fn load_all_data(
        &mut self,
        ctx: &mut GgmlContext,
        bufs: &[BufferType],
        mut progress_callback: Option<&mut ProgressCallback>,
    ) -> Result<bool> {
        if self.files.is_empty() {
            if let Some(cb) = self.set_tensor_data.take() {
                let mut cb = cb;
                for t in &mut ctx.tensors {
                    cb(t);
                }
                self.set_tensor_data = Some(cb);
            }
            return Ok(true);
        }
        if self.size_data == 0 {
            return Err(Error::new("llama_model_loader: call init_mappings() first"));
        }

        let mut names: Vec<String> = ctx.tensors.iter().map(|t| t.name.clone()).collect();
        if !self.use_mmap {
            names.sort_by(|a, b| {
                let staged = |name: &str| {
                    self.ctx_map
                        .iter()
                        .find(|(_, entry)| entry.tensors.iter().any(|t| t.name == name))
                        .map_or(false, |(key, _)| {
                            if bufs.is_empty() {
                                !matches!(key.buft, BufferType::Cpu | BufferType::Host)
                            } else {
                                !bufs.contains(&key.buft)
                            }
                        })
                };
                let na = self.get_weight(a).map_or(0, |w| ggml_nbytes(&w.tensor));
                let nb = self.get_weight(b).map_or(0, |w| ggml_nbytes(&w.tensor));
                staged(b).cmp(&staged(a)).then_with(|| nb.cmp(&na))
            });
        }

        for name in &names {
            let w = match self.get_weight(name) {
                Some(w) => w.clone(),
                None => continue, // split-expert tensors without a weight
            };
            if let Some(cb) = progress_callback.as_deref_mut() {
                if !cb(self.size_done as f32 / self.size_data as f32) {
                    return Ok(false);
                }
            }
            let n_size = ggml_nbytes(&w.tensor);
            let from_mapping = self.use_mmap || self.lazy.has_name(name);
            let data = if from_mapping {
                let mapping = self
                    .mappings
                    .get(w.idx as usize)
                    .ok_or_else(|| Error::new("missing mapping for weight"))?;
                let start = w.offs;
                mapping
                    .addr()
                    .get(start..start + n_size)
                    .ok_or_else(|| Error::new("mapping range out of bounds"))?
                    .to_vec()
            } else {
                let file = self
                    .files
                    .get_mut(w.idx as usize)
                    .ok_or_else(|| Error::new("missing file for weight"))?;
                file.read_at(w.offs, n_size)?
            };
            if self.check_tensors && !validate_row_data(w.tensor.type_, &data, n_size) {
                return Err(Error::new(format!("tensor '{}' has invalid data", name)));
            }

            if from_mapping {
                let used = self
                    .mmaps_used
                    .get_mut(w.idx as usize)
                    .ok_or_else(|| Error::new("missing mmap usage entry"))?;
                used.0 = used.0.min(w.offs);
                used.1 = used.1.max(w.offs + n_size);
            }

            if let Some(tensor) = ctx.tensors.iter_mut().find(|t| &t.name == name) {
                tensor.data = data;
            }
            self.size_done += n_size;
        }

        if self.size_done >= self.size_data {
            if self.use_mmap {
                for idx in 0..self.mappings.len() {
                    let (first, last) = self.mmaps_used[idx];
                    self.mappings[idx].unmap_fragment(0, first);
                    if last != 0 {
                        let size = self.mappings[idx].size();
                        self.mappings[idx].unmap_fragment(last, size);
                    }
                }
            }
            if let Some(cb) = progress_callback.as_deref_mut() {
                return Ok(cb(1.0));
            }
        }
        Ok(true)
    }

    pub fn ftype_name(&self) -> String {
        ftype_name(self.ftype)
    }

    pub fn print_info(&mut self) {
        let bpw = if self.n_elements == 0 {
            0.0
        } else {
            self.n_bytes as f64 * 8.0 / self.n_elements as f64
        };
        let msg = if self.n_bytes < GIB {
            format!(
                "llama_model_loader: file size = {:.2} MiB ({:.2} BPW)",
                self.n_bytes as f64 / MIB as f64,
                bpw
            )
        } else {
            format!(
                "llama_model_loader: file size = {:.2} GiB ({:.2} BPW)",
                self.n_bytes as f64 / GIB as f64,
                bpw
            )
        };
        self.log_info(format!(
            "llama_model_loader: file format = {}",
            llama_file_version_name(self.fver)
        ));
        self.log_info(format!(
            "llama_model_loader: file type   = {}",
            self.ftype_name()
        ));
        self.log_info(msg);
    }
}

fn override_value_string(ov: &KvOverride) -> String {
    match ov.tag {
        KvOverrideType::Bool => ov.val_bool.to_string(),
        KvOverrideType::Int => ov.val_i64.to_string(),
        KvOverrideType::Float => format!("{:.6}", ov.val_f64),
        KvOverrideType::Str => ov.val_str.clone(),
    }
}

fn gguf_get_version_local(ctx: &GgufContext) -> u32 {
    ctx.version
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blk_layer_parsing_matches_sscanf() {
        assert_eq!(parse_blk_layer("blk.12.attn_q.weight"), 12);
        assert_eq!(parse_blk_layer("blk.0.ffn_down_exps.weight"), 0);
        assert_eq!(parse_blk_layer("output.weight"), -1);
        assert_eq!(parse_blk_layer("blk.x.attn_q"), -1);
        assert_eq!(parse_blk_layer("blk.12attn_q"), -1);
    }

    #[test]
    fn split_names_round_trip() {
        let path = split_path("models/llama", 1, 4);
        assert_eq!(path, "models/llama-00002-of-00004.gguf");
        assert_eq!(split_prefix(&path, 1, 4), Some("models/llama".to_string()));
        assert_eq!(split_prefix("models/llama.gguf", 0, 4), None);
    }

    #[test]
    fn range_merge_and_subtract() {
        let merged = merge_ranges(vec![
            MappingRange { first: 8, last: 12 },
            MappingRange { first: 0, last: 4 },
            MappingRange { first: 4, last: 9 },
        ]);
        assert_eq!(merged, vec![MappingRange { first: 0, last: 12 }]);
        let sub = sub_ranges(
            &[MappingRange { first: 0, last: 10 }],
            &[MappingRange { first: 3, last: 5 }],
        );
        assert_eq!(
            sub,
            vec![
                MappingRange { first: 0, last: 3 },
                MappingRange { first: 5, last: 10 },
            ]
        );
    }

    #[test]
    fn tensor_override_patterns() {
        assert!(regex_search(
            "blk.12.attn_q.weight",
            r"^blk\.[0-9]+\.attn_q"
        ));
        assert!(regex_search("blk.3.ffn_gate.weight", r"ffn_[gu]ate"));
        assert!(!regex_search(
            "blk.3.ffn_down.weight",
            r"^blk\.[0-9]+\.attn_q"
        ));
    }

    #[test]
    fn metadata_getters_and_overrides() {
        let mut md = GgufContext::default();
        md.kv.push(GgufKv::new_scalar("test.i32", 7i32));
        md.kv.push(GgufKv::new_array("test.arr", &[1u32, 2, 3]));
        let overrides = [KvOverride::int("test.i32", 42)];
        let mut splits = Vec::new();
        let loader = ModelLoader::new(
            Some(md),
            None,
            "",
            &mut splits,
            None,
            LoadMode::None,
            false,
            false,
            false,
            &overrides,
            &[],
        )
        .unwrap();

        let mut value = 0i32;
        assert!(loader.get_key("test.i32", &mut value, true).unwrap());
        assert_eq!(value, 42);
        assert_eq!(
            loader.get_arr::<u32>("test.arr", true).unwrap(),
            Some(vec![1, 2, 3])
        );
        assert!(loader.get_key::<f32>("test.i32", &mut 0.0, true).is_err());
        assert!(loader
            .get_key::<i32>("test.missing", &mut value, true)
            .is_err());
    }

    #[test]
    fn ftype_names() {
        assert_eq!(ftype_name(ftype::MOSTLY_Q4_K_M), "Q4_K - Medium");
        assert_eq!(
            ftype_name(ftype::MOSTLY_Q4_K_M | ftype::GUESSED),
            "(guessed) Q4_K - Medium"
        );
    }
}

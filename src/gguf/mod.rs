//! Rust translation of `gguf.cpp` / `gguf.h`.
//!
//! GGUF files have the following structure:
//!
//! 1. File magic "GGUF" (4 bytes).
//! 2. File version (uint32_t).
//! 3. Number of ggml tensors in file (int64_t).
//! 4. Number of key-value-pairs in file (int64_t).
//! 5. For each KV pair:
//!    1. The key (string).
//!    2. The value type (gguf_type).
//!    3a. If the value type is GGUF_TYPE_ARRAY:
//!       1. The type of the array (gguf_type).
//!       2. The number of elements in the array (uint64_t).
//!       3. The binary representation of each element in the array.
//!    3b. Otherwise:
//!       1. The binary representation of the value.
//! 6. For each ggml tensor:
//!    1. The tensor name (string).
//!    2. The number of dimensions of the tensor (uint32_t).
//!    3. For each dimension:
//!       1. The size of the tensor in the dimension (int64_t).
//!    4. The tensor data type (ggml_type).
//!    5. The tensor data offset in the tensor data binary blob (uint64_t).
//! 7. The tensor data binary blob (optional, aligned).
//!
//! Strings are serialized as the string length (uint64_t) followed by the C
//! string without the null terminator. All enums are stored as int32_t. All
//! bool values are stored as int8_t. If the special key "general.alignment"
//! (uint32_t) is defined it is used for alignment, otherwise
//! GGUF_DEFAULT_ALIGNMENT is used.
//!
//! The C++ code reads and writes typed values through raw byte copies in the
//! host byte order; `from_ne_bytes` / `to_ne_bytes` preserve that behaviour,
//! including the endianness check on the version field.
//!
//! Module maintainer: Johannes Gäßler (@JohannesGaessler, johannesg@5d6.de)

pub mod kv;
pub mod reader;
pub mod types;
pub mod writer;

#[cfg(test)]
mod tests;

pub use kv::*;
pub use reader::*;
pub use types::*;
pub use writer::*;

// ---------------------------------------------------------------------------
// gguf_tensor_info / gguf_context / gguf_init_params
// ---------------------------------------------------------------------------

/// Minimal `ggml_tensor` equivalent: only the info GGUF needs.
#[derive(Debug, Clone)]
pub struct GgmlTensor {
    pub name: String,
    /// Equivalent of `ne`, an array of GGML_MAX_DIMS elements; `ne[dim]` is 1
    /// for dim >= n_dims.
    pub ne: [i64; GGML_MAX_DIMS],
    pub type_: GgmlType,
    pub nb: [usize; GGML_MAX_DIMS],
    /// Owned bytes for the tensor data. Mirrors `ggml_tensor::data`.
    pub data: Vec<u8>,
}

impl Default for GgmlTensor {
    fn default() -> Self {
        Self {
            name: String::new(),
            ne: [1; GGML_MAX_DIMS],
            type_: GgmlType::F32,
            nb: [4; GGML_MAX_DIMS],
            data: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GgufTensorInfo {
    pub t: GgmlTensor,
    /// Offset from start of `data`, must be a multiple of `alignment`.
    pub offset: u64,
}

/// Stand-in for `ggml_context`, created when `GgufInitParams::ctx` is set.
#[derive(Debug, Clone, Default)]
pub struct GgmlContext {
    pub tensors: Vec<GgmlTensor>,
    pub no_alloc: bool,
}

#[derive(Debug, Clone)]
pub struct GgufContext {
    pub version: u32,
    pub kv: Vec<GgufKv>,
    pub info: Vec<GgufTensorInfo>,
    pub alignment: usize,
    /// Offset of `data` from beginning of file.
    pub offset: usize,
    /// Size of `data` in bytes.
    pub size: usize,
    /// The tensor data binary blob.
    pub data: Vec<u8>,
}

impl Default for GgufContext {
    fn default() -> Self {
        Self {
            version: GGUF_VERSION,
            kv: Vec::new(),
            info: Vec::new(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
            offset: 0,
            size: 0,
            data: Vec::new(),
        }
    }
}

pub struct GgufInitParams {
    pub no_alloc: bool,
    /// Safe equivalent of `ggml_context ** ctx`. Set to `Some(...)` to create
    /// a ggml context and allocate tensor data in it; the slot is overwritten
    /// with the created context on success.
    pub ctx: Option<GgmlContext>,
}

impl Default for GgufInitParams {
    fn default() -> Self {
        Self {
            no_alloc: false,
            ctx: None,
        }
    }
}

/// `size_t ggml_nbytes(const struct ggml_tensor *)`.
pub fn ggml_nbytes(t: &GgmlTensor) -> usize {
    let mut nelements = 1i128;
    for &ne in &t.ne {
        nelements *= ne as i128;
    }
    let blck = ggml_blck_size(t.type_).max(1) as i128;
    let nbytes = nelements * ggml_type_size(t.type_) as i128 / blck;
    if nbytes < 0 {
        return 0;
    }
    nbytes as usize
}

/// `int ggml_n_dims(const struct ggml_tensor *)`.
pub fn ggml_n_dims(t: &GgmlTensor) -> usize {
    let mut n_dims = GGML_MAX_DIMS;
    while n_dims > 1 && t.ne[n_dims - 1] == 1 {
        n_dims -= 1;
    }
    n_dims
}

/// `bool ggml_is_contiguous(const struct ggml_tensor *)`.
pub fn ggml_is_contiguous(t: &GgmlTensor) -> bool {
    if t.nb[0] != ggml_type_size(t.type_) {
        return false;
    }
    (1..GGML_MAX_DIMS).all(|i| t.nb[i] == t.nb[i - 1] * t.ne[i - 1].max(0) as usize)
}

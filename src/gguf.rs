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

use std::collections::HashSet;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::core::{Error, Result};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const GGUF_VERSION: u32 = 3;

pub const GGUF_KEY_GENERAL_ALIGNMENT: &str = "general.alignment";

pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

pub const GGUF_MAX_STRING_LENGTH: usize = 1024 * 1024 * 1024;
pub const GGUF_MAX_ARRAY_ELEMENTS: usize = 1024 * 1024 * 1024;

// Stand-ins for the ggml constants used by gguf.cpp. Kept local so this module
// stays self-contained until the ggml port is wired in.
pub const GGML_MAX_DIMS: usize = 4;
pub const GGML_MAX_NAME: usize = 64;
pub const GGML_TYPE_COUNT: usize = 31;

// ---------------------------------------------------------------------------
// gguf_type
// ---------------------------------------------------------------------------

/// Types that can be stored as GGUF KV data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
    Count = 13, // marks the end of the enum
}

impl GgufType {
    /// `GGUF_TYPE_SIZE` / `gguf_type_size`. 0 means undefined (string, array).
    pub fn type_size(self) -> usize {
        match self {
            GgufType::Uint8 => std::mem::size_of::<u8>(),
            GgufType::Int8 => std::mem::size_of::<i8>(),
            GgufType::Uint16 => std::mem::size_of::<u16>(),
            GgufType::Int16 => std::mem::size_of::<i16>(),
            GgufType::Uint32 => std::mem::size_of::<u32>(),
            GgufType::Int32 => std::mem::size_of::<i32>(),
            GgufType::Float32 => std::mem::size_of::<f32>(),
            GgufType::Bool => std::mem::size_of::<i8>(),
            GgufType::String => 0, // undefined
            GgufType::Array => 0,  // undefined
            GgufType::Uint64 => std::mem::size_of::<u64>(),
            GgufType::Int64 => std::mem::size_of::<i64>(),
            GgufType::Float64 => std::mem::size_of::<f64>(),
            GgufType::Count => 0,
        }
    }

    /// `GGUF_TYPE_NAME` / `gguf_type_name`. C++ returns null for unknown types.
    pub fn name(self) -> Option<&'static str> {
        match self {
            GgufType::Uint8 => Some("u8"),
            GgufType::Int8 => Some("i8"),
            GgufType::Uint16 => Some("u16"),
            GgufType::Int16 => Some("i16"),
            GgufType::Uint32 => Some("u32"),
            GgufType::Int32 => Some("i32"),
            GgufType::Float32 => Some("f32"),
            GgufType::Bool => Some("bool"),
            GgufType::String => Some("str"),
            GgufType::Array => Some("arr"),
            GgufType::Uint64 => Some("u64"),
            GgufType::Int64 => Some("i64"),
            GgufType::Float64 => Some("f64"),
            GgufType::Count => None,
        }
    }
}

impl TryFrom<i32> for GgufType {
    type Error = Error;

    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(GgufType::Uint8),
            1 => Ok(GgufType::Int8),
            2 => Ok(GgufType::Uint16),
            3 => Ok(GgufType::Int16),
            4 => Ok(GgufType::Uint32),
            5 => Ok(GgufType::Int32),
            6 => Ok(GgufType::Float32),
            7 => Ok(GgufType::Bool),
            8 => Ok(GgufType::String),
            9 => Ok(GgufType::Array),
            10 => Ok(GgufType::Uint64),
            11 => Ok(GgufType::Int64),
            12 => Ok(GgufType::Float64),
            _ => Err(Error::new(format!("invalid GGUF type {}", value))),
        }
    }
}

// static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");
const _: () = assert!(GgufType::Count as i32 == 13);

/// `gguf_type_size(enum gguf_type type)`.
pub fn gguf_type_size(ty: GgufType) -> usize {
    ty.type_size()
}

/// `gguf_type_name(enum gguf_type type)`.
pub fn gguf_type_name(ty: GgufType) -> Option<&'static str> {
    ty.name()
}

// ---------------------------------------------------------------------------
// ggml_type (local stand-in for the ggml port)
// ---------------------------------------------------------------------------

/// ggml tensor element types. Matches the numeric values checked by GGUF
/// readers (`0 <= type < GGML_TYPE_COUNT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 4,
    Q5_1 = 5,
    Q8_0 = 6,
    Q8_1 = 7,
    Q2K = 8,
    Q3K = 9,
    Q4K = 10,
    Q5K = 11,
    Q6K = 12,
    Q8K = 13,
    I8 = 14,
    I16 = 15,
    I32 = 16,
    F64 = 17,
    Iq2Xxs = 18,
    Iq2Xs = 19,
    Iq3Xxs = 20,
    Iq1S = 21,
    Iq4Nl = 22,
    Iq3S = 23,
    Iq2S = 24,
    Iq4Xs = 25,
    I64 = 26,
    Iq1M = 27,
    Bf16 = 28,
    Tq1_0 = 29,
    Tq2_0 = 30,
    Count = 31,
}

impl GgmlType {
    /// Size in bytes of one block (`ggml_type_size`).
    pub fn type_size(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => 18,
            GgmlType::Q4_1 => 20,
            GgmlType::Q5_0 => 22,
            GgmlType::Q5_1 => 24,
            GgmlType::Q8_0 => 34,
            GgmlType::Q8_1 => 40,
            GgmlType::Q2K => 84,
            GgmlType::Q3K => 110,
            GgmlType::Q4K => 144,
            GgmlType::Q5K => 176,
            GgmlType::Q6K => 210,
            GgmlType::Q8K => 292,
            GgmlType::I8 => 1,
            GgmlType::I16 => 2,
            GgmlType::I32 => 4,
            GgmlType::F64 => 8,
            GgmlType::Iq2Xxs => 65,
            GgmlType::Iq2Xs => 67,
            GgmlType::Iq3Xxs => 79,
            GgmlType::Iq1S => 65,
            GgmlType::Iq4Nl => 16,
            GgmlType::Iq3S => 79,
            GgmlType::Iq2S => 65,
            GgmlType::Iq4Xs => 83,
            GgmlType::I64 => 8,
            GgmlType::Iq1M => 65,
            GgmlType::Bf16 => 2,
            GgmlType::Tq1_0 => 32,
            GgmlType::Tq2_0 => 64,
            GgmlType::Count => 0,
        }
    }

    /// Number of elements per block (`ggml_blck_size`).
    pub fn blck_size(self) -> i64 {
        match self {
            GgmlType::F32
            | GgmlType::F16
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::F64
            | GgmlType::I64
            | GgmlType::Bf16 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1
            | GgmlType::Q8_0 | GgmlType::Q8_1 | GgmlType::Iq4Nl => 32,
            GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Q8K
            | GgmlType::Iq2Xxs
            | GgmlType::Iq2Xs
            | GgmlType::Iq3Xxs
            | GgmlType::Iq1S
            | GgmlType::Iq3S
            | GgmlType::Iq2S
            | GgmlType::Iq4Xs
            | GgmlType::Iq1M
            | GgmlType::Tq1_0
            | GgmlType::Tq2_0 => 256,
            GgmlType::Count => 0,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "f32",
            GgmlType::F16 => "f16",
            GgmlType::Q4_0 => "q4_0",
            GgmlType::Q4_1 => "q4_1",
            GgmlType::Q5_0 => "q5_0",
            GgmlType::Q5_1 => "q5_1",
            GgmlType::Q8_0 => "q8_0",
            GgmlType::Q8_1 => "q8_1",
            GgmlType::Q2K => "q2_K",
            GgmlType::Q3K => "q3_K",
            GgmlType::Q4K => "q4_K",
            GgmlType::Q5K => "q5_K",
            GgmlType::Q6K => "q6_K",
            GgmlType::Q8K => "q8_K",
            GgmlType::I8 => "i8",
            GgmlType::I16 => "i16",
            GgmlType::I32 => "i32",
            GgmlType::F64 => "f64",
            GgmlType::Iq2Xxs => "iq2_xxs",
            GgmlType::Iq2Xs => "iq2_xs",
            GgmlType::Iq3Xxs => "iq3_xxs",
            GgmlType::Iq1S => "iq1_s",
            GgmlType::Iq4Nl => "iq4_nl",
            GgmlType::Iq3S => "iq3_s",
            GgmlType::Iq2S => "iq2_s",
            GgmlType::Iq4Xs => "iq4_xs",
            GgmlType::I64 => "i64",
            GgmlType::Iq1M => "iq1_m",
            GgmlType::Bf16 => "bf16",
            GgmlType::Tq1_0 => "tq1_0",
            GgmlType::Tq2_0 => "tq2_0",
            GgmlType::Count => "unknown",
        }
    }
}

impl TryFrom<i32> for GgmlType {
    type Error = Error;

    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        if value < 0 || value >= GGML_TYPE_COUNT as i32 {
            return Err(Error::new(format!(
                "invalid ggml type {} (must be in [0, {}))",
                value, GGML_TYPE_COUNT
            )));
        }
        // All values in range have a variant with the same discriminant.
        Ok(ALL_GGML_TYPES[value as usize])
    }
}

const ALL_GGML_TYPES: [GgmlType; GGML_TYPE_COUNT] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
    GgmlType::Q8_1,
    GgmlType::Q2K,
    GgmlType::Q3K,
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::Q8K,
    GgmlType::I8,
    GgmlType::I16,
    GgmlType::I32,
    GgmlType::F64,
    GgmlType::Iq2Xxs,
    GgmlType::Iq2Xs,
    GgmlType::Iq3Xxs,
    GgmlType::Iq1S,
    GgmlType::Iq4Nl,
    GgmlType::Iq3S,
    GgmlType::Iq2S,
    GgmlType::Iq4Xs,
    GgmlType::I64,
    GgmlType::Iq1M,
    GgmlType::Bf16,
    GgmlType::Tq1_0,
    GgmlType::Tq2_0,
];

pub fn ggml_type_size(ty: GgmlType) -> usize {
    ty.type_size()
}

pub fn ggml_blck_size(ty: GgmlType) -> i64 {
    ty.blck_size()
}

pub fn ggml_type_name(ty: GgmlType) -> &'static str {
    ty.name()
}

/// `GGML_PAD(x, alignment)`: round up to a multiple of `alignment`.
pub fn gguf_pad(x: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        return x;
    }
    (x + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// type_to_gguf_type equivalent: encode/decode typed values without unsafe
// ---------------------------------------------------------------------------

/// Maps a Rust value type to its `gguf_type` and raw byte representation.
pub trait GgufValue: Copy {
    const TYPE: GgufType;

    fn encode(self) -> Vec<u8>;
    fn decode(src: &[u8]) -> Self;
}

macro_rules! impl_gguf_value {
    ($($t:ty => $ty:expr),+ $(,)?) => {
        $(
            impl GgufValue for $t {
                const TYPE: GgufType = $ty;

                fn encode(self) -> Vec<u8> {
                    self.to_ne_bytes().to_vec()
                }

                fn decode(src: &[u8]) -> Self {
                    let mut buf = [0u8; std::mem::size_of::<$t>()];
                    buf.copy_from_slice(&src[..buf.len()]);
                    <$t>::from_ne_bytes(buf)
                }
            }
        )+
    };
}

impl_gguf_value!(
    u8 => GgufType::Uint8,
    i8 => GgufType::Int8,
    u16 => GgufType::Uint16,
    i16 => GgufType::Int16,
    u32 => GgufType::Uint32,
    i32 => GgufType::Int32,
    f32 => GgufType::Float32,
    u64 => GgufType::Uint64,
    i64 => GgufType::Int64,
    f64 => GgufType::Float64,
);

// bool values are stored as int8_t on all platforms.
impl GgufValue for bool {
    const TYPE: GgufType = GgufType::Bool;

    fn encode(self) -> Vec<u8> {
        vec![if self { 1 } else { 0 }]
    }

    fn decode(src: &[u8]) -> Self {
        src[0] != 0
    }
}

// ---------------------------------------------------------------------------
// gguf_kv
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GgufKv {
    pub key: String,
    pub is_array: bool,
    pub type_: GgufType,
    pub data: Vec<u8>,
    pub data_string: Vec<String>,
}

impl GgufKv {
    pub fn new_scalar<T: GgufValue>(key: &str, value: T) -> Self {
        assert!(!key.is_empty());
        Self {
            key: key.to_owned(),
            is_array: false,
            type_: T::TYPE,
            data: value.encode(),
            data_string: Vec::new(),
        }
    }

    pub fn new_array<T: GgufValue>(key: &str, value: &[T]) -> Self {
        assert!(!key.is_empty());
        let mut data = Vec::with_capacity(value.len() * std::mem::size_of::<T>());
        for &v in value {
            data.extend_from_slice(&v.encode());
        }
        Self {
            key: key.to_owned(),
            is_array: true,
            type_: T::TYPE,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_string(key: &str, value: &str) -> Self {
        assert!(!key.is_empty());
        Self {
            key: key.to_owned(),
            is_array: false,
            type_: GgufType::String,
            data: Vec::new(),
            data_string: vec![value.to_owned()],
        }
    }

    pub fn new_string_array<S: AsRef<str>>(key: &str, value: &[S]) -> Self {
        assert!(!key.is_empty());
        Self {
            key: key.to_owned(),
            is_array: true,
            type_: GgufType::String,
            data: Vec::new(),
            data_string: value.iter().map(|s| s.as_ref().to_owned()).collect(),
        }
    }

    pub fn get_key(&self) -> &str {
        &self.key
    }

    pub fn get_type(&self) -> GgufType {
        self.type_
    }

    pub fn get_ne(&self) -> usize {
        if self.type_ == GgufType::String {
            let ne = self.data_string.len();
            assert!(self.is_array || ne == 1);
            return ne;
        }
        let type_size = gguf_type_size(self.type_);
        assert!(type_size > 0 && self.data.len() % type_size == 0);
        let ne = self.data.len() / type_size;
        assert!(self.is_array || ne == 1);
        ne
    }

    pub fn get_val<T: GgufValue>(&self, i: usize) -> T {
        assert!(T::TYPE == self.type_);
        let type_size = gguf_type_size(self.type_);
        assert!(type_size > 0 && self.data.len() % type_size == 0);
        assert!(self.data.len() >= (i + 1) * type_size);
        T::decode(&self.data[i * type_size..(i + 1) * type_size])
    }

    pub fn get_str(&self, i: usize) -> &str {
        assert!(self.type_ == GgufType::String);
        assert!(self.data_string.len() > i);
        &self.data_string[i]
    }

    /// `void cast(const enum gguf_type new_type)`.
    pub fn cast(&mut self, new_type: GgufType) {
        let new_type_size = gguf_type_size(new_type);
        assert!(new_type_size > 0 && self.data.len() % new_type_size == 0);
        self.type_ = new_type;
    }
}

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

// ---------------------------------------------------------------------------
// reader
// ---------------------------------------------------------------------------

/// Callback to simulate or wrap a FILE pointer: read up to `output.len()`
/// bytes at `offset` into `output` and return the number of bytes read.
pub type GgufReaderCallback<'a> = dyn FnMut(u64, &mut [u8]) -> usize + 'a;

pub struct GgufReader<F>
where
    F: FnMut(u64, &mut [u8]) -> usize,
{
    callback: F,
    max_chunk_read: usize,
    start_offset: u64,
    data_offset: u64,
    nbytes_remain: u64,
}

impl<F> GgufReader<F>
where
    F: FnMut(u64, &mut [u8]) -> usize,
{
    pub fn new(
        callback: F,
        max_chunk_read: usize,
        data_offset: u64,
        nbytes_remain: u64,
    ) -> Self {
        assert!(max_chunk_read > 0);
        Self {
            callback,
            max_chunk_read,
            start_offset: data_offset,
            data_offset,
            nbytes_remain,
        }
    }

    fn read_raw(&mut self, dst: &mut [u8]) -> usize {
        let size = dst.len();
        if size == 0 {
            return 0;
        }

        let mut total_nread = 0usize;
        let mut reached_eof = false;

        while total_nread < size {
            let chunk_size = self.max_chunk_read.min(size - total_nread);
            if self.data_offset.wrapping_add(total_nread as u64) < self.data_offset {
                break;
            }
            let nread = (self.callback)(
                self.data_offset + total_nread as u64,
                &mut dst[total_nread..total_nread + chunk_size],
            );
            let nread = nread.min(chunk_size);
            total_nread += nread;
            if nread != chunk_size {
                reached_eof = true;
                break;
            }
        }

        self.data_offset += total_nread as u64;
        assert!(total_nread as u64 <= self.nbytes_remain);
        self.nbytes_remain -= total_nread as u64;

        if reached_eof {
            self.nbytes_remain = 0;
        }

        total_nread
    }

    pub fn read_bytes(&mut self, dst: &mut [u8]) -> Result<()> {
        let size = dst.len();
        if size as u64 > self.nbytes_remain {
            return Err(Error::new("gguf_reader: read past remaining bytes"));
        }
        if self.read_raw(dst) != size {
            return Err(Error::new("gguf_reader: failed to read raw bytes"));
        }
        Ok(())
    }

    pub fn read<T: GgufValue>(&mut self) -> Result<T> {
        let mut buf = [0u8; std::mem::size_of::<T>()];
        self.read_bytes(&mut buf)?;
        Ok(T::decode(&buf))
    }

    pub fn read_bool(&mut self) -> Result<bool> {
        let tmp = self.read::<i8>()?;
        Ok(tmp != 0)
    }

    pub fn read_string(&mut self) -> Result<String> {
        let size = self.read::<u64>()?;
        if size > GGUF_MAX_STRING_LENGTH as u64 {
            return Err(Error::new(format!(
                "gguf_reader: string length {} exceeds maximum {}",
                size, GGUF_MAX_STRING_LENGTH
            )));
        }
        if size > self.nbytes_remain {
            return Err(Error::new(format!(
                "gguf_reader: string length {} exceeds remaining file size {} bytes",
                size, self.nbytes_remain
            )));
        }
        let mut buf = vec![0u8; size as usize];
        if self.read_raw(&mut buf) != size as usize {
            return Err(Error::new("gguf_reader: failed to read string"));
        }
        String::from_utf8(buf)
            .map_err(|e| Error::new(format!("gguf_reader: invalid UTF-8 string: {}", e)))
    }

    pub fn read_vec<T: GgufValue>(&mut self, n: usize) -> Result<Vec<T>> {
        if n > GGUF_MAX_ARRAY_ELEMENTS {
            return Err(Error::new("gguf_reader: array too large"));
        }
        let type_size = std::mem::size_of::<T>();
        if type_size > 0 && n > usize::MAX / type_size {
            return Err(Error::new("gguf_reader: array size overflow"));
        }
        let total = n
            .checked_mul(type_size)
            .ok_or_else(|| Error::new("gguf_reader: array size overflow"))?;
        if self.nbytes_remain < total as u64 {
            return Err(Error::new("gguf_reader: array exceeds remaining bytes"));
        }
        let mut dst = Vec::with_capacity(n);
        for _ in 0..n {
            dst.push(self.read::<T>()?);
        }
        Ok(dst)
    }

    pub fn read_bool_vec(&mut self, n: usize) -> Result<Vec<bool>> {
        if n > GGUF_MAX_ARRAY_ELEMENTS {
            return Err(Error::new("gguf_reader: array too large"));
        }
        if self.nbytes_remain < n as u64 {
            return Err(Error::new("gguf_reader: array exceeds remaining bytes"));
        }
        let mut dst = Vec::with_capacity(n);
        for _ in 0..n {
            dst.push(self.read_bool()?);
        }
        Ok(dst)
    }

    pub fn read_string_vec(&mut self, n: usize) -> Result<Vec<String>> {
        if n > GGUF_MAX_ARRAY_ELEMENTS {
            return Err(Error::new("gguf_reader: array too large"));
        }
        // Strings are prefixed with their length, so account for that.
        if n > usize::MAX / std::mem::size_of::<u64>() {
            return Err(Error::new("gguf_reader: string array size overflow"));
        }
        let prefix_bytes = n
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| Error::new("gguf_reader: string array size overflow"))?;
        if self.nbytes_remain < prefix_bytes as u64 {
            return Err(Error::new("gguf_reader: string array exceeds remaining bytes"));
        }
        let mut dst = Vec::with_capacity(n);
        for _ in 0..n {
            dst.push(self.read_string()?);
        }
        Ok(dst)
    }

    /// Current absolute offset in the file/buffer.
    pub fn tell(&self) -> u64 {
        self.data_offset
    }

    /// Position in the file where the GGUF data starts; alignment is relative
    /// to it, not to the file.
    pub fn start(&self) -> u64 {
        self.start_offset
    }

    pub fn seek(&mut self, absolute_offset: u64) -> bool {
        let end_offset = self.data_offset + self.nbytes_remain;
        if absolute_offset > end_offset {
            return false;
        }
        self.data_offset = absolute_offset;
        self.nbytes_remain = end_offset - absolute_offset;
        true
    }
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

pub fn gguf_init_empty() -> GgufContext {
    GgufContext::default()
}

fn gguf_read_emplace_helper<T, F>(
    gr: &mut GgufReader<F>,
    key: &str,
    is_array: bool,
    n: usize,
) -> Result<GgufKv>
where
    T: GgufValue,
    F: FnMut(u64, &mut [u8]) -> usize,
{
    if is_array {
        let value = gr.read_vec::<T>(n)?;
        Ok(GgufKv::new_array(key, &value))
    } else {
        let value = gr.read::<T>()?;
        Ok(GgufKv::new_scalar(key, value))
    }
}

fn gguf_init_from_reader<F>(
    mut gr: GgufReader<F>,
    params: &mut GgufInitParams,
) -> Result<GgufContext>
where
    F: FnMut(u64, &mut [u8]) -> usize,
{
    let mut ctx = GgufContext::default();

    // file magic
    {
        let mut magic = [0u8; 4];
        gr.read_bytes(&mut magic)
            .map_err(|_| Error::new("gguf_init_from_reader: failed to read magic"))?;

        if magic != *GGUF_MAGIC {
            let printable = |c: u8| {
                if c.is_ascii_graphic() || c == b' ' {
                    c as char
                } else {
                    '?'
                }
            };
            return Err(Error::new(format!(
                "gguf_init_from_reader: invalid magic characters: '{}{}{}{}', expected 'GGUF'",
                printable(magic[0]),
                printable(magic[1]),
                printable(magic[2]),
                printable(magic[3]),
            )));
        }
    }

    // header
    ctx.version = gr.read::<u32>()?;
    if ctx.version == 0 {
        return Err(Error::new(format!(
            "gguf_init_from_reader: bad GGUF version: {}",
            ctx.version
        )));
    }

    /*
     * bit layout is different when reading non-native endian models.
     * assuming that the GGUF version is 3, the non-native endian model
     * would read it as 0x30000000. we can use the AND operation against
     * the last 4 hexadecimal digits to check if the model is the same
     * endianness as the host system.
     */
    if (ctx.version & 0x0000_FFFF) == 0x0000_0000 {
        return Err(Error::new(format!(
            "gguf_init_from_reader: failed to load model: this GGUF file version {} is extremely large, is there a mismatch between the host and model endianness?",
            ctx.version
        )));
    }
    if ctx.version == 1 {
        return Err(Error::new(
            "gguf_init_from_reader: GGUFv1 is no longer supported, please use a more up-to-date version",
        ));
    }
    if ctx.version > GGUF_VERSION {
        return Err(Error::new(format!(
            "gguf_init_from_reader: this GGUF file is version {} but this software only supports up to version {}",
            ctx.version, GGUF_VERSION
        )));
    }

    let n_tensors = gr.read::<i64>()?;
    if n_tensors < 0
        || n_tensors as u64 > (usize::MAX / std::mem::size_of::<GgufTensorInfo>()) as u64
    {
        return Err(Error::new(format!(
            "gguf_init_from_reader: number of tensors is {} but must be in [0, {}]",
            n_tensors,
            usize::MAX / std::mem::size_of::<GgufTensorInfo>()
        )));
    }

    let n_kv = gr.read::<i64>()?;
    if n_kv < 0 || n_kv as u64 > (usize::MAX / std::mem::size_of::<GgufKv>()) as u64 {
        return Err(Error::new(format!(
            "gguf_init_from_reader: number of key value pairs is {} but must be in [0, {}]",
            n_kv,
            usize::MAX / std::mem::size_of::<GgufKv>()
        )));
    }

    // KV pairs
    {
        let mut seen_keys = HashSet::new();

        for i in 0..n_kv {
            let key = gr.read_string()?;
            if key.is_empty() {
                return Err(Error::new(format!(
                    "gguf_init_from_reader: key {} is empty",
                    i
                )));
            }
            if !seen_keys.insert(key.clone()) {
                return Err(Error::new(format!(
                    "gguf_init_from_reader: duplicate key '{}' for KV pair {}",
                    key, i
                )));
            }

            let mut type_i = gr.read::<i32>()?;
            let mut is_array = false;
            let mut n = 1usize;

            if type_i == GgufType::Array as i32 {
                is_array = true;
                type_i = gr.read::<i32>()?;
                let n_u64 = gr.read::<u64>()?;
                if n_u64 > GGUF_MAX_ARRAY_ELEMENTS as u64 || n_u64 > usize::MAX as u64 {
                    return Err(Error::new(format!(
                        "gguf_init_from_reader: array length {} for key '{}' is too large",
                        n_u64, key
                    )));
                }
                n = n_u64 as usize;
            }

            let ty = GgufType::try_from(type_i).map_err(|_| {
                Error::new(format!(
                    "gguf_init_from_reader: key '{}' has invalid GGUF type {}",
                    key, type_i
                ))
            })?;

            let kv = match ty {
                GgufType::Uint8 => {
                    gguf_read_emplace_helper::<u8, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Int8 => {
                    gguf_read_emplace_helper::<i8, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Uint16 => {
                    gguf_read_emplace_helper::<u16, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Int16 => {
                    gguf_read_emplace_helper::<i16, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Uint32 => {
                    gguf_read_emplace_helper::<u32, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Int32 => {
                    gguf_read_emplace_helper::<i32, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Float32 => {
                    gguf_read_emplace_helper::<f32, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Uint64 => {
                    gguf_read_emplace_helper::<u64, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Int64 => {
                    gguf_read_emplace_helper::<i64, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Float64 => {
                    gguf_read_emplace_helper::<f64, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Bool => {
                    if is_array {
                        GgufKv::new_array(&key, &gr.read_bool_vec(n)?)
                    } else {
                        GgufKv::new_scalar(&key, gr.read_bool()?)
                    }
                }
                GgufType::String => {
                    if is_array {
                        GgufKv::new_string_array(&key, &gr.read_string_vec(n)?)
                    } else {
                        GgufKv::new_string(&key, &gr.read_string()?)
                    }
                }
                GgufType::Array | GgufType::Count => {
                    return Err(Error::new(format!(
                        "gguf_init_from_reader: key '{}' has invalid GGUF type {}",
                        key, type_i
                    )));
                }
            };
            ctx.kv.push(kv);
        }

        debug_assert_eq!(ctx.kv.len() as i64, n_kv);

        let alignment_idx = gguf_find_key(&ctx, GGUF_KEY_GENERAL_ALIGNMENT);
        if alignment_idx != -1 && gguf_get_kv_type(&ctx, alignment_idx) != GgufType::Uint32 {
            return Err(Error::new(format!(
                "gguf_init_from_reader: key '{}' must be of type {} but is {}",
                GGUF_KEY_GENERAL_ALIGNMENT,
                gguf_type_name(GgufType::Uint32).unwrap_or("unknown"),
                gguf_type_name(gguf_get_kv_type(&ctx, alignment_idx)).unwrap_or("unknown")
            )));
        }
        ctx.alignment = if alignment_idx == -1 {
            GGUF_DEFAULT_ALIGNMENT
        } else {
            gguf_get_val_u32(&ctx, alignment_idx) as usize
        };

        if ctx.alignment == 0 || (ctx.alignment & (ctx.alignment - 1)) != 0 {
            return Err(Error::new(format!(
                "gguf_init_from_reader: alignment {} is not a power of 2",
                ctx.alignment
            )));
        }
    }

    // read the tensor info
    let mut seen_tensor_names = HashSet::new();

    for i in 0..n_tensors {
        let mut t = GgmlTensor::default();

        // tensor name
        let name = gr.read_string()?;
        if name.len() >= GGML_MAX_NAME {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor name {} is too long: {} >= {}",
                i,
                name.len(),
                GGML_MAX_NAME
            )));
        }
        t.name = name.clone();

        // make sure there are no duplicate tensor names
        if !seen_tensor_names.insert(name) {
            return Err(Error::new(format!(
                "gguf_init_from_reader: duplicate tensor name '{}' for tensor {}",
                t.name, i
            )));
        }

        // tensor shape
        let n_dims = gr.read::<u32>()?;
        if n_dims > GGML_MAX_DIMS as u32 {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor '{}' has invalid number of dimensions: {} > {}",
                t.name, n_dims, GGML_MAX_DIMS
            )));
        }
        for j in 0..GGML_MAX_DIMS {
            t.ne[j] = 1;
            if j < n_dims as usize {
                t.ne[j] = gr.read::<i64>()?;
            }

            // check that all ne are non-negative
            if t.ne[j] < 0 {
                return Err(Error::new(format!(
                    "gguf_init_from_reader: tensor '{}' dimension {} has invalid number of elements: {} < 0",
                    t.name, j, t.ne[j]
                )));
            }
        }

        // check that the total number of elements is representable
        // (a zero-element tensor is trivially representable)
        let mut ok = true;
        let mut current_nelements = t.ne[0];
        for j in 1..GGML_MAX_DIMS {
            if t.ne[j] > 1 {
                // 1 or 0 won't cause an overflow
                if i64::MAX / t.ne[j] < current_nelements {
                    ok = false;
                    break;
                }
            }
            current_nelements *= t.ne[j];
        }
        if !ok {
            return Err(Error::new(format!(
                "gguf_init_from_reader: incorrect shape for a tensor '{}' may cause overflow errors: ({}, {}, {}, {}) is >= INT64_MAX ({})",
                t.name, t.ne[0], t.ne[1], t.ne[2], t.ne[3], i64::MAX
            )));
        }

        // tensor type
        let type_i = gr.read::<i32>()?;

        // check that tensor type is within defined range
        let ty = GgmlType::try_from(type_i).map_err(|_| {
            Error::new(format!(
                "gguf_init_from_reader: tensor '{}' has invalid ggml type {}. should be in [0, {})",
                t.name, type_i, GGML_TYPE_COUNT
            ))
        })?;
        let type_size = ggml_type_size(ty);
        let blck_size = ggml_blck_size(ty);

        // check that row size is divisible by block size
        if blck_size == 0 || t.ne[0] % blck_size != 0 {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor '{}' of type {} ({}) has {} elements per row, not a multiple of block size ({})",
                t.name, type_i, ggml_type_name(ty), t.ne[0], blck_size
            )));
        }

        // check that the size of the tensor in bytes is representable
        let nelements = t.ne.iter().try_fold(1i128, |acc, &ne| acc.checked_mul(ne as i128));
        let blocks = match nelements {
            Some(ne) => ne / blck_size as i128,
            None => i128::MAX,
        };
        if blocks > usize::MAX as i128 / type_size.max(1) as i128 {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor '{}' with shape ({}, {}, {}, {}) has a size in bytes > {}",
                t.name, t.ne[0], t.ne[1], t.ne[2], t.ne[3], usize::MAX
            )));
        }

        // calculate byte offsets given the tensor shape and type
        t.type_ = ty;
        t.nb[0] = type_size;
        t.nb[1] = t.nb[0] * (t.ne[0] as usize / blck_size as usize);
        for j in 2..GGML_MAX_DIMS {
            t.nb[j] = t.nb[j - 1] * t.ne[j - 1] as usize;
        }

        // tensor data offset within buffer
        let offset = gr.read::<u64>()?;
        ctx.info.push(GgufTensorInfo { t, offset });
    }

    debug_assert_eq!(ctx.info.len() as i64, n_tensors);

    // we require the data section to be aligned, so take into account any padding
    if n_tensors > 0 {
        let rel = gr.tell() - gr.start();
        let aligned = gr.start() + gguf_pad(rel as usize, ctx.alignment) as u64;
        if !gr.seek(aligned) {
            return Err(Error::new(
                "gguf_init_from_reader: failed to seek to beginning of data section",
            ));
        }
    }

    // store the current file offset - this is where the data section starts
    ctx.offset = gr.tell() as usize;

    // compute the total size of the data section, taking into account the alignment
    for ti in &ctx.info {
        if ti.offset != ctx.size as u64 {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor '{}' has offset {}, expected {}",
                ti.t.name, ti.offset, ctx.size
            )));
        }
        let nbytes = ggml_nbytes(&ti.t);
        // GGML_PAD adds (alignment - 1) to nbytes. When nbytes is near
        // SIZE_MAX this wraps to 0 and makes the check below dead.
        if nbytes > usize::MAX - (ctx.alignment - 1) {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor '{}' size {} overflows after padding (alignment {})",
                ti.t.name, nbytes, ctx.alignment
            )));
        }
        let padded_size = gguf_pad(nbytes, ctx.alignment);
        if usize::MAX - ctx.size < padded_size {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor '{}' size overflow, cannot accumulate size {} + {}",
                ti.t.name, ctx.size, padded_size
            )));
        }
        ctx.size += padded_size;
    }

    // load the tensor data only if requested
    if params.ctx.is_some() {
        // if the provided gguf_context is no_alloc, then we create "empty"
        // tensors and do not read the binary blob; otherwise we load the
        // binary blob and point the tensors at the appropriate locations.

        let mut blob = Vec::new();
        if !params.no_alloc {
            blob.resize(ctx.size, 0u8);
            gr.read_bytes(&mut blob).map_err(|_| {
                Error::new("gguf_init_from_reader: failed to read tensor data binary blob")
            })?;
            ctx.data = blob;
        }

        let mut tensors = Vec::with_capacity(ctx.info.len());
        for info in &ctx.info {
            let mut cur = info.t.clone();
            if !params.no_alloc {
                let start = info.offset as usize;
                let end = start + ggml_nbytes(&info.t);
                cur.data = ctx.data[start..end].to_vec();
            }
            tensors.push(cur);
        }

        params.ctx = Some(GgmlContext {
            tensors,
            no_alloc: params.no_alloc,
        });
    }

    Ok(ctx)
}

/// max_chunk_read == 0 means no limit.
pub fn gguf_init_from_callback<F>(
    callback: F,
    max_chunk_read: usize,
    max_expected_size: u64,
    params: &mut GgufInitParams,
) -> Result<GgufContext>
where
    F: FnMut(u64, &mut [u8]) -> usize,
{
    let gr = GgufReader::new(
        callback,
        if max_chunk_read == 0 {
            usize::MAX
        } else {
            max_chunk_read
        },
        0,
        max_expected_size,
    );
    gguf_init_from_reader(gr, params)
}

struct GgufFileReaderState {
    file: File,
    offset: u64,
}

fn gguf_file_reader_callback(
    state: &mut GgufFileReaderState,
    output: &mut [u8],
    offset: u64,
) -> usize {
    assert!(!output.is_empty());

    if state.offset != offset {
        if state.file.seek(SeekFrom::Start(offset)).is_err() {
            return 0;
        }
        state.offset = offset;
    }

    match state.file.read(output) {
        Ok(nread) => {
            state.offset += nread as u64;
            nread
        }
        Err(_) => 0,
    }
}

/// Remaining bytes from the current file position to EOF.
fn gguf_file_remain(file: &mut File) -> Result<u64> {
    let io_err = |e: std::io::Error| Error::new(e.to_string());
    let cur = file.stream_position().map_err(io_err)?;
    let end = file.seek(SeekFrom::End(0)).map_err(io_err)?;
    file.seek(SeekFrom::Start(cur)).map_err(io_err)?;
    Ok(end.saturating_sub(cur))
}

pub fn gguf_init_from_file_ptr(mut file: File, params: &mut GgufInitParams) -> Result<GgufContext> {
    let cur = file
        .stream_position()
        .map_err(|e| Error::new(e.to_string()))?;
    let remain = gguf_file_remain(&mut file)?;

    let mut state = GgufFileReaderState { file, offset: cur };
    let gr = GgufReader::new(
        |offset, output| gguf_file_reader_callback(&mut state, output, offset),
        usize::MAX,
        cur,
        remain,
    );
    gguf_init_from_reader(gr, params)
}

/// Buffer reader: `data` is the whole backing buffer.
pub fn gguf_init_from_buffer(data: &[u8], params: &mut GgufInitParams) -> Result<GgufContext> {
    if data.is_empty() {
        return Err(Error::new("gguf_init_from_buffer: empty buffer"));
    }

    let gr = GgufReader::new(
        |offset: u64, output: &mut [u8]| -> usize {
            assert!(!output.is_empty());
            if offset > data.len() as u64 || output.len() > data.len() - offset as usize {
                return 0;
            }
            let data_offset = offset as usize;
            let nread = output.len().min(data.len() - data_offset);
            output[..nread].copy_from_slice(&data[data_offset..data_offset + nread]);
            nread
        },
        usize::MAX,
        0,
        data.len() as u64,
    );
    gguf_init_from_reader(gr, params)
}

pub fn gguf_init_from_file(fname: &str, params: &mut GgufInitParams) -> Result<GgufContext> {
    let file = File::open(fname)
        .map_err(|e| Error::new(format!("gguf_init_from_file: failed to open '{}' ({})", fname, e)))?;
    gguf_init_from_file_ptr(file, params)
}

/// Present for API parity. Rust drops the context automatically.
pub fn gguf_free(_ctx: GgufContext) {}

// ---------------------------------------------------------------------------
// getters
// ---------------------------------------------------------------------------

fn key_index(ctx: &GgufContext, key_id: i64) -> usize {
    assert!(key_id >= 0 && (key_id as usize) < ctx.kv.len());
    key_id as usize
}

fn tensor_index(ctx: &GgufContext, tensor_id: i64) -> usize {
    assert!(tensor_id >= 0 && (tensor_id as usize) < ctx.info.len());
    tensor_id as usize
}

pub fn gguf_get_version(ctx: &GgufContext) -> u32 {
    ctx.version
}

pub fn gguf_get_alignment(ctx: &GgufContext) -> usize {
    ctx.alignment
}

/// Padded to `gguf_get_alignment` if and only if the context has a tensor.
pub fn gguf_get_data_offset(ctx: &GgufContext) -> usize {
    ctx.offset
}

pub fn gguf_get_n_kv(ctx: &GgufContext) -> i64 {
    ctx.kv.len() as i64
}

/// Returns -1 if key is not found.
pub fn gguf_find_key(ctx: &GgufContext, key: &str) -> i64 {
    for (i, kv) in ctx.kv.iter().enumerate() {
        if key == kv.get_key() {
            return i as i64;
        }
    }
    -1
}

pub fn gguf_get_key(ctx: &GgufContext, key_id: i64) -> &str {
    ctx.kv[key_index(ctx, key_id)].get_key()
}

pub fn gguf_get_kv_type(ctx: &GgufContext, key_id: i64) -> GgufType {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    if kv.is_array {
        GgufType::Array
    } else {
        kv.get_type()
    }
}

pub fn gguf_get_arr_type(ctx: &GgufContext, key_id: i64) -> GgufType {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    assert!(kv.is_array);
    kv.get_type()
}

fn get_val_helper<T: GgufValue>(ctx: &GgufContext, key_id: i64) -> T {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    assert_eq!(kv.get_ne(), 1);
    kv.get_val(0)
}

/// Will panic if the wrong type is used for the key.
pub fn gguf_get_val_u8(ctx: &GgufContext, key_id: i64) -> u8 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_i8(ctx: &GgufContext, key_id: i64) -> i8 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_u16(ctx: &GgufContext, key_id: i64) -> u16 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_i16(ctx: &GgufContext, key_id: i64) -> i16 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_u32(ctx: &GgufContext, key_id: i64) -> u32 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_i32(ctx: &GgufContext, key_id: i64) -> i32 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_f32(ctx: &GgufContext, key_id: i64) -> f32 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_u64(ctx: &GgufContext, key_id: i64) -> u64 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_i64(ctx: &GgufContext, key_id: i64) -> i64 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_f64(ctx: &GgufContext, key_id: i64) -> f64 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_bool(ctx: &GgufContext, key_id: i64) -> bool {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_str(ctx: &GgufContext, key_id: i64) -> &str {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    assert_eq!(kv.get_ne(), 1);
    kv.get_str(0)
}

pub fn gguf_get_val_data(ctx: &GgufContext, key_id: i64) -> &[u8] {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    assert_eq!(kv.get_ne(), 1);
    assert!(kv.get_type() != GgufType::String);
    &kv.data
}

pub fn gguf_get_arr_n(ctx: &GgufContext, key_id: i64) -> usize {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    if kv.type_ == GgufType::String {
        return kv.data_string.len();
    }
    let type_size = gguf_type_size(kv.type_);
    assert!(type_size > 0 && kv.data.len() % type_size == 0);
    kv.data.len() / type_size
}

/// Raw pointer to the first element of the array in C++. For bool arrays the
/// elements are always stored as int8 on all platforms.
pub fn gguf_get_arr_data(ctx: &GgufContext, key_id: i64) -> &[u8] {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    assert!(kv.get_type() != GgufType::String);
    &kv.data
}

/// Get the ith C string from the array with the given key_id.
pub fn gguf_get_arr_str(ctx: &GgufContext, key_id: i64, i: usize) -> &str {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    assert!(kv.get_type() == GgufType::String);
    &kv.data_string[i]
}

pub fn gguf_get_n_tensors(ctx: &GgufContext) -> i64 {
    ctx.info.len() as i64
}

/// Returns -1 if the tensor is not found.
pub fn gguf_find_tensor(ctx: &GgufContext, name: &str) -> i64 {
    for (i, info) in ctx.info.iter().enumerate() {
        if name == info.t.name {
            return i as i64;
        }
    }
    -1
}

pub fn gguf_get_tensor_offset(ctx: &GgufContext, tensor_id: i64) -> usize {
    ctx.info[tensor_index(ctx, tensor_id)].offset as usize
}

pub fn gguf_get_tensor_name(ctx: &GgufContext, tensor_id: i64) -> &str {
    &ctx.info[tensor_index(ctx, tensor_id)].t.name
}

/// `ne[dim]` is 1 for dim >= n_dims.
pub fn gguf_get_tensor_ne(ctx: &GgufContext, tensor_id: i64) -> &[i64; GGML_MAX_DIMS] {
    &ctx.info[tensor_index(ctx, tensor_id)].t.ne
}

pub fn gguf_get_tensor_type(ctx: &GgufContext, tensor_id: i64) -> GgmlType {
    ctx.info[tensor_index(ctx, tensor_id)].t.type_
}

pub fn gguf_get_tensor_size(ctx: &GgufContext, tensor_id: i64) -> usize {
    ggml_nbytes(&ctx.info[tensor_index(ctx, tensor_id)].t)
}

// ---------------------------------------------------------------------------
// setters
// ---------------------------------------------------------------------------

/// Removes key if it exists, returns the id the key had prior to removal.
pub fn gguf_remove_key(ctx: &mut GgufContext, key: &str) -> i64 {
    let key_id = gguf_find_key(ctx, key);
    if key_id >= 0 {
        ctx.kv.remove(key_id as usize);
    }
    key_id
}

fn check_reserved_key(key: &str) {
    assert!(
        key != GGUF_KEY_GENERAL_ALIGNMENT,
        "general.alignment must be type u32"
    );
}

macro_rules! define_set_val {
    ($name:ident, $t:ty, $ctor:ident) => {
        pub fn $name(ctx: &mut GgufContext, key: &str, val: $t) {
            check_reserved_key(key);
            gguf_remove_key(ctx, key);
            ctx.kv.push(GgufKv::$ctor(key, val));
        }
    };
}

define_set_val!(gguf_set_val_u8, u8, new_scalar);
define_set_val!(gguf_set_val_i8, i8, new_scalar);
define_set_val!(gguf_set_val_u16, u16, new_scalar);
define_set_val!(gguf_set_val_i16, i16, new_scalar);
define_set_val!(gguf_set_val_i32, i32, new_scalar);
define_set_val!(gguf_set_val_f32, f32, new_scalar);
define_set_val!(gguf_set_val_u64, u64, new_scalar);
define_set_val!(gguf_set_val_i64, i64, new_scalar);
define_set_val!(gguf_set_val_f64, f64, new_scalar);
define_set_val!(gguf_set_val_bool, bool, new_scalar);

pub fn gguf_set_val_u32(ctx: &mut GgufContext, key: &str, val: u32) {
    if key == GGUF_KEY_GENERAL_ALIGNMENT {
        assert!(
            val > 0 && (val & (val - 1)) == 0,
            "general.alignment must be power of 2"
        );
    }
    gguf_remove_key(ctx, key);
    ctx.kv.push(GgufKv::new_scalar(key, val));
}

pub fn gguf_set_val_str(ctx: &mut GgufContext, key: &str, val: &str) {
    check_reserved_key(key);
    gguf_remove_key(ctx, key);
    ctx.kv.push(GgufKv::new_string(key, val));
}

/// Creates a new array with n elements of the given type and copies
/// `n * gguf_type_size(type)` bytes from `data`.
pub fn gguf_set_arr_data(ctx: &mut GgufContext, key: &str, ty: GgufType, data: &[u8], n: usize) {
    check_reserved_key(key);
    gguf_remove_key(ctx, key);

    let nbytes = n * gguf_type_size(ty);
    assert!(data.len() >= nbytes);
    let mut kv = GgufKv::new_array(key, &data[..nbytes].to_vec());
    // gguf_kv::cast(new_type)
    if gguf_type_size(ty) > 0 {
        kv.cast(ty);
    } else {
        kv.type_ = ty;
    }
    ctx.kv.push(kv);
}

/// Creates a new array with n strings and copies the corresponding strings.
pub fn gguf_set_arr_str(ctx: &mut GgufContext, key: &str, data: &[&str], n: usize) {
    check_reserved_key(key);
    gguf_remove_key(ctx, key);
    ctx.kv.push(GgufKv::new_string_array(key, &data[..n]));
}

/// Set or add KV pairs from another context.
pub fn gguf_set_kv(ctx: &mut GgufContext, src: &GgufContext) {
    for kv in &src.kv {
        let key = kv.get_key();

        if !kv.is_array {
            match kv.get_type() {
                GgufType::Uint8 => gguf_set_val_u8(ctx, key, kv.get_val(0)),
                GgufType::Int8 => gguf_set_val_i8(ctx, key, kv.get_val(0)),
                GgufType::Uint16 => gguf_set_val_u16(ctx, key, kv.get_val(0)),
                GgufType::Int16 => gguf_set_val_i16(ctx, key, kv.get_val(0)),
                GgufType::Uint32 => gguf_set_val_u32(ctx, key, kv.get_val(0)),
                GgufType::Int32 => gguf_set_val_i32(ctx, key, kv.get_val(0)),
                GgufType::Float32 => gguf_set_val_f32(ctx, key, kv.get_val(0)),
                GgufType::Uint64 => gguf_set_val_u64(ctx, key, kv.get_val(0)),
                GgufType::Int64 => gguf_set_val_i64(ctx, key, kv.get_val(0)),
                GgufType::Float64 => gguf_set_val_f64(ctx, key, kv.get_val(0)),
                GgufType::Bool => gguf_set_val_bool(ctx, key, kv.get_val(0)),
                GgufType::String => gguf_set_val_str(ctx, key, kv.get_str(0)),
                GgufType::Array | GgufType::Count => panic!("invalid type"),
            }
            continue;
        }

        let ne = kv.get_ne();
        match kv.get_type() {
            GgufType::Uint8
            | GgufType::Int8
            | GgufType::Uint16
            | GgufType::Int16
            | GgufType::Uint32
            | GgufType::Int32
            | GgufType::Float32
            | GgufType::Uint64
            | GgufType::Int64
            | GgufType::Float64
            | GgufType::Bool => {
                gguf_set_arr_data(ctx, key, kv.get_type(), &kv.data, ne);
            }
            GgufType::String => {
                let tmp: Vec<&str> = (0..ne).map(|j| kv.get_str(j)).collect();
                gguf_set_arr_str(ctx, key, &tmp, ne);
            }
            GgufType::Array | GgufType::Count => panic!("invalid type"),
        }
    }
}

/// Add a tensor to the GGUF context; the tensor name must be unique.
pub fn gguf_add_tensor(ctx: &mut GgufContext, tensor: &GgmlTensor) {
    assert!(!tensor.name.is_empty());
    if gguf_find_tensor(ctx, &tensor.name) != -1 {
        panic!("duplicate tensor name: {}", tensor.name);
    }

    let offset = if ctx.info.is_empty() {
        0
    } else {
        let last = ctx.info.last().unwrap();
        last.offset + gguf_pad(ggml_nbytes(&last.t), ctx.alignment) as u64
    };
    ctx.info.push(GgufTensorInfo {
        t: tensor.clone(),
        offset,
    });
}

/// After changing a tensor's type, offsets of all tensors with higher indices
/// are immediately recalculated so the data stays one contiguous block.
pub fn gguf_set_tensor_type(ctx: &mut GgufContext, name: &str, ty: GgmlType) {
    let tensor_id = gguf_find_tensor(ctx, name);
    assert!(tensor_id >= 0, "tensor not found: {}", name);
    let tensor_id = tensor_id as usize;

    let type_size = ggml_type_size(ty);
    let blck_size = ggml_blck_size(ty);

    {
        let t = &mut ctx.info[tensor_id].t;
        t.type_ = ty;
        assert!(
            blck_size > 0 && t.ne[0] % blck_size == 0,
            "tensor row size not divisible by block size of new type"
        );

        t.nb[0] = type_size;
        t.nb[1] = t.nb[0] * (t.ne[0] as usize / blck_size.max(1) as usize);
        for j in 2..GGML_MAX_DIMS {
            t.nb[j] = t.nb[j - 1] * t.ne[j - 1] as usize;
        }
    }

    // update offsets
    let n_tensors = gguf_get_n_tensors(ctx) as usize;
    for i in (tensor_id + 1)..n_tensors {
        let prev_nbytes = ggml_nbytes(&ctx.info[i - 1].t);
        ctx.info[i].offset =
            ctx.info[i - 1].offset + gguf_pad(prev_nbytes, ctx.alignment) as u64;
    }
}

/// Assumes that at least `gguf_get_tensor_size` bytes can be read from data.
pub fn gguf_set_tensor_data(ctx: &mut GgufContext, name: &str, data: &[u8]) {
    let tensor_id = gguf_find_tensor(ctx, name);
    assert!(tensor_id >= 0, "tensor not found: {}", name);
    ctx.info[tensor_id as usize].t.data = data.to_vec();
}

// ---------------------------------------------------------------------------
// writer
// ---------------------------------------------------------------------------

pub trait GgufWriter {
    fn write_i8(&mut self, val: i8) -> Result<()>;
    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()>;
    fn written_bytes(&self) -> usize;

    fn write<T: GgufValue>(&mut self, val: T) -> Result<()> {
        self.write_bytes(&val.encode())
    }

    fn write_bool(&mut self, val: bool) -> Result<()> {
        self.write_i8(if val { 1 } else { 0 })
    }

    fn write_string(&mut self, val: &str) -> Result<()> {
        self.write(val.len() as u64)?;
        self.write_bytes(val.as_bytes())
    }

    fn write_guf_type(&mut self, val: GgufType) -> Result<()> {
        self.write(val as i32)
    }

    fn write_ggml_type(&mut self, val: GgmlType) -> Result<()> {
        self.write(val as i32)
    }

    fn write_kv(&mut self, kv: &GgufKv) -> Result<()> {
        let ne = kv.get_ne() as u64;

        self.write_string(kv.get_key())?;

        if kv.is_array {
            self.write_guf_type(GgufType::Array)?;
            self.write_guf_type(kv.get_type())?;
            self.write(ne)?;
        } else {
            self.write_guf_type(kv.get_type())?;
        }

        match kv.get_type() {
            GgufType::Uint8
            | GgufType::Int8
            | GgufType::Uint16
            | GgufType::Int16
            | GgufType::Uint32
            | GgufType::Int32
            | GgufType::Float32
            | GgufType::Uint64
            | GgufType::Int64
            | GgufType::Float64 => {
                self.write_bytes(&kv.data)?;
            }
            GgufType::Bool => {
                for i in 0..ne as usize {
                    self.write_bool(kv.get_val(i))?;
                }
            }
            GgufType::String => {
                for i in 0..ne as usize {
                    self.write_string(kv.get_str(i))?;
                }
            }
            GgufType::Array | GgufType::Count => panic!("invalid type"),
        }
        Ok(())
    }

    fn write_tensor_meta(&mut self, info: &GgufTensorInfo) -> Result<()> {
        self.write_string(&info.t.name)?;

        let n_dims = ggml_n_dims(&info.t);
        self.write(n_dims as u32)?;

        for j in 0..n_dims {
            self.write(info.t.ne[j])?;
        }
        self.write_ggml_type(info.t.type_)?;
        self.write(info.offset)?;
        Ok(())
    }

    fn pad(&mut self, alignment: usize) -> Result<()> {
        while self.written_bytes() % alignment != 0 {
            self.write_i8(0)?;
        }
        Ok(())
    }

    fn write_tensor_data(
        &mut self,
        info: &GgufTensorInfo,
        offset_data: usize,
        alignment: usize,
    ) -> Result<()> {
        assert_eq!(self.written_bytes() - offset_data, info.offset as usize);
        assert!(ggml_is_contiguous(&info.t));

        let nbytes = ggml_nbytes(&info.t);
        let mut bytes = vec![0u8; nbytes];
        let n = nbytes.min(info.t.data.len());
        bytes[..n].copy_from_slice(&info.t.data[..n]);
        self.write_bytes(&bytes)?;

        self.pad(alignment)
    }
}

/// Vector buffer based writer.
pub struct GgufWriterBuf {
    pub buf: Vec<u8>,
}

impl GgufWriterBuf {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }
}

impl Default for GgufWriterBuf {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter for GgufWriterBuf {
    fn write_i8(&mut self, val: i8) -> Result<()> {
        self.buf.push(val as u8);
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    fn written_bytes(&self) -> usize {
        self.buf.len()
    }
}

/// File based writer.
pub struct GgufWriterFile {
    pub file: File,
    written: usize,
}

impl GgufWriterFile {
    pub fn new(file: File) -> Self {
        Self { file, written: 0 }
    }
}

impl GgufWriter for GgufWriterFile {
    fn write_i8(&mut self, val: i8) -> Result<()> {
        self.file
            .write_all(&[val as u8])
            .map_err(|e| Error::new(e.to_string()))?;
        self.written += 1;
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.file
            .write_all(bytes)
            .map_err(|e| Error::new(e.to_string()))?;
        self.written += bytes.len();
        Ok(())
    }

    fn written_bytes(&self) -> usize {
        self.written
    }
}

pub fn gguf_write_out<W: GgufWriter>(
    ctx: &GgufContext,
    gw: &mut W,
    only_meta: bool,
) -> Result<()> {
    let n_kv = gguf_get_n_kv(ctx);
    let n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.write_bytes(GGUF_MAGIC)?;
    gw.write(ctx.version)?;
    gw.write(n_tensors)?;
    gw.write(n_kv)?;

    // write key-value pairs
    for kv in &ctx.kv {
        gw.write_kv(kv)?;
    }

    // write tensor info
    for info in &ctx.info {
        gw.write_tensor_meta(info)?;
    }

    // we require the data section to be aligned
    gw.pad(ctx.alignment)?;

    if only_meta {
        return Ok(());
    }

    let offset_data = gw.written_bytes();

    // write tensor data
    for info in &ctx.info {
        gw.write_tensor_data(info, offset_data, ctx.alignment)?;
    }

    Ok(())
}

pub fn gguf_write_to_buf(ctx: &GgufContext, only_meta: bool) -> Vec<u8> {
    let mut gw = GgufWriterBuf::new();
    gguf_write_out(ctx, &mut gw, only_meta).expect("gguf buffer write failed");
    gw.buf
}

pub fn gguf_write_to_file_ptr(ctx: &GgufContext, file: File, only_meta: bool) -> bool {
    let mut gw = GgufWriterFile::new(file);
    match gguf_write_out(ctx, &mut gw, only_meta) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("gguf_write_to_file_ptr: failed to write GGUF data: {}", e);
            false
        }
    }
}

pub fn gguf_write_to_file(ctx: &GgufContext, fname: &str, only_meta: bool) -> bool {
    let file = match File::create(fname) {
        Ok(file) => file,
        Err(e) => {
            eprintln!(
                "gguf_write_to_file: failed to open file '{}' for writing GGUF data ({})",
                fname, e
            );
            return false;
        }
    };

    let success = gguf_write_to_file_ptr(ctx, file, only_meta);
    if !success {
        eprintln!(
            "gguf_write_to_file: failed to write GGUF data into '{}'",
            fname
        );
    }
    success
}

/// Size in bytes of the meta data (header, KV pairs, tensor info) including
/// padding.
pub fn gguf_get_meta_size(ctx: &GgufContext) -> usize {
    gguf_write_to_buf(ctx, true).len()
}

/// Writes the meta data to a freshly allocated byte vector.
pub fn gguf_get_meta_data(ctx: &GgufContext) -> Vec<u8> {
    gguf_write_to_buf(ctx, true)
}

// ---------------------------------------------------------------------------
// tests for pure helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_type_size_and_name_table() {
        assert_eq!(gguf_type_size(GgufType::Uint8), 1);
        assert_eq!(gguf_type_size(GgufType::Float32), 4);
        assert_eq!(gguf_type_size(GgufType::Bool), 1);
        assert_eq!(gguf_type_size(GgufType::String), 0);
        assert_eq!(gguf_type_size(GgufType::Array), 0);
        assert_eq!(gguf_type_size(GgufType::Float64), 8);
        assert_eq!(gguf_type_name(GgufType::Bool), Some("bool"));
        assert_eq!(gguf_type_name(GgufType::Count), None);
        assert_eq!(GgufType::Count as i32, 13);
    }

    #[test]
    fn pad_rounds_up_to_alignment() {
        assert_eq!(gguf_pad(0, 32), 0);
        assert_eq!(gguf_pad(1, 32), 32);
        assert_eq!(gguf_pad(32, 32), 32);
        assert_eq!(gguf_pad(33, 32), 64);
        assert_eq!(gguf_pad(7, 8), 8);
    }

    #[test]
    fn ggml_type_size_matches_block_math() {
        let t = GgmlType::Q4_0;
        assert_eq!(ggml_type_size(t), 18);
        assert_eq!(ggml_blck_size(t), 32);
        let mut tensor = GgmlTensor::default();
        tensor.type_ = t;
        tensor.ne = [32, 1, 1, 1];
        assert_eq!(ggml_nbytes(&tensor), 18);
    }
}

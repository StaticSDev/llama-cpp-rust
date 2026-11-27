//! GGUF/ggml type constants, enums, and type-size/name tables.

use std::convert::TryFrom;

use crate::core::Error;

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
pub const GGML_TYPE_COUNT: usize = 44;

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
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
    F8E4M3 = 43,
    Count = 44,
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
            GgmlType::Mxfp4 => 17,
            GgmlType::Nvfp4 => 36,
            GgmlType::Q1_0 => 18,
            GgmlType::Q2_0 => 18,
            GgmlType::F8E4M3 => 1,
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
            | GgmlType::Bf16
            | GgmlType::F8E4M3 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1
            | GgmlType::Q8_0 | GgmlType::Q8_1 | GgmlType::Iq4Nl | GgmlType::Mxfp4 => 32,
            GgmlType::Nvfp4 | GgmlType::Q2_0 => 64,
            GgmlType::Q1_0 => 128,
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
            GgmlType::Mxfp4 => "mxfp4",
            GgmlType::Nvfp4 => "nvfp4",
            GgmlType::Q1_0 => "q1_0",
            GgmlType::Q2_0 => "q2_0",
            GgmlType::F8E4M3 => "f8_e4m3",
            GgmlType::Count => "unknown",
        }
    }
}

impl TryFrom<i32> for GgmlType {
    type Error = Error;

    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        let ty = match value {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            15 => GgmlType::Q8K,
            16 => GgmlType::Iq2Xxs,
            17 => GgmlType::Iq2Xs,
            18 => GgmlType::Iq3Xxs,
            19 => GgmlType::Iq1S,
            20 => GgmlType::Iq4Nl,
            21 => GgmlType::Iq3S,
            22 => GgmlType::Iq2S,
            23 => GgmlType::Iq4Xs,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            29 => GgmlType::Iq1M,
            30 => GgmlType::Bf16,
            34 => GgmlType::Tq1_0,
            35 => GgmlType::Tq2_0,
            39 => GgmlType::Mxfp4,
            40 => GgmlType::Nvfp4,
            41 => GgmlType::Q1_0,
            42 => GgmlType::Q2_0,
            43 => GgmlType::F8E4M3,
            _ => {
                return Err(Error::new(format!(
                    "invalid ggml type {value} (removed or out of range)"
                )));
            }
        };
        Ok(ty)
    }
}

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
                    let n = std::mem::size_of::<$t>();
                    let mut buf = [0u8; std::mem::size_of::<$t>()];
                    buf.copy_from_slice(&src[..n]);
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

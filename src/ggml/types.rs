//! Core ggml types: constants, enums, structs and header helpers.
//!
//! Ported from `ggml.h`, `ggml-impl.h` and `ggml-common.h`. Opaque C pointers
//! become `usize` handles (`0` is null), `char[]` becomes `String`, and C unions
//! become pairs of fields with accessors.

use std::mem::size_of;

use crate::core::{Error, Result};

// ---------------------------------------------------------------------------
// ggml.h constants
// ---------------------------------------------------------------------------

pub const GGML_FILE_MAGIC: u32 = 0x6767_6d6c;
pub const GGML_FILE_VERSION: i32 = 2;
pub const GGML_QNT_VERSION: i32 = 2;
pub const GGML_QNT_VERSION_FACTOR: i32 = 1000;

pub const GGML_MAX_DIMS: usize = 4;
pub const GGML_MAX_PARAMS: usize = 2048;
pub const GGML_MAX_SRC: usize = 10;
pub const GGML_MAX_N_THREADS: usize = 512;
pub const GGML_MAX_OP_PARAMS: usize = 64;
pub const GGML_MAX_NAME: usize = 64;
pub const GGML_DEFAULT_N_THREADS: i32 = 4;
pub const GGML_DEFAULT_GRAPH_SIZE: usize = 2048;
/// 64-bit targets use 16-byte alignment (32-bit uses 4, wasm32 uses 8).
pub const GGML_MEM_ALIGN: usize = 16;

pub const GGML_EXIT_SUCCESS: i32 = 0;
pub const GGML_EXIT_ABORTED: i32 = 1;

pub const GGML_ROPE_TYPE_NORMAL: i32 = 0;
pub const GGML_ROPE_TYPE_NEOX: i32 = 2;
pub const GGML_ROPE_TYPE_MROPE: i32 = 8;
pub const GGML_ROPE_TYPE_VISION: i32 = 24;
pub const GGML_ROPE_TYPE_IMROPE: i32 = 40;
pub const GGML_MROPE_SECTIONS: usize = 4;

pub const GGML_TENSOR_FLAG_INPUT: i32 = 1;
pub const GGML_TENSOR_FLAG_OUTPUT: i32 = 2;
pub const GGML_TENSOR_FLAG_PARAM: i32 = 4;
pub const GGML_TENSOR_FLAG_LOSS: i32 = 8;
pub const GGML_TENSOR_FLAG_COMPUTE: i32 = 16;

/// `GGML_PAD(x, n)`: round `x` up to a multiple of `n`.
pub fn ggml_pad(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}

// ---------------------------------------------------------------------------
// ggml-impl.h constants
// ---------------------------------------------------------------------------

/// Alignment guaranteed by GGUF for mmap'd tensors.
pub const TENSOR_ALIGNMENT: usize = 32;
pub const BITSET_SHR: usize = 5;
pub const BITSET_MASK: usize = 31;
pub const GGML_HASHSET_FULL: usize = usize::MAX;
pub const GGML_HASHSET_ALREADY_EXISTS: usize = usize::MAX - 1;

// ---------------------------------------------------------------------------
// ggml-common.h constants
// ---------------------------------------------------------------------------

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
pub const IQ3S_N_SCALE: usize = QK_K / 64;

pub const QK1_0: usize = 128;
pub const QK2_0: usize = 64;
pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK_MXFP4: usize = 32;
pub const QK_NVFP4: usize = 64;
pub const QK_NVFP4_SUB: usize = 16;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;
pub const QK4_NL: usize = 32;

pub const QR1_0: usize = 1;
pub const QI1_0: usize = QK1_0 / 32;
pub const QR2_0: usize = 1;
pub const QI2_0: usize = QK2_0 / 32;
pub const QR4_0: usize = 2;
pub const QI4_0: usize = QK4_0 / (4 * QR4_0);
pub const QR4_1: usize = 2;
pub const QI4_1: usize = QK4_1 / (4 * QR4_1);
pub const QR_MXFP4: usize = 2;
pub const QI_MXFP4: usize = QK_MXFP4 / (4 * QR_MXFP4);
pub const QR_NVFP4: usize = 2;
pub const QI_NVFP4: usize = QK_NVFP4 / (4 * QR_NVFP4);
pub const QR5_0: usize = 2;
pub const QI5_0: usize = QK5_0 / (4 * QR5_0);
pub const QR5_1: usize = 2;
pub const QI5_1: usize = QK5_1 / (4 * QR5_1);
pub const QR8_0: usize = 1;
pub const QI8_0: usize = QK8_0 / (4 * QR8_0);
pub const QR8_1: usize = 1;
pub const QI8_1: usize = QK8_1 / (4 * QR8_1);
pub const QR2_K: usize = 4;
pub const QI2_K: usize = QK_K / (4 * QR2_K);
pub const QR3_K: usize = 4;
pub const QI3_K: usize = QK_K / (4 * QR3_K);
pub const QR4_K: usize = 2;
pub const QI4_K: usize = QK_K / (4 * QR4_K);
pub const QR5_K: usize = 2;
pub const QI5_K: usize = QK_K / (4 * QR5_K);
pub const QR6_K: usize = 2;
pub const QI6_K: usize = QK_K / (4 * QR6_K);
pub const QR2_XXS: usize = 4;
pub const QI2_XXS: usize = QK_K / (4 * QR2_XXS);
pub const QR2_XS: usize = 4;
pub const QI2_XS: usize = QK_K / (4 * QR2_XS);
pub const QR2_S: usize = 4;
pub const QI2_S: usize = QK_K / (4 * QR2_S);
pub const QR3_XXS: usize = 4;
pub const QI3_XXS: usize = QK_K / (4 * QR3_XXS);
pub const QR3_XS: usize = 4;
pub const QI3_XS: usize = QK_K / (4 * QR3_XS);
pub const QR1_S: usize = 8;
pub const QI1_S: usize = QK_K / (4 * QR1_S);
pub const QR1_M: usize = 8;
pub const QI1_M: usize = QK_K / (4 * QR1_M);
pub const QR4_NL: usize = 2;
pub const QI4_NL: usize = QK4_NL / (4 * QR4_NL);
pub const QR4_XS: usize = 2;
pub const QI4_XS: usize = QK_K / (4 * QR4_XS);
pub const QR3_S: usize = 4;
pub const QI3_S: usize = QK_K / (4 * QR3_S);

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// `enum ggml_status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlStatus {
    AllocFailed = -2,
    Failed = -1,
    Success = 0,
    Aborted = 1,
}

/// `enum ggml_type`. Gap values are retired types kept out of the enum.
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

pub const GGML_TYPE_COUNT: usize = GgmlType::Count as usize;

/// `enum ggml_prec`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlPrec {
    Undefined = 0,
    F32 = 10,
    Bf16 = 15,
    F16 = 20,
    Q8 = 30,
    Q4 = 40,
}

/// `enum ggml_op_hint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlOpHint {
    None = 0,
    Src0IsHadamard = 1,
}

/// `enum ggml_ftype` (deprecated in upstream).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlFtype {
    Unknown = -1,
    AllF32 = 0,
    MostlyF16 = 1,
    MostlyQ4_0 = 2,
    MostlyQ4_1 = 3,
    MostlyQ4_1SomeF16 = 4,
    MostlyQ8_0 = 7,
    MostlyQ5_0 = 8,
    MostlyQ5_1 = 9,
    MostlyQ2K = 10,
    MostlyQ3K = 11,
    MostlyQ4K = 12,
    MostlyQ5K = 13,
    MostlyQ6K = 14,
    MostlyIq2Xxs = 15,
    MostlyIq2Xs = 16,
    MostlyIq3Xxs = 17,
    MostlyIq1S = 18,
    MostlyIq4Nl = 19,
    MostlyIq3S = 20,
    MostlyIq2S = 21,
    MostlyIq4Xs = 22,
    MostlyIq1M = 23,
    MostlyBf16 = 24,
    MostlyMxfp4 = 25,
    MostlyNvfp4 = 26,
    MostlyQ1_0 = 27,
    MostlyQ2_0 = 28,
    MostlyF8E4M3 = 29,
}

/// `enum ggml_op`. Values follow declaration order in `ggml.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlOp {
    None = 0,
    Dup = 1,
    Add = 2,
    AddId = 3,
    Acc = 4,
    Sub = 5,
    Mul = 6,
    Div = 7,
    Sqr = 8,
    Sqrt = 9,
    Log = 10,
    Sin = 11,
    Cos = 12,
    Sum = 13,
    SumRows = 14,
    Cumsum = 15,
    Mean = 16,
    Argmax = 17,
    CountEqual = 18,
    Repeat = 19,
    RepeatBack = 20,
    Concat = 21,
    SiluBack = 22,
    Norm = 23,
    RmsNorm = 24,
    RmsNormBack = 25,
    GroupNorm = 26,
    L2Norm = 27,
    MulMat = 28,
    MulMatId = 29,
    OutProd = 30,
    Scale = 31,
    Set = 32,
    Cpy = 33,
    Cont = 34,
    Reshape = 35,
    View = 36,
    Permute = 37,
    Transpose = 38,
    GetRows = 39,
    GetRowsBack = 40,
    SetRows = 41,
    Diag = 42,
    SoftMax = 43,
    SoftMaxBack = 44,
    Rope = 45,
    RopeBack = 46,
    Clamp = 47,
    ConvTranspose1d = 48,
    Im2col = 49,
    Im2colBack = 50,
    Im2col3d = 51,
    Col2im1d = 52,
    Conv2d = 53,
    Conv3d = 54,
    Conv2dDw = 55,
    ConvTranspose2d = 56,
    Pool1d = 57,
    Pool2d = 58,
    Pool2dBack = 59,
    Upscale = 60,
    Pad = 61,
    PadReflect1d = 62,
    Roll = 63,
    Arange = 64,
    TimestepEmbedding = 65,
    Argsort = 66,
    TopK = 67,
    LeakyRelu = 68,
    Tri = 69,
    Fill = 70,
    FlashAttnExt = 71,
    FlashAttnBack = 72,
    SsmConv = 73,
    SsmScan = 74,
    WinPart = 75,
    WinUnpart = 76,
    GetRelPos = 77,
    AddRelPos = 78,
    RwkvWkv6 = 79,
    GatedLinearAttn = 80,
    RwkvWkv7 = 81,
    SolveTri = 82,
    GatedDeltaNet = 83,
    LightningIndexer = 84,
    Dsv4HcComb = 85,
    Dsv4HcPre = 86,
    Dsv4HcPost = 87,
    TopKMask = 88,
    Unary = 89,
    MapCustom1 = 90,
    MapCustom2 = 91,
    MapCustom3 = 92,
    Custom = 93,
    CrossEntropyLoss = 94,
    CrossEntropyLossBack = 95,
    OptStepAdamw = 96,
    OptStepSgd = 97,
    Glu = 98,
    Sleep = 99,
    SigmoidBack = 100,
    Count = 101,
}

/// `enum ggml_unary_op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlUnaryOp {
    Abs = 0,
    Sgn = 1,
    Neg = 2,
    Step = 3,
    Tanh = 4,
    Elu = 5,
    Relu = 6,
    Sigmoid = 7,
    Gelu = 8,
    GeluQuick = 9,
    Silu = 10,
    Hardswish = 11,
    Hardsigmoid = 12,
    Exp = 13,
    Expm1 = 14,
    Softplus = 15,
    GeluErf = 16,
    Xielu = 17,
    Floor = 18,
    Ceil = 19,
    Round = 20,
    Trunc = 21,
    Count = 22,
}

/// `enum ggml_glu_op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlGluOp {
    Reglu = 0,
    Geglu = 1,
    Swiglu = 2,
    SwigluOai = 3,
    GegluErf = 4,
    GegluQuick = 5,
    SwigluClamp = 6,
    Count = 7,
}

/// `enum ggml_object_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlObjectType {
    Tensor = 0,
    Graph = 1,
    WorkBuffer = 2,
}

/// `enum ggml_log_level`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlLogLevel {
    None = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Cont = 5,
}

/// `enum ggml_tensor_flag`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlTensorFlag {
    Input = 1,
    Output = 2,
    Param = 4,
    Loss = 8,
    Compute = 16,
}

/// `enum ggml_tri_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlTriType {
    UpperDiag = 0,
    Upper = 1,
    LowerDiag = 2,
    Lower = 3,
}

/// `enum ggml_cgraph_eval_order` from `ggml-impl.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlCgraphEvalOrder {
    LeftToRight = 0,
    RightToLeft = 1,
    Count = 2,
}

macro_rules! enum_try_from_i32 {
    ($ty:ident: $($variant:ident),+ $(,)?) => {
        impl TryFrom<i32> for $ty {
            type Error = Error;

            fn try_from(value: i32) -> std::result::Result<Self, Error> {
                match value {
                    $(v if v == Self::$variant as i32 => Ok(Self::$variant),)+
                    _ => Err(Error::new(format!("invalid {} value {}", stringify!($ty), value))),
                }
            }
        }
    };
}

enum_try_from_i32!(GgmlStatus: AllocFailed, Failed, Success, Aborted);
enum_try_from_i32!(GgmlType: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K,
    Q5K, Q6K, Q8K, Iq2Xxs, Iq2Xs, Iq3Xxs, Iq1S, Iq4Nl, Iq3S, Iq2S, Iq4Xs, I8, I16, I32, I64,
    F64, Iq1M, Bf16, Tq1_0, Tq2_0, Mxfp4, Nvfp4, Q1_0, Q2_0, F8E4M3, Count);
enum_try_from_i32!(GgmlPrec: Undefined, F32, Bf16, F16, Q8, Q4);
enum_try_from_i32!(GgmlOpHint: None, Src0IsHadamard);
enum_try_from_i32!(GgmlFtype: Unknown, AllF32, MostlyF16, MostlyQ4_0, MostlyQ4_1,
    MostlyQ4_1SomeF16, MostlyQ8_0, MostlyQ5_0, MostlyQ5_1, MostlyQ2K, MostlyQ3K, MostlyQ4K,
    MostlyQ5K, MostlyQ6K, MostlyIq2Xxs, MostlyIq2Xs, MostlyIq3Xxs, MostlyIq1S, MostlyIq4Nl,
    MostlyIq3S, MostlyIq2S, MostlyIq4Xs, MostlyIq1M, MostlyBf16, MostlyMxfp4, MostlyNvfp4,
    MostlyQ1_0, MostlyQ2_0, MostlyF8E4M3);
enum_try_from_i32!(GgmlOp: None, Dup, Add, AddId, Acc, Sub, Mul, Div, Sqr, Sqrt, Log, Sin, Cos,
    Sum, SumRows, Cumsum, Mean, Argmax, CountEqual, Repeat, RepeatBack, Concat, SiluBack, Norm,
    RmsNorm, RmsNormBack, GroupNorm, L2Norm, MulMat, MulMatId, OutProd, Scale, Set, Cpy, Cont,
    Reshape, View, Permute, Transpose, GetRows, GetRowsBack, SetRows, Diag, SoftMax, SoftMaxBack,
    Rope, RopeBack, Clamp, ConvTranspose1d, Im2col, Im2colBack, Im2col3d, Col2im1d, Conv2d,
    Conv3d, Conv2dDw, ConvTranspose2d, Pool1d, Pool2d, Pool2dBack, Upscale, Pad, PadReflect1d,
    Roll, Arange, TimestepEmbedding, Argsort, TopK, LeakyRelu, Tri, Fill, FlashAttnExt,
    FlashAttnBack, SsmConv, SsmScan, WinPart, WinUnpart, GetRelPos, AddRelPos, RwkvWkv6,
    GatedLinearAttn, RwkvWkv7, SolveTri, GatedDeltaNet, LightningIndexer, Dsv4HcComb, Dsv4HcPre,
    Dsv4HcPost, TopKMask, Unary, MapCustom1, MapCustom2, MapCustom3, Custom, CrossEntropyLoss,
    CrossEntropyLossBack, OptStepAdamw, OptStepSgd, Glu, Sleep, SigmoidBack, Count);
enum_try_from_i32!(GgmlUnaryOp: Abs, Sgn, Neg, Step, Tanh, Elu, Relu, Sigmoid, Gelu, GeluQuick,
    Silu, Hardswish, Hardsigmoid, Exp, Expm1, Softplus, GeluErf, Xielu, Floor, Ceil, Round, Trunc,
    Count);
enum_try_from_i32!(GgmlGluOp: Reglu, Geglu, Swiglu, SwigluOai, GegluErf, GegluQuick, SwigluClamp,
    Count);
enum_try_from_i32!(GgmlObjectType: Tensor, Graph, WorkBuffer);
enum_try_from_i32!(GgmlLogLevel: None, Debug, Info, Warn, Error, Cont);
enum_try_from_i32!(GgmlTensorFlag: Input, Output, Param, Loss, Compute);
enum_try_from_i32!(GgmlTriType: UpperDiag, Upper, LowerDiag, Lower);
enum_try_from_i32!(GgmlCgraphEvalOrder: LeftToRight, RightToLeft, Count);

const _: () = assert!(GgmlType::Count as i32 == 44);
const _: () = assert!(GgmlOp::Count as i32 == 101);
const _: () = assert!(GGML_MAX_OP_PARAMS / size_of::<i32>() == 16);

// ---------------------------------------------------------------------------
// Scalar and block types
// ---------------------------------------------------------------------------

pub type GgmlHalf = u16;
pub type GgmlHalf2 = [u16; 2];

/// `ggml_fp16_t`.
pub type GgmlFp16 = u16;

/// `ggml_bf16_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct GgmlBf16 {
    pub bits: u16,
}

/// `ggml_fp8_e4m3_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct GgmlFp8E4M3 {
    pub bits: u8,
}

/// `ggml_guid`.
pub type GgmlGuid = [u8; 16];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ1_0 {
    pub d: GgmlHalf,
    pub qs: [u8; QK1_0 / 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ2_0 {
    pub d: GgmlHalf,
    pub qs: [u8; QK2_0 / 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub d: GgmlHalf,
    pub qs: [u8; QK4_0 / 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub d: GgmlHalf,
    pub m: GgmlHalf,
    pub qs: [u8; QK4_1 / 2],
}

impl BlockQ4_1 {
    /// `dm` in the C union view.
    pub fn dm(&self) -> GgmlHalf2 {
        [self.d, self.m]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockMxfp4 {
    pub e: u8,
    pub qs: [u8; QK_MXFP4 / 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockNvfp4 {
    pub d: [u8; QK_NVFP4 / QK_NVFP4_SUB],
    pub qs: [u8; QK_NVFP4 / 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub d: GgmlHalf,
    pub qh: [u8; 4],
    pub qs: [u8; QK5_0 / 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub d: GgmlHalf,
    pub m: GgmlHalf,
    pub qh: [u8; 4],
    pub qs: [u8; QK5_1 / 2],
}

impl BlockQ5_1 {
    pub fn dm(&self) -> GgmlHalf2 {
        [self.d, self.m]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: GgmlHalf,
    pub qs: [i8; QK8_0],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub d: GgmlHalf,
    pub s: GgmlHalf,
    pub qs: [i8; QK8_1],
}

impl BlockQ8_1 {
    pub fn ds(&self) -> GgmlHalf2 {
        [self.d, self.s]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockTq1_0 {
    pub qs: [u8; (QK_K - 4 * QK_K / 64) / 5],
    pub qh: [u8; QK_K / 64],
    pub d: GgmlHalf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockTq2_0 {
    pub qs: [u8; QK_K / 4],
    pub d: GgmlHalf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ2K {
    pub d: GgmlHalf,
    pub dmin: GgmlHalf,
    pub scales: [u8; QK_K / 16],
    pub qs: [u8; QK_K / 4],
}

impl BlockQ2K {
    pub fn dm(&self) -> GgmlHalf2 {
        [self.d, self.dmin]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ3K {
    pub hmask: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 4],
    pub scales: [u8; 12],
    pub d: GgmlHalf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ4K {
    pub d: GgmlHalf,
    pub dmin: GgmlHalf,
    pub scales: [u8; K_SCALE_SIZE],
    pub qs: [u8; QK_K / 2],
}

impl BlockQ4K {
    pub fn dm(&self) -> GgmlHalf2 {
        [self.d, self.dmin]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ5K {
    pub d: GgmlHalf,
    pub dmin: GgmlHalf,
    pub scales: [u8; K_SCALE_SIZE],
    pub qh: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 2],
}

impl BlockQ5K {
    pub fn dm(&self) -> GgmlHalf2 {
        [self.d, self.dmin]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: GgmlHalf,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockIq2Xxs {
    pub d: GgmlHalf,
    pub qs: [u16; QK_K / 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockIq2Xs {
    pub d: GgmlHalf,
    pub qs: [u16; QK_K / 8],
    pub scales: [u8; QK_K / 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockIq2S {
    pub d: GgmlHalf,
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub scales: [u8; QK_K / 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockIq3Xxs {
    pub d: GgmlHalf,
    pub qs: [u8; 3 * QK_K / 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockIq3S {
    pub d: GgmlHalf,
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub signs: [u8; QK_K / 8],
    pub scales: [u8; IQ3S_N_SCALE],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockIq1S {
    pub d: GgmlHalf,
    pub qs: [u8; QK_K / 8],
    pub qh: [u16; QK_K / 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockIq1M {
    pub qs: [u8; QK_K / 8],
    pub qh: [u8; QK_K / 16],
    pub scales: [u8; QK_K / 32],
}

/// `iq1m_scale_t` union, stored as its raw bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Iq1mScale {
    pub bits: u16,
}

impl Iq1mScale {
    pub fn f16(self) -> GgmlHalf {
        self.bits
    }

    pub fn u16(self) -> u16 {
        self.bits
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockIq4Nl {
    pub d: GgmlHalf,
    pub qs: [u8; QK4_NL / 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockIq4Xs {
    pub d: GgmlHalf,
    pub scales_h: u16,
    pub scales_l: [u8; QK_K / 64],
    pub qs: [u8; QK_K / 2],
}

const _: () = assert!(size_of::<BlockQ4_0>() == 2 + QK4_0 / 2);
const _: () = assert!(size_of::<BlockQ4_1>() == 4 + QK4_1 / 2);
const _: () = assert!(size_of::<BlockQ5_0>() == 2 + 4 + QK5_0 / 2);
const _: () = assert!(size_of::<BlockQ8_1>() == 4 + QK8_1);
const _: () = assert!(size_of::<BlockQ4K>() == 4 + K_SCALE_SIZE + QK_K / 2);
const _: () = assert!(size_of::<BlockQ8K>() == 4 + QK_K + QK_K / 16 * 2);
const _: () = assert!(size_of::<BlockIq4Xs>() == 4 + QK_K / 64 + QK_K / 2);

impl GgmlType {
    /// `ggml_blck_size`: elements represented by one block.
    pub fn blck_size(self) -> i64 {
        match self {
            GgmlType::F32
            | GgmlType::F16
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64
            | GgmlType::F64
            | GgmlType::Bf16
            | GgmlType::F8E4M3 => 1,
            GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q8_1
            | GgmlType::Iq4Nl
            | GgmlType::Mxfp4 => 32,
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

    /// `ggml_type_size`: bytes per block, derived from the block layouts.
    pub fn type_size(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => size_of::<BlockQ4_0>(),
            GgmlType::Q4_1 => size_of::<BlockQ4_1>(),
            GgmlType::Q5_0 => size_of::<BlockQ5_0>(),
            GgmlType::Q5_1 => size_of::<BlockQ5_1>(),
            GgmlType::Q8_0 => size_of::<BlockQ8_0>(),
            GgmlType::Q8_1 => size_of::<BlockQ8_1>(),
            GgmlType::Q2K => size_of::<BlockQ2K>(),
            GgmlType::Q3K => size_of::<BlockQ3K>(),
            GgmlType::Q4K => size_of::<BlockQ4K>(),
            GgmlType::Q5K => size_of::<BlockQ5K>(),
            GgmlType::Q6K => size_of::<BlockQ6K>(),
            GgmlType::Q8K => size_of::<BlockQ8K>(),
            GgmlType::Iq2Xxs => size_of::<BlockIq2Xxs>(),
            GgmlType::Iq2Xs => size_of::<BlockIq2Xs>(),
            GgmlType::Iq3Xxs => size_of::<BlockIq3Xxs>(),
            GgmlType::Iq1S => size_of::<BlockIq1S>(),
            GgmlType::Iq4Nl => size_of::<BlockIq4Nl>(),
            GgmlType::Iq3S => size_of::<BlockIq3S>(),
            GgmlType::Iq2S => size_of::<BlockIq2S>(),
            GgmlType::Iq4Xs => size_of::<BlockIq4Xs>(),
            GgmlType::I8 => 1,
            GgmlType::I16 => 2,
            GgmlType::I32 => 4,
            GgmlType::I64 => 8,
            GgmlType::F64 => 8,
            GgmlType::Iq1M => size_of::<BlockIq1M>(),
            GgmlType::Bf16 => 2,
            GgmlType::Tq1_0 => size_of::<BlockTq1_0>(),
            GgmlType::Tq2_0 => size_of::<BlockTq2_0>(),
            GgmlType::Mxfp4 => size_of::<BlockMxfp4>(),
            GgmlType::Nvfp4 => size_of::<BlockNvfp4>(),
            GgmlType::Q1_0 => size_of::<BlockQ1_0>(),
            GgmlType::Q2_0 => size_of::<BlockQ2_0>(),
            GgmlType::F8E4M3 => 1,
            GgmlType::Count => 0,
        }
    }

    /// `ggml_row_size`: bytes needed for `ne` elements in one row.
    pub fn row_size(self, ne: i64) -> usize {
        let bs = self.blck_size();
        if bs <= 0 || ne <= 0 {
            return 0;
        }
        let nb = self.type_size();
        ((ne as usize) / (bs as usize)) * nb
    }

    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q8_0
                | GgmlType::Q8_1
                | GgmlType::Q2K
                | GgmlType::Q3K
                | GgmlType::Q4K
                | GgmlType::Q5K
                | GgmlType::Q6K
                | GgmlType::Q8K
                | GgmlType::Iq2Xxs
                | GgmlType::Iq2Xs
                | GgmlType::Iq3Xxs
                | GgmlType::Iq1S
                | GgmlType::Iq4Nl
                | GgmlType::Iq3S
                | GgmlType::Iq2S
                | GgmlType::Iq4Xs
                | GgmlType::Iq1M
                | GgmlType::Tq1_0
                | GgmlType::Tq2_0
                | GgmlType::Mxfp4
                | GgmlType::Nvfp4
                | GgmlType::Q1_0
                | GgmlType::Q2_0
                | GgmlType::F8E4M3
        )
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
            GgmlType::Iq2Xxs => "iq2_xxs",
            GgmlType::Iq2Xs => "iq2_xs",
            GgmlType::Iq3Xxs => "iq3_xxs",
            GgmlType::Iq1S => "iq1_s",
            GgmlType::Iq4Nl => "iq4_nl",
            GgmlType::Iq3S => "iq3_s",
            GgmlType::Iq2S => "iq2_s",
            GgmlType::Iq4Xs => "iq4_xs",
            GgmlType::I8 => "i8",
            GgmlType::I16 => "i16",
            GgmlType::I32 => "i32",
            GgmlType::I64 => "i64",
            GgmlType::F64 => "f64",
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

/// Opaque handle to a tensor allocated by a ggml context; 0 means null.
pub type GgmlTensorRef = usize;
/// Opaque handle to a backend buffer; 0 means null.
pub type GgmlBufferRef = usize;
/// Opaque handle to a ggml context; 0 means null.
pub type GgmlContextRef = usize;

pub const GGML_NULL: usize = 0;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// `struct ggml_init_params`. `mem_buffer` is an opaque handle (0 = internal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlInitParams {
    pub mem_size: usize,
    pub mem_buffer: usize,
    pub no_alloc: bool,
}

impl Default for GgmlInitParams {
    fn default() -> Self {
        Self { mem_size: 0, mem_buffer: 0, no_alloc: false }
    }
}

/// `struct ggml_tensor`. Opaque pointers are `usize` handles, 0 means null.
#[derive(Debug, Clone, PartialEq)]
pub struct GgmlTensor {
    pub type_: GgmlType,
    pub buffer: GgmlBufferRef,
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
    pub op: GgmlOp,
    /// `int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]`.
    pub op_params: [i32; GGML_MAX_OP_PARAMS / size_of::<i32>()],
    pub flags: i32,
    pub src: [GgmlTensorRef; GGML_MAX_SRC],
    pub view_src: GgmlTensorRef,
    pub view_offs: usize,
    /// Opaque data pointer or buffer-relative handle; 0 means null.
    pub data: usize,
    pub name: String,
    /// Opaque backend-specific pointer, represented as a handle.
    pub extra: usize,
    pub padding: [u8; 8],
}

impl Default for GgmlTensor {
    fn default() -> Self {
        Self {
            type_: GgmlType::F32,
            buffer: GGML_NULL,
            ne: [0; GGML_MAX_DIMS],
            nb: [0; GGML_MAX_DIMS],
            op: GgmlOp::None,
            op_params: [0; GGML_MAX_OP_PARAMS / size_of::<i32>()],
            flags: 0,
            src: [GGML_NULL; GGML_MAX_SRC],
            view_src: GGML_NULL,
            view_offs: 0,
            data: GGML_NULL,
            name: String::new(),
            extra: GGML_NULL,
            padding: [0; 8],
        }
    }
}

impl GgmlTensor {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_name(&mut self, name: &str) {
        self.name = name.chars().take(GGML_MAX_NAME).collect();
    }

    pub fn is_view(&self) -> bool {
        self.view_src != GGML_NULL
    }
}

/// `ggml_bitset_t`.
pub type GgmlBitset = u32;

/// `struct ggml_hash_set`. `keys[i]` is meaningful only when bit `i` is set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlHashSet {
    pub size: usize,
    pub used: Vec<GgmlBitset>,
    pub keys: Vec<GgmlTensorRef>,
}

impl Default for GgmlHashSet {
    fn default() -> Self {
        Self::new(0)
    }
}

impl GgmlHashSet {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            used: vec![0; ggml_bitset_size(size)],
            keys: vec![GGML_NULL; size],
        }
    }

    pub fn reset(&mut self) {
        self.used.fill(0);
        self.keys.fill(GGML_NULL);
    }
}

/// `struct ggml_cgraph` from `ggml-impl.h`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlCgraph {
    pub size: i32,
    pub n_nodes: i32,
    pub n_leafs: i32,
    pub nodes: Vec<GgmlTensorRef>,
    pub grads: Vec<GgmlTensorRef>,
    pub grad_accs: Vec<GgmlTensorRef>,
    pub leafs: Vec<GgmlTensorRef>,
    pub use_counts: Vec<i32>,
    pub visited_hash_set: GgmlHashSet,
    pub order: GgmlCgraphEvalOrder,
    pub uid: u64,
}

impl Default for GgmlCgraph {
    fn default() -> Self {
        Self {
            size: 0,
            n_nodes: 0,
            n_leafs: 0,
            nodes: Vec::new(),
            grads: Vec::new(),
            grad_accs: Vec::new(),
            leafs: Vec::new(),
            use_counts: Vec::new(),
            visited_hash_set: GgmlHashSet::default(),
            order: GgmlCgraphEvalOrder::LeftToRight,
            uid: 0,
        }
    }
}

pub const GGML_TENSOR_SIZE: usize = size_of::<GgmlTensor>();

// ---------------------------------------------------------------------------
// ggml-impl.h helpers
// ---------------------------------------------------------------------------

pub fn min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

pub fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

/// `ggml_up32`.
pub fn ggml_up32(n: i32) -> i32 {
    (n + 31) & !31
}

/// `ggml_up`: round `n` up to a power-of-two multiple `m`.
pub fn ggml_up(n: i32, m: i32) -> i32 {
    assert!(m > 0 && (m & (m - 1)) == 0);
    (n + m - 1) & !(m - 1)
}

pub fn ggml_are_same_layout(a: &GgmlTensor, b: &GgmlTensor) -> bool {
    if a.type_ != b.type_ {
        return false;
    }
    (0..GGML_MAX_DIMS).all(|i| a.ne[i] == b.ne[i] && a.nb[i] == b.nb[i])
}

pub fn ggml_are_same_shape(a: &GgmlTensor, b: &GgmlTensor) -> bool {
    (0..GGML_MAX_DIMS).all(|i| a.ne[i] == b.ne[i])
}

pub fn ggml_op_is_empty(op: GgmlOp) -> bool {
    matches!(
        op,
        GgmlOp::None | GgmlOp::Reshape | GgmlOp::Transpose | GgmlOp::View | GgmlOp::Permute
    )
}

pub fn ggml_impl_is_view(t: &GgmlTensor) -> bool {
    t.view_src != GGML_NULL
}

pub fn ggml_compute_softplus_f32(input: f32) -> f32 {
    if input > 20.0 { input } else { (input.exp()).ln_1p() }
}

// op params: the C code aliases the int32 storage as float

pub fn ggml_set_op_params(tensor: &mut GgmlTensor, params: &[i32]) -> Result<()> {
    if params.len() > GGML_MAX_OP_PARAMS {
        return Err(Error::new("ggml_set_op_params: params too large"));
    }
    tensor.op_params[..params.len()].copy_from_slice(params);
    Ok(())
}

fn op_param_index(i: usize) -> Result<usize> {
    if i < GGML_MAX_OP_PARAMS / size_of::<i32>() {
        Ok(i)
    } else {
        Err(Error::new("op_params index out of range"))
    }
}

pub fn ggml_get_op_params_i32(tensor: &GgmlTensor, i: usize) -> Result<i32> {
    Ok(tensor.op_params[op_param_index(i)?])
}

pub fn ggml_get_op_params_f32(tensor: &GgmlTensor, i: usize) -> Result<f32> {
    Ok(f32::from_bits(tensor.op_params[op_param_index(i)?] as u32))
}

pub fn ggml_set_op_params_i32(tensor: &mut GgmlTensor, i: usize, value: i32) -> Result<()> {
    let i = op_param_index(i)?;
    tensor.op_params[i] = value;
    Ok(())
}

pub fn ggml_set_op_params_f32(tensor: &mut GgmlTensor, i: usize, value: f32) -> Result<()> {
    let i = op_param_index(i)?;
    tensor.op_params[i] = value.to_bits() as i32;
    Ok(())
}

// bitset

pub fn ggml_bitset_size(n: usize) -> usize {
    (n + BITSET_MASK) >> BITSET_SHR
}

pub fn ggml_bitset_get(bitset: &[GgmlBitset], i: usize) -> bool {
    bitset
        .get(i >> BITSET_SHR)
        .map_or(false, |w| w & (1u32 << (i & BITSET_MASK)) != 0)
}

/// Returns `false` when `i` is outside `bitset`.
pub fn ggml_bitset_set(bitset: &mut [GgmlBitset], i: usize) -> bool {
    match bitset.get_mut(i >> BITSET_SHR) {
        Some(w) => {
            *w |= 1u32 << (i & BITSET_MASK);
            true
        }
        None => false,
    }
}

/// Returns `false` when `i` is outside `bitset`.
pub fn ggml_bitset_clear(bitset: &mut [GgmlBitset], i: usize) -> bool {
    match bitset.get_mut(i >> BITSET_SHR) {
        Some(w) => {
            *w &= !(1u32 << (i & BITSET_MASK));
            true
        }
        None => false,
    }
}

// hash set

pub fn ggml_hash_set_new(size: usize) -> GgmlHashSet {
    GgmlHashSet::new(size)
}

/// Hash of an opaque tensor handle; low 4 bits are always zero in C.
pub fn ggml_hash(p: GgmlTensorRef) -> usize {
    p >> 4
}

pub fn ggml_hash_find(hash_set: &GgmlHashSet, key: GgmlTensorRef) -> usize {
    if hash_set.size == 0 {
        return GGML_HASHSET_FULL;
    }
    let h = ggml_hash(key) % hash_set.size;
    let mut i = h;
    loop {
        let used = ggml_bitset_get(&hash_set.used, i);
        if !used || hash_set.keys.get(i).copied() == Some(key) {
            return i;
        }
        i = (i + 1) % hash_set.size;
        if i == h {
            return GGML_HASHSET_FULL;
        }
    }
}

pub fn ggml_hash_contains(hash_set: &GgmlHashSet, key: GgmlTensorRef) -> bool {
    let i = ggml_hash_find(hash_set, key);
    i != GGML_HASHSET_FULL && ggml_bitset_get(&hash_set.used, i)
}

/// Returns the slot index, or `GGML_HASHSET_ALREADY_EXISTS`.
pub fn ggml_hash_insert(hash_set: &mut GgmlHashSet, key: GgmlTensorRef) -> Result<usize> {
    if hash_set.size == 0 {
        return Err(Error::new("ggml_hash_insert: hash set is empty"));
    }
    let h = ggml_hash(key) % hash_set.size;
    let mut i = h;
    loop {
        if !ggml_bitset_get(&hash_set.used, i) {
            if !ggml_bitset_set(&mut hash_set.used, i) {
                return Err(Error::new("ggml_hash_insert: bitset index out of range"));
            }
            match hash_set.keys.get_mut(i) {
                Some(k) => *k = key,
                None => return Err(Error::new("ggml_hash_insert: keys index out of range")),
            }
            return Ok(i);
        }
        if hash_set.keys.get(i).copied() == Some(key) {
            return Ok(GGML_HASHSET_ALREADY_EXISTS);
        }
        i = (i + 1) % hash_set.size;
        if i == h {
            return Err(Error::new("ggml_hash_insert: hash table full"));
        }
    }
}

pub fn ggml_hash_find_or_insert(hash_set: &mut GgmlHashSet, key: GgmlTensorRef) -> Result<usize> {
    if hash_set.size == 0 {
        return Err(Error::new("ggml_hash_find_or_insert: hash set is empty"));
    }
    let h = ggml_hash(key) % hash_set.size;
    let mut i = h;
    loop {
        if !ggml_bitset_get(&hash_set.used, i) {
            if !ggml_bitset_set(&mut hash_set.used, i) {
                return Err(Error::new("ggml_hash_find_or_insert: bitset index out of range"));
            }
            match hash_set.keys.get_mut(i) {
                Some(k) => *k = key,
                None => return Err(Error::new("ggml_hash_find_or_insert: keys index out of range")),
            }
            return Ok(i);
        }
        if hash_set.keys.get(i).copied() == Some(key) {
            return Ok(i);
        }
        i = (i + 1) % hash_set.size;
        if i == h {
            return Err(Error::new("ggml_hash_find_or_insert: hash table full"));
        }
    }
}

// graph helpers

pub fn ggml_node_get_use_count(cgraph: &GgmlCgraph, node_idx: usize) -> i32 {
    let node = match cgraph.nodes.get(node_idx) {
        Some(n) => *n,
        None => return 0,
    };
    let hash_pos = ggml_hash_find(&cgraph.visited_hash_set, node);
    if hash_pos == GGML_HASHSET_FULL
        || !ggml_bitset_get(&cgraph.visited_hash_set.used, hash_pos)
    {
        return 0;
    }
    cgraph.use_counts.get(hash_pos).copied().unwrap_or(0)
}

/// `resolve` replaces C pointer dereferencing on a `GgmlTensorRef`.
pub fn ggml_node_has_n_uses<'a, F>(
    cgraph: &GgmlCgraph,
    node_idx: usize,
    resolve: &F,
    n_uses: i32,
) -> bool
where
    F: Fn(GgmlTensorRef) -> Option<&'a GgmlTensor>,
{
    if ggml_node_get_use_count(cgraph, node_idx) != n_uses {
        return false;
    }
    let handle = match cgraph.nodes.get(node_idx) {
        Some(h) => *h,
        None => return false,
    };
    let node = match resolve(handle) {
        Some(n) => n,
        None => return false,
    };
    if node.is_view() || node.flags & GGML_TENSOR_FLAG_OUTPUT != 0 {
        return false;
    }
    true
}

/// `ggml_can_fuse_ext`. Indices are `usize` and `resolve` replaces pointer access.
pub fn ggml_can_fuse_ext<'a, F>(
    cgraph: &GgmlCgraph,
    node_idxs: &[usize],
    ops: &[GgmlOp],
    resolve: &F,
) -> bool
where
    F: Fn(GgmlTensorRef) -> Option<&'a GgmlTensor>,
{
    for i in 0..ops.len() {
        let idx = match node_idxs.get(i) {
            Some(idx) => *idx,
            None => return false,
        };
        if idx >= cgraph.n_nodes as usize {
            return false;
        }
        let op = match ops.get(i) {
            Some(op) => *op,
            None => return false,
        };
        let node = match cgraph.nodes.get(idx).and_then(|h| resolve(*h)) {
            Some(n) => n,
            None => return false,
        };
        if node.op != op || node.flags & GGML_TENSOR_FLAG_COMPUTE == 0 {
            return false;
        }
        if i < ops.len() - 1 && !ggml_node_has_n_uses(cgraph, idx, resolve, 1) {
            return false;
        }
        if i > 0 {
            let prev = match cgraph.nodes.get(node_idxs[i - 1]).and_then(|h| resolve(*h)) {
                Some(p) => p,
                None => return false,
            };
            if node.src.get(0) != Some(&prev_handle(cgraph, node_idxs[i - 1]))
                && node.src.get(1) != Some(&prev_handle(cgraph, node_idxs[i - 1]))
            {
                return false;
            }
            if !ggml_are_same_shape(node, prev) {
                return false;
            }
        }
    }
    true
}

fn prev_handle(cgraph: &GgmlCgraph, idx: usize) -> GgmlTensorRef {
    cgraph.nodes.get(idx).copied().unwrap_or(GGML_NULL)
}

pub fn ggml_can_fuse<'a, F>(
    cgraph: &GgmlCgraph,
    node_idx: usize,
    ops: &[GgmlOp],
    resolve: &F,
) -> bool
where
    F: Fn(GgmlTensorRef) -> Option<&'a GgmlTensor>,
{
    assert!(ops.len() < 32);
    if node_idx + ops.len() > cgraph.n_nodes as usize {
        return false;
    }
    let idxs: Vec<usize> = (node_idx..node_idx + ops.len()).collect();
    ggml_can_fuse_ext(cgraph, &idxs, ops, resolve)
}

// fp16 / bf16 / fp8 conversions

pub fn fp32_from_bits(w: u32) -> f32 {
    f32::from_bits(w)
}

pub fn fp32_to_bits(f: f32) -> u32 {
    f.to_bits()
}

/// `ggml_compute_fp16_to_fp32`, bit-identical to the C header helper.
pub fn ggml_compute_fp16_to_fp32(h: GgmlFp16) -> f32 {
    let w = (h as u32) << 16;
    let sign = w & 0x8000_0000;
    let two_w = w.wrapping_add(w);
    let exp_offset: u32 = 0xE0 << 23;
    let exp_scale = fp32_from_bits(0x0780_0000);
    let normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    let magic_mask: u32 = 126 << 23;
    let magic_bias = 0.5f32;
    let denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
    let denormalized_cutoff: u32 = 1 << 27;
    let result = sign
        | (if two_w < denormalized_cutoff {
            fp32_to_bits(denormalized_value)
        } else {
            fp32_to_bits(normalized_value)
        });
    fp32_from_bits(result)
}

/// `ggml_compute_fp32_to_fp16`, bit-identical to the C header helper.
pub fn ggml_compute_fp32_to_fp16(f: f32) -> GgmlFp16 {
    let scale_to_inf = fp32_from_bits(0x7780_0000);
    let scale_to_zero = fp32_from_bits(0x0880_0000);
    let mut base = (f.abs() * scale_to_inf) * scale_to_zero;
    let w = fp32_to_bits(f);
    let shl1_w = w.wrapping_add(w);
    let sign = w & 0x8000_0000;
    let mut bias = shl1_w & 0xFF00_0000;
    if bias < 0x7100_0000 {
        bias = 0x7100_0000;
    }
    base = fp32_from_bits((bias >> 1) + 0x0780_0000) + base;
    let bits = fp32_to_bits(base);
    let exp_bits = (bits >> 13) & 0x0000_7C00;
    let mantissa_bits = bits & 0x0000_0FFF;
    let nonsign = exp_bits + mantissa_bits;
    ((sign >> 16) | (if shl1_w > 0xFF00_0000 { 0x7E00 } else { nonsign })) as u16
}

pub fn ggml_compute_bf16_to_fp32(h: GgmlBf16) -> f32 {
    f32::from_bits((h.bits as u32) << 16)
}

pub fn ggml_compute_fp32_to_bf16(f: f32) -> GgmlBf16 {
    let u = fp32_to_bits(f);
    if u & 0x7fff_ffff > 0x7f80_0000 {
        return GgmlBf16 { bits: ((u >> 16) | 64) as u16 };
    }
    GgmlBf16 { bits: (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16 }
}

/// `ggml_e8m0_to_fp32`.
pub fn ggml_e8m0_to_fp32(x: u8) -> f32 {
    let bits = if x == 0 { 0x0040_0000 } else { (x as u32) << 23 };
    f32::from_bits(bits)
}

/// `ggml_e8m0_to_fp32_half`, equal to `ggml_e8m0_to_fp32(x) / 2`.
pub fn ggml_e8m0_to_fp32_half(x: u8) -> f32 {
    let bits = if x < 2 {
        0x0020_0000u32 << x
    } else {
        (x as u32 - 1) << 23
    };
    f32::from_bits(bits)
}

fn ldexp_f32(x: f32, exp: i32) -> f32 {
    x * 2.0f32.powi(exp)
}

/// `ggml_ue4m3_to_fp32`: unsigned E4M3 scale, returns value * 0.5.
pub fn ggml_ue4m3_to_fp32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as i32;
    let raw = if exp == 0 {
        ldexp_f32(man as f32, -9)
    } else {
        ldexp_f32(1.0 + man as f32 / 8.0, exp - 7)
    };
    raw * 0.5
}

/// `ggml_fp32_to_ue4m3`.
pub fn ggml_fp32_to_ue4m3(x: f32) -> u8 {
    if !(x > 0.0) {
        return 0;
    }
    let x = x.min(448.0);
    let bits = fp32_to_bits(x);
    let fp32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp32_man = ((bits >> 20) & 0x7) as i32;
    let ue4m3_exp = fp32_exp + 7;
    if ue4m3_exp <= 0 {
        let man = (x * 512.0 + 0.5) as i32;
        if man < 1 {
            return 0;
        }
        return man.min(7) as u8;
    }
    if ue4m3_exp >= 15 {
        return 0x7E;
    }
    let round_bit = ((bits >> 19) & 1) as i32;
    let mut ue4m3_man = fp32_man + round_bit;
    let mut ue4m3_exp = ue4m3_exp;
    if ue4m3_man > 7 {
        ue4m3_man = 0;
        ue4m3_exp += 1;
        if ue4m3_exp >= 15 {
            return 0x7E;
        }
    }
    ((ue4m3_exp << 3) | ue4m3_man) as u8
}

/// `ggml_f8_e4m3_to_fp32` (OCP FP8, E4M3).
pub fn ggml_f8_e4m3_to_fp32(x: u8) -> f32 {
    let ax = x & 0x7F;
    if ax == 0x7F {
        return f32::NAN;
    }
    let exp = ((ax >> 3) & 0xF) as u32;
    let man = (ax & 0x7) as u32;
    let value = if exp == 0 {
        man as f32 * (1.0 / 512.0)
    } else {
        fp32_from_bits(((exp + 120) << 23) | (man << 20))
    };
    fp32_from_bits(fp32_to_bits(value) | ((x as u32 & 0x80) << 24))
}

pub fn ggml_round_to_nearest_even(x: f32) -> i32 {
    let value = x.floor() as i32;
    let fraction = x - value as f32;
    if fraction > 0.5 || (fraction == 0.5 && (value & 1) != 0) {
        value + 1
    } else {
        value
    }
}

/// `ggml_fp32_to_f8_e4m3`, saturating conversion.
pub fn ggml_fp32_to_f8_e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let x = x.abs();
    if x.is_nan() {
        return 0x7F;
    }
    if x == 0.0 {
        return sign;
    }
    if x.is_infinite() || x >= 448.0 {
        return sign | 0x7E;
    }
    if x < 0.015625 {
        return sign | ggml_round_to_nearest_even(x * 512.0) as u8;
    }
    let bits = fp32_to_bits(x);
    let float_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa_bits = (bits & 0x007f_ffff) | 0x3f80_0000;
    let mantissa = fp32_from_bits(mantissa_bits);
    let mut encoded_exp = float_exp + 7;
    let mut encoded_man = ggml_round_to_nearest_even((mantissa - 1.0) * 8.0);
    if encoded_man == 8 {
        encoded_man = 0;
        encoded_exp += 1;
    }
    if encoded_exp > 15 || (encoded_exp == 15 && encoded_man > 6) {
        return sign | 0x7E;
    }
    sign | ((encoded_exp << 3) | encoded_man) as u8
}

// ---------------------------------------------------------------------------
// Tests for the bit-exact helpers and numeric values
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_values_match_headers() {
        assert_eq!(GgmlType::F8E4M3 as i32, 43);
        assert_eq!(GgmlType::Count as i32, 44);
        assert_eq!(GgmlOp::SigmoidBack as i32, 100);
        assert_eq!(GgmlOp::Count as i32, 101);
        assert_eq!(GgmlFtype::MostlyQ4_1SomeF16 as i32, 4);
        assert_eq!(GgmlFtype::MostlyQ8_0 as i32, 7);
    }

    #[test]
    fn block_sizes_match_static_asserts() {
        assert_eq!(size_of::<BlockQ1_0>(), 18);
        assert_eq!(size_of::<BlockQ2_0>(), 18);
        assert_eq!(size_of::<BlockQ4_0>(), 18);
        assert_eq!(size_of::<BlockQ4_1>(), 20);
        assert_eq!(size_of::<BlockMxfp4>(), 17);
        assert_eq!(size_of::<BlockNvfp4>(), 36);
        assert_eq!(size_of::<BlockQ5_0>(), 22);
        assert_eq!(size_of::<BlockQ5_1>(), 24);
        assert_eq!(size_of::<BlockQ8_0>(), 34);
        assert_eq!(size_of::<BlockQ8_1>(), 36);
        assert_eq!(size_of::<BlockTq1_0>(), 54);
        assert_eq!(size_of::<BlockTq2_0>(), 66);
        assert_eq!(size_of::<BlockQ2K>(), 84);
        assert_eq!(size_of::<BlockQ3K>(), 110);
        assert_eq!(size_of::<BlockQ4K>(), 144);
        assert_eq!(size_of::<BlockQ5K>(), 176);
        assert_eq!(size_of::<BlockQ6K>(), 210);
        assert_eq!(size_of::<BlockQ8K>(), 292);
        assert_eq!(size_of::<BlockIq2Xxs>(), 66);
        assert_eq!(size_of::<BlockIq2Xs>(), 74);
        assert_eq!(size_of::<BlockIq2S>(), 82);
        assert_eq!(size_of::<BlockIq3Xxs>(), 98);
        assert_eq!(size_of::<BlockIq3S>(), 110);
        assert_eq!(size_of::<BlockIq1S>(), 50);
        assert_eq!(size_of::<BlockIq1M>(), 56);
        assert_eq!(size_of::<BlockIq4Nl>(), 18);
        assert_eq!(size_of::<BlockIq4Xs>(), 136);
    }

    #[test]
    fn fp16_round_trip() {
        for value in [0.0f32, 1.0, -1.0, 0.5, 65504.0] {
            let h = ggml_compute_fp32_to_fp16(value);
            assert_eq!(ggml_compute_fp16_to_fp32(h), value);
        }
        assert_eq!(ggml_compute_fp32_to_fp16(6.103515625e-5), 0x0400);
        assert_eq!(ggml_compute_fp16_to_fp32(0x7c00), f32::INFINITY);
    }

    #[test]
    fn bf16_round_trip() {
        let h = ggml_compute_fp32_to_bf16(1.0);
        assert_eq!(ggml_compute_bf16_to_fp32(h), 1.0);
        assert_eq!(h.bits, 0x3f80);
    }

    #[test]
    fn fp8_conversions() {
        assert_eq!(ggml_f8_e4m3_to_fp32(0x7E), 448.0);
        assert_eq!(ggml_fp32_to_f8_e4m3(448.0), 0x7E);
        assert_eq!(ggml_fp32_to_f8_e4m3(1.0), 0x38);
        assert_eq!(ggml_e8m0_to_fp32(127), 1.0);
        assert_eq!(ggml_e8m0_to_fp32_half(128), 1.0);
    }

    #[test]
    fn bitset_and_hash() {
        let mut set = GgmlHashSet::new(8);
        assert_eq!(ggml_hash_insert(&mut set, 16).unwrap(), 1);
        assert_eq!(ggml_hash_contains(&set, 16), true);
        assert_eq!(ggml_hash_find(&set, 32), 2);
        assert_eq!(ggml_hash_find_or_insert(&mut set, 32).unwrap(), 2);
        assert_eq!(ggml_hash_insert(&mut set, 16).unwrap(), GGML_HASHSET_ALREADY_EXISTS);
        for key in [48usize, 64, 80, 96, 112, 128] {
            assert!(ggml_hash_insert(&mut set, key).is_ok());
        }
        assert_eq!(ggml_hash_find(&set, 144), GGML_HASHSET_FULL);
        ggml_bitset_clear(&mut set.used, 1);
        assert!(!ggml_bitset_get(&set.used, 1));
    }

    #[test]
    fn type_tables() {
        assert_eq!(GgmlType::Q5_0 as i32, 6);
        assert_eq!(GgmlType::Mxfp4.type_size(), 17);
        assert_eq!(GgmlType::Q8_1.type_size(), 36);
        assert_eq!(GgmlType::Q4K.row_size(256), 144);
        assert!(GgmlType::Nvfp4.is_quantized());
        assert!(!GgmlType::F32.is_quantized());
    }
}

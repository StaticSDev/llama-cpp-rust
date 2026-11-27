//! Rust port of the `ggml_type` / `ggml_op` enums and their lookup tables.
//!
//! Values, names and block sizes come from `ggml.h` and `ggml.c`. Removed
//! type slots (4, 5, 31, 32, 33, 36, 37, 38) keep their numeric gap so that
//! on-disk GGUF type ids stay stable.

// Enum variants keep upstream spelling (`Q4_K`, `IQ2_XXS`, ...).
#![allow(non_camel_case_types)]

use std::convert::TryFrom;

use crate::core::Error;

/// `GGML_MAX_DIMS`.
pub const GGML_MAX_DIMS: usize = 4;
/// `GGML_MAX_SRC`.
pub const GGML_MAX_SRC: usize = 10;
/// `GGML_MAX_OP_PARAMS`, in bytes.
pub const GGML_MAX_OP_PARAMS: usize = 64;
/// Number of `i32` slots in `ggml_tensor::op_params`.
pub const GGML_MAX_OP_PARAMS_I32: usize = GGML_MAX_OP_PARAMS / 4;
/// `GGML_MAX_NAME`.
pub const GGML_MAX_NAME: usize = 64;
/// `GGML_MEM_ALIGN` on a 64-bit target.
pub const GGML_MEM_ALIGN: usize = 16;
/// `GGML_TYPE_COUNT`.
pub const GGML_TYPE_COUNT: usize = 44;
/// `GGML_OP_COUNT`.
pub const GGML_OP_COUNT: usize = 101;
/// `GGML_UNARY_OP_COUNT`.
pub const GGML_UNARY_OP_COUNT: usize = 22;
/// `GGML_GLU_OP_COUNT`.
pub const GGML_GLU_OP_COUNT: usize = 7;

/// `GGML_PAD(x, n)`.
pub const fn ggml_pad(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}

/// `enum ggml_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
    F8_E4M3 = 43,
}

impl GgmlType {
    /// Numeric GGUF/ggml type id.
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    /// Convert from a GGUF type id. Removed slots return `Error`.
    pub fn from_i32(value: i32) -> Result<Self, Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            20 => Ok(Self::IQ4_NL),
            21 => Ok(Self::IQ3_S),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1_M),
            30 => Ok(Self::BF16),
            34 => Ok(Self::TQ1_0),
            35 => Ok(Self::TQ2_0),
            39 => Ok(Self::MXFP4),
            40 => Ok(Self::NVFP4),
            41 => Ok(Self::Q1_0),
            42 => Ok(Self::Q2_0),
            43 => Ok(Self::F8_E4M3),
            _ => Err(Error::new(format!("invalid ggml type: {}", value))),
        }
    }
}

impl TryFrom<i32> for GgmlType {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Self::from_i32(value)
    }
}

impl Default for GgmlType {
    fn default() -> Self {
        Self::F32
    }
}

/// One row of `type_traits` in `ggml.c`.
#[derive(Debug, Clone, Copy)]
pub struct GgmlTypeTraits {
    pub type_name: &'static str,
    /// Elements per quantized block, or 1 for plain scalar types.
    pub blck_size: i64,
    /// Byte size of one block (one element for unquantized types).
    pub type_size: usize,
    pub is_quantized: bool,
}

/// `type_traits[type]` from `ggml.c`.
///
/// Block byte sizes follow the `static_assert`s in `ggml-common.h` with
/// `QK_K = 256`, `QK4_NL = 32`, `K_SCALE_SIZE = 12`.
pub const fn ggml_type_traits(t: GgmlType) -> GgmlTypeTraits {
    use GgmlType::*;
    let (type_name, blck_size, type_size, is_quantized) = match t {
        F32 => ("f32", 1, 4, false),
        F16 => ("f16", 1, 2, false),
        Q4_0 => ("q4_0", 32, 18, true),
        Q4_1 => ("q4_1", 32, 20, true),
        Q5_0 => ("q5_0", 32, 22, true),
        Q5_1 => ("q5_1", 32, 24, true),
        Q8_0 => ("q8_0", 32, 34, true),
        Q8_1 => ("q8_1", 32, 36, true),
        Q2_K => ("q2_K", 256, 84, true),
        Q3_K => ("q3_K", 256, 110, true),
        Q4_K => ("q4_K", 256, 144, true),
        Q5_K => ("q5_K", 256, 176, true),
        Q6_K => ("q6_K", 256, 210, true),
        Q8_K => ("q8_K", 256, 292, true),
        IQ2_XXS => ("iq2_xxs", 256, 66, true),
        IQ2_XS => ("iq2_xs", 256, 74, true),
        IQ3_XXS => ("iq3_xxs", 256, 98, true),
        IQ1_S => ("iq1_s", 256, 50, true),
        IQ4_NL => ("iq4_nl", 32, 18, true),
        IQ3_S => ("iq3_s", 256, 110, true),
        IQ2_S => ("iq2_s", 256, 82, true),
        IQ4_XS => ("iq4_xs", 256, 136, true),
        I8 => ("i8", 1, 1, false),
        I16 => ("i16", 1, 2, false),
        I32 => ("i32", 1, 4, false),
        I64 => ("i64", 1, 8, false),
        F64 => ("f64", 1, 8, false),
        IQ1_M => ("iq1_m", 256, 56, true),
        BF16 => ("bf16", 1, 2, false),
        TQ1_0 => ("tq1_0", 256, 54, true),
        TQ2_0 => ("tq2_0", 256, 66, true),
        MXFP4 => ("mxfp4", 32, 17, true),
        NVFP4 => ("nvfp4", 64, 36, true),
        Q1_0 => ("q1_0", 128, 18, true),
        Q2_0 => ("q2_0", 64, 18, true),
        F8_E4M3 => ("f8_e4m3", 1, 1, false),
    };
    GgmlTypeTraits {
        type_name,
        blck_size,
        type_size,
        is_quantized,
    }
}

/// `ggml_blck_size`.
pub const fn ggml_blck_size(t: GgmlType) -> i64 {
    ggml_type_traits(t).blck_size
}

/// `ggml_type_size`.
pub const fn ggml_type_size(t: GgmlType) -> usize {
    ggml_type_traits(t).type_size
}

/// `ggml_type_name`.
pub const fn ggml_type_name(t: GgmlType) -> &'static str {
    ggml_type_traits(t).type_name
}

/// `ggml_is_quantized`.
pub const fn ggml_is_quantized(t: GgmlType) -> bool {
    ggml_type_traits(t).is_quantized
}

/// `ggml_row_size`. The C version asserts `ne % blck_size == 0`.
pub fn ggml_row_size(t: GgmlType, ne: i64) -> Result<usize, Error> {
    let bs = ggml_blck_size(t);
    if ne < 0 || bs <= 0 || ne % bs != 0 {
        return Err(Error::new(format!(
            "ggml_row_size: ne {} not a multiple of block size {}",
            ne, bs
        )));
    }
    Ok(ggml_type_size(t) * (ne as usize) / (bs as usize))
}

/// `enum ggml_op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlOp {
    None = 0,
    Dup,
    Add,
    AddId,
    Acc,
    Sub,
    Mul,
    Div,
    Sqr,
    Sqrt,
    Log,
    Sin,
    Cos,
    Sum,
    SumRows,
    Cumsum,
    Mean,
    Argmax,
    CountEqual,
    Repeat,
    RepeatBack,
    Concat,
    SiluBack,
    Norm,
    RmsNorm,
    RmsNormBack,
    GroupNorm,
    L2Norm,

    MulMat,
    MulMatId,
    OutProd,

    Scale,
    Set,
    Cpy,
    Cont,
    Reshape,
    View,
    Permute,
    Transpose,
    GetRows,
    GetRowsBack,
    SetRows,
    Diag,
    SoftMax,
    SoftMaxBack,
    Rope,
    RopeBack,
    Clamp,
    ConvTranspose1D,
    Im2Col,
    Im2ColBack,
    Im2Col3D,
    Col2Im1D,
    Conv2D,
    Conv3D,
    Conv2DDw,
    ConvTranspose2D,
    Pool1D,
    Pool2D,
    Pool2DBack,
    Upscale,
    Pad,
    PadReflect1D,
    Roll,
    Arange,
    TimestepEmbedding,
    Argsort,
    TopK,
    LeakyRelu,
    Tri,
    Fill,

    FlashAttnExt,
    FlashAttnBack,
    SsmConv,
    SsmScan,
    WinPart,
    WinUnpart,
    GetRelPos,
    AddRelPos,
    RwkvWkv6,
    GatedLinearAttn,
    RwkvWkv7,
    SolveTri,
    GatedDeltaNet,
    LightningIndexer,
    Dsv4HcComb,
    Dsv4HcPre,
    Dsv4HcPost,
    TopKMask,

    Unary,

    MapCustom1,
    MapCustom2,
    MapCustom3,

    Custom,

    CrossEntropyLoss,
    CrossEntropyLossBack,
    OptStepAdamw,
    OptStepSgd,

    Glu,

    Sleep,

    SigmoidBack,

    /// `GGML_OP_COUNT`, the number of real ops.
    Count,
}

impl Default for GgmlOp {
    fn default() -> Self {
        Self::None
    }
}

impl GgmlOp {
    /// Numeric op id.
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    /// Convert from a numeric op id.
    pub fn from_i32(value: i32) -> Result<Self, Error> {
        if value < 0 || value >= GGML_OP_COUNT as i32 {
            return Err(Error::new(format!("invalid ggml op: {}", value)));
        }
        // Safety is provided by the range check; the enum is contiguous.
        Ok(ALL_OPS[value as usize])
    }
}

/// All `ggml_op` values in discriminant order.
pub const ALL_OPS: [GgmlOp; GGML_OP_COUNT] = {
    use GgmlOp::*;
    [
        None, Dup, Add, AddId, Acc, Sub, Mul, Div, Sqr, Sqrt, Log, Sin, Cos, Sum, SumRows, Cumsum,
        Mean, Argmax, CountEqual, Repeat, RepeatBack, Concat, SiluBack, Norm, RmsNorm, RmsNormBack,
        GroupNorm, L2Norm, MulMat, MulMatId, OutProd, Scale, Set, Cpy, Cont, Reshape, View, Permute,
        Transpose, GetRows, GetRowsBack, SetRows, Diag, SoftMax, SoftMaxBack, Rope, RopeBack, Clamp,
        ConvTranspose1D, Im2Col, Im2ColBack, Im2Col3D, Col2Im1D, Conv2D, Conv3D, Conv2DDw,
        ConvTranspose2D, Pool1D, Pool2D, Pool2DBack, Upscale, Pad, PadReflect1D, Roll, Arange,
        TimestepEmbedding, Argsort, TopK, LeakyRelu, Tri, Fill, FlashAttnExt, FlashAttnBack,
        SsmConv, SsmScan, WinPart, WinUnpart, GetRelPos, AddRelPos, RwkvWkv6, GatedLinearAttn,
        RwkvWkv7, SolveTri, GatedDeltaNet, LightningIndexer, Dsv4HcComb, Dsv4HcPre, Dsv4HcPost,
        TopKMask, Unary, MapCustom1, MapCustom2, MapCustom3, Custom, CrossEntropyLoss,
        CrossEntropyLossBack, OptStepAdamw, OptStepSgd, Glu, Sleep, SigmoidBack,
    ]
};

/// `GGML_OP_NAME[op]`.
pub const fn ggml_op_name(op: GgmlOp) -> &'static str {
    use GgmlOp::*;
    match op {
        None => "NONE",
        Dup => "DUP",
        Add => "ADD",
        AddId => "ADD_ID",
        Acc => "ACC",
        Sub => "SUB",
        Mul => "MUL",
        Div => "DIV",
        Sqr => "SQR",
        Sqrt => "SQRT",
        Log => "LOG",
        Sin => "SIN",
        Cos => "COS",
        Sum => "SUM",
        SumRows => "SUM_ROWS",
        Cumsum => "CUMSUM",
        Mean => "MEAN",
        Argmax => "ARGMAX",
        CountEqual => "COUNT_EQUAL",
        Repeat => "REPEAT",
        RepeatBack => "REPEAT_BACK",
        Concat => "CONCAT",
        SiluBack => "SILU_BACK",
        Norm => "NORM",
        RmsNorm => "RMS_NORM",
        RmsNormBack => "RMS_NORM_BACK",
        GroupNorm => "GROUP_NORM",
        L2Norm => "L2_NORM",
        MulMat => "MUL_MAT",
        MulMatId => "MUL_MAT_ID",
        OutProd => "OUT_PROD",
        Scale => "SCALE",
        Set => "SET",
        Cpy => "CPY",
        Cont => "CONT",
        Reshape => "RESHAPE",
        View => "VIEW",
        Permute => "PERMUTE",
        Transpose => "TRANSPOSE",
        GetRows => "GET_ROWS",
        GetRowsBack => "GET_ROWS_BACK",
        SetRows => "SET_ROWS",
        Diag => "DIAG",
        SoftMax => "SOFT_MAX",
        SoftMaxBack => "SOFT_MAX_BACK",
        Rope => "ROPE",
        RopeBack => "ROPE_BACK",
        Clamp => "CLAMP",
        ConvTranspose1D => "CONV_TRANSPOSE_1D",
        Im2Col => "IM2COL",
        Im2ColBack => "IM2COL_BACK",
        Im2Col3D => "IM2COL_3D",
        Col2Im1D => "COL2IM_1D",
        Conv2D => "CONV_2D",
        Conv3D => "CONV_3D",
        Conv2DDw => "CONV_2D_DW",
        ConvTranspose2D => "CONV_TRANSPOSE_2D",
        Pool1D => "POOL_1D",
        Pool2D => "POOL_2D",
        Pool2DBack => "POOL_2D_BACK",
        Upscale => "UPSCALE",
        Pad => "PAD",
        PadReflect1D => "PAD_REFLECT_1D",
        Roll => "ROLL",
        Arange => "ARANGE",
        TimestepEmbedding => "TIMESTEP_EMBEDDING",
        Argsort => "ARGSORT",
        TopK => "TOP_K",
        LeakyRelu => "LEAKY_RELU",
        Tri => "TRI",
        Fill => "FILL",
        FlashAttnExt => "FLASH_ATTN_EXT",
        FlashAttnBack => "FLASH_ATTN_BACK",
        SsmConv => "SSM_CONV",
        SsmScan => "SSM_SCAN",
        WinPart => "WIN_PART",
        WinUnpart => "WIN_UNPART",
        GetRelPos => "GET_REL_POS",
        AddRelPos => "ADD_REL_POS",
        RwkvWkv6 => "RWKV_WKV6",
        GatedLinearAttn => "GATED_LINEAR_ATTN",
        RwkvWkv7 => "RWKV_WKV7",
        SolveTri => "SOLVE_TRI",
        GatedDeltaNet => "GATED_DELTA_NET",
        LightningIndexer => "LIGHTNING_INDEXER",
        Dsv4HcComb => "DSV4_HC_COMB",
        Dsv4HcPre => "DSV4_HC_PRE",
        Dsv4HcPost => "DSV4_HC_POST",
        TopKMask => "TOP_K_MASK",
        Unary => "UNARY",
        MapCustom1 => "MAP_CUSTOM1",
        MapCustom2 => "MAP_CUSTOM2",
        MapCustom3 => "MAP_CUSTOM3",
        Custom => "CUSTOM",
        CrossEntropyLoss => "CROSS_ENTROPY_LOSS",
        CrossEntropyLossBack => "CROSS_ENTROPY_LOSS_BACK",
        OptStepAdamw => "OPT_STEP_ADAMW",
        OptStepSgd => "OPT_STEP_SGD",
        Glu => "GLU",
        Sleep => "SLEEP",
        SigmoidBack => "SIGMOID_BACK",
        Count => "COUNT",
    }
}

/// `GGML_OP_SYMBOL[op]`.
pub const fn ggml_op_symbol(op: GgmlOp) -> &'static str {
    use GgmlOp::*;
    match op {
        None => "none",
        Dup => "x",
        Add => "x+y",
        AddId => "x[i]+y",
        Acc => "view(x,nb,offset)+=y->x",
        Sub => "x-y",
        Mul => "x*y",
        Div => "x/y",
        Sqr => "x^2",
        Sqrt => "√x",
        Log => "log(x)",
        Sin => "sin(x)",
        Cos => "cos(x)",
        Sum => "Σx",
        SumRows => "Σx_k",
        Cumsum => "cumsum(x)",
        Mean => "Σx/n",
        Argmax => "argmax(x)",
        CountEqual => "count_equal(x)",
        Repeat => "repeat(x)",
        RepeatBack => "repeat_back(x)",
        Concat => "concat(x, y)",
        SiluBack => "silu_back(x)",
        Norm => "norm(x)",
        RmsNorm => "rms_norm(x)",
        RmsNormBack => "rms_norm_back(x)",
        GroupNorm => "group_norm(x)",
        L2Norm => "l2_norm(x)",
        MulMat => "X*Y",
        MulMatId => "X[i]*Y",
        OutProd => "X*Y",
        Scale => "x*v",
        Set => "y-\\>view(x)",
        Cpy => "x-\\>y",
        Cont => "cont(x)",
        Reshape => "reshape(x)",
        View => "view(x)",
        Permute => "permute(x)",
        Transpose => "transpose(x)",
        GetRows => "get_rows(x)",
        GetRowsBack => "get_rows_back(x)",
        SetRows => "set_rows(x)",
        Diag => "diag(x)",
        SoftMax => "soft_max(x)",
        SoftMaxBack => "soft_max_back(x)",
        Rope => "rope(x)",
        RopeBack => "rope_back(x)",
        Clamp => "clamp(x)",
        ConvTranspose1D => "conv_transpose_1d(x)",
        Im2Col => "im2col(x)",
        Im2ColBack => "im2col_back(x)",
        Im2Col3D => "im2col_3d(x)",
        Col2Im1D => "col2im_1d(x)",
        Conv2D => "conv_2d(x)",
        Conv3D => "conv_3d(x)",
        Conv2DDw => "conv_2d_dw(x)",
        ConvTranspose2D => "conv_transpose_2d(x)",
        Pool1D => "pool_1d(x)",
        Pool2D => "pool_2d(x)",
        Pool2DBack => "pool_2d_back(x)",
        Upscale => "upscale(x)",
        Pad => "pad(x)",
        PadReflect1D => "pad_reflect_1d(x)",
        Roll => "roll(x)",
        Arange => "arange(start, stop, step)",
        TimestepEmbedding => "timestep_embedding(timesteps, dim, max_period)",
        Argsort => "argsort(x)",
        TopK => "top_k(x)",
        LeakyRelu => "leaky_relu(x)",
        Tri => "tri(x)",
        Fill => "fill(x, c)",
        FlashAttnExt => "flash_attn_ext(x)",
        FlashAttnBack => "flash_attn_back(x)",
        SsmConv => "ssm_conv(x)",
        SsmScan => "ssm_scan(x)",
        WinPart => "win_part(x)",
        WinUnpart => "win_unpart(x)",
        GetRelPos => "get_rel_pos(x)",
        AddRelPos => "add_rel_pos(x)",
        RwkvWkv6 => "rwkv_wkv6(k, v, r, tf, td, s)",
        GatedLinearAttn => "gated_linear_attn(k, v, q, gate, s)",
        RwkvWkv7 => "rwkv_wkv7(r, w, k, v, a, b, s)",
        SolveTri => "A X = B, A triangular, solve X",
        GatedDeltaNet => "gated_delta_net(q, k, v, g, beta, s)",
        LightningIndexer => "lightning_indexer(q, k, weights, mask)",
        Dsv4HcComb => "dsv4_hc_comb(mixes, scale, base)",
        Dsv4HcPre => "dsv4_hc_pre(x, weights)",
        Dsv4HcPost => "dsv4_hc_post(x, residual, post, comb)",
        TopKMask => "top_k_mask(mask, top_k, score, blocks, extra)",
        Unary => "unary(x)",
        MapCustom1 => "map_custom(x)",
        MapCustom2 => "map_custom(x,y)",
        MapCustom3 => "map_custom(x,y,z)",
        Custom => "custom(x)",
        CrossEntropyLoss => "cross_entropy_loss(x,y)",
        CrossEntropyLossBack => "cross_entropy_loss_back(x,y)",
        OptStepAdamw => "adamw(x)",
        OptStepSgd => "sgd(x)",
        Glu => "glu(x)",
        Sleep => "sleep(x)",
        SigmoidBack => "sigmoid_back(x)",
        Count => "count",
    }
}

/// `enum ggml_unary_op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlUnaryOp {
    Abs,
    Sgn,
    Neg,
    Step,
    Tanh,
    Elu,
    Relu,
    Sigmoid,
    Gelu,
    GeluQuick,
    Silu,
    Hardswish,
    Hardsigmoid,
    Exp,
    Expm1,
    Softplus,
    GeluErf,
    Xielu,
    Floor,
    Ceil,
    Round,
    Trunc,
}

impl GgmlUnaryOp {
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    pub fn from_i32(value: i32) -> Result<Self, Error> {
        if value < 0 || value >= GGML_UNARY_OP_COUNT as i32 {
            return Err(Error::new(format!("invalid unary op: {}", value)));
        }
        Ok(UNARY_OPS[value as usize])
    }
}

/// All `ggml_unary_op` values in discriminant order.
pub const UNARY_OPS: [GgmlUnaryOp; GGML_UNARY_OP_COUNT] = {
    use GgmlUnaryOp::*;
    [
        Abs, Sgn, Neg, Step, Tanh, Elu, Relu, Sigmoid, Gelu, GeluQuick, Silu, Hardswish,
        Hardsigmoid, Exp, Expm1, Softplus, GeluErf, Xielu, Floor, Ceil, Round, Trunc,
    ]
};

/// `GGML_UNARY_OP_NAME[op]`.
pub const fn ggml_unary_op_name(op: GgmlUnaryOp) -> &'static str {
    use GgmlUnaryOp::*;
    match op {
        Abs => "ABS",
        Sgn => "SGN",
        Neg => "NEG",
        Step => "STEP",
        Tanh => "TANH",
        Elu => "ELU",
        Relu => "RELU",
        Sigmoid => "SIGMOID",
        Gelu => "GELU",
        GeluQuick => "GELU_QUICK",
        Silu => "SILU",
        Hardswish => "HARDSWISH",
        Hardsigmoid => "HARDSIGMOID",
        Exp => "EXP",
        Expm1 => "EXPM1",
        Softplus => "SOFTPLUS",
        GeluErf => "GELU_ERF",
        Xielu => "XIELU",
        Floor => "FLOOR",
        Ceil => "CEIL",
        Round => "ROUND",
        Trunc => "TRUNC",
    }
}

/// `enum ggml_glu_op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlGluOp {
    Reglu,
    Geglu,
    Swiglu,
    SwigluOai,
    GegluErf,
    GegluQuick,
    SwigluClamp,
}

impl GgmlGluOp {
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    pub fn from_i32(value: i32) -> Result<Self, Error> {
        if value < 0 || value >= GGML_GLU_OP_COUNT as i32 {
            return Err(Error::new(format!("invalid glu op: {}", value)));
        }
        Ok(GLU_OPS[value as usize])
    }
}

/// All `ggml_glu_op` values in discriminant order.
pub const GLU_OPS: [GgmlGluOp; GGML_GLU_OP_COUNT] = {
    use GgmlGluOp::*;
    [
        Reglu,
        Geglu,
        Swiglu,
        SwigluOai,
        GegluErf,
        GegluQuick,
        SwigluClamp,
    ]
};

/// `GGML_GLU_OP_NAME[op]`.
pub const fn ggml_glu_op_name(op: GgmlGluOp) -> &'static str {
    use GgmlGluOp::*;
    match op {
        Reglu => "REGLU",
        Geglu => "GEGLU",
        Swiglu => "SWIGLU",
        SwigluOai => "SWIGLU_OAI",
        GegluErf => "GEGLU_ERF",
        GegluQuick => "GEGLU_QUICK",
        SwigluClamp => "SWIGLU_CLAMP",
    }
}

/// `enum ggml_ftype`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    MostlyQ2_K = 10,
    MostlyQ3_K = 11,
    MostlyQ4_K = 12,
    MostlyQ5_K = 13,
    MostlyQ6_K = 14,
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

/// `ggml_ftype_to_ggml_type`.
pub fn ggml_ftype_to_ggml_type(ftype: GgmlFtype) -> Result<GgmlType, Error> {
    use GgmlFtype::*;
    let t = match ftype {
        AllF32 => GgmlType::F32,
        MostlyF16 => GgmlType::F16,
        MostlyBf16 => GgmlType::BF16,
        MostlyQ4_0 => GgmlType::Q4_0,
        MostlyQ4_1 => GgmlType::Q4_1,
        MostlyQ1_0 => GgmlType::Q1_0,
        MostlyQ2_0 => GgmlType::Q2_0,
        MostlyQ5_0 => GgmlType::Q5_0,
        MostlyQ5_1 => GgmlType::Q5_1,
        MostlyQ8_0 => GgmlType::Q8_0,
        MostlyMxfp4 => GgmlType::MXFP4,
        MostlyNvfp4 => GgmlType::NVFP4,
        MostlyF8E4M3 => GgmlType::F8_E4M3,
        MostlyQ2_K => GgmlType::Q2_K,
        MostlyQ3_K => GgmlType::Q3_K,
        MostlyQ4_K => GgmlType::Q4_K,
        MostlyQ5_K => GgmlType::Q5_K,
        MostlyQ6_K => GgmlType::Q6_K,
        MostlyIq2Xxs => GgmlType::IQ2_XXS,
        MostlyIq2Xs => GgmlType::IQ2_XS,
        MostlyIq3Xxs => GgmlType::IQ3_XXS,
        MostlyIq1S => GgmlType::IQ1_S,
        MostlyIq1M => GgmlType::IQ1_M,
        MostlyIq4Nl => GgmlType::IQ4_NL,
        MostlyIq4Xs => GgmlType::IQ4_XS,
        MostlyIq3S => GgmlType::IQ3_S,
        MostlyIq2S => GgmlType::IQ2_S,
        Unknown | MostlyQ4_1SomeF16 => {
            return Err(Error::new("ggml_ftype_to_ggml_type: unsupported ftype"))
        }
    };
    Ok(t)
}

/// `enum ggml_tensor_flag`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlTensorFlag;

impl GgmlTensorFlag {
    pub const INPUT: i32 = 1;
    pub const OUTPUT: i32 = 2;
    pub const PARAM: i32 = 4;
    pub const LOSS: i32 = 8;
    pub const COMPUTE: i32 = 16;
}

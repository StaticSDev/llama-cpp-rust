//! Minimal tensor view used by the backend interface.
//!
//! The C++ backend code reads a handful of `ggml_tensor` fields. This module
//! keeps those fields in one owned struct so the backend traits need no
//! dependency on the rest of the ggml port. Byte storage is shared through
//! `Rc<RefCell<Vec<u8>>>`, which stands in for a `void * data` pointer into a
//! backend buffer.

use std::cell::RefCell;
use std::rc::Rc;

use crate::gguf::{ggml_blck_size, ggml_type_size, GgmlType};

use super::buffer::BufferRef;
use super::types::{MAX_DIMS, MAX_SRC};

/// `ggml_tensor *`.
pub type TensorRef = Rc<RefCell<Tensor>>;

/// Location of tensor data inside a buffer's byte storage.
#[derive(Clone, Debug)]
pub struct TensorData {
    pub storage: Rc<RefCell<Vec<u8>>>,
    pub offset: usize,
}

/// `enum ggml_op`, values match the C enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Op {
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
    Conv2DDW,
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
    Count,
}

impl Default for Op {
    fn default() -> Self {
        Self::None
    }
}

/// `struct ggml_cgraph` view: the parts the backend interface uses.
#[derive(Debug, Default)]
pub struct ComputeGraph {
    pub nodes: Vec<TensorRef>,
    pub leafs: Vec<TensorRef>,
}

/// `struct ggml_cgraph *`.
pub type GraphRef = Rc<RefCell<ComputeGraph>>;

/// `GGML_TENSOR_FLAG_*` values from `ggml.h`.
pub const TENSOR_FLAG_INPUT: u32 = 1;
pub const TENSOR_FLAG_OUTPUT: u32 = 2;
pub const TENSOR_FLAG_PARAM: u32 = 4;
pub const TENSOR_FLAG_LOSS: u32 = 8;
pub const TENSOR_FLAG_COMPUTE: u32 = 16;

/// Backend-visible view of `struct ggml_tensor`.
#[derive(Debug)]
pub struct Tensor {
    pub name: String,
    pub ne: [i64; MAX_DIMS],
    pub nb: [usize; MAX_DIMS],
    pub type_: GgmlType,
    pub op: Op,
    pub flags: u32,
    pub data: Option<TensorData>,
    pub buffer: Option<BufferRef>,
    pub view_src: Option<TensorRef>,
    pub view_offs: usize,
    pub src: [Option<TensorRef>; MAX_SRC],
    /// Stands in for `ggml_tensor::op_params`; used by meta op checks.
    pub op_params: [u8; 64],
    /// Whether FLASH_ATTN_EXT uses an indexed bias (derived from op_params).
    pub indexed_bias: bool,
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            name: String::new(),
            ne: [1; MAX_DIMS],
            nb: [4; MAX_DIMS],
            type_: GgmlType::F32,
            op: Op::None,
            flags: 0,
            data: None,
            buffer: None,
            view_src: None,
            view_offs: 0,
            src: std::array::from_fn(|_| None),
            op_params: [0; 64],
            indexed_bias: false,
        }
    }
}

impl Tensor {
    pub fn new(name: impl Into<String>, ne: [i64; MAX_DIMS], type_: GgmlType) -> Self {
        let ts = ggml_type_size(type_);
        let mut nb = [ts; MAX_DIMS];
        nb[0] = ts;
        for i in 1..MAX_DIMS {
            nb[i] = nb[i - 1] * ne[i - 1].max(0) as usize;
        }
        Self {
            name: name.into(),
            ne,
            nb,
            type_,
            ..Self::default()
        }
    }

    /// `ggml_nbytes`.
    pub fn nbytes(&self) -> usize {
        nbytes(self)
    }
}

/// `size_t ggml_nbytes(const struct ggml_tensor *)`.
pub fn nbytes(t: &Tensor) -> usize {
    let mut nelements = 1i128;
    for &ne in &t.ne {
        nelements *= ne as i128;
    }
    let blck = ggml_blck_size(t.type_).max(1) as i128;
    let n = nelements * ggml_type_size(t.type_) as i128 / blck;
    if n < 0 {
        0
    } else {
        n as usize
    }
}

pub fn nbytes_ref(t: &TensorRef) -> usize {
    nbytes(&t.borrow())
}

/// `int ggml_n_dims`.
pub fn n_dims(t: &Tensor) -> usize {
    let mut d = MAX_DIMS;
    while d > 1 && t.ne[d - 1] == 1 {
        d -= 1;
    }
    d
}

/// `bool ggml_is_contiguous`.
pub fn is_contiguous(t: &Tensor) -> bool {
    if t.nb[0] != ggml_type_size(t.type_) {
        return false;
    }
    (1..MAX_DIMS).all(|i| t.nb[i] == t.nb[i - 1] * t.ne[i - 1].max(0) as usize)
}

/// `bool ggml_are_same_layout`.
pub fn are_same_layout(a: &Tensor, b: &Tensor) -> bool {
    a.ne == b.ne && a.nb == b.nb
}

/// `bool ggml_op_is_empty`.
pub fn op_is_empty(op: Op) -> bool {
    matches!(
        op,
        Op::None | Op::Reshape | Op::Transpose | Op::View | Op::Permute
    )
}

/// `bool ggml_is_view_op`.
pub fn is_view_op(op: Op) -> bool {
    matches!(op, Op::View | Op::Reshape | Op::Permute | Op::Transpose)
}

/// `bool ggml_op_alloc_size_may_expand` from `ggml-backend.cpp`.
pub fn op_alloc_size_may_expand(op: Op) -> bool {
    matches!(
        op,
        Op::FlashAttnExt
            | Op::MulMat
            | Op::MulMatId
            | Op::Cumsum
            | Op::Argsort
            | Op::TopK
            | Op::Argmax
            | Op::Conv2D
    )
}

/// Buffer that owns the tensor data, following `view_src` like the C++ code.
pub fn effective_buffer(t: &TensorRef) -> Option<BufferRef> {
    let b = t.borrow();
    b.view_src
        .as_ref()
        .and_then(|s| s.borrow().buffer.clone())
        .or_else(|| b.buffer.clone())
}

/// Data location of a tensor.
pub fn data_location(t: &TensorRef) -> Option<TensorData> {
    t.borrow().data.clone()
}

/// `GGML_PAD(x, alignment)`.
pub fn pad(x: usize, alignment: usize) -> usize {
    crate::gguf::gguf_pad(x, alignment)
}

/// Stand-in for `ggml_flash_attn_ext_has_indexed_bias`.
pub fn flash_attn_ext_has_indexed_bias(t: &Tensor) -> bool {
    t.op == Op::FlashAttnExt && t.indexed_bias
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nbytes_and_alignment() {
        let t = Tensor::new("x", [4, 2, 1, 1], GgmlType::F32);
        assert_eq!(t.nbytes(), 32);
        assert_eq!(pad(33, 32), 64);
    }

    #[test]
    fn op_tables_match_reference() {
        assert!(op_alloc_size_may_expand(Op::MulMat));
        assert!(!op_alloc_size_may_expand(Op::Add));
        assert!(is_view_op(Op::View));
        assert!(op_is_empty(Op::None));
    }
}

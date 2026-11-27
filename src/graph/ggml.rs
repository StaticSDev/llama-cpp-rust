//! Local stand-ins for the ggml tensor API used by the graph builder.
//!
//! The C++ builder allocates tensors in a `ggml_context` and records ops in a
//! `ggml_cgraph`.  Here each tensor is an `Rc<RefCell<TensorData>>`: ops build
//! new nodes with the same shape/stride metadata ggml would compute, and the
//! `Graph` recorder walks the `src` links.  Input tensors own a host byte
//! buffer so the `set_input()` methods can copy data without FFI or `unsafe`.

use std::cell::RefCell;
use std::rc::Rc;

/// `ggml_type` values referenced by the graph builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Type {
    F32,
    F16,
    BF16,
    I32,
    I64,
    F8E4M3,
    NVFP4,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Type {
    /// `ggml_element_size`.
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
            Self::F16 | Self::BF16 => 2,
            Self::F8E4M3 | Self::NVFP4 => 1,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    /// `ggml_blck_size`.
    pub fn blck_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::I32 | Self::I64
            | Self::F8E4M3 | Self::NVFP4 => 1,
            _ => 32,
        }
    }

    /// `ggml_row_size`.
    pub fn row_size(self, n: i64) -> usize {
        let n = n.max(0) as usize;
        (n / self.blck_size()) * self.type_size()
    }
}

/// `ggml_op` values needed to recognize graph nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    None,
    Add,
    AddId,
    Abs,
    Arange,
    ArgsortTopK,
    Cast,
    Cont,
    Cpy,
    Div,
    Dup,
    Exp,
    Fill,
    FlashAttnExt,
    Gelu,
    GeluErf,
    Geglu,
    GetRows,
    GroupNorm,
    Log,
    Ceil,
    Clamp,
    Mul,
    MulMat,
    MulMatId,
    Norm,
    Pad,
    Permute,
    Relu,
    RmsNorm,
    Scale,
    SetRows,
    Sigmoid,
    Silu,
    SoftMax,
    Softplus,
    Sqr,
    Sqrt,
    SumRows,
    Tanh,
    TopKMask,
    Transpose,
    Unary,
    View,
    Swiglu,
    SwigluSplit,
    SwigluOai,
    GegluSplit,
    GegluErfSplit,
    RegluSplit,
    ReluSqr,
    Situ,
    Other,
}

/// `ggml_prec`: accumulation precision flags set on graph nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prec {
    F32,
    F16,
    BF16,
}

#[derive(Debug)]
pub struct TensorData {
    pub ty: Type,
    pub ne: [i64; 4],
    pub nb: [usize; 4],
    pub offset: usize,
    pub name: String,
    pub op: Op,
    pub src: Vec<Option<Tensor>>,
    pub is_input: bool,
    pub is_output: bool,
    pub contiguous: bool,
    pub prec: Prec,
    pub n_kv_max: i32,
    pub data: Vec<u8>,
}

/// Shared handle to a tensor node. Clones point at the same node, as C++
/// pointers do.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub inner: Rc<RefCell<TensorData>>,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}
impl Eq for Tensor {}

impl Tensor {
    /// `ggml_new_tensor`: allocate the host buffer for an input/weight tensor.
    pub fn new(ty: Type, ne: [i64; 4]) -> Self {
        let mut nb = [0usize; 4];
        nb[0] = ty.type_size();
        for i in 1..4 {
            nb[i] = nb[i - 1] * ne[i - 1].max(0) as usize;
        }
        let nbytes = ty.row_size(ne[0]) * ne[1].max(0) as usize
            * ne[2].max(0) as usize
            * ne[3].max(0) as usize;
        Self {
            inner: Rc::new(RefCell::new(TensorData {
                ty,
                ne,
                nb,
                offset: 0,
                name: String::new(),
                op: Op::None,
                src: Vec::new(),
                is_input: false,
                is_output: false,
                contiguous: true,
                prec: Prec::F32,
                n_kv_max: 0,
                data: vec![0u8; nbytes],
            })),
        }
    }

    pub fn new_1d(ty: Type, ne0: i64) -> Self {
        Self::new(ty, [ne0, 1, 1, 1])
    }

    pub fn new_2d(ty: Type, ne0: i64, ne1: i64) -> Self {
        Self::new(ty, [ne0, ne1, 1, 1])
    }

    pub fn new_3d(ty: Type, ne0: i64, ne1: i64, ne2: i64) -> Self {
        Self::new(ty, [ne0, ne1, ne2, 1])
    }

    pub fn new_4d(ty: Type, ne: [i64; 4]) -> Self {
        Self::new(ty, ne)
    }

    /// Identity for the caches keyed by tensor pointer.
    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.inner) as usize
    }

    pub fn ne(&self, i: usize) -> i64 {
        self.inner.borrow().ne[i]
    }

    pub fn ne_all(&self) -> [i64; 4] {
        self.inner.borrow().ne
    }

    pub fn set_ne(&self, i: usize, value: i64) {
        self.inner.borrow_mut().ne[i] = value;
    }

    pub fn nb(&self, i: usize) -> usize {
        self.inner.borrow().nb[i]
    }

    pub fn nb_all(&self) -> [usize; 4] {
        self.inner.borrow().nb
    }

    pub fn ty(&self) -> Type {
        self.inner.borrow().ty
    }

    pub fn op(&self) -> Op {
        self.inner.borrow().op
    }

    pub fn nelements(&self) -> usize {
        let ne = self.ne_all();
        ne.iter().map(|&n| n.max(0) as usize).product()
    }

    /// `ggml_nbytes`.
    pub fn nbytes(&self) -> usize {
        let ne = self.ne_all();
        self.ty().row_size(ne[0]) * ne[1].max(0) as usize
            * ne[2].max(0) as usize
            * ne[3].max(0) as usize
    }

    pub fn name(&self) -> String {
        self.inner.borrow().name.clone()
    }

    pub fn set_name(&self, name: impl Into<String>) {
        self.inner.borrow_mut().name = name.into();
    }

    pub fn is_input(&self) -> bool {
        self.inner.borrow().is_input
    }

    pub fn is_output(&self) -> bool {
        self.inner.borrow().is_output
    }

    pub fn set_input(&self) {
        self.inner.borrow_mut().is_input = true;
    }

    pub fn set_output(&self) {
        self.inner.borrow_mut().is_output = true;
    }

    pub fn is_contiguous(&self) -> bool {
        self.inner.borrow().contiguous
    }

    pub fn set_contiguous(&self, value: bool) {
        self.inner.borrow_mut().contiguous = value;
    }

    pub fn src(&self, i: usize) -> Option<Tensor> {
        self.inner.borrow().src.get(i).cloned().flatten()
    }

    pub fn set_src(&self, i: usize, tensor: Option<Tensor>) {
        let mut d = self.inner.borrow_mut();
        if d.src.len() <= i {
            d.src.resize(i + 1, None);
        }
        d.src[i] = tensor;
    }

    pub fn add_src(&self, tensor: Tensor) {
        self.inner.borrow_mut().src.push(Some(tensor));
    }

    pub fn prec(&self) -> Prec {
        self.inner.borrow().prec
    }

    pub fn set_prec(&self, prec: Prec) {
        self.inner.borrow_mut().prec = prec;
    }

    pub fn set_n_kv_max(&self, n: i32) {
        self.inner.borrow_mut().n_kv_max = n;
    }

    /// `ggml_are_same_shape`.
    pub fn same_shape(&self, other: &Tensor) -> bool {
        self.ne_all() == other.ne_all()
    }

    /// `ggml_are_same_stride`.
    pub fn same_stride(&self, other: &Tensor) -> bool {
        self.nb_all() == other.nb_all()
    }

    /// Zero the host buffer of an input tensor.
    pub fn zero_data(&self) {
        self.inner.borrow_mut().data.fill(0);
    }

    /// Read back host data, for tests and input verification.
    pub fn data_i32(&self) -> Vec<i32> {
        let d = self.inner.borrow();
        let n = d.data.len() / 4;
        (0..n)
            .map(|i| i32::from_le_bytes(d.data[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect()
    }

    pub fn data_i64(&self) -> Vec<i64> {
        let d = self.inner.borrow();
        let n = d.data.len() / 8;
        (0..n)
            .map(|i| i64::from_le_bytes(d.data[i * 8..i * 8 + 8].try_into().unwrap()))
            .collect()
    }
}

impl Tensor {
    /// Copy host data into an input tensor (stand-in for
    /// `ggml_backend_tensor_set`).
    pub fn set_data_i32(&self, values: &[i32]) {
        let mut d = self.inner.borrow_mut();
        let n = values.len().min(d.data.len() / 4);
        for (i, v) in values.iter().take(n).enumerate() {
            d.data[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }

    pub fn set_data_i64(&self, values: &[i64]) {
        let mut d = self.inner.borrow_mut();
        let n = values.len().min(d.data.len() / 8);
        for (i, v) in values.iter().take(n).enumerate() {
            d.data[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
    }

    pub fn set_data_f32(&self, values: &[f32]) {
        let mut d = self.inner.borrow_mut();
        let n = values.len().min(d.data.len() / 4);
        for (i, v) in values.iter().take(n).enumerate() {
            d.data[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }

    /// Copy raw bytes, for tensors whose logical element type differs from
    /// the host representation (for example the I32 CLS indices).
    pub fn set_data_raw(&self, values: &[u8]) {
        let mut d = self.inner.borrow_mut();
        let n = values.len().min(d.data.len());
        d.data[..n].copy_from_slice(&values[..n]);
    }

    pub fn data_f32(&self) -> Vec<f32> {
        let d = self.inner.borrow();
        let n = d.data.len() / 4;
        (0..n)
            .map(|i| f32::from_le_bytes(d.data[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect()
    }
}

fn contiguous_strides(ty: Type, ne: [i64; 4]) -> [usize; 4] {
    let mut nb = [0usize; 4];
    nb[0] = ty.type_size();
    for i in 1..4 {
        nb[i] = nb[i - 1] * ne[i - 1].max(0) as usize;
    }
    nb
}

fn node(op: Op, ty: Type, ne: [i64; 4], src: &[Option<Tensor>]) -> Tensor {
    let t = Tensor::new(ty, ne);
    {
        let mut d = t.inner.borrow_mut();
        d.op = op;
        d.nb = contiguous_strides(ty, ne);
        d.contiguous = true;
        d.src = src.to_vec();
    }
    t
}

fn broadcast_ne(a: &Tensor, b: &Tensor) -> [i64; 4] {
    let na = a.ne_all();
    let nb = b.ne_all();
    let mut ne = [1i64; 4];
    for i in 0..4 {
        ne[i] = if na[i] == 1 { nb[i] } else if nb[i] == 1 { na[i] } else { na[i].max(nb[i]) };
    }
    ne
}

fn binary(op: Op, a: &Tensor, b: &Tensor) -> Tensor {
    node(op, a.ty(), broadcast_ne(a, b), &[Some(a.clone()), Some(b.clone())])
}

fn unary(op: Op, a: &Tensor) -> Tensor {
    node(op, a.ty(), a.ne_all(), &[Some(a.clone())])
}

// --- constructors -----------------------------------------------------------

pub fn new_tensor_1d(ty: Type, ne0: i64) -> Tensor {
    Tensor::new_1d(ty, ne0)
}
pub fn new_tensor_2d(ty: Type, ne0: i64, ne1: i64) -> Tensor {
    Tensor::new_2d(ty, ne0, ne1)
}
pub fn new_tensor_3d(ty: Type, ne0: i64, ne1: i64, ne2: i64) -> Tensor {
    Tensor::new_3d(ty, ne0, ne1, ne2)
}
pub fn new_tensor_4d(ty: Type, ne: [i64; 4]) -> Tensor {
    Tensor::new_4d(ty, ne)
}

// --- views / reshape --------------------------------------------------------

pub fn view_1d(a: &Tensor, ne0: i64, offset: usize) -> Tensor {
    let t = Tensor::new_1d(a.ty(), ne0);
    {
        let mut d = t.inner.borrow_mut();
        d.op = Op::View;
        d.offset = a.nb(0) * offset + a.inner.borrow().offset;
        d.nb = a.nb_all();
        d.contiguous = false;
        d.src = vec![Some(a.clone())];
    }
    t
}

pub fn view_2d(a: &Tensor, ne0: i64, ne1: i64, nb1: usize, offset: usize) -> Tensor {
    let t = Tensor::new_2d(a.ty(), ne0, ne1);
    {
        let mut d = t.inner.borrow_mut();
        d.op = Op::View;
        d.nb = [a.ty().type_size(), nb1, nb1 * ne1.max(0) as usize, nb1 * ne1.max(0) as usize];
        d.offset = offset + a.inner.borrow().offset;
        d.contiguous = false;
        d.src = vec![Some(a.clone())];
    }
    t
}

pub fn view_3d(
    a: &Tensor,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    nb1: usize,
    nb2: usize,
    offset: usize,
) -> Tensor {
    let t = Tensor::new_3d(a.ty(), ne0, ne1, ne2);
    {
        let mut d = t.inner.borrow_mut();
        d.op = Op::View;
        d.nb = [a.ty().type_size(), nb1, nb2, nb2 * ne2.max(0) as usize];
        d.offset = offset + a.inner.borrow().offset;
        d.contiguous = false;
        d.src = vec![Some(a.clone())];
    }
    t
}

pub fn view_4d(
    a: &Tensor,
    ne: [i64; 4],
    nb1: usize,
    nb2: usize,
    nb3: usize,
    offset: usize,
) -> Tensor {
    let t = Tensor::new_4d(a.ty(), ne);
    {
        let mut d = t.inner.borrow_mut();
        d.op = Op::View;
        d.nb = [a.ty().type_size(), nb1, nb2, nb3];
        d.offset = offset + a.inner.borrow().offset;
        d.contiguous = false;
        d.src = vec![Some(a.clone())];
    }
    t
}

fn reshape(a: &Tensor, ne: [i64; 4]) -> Tensor {
    let t = Tensor::new_4d(a.ty(), ne);
    {
        let mut d = t.inner.borrow_mut();
        d.op = Op::View;
        d.nb = contiguous_strides(a.ty(), ne);
        d.offset = a.inner.borrow().offset;
        d.contiguous = true;
        d.src = vec![Some(a.clone())];
    }
    t
}

pub fn reshape_1d(a: &Tensor, ne0: i64) -> Tensor {
    reshape(a, [ne0, 1, 1, 1])
}
pub fn reshape_2d(a: &Tensor, ne0: i64, ne1: i64) -> Tensor {
    reshape(a, [ne0, ne1, 1, 1])
}
pub fn reshape_3d(a: &Tensor, ne0: i64, ne1: i64, ne2: i64) -> Tensor {
    reshape(a, [ne0, ne1, ne2, 1])
}
pub fn reshape_4d(a: &Tensor, ne: [i64; 4]) -> Tensor {
    reshape(a, ne)
}

pub fn permute(a: &Tensor, a0: usize, a1: usize, a2: usize, a3: usize) -> Tensor {
    let old = a.ne_all();
    let nb = a.nb_all();
    let axes = [a0, a1, a2, a3];
    let ne = axes.map(|x| old[x]);
    let new_nb = axes.map(|x| nb[x]);
    let t = Tensor::new_4d(a.ty(), ne);
    {
        let mut d = t.inner.borrow_mut();
        d.op = Op::Permute;
        d.nb = new_nb;
        d.offset = a.inner.borrow().offset;
        d.contiguous = false;
        d.src = vec![Some(a.clone())];
    }
    t
}

pub fn transpose(a: &Tensor) -> Tensor {
    permute(a, 1, 0, 2, 3)
}

pub fn cont(a: &Tensor) -> Tensor {
    if a.is_contiguous() {
        return a.clone();
    }
    let t = unary(Op::Cont, a);
    t.set_contiguous(true);
    t
}

pub fn cont_2d(a: &Tensor, ne0: i64, ne1: i64) -> Tensor {
    let t = reshape_2d(a, ne0, ne1);
    t.set_contiguous(true);
    t
}

pub fn dup(a: &Tensor) -> Tensor {
    node(Op::Dup, a.ty(), a.ne_all(), &[Some(a.clone())])
}

pub fn concat(a: &Tensor, b: &Tensor, dim: i32) -> Tensor {
    let mut ne = broadcast_ne(a, b);
    let d = dim.max(0) as usize;
    ne[d] = a.ne(d) + b.ne(d);
    node(Op::Other, a.ty(), ne, &[Some(a.clone()), Some(b.clone())])
}

pub fn repeat_4d(a: &Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> Tensor {
    node(
        Op::Other,
        a.ty(),
        [ne0, ne1, ne2, ne3],
        &[Some(a.clone())],
    )
}

pub fn pad(a: &Tensor, p0: i64, p1: i64, p2: i64, p3: i64) -> Tensor {
    let ne = a.ne_all();
    let ne = [ne[0] + p0, ne[1] + p1, ne[2] + p2, ne[3] + p3];
    node(Op::Pad, a.ty(), ne, &[Some(a.clone())])
}

// --- arithmetic -------------------------------------------------------------

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    binary(Op::Add, a, b)
}
pub fn add_id(a: &Tensor, b: &Tensor, ids: &Tensor) -> Tensor {
    node(Op::AddId, a.ty(), a.ne_all(), &[Some(a.clone()), Some(b.clone()), Some(ids.clone())])
}
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    binary(Op::Other, a, b)
}
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    binary(Op::Mul, a, b)
}
pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    binary(Op::Div, a, b)
}
pub fn scale(a: &Tensor, _value: f32) -> Tensor {
    unary(Op::Scale, a)
}
pub fn scale_bias(a: &Tensor, _scale: f32, _bias: f32) -> Tensor {
    unary(Op::Scale, a)
}
pub fn sqr(a: &Tensor) -> Tensor {
    unary(Op::Sqr, a)
}
pub fn sqrt(a: &Tensor) -> Tensor {
    unary(Op::Sqrt, a)
}
pub fn abs(a: &Tensor) -> Tensor {
    unary(Op::Abs, a)
}
pub fn log(a: &Tensor) -> Tensor {
    unary(Op::Log, a)
}
pub fn ceil(a: &Tensor) -> Tensor {
    unary(Op::Ceil, a)
}
pub fn exp(a: &Tensor) -> Tensor {
    unary(Op::Exp, a)
}
pub fn clamp(a: &Tensor, _lo: f32, _hi: f32) -> Tensor {
    unary(Op::Clamp, a)
}
pub fn tanh(a: &Tensor) -> Tensor {
    unary(Op::Tanh, a)
}
pub fn relu(a: &Tensor) -> Tensor {
    unary(Op::Relu, a)
}
pub fn sigmoid(a: &Tensor) -> Tensor {
    unary(Op::Sigmoid, a)
}
pub fn silu(a: &Tensor) -> Tensor {
    unary(Op::Silu, a)
}
pub fn gelu(a: &Tensor) -> Tensor {
    unary(Op::Gelu, a)
}
pub fn gelu_erf(a: &Tensor) -> Tensor {
    unary(Op::GeluErf, a)
}
pub fn softplus(a: &Tensor) -> Tensor {
    unary(Op::Softplus, a)
}
pub fn cast(a: &Tensor, ty: Type) -> Tensor {
    let t = node(Op::Cast, ty, a.ne_all(), &[Some(a.clone())]);
    t
}

pub fn unary_op(a: &Tensor, _op: u32) -> Tensor {
    unary(Op::Unary, a)
}

pub fn swiglu(a: &Tensor) -> Tensor {
    unary(Op::Swiglu, a)
}
pub fn geglu(a: &Tensor) -> Tensor {
    unary(Op::GegluSplit, a)
}
pub fn geglu_erf(a: &Tensor) -> Tensor {
    unary(Op::GegluErfSplit, a)
}
pub fn reglu(a: &Tensor) -> Tensor {
    unary(Op::RegluSplit, a)
}
pub fn swiglu_split(gate: &Tensor, up: &Tensor) -> Tensor {
    binary(Op::SwigluSplit, gate, up)
}
pub fn swiglu_clamp(gate: &Tensor, up: &Tensor, _limit: f32) -> Tensor {
    binary(Op::Swiglu, gate, up)
}
pub fn swiglu_oai(gate: &Tensor, up: &Tensor, _alpha: f32, _limit: f32) -> Tensor {
    binary(Op::SwigluOai, gate, up)
}
pub fn geglu_split(gate: &Tensor, up: &Tensor) -> Tensor {
    binary(Op::GegluSplit, gate, up)
}
pub fn geglu_erf_split(gate: &Tensor, up: &Tensor) -> Tensor {
    binary(Op::GegluErfSplit, gate, up)
}
pub fn reglu_split(gate: &Tensor, up: &Tensor) -> Tensor {
    binary(Op::RegluSplit, gate, up)
}
pub fn situ(gate: &Tensor, up: &Tensor, _beta: f32, _lb: f32) -> Tensor {
    binary(Op::Situ, gate, up)
}

pub fn norm(a: &Tensor, _eps: f32) -> Tensor {
    unary(Op::Norm, a)
}
pub fn rms_norm(a: &Tensor, _eps: f32) -> Tensor {
    unary(Op::RmsNorm, a)
}
pub fn group_norm(a: &Tensor, _groups: i64, _eps: f32) -> Tensor {
    unary(Op::GroupNorm, a)
}
pub fn soft_max(a: &Tensor) -> Tensor {
    unary(Op::SoftMax, a)
}
pub fn soft_max_ext(a: &Tensor, mask: &Tensor, _scale: f32, _max_bias: f32) -> Tensor {
    node(Op::SoftMax, a.ty(), a.ne_all(), &[Some(a.clone()), Some(mask.clone())])
}
pub fn soft_max_add_sinks(a: &Tensor, sinks: &Tensor) {
    a.add_src(sinks.clone());
}
pub fn sum_rows(a: &Tensor) -> Tensor {
    let ne = a.ne_all();
    node(Op::SumRows, a.ty(), [1, ne[1], ne[2], ne[3]], &[Some(a.clone())])
}
pub fn argsort_top_k(a: &Tensor, k: i64) -> Tensor {
    let ne = a.ne_all();
    node(Op::ArgsortTopK, Type::I32, [k, ne[1], ne[2], ne[3]], &[Some(a.clone())])
}
pub fn arange(start: f32, stop: f32, step: f32) -> Tensor {
    let steps = ((stop - start) / step).ceil().max(0.0) as i64;
    node(Op::Arange, Type::F32, [steps, 1, 1, 1], &[])
}
pub fn fill(a: &Tensor, _value: f32) -> Tensor {
    unary(Op::Fill, a)
}
pub fn fill_inplace(a: &Tensor, _value: f32) -> Tensor {
    let t = unary(Op::Fill, a);
    t
}

pub fn mul_mat(a: &Tensor, b: &Tensor) -> Tensor {
    let na = a.ne_all();
    let nb = b.ne_all();
    let ne = [na[1], nb[1], nb[2], nb[3]];
    node(Op::MulMat, Type::F32, ne, &[Some(a.clone()), Some(b.clone())])
}

pub fn mul_mat_id(a: &Tensor, b: &Tensor, ids: &Tensor) -> Tensor {
    let na = a.ne_all();
    let nb = b.ne_all();
    let ni = ids.ne_all();
    let ne = [na[1], ni[0], nb[2], 1];
    node(
        Op::MulMatId,
        Type::F32,
        ne,
        &[Some(a.clone()), Some(b.clone()), Some(ids.clone())],
    )
}

pub fn get_rows(a: &Tensor, ids: &Tensor) -> Tensor {
    let na = a.ne_all();
    let ni = ids.ne_all();
    let ne = [na[0], ni[0], ni[1], ni[2]];
    node(Op::GetRows, a.ty(), ne, &[Some(a.clone()), Some(ids.clone())])
}

pub fn set_rows(dst: &Tensor, src: &Tensor, ids: &Tensor) -> Tensor {
    node(
        Op::SetRows,
        dst.ty(),
        dst.ne_all(),
        &[Some(dst.clone()), Some(src.clone()), Some(ids.clone())],
    )
}

pub fn cpy(src: &Tensor, dst: &Tensor) -> Tensor {
    node(Op::Cpy, dst.ty(), dst.ne_all(), &[Some(src.clone())])
}

pub fn top_k_mask(mask: &Tensor, top_k: &Tensor, _min_top_k: Option<&Tensor>, _output_rows: Option<&Tensor>, _bias: Option<&Tensor>) -> Tensor {
    node(
        Op::TopKMask,
        mask.ty(),
        mask.ne_all(),
        &[Some(mask.clone()), Some(top_k.clone())],
    )
}

pub fn flash_attn_ext(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, rows: Option<&Tensor>, _scale: f32, _max_bias: f32, _softcap: f32) -> Tensor {
    let nq = q.ne_all();
    let t = node(
        Op::FlashAttnExt,
        Type::F32,
        [nq[0], nq[1], nq[2], nq[3]],
        &[Some(q.clone()), Some(k.clone()), Some(v.clone()), Some(mask.clone()), rows.cloned(), None],
    );
    t
}

pub fn flash_attn_ext_rows(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, rows: &Tensor, scale: f32, max_bias: f32, softcap: f32) -> Tensor {
    flash_attn_ext(q, k, v, mask, Some(rows), scale, max_bias, softcap)
}

pub fn flash_attn_ext_add_sinks(a: &Tensor, sinks: &Tensor) {
    a.set_src(4, Some(sinks.clone()));
}

pub fn flash_attn_ext_set_n_kv_max(a: &Tensor, n: i32) {
    a.set_n_kv_max(n);
}

pub fn flash_attn_ext_add_indexed_bias(a: &Tensor, bias: Option<&Tensor>, indices: Option<&Tensor>) {
    a.set_src(6, bias.cloned());
    a.set_src(7, indices.cloned());
}

pub fn prec_set_acc(a: &Tensor, p: Prec) {
    a.set_prec(p);
}

pub fn pool_1d(a: &Tensor, _op: Op, k: i64, _d: i64, _p: i64) -> Tensor {
    let ne = a.ne_all();
    node(Op::Other, a.ty(), [(ne[0] + k - 1) / k, ne[1], ne[2], ne[3]], &[Some(a.clone())])
}

pub fn rope(a: &Tensor, pos: &Tensor) -> Tensor {
    node(Op::Other, a.ty(), a.ne_all(), &[Some(a.clone()), Some(pos.clone())])
}

pub fn rope_multi(a: &Tensor, pos: &Tensor) -> Tensor {
    rope(a, pos)
}

pub fn rope_set_offset(a: &Tensor, _n_offs: i64) -> Tensor {
    unary(Op::Other, a)
}

pub fn format_name(a: &Tensor, text: impl Into<String>) {
    a.set_name(text);
}

/// `ggml_cgraph`: record the ops reachable from a tensor, in dependency order.
#[derive(Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Tensor>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn node(&self, i: usize) -> Option<&Tensor> {
        self.nodes.get(i)
    }

    /// `ggml_build_forward_expand`: append dependencies of `t` to the graph.
    pub fn build_forward_expand(&mut self, t: &Tensor) {
        let mut seen: Vec<usize> = self.nodes.iter().map(|n| n.id()).collect();
        let mut order = Vec::new();
        expand_rec(t, &mut seen, &mut order);
        self.nodes.extend(order);
    }

    /// `ggml_build_forward_order`: same dependency walk, used to pin GET_ROWS
    /// nodes to the front of the graph.
    pub fn build_forward_order(&mut self, t: &Tensor) {
        self.build_forward_expand(t);
    }

    /// `ggml_build_forward_select`: build every branch, keep `idx`.
    pub fn build_forward_select(&mut self, inps: &[Tensor], idx: usize) -> Tensor {
        for t in inps {
            self.build_forward_expand(t);
        }
        inps.get(idx).cloned().unwrap_or_else(|| inps[0].clone())
    }
}

fn expand_rec(t: &Tensor, seen: &mut Vec<usize>, order: &mut Vec<Tensor>) {
    let id = t.id();
    if seen.contains(&id) {
        return;
    }
    seen.push(id);
    let srcs = t.inner.borrow().src.clone();
    for s in srcs.into_iter().flatten() {
        expand_rec(&s, seen, order);
    }
    order.push(t.clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_follow_ggml_rules() {
        let w = new_tensor_2d(Type::F32, 8, 16);
        let cur = new_tensor_2d(Type::F32, 8, 3);
        let mm = mul_mat(&w, &cur);
        assert_eq!(mm.ne_all(), [16, 3, 1, 1]);

        let ids = new_tensor_1d(Type::I32, 4);
        let rows = get_rows(&w, &ids);
        assert_eq!(rows.ne_all(), [8, 4, 1, 1]);

        let p = permute(&rows, 0, 2, 1, 3);
        assert_eq!(p.ne_all(), [8, 1, 4, 1]);
    }

    #[test]
    fn graph_records_dependencies_once() {
        let a = new_tensor_2d(Type::F32, 4, 2);
        let b = new_tensor_2d(Type::F32, 4, 2);
        let c = add(&a, &b);
        let d = mul(&c, &a);
        let mut gf = Graph::new();
        gf.build_forward_expand(&d);
        assert_eq!(gf.n_nodes(), 4);
        gf.build_forward_expand(&c);
        assert_eq!(gf.n_nodes(), 4);
    }

    #[test]
    fn input_data_writes_are_typed() {
        let t = new_tensor_1d(Type::I32, 3);
        t.set_data_i32(&[7, 8, 9]);
        assert_eq!(t.data_i32(), vec![7, 8, 9]);
    }
}

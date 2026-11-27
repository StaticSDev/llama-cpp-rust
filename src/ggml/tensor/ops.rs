//! Arithmetic, unary, reduction and set/scale graph operations.

use std::sync::Arc;

use crate::core::{Error, Result};

use super::create::{
    ggml_dup_tensor, ggml_new_tensor, ggml_new_tensor_1d, ggml_new_tensor_4d, ggml_view_tensor,
};
use super::helpers::{
    ggml_are_same_shape, ggml_can_repeat, ggml_can_repeat_rows, ggml_is_contiguous,
    ggml_is_contiguous_rows, ggml_is_matrix, ggml_nelements,
};
use super::types::{ggml_is_quantized, GgmlOp, GgmlType, GgmlUnaryOp, GGML_MAX_DIMS};
use super::views::finish;
use super::Tensor;

/// Shared body of the in-place/out-of-place unary ops.
fn unary_result(a: &Arc<Tensor>, inplace: bool) -> Result<Tensor> {
    let t = if inplace { ggml_view_tensor(a)? } else { ggml_dup_tensor(a)? };
    Ok(Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone()))
}

/// Shared body of the in-place/out-of-place binary ops.
fn binary_result(a: &Arc<Tensor>, b: &Arc<Tensor>, inplace: bool, check_repeat: bool) -> Result<Tensor> {
    if check_repeat && !ggml_can_repeat(b, a) {
        return Err(Error::new("binary op: b cannot be repeated to match a"));
    }
    unary_result(a, inplace)
}

/// `ggml_dup_impl`.
pub fn ggml_dup_impl(a: &Arc<Tensor>, inplace: bool) -> Result<Arc<Tensor>> {
    let t = unary_result(a, inplace)?;
    Ok(finish(t, GgmlOp::Dup, &[a]))
}

/// `ggml_dup`.
pub fn ggml_dup(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    ggml_dup_impl(a, false)
}

/// `ggml_dup_inplace`.
pub fn ggml_dup_inplace(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    ggml_dup_impl(a, true)
}

/// `ggml_add_impl`.
pub fn ggml_add_impl(a: &Arc<Tensor>, b: &Arc<Tensor>, inplace: bool) -> Result<Arc<Tensor>> {
    let t = binary_result(a, b, inplace, true)?;
    Ok(finish(t, GgmlOp::Add, &[a, b]))
}

/// `ggml_add`.
pub fn ggml_add(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    ggml_add_impl(a, b, false)
}

/// `ggml_add_inplace`.
pub fn ggml_add_inplace(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    ggml_add_impl(a, b, true)
}

/// `ggml_add_cast`.
pub fn ggml_add_cast(a: &Arc<Tensor>, b: &Arc<Tensor>, type_: GgmlType) -> Result<Arc<Tensor>> {
    if !ggml_can_repeat_rows(b, a) {
        return Err(Error::new("ggml_add_cast: b cannot be repeated to match a"));
    }
    if !(ggml_is_quantized(a.type_) || a.type_ == GgmlType::F16 || a.type_ == GgmlType::BF16) {
        return Err(Error::new("ggml_add_cast: unsupported source type"));
    }
    let t = ggml_new_tensor(type_, GGML_MAX_DIMS, &a.ne)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::Add, &[a, b]))
}

/// `ggml_add_id`.
pub fn ggml_add_id(a: &Arc<Tensor>, b: &Arc<Tensor>, ids: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    if a.ne[0] != b.ne[0] || a.ne[1] != ids.ne[0] || a.ne[2] != ids.ne[1] || ids.type_ != GgmlType::I32 {
        return Err(Error::new("ggml_add_id: incompatible shapes"));
    }
    let t = ggml_dup_tensor(a)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::AddId, &[a, b, ids]))
}

/// Pack the five `size_t`/bool values used by ACC and SET into op params.
fn set_nb_params(t: &mut Tensor, nb1: usize, nb2: usize, nb3: usize, offset: usize, inplace: bool) -> Result<()> {
    if offset >= 1 << 30 {
        return Err(Error::new("op offset must be less than 1 << 30"));
    }
    t.set_op_params_i32_slice(&[
        nb1 as i32,
        nb2 as i32,
        nb3 as i32,
        offset as i32,
        if inplace { 1 } else { 0 },
    ]);
    Ok(())
}

/// `ggml_acc_impl`.
pub fn ggml_acc_impl(
    a: &Arc<Tensor>,
    b: &Arc<Tensor>,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    offset: usize,
    inplace: bool,
) -> Result<Arc<Tensor>> {
    if ggml_nelements(b) > ggml_nelements(a) {
        return Err(Error::new("ggml_acc: b has more elements than a"));
    }
    if !ggml_is_contiguous(a) || a.type_ != GgmlType::F32 || b.type_ != GgmlType::F32 {
        return Err(Error::new("ggml_acc: tensors must be contiguous F32"));
    }
    let mut t = unary_result(a, inplace)?;
    set_nb_params(&mut t, nb1, nb2, nb3, offset, inplace)?;
    Ok(finish(t, GgmlOp::Acc, &[a, b]))
}

/// `ggml_acc`.
pub fn ggml_acc(
    a: &Arc<Tensor>,
    b: &Arc<Tensor>,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    offset: usize,
) -> Result<Arc<Tensor>> {
    ggml_acc_impl(a, b, nb1, nb2, nb3, offset, false)
}

/// `ggml_acc_inplace`.
pub fn ggml_acc_inplace(
    a: &Arc<Tensor>,
    b: &Arc<Tensor>,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    offset: usize,
) -> Result<Arc<Tensor>> {
    ggml_acc_impl(a, b, nb1, nb2, nb3, offset, true)
}

macro_rules! binary_op {
    ($name:ident, $inplace:ident, $op:expr) => {
        /// Out-of-place form.
        pub fn $name(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
            binary_result_check(a, b, false, $op)
        }
        /// In-place form.
        pub fn $inplace(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
            binary_result_check(a, b, true, $op)
        }
    };
}

fn binary_result_check(a: &Arc<Tensor>, b: &Arc<Tensor>, inplace: bool, op: GgmlOp) -> Result<Arc<Tensor>> {
    let t = binary_result(a, b, inplace, true)?;
    Ok(finish(t, op, &[a, b]))
}

binary_op!(ggml_sub, ggml_sub_inplace, GgmlOp::Sub);
binary_op!(ggml_mul, ggml_mul_inplace, GgmlOp::Mul);
binary_op!(ggml_div, ggml_div_inplace, GgmlOp::Div);

macro_rules! unary_op {
    ($name:ident, $inplace:ident, $op:expr) => {
        /// Out-of-place form.
        pub fn $name(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
            unary_op_build(a, false, $op)
        }
        /// In-place form.
        pub fn $inplace(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
            unary_op_build(a, true, $op)
        }
    };
}

fn unary_op_build(a: &Arc<Tensor>, inplace: bool, op: GgmlOp) -> Result<Arc<Tensor>> {
    let t = unary_result(a, inplace)?;
    Ok(finish(t, op, &[a]))
}

unary_op!(ggml_sqr, ggml_sqr_inplace, GgmlOp::Sqr);
unary_op!(ggml_sqrt, ggml_sqrt_inplace, GgmlOp::Sqrt);
unary_op!(ggml_log, ggml_log_inplace, GgmlOp::Log);
unary_op!(ggml_sin, ggml_sin_inplace, GgmlOp::Sin);
unary_op!(ggml_cos, ggml_cos_inplace, GgmlOp::Cos);

/// `ggml_sum`.
pub fn ggml_sum(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    let t = ggml_new_tensor_1d(a.type_, 1)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::Sum, &[a]))
}

/// `ggml_sum_rows`.
pub fn ggml_sum_rows(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    let mut ne = [1i64; GGML_MAX_DIMS];
    ne[1..].copy_from_slice(&a.ne[1..]);
    let t = ggml_new_tensor(a.type_, GGML_MAX_DIMS, &ne)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::SumRows, &[a]))
}

/// `ggml_cumsum`.
pub fn ggml_cumsum(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    if a.type_ != GgmlType::F32 {
        return Err(Error::new("ggml_cumsum: a must be F32"));
    }
    let t = ggml_dup_tensor(a)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::Cumsum, &[a]))
}

/// `ggml_mean`.
pub fn ggml_mean(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    let ne = [1, a.ne[1], a.ne[2], a.ne[3]];
    let t = ggml_new_tensor(GgmlType::F32, GGML_MAX_DIMS, &ne)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::Mean, &[a]))
}

/// `ggml_argmax`.
pub fn ggml_argmax(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    if !ggml_is_matrix(a) || a.ne[0] > i32::MAX as i64 {
        return Err(Error::new("ggml_argmax: a must be a matrix with <= INT32_MAX columns"));
    }
    let t = ggml_new_tensor_1d(GgmlType::I32, a.ne[1])?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::Argmax, &[a]))
}

/// `ggml_count_equal`.
pub fn ggml_count_equal(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    if !ggml_are_same_shape(a, b) {
        return Err(Error::new("ggml_count_equal: shapes differ"));
    }
    let t = ggml_new_tensor_1d(GgmlType::I64, 1)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::CountEqual, &[a, b]))
}

/// `ggml_repeat`.
pub fn ggml_repeat(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    if !ggml_can_repeat(a, b) {
        return Err(Error::new("ggml_repeat: cannot repeat a to b"));
    }
    let t = ggml_new_tensor(a.type_, GGML_MAX_DIMS, &b.ne)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::Repeat, &[a]))
}

/// `ggml_repeat_4d`.
pub fn ggml_repeat_4d(
    a: &Arc<Tensor>,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
) -> Result<Arc<Tensor>> {
    let ne = [ne0, ne1, ne2, ne3];
    let can_repeat = super::helpers::ggml_is_empty(a)
        || ne.iter().zip(a.ne.iter()).all(|(n, m)| *n % *m == 0);
    if !can_repeat {
        return Err(Error::new("ggml_repeat_4d: cannot repeat a"));
    }
    let t = ggml_new_tensor_4d(a.type_, ne0, ne1, ne2, ne3)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::Repeat, &[a]))
}

/// `ggml_repeat_back`.
pub fn ggml_repeat_back(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    if !ggml_can_repeat(b, a) {
        return Err(Error::new("ggml_repeat_back: cannot repeat b to a"));
    }
    let t = ggml_new_tensor(a.type_, GGML_MAX_DIMS, &b.ne)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::RepeatBack, &[a]))
}

/// `ggml_concat`.
pub fn ggml_concat(a: &Arc<Tensor>, b: &Arc<Tensor>, dim: usize) -> Result<Arc<Tensor>> {
    if dim >= GGML_MAX_DIMS || a.type_ != b.type_ {
        return Err(Error::new("ggml_concat: invalid dim or type mismatch"));
    }
    let mut ne = [0i64; GGML_MAX_DIMS];
    for d in 0..GGML_MAX_DIMS {
        if d == dim {
            ne[d] = a.ne[d] + b.ne[d];
        } else {
            if a.ne[d] != b.ne[d] {
                return Err(Error::new("ggml_concat: shape mismatch"));
            }
            ne[d] = a.ne[d];
        }
    }
    let mut t = ggml_new_tensor(a.type_, GGML_MAX_DIMS, &ne)?;
    Arc::get_mut(&mut t).expect("fresh tensor is not shared").set_op_params_i32(0, dim as i32);
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::Concat, &[a, b]))
}

/// `ggml_scale_impl`.
pub fn ggml_scale_impl(a: &Arc<Tensor>, s: f32, b: f32, inplace: bool) -> Result<Arc<Tensor>> {
    if !ggml_is_contiguous_rows(a) {
        return Err(Error::new("ggml_scale: a is not contiguous by rows"));
    }
    let mut t = unary_result(a, inplace)?;
    t.set_op_params_f32_slice(&[s, b]);
    Ok(finish(t, GgmlOp::Scale, &[a]))
}

/// `ggml_scale`.
pub fn ggml_scale(a: &Arc<Tensor>, s: f32) -> Result<Arc<Tensor>> {
    ggml_scale_impl(a, s, 0.0, false)
}

/// `ggml_scale_inplace`.
pub fn ggml_scale_inplace(a: &Arc<Tensor>, s: f32) -> Result<Arc<Tensor>> {
    ggml_scale_impl(a, s, 0.0, true)
}

/// `ggml_scale_bias`.
pub fn ggml_scale_bias(a: &Arc<Tensor>, s: f32, b: f32) -> Result<Arc<Tensor>> {
    ggml_scale_impl(a, s, b, false)
}

/// `ggml_scale_bias_inplace`.
pub fn ggml_scale_bias_inplace(a: &Arc<Tensor>, s: f32, b: f32) -> Result<Arc<Tensor>> {
    ggml_scale_impl(a, s, b, true)
}

/// `ggml_set_impl`: place `b` into `a` with explicit strides and offset.
pub fn ggml_set_impl(
    a: &Arc<Tensor>,
    b: &Arc<Tensor>,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    offset: usize,
    inplace: bool,
) -> Result<Arc<Tensor>> {
    if ggml_nelements(a) < ggml_nelements(b) {
        return Err(Error::new("ggml_set: a has fewer elements than b"));
    }
    let mut t = unary_result(a, inplace)?;
    set_nb_params(&mut t, nb1, nb2, nb3, offset, inplace)?;
    Ok(finish(t, GgmlOp::Set, &[a, b]))
}

/// `ggml_set`.
pub fn ggml_set(
    a: &Arc<Tensor>,
    b: &Arc<Tensor>,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    offset: usize,
) -> Result<Arc<Tensor>> {
    ggml_set_impl(a, b, nb1, nb2, nb3, offset, false)
}

/// `ggml_set_inplace`.
pub fn ggml_set_inplace(
    a: &Arc<Tensor>,
    b: &Arc<Tensor>,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    offset: usize,
) -> Result<Arc<Tensor>> {
    ggml_set_impl(a, b, nb1, nb2, nb3, offset, true)
}

/// `ggml_set_1d`.
pub fn ggml_set_1d(a: &Arc<Tensor>, b: &Arc<Tensor>, offset: usize) -> Result<Arc<Tensor>> {
    ggml_set_impl(a, b, a.nb[1], a.nb[2], a.nb[3], offset, false)
}

/// `ggml_set_1d_inplace`.
pub fn ggml_set_1d_inplace(a: &Arc<Tensor>, b: &Arc<Tensor>, offset: usize) -> Result<Arc<Tensor>> {
    ggml_set_impl(a, b, a.nb[1], a.nb[2], a.nb[3], offset, true)
}

/// `ggml_set_2d`.
pub fn ggml_set_2d(a: &Arc<Tensor>, b: &Arc<Tensor>, nb1: usize, offset: usize) -> Result<Arc<Tensor>> {
    ggml_set_impl(a, b, nb1, a.nb[2], a.nb[3], offset, false)
}

/// `ggml_set_2d_inplace`.
pub fn ggml_set_2d_inplace(a: &Arc<Tensor>, b: &Arc<Tensor>, nb1: usize, offset: usize) -> Result<Arc<Tensor>> {
    ggml_set_impl(a, b, nb1, a.nb[2], a.nb[3], offset, true)
}

/// `ggml_unary_impl`.
pub fn ggml_unary_impl(a: &Arc<Tensor>, op: GgmlUnaryOp, inplace: bool) -> Result<Arc<Tensor>> {
    if !ggml_is_contiguous_rows(a) {
        return Err(Error::new("ggml_unary: a is not contiguous by rows"));
    }
    let mut t = unary_result(a, inplace)?;
    t.set_op_params_i32(0, op.as_i32());
    Ok(finish(t, GgmlOp::Unary, &[a]))
}

/// `ggml_unary`.
pub fn ggml_unary(a: &Arc<Tensor>, op: GgmlUnaryOp) -> Result<Arc<Tensor>> {
    ggml_unary_impl(a, op, false)
}

/// `ggml_unary_inplace`.
pub fn ggml_unary_inplace(a: &Arc<Tensor>, op: GgmlUnaryOp) -> Result<Arc<Tensor>> {
    ggml_unary_impl(a, op, true)
}

macro_rules! unary_wrapper {
    ($out:ident, $inplace:ident, $variant:ident) => {
        /// Out-of-place unary form.
        pub fn $out(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
            ggml_unary(a, GgmlUnaryOp::$variant)
        }
        /// In-place unary form.
        pub fn $inplace(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
            ggml_unary_inplace(a, GgmlUnaryOp::$variant)
        }
    };
}

unary_wrapper!(ggml_abs, ggml_abs_inplace, Abs);
unary_wrapper!(ggml_sgn, ggml_sgn_inplace, Sgn);
unary_wrapper!(ggml_neg, ggml_neg_inplace, Neg);
unary_wrapper!(ggml_step, ggml_step_inplace, Step);
unary_wrapper!(ggml_tanh, ggml_tanh_inplace, Tanh);
unary_wrapper!(ggml_elu, ggml_elu_inplace, Elu);
unary_wrapper!(ggml_relu, ggml_relu_inplace, Relu);
unary_wrapper!(ggml_sigmoid, ggml_sigmoid_inplace, Sigmoid);
unary_wrapper!(ggml_gelu, ggml_gelu_inplace, Gelu);
unary_wrapper!(ggml_gelu_quick, ggml_gelu_quick_inplace, GeluQuick);
unary_wrapper!(ggml_silu, ggml_silu_inplace, Silu);
unary_wrapper!(ggml_hardswish, ggml_hardswish_inplace, Hardswish);
unary_wrapper!(ggml_hardsigmoid, ggml_hardsigmoid_inplace, Hardsigmoid);
unary_wrapper!(ggml_exp, ggml_exp_inplace, Exp);
unary_wrapper!(ggml_expm1, ggml_expm1_inplace, Expm1);
unary_wrapper!(ggml_softplus, ggml_softplus_inplace, Softplus);
unary_wrapper!(ggml_gelu_erf, ggml_gelu_erf_inplace, GeluErf);
unary_wrapper!(ggml_xielu, ggml_xielu_inplace, Xielu);
unary_wrapper!(ggml_floor, ggml_floor_inplace, Floor);
unary_wrapper!(ggml_ceil, ggml_ceil_inplace, Ceil);
unary_wrapper!(ggml_round, ggml_round_inplace, Round);
unary_wrapper!(ggml_trunc, ggml_trunc_inplace, Trunc);

/// `ggml_leaky_relu`.
pub fn ggml_leaky_relu(a: &Arc<Tensor>, negative_slope: f32, inplace: bool) -> Result<Arc<Tensor>> {
    let mut t = unary_result(a, inplace)?;
    t.set_op_params_f32(0, negative_slope);
    Ok(finish(t, GgmlOp::LeakyRelu, &[a]))
}

/// `ggml_clamp_impl`.
pub fn ggml_clamp_impl(a: &Arc<Tensor>, min: f32, max: f32, inplace: bool) -> Result<Arc<Tensor>> {
    let mut t = unary_result(a, inplace)?;
    t.set_op_params_f32_slice(&[min, max]);
    Ok(finish(t, GgmlOp::Clamp, &[a]))
}

/// `ggml_clamp`.
pub fn ggml_clamp(a: &Arc<Tensor>, min: f32, max: f32) -> Result<Arc<Tensor>> {
    ggml_clamp_impl(a, min, max, false)
}

/// `ggml_clamp_inplace`.
pub fn ggml_clamp_inplace(a: &Arc<Tensor>, min: f32, max: f32) -> Result<Arc<Tensor>> {
    ggml_clamp_impl(a, min, max, true)
}

/// `ggml_tri`.
pub fn ggml_tri(a: &Arc<Tensor>, tri_type: i32) -> Result<Arc<Tensor>> {
    if !(a.type_ == GgmlType::F32 || a.type_ == GgmlType::F16 || a.type_ == GgmlType::BF16) {
        return Err(Error::new("ggml_tri: unsupported type"));
    }
    if !ggml_is_contiguous(a) || a.ne[0] != a.ne[1] {
        return Err(Error::new("ggml_tri: a must be a contiguous square matrix"));
    }
    let mut t = ggml_dup_tensor(a)?;
    Arc::get_mut(&mut t).expect("fresh tensor is not shared").set_op_params_i32(0, tri_type);
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::Tri, &[a]))
}

/// `ggml_fill_impl`.
pub fn ggml_fill_impl(a: &Arc<Tensor>, c: f32, inplace: bool) -> Result<Arc<Tensor>> {
    if !(a.type_ == GgmlType::F32 || a.type_ == GgmlType::F16) || !ggml_is_contiguous(a) {
        return Err(Error::new("ggml_fill: a must be contiguous F32 or F16"));
    }
    let mut t = unary_result(a, inplace)?;
    t.set_op_params_f32(0, c);
    Ok(finish(t, GgmlOp::Fill, &[a]))
}

/// `ggml_fill`.
pub fn ggml_fill(a: &Arc<Tensor>, c: f32) -> Result<Arc<Tensor>> {
    ggml_fill_impl(a, c, false)
}

/// `ggml_fill_inplace`.
pub fn ggml_fill_inplace(a: &Arc<Tensor>, c: f32) -> Result<Arc<Tensor>> {
    ggml_fill_impl(a, c, true)
}

/// `ggml_arange`.
pub fn ggml_arange(start: f32, stop: f32, step: f32) -> Result<Arc<Tensor>> {
    if stop <= start || step <= 0.0 {
        return Err(Error::new("ggml_arange: require stop > start and step > 0"));
    }
    let steps = ((stop - start) / step).ceil() as i64;
    let mut t = ggml_new_tensor_1d(GgmlType::F32, steps)?;
    {
        let out = Arc::get_mut(&mut t).expect("fresh tensor is not shared");
        out.set_op_params_f32_slice(&[start, stop, step]);
        out.op = GgmlOp::Arange;
    }
    Ok(Arc::new(Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone())))
}

/// `ggml_set_zero`: zero the tensor bytes in place.
pub fn ggml_set_zero(a: &Arc<Tensor>) -> Result<()> {
    a.set_zero()
}

//! Remaining graph-building ops: matrix products, normalizations and softmax.

use std::sync::Arc;

use crate::core::{Error, Result};

use super::create::{ggml_dup_tensor, ggml_new_tensor};
use super::helpers::{ggml_is_contiguous, ggml_is_transposed};
use super::types::{GgmlOp, GgmlType, GGML_MAX_DIMS};
use super::views::finish;
use super::Tensor;

/// `ggml_can_mul_mat` (static inline in `ggml.c`).
pub fn ggml_can_mul_mat(t0: &Tensor, t1: &Tensor) -> bool {
    t0.ne[0] == t1.ne[0]
        && t1.ne[2] % t0.ne[2] == 0
        && t1.ne[3] % t0.ne[3] == 0
}

/// `ggml_can_out_prod` (static inline in `ggml.c`).
pub fn ggml_can_out_prod(t0: &Tensor, t1: &Tensor) -> bool {
    t0.ne[1] == t1.ne[1]
        && t1.ne[2] % t0.ne[2] == 0
        && t1.ne[3] % t0.ne[3] == 0
}

/// `ggml_mul_mat`.
pub fn ggml_mul_mat(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    if !ggml_can_mul_mat(a, b) || ggml_is_transposed(a) {
        return Err(Error::new("ggml_mul_mat: incompatible shapes"));
    }
    let ne = [a.ne[1], b.ne[1], b.ne[2], b.ne[3]];
    let t = ggml_new_tensor(GgmlType::F32, GGML_MAX_DIMS, &ne)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::MulMat, &[a, b]))
}

/// `ggml_mul_mat_id`.
pub fn ggml_mul_mat_id(
    as_: &Arc<Tensor>,
    b: &Arc<Tensor>,
    ids: &Arc<Tensor>,
) -> Result<Arc<Tensor>> {
    if ggml_is_transposed(as_)
        || ids.type_ != GgmlType::I32
        || as_.ne[3] != 1
        || b.ne[3] != 1
        || ids.ne[2] != 1
        || ids.ne[3] != 1
        || ids.ne[1] != b.ne[2]
        || as_.ne[0] != b.ne[0]
        || ids.ne[0] % b.ne[1] != 0
    {
        return Err(Error::new("ggml_mul_mat_id: incompatible shapes"));
    }
    let ne = [as_.ne[1], ids.ne[0], b.ne[2], 1];
    let t = ggml_new_tensor(GgmlType::F32, GGML_MAX_DIMS, &ne)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::MulMatId, &[as_, b, ids]))
}

/// `ggml_out_prod`.
pub fn ggml_out_prod(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    if !ggml_can_out_prod(a, b) || ggml_is_transposed(a) {
        return Err(Error::new("ggml_out_prod: incompatible shapes"));
    }
    let ne = [a.ne[0], b.ne[0], b.ne[2], b.ne[3]];
    let t = ggml_new_tensor(GgmlType::F32, GGML_MAX_DIMS, &ne)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::OutProd, &[a, b]))
}

fn norm_impl(a: &Arc<Tensor>, eps: f32, inplace: bool, op: GgmlOp) -> Result<Arc<Tensor>> {
    let t = if inplace { super::create::ggml_view_tensor(a)? } else { ggml_dup_tensor(a)? };
    let mut t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    t.set_op_params_f32(0, eps);
    Ok(finish(t, op, &[a]))
}

/// `ggml_norm`.
pub fn ggml_norm(a: &Arc<Tensor>, eps: f32) -> Result<Arc<Tensor>> {
    norm_impl(a, eps, false, GgmlOp::Norm)
}

/// `ggml_norm_inplace`.
pub fn ggml_norm_inplace(a: &Arc<Tensor>, eps: f32) -> Result<Arc<Tensor>> {
    norm_impl(a, eps, true, GgmlOp::Norm)
}

/// `ggml_rms_norm`.
pub fn ggml_rms_norm(a: &Arc<Tensor>, eps: f32) -> Result<Arc<Tensor>> {
    norm_impl(a, eps, false, GgmlOp::RmsNorm)
}

/// `ggml_rms_norm_inplace`.
pub fn ggml_rms_norm_inplace(a: &Arc<Tensor>, eps: f32) -> Result<Arc<Tensor>> {
    norm_impl(a, eps, true, GgmlOp::RmsNorm)
}

/// `ggml_rms_norm_back`.
pub fn ggml_rms_norm_back(a: &Arc<Tensor>, b: &Arc<Tensor>, eps: f32) -> Result<Arc<Tensor>> {
    let mut t = Arc::try_unwrap(ggml_dup_tensor(a)?).unwrap_or_else(|t| (*t).clone());
    t.set_op_params_f32(0, eps);
    Ok(finish(t, GgmlOp::RmsNormBack, &[a, b]))
}

/// `ggml_group_norm_impl`.
pub fn ggml_group_norm_impl(
    a: &Arc<Tensor>,
    n_groups: i32,
    eps: f32,
    inplace: bool,
) -> Result<Arc<Tensor>> {
    let t = if inplace { super::create::ggml_view_tensor(a)? } else { ggml_dup_tensor(a)? };
    let mut t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    t.set_op_params_i32(0, n_groups);
    t.set_op_params_f32(1, eps);
    Ok(finish(t, GgmlOp::GroupNorm, &[a]))
}

/// `ggml_group_norm`.
pub fn ggml_group_norm(a: &Arc<Tensor>, n_groups: i32, eps: f32) -> Result<Arc<Tensor>> {
    ggml_group_norm_impl(a, n_groups, eps, false)
}

/// `ggml_group_norm_inplace`.
pub fn ggml_group_norm_inplace(a: &Arc<Tensor>, n_groups: i32, eps: f32) -> Result<Arc<Tensor>> {
    ggml_group_norm_impl(a, n_groups, eps, true)
}

fn l2_norm_impl(a: &Arc<Tensor>, eps: f32, inplace: bool) -> Result<Arc<Tensor>> {
    norm_impl(a, eps, inplace, GgmlOp::L2Norm)
}

/// `ggml_l2_norm`.
pub fn ggml_l2_norm(a: &Arc<Tensor>, eps: f32) -> Result<Arc<Tensor>> {
    l2_norm_impl(a, eps, false)
}

/// `ggml_l2_norm_inplace`.
pub fn ggml_l2_norm_inplace(a: &Arc<Tensor>, eps: f32) -> Result<Arc<Tensor>> {
    l2_norm_impl(a, eps, true)
}

/// `ggml_soft_max_impl` with optional mask. `mask` is `None` for plain softmax.
pub fn ggml_soft_max_impl(
    a: &Arc<Tensor>,
    mask: Option<&Arc<Tensor>>,
    scale: f32,
    max_bias: f32,
    inplace: bool,
) -> Result<Arc<Tensor>> {
    if !ggml_is_contiguous(a) {
        return Err(Error::new("ggml_soft_max: a is not contiguous"));
    }
    if let Some(m) = mask {
        if !(m.type_ == GgmlType::F16 || m.type_ == GgmlType::F32)
            || !ggml_is_contiguous(m)
            || m.ne[0] != a.ne[0]
            || m.ne[1] < a.ne[1]
            || a.ne[2] % m.ne[2] != 0
            || a.ne[3] % m.ne[3] != 0
        {
            return Err(Error::new("ggml_soft_max: incompatible mask"));
        }
    }
    if max_bias > 0.0 && mask.is_none() {
        return Err(Error::new("ggml_soft_max: max_bias requires a mask"));
    }
    let t = if inplace { super::create::ggml_view_tensor(a)? } else { ggml_dup_tensor(a)? };
    let mut t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    t.set_op_params_f32_slice(&[scale, max_bias]);
    let mut srcs: Vec<&Arc<Tensor>> = vec![a];
    if let Some(m) = mask {
        srcs.push(m);
    }
    Ok(finish(t, GgmlOp::SoftMax, &srcs))
}

/// `ggml_soft_max`.
pub fn ggml_soft_max(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    ggml_soft_max_impl(a, None, 1.0, 0.0, false)
}

/// `ggml_silu_back`.
pub fn ggml_silu_back(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    let t = ggml_dup_tensor(a)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::SiluBack, &[a, b]))
}

/// `ggml_sigmoid_back`.
pub fn ggml_sigmoid_back(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    let t = ggml_dup_tensor(a)?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::SigmoidBack, &[a, b]))
}

/// `ggml_argsort`. `order` is the raw `ggml_sort_order` value.
pub fn ggml_argsort(a: &Arc<Tensor>, order: i32) -> Result<Arc<Tensor>> {
    if a.ne[0] > i32::MAX as i64 {
        return Err(Error::new("ggml_argsort: too many elements"));
    }
    let mut t = ggml_new_tensor(GgmlType::I32, GGML_MAX_DIMS, &a.ne)?;
    Arc::get_mut(&mut t).expect("fresh tensor is not shared").set_op_params_i32(0, order);
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::Argsort, &[a]))
}

/// `ggml_top_k`.
pub fn ggml_top_k(a: &Arc<Tensor>, k: i64) -> Result<Arc<Tensor>> {
    if k < 0 || a.ne[0] < k {
        return Err(Error::new("ggml_top_k: invalid k"));
    }
    let t = ggml_new_tensor(GgmlType::I32, GGML_MAX_DIMS, &[k, a.ne[1], a.ne[2], a.ne[3]])?;
    let t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    Ok(finish(t, GgmlOp::TopK, &[a]))
}

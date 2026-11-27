//! Reshape, strided views, permutation, transposition and copy-style ops.

use std::sync::Arc;

use crate::core::{Error, Result};

use super::create::{ggml_new_tensor, ggml_new_tensor_4d, ggml_view_tensor};
use super::helpers::{ggml_is_contiguous, ggml_nelements};
use super::structure::new_tensor_impl;
use super::types::{
    ggml_type_size, GgmlOp, GgmlType, GGML_MAX_DIMS,
};
use super::Tensor;

/// Attach an op and its sources to a freshly built tensor.
pub(crate) fn finish(mut t: Tensor, op: GgmlOp, srcs: &[&Arc<Tensor>]) -> Arc<Tensor> {
    t.op = op;
    for (i, s) in srcs.iter().enumerate() {
        t.src[i] = Some((*s).clone());
    }
    Arc::new(t)
}

/// `ggml_reshape`.
pub fn ggml_reshape(a: &Arc<Tensor>, b: &Tensor) -> Result<Arc<Tensor>> {
    if !ggml_is_contiguous(a) {
        return Err(Error::new("ggml_reshape: a is not contiguous"));
    }
    if ggml_nelements(a) != ggml_nelements(b) {
        return Err(Error::new("ggml_reshape: element count mismatch"));
    }
    let mut result = new_tensor_impl(a.type_, GGML_MAX_DIMS, &b.ne, None, Some(a.clone()), 0, false)?;
    result.format_name(&format!("{} (reshaped)", a.name));
    Ok(finish(result, GgmlOp::Reshape, &[a]))
}

/// `ggml_reshape_1d`.
pub fn ggml_reshape_1d(a: &Arc<Tensor>, ne0: i64) -> Result<Arc<Tensor>> {
    reshape_n(a, &[ne0])
}

/// `ggml_reshape_2d`.
pub fn ggml_reshape_2d(a: &Arc<Tensor>, ne0: i64, ne1: i64) -> Result<Arc<Tensor>> {
    reshape_n(a, &[ne0, ne1])
}

/// `ggml_reshape_3d`.
pub fn ggml_reshape_3d(a: &Arc<Tensor>, ne0: i64, ne1: i64, ne2: i64) -> Result<Arc<Tensor>> {
    reshape_n(a, &[ne0, ne1, ne2])
}

/// `ggml_reshape_4d`.
pub fn ggml_reshape_4d(
    a: &Arc<Tensor>,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
) -> Result<Arc<Tensor>> {
    reshape_n(a, &[ne0, ne1, ne2, ne3])
}

fn reshape_n(a: &Arc<Tensor>, ne: &[i64]) -> Result<Arc<Tensor>> {
    if !ggml_is_contiguous(a) {
        return Err(Error::new("ggml_reshape: a is not contiguous"));
    }
    let wanted: i64 = ne.iter().product();
    if ggml_nelements(a) != wanted {
        return Err(Error::new("ggml_reshape: element count mismatch"));
    }
    let mut result = new_tensor_impl(a.type_, ne.len(), ne, None, Some(a.clone()), 0, false)?;
    result.format_name(&format!("{} (reshaped)", a.name));
    Ok(finish(result, GgmlOp::Reshape, &[a]))
}

/// `ggml_view_impl`: the common path for all strided views.
pub fn ggml_view_impl(
    a: &Arc<Tensor>,
    n_dims: usize,
    ne: &[i64],
    nb: Option<&[usize]>,
    offset: usize,
) -> Result<Arc<Tensor>> {
    if nb.is_some() && a.nb[0] != ggml_type_size(a.type_) {
        return Err(Error::new("ggml_view: source dim 0 is not contiguous"));
    }
    let mut result = new_tensor_impl(a.type_, n_dims, ne, nb, Some(a.clone()), offset, false)?;
    result.format_name(&format!("{} (view)", a.name));
    result.set_op_params_usize(0, offset);
    Ok(finish(result, GgmlOp::View, &[a]))
}

/// `ggml_view_1d`.
pub fn ggml_view_1d(a: &Arc<Tensor>, ne0: i64, offset: usize) -> Result<Arc<Tensor>> {
    ggml_view_impl(a, 1, &[ne0], None, offset)
}

/// `ggml_view_2d`.
pub fn ggml_view_2d(
    a: &Arc<Tensor>,
    ne0: i64,
    ne1: i64,
    nb1: usize,
    offset: usize,
) -> Result<Arc<Tensor>> {
    let ne = [ne0, ne1];
    let nb = [
        ggml_type_size(a.type_),
        nb1,
        nb1.wrapping_mul(ne1 as usize),
        nb1.wrapping_mul(ne1 as usize),
    ];
    ggml_view_impl(a, 2, &ne, Some(&nb), offset)
}

/// `ggml_view_3d`.
pub fn ggml_view_3d(
    a: &Arc<Tensor>,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    nb1: usize,
    nb2: usize,
    offset: usize,
) -> Result<Arc<Tensor>> {
    let ne = [ne0, ne1, ne2];
    let nb = [
        ggml_type_size(a.type_),
        nb1,
        nb2,
        nb2.wrapping_mul(ne2 as usize),
    ];
    ggml_view_impl(a, 3, &ne, Some(&nb), offset)
}

/// `ggml_view_4d`.
pub fn ggml_view_4d(
    a: &Arc<Tensor>,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    offset: usize,
) -> Result<Arc<Tensor>> {
    let ne = [ne0, ne1, ne2, ne3];
    let nb = [ggml_type_size(a.type_), nb1, nb2, nb3];
    ggml_view_impl(a, 4, &ne, Some(&nb), offset)
}

/// `ggml_permute`.
pub fn ggml_permute(
    a: &Arc<Tensor>,
    axis0: usize,
    axis1: usize,
    axis2: usize,
    axis3: usize,
) -> Result<Arc<Tensor>> {
    let axes = [axis0, axis1, axis2, axis3];
    let mut seen = [false; GGML_MAX_DIMS];
    for &ax in &axes {
        if ax >= GGML_MAX_DIMS || seen[ax] {
            return Err(Error::new("ggml_permute: axes must be a permutation of 0..4"));
        }
        seen[ax] = true;
    }

    let mut t = ggml_view_tensor(a)?;
    Arc::get_mut(&mut t).expect("fresh view is not shared").format_name(&format!("{} (permuted)", a.name));

    let mut ne = [0i64; GGML_MAX_DIMS];
    let mut nb = [0usize; GGML_MAX_DIMS];
    for (src_axis, &dst) in axes.iter().enumerate() {
        ne[dst] = a.ne[src_axis];
        nb[dst] = a.nb[src_axis];
    }
    {
        let out = Arc::get_mut(&mut t).expect("fresh view is not shared");
        out.ne = ne;
        out.nb = nb;
        out.set_op_params_i32_slice(&[axis0 as i32, axis1 as i32, axis2 as i32, axis3 as i32]);
    }
    Ok(finish(Arc::try_unwrap(t).unwrap_or_else(|a| (*a).clone()), GgmlOp::Permute, &[a]))
}

/// `ggml_transpose`: swap dimensions 0 and 1.
pub fn ggml_transpose(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    let mut t = ggml_view_tensor(a)?;
    {
        let out = Arc::get_mut(&mut t).expect("fresh view is not shared");
        out.format_name(&format!("{} (transposed)", a.name));
        out.ne.swap(0, 1);
        out.nb.swap(0, 1);
    }
    Ok(finish(Arc::try_unwrap(t).unwrap_or_else(|a| (*a).clone()), GgmlOp::Transpose, &[a]))
}

/// `ggml_cont`.
pub fn ggml_cont(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    let mut t = super::create::ggml_dup_tensor(a)?;
    Arc::get_mut(&mut t).expect("fresh tensor is not shared").format_name(&format!("{} (cont)", a.name));
    Ok(finish(Arc::try_unwrap(t).unwrap_or_else(|a| (*a).clone()), GgmlOp::Cont, &[a]))
}

/// `ggml_cont_4d`.
pub fn ggml_cont_4d(
    a: &Arc<Tensor>,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
) -> Result<Arc<Tensor>> {
    if ggml_nelements(a) != ne0 * ne1 * ne2 * ne3 {
        return Err(Error::new("ggml_cont_4d: element count mismatch"));
    }
    let mut t = ggml_new_tensor_4d(a.type_, ne0, ne1, ne2, ne3)?;
    Arc::get_mut(&mut t).expect("fresh tensor is not shared").format_name(&format!("{} (cont)", a.name));
    Ok(finish(Arc::try_unwrap(t).unwrap_or_else(|a| (*a).clone()), GgmlOp::Cont, &[a]))
}

/// `ggml_cont_1d`.
pub fn ggml_cont_1d(a: &Arc<Tensor>, ne0: i64) -> Result<Arc<Tensor>> {
    ggml_cont_4d(a, ne0, 1, 1, 1)
}

/// `ggml_cont_2d`.
pub fn ggml_cont_2d(a: &Arc<Tensor>, ne0: i64, ne1: i64) -> Result<Arc<Tensor>> {
    ggml_cont_4d(a, ne0, ne1, 1, 1)
}

/// `ggml_cont_3d`.
pub fn ggml_cont_3d(a: &Arc<Tensor>, ne0: i64, ne1: i64, ne2: i64) -> Result<Arc<Tensor>> {
    ggml_cont_4d(a, ne0, ne1, ne2, 1)
}

/// `ggml_cpy`: a view of `b` that copies `a` into it.
pub fn ggml_cpy(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    if ggml_nelements(a) != ggml_nelements(b) {
        return Err(Error::new("ggml_cpy: element count mismatch"));
    }
    let mut t = ggml_view_tensor(b)?;
    {
        let out = Arc::get_mut(&mut t).expect("fresh view is not shared");
        let name = if !b.name.is_empty() {
            format!("{} (copy of {})", b.name, a.name)
        } else {
            format!("{} (copy)", a.name)
        };
        out.format_name(&name);
    }
    Ok(finish(Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone()), GgmlOp::Cpy, &[a, b]))
}

/// `ggml_cast`.
///
/// The C version also stores `src[1] = result` for backend consistency. That
/// self-reference cannot be built with `Arc` and is not needed without a
/// backend, so `src[1]` stays empty here.
pub fn ggml_cast(a: &Arc<Tensor>, type_: GgmlType) -> Result<Arc<Tensor>> {
    let mut t = ggml_new_tensor(type_, GGML_MAX_DIMS, &a.ne)?;
    Arc::get_mut(&mut t).expect("fresh tensor is not shared").format_name(&format!("{} (copy)", a.name));
    Ok(finish(Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone()), GgmlOp::Cpy, &[a]))
}

/// `ggml_diag`.
pub fn ggml_diag(a: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    if a.ne[1] != 1 {
        return Err(Error::new("ggml_diag: a must be a vector"));
    }
    let ne = [a.ne[0], a.ne[0], a.ne[2], a.ne[3]];
    let t = ggml_new_tensor(a.type_, GGML_MAX_DIMS, &ne)?;
    Ok(finish(Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone()), GgmlOp::Diag, &[a]))
}

/// `ggml_get_rows`.
pub fn ggml_get_rows(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    if a.ne[2] != b.ne[1] || a.ne[3] != b.ne[2] || b.ne[3] != 1 {
        return Err(Error::new("ggml_get_rows: incompatible shapes"));
    }
    if b.type_ != GgmlType::I32 {
        return Err(Error::new("ggml_get_rows: indices must be I32"));
    }
    let type_ = if a.type_ == GgmlType::I32 { GgmlType::I32 } else { GgmlType::F32 };
    let t = ggml_new_tensor_4d(type_, a.ne[0], b.ne[0], b.ne[1], b.ne[2])?;
    Ok(finish(Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone()), GgmlOp::GetRows, &[a, b]))
}

/// `ggml_get_rows_back`.
pub fn ggml_get_rows_back(
    a: &Arc<Tensor>,
    b: &Arc<Tensor>,
    c: &Arc<Tensor>,
) -> Result<Arc<Tensor>> {
    if b.type_ != GgmlType::I32 {
        return Err(Error::new("ggml_get_rows_back: indices must be I32"));
    }
    let t = super::create::ggml_new_tensor(GgmlType::F32, 2, &[c.ne[0], c.ne[1]])?;
    // The C version stores only src[0] and src[1]; `c` supplies the shape.
    Ok(finish(Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone()), GgmlOp::GetRowsBack, &[a, b]))
}

/// `ggml_set_rows`.
pub fn ggml_set_rows(a: &Arc<Tensor>, b: &Arc<Tensor>, c: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    if a.ne[0] != b.ne[0] || a.ne[2] != b.ne[2] || a.ne[3] != b.ne[3] || b.ne[1] != c.ne[0] {
        return Err(Error::new("ggml_set_rows: incompatible shapes"));
    }
    if b.ne[2] % c.ne[1] != 0 || b.ne[3] % c.ne[2] != 0 || c.ne[3] != 1 {
        return Err(Error::new("ggml_set_rows: incompatible shapes"));
    }
    let t = ggml_view_tensor(a)?;
    let mut t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
    {
        // Legacy source order: b, c, a (see upstream PR 16063).
        t.src[0] = Some(b.clone());
        t.src[1] = Some(c.clone());
        t.src[2] = Some(a.clone());
    }
    Ok(finish(t, GgmlOp::SetRows, &[]))
}

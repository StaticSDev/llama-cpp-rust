//! Row/byte helpers and tensor predicates from `ggml.c`.

use std::cell::Ref;

use crate::core::{Error, Result};

use super::structure::Tensor;
use super::types::{
    ggml_blck_size, ggml_glu_op_name, ggml_op_name, ggml_type_size, ggml_unary_op_name, GgmlGluOp,
    GgmlUnaryOp, GGML_MAX_DIMS, GGML_MEM_ALIGN,
};

/// `ggml_nelements`.
pub fn ggml_nelements(t: &Tensor) -> i64 {
    t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]
}

/// `ggml_nrows`.
pub fn ggml_nrows(t: &Tensor) -> i64 {
    t.ne[1] * t.ne[2] * t.ne[3]
}

/// `ggml_nbytes`. Returns 0 when any dimension is non-positive.
pub fn ggml_nbytes(t: &Tensor) -> usize {
    if t.ne.iter().any(|&n| n <= 0) {
        return 0;
    }
    let blck_size = ggml_blck_size(t.type_);
    if blck_size == 1 {
        let mut nbytes = ggml_type_size(t.type_);
        for i in 0..GGML_MAX_DIMS {
            nbytes = nbytes.wrapping_add((t.ne[i] as usize - 1).wrapping_mul(t.nb[i]));
        }
        nbytes
    } else {
        let bs = blck_size as usize;
        let mut nbytes = (t.ne[0] as usize).wrapping_mul(t.nb[0]) / bs;
        for i in 1..GGML_MAX_DIMS {
            nbytes = nbytes.wrapping_add((t.ne[i] as usize - 1).wrapping_mul(t.nb[i]));
        }
        nbytes
    }
}

/// `ggml_nbytes_pad`.
pub fn ggml_nbytes_pad(t: &Tensor) -> usize {
    let n = ggml_nbytes(t);
    n.wrapping_add(GGML_MEM_ALIGN - 1) & !(GGML_MEM_ALIGN - 1)
}

/// `ggml_element_size`.
pub fn ggml_element_size(t: &Tensor) -> usize {
    ggml_type_size(t.type_)
}

/// `ggml_is_scalar`.
pub fn ggml_is_scalar(t: &Tensor) -> bool {
    t.ne.iter().all(|&n| n == 1)
}

/// `ggml_is_vector`.
pub fn ggml_is_vector(t: &Tensor) -> bool {
    t.ne[1] == 1 && t.ne[2] == 1 && t.ne[3] == 1
}

/// `ggml_is_matrix`.
pub fn ggml_is_matrix(t: &Tensor) -> bool {
    t.ne[2] == 1 && t.ne[3] == 1
}

/// `ggml_is_3d`.
pub fn ggml_is_3d(t: &Tensor) -> bool {
    t.ne[3] == 1
}

/// `ggml_n_dims`.
pub fn ggml_n_dims(t: &Tensor) -> usize {
    for i in (1..GGML_MAX_DIMS).rev() {
        if t.ne[i] > 1 {
            return i + 1;
        }
    }
    1
}

/// `ggml_is_transposed`.
pub fn ggml_is_transposed(t: &Tensor) -> bool {
    t.nb[0] > t.nb[1]
}

/// `ggml_is_permuted`.
pub fn ggml_is_permuted(t: &Tensor) -> bool {
    t.nb[0] > t.nb[1] || t.nb[1] > t.nb[2] || t.nb[2] > t.nb[3]
}

/// `ggml_is_empty`.
pub fn ggml_is_empty(t: &Tensor) -> bool {
    t.ne.iter().any(|&n| n == 0)
}

/// `ggml_is_view`.
pub fn ggml_is_view(t: &Tensor) -> bool {
    t.view_src.is_some()
}

/// `ggml_is_contiguous_m_n`, the shared implementation of the contiguous helpers.
pub fn ggml_is_contiguous_m_n(t: &Tensor, m: usize, n: usize) -> bool {
    let mut next_nb = ggml_type_size(t.type_);
    if t.ne[0] != ggml_blck_size(t.type_) && t.nb[0] != next_nb {
        return false;
    }
    next_nb = next_nb.wrapping_mul(t.ne[0] as usize / ggml_blck_size(t.type_) as usize);
    for i in 1..n {
        if i > m {
            if t.ne[i] != 1 && t.nb[i] != next_nb {
                return false;
            }
            next_nb = next_nb.wrapping_mul(t.ne[i] as usize);
        } else {
            next_nb = (t.ne[i] as usize).wrapping_mul(t.nb[i]);
        }
    }
    true
}

/// `ggml_is_contiguous` / `ggml_is_contiguous_0`.
pub fn ggml_is_contiguous(t: &Tensor) -> bool {
    ggml_is_contiguous_m_n(t, 0, GGML_MAX_DIMS)
}

/// `ggml_is_contiguous_0`.
pub fn ggml_is_contiguous_0(t: &Tensor) -> bool {
    ggml_is_contiguous_m_n(t, 0, GGML_MAX_DIMS)
}

/// `ggml_is_contiguous_1`.
pub fn ggml_is_contiguous_1(t: &Tensor) -> bool {
    ggml_is_contiguous_m_n(t, 1, GGML_MAX_DIMS)
}

/// `ggml_is_contiguous_2`.
pub fn ggml_is_contiguous_2(t: &Tensor) -> bool {
    ggml_is_contiguous_m_n(t, 2, GGML_MAX_DIMS)
}

/// `ggml_is_contiguous_to_1`.
pub fn ggml_is_contiguous_to_1(t: &Tensor) -> bool {
    ggml_is_contiguous_m_n(t, 0, 1)
}

/// `ggml_is_contiguous_to_2`.
pub fn ggml_is_contiguous_to_2(t: &Tensor) -> bool {
    ggml_is_contiguous_m_n(t, 0, 2)
}

/// `ggml_is_contiguous_to_3`.
pub fn ggml_is_contiguous_to_3(t: &Tensor) -> bool {
    ggml_is_contiguous_m_n(t, 0, 3)
}

/// `ggml_is_contiguously_allocated`.
pub fn ggml_is_contiguously_allocated(t: &Tensor) -> bool {
    let expected = (ggml_nelements(t) as usize)
        .wrapping_mul(ggml_type_size(t.type_))
        / ggml_blck_size(t.type_) as usize;
    ggml_nbytes(t) == expected
}

/// `ggml_is_contiguous_channels`.
pub fn ggml_is_contiguous_channels(t: &Tensor) -> bool {
    t.nb[0] > t.nb[2]
        && t.nb[1] > t.nb[0]
        && t.nb[2] == ggml_type_size(t.type_)
}

/// `ggml_is_contiguous_rows`.
pub fn ggml_is_contiguous_rows(t: &Tensor) -> bool {
    t.ne[0] == ggml_blck_size(t.type_) || t.nb[0] == ggml_type_size(t.type_)
}

/// `ggml_is_padded_1d` (static inline in `ggml.c`).
pub fn ggml_is_padded_1d(t: &Tensor) -> bool {
    t.nb[0] == ggml_type_size(t.type_)
        && t.nb[2] == t.nb[1].wrapping_mul(t.ne[1] as usize)
        && t.nb[3] == t.nb[2].wrapping_mul(t.ne[2] as usize)
}

/// `ggml_are_same_shape`.
pub fn ggml_are_same_shape(t0: &Tensor, t1: &Tensor) -> bool {
    t0.ne == t1.ne
}

/// `ggml_are_same_stride`.
pub fn ggml_are_same_stride(t0: &Tensor, t1: &Tensor) -> bool {
    t0.nb == t1.nb
}

/// `ggml_can_repeat`.
pub fn ggml_can_repeat(t0: &Tensor, t1: &Tensor) -> bool {
    if ggml_is_empty(t0) {
        return ggml_is_empty(t1);
    }
    t1.ne.iter().zip(t0.ne.iter()).all(|(a, b)| *a % *b == 0)
}

/// `ggml_can_repeat_rows` (static inline in `ggml.c`).
pub fn ggml_can_repeat_rows(t0: &Tensor, t1: &Tensor) -> bool {
    t0.ne[0] == t1.ne[0] && ggml_can_repeat(t0, t1)
}

/// `ggml_unravel_index`. Returns `[i0, i1, i2, i3]`.
pub fn ggml_unravel_index(t: &Tensor, i: i64) -> [i64; GGML_MAX_DIMS] {
    let ne0 = t.ne[0];
    let ne1 = t.ne[1];
    let ne2 = t.ne[2];
    let i3 = i / (ne2 * ne1 * ne0);
    let i2 = (i - i3 * ne2 * ne1 * ne0) / (ne1 * ne0);
    let i1 = (i - i3 * ne2 * ne1 * ne0 - i2 * ne1 * ne0) / ne0;
    let i0 = i - i3 * ne2 * ne1 * ne0 - i2 * ne1 * ne0 - i1 * ne0;
    [i0, i1, i2, i3]
}

/// `ggml_get_data`.
pub fn ggml_get_data(t: &Tensor) -> Option<Ref<'_, [u8]>> {
    t.data_slice()
}

/// `ggml_get_unary_op`.
pub fn ggml_get_unary_op(t: &Tensor) -> Result<GgmlUnaryOp> {
    if t.op != super::types::GgmlOp::Unary {
        return Err(Error::new("ggml_get_unary_op: tensor op is not UNARY"));
    }
    GgmlUnaryOp::from_i32(t.get_op_params_i32(0))
}

/// `ggml_get_glu_op`.
pub fn ggml_get_glu_op(t: &Tensor) -> Result<GgmlGluOp> {
    if t.op != super::types::GgmlOp::Glu {
        return Err(Error::new("ggml_get_glu_op: tensor op is not GLU"));
    }
    GgmlGluOp::from_i32(t.get_op_params_i32(0))
}

/// `ggml_op_desc`.
pub fn ggml_op_desc(t: &Tensor) -> Result<&'static str> {
    match t.op {
        super::types::GgmlOp::Unary => Ok(ggml_unary_op_name(ggml_get_unary_op(t)?)),
        super::types::GgmlOp::Glu => Ok(ggml_glu_op_name(ggml_get_glu_op(t)?)),
        op => Ok(ggml_op_name(op)),
    }
}

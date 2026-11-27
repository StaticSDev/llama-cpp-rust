//! Tensor creation and duplication (`ggml_new_tensor*`, `ggml_dup_tensor`).

use std::sync::Arc;

use crate::core::Result;

use super::structure::new_tensor_impl;
use super::types::{GgmlType, GGML_MAX_DIMS};
use super::Tensor;

/// Build a shared graph node from an owned tensor.
pub fn share(t: Tensor) -> Arc<Tensor> {
    Arc::new(t)
}

/// `ggml_new_tensor`.
pub fn ggml_new_tensor(type_: GgmlType, n_dims: usize, ne: &[i64]) -> Result<Arc<Tensor>> {
    Ok(share(new_tensor_impl(type_, n_dims, ne, None, None, 0, false)?))
}

/// `ggml_new_tensor_1d`.
pub fn ggml_new_tensor_1d(type_: GgmlType, ne0: i64) -> Result<Arc<Tensor>> {
    ggml_new_tensor(type_, 1, &[ne0])
}

/// `ggml_new_tensor_2d`.
pub fn ggml_new_tensor_2d(type_: GgmlType, ne0: i64, ne1: i64) -> Result<Arc<Tensor>> {
    ggml_new_tensor(type_, 2, &[ne0, ne1])
}

/// `ggml_new_tensor_3d`.
pub fn ggml_new_tensor_3d(type_: GgmlType, ne0: i64, ne1: i64, ne2: i64) -> Result<Arc<Tensor>> {
    ggml_new_tensor(type_, 3, &[ne0, ne1, ne2])
}

/// `ggml_new_tensor_4d`.
pub fn ggml_new_tensor_4d(
    type_: GgmlType,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
) -> Result<Arc<Tensor>> {
    ggml_new_tensor(type_, 4, &[ne0, ne1, ne2, ne3])
}

/// `ggml_dup_tensor`: same type and shape, freshly allocated.
pub fn ggml_dup_tensor(src: &Tensor) -> Result<Arc<Tensor>> {
    ggml_new_tensor(src.type_, GGML_MAX_DIMS, &src.ne)
}

/// `ggml_view_tensor`: a full view over `src` with `src`'s strides.
pub fn ggml_view_tensor(src: &Arc<Tensor>) -> Result<Arc<Tensor>> {
    let mut result = new_tensor_impl(
        src.type_,
        GGML_MAX_DIMS,
        &src.ne,
        Some(&src.nb),
        Some(src.clone()),
        0,
        false,
    )?;
    result.format_name(&format!("{} (view)", src.name));
    Ok(share(result))
}

/// `ggml_new_buffer`: allocate `nbytes` as a plain storage tensor.
pub fn ggml_new_buffer(nbytes: usize) -> Result<Arc<Tensor>> {
    let mut t = ggml_new_tensor_1d(GgmlType::I8, nbytes as i64)?;
    Arc::get_mut(&mut t)
        .expect("new buffer is not shared")
        .format_name("(buffer)");
    Ok(t)
}

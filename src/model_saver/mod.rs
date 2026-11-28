//! Rust translation of `llama-model-saver.cpp`.
//!
//! C++ `llama_model_saver` collects hparams, vocab and tensors into a GGUF
//! context, then writes it with `gguf_write_to_file`. Callers supply the
//! tensors and metadata here; serialization stays in `crate::gguf`.

use std::convert::TryFrom;
use std::fs::File;

use crate::core::{Error, Result};
use crate::gguf::{
    ggml_blck_size, ggml_nbytes, ggml_type_size, gguf_add_tensor, gguf_find_tensor,
    gguf_init_empty, gguf_set_arr_data, gguf_set_arr_str, gguf_set_val_bool, gguf_set_val_f32,
    gguf_set_val_i32, gguf_set_val_i64, gguf_set_val_str, gguf_set_val_u32, gguf_set_val_u64,
    gguf_write_out, GgmlTensor, GgmlType, GgufContext, GgufValue, GgufWriterFile,
    GGML_MAX_DIMS, GGML_MAX_NAME, GGUF_KEY_GENERAL_ALIGNMENT,
};

#[cfg(test)]
mod tests;

/// One tensor to write. `shape` maps to `ggml_tensor::ne`, `data` holds exactly
/// `ggml_nbytes(tensor)` bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct SavedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub ty: GgmlType,
    pub data: Vec<u8>,
}

/// Metadata value accepted by the GGUF writer.
#[derive(Debug, Clone, PartialEq)]
pub enum SavedValue {
    String(String),
    Bool(bool),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    StringArray(Vec<String>),
    I32Array(Vec<i32>),
    F32Array(Vec<f32>),
}

/// Write tensors and metadata to a GGUF file.
pub fn save_model(
    path: &str,
    tensors: &[SavedTensor],
    metadata: &[(String, SavedValue)],
) -> Result<()> {
    let mut ctx = gguf_init_empty();
    for (key, value) in metadata {
        set_metadata(&mut ctx, key, value)?;
        if key == GGUF_KEY_GENERAL_ALIGNMENT {
            if let SavedValue::U32(v) = value {
                ctx.alignment = *v as usize;
            }
        }
    }

    for tensor in tensors {
        if gguf_find_tensor(&ctx, &tensor.name) >= 0 {
            return Err(Error::new(format!("duplicate tensor name: {}", tensor.name)));
        }
        gguf_add_tensor(&mut ctx, &to_ggml_tensor(tensor)?);
    }

    let file = File::create(path)
        .map_err(|e| Error::new(format!("failed to create '{}' ({})", path, e)))?;
    gguf_write_out(&ctx, &mut GgufWriterFile::new(file), false)
}

fn set_metadata(ctx: &mut GgufContext, key: &str, value: &SavedValue) -> Result<()> {
    if key.is_empty() {
        return Err(Error::new("metadata key is empty"));
    }
    if key == GGUF_KEY_GENERAL_ALIGNMENT
        && !matches!(value, SavedValue::U32(v) if *v > 0 && v.is_power_of_two())
    {
        return Err(Error::new(format!(
            "{} must be a power-of-two u32",
            GGUF_KEY_GENERAL_ALIGNMENT
        )));
    }

    match value {
        SavedValue::String(v) => gguf_set_val_str(ctx, key, v),
        SavedValue::Bool(v) => gguf_set_val_bool(ctx, key, *v),
        SavedValue::U32(v) => gguf_set_val_u32(ctx, key, *v),
        SavedValue::I32(v) => gguf_set_val_i32(ctx, key, *v),
        SavedValue::U64(v) => gguf_set_val_u64(ctx, key, *v),
        SavedValue::I64(v) => gguf_set_val_i64(ctx, key, *v),
        SavedValue::F32(v) => gguf_set_val_f32(ctx, key, *v),
        SavedValue::StringArray(v) => {
            let strings: Vec<&str> = v.iter().map(String::as_str).collect();
            gguf_set_arr_str(ctx, key, &strings, strings.len());
        }
        SavedValue::I32Array(v) => set_array(ctx, key, v),
        SavedValue::F32Array(v) => set_array(ctx, key, v),
    }
    Ok(())
}

fn set_array<T: GgufValue>(ctx: &mut GgufContext, key: &str, values: &[T]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.encode()).collect();
    gguf_set_arr_data(ctx, key, T::TYPE, &bytes, values.len());
}

fn to_ggml_tensor(tensor: &SavedTensor) -> Result<GgmlTensor> {
    if tensor.name.is_empty() || tensor.name.len() >= GGML_MAX_NAME {
        return Err(Error::new(format!(
            "invalid tensor name '{}'",
            tensor.name
        )));
    }
    if tensor.shape.is_empty() || tensor.shape.len() > GGML_MAX_DIMS {
        return Err(Error::new(format!(
            "tensor '{}' has invalid number of dimensions: {}",
            tensor.name,
            tensor.shape.len()
        )));
    }
    if tensor.ty == GgmlType::Count {
        return Err(Error::new("invalid ggml type: Count"));
    }

    let block = ggml_blck_size(tensor.ty);
    if block <= 0 || tensor.shape[0] % block as usize != 0 {
        return Err(Error::new(format!(
            "tensor '{}' row size {} is not a multiple of block size {}",
            tensor.name, tensor.shape[0], block
        )));
    }

    let mut ne = [1i64; GGML_MAX_DIMS];
    for (i, &dim) in tensor.shape.iter().enumerate() {
        ne[i] = i64::try_from(dim)
            .map_err(|_| Error::new(format!("tensor '{}' has oversized dimension", tensor.name)))?;
    }

    // gguf_write_out requires contiguous strides; nb is not serialized.
    let mut nb = [0usize; GGML_MAX_DIMS];
    nb[0] = ggml_type_size(tensor.ty);
    for i in 1..GGML_MAX_DIMS {
        nb[i] = nb[i - 1] * ne[i - 1].max(0) as usize;
    }

    let t = GgmlTensor {
        name: tensor.name.clone(),
        ne,
        type_: tensor.ty,
        nb,
        data: tensor.data.clone(),
    };
    let expected = ggml_nbytes(&t);
    if tensor.data.len() != expected {
        return Err(Error::new(format!(
            "tensor '{}' has {} data bytes, expected {}",
            tensor.name,
            tensor.data.len(),
            expected
        )));
    }
    Ok(t)
}

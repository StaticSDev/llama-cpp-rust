//! Load a Qwen3.5 GGUF into the CPU runtime's [`Weights`] container.

use crate::core::{Error, Result};
use crate::gguf::{
    gguf_find_key, gguf_get_arr_data, gguf_get_arr_n, gguf_get_arr_str, gguf_get_tensor_name,
    gguf_get_val_f32, gguf_get_val_u32, gguf_init_from_file, GgmlType, GgufContext,
    GgufInitParams, GgufType,
};
use crate::runtime::types::{ModelParams, Weight, Weights};
use crate::vocab::metadata::Metadata;
use crate::vocab::LlamaVocab;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

fn key_id(ctx: &GgufContext, key: &str) -> Result<i64> {
    let id = gguf_find_key(ctx, key);
    if id < 0 {
        return Err(Error::new(format!("missing GGUF metadata key '{key}'")));
    }
    Ok(id)
}

fn get_u32(ctx: &GgufContext, key: &str) -> Result<u32> {
    Ok(gguf_get_val_u32(ctx, key_id(ctx, key)?))
}

fn get_f32(ctx: &GgufContext, key: &str) -> Result<f32> {
    Ok(gguf_get_val_f32(ctx, key_id(ctx, key)?))
}

fn decode_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn decode_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Mirror every GGUF KV into the maps `LlamaVocab::load` reads.
fn build_metadata(ctx: &GgufContext) -> Metadata {
    let mut meta = Metadata::default();
    for i in 0..ctx.kv.len() {
        let id = i as i64;
        let kv = &ctx.kv[i];
        let key = kv.key.clone();
        if kv.is_array {
            match kv.type_ {
                GgufType::String => {
                    let n = gguf_get_arr_n(ctx, id);
                    let vals = (0..n)
                        .map(|j| gguf_get_arr_str(ctx, id, j).to_owned())
                        .collect();
                    meta.string_arrays.insert(key, vals);
                }
                GgufType::Int32 => {
                    let vals = decode_i32(gguf_get_arr_data(ctx, id));
                    meta.i32_arrays.insert(key, vals);
                }
                GgufType::Float32 => {
                    let vals = decode_f32(gguf_get_arr_data(ctx, id));
                    meta.f32_arrays.insert(key, vals);
                }
                _ => {
                    meta.blobs.insert(key, gguf_get_arr_data(ctx, id).to_vec());
                }
            }
        } else {
            match kv.type_ {
                GgufType::String => {
                    meta.strings.insert(key, kv.get_str(0).to_owned());
                }
                GgufType::Bool => {
                    meta.bools.insert(key, kv.get_val::<bool>(0));
                }
                GgufType::Uint8 => {
                    meta.u32s.insert(key, kv.get_val::<u8>(0) as u32);
                }
                GgufType::Uint16 => {
                    meta.u32s.insert(key, kv.get_val::<u16>(0) as u32);
                }
                GgufType::Uint32 => {
                    meta.u32s.insert(key, kv.get_val::<u32>(0));
                }
                GgufType::Uint64 => {
                    meta.u32s.insert(key, kv.get_val::<u64>(0) as u32);
                }
                GgufType::Int8 => {
                    meta.u32s.insert(key, kv.get_val::<i8>(0) as u32);
                }
                GgufType::Int16 => {
                    meta.u32s.insert(key, kv.get_val::<i16>(0) as u32);
                }
                GgufType::Int32 => {
                    meta.u32s.insert(key, kv.get_val::<i32>(0) as u32);
                }
                GgufType::Int64 => {
                    meta.u32s.insert(key, kv.get_val::<i64>(0) as u32);
                }
                _ => {}
            }
        }
    }
    meta
}

/// Load model parameters, tensors and tokenizer from `path`.
pub fn load(path: &str) -> Result<Weights> {
    load_with_gpu(path, false)
}

/// Load a model, uploading its tensors to the GPU when `use_gpu` is set.
pub fn load_with_gpu(path: &str, use_gpu: bool) -> Result<Weights> {
    let ctx = gguf_init_from_file(path, &mut GgufInitParams::default())?;

    // The GGUF reader only fills `ctx.data` when a ggml context is requested.
    // Read the tensor blob directly so we do not also duplicate every tensor.
    let mut file = File::open(path).map_err(|e| Error::new(format!("open {path}: {e}")))?;
    file.seek(SeekFrom::Start(ctx.offset as u64))
        .map_err(|e| Error::new(format!("seek tensor blob: {e}")))?;
    let mut blob = vec![0u8; ctx.size];
    file.read_exact(&mut blob)
        .map_err(|e| Error::new(format!("read tensor blob: {e}")))?;

    let vocab_size = gguf_get_arr_n(&ctx, key_id(&ctx, "tokenizer.ggml.tokens")?);

    // Parsed for presence; the runtime derives head_dim from value_length.
    let _key_length = get_u32(&ctx, "qwen35.attention.key_length")?;

    let sections_id = key_id(&ctx, "qwen35.rope.dimension_sections")?;
    let sections = decode_u32(gguf_get_arr_data(&ctx, sections_id));
    let mut rope_sections = [0usize; 4];
    for (dst, src) in rope_sections.iter_mut().zip(sections.iter()) {
        *dst = *src as usize;
    }

    let params = ModelParams {
        n_layer: get_u32(&ctx, "qwen35.block_count")? as usize,
        n_embd: get_u32(&ctx, "qwen35.embedding_length")? as usize,
        n_head: get_u32(&ctx, "qwen35.attention.head_count")? as usize,
        n_head_kv: get_u32(&ctx, "qwen35.attention.head_count_kv")? as usize,
        head_dim: get_u32(&ctx, "qwen35.attention.value_length")? as usize,
        n_ff: get_u32(&ctx, "qwen35.feed_forward_length")? as usize,
        n_ctx: get_u32(&ctx, "qwen35.context_length")? as usize,
        vocab_size,
        eps: get_f32(&ctx, "qwen35.attention.layer_norm_rms_epsilon")?,
        rope_base: get_f32(&ctx, "qwen35.rope.freq_base")?,
        rope_dim: get_u32(&ctx, "qwen35.rope.dimension_count")? as usize,
        rope_sections,
        conv_kernel: get_u32(&ctx, "qwen35.ssm.conv_kernel")? as usize,
        ssm_inner: get_u32(&ctx, "qwen35.ssm.inner_size")? as usize,
        ssm_state: get_u32(&ctx, "qwen35.ssm.state_size")? as usize,
        ssm_group: get_u32(&ctx, "qwen35.ssm.group_count")? as usize,
        ssm_dt_rank: get_u32(&ctx, "qwen35.ssm.time_step_rank")? as usize,
        full_attn_interval: get_u32(&ctx, "qwen35.full_attention_interval")? as usize,
    };

    let mut map = HashMap::with_capacity(ctx.info.len());
    // Small F32 tensors (norms, conv kernels, biases) stay on the host even
    // in GPU mode: reading them one row at a time costs more than the math.
    let mut small = HashMap::new();
    for (i, info) in ctx.info.iter().enumerate() {
        let name = gguf_get_tensor_name(&ctx, i as i64).to_owned();
        let ne = [info.t.ne[0] as usize, info.t.ne[1] as usize];
        let offset = info.offset as usize;
        if info.t.type_ == GgmlType::F32 {
            let count = ne[0] * ne[1];
            if count <= 1 << 20 {
                let end = offset + count * 4;
                if let Some(bytes) = blob.get(offset..end) {
                    let values = bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    small.insert(name.clone(), values);
                }
            }
        }
        map.insert(
            name,
            Weight {
                ne,
                ty: info.t.type_,
                offset,
            },
        );
    }

    let meta = build_metadata(&ctx);
    let mut vocab = LlamaVocab::new();
    vocab.load(&meta)?;

    #[cfg(feature = "cuda")]
    let mut data = Arc::new(blob);
    #[cfg(not(feature = "cuda"))]
    let data = Arc::new(blob);

    #[cfg(feature = "cuda")]
    let gpu = if use_gpu {
        let gpu = crate::runtime::cuda::CudaWeights::new(data.as_slice(), map.values())?;
        // Free the host copy after upload to stay within the RAM budget.
        let _empty = std::mem::take(Arc::make_mut(&mut data));
        Some(Arc::new(gpu))
    } else {
        None
    };

    #[cfg(not(feature = "cuda"))]
    if use_gpu {
        return Err(Error::new("built without the cuda feature"));
    }

    Ok(Weights {
        map,
        small,
        data,
        params,
        vocab,
        #[cfg(feature = "cuda")]
        gpu,
    })
}

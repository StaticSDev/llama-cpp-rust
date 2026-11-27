//! `llama_model_quantize_impl`: the quantization driver.

use std::collections::{BTreeMap, HashMap};

use crate::arch::arch_from_string;
use crate::core::{Error, Result};

use super::gguf_io::{self, write_gguf, GgufKv, OutputTensor};
use super::imatrix;
use super::kernels::{dequantize_slab, quantize_slab, NativeQuantizer};
use super::names::{
    tensor_allows_quantization, tensor_name_match_token_embd, tensor_requires_imatrix,
};
use super::params::{ftype_default_type, FType, KvOverrideType, QuantizeParams};
use super::state::{metadata_from_tensors, ModelInfo, QuantizeState, TensorMetadata};
use super::target_bpw::target_bpw_type;
use super::types::{QTensor, QType};
use super::GGUF_ALIGNMENT;

const GGML_QNT_VERSION: u32 = 2;
const LLM_KV_GENERAL_QUANTIZATION_VERSION: &str = "general.quantization_version";
const LLM_KV_GENERAL_FILE_TYPE: &str = "general.file_type";
const LLM_KV_SPLIT_NO: &str = "split.no";
const LLM_KV_SPLIT_COUNT: &str = "split.count";
const LLM_KV_SPLIT_TENSORS_COUNT: &str = "split.tensors.count";
const LLM_KV_BLOCK_COUNT: &str = "block_count";
#[allow(dead_code)]
const LLM_KV_ATTENTION_SHARED_KV_LAYERS: &str = "attention.shared_kv_layers";

fn kv_u32(kv: &[GgufKv], key: &str) -> u32 {
    kv.iter()
        .find(|k| k.key == key)
        .and_then(|k| k.as_u32())
        .unwrap_or(0)
}

#[allow(dead_code)]
fn kv_i32(kv: &[GgufKv], key: &str) -> i32 {
    kv.iter()
        .find(|k| k.key == key)
        .and_then(|k| k.as_i32())
        .unwrap_or(0)
}

fn model_info_from_gguf(gguf: &gguf_io::LoadedGguf) -> ModelInfo {
    let arch = arch_from_string(&gguf.arch_name);
    let n_head = kv_u32(&gguf.kv, "attention.head_count");
    let n_head_kv = kv_u32(&gguf.kv, "attention.head_count_kv");
    let is_70b = arch == crate::arch::Arch::Llama && gguf.block_count == 80 && n_head != n_head_kv;
    let n_gqa = if n_head_kv == 0 {
        1
    } else {
        n_head / n_head_kv
    };
    let n_expert = kv_u32(&gguf.kv, "expert.count").max(1);
    ModelInfo {
        arch,
        is_70b,
        n_expert,
        n_layer_all: gguf.block_count,
        n_gqa,
    }
}

fn apply_kv_overrides(kv: &mut Vec<GgufKv>, overrides: &[super::params::KvOverride]) {
    for o in overrides {
        if o.key.is_empty() {
            continue;
        }
        let existing = kv.iter().position(|k| k.key == o.key);
        let mut entry = match existing {
            Some(i) => kv.remove(i),
            None => GgufKv {
                key: o.key.clone(),
                type_: 8,
                is_array: false,
                elem_type: 8,
                data: Vec::new(),
                strings: vec![String::new()],
            },
        };
        match o.tag {
            KvOverrideType::Float => entry.set_f32(o.val_f64 as f32),
            KvOverrideType::Int => entry.set_u32(o.val_i64.unsigned_abs() as u32),
            KvOverrideType::Bool => {
                entry.type_ = 7;
                entry.elem_type = 7;
                entry.data = vec![o.val_bool as u8];
                entry.strings.clear();
            }
            KvOverrideType::Str => {
                entry.type_ = 8;
                entry.elem_type = 8;
                entry.data.clear();
                entry.strings = vec![o.val_str.clone()];
            }
        }
        kv.push(entry);
    }
}

fn prepare_tensors(
    gguf: &gguf_io::LoadedGguf,
    params: &QuantizeParams,
) -> (
    Vec<QTensor>,
    BTreeMap<i32, String>,
    i32,
    std::collections::HashSet<String>,
) {
    let prune = &params.prune_layers;
    let mut mapped = BTreeMap::new();
    let mut next_id = 0i32;
    let n_unused_from = if gguf.shared_kv > 0 && gguf.shared_kv <= gguf.block_count {
        (gguf.block_count - gguf.shared_kv) as i32
    } else {
        -1
    };
    let mut tensors = Vec::new();
    for t in &gguf.tensors {
        let remapped = imatrix::remap_layer(&t.name, prune, &mut mapped, &mut next_id);
        if remapped.is_empty() {
            continue;
        }
        let mut out = t.clone();
        out.name = remapped;
        tensors.push(out);
    }
    if params.keep_split {
        tensors.sort_by_key(|t| t.split);
    }
    let unused = imatrix::unused_kv_tensors(&tensors, n_unused_from);
    (tensors, mapped, next_id, unused)
}

fn build_locked(
    qs: &QuantizeState,
    tensors: &[QTensor],
    metadata: &[TensorMetadata],
    unused: &std::collections::HashSet<String>,
) -> HashMap<String, QType> {
    let mut locked = HashMap::new();
    for (i, t) in tensors.iter().enumerate() {
        if !metadata[i].allows_quantization {
            continue;
        }
        let mut lt = qs.tensor_type_override_for(&t.name);
        if metadata[i].category == super::names::TensorCategory::TokenEmbd {
            let named = lt != QType::Count && t.name == "per_layer_token_embd.weight";
            if qs.params.token_embedding_type != QType::Count && !named {
                lt = qs.params.token_embedding_type;
            } else if lt == QType::Count
                && qs.has_tied_embeddings
                && qs.params.output_tensor_type != QType::Count
            {
                lt = qs.params.output_tensor_type;
            }
        } else if metadata[i].category == super::names::TensorCategory::Output
            && qs.params.output_tensor_type != QType::Count
        {
            lt = qs.params.output_tensor_type;
        }
        if lt != QType::Count {
            locked.insert(t.name.clone(), lt);
        }
    }
    // Unused KV tensors keep their converted type instead of being budgeted.
    for (i, t) in tensors.iter().enumerate() {
        if metadata[i].allows_quantization && unused.contains(&t.name) {
            locked.insert(t.name.clone(), metadata[i].target_type);
        }
    }
    locked
}

/// Port of `llama_model_quantize_impl`.
pub fn llama_model_quantize_impl(
    fname_inp: &str,
    fname_out: &str,
    params: &QuantizeParams,
) -> Result<()> {
    let quant = NativeQuantizer;
    let mut ftype = params.ftype;
    let nthread = if params.nthread > 0 {
        params.nthread as usize
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    };
    let default_type = ftype_default_type(ftype);
    if default_type == QType::Count {
        return Err(Error::new(format!(
            "invalid output file type {}",
            ftype as i32
        )));
    }

    let gguf = gguf_io::read_gguf(fname_inp)?;
    let model = model_info_from_gguf(&gguf);
    let mut qs = QuantizeState::new(model, params);
    if params.only_copy {
        ftype = FType::from_i32(gguf.source_ftype).unwrap_or(ftype);
    }

    let (values, activations, statistics, stats_schema) = imatrix::prepare_side_data(params)?;
    qs.has_imatrix = params.has_imatrix();
    qs.has_activations = params.has_activations();
    if qs.has_imatrix {
        crate::llama_log_info!(
            "llama_model_quantize_impl: have importance matrix data with {} entries",
            values.len()
        );
    }
    if qs.has_activations {
        crate::llama_log_info!(" - {} activations", activations.len());
    }
    if params.has_statistics() {
        crate::llama_log_info!(" and {} statistics", statistics.len());
    }
    crate::llama_log_info!("\n");

    let mut out_kv = gguf.kv.clone();
    gguf_io::set_u32(
        &mut out_kv,
        LLM_KV_GENERAL_QUANTIZATION_VERSION,
        GGML_QNT_VERSION,
    );
    gguf_io::set_u32(&mut out_kv, LLM_KV_GENERAL_FILE_TYPE, ftype.as_i32() as u32);
    gguf_io::remove_key(&mut out_kv, LLM_KV_SPLIT_NO);
    gguf_io::remove_key(&mut out_kv, LLM_KV_SPLIT_COUNT);
    gguf_io::remove_key(&mut out_kv, LLM_KV_SPLIT_TENSORS_COUNT);
    apply_kv_overrides(&mut out_kv, &params.kv_overrides);

    let (mut tensors, mapped, blk_id, unused) = prepare_tensors(&gguf, params);
    if !params.prune_layers.is_empty() {
        gguf_io::set_u32(&mut out_kv, LLM_KV_BLOCK_COUNT, blk_id as u32);
    }

    let mut metadata = metadata_from_tensors(&tensors);
    qs.init_counters(&mut metadata);

    let mut n_split = 1u16;
    if params.keep_split {
        for t in &tensors {
            n_split = n_split.max(t.split + 1);
        }
    }
    if n_split > 1 {
        gguf_io::set_u32(&mut out_kv, LLM_KV_SPLIT_COUNT, n_split as u32);
        gguf_io::set_u32(
            &mut out_kv,
            LLM_KV_SPLIT_TENSORS_COUNT,
            tensors.len() as u32,
        );
    }

    let mut will_require_imatrix = false;
    for (i, t) in tensors.iter().enumerate() {
        metadata[i].allows_quantization = tensor_allows_quantization(params, qs.model.arch, t);
        metadata[i].target_type = if metadata[i].allows_quantization {
            qs.tensor_get_type(params, t, default_type, &metadata[i])?
        } else {
            t.ty
        };
        if metadata[i].allows_quantization && unused.contains(&t.name) {
            metadata[i].target_type = qs.tensor_type_fallback(t, QType::Q4_0)?;
            crate::llama_log_info!(
                "llama_model_quantize_impl: unused tensor ({} shares an earlier layer's kv cache) - converting to {}\n",
                metadata[i].name,
                metadata[i].target_type.name()
            );
        }
        metadata[i].requires_imatrix =
            tensor_requires_imatrix(&t.name, metadata[i].target_type, ftype);
        if params.has_imatrix() {
            metadata[i].remapped_imatrix_name =
                imatrix::remap_imatrix(&t.name, &mapped).unwrap_or_else(|_| t.name.clone());
        } else if metadata[i].allows_quantization && metadata[i].requires_imatrix {
            if params.dry_run {
                will_require_imatrix = true;
            } else {
                return Err(Error::new(format!(
                    "this quantization requires an imatrix! offending tensor: {}, target type: {}",
                    metadata[i].name,
                    metadata[i].target_type.name()
                )));
            }
        }
    }

    if values.is_empty() && qs.has_imatrix {
        imatrix::report_missing_imatrix(&values, &metadata, &unused);
    }

    if params.target_requested() && !params.only_copy {
        if !params.has_imatrix() {
            return Err(Error::new(
                "--target-bpw/--target-size require an imatrix but none was provided",
            ));
        }
        let locked = build_locked(&qs, &tensors, &metadata, &unused);
        let locked_ref = if locked.is_empty() {
            None
        } else {
            Some(&locked)
        };
        let bpw = target_bpw_type(
            &mut qs,
            &quant,
            &tensors,
            &metadata,
            &mapped,
            &gguf.meta_bytes,
            gguf.meta_size,
            &gguf.general_name,
            &gguf.arch_name,
            Some(&values),
            if activations.is_empty() {
                None
            } else {
                Some(&activations)
            },
            if statistics.is_empty() {
                None
            } else {
                Some(&statistics)
            },
            if stats_schema.is_empty() {
                None
            } else {
                Some(&stats_schema)
            },
            locked_ref,
            nthread,
        )?;
        for (i, t) in tensors.iter_mut().enumerate() {
            if let Some(&ty) = bpw.get(&t.name) {
                metadata[i].target_type = ty;
                metadata[i].requires_imatrix = tensor_requires_imatrix(&t.name, ty, ftype);
            }
        }
    }

    let mut total_size_org = 0usize;
    let mut total_size_new = 0usize;
    let max_buf = params.max_buf_size.max(1);
    let mut outputs: Vec<Vec<u8>> = Vec::with_capacity(tensors.len());

    for (i, t) in tensors.iter().enumerate() {
        if params.aborted() {
            return Err(Error::new("interrupted"));
        }
        let cur_type = t.ty;
        let new_type = metadata[i].target_type;
        let quantize = cur_type != new_type;
        let tensor_size = t.current_nbytes();
        crate::llama_log_info!(
            "[{:4}/{:4}] {:36} - type = {:6}, ",
            i + 1,
            tensors.len(),
            t.name,
            cur_type.name()
        );

        let mut new_size = 0usize;
        let mut out_data = Vec::new();
        if params.dry_run {
            if quantize {
                new_size = t.nrows().max(0) as usize * new_type.row_size(t.ne[0]);
                crate::llama_log_info!(
                    "size = {:.2} MiB -> {:.2} MiB ({})\n",
                    tensor_size as f64 / 1024.0 / 1024.0,
                    new_size as f64 / 1024.0 / 1024.0,
                    new_type.name()
                );
                if !will_require_imatrix && metadata[i].requires_imatrix {
                    will_require_imatrix = true;
                }
            } else {
                new_size = tensor_size;
                crate::llama_log_info!("size = {:.2} MiB\n", new_size as f64 / 1024.0 / 1024.0);
            }
        } else if !quantize {
            new_size = tensor_size;
            crate::llama_log_info!("size = {:.2} MiB\n", tensor_size as f64 / 1024.0 / 1024.0);
            out_data.extend_from_slice(&t.data);
        } else {
            let imatrix = if values.is_empty() {
                None
            } else {
                let name = &metadata[i].remapped_imatrix_name;
                match values.get(name) {
                    Some(v)
                        if v.len() == t.ne[0] as usize * t.ne[2] as usize * t.ne[3] as usize =>
                    {
                        Some(v.as_slice())
                    }
                    Some(_) if tensor_name_match_token_embd(&t.name) => None,
                    Some(v) => {
                        return Err(Error::new(format!(
                            "imatrix size {} is different from tensor size {} for {}",
                            v.len(),
                            t.ne[0] * t.ne[2] * t.ne[3],
                            t.name
                        )))
                    }
                    None => None,
                }
            };
            if imatrix.is_none() && metadata[i].requires_imatrix {
                return Err(Error::new(format!(
                    "Missing importance matrix for tensor {} in a very low-bit quantization",
                    t.name
                )));
            }
            if cur_type.is_quantized() && !params.allow_requantize {
                return Err(Error::new(format!(
                    "requantizing from type {} is disabled",
                    cur_type.name()
                )));
            }
            crate::llama_log_info!("converting to {:7} ", new_type.name());

            let n_per_row = t.ne[0].max(0) as usize;
            let nrows_total = t.nrows().max(0);
            let row_size_src = t.row_size(cur_type);
            let row_size_dst = t.row_size(new_type);
            if row_size_src == 0 || row_size_dst == 0 {
                return Err(Error::new(format!(
                    "invalid row size for tensor {}",
                    t.name
                )));
            }
            let bytes_per_row = row_size_src
                + row_size_dst
                + if cur_type == QType::F32 {
                    0
                } else {
                    n_per_row * 4
                };
            let nrows_slab = (max_buf / bytes_per_row.max(1)) as i64;
            let nrows_slab = nrows_slab.clamp(1, nrows_total.max(1));
            let min_chunk_size = 32 * 512i64;
            let chunk_size = if n_per_row as i64 >= min_chunk_size {
                n_per_row as i64
            } else {
                n_per_row as i64 * ((min_chunk_size + n_per_row as i64 - 1) / n_per_row as i64)
            };
            let _ = chunk_size;

            let mut slab_f32 = Vec::new();
            let mut work = Vec::new();
            let mut ir = 0i64;
            while ir < nrows_total {
                let nrows_cur = nrows_slab.min(nrows_total - ir);
                let start = ir as usize * row_size_src;
                let end = start + nrows_cur as usize * row_size_src;
                let src = t.data.get(start..end).ok_or_else(|| {
                    Error::new(format!("tensor {} data range out of bounds", t.name))
                })?;
                if cur_type == QType::F32 {
                    slab_f32.clear();
                    for chunk in src.chunks_exact(4) {
                        slab_f32.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                } else {
                    slab_f32.resize(nrows_cur as usize * n_per_row, 0.0);
                    dequantize_slab(&quant, cur_type, src, &mut slab_f32, nthread)?;
                }
                let nrows_per_expert = t.ne[1].max(1);
                let size = quantize_slab(
                    &quant,
                    new_type,
                    &slab_f32,
                    &mut work,
                    ir,
                    nrows_cur,
                    nrows_per_expert,
                    n_per_row,
                    imatrix,
                )?;
                out_data.extend_from_slice(&work[..size]);
                new_size += size;
                ir += nrows_cur;
            }
            crate::llama_log_info!(
                "size = {:.2} MiB -> {:.2} MiB\n",
                tensor_size as f64 / 1024.0 / 1024.0,
                new_size as f64 / 1024.0 / 1024.0
            );
        }
        total_size_org += tensor_size;
        total_size_new += new_size;
        outputs.push(out_data);
    }

    if !params.dry_run {
        let mut final_tensors = Vec::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            final_tensors.push(OutputTensor {
                name: &t.name,
                ne: t.ne,
                ty: metadata[i].target_type,
                data: &outputs[i],
                split: t.split,
            });
        }
        if params.keep_split && n_split > 1 {
            for s in 0..n_split {
                gguf_io::set_u32(&mut out_kv, LLM_KV_SPLIT_NO, s as u32);
                let split_tensors: Vec<OutputTensor<'_>> = final_tensors
                    .iter()
                    .filter(|t| t.split == s)
                    .cloned()
                    .collect();
                let fname = format!(
                    "{}-{:05}-of-{:05}.gguf",
                    fname_out.trim_end_matches(".gguf"),
                    s,
                    n_split
                );
                write_gguf(&fname, &mut out_kv, &split_tensors, GGUF_ALIGNMENT)?;
            }
        } else {
            write_gguf(fname_out, &mut out_kv, &final_tensors, GGUF_ALIGNMENT)?;
        }
    }

    crate::llama_log_info!(
        "llama_model_quantize_impl: model size  = {:8.2} MiB\n",
        total_size_org as f64 / 1024.0 / 1024.0
    );
    crate::llama_log_info!(
        "llama_model_quantize_impl: quant size  = {:8.2} MiB\n",
        total_size_new as f64 / 1024.0 / 1024.0
    );
    if !params.imatrix.is_empty() && params.dry_run && will_require_imatrix {
        crate::llama_log_warn!("llama_model_quantize_impl: WARNING: dry run completed successfully, but actually completing this quantization will require an imatrix!\n");
    }
    if qs.n_fallback > 0 {
        crate::llama_log_warn!(
            "llama_model_quantize_impl: WARNING: {} of {} tensor(s) required fallback quantization\n",
            qs.n_fallback,
            tensors.len()
        );
    }
    Ok(())
}

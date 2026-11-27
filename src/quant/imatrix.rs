//! Importance-matrix ingestion, layer pruning/remapping and coverage checks.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::core::{Error, Result};

use super::names::{tensor_name_match_output_weight, tensor_name_match_token_embd};
use super::params::{imatrix_map, validate_side_data, QuantizeParams};
use super::types::QTensor;

/// `values_data`, `activations_data` and `statistics_data` in the C++ driver.
pub type SideData = HashMap<String, Vec<f32>>;

/// Build the three side-data maps and validate their entries.
pub fn prepare_side_data(
    params: &QuantizeParams,
) -> Result<(SideData, SideData, SideData, Vec<String>)> {
    let values = imatrix_map(&params.imatrix);
    let activations = imatrix_map(&params.activations);
    let statistics = imatrix_map(&params.statistics);
    validate_side_data(&params.imatrix, "imatrix contains non-finite value")?;
    validate_side_data(&params.activations, "activations contain non-finite value")?;
    Ok((values, activations, statistics, params.stats_schema.clone()))
}

/// `remap_layer`: rename block indices after pruning layers.
pub fn remap_layer(
    orig_name: &str,
    prune: &[i32],
    mapped: &mut BTreeMap<i32, String>,
    next_id: &mut i32,
) -> String {
    if prune.is_empty() {
        return orig_name.to_owned();
    }
    let Some(blk) = super::names::parse_block(orig_name) else {
        return orig_name.to_owned();
    };
    if !mapped.contains_key(&blk) {
        if prune.contains(&blk) {
            mapped.insert(blk, String::new());
        } else if blk < prune[0] {
            mapped.insert(blk, blk.to_string());
            *next_id = blk + 1;
        } else {
            mapped.insert(blk, next_id.to_string());
            *next_id += 1;
        }
    }
    let mapped_id = &mapped[&blk];
    if mapped_id.is_empty() {
        return String::new();
    }
    let dot = orig_name.find('.').unwrap_or(0) + 1;
    let end = orig_name[dot..]
        .find('.')
        .map(|p| dot + p)
        .unwrap_or(orig_name.len());
    let mut out = String::with_capacity(orig_name.len());
    out.push_str(&orig_name[..dot]);
    out.push_str(mapped_id);
    out.push_str(&orig_name[end..]);
    out
}

/// `remap_imatrix`: translate an imatrix name back to its original layer id.
pub fn remap_imatrix(orig_name: &str, mapped: &BTreeMap<i32, String>) -> Result<String> {
    if mapped.is_empty() {
        return Ok(orig_name.to_owned());
    }
    let Some(blk) = super::names::parse_block(orig_name) else {
        return Ok(orig_name.to_owned());
    };
    let blk_text = blk.to_string();
    for (&original, remapped) in mapped {
        if *remapped == blk_text {
            let dot = orig_name.find('.').unwrap_or(0) + 1;
            let end = orig_name[dot..]
                .find('.')
                .map(|p| dot + p)
                .unwrap_or(orig_name.len());
            let mut out = String::with_capacity(orig_name.len());
            out.push_str(&orig_name[..dot]);
            out.push_str(&original.to_string());
            out.push_str(&orig_name[end..]);
            return Ok(out);
        }
    }
    Err(Error::new(format!(
        "imatrix mapping error for {}",
        orig_name
    )))
}

/// Build the set of tensors whose KV cache is shared with an earlier layer.
pub fn unused_kv_tensors(tensors: &[QTensor], n_unused_from: i32) -> HashSet<String> {
    let mut set = HashSet::new();
    for t in tensors {
        if super::names::tensor_name_match_unused_kv(&t.name, n_unused_from) {
            set.insert(t.name.clone());
        }
    }
    set
}

/// Warn about quantizable tensors with no importance-matrix data.
pub fn report_missing_imatrix(
    values: &SideData,
    metadata: &[super::state::TensorMetadata],
    unused: &HashSet<String>,
) {
    let mut missing = Vec::new();
    for tm in metadata {
        if !tm.allows_quantization
            || unused.contains(&tm.name)
            || tensor_name_match_token_embd(&tm.name)
            || tensor_name_match_output_weight(&tm.name)
        {
            continue;
        }
        if !values.contains_key(&tm.remapped_imatrix_name) {
            missing.push(tm.name.clone());
        }
    }
    if !missing.is_empty() {
        crate::llama_log_warn!(
            "llama_model_quantize_impl: {} quantizable tensors have no imatrix data:\n",
            missing.len()
        );
        for name in missing {
            crate::llama_log_warn!("llama_model_quantize_impl:     {}\n", name);
        }
    }
}

/// Look up an importance-matrix row slice, broadcasting one slice if needed.
pub fn broadcast<'a>(
    src: Option<&'a Vec<f32>>,
    storage: &'a mut Vec<f32>,
    ne2: usize,
    n_per_row: usize,
) -> Option<&'a [f32]> {
    let src = src?;
    let req = ne2 * n_per_row;
    if src.len() == req {
        return Some(&src[..]);
    }
    if src.len() == n_per_row && n_per_row > 0 {
        storage.resize(req, 0.0);
        for s in 0..ne2 {
            let start = s * n_per_row;
            storage[start..start + n_per_row].copy_from_slice(&src[..n_per_row]);
        }
        return Some(&storage[..]);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaps_layers() {
        let prune = vec![1, 3];
        let mut mapped = BTreeMap::new();
        let mut next = 0;
        assert_eq!(
            remap_layer("blk.0.attn_v.weight", &prune, &mut mapped, &mut next),
            "blk.0.attn_v.weight"
        );
        assert_eq!(next, 1);
        assert_eq!(
            remap_layer("blk.2.attn_v.weight", &prune, &mut mapped, &mut next),
            "blk.1.attn_v.weight"
        );
        assert_eq!(
            remap_layer("blk.4.attn_v.weight", &prune, &mut mapped, &mut next),
            "blk.2.attn_v.weight"
        );
        assert!(remap_layer("blk.3.attn_v.weight", &prune, &mut mapped, &mut next).is_empty());
        assert_eq!(
            remap_imatrix("blk.1.attn_v.weight", &mapped).unwrap(),
            "blk.2.attn_v.weight"
        );
    }
}

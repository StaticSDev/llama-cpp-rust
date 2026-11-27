//! `--target-bpw` / `--target-size` tensor-type selection.
//!
//! Direct port of `target_bpw_type` in `llama-quant.cpp`: sampling, error
//! metrics, Pareto pruning and the Lagrangian/auction solver.

use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{Read, Write};

use crate::core::Mt19937;
use crate::core::{Error, Result};

use super::imatrix::{broadcast, remap_imatrix, SideData};
use super::kernels::Quantizer;
use super::names::{
    tensor_allows_quantization, tensor_get_category, tensor_name_match_output_weight,
    tensor_name_match_token_embd, TensorCategory,
};
use super::params::QuantizeParams;
use super::regex_lite::regex_match;
use super::state::{QuantizeState, TensorMetadata};
use super::target_metrics::{
    cnif, compute_quant_error, outlier_smoothing, randomized_svd_condition_number, squash_kappa,
    winsorize_values, MseCache, QuantError, TensorScore, WceCache,
};
use super::target_types::*;
use super::types::{fallback_type_for, QTensor, QType, TYPE_COUNT};

const STAT_NAMES: [&str; 5] = ["sum_sq", "h_norm", "l2_dist", "cossim", "pearson"];

fn tensor_bytes(t: &QTensor, q: QType) -> usize {
    t.nrows().max(0) as usize * q.row_size(t.ne[0])
}

fn tensor_bpw(t: &QTensor, q: QType) -> f64 {
    let n = t.nelements();
    if n == 0 {
        return 0.0;
    }
    tensor_bytes(t, q) as f64 * 8.0 / n as f64
}

fn is_compatible(t: &QTensor, q: QType) -> bool {
    let blck = q.blck_size();
    blck <= 1 || t.ne[0] % blck == 0
}

fn make_compatible(t: &QTensor, q: QType) -> QType {
    if is_compatible(t, q) {
        return q;
    }
    let fb = fallback_type_for(q).unwrap_or(QType::Count);
    if fb != QType::Count && is_compatible(t, fb) {
        fb
    } else {
        QType::F16
    }
}

fn can_quantize(qs: &QuantizeState, params: &QuantizeParams, t: &QTensor) -> bool {
    if t.n_dims() < 2 || t.n_dims() > 3 {
        return false;
    }
    tensor_allows_quantization(params, qs.model.arch, t)
}

fn has_side_data(map: Option<&SideData>, name: &str, n2: usize, n_per_row: usize) -> bool {
    let Some(m) = map else { return false };
    match m.get(name) {
        Some(v) => v.len() == n2 * n_per_row || v.len() == n_per_row,
        None => false,
    }
}

fn sample_count(n: i64, rows: i64, n2: i64, has_acts: bool) -> i64 {
    let tensor_budget = (if has_acts { 1.0 } else { 0.5 }) * 1024.0 * 1024.0;
    let scale = (rows.max(1) as f64 / 4096.0).sqrt().clamp(0.5, 2.0);
    let slice_budget = tensor_budget * scale / n2.max(1) as f64;
    let min_r = if has_acts { 512 } else { 256 };
    let max_r = 4096;
    let mut tr = (slice_budget / n.max(1) as f64).round() as i64;
    tr = min_r.max(tr.min(rows.min(max_r)));
    if rows <= min_r * 2 {
        tr = rows;
    }
    tr
}

fn model_id(
    metadata_bytes: &[u8],
    values: Option<&SideData>,
    activations: Option<&SideData>,
    statistics: Option<&SideData>,
    stats_schema: Option<&[String]>,
    mapped: &std::collections::BTreeMap<i32, String>,
) -> u64 {
    let mut h = djb2_hash(metadata_bytes, 5381);
    for m in [values, activations, statistics] {
        let present = m.is_some() as u8;
        h = djb2_hash(&[present], h);
        let Some(map) = m else { continue };
        let mut names: Vec<&String> = map.keys().collect();
        names.sort();
        h = hash_u64(h, names.len() as u64);
        for name in names {
            h = hash_u64(h, name.len() as u64);
            h = djb2_hash(name.as_bytes(), h);
            let v = &map[name];
            h = hash_u64(h, v.len() as u64);
            h = djb2_hash(&f32_bytes(v), h);
        }
    }
    let has_schema = stats_schema.is_some() as u8;
    h = djb2_hash(&[has_schema], h);
    if let Some(schema) = stats_schema {
        h = hash_u64(h, schema.len() as u64);
        for name in schema {
            h = hash_u64(h, name.len() as u64);
            h = djb2_hash(name.as_bytes(), h);
        }
    }
    for (&blk, num) in mapped {
        h = hash_u64(h, blk as u64);
        h = hash_u64(h, num.len() as u64);
        h = djb2_hash(num.as_bytes(), h);
    }
    h
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * std::mem::size_of::<f32>());
    for &x in v {
        out.extend_from_slice(&x.to_ne_bytes());
    }
    out
}

fn checkpoint_path(qs: &QuantizeParams, model_name: &str, arch_name: &str, id: u64) -> String {
    let mut name = model_name.to_owned();
    if let Some(pos) = name.rfind('/') {
        name = name[pos + 1..].to_owned();
    }
    name = name.replace(' ', "_");
    if name.is_empty() {
        name = arch_name.to_owned();
    }
    let mut path = format!("{}-{:016x}.bpw_state", name, id);
    if !qs.state_file.is_empty() {
        path = qs.state_file.clone();
    }
    path
}

fn save_state(path: &str, model_id: u64, all: &[TypeChoice], tensors: &[QTensor]) {
    let tmp = format!("{}.tmp", path);
    let mut ofs = match File::create(&tmp) {
        Ok(f) => f,
        Err(e) => {
            crate::llama_log_warn!(
                "\n\ttarget_bpw_type: failed to save progress to {} (open error): {}\n",
                tmp,
                e
            );
            return;
        }
    };
    let mut write_u64 =
        |v: u64, f: &mut File| -> std::io::Result<()> { f.write_all(&v.to_le_bytes()) };
    let mut write_i32 =
        |v: i32, f: &mut File| -> std::io::Result<()> { f.write_all(&v.to_le_bytes()) };
    let mut write_f32 =
        |v: f32, f: &mut File| -> std::io::Result<()> { f.write_all(&v.to_le_bytes()) };
    let mut write_f64 =
        |v: f64, f: &mut File| -> std::io::Result<()> { f.write_all(&v.to_le_bytes()) };
    let ok = (|| -> std::io::Result<()> {
        write_u64(STATE_MAGIC, &mut ofs)?;
        write_u64(model_id, &mut ofs)?;
        write_u64(all.len() as u64, &mut ofs)?;
        for tn in all {
            let name = &tensors[tn.tensor].name;
            write_u64(name.len() as u64, &mut ofs)?;
            ofs.write_all(name.as_bytes())?;
            write_u64(tn.candidates.len() as u64, &mut ofs)?;
            write_i32(tn.choice, &mut ofs)?;
            write_f32(tn.min_bpw, &mut ofs)?;
            write_f32(tn.max_bpw, &mut ofs)?;
            write_u64(tn.n_elements as u64, &mut ofs)?;
            for c in &tn.candidates {
                write_i32(c.ty as i32, &mut ofs)?;
                write_f32(c.bpw, &mut ofs)?;
                write_u64(c.bytes as u64, &mut ofs)?;
                write_f64(c.error, &mut ofs)?;
            }
        }
        ofs.flush()
    })();
    if ok.is_err() {
        let _ = std::fs::remove_file(&tmp);
        crate::llama_log_warn!(
            "\n\ttarget_bpw_type: failed to save progress to {} (write error)\n",
            tmp
        );
        return;
    }
    drop(ofs);
    let _ = std::fs::remove_file(path);
    if std::fs::rename(&tmp, path).is_err() {
        crate::llama_log_warn!(
            "\n\ttarget_bpw_type: failed to save progress to {} (rename error), kept in {}\n",
            path,
            tmp
        );
        return;
    }
    crate::llama_log_info!(
        "\ttarget_bpw_type: saved target progress for {} tensors to {}\n",
        all.len(),
        path
    );
}

fn read_u64_file(f: &mut File) -> Option<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).ok().map(|_| u64::from_le_bytes(buf))
}

fn load_state(path: &str, id: u64) -> HashMap<String, TypeChoice> {
    let mut ifs = match File::open(path) {
        Ok(f) => f,
        Err(_) => return HashMap::new(),
    };
    let (Some(magic), Some(file_id)) = (read_u64_file(&mut ifs), read_u64_file(&mut ifs)) else {
        return HashMap::new();
    };
    if file_id != id {
        crate::llama_log_warn!("target_bpw_type: state file does not match this model, imatrix or pruned layers, ignoring\n");
        return HashMap::new();
    }
    if magic != STATE_MAGIC {
        crate::llama_log_warn!("target_bpw_type: invalid state file, ignoring\n");
        return HashMap::new();
    }
    let Some(n) = read_u64_file(&mut ifs) else {
        return HashMap::new();
    };
    let mut out = HashMap::new();
    for _ in 0..n {
        let Some(len) = read_u64_file(&mut ifs) else {
            break;
        };
        if len >= 64 {
            crate::llama_log_warn!(
                "target_bpw_type: state file name length exceeds limits, ignoring\n"
            );
            return HashMap::new();
        }
        let mut name_bytes = vec![0u8; len as usize];
        if ifs.read_exact(&mut name_bytes).is_err() {
            break;
        }
        let name = String::from_utf8_lossy(&name_bytes).into_owned();
        let Some(sz) = read_u64_file(&mut ifs) else {
            break;
        };
        if sz == 0 || sz > QUANT_TYPES.len() as u64 {
            crate::llama_log_warn!(
                "target_bpw_type: candidate count out of range, ignoring state file\n"
            );
            return HashMap::new();
        }
        let mut choice = [0u8; 4];
        if ifs.read_exact(&mut choice).is_err() {
            break;
        }
        let mut fbuf = [0u8; 4];
        let mut dbuf = [0u8; 8];
        if ifs.read_exact(&mut fbuf).is_err() {
            break;
        }
        let min_bpw = f32::from_le_bytes(fbuf);
        if ifs.read_exact(&mut fbuf).is_err() {
            break;
        }
        let max_bpw = f32::from_le_bytes(fbuf);
        let Some(ne) = read_u64_file(&mut ifs) else {
            break;
        };
        let mut tc = TypeChoice {
            choice: i32::from_le_bytes(choice),
            min_bpw,
            max_bpw,
            n_elements: ne as usize,
            ..TypeChoice::default()
        };
        for _ in 0..sz {
            if ifs.read_exact(&mut choice).is_err() {
                break;
            }
            let t = i32::from_le_bytes(choice);
            if !QUANT_TYPES.iter().any(|q| *q as i32 == t) {
                crate::llama_log_warn!("target_bpw_type: candidate type is not in the selection pool, ignoring state file\n");
                return HashMap::new();
            }
            if ifs.read_exact(&mut fbuf).is_err() {
                break;
            }
            let bpw = f32::from_le_bytes(fbuf);
            if ifs.read_exact(&mut dbuf).is_err() {
                break;
            }
            let b = u64::from_le_bytes(dbuf);
            if ifs.read_exact(&mut dbuf).is_err() {
                break;
            }
            let error = f64::from_le_bytes(dbuf);
            tc.candidates.push(TypeScores {
                ty: QType::from_i32(t).unwrap_or(QType::Count),
                bpw,
                bytes: b as usize,
                error,
                ..TypeScores::default()
            });
        }
        out.insert(name, tc);
    }
    crate::llama_log_info!(
        "\ttarget_bpw_type: using {} (data for {} tensors loaded)\n",
        path,
        out.len()
    );
    out
}

pub fn simplify_pareto(candidates: &mut Vec<TypeScores>) {
    candidates.sort_by(|a, b| {
        a.bytes.cmp(&b.bytes).then_with(|| {
            a.error
                .partial_cmp(&b.error)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.ty.cmp(&b.ty))
        })
    });
    candidates.dedup_by(|a, b| a.bytes == b.bytes);
    let mut hull = Vec::new();
    let mut min_err = LQ_INFINITE;
    for c in candidates.iter() {
        if c.error < min_err {
            min_err = c.error;
            hull.push(c.clone());
        }
    }
    *candidates = hull;
    if candidates.len() < 3 {
        return;
    }
    let cross = |a: &TypeScores, b: &TypeScores, c: &TypeScores| -> f64 {
        (b.bytes as f64 - a.bytes as f64) * (c.error - a.error)
            - (c.bytes as f64 - a.bytes as f64) * (b.error - a.error)
    };
    let mut convex: Vec<TypeScores> = Vec::new();
    for c in candidates.iter() {
        while convex.len() >= 2
            && cross(&convex[convex.len() - 2], convex.last().unwrap(), c) <= EPSILON
        {
            convex.pop();
        }
        convex.push(c.clone());
    }
    *candidates = convex;
}

/// Build the CNIF scores used to scale candidate errors.
fn build_cnif(
    tensors: &[QTensor],
    metadata: &[TensorMetadata],
    qs: &QuantizeState,
    values: Option<&SideData>,
    activations: Option<&SideData>,
    statistics: Option<&SideData>,
    stats_schema: Option<&[String]>,
    mapped: &std::collections::BTreeMap<i32, String>,
) -> HashMap<String, f32> {
    let Some(stats) = statistics.filter(|s| !s.is_empty()) else {
        return HashMap::new();
    };
    let mut stat_pos = [0usize; 5];
    let mut min_stats = 0usize;
    let mut use_cnif = true;
    for (i, name) in STAT_NAMES.iter().enumerate() {
        let n = stats_schema.map(|s| s.len()).unwrap_or(0);
        let mut p = 0;
        while p < n && stats_schema.unwrap()[p] != *name {
            p += 1;
        }
        if p == n {
            crate::llama_log_warn!("target_bpw_type: statistics have no {} metric, skipping CNIF; model quality may be compromised\n", name);
            use_cnif = false;
        }
        stat_pos[i] = p;
        min_stats = min_stats.max(p + 1);
    }
    if !use_cnif {
        return HashMap::new();
    }

    let corr_error = |x: f32| -> f32 {
        if !x.is_finite() {
            return 0.0;
        }
        1.0 - x.clamp(-1.0, 1.0).abs()
    };
    let mut mse_scores = Vec::new();
    let mut wce_scores = Vec::new();
    for t in tensors.iter() {
        if !can_quantize(qs, &qs.params, t) {
            continue;
        }
        let name = remap_imatrix(&t.name, mapped).unwrap_or_else(|_| t.name.clone());
        let Some(ts) = stats.get(&name).filter(|v| v.len() >= min_stats) else {
            continue;
        };
        if tensor_name_match_token_embd(&t.name) || tensor_name_match_output_weight(&t.name) {
            continue;
        }
        let h_norm = if ts[stat_pos[1]].is_finite() {
            ts[stat_pos[1]].clamp(0.0, 100.0)
        } else {
            100.0
        };
        let concentration = 1.0 - h_norm / 100.0;
        let energy = if ts[stat_pos[0]].is_finite() {
            ts[stat_pos[0]].max(0.0)
        } else {
            0.0
        };
        let l2_dist = if ts[stat_pos[2]].is_finite() {
            ts[stat_pos[2]].max(0.0)
        } else {
            0.0
        };
        let rel_l2 = l2_dist / (energy.sqrt() + EPSILON as f32);
        let density = rel_l2 / (1.0 + rel_l2);
        let fragility = 0.5 * (corr_error(ts[stat_pos[3]]) + corr_error(ts[stat_pos[4]]));
        let n_per_row = t.ne[0] as usize;
        let ne2 = if t.ne[2] > 0 { t.ne[2] } else { 1 } as usize;
        let has_vals = has_side_data(values, &name, ne2, n_per_row);
        let has_acts = has_side_data(activations, &name, ne2, n_per_row);
        let use_wce = has_vals && has_acts && is_angle_sensitive(&name);
        let score = if use_wce {
            0.75 * concentration + 0.15 * density + 0.10 * fragility
        } else {
            0.40 * concentration + 0.35 * density + 0.25 * fragility
        };
        let scores = if use_wce {
            &mut wce_scores
        } else {
            &mut mse_scores
        };
        scores.push(TensorScore { name, score });
    }
    let mut out = cnif(&mut mse_scores, 0.40);
    out.extend(cnif(&mut wce_scores, 0.25));
    out
}

/// Process one tensor and build its candidate set.
#[allow(clippy::too_many_arguments)]
fn process_tensor(
    quant: &impl Quantizer,
    qs: &mut QuantizeState,
    params: &QuantizeParams,
    tensors: &[QTensor],
    idx: usize,
    mapped: &std::collections::BTreeMap<i32, String>,
    values: Option<&SideData>,
    activations: Option<&SideData>,
    statistics: Option<&SideData>,
    cnif_scores: &HashMap<String, f32>,
    bpw_data: &HashMap<String, TypeChoice>,
    locked: Option<&HashMap<String, QType>>,
) -> Result<TypeChoice> {
    let t = &tensors[idx];
    let name = t.name.clone();
    let remapped = remap_imatrix(&name, mapped).unwrap_or_else(|_| name.clone());

    if let Some(cached) = bpw_data.get(&name) {
        let mut tc = cached.clone();
        tc.tensor = idx;
        if qs.has_ignored_types {
            tc.candidates
                .retain(|c| !(c.ty != QType::Count && qs.is_ignored(c.ty)));
            if tc.candidates.is_empty() {
                let is_locked = locked.map(|m| m.contains_key(&name)).unwrap_or(false);
                if !is_locked {
                    return Err(Error::new(format!(
                        "target_bpw_type: tensor '{}' has {} quant type candidates left after --target-exclude",
                        name,
                        tc.candidates.len()
                    )));
                }
                let bytes = t.current_nbytes();
                let n = t.nelements().max(1);
                tc.candidates.push(TypeScores {
                    ty: t.ty,
                    bpw: (bytes * 8) as f32 / n as f32,
                    bytes,
                    ..TypeScores::default()
                });
            }
            tc.choice = 0;
        }
        return Ok(tc);
    }

    crate::llama_log_info!(
        "\ttarget_bpw_type: - processing tensor {:45} \t({:12} elements)\n",
        name,
        t.nelements()
    );
    let tensor_nbytes = t.current_nbytes();
    let row_nbytes = t.row_size(t.ty);
    let max_buf_worker = (params.max_buf_size.max(1)) / 1;
    let stream_rows = row_nbytes > 0 && tensor_nbytes > max_buf_worker;

    let n_per_row = t.ne[0];
    let nrows_total = t.ne[1];
    let ne2 = if t.ne[2] > 0 { t.ne[2] } else { 1 };
    let has_acts = activations.is_some();
    let rows_to_sample = sample_count(n_per_row, nrows_total, ne2, has_acts);

    let mut f32_sample = Vec::new();
    let mut rows_sample = vec![0i64; ne2 as usize];
    let row_bytes = if stream_rows {
        row_nbytes
    } else {
        tensor_nbytes
    };
    let mut row_buf = vec![0u8; row_bytes];

    for slice in 0..ne2 {
        let mut rng =
            Mt19937::new((djb2_hash(name.as_bytes(), HASH_MAGIC) ^ (slice as u64)) as u32);
        let limit = nrows_total.max(1).min(rows_to_sample);
        let stride = nrows_total as f64 / limit as f64;
        let offset = if stride > 1.0 {
            (rng.canonical_f64() * ((stride - 1.0) as f64 + 1.0)) as i64
        } else {
            0
        };
        let mut count = 0i64;
        for i in 0..limit {
            let r = offset + (i as f64 * stride).round() as i64;
            if r >= nrows_total {
                break;
            }
            let off = ((slice * nrows_total + r) as usize) * row_nbytes;
            let src = t
                .data
                .get(off..off + row_nbytes)
                .ok_or_else(|| Error::new(format!("tensor {} data out of range", name)))?;
            let begin = f32_sample.len();
            f32_sample.resize(begin + n_per_row as usize, 0.0);
            quant.dequantize_row(t.ty, src, &mut f32_sample[begin..])?;
            count += 1;
        }
        rows_sample[slice as usize] = count;
    }

    let mut smoothed_sample = Vec::new();
    let is_outlier_heavy = remapped.contains("ffn_down");
    if is_outlier_heavy {
        outlier_smoothing(
            &f32_sample,
            &mut smoothed_sample,
            (rows_to_sample * ne2).max(0) as usize,
            n_per_row as usize,
        );
    }

    let mut val_cache = Vec::new();
    let mut act_cache = Vec::new();
    let ne2us = ne2 as usize;
    let npr = n_per_row as usize;
    let val = values.and_then(|map| map.get(&remapped));
    let act = activations.and_then(|map| map.get(&remapped));
    let val_vec = broadcast(val, &mut val_cache, ne2us, npr);
    let act_vec = broadcast(act, &mut act_cache, ne2us, npr);
    let val_metric = val_vec.map(|v| winsorize_values(v, ne2us, npr));
    let use_wce = val_vec.is_some() && act_vec.is_some() && is_angle_sensitive(&remapped);

    let mut total_rows_sampled = 0usize;
    for &r in &rows_sample {
        total_rows_sampled += r.max(0) as usize;
    }
    let mut ref_wce = WceCache::default();
    let mut ref_mse = MseCache::default();
    if use_wce {
        let v = val_metric.as_deref();
        let mut off = 0usize;
        for s in 0..ne2us {
            let rs = rows_sample[s].max(0) as usize;
            if rs == 0 {
                continue;
            }
            let slice = v.and_then(|m| m.get(s * npr..));
            for _ in 0..rs {
                let wx = &f32_sample[off..off + npr];
                let mut norm_x = 0.0f64;
                for (j, &x) in wx.iter().enumerate() {
                    let w = slice.map(|m| (m[j] as f64).max(0.0)).unwrap_or(1.0);
                    norm_x += x as f64 * x as f64 * w;
                }
                ref_wce.row_sq_norm.push(norm_x);
                off += npr;
            }
        }
    } else {
        let has_vals = val_vec.is_some();
        let has_acts = act_vec.is_some();
        let mut off = 0usize;
        for s in 0..ne2us {
            let rs = rows_sample[s].max(0) as usize;
            let val_slice = if has_vals {
                val_metric.as_deref().and_then(|m| m.get(s * npr..))
            } else {
                None
            };
            let act_slice = if has_acts {
                act_vec.and_then(|m| m.get(s * npr..))
            } else {
                None
            };
            for _ in 0..rs {
                let x = &f32_sample[off..off + npr];
                let mut sum = 0.0f64;
                let mut bias_sum = 0.0f64;
                if let (Some(v), Some(a)) = (val_slice, act_slice) {
                    for j in 0..npr {
                        let aj = a[j] as f64;
                        let variance = (v[j] as f64 - aj * aj).max(0.0);
                        sum += variance * x[j] as f64 * x[j] as f64;
                        bias_sum += aj * x[j] as f64;
                    }
                    sum += bias_sum * bias_sum;
                } else if let Some(v) = val_slice {
                    for j in 0..npr {
                        sum += (v[j] as f64).max(0.0) * x[j] as f64 * x[j] as f64;
                    }
                } else {
                    for j in 0..npr {
                        sum += x[j] as f64 * x[j] as f64;
                    }
                }
                ref_mse.row_sq_norm.push(sum);
                off += npr;
            }
        }
    }

    let kappa_factor = if statistics.map(|s| !s.is_empty()).unwrap_or(false) {
        squash_kappa(randomized_svd_condition_number(
            &f32_sample,
            npr,
            total_rows_sampled,
        ))
    } else {
        1.0
    };

    let valid_matrix = val_vec.is_some();
    let mut valid_types = Vec::new();
    let mut max_row_sz = 0usize;
    for &ty in &QUANT_TYPES {
        if is_iq(ty) && !valid_matrix {
            continue;
        }
        if qs.is_ignored(ty) {
            continue;
        }
        let compat = make_compatible(t, ty);
        if qs.is_ignored(compat) || !is_compatible(t, compat) {
            continue;
        }
        valid_types.push(compat);
        max_row_sz = max_row_sz.max(compat.row_size(n_per_row));
    }
    valid_types.sort();
    valid_types.dedup();

    let mut q_buf = Vec::new();
    let mut dq_buf = Vec::new();
    if total_rows_sampled > 0 && max_row_sz > 0 {
        q_buf.reserve(max_row_sz + 256);
        dq_buf.reserve(npr);
    }
    let cnif_score = cnif_scores.get(&remapped).copied().unwrap_or(1.0);
    let scaling_factor = cnif_score * kappa_factor;
    let ip_sensitive = super::target_types::is_inner_product_sensitive(&remapped);
    let ptr_ref_wce = if use_wce && !ref_wce.row_sq_norm.is_empty() {
        Some(&ref_wce)
    } else {
        None
    };
    let ptr_ref_mse = if !use_wce && !ref_mse.row_sq_norm.is_empty() {
        Some(&ref_mse)
    } else {
        None
    };

    let mut evaluations = Vec::new();
    for vt in valid_types {
        if params.aborted() {
            return Err(Error::new("interrupted"));
        }
        let bpw = vt.bits_per_weight() as f32;
        let sample = if is_outlier_heavy && bpw < 3.0 {
            &smoothed_sample
        } else {
            &f32_sample
        };
        let qe: QuantError = compute_quant_error(
            quant,
            t,
            vt,
            sample,
            &rows_sample,
            val_vec,
            act_vec,
            &mut q_buf,
            &mut dq_buf,
            ptr_ref_wce,
            ptr_ref_mse,
            val_metric.as_deref(),
        );
        let mut error = qe.error;
        if ip_sensitive {
            error *= 1.0 + 0.5 * 2.0f64.powf(3.0 - bpw as f64);
        }
        evaluations.push(TypeScores {
            ty: vt,
            bpw: tensor_bpw(t, vt) as f32,
            bytes: tensor_bytes(t, vt),
            error: error * scaling_factor as f64,
            mse: qe.mse,
            wce: qe.wce,
        });
    }

    let mut ch = TypeChoice {
        tensor: idx,
        n_elements: t.nelements(),
        ..TypeChoice::default()
    };
    for ev in evaluations {
        if ev.bytes > 0 {
            ch.candidates.push(ev);
        }
    }
    if ch.candidates.is_empty() {
        let is_locked = locked.map(|m| m.contains_key(&name)).unwrap_or(false);
        if qs.has_ignored_types && !is_locked {
            return Err(Error::new(format!(
                "target_bpw_type: tensor '{}' has {} quant type candidates left after --target-exclude",
                name, ch.candidates.len()
            )));
        }
        let bytes = t.current_nbytes();
        ch.candidates.push(TypeScores {
            ty: t.ty,
            bpw: (bytes * 8) as f32 / ch.n_elements.max(1) as f32,
            bytes,
            ..TypeScores::default()
        });
    }
    simplify_pareto(&mut ch.candidates);
    ch.choice = 0;
    ch.min_bpw = ch.candidates.first().map(|c| c.bpw).unwrap_or(0.0);
    ch.max_bpw = ch.candidates.last().map(|c| c.bpw).unwrap_or(0.0);
    Ok(ch)
}

fn lagrangian_relaxation(
    all: &[TypeChoice],
    mu: f64,
    choices: &mut Vec<i32>,
    bytes: &mut usize,
    cost: &mut f64,
) {
    choices.resize(all.len(), 0);
    *bytes = 0;
    *cost = 0.0;
    for (i, tn) in all.iter().enumerate() {
        let eff_mu = mu / tn.boost as f64;
        let mut best = 0usize;
        for j in 1..tn.candidates.len() {
            let lr = (tn.candidates[j].error - tn.candidates[best].error)
                + eff_mu * (tn.candidates[j].bytes as f64 - tn.candidates[best].bytes as f64) * 8.0;
            if lr < -EPSILON
                || (lr.abs() <= EPSILON && tn.candidates[j].bytes < tn.candidates[best].bytes)
            {
                best = j;
            }
        }
        choices[i] = best as i32;
        *bytes += tn.candidates[best].bytes;
        *cost += tn.candidates[best].error;
    }
}

#[derive(Clone, Copy, Debug)]
struct TensorUpgrade {
    index: usize,
    next_choice: usize,
    score: f64,
}

impl PartialEq for TensorUpgrade {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for TensorUpgrade {}
impl PartialOrd for TensorUpgrade {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TensorUpgrade {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn allocate_remaining(all: &mut [TypeChoice], budget: usize) {
    let mut cb: usize = all
        .iter()
        .map(|tn| tn.candidates[tn.choice as usize].bytes)
        .sum();
    let mut queue = BinaryHeap::new();
    for i in 0..all.len() {
        let tn = &all[i];
        let next = tn.choice as usize + 1;
        if next < tn.candidates.len() {
            let err =
                (tn.candidates[tn.choice as usize].error - tn.candidates[next].error).max(0.0);
            let delta =
                tn.candidates[next].bytes as f64 - tn.candidates[tn.choice as usize].bytes as f64;
            if delta > EPSILON {
                let mut ratio = err / delta;
                ratio *= tn.boost as f64;
                queue.push(TensorUpgrade {
                    index: i,
                    next_choice: next,
                    score: ratio,
                });
            }
        }
    }
    while let Some(top) = queue.pop() {
        let c = all[top.index].choice as usize;
        if c >= top.next_choice {
            continue;
        }
        let delta =
            all[top.index].candidates[top.next_choice].bytes - all[top.index].candidates[c].bytes;
        if cb + delta <= budget {
            cb += delta;
            all[top.index].choice = top.next_choice as i32;
            let tn = &all[top.index];
            let next = tn.choice as usize + 1;
            if next < tn.candidates.len() {
                let err =
                    (tn.candidates[tn.choice as usize].error - tn.candidates[next].error).max(0.0);
                let d = tn.candidates[next].bytes as f64
                    - tn.candidates[tn.choice as usize].bytes as f64;
                if d > EPSILON {
                    let mut ratio = err / d;
                    ratio *= tn.boost as f64;
                    queue.push(TensorUpgrade {
                        index: top.index,
                        next_choice: next,
                        score: ratio,
                    });
                }
            }
        }
    }
}

fn current_bytes(all: &[TypeChoice]) -> usize {
    all.iter()
        .map(|tn| tn.candidates[tn.choice as usize].bytes)
        .sum()
}

fn total_cost(all: &[TypeChoice]) -> f64 {
    all.iter()
        .map(|tn| tn.boost as f64 * tn.candidates[tn.choice as usize].error)
        .sum()
}

fn run_auction(all: &mut [TypeChoice], budget: usize) {
    let min_bytes: usize = all
        .iter()
        .map(|tn| tn.candidates.first().map(|c| c.bytes).unwrap_or(0))
        .sum();
    let max_bytes: usize = all
        .iter()
        .map(|tn| tn.candidates.last().map(|c| c.bytes).unwrap_or(0))
        .sum();
    if budget <= min_bytes {
        for tn in all.iter_mut() {
            tn.choice = 0;
        }
        return;
    }
    if budget >= max_bytes {
        for tn in all.iter_mut() {
            tn.choice = tn.candidates.len() as i32 - 1;
        }
        return;
    }

    let mut mu_lo = 0.0f64;
    let mut mu_hi = 1.0f64;
    let mut ch_lo = Vec::new();
    let mut ch_hi = Vec::new();
    let mut ch_under = Vec::new();
    let mut ch_over = Vec::new();
    let mut bt_lo = 0usize;
    let mut bt_hi = 0usize;
    let mut dummy = 0.0f64;
    lagrangian_relaxation(all, mu_lo, &mut ch_lo, &mut bt_lo, &mut dummy);
    let mut safety = 0;
    loop {
        lagrangian_relaxation(all, mu_hi, &mut ch_hi, &mut bt_hi, &mut dummy);
        if bt_hi <= budget || bt_hi == usize::MAX {
            break;
        }
        mu_hi *= 2.0;
        safety += 1;
        if safety >= 60 {
            break;
        }
    }
    let mut gap_under = LQ_INFINITE;
    let mut gap_over = LQ_INFINITE;
    for _ in 0..40 {
        let mu = 0.5 * (mu_lo + mu_hi);
        let mut ch_mid = Vec::new();
        let mut bt_mid = 0usize;
        let mut cost_mid = 0.0f64;
        lagrangian_relaxation(all, mu, &mut ch_mid, &mut bt_mid, &mut cost_mid);
        let _ = cost_mid;
        let gap = (bt_mid as f64 - budget as f64).abs();
        if bt_mid > budget {
            mu_lo = mu;
            if gap < gap_over {
                gap_over = gap;
                ch_over = ch_mid;
            }
        } else {
            mu_hi = mu;
            if gap < gap_under {
                gap_under = gap;
                ch_under = ch_mid;
            }
        }
    }
    if !ch_under.is_empty() {
        for (i, tn) in all.iter_mut().enumerate() {
            tn.choice = ch_under[i];
        }
    } else if !ch_over.is_empty() {
        for (i, tn) in all.iter_mut().enumerate() {
            tn.choice = ch_over[i];
        }
    } else if bt_hi <= budget && !ch_hi.is_empty() {
        for (i, tn) in all.iter_mut().enumerate() {
            tn.choice = ch_hi[i];
        }
    } else {
        for tn in all.iter_mut() {
            tn.choice = 0;
        }
    }

    allocate_remaining(all, budget);

    let under_choices: Vec<i32> = all.iter().map(|tn| tn.choice).collect();
    let under_cost = total_cost(all);
    let over: Vec<i32> = if ch_over.is_empty() { ch_lo } else { ch_over };
    if over.len() == all.len() {
        for (i, tn) in all.iter_mut().enumerate() {
            tn.choice = over[i];
        }
    }
    let mut cb = current_bytes(all);
    while cb > budget {
        let mut best: Option<usize> = None;
        let mut best_score = LQ_INFINITE;
        for (i, tn) in all.iter().enumerate() {
            let c = tn.choice as usize;
            if c == 0 {
                continue;
            }
            let err =
                tn.boost as f64 * (tn.candidates[c - 1].error - tn.candidates[c].error).max(0.0);
            let freed = tn.candidates[c].bytes - tn.candidates[c - 1].bytes;
            let score = err / freed.min(cb - budget).max(1) as f64;
            if score < best_score {
                best_score = score;
                best = Some(i);
            }
        }
        let Some(i) = best else { break };
        let c = all[i].choice as usize;
        cb -= all[i].candidates[c].bytes - all[i].candidates[c - 1].bytes;
        all[i].choice = c as i32 - 1;
    }
    allocate_remaining(all, budget);
    if current_bytes(all) > budget || total_cost(all) >= under_cost * (1.0 - DOWNWARD_THRESHOLD) {
        for (i, tn) in all.iter_mut().enumerate() {
            tn.choice = under_choices[i];
        }
    }
}

fn build_mix(
    tensors: &[QTensor],
    all: &[TypeChoice],
    pinned: &[TypeChoice],
) -> HashMap<String, QType> {
    let mut optimized: HashMap<&str, &TypeChoice> = HashMap::new();
    for tn in all {
        optimized.insert(&tensors[tn.tensor].name, tn);
    }
    let mut pinned_map: HashMap<&str, &TypeChoice> = HashMap::new();
    for tn in pinned {
        pinned_map.insert(&tensors[tn.tensor].name, tn);
    }
    let mut mix = HashMap::new();
    crate::llama_log_info!("\ttarget_bpw_type: tensor quantization mix:\n");
    for t in tensors {
        let name = t.name.as_str();
        let ptr = pinned_map.get(name).or_else(|| optimized.get(name));
        if let Some(tn) = ptr {
            let choice = &tn.candidates[tn.choice as usize];
            crate::llama_log_info!(
                "\ttarget_bpw_type: {:45} {}\t{:8}, \t{:.4} bpw,\terror: {:.4}{}\n",
                name,
                if tn.boost > 1.0 { "up" } else { "-" },
                choice.ty.name(),
                choice.bpw,
                choice.error,
                if pinned_map.contains_key(name) {
                    " (pinned)"
                } else {
                    ""
                }
            );
            mix.insert(name.to_owned(), choice.ty);
        }
    }
    mix
}

/// Port of `target_bpw_type`.
#[allow(clippy::too_many_arguments)]
pub fn target_bpw_type(
    qs: &mut QuantizeState,
    quant: &impl Quantizer,
    tensors: &[QTensor],
    metadata: &[TensorMetadata],
    mapped: &std::collections::BTreeMap<i32, String>,
    metadata_bytes: &[u8],
    meta_size: usize,
    model_name: &str,
    arch_name: &str,
    values: Option<&SideData>,
    activations: Option<&SideData>,
    statistics: Option<&SideData>,
    stats_schema: Option<&[String]>,
    locked: Option<&HashMap<String, QType>>,
    nthread: usize,
) -> Result<HashMap<String, QType>> {
    let params = qs.params.clone();
    if qs.has_ignored_types {
        if QUANT_TYPES.iter().all(|t| qs.is_ignored(*t)) {
            return Err(Error::new(
                "target_bpw_type: --target-exclude bars every quant type in the selection pool",
            ));
        }
    }

    let id = model_id(
        metadata_bytes,
        values,
        activations,
        statistics,
        stats_schema,
        mapped,
    );
    let checkpoint = checkpoint_path(&params, model_name, arch_name, id);
    let mut bpw_data = if !params.state_file.is_empty() {
        load_state(&checkpoint, id)
    } else {
        HashMap::new()
    };

    let cnif_scores = build_cnif(
        tensors,
        metadata,
        qs,
        values,
        activations,
        statistics,
        stats_schema,
        mapped,
    );
    let mut all_tensors: Vec<TypeChoice> = Vec::new();
    let mut interrupted = false;
    let mut workers = nthread.max(1);
    workers = workers.min(tensors.len().max(1));
    let _ = workers;

    for (i, t) in tensors.iter().enumerate() {
        if params.aborted() {
            interrupted = true;
            break;
        }
        if !can_quantize(qs, &params, t) {
            continue;
        }
        match process_tensor(
            quant,
            qs,
            &params,
            tensors,
            i,
            mapped,
            values,
            activations,
            statistics,
            &cnif_scores,
            &bpw_data,
            locked,
        ) {
            Ok(tc) => all_tensors.push(tc),
            Err(e) if e.message == "interrupted" => {
                interrupted = true;
                break;
            }
            Err(e) => return Err(e),
        }
    }

    if interrupted {
        let reached: std::collections::HashSet<String> = all_tensors
            .iter()
            .map(|tc| tensors[tc.tensor].name.clone())
            .collect();
        for (i, t) in tensors.iter().enumerate() {
            let Some(mut tc) = bpw_data.get(&t.name).cloned() else {
                continue;
            };
            if reached.contains(&t.name) {
                continue;
            }
            tc.tensor = i;
            all_tensors.push(tc);
        }
    }
    if interrupted {
        if params.state_file.is_empty() {
            crate::llama_log_info!("\n\ttarget_bpw_type: interrupted\n");
        } else if qs.has_ignored_types {
            crate::llama_log_warn!("\n\ttarget_bpw_type: interrupted; state checkpoint is not written while --target-exclude is active\n");
        } else {
            crate::llama_log_info!(
                "\n\ttarget_bpw_type: interrupted, saving progress for {} tensors to {}\n",
                all_tensors.len(),
                checkpoint
            );
            save_state(&checkpoint, id, &all_tensors, tensors);
        }
        return Err(Error::new("interrupted"));
    }
    if !params.state_file.is_empty() {
        if qs.has_ignored_types {
            crate::llama_log_warn!("\ttarget_bpw_type: state checkpoint cannot be saved while --target-exclude is active\n");
        } else {
            save_state(&checkpoint, id, &all_tensors, tensors);
        }
    }
    if all_tensors.is_empty() {
        return Ok(HashMap::new());
    }

    let mut n_elements = 0usize;
    let mut nq_bytes = 0usize;
    for t in tensors.iter() {
        n_elements += t.nelements();
        if !can_quantize(qs, &params, t) {
            nq_bytes += t.current_nbytes();
        }
    }
    let _ = metadata;

    let mut pinned_tensors = Vec::new();
    if let Some(locked) = locked.filter(|m| !m.is_empty()) {
        let mut variable = Vec::new();
        for mut tn in all_tensors {
            let name = tensors[tn.tensor].name.clone();
            if let Some(&lt) = locked.get(&name) {
                let mc = qs.tensor_type_fallback(&tensors[tn.tensor], lt)?;
                let mut idx = tn.candidates.iter().position(|c| c.ty == mc);
                if idx.is_none() {
                    tn.candidates.push(TypeScores {
                        ty: mc,
                        bpw: tensor_bpw(&tensors[tn.tensor], mc) as f32,
                        bytes: tensor_bytes(&tensors[tn.tensor], mc),
                        error: f32::NAN as f64,
                        ..TypeScores::default()
                    });
                    idx = Some(tn.candidates.len() - 1);
                }
                tn.choice = idx.unwrap() as i32;
                pinned_tensors.push(tn);
            } else {
                variable.push(tn);
            }
        }
        all_tensors = variable;
    }

    let pinned_bytes: usize = pinned_tensors
        .iter()
        .map(|tc| tc.candidates[tc.choice as usize].bytes)
        .sum();
    let min_total: usize = all_tensors
        .iter()
        .map(|tn| tn.candidates.first().map(|c| c.bytes).unwrap_or(0))
        .sum();
    let max_total: usize = all_tensors
        .iter()
        .map(|tn| tn.candidates.last().map(|c| c.bytes).unwrap_or(0))
        .sum();
    let total_fixed = nq_bytes + pinned_bytes;
    let total_budget = if params.target_size != -1 {
        params.target_size.max(0) as usize
    } else {
        (params.target_bpw as f64 * n_elements.max(1) as f64 / 8.0).round() as usize
    };
    let total_fixed = total_fixed
        + if params.target_size != -1 {
            meta_size
        } else {
            0
        };
    let pinned_min: usize = pinned_tensors
        .iter()
        .map(|tc| tc.candidates.first().map(|c| c.bytes).unwrap_or(0))
        .sum();
    let min_bytes = total_fixed + min_total;
    let unpinned_min = min_bytes - pinned_bytes.min(min_bytes) + pinned_min;
    let clamped = total_budget < min_bytes;
    let size_str = |bytes: usize| -> String {
        if params.target_size != -1 {
            format!("{} bytes", bytes)
        } else {
            format!("{:.4} bpw", bytes as f64 * 8.0 / n_elements.max(1) as f64)
        }
    };
    let requested = if params.target_size != -1 {
        format!("file size {}", total_budget)
    } else {
        format!("bpw {:.4}", params.target_bpw)
    };
    let budget_bytes = if !clamped {
        let mut b = total_budget.saturating_sub(total_fixed);
        if !pinned_tensors.is_empty() && b > max_total {
            crate::llama_log_warn!("\ttarget_bpw_type: target {} is not reachable with the pinned tensors (maximum ~{}); clamping to maximum\n", requested, size_str(total_fixed + max_total));
        }
        b
    } else {
        if total_budget >= unpinned_min {
            return Err(Error::new(format!(
                "target_bpw_type: target {} is not reachable with the pinned tensors (minimum ~{})",
                requested,
                size_str(min_bytes)
            )));
        }
        let pinned_note = if pinned_tensors.is_empty() {
            String::new()
        } else {
            format!(" (~{} with the pinned tensors)", size_str(min_bytes))
        };
        crate::llama_log_warn!("\ttarget_bpw_type: requested {} is smaller than minimum possible model size (~{}); clamping to minimum{}\n", requested, size_str(unpinned_min), pinned_note);
        min_total
    };

    if !pinned_tensors.is_empty() && !all_tensors.is_empty() {
        let pinned_elements: usize = pinned_tensors.iter().map(|tc| tc.n_elements).sum();
        let rest_elements: usize = all_tensors.iter().map(|tn| tn.n_elements).sum();
        let other_fixed = total_fixed.saturating_sub(pinned_bytes);
        let shared_bpw = (8.0 * (total_budget as f64 - other_fixed as f64)
            / (pinned_elements + rest_elements).max(1) as f64)
            .max(0.0);
        let rest_bpw = 8.0 * budget_bytes.min(max_total) as f64 / rest_elements.max(1) as f64;
        crate::llama_log_info!("\ttarget_bpw_type: pinning {} tensors ({:.2}% of weights) leaves {:.4} bpw available for the rest\n",
            pinned_tensors.len(),
            100.0 * pinned_elements as f64 / (pinned_elements + rest_elements).max(1) as f64,
            rest_bpw);
        if !clamped && rest_bpw < BOOST_TAPER && shared_bpw >= BOOST_TAPER {
            crate::llama_log_warn!("\ttarget_bpw_type: remaining bpw is below the {:.1} safety limit; model's quality may be severely compromised\n", BOOST_TAPER);
        }
    }

    if budget_bytes <= min_total {
        for tn in all_tensors.iter_mut() {
            tn.choice = 0;
        }
        return Ok(build_mix(tensors, &all_tensors, &pinned_tensors));
    }
    if budget_bytes >= max_total {
        for tn in all_tensors.iter_mut() {
            tn.choice = tn.candidates.len() as i32 - 1;
        }
        return Ok(build_mix(tensors, &all_tensors, &pinned_tensors));
    }

    let total_elements: usize = all_tensors.iter().map(|t| t.n_elements).sum();
    let global_bpw = 8.0 * budget_bytes as f64 / total_elements.max(1) as f64;
    for tn in all_tensors.iter_mut() {
        let tensor_name = &tensors[tn.tensor].name;
        if tensor_name.contains("ffn_down.weight") {
            if global_bpw < BOOST_TAPER {
                for c in tn.candidates.iter_mut() {
                    c.error *= 1.0 + 2.0f64.powf(3.0 - c.bpw as f64);
                }
            } else {
                tn.boost = FFN_DOWN_BOOST;
            }
        } else if tensor_name == "output.weight" {
            tn.boost = OUTPUT_BOOST;
        } else if tensor_name.contains("attn_v") {
            tn.boost = ATTN_V_BOOST;
        } else if qs.has_tied_embeddings
            && tensor_name == "token_embd.weight"
            && global_bpw >= BOOST_TAPER
        {
            let embd_share = tn.n_elements as f64 / total_elements.max(1) as f64;
            tn.boost = if embd_share <= 0.10 {
                TIED_EMBD_BOOST
            } else {
                (TIED_EMBD_BOOST as f64 * (1.0 - 3.0 * (embd_share - 0.10))).max(1.25) as f32
            };
        }
    }

    run_auction(&mut all_tensors, budget_bytes);

    // Dense ffn_down cap.
    if global_bpw >= BOOST_TAPER {
        let role: Vec<usize> = all_tensors
            .iter()
            .enumerate()
            .filter(|(_, tn)| {
                regex_match(&tensors[tn.tensor].name, r"blk\.[0-9]+\.ffn_down\.weight")
            })
            .map(|(i, _)| i)
            .collect();
        if role.len() >= 8 {
            let mut bpws: Vec<f64> = role
                .iter()
                .map(|&i| all_tensors[i].candidates[all_tensors[i].choice as usize].bpw as f64)
                .collect();
            bpws.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = bpws.len() / 2;
            let median_bpw = if bpws.len() % 2 != 0 {
                bpws[mid]
            } else {
                0.5 * (bpws[mid - 1] + bpws[mid])
            };
            let outliers: Vec<usize> = role
                .iter()
                .copied()
                .filter(|&i| {
                    median_bpw
                        - all_tensors[i].candidates[all_tensors[i].choice as usize].bpw as f64
                        > 2.0
                })
                .collect();
            if !outliers.is_empty() {
                let family_rank = |gt: QType| -> i32 {
                    match gt {
                        QType::Q2K | QType::Q3K | QType::Q4K | QType::Q5K | QType::Q6K => 0,
                        _ if is_iq(gt) => 1,
                        _ => 2,
                    }
                };
                let mut fill = QType::Q8_0;
                let mut fill_bpw = LQ_INFINITE;
                let mut fill_rank = 3;
                for &ty in &QUANT_TYPES {
                    if !ty.is_quantized() || qs.is_ignored(ty) {
                        continue;
                    }
                    let nominal = ty.bits_per_weight();
                    if nominal < median_bpw {
                        continue;
                    }
                    let rank = family_rank(ty);
                    if nominal < fill_bpw || (nominal == fill_bpw && rank < fill_rank) {
                        fill = ty;
                        fill_bpw = nominal;
                        fill_rank = rank;
                    }
                }
                if fill_bpw == LQ_INFINITE && qs.is_ignored(fill) {
                    crate::llama_log_warn!("\ttarget_bpw_type: ffn_down type {} is ignored by --target-exclude, keeping the auto-selected type\n", fill.name());
                } else {
                    let mut locked_bytes = 0usize;
                    let mut remaining = Vec::new();
                    for (i, mut tn) in all_tensors.into_iter().enumerate() {
                        if outliers.binary_search(&i).is_err() {
                            remaining.push(tn);
                            continue;
                        }
                        let mc = make_compatible(&tensors[tn.tensor], fill);
                        if qs.is_ignored(mc) {
                            remaining.push(tn);
                            continue;
                        }
                        let mut idx = tn.candidates.iter().position(|c| c.ty == mc);
                        if idx.is_none() {
                            tn.candidates.push(TypeScores {
                                ty: mc,
                                bpw: tensor_bpw(&tensors[tn.tensor], mc) as f32,
                                bytes: tensor_bytes(&tensors[tn.tensor], mc),
                                error: f32::NAN as f64,
                                ..TypeScores::default()
                            });
                            idx = Some(tn.candidates.len() - 1);
                        }
                        tn.choice = idx.unwrap() as i32;
                        locked_bytes += tn.candidates[idx.unwrap()].bytes;
                        pinned_tensors.push(tn);
                    }
                    all_tensors = remaining;
                    run_auction(&mut all_tensors, budget_bytes.saturating_sub(locked_bytes));
                }
            }
        }
    }

    // Expert down-projection cap.
    if global_bpw < BOOST_TAPER {
        let re_down = r"blk\.[0-9]+\.ffn_down_exps\.weight";
        let re_gate = r"blk\.[0-9]+\.ffn_gate_exps\.weight";
        let re_up = r"blk\.[0-9]+\.ffn_up_exps\.weight";
        let mean_bpw = |pattern: &str, members: &mut Vec<usize>| -> f64 {
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for (i, tn) in all_tensors.iter().enumerate() {
                if !regex_match(&tensors[tn.tensor].name, pattern) {
                    continue;
                }
                num += tn.n_elements as f64 * tn.candidates[tn.choice as usize].bpw as f64;
                den += tn.n_elements as f64;
                members.push(i);
            }
            if den > 0.0 {
                num / den
            } else {
                -1.0
            }
        };
        let mut role = Vec::new();
        let mut siblings = Vec::new();
        let down_bpw = mean_bpw(re_down, &mut role);
        let gate_bpw = mean_bpw(re_gate, &mut siblings);
        let up_bpw = mean_bpw(re_up, &mut siblings);
        let sibling_bpw = gate_bpw.max(up_bpw);
        siblings.sort_unstable();
        if !role.is_empty() && sibling_bpw > 0.0 && down_bpw > sibling_bpw {
            let outliers: Vec<usize> = role
                .iter()
                .copied()
                .filter(|&i| {
                    all_tensors[i].candidates[all_tensors[i].choice as usize].bpw as f64
                        > sibling_bpw
                })
                .collect();
            if !outliers.is_empty() {
                let mut locked_bytes = 0usize;
                let mut remaining = Vec::new();
                for (i, mut tn) in all_tensors.into_iter().enumerate() {
                    if outliers.binary_search(&i).is_err() {
                        if siblings.binary_search(&i).is_ok() {
                            locked_bytes += tn.candidates[tn.choice as usize].bytes;
                            pinned_tensors.push(tn);
                            continue;
                        }
                        if role.binary_search(&i).is_ok() {
                            let chosen = tn.candidates[tn.choice as usize].ty;
                            let kept: Vec<TypeScores> = tn
                                .candidates
                                .iter()
                                .filter(|c| c.bpw as f64 <= sibling_bpw)
                                .cloned()
                                .collect();
                            if let Some(new_choice) = kept.iter().position(|c| c.ty == chosen) {
                                tn.candidates = kept;
                                tn.choice = new_choice as i32;
                            }
                        }
                        remaining.push(tn);
                        continue;
                    }
                    let mut idx: Option<usize> = None;
                    for (j, c) in tn.candidates.iter().enumerate() {
                        if c.bpw as f64 > sibling_bpw {
                            continue;
                        }
                        if idx.is_none() || c.bpw > tn.candidates[idx.unwrap()].bpw {
                            idx = Some(j);
                        }
                    }
                    let Some(j) = idx else {
                        remaining.push(tn);
                        continue;
                    };
                    tn.choice = j as i32;
                    locked_bytes += tn.candidates[j].bytes;
                    pinned_tensors.push(tn);
                }
                all_tensors = remaining;
                run_auction(&mut all_tensors, budget_bytes.saturating_sub(locked_bytes));
            }
        }
    }

    Ok(build_mix(tensors, &all_tensors, &pinned_tensors))
}

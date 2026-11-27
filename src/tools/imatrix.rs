//! Rust port of `tools/imatrix/imatrix.cpp` and `common/imatrix-loader.{h,cpp}`.
//!
//! The pure data path (accumulation, statistics, file formats, table output)
//! is translated directly. Operations that need a live ggml backend or a
//! loaded model are expressed through the [`Runtime`] trait and plain-data
//! stand-ins, so no FFI or external crate is required.

use crate::common::{self, Params, SpeculativeType};
use crate::core::{Error, Result, Token};
use crate::gguf::{
    gguf_add_tensor, gguf_find_key, gguf_find_tensor, gguf_get_arr_n, gguf_get_arr_str,
    gguf_get_arr_type, gguf_get_kv_type, gguf_get_n_tensors, gguf_get_tensor_name,
    gguf_get_val_str, gguf_get_val_u16, gguf_get_val_u32, gguf_init_empty, gguf_init_from_file,
    gguf_set_arr_str, gguf_set_val_str, gguf_set_val_u32, gguf_write_to_file, GgmlContext,
    GgufContext, GgufInitParams, GgufType, GgmlTensor, GgmlType, GGML_MAX_DIMS,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

const LLM_KV_IMATRIX_DATASETS: &str = "imatrix.datasets";
const LLM_KV_IMATRIX_CHUNK_COUNT: &str = "imatrix.chunk_count";
const LLM_KV_IMATRIX_CHUNK_SIZE: &str = "imatrix.chunk_size";
const LLM_KV_IMATRIX_STATS_SCHEMA: &str = "imatrix.stats_schema";
const LLM_KV_IMATRIX_N_LAYER_NEXTN: &str = "imatrix.n_layer_nextn";

const STATS_SCHEMA: [&str; 12] = [
    "sum_sq",
    "mean",
    "elements",
    "std_deviation",
    "skewness",
    "kurtosis",
    "gain",
    "h_norm",
    "l2_dist",
    "cossim",
    "pearson",
    "covariance",
];

const F32_SIZE: usize = std::mem::size_of::<f32>();

fn io_result<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| Error::new(e.to_string()))
}

// ---------------------------------------------------------------------------
// data types
// ---------------------------------------------------------------------------

/// `struct Stats`.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub activations: Vec<f32>,
    pub values: Vec<f32>,
    pub counts: Vec<i64>,
}

/// `struct tensor_statistics`.
#[derive(Clone, Debug)]
pub struct TensorStatistics {
    pub tensor: String,
    pub legacy: bool,
    pub sum: f64,
    pub mean: f32,
    pub elements: i64,
    pub std_deviation: f32,
    pub skewness: f32,
    pub kurtosis: f32,
    pub gain: f32,
    pub entropy: f32,
    pub l2_dist: f32,
    pub cossim: f32,
    pub pearson: f32,
    pub covariance: f32,
    pub cov_sum: f64,
    pub var_c_sum: f64,
    pub var_p_sum: f64,
    pub dot_prod: f64,
    pub norm1_sq: f64,
    pub norm2_sq: f64,
    pub l2_dist_sq: f64,
    pub sum_prev: f64,
    pub elements_prev: i64,
    pub n_features: i64,
}

impl Default for TensorStatistics {
    fn default() -> Self {
        let fnan = f32::NAN;
        Self {
            tensor: String::new(),
            legacy: true,
            sum: 0.0,
            mean: 0.0,
            elements: 0,
            std_deviation: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            gain: fnan,
            entropy: 0.0,
            l2_dist: fnan,
            cossim: fnan,
            pearson: fnan,
            covariance: fnan,
            cov_sum: 0.0,
            var_c_sum: 0.0,
            var_p_sum: 0.0,
            dot_prod: 0.0,
            norm1_sq: 0.0,
            norm2_sq: 0.0,
            l2_dist_sq: 0.0,
            sum_prev: 0.0,
            elements_prev: 0,
            n_features: 0,
        }
    }
}

/// `struct common_imatrix_entry`.
#[derive(Clone, Debug, Default)]
pub struct ImatrixEntry {
    pub sums: Vec<f32>,
    pub activations: Vec<f32>,
    pub counts: Vec<i64>,
    pub stats: Vec<f32>,
}

/// `struct common_imatrix`. Entries use an ordered map like the C++ `std::map`.
#[derive(Clone, Debug, Default)]
pub struct Imatrix {
    pub entries: BTreeMap<String, ImatrixEntry>,
    pub datasets: Vec<String>,
    pub stats_schema: Vec<String>,
    pub chunk_count: i32,
    pub chunk_size: i32,
    pub n_layer_nextn: i32,
    pub is_legacy: bool,
    pub has_metadata: bool,
}

/// `struct results_log_softmax`.
#[derive(Clone, Copy, Debug)]
struct ResultsLogSoftmax {
    log_softmax: f64,
    logit: f32,
    prob: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectOp {
    MulMat,
    MulMatId,
    Other,
}

/// Plain-data view of the `ggml_tensor` arguments the collector callback uses.
/// Byte strides `nb` mirror `ggml_tensor::nb`; `src1_data` is the F32 payload.
#[derive(Clone, Debug)]
pub struct CollectTensor<'a> {
    pub op: CollectOp,
    pub src0_name: &'a str,
    pub src0_ne: [i64; GGML_MAX_DIMS],
    pub src1_ne: [i64; GGML_MAX_DIMS],
    pub src1_nb: [usize; GGML_MAX_DIMS],
    pub src1_is_f32: bool,
    pub src1_data: &'a [f32],
    pub ids_ne: [i64; 2],
    pub ids_nb: [usize; 2],
    pub ids_data: &'a [u8],
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeBatchRow {
    pub id: Token,
    pub pos: i32,
    pub seq: i32,
    pub logits: bool,
    pub embd: Option<Vec<f32>>,
}

/// Plain-data stand-in for `common_batch`.
#[derive(Clone, Debug, Default)]
pub struct RuntimeBatch {
    pub rows: Vec<RuntimeBatchRow>,
}

impl RuntimeBatch {
    pub fn clear(&mut self) {
        self.rows.clear();
    }

    pub fn size(&self) -> usize {
        self.rows.len()
    }

    pub fn add(&mut self, id: Token, pos: i32, seq: i32, logits: bool) -> usize {
        let idx = self.rows.len();
        self.rows.push(RuntimeBatchRow {
            id,
            pos,
            seq,
            logits,
            embd: None,
        });
        idx
    }

    pub fn set_embd_state(&mut self, idx: usize, embd: Vec<f32>) {
        if let Some(row) = self.rows.get_mut(idx) {
            row.embd = Some(embd);
        }
    }
}

/// `struct nextn_model_info`.
#[derive(Clone, Debug)]
struct NextnModelInfo {
    has_layers: bool,
    own_lm_head: Vec<bool>,
}

/// `struct model_file_shape`.
#[derive(Clone, Debug, Default)]
pub struct ModelFileShape {
    pub valid: bool,
    pub is_split: bool,
    pub arch: String,
    pub n_layer_all: u32,
    pub n_embd_out: u32,
    pub nextn_idx: BTreeSet<u32>,
    pub has_trunk: bool,
    pub has_nextn: bool,
    pub n_nextn_layers: u32,
    pub first_nextn: String,
}

impl ModelFileShape {
    pub fn n_trunk(&self) -> u32 {
        self.n_layer_all.saturating_sub(self.n_nextn_layers)
    }
}

/// Plain-data stand-in for `struct nextn_collector`.
#[derive(Clone, Debug, Default)]
pub struct NextnCollector {
    pub n_embd: usize,
    pub own_lm_head: bool,
    pub pending_h: Vec<f32>,
}

/// Backend boundary. A caller with a live llama.cpp runtime implements this;
/// the std-only port contains no FFI implementation of it.
pub trait Runtime {
    fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<Token>>;
    fn vocab_n_tokens(&self) -> usize;
    fn add_bos(&self) -> bool;
    fn add_eos(&self) -> bool;
    fn bos_token(&self) -> Token;
    fn pooling_is_last(&self) -> bool;
    fn clear_memory(&mut self, nextn: bool);
    fn process(&mut self, batch: &RuntimeBatch) -> Result<()>;
    fn logits(&self) -> &[f32];
    fn logits_ith(&self, i: usize) -> &[f32];
    fn nextn_embeddings_ith(&self, i: usize) -> Option<Vec<f32>>;
    fn nextn_decode(&mut self, batch: &RuntimeBatch) -> Result<()>;
    fn synchronize(&mut self);
    fn print_perf(&self);
    fn init(&mut self, params: &Params) -> Result<()>;
    fn is_initialized(&self) -> bool;
    fn n_ctx_train(&self) -> i32;
    fn n_layer(&self) -> i32;
    fn n_layer_nextn(&self) -> i32;
    fn n_embd_out(&self) -> i32;

    fn draft_n_layer(&self) -> i32 {
        self.n_layer()
    }

    fn draft_n_layer_nextn(&self) -> i32 {
        self.n_layer_nextn()
    }

    fn load_draft(&mut self, path: &str, params: &Params) -> Result<()>;
    fn draft_compatible(&self) -> bool;
    fn init_nextn(&mut self, params: &Params, own_lm_head: bool) -> Result<()>;
    fn set_embeddings_nextn(&mut self, enabled: bool, masked: bool);
}

// ---------------------------------------------------------------------------
// pure helpers
// ---------------------------------------------------------------------------

/// Remove any prefix and suffixes: `CUDA0#blk.0.attn_k.weight#0` -> `blk.0.attn_k.weight`.
fn filter_tensor_name(name: &str) -> String {
    match name.find('#') {
        Some(hash) => {
            let p = &name[hash + 1..];
            match p.find('#') {
                Some(q) => p[..q].to_owned(),
                None => p.to_owned(),
            }
        }
        None => name.to_owned(),
    }
}

/// `process_tensor_name`: split a tensor name into its layer and short name.
fn process_tensor_name(input: &str) -> (String, String) {
    let name: Vec<&str> = input.split('.').collect();
    let mut layer = String::new();
    let mut tensor = String::new();

    for (i, item) in name.iter().enumerate() {
        if *item == "blk" && i + 1 < name.len() {
            layer = name[i + 1].to_owned();
            break;
        }
    }

    for i in 0..name.len() {
        if name[i] == "weight" && i > 0 {
            let mut j = 0usize;
            while j < name.len() {
                if name[j] == "blk" {
                    j += if name.len() > 4 { 1 } else { 2 };
                    j += 1;
                    continue;
                }
                if j == i {
                    break;
                }
                if !tensor.is_empty() {
                    tensor.push('.');
                }
                tensor.push_str(name[j]);
                j += 1;
            }
            break;
        }
    }

    if tensor.is_empty() {
        tensor = input.to_owned();
    }
    if layer.is_empty() {
        layer = "-".to_owned();
    }

    (layer, tensor)
}

/// `compute_tensor_averages`.
fn compute_tensor_averages(stats: &Stats, use_activations: bool) -> Vec<f32> {
    if stats.counts.is_empty() {
        return Vec::new();
    }

    let n_mat = stats.counts.len();
    let src = if use_activations {
        &stats.activations
    } else {
        &stats.values
    };
    let len = src.len();
    if len == 0 || len % n_mat != 0 {
        return Vec::new();
    }

    let row = len / n_mat;
    let mut vec = vec![f32::NAN; len];
    let mut has_valid = false;

    for m in 0..n_mat {
        let c = stats.counts[m] as f32;
        if c <= 0.0 {
            continue;
        }
        has_valid = true;
        let scale = 1.0f32 / c;
        let off = m * row;
        for j in 0..row {
            vec[off + j] = src[off + j] * scale;
        }
    }

    if !has_valid {
        Vec::new()
    } else {
        vec
    }
}

/// `compute_vector_statistics` (single entry).
fn compute_vector_statistics(tstats: &mut Vec<TensorStatistics>, name: &str, e: &Stats) -> bool {
    let fnan = f32::NAN;
    let legacy = e.activations.is_empty();
    let n_mat = e.counts.len();
    let len = if legacy {
        e.values.len()
    } else {
        e.activations.len()
    };

    if n_mat == 0 || len == 0 || len % n_mat != 0 {
        eprintln!(
            "compute_vector_statistics: data size mismatch or empty for tensor {}",
            name
        );
        return false;
    }
    if !legacy && e.values.len() != len {
        eprintln!(
            "compute_vector_statistics: activations/values size mismatch for {}",
            name
        );
        return false;
    }

    let row_size = len / n_mat;
    let mut sum = 0.0f64;
    let mut mean = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_cu_diff = 0.0f64;
    let mut sum_qd_diff = 0.0f64;
    let mut sum_energy = 0.0f64;
    let mut valid_n = 0usize;

    // Mean
    for i in 0..n_mat {
        let c = e.counts[i] as f32;
        if c <= 0.0 {
            continue;
        }
        let inv_c = 1.0f64 / c as f64;
        let off = i * row_size;
        for j in 0..row_size {
            let v_act = if legacy {
                0.0
            } else {
                e.activations[off + j] as f64 * inv_c
            };
            let v_val = e.values[off + j] as f64 * inv_c;
            let v = if legacy { v_val } else { v_act };
            if !v.is_finite() || !v_val.is_finite() {
                continue;
            }
            sum += v_val;
            valid_n += 1;
            let delta = v - mean;
            mean += delta / valid_n as f64;
            if v_val > 0.0 {
                sum_energy += v_val;
            }
        }
    }

    if valid_n == 0 {
        return false;
    }

    let mut entropy = 0.0f64;

    // Std Dev, Skew, Kurtosis, Entropy
    let inv_sum_energy = if sum_energy > 0.0 {
        1.0 / sum_energy
    } else {
        0.0
    };
    let log2_inv = 1.0f64 / 2.0f64.ln();

    for i in 0..n_mat {
        let c = e.counts[i] as f32;
        if c <= 0.0 {
            continue;
        }
        let inv_c = 1.0f64 / c as f64;
        let off = i * row_size;
        for j in 0..row_size {
            let v_act = if legacy {
                0.0
            } else {
                e.activations[off + j] as f64 * inv_c
            };
            let v_val = e.values[off + j] as f64 * inv_c;
            let v = if legacy { v_val } else { v_act };
            if !v.is_finite() || !v_val.is_finite() {
                continue;
            }
            let diff = v - mean;
            sum_sq_diff += diff * diff;
            sum_cu_diff += diff * diff * diff;
            sum_qd_diff += diff * diff * diff * diff;

            if inv_sum_energy > 0.0 {
                let v_energy = e.values[off + j] as f64 * inv_c;
                let p = v_energy.max(0.0) * inv_sum_energy;
                if p > 1e-10 {
                    entropy -= p * p.ln() * log2_inv;
                }
            }
        }
    }

    let variance = if valid_n > 1 {
        sum_sq_diff / valid_n as f64
    } else {
        0.0
    };
    let std_deviation = (variance.max(0.0) as f32).sqrt();
    let mut skewness = 0.0f32;
    let mut kurtosis = 0.0f32;
    if std_deviation > 1e-10f32 {
        let m2 = sum_sq_diff / valid_n as f64;
        skewness = (sum_cu_diff / valid_n as f64 / (m2 * m2.sqrt())) as f32;
        kurtosis = (sum_qd_diff / valid_n as f64 / (m2 * m2) - 3.0) as f32;
    }

    let mut ts = TensorStatistics::default();
    ts.tensor = name.to_owned();
    ts.legacy = legacy;
    ts.sum = sum;
    ts.mean = mean as f32;
    ts.elements = valid_n as i64;
    ts.std_deviation = std_deviation;
    ts.skewness = skewness;
    ts.kurtosis = kurtosis;
    ts.gain = fnan;
    ts.entropy = entropy as f32;
    ts.l2_dist = fnan;
    ts.cossim = fnan;
    ts.pearson = fnan;
    ts.covariance = fnan;
    tstats.push(ts);

    true
}

/// `nextn_layer_start`.
fn nextn_layer_start(tstats: &[TensorStatistics], n_layer_nextn: i32) -> i32 {
    let mut max_blk = -1;
    let mut first = i32::MAX;

    for ts in tstats {
        let (layer, name) = process_tensor_name(&ts.tensor);
        let blk = match layer.parse::<i32>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        max_blk = max_blk.max(blk);
        if name.starts_with("nextn.") {
            first = first.min(blk);
        }
    }

    if n_layer_nextn > 0 && max_blk >= 0 {
        return 0.max(max_blk + 1 - n_layer_nextn);
    }

    first
}

/// `layer_label`.
fn layer_label(blk: i32, nextn_start: i32) -> String {
    if blk < 0 || blk == i32::MAX {
        return "-".to_owned();
    }
    if blk >= nextn_start {
        return format!("mtp{}", blk - nextn_start);
    }
    blk.to_string()
}

/// `compute_tensor_statistics`: compare each tensor with its preceding layer.
fn compute_tensor_statistics(
    tstats: &mut [TensorStatistics],
    mstats: &BTreeMap<String, Stats>,
    nextn_start: i32,
) {
    let fnan = f32::NAN;
    let mut tensor_map: BTreeMap<String, usize> = BTreeMap::new();
    for (i, ts) in tstats.iter().enumerate() {
        tensor_map.insert(ts.tensor.clone(), i);
    }

    for idx in 0..tstats.len() {
        let tensor_name = tstats[idx].tensor.clone();
        let (layer_str, _dummy) = process_tensor_name(&tensor_name);
        let blk = match layer_str.parse::<i32>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if blk <= 0 {
            continue;
        }
        if blk == nextn_start {
            continue;
        }

        let blk_first = if blk > nextn_start { nextn_start } else { 0 };
        let pattern = format!("blk.{}", layer_str);
        let blk_start_pos = match tensor_name.find(&pattern) {
            Some(v) => v,
            None => continue,
        };

        let mut found: Option<usize> = None;
        let mut prev = blk - 1;
        let end = blk_start_pos + layer_str.len() + 4;
        while prev >= blk_first && found.is_none() {
            let replacement = format!("blk.{}", prev);
            let tname = format!(
                "{}{}{}",
                &tensor_name[..blk_start_pos],
                replacement,
                &tensor_name[end..]
            );
            found = tensor_map.get(&tname).copied();
            prev -= 1;
        }

        let prev_idx = match found {
            Some(v) => v,
            None => {
                eprintln!(
                    "compute_tensor_statistics: no preceding-layer tensor for '{}'",
                    tensor_name
                );
                continue;
            }
        };

        let curr_e = match mstats.get(&tensor_name) {
            Some(v) => v,
            None => continue,
        };
        let prev_ts_name = tstats[prev_idx].tensor.clone();
        let prev_e = match mstats.get(&prev_ts_name) {
            Some(v) => v,
            None => continue,
        };

        let use_activations = !tstats[idx].legacy && !tstats[prev_idx].legacy;
        let curr_avg = compute_tensor_averages(curr_e, use_activations);
        let prev_avg = compute_tensor_averages(prev_e, use_activations);
        if curr_avg.is_empty() || curr_avg.len() != prev_avg.len() {
            continue;
        }

        let mut dot_prod = 0.0f64;
        let mut norm1_sq = 0.0f64;
        let mut norm2_sq = 0.0f64;
        let mut l2_dist_sq = 0.0f64;
        let mut sum_c = 0.0f64;
        let mut sum_p = 0.0f64;
        let n = curr_avg.len();
        let mut valid_n = 0usize;

        // Sums for Means
        for i in 0..n {
            let c_val = curr_avg[i] as f64;
            let p_val = prev_avg[i] as f64;
            if c_val.is_finite() && p_val.is_finite() {
                sum_c += c_val;
                sum_p += p_val;
                valid_n += 1;
            }
        }
        if valid_n == 0 {
            continue;
        }
        let mean_c = sum_c / valid_n as f64;
        let mean_p = sum_p / valid_n as f64;

        let mut cov_sum = 0.0f64;
        let mut var_c_sum = 0.0f64;
        let mut var_p_sum = 0.0f64;

        // Metrics
        for i in 0..n {
            let c_val = curr_avg[i] as f64;
            let p_val = prev_avg[i] as f64;
            if !c_val.is_finite() || !p_val.is_finite() {
                continue;
            }
            dot_prod += c_val * p_val;
            norm1_sq += c_val * c_val;
            norm2_sq += p_val * p_val;
            let diff = c_val - p_val;
            l2_dist_sq += diff * diff;

            let dc = c_val - mean_c;
            let dp = p_val - mean_p;
            cov_sum += dc * dp;
            var_c_sum += dc * dc;
            var_p_sum += dp * dp;
        }

        let prev_sum = tstats[prev_idx].sum;
        let prev_elements = tstats[prev_idx].elements;
        let ts = &mut tstats[idx];
        ts.n_features = valid_n as i64;
        ts.dot_prod = dot_prod;
        ts.norm1_sq = norm1_sq;
        ts.norm2_sq = norm2_sq;
        ts.cov_sum = cov_sum;
        ts.var_c_sum = var_c_sum;
        ts.var_p_sum = var_p_sum;
        ts.l2_dist_sq = l2_dist_sq;
        ts.l2_dist = l2_dist_sq.sqrt() as f32;

        if valid_n > 1 {
            ts.covariance = (cov_sum / valid_n as f64) as f32;
        }

        if norm1_sq > 0.0 && norm2_sq > 0.0 {
            ts.cossim = (dot_prod / (norm1_sq.sqrt() * norm2_sq.sqrt())) as f32;
            ts.cossim = ts.cossim.clamp(-1.0, 1.0);
        } else {
            ts.cossim = if norm1_sq == 0.0 && norm2_sq == 0.0 {
                fnan
            } else {
                0.0
            };
        }

        if var_c_sum > 0.0 && var_p_sum > 0.0 {
            ts.pearson = (cov_sum / (var_c_sum.sqrt() * var_p_sum.sqrt())) as f32;
            ts.pearson = ts.pearson.clamp(-1.0, 1.0);
        } else {
            ts.pearson = if var_c_sum == 0.0 && var_p_sum == 0.0 {
                fnan
            } else {
                0.0
            };
        }

        if prev_sum > 1e-10f32 as f64 {
            let gain = (ts.sum / ts.elements as f64).sqrt()
                / (prev_sum / prev_elements as f64).sqrt();
            ts.gain = gain as f32;
            ts.sum_prev = prev_sum;
            ts.elements_prev = prev_elements;
        } else {
            ts.gain = if ts.sum <= 1e-10f32 as f64 { 1.0 } else { fnan };
        }
    }
}

#[derive(Default)]
struct LayerAggregation {
    sum_dot_prod: f64,
    sum_norm1_sq: f64,
    sum_norm2_sq: f64,
    sum_l2_dist_sq: f64,
    sum_cov: f64,
    sum_var_c: f64,
    sum_var_p: f64,
    sum_energy_curr: f64,
    sum_energy_prev: f64,
    sum_elements_curr: i64,
    sum_elements_prev: i64,
    sum_n_features: i64,
    n_tensors: i32,
}

/// `compute_layer_statistics`.
fn compute_layer_statistics(
    tstats: &[TensorStatistics],
    layer_cossim: &mut BTreeMap<i32, f32>,
    layer_l2_dist: &mut BTreeMap<i32, f32>,
    layer_pearson: &mut BTreeMap<i32, f32>,
    layer_covariance: &mut BTreeMap<i32, f32>,
    layer_gain: &mut BTreeMap<i32, f32>,
) {
    let fnan = f32::NAN;
    let mut laggr: BTreeMap<i32, LayerAggregation> = BTreeMap::new();

    for ts in tstats {
        let (layer_str, _dummy) = process_tensor_name(&ts.tensor);
        let blk = match layer_str.parse::<i32>() {
            Ok(v) => v,
            Err(_) => {
                if layer_str == "-" {
                    -1
                } else {
                    continue;
                }
            }
        };
        if blk <= 0 {
            continue;
        }
        if ts.norm1_sq == 0.0 && ts.norm2_sq == 0.0 && ts.l2_dist_sq == 0.0 {
            continue;
        }

        let entry = laggr.entry(blk).or_default();
        entry.sum_dot_prod += ts.dot_prod;
        entry.sum_norm1_sq += ts.norm1_sq;
        entry.sum_norm2_sq += ts.norm2_sq;
        entry.sum_l2_dist_sq += ts.l2_dist_sq;
        entry.sum_cov += ts.cov_sum;
        entry.sum_var_c += ts.var_c_sum;
        entry.sum_var_p += ts.var_p_sum;
        entry.n_tensors += 1;
        if ts.n_features > 0 {
            entry.sum_n_features += ts.n_features;
        }
        if ts.elements_prev > 0 {
            entry.sum_energy_curr += ts.sum;
            entry.sum_energy_prev += ts.sum_prev;
            entry.sum_elements_curr += ts.elements;
            entry.sum_elements_prev += ts.elements_prev;
        }
    }

    for (layer, agg) in &laggr {
        if agg.n_tensors == 0 {
            continue;
        }

        let mut cossim = 0.0f32;
        if agg.sum_norm1_sq > 0.0 && agg.sum_norm2_sq > 0.0 {
            cossim = (agg.sum_dot_prod
                / (agg.sum_norm1_sq.sqrt() * agg.sum_norm2_sq.sqrt())) as f32;
            cossim = cossim.clamp(-1.0, 1.0);
        } else if agg.sum_norm1_sq == 0.0 && agg.sum_norm2_sq == 0.0 {
            cossim = fnan;
        }

        let mut gain = fnan;
        if agg.sum_elements_curr > 0
            && agg.sum_elements_prev > 0
            && agg.sum_energy_prev > 0.0
        {
            let rms_curr = (agg.sum_energy_curr / agg.sum_elements_curr as f64).sqrt();
            let rms_prev = (agg.sum_energy_prev / agg.sum_elements_prev as f64).sqrt();
            gain = (rms_curr / rms_prev) as f32;
        }

        layer_cossim.insert(*layer, cossim);
        layer_l2_dist.insert(*layer, agg.sum_l2_dist_sq.sqrt() as f32);
        layer_gain.insert(*layer, gain);

        if agg.sum_n_features > 0 {
            layer_covariance.insert(*layer, (agg.sum_cov / agg.sum_n_features as f64) as f32);
        } else {
            layer_covariance.insert(*layer, fnan);
        }

        if agg.sum_var_c > 0.0 && agg.sum_var_p > 0.0 {
            let pearson = (agg.sum_cov / (agg.sum_var_c.sqrt() * agg.sum_var_p.sqrt())) as f32;
            layer_pearson.insert(*layer, pearson.clamp(-1.0, 1.0));
        } else if agg.sum_var_c == 0.0 && agg.sum_var_p == 0.0 {
            layer_pearson.insert(*layer, fnan);
        } else {
            layer_pearson.insert(*layer, 0.0);
        }
    }
}

fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Round to nearest instead of truncating.
fn rows_to_chunks(n_rows: i64, chunk_size: i32) -> i32 {
    let cs = chunk_size as i64;
    ((n_rows + cs / 2) / cs) as i32
}

fn tensor_nelements(t: &GgmlTensor) -> usize {
    let mut n = 1i128;
    for &ne in &t.ne {
        n *= ne as i128;
    }
    if n <= 0 {
        0
    } else {
        n as usize
    }
}

fn tensor_f32_data(t: &GgmlTensor, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for chunk in t.data.chunks_exact(F32_SIZE).take(n) {
        v.push(f32::from_ne_bytes(chunk.try_into().unwrap()));
    }
    v
}

/// `softmax` (kept for parity with the reference; unused by this tool).
#[allow(dead_code)]
fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut probs = vec![0.0f32; logits.len()];
    if logits.is_empty() {
        return probs;
    }

    let mut max_logit = logits[0];
    for &v in logits {
        max_logit = max_logit.max(v);
    }

    let mut sum_exp = 0.0f64;
    for i in 0..logits.len() {
        let logit = logits[i] - max_logit;
        let exp_logit = logit.exp();
        sum_exp += exp_logit as f64;
        probs[i] = exp_logit;
    }
    for p in &mut probs {
        *p = (*p as f64 / sum_exp) as f32;
    }
    probs
}

/// `log_softmax`.
fn log_softmax(n_vocab: usize, logits: &[f32], tok: usize) -> ResultsLogSoftmax {
    let mut max_logit = logits[0];
    for i in 1..n_vocab {
        max_logit = max_logit.max(logits[i]);
    }
    let mut sum_exp = 0.0f64;
    for i in 0..n_vocab {
        sum_exp += (logits[i] - max_logit).exp() as f64;
    }
    ResultsLogSoftmax {
        log_softmax: (logits[tok] - max_logit) as f64 - sum_exp.ln(),
        logit: logits[tok],
        prob: (logits[tok] - max_logit).exp() / sum_exp as f32,
    }
}

#[derive(Debug)]
struct LogitState<'a> {
    counter: usize,
    n_token: usize,
    nll: f64,
    nll2: f64,
    logits: &'a [f32],
    tokens: &'a [Token],
    logit_history: &'a mut [f32],
    prob_history: &'a mut [f32],
}

/// `process_logits` with the same worker/counter scheme as the C++ version.
fn process_logits(
    n_vocab: usize,
    logits: &[f32],
    tokens: &[Token],
    nll: &mut f64,
    nll2: &mut f64,
    logit_history: &mut [f32],
    prob_history: &mut [f32],
    n_threads: usize,
) {
    let n_token = logit_history.len().min(prob_history.len());
    let state = LogitState {
        counter: 0,
        n_token,
        nll: 0.0,
        nll2: 0.0,
        logits,
        tokens,
        logit_history,
        prob_history,
    };
    let shared = Arc::new(Mutex::new(state));

    let compute = |s: &Arc<Mutex<LogitState<'_>>>| {
        let mut local_nll = 0.0f64;
        let mut local_nll2 = 0.0f64;
        loop {
            let mut guard = s.lock().unwrap();
            let i = guard.counter;
            guard.counter += 1;
            if i >= guard.n_token {
                guard.nll += local_nll;
                guard.nll2 += local_nll2;
                break;
            }
            let results = log_softmax(n_vocab, &guard.logits[i * n_vocab..], guard.tokens[i + 1] as usize);
            let v = -results.log_softmax;
            local_nll += v;
            local_nll2 += v * v;
            guard.logit_history[i] = results.logit;
            guard.prob_history[i] = results.prob;
        }
    };

    thread::scope(|scope| {
        for _ in 0..n_threads {
            scope.spawn(|| compute(&shared));
        }
        compute(&shared);
    });

    let finished = Arc::try_unwrap(shared).unwrap().into_inner().unwrap();
    *nll = finished.nll;
    *nll2 = finished.nll2;
}

fn print_usage(program: &str) {
    println!(
        "\nexample usage:\n\n    {} \\\n       -m model.gguf -f some-text.txt [-o imatrix.gguf] [--output-format {{gguf,dat}}] [--no-ppl] \\\n       [--process-output] [--nextn] [--model-draft draft.gguf] [--chunk 123] [--save-frequency 0] \\\n       [--output-frequency 10] [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...] \\\n       [--parse-special] [--show-statistics] [...]\n",
        program
    );
}

// ---------------------------------------------------------------------------
// IMatrixCollector
// ---------------------------------------------------------------------------

pub struct IMatrixCollector {
    stats: BTreeMap<String, Stats>,
    params: Params,
    datasets: Vec<String>,
    last_chunk: i32,
    chunk_size: i32,
    n_layer_nextn: i32,
}

impl Default for IMatrixCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl IMatrixCollector {
    pub fn new() -> Self {
        Self {
            stats: BTreeMap::new(),
            params: Params::default(),
            datasets: Vec::new(),
            last_chunk: 0,
            chunk_size: 0,
            n_layer_nextn: 0,
        }
    }

    pub fn set_params(&mut self, params: Params) {
        self.params = params;
    }

    pub fn mstats(&self) -> &BTreeMap<String, Stats> {
        &self.stats
    }

    pub fn set_n_layer_nextn(&mut self, n_layer_nextn: i32) {
        self.n_layer_nextn = n_layer_nextn;
    }

    pub fn get_n_layer_nextn(&self) -> i32 {
        self.n_layer_nextn
    }

    fn e_chunk_size(&self) -> i32 {
        if self.chunk_size > 0 {
            self.chunk_size
        } else {
            self.params.n_ctx / self.params.n_parallel
        }
    }

    /// `IMatrixCollector::collect_imatrix`.
    pub fn collect_imatrix(&mut self, t: &CollectTensor<'_>, ask: bool) -> bool {
        let wname = filter_tensor_name(t.src0_name);
        let chunk_size = self.params.n_ctx / self.params.n_parallel;

        if ask {
            if t.op == CollectOp::MulMatId {
                return true;
            }
            if t.op != CollectOp::MulMat {
                return false;
            }
            if t.src1_ne[1] < 16 || !t.src1_is_f32 {
                return false;
            }
            let is_output = wname == "output.weight"
                || wname == "token_embd.weight"
                || wname == "nextn.post_projection.weight";
            let is_nextn = wname == "nextn.pre_projection.weight";
            if !(wname.starts_with("blk.")
                || (self.params.process_output && is_output)
                || (self.params.load_mtp && is_nextn))
            {
                return false;
            }
            return true;
        }

        if t.op == CollectOp::MulMatId {
            // ids -> [n_experts_used, n_tokens]
            // src1 -> [cols, n_expert_used, n_tokens]
            let n_as = t.src0_ne[2];
            let n_ids = t.ids_ne[0];
            let n_tokens = t.src1_ne[2];
            let ne0 = t.src1_ne[0];

            {
                let e = self.stats.entry(wname.clone()).or_default();

                if e.counts.len() == 1 && n_as > 1 {
                    let c0 = e.counts[0];
                    e.counts.resize(n_as as usize, c0);
                }
                if e.values.is_empty() {
                    e.activations.resize((ne0 * n_as) as usize, 0.0);
                    e.values.resize((ne0 * n_as) as usize, 0.0);
                    e.counts.resize(n_as as usize, 0);
                } else if e.values.len() != (ne0 * n_as) as usize {
                    eprintln!(
                        "collect_imatrix: inconsistent size for {} ({} vs {})",
                        wname,
                        e.values.len(),
                        ne0 * n_as
                    );
                    return false;
                } else if e.counts.len() != n_as as usize {
                    eprintln!(
                        "collect_imatrix: inconsistent expert count for {} ({} vs {})",
                        wname,
                        e.counts.len(),
                        n_as
                    );
                    return false;
                }

                let has_act = !e.activations.is_empty();
                let mut touched = vec![false; n_as as usize];

                // single pass over the routing ids
                for idx in 0..n_ids {
                    for row in 0..n_tokens {
                        let off = row as usize * t.ids_nb[1] + idx as usize * t.ids_nb[0];
                        let bytes: [u8; 4] = match t.ids_data[off..off + 4].try_into() {
                            Ok(v) => v,
                            Err(_) => return false,
                        };
                        let ex = i32::from_ne_bytes(bytes);
                        if ex < 0 || ex >= n_as as i32 {
                            return false;
                        }
                        let ex = ex as usize;
                        let i11 = idx % t.src1_ne[1];
                        let xoff = (i11 as usize * t.src1_nb[1] + row as usize * t.src1_nb[2])
                            / F32_SIZE;
                        let acc_off = ex * ne0 as usize;

                        e.counts[ex] += 1;
                        touched[ex] = true;
                        for j in 0..ne0 as usize {
                            let x = t.src1_data[xoff + j];
                            e.values[acc_off + j] += x * x;
                            if has_act {
                                e.activations[acc_off + j] += x;
                            }
                        }
                    }
                }

                // check only experts that were routed to and touched
                for ex in 0..n_as as usize {
                    let start = ex * ne0 as usize;
                    if touched[ex] && !all_finite(&e.values[start..start + ne0 as usize]) {
                        eprintln!(
                            "collect_imatrix: non-finite values detected in {}",
                            wname
                        );
                        return false;
                    }
                }
            }

            for ex in 0..n_as as usize {
                let count = self.stats[&wname].counts[ex];
                let n_chunk = rows_to_chunks(count, chunk_size);
                if n_chunk > self.last_chunk {
                    let chunk_step = n_chunk - self.last_chunk;
                    self.last_chunk = n_chunk;
                    if self.params.n_out_freq != 0
                        && (self.last_chunk % self.params.n_out_freq) / chunk_step == 0
                    {
                        if let Err(err) = self.save_imatrix(-1) {
                            eprintln!("collect_imatrix: {}", err);
                        }
                    }
                    if self.params.n_save_freq > 0
                        && (self.last_chunk % self.params.n_save_freq) / chunk_step == 0
                    {
                        if let Err(err) = self.save_imatrix(self.last_chunk) {
                            eprintln!("collect_imatrix: {}", err);
                        }
                    }
                }
            }
        } else {
            let ne0 = t.src1_ne[0];
            let n_mat = t.src0_ne[2] * t.src0_ne[3];

            {
                let e = self.stats.entry(wname.clone()).or_default();

                // use a single count per dense tensor
                if e.counts.len() > 1 {
                    let all_equal = e.counts.iter().all(|&c| c == e.counts[0]);
                    if all_equal {
                        let c0 = e.counts[0];
                        e.counts.resize(1, c0);
                    }
                }
                if e.values.is_empty() {
                    e.activations.resize((ne0 * n_mat) as usize, 0.0);
                    e.values.resize((ne0 * n_mat) as usize, 0.0);
                    e.counts.resize(1, 0);
                } else if e.values.len() != (ne0 * n_mat) as usize {
                    eprintln!(
                        "collect_imatrix: inconsistent size for {} ({} vs {})",
                        wname,
                        e.values.len(),
                        ne0 * n_mat
                    );
                    return false;
                }

                let has_act = !e.activations.is_empty();
                for i3 in 0..t.src1_ne[3] {
                    for i2 in 0..t.src1_ne[2] {
                        let mat_id = (i3 % t.src0_ne[3]) * t.src0_ne[2] + (i2 % t.src0_ne[2]);
                        let acc_off = mat_id as usize * ne0 as usize;
                        for row in 0..t.src1_ne[1] {
                            let xoff = (row as usize * t.src1_nb[1]
                                + i2 as usize * t.src1_nb[2]
                                + i3 as usize * t.src1_nb[3])
                                / F32_SIZE;
                            for j in 0..ne0 as usize {
                                let x = t.src1_data[xoff + j];
                                e.values[acc_off + j] += x * x;
                                if has_act {
                                    e.activations[acc_off + j] += x;
                                }
                            }
                        }
                    }
                }

                if !all_finite(&e.values) {
                    eprintln!("collect_imatrix: non-finite values detected in {}", wname);
                    return false;
                }

                let n_rows = t.src1_ne[1] * t.src1_ne[2] * t.src1_ne[3];
                for c in e.counts.iter_mut() {
                    *c += n_rows / n_mat;
                }
            }

            let n_counts = self.stats[&wname].counts.len();
            for i in 0..n_counts {
                let count = self.stats[&wname].counts[i];
                let n_chunk = rows_to_chunks(count, chunk_size);
                if n_chunk > self.last_chunk {
                    let chunk_step = n_chunk - self.last_chunk;
                    self.last_chunk = n_chunk;
                    if self.params.n_out_freq != 0
                        && (self.last_chunk % self.params.n_out_freq) / chunk_step == 0
                    {
                        if let Err(err) = self.save_imatrix(-1) {
                            eprintln!("collect_imatrix: {}", err);
                        }
                    }
                    if self.params.n_save_freq > 0
                        && (self.last_chunk % self.params.n_save_freq) / chunk_step == 0
                    {
                        if let Err(err) = self.save_imatrix(self.last_chunk) {
                            eprintln!("collect_imatrix: {}", err);
                        }
                    }
                }
            }
        }

        true
    }

    /// `IMatrixCollector::save_imatrix_legacy`.
    pub fn save_imatrix_legacy(&self, ncall: i32) -> Result<()> {
        let mut fname = self.params.out_file.clone();
        if ncall > 0 {
            fname = format!("{}.at_{}", fname, ncall);
        }

        let mut to_store: Vec<String> = Vec::new();
        let mut is_first = true;

        for (name, stat) in &self.stats {
            let n_all = stat.counts.len();
            if n_all == 0 {
                continue;
            }
            let n_zeros = stat.counts.iter().filter(|&&c| c == 0).count();
            if n_zeros != 0 && is_first {
                println!();
                is_first = false;
            }
            if n_zeros == n_all {
                eprintln!(
                    "save_imatrix_legacy: entry '{:>40}' has no data - skipping",
                    name
                );
                continue;
            }
            if n_zeros > 0 {
                eprintln!(
                    "save_imatrix_legacy: entry '{:>40}' has partial data ({:.2}%)",
                    name,
                    100.0f32 * (n_all - n_zeros) as f32 / n_all as f32
                );
            }
            to_store.push(name.clone());
        }

        if to_store.len() < self.stats.len() {
            eprintln!(
                "save_imatrix_legacy: storing only {} out of {} entries",
                to_store.len(),
                self.stats.len()
            );
        }

        to_store.sort();

        let chunk_size = self.e_chunk_size() as i64;
        let mut out = File::create(&fname)
            .map_err(|e| Error::new(format!("failed to open '{}': {}", fname, e)))?;

        io_result(out.write_all(&(to_store.len() as i32).to_ne_bytes()))?;
        for name in &to_store {
            let stat = &self.stats[name];
            io_result(out.write_all(&(name.len() as i32).to_ne_bytes()))?;
            io_result(out.write_all(name.as_bytes()))?;

            let max_count = stat.counts.iter().copied().max().unwrap_or(0);
            let ncall = (max_count + chunk_size - 1) / chunk_size;
            io_result(out.write_all(&(ncall as i32).to_ne_bytes()))?;

            let nval = stat.values.len() as i32;
            let nmat = stat.counts.len() as i32;
            io_result(out.write_all(&nval.to_ne_bytes()))?;
            if nval > 0 && nmat > 0 {
                let row = (nval / nmat) as usize;
                for i in 0..nval as usize {
                    let mut count = stat.counts[i / row] as f32;
                    let mut value = stat.values[i];
                    if count == 0.0 {
                        value = 1.0;
                        count = 1.0;
                    }
                    let out_val = (value / count) * ncall as f32;
                    io_result(out.write_all(&out_val.to_ne_bytes()))?;
                }
            }
        }

        io_result(out.write_all(&self.last_chunk.to_ne_bytes()))?;

        let dataset = if self.params.prompt_file.is_empty() && !self.datasets.is_empty() {
            self.datasets.last().unwrap().clone()
        } else {
            self.params.prompt_file.clone()
        };
        io_result(out.write_all(&(dataset.len() as i32).to_ne_bytes()))?;
        io_result(out.write_all(dataset.as_bytes()))?;

        println!();
        println!(
            "save_imatrix_legacy: stored collected data after {} chunks in {}",
            self.last_chunk, fname
        );
        Ok(())
    }

    fn add_f32_tensor(&self, ctx: &mut GgufContext, name: &str, values: &[f32], ne: &[i64; 2]) {
        let mut tensor = GgmlTensor {
            name: name.to_owned(),
            ne: [1; GGML_MAX_DIMS],
            type_: GgmlType::F32,
            nb: [F32_SIZE; GGML_MAX_DIMS],
            data: Vec::with_capacity(values.len() * F32_SIZE),
        };
        tensor.ne[0] = ne[0];
        tensor.ne[1] = ne[1];
        tensor.nb[1] = tensor.nb[0] * ne[0] as usize;
        tensor.nb[2] = tensor.nb[1] * ne[1] as usize;
        tensor.nb[3] = tensor.nb[2];
        for &v in values {
            tensor.data.extend_from_slice(&v.to_ne_bytes());
        }
        gguf_add_tensor(ctx, &tensor);
    }

    /// `IMatrixCollector::save_imatrix`.
    pub fn save_imatrix(&self, n_chunk: i32) -> Result<()> {
        let mut fname = self.params.out_file.clone();
        let use_legacy_format = self.params.imat_dat;

        if use_legacy_format > 0 {
            return self.save_imatrix_legacy(n_chunk);
        }
        if use_legacy_format == 0 && !common::string_ends_with(&fname, ".gguf") {
            eprintln!("\nsave_imatrix: saving imatrix using GGUF format with a different suffix than .gguf");
            eprintln!("save_imatrix: if you want the previous imatrix format, use --output-format dat");
        }
        if n_chunk > 0 {
            fname = format!("{}.at_{}", fname, n_chunk);
        }

        let mut to_store: Vec<String> = Vec::new();
        let mut is_first = true;
        for (name, stat) in &self.stats {
            let n_all = stat.counts.len();
            let n_zeros = stat.counts.iter().filter(|&&c| c == 0).count();
            if n_zeros != 0 && is_first {
                println!();
                is_first = false;
            }
            if n_zeros > 0 {
                eprintln!(
                    "save_imatrix: entry '{:>40}' has partial data ({:.2}%)",
                    name,
                    100.0f32 * (n_all - n_zeros) as f32 / n_all as f32
                );
            }
            to_store.push(name.clone());
        }
        to_store.sort();

        let mut tstats: Vec<TensorStatistics> = Vec::with_capacity(self.stats.len());
        for (name, stats) in &self.stats {
            compute_vector_statistics(&mut tstats, name, stats);
        }
        if !tstats.is_empty() {
            let nextn_start = nextn_layer_start(&tstats, self.n_layer_nextn);
            compute_tensor_statistics(&mut tstats, &self.stats, nextn_start);
        }
        let mut tstat_index: BTreeMap<String, usize> = BTreeMap::new();
        for (i, ts) in tstats.iter().enumerate() {
            tstat_index.insert(ts.tensor.clone(), i);
        }

        let mut ctx = gguf_init_empty();

        let mut datasets: Vec<&str> = Vec::with_capacity(self.datasets.len() + 1);
        for d in &self.datasets {
            datasets.push(d.as_str());
        }
        if !self.params.prompt_file.is_empty() {
            datasets.push(self.params.prompt_file.as_str());
        }

        gguf_set_val_str(&mut ctx, "general.type", "imatrix");
        gguf_set_arr_str(&mut ctx, LLM_KV_IMATRIX_DATASETS, &datasets, datasets.len());
        gguf_set_val_u32(&mut ctx, LLM_KV_IMATRIX_CHUNK_COUNT, self.last_chunk as u32);
        gguf_set_val_u32(&mut ctx, LLM_KV_IMATRIX_CHUNK_SIZE, self.e_chunk_size() as u32);
        if self.n_layer_nextn > 0 {
            gguf_set_val_u32(
                &mut ctx,
                LLM_KV_IMATRIX_N_LAYER_NEXTN,
                self.n_layer_nextn as u32,
            );
        }
        gguf_set_arr_str(&mut ctx, LLM_KV_IMATRIX_STATS_SCHEMA, &STATS_SCHEMA, STATS_SCHEMA.len());

        for name in &to_store {
            let stat = &self.stats[name];
            let nval = stat.values.len() as i32;
            let nmat = stat.counts.len() as i32;
            if nval > 0 && nmat > 0 {
                self.add_f32_tensor(
                    &mut ctx,
                    &format!("{}.in_sum2", name),
                    &stat.values,
                    &[(nval / nmat) as i64, nmat as i64],
                );
                let counts_f32: Vec<f32> = stat.counts.iter().map(|&c| c as f32).collect();
                self.add_f32_tensor(
                    &mut ctx,
                    &format!("{}.counts", name),
                    &counts_f32,
                    &[1, nmat as i64],
                );
                if !stat.activations.is_empty() {
                    let nact = stat.activations.len() as i32;
                    self.add_f32_tensor(
                        &mut ctx,
                        &format!("{}.in_sum", name),
                        &stat.activations,
                        &[(nact / nmat) as i64, nmat as i64],
                    );
                }
            } else {
                eprintln!("save_imatrix: no data for tensor {}", name);
            }

            let fnan = f32::NAN;
            let mut stats_vals = [0.0f32; 12];
            if let Some(&i) = tstat_index.get(name) {
                let ts = &tstats[i];
                stats_vals[0] = ts.sum as f32;
                stats_vals[1] = ts.mean;
                stats_vals[2] = ts.elements as f32;
                stats_vals[3] = ts.std_deviation;
                stats_vals[4] = ts.skewness;
                stats_vals[5] = ts.kurtosis;
                stats_vals[6] = ts.gain;
                stats_vals[7] = if ts.elements > 1 {
                    100.0f32 * (ts.entropy / (ts.elements as f32).log2())
                } else {
                    fnan
                };
                stats_vals[8] = ts.l2_dist;
                stats_vals[9] = ts.cossim;
                stats_vals[10] = ts.pearson;
                stats_vals[11] = ts.covariance;
            }
            self.add_f32_tensor(&mut ctx, &format!("{}.stats", name), &stats_vals, &[12, 1]);
        }

        if !gguf_write_to_file(&ctx, &fname, false) {
            return Err(Error::new(format!("failed to write imatrix to '{}'", fname)));
        }

        println!();
        println!(
            "save_imatrix: stored collected data after {} chunks in {}",
            self.last_chunk, fname
        );
        Ok(())
    }

    /// `IMatrixCollector::load_imatrix`.
    pub fn load_imatrix(&mut self, file_name: &str) -> Result<()> {
        let mut loaded = Imatrix::default();
        if !common_imatrix_load(file_name, &mut loaded) {
            return Err(Error::new(format!("failed to load imatrix '{}'", file_name)));
        }

        let is_legacy = loaded.is_legacy;
        if loaded.n_layer_nextn > 0 {
            if self.n_layer_nextn == 0 {
                self.n_layer_nextn = loaded.n_layer_nextn;
            } else if self.n_layer_nextn != loaded.n_layer_nextn {
                eprintln!(
                    "load_imatrix: NextN layer count mismatch in {}: {} != {}, using {}",
                    file_name, loaded.n_layer_nextn, self.n_layer_nextn, self.n_layer_nextn
                );
            }
        }

        if !is_legacy && loaded.chunk_size > 0 {
            if self.chunk_size == 0 {
                self.chunk_size = loaded.chunk_size;
            } else if self.chunk_size != loaded.chunk_size {
                eprintln!(
                    "load_imatrix: chunk size mismatch in {}: {} != {}, using {}",
                    file_name, loaded.chunk_size, self.chunk_size, self.chunk_size
                );
            }
        }

        let chunk_size = self.e_chunk_size();

        for (name, entry) in &loaded.entries {
            let e = self.stats.entry(name.clone()).or_default();

            if is_legacy {
                if e.values.is_empty() {
                    e.values.resize(entry.sums.len(), 0.0);
                    e.counts.resize(1, 0);
                }
                for j in 0..entry.sums.len() {
                    e.values[j] += entry.sums[j] * chunk_size as f32;
                }
                for j in 0..e.counts.len() {
                    e.counts[j] += entry.counts[0] * chunk_size as i64;
                }
                e.activations.clear();
            } else {
                let nval = entry.sums.len();
                let ncounts = entry.counts.len();
                let nact = entry.activations.len();
                let first_contribution = e.values.is_empty();

                if e.values.is_empty() {
                    e.values.resize(nval, 0.0);
                } else if nval != e.values.len() {
                    eprintln!(
                        "load_imatrix: mismatched sums size for {}: {} != {}",
                        name,
                        nval,
                        e.values.len()
                    );
                    return Err(Error::new("mismatched sums size"));
                }

                if e.counts.is_empty() {
                    e.counts.resize(ncounts, 0);
                } else if e.counts.len() == 1 && ncounts > 1 {
                    let c0 = e.counts[0];
                    e.counts.resize(ncounts, c0);
                } else if ncounts != e.counts.len() {
                    eprintln!(
                        "load_imatrix: mismatched counts size for {}: {} != {}",
                        name,
                        ncounts,
                        e.counts.len()
                    );
                    return Err(Error::new("mismatched counts size"));
                }

                for j in 0..nval {
                    e.values[j] += entry.sums[j];
                }
                for j in 0..ncounts {
                    e.counts[j] += entry.counts[j];
                }

                if nact > 0 && (first_contribution || !e.activations.is_empty()) {
                    if nact != nval {
                        eprintln!(
                            "load_imatrix: mismatched activations size for {}: {} != {}",
                            name, nact, nval
                        );
                        return Err(Error::new("mismatched activations size"));
                    }
                    if e.activations.is_empty() {
                        e.activations.resize(nact, 0.0);
                    } else if nact != e.activations.len() {
                        eprintln!(
                            "load_imatrix: mismatched activations size for {}: {} != {}",
                            name,
                            nact,
                            e.activations.len()
                        );
                        return Err(Error::new("mismatched activations size"));
                    }
                    for j in 0..nact {
                        e.activations[j] += entry.activations[j];
                    }
                } else {
                    e.activations.clear();
                }
            }
        }

        self.datasets.extend(loaded.datasets.iter().cloned());

        let max_count = self
            .stats
            .values()
            .flat_map(|s| s.counts.iter().copied())
            .max()
            .unwrap_or(0);
        self.last_chunk = rows_to_chunks(max_count, chunk_size);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// imatrix-loader
// ---------------------------------------------------------------------------

fn read_i32(file: &mut File) -> Option<i32> {
    let mut buf = [0u8; 4];
    if file.read_exact(&mut buf).is_err() {
        return None;
    }
    Some(i32::from_ne_bytes(buf))
}

/// `common_imatrix_load_legacy`.
fn common_imatrix_load_legacy(fname: &str, imatrix: &mut Imatrix) -> bool {
    let mut file = match File::open(fname) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("common_imatrix_load_legacy: failed to open {}", fname);
            return false;
        }
    };

    let n_entries = match read_i32(&mut file) {
        Some(v) => v,
        None => {
            eprintln!("common_imatrix_load_legacy: no data in file {}", fname);
            return false;
        }
    };
    if n_entries < 1 {
        eprintln!("common_imatrix_load_legacy: no data in file {}", fname);
        return false;
    }

    for i in 0..n_entries {
        let len = match read_i32(&mut file) {
            Some(v) => v,
            None => {
                eprintln!(
                    "common_imatrix_load_legacy: failed reading name for entry {} from {}",
                    i + 1,
                    fname
                );
                return false;
            }
        };
        let mut name_bytes = vec![0u8; len.max(0) as usize];
        if file.read_exact(&mut name_bytes).is_err() {
            eprintln!(
                "common_imatrix_load_legacy: failed reading name for entry {} from {}",
                i + 1,
                fname
            );
            return false;
        }
        if len < 0 {
            return false;
        }
        let name = String::from_utf8_lossy(&name_bytes).into_owned();

        let ncall = read_i32(&mut file).unwrap_or(0);
        let nval = match read_i32(&mut file) {
            Some(v) => v,
            None => 0,
        };
        if nval < 1 {
            eprintln!(
                "common_imatrix_load_legacy: failed reading number of values for entry {}",
                i
            );
            return false;
        }

        let mut raw = vec![0u8; nval as usize * F32_SIZE];
        if file.read_exact(&mut raw).is_err() {
            eprintln!(
                "common_imatrix_load_legacy: failed reading data for entry {}",
                i
            );
            return false;
        }
        let sums: Vec<f32> = raw
            .chunks_exact(F32_SIZE)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();

        let e = imatrix.entries.entry(name).or_default();
        e.sums = sums;
        e.counts = vec![ncall as i64];
    }

    // the trailing data (chunk count + dataset name) is optional
    let has_more = {
        let mut b = [0u8; 1];
        match file.read(&mut b) {
            Ok(0) => false,
            Ok(_) => true,
            Err(_) => false,
        }
    };
    if has_more {
        let _ = file.seek(SeekFrom::Current(-1));
        if let Some(n_calls) = read_i32(&mut file) {
            imatrix.chunk_count = n_calls;
            if let Some(len) = read_i32(&mut file) {
                if len > 0 {
                    let mut dataset = vec![0u8; len as usize];
                    if file.read_exact(&mut dataset).is_ok() {
                        imatrix
                            .datasets
                            .push(String::from_utf8_lossy(&dataset).into_owned());
                    }
                }
            }
        }
    }

    imatrix.chunk_size = 0;
    imatrix.is_legacy = true;
    true
}

#[derive(Default)]
struct SumTensors {
    in_sum: Option<i64>,
    in_sum2: Option<i64>,
    counts: Option<i64>,
    stats: Option<i64>,
}

/// `common_imatrix_load`.
fn common_imatrix_load(fname: &str, imatrix: &mut Imatrix) -> bool {
    let mut params = GgufInitParams {
        no_alloc: false,
        ctx: Some(GgmlContext::default()),
    };
    let ctx = match gguf_init_from_file(fname, &mut params) {
        Ok(c) => c,
        Err(_) => return common_imatrix_load_legacy(fname, imatrix),
    };

    let n_entries = gguf_get_n_tensors(&ctx);
    if n_entries < 1 {
        eprintln!("common_imatrix_load: no data in file {}", fname);
        return false;
    }

    let datasets_key = gguf_find_key(&ctx, LLM_KV_IMATRIX_DATASETS);
    let chunk_count_key = gguf_find_key(&ctx, LLM_KV_IMATRIX_CHUNK_COUNT);
    let chunk_size_key = gguf_find_key(&ctx, LLM_KV_IMATRIX_CHUNK_SIZE);
    let nextn_key = gguf_find_key(&ctx, LLM_KV_IMATRIX_N_LAYER_NEXTN);

    if datasets_key != -1
        && gguf_get_kv_type(&ctx, datasets_key) == GgufType::Array
        && gguf_get_arr_type(&ctx, datasets_key) == GgufType::String
    {
        let n = gguf_get_arr_n(&ctx, datasets_key);
        for i in 0..n {
            imatrix
                .datasets
                .push(gguf_get_arr_str(&ctx, datasets_key, i).to_owned());
        }
    }

    imatrix.has_metadata = datasets_key != -1 && chunk_count_key != -1 && chunk_size_key != -1;
    imatrix.chunk_count = if chunk_count_key != -1 {
        gguf_get_val_u32(&ctx, chunk_count_key) as i32
    } else {
        0
    };
    imatrix.chunk_size = if chunk_size_key != -1 {
        gguf_get_val_u32(&ctx, chunk_size_key) as i32
    } else {
        0
    };
    imatrix.n_layer_nextn = if nextn_key != -1 {
        gguf_get_val_u32(&ctx, nextn_key) as i32
    } else {
        0
    };

    let schema_idx = gguf_find_key(&ctx, LLM_KV_IMATRIX_STATS_SCHEMA);
    if schema_idx != -1
        && gguf_get_kv_type(&ctx, schema_idx) == GgufType::Array
        && gguf_get_arr_type(&ctx, schema_idx) == GgufType::String
    {
        let n_schema = gguf_get_arr_n(&ctx, schema_idx);
        imatrix.stats_schema.clear();
        for i in 0..n_schema {
            imatrix
                .stats_schema
                .push(gguf_get_arr_str(&ctx, schema_idx, i).to_owned());
        }
    } else {
        if schema_idx != -1 {
            eprintln!(
                "common_imatrix_load: {} in {} is not an array of strings, using the default order",
                LLM_KV_IMATRIX_STATS_SCHEMA, fname
            );
        }
        imatrix.stats_schema = STATS_SCHEMA.iter().map(|s| (*s).to_owned()).collect();
    }

    let mut sums_counts_for: BTreeMap<String, SumTensors> = BTreeMap::new();
    for i in 0..n_entries {
        let name = gguf_get_tensor_name(&ctx, i).to_owned();
        if name.is_empty() {
            continue;
        }
        let mut base = name.clone();
        let sc = sums_counts_for.entry(name.clone()).or_default();
        if common::string_remove_suffix(&mut base, ".in_sum") {
            sc.in_sum = Some(i);
        } else {
            let mut base = name.clone();
            if common::string_remove_suffix(&mut base, ".in_sum2") {
                sc.in_sum2 = Some(i);
            } else {
                let mut base = name.clone();
                if common::string_remove_suffix(&mut base, ".counts") {
                    sc.counts = Some(i);
                } else {
                    let mut base = name.clone();
                    if common::string_remove_suffix(&mut base, ".stats") {
                        sc.stats = Some(i);
                    }
                }
            }
        }
    }

    let tensors = params.ctx.as_ref().map(|c| c.tensors.as_slice()).unwrap_or(&[]);

    for (name, sc) in &sums_counts_for {
        let (in_sum, in_sum2, counts, stats) = match (sc.in_sum, sc.in_sum2, sc.counts, sc.stats) {
            (a, Some(b), Some(c), d) => (a, b, c, d),
            _ => {
                eprintln!("common_imatrix_load: mismatched sums and counts for {}", name);
                return false;
            }
        };

        let t_sum2 = &ctx.info[in_sum2 as usize].t;
        let t_counts = &ctx.info[counts as usize].t;
        if let Some(i) = in_sum {
            if tensor_nelements(&ctx.info[i as usize].t) != tensor_nelements(t_sum2) {
                eprintln!("common_imatrix_load: mismatched sums and counts for {}", name);
                return false;
            }
        }
        if t_sum2.type_ != GgmlType::F32 || t_counts.type_ != GgmlType::F32 {
            eprintln!("common_imatrix_load: sums and counts for {} must be F32", name);
            return false;
        }

        let nval = tensor_nelements(t_sum2);
        let ncounts = tensor_nelements(t_counts);
        let sum2_idx = gguf_find_tensor(&ctx, &ctx.info[in_sum2 as usize].t.name);
        let count_idx = gguf_find_tensor(&ctx, &t_counts.name);
        if sum2_idx < 0 || count_idx < 0 || sum2_idx as usize >= tensors.len() || count_idx as usize >= tensors.len() {
            eprintln!("common_imatrix_load: missing tensor data for {}", name);
            return false;
        }

        let e = imatrix.entries.entry(name.clone()).or_default();
        e.sums = tensor_f32_data(&tensors[sum2_idx as usize], nval);
        e.counts = tensor_f32_data(&tensors[count_idx as usize], ncounts)
            .iter()
            .map(|&c| c as f64)
            .map(|c| c.round() as i64)
            .collect();

        if let Some(i) = in_sum {
            if tensor_nelements(&ctx.info[i as usize].t) == nval {
                let sum_idx = gguf_find_tensor(&ctx, &ctx.info[i as usize].t.name);
                if sum_idx >= 0 && (sum_idx as usize) < tensors.len() {
                    e.activations = tensor_f32_data(&tensors[sum_idx as usize], nval);
                }
            }
        }

        if let Some(i) = stats {
            let t_stats = &ctx.info[i as usize].t;
            if t_stats.type_ == GgmlType::F32 {
                let stats_idx = gguf_find_tensor(&ctx, &t_stats.name);
                if stats_idx >= 0 && (stats_idx as usize) < tensors.len() {
                    let n_stats = tensor_nelements(t_stats);
                    e.stats = tensor_f32_data(&tensors[stats_idx as usize], n_stats);
                }
            }
        }
    }

    true
}

// ---------------------------------------------------------------------------
// model shape / NextN helpers
// ---------------------------------------------------------------------------

/// `nextn_read_model_info`.
fn nextn_read_model_info(
    model_path: &str,
    n_layer: i32,
    n_heads: i32,
) -> NextnModelInfo {
    let mut info = NextnModelInfo {
        has_layers: false,
        own_lm_head: vec![false; n_heads.max(0) as usize],
    };

    let mut params = GgufInitParams {
        no_alloc: true,
        ctx: None,
    };
    let ctx = match gguf_init_from_file(model_path, &mut params) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("nextn_read_model_info: cannot read '{}'", model_path);
            return info;
        }
    };

    let n_tensors = gguf_get_n_tensors(&ctx);
    for head in 0..n_heads.max(0) {
        let prefix = format!("blk.{}.nextn.", n_layer + head);
        let lm_head = format!("{}shared_head_head.weight", prefix);
        for i in 0..n_tensors {
            let name = gguf_get_tensor_name(&ctx, i);
            if !name.starts_with(&prefix) {
                continue;
            }
            info.has_layers = true;
            if name == lm_head {
                info.own_lm_head[head as usize] = true;
            }
        }
    }

    info
}

/// `model_read_file_shape`.
fn model_read_file_shape(model_path: &str) -> ModelFileShape {
    let mut shape = ModelFileShape::default();
    let mut params = GgufInitParams {
        no_alloc: true,
        ctx: None,
    };
    let ctx = match gguf_init_from_file(model_path, &mut params) {
        Ok(c) => c,
        Err(_) => return shape,
    };
    shape.valid = true;

    let arch_key = gguf_find_key(&ctx, "general.architecture");
    if arch_key >= 0 && gguf_get_kv_type(&ctx, arch_key) == GgufType::String {
        shape.arch = gguf_get_val_str(&ctx, arch_key).to_owned();
        let prefix = format!("{}.", shape.arch);

        let block_key = gguf_find_key(&ctx, &format!("{}block_count", prefix));
        if block_key >= 0 && gguf_get_kv_type(&ctx, block_key) == GgufType::Uint32 {
            shape.n_layer_all = gguf_get_val_u32(&ctx, block_key);
        }

        let nextn_key = gguf_find_key(&ctx, &format!("{}nextn_predict_layers", prefix));
        if nextn_key >= 0 && gguf_get_kv_type(&ctx, nextn_key) == GgufType::Uint32 {
            shape.n_nextn_layers = gguf_get_val_u32(&ctx, nextn_key);
        }

        let embd_out_key = gguf_find_key(&ctx, &format!("{}embedding_length_out", prefix));
        let embd_key = gguf_find_key(&ctx, &format!("{}embedding_length", prefix));
        if embd_out_key >= 0 && gguf_get_kv_type(&ctx, embd_out_key) == GgufType::Uint32 {
            shape.n_embd_out = gguf_get_val_u32(&ctx, embd_out_key);
        } else if embd_key >= 0 && gguf_get_kv_type(&ctx, embd_key) == GgufType::Uint32 {
            shape.n_embd_out = gguf_get_val_u32(&ctx, embd_key);
        }
    }

    let split_key = gguf_find_key(&ctx, "split.count");
    if split_key >= 0 && gguf_get_kv_type(&ctx, split_key) == GgufType::Uint16 {
        shape.is_split = gguf_get_val_u16(&ctx, split_key) > 1;
    }

    let n_tensors = gguf_get_n_tensors(&ctx);
    for i in 0..n_tensors {
        let name = gguf_get_tensor_name(&ctx, i);
        if name.contains(".nextn.") {
            shape.has_nextn = true;
            if shape.first_nextn.is_empty() {
                shape.first_nextn = name.to_owned();
            }
            if let Some(rest) = name.strip_prefix("blk.") {
                if let Some(dot) = rest.find('.') {
                    if dot > 0 && rest[dot + 1..].starts_with("nextn.") {
                        let mut idx = 0u32;
                        let mut ok = true;
                        for c in rest[..dot].bytes() {
                            if !c.is_ascii_digit() {
                                ok = false;
                                break;
                            }
                            idx = idx.saturating_mul(10).saturating_add((c - b'0') as u32);
                        }
                        if ok {
                            shape.nextn_idx.insert(idx);
                        }
                    }
                }
            }
        } else if name.starts_with("blk.0.") {
            shape.has_trunk = true;
        }
    }

    shape
}

impl NextnCollector {
    fn clear_memory(&mut self) {
        self.pending_h.clear();
    }

    /// `nextn_collector::decode`. Hidden states are supplied by the runtime.
    /// Returns the NextN batch for the runtime to evaluate.
    fn decode(&mut self, trunk: &RuntimeBatch, hidden: &[Vec<f32>]) -> Result<RuntimeBatch> {
        let n_last = trunk.size() as i32 - 1;
        let mut batch = RuntimeBatch::default();

        if !self.pending_h.is_empty() {
            let first = trunk.rows.first().ok_or_else(|| Error::new("empty trunk batch"))?;
            let idx = batch.add(first.id, first.pos, 0, self.own_lm_head);
            batch.set_embd_state(idx, self.pending_h.clone());
        }

        let mut h_last: Option<Vec<f32>> = None;
        for i in 0..=n_last {
            let h = hidden
                .get(i as usize)
                .ok_or_else(|| Error::new(format!("no NextN hidden state for row {}", i)))?;
            if i == n_last {
                h_last = Some(h.clone());
                break;
            }
            let next = trunk
                .rows
                .get((i + 1) as usize)
                .ok_or_else(|| Error::new("trunk batch too short"))?;
            let idx = batch.add(next.id, next.pos, 0, self.own_lm_head);
            batch.set_embd_state(idx, h.clone());
        }

        if let Some(h) = h_last {
            self.pending_h = h;
        }
        Ok(batch)
    }
}

/// `compute_imatrix`. Inference is provided by `Runtime`; all surrounding
/// token/chunk/logit processing is translated directly.
fn compute_imatrix(
    ctx: &mut dyn Runtime,
    params: &Params,
    n_ctx: i32,
    mut nextn: Option<&mut NextnCollector>,
) -> Result<()> {
    let add_bos = ctx.add_bos();
    let add_eos = ctx.add_eos();
    if !ctx.pooling_is_last() && add_eos {
        return Err(Error::new("assertion failed: EOS with non-LAST pooling"));
    }

    let tim1 = Instant::now();
    println!("compute_imatrix: tokenizing the input ..");
    let mut tokens = ctx.tokenize(&params.prompt, true, params.parse_special)?;
    let tim2 = Instant::now();
    println!(
        "compute_imatrix: tokenization took {} ms",
        tim2.duration_since(tim1).as_micros() as f64 * 1e-3
    );

    if params.i_chunk > 0 {
        if ((params.i_chunk + 2) * n_ctx) as usize >= tokens.len() {
            return Err(Error::new(format!(
                "there will be not enough tokens left after removing {} chunks",
                params.i_chunk
            )));
        }
        println!(
            "compute_imatrix: removing initial {} chunks ({} tokens)",
            params.i_chunk,
            params.i_chunk * n_ctx
        );
        tokens.drain(..(params.i_chunk * n_ctx) as usize);
    }

    if (tokens.len() as i32) < 2 * n_ctx {
        eprintln!(
            "compute_imatrix: you need at least {} tokens for a context of {} tokens",
            2 * n_ctx,
            n_ctx
        );
        eprintln!(
            "compute_imatrix: the data file you provided tokenizes to only {} tokens",
            tokens.len()
        );
        return Err(Error::new("not enough tokens"));
    }

    let mut logit_history = vec![0.0f32; tokens.len()];
    let mut prob_history = vec![0.0f32; tokens.len()];

    let n_chunk_max = tokens.len() as i32 / n_ctx;
    let n_chunk = if params.n_chunks < 0 {
        n_chunk_max
    } else {
        params.n_chunks.min(n_chunk_max)
    };
    let n_vocab = ctx.vocab_n_tokens();
    let n_batch = params.n_batch;

    let mut count = 0i32;
    let mut nll = 0.0f64;
    let mut nll2 = 0.0f64;

    let num_batches = (n_ctx + n_batch - 1) / n_batch;
    let n_seq = 1.max(n_batch / n_ctx);

    if n_batch >= n_ctx && n_batch % n_ctx != 0 {
        return Err(Error::new("invalid batch size: n_batch must be a multiple of n_ctx"));
    }
    if params.n_ctx != n_seq * n_ctx {
        return Err(Error::new("invalid context size: n_ctx must equal n_seq * n_ctx"));
    }

    let mut batch = RuntimeBatch::default();
    let mut logits: Vec<f32> = Vec::new();
    if params.compute_ppl && num_batches > 1 {
        logits.reserve(n_ctx as usize * n_vocab);
    }

    println!(
        "compute_imatrix: computing over {} chunks, n_ctx={}, batch_size={}, n_seq={}",
        n_chunk, n_ctx, n_batch, n_seq
    );

    let n_threads = thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1))
        .unwrap_or(0);

    let mut i = 0;
    while i < n_chunk {
        let start = i * n_ctx;
        let end = start + n_ctx;
        let n_seq_batch = n_seq.min(n_chunk - i);
        let t_start = Instant::now();

        ctx.clear_memory(nextn.is_some());
        if let Some(nx) = nextn.as_mut() {
            nx.clear_memory();
        }

        for j in 0..num_batches {
            let batch_start = start + j * n_batch;
            let batch_size = (end - batch_start).min(n_batch);
            batch.clear();

            for seq in 0..n_seq_batch {
                let seq_start = (batch_start + seq * n_ctx) as usize;
                let token_org = tokens[seq_start];
                if add_bos && j == 0 {
                    tokens[seq_start] = ctx.bos_token();
                }
                for k in 0..batch_size {
                    batch.add(
                        tokens[seq_start + k as usize],
                        j * n_batch + k,
                        seq,
                        true,
                    );
                }
                tokens[seq_start] = token_org;
            }

            ctx.process(&batch)?;

            if let Some(nx) = nextn.as_mut() {
                let mut hidden = Vec::with_capacity(batch.size());
                for k in 0..batch.size() {
                    hidden.push(
                        ctx.nextn_embeddings_ith(k)
                            .ok_or_else(|| Error::new(format!("no NextN hidden state for row {}", k)))?,
                    );
                }
                let nx_batch = nx.decode(&batch, &hidden)?;
                ctx.nextn_decode(&nx_batch)?;
            }

            if params.compute_ppl && num_batches > 1 {
                let bl = ctx.logits();
                let n = batch_size as usize * n_vocab;
                logits.extend_from_slice(&bl[..n.min(bl.len())]);
            }
        }

        if i == 0 {
            ctx.synchronize();
            let t_total = t_start.elapsed().as_secs_f32();
            print!("compute_imatrix: {:.2} seconds per pass - ETA ", t_total);
            let mut total_seconds = (t_total * n_chunk as f32 / n_seq as f32) as i32;
            if total_seconds >= 60 * 60 {
                print!("{} hours ", total_seconds / (60 * 60));
                total_seconds %= 60 * 60;
            }
            println!("{:.2} minutes", total_seconds as f64 / 60.0);
        }

        if params.compute_ppl {
            let first = n_ctx / 2;
            for seq in 0..n_seq_batch {
                let all_logits: &[f32] = if num_batches > 1 {
                    &logits[first as usize * n_vocab..]
                } else {
                    ctx.logits_ith((seq * n_ctx) as usize)
                };
                let tok_start = (start + seq * n_ctx + first) as usize;
                let n_tok = (n_ctx - 1 - first) as usize;
                process_logits(
                    n_vocab,
                    all_logits,
                    &tokens[tok_start..tok_start + n_tok + 1],
                    &mut nll,
                    &mut nll2,
                    &mut logit_history[tok_start..tok_start + n_tok],
                    &mut prob_history[tok_start..tok_start + n_tok],
                    n_threads,
                );
                count += n_ctx - first - 1;
                print!("[{}]{:.4},", i + seq + 1, (nll / count as f64).exp());
            }
            let _ = std::io::stdout().flush();
            logits.clear();
        }

        i += n_seq;
    }

    println!();

    if params.compute_ppl {
        nll2 /= count as f64;
        nll /= count as f64;
        let ppl = nll.exp();
        nll2 -= nll * nll;
        if nll2 > 0.0 {
            nll2 = (nll2 / (count as f64 - 1.0)).sqrt();
            println!("Final estimate: PPL = {:.4} +/- {:.5}", ppl, nll2 * ppl);
        } else {
            println!("Unexpected negative standard deviation of log(prob)");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// statistics display
// ---------------------------------------------------------------------------

fn label_fmt(s: &str, w: usize) -> String {
    if s.len() <= w {
        s.to_owned()
    } else {
        format!("..{}", &s[s.len() - (w - 2)..])
    }
}

#[derive(Default)]
struct LayerStatsAgg {
    layer_sum: f32,
    n: i64,
}

/// `show_statistics`.
fn show_statistics(params: &Params) -> Result<bool> {
    if params.in_files.is_empty() {
        return Ok(false);
    }

    let mut collector = IMatrixCollector::new();
    collector.set_params(params.clone());
    if collector.load_imatrix(&params.in_files[0]).is_err() {
        return Ok(false);
    }

    let mut ts: Vec<TensorStatistics> = Vec::with_capacity(collector.mstats().len());
    for (name, stats) in collector.mstats() {
        compute_vector_statistics(&mut ts, name, stats);
    }
    if ts.is_empty() {
        return Ok(false);
    }

    let n_legacy = ts.iter().filter(|t| t.legacy).count();
    if n_legacy > 0 && n_legacy < ts.len() {
        eprintln!(
            "show_statistics: {} of {} tensors have no activation data, using the legacy layout",
            n_legacy,
            ts.len()
        );
    }
    let legacy = n_legacy > 0;
    let nextn_start = nextn_layer_start(&ts, collector.get_n_layer_nextn());
    compute_tensor_statistics(&mut ts, collector.mstats(), nextn_start);

    ts.sort_by(|a, b| {
        let (lay_a, name_a) = process_tensor_name(&a.tensor);
        let (lay_b, name_b) = process_tensor_name(&b.tensor);
        let blk_a = match lay_a.parse::<i32>() {
            Ok(v) => v,
            Err(_) => {
                if a.tensor.contains("output") {
                    i32::MAX
                } else {
                    i32::MAX - 1
                }
            }
        };
        let blk_b = match lay_b.parse::<i32>() {
            Ok(v) => v,
            Err(_) => {
                if b.tensor.contains("output") {
                    i32::MAX
                } else {
                    i32::MAX - 1
                }
            }
        };
        blk_a.cmp(&blk_b).then_with(|| name_a.cmp(&name_b))
    });

    let mut ls: BTreeMap<i32, LayerStatsAgg> = BTreeMap::new();

    const W_LAY: usize = 6;
    const W_NAM: usize = 40;
    const SEP: &str = " | ";
    let fnan = f32::NAN;

    println!(
        "\nComputing tensor statistics for {} ({} tensors)",
        params.in_files[0],
        ts.len()
    );

    if legacy {
        println!(
            "\n{:>w$}{}{:<n$}{}{:>10}{:>10}{:>12}{:>12}{:>9}{}{:>17}{:>8}{}{:>10}{:>10}",
            "Layer", SEP, "Tensor", SEP, "Mean", "StdDev", "Skew", "Kurt", "H Norm", SEP,
            "∑ E[A²]", "Gain", SEP, "PCC", "Cov", w = W_LAY, n = W_NAM
        );
        println!("{}", "-".repeat(153));
    } else {
        println!(
            "\n{:>w$}{}{:<n$}{}{:>10}{:>10}{:>12}{:>12}{:>9}{}{:>17}{:>8}{}{:>12}{:>10}{:>10}",
            "Layer", SEP, "Tensor", SEP, "Mean", "StdDev", "Skew", "Kurt", "H Norm", SEP,
            "∑ E[A²]", "Gain", SEP, "L2 Dist", "PCC", "Cov", w = W_LAY, n = W_NAM
        );
        println!("{}", "-".repeat(165));
    }

    for tstat in &ts {
        let (layer, _name) = process_tensor_name(&tstat.tensor);
        let h_norm = if tstat.elements > 1 {
            100.0f32 * (tstat.entropy / (tstat.elements as f32).log2())
        } else {
            fnan
        };

        let blk = match layer.parse::<i32>() {
            Ok(v) => v,
            Err(_) => {
                if tstat.tensor.contains("output") {
                    i32::MAX
                } else {
                    -1
                }
            }
        };
        let label = layer_label(blk, nextn_start);
        let short = label_fmt(&tstat.tensor, W_NAM);

        if legacy {
            println!(
                "{:>w$}{}{:<n$}{}{:10.4}{:10.4}{:12.4}{:12.4}{:8.2}%{}{:14.4}{:8.2}{}{:10.4}{:10.4}",
                label, SEP, short, SEP, tstat.mean, tstat.std_deviation, tstat.skewness,
                tstat.kurtosis, h_norm, SEP, tstat.sum, tstat.gain, SEP, tstat.pearson,
                tstat.covariance, w = W_LAY, n = W_NAM
            );
        } else {
            println!(
                "{:>w$}{}{:<n$}{}{:10.4}{:10.4}{:12.4}{:12.4}{:8.2}%{}{:14.4}{:8.2}{}{:12.4}{:10.4}{:10.4}",
                label, SEP, short, SEP, tstat.mean, tstat.std_deviation, tstat.skewness,
                tstat.kurtosis, h_norm, SEP, tstat.sum, tstat.gain, SEP, tstat.l2_dist,
                tstat.pearson, tstat.covariance, w = W_LAY, n = W_NAM
            );
        }

        let entry = ls.entry(blk).or_default();
        entry.layer_sum += tstat.sum as f32;
        entry.n += tstat.elements;
    }

    let mut layer_cossim: BTreeMap<i32, f32> = BTreeMap::new();
    let mut layer_l2_dist: BTreeMap<i32, f32> = BTreeMap::new();
    let mut layer_pearson: BTreeMap<i32, f32> = BTreeMap::new();
    let mut layer_covariance: BTreeMap<i32, f32> = BTreeMap::new();
    let mut layer_gain: BTreeMap<i32, f32> = BTreeMap::new();
    compute_layer_statistics(
        &ts,
        &mut layer_cossim,
        &mut layer_l2_dist,
        &mut layer_pearson,
        &mut layer_covariance,
        &mut layer_gain,
    );

    let mut layers = 0usize;
    let mut trunk_layers = 0usize;
    let mut min = i32::MAX;
    let mut max = -1;
    for (&layer, stats) in &ls {
        if layer >= 0 && layer < 9999 && stats.n > 0 {
            layers += 1;
            if layer < nextn_start {
                trunk_layers += 1;
                min = min.min(layer);
                max = max.max(layer);
            }
        }
    }

    if trunk_layers > 0 {
        let expected = (max - min + 1) as usize;
        if trunk_layers != expected {
            eprintln!(
                "\nshow_statistics: layer sequence gap detected (found {} layers in range {}-{}, expected {}); layer statistics will not be shown",
                trunk_layers, min, max, expected
            );
            return Ok(false);
        }
    }

    println!(
        "\nComputing layer statistics for {} ({} layers)\n",
        params.in_files[0],
        layers
    );

    if legacy {
        println!(
            "{:>w$}{}{:>17}{:>8}{}{:>9}{:>9}{:>12}",
            "Layer", SEP, "∑ E[A²]", "Gain", SEP, "CosSim", "PCC", "Cov", w = W_LAY
        );
        println!("{}", "-".repeat(64));
    } else {
        println!(
            "{:>w$}{}{:>17}{:>8}{}{:>12}{:>9}{:>9}{:>12}",
            "Layer", SEP, "∑ E[A²]", "Gain", SEP, "L2 Dist", "CosSim", "PCC", "Cov", w = W_LAY
        );
        println!("{}", "-".repeat(76));
    }

    let get_layer_stat = |map: &BTreeMap<i32, f32>, layer: i32| -> f32 {
        map.get(&layer).copied().unwrap_or(fnan)
    };

    for (&layer, stats) in &ls {
        if layer < 0 || stats.n == 0 {
            continue;
        }
        let skip = layer == 0 || layer == i32::MAX;
        let lgn = if skip { fnan } else { get_layer_stat(&layer_gain, layer) };
        let ll2 = if skip { fnan } else { get_layer_stat(&layer_l2_dist, layer) };
        let lcs = if skip { fnan } else { get_layer_stat(&layer_cossim, layer) };
        let lpc = if skip { fnan } else { get_layer_stat(&layer_pearson, layer) };
        let lcv = if skip { fnan } else { get_layer_stat(&layer_covariance, layer) };
        let lyr = layer_label(layer, nextn_start);

        if legacy {
            println!(
                "{:>w$}{}{:14.4}{:8.2}{}{:9.4}{:9.4}{:12.4}",
                lyr, SEP, stats.layer_sum, lgn, SEP, lcs, lpc, lcv, w = W_LAY
            );
        } else {
            println!(
                "{:>w$}{}{:14.4}{:8.2}{}{:12.4}{:9.4}{:9.4}{:12.4}",
                lyr, SEP, stats.layer_sum, lgn, SEP, ll2, lcs, lpc, lcv, w = W_LAY
            );
        }
    }

    println!();
    Ok(true)
}

// ---------------------------------------------------------------------------
// CLI params + main orchestration
// ---------------------------------------------------------------------------

fn parse_i32(value: &str) -> Result<i32> {
    value
        .parse::<i32>()
        .map_err(|_| Error::new(format!("invalid integer argument: '{}'", value)))
}

fn take_value(args: &[String], i: &mut usize, program: &str, name: &str) -> Option<String> {
    *i += 1;
    match args.get(*i) {
        Some(v) => Some(v.clone()),
        None => {
            eprintln!("{}: missing value for {}", program, name);
            None
        }
    }
}

/// Minimal `common_params_parse` covering the options used by this example.
fn common_params_parse(args: &[String], params: &mut Params) -> bool {
    if args.is_empty() {
        return true;
    }
    let program = &args[0];
    let mut i = 1usize;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-h" | "--help" => {
                print_usage(program);
                return false;
            }
            "-m" | "--model" => match take_value(args, &mut i, program, arg) {
                Some(v) => params.model.path = v,
                None => return false,
            },
            "-md" | "--model-draft" => match take_value(args, &mut i, program, arg) {
                Some(v) => params.speculative.draft.mparams.path = v,
                None => return false,
            },
            "-f" | "--file" => match take_value(args, &mut i, program, arg) {
                Some(v) => match std::fs::read_to_string(&v) {
                    Ok(text) => {
                        params.prompt = text;
                        params.prompt_file = v;
                    }
                    Err(err) => {
                        eprintln!("{}: failed to read '{}': {}", program, v, err);
                        return false;
                    }
                },
                None => return false,
            },
            "-p" | "--prompt" => match take_value(args, &mut i, program, arg) {
                Some(v) => params.prompt = v,
                None => return false,
            },
            "-o" | "--output" | "--output-file" => match take_value(args, &mut i, program, arg) {
                Some(v) => params.out_file = v,
                None => return false,
            },
            "--output-format" => match take_value(args, &mut i, program, arg) {
                Some(v) => match v.as_str() {
                    "gguf" => params.imat_dat = 0,
                    "dat" => params.imat_dat = 1,
                    _ => {
                        eprintln!("{}: invalid output format '{}'", program, v);
                        return false;
                    }
                },
                None => return false,
            },
            "--no-ppl" => params.compute_ppl = false,
            "--process-output" => params.process_output = true,
            "--nextn" => params.load_mtp = true,
            "--parse-special" => params.parse_special = true,
            "--show-statistics" => params.show_statistics = true,
            "--in-file" => match take_value(args, &mut i, program, arg) {
                Some(v) => params.in_files.push(v),
                None => return false,
            },
            "--chunk" => match take_value(args, &mut i, program, arg).and_then(|v| parse_i32(&v).ok()) {
                Some(v) => params.n_chunks = v,
                None => return false,
            },
            "--save-frequency" => match take_value(args, &mut i, program, arg).and_then(|v| parse_i32(&v).ok()) {
                Some(v) => params.n_save_freq = v,
                None => return false,
            },
            "--output-frequency" => match take_value(args, &mut i, program, arg).and_then(|v| parse_i32(&v).ok()) {
                Some(v) => params.n_out_freq = v,
                None => return false,
            },
            "-c" | "--ctx-size" => match take_value(args, &mut i, program, arg).and_then(|v| parse_i32(&v).ok()) {
                Some(v) => params.n_ctx = v,
                None => return false,
            },
            "-b" | "--batch-size" => match take_value(args, &mut i, program, arg).and_then(|v| parse_i32(&v).ok()) {
                Some(v) => params.n_batch = v,
                None => return false,
            },
            "-n" | "--n-predict" => match take_value(args, &mut i, program, arg).and_then(|v| parse_i32(&v).ok()) {
                Some(v) => params.n_predict = v,
                None => return false,
            },
            unknown => {
                eprintln!("{}: unknown argument '{}'", program, unknown);
                return false;
            }
        }
        i += 1;
    }
    true
}

/// `main`. The backend is supplied through `runtime`.
pub fn main(args: &[String], runtime: &mut dyn Runtime) -> Result<i32> {
    let mut params = Params::default();
    params.out_file = "imatrix.gguf".into();
    params.n_ctx = 512;
    params.escape = false;

    if !common_params_parse(args, &mut params) {
        return Ok(1);
    }

    let use_draft = !params.speculative.draft.mparams.path.is_empty();
    if use_draft {
        params.load_mtp = true;
        params.speculative.draft.n_max = 0;
    }

    if params.show_statistics {
        if !show_statistics(&params)? {
            return Ok(1);
        }
        return Ok(0);
    }

    let n_ctx = params.n_ctx;
    if n_ctx <= 0 {
        eprintln!("main: imatrix tool requires '--ctx-size' > 0");
        return Ok(1);
    }

    {
        let n_seq = 1.max(params.n_batch / n_ctx);
        let n_kv = n_seq * n_ctx;
        if params.load_mtp && n_seq > 1 {
            eprintln!("main: '--nextn' and '--model-draft' need a single sequence per batch, set '--batch-size' to at most '--ctx-size' ({})", n_ctx);
            return Ok(1);
        }
        params.n_parallel = n_seq;
        params.n_ctx = n_kv;
        params.n_batch = params.n_batch.min(n_kv);
    }

    let mut collector = IMatrixCollector::new();
    collector.set_params(params.clone());

    for in_file in &params.in_files {
        println!("main : loading imatrix from '{}'", in_file);
        if collector.load_imatrix(in_file).is_err() {
            eprintln!("main : failed to load {}", in_file);
            return Ok(1);
        }
    }

    if params.prompt.is_empty() {
        println!("No prompt provided; combining precomputed matrices only.");
        if params.in_files.is_empty() {
            eprintln!("Error: No prompt provided and no precomputed matrices (--in-file) to combine.");
            return Ok(1);
        }
        if params.in_files.len() == 1 {
            println!("main : saving imatrix to '{}'", params.out_file);
        } else {
            println!("main : saving combined imatrix to '{}'", params.out_file);
        }
        collector.save_imatrix(-1)?;
        return Ok(0);
    }

    let shape = model_read_file_shape(&params.model.path);
    if use_draft && shape.n_nextn_layers > 0 {
        eprintln!("main: model already includes NextN layers");
        return Ok(1);
    }
    if use_draft && shape.has_nextn {
        eprintln!(
            "main: the model already includes NextN layers (NextN tensor '{}')",
            shape.first_nextn
        );
        return Ok(1);
    }
    if !use_draft && !shape.has_trunk && (shape.n_nextn_layers > 0 || shape.has_nextn) {
        eprintln!(
            "main: '{}' is a NextN draft, use with --model-draft/-md",
            params.model.path
        );
        return Ok(1);
    }
    if use_draft {
        let path_md = &params.speculative.draft.mparams.path;
        let shape_md = model_read_file_shape(path_md);
        if !shape_md.valid || shape_md.is_split {
            eprintln!("main: '{}' is not a readable single-shard GGUF", path_md);
            return Ok(1);
        }
        if shape.valid && shape_md.arch != shape.arch {
            eprintln!(
                "main: the draft architecture '{}' does not match the target architecture '{}'",
                shape_md.arch, shape.arch
            );
            return Ok(1);
        }
        if shape_md.n_nextn_layers == 0 || shape_md.has_trunk {
            eprintln!(
                "main: '{}' is not a draft (no NextN layers or trunk tensors present)",
                path_md
            );
            return Ok(1);
        }
        if shape_md.n_nextn_layers > 1 {
            eprintln!(
                "main: multi-layer NextN drafts are not supported (found {})",
                shape_md.n_nextn_layers
            );
            return Ok(1);
        }
        if shape_md.n_trunk() == 0 {
            eprintln!("main: NextN layers sharing the trunk's KV cache are not supported");
            return Ok(1);
        }
        if shape.valid && !shape_md.nextn_idx.contains(&shape.n_trunk()) {
            eprintln!(
                "main: no NextN tensors at layer index {} in '{}' found",
                shape.n_trunk(),
                path_md
            );
            return Ok(1);
        }
        if shape.valid && shape_md.n_embd_out != shape.n_embd_out {
            eprintln!(
                "main: the draft output width {} does not match the target's {}",
                shape_md.n_embd_out, shape.n_embd_out
            );
            return Ok(1);
        }
        for t in &params.speculative.types {
            if *t != SpeculativeType::None && *t != SpeculativeType::DraftMtp {
                eprintln!(
                    "main: speculative type '{:?}' is not supported with --model-draft/-md (only 'draft-mtp')",
                    t
                );
                return Ok(1);
            }
        }
    }

    runtime.init(&params)?;
    if !runtime.is_initialized() {
        eprintln!("main : failed to init");
        return Ok(1);
    }

    let n_ctx_train = runtime.n_ctx_train();
    if params.n_ctx > n_ctx_train {
        eprintln!(
            "main: model was trained on only {} context tokens ({} specified)",
            n_ctx_train, params.n_ctx
        );
    }

    if use_draft {
        runtime.load_draft(&params.speculative.draft.mparams.path, &params)?;
        if !runtime.draft_compatible() {
            eprintln!("main: the target and draft vocab are not compatible");
            return Ok(1);
        }
    }

    let mut nextn: Option<NextnCollector> = None;
    if params.load_mtp {
        let n_heads = if use_draft {
            runtime.draft_n_layer_nextn()
        } else {
            runtime.n_layer_nextn()
        };
        let n_trunk_src = if use_draft {
            runtime.draft_n_layer()
        } else {
            runtime.n_layer()
        };
        let n_trunk = runtime.n_layer();
        let path_src = if use_draft {
            params.speculative.draft.mparams.path.as_str()
        } else {
            params.model.path.as_str()
        };
        let flag_src = if use_draft { "'-md'" } else { "'--nextn'" };
        let info = if n_heads > 0 {
            nextn_read_model_info(path_src, n_trunk_src, n_heads)
        } else {
            NextnModelInfo {
                has_layers: false,
                own_lm_head: Vec::new(),
            }
        };

        if n_heads == 0 {
            eprintln!("main: the model has no NextN layers, {} has no effect", flag_src);
        } else if n_trunk_src == 0 {
            eprintln!("main: NextN layers sharing the trunk's KV cache are not supported");
            return Ok(1);
        } else if n_heads > 1 {
            eprintln!("main: multi-layer NextN drafts are not supported (found {})", n_heads);
            return Ok(1);
        } else if !info.has_layers {
            eprintln!("main: no NextN tensor in '{}', {} has no effect", path_src, flag_src);
        } else {
            runtime.init_nextn(&params, info.own_lm_head.get(0).copied().unwrap_or(false))?;
            runtime.set_embeddings_nextn(true, false);
            collector.set_n_layer_nextn(n_heads);
            println!(
                "main: processing {} NextN layer(s) from block {}",
                n_heads, n_trunk
            );
            nextn = Some(NextnCollector {
                n_embd: runtime.n_embd_out() as usize,
                own_lm_head: info.own_lm_head.get(0).copied().unwrap_or(false),
                pending_h: Vec::new(),
            });
        }
    }

    println!();
    println!("{}", common::common_params_get_system_info(&params));

    if compute_imatrix(runtime, &params, n_ctx, nextn.as_mut()).is_err() {
        return Ok(1);
    }

    collector.save_imatrix(-1)?;

    println!();
    runtime.print_perf();

    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_name_processing() {
        assert_eq!(
            process_tensor_name("blk.0.attn_k.weight"),
            ("0".to_owned(), "blk.0.attn_k.weight".to_owned())
        );
        assert_eq!(
            process_tensor_name("blk.0.ffn_up.exps.weight"),
            ("0".to_owned(), "ffn_up.exps".to_owned())
        );
        assert_eq!(
            process_tensor_name("output.weight"),
            ("-".to_owned(), "output".to_owned())
        );
    }

    #[test]
    fn rows_to_chunks_rounds_to_nearest() {
        assert_eq!(rows_to_chunks(25, 10), 3);
        assert_eq!(rows_to_chunks(24, 10), 2);
    }

    #[test]
    fn tensor_averages_scale_by_count() {
        let stats = Stats {
            values: vec![4.0, 8.0],
            ..Default::default()
        };
        let mut expected = stats.clone();
        expected.counts = vec![2];
        assert_eq!(compute_tensor_averages(&expected, false), vec![2.0, 4.0]);
        assert!(compute_tensor_averages(&stats, false).is_empty());
    }
}

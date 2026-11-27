//! Quality metrics for the bpw optimizer: quant error, CNIF and condition number.

use crate::core::Result;

use super::kernels::Quantizer;
use super::target_types::{is_angle_sensitive, EPSILON, LQ_INFINITE};
use super::types::{QTensor, QType};

#[derive(Clone, Debug, Default)]
pub struct QuantError {
    pub error: f64,
    pub mse: f64,
    pub wce: f64,
}

#[derive(Clone, Debug, Default)]
pub struct MseCache {
    pub row_sq_norm: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct WceCache {
    pub row_sq_norm: Vec<f64>,
}

fn trimmed_mean(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    if v.len() < 100 {
        return v.iter().sum::<f64>() / v.len() as f64;
    }
    let k = (v.len() as f64 * 0.01) as usize;
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[k..v.len() - k].iter().sum::<f64>() / (v.len() - 2 * k).max(1) as f64
}

/// `compute_quant_error` from `target_bpw_type`.
#[allow(clippy::too_many_arguments)]
pub fn compute_quant_error(
    quant: &impl Quantizer,
    t: &QTensor,
    quant_type: QType,
    f32_sample: &[f32],
    rows_sample: &[i64],
    values_sample: Option<&[f32]>,
    activations_sample: Option<&[f32]>,
    quantized_buffer: &mut Vec<u8>,
    dequantized_buffer: &mut Vec<f32>,
    ref_wce: Option<&WceCache>,
    ref_mse: Option<&MseCache>,
    values_winsorized: Option<&[f32]>,
) -> QuantError {
    let mut qe = QuantError {
        error: f64::INFINITY,
        ..QuantError::default()
    };
    let n_per_row = t.ne[0].max(0) as usize;
    let ne2 = if t.ne[2] > 0 { t.ne[2] } else { 1 } as usize;
    let sample_rows = if n_per_row > 0 {
        f32_sample.len() / n_per_row
    } else {
        0
    };
    if sample_rows == 0 {
        qe.error = 0.0;
        return qe;
    }
    let row_sz = quant_type.row_size(n_per_row as i64);
    if quantized_buffer.len() < row_sz + 256 {
        quantized_buffer.resize(row_sz + 256, 0);
    }
    if dequantized_buffer.len() < n_per_row {
        dequantized_buffer.resize(n_per_row, 0.0);
    }

    let mut quant_dequant_row = |src: &[f32], dst: &mut [f32], v: Option<&[f32]>| -> Result<()> {
        let size = quant.quantize_row(quant_type, src, quantized_buffer, v)?;
        quant.dequantize_row(quant_type, &quantized_buffer[..size], dst)
    };

    let has_vals = values_sample.is_some();
    let has_acts = activations_sample.is_some();
    let use_wce = has_acts && has_vals && is_angle_sensitive(&t.name);
    let valw = values_winsorized.or(values_sample);

    let mut local_row_sq_norm: Vec<f64> = Vec::new();
    let ptr_row_sq_norm: Option<&Vec<f64>> = if !use_wce {
        if let Some(cache) = ref_mse {
            Some(&cache.row_sq_norm)
        } else {
            let mut off = 0usize;
            for s in 0..ne2 {
                let rs = rows_sample.get(s).copied().unwrap_or(0).max(0) as usize;
                let val = if has_vals {
                    valw.and_then(|v| v.get(s * n_per_row..))
                } else {
                    None
                };
                let act = if has_acts {
                    activations_sample.and_then(|a| a.get(s * n_per_row..))
                } else {
                    None
                };
                for _ in 0..rs {
                    let x = &f32_sample[off..off + n_per_row];
                    let mut sum = 0.0f64;
                    let mut bias_sum = 0.0f64;
                    if let (Some(v), Some(a)) = (val, act) {
                        for j in 0..n_per_row {
                            let act_j = a[j] as f64;
                            let variance = (v[j] as f64 - act_j * act_j).max(0.0);
                            sum += variance * x[j] as f64 * x[j] as f64;
                            bias_sum += act_j * x[j] as f64;
                        }
                        sum += bias_sum * bias_sum;
                    } else if let Some(v) = val {
                        for j in 0..n_per_row {
                            sum += (v[j] as f64).max(0.0) * x[j] as f64 * x[j] as f64;
                        }
                    } else {
                        for j in 0..n_per_row {
                            sum += x[j] as f64 * x[j] as f64;
                        }
                    }
                    local_row_sq_norm.push(sum);
                    off += n_per_row;
                }
            }
            Some(&local_row_sq_norm)
        }
    } else {
        None
    };

    if use_wce {
        let mut total_cos_error = 0.0f64;
        let mut off = 0usize;
        let mut sample_idx = 0usize;
        let cached_norm_x = ref_wce.filter(|c| !c.row_sq_norm.is_empty());
        for s in 0..ne2 {
            let rs = rows_sample.get(s).copied().unwrap_or(0).max(0) as usize;
            if rs == 0 {
                continue;
            }
            let v = valw.and_then(|m| m.get(s * n_per_row..));
            let mut slice_sum = 0.0f64;
            for _ in 0..rs {
                let wx = &f32_sample[off..off + n_per_row];
                if quant_dequant_row(wx, dequantized_buffer, v).is_err() {
                    return qe;
                }
                let wy = &dequantized_buffer[..n_per_row];
                let mut dot = 0.0f64;
                let mut ny = 0.0f64;
                let mut nx = 0.0f64;
                let calc_nx = cached_norm_x.is_none();
                if calc_nx {
                    for j in 0..n_per_row {
                        let w = v.map(|m| (m[j] as f64).max(0.0)).unwrap_or(1.0);
                        let xj = wx[j] as f64;
                        let yj = wy[j] as f64;
                        let yw = yj * w;
                        dot += xj * yw;
                        ny += yj * yw;
                        nx += xj * xj * w;
                    }
                } else {
                    nx = cached_norm_x.unwrap().row_sq_norm[sample_idx];
                    for j in 0..n_per_row {
                        let w = v.map(|m| (m[j] as f64).max(0.0)).unwrap_or(1.0);
                        let yj = wy[j] as f64;
                        let yw = yj * w;
                        dot += wx[j] as f64 * yw;
                        ny += yj * yw;
                    }
                }
                let norm_prod = nx * ny;
                let mut cos_sim = if norm_prod <= EPSILON {
                    if nx <= EPSILON && ny <= EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    dot / norm_prod.sqrt()
                };
                cos_sim = cos_sim.clamp(-1.0, 1.0);
                slice_sum += 1.0 - cos_sim;
                off += n_per_row;
                sample_idx += 1;
            }
            total_cos_error += slice_sum / rs as f64 * t.ne[1] as f64;
        }
        qe.wce = total_cos_error;
        qe.error = qe.wce;
        return qe;
    }

    let mut off = 0usize;
    let mut row_idx = 0usize;
    let mut total_wmse = 0.0f64;
    let norms = ptr_row_sq_norm.expect("MSE reference norms");
    for s in 0..ne2 {
        let rs = rows_sample.get(s).copied().unwrap_or(0).max(0) as usize;
        if rs == 0 {
            continue;
        }
        let val = if has_vals {
            valw.and_then(|v| v.get(s * n_per_row..))
        } else {
            None
        };
        let mval = val;
        let act = if has_acts {
            activations_sample.and_then(|a| a.get(s * n_per_row..))
        } else {
            None
        };
        let mut slice_mse_norm = Vec::with_capacity(rs);
        for _ in 0..rs {
            let x = &f32_sample[off..off + n_per_row];
            if quant_dequant_row(x, dequantized_buffer, val).is_err() {
                return qe;
            }
            let y = &dequantized_buffer[..n_per_row];
            let mut w_err = 0.0f64;
            let mut bias_num = 0.0f64;
            if let (Some(m), Some(a)) = (mval, act) {
                for j in 0..n_per_row {
                    let w = (m[j] as f64).max(0.0);
                    let e = y[j] as f64 - x[j] as f64;
                    w_err += w * e * e;
                    bias_num += a[j] as f64 * e;
                }
                w_err += bias_num * bias_num;
            } else if let Some(m) = mval {
                for j in 0..n_per_row {
                    let w = (m[j] as f64).max(0.0);
                    let e = y[j] as f64 - x[j] as f64;
                    w_err += w * e * e;
                }
            } else {
                for j in 0..n_per_row {
                    let e = y[j] as f64 - x[j] as f64;
                    w_err += e * e;
                }
            }
            let rsn = norms.get(row_idx).copied().unwrap_or(0.0);
            let m_norm = if rsn > EPSILON { w_err / rsn } else { 0.0 };
            slice_mse_norm.push(if m_norm.is_finite() {
                m_norm
            } else {
                LQ_INFINITE
            });
            off += n_per_row;
            row_idx += 1;
        }
        total_wmse += trimmed_mean(&mut slice_mse_norm) * t.ne[1] as f64;
    }
    qe.mse = total_wmse;
    qe.error = total_wmse;
    qe
}

/// Clip rows at four standard deviations to spread outlier energy.
pub fn outlier_smoothing(src: &[f32], dst: &mut Vec<f32>, n_rows: usize, len: usize) {
    dst.resize(src.len(), 0.0);
    for r in 0..n_rows {
        let row = &src[r * len..(r + 1) * len];
        let mut mean_d = 0.0f64;
        let mut m2 = 0.0f64;
        for (i, &v) in row.iter().enumerate() {
            let delta = v as f64 - mean_d;
            mean_d += delta / (i + 1) as f64;
            m2 += delta * (v as f64 - mean_d);
        }
        let mean = mean_d as f32;
        let variance = m2 / len as f64;
        let std_dev = (variance.max(0.0) as f32).sqrt();
        let clip = 4.0f32 * std_dev;
        let out = &mut dst[r * len..(r + 1) * len];
        if clip > 1e-4 {
            for (o, &v) in out.iter_mut().zip(row.iter()) {
                *o = v.clamp(mean - clip, mean + clip);
            }
        } else {
            out.copy_from_slice(row);
        }
    }
}

/// Winsorize positive importance values using a log-space MAD estimate.
pub fn winsorize_values(src: &[f32], ne2: usize, n_per_row: usize) -> Vec<f32> {
    let mut out = src.to_vec();
    for s in 0..ne2 {
        let start = s * n_per_row;
        let mut scratch = Vec::new();
        for &v in &src[start..start + n_per_row] {
            if v > 0.0 {
                scratch.push(v.ln());
            }
        }
        if scratch.len() < 8 || scratch.len() < n_per_row / 2 {
            continue;
        }
        let mid = scratch.len() / 2;
        scratch.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        let med = scratch[mid];
        for d in scratch.iter_mut() {
            *d = (*d - med).abs();
        }
        scratch.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        let mad = scratch[mid];
        if mad <= 1e-6 * med.abs().max(1.0) {
            continue;
        }
        let cap = (med as f64 + 4.0 * 1.4826 * mad as f64).exp() as f32;
        for v in out[start..start + n_per_row].iter_mut() {
            if *v > cap {
                *v = cap;
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct TensorScore {
    pub name: String,
    pub score: f32,
}

fn sort_scores(scores: &mut [TensorScore]) {
    scores.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });
}

fn median(values: &[f32]) -> f32 {
    let mid = values.len() / 2;
    if values.len() % 2 != 0 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    }
}

fn rank(scores: &mut [TensorScore], beta: f32, out: &mut std::collections::HashMap<String, f32>) {
    if scores.is_empty() {
        return;
    }
    sort_scores(scores);
    if scores.len() == 1 {
        out.insert(scores[0].name.clone(), 1.0);
        return;
    }
    let mut i = 0;
    while i < scores.len() {
        let mut j = i + 1;
        while j < scores.len() && (scores[j].score - scores[i].score).abs() <= 1e-6 {
            j += 1;
        }
        let p = 0.5 * (i + j - 1) as f32 / (scores.len() - 1) as f32;
        let scale = (beta * (2.0 * p - 1.0)).exp();
        for k in i..j {
            out.insert(scores[k].name.clone(), scale);
        }
        i = j;
    }
}

/// Centered-Normalized Importance Factor scoring.
pub fn cnif(scores: &mut [TensorScore], beta: f32) -> std::collections::HashMap<String, f32> {
    const MIN_SCORES: usize = 8;
    const EPS: f32 = 1e-6;
    const MAD_FACTOR: f32 = 1.4826;
    let mut out = std::collections::HashMap::new();
    if scores.len() < MIN_SCORES {
        rank(scores, beta, &mut out);
        return out;
    }
    sort_scores(scores);
    let med = median(&scores.iter().map(|s| s.score).collect::<Vec<_>>());
    let mut deviations: Vec<f32> = scores.iter().map(|s| (s.score - med).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = median(&deviations);
    if mad <= EPS {
        rank(scores, beta, &mut out);
        return out;
    }
    let inv_scale = 1.0 / (MAD_FACTOR * mad + EPS);
    for s in scores.iter() {
        let z = ((s.score - med) * inv_scale).clamp(-4.0, 4.0);
        out.insert(s.name.clone(), (beta * (0.8 * z).tanh()).exp());
    }
    out
}

/// Map [1, inf) to [1, 5] with a tanh squasher.
pub fn squash_kappa(kappa: f32) -> f32 {
    const M: f32 = 5.0;
    const ALPHA: f32 = 0.15;
    if !kappa.is_finite() {
        return M;
    }
    1.0 + (M - 1.0) * (ALPHA * kappa.ln()).tanh()
}

fn jacobi_diag(a: &mut [f32], dim: usize, max_sweeps: usize) {
    const EPS_EIG: f64 = EPSILON;
    for _ in 0..max_sweeps {
        let mut converged = true;
        for p in 0..dim.saturating_sub(1) {
            for q in p + 1..dim {
                let app = a[p * dim + p];
                let aqq = a[q * dim + q];
                let apq = a[p * dim + q];
                if (apq as f64).abs() <= EPS_EIG {
                    continue;
                }
                converged = false;
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                a[p * dim + p] = app - t * apq;
                a[q * dim + q] = aqq + t * apq;
                a[p * dim + q] = 0.0;
                a[q * dim + p] = 0.0;
                for r in 0..dim {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * dim + p];
                    let arq = a[r * dim + q];
                    a[r * dim + p] = c * arp - s * arq;
                    a[p * dim + r] = a[r * dim + p];
                    a[r * dim + q] = s * arp + c * arq;
                    a[q * dim + r] = a[r * dim + q];
                }
            }
        }
        if converged {
            break;
        }
    }
}

fn matmul_a(a: &[f32], rows: usize, cols_a: usize, b: &[f32], cols_b: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; rows * cols_b];
    for i in 0..rows {
        for l in 0..cols_a {
            let aik = a[i * cols_a + l];
            if aik == 0.0 {
                continue;
            }
            for j in 0..cols_b {
                c[i * cols_b + j] += aik * b[l * cols_b + j];
            }
        }
    }
    c
}

fn matmul_b(a: &[f32], rows_a: usize, cols_a: usize, b: &[f32], cols_b: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; cols_a * cols_b];
    for i in 0..cols_a {
        for l in 0..rows_a {
            let aki = a[l * cols_a + i];
            if aki == 0.0 {
                continue;
            }
            for j in 0..cols_b {
                c[i * cols_b + j] += aki * b[l * cols_b + j];
            }
        }
    }
    c
}

fn mgs_qr(y: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut q = vec![0.0f32; rows * cols];
    let mut v = vec![0.0f32; rows];
    for j in 0..cols {
        for i in 0..rows {
            v[i] = y[i * cols + j];
        }
        for b in 0..j {
            let dot: f64 = (0..rows)
                .map(|i| v[i] as f64 * q[i * cols + b] as f64)
                .sum();
            for i in 0..rows {
                v[i] -= dot as f32 * q[i * cols + b];
            }
        }
        let norm_sq: f64 = v.iter().map(|&x| x as f64 * x as f64).sum();
        let norm = (norm_sq as f32).sqrt();
        if norm > 1e-8 {
            let inv = 1.0 / norm;
            for i in 0..rows {
                q[i * cols + j] = v[i] * inv;
            }
        }
    }
    q
}

fn eigenvalues_sorted(mut matrix: Vec<f32>, dim: usize) -> Vec<f32> {
    jacobi_diag(&mut matrix, dim, 20);
    let mut ev: Vec<f32> = (0..dim).map(|i| matrix[i * dim + i]).collect();
    ev.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    ev
}

fn condition_number(ev: &[f32], k: usize) -> f32 {
    const EPS_EIG: f64 = EPSILON;
    if ev.len() < k {
        return f64::INFINITY as f32;
    }
    let sigma_max = (ev[0].max(0.0) as f64).sqrt();
    let sigma_min = (ev[k - 1].max(0.0) as f64).sqrt();
    if sigma_min <= EPS_EIG {
        return f64::INFINITY as f32;
    }
    (sigma_max / sigma_min) as f32
}

/// HMT randomized SVD condition number estimator (kappa = sigma_1 / sigma_20).
pub fn randomized_svd_condition_number(
    f32_sample: &[f32],
    n_per_row: usize,
    total_rows_sampled: usize,
) -> f32 {
    const TARGET_RANK: usize = 25;
    const K: usize = 20;
    const EPS_NORM: f32 = 1e-8;
    if total_rows_sampled < 1 || n_per_row < 1 {
        return f64::INFINITY as f32;
    }
    let m = total_rows_sampled;
    let n = n_per_row;

    let mut norm_matrix = Vec::with_capacity(m * n);
    for r in 0..m {
        let row = &f32_sample[r * n..(r + 1) * n];
        let mut l2_sq = 0.0f64;
        let mut finite = true;
        for &v in row {
            if !v.is_finite() {
                finite = false;
                break;
            }
            l2_sq += v as f64 * v as f64;
        }
        if !finite {
            continue;
        }
        let l2 = (l2_sq as f32).sqrt();
        if l2 <= EPS_NORM {
            continue;
        }
        let inv = 1.0 / l2;
        for &v in row {
            norm_matrix.push(v * inv);
        }
    }
    let valid_m = norm_matrix.len() / n;
    if valid_m < 2 {
        return f64::INFINITY as f32;
    }
    let hmt_direct = valid_m >= TARGET_RANK;
    let hmt_transpose = !hmt_direct && n >= TARGET_RANK;

    let mut rng = crate::core::Mt19937::new(
        ((n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (valid_m as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)) as u32,
    );

    if hmt_direct {
        let mut omega = vec![0.0f32; n * TARGET_RANK];
        for v in &mut omega {
            *v = rng.normal_f32();
        }
        let y0 = matmul_a(&norm_matrix, valid_m, n, &omega, TARGET_RANK);
        let z = matmul_b(&norm_matrix, valid_m, n, &y0, TARGET_RANK);
        let y1 = matmul_a(&norm_matrix, valid_m, n, &z, TARGET_RANK);
        let q = mgs_qr(&y1, valid_m, TARGET_RANK);
        let b = matmul_b(&q, valid_m, TARGET_RANK, &norm_matrix, n);
        let mut c = vec![0.0f32; TARGET_RANK * TARGET_RANK];
        for i in 0..TARGET_RANK {
            for j in 0..TARGET_RANK {
                let sum: f64 = (0..n)
                    .map(|l| b[i * n + l] as f64 * b[j * n + l] as f64)
                    .sum();
                c[i * TARGET_RANK + j] = sum as f32;
            }
        }
        return condition_number(&eigenvalues_sorted(c, TARGET_RANK), K);
    }

    if hmt_transpose {
        let mut omega = vec![0.0f32; valid_m * TARGET_RANK];
        for v in &mut omega {
            *v = rng.normal_f32();
        }
        let y0 = matmul_b(&norm_matrix, valid_m, n, &omega, TARGET_RANK);
        let z = matmul_a(&norm_matrix, valid_m, n, &y0, TARGET_RANK);
        let y1 = matmul_b(&norm_matrix, valid_m, n, &z, TARGET_RANK);
        let q = mgs_qr(&y1, n, TARGET_RANK);
        let mut b = vec![0.0f32; TARGET_RANK * valid_m];
        for j in 0..TARGET_RANK {
            for k in 0..valid_m {
                let sum: f64 = (0..n)
                    .map(|i| q[i * TARGET_RANK + j] as f64 * norm_matrix[k * n + i] as f64)
                    .sum();
                b[j * valid_m + k] = sum as f32;
            }
        }
        let mut c = vec![0.0f32; TARGET_RANK * TARGET_RANK];
        for i in 0..TARGET_RANK {
            for j in 0..TARGET_RANK {
                let sum: f64 = (0..valid_m)
                    .map(|l| b[i * valid_m + l] as f64 * b[j * valid_m + l] as f64)
                    .sum();
                c[i * TARGET_RANK + j] = sum as f32;
            }
        }
        return condition_number(&eigenvalues_sorted(c, TARGET_RANK), K);
    }

    let gram_rows = valid_m <= n;
    let gram_dim = if gram_rows { valid_m } else { n };
    let mut g = vec![0.0f32; gram_dim * gram_dim];
    for i in 0..gram_dim {
        for j in i..gram_dim {
            let sum: f64 = if gram_rows {
                (0..n)
                    .map(|l| norm_matrix[i * n + l] as f64 * norm_matrix[j * n + l] as f64)
                    .sum()
            } else {
                (0..valid_m)
                    .map(|l| norm_matrix[l * n + i] as f64 * norm_matrix[l * n + j] as f64)
                    .sum()
            };
            g[i * gram_dim + j] = sum as f32;
            g[j * gram_dim + i] = sum as f32;
        }
    }
    condition_number(&eigenvalues_sorted(g, gram_dim), K)
}

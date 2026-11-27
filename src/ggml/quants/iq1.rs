//! IQ1_S and IQ1_M quantizers.

use crate::core::{Error, Result};
use crate::ggml::quants::blocks::*;
use crate::ggml::quants::iq2::pattern2;
use crate::ggml::quants::iq_codebooks::*;
use crate::ggml::quants::util::*;

const IQ1S_BLOCK_SIZE: usize = 32;
const IQ1M_BLOCK_SIZE: usize = 16;
const IQ1S_DELTA: f32 = 0.125;

fn aligned(k: usize) -> Result<usize> {
    if k % QK_K != 0 {
        return Err(Error::new("quantize: k not divisible by QK_K"));
    }
    Ok(k / QK_K)
}

fn check_dst<T>(dst: &[T], nb: usize) -> Result<()> {
    if dst.len() < nb {
        return Err(Error::new("quantize: destination buffer too small"));
    }
    Ok(())
}

fn sorted_pairs(xb: &[f32]) -> Vec<(f32, usize)> {
    let mut pairs: Vec<(f32, usize)> = xb.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    pairs
}

pub fn quantize_row_iq1_s_impl(x: &[f32], y: &mut [BlockIq1S], qw: &[f32]) -> Result<()> {
    let data = iq2_data(Iq2Kind::Iq1S)?;
    let nb = aligned(x.len())?;
    check_dst(y, nb)?;
    if qw.len() < x.len() {
        return Err(Error::new("quantize: importance matrix row too short"));
    }
    let x_p = [-1.0 + IQ1S_DELTA, IQ1S_DELTA, 1.0 + IQ1S_DELTA];
    let x_m = [-1.0 - IQ1S_DELTA, -IQ1S_DELTA, 1.0 - IQ1S_DELTA];
    let mut scales = [0.0f32; QK_K / IQ1S_BLOCK_SIZE];
    let mut weight = [0.0f32; IQ1S_BLOCK_SIZE];
    let mut l = [0i8; IQ1S_BLOCK_SIZE];
    let mut index = [0usize; IQ1S_BLOCK_SIZE / 8];
    let mut shifts = [0i8; QK_K / IQ1S_BLOCK_SIZE];

    for ibl in 0..nb {
        y[ibl] = BlockIq1S {
            d: fp32_to_fp16(0.0),
            qs: [0; QK_K / 8],
            qh: [0; QK_K / 32],
        };
        let xb = &x[QK_K * ibl..(ibl + 1) * QK_K];
        let mut sumx2 = 0.0f32;
        for v in xb {
            sumx2 += v * v;
        }
        let sigma2 = 2.0 * sumx2 / QK_K as f32;
        let mut max_scale = 0.0f32;

        for ib in 0..QK_K / IQ1S_BLOCK_SIZE {
            let xs = &xb[IQ1S_BLOCK_SIZE * ib..IQ1S_BLOCK_SIZE * (ib + 1)];
            let qwi = &qw[QK_K * ibl + IQ1S_BLOCK_SIZE * ib..];
            for i in 0..IQ1S_BLOCK_SIZE {
                weight[i] = qwi[i] * (sigma2 + xs[i] * xs[i]).sqrt();
            }
            let mut max = xs[0].abs();
            for i in 1..IQ1S_BLOCK_SIZE {
                max = max.max(xs[i].abs());
            }
            if max < GROUP_MAX_EPS_IQ1_S {
                scales[ib] = 0.0;
                shifts[ib] = 1;
                l.fill(1);
                continue;
            }
            let pairs = sorted_pairs(xs);
            let mut sumx = [0.0f32; IQ1S_BLOCK_SIZE + 1];
            let mut sumw = [0.0f32; IQ1S_BLOCK_SIZE + 1];
            for j in 0..IQ1S_BLOCK_SIZE {
                let i = pairs[j].1;
                sumx[j + 1] = sumx[j] + weight[i] * xs[i];
                sumw[j + 1] = sumw[j] + weight[i];
            }
            let mut best_score = -f32::MAX;
            let mut scale = max;
            let mut besti1 = -1i32;
            let mut besti2 = -1i32;
            let mut best_shift = 0i8;
            for i1 in 0..=IQ1S_BLOCK_SIZE {
                for i2 in i1..=IQ1S_BLOCK_SIZE {
                    for (xx, sg) in [(&x_p, 1i8), (&x_m, -1i8)] {
                        let sumqx = (sumx[i1] - sumx[0]) * xx[0]
                            + (sumx[i2] - sumx[i1]) * xx[1]
                            + (sumx[IQ1S_BLOCK_SIZE] - sumx[i2]) * xx[2];
                        let sumq2 = (sumw[i1] - sumw[0]) * xx[0] * xx[0]
                            + (sumw[i2] - sumw[i1]) * xx[1] * xx[1]
                            + (sumw[IQ1S_BLOCK_SIZE] - sumw[i2]) * xx[2] * xx[2];
                        if sumq2 > 0.0 && sumqx * sumqx > best_score * sumq2 {
                            scale = sumqx / sumq2;
                            best_score = scale * sumqx;
                            besti1 = i1 as i32;
                            besti2 = i2 as i32;
                            best_shift = sg;
                        }
                    }
                }
            }
            if besti1 < 0 || besti2 < 0 || best_shift == 0 {
                scales[ib] = 0.0;
                shifts[ib] = 1;
                l.fill(1);
                continue;
            }
            let i1 = besti1 as usize;
            let i2 = besti2 as usize;
            for j in 0..i1 {
                l[pairs[j].1] = 0;
            }
            for j in i1..i2 {
                l[pairs[j].1] = 1;
            }
            for j in i2..IQ1S_BLOCK_SIZE {
                l[pairs[j].1] = 2;
            }
            if scale < 0.0 {
                for j in 0..IQ1S_BLOCK_SIZE {
                    l[j] = 2 - l[j];
                }
                scale = -scale;
                best_shift = -best_shift;
            }
            let xx = if best_shift == 1 { &x_p } else { &x_m };
            let mut all_on_grid = true;
            for k in 0..IQ1S_BLOCK_SIZE / 8 {
                let u = pattern2(&l, k);
                let mut gi = data.map[u];
                if gi < 0 {
                    all_on_grid = false;
                    let nbors = neighbours_of(&data, u);
                    gi = iq1_find_best_neighbour2(
                        &data,
                        nbors,
                        &xs[8 * k..8 * k + 8],
                        &weight[8 * k..8 * k + 8],
                        scale,
                        xx,
                        &mut l[8 * k..8 * k + 8],
                    ) as i32;
                }
                index[k] = gi as usize;
            }
            if !all_on_grid {
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for k in 0..IQ1S_BLOCK_SIZE / 8 {
                    let levels = levels8(data.grid[index[k]]);
                    for j in 0..8 {
                        let w = weight[8 * k + j];
                        let q = xx[((levels[j] - 1) / 2) as usize];
                        sumqx += w * q as f32 * xs[8 * k + j];
                        sumq2 += w * q as f32 * q as f32;
                    }
                }
                if sumqx > 0.0 && sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            let mut h = 0u16;
            for k in 0..IQ1S_BLOCK_SIZE / 8 {
                y[ibl].qs[(IQ1S_BLOCK_SIZE / 8) * ib + k] = (index[k] & 255) as u8;
                h |= ((index[k] >> 8) as u16) << (3 * k);
            }
            y[ibl].qh[ib] = h;
            scales[ib] = scale;
            shifts[ib] = best_shift;
            max_scale = max_scale.max(scale);
        }
        if max_scale == 0.0 {
            continue;
        }
        let d = max_scale / 15.0;
        y[ibl].d = fp32_to_fp16(d * 1.125);
        let id = 1.0 / d;
        for ib in 0..QK_K / IQ1S_BLOCK_SIZE {
            let mut lv = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 7);
            if shifts[ib] == -1 {
                lv |= 8;
            }
            y[ibl].qh[ib] |= (lv as u16) << 12;
        }
    }
    Ok(())
}

pub fn quantize_iq1_s(
    src: &[f32],
    dst: &mut [BlockIq1S],
    nrow: usize,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
) -> Result<usize> {
    let nb = aligned(n_per_row)?;
    let total = nrow * n_per_row;
    if src.len() < total {
        return Err(Error::new("quantize: input too short"));
    }
    let qw = imatrix.ok_or_else(|| Error::new("iq1_s: missing quantization weights"))?;
    if qw.len() < total {
        return Err(Error::new("quantize: importance matrix too short"));
    }
    check_dst(dst, nrow * nb)?;
    for row in 0..nrow {
        quantize_row_iq1_s_impl(
            &src[row * n_per_row..(row + 1) * n_per_row],
            &mut dst[row * nb..(row + 1) * nb],
            &qw[row * n_per_row..(row + 1) * n_per_row],
        )?;
    }
    Ok(nrow * nb * std::mem::size_of::<BlockIq1S>())
}

pub fn quantize_row_iq1_m_impl(x: &[f32], y: &mut [BlockIq1M], qw: Option<&[f32]>) -> Result<()> {
    let data = iq2_data(Iq2Kind::Iq1M)?;
    let nb = aligned(x.len())?;
    check_dst(y, nb)?;
    let x_p = [-1.0 + IQ1S_DELTA, IQ1S_DELTA, 1.0 + IQ1S_DELTA];
    let x_m = [-1.0 - IQ1S_DELTA, -IQ1S_DELTA, 1.0 - IQ1S_DELTA];
    const MASKS: [u8; 4] = [0x00, 0x80, 0x08, 0x88];
    let mut scales = [0.0f32; QK_K / IQ1M_BLOCK_SIZE];
    let mut weight = [0.0f32; IQ1M_BLOCK_SIZE];
    let mut l = [0i8; IQ1M_BLOCK_SIZE];
    let mut index = [0usize; IQ1M_BLOCK_SIZE / 8];
    let mut shifts = [0i8; QK_K / IQ1M_BLOCK_SIZE];

    for ibl in 0..nb {
        y[ibl] = BlockIq1M {
            qs: [0; QK_K / 8],
            qh: [0; QK_K / 16],
            scales: [0; QK_K / 32],
        };
        let xb = &x[QK_K * ibl..(ibl + 1) * QK_K];
        let mut sumx2 = 0.0f32;
        for v in xb {
            sumx2 += v * v;
        }
        let sigma2 = 2.0 * sumx2 / QK_K as f32;
        let mut max_scale = 0.0f32;

        for ib in 0..QK_K / IQ1M_BLOCK_SIZE {
            let xs = &xb[IQ1M_BLOCK_SIZE * ib..IQ1M_BLOCK_SIZE * (ib + 1)];
            match qw {
                Some(qw) => {
                    let qwi = &qw[QK_K * ibl + IQ1M_BLOCK_SIZE * ib..];
                    for i in 0..IQ1M_BLOCK_SIZE {
                        weight[i] = qwi[i] * (sigma2 + xs[i] * xs[i]).sqrt();
                    }
                }
                None => {
                    for i in 0..IQ1M_BLOCK_SIZE {
                        weight[i] = xs[i] * xs[i];
                    }
                }
            }
            let mut max = xs[0].abs();
            for i in 1..IQ1M_BLOCK_SIZE {
                max = max.max(xs[i].abs());
            }
            if max < GROUP_MAX_EPS_IQ1_M {
                scales[ib] = 0.0;
                shifts[ib] = 0;
                l.fill(1);
                continue;
            }
            let pairs = sorted_pairs(xs);
            let mut px = [[0.0f32; IQ1M_BLOCK_SIZE + 1]; 2];
            let mut pw = [[0.0f32; IQ1M_BLOCK_SIZE + 1]; 2];
            for j in 0..IQ1M_BLOCK_SIZE {
                let i = pairs[j].1;
                let h = if i < IQ1M_BLOCK_SIZE / 2 { 0 } else { 1 };
                px[h][j + 1] = px[h][j] + weight[i] * xs[i];
                px[1 - h][j + 1] = px[1 - h][j];
                pw[h][j + 1] = pw[h][j] + weight[i];
                pw[1 - h][j + 1] = pw[1 - h][j];
            }
            let txs = [px[0][IQ1M_BLOCK_SIZE], px[1][IQ1M_BLOCK_SIZE]];
            let tws = [pw[0][IQ1M_BLOCK_SIZE], pw[1][IQ1M_BLOCK_SIZE]];
            let p2 = x_p[2];
            let m2 = x_m[2];
            let cp1 = x_p[0] * x_p[0] - x_p[1] * x_p[1];
            let cp2 = x_p[1] * x_p[1] - x_p[2] * x_p[2];
            let cm1 = x_m[0] * x_m[0] - x_m[1] * x_m[1];
            let cm2 = x_m[1] * x_m[1] - x_m[2] * x_m[2];
            let mut best_score = -f32::MAX;
            let mut scale = max;
            let mut besti1 = -1i32;
            let mut besti2 = -1i32;
            let mut best_k = -1i32;
            for i1 in 0..=IQ1M_BLOCK_SIZE {
                for i2 in i1..=IQ1M_BLOCK_SIZE {
                    let mut qx_p = [0.0f32; 2];
                    let mut qx_m = [0.0f32; 2];
                    let mut q2_p = [0.0f32; 2];
                    let mut q2_m = [0.0f32; 2];
                    for h in 0..2 {
                        let sx = px[h][i1] + px[h][i2];
                        qx_p[h] = txs[h] * p2 - sx;
                        qx_m[h] = txs[h] * m2 - sx;
                        q2_p[h] = tws[h] * p2 * p2 + pw[h][i1] * cp1 + pw[h][i2] * cp2;
                        q2_m[h] = tws[h] * m2 * m2 + pw[h][i1] * cm1 + pw[h][i2] * cm2;
                    }
                    let sumqx = [
                        qx_p[0] + qx_p[1],
                        qx_p[0] + qx_m[1],
                        qx_m[0] + qx_p[1],
                        qx_m[0] + qx_m[1],
                    ];
                    let sumq2 = [
                        q2_p[0] + q2_p[1],
                        q2_p[0] + q2_m[1],
                        q2_m[0] + q2_p[1],
                        q2_m[0] + q2_m[1],
                    ];
                    for k in 0..4 {
                        if sumq2[k] > 0.0 && sumqx[k] * sumqx[k] > best_score * sumq2[k] {
                            scale = sumqx[k] / sumq2[k];
                            best_score = scale * sumqx[k];
                            besti1 = i1 as i32;
                            besti2 = i2 as i32;
                            best_k = k as i32;
                        }
                    }
                }
            }
            if besti1 < 0 || besti2 < 0 || best_k < 0 {
                scales[ib] = 0.0;
                shifts[ib] = 0;
                l.fill(1);
                continue;
            }
            let i1 = besti1 as usize;
            let i2 = besti2 as usize;
            for j in 0..i1 {
                l[pairs[j].1] = 0;
            }
            for j in i1..i2 {
                l[pairs[j].1] = 1;
            }
            for j in i2..IQ1M_BLOCK_SIZE {
                l[pairs[j].1] = 2;
            }
            if scale < 0.0 {
                for j in 0..IQ1M_BLOCK_SIZE {
                    l[j] = 2 - l[j];
                }
                scale = -scale;
                best_k = match best_k {
                    0 => 3,
                    1 => 2,
                    2 => 1,
                    _ => 0,
                };
            }
            let mut all_on_grid = true;
            for k in 0..IQ1M_BLOCK_SIZE / 8 {
                let xx = if k == 0 {
                    if best_k < 2 { &x_p } else { &x_m }
                } else if best_k % 2 == 0 {
                    &x_p
                } else {
                    &x_m
                };
                let u = pattern2(&l, k);
                let mut gi = data.map[u];
                if gi < 0 {
                    all_on_grid = false;
                    let nbors = neighbours_of(&data, u);
                    gi = iq1_find_best_neighbour2(
                        &data,
                        nbors,
                        &xs[8 * k..8 * k + 8],
                        &weight[8 * k..8 * k + 8],
                        scale,
                        xx,
                        &mut l[8 * k..8 * k + 8],
                    ) as i32;
                }
                index[k] = gi as usize;
            }
            if !all_on_grid {
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for k in 0..IQ1M_BLOCK_SIZE / 8 {
                    let xx = if k == 0 {
                        if best_k < 2 { &x_p } else { &x_m }
                    } else if best_k % 2 == 0 {
                        &x_p
                    } else {
                        &x_m
                    };
                    let levels = levels8(data.grid[index[k]]);
                    for j in 0..8 {
                        let w = weight[8 * k + j];
                        let q = xx[((levels[j] - 1) / 2) as usize];
                        sumqx += w * q as f32 * xs[8 * k + j];
                        sumq2 += w * q as f32 * q as f32;
                    }
                }
                if sumqx > 0.0 && sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            y[ibl].qs[2 * ib] = (index[0] & 255) as u8;
            y[ibl].qs[2 * ib + 1] = (index[1] & 255) as u8;
            y[ibl].qh[ib] = ((index[0] >> 8) | ((index[1] >> 8) << 4)) as u8;
            scales[ib] = scale;
            shifts[ib] = best_k as i8;
            max_scale = max_scale.max(scale);
        }
        if max_scale == 0.0 {
            continue;
        }
        let mut sc = [0u16; QK_K / 32 / 2];
        let mut d = max_scale / 15.0;
        let id = 1.0 / d;
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;
        for ib in 0..QK_K / IQ1M_BLOCK_SIZE {
            let lv = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 7);
            sc[ib / 4] |= (lv as u16) << (3 * (ib % 4));
            y[ibl].qh[ib] |= MASKS[shifts[ib] as usize];
            let xs = &xb[IQ1M_BLOCK_SIZE * ib..IQ1M_BLOCK_SIZE * (ib + 1)];
            match qw {
                Some(qw) => {
                    let qwi = &qw[QK_K * ibl + IQ1M_BLOCK_SIZE * ib..];
                    for i in 0..IQ1M_BLOCK_SIZE {
                        weight[i] = qwi[i] * (sigma2 + xs[i] * xs[i]).sqrt();
                    }
                }
                None => {
                    for i in 0..IQ1M_BLOCK_SIZE {
                        weight[i] = xs[i] * xs[i];
                    }
                }
            }
            for k in 0..IQ1M_BLOCK_SIZE / 8 {
                let xx = if k == 0 {
                    if shifts[ib] < 2 { &x_p } else { &x_m }
                } else if shifts[ib] % 2 == 0 {
                    &x_p
                } else {
                    &x_m
                };
                let gi = y[ibl].qs[2 * ib + k] as usize
                    | ((((y[ibl].qh[ib] as usize) << (8 - 4 * k)) & 0x700));
                let levels = levels8(data.grid[gi]);
                for j in 0..8 {
                    let w = weight[8 * k + j];
                    let q = xx[((levels[j] - 1) / 2) as usize] * (2 * lv + 1) as f32;
                    sumqx += w * q * xs[8 * k + j];
                    sumq2 += w * q * q;
                }
            }
        }
        if sumq2 > 0.0 {
            d = sumqx / sumq2;
        }
        let s = fp32_to_fp16(d * 1.1125) as u16;
        sc[0] |= (s & 0x000F) << 12;
        sc[1] |= (s & 0x00F0) << 8;
        sc[2] |= (s & 0x0F00) << 4;
        sc[3] |= s & 0xF000;
        for i in 0..sc.len() {
            y[ibl].scales[2 * i] = (sc[i] & 255) as u8;
            y[ibl].scales[2 * i + 1] = (sc[i] >> 8) as u8;
        }
    }
    Ok(())
}

pub fn quantize_iq1_m(
    src: &[f32],
    dst: &mut [BlockIq1M],
    nrow: usize,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
) -> Result<usize> {
    let nb = aligned(n_per_row)?;
    let total = nrow * n_per_row;
    if src.len() < total {
        return Err(Error::new("quantize: input too short"));
    }
    if let Some(qw) = imatrix {
        if qw.len() < total {
            return Err(Error::new("quantize: importance matrix too short"));
        }
    }
    check_dst(dst, nrow * nb)?;
    for row in 0..nrow {
        let q = imatrix.map(|qw| &qw[row * n_per_row..(row + 1) * n_per_row]);
        quantize_row_iq1_m_impl(
            &src[row * n_per_row..(row + 1) * n_per_row],
            &mut dst[row * nb..(row + 1) * nb],
            q,
        )?;
    }
    Ok(nrow * nb * std::mem::size_of::<BlockIq1M>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_pairs_keeps_original_indices() {
        let x = [3.0f32, -1.0, 2.0];
        let pairs = sorted_pairs(&x);
        assert_eq!(pairs[0], (-1.0, 1));
        assert_eq!(pairs[2], (3.0, 0));
    }
}

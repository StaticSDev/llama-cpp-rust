//! IQ3_XXS and IQ3_S quantizers (D4 lattice).

use crate::core::{Error, Result};
use crate::ggml::quants::blocks::*;
use crate::ggml::quants::iq_codebooks::*;
use crate::ggml::quants::iq2::prepare_signs;
use crate::ggml::quants::util::*;

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

#[inline]
fn pattern3(l: &[i8], k: usize) -> usize {
    let mut u = 0usize;
    for i in 0..4 {
        u |= (l[4 * k + i] as usize) << (3 * i);
    }
    u
}

#[allow(dead_code)]
struct Iq3Result {
    scales: Vec<f32>,
    qs: Vec<u8>,
    qh: Vec<u8>,
    signs: Vec<u8>,
    scale_bytes: Vec<u8>,
    d: Vec<u16>,
}

fn quantize_iq3_core(x: &[f32], qw: Option<&[f32]>, is_s: bool) -> Result<Iq3Result> {
    let grid_size = if is_s { 512 } else { 256 };
    let data = iq3_data(grid_size)?;
    const KMAXQ: i32 = 8;
    let nb = aligned(x.len())?;
    const NSUB: usize = QK_K / 32;
    const NGROUPS: usize = 8;

    let mut scales = vec![0.0f32; nb * NSUB];
    let mut qs_buf = vec![0u8; nb * (3 * QK_K / 8)];
    let mut qh_buf = vec![0u8; nb * (QK_K / 32)];
    let mut signs_buf = vec![0u8; nb * (QK_K / 8)];
    let mut scale_buf = vec![0u8; nb * (QK_K / 64)];
    let mut d_buf = vec![0u16; nb];
    let mut weight = vec![0.0f32; 32];
    let mut waux = [0.0f32; 32];
    let mut xval = [0.0f32; 32];
    let mut l = [0i8; 32];
    let mut laux = [0i8; 32];
    let mut block_signs = [0u8; 4];
    let mut is_on_grid = [false; NGROUPS];
    let mut is_on_grid_aux = [false; NGROUPS];

    for ibl in 0..nb {
        let xb = &x[QK_K * ibl..(ibl + 1) * QK_K];
        let mut sumx2 = 0.0f32;
        for v in xb {
            sumx2 += v * v;
        }
        let sigma2 = 2.0 * sumx2 / QK_K as f32;
        let mut max_scale = 0.0f32;

        for ib in 0..NSUB {
            let xs = &xb[32 * ib..32 * ib + 32];
            match qw {
                Some(qw) => {
                    let qwi = &qw[QK_K * ibl + 32 * ib..];
                    for i in 0..32 {
                        weight[i] = qwi[i] * (sigma2 + xs[i] * xs[i]).sqrt();
                    }
                }
                None => {
                    for i in 0..32 {
                        weight[i] = xs[i] * xs[i];
                    }
                }
            }
            for i in 0..32 {
                waux[i] = weight[i].sqrt();
            }
            prepare_signs(xs, &weight, &mut xval, &mut block_signs, !is_s);
            let mut max = xval[0];
            for i in 1..32 {
                max = max.max(xval[i]);
            }
            l.fill(0);
            if (is_s && max == 0.0) || (!is_s && max < GROUP_MAX_EPS_IQ3_XXS) {
                scales[ibl * NSUB + ib] = 0.0;
                continue;
            }
            let mut best = 0.0f32;
            let mut scale = max / (2 * KMAXQ - 1) as f32;
            for k in 0..NGROUPS {
                is_on_grid[k] = !is_s;
            }
            let is_range: std::ops::RangeInclusive<i32> = if is_s { -9..=9 } else { -15..=15 };
            for is in is_range {
                let id = (15.0 + is as f32 * 0.2) / max;
                let this_scale = 1.0 / id;
                for k in 0..NGROUPS {
                    for i in 0..4 {
                        let lv = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0))
                            .clamp(0, KMAXQ - 1);
                        laux[4 * k + i] = lv as i8;
                    }
                    let u = pattern3(&laux, k);
                    let gi = data.map[u];
                    is_on_grid_aux[k] = gi >= 0;
                    if gi < 0 {
                        let nbors = neighbours_of_iq3(&data, u);
                        let _ = iq3_find_best_neighbour(
                            &data,
                            nbors,
                            &xval[4 * k..4 * k + 4],
                            &waux[4 * k..4 * k + 4],
                            this_scale,
                            &mut laux[4 * k..4 * k + 4],
                        );
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += weight[i] * xval[i] * q;
                    sumq2 += weight[i] * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    l.copy_from_slice(&laux);
                    is_on_grid = is_on_grid_aux;
                }
            }
            let n_not_on_grid = is_on_grid.iter().filter(|v| !**v).count();
            if n_not_on_grid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..NGROUPS {
                    if !is_s && is_on_grid[k] {
                        continue;
                    }
                    let mut u = 0usize;
                    for i in 0..4 {
                        let lv = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0))
                            .clamp(0, KMAXQ - 1);
                        l[4 * k + i] = lv as i8;
                        u |= (lv as usize) << (3 * i);
                    }
                    let gi = data.map[u];
                    if gi < 0 {
                        let nbors = neighbours_of_iq3(&data, u);
                        let gi = iq3_find_best_neighbour(
                            &data,
                            nbors,
                            &xval[4 * k..4 * k + 4],
                            &waux[4 * k..4 * k + 4],
                            scale,
                            &mut l[4 * k..4 * k + 4],
                        );
                        let levels = levels4(data.grid[gi]);
                        for i in 0..4 {
                            l[4 * k + i] = (levels[i] - 1) / 2;
                        }
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let q = 2.0 * l[i] as f32 + 1.0;
                    sumqx += weight[i] * xval[i] * q;
                    sumq2 += weight[i] * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            if scale < 0.0 {
                scale = -scale;
                for k in 0..4 {
                    block_signs[k] = !block_signs[k];
                }
            }
            for k in 0..NGROUPS {
                let u = pattern3(&l, k);
                let gi = data.map[u];
                if gi < 0 {
                    return Err(Error::new("iq3: quantization point not on grid"));
                }
                let pos = ibl * (3 * QK_K / 8) + 8 * ib + k;
                qs_buf[pos] = (gi & 255) as u8;
                qh_buf[ibl * (QK_K / 32) + ib] |= ((gi >> 8) as u8) << k;
            }
            if is_s {
                for k in 0..4 {
                    signs_buf[ibl * (QK_K / 8) + 4 * ib + k] = block_signs[k];
                }
            } else {
                let packed = block_signs[0] as u32
                    | ((block_signs[1] as u32) << 7)
                    | ((block_signs[2] as u32) << 14)
                    | ((block_signs[3] as u32) << 21);
                let pos = ibl * (3 * QK_K / 8) + QK_K / 4 + 4 * ib;
                qs_buf[pos..pos + 4].copy_from_slice(&packed.to_le_bytes());
            }
            scales[ibl * NSUB + ib] = scale;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            continue;
        }
        let d = max_scale / 31.0;
        let dd = if is_s { d * 1.033 } else { d * 1.0125 };
        let id = 1.0 / d;
        for ib in 0..NSUB {
            let lv = nearest_int(0.5 * (id * scales[ibl * NSUB + ib] - 1.0)).clamp(0, 15);
            if is_s {
                if ib % 2 == 0 {
                    scale_buf[ibl * (QK_K / 64) + ib / 2] = lv as u8;
                } else {
                    scale_buf[ibl * (QK_K / 64) + ib / 2] |= (lv as u8) << 4;
                }
            } else {
                let pos = ibl * (3 * QK_K / 8) + QK_K / 4 + 4 * ib;
                let mut packed = u32::from_le_bytes([
                    qs_buf[pos], qs_buf[pos + 1], qs_buf[pos + 2], qs_buf[pos + 3],
                ]);
                packed |= (lv as u32) << 28;
                qs_buf[pos..pos + 4].copy_from_slice(&packed.to_le_bytes());
            }
        }
        d_buf[ibl] = fp32_to_fp16(dd);
    }
    Ok(Iq3Result {
        scales,
        qs: qs_buf,
        qh: qh_buf,
        signs: signs_buf,
        scale_bytes: scale_buf,
        d: d_buf,
    })
}

pub fn quantize_row_iq3_xxs_impl(x: &[f32], y: &mut [BlockIq3Xxs], qw: Option<&[f32]>) -> Result<()> {
    let result = quantize_iq3_core(x, qw, false)?;
    let nb = aligned(x.len())?;
    check_dst(y, nb)?;
    for ibl in 0..nb {
        y[ibl].d = result.d[ibl];
        y[ibl].qs.copy_from_slice(
            &result.qs[ibl * (3 * QK_K / 8)..(ibl + 1) * (3 * QK_K / 8)],
        );
    }
    Ok(())
}

pub fn quantize_row_iq3_s_impl(x: &[f32], y: &mut [BlockIq3S], qw: Option<&[f32]>) -> Result<()> {
    let result = quantize_iq3_core(x, qw, true)?;
    let nb = aligned(x.len())?;
    check_dst(y, nb)?;
    for ibl in 0..nb {
        y[ibl] = BlockIq3S {
            d: result.d[ibl],
            qs: [0; QK_K / 4],
            qh: [0; QK_K / 32],
            signs: [0; QK_K / 8],
            scales: [0; IQ3S_N_SCALE],
        };
        y[ibl].qs.copy_from_slice(&result.qs[ibl * (QK_K / 4)..(ibl + 1) * (QK_K / 4)]);
        y[ibl].qh.copy_from_slice(&result.qh[ibl * (QK_K / 32)..(ibl + 1) * (QK_K / 32)]);
        y[ibl].signs.copy_from_slice(&result.signs[ibl * (QK_K / 8)..(ibl + 1) * (QK_K / 8)]);
        y[ibl].scales.copy_from_slice(
            &result.scale_bytes[ibl * (QK_K / 64)..(ibl + 1) * (QK_K / 64)],
        );
    }
    Ok(())
}

pub fn quantize_iq3_xxs(
    src: &[f32],
    dst: &mut [BlockIq3Xxs],
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
        quantize_row_iq3_xxs_impl(
            &src[row * n_per_row..(row + 1) * n_per_row],
            &mut dst[row * nb..(row + 1) * nb],
            q,
        )?;
    }
    Ok(nrow * nb * std::mem::size_of::<BlockIq3Xxs>())
}

pub fn quantize_iq3_s(
    src: &[f32],
    dst: &mut [BlockIq3S],
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
        quantize_row_iq3_s_impl(
            &src[row * n_per_row..(row + 1) * n_per_row],
            &mut dst[row * nb..(row + 1) * nb],
            q,
        )?;
    }
    Ok(nrow * nb * std::mem::size_of::<BlockIq3S>())
}

pub fn quantize_row_iq3_xxs_ref(x: &[f32], y: &mut [BlockIq3Xxs], k: usize) -> Result<()> {
    if k % QK_K != 0 {
        return Err(Error::new("quantize: k not divisible by QK_K"));
    }
    quantize_iq3_xxs(x, y, 1, k, None)?;
    Ok(())
}

pub fn quantize_row_iq3_s_ref(x: &[f32], y: &mut [BlockIq3S], k: usize) -> Result<()> {
    if k % QK_K != 0 {
        return Err(Error::new("quantize: k not divisible by QK_K"));
    }
    quantize_iq3_s(x, y, 1, k, None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq3_codebook_is_ready() {
        let data = iq3_data(512).unwrap();
        assert_eq!(data.grid.len(), 512);
    }
}

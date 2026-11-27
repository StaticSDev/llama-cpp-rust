//! IQ2_XXS, IQ2_XS and IQ2_S quantizers.

use crate::core::{Error, Result};
use crate::ggml::quants::blocks::*;
use crate::ggml::quants::iq_codebooks::*;
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

/// Extract the sign bits and (for 2/3-bit grids) fix odd parity.
pub(crate) fn prepare_signs(
    xb: &[f32],
    weight: &[f32],
    xval: &mut [f32],
    block_signs: &mut [u8],
    fix_parity: bool,
) {
    for k in 0..block_signs.len() {
        let mut nflip = 0;
        let mut s = 0u8;
        for i in 0..8 {
            let v = xb[8 * k + i];
            if v >= 0.0 {
                xval[8 * k + i] = v;
            } else {
                xval[8 * k + i] = -v;
                nflip += 1;
                s |= 1 << i;
            }
        }
        if fix_parity && nflip % 2 != 0 {
            let mut imin = 0;
            let mut min = weight[8 * k] * xb[8 * k] * xb[8 * k];
            for i in 1..8 {
                let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                if ax < min {
                    min = ax;
                    imin = i;
                }
            }
            xval[8 * k + imin] = -xval[8 * k + imin];
            s ^= 1 << imin;
        }
        block_signs[k] = if fix_parity { s & 127 } else { s };
    }
}

#[inline]
pub(crate) fn pattern2(l: &[i8], k: usize) -> usize {
    let mut u = 0usize;
    for i in 0..8 {
        u |= (l[8 * k + i] as usize) << (2 * i);
    }
    u
}

fn grid_index2(data: &Iq2Data, u: usize) -> i32 {
    data.map[u]
}

fn resolve_neighbour2(
    data: &Iq2Data,
    cache: &mut Iq2NeighbourCache,
    u: usize,
    xval: &[f32],
    waux: &[f32],
    scale: f32,
    l: &mut [i8],
) -> i32 {
    let mapped = data.map[u];
    if mapped >= 0 {
        mapped
    } else {
        let nb = neighbours_of(data, u);
        iq2_find_best_neighbour_cached(cache, u, nb, data, xval, waux, scale, l) as i32
    }
}

pub fn quantize_row_iq2_xxs_impl(x: &[f32], y: &mut [BlockIq2Xxs], qw: &[f32]) -> Result<()> {
    let data = iq2_data(Iq2Kind::Iq2Xxs)?;
    let nb = aligned(x.len())?;
    check_dst(y, nb)?;
    if qw.len() < x.len() {
        return Err(Error::new("quantize: importance matrix row too short"));
    }
    const KMAXQ: i32 = 3;
    let mut scales = [0.0f32; QK_K / 32];
    let mut weight = [0.0f32; 32];
    let mut xval = [0.0f32; 32];
    let mut l = [0i8; 32];
    let mut lu8 = [0u8; 32];
    let mut laux = [0i8; 32];
    let mut laux_prev = [0i8; 32];
    let mut waux = [0.0f32; 32];
    let mut block_signs = [0u8; 4];
    let mut q2 = [0u32; 2 * (QK_K / 32)];
    let mut caches: [Iq2NeighbourCache; 4] = std::array::from_fn(|_| Iq2NeighbourCache::new());

    for ibl in 0..nb {
        y[ibl].d = fp32_to_fp16(0.0);
        q2.fill(0);
        let xb = &x[QK_K * ibl..(ibl + 1) * QK_K];
        let mut sumx2 = 0.0f32;
        for v in xb {
            sumx2 += v * v;
        }
        let sigma2 = sumx2 / QK_K as f32;
        let mut max_scale = 0.0f32;

        for ib in 0..QK_K / 32 {
            let xs = &xb[32 * ib..32 * ib + 32];
            let qwi = &qw[QK_K * ibl + 32 * ib..];
            for i in 0..32 {
                weight[i] = qwi[i] * (sigma2 + xs[i] * xs[i]).sqrt();
                waux[i] = weight[i].sqrt();
            }
            prepare_signs(xs, &weight, &mut xval, &mut block_signs, true);
            let mut max = xval[0];
            for i in 1..32 {
                max = max.max(xval[i]);
            }
            if max < GROUP_MAX_EPS {
                scales[ib] = 0.0;
                l.fill(0);
                continue;
            }
            let mut scale = make_qp_quants(&xval, KMAXQ + 1, &mut lu8, &weight);
            for i in 0..32 {
                l[i] = lu8[i] as i8;
            }
            let eff_max = scale * KMAXQ as f32;
            if eff_max <= 0.0 {
                scales[ib] = 0.0;
                l.fill(0);
                continue;
            }
            let mut best = 0.0f32;
            for k in 0..4 {
                caches[k].init(&xval[8 * k..8 * k + 8], &waux[8 * k..8 * k + 8]);
            }
            for is in -6..=6 {
                let id = (5.0 + is as f32 * 0.1) / eff_max;
                let this_scale = 1.0 / id;
                for k in 0..4 {
                    for i in 0..8 {
                        let lv = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0))
                            .clamp(0, KMAXQ - 1);
                        laux[8 * k + i] = lv as i8;
                    }
                    let u = pattern2(&laux, k);
                    let _ = resolve_neighbour2(
                        &data,
                        &mut caches[k],
                        u,
                        &xval[8 * k..8 * k + 8],
                        &waux[8 * k..8 * k + 8],
                        this_scale,
                        &mut laux[8 * k..8 * k + 8],
                    );
                }
                if is > -6 && laux == laux_prev {
                    continue;
                }
                laux_prev.copy_from_slice(&laux);
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
                }
            }
            if scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..4 {
                    let mut u = 0usize;
                    for i in 0..8 {
                        let lv = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0))
                            .clamp(0, KMAXQ - 1);
                        l[8 * k + i] = lv as i8;
                        u |= (lv as usize) << (2 * i);
                    }
                    let gi = resolve_neighbour2(
                        &data,
                        &mut caches[k],
                        u,
                        &xval[8 * k..8 * k + 8],
                        &waux[8 * k..8 * k + 8],
                        scale,
                        &mut l[8 * k..8 * k + 8],
                    );
                    if gi >= 0 {
                        let levels = levels8(data.grid[gi as usize]);
                        for i in 0..8 {
                            l[8 * k + i] = (levels[i] - 1) / 2;
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
                    block_signs[k] = !block_signs[k] & 127;
                }
            }
            for k in 0..4 {
                let u = pattern2(&l, k);
                let gi = grid_index2(&data, u);
                if gi < 0 {
                    return Err(Error::new("iq2_xxs: quantization point not on grid"));
                }
                q2[2 * ib] |= (gi as u32) << (8 * k);
                q2[2 * ib + 1] |= (block_signs[k] as u32) << (7 * k);
            }
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }
        if max_scale == 0.0 {
            y[ibl].qs.fill(0);
            continue;
        }
        let d = max_scale / 31.0;
        y[ibl].d = fp32_to_fp16(d);
        let id = 1.0 / d;
        for ib in 0..QK_K / 32 {
            let lv = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
            q2[2 * ib + 1] |= (lv as u32) << 28;
        }
        for j in 0..q2.len() {
            y[ibl].qs[2 * j] = q2[j] as u16;
            y[ibl].qs[2 * j + 1] = (q2[j] >> 16) as u16;
        }
    }
    Ok(())
}

fn quantize_iq2_impl_common(
    x: &[f32],
    qw: Option<&[f32]>,
    kind: Iq2Kind,
    nb_sub: usize,
    cache_count: usize,
) -> Result<(Vec<f32>, Vec<u16>, Vec<u8>)> {
    let data = iq2_data(kind)?;
    let nb = aligned(x.len())?;
    let mut all_scales = Vec::with_capacity(nb * nb_sub);
    let mut all_q2 = Vec::with_capacity(nb * 2 * nb_sub);
    let mut all_signs = Vec::with_capacity(nb * 2 * nb_sub);
    let mut weight = vec![0.0f32; 16];
    let mut xval = [0.0f32; 16];
    let mut l = [0i8; 16];
    let mut laux = [0i8; 16];
    let mut laux_prev = [0i8; 16];
    let mut waux = [0.0f32; 16];
    let mut is_on_grid = [false; 2];
    let mut is_on_grid_aux = [false; 2];
    let mut is_on_grid_prev = [false; 2];
    let mut block_signs = [0u8; 2];
    let mut caches: Vec<Iq2NeighbourCache> = (0..cache_count)
        .map(|_| Iq2NeighbourCache::new())
        .collect();

    for ibl in 0..nb {
        let xb = &x[QK_K * ibl..(ibl + 1) * QK_K];
        let mut sumx2 = 0.0f32;
        for v in xb {
            sumx2 += v * v;
        }
        let sigma2 = if kind == Iq2Kind::Iq2S {
            2.0 * sumx2 / QK_K as f32
        } else {
            sumx2 / QK_K as f32
        };
        let mut max_scale = 0.0f32;
        for ib in 0..nb_sub {
            let xs = &xb[16 * ib..16 * ib + 16];
            match qw {
                Some(qw) => {
                    let qwi = &qw[QK_K * ibl + 16 * ib..];
                    for i in 0..16 {
                        weight[i] = qwi[i] * (sigma2 + xs[i] * xs[i]).sqrt();
                    }
                }
                None => {
                    for i in 0..16 {
                        weight[i] = if kind == Iq2Kind::Iq2S {
                            0.25 * sigma2 + xs[i] * xs[i]
                        } else {
                            xs[i] * xs[i]
                        };
                    }
                }
            }
            for i in 0..16 {
                waux[i] = weight[i].sqrt();
            }
            prepare_signs(
                xs,
                &weight,
                &mut xval,
                &mut block_signs,
                kind != Iq2Kind::Iq2S,
            );
            let mut max = xval[0];
            for i in 1..16 {
                max = max.max(xval[i]);
            }
            l.fill(0);
            if max < GROUP_MAX_EPS_IQ2_S {
                all_scales.push(0.0);
                all_q2.push(0);
                all_q2.push(0);
                all_signs.push(0);
                all_signs.push(0);
                continue;
            }
            let mut best = 0.0f32;
            let mut scale = max / 5.0;
            is_on_grid[0] = true;
            is_on_grid[1] = true;
            for k in 0..2 {
                caches[k].init(&xval[8 * k..8 * k + 8], &waux[8 * k..8 * k + 8]);
            }
            for is in -9..=9 {
                let id = (5.0 + is as f32 * 0.1) / max;
                let this_scale = 1.0 / id;
                for k in 0..2 {
                    for i in 0..8 {
                        let lv = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0))
                            .clamp(0, 2);
                        laux[8 * k + i] = lv as i8;
                    }
                    let u = pattern2(&laux, k);
                    is_on_grid_aux[k] = data.map[u] >= 0;
                    let _ = resolve_neighbour2(
                        &data,
                        &mut caches[k],
                        u,
                        &xval[8 * k..8 * k + 8],
                        &waux[8 * k..8 * k + 8],
                        this_scale,
                        &mut laux[8 * k..8 * k + 8],
                    );
                }
                if is > -9
                    && laux == laux_prev
                    && is_on_grid_aux[0] == is_on_grid_prev[0]
                    && is_on_grid_aux[1] == is_on_grid_prev[1]
                {
                    continue;
                }
                laux_prev.copy_from_slice(&laux);
                is_on_grid_prev = is_on_grid_aux;
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..16 {
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
                for k in 0..2 {
                    if is_on_grid[k] && kind != Iq2Kind::Iq2S {
                        continue;
                    }
                    let mut u = 0usize;
                    for i in 0..8 {
                        let lv = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0))
                            .clamp(0, 2);
                        l[8 * k + i] = lv as i8;
                        u |= (lv as usize) << (2 * i);
                    }
                    let _ = resolve_neighbour2(
                        &data,
                        &mut caches[k],
                        u,
                        &xval[8 * k..8 * k + 8],
                        &waux[8 * k..8 * k + 8],
                        scale,
                        &mut l[8 * k..8 * k + 8],
                    );
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..16 {
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
                for k in 0..2 {
                    block_signs[k] = !block_signs[k];
                }
            }
            for k in 0..2 {
                let u = pattern2(&l, k);
                let gi = grid_index2(&data, u);
                if gi < 0 {
                    return Err(Error::new("iq2: quantization point not on grid"));
                }
                all_q2.push(gi as u16);
                all_signs.push(block_signs[k]);
            }
            all_scales.push(scale);
            max_scale = max_scale.max(scale);
        }
    }
    Ok((all_scales, all_q2, all_signs))
}

pub fn quantize_row_iq2_xs_impl(x: &[f32], y: &mut [BlockIq2Xs], qw: Option<&[f32]>) -> Result<()> {
    let (scales, q2, signs) = quantize_iq2_impl_common(x, qw, Iq2Kind::Iq2Xs, QK_K / 16, 2)?;
    let nb = aligned(x.len())?;
    check_dst(y, nb)?;
    let mut max_scale = 0.0f32;
    for ibl in 0..nb {
        for ib in 0..QK_K / 16 {
            max_scale = max_scale.max(scales[ibl * (QK_K / 16) + ib]);
        }
    }
    let d = max_scale / 31.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    for ibl in 0..nb {
        y[ibl] = BlockIq2Xs {
            d: fp32_to_fp16(d),
            qs: [0; QK_K / 8],
            scales: [0; QK_K / 32],
        };
        for ib in 0..QK_K / 16 {
            let s = scales[ibl * (QK_K / 16) + ib];
            let lv = nearest_int(0.5 * (id * s - 1.0)).clamp(0, 15) as u8;
            if ib % 2 == 0 {
                y[ibl].scales[ib / 2] = lv;
            } else {
                y[ibl].scales[ib / 2] |= lv << 4;
            }
        }
        for j in 0..QK_K / 8 {
            let idx = ibl * 2 * (QK_K / 16) + j;
            y[ibl].qs[j] = q2[idx] | ((signs[idx] as u16) << 9);
        }
    }
    Ok(())
}

pub fn quantize_row_iq2_s_impl(x: &[f32], y: &mut [BlockIq2S], qw: Option<&[f32]>) -> Result<()> {
    let (scales, q2, signs) = quantize_iq2_impl_common(x, qw, Iq2Kind::Iq2S, QK_K / 16, 2)?;
    let nb = aligned(x.len())?;
    check_dst(y, nb)?;
    let mut max_scale = 0.0f32;
    for ibl in 0..nb {
        for ib in 0..QK_K / 16 {
            max_scale = max_scale.max(scales[ibl * (QK_K / 16) + ib]);
        }
    }
    let d = max_scale / 31.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    for ibl in 0..nb {
        y[ibl] = BlockIq2S {
            d: fp32_to_fp16(d * 0.9875),
            qs: [0; QK_K / 4],
            qh: [0; QK_K / 32],
            scales: [0; QK_K / 32],
        };
        for ib in 0..QK_K / 16 {
            let s = scales[ibl * (QK_K / 16) + ib];
            let lv = nearest_int(0.5 * (id * s - 1.0)).clamp(0, 15) as u8;
            if ib % 2 == 0 {
                y[ibl].scales[ib / 2] = lv;
            } else {
                y[ibl].scales[ib / 2] |= lv << 4;
            }
        }
        for i8idx in 0..QK_K / 8 {
            let idx = ibl * 2 * (QK_K / 16) + i8idx;
            let gi = q2[idx];
            y[ibl].qs[i8idx] = (gi & 255) as u8;
            y[ibl].qh[i8idx / 4] |= ((gi >> 8) as u8) << (2 * (i8idx % 4));
            y[ibl].qs[QK_K / 8 + i8idx] = signs[idx];
        }
    }
    Ok(())
}

fn quantize_iq2_rows<B, F>(
    src: &[f32],
    dst: &mut [B],
    nrow: usize,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
    row_fn: F,
) -> Result<usize>
where
    F: Fn(&[f32], &mut [B], Option<&[f32]>) -> Result<()>,
{
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
        let x = &src[row * n_per_row..(row + 1) * n_per_row];
        let q = imatrix.map(|qw| &qw[row * n_per_row..(row + 1) * n_per_row]);
        row_fn(x, &mut dst[row * nb..(row + 1) * nb], q)?;
    }
    Ok(nrow * nb * std::mem::size_of::<B>())
}

pub fn quantize_iq2_xxs(
    src: &[f32],
    dst: &mut [BlockIq2Xxs],
    nrow: usize,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
) -> Result<usize> {
    let nb = aligned(n_per_row)?;
    let total = nrow * n_per_row;
    if src.len() < total {
        return Err(Error::new("quantize: input too short"));
    }
    let qw = imatrix.ok_or_else(|| Error::new("iq2_xxs: missing quantization weights"))?;
    if qw.len() < total {
        return Err(Error::new("quantize: importance matrix too short"));
    }
    check_dst(dst, nrow * nb)?;
    for row in 0..nrow {
        quantize_row_iq2_xxs_impl(
            &src[row * n_per_row..(row + 1) * n_per_row],
            &mut dst[row * nb..(row + 1) * nb],
            &qw[row * n_per_row..(row + 1) * n_per_row],
        )?;
    }
    Ok(nrow * nb * std::mem::size_of::<BlockIq2Xxs>())
}

pub fn quantize_iq2_xs(
    src: &[f32],
    dst: &mut [BlockIq2Xs],
    nrow: usize,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
) -> Result<usize> {
    quantize_iq2_rows(src, dst, nrow, n_per_row, imatrix, quantize_row_iq2_xs_impl)
}

pub fn quantize_iq2_s(
    src: &[f32],
    dst: &mut [BlockIq2S],
    nrow: usize,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
) -> Result<usize> {
    quantize_iq2_rows(src, dst, nrow, n_per_row, imatrix, quantize_row_iq2_s_impl)
}

pub fn quantize_row_iq2_s_ref(x: &[f32], y: &mut [BlockIq2S], k: usize) -> Result<()> {
    if k % QK_K != 0 {
        return Err(Error::new("quantize: k not divisible by QK_K"));
    }
    quantize_iq2_s(x, y, 1, k, None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq2_xxs_codebook_is_ready() {
        let data = iq2_data(Iq2Kind::Iq2Xxs).unwrap();
        assert_eq!(data.grid.len(), 256);
    }
}

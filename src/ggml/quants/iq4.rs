//! IQ4_NL and IQ4_XS quantizers.

use crate::core::{Error, Result};
use crate::ggml::quants::blocks::*;
use crate::ggml::quants::util::{best_index_iq4nl, fp32_to_fp16, nearest_int, GROUP_MAX_EPS, KVALUES_IQ4NL};

fn aligned(k: usize, qk: usize) -> Result<usize> {
    if k % qk != 0 {
        return Err(Error::new("quantize: k not divisible by block size"));
    }
    Ok(k / qk)
}

fn check_dst<T>(dst: &[T], nb: usize) -> Result<()> {
    if dst.len() < nb {
        return Err(Error::new("quantize: destination buffer too small"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn quantize_row_iq4_impl(
    x: &[f32],
    qw: Option<&[f32]>,
    ntry: i32,
    super_block_size: usize,
    block_size: usize,
) -> Result<(u16, Vec<u8>, u16, Vec<u8>)> {
    let mut sigma2 = 0.0f32;
    for j in 0..super_block_size {
        sigma2 += x[j] * x[j];
    }
    sigma2 *= 2.0 / super_block_size as f32;

    let mut mid = [0.0f32; 15];
    let mut fval = [0.0f32; 16];
    for k in 0..15 {
        mid[k] = 0.5 * (KVALUES_IQ4NL[k] as f32 + KVALUES_IQ4NL[k + 1] as f32);
    }
    for k in 0..16 {
        fval[k] = KVALUES_IQ4NL[k] as f32;
    }

    let nsub = super_block_size / block_size;
    let mut q4 = vec![0u8; super_block_size / 2];
    let mut scales = vec![0.0f32; nsub];
    let mut weight = vec![0.0f32; block_size];
    let mut l = vec![0u8; super_block_size];
    let mut max_scale = 0.0f32;
    let mut amax_scale = 0.0f32;

    for ib in 0..nsub {
        let xb = &x[ib * block_size..(ib + 1) * block_size];
        match qw {
            Some(qw) => {
                for j in 0..block_size {
                    weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
                }
            }
            None => {
                for j in 0..block_size {
                    weight[j] = xb[j] * xb[j];
                }
            }
        }
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for j in 0..block_size {
            let ax = xb[j].abs();
            if ax > amax {
                amax = ax;
                max = xb[j];
            }
        }
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        let mut d = if ntry > 0 {
            -max / KVALUES_IQ4NL[0] as f32
        } else {
            max / KVALUES_IQ4NL[0] as f32
        };
        let nt: usize = if ntry >= 0 { (2 * ntry + 2) as usize } else { 1 };
        let mut idt = [0.0f32; 16];
        let mut sqx = [0.0f32; 16];
        let mut sq2 = [0.0f32; 16];
        idt[0] = 1.0 / d;
        for t in 1..nt {
            idt[t] = (t as f32 - 1.0 - ntry as f32 + KVALUES_IQ4NL[0] as f32) / max;
        }
        for j in 0..block_size {
            l[ib * block_size + j] = best_index_iq4nl(&mid, idt[0] * xb[j]) as u8;
        }
        for j in 0..block_size {
            let w = weight[j];
            for t in 0..nt {
                let q = fval[best_index_iq4nl(&mid, idt[t] * xb[j]) as usize];
                sqx[t] += w * q * xb[j];
                sq2[t] += w * q * q;
            }
        }
        d = if sq2[0] > 0.0 { sqx[0] / sq2[0] } else { 0.0 };
        let mut best = d * sqx[0];
        for t in 1..nt {
            if sq2[t] > 0.0 && sqx[t] * sqx[t] > best * sq2[t] {
                d = sqx[t] / sq2[t];
                best = d * sqx[t];
            }
        }
        scales[ib] = d;
        let abs_d = d.abs();
        if abs_d > amax_scale {
            amax_scale = abs_d;
            max_scale = d;
        }
    }

    let mut scales_h = 0u16;
    let mut scales_l = vec![0u8; nsub / 2];
    let dh = if nsub > 1 {
        let d = -max_scale / 32.0;
        let dh = fp32_to_fp16(d);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for ib in 0..nsub {
            let lv = nearest_int(id * scales[ib]).clamp(-32, 31);
            let dl = d * lv as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            let xb = &x[ib * block_size..(ib + 1) * block_size];
            for j in 0..block_size {
                l[ib * block_size + j] = best_index_iq4nl(&mid, idl * xb[j]) as u8;
            }
            let lv = (lv + 32) as u16;
            let l_l = (lv & 0xF) as u8;
            let l_h = lv >> 4;
            if ib % 2 == 0 {
                scales_l[ib / 2] = l_l;
            } else {
                scales_l[ib / 2] |= l_l << 4;
            }
            scales_h |= l_h << (2 * (ib % 8));
        }
        dh
    } else {
        let dh = fp32_to_fp16(scales[0]);
        if ntry > 0 {
            let id = if scales[0] != 0.0 { 1.0 / scales[0] } else { 0.0 };
            for j in 0..super_block_size {
                l[j] = best_index_iq4nl(&mid, id * x[j]) as u8;
            }
        }
        dh
    };

    for i in 0..super_block_size / 32 {
        for j in 0..16 {
            q4[16 * i + j] = l[32 * i + j] | (l[32 * i + 16 + j] << 4);
        }
    }
    Ok((dh, q4, scales_h, scales_l))
}

pub fn quantize_row_iq4_nl_ref(x: &[f32], y: &mut [BlockIq4Nl], k: usize) -> Result<()> {
    let nb = aligned(k, QK4_NL)?;
    check_dst(y, nb)?;
    if x.len() < k {
        return Err(Error::new("quantize: input slice shorter than k"));
    }
    for ibl in 0..nb {
        let (d, q4, _, _) = quantize_row_iq4_impl(
            &x[QK4_NL * ibl..QK4_NL * (ibl + 1)],
            None,
            -1,
            QK4_NL,
            QK4_NL,
        )?;
        y[ibl] = BlockIq4Nl {
            d,
            qs: q4.try_into().unwrap(),
        };
    }
    Ok(())
}

pub fn quantize_iq4_nl(
    src: &[f32],
    dst: &mut [BlockIq4Nl],
    nrow: usize,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
) -> Result<usize> {
    let nb_row = aligned(n_per_row, QK4_NL)?;
    let total = nrow * n_per_row;
    if src.len() < total {
        return Err(Error::new("quantize: input too short"));
    }
    if let Some(qw) = imatrix {
        if qw.len() < total {
            return Err(Error::new("quantize: importance matrix too short"));
        }
    }
    check_dst(dst, nrow * nb_row)?;
    for row in 0..nrow {
        for ibl in 0..nb_row {
            let start = row * n_per_row + QK4_NL * ibl;
            let q = imatrix.map(|qw| &qw[start..start + QK4_NL]);
            let (d, q4, _, _) =
                quantize_row_iq4_impl(&src[start..start + QK4_NL], q, 7, QK4_NL, QK4_NL)?;
            dst[row * nb_row + ibl] = BlockIq4Nl {
                d,
                qs: q4.try_into().unwrap(),
            };
        }
    }
    Ok(nrow * nb_row * std::mem::size_of::<BlockIq4Nl>())
}

pub fn quantize_iq4_xs(
    src: &[f32],
    dst: &mut [BlockIq4Xs],
    nrow: usize,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
) -> Result<usize> {
    let nb_row = aligned(n_per_row, QK_K)?;
    let total = nrow * n_per_row;
    if src.len() < total {
        return Err(Error::new("quantize: input too short"));
    }
    if let Some(qw) = imatrix {
        if qw.len() < total {
            return Err(Error::new("quantize: importance matrix too short"));
        }
    }
    check_dst(dst, nrow * nb_row)?;
    for row in 0..nrow {
        for ibl in 0..nb_row {
            let start = row * n_per_row + QK_K * ibl;
            let q = imatrix.map(|qw| &qw[start..start + QK_K]);
            let (d, q4, scales_h, scales_l) =
                quantize_row_iq4_impl(&src[start..start + QK_K], q, 7, QK_K, 32)?;
            dst[row * nb_row + ibl] = BlockIq4Xs {
                d,
                scales_h,
                scales_l: scales_l.try_into().unwrap(),
                qs: q4.try_into().unwrap(),
            };
        }
    }
    Ok(nrow * nb_row * std::mem::size_of::<BlockIq4Xs>())
}

pub fn quantize_row_iq4_xs_ref(x: &[f32], y: &mut [BlockIq4Xs], k: usize) -> Result<()> {
    if k % QK_K != 0 {
        return Err(Error::new("quantize: k not divisible by QK_K"));
    }
    quantize_iq4_xs(x, y, 1, k, None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq4_nl_quantizes_all_zero_block() {
        let x = [0.0f32; QK4_NL];
        let mut b = [BlockIq4Nl { d: 1, qs: [255; QK4_NL / 2] }];
        quantize_row_iq4_nl_ref(&x, &mut b, QK4_NL).unwrap();
        assert_eq!(b[0].d, 0);
        assert!(b[0].qs.iter().all(|v| *v == 0));
    }
}

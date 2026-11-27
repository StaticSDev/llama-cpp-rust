#![allow(non_snake_case)]

//! 2-8 bit quantization in super-blocks of 256 elements.

use crate::core::{Error, Result};
use crate::ggml::quants::blocks::*;
use crate::ggml::quants::util::*;

fn check_aligned(k: usize, qk: usize) -> Result<usize> {
    if k % qk != 0 {
        return Err(Error::new("quantize: k not divisible by QK_K"));
    }
    Ok(k / qk)
}

fn check_input(x: &[f32], k: usize) -> Result<()> {
    if x.len() < k {
        return Err(Error::new("quantize: input slice shorter than k"));
    }
    Ok(())
}

fn check_dst<T>(dst: &[T], nb: usize) -> Result<()> {
    if dst.len() < nb {
        return Err(Error::new("quantize: destination buffer too small"));
    }
    Ok(())
}

pub fn quantize_row_q2_K_ref(x: &[f32], y: &mut [BlockQ2K], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 16];
    let mut weights = [0.0f32; 16];
    let mut mins = [0.0f32; QK_K / 16];
    let mut scales = [0.0f32; QK_K / 16];
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..QK_K / 16 {
            for (k_idx, w) in weights.iter_mut().enumerate() {
                *w = xb[16 * j + k_idx].abs();
            }
            scales[j] = make_qkx2_quants(
                &xb[16 * j..16 * j + 16],
                &weights,
                3,
                &mut l[16 * j..16 * j + 16],
                &mut mins[j],
                &mut laux,
                -0.5,
                0.1,
                15,
                true,
            );
            max_scale = max_scale.max(scales[j]);
            max_min = max_min.max(mins[j]);
        }
        if max_scale > 0.0 {
            let iscale = 15.0 / max_scale;
            for j in 0..QK_K / 16 {
                y[i].scales[j] = nearest_int(iscale * scales[j]) as u8;
            }
            y[i].d = fp32_to_fp16(max_scale / 15.0);
        } else {
            y[i].scales.fill(0);
            y[i].d = fp32_to_fp16(0.0);
        }
        if max_min > 0.0 {
            let iscale = 15.0 / max_min;
            for j in 0..QK_K / 16 {
                y[i].scales[j] |= (nearest_int(iscale * mins[j]) as u8) << 4;
            }
            y[i].dmin = fp32_to_fp16(max_min / 15.0);
        } else {
            y[i].dmin = fp32_to_fp16(0.0);
        }
        for j in 0..QK_K / 16 {
            let d = fp16_to_fp32(y[i].d) * (y[i].scales[j] & 0xF) as f32;
            if d == 0.0 {
                continue;
            }
            let dm = fp16_to_fp32(y[i].dmin) * (y[i].scales[j] >> 4) as f32;
            for ii in 0..16 {
                let lv = nearest_int((xb[16 * j + ii] + dm) / d).clamp(0, 3);
                l[16 * j + ii] = lv as u8;
            }
        }
        for j in (0..QK_K).step_by(128) {
            for n in 0..32 {
                y[i].qs[j / 4 + n] = l[j + n]
                    | (l[j + n + 32] << 2)
                    | (l[j + n + 64] << 4)
                    | (l[j + n + 96] << 6);
            }
        }
    }
    Ok(())
}

pub fn dequantize_row_q2_K(x: &[BlockQ2K], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let min = fp16_to_fp32(x[i].dmin);
        let mut is = 0usize;
        let mut out = i * QK_K;
        for n in (0..QK_K).step_by(128) {
            let mut shift = 0u32;
            for _ in 0..4 {
                let sc = x[i].scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    let q = ((x[i].qs[n / 4 + l] >> shift) & 3) as i8;
                    y[out + l] = dl * q as f32 - ml;
                }
                out += 16;
                let sc = x[i].scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    let q = ((x[i].qs[n / 4 + l + 16] >> shift) & 3) as i8;
                    y[out + l] = dl * q as f32 - ml;
                }
                out += 16;
                shift += 2;
            }
        }
    }
    Ok(())
}

fn quantize_row_q2_K_impl(x: &[f32], y: &mut [BlockQ2K], qw: Option<&[f32]>) -> Result<()> {
    let n_per_row = x.len();
    let nb = check_aligned(n_per_row, QK_K)?;
    check_dst(y, nb)?;
    let qw = qw.ok_or_else(|| Error::new("q2_K: missing quantization weights"))?;
    if qw.len() < n_per_row {
        return Err(Error::new("quantize: importance matrix row too short"));
    }
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 16];
    let mut mins = [0.0f32; QK_K / 16];
    let mut scales = [0.0f32; QK_K / 16];
    let mut sw = [0.0f32; QK_K / 16];
    let mut weight = [0.0f32; 16];
    let mut ls = [0u8; QK_K / 16];
    let mut lm = [0u8; QK_K / 16];
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        sw.fill(0.0);
        let mut sumx2 = 0.0f32;
        for j in 0..QK_K {
            sumx2 += xb[j] * xb[j];
        }
        let sigma2 = sumx2 / QK_K as f32;
        for j in 0..QK_K / 16 {
            let qwi = &qw[i * QK_K + 16 * j..];
            for k in 0..16 {
                weight[k] = qwi[k] * (sigma2 + xb[16 * j + k] * xb[16 * j + k]).sqrt();
            }
            for k in 0..QK_K / 16 {
                sw[j] += weight[k];
            }
            scales[j] = make_qkx3_quants(
                &xb[16 * j..16 * j + 16],
                Some(&weight),
                3,
                &mut l[16 * j..16 * j + 16],
                &mut mins[j],
                &mut laux,
                -0.9,
                0.05,
                36,
                false,
            );
        }
        let dm = make_qp_quants(&scales, 15, &mut ls, &sw);
        let mm = make_qp_quants(&mins, 15, &mut lm, &sw);
        y[i].d = fp32_to_fp16(dm);
        y[i].dmin = fp32_to_fp16(mm);
        let dm = fp16_to_fp32(y[i].d);
        let mm = fp16_to_fp32(y[i].dmin);
        for j in 0..QK_K / 16 {
            y[i].scales[j] = ls[j] | (lm[j] << 4);
        }
        for j in 0..QK_K / 16 {
            let d = dm * (y[i].scales[j] & 0xF) as f32;
            if d == 0.0 {
                continue;
            }
            let m = mm * (y[i].scales[j] >> 4) as f32;
            for ii in 0..16 {
                let lv = nearest_int((xb[16 * j + ii] + m) / d).clamp(0, 3);
                l[16 * j + ii] = lv as u8;
            }
        }
        for j in (0..QK_K).step_by(128) {
            for n in 0..32 {
                y[i].qs[j / 4 + n] = l[j + n]
                    | (l[j + n + 32] << 2)
                    | (l[j + n + 64] << 4)
                    | (l[j + n + 96] << 6);
            }
        }
    }
    Ok(())
}

pub fn quantize_row_q3_K_ref(x: &[f32], y: &mut [BlockQ3K], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    let mut l = [0i8; QK_K];
    let mut scales = [0.0f32; QK_K / 16];
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut max_scale = 0.0f32;
        let mut amax = 0.0f32;
        for j in 0..QK_K / 16 {
            scales[j] = make_q3_quants(&xb[16 * j..16 * j + 16], 4, &mut l[16 * j..16 * j + 16], true);
            let scale = scales[j].abs();
            if scale > amax {
                amax = scale;
                max_scale = scales[j];
            }
        }
        y[i].scales.fill(0);
        if max_scale != 0.0 {
            let iscale = -32.0 / max_scale;
            for j in 0..QK_K / 16 {
                let mut lv = nearest_int(iscale * scales[j]).clamp(-32, 31) + 32;
                if j < 8 {
                    y[i].scales[j] = (lv & 0xF) as u8;
                } else {
                    y[i].scales[j - 8] |= ((lv & 0xF) << 4) as u8;
                }
                lv >>= 4;
                y[i].scales[j % 4 + 8] |= (lv << (2 * (j / 4))) as u8;
            }
            y[i].d = fp32_to_fp16(1.0 / iscale);
        } else {
            y[i].d = fp32_to_fp16(0.0);
        }
        for j in 0..QK_K / 16 {
            let mut sc = if j < 8 {
                (y[i].scales[j] & 0xF) as i8
            } else {
                (y[i].scales[j - 8] >> 4) as i8
            };
            sc = (sc | ((((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) as i8) << 4)) - 32;
            let d = fp16_to_fp32(y[i].d) * sc as f32;
            if d == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let lv = nearest_int(xb[16 * j + ii] / d).clamp(-4, 3);
                l[16 * j + ii] = (lv + 4) as i8;
            }
        }
        y[i].hmask.fill(0);
        let mut m = 0usize;
        let mut hm = 1u8;
        for j in 0..QK_K {
            if l[j] > 3 {
                y[i].hmask[m] |= hm;
                l[j] -= 4;
            }
            m += 1;
            if m == QK_K / 8 {
                m = 0;
                hm <<= 1;
            }
        }
        for j in (0..QK_K).step_by(128) {
            for n in 0..32 {
                y[i].qs[j / 4 + n] = l[j + n] as u8
                    | ((l[j + n + 32] as u8) << 2)
                    | ((l[j + n + 64] as u8) << 4)
                    | ((l[j + n + 96] as u8) << 6);
            }
        }
    }
    Ok(())
}

pub fn dequantize_row_q3_K(x: &[BlockQ3K], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    for i in 0..nb {
        let d_all = fp16_to_fp32(x[i].d);
        let mut aux = [0u32; 4];
        for j in 0..3 {
            let start = 4 * j;
            aux[j] = u32::from_le_bytes([
                x[i].scales[start],
                x[i].scales[start + 1],
                x[i].scales[start + 2],
                x[i].scales[start + 3],
            ]);
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let mut scale_bytes = [0u8; 16];
        for j in 0..4 {
            scale_bytes[4 * j..4 * j + 4].copy_from_slice(&aux[j].to_le_bytes());
        }
        let mut m = 1u8;
        let mut is = 0usize;
        let mut out = i * QK_K;
        for n in (0..QK_K).step_by(128) {
            let mut shift = 0u32;
            for _ in 0..4 {
                let sc = scale_bytes[is] as i8 - 32;
                is += 1;
                let dl = d_all * sc as f32;
                for l in 0..16 {
                    let q = ((x[i].qs[n / 4 + l] >> shift) & 3) as i8;
                    let h = if x[i].hmask[l] & m != 0 { 0 } else { 4 };
                    y[out + l] = dl * (q - h) as f32;
                }
                out += 16;
                let sc = scale_bytes[is] as i8 - 32;
                is += 1;
                let dl = d_all * sc as f32;
                for l in 0..16 {
                    let q = ((x[i].qs[n / 4 + l + 16] >> shift) & 3) as i8;
                    let h = if x[i].hmask[l + 16] & m != 0 { 0 } else { 4 };
                    y[out + l] = dl * (q - h) as f32;
                }
                out += 16;
                shift += 2;
                m <<= 1;
            }
        }
    }
    Ok(())
}

fn quantize_row_q3_K_impl(x: &[f32], y: &mut [BlockQ3K], qw: Option<&[f32]>) -> Result<()> {
    let n_per_row = x.len();
    let nb = check_aligned(n_per_row, QK_K)?;
    check_dst(y, nb)?;
    let mut l = [0i8; QK_K];
    let mut scales = [0.0f32; QK_K / 16];
    let mut weight = [0.0f32; 16];
    let mut sw = [0.0f32; QK_K / 16];
    let mut ls = [0i8; QK_K / 16];
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut sumx2 = 0.0f32;
        for j in 0..QK_K {
            sumx2 += xb[j] * xb[j];
        }
        let sigma2 = 2.0 * sumx2 / QK_K as f32;
        for j in 0..QK_K / 16 {
            match qw {
                Some(qw) => {
                    let qwi = &qw[i * QK_K + 16 * j..];
                    for k in 0..16 {
                        weight[k] = qwi[k] * (sigma2 + xb[16 * j + k] * xb[16 * j + k]).sqrt();
                    }
                }
                None => {
                    for k in 0..16 {
                        weight[k] = xb[16 * j + k] * xb[16 * j + k];
                    }
                }
            }
            let mut sumw = 0.0f32;
            for k in 0..16 {
                sumw += weight[k];
            }
            sw[j] = sumw;
            scales[j] = make_qx_quants(&xb[16 * j..16 * j + 16], 4, &mut l[16 * j..16 * j + 16], 1, Some(&weight));
        }
        y[i].scales.fill(0);
        let d_block = make_qx_quants(&scales, 32, &mut ls, 1, Some(&sw));
        for j in 0..QK_K / 16 {
            let mut lv = ls[j] as i32;
            if j < 8 {
                y[i].scales[j] = (lv & 0xF) as u8;
            } else {
                y[i].scales[j - 8] |= ((lv & 0xF) << 4) as u8;
            }
            lv >>= 4;
            y[i].scales[j % 4 + 8] |= (lv << (2 * (j / 4))) as u8;
        }
        y[i].d = fp32_to_fp16(d_block);
        for j in 0..QK_K / 16 {
            let mut sc = if j < 8 {
                (y[i].scales[j] & 0xF) as i8
            } else {
                (y[i].scales[j - 8] >> 4) as i8
            };
            sc = (sc | ((((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) as i8) << 4)) - 32;
            let d = fp16_to_fp32(y[i].d) * sc as f32;
            if d == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let lv = nearest_int(xb[16 * j + ii] / d).clamp(-4, 3);
                l[16 * j + ii] = (lv + 4) as i8;
            }
        }
        y[i].hmask.fill(0);
        let mut m = 0usize;
        let mut hm = 1u8;
        for j in 0..QK_K {
            if l[j] > 3 {
                y[i].hmask[m] |= hm;
                l[j] -= 4;
            }
            m += 1;
            if m == QK_K / 8 {
                m = 0;
                hm <<= 1;
            }
        }
        for j in (0..QK_K).step_by(128) {
            for n in 0..32 {
                y[i].qs[j / 4 + n] = l[j + n] as u8
                    | ((l[j + n + 32] as u8) << 2)
                    | ((l[j + n + 64] as u8) << 4)
                    | ((l[j + n + 96] as u8) << 6);
            }
        }
    }
    Ok(())
}

pub fn quantize_row_q4_K_ref(x: &[f32], y: &mut [BlockQ4K], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut weights = [0.0f32; 32];
    let mut mins = [0.0f32; QK_K / 32];
    let mut scales = [0.0f32; QK_K / 32];
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..QK_K / 32 {
            let mut sum_x2 = 0.0f32;
            for k in 0..32 {
                sum_x2 += xb[32 * j + k] * xb[32 * j + k];
            }
            let av_x = (sum_x2 / 32.0).sqrt();
            for k in 0..32 {
                weights[k] = av_x + xb[32 * j + k].abs();
            }
            scales[j] = make_qkx2_quants(
                &xb[32 * j..32 * j + 32],
                &weights,
                15,
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                &mut laux,
                -1.0,
                0.1,
                20,
                false,
            );
            max_scale = max_scale.max(scales[j]);
            max_min = max_min.max(mins[j]);
        }
        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        for j in 0..QK_K / 32 {
            let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
            let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
            if j < 4 {
                y[i].scales[j] = ls;
                y[i].scales[j + 4] = lm;
            } else {
                y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j - 4] |= (ls >> 4) << 6;
                y[i].scales[j] |= (lm >> 4) << 6;
            }
        }
        y[i].d = fp32_to_fp16(max_scale / 63.0);
        y[i].dmin = fp32_to_fp16(max_min / 63.0);
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &y[i].scales);
            let d = fp16_to_fp32(y[i].d) * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = fp16_to_fp32(y[i].dmin) * m as f32;
            for ii in 0..32 {
                let lv = nearest_int((xb[32 * j + ii] + dm) / d).clamp(0, 15);
                l[32 * j + ii] = lv as u8;
            }
        }
        let mut q = 0usize;
        for j in (0..QK_K).step_by(64) {
            for n in 0..32 {
                y[i].qs[q + n] = l[j + n] | (l[j + n + 32] << 4);
            }
            q += 32;
        }
    }
    Ok(())
}

pub fn dequantize_row_q4_K(x: &[BlockQ4K], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let min = fp16_to_fp32(x[i].dmin);
        let mut is = 0usize;
        let mut q = 0usize;
        let mut out = i * QK_K;
        for _ in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, &x[i].scales);
            let d1 = d * sc1 as f32;
            let m1 = min * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, &x[i].scales);
            let d2 = d * sc2 as f32;
            let m2 = min * m2 as f32;
            for l in 0..32 {
                y[out + l] = d1 * (x[i].qs[q + l] & 0xF) as f32 - m1;
            }
            for l in 0..32 {
                y[out + l + 32] = d2 * (x[i].qs[q + l] >> 4) as f32 - m2;
            }
            q += 32;
            is += 2;
            out += 64;
        }
    }
    Ok(())
}

fn quantize_row_q4_K_impl(x: &[f32], y: &mut [BlockQ4K], qw: Option<&[f32]>) -> Result<()> {
    let n_per_row = x.len();
    let nb = check_aligned(n_per_row, QK_K)?;
    check_dst(y, nb)?;
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut ls = [0u8; QK_K / 32];
    let mut lm = [0u8; QK_K / 32];
    let mut weights = [0.0f32; 32];
    let mut sw = [0.0f32; QK_K / 32];
    let mut mins = [0.0f32; QK_K / 32];
    let mut scales = [0.0f32; QK_K / 32];
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut sum_x2 = 0.0f32;
        for k in 0..QK_K {
            sum_x2 += xb[k] * xb[k];
        }
        let sigma2 = 2.0 * sum_x2 / QK_K as f32;
        let av_x = sigma2.sqrt();
        for j in 0..QK_K / 32 {
            match qw {
                Some(qw) => {
                    let qwi = &qw[i * QK_K + 32 * j..];
                    for k in 0..32 {
                        weights[k] = qwi[k] * (sigma2 + xb[32 * j + k] * xb[32 * j + k]).sqrt();
                    }
                }
                None => {
                    for k in 0..32 {
                        weights[k] = av_x + xb[32 * j + k].abs();
                    }
                }
            }
            let mut sumw = 0.0f32;
            for k in 0..32 {
                sumw += weights[k];
            }
            sw[j] = sumw;
            scales[j] = make_qkx3_quants(
                &xb[32 * j..32 * j + 32],
                Some(&weights),
                15,
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                &mut laux,
                -0.9,
                0.05,
                36,
                false,
            );
        }
        let d_block = make_qp_quants(&scales, 63, &mut ls, &sw);
        let m_block = make_qp_quants(&mins, 63, &mut lm, &sw);
        for j in 0..QK_K / 32 {
            let lv_s = ls[j];
            let lv_m = lm[j];
            if j < 4 {
                y[i].scales[j] = lv_s;
                y[i].scales[j + 4] = lv_m;
            } else {
                y[i].scales[j + 4] = (lv_s & 0xF) | ((lv_m & 0xF) << 4);
                y[i].scales[j - 4] |= (lv_s >> 4) << 6;
                y[i].scales[j] |= (lv_m >> 4) << 6;
            }
        }
        y[i].d = fp32_to_fp16(d_block);
        y[i].dmin = fp32_to_fp16(m_block);
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &y[i].scales);
            let d = fp16_to_fp32(y[i].d) * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = fp16_to_fp32(y[i].dmin) * m as f32;
            for ii in 0..32 {
                let lv = nearest_int((xb[32 * j + ii] + dm) / d).clamp(0, 15);
                l[32 * j + ii] = lv as u8;
            }
        }
        let mut q = 0usize;
        for j in (0..QK_K).step_by(64) {
            for n in 0..32 {
                y[i].qs[q + n] = l[j + n] | (l[j + n + 32] << 4);
            }
            q += 32;
        }
    }
    Ok(())
}

pub fn quantize_row_q5_K_ref(x: &[f32], y: &mut [BlockQ5K], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut weights = [0.0f32; 32];
    let mut mins = [0.0f32; QK_K / 32];
    let mut scales = [0.0f32; QK_K / 32];
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..QK_K / 32 {
            let mut sum_x2 = 0.0f32;
            for k in 0..32 {
                sum_x2 += xb[32 * j + k] * xb[32 * j + k];
            }
            let av_x = (sum_x2 / 32.0).sqrt();
            for k in 0..32 {
                weights[k] = av_x + xb[32 * j + k].abs();
            }
            scales[j] = make_qkx2_quants(
                &xb[32 * j..32 * j + 32],
                &weights,
                31,
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                &mut laux,
                -0.5,
                0.1,
                15,
                false,
            );
            max_scale = max_scale.max(scales[j]);
            max_min = max_min.max(mins[j]);
        }
        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        for j in 0..QK_K / 32 {
            let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
            let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
            if j < 4 {
                y[i].scales[j] = ls;
                y[i].scales[j + 4] = lm;
            } else {
                y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j - 4] |= (ls >> 4) << 6;
                y[i].scales[j] |= (lm >> 4) << 6;
            }
        }
        y[i].d = fp32_to_fp16(max_scale / 63.0);
        y[i].dmin = fp32_to_fp16(max_min / 63.0);
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &y[i].scales);
            let d = fp16_to_fp32(y[i].d) * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = fp16_to_fp32(y[i].dmin) * m as f32;
            for ii in 0..32 {
                let lv = nearest_int((xb[32 * j + ii] + dm) / d).clamp(0, 31);
                l[32 * j + ii] = lv as u8;
            }
        }
        y[i].qh.fill(0);
        let mut m1 = 1u8;
        let mut m2 = 2u8;
        let mut ql = 0usize;
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                let mut l1 = l[n + j] as i32;
                if l1 > 15 {
                    l1 -= 16;
                    y[i].qh[j] |= m1;
                }
                let mut l2 = l[n + j + 32] as i32;
                if l2 > 15 {
                    l2 -= 16;
                    y[i].qh[j] |= m2;
                }
                y[i].qs[ql + j] = (l1 as u8) | ((l2 as u8) << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
            ql += 32;
        }
    }
    Ok(())
}

pub fn dequantize_row_q5_K(x: &[BlockQ5K], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let min = fp16_to_fp32(x[i].dmin);
        let mut is = 0usize;
        let mut ql = 0usize;
        let mut out = i * QK_K;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for _ in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, &x[i].scales);
            let d1 = d * sc1 as f32;
            let m1 = min * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, &x[i].scales);
            let d2 = d * sc2 as f32;
            let m2 = min * m2 as f32;
            for l in 0..32 {
                let v = (x[i].qs[ql + l] & 0xF) as f32
                    + if x[i].qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                y[out + l] = d1 * v - m1;
            }
            for l in 0..32 {
                let v = (x[i].qs[ql + l] >> 4) as f32
                    + if x[i].qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                y[out + l + 32] = d2 * v - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
            out += 64;
        }
    }
    Ok(())
}

fn quantize_row_q5_K_impl(x: &[f32], y: &mut [BlockQ5K], qw: Option<&[f32]>) -> Result<()> {
    let n_per_row = x.len();
    let nb = check_aligned(n_per_row, QK_K)?;
    check_dst(y, nb)?;
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut ls = [0u8; QK_K / 32];
    let mut lm = [0u8; QK_K / 32];
    let mut weights = [0.0f32; 32];
    let mut sw = [0.0f32; QK_K / 32];
    let mut mins = [0.0f32; QK_K / 32];
    let mut scales = [0.0f32; QK_K / 32];
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut sum_x2 = 0.0f32;
        for k in 0..QK_K {
            sum_x2 += xb[k] * xb[k];
        }
        let sigma2 = 2.0 * sum_x2 / QK_K as f32;
        let av_x = sigma2.sqrt();
        for j in 0..QK_K / 32 {
            match qw {
                Some(qw) => {
                    let qwi = &qw[i * QK_K + 32 * j..];
                    for k in 0..32 {
                        weights[k] = qwi[k] * (sigma2 + xb[32 * j + k] * xb[32 * j + k]).sqrt();
                    }
                }
                None => {
                    for k in 0..32 {
                        weights[k] = av_x + xb[32 * j + k].abs();
                    }
                }
            }
            let mut sumw = 0.0f32;
            for k in 0..32 {
                sumw += weights[k];
            }
            sw[j] = sumw;
            scales[j] = make_qkx3_quants(
                &xb[32 * j..32 * j + 32],
                Some(&weights),
                31,
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                &mut laux,
                -0.9,
                0.05,
                36,
                false,
            );
        }
        let d_block = make_qp_quants(&scales, 63, &mut ls, &sw);
        let m_block = make_qp_quants(&mins, 63, &mut lm, &sw);
        for j in 0..QK_K / 32 {
            let lv_s = ls[j].min(63);
            let lv_m = lm[j].min(63);
            if j < 4 {
                y[i].scales[j] = lv_s;
                y[i].scales[j + 4] = lv_m;
            } else {
                y[i].scales[j + 4] = (lv_s & 0xF) | ((lv_m & 0xF) << 4);
                y[i].scales[j - 4] |= (lv_s >> 4) << 6;
                y[i].scales[j] |= (lv_m >> 4) << 6;
            }
        }
        y[i].d = fp32_to_fp16(d_block);
        y[i].dmin = fp32_to_fp16(m_block);
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &y[i].scales);
            let d = fp16_to_fp32(y[i].d) * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = fp16_to_fp32(y[i].dmin) * m as f32;
            for ii in 0..32 {
                let lv = nearest_int((xb[32 * j + ii] + dm) / d).clamp(0, 31);
                l[32 * j + ii] = lv as u8;
            }
        }
        y[i].qh.fill(0);
        let mut m1 = 1u8;
        let mut m2 = 2u8;
        let mut ql = 0usize;
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                let mut l1 = l[n + j] as i32;
                if l1 > 15 {
                    l1 -= 16;
                    y[i].qh[j] |= m1;
                }
                let mut l2 = l[n + j + 32] as i32;
                if l2 > 15 {
                    l2 -= 16;
                    y[i].qh[j] |= m2;
                }
                y[i].qs[ql + j] = (l1 as u8) | ((l2 as u8) << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
            ql += 32;
        }
    }
    Ok(())
}

pub fn quantize_row_q6_K_ref(x: &[f32], y: &mut [BlockQ6K], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    let mut l = [0i8; QK_K];
    let mut scales = [0.0f32; QK_K / 16];
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;
        for ib in 0..QK_K / 16 {
            let scale = make_qx_quants(&xb[16 * ib..16 * ib + 16], 32, &mut l[16 * ib..16 * ib + 16], 1, None);
            scales[ib] = scale;
            let abs_scale = scale.abs();
            if abs_scale > max_abs_scale {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }
        if max_abs_scale < GROUP_MAX_EPS {
            y[i] = BlockQ6K {
                ql: [0; QK_K / 2],
                qh: [0; QK_K / 4],
                scales: [0; QK_K / 16],
                d: fp32_to_fp16(0.0),
            };
            continue;
        }
        let iscale = -128.0 / max_scale;
        y[i].d = fp32_to_fp16(1.0 / iscale);
        for ib in 0..QK_K / 16 {
            y[i].scales[ib] = nearest_int(iscale * scales[ib]).min(127) as i8;
        }
        for j in 0..QK_K / 16 {
            let d = fp16_to_fp32(y[i].d) * y[i].scales[j] as f32;
            if d == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let lv = nearest_int(xb[16 * j + ii] / d).clamp(-32, 31);
                l[16 * j + ii] = (lv + 32) as i8;
            }
        }
        let mut ql = 0usize;
        let mut qh = 0usize;
        for j in (0..QK_K).step_by(128) {
            for n in 0..32 {
                let q1 = (l[j + n] & 0xF) as u8;
                let q2 = (l[j + n + 32] & 0xF) as u8;
                let q3 = (l[j + n + 64] & 0xF) as u8;
                let q4 = (l[j + n + 96] & 0xF) as u8;
                y[i].ql[ql + n] = q1 | (q3 << 4);
                y[i].ql[ql + 32 + n] = q2 | (q4 << 4);
                y[i].qh[qh + n] = ((l[j + n] >> 4) as u8)
                    | (((l[j + n + 32] >> 4) as u8) << 2)
                    | (((l[j + n + 64] >> 4) as u8) << 4)
                    | (((l[j + n + 96] >> 4) as u8) << 6);
            }
            ql += 64;
            qh += 32;
        }
    }
    Ok(())
}

pub fn dequantize_row_q6_K(x: &[BlockQ6K], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let mut out = i * QK_K;
        let mut ql = 0usize;
        let mut qh = 0usize;
        let mut sc = 0usize;
        for _ in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((x[i].ql[ql + l] & 0xF) | (((x[i].qh[qh + l] >> 0) & 3) << 4)) as i8 - 32;
                let q2 = ((x[i].ql[ql + 32 + l] & 0xF) | (((x[i].qh[qh + l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((x[i].ql[ql + l] >> 4) | (((x[i].qh[qh + l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((x[i].ql[ql + 32 + l] >> 4) | (((x[i].qh[qh + l] >> 6) & 3) << 4)) as i8 - 32;
                y[out + l] = d * x[i].scales[sc + is] as f32 * q1 as f32;
                y[out + l + 32] = d * x[i].scales[sc + is + 2] as f32 * q2 as f32;
                y[out + l + 64] = d * x[i].scales[sc + is + 4] as f32 * q3 as f32;
                y[out + l + 96] = d * x[i].scales[sc + is + 6] as f32 * q4 as f32;
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    Ok(())
}

fn quantize_row_q6_K_impl(x: &[f32], y: &mut [BlockQ6K], qw: Option<&[f32]>) -> Result<()> {
    let n_per_row = x.len();
    let nb = check_aligned(n_per_row, QK_K)?;
    check_dst(y, nb)?;
    let mut l = [0i8; QK_K];
    let mut scales = [0.0f32; QK_K / 16];
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;
        for ib in 0..QK_K / 16 {
            let scale = make_qx_quants(
                &xb[16 * ib..16 * ib + 16],
                32,
                &mut l[16 * ib..16 * ib + 16],
                1,
                qw.map(|qw| &qw[i * QK_K + 16 * ib..16 * ib + 16 + i * QK_K]),
            );
            scales[ib] = scale;
            let abs_scale = scale.abs();
            if abs_scale > max_abs_scale {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }
        if max_abs_scale < GROUP_MAX_EPS {
            y[i] = BlockQ6K {
                ql: [0; QK_K / 2],
                qh: [0; QK_K / 4],
                scales: [0; QK_K / 16],
                d: fp32_to_fp16(0.0),
            };
            continue;
        }
        let iscale = -128.0 / max_scale;
        y[i].d = fp32_to_fp16(1.0 / iscale);
        for ib in 0..QK_K / 16 {
            y[i].scales[ib] = nearest_int(iscale * scales[ib]).min(127) as i8;
        }
        for j in 0..QK_K / 16 {
            let d = fp16_to_fp32(y[i].d) * y[i].scales[j] as f32;
            if d == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let lv = nearest_int(xb[16 * j + ii] / d).clamp(-32, 31);
                l[16 * j + ii] = (lv + 32) as i8;
            }
        }
        let mut ql = 0usize;
        let mut qh = 0usize;
        for j in (0..QK_K).step_by(128) {
            for n in 0..32 {
                let q1 = (l[j + n] & 0xF) as u8;
                let q2 = (l[j + n + 32] & 0xF) as u8;
                let q3 = (l[j + n + 64] & 0xF) as u8;
                let q4 = (l[j + n + 96] & 0xF) as u8;
                y[i].ql[ql + n] = q1 | (q3 << 4);
                y[i].ql[ql + 32 + n] = q2 | (q4 << 4);
                y[i].qh[qh + n] = ((l[j + n] >> 4) as u8)
                    | (((l[j + n + 32] >> 4) as u8) << 2)
                    | (((l[j + n + 64] >> 4) as u8) << 4)
                    | (((l[j + n + 96] >> 4) as u8) << 6);
            }
            ql += 64;
            qh += 32;
        }
    }
    Ok(())
}

pub fn quantize_row_q8_K_ref(x: &[f32], y: &mut [BlockQ8K], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for j in 0..QK_K {
            let ax = xb[j].abs();
            if ax > amax {
                amax = ax;
                max = xb[j];
            }
        }
        if amax == 0.0 {
            y[i].d = 0.0;
            y[i].qs.fill(0);
            y[i].bsums.fill(0);
            continue;
        }
        let iscale = -127.0 / max;
        for j in 0..QK_K {
            y[i].qs[j] = nearest_int(iscale * xb[j]).min(127) as i8;
        }
        for j in 0..QK_K / 16 {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += y[i].qs[j * 16 + ii] as i32;
            }
            y[i].bsums[j] = sum as i16;
        }
        y[i].d = 1.0 / iscale;
    }
    Ok(())
}

pub fn dequantize_row_q8_K(x: &[BlockQ8K], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        for j in 0..QK_K {
            y[i * QK_K + j] = x[i].d * x[i].qs[j] as f32;
        }
    }
    Ok(())
}

macro_rules! quantize_k {
    ($name:ident, $block:ty, $row:ident, $impl:ident) => {
        pub fn $name(
            src: &[f32],
            dst: &mut [$block],
            nrow: usize,
            n_per_row: usize,
            imatrix: Option<&[f32]>,
        ) -> Result<usize> {
            let total = nrow.checked_mul(n_per_row).unwrap_or(usize::MAX);
            check_input(src, total)?;
            let row_blocks = check_aligned(n_per_row, QK_K)?;
            match imatrix {
                None => $row(src, dst, total)?,
                Some(qw) => {
                    if qw.len() < total {
                        return Err(Error::new("quantize: importance matrix too short"));
                    }
                    check_dst(dst, nrow * row_blocks)?;
                    for row in 0..nrow {
                        let x = &src[row * n_per_row..(row + 1) * n_per_row];
                        let y = &mut dst[row * row_blocks..(row + 1) * row_blocks];
                        $impl(x, y, Some(&qw[row * n_per_row..(row + 1) * n_per_row]))?;
                    }
                }
            }
            Ok(nrow * row_blocks * std::mem::size_of::<$block>())
        }
    };
}

quantize_k!(quantize_q2_K, BlockQ2K, quantize_row_q2_K_ref, quantize_row_q2_K_impl);
quantize_k!(quantize_q3_K, BlockQ3K, quantize_row_q3_K_ref, quantize_row_q3_K_impl);
quantize_k!(quantize_q4_K, BlockQ4K, quantize_row_q4_K_ref, quantize_row_q4_K_impl);
quantize_k!(quantize_q5_K, BlockQ5K, quantize_row_q5_K_ref, quantize_row_q5_K_impl);
quantize_k!(quantize_q6_K, BlockQ6K, quantize_row_q6_K_ref, quantize_row_q6_K_impl);

pub fn quantize_q8_K(
    src: &[f32],
    dst: &mut [BlockQ8K],
    nrow: usize,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
) -> Result<usize> {
    let _ = imatrix;
    let total = nrow.checked_mul(n_per_row).unwrap_or(usize::MAX);
    check_input(src, total)?;
    quantize_row_q8_K_ref(src, dst, total)?;
    Ok(nrow * (n_per_row / QK_K) * std::mem::size_of::<BlockQ8K>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_k_round_trip() {
        let x: Vec<f32> = (0..QK_K).map(|i| (i as f32 % 31.0 - 15.5) * 0.25).collect();
        let mut b = [BlockQ4K {
            d: 0,
            dmin: 0,
            scales: [0; K_SCALE_SIZE],
            qs: [0; QK_K / 2],
        }];
        let mut y = [0.0f32; QK_K];
        quantize_row_q4_K_ref(&x, &mut b, QK_K).unwrap();
        dequantize_row_q4_K(&b, &mut y, QK_K).unwrap();
        let mut max_err = 0.0f32;
        for i in 0..QK_K {
            max_err = max_err.max((y[i] - x[i]).abs());
        }
        assert!(max_err < 0.3, "max error {}", max_err);
    }

    #[test]
    fn q6_k_round_trip() {
        let x: Vec<f32> = (0..QK_K).map(|i| ((i * 3) % 61) as f32 - 30.0).collect();
        let mut b = [BlockQ6K {
            ql: [0; QK_K / 2],
            qh: [0; QK_K / 4],
            scales: [0; QK_K / 16],
            d: 0,
        }];
        let mut y = [0.0f32; QK_K];
        quantize_row_q6_K_ref(&x, &mut b, QK_K).unwrap();
        dequantize_row_q6_K(&b, &mut y, QK_K).unwrap();
        let mut max_err = 0.0f32;
        for i in 0..QK_K {
            max_err = max_err.max((y[i] - x[i]).abs());
        }
        assert!(max_err < 0.5, "max error {}", max_err);
    }
}

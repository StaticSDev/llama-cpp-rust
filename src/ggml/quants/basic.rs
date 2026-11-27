//! Reference quantizers for the small (non-super) block formats.

use crate::core::{Error, Result};
use crate::ggml::quants::blocks::*;
use crate::ggml::quants::util::*;

fn check_aligned(k: usize, qk: usize) -> Result<usize> {
    if k % qk != 0 {
        return Err(Error::new("quantize: k not divisible by block size"));
    }
    Ok(k / qk)
}

fn check_input(x: &[f32], k: usize) -> Result<()> {
    if x.len() < k {
        return Err(Error::new("quantize: input slice shorter than k"));
    }
    Ok(())
}

pub fn quantize_row_q1_0_ref(x: &[f32], y: &mut [BlockQ1_0], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK1_0)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    for i in 0..nb {
        let mut sum_abs = 0.0f32;
        for j in 0..QK1_0 {
            sum_abs += x[i * QK1_0 + j].abs();
        }
        let d = sum_abs / QK1_0 as f32;
        y[i].d = fp32_to_fp16(d);
        y[i].qs.fill(0);
        for j in 0..QK1_0 {
            if x[i * QK1_0 + j] >= 0.0 {
                y[i].qs[j / 8] |= 1 << (j % 8);
            }
        }
    }
    Ok(())
}

pub fn quantize_row_q2_0_ref(x: &[f32], y: &mut [BlockQ2_0], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK2_0)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    for i in 0..nb {
        let mut amax = 0.0f32;
        for j in 0..QK2_0 {
            let a = x[i * QK2_0 + j].abs();
            if a > amax {
                amax = a;
            }
        }
        let d = amax;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        y[i].d = fp32_to_fp16(d);
        y[i].qs.fill(0);
        for j in 0..QK2_0 {
            let w = x[i * QK2_0 + j];
            let q = ((w * id).round() as i32 + 1).clamp(0, 3);
            y[i].qs[j / 4] |= (q as u8) << ((j % 4) * 2);
        }
    }
    Ok(())
}

pub fn quantize_row_q4_0_ref(x: &[f32], y: &mut [BlockQ4_0], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK4_0)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    for i in 0..nb {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for j in 0..QK4_0 {
            let v = x[i * QK4_0 + j];
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        y[i].d = fp32_to_fp16(d);
        for j in 0..QK4_0 / 2 {
            let x0 = x[i * QK4_0 + j] * id;
            let x1 = x[i * QK4_0 + QK4_0 / 2 + j] * id;
            let xi0 = ((x0 + 8.5) as i8).min(15) as u8;
            let xi1 = ((x1 + 8.5) as i8).min(15) as u8;
            y[i].qs[j] = xi0 | (xi1 << 4);
        }
    }
    Ok(())
}

pub fn quantize_row_q4_1_ref(x: &[f32], y: &mut [BlockQ4_1], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK4_1)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    for i in 0..nb {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for j in 0..QK4_1 {
            let v = x[i * QK4_1 + j];
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let d = (max - min) / ((1 << 4) - 1) as f32;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        y[i].d = fp32_to_fp16(d);
        y[i].m = fp32_to_fp16(min);
        for j in 0..QK4_1 / 2 {
            let x0 = (x[i * QK4_1 + j] - min) * id;
            let x1 = (x[i * QK4_1 + QK4_1 / 2 + j] - min) * id;
            let xi0 = ((x0 + 0.5) as i8).min(15) as u8;
            let xi1 = ((x1 + 0.5) as i8).min(15) as u8;
            y[i].qs[j] = xi0 | (xi1 << 4);
        }
    }
    Ok(())
}

pub fn quantize_row_q5_0_ref(x: &[f32], y: &mut [BlockQ5_0], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK5_0)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    for i in 0..nb {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for j in 0..QK5_0 {
            let v = x[i * QK5_0 + j];
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        y[i].d = fp32_to_fp16(d);
        let mut qh = 0u32;
        for j in 0..QK5_0 / 2 {
            let x0 = x[i * QK5_0 + j] * id;
            let x1 = x[i * QK5_0 + QK5_0 / 2 + j] * id;
            let xi0 = ((x0 + 16.5) as i8).min(31) as u32;
            let xi1 = ((x1 + 16.5) as i8).min(31) as u32;
            y[i].qs[j] = ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4)) as u8;
            qh |= ((xi0 & 0x10) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10) >> 4) << (j + QK5_0 / 2);
        }
        y[i].qh = qh.to_le_bytes();
    }
    Ok(())
}

pub fn quantize_row_q5_1_ref(x: &[f32], y: &mut [BlockQ5_1], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK5_1)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    for i in 0..nb {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for j in 0..QK5_1 {
            let v = x[i * QK5_1 + j];
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let d = (max - min) / ((1 << 5) - 1) as f32;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        y[i].d = fp32_to_fp16(d);
        y[i].m = fp32_to_fp16(min);
        let mut qh = 0u32;
        for j in 0..QK5_1 / 2 {
            let x0 = (x[i * QK5_1 + j] - min) * id;
            let x1 = (x[i * QK5_1 + QK5_1 / 2 + j] - min) * id;
            let xi0 = (x0 + 0.5) as u8;
            let xi1 = (x1 + 0.5) as u8;
            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= (((xi0 & 0x10) as u32) >> 4) << (j + 0);
            qh |= (((xi1 & 0x10) as u32) >> 4) << (j + QK5_1 / 2);
        }
        y[i].qh = qh.to_le_bytes();
    }
    Ok(())
}

pub fn quantize_row_q8_0_ref(x: &[f32], y: &mut [BlockQ8_0], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK8_0)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    for i in 0..nb {
        let mut amax = 0.0f32;
        for j in 0..QK8_0 {
            amax = amax.max(x[i * QK8_0 + j].abs());
        }
        let d = amax / ((1 << 7) - 1) as f32;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        y[i].d = fp32_to_fp16(d);
        for j in 0..QK8_0 {
            y[i].qs[j] = (x[i * QK8_0 + j] * id).round() as i8;
        }
    }
    Ok(())
}

pub fn quantize_row_q8_1_ref(x: &[f32], y: &mut [BlockQ8_1], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK8_1)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    for i in 0..nb {
        let mut amax = 0.0f32;
        for j in 0..QK8_1 {
            amax = amax.max(x[i * QK8_1 + j].abs());
        }
        let d = amax / ((1 << 7) - 1) as f32;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        y[i].d = fp32_to_fp16(d);
        let mut sum = 0i32;
        for j in 0..QK8_1 / 2 {
            let v0 = (x[i * QK8_1 + j] * id).round() as i8;
            let v1 = (x[i * QK8_1 + QK8_1 / 2 + j] * id).round() as i8;
            y[i].qs[j] = v0;
            y[i].qs[QK8_1 / 2 + j] = v1;
            sum += v0 as i32;
            sum += v1 as i32;
        }
        y[i].s = fp32_to_fp16(sum as f32 * d);
    }
    Ok(())
}

pub fn quantize_row_mxfp4_ref(x: &[f32], y: &mut [BlockMxfp4], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_MXFP4)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    for i in 0..nb {
        let mut amax = 0.0f32;
        for j in 0..QK_MXFP4 {
            let v = x[i * QK_MXFP4 + j];
            if amax < v.abs() {
                amax = v.abs();
            }
        }
        let e = if amax > 0.0 {
            (amax.log2().floor() - 2.0 + 127.0) as u8
        } else {
            0
        };
        let d = e8m0_to_fp32_half(e);
        y[i].e = e;
        for j in 0..QK_MXFP4 / 2 {
            let x0 = best_index_mxfp4(x[i * QK_MXFP4 + j], d) as u8;
            let x1 = best_index_mxfp4(x[i * QK_MXFP4 + QK_MXFP4 / 2 + j], d) as u8;
            y[i].qs[j] = x0 | (x1 << 4);
        }
    }
    Ok(())
}

pub fn quantize_row_nvfp4_ref(x: &[f32], y: &mut [BlockNvfp4], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_NVFP4)?;
    check_input(x, k)?;
    check_dst(y, nb)?;
    let n_sub = QK_NVFP4 / QK_NVFP4_SUB;
    for i in 0..nb {
        for s in 0..n_sub {
            let xb = &x[i * QK_NVFP4 + s * QK_NVFP4_SUB..];
            let mut amax = 0.0f32;
            for j in 0..QK_NVFP4_SUB {
                if amax < xb[j].abs() {
                    amax = xb[j].abs();
                }
            }
            let ue = fp32_to_ue4m3(amax / 6.0);
            y[i].d[s] = ue;
            let d = ue4m3_to_fp32(ue);
            for j in 0..QK_NVFP4_SUB / 2 {
                let x0 = best_index_mxfp4(xb[j], d) as u8;
                let x1 = best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d) as u8;
                y[i].qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
            }
        }
    }
    Ok(())
}

pub fn quantize_row_f8_e4m3_ref(x: &[f32], y: &mut [Fp8E4M3], k: usize) -> Result<()> {
    check_input(x, k)?;
    check_dst(y, k)?;
    for i in 0..k {
        y[i].bits = fp32_to_f8_e4m3(x[i]);
    }
    Ok(())
}

pub fn dequantize_row_q1_0(x: &[BlockQ1_0], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK1_0)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let neg_d = -d;
        for j in 0..QK1_0 {
            let bit = (x[i].qs[j / 8] >> (j % 8)) & 1;
            y[i * QK1_0 + j] = if bit != 0 { d } else { neg_d };
        }
    }
    Ok(())
}

pub fn dequantize_row_q2_0(x: &[BlockQ2_0], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK2_0)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for j in 0..QK2_0 {
            let q = (x[i].qs[j / 4] >> ((j % 4) * 2)) & 0x03;
            y[i * QK2_0 + j] = (q as i32 - 1) as f32 * d;
        }
    }
    Ok(())
}

pub fn dequantize_row_q4_0(x: &[BlockQ4_0], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK4_0)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for j in 0..QK4_0 / 2 {
            let x0 = (x[i].qs[j] & 0x0F) as i32 - 8;
            let x1 = (x[i].qs[j] >> 4) as i32 - 8;
            y[i * QK4_0 + j] = x0 as f32 * d;
            y[i * QK4_0 + j + QK4_0 / 2] = x1 as f32 * d;
        }
    }
    Ok(())
}

pub fn dequantize_row_q4_1(x: &[BlockQ4_1], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK4_1)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let m = fp16_to_fp32(x[i].m);
        for j in 0..QK4_1 / 2 {
            let x0 = (x[i].qs[j] & 0x0F) as f32;
            let x1 = (x[i].qs[j] >> 4) as f32;
            y[i * QK4_1 + j] = x0 * d + m;
            y[i * QK4_1 + j + QK4_1 / 2] = x1 * d + m;
        }
    }
    Ok(())
}

pub fn dequantize_row_q5_0(x: &[BlockQ5_0], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK5_0)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let qh = u32::from_le_bytes(x[i].qh);
        for j in 0..QK5_0 / 2 {
            let xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((x[i].qs[j] & 0x0F) as u32 | xh_0) as i32 - 16;
            let x1 = ((x[i].qs[j] >> 4) as u32 | xh_1) as i32 - 16;
            y[i * QK5_0 + j] = x0 as f32 * d;
            y[i * QK5_0 + j + QK5_0 / 2] = x1 as f32 * d;
        }
    }
    Ok(())
}

pub fn dequantize_row_q5_1(x: &[BlockQ5_1], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK5_1)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let m = fp16_to_fp32(x[i].m);
        let qh = u32::from_le_bytes(x[i].qh);
        for j in 0..QK5_1 / 2 {
            let xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = (x[i].qs[j] & 0x0F) as u32 | xh_0;
            let x1 = (x[i].qs[j] >> 4) as u32 | xh_1;
            y[i * QK5_1 + j] = x0 as f32 * d + m;
            y[i * QK5_1 + j + QK5_1 / 2] = x1 as f32 * d + m;
        }
    }
    Ok(())
}

pub fn dequantize_row_q8_0(x: &[BlockQ8_0], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK8_0)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for j in 0..QK8_0 {
            y[i * QK8_0 + j] = x[i].qs[j] as f32 * d;
        }
    }
    Ok(())
}

pub fn dequantize_row_mxfp4(x: &[BlockMxfp4], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_MXFP4)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = e8m0_to_fp32_half(x[i].e);
        for j in 0..QK_MXFP4 / 2 {
            let x0 = KVALUES_MXFP4[(x[i].qs[j] & 0x0F) as usize] as f32;
            let x1 = KVALUES_MXFP4[(x[i].qs[j] >> 4) as usize] as f32;
            y[i * QK_MXFP4 + j] = x0 * d;
            y[i * QK_MXFP4 + j + QK_MXFP4 / 2] = x1 * d;
        }
    }
    Ok(())
}

pub fn dequantize_row_nvfp4(x: &[BlockNvfp4], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_NVFP4)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    let n_sub = QK_NVFP4 / QK_NVFP4_SUB;
    for i in 0..nb {
        for s in 0..n_sub {
            let d = ue4m3_to_fp32(x[i].d[s]);
            for j in 0..QK_NVFP4_SUB / 2 {
                let v0 = KVALUES_MXFP4[(x[i].qs[s * (QK_NVFP4_SUB / 2) + j] & 0x0F) as usize] as f32;
                let v1 = KVALUES_MXFP4[(x[i].qs[s * (QK_NVFP4_SUB / 2) + j] >> 4) as usize] as f32;
                y[i * QK_NVFP4 + s * QK_NVFP4_SUB + j] = v0 * d;
                y[i * QK_NVFP4 + s * QK_NVFP4_SUB + j + QK_NVFP4_SUB / 2] = v1 * d;
            }
        }
    }
    Ok(())
}

pub fn dequantize_row_f8_e4m3(x: &[Fp8E4M3], y: &mut [f32], k: usize) -> Result<()> {
    check_dst(x, k)?;
    check_dst(y, k)?;
    for i in 0..k {
        y[i] = f8_e4m3_to_fp32(x[i].bits);
    }
    Ok(())
}

fn quantize_row_q4_0_impl(x: &[f32], y: &mut [BlockQ4_0], qw: &[f32]) -> Result<()> {
    let n_per_row = x.len();
    let nb = check_aligned(n_per_row, QK4_0)?;
    check_dst(y, nb)?;
    if qw.len() < n_per_row {
        return Err(Error::new("quantize: importance matrix row too short"));
    }
    let mut sum_x2 = 0.0f32;
    for j in 0..n_per_row {
        sum_x2 += x[j] * x[j];
    }
    let sigma2 = sum_x2 / n_per_row as f32;
    let mut weight = [0.0f32; QK4_0];
    let mut l = [0i8; QK4_0];
    for ib in 0..nb {
        let xb = &x[QK4_0 * ib..];
        let qwi = &qw[QK4_0 * ib..];
        for j in 0..QK4_0 {
            weight[j] = qwi[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }
        let d = make_qx_quants(&xb[..QK4_0], 8, &mut l, 1, Some(&weight));
        y[ib].d = fp32_to_fp16(d);
        for j in 0..16 {
            y[ib].qs[j] = (l[j] as u8) | ((l[j + 16] as u8) << 4);
        }
    }
    Ok(())
}

fn quantize_row_q4_1_impl(x: &[f32], y: &mut [BlockQ4_1], qw: &[f32]) -> Result<()> {
    let n_per_row = x.len();
    let nb = check_aligned(n_per_row, QK4_1)?;
    check_dst(y, nb)?;
    if qw.len() < n_per_row {
        return Err(Error::new("quantize: importance matrix row too short"));
    }
    let mut sum_x2 = 0.0f32;
    for j in 0..n_per_row {
        sum_x2 += x[j] * x[j];
    }
    let sigma2 = sum_x2 / n_per_row as f32;
    let mut weight = [0.0f32; QK4_1];
    let mut l = [0u8; QK4_1];
    let mut laux = [0u8; QK4_1];
    for ib in 0..nb {
        let xb = &x[QK4_1 * ib..];
        let qwi = &qw[QK4_1 * ib..];
        for j in 0..QK4_1 {
            weight[j] = qwi[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }
        let mut min = 0.0f32;
        let d = make_qkx3_quants(
            &xb[..QK4_1],
            Some(&weight),
            15,
            &mut l,
            &mut min,
            &mut laux,
            -0.9,
            0.05,
            36,
            false,
        );
        y[ib].d = fp32_to_fp16(d);
        y[ib].m = fp32_to_fp16(-min);
        for j in 0..16 {
            y[ib].qs[j] = l[j] | (l[j + 16] << 4);
        }
    }
    Ok(())
}

fn quantize_row_q5_0_impl(x: &[f32], y: &mut [BlockQ5_0], qw: &[f32]) -> Result<()> {
    let n_per_row = x.len();
    let nb = check_aligned(n_per_row, QK5_0)?;
    check_dst(y, nb)?;
    if qw.len() < n_per_row {
        return Err(Error::new("quantize: importance matrix row too short"));
    }
    let mut sum_x2 = 0.0f32;
    for j in 0..n_per_row {
        sum_x2 += x[j] * x[j];
    }
    let sigma2 = sum_x2 / n_per_row as f32;
    let mut weight = [0.0f32; QK5_0];
    let mut l = [0i8; QK5_0];
    for ib in 0..nb {
        let xb = &x[QK5_0 * ib..];
        let qwi = &qw[QK5_0 * ib..];
        for j in 0..QK5_0 {
            weight[j] = qwi[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }
        let d = make_qx_quants(&xb[..QK5_0], 16, &mut l, 1, Some(&weight));
        y[ib].d = fp32_to_fp16(d);
        let mut qh = 0u32;
        for j in 0..16 {
            let xi0 = l[j] as u32;
            let xi1 = l[j + 16] as u32;
            y[ib].qs[j] = ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4)) as u8;
            qh |= ((xi0 & 0x10) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10) >> 4) << (j + QK5_0 / 2);
        }
        y[ib].qh = qh.to_le_bytes();
    }
    Ok(())
}

fn quantize_row_q5_1_impl(x: &[f32], y: &mut [BlockQ5_1], qw: &[f32]) -> Result<()> {
    let n_per_row = x.len();
    let nb = check_aligned(n_per_row, QK5_1)?;
    check_dst(y, nb)?;
    if qw.len() < n_per_row {
        return Err(Error::new("quantize: importance matrix row too short"));
    }
    let mut sum_x2 = 0.0f32;
    for j in 0..n_per_row {
        sum_x2 += x[j] * x[j];
    }
    let sigma2 = sum_x2 / n_per_row as f32;
    let mut weight = [0.0f32; QK5_1];
    let mut l = [0u8; QK5_1];
    let mut laux = [0u8; QK5_1];
    for ib in 0..nb {
        let xb = &x[QK5_1 * ib..];
        let qwi = &qw[QK5_1 * ib..];
        for j in 0..QK5_1 {
            weight[j] = qwi[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }
        let mut min = 0.0f32;
        let d = make_qkx3_quants(
            &xb[..QK5_1],
            Some(&weight),
            31,
            &mut l,
            &mut min,
            &mut laux,
            -0.9,
            0.05,
            36,
            false,
        );
        y[ib].d = fp32_to_fp16(d);
        y[ib].m = fp32_to_fp16(-min);
        let mut qh = 0u32;
        for j in 0..16 {
            let xi0 = l[j] as u32;
            let xi1 = l[j + 16] as u32;
            y[ib].qs[j] = ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4)) as u8;
            qh |= ((xi0 & 0x10) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10) >> 4) << (j + QK5_1 / 2);
        }
        y[ib].qh = qh.to_le_bytes();
    }
    Ok(())
}

macro_rules! quantize_simple {
    ($name:ident, $block:ty, $row:ident, $qk:expr) => {
        pub fn $name(
            src: &[f32],
            dst: &mut [$block],
            nrow: usize,
            n_per_row: usize,
            imatrix: Option<&[f32]>,
        ) -> Result<usize> {
            let _ = imatrix;
            let total = nrow.checked_mul(n_per_row).unwrap_or(usize::MAX);
            let nb = check_aligned(total, $qk)?;
            check_input(src, total)?;
            check_dst(dst, nb)?;
            $row(src, dst, total)?;
            Ok(total / $qk * std::mem::size_of::<$block>())
        }
    };
}

macro_rules! quantize_with_imatrix {
    ($name:ident, $block:ty, $row:ident, $impl:ident, $qk:expr) => {
        pub fn $name(
            src: &[f32],
            dst: &mut [$block],
            nrow: usize,
            n_per_row: usize,
            imatrix: Option<&[f32]>,
        ) -> Result<usize> {
            let total = nrow.checked_mul(n_per_row).unwrap_or(usize::MAX);
            check_input(src, total)?;
            match imatrix {
                None => {
                    let nb = check_aligned(total, $qk)?;
                    check_dst(dst, nb)?;
                    $row(src, dst, total)?;
                }
                Some(qw) => {
                    if qw.len() < total {
                        return Err(Error::new("quantize: importance matrix too short"));
                    }
                    let row_blocks = check_aligned(n_per_row, $qk)?;
                    check_dst(dst, nrow * row_blocks)?;
                    for row in 0..nrow {
                        let x = &src[row * n_per_row..(row + 1) * n_per_row];
                        let q = &qw[row * n_per_row..(row + 1) * n_per_row];
                        let y = &mut dst[row * row_blocks..(row + 1) * row_blocks];
                        $impl(x, y, q)?;
                    }
                }
            }
            Ok(total / $qk * std::mem::size_of::<$block>())
        }
    };
}

quantize_simple!(quantize_q1_0, BlockQ1_0, quantize_row_q1_0_ref, QK1_0);
quantize_simple!(quantize_q2_0, BlockQ2_0, quantize_row_q2_0_ref, QK2_0);
quantize_with_imatrix!(quantize_q4_0, BlockQ4_0, quantize_row_q4_0_ref, quantize_row_q4_0_impl, QK4_0);
quantize_with_imatrix!(quantize_q4_1, BlockQ4_1, quantize_row_q4_1_ref, quantize_row_q4_1_impl, QK4_1);
quantize_with_imatrix!(quantize_q5_0, BlockQ5_0, quantize_row_q5_0_ref, quantize_row_q5_0_impl, QK5_0);
quantize_with_imatrix!(quantize_q5_1, BlockQ5_1, quantize_row_q5_1_ref, quantize_row_q5_1_impl, QK5_1);
quantize_simple!(quantize_q8_0, BlockQ8_0, quantize_row_q8_0_ref, QK8_0);
quantize_simple!(quantize_mxfp4, BlockMxfp4, quantize_row_mxfp4_ref, QK_MXFP4);
quantize_simple!(quantize_nvfp4, BlockNvfp4, quantize_row_nvfp4_ref, QK_NVFP4);
quantize_simple!(quantize_f8_e4m3, Fp8E4M3, quantize_row_f8_e4m3_ref, 1);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_round_trip_small_error() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let mut b = [BlockQ4_0 { d: 0, qs: [0; 16] }];
        let mut y = [0.0f32; 32];
        quantize_row_q4_0_ref(&x, &mut b, 32).unwrap();
        dequantize_row_q4_0(&b, &mut y, 32).unwrap();
        for i in 0..32 {
            assert!((y[i] - x[i]).abs() < 0.12);
        }
    }

    #[test]
    fn q5_0_high_bits_round_trip() {
        let x: Vec<f32> = (0..32).map(|i| ((i * 7) % 32) as f32 - 16.0).collect();
        let mut b = [BlockQ5_0 { d: 0, qh: [0; 4], qs: [0; 16] }];
        let mut y = [0.0f32; 32];
        quantize_row_q5_0_ref(&x, &mut b, 32).unwrap();
        dequantize_row_q5_0(&b, &mut y, 32).unwrap();
        for i in 0..32 {
            assert!((y[i] - x[i]).abs() < 1.0);
        }
    }

    #[test]
    fn mxfp4_round_trip() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let mut b = [BlockMxfp4 { e: 0, qs: [0; 16] }];
        let mut y = [0.0f32; 32];
        quantize_row_mxfp4_ref(&x, &mut b, 32).unwrap();
        dequantize_row_mxfp4(&b, &mut y, 32).unwrap();
        for i in 0..32 {
            assert!((y[i] - x[i]).abs() < 2.0);
        }
    }

    #[test]
    fn rejects_unaligned_k() {
        let x = [0.0f32; 33];
        let mut b = [BlockQ4_0 { d: 0, qs: [0; 16] }];
        assert!(quantize_row_q4_0_ref(&x, &mut b, 33).is_err());
    }
}

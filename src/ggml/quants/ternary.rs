//! Ternary quantizers (BitNet b1.58 and TriLMs).

use crate::core::{Error, Result};
use crate::ggml::quants::blocks::*;
use crate::ggml::quants::util::{check_dst, fp16_to_fp32, fp32_to_fp16};

fn check_aligned(k: usize) -> Result<usize> {
    if k % QK_K != 0 {
        return Err(Error::new("quantize: k not divisible by QK_K"));
    }
    Ok(k / QK_K)
}

pub fn quantize_row_tq1_0_ref(x: &[f32], y: &mut [BlockTq1_0], k: usize) -> Result<()> {
    let nb = check_aligned(k)?;
    if x.len() < k {
        return Err(Error::new("quantize: input slice shorter than k"));
    }
    check_dst(y, nb)?;
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut amax = 0.0f32;
        for j in 0..QK_K {
            amax = amax.max(xb[j].abs());
        }
        let d = amax;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        y[i].d = fp32_to_fp16(d);
        let mut pos = 0usize;
        for chunk in [32usize, 16] {
            let start = if chunk == 32 { 0 } else { 32 };
            for j in (start..start + chunk).step_by(chunk) {
                for m in 0..chunk {
                    let mut q = 0u32;
                    for n in 0..5 {
                        let xi = (xb[pos + m + n * chunk] * id).round() as i32 + 1;
                        q *= 3;
                        q += xi as u32;
                    }
                    let q = ((q * 256 + (243 - 1)) / 243) as u8;
                    y[i].qs[j + m] = q;
                }
                pos += 5 * chunk;
            }
        }
        for j in 0..y[i].qh.len() {
            let mut q = 0u32;
            for m in 0..4 {
                let xi = (xb[pos + j + m * y[i].qh.len()] * id).round() as i32 + 1;
                q *= 3;
                q += xi as u32;
            }
            q *= 3;
            let q = ((q * 256 + (243 - 1)) / 243) as u8;
            y[i].qh[j] = q;
        }
    }
    Ok(())
}

pub fn quantize_row_tq2_0_ref(x: &[f32], y: &mut [BlockTq2_0], k: usize) -> Result<()> {
    let nb = check_aligned(k)?;
    if x.len() < k {
        return Err(Error::new("quantize: input slice shorter than k"));
    }
    check_dst(y, nb)?;
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut amax = 0.0f32;
        for j in 0..QK_K {
            amax = amax.max(xb[j].abs());
        }
        let d = amax;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        y[i].d = fp32_to_fp16(d);
        let mut pos = 0usize;
        for j in (0..y[i].qs.len()).step_by(32) {
            for m in 0..32 {
                let mut q = 0u32;
                for n in 0..4 {
                    let xi = (xb[pos + m + n * 32] * id).round() as i32 + 1;
                    q += ((xi & 3) as u32) << (2 * n);
                }
                y[i].qs[j + m] = q as u8;
            }
            pos += 4 * 32;
        }
    }
    Ok(())
}

pub fn quantize_tq1_0(
    src: &[f32],
    dst: &mut [BlockTq1_0],
    nrow: usize,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
) -> Result<usize> {
    let _ = imatrix;
    let total = nrow.checked_mul(n_per_row).unwrap_or(usize::MAX);
    quantize_row_tq1_0_ref(src, dst, total)?;
    Ok(nrow * (n_per_row / QK_K) * std::mem::size_of::<BlockTq1_0>())
}

pub fn quantize_tq2_0(
    src: &[f32],
    dst: &mut [BlockTq2_0],
    nrow: usize,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
) -> Result<usize> {
    let _ = imatrix;
    let total = nrow.checked_mul(n_per_row).unwrap_or(usize::MAX);
    quantize_row_tq2_0_ref(src, dst, total)?;
    Ok(nrow * (n_per_row / QK_K) * std::mem::size_of::<BlockTq2_0>())
}

pub fn dequantize_row_tq1_0(x: &[BlockTq1_0], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    const POW3: [u32; 6] = [1, 3, 9, 27, 81, 243];
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let mut out = i * QK_K;
        for chunk in [32usize, 16] {
            let start = if chunk == 32 { 0 } else { 32 };
            for j in (start..start + chunk).step_by(chunk) {
                for n in 0..5 {
                    for m in 0..chunk {
                        let q = (x[i].qs[j + m] as u32 * POW3[n]) as u8 as u32;
                        let xi = ((q * 3) >> 8) as i32;
                        y[out] = (xi - 1) as f32 * d;
                        out += 1;
                    }
                }
            }
        }
        for n in 0..4 {
            for j in 0..x[i].qh.len() {
                let q = (x[i].qh[j] as u32 * POW3[n]) as u8 as u32;
                let xi = ((q * 3) >> 8) as i32;
                y[out] = (xi - 1) as f32 * d;
                out += 1;
            }
        }
    }
    Ok(())
}

pub fn dequantize_row_tq2_0(x: &[BlockTq2_0], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k)?;
    check_dst(x, nb)?;
    check_dst(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let mut out = i * QK_K;
        for j in (0..x[i].qs.len()).step_by(32) {
            for l in 0..4 {
                for m in 0..32 {
                    let q = ((x[i].qs[j + m] >> (l * 2)) & 3) as i32;
                    y[out] = (q - 1) as f32 * d;
                    out += 1;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tq2_0_round_trip() {
        let x: Vec<f32> = (0..QK_K).map(|i| ((i % 3) as f32 - 1.0)).collect();
        let mut b = [BlockTq2_0 {
            qs: [0; QK_K / 4],
            d: 0,
        }];
        let mut y = [0.0f32; QK_K];
        quantize_row_tq2_0_ref(&x, &mut b, QK_K).unwrap();
        dequantize_row_tq2_0(&b, &mut y, QK_K).unwrap();
        for i in 0..QK_K {
            assert_eq!(y[i], x[i]);
        }
    }

    #[test]
    fn tq1_0_round_trip() {
        let x: Vec<f32> = (0..QK_K).map(|i| ((i % 3) as f32 - 1.0)).collect();
        let mut b = [BlockTq1_0 {
            qs: [0; (QK_K - 4 * QK_K / 64) / 5],
            qh: [0; QK_K / 64],
            d: 0,
        }];
        let mut y = [0.0f32; QK_K];
        quantize_row_tq1_0_ref(&x, &mut b, QK_K).unwrap();
        dequantize_row_tq1_0(&b, &mut y, QK_K).unwrap();
        for i in 0..QK_K {
            assert_eq!(y[i], x[i]);
        }
    }
}

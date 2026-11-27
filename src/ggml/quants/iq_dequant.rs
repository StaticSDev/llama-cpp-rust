//! Dequantizers for the importance-matrix (IQ) block formats.

use crate::core::{Error, Result};
use crate::ggml::quants::blocks::*;
use crate::ggml::quants::tables::*;
use crate::ggml::quants::util::{fp16_to_fp32, KVALUES_IQ4NL};

const IQ1S_DELTA: f32 = 0.125;

fn check_aligned(k: usize, qk: usize) -> Result<usize> {
    if k % qk != 0 {
        return Err(Error::new("dequantize: k not divisible by block size"));
    }
    Ok(k / qk)
}

fn check_blocks<T>(x: &[T], nb: usize) -> Result<()> {
    if x.len() < nb {
        return Err(Error::new("dequantize: block slice too short"));
    }
    Ok(())
}

fn check_out(y: &[f32], k: usize) -> Result<()> {
    if y.len() < k {
        return Err(Error::new("dequantize: output slice too short"));
    }
    Ok(())
}

#[inline]
fn grid8_u64(word: u64) -> [u8; 8] {
    word.to_le_bytes()
}

#[inline]
fn grid4_u32(word: u32) -> [u8; 4] {
    word.to_le_bytes()
}

#[inline]
fn grid8_i8(word: u64) -> [i8; 8] {
    word.to_le_bytes().map(|b| b as i8)
}

pub fn dequantize_row_iq2_xxs(x: &[BlockIq2Xxs], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_blocks(x, nb)?;
    check_out(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for ib32 in 0..QK_K / 32 {
            let mut aux32 = [0u32; 2];
            for j in 0..2 {
                let start = 4 * ib32 + 2 * j;
                let w0 = x[i].qs[start];
                let w1 = x[i].qs[start + 1];
                aux32[j] = u32::from_le_bytes([
                    w0 as u8,
                    (w0 >> 8) as u8,
                    w1 as u8,
                    (w1 >> 8) as u8,
                ]);
            }
            let db = d * (0.5 + (aux32[1] >> 28) as f32) * 0.25;
            let mut out = i * QK_K + 32 * ib32;
            for l in 0..4 {
                let idx = ((aux32[0] >> (8 * l)) & 0xFF) as usize;
                let grid = grid8_u64(IQ2XXS_GRID[idx]);
                let signs = KSIGNS_IQ2XS[((aux32[1] >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    y[out + j] = db * grid[j] as f32 * sign;
                }
                out += 8;
            }
        }
    }
    Ok(())
}

pub fn dequantize_row_iq2_xs(x: &[BlockIq2Xs], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_blocks(x, nb)?;
    check_out(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for ib32 in 0..QK_K / 32 {
            let db0 = d * (0.5 + (x[i].scales[ib32] & 0xF) as f32) * 0.25;
            let db1 = d * (0.5 + (x[i].scales[ib32] >> 4) as f32) * 0.25;
            let mut out = i * QK_K + 32 * ib32;
            for l in 0..4 {
                let q = x[i].qs[4 * ib32 + l];
                let grid = grid8_u64(IQ2XS_GRID[(q & 511) as usize]);
                let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                let db = if l / 2 == 0 { db0 } else { db1 };
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    y[out + j] = db * grid[j] as f32 * sign;
                }
                out += 8;
            }
        }
    }
    Ok(())
}

pub fn dequantize_row_iq2_s(x: &[BlockIq2S], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_blocks(x, nb)?;
    check_out(y, k)?;
    let signs_offset = QK_K / 8;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for ib32 in 0..QK_K / 32 {
            let db0 = d * (0.5 + (x[i].scales[ib32] & 0xF) as f32) * 0.25;
            let db1 = d * (0.5 + (x[i].scales[ib32] >> 4) as f32) * 0.25;
            let mut out = i * QK_K + 32 * ib32;
            for l in 0..4 {
                let idx = x[i].qs[4 * ib32 + l] as u32
                    | (((x[i].qh[ib32] as u32) << (8 - 2 * l)) & 0x300);
                let grid = grid8_u64(IQ2S_GRID[idx as usize]);
                let signs = x[i].qs[signs_offset + 4 * ib32 + l];
                let db = if l / 2 == 0 { db0 } else { db1 };
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    y[out + j] = db * grid[j] as f32 * sign;
                }
                out += 8;
            }
        }
    }
    Ok(())
}

pub fn dequantize_row_iq3_xxs(x: &[BlockIq3Xxs], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_blocks(x, nb)?;
    check_out(y, k)?;
    let ss_offset = QK_K / 4;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for ib32 in 0..QK_K / 32 {
            let start = ss_offset + 4 * ib32;
            let aux32 = u32::from_le_bytes(
                [x[i].qs[start], x[i].qs[start + 1], x[i].qs[start + 2], x[i].qs[start + 3]],
            );
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            let mut out = i * QK_K + 32 * ib32;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let grid1 = grid4_u32(IQ3XXS_GRID[x[i].qs[8 * ib32 + 2 * l] as usize]);
                let grid2 = grid4_u32(IQ3XXS_GRID[x[i].qs[8 * ib32 + 2 * l + 1] as usize]);
                for j in 0..4 {
                    let s1 = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let s2 = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    y[out + j] = db * grid1[j] as f32 * s1;
                    y[out + j + 4] = db * grid2[j] as f32 * s2;
                }
                out += 8;
            }
        }
    }
    Ok(())
}

pub fn dequantize_row_iq3_s(x: &[BlockIq3S], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_blocks(x, nb)?;
    check_out(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let mut qs = 0usize;
        let mut qh = 0usize;
        let mut signs = 0usize;
        for ib32 in (0..QK_K / 32).step_by(2) {
            let db1 = d * (1.0 + 2.0 * (x[i].scales[ib32 / 2] & 0xF) as f32);
            let db2 = d * (1.0 + 2.0 * (x[i].scales[ib32 / 2] >> 4) as f32);
            let mut out = i * QK_K + 32 * ib32;
            for half in 0..2 {
                let db = if half == 0 { db1 } else { db2 };
                let h = x[i].qh[qh + half];
                for l in 0..4 {
                    let idx1 = x[i].qs[qs + 2 * l] as u32
                        | (((h as u32) << (8 - 2 * l)) & 0x100);
                    let idx2 = x[i].qs[qs + 2 * l + 1] as u32
                        | (((h as u32) << (7 - 2 * l)) & 0x100);
                    let grid1 = grid4_u32(IQ3S_GRID[idx1 as usize]);
                    let grid2 = grid4_u32(IQ3S_GRID[idx2 as usize]);
                    let s = x[i].signs[signs + l];
                    for j in 0..4 {
                        let s1 = if s & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                        let s2 = if s & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                        y[out + j] = db * grid1[j] as f32 * s1;
                        y[out + j + 4] = db * grid2[j] as f32 * s2;
                    }
                    out += 8;
                }
                qs += 8;
                signs += 4;
            }
            qh += 2;
        }
    }
    Ok(())
}

pub fn dequantize_row_iq1_s(x: &[BlockIq1S], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_blocks(x, nb)?;
    check_out(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for ib in 0..QK_K / 32 {
            let qh = x[i].qh[ib];
            let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
            let delta = if qh & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
            let mut out = i * QK_K + 32 * ib;
            for l in 0..4 {
                let idx = x[i].qs[4 * ib + l] as usize
                    | ((((qh >> (3 * l)) & 7) as usize) << 8);
                let grid = grid8_i8(IQ1S_GRID[idx]);
                for j in 0..8 {
                    y[out + j] = dl * (grid[j] as f32 + delta);
                }
                out += 8;
            }
        }
    }
    Ok(())
}

pub fn dequantize_row_iq1_m(x: &[BlockIq1M], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_blocks(x, nb)?;
    check_out(y, k)?;
    for i in 0..nb {
        let sc = [
            u16::from_le_bytes([x[i].scales[0], x[i].scales[1]]),
            u16::from_le_bytes([x[i].scales[2], x[i].scales[3]]),
            u16::from_le_bytes([x[i].scales[4], x[i].scales[5]]),
            u16::from_le_bytes([x[i].scales[6], x[i].scales[7]]),
        ];
        let scale = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        let d = fp16_to_fp32(scale);
        for ib in 0..QK_K / 32 {
            let dl1 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) as f32 + 1.0);
            let dl2 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) as f32 + 1.0);
            let q0 = x[i].qh[2 * ib];
            let q1 = x[i].qh[2 * ib + 1];
            let idx = [
                x[i].qs[4 * ib] as u16 | (((q0 as u16) << 8) & 0x700),
                x[i].qs[4 * ib + 1] as u16 | (((q0 as u16) << 4) & 0x700),
                x[i].qs[4 * ib + 2] as u16 | (((q1 as u16) << 8) & 0x700),
                x[i].qs[4 * ib + 3] as u16 | (((q1 as u16) << 4) & 0x700),
            ];
            let delta = [
                if q0 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                if q0 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                if q1 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                if q1 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
            ];
            let mut out = i * QK_K + 32 * ib;
            for l in 0..2 {
                let grid = grid8_i8(IQ1S_GRID[idx[l] as usize]);
                for j in 0..8 {
                    y[out + j] = dl1 * (grid[j] as f32 + delta[l]);
                }
                out += 8;
            }
            for l in 2..4 {
                let grid = grid8_i8(IQ1S_GRID[idx[l] as usize]);
                for j in 0..8 {
                    y[out + j] = dl2 * (grid[j] as f32 + delta[l]);
                }
                out += 8;
            }
        }
    }
    Ok(())
}

pub fn dequantize_row_iq4_nl(x: &[BlockIq4Nl], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK4_NL)?;
    check_blocks(x, nb)?;
    check_out(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for j in 0..QK4_NL / 2 {
            y[i * QK4_NL + j] = d * KVALUES_IQ4NL[(x[i].qs[j] & 0xF) as usize] as f32;
            y[i * QK4_NL + j + QK4_NL / 2] = d * KVALUES_IQ4NL[(x[i].qs[j] >> 4) as usize] as f32;
        }
    }
    Ok(())
}

pub fn dequantize_row_iq4_xs(x: &[BlockIq4Xs], y: &mut [f32], k: usize) -> Result<()> {
    let nb = check_aligned(k, QK_K)?;
    check_blocks(x, nb)?;
    check_out(y, k)?;
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for ib in 0..QK_K / 32 {
            let ls = ((x[i].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as u32
                | (((x[i].scales_h >> (2 * ib)) & 3) as u32) << 4;
            let dl = d * (ls as i32 - 32) as f32;
            let out = i * QK_K + 32 * ib;
            for j in 0..16 {
                let q = x[i].qs[16 * ib + j];
                y[out + j] = dl * KVALUES_IQ4NL[(q & 0xF) as usize] as f32;
                y[out + j + 16] = dl * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_bytes_are_little_endian() {
        let g = grid8_u64(0x0807_0605_0403_0201);
        assert_eq!(g, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn iq4_nl_dequantizes_zero_scale_to_zero() {
        let x: Vec<f32> = vec![0.0; QK4_NL];
        let mut b = [BlockIq4Nl { d: 0, qs: [0; QK4_NL / 2] }];
        crate::ggml::quants::iq4::quantize_row_iq4_nl_ref(&x, &mut b, QK4_NL).unwrap();
        let mut y = [1.0f32; QK4_NL];
        dequantize_row_iq4_nl(&b, &mut y, QK4_NL).unwrap();
        assert!(y.iter().all(|v| *v == 0.0));
    }
}

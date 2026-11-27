//! Tensor dequantization/quantization used by the driver.
//!
//! `llama-quant.cpp` calls into ggml for the actual kernels. The Rust port has
//! no ggml runtime, so that dependency is expressed as the `Quantizer` trait.
//! `NativeQuantizer` implements the legacy types in pure Rust; other types
//! return a descriptive error so a backend can supply its own implementation.

use crate::core::{Error, Result};

use super::types::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, QType};

/// Equivalent of the ggml type traits used by `llama-quant.cpp`.
pub trait Quantizer {
    /// `qtype->to_float` plus the F16/BF16 conversions.
    fn dequantize_row(&self, ty: QType, src: &[u8], dst: &mut [f32]) -> Result<()>;
    /// `ggml_quantize_chunk` for one row.
    fn quantize_row(
        &self,
        ty: QType,
        src: &[f32],
        dst: &mut [u8],
        imatrix: Option<&[f32]>,
    ) -> Result<usize>;
    /// `ggml_validate_row_data`.
    fn validate_row(&self, ty: QType, data: &[u8]) -> bool;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NativeQuantizer;

/// Pick a scale with a small weighted-MSE search when an imatrix is present.
fn weighted_scale(x: &[f32], weights: Option<&[f32]>, initial: f32, span: f32) -> f32 {
    let Some(w) = weights else {
        return initial;
    };
    if w.len() < x.len() || !initial.is_finite() || initial <= 0.0 {
        return initial;
    }
    let mut best = initial;
    let mut best_err = f64::INFINITY;
    for k in -4..=4 {
        let d = initial * (1.0 + 0.02 * k as f32);
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let mut err = 0.0f64;
        for (j, &v) in x.iter().enumerate() {
            let q = ((v * id).round() as f64).clamp(-span as f64, span as f64 - 1.0);
            let e = v as f64 - q * d as f64;
            err += w[j] as f64 * e * e;
        }
        if err < best_err {
            best_err = err;
            best = d;
        }
    }
    best
}

fn read_f16(src: &[u8], i: usize) -> f32 {
    let b = [src[2 * i], src[2 * i + 1]];
    f16_to_f32(u16::from_le_bytes(b))
}

fn write_f16(dst: &mut [u8], i: usize, value: f32) {
    let b = f32_to_f16(value).to_le_bytes();
    dst[2 * i] = b[0];
    dst[2 * i + 1] = b[1];
}

fn max_abs(x: &[f32]) -> (f32, f32) {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &v in x {
        if amax < v.abs() {
            amax = v.abs();
            max = v;
        }
    }
    (amax, max)
}

fn min_max(x: &[f32]) -> (f32, f32) {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for &v in x {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo, hi)
}

fn quant_q4_0(x: &[f32], dst: &mut [u8], w: Option<&[f32]>) {
    let (_, max) = max_abs(x);
    let d = weighted_scale(x, w, max / -8.0, 8.0);
    let id = if d == 0.0 { 0.0 } else { 1.0 / d };
    write_f16(dst, 0, d);
    for j in 0..16 {
        let a = ((x[j] * id) + 8.5) as i32;
        let b = ((x[16 + j] * id) + 8.5) as i32;
        let qa = a.clamp(0, 15) as u8;
        let qb = b.clamp(0, 15) as u8;
        dst[2 + j] = qa | (qb << 4);
    }
}

fn dequant_q4_0(src: &[u8], dst: &mut [f32]) {
    let d = read_f16(src, 0);
    for j in 0..16 {
        let q = src[2 + j];
        dst[j] = ((q & 0x0f) as i32 - 8) as f32 * d;
        dst[16 + j] = ((q >> 4) as i32 - 8) as f32 * d;
    }
}

fn quant_q4_1(x: &[f32], dst: &mut [u8], w: Option<&[f32]>) {
    let (lo, hi) = min_max(x);
    let d = weighted_scale(x, w, (hi - lo) / 15.0, 15.0);
    let id = if d == 0.0 { 0.0 } else { 1.0 / d };
    let min = if w.is_some() { lo } else { lo };
    write_f16(dst, 0, d);
    write_f16(dst, 1, min);
    for j in 0..16 {
        let a = (((x[j] - min) * id) + 0.5) as i32;
        let b = (((x[16 + j] - min) * id) + 0.5) as i32;
        dst[4 + j] = a.clamp(0, 15) as u8 | ((b.clamp(0, 15) as u8) << 4);
    }
}

fn dequant_q4_1(src: &[u8], dst: &mut [f32]) {
    let d = read_f16(src, 0);
    let m = read_f16(src, 1);
    for j in 0..16 {
        let q = src[4 + j];
        dst[j] = (q & 0x0f) as f32 * d + m;
        dst[16 + j] = (q >> 4) as f32 * d + m;
    }
}

fn quant_q5_0(x: &[f32], dst: &mut [u8], w: Option<&[f32]>) {
    let (_, max) = max_abs(x);
    let d = weighted_scale(x, w, max / -16.0, 16.0);
    let id = if d == 0.0 { 0.0 } else { 1.0 / d };
    write_f16(dst, 0, d);
    let mut qh = 0u32;
    for j in 0..16 {
        let a = ((x[j] * id) + 16.5) as i32;
        let b = ((x[16 + j] * id) + 16.5) as i32;
        let qa = a.clamp(0, 31) as u32;
        let qb = b.clamp(0, 31) as u32;
        dst[6 + j] = ((qa & 0x0f) | ((qb & 0x0f) << 4)) as u8;
        qh |= ((qa >> 4) & 1) << j;
        qh |= ((qb >> 4) & 1) << (j + 16);
    }
    dst[2..6].copy_from_slice(&qh.to_le_bytes());
}

fn dequant_q5_0(src: &[u8], dst: &mut [f32]) {
    let d = read_f16(src, 0);
    let mut qh = [0u8; 4];
    qh.copy_from_slice(&src[2..6]);
    let qh = u32::from_le_bytes(qh);
    for j in 0..16 {
        let q = src[6 + j];
        let qa = (q & 0x0f) as u32 | ((qh >> j) & 1) << 4;
        let qb = (q >> 4) as u32 | ((qh >> (j + 16)) & 1) << 4;
        dst[j] = (qa as i32 - 16) as f32 * d;
        dst[16 + j] = (qb as i32 - 16) as f32 * d;
    }
}

fn quant_q5_1(x: &[f32], dst: &mut [u8], w: Option<&[f32]>) {
    let (lo, hi) = min_max(x);
    let d = weighted_scale(x, w, (hi - lo) / 31.0, 31.0);
    let id = if d == 0.0 { 0.0 } else { 1.0 / d };
    write_f16(dst, 0, d);
    write_f16(dst, 1, lo);
    let mut qh = 0u32;
    for j in 0..16 {
        let a = (((x[j] - lo) * id) + 0.5) as i32;
        let b = (((x[16 + j] - lo) * id) + 0.5) as i32;
        let qa = a.clamp(0, 31) as u32;
        let qb = b.clamp(0, 31) as u32;
        dst[8 + j] = ((qa & 0x0f) | ((qb & 0x0f) << 4)) as u8;
        qh |= ((qa >> 4) & 1) << j;
        qh |= ((qb >> 4) & 1) << (j + 16);
    }
    dst[4..8].copy_from_slice(&qh.to_le_bytes());
}

fn dequant_q5_1(src: &[u8], dst: &mut [f32]) {
    let d = read_f16(src, 0);
    let m = read_f16(src, 1);
    let mut qh = [0u8; 4];
    qh.copy_from_slice(&src[4..8]);
    let qh = u32::from_le_bytes(qh);
    for j in 0..16 {
        let q = src[8 + j];
        let qa = (q & 0x0f) as u32 | ((qh >> j) & 1) << 4;
        let qb = (q >> 4) as u32 | ((qh >> (j + 16)) & 1) << 4;
        dst[j] = qa as f32 * d + m;
        dst[16 + j] = qb as f32 * d + m;
    }
}

fn quant_q8_0(x: &[f32], dst: &mut [u8], w: Option<&[f32]>) {
    let (amax, _) = max_abs(x);
    let d = weighted_scale(x, w, amax / 127.0, 127.0);
    let id = if d == 0.0 { 0.0 } else { 1.0 / d };
    write_f16(dst, 0, d);
    for (j, &v) in x.iter().enumerate().take(32) {
        dst[2 + j] = (v * id).round().clamp(-128.0, 127.0) as i8 as u8;
    }
}

fn dequant_q8_0(src: &[u8], dst: &mut [f32]) {
    let d = read_f16(src, 0);
    for j in 0..32 {
        dst[j] = (src[2 + j] as i8) as f32 * d;
    }
}

impl Quantizer for NativeQuantizer {
    fn dequantize_row(&self, ty: QType, src: &[u8], dst: &mut [f32]) -> Result<()> {
        match ty {
            QType::F32 => {
                if src.len() < dst.len() * 4 {
                    return Err(Error::new("f32 row data is too short"));
                }
                for (i, out) in dst.iter_mut().enumerate() {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&src[i * 4..i * 4 + 4]);
                    *out = f32::from_le_bytes(b);
                }
            }
            QType::F16 => {
                if src.len() < dst.len() * 2 {
                    return Err(Error::new("f16 row data is too short"));
                }
                for (i, out) in dst.iter_mut().enumerate() {
                    *out = f16_to_f32(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]));
                }
            }
            QType::Bf16 => {
                if src.len() < dst.len() * 2 {
                    return Err(Error::new("bf16 row data is too short"));
                }
                for (i, out) in dst.iter_mut().enumerate() {
                    *out = bf16_to_f32(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]));
                }
            }
            QType::Q4_0 | QType::Q4_1 | QType::Q5_0 | QType::Q5_1 | QType::Q8_0 => {
                let bs = ty.blck_size() as usize;
                if dst.len() % bs != 0 {
                    return Err(Error::new(format!(
                        "row length {} is not divisible by block size {}",
                        dst.len(),
                        bs
                    )));
                }
                let row = ty.row_size(dst.len() as i64);
                if src.len() < row {
                    return Err(Error::new(format!("{} row data is too short", ty.name())));
                }
                let blck_bytes = ty.type_size();
                let nblocks = dst.len() / bs;
                for b in 0..nblocks {
                    let s = &src[b * blck_bytes..(b + 1) * blck_bytes];
                    let d = &mut dst[b * bs..(b + 1) * bs];
                    match ty {
                        QType::Q4_0 => dequant_q4_0(s, d),
                        QType::Q4_1 => dequant_q4_1(s, d),
                        QType::Q5_0 => dequant_q5_0(s, d),
                        QType::Q5_1 => dequant_q5_1(s, d),
                        QType::Q8_0 => dequant_q8_0(s, d),
                        _ => unreachable!(),
                    }
                }
            }
            other => {
                return Err(Error::new(format!(
                    "no pure-Rust dequantizer for type {}; supply a Quantizer implementation",
                    other.name()
                )))
            }
        }
        Ok(())
    }

    fn quantize_row(
        &self,
        ty: QType,
        src: &[f32],
        dst: &mut [u8],
        imatrix: Option<&[f32]>,
    ) -> Result<usize> {
        let row = ty.row_size(src.len() as i64);
        if dst.len() < row {
            return Err(Error::new(format!(
                "{} output row buffer is too short",
                ty.name()
            )));
        }
        match ty {
            QType::F32 => {
                for (i, &v) in src.iter().enumerate() {
                    dst[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            QType::F16 => {
                for (i, &v) in src.iter().enumerate() {
                    dst[i * 2..i * 2 + 2].copy_from_slice(&f32_to_f16(v).to_le_bytes());
                }
            }
            QType::Bf16 => {
                for (i, &v) in src.iter().enumerate() {
                    dst[i * 2..i * 2 + 2].copy_from_slice(&f32_to_bf16(v).to_le_bytes());
                }
            }
            QType::Q4_0 | QType::Q4_1 | QType::Q5_0 | QType::Q5_1 | QType::Q8_0 => {
                let bs = ty.blck_size() as usize;
                if src.len() % bs != 0 {
                    return Err(Error::new(format!(
                        "row length {} is not divisible by block size {}",
                        src.len(),
                        bs
                    )));
                }
                let bb = ty.type_size();
                let nblocks = src.len() / bs;
                for b in 0..nblocks {
                    let x = &src[b * bs..(b + 1) * bs];
                    let w = imatrix.and_then(|m| m.get(b * bs..(b + 1) * bs));
                    let out = &mut dst[b * bb..(b + 1) * bb];
                    match ty {
                        QType::Q4_0 => quant_q4_0(x, out, w),
                        QType::Q4_1 => quant_q4_1(x, out, w),
                        QType::Q5_0 => quant_q5_0(x, out, w),
                        QType::Q5_1 => quant_q5_1(x, out, w),
                        QType::Q8_0 => quant_q8_0(x, out, w),
                        _ => unreachable!(),
                    }
                }
            }
            other => {
                return Err(Error::new(format!(
                    "no pure-Rust quantizer for type {}; supply a Quantizer implementation",
                    other.name()
                )))
            }
        }
        Ok(row)
    }

    fn validate_row(&self, ty: QType, data: &[u8]) -> bool {
        let bs = ty.blck_size();
        bs > 0 && ty.type_size() > 0 && data.len() % ty.type_size() == 0
    }
}

/// `llama_tensor_dequantize_impl`: split a slab into thread-sized ranges.
pub fn dequantize_slab(
    quant: &impl Quantizer,
    ty: QType,
    src: &[u8],
    dst: &mut [f32],
    nthread: usize,
) -> Result<()> {
    if ty != QType::F16 && ty != QType::Bf16 && ty != QType::F8E4M3 && !ty.is_quantized() {
        return Err(Error::new(format!(
            "cannot dequantize/convert tensor type {}",
            ty.name()
        )));
    }
    if nthread < 2 {
        return quant.dequantize_row(ty, src, dst);
    }
    let block_size = if ty == QType::F16 || ty == QType::Bf16 {
        1usize
    } else {
        ty.blck_size().max(1) as usize
    };
    let block_bytes = ty.type_size();
    if dst.len() % block_size != 0 {
        return Err(Error::new("element count is not divisible by block size"));
    }
    let nblocks = dst.len() / block_size;
    let per = nblocks / nthread;
    let spare = nblocks - per * nthread;
    let mut in_off = 0;
    let mut out_off = 0;
    for t in 0..nthread {
        let blocks = per + if t == nthread - 1 { spare } else { 0 };
        let elems = blocks * block_size;
        let bytes = blocks * block_bytes;
        quant.dequantize_row(
            ty,
            src.get(in_off..in_off + bytes).unwrap_or(&[]),
            &mut dst[out_off..out_off + elems],
        )?;
        in_off += bytes;
        out_off += elems;
    }
    Ok(())
}

/// `llama_tensor_quantize_impl`: rows are split at expert boundaries.
pub fn quantize_slab(
    quant: &impl Quantizer,
    new_type: QType,
    src: &[f32],
    dst: &mut Vec<u8>,
    first_row: i64,
    nrows: i64,
    nrows_per_expert: i64,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
) -> Result<usize> {
    let row_size = new_type.row_size(n_per_row as i64);
    let mut written = 0usize;
    let mut row = 0i64;
    while row < nrows {
        let global = first_row + row;
        let this_nrow = (nrows - row).min(nrows_per_expert - global % nrows_per_expert.max(1));
        // Quantize row groups so each expert gets its own imatrix slice.
        let end = (row + this_nrow) as usize * row_size;
        if dst.len() < end {
            dst.resize(end, 0);
        }
        for r in 0..this_nrow as usize {
            let row_global = global + r as i64;
            let slice = if let Some(im) = imatrix {
                let idx = (row_global / nrows_per_expert.max(1)) as usize * n_per_row;
                im.get(idx..idx + n_per_row)
            } else {
                None
            };
            let out_begin = (row as usize + r) * row_size;
            let size = quant.quantize_row(
                new_type,
                &src[(row as usize + r) * n_per_row..(row as usize + r + 1) * n_per_row],
                &mut dst[out_begin..end],
                slice,
            )?;
            if !quant.validate_row(new_type, &dst[out_begin..out_begin + size]) {
                return Err(Error::new("quantized data validation failed"));
            }
        }
        written += this_nrow as usize * row_size;
        row += this_nrow;
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_round_trip() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let q = NativeQuantizer;
        let mut buf = vec![0u8; 34];
        let n = q.quantize_row(QType::Q8_0, &x, &mut buf, None).unwrap();
        assert_eq!(n, 34);
        let mut out = vec![0.0f32; 32];
        q.dequantize_row(QType::Q8_0, &buf, &mut out).unwrap();
        for (a, b) in x.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }
}

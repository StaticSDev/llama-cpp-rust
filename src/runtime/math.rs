//! Small numeric helpers for the CPU runtime.

use crate::core::{Error, Result};
use crate::ggml::quants::{
    dequantize_row_q4_K, dequantize_row_q5_K, dequantize_row_q6_K, dequantize_row_q8_0,
    fp16_to_fp32, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8_0, QK8_0, QK_K,
};
use crate::ggml::types::GgmlBf16;
use crate::ggml::types::ggml_compute_bf16_to_fp32;
use crate::gguf::GgmlType;
use crate::runtime::types::Weight;

fn take<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8]> {
    let end = pos.checked_add(n).ok_or_else(|| Error::new("block size overflow"))?;
    let slice = bytes
        .get(*pos..end)
        .ok_or_else(|| Error::new("truncated quantized block"))?;
    *pos = end;
    Ok(slice)
}

fn u16_at(bytes: &[u8], pos: &mut usize) -> Result<u16> {
    let b = take(bytes, pos, 2)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

/// Dequantize one row of a GGUF tensor into `out`.
pub fn dequant_row(ty: GgmlType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let n = out.len();
    match ty {
        GgmlType::F32 => {
            for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        GgmlType::F16 => {
            for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *dst = fp16_to_fp32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        GgmlType::Bf16 => {
            for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                *dst = ggml_compute_bf16_to_fp32(GgmlBf16 { bits });
            }
        }
        GgmlType::Q8_0 => {
            let mut p = 0;
            let mut blocks = Vec::with_capacity(n / QK8_0);
            for _ in 0..n / QK8_0 {
                let d = u16_at(bytes, &mut p)?;
                let qs = take(bytes, &mut p, QK8_0)?;
                let mut arr = [0i8; QK8_0];
                for (dst, src) in arr.iter_mut().zip(qs.iter()) {
                    *dst = *src as i8;
                }
                blocks.push(BlockQ8_0 { d, qs: arr });
            }
            dequantize_row_q8_0(&blocks, out, n)?;
        }
        GgmlType::Q4K => {
            let mut p = 0;
            let mut blocks = Vec::with_capacity(n / QK_K);
            for _ in 0..n / QK_K {
                let d = u16_at(bytes, &mut p)?;
                let dmin = u16_at(bytes, &mut p)?;
                let s = take(bytes, &mut p, 12)?;
                let q = take(bytes, &mut p, QK_K / 2)?;
                let mut scales = [0u8; 12];
                let mut qs = [0u8; QK_K / 2];
                scales.copy_from_slice(s);
                qs.copy_from_slice(q);
                blocks.push(BlockQ4K { d, dmin, scales, qs });
            }
            dequantize_row_q4_K(&blocks, out, n)?;
        }
        GgmlType::Q5K => {
            let mut p = 0;
            let mut blocks = Vec::with_capacity(n / QK_K);
            for _ in 0..n / QK_K {
                let d = u16_at(bytes, &mut p)?;
                let dmin = u16_at(bytes, &mut p)?;
                let s = take(bytes, &mut p, 12)?;
                let h = take(bytes, &mut p, QK_K / 8)?;
                let q = take(bytes, &mut p, QK_K / 2)?;
                let mut scales = [0u8; 12];
                let mut qh = [0u8; QK_K / 8];
                let mut qs = [0u8; QK_K / 2];
                scales.copy_from_slice(s);
                qh.copy_from_slice(h);
                qs.copy_from_slice(q);
                blocks.push(BlockQ5K { d, dmin, scales, qh, qs });
            }
            dequantize_row_q5_K(&blocks, out, n)?;
        }
        GgmlType::Q6K => {
            let mut p = 0;
            let mut blocks = Vec::with_capacity(n / QK_K);
            for _ in 0..n / QK_K {
                let l = take(bytes, &mut p, QK_K / 2)?;
                let h = take(bytes, &mut p, QK_K / 4)?;
                let s = take(bytes, &mut p, QK_K / 16)?;
                let d = u16_at(bytes, &mut p)?;
                let mut ql = [0u8; QK_K / 2];
                let mut qh = [0u8; QK_K / 4];
                let mut scales = [0i8; QK_K / 16];
                ql.copy_from_slice(l);
                qh.copy_from_slice(h);
                for (dst, src) in scales.iter_mut().zip(s.iter()) {
                    *dst = *src as i8;
                }
                blocks.push(BlockQ6K { ql, qh, scales, d });
            }
            dequantize_row_q6_K(&blocks, out, n)?;
        }
        other => return Err(Error::new(format!("unsupported tensor type {other:?}"))),
    }
    Ok(())
}

/// `out = W * x`, dequantizing one row at a time to keep memory low.
pub fn matvec(data: &[u8], w: &Weight, x: &[f32], out: &mut [f32]) -> Result<()> {
    if x.len() != w.in_features() || out.len() != w.out_features() {
        return Err(Error::new("matvec shape mismatch"));
    }

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if threads < 2 || out.len() < 512 {
        return matvec_range(data, w, x, out, 0, out.len());
    }

    // Rows are independent: split the output and dequantize each row in its
    // own worker. This keeps the whole model resident as quantized bytes.
    let chunk = (out.len() + threads - 1) / threads;
    let mut errors = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for (i, part) in out.chunks_mut(chunk).enumerate() {
            let start = i * chunk;
            handles.push(scope.spawn(move || {
                matvec_range(data, w, x, part, start, start + part.len())
            }));
        }
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => errors.push(err),
                Err(_) => errors.push(Error::new("matvec worker panicked")),
            }
        }
    });
    match errors.into_iter().next() {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

fn matvec_range(
    data: &[u8],
    w: &Weight,
    x: &[f32],
    out: &mut [f32],
    first: usize,
    last: usize,
) -> Result<()> {
    let mut row = vec![0.0f32; w.in_features()];
    for r in first..last.min(w.out_features()) {
        dequant_row(w.ty, w.row(data, r)?, &mut row)?;
        let mut sum = 0.0f32;
        for (a, b) in row.iter().zip(x.iter()) {
            sum += a * b;
        }
        out[r - first] = sum;
    }
    Ok(())
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    x.iter()
        .zip(w.iter())
        .map(|(v, g)| v * scale * g)
        .collect()
}

/// NeoX-style RoPE on the first `n_rot` channels. Text MRoPE uses the same
/// position for every axis, so this matches the interleaved variant.
pub fn rope_inplace(x: &mut [f32], pos: usize, n_rot: usize, base: f32) {
    let half = n_rot / 2;
    for i in 0..half {
        let theta = pos as f32 * base.powf(-((2 * i) as f32) / n_rot as f32);
        let c = theta.cos();
        let s = theta.sin();
        let a = x[i];
        let b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

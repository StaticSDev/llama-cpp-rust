//! `ggml_type` stand-ins and tensor helpers used by the quantizer.
//!
//! The Rust port has no ggml runtime, so this module carries the type table
//! needed by `llama-quant.cpp`. Numeric values match the ggml enum used by the
//! reference tree (including MXFP4, NVFP4 and Q1_0/Q2_0).

use crate::core::{Error, Result};

/// `enum ggml_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum QType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
    F8E4M3 = 43,
    Count = 44,
}

pub const TYPE_COUNT: usize = QType::Count as usize;

impl QType {
    pub const TYPE_COUNT: usize = TYPE_COUNT;

    /// `ggml_type_size(type)`: bytes per block.
    pub fn type_size(self) -> usize {
        match self {
            QType::F32 | QType::I32 => 4,
            QType::F16 | QType::Bf16 | QType::I16 => 2,
            QType::I8 | QType::F8E4M3 => 1,
            QType::I64 | QType::F64 => 8,
            QType::Q1_0 => 18,
            QType::Q2_0 => 18,
            QType::Q4_0 | QType::Iq4Nl => 18,
            QType::Q4_1 => 20,
            QType::Q5_0 => 22,
            QType::Q5_1 => 24,
            QType::Q8_0 => 34,
            QType::Q8_1 => 40,
            QType::Q2K => 84,
            QType::Q3K => 110,
            QType::Q4K => 144,
            QType::Q5K => 176,
            QType::Q6K => 210,
            QType::Q8K => 292,
            QType::Iq2Xxs | QType::Iq1S | QType::Iq2S | QType::Iq1M => 65,
            QType::Iq2Xs => 67,
            QType::Iq3Xxs | QType::Iq3S => 79,
            QType::Iq4Xs => 83,
            QType::Tq1_0 => 32,
            QType::Tq2_0 => 64,
            QType::Mxfp4 => 17,
            QType::Nvfp4 => 36,
            QType::Count => 0,
        }
    }

    /// `ggml_blck_size(type)`: elements per block.
    pub fn blck_size(self) -> i64 {
        match self {
            QType::F32
            | QType::F16
            | QType::Bf16
            | QType::F64
            | QType::F8E4M3
            | QType::I8
            | QType::I16
            | QType::I32
            | QType::I64 => 1,
            QType::Q1_0 => 128,
            QType::Q2_0 => 64,
            QType::Q4_0
            | QType::Q4_1
            | QType::Q5_0
            | QType::Q5_1
            | QType::Q8_0
            | QType::Q8_1
            | QType::Iq4Nl
            | QType::Mxfp4 => 32,
            QType::Nvfp4 => 64,
            QType::Q2K
            | QType::Q3K
            | QType::Q4K
            | QType::Q5K
            | QType::Q6K
            | QType::Q8K
            | QType::Iq2Xxs
            | QType::Iq2Xs
            | QType::Iq3Xxs
            | QType::Iq1S
            | QType::Iq3S
            | QType::Iq2S
            | QType::Iq4Xs
            | QType::Iq1M
            | QType::Tq1_0
            | QType::Tq2_0 => 256,
            QType::Count => 0,
        }
    }

    /// `ggml_type_name(type)`.
    pub fn name(self) -> &'static str {
        match self {
            QType::F32 => "f32",
            QType::F16 => "f16",
            QType::Q4_0 => "q4_0",
            QType::Q4_1 => "q4_1",
            QType::Q5_0 => "q5_0",
            QType::Q5_1 => "q5_1",
            QType::Q8_0 => "q8_0",
            QType::Q8_1 => "q8_1",
            QType::Q2K => "q2_K",
            QType::Q3K => "q3_K",
            QType::Q4K => "q4_K",
            QType::Q5K => "q5_K",
            QType::Q6K => "q6_K",
            QType::Q8K => "q8_K",
            QType::Iq2Xxs => "iq2_xxs",
            QType::Iq2Xs => "iq2_xs",
            QType::Iq3Xxs => "iq3_xxs",
            QType::Iq1S => "iq1_s",
            QType::Iq4Nl => "iq4_nl",
            QType::Iq3S => "iq3_s",
            QType::Iq2S => "iq2_s",
            QType::Iq4Xs => "iq4_xs",
            QType::I8 => "i8",
            QType::I16 => "i16",
            QType::I32 => "i32",
            QType::I64 => "i64",
            QType::F64 => "f64",
            QType::Iq1M => "iq1_m",
            QType::Bf16 => "bf16",
            QType::Tq1_0 => "tq1_0",
            QType::Tq2_0 => "tq2_0",
            QType::Mxfp4 => "mxfp4",
            QType::Nvfp4 => "nvfp4",
            QType::Q1_0 => "q1_0",
            QType::Q2_0 => "q2_0",
            QType::F8E4M3 => "f8_e4m3",
            QType::Count => "unknown",
        }
    }

    /// `ggml_is_quantized(type)`.
    pub fn is_quantized(self) -> bool {
        match self {
            QType::I8
            | QType::I16
            | QType::I32
            | QType::I64
            | QType::F32
            | QType::F16
            | QType::F64
            | QType::Bf16
            | QType::F8E4M3
            | QType::Count => false,
            _ => true,
        }
    }

    /// `8.0 * ggml_type_size(t) / ggml_blck_size(t)`.
    pub fn bits_per_weight(self) -> f64 {
        if self.blck_size() == 0 {
            return 0.0;
        }
        8.0 * self.type_size() as f64 / self.blck_size() as f64
    }

    pub fn from_i32(value: i32) -> Option<Self> {
        let all = [
            QType::F32,
            QType::F16,
            QType::Q4_0,
            QType::Q4_1,
            QType::Q5_0,
            QType::Q5_1,
            QType::Q8_0,
            QType::Q8_1,
            QType::Q2K,
            QType::Q3K,
            QType::Q4K,
            QType::Q5K,
            QType::Q6K,
            QType::Q8K,
            QType::Iq2Xxs,
            QType::Iq2Xs,
            QType::Iq3Xxs,
            QType::Iq1S,
            QType::Iq4Nl,
            QType::Iq3S,
            QType::Iq2S,
            QType::Iq4Xs,
            QType::I8,
            QType::I16,
            QType::I32,
            QType::I64,
            QType::F64,
            QType::Iq1M,
            QType::Bf16,
            QType::Tq1_0,
            QType::Tq2_0,
            QType::Mxfp4,
            QType::Nvfp4,
            QType::Q1_0,
            QType::Q2_0,
            QType::F8E4M3,
        ];
        all.into_iter().find(|t| *t as i32 == value)
    }

    /// Parse the names accepted by `--tensor-type` and `--output-tensor-type`.
    pub fn from_name(name: &str) -> Option<Self> {
        let n = name.to_ascii_lowercase();
        let mut t = (0..QType::TYPE_COUNT as i32).filter_map(QType::from_i32);
        t.find(|ty| ty.name() == n)
    }

    /// `ggml_row_size(type, ne0)`.
    pub fn row_size(self, ne0: i64) -> usize {
        let blck = self.blck_size();
        if blck <= 0 || ne0 <= 0 {
            return 0;
        }
        self.type_size() * ne0 as usize / blck as usize
    }
}

/// `fallback_type_for`: map an incompatible target to a block-size-32 type.
pub fn fallback_type_for(target: QType) -> Result<QType> {
    match target {
        QType::Iq1S
        | QType::Iq1M
        | QType::Iq2Xxs
        | QType::Iq2Xs
        | QType::Iq2S
        | QType::Iq3Xxs
        | QType::Iq3S
        | QType::Iq4Xs => Ok(QType::Iq4Nl),
        QType::Q2_0 | QType::Q2K | QType::Q3K | QType::Tq1_0 | QType::Tq2_0 => Ok(QType::Q4_0),
        QType::Q4K => Ok(QType::Q5_0),
        QType::Q5K => Ok(QType::Q5_1),
        QType::Q6K => Ok(QType::Q8_0),
        other if other.blck_size() <= 32 => Ok(QType::Count),
        other => Err(Error::new(format!(
            "no tensor type fallback is defined for type {}",
            other.name()
        ))),
    }
}

/// Minimal `ggml_tensor` equivalent used by the quantizer.
#[derive(Clone, Debug)]
pub struct QTensor {
    pub name: String,
    pub ne: [i64; 4],
    pub ty: QType,
    /// Raw tensor bytes in the source file type.
    pub data: Vec<u8>,
    pub split: u16,
    pub offset: u64,
}

impl QTensor {
    pub fn n_dims(&self) -> usize {
        let mut n = 4;
        while n > 1 && self.ne[n - 1] == 1 {
            n -= 1;
        }
        n
    }

    pub fn nelements(&self) -> usize {
        (self.ne[0] as i128 * self.ne[1] as i128 * self.ne[2] as i128 * self.ne[3] as i128).max(0)
            as usize
    }

    pub fn nrows(&self) -> i64 {
        self.ne[1] * self.ne[2] * self.ne[3]
    }

    pub fn row_size(&self, ty: QType) -> usize {
        ty.row_size(self.ne[0])
    }

    pub fn nbytes(&self, ty: QType) -> usize {
        ty.row_size(self.ne[0]) * self.nrows().max(0) as usize
    }

    pub fn current_nbytes(&self) -> usize {
        self.nbytes(self.ty)
    }
}

/// `ggml_fp16_to_fp32`.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e: i32 = -14;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// `ggml_fp32_to_fp16`.
pub fn f32_to_f16(x: f32) -> u16 {
    let u = x.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let exp = ((u >> 23) & 0xff) as i32 - 127 + 15;
    let mant = u & 0x7f_ffff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (1 - exp);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    if exp >= 31 {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    sign | ((exp as u16) << 10) | ((mant + 0x1000) >> 13) as u16
}

/// `ggml_bf16_to_fp32`.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `ggml_fp32_to_bf16` (round to nearest even).
pub fn f32_to_bf16(x: f32) -> u16 {
    let u = x.to_bits();
    let rounding_bias = 0x7fff + ((u >> 16) & 1);
    ((u.wrapping_add(rounding_bias)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip() {
        for x in [0.0f32, 1.0, -1.0, 0.5, 3.25] {
            let h = f32_to_f16(x);
            assert!((f16_to_f32(h) - x).abs() < 1e-3);
        }
    }

    #[test]
    fn fallback_maps_block_sizes() {
        assert_eq!(fallback_type_for(QType::Q2K).unwrap(), QType::Q4_0);
        assert_eq!(fallback_type_for(QType::Q4K).unwrap(), QType::Q5_0);
        assert_eq!(fallback_type_for(QType::Q8_0).unwrap(), QType::Count);
    }
}

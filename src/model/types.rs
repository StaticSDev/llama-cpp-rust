//! Types declared by `llama-model.h` plus local stand-ins for ggml objects.
//!
//! Pointer-valued `ggml_tensor *` fields become `Option<Tensor>`. A `Tensor`
//! here owns its bytes; no FFI and no global allocator is involved.

use crate::core::Token;
use crate::gguf::GgmlType;

/// `llm_type` from `llama-model.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum LlmType {
    Unknown,
    T14M,
    T17M,
    T22M,
    T33M,
    T47M,
    T60M,
    T70M,
    T80M,
    T109M,
    T137M,
    T140M,
    T149M,
    T160M,
    T190M,
    T220M,
    T230M,
    T250M,
    T256M,
    T270M,
    T335M,
    T350M,
    T360M,
    T395M,
    T410M,
    T450M,
    T475M,
    T558M,
    T700M,
    T770M,
    T780M,
    T950M,
    T03B,
    T05B,
    T06B,
    T08B,
    T1B,
    T1_2B,
    T1_3B,
    T1_4B,
    T1_5B,
    T1_6B,
    T1_7B,
    T1_8B,
    T2B,
    T2_6B,
    T2_8B,
    T2_9B,
    T3B,
    T4B,
    T6B,
    T69B,
    T7B,
    T8B,
    T9B,
    T10B,
    T11B,
    T12B,
    T13B,
    T14B,
    T15B,
    T16B,
    T20B,
    T26B,
    T27B,
    T30B,
    T31B,
    T32B,
    T34B,
    T35B,
    T36B,
    T40B,
    T65B,
    T70B,
    T120B,
    T142B,
    T236B,
    T290B,
    T314B,
    T405B,
    T456B,
    T671B,
    Small,
    Medium,
    Large,
    Xl,
    A17B,
    A27B,
    X8x7B,
    X8x22B,
    X16x12B,
    X16x38B,
    T10B128x366B,
    T57BA14B,
    T816BA42B,
    T78BA35B,
    T17B16E,
    T17B128E,
    A13B,
    T1BA400M,
    T3BA800M,
    T7BA1B,
    T8BA1B,
    T79BA13B,
    T12BA25B,
    T16BA1B,
    T21BA3B,
    T24BA2B,
    T26BA4B,
    T29BA4B,
    T30BA3B,
    T31BA35B,
    T32BA9B,
    T35BA3B,
    T48BA3B,
    T75BA9B,
    T80BA3B,
    A3B,
    T100BA6B,
    T102BA12B,
    T106BA12B,
    T118BA8B,
    T120BA12B,
    T122BA10B,
    T124BA51B,
    T196BA11B,
    T230BA10B,
    T428BA23B,
    T235BA22B,
    T288BA19B,
    T300BA47B,
    T310BA15B,
    T355BA32B,
    T397BA17B,
    T685BA37B,
    T744BA40B,
    T28TA50B,
    T320BA18B,
    E2B,
    E4B,
}

impl LlmType {
    /// `llm_type_name`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Unknown => "?B",
            Self::T14M => "14M",
            Self::T17M => "17M",
            Self::T22M => "22M",
            Self::T33M => "33M",
            Self::T47M => "47M",
            Self::T60M => "60M",
            Self::T70M => "70M",
            Self::T80M => "80M",
            Self::T109M => "109M",
            Self::T137M => "137M",
            Self::T140M => "140M",
            Self::T149M => "149M",
            Self::T160M => "160M",
            Self::T190M => "190M",
            Self::T220M => "220M",
            Self::T230M => "230M",
            Self::T250M => "250M",
            Self::T256M => "256M",
            Self::T270M => "270M",
            Self::T335M => "335M",
            Self::T350M => "350M",
            Self::T360M => "360M",
            Self::T395M => "395M",
            Self::T410M => "410M",
            Self::T450M => "450M",
            Self::T475M => "475M",
            Self::T558M => "558M",
            Self::T700M => "700M",
            Self::T770M => "770M",
            Self::T780M => "780M",
            Self::T950M => "950M",
            Self::T03B => "0.3B",
            Self::T05B => "0.5B",
            Self::T06B => "0.6B",
            Self::T08B => "0.8B",
            Self::T1B => "1B",
            Self::T1_2B => "1.2B",
            Self::T1_3B => "1.3B",
            Self::T1_4B => "1.4B",
            Self::T1_5B => "1.5B",
            Self::T1_6B => "1.6B",
            Self::T1_7B => "1.7B",
            Self::T1_8B => "1.8B",
            Self::T2B => "2B",
            Self::T2_6B => "2.6B",
            Self::T2_8B => "2.8B",
            Self::T2_9B => "2.9B",
            Self::T3B => "3B",
            Self::T4B => "4B",
            Self::T6B => "6B",
            Self::T69B => "6.9B",
            Self::T7B => "7B",
            Self::T8B => "8B",
            Self::T9B => "9B",
            Self::T10B => "10B",
            Self::T11B => "11B",
            Self::T12B => "12B",
            Self::T13B => "13B",
            Self::T14B => "14B",
            Self::T15B => "15B",
            Self::T16B => "16B",
            Self::T20B => "20B",
            Self::T26B => "26B",
            Self::T27B => "27B",
            Self::T30B => "30B",
            Self::T31B => "31B",
            Self::T32B => "32B",
            Self::T34B => "34B",
            Self::T35B => "35B",
            Self::T36B => "36B",
            Self::T40B => "40B",
            Self::T65B => "65B",
            Self::T70B => "70B",
            Self::T120B => "120B",
            Self::T142B => "142B",
            Self::T236B => "236B",
            Self::T290B => "290B",
            Self::T314B => "314B",
            Self::T405B => "405B",
            Self::T456B => "456B",
            Self::T671B => "671B",
            Self::Small => "0.1B",
            Self::Medium => "0.4B",
            Self::Large => "0.8B",
            Self::Xl => "1.5B",
            Self::A17B => "A1.7B",
            Self::A27B => "A2.7B",
            Self::X8x7B => "8x7B",
            Self::X8x22B => "8x22B",
            Self::X16x12B => "16x12B",
            Self::X16x38B => "16x3.8B",
            Self::T10B128x366B => "10B+128x3.66B",
            Self::T57BA14B => "57B.A14B",
            Self::T816BA42B => "816B.A42B",
            Self::T78BA35B => "78B.A3.5B",
            Self::T17B16E => "17Bx16E (Scout)",
            Self::T17B128E => "17Bx128E (Maverick)",
            Self::A13B => "A13B",
            Self::T1BA400M => "1B.A400M",
            Self::T3BA800M => "3B.A800M",
            Self::T7BA1B => "7B.A1B",
            Self::T8BA1B => "8B.A1B",
            Self::T79BA13B => "7.9B.A1.3B",
            Self::T12BA25B => "12B.A2.5B",
            Self::T16BA1B => "16B.A1B",
            Self::T21BA3B => "21B.A3B",
            Self::T24BA2B => "24B.A2B",
            Self::T26BA4B => "26B.A4B",
            Self::T29BA4B => "29B.A4B",
            Self::T30BA3B => "30B.A3B",
            Self::T31BA35B => "31B.A3.5B",
            Self::T32BA9B => "32B.A9B",
            Self::T35BA3B => "35B.A3B",
            Self::T48BA3B => "48B.A3B",
            Self::T75BA9B => "75B.A9B",
            Self::T80BA3B => "80B.A3B",
            Self::A3B => "A3B",
            Self::T100BA6B => "100B.A6B",
            Self::T102BA12B => "102B.A12B",
            Self::T106BA12B => "106B.A12B",
            Self::T118BA8B => "118B.A8B",
            Self::T120BA12B => "120B.A12B",
            Self::T122BA10B => "122B.A10B",
            Self::T124BA51B => "124B.A5.1B",
            Self::T196BA11B => "196B.A11B",
            Self::T230BA10B => "230B.A10B",
            Self::T428BA23B => "428B.A23B",
            Self::T235BA22B => "235B.A22B",
            Self::T288BA19B => "288B.A19B",
            Self::T300BA47B => "300B.A47B",
            Self::T310BA15B => "310B.A15B",
            Self::T355BA32B => "355B.A32B",
            Self::T397BA17B => "397B.A17B",
            Self::T685BA37B => "685B.A37B",
            Self::T744BA40B => "744B.A40B",
            Self::T28TA50B => "2.8T.A50B",
            Self::T320BA18B => "320B.A18B",
            Self::E2B => "E2B",
            Self::E4B => "E4B",
        }
    }
}

/// `llama_ftype` is an integer enum in `llama.h`; keep the raw value so new
/// GGUF file types do not need a source change.
pub type Ftype = i32;

pub const LLAMA_FTYPE_ALL_F32: Ftype = 0;

pub fn ftype_name(ftype: Ftype) -> &'static str {
    match ftype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        39 => "NVFP4",
        _ => "unknown",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitMode {
    None,
    Layer,
    Row,
    Expert,
    Tensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadMode {
    Auto,
    Mmap,
    MmapMlock,
    Mlock,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LazyMode {
    Auto,
    On,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeType {
    None,
    Normal,
    NeoX,
    Mrope,
    Imrope,
}

impl From<crate::hparams::RopeType> for RopeType {
    fn from(value: crate::hparams::RopeType) -> Self {
        match value {
            crate::hparams::RopeType::None => Self::None,
            crate::hparams::RopeType::Normal => Self::Normal,
            crate::hparams::RopeType::Mrope => Self::Mrope,
            crate::hparams::RopeType::Imrope => Self::Imrope,
        }
    }
}

impl RopeType {
    /// Convert back to the hparams enum; `NeoX` is only used by indexers.
    pub fn to_hparams(self) -> crate::hparams::RopeType {
        match self {
            Self::None => crate::hparams::RopeType::None,
            Self::Normal | Self::NeoX => crate::hparams::RopeType::Normal,
            Self::Mrope => crate::hparams::RopeType::Mrope,
            Self::Imrope => crate::hparams::RopeType::Imrope,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeScalingType {
    Unspecified,
    None,
    Linear,
    Yarn,
    Longrope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DflashVariant {
    None,
    Dflash,
    Dflash2,
    Dspark,
    Lilicorr,
}

/// Activation precision requested by `llama_prec_policy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Undefined,
    F32,
    F16,
    Bf16,
    Q8,
}

/// Flags accepted by `llama_model_loader::create_tensor`.
pub mod tensor_flags {
    pub const DUPLICATED: u32 = 1 << 0;
    pub const NOT_REQUIRED: u32 = 1 << 1;
    pub const SKIP: u32 = 1 << 2;
    pub const SKIP_IF_VIRTUAL: u32 = 1 << 3;
    pub const ALLOW_RESHAPE: u32 = 1 << 4;
    pub const READ_LAZY: u32 = 1 << 5;
}

/// Owned stand-in for a `ggml_tensor`.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub ne: Vec<i64>,
    pub ty: GgmlType,
    pub data: Vec<u8>,
    pub buffer: Option<String>,
}

impl Tensor {
    pub fn new(name: impl Into<String>, ne: &[i64], ty: GgmlType) -> Self {
        let mut dims = ne.to_vec();
        while dims.len() < 4 {
            dims.push(1);
        }
        dims.truncate(4);
        Self {
            name: name.into(),
            ne: dims,
            ty,
            data: Vec::new(),
            buffer: None,
        }
    }

    pub fn n_dims(&self) -> usize {
        let mut n = 4;
        while n > 1 && self.ne[n - 1] == 1 {
            n -= 1;
        }
        n
    }

    pub fn nelements(&self) -> usize {
        self.ne.iter().map(|&n| n.max(0) as usize).product()
    }

    /// `ggml_nbytes`: quantized types store several elements per block.
    pub fn nbytes(&self) -> usize {
        let blck = crate::gguf::ggml_blck_size(self.ty).max(1) as usize;
        let type_size = crate::gguf::ggml_type_size(self.ty);
        let elements = self.nelements();
        if type_size == 0 {
            return 0;
        }
        (elements / blck) * type_size
    }

    /// `ggml_row_size`: rows are block-aligned.
    pub fn row_size(&self) -> usize {
        let blck = crate::gguf::ggml_blck_size(self.ty).max(1) as usize;
        let type_size = crate::gguf::ggml_type_size(self.ty);
        let n = self.ne[0].max(0) as usize;
        ((n + blck - 1) / blck) * type_size
    }

    /// `llama_tensor_rows_to_f32`. F16 and BF16 are decoded; quantized types
    /// are reported as unsupported instead of silently returning zeros.
    pub fn rows_to_f32(&self, row0: usize, n_rows: usize) -> crate::core::Result<Vec<f32>> {
        let n = self.ne[0].max(0) as usize;
        let nelements = n_rows
            .checked_mul(n)
            .ok_or_else(|| crate::core::Error::new("tensor row count overflow"))?;
        let row_bytes = self.row_size();
        let start = row0
            .checked_mul(row_bytes)
            .ok_or_else(|| crate::core::Error::new("tensor row offset overflow"))?;
        let end = start
            .checked_add(n_rows.checked_mul(row_bytes).unwrap_or(usize::MAX))
            .ok_or_else(|| crate::core::Error::new("tensor row range overflow"))?;
        let bytes = self
            .data
            .get(start..end.min(self.data.len()))
            .ok_or_else(|| {
                crate::core::Error::new(format!(
                    "tensor '{}' has no data for rows {}..{}",
                    self.name,
                    row0,
                    row0 + n_rows
                ))
            })?;
        let mut out = Vec::with_capacity(nelements);
        match self.ty {
            GgmlType::F32 => {
                for chunk in bytes.chunks_exact(4).take(nelements) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            GgmlType::F16 => {
                for chunk in bytes.chunks_exact(2).take(nelements) {
                    let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(f16_to_f32(h));
                }
            }
            GgmlType::Bf16 => {
                for chunk in bytes.chunks_exact(2).take(nelements) {
                    let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(f32::from_bits((h as u32) << 16));
                }
            }
            _ => {
                return Err(crate::core::Error::new(format!(
                    "unsupported tensor type for dequantization: {:?}",
                    self.ty
                )));
            }
        }
        if out.len() != nelements {
            return Err(crate::core::Error::new(format!(
                "short tensor data for '{}': {} of {} elements",
                self.name,
                out.len(),
                nelements
            )));
        }
        Ok(out)
    }
}

/// IEEE 754 binary16 to f32, without a dependency.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = -14i32;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            sign | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | (((exp + 112) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// `llama_layer_posnet`.
#[derive(Clone, Debug, Default)]
pub struct LayerPosnet {
    pub norm1: Option<Tensor>,
    pub norm1_b: Option<Tensor>,
    pub conv1: Option<Tensor>,
    pub conv1_b: Option<Tensor>,
    pub norm2: Option<Tensor>,
    pub norm2_b: Option<Tensor>,
    pub conv2: Option<Tensor>,
    pub conv2_b: Option<Tensor>,
    pub attn_norm: Option<Tensor>,
    pub attn_norm_b: Option<Tensor>,
    pub attn_q: Option<Tensor>,
    pub attn_q_b: Option<Tensor>,
    pub attn_k: Option<Tensor>,
    pub attn_k_b: Option<Tensor>,
    pub attn_v: Option<Tensor>,
    pub attn_v_b: Option<Tensor>,
    pub attn_o: Option<Tensor>,
    pub attn_o_b: Option<Tensor>,
    pub norm: Option<Tensor>,
    pub norm_b: Option<Tensor>,
}

/// `llama_layer_convnext`.
#[derive(Clone, Debug, Default)]
pub struct LayerConvnext {
    pub dw: Option<Tensor>,
    pub dw_b: Option<Tensor>,
    pub norm: Option<Tensor>,
    pub norm_b: Option<Tensor>,
    pub pw1: Option<Tensor>,
    pub pw1_b: Option<Tensor>,
    pub pw2: Option<Tensor>,
    pub pw2_b: Option<Tensor>,
    pub gamma: Option<Tensor>,
}

/// `llama_layer_shortconv`.
#[derive(Clone, Debug, Default)]
pub struct LayerShortconv {
    pub in_proj: Option<Tensor>,
    pub conv: Option<Tensor>,
    pub out_proj: Option<Tensor>,
}

/// `llama_layer_zaya1`.
#[derive(Clone, Debug, Default)]
pub struct LayerZaya1 {
    pub cca_conv_grp: Option<Tensor>,
    pub cca_conv_grp_b: Option<Tensor>,
    pub cca_temp: Option<Tensor>,
    pub attn_res_hs: Option<Tensor>,
    pub ffn_res_hs: Option<Tensor>,
    pub router_eda: Option<Tensor>,
    pub router_norm: Option<Tensor>,
    pub router_fc1: Option<Tensor>,
    pub router_fc1_b: Option<Tensor>,
    pub router_fc2: Option<Tensor>,
    pub router_fc2_b: Option<Tensor>,
    pub router_out: Option<Tensor>,
}

/// `llama_layer_nextn`.
#[derive(Clone, Debug, Default)]
pub struct LayerNextn {
    pub eh_proj: Option<Tensor>,
    pub eh_proj_s: Option<Tensor>,
    pub eh_proj_in_s: Option<Tensor>,
    pub embed_tokens: Option<Tensor>,
    pub enorm: Option<Tensor>,
    pub hnorm: Option<Tensor>,
    pub shared_head_head: Option<Tensor>,
    pub shared_head_head_s: Option<Tensor>,
    pub shared_head_head_in_s: Option<Tensor>,
    pub shared_head_norm: Option<Tensor>,
    pub hc_head_norm: Option<Tensor>,
    pub hc_head_down: Option<Tensor>,
    pub hc_head_up: Option<Tensor>,
}

/// `llama_layer_switch_lora`.
#[derive(Clone, Debug, Default)]
pub struct LayerSwitchLora {
    pub a_q: Option<Tensor>,
    pub b_q: Option<Tensor>,
    pub a_k: Option<Tensor>,
    pub b_k: Option<Tensor>,
    pub a_v: Option<Tensor>,
    pub b_v: Option<Tensor>,
    pub a_o: Option<Tensor>,
    pub b_o: Option<Tensor>,
    pub a_gate: Option<Tensor>,
    pub b_gate: Option<Tensor>,
    pub a_up: Option<Tensor>,
    pub b_up: Option<Tensor>,
    pub a_down: Option<Tensor>,
    pub b_down: Option<Tensor>,
}

/// `llama_layer` from `llama-model.h`.
#[derive(Clone, Debug, Default)]
pub struct Layer {
    // normalization
    pub attn_norm: Option<Tensor>,
    pub attn_norm_b: Option<Tensor>,
    pub attn_norm_2: Option<Tensor>,
    pub attn_norm_2_b: Option<Tensor>,
    pub attn_q_norm: Option<Tensor>,
    pub attn_q_norm_b: Option<Tensor>,
    pub attn_k_norm: Option<Tensor>,
    pub attn_k_norm_b: Option<Tensor>,
    pub attn_out_norm: Option<Tensor>,
    pub attn_out_norm_b: Option<Tensor>,
    pub attn_q_a_norm: Option<Tensor>,
    pub attn_kv_a_norm: Option<Tensor>,
    pub attn_sub_norm: Option<Tensor>,
    pub attn_post_norm: Option<Tensor>,
    pub ffn_sub_norm: Option<Tensor>,
    pub attn_norm_cross: Option<Tensor>,
    pub attn_norm_enc: Option<Tensor>,
    pub ssm_norm: Option<Tensor>,
    pub ssm_dt_norm: Option<Tensor>,
    pub ssm_b_norm: Option<Tensor>,
    pub ssm_c_norm: Option<Tensor>,

    // attention
    pub wq: Option<Tensor>,
    pub wk: Option<Tensor>,
    pub wv: Option<Tensor>,
    pub k_cache_s: Option<Tensor>,
    pub v_cache_s: Option<Tensor>,
    pub wo: Option<Tensor>,
    pub wqkv: Option<Tensor>,
    pub wg: Option<Tensor>,
    pub wq_a: Option<Tensor>,
    pub wq_b: Option<Tensor>,
    pub wkv_a_mqa: Option<Tensor>,
    pub wkv_b: Option<Tensor>,
    pub wkv: Option<Tensor>,
    pub wk_b: Option<Tensor>,
    pub wv_b: Option<Tensor>,
    pub wqkv_b: Option<Tensor>,
    pub wo_a: Option<Tensor>,
    pub wo_b: Option<Tensor>,
    pub wq_cross: Option<Tensor>,
    pub wk_cross: Option<Tensor>,
    pub wv_cross: Option<Tensor>,
    pub wo_cross: Option<Tensor>,
    pub wq_enc: Option<Tensor>,
    pub wk_enc: Option<Tensor>,
    pub wv_enc: Option<Tensor>,
    pub wo_enc: Option<Tensor>,
    pub wqkv_gate: Option<Tensor>,
    pub wqkv_gate_b: Option<Tensor>,
    pub attn_v_gate: Option<Tensor>,
    pub attn_v_gate_b: Option<Tensor>,
    pub attn_v_exps: Option<Tensor>,

    // relative position bias
    pub attn_rel_b: Option<Tensor>,
    pub attn_rel_b_enc: Option<Tensor>,
    pub attn_rel_b_cross: Option<Tensor>,

    // FFN normalization
    pub ffn_norm: Option<Tensor>,
    pub ffn_norm_b: Option<Tensor>,
    pub ffn_post_norm: Option<Tensor>,
    pub ffn_post_norm_1: Option<Tensor>,
    pub ffn_post_norm_2: Option<Tensor>,
    pub ffn_pre_norm_2: Option<Tensor>,
    pub ffn_norm_ada_down: Option<Tensor>,
    pub ffn_norm_ada_up: Option<Tensor>,
    pub layer_out_norm: Option<Tensor>,
    pub layer_out_norm_b: Option<Tensor>,
    pub ffn_norm_exps: Option<Tensor>,
    pub ffn_norm_enc: Option<Tensor>,

    // dense FFN
    pub ffn_gate: Option<Tensor>,
    pub ffn_down: Option<Tensor>,
    pub ffn_up: Option<Tensor>,
    pub ffn_gate_enc: Option<Tensor>,
    pub ffn_down_enc: Option<Tensor>,
    pub ffn_up_enc: Option<Tensor>,

    // MoE FFN
    pub ffn_gate_inp: Option<Tensor>,
    pub ffn_gate_inp_s: Option<Tensor>,
    pub ffn_gate_exps: Option<Tensor>,
    pub ffn_down_exps: Option<Tensor>,
    pub ffn_up_exps: Option<Tensor>,
    pub ffn_gate_up_exps: Option<Tensor>,
    pub ffn_gate_inp_b: Option<Tensor>,
    pub ffn_gate_exps_b: Option<Tensor>,
    pub ffn_down_exps_b: Option<Tensor>,
    pub ffn_up_exps_b: Option<Tensor>,
    pub ffn_gate_up_exps_b: Option<Tensor>,
    pub ffn_gate_exps_s: Option<Tensor>,
    pub ffn_down_exps_s: Option<Tensor>,
    pub ffn_up_exps_s: Option<Tensor>,
    pub ffn_latent_down: Option<Tensor>,
    pub ffn_latent_up: Option<Tensor>,
    pub ffn_pre_exp_norm: Option<Tensor>,
    pub ffn_post_exp_norm: Option<Tensor>,

    // residual coefficients
    pub attn_resid_alpha: Option<Tensor>,
    pub attn_resid_beta: Option<Tensor>,
    pub ffn_resid_alpha: Option<Tensor>,
    pub ffn_resid_beta: Option<Tensor>,
    pub attn_vres_lambda: Option<Tensor>,

    // shared and adjugate experts
    pub ffn_gate_inp_shexp: Option<Tensor>,
    pub ffn_gate_shexp: Option<Tensor>,
    pub ffn_down_shexp: Option<Tensor>,
    pub ffn_up_shexp: Option<Tensor>,
    pub ffn_gate_chexps: Option<Tensor>,
    pub ffn_down_chexps: Option<Tensor>,
    pub ffn_up_chexps: Option<Tensor>,

    // FFN biases
    pub ffn_gate_b: Option<Tensor>,
    pub ffn_down_b: Option<Tensor>,
    pub ffn_up_b: Option<Tensor>,
    pub ffn_act: Option<Tensor>,
    pub ffn_exp_probs_b: Option<Tensor>,
    pub ffn_exp_probs_b_vl: Option<Tensor>,
    pub ffn_gate_tid2eid: Option<Tensor>,

    // dflash
    pub dflash_attn_conv_base: Option<Tensor>,
    pub dflash_attn_conv_proj: Option<Tensor>,
    pub dflash_ffn_conv_base: Option<Tensor>,
    pub dflash_ffn_conv_proj: Option<Tensor>,
    pub dflash_ctx_mix: Option<Tensor>,
    pub dflash_ctx_norm: Option<Tensor>,
    pub dflash_wk_ctx: Option<Tensor>,
    pub dflash_wv_ctx: Option<Tensor>,
    pub dflash_attn_k_ctx_norm: Option<Tensor>,

    // lilicorr
    pub lilicorr_attn_norm: Option<Tensor>,
    pub lilicorr_mlp_norm: Option<Tensor>,
    pub lilicorr_attn_in_proj: Option<Tensor>,
    pub lilicorr_attn_in_proj_b: Option<Tensor>,
    pub lilicorr_attn_out_proj: Option<Tensor>,
    pub lilicorr_attn_out_proj_b: Option<Tensor>,
    pub lilicorr_attn_out_proj_s: Option<Tensor>,
    pub lilicorr_mlp_0: Option<Tensor>,
    pub lilicorr_mlp_0_b: Option<Tensor>,
    pub lilicorr_mlp_0_s: Option<Tensor>,
    pub lilicorr_mlp_2: Option<Tensor>,
    pub lilicorr_mlp_2_b: Option<Tensor>,
    pub lilicorr_mlp_2_s: Option<Tensor>,

    // Mamba projections and state
    pub ssm_in: Option<Tensor>,
    pub ssm_x: Option<Tensor>,
    pub ssm_dt: Option<Tensor>,
    pub ssm_out: Option<Tensor>,
    pub ssm_conv1d: Option<Tensor>,
    pub ssm_a: Option<Tensor>,
    pub ssm_d: Option<Tensor>,
    pub ssm_conv1d_b: Option<Tensor>,
    pub ssm_dt_b: Option<Tensor>,
    pub ssm_beta_alpha: Option<Tensor>,
    pub ssm_alpha: Option<Tensor>,

    // RWKV
    pub time_mix_w1: Option<Tensor>,
    pub time_mix_w2: Option<Tensor>,
    pub time_mix_lerp_x: Option<Tensor>,
    pub time_mix_lerp_w: Option<Tensor>,
    pub time_mix_lerp_k: Option<Tensor>,
    pub time_mix_lerp_v: Option<Tensor>,
    pub time_mix_lerp_r: Option<Tensor>,
    pub time_mix_lerp_g: Option<Tensor>,
    pub time_mix_lerp_fused: Option<Tensor>,
    pub time_mix_first: Option<Tensor>,
    pub time_mix_decay: Option<Tensor>,
    pub time_mix_decay_w1: Option<Tensor>,
    pub time_mix_decay_w2: Option<Tensor>,
    pub time_mix_key: Option<Tensor>,
    pub time_mix_key_b: Option<Tensor>,
    pub time_mix_value: Option<Tensor>,
    pub time_mix_value_b: Option<Tensor>,
    pub time_mix_receptance: Option<Tensor>,
    pub time_mix_receptance_b: Option<Tensor>,
    pub time_mix_gate: Option<Tensor>,
    pub time_mix_w0: Option<Tensor>,
    pub time_mix_a0: Option<Tensor>,
    pub time_mix_a1: Option<Tensor>,
    pub time_mix_a2: Option<Tensor>,
    pub time_mix_v0: Option<Tensor>,
    pub time_mix_v1: Option<Tensor>,
    pub time_mix_v2: Option<Tensor>,
    pub time_mix_g1: Option<Tensor>,
    pub time_mix_g2: Option<Tensor>,
    pub time_mix_k_k: Option<Tensor>,
    pub time_mix_k_a: Option<Tensor>,
    pub time_mix_r_k: Option<Tensor>,
    pub time_mix_ln: Option<Tensor>,
    pub time_mix_ln_b: Option<Tensor>,
    pub time_mix_output: Option<Tensor>,
    pub channel_mix_lerp_k: Option<Tensor>,
    pub channel_mix_lerp_r: Option<Tensor>,
    pub channel_mix_key: Option<Tensor>,
    pub channel_mix_receptance: Option<Tensor>,
    pub channel_mix_value: Option<Tensor>,

    // long rope factors
    pub rope_long: Option<Tensor>,
    pub rope_short: Option<Tensor>,
    pub rope_freqs: Option<Tensor>,

    // bitnet and input scales
    pub wq_s: Option<Tensor>,
    pub wk_s: Option<Tensor>,
    pub wv_s: Option<Tensor>,
    pub wo_s: Option<Tensor>,
    pub wqkv_s: Option<Tensor>,
    pub wqkv_gate_s: Option<Tensor>,
    pub ffn_gate_s: Option<Tensor>,
    pub ffn_up_s: Option<Tensor>,
    pub ffn_down_s: Option<Tensor>,
    pub ffn_gate_shexp_s: Option<Tensor>,
    pub ffn_up_shexp_s: Option<Tensor>,
    pub ffn_down_shexp_s: Option<Tensor>,
    pub ssm_in_s: Option<Tensor>,
    pub ssm_out_s: Option<Tensor>,
    pub ssm_alpha_s: Option<Tensor>,
    pub ssm_beta_s: Option<Tensor>,
    pub wq_in_s: Option<Tensor>,
    pub wk_in_s: Option<Tensor>,
    pub wv_in_s: Option<Tensor>,
    pub wo_in_s: Option<Tensor>,
    pub wqkv_in_s: Option<Tensor>,
    pub wqkv_gate_in_s: Option<Tensor>,
    pub ffn_gate_in_s: Option<Tensor>,
    pub ffn_up_in_s: Option<Tensor>,
    pub ffn_down_in_s: Option<Tensor>,
    pub ffn_gate_exps_in_s: Option<Tensor>,
    pub ffn_down_exps_in_s: Option<Tensor>,
    pub ffn_up_exps_in_s: Option<Tensor>,
    pub ffn_gate_shexp_in_s: Option<Tensor>,
    pub ffn_up_shexp_in_s: Option<Tensor>,
    pub ffn_down_shexp_in_s: Option<Tensor>,
    pub ssm_in_in_s: Option<Tensor>,
    pub ssm_out_in_s: Option<Tensor>,
    pub ssm_alpha_in_s: Option<Tensor>,
    pub ssm_beta_in_s: Option<Tensor>,

    // altup and laurel
    pub per_layer_inp_gate: Option<Tensor>,
    pub per_layer_proj: Option<Tensor>,
    pub per_layer_post_norm: Option<Tensor>,
    pub altup_correct_coef: Option<Tensor>,
    pub altup_correct_scale: Option<Tensor>,
    pub altup_predict_coef: Option<Tensor>,
    pub altup_router: Option<Tensor>,
    pub altup_router_norm: Option<Tensor>,
    pub laurel_l: Option<Tensor>,
    pub laurel_r: Option<Tensor>,
    pub laurel_post_norm: Option<Tensor>,
    pub attn_sinks: Option<Tensor>,

    // DeepSeek-V4
    pub attn_kv_norm: Option<Tensor>,
    pub hc_attn_fn: Option<Tensor>,
    pub hc_attn_base: Option<Tensor>,
    pub hc_attn_scale: Option<Tensor>,
    pub hc_ffn_fn: Option<Tensor>,
    pub hc_ffn_base: Option<Tensor>,
    pub hc_ffn_scale: Option<Tensor>,
    pub attn_comp_wkv: Option<Tensor>,
    pub attn_comp_wgate: Option<Tensor>,
    pub attn_comp_ape: Option<Tensor>,
    pub attn_comp_norm: Option<Tensor>,
    pub indexer_comp_wkv: Option<Tensor>,
    pub indexer_comp_wgate: Option<Tensor>,
    pub indexer_comp_ape: Option<Tensor>,
    pub indexer_comp_norm: Option<Tensor>,

    // cogvlm
    pub visexp_attn_wqkv: Option<Tensor>,
    pub visexp_attn_wo: Option<Tensor>,
    pub visexp_ffn_gate: Option<Tensor>,
    pub visexp_ffn_down: Option<Tensor>,
    pub visexp_ffn_up: Option<Tensor>,

    // xIELU
    pub ffn_act_alpha_n: Option<Tensor>,
    pub ffn_act_alpha_p: Option<Tensor>,
    pub ffn_act_beta: Option<Tensor>,
    pub ffn_act_eps: Option<Tensor>,

    // Kimi KDA
    pub ssm_q_conv: Option<Tensor>,
    pub ssm_k_conv: Option<Tensor>,
    pub ssm_v_conv: Option<Tensor>,
    pub ssm_f_a: Option<Tensor>,
    pub ssm_f_b: Option<Tensor>,
    pub ssm_beta: Option<Tensor>,
    pub ssm_rda: Option<Tensor>,
    pub ssm_g_a: Option<Tensor>,
    pub ssm_g_b: Option<Tensor>,
    pub ssm_o_norm: Option<Tensor>,
    pub ssm_g: Option<Tensor>,
    pub attn_res_score: Option<Tensor>,
    pub ffn_res_score: Option<Tensor>,
    pub ffn_routed_down: Option<Tensor>,
    pub ffn_routed_up: Option<Tensor>,
    pub ffn_routed_norm: Option<Tensor>,

    // DSA / indexer
    pub indexer_k_norm: Option<Tensor>,
    pub indexer_k_norm_b: Option<Tensor>,
    pub indexer_proj: Option<Tensor>,
    pub indexer_attn_k: Option<Tensor>,
    pub indexer_attn_q_b: Option<Tensor>,
    pub indexer_q_norm: Option<Tensor>,
    pub indexer_q_norm_b: Option<Tensor>,
    pub attn_ssmax_s: Option<Tensor>,
    pub attn_ssmax_b: Option<Tensor>,
    pub indexer_ssmax_s: Option<Tensor>,
    pub indexer_ssmax_b: Option<Tensor>,
    pub indexer_kpool_gate: Option<Tensor>,
    pub indexer_kpool_ape: Option<Tensor>,
    pub indexer_kpool_value: Option<Tensor>,

    // deepseek41
    pub engram_kv: Option<Tensor>,
    pub engram_gate: Option<Tensor>,
    pub engram_kv_b: Option<Tensor>,
    pub engram_lookup_norm: Option<Tensor>,
    pub engram_pre_gamma: Option<Tensor>,
    pub engram_hc_scale: Option<Tensor>,

    // MSA
    pub index_q_proj: Option<Tensor>,
    pub index_k_proj: Option<Tensor>,
    pub index_q_norm: Option<Tensor>,
    pub index_k_norm: Option<Tensor>,

    // hyper-connections
    pub hc_attn_norm: Option<Tensor>,
    pub hc_attn_down: Option<Tensor>,
    pub hc_attn_up: Option<Tensor>,
    pub hc_attn_inject: Option<Tensor>,
    pub hc_ffn_norm: Option<Tensor>,
    pub hc_ffn_down: Option<Tensor>,
    pub hc_ffn_up: Option<Tensor>,
    pub hc_ffn_inject: Option<Tensor>,
    pub hc_attn_down_inject: Option<Tensor>,
    pub hc_ffn_down_inject: Option<Tensor>,
    pub wqkv_z: Option<Tensor>,
    pub ssm_beta_alpha_cat: Option<Tensor>,

    // PLE
    pub ple_key: Option<Tensor>,
    pub ple_value: Option<Tensor>,
    pub ple_norm_key: Option<Tensor>,
    pub ple_norm_query: Option<Tensor>,
    pub ple_norm_conv: Option<Tensor>,
    pub ple_conv1d: Option<Tensor>,

    pub out_scale: Option<Tensor>,

    // metis16
    pub hc_attn_mix: Option<Tensor>,
    pub hc_attn_read: Option<Tensor>,
    pub hc_attn_write: Option<Tensor>,
    pub hc_ffn_mix: Option<Tensor>,
    pub hc_ffn_read: Option<Tensor>,
    pub hc_ffn_write: Option<Tensor>,
    pub attn_pass_lora_down: Option<Tensor>,
    pub attn_pass_lora_up: Option<Tensor>,
    pub attn_pass_gate: Option<Tensor>,
    pub ffn_exp_embd: Option<Tensor>,

    // limite
    pub attn_ve_gate: Option<Tensor>,
    pub attn_xsa_alpha: Option<Tensor>,
    pub attn_resid_scale: Option<Tensor>,
    pub attn_resid_scale_b: Option<Tensor>,
    pub ffn_resid_scale: Option<Tensor>,
    pub ffn_resid_scale_b: Option<Tensor>,
    pub mudd_down: Option<Tensor>,
    pub mudd_up: Option<Tensor>,
    pub mudd_up_b: Option<Tensor>,

    // inkling2
    pub attn_rel: Option<Tensor>,
    pub attn_rel_bias: Option<Tensor>,
    pub attn_q_clip: Option<Tensor>,
    pub attn_rel_clip: Option<Tensor>,
    pub attn_sink_proj: Option<Tensor>,
    pub attn_q_inp_norm: Option<Tensor>,
    pub attn_k_inp_norm: Option<Tensor>,
    pub attn_v_inp_norm: Option<Tensor>,
    pub attn_k_conv: Option<Tensor>,
    pub attn_v_conv: Option<Tensor>,
    pub attn_k_conv_proj: Option<Tensor>,
    pub attn_v_conv_proj: Option<Tensor>,
    pub attn_post_conv: Option<Tensor>,
    pub attn_res_query: Option<Tensor>,
    pub ffn_post_conv: Option<Tensor>,
    pub ffn_res_query: Option<Tensor>,

    pub posnet: LayerPosnet,
    pub convnext: LayerConvnext,
    pub shortconv: LayerShortconv,
    pub zaya1: LayerZaya1,
    pub nextn: LayerNextn,
    pub switch_lora: LayerSwitchLora,
}

/// `llama_rope_params`.
#[derive(Clone, Debug)]
pub struct RopeParams {
    pub n_rot: i32,
    pub n_offs: i32,
    pub rope_type: RopeType,
    pub n_ctx_orig: i32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub factors: Option<String>,
    pub k_norm: Option<String>,
}

impl Default for RopeParams {
    fn default() -> Self {
        Self {
            n_rot: 0,
            n_offs: 0,
            rope_type: RopeType::None,
            n_ctx_orig: 0,
            freq_base: 0.0,
            freq_scale: 0.0,
            ext_factor: 0.0,
            attn_factor: 0.0,
            beta_fast: 0.0,
            beta_slow: 0.0,
            factors: None,
            k_norm: None,
        }
    }
}

/// `llama_hadamard_transform`; tensors are named rather than pointer-keyed.
#[derive(Clone, Debug, Default)]
pub struct HadamardTransform {
    pub rot: String,
    pub signs: Option<String>,
    pub perm_hd: i64,
    pub perm_nk: i64,
    pub perm_rep: i64,
}

impl HadamardTransform {
    /// `set_gdn_v_perm`: reject geometries that cannot describe the tiling.
    pub fn set_gdn_v_perm(&mut self, n_in: i64, n_v: i64, n_k: i64) -> bool {
        if n_k <= 0 || n_v <= 0 || n_v % n_k != 0 || n_in % n_v != 0 {
            return false;
        }
        self.perm_hd = n_in / n_v;
        self.perm_nk = n_k;
        self.perm_rep = n_v / n_k;
        true
    }
}

/// `llama_hadamard`.
#[derive(Clone, Debug, Default)]
pub struct Hadamard {
    pub weight_blocks: std::collections::BTreeMap<String, u32>,
    pub inverse_blocks: std::collections::BTreeMap<String, u32>,
    pub sign_data: std::collections::BTreeMap<i32, Vec<i32>>,
    pub gdn_v_grouped: bool,
    pub tied_output: bool,
    pub rot: std::collections::BTreeMap<String, HadamardTransform>,
    pub inv: std::collections::BTreeMap<String, HadamardTransform>,
}

/// `llama_prec_policy`. Names are used as keys instead of tensor pointers.
#[derive(Clone, Debug, Default)]
pub struct PrecPolicy {
    pub prec_acc: std::collections::BTreeMap<String, Precision>,
    pub prec_src1: std::collections::BTreeMap<String, Precision>,
}

impl PrecPolicy {
    pub fn set_acc(&mut self, weight: Option<&Tensor>, prec: Precision) {
        if let Some(w) = weight {
            self.prec_acc.insert(w.name.clone(), prec);
        }
    }

    pub fn set_src1(&mut self, weight: Option<&Tensor>, prec: Precision) {
        if let Some(w) = weight {
            self.prec_src1.insert(w.name.clone(), prec);
        }
    }

    pub fn get_src1(&self, weight: &Tensor) -> Precision {
        self.prec_src1
            .get(&weight.name)
            .copied()
            .unwrap_or(Precision::Undefined)
    }

    /// `apply`: records the requested precision on the result tensor name.
    pub fn apply(&self, result: &mut Tensor, weight: Option<&Tensor>) -> bool {
        let Some(w) = weight else {
            return false;
        };
        let key = w.name.clone();
        let mut applied = false;
        if let Some(prec) = self.prec_acc.get(&key) {
            result.data.extend_from_slice(&[*prec as u8]);
            applied = true;
        }
        if let Some(prec) = self.prec_src1.get(&key) {
            result.data.extend_from_slice(&[*prec as u8]);
            applied = true;
        }
        applied
    }
}

/// `llama_device`.
#[derive(Clone, Debug)]
pub struct Device {
    pub name: String,
    pub is_meta: bool,
    pub mmap_support: bool,
    pub free_memory: u64,
    pub total_memory: u64,
    pub backend_id: usize,
}

impl Device {
    pub fn cpu() -> Self {
        Self {
            name: "CPU".to_owned(),
            is_meta: false,
            mmap_support: true,
            free_memory: 0,
            total_memory: 0,
            backend_id: 0,
        }
    }
}

/// A tensor buffer type stand-in; identity is `(device, name)`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BufferType {
    pub name: String,
    pub device: usize,
}

/// `ggml_backend_buffer` stand-in with ownership accounting.
#[derive(Clone, Debug)]
pub struct Buffer {
    pub buft: BufferType,
    pub size: usize,
    pub usage: String,
    pub base_is_null: bool,
}

/// `llama_model_params`.
#[derive(Clone, Debug)]
pub struct ModelParams {
    pub devices: Vec<Device>,
    pub tensor_buft_overrides: Vec<TensorBuftOverride>,
    pub n_gpu_layers: i32,
    pub split_mode: SplitMode,
    pub load_mode: LoadMode,
    pub lazy_mode: LazyMode,
    pub main_gpu: usize,
    pub tensor_split: Vec<f32>,
    /// Returns false to cancel tensor loading, like the C++ callback.
    pub progress_callback: Option<fn(f32) -> bool>,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            devices: vec![Device::cpu()],
            tensor_buft_overrides: Vec::new(),
            n_gpu_layers: -1,
            split_mode: SplitMode::Layer,
            load_mode: LoadMode::Auto,
            lazy_mode: LazyMode::Auto,
            main_gpu: 0,
            tensor_split: Vec::new(),
            progress_callback: None,
            vocab_only: false,
            check_tensors: false,
            use_extra_bufts: true,
            no_host: false,
            no_alloc: false,
            load_mtp: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorBuftOverride {
    pub pattern: String,
    pub buft: String,
}

/// `llama_meta_device_get_split_state_userdata` fields.
#[derive(Clone, Debug)]
pub struct SplitStateUserdata {
    pub n_devices: usize,
}

/// `ggml_backend_meta_split_state` stand-in.
#[derive(Clone, Debug)]
pub struct MetaSplitState {
    pub axis: i32,
    pub ne: Vec<i64>,
    pub nr: Vec<u32>,
    pub n_segments: usize,
}

impl Default for MetaSplitState {
    fn default() -> Self {
        Self {
            axis: -1,
            ne: vec![0; 16],
            nr: vec![1; 4],
            n_segments: 1,
        }
    }
}

/// `llama_model` member `target_layer_ids` element type.
pub type TargetLayerId = i32;

/// Token id convenience alias, matching the rest of the port.
pub type ModelToken = Token;

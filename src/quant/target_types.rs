//! Types used by the `--target-bpw` / `--target-size` optimizer.

use super::types::QType;

/// Candidate quantization type and its measured properties.
#[derive(Clone, Debug)]
pub struct TypeScores {
    pub ty: QType,
    pub bpw: f32,
    pub bytes: usize,
    pub error: f64,
    pub mse: f64,
    pub wce: f64,
}

impl Default for TypeScores {
    fn default() -> Self {
        Self {
            ty: QType::Count,
            bpw: 0.0,
            bytes: 0,
            error: f64::INFINITY,
            mse: 0.0,
            wce: 0.0,
        }
    }
}

/// One tensor's candidate set and chosen candidate.
#[derive(Clone, Debug)]
pub struct TypeChoice {
    pub tensor: usize,
    pub candidates: Vec<TypeScores>,
    pub choice: i32,
    pub min_bpw: f32,
    pub max_bpw: f32,
    pub n_elements: usize,
    pub boost: f32,
}

impl Default for TypeChoice {
    fn default() -> Self {
        Self {
            tensor: 0,
            candidates: Vec::new(),
            choice: -1,
            min_bpw: 0.0,
            max_bpw: 0.0,
            n_elements: 0,
            boost: 1.0,
        }
    }
}

pub const QUANT_TYPES: [QType; 21] = [
    QType::Iq1S,
    QType::Iq1M,
    QType::Iq2Xxs,
    QType::Iq2Xs,
    QType::Iq2S,
    QType::Iq3Xxs,
    QType::Iq3S,
    QType::Iq4Xs,
    QType::Iq4Nl,
    QType::Q2K,
    QType::Q3K,
    QType::Q4_0,
    QType::Q4_1,
    QType::Q4K,
    QType::Q5_0,
    QType::Q5_1,
    QType::Q5K,
    QType::Q6K,
    QType::Q8_0,
    QType::F16,
    QType::Bf16,
];

pub const EPSILON: f64 = 1e-12;
pub const LQ_INFINITE: f64 = f64::INFINITY;
pub const STATE_MAGIC: u64 = 0x4250_572d_5632;
pub const HASH_MAGIC: u64 = 0xeaba_da55_cafe_d00d;

pub const FFN_DOWN_BOOST: f32 = 6.0;
pub const OUTPUT_BOOST: f32 = 3.0;
pub const TIED_EMBD_BOOST: f32 = 2.5;
pub const ATTN_V_BOOST: f32 = 1.25;
pub const BOOST_TAPER: f64 = 2.2;
pub const DOWNWARD_THRESHOLD: f64 = 0.08;

pub fn is_iq(ty: QType) -> bool {
    matches!(
        ty,
        QType::Iq1S
            | QType::Iq1M
            | QType::Iq2Xxs
            | QType::Iq2Xs
            | QType::Iq2S
            | QType::Iq3Xxs
            | QType::Iq3S
            | QType::Iq4Nl
            | QType::Iq4Xs
    )
}

pub fn is_angle_sensitive(name: &str) -> bool {
    name.contains("attn_q.weight")
        || name.contains("attn_k.weight")
        || name.contains("attn_qkv.weight")
        || name.contains("attn_q_a.weight")
        || name.contains("attn_q_b.weight")
}

pub fn is_inner_product_sensitive(name: &str) -> bool {
    name.contains("attn_q")
        || name.contains("attn_k")
        || name.contains("attn_o")
        || name.contains("attn_v")
}

/// DJB2 hash used for sampling seeds and checkpoint identifiers.
pub fn djb2_hash(data: &[u8], mut h: u64) -> u64 {
    for &b in data {
        h = h.wrapping_shl(5).wrapping_add(h).wrapping_add(b as u64);
    }
    if h == 0 {
        HASH_MAGIC
    } else {
        h
    }
}

pub fn hash_u64(h: u64, value: u64) -> u64 {
    djb2_hash(&value.to_ne_bytes(), h)
}

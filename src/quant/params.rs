//! `llama_model_quantize_params` and the file-type table.

use std::sync::Arc;

use crate::core::Result;

use super::types::QType;

/// `enum llama_ftype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum FType {
    AllF32 = 0,
    MostlyF16 = 1,
    MostlyQ4_0 = 2,
    MostlyQ4_1 = 3,
    MostlyQ8_0 = 7,
    MostlyQ5_0 = 8,
    MostlyQ5_1 = 9,
    MostlyQ2K = 10,
    MostlyQ3KS = 11,
    MostlyQ3KM = 12,
    MostlyQ3KL = 13,
    MostlyQ4KS = 14,
    MostlyQ4KM = 15,
    MostlyQ5KS = 16,
    MostlyQ5KM = 17,
    MostlyQ6K = 18,
    MostlyIq2Xxs = 19,
    MostlyIq2Xs = 20,
    MostlyQ2KS = 21,
    MostlyIq3Xs = 22,
    MostlyIq3Xxs = 23,
    MostlyIq1S = 24,
    MostlyIq4Nl = 25,
    MostlyIq3S = 26,
    MostlyIq3M = 27,
    MostlyIq2S = 28,
    MostlyIq2M = 29,
    MostlyIq4Xs = 30,
    MostlyIq1M = 31,
    MostlyBf16 = 32,
    MostlyTq1_0 = 36,
    MostlyTq2_0 = 37,
    MostlyMxfp4Moe = 38,
    MostlyNvfp4 = 39,
    MostlyQ1_0 = 40,
    MostlyQ2_0 = 41,
    MostlyF8E4M3 = 42,
    Guessed = 1024,
}

impl FType {
    pub fn from_i32(value: i32) -> Option<Self> {
        let all = [
            FType::AllF32,
            FType::MostlyF16,
            FType::MostlyQ4_0,
            FType::MostlyQ4_1,
            FType::MostlyQ8_0,
            FType::MostlyQ5_0,
            FType::MostlyQ5_1,
            FType::MostlyQ2K,
            FType::MostlyQ3KS,
            FType::MostlyQ3KM,
            FType::MostlyQ3KL,
            FType::MostlyQ4KS,
            FType::MostlyQ4KM,
            FType::MostlyQ5KS,
            FType::MostlyQ5KM,
            FType::MostlyQ6K,
            FType::MostlyIq2Xxs,
            FType::MostlyIq2Xs,
            FType::MostlyQ2KS,
            FType::MostlyIq3Xs,
            FType::MostlyIq3Xxs,
            FType::MostlyIq1S,
            FType::MostlyIq4Nl,
            FType::MostlyIq3S,
            FType::MostlyIq3M,
            FType::MostlyIq2S,
            FType::MostlyIq2M,
            FType::MostlyIq4Xs,
            FType::MostlyIq1M,
            FType::MostlyBf16,
            FType::MostlyTq1_0,
            FType::MostlyTq2_0,
            FType::MostlyMxfp4Moe,
            FType::MostlyNvfp4,
            FType::MostlyQ1_0,
            FType::MostlyQ2_0,
            FType::MostlyF8E4M3,
            FType::Guessed,
        ];
        all.into_iter().find(|f| *f as i32 == value)
    }

    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

/// `llama_ftype_get_default_type`: default tensor type for a file type.
pub fn ftype_default_type(ftype: FType) -> QType {
    match ftype {
        FType::MostlyQ4_0 => QType::Q4_0,
        FType::MostlyQ4_1 => QType::Q4_1,
        FType::MostlyQ5_0 => QType::Q5_0,
        FType::MostlyQ5_1 => QType::Q5_1,
        FType::MostlyQ8_0 => QType::Q8_0,
        FType::MostlyF16 => QType::F16,
        FType::MostlyBf16 => QType::Bf16,
        FType::AllF32 => QType::F32,
        FType::MostlyQ1_0 => QType::Q1_0,
        FType::MostlyQ2_0 => QType::Q2_0,
        FType::MostlyMxfp4Moe => QType::Mxfp4,
        FType::MostlyQ2KS | FType::MostlyQ2K => QType::Q2K,
        FType::MostlyIq3Xs => QType::Iq3S,
        FType::MostlyQ3KS | FType::MostlyQ3KM | FType::MostlyQ3KL => QType::Q3K,
        FType::MostlyQ4KS | FType::MostlyQ4KM => QType::Q4K,
        FType::MostlyQ5KS | FType::MostlyQ5KM => QType::Q5K,
        FType::MostlyQ6K => QType::Q6K,
        FType::MostlyTq1_0 => QType::Tq1_0,
        FType::MostlyTq2_0 => QType::Tq2_0,
        FType::MostlyIq2Xxs => QType::Iq2Xxs,
        FType::MostlyIq2Xs => QType::Iq2Xs,
        FType::MostlyIq2S => QType::Iq2Xs,
        FType::MostlyIq2M => QType::Iq2S,
        FType::MostlyIq3Xxs => QType::Iq3Xxs,
        FType::MostlyIq1S => QType::Iq1S,
        FType::MostlyIq1M => QType::Iq1M,
        FType::MostlyIq4Nl => QType::Iq4Nl,
        FType::MostlyIq4Xs => QType::Iq4Xs,
        FType::MostlyIq3S | FType::MostlyIq3M => QType::Iq3S,
        _ => QType::Count,
    }
}

/// One `llama_model_imatrix_data` entry.
#[derive(Clone, Debug)]
pub struct ImatrixData {
    pub name: String,
    pub data: Vec<f32>,
}

/// One `llama_model_kv_override` entry.
#[derive(Clone, Debug)]
pub struct KvOverride {
    pub key: String,
    pub tag: KvOverrideType,
    pub val_f64: f64,
    pub val_i64: i64,
    pub val_bool: bool,
    pub val_str: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvOverrideType {
    Float,
    Int,
    Bool,
    Str,
}

/// One `tensor_type_option` entry (`--tensor-type pattern=type`).
#[derive(Clone, Debug)]
pub struct TensorTypeOverride {
    pub pattern: String,
    pub ty: QType,
}

/// `llama_model_quantize_params`.
#[derive(Clone)]
pub struct QuantizeParams {
    pub nthread: i32,
    pub ftype: FType,
    pub output_tensor_type: QType,
    pub token_embedding_type: QType,
    pub allow_requantize: bool,
    pub quantize_output_tensor: bool,
    pub only_copy: bool,
    pub pure: bool,
    pub keep_split: bool,
    pub dry_run: bool,
    pub imatrix: Vec<ImatrixData>,
    pub kv_overrides: Vec<KvOverride>,
    pub tt_overrides: Vec<TensorTypeOverride>,
    pub prune_layers: Vec<i32>,
    pub max_buf_size: usize,
    pub activations: Vec<ImatrixData>,
    pub statistics: Vec<ImatrixData>,
    pub stats_schema: Vec<String>,
    pub target_bpw: f32,
    pub target_size: i64,
    pub state_file: String,
    pub target_exclude: Vec<QType>,
    pub abort_callback: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
}

/// `llama_model_quantize_default_params`.
pub const LLAMA_QUANT_MAX_BUF_SIZE: usize = 8usize * 1024 * 1024 * 1024;

impl Default for QuantizeParams {
    fn default() -> Self {
        Self {
            nthread: 0,
            ftype: FType::MostlyQ8_0,
            output_tensor_type: QType::Count,
            token_embedding_type: QType::Count,
            allow_requantize: false,
            quantize_output_tensor: true,
            only_copy: false,
            pure: false,
            keep_split: false,
            dry_run: false,
            imatrix: Vec::new(),
            kv_overrides: Vec::new(),
            tt_overrides: Vec::new(),
            prune_layers: Vec::new(),
            max_buf_size: LLAMA_QUANT_MAX_BUF_SIZE,
            activations: Vec::new(),
            statistics: Vec::new(),
            stats_schema: Vec::new(),
            target_bpw: -1.0,
            target_size: -1,
            state_file: String::new(),
            target_exclude: Vec::new(),
            abort_callback: None,
        }
    }
}

impl QuantizeParams {
    pub fn has_imatrix(&self) -> bool {
        !self.imatrix.is_empty()
    }

    pub fn has_activations(&self) -> bool {
        !self.activations.is_empty()
    }

    pub fn has_statistics(&self) -> bool {
        !self.statistics.is_empty()
    }

    pub fn target_requested(&self) -> bool {
        self.target_bpw != -1.0 || self.target_size != -1
    }

    pub fn aborted(&self) -> bool {
        self.abort_callback.as_ref().map(|f| f()).unwrap_or(false)
    }
}

/// Metadata needed to build a model description for `llama_quant_compute_types`.
#[derive(Clone, Debug)]
pub struct ModelQuantizeDesc {
    pub architecture: String,
    pub n_layer: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_embd: u32,
    pub n_embd_head_k: u32,
    pub n_embd_head_v: u32,
    pub n_ff: u32,
    pub n_expert: u32,
}

impl Default for ModelQuantizeDesc {
    fn default() -> Self {
        Self {
            architecture: String::new(),
            n_layer: 0,
            n_head: 0,
            n_head_kv: 0,
            n_embd: 0,
            n_embd_head_k: 0,
            n_embd_head_v: 0,
            n_ff: 0,
            n_expert: 0,
        }
    }
}

/// Validate the side-data maps as the driver does before quantization.
pub fn validate_side_data(data: &[ImatrixData], label: &str) -> Result<usize> {
    for entry in data {
        for &v in &entry.data {
            if !v.is_finite() {
                return Err(crate::core::Error::new(format!("{} {}", label, v)));
            }
        }
    }
    Ok(data.len())
}

/// Build a name -> values map from side data, mirroring `emplace` in C++.
pub fn imatrix_map(data: &[ImatrixData]) -> std::collections::HashMap<String, Vec<f32>> {
    let mut out = std::collections::HashMap::new();
    for e in data {
        out.entry(e.name.clone()).or_insert_with(|| e.data.clone());
    }
    out
}

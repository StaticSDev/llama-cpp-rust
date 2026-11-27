//! Quantization state, tensor metadata and target-type selection.

use crate::arch::Arch;
use crate::core::{Error, Result};

use super::names::{
    category_is_attn_v, tensor_allows_quantization, tensor_is_expert,
    TensorCategory,
};
use super::params::{FType, QuantizeParams};
use super::regex_lite::regex_search;
use super::types::{fallback_type_for, QTensor, QType, TYPE_COUNT};

/// The subset of model state that type selection depends on.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub arch: Arch,
    pub is_70b: bool,
    pub n_expert: u32,
    pub n_layer_all: u32,
    pub n_gqa: u32,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            arch: Arch::Llama,
            is_70b: false,
            n_expert: 1,
            n_layer_all: 0,
            n_gqa: 1,
        }
    }
}

/// Per-tensor metadata computed once before quantization.
#[derive(Clone, Debug)]
pub struct TensorMetadata {
    pub name: String,
    pub target_type: QType,
    pub category: TensorCategory,
    pub remapped_imatrix_name: String,
    pub allows_quantization: bool,
    pub requires_imatrix: bool,
}

impl Default for TensorMetadata {
    fn default() -> Self {
        Self {
            name: String::new(),
            target_type: QType::Count,
            category: TensorCategory::Other,
            remapped_imatrix_name: String::new(),
            allows_quantization: false,
            requires_imatrix: false,
        }
    }
}

/// `quantize_state_impl`.
pub struct QuantizeState {
    pub model: ModelInfo,
    pub params: QuantizeParams,
    pub n_attention_wv: usize,
    pub n_ffn_down: usize,
    pub n_ffn_gate: usize,
    pub n_ffn_up: usize,
    pub i_attention_wv: usize,
    pub i_ffn_down: usize,
    pub i_ffn_gate: usize,
    pub i_ffn_up: usize,
    pub n_fallback: usize,
    pub has_imatrix: bool,
    pub has_tied_embeddings: bool,
    pub has_activations: bool,
    pub tensor_type_patterns: Vec<(String, QType)>,
    pub ignored_types: [bool; TYPE_COUNT],
    pub has_ignored_types: bool,
}

impl QuantizeState {
    pub fn new(model: ModelInfo, params: &QuantizeParams) -> Self {
        let mut ignored_types = [false; TYPE_COUNT];
        let mut has_ignored_types = false;
        for &t in &params.target_exclude {
            let v = t as usize;
            if (t != QType::Count) && v < TYPE_COUNT {
                ignored_types[v] = true;
                has_ignored_types = true;
            }
        }
        Self {
            model,
            params: params.clone(),
            n_attention_wv: 0,
            n_ffn_down: 0,
            n_ffn_gate: 0,
            n_ffn_up: 0,
            i_attention_wv: 0,
            i_ffn_down: 0,
            i_ffn_gate: 0,
            i_ffn_up: 0,
            n_fallback: 0,
            has_imatrix: false,
            has_tied_embeddings: true,
            has_activations: false,
            tensor_type_patterns: params
                .tt_overrides
                .iter()
                .map(|o| (o.pattern.clone(), o.ty))
                .collect(),
            ignored_types,
            has_ignored_types,
        }
    }

    pub fn reset_counters(&mut self) {
        self.n_attention_wv = 0;
        self.n_ffn_down = 0;
        self.n_ffn_gate = 0;
        self.n_ffn_up = 0;
        self.i_attention_wv = 0;
        self.i_ffn_down = 0;
        self.i_ffn_gate = 0;
        self.i_ffn_up = 0;
        self.n_fallback = 0;
        self.has_imatrix = false;
        self.has_tied_embeddings = true;
    }

    /// `tensor_type_override_for`.
    pub fn tensor_type_override_for(&self, name: &str) -> QType {
        for (pattern, ty) in &self.tensor_type_patterns {
            if regex_search(name, pattern) {
                return *ty;
            }
        }
        QType::Count
    }

    pub fn is_ignored(&self, t: QType) -> bool {
        t != QType::Count && self.ignored_types[t as usize]
    }

    /// `tensor_type_fallback`.
    pub fn tensor_type_fallback(&mut self, tensor: &QTensor, target: QType) -> Result<QType> {
        let ncols = tensor.ne[0];
        let qk_k = target.blck_size();
        if qk_k <= 0 || ncols % qk_k == 0 {
            return Ok(target);
        }
        self.n_fallback += 1;
        crate::llama_log_warn!(
            "warning: {:36} - ncols {:6} not divisible by {:3} (required for type {:7}) ",
            tensor.name,
            ncols,
            qk_k,
            target.name()
        );
        let mut fallback = fallback_type_for(target)?;
        if fallback == QType::Count
            || fallback.blck_size() <= 0
            || ncols % fallback.blck_size() != 0
        {
            crate::llama_log_warn!("(WARNING: must use F16 due to unusual shape) ");
            fallback = QType::F16;
        }
        crate::llama_log_warn!("-> falling back to {:7}\n", fallback.name());
        Ok(fallback)
    }

    /// `init_quantize_state_counters`.
    pub fn init_counters(&mut self, metadata: &mut [TensorMetadata]) {
        for tm in metadata.iter_mut() {
            tm.category = super::names::tensor_get_category(&tm.name);
            if category_is_attn_v(tm.category) {
                self.n_attention_wv += 1;
            }
            if tm.category == TensorCategory::Output {
                self.has_tied_embeddings = false;
            }
        }
        let n = self.model.n_layer_all as usize;
        self.n_ffn_down = n;
        self.n_ffn_gate = n;
        self.n_ffn_up = n;
    }
}

fn use_more_bits(i_layer: usize, n_layers: usize) -> bool {
    i_layer < n_layers / 8 || i_layer >= 7 * n_layers / 8 || (i_layer - n_layers / 8) % 3 == 2
}

impl QuantizeState {
    fn layer_info(&self, i_layer: usize, n_layer: usize, name: &str) -> Result<(usize, usize)> {
        if self.model.n_expert > 1 {
            let Some(layer) = super::names::parse_block(name) else {
                return Err(Error::new(format!(
                    "Failed to determine layer for tensor {}",
                    name
                )));
            };
            if layer < 0 || layer as usize >= n_layer {
                return Err(Error::new(format!(
                    "Bad layer {} for tensor {}. Must be in [0, {})",
                    layer, name, n_layer
                )));
            }
            return Ok((layer as usize, n_layer));
        }
        Ok((i_layer.min(n_layer.saturating_sub(1)), n_layer))
    }

    /// `llama_tensor_get_type_impl`.
    pub fn tensor_get_type_impl(
        &mut self,
        mut new_type: QType,
        tensor: &QTensor,
        ftype: FType,
        category: TensorCategory,
    ) -> Result<QType> {
        let name = tensor.name.as_str();
        let arch = self.model.arch;

        if arch == Arch::Glm5Next
            && (name.contains("attn_q_a")
                || name.contains("attn_q_b")
                || name.contains("nextn.eh_proj"))
            && !matches!(new_type, QType::F32 | QType::Bf16 | QType::F16)
        {
            return Ok(QType::Q8_0);
        }

        if arch == Arch::Raon1Mini
            && matches!(ftype, FType::MostlyQ6K | FType::MostlyQ8_0)
            && (name.contains("attn_kv_a_mqa.weight")
                || name.contains("attn_k_b.weight")
                || name.contains("attn_v_b.weight"))
        {
            return Ok(tensor.ty);
        }

        if arch == Arch::MuseSpark
            && matches!(
                category,
                TensorCategory::FfnLatent | TensorCategory::AttentionGate
            )
            && new_type.is_quantized()
            && new_type.bits_per_weight() < QType::Q8_0.bits_per_weight()
        {
            new_type = QType::Q8_0;
        }

        if category == TensorCategory::Output
            || (self.has_tied_embeddings && category == TensorCategory::TokenEmbd)
        {
            if self.params.output_tensor_type != QType::Count {
                new_type = self.params.output_tensor_type;
            } else {
                let nx = tensor.ne[0];
                let qk_k = new_type.blck_size();
                if ftype == FType::MostlyMxfp4Moe {
                    new_type = QType::Q8_0;
                } else if arch == Arch::Falcon || (qk_k > 0 && nx % qk_k != 0) {
                    new_type = QType::Q8_0;
                } else if matches!(
                    ftype,
                    FType::MostlyIq2Xxs
                        | FType::MostlyIq2Xs
                        | FType::MostlyIq3Xxs
                        | FType::MostlyIq1S
                        | FType::MostlyIq2S
                        | FType::MostlyIq2M
                        | FType::MostlyIq1M
                ) {
                    new_type = QType::Q5K;
                } else if new_type != QType::Q8_0 {
                    new_type = QType::Q6K;
                }
            }
        } else if ftype == FType::MostlyMxfp4Moe {
            new_type = if tensor_is_expert(name) {
                QType::Mxfp4
            } else {
                QType::Q8_0
            };
        } else if category == TensorCategory::TokenEmbd {
            if self.params.token_embedding_type != QType::Count {
                new_type = self.params.token_embedding_type;
            } else if matches!(
                ftype,
                FType::MostlyIq2Xxs | FType::MostlyIq2Xs | FType::MostlyIq1S | FType::MostlyIq1M
            ) {
                new_type = QType::Q2K;
            } else if matches!(ftype, FType::MostlyIq2S | FType::MostlyIq2M) {
                new_type = QType::Iq3S;
            } else if ftype == FType::MostlyIq3Xxs {
                new_type = QType::Iq3S;
            } else if matches!(
                ftype,
                FType::MostlyTq1_0 | FType::MostlyTq2_0 | FType::MostlyQ2_0
            ) {
                new_type = QType::Q4K;
            }
        } else if matches!(
            ftype,
            FType::MostlyIq2Xxs
                | FType::MostlyIq2Xs
                | FType::MostlyIq1S
                | FType::MostlyIq2S
                | FType::MostlyIq2M
                | FType::MostlyIq1M
        ) {
            let iq2_sm = matches!(ftype, FType::MostlyIq2S | FType::MostlyIq2M);
            if category_is_attn_v(category) {
                if self.model.n_gqa >= 4 || self.model.n_expert >= 4 {
                    new_type = QType::Q4K;
                } else {
                    new_type = if iq2_sm { QType::Iq3S } else { QType::Q2K };
                }
                self.i_attention_wv += 1;
            } else if self.model.n_expert == 8 && category == TensorCategory::AttentionK {
                new_type = QType::Q4K;
            } else if category == TensorCategory::FfnDown {
                if self.i_ffn_down < self.n_ffn_down / 8 {
                    new_type = if iq2_sm { QType::Iq3S } else { QType::Q2K };
                }
                self.i_ffn_down += 1;
            } else if category == TensorCategory::AttentionOutput {
                if self.model.n_expert == 8 {
                    new_type = QType::Q5K;
                } else if matches!(ftype, FType::MostlyIq1S | FType::MostlyIq1M) {
                    new_type = QType::Iq2Xxs;
                } else if iq2_sm {
                    new_type = QType::Iq3S;
                }
            }
        } else if category_is_attn_v(category) {
            if ftype == FType::MostlyQ2K {
                new_type = if self.model.n_gqa >= 4 {
                    QType::Q4K
                } else {
                    QType::Q3K
                };
            } else if ftype == FType::MostlyQ2KS && self.model.n_gqa >= 4 {
                new_type = QType::Q4K;
            } else if ftype == FType::MostlyIq3Xxs {
                new_type = if self.model.n_gqa >= 4 {
                    QType::Q4K
                } else if !self.has_imatrix {
                    QType::Iq3S
                } else {
                    QType::Iq3Xxs
                };
            } else if matches!(ftype, FType::MostlyIq3Xs | FType::MostlyIq3S)
                && self.model.n_gqa >= 4
            {
                new_type = QType::Q4K;
            } else if ftype == FType::MostlyIq3M {
                new_type = QType::Q4K;
            } else if ftype == FType::MostlyQ3KM {
                new_type = if self.i_attention_wv < 2 {
                    QType::Q5K
                } else {
                    QType::Q4K
                };
            } else if ftype == FType::MostlyQ3KL {
                new_type = QType::Q5K;
            } else if matches!(ftype, FType::MostlyIq4Nl | FType::MostlyIq4Xs)
                && self.model.n_gqa >= 4
            {
                new_type = QType::Q5K;
            } else if matches!(ftype, FType::MostlyQ4KM | FType::MostlyQ5KM)
                && use_more_bits(self.i_attention_wv, self.n_attention_wv)
            {
                new_type = QType::Q6K;
            } else if ftype == FType::MostlyQ4KS && self.i_attention_wv < 4 {
                new_type = QType::Q5K;
            }
            if self.model.is_70b && matches!(new_type, QType::Q3K | QType::Q4K) {
                new_type = QType::Q5K;
            }
            if self.model.n_expert == 8 {
                new_type = QType::Q8_0;
            }
            self.i_attention_wv += 1;
        } else if category == TensorCategory::AttentionK {
            if self.model.n_expert == 8 {
                new_type = QType::Q8_0;
            } else if ftype == FType::MostlyIq3Xs {
                new_type = QType::Iq3Xxs;
            } else if ftype == FType::MostlyIq3Xxs {
                new_type = QType::Iq2S;
            }
        } else if category == TensorCategory::AttentionQ {
            if ftype == FType::MostlyIq3Xs {
                new_type = QType::Iq3Xxs;
            } else if ftype == FType::MostlyIq3Xxs {
                new_type = QType::Iq2S;
            }
        } else if category == TensorCategory::FfnDown {
            let (i_layer, n_layer) = self.layer_info(self.i_ffn_down, self.n_ffn_down, name)?;
            if ftype == FType::MostlyQ2K {
                new_type = QType::Q3K;
            } else if ftype == FType::MostlyQ2KS && i_layer < n_layer / 8 {
                new_type = QType::Q4K;
            } else if ftype == FType::MostlyIq3Xxs && !self.has_imatrix {
                new_type = if i_layer < n_layer / 8 {
                    QType::Q4K
                } else {
                    QType::Q3K
                };
            } else if ftype == FType::MostlyQ3KM {
                new_type = if i_layer < n_layer / 16 {
                    QType::Q5K
                } else if arch != Arch::Falcon || use_more_bits(i_layer, n_layer) {
                    QType::Q4K
                } else {
                    QType::Q3K
                };
            } else if ftype == FType::MostlyIq3M
                && (i_layer < n_layer / 8
                    || (self.model.n_expert == 8 && use_more_bits(i_layer, n_layer)))
            {
                new_type = QType::Q4K;
            } else if ftype == FType::MostlyQ3KL {
                new_type = if arch == Arch::Falcon {
                    QType::Q4K
                } else {
                    QType::Q5K
                };
            } else if ftype == FType::MostlyQ4KM {
                if arch == Arch::Falcon {
                    new_type = if i_layer < n_layer / 16 {
                        QType::Q6K
                    } else if use_more_bits(i_layer, n_layer) {
                        QType::Q5K
                    } else {
                        QType::Q4K
                    };
                } else if use_more_bits(i_layer, n_layer) {
                    new_type = QType::Q6K;
                }
            } else if i_layer < n_layer / 8
                && matches!(ftype, FType::MostlyIq4Nl | FType::MostlyIq4Xs)
                && !self.has_imatrix
            {
                new_type = QType::Q5K;
            } else if ftype == FType::MostlyQ5KM && use_more_bits(i_layer, n_layer) {
                new_type = QType::Q6K;
            } else if ftype == FType::MostlyQ4KS && arch != Arch::Falcon && i_layer < n_layer / 8 {
                new_type = QType::Q5K;
            } else if matches!(ftype, FType::MostlyQ4_0 | FType::MostlyQ5_0)
                && self.has_imatrix
                && i_layer < n_layer / 8
            {
                new_type = if ftype == FType::MostlyQ4_0 {
                    QType::Q4_1
                } else {
                    QType::Q5_1
                };
            }
            self.i_ffn_down += 1;
        } else if category == TensorCategory::AttentionOutput {
            if arch != Arch::Falcon {
                if self.model.n_expert == 8 {
                    if matches!(
                        ftype,
                        FType::MostlyQ2K
                            | FType::MostlyIq3Xs
                            | FType::MostlyIq3Xxs
                            | FType::MostlyQ3KS
                            | FType::MostlyQ3KM
                            | FType::MostlyIq4Nl
                            | FType::MostlyQ4KS
                            | FType::MostlyQ4KM
                            | FType::MostlyIq3S
                            | FType::MostlyIq3M
                            | FType::MostlyIq4Xs
                    ) {
                        new_type = QType::Q5K;
                    }
                } else if ftype == FType::MostlyQ2K {
                    new_type = QType::Q3K;
                } else if ftype == FType::MostlyIq3Xxs {
                    new_type = QType::Iq3S;
                } else if ftype == FType::MostlyQ3KM {
                    new_type = QType::Q4K;
                } else if ftype == FType::MostlyQ3KL {
                    new_type = QType::Q5K;
                } else if ftype == FType::MostlyIq3M {
                    new_type = QType::Q4K;
                }
            } else if ftype == FType::MostlyQ3KL {
                new_type = QType::Q4K;
            }
        } else if category == TensorCategory::AttentionQkv {
            if matches!(
                ftype,
                FType::MostlyQ3KM | FType::MostlyQ3KL | FType::MostlyIq3M
            ) {
                new_type = QType::Q4K;
            } else if ftype == FType::MostlyQ4KM {
                new_type = QType::Q5K;
            } else if ftype == FType::MostlyQ5KM {
                new_type = QType::Q6K;
            }
        } else if category == TensorCategory::FfnGate {
            let (i_layer, n_layer) = self.layer_info(self.i_ffn_gate, self.n_ffn_gate, name)?;
            if ftype == FType::MostlyIq3Xs && i_layer >= n_layer / 8 && i_layer < 7 * n_layer / 8 {
                new_type = QType::Iq3Xxs;
            }
            self.i_ffn_gate += 1;
        } else if category == TensorCategory::FfnUp {
            let (i_layer, n_layer) = self.layer_info(self.i_ffn_up, self.n_ffn_up, name)?;
            if ftype == FType::MostlyIq3Xs && i_layer >= n_layer / 8 && i_layer < 7 * n_layer / 8 {
                new_type = QType::Iq3Xxs;
            }
            self.i_ffn_up += 1;
        }

        if arch == Arch::Raon1Mini && new_type.is_quantized() {
            let k_4_5 = matches!(
                ftype,
                FType::MostlyQ4KS | FType::MostlyQ4KM | FType::MostlyQ5KS | FType::MostlyQ5KM
            );
            let low = matches!(
                ftype,
                FType::MostlyIq4Xs
                    | FType::MostlyIq4Nl
                    | FType::MostlyQ3KS
                    | FType::MostlyQ3KM
                    | FType::MostlyQ3KL
                    | FType::MostlyIq3Xxs
                    | FType::MostlyIq3Xs
                    | FType::MostlyIq3S
                    | FType::MostlyIq3M
                    | FType::MostlyQ2K
                    | FType::MostlyQ2KS
                    | FType::MostlyIq2Xxs
                    | FType::MostlyIq2Xs
                    | FType::MostlyIq2S
                    | FType::MostlyIq2M
                    | FType::MostlyIq1S
                    | FType::MostlyIq1M
            );
            let floor = if matches!(
                category,
                TensorCategory::AttentionOutput | TensorCategory::AttentionGate
            ) {
                if k_4_5 {
                    QType::Q8_0
                } else if low {
                    QType::Q6K
                } else {
                    QType::Count
                }
            } else if name.contains("attn_q_a.weight")
                || name.contains("attn_q_b.weight")
                || name.contains("attn_k_b.weight")
                || name.contains("attn_kv_a_mqa.weight")
                || name.contains("attn_v_b.weight")
                || name.contains("_shexp.weight")
            {
                if k_4_5 || low {
                    QType::Q6K
                } else {
                    QType::Count
                }
            } else if matches!(
                category,
                TensorCategory::AttentionQ
                    | TensorCategory::AttentionK
                    | TensorCategory::AttentionV
            ) || name.contains("ssm_g.weight")
            {
                if k_4_5 {
                    QType::Q6K
                } else {
                    QType::Count
                }
            } else {
                QType::Count
            };
            if floor != QType::Count && new_type.bits_per_weight() < floor.bits_per_weight() {
                new_type = floor;
            }
        }

        Ok(new_type)
    }

    /// `llama_tensor_get_type`.
    pub fn tensor_get_type(
        &mut self,
        params: &QuantizeParams,
        tensor: &QTensor,
        default_type: QType,
        tm: &TensorMetadata,
    ) -> Result<QType> {
        if params.target_requested() || !tensor_allows_quantization(params, self.model.arch, tensor)
        {
            return Ok(tensor.ty);
        }
        if params.token_embedding_type != QType::Count && tm.category == TensorCategory::TokenEmbd {
            let named = tensor.name == "per_layer_token_embd.weight"
                && self.tensor_type_override_for(&tensor.name) != QType::Count;
            if !named {
                return Ok(params.token_embedding_type);
            }
        }
        if params.output_tensor_type != QType::Count && tm.category == TensorCategory::Output {
            return Ok(params.output_tensor_type);
        }

        let mut new_type = default_type;
        if default_type.is_quantized() {
            let manual = self.tensor_type_override_for(&tensor.name);
            if manual != QType::Count && manual != new_type {
                crate::llama_log_warn!(
                    "llama_tensor_get_type: {:36} - applying manual override: {} -> {}\n",
                    tensor.name,
                    new_type.name(),
                    manual.name()
                );
                new_type = manual;
            }
            if manual == QType::Count && !params.pure {
                new_type =
                    self.tensor_get_type_impl(new_type, tensor, params.ftype, tm.category)?;
            }
            new_type = self.tensor_type_fallback(tensor, new_type)?;
        }
        Ok(new_type)
    }
}

/// `llama_quant_compute_types` helper: build metadata from names.
pub fn metadata_from_tensors(tensors: &[QTensor]) -> Vec<TensorMetadata> {
    tensors
        .iter()
        .map(|t| TensorMetadata {
            name: t.name.clone(),
            ..TensorMetadata::default()
        })
        .collect()
}

/// `llama_quant_model_from_metadata`: infer the fields used by quantization.
pub fn model_info_from_desc(desc: &super::params::ModelQuantizeDesc) -> Result<ModelInfo> {
    if desc.n_layer == 0 || desc.n_layer as usize > crate::hparams::LLAMA_MAX_LAYERS {
        return Err(Error::new("invalid layer count in model description"));
    }
    let arch = crate::arch::arch_from_string(&desc.architecture);
    let is_70b = arch == Arch::Llama && desc.n_layer == 80 && desc.n_head != desc.n_head_kv;
    let n_gqa = if desc.n_head_kv == 0 {
        1
    } else {
        desc.n_head / desc.n_head_kv
    };
    Ok(ModelInfo {
        arch,
        is_70b,
        n_expert: desc.n_expert.max(1),
        n_layer_all: desc.n_layer,
        n_gqa,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ftype_defaults() {
        assert_eq!(
            super::super::params::ftype_default_type(FType::MostlyQ4KM),
            QType::Q4K
        );
        assert_eq!(
            super::super::params::ftype_default_type(FType::MostlyIq2S),
            QType::Iq2Xs
        );
    }

    #[test]
    fn override_patterns() {
        let params = QuantizeParams {
            tt_overrides: vec![super::super::params::TensorTypeOverride {
                pattern: r"blk\.[0-9]+\.attn_v\.weight".into(),
                ty: QType::Q8_0,
            }],
            ..QuantizeParams::default()
        };
        let qs = QuantizeState::new(ModelInfo::default(), &params);
        assert_eq!(
            qs.tensor_type_override_for("blk.1.attn_v.weight"),
            QType::Q8_0
        );
        assert_eq!(
            qs.tensor_type_override_for("blk.1.attn_k.weight"),
            QType::Count
        );
    }
}

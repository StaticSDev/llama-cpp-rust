//! Public getters, rope parameter resolution, memory selection and graph
//! assembly steps from `llama-model.cpp`.

use super::model::LlamaModel;
use super::types::*;
use crate::arch::{arch_is_diffusion, arch_is_hybrid, arch_is_recurrent, Arch};
use crate::core::{Error, Result, Token};
use crate::cparams::{CParams, ContextType};

impl LlamaModel {
    /// `llama_model_n_ctx_train`.
    pub fn n_ctx_train(&self) -> u32 {
        self.hparams.n_ctx_train
    }

    /// `llama_model_n_embd`.
    pub fn n_embd(&self) -> u32 {
        self.hparams.n_embd
    }

    /// `llama_model_n_embd_inp`.
    pub fn n_embd_inp(&self) -> u32 {
        self.hparams.n_embd_inp()
    }

    /// `llama_model_n_embd_out`.
    pub fn n_embd_out(&self) -> u32 {
        self.hparams.n_embd_out()
    }

    /// `llama_model_n_layer`.
    pub fn n_layer(&self) -> u32 {
        self.hparams.n_layer()
    }

    /// `llama_model_n_layer_nextn`.
    pub fn n_layer_nextn(&self) -> u32 {
        self.hparams.n_layer_nextn
    }

    /// `llama_model_dflash_variant`.
    pub fn dflash_variant(&self) -> DflashVariant {
        if self.arch == Arch::Dflash {
            self.dflash_variant
        } else {
            DflashVariant::None
        }
    }

    /// `llama_model_dflash_selector_top_k`.
    pub fn dflash_selector_top_k(&self) -> u32 {
        self.hparams.dflash_selector_top_k
    }

    /// `llama_model_n_head`.
    pub fn n_head(&self) -> u32 {
        self.hparams.n_head(0).unwrap_or(0)
    }

    /// `llama_model_n_head_kv`.
    pub fn n_head_kv(&self) -> u32 {
        self.hparams.n_head_kv(0).unwrap_or(0)
    }

    /// `llama_model_n_swa`: DSV4 has SWA but cannot be used for rollback.
    pub fn n_swa(&self) -> u32 {
        if matches!(self.arch, Arch::Deepseek4 | Arch::Deepseek41) {
            return 0;
        }
        self.hparams.n_swa
    }

    pub fn n_cls_out(&self) -> u32 {
        self.hparams.n_cls_out
    }

    pub fn cls_label(&self, i: usize) -> Option<&str> {
        self.classifier_labels.get(i).map(|s| s.as_str())
    }

    pub fn decision_label(&self, i: usize) -> Option<&str> {
        self.decision_labels.get(i).map(|s| s.as_str())
    }

    pub fn rope_type(&self) -> RopeType {
        self.hparams.rope_type.into()
    }

    pub fn rope_freq_scale_train(&self) -> f32 {
        self.hparams.rope_freq_scale_train
    }

    pub fn meta_val_str(&self, key: &str) -> Option<&str> {
        self.gguf_kv.get(key).map(|s| s.as_str())
    }

    pub fn meta_count(&self) -> usize {
        self.gguf_kv.len()
    }

    pub fn meta_key_by_index(&self, i: usize) -> Option<&str> {
        self.gguf_kv.keys().nth(i).map(|s| s.as_str())
    }

    pub fn meta_val_str_by_index(&self, i: usize) -> Option<&str> {
        self.gguf_kv.values().nth(i).map(|s| s.as_str())
    }

    pub fn meta_arr_n(&self, key: &str) -> Option<usize> {
        self.gguf_kv_arr.get(key).map(|v| v.len())
    }

    pub fn meta_arr_str(&self, key: &str, i: usize) -> Option<&str> {
        self.gguf_kv_arr
            .get(key)
            .and_then(|v| v.get(i))
            .map(|s| s.as_str())
    }

    /// `llama_model_meta_key_str`.
    pub fn meta_key_str(key: i32) -> Option<&'static str> {
        match key {
            0 => Some("general.sampling.sequence"),
            1 => Some("general.sampling.top_k"),
            2 => Some("general.sampling.top_p"),
            3 => Some("general.sampling.min_p"),
            4 => Some("general.sampling.xtc_probability"),
            5 => Some("general.sampling.xtc_threshold"),
            6 => Some("general.sampling.temp"),
            7 => Some("general.sampling.penalty_last_n"),
            8 => Some("general.sampling.penalty_repeat"),
            9 => Some("general.sampling.mirostat"),
            10 => Some("general.sampling.mirostat_tau"),
            11 => Some("general.sampling.mirostat_eta"),
            _ => None,
        }
    }

    pub fn n_params(&self) -> u64 {
        self.n_elements()
    }

    /// `llama_model_has_encoder`.
    pub fn has_encoder(&self) -> bool {
        matches!(
            self.arch,
            Arch::T5 | Arch::T5encoder | Arch::Eagle3 | Arch::Dflash
        )
    }

    /// `llama_model_has_decoder`.
    pub fn has_decoder(&self) -> bool {
        self.arch != Arch::T5encoder
    }

    pub fn is_recurrent(&self) -> bool {
        arch_is_recurrent(self.arch)
    }

    pub fn is_hybrid(&self) -> bool {
        arch_is_hybrid(self.arch)
    }

    pub fn is_diffusion(&self) -> bool {
        arch_is_diffusion(self.arch)
    }

    pub fn n_expert(&self) -> u32 {
        self.hparams.n_expert
    }

    pub fn n_devices_i32(&self) -> i32 {
        self.devices.len() as i32
    }

    pub fn get_device(&self, i: usize) -> Option<&Device> {
        self.devices.get(i)
    }

    pub fn target_layer_ids(&self) -> &[i32] {
        &self.target_layer_ids
    }

    pub fn n_tokens_delay(&self) -> u32 {
        self.hparams.n_tokens_delay
    }

    pub fn classifier_free_guidance(&self) -> bool {
        self.hparams.classifier_free_guidance
    }

    /// `llama_model::get_rope_params`.
    pub fn get_rope_params(&self, cparams: &CParams, il: u32) -> Result<RopeParams> {
        let has_rope = self.hparams.has_rope(il)?;
        let n_rot = self.hparams.n_rot(il)?;
        let mut res = RopeParams {
            n_rot: if has_rope { n_rot as i32 } else { 0 },
            n_offs: if self.hparams.n_lora_kv > 0 {
                self.hparams.n_embd_head_k(il)? as i32 - n_rot as i32
            } else {
                0
            },
            rope_type: self.hparams.rope_type.into(),
            n_ctx_orig: cparams.n_ctx_orig_yarn as i32,
            freq_base: self.get_rope_freq_base(cparams, il)?,
            freq_scale: self.get_rope_freq_scale(cparams, il)?,
            ext_factor: cparams.yarn_ext_factor,
            attn_factor: cparams.yarn_attn_factor,
            beta_fast: cparams.yarn_beta_fast,
            beta_slow: cparams.yarn_beta_slow,
            factors: self.get_rope_factors(cparams, il),
            k_norm: None,
        };
        if let Some(t) = &self
            .layers
            .get(il as usize)
            .and_then(|l| l.attn_k_norm.clone())
        {
            res.k_norm = Some(t.name.clone());
        }
        Ok(res)
    }

    /// `llama_model::get_rope_params_indexer`.
    pub fn get_rope_params_indexer(&self, cparams: &CParams, il: u32) -> Result<RopeParams> {
        let mut res = self.get_rope_params(cparams, il)?;
        res.n_offs = 0;
        res.rope_type = RopeType::NeoX;
        res.freq_base = cparams.rope_freq_base;
        res.freq_scale = cparams.rope_freq_scale;
        res.factors = None;
        Ok(res)
    }

    /// `llama_model::get_rope_freq_base`.
    pub fn get_rope_freq_base(&self, cparams: &CParams, il: u32) -> Result<f32> {
        if self.hparams.is_swa(il)? {
            Ok(self.hparams.rope_freq_base_train_swa)
        } else {
            Ok(cparams.rope_freq_base)
        }
    }

    /// `llama_model::get_rope_freq_scale`.
    pub fn get_rope_freq_scale(&self, cparams: &CParams, il: u32) -> Result<f32> {
        if self.hparams.is_swa(il)? {
            Ok(self.hparams.rope_freq_scale_train_swa)
        } else {
            Ok(cparams.rope_freq_scale)
        }
    }

    /// `llama_model::get_rope_factors`: name of the selected factor tensor.
    pub fn get_rope_factors(&self, cparams: &CParams, il: u32) -> Option<String> {
        let layer = self.layers.get(il as usize)?;
        if let Some(ref freq) = layer.rope_freqs {
            return Some(freq.name.clone());
        }
        if cparams.n_ctx_seq > self.hparams.n_ctx_orig_yarn {
            layer
                .rope_long
                .as_ref()
                .or_else(|| layer.rope_short.as_ref())
                .map(|t| t.name.clone())
        } else {
            layer
                .rope_short
                .as_ref()
                .or_else(|| layer.rope_long.as_ref())
                .map(|t| t.name.clone())
        }
    }

    /// `llama_tensor_rows_to_f32` + `llama_model_get_tok_embd`.
    pub fn get_tok_embd(&self) -> Result<(u32, Vec<f32>)> {
        if self.vocab.n_tokens() == 0 || self.tok_embd.is_none() {
            return Ok((0, Vec::new()));
        }
        let tensor = self.tok_embd.as_ref().unwrap();
        let nelements = tensor.nelements();
        if nelements > u32::MAX as usize {
            return Err(Error::new("token embedding table exceeds UINT32_MAX"));
        }
        let rows = tensor.ne[1].max(0) as usize;
        let values = tensor.rows_to_f32(0, rows)?;
        Ok((nelements as u32, values))
    }

    /// `llama_model_get_tok_embd_rows`.
    pub fn get_tok_embd_rows(&self, tokens: &[Token]) -> Result<(u32, Vec<f32>)> {
        if self.vocab.n_tokens() == 0 || self.tok_embd.is_none() || tokens.is_empty() {
            return Ok((0, Vec::new()));
        }
        let tensor = self.tok_embd.as_ref().unwrap();
        let mut out = Vec::with_capacity(tokens.len() * tensor.ne[0].max(0) as usize);
        for &token in tokens {
            if token < 0 || token as i64 >= tensor.ne[1] {
                return Ok((0, Vec::new()));
            }
            out.extend(tensor.rows_to_f32(token as usize, 1)?);
        }
        Ok(((tokens.len() as u32) * tensor.ne[0].max(0) as u32, out))
    }
}

/// Kinds selected by `llama_model::create_memory`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemoryKind {
    #[default]
    None,
    Kv,
    KvIsWa,
    KvDsa,
    KvDsaIsWa,
    KvMsa,
    KvDsv4,
    Recurrent,
    Hybrid,
    HybridIsWa,
    HybridIdx,
}

/// Filter description recorded by memory selection.
#[derive(Clone, Debug, Default)]
pub struct MemoryDescriptor {
    pub kind: MemoryKind,
    pub filter_attn: Option<String>,
    pub filter_recr: Option<String>,
    pub filter_idx: Option<String>,
    pub reuse: Option<String>,
    pub share: Option<String>,
    pub n_ctx_seq: u32,
    pub n_seq_max: u32,
    pub n_ubatch: u32,
    pub no_alloc: bool,
}

impl LlamaModel {
    /// Structural port of `llama_model::create_memory`.
    pub fn memory_descriptor(
        &self,
        params: &crate::cparams::CParams,
        no_alloc: bool,
    ) -> Result<MemoryDescriptor> {
        use MemoryKind::*;
        let arch = self.arch;
        let h = &self.hparams;
        let mut desc = MemoryDescriptor {
            n_ctx_seq: params.n_ctx_seq,
            n_seq_max: params.n_seq_max,
            n_ubatch: params.n_ubatch,
            no_alloc,
            ..Default::default()
        };
        if (arch == Arch::Lfm2 && !h.causal_attn && h.n_layer_decision > 0)
            || (arch == Arch::Lfm3 && !h.causal_attn)
        {
            desc.kind = None;
            return Ok(desc);
        }

        let is_mtp = params.ctx_type == ContextType::Mtp;
        match arch {
            Arch::Bert
            | Arch::JinaBertV2
            | Arch::JinaBertV3
            | Arch::NomicBert
            | Arch::NomicBertMoe
            | Arch::NeoBert
            | Arch::Eurobert
            | Arch::WavtokenizerDec
            | Arch::ModernBert
            | Arch::GemmaEmbedding
            | Arch::GemmaEmbedding2
            | Arch::Dream
            | Arch::Llada
            | Arch::LladaMoe
            | Arch::Rnd1
            | Arch::Clef
            | Arch::Kokoro
            | Arch::TinyVoca => {
                desc.kind = None;
            }
            Arch::MinimaxM3 => {
                desc.kind = KvMsa;
                desc.filter_idx = Some("il < n_layer && is_indexer_full(il)".to_owned());
            }
            Arch::GlmDsa | Arch::Deepseek32 => {
                if is_mtp && h.n_layer_nextn > 0 {
                    desc.kind = Kv;
                    desc.filter_attn = Some("il >= n_layer".to_owned());
                } else {
                    desc.kind = KvDsa;
                    if h.n_layer_nextn > 0 {
                        desc.filter_attn = Some("il < n_layer".to_owned());
                    }
                    desc.filter_idx = Some(
                        if arch == Arch::GlmDsa {
                            "il < n_layer && is_indexer_full(il)"
                        } else {
                            "il < n_layer"
                        }
                        .to_owned(),
                    );
                }
            }
            Arch::MistralLarge4 => {
                desc.kind = KvDsa;
                desc.filter_idx = Some("is_indexer_full(il)".to_owned());
                desc.reuse = Some("has_kv(il) ? -1 : kv_source(il)".to_owned());
            }
            Arch::Glm5Next => {
                desc.kind = HybridIdx;
                if is_mtp {
                    if h.n_layer_nextn == 0 {
                        return Err(Error::new(
                            "GLM5-Next MTP requires the NextN block, convert without --no-mtp",
                        ));
                    }
                    desc.filter_attn = Some("il >= n_layer".to_owned());
                    desc.filter_idx = Some("il >= n_layer".to_owned());
                    desc.filter_recr = Some("false".to_owned());
                } else {
                    desc.filter_attn = Some("il < n_layer && !is_recr(il)".to_owned());
                    desc.filter_recr = Some("il < n_layer && is_recr(il)".to_owned());
                    desc.filter_idx =
                        Some("il < n_layer && !is_recr(il) && is_indexer_full(il)".to_owned());
                }
            }
            Arch::Moshi => {
                desc.kind = Kv;
                desc.n_ctx_seq = h.n_swa;
            }
            Arch::HyV4 => {
                if h.indexer_top_k == 0 {
                    desc.kind = Kv;
                } else {
                    desc.kind = KvDsa;
                    desc.filter_idx = Some("is_indexer_full(il)".to_owned());
                }
            }
            Arch::Dots3note | Arch::HyV4Flash => {
                if is_mtp && h.n_layer_nextn > 0 {
                    desc.kind = Kv;
                    desc.filter_attn = Some("il >= n_layer".to_owned());
                } else {
                    desc.kind = KvDsaIsWa;
                    if h.n_layer_nextn > 0 {
                        desc.filter_attn = Some("il < n_layer".to_owned());
                    }
                    desc.filter_idx = Some("il < n_layer && is_indexer_full(il)".to_owned());
                }
            }
            Arch::Deepseek41 | Arch::Deepseek4 => {
                if is_mtp {
                    desc.kind = KvIsWa;
                    desc.filter_attn = Some("il >= n_layer".to_owned());
                } else {
                    desc.kind = KvDsv4;
                }
            }
            Arch::Dflash if h.dsv4_hc_mult > 0 || h.n_lora_kv > 0 => {
                desc.kind = KvIsWa;
            }
            _ => {
                let mtp_on_hybrid_qwen = is_mtp
                    && matches!(
                        arch,
                        Arch::Qwen3next | Arch::Qwen35 | Arch::Qwen35moe | Arch::Bailingmoe3
                    );
                let mtp_on_hybrid_nemotron = is_mtp && arch == Arch::NemotronHMoe;
                if arch_is_recurrent(arch) {
                    desc.kind = Recurrent;
                } else if arch_is_hybrid(arch) && !mtp_on_hybrid_qwen && !mtp_on_hybrid_nemotron {
                    let needs_idx = matches!(arch, Arch::Qwen4exp | Arch::KimiK31);
                    desc.filter_attn = Some("!is_recr(il)".to_owned());
                    desc.filter_recr = Some("is_recr(il)".to_owned());
                    for a in [
                        Arch::FalconH1,
                        Arch::Inkling2,
                        Arch::Inkling2Loop,
                        Arch::Zaya1,
                    ] {
                        if arch == a {
                            desc.filter_attn = Some("true".to_owned());
                            desc.filter_recr = Some("true".to_owned());
                        }
                    }
                    if matches!(arch, Arch::NemotronH | Arch::NemotronHMoe) {
                        desc.filter_attn = Some("!is_recr(il) && n_ff(il)==0".to_owned());
                        desc.filter_recr = Some("is_recr(il) && n_ff(il)==0".to_owned());
                    }
                    if matches!(arch, Arch::Qwen4exp | Arch::KimiK31)
                        && arch == Arch::Qwen4exp
                        && h.indexer_head_size > 0
                    {
                        desc.filter_idx = Some("!is_recr(il)".to_owned());
                    }
                    if arch == Arch::Qwen4exp && is_mtp {
                        desc.filter_attn = Some("il >= n_layer".to_owned());
                        desc.filter_recr = Some("false".to_owned());
                        if desc.filter_idx.is_some() {
                            desc.filter_idx = Some("il >= n_layer".to_owned());
                        }
                    }
                    desc.kind = if h.swa_type != crate::hparams::SwaType::None {
                        HybridIsWa
                    } else if needs_idx {
                        HybridIdx
                    } else {
                        Hybrid
                    };
                } else {
                    desc.kind = Kv;
                    if matches!(arch, Arch::Gemma3n | Arch::Gemma4) {
                        desc.reuse = Some("has_kv(il) ? -1 : kv_source(il)".to_owned());
                    }
                    if mtp_on_hybrid_qwen || mtp_on_hybrid_nemotron {
                        desc.filter_attn = Some("il >= n_layer".to_owned());
                    }
                    if h.n_layer_nextn > 0 && h.n_layer() > 0 && h.router_layer < 0 {
                        desc.filter_attn = Some(
                            if is_mtp {
                                "il >= n_layer"
                            } else {
                                "il < n_layer"
                            }
                            .to_owned(),
                        );
                    }
                    if h.swa_type != crate::hparams::SwaType::None {
                        if arch == Arch::Gemma4Assistant {
                            desc.share = Some("swa ? n_layer_other-2 : n_layer_other-1".to_owned());
                        }
                        desc.kind = KvIsWa;
                    }
                }
            }
        }
        Ok(desc)
    }

    /// The sequence of graph-building phases from `llama_model::build_graph`.
    pub fn build_graph_plan(&self) -> GraphPlan {
        let mut steps = vec!["build_arch_graph".to_owned()];
        steps.push("build_pooling".to_owned());
        if self.output.is_some() {
            steps.push("build_logits".to_owned());
        }
        steps.push("build_sampling".to_owned());
        if self.dense_2_out_layers.is_some() || self.dense_3_out_layers.is_some() {
            steps.push("build_dense_out".to_owned());
        }
        steps.push("build_layer_inp_dev".to_owned());
        steps.push("build_h_nextn_dev".to_owned());
        steps.push("set_outputs".to_owned());
        GraphPlan { steps }
    }
}

/// Description of a graph build, in place of a ggml_cgraph pointer.
#[derive(Clone, Debug)]
pub struct GraphPlan {
    pub steps: Vec<String>,
}

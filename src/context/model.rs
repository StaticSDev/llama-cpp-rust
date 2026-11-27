//! Local model stand-in used by the context port.
//!
//! The C++ context reads hparams, vocabulary, architecture and tensors from
//! `llama_model`. The Rust tree has no model module yet, so this file owns the
//! small view the context needs. The graph builder stays in `graph.rs`.

use std::collections::{BTreeMap, BTreeSet};

use super::params::FusedOp;
use crate::core::{Error, Result};
use crate::hparams::Hparams;

/// Architectures consulted by the context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Other,
    Grok,
    T5,
    Dflash,
    Dspark,
    Gemma4Assistant,
    Eagle3,
    HyV4Flash,
    KimiLinear,
    MiniMax01,
    Qwen35,
    Qwen35Moe,
    Qwen4Exp,
    CloverMoe,
    KimiK3,
    KimiK31,
    Glm5Next,
    BailingMoe3,
    Raon1Mini,
    Deepseek2,
    Deepseek4,
    Deepseek41,
    InKling2Loop,
    GlmDsa,
    Dots3Note,
    MistralLarge4,
    HyV4,
    MiniMaxM3,
}

impl Arch {
    pub fn name(self) -> &'static str {
        match self {
            Self::Other => "other",
            Self::Grok => "grok",
            Self::T5 => "t5",
            Self::Dflash => "dflash",
            Self::Dspark => "dspark",
            Self::Gemma4Assistant => "gemma4-assistant",
            Self::Eagle3 => "eagle3",
            Self::HyV4Flash => "hy-v4-flash",
            Self::KimiLinear => "kimi-linear",
            Self::MiniMax01 => "minimax-01",
            Self::Qwen35 => "qwen35",
            Self::Qwen35Moe => "qwen35moe",
            Self::Qwen4Exp => "qwen4exp",
            Self::CloverMoe => "clover-moe",
            Self::KimiK3 => "kimi-k3",
            Self::KimiK31 => "kimi-k31",
            Self::Glm5Next => "glm5-next",
            Self::BailingMoe3 => "bailingmoe3",
            Self::Raon1Mini => "raon1-mini",
            Self::Deepseek2 => "deepseek2",
            Self::Deepseek4 => "deepseek4",
            Self::Deepseek41 => "deepseek41",
            Self::InKling2Loop => "inkling2-loop",
            Self::GlmDsa => "glm-dsa",
            Self::Dots3Note => "dots3note",
            Self::MistralLarge4 => "mistral-large4",
            Self::HyV4 => "hy-v4",
            Self::MiniMaxM3 => "minimax-m3",
        }
    }

    /// `llm_arch_supports_mixed_batch`.
    pub fn supports_mixed_batch(self) -> bool {
        matches!(self, Self::Dflash | Self::Dspark | Self::Eagle3 | Self::T5)
    }

    /// `llm_arch_supports_rs_rollback`.
    pub fn supports_rs_rollback(self) -> bool {
        matches!(
            self,
            Self::Qwen35 | Self::Qwen35Moe | Self::Qwen4Exp | Self::CloverMoe
        )
    }

    /// Archs that do a full KV replay after a partial recurrent rollback.
    pub fn uses_rs_replay(self) -> bool {
        matches!(
            self,
            Self::Qwen35
                | Self::Qwen35Moe
                | Self::Qwen4Exp
                | Self::CloverMoe
                | Self::KimiK3
                | Self::KimiK31
                | Self::Glm5Next
                | Self::BailingMoe3
                | Self::Raon1Mini
        )
    }

    /// Archs that reject a quantized V cache without flash attention.
    pub fn rejects_fp8_cache(self) -> bool {
        matches!(
            self,
            Self::Deepseek2
                | Self::Deepseek4
                | Self::Deepseek41
                | Self::GlmDsa
                | Self::Glm5Next
                | Self::Dots3Note
                | Self::MistralLarge4
                | Self::HyV4
                | Self::MiniMaxM3
                | Self::Qwen4Exp
                | Self::Dflash
        )
    }

    /// Archs whose reserve graph grows with a squared diagonal-decay input.
    pub fn reserve_single_seq(self) -> bool {
        matches!(self, Self::KimiLinear | Self::MiniMax01)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitMode {
    None,
    Layer,
    Tensor,
}

/// Vocabulary view used by the context.
#[derive(Clone, Debug, Default)]
pub struct VocabInfo {
    pub n_tokens: u32,
    pub n_cls_out: u32,
}

/// A LoRA adapter descriptor, used for node estimates and sameness checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoraAdapter {
    pub id: usize,
    pub scale_bits: u32,
    pub n_nodes: u32,
}

/// One active LoRA adapter and its scale in a context.
#[derive(Clone, Debug, PartialEq)]
pub struct LoraRef {
    pub id: usize,
    pub scale: f32,
}

/// Control vector descriptor. `apply` writes the host-side vector.
#[derive(Clone, Debug, Default)]
pub struct CVecAdapter {
    pub data: Vec<f32>,
    pub n_embd: usize,
    pub il_start: i32,
    pub il_end: i32,
    pub valid: bool,
}

impl CVecAdapter {
    pub fn apply(
        &mut self,
        model: &Model,
        data: &[f32],
        len: usize,
        n_embd: i32,
        il_start: i32,
        il_end: i32,
    ) -> Result<()> {
        if n_embd <= 0 || len != n_embd as usize * (model.hparams.n_layer() as usize + 1) {
            return Err(Error::new("cvec: invalid data size"));
        }
        if il_start < 0 || il_end < il_start || il_end as u32 > model.hparams.n_layer() + 1 {
            return Err(Error::new("cvec: invalid layer range"));
        }
        self.data = data.to_vec();
        self.n_embd = n_embd as usize;
        self.il_start = il_start;
        self.il_end = il_end;
        self.valid = true;
        Ok(())
    }
}

/// Hadamard rotation tables used by the reserved-graph check.
#[derive(Clone, Debug, Default)]
pub struct HadamardInfo {
    /// Tensor id -> rotation tensor id and hint.
    pub rot: BTreeMap<usize, (usize, i32)>,
    /// Tensor id -> inverse table id.
    pub inv: BTreeSet<usize>,
}

/// `llama_model` view consumed by `llama_context`.
#[derive(Clone, Debug)]
pub struct Model {
    pub arch: Arch,
    pub hparams: Hparams,
    pub vocab: VocabInfo,
    pub devices: Vec<String>,
    pub n_gpu_layers: u32,
    pub split_mode: SplitMode,
    pub no_alloc: bool,
    pub has_tensor_overrides: bool,
    pub has_tok_embd: bool,
    pub has_output: bool,
    pub loras: Vec<LoraAdapter>,
    pub hadamard: HadamardInfo,
    pub fused_support: BTreeSet<FusedOp>,
    pub nodes_per_tensor: u32,
    pub tensors: u32,
    pub model_bytes: u64,
    /// Tensor names offered to the optimizer parameter filter.
    pub trainable_tensors: Vec<String>,
}

impl Model {
    pub fn new(hparams: Hparams, vocab: VocabInfo) -> Self {
        let mut fused_support = BTreeSet::new();
        fused_support.insert(FusedOp::FlashAttn);
        fused_support.insert(FusedOp::FlashAttnKvRows);
        Self {
            arch: Arch::Other,
            hparams,
            vocab,
            devices: vec!["CPU".to_string()],
            n_gpu_layers: 0,
            split_mode: SplitMode::None,
            no_alloc: false,
            has_tensor_overrides: false,
            has_tok_embd: true,
            has_output: true,
            loras: Vec::new(),
            hadamard: HadamardInfo::default(),
            fused_support,
            nodes_per_tensor: 4,
            tensors: 1,
            model_bytes: 0,
            trainable_tensors: vec![
                "type_embd.weight".to_string(),
                "output_norm.weight".to_string(),
                "output.weight".to_string(),
                "token_embd.weight".to_string(),
                "rope_freqs.weight".to_string(),
            ],
        }
    }

    pub fn no_alloc(&self) -> bool {
        self.no_alloc
    }

    pub fn n_tensors(&self) -> u32 {
        self.tensors
    }

    pub fn graph_nodes_per_tensor(&self) -> u32 {
        self.nodes_per_tensor
    }

    pub fn dev_layer(&self, _il: u32) -> Option<String> {
        if self.devices.is_empty() {
            None
        } else {
            Some(self.devices[0].clone())
        }
    }

    pub fn dev_output(&self) -> Option<String> {
        self.devices.last().cloned()
    }

    pub fn memory_breakdown(&self) -> BTreeMap<String, usize> {
        let mut res = BTreeMap::new();
        let buft = self.dev_output().unwrap_or_else(|| "CPU".to_string());
        res.insert(buft, self.model_bytes as usize);
        res
    }
}

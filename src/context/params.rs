//! Port of `llama_context_params` and the constants around it.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use crate::cparams::{ContextType, PoolingType, RopeScalingType, LLAMA_MAX_SEQ};
use crate::kv_cache::GgmlType;

use super::LlamaContext;

/// `llama_attention_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionType {
    Unspecified,
    Causal,
    NonCausal,
}

/// `llama_flash_attn_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashAttnType {
    Disabled,
    Auto,
    Enabled,
}

/// `llm_graph_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphType {
    Default,
    Encoder,
    DecoderMtp,
}

/// `llm_fused_op`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FusedOp {
    FlashAttn,
    FlashAttnKvRows,
    Dsv4HcPre,
    Dsv4HcComb,
    Dsv4HcPost,
    TopKMask,
}

impl FusedOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::FlashAttn => "Flash Attention",
            Self::FlashAttnKvRows => "Flash Attention over K/V rows",
            Self::Dsv4HcPre => "fused DeepSeek V4 HC pre",
            Self::Dsv4HcComb => "fused DeepSeek V4 HC comb",
            Self::Dsv4HcPost => "fused DeepSeek V4 HC post",
            Self::TopKMask => "fused top-k mask",
        }
    }
}

/// Descriptor used by `resolve_fused_ops`.
#[derive(Clone, Copy, Debug)]
pub struct FusedOpProbe {
    pub op: FusedOp,
    pub name: &'static str,
    pub n_tokens_per_seq: u32,
}

pub const FUSED_PROBE_FLASH_ATTN: FusedOpProbe = FusedOpProbe {
    op: FusedOp::FlashAttn,
    name: "Flash Attention",
    n_tokens_per_seq: 1,
};
pub const FUSED_PROBE_FLASH_ATTN_KV_ROWS: FusedOpProbe = FusedOpProbe {
    op: FusedOp::FlashAttnKvRows,
    name: "Flash Attention over K/V rows",
    n_tokens_per_seq: 1,
};
pub const FUSED_PROBE_DSV4_HC_PRE: FusedOpProbe = FusedOpProbe {
    op: FusedOp::Dsv4HcPre,
    name: "fused DeepSeek V4 HC pre",
    n_tokens_per_seq: 1,
};
pub const FUSED_PROBE_DSV4_HC_COMB: FusedOpProbe = FusedOpProbe {
    op: FusedOp::Dsv4HcComb,
    name: "fused DeepSeek V4 HC comb",
    n_tokens_per_seq: 1,
};
pub const FUSED_PROBE_DSV4_HC_POST: FusedOpProbe = FusedOpProbe {
    op: FusedOp::Dsv4HcPost,
    name: "fused DeepSeek V4 HC post",
    n_tokens_per_seq: 1,
};
pub const FUSED_PROBE_TOP_K_MASK: FusedOpProbe = FusedOpProbe {
    op: FusedOp::TopKMask,
    name: "fused top-k mask",
    n_tokens_per_seq: 1,
};

/// Stand-in for a backend sampler chain assigned to one sequence.
#[derive(Clone, Debug)]
pub struct BackendSampler {
    pub name: String,
    pub n_chain: usize,
    pub can_offload: bool,
    pub n_nodes: usize,
    /// Outputs claimed in the current logical batch.
    pub n_outputs: u32,
}

impl BackendSampler {
    /// `llama_sampler_backend_begin`.
    pub fn begin(&mut self) {
        self.n_outputs = 0;
    }
}

/// `llama_context_params`.
#[derive(Clone)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: ContextType,
    pub rope_scaling_type: RopeScalingType,
    pub pooling_type: PoolingType,
    pub attention_type: AttentionType,
    pub flash_attn_type: FlashAttnType,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub type_k: GgmlType,
    pub type_v: GgmlType,
    pub moe_cache_size: usize,
    pub moe_cache_cpu_share: f32,
    pub abort_callback: Option<Arc<dyn Fn() -> bool>>,
    pub samplers: Vec<SamplerConfig>,
    pub ctx_other: Option<Rc<RefCell<LlamaContext>>>,
    pub ctx_state_src: Option<Rc<RefCell<LlamaContext>>>,
}

/// `llama_context_params::samplers` entry.
#[derive(Clone)]
pub struct SamplerConfig {
    pub seq_id: crate::core::SeqId,
    pub sampler: BackendSampler,
}

/// `llama_context_default_params()`.
pub fn default_params() -> ContextParams {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1);
    ContextParams {
        n_ctx: 512,
        n_batch: 2048,
        n_ubatch: 512,
        n_seq_max: 1,
        n_rs_seq: 0,
        n_outputs_max: 0,
        n_outputs_max_per_seq: 1,
        n_threads,
        n_threads_batch: n_threads,
        ctx_type: ContextType::Default,
        rope_scaling_type: RopeScalingType::Unspecified,
        pooling_type: PoolingType::Unspecified,
        attention_type: AttentionType::Unspecified,
        flash_attn_type: FlashAttnType::Auto,
        rope_freq_base: 0.0,
        rope_freq_scale: 0.0,
        yarn_ext_factor: -1.0,
        yarn_attn_factor: -1.0,
        yarn_beta_fast: -1.0,
        yarn_beta_slow: -1.0,
        yarn_orig_ctx: 0,
        embeddings: false,
        offload_kqv: true,
        no_perf: true,
        op_offload: true,
        swa_full: true,
        kv_unified: false,
        type_k: GgmlType::F16,
        type_v: GgmlType::F16,
        moe_cache_size: 0,
        moe_cache_cpu_share: 0.0,
        abort_callback: None,
        samplers: Vec::new(),
        ctx_other: None,
        ctx_state_src: None,
    }
}

impl Default for ContextParams {
    fn default() -> Self {
        default_params()
    }
}

/// `ggml_status` values used by `encode`, `decode` and `process_ubatch`.
pub mod status {
    pub const SUCCESS: i32 = 0;
    pub const ABORTED: i32 = 2;
    pub const ALLOC_FAILED: i32 = -2;
    pub const FAILED: i32 = -3;
}

/// `max` helper mirroring the C++ clamp on sequence count.
pub fn clamp_seq_max(n_seq_max: u32) -> u32 {
    n_seq_max.max(1).min(LLAMA_MAX_SEQ)
}

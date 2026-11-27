//! Port of `src/llama-cparams.h` and `src/llama-cparams.cpp`.

pub const LLAMA_MAX_SEQ: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextType {
    Default,
    Mtp,
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
pub enum PoolingType {
    Unspecified,
    None,
    Mean,
    Cls,
    Last,
    Rank,
}

/// `llama_cparams`: inference context parameters.
#[derive(Clone, Debug)]
pub struct CParams {
    pub n_ctx: u32,
    pub n_ctx_seq: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub rs_replay: bool,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,

    pub nextn_layer_offset: i32,

    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,

    pub n_ctx_orig_yarn: u32,
    // Not exposed in GGUF: every YaRN model uses the same values.
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,

    pub embeddings: bool,
    pub embeddings_nextn: bool,
    pub embeddings_nextn_masked: bool,
    pub causal_attn: bool,
    pub offload_kqv: bool,
    pub flash_attn: bool,
    pub auto_fa: bool,
    pub fused_dsv4_hc_pre: bool,
    pub fused_dsv4_hc_comb: bool,
    pub fused_dsv4_hc_post: bool,
    pub auto_fhc: bool,
    pub fused_top_k_mask: bool,
    pub auto_ftkm: bool,
    pub fused_kv_rows: bool,
    pub auto_fkvr: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub kv_unified: bool,
    pub pipeline_parallel: bool,
    pub training: bool,
    pub moe_random_routing: bool,

    pub moe_cache_size: usize,
    pub moe_cache_cpu_share: f32,

    pub embeddings_layer_inp: Vec<bool>,
    pub embeddings_layer_inp_dev: bool,
    pub embeddings_nextn_dev: bool,

    pub ctx_type: ContextType,
    pub rope_scaling_type: RopeScalingType,
    pub pooling_type: PoolingType,
}

impl Default for CParams {
    fn default() -> Self {
        Self {
            n_ctx: 0,
            n_ctx_seq: 0,
            n_batch: 0,
            n_ubatch: 0,
            n_seq_max: 0,
            n_rs_seq: 0,
            rs_replay: false,
            n_outputs_max: 0,
            n_outputs_max_per_seq: 0,
            n_threads: 0,
            n_threads_batch: 0,
            nextn_layer_offset: 0,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            n_ctx_orig_yarn: 0,
            yarn_ext_factor: 0.0,
            yarn_attn_factor: 0.0,
            yarn_beta_fast: 0.0,
            yarn_beta_slow: 0.0,
            embeddings: false,
            embeddings_nextn: false,
            embeddings_nextn_masked: false,
            causal_attn: false,
            offload_kqv: false,
            flash_attn: false,
            auto_fa: false,
            fused_dsv4_hc_pre: false,
            fused_dsv4_hc_comb: false,
            fused_dsv4_hc_post: false,
            auto_fhc: false,
            fused_top_k_mask: false,
            auto_ftkm: false,
            fused_kv_rows: false,
            auto_fkvr: false,
            no_perf: false,
            op_offload: false,
            kv_unified: false,
            pipeline_parallel: false,
            training: false,
            moe_random_routing: false,
            moe_cache_size: 0,
            moe_cache_cpu_share: 0.0,
            embeddings_layer_inp: Vec::new(),
            embeddings_layer_inp_dev: false,
            embeddings_nextn_dev: false,
            ctx_type: ContextType::Default,
            rope_scaling_type: RopeScalingType::Unspecified,
            pooling_type: PoolingType::Unspecified,
        }
    }
}

/// `llama_max_parallel_sequences`.
pub fn max_parallel_sequences() -> usize {
    LLAMA_MAX_SEQ as usize
}

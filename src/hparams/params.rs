use crate::core::{Token, LLAMA_TOKEN_NULL};

// bump if necessary
pub const LLAMA_MAX_LAYERS: usize = 512;
pub const LLAMA_MAX_EXPERTS: usize = 1024; // Kimi K3
pub const LLAMA_MAX_PLE_NGRAM: usize = 8;  // qwen4exp, deepseek41 (all hash modules)
pub const LLAMA_MAX_PLE_HEADS: usize = 64; // qwen4exp, deepseek41 (all hash modules)


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ExpertGatingFuncType {
    None = 0,
    Softmax = 1,
    Sigmoid = 2,
    /// Applied to the router weights instead of the logits.
    SoftmaxWeight = 3,
    SqrtSoftplus = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SwaType {
    None = 0,
    Standard = 1,
    Chunked = 2,
    Symmetric = 3,
}

/// How a non-causal mask is built with `llama_set_causal_attn(ctx, false)`
/// (e.g. mtmd decoding image tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum NonCausalType {
    /// All layers non-causal, SWA still applied (gemma 3, qwen-vl, ...).
    All = 0,
    /// SWA layers non-causal, dense layers stay causal (gemma 4).
    SwaOnly = 1,
    /// All layers non-causal, SWA not applied inside the current ubatch (deepseek 4).
    SwaFull = 2,
    /// Dense layers non-causal, SWA layers stay causal (lfm3 d1).
    DenseOnly = 3,
}

/// M-RoPE: which input position slot feeds each RoPE section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum RopeSectionOrder {
    Unspecified = -1,
    /// Default: slot i feeds section i.
    Tyxz = 0,
    /// MiniCPM-V 4.7: time last.
    Zyxt = 1,
}

/// Minimal stand-ins for enums declared in `llama.h` / `llama-graph.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum PoolingType {
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
    Rank = 4,
    Unspecified = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum RopeType {
    None = 0,
    Normal = 1,
    Mrope = 2,
    Imrope = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum RopeScalingType {
    None = 0,
    Linear = 1,
    Yarn = 2,
    Extrapolation = 3,
}

/// Resolved FFN gated activation flavor. `None` is the sentinel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FfnOpType {
    None = 0,
}

/// Activation of the classifier head (RANK).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum UnaryOp {
    Tanh = 0,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HparamsPosnet {
    pub n_embd: u32,
    pub n_layer: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HparamsConvnext {
    pub n_embd: u32,
    pub n_layer: u32,
}

#[derive(Clone, Debug)]
pub struct Hparams {
    // note: the `_impl` suffix avoids a name conflict between a member and its getter
    pub vocab_only: bool,
    pub rope_finetuned: bool,
    pub use_par_res: bool,
    pub swin_norm: bool,
    pub norm_before_residual: bool,
    pub norm_before_fc: bool,

    /// Context size the model was trained on.
    pub n_ctx_train: u32,
    pub n_embd: u32,
    pub n_layer_all: u32,
    pub n_layer_nextn: u32,
    /// Trailing blocks that form the decision head.
    pub n_layer_decision: u32,

    /// granite-switch: single-head "router" KV layer encoding per-token adapter
    /// selection. -1 when the model has no such layer.
    pub router_layer: i32,
    pub n_expert: u32,
    pub n_rel_attn_bkts: u32,
    /// MoVA value experts (K2 Horizon).
    pub n_value_expert: u32,
    pub n_value_expert_used: u32,

    // TODO: this needs to be reworked
    /// If non-negative, the first `n_layer_kv_from_start` layers have KV cache.
    pub n_layer_kv_from_start: i32,

    /// Layer il reads the KV cache of layer `kv_source_impl[il]` (own cache if equal to il).
    pub kv_source_impl: [u32; LLAMA_MAX_LAYERS],

    /// Different head size for full_attention and SWA layers.
    /// d_q is assumed equal to d_k; there are n_head q heads and only n_head_kv k-v heads.
    pub n_embd_head_k_full: u32,
    /// Dimension of values (d_v) aka n_embd_head.
    pub n_embd_head_v_full: u32,
    pub n_embd_head_k_swa: u32,
    pub n_embd_head_v_swa: u32,

    pub n_rot_full: u32,
    pub n_rot_swa: u32,

    /// deepseek2 MLA converts into MQA with larger heads, then decompresses to MHA.
    pub n_embd_head_k_mla_impl: u32,
    pub n_embd_head_v_mla_impl: u32,

    // for WavTokenizer
    pub posnet: HparamsPosnet,
    pub convnext: HparamsConvnext,

    pub n_shortconv_l_cache: u32,
    pub n_sconv_channels: u32,

    pub n_head_arr: [u32; LLAMA_MAX_LAYERS],
    pub n_head_kv_arr: [u32; LLAMA_MAX_LAYERS],
    pub n_ff_arr: [u32; LLAMA_MAX_LAYERS],

    /// Per-layer expert feed-forward size.
    pub n_ff_exp_arr: [u32; LLAMA_MAX_LAYERS],
    /// Per-layer top-k expert routing count.
    pub n_expert_used_arr: [u32; LLAMA_MAX_LAYERS],

    pub n_layer_dense_lead: u32,
    pub n_lora_q: u32,
    pub n_lora_kv: u32,
    pub n_ff_shexp: u32,
    pub n_ff_chexp: u32,
    pub n_expert_shared: u32,
    pub n_norm_groups: u32,
    pub n_expert_groups: u32,
    pub n_group_used: u32,
    pub n_group_experts: u32,

    // MLA + SWA (i.e. dots3note)
    pub n_lora_kv_swa: u32,
    pub n_embd_head_k_mla_swa: u32,
    pub n_embd_head_v_mla_swa: u32,

    pub expert_group_scale: f32,
    pub expert_weights_scale: f32,
    pub expert_weights_norm: bool,
    /// > 0: weights / (sum + eps), else the sum is clamped.
    pub expert_weights_norm_eps: f32,
    /// Compute the router logits in F32.
    pub expert_router_f32: bool,
    pub expert_gating_func: u32,
    pub moe_every_n_layers: u32,
    pub moe_latent_size: u32,

    pub f_norm_eps: f32,
    pub f_norm_rms_eps: f32,
    pub f_norm_group_eps: f32,
    /// RMS norm of the MLA q/kv latents (Raon-1 mini).
    pub f_norm_rms_eps_latent: f32,

    // post norms use their own epsilon
    pub f_post_norm_eps: f32,

    pub f_attn_logit_softcapping: f32,
    pub f_router_logit_softcapping: f32,
    pub f_final_logit_softcapping: f32,

    // for RWKV
    pub rescale_every_n_layers: u32,
    pub time_mix_extra_dim: u32,
    pub time_decay_extra_dim: u32,
    pub wkv_head_size: u32,
    pub token_shift_count: u32,
    pub n_lora_decay: u32,
    pub n_lora_iclr: u32,
    pub n_lora_value_res_mix: u32,
    pub n_lora_gate: u32,

    pub rope_attn_factor: f32,
    pub rope_freq_base_train: f32,
    pub rope_freq_base_train_swa: f32,
    pub rope_freq_scale_train: f32,
    pub rope_freq_scale_train_swa: f32,
    /// NTK-aware alpha for XDRoPE.
    pub rope_scaling_alpha: f32,

    pub n_ctx_orig_yarn: u32,
    pub rope_yarn_log_mul: f32,

    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,

    pub rope_sections: [i32; 4],
    pub rope_section_order: RopeSectionOrder,

    /// Per-layer RoPE enable flags (1 = RoPE, 0 = NoPE).
    /// By default all layers use RoPE (controlled by `rope_finetuned`).
    pub rope_pattern: [u32; LLAMA_MAX_LAYERS],

    // Sliding Window Attention
    pub swa_type: SwaType,
    /// Size of the sliding window (0 - no SWA).
    pub n_swa: u32,

    /// see `NonCausalType`
    /// note: for SwaFull, older tokens outside the current ubatch are still window-clipped
    pub non_causal_type: NonCausalType,

    /// 1 = SWA layer, 0 = dense layer. By default all layers are dense.
    /// u32 for compatibility reason.
    pub is_swa_impl: [u32; LLAMA_MAX_LAYERS],

    /// For hybrid state space models.
    pub is_recr_impl: [u32; LLAMA_MAX_LAYERS],

    // for State Space Models
    pub ssm_d_conv: u32,
    pub ssm_d_inner: u32,
    pub ssm_d_state: u32,
    pub ssm_dt_rank: u32,
    pub ssm_n_group: u32,
    /// 0 = use `f_norm_rms_eps`.
    pub ssm_norm_eps: f32,

    // for MiniMax-Text-01 linear attention
    pub n_embd_head_la: u32,

    // for Kimi Linear KDA
    pub n_embd_head_kda: u32,
    pub kda_safe_gate: bool,

    // for ZAYA1 CCA layers: q + k channels in the conv state
    pub n_cca_qk: u32,

    // for ZAYA1: width of the router MLP
    pub n_router_hidden: u32,

    // kimi-k3
    /// Routed expert hidden size (0 = experts run at n_embd).
    pub n_expert_latent: u32,
    /// 0 = no cross-layer attention residuals.
    pub attn_res_block_size: u32,
    pub kda_gate_lower_bound: f32,
    pub situ_beta: f32,
    /// 0 = no linear-beta transform on the up branch.
    pub situ_linear_beta: f32,

    // hrm-text (looped H/L stacks)
    pub n_hrm_layers_per_stack: u32,
    pub n_hrm_h_cycles: u32,
    pub n_hrm_l_cycles: u32,
    pub hrm_prefix_lm: bool,

    // metis16 (passes over shared layers, depth memory)
    pub metis_n_pass: u32,
    pub metis_n_mem: u32,
    pub metis_n_route: u32,
    pub metis_pass_lora_rank: u32,
    pub metis_cont_limit: f32,

    // limite
    pub f_xsa_eps: f32,
    /// scale, shift, temperature
    pub final_logit_sigmoid_capping: [f32; 3],
    pub n_mudd_ff: u32,
    pub n_mudd_taps: u32,
    /// [n_layer][n_mudd_taps], -1 = no mixing.
    pub mudd_tap_idx: [i32; LLAMA_MAX_LAYERS],

    // inkling2
    pub n_relative: u32,
    pub n_relative_extent: u32,
    pub n_vocab_unpadded: u32,
    pub n_vocab_padded: u32,
    pub n_log_scale_floor: u32,
    pub f_log_scale_alpha: f32,
    pub f_qk_clip: f32,
    pub f_relative_clip: f32,

    pub ssm_dt_b_c_rms: bool,

    pub f_clamp_kqv: f32,
    pub f_max_alibi_bias: f32,
    pub f_logit_scale: f32,

    // Additional scale factors (Granite/Granite MoE)
    pub f_residual_scale: f32,
    pub f_embedding_scale: f32,
    pub f_attention_scale: f32,

    // grok-2
    pub f_attn_out_scale: f32,
    pub attn_temp_length: u32,

    pub f_attn_value_scale: f32,

    pub causal_attn: bool,
    pub use_alibi: bool,
    pub attn_soft_cap: bool,
    pub use_kq_norm: bool,

    // for Classifiers
    pub n_cls_out: u32,

    /// Input embedding dimension (0 = use n_embd).
    pub n_embd_inp_impl: u32,

    /// Encoder input embedding dimension (0 = use `n_embd_inp()`).
    /// e.g. the eagle3 encoder fuses target_layers * target_hidden features.
    pub n_embd_inp_enc_impl: u32,

    /// Output embedding dimension (0 = use n_embd).
    pub n_embd_out_impl: u32,

    pub dflash_block_size: u32,
    pub dflash_conv_kernel_size: u32,
    pub dflash_conv_group_size: u32,
    pub dflash_selector_rank: u32,
    pub dflash_selector_top_k: u32,
    /// DFlash2 softcaps the selector scores instead of the logits.
    pub dflash_softcap_post: bool,
    /// rnn markov head reads f_rnn from embd_state.
    pub dspark_f_rnn: bool,
    pub lilicorr_hidden_size: u32,
    pub lilicorr_candidate_topk: u32,
    pub lilicorr_num_heads: u32,
    pub lilicorr_layers: u32,
    pub lilicorr_vector_eps: f32,
    pub lilicorr_logit_scale: f32,

    // llama4 smallthinker
    pub n_moe_layer_step: u32,
    pub n_no_rope_layer_step: u32,
    pub n_attn_temp_floor_scale: u32,
    pub f_attn_temp_scale: f32,
    /// Offset position index.
    pub f_attn_temp_offset: f32,
    /// Use (1 + (pos + offset) / floor_scale) ^ scale instead of the llama4 log form.
    pub attn_temp_pow: bool,

    /// Realtime models: tokens the output lags behind the audio input.
    pub n_tokens_delay: u32,

    /// Each sequence is paired with an unconditional one.
    pub classifier_free_guidance: bool,

    // gemma3n altup
    /// altup_num_inputs
    pub n_altup: u32,
    /// altup_active_idx
    pub i_altup_act: u32,
    pub laurel_rank: u32,
    pub n_embd_altup: u32,

    // needed for sentence-transformers dense layers
    /// in_features of the 2_Dense
    pub dense_2_feat_in: u32,
    /// out_features of the 2_Dense
    pub dense_2_feat_out: u32,
    /// in_features of the 3_Dense
    pub dense_3_feat_in: u32,
    /// out_features of the 3_Dense
    pub dense_3_feat_out: u32,

    // xIELU
    pub xielu_alpha_n: [f32; LLAMA_MAX_LAYERS],
    pub xielu_alpha_p: [f32; LLAMA_MAX_LAYERS],
    pub xielu_beta: [f32; LLAMA_MAX_LAYERS],
    pub xielu_eps: [f32; LLAMA_MAX_LAYERS],

    // DSA (deepseek sparse attention)
    pub indexer_n_head: u32,
    pub indexer_head_size: u32,
    pub indexer_top_k: u32,
    /// k-pool size
    pub indexer_kpool: u32,
    pub indexer_kpool_select_tail: bool,
    /// Head-size slots per cached indexer row; the last holds the pooled key.
    pub indexer_kpool_row: u32,
    /// Pools are consecutive cells in sequence order, not runs of consecutive positions.
    pub indexer_kpool_by_order: bool,
    // MSA
    pub indexer_block_size: u32,
    pub indexer_local_blocks: u32,

    /// Indexer is full (true) or shared (false).
    /// Shared indexers reuse top-k from the previous full layer.
    pub is_indexer_full_impl: [u32; LLAMA_MAX_LAYERS],

    // DeepSeek-V4
    pub dsv4_o_group_count: u32,
    pub dsv4_o_lora_rank: u32,
    pub dsv4_hc_mult: u32,
    /// DeepSeek-V4.1 DSpark draft: lagged pre-mix, no hc_head.
    pub dsv4_hc_single_pass: bool,
    pub dsv4_hc_sinkhorn_iters: u32,
    pub dsv4_hash_layer_count: u32,
    pub dsv4_compress_rope_base: f32,
    pub dsv4_hc_eps: f32,
    pub dsv4_compress_ratios: [u32; LLAMA_MAX_LAYERS],

    // deepseek41
    /// Layer whose compressed KV this layer reads, -1 = none.
    pub dsv41_kv_src: [i32; LLAMA_MAX_LAYERS],
    /// Layer whose top-k this layer uses, -1 = none.
    pub dsv41_topk_src: [i32; LLAMA_MAX_LAYERS],
    /// -1 = no candidate block selection.
    pub dsv41_cand_src: i32,
    pub dsv41_cand_topk_blocks: u32,
    pub dsv41_cand_block_size: u32,

    /// 0 = full rank (DeepSeek-V4).
    pub hc_low_rank: u32,

    /// Scale of the hyper-connection post gate (DeepSeek-V4 hardcodes 2.0).
    pub hc_magnitude: f32,

    pub ple_ngram_size: u32,
    pub ple_heads_per_ngram: u32,
    pub ple_conv_kernel: u32,
    /// (ngram_size - 1) * heads_per_ngram
    pub ple_n_heads: u32,
    pub ple_head_dim: u32,
    pub ple_eos_token_id: u32,
    /// Id the PLE hash stands in for at image positions; 0 falls back to EOS.
    pub ple_image_token_id: u32,
    /// Id the KV cells record at embedding (image) positions for the n-gram history.
    pub ple_embd_token: Token,
    /// Hash constant sets: 1 shared by all PLE layers (qwen4exp), or 1 per layer (deepseek41).
    pub ple_n_modules: u32,
    /// deepseek41 Engram: token that pads the n-gram history.
    pub engram_pad_id: u32,
    /// The file lists PLE layer indices, so one bit per layer is enough.
    pub is_ple_impl: [bool; LLAMA_MAX_LAYERS],
    /// Hash multipliers reach ~2e13 and must stay 64-bit.
    pub ple_layer_multipliers: [u64; LLAMA_MAX_PLE_NGRAM],
    /// Head offsets and vocab sizes are token-space indices; the gather truncates to int32.
    pub ple_head_offsets: [u32; LLAMA_MAX_PLE_HEADS],
    pub ple_head_vocab_sizes: [u32; LLAMA_MAX_PLE_HEADS],

    // qwen3vl deepstack
    // When parsed from GGUF, the first N layers consume the first N deepstack
    // embeddings. Use `deepstack_mapping_arr` for a more complex mapping; set
    // `n_deepstack_layers` to the number of unique deepstack layers so
    // `n_embd_inp()` stays accurate (see granite.cpp).
    // TODO: can be expressed via the new `n_embd_inp_impl` and remove this param
    pub n_deepstack_layers: u32,

    /// Deepstack layer array (Granite4 Vision): -1 = none, >=0 = embedding index.
    pub deepstack_mapping_arr: [i32; LLAMA_MAX_LAYERS],

    // gemma4 per-layer embedding
    pub n_embd_per_layer: u32,

    // needed by encoder-decoder models (e.g. T5, FLAN-T5)
    pub dec_start_token_id: Token,
    pub dec_n_layer: u32,

    pub pooling_type: PoolingType,
    /// Pooling before the classifier head (RANK).
    pub pooling_type_cls: PoolingType,
    pub rope_type: RopeType,
    pub rope_scaling_type_train: RopeScalingType,

    // Resolved FFN gated activation flavor for archs that read
    // `<arch>.hidden_activation` from the GGUF (e.g. ModernBert derivatives).
    // Defaults to `FfnOpType::None` (sentinel = 0); the mapping from the GGUF
    // string to a real op is done at hparam-load time, mirroring
    // `rope_scaling_type_train`.
    pub llm_ffn_op: FfnOpType,
    pub act_cls: UnaryOp,

    // Step35: optional per-layer clamps for (Swi)GLU
    /// Clamping for expert FFN.
    pub swiglu_clamp_exp: [f32; LLAMA_MAX_LAYERS],
    /// Shared expert.
    pub swiglu_clamp_shexp: [f32; LLAMA_MAX_LAYERS],
    /// Dense FFN.
    pub swiglu_clamp: [f32; LLAMA_MAX_LAYERS],
}

impl Default for Hparams {
    fn default() -> Self {
        Self {
            vocab_only: false,
            rope_finetuned: false,
            use_par_res: false,
            swin_norm: false,
            norm_before_residual: false,
            norm_before_fc: false,

            n_ctx_train: 0,
            n_embd: 0,
            n_layer_all: 0,
            n_layer_nextn: 0,
            n_layer_decision: 0,

            router_layer: -1,
            n_expert: 0,
            n_rel_attn_bkts: 0,
            n_value_expert: 0,
            n_value_expert_used: 0,

            n_layer_kv_from_start: -1,
            kv_source_impl: [0; LLAMA_MAX_LAYERS],

            n_embd_head_k_full: 0,
            n_embd_head_v_full: 0,
            n_embd_head_k_swa: 0,
            n_embd_head_v_swa: 0,

            n_rot_full: 0,
            n_rot_swa: 0,

            n_embd_head_k_mla_impl: 0,
            n_embd_head_v_mla_impl: 0,

            posnet: HparamsPosnet::default(),
            convnext: HparamsConvnext::default(),

            n_shortconv_l_cache: 0,
            n_sconv_channels: 0,

            n_head_arr: [0; LLAMA_MAX_LAYERS],
            n_head_kv_arr: [0; LLAMA_MAX_LAYERS],
            n_ff_arr: [0; LLAMA_MAX_LAYERS],
            n_ff_exp_arr: [0; LLAMA_MAX_LAYERS],
            n_expert_used_arr: [0; LLAMA_MAX_LAYERS],

            n_layer_dense_lead: 0,
            n_lora_q: 0,
            n_lora_kv: 0,
            n_ff_shexp: 0,
            n_ff_chexp: 0,
            n_expert_shared: 0,
            n_norm_groups: 0,
            n_expert_groups: 0,
            n_group_used: 0,
            n_group_experts: 0,

            n_lora_kv_swa: 0,
            n_embd_head_k_mla_swa: 0,
            n_embd_head_v_mla_swa: 0,

            expert_group_scale: 0.05,
            expert_weights_scale: 0.0,
            expert_weights_norm: false,
            expert_weights_norm_eps: 0.0,
            expert_router_f32: false,
            expert_gating_func: ExpertGatingFuncType::None as u32,
            moe_every_n_layers: 0,
            moe_latent_size: 0,

            f_norm_eps: 0.0,
            f_norm_rms_eps: 0.0,
            f_norm_group_eps: 0.0,
            f_norm_rms_eps_latent: 0.0,

            f_post_norm_eps: 0.0,

            f_attn_logit_softcapping: 50.0,
            f_router_logit_softcapping: 30.0,
            f_final_logit_softcapping: 30.0,

            rescale_every_n_layers: 0,
            time_mix_extra_dim: 0,
            time_decay_extra_dim: 0,
            wkv_head_size: 0,
            token_shift_count: 2,
            n_lora_decay: 0,
            n_lora_iclr: 0,
            n_lora_value_res_mix: 0,
            n_lora_gate: 0,

            rope_attn_factor: 1.0,
            rope_freq_base_train: 0.0,
            rope_freq_base_train_swa: 10000.0,
            rope_freq_scale_train: 0.0,
            rope_freq_scale_train_swa: 1.0,
            rope_scaling_alpha: 0.0,

            n_ctx_orig_yarn: 0,
            rope_yarn_log_mul: 0.0,

            yarn_ext_factor: -1.0,
            yarn_attn_factor: 1.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,

            rope_sections: [0; 4],
            rope_section_order: RopeSectionOrder::Tyxz,
            rope_pattern: [0; LLAMA_MAX_LAYERS],

            swa_type: SwaType::None,
            n_swa: 0,
            non_causal_type: NonCausalType::All,
            is_swa_impl: [0; LLAMA_MAX_LAYERS],
            is_recr_impl: [0; LLAMA_MAX_LAYERS],

            ssm_d_conv: 0,
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_dt_rank: 0,
            ssm_n_group: 0,
            ssm_norm_eps: 0.0,

            n_embd_head_la: 0,

            n_embd_head_kda: 0,
            kda_safe_gate: false,

            n_cca_qk: 0,

            n_router_hidden: 0,

            n_expert_latent: 0,
            attn_res_block_size: 0,
            kda_gate_lower_bound: f32::NEG_INFINITY,
            situ_beta: 1.0,
            situ_linear_beta: 0.0,

            n_hrm_layers_per_stack: 0,
            n_hrm_h_cycles: 0,
            n_hrm_l_cycles: 0,
            hrm_prefix_lm: false,

            metis_n_pass: 0,
            metis_n_mem: 0,
            metis_n_route: 0,
            metis_pass_lora_rank: 0,
            metis_cont_limit: 0.0,

            f_xsa_eps: 0.0,
            final_logit_sigmoid_capping: [0.0; 3],
            n_mudd_ff: 0,
            n_mudd_taps: 0,
            mudd_tap_idx: [0; LLAMA_MAX_LAYERS],

            n_relative: 0,
            n_relative_extent: 0,
            n_vocab_unpadded: 0,
            n_vocab_padded: 0,
            n_log_scale_floor: 0,
            f_log_scale_alpha: 0.0,
            f_qk_clip: 0.0,
            f_relative_clip: 0.0,

            ssm_dt_b_c_rms: false,

            f_clamp_kqv: 0.0,
            f_max_alibi_bias: 0.0,
            f_logit_scale: 0.0,

            f_residual_scale: 0.0,
            f_embedding_scale: 0.0,
            f_attention_scale: 0.0,

            f_attn_out_scale: 0.0,
            attn_temp_length: 0,

            f_attn_value_scale: 0.0,

            causal_attn: true,
            use_alibi: false,
            attn_soft_cap: false,
            use_kq_norm: false,

            n_cls_out: 1,

            n_embd_inp_impl: 0,
            n_embd_inp_enc_impl: 0,
            n_embd_out_impl: 0,

            dflash_block_size: 0,
            dflash_conv_kernel_size: 0,
            dflash_conv_group_size: 0,
            dflash_selector_rank: 0,
            dflash_selector_top_k: 0,
            dflash_softcap_post: false,
            dspark_f_rnn: false,
            lilicorr_hidden_size: 0,
            lilicorr_candidate_topk: 0,
            lilicorr_num_heads: 0,
            lilicorr_layers: 0,
            lilicorr_vector_eps: 0.0,
            lilicorr_logit_scale: 0.0,

            n_moe_layer_step: 0,
            n_no_rope_layer_step: 4,
            n_attn_temp_floor_scale: 0,
            f_attn_temp_scale: 0.0,
            f_attn_temp_offset: 0.0,
            attn_temp_pow: false,

            n_tokens_delay: 0,

            classifier_free_guidance: false,

            n_altup: 4,
            i_altup_act: 0,
            laurel_rank: 64,
            n_embd_altup: 256,

            dense_2_feat_in: 0,
            dense_2_feat_out: 0,
            dense_3_feat_in: 0,
            dense_3_feat_out: 0,

            xielu_alpha_n: [0.0; LLAMA_MAX_LAYERS],
            xielu_alpha_p: [0.0; LLAMA_MAX_LAYERS],
            xielu_beta: [0.0; LLAMA_MAX_LAYERS],
            xielu_eps: [0.0; LLAMA_MAX_LAYERS],

            indexer_n_head: 0,
            indexer_head_size: 0,
            indexer_top_k: 0,
            indexer_kpool: 0,
            indexer_kpool_select_tail: true,
            indexer_kpool_row: 3,
            indexer_kpool_by_order: false,
            indexer_block_size: 0,
            indexer_local_blocks: 0,

            is_indexer_full_impl: [0; LLAMA_MAX_LAYERS],

            dsv4_o_group_count: 0,
            dsv4_o_lora_rank: 0,
            dsv4_hc_mult: 0,
            dsv4_hc_single_pass: false,
            dsv4_hc_sinkhorn_iters: 0,
            dsv4_hash_layer_count: 0,
            dsv4_compress_rope_base: 0.0,
            dsv4_hc_eps: 0.0,
            dsv4_compress_ratios: [0; LLAMA_MAX_LAYERS],

            dsv41_kv_src: [0; LLAMA_MAX_LAYERS],
            dsv41_topk_src: [0; LLAMA_MAX_LAYERS],
            dsv41_cand_src: -1,
            dsv41_cand_topk_blocks: 0,
            dsv41_cand_block_size: 0,

            hc_low_rank: 0,
            hc_magnitude: 0.0,

            ple_ngram_size: 0,
            ple_heads_per_ngram: 0,
            ple_conv_kernel: 0,
            ple_n_heads: 0,
            ple_head_dim: 0,
            ple_eos_token_id: 0,
            ple_image_token_id: 0,
            ple_embd_token: LLAMA_TOKEN_NULL,
            ple_n_modules: 1,
            engram_pad_id: 0,
            is_ple_impl: [false; LLAMA_MAX_LAYERS],
            ple_layer_multipliers: [0; LLAMA_MAX_PLE_NGRAM],
            ple_head_offsets: [0; LLAMA_MAX_PLE_HEADS],
            ple_head_vocab_sizes: [0; LLAMA_MAX_PLE_HEADS],

            n_deepstack_layers: 0,
            deepstack_mapping_arr: [0; LLAMA_MAX_LAYERS],
            n_embd_per_layer: 0,

            dec_start_token_id: LLAMA_TOKEN_NULL,
            dec_n_layer: 0,

            pooling_type: PoolingType::None,
            pooling_type_cls: PoolingType::Unspecified,
            rope_type: RopeType::None,
            rope_scaling_type_train: RopeScalingType::None,

            llm_ffn_op: FfnOpType::None,
            act_cls: UnaryOp::Tanh,

            swiglu_clamp_exp: [0.0; LLAMA_MAX_LAYERS],
            swiglu_clamp_shexp: [0.0; LLAMA_MAX_LAYERS],
            swiglu_clamp: [0.0; LLAMA_MAX_LAYERS],
        }
    }
}

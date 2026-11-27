//! `llama_model` itself: construction, owned state and the load pipeline.

use super::buffers::{assign_devices, ModelImpl};
use super::types::*;
use super::{hparams as hp, tensors as tn, LlamaModelLoader};
use crate::arch::{arch_supports_sm_tensor, Arch};
use crate::core::{Error, Result, Token};
use crate::hparams::Hparams;
use crate::vocab::LlamaVocab;
use std::collections::{BTreeMap, BTreeSet};

/// Structural port of `struct llama_model`.
pub struct LlamaModel {
    pub type_: LlmType,
    pub arch: Arch,
    pub name: String,
    pub hparams: Hparams,
    pub vocab: LlamaVocab,
    pub prec_policy: PrecPolicy,
    pub hdmd: Hadamard,
    /// Weight name -> scale tensor name.
    pub w_scales: BTreeMap<String, String>,
    pub attn_rot: bool,
    /// `llama_model_dflash::variant`, stored on the base model for the port.
    pub dflash_variant: DflashVariant,
    pub classifier_labels: Vec<String>,
    pub decision_labels: Vec<String>,
    pub engram_token_map: Vec<i32>,
    pub engram_multipliers: Vec<u64>,

    // model-level tensors
    pub tok_embd: Option<Tensor>,
    pub type_embd: Option<Tensor>,
    pub pos_embd: Option<Tensor>,
    pub tok_norm: Option<Tensor>,
    pub tok_norm_b: Option<Tensor>,
    pub output_norm: Option<Tensor>,
    pub output_res_score: Option<Tensor>,
    pub output_res_query: Option<Tensor>,
    pub output_norm_b: Option<Tensor>,
    pub output: Option<Tensor>,
    pub output_b: Option<Tensor>,
    pub output_norm_enc: Option<Tensor>,
    pub output_s: Option<Tensor>,
    pub output_in_s: Option<Tensor>,
    pub nextn_proj_pre: Option<Tensor>,
    pub nextn_proj_post: Option<Tensor>,
    pub hrm_z_l_init: Option<Tensor>,
    pub stream_embd: Option<Tensor>,
    pub depth_mem_state_write: Option<Tensor>,
    pub depth_mem_meta_write: Option<Tensor>,
    pub depth_mem_entry_bias: Option<Tensor>,
    pub depth_mem_query: Option<Tensor>,
    pub depth_mem_query_b: Option<Tensor>,
    pub depth_mem_key: Option<Tensor>,
    pub depth_mem_key_b: Option<Tensor>,
    pub depth_mem_value: Option<Tensor>,
    pub depth_mem_value_b: Option<Tensor>,
    pub depth_mem_out: Option<Tensor>,
    pub depth_mem_out_b: Option<Tensor>,
    pub depth_mem_route_state: Option<Tensor>,
    pub depth_mem_route_retr: Option<Tensor>,
    pub depth_mem_route_diff: Option<Tensor>,
    pub depth_mem_route_hist: Option<Tensor>,
    pub depth_mem_route_b: Option<Tensor>,
    pub depth_mem_stream_gate: Option<Tensor>,
    pub depth_mem_stream_gate_b: Option<Tensor>,
    pub cont_hidden_state: Option<Tensor>,
    pub cont_hidden_retr: Option<Tensor>,
    pub cont_hidden_diff: Option<Tensor>,
    pub cont_hidden_route: Option<Tensor>,
    pub cont_hidden_b: Option<Tensor>,
    pub cont_output: Option<Tensor>,
    pub cont_output_b: Option<Tensor>,
    pub hc_head_fn: Option<Tensor>,
    pub hc_head_base: Option<Tensor>,
    pub hc_head_scale: Option<Tensor>,
    pub cls: Option<Tensor>,
    pub cls_b: Option<Tensor>,
    pub cls_out: Option<Tensor>,
    pub cls_out_b: Option<Tensor>,
    pub cls_norm: Option<Tensor>,
    pub cls_norm_b: Option<Tensor>,
    pub conv1d: Option<Tensor>,
    pub conv1d_b: Option<Tensor>,
    pub altup_proj: Option<Tensor>,
    pub altup_unembd_proj: Option<Tensor>,
    pub per_layer_tok_embd: Option<Tensor>,
    pub per_layer_tok_embd_s: Option<Tensor>,
    pub per_layer_tok_embd_scale: f32,
    pub value_embd: Option<Tensor>,
    pub hc_head_norm: Option<Tensor>,
    pub hc_head_down: Option<Tensor>,
    pub hc_head_up: Option<Tensor>,
    pub per_layer_model_proj: Option<Tensor>,
    pub per_layer_proj_norm: Option<Tensor>,
    pub fc: Option<Tensor>,
    pub fc_s: Option<Tensor>,
    pub fc_in_s: Option<Tensor>,
    pub d2t: Option<Tensor>,
    pub dspark_markov_w1: Option<Tensor>,
    pub dspark_markov_w2: Option<Tensor>,
    pub dspark_markov_w2_s: Option<Tensor>,
    pub dspark_conf_proj: Option<Tensor>,
    pub dspark_conf_proj_b: Option<Tensor>,
    pub dspark_joint_proj: Option<Tensor>,
    pub dspark_joint_proj_b: Option<Tensor>,
    pub dspark_mask_embd: Option<Tensor>,
    pub dspark_markov_w2_b: Option<Tensor>,
    pub dspark_markov_ctx_mix: Option<Tensor>,
    pub dspark_markov_ctx_norm: Option<Tensor>,
    pub res_scale_in: Option<Tensor>,
    pub res_scale_in_b: Option<Tensor>,
    pub dflash_selector_prev: Option<Tensor>,
    pub dflash_selector_next: Option<Tensor>,
    pub dflash_selector_hidden: Option<Tensor>,
    pub lilicorr_token_proj: Option<Tensor>,
    pub lilicorr_token_proj_b: Option<Tensor>,
    pub lilicorr_token_proj_s: Option<Tensor>,
    pub lilicorr_pass_hidden_proj: Option<Tensor>,
    pub lilicorr_pass_hidden_proj_b: Option<Tensor>,
    pub lilicorr_pass_hidden_proj_s: Option<Tensor>,
    pub lilicorr_feature_norm: Option<Tensor>,
    pub lilicorr_feature_norm_b: Option<Tensor>,
    pub lilicorr_feature_mlp_1: Option<Tensor>,
    pub lilicorr_feature_mlp_1_b: Option<Tensor>,
    pub lilicorr_feature_mlp_3: Option<Tensor>,
    pub lilicorr_feature_mlp_3_b: Option<Tensor>,
    pub lilicorr_slot_embedding: Option<Tensor>,
    pub lilicorr_rank_embedding: Option<Tensor>,
    pub lilicorr_attention_bias: Option<Tensor>,
    pub lilicorr_context_proj: Option<Tensor>,
    pub lilicorr_context_proj_b: Option<Tensor>,
    pub lilicorr_context_proj_s: Option<Tensor>,
    pub lilicorr_output_norm: Option<Tensor>,
    pub lilicorr_anchor_norm: Option<Tensor>,
    pub lilicorr_factor_input_proj: Option<Tensor>,
    pub lilicorr_factor_input_proj_b: Option<Tensor>,
    pub lilicorr_out_head: Option<Tensor>,
    pub lilicorr_out_head_b: Option<Tensor>,
    pub lilicorr_in_head: Option<Tensor>,
    pub lilicorr_in_head_b: Option<Tensor>,
    pub lilicorr_anchor_out_head: Option<Tensor>,
    pub lilicorr_anchor_out_head_b: Option<Tensor>,

    pub target_layer_ids: Vec<i32>,
    pub layers: Vec<Layer>,
    pub dense_2_out_layers: Option<Tensor>,
    pub dense_2_out_layers_b: Option<Tensor>,
    pub dense_3_out_layers: Option<Tensor>,

    pub gguf_kv: BTreeMap<String, String>,
    pub gguf_kv_arr: BTreeMap<String, Vec<String>>,
    pub devices: Vec<Device>,
    pub tensors_by_name: Vec<(String, Tensor)>,
    pub loras: BTreeSet<String>,
    pub can_prefetch: Vec<String>,
    pub t_load_us: i64,
    pub t_start_us: i64,

    pub params: ModelParams,
    pub impl_state: ModelImpl,
}

impl LlamaModel {
    /// `llama_model_create(arch, params)`.
    pub fn create(arch: Arch, params: ModelParams) -> Result<Self> {
        if arch == Arch::Unknown {
            return Err(Error::new("unknown model architecture: 'unknown'"));
        }
        if params.split_mode == SplitMode::Tensor && !arch_supports_sm_tensor(arch) {
            return Err(Error::new(format!(
                "LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '{}'",
                arch.name()
            )));
        }
        let mut res = Self::new(arch, params);
        res.impl_state = assign_devices(&res)?;
        Ok(res)
    }

    /// `llama_model_create(ml, params)`.
    pub fn create_from_loader(ml: &LlamaModelLoader, params: ModelParams) -> Result<Self> {
        if ml.arch == Arch::Unknown {
            return Err(Error::new(format!(
                "unknown model architecture: '{}'",
                ml.arch.name()
            )));
        }
        Self::create(ml.arch, params)
    }

    pub fn new(arch: Arch, params: ModelParams) -> Self {
        Self {
            type_: LlmType::Unknown,
            arch,
            name: "n/a".to_owned(),
            hparams: Hparams::default(),
            vocab: LlamaVocab::new(),
            prec_policy: PrecPolicy::default(),
            hdmd: Hadamard::default(),
            w_scales: BTreeMap::new(),
            attn_rot: true,
            dflash_variant: DflashVariant::None,
            classifier_labels: Vec::new(),
            decision_labels: Vec::new(),
            engram_token_map: Vec::new(),
            engram_multipliers: Vec::new(),
            tok_embd: None,
            type_embd: None,
            pos_embd: None,
            tok_norm: None,
            tok_norm_b: None,
            output_norm: None,
            output_res_score: None,
            output_res_query: None,
            output_norm_b: None,
            output: None,
            output_b: None,
            output_norm_enc: None,
            output_s: None,
            output_in_s: None,
            nextn_proj_pre: None,
            nextn_proj_post: None,
            hrm_z_l_init: None,
            stream_embd: None,
            depth_mem_state_write: None,
            depth_mem_meta_write: None,
            depth_mem_entry_bias: None,
            depth_mem_query: None,
            depth_mem_query_b: None,
            depth_mem_key: None,
            depth_mem_key_b: None,
            depth_mem_value: None,
            depth_mem_value_b: None,
            depth_mem_out: None,
            depth_mem_out_b: None,
            depth_mem_route_state: None,
            depth_mem_route_retr: None,
            depth_mem_route_diff: None,
            depth_mem_route_hist: None,
            depth_mem_route_b: None,
            depth_mem_stream_gate: None,
            depth_mem_stream_gate_b: None,
            cont_hidden_state: None,
            cont_hidden_retr: None,
            cont_hidden_diff: None,
            cont_hidden_route: None,
            cont_hidden_b: None,
            cont_output: None,
            cont_output_b: None,
            hc_head_fn: None,
            hc_head_base: None,
            hc_head_scale: None,
            cls: None,
            cls_b: None,
            cls_out: None,
            cls_out_b: None,
            cls_norm: None,
            cls_norm_b: None,
            conv1d: None,
            conv1d_b: None,
            altup_proj: None,
            altup_unembd_proj: None,
            per_layer_tok_embd: None,
            per_layer_tok_embd_s: None,
            per_layer_tok_embd_scale: 1.0,
            value_embd: None,
            hc_head_norm: None,
            hc_head_down: None,
            hc_head_up: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            fc: None,
            fc_s: None,
            fc_in_s: None,
            d2t: None,
            dspark_markov_w1: None,
            dspark_markov_w2: None,
            dspark_markov_w2_s: None,
            dspark_conf_proj: None,
            dspark_conf_proj_b: None,
            dspark_joint_proj: None,
            dspark_joint_proj_b: None,
            dspark_mask_embd: None,
            dspark_markov_w2_b: None,
            dspark_markov_ctx_mix: None,
            dspark_markov_ctx_norm: None,
            res_scale_in: None,
            res_scale_in_b: None,
            dflash_selector_prev: None,
            dflash_selector_next: None,
            dflash_selector_hidden: None,
            lilicorr_token_proj: None,
            lilicorr_token_proj_b: None,
            lilicorr_token_proj_s: None,
            lilicorr_pass_hidden_proj: None,
            lilicorr_pass_hidden_proj_b: None,
            lilicorr_pass_hidden_proj_s: None,
            lilicorr_feature_norm: None,
            lilicorr_feature_norm_b: None,
            lilicorr_feature_mlp_1: None,
            lilicorr_feature_mlp_1_b: None,
            lilicorr_feature_mlp_3: None,
            lilicorr_feature_mlp_3_b: None,
            lilicorr_slot_embedding: None,
            lilicorr_rank_embedding: None,
            lilicorr_attention_bias: None,
            lilicorr_context_proj: None,
            lilicorr_context_proj_b: None,
            lilicorr_context_proj_s: None,
            lilicorr_output_norm: None,
            lilicorr_anchor_norm: None,
            lilicorr_factor_input_proj: None,
            lilicorr_factor_input_proj_b: None,
            lilicorr_out_head: None,
            lilicorr_out_head_b: None,
            lilicorr_in_head: None,
            lilicorr_in_head_b: None,
            lilicorr_anchor_out_head: None,
            lilicorr_anchor_out_head_b: None,
            target_layer_ids: Vec::new(),
            layers: Vec::new(),
            dense_2_out_layers: None,
            dense_2_out_layers_b: None,
            dense_3_out_layers: None,
            gguf_kv: BTreeMap::new(),
            gguf_kv_arr: BTreeMap::new(),
            devices: params.devices.clone(),
            tensors_by_name: Vec::new(),
            loras: BTreeSet::new(),
            can_prefetch: Vec::new(),
            t_load_us: 0,
            t_start_us: 0,
            params,
            impl_state: ModelImpl::default(),
        }
    }

    /// Full load pipeline: stats, hparams, vocab and tensors.
    pub fn load(&mut self, ml: &mut LlamaModelLoader) -> Result<bool> {
        self.load_stats(ml);
        hp::load_hparams(self, ml)?;
        hp::load_vocab(self, ml)?;
        let done = tn::load_tensors(self, ml)?;
        Ok(done)
    }

    /// `load_stats`.
    pub fn load_stats(&mut self, ml: &LlamaModelLoader) {
        self.impl_state.n_elements = ml.n_elements;
        self.impl_state.n_bytes = ml.n_bytes;
    }

    /// `llama_model::arch_name`.
    pub fn arch_name(&self) -> String {
        crate::arch::arch_name(self.arch).to_owned()
    }

    /// `llama_model::type_name`.
    pub fn type_name(&self) -> String {
        self.type_.name().to_owned()
    }

    /// `llama_model::desc`.
    pub fn desc(&self) -> String {
        self.impl_state.desc_str.clone()
    }

    /// `llama_model::ftype`.
    pub fn ftype(&self) -> Ftype {
        self.impl_state.ftype
    }

    /// `llama_model::size` (file size in bytes).
    pub fn size(&self) -> usize {
        self.impl_state.n_bytes
    }

    /// `llama_model::n_elements` (total parameter count).
    pub fn n_elements(&self) -> u64 {
        self.impl_state.n_elements
    }

    /// `llama_model::n_tensors`.
    pub fn n_tensors(&self) -> usize {
        self.tensors_by_name.len()
    }

    /// `llama_model::n_devices`.
    pub fn n_devices(&self) -> usize {
        self.devices.len()
    }

    /// `llama_model::tensor_split`.
    pub fn tensor_split(&self) -> &[f32] {
        if self.impl_state.tensor_split_owned.is_empty() {
            &self.params.tensor_split
        } else {
            &self.impl_state.tensor_split_owned
        }
    }

    /// `llama_model::n_gpu_layers`: plus one for the output layer.
    pub fn n_gpu_layers(&self) -> u32 {
        if self.params.n_gpu_layers >= 0 {
            self.params.n_gpu_layers as u32
        } else {
            self.hparams.n_layer_all + 1
        }
    }

    pub fn split_mode(&self) -> SplitMode {
        self.params.split_mode
    }

    pub fn no_alloc(&self) -> bool {
        self.params.no_alloc
    }

    pub fn has_tensor_overrides(&self) -> bool {
        self.impl_state.has_tensor_overrides
    }

    /// `llama_model::get_tensor`.
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors_by_name
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
    }

    pub fn get_tensor_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.tensors_by_name
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
    }

    /// `llama_model::get_w_scale`.
    pub fn get_w_scale(&self, w: &Tensor) -> Option<&Tensor> {
        self.w_scales
            .get(&w.name)
            .and_then(|name| self.get_tensor(name))
    }

    /// `llama_model::dev_layer`.
    pub fn dev_layer(&self, il: usize) -> Result<usize> {
        self.impl_state
            .dev_layer
            .get(il)
            .map(|d| d.device)
            .ok_or_else(|| Error::new(format!("layer device out of range: {}", il)))
    }

    pub fn dev_output(&self) -> usize {
        self.impl_state.dev_output.device
    }

    /// `llama_model::select_buft`.
    pub fn select_buft(&self, il: usize) -> Result<BufferType> {
        let dev = self
            .impl_state
            .dev_layer
            .get(il)
            .ok_or_else(|| Error::new(format!("layer device out of range: {}", il)))?;
        super::buffers::select_buft(&dev.bufts, self.hparams.n_embd)
    }

    /// `llama_model::memory_breakdown`.
    pub fn memory_breakdown(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for (_, bufs) in &self.impl_state.ctxs_bufs {
            for buf in bufs {
                *out.entry(buf.buft.name.clone()).or_insert(0) += buf.size;
            }
        }
        out
    }

    /// `llama_model::print_info`, rendered as a string for the pure-Rust port.
    pub fn print_info(&self) -> String {
        hp::print_info(self)
    }

    /// Load the layer list for arch-specific tensor creation.
    pub fn resize_layers(&mut self) {
        self.layers = vec![Layer::default(); self.hparams.n_layer_all as usize];
    }

    /// Track an added LoRA adapter (the C++ set owns the pointers).
    pub fn add_lora(&mut self, name: impl Into<String>) {
        self.loras.insert(name.into());
    }

    pub fn remove_lora(&mut self, name: &str) {
        self.loras.remove(name);
    }

    /// The tensors for a layer, for arch modules building their own models.
    pub fn layer(&self, il: usize) -> Result<&Layer> {
        self.layers
            .get(il)
            .ok_or_else(|| Error::new(format!("layer out of range: {}", il)))
    }

    pub fn layer_mut(&mut self, il: usize) -> Result<&mut Layer> {
        self.layers
            .get_mut(il)
            .ok_or_else(|| Error::new(format!("layer out of range: {}", il)))
    }

    /// Reset RoPE rotation state (`llama_model_set_attn_rot`).
    pub fn set_attn_rot(&mut self, value: bool) {
        self.attn_rot = value;
    }

    /// Token embedding table rows, when the tensor is available.
    pub fn n_embd_table_rows(&self) -> usize {
        self.tok_embd
            .as_ref()
            .map(|t| t.ne[1].max(0) as usize)
            .unwrap_or(0)
    }

    /// Decoder start token; mirrors `llama_model_decoder_start_token`.
    pub fn decoder_start_token(&self) -> Token {
        self.hparams.dec_start_token_id
    }

    /// `llama_model_chat_template` for the default template name.
    pub fn chat_template(&self, name: Option<&str>) -> Option<&str> {
        let key = if let Some(name) = name {
            format!("{}.tokenizer.chat_template", name)
        } else {
            format!("{}.tokenizer.chat_template", self.arch.name())
        };
        self.gguf_kv.get(&key).map(|s| s.as_str()).or_else(|| {
            if name.is_none()
                && self.vocab.get_pre_type() == crate::vocab::VocabPreType::Tekken
                && self.layers.len() == 40
            {
                Some("mistral-v7-tekken")
            } else {
                None
            }
        })
    }

    /// `llama_internal_get_tensor_map`.
    pub fn internal_tensor_map(&self) -> &[(String, Tensor)] {
        &self.tensors_by_name
    }

    /// Delegate used by arch-specific modules that need custom hparams.
    pub fn load_arch_hparams_hook(&mut self, ml: &LlamaModelLoader) -> Result<()> {
        hp::load_arch_hparams(self, ml)
    }

    /// Delegate used by arch-specific modules that need custom tensors.
    pub fn load_arch_tensors_hook(&mut self, ml: &mut LlamaModelLoader) -> Result<()> {
        tn::load_arch_tensors(self, ml)
    }
}

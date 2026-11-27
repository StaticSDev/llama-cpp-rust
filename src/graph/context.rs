//! `llm_graph_context`: the state shared by every `build_*` method.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::batch::Ubatch;
use crate::cparams::{CParams, ContextType, PoolingType};
use crate::core::{Error, Result, SeqId};
use crate::hparams::{Hparams, RopeType};

use super::ggml::{Graph, Tensor};
use super::support::{
    Backend, Cvec, LoraAdapter, MemoryContext, MoeCache, Model, PrecPolicy, Sampler, Scheduler,
};
use super::types::{GraphParams, GraphResult};

/// `llm_graph_context`.
pub struct GraphContext {
    pub model: Rc<Model>,
    pub hparams: Hparams,
    pub cparams: CParams,
    pub ubatch: Ubatch,
    pub prec_policy: PrecPolicy,
    pub n_embd: i64,
    pub n_layer: i64,
    pub n_layer_nextn: i64,
    pub n_rot: i64,
    pub n_ctx: i64,
    pub n_head: i64,
    pub n_head_kv: i64,
    pub n_embd_head_k: i64,
    pub n_embd_k_gqa: i64,
    pub n_embd_head_v: i64,
    pub n_embd_v_gqa: i64,
    pub n_expert: i64,
    pub n_expert_used: i64,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub norm_eps: f32,
    pub norm_rms_eps: f32,
    pub n_tokens: i64,
    pub n_outputs: i64,
    pub n_ctx_orig: i32,
    pub pooling_type: PoolingType,
    pub rope_type: RopeType,
    pub sched: Rc<RefCell<Scheduler>>,
    pub backend_cpu: Backend,
    pub cvec: Option<Rc<Cvec>>,
    pub loras: Option<Rc<Vec<(Rc<LoraAdapter>, f32)>>>,
    pub mctx: Option<Rc<dyn MemoryContext>>,
    pub cross: Option<Rc<super::types::Cross>>,
    pub moe_cache: Option<Rc<dyn MoeCache>>,
    pub layer_inp_dev: Option<Tensor>,
    pub layer_inp_src: Option<Tensor>,
    pub h_nextn_dev: Option<Tensor>,
    pub h_nextn_src: Option<Tensor>,
    pub samplers: BTreeMap<SeqId, Rc<Sampler>>,
    pub cb_func: Option<Rc<dyn Fn(&Ubatch, &Tensor, &str, i32)>>,
    pub res: Rc<RefCell<GraphResult>>,
    pub gf: Rc<RefCell<Graph>>,
}

impl GraphContext {
    pub fn new(params: &GraphParams) -> Result<Self> {
        let model = params
            .model
            .clone()
            .ok_or_else(|| Error::new("llm_graph_context: missing model"))?;
        let res = params
            .res
            .clone()
            .ok_or_else(|| Error::new("llm_graph_context: missing graph result"))?;

        let hparams = model.hparams.clone();
        let cparams = params.cparams.clone();
        let ubatch = params.ubatch.clone();

        let n_embd = hparams.n_embd as i64;
        let n_layer = hparams.n_layer() as i64;
        let n_layer_nextn = hparams.n_layer_nextn as i64;
        let n_rot = hparams.n_rot(0).unwrap_or(0) as i64;
        let n_head = hparams.n_head(0).unwrap_or(0) as i64;
        let n_head_kv = hparams.n_head_kv(0).unwrap_or(0) as i64;
        let n_embd_head_k = hparams.n_embd_head_k(0).unwrap_or(0) as i64;
        let n_embd_k_gqa = hparams.n_embd_k_gqa(0).unwrap_or(0) as i64;
        let n_embd_head_v = hparams.n_embd_head_v(0).unwrap_or(0) as i64;
        let n_embd_v_gqa = hparams.n_embd_v_gqa(0).unwrap_or(0) as i64;
        let n_expert = hparams.n_expert as i64;
        let n_expert_used = hparams.n_expert_used_max().unwrap_or(0) as i64;
        let norm_eps = hparams.f_norm_eps;
        let norm_rms_eps = hparams.f_norm_rms_eps;
        let rope_type = hparams.rope_type;

        let gf = res.borrow().get_gf();
        res.borrow_mut().set_params(params);

        Ok(Self {
            model,
            hparams,
            cparams,
            ubatch,
            prec_policy: params.model.as_ref().map(|m| m.prec_policy).unwrap_or_default(),
            n_embd,
            n_layer,
            n_layer_nextn,
            n_rot,
            n_ctx: params.cparams.n_ctx as i64,
            n_head,
            n_head_kv,
            n_embd_head_k,
            n_embd_k_gqa,
            n_embd_head_v,
            n_embd_v_gqa,
            n_expert,
            n_expert_used,
            freq_base: params.cparams.rope_freq_base,
            freq_scale: params.cparams.rope_freq_scale,
            ext_factor: params.cparams.yarn_ext_factor,
            attn_factor: params.cparams.yarn_attn_factor,
            beta_fast: params.cparams.yarn_beta_fast,
            beta_slow: params.cparams.yarn_beta_slow,
            norm_eps,
            norm_rms_eps,
            n_tokens: params.ubatch.n_tokens as i64,
            n_outputs: params.n_outputs as i64,
            n_ctx_orig: params.cparams.n_ctx_orig_yarn as i32,
            pooling_type: params.cparams.pooling_type,
            rope_type,
            sched: Rc::new(RefCell::new(Scheduler::default())),
            backend_cpu: Backend::Cpu,
            cvec: params.cvec.clone(),
            loras: params.loras.clone(),
            mctx: params.mctx.clone(),
            cross: params.cross.clone(),
            moe_cache: params.moe_cache.clone(),
            layer_inp_dev: params.layer_inp_dev.clone(),
            layer_inp_src: params.layer_inp_src.clone(),
            h_nextn_dev: params.h_nextn_dev.clone(),
            h_nextn_src: params.h_nextn_src.clone(),
            samplers: params.samplers.clone(),
            cb_func: params.cb.clone(),
            res,
            gf,
        })
    }

    /// `llm_graph_context::cb`.
    pub fn cb(&self, cur: &Tensor, name: &str, il: i32) {
        if let Some(cb) = &self.cb_func {
            cb(&self.ubatch, cur, name, il);
        }
    }

    /// True when the last layer must be narrowed before capturing `h_nextn`.
    pub fn crop_before_nextn(&self, inp_out_ids: Option<&Tensor>) -> bool {
        inp_out_ids.is_some()
            && (!self.cparams.embeddings_nextn || self.cparams.embeddings_nextn_masked)
    }

    /// True when `h_nextn` must be narrowed after capture.
    pub fn crop_after_nextn(&self, inp_out_ids: Option<&Tensor>) -> bool {
        inp_out_ids.is_some()
            && self.cparams.embeddings_nextn
            && !self.cparams.embeddings_nextn_masked
    }

    /// `llm_graph_context::build_cvec`.
    pub fn build_cvec(&self, cur: &Tensor, il: i32) -> Tensor {
        match &self.cvec {
            Some(cvec) => cvec.apply_to(cur, il),
            None => cur.clone(),
        }
    }

    /// True when a per-layer MTP state input is present.
    pub fn is_mtp(&self) -> bool {
        self.cparams.ctx_type == ContextType::Mtp
    }

    /// Expand a node and record it in the compute graph.
    pub fn expand(&self, t: &Tensor) {
        self.gf.borrow_mut().build_forward_expand(t);
    }
}

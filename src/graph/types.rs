//! Types declared in `llama-graph.h`: enums, graph parameters, results and
//! the small records shared by the builder methods.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::batch::{Pos, Ubatch};
use crate::cparams::CParams;
use crate::core::SeqId;
use crate::hparams::LLAMA_MAX_LAYERS;

use super::ggml::{Graph, Tensor};
use super::support::{Cvec, MemoryContext, MoeCache, Model, Sampler, LoraAdapter};

/// `llm_graph_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphType {
    Default,
    Encoder,
    Decoder,
    DecoderMtp,
}

/// `llm_fused_op`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedOp {
    FlashAttn,
    FlashAttnKvRows,
    Dsv4HcPre,
    Dsv4HcComb,
    Dsv4HcPost,
    TopKMask,
}

/// `llm_ffn_op_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnOpType {
    /// Sentinel: architectures must assign before use.
    None,
    Silu,
    Gelu,
    GeluErf,
    Relu,
    ReluSqr,
    Swiglu,
    Geglu,
    GegluErf,
    Reglu,
    SwigluOaiMoe,
    Situ,
}

/// `llm_ffn_gate_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnGateType {
    Seq,
    /// `ffn_gate` is parallel to `ffn_up`.
    Par,
}

/// `llm_norm_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormType {
    Norm,
    NormRms,
    NormGroup,
}

/// `llama_cross`: encoder output handed to a decoder graph.
#[derive(Clone, Debug, Default)]
pub struct Cross {
    pub n_embd: i64,
    pub n_enc: i64,
    /// Embeddings copied to host memory (temporary in llama.cpp).
    pub v_embd: Vec<f32>,
    /// Sequence ids visible to each encoded token.
    pub seq_ids_enc: Vec<BTreeSet<SeqId>>,
}

/// `llm_attn_mask`: KQ mask plus optional per-sequence cell rows.
#[derive(Clone, Debug, Default)]
pub struct AttnMask {
    pub raw: Option<Tensor>,
    /// I32 [n_kv_rows, n_slices].
    pub rows: Option<Tensor>,
    pub n_kv: u32,
    /// Fill every token of a sequence from its first mask row.
    pub block_uniform: bool,
}

impl AttnMask {
    pub fn has_rows(&self) -> bool {
        self.rows.is_some()
    }
}

/// `llm_graph_params`.
pub struct GraphParams {
    pub model: Option<Rc<Model>>,
    pub cparams: CParams,
    /// Intentionally a copy, as in C++.
    pub ubatch: Ubatch,
    pub gtype: GraphType,
    pub cvec: Option<Rc<Cvec>>,
    pub loras: Option<Rc<Vec<(Rc<LoraAdapter>, f32)>>>,
    pub mctx: Option<Rc<dyn MemoryContext>>,
    pub cross: Option<Rc<Cross>>,
    pub moe_cache: Option<Rc<dyn MoeCache>>,
    pub layer_inp_dev: Option<Tensor>,
    pub layer_inp_src: Option<Tensor>,
    pub h_nextn_dev: Option<Tensor>,
    pub h_nextn_src: Option<Tensor>,
    pub samplers: BTreeMap<SeqId, Rc<Sampler>>,
    pub n_outputs: u32,
    pub cb: Option<Rc<dyn Fn(&Ubatch, &Tensor, &str, i32)>>,
    pub res: Option<Rc<RefCell<GraphResult>>>,
}

impl GraphParams {
    pub fn samplers_equal(
        lhs: &BTreeMap<SeqId, Rc<Sampler>>,
        rhs: &BTreeMap<SeqId, Rc<Sampler>>,
    ) -> bool {
        if lhs.len() != rhs.len() {
            return false;
        }
        lhs.iter().all(|(seq_id, sampler)| {
            matches!(rhs.get(seq_id), Some(other) if Rc::ptr_eq(sampler, other))
        })
    }

    /// True when `other` would produce a graph with the same topology.
    pub fn allow_reuse(&self, other: &GraphParams) -> bool {
        let a = &self.ubatch;
        let b = &other.ubatch;

        let mut can_reuse_ubatch = a.equal_seqs() == b.equal_seqs()
            && a.n_tokens == b.n_tokens
            && a.n_seq_tokens == b.n_seq_tokens
            && a.n_seqs == b.n_seqs
            && a.n_seqs_unq == b.n_seqs_unq
            && a.is_mixed() == b.is_mixed()
            && ((a.token.is_empty() && b.token.is_empty())
                || (a.embd.is_empty() && b.embd.is_empty())
                || (!a.token.is_empty()
                    && !b.token.is_empty()
                    && !a.embd.is_empty()
                    && !b.embd.is_empty()))
            && a.embd_state.is_empty() == b.embd_state.is_empty()
            && a.embd_row.is_empty() == b.embd_row.is_empty();

        if can_reuse_ubatch && a.equal_seqs() {
            if a.seq_id_unq.is_empty() {
                // The old ubatch did not own its data; sequence ids cannot be
                // verified, so graph reuse is not safe.
                can_reuse_ubatch = false;
            } else {
                for s in 0..a.n_seqs_unq as usize {
                    if a.seq_id_unq.get(s) != b.seq_id_unq.get(s) {
                        can_reuse_ubatch = false;
                    }
                }
            }
        }

        if !can_reuse_ubatch {
            return false;
        }
        if self.n_outputs != other.n_outputs {
            return false;
        }
        if !Self::samplers_equal(&self.samplers, &other.samplers) {
            return false;
        }
        if !self.samplers.is_empty() {
            if a.seq_id.is_empty() || b.seq_id.is_empty() {
                return false;
            }
            for i in 0..a.n_tokens as usize {
                if a.output.get(i) != b.output.get(i)
                    || a.seq_id.get(i).and_then(|v| v.first())
                        != b.seq_id.get(i).and_then(|v| v.first())
                {
                    return false;
                }
            }
        }
        if self.cparams.nextn_layer_offset != other.cparams.nextn_layer_offset {
            return false;
        }

        self.cparams.embeddings == other.cparams.embeddings
            && self.cparams.embeddings_nextn == other.cparams.embeddings_nextn
            && self.cparams.embeddings_nextn_masked == other.cparams.embeddings_nextn_masked
            && self.cparams.causal_attn == other.cparams.causal_attn
            && same_opt_tensor(&self.layer_inp_dev, &other.layer_inp_dev)
            && same_opt_tensor(&self.layer_inp_src, &other.layer_inp_src)
            && same_opt_tensor(&self.h_nextn_dev, &other.h_nextn_dev)
            && same_opt_tensor(&self.h_nextn_src, &other.h_nextn_src)
            && same_opt_rc_model(&self.model, &other.model)
            && self.gtype == other.gtype
            && same_opt_rc(&self.cvec, &other.cvec)
            && same_opt_rc(&self.loras, &other.loras)
            && same_opt_rc(&self.cross, &other.cross)
    }
}

fn same_opt_tensor(a: &Option<Tensor>, b: &Option<Tensor>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

fn same_opt_rc<T>(a: &Option<Rc<T>>, b: &Option<Rc<T>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Rc::ptr_eq(a, b),
        _ => false,
    }
}

fn same_opt_rc_model(a: &Option<Rc<Model>>, b: &Option<Rc<Model>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Rc::ptr_eq(a, b),
        _ => false,
    }
}

/// `llm_graph_fused_node`.
#[derive(Clone, Debug)]
pub struct GraphFusedNode {
    pub op: FusedOp,
    pub tensor: Tensor,
    pub il: i32,
}

/// `llm_graph_input_i`: the interface implemented by every graph input.
pub trait GraphInput {
    fn set_input(&mut self, ubatch: &Ubatch);
    /// Return true when tensors built from `params` match the stored ones.
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let _ = params;
        false
    }
    fn name(&self) -> &'static str {
        "graph_input"
    }
}

/// `llm_graph_result`: outputs and inputs handed back to the context.
pub struct GraphResult {
    pub t_inp_tokens: Option<Tensor>,
    pub t_inp_embd: Option<Tensor>,
    pub t_logits: Option<Tensor>,
    pub t_embd: Option<Tensor>,
    pub t_embd_pooled: Option<Tensor>,
    /// Hidden state before the final output norm, [n_embd, n_outputs].
    pub t_h_nextn: Option<Tensor>,
    pub t_layer_inp: Vec<Option<Tensor>>,
    pub t_sampled: Vec<Option<Tensor>>,
    pub t_sampled_probs: Vec<Option<Tensor>>,
    pub t_sampled_logits: Vec<Option<Tensor>>,
    pub t_candidates: Vec<Option<Tensor>>,
    pub t_sampling_all: Vec<Tensor>,
    /// [0] is the random-routing input, the rest are views of it.
    pub t_moe_ids: Vec<Tensor>,
    pub inputs: Vec<Rc<RefCell<dyn GraphInput>>>,
    pub fused_nodes: Vec<GraphFusedNode>,
    /// Keyed by (input id, rotation id).
    pub hdmd_inputs: BTreeMap<(usize, usize), Tensor>,
    pub gf: Rc<RefCell<Graph>>,
    pub max_nodes: i64,
    pub params: Option<GraphParams>,
    pub debug: i32,
}

impl GraphResult {
    pub fn new(max_nodes: i64) -> Self {
        Self {
            t_inp_tokens: None,
            t_inp_embd: None,
            t_logits: None,
            t_embd: None,
            t_embd_pooled: None,
            t_h_nextn: None,
            t_layer_inp: vec![None; LLAMA_MAX_LAYERS + 1],
            t_sampled: Vec::new(),
            t_sampled_probs: Vec::new(),
            t_sampled_logits: Vec::new(),
            t_candidates: Vec::new(),
            t_sampling_all: Vec::new(),
            t_moe_ids: Vec::new(),
            inputs: Vec::new(),
            fused_nodes: Vec::new(),
            hdmd_inputs: BTreeMap::new(),
            gf: Rc::new(RefCell::new(Graph::new())),
            max_nodes,
            params: None,
            debug: 0,
        }
    }

    pub fn get_max_nodes(&self) -> i64 {
        self.max_nodes
    }

    pub fn set_max_nodes(&mut self, max_nodes: i64) {
        self.max_nodes = max_nodes;
    }

    /// Reset outputs, inputs and the compute graph; keep the configured size.
    pub fn reset(&mut self) {
        self.t_inp_tokens = None;
        self.t_inp_embd = None;
        self.t_logits = None;
        self.t_embd = None;
        self.t_embd_pooled = None;
        self.t_h_nextn = None;
        self.t_layer_inp = vec![None; LLAMA_MAX_LAYERS + 1];
        self.t_sampled.clear();
        self.t_sampled_probs.clear();
        self.t_sampled_logits.clear();
        self.t_candidates.clear();
        self.t_sampling_all.clear();
        self.t_moe_ids.clear();
        self.params = None;
        self.inputs.clear();
        self.fused_nodes.clear();
        self.hdmd_inputs.clear();
        self.gf = Rc::new(RefCell::new(Graph::new()));
    }

    pub fn get_inp_tokens(&self) -> Option<&Tensor> {
        self.t_inp_tokens.as_ref()
    }
    pub fn get_logits(&self) -> Option<&Tensor> {
        self.t_logits.as_ref()
    }
    pub fn get_embd(&self) -> Option<&Tensor> {
        self.t_embd.as_ref()
    }
    pub fn get_embd_pooled(&self) -> Option<&Tensor> {
        self.t_embd_pooled.as_ref()
    }
    pub fn get_h_nextn(&self) -> Option<&Tensor> {
        self.t_h_nextn.as_ref()
    }
    pub fn get_layer_inp(&self, il: usize) -> Option<&Tensor> {
        self.t_layer_inp.get(il).and_then(|t| t.as_ref())
    }
    pub fn get_gf(&self) -> Rc<RefCell<Graph>> {
        self.gf.clone()
    }

    pub fn set_inputs(&mut self, ubatch: &Ubatch) {
        for input in &mut self.inputs {
            input.borrow_mut().set_input(ubatch);
        }
    }

    pub fn set_outputs(&mut self, params: &GraphParams) {
        if let Some(t) = &self.t_logits {
            t.set_output();
        }
        if let Some(t) = &self.t_embd {
            t.set_output();
        }
        if let Some(t) = &self.t_embd_pooled {
            t.set_output();
        }
        if let Some(t) = &self.t_h_nextn {
            t.set_output();
        }
        {
            let enabled = &params.cparams.embeddings_layer_inp;
            if !params.cparams.embeddings_layer_inp_dev {
                for (il, flag) in enabled.iter().enumerate() {
                    if *flag {
                        if let Some(t) = self.t_layer_inp.get(il).and_then(|t| t.as_ref()) {
                            t.set_output();
                        }
                    }
                }
            }
        }
        for t in &self.t_sampling_all {
            t.set_output();
        }
    }

    /// Try to reuse the graph; inputs may rebind their memory context.
    pub fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let Some(prev) = &self.params else {
            return false;
        };
        if !prev.allow_reuse(params) {
            return false;
        }
        let mut res = true;
        for input in &mut self.inputs {
            res &= input.borrow_mut().can_reuse(params);
        }
        res
    }

    /// Keep a strong handle on the concrete input and register it for
    /// `set_inputs`/`can_reuse`. Mirrors `add_input` returning a raw pointer.
    pub fn add_input<I: GraphInput + 'static>(&mut self, input: I) -> Rc<RefCell<I>> {
        let rc = Rc::new(RefCell::new(input));
        self.inputs.push(rc.clone());
        rc
    }

    pub fn add_fused_node(&mut self, node: GraphFusedNode) {
        self.fused_nodes.push(node);
    }

    pub fn get_hdmd_input(&self, cur: &Tensor, rot: &Tensor) -> Option<Tensor> {
        self.hdmd_inputs.get(&(cur.id(), rot.id())).cloned()
    }

    pub fn set_hdmd_input(&mut self, cur: &Tensor, rot: &Tensor, res: Tensor) {
        self.hdmd_inputs.insert((cur.id(), rot.id()), res);
    }

    pub fn set_params(&mut self, params: &GraphParams) {
        self.params = Some(GraphParams {
            model: params.model.clone(),
            cparams: params.cparams.clone(),
            ubatch: params.ubatch.clone(),
            gtype: params.gtype,
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
            n_outputs: params.n_outputs,
            cb: params.cb.clone(),
            // Keep the copy acyclic: the original parameters already own the
            // result handle, and only `allow_reuse` fields matter here.
            res: None,
        });
    }
}

/// `llm_graph_qkv`.
#[derive(Clone, Debug)]
pub struct GraphQkv {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

/// `llm_graph_amax_src1`.
#[derive(Clone, Debug, Default)]
pub struct GraphAmaxSrc1 {
    pub res: Option<Tensor>,
    pub scale_s: Option<Tensor>,
}

/// `llama_relative_position_bucket` inputs use `Pos` from the batch module.
pub type Position = Pos;

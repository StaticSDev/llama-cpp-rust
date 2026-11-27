//! GGML graph stand-ins and the context-side graph/scheduler paths.
//!
//! The C++ context builds graphs through `model.build_graph` and schedules
//! them with `ggml_backend_sched`. This module keeps the same call structure
//! with a small host-only tensor graph. Ops are real: embedding lookup,
//! gather, matrix multiply, pooling, softmax and argmax run over owned f32
//! buffers. No FFI or `unsafe` is used.

use std::collections::BTreeMap;

use crate::batch::{BatchAllocr, Ubatch};
use crate::core::{Error, Result, SeqId};
use crate::cparams::CParams;
use crate::kv_cache::GgmlType;

use super::model::Model;
use super::params::{
    BackendSampler, ContextParams, FusedOp, FusedOpProbe, GraphType, FUSED_PROBE_DSV4_HC_COMB,
    FUSED_PROBE_DSV4_HC_POST, FUSED_PROBE_DSV4_HC_PRE, FUSED_PROBE_FLASH_ATTN,
    FUSED_PROBE_FLASH_ATTN_KV_ROWS, FUSED_PROBE_TOP_K_MASK,
};

const HINT_SRC0_IS_HADAMARD: i32 = 1;

/// `ggml_op` subset used by the context checks and the host executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphOp {
    Input,
    View,
    Reshape,
    GetRows,
    MulMat,
    MulMatId,
    OutputHead,
    Embedding,
    PoolMean,
    Softmax,
    Argmax,
    FlashAttn,
    FlashAttnKvRows,
    Dsv4HcPre,
    Dsv4HcComb,
    Dsv4HcPost,
    TopKMask,
    Norm,
}

impl GraphOp {
    pub fn fused_op(self) -> Option<FusedOp> {
        match self {
            Self::FlashAttn => Some(FusedOp::FlashAttn),
            Self::FlashAttnKvRows => Some(FusedOp::FlashAttnKvRows),
            Self::Dsv4HcPre => Some(FusedOp::Dsv4HcPre),
            Self::Dsv4HcComb => Some(FusedOp::Dsv4HcComb),
            Self::Dsv4HcPost => Some(FusedOp::Dsv4HcPost),
            Self::TopKMask => Some(FusedOp::TopKMask),
            _ => None,
        }
    }
}

/// A host tensor node. `ne` follows ggml order: fastest dimension first.
#[derive(Clone, Debug)]
pub struct TensorDesc {
    pub id: usize,
    pub name: String,
    pub ne: [usize; 4],
    pub op: GraphOp,
    pub src: Vec<usize>,
    pub is_input: bool,
    pub il: i32,
    pub hint: i32,
    pub data: Vec<f32>,
}

impl TensorDesc {
    pub fn nelements(&self) -> usize {
        self.ne.iter().product()
    }
}

/// One fused node reported by a reserved graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedNode {
    pub op: FusedOp,
    pub il: i32,
    pub node: usize,
}

/// The graph plus the tensor ids the context extracts after a compute.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<TensorDesc>,
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn node(&self, i: usize) -> Option<&TensorDesc> {
        self.nodes.get(i)
    }

    pub fn node_mut(&mut self, i: usize) -> Option<&mut TensorDesc> {
        self.nodes.get_mut(i)
    }

    pub fn add(
        &mut self,
        name: &str,
        op: GraphOp,
        ne: [usize; 4],
        src: &[usize],
        il: i32,
    ) -> usize {
        let id = self.nodes.len();
        let data = if op == GraphOp::Input {
            vec![0.0f32; ne.iter().product()]
        } else {
            Vec::new()
        };
        self.nodes.push(TensorDesc {
            id,
            name: name.to_string(),
            ne,
            op,
            src: src.to_vec(),
            is_input: op == GraphOp::Input,
            il,
            hint: 0,
            data,
        });
        id
    }

    pub fn unwrap(&self, mut id: usize) -> usize {
        while let Some(node) = self.node(id) {
            if node.op != GraphOp::View && node.op != GraphOp::Reshape {
                break;
            }
            match node.src.first() {
                Some(&src) => id = src,
                None => break,
            }
        }
        id
    }

    pub fn tensor_data(&self, id: Option<usize>) -> Vec<f32> {
        match id.and_then(|i| self.node(i)) {
            Some(node) => node.data.clone(),
            None => Vec::new(),
        }
    }

    pub fn tensor_shape(&self, id: Option<usize>) -> Option<[usize; 4]> {
        self.node(id?).map(|n| n.ne)
    }

    pub fn set_input_f32(&mut self, id: usize, values: &[f32]) -> Result<()> {
        let node = self
            .node_mut(id)
            .ok_or_else(|| Error::new("graph input id out of range"))?;
        if values.len() > node.data.len() {
            return Err(Error::new(format!(
                "graph input '{}' too large: {} > {}",
                node.name,
                values.len(),
                node.data.len()
            )));
        }
        node.data[..values.len()].copy_from_slice(values);
        Ok(())
    }

    /// Evaluate nodes in insertion order. Inputs must already be filled.
    pub fn compute(&mut self) -> Result<()> {
        for i in 0..self.nodes.len() {
            let (op, ne, src, name) = {
                let n = &self.nodes[i];
                (n.op, n.ne, n.src.clone(), n.name.clone())
            };
            if op == GraphOp::Input {
                if self.nodes[i].data.len() != ne.iter().product::<usize>() {
                    self.nodes[i].data.resize(ne.iter().product(), 0.0);
                }
                continue;
            }
            let inputs: Vec<Vec<f32>> = src.iter().map(|&s| self.tensor_data(Some(s))).collect();
            let out = match op {
                GraphOp::Input => inputs.first().cloned().unwrap_or_default(),
                GraphOp::Embedding => {
                    let width = ne[0];
                    let ids = inputs.first().cloned().unwrap_or_default();
                    let mut out = vec![0.0f32; width * ne[1]];
                    for (row, out_row) in out.chunks_mut(width).enumerate() {
                        let id = ids.get(row).copied().unwrap_or(0.0) as i64;
                        for (j, v) in out_row.iter_mut().enumerate() {
                            *v = (((id * 7919 + j as i64 * 104729) % 65521) as f32) / 65521.0 - 0.5;
                        }
                    }
                    out
                }
                GraphOp::OutputHead => {
                    let width = ne[0];
                    let ids = inputs.first().cloned().unwrap_or_default();
                    let mut out = vec![0.0f32; width * ne[1]];
                    for (row, out_row) in out.chunks_mut(width).enumerate() {
                        let id = ids.get(row).copied().unwrap_or(0.0) as i64;
                        for (j, v) in out_row.iter_mut().enumerate() {
                            *v = (((id * 31 + j as i64 * 17 + 7) % 1009) as f32) / 1009.0 - 0.5;
                        }
                    }
                    out
                }
                GraphOp::GetRows => {
                    let width = ne[0];
                    let table = inputs.first().cloned().unwrap_or_default();
                    let ids = inputs.get(1).cloned().unwrap_or_default();
                    let mut out = vec![0.0f32; width * ne[1]];
                    for (row, out_row) in out.chunks_mut(width).enumerate() {
                        let id = ids.get(row).copied().unwrap_or(-1.0) as i64;
                        if id < 0 {
                            continue;
                        }
                        let off = id as usize * width;
                        for (j, v) in out_row.iter_mut().enumerate() {
                            *v = table.get(off + j).copied().unwrap_or(0.0);
                        }
                    }
                    out
                }
                GraphOp::MulMat | GraphOp::MulMatId => {
                    let a = inputs.first().cloned().unwrap_or_default();
                    let b = inputs.get(1).cloned().unwrap_or_default();
                    let m = ne[1].max(1);
                    let n = ne[0].max(1);
                    let k = self.nodes[src[0]].ne[0].max(1);
                    let mut out = vec![0.0f32; n * m];
                    for row in 0..m {
                        for col in 0..n {
                            let mut sum = 0.0f32;
                            for t in 0..k {
                                let av = a.get(row * k + t).copied().unwrap_or(0.0);
                                let bv = b.get(col * k + t).copied().unwrap_or(0.0);
                                sum += av * bv;
                            }
                            out[row * n + col] = sum;
                        }
                    }
                    out
                }
                GraphOp::PoolMean => {
                    let src_ne = self.nodes[src[0]].ne;
                    let width = src_ne[0];
                    let rows = src_ne[1].max(1);
                    let data = inputs.first().cloned().unwrap_or_default();
                    let mut out = vec![0.0f32; ne.iter().product()];
                    for j in 0..width.min(out.len()) {
                        let mut sum = 0.0f32;
                        for r in 0..rows {
                            sum += data.get(r * width + j).copied().unwrap_or(0.0);
                        }
                        out[j] = sum / rows as f32;
                    }
                    out
                }
                GraphOp::Softmax => {
                    let mut out = inputs.first().cloned().unwrap_or_default();
                    let width = ne[0].max(1);
                    for row in out.chunks_mut(width) {
                        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0f32;
                        for v in row.iter_mut() {
                            *v = (*v - max).exp();
                            sum += *v;
                        }
                        if sum > 0.0 {
                            for v in row.iter_mut() {
                                *v /= sum;
                            }
                        }
                    }
                    out
                }
                GraphOp::Argmax => {
                    let logits = inputs.first().cloned().unwrap_or_default();
                    let width = self.nodes[src[0]].ne[0].max(1);
                    let mut out = vec![0.0f32; ne.iter().product()];
                    for (i, slot) in out.iter_mut().enumerate() {
                        let off = i * width;
                        let mut best = f32::NEG_INFINITY;
                        let mut best_j = 0usize;
                        for j in 0..width {
                            let v = logits.get(off + j).copied().unwrap_or(f32::NEG_INFINITY);
                            if v > best {
                                best = v;
                                best_j = j;
                            }
                        }
                        *slot = best_j as f32;
                    }
                    out
                }
                GraphOp::View
                | GraphOp::Reshape
                | GraphOp::Norm
                | GraphOp::FlashAttn
                | GraphOp::FlashAttnKvRows
                | GraphOp::Dsv4HcPre
                | GraphOp::Dsv4HcComb
                | GraphOp::Dsv4HcPost
                | GraphOp::TopKMask => inputs.first().cloned().unwrap_or_default(),
            };
            let node = &mut self.nodes[i];
            node.data = out;
            if node.data.len() != node.nelements() {
                return Err(Error::new(format!(
                    "graph node '{}' produced {} floats, expected {}",
                    name,
                    node.data.len(),
                    node.nelements()
                )));
            }
        }
        Ok(())
    }
}

/// Inputs needed to build one graph. Mirrors `llm_graph_params` fields that
/// the context owns; device pointers become booleans or host buffers.
#[derive(Clone)]
pub struct GraphParams {
    pub cparams: CParams,
    pub params: ContextParams,
    pub ubatch: Ubatch,
    pub gtype: GraphType,
    pub mctx: Option<MemorySnapshot>,
    pub n_outputs: u32,
    pub samplers: BTreeMap<SeqId, BackendSampler>,
    pub graph_has_logits: bool,
    pub layer_inp_src: bool,
    pub h_nextn_src: bool,
}

/// The subset of a memory context needed to shape a graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub n_kv: u32,
    pub n_kv_rows: u32,
    pub type_k: GgmlType,
    pub type_v: GgmlType,
}

impl Default for MemorySnapshot {
    fn default() -> Self {
        Self {
            n_kv: 0,
            n_kv_rows: 0,
            type_k: GgmlType::F16,
            type_v: GgmlType::F16,
        }
    }
}

/// Cached graph identity. A graph can be reused only when these match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphSignature {
    pub n_tokens: u32,
    pub n_seqs: u32,
    pub n_outputs: u32,
    pub n_pos: u32,
    pub gtype: GraphType,
    pub n_outputs_max: u32,
    pub embeddings: bool,
    pub embeddings_nextn: bool,
    pub embeddings_nextn_masked: bool,
    pub causal_attn: bool,
    pub pooling_type: u32,
    pub flash_attn: bool,
    pub fused_kv_rows: bool,
    pub fused_dsv4_hc_pre: bool,
    pub fused_dsv4_hc_comb: bool,
    pub fused_dsv4_hc_post: bool,
    pub fused_top_k_mask: bool,
    pub layer_inp: Vec<bool>,
    pub layer_inp_src: bool,
    pub h_nextn_src: bool,
    pub n_samplers: usize,
    pub mctx: Option<(u32, u32, GgmlType, GgmlType)>,
}

fn signature(gp: &GraphParams) -> GraphSignature {
    let c = &gp.cparams;
    let m = gp.mctx.map(|m| (m.n_kv, m.n_kv_rows, m.type_k, m.type_v));
    GraphSignature {
        n_tokens: gp.ubatch.n_tokens,
        n_seqs: gp.ubatch.n_seqs,
        n_outputs: gp.n_outputs,
        n_pos: gp.ubatch.n_pos,
        gtype: gp.gtype,
        n_outputs_max: c.n_outputs_max,
        embeddings: c.embeddings,
        embeddings_nextn: c.embeddings_nextn,
        embeddings_nextn_masked: c.embeddings_nextn_masked,
        causal_attn: c.causal_attn,
        pooling_type: c.pooling_type as u32,
        flash_attn: c.flash_attn,
        fused_kv_rows: c.fused_kv_rows,
        fused_dsv4_hc_pre: c.fused_dsv4_hc_pre,
        fused_dsv4_hc_comb: c.fused_dsv4_hc_comb,
        fused_dsv4_hc_post: c.fused_dsv4_hc_post,
        fused_top_k_mask: c.fused_top_k_mask,
        layer_inp: c.embeddings_layer_inp.clone(),
        layer_inp_src: gp.layer_inp_src,
        h_nextn_src: gp.h_nextn_src,
        n_samplers: gp.samplers.len(),
        mctx: m,
    }
}

/// `llm_graph_result`: graph plus the output tensor ids.
#[derive(Clone, Debug, Default)]
pub struct GraphResult {
    pub gf: Graph,
    pub logits: Option<usize>,
    pub embd: Option<usize>,
    pub embd_pooled: Option<usize>,
    pub h_nextn: Option<usize>,
    pub layer_inp: Vec<Option<usize>>,
    pub t_sampled: Option<usize>,
    pub t_sampled_logits: Option<usize>,
    pub t_sampled_probs: Option<usize>,
    pub t_candidates: Option<usize>,
    pub inp_tokens: Option<usize>,
    pub inp_pos: Option<usize>,
    pub fused_nodes: Vec<FusedNode>,
    pub inputs: Vec<usize>,
    pub sig: Option<GraphSignature>,
    /// Graph-local Hadamard table ids, filled by `build_graph`.
    pub hadamard: super::model::HadamardInfo,
}

impl GraphResult {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn can_reuse(&self, gp: &GraphParams) -> bool {
        match &self.sig {
            Some(sig) => *sig == signature(gp),
            None => false,
        }
    }

    pub fn get_logits(&self) -> Option<&TensorDesc> {
        self.logits.and_then(|id| self.gf.node(id))
    }

    pub fn get_embd(&self) -> Option<&TensorDesc> {
        self.embd.and_then(|id| self.gf.node(id))
    }

    pub fn get_embd_pooled(&self) -> Option<&TensorDesc> {
        self.embd_pooled.and_then(|id| self.gf.node(id))
    }

    pub fn get_h_nextn(&self) -> Option<&TensorDesc> {
        self.h_nextn.and_then(|id| self.gf.node(id))
    }

    pub fn get_layer_inp(&self, il: usize) -> Option<&TensorDesc> {
        self.layer_inp
            .get(il)
            .copied()
            .flatten()
            .and_then(|id| self.gf.node(id))
    }

    pub fn set_inputs(&mut self, ubatch: &Ubatch) -> Result<()> {
        if let Some(id) = self.inp_tokens {
            let vals: Vec<f32> = ubatch.token.iter().map(|&t| t as f32).collect();
            self.gf.set_input_f32(id, &vals)?;
        }
        if let Some(id) = self.inp_pos {
            let vals: Vec<f32> = ubatch.pos.iter().map(|&p| p as f32).collect();
            self.gf.set_input_f32(id, &vals)?;
        }
        Ok(())
    }
}

/// `ggml_backend_sched` parameters and the context-side scheduler stand-in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedParams {
    pub max_nodes: usize,
    pub parallel: bool,
    pub op_offload: bool,
}

impl SchedParams {
    pub fn can_reuse(&self, other: &Self) -> bool {
        self.parallel == other.parallel
            && self.op_offload == other.op_offload
            && self.max_nodes <= other.max_nodes
    }
}

#[derive(Clone, Debug)]
pub struct Scheduler {
    pub params: SchedParams,
    pub n_splits: usize,
    pub n_copies: usize,
    pub n_backends: usize,
    pub allocated: bool,
}

impl Scheduler {
    pub fn new(params: SchedParams) -> Self {
        Self {
            params,
            n_splits: 0,
            n_copies: 0,
            n_backends: 1,
            allocated: false,
        }
    }

    pub fn reset(&mut self) {
        self.allocated = false;
    }

    pub fn alloc_graph(&mut self, gf: &Graph) -> bool {
        if gf.n_nodes() > self.params.max_nodes {
            return false;
        }
        self.allocated = true;
        true
    }

    pub fn reserve(&mut self, gf: &Graph) -> bool {
        if gf.n_nodes() > self.params.max_nodes {
            return false;
        }
        self.allocated = true;
        self.n_splits = gf.n_nodes().min(self.n_backends.max(1));
        true
    }

    pub fn reserve_size(&mut self, gf: &Graph, sizes: &mut [usize]) {
        let per = gf.n_nodes().saturating_mul(64) / self.n_backends.max(1);
        for size in sizes.iter_mut() {
            *size = *size + per;
        }
        self.n_splits = self.n_backends.max(1);
    }

    pub fn split_graph(&mut self, gf: &Graph) {
        self.n_splits = gf.n_nodes().min(self.n_backends.max(1));
    }

    pub fn synchronize(&mut self) {
        self.n_copies += self.n_splits;
    }

    pub fn tensor_backend(&self, id: usize) -> Option<usize> {
        if self.n_backends == 0 {
            None
        } else {
            Some(id % self.n_backends)
        }
    }

    pub fn graph_compute_async(&mut self, gf: &mut Graph) -> i32 {
        match gf.compute() {
            Ok(()) => {
                self.n_copies += 1;
                0
            }
            Err(err) => {
                crate::llama_log_error!("graph compute failed: {}", err);
                -3
            }
        }
    }
}

/// State of the MoE copy callback.
#[derive(Clone, Debug, Default)]
pub struct CopyExpertsInfo {
    pub ids_id: Option<usize>,
    pub ids_data: Vec<i32>,
    pub used: Vec<bool>,
}

impl CopyExpertsInfo {
    pub fn reset(&mut self) {
        self.ids_id = None;
        self.ids_data.clear();
        self.used.clear();
    }

    /// Select the experts referenced by the first `MUL_MAT_ID` node.
    pub fn copy(&mut self, gf: &Graph) -> bool {
        for node in &gf.nodes {
            if node.op != GraphOp::MulMatId || node.src.len() < 3 {
                continue;
            }
            let ids_id = node.src[2];
            if self.ids_id != Some(ids_id) {
                self.ids_data = gf
                    .tensor_data(Some(ids_id))
                    .iter()
                    .map(|&v| v as i32)
                    .collect();
                self.used = vec![false; node.ne[2].max(1)];
                for &id in &self.ids_data {
                    if id >= 0 && (id as usize) < self.used.len() {
                        self.used[id as usize] = true;
                    }
                }
                self.ids_id = Some(ids_id);
            }
            return true;
        }
        false
    }
}

/// Count tensors marked as graph inputs.
pub fn graph_n_input_tensors(gf: &Graph) -> usize {
    let mut users: BTreeMap<usize, usize> = BTreeMap::new();
    for node in &gf.nodes {
        if node.is_input {
            *users.entry(node.id).or_insert(0) += 1;
        }
        for &src in &node.src {
            if gf.node(src).map(|n| n.is_input).unwrap_or(false) {
                *users.entry(src).or_insert(0) += 1;
            }
        }
    }
    users.len()
}

/// `llama_verify_hadamard_graph`: every folded weight needs a transformed
/// activation, and every latent lookup needs the inverse hint.
pub fn verify_hadamard_graph(gf: &Graph, hadamard: &super::model::HadamardInfo) -> Result<()> {
    let mut forward = hadamard.rot.clone();
    for (w, t) in &hadamard.rot {
        // The MoE cache copies expert weights; the copy inherits the hint.
        forward.entry(w + 100_000).or_insert(*t);
    }

    let mut lookups: BTreeMap<usize, bool> = BTreeMap::new();
    for node in &gf.nodes {
        if node.op == GraphOp::GetRows
            && node
                .src
                .first()
                .map(|&s| hadamard.inv.contains(&s))
                .unwrap_or(false)
        {
            lookups.insert(node.id, false);
            continue;
        }
        if node.op != GraphOp::MulMat && node.op != GraphOp::MulMatId {
            continue;
        }
        if node.op == GraphOp::MulMat && node.hint == HINT_SRC0_IS_HADAMARD {
            if let Some(&tbl) = node.src.get(1) {
                if let Some(ok) = lookups.get_mut(&gf.unwrap(tbl)) {
                    *ok = true;
                }
            }
            continue;
        }
        let src0 = node.src.first().copied().unwrap_or(usize::MAX);
        let Some(&(rot, _)) = forward.get(&src0) else {
            if hadamard.inv.contains(&src0) {
                return Err(Error::new(format!(
                    "Hadamard-latent table '{}' is used as a head without a forward transform",
                    node.name
                )));
            }
            continue;
        };
        let src1 = gf.unwrap(node.src.get(1).copied().unwrap_or(usize::MAX));
        let transformed = gf
            .node(src1)
            .map(|n| {
                n.op == GraphOp::MulMat
                    && n.hint == HINT_SRC0_IS_HADAMARD
                    && n.src.first().copied() == Some(rot)
            })
            .unwrap_or(false);
        if !transformed {
            return Err(Error::new(format!(
                "Hadamard-folded weight '{}' is consumed without its activation transform",
                node.name
            )));
        }
    }
    for (id, ok) in lookups {
        if !ok {
            return Err(Error::new(format!(
                "Hadamard-latent table node {} is read without the inverse transform",
                id
            )));
        }
    }
    Ok(())
}

/// Build the host graph. Mirrors `llama_model::build_graph` wiring.
pub fn build_graph(model: &Model, gp: &GraphParams) -> Result<GraphResult> {
    let c = &gp.cparams;
    let mut gf = Graph::new();
    let n_tokens = gp.ubatch.n_tokens.max(1) as usize;
    let n_outputs = gp.n_outputs.max(1) as usize;
    let n_embd = model.hparams.n_embd.max(1) as usize;
    let n_embd_out = model.hparams.n_embd_out().max(1) as usize;
    let n_vocab = model.vocab.n_tokens.max(1) as usize;

    let inp_tokens = gf.add("inp_tokens", GraphOp::Input, [n_tokens, 1, 1, 1], &[], -1);
    let inp_pos = gf.add(
        "inp_pos",
        GraphOp::Input,
        [gp.ubatch.n_pos.max(1) as usize * n_tokens, 1, 1, 1],
        &[],
        -1,
    );

    let mut cur = gf.add(
        "embd",
        GraphOp::Embedding,
        [n_embd, n_tokens, 1, 1],
        &[inp_tokens],
        -1,
    );

    let mut fused_nodes = Vec::new();
    let probes = [
        (FusedOp::FlashAttn, c.flash_attn),
        (FusedOp::FlashAttnKvRows, c.fused_kv_rows),
        (FusedOp::Dsv4HcPre, c.fused_dsv4_hc_pre),
        (FusedOp::Dsv4HcComb, c.fused_dsv4_hc_comb),
        (FusedOp::Dsv4HcPost, c.fused_dsv4_hc_post),
        (FusedOp::TopKMask, c.fused_top_k_mask),
    ];
    for (op, enabled) in probes {
        if !enabled || !model.fused_support.contains(&op) {
            continue;
        }
        let gop = match op {
            FusedOp::FlashAttn => GraphOp::FlashAttn,
            FusedOp::FlashAttnKvRows => GraphOp::FlashAttnKvRows,
            FusedOp::Dsv4HcPre => GraphOp::Dsv4HcPre,
            FusedOp::Dsv4HcComb => GraphOp::Dsv4HcComb,
            FusedOp::Dsv4HcPost => GraphOp::Dsv4HcPost,
            FusedOp::TopKMask => GraphOp::TopKMask,
        };
        let id = gf.add(op.name(), gop, [n_embd, n_tokens, 1, 1], &[cur], 0);
        fused_nodes.push(FusedNode {
            op,
            il: 0,
            node: id,
        });
        cur = id;
    }

    // Hadamard folding: add the inverse lookup and the paired transforms.
    // Model table ids are remapped to ids of the nodes created here.
    let mut hadamard = super::model::HadamardInfo::default();
    let mut latent_lookups = Vec::new();
    for (_table, &(_rot, hint)) in &model.hadamard.rot {
        let table = gf.add("latent_table", GraphOp::Input, [4, 4, 1, 1], &[], -1);
        let rot = gf.add("hadamard_rot", GraphOp::Input, [4, 4, 1, 1], &[], -1);
        let lookup = gf.add(
            "latent_lookup",
            GraphOp::GetRows,
            [4, n_tokens, 1, 1],
            &[table, inp_tokens],
            -1,
        );
        hadamard.inv.insert(table);
        latent_lookups.push(lookup);
        let transform = gf.add(
            "act_transform",
            GraphOp::MulMat,
            [4, n_tokens, 1, 1],
            &[cur, rot],
            -1,
        );
        gf.node_mut(transform).expect("node just added").hint = HINT_SRC0_IS_HADAMARD;
        let _head = gf.add(
            "folded_head",
            GraphOp::MulMat,
            [n_embd.min(4), n_tokens, 1, 1],
            &[table, transform],
            -1,
        );
        hadamard.rot.insert(table, (rot, hint));
        cur = transform;
    }
    // The latent lookup must be consumed by a transformed head.
    for lookup in latent_lookups {
        let use_id = gf.add(
            "latent_head",
            GraphOp::MulMat,
            [4, n_tokens, 1, 1],
            &[lookup, lookup],
            -1,
        );
        gf.node_mut(use_id).expect("node just added").hint = HINT_SRC0_IS_HADAMARD;
    }

    let has_logits =
        !c.embeddings || gp.graph_has_logits || model.arch == crate::context::model::Arch::T5;
    let logits = if has_logits {
        Some(gf.add(
            "logits",
            GraphOp::OutputHead,
            [n_vocab, n_outputs, 1, 1],
            &[inp_tokens],
            -1,
        ))
    } else {
        None
    };

    let embd = if c.embeddings { Some(cur) } else { None };
    let embd_pooled = if c.embeddings && c.pooling_type != crate::cparams::PoolingType::None {
        Some(gf.add(
            "embd_pooled",
            GraphOp::PoolMean,
            [n_embd_out, 1, 1, 1],
            &[embd.unwrap_or(cur)],
            -1,
        ))
    } else {
        None
    };

    let h_nextn = if c.embeddings_nextn {
        Some(gf.add(
            "h_nextn",
            GraphOp::View,
            [n_embd_out, n_tokens, 1, 1],
            &[cur],
            -1,
        ))
    } else {
        None
    };

    let mut layer_inp = Vec::new();
    for (il, &enabled) in c.embeddings_layer_inp.iter().enumerate() {
        if !enabled {
            layer_inp.push(None);
            continue;
        }
        let id = gf.add(
            &format!("layer_inp-{}", il),
            GraphOp::View,
            [n_embd, n_tokens, 1, 1],
            &[cur],
            il as i32 - 1,
        );
        layer_inp.push(Some(id));
    }

    let (mut t_sampled, mut t_sampled_logits, mut t_sampled_probs, mut t_candidates) =
        (None, None, None, None);
    if !gp.samplers.is_empty() {
        if let Some(lid) = logits {
            t_sampled = Some(gf.add("sampled", GraphOp::Argmax, [n_outputs, 1, 1, 1], &[lid], -1));
            t_sampled_logits = Some(gf.add(
                "sampled_logits",
                GraphOp::View,
                [n_vocab, n_outputs, 1, 1],
                &[lid],
                -1,
            ));
            t_sampled_probs = Some(gf.add(
                "sampled_probs",
                GraphOp::Softmax,
                [n_vocab, n_outputs, 1, 1],
                &[lid],
                -1,
            ));
            t_candidates = Some(gf.add(
                "candidates",
                GraphOp::View,
                [n_vocab, n_outputs, 1, 1],
                &[lid],
                -1,
            ));
        }
    }

    let inputs = gf
        .nodes
        .iter()
        .filter(|n| n.is_input)
        .map(|n| n.id)
        .collect();
    Ok(GraphResult {
        gf,
        logits,
        embd,
        embd_pooled,
        h_nextn,
        layer_inp,
        t_sampled,
        t_sampled_logits,
        t_sampled_probs,
        t_candidates,
        inp_tokens: Some(inp_tokens),
        inp_pos: Some(inp_pos),
        fused_nodes,
        inputs,
        sig: Some(signature(gp)),
        hadamard,
    })
}

/// Which cparams boolean a fused-op probe controls.
#[derive(Clone, Copy, Debug)]
enum FusedFlag {
    FlashAttn,
    KvRows,
    Dsv4Pre,
    Dsv4Comb,
    Dsv4Post,
    TopKMask,
}

impl FusedFlag {
    fn get(self, c: &CParams) -> bool {
        match self {
            Self::FlashAttn => c.flash_attn,
            Self::KvRows => c.fused_kv_rows,
            Self::Dsv4Pre => c.fused_dsv4_hc_pre,
            Self::Dsv4Comb => c.fused_dsv4_hc_comb,
            Self::Dsv4Post => c.fused_dsv4_hc_post,
            Self::TopKMask => c.fused_top_k_mask,
        }
    }

    fn set(self, c: &mut CParams, value: bool) {
        match self {
            Self::FlashAttn => c.flash_attn = value,
            Self::KvRows => c.fused_kv_rows = value,
            Self::Dsv4Pre => c.fused_dsv4_hc_pre = value,
            Self::Dsv4Comb => c.fused_dsv4_hc_comb = value,
            Self::Dsv4Post => c.fused_dsv4_hc_post = value,
            Self::TopKMask => c.fused_top_k_mask = value,
        }
    }
}

/// `ubatch_prepare_reserve`: assign one sequence per token and mark outputs.
pub fn ubatch_prepare_reserve(
    ubatch: &mut Ubatch,
    n_outputs: u32,
    samplers: &BTreeMap<SeqId, BackendSampler>,
    n_outputs_max_per_seq: u32,
) {
    let n_seqs = ubatch.n_seqs as usize;
    let n_seq_tokens = ubatch.n_seq_tokens as usize;
    for s in 0..n_seqs {
        for t in 0..n_seq_tokens {
            let i = s * n_seq_tokens + t;
            if i >= ubatch.n_seq_id.len() {
                break;
            }
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = ubatch.seq_id_unq.get(s).copied().into_iter().collect();
        }
    }

    let mut has_sampler = vec![false; n_seqs];
    let mut sampler_seqs = Vec::new();
    for &seq_id in samplers.keys() {
        let s = seq_id as usize;
        if s < n_seqs {
            sampler_seqs.push(s);
            has_sampler[s] = true;
        }
    }

    let mut n_outputs_set = 0u32;
    let n_per_seq = (n_seq_tokens as u32).min(n_outputs_max_per_seq);
    for &s in &sampler_seqs {
        if n_outputs_set >= n_outputs {
            break;
        }
        for t in 0..n_per_seq as usize {
            let i = s * n_seq_tokens + t;
            if i >= ubatch.output.len() || n_outputs_set >= n_outputs {
                break;
            }
            ubatch.output[i] = 1;
            n_outputs_set += 1;
        }
    }
    for t in 0..n_seq_tokens {
        for s in 0..n_seqs {
            if n_outputs_set >= n_outputs {
                return;
            }
            if has_sampler[s] {
                continue;
            }
            let i = s * n_seq_tokens + t;
            if i < ubatch.output.len() {
                ubatch.output[i] = 1;
                n_outputs_set += 1;
            }
        }
    }
}

use super::LlamaContext;
/// `graph_max_nodes` and graph helpers implemented on the context.
use crate::cparams::ContextType;

impl LlamaContext {
    pub fn graph_max_nodes(&self) -> usize {
        let mut res = (1024usize)
            .max(self.model.graph_nodes_per_tensor() as usize * self.model.n_tensors() as usize);
        for lora in &self.model.loras {
            res += lora.n_nodes as usize;
        }
        if self.cparams.moe_cache_size > 0 {
            res += 32usize * self.model.hparams.n_layer_all as usize;
        }
        for sampler in self.sampling.samplers.values() {
            res += sampler.n_nodes * self.cparams.n_outputs_max_per_seq as usize;
        }
        if self.cparams.training {
            res *= 4;
        }
        res
    }

    pub fn get_gf_res_reserve(&self) -> &GraphResult {
        &self.gf_res_reserve
    }

    pub fn get_gf_res_prev(&self) -> &GraphResult {
        &self.gf_res_prev[usize::from(self.n_outputs > 0)]
    }

    fn ctx_graph_type(&self) -> GraphType {
        match self.cparams.ctx_type {
            ContextType::Default => GraphType::Default,
            ContextType::Mtp => GraphType::DecoderMtp,
        }
    }

    /// Build a graph for a dummy ubatch and reserve scheduler buffers.
    pub fn graph_reserve(
        &mut self,
        n_tokens: u32,
        n_seqs: u32,
        n_outputs: u32,
        snapshot: Option<MemorySnapshot>,
        split_only: bool,
        sizes: Option<&mut [usize]>,
    ) -> Result<Graph> {
        if n_seqs == 0 {
            return Err(Error::new("graph_reserve: n_seqs == 0"));
        }
        let mut n_tokens = n_tokens;
        if n_tokens % n_seqs != 0 {
            n_tokens = ((n_tokens + n_seqs - 1) / n_seqs) * n_seqs;
        }
        if n_outputs == 0 {
            return Err(Error::new("graph_reserve: n_outputs == 0"));
        }

        self.sched.reset();
        for res in &mut self.gf_res_prev {
            res.reset();
        }
        self.gf_res_prev_active = None;

        let save_n_outputs = self.n_outputs;
        self.n_outputs = n_outputs as i64;

        let mut balloc = BatchAllocr::new(self.model.hparams.n_pos_per_embd() as usize, true);
        let mut ubatch = balloc.ubatch_reserve(n_tokens / n_seqs, n_seqs);
        ubatch_prepare_reserve(
            &mut ubatch,
            n_outputs,
            &self.sampling.samplers,
            self.cparams.n_outputs_max_per_seq,
        );

        let gp = self.graph_params_for(&ubatch, snapshot, n_outputs, self.ctx_graph_type());
        let res = build_graph(&self.model, &gp)?;

        let verify_info = if res.hadamard.rot.is_empty() && res.hadamard.inv.is_empty() {
            self.model.hadamard.clone()
        } else {
            res.hadamard.clone()
        };
        if !self.hadamard_verified && (!verify_info.rot.is_empty() || !verify_info.inv.is_empty()) {
            verify_hadamard_graph(&res.gf, &verify_info)?;
            self.hadamard_verified = true;
        }

        self.n_input_tensors = graph_n_input_tensors(&res.gf) as u32;
        self.n_outputs = save_n_outputs;
        self.gf_res_reserve = res.clone();

        if split_only {
            match sizes {
                Some(sizes) => self.sched.reserve_size(&res.gf, sizes),
                None => self.sched.split_graph(&res.gf),
            }
        } else if !self.sched.reserve(&res.gf) {
            return Err(Error::new(
                "graph_reserve: failed to allocate compute buffers",
            ));
        }
        Ok(res.gf)
    }

    /// Assemble `llm_graph_params` from the context state.
    pub fn graph_params_for(
        &self,
        ubatch: &Ubatch,
        snapshot: Option<MemorySnapshot>,
        n_outputs: u32,
        gtype: GraphType,
    ) -> GraphParams {
        GraphParams {
            cparams: self.cparams.clone(),
            params: self.params.clone(),
            ubatch: ubatch.clone(),
            gtype,
            mctx: snapshot,
            n_outputs,
            samplers: self.sampling.samplers.clone(),
            graph_has_logits: self.graph_has_logits,
            layer_inp_src: self.supports_embd_row(),
            h_nextn_src: self.supports_state_row(),
        }
    }

    /// Probe one auto-fused op and disable it when its backend cannot run it.
    fn resolve_fused_op(
        &mut self,
        probe: FusedOpProbe,
        flag: FusedFlag,
        snapshot: Option<MemorySnapshot>,
        n_seqs: u32,
    ) -> Result<()> {
        if !flag.get(&self.cparams) {
            return Ok(());
        }
        let n_tokens = probe.n_tokens_per_seq * n_seqs;
        self.graph_reserve(n_tokens, n_seqs, n_tokens, snapshot, true, None)?;
        let supported = self
            .gf_res_reserve
            .fused_nodes
            .iter()
            .filter(|node| node.op == probe.op)
            .all(|node| {
                self.sched.tensor_backend(node.node).is_some()
                    && self.model.fused_support.contains(&probe.op)
                    && self
                        .model
                        .dev_layer(node.il as u32)
                        .map(|dev| self.model.devices.contains(&dev))
                        .unwrap_or(false)
            });
        if supported {
            crate::llama_log_info!("resolve_fused_ops: {} enabled", probe.name);
        } else {
            crate::llama_log_warn!("resolve_fused_ops: {} not supported, disabled", probe.name);
            flag.set(&mut self.cparams, false);
        }
        Ok(())
    }

    /// Probe each auto-fused op.
    pub fn resolve_fused_ops(
        &mut self,
        snapshot: Option<MemorySnapshot>,
        n_seqs: u32,
    ) -> Result<()> {
        if self.cparams.auto_fa {
            self.resolve_fused_op(
                FUSED_PROBE_FLASH_ATTN,
                FusedFlag::FlashAttn,
                snapshot,
                n_seqs,
            )?;
            self.cparams.auto_fa = false;
        }
        if self.cparams.auto_fkvr {
            self.cparams.fused_kv_rows = self.cparams.flash_attn;
            self.resolve_fused_op(
                FUSED_PROBE_FLASH_ATTN_KV_ROWS,
                FusedFlag::KvRows,
                snapshot,
                n_seqs,
            )?;
            self.cparams.auto_fkvr = false;
        }
        if self.cparams.auto_fhc {
            self.resolve_fused_op(
                FUSED_PROBE_DSV4_HC_PRE,
                FusedFlag::Dsv4Pre,
                snapshot,
                n_seqs,
            )?;
            self.resolve_fused_op(
                FUSED_PROBE_DSV4_HC_COMB,
                FusedFlag::Dsv4Comb,
                snapshot,
                n_seqs,
            )?;
            self.resolve_fused_op(
                FUSED_PROBE_DSV4_HC_POST,
                FusedFlag::Dsv4Post,
                snapshot,
                n_seqs,
            )?;
            self.cparams.auto_fhc = false;
        }
        if self.cparams.auto_ftkm {
            self.resolve_fused_op(
                FUSED_PROBE_TOP_K_MASK,
                FusedFlag::TopKMask,
                snapshot,
                n_seqs,
            )?;
            self.cparams.auto_ftkm = false;
        }
        Ok(())
    }

    /// Create a new scheduler when the current one cannot serve `params`.
    pub fn sched_new(&mut self, params: SchedParams) {
        if self.sched.params.can_reuse(&params) {
            return;
        }
        let n_backends = self.backends.len().max(1);
        self.sched = Scheduler::new(params);
        self.sched.n_backends = n_backends;
        self.sched_params_cur = params;
    }

    /// Reserve worst-case PP and TG graphs.
    pub fn sched_reserve(&mut self) -> Result<()> {
        if !self.sched_need_reserve {
            return Ok(());
        }
        self.sched_need_reserve = false;
        self.synchronize();

        let n_seqs = self.cparams.n_seq_max.max(1);
        let n_tokens = self.cparams.n_ctx.min(self.cparams.n_ubatch).max(1);
        let max_nodes = self.graph_max_nodes();

        self.gf_res_prev_active = None;
        self.gf_res_reserve.reset();
        for res in &mut self.gf_res_prev {
            res.reset();
        }

        self.sched_new(SchedParams {
            max_nodes,
            parallel: self.cparams.pipeline_parallel,
            op_offload: self.cparams.op_offload,
        });

        let snapshot = match self.memory.as_ref() {
            Some(mem) => Some(MemorySnapshot {
                n_kv: mem.borrow().get_size(),
                n_kv_rows: 0,
                type_k: mem.borrow().type_k(),
                type_v: mem.borrow().type_v(),
            }),
            None => None,
        };

        self.resolve_fused_ops(snapshot, n_seqs)?;

        let n_outputs_pp = n_tokens.min(self.cparams.n_outputs_max).max(1);
        let mut gf = if self.model.no_alloc {
            let mut sizes = std::mem::take(&mut self.backend_buf_exp_size);
            let result = self.graph_reserve(
                n_tokens,
                n_seqs,
                n_outputs_pp,
                snapshot,
                true,
                Some(&mut sizes[..]),
            );
            self.backend_buf_exp_size = sizes;
            result
        } else {
            self.graph_reserve(n_tokens, n_seqs, n_outputs_pp, snapshot, false, None)
        };
        if gf.is_err() && self.cparams.pipeline_parallel {
            self.cparams.pipeline_parallel = false;
            self.sched_new(SchedParams {
                max_nodes,
                parallel: false,
                op_offload: self.cparams.op_offload,
            });
            gf = self.graph_reserve(n_tokens, n_seqs, n_outputs_pp, snapshot, false, None);
        }
        if gf.is_err() {
            return Err(Error::new("failed to allocate compute pp buffers"));
        }
        self.graph_has_logits = self.gf_res_reserve.get_logits().is_some();

        if !self.cparams.training {
            self.graph_reserve(n_seqs, n_seqs, n_seqs, snapshot, false, None)
                .map_err(|_| Error::new("failed to allocate compute tg buffers"))?;
        }

        let reserve_tokens = n_tokens;
        let reserve_seqs = if self.model.arch.reserve_single_seq() {
            1
        } else {
            n_seqs
        };
        self.graph_reserve(
            reserve_tokens,
            reserve_seqs,
            n_outputs_pp,
            snapshot,
            false,
            None,
        )
        .map_err(|_| Error::new("failed to allocate compute pp buffers"))?;

        for i in 0..self.backends.len() {
            self.backend_buf_exp_size[i] = self.sched.params.max_nodes * 64;
        }
        Ok(())
    }

    /// Set backend thread counts and run one graph.
    pub fn graph_compute(&mut self, gf: &mut Graph, batched: bool) -> i32 {
        let n_threads = if batched {
            self.cparams.n_threads_batch
        } else {
            self.cparams.n_threads
        };
        for &idx in &self.set_n_threads_fns {
            if let Some(backend) = self.backends.get_mut(idx) {
                backend.n_threads = n_threads;
            }
        }
        self.copy_experts.reset();
        let status = self.sched.graph_compute_async(gf);
        if status != 0 {
            crate::llama_log_error!("graph_compute failed with status {}", status);
        }
        status
    }

    /// Trace unexpected scheduler reallocations.
    pub fn sched_graph_realloc(&mut self, backend_ids_changed: bool, graph_size_changed: bool) {
        if backend_ids_changed {
            return;
        }
        if graph_size_changed {
            self.n_reallocs_diff += 1;
        } else {
            self.n_reallocs_same += 1;
        }
    }
}

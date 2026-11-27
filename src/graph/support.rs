//! Local stand-ins for model, adapter, memory and sampling interfaces used by
//! `llama-graph.cpp`. They hold real data and implement the same operations on
//! graph tensors; a host backend supplies the concrete instances.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::arch::Arch;
use crate::batch::{Pos, Ubatch};
use crate::cparams::CParams;
use crate::core::SeqId;
use crate::hparams::{Hparams, RopeType};

use super::ggml::{
    self, add, mul, new_tensor_1d, new_tensor_3d, Op, Prec, Tensor, Type,
};
use super::types::{AttnMask, FfnOpType};

// ---------------------------------------------------------------------------
// Backend / scheduler
// ---------------------------------------------------------------------------

/// `ggml_backend_dev_type` values used by the graph builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Gpu,
}

impl Backend {
    pub fn is_cpu(self) -> bool {
        self == Self::Cpu
    }

    pub fn supports_op(self, _op: Op) -> bool {
        true
    }
}

/// `ggml_backend_sched_t`: records forced tensor placements.
#[derive(Debug, Default)]
pub struct Scheduler {
    forced: BTreeMap<usize, Backend>,
}

impl Scheduler {
    pub fn set_tensor_backend(&mut self, tensor: &Tensor, backend: Backend) {
        self.forced.insert(tensor.id(), backend);
    }

    pub fn tensor_backend(&self, tensor: &Tensor) -> Option<Backend> {
        self.forced.get(&tensor.id()).copied()
    }
}

// ---------------------------------------------------------------------------
// Precision policy, Hadamard folds, LoRA, control vectors
// ---------------------------------------------------------------------------

/// `llama_prec_policy`: per-node accumulation precision.
#[derive(Clone, Copy, Debug)]
pub struct PrecPolicy {
    pub acc: Prec,
    pub src1: Prec,
}

impl Default for PrecPolicy {
    fn default() -> Self {
        Self { acc: Prec::F32, src1: Prec::F32 }
    }
}

impl PrecPolicy {
    /// `prec_policy.apply(res, w)`.
    pub fn apply(&self, res: &Tensor, _w: &Tensor) {
        ggml::prec_set_acc(res, self.acc);
    }

    pub fn get_src1(&self, _w: &Tensor) -> Prec {
        self.src1
    }
}

/// One Hadamard transform entry from `llama_hadamard`.
#[derive(Clone, Debug)]
pub struct HadamardRotation {
    pub rot: Tensor,
    pub signs: Option<Tensor>,
    pub perm_rep: usize,
    pub perm_hd: usize,
    pub perm_nk: usize,
}

impl HadamardRotation {
    pub fn new(rot: Tensor) -> Self {
        Self { rot, signs: None, perm_rep: 1, perm_hd: 1, perm_nk: 1 }
    }
}

/// `llama_hadamard`: folded weights keyed by tensor id.
#[derive(Clone, Debug, Default)]
pub struct Hadamard {
    pub rot: BTreeMap<usize, HadamardRotation>,
    pub inv: BTreeMap<usize, HadamardRotation>,
}

impl Hadamard {
    pub fn get_rot(&self, w: &Tensor) -> Option<&HadamardRotation> {
        self.rot.get(&w.id())
    }
    pub fn get_inv(&self, w: &Tensor) -> Option<&HadamardRotation> {
        self.inv.get(&w.id())
    }
}

/// `llama_adapter_lora_weight`.
#[derive(Clone, Debug)]
pub struct LoraWeight {
    pub a: Tensor,
    pub b: Tensor,
}

impl LoraWeight {
    pub fn get_scale(&self, alpha: f32, adapter_scale: f32) -> f32 {
        let rank = self.b.ne(0) as f32;
        if alpha != 0.0 { adapter_scale * alpha / rank } else { adapter_scale }
    }
}

/// `llama_adapter_lora`.
#[derive(Clone, Debug, Default)]
pub struct LoraAdapter {
    pub alpha: f32,
    pub weights: BTreeMap<usize, LoraWeight>,
}

impl LoraAdapter {
    pub fn get_weight(&self, w: &Tensor) -> Option<LoraWeight> {
        self.weights.get(&w.id()).cloned()
    }
}

/// `llama_adapter_cvec`: optional per-token scale and shift.
#[derive(Clone, Debug, Default)]
pub struct Cvec {
    pub scale: Option<Tensor>,
    pub shift: Option<Tensor>,
}

impl Cvec {
    pub fn apply_to(&self, cur: &Tensor, _il: i32) -> Tensor {
        let mut out = cur.clone();
        if let Some(scale) = &self.scale {
            out = mul(&out, scale);
        }
        if let Some(shift) = &self.shift {
            out = add(&out, shift);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Model / layer / RoPE
// ---------------------------------------------------------------------------

/// `llama_rope_params`.
#[derive(Clone, Debug)]
pub struct RopeParams {
    pub n_rot: u32,
    pub n_offs: u32,
    pub rope_type: u32,
    pub factors: Option<Tensor>,
    pub n_ctx_orig: u32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl Default for RopeParams {
    fn default() -> Self {
        Self {
            n_rot: 0,
            n_offs: 0,
            rope_type: 0,
            factors: None,
            n_ctx_orig: 0,
            freq_base: 0.0,
            freq_scale: 0.0,
            ext_factor: 0.0,
            attn_factor: 0.0,
            beta_fast: 0.0,
            beta_slow: 0.0,
        }
    }
}

impl RopeParams {
    /// `GGML_ROPE_TYPE_MROPE`.
    pub const MROPE: u32 = 0x1_0000;
    pub fn is_mrope(&self) -> bool {
        self.rope_type & Self::MROPE != 0
    }
}

fn rope_type_bits(rope_type: RopeType) -> u32 {
    match rope_type {
        RopeType::None => 0,
        RopeType::Normal => 0,
        RopeType::Mrope => RopeParams::MROPE,
        RopeType::Imrope => 0x2_0000,
    }
}

/// `llama_layer`: the projection tensors consumed by `build_qkv`.
#[derive(Clone, Debug, Default)]
pub struct Layer {
    pub wq: Option<Tensor>,
    pub wq_b: Option<Tensor>,
    pub wq_s: Option<Tensor>,
    pub wk: Option<Tensor>,
    pub wk_b: Option<Tensor>,
    pub wk_s: Option<Tensor>,
    pub wv: Option<Tensor>,
    pub wv_b: Option<Tensor>,
    pub wv_s: Option<Tensor>,
    pub wqkv: Option<Tensor>,
    pub wqkv_b: Option<Tensor>,
    pub wqkv_s: Option<Tensor>,
}

/// `llama_model` subset needed by the graph builder.
pub struct Model {
    pub arch: Arch,
    pub hparams: Hparams,
    pub prec_policy: PrecPolicy,
    pub hdmd: Option<Hadamard>,
    /// Per-layer device placement.
    pub devices: Vec<Backend>,
    /// Registered weight scales, keyed by tensor id.
    pub w_scales: BTreeMap<usize, Tensor>,
}

impl Model {
    pub fn new(arch: Arch, hparams: Hparams) -> Self {
        Self {
            arch,
            hparams,
            prec_policy: PrecPolicy::default(),
            hdmd: None,
            devices: Vec::new(),
            w_scales: BTreeMap::new(),
        }
    }

    pub fn dev_layer(&self, il: i32) -> Backend {
        if il < 0 {
            return Backend::Cpu;
        }
        self.devices.get(il as usize).copied().unwrap_or(Backend::Cpu)
    }

    pub fn get_w_scale(&self, w: &Tensor) -> Option<Tensor> {
        self.w_scales.get(&w.id()).cloned()
    }

    pub fn get_rope_params(&self, cparams: &CParams, il: i32) -> RopeParams {
        let ilu = il.max(0) as u32;
        let has_rope = self.hparams.has_rope(ilu).unwrap_or(true);
        let n_rot = if has_rope { self.hparams.n_rot(ilu).unwrap_or(0) } else { 0 };
        let n_offs = if self.hparams.n_lora_kv > 0 {
            self.hparams.n_embd_head_k(ilu).unwrap_or(0).saturating_sub(n_rot)
        } else {
            0
        };
        RopeParams {
            n_rot,
            n_offs,
            rope_type: rope_type_bits(self.hparams.rope_type),
            factors: None,
            n_ctx_orig: cparams.n_ctx_orig_yarn,
            freq_base: cparams.rope_freq_base,
            freq_scale: cparams.rope_freq_scale,
            ext_factor: cparams.yarn_ext_factor,
            attn_factor: cparams.yarn_attn_factor,
            beta_fast: cparams.yarn_beta_fast,
            beta_slow: cparams.yarn_beta_slow,
        }
    }

    pub fn get_rope_params_indexer(&self, cparams: &CParams, il: i32) -> RopeParams {
        let mut rp = self.get_rope_params(cparams, il);
        rp.n_rot = self.hparams.n_rot(il.max(0) as u32).unwrap_or(0);
        rp.n_offs = self.hparams.n_embd_head_k(il.max(0) as u32).unwrap_or(0) - rp.n_rot;
        rp
    }
}

// ---------------------------------------------------------------------------
// Memory contexts
// ---------------------------------------------------------------------------

/// Compression plan returned by a DSV4 cache context.
#[derive(Clone, Debug, Default)]
pub struct Dsv4CompPlan {
    pub n_kv: u32,
    pub n_stream: i64,
    pub n_visible: Vec<i32>,
    pub state_pos: Vec<i32>,
    pub state_persist_src_idxs: Vec<i32>,
    pub state_persist_dst_idxs: Vec<i32>,
    pub state_restore_src_idxs: Vec<i32>,
    pub state_restore_dst_idxs: Vec<i32>,
    pub state_snapshot_src_idxs: Vec<i32>,
    pub state_snapshot_dst_idxs: Vec<i32>,
    pub state_read_idxs: Vec<i32>,
    pub state_write_idxs: Vec<i64>,
    pub state_write_pos: Vec<i32>,
}

/// A stand-in for any `llama_memory_context_i` subclass. Fields that the
/// active architecture does not use stay empty, as in the C++ subclasses that
/// do not implement the corresponding methods.
#[derive(Default)]
pub struct MemoryStandIn {
    pub n_kv: u32,
    pub n_kv_rows: u32,
    pub n_rs: usize,
    pub head: u32,
    pub rs_z: i32,
    pub rs_size: usize,
    pub rs_in_place: bool,
    pub rs_snapshots: bool,
    pub rs_rep: Vec<usize>,
    pub k_rot: Vec<Option<Tensor>>,
    pub v_rot: Vec<Option<Tensor>>,
    pub k_scale: BTreeMap<i32, Tensor>,
    pub v_scale: BTreeMap<i32, Tensor>,
    pub base: Option<Rc<dyn MemoryContext>>,
    pub swa: Option<Rc<dyn MemoryContext>>,
    pub mla: Option<Rc<dyn MemoryContext>>,
    pub lid: Option<Rc<dyn MemoryContext>>,
    pub idx: Option<Rc<dyn MemoryContext>>,
    pub raw: Option<Rc<dyn MemoryContext>>,
    pub dsa: Option<Rc<dyn MemoryContext>>,
    pub attn: Option<Rc<dyn MemoryContext>>,
    pub recr: Option<Rc<dyn MemoryContext>>,
    pub plan_csa: Dsv4CompPlan,
    pub plan_hca: Dsv4CompPlan,
    pub plan_lid: Dsv4CompPlan,
}

/// The memory interfaces used by the graph input and attention builders.
pub trait MemoryContext {
    fn get_n_kv(&self) -> u32 {
        0
    }
    fn get_n_kv_rows(&self) -> u32 {
        0
    }

    fn build_input_k_idxs(&self, ubatch: &Ubatch) -> Tensor {
        let t = new_tensor_1d(Type::I64, ubatch.n_tokens as i64);
        t.set_input();
        t
    }
    fn build_input_v_idxs(&self, ubatch: &Ubatch) -> Tensor {
        self.build_input_k_idxs(ubatch)
    }

    fn set_input_k_idxs(&self, dst: &Tensor, ubatch: &Ubatch) {
        let n = ubatch.n_tokens as usize;
        let idxs: Vec<i64> = (0..n)
            .map(|i| ubatch.batch_idxs.get(i).copied().unwrap_or(i) as i64)
            .collect();
        dst.set_data_i64(&idxs);
    }
    fn set_input_v_idxs(&self, dst: &Tensor, ubatch: &Ubatch) {
        self.set_input_k_idxs(dst, ubatch)
    }

    /// DSV4 raw caches address rows without a batch: write sequential rows.
    fn set_input_k_idxs_raw(&self, dst: &Tensor) {
        let n = dst.ne(0).max(0) as usize;
        let data: Vec<i64> = (0..n).map(|i| i as i64).collect();
        dst.set_data_i64(&data);
    }

    fn set_input_attn_mask(&self, mask: &mut AttnMask, ubatch: &Ubatch, causal: bool) {
        let Some(raw) = mask.raw.as_ref() else { return };
        let n_kv = raw.ne(0).max(0) as usize;
        let n_tokens = ubatch.n_tokens as usize;
        let mut data = vec![-f32::INFINITY; n_kv * n_tokens];
        for i1 in 0..n_tokens {
            let s1 = ubatch.seq_id.get(i1).and_then(|v| v.first()).copied().unwrap_or(0);
            let p1 = ubatch.pos.get(i1).copied().unwrap_or(0);
            for i0 in 0..n_tokens {
                let s0 = ubatch.seq_id.get(i0).and_then(|v| v.first()).copied().unwrap_or(0);
                let p0 = ubatch.pos.get(i0).copied().unwrap_or(0);
                if s0 != s1 || (causal && p0 > p1) || i0 >= n_kv {
                    continue;
                }
                data[i1 * n_kv + i0] = 0.0;
            }
        }
        raw.set_data_f32(&data);
    }

    fn set_input_pos_bucket(&self, dst: &Tensor, ubatch: &Ubatch) {
        let n_tokens = ubatch.n_tokens as usize;
        let mut data = vec![0i32; n_tokens * n_tokens];
        for j in 0..n_tokens {
            for i in 0..n_tokens {
                let x = ubatch.pos.get(i).copied().unwrap_or(0);
                let y = ubatch.pos.get(j).copied().unwrap_or(0);
                data[j * n_tokens + i] = super::output::relative_position_bucket(x, y, 32, true);
            }
        }
        dst.set_data_i32(&data);
    }

    fn set_input_relative_pos(&self, dst: &Tensor, ubatch: &Ubatch, extent: u32) {
        let n_kv = dst.ne(0).max(0) as usize;
        let n_tokens = ubatch.n_tokens as usize;
        let mut data = vec![extent as i32; n_kv * n_tokens];
        for i in 0..n_tokens {
            let p1 = ubatch.pos.get(i).copied().unwrap_or(0);
            for j in 0..n_kv {
                let p0 = if j < n_tokens {
                    ubatch.pos.get(j).copied().unwrap_or(0)
                } else {
                    p1 - extent as Pos
                };
                let dist = p1 - p0;
                data[i * n_kv + j] = if dist < 0 || dist > extent as i32 {
                    extent as i32
                } else {
                    dist
                };
            }
        }
        dst.set_data_i32(&data);
    }

    fn get_k_rot(&self) -> Vec<Option<Tensor>> {
        Vec::new()
    }
    fn get_v_rot(&self) -> Vec<Option<Tensor>> {
        Vec::new()
    }
    fn get_k_scale(&self, _il: i32) -> Option<Tensor> {
        None
    }
    fn get_v_scale(&self, _il: i32) -> Option<Tensor> {
        None
    }

    fn cpy_k(&self, src: &Tensor, idxs: &Tensor, _il: i32) -> Tensor {
        ggml::set_rows(&self.get_k(0), src, idxs)
    }
    fn cpy_v(&self, src: &Tensor, idxs: &Tensor, _il: i32) -> Tensor {
        ggml::set_rows(&self.get_v(0), src, idxs)
    }
    fn get_k(&self, _il: i32) -> Tensor {
        new_tensor_3d(Type::F32, 1, 1, self.get_n_kv() as i64)
    }
    fn get_v(&self, _il: i32) -> Tensor {
        new_tensor_3d(Type::F32, 1, 1, self.get_n_kv() as i64)
    }

    fn get_base(&self) -> Option<Rc<dyn MemoryContext>> {
        None
    }
    fn get_swa(&self) -> Option<Rc<dyn MemoryContext>> {
        None
    }
    fn get_mla(&self) -> Option<Rc<dyn MemoryContext>> {
        None
    }
    fn get_lid(&self) -> Option<Rc<dyn MemoryContext>> {
        None
    }
    fn get_idx(&self) -> Option<Rc<dyn MemoryContext>> {
        None
    }
    fn get_raw(&self) -> Option<Rc<dyn MemoryContext>> {
        None
    }
    fn get_dsa(&self) -> Option<Rc<dyn MemoryContext>> {
        None
    }
    fn get_attn(&self) -> Option<Rc<dyn MemoryContext>> {
        None
    }
    fn get_recr(&self) -> Option<Rc<dyn MemoryContext>> {
        None
    }

    fn get_csa_plan(&self, _ubatch: &Ubatch) -> Dsv4CompPlan {
        Dsv4CompPlan::default()
    }
    fn get_hca_plan(&self, _ubatch: &Ubatch) -> Dsv4CompPlan {
        Dsv4CompPlan::default()
    }
    fn get_lid_plan(&self, _ubatch: &Ubatch) -> Dsv4CompPlan {
        Dsv4CompPlan::default()
    }

    fn get_n_rs(&self) -> usize {
        0
    }
    fn s_copy(&self, i: usize) -> i32 {
        i as i32
    }
    fn get_rs_rep(&self, _s: usize) -> usize {
        0
    }
    fn get_head(&self) -> u32 {
        0
    }
    fn get_rs_z(&self) -> i32 {
        0
    }
    fn get_rs_in_place(&self) -> bool {
        false
    }
    fn get_rs_snapshots(&self) -> bool {
        false
    }
    fn get_size(&self) -> usize {
        0
    }
    fn get_r_l(&self, _il: i32) -> Tensor {
        new_tensor_1d(Type::F32, 1)
    }
}

impl MemoryContext for MemoryStandIn {
    fn get_n_kv(&self) -> u32 {
        self.n_kv
    }
    fn get_n_kv_rows(&self) -> u32 {
        self.n_kv_rows
    }
    fn get_k_rot(&self) -> Vec<Option<Tensor>> {
        self.k_rot.clone()
    }
    fn get_v_rot(&self) -> Vec<Option<Tensor>> {
        self.v_rot.clone()
    }
    fn get_k_scale(&self, il: i32) -> Option<Tensor> {
        self.k_scale.get(&il).cloned()
    }
    fn get_v_scale(&self, il: i32) -> Option<Tensor> {
        self.v_scale.get(&il).cloned()
    }
    fn get_k(&self, _il: i32) -> Tensor {
        new_tensor_3d(Type::F32, 1, 1, self.n_kv as i64)
    }
    fn get_v(&self, _il: i32) -> Tensor {
        new_tensor_3d(Type::F32, 1, 1, self.n_kv as i64)
    }
    fn get_base(&self) -> Option<Rc<dyn MemoryContext>> {
        self.base.clone()
    }
    fn get_swa(&self) -> Option<Rc<dyn MemoryContext>> {
        self.swa.clone()
    }
    fn get_mla(&self) -> Option<Rc<dyn MemoryContext>> {
        self.mla.clone()
    }
    fn get_lid(&self) -> Option<Rc<dyn MemoryContext>> {
        self.lid.clone()
    }
    fn get_idx(&self) -> Option<Rc<dyn MemoryContext>> {
        self.idx.clone()
    }
    fn get_raw(&self) -> Option<Rc<dyn MemoryContext>> {
        self.raw.clone()
    }
    fn get_dsa(&self) -> Option<Rc<dyn MemoryContext>> {
        self.dsa.clone()
    }
    fn get_attn(&self) -> Option<Rc<dyn MemoryContext>> {
        self.attn.clone()
    }
    fn get_recr(&self) -> Option<Rc<dyn MemoryContext>> {
        self.recr.clone()
    }
    fn get_csa_plan(&self, _ubatch: &Ubatch) -> Dsv4CompPlan {
        self.plan_csa.clone()
    }
    fn get_hca_plan(&self, _ubatch: &Ubatch) -> Dsv4CompPlan {
        self.plan_hca.clone()
    }
    fn get_lid_plan(&self, _ubatch: &Ubatch) -> Dsv4CompPlan {
        self.plan_lid.clone()
    }
    fn get_n_rs(&self) -> usize {
        self.n_rs
    }
    fn s_copy(&self, i: usize) -> i32 {
        if self.rs_in_place {
            i as i32
        } else {
            self.head as i32 + i as i32
        }
    }
    fn get_rs_rep(&self, s: usize) -> usize {
        self.rs_rep.get(s).copied().unwrap_or(0)
    }
    fn get_head(&self) -> u32 {
        self.head
    }
    fn get_rs_z(&self) -> i32 {
        self.rs_z
    }
    fn get_rs_in_place(&self) -> bool {
        self.rs_in_place
    }
    fn get_rs_snapshots(&self) -> bool {
        self.rs_snapshots
    }
    fn get_size(&self) -> usize {
        self.rs_size
    }
    fn get_r_l(&self, _il: i32) -> Tensor {
        new_tensor_1d(Type::F32, (self.n_rs * self.rs_size).max(1) as i64)
    }
}

// ---------------------------------------------------------------------------
// Sampling / MoE cache
// ---------------------------------------------------------------------------

/// `llama_sampler_data` passed to a backend sampler.
#[derive(Clone, Debug, Default)]
pub struct SamplerData {
    pub logits: Option<Tensor>,
    pub probs: Option<Tensor>,
    pub sampled: Option<Tensor>,
    pub candidates: Option<Tensor>,
}

/// `llama_sampler`: optional backend hooks. A sampler without a hook is a
/// host-only sampler and the graph leaves that output untouched.
pub struct Sampler {
    pub on_reset: Option<Box<dyn Fn()>>,
    pub on_set_input: Option<Box<dyn Fn()>>,
    pub on_apply: Option<Box<dyn Fn(&Rc<RefCell<ggml::Graph>>, &mut SamplerData)>>,
}

impl Sampler {
    pub fn new() -> Self {
        Self { on_reset: None, on_set_input: None, on_apply: None }
    }

    pub fn backend_reset(&self) {
        if let Some(f) = &self.on_reset {
            f();
        }
    }
    pub fn backend_set_input(&self) {
        if let Some(f) = &self.on_set_input {
            f();
        }
    }
    pub fn backend_apply(&self, gf: &Rc<RefCell<ggml::Graph>>, data: &mut SamplerData) {
        if let Some(f) = &self.on_apply {
            f(gf, data);
        }
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new()
    }
}

/// `llama_moe_cache_lookup`.
#[derive(Clone, Debug, Default)]
pub struct MoeCacheLookup {
    /// Expert slots; `None` means the router ids are used directly.
    pub slots: Option<Tensor>,
}

/// Arguments handed to `llama_moe_cache::build_lookup`.
#[derive(Clone, Debug)]
pub struct MoeLookupArgs {
    pub il: i32,
    pub up_exps: Tensor,
    pub gate_exps: Tensor,
    pub down_exps: Tensor,
    pub gate_up_exps: Option<Tensor>,
    pub weight_before_ffn: bool,
}

/// `llama_moe_cache`: cached-expert lookup and expert building hooks.
pub trait MoeCache {
    /// Expert weight tensor cached for `w`, if any.
    fn get_experts(&self, _w: &Tensor) -> Option<Tensor> {
        None
    }

    fn build_lookup(
        &self,
        _args: &MoeLookupArgs,
        _selected_experts: &Tensor,
    ) -> MoeCacheLookup {
        MoeCacheLookup::default()
    }

    fn build_experts(
        &self,
        cur: &Tensor,
        ids: &Tensor,
        slots: Option<&Tensor>,
        cached: bool,
        build: &dyn Fn(&Tensor, &Tensor, Option<&Tensor>, bool) -> Tensor,
    ) -> Tensor {
        let _ = slots;
        let _ = cached;
        build(cur, ids, None, true)
    }
}

/// Default cache: the router ids address the experts directly.
#[derive(Default)]
pub struct SimpleMoeCache;

impl MoeCache for SimpleMoeCache {}

/// Helper used by the graph builder for expert gating flavors.
pub fn gating_op_from(hparams: &Hparams) -> crate::hparams::ExpertGatingFuncType {
    match hparams.expert_gating_func {
        1 => crate::hparams::ExpertGatingFuncType::Softmax,
        2 => crate::hparams::ExpertGatingFuncType::Sigmoid,
        3 => crate::hparams::ExpertGatingFuncType::SoftmaxWeight,
        4 => crate::hparams::ExpertGatingFuncType::SqrtSoftplus,
        _ => crate::hparams::ExpertGatingFuncType::None,
    }
}

/// Convert the stored gating code to the FFN op used by `build_moe_ffn`.
pub fn ffn_op_default() -> FfnOpType {
    FfnOpType::Silu
}

/// Number of KV slices of an ubatch (`llama_kv_cache::get_n_kv_slices`).
pub fn get_n_kv_slices(ubatch: &Ubatch) -> u32 {
    if ubatch.n_seqs_unq == 1 {
        return 1;
    }
    if ubatch.equal_seqs() && ubatch.n_seqs == ubatch.n_seqs_unq {
        return ubatch.n_seqs;
    }
    if ubatch.n_tokens != 0 && ubatch.n_tokens % ubatch.n_seqs_unq == 0 {
        let n_tps = ubatch.n_tokens / ubatch.n_seqs_unq;
        let mut is_runs = true;
        for i in 0..ubatch.n_tokens as usize {
            let is_single = ubatch.n_seq_id.get(i).copied().unwrap_or(0) == 1;
            let first = ubatch.seq_id.get(i - i % n_tps as usize).and_then(|v| v.first());
            let cur = ubatch.seq_id.get(i).and_then(|v| v.first());
            if !is_single || first != cur {
                is_runs = false;
                break;
            }
        }
        if is_runs {
            return ubatch.n_seqs_unq;
        }
    }
    1
}

/// `GGML_PAD` used for the KV dimension of no-cache masks.
pub fn pad_to(value: i64, multiple: i64) -> i64 {
    ((value + multiple - 1) / multiple) * multiple
}

/// Reserved for sampler maps keyed by sequence id.
pub type SeqSamplerMap = BTreeMap<SeqId, Rc<dyn Fn()>>;

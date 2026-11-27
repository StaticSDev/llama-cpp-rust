//! DSV4 graph input helpers, translated from the `dsv4_*` functions in
//! `llama-graph.cpp`.

use std::cell::RefCell;
use std::rc::Rc;

use crate::batch::Ubatch;
use crate::cparams::CParams;
use crate::hparams::Hparams;

use super::ggml::{new_tensor_1d, new_tensor_4d, Tensor, Type};
use super::support::{Dsv4CompPlan, MemoryContext};
use super::types::{AttnMask, GraphInput, GraphParams};

/// `llm_graph_input_dsv4::comp_input`.
#[derive(Clone, Debug, Default)]
pub struct Dsv4CompInput {
    pub state_pos: Option<Tensor>,
    pub state_persist_src_idxs: Option<Tensor>,
    pub state_persist_dst_idxs: Option<Tensor>,
    pub state_restore_src_idxs: Option<Tensor>,
    pub state_restore_dst_idxs: Option<Tensor>,
    pub state_snapshot_src_idxs: Option<Tensor>,
    pub state_snapshot_dst_idxs: Option<Tensor>,
    pub state_read_idxs: Option<Tensor>,
    pub state_write_idxs: Option<Tensor>,
    pub state_write_pos: Option<Tensor>,
    pub kq_mask: Option<Tensor>,
    pub k_rot: Vec<Option<Tensor>>,
}

fn set_i32(dst: &Option<Tensor>, src: &[i32]) {
    if let Some(t) = dst {
        t.set_data_i32(src);
    }
}

fn set_i64(dst: &Option<Tensor>, src: &[i64]) {
    if let Some(t) = dst {
        t.set_data_i64(src);
    }
}

fn set_kq_mask(dst: &Option<Tensor>, plan: &Dsv4CompPlan, n_tokens: u32, n_stream: i64) {
    let Some(t) = dst else { return };
    let n_kv = t.ne(0).max(0) as usize;
    let n = n_tokens as usize;
    let mut data = vec![0.0f32; n * n_kv];
    for i in 0..n {
        let n_visible = plan.n_visible.get(i).copied().unwrap_or(0).max(0) as usize;
        for j in 0..n_kv {
            data[i * n_kv + j] = if j < n_visible { 0.0 } else { -f32::INFINITY };
        }
    }
    t.set_data_f32(&data);
    let _ = n_stream;
}

pub fn build_raw_kq_mask(
    mctx: &dyn MemoryContext,
    ubatch: &Ubatch,
    cparams: &CParams,
    n_stream: i64,
) -> Tensor {
    let n_kv = mctx.get_n_kv() as i64;
    let n_tokens = ubatch.n_tokens as i64;
    let t = new_tensor_4d(
        if cparams.flash_attn { Type::F16 } else { Type::F32 },
        [n_kv, n_tokens / n_stream.max(1), 1, n_stream],
    );
    t.set_input();
    t.set_name("attn_inp_kq_mask");
    t
}

pub fn can_reuse_raw_kq_mask(
    mask: &Tensor,
    mctx: &dyn MemoryContext,
    ubatch: &Ubatch,
    n_stream: i64,
) -> bool {
    mask.ne(0) == mctx.get_n_kv() as i64
        && mask.ne(1) == ubatch.n_tokens as i64 / n_stream
        && mask.ne(2) == 1
        && mask.ne(3) == n_stream
}

fn plan_positions(values: &[i32]) -> String {
    let mut out = String::from("[");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&v.to_string());
    }
    out.push(']');
    out
}

fn compress_debug() -> bool {
    std::env::var("LLAMA_DSV4_COMPRESS_DEBUG")
        .map(|v| v.parse::<i32>().unwrap_or(0) > 0)
        .unwrap_or(false)
}

pub fn set_comp_inputs(
    inp: &Dsv4CompInput,
    plan: &Dsv4CompPlan,
    name: &str,
    debug: bool,
    n_tokens: u32,
    n_stream: i64,
) {
    set_i32(&inp.state_pos, &plan.state_pos);
    set_i32(&inp.state_persist_src_idxs, &plan.state_persist_src_idxs);
    set_i32(&inp.state_persist_dst_idxs, &plan.state_persist_dst_idxs);
    set_i32(&inp.state_restore_src_idxs, &plan.state_restore_src_idxs);
    set_i32(&inp.state_restore_dst_idxs, &plan.state_restore_dst_idxs);
    set_i32(&inp.state_snapshot_src_idxs, &plan.state_snapshot_src_idxs);
    set_i32(&inp.state_snapshot_dst_idxs, &plan.state_snapshot_dst_idxs);
    set_i32(&inp.state_read_idxs, &plan.state_read_idxs);
    set_i64(&inp.state_write_idxs, &plan.state_write_idxs);
    set_i32(&inp.state_write_pos, &plan.state_write_pos);
    set_kq_mask(&inp.kq_mask, plan, n_tokens, n_stream);

    if debug || compress_debug() {
        crate::llama_log_info!(
            "dsv4_set_comp_inputs: {} n_tokens={}, n_stream={}, state_persist_dst={}, state_write_pos={}",
            name,
            n_tokens,
            n_stream,
            plan_positions(&plan.state_persist_dst_idxs),
            plan_positions(&plan.state_write_pos)
        );
    }
}

fn can_reuse_tensor_1d(t: &Option<Tensor>, ne0: usize) -> bool {
    matches!(t, None if ne0 == 0) || matches!(t, Some(tensor) if tensor.ne(0) == ne0 as i64)
}

fn can_reuse_kq_mask_1d(
    t: &Option<Tensor>,
    plan: &Dsv4CompPlan,
    n_tokens: u32,
    n_stream: i64,
) -> bool {
    if plan.n_kv == 0 {
        return t.is_none();
    }
    matches!(t, Some(mask)
        if mask.ne(0) == plan.n_kv as i64
            && mask.ne(1) == n_tokens as i64 / n_stream
            && mask.ne(2) == 1
            && mask.ne(3) == n_stream)
}

pub fn can_reuse_comp_input(
    inp: &Dsv4CompInput,
    plan: &Dsv4CompPlan,
    n_tokens: u32,
    n_stream: i64,
) -> bool {
    let mut res = true;
    res &= can_reuse_tensor_1d(&inp.state_pos, plan.state_pos.len());
    res &= can_reuse_tensor_1d(&inp.state_persist_src_idxs, plan.state_persist_src_idxs.len());
    res &= can_reuse_tensor_1d(&inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.len());
    res &= can_reuse_tensor_1d(&inp.state_restore_src_idxs, plan.state_restore_src_idxs.len());
    res &= can_reuse_tensor_1d(&inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.len());
    res &= can_reuse_tensor_1d(&inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.len());
    res &= can_reuse_tensor_1d(&inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.len());
    res &= can_reuse_tensor_1d(&inp.state_read_idxs, plan.state_read_idxs.len());
    res &= can_reuse_tensor_1d(&inp.state_write_idxs, plan.state_write_idxs.len());
    res &= can_reuse_tensor_1d(&inp.state_write_pos, plan.state_write_pos.len());
    res &= can_reuse_kq_mask_1d(&inp.kq_mask, plan, n_tokens, n_stream);
    res
}

fn build_input_1d(ty: Type, ne0: usize, name: &str) -> Option<Tensor> {
    if ne0 == 0 {
        return None;
    }
    let t = new_tensor_1d(ty, ne0 as i64);
    t.set_input();
    t.set_name(name);
    Some(t)
}

pub fn build_comp_inputs(
    inp: &mut Dsv4CompInput,
    plan: &Dsv4CompPlan,
    name: &str,
    cparams: &CParams,
    n_stream: i64,
) {
    inp.state_pos = build_input_1d(Type::I32, plan.state_pos.len(), &format!("dsv4_{}_state_pos", name));
    inp.state_persist_src_idxs = build_input_1d(
        Type::I32, plan.state_persist_src_idxs.len(), &format!("dsv4_{}_state_persist_src_idxs", name));
    inp.state_persist_dst_idxs = build_input_1d(
        Type::I32, plan.state_persist_dst_idxs.len(), &format!("dsv4_{}_state_persist_dst_idxs", name));
    inp.state_restore_src_idxs = build_input_1d(
        Type::I32, plan.state_restore_src_idxs.len(), &format!("dsv4_{}_state_restore_src_idxs", name));
    inp.state_restore_dst_idxs = build_input_1d(
        Type::I32, plan.state_restore_dst_idxs.len(), &format!("dsv4_{}_state_restore_dst_idxs", name));
    inp.state_snapshot_src_idxs = build_input_1d(
        Type::I32, plan.state_snapshot_src_idxs.len(), &format!("dsv4_{}_state_snapshot_src_idxs", name));
    inp.state_snapshot_dst_idxs = build_input_1d(
        Type::I32, plan.state_snapshot_dst_idxs.len(), &format!("dsv4_{}_state_snapshot_dst_idxs", name));
    inp.state_read_idxs = build_input_1d(
        Type::I32, plan.state_read_idxs.len(), &format!("dsv4_{}_state_read_idxs", name));
    inp.state_write_idxs = build_input_1d(
        Type::I64, plan.state_write_idxs.len(), &format!("dsv4_{}_state_write_idxs", name));
    inp.state_write_pos = build_input_1d(
        Type::I32, plan.state_write_pos.len(), &format!("dsv4_{}_state_write_pos", name));

    if plan.n_kv > 0 {
        let n_tokens = plan.n_visible.len() as i64;
        let ty = if name == "lid" || cparams.flash_attn { Type::F16 } else { Type::F32 };
        let t = new_tensor_4d(ty, [plan.n_kv as i64, n_tokens / n_stream.max(1), 1, n_stream]);
        t.set_input();
        t.set_name(&format!("dsv4_{}_kq_mask", name));
        inp.kq_mask = Some(t);
    }
}

/// `llm_graph_input_dsv4_raw`.
#[derive(Default)]
pub struct GraphInputDsv4Raw {
    pub self_k_idxs: Option<Tensor>,
    pub self_attn_mask: AttnMask,
    pub self_k_rot: Vec<Option<Tensor>>,
    pub cparams: CParams,
    pub mctx: Option<Rc<dyn MemoryContext>>,
}

impl GraphInputDsv4Raw {
    pub fn new(cparams: &CParams, mctx: Rc<dyn MemoryContext>) -> Self {
        Self { cparams: cparams.clone(), mctx: Some(mctx), ..Default::default() }
    }
}

impl GraphInput for GraphInputDsv4Raw {
    fn name(&self) -> &'static str {
        "llm_graph_input_dsv4_raw"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(mctx) = self.mctx.clone() else { return };
        if let Some(idxs) = &self.self_k_idxs {
            mctx.set_input_k_idxs_raw(idxs);
        }
        if self.self_attn_mask.raw.is_some() {
            mctx.set_input_attn_mask(&mut self.self_attn_mask, ubatch, self.cparams.causal_attn);
        }
    }
}

/// `llm_graph_input_dsv4`.
#[derive(Default)]
pub struct GraphInputDsv4 {
    pub inp_raw: Option<Rc<RefCell<GraphInputDsv4Raw>>>,
    pub inp_csa: Dsv4CompInput,
    pub inp_hca: Dsv4CompInput,
    pub inp_lid: Dsv4CompInput,
    pub cparams: CParams,
    pub mctx: Option<Rc<dyn MemoryContext>>,
    pub debug: i32,
}

impl GraphInputDsv4 {
    pub fn new(
        cparams: &CParams,
        inp_raw: Rc<RefCell<GraphInputDsv4Raw>>,
        mctx: Rc<dyn MemoryContext>,
    ) -> Self {
        Self {
            inp_raw: Some(inp_raw),
            cparams: cparams.clone(),
            mctx: Some(mctx),
            debug: 0,
            ..Default::default()
        }
    }

    pub fn get_raw(&self) -> Option<Rc<RefCell<GraphInputDsv4Raw>>> {
        self.inp_raw.clone()
    }
    pub fn get_csa(&self) -> &Dsv4CompInput { &self.inp_csa }
    pub fn get_hca(&self) -> &Dsv4CompInput { &self.inp_hca }
    pub fn get_lid(&self) -> &Dsv4CompInput { &self.inp_lid }
}

impl GraphInput for GraphInputDsv4 {
    fn name(&self) -> &'static str {
        "llm_graph_input_dsv4"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(mctx) = self.mctx.clone() else { return };
        let plan_csa = mctx.get_csa_plan(ubatch);
        let plan_hca = mctx.get_hca_plan(ubatch);
        let plan_lid = mctx.get_lid_plan(ubatch);
        let n_stream = plan_csa.n_stream;

        if let Some(raw) = &self.inp_raw {
            if let Some(raw_ctx) = mctx.get_raw() {
                raw.borrow_mut().mctx = Some(raw_ctx);
            }
            raw.borrow_mut().set_input(ubatch);
        }

        set_comp_inputs(&self.inp_csa, &plan_csa, "csa", self.debug > 0, ubatch.n_tokens, n_stream);
        set_comp_inputs(&self.inp_hca, &plan_hca, "hca", self.debug > 0, ubatch.n_tokens, n_stream);
        set_comp_inputs(&self.inp_lid, &plan_lid, "lid", self.debug > 0, ubatch.n_tokens, n_stream);
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = params.mctx.clone() else { return false };
        self.mctx = Some(mctx.clone());
        let plan_csa = mctx.get_csa_plan(&params.ubatch);
        let plan_hca = mctx.get_hca_plan(&params.ubatch);
        let plan_lid = mctx.get_lid_plan(&params.ubatch);
        let n_stream = plan_csa.n_stream;
        let mut res = true;

        if let (Some(raw), Some(raw_ctx)) = (&self.inp_raw, mctx.get_raw()) {
            let mut raw = raw.borrow_mut();
            raw.mctx = Some(raw_ctx.clone());
            if let Some(idxs) = &raw.self_k_idxs {
                res &= idxs.ne(0) == raw_ctx.get_n_kv() as i64;
            }
            if let Some(mask) = &raw.self_attn_mask.raw {
                res &= can_reuse_raw_kq_mask(mask, raw_ctx.as_ref(), &params.ubatch, n_stream);
            }
        } else {
            res = false;
        }

        res &= can_reuse_comp_input(&self.inp_csa, &plan_csa, params.ubatch.n_tokens, n_stream);
        res &= can_reuse_comp_input(&self.inp_hca, &plan_hca, params.ubatch.n_tokens, n_stream);
        res &= can_reuse_comp_input(&self.inp_lid, &plan_lid, params.ubatch.n_tokens, n_stream);
        res
    }
}

/// Keep `Hparams` referenced here so the module documents its dependency on
/// the parsed hyperparameters, matching the C++ header's includes.
pub type Dsv4Hparams = Hparams;

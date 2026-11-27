//! `llm_graph_input_*` classes from `llama-graph.h` / `llama-graph.cpp`.
//!
//! Each input owns the tensors the builder created and knows how to populate
//! them from an `Ubatch`. `can_reuse` rebinds memory contexts and checks that
//! the stored tensor shapes still match.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::rc::Rc;

use crate::arch::Arch;
use crate::batch::Ubatch;
use crate::cparams::{CParams, PoolingType};
use crate::core::{Mt19937, SeqId};
use crate::hparams::{Hparams, RopeSectionOrder, SwaType};

use super::ggml::{new_tensor_2d, new_tensor_4d, Tensor, Type};
use super::support::{get_n_kv_slices, MemoryContext};
use super::types::{AttnMask, Cross, GraphInput, GraphParams};

// ---------------------------------------------------------------------------
// Dedup helpers shared by the attention inputs
// ---------------------------------------------------------------------------

fn mask_type(flash: bool, force_f16: bool) -> Type {
    if flash || force_f16 { Type::F16 } else { Type::F32 }
}

/// `build_attn_inp_kq_mask`.
pub fn build_attn_inp_kq_mask(
    mctx: &dyn MemoryContext,
    ubatch: &Ubatch,
    cparams: &CParams,
    force_f16: bool,
) -> Tensor {
    let n_kv = mctx.get_n_kv() as i64;
    let n_tokens = ubatch.n_tokens as i64;
    let n_stream = if cparams.kv_unified { 1 } else { ubatch.n_seqs_unq as i64 };
    let t = new_tensor_4d(
        mask_type(cparams.flash_attn, force_f16),
        [n_kv, n_tokens / n_stream.max(1), 1, n_stream],
    );
    t.set_input();
    t.set_name("attn_inp_kq_mask");
    t
}

pub fn can_reuse_kq_mask(
    mask: &Tensor,
    mctx: &dyn MemoryContext,
    ubatch: &Ubatch,
    cparams: &CParams,
) -> bool {
    let n_kv = mctx.get_n_kv() as i64;
    let n_tokens = ubatch.n_tokens as i64;
    let n_stream = if cparams.kv_unified { 1 } else { ubatch.n_seqs_unq as i64 };
    mask.ne(0) == n_kv
        && mask.ne(1) == n_tokens / n_stream.max(1)
        && mask.ne(2) == 1
        && mask.ne(3) == n_stream
}

pub fn can_reuse_attn_mask(
    attn_mask: &AttnMask,
    mctx: &dyn MemoryContext,
    ubatch: &Ubatch,
    cparams: &CParams,
) -> bool {
    let n_kv_rows = if cparams.fused_kv_rows { mctx.get_n_kv_rows() } else { 0 };
    let n_slices = get_n_kv_slices(ubatch);
    let n_tokens = ubatch.n_tokens;
    let use_rows = n_kv_rows > 0 && n_slices > 0;

    if use_rows != attn_mask.has_rows() {
        return false;
    }
    if !use_rows {
        return matches!(&attn_mask.raw, Some(raw) if can_reuse_kq_mask(raw, mctx, ubatch, cparams));
    }
    let (Some(raw), Some(rows)) = (&attn_mask.raw, &attn_mask.rows) else { return false };
    attn_mask.n_kv == mctx.get_n_kv()
        && rows.ne(0) == n_kv_rows as i64
        && rows.ne(1) == n_slices as i64
        && raw.ne(0) == n_kv_rows as i64
        && raw.ne(1) == (n_tokens / n_slices) as i64
        && raw.ne(3) == n_slices as i64
}

/// `build_attn_inp_attn_mask`.
pub fn build_attn_inp_attn_mask(
    mctx: &dyn MemoryContext,
    ubatch: &Ubatch,
    cparams: &CParams,
) -> AttnMask {
    let mut attn_mask = AttnMask {
        n_kv: mctx.get_n_kv(),
        ..Default::default()
    };
    let n_kv_rows = if cparams.fused_kv_rows { mctx.get_n_kv_rows() } else { 0 };
    if n_kv_rows == 0 {
        attn_mask.raw = Some(build_attn_inp_kq_mask(mctx, ubatch, cparams, false));
    } else {
        let n_slices = get_n_kv_slices(ubatch);
        let rows = new_tensor_2d(Type::I32, n_kv_rows as i64, n_slices as i64);
        rows.set_input();
        rows.set_name("attn_inp_kv_rows");
        attn_mask.rows = Some(rows);

        let raw = new_tensor_4d(
            Type::F16,
            [n_kv_rows as i64, (ubatch.n_tokens / n_slices.max(1)) as i64, 1, n_slices as i64],
        );
        raw.set_input();
        raw.set_name("attn_inp_kq_mask");
        attn_mask.raw = Some(raw);
    }
    attn_mask
}

/// Number of token rows in a mixed ubatch.
pub fn graph_n_tok_rows(ubatch: &Ubatch) -> i64 {
    if !ubatch.is_mixed() {
        return ubatch.n_tokens as i64;
    }
    ubatch.type_.iter().filter(|&&t| t == 0).count() as i64
}

fn fill_no_cache_mask(
    data: &mut [f32],
    n_tokens: i64,
    n_kv: i64,
    ubatch: &Ubatch,
    n_swa: u32,
    swa_type: SwaType,
    causal: bool,
    group: Option<&[i32]>,
    use_alibi: bool,
) {
    data.fill(-f32::INFINITY);
    for i1 in 0..n_tokens as usize {
        let s1 = ubatch.seq_id.get(i1).and_then(|v| v.first()).copied().unwrap_or(0);
        let p1 = ubatch.pos.get(i1).copied().unwrap_or(0);
        let g1 = group.and_then(|g| g.get(i1)).copied().unwrap_or(0);
        let dst = i1 * n_kv as usize;
        for i0 in 0..n_tokens as usize {
            let s0 = ubatch.seq_id.get(i0).and_then(|v| v.first()).copied().unwrap_or(0);
            let p0 = ubatch.pos.get(i0).copied().unwrap_or(0);
            let g0 = group.and_then(|g| g.get(i0)).copied().unwrap_or(0);
            if s0 != s1 || (g0 != 0 && g1 != 0 && g0 != g1) || (causal && p0 > p1) {
                continue;
            }
            if Hparams::is_masked_swa(n_swa, swa_type, p0, p1) {
                continue;
            }
            if dst + i0 < data.len() {
                data[dst + i0] = if use_alibi { -(p0 - p1).abs() as f32 } else { 0.0 };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Token / embedding inputs
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct GraphInputEmbd {
    pub tokens: Option<Tensor>,
    pub embd: Option<Tensor>,
    pub mixed_tokens: Option<Tensor>,
    pub mixed_slots: Option<Tensor>,
    pub mixed_embd: Option<Tensor>,
    pub scale_rows: Option<Tensor>,
    pub embd_rows: Option<Tensor>,
    pub scale_tok: f32,
    pub n_embd: i64,
}

impl GraphInputEmbd {
    pub fn new(n_embd: i64) -> Self {
        Self { n_embd, scale_tok: 1.0, ..Default::default() }
    }
}

impl GraphInput for GraphInputEmbd {
    fn name(&self) -> &'static str {
        "llm_graph_input_embd"
    }

    fn set_input(&mut self, ubatch: &Ubatch) {
        if !ubatch.token.is_empty() {
            if let Some(t) = &self.tokens {
                t.set_data_i32(&ubatch.token);
            }
        }
        if !ubatch.embd_row.is_empty() {
            if let Some(t) = &self.embd_rows {
                t.set_data_i32(&ubatch.embd_row);
            }
        }
        if !ubatch.embd.is_empty() {
            if let Some(t) = &self.embd {
                if !ubatch.is_mixed() {
                    t.set_data_f32(&ubatch.embd);
                }
            }
        }
        if ubatch.is_mixed() {
            if let (Some(mixed_tokens), Some(mixed_slots), Some(mixed_embd)) =
                (&self.mixed_tokens, &self.mixed_slots, &self.mixed_embd)
            {
                let mut ids = Vec::new();
                let mut slots = Vec::new();
                for (i, kind) in ubatch.type_.iter().enumerate() {
                    if *kind == 0 {
                        ids.push(ubatch.token.get(i).copied().unwrap_or(0));
                        slots.push(i as i64);
                    }
                }
                mixed_tokens.set_data_i32(&ids);
                mixed_slots.set_data_i64(&slots);
                mixed_embd.set_data_f32(&ubatch.embd);
            }
        }
        if let Some(scale_rows) = &self.scale_rows {
            let n = ubatch.n_tokens as usize;
            let data: Vec<f32> = (0..n)
                .map(|i| {
                    let is_embd = ubatch.token.is_empty()
                        || (ubatch.is_mixed() && ubatch.type_.get(i).copied().unwrap_or(0) != 0);
                    if is_embd { 1.0 } else { self.scale_tok }
                })
                .collect();
            scale_rows.set_data_f32(&data);
        }
    }

    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let u = &params.ubatch;
        let mut res = true;
        res &= u.token.is_empty() || matches!(&self.tokens, Some(t) if t.ne(0) == u.n_tokens as i64);
        res &= u.embd.is_empty() || matches!(&self.embd, Some(t) if t.ne(1) == u.n_tokens as i64);
        res &= matches!(&self.mixed_tokens, None)
            || matches!(&self.mixed_tokens, Some(t) if t.ne(0) == graph_n_tok_rows(u));
        res &= matches!(&self.mixed_embd, None)
            || matches!(&self.mixed_embd, Some(t) if t.ne(1) == u.n_tokens as i64);
        res &= matches!(&self.scale_rows, None)
            || matches!(&self.scale_rows, Some(t) if t.ne(1) == u.n_tokens as i64);
        res &= u.embd_row.is_empty()
            || matches!(&self.embd_rows, Some(t) if t.ne(0) == u.n_tokens as i64);
        res
    }
}

#[derive(Default)]
pub struct GraphInputEmbdH {
    pub tokens: Option<Tensor>,
    pub embd: Option<Tensor>,
    pub h: Option<Tensor>,
    pub h_rows: Option<Tensor>,
    pub n_embd: i64,
    pub n_embd_state: i64,
}

impl GraphInputEmbdH {
    pub fn new(n_embd: i64, n_embd_state: i64) -> Self {
        Self { n_embd, n_embd_state, ..Default::default() }
    }
}

impl GraphInput for GraphInputEmbdH {
    fn name(&self) -> &'static str {
        "llm_graph_input_embd_h"
    }

    fn set_input(&mut self, ubatch: &Ubatch) {
        let n_tokens = ubatch.n_tokens as usize;
        if !ubatch.token.is_empty() {
            if let Some(t) = &self.tokens {
                t.set_data_i32(&ubatch.token[..n_tokens.min(ubatch.token.len())]);
            }
        } else if let Some(embd) = &self.embd {
            let n = (n_tokens * self.n_embd as usize).min(ubatch.embd.len());
            embd.set_data_f32(&ubatch.embd[..n]);
        }
        if let Some(h_rows) = &self.h_rows {
            let rows = &ubatch.embd_state_row[..n_tokens.min(ubatch.embd_state_row.len())];
            h_rows.set_data_i32(rows);
            return;
        }
        if let Some(h) = &self.h {
            let n = (n_tokens * self.n_embd_state as usize).min(ubatch.embd_state.len());
            h.set_data_f32(&ubatch.embd_state[..n]);
        }
    }

    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let u = &params.ubatch;
        let mut res = true;
        res &= u.token.is_empty() || matches!(&self.tokens, Some(t) if t.ne(0) == u.n_tokens as i64);
        res &= u.embd.is_empty() || matches!(&self.embd, Some(t) if t.ne(1) == u.n_tokens as i64);
        res &= u.embd_state.is_empty()
            || matches!(&self.h, Some(t) if t.ne(1) == u.n_tokens as i64);
        let has_rows = !u.embd_state_row.is_empty();
        res &= has_rows == self.h_rows.is_some();
        res &= !has_rows
            || matches!(&self.h_rows, Some(t) if t.ne(0) == u.n_tokens as i64);
        res
    }
}

#[derive(Default)]
pub struct GraphInputBatchIdxs {
    pub idxs: Option<Tensor>,
}

impl GraphInput for GraphInputBatchIdxs {
    fn name(&self) -> &'static str {
        "llm_graph_input_batch_idxs"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        if let Some(idxs) = &self.idxs {
            let n = ubatch.n_tokens as usize;
            let data: Vec<i64> = (0..n)
                .map(|i| ubatch.batch_idxs.get(i).copied().unwrap_or(i) as i64)
                .collect();
            idxs.set_data_i64(&data);
        }
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        matches!(&self.idxs, Some(t) if t.ne(0) == params.ubatch.n_tokens as i64)
    }
}

// ---------------------------------------------------------------------------
// Position / temperature inputs
// ---------------------------------------------------------------------------

pub struct GraphInputPos {
    pub pos: Option<Tensor>,
    pub n_pos_per_embd: u32,
    pub rope_section_order: RopeSectionOrder,
}

impl Default for GraphInputPos {
    fn default() -> Self {
        Self { pos: None, n_pos_per_embd: 1, rope_section_order: RopeSectionOrder::Tyxz }
    }
}

impl GraphInputPos {
    pub fn new(n_pos_per_embd: u32, rope_section_order: RopeSectionOrder) -> Self {
        Self { pos: None, n_pos_per_embd, rope_section_order }
    }
}

impl GraphInput for GraphInputPos {
    fn name(&self) -> &'static str {
        "llm_graph_input_pos"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let (Some(pos), true) = (&self.pos, !ubatch.pos.is_empty()) else { return };
        let n_tokens = ubatch.n_tokens as usize;
        let has_embd = ubatch.is_mixed() || ubatch.token.is_empty();
        if self.rope_section_order == RopeSectionOrder::Tyxz || !has_embd {
            let n = (n_tokens * self.n_pos_per_embd as usize).min(ubatch.pos.len());
            pos.set_data_i32(&ubatch.pos[..n]);
            return;
        }
        let slot_of_section: [usize; 4] = match self.rope_section_order {
            RopeSectionOrder::Zyxt => [3, 1, 2, 0],
            RopeSectionOrder::Unspecified | RopeSectionOrder::Tyxz => [0, 1, 2, 3],
        };
        let mut data = vec![0i32; n_tokens * self.n_pos_per_embd as usize];
        for i in 0..n_tokens {
            let is_embd = if ubatch.is_mixed() {
                ubatch.type_.get(i).copied().unwrap_or(0) != 0
            } else {
                true
            };
            for s in 0..4 {
                let slot = if is_embd { slot_of_section[s] } else { s };
                let value = ubatch.pos.get(slot * n_tokens + i).copied().unwrap_or(0);
                data[s * n_tokens + i] = value;
            }
        }
        pos.set_data_i32(&data);
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        matches!(&self.pos, Some(t) if t.ne(0) == params.ubatch.n_tokens as i64 * self.n_pos_per_embd as i64)
    }
}

#[derive(Default)]
pub struct GraphInputAttnTemp {
    pub attn_scale: Option<Tensor>,
    pub n_attn_temp_floor_scale: u32,
    pub f_attn_temp_scale: f32,
    pub f_attn_temp_offset: f32,
    pub attn_temp_pow: bool,
}

impl GraphInputAttnTemp {
    pub fn new(
        n_attn_temp_floor_scale: u32,
        f_attn_temp_scale: f32,
        f_attn_temp_offset: f32,
        attn_temp_pow: bool,
    ) -> Self {
        Self {
            attn_scale: None,
            n_attn_temp_floor_scale,
            f_attn_temp_scale,
            f_attn_temp_offset,
            attn_temp_pow,
        }
    }
}

impl GraphInput for GraphInputAttnTemp {
    fn name(&self) -> &'static str {
        "llm_graph_input_attn_temp"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let (Some(attn_scale), true) = (&self.attn_scale, !ubatch.pos.is_empty()) else { return };
        let n_tokens = ubatch.n_tokens as usize;
        let mut data = vec![0.0f32; n_tokens];
        if self.attn_temp_pow {
            for (i, pos) in data.iter_mut().enumerate().take(n_tokens) {
                let p = ubatch.pos.get(i).copied().unwrap_or(0) as f64;
                *pos = (self.f_attn_temp_scale as f64
                    * (1.0 + (p + self.f_attn_temp_offset as f64) / self.n_attn_temp_floor_scale as f64).ln_1p())
                    as f32;
            }
        } else {
            for (i, pos) in data.iter_mut().enumerate().take(n_tokens) {
                let p = ubatch.pos.get(i).copied().unwrap_or(0) as f32;
                *pos = (((p + self.f_attn_temp_offset) / self.n_attn_temp_floor_scale as f32)
                    .floor()
                    + 1.0)
                    .ln()
                    * self.f_attn_temp_scale
                    + 1.0;
            }
        }
        attn_scale.set_data_f32(&data);
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        self.attn_temp_pow
            && matches!(&self.attn_scale, Some(t) if t.ne(2) == params.ubatch.n_tokens as i64)
    }
}

#[derive(Default)]
pub struct GraphInputPosBucket {
    pub pos_bucket: Option<Tensor>,
    pub n_rel_attn_bkts: u32,
}

impl GraphInputPosBucket {
    pub fn new(hparams: &Hparams) -> Self {
        Self { pos_bucket: None, n_rel_attn_bkts: hparams.n_rel_attn_bkts }
    }
}

impl GraphInput for GraphInputPosBucket {
    fn name(&self) -> &'static str {
        "llm_graph_input_pos_bucket"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        if let Some(pos_bucket) = &self.pos_bucket {
            let n = ubatch.n_tokens as usize;
            let mut data = vec![0i32; n * n];
            for j in 0..n {
                for i in 0..n {
                    let x = ubatch.pos.get(i).copied().unwrap_or(0);
                    let y = ubatch.pos.get(j).copied().unwrap_or(0);
                    data[j * n + i] =
                        super::output::relative_position_bucket(x, y, self.n_rel_attn_bkts as u64, true);
                }
            }
            pos_bucket.set_data_i32(&data);
        }
    }
}

#[derive(Clone)]
pub struct GraphInputPosBucketKv {
    pub pos_bucket: Option<Tensor>,
    pub hparams: Hparams,
    pub mctx: Rc<dyn MemoryContext>,
}

impl GraphInputPosBucketKv {
    pub fn new(hparams: &Hparams, mctx: Rc<dyn MemoryContext>) -> Self {
        Self { pos_bucket: None, hparams: hparams.clone(), mctx }
    }
}

impl GraphInput for GraphInputPosBucketKv {
    fn name(&self) -> &'static str {
        "llm_graph_input_pos_bucket_kv"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        if let Some(pos_bucket) = &self.pos_bucket {
            self.mctx.set_input_pos_bucket(pos_bucket, ubatch);
        }
    }
}

#[derive(Clone)]
pub struct GraphInputOutIds {
    pub out_ids: Option<Tensor>,
    pub hparams: Hparams,
    pub cparams: CParams,
    pub n_outputs: u32,
}

impl GraphInputOutIds {
    pub fn new(hparams: &Hparams, cparams: &CParams, n_outputs: u32) -> Self {
        Self {
            out_ids: None,
            hparams: hparams.clone(),
            cparams: cparams.clone(),
            n_outputs,
        }
    }
}

impl GraphInput for GraphInputOutIds {
    fn name(&self) -> &'static str {
        "llm_graph_input_out_ids"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(out_ids) = &self.out_ids else { return };
        let n_tokens = ubatch.n_tokens as usize;
        if self.n_outputs as usize == n_tokens {
            let data: Vec<i32> = (0..n_tokens as i32).collect();
            out_ids.set_data_i32(&data);
            return;
        }
        let mut data = Vec::new();
        for (i, flag) in ubatch.output.iter().enumerate().take(n_tokens) {
            if *flag != 0 {
                data.push(i as i32);
            }
        }
        out_ids.set_data_i32(&data);
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        self.n_outputs == params.n_outputs
    }
}

#[derive(Default)]
pub struct GraphInputMoeIds {
    pub ids: Option<Tensor>,
    pub n_expert: usize,
}

impl GraphInputMoeIds {
    pub fn new(n_expert: usize) -> Self {
        Self { ids: None, n_expert }
    }
}

impl GraphInput for GraphInputMoeIds {
    fn name(&self) -> &'static str {
        "llm_graph_input_moe_ids"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(ids) = &self.ids else { return };
        let n_used = ids.ne(0).max(0) as usize;
        let n_tokens = ids.ne(1).max(0) as usize;
        let seed = ubatch.pos.first().copied().unwrap_or(0) as u32;
        let mut rng = Mt19937::new(seed);
        let mut perm: Vec<usize> = (0..self.n_expert).collect();
        let mut data: Vec<i32> = ids.data_i32();
        if data.len() < n_used * n_tokens {
            data.resize(n_used * n_tokens, 0);
        }
        for row in 0..n_tokens {
            for j in 0..n_used {
                let swap = j + (rng.next_u32() as usize % (self.n_expert - j));
                perm.swap(j, swap);
                data[row * n_used + j] = perm[j] as i32;
            }
        }
        ids.set_data_i32(&data);
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        matches!(&self.ids, Some(t) if t.ne(1) == params.ubatch.n_tokens as i64)
    }
}

fn mean_pooling(cparams: &CParams) -> bool {
    cparams.embeddings
        && matches!(cparams.pooling_type, PoolingType::Mean | PoolingType::Rank)
}

fn cls_pooling(cparams: &CParams) -> bool {
    cparams.embeddings
        && matches!(
            cparams.pooling_type,
            PoolingType::Cls | PoolingType::Rank | PoolingType::Last
        )
}

#[derive(Clone)]
pub struct GraphInputMean {
    pub mean: Option<Tensor>,
    pub cparams: CParams,
}

impl GraphInputMean {
    pub fn new(cparams: &CParams) -> Self {
        Self { mean: None, cparams: cparams.clone() }
    }
}

impl GraphInput for GraphInputMean {
    fn name(&self) -> &'static str {
        "llm_graph_input_mean"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        if !mean_pooling(&self.cparams) {
            return;
        }
        let Some(mean) = &self.mean else { return };
        let n_tokens = ubatch.n_tokens as usize;
        let n_seq_tokens = ubatch.n_seq_tokens as usize;
        let n_seqs_unq = ubatch.n_seqs_unq as usize;
        let mut data = vec![0.0f32; n_tokens * n_seqs_unq];
        let mut sums = vec![0u64; n_seqs_unq];
        let mut i = 0;
        while i < n_tokens {
            for seq_id in ubatch.seq_id.get(i).into_iter().flatten() {
                let seq_idx = ubatch.seq_idx.get(*seq_id as usize).copied().unwrap_or(0) as usize;
                if seq_idx < n_seqs_unq {
                    sums[seq_idx] += n_seq_tokens as u64;
                }
            }
            i += n_seq_tokens.max(1);
        }
        let div: Vec<f32> = sums.iter().map(|&s| if s > 0 { 1.0 / s as f32 } else { 0.0 }).collect();
        let mut i = 0;
        while i < n_tokens {
            for seq_id in ubatch.seq_id.get(i).into_iter().flatten() {
                let seq_idx = ubatch.seq_idx.get(*seq_id as usize).copied().unwrap_or(0) as usize;
                if seq_idx >= n_seqs_unq {
                    continue;
                }
                for j in 0..n_seq_tokens {
                    if i + j < n_tokens {
                        data[seq_idx * n_tokens + i + j] = div[seq_idx];
                    }
                }
            }
            i += n_seq_tokens.max(1);
        }
        mean.set_data_f32(&data);
    }
}

#[derive(Clone)]
pub struct GraphInputCls {
    pub cls: Option<Tensor>,
    pub cparams: CParams,
    pub arch: Arch,
}

impl GraphInputCls {
    pub fn new(cparams: &CParams, arch: Arch) -> Self {
        Self { cls: None, cparams: cparams.clone(), arch }
    }
}

impl GraphInput for GraphInputCls {
    fn name(&self) -> &'static str {
        "llm_graph_input_cls"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        if !cls_pooling(&self.cparams) {
            return;
        }
        let Some(cls) = &self.cls else { return };
        let n_tokens = ubatch.n_tokens as usize;
        let n_seqs_unq = ubatch.n_seqs_unq as usize;
        let mut data = vec![0u32; n_seqs_unq];
        let mut target_pos = vec![-1i32; n_seqs_unq];
        let mut target_row = vec![-1i32; n_seqs_unq];
        let last = self.cparams.pooling_type == PoolingType::Last
            || (self.cparams.pooling_type == PoolingType::Rank && self.cparams.causal_attn);
        for i in 0..n_tokens {
            let pos = ubatch.pos.get(i).copied().unwrap_or(0);
            for seq_id in ubatch.seq_id.get(i).into_iter().flatten() {
                let seq_idx = ubatch.seq_idx.get(*seq_id as usize).copied().unwrap_or(0) as usize;
                if seq_idx >= n_seqs_unq {
                    continue;
                }
                if target_pos[seq_idx] == -1
                    || (last && pos >= target_pos[seq_idx])
                    || (!last && pos < target_pos[seq_idx])
                {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i as i32;
                }
            }
        }
        for s in 0..n_seqs_unq {
            if target_row[s] >= 0 {
                data[s] = target_row[s] as u32;
            }
        }
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        cls.set_data_raw(&bytes);
    }
}

// ---------------------------------------------------------------------------
// Recurrent state input
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct GraphInputRs {
    pub s_copy: Option<Tensor>,
    pub s_copy_main: Option<Tensor>,
    pub s_copy_tail: Option<Tensor>,
    pub has_rep: bool,
    pub n_conv: i64,
    pub n_pend: i64,
    pub rep_n: Option<Tensor>,
    pub rep_cols: Option<Tensor>,
    pub rep_cols_r: Option<Tensor>,
    pub rep_cols_p: Option<Tensor>,
    pub mctx: Option<Rc<dyn MemoryContext>>,
    pub head: u32,
    pub rs_z: i32,
    pub rs_in_place: bool,
    pub s_copy_rows: Option<Tensor>,
}

impl GraphInputRs {
    pub fn new(mctx: Rc<dyn MemoryContext>) -> Self {
        Self { mctx: Some(mctx), ..Default::default() }
    }

    fn set_input_rep(&mut self, mctx: &dyn MemoryContext, ubatch: &Ubatch) {
        if !self.has_rep {
            return;
        }
        let n_seqs = ubatch.n_seqs as usize;
        let (Some(rep_n), Some(rep_cols)) = (&self.rep_n, &self.rep_cols) else { return };
        let n_conv = self.n_conv.max(0) as usize;
        let mut reps = vec![0i32; n_seqs];
        let mut cols = vec![0i32; n_seqs * n_conv];
        for s in 0..n_seqs {
            let n_rep = mctx.get_rs_rep(s);
            reps[s] = n_rep as i32;
            for c in 0..n_conv {
                cols[s * n_conv + c] = (n_rep + c) as i32;
            }
        }
        rep_n.set_data_i32(&reps);
        rep_cols.set_data_i32(&cols);
    }
}

impl GraphInput for GraphInputRs {
    fn name(&self) -> &'static str {
        "llm_graph_input_rs"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(mctx) = self.mctx.clone() else { return };
        if let Some(s_copy) = &self.s_copy {
            let n_rs = mctx.get_n_rs();
            let data: Vec<i32> = (0..n_rs).map(|i| mctx.s_copy(i)).collect();
            s_copy.set_data_i32(&data);
        }
        self.set_input_rep(mctx.as_ref(), ubatch);
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = params.mctx.clone() else { return false };
        self.mctx = Some(mctx.clone());
        let mut res = true;
        res &= matches!(&self.s_copy, Some(t) if t.ne(0) == mctx.get_n_rs() as i64);
        res &= matches!(&self.s_copy_main, Some(t) if t.ne(0) == params.ubatch.n_seqs as i64);
        res &= self.head == mctx.get_head();
        res &= self.rs_z == mctx.get_rs_z();
        res &= self.rs_in_place == mctx.get_rs_in_place();
        res
    }
}

#[derive(Clone)]
pub struct GraphInputCrossEmbd {
    pub cross_embd: Option<Tensor>,
    pub cross: Rc<Cross>,
}

impl GraphInputCrossEmbd {
    pub fn new(cross: Rc<Cross>) -> Self {
        Self { cross_embd: None, cross }
    }
}

impl GraphInput for GraphInputCrossEmbd {
    fn name(&self) -> &'static str {
        "llm_graph_input_cross_embd"
    }
    fn set_input(&mut self, _ubatch: &Ubatch) {
        if let Some(cross_embd) = &self.cross_embd {
            if !self.cross.v_embd.is_empty() {
                cross_embd.set_data_f32(&self.cross.v_embd);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Attention inputs
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct GraphInputAttnNoCache {
    pub self_attn_mask: AttnMask,
    pub self_attn_mask_swa: AttnMask,
    pub hparams: Hparams,
    pub cparams: CParams,
    pub debug: i32,
}

impl GraphInputAttnNoCache {
    pub fn new(hparams: &Hparams, cparams: &CParams) -> Self {
        Self {
            self_attn_mask: AttnMask::default(),
            self_attn_mask_swa: AttnMask::default(),
            hparams: hparams.clone(),
            cparams: cparams.clone(),
            debug: 0,
        }
    }
}

impl GraphInput for GraphInputAttnNoCache {
    fn name(&self) -> &'static str {
        "llm_graph_input_attn_no_cache"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(raw) = self.self_attn_mask.raw.as_ref() else { return };
        let n_kv = raw.ne(0);
        let n_tokens = ubatch.n_tokens as i64;
        let causal = self.cparams.causal_attn;
        let causal_swa = causal
            || self.hparams.non_causal_type == crate::hparams::NonCausalType::DenseOnly;

        if let Some(data) = self.self_attn_mask.raw.as_ref() {
            let n = data.nelements();
            let mut buf = vec![0.0f32; n];
            fill_no_cache_mask(
                &mut buf, n_tokens, n_kv, ubatch, 0, SwaType::None, causal,
                Some(&ubatch.decision_group), self.hparams.use_alibi,
            );
            data.set_data_f32(&buf);
        }

        if self.hparams.swa_type != SwaType::None {
            if let Some(swa) = self.self_attn_mask_swa.raw.as_ref() {
                let n = swa.nelements();
                let mut buf = vec![0.0f32; n];
                fill_no_cache_mask(
                    &mut buf, n_tokens, n_kv, ubatch, self.hparams.n_swa,
                    self.hparams.swa_type, causal_swa, None, self.hparams.use_alibi,
                );
                swa.set_data_f32(&buf);
            }
        }
    }
}

#[derive(Clone)]
pub struct GraphInputAttnKv {
    pub self_k_idxs: Option<Tensor>,
    pub self_v_idxs: Option<Tensor>,
    pub self_attn_mask: AttnMask,
    pub self_k_rot: Vec<Option<Tensor>>,
    pub self_v_rot: Vec<Option<Tensor>>,
    pub hparams: Hparams,
    pub cparams: CParams,
    pub mctx: Option<Rc<dyn MemoryContext>>,
}

impl GraphInputAttnKv {
    pub fn new(hparams: &Hparams, cparams: &CParams, mctx: Rc<dyn MemoryContext>) -> Self {
        Self {
            self_k_idxs: None,
            self_v_idxs: None,
            self_attn_mask: AttnMask::default(),
            self_k_rot: Vec::new(),
            self_v_rot: Vec::new(),
            hparams: hparams.clone(),
            cparams: cparams.clone(),
            mctx: Some(mctx),
        }
    }

    pub fn get_k_idxs(&self) -> Option<Tensor> {
        self.self_k_idxs.clone()
    }
    pub fn get_v_idxs(&self) -> Option<Tensor> {
        self.self_v_idxs.clone()
    }
}

impl GraphInput for GraphInputAttnKv {
    fn name(&self) -> &'static str {
        "llm_graph_input_attn_kv"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(mctx) = self.mctx.clone() else { return };
        if let Some(idxs) = &self.self_k_idxs {
            mctx.set_input_k_idxs(idxs, ubatch);
        }
        if let Some(idxs) = &self.self_v_idxs {
            mctx.set_input_v_idxs(idxs, ubatch);
        }
        if self.self_attn_mask.raw.is_some() {
            mctx.set_input_attn_mask(&mut self.self_attn_mask, ubatch, self.cparams.causal_attn);
        }
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = params.mctx.clone() else { return false };
        self.mctx = Some(mctx.clone());
        let mut res = true;
        res &= matches!(&self.self_k_idxs, Some(t) if t.ne(0) == params.ubatch.n_tokens as i64);
        res &= can_reuse_attn_mask(&self.self_attn_mask, mctx.as_ref(), &params.ubatch, &params.cparams);
        res
    }
}

#[derive(Clone)]
pub struct GraphInputAttnK {
    pub self_k_idxs: Option<Tensor>,
    pub self_attn_mask: AttnMask,
    pub hparams: Hparams,
    pub cparams: CParams,
    pub mctx: Option<Rc<dyn MemoryContext>>,
}

impl GraphInputAttnK {
    pub fn new(hparams: &Hparams, cparams: &CParams, mctx: Rc<dyn MemoryContext>) -> Self {
        Self {
            self_k_idxs: None,
            self_attn_mask: AttnMask::default(),
            hparams: hparams.clone(),
            cparams: cparams.clone(),
            mctx: Some(mctx),
        }
    }
    pub fn get_k_idxs(&self) -> Option<Tensor> {
        self.self_k_idxs.clone()
    }

    /// `can_reuse_impl`: reuses the currently bound memory context.
    pub fn can_reuse_impl(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = self.mctx.clone() else { return false };
        let mut res = true;
        res &= matches!(&self.self_k_idxs, Some(t) if t.ne(0) == params.ubatch.n_tokens as i64);
        res &= can_reuse_attn_mask(&self.self_attn_mask, mctx.as_ref(), &params.ubatch, &params.cparams);
        res
    }
}

impl GraphInput for GraphInputAttnK {
    fn name(&self) -> &'static str {
        "llm_graph_input_attn_k"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(mctx) = self.mctx.clone() else { return };
        if let Some(idxs) = &self.self_k_idxs {
            mctx.set_input_k_idxs(idxs, ubatch);
        }
        mctx.set_input_attn_mask(&mut self.self_attn_mask, ubatch, self.cparams.causal_attn);
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        if let Some(mctx) = params.mctx.clone() {
            self.mctx = Some(mctx);
        }
        self.can_reuse_impl(params)
    }
}

#[derive(Clone)]
pub struct GraphInputAttnKDsa {
    pub self_k_idxs_mla: Option<Tensor>,
    pub self_k_idxs_lid: Option<Tensor>,
    pub self_attn_mask_mla: AttnMask,
    pub self_attn_mask_lid: AttnMask,
    pub self_k_rot_lid: Vec<Option<Tensor>>,
    pub hparams: Hparams,
    pub cparams: CParams,
    pub mctx: Option<Rc<dyn MemoryContext>>,
}

impl GraphInputAttnKDsa {
    pub fn new(hparams: &Hparams, cparams: &CParams, mctx: Rc<dyn MemoryContext>) -> Self {
        Self {
            self_k_idxs_mla: None,
            self_k_idxs_lid: None,
            self_attn_mask_mla: AttnMask::default(),
            self_attn_mask_lid: AttnMask::default(),
            self_k_rot_lid: Vec::new(),
            hparams: hparams.clone(),
            cparams: cparams.clone(),
            mctx: Some(mctx),
        }
    }
    pub fn get_k_idxs_mla(&self) -> Option<Tensor> {
        self.self_k_idxs_mla.clone()
    }
    pub fn get_k_idxs_lid(&self) -> Option<Tensor> {
        self.self_k_idxs_lid.clone()
    }
    pub fn can_reuse_impl(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = self.mctx.clone() else { return false };
        let Some(mla) = mctx.get_mla() else { return false };
        let Some(lid) = mctx.get_lid() else { return false };
        let mut res = true;
        res &= matches!(&self.self_k_idxs_mla, Some(t) if t.ne(0) == params.ubatch.n_tokens as i64);
        res &= matches!(&self.self_k_idxs_lid, Some(t) if t.ne(0) == params.ubatch.n_tokens as i64);
        res &= can_reuse_attn_mask(&self.self_attn_mask_mla, mla.as_ref(), &params.ubatch, &params.cparams);
        res &= can_reuse_attn_mask(&self.self_attn_mask_lid, lid.as_ref(), &params.ubatch, &params.cparams);
        res
    }
}

impl GraphInput for GraphInputAttnKDsa {
    fn name(&self) -> &'static str {
        "llm_graph_input_attn_k_dsa"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(mctx) = self.mctx.clone() else { return };
        let (Some(mla), Some(lid)) = (mctx.get_mla(), mctx.get_lid()) else { return };
        if let Some(idxs) = &self.self_k_idxs_mla {
            mla.set_input_k_idxs(idxs, ubatch);
        }
        if self.self_attn_mask_mla.raw.as_ref().is_some() {
            mla.set_input_attn_mask(&mut self.self_attn_mask_mla, ubatch, self.cparams.causal_attn);
        }
        if let Some(idxs) = &self.self_k_idxs_lid {
            lid.set_input_k_idxs(idxs, ubatch);
        }
        lid.set_input_attn_mask(&mut self.self_attn_mask_lid, ubatch, self.cparams.causal_attn);
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        if let Some(mctx) = params.mctx.clone() {
            self.mctx = Some(mctx);
        }
        self.can_reuse_impl(params)
    }
}

#[derive(Default)]
pub struct GraphInputAttnKDsaIswa {
    pub inp_dsa: Option<Rc<RefCell<GraphInputAttnKDsa>>>,
    pub inp_swa: Option<Rc<RefCell<GraphInputAttnK>>>,
    pub mctx: Option<Rc<dyn MemoryContext>>,
}

impl GraphInputAttnKDsaIswa {
    pub fn new(
        inp_dsa: Rc<RefCell<GraphInputAttnKDsa>>,
        inp_swa: Rc<RefCell<GraphInputAttnK>>,
        mctx: Rc<dyn MemoryContext>,
    ) -> Self {
        Self { inp_dsa: Some(inp_dsa), inp_swa: Some(inp_swa), mctx: Some(mctx) }
    }
}

impl GraphInput for GraphInputAttnKDsaIswa {
    fn name(&self) -> &'static str {
        "llm_graph_input_attn_k_dsa_iswa"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        if let Some(inp) = &self.inp_dsa {
            inp.borrow_mut().set_input(ubatch);
        }
        if let Some(inp) = &self.inp_swa {
            inp.borrow_mut().set_input(ubatch);
        }
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = params.mctx.clone() else { return false };
        self.mctx = Some(mctx.clone());
        let mut res = true;
        if let (Some(dsa), Some(swa), Some(m), Some(s)) = (
            &self.inp_dsa, &self.inp_swa, mctx.get_dsa(), mctx.get_swa(),
        ) {
            dsa.borrow_mut().mctx = Some(m);
            swa.borrow_mut().mctx = Some(s);
            res &= dsa.borrow_mut().can_reuse_impl(params);
            res &= swa.borrow_mut().can_reuse_impl(params);
        } else {
            res = false;
        }
        res
    }
}

pub struct GraphInputAttnKvMsa {
    pub base: GraphInputAttnKv,
    pub self_k_idxs_idx: Option<Tensor>,
    pub mctx_msa: Option<Rc<dyn MemoryContext>>,
}

impl GraphInputAttnKvMsa {
    pub fn new(
        hparams: &Hparams,
        cparams: &CParams,
        mctx: Rc<dyn MemoryContext>,
    ) -> Self {
        let Some(base_ctx) = mctx.get_base() else {
            return Self { base: GraphInputAttnKv::new(hparams, cparams, mctx.clone()), self_k_idxs_idx: None, mctx_msa: Some(mctx) };
        };
        Self {
            base: GraphInputAttnKv::new(hparams, cparams, base_ctx),
            self_k_idxs_idx: None,
            mctx_msa: Some(mctx),
        }
    }
    pub fn get_k_idxs_idx(&self) -> Option<Tensor> {
        self.self_k_idxs_idx.clone()
    }
}

impl GraphInput for GraphInputAttnKvMsa {
    fn name(&self) -> &'static str {
        "llm_graph_input_attn_kv_msa"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        self.base.set_input(ubatch);
        if let (Some(idxs), Some(mctx)) = (&self.self_k_idxs_idx, &self.mctx_msa) {
            if let Some(idx_ctx) = mctx.get_idx() {
                idx_ctx.set_input_k_idxs(idxs, ubatch);
            }
        }
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = params.mctx.clone() else { return false };
        self.mctx_msa = Some(mctx.clone());
        if let Some(base) = mctx.get_base() {
            self.base.mctx = Some(base);
        }
        let mut res = true;
        res &= matches!(&self.base.self_k_idxs, Some(t) if t.ne(0) == params.ubatch.n_tokens as i64);
        if let Some(idxs) = &self.self_k_idxs_idx {
            res &= idxs.ne(0) == params.ubatch.n_tokens as i64;
        }
        let Some(base) = self.base.mctx.clone() else { return false };
        res &= can_reuse_attn_mask(&self.base.self_attn_mask, base.as_ref(), &params.ubatch, &params.cparams);
        res
    }
}

#[derive(Clone)]
pub struct GraphInputAttnKvIswa {
    pub self_k_idxs: Option<Tensor>,
    pub self_v_idxs: Option<Tensor>,
    pub self_k_idxs_swa: Option<Tensor>,
    pub self_v_idxs_swa: Option<Tensor>,
    pub self_attn_mask: AttnMask,
    pub self_attn_mask_swa: AttnMask,
    pub self_k_rot: Vec<Option<Tensor>>,
    pub self_v_rot: Vec<Option<Tensor>>,
    pub self_k_rot_swa: Vec<Option<Tensor>>,
    pub self_v_rot_swa: Vec<Option<Tensor>>,
    pub hparams: Hparams,
    pub cparams: CParams,
    pub mctx: Option<Rc<dyn MemoryContext>>,
}

impl GraphInputAttnKvIswa {
    pub fn new(hparams: &Hparams, cparams: &CParams, mctx: Rc<dyn MemoryContext>) -> Self {
        Self {
            self_k_idxs: None, self_v_idxs: None,
            self_k_idxs_swa: None, self_v_idxs_swa: None,
            self_attn_mask: AttnMask::default(), self_attn_mask_swa: AttnMask::default(),
            self_k_rot: Vec::new(), self_v_rot: Vec::new(),
            self_k_rot_swa: Vec::new(), self_v_rot_swa: Vec::new(),
            hparams: hparams.clone(), cparams: cparams.clone(), mctx: Some(mctx),
        }
    }
    pub fn get_k_idxs(&self) -> Option<Tensor> { self.self_k_idxs.clone() }
    pub fn get_v_idxs(&self) -> Option<Tensor> { self.self_v_idxs.clone() }
    pub fn get_k_idxs_swa(&self) -> Option<Tensor> { self.self_k_idxs_swa.clone() }
    pub fn get_v_idxs_swa(&self) -> Option<Tensor> { self.self_v_idxs_swa.clone() }
}

impl GraphInput for GraphInputAttnKvIswa {
    fn name(&self) -> &'static str {
        "llm_graph_input_attn_kv_iswa"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(mctx) = self.mctx.clone() else { return };
        let (Some(base), Some(swa)) = (mctx.get_base(), mctx.get_swa()) else { return };
        if let Some(idxs) = &self.self_k_idxs {
            base.set_input_k_idxs(idxs, ubatch);
            if let Some(v) = &self.self_v_idxs {
                base.set_input_v_idxs(v, ubatch);
            }
        }
        base.set_input_attn_mask(&mut self.self_attn_mask, ubatch, self.cparams.causal_attn);
        if let Some(idxs) = &self.self_k_idxs_swa {
            swa.set_input_k_idxs(idxs, ubatch);
            if let Some(v) = &self.self_v_idxs_swa {
                swa.set_input_v_idxs(v, ubatch);
            }
        }
        swa.set_input_attn_mask(&mut self.self_attn_mask_swa, ubatch, self.cparams.causal_attn);
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = params.mctx.clone() else { return false };
        self.mctx = Some(mctx.clone());
        let (Some(base), Some(swa)) = (mctx.get_base(), mctx.get_swa()) else { return false };
        let mut res = true;
        if let Some(idxs) = &self.self_k_idxs {
            res &= idxs.ne(0) == params.ubatch.n_tokens as i64;
        }
        if self.self_attn_mask.raw.is_some() {
            res &= can_reuse_attn_mask(&self.self_attn_mask, base.as_ref(), &params.ubatch, &params.cparams);
        }
        if let Some(idxs) = &self.self_k_idxs_swa {
            res &= idxs.ne(0) == params.ubatch.n_tokens as i64;
        }
        if self.self_attn_mask_swa.raw.is_some() {
            res &= can_reuse_attn_mask(&self.self_attn_mask_swa, swa.as_ref(), &params.ubatch, &params.cparams);
        }
        res
    }
}

#[derive(Clone)]
pub struct GraphInputAttnKIswa {
    pub self_k_idxs: Option<Tensor>,
    pub self_k_idxs_swa: Option<Tensor>,
    pub self_attn_mask: AttnMask,
    pub self_attn_mask_swa: AttnMask,
    pub self_k_rot: Vec<Option<Tensor>>,
    pub self_k_rot_swa: Vec<Option<Tensor>>,
    pub hparams: Hparams,
    pub cparams: CParams,
    pub mctx: Option<Rc<dyn MemoryContext>>,
}

impl GraphInputAttnKIswa {
    pub fn new(hparams: &Hparams, cparams: &CParams, mctx: Rc<dyn MemoryContext>) -> Self {
        Self {
            self_k_idxs: None,
            self_k_idxs_swa: None,
            self_attn_mask: AttnMask::default(),
            self_attn_mask_swa: AttnMask::default(),
            self_k_rot: Vec::new(),
            self_k_rot_swa: Vec::new(),
            hparams: hparams.clone(),
            cparams: cparams.clone(),
            mctx: Some(mctx),
        }
    }
    pub fn get_k_idxs(&self) -> Option<Tensor> { self.self_k_idxs.clone() }
    pub fn get_k_idxs_swa(&self) -> Option<Tensor> { self.self_k_idxs_swa.clone() }
}

impl GraphInput for GraphInputAttnKIswa {
    fn name(&self) -> &'static str {
        "llm_graph_input_attn_k_iswa"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(mctx) = self.mctx.clone() else { return };
        let (Some(base), Some(swa)) = (mctx.get_base(), mctx.get_swa()) else { return };
        if let Some(idxs) = &self.self_k_idxs {
            base.set_input_k_idxs(idxs, ubatch);
        }
        base.set_input_attn_mask(&mut self.self_attn_mask, ubatch, self.cparams.causal_attn);
        if let Some(idxs) = &self.self_k_idxs_swa {
            swa.set_input_k_idxs(idxs, ubatch);
        }
        swa.set_input_attn_mask(&mut self.self_attn_mask_swa, ubatch, self.cparams.causal_attn);
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = params.mctx.clone() else { return false };
        self.mctx = Some(mctx.clone());
        let (Some(base), Some(swa)) = (mctx.get_base(), mctx.get_swa()) else { return false };
        let mut res = true;
        if let Some(idxs) = &self.self_k_idxs {
            res &= idxs.ne(0) == params.ubatch.n_tokens as i64;
        }
        if self.self_attn_mask.raw.is_some() {
            res &= can_reuse_attn_mask(&self.self_attn_mask, base.as_ref(), &params.ubatch, &params.cparams);
        }
        if let Some(idxs) = &self.self_k_idxs_swa {
            res &= idxs.ne(0) == params.ubatch.n_tokens as i64;
        }
        if self.self_attn_mask_swa.raw.is_some() {
            res &= can_reuse_attn_mask(&self.self_attn_mask_swa, swa.as_ref(), &params.ubatch, &params.cparams);
        }
        res
    }
}

#[derive(Clone)]
pub struct GraphInputAttnCross {
    pub cross_attn_mask: AttnMask,
    pub cross: Option<Rc<Cross>>,
}

impl GraphInputAttnCross {
    pub fn new(cross: Rc<Cross>) -> Self {
        Self { cross_attn_mask: AttnMask::default(), cross: Some(cross) }
    }
}

impl GraphInput for GraphInputAttnCross {
    fn name(&self) -> &'static str {
        "llm_graph_input_attn_cross"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(raw) = self.cross_attn_mask.raw.as_ref() else { return };
        let Some(cross) = &self.cross else { return };
        let n_enc = raw.ne(0).max(0) as usize;
        let n_tokens = ubatch.n_tokens as usize;
        let mut data = vec![-f32::INFINITY; n_enc * n_tokens];
        for (i, row) in data.chunks_mut(n_enc).enumerate().take(n_tokens) {
            let seq_ids = ubatch.seq_id.get(i).cloned().unwrap_or_default();
            for (j, val) in row.iter_mut().enumerate() {
                let visible = seq_ids.iter().any(|id| {
                    cross.seq_ids_enc.get(j).is_some_and(|set| set.contains(id))
                });
                if visible {
                    *val = 0.0;
                }
            }
        }
        raw.set_data_f32(&data);
    }
}

// ---------------------------------------------------------------------------
// Hybrid inputs
// ---------------------------------------------------------------------------

fn set_rs_copy(mctx: &dyn MemoryContext, inp_rs: &GraphInputRs) {
    let Some(s_copy) = &inp_rs.s_copy else { return };
    let n_rs = mctx.get_n_rs();
    let data: Vec<i32> = (0..n_rs).map(|i| mctx.s_copy(i)).collect();
    s_copy.set_data_i32(&data);
}

fn reuse_rs(inp_rs: &mut GraphInputRs, mctx: &dyn MemoryContext, params: &GraphParams) -> bool {
    let mut res = true;
    res &= matches!(&inp_rs.s_copy, Some(t) if t.ne(0) == mctx.get_n_rs() as i64);
    res &= matches!(&inp_rs.s_copy_main, Some(t) if t.ne(0) == params.ubatch.n_seqs as i64);
    res &= inp_rs.head == mctx.get_head();
    res &= inp_rs.rs_z == mctx.get_rs_z();
    res &= inp_rs.rs_in_place == mctx.get_rs_in_place();
    res
}

#[derive(Default)]
pub struct GraphInputMemHybrid {
    pub inp_attn: Option<Rc<RefCell<GraphInputAttnKv>>>,
    pub inp_rs: Option<Rc<RefCell<GraphInputRs>>>,
    pub cparams: CParams,
    pub mctx: Option<Rc<dyn MemoryContext>>,
}

impl GraphInputMemHybrid {
    pub fn new(
        cparams: &CParams,
        inp_attn: Rc<RefCell<GraphInputAttnKv>>,
        inp_rs: Rc<RefCell<GraphInputRs>>,
        mctx: Rc<dyn MemoryContext>,
    ) -> Self {
        Self {
            inp_attn: Some(inp_attn),
            inp_rs: Some(inp_rs),
            cparams: cparams.clone(),
            mctx: Some(mctx),
        }
    }
}

impl GraphInput for GraphInputMemHybrid {
    fn name(&self) -> &'static str {
        "llm_graph_input_mem_hybrid"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(mctx) = self.mctx.clone() else { return };
        let (Some(attn_ctx), Some(recr_ctx)) = (mctx.get_attn(), mctx.get_recr()) else { return };
        if let Some(inp_attn) = &self.inp_attn {
            let mut inp = inp_attn.borrow_mut();
            if let Some(idxs) = &inp.self_k_idxs {
                attn_ctx.set_input_k_idxs(idxs, ubatch);
            }
            if let Some(idxs) = &inp.self_v_idxs {
                attn_ctx.set_input_v_idxs(idxs, ubatch);
            }
            attn_ctx.set_input_attn_mask(&mut inp.self_attn_mask, ubatch, self.cparams.causal_attn);
        }
        if let Some(inp_rs) = &self.inp_rs {
            let mut rs = inp_rs.borrow_mut();
            set_rs_copy(recr_ctx.as_ref(), &rs);
            rs.set_input_rep(recr_ctx.as_ref(), ubatch);
        }
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = params.mctx.clone() else { return false };
        self.mctx = Some(mctx.clone());
        let (Some(attn_ctx), Some(recr_ctx)) = (mctx.get_attn(), mctx.get_recr()) else { return false };
        let mut res = true;
        if let Some(inp) = &self.inp_attn {
            let inp = inp.borrow_mut();
            res &= matches!(&inp.self_k_idxs, Some(t) if t.ne(0) == params.ubatch.n_tokens as i64);
            res &= can_reuse_attn_mask(&inp.self_attn_mask, attn_ctx.as_ref(), &params.ubatch, &params.cparams);
        } else {
            res = false;
        }
        if let Some(inp) = &self.inp_rs {
            res &= reuse_rs(&mut inp.borrow_mut(), recr_ctx.as_ref(), params);
        } else {
            res = false;
        }
        res
    }
}

#[derive(Default)]
pub struct GraphInputMemHybridK {
    pub inp_attn: Option<Rc<RefCell<GraphInputAttnK>>>,
    pub inp_rs: Option<Rc<RefCell<GraphInputRs>>>,
    pub cparams: CParams,
    pub mctx: Option<Rc<dyn MemoryContext>>,
}

impl GraphInputMemHybridK {
    pub fn new(
        cparams: &CParams,
        inp_attn: Rc<RefCell<GraphInputAttnK>>,
        inp_rs: Rc<RefCell<GraphInputRs>>,
        mctx: Rc<dyn MemoryContext>,
    ) -> Self {
        Self {
            inp_attn: Some(inp_attn),
            inp_rs: Some(inp_rs),
            cparams: cparams.clone(),
            mctx: Some(mctx),
        }
    }
}

impl GraphInput for GraphInputMemHybridK {
    fn name(&self) -> &'static str {
        "llm_graph_input_mem_hybrid_k"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(mctx) = self.mctx.clone() else { return };
        let (Some(attn_ctx), Some(recr_ctx)) = (mctx.get_attn(), mctx.get_recr()) else { return };
        if let Some(inp_attn) = &self.inp_attn {
            let mut inp = inp_attn.borrow_mut();
            if let Some(idxs) = &inp.self_k_idxs {
                attn_ctx.set_input_k_idxs(idxs, ubatch);
            }
            if inp.self_attn_mask.raw.is_some() {
                attn_ctx.set_input_attn_mask(&mut inp.self_attn_mask, ubatch, self.cparams.causal_attn);
            }
        }
        if let Some(inp_rs) = &self.inp_rs {
            let mut rs = inp_rs.borrow_mut();
            set_rs_copy(recr_ctx.as_ref(), &rs);
            rs.set_input_rep(recr_ctx.as_ref(), ubatch);
        }
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = params.mctx.clone() else { return false };
        self.mctx = Some(mctx.clone());
        let (Some(attn_ctx), Some(recr_ctx)) = (mctx.get_attn(), mctx.get_recr()) else { return false };
        let mut res = true;
        if let Some(inp) = &self.inp_attn {
            let inp = inp.borrow_mut();
            res &= matches!(&inp.self_k_idxs, Some(t) if t.ne(0) == params.ubatch.n_tokens as i64);
            res &= can_reuse_attn_mask(&inp.self_attn_mask, attn_ctx.as_ref(), &params.ubatch, &params.cparams);
        } else {
            res = false;
        }
        if let Some(inp) = &self.inp_rs {
            res &= reuse_rs(&mut inp.borrow_mut(), recr_ctx.as_ref(), params);
        } else {
            res = false;
        }
        res
    }
}

#[derive(Default)]
pub struct GraphInputMemHybridIswa {
    pub inp_attn: Option<Rc<RefCell<GraphInputAttnKvIswa>>>,
    pub inp_rs: Option<Rc<RefCell<GraphInputRs>>>,
    pub cparams: CParams,
    pub mctx: Option<Rc<dyn MemoryContext>>,
}

impl GraphInputMemHybridIswa {
    pub fn new(
        cparams: &CParams,
        inp_attn: Rc<RefCell<GraphInputAttnKvIswa>>,
        inp_rs: Rc<RefCell<GraphInputRs>>,
        mctx: Rc<dyn MemoryContext>,
    ) -> Self {
        Self {
            inp_attn: Some(inp_attn),
            inp_rs: Some(inp_rs),
            cparams: cparams.clone(),
            mctx: Some(mctx),
        }
    }
}

impl GraphInput for GraphInputMemHybridIswa {
    fn name(&self) -> &'static str {
        "llm_graph_input_mem_hybrid_iswa"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(mctx) = self.mctx.clone() else { return };
        let (Some(attn_ctx), Some(recr_ctx)) = (mctx.get_attn(), mctx.get_recr()) else { return };
        let (Some(base), Some(swa)) = (attn_ctx.get_base(), attn_ctx.get_swa()) else { return };
        if let Some(inp) = &self.inp_attn {
            let mut inp = inp.borrow_mut();
            if let Some(idxs) = &inp.self_k_idxs {
                base.set_input_k_idxs(idxs, ubatch);
                if let Some(v) = &inp.self_v_idxs {
                    base.set_input_v_idxs(v, ubatch);
                }
            }
            if inp.self_attn_mask.raw.is_some() {
                base.set_input_attn_mask(&mut inp.self_attn_mask, ubatch, self.cparams.causal_attn);
            }
            if let Some(idxs) = &inp.self_k_idxs_swa {
                swa.set_input_k_idxs(idxs, ubatch);
                if let Some(v) = &inp.self_v_idxs_swa {
                    swa.set_input_v_idxs(v, ubatch);
                }
            }
            if inp.self_attn_mask_swa.raw.is_some() {
                swa.set_input_attn_mask(&mut inp.self_attn_mask_swa, ubatch, self.cparams.causal_attn);
            }
        }
        if let Some(inp_rs) = &self.inp_rs {
            let mut rs = inp_rs.borrow_mut();
            set_rs_copy(recr_ctx.as_ref(), &rs);
            rs.set_input_rep(recr_ctx.as_ref(), ubatch);
        }
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = params.mctx.clone() else { return false };
        self.mctx = Some(mctx.clone());
        let (Some(attn_ctx), Some(recr_ctx)) = (mctx.get_attn(), mctx.get_recr()) else { return false };
        let (Some(base), Some(swa)) = (attn_ctx.get_base(), attn_ctx.get_swa()) else { return false };
        let mut res = true;
        if let Some(inp) = &self.inp_attn {
            let inp = inp.borrow_mut();
            if let Some(idxs) = &inp.self_k_idxs {
                res &= idxs.ne(0) == params.ubatch.n_tokens as i64;
            }
            res &= can_reuse_attn_mask(&inp.self_attn_mask, base.as_ref(), &params.ubatch, &params.cparams);
            if let Some(idxs) = &inp.self_k_idxs_swa {
                res &= idxs.ne(0) == params.ubatch.n_tokens as i64;
            }
            res &= can_reuse_attn_mask(&inp.self_attn_mask_swa, swa.as_ref(), &params.ubatch, &params.cparams);
        } else {
            res = false;
        }
        if let Some(inp) = &self.inp_rs {
            res &= reuse_rs(&mut inp.borrow_mut(), recr_ctx.as_ref(), params);
        } else {
            res = false;
        }
        res
    }
}

#[derive(Default)]
pub struct GraphInputMemHybridIswaRel {
    pub inp_attn: Option<Rc<RefCell<GraphInputAttnKvIswa>>>,
    pub inp_rs: Option<Rc<RefCell<GraphInputRs>>>,
    pub self_rel_pos: Option<Tensor>,
    pub self_rel_pos_swa: Option<Tensor>,
    pub extent: u32,
    pub extent_swa: u32,
    pub cparams: CParams,
    pub mctx: Option<Rc<dyn MemoryContext>>,
}

impl GraphInputMemHybridIswaRel {
    pub fn new(
        cparams: &CParams,
        inp_attn: Rc<RefCell<GraphInputAttnKvIswa>>,
        inp_rs: Rc<RefCell<GraphInputRs>>,
        mctx: Rc<dyn MemoryContext>,
        extent: u32,
        extent_swa: u32,
    ) -> Self {
        Self {
            inp_attn: Some(inp_attn),
            inp_rs: Some(inp_rs),
            self_rel_pos: None,
            self_rel_pos_swa: None,
            extent,
            extent_swa,
            cparams: cparams.clone(),
            mctx: Some(mctx),
        }
    }
}

impl GraphInput for GraphInputMemHybridIswaRel {
    fn name(&self) -> &'static str {
        "llm_graph_input_mem_hybrid_iswa_rel"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let Some(mctx) = self.mctx.clone() else { return };
        let (Some(attn_ctx), Some(recr_ctx)) = (mctx.get_attn(), mctx.get_recr()) else { return };
        let (Some(base), Some(swa)) = (attn_ctx.get_base(), attn_ctx.get_swa()) else { return };
        if let Some(inp) = &self.inp_attn {
            let mut inp = inp.borrow_mut();
            if let Some(idxs) = &inp.self_k_idxs {
                base.set_input_k_idxs(idxs, ubatch);
                if let Some(v) = &inp.self_v_idxs {
                    base.set_input_v_idxs(v, ubatch);
                }
            }
            if inp.self_attn_mask.raw.is_some() {
                base.set_input_attn_mask(&mut inp.self_attn_mask, ubatch, self.cparams.causal_attn);
            }
            if let Some(rel) = &self.self_rel_pos {
                base.set_input_relative_pos(rel, ubatch, self.extent);
            }
            if let Some(idxs) = &inp.self_k_idxs_swa {
                swa.set_input_k_idxs(idxs, ubatch);
                if let Some(v) = &inp.self_v_idxs_swa {
                    swa.set_input_v_idxs(v, ubatch);
                }
            }
            if inp.self_attn_mask_swa.raw.is_some() {
                swa.set_input_attn_mask(&mut inp.self_attn_mask_swa, ubatch, self.cparams.causal_attn);
            }
            if let Some(rel) = &self.self_rel_pos_swa {
                swa.set_input_relative_pos(rel, ubatch, self.extent_swa);
            }
        }
        if let Some(inp_rs) = &self.inp_rs {
            let mut rs = inp_rs.borrow_mut();
            set_rs_copy(recr_ctx.as_ref(), &rs);
            rs.set_input_rep(recr_ctx.as_ref(), ubatch);
        }
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        let Some(mctx) = params.mctx.clone() else { return false };
        self.mctx = Some(mctx.clone());
        let (Some(attn_ctx), Some(recr_ctx)) = (mctx.get_attn(), mctx.get_recr()) else { return false };
        let (Some(base), Some(swa)) = (attn_ctx.get_base(), attn_ctx.get_swa()) else { return false };
        let mut res = true;
        if let Some(inp) = &self.inp_attn {
            let inp = inp.borrow_mut();
            if let Some(idxs) = &inp.self_k_idxs {
                res &= idxs.ne(0) == params.ubatch.n_tokens as i64;
            }
            if let Some(raw) = &inp.self_attn_mask.raw {
                res &= can_reuse_kq_mask(raw, base.as_ref(), &params.ubatch, &params.cparams);
            }
            if let Some(rel) = &self.self_rel_pos {
                res &= rel.ne(0) == base.get_n_kv() as i64;
                res &= rel.ne(1) == params.ubatch.n_tokens as i64;
            }
            if let Some(idxs) = &inp.self_k_idxs_swa {
                res &= idxs.ne(0) == params.ubatch.n_tokens as i64;
            }
            if let Some(raw) = &inp.self_attn_mask_swa.raw {
                res &= can_reuse_kq_mask(raw, swa.as_ref(), &params.ubatch, &params.cparams);
            }
            if let Some(rel) = &self.self_rel_pos_swa {
                res &= rel.ne(0) == swa.get_n_kv() as i64;
                res &= rel.ne(1) == params.ubatch.n_tokens as i64;
            }
        } else {
            res = false;
        }
        if let Some(inp) = &self.inp_rs {
            res &= reuse_rs(&mut inp.borrow_mut(), recr_ctx.as_ref(), params);
        } else {
            res = false;
        }
        res
    }
}

#[derive(Default)]
pub struct GraphInputSampling {
    pub samplers: BTreeMap<SeqId, Rc<super::support::Sampler>>,
}

impl GraphInputSampling {
    pub fn new(samplers: &BTreeMap<SeqId, Rc<super::support::Sampler>>) -> Self {
        Self { samplers: samplers.clone() }
    }
}

impl GraphInput for GraphInputSampling {
    fn name(&self) -> &'static str {
        "llm_graph_input_sampling"
    }
    fn set_input(&mut self, ubatch: &Ubatch) {
        let mut active = HashSet::new();
        for (i, flag) in ubatch.output.iter().enumerate().take(ubatch.n_tokens as usize) {
            if *flag != 0 {
                if let Some(seq_id) = ubatch.seq_id.get(i).and_then(|v| v.first()) {
                    active.insert(*seq_id);
                }
            }
        }
        for seq_id in active {
            if let Some(sampler) = self.samplers.get(&seq_id) {
                sampler.backend_set_input();
            }
        }
    }
    fn can_reuse(&mut self, params: &GraphParams) -> bool {
        if self.samplers.len() != params.samplers.len() {
            return false;
        }
        params.samplers.iter().all(|(seq_id, sampler)| {
            matches!(self.samplers.get(seq_id), Some(cur) if Rc::ptr_eq(cur, sampler))
        })
    }
}

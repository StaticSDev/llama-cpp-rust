//! Output paths: pooling, logits, sampling and layer-input device writes.

use std::collections::BTreeMap;

use super::context::GraphContext;
use super::ggml::{
    self, add, get_rows, mul_mat, new_tensor_1d, pad, reshape_2d, set_rows, soft_max,
    unary_op, view_1d, view_2d, Tensor, Type,
};
use super::inputs::GraphInputBatchIdxs;
use super::support::SamplerData;
use super::types::NormType;
use crate::arch::Arch;
use crate::cparams::PoolingType;
use crate::core::SeqId;

/// `llama_relative_position_bucket`.
pub fn relative_position_bucket(x: i32, y: i32, mut n_buckets: u64, bidirectional: bool) -> i32 {
    const MAX_DISTANCE: i64 = 128;
    if bidirectional {
        n_buckets >>= 1;
    }
    let max_exact = (n_buckets >> 1) as i64;
    let mut relative_position = x - y;
    let mut relative_bucket = 0i32;

    if bidirectional {
        relative_bucket += (relative_position > 0) as i32 * n_buckets as i32;
        relative_position = relative_position.abs();
    } else {
        relative_position = -relative_position.min(0);
    }

    let numerator = (max_exact as f32 + ((1.0 * relative_position as f32 / max_exact as f32).ln())
        * ((n_buckets as f32 - max_exact as f32) / (1.0 * MAX_DISTANCE as f32 / max_exact as f32).ln()))
        .floor() as i32;
    let mut large = numerator;
    large = large.min(n_buckets as i32 - 1);
    relative_bucket += if relative_position < max_exact as i32 { relative_position } else { large };
    relative_bucket
}

impl GraphContext {
    pub fn build_dense_out(
        &self,
        dense_2: Option<&Tensor>,
        dense_2_b: Option<&Tensor>,
        dense_3: Option<&Tensor>,
    ) {
        if !self.cparams.embeddings || (dense_2.is_none() && dense_2_b.is_none() && dense_3.is_none()) {
            return;
        }
        let mut cur = self
            .res
            .borrow()
            .t_embd_pooled
            .clone()
            .or_else(|| self.res.borrow().t_embd.clone())
            .unwrap_or_else(|| new_tensor_1d(Type::F32, 1));
        if let Some(dense_2) = dense_2 {
            cur = mul_mat(dense_2, &cur);
        }
        if let Some(b) = dense_2_b {
            cur = add(&cur, b);
        }
        if let Some(dense_3) = dense_3 {
            cur = mul_mat(dense_3, &cur);
        }
        self.cb(&cur, "result_embd_pooled", -1);
        self.res.borrow_mut().t_embd_pooled = Some(cur.clone());
        self.expand(&cur);
    }

    pub fn build_pooling(
        &self,
        cls: Option<&Tensor>,
        cls_b: Option<&Tensor>,
        cls_out: Option<&Tensor>,
        cls_out_b: Option<&Tensor>,
        cls_norm: Option<&Tensor>,
    ) {
        let arch = self.model.arch;
        if !self.cparams.embeddings {
            let (no_logits, embd) = {
                let res = self.res.borrow();
                (res.t_logits.is_none(), res.t_embd.clone())
            };
            if no_logits {
                if let Some(embd) = embd {
                    self.expand(&embd);
                }
            }
            return;
        }
        let inp = self.res.borrow().t_embd.clone().unwrap_or_else(|| new_tensor_1d(Type::F32, 1));

        let cur = match self.pooling_type {
            PoolingType::None => inp,
            PoolingType::Mean => {
                let inp_mean = self.build_inp_mean();
                let t = ggml::cont(&ggml::transpose(&inp));
                mul_mat(&t, &inp_mean)
            }
            PoolingType::Cls | PoolingType::Last => {
                let inp_cls = self.build_inp_cls();
                get_rows(&inp, &inp_cls)
            }
            PoolingType::Rank => {
                let mut x = if matches!(
                    self.hparams.pooling_type_cls,
                    crate::hparams::PoolingType::Mean
                ) {
                    let inp_mean = self.build_inp_mean();
                    let t = ggml::cont(&ggml::transpose(&inp));
                    mul_mat(&t, &inp_mean)
                } else {
                    let inp_cls = self.build_inp_cls();
                    get_rows(&inp, &inp_cls)
                };
                if let Some(cls) = cls {
                    x = mul_mat(cls, &x);
                    if let Some(b) = cls_b {
                        x = add(&x, b);
                    }
                    x = unary_op(&x, self.hparams.act_cls as u32);
                    if let Some(norm) = cls_norm {
                        x = self.build_norm(&x, Some(norm), None, NormType::Norm, -1);
                    }
                }
                if let Some(cls_out) = cls_out {
                    x = mul_mat(cls_out, &x);
                    if let Some(b) = cls_out_b {
                        x = add(&x, b);
                    }
                }
                if arch == Arch::Qwen3 || arch == Arch::Qwen3vl {
                    x = soft_max(&x);
                }
                x
            }
            PoolingType::Unspecified => inp,
        };
        self.cb(&cur, "result_embd_pooled", -1);
        self.res.borrow_mut().t_embd_pooled = Some(cur.clone());
        self.expand(&cur);
    }

    pub fn build_logits(&self) {
        if self.cparams.embeddings && self.pooling_type != PoolingType::None {
            self.res.borrow_mut().t_logits = None;
        }
        let logits = self.res.borrow().t_logits.clone();
        if let Some(logits) = logits {
            self.expand(&logits);
        }
    }

    pub fn build_layer_inp_dev(&self) {
        if !self.cparams.embeddings_layer_inp_dev {
            return;
        }
        let Some(layer_inp_dev) = self.layer_inp_dev.clone() else { return };
        let mut inp = GraphInputBatchIdxs::default();
        let idxs = new_tensor_1d(Type::I64, self.n_tokens);
        idxs.set_input();
        inp.idxs = Some(idxs.clone());

        let mut col = 0i64;
        for (il, enabled) in self.cparams.embeddings_layer_inp.iter().enumerate() {
            if !*enabled {
                continue;
            }
            let src = self.res.borrow().t_layer_inp.get(il).cloned().flatten();
            let Some(src) = src else { continue };
            let dst = view_2d(
                &layer_inp_dev,
                self.n_embd,
                layer_inp_dev.ne(1),
                layer_inp_dev.nb(1),
                col as usize * self.n_embd as usize * idxs.ty().type_size(),
            );
            let rows = reshape_2d(&src, self.n_embd, self.n_tokens);
            self.expand(&set_rows(&dst, &rows, &idxs));
            col += 1;
        }
        self.res.borrow_mut().add_input(inp);
    }

    pub fn build_h_nextn_dev(&self) {
        let src = self.res.borrow().t_h_nextn.clone();
        let (Some(src), Some(h_nextn_dev)) = (src, self.h_nextn_dev.clone()) else { return };
        let mut inp = GraphInputBatchIdxs::default();
        let idxs = new_tensor_1d(Type::I64, self.n_tokens);
        idxs.set_input();
        inp.idxs = Some(idxs.clone());
        let rows = reshape_2d(&src, src.ne(0), self.n_tokens);
        self.expand(&set_rows(&h_nextn_dev, &rows, &idxs));
        self.res.borrow_mut().add_input(inp);
    }

    pub fn build_sampling(&self) {
        let logits = self.res.borrow().t_logits.clone();
        if self.samplers.is_empty() || logits.is_none() {
            return;
        }
        let logits = logits.unwrap();
        let inp_sampling = super::inputs::GraphInputSampling::new(&self.samplers);
        self.res.borrow_mut().add_input(inp_sampling);

        let mut sampling_rows: BTreeMap<SeqId, Vec<u32>> = BTreeMap::new();
        let mut n_rows = 0u32;
        for i in 0..self.ubatch.n_tokens as usize {
            if self.ubatch.output.get(i).copied().unwrap_or(0) != 0 {
                let seq_id = self.ubatch.seq_id.get(i).and_then(|v| v.first()).copied().unwrap_or(0);
                sampling_rows.entry(seq_id).or_default().push(n_rows);
                n_rows += 1;
            }
        }
        {
            let mut res = self.res.borrow_mut();
            res.t_sampled = vec![None; n_rows as usize];
            res.t_sampled_probs = vec![None; n_rows as usize];
            res.t_sampled_logits = vec![None; n_rows as usize];
            res.t_candidates = vec![None; n_rows as usize];
        }

        let logits_t = pad(&logits, 0, 1, 0, 0);
        for sampler in self.samplers.values() {
            sampler.backend_reset();
        }

        let no_rows: Vec<u32> = Vec::new();
        for (seq_id, sampler) in &self.samplers {
            let rows = sampling_rows.get(seq_id).unwrap_or(&no_rows);
            for i in 0..self.cparams.n_outputs_max_per_seq as usize {
                let active = i < rows.len();
                let row = if active { rows[i] as i64 } else { 0 };
                let mut logits_seq = view_1d(&logits_t, logits_t.ne(0), row as usize * logits_t.nb(1));
                ggml::format_name(&logits_seq, format!("logits_seq_{}_{}", seq_id, i));
                let mut data = SamplerData {
                    logits: Some(logits_seq.clone()),
                    probs: None,
                    sampled: None,
                    candidates: None,
                };
                sampler.backend_apply(&self.gf, &mut data);
                logits_seq = data.logits.clone().unwrap_or(logits_seq);

                if let Some(sampled) = &data.sampled {
                    if active {
                        self.res.borrow_mut().t_sampled[rows[i] as usize] = Some(sampled.clone());
                    }
                    let outs = [logits_t.clone(), sampled.clone()];
                    let idx = if active { 1 } else { 0 };
                    self.gf.borrow_mut().build_forward_select(&outs, idx);
                }
                if let Some(probs) = &data.probs {
                    if active {
                        self.res.borrow_mut().t_sampled_probs[rows[i] as usize] = Some(probs.clone());
                    }
                    let outs = [logits_t.clone(), probs.clone()];
                    let idx = if active { 1 } else { 0 };
                    self.gf.borrow_mut().build_forward_select(&outs, idx);
                }
                if let Some(new_logits) = &data.logits {
                    if active {
                        self.res.borrow_mut().t_sampled_logits[rows[i] as usize] = Some(new_logits.clone());
                    }
                    let outs = [logits_t.clone(), new_logits.clone()];
                    let idx = if active { 1 } else { 0 };
                    self.gf.borrow_mut().build_forward_select(&outs, idx);
                }
                if let Some(candidates) = &data.candidates {
                    if active {
                        self.res.borrow_mut().t_candidates[rows[i] as usize] = Some(candidates.clone());
                    }
                    let outs = [logits_t.clone(), candidates.clone()];
                    let idx = if active { 1 } else { 0 };
                    self.gf.borrow_mut().build_forward_select(&outs, idx);
                }

                let mut res = self.res.borrow_mut();
                if let Some(t) = data.sampled { res.t_sampling_all.push(t); }
                if let Some(t) = data.probs { res.t_sampling_all.push(t); }
                if let Some(t) = data.logits { res.t_sampling_all.push(t); }
                if let Some(t) = data.candidates { res.t_sampling_all.push(t); }
            }
        }
    }
}

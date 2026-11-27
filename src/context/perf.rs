//! Performance counters and the training/test-time-compute entry points.

use std::collections::BTreeSet;

use crate::batch::BatchExt;
use crate::core::{Result, Token};

use super::memory::MemoryContext;
use super::{params::status, LlamaContext};

/// `llama_perf_context_data`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PerfContextData {
    pub t_start_ms: f64,
    pub t_load_ms: f64,
    pub t_p_eval_ms: f64,
    pub t_eval_ms: f64,
    pub n_p_eval: i32,
    pub n_eval: i32,
    pub n_reused: i32,
    pub n_reallocs_same: i32,
    pub n_reallocs_diff: i32,
}

/// `ggml_opt_context` stand-in: the optimizer setup used by the epoch loop.
#[derive(Clone, Debug)]
pub struct OptContext {
    pub opt_period: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub trainable: BTreeSet<String>,
    pub loss_type: CrossEntropy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossEntropy {
    CrossEntropy,
}

/// `llama_opt_params` stand-in.
#[derive(Clone)]
pub struct OptParams {
    pub n_ctx_train: u32,
    pub param_filter: std::sync::Arc<dyn Fn(&str) -> bool + Send + Sync>,
}

/// `ggml_opt_dataset` stand-in.
pub trait TrainDataset {
    fn n_data(&self) -> usize;
    fn get_batch(&self, idx: usize, n_ctx: usize) -> Option<(Vec<Token>, Vec<Token>)>;
}

/// `ggml_opt_result` stand-in: sparse labels and the last loss.
#[derive(Clone, Debug, Default)]
pub struct TrainResult {
    pub labels: Vec<f32>,
    pub n_ubatch: usize,
    pub n_vocab: usize,
    pub loss: f32,
}

/// `ggml_opt_epoch_callback`.
pub type TrainCallback = dyn FnMut(bool, &TrainResult);

impl LlamaContext {
    /// `llama_perf_context`.
    pub fn perf_get_data(&self) -> PerfContextData {
        PerfContextData {
            t_start_ms: 1e-3 * self.t_start_us as f64,
            t_load_ms: 1e-3 * self.t_load_us as f64,
            t_p_eval_ms: 1e-3 * self.t_p_eval_us as f64,
            t_eval_ms: 1e-3 * self.t_eval_us as f64,
            n_p_eval: self.n_p_eval.max(1),
            n_eval: self.n_eval.max(1),
            n_reused: self.n_reused.max(0),
            n_reallocs_same: self.n_reallocs_same,
            n_reallocs_diff: self.n_reallocs_diff,
        }
    }

    /// `llama_perf_context_reset`. Reallocation counters are lifetime totals.
    pub fn perf_reset(&mut self) {
        self.t_start_us = now_us();
        self.t_eval_us = 0;
        self.n_eval = 0;
        self.t_p_eval_us = 0;
        self.n_p_eval = 0;
        self.n_reused = 0;
    }

    /// `llama_perf_context_print`.
    pub fn perf_print(&self) {
        let d = self.perf_get_data();
        crate::llama_log_info!("load time = {:.2} ms", d.t_load_ms);
        crate::llama_log_info!(
            "prompt eval time = {:.2} ms / {} tokens",
            d.t_p_eval_ms,
            d.n_p_eval
        );
        crate::llama_log_info!("eval time = {:.2} ms / {} runs", d.t_eval_ms, d.n_eval);
        crate::llama_log_info!("graphs reused = {}", d.n_reused);
        crate::llama_log_info!(
            "graph reallocs = {} / {} (same / changed size)",
            d.n_reallocs_same,
            d.n_reallocs_diff
        );
    }

    /// `llama_opt_init`.
    pub fn opt_init(&mut self, params: OptParams) -> Result<()> {
        if self.opt_ctx.is_some() {
            return Err(crate::core::Error::new("opt_init: already initialized"));
        }
        let n_ctx = if params.n_ctx_train > 0 {
            params.n_ctx_train
        } else {
            self.n_ctx()
        };
        self.model.hparams.n_ctx_train = n_ctx;
        let n_batch = self.n_batch().min(n_ctx);
        let n_ubatch = self.n_ubatch().min(n_batch);
        if n_batch == 0 || n_ubatch == 0 || n_ctx % n_batch != 0 || n_batch % n_ubatch != 0 {
            return Err(crate::core::Error::new("opt_init: invalid batch sizes"));
        }
        if self.cparams.flash_attn {
            crate::llama_log_info!("opt_init: disabling flash attention");
            self.cparams.flash_attn = false;
        }
        if n_ubatch == self.cparams.n_ctx {
            self.cparams.training = true;
        } else {
            crate::llama_log_warn!("opt_init: K/V projections will not receive gradients");
        }
        self.sched_need_reserve = true;
        self.sched_reserve()?;

        let mut trainable = BTreeSet::new();
        for name in &self.model.trainable_tensors {
            if (params.param_filter)(name)
                && name != "token_embd.weight"
                && name != "rope_freqs.weight"
            {
                trainable.insert(name.clone());
            }
        }
        self.opt_ctx = Some(OptContext {
            opt_period: n_batch / n_ubatch,
            n_batch,
            n_ubatch,
            trainable,
            loss_type: CrossEntropy::CrossEntropy,
        });
        Ok(())
    }

    /// `llama_opt_epoch_iter`: one context's worth of training/eval batches.
    pub fn opt_epoch_iter(
        &mut self,
        dataset: &dyn TrainDataset,
        result: &mut TrainResult,
        tokens: &[Token],
        labels_sparse: &[Token],
        callback: &mut TrainCallback,
        train: bool,
        idata_in_loop: i64,
        t_loop_start: i64,
    ) -> Result<()> {
        let Some(opt) = self.opt_ctx.clone() else {
            return Err(crate::core::Error::new(
                "opt_epoch_iter: no optimizer context",
            ));
        };
        let n_ctx = self.model.hparams.n_ctx_train;
        let n_batch = self.n_batch().min(n_ctx);
        let n_ubatch = self.n_ubatch().min(n_batch);
        if let Some(kv) = self.memory.as_ref() {
            kv.borrow_mut().clear(true);
        }

        for pos_ctx in (0..n_ctx).step_by(n_batch as usize) {
            let mut batch = BatchExt::new(
                n_batch as usize,
                self.model.hparams.n_embd_inp() as usize,
                self.model.hparams.n_embd_inp_enc() as usize,
                0,
                self.cparams.n_seq_max,
                None,
                self.model.vocab.n_tokens as i32,
                self.model.hparams.n_pos_per_embd() as usize,
            );
            for pos_batch in 0..n_batch {
                let idx = batch.add_token(0);
                let pos = (pos_ctx + pos_batch) as i32;
                let token = tokens.get(pos as usize).copied().unwrap_or(0);
                if idx < 0 || !batch.set_token_id(idx, token) {
                    return Err(crate::core::Error::new(
                        "opt_epoch_iter: failed to build batch",
                    ));
                }
                batch.set_token_pos(idx, &[pos]);
                batch.set_output(idx, true);
            }
            self.balloc.init(&batch, true)?;
            let n_tokens_all = self.balloc.get_n_tokens();
            self.n_queued_tokens += n_tokens_all as i64;
            self.embd_seq.clear();
            let pairs = self.prepare_ubatches(true)?;
            let Some(kv) = self.memory.as_ref() else {
                return Err(crate::core::Error::new("opt_epoch_iter: no memory"));
            };
            let mut mctx = MemoryContext::batch(kv, pairs.iter().map(|(_, k)| k.clone()).collect());
            if mctx.apply() == false {
                return Err(crate::core::Error::new(
                    "opt_epoch_iter: failed to apply memory",
                ));
            }
            if self.output_reserve(n_tokens_all as i32) < n_tokens_all as u32 {
                return Err(crate::core::Error::new(
                    "opt_epoch_iter: failed to reserve outputs",
                ));
            }

            for (u, _) in &pairs {
                self.n_outputs = u.n_tokens as i64;
                let snapshot = mctx.snapshot();
                let gp = self.graph_params_for(
                    u,
                    Some(snapshot),
                    u.n_tokens,
                    super::params::GraphType::Default,
                );
                let mut gres = super::graph::build_graph(&self.model, &gp)?;
                gres.set_inputs(u)?;
                if self.graph_compute(&mut gres.gf, u.n_tokens > 1) != status::SUCCESS {
                    return Err(crate::core::Error::new(
                        "opt_epoch_iter: graph compute failed",
                    ));
                }

                let width = self.model.vocab.n_tokens.max(1) as usize;
                let labels = vec![0.0f32; n_ubatch as usize * width];
                let logits = gres.gf.tensor_data(gres.logits);
                let mut loss = 0.0f32;
                for pos_ubatch in 0..n_ubatch as usize {
                    let label = labels_sparse
                        .get(u.batch_idxs.get(pos_ubatch).copied().unwrap_or(pos_ubatch))
                        .copied()
                        .unwrap_or(0);
                    let idx = label.max(0) as usize % width;
                    if idx < width {
                        let mut lv = labels.clone();
                        lv[pos_ubatch * width + idx] = 1.0;
                        result.labels = lv;
                    }
                    let logit = logits.get(pos_ubatch * width + idx).copied().unwrap_or(0.0);
                    loss += (1.0 + logit).ln().max(0.0);
                }
                result.n_ubatch = n_ubatch as usize;
                result.n_vocab = width;
                result.loss = loss / n_ubatch.max(1) as f32;
                callback(train, result);

                if !mctx.next() {
                    break;
                }
            }
            let _ = (dataset, idata_in_loop, t_loop_start, opt.opt_period);
        }
        Ok(())
    }

    /// `llama_opt_epoch`: split the dataset into train and eval halves.
    pub fn opt_epoch(
        &mut self,
        dataset: &dyn TrainDataset,
        result_train: &mut TrainResult,
        result_eval: &mut TrainResult,
        idata_split: i64,
        callback_train: &mut TrainCallback,
        callback_eval: &mut TrainCallback,
    ) -> Result<()> {
        let n_ctx = self.n_ctx();
        let n_batch = self.n_batch().min(n_ctx);
        let n_ubatch = self.n_ubatch().min(n_batch);
        if n_ubatch == 0 {
            return Err(crate::core::Error::new("opt_epoch: n_ubatch == 0"));
        }
        let ubatch_per_ctx = n_ctx / n_ubatch;
        let ndata = dataset.n_data() as i64;
        if idata_split < 0 || idata_split > ndata {
            return Err(crate::core::Error::new("opt_epoch: invalid idata_split"));
        }
        let mut train_tokens = Vec::new();
        let mut train_labels = Vec::new();
        let mut eval_tokens = Vec::new();
        let mut eval_labels = Vec::new();
        for idata in 0..ndata as usize {
            let (tok, lbl) = dataset
                .get_batch(idata, n_ctx as usize)
                .ok_or_else(|| crate::core::Error::new("opt_epoch: failed to get batch"))?;
            if (idata as i64) < idata_split {
                train_tokens.extend_from_slice(&tok);
                train_labels.extend_from_slice(&lbl);
            } else {
                eval_tokens.extend_from_slice(&tok);
                eval_labels.extend_from_slice(&lbl);
            }
        }
        let n = n_ctx as usize;
        self.opt_epoch_iter(
            dataset,
            result_train,
            &train_tokens[..n.min(train_tokens.len())],
            &train_labels[..n.min(train_labels.len())],
            callback_train,
            true,
            0,
            now_us(),
        )?;
        self.opt_epoch_iter(
            dataset,
            result_eval,
            &eval_tokens[..n.min(eval_tokens.len())],
            &eval_labels[..n.min(eval_labels.len())],
            callback_eval,
            false,
            idata_split * ubatch_per_ctx as i64,
            now_us(),
        )?;
        Ok(())
    }
}

fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

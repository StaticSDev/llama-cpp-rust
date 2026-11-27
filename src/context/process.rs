//! `process_ubatch`, `encode` and `decode`.

use crate::batch::{BatchExt, Ubatch};
use crate::core::SeqId;
use crate::cparams::PoolingType;
use crate::kv_cache;

use super::graph::{build_graph, GraphParams, GraphResult};
use super::memory::MemoryContext;
use super::model::Arch;
use super::params::{status, GraphType};
use super::LlamaContext;

/// True when an output token has a sequence without a backend sampler.
pub fn needs_raw_logits(
    ubatch: &Ubatch,
    samplers: &std::collections::BTreeMap<SeqId, super::params::BackendSampler>,
) -> bool {
    for i in 0..ubatch.n_tokens as usize {
        if ubatch.output.get(i).copied().unwrap_or(0) == 0 {
            continue;
        }
        let n_seq = ubatch.n_seq_id.get(i).copied().unwrap_or(0).max(0) as usize;
        for j in 0..n_seq {
            let seq_id = match ubatch.seq_id.get(i).and_then(|v| v.get(j)) {
                Some(&s) => s,
                None => break,
            };
            if !samplers.contains_key(&seq_id) {
                return true;
            }
        }
    }
    false
}

impl LlamaContext {
    /// Process one micro-batch with a specific graph type.
    pub fn process_ubatch(
        &mut self,
        ubatch: &Ubatch,
        gtype: GraphType,
        mut mctx: Option<&mut MemoryContext>,
        ret: &mut i32,
    ) -> Option<GraphResult> {
        if let Some(mctx) = mctx.as_mut() {
            if !mctx.apply() {
                crate::llama_log_error!("process_ubatch: failed to apply memory context");
                *ret = status::FAILED;
                return None;
            }
        }

        let snapshot = mctx.map(|m| m.snapshot());
        let gp: GraphParams =
            self.graph_params_for(ubatch, snapshot, self.n_outputs.max(1) as u32, gtype);
        let slot = usize::from(self.n_outputs > 0);

        let mut res = std::mem::take(&mut self.gf_res_prev[slot]);
        if !self.graph_reuse_disable && self.gf_res_prev_active == Some(slot) && res.can_reuse(&gp)
        {
            self.n_reused += 1;
        } else {
            self.gf_res_prev_active = None;
            res.reset();
            self.sched.reset();
            match build_graph(&self.model, &gp) {
                Ok(gf) => res = gf,
                Err(err) => {
                    crate::llama_log_error!("process_ubatch: failed to initialize graph: {}", err);
                    *ret = status::FAILED;
                    self.gf_res_prev[slot] = res;
                    return None;
                }
            }
            if !self.sched.alloc_graph(&res.gf) {
                crate::llama_log_error!("process_ubatch: failed to allocate graph");
                *ret = status::ALLOC_FAILED;
                self.gf_res_prev[slot] = res;
                return None;
            }
            self.gf_res_prev_active = Some(slot);
        }

        if let Err(err) = res.set_inputs(ubatch) {
            crate::llama_log_error!("process_ubatch: failed to set inputs: {}", err);
            *ret = status::FAILED;
            self.gf_res_prev[slot] = res;
            return None;
        }

        let graph_status = self.graph_compute(&mut res.gf, ubatch.n_tokens > 1);
        if graph_status != status::SUCCESS {
            crate::llama_log_error!("process_ubatch: compute status {}", graph_status);
            *ret = graph_status;
            self.gf_res_prev[slot] = res;
            return None;
        }

        let out = res.clone();
        self.gf_res_prev[slot] = res;
        *ret = status::SUCCESS;
        Some(out)
    }

    /// `llama_encode`: one non-causal pass over the whole batch.
    pub fn encode(&mut self, batch_inp: &BatchExt) -> i32 {
        if batch_inp.tokens.is_empty() {
            crate::llama_log_error!("encode: n_tokens == 0");
            return -1;
        }
        let n_embd_inp_enc = self.model.hparams.n_embd_inp_enc();
        if batch_inp.n_embd > 0 && batch_inp.n_embd as u32 != n_embd_inp_enc {
            crate::llama_log_error!(
                "encode: embd row width {} does not match the encoder input {}",
                batch_inp.n_embd,
                n_embd_inp_enc
            );
            return -1;
        }
        if batch_inp.tokens[0].embd_row >= 0 {
            crate::llama_log_error!("encode: embd row batches are not supported");
            return -1;
        }

        if self
            .balloc
            .init(batch_inp, self.memory.is_some() || self.cparams.embeddings)
            .is_err()
        {
            crate::llama_log_error!("encode: failed to initialize batch");
            return -1;
        }
        let n_tokens = self.balloc.get_n_tokens();
        if n_tokens == 0 {
            return -1;
        }
        let ubatch = self.balloc.split_simple(n_tokens as u32);
        if n_tokens as u32 > self.cparams.n_ubatch {
            crate::llama_log_error!(
                "encode: n_tokens {} > n_ubatch {}, increase n_ubatch (-ub)",
                n_tokens,
                self.cparams.n_ubatch
            );
            return -1;
        }
        if !self.embd_seq.is_empty() {
            self.synchronize();
        }
        self.embd_seq.clear();
        if self.t_compute_start_us == 0 {
            self.t_compute_start_us = now_us();
        }
        if self.sched_reserve().is_err() {
            return -2;
        }
        self.n_queued_tokens += n_tokens as i64;

        let n_outputs_all = self.balloc.get_n_outputs() as usize;
        if self.output_reserve(n_outputs_all as i32) < n_outputs_all as u32 {
            crate::llama_log_error!(
                "encode: could not reserve space for {} outputs",
                n_outputs_all
            );
            return -2;
        }
        for (i, &out_id) in self.balloc.get_out_ids().iter().enumerate() {
            if i < self.output_ids.len() && out_id < self.output_ids.len() {
                self.output_ids[out_id] = i as i64;
            }
        }
        self.n_outputs = n_outputs_all as i64;

        let causal_org = self.cparams.causal_attn;
        self.cparams.causal_attn = false;
        let mut status = status::SUCCESS;
        let res = self.process_ubatch(&ubatch, GraphType::Encoder, None, &mut status);
        self.cparams.causal_attn = causal_org;
        let res = match res {
            Some(res) => res,
            None => return encode_status(status),
        };

        let n_vocab = self.model.vocab.n_tokens.max(1) as usize;
        let n_embd_out = self.model.hparams.n_embd_out().max(1) as usize;
        self.extract_logits(&res, 0, n_outputs_all);
        if self.embd.has_data() && res.get_embd().is_some() {
            self.extract_embeddings(&res, &ubatch, 0, n_outputs_all);
        }
        if self.cparams.embeddings_nextn && self.cparams.pooling_type == PoolingType::None {
            self.extract_nextn(&res, 0, 0, n_outputs_all, n_tokens);
        }
        let _ = (n_vocab, n_embd_out);

        if self.model.arch == Arch::T5 && res.get_embd().is_some() {
            self.synchronize();
            let shape = res.gf.tensor_shape(res.embd).unwrap_or([0; 4]);
            self.cross.n_embd = shape[0];
            self.cross.n_enc = shape[1];
            self.cross.v_embd = self.embd.as_vec();
            self.cross.seq_ids_enc = ubatch.seq_id.clone();
        }
        0
    }

    /// `llama_decode`: batching, memory updates and output extraction.
    pub fn decode(&mut self, batch_inp: &BatchExt) -> i32 {
        if self.memory.is_none() {
            crate::llama_log_debug!("decode: no memory, calling encode");
            return self.encode(batch_inp);
        }
        if batch_inp.tokens.is_empty() {
            crate::llama_log_error!("decode: n_tokens == 0");
            return -1;
        }
        if batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp {
            crate::llama_log_error!(
                "decode: embd row width {} does not match decoder input {}",
                batch_inp.n_embd,
                batch_inp.n_embd_inp
            );
            return -1;
        }
        let has_embd_row = batch_inp.tokens[0].embd_row >= 0;
        if has_embd_row && !self.supports_embd_row() {
            crate::llama_log_error!("decode: this context cannot read embd rows");
            return -1;
        }
        let has_state_row = batch_inp.tokens[0].state_row >= 0;
        if has_state_row && !self.supports_state_row() {
            crate::llama_log_error!("decode: this context cannot read state rows");
            return -1;
        }

        let output_all = self.cparams.embeddings;
        let embd_all = output_all || !self.cparams.causal_attn;
        let has_samplers = !self.sampling.samplers.is_empty();
        let n_seq_max = if self.cparams.kv_unified {
            crate::cparams::LLAMA_MAX_SEQ
        } else {
            self.cparams.n_seq_max
        };

        if has_samplers {
            let mut counts = vec![0i32; n_seq_max as usize];
            for tok in &batch_inp.tokens {
                if !output_all && !tok.output {
                    continue;
                }
                for &seq_id in &tok.seq_ids {
                    if seq_id >= n_seq_max {
                        continue;
                    }
                    let entry = counts[seq_id as usize];
                    counts[seq_id as usize] += 1;
                    if self.sampling.samplers.contains_key(&seq_id)
                        && counts[seq_id as usize] > self.cparams.n_outputs_max_per_seq as i32
                    {
                        crate::llama_log_error!(
                            "decode: backend sampling supports at most {} outputs per sequence (seq_id {})",
                            self.cparams.n_outputs_max_per_seq,
                            seq_id
                        );
                        return -1;
                    }
                    let _ = entry;
                }
            }
        }

        if self.balloc.init(batch_inp, output_all).is_err() {
            crate::llama_log_error!("decode: failed to initialize batch");
            return -1;
        }
        let n_tokens_all = self.balloc.get_n_tokens();
        let n_outputs_all = self.balloc.get_n_outputs() as usize;
        if output_all && n_outputs_all != n_tokens_all {
            crate::llama_log_error!("decode: pooled embedding requires all tokens as outputs");
            return -1;
        }
        if n_tokens_all > self.cparams.n_batch as usize {
            crate::llama_log_error!("decode: n_tokens exceeds n_batch");
            return -2;
        }
        if !self.cparams.causal_attn && (self.cparams.n_ubatch as usize) < n_tokens_all {
            crate::llama_log_error!("decode: non-causal attention requires n_ubatch >= n_tokens");
            return -2;
        }

        if !self.embd_seq.is_empty() {
            self.synchronize();
        }
        self.embd_seq.clear();
        if self.t_compute_start_us == 0 {
            self.t_compute_start_us = now_us();
        }
        self.n_queued_tokens += n_tokens_all as i64;
        self.output_swaps.clear();
        self.embd_batch_idxs.clear();
        if self.sched_reserve().is_err() {
            return -2;
        }
        let _ = self.memory_update(false);

        let Some(kv) = self.memory.as_ref().cloned() else {
            return -2;
        };
        let pairs = match self.prepare_ubatches(embd_all) {
            Ok(pairs) => pairs,
            Err(_) => return -2,
        };
        let kv_ubatches: Vec<_> = pairs.iter().map(|(_, k)| k.clone()).collect();
        let mut mctx = MemoryContext::batch(&kv, kv_ubatches);
        let mut did_optimize = false;
        loop {
            match mctx.get_status() {
                kv_cache::MemoryStatus::Success => break,
                kv_cache::MemoryStatus::FailedPrepare => {
                    if !did_optimize {
                        did_optimize = true;
                        if matches!(self.memory_update(true), Ok(true)) {
                            continue;
                        }
                    }
                    crate::llama_log_warn!("decode: failed to find a memory slot for the batch");
                    return 1;
                }
                _ => {
                    crate::llama_log_error!("decode: unexpected memory context status");
                    return -2;
                }
            }
        }

        if self.output_reserve(n_outputs_all as i32) < n_outputs_all as u32 {
            crate::llama_log_error!(
                "decode: could not reserve space for {} outputs",
                n_outputs_all
            );
            return -2;
        }
        for sampler in self.sampling.samplers.values_mut() {
            sampler.begin();
        }

        let mut n_outputs_prev = 0usize;
        let mut n_tokens_prev = 0usize;
        let mut ubatch_idx = 0usize;
        let n_outputs_eq_tokens = n_outputs_all == n_tokens_all;
        let gtype = match self.cparams.ctx_type {
            crate::cparams::ContextType::Default => GraphType::Default,
            crate::cparams::ContextType::Mtp => GraphType::DecoderMtp,
        };

        loop {
            let ubatch = pairs[ubatch_idx].0.clone();
            let n_outputs_new = if n_outputs_eq_tokens {
                ubatch.n_tokens as usize
            } else {
                ubatch.output.iter().filter(|&&o| o != 0).count()
            };
            self.n_outputs = n_outputs_new as i64;

            let mut ret_code = status::SUCCESS;
            let res = self.process_ubatch(&ubatch, gtype, Some(&mut mctx), &mut ret_code);
            let Some(res) = res else {
                // Remove all positions of the failed ubatch from the cache.
                if let Some(kv) = self.memory.as_ref() {
                    let mut cache = kv.borrow_mut();
                    let mut pos_min = vec![i32::MAX; crate::batch::LLAMA_MAX_SEQ];
                    for i in 0..ubatch.n_tokens as usize {
                        let seq_id = ubatch
                            .seq_id
                            .get(i)
                            .and_then(|v| v.first())
                            .copied()
                            .unwrap_or(0);
                        let pos = ubatch.pos.get(i).copied().unwrap_or(0);
                        if (seq_id as usize) < pos_min.len() {
                            pos_min[seq_id as usize] = pos_min[seq_id as usize].min(pos);
                        }
                    }
                    for (s, &p) in pos_min.iter().enumerate() {
                        if p != i32::MAX {
                            crate::llama_log_warn!(
                                "decode: removing memory entries for seq_id {}, pos [{}, +inf)",
                                s,
                                p
                            );
                            cache.seq_rm(s as i32, p, -1);
                        }
                    }
                }
                return match ret_code {
                    status::ABORTED => 2,
                    status::ALLOC_FAILED => -2,
                    _ => -3,
                };
            };

            if self.logits.has_data()
                && res.get_logits().is_some()
                && n_outputs_new > 0
                && needs_raw_logits(&ubatch, &self.sampling.samplers)
            {
                self.extract_logits(&res, n_outputs_prev, n_outputs_new);
            }
            if self.embd.has_data() && res.get_embd().is_some() && n_outputs_new > 0 {
                self.extract_embeddings(&res, &ubatch, n_outputs_prev, n_outputs_new);
            }
            let extract_all =
                self.extract_layer_inputs(&res, n_tokens_prev, ubatch.n_tokens as usize);
            let nextn_all = self.extract_nextn(
                &res,
                n_outputs_prev,
                n_tokens_prev,
                n_outputs_new,
                ubatch.n_tokens as usize,
            );
            if extract_all || nextn_all {
                self.embd_batch_idxs.extend_from_slice(&ubatch.batch_idxs);
            }
            self.extract_sampling(&res, n_outputs_prev);

            n_outputs_prev += n_outputs_new;
            n_tokens_prev += ubatch.n_tokens as usize;
            ubatch_idx += 1;
            if !mctx.next() {
                break;
            }
        }

        self.n_outputs = n_outputs_all as i64;
        if self.n_outputs > 0 {
            let out_ids = self.balloc.get_out_ids().to_vec();
            let mut sorted = true;
            for (i, &out_id) in out_ids.iter().enumerate() {
                if i >= self.output_ids.len() || out_id >= self.output_ids.len() {
                    continue;
                }
                self.output_ids[out_id] = i as i64;
                if out_id != i {
                    sorted = false;
                }
            }
            if !sorted && out_ids.len() > 1 {
                let mut out_ids = out_ids;
                for i in 0..out_ids.len() - 1 {
                    let mut j_min = i;
                    for j in i + 1..out_ids.len() {
                        if out_ids[j] < out_ids[j_min] {
                            j_min = j;
                        }
                    }
                    if j_min == i {
                        continue;
                    }
                    out_ids.swap(i, j_min);
                    self.output_swaps.push((i as u32, j_min as u32));
                }
                self.output_ids.fill(-1);
                for (i, &out_id) in out_ids.iter().enumerate() {
                    if out_id < self.output_ids.len() {
                        self.output_ids[out_id] = i as i64;
                    }
                }
            }
        }
        0
    }
}

fn encode_status(status: i32) -> i32 {
    match status {
        status::ABORTED => 2,
        status::ALLOC_FAILED => -2,
        status::FAILED => -3,
        _ => 0,
    }
}

fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

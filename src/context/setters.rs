//! Context setters and adapter wiring.

use std::cell::RefCell;
use std::rc::Rc;

use crate::core::{Error, Result, SeqId};
use crate::cparams::ContextType;

use super::model::{LoraRef, Model};
use super::params::BackendSampler;
use super::LlamaContext;

impl LlamaContext {
    /// `llama_attach_threadpool`.
    pub fn attach_threadpool(
        &mut self,
        threadpool: Option<usize>,
        threadpool_batch: Option<usize>,
    ) {
        self.threadpool = threadpool;
        self.threadpool_batch = threadpool_batch.or(threadpool);
    }

    /// `llama_detach_threadpool`.
    pub fn detach_threadpool(&mut self) {
        self.threadpool = None;
        self.threadpool_batch = None;
    }

    /// `llama_set_n_threads`.
    pub fn set_n_threads(&mut self, n_threads: i32, n_threads_batch: i32) {
        self.cparams.n_threads = n_threads;
        self.cparams.n_threads_batch = n_threads_batch;
        for backend in &mut self.backends {
            backend.n_threads = n_threads;
        }
    }

    /// `llama_set_abort_callback`.
    pub fn set_abort_callback(&mut self, callback: Option<std::sync::Arc<dyn Fn() -> bool>>) {
        self.params.abort_callback = callback.clone();
        self.abort_callback = callback;
        for backend in &mut self.backends {
            backend.abortable = self.abort_callback.is_some();
        }
    }

    /// `llama_set_embeddings`.
    pub fn set_embeddings(&mut self, value: bool) {
        if self.cparams.embeddings != value {
            self.sched_need_reserve = true;
        }
        self.cparams.embeddings = value;
    }

    /// `llama_set_embeddings_nextn`.
    pub fn set_embeddings_nextn(&mut self, value: bool, masked: bool) {
        if self.cparams.embeddings_nextn != value || self.cparams.embeddings_nextn_masked != masked
        {
            self.sched_need_reserve = true;
        }
        self.cparams.embeddings_nextn = value;
        self.cparams.embeddings_nextn_masked = masked;
        if self.cparams.embeddings_nextn_dev && (!value || masked) {
            self.cparams.embeddings_nextn_dev = false;
            self.h_nextn_dev = None;
        }
    }

    /// `llama_set_embeddings_layer_inp`.
    pub fn set_embeddings_layer_inp(&mut self, lid: u32, enable: bool) {
        if lid as usize >= self.cparams.embeddings_layer_inp.len() {
            crate::llama_log_error!("set_embeddings_layer_inp: layer id out of range");
            return;
        }
        self.cparams.embeddings_layer_inp[lid as usize] = enable;
        self.sched_need_reserve = true;
        if self.cparams.embeddings_layer_inp_dev {
            self.layer_inp_dev_alloc();
        }
    }

    /// `llama_set_embeddings_layer_inp_dev`.
    pub fn set_embeddings_layer_inp_dev(&mut self, value: bool) {
        if self.cparams.embeddings_layer_inp_dev != value {
            self.sched_need_reserve = true;
        }
        self.cparams.embeddings_layer_inp_dev = value;
        self.layer_inp_dev_alloc();
    }

    /// Allocate the host stand-in for the layer-input device buffer.
    pub fn layer_inp_dev_alloc(&mut self) {
        self.synchronize();
        self.layer_inp_dev = None;
        if !self.cparams.embeddings_layer_inp_dev {
            return;
        }
        let n_layer_inp = self
            .cparams
            .embeddings_layer_inp
            .iter()
            .filter(|&&v| v)
            .count();
        if n_layer_inp == 0 {
            return;
        }
        let n_embd = self.model.hparams.n_embd.max(1) as usize;
        let n_batch = self.cparams.n_batch.max(1) as usize;
        self.layer_inp_dev = Some(vec![0.0f32; n_layer_inp * n_embd * n_batch]);
    }

    /// Allocate the h_nextn host buffer. False when the context has no output device.
    pub fn h_nextn_dev_alloc(&mut self) -> bool {
        self.synchronize();
        self.h_nextn_dev = None;
        if !self.cparams.embeddings_nextn_dev {
            return true;
        }
        if self.backends.is_empty() {
            crate::llama_log_warn!("h_nextn_dev_alloc: no output backend, using host rows");
            self.cparams.embeddings_nextn_dev = false;
            return false;
        }
        let n_embd = self.model.hparams.n_embd_out().max(1) as usize;
        let rows = self.cparams.n_batch.max(1) as usize + self.cparams.n_seq_max.max(1) as usize;
        self.h_nextn_dev = Some(vec![0.0f32; n_embd * rows]);
        true
    }

    /// `llama_set_embeddings_nextn_dev`.
    pub fn set_embeddings_nextn_dev(&mut self, value: bool) -> bool {
        if value && (!self.cparams.embeddings_nextn || self.cparams.embeddings_nextn_masked) {
            return false;
        }
        if self.cparams.embeddings_nextn_dev != value {
            self.sched_need_reserve = true;
        }
        self.cparams.embeddings_nextn_dev = value;
        self.h_nextn_dev_alloc() && self.cparams.embeddings_nextn_dev == value
    }

    /// `llama_get_embeddings_nextn_rows`.
    pub fn get_h_nextn_rows(&self, row0: i64, n_rows: i64, out: &mut [f32]) -> bool {
        let Some(buf) = self.h_nextn_dev.as_ref() else {
            return false;
        };
        if row0 < 0 || n_rows < 0 {
            return false;
        }
        let n_embd = self.model.hparams.n_embd_out().max(1) as usize;
        let n_total = buf.len() / n_embd;
        if (row0 + n_rows) as usize > n_total {
            return false;
        }
        let n = n_rows as usize * n_embd;
        if out.len() < n {
            return false;
        }
        let begin = row0 as usize * n_embd;
        out[..n].copy_from_slice(&buf[begin..begin + n]);
        true
    }

    /// `llama_set_embeddings_nextn_rows`.
    pub fn set_h_nextn_rows(&mut self, row0: i64, n_rows: i64, data: &[f32]) -> bool {
        let Some(buf) = self.h_nextn_dev.as_mut() else {
            return false;
        };
        if row0 < 0 || n_rows < 0 {
            return false;
        }
        let n_embd = self.model.hparams.n_embd_out().max(1) as usize;
        let n_total = buf.len() / n_embd;
        if (row0 + n_rows) as usize > n_total {
            return false;
        }
        let n = n_rows as usize * n_embd;
        if data.len() < n {
            return false;
        }
        let begin = row0 as usize * n_embd;
        buf[begin..begin + n].copy_from_slice(&data[..n]);
        true
    }

    /// `llama_supports_embd_row`.
    pub fn supports_embd_row(&self) -> bool {
        if self.model.arch != super::model::Arch::Dflash {
            return false;
        }
        let Some(other) = self.params.ctx_other.as_ref() else {
            return false;
        };
        if other.borrow().layer_inp_dev.is_none() {
            return false;
        }
        !self.backends.is_empty()
    }

    /// `llama_set_ctx_state_src`.
    pub fn set_ctx_state_src(&mut self, src: Option<Rc<RefCell<LlamaContext>>>) -> bool {
        let changed = match (self.params.ctx_state_src.as_ref(), src.as_ref()) {
            (Some(a), Some(b)) => !Rc::ptr_eq(a, b),
            (None, None) => false,
            _ => true,
        };
        if changed {
            self.sched_need_reserve = true;
        }
        self.params.ctx_state_src = src;
        self.supports_state_row()
    }

    /// `supports_state_row`: MTP contexts read h_nextn rows from ctx_state_src.
    pub fn supports_state_row(&self) -> bool {
        if self.cparams.ctx_type != ContextType::Mtp {
            return false;
        }
        let Some(src) = self.params.ctx_state_src.as_ref() else {
            return false;
        };
        let src = src.borrow();
        if src.h_nextn_dev.is_none() {
            return false;
        }
        !self.backends.is_empty()
    }

    /// `llama_set_nextn_layer_offset`.
    pub fn set_nextn_layer_offset(&mut self, offset: i32) {
        self.cparams.nextn_layer_offset = offset;
    }

    /// `llama_set_causal_attn`.
    pub fn set_causal_attn(&mut self, value: bool) {
        // Graph shapes must not depend on causal_attn, so no reserve.
        self.cparams.causal_attn = value;
    }

    /// `llama_get_causal_attn`.
    pub fn get_causal_attn(&self) -> bool {
        self.cparams.causal_attn
    }

    /// `llama_set_sampler`.
    pub fn set_sampler(&mut self, seq_id: SeqId, sampler: Option<BackendSampler>) -> bool {
        if sampler.is_none() && !self.sampling.samplers.contains_key(&seq_id) {
            return true;
        }
        if sampler.is_some() && self.model.split_mode == super::model::SplitMode::Tensor {
            crate::llama_log_warn!(
                "set_sampler: backend sampling not supported with tensor split, using CPU"
            );
            if self.sampling.samplers.remove(&seq_id).is_some() {
                self.sched_need_reserve = true;
            }
            return false;
        }

        let can_offload = sampler
            .as_ref()
            .map(|s| s.can_offload && s.n_chain > 0)
            .unwrap_or(false);

        match sampler {
            Some(sampler) if can_offload => {
                self.sampling.samplers.insert(seq_id, sampler);
                self.sched_need_reserve = true;
                true
            }
            Some(sampler) => {
                crate::llama_log_warn!(
                    "set_sampler: sampler '{}' for seq_id {} cannot be offloaded",
                    sampler.name,
                    seq_id
                );
                if self.sampling.samplers.remove(&seq_id).is_some() {
                    self.sched_need_reserve = true;
                }
                false
            }
            None => {
                self.sampling.samplers.remove(&seq_id);
                self.sched_need_reserve = true;
                true
            }
        }
    }

    /// `llama_set_adapters_lora`.
    pub fn set_adapters_lora(&mut self, adapters: &[(usize, f32)]) {
        if self.adapters_lora_are_same(adapters) {
            return;
        }
        self.loras = adapters
            .iter()
            .filter(|(_, scale)| *scale != 0.0)
            .map(|&(id, scale)| LoraRef { id, scale })
            .collect();
        self.sched_need_reserve = true;
    }

    /// `adapters_lora_are_same`.
    pub fn adapters_lora_are_same(&self, adapters: &[(usize, f32)]) -> bool {
        let mut n_non_zero = 0usize;
        for &(id, scale) in adapters {
            if scale == 0.0 {
                continue;
            }
            n_non_zero += 1;
            if !self.loras.iter().any(|l| l.id == id && l.scale == scale) {
                return false;
            }
        }
        n_non_zero == self.loras.len()
    }

    /// `llama_set_adapter_cvec`.
    pub fn set_adapter_cvec(
        &mut self,
        data: &[f32],
        len: usize,
        n_embd: i32,
        il_start: i32,
        il_end: i32,
    ) -> Result<()> {
        let model: Model = self.model.clone();
        let res = self.cvec.apply(&model, data, len, n_embd, il_start, il_end);
        self.sched_need_reserve = true;
        res.map_err(|e| Error::new(e.message))
    }
}

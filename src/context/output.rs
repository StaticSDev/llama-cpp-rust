//! Output buffers, sampling buffers and the public output getters.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::batch::Ubatch;
use crate::core::{Error, Result, SeqId, Token, LLAMA_TOKEN_NULL};
use crate::cparams::PoolingType;

use super::graph::GraphResult;
use super::params::BackendSampler;
use super::LlamaContext;

/// `buffer_view<T>` over one host arena. Safe Rust cannot hand out raw
/// pointers, so getters return owned copies while the arena stays shared.
#[derive(Clone, Debug, Default)]
pub struct BufferView<T: Copy + Default> {
    pub data: Rc<RefCell<Vec<T>>>,
    pub offset: usize,
    pub size: usize,
}

impl<T: Copy + Default> BufferView<T> {
    pub fn new(data: Rc<RefCell<Vec<T>>>, offset: usize, size: usize) -> Self {
        Self { data, offset, size }
    }

    pub fn has_data(&self) -> bool {
        self.size > 0 && !self.data.borrow().is_empty()
    }

    pub fn as_vec(&self) -> Vec<T> {
        let arena = self.data.borrow();
        let end = (self.offset + self.size).min(arena.len());
        if self.offset < end {
            arena[self.offset..end].to_vec()
        } else {
            Vec::new()
        }
    }

    pub fn get(&self, i: usize) -> Option<T> {
        if i >= self.size {
            return None;
        }
        self.data.borrow().get(self.offset + i).copied()
    }

    pub fn set(&mut self, i: usize, value: T) {
        if i < self.size {
            if let Some(slot) = self.data.borrow_mut().get_mut(self.offset + i) {
                *slot = value;
            }
        }
    }

    pub fn fill(&mut self, value: T) {
        let mut arena = self.data.borrow_mut();
        let end = (self.offset + self.size).min(arena.len());
        if self.offset < end {
            arena[self.offset..end].fill(value);
        }
    }

    pub fn swap(&mut self, i: usize, j: usize) {
        if i == j || i >= self.size || j >= self.size {
            return;
        }
        let mut arena = self.data.borrow_mut();
        arena.swap(self.offset + i, self.offset + j);
    }

    pub fn swap_row(&mut self, i0: usize, i1: usize, stride: usize) {
        if i0 == i1 || stride == 0 {
            return;
        }
        let mut arena = self.data.borrow_mut();
        for k in 0..stride {
            let a = self.offset + i0 * stride + k;
            let b = self.offset + i1 * stride + k;
            if a < arena.len() && b < arena.len() {
                arena.swap(a, b);
            }
        }
    }
}

/// `sampling_info`.
#[derive(Clone, Debug, Default)]
pub struct SamplingInfo {
    pub samplers: BTreeMap<SeqId, BackendSampler>,
    pub logits: BufferView<f32>,
    pub sampled: BufferView<Token>,
    pub probs: BufferView<f32>,
    pub candidates: BufferView<Token>,
    pub logits_count: Vec<u32>,
    pub probs_count: Vec<u32>,
    pub candidates_count: Vec<u32>,
    pub token_ids_full_vocab: Vec<Token>,
}

/// Copy one graph tensor row block into a host view.
pub(crate) fn copy_tensor_rows(
    res: &GraphResult,
    id: Option<usize>,
    dst: &mut BufferView<f32>,
    row_offset: usize,
    stride: usize,
    mut counts: Option<&mut Vec<u32>>,
) {
    let data = res.gf.tensor_data(id);
    if !dst.has_data() || stride == 0 {
        return;
    }
    for (i, chunk) in data.chunks(stride).enumerate() {
        let row = row_offset + i;
        if row >= dst.size {
            break;
        }
        {
            let mut arena = dst.data.borrow_mut();
            let off = dst.offset + row * stride;
            let end = (off + chunk.len()).min(arena.len());
            if off < end {
                arena[off..end].copy_from_slice(&chunk[..end - off]);
            }
        }
        if let Some(counts) = counts.as_deref_mut() {
            if row < counts.len() {
                counts[row] = chunk.len() as u32;
            }
        }
    }
}

/// Copy token sampling tensors into a token view.
pub(crate) fn copy_token_rows(
    res: &GraphResult,
    id: Option<usize>,
    dst: &mut BufferView<Token>,
    row_offset: usize,
) {
    let data = res.gf.tensor_data(id);
    if !dst.has_data() {
        return;
    }
    for (i, &value) in data.iter().enumerate() {
        let row = row_offset + i;
        if row >= dst.size {
            break;
        }
        dst.set(row, value as Token);
    }
}

impl LlamaContext {
    /// `output_reserve`: grow output arenas and rebind every view.
    pub fn output_reserve(&mut self, n_outputs: i32) -> u32 {
        let n_outputs = n_outputs.max(self.cparams.n_seq_max as i32).max(1) as usize;
        let n_batch = self.cparams.n_batch.max(1) as usize;
        let n_vocab = self.model.vocab.n_tokens.max(1) as usize;
        let n_embd = self.model.hparams.n_embd.max(1) as usize;
        let n_embd_out = self.model.hparams.n_embd_out().max(1) as usize;

        let mut has_logits = !self.cparams.embeddings || self.graph_has_logits;
        let has_embd = self.cparams.embeddings;
        let has_embd_nextn = self.cparams.embeddings_nextn;
        if self.model.arch == super::model::Arch::T5 {
            has_logits = true;
        }

        let logits_size = if has_logits { n_vocab * n_outputs } else { 0 };
        let embd_size = if has_embd { n_embd_out * n_outputs } else { 0 };
        let mut embd_nextn_size = if has_embd_nextn {
            n_embd_out * n_outputs
        } else {
            0
        };
        if has_embd_nextn && !self.cparams.embeddings_nextn_masked {
            embd_nextn_size = n_embd_out * n_batch;
        }

        let mut layer_sizes = Vec::with_capacity(self.cparams.embeddings_layer_inp.len());
        for &enabled in &self.cparams.embeddings_layer_inp {
            if enabled && !self.cparams.embeddings_layer_inp_dev {
                layer_sizes.push(n_embd * n_batch);
            } else {
                layer_sizes.push(0);
            }
        }

        let has_sampling = !self.sampling.samplers.is_empty();
        let backend_float = if has_sampling {
            2 * n_vocab * n_outputs
        } else {
            0
        };
        let backend_token = if has_sampling {
            (1 + n_vocab) * n_outputs
        } else {
            0
        };

        let need_f32 = logits_size
            + embd_size
            + embd_nextn_size
            + layer_sizes.iter().sum::<usize>()
            + backend_float;
        if self.output_f32.borrow().len() < need_f32 {
            *self.output_f32.borrow_mut() = vec![0.0f32; need_f32];
        } else {
            self.output_f32.borrow_mut().fill(0.0);
        }
        if self.output_tokens.borrow().len() < backend_token {
            *self.output_tokens.borrow_mut() = vec![LLAMA_TOKEN_NULL; backend_token];
        } else {
            self.output_tokens.borrow_mut().fill(LLAMA_TOKEN_NULL);
        }

        let mut off = 0usize;
        self.logits = BufferView::new(
            Rc::clone(&self.output_f32),
            off,
            if has_logits { logits_size } else { 0 },
        );
        off += logits_size;
        self.embd = BufferView::new(
            Rc::clone(&self.output_f32),
            off,
            if has_embd { embd_size } else { 0 },
        );
        off += embd_size;
        self.embd_nextn = BufferView::new(
            Rc::clone(&self.output_f32),
            off,
            if has_embd_nextn { embd_nextn_size } else { 0 },
        );
        off += embd_nextn_size;

        self.embd_layer_inp.clear();
        for &size in &layer_sizes {
            self.embd_layer_inp
                .push(BufferView::new(Rc::clone(&self.output_f32), off, size));
            off += size;
        }

        if has_sampling {
            self.sampling.logits =
                BufferView::new(Rc::clone(&self.output_f32), off, n_vocab * n_outputs);
            off += n_vocab * n_outputs;
            self.sampling.probs =
                BufferView::new(Rc::clone(&self.output_f32), off, n_vocab * n_outputs);
            self.sampling.sampled = BufferView::new(Rc::clone(&self.output_tokens), 0, n_outputs);
            self.sampling.candidates = BufferView::new(
                Rc::clone(&self.output_tokens),
                n_outputs,
                n_vocab * n_outputs,
            );
            self.sampling.logits_count = vec![0; n_outputs];
            self.sampling.probs_count = vec![0; n_outputs];
            self.sampling.candidates_count = vec![0; n_outputs];
            self.sampling.sampled.fill(LLAMA_TOKEN_NULL);
        } else {
            self.sampling.logits = BufferView::default();
            self.sampling.probs = BufferView::default();
            self.sampling.sampled = BufferView::default();
            self.sampling.candidates = BufferView::default();
            self.sampling.logits_count.clear();
            self.sampling.probs_count.clear();
            self.sampling.candidates_count.clear();
        }

        if self.output_ids.is_empty() {
            self.output_ids = vec![-1; n_batch];
        }
        self.n_outputs = 0;
        n_outputs as u32
    }

    /// Map a batch token index to an output row.
    pub fn output_resolve_row(&self, i: i32) -> Result<usize> {
        let n_outputs = self.n_outputs.max(0) as usize;
        let j: i64 = if i < 0 {
            let j = n_outputs as i64 + i as i64;
            if j < 0 {
                return Err(Error::new(format!(
                    "negative index out of range [0, {})",
                    n_outputs
                )));
            }
            j
        } else if i as usize >= self.output_ids.len() {
            return Err(Error::new(format!(
                "out of range [0, {})",
                self.output_ids.len()
            )));
        } else {
            self.output_ids[i as usize]
        };
        if j < 0 {
            return Err(Error::new(format!("batch.logits[{}] != true", i)));
        }
        if j as usize >= n_outputs {
            return Err(Error::new(format!(
                "corrupt output buffer (j={}, n_outputs={})",
                j, n_outputs
            )));
        }
        Ok(j as usize)
    }

    /// Apply the deferred output swaps and token-row permutations.
    pub fn output_reorder(&mut self) {
        let n_vocab = self.model.vocab.n_tokens.max(1) as usize;
        let n_embd = self.model.hparams.n_embd.max(1) as usize;
        let n_embd_out = self.model.hparams.n_embd_out().max(1) as usize;

        for &(i0, i1) in &self.output_swaps {
            let (i0, i1) = (i0 as usize, i1 as usize);
            if self.logits.size > 0 {
                self.logits.swap_row(i0, i1, n_vocab);
            }
            if self.embd.size > 0 {
                self.embd.swap_row(i0, i1, n_embd_out);
            }
            if self.embd_nextn.size > 0 && self.cparams.embeddings_nextn_masked {
                self.embd_nextn.swap_row(i0, i1, n_embd_out);
            }
            if !self.sampling.samplers.is_empty() {
                self.sampling.logits.swap_row(i0, i1, n_vocab);
                self.sampling.probs.swap_row(i0, i1, n_vocab);
                self.sampling.candidates.swap_row(i0, i1, n_vocab);
                self.sampling.sampled.swap(i0, i1);
                if i0 < self.sampling.logits_count.len() && i1 < self.sampling.logits_count.len() {
                    self.sampling.logits_count.swap(i0, i1);
                    self.sampling.probs_count.swap(i0, i1);
                    self.sampling.candidates_count.swap(i0, i1);
                }
            }
        }
        self.output_swaps.clear();

        let mut i = 0usize;
        while i < self.embd_batch_idxs.len() {
            let j = self.embd_batch_idxs[i];
            if j == i {
                i += 1;
                continue;
            }
            if j >= self.embd_batch_idxs.len() {
                break;
            }
            if self.embd_nextn.has_data() && !self.cparams.embeddings_nextn_masked {
                self.embd_nextn.swap_row(i, j, n_embd_out);
            }
            for layer in &mut self.embd_layer_inp {
                if layer.has_data() {
                    layer.swap_row(i, j, n_embd);
                }
            }
            self.embd_batch_idxs.swap(i, j);
        }
        self.embd_batch_idxs.clear();
    }

    /// Copy enabled layer-input tensors into host rows.
    pub fn extract_layer_inputs(
        &mut self,
        res: &GraphResult,
        token_offset: usize,
        n_tokens: usize,
    ) -> bool {
        if self.cparams.embeddings_layer_inp_dev || n_tokens == 0 {
            return false;
        }
        let mut extracted = false;
        let n_embd = self.model.hparams.n_embd.max(1) as usize;
        for il in 0..self.cparams.embeddings_layer_inp.len() {
            if !self.cparams.embeddings_layer_inp[il] {
                continue;
            }
            let Some(tensor) = res.get_layer_inp(il) else {
                continue;
            };
            let data = tensor.data.clone();
            let row_floats = n_embd;
            let dst_offset = token_offset * row_floats;
            let nfloats = data.len().min(n_tokens * row_floats);
            if let Some(dst) = self.embd_layer_inp.get_mut(il) {
                let mut arena = dst.data.borrow_mut();
                let begin = dst.offset + dst_offset;
                let end = (begin + nfloats).min(arena.len());
                if begin < end {
                    arena[begin..end].copy_from_slice(&data[..end - begin]);
                }
            }
            extracted = true;
        }
        extracted
    }

    /// Copy logits out of a processed graph.
    pub fn extract_logits(&mut self, res: &GraphResult, n_outputs_prev: usize, n_outputs: usize) {
        if n_outputs == 0 {
            return;
        }
        let n_vocab = self.model.vocab.n_tokens.max(1) as usize;
        let mut tmp = BufferView::new(Rc::clone(&self.output_f32), 0, 0);
        // Copy directly into the logits view at the row offset.
        if !self.logits.has_data() {
            return;
        }
        tmp.data = Rc::clone(&self.logits.data);
        tmp.offset = self.logits.offset + n_outputs_prev * n_vocab;
        tmp.size = n_outputs * n_vocab;
        copy_tensor_rows(res, res.logits, &mut tmp, 0, n_vocab, None);
    }

    /// Copy token or pooled embeddings out of a processed graph.
    pub fn extract_embeddings(
        &mut self,
        res: &GraphResult,
        ubatch: &Ubatch,
        n_outputs_prev: usize,
        n_outputs: usize,
    ) {
        let n_embd_out = self.model.hparams.n_embd_out().max(1) as usize;
        let source = if res.get_embd_pooled().is_some() {
            res.embd_pooled
        } else {
            res.embd
        };
        match self.cparams.pooling_type {
            PoolingType::None => {
                if n_outputs == 0 || !self.embd.has_data() {
                    return;
                }
                let data = res.gf.tensor_data(source);
                let mut arena = self.embd.data.borrow_mut();
                let off = self.embd.offset + n_outputs_prev * n_embd_out;
                let n = data.len().min(n_outputs * n_embd_out);
                let end = (off + n).min(arena.len());
                if off < end {
                    arena[off..end].copy_from_slice(&data[..end - off]);
                }
            }
            PoolingType::Mean | PoolingType::Cls | PoolingType::Last | PoolingType::Rank => {
                let data = res.gf.tensor_data(source);
                let width = if self.cparams.pooling_type == PoolingType::Rank {
                    self.model.hparams.n_cls_out.max(1) as usize
                } else {
                    n_embd_out
                };
                for &seq_id in &ubatch.seq_id_unq {
                    let idx = ubatch.seq_idx.get(seq_id as usize).copied().unwrap_or(-1);
                    if idx < 0 {
                        continue;
                    }
                    let mut row = vec![0.0f32; width];
                    let begin = idx as usize * width;
                    for (j, v) in row.iter_mut().enumerate() {
                        *v = data.get(begin + j).copied().unwrap_or(0.0);
                    }
                    self.embd_seq.insert(seq_id, row);
                }
            }
            PoolingType::Unspecified => {}
        }
    }

    /// Copy nextn hidden rows out of a processed graph.
    pub fn extract_nextn(
        &mut self,
        res: &GraphResult,
        n_outputs_prev: usize,
        n_tokens_prev: usize,
        n_outputs: usize,
        n_tokens: usize,
    ) -> bool {
        let masked = self.cparams.embeddings_nextn_masked;
        let n_rows = if masked { n_outputs } else { n_tokens };
        let offset = if masked {
            n_outputs_prev
        } else {
            n_tokens_prev
        };
        if n_rows == 0
            || !self.embd_nextn.has_data()
            || self.cparams.pooling_type != PoolingType::None
        {
            return false;
        }
        let data = res.gf.tensor_data(res.h_nextn);
        let n_embd = self.model.hparams.n_embd_out().max(1) as usize;
        let mut arena = self.embd_nextn.data.borrow_mut();
        let begin = self.embd_nextn.offset + offset * n_embd;
        let n = data.len().min(n_rows * n_embd);
        let end = (begin + n).min(arena.len());
        if begin < end {
            arena[begin..end].copy_from_slice(&data[..end - begin]);
        }
        !masked
    }

    /// Copy backend sampling tensors into host buffers.
    pub fn extract_sampling(&mut self, res: &GraphResult, n_outputs_prev: usize) {
        if self.sampling.samplers.is_empty() {
            return;
        }
        let n_vocab = self.model.vocab.n_tokens.max(1) as usize;
        copy_token_rows(
            res,
            res.t_sampled,
            &mut self.sampling.sampled,
            n_outputs_prev,
        );
        {
            let mut counts = std::mem::take(&mut self.sampling.logits_count);
            copy_tensor_rows(
                res,
                res.t_sampled_logits,
                &mut self.sampling.logits,
                n_outputs_prev,
                n_vocab,
                Some(&mut counts),
            );
            self.sampling.logits_count = counts;
        }
        {
            let mut counts = std::mem::take(&mut self.sampling.probs_count);
            copy_tensor_rows(
                res,
                res.t_sampled_probs,
                &mut self.sampling.probs,
                n_outputs_prev,
                n_vocab,
                Some(&mut counts),
            );
            self.sampling.probs_count = counts;
        }
        {
            let mut counts = std::mem::take(&mut self.sampling.candidates_count);
            let data = res.gf.tensor_data(res.t_candidates);
            let row = n_outputs_prev * n_vocab;
            let mut arena = self.sampling.candidates.data.borrow_mut();
            let n = data.len().min(n_vocab.saturating_sub(row));
            let end = (self.sampling.candidates.offset + row + n).min(arena.len());
            let begin = self.sampling.candidates.offset + row;
            if begin < end {
                for (i, &v) in data.iter().take(end - begin).enumerate() {
                    arena[begin + i] = v as Token;
                }
            }
            if n_outputs_prev < counts.len() {
                counts[n_outputs_prev] = n as u32;
            }
            self.sampling.candidates_count = counts;
        }
    }

    /// `llama_get_logits`.
    pub fn get_logits(&mut self) -> Vec<f32> {
        self.synchronize();
        self.output_reorder();
        self.logits.as_vec()
    }

    /// `llama_get_logits_ith`, falling back to sampled logits.
    pub fn get_logits_ith(&mut self, i: i32) -> Option<Vec<f32>> {
        self.synchronize();
        if let Some(sampled) = self.get_sampled_logits_ith(i) {
            return Some(sampled);
        }
        self.output_reorder();
        if !self.logits.has_data() {
            return None;
        }
        let j = self.output_resolve_row(i).ok()?;
        let n_vocab = self.model.vocab.n_tokens.max(1) as usize;
        let data = self.logits.as_vec();
        data.get(j * n_vocab..(j + 1) * n_vocab).map(|s| s.to_vec())
    }

    /// `llama_get_embeddings`.
    pub fn get_embeddings(&mut self) -> Vec<f32> {
        self.synchronize();
        self.output_reorder();
        self.embd.as_vec()
    }

    /// `llama_get_embeddings_ith`.
    pub fn get_embeddings_ith(&mut self, i: i32) -> Option<Vec<f32>> {
        self.synchronize();
        self.output_reorder();
        if !self.embd.has_data() {
            return None;
        }
        let j = self.output_resolve_row(i).ok()?;
        let n = self.model.hparams.n_embd_out().max(1) as usize;
        let data = self.embd.as_vec();
        data.get(j * n..(j + 1) * n).map(|s| s.to_vec())
    }

    /// `llama_get_embeddings_seq`.
    pub fn get_embeddings_seq(&self, seq_id: SeqId) -> Option<Vec<f32>> {
        self.embd_seq.get(&seq_id).cloned()
    }

    /// `llama_get_embeddings_nextn`.
    pub fn get_embeddings_nextn(&mut self) -> Vec<f32> {
        self.synchronize();
        self.output_reorder();
        self.embd_nextn.as_vec()
    }

    /// `llama_get_embeddings_nextn_ith`.
    pub fn get_embeddings_nextn_ith(&mut self, i: i32) -> Option<Vec<f32>> {
        self.synchronize();
        self.output_reorder();
        let n = self.model.hparams.n_embd_out().max(1) as usize;
        let data = self.embd_nextn.as_vec();
        if !self.cparams.embeddings_nextn_masked {
            if i < 0 || (i as usize + 1) * n > data.len() {
                return None;
            }
            return data
                .get(i as usize * n..(i as usize + 1) * n)
                .map(|s| s.to_vec());
        }
        if data.is_empty() {
            return None;
        }
        let j = self.output_resolve_row(i).ok()?;
        data.get(j * n..(j + 1) * n).map(|s| s.to_vec())
    }

    /// `llama_get_embeddings_layer_inp`.
    pub fn get_embeddings_layer_inp(&mut self, lid: u32) -> Option<Vec<f32>> {
        self.synchronize();
        self.output_reorder();
        self.embd_layer_inp
            .get(lid as usize)
            .filter(|v| v.has_data())
            .map(|v| v.as_vec())
    }

    /// `llama_get_sampled_tokens`.
    pub fn get_sampled_tokens(&self) -> Vec<Token> {
        self.sampling.sampled.as_vec()
    }

    /// `llama_get_sampled_token_ith`.
    pub fn get_sampled_token_ith(&mut self, idx: i32) -> Token {
        self.synchronize();
        self.output_reorder();
        if !self.sampling.sampled.has_data() {
            return LLAMA_TOKEN_NULL;
        }
        match self.output_resolve_row(idx) {
            Ok(row) => self.sampling.sampled.get(row).unwrap_or(LLAMA_TOKEN_NULL),
            Err(err) => {
                crate::llama_log_error!("invalid backend sampled token id {}: {}", idx, err);
                LLAMA_TOKEN_NULL
            }
        }
    }

    fn sampled_row(&mut self, idx: i32) -> Option<usize> {
        self.synchronize();
        self.output_reorder();
        self.output_resolve_row(idx).ok()
    }

    /// `llama_get_sampled_probs_ith`.
    pub fn get_sampled_probs_ith(&mut self, idx: i32) -> Option<Vec<f32>> {
        if !self.sampling.probs.has_data() {
            return None;
        }
        let row = self.sampled_row(idx)?;
        if row >= self.sampling.probs_count.len() || self.sampling.probs_count[row] == 0 {
            return None;
        }
        let n_vocab = self.model.vocab.n_tokens.max(1) as usize;
        let data = self.sampling.probs.as_vec();
        data.get(row * n_vocab..(row + 1) * n_vocab)
            .map(|s| s.to_vec())
    }

    /// `llama_get_sampled_logits_ith`.
    pub fn get_sampled_logits_ith(&mut self, idx: i32) -> Option<Vec<f32>> {
        if !self.sampling.logits.has_data() {
            return None;
        }
        let row = self.sampled_row(idx)?;
        if row >= self.sampling.logits_count.len() || self.sampling.logits_count[row] == 0 {
            return None;
        }
        let n_vocab = self.model.vocab.n_tokens.max(1) as usize;
        let data = self.sampling.logits.as_vec();
        data.get(row * n_vocab..(row + 1) * n_vocab)
            .map(|s| s.to_vec())
    }

    /// `llama_get_sampled_candidates_ith`.
    pub fn get_sampled_candidates_ith(&mut self, idx: i32) -> Vec<Token> {
        let row = self.sampled_row(idx);
        let n_vocab = self.model.vocab.n_tokens.max(1) as usize;
        if let Some(row) = row {
            if self.sampling.candidates.has_data()
                && row < self.sampling.candidates_count.len()
                && self.sampling.candidates_count[row] > 0
            {
                let data = self.sampling.candidates.as_vec();
                if let Some(slice) = data.get(row * n_vocab..(row + 1) * n_vocab) {
                    return slice.to_vec();
                }
            }
        }
        self.sampling.token_ids_full_vocab.clone()
    }

    /// `llama_get_sampled_candidates_count_ith`.
    pub fn get_sampled_candidates_count(&mut self, idx: i32) -> u32 {
        if !self.sampling.candidates.has_data() {
            return 0;
        }
        match self.sampled_row(idx) {
            Some(row) => self
                .sampling
                .candidates_count
                .get(row)
                .copied()
                .unwrap_or(0),
            None => 0,
        }
    }

    /// `llama_get_sampled_logits_count_ith`.
    pub fn get_sampled_logits_count(&mut self, idx: i32) -> u32 {
        if !self.sampling.logits.has_data() {
            return self.model.vocab.n_tokens;
        }
        match self.sampled_row(idx) {
            Some(row) => self.sampling.logits_count.get(row).copied().unwrap_or(0),
            None => 0,
        }
    }

    /// `llama_get_sampled_probs_count_ith`.
    pub fn get_sampled_probs_count(&mut self, idx: i32) -> u32 {
        if !self.sampling.probs.has_data() {
            return 0;
        }
        match self.sampled_row(idx) {
            Some(row) => self.sampling.probs_count.get(row).copied().unwrap_or(0),
            None => 0,
        }
    }
}

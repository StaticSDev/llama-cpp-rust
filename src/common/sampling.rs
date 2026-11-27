//! Rust port of `common/sampling.h` and `common/sampling.cpp`.
//!
//! C++ handles become local substitutes: [`SamplingContext`] supplies the
//! logits of one output row, the sampler chain comes from `sampler`, and
//! grammar constraints are a `dyn Grammar`. No reasoning-budget sampler
//! exists in this crate, so its calls are omitted.

use std::cmp::Ordering;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::context::PerfContextData;
use crate::core::{Error, Mt19937, Result, Token, LLAMA_TOKEN_NULL};
use crate::sampler::{
    get_seed, perf_sampler, Chain, Grammar, RingBuffer, Sampler, TokenData, TokenDataArray,
};
use crate::vocab::LlamaVocab;

use super::{
    common_token_to_piece, common_tokenize, GrammarTrigger, GrammarTriggerType, ParamsSampling,
    SamplerType,
};

/// Supplies the data a C++ `llama_context` gives to the common sampler.
pub trait SamplingContext {
    fn n_vocab(&self) -> usize;
    fn sampled_probs(&mut self, idx: i32) -> Option<Vec<f32>>;
    fn sampled_logits(&mut self, idx: i32) -> Option<Vec<f32>>;
    fn sampled_ids(&mut self, idx: i32) -> Vec<Token>;
    fn sampled_token(&mut self, idx: i32) -> Token;
    fn logits(&mut self, idx: i32) -> Option<Vec<f32>>;
    fn is_eog(&self, _token: Token) -> bool {
        false
    }
    fn synchronize(&mut self) {}
}

impl SamplingContext for crate::context::LlamaContext {
    fn n_vocab(&self) -> usize {
        self.model.vocab.n_tokens as usize
    }
    fn sampled_probs(&mut self, idx: i32) -> Option<Vec<f32>> {
        self.get_sampled_probs_ith(idx)
    }
    fn sampled_logits(&mut self, idx: i32) -> Option<Vec<f32>> {
        self.get_sampled_logits_ith(idx)
    }
    fn sampled_ids(&mut self, idx: i32) -> Vec<Token> {
        self.get_sampled_candidates_ith(idx)
    }
    fn sampled_token(&mut self, idx: i32) -> Token {
        self.get_sampled_token_ith(idx)
    }
    fn logits(&mut self, idx: i32) -> Option<Vec<f32>> {
        self.get_logits_ith(idx)
    }
    fn synchronize(&mut self) {
        crate::context::LlamaContext::synchronize(self)
    }
}

/// `common_sampler`. The chain is built by the caller because C++ needs a
/// model and vocab handle for several of its samplers.
pub struct CommonSampler {
    pub params: ParamsSampling,
    chain: Chain,
    grammar: Option<Box<dyn Grammar>>,
    prev: RingBuffer<Token>,
    cur_p: TokenDataArray,
    rng: Mt19937,
    t_total_us: i64,
}

impl CommonSampler {
    /// `common_sampler_init`, with the sampler chain supplied by the caller.
    pub fn new(
        chain: Chain,
        params: ParamsSampling,
        grammar: Option<Box<dyn Grammar>>,
    ) -> Result<Self> {
        validate_params(&params)?;
        let mut params = params;
        if grammar.is_some() {
            params.backend_sampling = false;
        }
        let seed = get_seed(&chain) ^ 0x9e37_79b9;
        let capacity = params.n_prev.max(32).max(1) as usize;
        Ok(Self {
            params,
            chain,
            grammar,
            prev: RingBuffer::new(capacity),
            cur_p: TokenDataArray::default(),
            rng: Mt19937::new(seed),
            t_total_us: 0,
        })
    }

    /// `common_sampler_accept`.
    pub fn accept(&mut self, token: Token, is_generated: bool) {
        let start = self.tm_start();
        if is_generated && self.grammar_should_apply() {
            if let Some(grammar) = self.grammar.as_mut() {
                grammar.accept(token);
            }
        }
        self.chain.accept(token);
        self.prev.push_back(token);
        self.tm_end(start);
    }

    /// `common_sampler_reset`. The grammar sampler keeps its state, as in C++.
    pub fn reset(&mut self) {
        let start = self.tm_start();
        self.prev.clear();
        self.chain.reset();
        self.tm_end(start);
    }

    /// `common_sampler_clone`.
    pub fn clone_sampler(&self) -> Self {
        self.clone()
    }

    /// `common_sampler_copy`. Grammar state is cloned because `Grammar` has no
    /// in-place copy operation.
    pub fn copy_from(&mut self, src: &Self) -> Result<()> {
        if self.grammar.is_some() != src.grammar.is_some() {
            return Err(Error::new("common_sampler_copy: grammar mismatch"));
        }
        if self.chain.len() != src.chain.len() {
            return Err(Error::new("common_sampler_copy: chain length mismatch"));
        }
        self.chain.copy_state(&src.chain);
        self.grammar = src.grammar.as_ref().map(|g| g.clone_box());
        self.params = src.params.clone();
        self.prev = src.prev.clone();
        self.cur_p = src.cur_p.clone();
        self.rng = src.rng.clone();
        self.t_total_us = src.t_total_us;
        Ok(())
    }

    /// `common_sampler_get`.
    pub fn chain(&self) -> &Chain {
        &self.chain
    }

    /// Mutable access to the underlying chain.
    pub fn chain_mut(&mut self) -> &mut Chain {
        &mut self.chain
    }

    /// `common_sampler_sample`.
    pub fn sample(
        &mut self,
        ctx: &mut dyn SamplingContext,
        idx: i32,
        grammar_first: bool,
    ) -> Result<Token> {
        ctx.synchronize();
        let start = self.tm_start();
        let result = self.sample_impl(ctx, idx, grammar_first);
        self.tm_end(start);
        result
    }

    /// `common_sampler_sample_and_accept_n`.
    pub fn sample_and_accept_n(
        &mut self,
        ctx: &mut dyn SamplingContext,
        idxs: &[i32],
        draft: &[Token],
        grammar_first: bool,
    ) -> Result<Vec<Token>> {
        if idxs.len() != draft.len() + 1 {
            return Err(Error::new("idxs.len() must be draft.size() + 1"));
        }
        let mut result = Vec::with_capacity(idxs.len());
        let mut i = 0usize;
        while i < draft.len() {
            let id = self.sample(ctx, idxs[i], grammar_first)?;
            self.accept(id, true);
            result.push(id);
            if draft[i] != id || (ctx.is_eog(id) && i + 1 < draft.len()) {
                break;
            }
            i += 1;
        }
        if i == draft.len() {
            let id = self.sample(ctx, idxs[i], grammar_first)?;
            self.accept(id, true);
            result.push(id);
        }
        Ok(result)
    }

    /// `common_sampler_sample_and_accept_n_rejection`.
    pub fn sample_and_accept_n_rejection(
        &mut self,
        ctx: &mut dyn SamplingContext,
        idxs: &[i32],
        draft: &[Token],
        draft_q: &[Vec<TokenData>],
        grammar_first: bool,
    ) -> Result<Vec<Token>> {
        if idxs.len() != draft.len() + 1 {
            return Err(Error::new("idxs.len() must be draft.size() + 1"));
        }
        if draft_q.len() != draft.len() {
            return Err(Error::new("draft_q must have one entry per draft token"));
        }

        let mut result = Vec::with_capacity(idxs.len());
        let mut i = 0usize;
        while i < draft.len() {
            let id_tgt = self.sample(ctx, idxs[i], grammar_first)?;
            self.sort_candidates();

            let masked = !grammar_first && self.grammar_should_apply();
            let mut cand: Vec<TokenData> = Vec::new();
            if masked {
                cand = self.cur_p.data.clone();
                let mut arr = TokenDataArray::new(cand);
                if let Some(grammar) = self.grammar.as_mut() {
                    grammar.apply(&mut arr);
                }
                cand = arr.data;
            }

            let cur = &self.cur_p.data;
            let p_raw = |k: usize| -> f32 {
                if masked
                    && cand
                        .get(k)
                        .map(|td| td.logit == f32::NEG_INFINITY)
                        .unwrap_or(false)
                {
                    0.0
                } else {
                    cur[k].p
                }
            };
            let mut p_sum = 0.0f32;
            if masked {
                for k in 0..cur.len() {
                    p_sum += p_raw(k);
                }
            }
            let p_norm = if masked && p_sum > 0.0 {
                1.0 / p_sum
            } else {
                1.0
            };
            let p_of = |k: usize| p_raw(k) * p_norm;

            let q = &draft_q[i];
            let q_x = prob_of(q, draft[i]);
            let mut p_x = 0.0f32;
            for (k, td) in cur.iter().enumerate() {
                if td.id == draft[i] {
                    p_x = p_of(k);
                    break;
                }
            }

            if q_x > 0.0 && (p_x >= q_x || self.rng.canonical_f32() < p_x / q_x) {
                self.accept(draft[i], true);
                result.push(draft[i]);
                i += 1;
                continue;
            }

            let mut residual: Vec<TokenData> = Vec::new();
            let mut sum = 0.0f32;
            for k in 0..cur.len() {
                let r = p_of(k) - prob_of(q, cur[k].id);
                if r > 0.0 {
                    residual.push(TokenData {
                        id: cur[k].id,
                        logit: 0.0,
                        p: r,
                    });
                    sum += r;
                }
            }

            let mut id = id_tgt;
            if sum > 0.0 {
                let mut u = self.rng.canonical_f32() * sum;
                for e in &residual {
                    id = e.id;
                    u -= e.p;
                    if u <= 0.0 {
                        break;
                    }
                }
            }
            self.accept(id, true);
            result.push(id);
            break;
        }

        if i == draft.len() {
            let id = self.sample(ctx, idxs[i], grammar_first)?;
            self.accept(id, true);
            result.push(id);
        }
        Ok(result)
    }

    /// `common_sampler_get_seed`.
    pub fn seed(&self) -> u32 {
        get_seed(&self.chain)
    }

    /// `common_sampler_reasoning_budget_force`: no reasoning-budget sampler.
    pub fn reasoning_budget_force(&self) -> bool {
        false
    }

    /// `common_sampler_get_candidates`.
    pub fn get_candidates(&mut self, do_sort: bool) -> &TokenDataArray {
        if do_sort {
            self.sort_candidates();
        }
        &self.cur_p
    }

    /// `common_sampler_last`.
    pub fn last(&self) -> Result<Token> {
        if self.prev.is_empty() {
            return Err(Error::new("ring buffer is empty"));
        }
        Ok(self.prev.rat(0))
    }

    /// `common_sampler_print`.
    pub fn print(&self) -> String {
        let mut result = String::from("logits ");
        for i in 0..self.chain.len() {
            if let Some(smpl) = self.chain.get(i) {
                result.push_str("-> ");
                result.push_str(smpl.name());
                result.push(' ');
            }
        }
        result
    }

    /// `common_sampler_prev_str`.
    pub fn prev_str(&self, vocab: &LlamaVocab, n: usize) -> Result<String> {
        let n = n.min(self.prev.len());
        if n == 0 {
            return Ok(String::new());
        }
        let mut result = String::with_capacity(8 * n);
        for i in (0..n).rev() {
            let id = self.prev.rat(i);
            if id == LLAMA_TOKEN_NULL {
                return Err(Error::new("null token in the sampling history"));
            }
            result.push_str(&common_token_to_piece(vocab, id, true));
        }
        Ok(result)
    }

    /// `common_perf_print`, with context counters supplied by the caller.
    pub fn perf_print(&self, ctx: Option<&PerfContextData>) {
        let t_sampling_ms = 1e-3 * self.t_total_us as f64;
        let data = perf_sampler(&self.chain);
        crate::llama_log_info!("sampling time = {:10.2} ms", t_sampling_ms);
        crate::llama_log_info!(
            "samplers time = {:10.2} ms / {:5} tokens",
            data.t_sample_ms,
            data.n_sample
        );

        if let Some(ctx) = ctx {
            let now_us = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_micros() as i64)
                .unwrap_or(0);
            let t_end_ms = 1e-3 * now_us as f64;
            let t_total_ms = t_end_ms - ctx.t_start_ms;
            let t_unacc_ms = t_total_ms - (t_sampling_ms + ctx.t_p_eval_ms + ctx.t_eval_ms);
            let t_unacc_pc = if t_total_ms != 0.0 {
                100.0 * t_unacc_ms / t_total_ms
            } else {
                0.0
            };
            crate::llama_log_info!("load time = {:10.2} ms", ctx.t_load_ms);
            crate::llama_log_info!(
                "prompt eval time = {:10.2} ms / {:5} tokens ({:8.2} ms per token, {:8.2} tokens per second)",
                ctx.t_p_eval_ms,
                ctx.n_p_eval,
                ctx.t_p_eval_ms / ctx.n_p_eval as f64,
                1e3 / ctx.t_p_eval_ms * ctx.n_p_eval as f64
            );
            crate::llama_log_info!(
                "eval time = {:10.2} ms / {:5} runs   ({:8.2} ms per token, {:8.2} tokens per second)",
                ctx.t_eval_ms,
                ctx.n_eval,
                ctx.t_eval_ms / ctx.n_eval as f64,
                1e3 / ctx.t_eval_ms * ctx.n_eval as f64
            );
            crate::llama_log_info!(
                "total time = {:10.2} ms / {:5} tokens",
                t_end_ms - ctx.t_start_ms,
                ctx.n_p_eval + ctx.n_eval
            );
            crate::llama_log_info!(
                "unaccounted time = {:10.2} ms / {:5.1} %      (total - sampling - prompt eval - eval) / (total)",
                t_unacc_ms,
                t_unacc_pc
            );
            crate::llama_log_info!("graphs reused = {:10}", ctx.n_reused);
            crate::llama_log_info!(
                "graph reallocs = {:10}    / {:5}        (same size) / (changed size)",
                ctx.n_reallocs_same,
                ctx.n_reallocs_diff
            );
        }
    }

    fn grammar_should_apply(&self) -> bool {
        // With no reasoning-budget sampler there is no thinking state to suppress grammar.
        self.grammar.is_some()
    }

    fn tm_start(&self) -> Option<Instant> {
        if self.params.no_perf {
            None
        } else {
            Some(Instant::now())
        }
    }

    fn tm_end(&mut self, start: Option<Instant>) {
        if let Some(start) = start {
            self.t_total_us += start.elapsed().as_micros() as i64;
        }
    }

    fn set_logits(&mut self, ctx: &mut dyn SamplingContext, idx: i32) -> Result<()> {
        let mut cur = Vec::new();
        if let Some(probs) = ctx.sampled_probs(idx) {
            let ids = ctx.sampled_ids(idx);
            let logits = ctx.sampled_logits(idx).unwrap_or_default();
            let n = probs.len().min(ids.len());
            for i in 0..n {
                cur.push(TokenData {
                    id: ids[i],
                    logit: logits.get(i).copied().unwrap_or(0.0),
                    p: probs[i],
                });
            }
        } else if let Some(logits) = ctx.sampled_logits(idx) {
            let ids = ctx.sampled_ids(idx);
            let n = logits.len().min(ids.len());
            for i in 0..n {
                cur.push(TokenData {
                    id: ids[i],
                    logit: logits[i],
                    p: 0.0,
                });
            }
        } else {
            let logits = ctx
                .logits(idx)
                .ok_or_else(|| Error::new("common_sampler_set_logits: no logits"))?;
            let n = logits.len().min(ctx.n_vocab());
            for i in 0..n {
                cur.push(TokenData {
                    id: i as Token,
                    logit: logits[i],
                    p: 0.0,
                });
            }
        }
        self.cur_p = TokenDataArray {
            data: cur,
            selected: -1,
            sorted: false,
        };
        Ok(())
    }

    fn sample_impl(
        &mut self,
        ctx: &mut dyn SamplingContext,
        idx: i32,
        grammar_first: bool,
    ) -> Result<Token> {
        self.set_logits(ctx, idx)?;

        let id = ctx.sampled_token(idx);
        if id != LLAMA_TOKEN_NULL {
            if let Some(pos) = self.cur_p.data.iter().position(|td| td.id == id) {
                self.cur_p.selected = pos as i32;
            }
            return Ok(id);
        }

        if grammar_first && self.grammar_should_apply() {
            if let Some(grammar) = self.grammar.as_mut() {
                grammar.apply(&mut self.cur_p);
            }
        }
        self.chain.apply(&mut self.cur_p);
        let id = self.selected_token()?;

        if grammar_first || !self.grammar_should_apply() {
            return Ok(id);
        }

        let mut single = TokenDataArray::new(vec![TokenData {
            id,
            logit: 1.0,
            p: 0.0,
        }]);
        if let Some(grammar) = self.grammar.as_mut() {
            grammar.apply(&mut single);
        }
        if single
            .data
            .first()
            .map(|td| td.logit != f32::NEG_INFINITY)
            .unwrap_or(false)
        {
            return Ok(id);
        }

        self.set_logits(ctx, idx)?;
        if self.grammar_should_apply() {
            if let Some(grammar) = self.grammar.as_mut() {
                grammar.apply(&mut self.cur_p);
            }
        }
        self.chain.apply(&mut self.cur_p);
        self.selected_token()
    }

    fn selected_token(&self) -> Result<Token> {
        let selected = self.cur_p.selected;
        if selected < 0 || selected as usize >= self.cur_p.data.len() {
            return Err(Error::new(
                "no selected token during sampling - check your sampling configuration",
            ));
        }
        Ok(self.cur_p.data[selected as usize].id)
    }

    fn sort_candidates(&mut self) {
        if self.cur_p.sorted {
            return;
        }
        let start = self.tm_start();
        let selected = if self.cur_p.selected >= 0 {
            self.cur_p
                .data
                .get(self.cur_p.selected as usize)
                .map(|td| td.id)
        } else {
            None
        };
        self.cur_p
            .data
            .sort_unstable_by(|a, b| b.p.partial_cmp(&a.p).unwrap_or(Ordering::Equal));
        if let Some(id) = selected {
            if let Some(pos) = self.cur_p.data.iter().position(|td| td.id == id) {
                self.cur_p.selected = pos as i32;
            }
        }
        self.cur_p.sorted = true;
        self.tm_end(start);
    }
}

impl Clone for CommonSampler {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            chain: self.chain.clone_chain(),
            grammar: self.grammar.as_ref().map(|g| g.clone_box()),
            prev: self.prev.clone(),
            cur_p: self.cur_p.clone(),
            rng: self.rng.clone(),
            t_total_us: self.t_total_us,
        }
    }
}

fn validate_params(params: &ParamsSampling) -> Result<()> {
    if !params.penalty_repeat.is_finite()
        || params.penalty_repeat <= 0.0
        || !(1.0f32 / params.penalty_repeat).is_finite()
    {
        return Err(Error::new(
            "penalty_repeat must be finite and greater than 0",
        ));
    }
    if !params.penalty_freq.is_finite() {
        return Err(Error::new("penalty_freq must be finite"));
    }
    if !params.penalty_present.is_finite() {
        return Err(Error::new("penalty_present must be finite"));
    }
    Ok(())
}

fn prob_of(data: &[TokenData], id: Token) -> f32 {
    for td in data {
        if td.id == id {
            return td.p;
        }
    }
    0.0
}

/// `common_sampler_types_from_chars`.
pub fn sampler_types_from_chars(chars: &str) -> Vec<SamplerType> {
    let all = [
        SamplerType::Dry,
        SamplerType::TopK,
        SamplerType::TypicalP,
        SamplerType::TopP,
        SamplerType::TopNSigma,
        SamplerType::MinP,
        SamplerType::Temperature,
        SamplerType::Xtc,
        SamplerType::Infill,
        SamplerType::Penalties,
        SamplerType::AdaptiveP,
    ];
    let mut result = Vec::new();
    for c in chars.chars() {
        if let Some(t) = all.iter().find(|t| t.as_char() == c) {
            result.push(*t);
        } else {
            crate::llama_log_warn!(
                "sampler_types_from_chars: unable to match sampler by char '{}'",
                c
            );
        }
    }
    result
}

/// `common_sampling_add_preserved_tokens`.
pub fn add_preserved_tokens(
    sampling: &mut ParamsSampling,
    vocab: &LlamaVocab,
    tokens: &[String],
) -> Result<()> {
    for token in tokens {
        let ids = common_tokenize(vocab, token, false, true)?;
        if ids.len() == 1 {
            sampling.preserved_tokens.insert(ids[0]);
        }
    }
    Ok(())
}

/// `common_sampling_add_grammar_triggers`.
pub fn add_grammar_triggers(
    sampling: &mut ParamsSampling,
    vocab: &LlamaVocab,
    triggers: Vec<GrammarTrigger>,
) -> Result<()> {
    for trigger in triggers {
        if trigger.type_ == GrammarTriggerType::Word {
            let ids = common_tokenize(vocab, &trigger.value, false, true)?;
            if ids.len() == 1 {
                let token = ids[0];
                if !sampling.preserved_tokens.contains(&token) {
                    return Err(Error::new(format!(
                        "Grammar trigger word should be marked as preserved token: {}",
                        trigger.value
                    )));
                }
                sampling.grammar_triggers.push(GrammarTrigger {
                    type_: GrammarTriggerType::Token,
                    value: trigger.value,
                    token,
                });
            } else {
                sampling.grammar_triggers.push(trigger);
            }
        } else {
            sampling.grammar_triggers.push(trigger);
        }
    }
    if sampling.grammar_lazy && sampling.grammar_triggers.is_empty() {
        return Err(Error::new("Error: no triggers set for lazy grammar!"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{init_chain, ChainParams};

    #[test]
    fn chars_map_to_sampler_types() {
        assert_eq!(
            sampler_types_from_chars("kpt"),
            vec![
                SamplerType::TopK,
                SamplerType::TopP,
                SamplerType::Temperature
            ]
        );
        assert!(sampler_types_from_chars("?").is_empty());
    }

    #[test]
    fn sampler_holds_history_and_copies() {
        let mut sampler = CommonSampler::new(
            init_chain(ChainParams::default()),
            ParamsSampling::default(),
            None,
        )
        .unwrap();
        sampler.accept(7, true);
        sampler.accept(9, true);
        assert_eq!(sampler.last().unwrap(), 9);
        assert_eq!(sampler.prev.to_vector(), vec![7, 9]);
        assert!(sampler.print().starts_with("logits"));

        let mut copy = sampler.clone_sampler();
        copy.reset();
        assert!(copy.prev.is_empty());
        copy.copy_from(&sampler).unwrap();
        assert_eq!(copy.prev.to_vector(), vec![7, 9]);
        assert_eq!(copy.seed(), sampler.seed());
    }
}

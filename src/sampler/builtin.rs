use crate::core::{Token, LLAMA_TOKEN_NULL};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use super::candidate::{
    sample_dist, softmax_impl, temp_impl, top_k_impl, Grammar, LogitBias, TokenData,
    TokenDataArray, Vocab, LLAMA_DRY_MAX_SEQ_BREAKERS,
};
use super::chain::{partial_sort, partial_sort_inplace, RingBuffer, Sampler};
use super::rng::{get_rng_seed, SamplerRng};

#[derive(Clone)]
struct Empty {
    name: String,
}

pub fn init_empty(name: &str) -> Box<dyn Sampler> {
    Box::new(Empty {
        name: name.to_owned(),
    })
}

impl Sampler for Empty {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&mut self, _cur: &mut TokenDataArray) {}

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct Greedy;

pub fn init_greedy() -> Box<dyn Sampler> {
    Box::new(Greedy)
}

impl Sampler for Greedy {
    fn name(&self) -> &str {
        "greedy"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        if cur.data.is_empty() {
            cur.selected = -1;
            return;
        }
        cur.selected = 0;
        for i in 1..cur.data.len() {
            if cur.data[i].logit > cur.data[cur.selected as usize].logit {
                cur.selected = i as i32;
            }
        }
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// dist
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Dist {
    seed: u32,
    seed_cur: u32,
    rng: SamplerRng,
    backend_transactional: bool,
    rng_backend: SamplerRng,
    n_backend_draws_generated: usize,
    n_backend_draws_committed: usize,
}

pub fn init_dist(seed: u32) -> Box<dyn Sampler> {
    let seed_cur = get_rng_seed(seed);
    Box::new(Dist {
        seed,
        seed_cur,
        rng: SamplerRng::new(seed_cur),
        backend_transactional: false,
        rng_backend: SamplerRng::new(seed_cur),
        n_backend_draws_generated: 0,
        n_backend_draws_committed: 0,
    })
}

impl Sampler for Dist {
    fn name(&self) -> &str {
        "dist"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        // edge cases
        if cur.data.is_empty() {
            cur.selected = -1;
            return;
        }

        cur.selected = 0;

        if cur.data.len() == 1 {
            // keep the RNG state aligned with backend sampling, which draws once per output
            let _ = self.rng.uniform_f64();
            cur.data[0].p = 1.0;
            return;
        }

        // max logit for numerical stability
        let mut max_l = cur.data[0].logit;
        if !cur.sorted {
            for td in &cur.data[1..] {
                max_l = max_l.max(td.logit);
            }
        }

        // apply softmax to obtain the probabilities
        let mut sum_cum = 0.0f64;
        for td in &mut cur.data {
            let p = (td.logit - max_l).exp();
            td.p = p;
            sum_cum += p as f64;
        }

        // sample from the obtained probabilities and normalize the probs in a single pass
        let rnd = self.rng.uniform_f64();

        let mut sum_run = 0.0f64;
        let sum_tgt = sum_cum * rnd;
        let mut found = false;
        for i in 0..cur.data.len() {
            if !found {
                // accumulate probs until we reach the target sum
                sum_run += cur.data[i].p as f64;
                if sum_run >= sum_tgt {
                    cur.selected = i as i32;
                    found = true;
                }
            }
            // normalize probs
            cur.data[i].p = (cur.data[i].p as f64 / sum_cum) as f32;
        }

        // fallback to the last token
        if !found {
            cur.selected = (cur.data.len() - 1) as i32;
        }
    }

    fn accept(&mut self, _token: Token) {
        if !self.backend_transactional
            || self.n_backend_draws_committed >= self.n_backend_draws_generated
        {
            return;
        }
        let _ = self.rng.uniform_f64();
        self.n_backend_draws_committed += 1;
    }

    fn reset(&mut self) {
        self.seed_cur = get_rng_seed(self.seed);
        self.rng.reseed(self.seed_cur);
        self.rng_backend = self.rng.clone();
        self.n_backend_draws_generated = 0;
        self.n_backend_draws_committed = 0;
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }

    fn seed_cur(&self) -> Option<u32> {
        Some(self.seed_cur)
    }

    fn backend_begin(&mut self) {
        if self.backend_transactional {
            self.rng_backend = self.rng.clone();
            self.n_backend_draws_generated = 0;
            self.n_backend_draws_committed = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// top-k
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct TopK {
    k: i32,
}

pub fn init_top_k(k: i32) -> Box<dyn Sampler> {
    if k <= 0 {
        return init_empty("?top-k");
    }
    Box::new(TopK { k })
}

impl Sampler for TopK {
    fn name(&self) -> &str {
        "top-k"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        top_k_impl(cur, self.k);
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// top-p
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct TopP {
    p: f32,
    min_keep: usize,
    buf_sort: Vec<TokenData>,
}

pub fn init_top_p(p: f32, min_keep: usize) -> Box<dyn Sampler> {
    if p >= 1.0 {
        return init_empty("?top-p");
    }
    Box::new(TopP {
        p,
        min_keep,
        buf_sort: Vec::new(),
    })
}

impl Sampler for TopP {
    fn name(&self) -> &str {
        "top-p"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        if self.p >= 1.0 {
            return;
        }

        softmax_impl(cur, false);

        let mut k = cur.data.len();
        let mut use_buf = false;

        // if not sorted, try adaptive top-k sorting
        if !cur.sorted && cur.data.len() > 1024 {
            k = 256.min(cur.data.len());
            self.buf_sort = partial_sort(cur, k);
            use_buf = true;
        } else if !cur.sorted {
            // small candidates -> sort inplace
            partial_sort_inplace(cur, k);
        }

        // Compute the cumulative probabilities
        let mut cum_sum = 0.0f32;
        let mut last_idx = cur.data.len();

        for i in 0..cur.data.len() {
            let pi = if use_buf {
                self.buf_sort[i].p
            } else {
                cur.data[i].p
            };
            cum_sum += pi;

            // Check if the running sum is at least p or if we have kept at least min_keep tokens
            // we set the last index to i+1 to indicate that the current iterate should be included
            if cum_sum >= self.p && i + 1 >= self.min_keep {
                last_idx = i + 1;
                break;
            }

            // we exceeded the current top-k heuristic -> increase k and continue
            if !cur.sorted && i == k - 1 {
                k = cur.data.len();
                self.buf_sort = partial_sort(cur, k);
                use_buf = true;
            }
        }

        // Resize the output vector to keep only the top-p tokens
        if !cur.sorted {
            for i in 0..last_idx {
                cur.data[i] = self.buf_sort[i];
            }
            cur.sorted = true;
        }
        cur.data.truncate(last_idx);
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// min-p
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct MinP {
    p: f32,
    min_keep: usize,
}

pub fn init_min_p(p: f32, min_keep: usize) -> Box<dyn Sampler> {
    if p <= 0.0 {
        return init_empty("?min-p");
    }
    Box::new(MinP { p, min_keep })
}

impl Sampler for MinP {
    fn name(&self) -> &str {
        "min-p"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        if self.p <= 0.0 || cur.data.is_empty() {
            return;
        }

        let mut min_p_applied = false;

        // if the cur_p aren't sorted, try the unsorted implementation first
        if !cur.sorted {
            let mut filtered_tokens = Vec::new();

            let mut max_logit = -f32::MAX;
            for td in &cur.data {
                max_logit = max_logit.max(td.logit);
            }
            let min_logit = max_logit + self.p.ln(); // min logit for p_i >= p * p_max

            for td in &cur.data {
                if td.logit >= min_logit {
                    filtered_tokens.push(*td);
                }
            }

            // if we have enough values the operation was a success
            if !filtered_tokens.is_empty() && filtered_tokens.len() >= self.min_keep {
                for i in 0..filtered_tokens.len() {
                    cur.data[i] = filtered_tokens[i];
                }
                cur.data.truncate(filtered_tokens.len());
                min_p_applied = true;
            }
        }

        // if the cur_p are sorted or the unsorted implementation failed, use this implementation
        if !min_p_applied {
            // Sort the logits in descending order
            if !cur.sorted {
                partial_sort_inplace(cur, cur.data.len());
            }

            let min_logit = cur.data[0].logit + self.p.ln(); // min logit for p_i >= p * p_max
            let mut i = 1usize; // first token always matches

            while i < cur.data.len() {
                if cur.data[i].logit < min_logit && i >= self.min_keep {
                    break; // prob too small
                }
                i += 1;
            }

            // Resize the output vector to keep only the matching tokens
            cur.data.truncate(i);
        }
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// typical-p
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Typical {
    p: f32,
    min_keep: usize,
}

pub fn init_typical(p: f32, min_keep: usize) -> Box<dyn Sampler> {
    if p >= 1.0 {
        return init_empty("?typical");
    }
    Box::new(Typical { p, min_keep })
}

impl Sampler for Typical {
    fn name(&self) -> &str {
        "typical"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        // Reference implementation:
        // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
        if self.p >= 1.0 {
            return;
        }

        // Compute the softmax of logits and calculate entropy
        softmax_impl(cur, true);

        let mut entropy = 0.0f32;
        for td in &cur.data {
            entropy += -td.p * td.p.ln();
        }

        // Compute the absolute difference between negative log probability and entropy
        let mut shifted_scores = Vec::with_capacity(cur.data.len());
        for td in &cur.data {
            shifted_scores.push((-td.p.ln() - entropy).abs());
        }

        // Sort tokens based on the shifted_scores and their corresponding indices
        let mut indices: Vec<usize> = (0..cur.data.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            shifted_scores[a]
                .partial_cmp(&shifted_scores[b])
                .unwrap_or(Ordering::Equal)
        });

        // Compute the cumulative probabilities
        let mut cum_sum = 0.0f32;
        let mut last_idx = indices.len();

        for (i, &idx) in indices.iter().enumerate() {
            cum_sum += cur.data[idx].p;

            // Check if the running sum is greater than typical or if we have kept min_keep tokens
            if cum_sum > self.p && (self.min_keep == 0 || i >= self.min_keep - 1) {
                last_idx = i + 1;
                break;
            }
        }

        // Resize the output vector to keep only the locally typical tokens
        let mut cur_p_new = Vec::with_capacity(last_idx);
        for &idx in &indices[..last_idx] {
            cur_p_new.push(cur.data[idx]);
        }
        cur.data = cur_p_new;
        cur.sorted = false;
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// temp
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Temp {
    temp: f32,
}

pub fn init_temp(temp: f32) -> Box<dyn Sampler> {
    if temp == 1.0 {
        return init_empty("?temp");
    }
    Box::new(Temp { temp })
}

impl Sampler for Temp {
    fn name(&self) -> &str {
        "temp"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        temp_impl(cur, self.temp);
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// temp-ext
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct TempExt {
    temp: f32,
    delta: f32,
    exponent: f32,
}

pub fn init_temp_ext(temp: f32, delta: f32, exponent: f32) -> Box<dyn Sampler> {
    if temp == 1.0 && delta <= 0.0 {
        return init_empty("?temp-ext");
    }
    Box::new(TempExt {
        temp,
        delta,
        exponent,
    })
}

impl Sampler for TempExt {
    fn name(&self) -> &str {
        "temp-ext"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        if self.delta > 0.0 {
            let min_temp = 0.0f32.max(self.temp - self.delta);
            let max_temp = self.temp + self.delta;

            let exponent_val = self.exponent;

            // no need to do anything if there is only one (or zero) candidates
            if cur.data.len() <= 1 {
                return;
            }

            // Calculate maximum possible entropy
            let max_entropy = -(1.0f32 / cur.data.len() as f32).ln();

            softmax_impl(cur, true);

            // Calculate entropy of the softmax probabilities
            let mut entropy = 0.0f32;
            for td in &cur.data {
                if td.p > 0.0 {
                    // Ensure no log(0)
                    entropy -= td.p * td.p.ln();
                }
            }

            // Normalize the entropy (max_entropy cannot be 0 here because size != 1)
            let normalized_entropy = entropy / max_entropy;

            // Map the normalized entropy to the desired temperature range
            let dyn_temp = min_temp + (max_temp - min_temp) * normalized_entropy.powf(exponent_val);

            // Apply the dynamically calculated temperature scaling
            temp_impl(cur, dyn_temp);

            // Re-compute softmax probabilities after scaling logits with dynamic temperature
            let max_l_double = cur.data[0].logit as f64;
            let mut cum_sum_double = 0.0f64;
            for td in &mut cur.data {
                let p = (td.logit as f64 - max_l_double).exp();
                td.p = p as f32;
                cum_sum_double += p;
            }
            for td in &mut cur.data {
                td.p = (td.p as f64 / cum_sum_double) as f32;
            }
        } else {
            temp_impl(cur, self.temp);
        }
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// xtc
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Xtc {
    probability: f32,
    threshold: f32,
    min_keep: usize,
    seed: u32,
    seed_cur: u32,
    rng: SamplerRng,
}

pub fn init_xtc(p: f32, t: f32, min_keep: usize, seed: u32) -> Box<dyn Sampler> {
    if p <= 0.0 || t > 0.5 {
        return init_empty("?xtc");
    }
    let seed_cur = get_rng_seed(seed);
    Box::new(Xtc {
        probability: p,
        threshold: t,
        min_keep,
        seed,
        seed_cur,
        rng: SamplerRng::new(seed_cur),
    })
}

impl Sampler for Xtc {
    fn name(&self) -> &str {
        "xtc"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        if self.probability <= 0.0 || self.threshold > 0.5 || cur.data.len() < 2 {
            return;
        }

        let chance = self.rng.uniform_f32();
        if chance > self.probability {
            return;
        }

        softmax_impl(cur, true);

        let mut pos_last = 0usize;
        for i in 0..cur.data.len() {
            if cur.data[i].p >= self.threshold {
                pos_last = i;
            } else {
                break;
            }
        }

        if cur.data.len() - pos_last >= self.min_keep && pos_last > 0 {
            cur.data.drain(0..pos_last);
        }
    }

    fn reset(&mut self) {
        self.seed_cur = get_rng_seed(self.seed);
        self.rng.reseed(self.seed_cur);
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// mirostat
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Mirostat {
    n_vocab: i32,
    seed: u32,
    seed_cur: u32,
    tau: f32,
    eta: f32,
    m: i32,
    mu: f32,
    rng: SamplerRng,
}

pub fn init_mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Box<dyn Sampler> {
    let seed_cur = get_rng_seed(seed);
    Box::new(Mirostat {
        n_vocab,
        seed,
        seed_cur,
        tau,
        eta,
        m,
        mu: 2.0 * tau,
        rng: SamplerRng::new(seed_cur),
    })
}

impl Sampler for Mirostat {
    fn name(&self) -> &str {
        "mirostat"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        softmax_impl(cur, true);

        // Estimate s_hat using the most probable m tokens
        let mut s_hat = 0.0f32;
        let mut sum_ti_bi = 0.0f32;
        let mut sum_ti_sq = 0.0f32;
        let limit = (self.m - 1).max(0) as usize;
        let avail = cur.data.len().saturating_sub(1);
        for i in 0..limit.min(avail) {
            let t_i = ((i + 2) as f32 / (i + 1) as f32).ln();
            let b_i = (cur.data[i].p / cur.data[i + 1].p).ln();
            sum_ti_bi += t_i * b_i;
            sum_ti_sq += t_i * t_i;
        }
        s_hat = sum_ti_bi / sum_ti_sq;

        // Compute k from the estimated s_hat and target surprise value
        let epsilon_hat = s_hat - 1.0;
        let k = ((epsilon_hat * 2.0f32.powf(self.mu))
            / (1.0 - (self.n_vocab as f32).powf(-epsilon_hat)))
        .powf(1.0 / s_hat);

        top_k_impl(cur, (k as i32).max(1));
        softmax_impl(cur, true);

        let idx = sample_dist(cur, &mut self.rng);
        cur.selected = idx as i32;

        let observed_surprise = -cur.data[idx].p.log2();
        let e = observed_surprise - self.tau;

        // Update mu using the learning rate and error
        self.mu -= self.eta * e;
    }

    fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
        self.seed_cur = get_rng_seed(self.seed);
        self.rng.reseed(self.seed_cur);
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }

    fn seed_cur(&self) -> Option<u32> {
        Some(self.seed_cur)
    }
}

// ---------------------------------------------------------------------------
// mirostat v2
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct MirostatV2 {
    seed: u32,
    seed_cur: u32,
    tau: f32,
    eta: f32,
    mu: f32,
    rng: SamplerRng,
}

pub fn init_mirostat_v2(seed: u32, tau: f32, eta: f32) -> Box<dyn Sampler> {
    let seed_cur = get_rng_seed(seed);
    Box::new(MirostatV2 {
        seed,
        seed_cur,
        tau,
        eta,
        mu: 2.0 * tau,
        rng: SamplerRng::new(seed_cur),
    })
}

impl Sampler for MirostatV2 {
    fn name(&self) -> &str {
        "mirostat-v2"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        if cur.data.is_empty() {
            return;
        }

        softmax_impl(cur, true);

        // Truncate the words with surprise values greater than mu
        let mut size = 0usize;
        for td in &cur.data {
            if -td.p.log2() > self.mu {
                break;
            }
            size += 1;
        }
        cur.data.truncate(size.max(1).min(cur.data.len()));

        // Normalize the probabilities of the remaining words
        softmax_impl(cur, true);

        let idx = sample_dist(cur, &mut self.rng);
        cur.selected = idx as i32;

        let observed_surprise = -cur.data[idx].p.log2();
        let e = observed_surprise - self.tau;

        // Update mu using the learning rate and error
        self.mu -= self.eta * e;
    }

    fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
        self.seed_cur = get_rng_seed(self.seed);
        self.rng.reseed(self.seed_cur);
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }

    fn seed_cur(&self) -> Option<u32> {
        Some(self.seed_cur)
    }
}

// ---------------------------------------------------------------------------
// grammar
// ---------------------------------------------------------------------------

struct GrammarSampler {
    grammar_str: String,
    grammar_root: String,
    grammar: Option<Box<dyn Grammar>>,
}

pub fn init_grammar(
    grammar_str: Option<&str>,
    grammar_root: Option<&str>,
    grammar: Option<Box<dyn Grammar>>,
) -> Box<dyn Sampler> {
    Box::new(GrammarSampler {
        grammar_str: grammar_str.unwrap_or("").to_owned(),
        grammar_root: grammar_root.unwrap_or("").to_owned(),
        grammar,
    })
}

impl Sampler for GrammarSampler {
    fn name(&self) -> &str {
        "grammar"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        if let Some(grammar) = &mut self.grammar {
            grammar.apply(cur);
        }
    }

    fn accept(&mut self, token: Token) {
        if let Some(grammar) = &mut self.grammar {
            grammar.accept(token);
        }
    }

    fn reset(&mut self) {
        if let Some(grammar) = &mut self.grammar {
            grammar.reset();
        }
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(GrammarSampler {
            grammar_str: self.grammar_str.clone(),
            grammar_root: self.grammar_root.clone(),
            grammar: self.grammar.as_ref().map(|g| g.clone_box()),
        })
    }
}

// ---------------------------------------------------------------------------
// penalties
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Penalties {
    n_vocab: i32,
    penalty_last_n: i32,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,
    prev: RingBuffer<Token>,
    token_count: HashMap<Token, i32>,
}

impl Penalties {
    fn is_disabled(last_n: i32, repeat: f32, freq: f32, present: f32) -> bool {
        last_n == 0 || (repeat == 1.0 && freq == 0.0 && present == 0.0)
    }

    fn disabled(&self) -> bool {
        Self::is_disabled(
            self.penalty_last_n,
            self.penalty_repeat,
            self.penalty_freq,
            self.penalty_present,
        )
    }
}

pub fn init_penalties(
    n_vocab: i32,
    penalty_last_n: i32,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,
) -> Box<dyn Sampler> {
    let penalty_last_n = penalty_last_n.max(0);
    if Penalties::is_disabled(
        penalty_last_n,
        penalty_repeat,
        penalty_freq,
        penalty_present,
    ) {
        return init_empty("?penalties");
    }
    Box::new(Penalties {
        n_vocab,
        penalty_last_n,
        penalty_repeat,
        penalty_freq,
        penalty_present,
        prev: RingBuffer::new(penalty_last_n as usize),
        token_count: HashMap::new(),
    })
}

impl Sampler for Penalties {
    fn name(&self) -> &str {
        "penalties"
    }

    fn accept(&mut self, token: Token) {
        if self.penalty_last_n == 0 {
            return;
        }

        *self.token_count.entry(token).or_insert(0) += 1;

        // if the ring buffer is full, remove the oldest token
        if self.prev.len() >= self.penalty_last_n as usize {
            let old = self.prev.front();
            let remove = {
                let count = self
                    .token_count
                    .get_mut(&old)
                    .expect("penalties: missing token count");
                *count -= 1;
                *count == 0
            };
            if remove {
                self.token_count.remove(&old);
            }
        }

        self.prev.push_back(token);
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        if self.disabled() {
            return;
        }

        // Apply frequency and presence penalties to the cur_p
        for td in &mut cur.data {
            if let Some(&count) = self.token_count.get(&td.id) {
                // The academic publication divided the logit. That would make tokens with
                // negative logits more likely, so we multiply instead of dividing.
                if td.logit <= 0.0 {
                    td.logit *= self.penalty_repeat;
                } else {
                    td.logit /= self.penalty_repeat;
                }
                let present = if count > 0 {
                    self.penalty_present
                } else {
                    0.0
                };
                td.logit -= count as f32 * self.penalty_freq + present;
            }
        }

        cur.sorted = false;
    }

    fn reset(&mut self) {
        self.prev.clear();
        self.token_count.clear();
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// top-n-sigma
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct TopNSigma {
    n: f32,
}

pub fn init_top_n_sigma(n: f32) -> Box<dyn Sampler> {
    if n <= 0.0 {
        return init_empty("?top-n-sigma");
    }
    Box::new(TopNSigma { n })
}

impl Sampler for TopNSigma {
    fn name(&self) -> &str {
        "top-n-sigma"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        if self.n <= 0.0 || cur.data.len() <= 1 {
            return;
        }

        // find max logit and calculate mean
        let mut max = cur.data[0].logit;
        let mut logits_sum = 0.0f32;
        let mut valid_count = 0usize;
        for td in &cur.data {
            // Only count non-negative infinity values
            if td.logit != f32::NEG_INFINITY {
                max = max.max(td.logit);
                logits_sum += td.logit;
                valid_count += 1;
            }
        }
        let mean = if valid_count > 0 {
            logits_sum / valid_count as f32
        } else {
            0.0
        };

        // calculate standard deviation
        let mut acc = 0.0f32;
        for td in &cur.data {
            // Skip -infinity in std calculation
            if td.logit != f32::NEG_INFINITY {
                acc += (td.logit - mean).powi(2);
            }
        }
        let std = if valid_count > 0 {
            (acc / valid_count as f32).sqrt()
        } else {
            0.0
        };

        // apply mask
        for td in &mut cur.data {
            if td.logit < max - self.n * std {
                td.logit = f32::NEG_INFINITY;
            }
        }
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// DRY
// ---------------------------------------------------------------------------

/// `llama_sampler_dry`.
struct Dry {
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: i32,
    dry_penalty_last_n: i32,
    dry_processed_breakers: HashMap<Token, Vec<Vec<Token>>>,
    dry_repeat_count: Vec<i32>,
    dry_max_token_repeat: HashMap<Token, i32>,
    last_tokens: RingBuffer<Token>,
}

impl Clone for Dry {
    fn clone(&self) -> Self {
        Self {
            dry_multiplier: self.dry_multiplier,
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_penalty_last_n: self.dry_penalty_last_n,
            dry_processed_breakers: self.dry_processed_breakers.clone(),
            dry_repeat_count: self.dry_repeat_count.clone(),
            dry_max_token_repeat: self.dry_max_token_repeat.clone(),
            last_tokens: self.last_tokens.clone(),
        }
    }
}

fn contains_subslice(hay: &[u8], needle: &[u8]) -> bool {
    needle.len() <= hay.len()
        && (0..=hay.len() - needle.len()).any(|i| &hay[i..i + needle.len()] == needle)
}

// Ported from Koboldcpp, original PR: https://github.com/LostRuins/koboldcpp/pull/982
fn get_overlapping_token_sequences(
    vocab: &dyn Vocab,
    vocab_words: &[Vec<u8>],
    text: &[u8],
    token_sequences: &mut HashMap<Token, Vec<Vec<Token>>>,
    max_tail_len: i32,
) {
    // the tail tokenization only depends on how many characters of str were matched,
    // so compute each one at most once
    let mut tail_cache: Vec<Option<Vec<Token>>> = (0..=text.len()).map(|_| None).collect();

    for (token_id, word) in vocab_words.iter().enumerate() {
        let token_id = token_id as Token;
        if contains_subslice(word, text) {
            token_sequences.entry(token_id).or_default();
            continue;
        }

        let word_len = word.len();
        let str_len = text.len();
        let mut pos: isize = -1;
        loop {
            // find the next occurrence of text[0] at pos + 1
            let start = (pos + 1) as usize;
            let p = match word[start..].iter().position(|&c| c == text[0]) {
                Some(off) => start + off,
                None => break,
            };
            pos = p as isize;

            let mut i = 0usize;
            let mut matched = true;
            while i < str_len && p + i < word_len {
                if word[p + i] != text[i] {
                    matched = false;
                    break;
                }
                i += 1;
            }

            if matched {
                if tail_cache[i].is_none() {
                    let mut tail = vocab.tokenize(&text[i..], false, false);
                    if max_tail_len >= 0 && tail.len() > max_tail_len as usize {
                        tail.truncate(max_tail_len as usize);
                    }
                    tail_cache[i] = Some(tail);
                }
                let tokenization = tail_cache[i].clone().expect("tail cache");
                let entry = token_sequences.entry(token_id).or_default();
                // Ensure we don't already have a duplicate matching tokenization
                if !entry.iter().any(|t| *t == tokenization) {
                    entry.push(tokenization);
                }
            }
        }
    }
}

pub fn init_dry(
    vocab: &dyn Vocab,
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: i32,
    dry_penalty_last_n: i32,
    seq_breakers: &[&str],
) -> Box<dyn Sampler> {
    let dry_penalty_last_n = dry_penalty_last_n.max(0);
    let dry_enabled = dry_multiplier != 0.0 && dry_base >= 1.0 && dry_penalty_last_n != 0;

    if !dry_enabled {
        return init_empty("?dry");
    }

    let mut processed_breakers: HashMap<Token, Vec<Vec<Token>>> = HashMap::new();
    if !seq_breakers.is_empty() {
        // each distinct breaker costs a full vocabulary scan: drop duplicates and stop at the cap
        let mut seen_breakers: HashSet<Vec<u8>> = HashSet::new();

        // detokenize the vocabulary once instead of rescanning it for every breaker
        let mut vocab_words = Vec::with_capacity(vocab.n_tokens());
        for token_id in 0..vocab.n_tokens() as Token {
            vocab_words.push(vocab.detokenize(token_id, true));
        }

        for &breaker in seq_breakers {
            if breaker.is_empty() {
                continue;
            }
            let mut sequence_break = breaker.as_bytes().to_vec();
            if sequence_break.len() > 40 {
                sequence_break.truncate(40);
            }
            if !seen_breakers.insert(sequence_break.clone()) {
                continue;
            }
            if seen_breakers.len() > LLAMA_DRY_MAX_SEQ_BREAKERS {
                break;
            }
            get_overlapping_token_sequences(
                vocab,
                &vocab_words,
                &sequence_break,
                &mut processed_breakers,
                20,
            );
        }
    }

    Box::new(Dry {
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        dry_penalty_last_n,
        dry_processed_breakers: processed_breakers,
        dry_repeat_count: vec![0; dry_penalty_last_n as usize],
        dry_max_token_repeat: HashMap::new(),
        last_tokens: RingBuffer::new(dry_penalty_last_n as usize),
    })
}

/// Wrapper for `test-sampling.cpp`: token-based sequence breakers need no vocab.
pub fn init_dry_testing(
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: i32,
    dry_penalty_last_n: i32,
    seq_breakers: &[Vec<Token>],
) -> Box<dyn Sampler> {
    let dry_penalty_last_n = dry_penalty_last_n.max(0);
    if dry_multiplier == 0.0 || dry_base < 1.0 || dry_penalty_last_n == 0 {
        return init_empty("?dry");
    }

    let mut processed_breakers: HashMap<Token, Vec<Vec<Token>>> = HashMap::new();
    for breaker in seq_breakers {
        if breaker.is_empty() {
            continue;
        }
        let head = breaker[0];
        let tail = breaker[1..].to_vec();
        processed_breakers.entry(head).or_default().push(tail);
    }

    Box::new(Dry {
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        dry_penalty_last_n,
        dry_processed_breakers: processed_breakers,
        dry_repeat_count: vec![0; dry_penalty_last_n as usize],
        dry_max_token_repeat: HashMap::new(),
        last_tokens: RingBuffer::new(dry_penalty_last_n as usize),
    })
}

impl Sampler for Dry {
    fn name(&self) -> &str {
        "dry"
    }

    fn accept(&mut self, token: Token) {
        if self.dry_multiplier == 0.0 || self.dry_base < 1.0 || self.dry_penalty_last_n == 0 {
            return;
        }
        self.last_tokens.push_back(token);
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        if self.dry_multiplier == 0.0 || self.dry_base < 1.0 || self.dry_penalty_last_n == 0 {
            return;
        }

        let last_n_repeat = self.last_tokens.len().min(self.dry_penalty_last_n as usize);
        if last_n_repeat as i32 <= self.dry_allowed_length {
            return;
        }

        self.dry_repeat_count.clear();
        self.dry_repeat_count.resize(last_n_repeat, 0);
        self.dry_max_token_repeat.clear();

        // Step 1: Look for restart sequences to limit the maximum repetition length.
        // Work backwards through the context looking for any token that begins a
        // restart sequence. The longest matching sequence is used to limit it.
        let mut rep_limit = last_n_repeat as i32;
        for i in 0..last_n_repeat {
            let token = self.last_tokens.rat(i);
            let mut longest_match: i32 = -1;
            if let Some(tails) = self.dry_processed_breakers.get(&token) {
                for tail in tails {
                    // (*it) does not contain the head character
                    let seq_len = tail.len() as i32;
                    if seq_len > longest_match && seq_len <= i as i32 {
                        let mut matched = true;
                        for (offset, &t) in tail.iter().enumerate() {
                            // -1 when indexing last_tokens: the head is already matched
                            if t != self.last_tokens.rat(i - offset - 1) {
                                matched = false;
                                break;
                            }
                        }
                        if matched {
                            longest_match = seq_len;
                        }
                    }
                }
            }
            if longest_match >= 0 {
                rep_limit = i as i32 - longest_match;
                break;
            }
        }
        if rep_limit < self.dry_allowed_length {
            return;
        }

        // Step 2: reverse Z-algorithm: positions and lengths of suffixes appearing
        // elsewhere in the context, limited to rep_limit. O(N) despite nested loops.
        {
            let last = last_n_repeat - 1;
            let mut rt = 0i32;
            let mut lt = 0i32;

            for k in 1..last_n_repeat {
                let kk = k as i32;
                if kk > rt {
                    // If k is outside the current Z-box, do naive computation.
                    let mut n = 0i32;
                    while n + kk < last_n_repeat as i32
                        && self.last_tokens.rat(n as usize)
                            == self.last_tokens.rat((n + kk) as usize)
                    {
                        n += 1;
                    }
                    self.dry_repeat_count[last - k] = n.min(rep_limit);
                    if n > 0 {
                        lt = kk;
                        rt = kk + n - 1;
                    }
                } else {
                    // If k is inside the current Z-box, consider two cases.
                    let p = kk - lt; // Pair index.
                    let right_part_len = rt - kk + 1;
                    let prev = self.dry_repeat_count[last - p as usize];
                    if prev < right_part_len {
                        self.dry_repeat_count[last - k] = prev.min(rep_limit);
                    } else {
                        let mut i = rt + 1;
                        while i < last_n_repeat as i32
                            && self.last_tokens.rat(i as usize)
                                == self.last_tokens.rat((i - kk) as usize)
                        {
                            i += 1;
                        }
                        self.dry_repeat_count[last - k] = (i - kk).min(rep_limit);
                        lt = kk;
                        rt = i - 1;
                    }
                }
            }
        }

        // Step 3: find the maximum repeat length generated by emitting each new token.
        for i in 0..last_n_repeat - 1 {
            let repeat_len = self.dry_repeat_count[i];
            if repeat_len >= self.dry_allowed_length {
                let token = self.last_tokens.rat(last_n_repeat - 2 - i);
                let update = match self.dry_max_token_repeat.get(&token) {
                    Some(&v) => v < repeat_len,
                    None => true,
                };
                if update {
                    self.dry_max_token_repeat.insert(token, repeat_len);
                }
            }
        }

        // Step 4: Apply logit penalties based on the maximum repeat length.
        // Prevent floating point overflow in pow(base, exponent).
        const FLOAT_MAX_LOG: f32 = 88.7228391;
        let max_exponent = if self.dry_base > 1.000001 {
            (FLOAT_MAX_LOG / self.dry_base.ln()) as i32
        } else {
            0
        };

        for td in &mut cur.data {
            if let Some(&repeat) = self.dry_max_token_repeat.get(&td.id) {
                // Apply penalty only if it's not a single-token sequence breaker
                let is_single_token_breaker = self
                    .dry_processed_breakers
                    .get(&td.id)
                    .map_or(false, |tails| tails.iter().any(|tail| tail.is_empty()));
                if !is_single_token_breaker {
                    let mut repeat_exp = repeat - self.dry_allowed_length;
                    if max_exponent > 0 && repeat_exp > max_exponent {
                        repeat_exp = max_exponent;
                    }
                    let penalty = self.dry_multiplier * self.dry_base.powi(repeat_exp);
                    td.logit -= penalty;
                }
            }
        }

        cur.sorted = false;
    }

    fn reset(&mut self) {
        self.last_tokens.clear();
        self.dry_repeat_count.clear();
        self.dry_max_token_repeat.clear();
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// adaptive-p
// ---------------------------------------------------------------------------

const DISTRIBUTION_WIDTH: f32 = 0.3;
const PEAK_LOGIT_VALUE: f32 = 5.0;
const SHARPNESS: f32 = 10.0;
const INV_WIDTH: f32 = 1.0 / DISTRIBUTION_WIDTH;

#[derive(Clone)]
struct AdaptiveP {
    target: f32,
    decay: f32,
    seed: u32,
    seed_cur: u32,
    rng: SamplerRng,
    weighted_sum: f32,
    total_weight: f32,
    original_probs: Vec<f32>,
    pending_token_id: Token,
    pending_token_idx: i32,
}

pub fn init_adaptive_p(target: f32, decay: f32, seed: u32) -> Box<dyn Sampler> {
    let seed_cur = get_rng_seed(seed);
    let clamped_decay = decay.clamp(0.0, 0.99);
    Box::new(AdaptiveP {
        target,
        decay: clamped_decay,
        seed,
        seed_cur,
        rng: SamplerRng::new(seed_cur),
        weighted_sum: target / (1.0 - clamped_decay),
        total_weight: 1.0 / (1.0 - clamped_decay),
        original_probs: Vec::new(),
        pending_token_id: LLAMA_TOKEN_NULL,
        pending_token_idx: -1,
    })
}

impl Sampler for AdaptiveP {
    fn name(&self) -> &str {
        "adaptive-p"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        softmax_impl(cur, false);

        if self.target < 0.0 {
            // at negative target values, adaptive-p is a no-op
            let idx = sample_dist(cur, &mut self.rng);
            cur.selected = idx as i32;
            return;
        }

        // store the original probabilities
        self.original_probs.clear();
        for td in &cur.data {
            self.original_probs.push(td.p);
        }

        // using the EMA, compute the adapted target probability for this step
        let target = self.target.clamp(0.0, 1.0);
        let adapted_target = if self.total_weight == 0.0 {
            target
        } else {
            (2.0 * target - self.weighted_sum / self.total_weight).clamp(0.0, 1.0)
        };

        // quadratic near target for fine differentiation, transitioning to linear decay
        // in the tails. Unbounded negative logits suppress far-from-target tokens.
        for td in &mut cur.data {
            if td.logit == f32::NEG_INFINITY {
                // don't transform logits masked out by e.g. min-p and top-p
                continue;
            }
            let dist = ((td.p - adapted_target) * INV_WIDTH).abs();
            td.logit = PEAK_LOGIT_VALUE - SHARPNESS * dist * dist / (1.0 + dist);
        }

        // softmax and sample from the transformed distribution
        softmax_impl(cur, false);
        let idx = sample_dist(cur, &mut self.rng);
        cur.selected = idx as i32;

        self.pending_token_id = cur.data[idx].id;
        self.pending_token_idx = idx as i32;
    }

    fn accept(&mut self, token: Token) {
        if self.pending_token_id == token {
            let idx = self.pending_token_idx;
            if idx >= 0 && (idx as usize) < self.original_probs.len() {
                // update EMA with the original probability of the selected token
                self.weighted_sum =
                    self.original_probs[idx as usize] + self.decay * self.weighted_sum;
                self.total_weight = 1.0 + self.decay * self.total_weight;
            }
        }
        self.pending_token_id = LLAMA_TOKEN_NULL;
        self.pending_token_idx = -1;
    }

    fn reset(&mut self) {
        // target and decay never change after init; original_probs is overwritten on apply
        self.weighted_sum = self.target / (1.0 - self.decay);
        self.total_weight = 1.0 / (1.0 - self.decay);
        self.pending_token_id = LLAMA_TOKEN_NULL;
        self.pending_token_idx = -1;
        self.seed_cur = get_rng_seed(self.seed);
        self.rng.reseed(self.seed_cur);
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// logit-bias
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct LogitBiasSampler {
    n_vocab: i32,
    logit_bias: Vec<LogitBias>,
    to_search: Vec<LogitBias>,
}

pub fn init_logit_bias(n_vocab: i32, logit_bias: &[LogitBias]) -> Box<dyn Sampler> {
    if logit_bias.is_empty() {
        return init_empty("?logit-bias");
    }
    Box::new(LogitBiasSampler {
        n_vocab,
        logit_bias: logit_bias.to_vec(),
        to_search: Vec::new(),
    })
}

impl Sampler for LogitBiasSampler {
    fn name(&self) -> &str {
        "logit-bias"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        if self.logit_bias.is_empty() {
            return;
        }

        self.to_search.clear();

        // update the candidates that have not been shuffled in the vocabulary (i.e. idx == id)
        for &lb in &self.logit_bias {
            if lb.token >= 0
                && cur.data.len() > lb.token as usize
                && cur.data[lb.token as usize].id == lb.token
            {
                cur.data[lb.token as usize].logit += lb.bias;
            } else {
                self.to_search.push(lb);
            }
        }

        if self.to_search.is_empty() {
            return;
        }

        // search for the remaining candidates that were not found in the previous step
        for td in &mut cur.data {
            for &lb in &self.to_search {
                if td.id == lb.token {
                    td.logit += lb.bias;
                    break;
                }
            }
        }
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// infill
// ---------------------------------------------------------------------------

struct Infill {
    vocab: Rc<dyn Vocab>,
}

pub fn init_infill(vocab: Rc<dyn Vocab>) -> Box<dyn Sampler> {
    Box::new(Infill { vocab })
}

impl Sampler for Infill {
    fn name(&self) -> &str {
        "infill"
    }

    fn apply(&mut self, cur: &mut TokenDataArray) {
        softmax_impl(cur, true);

        let mut p_txt_sum = 0.0f32;
        let mut p_eog_sum = 0.0f32;
        for td in &cur.data {
            if self.vocab.is_eog(td.id) {
                p_eog_sum += td.p;
            } else {
                p_txt_sum += td.p;
            }
        }

        if 3.0 * p_eog_sum * cur.data.len() as f32 > p_txt_sum {
            // keep just the EOG tokens
            let size_org = cur.data.len();
            let mut kept = Vec::new();
            let mut p_sum = 0.0f32;
            for td in cur.data.iter().take(size_org) {
                if self.vocab.is_eog(td.id) {
                    p_sum += td.p;
                    kept.push(*td);
                }
            }
            cur.data = kept;
            for td in &mut cur.data {
                td.p /= p_sum;
            }
            return;
        }

        // combine tokens with common prefix
        for i0 in 0..cur.data.len() {
            for i1 in 0..cur.data.len() {
                let (l0, p0, id0) = (cur.data[i0].logit, cur.data[i0].p, cur.data[i0].id);
                let (l1, p1, id1) = (cur.data[i1].logit, cur.data[i1].p, cur.data[i1].id);
                if l0 == f32::NEG_INFINITY {
                    break;
                }
                if i0 == i1 || l1 == f32::NEG_INFINITY {
                    continue;
                }

                let b0 = self.vocab.token_to_piece(id0);
                let b1 = self.vocab.token_to_piece(id1);

                // token i0 is a prefix of token i1
                if !b0.is_empty() && b0.len() <= b1.len() && b0[..] == b1[..b0.len()] {
                    let (mut dst, mut src) = (i0, i1);
                    // merge into the token with higher probability
                    if p1 > p0 {
                        std::mem::swap(&mut dst, &mut src);
                    }
                    cur.data[dst].p += cur.data[src].p;
                    cur.data[src].logit = f32::NEG_INFINITY;
                    cur.data[src].p = 0.0;
                }
            }
        }

        let mut n_non_eog = 0usize;
        let size_org = cur.data.len();
        let mut p_sum = 0.0f32;
        let mut thold = 0.2f32;
        let mut kept = Vec::new();
        for td in cur.data.iter().take(size_org) {
            let is_eog = self.vocab.is_eog(td.id);
            if td.p < thold && !is_eog {
                continue;
            }
            if !is_eog {
                n_non_eog += 1;
            }
            p_sum += td.p;
            kept.push(*td);
        }
        cur.data = kept;

        // if no non-EOG tokens are left -> reduce cur_p to single EOT token
        if n_non_eog == 0 {
            cur.data.truncate(1);
            if cur.data.is_empty() {
                cur.data.push(TokenData::default());
            }
            let mut id = self.vocab.token_eot();
            if id == LLAMA_TOKEN_NULL {
                id = self.vocab.token_eos();
            }
            cur.data[0].id = id;
            cur.data[0].logit = 1.0;
            return;
        }

        for td in &mut cur.data {
            td.p /= p_sum;
        }

        let size_org = cur.data.len();
        p_sum = 0.0;
        thold = 1.0 / (n_non_eog + 1) as f32;
        let mut kept = Vec::new();
        for td in cur.data.iter().take(size_org) {
            let is_eog = self.vocab.is_eog(td.id);
            if td.p < thold && !is_eog {
                continue;
            }
            p_sum += td.p;
            kept.push(*td);
        }
        cur.data = kept;

        for td in &mut cur.data {
            td.p /= p_sum;
        }
    }

    fn clone_sampler(&self) -> Box<dyn Sampler> {
        Box::new(Infill {
            vocab: self.vocab.clone(),
        })
    }
}

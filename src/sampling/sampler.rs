//! CPU sampling algorithms from `llama-sampler.cpp`.
//!
//! Chain order follows the default `common_params_sampling` order in
//! `common/common.h`: top-k, typical-p, top-p, min-p, temperature, dist.
//! `min_keep` is fixed at 0 because this port has no public knob for it.

use crate::core::{Mt19937, LLAMA_TOKEN_NULL};

/// `llama_token_data`: one candidate in the sampler chain.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    id: i32,
    logit: f32,
    p: f32,
}

/// Self-contained CPU sampler chain.
///
/// Defaults are neutral (no filter, temperature 1, seed 0), so a fresh
/// [`Sampler`] samples from the full softmax distribution.
#[derive(Clone, Debug)]
pub struct Sampler {
    temp: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    typical_p: f32,
    seed: u32,
    rng: Mt19937,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temp: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            seed: 0,
            rng: Mt19937::new(0),
        }
    }
}

impl Sampler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Temperature scaling. `temp <= 0` keeps only the argmax (greedy).
    pub fn temp(mut self, temp: f32) -> Self {
        self.temp = temp;
        self
    }

    /// Keep the `k` highest-logit candidates. `0` disables the filter.
    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = k;
        self
    }

    /// Nucleus cutoff. `1.0` disables the filter; `0.0` keeps only the top token.
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Probability floor relative to the highest candidate. `0.0` disables it.
    pub fn min_p(mut self, p: f32) -> Self {
        self.min_p = p;
        self
    }

    /// Entropy-based locally typical cutoff. `1.0` disables the filter.
    pub fn typical_p(mut self, p: f32) -> Self {
        self.typical_p = p;
        self
    }

    /// Seed for the deterministic `Mt19937` draw. Resets the RNG.
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self.rng = Mt19937::new(seed);
        self
    }

    /// Reset RNG state to the configured seed. Sampler settings stay unchanged.
    pub fn clear(&mut self) {
        self.rng = Mt19937::new(self.seed);
    }

    /// Apply the chain and return the sampled token id.
    ///
    /// `logits[i]` is the logit of token `i`. The slice is not modified;
    /// candidate mutation happens on an internal copy. Invalid input (empty,
    /// NaN or infinite logits, or NaN settings) yields `LLAMA_TOKEN_NULL`.
    pub fn sample(&mut self, logits: &mut [f32]) -> i32 {
        if logits.is_empty()
            || !logits.iter().all(|l| l.is_finite())
            || self.temp.is_nan()
            || self.top_p.is_nan()
            || self.min_p.is_nan()
            || self.typical_p.is_nan()
        {
            return LLAMA_TOKEN_NULL;
        }

        let mut cur: Vec<Candidate> = logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| Candidate {
                id: i as i32,
                logit,
                p: 0.0,
            })
            .collect();
        let mut sorted = false;

        if self.top_k > 0 {
            let k = (self.top_k as usize).min(cur.len());
            if !sorted {
                sort_by_logit(&mut cur);
                sorted = true;
            }
            cur.truncate(k);
        }

        if self.typical_p < 1.0 {
            apply_typical(&mut cur, self.typical_p, &mut sorted);
        }

        if self.top_p < 1.0 {
            apply_top_p(&mut cur, self.top_p, &mut sorted);
        }

        if self.min_p > 0.0 {
            apply_min_p(&mut cur, self.min_p, &mut sorted);
        }

        if self.temp <= 0.0 {
            greedy_temp(&mut cur);
        } else {
            for c in &mut cur {
                c.logit /= self.temp;
            }
            if !cur.iter().all(|c| c.logit.is_finite()) {
                return LLAMA_TOKEN_NULL;
            }
        }

        sample_dist(&mut cur, &mut self.rng)
    }
}

/// Stable sort by descending logit; ties break by ascending token id.
fn sort_by_logit(cur: &mut [Candidate]) {
    cur.sort_by(|a, b| b.logit.total_cmp(&a.logit).then(a.id.cmp(&b.id)));
}

/// `llama_sampler_softmax_impl`.
fn softmax(cur: &mut [Candidate], do_sort: bool, sorted: &mut bool) {
    if cur.is_empty() {
        return;
    }

    if do_sort && !*sorted {
        sort_by_logit(cur);
        *sorted = true;
    }

    let mut max_l = cur[0].logit;
    if !*sorted {
        for c in &cur[1..] {
            max_l = max_l.max(c.logit);
        }
    }

    let mut cum_sum = 0.0f32;
    for c in cur.iter_mut() {
        c.p = (c.logit - max_l).exp();
        cum_sum += c.p;
    }
    // Finite logits always give exp(0) == 1 for the maximum, so cum_sum > 0.
    for c in cur.iter_mut() {
        c.p /= cum_sum;
    }
}

/// `llama_sampler_top_p_apply`. The >1024 adaptive-sort path is replaced by a
/// full sort: it yields the same kept set and draws no extra random numbers.
fn apply_top_p(cur: &mut Vec<Candidate>, p: f32, sorted: &mut bool) {
    if cur.is_empty() {
        return;
    }

    softmax(cur, false, sorted);
    if !*sorted {
        sort_by_logit(cur);
        *sorted = true;
    }

    let mut cum_sum = 0.0f32;
    let mut last_idx = cur.len();
    for (i, c) in cur.iter().enumerate() {
        cum_sum += c.p;
        if cum_sum >= p {
            last_idx = i + 1;
            break;
        }
    }
    cur.truncate(last_idx);
}

/// `llama_sampler_min_p_apply`.
fn apply_min_p(cur: &mut Vec<Candidate>, p: f32, sorted: &mut bool) {
    if cur.is_empty() {
        return;
    }

    let mut applied = false;
    if !*sorted {
        let max_logit = cur.iter().map(|c| c.logit).fold(f32::MIN, f32::max);
        let min_logit = max_logit + p.ln();
        let filtered: Vec<Candidate> = cur
            .iter()
            .copied()
            .filter(|c| c.logit >= min_logit)
            .collect();
        if !filtered.is_empty() {
            *cur = filtered;
            applied = true;
        }
    }

    if !applied {
        if !*sorted {
            sort_by_logit(cur);
            *sorted = true;
        }
        let min_logit = cur[0].logit + p.ln();
        let mut i = 1;
        while i < cur.len() && cur[i].logit >= min_logit {
            i += 1;
        }
        cur.truncate(i);
    }
}

/// `llama_sampler_typical_apply`.
fn apply_typical(cur: &mut Vec<Candidate>, p: f32, sorted: &mut bool) {
    if cur.is_empty() {
        return;
    }

    softmax(cur, true, sorted);

    // x*log(x) -> 0 as x -> 0; skipping zero-probability tokens avoids NaN.
    let mut entropy = 0.0f32;
    for c in cur.iter() {
        if c.p > 0.0 {
            entropy -= c.p * c.p.ln();
        }
    }

    let mut scored: Vec<(f32, usize, Candidate)> = cur
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let shift = if c.p > 0.0 {
                (-c.p.ln() - entropy).abs()
            } else {
                f32::INFINITY
            };
            (shift, i, *c)
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut cum_sum = 0.0f32;
    let mut last_idx = scored.len();
    for (i, (_, _, c)) in scored.iter().enumerate() {
        cum_sum += c.p;
        if cum_sum > p {
            last_idx = i + 1;
            break;
        }
    }

    for (dst, (_, _, c)) in cur.iter_mut().zip(scored.into_iter().take(last_idx)) {
        *dst = c;
    }
    cur.truncate(last_idx);
    *sorted = false;
}

/// `llama_sampler_temp_impl` for `temp <= 0`.
fn greedy_temp(cur: &mut [Candidate]) {
    let mut max_i = 0usize;
    let mut max_l = cur[0].logit;
    for i in 1..cur.len() {
        if cur[i].logit > max_l {
            cur[max_i].logit = f32::NEG_INFINITY;
            max_i = i;
            max_l = cur[i].logit;
        } else {
            cur[i].logit = f32::NEG_INFINITY;
        }
    }
}

/// `llama_sampler_dist_apply`.
fn sample_dist(cur: &mut [Candidate], rng: &mut Mt19937) -> i32 {
    if cur.is_empty() {
        return LLAMA_TOKEN_NULL;
    }

    if cur.len() == 1 {
        // Match backend sampling: one draw per output even for a single token.
        let _ = rng.canonical_f64();
        cur[0].p = 1.0;
        return cur[0].id;
    }

    let max_l = cur
        .iter()
        .map(|c| c.logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum_cum = 0.0f64;
    for c in cur.iter_mut() {
        c.p = (c.logit - max_l).exp();
        sum_cum += c.p as f64;
    }
    if !sum_cum.is_finite() || sum_cum <= 0.0 {
        return LLAMA_TOKEN_NULL;
    }

    let sum_tgt = sum_cum * rng.canonical_f64();
    let mut sum_run = 0.0f64;
    let mut selected = cur.len() - 1;
    let mut found = false;
    for (i, c) in cur.iter_mut().enumerate() {
        if !found {
            sum_run += c.p as f64;
            if sum_run >= sum_tgt {
                selected = i;
                found = true;
            }
        }
        c.p /= sum_cum as f32;
    }
    cur[selected].id
}

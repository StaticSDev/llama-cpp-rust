use crate::core::Token;
use std::cmp::Ordering;

use super::chain::partial_sort_inplace;
use super::rng::SamplerRng;

/// `LLAMA_DEFAULT_SEED` from `llama.h`.
pub const LLAMA_DEFAULT_SEED: u32 = u32::MAX;

/// `LLAMA_DRY_MAX_SEQ_BREAKERS` from `llama.h`.
pub const LLAMA_DRY_MAX_SEQ_BREAKERS: usize = 64;

/// `llama_token_data`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TokenData {
    pub id: Token,
    pub logit: f32,
    pub p: f32,
}

/// `llama_token_data_array`. `data.len()` is the C++ `size` field.
#[derive(Clone, Debug, Default)]
pub struct TokenDataArray {
    pub data: Vec<TokenData>,
    pub selected: i32,
    pub sorted: bool,
}

impl TokenDataArray {
    pub fn new(data: Vec<TokenData>) -> Self {
        Self {
            data,
            selected: -1,
            sorted: false,
        }
    }
}

/// `llama_sampler_chain_params`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChainParams {
    pub no_perf: bool,
}

/// `llama_logit_bias`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LogitBias {
    pub token: Token,
    pub bias: f32,
}

/// Stand-in for the parts of `llama_vocab` used by samplers.
pub trait Vocab {
    fn n_tokens(&self) -> usize;
    fn detokenize(&self, token: Token, special: bool) -> Vec<u8>;
    fn tokenize(&self, text: &[u8], bos: bool, special: bool) -> Vec<Token>;
    fn is_eog(&self, token: Token) -> bool;
    fn token_eot(&self) -> Token;
    fn token_eos(&self) -> Token;
    fn token_to_piece(&self, token: Token) -> Vec<u8>;
}

/// Stand-in for `llama_grammar`: forwards candidate filtering and accept.
pub trait Grammar {
    fn apply(&mut self, cur: &mut TokenDataArray);
    fn accept(&mut self, token: Token);
    fn reset(&mut self);
    fn clone_box(&self) -> Box<dyn Grammar>;
}

pub(crate) fn cmp_logit(a: &TokenData, b: &TokenData) -> Ordering {
    b.logit.partial_cmp(&a.logit).unwrap_or(Ordering::Equal)
}

pub(crate) fn sample_dist(cur: &TokenDataArray, rng: &mut SamplerRng) -> usize {
    if cur.data.is_empty() {
        return 0;
    }

    let mut cum = Vec::with_capacity(cur.data.len());
    let mut sum = 0.0f64;
    for td in &cur.data {
        sum += td.p as f64;
        cum.push(sum);
    }
    if sum <= 0.0 {
        return 0;
    }

    let r = rng.uniform_f64() * sum;
    // std::upper_bound on the cumulative weights
    let mut lo = 0usize;
    let mut hi = cum.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cum[mid] <= r {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo.min(cur.data.len() - 1)
}

pub(crate) fn temp_impl(cur: &mut TokenDataArray, temp: f32) {
    if cur.data.is_empty() {
        return;
    }

    if temp <= 0.0 {
        // find the token with the highest logit and set the rest to -inf
        let mut max_i = 0usize;
        let mut max_l = cur.data[0].logit;

        for i in 1..cur.data.len() {
            if cur.data[i].logit > max_l {
                cur.data[max_i].logit = f32::NEG_INFINITY;
                max_i = i;
                max_l = cur.data[i].logit;
            } else {
                cur.data[i].logit = f32::NEG_INFINITY;
            }
        }
        return;
    }

    for td in &mut cur.data {
        td.logit /= temp;
    }
}

pub(crate) fn softmax_impl(cur: &mut TokenDataArray, do_sort: bool) {
    assert!(!cur.data.is_empty(), "softmax: empty candidates");

    // Sort the logits in descending order if requested
    if do_sort && !cur.sorted {
        partial_sort_inplace(cur, cur.data.len());
    }

    let mut max_l = cur.data[0].logit;
    if !cur.sorted {
        for td in &cur.data[1..] {
            max_l = max_l.max(td.logit);
        }
    }

    let mut cum_sum = 0.0f32;
    for td in &mut cur.data {
        let p = (td.logit - max_l).exp();
        td.p = p;
        cum_sum += p;
    }
    for td in &mut cur.data {
        td.p /= cum_sum;
    }
}

pub(crate) fn top_k_impl(cur: &mut TokenDataArray, k: i32) {
    if k <= 0 {
        return;
    }

    let k = (k as usize).min(cur.data.len());

    // Sort scores in descending order
    if !cur.sorted {
        partial_sort_inplace(cur, k);
    }

    cur.data.truncate(k);
}

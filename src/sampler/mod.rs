//! Rust port of `llama-sampler.cpp` and `llama-sampler.h`.
//!
//! CPU-side sampling algorithms are ported in full. The ggml backend graph
//! helpers (`backend_init`, `backend_apply`, `backend_set_input`, graph probes
//! and `backend_n_nodes`) are not present: this crate has no ggml layer. The
//! backend hooks that only manage sampler state (`backend_begin`,
//! `backend_reset` bookkeeping) are kept where they do not need a graph.
//!
//! Vocabulary- and grammar-dependent samplers talk to small local traits
//! (`Vocab`, `Grammar`) that stand in for `llama_vocab` / `llama_grammar`.

mod builtin;
mod candidate;
mod chain;
mod rng;

#[cfg(test)]
mod tests;

pub use builtin::{
    init_adaptive_p, init_dry, init_dry_testing, init_dist, init_empty, init_greedy, init_grammar,
    init_infill, init_logit_bias, init_min_p, init_mirostat, init_mirostat_v2, init_penalties,
    init_temp, init_temp_ext, init_top_k, init_top_n_sigma, init_top_p, init_typical, init_xtc,
};
pub use candidate::{
    ChainParams, Grammar, LogitBias, TokenData, TokenDataArray, Vocab, LLAMA_DEFAULT_SEED,
    LLAMA_DRY_MAX_SEQ_BREAKERS,
};
pub use chain::{
    backend_begin, chain_add, chain_get, chain_n, chain_remove, copy_sampler, get_seed, init_chain,
    perf_sampler, perf_sampler_reset, sample, Chain, PerfSamplerData, RingBuffer, Sampler,
};
pub use rng::get_rng_seed;

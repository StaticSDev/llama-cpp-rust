//! Rust port of `common/speculative.h` and the pure helpers in `common/speculative.cpp`.
//!
//! Only data helpers are ported. The C++ implementation classes drive a live
//! target context, draft context, sampler chain and backend device; those need
//! FFI and are listed as omissions at the end of this file.

use crate::common::{Params, ParamsSpeculative, SpeculativeType, SplitMode};
use crate::core::{Error, Result, Token};
use crate::cparams::PoolingType;
use crate::gguf::{
    gguf_find_key, gguf_find_tensor, gguf_get_kv_type, gguf_get_val_str, gguf_get_val_u32,
    gguf_init_from_file, GgufInitParams, GgufType,
};
use crate::sampler::{TokenData, LLAMA_DEFAULT_SEED};
use crate::vocab::{
    llama_vocab_bos, llama_vocab_eos, llama_vocab_get_add_bos, llama_vocab_get_add_eos,
    llama_vocab_get_text, llama_vocab_n_tokens, llama_vocab_type, LlamaVocab,
};
use std::sync::OnceLock;

/// `SPEC_VOCAB_MAX_SIZE_DIFFERENCE`.
pub const SPEC_VOCAB_MAX_SIZE_DIFFERENCE: i32 = 128;
/// `SPEC_VOCAB_CHECK_START_TOKEN_ID`.
pub const SPEC_VOCAB_CHECK_START_TOKEN_ID: i32 = 5;
/// `COMMON_SPECULATIVE_TYPE_COUNT`.
pub const SPECULATIVE_TYPE_COUNT: usize = 13;

const ALL_TYPES: [SpeculativeType; SPECULATIVE_TYPE_COUNT] = [
    SpeculativeType::None,
    SpeculativeType::DraftSimple,
    SpeculativeType::DraftEagle3,
    SpeculativeType::DraftMtp,
    SpeculativeType::DraftDflash,
    SpeculativeType::DraftDspark,
    SpeculativeType::DraftDflash2,
    SpeculativeType::DraftLilicorr,
    SpeculativeType::NgramSimple,
    SpeculativeType::NgramMapK,
    SpeculativeType::NgramMapK4v,
    SpeculativeType::NgramMod,
    SpeculativeType::NgramCache,
];

/// `common_speculative_output_limits`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeculativeOutputLimits {
    pub total: i32,
    pub per_seq: i32,
}

/// Data-only form of `common_speculative_draft_params`; the pointer fields of
/// the C++ struct become owned vectors here.
#[derive(Clone, Debug)]
pub struct SpeculativeDraftParams {
    pub drafting: bool,
    pub n_max: i32,
    pub pos0: i32,
    pub id_last: Token,
    pub prompt: Vec<Token>,
    pub result: Vec<Token>,
    pub result_q: Option<Vec<Vec<TokenData>>>,
    pub temp: f32,
    pub seed: u32,
}

impl Default for SpeculativeDraftParams {
    fn default() -> Self {
        Self {
            drafting: false,
            n_max: -1,
            pos0: 0,
            id_last: crate::core::LLAMA_TOKEN_NULL,
            prompt: Vec::new(),
            result: Vec::new(),
            result_q: None,
            temp: 1.0,
            seed: LLAMA_DEFAULT_SEED,
        }
    }
}

/// Data-only mirror of the counters on `common_speculative_impl`, used by
/// `common_speculative_print_stats`.
#[derive(Clone, Debug)]
pub struct SpeculativeStats {
    pub type_: SpeculativeType,
    pub n_seq: u32,
    pub n_max: i32,
    pub n_call_begin: usize,
    pub n_call_draft: usize,
    pub n_call_accept: usize,
    pub n_gen_drafts: usize,
    pub n_acc_drafts: usize,
    pub n_gen_tokens: usize,
    pub n_acc_tokens: usize,
    pub n_acc_tokens_per_pos: Vec<usize>,
    pub gen_perf: bool,
    pub t_begin_us: i64,
    pub t_draft_us: i64,
    pub t_accept_us: i64,
}

impl Default for SpeculativeStats {
    fn default() -> Self {
        Self {
            type_: SpeculativeType::None,
            n_seq: 0,
            n_max: 0,
            n_call_begin: 0,
            n_call_draft: 0,
            n_call_accept: 0,
            n_gen_drafts: 0,
            n_acc_drafts: 0,
            n_gen_tokens: 0,
            n_acc_tokens: 0,
            n_acc_tokens_per_pos: Vec::new(),
            gen_perf: true,
            t_begin_us: 0,
            t_draft_us: 0,
            t_accept_us: 0,
        }
    }
}

/// `common_speculative_type_to_str`.
pub fn common_speculative_type_to_str(type_: SpeculativeType) -> &'static str {
    match type_ {
        SpeculativeType::None => "none",
        SpeculativeType::DraftSimple => "draft-simple",
        SpeculativeType::DraftEagle3 => "draft-eagle3",
        SpeculativeType::DraftMtp => "draft-mtp",
        SpeculativeType::DraftDflash => "draft-dflash",
        SpeculativeType::DraftDspark => "draft-dspark",
        SpeculativeType::DraftDflash2 => "draft-dflash2",
        SpeculativeType::DraftLilicorr => "draft-lilicorr",
        SpeculativeType::NgramSimple => "ngram-simple",
        SpeculativeType::NgramMapK => "ngram-map-k",
        SpeculativeType::NgramMapK4v => "ngram-map-k4v",
        SpeculativeType::NgramMod => "ngram-mod",
        SpeculativeType::NgramCache => "ngram-cache",
    }
}

/// `common_speculative_type_from_name`. `None` stands for the C++ sentinel
/// `COMMON_SPECULATIVE_TYPE_COUNT`, which the enum here has no variant for.
pub fn common_speculative_type_from_name(name: &str) -> Option<SpeculativeType> {
    match name {
        "none" => Some(SpeculativeType::None),
        "draft-simple" => Some(SpeculativeType::DraftSimple),
        "draft-eagle3" => Some(SpeculativeType::DraftEagle3),
        "draft-mtp" => Some(SpeculativeType::DraftMtp),
        "draft-dflash" => Some(SpeculativeType::DraftDflash),
        "draft-dspark" => Some(SpeculativeType::DraftDspark),
        "draft-dflash2" => Some(SpeculativeType::DraftDflash2),
        "draft-lilicorr" => Some(SpeculativeType::DraftLilicorr),
        "ngram-simple" => Some(SpeculativeType::NgramSimple),
        "ngram-map-k" => Some(SpeculativeType::NgramMapK),
        "ngram-map-k4v" => Some(SpeculativeType::NgramMapK4v),
        "ngram-mod" => Some(SpeculativeType::NgramMod),
        "ngram-cache" => Some(SpeculativeType::NgramCache),
        _ => None,
    }
}

/// `common_speculative_type_name_str`.
pub fn common_speculative_type_name_str(types: &[SpeculativeType]) -> String {
    types
        .iter()
        .map(|&t| common_speculative_type_to_str(t))
        .collect::<Vec<_>>()
        .join(",")
}

/// `common_speculative_all_types_str`.
pub fn common_speculative_all_types_str() -> &'static str {
    static ALL_TYPES_STR: OnceLock<String> = OnceLock::new();
    ALL_TYPES_STR.get_or_init(|| common_speculative_type_name_str(&ALL_TYPES))
}

/// `common_speculative_types_from_names`. The C++ function throws on an
/// unknown name; this returns an error instead.
pub fn common_speculative_types_from_names(names: &[String]) -> Result<Vec<SpeculativeType>> {
    let mut types = Vec::with_capacity(names.len());
    for name in names {
        match common_speculative_type_from_name(name) {
            Some(SpeculativeType::None) => return Ok(vec![SpeculativeType::None]),
            Some(t) => types.push(t),
            None => {
                return Err(Error::new(format!("unknown speculative type: {name}")));
            }
        }
    }
    Ok(types)
}

/// `common_speculative_types_from_gguf`. A missing or mistyped `block_count`
/// key returns no types; the C++ code reads it unchecked.
pub fn common_speculative_types_from_gguf(path: &str) -> Vec<SpeculativeType> {
    let mut gguf_params = GgufInitParams {
        no_alloc: true,
        ctx: None,
    };
    let Ok(ctx) = gguf_init_from_file(path, &mut gguf_params) else {
        return Vec::new();
    };

    let arch_id = gguf_find_key(&ctx, "general.architecture");
    if arch_id < 0 || gguf_get_kv_type(&ctx, arch_id) != GgufType::String {
        return Vec::new();
    }
    let arch = gguf_get_val_str(&ctx, arch_id);
    if arch == "dflash" {
        return vec![SpeculativeType::DraftDflash];
    }

    let block_id = gguf_find_key(&ctx, &format!("{arch}.block_count"));
    if block_id < 0 || gguf_get_kv_type(&ctx, block_id) != GgufType::Uint32 {
        return Vec::new();
    }
    let block_count = gguf_get_val_u32(&ctx, block_id);
    let tensor = format!("blk.{}.nextn.eh_proj.weight", block_count.saturating_sub(1));
    if gguf_find_tensor(&ctx, &tensor) >= 0 {
        return vec![SpeculativeType::DraftMtp];
    }
    Vec::new()
}

/// `common_speculative_n_max(const common_params_speculative *)`.
pub fn common_speculative_n_max(spec: &ParamsSpeculative) -> i32 {
    let mut n_max = 0;
    for &t in &spec.types {
        match t {
            SpeculativeType::DraftSimple
            | SpeculativeType::DraftEagle3
            | SpeculativeType::DraftMtp
            | SpeculativeType::DraftDflash
            | SpeculativeType::DraftDspark
            | SpeculativeType::DraftDflash2
            | SpeculativeType::DraftLilicorr => {
                n_max = n_max.max(spec.draft.n_max.max(0));
            }
            SpeculativeType::NgramSimple => {
                n_max = n_max.max(spec.ngram_simple.size_m as i32);
            }
            SpeculativeType::NgramMapK => {
                n_max = n_max.max(spec.ngram_map_k.size_m as i32);
            }
            SpeculativeType::NgramMapK4v => {
                n_max = n_max.max(spec.ngram_map_k4v.size_m as i32);
            }
            SpeculativeType::NgramMod => {
                n_max = n_max.max(spec.ngram_mod.n_max.max(0));
            }
            SpeculativeType::NgramCache => {
                n_max = n_max.max(8);
            }
            SpeculativeType::None => {}
        }
    }
    n_max
}

/// `common_speculative_n_max(const common_speculative *)`. The C++ object
/// holds initialized implementations; here callers pass each impl's `n_max`.
pub fn common_speculative_n_max_from_drafts(draft_n_max: &[i32]) -> i32 {
    draft_n_max.iter().fold(0, |acc, &n| acc.max(n.max(0)))
}

/// `common_speculative_synth_rates_resolve`.
pub fn common_speculative_synth_rates_resolve(
    spec: &ParamsSpeculative,
    n_max: i32,
) -> Result<Vec<f64>> {
    let has_length = spec.synth_len != -1.0;
    let has_rates = !spec.synth_rates.is_empty();

    if !has_length && !has_rates {
        return Ok(Vec::new());
    }
    if has_length && has_rates {
        return Err(Error::new(
            "synthetic acceptance length and rates are mutually exclusive",
        ));
    }
    if n_max <= 0 {
        return Err(Error::new(
            "synthetic acceptance requires at least one speculative token",
        ));
    }

    if has_rates {
        let rates = &spec.synth_rates;
        if rates.len() != n_max as usize {
            return Err(Error::new(format!(
                "synthetic acceptance rates must contain {n_max} values, got {}",
                rates.len()
            )));
        }
        for i in 0..rates.len() {
            let rate = rates[i];
            if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
                return Err(Error::new(
                    "synthetic acceptance rates must be finite and within [0, 1]",
                ));
            }
            if i > 0 && rate > rates[i - 1] {
                return Err(Error::new(
                    "synthetic acceptance rates must be monotonically non-increasing",
                ));
            }
        }
        return Ok(rates.clone());
    }

    let length = spec.synth_len;
    let length_max = n_max as f64 + 1.0;
    if !length.is_finite() || !(1.0..=length_max).contains(&length) {
        return Err(Error::new(format!(
            "synthetic acceptance length must be finite and within [1, {length_max:.0}]"
        )));
    }

    let mut p = 0.0;
    if length == length_max {
        p = 1.0;
    } else if length > 1.0 {
        let mut p_min = 0.0;
        let mut p_max = 1.0;
        for _ in 0..32 {
            let p_mid = 0.5 * (p_min + p_max);
            let mut sum = 0.0;
            let mut term = p_mid;
            for _ in 0..n_max {
                sum += term;
                term *= p_mid;
            }
            if sum < length - 1.0 {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    let mut rates = Vec::with_capacity(n_max as usize);
    let mut rate = p;
    for _ in 0..n_max {
        rates.push(rate);
        rate *= p;
    }
    Ok(rates)
}

/// Conditional acceptance probabilities built by `common_speculative_init`
/// from the resolved rates. The C++ state stores this as `synth_probs`.
pub fn common_speculative_synth_probs(rates: &[f64]) -> Vec<f64> {
    let mut probs = Vec::with_capacity(rates.len());
    let mut rate_prev = 1.0;
    for &rate in rates {
        probs.push(if rate_prev > 0.0 { rate / rate_prev } else { 0.0 });
        rate_prev = rate;
    }
    probs
}

/// `common_base_params_to_speculative`.
pub fn common_base_params_to_speculative(params: &Params) -> Params {
    let has_draft = params.speculative.has_dft();
    let params_spec = &params.speculative.draft;
    let mut result = params.clone();

    result.embedding = false;
    result.pooling_type = PoolingType::Unspecified;

    if has_draft {
        // A Rust device vector has no null sentinel, so its length is the C++
        // count of non-null devices.
        if !params_spec.devices.is_empty() {
            result.devices = params_spec.devices.clone();
        }
        result.model = params_spec.mparams.clone();
        result.n_gpu_layers = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides.clone();
        if params_spec.devices.len() == 1 {
            result.split_mode = SplitMode::Layer;
        }
        if params_spec.cpuparams.n_threads > 0 {
            result.cpuparams.n_threads = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }
    }

    result.cache_type_k = params_spec.cache_type_k;
    result.cache_type_v = params_spec.cache_type_v;
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;
    if params_spec.n_ubatch > 0 {
        result.n_ubatch = params_spec.n_ubatch;
    }
    result.moe_cache_size = 0;

    let has_block_draft = params.speculative.types.iter().any(|&t| {
        matches!(
            t,
            SpeculativeType::DraftDflash
                | SpeculativeType::DraftDspark
                | SpeculativeType::DraftDflash2
                | SpeculativeType::DraftLilicorr
        )
    });
    if has_block_draft {
        let per_seq = 1.max(params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        if params_spec.backend_sampling {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    result
}

/// `common_speculative_get_output_limits`.
pub fn common_speculative_get_output_limits(
    n_batch: i32,
    n_parallel: i32,
    n_draft: i32,
) -> SpeculativeOutputLimits {
    let per_seq = 1i64 + n_draft.max(0) as i64;
    let total = n_parallel as i64 * per_seq;
    SpeculativeOutputLimits {
        total: total.min(n_batch as i64) as i32,
        per_seq: per_seq.min(n_batch as i64) as i32,
    }
}

/// `common_speculative_are_compatible`, fed by vocab handles instead of
/// target and draft model handles.
pub fn common_speculative_are_compatible(
    vocab_tgt: &LlamaVocab,
    vocab_dft: &LlamaVocab,
) -> bool {
    if llama_vocab_type(vocab_tgt) != llama_vocab_type(vocab_dft) {
        return false;
    }

    let add_bos_tgt = llama_vocab_get_add_bos(vocab_tgt);
    let add_bos_dft = llama_vocab_get_add_bos(vocab_dft);
    if add_bos_tgt != add_bos_dft
        || (add_bos_tgt && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))
    {
        return false;
    }

    let add_eos_tgt = llama_vocab_get_add_eos(vocab_tgt);
    let add_eos_dft = llama_vocab_get_add_eos(vocab_dft);
    if add_eos_tgt != add_eos_dft
        || (add_eos_tgt && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))
    {
        return false;
    }

    let n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
    let n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
    let vocab_diff = (n_vocab_tgt as i64 - n_vocab_dft as i64).abs();
    if vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE as i64 {
        return false;
    }

    for i in SPEC_VOCAB_CHECK_START_TOKEN_ID..n_vocab_tgt.min(n_vocab_dft) {
        if llama_vocab_get_text(vocab_tgt, i) != llama_vocab_get_text(vocab_dft, i) {
            return false;
        }
    }

    true
}

/// `common_speculative_print_stats`, returning the trace text instead of
/// sending it to the C++ logger (the `spec print_stats: ` log prefix is dropped).
pub fn common_speculative_print_stats(stats: &[SpeculativeStats]) -> String {
    let mut out = String::new();

    for s in stats {
        let str_perf = if s.gen_perf {
            format!(
                ", dur(b,g,a) = {:.3}, {:.3}, {:.3} ms",
                s.t_begin_us as f64 / 1000.0,
                s.t_draft_us as f64 / 1000.0,
                s.t_accept_us as f64 / 1000.0
            )
        } else {
            String::new()
        };

        let str_stats = if s.n_call_accept > 0 {
            let mean = 1.0 + s.n_acc_tokens as f64 / s.n_call_accept as f64;
            let per_pos = s
                .n_acc_tokens_per_pos
                .iter()
                .map(|&n| format!("{:.3}", n as f64 / s.n_call_accept as f64))
                .collect::<Vec<_>>()
                .join(", ");
            format!(", #mean acc len = {mean:.2}, #acc rate/pos = ({per_pos})")
        } else {
            String::new()
        };

        out.push_str(&format!(
            "statistics {:>16}: #calls(b,g,a) = {:4} {:6} {:6}, #gen drafts = {:6}, #acc drafts = {:5}, #gen tokens = {:6}, #acc tokens = {:5}{}{}\n",
            common_speculative_type_to_str(s.type_),
            s.n_call_begin,
            s.n_call_draft,
            s.n_call_accept,
            s.n_gen_drafts,
            s.n_acc_drafts,
            s.n_gen_tokens,
            s.n_acc_tokens,
            str_stats,
            str_perf
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_names_round_trip() {
        for &t in &ALL_TYPES {
            assert_eq!(common_speculative_type_from_name(common_speculative_type_to_str(t)), Some(t));
        }
        assert_eq!(
            common_speculative_type_name_str(&[SpeculativeType::None, SpeculativeType::DraftMtp]),
            "none,draft-mtp"
        );
        assert!(common_speculative_type_from_name("bogus").is_none());
        assert_eq!(common_speculative_all_types_str().split(',').count(), SPECULATIVE_TYPE_COUNT);
    }

    #[test]
    fn types_from_names_handles_none_and_errors() {
        let names = vec!["ngram-map-k".to_owned(), "draft-mtp".to_owned()];
        assert_eq!(
            common_speculative_types_from_names(&names).unwrap(),
            vec![SpeculativeType::NgramMapK, SpeculativeType::DraftMtp]
        );
        assert_eq!(
            common_speculative_types_from_names(&["none".to_owned()]).unwrap(),
            vec![SpeculativeType::None]
        );
        assert!(common_speculative_types_from_names(&["bogus".to_owned()]).is_err());
    }

    #[test]
    fn n_max_uses_largest_configured_drafter() {
        let mut spec = ParamsSpeculative::default();
        spec.types = vec![SpeculativeType::DraftSimple, SpeculativeType::NgramCache];
        spec.draft.n_max = 3;
        assert_eq!(common_speculative_n_max(&spec), 8);
        spec.types = vec![SpeculativeType::NgramMod];
        spec.ngram_mod.n_max = 64;
        assert_eq!(common_speculative_n_max(&spec), 64);
        assert_eq!(common_speculative_n_max_from_drafts(&[-1, 5, 2]), 5);
    }

    #[test]
    fn synth_rates_resolve_validates_and_solves_length() {
        let mut spec = ParamsSpeculative::default();
        assert!(common_speculative_synth_rates_resolve(&spec, 2)
            .unwrap()
            .is_empty());

        spec.synth_len = 2.0;
        let rates = common_speculative_synth_rates_resolve(&spec, 2).unwrap();
        assert_eq!(rates.len(), 2);
        let mean = 1.0 + rates.iter().sum::<f64>();
        assert!((mean - 2.0).abs() < 1e-3);
        assert_eq!(common_speculative_synth_probs(&rates).len(), 2);

        spec.synth_rates = vec![0.5, 0.25];
        assert!(common_speculative_synth_rates_resolve(&spec, 2).is_err());
        spec.synth_len = -1.0;
        assert_eq!(
            common_speculative_synth_rates_resolve(&spec, 2).unwrap(),
            vec![0.5, 0.25]
        );
        spec.synth_rates = vec![0.5];
        assert!(common_speculative_synth_rates_resolve(&spec, 2).is_err());
        spec.synth_rates = vec![0.25, 0.5];
        assert!(common_speculative_synth_rates_resolve(&spec, 2).is_err());
    }

    #[test]
    fn output_limits_clamp_to_batch() {
        assert_eq!(
            common_speculative_get_output_limits(8, 2, 3),
            SpeculativeOutputLimits {
                total: 8,
                per_seq: 4
            }
        );
    }

    #[test]
    fn base_params_applies_draft_overrides() {
        let mut params = Params::default();
        params.embedding = true;
        params.speculative.types = vec![SpeculativeType::DraftDflash];
        params.speculative.draft.n_max = 3;
        params.n_parallel = 2;
        let out = common_base_params_to_speculative(&params);
        assert!(!out.embedding);
        assert_eq!(out.n_outputs_max, 8);
        assert_eq!(out.n_outputs_max_per_seq, 4);
        assert_eq!(out.moe_cache_size, 0);
    }
}

// Omissions (require a live model/context/sampler/backend, all FFI):
// - every common_speculative_impl_* class (draft-simple, eagle3, mtp, dflash,
//   dspark, dflash2, lilicorr, ngram-simple/map-k/mod/cache) and spec_retune.
// - common_speculative_init/free, init_from_params and init_result, begin,
//   process, draft, accept, get_draft_params, get_state and set_state.
// - the static helpers tied to those classes: get_common_ngram_map,
//   create_state_ngram_* and common_get_enabled_speculative_configs.
// - SPC_* logging is dropped; helpers return data instead of writing log lines.

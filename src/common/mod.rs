//! Rust port of `common/common.h` and `common/common.cpp`.
//!
//! Pure-data helpers only. Functions whose C++ bodies need a live model,
//! context, sampler or backend handle stay out of this module; see the report
//! at the end of the porting task for the exact list. File paths stay `String`
//! or `PathBuf` and device handles become indices, so no FFI is needed.

use crate::core::{Error, Result, Token};
use crate::cparams::RopeScalingType as ContextRopeScalingType;
use crate::ggml::threading::{GGML_MAX_N_THREADS, SchedPriority, ThreadpoolParams};
use crate::gguf::{
    gguf_find_key, gguf_get_kv_type, gguf_get_n_tensors, gguf_get_tensor_name, gguf_get_tensor_type,
    ggml_n_dims, gguf_get_val_str, gguf_get_val_u32, gguf_init_from_file, GgufInitParams, GgufType,
    GgmlContext, GgmlType,
};
use crate::sampler::LogitBias;
use crate::unicode::unicode_cpt_from_utf8;
use crate::vocab::LlamaVocab;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub mod arg;
pub mod chat_parser;
pub mod console;
pub mod download;
pub mod jinja;
pub mod json_schema;
pub mod misc;
pub mod ngram;
pub mod sampling;
pub mod speculative;

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

/// `COMMON_RECURSIVE_LIMIT`.
pub const COMMON_RECURSIVE_LIMIT: usize = 128;
/// `llama_max_devices()` from `llama.cpp`.
pub const LLAMA_MAX_DEVICES: usize = 16;
/// `COMMON_DECISION_LFM3_D1_SET_SIZE`.
pub const COMMON_DECISION_LFM3_D1_SET_SIZE: usize = 16;

pub const LLM_KV_SPLIT_NO: &str = "split.no";
pub const LLM_KV_SPLIT_COUNT: &str = "split.count";
pub const LLM_KV_SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

pub const LLM_FFN_EXPS_REGEX: &str = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
pub const LLM_FFN_DENSE_REGEX: &str = "\\.ffn_(up|down|gate)\\.";

pub const META_SAMPLING_SEQUENCE: &str = "general.sampling.sequence";
pub const META_SAMPLING_TOP_K: &str = "general.sampling.top_k";
pub const META_SAMPLING_TOP_P: &str = "general.sampling.top_p";
pub const META_SAMPLING_MIN_P: &str = "general.sampling.min_p";
pub const META_SAMPLING_XTC_PROBABILITY: &str = "general.sampling.xtc_probability";
pub const META_SAMPLING_XTC_THRESHOLD: &str = "general.sampling.xtc_threshold";
pub const META_SAMPLING_TEMP: &str = "general.sampling.temp";
pub const META_SAMPLING_PENALTY_LAST_N: &str = "general.sampling.penalty_last_n";
pub const META_SAMPLING_PENALTY_REPEAT: &str = "general.sampling.penalty_repeat";
pub const META_SAMPLING_MIROSTAT: &str = "general.sampling.mirostat";
pub const META_SAMPLING_MIROSTAT_TAU: &str = "general.sampling.mirostat_tau";
pub const META_SAMPLING_MIROSTAT_ETA: &str = "general.sampling.mirostat_eta";

pub const SAMPLING_CONFIG_SAMPLERS: u64 = 1 << 0;
pub const SAMPLING_CONFIG_TOP_K: u64 = 1 << 1;
pub const SAMPLING_CONFIG_TOP_P: u64 = 1 << 2;
pub const SAMPLING_CONFIG_MIN_P: u64 = 1 << 3;
pub const SAMPLING_CONFIG_XTC_PROBABILITY: u64 = 1 << 4;
pub const SAMPLING_CONFIG_XTC_THRESHOLD: u64 = 1 << 5;
pub const SAMPLING_CONFIG_TEMP: u64 = 1 << 6;
pub const SAMPLING_CONFIG_PENALTY_LAST_N: u64 = 1 << 7;
pub const SAMPLING_CONFIG_PENALTY_REPEAT: u64 = 1 << 8;
pub const SAMPLING_CONFIG_MIROSTAT: u64 = 1 << 9;
pub const SAMPLING_CONFIG_MIROSTAT_TAU: u64 = 1 << 10;
pub const SAMPLING_CONFIG_MIROSTAT_ETA: u64 = 1 << 11;

// ---------------------------------------------------------------------------
// enums
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum LlamaExample {
    Batched,
    Debug,
    Common,
    Speculative,
    Completion,
    Cli,
    Embedding,
    Perplexity,
    Retrieval,
    Passkey,
    Imatrix,
    Bench,
    Server,
    CvectorGenerator,
    ExportLora,
    Mtmd,
    Lookup,
    Parallel,
    Tts,
    Diffusion,
    Finetune,
    FitParams,
    Results,
    ExportGraphOps,
    Download,
    Tokenize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SamplerType {
    None = 0,
    Dry = 1,
    TopK = 2,
    TopP = 3,
    MinP = 4,
    TypicalP = 6,
    Temperature = 7,
    Xtc = 8,
    Infill = 9,
    Penalties = 10,
    TopNSigma = 11,
    AdaptiveP = 12,
}

impl SamplerType {
    /// `common_sampler_type_to_str`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dry => "dry",
            Self::TopK => "top_k",
            Self::TypicalP => "typ_p",
            Self::TopP => "top_p",
            Self::TopNSigma => "top_n_sigma",
            Self::MinP => "min_p",
            Self::Temperature => "temperature",
            Self::Xtc => "xtc",
            Self::Infill => "infill",
            Self::Penalties => "penalties",
            Self::AdaptiveP => "adaptive_p",
            Self::None => "",
        }
    }

    /// `common_sampler_type_to_chr`.
    pub fn as_char(self) -> char {
        match self {
            Self::Dry => 'd',
            Self::TopK => 'k',
            Self::TypicalP => 'y',
            Self::TopP => 'p',
            Self::TopNSigma => 's',
            Self::MinP => 'm',
            Self::Temperature => 't',
            Self::Xtc => 'x',
            Self::Infill => 'i',
            Self::Penalties => 'e',
            Self::AdaptiveP => 'a',
            Self::None => '?',
        }
    }
}

/// `common_sampler_types_from_names`, with the same aliases.
pub fn sampler_types_from_names(names: &[String]) -> Vec<SamplerType> {
    let mut map: BTreeMap<String, SamplerType> = BTreeMap::new();
    for t in [
        SamplerType::Dry,
        SamplerType::TopK,
        SamplerType::TopP,
        SamplerType::TopNSigma,
        SamplerType::TypicalP,
        SamplerType::MinP,
        SamplerType::Temperature,
        SamplerType::Xtc,
        SamplerType::Infill,
        SamplerType::Penalties,
        SamplerType::AdaptiveP,
    ] {
        let canonical = t.as_str();
        map.insert(canonical.to_owned(), t);
        if canonical.contains('_') {
            map.insert(canonical.replace('_', "-"), t);
            map.insert(canonical.replace('_', ""), t);
        }
    }
    map.insert("nucleus".to_owned(), SamplerType::TopP);
    map.insert("temp".to_owned(), SamplerType::Temperature);
    map.insert("typ".to_owned(), SamplerType::TypicalP);

    let mut result = Vec::with_capacity(names.len());
    for name in names {
        let lower = name.to_lowercase();
        if let Some(t) = map.get(&lower) {
            result.push(*t);
        } else {
            eprintln!("sampler_types_from_names: unable to match sampler by name '{lower}'");
        }
    }
    result
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DimreMethod {
    #[default]
    Pca,
    Mean,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ConversationMode {
    Disabled = 0,
    Enabled = 1,
    Auto = 2,
}

impl Default for ConversationMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum GrammarTriggerType {
    Token = 0,
    Word = 1,
    Pattern = 2,
    PatternFull = 3,
}

#[derive(Clone, Debug)]
pub struct GrammarTrigger {
    pub type_: GrammarTriggerType,
    pub value: String,
    pub token: Token,
}

impl Default for GrammarTrigger {
    fn default() -> Self {
        Self {
            type_: GrammarTriggerType::Token,
            value: String::new(),
            token: crate::core::LLAMA_TOKEN_NULL,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SpeculativeType {
    None = 0,
    DraftSimple = 1,
    DraftEagle3 = 2,
    DraftMtp = 3,
    DraftDflash = 4,
    DraftDspark = 5,
    DraftDflash2 = 6,
    DraftLilicorr = 7,
    NgramSimple = 8,
    NgramMapK = 9,
    NgramMapK4v = 10,
    NgramMod = 11,
    NgramCache = 12,
}

impl Default for SpeculativeType {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum GrammarType {
    None = 0,
    User = 1,
    OutputFormat = 2,
    ToolCalls = 3,
}

#[derive(Clone, Debug)]
pub struct CommonGrammar {
    pub type_: GrammarType,
    pub grammar: String,
}

impl Default for CommonGrammar {
    fn default() -> Self {
        Self {
            type_: GrammarType::None,
            grammar: String::new(),
        }
    }
}

impl CommonGrammar {
    pub fn new(type_: GrammarType, grammar: String) -> Self {
        assert!(type_ != GrammarType::None || !grammar.is_empty());
        Self { type_, grammar }
    }

    pub fn empty(&self) -> bool {
        self.type_ == GrammarType::None || self.grammar.is_empty()
    }

    pub fn value(&self) -> &str {
        &self.grammar
    }

    pub fn needs_prefill(&self) -> bool {
        self.type_ == GrammarType::OutputFormat || self.type_ == GrammarType::ToolCalls
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ReasoningFormat {
    None,
    Auto,
    DeepseekLegacy,
    Deepseek,
}

impl Default for ReasoningFormat {
    fn default() -> Self {
        Self::Deepseek
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DecisionType {
    None,
    Openjev,
    Lev,
    Kev,
    Nimble,
    Laya,
    Clef,
    PplxDecider,
    VllmDecision2,
    VllmD3,
    Lfm2D1,
    Lfm2D1Omni,
    GraniteSwitch,
    H2oLightning,
    MinicpmOcrJev,
    Lfm3D1,
    Lfm3D1Omni,
    Unknown,
}

impl Default for DecisionType {
    fn default() -> Self {
        Self::Unknown
    }
}

impl DecisionType {
    /// `COMMON_DECISION_TYPE_NAMES` lookup.
    pub fn from_name(s: &str) -> Self {
        match s {
            "openjev" => Self::Openjev,
            "lev" => Self::Lev,
            "kev" => Self::Kev,
            "nimble" => Self::Nimble,
            "laya" => Self::Laya,
            "clef" => Self::Clef,
            "pplx-decider" => Self::PplxDecider,
            "vllm-decision2" => Self::VllmDecision2,
            "vllm-d3" => Self::VllmD3,
            "lfm2-d1" => Self::Lfm2D1,
            "lfm2-d1-omni" => Self::Lfm2D1Omni,
            "granite-switch" => Self::GraniteSwitch,
            "h2o-lightning" => Self::H2oLightning,
            "openbmb-minicpm-ocr-jev" => Self::MinicpmOcrJev,
            "lfm3-d1" => Self::Lfm3D1,
            "lfm3-d1-omni" => Self::Lfm3D1Omni,
            _ => Self::Unknown,
        }
    }

    pub fn as_name(self) -> Option<&'static str> {
        match self {
            Self::Openjev => Some("openjev"),
            Self::Lev => Some("lev"),
            Self::Kev => Some("kev"),
            Self::Nimble => Some("nimble"),
            Self::Laya => Some("laya"),
            Self::Clef => Some("clef"),
            Self::PplxDecider => Some("pplx-decider"),
            Self::VllmDecision2 => Some("vllm-decision2"),
            Self::VllmD3 => Some("vllm-d3"),
            Self::Lfm2D1 => Some("lfm2-d1"),
            Self::Lfm2D1Omni => Some("lfm2-d1-omni"),
            Self::GraniteSwitch => Some("granite-switch"),
            Self::H2oLightning => Some("h2o-lightning"),
            Self::MinicpmOcrJev => Some("openbmb-minicpm-ocr-jev"),
            Self::Lfm3D1 => Some("lfm3-d1"),
            Self::Lfm3D1Omni => Some("lfm3-d1-omni"),
            _ => None,
        }
    }
}

/// `common_get_decision_type`, fed by metadata instead of a model handle.
pub fn common_get_decision_type(
    general_architecture: Option<&str>,
    decision_type: Option<&str>,
) -> DecisionType {
    if general_architecture.is_none() || decision_type.is_none() {
        return DecisionType::None;
    }
    DecisionType::from_name(decision_type.unwrap_or_default())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SplitMode {
    None = 0,
    Layer = 1,
    Row = 2,
    Tensor = 3,
}

impl Default for SplitMode {
    fn default() -> Self {
        Self::Layer
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum LoadMode {
    Auto = -1,
    None = 0,
    Mmap = 1,
    Mlock = 2,
    MmapMlock = 3,
    DirectIo = 4,
}

impl Default for LoadMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum LazyMode {
    Off = 0,
    Auto = 1,
    On = 2,
}

impl Default for LazyMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum AttentionType {
    Unspecified = -1,
    Causal = 0,
    NonCausal = 1,
}

impl Default for AttentionType {
    fn default() -> Self {
        Self::Unspecified
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum FlashAttnType {
    Auto = -1,
    Disabled = 0,
    Enabled = 1,
}

impl Default for FlashAttnType {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ContextSeqRmType {
    No = 0,
    Part = 1,
    Full = 2,
    Rs = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum OptimizerType {
    Adamw,
    Sgd,
    Count,
}

impl Default for OptimizerType {
    fn default() -> Self {
        Self::Adamw
    }
}

/// `common_opt_get_optimizer`, case insensitive.
pub fn common_opt_get_optimizer(name: &str) -> OptimizerType {
    if name.eq_ignore_ascii_case("adamw") {
        OptimizerType::Adamw
    } else if name.eq_ignore_ascii_case("sgd") {
        OptimizerType::Sgd
    } else {
        OptimizerType::Count
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum KvOverrideType {
    Int,
    Float,
    Bool,
    Str,
}

/// `llama_model_kv_override` without the fixed C buffers.
#[derive(Clone, Debug, PartialEq)]
pub struct KvOverride {
    pub key: String,
    pub tag: KvOverrideType,
    pub val_i64: i64,
    pub val_f64: f64,
    pub val_bool: bool,
    pub val_str: String,
}

impl Default for KvOverride {
    fn default() -> Self {
        Self {
            key: String::new(),
            tag: KvOverrideType::Int,
            val_i64: 0,
            val_f64: 0.0,
            val_bool: false,
            val_str: String::new(),
        }
    }
}

/// `llama_model_tensor_buft_override`, with the buffer type kept as a name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorBuftOverride {
    pub pattern: String,
    pub buffer_type: String,
}

impl Default for TensorBuftOverride {
    fn default() -> Self {
        Self {
            pattern: String::new(),
            buffer_type: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// CPU utils
// ---------------------------------------------------------------------------

/// `common_cpu_params`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuParams {
    pub n_threads: i32,
    pub cpumask: Vec<bool>,
    pub mask_valid: bool,
    pub priority: SchedPriority,
    pub strict_cpu: bool,
    pub poll: u32,
}

impl Default for CpuParams {
    fn default() -> Self {
        Self {
            n_threads: -1,
            cpumask: vec![false; GGML_MAX_N_THREADS],
            mask_valid: false,
            priority: SchedPriority::Normal,
            strict_cpu: false,
            poll: 50,
        }
    }
}

/// Linux sysfs sibling sets, then the C++ thread-count fallback.
pub fn common_cpu_get_num_physical_cores() -> i32 {
    #[cfg(target_os = "linux")]
    {
        let mut siblings = HashSet::new();
        let mut cpu = 0u32;
        loop {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings");
            let Ok(text) = fs::read_to_string(path) else { break };
            siblings.insert(text);
            cpu = cpu.saturating_add(1);
        }
        if !siblings.is_empty() {
            return siblings.len() as i32;
        }
    }
    let n = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(0);
    if n > 0 {
        if n <= 4 { n } else { n / 2 }
    } else {
        4
    }
}

/// CPU count useful for math. Hybrid-core pinning needs OS calls, so on Linux
/// this is the physical-core count.
pub fn common_cpu_get_num_math() -> i32 {
    common_cpu_get_num_physical_cores()
}

/// `postprocess_cpu_params`. The warning is printed instead of logged.
pub fn postprocess_cpu_params(cpuparams: &mut CpuParams, role_model: Option<&CpuParams>) {
    if cpuparams.n_threads < 0 {
        match role_model {
            Some(model) => *cpuparams = model.clone(),
            None => cpuparams.n_threads = common_cpu_get_num_math(),
        }
    }

    let n_set = cpuparams.cpumask.iter().filter(|&&b| b).count() as i32;
    if n_set > 0 && n_set < cpuparams.n_threads {
        eprintln!(
            "Not enough set bits in CPU mask ({n_set}) to satisfy requested thread count: {}",
            cpuparams.n_threads
        );
    }
}

/// `parse_cpu_range`: `[<start>]-[<end>]`.
pub fn parse_cpu_range(range: &str, boolmask: &mut [bool; GGML_MAX_N_THREADS]) -> bool {
    let Some(dash) = range.find('-') else {
        eprintln!("Format of CPU range is invalid! Expected [<start>]-[<end>].");
        return false;
    };

    let start = if dash == 0 {
        0
    } else {
        match range[..dash].parse::<usize>() {
            Ok(v) if v < GGML_MAX_N_THREADS => v,
            _ => {
                eprintln!("Start index out of bounds!");
                return false;
            }
        }
    };

    let end = if dash == range.len() - 1 {
        GGML_MAX_N_THREADS - 1
    } else {
        match range[dash + 1..].parse::<usize>() {
            Ok(v) if v < GGML_MAX_N_THREADS => v,
            _ => {
                eprintln!("End index out of bounds!");
                return false;
            }
        }
    };

    for i in start..=end {
        boolmask[i] = true;
    }
    true
}

/// `parse_cpu_mask`, with an optional `0x` prefix.
pub fn parse_cpu_mask(mask: &str, boolmask: &mut [bool; GGML_MAX_N_THREADS]) -> bool {
    let start = if mask.len() >= 2 && mask.starts_with("0x") {
        2
    } else {
        0
    };
    let num_digits = (mask.len() - start).min(128);
    let end = start + num_digits;

    for (i, c) in mask.as_bytes()[start..end].iter().enumerate() {
        let id = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => {
                eprintln!(
                    "Invalid hex character '{}' at position {}",
                    *c as char,
                    start + i
                );
                return false;
            }
        };
        let n = num_digits * 4 - 1 - i * 4;
        boolmask[n] |= id & 8 != 0;
        boolmask[n - 1] |= id & 4 != 0;
        boolmask[n - 2] |= id & 2 != 0;
        boolmask[n - 3] |= id & 1 != 0;
    }
    true
}

/// `ggml_threadpool_params_from_cpu_params`.
pub fn ggml_threadpool_params_from_cpu_params(params: &CpuParams) -> ThreadpoolParams {
    let mut tpp = ThreadpoolParams::new(params.n_threads.max(0) as usize);
    if params.mask_valid {
        tpp.cpumask.copy_from_slice(&params.cpumask);
    }
    tpp.prio = params.priority.clone();
    tpp.poll = params.poll;
    tpp.strict_cpu = params.strict_cpu;
    tpp
}

/// Drop-in for `common_time_meas`: adds elapsed microseconds to `t_acc`.
pub struct CommonTimeMeas<'a> {
    t_start: Option<Instant>,
    t_acc: &'a mut i64,
}

impl<'a> CommonTimeMeas<'a> {
    pub fn new(t_acc: &'a mut i64, disable: bool) -> Self {
        Self {
            t_start: if disable { None } else { Some(Instant::now()) },
            t_acc,
        }
    }
}

impl Drop for CommonTimeMeas<'_> {
    fn drop(&mut self) {
        if let Some(start) = self.t_start {
            *self.t_acc += start.elapsed().as_micros() as i64;
        }
    }
}

// ---------------------------------------------------------------------------
// model params
// ---------------------------------------------------------------------------

/// `common_adapter_lora_info`, without the loaded adapter pointer.
#[derive(Clone, Debug, Default)]
pub struct AdapterLoraInfo {
    pub path: String,
    pub scale: f32,
    pub task_name: String,
    pub prompt_prefix: String,
}

#[derive(Clone, Debug, Default)]
pub struct ControlVectorData {
    pub n_embd: i32,
    /// Data for layers [1, n_layer].
    pub data: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct ControlVectorLoadInfo {
    pub strength: f32,
    pub fname: String,
}

#[derive(Clone, Debug)]
pub struct ParamsSampling {
    pub seed: u32,
    pub n_prev: i32,
    pub n_probs: i32,
    pub min_keep: i32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub xtc_probability: f32,
    pub xtc_threshold: f32,
    pub typ_p: f32,
    pub temp: f32,
    pub dynatemp_range: f32,
    pub dynatemp_exponent: f32,
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
    pub adaptive_target: f32,
    pub adaptive_decay: f32,
    pub mirostat: i32,
    pub top_n_sigma: f32,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub ignore_eos: bool,
    pub no_perf: bool,
    pub timing_per_token: bool,
    pub user_sampling_config: u64,
    pub dry_sequence_breakers: Vec<String>,
    pub samplers: Vec<SamplerType>,
    pub grammar: CommonGrammar,
    pub grammar_lazy: bool,
    pub grammar_triggers: Vec<GrammarTrigger>,
    pub preserved_tokens: BTreeSet<Token>,
    pub logit_bias: Vec<LogitBias>,
    pub logit_bias_eog: Vec<LogitBias>,
    pub generation_prompt: String,
    pub reasoning_budget_tokens: i32,
    pub reasoning_budget_start: Vec<Token>,
    pub reasoning_budget_end: Vec<Vec<Token>>,
    pub reasoning_budget_forced: Vec<Token>,
    pub reasoning_budget_message: String,
    pub reasoning_control: bool,
    pub backend_sampling: bool,
}

impl Default for ParamsSampling {
    fn default() -> Self {
        Self {
            seed: u32::MAX,
            n_prev: 64,
            n_probs: 0,
            min_keep: 0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            xtc_probability: 0.0,
            xtc_threshold: 0.10,
            typ_p: 1.0,
            temp: 0.80,
            dynatemp_range: 0.0,
            dynatemp_exponent: 1.0,
            penalty_last_n: 64,
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 64,
            adaptive_target: -1.0,
            adaptive_decay: 0.90,
            mirostat: 0,
            top_n_sigma: -1.0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.10,
            ignore_eos: false,
            no_perf: false,
            timing_per_token: false,
            user_sampling_config: 0,
            dry_sequence_breakers: vec!["\n".into(), ":".into(), "\"".into(), "*".into()],
            samplers: vec![
                SamplerType::Penalties,
                SamplerType::Dry,
                SamplerType::TopNSigma,
                SamplerType::TopK,
                SamplerType::TypicalP,
                SamplerType::TopP,
                SamplerType::MinP,
                SamplerType::Xtc,
                SamplerType::Temperature,
            ],
            grammar: CommonGrammar::default(),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            preserved_tokens: BTreeSet::new(),
            logit_bias: Vec::new(),
            logit_bias_eog: Vec::new(),
            generation_prompt: String::new(),
            reasoning_budget_tokens: -1,
            reasoning_budget_start: Vec::new(),
            reasoning_budget_end: Vec::new(),
            reasoning_budget_forced: Vec::new(),
            reasoning_budget_message: String::new(),
            reasoning_control: false,
            backend_sampling: false,
        }
    }
}

impl ParamsSampling {
    /// `common_params_sampling::print`.
    pub fn print(&self) -> String {
        format!(
            "\trepeat_last_n = {}, repeat_penalty = {:.3}, frequency_penalty = {:.3}, presence_penalty = {:.3}\n\
             \tdry_multiplier = {:.3}, dry_base = {:.3}, dry_allowed_length = {}, dry_penalty_last_n = {}\n\
             \ttop_k = {}, top_p = {:.3}, min_p = {:.3}, xtc_probability = {:.3}, xtc_threshold = {:.3}, typical_p = {:.3}, top_n_sigma = {:.3}, temp = {:.3}\n\
             \tmirostat = {}, mirostat_lr = {:.3}, mirostat_ent = {:.3}, adaptive_target = {:.3}, adaptive_decay = {:.3}",
            self.penalty_last_n,
            self.penalty_repeat,
            self.penalty_freq,
            self.penalty_present,
            self.dry_multiplier,
            self.dry_base,
            self.dry_allowed_length,
            self.dry_penalty_last_n,
            self.top_k,
            self.top_p,
            self.min_p,
            self.xtc_probability,
            self.xtc_threshold,
            self.typ_p,
            self.top_n_sigma,
            self.temp,
            self.mirostat,
            self.mirostat_eta,
            self.mirostat_tau,
            self.adaptive_target,
            self.adaptive_decay
        )
    }
}

/// `common_init_sampler_from_model`, with GGUF metadata as input.
pub fn common_init_sampler_from_meta(
    meta: &BTreeMap<String, String>,
    sparams: &mut ParamsSampling,
) {
    let config = sparams.user_sampling_config;

    let get_i32 = |key: &str, dst: &mut i32, user: u64| {
        if config_bit(config, user) {
            return;
        }
        if let Some(v) = meta.get(key).and_then(|s| parse_i64_prefix(s)) {
            *dst = v as i32;
        }
    };
    let get_f32 = |key: &str, dst: &mut f32, user: u64| {
        if config_bit(config, user) {
            return;
        }
        if let Some(v) = meta.get(key).and_then(|s| parse_f64_prefix(s)) {
            *dst = v as f32;
        }
    };

    if !config_bit(config, SAMPLING_CONFIG_SAMPLERS) {
        if let Some(seq) = meta.get(META_SAMPLING_SEQUENCE) {
            let names: Vec<String> = string_split_char(seq, ';');
            let parsed = sampler_types_from_names(&names);
            if !parsed.is_empty() {
                sparams.samplers = parsed;
            }
        }
    }

    get_i32(META_SAMPLING_TOP_K, &mut sparams.top_k, SAMPLING_CONFIG_TOP_K);
    get_f32(META_SAMPLING_TOP_P, &mut sparams.top_p, SAMPLING_CONFIG_TOP_P);
    get_f32(META_SAMPLING_MIN_P, &mut sparams.min_p, SAMPLING_CONFIG_MIN_P);
    get_f32(
        META_SAMPLING_XTC_PROBABILITY,
        &mut sparams.xtc_probability,
        SAMPLING_CONFIG_XTC_PROBABILITY,
    );
    get_f32(
        META_SAMPLING_XTC_THRESHOLD,
        &mut sparams.xtc_threshold,
        SAMPLING_CONFIG_XTC_THRESHOLD,
    );
    get_f32(META_SAMPLING_TEMP, &mut sparams.temp, SAMPLING_CONFIG_TEMP);
    get_i32(
        META_SAMPLING_PENALTY_LAST_N,
        &mut sparams.penalty_last_n,
        SAMPLING_CONFIG_PENALTY_LAST_N,
    );
    get_f32(
        META_SAMPLING_PENALTY_REPEAT,
        &mut sparams.penalty_repeat,
        SAMPLING_CONFIG_PENALTY_REPEAT,
    );
    get_i32(
        META_SAMPLING_MIROSTAT,
        &mut sparams.mirostat,
        SAMPLING_CONFIG_MIROSTAT,
    );
    get_f32(
        META_SAMPLING_MIROSTAT_TAU,
        &mut sparams.mirostat_tau,
        SAMPLING_CONFIG_MIROSTAT_TAU,
    );
    get_f32(
        META_SAMPLING_MIROSTAT_ETA,
        &mut sparams.mirostat_eta,
        SAMPLING_CONFIG_MIROSTAT_ETA,
    );
}

fn config_bit(config: u64, user: u64) -> bool {
    config & user != 0
}

#[derive(Clone, Debug, Default)]
pub struct ParamsModel {
    pub path: String,
    pub url: String,
    pub hf_repo: String,
    pub hf_file: String,
    pub docker_repo: String,
    pub shards: Vec<String>,
}

impl ParamsModel {
    /// `common_params_model::get_paths`.
    pub fn get_paths(&self) -> Vec<String> {
        let mut paths = vec![self.path.clone()];
        for shard in &self.shards {
            if !paths.contains(shard) {
                paths.push(shard.clone());
            }
        }
        paths
    }

    pub fn get_name(&self) -> &str {
        if !self.hf_repo.is_empty() {
            &self.hf_repo
        } else if !self.docker_repo.is_empty() {
            &self.docker_repo
        } else {
            &self.path
        }
    }

    pub fn is_empty(&self) -> bool {
        self.get_name().is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct ParamsSpeculativeDraft {
    pub n_max: i32,
    pub n_min: i32,
    pub p_split: f32,
    pub p_min: f32,
    pub backend_sampling: bool,
    pub probabilistic: bool,
    pub mparams: ParamsModel,
    pub n_gpu_layers: i32,
    pub n_ubatch: i32,
    pub cache_type_k: GgmlType,
    pub cache_type_v: GgmlType,
    pub cpuparams: CpuParams,
    pub cpuparams_batch: CpuParams,
    pub devices: Vec<usize>,
    pub tensor_buft_overrides: Vec<TensorBuftOverride>,
}

impl Default for ParamsSpeculativeDraft {
    fn default() -> Self {
        Self {
            n_max: 3,
            n_min: 0,
            p_split: 0.1,
            p_min: 0.0,
            backend_sampling: true,
            probabilistic: false,
            mparams: ParamsModel::default(),
            n_gpu_layers: -1,
            n_ubatch: 0,
            cache_type_k: GgmlType::F16,
            cache_type_v: GgmlType::F16,
            cpuparams: CpuParams::default(),
            cpuparams_batch: CpuParams::default(),
            devices: Vec::new(),
            tensor_buft_overrides: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamsSpeculativeNgramMod {
    pub n_match: i32,
    pub n_max: i32,
    pub n_min: i32,
}

impl Default for ParamsSpeculativeNgramMod {
    fn default() -> Self {
        Self {
            n_match: 24,
            n_max: 64,
            n_min: 48,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamsSpeculativeNgramMap {
    pub size_n: u16,
    pub size_m: u16,
    pub min_hits: u16,
}

impl Default for ParamsSpeculativeNgramMap {
    fn default() -> Self {
        Self {
            size_n: 12,
            size_m: 48,
            min_hits: 1,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ParamsSpeculativeNgramCache {
    pub lookup_cache_static: String,
    pub lookup_cache_dynamic: String,
}

#[derive(Clone, Debug)]
pub struct ParamsSpeculative {
    pub types: Vec<SpeculativeType>,
    pub synth_len: f64,
    pub synth_rates: Vec<f64>,
    pub draft: ParamsSpeculativeDraft,
    pub ngram_mod: ParamsSpeculativeNgramMod,
    pub ngram_simple: ParamsSpeculativeNgramMap,
    pub ngram_map_k: ParamsSpeculativeNgramMap,
    pub ngram_map_k4v: ParamsSpeculativeNgramMap,
    pub ngram_cache: ParamsSpeculativeNgramCache,
}

impl Default for ParamsSpeculative {
    fn default() -> Self {
        Self {
            types: vec![SpeculativeType::None],
            synth_len: -1.0,
            synth_rates: Vec::new(),
            draft: ParamsSpeculativeDraft::default(),
            ngram_mod: ParamsSpeculativeNgramMod::default(),
            ngram_simple: ParamsSpeculativeNgramMap::default(),
            ngram_map_k: ParamsSpeculativeNgramMap::default(),
            ngram_map_k4v: ParamsSpeculativeNgramMap::default(),
            ngram_cache: ParamsSpeculativeNgramCache::default(),
        }
    }
}

impl ParamsSpeculative {
    pub fn has_dft(&self) -> bool {
        !self.draft.mparams.is_empty()
    }

    pub fn has_synth(&self) -> bool {
        self.synth_len != -1.0 || !self.synth_rates.is_empty()
    }

    /// `common_params_speculative::need_n_rs_seq`.
    pub fn need_n_rs_seq(&self) -> u32 {
        let needs_rs_seq = self.types.iter().any(|t| {
            matches!(
                t,
                SpeculativeType::DraftMtp
                    | SpeculativeType::DraftEagle3
                    | SpeculativeType::DraftDflash
                    | SpeculativeType::DraftDspark
                    | SpeculativeType::DraftDflash2
                    | SpeculativeType::DraftLilicorr
            )
        });
        if needs_rs_seq {
            self.draft.n_max.max(0) as u32
        } else {
            0
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParamsDiffusion {
    pub steps: i32,
    pub visual_mode: bool,
    pub eps: f32,
    pub block_length: i32,
    pub algorithm: i32,
    pub alg_temp: f32,
    pub cfg_scale: f32,
    pub add_gumbel_noise: bool,
}

impl Default for ParamsDiffusion {
    fn default() -> Self {
        Self {
            steps: 128,
            visual_mode: false,
            eps: 0.0,
            block_length: 0,
            algorithm: 4,
            alg_temp: 0.0,
            cfg_scale: 0.0,
            add_gumbel_noise: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LrOpt {
    pub lr0: f32,
    pub lr_min: f32,
    pub decay_epochs: f32,
    pub scale_epoch: f32,
    pub wd: f32,
    pub epochs: u32,
    pub epoch: u32,
}

impl Default for LrOpt {
    fn default() -> Self {
        Self {
            lr0: 1e-5,
            lr_min: -1.0,
            decay_epochs: -1.0,
            scale_epoch: 0.0,
            wd: 0.0,
            epochs: 2,
            epoch: 0,
        }
    }
}

impl LrOpt {
    const LOG_2: f32 = std::f32::consts::LN_2;

    pub fn init(&mut self) {
        if self.lr_min > 0.0 && self.lr_min < self.lr0 {
            let nhalf = (self.lr0 / self.lr_min).ln() / Self::LOG_2;
            let mut e = self.epochs as f32;
            if self.decay_epochs > 0.0 && self.decay_epochs < e {
                e = self.decay_epochs;
            } else {
                self.decay_epochs = e;
            }
            self.scale_epoch = nhalf / e;
        }
    }

    pub fn get_lr_for(&self, epoch: f32) -> f32 {
        if self.lr_min <= 0.0 {
            self.lr0
        } else if epoch >= self.decay_epochs {
            self.lr_min
        } else {
            self.lr0 * 0.5f32.powf(epoch * self.scale_epoch)
        }
    }

    pub fn get_lr(&self) -> f32 {
        self.get_lr_for(self.epoch as f32)
    }
}

/// `ggml_opt_optimizer_params`, reduced to the fields common.cpp sets.
#[derive(Clone, Copy, Debug, Default)]
pub struct OptimizerParams {
    pub alpha: f32,
    pub wd: f32,
}

pub fn common_opt_lr_pars(lr: &LrOpt) -> OptimizerParams {
    OptimizerParams {
        alpha: lr.get_lr(),
        wd: lr.wd,
    }
}

/// `common_opt_dataset_init`, returning owned data instead of a GGML handle.
#[derive(Clone, Debug, Default)]
pub struct Dataset {
    pub data: Vec<Token>,
    pub labels: Vec<Token>,
}

pub fn common_opt_dataset_init(
    tokens: &[Token],
    n_ctx: usize,
    stride: usize,
) -> Result<Dataset> {
    if stride == 0 || tokens.len() <= n_ctx {
        return Err(Error::new("invalid dataset size or stride"));
    }
    let ndata = (tokens.len() - n_ctx - 1) / stride;
    if ndata == 0 {
        return Err(Error::new("no complete datapoints"));
    }
    let mut dataset = Dataset {
        data: vec![0; ndata * n_ctx],
        labels: vec![0; ndata * n_ctx],
    };
    for i in 0..ndata {
        let begin = i * stride;
        dataset.data[i * n_ctx..(i + 1) * n_ctx]
            .copy_from_slice(&tokens[begin..begin + n_ctx]);
        dataset.labels[i * n_ctx..(i + 1) * n_ctx]
            .copy_from_slice(&tokens[begin + 1..begin + 1 + n_ctx]);
    }
    Ok(dataset)
}

/// `common_params`: every field that does not need a live C++ object.
#[derive(Clone, Debug)]
pub struct Params {
    pub n_predict: i32,
    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_keep: i32,
    pub n_chunks: i32,
    pub n_parallel: i32,
    pub n_sequences: i32,
    pub n_outputs_max: i32,
    pub n_outputs_max_per_seq: i32,
    pub grp_attn_n: i32,
    pub grp_attn_w: i32,
    pub n_print: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: i32,

    pub devices: Vec<usize>,
    pub n_gpu_layers: i32,
    pub main_gpu: i32,
    /// `tensor_split[128]`; kept as a Vec so `Default` works for any length.
    pub tensor_split: Vec<f32>,
    pub fit_params: bool,
    pub fit_params_print: bool,
    pub fit_params_min_ctx: i32,
    pub fit_params_target: Vec<usize>,
    pub split_mode: SplitMode,
    pub load_mode: LoadMode,
    pub lazy_mode: LazyMode,
    pub cpuparams: CpuParams,
    pub cpuparams_batch: CpuParams,
    pub numa: u32,
    pub rope_scaling_type: ContextRopeScalingType,
    pub pooling_type: crate::cparams::PoolingType,
    pub attention_type: AttentionType,
    pub flash_attn_type: FlashAttnType,

    pub sampling: ParamsSampling,
    pub speculative: ParamsSpeculative,
    pub diffusion: ParamsDiffusion,
    pub model: ParamsModel,

    pub model_alias: BTreeSet<String>,
    pub model_tags: BTreeSet<String>,
    pub hf_token: String,
    pub prompt: String,
    pub system_prompt: String,
    pub prompt_file: String,
    pub path_prompt_cache: String,
    pub input_prefix: String,
    pub input_suffix: String,
    pub logits_file: String,
    pub path_prompts_log_dir: String,

    pub logits_output_dir: String,
    pub save_logits: bool,
    pub tensor_filter: Vec<String>,

    pub in_files: Vec<String>,
    pub antiprompt: Vec<String>,
    pub kv_overrides: Vec<KvOverride>,
    pub tensor_buft_overrides: Vec<TensorBuftOverride>,

    pub lora_init_without_apply: bool,
    pub lora_adapters: Vec<AdapterLoraInfo>,
    pub control_vectors: Vec<ControlVectorLoadInfo>,

    pub verbosity: i32,
    pub control_vector_layer_start: i32,
    pub control_vector_layer_end: i32,
    pub offline: bool,

    pub ppl_stride: i32,
    pub ppl_output_type: i32,
    pub hellaswag: bool,
    pub hellaswag_tasks: usize,
    pub winogrande: bool,
    pub winogrande_tasks: usize,
    pub multiple_choice: bool,
    pub multiple_choice_tasks: usize,
    pub kl_divergence: bool,
    pub check: bool,

    pub usage: bool,
    pub completion: bool,
    pub use_color: bool,
    pub special: bool,
    pub interactive: bool,
    pub interactive_first: bool,
    pub prompt_cache_all: bool,
    pub prompt_cache_ro: bool,

    pub escape: bool,
    pub multiline_input: bool,
    pub simple_io: bool,
    pub cont_batching: bool,
    pub no_perf: bool,
    pub show_timings: bool,
    pub ctx_shift: bool,
    pub swa_full: bool,
    pub kv_unified: bool,

    pub input_prefix_bos: bool,
    pub verbose_prompt: bool,
    pub display_prompt: bool,
    pub no_kv_offload: bool,
    pub warmup: bool,
    pub check_tensors: bool,
    pub no_op_offload: bool,
    pub no_extra_bufts: bool,
    pub no_host: bool,
    pub load_mtp: bool,
    pub single_turn: bool,

    pub cache_type_k: GgmlType,
    pub cache_type_v: GgmlType,
    pub moe_cache_size: usize,
    pub moe_cache_cpu_share: f32,
    pub conversation_mode: ConversationMode,

    pub mmproj: ParamsModel,
    pub mmproj_use_gpu: bool,
    pub no_mmproj: bool,
    pub image: Vec<String>,
    pub image_min_tokens: i32,
    pub image_max_tokens: i32,
    pub mtmd_batch_max_tokens: i32,

    pub video_fps: f32,
    pub video_timestamp_interval_ms: i64,
    pub video_ffmpeg_bin_dir: String,

    pub lr: LrOpt,
    pub optimizer: OptimizerType,
    pub val_split: f32,

    pub embedding: bool,
    pub embd_normalize: i32,
    pub embd_out: String,
    pub embd_sep: String,
    pub cls_sep: String,

    pub port: i32,
    pub reuse_port: bool,
    pub timeout_read: i32,
    pub timeout_write: i32,
    pub sse_ping_interval: i32,
    pub n_threads_http: i32,
    pub n_cache_reuse: i32,
    pub cache_prompt: bool,
    pub cache_idle_slots: bool,
    pub n_ctx_checkpoints: i32,
    pub kv_unified_per_slot: i32,
    pub checkpoint_min_step: i32,
    pub cache_ram_mib: i32,
    pub cache_ram_explicit: bool,

    pub server_connect: bool,
    pub server_connect_code: String,
    pub public_path: String,
    pub api_prefix: String,
    pub chat_template: String,
    pub hostnames: Vec<String>,
    pub use_jinja: bool,
    pub cors_origins: String,
    pub cors_methods: String,
    pub cors_headers: String,
    pub cors_credentials: bool,
    pub cors_origins_explicit: bool,

    pub enable_chat_template: bool,
    pub force_pure_content_parser: bool,
    pub reasoning_format: ReasoningFormat,
    pub enable_reasoning: i32,
    pub prefill_assistant: bool,
    pub sleep_idle_seconds: i32,
    pub api_keys: Vec<String>,
    pub ssl_file_key: String,
    pub ssl_file_cert: String,
    pub default_template_kwargs: BTreeMap<String, String>,
    pub preserve_reasoning_specified: bool,

    pub server_base: String,

    pub ui: bool,
    pub ui_agent: bool,
    pub ui_mcp_proxy: bool,
    pub ui_mcp_proxy_allow_private_urls: bool,
    pub ui_config_json: String,
    pub endpoint_slots: bool,
    pub endpoint_props: bool,
    pub endpoint_metrics: bool,
    pub server_tools: Vec<String>,
    pub server_tools_runtime: String,
    pub server_tools_allow_unauth: bool,
    pub mcp_servers_config: String,
    pub mcp_servers_json: String,

    pub models_dir: String,
    pub models_preset: String,
    pub models_max: i32,
    pub models_memory_margin: i32,
    pub models_autoload: bool,
    pub models_preset_hf: String,
    pub log_json: bool,

    pub slot_save_path: String,
    pub media_path: String,
    pub media_allow_private_urls: bool,
    pub slot_prompt_similarity: f32,

    pub is_pp_shared: bool,
    pub is_tg_separate: bool,
    pub n_pp: Vec<i32>,
    pub n_tg: Vec<i32>,
    pub n_pl: Vec<i32>,

    pub context_files: Vec<String>,
    pub chunk_size: i32,
    pub chunk_separator: String,

    pub n_junk: i32,
    pub i_pos: i32,

    pub n_out_freq: i32,
    pub n_save_freq: i32,
    pub i_chunk: i32,
    pub imat_dat: i8,
    pub process_output: bool,
    pub compute_ppl: bool,
    pub show_statistics: bool,
    pub activation_statistics: bool,
    pub parse_special: bool,

    pub n_pca_batch: i32,
    pub n_pca_iterations: i32,
    pub cvector_dimre_method: DimreMethod,
    pub cvector_positive_file: String,
    pub cvector_negative_file: String,

    pub spm_infill: bool,
    pub batched_bench_output_jsonl: bool,

    pub tokenize_ids: bool,
    pub tokenize_stdin: bool,
    pub tokenize_no_bos: bool,
    pub tokenize_show_count: bool,

    pub out_file: String,
    /// `fn(progress) -> bool`, returning false aborts model loading.
    pub load_progress_callback: Option<fn(f32) -> bool>,
    pub no_alloc: bool,

    pub tts_lang: String,
    pub tts_voice: String,
    pub tts_speaker_file: String,
    pub tts_speaker_text: String,
    pub is_gen_docs: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            n_predict: -1,
            n_ctx: 0,
            n_batch: 2048,
            n_ubatch: 512,
            n_keep: 0,
            n_chunks: -1,
            n_parallel: 1,
            n_sequences: 1,
            n_outputs_max: 0,
            n_outputs_max_per_seq: 1,
            grp_attn_n: 1,
            grp_attn_w: 512,
            n_print: -1,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: -1.0,
            yarn_beta_fast: -1.0,
            yarn_beta_slow: -1.0,
            yarn_orig_ctx: 0,
            devices: Vec::new(),
            n_gpu_layers: -1,
            main_gpu: 0,
            tensor_split: vec![0.0; 128],
            fit_params: true,
            fit_params_print: false,
            fit_params_min_ctx: 4096,
            fit_params_target: vec![1024 * 1024 * 1024; LLAMA_MAX_DEVICES],
            split_mode: SplitMode::Layer,
            load_mode: LoadMode::Auto,
            lazy_mode: LazyMode::Auto,
            cpuparams: CpuParams::default(),
            cpuparams_batch: CpuParams::default(),
            numa: 0,
            rope_scaling_type: ContextRopeScalingType::Unspecified,
            pooling_type: crate::cparams::PoolingType::Unspecified,
            attention_type: AttentionType::Unspecified,
            flash_attn_type: FlashAttnType::Auto,
            sampling: ParamsSampling::default(),
            speculative: ParamsSpeculative::default(),
            diffusion: ParamsDiffusion::default(),
            model: ParamsModel::default(),
            model_alias: BTreeSet::new(),
            model_tags: BTreeSet::new(),
            hf_token: String::new(),
            prompt: String::new(),
            system_prompt: String::new(),
            prompt_file: String::new(),
            path_prompt_cache: String::new(),
            input_prefix: String::new(),
            input_suffix: String::new(),
            logits_file: String::new(),
            path_prompts_log_dir: String::new(),
            logits_output_dir: "data".into(),
            save_logits: false,
            tensor_filter: Vec::new(),
            in_files: Vec::new(),
            antiprompt: Vec::new(),
            kv_overrides: Vec::new(),
            tensor_buft_overrides: Vec::new(),
            lora_init_without_apply: false,
            lora_adapters: Vec::new(),
            control_vectors: Vec::new(),
            verbosity: 3,
            control_vector_layer_start: -1,
            control_vector_layer_end: -1,
            offline: false,
            ppl_stride: 0,
            ppl_output_type: 0,
            hellaswag: false,
            hellaswag_tasks: 400,
            winogrande: false,
            winogrande_tasks: 0,
            multiple_choice: false,
            multiple_choice_tasks: 0,
            kl_divergence: false,
            check: false,
            usage: false,
            completion: false,
            use_color: false,
            special: false,
            interactive: false,
            interactive_first: false,
            prompt_cache_all: false,
            prompt_cache_ro: false,
            escape: true,
            multiline_input: false,
            simple_io: false,
            cont_batching: true,
            no_perf: false,
            show_timings: true,
            ctx_shift: false,
            swa_full: false,
            kv_unified: false,
            input_prefix_bos: false,
            verbose_prompt: false,
            display_prompt: true,
            no_kv_offload: false,
            warmup: true,
            check_tensors: false,
            no_op_offload: false,
            no_extra_bufts: false,
            no_host: false,
            load_mtp: false,
            single_turn: false,
            cache_type_k: GgmlType::F16,
            cache_type_v: GgmlType::F16,
            moe_cache_size: 0,
            moe_cache_cpu_share: 0.0,
            conversation_mode: ConversationMode::Auto,
            mmproj: ParamsModel::default(),
            mmproj_use_gpu: true,
            no_mmproj: false,
            image: Vec::new(),
            image_min_tokens: -1,
            image_max_tokens: -1,
            mtmd_batch_max_tokens: 1024,
            video_fps: 4.0,
            video_timestamp_interval_ms: 5000,
            video_ffmpeg_bin_dir: String::new(),
            lr: LrOpt::default(),
            optimizer: OptimizerType::Adamw,
            val_split: 0.05,
            embedding: false,
            embd_normalize: 2,
            embd_out: String::new(),
            embd_sep: "\n".into(),
            cls_sep: "\t".into(),
            port: 9931,
            reuse_port: false,
            timeout_read: 3600,
            timeout_write: 3600,
            sse_ping_interval: 30,
            n_threads_http: -1,
            n_cache_reuse: 0,
            cache_prompt: true,
            cache_idle_slots: true,
            n_ctx_checkpoints: 32,
            kv_unified_per_slot: 0,
            checkpoint_min_step: 8192,
            cache_ram_mib: 8192,
            cache_ram_explicit: false,
            server_connect: false,
            server_connect_code: String::new(),
            public_path: String::new(),
            api_prefix: String::new(),
            chat_template: String::new(),
            hostnames: vec!["127.0.0.1".into()],
            use_jinja: true,
            cors_origins: "*".into(),
            cors_methods: "GET, POST, DELETE, OPTIONS".into(),
            cors_headers: "*".into(),
            cors_credentials: true,
            cors_origins_explicit: false,
            enable_chat_template: true,
            force_pure_content_parser: false,
            reasoning_format: ReasoningFormat::Deepseek,
            enable_reasoning: -1,
            prefill_assistant: true,
            sleep_idle_seconds: -1,
            api_keys: Vec::new(),
            ssl_file_key: String::new(),
            ssl_file_cert: String::new(),
            default_template_kwargs: BTreeMap::new(),
            preserve_reasoning_specified: false,
            server_base: String::new(),
            ui: true,
            ui_agent: false,
            ui_mcp_proxy: false,
            ui_mcp_proxy_allow_private_urls: false,
            ui_config_json: String::new(),
            endpoint_slots: true,
            endpoint_props: false,
            endpoint_metrics: false,
            server_tools: Vec::new(),
            server_tools_runtime: String::new(),
            server_tools_allow_unauth: false,
            mcp_servers_config: String::new(),
            mcp_servers_json: String::new(),
            models_dir: String::new(),
            models_preset: String::new(),
            models_max: 4,
            models_memory_margin: 0,
            models_autoload: true,
            models_preset_hf: String::new(),
            log_json: false,
            slot_save_path: String::new(),
            media_path: String::new(),
            media_allow_private_urls: false,
            slot_prompt_similarity: 0.1,
            is_pp_shared: false,
            is_tg_separate: false,
            n_pp: Vec::new(),
            n_tg: Vec::new(),
            n_pl: Vec::new(),
            context_files: Vec::new(),
            chunk_size: 64,
            chunk_separator: "\n".into(),
            n_junk: 250,
            i_pos: -1,
            n_out_freq: 10,
            n_save_freq: 0,
            i_chunk: 0,
            imat_dat: 0,
            process_output: false,
            compute_ppl: true,
            show_statistics: false,
            activation_statistics: false,
            parse_special: false,
            n_pca_batch: 100,
            n_pca_iterations: 1000,
            cvector_dimre_method: DimreMethod::Pca,
            cvector_positive_file: "tools/cvector-generator/positive.txt".into(),
            cvector_negative_file: "tools/cvector-generator/negative.txt".into(),
            spm_infill: false,
            batched_bench_output_jsonl: false,
            tokenize_ids: false,
            tokenize_stdin: false,
            tokenize_no_bos: false,
            tokenize_show_count: false,
            out_file: String::new(),
            load_progress_callback: None,
            no_alloc: false,
            tts_lang: String::new(),
            tts_voice: String::new(),
            tts_speaker_file: String::new(),
            tts_speaker_text: String::new(),
            is_gen_docs: false,
        }
    }
}

/// Stand-in for `llama_model_params`.
#[derive(Clone, Debug)]
pub struct ModelParams {
    pub devices: Vec<usize>,
    pub n_gpu_layers: i32,
    pub split_mode: SplitMode,
    pub load_mode: LoadMode,
    pub lazy_mode: LazyMode,
    pub main_gpu: i32,
    pub tensor_split: Vec<f32>,
    pub progress_callback: Option<fn(f32) -> bool>,
    pub kv_overrides: Vec<KvOverride>,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
    pub tensor_buft_overrides: Vec<TensorBuftOverride>,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            devices: Vec::new(),
            n_gpu_layers: -1,
            split_mode: SplitMode::Layer,
            load_mode: LoadMode::Auto,
            lazy_mode: LazyMode::Auto,
            main_gpu: 0,
            tensor_split: vec![0.0; LLAMA_MAX_DEVICES],
            progress_callback: None,
            kv_overrides: Vec::new(),
            check_tensors: false,
            use_extra_bufts: true,
            no_host: false,
            no_alloc: false,
            load_mtp: false,
            tensor_buft_overrides: Vec::new(),
        }
    }
}

/// Stand-in for `llama_context_params`.
#[derive(Clone, Debug)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub rope_scaling_type: ContextRopeScalingType,
    pub pooling_type: crate::cparams::PoolingType,
    pub attention_type: AttentionType,
    pub flash_attn_type: FlashAttnType,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: i32,
    pub type_k: GgmlType,
    pub type_v: GgmlType,
    pub moe_cache_size: usize,
    pub moe_cache_cpu_share: f32,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            n_ctx: 512,
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 1,
            n_rs_seq: 0,
            n_outputs_max: 0,
            n_outputs_max_per_seq: 1,
            n_threads: 4,
            n_threads_batch: 4,
            rope_scaling_type: ContextRopeScalingType::Unspecified,
            pooling_type: crate::cparams::PoolingType::Unspecified,
            attention_type: AttentionType::Unspecified,
            flash_attn_type: FlashAttnType::Auto,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: -1.0,
            yarn_beta_fast: -1.0,
            yarn_beta_slow: -1.0,
            yarn_orig_ctx: 0,
            type_k: GgmlType::F16,
            type_v: GgmlType::F16,
            moe_cache_size: 0,
            moe_cache_cpu_share: 0.0,
            embeddings: false,
            offload_kqv: true,
            no_perf: true,
            op_offload: true,
            swa_full: true,
            kv_unified: false,
        }
    }
}

/// `common_model_params_to_llama`.
pub fn common_model_params_to_llama(params: &Params) -> ModelParams {
    let mut mparams = ModelParams::default();
    mparams.devices = params.devices.clone();
    mparams.n_gpu_layers = params.n_gpu_layers;
    mparams.main_gpu = params.main_gpu;
    mparams.split_mode = params.split_mode;
    mparams.load_mode = params.load_mode;
    mparams.lazy_mode = params.lazy_mode;
    mparams.tensor_split = params.tensor_split.clone();
    mparams.check_tensors = params.check_tensors;
    mparams.use_extra_bufts = !params.no_extra_bufts;
    mparams.no_host = params.no_host;
    mparams.kv_overrides = params.kv_overrides.clone();
    mparams.tensor_buft_overrides = params.tensor_buft_overrides.clone();
    mparams.progress_callback = params.load_progress_callback;
    mparams.no_alloc = params.no_alloc;
    mparams.load_mtp = params.load_mtp
        || params.speculative.types.contains(&SpeculativeType::DraftMtp);
    mparams
}

/// `common_context_params_to_llama`.
pub fn common_context_params_to_llama(params: &Params) -> ContextParams {
    let mut cparams = ContextParams::default();
    cparams.n_ctx = params.n_ctx.max(0) as u32;
    cparams.n_seq_max = params.n_parallel.max(0) as u32;
    cparams.n_rs_seq = params.speculative.need_n_rs_seq();
    cparams.n_outputs_max = params.n_outputs_max.max(0) as u32;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq.max(0) as u32;
    cparams.n_batch = params.n_batch.max(0) as u32;
    cparams.n_ubatch = params.n_ubatch.max(0) as u32;
    cparams.n_threads = params.cpuparams.n_threads;
    cparams.n_threads_batch = if params.cpuparams_batch.n_threads == -1 {
        params.cpuparams.n_threads
    } else {
        params.cpuparams_batch.n_threads
    };
    cparams.embeddings = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base = params.rope_freq_base;
    cparams.rope_freq_scale = params.rope_freq_scale;
    cparams.yarn_ext_factor = params.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor;
    cparams.yarn_beta_fast = params.yarn_beta_fast;
    cparams.yarn_beta_slow = params.yarn_beta_slow;
    cparams.yarn_orig_ctx = params.yarn_orig_ctx;
    cparams.pooling_type = params.pooling_type;
    cparams.attention_type = params.attention_type;
    cparams.flash_attn_type = params.flash_attn_type;
    cparams.offload_kqv = !params.no_kv_offload;
    cparams.no_perf = params.no_perf;
    cparams.op_offload = !params.no_op_offload;
    cparams.swa_full = params.swa_full;
    cparams.kv_unified = params.kv_unified;
    cparams.type_k = params.cache_type_k;
    cparams.type_v = params.cache_type_v;
    cparams.moe_cache_size = params.moe_cache_size;
    cparams.moe_cache_cpu_share = params.moe_cache_cpu_share;
    cparams
}

/// `common_params_get_system_info` without `llama_print_system_info()`.
pub fn common_params_get_system_info(params: &Params) -> String {
    let batch = if params.cpuparams_batch.n_threads != -1 {
        format!(" (n_threads_batch = {})", params.cpuparams_batch.n_threads)
    } else {
        String::new()
    };
    let hardware = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "?".to_owned());
    format!(
        "system_info: n_threads = {}{} / {} | rust port",
        params.cpuparams.n_threads, batch, hardware
    )
}

/// `common_params_print_info`, returned as a string instead of logged.
pub fn common_params_print_info(params: &Params, _print_devices: bool) -> String {
    common_params_get_system_info(params)
}

// ---------------------------------------------------------------------------
// model metadata / split helpers
// ---------------------------------------------------------------------------

/// `common_gguf_info`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GgufInfo {
    pub decision_type: DecisionType,
    pub n_ctx_train: u32,
    pub has_chat_tmpl: bool,
}

/// `common_get_gguf_info`: reads metadata without allocating tensors.
pub fn common_get_gguf_info(fname: &str) -> GgufInfo {
    let mut info = GgufInfo::default();
    let mut init = GgufInitParams {
        no_alloc: true,
        ctx: None,
    };
    let Ok(ctx) = gguf_init_from_file(fname, &mut init) else {
        return info;
    };

    let arch_id = gguf_find_key(&ctx, "general.architecture");
    if arch_id < 0 || gguf_get_kv_type(&ctx, arch_id) != GgufType::String {
        return info;
    }
    let arch = gguf_get_val_str(&ctx, arch_id);
    if arch.is_empty() {
        return info;
    }

    let type_id = gguf_find_key(&ctx, &format!("{arch}.decision.type"));
    if type_id < 0 {
        info.decision_type = DecisionType::None;
    } else if gguf_get_kv_type(&ctx, type_id) == GgufType::String {
        info.decision_type = DecisionType::from_name(gguf_get_val_str(&ctx, type_id));
    }

    let ctx_id = gguf_find_key(&ctx, &format!("{arch}.context_length"));
    if ctx_id >= 0 && gguf_get_kv_type(&ctx, ctx_id) == GgufType::Uint32 {
        info.n_ctx_train = gguf_get_val_u32(&ctx, ctx_id);
    }

    info.has_chat_tmpl = gguf_find_key(&ctx, "tokenizer.chat_template") >= 0;
    info
}

/// `llm_ffn_block_regex`.
pub fn llm_ffn_block_regex(idx: i32, ffn_regex: &str) -> String {
    format!("blk\\.{idx}{ffn_regex}")
}

/// `llm_ffn_exps_cpu_override`.
pub fn llm_ffn_exps_cpu_override() -> TensorBuftOverride {
    TensorBuftOverride {
        pattern: LLM_FFN_EXPS_REGEX.to_owned(),
        buffer_type: "CPU".to_owned(),
    }
}

/// `llm_add_n_cpu_ffn_overrides`.
pub fn llm_add_n_cpu_ffn_overrides(
    n: i32,
    ffn_regex: &str,
    overrides: &mut Vec<TensorBuftOverride>,
) {
    for i in 0..n {
        overrides.push(TensorBuftOverride {
            pattern: llm_ffn_block_regex(i, ffn_regex),
            buffer_type: "CPU".to_owned(),
        });
    }
}

// ---------------------------------------------------------------------------
// token and embedding helpers
// ---------------------------------------------------------------------------

/// `common_tokenize` with a vocab instead of a context.
pub fn common_tokenize(
    vocab: &LlamaVocab,
    text: &str,
    add_special: bool,
    parse_special: bool,
) -> Result<Vec<Token>> {
    let tokens = vocab.tokenize(text, add_special, parse_special);
    if tokens.len() >= i32::MAX as usize {
        return Err(Error::new(
            "Tokenization failed: input text too large, tokenization result exceeds int32_t limit",
        ));
    }
    Ok(tokens)
}

/// `common_token_to_piece` with a vocab instead of a context.
pub fn common_token_to_piece(vocab: &LlamaVocab, token: Token, special: bool) -> String {
    let mut piece = vec![0u8; 16];
    let mut n = vocab.token_to_piece(token, &mut piece, 0, special);
    if n < 0 {
        piece.resize((-n) as usize, 0);
        n = vocab.token_to_piece(token, &mut piece, 0, special);
    }
    if n < 0 || n as usize > piece.len() {
        return String::new();
    }
    String::from_utf8_lossy(&piece[..n as usize]).into_owned()
}

/// `common_detokenize` with a vocab instead of a context.
pub fn common_detokenize(vocab: &LlamaVocab, tokens: &[Token], special: bool) -> String {
    vocab.detokenize_string(tokens, special)
}

/// `string_from(const llama_context *, tokens)`.
pub fn string_from_tokens(vocab: &LlamaVocab, tokens: &[Token]) -> String {
    let mut out = String::from("[ ");
    for (i, token) in tokens.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(&common_token_to_piece(vocab, *token, true));
        out.push_str("':");
        out.push_str(&token.to_string());
    }
    out.push_str(" ]");
    out
}

/// `common_embd_normalize`.
pub fn common_embd_normalize(inp: &[f32], out: &mut [f32], embd_norm: i32) {
    let mut sum = 0.0f64;
    match embd_norm {
        -1 => sum = 1.0,
        0 => {
            for &v in inp {
                sum = sum.max((v as f64).abs());
            }
            sum /= 32760.0;
        }
        2 => {
            for &v in inp {
                sum += (v as f64) * (v as f64);
            }
            sum = sum.sqrt();
        }
        _ => {
            for &v in inp {
                sum += (v as f64).abs().powf(embd_norm as f64);
            }
            sum = sum.powf(1.0 / embd_norm as f64);
        }
    }
    let norm = if sum > 0.0 { 1.0 / sum } else { 0.0 } as f32;
    for (dst, src) in out.iter_mut().zip(inp) {
        *dst = *src * norm;
    }
}

/// `common_embd_similarity_cos`.
pub fn common_embd_similarity_cos(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        sum += (x as f64) * (y as f64);
        sum1 += (x as f64) * (x as f64);
        sum2 += (y as f64) * (y as f64);
    }
    if sum1 == 0.0 || sum2 == 0.0 {
        return if sum1 == 0.0 && sum2 == 0.0 { 1.0 } else { 0.0 };
    }
    (sum / (sum1.sqrt() * sum2.sqrt())) as f32
}

// ---------------------------------------------------------------------------
// control vector utils
// ---------------------------------------------------------------------------

fn control_vector_load_one(info: &ControlVectorLoadInfo, n_layer: i32) -> ControlVectorData {
    let mut result = ControlVectorData {
        n_embd: -1,
        data: Vec::new(),
    };
    let mut init = GgufInitParams {
        no_alloc: false,
        ctx: Some(GgmlContext::default()),
    };
    let Ok(ctx) = gguf_init_from_file(&info.fname, &mut init) else {
        eprintln!("failed to load control vector file from {}", info.fname);
        return result;
    };

    let n_tensors = gguf_get_n_tensors(&ctx);
    if n_tensors == 0 {
        eprintln!("no direction tensors found in {}", info.fname);
    }

    for i in 0..n_tensors {
        let name = gguf_get_tensor_name(&ctx, i).to_owned();
        let dot = name.find('.');
        let layer_idx = match dot {
            Some(pos) if &name[..pos] == "direction" => name[pos + 1..].parse::<i32>().unwrap_or(-1),
            _ => -1,
        };

        if layer_idx < 0 {
            eprintln!("invalid/unparsable direction tensor layer index in {}", info.fname);
            result.n_embd = -1;
            break;
        } else if layer_idx == 0 {
            eprintln!("invalid (zero) direction tensor layer index in {}", info.fname);
            result.n_embd = -1;
            break;
        } else if layer_idx > n_layer {
            eprintln!(
                "direction tensor layer index {layer_idx} exceeds the model layer count {n_layer} in {}",
                info.fname
            );
            result.n_embd = -1;
            break;
        }

        let tensor = &ctx.info[i as usize].t;
        if gguf_get_tensor_type(&ctx, i) != GgmlType::F32 || tensor.type_ != GgmlType::F32 {
            eprintln!("invalid (non-F32) direction tensor type in {}", info.fname);
            result.n_embd = -1;
            break;
        }
        if ggml_n_dims(tensor) != 1 {
            eprintln!("invalid (non-1D) direction tensor shape in {}", info.fname);
            result.n_embd = -1;
            break;
        }

        let n_embd = tensor.ne[..ggml_n_dims(tensor)]
            .iter()
            .fold(1usize, |acc, &n| acc * n.max(1) as usize);
        if result.n_embd == -1 {
            result.n_embd = n_embd as i32;
        } else if result.n_embd != n_embd as i32 {
            eprintln!("direction tensor in {} does not match previous dimensions", info.fname);
            result.n_embd = -1;
            break;
        }

        let size = result.n_embd as usize * layer_idx as usize;
        if result.data.len() < size {
            result.data.resize(size, 0.0);
        }
        for j in 0..result.n_embd as usize {
            let start = j * 4;
            if start + 4 > tensor.data.len() {
                result.n_embd = -1;
                break;
            }
            let value = f32::from_ne_bytes([
                tensor.data[start],
                tensor.data[start + 1],
                tensor.data[start + 2],
                tensor.data[start + 3],
            ]);
            result.data[result.n_embd as usize * (layer_idx as usize - 1) + j] +=
                value * info.strength;
        }
        if result.n_embd == -1 {
            break;
        }
    }

    if result.n_embd == -1 {
        eprintln!("skipping {} due to invalid direction tensors", info.fname);
        result.data.clear();
    }
    result
}

/// `common_control_vector_load`.
pub fn common_control_vector_load(
    infos: &[ControlVectorLoadInfo],
    n_layer: i32,
) -> ControlVectorData {
    let mut result = ControlVectorData {
        n_embd: -1,
        data: Vec::new(),
    };
    for info in infos {
        let cur = control_vector_load_one(info, n_layer);
        if cur.n_embd == -1 {
            result.n_embd = -1;
            break;
        }
        if result.n_embd != -1 && result.n_embd != cur.n_embd {
            eprintln!(
                "control vectors in {} does not match previous dimensions",
                info.fname
            );
            result.n_embd = -1;
            break;
        }
        if result.n_embd == -1 {
            result = cur;
        } else {
            if result.data.len() < cur.data.len() {
                result.data.resize(cur.data.len(), 0.0);
            }
            for (dst, src) in result.data.iter_mut().zip(cur.data.iter()) {
                *dst += *src;
            }
        }
    }
    if result.n_embd == -1 {
        eprintln!("no valid control vector files passed");
        result.data.clear();
    }
    result
}

// ---------------------------------------------------------------------------
// string utils
// ---------------------------------------------------------------------------

/// `string_strip`.
pub fn string_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_ascii_whitespace())
}

/// `string_lcs`, over bytes like the C++ `string_view` version.
pub fn string_lcs(a: &str, b: &str) -> String {
    if a.is_empty() || b.is_empty() {
        return String::new();
    }
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let mut dp = vec![vec![0usize; bb.len() + 1]; ab.len() + 1];
    let mut best_len = 0usize;
    let mut best_end = 0usize;
    for i in 1..=ab.len() {
        for j in 1..=bb.len() {
            if ab[i - 1] == bb[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if dp[i][j] > best_len {
                    best_len = dp[i][j];
                    best_end = i;
                }
            }
        }
    }
    String::from_utf8_lossy(&ab[best_end - best_len..best_end]).into_owned()
}

/// `string_get_sortable_timestamp`. UTC: Rust std has no localtime.
pub fn string_get_sortable_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let days = secs.div_euclid(86400);
    let sod = secs.rem_euclid(86400);
    let (year, month, day) = civil_from_days(days);
    let hour = sod / 3600;
    let minute = (sod % 3600) / 60;
    let second = sod % 60;
    format!(
        "{year:04}_{month:02}_{day:02}-{hour:02}_{minute:02}_{second:02}.{:09}",
        now.subsec_nanos()
    )
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `string_replace_all`.
pub fn string_replace_all(s: &mut String, search: &str, replace: &str) {
    if search.is_empty() {
        return;
    }
    *s = s.replace(search, replace);
}

/// `regex_escape`.
pub fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, '.' | '^' | '$' | '|' | '(' | ')' | '*' | '+' | '?' | '[' | ']' | '{' | '}' | '\\') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// `string_join`.
pub fn string_join(values: &[String], separator: &str) -> String {
    values.join(separator)
}

/// `string_split` with a string delimiter.
pub fn string_split(s: &str, delimiter: &str) -> Vec<String> {
    if delimiter.is_empty() {
        return vec![s.to_owned()];
    }
    s.split(delimiter).map(str::to_owned).collect()
}

/// Specialized `string_split<T>` for `String`.
pub fn string_split_char(s: &str, delim: char) -> Vec<String> {
    s.split(delim).map(str::to_owned).collect()
}

/// Generic `string_split<T>` from the header, returning an error on bad items.
pub fn string_split_to<T: std::str::FromStr>(s: &str, delim: char) -> Result<Vec<T>> {
    let mut values = Vec::new();
    for token in s.split(delim) {
        match token.parse::<T>() {
            Ok(v) => values.push(v),
            Err(_) => {
                return Err(Error::new(format!("invalid value: \"{token}\"")));
            }
        }
    }
    Ok(values)
}

/// `string_repeat`.
pub fn string_repeat(s: &str, n: usize) -> String {
    s.repeat(n)
}

/// `string_starts_with(str, prefix)`.
pub fn string_starts_with(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

/// `string_ends_with`.
pub fn string_ends_with(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}

/// `string_remove_suffix`.
pub fn string_remove_suffix(s: &mut String, suffix: &str) -> bool {
    if s.ends_with(suffix) {
        s.truncate(s.len() - suffix.len());
        true
    } else {
        false
    }
}

/// `string_find_partial_stop`; `None` is C++ `std::string::npos`.
pub fn string_find_partial_stop(s: &str, stop: &str) -> Option<usize> {
    if s.is_empty() || stop.is_empty() {
        return None;
    }
    let sb = s.as_bytes();
    let st = stop.as_bytes();
    let max_len = sb.len().min(st.len());
    let last = sb[sb.len() - 1];
    for len in (1..=max_len).rev() {
        if st[len - 1] == last && sb.ends_with(&st[..len]) {
            return Some(sb.len() - len);
        }
    }
    None
}

fn parse_i64_prefix(s: &str) -> Option<i64> {
    let bytes = s.as_bytes();
    let mut end = 0;
    if bytes.first() == Some(&b'-') || bytes.first() == Some(&b'+') {
        end = 1;
    }
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == 0 || (end == 1 && !bytes[0].is_ascii_digit()) {
        return None;
    }
    s[..end].parse().ok()
}

fn parse_f64_prefix(s: &str) -> Option<f64> {
    for end in (1..=s.len()).rev() {
        if let Ok(v) = s[..end].parse::<f64>() {
            return Some(v);
        }
    }
    None
}

/// `string_parse_kv_override`.
pub fn string_parse_kv_override(data: &str, overrides: &mut Vec<KvOverride>) -> bool {
    let Some(sep) = data.find('=') else {
        eprintln!("string_parse_kv_override: malformed KV override '{data}'");
        return false;
    };
    if sep >= 128 {
        eprintln!("string_parse_kv_override: malformed KV override '{data}'");
        return false;
    }

    let mut kvo = KvOverride {
        key: data[..sep].to_owned(),
        ..KvOverride::default()
    };
    let value = &data[sep + 1..];
    if let Some(rest) = value.strip_prefix("int:") {
        kvo.tag = KvOverrideType::Int;
        kvo.val_i64 = parse_i64_prefix(rest).unwrap_or(0);
    } else if let Some(rest) = value.strip_prefix("float:") {
        kvo.tag = KvOverrideType::Float;
        kvo.val_f64 = parse_f64_prefix(rest).unwrap_or(0.0);
    } else if let Some(rest) = value.strip_prefix("bool:") {
        kvo.tag = KvOverrideType::Bool;
        match rest {
            "true" => kvo.val_bool = true,
            "false" => kvo.val_bool = false,
            _ => {
                eprintln!("string_parse_kv_override: invalid boolean value for KV override '{data}'");
                return false;
            }
        }
    } else if let Some(rest) = value.strip_prefix("str:") {
        if rest.len() > 127 {
            eprintln!(
                "string_parse_kv_override: malformed KV override '{data}', value cannot exceed 127 chars"
            );
            return false;
        }
        kvo.tag = KvOverrideType::Str;
        kvo.val_str = rest.to_owned();
    } else {
        eprintln!("string_parse_kv_override: invalid type for KV override '{data}'");
        return false;
    }
    overrides.push(kvo);
    true
}

/// `string_process_escapes`, in place.
pub fn string_process_escapes(input: &mut String) {
    let bytes = input.as_bytes().to_vec();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 1;
            match bytes[i] {
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'\'' => out.push(b'\''),
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                b'x' => {
                    if i + 2 < bytes.len()
                        && bytes[i + 1].is_ascii_hexdigit()
                        && bytes[i + 2].is_ascii_hexdigit()
                    {
                        let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or_default();
                        if let Ok(v) = u8::from_str_radix(hex, 16) {
                            i += 2;
                            out.push(v);
                            i += 1;
                            continue;
                        }
                    }
                    out.push(b'\\');
                    out.push(bytes[i]);
                }
                _ => {
                    out.push(b'\\');
                    out.push(bytes[i]);
                }
            }
            i += 1;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    *input = String::from_utf8_lossy(&out).into_owned();
}

/// `string_from(bool)`.
pub fn string_from_bool(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

/// `string_from(const std::vector<int> &)`.
pub fn string_from_ints(values: &[i32]) -> String {
    let body: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    format!("[ {} ]", body.join(", "))
}

/// `string_from(bool value)` used by the CLI printers.
pub fn string_from(value: bool) -> &'static str {
    string_from_bool(value)
}

fn glob_class_match(c: u8, pattern: &[u8], start: usize, end: usize) -> bool {
    let mut pos = start;
    let mut negated = false;
    if pos < end && pattern[pos] == b'!' {
        negated = true;
        pos += 1;
    }
    if pos < end && (pattern[pos] == b']' || pattern[pos] == b'-') {
        if pattern[pos] == c {
            return !negated;
        }
        pos += 1;
    }

    let mut matched = false;
    while pos < end {
        if pos + 2 < end && pattern[pos + 1] == b'-' && pattern[pos + 2] != b']' {
            if c >= pattern[pos] && c <= pattern[pos + 2] {
                matched = true;
                break;
            }
            pos += 3;
        } else {
            if pattern[pos] == c {
                matched = true;
                break;
            }
            pos += 1;
        }
    }
    if negated { !matched } else { matched }
}

fn glob_match_at(pattern: &[u8], mut p: usize, text: &[u8], mut s: usize, depth: usize) -> bool {
    if depth >= COMMON_RECURSIVE_LIMIT {
        return false;
    }
    while p < pattern.len() {
        if pattern[p] == b'*' && p + 1 < pattern.len() && pattern[p + 1] == b'*' {
            let rest = p + 2;
            loop {
                if glob_match_at(pattern, rest, text, s, depth + 1) {
                    return true;
                }
                if s == text.len() {
                    return false;
                }
                s += 1;
            }
        }
        if pattern[p] == b'*' {
            let rest = p + 1;
            while s < text.len() && text[s] != b'/' {
                if glob_match_at(pattern, rest, text, s, depth + 1) {
                    return true;
                }
                s += 1;
            }
            return glob_match_at(pattern, rest, text, s, depth + 1);
        }
        if pattern[p] == b'?' && s < text.len() && text[s] != b'/' {
            p += 1;
            s += 1;
            continue;
        }
        if pattern[p] == b'[' {
            let mut class_end = p + 1;
            if class_end < pattern.len() && (pattern[class_end] == b']' || pattern[class_end] == b'-') {
                class_end += 1;
            }
            while class_end < pattern.len() && pattern[class_end] != b']' {
                class_end += 1;
            }
            if class_end < pattern.len() {
                if s == text.len() || !glob_class_match(text[s], pattern, p + 1, class_end) {
                    return false;
                }
                p = class_end + 1;
                s += 1;
                continue;
            }
            if s == text.len() || text[s] != b'[' {
                return false;
            }
            p += 1;
            s += 1;
            continue;
        }
        if s >= text.len() || pattern[p] != text[s] {
            return false;
        }
        p += 1;
        s += 1;
    }
    s == text.len()
}

/// `glob_match`: `*` skips `/`, `**` spans `/`, `[]` is a char class.
pub fn glob_match(pattern: &str, s: &str) -> bool {
    glob_match_at(pattern.as_bytes(), 0, s.as_bytes(), 0, 0)
}

// ---------------------------------------------------------------------------
// environment, filesystem, TTY
// ---------------------------------------------------------------------------

/// `common_get_env`: unset reads as empty.
pub fn common_get_env(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

/// `common_set_env`: empty value unsets.
pub fn common_set_env(name: &str, value: &str) {
    if value.is_empty() {
        std::env::remove_var(name);
    } else {
        std::env::set_var(name, value);
    }
}

fn get_home_directory() -> Result<PathBuf> {
    let home = common_get_env("HOME");
    if !home.is_empty() {
        return Ok(PathBuf::from(home));
    }
    Err(Error::new("Failed to find $HOME directory"))
}

/// `fs_get_cache_directory`.
pub fn fs_get_cache_directory() -> Result<PathBuf> {
    let env = common_get_env("LLAMA_CACHE");
    if !env.is_empty() {
        return Ok(PathBuf::from(env));
    }
    #[cfg(target_os = "windows")]
    {
        let local = common_get_env("LOCALAPPDATA");
        if local.is_empty() {
            return Err(Error::new("Failed to find %LOCALAPPDATA% directory"));
        }
        return Ok(PathBuf::from(local).join("llama.cpp"));
    }
    #[cfg(target_os = "macos")]
    {
        return Ok(get_home_directory()?.join("Library/Caches/llama.cpp"));
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let xdg = common_get_env("XDG_CACHE_HOME");
        let base = if xdg.is_empty() {
            get_home_directory()?.join(".cache")
        } else {
            PathBuf::from(xdg)
        };
        Ok(base.join("llama.cpp"))
    }
}

/// `fs_get_config_directory`.
pub fn fs_get_config_directory() -> Result<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        let appdata = common_get_env("APPDATA");
        if appdata.is_empty() {
            return Err(Error::new("Failed to find %APPDATA% directory"));
        }
        return Ok(PathBuf::from(appdata).join("llama.cpp"));
    }
    #[cfg(not(target_os = "windows"))]
    {
        let xdg = common_get_env("XDG_CONFIG_HOME");
        let base = if xdg.is_empty() {
            get_home_directory()?.join(".config")
        } else {
            PathBuf::from(xdg)
        };
        Ok(base.join("llama.cpp"))
    }
}

/// `fs_get_cache_file`; the filename must not contain a path separator.
pub fn fs_get_cache_file(filename: &str) -> Result<PathBuf> {
    if filename.contains(std::path::MAIN_SEPARATOR) {
        return Err(Error::new("cache filename contains a path separator"));
    }
    let dir = fs_get_cache_directory()?;
    fs::create_dir_all(&dir)
        .map_err(|e| Error::new(format!("failed to create cache directory: {dir:?}: {e}")))?;
    Ok(dir.join(filename))
}

/// `fs_is_directory`.
pub fn fs_is_directory(path: &str) -> bool {
    Path::new(path).is_dir()
}

/// `fs_validate_filename`.
pub fn fs_validate_filename(filename: &str, allow_subdirs: bool) -> bool {
    if filename.is_empty() || filename.len() > 255 {
        return false;
    }
    let bytes = filename.as_bytes();
    let mut offset = 0usize;
    while offset < bytes.len() {
        let start = offset;
        let Ok(c) = unicode_cpt_from_utf8(bytes, &mut offset) else {
            return false;
        };
        let consumed = offset - start;
        if (consumed == 2 && c < 0x80)
            || (consumed == 3 && c < 0x800)
            || (consumed == 4 && c < 0x10000)
        {
            return false;
        }
        if c <= 0x1F
            || c == 0x7F
            || (0x80..=0x9F).contains(&c)
            || c == 0xFF0E
            || c == 0x2215
            || c == 0x2216
            || (0xD800..=0xDFFF).contains(&c)
            || c > 0x10FFFF
            || c == 0xFFFD
            || c == 0xFEFF
            || c == b':' as u32
            || c == b'*' as u32
            || c == b'?' as u32
            || c == b'"' as u32
            || c == b'<' as u32
            || c == b'>' as u32
            || c == b'|' as u32
        {
            return false;
        }
        if !allow_subdirs && (c == b'/' as u32 || c == b'\\' as u32) {
            return false;
        }
    }

    if filename.starts_with(' ') || filename.ends_with(' ') || filename.ends_with('.') {
        return false;
    }
    if filename.contains("..") || filename == "." {
        return false;
    }
    true
}

static WRITE_ID: AtomicU64 = AtomicU64::new(0);

/// `fs_write_atomic`: write a unique temp file, then rename it into place.
pub fn fs_write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let id = WRITE_ID.fetch_add(1, Ordering::Relaxed) as u128
        | ((SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            & 0xffff_ffff)
            << 64);
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(format!(".{id:x}.tmp"));
    let tmp = PathBuf::from(tmp);

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::new(format!("failed to create directory: {e}")))?;
        }
    }

    let write_result = fs::write(&tmp, data);
    let rename_result = match write_result {
        Ok(()) => fs::rename(&tmp, path),
        Err(e) => Err(e),
    };
    if let Err(e) = rename_result {
        let _ = fs::remove_file(&tmp);
        return Err(Error::new(format!(
            "failed to write file: {}: {e}",
            path.display()
        )));
    }
    Ok(())
}

/// `common_is_tty(stdout)` / `common_is_tty(stderr)` via `IsTerminal`.
pub fn stdout_is_tty() -> bool {
    std::io::stdout().is_terminal()
}

pub fn stderr_is_tty() -> bool {
    std::io::stderr().is_terminal()
}

/// `tty_can_use_colors`.
pub fn tty_can_use_colors() -> bool {
    if !common_get_env("NO_COLOR").is_empty() {
        return false;
    }
    if common_get_env("TERM") == "dumb" {
        return false;
    }
    stdout_is_tty() || stderr_is_tty()
}

/// `tty_enable_ansi`. Windows console mode needs OS calls; std cannot set it.
pub fn tty_enable_ansi() -> bool {
    true
}

/// `common_get_model_endpoint`.
pub fn common_get_model_endpoint() -> String {
    let mut endpoint = common_get_env("MODEL_ENDPOINT");
    if endpoint.is_empty() {
        endpoint = common_get_env("HF_ENDPOINT");
    }
    if endpoint.is_empty() {
        return "https://huggingface.co/".to_owned();
    }
    if !endpoint.ends_with('/') {
        endpoint.push('/');
    }
    endpoint
}

/// `common_get_model_or_exit`, as a `Result` instead of `exit`.
pub fn common_get_model_or_exit(arg: Option<&str>) -> Result<String> {
    if let Some(path) = arg {
        return Ok(path.to_owned());
    }
    let path = common_get_env("LLAMACPP_TEST_MODELFILE");
    if path.is_empty() {
        return Err(Error::new(
            "No model file provided. Set LLAMACPP_TEST_MODELFILE=<gguf_model_path>.",
        ));
    }
    Ok(path)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_helpers() {
        assert_eq!(string_strip(" \t hi \n"), "hi");
        assert_eq!(string_lcs("abcde", "abfde"), "ab");
        assert_eq!(string_join(&["a".into(), "b".into()], "-"), "a-b");
        assert_eq!(string_split("a::b::", "::"), vec!["a", "b", ""]);
        assert_eq!(string_split_char("a,b", ','), vec!["a", "b"]);
        assert_eq!(string_split_to::<i32>("1,2,3", ','), Ok(vec![1, 2, 3]));
        assert!(string_split_to::<i32>("1,x", ',').is_err());
        assert_eq!(string_repeat("ab", 3), "ababab");
        let mut s = "a.b.c".to_owned();
        string_replace_all(&mut s, ".", "-");
        assert_eq!(s, "a-b-c");
        assert_eq!(regex_escape("a.b[c]"), "a\\.b\\[c\\]");
        assert!(string_starts_with("foobar", "foo"));
        assert!(string_ends_with("foobar", "bar"));
        let mut s = "foobar".to_owned();
        assert!(string_remove_suffix(&mut s, "bar"));
        assert_eq!(s, "foo");
        assert_eq!(string_find_partial_stop("abcstop", "stop"), Some(3));
        assert_eq!(string_find_partial_stop("abc", "stop"), None);
    }

    #[test]
    fn escape_sequences() {
        let mut s = String::from("a\\nb\\x41\\q");
        string_process_escapes(&mut s);
        assert_eq!(s, "a\nbA\\q");
    }

    #[test]
    fn glob_patterns() {
        assert!(glob_match("src/**/*.rs", "src/common/mod.rs"));
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "a/c"));
        assert!(glob_match("[!a]bc", "xbc"));
        assert!(!glob_match("[!a]bc", "abc"));
        assert!(glob_match("[a-c]bc", "bbc"));
        assert!(glob_match("a[bc", "a[bc"));
    }

    #[test]
    fn cpu_masks_and_ranges() {
        let mut mask = [false; GGML_MAX_N_THREADS];
        assert!(parse_cpu_range("-3", &mut mask));
        assert!(mask[0] && mask[3] && !mask[4]);
        let mut mask2 = [false; GGML_MAX_N_THREADS];
        assert!(parse_cpu_mask("0x8000000000000001", &mut mask2));
        assert!(mask2[0] && mask2[63]);
        assert!(!parse_cpu_range("3", &mut mask2));
        assert!(parse_cpu_range("3-", &mut mask2));
        assert!(mask2[3] && mask2[GGML_MAX_N_THREADS - 1]);

        let mut cp = CpuParams::default();
        cp.n_threads = -1;
        postprocess_cpu_params(&mut cp, None);
        assert!(cp.n_threads > 0);
        let tpp = ggml_threadpool_params_from_cpu_params(&cp);
        assert_eq!(tpp.n_threads, cp.n_threads as usize);
    }

    #[test]
    fn kv_overrides() {
        let mut overrides = Vec::new();
        assert!(string_parse_kv_override("foo=int:12", &mut overrides));
        assert_eq!(overrides[0].val_i64, 12);
        assert!(string_parse_kv_override("foo=float:1.5rest", &mut overrides));
        assert_eq!(overrides[1].val_f64, 1.5);
        assert!(string_parse_kv_override("foo=bool:false", &mut overrides));
        assert!(!overrides[2].val_bool);
        assert!(string_parse_kv_override("foo=str:hello", &mut overrides));
        assert_eq!(overrides[3].val_str, "hello");
        assert!(!string_parse_kv_override("bad", &mut overrides));
        assert!(!string_parse_kv_override("foo=bool:yes", &mut overrides));
    }

    #[test]
    fn params_defaults() {
        let p = Params::default();
        assert_eq!(p.n_batch, 2048);
        assert_eq!(p.tensor_split.len(), 128);
        assert_eq!(p.fit_params_target.len(), LLAMA_MAX_DEVICES);
        assert_eq!(p.sampling.samplers.len(), 9);
        assert_eq!(p.sampling.dry_sequence_breakers.len(), 4);
        assert_eq!(p.speculative.need_n_rs_seq(), 0);
        let q = common_context_params_to_llama(&p);
        assert_eq!(q.n_batch, 2048);
        let m = common_model_params_to_llama(&p);
        assert!(m.use_extra_bufts);
        assert!(!m.load_mtp);
        assert!(common_params_get_system_info(&p).contains("n_threads = -1"));
    }

    #[test]
    fn speculative_and_model_helpers() {
        let mut spec = ParamsSpeculative::default();
        assert!(!spec.has_dft());
        assert!(!spec.has_synth());
        spec.types = vec![SpeculativeType::DraftMtp];
        assert_eq!(spec.need_n_rs_seq(), 3);
        spec.draft.mparams.path = "draft.gguf".into();
        assert!(spec.has_dft());
        spec.synth_rates.push(0.5);
        assert!(spec.has_synth());

        let mut model = ParamsModel::default();
        model.path = "a.gguf".into();
        model.shards = vec!["a.gguf".into(), "b.gguf".into()];
        assert_eq!(model.get_paths(), vec!["a.gguf", "b.gguf"]);
        assert_eq!(model.get_name(), "a.gguf");
        model.hf_repo = "org/repo".into();
        assert_eq!(model.get_name(), "org/repo");
    }

    #[test]
    fn sampler_names_and_metadata() {
        let names = vec![
            "top-k".to_owned(),
            "temp".to_owned(),
            "nucleus".to_owned(),
        ];
        assert_eq!(
            sampler_types_from_names(&names),
            vec![SamplerType::TopK, SamplerType::Temperature, SamplerType::TopP]
        );
        let mut meta = BTreeMap::new();
        meta.insert(META_SAMPLING_TOP_K.to_owned(), "11".to_owned());
        meta.insert(META_SAMPLING_TEMP.to_owned(), "0.4".to_owned());
        meta.insert(
            META_SAMPLING_SEQUENCE.to_owned(),
            "top_k;temp".to_owned(),
        );
        let mut sp = ParamsSampling::default();
        sp.user_sampling_config = SAMPLING_CONFIG_TEMP;
        common_init_sampler_from_meta(&meta, &mut sp);
        assert_eq!(sp.top_k, 11);
        assert_eq!(sp.temp, 0.80);
        assert_eq!(sp.samplers, vec![SamplerType::TopK, SamplerType::Temperature]);
        assert!(sp.print().contains("top_k = 11"));
    }

    #[test]
    fn embedding_helpers() {
        let inp = [3.0f32, 4.0];
        let mut out = [0.0f32; 2];
        common_embd_normalize(&inp, &mut out, 2);
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.8).abs() < 1e-6);
        assert_eq!(common_embd_similarity_cos(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        assert_eq!(common_embd_similarity_cos(&[0.0], &[1.0]), 0.0);
        assert_eq!(common_embd_similarity_cos(&[0.0], &[0.0]), 1.0);
    }

    #[test]
    fn filename_and_ffn_helpers() {
        assert!(fs_validate_filename("model.gguf", false));
        assert!(!fs_validate_filename("a/b", false));
        assert!(fs_validate_filename("a/b", true));
        assert!(!fs_validate_filename("..", false));
        assert!(!fs_validate_filename("bad:name", false));
        assert_eq!(
            llm_ffn_block_regex(3, LLM_FFN_DENSE_REGEX),
            "blk\\.3\\.ffn_(up|down|gate)\\."
        );
        let mut overrides = Vec::new();
        llm_add_n_cpu_ffn_overrides(2, LLM_FFN_EXPS_REGEX, &mut overrides);
        assert_eq!(overrides.len(), 2);
        assert!(overrides[0].pattern.starts_with("blk\\.0"));
    }

    #[test]
    fn decision_and_endpoint_helpers() {
        assert_eq!(DecisionType::from_name("lfm3-d1"), DecisionType::Lfm3D1);
        assert_eq!(DecisionType::Unknown.as_name(), None);
        assert_eq!(
            common_get_decision_type(Some("lfm3"), Some("lfm3-d1")),
            DecisionType::Lfm3D1
        );
        assert_eq!(
            common_get_decision_type(None, Some("openjev")),
            DecisionType::None
        );
        common_set_env("LCR_TEST_ENDPOINT", "https://example.org/models");
        let saved = common_get_env("LCR_TEST_ENDPOINT");
        assert_eq!(saved, "https://example.org/models");
        common_set_env("LCR_TEST_ENDPOINT", "");
        assert!(common_get_env("LCR_TEST_ENDPOINT").is_empty());
        assert!(string_from_bool(true) == "true");
        assert_eq!(string_from_ints(&[1, 2]), "[ 1, 2 ]");
        assert_eq!(string_from(true), "true");
    }

    #[test]
    fn timestamp_and_time_meas() {
        let ts = string_get_sortable_timestamp();
        assert_eq!(ts.len(), 29);
        assert!(ts.contains('-') && ts.contains('.'));
        let mut acc = 0i64;
        {
            let _meas = CommonTimeMeas::new(&mut acc, false);
        }
        let mut disabled = 0i64;
        {
            let _meas = CommonTimeMeas::new(&mut disabled, true);
        }
        assert_eq!(disabled, 0);
    }

    #[test]
    fn lr_opt_and_dataset() {
        let mut lr = LrOpt::default();
        lr.lr_min = 1e-6;
        lr.init();
        assert!(lr.scale_epoch > 0.0);
        let pars = common_opt_lr_pars(&lr);
        assert_eq!(pars.alpha, lr.get_lr());
        assert_eq!(common_opt_get_optimizer("SGD"), OptimizerType::Sgd);
        assert_eq!(common_opt_get_optimizer("nope"), OptimizerType::Count);

        let dataset = common_opt_dataset_init(&[0, 1, 2, 3, 4], 2, 1).unwrap();
        assert_eq!(dataset.data, vec![0, 1, 1, 2]);
        assert_eq!(dataset.labels, vec![1, 2, 2, 3]);
    }
}

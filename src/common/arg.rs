//! Rust port of `common/arg.h` and `common/arg.cpp`.
//!
//! Std-only translation of the CLI argument definitions and parser. Handlers
//! are plain function pointers, so an option needs no captured state. Features
//! that need a live backend, downloader, logger or preset engine are accepted
//! and stored but have no side effect; see the end-of-file notes.

use crate::api::LLAMA_DEFAULT_SEED;
use crate::common::{
    common_opt_get_optimizer, fs_get_cache_directory, fs_is_directory,
    llm_add_n_cpu_ffn_overrides, llm_ffn_exps_cpu_override, parse_cpu_mask, parse_cpu_range,
    postprocess_cpu_params, sampler_types_from_names, string_parse_kv_override,
    string_process_escapes, string_replace_all, string_split, string_split_char, string_split_to,
    string_strip,
    tty_can_use_colors, AttentionType, CommonGrammar, ConversationMode, DimreMethod, FlashAttnType,
    GgmlType, GrammarType, LoadMode, LLAMA_MAX_DEVICES, LLM_FFN_DENSE_REGEX,
    LLM_FFN_EXPS_REGEX, LlamaExample, OptimizerType, Params,
    ReasoningFormat, SAMPLING_CONFIG_MIN_P, SAMPLING_CONFIG_MIROSTAT, SAMPLING_CONFIG_MIROSTAT_ETA,
    SAMPLING_CONFIG_MIROSTAT_TAU, SAMPLING_CONFIG_PENALTY_LAST_N,
    SAMPLING_CONFIG_PENALTY_REPEAT, SAMPLING_CONFIG_SAMPLERS, SAMPLING_CONFIG_TEMP,
    SAMPLING_CONFIG_TOP_K, SAMPLING_CONFIG_TOP_P, SAMPLING_CONFIG_XTC_PROBABILITY,
    SAMPLING_CONFIG_XTC_THRESHOLD, SpeculativeType, SplitMode, TensorBuftOverride, LazyMode,
};
use crate::core::{Error, Result};
use crate::cparams::{PoolingType, RopeScalingType};
use crate::ggml::threading::{SchedPriority, GGML_MAX_N_THREADS};
use crate::json::Json;
use crate::sampler::LogitBias;
use std::cmp::Ordering;
use std::env;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

/// Pseudo-env variable that identifies preset-only arguments.
pub const COMMON_ARG_PRESET_LOAD_ON_STARTUP: &str = "__PRESET_LOAD_ON_STARTUP";
pub const COMMON_ARG_PRESET_STOP_TIMEOUT: &str = "__PRESET_STOP_TIMEOUT";
pub const COMMON_ARG_PRESET_DEDUP_CACHE_MODELS: &str = "__PRESET_DEDUP_CACHE_MODELS";

const N_LEADING_SPACES: usize = 40;
const N_CHAR_PER_LINE_HELP: usize = 70;

// ---------------------------------------------------------------------------
// argument description
// ---------------------------------------------------------------------------

/// One CLI action. Non-capturing closures coerce to these pointers.
#[derive(Clone, Copy, Debug)]
pub enum Handler {
    /// Preset placeholder with no runtime effect.
    None,
    Void(fn(&mut Params) -> Result<()>),
    Int(fn(&mut Params, i32) -> Result<()>),
    String(fn(&mut Params, String) -> Result<()>),
    Bool(fn(&mut Params, bool) -> Result<()>),
    StrStr(fn(&mut Params, String, String) -> Result<()>),
}

/// `common_arg`.
#[derive(Clone, Debug)]
pub struct CommonArg {
    pub examples: Vec<LlamaExample>,
    pub excludes: Vec<LlamaExample>,
    pub args: Vec<&'static str>,
    /// For negated args like `--no-xxx`.
    pub args_neg: Vec<&'static str>,
    pub value_hint: Option<String>,
    pub value_hint_2: Option<String>,
    pub env: Option<&'static str>,
    pub help: String,
    pub is_sampling: bool,
    pub is_spec: bool,
    pub is_preset_only: bool,
    pub handler: Handler,
}

impl CommonArg {
    fn string(
        args: &[&'static str],
        hint: impl Into<String>,
        help: impl Into<String>,
        handler: fn(&mut Params, String) -> Result<()>,
    ) -> Self {
        Self {
            examples: vec![LlamaExample::Common],
            excludes: Vec::new(),
            args: args.to_vec(),
            args_neg: Vec::new(),
            value_hint: Some(hint.into()),
            value_hint_2: None,
            env: None,
            help: help.into(),
            is_sampling: false,
            is_spec: false,
            is_preset_only: false,
            handler: Handler::String(handler),
        }
    }

    fn int(
        args: &[&'static str],
        hint: impl Into<String>,
        help: impl Into<String>,
        handler: fn(&mut Params, i32) -> Result<()>,
    ) -> Self {
        Self {
            examples: vec![LlamaExample::Common],
            excludes: Vec::new(),
            args: args.to_vec(),
            args_neg: Vec::new(),
            value_hint: Some(hint.into()),
            value_hint_2: None,
            env: None,
            help: help.into(),
            is_sampling: false,
            is_spec: false,
            is_preset_only: false,
            handler: Handler::Int(handler),
        }
    }

    fn void(
        args: &[&'static str],
        help: impl Into<String>,
        handler: fn(&mut Params) -> Result<()>,
    ) -> Self {
        Self {
            examples: vec![LlamaExample::Common],
            excludes: Vec::new(),
            args: args.to_vec(),
            args_neg: Vec::new(),
            value_hint: None,
            value_hint_2: None,
            env: None,
            help: help.into(),
            is_sampling: false,
            is_spec: false,
            is_preset_only: false,
            handler: Handler::Void(handler),
        }
    }

    fn boolean(
        args: &[&'static str],
        args_neg: &[&'static str],
        help: impl Into<String>,
        handler: fn(&mut Params, bool) -> Result<()>,
    ) -> Self {
        Self {
            examples: vec![LlamaExample::Common],
            excludes: Vec::new(),
            args: args.to_vec(),
            args_neg: args_neg.to_vec(),
            value_hint: None,
            value_hint_2: None,
            env: None,
            help: help.into(),
            is_sampling: false,
            is_spec: false,
            is_preset_only: false,
            handler: Handler::Bool(handler),
        }
    }

    fn str2(
        args: &[&'static str],
        hint: impl Into<String>,
        hint2: impl Into<String>,
        help: impl Into<String>,
        handler: fn(&mut Params, String, String) -> Result<()>,
    ) -> Self {
        Self {
            examples: vec![LlamaExample::Common],
            excludes: Vec::new(),
            args: args.to_vec(),
            args_neg: Vec::new(),
            value_hint: Some(hint.into()),
            value_hint_2: Some(hint2.into()),
            env: None,
            help: help.into(),
            is_sampling: false,
            is_spec: false,
            is_preset_only: false,
            handler: Handler::StrStr(handler),
        }
    }

    fn set_examples(mut self, examples: &[LlamaExample]) -> Self {
        self.examples = examples.to_vec();
        self
    }

    fn set_excludes(mut self, excludes: &[LlamaExample]) -> Self {
        self.excludes = excludes.to_vec();
        self
    }

    fn set_env(mut self, name: &'static str) -> Self {
        self.help = format!("{}\n(env: {})", self.help, name);
        self.env = Some(name);
        self
    }

    fn set_sampling(mut self) -> Self {
        self.is_sampling = true;
        self
    }

    fn set_spec(mut self) -> Self {
        self.is_spec = true;
        self
    }

    fn set_preset_only(mut self) -> Self {
        self.is_preset_only = true;
        self
    }

    fn in_example(&self, ex: LlamaExample) -> bool {
        self.examples.contains(&ex)
    }

    fn is_exclude(&self, ex: LlamaExample) -> bool {
        self.excludes.contains(&ex)
    }

    /// `get_value_from_env`: `None` means no usable variable is set.
    pub fn get_value_from_env(&self) -> Option<String> {
        let env_name = self.env?;
        if !self.args_neg.is_empty() {
            let mut neg_env = env_name.to_owned();
            string_replace_all(&mut neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
            if env::var(&neg_env).is_ok() {
                return Some("0".to_owned());
            }
        }
        env::var(env_name).ok()
    }

    pub fn has_value_from_env(&self) -> bool {
        let Some(env_name) = self.env else {
            return false;
        };
        if !self.args_neg.is_empty() {
            let mut neg_env = env_name.to_owned();
            string_replace_all(&mut neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
            if env::var(&neg_env).is_ok() {
                return true;
            }
        }
        env::var(env_name).is_ok()
    }

    /// `common_arg::to_string`, used in per-argument error messages and help.
    pub fn to_string(&self) -> String {
        let all_args = self.get_args();
        let leading = " ".repeat(N_LEADING_SPACES);
        let mut ss = String::new();
        for (i, arg) in all_args.iter().enumerate() {
            if i == 0 {
                if all_args.len() == 1 {
                    ss.push_str(arg);
                } else {
                    let tmp = format!("{arg}, ");
                    let spaces = " ".repeat((7i32 - tmp.len() as i32).max(0) as usize);
                    ss.push_str(&tmp);
                    ss.push_str(&spaces);
                }
            } else {
                ss.push_str(arg);
                if i + 1 != all_args.len() {
                    ss.push_str(", ");
                }
            }
        }
        if let Some(hint) = &self.value_hint {
            ss.push(' ');
            ss.push_str(hint);
        }
        if let Some(hint) = &self.value_hint_2 {
            ss.push(' ');
            ss.push_str(hint);
        }
        if ss.len() > N_LEADING_SPACES - 3 {
            ss.push('\n');
            ss.push_str(&leading);
        } else {
            ss.push_str(&" ".repeat(N_LEADING_SPACES - ss.len()));
        }
        for (i, line) in break_str_into_lines(&self.help, N_CHAR_PER_LINE_HELP)
            .iter()
            .enumerate()
        {
            if i > 0 {
                ss.push_str(&leading);
            }
            ss.push_str(line);
            ss.push('\n');
        }
        ss
    }

    /// All positive and negative aliases.
    pub fn get_args(&self) -> Vec<String> {
        let mut result: Vec<String> = self.args.iter().map(|s| (*s).to_owned()).collect();
        result.extend(self.args_neg.iter().map(|s| (*s).to_owned()));
        result
    }

    /// Env variable plus its `LLAMA_ARG_NO_` compatibility variant.
    pub fn get_env(&self) -> Vec<String> {
        let Some(env_name) = self.env else {
            return Vec::new();
        };
        let mut result = vec![env_name.to_owned()];
        if !self.args_neg.is_empty() {
            let mut neg_env = env_name.to_owned();
            string_replace_all(&mut neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
            result.push(neg_env);
        }
        result
    }

    fn apply_env(&self, params: &mut Params, value: &str) -> Result<()> {
        match self.handler {
            Handler::None => Ok(()),
            Handler::Void(h) => {
                if is_truthy(value) {
                    h(params)?;
                }
                Ok(())
            }
            Handler::Int(h) => {
                let v = parse_number_prefix(value)?;
                h(params, v)
            }
            Handler::Bool(h) => h(params, parse_bool_value(value)?),
            Handler::String(h) => h(params, value.to_owned()),
            Handler::StrStr(_) => Ok(()),
        }
    }

    fn apply_cli(
        &self,
        params: &mut Params,
        is_positive: bool,
        val: Option<String>,
        val2: Option<String>,
    ) -> Result<()> {
        match self.handler {
            Handler::None => Ok(()),
            Handler::Void(h) => h(params),
            Handler::Bool(h) => h(params, is_positive),
            Handler::Int(h) => h(params, parse_number_prefix(val.as_deref().unwrap_or_default())?),
            Handler::String(h) => h(params, val.unwrap_or_default()),
            Handler::StrStr(h) => h(params, val.unwrap_or_default(), val2.unwrap_or_default()),
        }
    }
}

impl PartialEq for CommonArg {
    fn eq(&self, other: &Self) -> bool {
        match (self.args.first(), other.args.first()) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for CommonArg {}

impl PartialOrd for CommonArg {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CommonArg {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.args.first(), other.args.first()) {
            (Some(a), Some(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

// ---------------------------------------------------------------------------
// utils
// ---------------------------------------------------------------------------

pub fn is_truthy(value: &str) -> bool {
    matches!(value, "on" | "enabled" | "true" | "1")
}

pub fn is_falsey(value: &str) -> bool {
    matches!(value, "off" | "disabled" | "false" | "0")
}

pub fn is_autoy(value: &str) -> bool {
    matches!(value, "auto" | "-1")
}

/// Namespace alias matching `common_arg_utils` in the C++ source.
pub mod common_arg_utils {
    pub use super::{is_autoy, is_falsey, is_truthy};
}

fn parse_bool_value(value: &str) -> Result<bool> {
    if is_truthy(value) {
        Ok(true)
    } else if is_falsey(value) {
        Ok(false)
    } else {
        Err(Error::new("invalid boolean value"))
    }
}

/// C++ `std::stoi`/`stof`/`stoul` accept a numeric prefix and ignore trailing text.
fn parse_number_prefix<T: std::str::FromStr>(value: &str) -> Result<T> {
    let text = value.trim_start();
    for end in (1..=text.len()).rev() {
        if let Ok(v) = text[..end].parse::<T>() {
            return Ok(v);
        }
    }
    Err(Error::new(format!("invalid numeric value: '{value}'")))
}

/// Parse a value that must be consumed completely, as C++ `std::stod(text, &pos)` checks.
fn parse_number_exact<T: std::str::FromStr>(value: &str) -> Result<T> {
    let text = string_strip(value);
    text.parse::<T>()
        .map_err(|_| Error::new(format!("invalid value: '{value}'")))
}

/// `parse_cpu_mask` for the `Vec<bool>` used by `CpuParams`.
fn cpu_mask(mask: &str, dst: &mut Vec<bool>) -> bool {
    let mut tmp = [false; GGML_MAX_N_THREADS];
    if !parse_cpu_mask(mask, &mut tmp) {
        return false;
    }
    dst.resize(GGML_MAX_N_THREADS, false);
    dst.copy_from_slice(&tmp);
    true
}

/// `parse_cpu_range` for the `Vec<bool>` used by `CpuParams`.
fn cpu_range(range: &str, dst: &mut Vec<bool>) -> bool {
    let mut tmp = [false; GGML_MAX_N_THREADS];
    if !parse_cpu_range(range, &mut tmp) {
        return false;
    }
    dst.resize(GGML_MAX_N_THREADS, false);
    dst.copy_from_slice(&tmp);
    true
}

fn hardware_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(0)
}

fn sched_priority(value: i32) -> SchedPriority {
    match value {
        -1 => SchedPriority::Low,
        1 => SchedPriority::Medium,
        2 => SchedPriority::High,
        3 => SchedPriority::Realtime,
        _ => SchedPriority::Normal,
    }
}

fn priority_value(value: &SchedPriority) -> i32 {
    match value {
        SchedPriority::Low => -1,
        SchedPriority::Normal => 0,
        SchedPriority::Medium => 1,
        SchedPriority::High => 2,
        SchedPriority::Realtime => 3,
    }
}

/// `parse_csv_row`: quoted fields and doubled quotes.
pub fn parse_csv_row(input: &str) -> Vec<String> {
    let chars: Vec<char> = input.chars().collect();
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut i = 0usize;
    while i < chars.len() {
        match chars[i] {
            '"' => {
                if !in_quotes {
                    if !field.is_empty() {
                        field.push('"');
                    } else {
                        in_quotes = true;
                    }
                } else if i + 1 < chars.len() && chars[i + 1] == '"' {
                    field.push('"');
                    i += 1;
                } else {
                    in_quotes = false;
                }
            }
            ',' => {
                if in_quotes {
                    field.push(',');
                } else {
                    fields.push(std::mem::take(&mut field));
                }
            }
            c => field.push(c),
        }
        i += 1;
    }
    fields.push(field);
    fields
}

fn break_str_into_lines(input: &str, max_char_per_line: usize) -> Vec<String> {
    let mut result = Vec::new();
    for line in input.lines() {
        if line.len() <= max_char_per_line {
            result.push(line.to_owned());
            continue;
        }
        let mut current = String::new();
        for word in line.split_whitespace() {
            let added = usize::from(!current.is_empty()) + word.len();
            if current.len() + added > max_char_per_line {
                if !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
                current = word.to_owned();
            } else {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(word);
            }
        }
        if !current.is_empty() {
            result.push(current);
        }
    }
    result
}

/// `read_file` from the C++ source.
fn read_file(fname: &str) -> Result<String> {
    fs::read_to_string(fname)
        .map_err(|_| Error::new(format!("error: failed to open file '{fname}'\n")))
}

fn read_bytes(fname: &str) -> Result<Vec<u8>> {
    fs::read(fname).map_err(|_| Error::new(format!("error: failed to open file '{fname}'\n")))
}

/// `clean_file_name` from the C++ source.
pub fn clean_file_name(fname: &str) -> String {
    let mut result = fname.to_owned();
    string_replace_all(&mut result, "\\", "_");
    string_replace_all(&mut result, "/", "_");
    result
}

const KV_CACHE_TYPES: &[GgmlType] = &[
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Bf16,
    GgmlType::Q8_0,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Iq4Nl,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
];

fn kv_cache_type_from_str(s: &str) -> Result<GgmlType> {
    for &t in KV_CACHE_TYPES {
        if t.name() == s {
            return Ok(t);
        }
    }
    Err(Error::new(format!("Unsupported cache type: {s}")))
}

fn get_all_kv_cache_types() -> String {
    let names: Vec<&str> = KV_CACHE_TYPES.iter().map(|t| t.name()).collect();
    names.join(", ")
}

fn spec_type_name(t: SpeculativeType) -> &'static str {
    match t {
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

fn spec_all_types() -> String {
    [
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
    ]
    .iter()
    .map(|t| spec_type_name(*t))
    .collect::<Vec<_>>()
    .join(",")
}

fn spec_type_name_str(types: &[SpeculativeType]) -> String {
    types
        .iter()
        .map(|t| spec_type_name(*t))
        .collect::<Vec<_>>()
        .join(",")
}

fn spec_types_from_names(names: &[String]) -> Result<Vec<SpeculativeType>> {
    let all = [
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
    let mut result = Vec::with_capacity(names.len());
    for name in names {
        let Some(t) = all.iter().find(|t| spec_type_name(**t) == name) else {
            return Err(Error::new(format!("unknown speculative type: {name}")));
        };
        if *t == SpeculativeType::None {
            return Ok(vec![SpeculativeType::None]);
        }
        result.push(*t);
    }
    Ok(result)
}

fn sampler_types_from_chars(chars: &str) -> Vec<crate::common::SamplerType> {
    use crate::common::SamplerType;
    chars
        .chars()
        .filter_map(|c| match c {
            'd' => Some(SamplerType::Dry),
            'k' => Some(SamplerType::TopK),
            'y' => Some(SamplerType::TypicalP),
            'p' => Some(SamplerType::TopP),
            's' => Some(SamplerType::TopNSigma),
            'm' => Some(SamplerType::MinP),
            't' => Some(SamplerType::Temperature),
            'x' => Some(SamplerType::Xtc),
            'i' => Some(SamplerType::Infill),
            'e' => Some(SamplerType::Penalties),
            'a' => Some(SamplerType::AdaptiveP),
            _ => None,
        })
        .collect()
}

fn reasoning_format_from(value: &str) -> Result<ReasoningFormat> {
    match value {
        "none" => Ok(ReasoningFormat::None),
        "auto" => Ok(ReasoningFormat::Auto),
        "deepseek" => Ok(ReasoningFormat::Deepseek),
        "deepseek-legacy" => Ok(ReasoningFormat::DeepseekLegacy),
        _ => Err(Error::new(format!("Unknown reasoning format: {value}"))),
    }
}

/// `parse_device_list`, reduced to device indices since no backend registry exists.
fn parse_device_list(value: &str) -> Result<Vec<usize>> {
    let names = string_split_char(value, ',');
    if names.is_empty() {
        return Err(Error::new("no devices specified"));
    }
    if names.len() == 1 && names[0] == "none" {
        return Ok(Vec::new());
    }
    let mut devices = Vec::with_capacity(names.len());
    for name in names {
        let idx = name
            .parse::<usize>()
            .map_err(|_| Error::new(format!("invalid device: {name}")))?;
        devices.push(idx);
    }
    Ok(devices)
}

/// `parse_tensor_buffer_overrides`, without backend buffer-type validation.
fn parse_tensor_buffer_overrides(
    value: &str,
    overrides: &mut Vec<TensorBuftOverride>,
) -> Result<()> {
    for item in string_split_char(value, ',') {
        let Some(pos) = item.find('=') else {
            return Err(Error::new("invalid value"));
        };
        overrides.push(TensorBuftOverride {
            pattern: item[..pos].to_owned(),
            buffer_type: item[pos + 1..].to_owned(),
        });
    }
    Ok(())
}

/// `common_get_default_local_path`.
pub fn common_get_default_local_path(url: &str) -> Result<String> {
    let mut name = url;
    if let Some(pos) = name.find('#') {
        name = &name[..pos];
    }
    if let Some(pos) = name.find('?') {
        name = &name[..pos];
    }
    let file = string_split(name, "/").pop().unwrap_or_default();
    let path = fs_get_cache_directory()?.join(file);
    Ok(path.to_string_lossy().into_owned())
}

/// Print the GGUF files found in the cache directory.
fn print_cached_models() {
    let Ok(dir) = fs_get_cache_directory() else {
        println!("number of models in cache: 0");
        return;
    };
    let mut models = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".gguf") {
                models.push(name);
            }
        }
    }
    models.sort();
    println!("number of models in cache: {}", models.len());
    for (i, model) in models.iter().enumerate() {
        println!("{:4}. {}", i + 1, model);
    }
}

// ---------------------------------------------------------------------------
// parser context
// ---------------------------------------------------------------------------

/// `common_params_context`.
#[derive(Clone, Debug)]
pub struct ParamsContext {
    pub ex: LlamaExample,
    pub params: Params,
    pub options: Vec<CommonArg>,
}

impl ParamsContext {
    pub fn new(params: Params) -> Self {
        Self {
            ex: LlamaExample::Common,
            params,
            options: Vec::new(),
        }
    }
}

fn add_opt(options: &mut Vec<CommonArg>, ex: LlamaExample, opt: CommonArg) {
    // download only exposes the handful of args explicitly tagged for it
    let inherit_common = ex != LlamaExample::Download;
    if (opt.in_example(ex) || (inherit_common && opt.in_example(LlamaExample::Common)))
        && !opt.is_exclude(ex)
    {
        options.push(opt);
    }
}

/// `common_params_parser_init`.
pub fn common_params_parser_init(params: &Params, ex: LlamaExample) -> ParamsContext {
    let mut params = params.clone();

    // per-example default params
    match ex {
        LlamaExample::Completion | LlamaExample::Mtmd => {
            params.use_jinja = false;
            if ex == LlamaExample::Mtmd {
                params.sampling.temp = 0.2;
            }
        }
        LlamaExample::Server => params.n_parallel = -1,
        LlamaExample::Tokenize => params.parse_special = true,
        LlamaExample::Tts => {
            params.out_file = "output.wav".into();
            params.sampling.penalty_repeat = 1.05;
            params.sampling.penalty_last_n = -1;
        }
        _ => {}
    }
    params.use_color = tty_can_use_colors();

    let mut ctx = ParamsContext::new(params);
    ctx.ex = ex;

    let mut sampler_type_chars = String::new();
    let mut sampler_type_names = String::new();
    for sampler in &ctx.params.sampling.samplers {
        sampler_type_chars.push(sampler.as_char());
        sampler_type_names.push_str(sampler.as_str());
        sampler_type_names.push(';');
    }
    if sampler_type_names.ends_with(';') {
        sampler_type_names.pop();
    }

    let mmproj_examples = &[
        LlamaExample::Mtmd,
        LlamaExample::Server,
        LlamaExample::Cli,
        LlamaExample::Tts,
    ];
    let spec_examples = &[
        LlamaExample::Speculative,
        LlamaExample::Server,
        LlamaExample::Cli,
    ];
    let dry_breakers = {
        let mut text = String::new();
        for (i, b) in ctx.params.sampling.dry_sequence_breakers.iter().enumerate() {
            if i > 0 {
                text.push_str(", '");
            } else {
                text.push('\'');
            }
            text.push_str(if b == "\n" { "\\n" } else { b });
            text.push('\'');
        }
        if text.is_empty() {
            "none".to_owned()
        } else {
            text
        }
    };

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["-h", "--help", "--usage"], "print usage and exit", |p| {
            p.usage = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Common, LlamaExample::Download]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--version"], "show version and build info", |_| {
            println!("llama.cpp rust port {}", option_env!("CARGO_PKG_VERSION").unwrap_or("0.1.0"));
            std::process::exit(0);
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["-cl", "--cache-list"], "show list of models in cache", |_| {
            print_cached_models();
            std::process::exit(0);
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--completion-bash"], "print source-able bash completion script for llama.cpp", |p| {
            p.completion = true;
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--server-base"], "URL", "connect to this server instead of starting a new one, example: 'http://localhost:9931' (default: none)", |p, v| {
            p.server_base = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--verbose-prompt"], format!("print a verbose prompt before generation (default: {})", ctx.params.verbose_prompt), |p| {
            p.verbose_prompt = true;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Completion, LlamaExample::Cli, LlamaExample::Embedding,
            LlamaExample::Retrieval,
        ]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--display-prompt"], &["--no-display-prompt"], format!("whether to print prompt at generation (default: {})", ctx.params.display_prompt), |p, v| {
            p.display_prompt = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion, LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-co", "--color"], "[on|off|auto]", "Colorize output to distinguish prompt and user input from generations ('on', 'off', or 'auto', default: 'auto')\n'auto' enables colors when output is to a terminal", |p, v| {
            if is_truthy(&v) {
                p.use_color = true;
            } else if is_falsey(&v) {
                p.use_color = false;
            } else if is_autoy(&v) {
                p.use_color = tty_can_use_colors();
            } else {
                return Err(Error::new(format!("error: unknown value for --color: '{v}'\n")));
            }
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Completion, LlamaExample::Cli, LlamaExample::Speculative,
            LlamaExample::Lookup,
        ]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-t", "--threads"], "N", format!("number of CPU threads to use during generation (default: {})", ctx.params.cpuparams.n_threads), |p, v| {
            p.cpuparams.n_threads = if v <= 0 { hardware_threads() } else { v };
            Ok(())
        })
        .set_env("LLAMA_ARG_THREADS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-tb", "--threads-batch"], "N", "number of threads to use during batch and prompt processing (default: same as --threads)", |p, v| {
            p.cpuparams_batch.n_threads = if v <= 0 { hardware_threads() } else { v };
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-C", "--cpu-mask"], "M", "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")", |p, v| {
            p.cpuparams.mask_valid = true;
            if !cpu_mask(&v, &mut p.cpuparams.cpumask) {
                return Err(Error::new("invalid cpumask"));
            }
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-Cr", "--cpu-range"], "lo-hi", "range of CPUs for affinity. Complements --cpu-mask", |p, v| {
            p.cpuparams.mask_valid = true;
            if !cpu_range(&v, &mut p.cpuparams.cpumask) {
                return Err(Error::new("invalid range"));
            }
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--cpu-strict"], "<0|1>", format!("use strict CPU placement (default: {})\n", ctx.params.cpuparams.strict_cpu), |p, v| {
            p.cpuparams.strict_cpu = parse_number_prefix::<u32>(&v)? != 0;
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--prio"], "N", format!("set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: {})\n", priority_value(&ctx.params.cpuparams.priority)), |p, v| {
            if v < -1 || v > 3 {
                return Err(Error::new("invalid value"));
            }
            p.cpuparams.priority = sched_priority(v);
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--poll"], "<0...100>", format!("use polling level to wait for work (0 - no polling, default: {})\n", ctx.params.cpuparams.poll), |p, v| {
            p.cpuparams.poll = parse_number_prefix(&v)?;
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-Cb", "--cpu-mask-batch"], "M", "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)", |p, v| {
            p.cpuparams_batch.mask_valid = true;
            if !cpu_mask(&v, &mut p.cpuparams_batch.cpumask) {
                return Err(Error::new("invalid cpumask"));
            }
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-Crb", "--cpu-range-batch"], "lo-hi", "ranges of CPUs for affinity. Complements --cpu-mask-batch", |p, v| {
            p.cpuparams_batch.mask_valid = true;
            if !cpu_range(&v, &mut p.cpuparams_batch.cpumask) {
                return Err(Error::new("invalid range"));
            }
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--cpu-strict-batch"], "<0|1>", "use strict CPU placement (default: same as --cpu-strict)", |p, v| {
            p.cpuparams_batch.strict_cpu = v != 0;
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--prio-batch"], "N", format!("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: {})\n", priority_value(&ctx.params.cpuparams_batch.priority)), |p, v| {
            if !(0..=3).contains(&v) {
                return Err(Error::new("invalid value"));
            }
            p.cpuparams_batch.priority = sched_priority(v);
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--poll-batch"], "<0|1>", "use polling to wait for work (default: same as --poll)", |p, v| {
            p.cpuparams_batch.poll = v.max(0) as u32;
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-lcs", "--lookup-cache-static"], "FNAME", "path to static lookup cache to use for lookup decoding (not updated by generation)", |p, v| {
            p.speculative.ngram_cache.lookup_cache_static = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Lookup, LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-lcd", "--lookup-cache-dynamic"], "FNAME", "path to dynamic lookup cache to use for lookup decoding (updated by generation)", |p, v| {
            p.speculative.ngram_cache.lookup_cache_dynamic = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Lookup, LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-c", "--ctx-size"], "N", format!("size of the prompt context (default: {}, 0 = loaded from model)", ctx.params.n_ctx), |p, v| {
            p.n_ctx = v;
            if v == 0 {
                p.fit_params_min_ctx = i32::MAX;
            }
            Ok(())
        })
        .set_env("LLAMA_ARG_CTX_SIZE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--kv-unified-per-slot"], "N", "context limit per parallel slot (default: unset, behavior unchanged).\nwhen set without -c/--ctx-size, the shared KV pool is sized to n_parallel*N", |p, v| {
            p.kv_unified_per_slot = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_KV_UNIFIED_PER_SLOT")
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-n", "--predict", "--n-predict"], "N", if ctx.ex == LlamaExample::Completion {
            format!("number of tokens to predict (default: {}, -1 = infinity, -2 = until context filled)", ctx.params.n_predict)
        } else {
            format!("number of tokens to predict (default: {}, -1 = infinity)", ctx.params.n_predict)
        }, |p, v| {
            p.n_predict = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_N_PREDICT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-b", "--batch-size"], "N", format!("logical maximum batch size (default: {})", ctx.params.n_batch), |p, v| {
            p.n_batch = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_BATCH"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-ub", "--ubatch-size"], "N", format!("physical maximum batch size (default: {})", ctx.params.n_ubatch), |p, v| {
            p.n_ubatch = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_UBATCH"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--keep"], "N", format!("number of tokens to keep from the initial prompt (default: {}, -1 = all)", ctx.params.n_keep), |p, v| {
            p.n_keep = v;
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--swa-full"], format!("use full-size SWA cache (default: {})\n[(more info)](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)", ctx.params.swa_full), |p| {
            p.swa_full = true;
            Ok(())
        })
        .set_env("LLAMA_ARG_SWA_FULL"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"], "N", format!("max number of context checkpoints to create per slot (default: {})[(more info)](https://github.com/ggml-org/llama.cpp/pull/15293)", ctx.params.n_ctx_checkpoints), |p, v| {
            p.n_ctx_checkpoints = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_CTX_CHECKPOINTS")
        .set_examples(&[LlamaExample::Server, LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-cms", "--checkpoint-min-step"], "N", format!("minimum spacing between context checkpoints in tokens (default: {}, 0 = no minimum)", ctx.params.checkpoint_min_step), |p, v| {
            if v < 0 {
                return Err(Error::new("checkpoint-min-step must be non-negative"));
            }
            p.checkpoint_min_step = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT")
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-cram", "--cache-ram"], "N", format!("set the maximum cache size in MiB (default: {}, 0 for decision models, -1 - no limit, 0 - disable)[(more info)](https://github.com/ggml-org/llama.cpp/pull/16391)", ctx.params.cache_ram_mib), |p, v| {
            p.cache_ram_mib = v;
            p.cache_ram_explicit = true;
            Ok(())
        })
        .set_env("LLAMA_ARG_CACHE_RAM")
        .set_examples(&[LlamaExample::Server, LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["-kvu", "--kv-unified"], &["-no-kvu", "--no-kv-unified"], "use single unified KV buffer shared across all sequences (default: enabled if number of slots is auto)", |p, v| {
            p.kv_unified = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_KV_UNIFIED")
        .set_examples(&[
            LlamaExample::Server, LlamaExample::Perplexity, LlamaExample::Batched,
            LlamaExample::Bench, LlamaExample::Parallel, LlamaExample::Speculative,
        ]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--cache-idle-slots"], &["--no-cache-idle-slots"], "save idle slots to the prompt cache on new task, and clear them when using unified KV (default: enabled, requires cache-ram)", |p, v| {
            p.cache_idle_slots = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_CACHE_IDLE_SLOTS")
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--context-shift"], &["--no-context-shift"], format!("whether to use context shift on infinite text generation (default: {})", if ctx.params.ctx_shift { "enabled" } else { "disabled" }), |p, v| {
            p.ctx_shift = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Completion, LlamaExample::Cli, LlamaExample::Server,
            LlamaExample::Imatrix, LlamaExample::Perplexity,
        ])
        .set_env("LLAMA_ARG_CONTEXT_SHIFT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--chunks"], "N", format!("max number of chunks to process (default: {}, -1 = all)", ctx.params.n_chunks), |p, v| {
            p.n_chunks = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Imatrix, LlamaExample::Perplexity, LlamaExample::Retrieval,
        ]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-fa", "--flash-attn"], "[on|off|auto]", format!("set Flash Attention use ('on', 'off', or 'auto', default: '{}')", match ctx.params.flash_attn_type { FlashAttnType::Enabled => "on", FlashAttnType::Disabled => "off", FlashAttnType::Auto => "auto" }), |p, v| {
            if is_truthy(&v) {
                p.flash_attn_type = FlashAttnType::Enabled;
            } else if is_falsey(&v) {
                p.flash_attn_type = FlashAttnType::Disabled;
            } else if is_autoy(&v) {
                p.flash_attn_type = FlashAttnType::Auto;
            } else {
                return Err(Error::new(format!("error: unknown value for --flash-attn: '{v}'\n")));
            }
            Ok(())
        })
        .set_env("LLAMA_ARG_FLASH_ATTN"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-p", "--prompt"], "PROMPT", "prompt to start generation with; for system message, use -sys", |p, v| {
            p.prompt = v;
            Ok(())
        })
        .set_excludes(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-sys", "--system-prompt"], "PROMPT", "system prompt to use with model (if applicable, depending on chat template)", |p, v| {
            p.system_prompt = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Completion, LlamaExample::Cli, LlamaExample::Diffusion,
            LlamaExample::Mtmd,
        ]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--perf"], &["--no-perf"], format!("whether to enable internal libllama performance timings (default: {})", !ctx.params.no_perf), |p, v| {
            p.no_perf = !v;
            p.sampling.no_perf = !v;
            Ok(())
        })
        .set_env("LLAMA_ARG_PERF"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--show-timings"], &["--no-show-timings"], format!("whether to show timing information after each response (default: {})", ctx.params.show_timings), |p, v| {
            p.show_timings = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Cli])
        .set_env("LLAMA_ARG_SHOW_TIMINGS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-f", "--file"], "FNAME", "a file containing the prompt (default: none)", |p, v| {
            p.prompt = read_file(&v)?;
            p.prompt_file = v;
            if p.prompt.ends_with('\n') {
                p.prompt.pop();
            }
            Ok(())
        })
        .set_excludes(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-sysf", "--system-prompt-file"], "FNAME", "a file containing the system prompt (default: none)", |p, v| {
            p.system_prompt = read_file(&v)?;
            if p.system_prompt.ends_with('\n') {
                p.system_prompt.pop();
            }
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Completion, LlamaExample::Cli, LlamaExample::Diffusion,
        ]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--in-file"], "FNAME", "an input file (use comma-separated values to specify multiple files)", |p, v| {
            for item in parse_csv_row(&v) {
                if fs::metadata(&item).is_err() {
                    return Err(Error::new(format!("error: failed to open file '{item}'\n")));
                }
                p.in_files.push(item);
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Imatrix]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-bf", "--binary-file"], "FNAME", "binary file containing the prompt (default: none)", |p, v| {
            let bytes = read_bytes(&v)?;
            p.prompt_file = v.clone();
            p.prompt = String::from_utf8_lossy(&bytes).into_owned();
            eprintln!("Read {} bytes from binary file {}", bytes.len(), v);
            Ok(())
        })
        .set_excludes(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["-e", "--escape"], &["--no-escape"], format!("whether to process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: {})", ctx.params.escape), |p, v| {
            p.escape = v;
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-ptc", "--print-token-count"], "N", format!("print token count every N tokens (default: {})", ctx.params.n_print), |p, v| {
            p.n_print = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--prompt-cache"], "FNAME", "file to cache prompt state for faster startup (default: none)", |p, v| {
            p.path_prompt_cache = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--prompt-cache-all"], "if specified, saves user input and generations to cache as well\n", |p| {
            p.prompt_cache_all = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--prompt-cache-ro"], "if specified, uses the prompt cache but does not update it", |p| {
            p.prompt_cache_ro = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-r", "--reverse-prompt"], "PROMPT", "halt generation at PROMPT, return control in interactive mode\n", |p, v| {
            p.antiprompt.push(v);
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Completion, LlamaExample::Cli, LlamaExample::Server,
        ]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["-sp", "--special"], format!("special tokens output enabled (default: {})", ctx.params.special), |p| {
            p.special = true;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Completion, LlamaExample::Cli, LlamaExample::Server,
        ]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["-cnv", "--conversation"], &["-no-cnv", "--no-conversation"], "whether to run in conversation mode:\n- does not print special tokens and suffix/prefix\n- interactive mode is also enabled\n(default: auto enabled if chat template is available)", |p, v| {
            p.conversation_mode = if v {
                ConversationMode::Enabled
            } else {
                ConversationMode::Disabled
            };
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["-st", "--single-turn"], "run conversation for a single turn only, then exit when done\nwill not be interactive if first turn is predefined with --prompt\n(default: false)", |p| {
            p.single_turn = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion, LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["-i", "--interactive"], format!("run in interactive mode (default: {})", ctx.params.interactive), |p| {
            p.interactive = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["-if", "--interactive-first"], format!("run in interactive mode and wait for input right away (default: {})", ctx.params.interactive_first), |p| {
            p.interactive_first = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["-mli", "--multiline-input"], "allows you to write or paste multiple lines without ending each in '\\'", |p| {
            p.multiline_input = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion, LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--in-prefix-bos"], "prefix BOS to user inputs, preceding the `--in-prefix` string", |p| {
            p.input_prefix_bos = true;
            p.enable_chat_template = false;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--in-prefix"], "STRING", "string to prefix user inputs with (default: empty)", |p, v| {
            p.input_prefix = v;
            p.enable_chat_template = false;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--in-suffix"], "STRING", "string to suffix after user inputs with (default: empty)", |p, v| {
            p.input_suffix = v;
            p.enable_chat_template = false;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--warmup"], &["--no-warmup"], format!("whether to perform warmup with an empty run (default: {})", if ctx.params.warmup { "enabled" } else { "disabled" }), |p, v| {
            p.warmup = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Completion, LlamaExample::Cli, LlamaExample::Server,
            LlamaExample::Mtmd, LlamaExample::Embedding, LlamaExample::Retrieval,
            LlamaExample::Perplexity, LlamaExample::Debug,
        ]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--spm-infill"], format!("use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this. (default: {})", if ctx.params.spm_infill { "enabled" } else { "disabled" }), |p| {
            p.spm_infill = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server]),
    );

    //
    // sampling params
    //

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--samplers"], "SAMPLERS", format!("samplers that will be used for generation in the order, separated by ';'\n(default: {})", sampler_type_names), |p, v| {
            let names = string_split_char(&v, ';');
            p.sampling.samplers = sampler_types_from_names(&names);
            p.sampling.user_sampling_config |= SAMPLING_CONFIG_SAMPLERS;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-s", "--seed"], "SEED", format!("RNG seed (default: {}, use random seed for {})", ctx.params.sampling.seed, LLAMA_DEFAULT_SEED), |p, v| {
            p.sampling.seed = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--sampler-seq", "--sampling-seq"], "SEQUENCE", format!("simplified sequence for samplers that will be used (default: {})", sampler_type_chars), |p, v| {
            p.sampling.samplers = sampler_types_from_chars(&v);
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--ignore-eos"], "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)", |p| {
            p.sampling.ignore_eos = true;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--temp", "--temperature"], "N", format!("temperature (default: {:.2})", ctx.params.sampling.temp as f64), |p, v| {
            p.sampling.temp = parse_number_prefix::<f32>(&v)?.max(0.0);
            p.sampling.user_sampling_config |= SAMPLING_CONFIG_TEMP;
            Ok(())
        })
        .set_sampling()
        .set_env("LLAMA_ARG_TEMPERATURE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--top-k"], "N", format!("top-k sampling (default: {}, 0 = disabled)", ctx.params.sampling.top_k), |p, v| {
            p.sampling.top_k = v;
            p.sampling.user_sampling_config |= SAMPLING_CONFIG_TOP_K;
            Ok(())
        })
        .set_sampling()
        .set_env("LLAMA_ARG_TOP_K"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--top-p"], "N", format!("top-p sampling (default: {:.2}, 1.0 = disabled)", ctx.params.sampling.top_p as f64), |p, v| {
            p.sampling.top_p = parse_number_prefix(&v)?;
            p.sampling.user_sampling_config |= SAMPLING_CONFIG_TOP_P;
            Ok(())
        })
        .set_sampling()
        .set_env("LLAMA_ARG_TOP_P"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--min-p"], "N", format!("min-p sampling (default: {:.2}, 0.0 = disabled)", ctx.params.sampling.min_p as f64), |p, v| {
            p.sampling.min_p = parse_number_prefix(&v)?;
            p.sampling.user_sampling_config |= SAMPLING_CONFIG_MIN_P;
            Ok(())
        })
        .set_sampling()
        .set_env("LLAMA_ARG_MIN_P"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--top-nsigma", "--top-n-sigma"], "N", format!("top-n-sigma sampling (default: {:.2}, -1.0 = disabled)", ctx.params.sampling.top_n_sigma), |p, v| {
            p.sampling.top_n_sigma = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--xtc-probability"], "N", format!("xtc probability (default: {:.2}, 0.0 = disabled)", ctx.params.sampling.xtc_probability as f64), |p, v| {
            p.sampling.xtc_probability = parse_number_prefix(&v)?;
            p.sampling.user_sampling_config |= SAMPLING_CONFIG_XTC_PROBABILITY;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--xtc-threshold"], "N", format!("xtc threshold (default: {:.2}, 1.0 = disabled)", ctx.params.sampling.xtc_threshold as f64), |p, v| {
            p.sampling.xtc_threshold = parse_number_prefix(&v)?;
            p.sampling.user_sampling_config |= SAMPLING_CONFIG_XTC_THRESHOLD;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--typical", "--typical-p"], "N", format!("locally typical sampling, parameter p (default: {:.2}, 1.0 = disabled)", ctx.params.sampling.typ_p as f64), |p, v| {
            p.sampling.typ_p = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--repeat-last-n"], "N", format!("last n tokens to consider for penalize (default: {}, 0 = disabled)", ctx.params.sampling.penalty_last_n), |p, v| {
            if v < 0 {
                return Err(Error::new(format!("error: invalid repeat-last-n = {v}\n")));
            }
            p.sampling.penalty_last_n = v;
            p.sampling.n_prev = p.sampling.n_prev.max(v);
            p.sampling.user_sampling_config |= SAMPLING_CONFIG_PENALTY_LAST_N;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--repeat-penalty"], "N", format!("penalize repeat sequence of tokens (default: {:.2}, 1.0 = disabled)", ctx.params.sampling.penalty_repeat as f64), |p, v| {
            let penalty: f32 = parse_number_prefix(&v)?;
            if !penalty.is_finite() || penalty <= 0.0 || !(1.0 / penalty).is_finite() {
                return Err(Error::new("error: repeat-penalty must be finite and greater than 0\n"));
            }
            p.sampling.penalty_repeat = penalty;
            p.sampling.user_sampling_config |= SAMPLING_CONFIG_PENALTY_REPEAT;
            Ok(())
        })
        .set_sampling()
        .set_env("LLAMA_ARG_REPEAT_PENALTY"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--presence-penalty"], "N", format!("repeat alpha presence penalty (default: {:.2}, 0.0 = disabled)", ctx.params.sampling.penalty_present as f64), |p, v| {
            let penalty: f32 = parse_number_prefix(&v)?;
            if !penalty.is_finite() {
                return Err(Error::new("error: presence-penalty must be finite\n"));
            }
            p.sampling.penalty_present = penalty;
            Ok(())
        })
        .set_sampling()
        .set_env("LLAMA_ARG_PRESENCE_PENALTY"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--frequency-penalty"], "N", format!("repeat alpha frequency penalty (default: {:.2}, 0.0 = disabled)", ctx.params.sampling.penalty_freq as f64), |p, v| {
            let penalty: f32 = parse_number_prefix(&v)?;
            if !penalty.is_finite() {
                return Err(Error::new("error: frequency-penalty must be finite\n"));
            }
            p.sampling.penalty_freq = penalty;
            Ok(())
        })
        .set_sampling()
        .set_env("LLAMA_ARG_FREQUENCY_PENALTY"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--dry-multiplier"], "N", format!("set DRY sampling multiplier (default: {:.2}, 0.0 = disabled)", ctx.params.sampling.dry_multiplier as f64), |p, v| {
            p.sampling.dry_multiplier = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--dry-base"], "N", format!("set DRY sampling base value (default: {:.2})", ctx.params.sampling.dry_base as f64), |p, v| {
            let base: f32 = parse_number_prefix(&v)?;
            if base >= 1.0 {
                p.sampling.dry_base = base;
            }
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--dry-allowed-length"], "N", format!("set allowed length for DRY sampling (default: {})", ctx.params.sampling.dry_allowed_length), |p, v| {
            p.sampling.dry_allowed_length = v;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--dry-penalty-last-n"], "N", format!("set DRY penalty for the last n tokens (default: {}, 0 = disable)", ctx.params.sampling.dry_penalty_last_n), |p, v| {
            if v < 0 {
                return Err(Error::new(format!("error: invalid dry-penalty-last-n = {v}\n")));
            }
            p.sampling.dry_penalty_last_n = v;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--dry-sequence-breaker"], "STRING", format!("add sequence breaker for DRY sampling, clearing out default breakers ({}) in the process; use \"none\" to not use any sequence breakers\n", dry_breakers), |p, v| {
            static DEFAULTS_CLEARED: AtomicBool = AtomicBool::new(false);
            if !DEFAULTS_CLEARED.swap(true, AtomicOrdering::Relaxed) {
                p.sampling.dry_sequence_breakers.clear();
            }
            if v == "none" {
                p.sampling.dry_sequence_breakers.clear();
            } else {
                p.sampling.dry_sequence_breakers.push(v);
            }
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--adaptive-target"], "N", format!("adaptive-p: select tokens near this probability (valid range 0.0 to 1.0; negative = disabled) (default: {:.2})\n[(more info)](https://github.com/ggml-org/llama.cpp/pull/17927)", ctx.params.sampling.adaptive_target), |p, v| {
            p.sampling.adaptive_target = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--adaptive-decay"], "N", format!("adaptive-p: decay rate for target adaptation over time. lower values are more reactive, higher values are more stable.\n(valid range 0.0 to 0.99) (default: {:.2})", ctx.params.sampling.adaptive_decay), |p, v| {
            p.sampling.adaptive_decay = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--dynatemp-range"], "N", format!("dynamic temperature range (default: {:.2}, 0.0 = disabled)", ctx.params.sampling.dynatemp_range as f64), |p, v| {
            p.sampling.dynatemp_range = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--dynatemp-exp"], "N", format!("dynamic temperature exponent (default: {:.2})", ctx.params.sampling.dynatemp_exponent as f64), |p, v| {
            p.sampling.dynatemp_exponent = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--mirostat"], "N", format!("use Mirostat sampling.\nTop K, Nucleus and Locally Typical samplers are ignored if used.\n(default: {}, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", ctx.params.sampling.mirostat), |p, v| {
            p.sampling.mirostat = v;
            p.sampling.user_sampling_config |= SAMPLING_CONFIG_MIROSTAT;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--mirostat-lr"], "N", format!("Mirostat learning rate, parameter eta (default: {:.2})", ctx.params.sampling.mirostat_eta as f64), |p, v| {
            p.sampling.mirostat_eta = parse_number_prefix(&v)?;
            p.sampling.user_sampling_config |= SAMPLING_CONFIG_MIROSTAT_ETA;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--mirostat-ent"], "N", format!("Mirostat target entropy, parameter tau (default: {:.2})", ctx.params.sampling.mirostat_tau as f64), |p, v| {
            p.sampling.mirostat_tau = parse_number_prefix(&v)?;
            p.sampling.user_sampling_config |= SAMPLING_CONFIG_MIROSTAT_TAU;
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-l", "--logit-bias"], "TOKEN_ID(+/-)BIAS", "modifies the likelihood of token appearing in the completion,\ni.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\nor `--logit-bias 15043-1` to decrease likelihood of token ' Hello'", |p, v| {
            let pos = v.char_indices().find(|(_, c)| *c == '+' || *c == '-');
            let Some((pos, sign)) = pos else {
                return Err(Error::new("invalid input format"));
            };
            let token: i32 = parse_number_prefix(&v[..pos]).map_err(|_| Error::new("invalid input format"))?;
            let bias = parse_number_prefix::<f32>(&v[pos + sign.len_utf8()..])
                .map_err(|_| Error::new("invalid input format"))?;
            p.sampling.logit_bias.push(LogitBias {
                token,
                bias: if sign == '-' { -bias } else { bias },
            });
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--grammar"], "GRAMMAR", "BNF-like grammar to constrain generations (see samples in grammars/ dir)", |p, v| {
            p.sampling.grammar = CommonGrammar::new(GrammarType::User, v);
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--grammar-file"], "FNAME", "file to read grammar from", |p, v| {
            p.sampling.grammar = CommonGrammar::new(GrammarType::User, read_file(&v)?);
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-j", "--json-schema"], "SCHEMA", "JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON object", |p, v| {
            Json::parse(&v)?;
            // shortcut: json_schema_to_grammar is not ported; keep the raw schema
            p.sampling.grammar = CommonGrammar::new(GrammarType::OutputFormat, v);
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-jf", "--json-schema-file"], "FILE", "File containing a JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON object", |p, v| {
            let schema = read_file(&v)?;
            Json::parse(&schema)?;
            // shortcut: json_schema_to_grammar is not ported; keep the raw schema
            p.sampling.grammar = CommonGrammar::new(GrammarType::OutputFormat, schema);
            Ok(())
        })
        .set_sampling(),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["-bs", "--backend-sampling"], "enable backend sampling (experimental) (default: disabled)", |p| {
            p.sampling.backend_sampling = true;
            Ok(())
        })
        .set_sampling()
        .set_env("LLAMA_ARG_BACKEND_SAMPLING"),
    );

    //
    // model / runtime params
    //

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--pooling"], "{none,mean,cls,last,rank}", "pooling type for embeddings, use model default if unspecified", |p, v| {
            p.pooling_type = match v.as_str() {
                "none" => PoolingType::None,
                "mean" => PoolingType::Mean,
                "cls" => PoolingType::Cls,
                "last" => PoolingType::Last,
                "rank" => PoolingType::Rank,
                _ => return Err(Error::new("invalid value")),
            };
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Embedding, LlamaExample::Retrieval, LlamaExample::Server,
            LlamaExample::Debug,
        ])
        .set_env("LLAMA_ARG_POOLING"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--attention"], "{causal,non-causal}", "attention type for embeddings, use model default if unspecified", |p, v| {
            p.attention_type = match v.as_str() {
                "causal" => AttentionType::Causal,
                "non-causal" => AttentionType::NonCausal,
                _ => return Err(Error::new("invalid value")),
            };
            Ok(())
        })
        .set_examples(&[LlamaExample::Embedding]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--rope-scaling"], "{none,linear,yarn}", "RoPE frequency scaling method, defaults to linear unless specified by the model", |p, v| {
            p.rope_scaling_type = match v.as_str() {
                "none" => RopeScalingType::None,
                "linear" => RopeScalingType::Linear,
                "yarn" => RopeScalingType::Yarn,
                _ => return Err(Error::new("invalid value")),
            };
            Ok(())
        })
        .set_env("LLAMA_ARG_ROPE_SCALING_TYPE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--rope-scale"], "N", "RoPE context scaling factor, expands context by a factor of N", |p, v| {
            p.rope_freq_scale = 1.0 / parse_number_prefix::<f32>(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_ROPE_SCALE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--rope-freq-base"], "N", "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)", |p, v| {
            p.rope_freq_base = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_ROPE_FREQ_BASE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--rope-freq-scale"], "N", "RoPE frequency scaling factor, expands context by a factor of 1/N", |p, v| {
            p.rope_freq_scale = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_ROPE_FREQ_SCALE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--yarn-orig-ctx"], "N", format!("YaRN: original context size of model (default: {} = model training context size)", ctx.params.yarn_orig_ctx), |p, v| {
            p.yarn_orig_ctx = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_YARN_ORIG_CTX"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--yarn-ext-factor"], "N", format!("YaRN: extrapolation mix factor (default: {:.2}, 0.0 = full interpolation)", ctx.params.yarn_ext_factor), |p, v| {
            p.yarn_ext_factor = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_YARN_EXT_FACTOR"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--yarn-attn-factor"], "N", format!("YaRN: scale sqrt(t) or attention magnitude (default: {:.2})", ctx.params.yarn_attn_factor), |p, v| {
            p.yarn_attn_factor = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_YARN_ATTN_FACTOR"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--yarn-beta-slow"], "N", format!("YaRN: high correction dim or alpha (default: {:.2})", ctx.params.yarn_beta_slow), |p, v| {
            p.yarn_beta_slow = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_YARN_BETA_SLOW"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--yarn-beta-fast"], "N", format!("YaRN: low correction dim or beta (default: {:.2})", ctx.params.yarn_beta_fast), |p, v| {
            p.yarn_beta_fast = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_YARN_BETA_FAST"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-gan", "--grp-attn-n"], "N", format!("group-attention factor (default: {})", ctx.params.grp_attn_n), |p, v| {
            p.grp_attn_n = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_GRP_ATTN_N")
        .set_examples(&[LlamaExample::Completion, LlamaExample::Passkey]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-gaw", "--grp-attn-w"], "N", format!("group-attention width (default: {})", ctx.params.grp_attn_w), |p, v| {
            p.grp_attn_w = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_GRP_ATTN_W")
        .set_examples(&[LlamaExample::Completion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["-kvo", "--kv-offload"], &["-nkvo", "--no-kv-offload"], format!("whether to enable KV cache offloading (default: {})", if ctx.params.no_kv_offload { "disabled" } else { "enabled" }), |p, v| {
            p.no_kv_offload = !v;
            Ok(())
        })
        .set_env("LLAMA_ARG_KV_OFFLOAD"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--repack"], &["-nr", "--no-repack"], format!("whether to enable weight repacking (default: {})", if ctx.params.no_extra_bufts { "disabled" } else { "enabled" }), |p, v| {
            p.no_extra_bufts = !v;
            Ok(())
        })
        .set_env("LLAMA_ARG_REPACK"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--no-host"], "bypass host buffer allowing extra buffers to be used", |p| {
            p.no_host = true;
            Ok(())
        })
        .set_env("LLAMA_ARG_NO_HOST"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-ctk", "--cache-type-k"], "TYPE", format!("KV cache data type for K\nallowed values: {}\n(default: {})", get_all_kv_cache_types(), ctx.params.cache_type_k.name()), |p, v| {
            p.cache_type_k = kv_cache_type_from_str(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_CACHE_TYPE_K"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-ctv", "--cache-type-v"], "TYPE", format!("KV cache data type for V\nallowed values: {}\n(default: {})", get_all_kv_cache_types(), ctx.params.cache_type_v.name()), |p, v| {
            p.cache_type_v = kv_cache_type_from_str(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_CACHE_TYPE_V"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--hellaswag"], "compute HellaSwag score over random tasks from datafile supplied with -f", |p| {
            p.hellaswag = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Perplexity]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--hellaswag-tasks"], "N", format!("number of tasks to use when computing the HellaSwag score (default: {})", ctx.params.hellaswag_tasks), |p, v| {
            p.hellaswag_tasks = v.max(0) as usize;
            Ok(())
        })
        .set_examples(&[LlamaExample::Perplexity]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--winogrande"], "compute Winogrande score over random tasks from datafile supplied with -f", |p| {
            p.winogrande = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Perplexity]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--winogrande-tasks"], "N", format!("number of tasks to use when computing the Winogrande score (default: {})", ctx.params.winogrande_tasks), |p, v| {
            p.winogrande_tasks = v.max(0) as usize;
            Ok(())
        })
        .set_examples(&[LlamaExample::Perplexity]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--multiple-choice"], "compute multiple choice score over random tasks from datafile supplied with -f", |p| {
            p.multiple_choice = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Perplexity]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--multiple-choice-tasks"], "N", format!("number of tasks to use when computing the multiple choice score (default: {})", ctx.params.multiple_choice_tasks), |p, v| {
            p.multiple_choice_tasks = v.max(0) as usize;
            Ok(())
        })
        .set_examples(&[LlamaExample::Perplexity]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--kl-divergence"], "computes KL-divergence to logits provided via --kl-divergence-base", |p| {
            p.kl_divergence = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Perplexity]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--save-all-logits", "--kl-divergence-base"], "FNAME", "set logits file", |p, v| {
            p.logits_file = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Perplexity]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--ppl-stride"], "N", format!("stride for perplexity calculation (default: {})", ctx.params.ppl_stride), |p, v| {
            p.ppl_stride = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Perplexity]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--ppl-output-type"], "<0|1>", format!("output type for perplexity calculation (default: {})", ctx.params.ppl_output_type), |p, v| {
            p.ppl_output_type = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Perplexity]),
    );
    if ctx.ex == LlamaExample::Server {
        add_opt(
            &mut ctx.options,
            ctx.ex,
            CommonArg::int(&["-np", "--parallel"], "N", format!("number of server slots (default: {}, -1 = auto)", ctx.params.n_parallel), |p, v| {
                if v == 0 {
                    return Err(Error::new("error: invalid value for n_parallel\n"));
                }
                p.n_parallel = v;
                Ok(())
            })
            .set_env("LLAMA_ARG_N_PARALLEL")
            .set_examples(&[LlamaExample::Server]),
        );
    } else {
        add_opt(
            &mut ctx.options,
            ctx.ex,
            CommonArg::int(&["-np", "--parallel"], "N", format!("number of parallel sequences to decode (default: {})", ctx.params.n_parallel), |p, v| {
                p.n_parallel = v;
                Ok(())
            })
            .set_env("LLAMA_ARG_N_PARALLEL"),
        );
    }
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-ns", "--sequences"], "N", format!("number of sequences to decode (default: {})", ctx.params.n_sequences), |p, v| {
            p.n_sequences = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Parallel]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["-cb", "--cont-batching"], &["-nocb", "--no-cont-batching"], format!("whether to enable continuous batching (a.k.a dynamic batching) (default: {})", if ctx.params.cont_batching { "enabled" } else { "disabled" }), |p, v| {
            p.cont_batching = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_CONT_BATCHING"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-mm", "--mmproj"], "FILE", "path to a multimodal projector file. see tools/mtmd/README.md\nnote: if -hf is used, this argument can be omitted", |p, v| {
            p.mmproj.path = v;
            Ok(())
        })
        .set_examples(mmproj_examples)
        .set_env("LLAMA_ARG_MMPROJ"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-mmu", "--mmproj-url"], "URL", "URL to a multimodal projector file. see tools/mtmd/README.md", |p, v| {
            p.mmproj.url = v;
            Ok(())
        })
        .set_examples(mmproj_examples)
        .set_env("LLAMA_ARG_MMPROJ_URL"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--mmproj-auto"], &["--no-mmproj", "--no-mmproj-auto"], format!("whether to use multimodal projector file (if available), useful when using -hf (default: {})", if ctx.params.no_mmproj { "disabled" } else { "enabled" }), |p, v| {
            p.no_mmproj = !v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Mtmd, LlamaExample::Server, LlamaExample::Cli,
            LlamaExample::Download,
        ])
        .set_env("LLAMA_ARG_MMPROJ_AUTO"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--mmproj-offload"], &["--no-mmproj-offload"], format!("whether to enable GPU offloading for multimodal projector (default: {})", if ctx.params.mmproj_use_gpu { "enabled" } else { "disabled" }), |p, v| {
            p.mmproj_use_gpu = v;
            Ok(())
        })
        .set_examples(mmproj_examples)
        .set_env("LLAMA_ARG_MMPROJ_OFFLOAD"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-mmdev", "--mmproj-device"], "DEVICE", "device to use for multimodal projector (none = don't offload, default: follows --device)\nuse --list-devices to see a list of available devices", |p, v| {
            if v == "none" {
                p.mmproj_use_gpu = false;
                return Ok(());
            }
            let devices = parse_device_list(&v)?;
            if devices.len() > 1 {
                return Err(Error::new("only one device may be specified for mmproj"));
            }
            p.mmproj_use_gpu = true;
            Ok(())
        })
        .set_examples(mmproj_examples)
        .set_env("MTMD_BACKEND_DEVICE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--image", "--audio", "--video"], "FILE", "path to an image, audio, or video file. use with multimodal models, use comma-separated values for multiple files\n", |p, v| {
            p.image.extend(parse_csv_row(&v));
            Ok(())
        })
        .set_examples(&[LlamaExample::Mtmd, LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--image-min-tokens"], "N", "minimum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)", |p, v| {
            p.image_min_tokens = v;
            Ok(())
        })
        .set_examples(mmproj_examples)
        .set_env("LLAMA_ARG_IMAGE_MIN_TOKENS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--image-max-tokens"], "N", "maximum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)", |p, v| {
            p.image_max_tokens = v;
            Ok(())
        })
        .set_examples(mmproj_examples)
        .set_env("LLAMA_ARG_IMAGE_MAX_TOKENS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--mtmd-batch-max-tokens"], "N", format!("maximum number of image tokens per batch when encoding images (default: {})", ctx.params.mtmd_batch_max_tokens), |p, v| {
            p.mtmd_batch_max_tokens = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_MTMD_BATCH_MAX_TOKENS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--video-fps"], "N", format!("target video frame rate (default: {:.1})", ctx.params.video_fps), |p, v| {
            p.video_fps = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_examples(mmproj_examples)
        .set_env("LLAMA_ARG_VIDEO_FPS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--video-timestamp-interval"], "N", format!("interval in milliseconds between text timestamps (default: {})", ctx.params.video_timestamp_interval_ms), |p, v| {
            p.video_timestamp_interval_ms = v as i64;
            Ok(())
        })
        .set_examples(mmproj_examples)
        .set_env("LLAMA_ARG_VIDEO_TIMESTAMP_INTERVAL"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--video-ffmpeg-dir"], "DIR", "path to the directory containing ffmpeg and ffprobe (default: search in PATH)", |p, v| {
            p.video_ffmpeg_bin_dir = v;
            Ok(())
        })
        .set_examples(mmproj_examples)
        .set_env("LLAMA_ARG_VIDEO_FFMPEG_DIR"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--rpc"], "SERVERS", "comma-separated list of RPC servers (host:port)", |_, _| {
            // shortcut: no backend registry in the Rust port, so no devices are added
            Ok(())
        })
        .set_env("LLAMA_ARG_RPC"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-lm", "--load-mode"], "MODE", "model loading mode (default: auto)\n- auto: mmap, unless a device does not support it\n- none: no special loading mode\n- mmap: memory-map model (if mmap disabled, slower load but may reduce pageouts if not using mlock)\n- mlock: force system to keep model in RAM rather than swapping or compressing\n- mmap+mlock: mmap + force system to keep model in RAM rather than swapping or compressing\n- dio: use DirectIO if available\n", |p, v| {
            p.load_mode = match v.as_str() {
                "auto" => LoadMode::Auto,
                "none" => LoadMode::None,
                "mmap" => LoadMode::Mmap,
                "mlock" => LoadMode::Mlock,
                "mmap+mlock" => LoadMode::MmapMlock,
                "dio" => LoadMode::DirectIo,
                _ => return Err(Error::new("invalid value")),
            };
            Ok(())
        })
        .set_env("LLAMA_ARG_LOAD_MODE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-lzm", "--lazy-mode"], "MODE", "on-demand reading of certain tensors, for example per-layer embeddings (default: auto)\n- on: read the rows of such tensors from disk on demand instead of keeping them resident (requires mmap)\n- auto: on, but only for tensors larger than 4 GiB\n- off: always keep them resident", |p, v| {
            p.lazy_mode = match v.as_str() {
                "on" => LazyMode::On,
                "auto" => LazyMode::Auto,
                "off" => LazyMode::Off,
                _ => return Err(Error::new("invalid value")),
            };
            Ok(())
        })
        .set_env("LLAMA_ARG_LAZY_MODE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--numa"], "TYPE", "attempt optimizations that help on some NUMA systems\n- distribute: spread execution evenly over all nodes\n- isolate: only spawn threads on CPUs on the node that execution started on\n- numactl: use the CPU map provided by numactl\nif run without this previously, it is recommended to drop the system page cache before using this\nsee https://github.com/ggml-org/llama.cpp/issues/1437", |p, v| {
            p.numa = match v.as_str() {
                "distribute" | "" => 0,
                "isolate" => 1,
                "numactl" => 2,
                _ => return Err(Error::new("invalid value")),
            };
            Ok(())
        })
        .set_env("LLAMA_ARG_NUMA"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-dev", "--device"], "<dev1,dev2,..>", "comma-separated list of devices to use for offloading (none = don't offload)\nuse --list-devices to see a list of available devices", |p, v| {
            p.devices = parse_device_list(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_DEVICE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--list-devices"], "print list of available devices and exit", |_| {
            println!("Available devices:\n  (none)");
            std::process::exit(0);
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-ot", "--override-tensor"], "<tensor name pattern>=<buffer type>,...", "override tensor buffer type", |p, v| {
            parse_tensor_buffer_overrides(&v, &mut p.tensor_buft_overrides)
        })
        .set_env("LLAMA_ARG_OVERRIDE_TENSOR"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["-cmoe", "--cpu-moe"], "keep all Mixture of Experts (MoE) weights in the CPU", |p| {
            p.tensor_buft_overrides.push(llm_ffn_exps_cpu_override());
            Ok(())
        })
        .set_env("LLAMA_ARG_CPU_MOE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-ncmoe", "--n-cpu-moe"], "N", "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU", |p, v| {
            if v < 0 {
                return Err(Error::new("invalid value"));
            }
            llm_add_n_cpu_ffn_overrides(v, LLM_FFN_EXPS_REGEX, &mut p.tensor_buft_overrides);
            Ok(())
        })
        .set_env("LLAMA_ARG_N_CPU_MOE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--moe-cache-mib"], "N", "GPU cache size in MiB for the MoE experts kept in the CPU. with multiple GPUs, it is split among them like the layers (--tensor-split) (default: 0, disabled)", |p, v| {
            if v < 0 {
                return Err(Error::new("invalid value"));
            }
            p.moe_cache_size = (v as usize) * 1024 * 1024;
            Ok(())
        })
        .set_env("LLAMA_ARG_MOE_CACHE_MIB"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--moe-cache-cpu-share"], "F", format!("share of the MoE cache misses that the CPU computes from the host experts instead of uploading them to the GPU once the cache is full, from 0 to 1 (default: {:.2})", ctx.params.moe_cache_cpu_share), |p, v| {
            let share: f32 = parse_number_prefix(&v)?;
            if !(0.0..=1.0).contains(&share) {
                return Err(Error::new("invalid value"));
            }
            p.moe_cache_cpu_share = share;
            Ok(())
        })
        .set_env("LLAMA_ARG_MOE_CACHE_CPU_SHARE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-ncffn", "--n-cpu-ffn"], "N", "keep the dense FFN weights of the first N layers in the CPU\n(dense models; for MoE expert weights use --n-cpu-moe)", |p, v| {
            if v < 0 {
                return Err(Error::new("invalid value"));
            }
            llm_add_n_cpu_ffn_overrides(v, LLM_FFN_DENSE_REGEX, &mut p.tensor_buft_overrides);
            Ok(())
        })
        .set_env("LLAMA_ARG_N_CPU_FFN"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-ngl", "--gpu-layers", "--n-gpu-layers"], "N", format!("max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: {})", if ctx.params.n_gpu_layers == -1 { "auto" } else { "all" }), |p, v| {
            p.n_gpu_layers = match v.as_str() {
                "auto" => -1,
                "all" => -2,
                other => parse_number_prefix(other)?,
            };
            Ok(())
        })
        .set_env("LLAMA_ARG_N_GPU_LAYERS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-sm", "--split-mode"], "{none,layer,row,tensor}", "how to split the model across multiple GPUs, one of:\n- none: use one GPU only\n- layer (default): split layers and KV across GPUs (pipelined)\n- row: split weight across GPUs by rows (parallelized)\n- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)", |p, v| {
            p.split_mode = match v.as_str() {
                "none" => SplitMode::None,
                "layer" => SplitMode::Layer,
                "row" => SplitMode::Row,
                "tensor" => SplitMode::Tensor,
                _ => return Err(Error::new("invalid value")),
            };
            Ok(())
        })
        .set_env("LLAMA_ARG_SPLIT_MODE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-ts", "--tensor-split"], "N0,N1,N2,...", "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1", |p, v| {
            let parts: Vec<&str> = v.split(|c| c == ',' || c == '/').filter(|s| !s.is_empty()).collect();
            if parts.len() > LLAMA_MAX_DEVICES {
                return Err(Error::new(format!("got {} input configs, but system only has {} devices", parts.len(), LLAMA_MAX_DEVICES)));
            }
            for i in 0..LLAMA_MAX_DEVICES {
                p.tensor_split[i] = if i < parts.len() {
                    parse_number_prefix(parts[i])?
                } else {
                    0.0
                };
            }
            Ok(())
        })
        .set_env("LLAMA_ARG_TENSOR_SPLIT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-mg", "--main-gpu"], "INDEX", format!("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: {})", ctx.params.main_gpu), |p, v| {
            p.main_gpu = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_MAIN_GPU"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-fit", "--fit"], "[on|off]", format!("whether to adjust unset arguments to fit in device memory ('on' or 'off', default: '{}')", if ctx.params.fit_params { "on" } else { "off" }), |p, v| {
            if is_truthy(&v) {
                p.fit_params = true;
            } else if is_falsey(&v) {
                p.fit_params = false;
            } else {
                return Err(Error::new(format!("error: unknown value for --fit: '{v}'\n")));
            }
            Ok(())
        })
        .set_env("LLAMA_ARG_FIT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-fitp", "--fit-print"], "[on|off]", format!("print the estimated required memory ('on' or 'off', default: '{}')", if ctx.params.fit_params_print { "on" } else { "off" }), |p, v| {
            if is_truthy(&v) {
                p.fit_params_print = true;
            } else if is_falsey(&v) {
                p.fit_params_print = false;
            } else {
                return Err(Error::new(format!("error: unknown value for --fit-print: '{v}'\n")));
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::FitParams])
        .set_env("LLAMA_ARG_FIT_ESTIMATE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-fitt", "--fit-target"], "MiB0,MiB1,MiB2,...", format!("target margin per device for --fit, comma-separated list of values, single value is broadcast across all devices, default: {}", ctx.params.fit_params_target[0] / (1024 * 1024)), |p, v| {
            let parts: Vec<&str> = v.split(|c| c == ',' || c == '/').filter(|s| !s.is_empty()).collect();
            if parts.len() > LLAMA_MAX_DEVICES {
                return Err(Error::new(format!("got {} input configs, but system only has {} devices", parts.len(), LLAMA_MAX_DEVICES)));
            }
            let mut values = Vec::with_capacity(parts.len());
            for part in parts {
                let mib: u64 = parse_number_prefix(part)?;
                values.push((mib.saturating_mul(1024 * 1024)) as usize);
            }
            if values.len() == 1 {
                p.fit_params_target.fill(values[0]);
            } else {
                for (dst, src) in p.fit_params_target.iter_mut().zip(values) {
                    *dst = src;
                }
            }
            Ok(())
        })
        .set_env("LLAMA_ARG_FIT_TARGET"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-fitc", "--fit-ctx"], "N", format!("minimum ctx size that can be set by --fit option, default: {}", ctx.params.fit_params_min_ctx as u32), |p, v| {
            p.fit_params_min_ctx = v;
            Ok(())
        })
        .set_env("LLAMA_ARG_FIT_CTX"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--check-tensors"], format!("check model tensor data for invalid values (default: {})", ctx.params.check_tensors), |p| {
            p.check_tensors = true;
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--override-kv"], "KEY=TYPE:VALUE,...", "advanced option to override model metadata by key. to specify multiple overrides, either use comma-separated values.\ntypes: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false,tokenizer.ggml.add_eos_token=bool:false", |p, v| {
            for item in parse_csv_row(&v) {
                if !string_parse_kv_override(&item, &mut p.kv_overrides) {
                    return Err(Error::new(format!("error: Invalid type for KV override: {item}\n")));
                }
            }
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--op-offload"], &["--no-op-offload"], format!("whether to offload host tensor operations to device (default: {})", if ctx.params.no_op_offload { "false" } else { "true" }), |p, v| {
            p.no_op_offload = !v;
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--lora"], "FNAME", "path to LoRA adapter (use comma-separated values to load multiple adapters)", |p, v| {
            for item in parse_csv_row(&v) {
                p.lora_adapters.push(crate::common::AdapterLoraInfo {
                    path: item,
                    scale: 1.0,
                    task_name: String::new(),
                    prompt_prefix: String::new(),
                });
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Common, LlamaExample::ExportLora]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--lora-scaled"], "FNAME:SCALE,...", "path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...)\nnote: use comma-separated values", |p, v| {
            for item in parse_csv_row(&v) {
                let parts: Vec<&str> = item.split(':').collect();
                if parts.len() != 2 {
                    return Err(Error::new("lora-scaled format: FNAME:SCALE"));
                }
                p.lora_adapters.push(crate::common::AdapterLoraInfo {
                    path: parts[0].to_owned(),
                    scale: parse_number_prefix(parts[1])?,
                    task_name: String::new(),
                    prompt_prefix: String::new(),
                });
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Common, LlamaExample::ExportLora]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--control-vector"], "FNAME", "add a control vector\nnote: use comma-separated values to add multiple control vectors", |p, v| {
            for item in parse_csv_row(&v) {
                p.control_vectors.push(crate::common::ControlVectorLoadInfo {
                    strength: 1.0,
                    fname: item,
                });
            }
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--control-vector-scaled"], "FNAME:SCALE,...", "add a control vector with user defined scaling SCALE\nnote: use comma-separated values (format: FNAME:SCALE,...)", |p, v| {
            for item in parse_csv_row(&v) {
                let parts: Vec<&str> = item.split(':').collect();
                if parts.len() != 2 {
                    return Err(Error::new("control-vector-scaled format: FNAME:SCALE"));
                }
                p.control_vectors.push(crate::common::ControlVectorLoadInfo {
                    strength: parse_number_prefix(parts[1])?,
                    fname: parts[0].to_owned(),
                });
            }
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::str2(&["--control-vector-layer-range"], "START", "END", "layer range to apply the control vector(s) to, start and end inclusive", |p, start, end| {
            p.control_vector_layer_start = parse_number_prefix(&start)?;
            p.control_vector_layer_end = parse_number_prefix(&end)?;
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-a", "--alias"], "STRING", "set model name aliases, comma-separated (to be used by API)", |p, v| {
            for alias in string_split_char(&v, ',') {
                let alias = string_strip(&alias);
                if !alias.is_empty() {
                    p.model_alias.insert(alias.to_owned());
                }
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_ALIAS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--tags"], "STRING", "set model tags, comma-separated (informational, not used for routing)", |p, v| {
            for tag in string_split_char(&v, ',') {
                let tag = string_strip(&tag);
                if !tag.is_empty() {
                    p.model_tags.insert(tag.to_owned());
                }
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_TAGS"),
    );

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-m", "--model"], "FNAME", if ctx.ex == LlamaExample::ExportLora {
            "model path from which to load base model".to_owned()
        } else {
            "model path to load".to_owned()
        }, |p, v| {
            p.model.path = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Common, LlamaExample::ExportLora, LlamaExample::Download,
            LlamaExample::Tokenize,
        ])
        .set_env("LLAMA_ARG_MODEL"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--ple", "--shard"], "FNAME,...", "extra GGUF shards for the model, comma separated; may be given more than once\npaths may live anywhere, including on different drives; --ple is an alias for --shard", |p, v| {
            p.model.shards.extend(parse_csv_row(&v));
            Ok(())
        })
        .set_examples(&[LlamaExample::Common])
        .set_env("LLAMA_ARG_SHARD"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-mu", "--model-url"], "MODEL_URL", "model download url (default: unused)", |p, v| {
            p.model.url = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Common, LlamaExample::Download, LlamaExample::Tokenize,
        ])
        .set_env("LLAMA_ARG_MODEL_URL"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-dr", "--docker-repo"], "[<repo>/]<model>[:quant]", "Docker Hub model repository. repo is optional, default to ai/. quant is optional, default to :latest.\nexample: gemma3\n(default: unused)", |p, v| {
            p.model.docker_repo = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Common, LlamaExample::Download, LlamaExample::Tokenize,
        ])
        .set_env("LLAMA_ARG_DOCKER_REPO"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-hf", "-hfr", "--hf-repo"], "<user>/<model>[:quant]", "Hugging Face model repository; quant is optional, case-insensitive, default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist.\nmmproj is also downloaded automatically if available. to disable, add --no-mmproj\nexample: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M\n(default: unused)", |p, v| {
            p.model.hf_repo = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Common, LlamaExample::Download, LlamaExample::Tokenize,
        ])
        .set_env("LLAMA_ARG_HF_REPO"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-hff", "--hf-file"], "FILE", "Hugging Face model file. If specified, it will override the quant in --hf-repo (default: unused)", |p, v| {
            p.model.hf_file = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Common, LlamaExample::Download, LlamaExample::Tokenize,
        ])
        .set_env("LLAMA_ARG_HF_FILE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-hft", "--hf-token"], "TOKEN", "Hugging Face access token (default: value from HF_TOKEN environment variable)", |p, v| {
            p.hf_token = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Common, LlamaExample::Download, LlamaExample::Tokenize,
        ])
        .set_env("HF_TOKEN"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--mtp"], "also download the multi-token prediction (MTP) head, if available (default: unused)", |p| {
            p.speculative.types.push(SpeculativeType::DraftMtp);
            Ok(())
        })
        .set_examples(&[LlamaExample::Download]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--dflash"], "also download the DFlash sidecar, if available (default: unused)", |p| {
            p.speculative.types.push(SpeculativeType::DraftDflash);
            Ok(())
        })
        .set_examples(&[LlamaExample::Download]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--eagle3"], "also download the Eagle3 sidecar, if available (default: unused)", |p| {
            p.speculative.types.push(SpeculativeType::DraftEagle3);
            Ok(())
        })
        .set_examples(&[LlamaExample::Download]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--context-file"], "FNAME", "file to load context from (use comma-separated values to specify multiple files)", |p, v| {
            for item in parse_csv_row(&v) {
                if fs::metadata(&item).is_err() {
                    return Err(Error::new(format!("error: failed to open file '{item}'\n")));
                }
                p.context_files.push(item);
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Retrieval]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--chunk-size"], "N", format!("minimum length of embedded text chunks (default: {})", ctx.params.chunk_size), |p, v| {
            p.chunk_size = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Retrieval]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--chunk-separator"], "STRING", format!("separator between chunks (default: '{}')", ctx.params.chunk_separator), |p, v| {
            p.chunk_separator = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Retrieval]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--junk"], "N", format!("number of times to repeat the junk text (default: {})", ctx.params.n_junk), |p, v| {
            p.n_junk = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Passkey, LlamaExample::Parallel]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--pos"], "N", format!("position of the passkey in the junk text (default: {})", ctx.params.i_pos), |p, v| {
            p.i_pos = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Passkey]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-o", "--output", "--output-file"], "FNAME", format!("output file (default: '{}')", ctx.params.out_file), |p, v| {
            p.out_file = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Imatrix, LlamaExample::CvectorGenerator, LlamaExample::ExportLora,
            LlamaExample::Tts, LlamaExample::Finetune, LlamaExample::Results,
            LlamaExample::ExportGraphOps, LlamaExample::Cli,
        ]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-ofreq", "--output-frequency"], "N", format!("output the imatrix every N iterations (default: {})", ctx.params.n_out_freq), |p, v| {
            p.n_out_freq = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Imatrix]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--output-format"], "{gguf,dat}", format!("output format for imatrix file (default: {})", if ctx.params.imat_dat > 0 { "dat" } else { "gguf" }), |p, v| {
            p.imat_dat = match v.as_str() {
                "gguf" => -1,
                "dat" => 1,
                _ => return Err(Error::new("invalid output format")),
            };
            Ok(())
        })
        .set_examples(&[LlamaExample::Imatrix]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--save-frequency"], "N", format!("save an imatrix copy every N iterations (default: {})", ctx.params.n_save_freq), |p, v| {
            p.n_save_freq = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Imatrix]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--process-output"], format!("collect data for the output tensor (default: {})", ctx.params.process_output), |p| {
            p.process_output = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Imatrix]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--nextn"], format!("collect data for MTP/NextN layers (default: {})", ctx.params.load_mtp), |p| {
            p.load_mtp = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Imatrix]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--ppl"], &["--no-ppl"], format!("whether to compute perplexity (default: {})", ctx.params.compute_ppl), |p, v| {
            p.compute_ppl = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Imatrix]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--chunk", "--from-chunk"], "N", format!("start processing the input from chunk N (default: {})", ctx.params.i_chunk), |p, v| {
            p.i_chunk = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Imatrix]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--show-statistics"], format!("show imatrix statistics and then exit (default: {})", ctx.params.show_statistics), |p| {
            p.show_statistics = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Imatrix]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--parse-special"], format!("parse special tokens (chat, tool, etc) (default: {})", ctx.params.parse_special), |p| {
            p.parse_special = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Imatrix]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--ids"], format!("only print the token IDs, in a Python-parseable list form like [1, 2, 3] (default: {})", ctx.params.tokenize_ids), |p| {
            p.tokenize_ids = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Tokenize]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--stdin"], format!("read the prompt from stdin (takes precedence over -f/--file and -p/--prompt) (default: {})", ctx.params.tokenize_stdin), |p| {
            p.tokenize_stdin = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Tokenize]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--no-bos"], format!("do not add a BOS token to the prompt, even if the model normally uses one (default: {})", ctx.params.tokenize_no_bos), |p| {
            p.tokenize_no_bos = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Tokenize]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--no-parse-special"], format!("do not parse special tokens (chat, tool, etc) (default: {})", !ctx.params.parse_special), |p| {
            p.parse_special = false;
            Ok(())
        })
        .set_examples(&[LlamaExample::Tokenize]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--show-count"], format!("print the total number of tokens (default: {})", ctx.params.tokenize_show_count), |p| {
            p.tokenize_show_count = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Tokenize]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["-pps"], format!("is the prompt shared across parallel sequences (default: {})", ctx.params.is_pp_shared), |p| {
            p.is_pp_shared = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Bench, LlamaExample::Parallel]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["-tgs"], format!("is the text generation separated across the different sequences (default: {})", ctx.params.is_tg_separate), |p| {
            p.is_tg_separate = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Bench, LlamaExample::Parallel]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-npp"], "n0,n1,...", "number of prompt tokens", |p, v| {
            p.n_pp.extend(string_split_to::<i32>(&v, ',')?);
            Ok(())
        })
        .set_examples(&[LlamaExample::Bench]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-ntg"], "n0,n1,...", "number of text generation tokens", |p, v| {
            p.n_tg.extend(string_split_to::<i32>(&v, ',')?);
            Ok(())
        })
        .set_examples(&[LlamaExample::Bench]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-npl"], "n0,n1,...", "number of parallel prompts", |p, v| {
            p.n_pl.extend(string_split_to::<i32>(&v, ',')?);
            Ok(())
        })
        .set_examples(&[LlamaExample::Bench]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--embd-normalize"], "N", format!("normalisation for embeddings (default: {}) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)", ctx.params.embd_normalize), |p, v| {
            p.embd_normalize = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Embedding, LlamaExample::Server, LlamaExample::Debug,
        ]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--embd-output-format"], "FORMAT", "empty = default, \"array\" = [[],[]...], \"json\" = openai style, \"json+\" = same \"json\" + cosine similarity matrix, \"raw\" = plain whitespace-delimited output (one embedding per line)", |p, v| {
            p.embd_out = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Embedding]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--embd-separator"], "STRING", "separator of embeddings (default \\n) for example \"<#sep#>\"", |p, v| {
            p.embd_sep = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Embedding]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--cls-separator"], "STRING", "separator of classification sequences (default \\t) for example \"<#seq#>\"", |p, v| {
            p.cls_sep = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Embedding]),
    );

    //
    // server params
    //

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--host"], "HOST", format!("IP addresses to listen on, comma-separated, or UNIX socket paths ending in .sock; with multiple TCP addresses, :: binds IPv6 only; overlapping addresses result in undefined behavior (default: {})", ctx.params.hostnames[0]), |p, v| {
            p.hostnames.clear();
            for host in parse_csv_row(&v) {
                let host = string_strip(&host);
                if !host.is_empty() {
                    p.hostnames.push(host.to_owned());
                }
            }
            if p.hostnames.is_empty() {
                return Err(Error::new("--host requires at least one address"));
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_HOST"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--port"], "PORT", format!("port to listen (default: {})", ctx.params.port), |p, v| {
            p.port = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_PORT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--reuse-port"], format!("allow multiple sockets to bind to the same port (default: {})", if ctx.params.reuse_port { "enabled" } else { "disabled" }), |p| {
            p.reuse_port = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_REUSE_PORT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--path"], "PATH", format!("path to serve static files from (default: {})", ctx.params.public_path), |p, v| {
            p.public_path = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_STATIC_PATH"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--cors-origins"], "ORIGINS", format!("comma-separated list of allowed origins for CORS (default: {})\nif set to special value 'localhost', only allow same-origin requests (e.g. the built-in UI)", ctx.params.cors_origins), |p, v| {
            p.cors_origins = v;
            p.cors_origins_explicit = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_CORS_ORIGINS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--cors-methods"], "METHODS", format!("comma-separated list of allowed methods for CORS (default: {})", ctx.params.cors_methods), |p, v| {
            p.cors_methods = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_CORS_METHODS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--cors-headers"], "HEADERS", format!("comma-separated list of allowed headers for CORS (default: {})", ctx.params.cors_headers), |p, v| {
            p.cors_headers = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_CORS_HEADERS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--cors-credentials"], &["--no-cors-credentials"], format!("whether to allow credentials for CORS (default: {})\nnote: if this is enabled and --cors-origins is set to * (default), the Origin header will be echoed back, and credentials will always be allowed", if ctx.params.cors_credentials { "enabled" } else { "disabled" }), |p, v| {
            p.cors_credentials = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_CORS_CREDENTIALS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--api-prefix"], "PREFIX", format!("prefix path the server serves from, without the trailing slash (default: {})", ctx.params.api_prefix), |p, v| {
            p.api_prefix = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_API_PREFIX"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--ui-config", "--webui-config"], "JSON", "JSON that provides default UI settings (overrides UI defaults)", |p, v| {
            p.ui_config_json = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_UI_CONFIG"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--ui-config-file", "--webui-config-file"], "PATH", "JSON file that provides default UI settings (overrides UI defaults)", |p, v| {
            p.ui_config_json = read_file(&v)?;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_UI_CONFIG_FILE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--ui-mcp-proxy", "--webui-mcp-proxy"], &["--no-ui-mcp-proxy", "--no-webui-mcp-proxy"], "experimental: whether to enable MCP CORS proxy, for Llama Chat to reach public MCP servers without CORS headers; local and private addresses are refused unless --ui-mcp-proxy-allow-private-urls is set - do not enable in untrusted environments (default: disabled)\nnote: for security reasons, this will limit --cors-origins to localhost by default", |p, v| {
            p.ui_mcp_proxy = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_UI_MCP_PROXY"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--ui-mcp-proxy-allow-private-urls"], "allow the MCP CORS proxy to reach local or private addresses (e.g. MCP servers on localhost or LAN) - anyone who can reach the proxy can then reach your local network (default: disabled, only public addresses)\nnote: only has effect with --ui-mcp-proxy", |p| {
            p.ui_mcp_proxy_allow_private_urls = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_UI_MCP_PROXY_ALLOW_PRIVATE_URLS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--tools"], "TOOL1,TOOL2,...", "experimental: whether to enable built-in tools for AI agents - do not enable in untrusted environments (default: no tools)\nspecify \"all\" to enable all tools\navailable tools: read_file, file_glob_search, grep_search, exec_shell_command, write_file, edit_file, get_info\nnote: for security reasons, this will limit --cors-origins to localhost by default", |p, v| {
            p.server_tools = parse_csv_row(&v);
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_TOOLS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--tools-runtime"], "OPTION", "experimental: run tools in a separate runtime environment (default: none, use host environment)\navailable options:\n  'docker:<image>', 'podman:<image>': spin up a new container and reuse it for all invocations, clean up on server exit\n  'docker-container:<id>', 'podman-container:<id>': use an existing container by ID, won't stop on server exit\n  'ssh:<target>': run tools on a remote POSIX host over SSH, key-based auth and a trusted host key are required\n", |p, v| {
            p.server_tools_runtime = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_TOOLS_RUNTIME"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--allow-unauthenticated-tools"], "allow server tools, MCP servers or MCP proxy on a non-loopback address without an API key - only for trusted networks (default: disabled)", |p| {
            p.server_tools_allow_unauth = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_ALLOW_UNAUTHENTICATED_TOOLS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--mcp-servers-config"], "PATH", "experimental: path to JSON file with MCP server definitions (Cursor-compatible format) - do not enable in untrusted environments (default: none)\nnote: for security reasons, this will limit --cors-origins to localhost by default", |p, v| {
            p.mcp_servers_config = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_MCP_SERVERS_CONFIG"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--mcp-servers-json"], "JSON", "experimental: inline JSON with MCP server definitions (Cursor-compatible format) - do not enable in untrusted environments (default: none)\nnote: for security reasons, this will limit --cors-origins to localhost by default", |p, v| {
            p.mcp_servers_json = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_MCP_SERVERS_JSON"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["-ag", "--agent"], &["-no-ag", "--no-agent"], "whether to enable CORS proxy and all built-in tools - do not enable in untrusted environments (default: disabled)\nnote: for security reasons, this will limit --cors-origins to localhost by default", |p, v| {
            if v {
                p.server_tools = vec!["all".to_owned()];
                p.ui_mcp_proxy = true;
            } else {
                p.server_tools.clear();
                p.ui_mcp_proxy = false;
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_AGENT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--ui-agent"], "enable agent mode in Llama Chat with nothing enabled on the server: the tools that run in the browser and MCP servers the browser reaches itself, no server tools and no MCP proxy (default: disabled)\nnote: not needed with --agent, --tools or --ui-mcp-proxy, they enable agent mode in Llama Chat too", |p| {
            p.ui_agent = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_UI_AGENT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--ui", "--webui"], &["--no-ui", "--no-webui"], format!("whether to enable Llama Chat, the built-in chat app (default: {})", if ctx.params.ui { "enabled" } else { "disabled" }), |p, v| {
            p.ui = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_UI"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--connect"], format!("(experimental) open a peer-to-peer tunnel for server using llama-connect (default: {})", if ctx.params.server_connect { "enabled" } else { "disabled" }), |p| {
            p.server_connect = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_CONNECT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--connect-code"], "CODE", "manually specify a 40-character code for --connect (default: generate a new one for each run)", |p, v| {
            let val = v.replace('-', "").replace(' ', "");
            if val.len() != 40 {
                return Err(Error::new(format!("error: invalid connect code '{v}', must be 40 characters\n")));
            }
            p.server_connect_code = val;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_CONNECT_CODE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--embedding", "--embeddings"], format!("restrict to only support embedding use case; use only with dedicated embedding models (default: {})", if ctx.params.embedding { "enabled" } else { "disabled" }), |p| {
            p.embedding = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server, LlamaExample::Debug])
        .set_env("LLAMA_ARG_EMBEDDINGS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--rerank", "--reranking"], "enable reranking endpoint on server (default: disabled)", |p| {
            p.embedding = true;
            p.pooling_type = PoolingType::Rank;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_RERANKING"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--api-key"], "KEY", "API key to use for authentication, multiple keys can be provided as a comma-separated list (default: none)", |p, v| {
            for key in parse_csv_row(&v) {
                if !key.is_empty() {
                    p.api_keys.push(key);
                }
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_API_KEY"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--api-key-file"], "FNAME", "path to file containing API keys, one per line; lines starting with a hash are treated as comments (default: none)", |p, v| {
            let text = read_file(&v)?;
            for key in text.lines() {
                if !key.is_empty() && !key.starts_with('#') {
                    p.api_keys.push(key.to_owned());
                }
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_API_KEY_FILE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--ssl-key-file"], "FNAME", "path to file a PEM-encoded SSL private key", |p, v| {
            p.ssl_file_key = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_SSL_KEY_FILE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--ssl-cert-file"], "FNAME", "path to file a PEM-encoded SSL certificate", |p, v| {
            p.ssl_file_cert = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_SSL_CERT_FILE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--chat-template-kwargs"], "STRING", "sets additional params for the json template parser, must be a valid json object string, e.g. '{\"key1\":\"value1\",\"key2\":\"value2\"}'", |p, v| {
            let parsed = Json::parse(&v)?;
            for item in parsed.items() {
                if item.key() == "enable_thinking" {
                    eprintln!("warning: Setting 'enable_thinking' via --chat-template-kwargs is deprecated. Use --reasoning on / --reasoning off instead.");
                }
                if item.key() == "preserve_reasoning" {
                    eprintln!("warning: Setting 'preserve_reasoning' via --chat-template-kwargs is deprecated. Use --reasoning-preserve / --no-reasoning-preserve instead.");
                }
                p.default_template_kwargs.insert(item.key().to_owned(), item.value().dump(-1)?);
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Server, LlamaExample::Cli])
        .set_env("LLAMA_ARG_CHAT_TEMPLATE_KWARGS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-to", "--timeout"], "N", format!("server read/write timeout in seconds (default: {})", ctx.params.timeout_read), |p, v| {
            p.timeout_read = v;
            p.timeout_write = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_TIMEOUT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--sse-ping-interval"], "N", format!("server SSE ping interval in seconds (-1 = disabled, default: {})", ctx.params.sse_ping_interval), |p, v| {
            p.sse_ping_interval = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_SSE_PING_INTERVAL"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--threads-http"], "N", format!("number of threads used to process HTTP requests (default: {})", ctx.params.n_threads_http), |p, v| {
            p.n_threads_http = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_THREADS_HTTP"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--cache-prompt"], &["--no-cache-prompt"], format!("whether to enable prompt caching (default: {})", if ctx.params.cache_prompt { "enabled" } else { "disabled" }), |p, v| {
            p.cache_prompt = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_CACHE_PROMPT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--cache-reuse"], "N", format!("min chunk size to attempt reusing from the cache via KV shifting, requires prompt caching to be enabled (default: {})\n[(card)](https://ggml.ai/f0.png)", ctx.params.n_cache_reuse), |p, v| {
            p.n_cache_reuse = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_CACHE_REUSE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--metrics"], format!("enable prometheus compatible metrics endpoint (default: {})", ctx.params.endpoint_metrics), |p| {
            p.endpoint_metrics = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_ENDPOINT_METRICS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--props"], format!("enable changing global properties via POST /props (default: {})", ctx.params.endpoint_props), |p| {
            p.endpoint_props = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_ENDPOINT_PROPS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--slots"], &["--no-slots"], format!("expose slots monitoring endpoint (default: {})", if ctx.params.endpoint_slots { "enabled" } else { "disabled" }), |p, v| {
            p.endpoint_slots = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_ENDPOINT_SLOTS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--slot-save-path"], "PATH", "path to save slot kv cache (default: disabled)", |p, v| {
            p.slot_save_path = v.clone();
            if !fs_is_directory(&p.slot_save_path) {
                return Err(Error::new(format!("not a directory: {v}")));
            }
            if !p.slot_save_path.ends_with(std::path::MAIN_SEPARATOR) {
                p.slot_save_path.push(std::path::MAIN_SEPARATOR);
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_SLOT_SAVE_PATH"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--media-path"], "PATH", "directory for loading local media files; files can be accessed via file:// URLs using relative paths (default: disabled)", |p, v| {
            p.media_path = v.clone();
            if !fs_is_directory(&p.media_path) {
                return Err(Error::new(format!("not a directory: {v}")));
            }
            if !p.media_path.ends_with(std::path::MAIN_SEPARATOR) {
                p.media_path.push(std::path::MAIN_SEPARATOR);
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--media-allow-private-urls"], "allow http(s) media URLs (e.g. image_url) that point to local or private addresses (default: disabled, only public addresses)", |p| {
            p.media_allow_private_urls = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_MEDIA_ALLOW_PRIVATE_URLS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--models-dir"], "PATH", "directory containing models for the router server (default: disabled)", |p, v| {
            p.models_dir = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_MODELS_DIR"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--models-preset"], "PATH", "path to INI file containing model presets for the router server (default: disabled)", |p, v| {
            p.models_preset = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_MODELS_PRESET"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--models-max"], "N", format!("for router server, maximum number of models to load simultaneously (default: {}, 0 = unlimited)", ctx.params.models_max), |p, v| {
            p.models_max = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_MODELS_MAX"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--models-memory-margin"], "N", format!("for router server, MiB of device memory to keep free per device, unload LRU models until a new one fits (default: {}, 0 = disabled)", ctx.params.models_memory_margin), |p, v| {
            p.models_memory_margin = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_MODELS_MEMORY_MARGIN"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--models-autoload"], &["--no-models-autoload"], format!("for router server, whether to automatically load models (default: {})", if ctx.params.models_autoload { "enabled" } else { "disabled" }), |p, v| {
            p.models_autoload = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_MODELS_AUTOLOAD"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--jinja"], &["--no-jinja"], format!("whether to use jinja template engine for chat (default: {})", if ctx.params.use_jinja { "enabled" } else { "disabled" }), |p, v| {
            p.use_jinja = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Server, LlamaExample::Completion, LlamaExample::Cli,
            LlamaExample::Mtmd,
        ])
        .set_env("LLAMA_ARG_JINJA"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--reasoning-format"], "FORMAT", "controls whether thought tags are allowed and/or extracted from the response, and in which format they're returned; one of:\n- none: leaves thoughts unparsed in `message.content`\n- deepseek: puts thoughts in `message.reasoning_content`\n- deepseek-legacy: keeps `<think>` tags in `message.content` while also populating `message.reasoning_content`\n(default: auto)", |p, v| {
            p.reasoning_format = reasoning_format_from(&v)?;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Server, LlamaExample::Completion, LlamaExample::Cli,
        ])
        .set_env("LLAMA_ARG_THINK"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-rea", "--reasoning"], "[on|off|auto]", "Use reasoning/thinking in the chat ('on', 'off', or 'auto', default: 'auto' (detect from template))", |p, v| {
            if is_truthy(&v) {
                p.enable_reasoning = 1;
                p.default_template_kwargs.insert("enable_thinking".to_owned(), "true".to_owned());
            } else if is_falsey(&v) {
                p.enable_reasoning = 0;
                p.default_template_kwargs.insert("enable_thinking".to_owned(), "false".to_owned());
            } else if is_autoy(&v) {
                p.enable_reasoning = -1;
            } else {
                return Err(Error::new(format!("error: unknown value for --reasoning: '{v}'\n")));
            }
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Server, LlamaExample::Completion, LlamaExample::Cli,
        ])
        .set_env("LLAMA_ARG_REASONING"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--reasoning-effort"], "LEVEL", "reasoning effort level given to the chat template: 'default' to keep the template default,\nor a level such as 'minimal', 'low', 'medium', 'high', 'xhigh' or 'max' (default: default)", |p, v| {
            if v == "default" {
                p.default_template_kwargs.remove("reasoning_effort");
            } else {
                let dump = Json::String(v.clone()).dump(-1)?;
                p.default_template_kwargs.insert("reasoning_effort".to_owned(), dump);
            }
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Server, LlamaExample::Completion, LlamaExample::Cli,
        ])
        .set_env("LLAMA_ARG_REASONING_EFFORT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--reasoning-budget"], "N", "token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)", |p, v| {
            if v < -1 {
                return Err(Error::new("invalid value"));
            }
            p.sampling.reasoning_budget_tokens = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Server, LlamaExample::Completion, LlamaExample::Cli,
        ])
        .set_env("LLAMA_ARG_THINK_BUDGET"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--reasoning-budget-message"], "MESSAGE", "message injected before the end-of-thinking tag when reasoning budget is exhausted (default: none)", |p, v| {
            p.sampling.reasoning_budget_message = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Server, LlamaExample::Completion, LlamaExample::Cli,
        ])
        .set_env("LLAMA_ARG_THINK_BUDGET_MESSAGE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--reasoning-preserve"], &["--no-reasoning-preserve"], "preserve reasoning trace in the full history, not just the last assistant message (default: enabled)\ncompatible with certain templates having 'supports_preserve_reasoning' capability\nexample: https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking", |p, v| {
            p.default_template_kwargs.insert("preserve_reasoning".to_owned(), if v { "true" } else { "false" }.to_owned());
            p.preserve_reasoning_specified = true;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Server, LlamaExample::Completion, LlamaExample::Cli,
        ])
        .set_env("LLAMA_ARG_REASONING_PRESERVE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--chat-template"], "JINJA_TEMPLATE", "set custom jinja chat template (default: template taken from model's metadata)\nif suffix/prefix are specified, template will be disabled\nonly commonly used templates are accepted (unless --jinja is set before this flag):\nlist of built-in templates:\n(omitted in the Rust port)", |p, v| {
            p.chat_template = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Completion, LlamaExample::Cli, LlamaExample::Server,
            LlamaExample::Mtmd,
        ])
        .set_env("LLAMA_ARG_CHAT_TEMPLATE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--chat-template-file"], "JINJA_TEMPLATE_FILE", "set custom jinja chat template file (default: template taken from model's metadata)\nif suffix/prefix are specified, template will be disabled\nonly commonly used templates are accepted (unless --jinja is set before this flag):\nlist of built-in templates:\n(omitted in the Rust port)", |p, v| {
            p.chat_template = read_file(&v)?;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Completion, LlamaExample::Cli, LlamaExample::Server,
        ])
        .set_env("LLAMA_ARG_CHAT_TEMPLATE_FILE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--skip-chat-parsing"], &["--no-skip-chat-parsing"], "force a pure content parser, even if a Jinja template is specified; model will output everything in the content section, including any reasoning and/or tool calls (default: disabled)", |p, v| {
            p.force_pure_content_parser = v;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Completion, LlamaExample::Cli, LlamaExample::Server,
        ])
        .set_env("LLAMA_ARG_SKIP_CHAT_PARSING"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--prefill-assistant"], &["--no-prefill-assistant"], "whether to prefill the assistant's response if the last message is an assistant message (default: prefill enabled)\nwhen this flag is set, if the last message is an assistant message then it will be treated as a full message and not prefilled\n", |p, v| {
            p.prefill_assistant = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server])
        .set_env("LLAMA_ARG_PREFILL_ASSISTANT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-sps", "--slot-prompt-similarity"], "SIMILARITY", format!("how much the prompt of a request must match the prompt of a slot in order to use that slot (default: {:.2}, 0.0 = disabled)\n", ctx.params.slot_prompt_similarity), |p, v| {
            p.slot_prompt_similarity = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--lora-init-without-apply"], format!("load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: {})", if ctx.params.lora_init_without_apply { "enabled" } else { "disabled" }), |p| {
            p.lora_init_without_apply = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--sleep-idle-seconds"], "SECONDS", format!("number of seconds of idleness after which the server will sleep (default: {}; -1 = disabled)", ctx.params.sleep_idle_seconds), |p, v| {
            if v == 0 || v < -1 {
                return Err(Error::new("invalid value: cannot be 0 or less than -1"));
            }
            p.sleep_idle_seconds = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--simple-io"], "use basic IO for better compatibility in subprocesses and limited consoles", |p| {
            p.simple_io = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Completion, LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--positive-file"], "FNAME", format!("positive prompts file, one prompt per line (default: '{}')", ctx.params.cvector_positive_file), |p, v| {
            p.cvector_positive_file = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::CvectorGenerator]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--negative-file"], "FNAME", format!("negative prompts file, one prompt per line (default: '{}')", ctx.params.cvector_negative_file), |p, v| {
            p.cvector_negative_file = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::CvectorGenerator]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--pca-batch"], "N", format!("batch size used for PCA. Larger batch runs faster, but uses more memory (default: {})", ctx.params.n_pca_batch), |p, v| {
            p.n_pca_batch = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::CvectorGenerator]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--pca-iter"], "N", format!("number of iterations used for PCA (default: {})", ctx.params.n_pca_iterations), |p, v| {
            p.n_pca_iterations = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::CvectorGenerator]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--method"], "{pca, mean}", "dimensionality reduction method to be used (default: pca)", |p, v| {
            p.cvector_dimre_method = match v.as_str() {
                "pca" => DimreMethod::Pca,
                "mean" => DimreMethod::Mean,
                _ => return Err(Error::new("invalid value")),
            };
            Ok(())
        })
        .set_examples(&[LlamaExample::CvectorGenerator]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--output-format"], "{md,jsonl}", "output format for batched-bench results (default: md)", |p, v| {
            p.batched_bench_output_jsonl = match v.as_str() {
                "jsonl" => true,
                "md" => false,
                _ => return Err(Error::new("invalid value")),
            };
            Ok(())
        })
        .set_examples(&[LlamaExample::Bench]),
    );

    //
    // logging
    //

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--log-disable"], "Log disable", |_| Ok(())),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--log-file"], "FNAME", "Log to file", |_, _| Ok(()))
            .set_env("LLAMA_ARG_LOG_FILE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--log-jsonl"], &["--no-log-jsonl"], "Log as JSONL (one JSON object per line) to stdout, this also disables colored logging (default: disabled)", |_, _| Ok(()))
            .set_env("LLAMA_ARG_LOG_JSONL"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--log-prompts-dir"], "PATH", "Log prompts to directory (auto-created if not present; only used for debugging, default: disabled)", |p, v| {
            p.path_prompts_log_dir = v.clone();
            if fs::create_dir_all(&v).is_err() {
                eprintln!("warning: failed to create prompts-log-dir '{v}'");
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Server, LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--log-colors"], "[on|off|auto]", "Set colored logging ('on', 'off', or 'auto', default: 'auto')\n'auto' enables colors when output is to a terminal", |_, v| {
            if !is_truthy(&v) && !is_falsey(&v) && !is_autoy(&v) {
                return Err(Error::new(format!("error: unknown value for --log-colors: '{v}'\n")));
            }
            Ok(())
        })
        .set_env("LLAMA_ARG_LOG_COLORS"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["-v", "--verbose", "--log-verbose"], "Set verbosity level to infinity (i.e. log all messages, useful for debugging)", |p| {
            p.verbosity = i32::MAX;
            Ok(())
        }),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--offline"], "Offline mode: forces use of cache, prevents network access", |p| {
            p.offline = true;
            Ok(())
        })
        .set_examples(&[
            LlamaExample::Common, LlamaExample::Download, LlamaExample::Tokenize,
        ])
        .set_env("LLAMA_ARG_OFFLINE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-lv", "--verbosity", "--log-verbosity"], "N", format!("Set the verbosity threshold. Messages with a higher verbosity will be ignored. Values:\n - 0: generic output\n - 1: error\n - 2: warning\n - 3: info\n - 4: trace (more info)\n - 5: debug\n(default: {})\n", ctx.params.verbosity), |p, v| {
            p.verbosity = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_LOG_VERBOSITY"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--log-prefix"], &["--no-log-prefix"], "Enable prefix in log messages", |_, _| Ok(()))
            .set_env("LLAMA_ARG_LOG_PREFIX"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--log-timestamps"], &["--no-log-timestamps"], "Enable timestamps in log messages", |_, _| Ok(()))
            .set_env("LLAMA_ARG_LOG_TIMESTAMPS"),
    );

    //
    // speculative params
    //

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft"], "<user>/<model>[:quant]", "Same as --hf-repo, but for the draft model (default: unused)", |p, v| {
            p.speculative.draft.mparams.hf_repo = v;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples)
        .set_env("LLAMA_ARG_SPEC_DRAFT_HF_REPO"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-draft-threads", "-td", "--threads-draft"], "N", "number of threads to use during generation (default: same as --threads)", |p, v| {
            p.speculative.draft.cpuparams.n_threads = if v <= 0 { hardware_threads() } else { v };
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-draft-threads-batch", "-tbd", "--threads-batch-draft"], "N", "number of threads to use during batch and prompt processing (default: same as --threads-draft)", |p, v| {
            p.speculative.draft.cpuparams_batch.n_threads = if v <= 0 { hardware_threads() } else { v };
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-cpu-mask", "-Cd", "--cpu-mask-draft"], "M", "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)", |p, v| {
            p.speculative.draft.cpuparams.mask_valid = true;
            if !cpu_mask(&v, &mut p.speculative.draft.cpuparams.cpumask) {
                return Err(Error::new("invalid cpumask"));
            }
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-cpu-range", "-Crd", "--cpu-range-draft"], "lo-hi", "Ranges of CPUs for affinity. Complements --cpu-mask-draft", |p, v| {
            p.speculative.draft.cpuparams.mask_valid = true;
            if !cpu_range(&v, &mut p.speculative.draft.cpuparams.cpumask) {
                return Err(Error::new("invalid range"));
            }
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-cpu-strict", "--cpu-strict-draft"], "<0|1>", "Use strict CPU placement for draft model (default: same as --cpu-strict)", |p, v| {
            p.speculative.draft.cpuparams.strict_cpu = parse_number_prefix::<u32>(&v)? != 0;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-draft-prio", "--prio-draft"], "N", format!("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: {})\n", priority_value(&ctx.params.speculative.draft.cpuparams.priority)), |p, v| {
            if !(0..=3).contains(&v) {
                return Err(Error::new("invalid value"));
            }
            p.speculative.draft.cpuparams.priority = sched_priority(v);
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-draft-poll", "--poll-draft"], "<0|1>", "Use polling to wait for draft model work (default: same as --poll)", |p, v| {
            p.speculative.draft.cpuparams.poll = v.max(0) as u32;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-cpu-mask-batch", "-Cbd", "--cpu-mask-batch-draft"], "M", "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)", |p, v| {
            p.speculative.draft.cpuparams_batch.mask_valid = true;
            if !cpu_mask(&v, &mut p.speculative.draft.cpuparams_batch.cpumask) {
                return Err(Error::new("invalid cpumask"));
            }
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-cpu-range-batch", "-Crbd", "--cpu-range-batch-draft"], "lo-hi", "Ranges of CPUs for affinity. Complements --cpu-mask-draft-batch)", |p, v| {
            p.speculative.draft.cpuparams_batch.mask_valid = true;
            if !cpu_range(&v, &mut p.speculative.draft.cpuparams_batch.cpumask) {
                return Err(Error::new("invalid cpumask"));
            }
            Ok(())
        })
        .set_spec()
        .set_examples(&[LlamaExample::Speculative]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-cpu-strict-batch", "--cpu-strict-batch-draft"], "<0|1>", "Use strict CPU placement for draft model (default: --cpu-strict-draft)", |p, v| {
            p.speculative.draft.cpuparams_batch.strict_cpu = parse_number_prefix::<u32>(&v)? != 0;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-draft-prio-batch", "--prio-batch-draft"], "N", format!("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: {})\n", priority_value(&ctx.params.speculative.draft.cpuparams_batch.priority)), |p, v| {
            if !(0..=3).contains(&v) {
                return Err(Error::new("invalid value"));
            }
            p.speculative.draft.cpuparams_batch.priority = sched_priority(v);
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-draft-poll-batch", "--poll-batch-draft"], "<0|1>", "Use polling to wait for draft model work (default: --poll-draft)", |p, v| {
            p.speculative.draft.cpuparams_batch.poll = v.max(0) as u32;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-type-k", "-ctkd", "--cache-type-k-draft"], "TYPE", format!("KV cache data type for K for the draft model\nallowed values: {}\n(default: {})", get_all_kv_cache_types(), ctx.params.speculative.draft.cache_type_k.name()), |p, v| {
            p.speculative.draft.cache_type_k = kv_cache_type_from_str(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-type-v", "-ctvd", "--cache-type-v-draft"], "TYPE", format!("KV cache data type for V for the draft model\nallowed values: {}\n(default: {})", get_all_kv_cache_types(), ctx.params.speculative.draft.cache_type_v.name()), |p, v| {
            p.speculative.draft.cache_type_v = kv_cache_type_from_str(&v)?;
            Ok(())
        })
        .set_env("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-override-tensor", "-otd", "--override-tensor-draft"], "<tensor name pattern>=<buffer type>,...", "override tensor buffer type for draft model", |p, v| {
            parse_tensor_buffer_overrides(&v, &mut p.speculative.draft.tensor_buft_overrides)
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--spec-draft-cpu-moe", "-cmoed", "--cpu-moe-draft"], "keep all Mixture of Experts (MoE) weights in the CPU for the draft model", |p| {
            p.speculative.draft.tensor_buft_overrides.push(llm_ffn_exps_cpu_override());
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples)
        .set_env("LLAMA_ARG_SPEC_DRAFT_CPU_MOE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-draft-n-cpu-moe", "--spec-draft-ncmoe", "-ncmoed", "--n-cpu-moe-draft"], "N", "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU for the draft model", |p, v| {
            if v < 0 {
                return Err(Error::new("invalid value"));
            }
            llm_add_n_cpu_ffn_overrides(v, LLM_FFN_EXPS_REGEX, &mut p.speculative.draft.tensor_buft_overrides);
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples)
        .set_env("LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-draft-n-max"], "N", format!("number of tokens to draft for speculative decoding (default: {})", ctx.params.speculative.draft.n_max), |p, v| {
            if v < 0 {
                return Err(Error::new("invalid value"));
            }
            p.speculative.draft.n_max = v;
            Ok(())
        })
        .set_spec()
        .set_examples(&[
            LlamaExample::Speculative, LlamaExample::Lookup, LlamaExample::Server,
            LlamaExample::Cli,
        ])
        .set_env("LLAMA_ARG_SPEC_DRAFT_N_MAX"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-draft-n-min"], "N", format!("minimum number of draft tokens to use for speculative decoding (default: {})", ctx.params.speculative.draft.n_min), |p, v| {
            p.speculative.draft.n_min = v;
            Ok(())
        })
        .set_spec()
        .set_examples(&[
            LlamaExample::Speculative, LlamaExample::Lookup, LlamaExample::Server,
            LlamaExample::Cli,
        ])
        .set_env("LLAMA_ARG_SPEC_DRAFT_N_MIN"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-draft-n-ubatch"], "N", format!("physical maximum batch size of the draft context, use a smaller value to spare VRAM (default: {} - same as --n-ubatch)", ctx.params.speculative.draft.n_ubatch), |p, v| {
            if v < 0 {
                return Err(Error::new("invalid value"));
            }
            p.speculative.draft.n_ubatch = v;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples)
        .set_env("LLAMA_ARG_SPEC_DRAFT_N_UBATCH"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-synth-len"], "L", "target mean synthetic acceptance length, including the target token (benchmarking only)", |p, v| {
            let length: f64 = parse_number_exact(&v)?;
            if length == -1.0 {
                return Err(Error::new("invalid value"));
            }
            p.speculative.synth_len = length;
            Ok(())
        })
        .set_spec()
        .set_examples(&[LlamaExample::Server, LlamaExample::Cli])
        .set_env("LLAMA_ARG_SPEC_SYNTH_LEN"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-synth-rates"], "P0,P1,...", "comma-separated unconditional per-position synthetic acceptance probabilities (benchmarking only)", |p, v| {
            let mut rates = Vec::new();
            for raw in string_split_char(&v, ',') {
                rates.push(parse_number_exact::<f64>(&raw)?);
            }
            p.speculative.synth_rates = rates;
            Ok(())
        })
        .set_spec()
        .set_examples(&[LlamaExample::Server, LlamaExample::Cli])
        .set_env("LLAMA_ARG_SPEC_SYNTH_RATES"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-p-split", "--draft-p-split"], "P", format!("speculative decoding split probability (default: {:.2})", ctx.params.speculative.draft.p_split), |p, v| {
            p.speculative.draft.p_split = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples)
        .set_env("LLAMA_ARG_SPEC_DRAFT_P_SPLIT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-p-min", "--draft-p-min"], "P", format!("minimum speculative decoding probability (greedy) (default: {:.2})", ctx.params.speculative.draft.p_min), |p, v| {
            p.speculative.draft.p_min = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples)
        .set_env("LLAMA_ARG_SPEC_DRAFT_P_MIN"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::boolean(&["--spec-draft-backend-sampling"], &["--no-spec-draft-backend-sampling"], format!("offload draft sampling to the backend (default: {})", if ctx.params.speculative.draft.backend_sampling { "enabled" } else { "disabled" }), |p, v| {
            p.speculative.draft.backend_sampling = v;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples)
        .set_env("LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-sampling"], "{greedy,probabilistic}", format!("how the draft is sampled: greedy takes its argmax, probabilistic samples it and has the target verify by rejection sampling (default: {})", if ctx.params.speculative.draft.probabilistic { "probabilistic" } else { "greedy" }), |p, v| {
            p.speculative.draft.probabilistic = match v.as_str() {
                "greedy" => false,
                "probabilistic" => true,
                _ => return Err(Error::new("invalid value, must be one of: greedy, probabilistic")),
            };
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples)
        .set_env("LLAMA_ARG_SPEC_DRAFT_SAMPLING"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-device", "-devd", "--device-draft"], "<dev1,dev2,..>", "comma-separated list of devices to use for offloading the draft model (none = don't offload, default: follows --device)\nuse --list-devices to see a list of available devices", |p, v| {
            p.speculative.draft.devices = parse_device_list(&v)?;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"], "N", format!("max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' (default: {})", if ctx.params.speculative.draft.n_gpu_layers == -1 { "auto" } else { "all" }), |p, v| {
            p.speculative.draft.n_gpu_layers = match v.as_str() {
                "auto" => -1,
                "all" => -2,
                other => parse_number_prefix(other)?,
            };
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples)
        .set_env("LLAMA_ARG_N_GPU_LAYERS_DRAFT"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-draft-model", "-md", "--model-draft"], "FNAME", "draft model for speculative decoding (default: unused)", |p, v| {
            p.speculative.draft.mparams.path = v.clone();
            p.speculative.draft.mparams.hf_file = v;
            Ok(())
        })
        .set_spec()
        .set_examples(&[
            LlamaExample::Speculative, LlamaExample::Server, LlamaExample::Cli,
            LlamaExample::Imatrix,
        ])
        .set_env("LLAMA_ARG_SPEC_DRAFT_MODEL"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--spec-type"], spec_all_types(), format!("comma-separated list of types of speculative decoding to use (default: {})\n", spec_type_name_str(&ctx.params.speculative.types)), |p, v| {
            let names = string_split_char(&v, ',');
            let types = spec_types_from_names(&names)?;
            p.speculative.types.extend(types);
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples)
        .set_env("LLAMA_ARG_SPEC_TYPE"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-mod-n-min"], "N", format!("minimum number of ngram tokens to use for ngram-based speculative decoding (default: {})", ctx.params.speculative.ngram_mod.n_min), |p, v| {
            if !(0..=1024).contains(&v) {
                return Err(Error::new("ngram n-min must be between 0 and 1024 inclusive"));
            }
            p.speculative.ngram_mod.n_min = v;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-mod-n-max"], "N", format!("maximum number of ngram tokens to use for ngram-based speculative decoding (default: {})", ctx.params.speculative.ngram_mod.n_max), |p, v| {
            if !(0..=1024).contains(&v) {
                return Err(Error::new("ngram n-max must be between 0 and 1024 inclusive"));
            }
            p.speculative.ngram_mod.n_max = v;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-mod-n-match"], "N", format!("ngram-mod lookup length (default: {})", ctx.params.speculative.ngram_mod.n_match), |p, v| {
            if !(1..=1024).contains(&v) {
                return Err(Error::new("ngram size N must be between 1 and 1024 inclusive"));
            }
            p.speculative.ngram_mod.n_match = v;
            Ok(())
        })
        .set_spec()
        .set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-simple-size-n"], "N", format!("ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: {})", ctx.params.speculative.ngram_simple.size_n), |p, v| {
            if !(1..=1024).contains(&v) { return Err(Error::new("ngram size N must be between 1 and 1024 inclusive")); }
            p.speculative.ngram_simple.size_n = v as u16;
            Ok(())
        }).set_spec().set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-simple-size-m"], "N", format!("ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: {})", ctx.params.speculative.ngram_simple.size_m), |p, v| {
            if !(1..=1024).contains(&v) { return Err(Error::new("ngram size M must be between 1 and 1024 inclusive")); }
            p.speculative.ngram_simple.size_m = v as u16;
            Ok(())
        }).set_spec().set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-simple-min-hits"], "N", format!("minimum hits for ngram-simple speculative decoding (default: {})", ctx.params.speculative.ngram_simple.min_hits), |p, v| {
            if v < 1 { return Err(Error::new("ngram min hits must be at least 1")); }
            p.speculative.ngram_simple.min_hits = v as u16;
            Ok(())
        }).set_spec().set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-map-k-size-n"], "N", format!("ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: {})", ctx.params.speculative.ngram_map_k.size_n), |p, v| {
            if !(1..=1024).contains(&v) { return Err(Error::new("ngram size N must be between 1 and 1024 inclusive")); }
            p.speculative.ngram_map_k.size_n = v as u16;
            Ok(())
        }).set_spec().set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-map-k-size-m"], "N", format!("ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: {})", ctx.params.speculative.ngram_map_k.size_m), |p, v| {
            if !(1..=1024).contains(&v) { return Err(Error::new("ngram size M must be between 1 and 1024 inclusive")); }
            p.speculative.ngram_map_k.size_m = v as u16;
            Ok(())
        }).set_spec().set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-map-k-min-hits"], "N", format!("minimum hits for ngram-map-k speculative decoding (default: {})", ctx.params.speculative.ngram_map_k.min_hits), |p, v| {
            if v < 1 { return Err(Error::new("ngram min hits must be at least 1")); }
            p.speculative.ngram_map_k.min_hits = v as u16;
            Ok(())
        }).set_spec().set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-map-k4v-size-n"], "N", format!("ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: {})", ctx.params.speculative.ngram_map_k4v.size_n), |p, v| {
            if !(1..=1024).contains(&v) { return Err(Error::new("ngram size N must be between 1 and 1024 inclusive")); }
            p.speculative.ngram_map_k4v.size_n = v as u16;
            Ok(())
        }).set_spec().set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-map-k4v-size-m"], "N", format!("ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: {})", ctx.params.speculative.ngram_map_k4v.size_m), |p, v| {
            if !(1..=1024).contains(&v) { return Err(Error::new("ngram size M must be between 1 and 1024 inclusive")); }
            p.speculative.ngram_map_k4v.size_m = v as u16;
            Ok(())
        }).set_spec().set_examples(spec_examples),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-map-k4v-min-hits"], "N", format!("minimum hits for ngram-map-k4v speculative decoding (default: {})", ctx.params.speculative.ngram_map_k4v.min_hits), |p, v| {
            if v < 1 { return Err(Error::new("ngram min hits must be at least 1")); }
            p.speculative.ngram_map_k4v.min_hits = v as u16;
            Ok(())
        }).set_spec().set_examples(spec_examples),
    );

    //
    // removed params
    //

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--draft", "--draft-n", "--draft-max"], "N", "the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max", |_, _| {
            Err(Error::new("the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max"))
        })
        .set_spec()
        .set_examples(&[
            LlamaExample::Speculative, LlamaExample::Lookup, LlamaExample::Server,
            LlamaExample::Cli,
        ])
        .set_env("LLAMA_ARG_DRAFT_MAX"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--draft-min", "--draft-n-min"], "N", "the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min", |_, _| {
            Err(Error::new("the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min"))
        })
        .set_spec()
        .set_examples(&[
            LlamaExample::Speculative, LlamaExample::Lookup, LlamaExample::Server,
            LlamaExample::Cli,
        ])
        .set_env("LLAMA_ARG_DRAFT_MIN"),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-size-n"], "N", "the argument has been removed. use the respective --spec-ngram-*-size-n or --spec-ngram-mod-n-match", |_, _| {
            Err(Error::new("the argument has been removed. use the respective --spec-ngram-*-size-n or --spec-ngram-mod-n-match"))
        }).set_spec().set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-size-m"], "N", "the argument has been removed. use the respective --spec-ngram-*-size-m", |_, _| {
            Err(Error::new("the argument has been removed. use the respective --spec-ngram-*-size-m"))
        }).set_spec().set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--spec-ngram-min-hits"], "N", "the argument has been removed. use the respective --spec-ngram-*-min-hits", |_, _| {
            Err(Error::new("the argument has been removed. use the respective --spec-ngram-*-min-hits"))
        }).set_spec().set_examples(&[LlamaExample::Server]),
    );

    //
    // TTS params
    //

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--tts-lang"], "FNAME", "language (ISO 639-1) for audio generation\nsee docs/multimodal/tts.md for per-model usage notes", |p, v| {
            p.tts_lang = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Tts, LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--tts-voice"], "VOICE", "voice for audio generation (model-dependent)\nsee docs/multimodal/tts.md for per-model usage notes", |p, v| {
            p.tts_voice = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Tts, LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--tts-speaker-file"], "FNAME", "speaker file path for audio generation", |p, v| {
            p.tts_speaker_file = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Tts]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--tts-speaker-text"], "TEXT", "transcript of --tts-speaker-file, for models that need it to clone the voice\nsee docs/multimodal/tts.md for per-model usage notes", |p, v| {
            p.tts_speaker_text = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Tts]),
    );

    //
    // diffusion params
    //

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--diffusion-steps"], "N", format!("number of diffusion steps (default: {})", ctx.params.diffusion.steps), |p, v| {
            p.diffusion.steps = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Diffusion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--diffusion-visual"], format!("enable visual diffusion mode (show progressive generation) (default: {})", ctx.params.diffusion.visual_mode), |p| {
            p.diffusion.visual_mode = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Diffusion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--diffusion-eps"], "F", format!("epsilon for timesteps (default: {:.6})", ctx.params.diffusion.eps as f64), |p, v| {
            p.diffusion.eps = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_examples(&[LlamaExample::Diffusion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--diffusion-algorithm"], "N", format!("diffusion algorithm: 0=DIFFUSION_ALGORITHM_ORIGIN, 1=DIFFUSION_ALGORITHM_ENTROPY_BASED, 2=DIFFUSION_ALGORITHM_MARGIN_BASED, 3=DIFFUSION_ALGORITHM_RANDOM, 4=DIFFUSION_ALGORITHM_CONFIDENCE_BASED (default: {})", ctx.params.diffusion.algorithm), |p, v| {
            p.diffusion.algorithm = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Diffusion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--diffusion-alg-temp"], "F", format!("dream algorithm temperature (default: {:.3})", ctx.params.diffusion.alg_temp as f64), |p, v| {
            p.diffusion.alg_temp = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_examples(&[LlamaExample::Diffusion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["--diffusion-block-length"], "N", format!("llada block length for generation (default: {})", ctx.params.diffusion.block_length), |p, v| {
            p.diffusion.block_length = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Diffusion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--diffusion-cfg-scale"], "F", format!("llada classifier-free guidance scale (default: {:.3})", ctx.params.diffusion.cfg_scale as f64), |p, v| {
            p.diffusion.cfg_scale = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_examples(&[LlamaExample::Diffusion]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--diffusion-add-gumbel-noise"], "F", format!("add gumbel noise to the logits if temp > 0.0 (default: {})", ctx.params.diffusion.add_gumbel_noise), |p, v| {
            let noise: f32 = parse_number_prefix(&v)?;
            p.diffusion.add_gumbel_noise = noise != 0.0;
            Ok(())
        })
        .set_examples(&[LlamaExample::Diffusion]),
    );

    //
    // finetune params
    //

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-lr", "--learning-rate"], "ALPHA", format!("adamw or sgd optimizer alpha (default: {:.2}); note: sgd alpha recommended ~10x (no momentum)", ctx.params.lr.lr0), |p, v| {
            p.lr.lr0 = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_examples(&[LlamaExample::Finetune]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-lr-min", "--learning-rate-min"], "ALPHA", format!("(if >0) final learning rate after decay (if -decay-epochs is set, default={:.2})", ctx.params.lr.lr_min), |p, v| {
            p.lr.lr_min = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_examples(&[LlamaExample::Finetune]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-decay-epochs", "--learning-rate-decay-epochs"], "ALPHA", format!("(if >0) decay learning rate to -lr-min after this many epochs (exponential decay, default={:.2})", ctx.params.lr.decay_epochs), |p, v| {
            p.lr.decay_epochs = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_examples(&[LlamaExample::Finetune]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-wd", "--weight-decay"], "WD", format!("adamw or sgd optimizer weight decay (0 is off; recommend very small e.g. 1e-9) (default: {:.2}).", ctx.params.lr.wd), |p, v| {
            p.lr.wd = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_examples(&[LlamaExample::Finetune]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-val-split", "--val-split"], "FRACTION", format!("fraction of data to use as validation set for training (default: {:.2}).", ctx.params.val_split), |p, v| {
            p.val_split = parse_number_prefix(&v)?;
            Ok(())
        })
        .set_examples(&[LlamaExample::Finetune]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::int(&["-epochs", "--epochs"], "N", format!("optimizer max # of epochs (default: {})", ctx.params.lr.epochs), |p, v| {
            p.lr.epochs = v.max(0) as u32;
            Ok(())
        })
        .set_examples(&[LlamaExample::Finetune]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["-opt", "--optimizer"], "sgd|adamw", "adamw or sgd", |p, v| {
            p.optimizer = common_opt_get_optimizer(&v);
            if p.optimizer == OptimizerType::Count {
                return Err(Error::new("invalid --optimizer, valid options: adamw, sgd"));
            }
            Ok(())
        })
        .set_examples(&[LlamaExample::Finetune]),
    );

    //
    // results / debug
    //

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--check"], format!("check rather than generate results (default: {})", ctx.params.check), |p| {
            p.check = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Results]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--save-logits"], format!("save final logits to files for verification (default: {})", ctx.params.save_logits), |p| {
            p.save_logits = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Debug]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--logits-output-dir"], "PATH", format!("directory for saving logits output files (default: {})", ctx.params.logits_output_dir), |p, v| {
            p.logits_output_dir = v;
            Ok(())
        })
        .set_examples(&[LlamaExample::Debug]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::string(&["--tensor-filter"], "REGEX", "filter tensor names for debug output (regex pattern, can be specified multiple times)", |p, v| {
            p.tensor_filter.push(v);
            Ok(())
        })
        .set_examples(&[LlamaExample::Debug]),
    );

    //
    // presets
    //

    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--embd-gemma-default"], "use default EmbeddingGemma model (note: can download weights from the internet)", |p| {
            p.model.hf_repo = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF".into();
            p.model.hf_file = "embeddinggemma-300M-qat-Q4_0.gguf".into();
            p.port = 8011;
            p.n_ubatch = 2048;
            p.n_batch = 2048;
            p.n_parallel = 32;
            p.n_ctx = 2048 * p.n_parallel;
            p.verbose_prompt = true;
            p.embedding = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Embedding, LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--fim-qwen-1.5b-default"], "use default Qwen 2.5 Coder 1.5B (note: can download weights from the internet)", |p| {
            p.model.hf_repo = "ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF".into();
            p.model.hf_file = "qwen2.5-coder-1.5b-q8_0.gguf".into();
            p.port = 8012;
            p.n_ubatch = 1024;
            p.n_batch = 1024;
            p.n_ctx = 0;
            p.n_cache_reuse = 256;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--fim-qwen-3b-default"], "use default Qwen 2.5 Coder 3B (note: can download weights from the internet)", |p| {
            p.model.hf_repo = "ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF".into();
            p.model.hf_file = "qwen2.5-coder-3b-q8_0.gguf".into();
            p.port = 8012;
            p.n_ubatch = 1024;
            p.n_batch = 1024;
            p.n_ctx = 0;
            p.n_cache_reuse = 256;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--fim-qwen-7b-default"], "use default Qwen 2.5 Coder 7B (note: can download weights from the internet)", |p| {
            p.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF".into();
            p.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf".into();
            p.port = 8012;
            p.n_ubatch = 1024;
            p.n_batch = 1024;
            p.n_ctx = 0;
            p.n_cache_reuse = 256;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--fim-qwen-7b-spec"], "use Qwen 2.5 Coder 7B + 0.5B draft for speculative decoding (note: can download weights from the internet)", |p| {
            p.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF".into();
            p.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf".into();
            p.speculative.draft.mparams.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF".into();
            p.speculative.draft.mparams.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf".into();
            p.port = 8012;
            p.n_ubatch = 1024;
            p.n_batch = 1024;
            p.n_ctx = 0;
            p.n_cache_reuse = 256;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--fim-qwen-14b-spec"], "use Qwen 2.5 Coder 14B + 0.5B draft for speculative decoding (note: can download weights from the internet)", |p| {
            p.model.hf_repo = "ggml-org/Qwen2.5-Coder-14B-Q8_0-GGUF".into();
            p.model.hf_file = "qwen2.5-coder-14b-q8_0.gguf".into();
            p.speculative.draft.mparams.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF".into();
            p.speculative.draft.mparams.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf".into();
            p.port = 8012;
            p.n_ubatch = 1024;
            p.n_batch = 1024;
            p.n_ctx = 0;
            p.n_cache_reuse = 256;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--fim-qwen-30b-default"], "use default Qwen 3 Coder 30B A3B Instruct (note: can download weights from the internet)", |p| {
            p.model.hf_repo = "ggml-org/Qwen3-Coder-30B-A3B-Instruct-Q8_0-GGUF".into();
            p.model.hf_file = "qwen3-coder-30b-a3b-instruct-q8_0.gguf".into();
            p.port = 8012;
            p.n_ubatch = 1024;
            p.n_batch = 1024;
            p.n_ctx = 0;
            p.n_cache_reuse = 256;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--gpt-oss-20b-default"], "use gpt-oss-20b (note: can download weights from the internet)", |p| {
            p.model.hf_repo = "ggml-org/gpt-oss-20b-GGUF".into();
            p.model.hf_file = "gpt-oss-20b-mxfp4.gguf".into();
            p.port = 8013;
            p.n_ubatch = 2048;
            p.n_batch = 32768;
            p.n_parallel = 2;
            p.n_ctx = 131072 * p.n_parallel;
            p.sampling.temp = 1.0;
            p.sampling.top_p = 1.0;
            p.sampling.top_k = 0;
            p.sampling.min_p = 0.01;
            p.use_jinja = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server, LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--gpt-oss-120b-default"], "use gpt-oss-120b (note: can download weights from the internet)", |p| {
            p.model.hf_repo = "ggml-org/gpt-oss-120b-GGUF".into();
            p.port = 8013;
            p.n_ubatch = 2048;
            p.n_batch = 32768;
            p.n_parallel = 2;
            p.n_ctx = 131072 * p.n_parallel;
            p.sampling.temp = 1.0;
            p.sampling.top_p = 1.0;
            p.sampling.top_k = 0;
            p.sampling.min_p = 0.01;
            p.use_jinja = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server, LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--vision-gemma-4b-default"], "use Gemma 3 4B QAT (note: can download weights from the internet)", |p| {
            p.model.hf_repo = "ggml-org/gemma-3-4b-it-qat-GGUF".into();
            p.port = 8014;
            p.n_ctx = 0;
            p.use_jinja = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server, LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--vision-gemma-12b-default"], "use Gemma 3 12B QAT (note: can download weights from the internet)", |p| {
            p.model.hf_repo = "ggml-org/gemma-3-12b-it-qat-GGUF".into();
            p.port = 8014;
            p.n_ctx = 0;
            p.use_jinja = true;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server, LlamaExample::Cli]),
    );
    add_opt(
        &mut ctx.options,
        ctx.ex,
        CommonArg::void(&["--spec-default"], "enable default speculative decoding config", |p| {
            p.speculative.types.push(SpeculativeType::NgramMod);
            p.speculative.ngram_mod.n_match = 24;
            p.speculative.ngram_mod.n_min = 48;
            p.speculative.ngram_mod.n_max = 64;
            Ok(())
        })
        .set_examples(&[LlamaExample::Server, LlamaExample::Cli]),
    );

    ctx
}

// ---------------------------------------------------------------------------
// parsing
// ---------------------------------------------------------------------------

/// What the caller should do after parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseOutcome {
    /// Continue with the parsed params.
    Parsed,
    /// Print usage and stop.
    Usage,
    /// Print shell completion and stop.
    Completion,
}

/// `common_params_parse_ex`.
pub fn common_params_parse_ex(args: &[String], ctx: &mut ParamsContext) -> Result<ParseOutcome> {
    // config files apply first, so env and CLI arguments override them
    // shortcut: system/user INI files are not applied (no preset engine in this port)

    let mut arg_to_options: std::collections::BTreeMap<String, (usize, bool)> =
        std::collections::BTreeMap::new();
    for (i, opt) in ctx.options.iter().enumerate() {
        for arg in &opt.args {
            arg_to_options.insert((*arg).to_owned(), (i, true));
        }
        for arg in &opt.args_neg {
            arg_to_options.insert((*arg).to_owned(), (i, false));
        }
    }

    // handle environment variables
    for i in 0..ctx.options.len() {
        let opt = ctx.options[i].clone();
        if let Some(value) = opt.get_value_from_env() {
            if let Err(e) = opt.apply_env(&mut ctx.params, &value) {
                return Err(Error::new(format!(
                    "error while handling environment variable \"{}\": {}\n\n",
                    opt.env.unwrap_or_default(),
                    e
                )));
            }
        }
    }

    // handle command line arguments
    let mut seen_args = std::collections::BTreeSet::new();
    let mut i = 1usize;
    while i < args.len() {
        let mut arg = args[i].clone();
        if arg.starts_with("--") {
            arg = arg.replace('_', "-");
        }
        let Some(&(opt_idx, is_positive)) = arg_to_options.get(&arg) else {
            return Err(Error::new(format!("error: invalid argument: {arg}")));
        };
        if !seen_args.insert(arg.clone()) && arg != "--spec-type" {
            eprintln!("DEPRECATED: argument '{arg}' specified multiple times, use comma-separated values instead (only last value will be used)");
        }
        let opt = ctx.options[opt_idx].clone();
        if opt.has_value_from_env() {
            eprintln!(
                "warn: {} environment variable is set, but will be overwritten by command line argument {}",
                opt.env.unwrap_or_default(),
                arg
            );
        }
        let consumes_one = matches!(opt.handler, Handler::Int(_) | Handler::String(_));
        let consumes_two = matches!(opt.handler, Handler::StrStr(_));
        let mut val = None;
        let mut val2 = None;
        if consumes_one || consumes_two {
            if i + 1 >= args.len() {
                return Err(Error::new("error while handling argument: expected value for argument"));
            }
            i += 1;
            val = Some(args[i].clone());
        }
        if consumes_two {
            if i + 1 >= args.len() {
                return Err(Error::new("error while handling argument: expected value for argument"));
            }
            i += 1;
            val2 = Some(args[i].clone());
        }
        if let Err(e) = opt.apply_cli(&mut ctx.params, is_positive, val, val2) {
            return Err(Error::new(format!(
                "error while handling argument \"{arg}\": {}\n\nusage:\n{}\n\nto show complete usage, run with -h",
                e,
                opt.to_string()
            )));
        }
        i += 1;
    }

    postprocess_cpu_params(&mut ctx.params.cpuparams, None);
    let model_cpu = ctx.params.cpuparams.clone();
    let batch_cpu = ctx.params.cpuparams_batch.clone();
    postprocess_cpu_params(&mut ctx.params.cpuparams_batch, Some(&model_cpu));
    postprocess_cpu_params(&mut ctx.params.speculative.draft.cpuparams, Some(&model_cpu));
    postprocess_cpu_params(
        &mut ctx.params.speculative.draft.cpuparams_batch,
        Some(&batch_cpu),
    );

    if ctx.params.prompt_cache_all && (ctx.params.interactive || ctx.params.interactive_first) {
        return Err(Error::new(
            "error: --prompt-cache-all not supported in interactive mode yet\n",
        ));
    }
    if ctx.params.sampling.ignore_eos && !ctx.params.sampling.grammar.empty() {
        return Err(Error::new(
            "error: --ignore-eos is not supported with --grammar or --json-schema\n",
        ));
    }

    let skip_model_download = matches!(
        ctx.ex,
        LlamaExample::Server | LlamaExample::Download | LlamaExample::ExportGraphOps
    );
    if !skip_model_download {
        // shortcut: model downloads are not performed by this port
        let model = &ctx.params.model;
        let can_skip_model = ctx.params.usage
            || ctx.params.completion
            || !ctx.params.server_base.is_empty();
        if !can_skip_model
            && model.path.is_empty()
            && model.hf_repo.is_empty()
            && model.url.is_empty()
            && model.docker_repo.is_empty()
        {
            return Err(Error::new("error: --model or -hf is required\n"));
        }
    }

    if ctx.params.escape {
        string_process_escapes(&mut ctx.params.prompt);
        string_process_escapes(&mut ctx.params.input_prefix);
        string_process_escapes(&mut ctx.params.input_suffix);
        for antiprompt in &mut ctx.params.antiprompt {
            string_process_escapes(antiprompt);
        }
        for seq_breaker in &mut ctx.params.sampling.dry_sequence_breakers {
            string_process_escapes(seq_breaker);
        }
    }

    if !ctx.params.kv_overrides.is_empty() {
        ctx.params.kv_overrides.push(crate::common::KvOverride::default());
    }

    let mcp_enabled =
        !ctx.params.mcp_servers_config.is_empty() || !ctx.params.mcp_servers_json.is_empty();
    if (!ctx.params.server_tools.is_empty() || mcp_enabled || ctx.params.ui_mcp_proxy)
        && !ctx.params.cors_origins_explicit
    {
        eprintln!("warning: server tools, MCP servers or MCP proxy are enabled, using localhost as default CORS origin (change via --cors-origins)");
        ctx.params.cors_origins = "localhost".into();
    }

    // shortcut: tensor_buft_overrides stay a Vec, so no fixed-size padding is needed

    // shortcut: chat template verification needs the Jinja engine and is not done here

    if !ctx.params.default_template_kwargs.contains_key("preserve_reasoning") {
        ctx.params
            .default_template_kwargs
            .insert("preserve_reasoning".into(), "true".into());
    }

    if ctx.params.usage {
        Ok(ParseOutcome::Usage)
    } else if ctx.params.completion {
        Ok(ParseOutcome::Completion)
    } else {
        Ok(ParseOutcome::Parsed)
    }
}

/// `common_params_parse`, returning an error instead of printing and exiting.
pub fn common_params_parse(
    args: &[String],
    params: &mut Params,
    ex: LlamaExample,
) -> Result<ParseOutcome> {
    let original = params.clone();
    let mut ctx = common_params_parser_init(&original, ex);
    match common_params_parse_ex(args, &mut ctx) {
        Ok(outcome) => {
            *params = ctx.params;
            if outcome == ParseOutcome::Parsed {
                params.lr.init();
            }
            Ok(outcome)
        }
        Err(e) => {
            *params = original;
            Err(e)
        }
    }
}

/// `common_params_print_usage`.
pub fn common_params_print_usage(ctx: &ParamsContext) {
    let mut common_options = Vec::new();
    let mut sampling_options = Vec::new();
    let mut spec_options = Vec::new();
    let mut specific_options = Vec::new();
    for opt in &ctx.options {
        if opt.is_sampling {
            sampling_options.push(opt);
        } else if opt.is_spec {
            spec_options.push(opt);
        } else if opt.in_example(ctx.ex) {
            specific_options.push(opt);
        } else if !opt.is_preset_only {
            common_options.push(opt);
        }
    }

    let mut first = true;
    let mut print_section = |header: &str, options: &[&CommonArg]| {
        if options.is_empty() {
            return;
        }
        print!("{}----- {} -----\n\n", if first { "" } else { "\n\n" }, header);
        first = false;
        for opt in options {
            print!("{}", opt.to_string());
        }
    };
    print_section("common params", &common_options);
    print_section("sampling params", &sampling_options);
    print_section("speculative params", &spec_options);
    print_section("example-specific params", &specific_options);
}

/// `common_params_print_completion`.
pub fn common_params_print_completion(ctx: &ParamsContext) {
    let mut common_options = Vec::new();
    let mut sampling_options = Vec::new();
    let mut spec_options = Vec::new();
    let mut specific_options = Vec::new();
    for opt in &ctx.options {
        if opt.is_sampling {
            sampling_options.push(opt);
        } else if opt.is_spec {
            spec_options.push(opt);
        } else if opt.in_example(ctx.ex) {
            specific_options.push(opt);
        } else if !opt.is_preset_only {
            common_options.push(opt);
        }
    }

    println!("_llama_completions() {{");
    println!("    local cur prev opts");
    println!("    COMPREPLY=()");
    println!("    cur=\"${{COMP_WORDS[COMP_CWORD]}}\"");
    println!("    prev=\"${{COMP_WORDS[COMP_CWORD-1]}}\"\n");
    print!("    opts=\"");
    for opt in &common_options {
        for arg in &opt.args {
            print!("{arg} ");
        }
    }
    for opt in &sampling_options {
        for arg in &opt.args {
            print!("{arg} ");
        }
    }
    for opt in &spec_options {
        for arg in &opt.args {
            print!("{arg} ");
        }
    }
    for opt in &specific_options {
        for arg in &opt.args {
            print!("{arg} ");
        }
    }
    println!("\"\n");
    println!("    case \"$prev\" in");
    println!("        --model|-m)");
    println!("            COMPREPLY=( $(compgen -f -X '!*.gguf' -- \"$cur\") $(compgen -d -- \"$cur\") )");
    println!("            return 0");
    println!("            ;;");
    println!("        --grammar-file)");
    println!("            COMPREPLY=( $(compgen -f -X '!*.gbnf' -- \"$cur\") $(compgen -d -- \"$cur\") )");
    println!("            return 0");
    println!("            ;;");
    println!("        --chat-template-file)");
    println!("            COMPREPLY=( $(compgen -f -X '!*.jinja' -- \"$cur\") $(compgen -d -- \"$cur\") )");
    println!("            return 0");
    println!("            ;;");
    println!("        *)");
    println!("            COMPREPLY=( $(compgen -W \"${{opts}}\" -- \"$cur\") )");
    println!("            return 0");
    println!("            ;;");
    println!("    esac");
    println!("}}\n");

    let executables = [
        "llama-batched",
        "llama-batched-bench",
        "llama-bench",
        "llama-cli",
        "llama-completion",
        "llama-convert-llama2c-to-ggml",
        "llama-cvector-generator",
        "llama-debug",
        "llama-diffusion-cli",
        "llama-embedding",
        "llama-eval-callback",
        "llama-export-lora",
        "llama-finetune",
        "llama-fit-params",
        "llama-gemma3-cli",
        "llama-gen-docs",
        "llama-gguf",
        "llama-gguf-hash",
        "llama-gguf-split",
        "llama-idle",
        "llama-imatrix",
        "llama-llava-cli",
        "llama-lookahead",
        "llama-lookup",
        "llama-lookup-create",
        "llama-lookup-merge",
        "llama-lookup-stats",
        "llama-minicpmv-cli",
        "llama-mtmd-cli",
        "llama-parallel",
        "llama-passkey",
        "llama-perplexity",
        "llama-q8dot",
        "llama-quantize",
        "llama-qwen2vl-cli",
        "llama-retrieval",
        "llama-save-load-state",
        "llama-server",
        "llama-simple",
        "llama-simple-chat",
        "llama-speculative",
        "llama-speculative-simple",
        "llama-tokenize",
        "llama-tts",
        "llama-vdot",
    ];
    for exe in executables {
        println!("complete -F _llama_completions {exe}");
    }
}

/// `common_params_to_map`.
pub fn common_params_to_map(
    args: &[String],
    ex: LlamaExample,
) -> Result<std::collections::BTreeMap<CommonArg, String>> {
    let params = Params::default();
    let ctx = common_params_parser_init(&params, ex);
    let mut arg_to_options = std::collections::BTreeMap::new();
    for opt in &ctx.options {
        for arg in &opt.args {
            arg_to_options.insert((*arg).to_owned(), opt.clone());
        }
        for arg in &opt.args_neg {
            arg_to_options.insert((*arg).to_owned(), opt.clone());
        }
    }

    let mut out_map = std::collections::BTreeMap::new();
    let mut seen_args = std::collections::BTreeSet::new();
    let mut i = 1usize;
    while i < args.len() {
        let mut arg = args[i].clone();
        if arg.starts_with("--") {
            arg = arg.replace('_', "-");
        }
        let Some(opt) = arg_to_options.get(&arg) else {
            return Err(Error::new(format!("error: invalid argument: {arg}")));
        };
        if !seen_args.insert(arg.clone()) && arg != "--spec-type" {
            eprintln!("DEPRECATED: argument '{arg}' specified multiple times, use comma-separated values instead (only last value will be used)");
        }
        let mut val = String::new();
        if opt.value_hint.is_none() && opt.value_hint_2.is_none() {
            let is_neg = opt.args_neg.contains(&arg.as_str());
            val = if is_neg { "0" } else { "1" }.to_owned();
        }
        if opt.value_hint.is_some() {
            if i + 1 >= args.len() {
                return Err(Error::new("expected value for argument"));
            }
            i += 1;
            val = args[i].clone();
        }
        if opt.value_hint_2.is_some() {
            return Err(Error::new("error: argument with 2 values is not yet supported\n"));
        }
        out_map.insert(opt.clone(), val);
        i += 1;
    }
    Ok(out_map)
}

/// `common_params_add_preset_options`.
pub fn common_params_add_preset_options(args: &mut Vec<CommonArg>) {
    args.push(
        CommonArg::string(&["load-on-startup"], "NAME", "in server router mode, autoload this model on startup", |_, _| Ok(()))
            .set_env(COMMON_ARG_PRESET_LOAD_ON_STARTUP)
            .set_preset_only(),
    );
    args.push(
        CommonArg::int(&["stop-timeout"], "SECONDS", "in server router mode, force-kill model instance after this many seconds of graceful shutdown", |_, _| Ok(()))
            .set_env(COMMON_ARG_PRESET_STOP_TIMEOUT)
            .set_preset_only(),
    );
    args.push(
        CommonArg::string(&["dedup-cache-models"], "0|1", "in server router mode, hide a cached model from the model list when this preset resolves to the same model file", |_, _| Ok(()))
            .set_env(COMMON_ARG_PRESET_DEDUP_CACHE_MODELS)
            .set_preset_only(),
    );
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthy_falsey_autoy() {
        assert!(is_truthy("on") && is_truthy("TRUE") == false);
        assert!(is_falsey("0") && is_falsey("disabled"));
        assert!(is_autoy("auto") && is_autoy("-1"));
        assert!(parse_bool_value("true").unwrap());
        assert!(!parse_bool_value("false").unwrap());
        assert!(parse_bool_value("maybe").is_err());
    }

    #[test]
    fn csv_parser() {
        assert_eq!(
            parse_csv_row("a,\"b, c\",\"d \"\"e\"\" f\""),
            vec!["a", "b, c", "d \"e\" f"]
        );
    }

    #[test]
    fn arg_helpers() {
        let opt = CommonArg::string(&["-m", "--model"], "FNAME", "model path to load", |p, v| {
            p.model.path = v;
            Ok(())
        });
        assert_eq!(opt.get_args(), vec!["-m", "--model"]);
        assert!(opt.get_env().is_empty());
        assert!(opt.to_string().contains("-m,") && opt.to_string().contains("--model"));
        let neg = CommonArg::boolean(&["--perf"], &["--no-perf"], "perf", |p, v| {
            p.no_perf = !v;
            Ok(())
        })
        .set_env("LLAMA_ARG_PERF");
        assert_eq!(neg.get_env(), vec!["LLAMA_ARG_PERF", "LLAMA_ARG_NO_PERF"]);
    }

    #[test]
    fn parse_basic_args() {
        let args: Vec<String> = ["prog", "-m", "model.gguf", "-c", "512", "--temp", "0.5", "-np", "2"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let mut params = Params::default();
        let outcome = common_params_parse(&args, &mut params, LlamaExample::Common).unwrap();
        assert_eq!(outcome, ParseOutcome::Parsed);
        assert_eq!(params.model.path, "model.gguf");
        assert_eq!(params.n_ctx, 512);
        assert!((params.sampling.temp - 0.5).abs() < 1e-6);
        assert_eq!(params.n_parallel, 2);
        assert!(params.default_template_kwargs.contains_key("preserve_reasoning"));

        let bad: Vec<String> = ["prog", "--not-an-arg"].iter().map(|s| (*s).to_owned()).collect();
        assert!(common_params_parse(&bad, &mut params, LlamaExample::Common).is_err());
        assert_eq!(params.model.path, "model.gguf");
    }

    #[test]
    fn bool_and_prefixed_bool() {
        let args: Vec<String> = ["prog", "-m", "x.gguf", "--display-prompt", "--no-escape"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let mut params = Params::default();
        common_params_parse(&args, &mut params, LlamaExample::Cli).unwrap();
        assert!(params.display_prompt);
        assert!(!params.escape);
    }

    #[test]
    fn to_map_and_preset_options() {
        let args: Vec<String> = ["prog", "--top-k", "10", "--no-perf"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let map = common_params_to_map(&args, LlamaExample::Common).unwrap();
        assert_eq!(map.len(), 2);
        let mut opts = Vec::new();
        common_params_add_preset_options(&mut opts);
        assert_eq!(opts.len(), 3);
        assert!(opts.iter().all(|o| o.is_preset_only));
    }
}

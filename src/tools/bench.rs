//! Rust port of `tools/llama-bench/llama-bench.cpp` and `tools/llama-bench/main.cpp`.
//!
//! This is a dry port. There are no live model, context, backend or threadpool
//! handles, so command line parsing, parameter expansion, model/context
//! parameter mapping, timing maths and all five result printers are translated
//! as data. The caller supplies the work being timed through `run_benchmark`
//! and model metadata through `TestMeta`. Hugging Face download, backend
//! enumeration, memory fitting and `test_prompt`/`test_gen` are not ported.

use crate::api::{
    BackendDevice, Buft, ContextParams, FlashAttnType, GgmlType, LazyMode, LoadMode, ModelParams,
    NumaStrategy, SplitMode, TensorBuftOverride as ApiTensorBuftOverride,
};
use crate::common::{
    common_cpu_get_num_math, llm_ffn_block_regex, string_split, string_strip, TensorBuftOverride,
    LLM_FFN_EXPS_REGEX, LLAMA_MAX_DEVICES,
};
use crate::core::{Error, Result};
use crate::ggml::threading::SchedPriority;
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// `llama_commit()` for this port.
pub const BUILD_COMMIT: &str = crate::api::LLAMA_VERSION;
/// `llama_build_number()` for this port.
pub const BUILD_NUMBER: i32 = 0;

// ---------------------------------------------------------------------------
// utility
// ---------------------------------------------------------------------------

fn io_error(e: io::Error) -> Error {
    Error::new(e.to_string())
}

fn invalid(arg: &str) -> Error {
    Error::new(format!("invalid parameter for argument: {arg}"))
}

fn next_value<'a>(args: &'a [String], i: &mut usize, arg: &str) -> Result<&'a str> {
    *i += 1;
    args.get(*i).map(String::as_str).ok_or_else(|| invalid(arg))
}

fn joined<T: std::fmt::Display>(values: &[T]) -> String {
    values.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}

fn joined_bool(values: &[bool]) -> String {
    values
        .iter()
        .map(|&v| if v { "1" } else { "0" })
        .collect::<Vec<_>>()
        .join(",")
}

fn pair_str(p: (i32, i32)) -> String {
    format!("{},{}", p.0, p.1)
}

/// `output_format_str` / `output_format_from_str`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    None,
    Csv,
    Json,
    Jsonl,
    Markdown,
    Sql,
}

impl OutputFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Csv => "csv",
            Self::Json => "json",
            Self::Jsonl => "jsonl",
            Self::Markdown => "md",
            Self::Sql => "sql",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "csv" => Some(Self::Csv),
            "json" => Some(Self::Json),
            "jsonl" => Some(Self::Jsonl),
            "md" => Some(Self::Markdown),
            "sql" => Some(Self::Sql),
            _ => None,
        }
    }
}

/// `ggml_type_name`.
pub fn ggml_type_name(t: GgmlType) -> &'static str {
    match t {
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::Q4_0 => "q4_0",
        GgmlType::Q4_1 => "q4_1",
        GgmlType::Q5_0 => "q5_0",
        GgmlType::Q5_1 => "q5_1",
        GgmlType::Q8_0 => "q8_0",
        GgmlType::Q8_1 => "q8_1",
        GgmlType::Q2K => "q2_K",
        GgmlType::Q3K => "q3_K",
        GgmlType::Q4K => "q4_K",
        GgmlType::Q5K => "q5_K",
        GgmlType::Q6K => "q6_K",
        GgmlType::Q8K => "q8_K",
        GgmlType::Iq2Xxs => "iq2_xxs",
        GgmlType::Iq2Xs => "iq2_xs",
        GgmlType::Iq3Xxs => "iq3_xxs",
        GgmlType::Iq1S => "iq1_s",
        GgmlType::Iq4Nl => "iq4_nl",
        GgmlType::Iq3S => "iq3_s",
        GgmlType::Iq2S => "iq2_s",
        GgmlType::Iq4Xs => "iq4_xs",
        GgmlType::I8 => "i8",
        GgmlType::I16 => "i16",
        GgmlType::I32 => "i32",
        GgmlType::I64 => "i64",
        GgmlType::F64 => "f64",
        GgmlType::Iq1M => "iq1_m",
        GgmlType::Bf16 => "bf16",
        GgmlType::Tq1_0 => "tq1_0",
        GgmlType::Tq2_0 => "tq2_0",
        GgmlType::Mxfp4 => "mxfp4",
        GgmlType::Nvfp4 => "nvfp4",
        GgmlType::Q1_0 => "q1_0",
        GgmlType::Q2_0 => "q2_0",
        GgmlType::F8E4M3 => "f8_e4m3",
        GgmlType::Count => "unknown",
    }
}

/// `ggml_type_from_name`, limited to the cache types accepted by llama-bench.
pub fn ggml_type_from_name(s: &str) -> Option<GgmlType> {
    match s {
        "f16" => Some(GgmlType::F16),
        "bf16" => Some(GgmlType::Bf16),
        "f8_e4m3" => Some(GgmlType::F8E4M3),
        "q8_0" => Some(GgmlType::Q8_0),
        "q4_0" => Some(GgmlType::Q4_0),
        "q4_1" => Some(GgmlType::Q4_1),
        "q5_0" => Some(GgmlType::Q5_0),
        "q5_1" => Some(GgmlType::Q5_1),
        "iq4_nl" => Some(GgmlType::Iq4Nl),
        _ => None,
    }
}

pub fn split_mode_str(mode: SplitMode) -> &'static str {
    match mode {
        SplitMode::None => "none",
        SplitMode::Layer => "layer",
        SplitMode::Row => "row",
        SplitMode::Tensor => "tensor",
    }
}

pub fn lazy_mode_str(mode: LazyMode) -> &'static str {
    match mode {
        LazyMode::Off => "off",
        LazyMode::Auto => "auto",
        LazyMode::On => "on",
    }
}

pub fn load_mode_name(mode: LoadMode) -> &'static str {
    match mode {
        LoadMode::Auto => "auto",
        LoadMode::None => "none",
        LoadMode::Mmap => "mmap",
        LoadMode::Mlock => "mlock",
        LoadMode::MmapMlock => "mmap+mlock",
        LoadMode::DirectIo => "dio",
    }
}

pub fn flash_attn_type_name(t: FlashAttnType) -> &'static str {
    match t {
        FlashAttnType::Auto => "auto",
        FlashAttnType::Disabled => "disabled",
        FlashAttnType::Enabled => "enabled",
    }
}

fn is_truthy(value: &str) -> bool {
    matches!(value, "on" | "enabled" | "true" | "1")
}

fn is_falsey(value: &str) -> bool {
    matches!(value, "off" | "disabled" | "false" | "0")
}

fn is_autoy(value: &str) -> bool {
    matches!(value, "auto" | "-1")
}

fn sched_priority_from_i32(value: i32) -> Option<SchedPriority> {
    match value {
        -1 => Some(SchedPriority::Low),
        0 => Some(SchedPriority::Normal),
        1 => Some(SchedPriority::Medium),
        2 => Some(SchedPriority::High),
        3 => Some(SchedPriority::Realtime),
        _ => None,
    }
}

fn sched_priority_as_i32(prio: &SchedPriority) -> i32 {
    match prio {
        SchedPriority::Low => -1,
        SchedPriority::Normal => 0,
        SchedPriority::Medium => 1,
        SchedPriority::High => 2,
        SchedPriority::Realtime => 3,
    }
}

fn parse_bool_list(s: &str) -> Result<Vec<bool>> {
    s.split(',')
        .map(|v| match v {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => Err(Error::new(format!("invalid boolean: \"{v}\""))),
        })
        .collect()
}

/// `parse_int_range`: `first[-last[(+|*)step]]`, repeated across commas.
///
/// Like the C++ regex search, unmatched text before a number is skipped.
pub fn parse_int_range(s: &str, allow_negative: bool) -> Result<Vec<i32>> {
    let bytes = s.as_bytes();
    let mut pos = 0usize;
    let mut result = Vec::new();

    while pos < bytes.len() {
        // regex_search skips leading text until a number can start.
        while pos < bytes.len() {
            let digit = bytes[pos].is_ascii_digit();
            let neg = allow_negative && bytes[pos] == b'-' && pos + 1 < bytes.len() && bytes[pos + 1].is_ascii_digit();
            if digit || neg {
                break;
            }
            pos += 1;
        }
        if pos >= bytes.len() {
            // C++ fails here only if the unconsumed suffix is not empty.
            return Err(Error::new("invalid range format"));
        }

        let start = pos;
        let negative = bytes[pos] == b'-';
        if negative {
            pos += 1;
        }
        let first_start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        let mut first = s[first_start..pos].parse::<i32>().map_err(|_| Error::new("invalid range format"))?;
        if negative {
            first = -first;
        }

        let mut last = first;
        let mut op = '+';
        let mut step = 1i32;
        if pos < bytes.len() && bytes[pos] == b'-' && pos + 1 < bytes.len() && bytes[pos + 1].is_ascii_digit() {
            pos += 1;
            let last_start = pos;
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            last = s[last_start..pos].parse::<i32>().map_err(|_| Error::new("invalid range format"))?;
            if pos + 1 < bytes.len() && matches!(bytes[pos], b'+' | b'*') && bytes[pos + 1].is_ascii_digit() {
                op = bytes[pos] as char;
                pos += 1;
                let step_start = pos;
                while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                    pos += 1;
                }
                step = s[step_start..pos].parse::<i32>().map_err(|_| Error::new("invalid range format"))?;
            }
        }

        // The regex match ends at a comma or at the end of the string.
        if pos < bytes.len() && bytes[pos] != b',' {
            return Err(Error::new("invalid range format"));
        }
        if pos == start {
            return Err(Error::new("invalid range format"));
        }
        if pos < bytes.len() {
            pos += 1;
        }

        let mut i = first;
        loop {
            if i > last {
                break;
            }
            result.push(i);
            let prev = i;
            i = match op {
                '+' => i.checked_add(step).ok_or_else(|| Error::new("invalid range"))?,
                '*' => i.checked_mul(step).ok_or_else(|| Error::new("invalid range"))?,
                _ => return Err(Error::new("invalid range format")),
            };
            if i <= prev {
                return Err(Error::new("invalid range"));
            }
        }
    }

    Ok(result)
}

/// `parse_devices_arg`. Device names stand in for backend device handles.
pub fn parse_devices_arg(value: &str) -> Result<Vec<String>> {
    let trimmed = string_strip(value);
    if trimmed.is_empty() {
        return Err(Error::new("no devices specified"));
    }
    if trimmed == "auto" {
        return Ok(Vec::new());
    }

    let mut devices = Vec::new();
    for name in trimmed.split('/') {
        let dev_name = string_strip(name);
        if dev_name.is_empty() {
            return Err(Error::new("invalid device specification"));
        }
        devices.push(dev_name.to_owned());
    }
    if devices.len() == 1 && devices[0] == "none" {
        return Ok(vec!["none".to_owned()]);
    }
    Ok(devices)
}

fn devices_to_string(devices: &[String]) -> String {
    if devices.is_empty() {
        return "auto".to_owned();
    }
    if devices.len() == 1 && devices[0] == "none" {
        return "none".to_owned();
    }
    devices.join("/")
}

// ---------------------------------------------------------------------------
// command line params
// ---------------------------------------------------------------------------

/// `cmd_params`.
#[derive(Clone, Debug)]
pub struct CmdParams {
    pub model: Vec<String>,
    pub hf_repo: Vec<String>,
    pub hf_file: Vec<String>,
    pub hf_token: String,
    pub offline: bool,
    pub n_prompt: Vec<i32>,
    pub n_gen: Vec<i32>,
    pub n_pg: Vec<(i32, i32)>,
    pub n_depth: Vec<i32>,
    pub n_batch: Vec<i32>,
    pub n_ubatch: Vec<i32>,
    pub type_k: Vec<GgmlType>,
    pub type_v: Vec<GgmlType>,
    pub n_threads: Vec<i32>,
    pub cpu_mask: Vec<String>,
    pub cpu_strict: Vec<bool>,
    pub poll: Vec<i32>,
    pub n_gpu_layers: Vec<i32>,
    pub n_cpu_moe: Vec<i32>,
    pub split_mode: Vec<SplitMode>,
    pub load_mode: Vec<LoadMode>,
    pub lazy_mode: Vec<LazyMode>,
    pub main_gpu: Vec<i32>,
    pub no_kv_offload: Vec<bool>,
    pub flash_attn: Vec<FlashAttnType>,
    pub devices: Vec<Vec<String>>,
    pub tensor_split: Vec<Vec<f32>>,
    pub tensor_buft_overrides: Vec<Vec<TensorBuftOverride>>,
    pub embeddings: Vec<bool>,
    pub no_op_offload: Vec<bool>,
    pub no_host: Vec<bool>,
    pub repack: Vec<bool>,
    pub fit_params_target: Vec<usize>,
    pub fit_params_min_ctx: Vec<u32>,
    pub numa: NumaStrategy,
    pub reps: i32,
    pub prio: SchedPriority,
    pub delay: i32,
    pub verbose: bool,
    pub progress: bool,
    pub no_warmup: bool,
    pub output_format: OutputFormat,
    pub output_format_stderr: OutputFormat,
}

impl Default for CmdParams {
    fn default() -> Self {
        Self {
            model: vec!["models/7B/ggml-model-q4_0.gguf".to_owned()],
            hf_repo: Vec::new(),
            hf_file: Vec::new(),
            hf_token: String::new(),
            offline: false,
            n_prompt: vec![512],
            n_gen: vec![128],
            n_pg: Vec::new(),
            n_depth: vec![0],
            n_batch: vec![2048],
            n_ubatch: vec![512],
            type_k: vec![GgmlType::F16],
            type_v: vec![GgmlType::F16],
            n_threads: vec![common_cpu_get_num_math()],
            cpu_mask: vec!["0x0".to_owned()],
            cpu_strict: vec![false],
            poll: vec![50],
            n_gpu_layers: vec![-1],
            n_cpu_moe: vec![0],
            split_mode: vec![SplitMode::Layer],
            load_mode: vec![LoadMode::Auto],
            lazy_mode: vec![LazyMode::Auto],
            main_gpu: vec![0],
            no_kv_offload: vec![false],
            flash_attn: vec![FlashAttnType::Auto],
            devices: vec![Vec::new()],
            tensor_split: vec![vec![0.0; LLAMA_MAX_DEVICES]],
            tensor_buft_overrides: vec![vec![TensorBuftOverride::default()]],
            embeddings: vec![false],
            no_op_offload: vec![false],
            no_host: vec![false],
            repack: vec![ModelParams::default().use_extra_bufts],
            fit_params_target: vec![0],
            fit_params_min_ctx: vec![0],
            numa: NumaStrategy::Disabled,
            reps: 5,
            prio: SchedPriority::Normal,
            delay: 0,
            verbose: false,
            progress: false,
            no_warmup: false,
            output_format: OutputFormat::Markdown,
            output_format_stderr: OutputFormat::None,
        }
    }
}

impl CmdParams {
    /// Parser start state: scalar defaults, but every list empty.
    fn blank() -> Self {
        let d = Self::default();
        Self {
            model: Vec::new(),
            hf_repo: Vec::new(),
            hf_file: Vec::new(),
            n_prompt: Vec::new(),
            n_gen: Vec::new(),
            n_pg: Vec::new(),
            n_depth: Vec::new(),
            n_batch: Vec::new(),
            n_ubatch: Vec::new(),
            type_k: Vec::new(),
            type_v: Vec::new(),
            n_threads: Vec::new(),
            cpu_mask: Vec::new(),
            cpu_strict: Vec::new(),
            poll: Vec::new(),
            n_gpu_layers: Vec::new(),
            n_cpu_moe: Vec::new(),
            split_mode: Vec::new(),
            load_mode: Vec::new(),
            lazy_mode: Vec::new(),
            main_gpu: Vec::new(),
            no_kv_offload: Vec::new(),
            flash_attn: Vec::new(),
            devices: Vec::new(),
            tensor_split: Vec::new(),
            tensor_buft_overrides: Vec::new(),
            embeddings: Vec::new(),
            no_op_offload: Vec::new(),
            no_host: Vec::new(),
            repack: Vec::new(),
            fit_params_target: Vec::new(),
            fit_params_min_ctx: Vec::new(),
            ..d
        }
    }

    /// Fill vector fields that no command line value reached.
    fn apply_defaults(&mut self) {
        macro_rules! set_default {
            ($field:ident) => {
                if self.$field.is_empty() {
                    self.$field = Self::default().$field;
                }
            };
        }
        set_default!(model);
        set_default!(n_prompt);
        set_default!(n_gen);
        set_default!(n_depth);
        set_default!(n_batch);
        set_default!(n_ubatch);
        set_default!(type_k);
        set_default!(type_v);
        set_default!(n_gpu_layers);
        set_default!(n_cpu_moe);
        set_default!(split_mode);
        set_default!(load_mode);
        set_default!(lazy_mode);
        set_default!(main_gpu);
        set_default!(no_kv_offload);
        set_default!(flash_attn);
        set_default!(devices);
        set_default!(tensor_split);
        set_default!(tensor_buft_overrides);
        set_default!(embeddings);
        set_default!(no_op_offload);
        set_default!(no_host);
        set_default!(repack);
        set_default!(n_threads);
        set_default!(cpu_mask);
        set_default!(cpu_strict);
        set_default!(poll);
        set_default!(fit_params_target);
        set_default!(fit_params_min_ctx);
    }
}

/// Result of `parse_cmd_params`: parameters, or an early help/version exit.
#[derive(Clone, Debug)]
pub enum ParseOutcome {
    Params(CmdParams),
    Exit,
}

/// `print_usage`.
pub fn print_usage(program: &str) {
    let d = CmdParams::default();
    println!("usage: {program} [options]");
    println!();
    println!("options:");
    println!("  -h, --help");
    println!("  --version                                   show version and build info");
    println!("  --numa <distribute|isolate|numactl>         numa mode (default: disabled)");
    println!("  -r, --repetitions <n>                       number of times to repeat each test (default: {})", d.reps);
    println!("  --prio <-1|0|1|2|3>                         process/thread priority (default: {})", sched_priority_as_i32(&d.prio));
    println!("  --delay <0...N> (seconds)                   delay between each test (default: {})", d.delay);
    println!("  -o, --output <csv|json|jsonl|md|sql>        output format printed to stdout (default: {})", d.output_format.as_str());
    println!("  -oe, --output-err <csv|json|jsonl|md|sql>   output format printed to stderr (default: {})", d.output_format_stderr.as_str());
    println!("  --list-devices                              list available devices and exit");
    println!("  -v, --verbose                               verbose output");
    println!("  --progress                                  print test progress indicators");
    println!("  --no-warmup                                 skip warmup runs before benchmarking");
    println!("  -fitt, --fit-target <MiB>                   fit model to device memory with this margin per device in MiB (default: off)");
    println!("  -fitc, --fit-ctx <n>                        minimum ctx size for --fit-target (default: 0)");
    println!();
    println!("test parameters:");
    println!("  -m, --model <filename>                            (default: {})", d.model.join(","));
    println!("  -hf, -hfr, --hf-repo <user>/<model>[:quant]       Hugging Face model repository; quant is optional, case-insensitive");
    println!("                                                    default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist.");
    println!("                                                    example: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M");
    println!("                                                    (default: unused)");
    println!("  -hff, --hf-file <file>                            Hugging Face model file. If specified, it will override the quant in --hf-repo");
    println!("                                                    (default: unused)");
    println!("  -hft, --hf-token <token>                          Hugging Face access token");
    println!("                                                    (default: value from HF_TOKEN environment variable)");
    println!("  --offline                                         Offline mode: forces use of cache, prevents network access");
    println!("                                                    (default: disabled)");
    println!("  -p, --n-prompt <n>                                (default: {})", joined(&d.n_prompt));
    println!("  -n, --n-gen <n>                                   (default: {})", joined(&d.n_gen));
    println!("  -pg <pp,tg>                                       (default: {})", d.n_pg.iter().map(|&p| pair_str(p)).collect::<Vec<_>>().join(","));
    println!("  -d, --n-depth <n>                                 (default: {})", joined(&d.n_depth));
    println!("  -b, --batch-size <n>                              (default: {})", joined(&d.n_batch));
    println!("  -ub, --ubatch-size <n>                            (default: {})", joined(&d.n_ubatch));
    println!("  -ctk, --cache-type-k <t>                          (default: {})", d.type_k.iter().map(|&t| ggml_type_name(t)).collect::<Vec<_>>().join(","));
    println!("  -ctv, --cache-type-v <t>                          (default: {})", d.type_v.iter().map(|&t| ggml_type_name(t)).collect::<Vec<_>>().join(","));
    println!("  -t, --threads <n>                                 (default: {})", joined(&d.n_threads));
    println!("  -C, --cpu-mask <hex,hex>                          (default: {})", d.cpu_mask.join(","));
    println!("  --cpu-strict <0|1>                                (default: {})", joined_bool(&d.cpu_strict));
    println!("  --poll <0...100>                                  (default: {})", joined(&d.poll));
    println!("  -ngl, --n-gpu-layers <n>                          (default: {})", joined(&d.n_gpu_layers));
    println!("  -ncmoe, --n-cpu-moe <n>                           (default: {})", joined(&d.n_cpu_moe));
    println!("  -sm, --split-mode <none|layer|row|tensor>         (default: {})", d.split_mode.iter().map(|&m| split_mode_str(m)).collect::<Vec<_>>().join(","));
    println!("  -mg, --main-gpu <i>                               (default: {})", joined(&d.main_gpu));
    println!("  -nkvo, --no-kv-offload <0|1>                      (default: {})", joined_bool(&d.no_kv_offload));
    println!("  -fa, --flash-attn <on|off|auto>                   (default: {})", d.flash_attn.iter().map(|&t| flash_attn_type_name(t)).collect::<Vec<_>>().join(","));
    println!("  -dev, --device <dev0/dev1/...>                    (default: auto)");
    println!("  -lm, --load-mode <auto|none|mmap|mlock|mmap+mlock|dio> (default: {})", d.load_mode.iter().map(|&m| load_mode_name(m)).collect::<Vec<_>>().join(","));
    println!("  -lzm, --lazy-mode <on|auto|off>                   (default: {})", d.lazy_mode.iter().map(|&m| lazy_mode_str(m)).collect::<Vec<_>>().join(","));
    println!("  -embd, --embeddings <0|1>                         (default: {})", joined_bool(&d.embeddings));
    println!("  -ts, --tensor-split <ts0/ts1/..>                  (default: 0)");
    println!("  -ot --override-tensor <tensor name pattern>=<buffer type>;...");
    println!("                                                    (default: disabled)");
    println!("  -nopo, --no-op-offload <0|1>                      (default: 0)");
    println!("  --no-host <0|1>                                   (default: {})", joined_bool(&d.no_host));
    println!("  --repack <0|1>                                    (default: {})", joined_bool(&d.repack));
    println!();
    println!(
        "Multiple values can be given for each parameter by separating them with ','\nor by specifying the parameter multiple times. Ranges can be given as\n'first-last' or 'first-last+step' or 'first-last*mult'."
    );
}

/// `parse_cmd_params`. `args` excludes the program name.
pub fn parse_cmd_params(program: &str, args: &[String]) -> Result<ParseOutcome> {
    let mut p = CmdParams::blank();
    p.hf_token = std::env::var("HF_TOKEN").unwrap_or_default();

    let mut i = 0usize;
    while i < args.len() {
        let mut arg = args[i].clone();
        if arg.starts_with("--") {
            arg = arg.replace('_', "-");
        }

        match arg.as_str() {
            "-h" | "--help" => {
                print_usage(program);
                return Ok(ParseOutcome::Exit);
            }
            "--version" => {
                println!("{}", crate::api::llama_version());
                return Ok(ParseOutcome::Exit);
            }
            "-m" | "--model" => p.model.extend(string_split(next_value(args, &mut i, &arg)?, ",")),
            "-hf" | "-hfr" | "--hf-repo" => p.hf_repo.extend(string_split(next_value(args, &mut i, &arg)?, ",")),
            "-hff" | "--hf-file" => p.hf_file.extend(string_split(next_value(args, &mut i, &arg)?, ",")),
            "-hft" | "--hf-token" => p.hf_token = next_value(args, &mut i, &arg)?.to_owned(),
            "--offline" => p.offline = true,
            "-p" | "--n-prompt" => p.n_prompt.extend(parse_int_range(next_value(args, &mut i, &arg)?, false)?),
            "-n" | "--n-gen" => p.n_gen.extend(parse_int_range(next_value(args, &mut i, &arg)?, false)?),
            "-pg" => {
                let parts: Vec<&str> = next_value(args, &mut i, &arg)?.split(',').collect();
                if parts.len() != 2 {
                    return Err(invalid(&arg));
                }
                let pp = parts[0].parse::<i32>().map_err(|_| invalid(&arg))?;
                let tg = parts[1].parse::<i32>().map_err(|_| invalid(&arg))?;
                p.n_pg.push((pp, tg));
            }
            "-d" | "--n-depth" => p.n_depth.extend(parse_int_range(next_value(args, &mut i, &arg)?, false)?),
            "-b" | "--batch-size" => p.n_batch.extend(parse_int_range(next_value(args, &mut i, &arg)?, false)?),
            "-ub" | "--ubatch-size" => p.n_ubatch.extend(parse_int_range(next_value(args, &mut i, &arg)?, false)?),
            "-ctk" | "--cache-type-k" => {
                let mut types = Vec::new();
                for t in string_split(next_value(args, &mut i, &arg)?, ",") {
                    match ggml_type_from_name(&t) {
                        Some(gt) => types.push(gt),
                        None => return Err(invalid(&arg)),
                    }
                }
                p.type_k.extend(types);
            }
            "-ctv" | "--cache-type-v" => {
                let mut types = Vec::new();
                for t in string_split(next_value(args, &mut i, &arg)?, ",") {
                    match ggml_type_from_name(&t) {
                        Some(gt) => types.push(gt),
                        None => return Err(invalid(&arg)),
                    }
                }
                p.type_v.extend(types);
            }
            "-dev" | "--device" => {
                for combo in string_split(next_value(args, &mut i, &arg)?, ",") {
                    p.devices.push(parse_devices_arg(&combo)?);
                }
            }
            "--list-devices" => {
                println!("no backend devices in the Rust port");
                return Ok(ParseOutcome::Exit);
            }
            "-t" | "--threads" => p.n_threads.extend(parse_int_range(next_value(args, &mut i, &arg)?, false)?),
            "-C" | "--cpu-mask" => p.cpu_mask.extend(string_split(next_value(args, &mut i, &arg)?, ",")),
            "--cpu-strict" => p.cpu_strict.extend(parse_bool_list(next_value(args, &mut i, &arg)?)?),
            "--poll" => p.poll.extend(parse_int_range(next_value(args, &mut i, &arg)?, false)?),
            "-ngl" | "--n-gpu-layers" => p.n_gpu_layers.extend(parse_int_range(next_value(args, &mut i, &arg)?, true)?),
            "-ncmoe" | "--n-cpu-moe" => p.n_cpu_moe.extend(parse_int_range(next_value(args, &mut i, &arg)?, false)?),
            "-sm" | "--split-mode" => {
                let mut modes = Vec::new();
                for m in string_split(next_value(args, &mut i, &arg)?, ",") {
                    let mode = match m.as_str() {
                        "none" => SplitMode::None,
                        "layer" => SplitMode::Layer,
                        "row" => SplitMode::Row,
                        "tensor" => SplitMode::Tensor,
                        _ => return Err(invalid(&arg)),
                    };
                    modes.push(mode);
                }
                p.split_mode.extend(modes);
            }
            "-lm" | "--load-mode" => {
                let mut modes = Vec::new();
                for m in string_split(next_value(args, &mut i, &arg)?, ",") {
                    let mode = match m.as_str() {
                        "auto" => LoadMode::Auto,
                        "none" => LoadMode::None,
                        "mmap" => LoadMode::Mmap,
                        "mlock" => LoadMode::Mlock,
                        "mmap+mlock" => LoadMode::MmapMlock,
                        "dio" => LoadMode::DirectIo,
                        _ => return Err(invalid(&arg)),
                    };
                    modes.push(mode);
                }
                p.load_mode.extend(modes);
            }
            "-lzm" | "--lazy-mode" => {
                let mut modes = Vec::new();
                for m in string_split(next_value(args, &mut i, &arg)?, ",") {
                    let mode = match m.as_str() {
                        "on" => LazyMode::On,
                        "auto" => LazyMode::Auto,
                        "off" => LazyMode::Off,
                        _ => return Err(invalid(&arg)),
                    };
                    modes.push(mode);
                }
                p.lazy_mode.extend(modes);
            }
            "-mg" | "--main-gpu" => p.main_gpu = parse_int_range(next_value(args, &mut i, &arg)?, false)?,
            "-nkvo" | "--no-kv-offload" => p.no_kv_offload.extend(parse_bool_list(next_value(args, &mut i, &arg)?)?),
            "--numa" => {
                p.numa = match next_value(args, &mut i, &arg)? {
                    "distribute" | "" => NumaStrategy::Distribute,
                    "isolate" => NumaStrategy::Isolate,
                    "numactl" => NumaStrategy::Numactl,
                    _ => return Err(invalid(&arg)),
                };
            }
            "-fa" | "--flash-attn" => {
                let mut types = Vec::new();
                for v in string_split(next_value(args, &mut i, &arg)?, ",") {
                    let t = if is_truthy(&v) {
                        FlashAttnType::Enabled
                    } else if is_falsey(&v) {
                        FlashAttnType::Disabled
                    } else if is_autoy(&v) {
                        FlashAttnType::Auto
                    } else {
                        return Err(invalid(&arg));
                    };
                    types.push(t);
                }
                p.flash_attn.extend(types);
            }
            "-embd" | "--embeddings" => p.embeddings.extend(parse_bool_list(next_value(args, &mut i, &arg)?)?),
            "-nopo" | "--no-op-offload" => p.no_op_offload.extend(parse_bool_list(next_value(args, &mut i, &arg)?)?),
            "--no-host" => p.no_host.extend(parse_bool_list(next_value(args, &mut i, &arg)?)?),
            "--repack" => p.repack.extend(parse_bool_list(next_value(args, &mut i, &arg)?)?),
            "-ts" | "--tensor-split" => {
                for ts in string_split(next_value(args, &mut i, &arg)?, ",") {
                    let parts: Vec<&str> = ts.split(|c: char| c == ';' || c == '/').collect();
                    if parts.len() > LLAMA_MAX_DEVICES {
                        return Err(Error::new("too many tensor split values"));
                    }
                    let mut split = vec![0.0f32; LLAMA_MAX_DEVICES];
                    for (idx, part) in parts.iter().enumerate() {
                        split[idx] = part.parse::<f32>().map_err(|_| invalid(&arg))?;
                    }
                    p.tensor_split.push(split);
                }
            }
            "-ot" | "--override-tensor" => {
                let groups = next_value(args, &mut i, &arg)?.to_owned();
                for group in groups.split(',') {
                    let mut overrides = Vec::new();
                    for entry in group.split(';') {
                        if entry.is_empty() {
                            continue;
                        }
                        let mut kv = entry.splitn(2, '=');
                        let pattern = kv.next().unwrap_or("");
                        let buft = match kv.next() {
                            Some(b) if !b.is_empty() => b,
                            _ => return Err(invalid(&arg)),
                        };
                        overrides.push(TensorBuftOverride {
                            pattern: pattern.to_owned(),
                            buffer_type: buft.to_owned(),
                        });
                    }
                    // The last entry is always the empty (null) terminator.
                    overrides.push(TensorBuftOverride::default());
                    p.tensor_buft_overrides.push(overrides);
                }
            }
            "-r" | "--repetitions" => {
                p.reps = next_value(args, &mut i, &arg)?.parse::<i32>().map_err(|_| invalid(&arg))?;
            }
            "--prio" => {
                let prio = next_value(args, &mut i, &arg)?.parse::<i32>().map_err(|_| invalid(&arg))?;
                p.prio = sched_priority_from_i32(prio).ok_or_else(|| invalid(&arg))?;
            }
            "--delay" => {
                p.delay = next_value(args, &mut i, &arg)?.parse::<i32>().map_err(|_| invalid(&arg))?;
            }
            "-o" | "--output" => {
                p.output_format = OutputFormat::from_str(next_value(args, &mut i, &arg)?).ok_or_else(|| invalid(&arg))?;
            }
            "-oe" | "--output-err" => {
                p.output_format_stderr = OutputFormat::from_str(next_value(args, &mut i, &arg)?).ok_or_else(|| invalid(&arg))?;
            }
            "-v" | "--verbose" => p.verbose = true,
            "--progress" => p.progress = true,
            "--no-warmup" => p.no_warmup = true,
            "-fitt" | "--fit-target" => {
                for v in next_value(args, &mut i, &arg)?.split(',') {
                    p.fit_params_target.push(v.parse::<usize>().map_err(|_| invalid(&arg))?);
                }
            }
            "-fitc" | "--fit-ctx" => {
                for v in next_value(args, &mut i, &arg)?.split(',') {
                    p.fit_params_min_ctx.push(v.parse::<u32>().map_err(|_| invalid(&arg))?);
                }
            }
            _ => return Err(invalid(&arg)),
        }

        i += 1;
    }

    p.apply_defaults();
    Ok(ParseOutcome::Params(p))
}

// ---------------------------------------------------------------------------
// parameter instances
// ---------------------------------------------------------------------------

/// `cmd_params_instance`.
#[derive(Clone, Debug, PartialEq)]
pub struct CmdParamsInstance {
    pub model: String,
    pub n_prompt: i32,
    pub n_gen: i32,
    pub n_depth: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub type_k: GgmlType,
    pub type_v: GgmlType,
    pub n_threads: i32,
    pub cpu_mask: String,
    pub cpu_strict: bool,
    pub poll: i32,
    pub n_gpu_layers: i32,
    pub n_cpu_moe: i32,
    pub split_mode: SplitMode,
    pub load_mode: LoadMode,
    pub lazy_mode: LazyMode,
    pub main_gpu: i32,
    pub no_kv_offload: bool,
    pub flash_attn: FlashAttnType,
    pub devices: Vec<String>,
    pub tensor_split: Vec<f32>,
    pub tensor_buft_overrides: Vec<TensorBuftOverride>,
    pub embeddings: bool,
    pub no_op_offload: bool,
    pub no_host: bool,
    pub repack: bool,
    pub fit_target: usize,
    pub fit_min_ctx: u32,
}

impl CmdParamsInstance {
    /// Placeholder values; `get_cmd_params_instances` replaces every field.
    fn seed() -> Self {
        Self {
            model: String::new(),
            n_prompt: 0,
            n_gen: 0,
            n_depth: 0,
            n_batch: 0,
            n_ubatch: 0,
            type_k: GgmlType::F16,
            type_v: GgmlType::F16,
            n_threads: 0,
            cpu_mask: String::new(),
            cpu_strict: false,
            poll: 0,
            n_gpu_layers: 0,
            n_cpu_moe: 0,
            split_mode: SplitMode::Layer,
            load_mode: LoadMode::Auto,
            lazy_mode: LazyMode::Auto,
            main_gpu: 0,
            no_kv_offload: false,
            flash_attn: FlashAttnType::Auto,
            devices: Vec::new(),
            tensor_split: Vec::new(),
            tensor_buft_overrides: Vec::new(),
            embeddings: false,
            no_op_offload: false,
            no_host: false,
            repack: false,
            fit_target: 0,
            fit_min_ctx: 0,
        }
    }

    /// `cmd_params_instance::to_llama_mparams`.
    pub fn to_llama_mparams(&self) -> ModelParams {
        let mut mp = ModelParams::default();
        mp.n_gpu_layers = self.n_gpu_layers;
        mp.devices = self
            .devices
            .iter()
            .filter(|d| d.as_str() != "none")
            .map(|d| BackendDevice(d.clone()))
            .collect();
        mp.split_mode = self.split_mode;
        mp.load_mode = self.load_mode;
        mp.lazy_mode = self.lazy_mode;
        mp.main_gpu = self.main_gpu;
        mp.tensor_split = self.tensor_split.clone();
        mp.no_host = self.no_host;
        mp.use_extra_bufts = self.repack;

        let mut overrides: Vec<ApiTensorBuftOverride> = self
            .tensor_buft_overrides
            .iter()
            .map(|o| ApiTensorBuftOverride {
                pattern: o.pattern.clone(),
                buft: if o.buffer_type.is_empty() {
                    None
                } else {
                    Some(Buft(o.buffer_type.clone()))
                },
            })
            .collect();

        if self.n_cpu_moe > 0 {
            if overrides.last().is_some_and(|o| o.pattern.is_empty()) {
                overrides.pop();
            }
            for i in 0..self.n_cpu_moe {
                overrides.push(ApiTensorBuftOverride {
                    pattern: llm_ffn_block_regex(i, LLM_FFN_EXPS_REGEX),
                    buft: Some(Buft("CPU".to_owned())),
                });
            }
            overrides.push(ApiTensorBuftOverride { pattern: String::new(), buft: None });
        }

        mp.tensor_buft_overrides = overrides;
        mp
    }

    /// `cmd_params_instance::to_llama_cparams`.
    pub fn to_llama_cparams(&self) -> ContextParams {
        let mut cp = ContextParams::default();
        cp.n_ctx = (self.n_prompt + self.n_gen + self.n_depth) as u32;
        cp.n_batch = self.n_batch as u32;
        cp.n_ubatch = self.n_ubatch as u32;
        cp.type_k = self.type_k;
        cp.type_v = self.type_v;
        cp.offload_kqv = !self.no_kv_offload;
        cp.flash_attn_type = self.flash_attn;
        cp.embeddings = self.embeddings;
        cp.op_offload = !self.no_op_offload;
        cp.swa_full = false;
        cp
    }

    /// `cmd_params_instance::equal_mparams`.
    pub fn equal_mparams(&self, other: &Self) -> bool {
        self.model == other.model
            && self.n_gpu_layers == other.n_gpu_layers
            && self.n_cpu_moe == other.n_cpu_moe
            && self.split_mode == other.split_mode
            && self.main_gpu == other.main_gpu
            && self.tensor_split == other.tensor_split
            && self.load_mode == other.load_mode
            && self.lazy_mode == other.lazy_mode
            && self.devices == other.devices
            && self.no_host == other.no_host
            && self.repack == other.repack
            && self.tensor_buft_overrides == other.tensor_buft_overrides
    }
}

fn expand<T: Clone>(
    instances: &mut Vec<CmdParamsInstance>,
    values: &[T],
    mut set: impl FnMut(&mut CmdParamsInstance, &T),
) {
    let mut out = Vec::with_capacity(instances.len().saturating_mul(values.len()));
    for inst in instances.iter() {
        for value in values {
            let mut next = inst.clone();
            set(&mut next, value);
            out.push(next);
        }
    }
    *instances = out;
}

/// `get_cmd_params_instances`. The dimension order minimizes model reloads.
pub fn get_cmd_params_instances(params: &CmdParams) -> Vec<CmdParamsInstance> {
    let mut instances = vec![CmdParamsInstance::seed()];

    expand(&mut instances, &params.model, |i, v| i.model = v.clone());
    expand(&mut instances, &params.fit_params_target, |i, &v| i.fit_target = v);
    expand(&mut instances, &params.fit_params_min_ctx, |i, &v| i.fit_min_ctx = v);
    expand(&mut instances, &params.n_gpu_layers, |i, &v| i.n_gpu_layers = v);
    expand(&mut instances, &params.n_cpu_moe, |i, &v| i.n_cpu_moe = v);
    expand(&mut instances, &params.split_mode, |i, &v| i.split_mode = v);
    expand(&mut instances, &params.load_mode, |i, &v| i.load_mode = v);
    expand(&mut instances, &params.lazy_mode, |i, &v| i.lazy_mode = v);
    expand(&mut instances, &params.main_gpu, |i, &v| i.main_gpu = v);
    expand(&mut instances, &params.devices, |i, v| i.devices = v.clone());
    expand(&mut instances, &params.tensor_split, |i, v| i.tensor_split = v.clone());
    expand(&mut instances, &params.tensor_buft_overrides, |i, v| i.tensor_buft_overrides = v.clone());
    expand(&mut instances, &params.no_host, |i, &v| i.no_host = v);
    expand(&mut instances, &params.repack, |i, &v| i.repack = v);
    expand(&mut instances, &params.embeddings, |i, &v| i.embeddings = v);
    expand(&mut instances, &params.no_op_offload, |i, &v| i.no_op_offload = v);
    expand(&mut instances, &params.n_batch, |i, &v| i.n_batch = v);
    expand(&mut instances, &params.n_ubatch, |i, &v| i.n_ubatch = v);
    expand(&mut instances, &params.type_k, |i, &v| i.type_k = v);
    expand(&mut instances, &params.type_v, |i, &v| i.type_v = v);
    expand(&mut instances, &params.no_kv_offload, |i, &v| i.no_kv_offload = v);
    expand(&mut instances, &params.flash_attn, |i, &v| i.flash_attn = v);
    expand(&mut instances, &params.n_threads, |i, &v| i.n_threads = v);
    expand(&mut instances, &params.cpu_mask, |i, v| i.cpu_mask = v.clone());
    expand(&mut instances, &params.cpu_strict, |i, &v| i.cpu_strict = v);
    expand(&mut instances, &params.n_depth, |i, &v| i.n_depth = v);
    expand(&mut instances, &params.poll, |i, &v| i.poll = v);

    let mut result = Vec::new();
    for inst in instances {
        for &n_prompt in &params.n_prompt {
            if n_prompt == 0 {
                continue;
            }
            let mut t = inst.clone();
            t.n_prompt = n_prompt;
            t.n_gen = 0;
            result.push(t);
        }
        for &n_gen in &params.n_gen {
            if n_gen == 0 {
                continue;
            }
            let mut t = inst.clone();
            t.n_prompt = 0;
            t.n_gen = n_gen;
            result.push(t);
        }
        for &(pp, tg) in &params.n_pg {
            if pp == 0 && tg == 0 {
                continue;
            }
            let mut t = inst.clone();
            t.n_prompt = pp;
            t.n_gen = tg;
            result.push(t);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// timing maths
// ---------------------------------------------------------------------------

/// `get_time_ns`.
pub fn get_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Time one closure, returning elapsed nanoseconds.
pub fn time_ns<F: FnOnce()>(f: F) -> u64 {
    let start = get_time_ns();
    f();
    get_time_ns().saturating_sub(start)
}

/// `avg<T>` for nanosecond samples (integer division, as in C++).
pub fn avg_u64(v: &[u64]) -> u64 {
    if v.is_empty() {
        return 0;
    }
    let sum = v.iter().fold(0u64, |a, &b| a.wrapping_add(b));
    sum / v.len() as u64
}

/// `stdev<T>` for nanosecond samples.
pub fn stdev_u64(v: &[u64]) -> u64 {
    if v.len() <= 1 {
        return 0;
    }
    let n = v.len() as f64;
    let mean = avg_u64(v) as f64;
    let sq_sum: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (sq_sum / (n - 1.0) - mean * mean * n / (n - 1.0))
        .max(0.0)
        .sqrt() as u64
}

/// `avg<T>` for token-per-second samples.
pub fn avg_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// `stdev<T>` for token-per-second samples.
pub fn stdev_f64(v: &[f64]) -> f64 {
    if v.len() <= 1 {
        return 0.0;
    }
    let n = v.len() as f64;
    let mean = avg_f64(v);
    let sq_sum: f64 = v.iter().map(|x| x * x).sum();
    (sq_sum / (n - 1.0) - mean * mean * n / (n - 1.0)).sqrt()
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

/// `%FT%TZ` UTC timestamp used for `test_time`.
fn utc_now_rfc3339() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let days = secs.div_euclid(86400);
    let sod = secs.rem_euclid(86400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

// ---------------------------------------------------------------------------
// results
// ---------------------------------------------------------------------------

/// Stand-in for the model and context handles used by `test`'s constructor.
#[derive(Clone, Debug, Default)]
pub struct TestMeta {
    pub model_type: String,
    pub model_size: u64,
    pub model_n_params: u64,
    pub backends: String,
    pub cpu_info: String,
    pub gpu_info: String,
}

/// `test`.
#[derive(Clone, Debug)]
pub struct Test {
    pub build_commit: String,
    pub build_number: i32,
    pub cpu_info: String,
    pub gpu_info: String,
    pub backends: String,
    pub model_filename: String,
    pub model_type: String,
    pub model_size: u64,
    pub model_n_params: u64,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_threads: i32,
    pub cpu_mask: String,
    pub cpu_strict: bool,
    pub poll: i32,
    pub type_k: GgmlType,
    pub type_v: GgmlType,
    pub n_gpu_layers: i32,
    pub n_cpu_moe: i32,
    pub split_mode: SplitMode,
    pub load_mode: LoadMode,
    pub lazy_mode: LazyMode,
    pub main_gpu: i32,
    pub no_kv_offload: bool,
    pub flash_attn: FlashAttnType,
    pub devices: Vec<String>,
    pub tensor_split: Vec<f32>,
    pub tensor_buft_overrides: Vec<TensorBuftOverride>,
    pub embeddings: bool,
    pub no_op_offload: bool,
    pub no_host: bool,
    pub repack: bool,
    pub fit_target: usize,
    pub fit_min_ctx: u32,
    pub n_prompt: i32,
    pub n_gen: i32,
    pub n_depth: i32,
    pub test_time: String,
    pub samples_ns: Vec<u64>,
}

impl Test {
    pub fn new(inst: &CmdParamsInstance, meta: &TestMeta) -> Self {
        Self {
            build_commit: BUILD_COMMIT.to_owned(),
            build_number: BUILD_NUMBER,
            cpu_info: meta.cpu_info.clone(),
            gpu_info: meta.gpu_info.clone(),
            backends: meta.backends.clone(),
            model_filename: inst.model.clone(),
            model_type: meta.model_type.clone(),
            model_size: meta.model_size,
            model_n_params: meta.model_n_params,
            n_batch: inst.n_batch,
            n_ubatch: inst.n_ubatch,
            n_threads: inst.n_threads,
            cpu_mask: inst.cpu_mask.clone(),
            cpu_strict: inst.cpu_strict,
            poll: inst.poll,
            type_k: inst.type_k,
            type_v: inst.type_v,
            n_gpu_layers: inst.n_gpu_layers,
            n_cpu_moe: inst.n_cpu_moe,
            split_mode: inst.split_mode,
            load_mode: inst.load_mode,
            lazy_mode: inst.lazy_mode,
            main_gpu: inst.main_gpu,
            no_kv_offload: inst.no_kv_offload,
            flash_attn: inst.flash_attn,
            devices: inst.devices.clone(),
            tensor_split: inst.tensor_split.clone(),
            tensor_buft_overrides: inst.tensor_buft_overrides.clone(),
            embeddings: inst.embeddings,
            no_op_offload: inst.no_op_offload,
            no_host: inst.no_host,
            repack: inst.repack,
            fit_target: inst.fit_target,
            fit_min_ctx: inst.fit_min_ctx,
            n_prompt: inst.n_prompt,
            n_gen: inst.n_gen,
            n_depth: inst.n_depth,
            test_time: utc_now_rfc3339(),
            samples_ns: Vec::new(),
        }
    }

    pub fn avg_ns(&self) -> u64 {
        avg_u64(&self.samples_ns)
    }

    pub fn stdev_ns(&self) -> u64 {
        stdev_u64(&self.samples_ns)
    }

    pub fn get_ts(&self) -> Vec<f64> {
        let n_tokens = self.n_prompt + self.n_gen;
        self.samples_ns
            .iter()
            .map(|&t| 1e9 * n_tokens as f64 / t as f64)
            .collect()
    }

    pub fn avg_ts(&self) -> f64 {
        avg_f64(&self.get_ts())
    }

    pub fn stdev_ts(&self) -> f64 {
        stdev_f64(&self.get_ts())
    }

    pub fn get_backend(&self) -> &str {
        if self.backends.is_empty() { "CPU" } else { &self.backends }
    }

    pub fn get_fields() -> &'static [&'static str] {
        &[
            "build_commit", "build_number", "cpu_info", "gpu_info", "backends",
            "model_filename", "model_type", "model_size", "model_n_params", "n_batch",
            "n_ubatch", "n_threads", "cpu_mask", "cpu_strict", "poll",
            "type_k", "type_v", "n_gpu_layers", "n_cpu_moe", "split_mode",
            "main_gpu", "no_kv_offload", "flash_attn", "devices", "tensor_split",
            "tensor_buft_overrides", "load_mode", "lazy_mode", "embeddings",
            "no_op_offload", "no_host", "repack", "fit_target", "fit_min_ctx",
            "n_prompt", "n_gen", "n_depth", "test_time", "avg_ns", "stddev_ns",
            "avg_ts", "stddev_ts",
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldType {
    String,
    Bool,
    Int,
    Float,
}

/// `test::get_field_type`.
pub fn get_field_type(field: &str) -> FieldType {
    if matches!(
        field,
        "build_number" | "n_batch" | "n_ubatch" | "n_threads" | "poll" | "model_size"
            | "model_n_params" | "n_gpu_layers" | "main_gpu" | "n_prompt" | "n_gen" | "n_depth"
            | "avg_ns" | "stddev_ns" | "no_op_offload" | "n_cpu_moe" | "fit_target"
            | "fit_min_ctx" | "flash_attn"
    ) {
        return FieldType::Int;
    }
    if matches!(
        field,
        "f16_kv" | "no_kv_offload" | "cpu_strict" | "embeddings" | "no_host" | "repack"
    ) {
        return FieldType::Bool;
    }
    if matches!(field, "avg_ts" | "stddev_ts") {
        return FieldType::Float;
    }
    FieldType::String
}

fn tensor_split_string(split: &[f32]) -> String {
    let mut max_nonzero = 0usize;
    for (i, &v) in split.iter().enumerate().take(LLAMA_MAX_DEVICES) {
        if v > 0.0 {
            max_nonzero = i;
        }
    }
    (0..=max_nonzero)
        .map(|i| format!("{:.2}", split.get(i).copied().unwrap_or(0.0)))
        .collect::<Vec<_>>()
        .join("/")
}

fn overrides_string(overrides: &[TensorBuftOverride]) -> String {
    if overrides.len() <= 1 {
        return "none".to_owned();
    }
    let last = overrides.len() - 1;
    overrides[..last]
        .iter()
        .map(|o| {
            if o.pattern.is_empty() {
                "none".to_owned()
            } else {
                format!("{}={}", o.pattern, o.buffer_type)
            }
        })
        .collect::<Vec<_>>()
        .join(";")
}

impl Test {
    /// `test::get_values`.
    pub fn get_values(&self) -> Vec<String> {
        vec![
            self.build_commit.clone(),
            self.build_number.to_string(),
            self.cpu_info.clone(),
            self.gpu_info.clone(),
            self.get_backend().to_owned(),
            self.model_filename.clone(),
            self.model_type.clone(),
            self.model_size.to_string(),
            self.model_n_params.to_string(),
            self.n_batch.to_string(),
            self.n_ubatch.to_string(),
            self.n_threads.to_string(),
            self.cpu_mask.clone(),
            (self.cpu_strict as i32).to_string(),
            self.poll.to_string(),
            ggml_type_name(self.type_k).to_owned(),
            ggml_type_name(self.type_v).to_owned(),
            self.n_gpu_layers.to_string(),
            self.n_cpu_moe.to_string(),
            split_mode_str(self.split_mode).to_owned(),
            self.main_gpu.to_string(),
            (self.no_kv_offload as i32).to_string(),
            (self.flash_attn as i32).to_string(),
            devices_to_string(&self.devices),
            tensor_split_string(&self.tensor_split),
            overrides_string(&self.tensor_buft_overrides),
            load_mode_name(self.load_mode).to_owned(),
            lazy_mode_str(self.lazy_mode).to_owned(),
            (self.embeddings as i32).to_string(),
            (self.no_op_offload as i32).to_string(),
            (self.no_host as i32).to_string(),
            (self.repack as i32).to_string(),
            self.fit_target.to_string(),
            self.fit_min_ctx.to_string(),
            self.n_prompt.to_string(),
            self.n_gen.to_string(),
            self.n_depth.to_string(),
            self.test_time.clone(),
            self.avg_ns().to_string(),
            self.stdev_ns().to_string(),
            self.avg_ts().to_string(),
            self.stdev_ts().to_string(),
        ]
    }

    /// `test::get_map`.
    pub fn get_map(&self) -> BTreeMap<String, String> {
        Self::get_fields()
            .iter()
            .zip(self.get_values())
            .map(|(f, v)| ((*f).to_owned(), v))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// printers
// ---------------------------------------------------------------------------

pub trait Printer {
    fn print_header(&mut self, out: &mut dyn Write, params: &CmdParams, backends: &str) -> Result<()>;
    fn print_test(&mut self, out: &mut dyn Write, t: &Test) -> Result<()>;
    fn print_footer(&mut self, out: &mut dyn Write) -> Result<()>;
}

fn escape_csv(field: &str) -> String {
    let mut out = String::from("\"");
    for c in field.chars() {
        if c == '"' {
            out.push('"');
        }
        out.push(c);
    }
    out.push('"');
    out
}

fn escape_json(value: &str) -> String {
    let mut out = String::new();
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) <= 0x1f => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn format_json_value(field: &str, value: &str) -> String {
    match get_field_type(field) {
        FieldType::String => format!("\"{}\"", escape_json(value)),
        FieldType::Bool => if value == "0" { "false".to_owned() } else { "true".to_owned() },
        _ => value.to_owned(),
    }
}

fn join_display<T: std::fmt::Display>(values: &[T]) -> String {
    values.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
}

#[derive(Default)]
pub struct CsvPrinter;

impl Printer for CsvPrinter {
    fn print_header(&mut self, out: &mut dyn Write, _params: &CmdParams, _backends: &str) -> Result<()> {
        writeln!(out, "{}", Test::get_fields().join(",")).map_err(io_error)
    }

    fn print_test(&mut self, out: &mut dyn Write, t: &Test) -> Result<()> {
        let values: Vec<String> = t.get_values().iter().map(|v| escape_csv(v)).collect();
        writeln!(out, "{}", values.join(",")).map_err(io_error)
    }

    fn print_footer(&mut self, _out: &mut dyn Write) -> Result<()> {
        Ok(())
    }
}

#[derive(Default)]
pub struct JsonPrinter {
    first: bool,
}

impl JsonPrinter {
    pub fn new() -> Self {
        Self { first: true }
    }

    fn print_fields(out: &mut dyn Write, fields: &[&str], values: &[String]) -> Result<()> {
        for (field, value) in fields.iter().zip(values) {
            writeln!(out, "    \"{field}\": {},", format_json_value(field, value)).map_err(io_error)?;
        }
        Ok(())
    }
}

impl Printer for JsonPrinter {
    fn print_header(&mut self, out: &mut dyn Write, _params: &CmdParams, _backends: &str) -> Result<()> {
        writeln!(out, "[").map_err(io_error)
    }

    fn print_test(&mut self, out: &mut dyn Write, t: &Test) -> Result<()> {
        if self.first {
            self.first = false;
        } else {
            write!(out, ",\n").map_err(io_error)?;
        }
        write!(out, "  {{\n").map_err(io_error)?;
        Self::print_fields(out, Test::get_fields(), &t.get_values())?;
        writeln!(out, "    \"samples_ns\": [ {} ],", join_display(&t.samples_ns)).map_err(io_error)?;
        writeln!(out, "    \"samples_ts\": [ {} ]", join_display(&t.get_ts())).map_err(io_error)?;
        write!(out, "  }}").map_err(io_error)
    }

    fn print_footer(&mut self, out: &mut dyn Write) -> Result<()> {
        write!(out, "\n]\n").map_err(io_error)
    }
}

#[derive(Default)]
pub struct JsonlPrinter;

impl JsonlPrinter {
    fn print_fields(out: &mut dyn Write, fields: &[&str], values: &[String]) -> Result<()> {
        for (field, value) in fields.iter().zip(values) {
            write!(out, "\"{field}\": {}, ", format_json_value(field, value)).map_err(io_error)?;
        }
        Ok(())
    }
}

impl Printer for JsonlPrinter {
    fn print_header(&mut self, _out: &mut dyn Write, _params: &CmdParams, _backends: &str) -> Result<()> {
        Ok(())
    }

    fn print_test(&mut self, out: &mut dyn Write, t: &Test) -> Result<()> {
        write!(out, "{{").map_err(io_error)?;
        Self::print_fields(out, Test::get_fields(), &t.get_values())?;
        write!(out, "\"samples_ns\": [ {} ],", join_display(&t.samples_ns)).map_err(io_error)?;
        writeln!(out, "\"samples_ts\": [ {} ]}}", join_display(&t.get_ts())).map_err(io_error)
    }

    fn print_footer(&mut self, _out: &mut dyn Write) -> Result<()> {
        Ok(())
    }
}

fn field_width(field: &str) -> i32 {
    match field {
        "model" => -30,
        "t/s" => 20,
        "size" | "params" => 10,
        "n_gpu_layers" => 3,
        "n_threads" => 7,
        "n_batch" => 7,
        "n_ubatch" => 8,
        "type_k" | "type_v" => 6,
        "split_mode" => 6,
        "load_mode" => 10,
        "flash_attn" => 3,
        "devices" => -12,
        "test" => 15,
        "no_op_offload" => 4,
        "no_host" => 4,
        "repack" => 3,
        _ => {
            let width = field.len().max(10) as i32;
            if get_field_type(field) == FieldType::String { -width } else { width }
        }
    }
}

fn field_display_name(field: &str) -> &str {
    match field {
        "n_gpu_layers" => "ngl",
        "split_mode" => "sm",
        "n_threads" => "threads",
        "no_kv_offload" => "nkvo",
        "flash_attn" => "fa",
        "load_mode" => "lm",
        "embeddings" => "embd",
        "no_op_offload" => "nopo",
        "no_host" => "noh",
        "repack" => "rpk",
        "devices" => "dev",
        "tensor_split" => "ts",
        "tensor_buft_overrides" => "ot",
        "fit_target" => "fitt",
        "fit_min_ctx" => "fitc",
        _ => field,
    }
}

/// `printf("%*s")` semantics, counting bytes like C++ does.
fn pad(value: &str, width: i32) -> String {
    let target = width.unsigned_abs() as usize;
    let len = value.len();
    if len >= target {
        return value.to_owned();
    }
    let spaces = " ".repeat(target - len);
    if width > 0 { format!("{spaces}{value}") } else { format!("{value}{spaces}") }
}

#[derive(Default)]
pub struct MarkdownPrinter {
    fields: Vec<String>,
}

impl Printer for MarkdownPrinter {
    fn print_header(&mut self, out: &mut dyn Write, params: &CmdParams, backends: &str) -> Result<()> {
        self.fields.clear();
        self.fields.push("model".to_owned());
        self.fields.push("size".to_owned());
        self.fields.push("params".to_owned());
        self.fields.push("backend".to_owned());

        let is_cpu_backend = backends.contains("CPU") || backends.contains("BLAS") || backends.contains("ZenDNN");
        if !is_cpu_backend {
            self.fields.push("n_gpu_layers".to_owned());
        }
        let defaults = CmdParams::default();
        if params.n_cpu_moe.len() > 1 || params.n_cpu_moe != defaults.n_cpu_moe {
            self.fields.push("n_cpu_moe".to_owned());
        }
        if params.n_threads.len() > 1 || params.n_threads != defaults.n_threads || is_cpu_backend {
            self.fields.push("n_threads".to_owned());
        }
        if params.cpu_mask.len() > 1 || params.cpu_mask != defaults.cpu_mask {
            self.fields.push("cpu_mask".to_owned());
        }
        if params.cpu_strict.len() > 1 || params.cpu_strict != defaults.cpu_strict {
            self.fields.push("cpu_strict".to_owned());
        }
        if params.poll.len() > 1 || params.poll != defaults.poll {
            self.fields.push("poll".to_owned());
        }
        if params.n_batch.len() > 1 || params.n_batch != defaults.n_batch {
            self.fields.push("n_batch".to_owned());
        }
        if params.n_ubatch.len() > 1 || params.n_ubatch != defaults.n_ubatch {
            self.fields.push("n_ubatch".to_owned());
        }
        if params.type_k.len() > 1 || params.type_k != defaults.type_k {
            self.fields.push("type_k".to_owned());
        }
        if params.type_v.len() > 1 || params.type_v != defaults.type_v {
            self.fields.push("type_v".to_owned());
        }
        if params.main_gpu.len() > 1 || params.main_gpu != defaults.main_gpu {
            self.fields.push("main_gpu".to_owned());
        }
        if params.split_mode.len() > 1 || params.split_mode != defaults.split_mode {
            self.fields.push("split_mode".to_owned());
        }
        if params.no_kv_offload.len() > 1 || params.no_kv_offload != defaults.no_kv_offload {
            self.fields.push("no_kv_offload".to_owned());
        }
        if params.flash_attn.len() > 1 || params.flash_attn != defaults.flash_attn {
            self.fields.push("flash_attn".to_owned());
        }
        if params.devices.len() > 1 || params.devices != defaults.devices {
            self.fields.push("devices".to_owned());
        }
        if params.tensor_split.len() > 1 || params.tensor_split != defaults.tensor_split {
            self.fields.push("tensor_split".to_owned());
        }
        if params.tensor_buft_overrides.len() > 1
            || params.tensor_buft_overrides != defaults.tensor_buft_overrides
        {
            self.fields.push("tensor_buft_overrides".to_owned());
        }
        if params.load_mode.len() > 1 || params.load_mode != defaults.load_mode {
            self.fields.push("load_mode".to_owned());
        }
        if params.lazy_mode.len() > 1 || params.lazy_mode != defaults.lazy_mode {
            self.fields.push("lazy_mode".to_owned());
        }
        if params.embeddings.len() > 1 || params.embeddings != defaults.embeddings {
            self.fields.push("embeddings".to_owned());
        }
        if params.no_op_offload.len() > 1 || params.no_op_offload != defaults.no_op_offload {
            self.fields.push("no_op_offload".to_owned());
        }
        if params.no_host.len() > 1 || params.no_host != defaults.no_host {
            self.fields.push("no_host".to_owned());
        }
        if params.repack.len() > 1 || params.repack != defaults.repack {
            self.fields.push("repack".to_owned());
        }
        if params.fit_params_target.len() > 1 || params.fit_params_target != defaults.fit_params_target {
            self.fields.push("fit_target".to_owned());
        }
        if params.fit_params_min_ctx.len() > 1 || params.fit_params_min_ctx != defaults.fit_params_min_ctx {
            self.fields.push("fit_min_ctx".to_owned());
        }
        self.fields.push("test".to_owned());
        self.fields.push("t/s".to_owned());

        write!(out, "|").map_err(io_error)?;
        for field in &self.fields {
            write!(out, " {} |", pad(field_display_name(field), field_width(field))).map_err(io_error)?;
        }
        writeln!(out).map_err(io_error)?;
        write!(out, "|").map_err(io_error)?;
        for field in &self.fields {
            let width = field_width(field).unsigned_abs() as usize;
            let tail = if field_width(field) > 0 { ":" } else { "-" };
            write!(out, " {}{} |", "-".repeat(width - 1), tail).map_err(io_error)?;
        }
        writeln!(out).map_err(io_error)
    }

    fn print_test(&mut self, out: &mut dyn Write, t: &Test) -> Result<()> {
        let vmap = t.get_map();
        write!(out, "|").map_err(io_error)?;
        for field in &self.fields {
            let value = match field.as_str() {
                "model" => t.model_type.clone(),
                "size" => {
                    if t.model_size < 1024 * 1024 * 1024 {
                        format!("{:.2} MiB", t.model_size as f64 / 1024.0 / 1024.0)
                    } else {
                        format!("{:.2} GiB", t.model_size as f64 / 1024.0 / 1024.0 / 1024.0)
                    }
                }
                "params" => {
                    if t.model_n_params < 1_000_000_000 {
                        format!("{:.2} M", t.model_n_params as f64 / 1e6)
                    } else {
                        format!("{:.2} B", t.model_n_params as f64 / 1e9)
                    }
                }
                "backend" => t.get_backend().to_owned(),
                "test" => {
                    let mut label = if t.n_prompt > 0 && t.n_gen == 0 {
                        format!("pp{}", t.n_prompt)
                    } else if t.n_gen > 0 && t.n_prompt == 0 {
                        format!("tg{}", t.n_gen)
                    } else {
                        format!("pp{}+tg{}", t.n_prompt, t.n_gen)
                    };
                    if t.n_depth > 0 {
                        label.push_str(&format!(" @ d{}", t.n_depth));
                    }
                    label
                }
                "t/s" => format!("{:.2} ± {:.2}", t.avg_ts(), t.stdev_ts()),
                other => vmap.get(other).cloned().unwrap_or_default(),
            };
            let width = field_width(field) + i32::from(field == "t/s");
            write!(out, " {} |", pad(&value, width)).map_err(io_error)?;
        }
        writeln!(out).map_err(io_error)
    }

    fn print_footer(&mut self, out: &mut dyn Write) -> Result<()> {
        writeln!(out, "\nbuild: {} ({})", BUILD_COMMIT, BUILD_NUMBER).map_err(io_error)
    }
}

fn sql_field_type(field: &str) -> &'static str {
    match get_field_type(field) {
        FieldType::String => "TEXT",
        FieldType::Bool | FieldType::Int => "INTEGER",
        FieldType::Float => "REAL",
    }
}

#[derive(Default)]
pub struct SqlPrinter;

impl Printer for SqlPrinter {
    fn print_header(&mut self, out: &mut dyn Write, _params: &CmdParams, _backends: &str) -> Result<()> {
        writeln!(out, "CREATE TABLE IF NOT EXISTS llama_bench (").map_err(io_error)?;
        let fields = Test::get_fields();
        for (i, field) in fields.iter().enumerate() {
            let tail = if i + 1 < fields.len() { "," } else { "" };
            writeln!(out, "  {field} {}{tail}", sql_field_type(field)).map_err(io_error)?;
        }
        writeln!(out, ");").map_err(io_error)?;
        writeln!(out).map_err(io_error)
    }

    fn print_test(&mut self, out: &mut dyn Write, t: &Test) -> Result<()> {
        write!(out, "INSERT INTO llama_bench ({}) ", Test::get_fields().join(", ")).map_err(io_error)?;
        write!(out, "VALUES (").map_err(io_error)?;
        let values = t.get_values();
        for (i, value) in values.iter().enumerate() {
            let tail = if i + 1 < values.len() { ", " } else { "" };
            write!(out, "'{value}'{tail}").map_err(io_error)?;
        }
        writeln!(out, ");").map_err(io_error)
    }

    fn print_footer(&mut self, _out: &mut dyn Write) -> Result<()> {
        Ok(())
    }
}

/// `create_printer`.
pub fn create_printer(format: OutputFormat) -> Option<Box<dyn Printer>> {
    match format {
        OutputFormat::None => None,
        OutputFormat::Csv => Some(Box::new(CsvPrinter::default())),
        OutputFormat::Json => Some(Box::new(JsonPrinter::new())),
        OutputFormat::Jsonl => Some(Box::new(JsonlPrinter::default())),
        OutputFormat::Markdown => Some(Box::new(MarkdownPrinter::default())),
        OutputFormat::Sql => Some(Box::new(SqlPrinter::default())),
    }
}

/// Print header, all tests and footer through one printer.
pub fn print_all(
    printer: &mut dyn Printer,
    out: &mut dyn Write,
    params: &CmdParams,
    tests: &[Test],
    backends: &str,
) -> Result<()> {
    printer.print_header(out, params, backends)?;
    for t in tests {
        printer.print_test(out, t)?;
    }
    printer.print_footer(out)
}

// ---------------------------------------------------------------------------
// benchmark loop
// ---------------------------------------------------------------------------

/// Mirror of the timing and repetition loop in `llama_bench`.
///
/// The caller supplies prompt, generation and depth work. `depth_cached`
/// reports whether the context state at the requested depth was restored;
/// when false, `depth` is run to rebuild it. No context state is stored here.
pub fn run_benchmark(
    test: &mut Test,
    params: &CmdParams,
    index: usize,
    count: usize,
    prompt: &mut dyn FnMut() -> Result<()>,
    gen: &mut dyn FnMut() -> Result<()>,
    depth: &mut dyn FnMut() -> Result<()>,
    depth_cached: &mut dyn FnMut(i32) -> bool,
) -> Result<()> {
    if params.delay > 0 {
        std::thread::sleep(Duration::from_secs(params.delay as u64));
    }

    if !params.no_warmup {
        if test.n_prompt > 0 {
            if params.progress {
                eprintln!("llama-bench: benchmark {index}/{count}: warmup prompt run");
            }
            prompt()?;
        }
        if test.n_gen > 0 {
            if params.progress {
                eprintln!("llama-bench: benchmark {index}/{count}: warmup generation run");
            }
            gen()?;
        }
    }

    for i in 0..params.reps.max(0) {
        if test.n_depth > 0 && !depth_cached(test.n_depth) {
            if params.progress {
                eprintln!("llama-bench: benchmark {index}/{count}: depth run {}/{}", i + 1, params.reps);
            }
            depth()?;
        }

        let start = get_time_ns();

        if test.n_prompt > 0 {
            if params.progress {
                eprintln!("llama-bench: benchmark {index}/{count}: prompt run {}/{}", i + 1, params.reps);
            }
            prompt()?;
        }
        if test.n_gen > 0 {
            if params.progress {
                eprintln!("llama-bench: benchmark {index}/{count}: generation run {}/{}", i + 1, params.reps);
            }
            gen()?;
        }

        test.samples_ns.push(get_time_ns().saturating_sub(start));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn int_ranges() {
        assert_eq!(parse_int_range("512", false).unwrap(), vec![512]);
        assert_eq!(parse_int_range("1-5+2,10", false).unwrap(), vec![1, 3, 5, 10]);
        assert_eq!(parse_int_range("1-10*2", false).unwrap(), vec![1, 2, 4, 8]);
        assert_eq!(parse_int_range("-2,4", true).unwrap(), vec![-2, 4]);
        assert_eq!(parse_int_range("-2", false).unwrap(), vec![2]);
        assert!(parse_int_range("5-1", false).unwrap().is_empty());
        assert!(parse_int_range("3-9+0", false).is_err());
    }

    #[test]
    fn defaults_match_upstream() {
        let p = CmdParams::default();
        assert_eq!(p.n_prompt, vec![512]);
        assert_eq!(p.n_gen, vec![128]);
        assert!(p.n_pg.is_empty());
        assert_eq!(p.type_k, vec![GgmlType::F16]);
        assert_eq!(p.n_gpu_layers, vec![-1]);
        assert_eq!(p.poll, vec![50]);
        assert_eq!(p.reps, 5);
        assert_eq!(p.output_format, OutputFormat::Markdown);
        assert_eq!(p.output_format_stderr, OutputFormat::None);
        assert_eq!(p.tensor_split[0].len(), LLAMA_MAX_DEVICES);
    }

    #[test]
    fn parse_all_common_options() {
        let a = args(&[
            "-m", "a.gguf,b.gguf", "-p", "1-3+1", "-n", "8", "-pg", "16,32",
            "-b", "1024", "-ub", "256", "-ctk", "q8_0", "-ctv", "bf16",
            "-t", "4", "--cpu-strict", "1", "-ngl", "-2", "-ncmoe", "2",
            "-sm", "row", "-lm", "mmap+mlock", "-lzm", "off", "-mg", "1",
            "-nkvo", "1", "-fa", "on", "-embd", "0", "--no-host", "1",
            "--repack", "0", "-r", "3", "--prio", "2", "--delay", "1",
            "-o", "jsonl", "-oe", "csv", "--no-warmup", "--progress", "-v",
            "-fitt", "512", "-fitc", "1024", "--numa", "isolate",
        ]);
        let ParseOutcome::Params(p) = parse_cmd_params("llama-bench", &a).unwrap() else {
            panic!("unexpected exit");
        };
        assert_eq!(p.model, vec!["a.gguf", "b.gguf"]);
        assert_eq!(p.n_prompt, vec![1, 2, 3]);
        assert_eq!(p.n_gen, vec![8]);
        assert_eq!(p.n_pg, vec![(16, 32)]);
        assert_eq!(p.type_k, vec![GgmlType::Q8_0]);
        assert_eq!(p.type_v, vec![GgmlType::Bf16]);
        assert_eq!(p.split_mode, vec![SplitMode::Row]);
        assert_eq!(p.load_mode, vec![LoadMode::MmapMlock]);
        assert_eq!(p.lazy_mode, vec![LazyMode::Off]);
        assert_eq!(p.flash_attn, vec![FlashAttnType::Enabled]);
        assert_eq!(p.n_gpu_layers, vec![-2]);
        assert_eq!(p.prio, SchedPriority::High);
        assert_eq!(p.numa, NumaStrategy::Isolate);
        assert_eq!(p.output_format, OutputFormat::Jsonl);
        assert_eq!(p.output_format_stderr, OutputFormat::Csv);
        assert_eq!(p.fit_params_target, vec![512]);
        assert_eq!(p.fit_params_min_ctx, vec![1024]);
        assert!(p.progress && p.verbose && p.no_warmup);
        assert_eq!(p.reps, 3);
        assert_eq!(p.delay, 1);
    }

    #[test]
    fn invalid_options_fail() {
        assert!(parse_cmd_params("bench", &args(&["-o", "xml"])).is_err());
        assert!(parse_cmd_params("bench", &args(&["-ctk", "q2_K"])).is_err());
        assert!(parse_cmd_params("bench", &args(&["--cpu-strict", "yes"])).is_err());
        assert!(parse_cmd_params("bench", &args(&["-pg", "1"])).is_err());
        assert!(parse_cmd_params("bench", &args(&["-m"])).is_err());
    }

    #[test]
    fn instance_order_and_counts() {
        let ParseOutcome::Params(p) = parse_cmd_params("bench", &args(&["-p", "128,256", "-n", "32", "-pg", "16,8"])).unwrap() else {
            panic!("unexpected exit");
        };
        let inst = get_cmd_params_instances(&p);
        assert_eq!(inst.len(), 4);
        assert_eq!((inst[0].n_prompt, inst[0].n_gen), (128, 0));
        assert_eq!((inst[1].n_prompt, inst[1].n_gen), (256, 0));
        assert_eq!((inst[2].n_prompt, inst[2].n_gen), (0, 32));
        assert_eq!((inst[3].n_prompt, inst[3].n_gen), (16, 8));
    }

    #[test]
    fn parameter_mapping() {
        let ParseOutcome::Params(p) = parse_cmd_params("bench", &args(&["-p", "128", "-n", "32", "-d", "5", "-nkvo", "1", "-fa", "on"])).unwrap() else {
            panic!("unexpected exit");
        };
        let instances = get_cmd_params_instances(&p);
        let cp = instances[0].to_llama_cparams();
        assert_eq!(cp.n_ctx, 133);
        assert_eq!(instances[1].to_llama_cparams().n_ctx, 37);
        let inst = &instances[0];
        assert_eq!(cp.n_batch, 2048);
        assert_eq!(cp.n_ubatch, 512);
        assert!(!cp.offload_kqv);
        assert_eq!(cp.flash_attn_type, FlashAttnType::Enabled);
        assert!(!cp.swa_full);
        assert!(cp.op_offload);

        let mp = inst.to_llama_mparams();
        assert_eq!(mp.n_gpu_layers, -1);
        assert_eq!(mp.split_mode, SplitMode::Layer);
        assert!(mp.use_extra_bufts);
    }

    #[test]
    fn n_cpu_moe_merges_overrides() {
        let ParseOutcome::Params(p) = parse_cmd_params("bench", &args(&["-ncmoe", "2", "-ot", "blk.0=CPU"])).unwrap() else {
            panic!("unexpected exit");
        };
        let inst = &get_cmd_params_instances(&p)[0];
        let mp = inst.to_llama_mparams();
        assert_eq!(mp.tensor_buft_overrides.len(), 4);
        assert_eq!(mp.tensor_buft_overrides[0].pattern, "blk.0");
        assert_eq!(mp.tensor_buft_overrides[1].pattern, "blk\\.0\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        assert_eq!(mp.tensor_buft_overrides[2].pattern, "blk\\.1\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        assert!(mp.tensor_buft_overrides[3].pattern.is_empty());
    }

    #[test]
    fn stats_and_tokens_per_second() {
        let inst = CmdParamsInstance::seed();
        let mut t = Test::new(&inst, &TestMeta::default());
        assert_eq!(t.avg_ns(), 0);
        assert_eq!(t.stdev_ns(), 0);
        t.n_prompt = 100;
        t.samples_ns = vec![1_000_000_000, 2_000_000_000];
        assert_eq!(t.avg_ns(), 1_500_000_000);
        let ts = t.get_ts();
        assert_eq!(ts.len(), 2);
        assert_eq!(ts[0], 100.0);
        assert_eq!(ts[1], 50.0);
        assert_eq!(t.avg_ts(), 75.0);
        assert!(t.stdev_ts() > 0.0);
    }

    #[test]
    fn printer_outputs() {
        let inst = CmdParamsInstance {
            model: "model.gguf".to_owned(),
            n_prompt: 16,
            n_gen: 0,
            n_batch: 512,
            n_ubatch: 512,
            n_threads: 4,
            ..CmdParamsInstance::seed()
        };
        let meta = TestMeta {
            model_type: "q4_0".to_owned(),
            model_size: 2 * 1024 * 1024,
            model_n_params: 7_000_000,
            ..TestMeta::default()
        };
        let mut t = Test::new(&inst, &meta);
        t.samples_ns = vec![100, 200];

        let mut out = Vec::new();
        let mut p = CsvPrinter::default();
        print_all(&mut p, &mut out, &CmdParams::default(), &[t.clone()], "CPU").unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("build_commit,build_number,"));
        assert!(text.lines().nth(1).unwrap().contains("\"model.gguf\""));

        let mut out = Vec::new();
        let mut p = JsonPrinter::new();
        print_all(&mut p, &mut out, &CmdParams::default(), &[t.clone()], "CPU").unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("\"samples_ns\": [ 100, 200 ]"));
        assert!(text.contains("\"no_kv_offload\": false"));
        assert!(text.trim_end().ends_with("]"));

        let mut out = Vec::new();
        let mut p = MarkdownPrinter::default();
        print_all(&mut p, &mut out, &CmdParams::default(), &[t.clone()], "CPU").unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("pp16"));
        assert!(text.contains("2.00 MiB"));
        assert!(text.contains("7.00 M"));
        assert!(text.contains("±"));

        let mut out = Vec::new();
        let mut p = SqlPrinter::default();
        print_all(&mut p, &mut out, &CmdParams::default(), &[t.clone()], "CPU").unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("CREATE TABLE IF NOT EXISTS llama_bench"));
        assert!(text.contains("INSERT INTO llama_bench"));

        let mut out = Vec::new();
        let mut p = JsonlPrinter::default();
        print_all(&mut p, &mut out, &CmdParams::default(), &[t], "CPU").unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("{"));
        assert!(text.contains("\"samples_ts\": ["));
    }

    #[test]
    fn benchmark_loop_records_samples() {
        let inst = CmdParamsInstance {
            n_prompt: 4,
            n_threads: 2,
            ..CmdParamsInstance::seed()
        };
        let mut t = Test::new(&inst, &TestMeta::default());
        let params = CmdParams { reps: 3, no_warmup: true, ..CmdParams::default() };
        let mut calls = 0;
        let mut prompt = || {
            calls += 1;
            Ok(())
        };
        let mut gen = || Ok(());
        let mut depth = || Ok(());
        let mut cached = |_d: i32| true;
        run_benchmark(&mut t, &params, 1, 1, &mut prompt, &mut gen, &mut depth, &mut cached).unwrap();
        assert_eq!(t.samples_ns.len(), 3);
        assert_eq!(calls, 3);
    }
}

//! Rust port of `tools/completion/completion.cpp`.
//!
//! The C++ driver talks to live model, context, sampler and console handles.
//! Those handle calls are mirrored through `crate::api` and `crate::chat`;
//! `crate::common::Params` is the port of `common_params`. Output goes to the
//! same streams as the C++ macros: `LOG` to stdout, `LOG_INF/WRN/ERR` to stderr.
//!
//! Omissions are listed at the end of the file.

use crate::api::{self, Batch, Context, Model, Sampler, TokenData, TokenDataArray, Vocab};
use crate::chat::{ChatMessage, ChatMsg, ChatTemplatesInputs};
use crate::common::{self, ConversationMode, Params, SamplerType};
use crate::core::{Error, Result, Token};
use crate::cparams;
use std::collections::VecDeque;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::rc::Rc;

const FN_NAME: &str = "llama_completion";

/// `common_chat_msg` plus the template source selected for this run.
pub struct ChatTemplates {
    source: Option<String>,
    explicit: bool,
}

impl ChatTemplates {
    /// `common_chat_templates_init`.
    pub fn init(model: &Model, override_src: &str) -> Self {
        let explicit_override = !override_src.is_empty();
        let source = if explicit_override {
            Some(override_src.to_owned())
        } else {
            model
                .chat_templates
                .get("default")
                .or_else(|| model.chat_templates.values().next())
                .cloned()
        };
        let explicit = explicit_override || source.is_some();
        Self { source, explicit }
    }

    /// `common_chat_templates_was_explicit`.
    pub fn was_explicit(&self) -> bool {
        self.explicit
    }

    /// `common_chat_format_example`.
    pub fn example(&self) -> Result<String> {
        match &self.source {
            Some(src) => crate::chat::chat_format_example(src),
            None => Err(Error::new("no chat template is available")),
        }
    }

    /// `common_chat_format_single`.
    pub fn format_single(
        &self,
        past: &[ChatMsg],
        new_msg: &ChatMsg,
        add_ass: bool,
    ) -> Result<String> {
        match &self.source {
            Some(src) => crate::chat::chat_format_single(src, past, new_msg, add_ass),
            None => Err(Error::new("no chat template is available")),
        }
    }

    /// `common_chat_templates_apply(...).prompt.str()`.
    pub fn apply(&self, inputs: &ChatTemplatesInputs) -> Result<String> {
        let src = self.source.as_deref().unwrap_or("");
        let messages: Vec<ChatMessage> = inputs
            .messages
            .iter()
            .map(|m| ChatMessage::new(m.role.clone(), m.content.clone()))
            .collect();
        let mut out = String::new();
        if crate::chat::chat_apply_template(src, &messages, &mut out, inputs.add_generation_prompt)
            < 0
        {
            return Err(Error::new(
                "this custom template is not supported, try using --jinja",
            ));
        }
        Ok(out)
    }
}

/// `chat_add_and_format` from the completion driver.
fn chat_add_and_format(
    templates: &ChatTemplates,
    chat_msgs: &mut Vec<ChatMsg>,
    role: &str,
    content: String,
) -> Result<String> {
    let new_msg = ChatMsg {
        role: role.to_owned(),
        content,
        ..ChatMsg::default()
    };
    let formatted = templates.format_single(chat_msgs, &new_msg, role == "user")?;
    chat_msgs.push(new_msg);
    eprintln!("formatted: '{formatted}'");
    Ok(formatted)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayType {
    Prompt,
    Error,
    UserInput,
    Reset,
}

/// Stand-in for `console` with the same display state transitions.
pub struct Console {
    pub display: DisplayType,
}

impl Console {
    /// `console::init(simple_io, use_color)`.
    pub fn init(_simple_io: bool, _use_color: bool) -> Self {
        Self {
            display: DisplayType::Reset,
        }
    }

    pub fn set_display(&mut self, display: DisplayType) {
        self.display = display;
    }

    /// `console::readline`: simple-mode semantics, including `/` and `\\`.
    pub fn readline(&mut self, line: &mut String, mut multiline: bool) -> bool {
        line.clear();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() || input.is_empty() {
            return false;
        }
        if input.ends_with('\n') {
            input.pop();
        }
        if input.ends_with('\r') {
            input.pop();
        }
        if let Some(last) = input.as_bytes().last().copied() {
            if last == b'/' {
                input.pop();
                input.push('\n');
                *line = input;
                return false;
            }
            if last == b'\\' {
                input.pop();
                multiline = !multiline;
            }
        }
        input.push('\n');
        *line = input;
        multiline
    }

    pub fn cleanup(&self) {}
}

/// `common_sampler`: the sampler chain plus the recent-token history.
pub struct SamplerState {
    chain: Sampler,
    prev: VecDeque<Token>,
    capacity: usize,
}

impl SamplerState {
    pub fn new(chain: Sampler, capacity: usize) -> Self {
        Self {
            chain,
            prev: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn chain(&self) -> &Sampler {
        &self.chain
    }

    pub fn chain_mut(&mut self) -> &mut Sampler {
        &mut self.chain
    }

    /// `common_sampler_sample`: apply the chain, do not accept here.
    pub fn sample(&mut self, logits: &[f32]) -> Result<Token> {
        let mut cur = TokenDataArray::new(
            logits
                .iter()
                .enumerate()
                .map(|(id, &logit)| TokenData {
                    id: id as Token,
                    logit,
                    p: 0.0,
                })
                .collect(),
        );
        api::llama_sampler_apply(&mut self.chain, &mut cur);
        let idx = usize::try_from(cur.selected)
            .map_err(|_| Error::new("llama_sampler_sample: invalid selected index"))?;
        cur.data
            .get(idx)
            .map(|t| t.id)
            .ok_or_else(|| Error::new("llama_sampler_sample: invalid selected index"))
    }

    /// `common_sampler_accept` (grammar sampling is not part of this port).
    pub fn accept(&mut self, token: Token, _accept_grammar: bool) {
        api::llama_sampler_accept(&mut self.chain, token);
        if self.capacity > 0 {
            if self.prev.len() == self.capacity {
                self.prev.pop_front();
            }
            self.prev.push_back(token);
        }
    }

    /// `common_sampler_last`.
    pub fn last(&self) -> Token {
        self.prev.back().copied().unwrap_or(api::LLAMA_TOKEN_NULL)
    }

    /// `common_sampler_prev_str`.
    pub fn prev_str(&self, vocab: &Vocab, n: usize) -> String {
        let n = n.min(self.prev.len());
        let mut out = String::with_capacity(8 * n);
        for i in (0..n).rev() {
            out.push_str(&token_to_piece(vocab, self.prev[i], false));
        }
        out
    }

    pub fn reset(&mut self) {
        self.prev.clear();
        api::llama_sampler_reset(&mut self.chain);
    }
}

/// Result of one completion run.
#[derive(Clone, Debug, Default)]
pub struct CompletionOutput {
    /// Text emitted by the display loop, in the same order as `tokens`.
    pub text: String,
    /// Generated tokens recorded as outgoing output.
    pub tokens: Vec<Token>,
    /// Tokens recorded as incoming requested input.
    pub input_tokens: Vec<Token>,
}

/// `print_usage`.
pub fn print_usage(program: &str) {
    eprintln!("\nexample usage:\n");
    eprintln!(
        "\n  text generation:     {program} -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n"
    );
    eprintln!(
        "\n  chat (conversation): {program} -m your_model.gguf -sys \"You are a helpful assistant\"\n"
    );
    eprintln!();
}

fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

fn file_is_empty(path: &str) -> bool {
    fs::metadata(path).map(|m| m.len() == 0).unwrap_or(false)
}

/// `common_token_to_piece` on the API vocabulary.
pub fn token_to_piece(vocab: &Vocab, token: Token, special: bool) -> String {
    let mut piece = vec![0u8; 16];
    let mut n = api::llama_token_to_piece(vocab, token, &mut piece, 0, special);
    if n < 0 {
        piece.resize((-n) as usize, 0);
        n = api::llama_token_to_piece(vocab, token, &mut piece, 0, special);
    }
    if n < 0 || n as usize > piece.len() {
        return String::new();
    }
    String::from_utf8_lossy(&piece[..n as usize]).into_owned()
}

/// `string_from(const llama_context *, tokens)`.
fn string_from_tokens(vocab: &Vocab, tokens: &[Token]) -> String {
    let mut out = String::from("[ ");
    for (i, token) in tokens.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(&token_to_piece(vocab, *token, true));
        out.push_str("':");
        out.push_str(&token.to_string());
    }
    out.push_str(" ]");
    out
}

/// `common_sampler_print`.
pub fn sampler_print(chain: &mut Sampler) -> String {
    let n = api::llama_sampler_chain_n(chain);
    let mut out = String::from("logits ");
    for i in 0..n {
        let name = api::llama_sampler_chain_get(chain, i)
            .and_then(|s| api::llama_sampler_name(s).map(str::to_owned))
            .unwrap_or_else(|| "?".to_owned());
        out.push_str("-> ");
        out.push_str(&name);
        out.push(' ');
    }
    out
}

/// `common_perf_print`.
pub fn perf_print(ctx: &Context, smpl: &Sampler) {
    let data_smpl = api::llama_perf_sampler(smpl);
    let data_ctx = api::llama_perf_context(ctx);

    eprintln!(
        "{}:    sampling time = {:10.2} ms\n",
        FN_NAME,
        1e-3 * sampler_total_us()
    );
    eprintln!(
        "{}:    samplers time = {:10.2} ms / {:5} tokens",
        FN_NAME, data_smpl.t_sample_ms, data_smpl.n_sample
    );

    let t_end_ms = api::llama_time_us() as f64 / 1000.0;
    let t_total_ms = t_end_ms - data_ctx.t_start_ms;
    let t_unacc_ms = t_total_ms - (sampler_total_us() / 1000.0 + data_ctx.t_p_eval_ms + data_ctx.t_eval_ms);
    let t_unacc_pc = 100.0 * t_unacc_ms / t_total_ms;

    eprintln!("{}:        load time = {:10.2} ms", FN_NAME, data_ctx.t_load_ms);
    eprintln!(
        "{}: prompt eval time = {:10.2} ms / {:5} tokens ({:8.2} ms per token, {:8.2} tokens per second)",
        FN_NAME,
        data_ctx.t_p_eval_ms,
        data_ctx.n_p_eval,
        data_ctx.t_p_eval_ms / data_ctx.n_p_eval as f64,
        1e3 / data_ctx.t_p_eval_ms * data_ctx.n_p_eval as f64
    );
    eprintln!(
        "{}:        eval time = {:10.2} ms / {:5} runs   ({:8.2} ms per token, {:8.2} tokens per second)",
        FN_NAME,
        data_ctx.t_eval_ms,
        data_ctx.n_eval,
        data_ctx.t_eval_ms / data_ctx.n_eval as f64,
        1e3 / data_ctx.t_eval_ms * data_ctx.n_eval as f64
    );
    eprintln!(
        "{}:       total time = {:10.2} ms / {:5} tokens",
        FN_NAME,
        t_total_ms,
        data_ctx.n_p_eval + data_ctx.n_eval
    );
    eprintln!(
        "{}: unaccounted time = {:10.2} ms / {:5.1} %      (total - sampling - prompt eval - eval) / (total)",
        FN_NAME, t_unacc_ms, t_unacc_pc
    );
    eprintln!("{}:    graphs reused = {:10}", FN_NAME, data_ctx.n_reused);
}

/// The sampler chain has no extra common-layer timing in this port.
fn sampler_total_us() -> f64 {
    0.0
}

fn to_numa(numa: u32) -> api::NumaStrategy {
    match numa {
        1 => api::NumaStrategy::Distribute,
        2 => api::NumaStrategy::Isolate,
        3 => api::NumaStrategy::Numactl,
        4 => api::NumaStrategy::Mirror,
        _ => api::NumaStrategy::Disabled,
    }
}

fn to_api_split(mode: common::SplitMode) -> api::SplitMode {
    match mode {
        common::SplitMode::None => api::SplitMode::None,
        common::SplitMode::Layer => api::SplitMode::Layer,
        common::SplitMode::Row => api::SplitMode::Row,
        common::SplitMode::Tensor => api::SplitMode::Tensor,
    }
}

fn to_api_load(mode: common::LoadMode) -> api::LoadMode {
    match mode {
        common::LoadMode::Auto => api::LoadMode::Auto,
        common::LoadMode::None => api::LoadMode::None,
        common::LoadMode::Mmap => api::LoadMode::Mmap,
        common::LoadMode::Mlock => api::LoadMode::Mlock,
        common::LoadMode::MmapMlock => api::LoadMode::MmapMlock,
        common::LoadMode::DirectIo => api::LoadMode::DirectIo,
    }
}

fn to_api_lazy(mode: common::LazyMode) -> api::LazyMode {
    match mode {
        common::LazyMode::Off => api::LazyMode::Off,
        common::LazyMode::Auto => api::LazyMode::Auto,
        common::LazyMode::On => api::LazyMode::On,
    }
}

fn to_api_kv(o: &common::KvOverride) -> api::ModelKvOverride {
    let (tag, value) = match o.tag {
        common::KvOverrideType::Int => (
            api::ModelKvOverrideType::Int,
            api::KvOverrideValue::I64(o.val_i64),
        ),
        common::KvOverrideType::Float => (
            api::ModelKvOverrideType::Float,
            api::KvOverrideValue::F64(o.val_f64),
        ),
        common::KvOverrideType::Bool => (
            api::ModelKvOverrideType::Bool,
            api::KvOverrideValue::Bool(o.val_bool),
        ),
        common::KvOverrideType::Str => (
            api::ModelKvOverrideType::Str,
            api::KvOverrideValue::Str(o.val_str.clone()),
        ),
    };
    api::ModelKvOverride {
        tag,
        key: o.key.clone(),
        value,
    }
}

/// `common_model_params_to_llama`, into the API stand-in type.
fn to_api_model_params(params: &Params) -> api::ModelParams {
    api::ModelParams {
        devices: params
            .devices
            .iter()
            .map(|&d| api::BackendDevice(d.to_string()))
            .collect(),
        tensor_buft_overrides: params
            .tensor_buft_overrides
            .iter()
            .map(|o| api::TensorBuftOverride {
                pattern: o.pattern.clone(),
                buft: Some(api::Buft(o.buffer_type.clone())),
            })
            .collect(),
        n_gpu_layers: params.n_gpu_layers,
        split_mode: to_api_split(params.split_mode),
        load_mode: to_api_load(params.load_mode),
        lazy_mode: to_api_lazy(params.lazy_mode),
        main_gpu: params.main_gpu,
        tensor_split: params.tensor_split.clone(),
        progress_callback: params.load_progress_callback,
        kv_overrides: params.kv_overrides.iter().map(to_api_kv).collect(),
        vocab_only: false,
        check_tensors: params.check_tensors,
        use_extra_bufts: !params.no_extra_bufts,
        no_host: params.no_host,
        no_alloc: params.no_alloc,
        load_mtp: params.load_mtp
            || params
                .speculative
                .types
                .contains(&common::SpeculativeType::DraftMtp),
    }
}

fn to_api_gtype(t: crate::gguf::GgmlType) -> api::GgmlType {
    use crate::gguf::GgmlType as G;
    match t {
        G::F32 => api::GgmlType::F32,
        G::F16 => api::GgmlType::F16,
        G::Q4_0 => api::GgmlType::Q4_0,
        G::Q4_1 => api::GgmlType::Q4_1,
        G::Q8_0 => api::GgmlType::Q8_0,
        G::Q8_1 => api::GgmlType::Q8_1,
        G::Q2K => api::GgmlType::Q2K,
        G::Q3K => api::GgmlType::Q3K,
        G::Q4K => api::GgmlType::Q4K,
        G::Q5K => api::GgmlType::Q5K,
        G::Q6K => api::GgmlType::Q6K,
        G::Q8K => api::GgmlType::Q8K,
        G::I8 => api::GgmlType::I8,
        G::I16 => api::GgmlType::I16,
        G::I32 => api::GgmlType::I32,
        G::I64 => api::GgmlType::I64,
        G::F64 => api::GgmlType::F64,
        G::Iq2Xxs => api::GgmlType::Iq2Xxs,
        G::Iq2Xs => api::GgmlType::Iq2Xs,
        G::Iq3Xxs => api::GgmlType::Iq3Xxs,
        G::Iq1S => api::GgmlType::Iq1S,
        G::Iq4Nl => api::GgmlType::Iq4Nl,
        G::Iq3S => api::GgmlType::Iq3S,
        G::Iq2S => api::GgmlType::Iq2S,
        G::Iq4Xs => api::GgmlType::Iq4Xs,
        G::Iq1M => api::GgmlType::Iq1M,
        G::Bf16 => api::GgmlType::Bf16,
        G::Tq1_0 => api::GgmlType::Tq1_0,
        G::Tq2_0 => api::GgmlType::Tq2_0,
        G::Q5_0
        | G::Q5_1
        | G::Mxfp4
        | G::Nvfp4
        | G::Q1_0
        | G::Q2_0
        | G::F8E4M3
        | G::Count => api::GgmlType::F16,
    }
}

fn to_api_pool(t: cparams::PoolingType) -> api::PoolingType {
    match t {
        cparams::PoolingType::Unspecified => api::PoolingType::Unspecified,
        cparams::PoolingType::None => api::PoolingType::None,
        cparams::PoolingType::Mean => api::PoolingType::Mean,
        cparams::PoolingType::Cls => api::PoolingType::Cls,
        cparams::PoolingType::Last => api::PoolingType::Last,
        cparams::PoolingType::Rank => api::PoolingType::Rank,
    }
}

fn to_api_attn(t: common::AttentionType) -> api::AttentionType {
    match t {
        common::AttentionType::Unspecified => api::AttentionType::Unspecified,
        common::AttentionType::Causal => api::AttentionType::Causal,
        common::AttentionType::NonCausal => api::AttentionType::NonCausal,
    }
}

fn to_api_flash(t: common::FlashAttnType) -> api::FlashAttnType {
    match t {
        common::FlashAttnType::Auto => api::FlashAttnType::Auto,
        common::FlashAttnType::Disabled => api::FlashAttnType::Disabled,
        common::FlashAttnType::Enabled => api::FlashAttnType::Enabled,
    }
}

fn to_api_rope(t: cparams::RopeScalingType) -> api::RopeScalingType {
    match t {
        cparams::RopeScalingType::Unspecified => api::RopeScalingType::Unspecified,
        cparams::RopeScalingType::None => api::RopeScalingType::None,
        cparams::RopeScalingType::Linear => api::RopeScalingType::Linear,
        cparams::RopeScalingType::Yarn => api::RopeScalingType::Yarn,
        cparams::RopeScalingType::Longrope => api::RopeScalingType::Longrope,
    }
}

/// `common_context_params_to_llama`, into the API stand-in type.
fn to_api_context_params(params: &Params) -> api::ContextParams {
    let mut cp = api::ContextParams::default();
    cp.n_ctx = params.n_ctx.max(0) as u32;
    cp.n_batch = params.n_batch.max(0) as u32;
    cp.n_ubatch = params.n_ubatch.max(0) as u32;
    cp.n_seq_max = params.n_parallel.max(0) as u32;
    cp.n_rs_seq = params.speculative.need_n_rs_seq();
    cp.n_outputs_max = params.n_outputs_max.max(0) as u32;
    cp.n_outputs_max_per_seq = params.n_outputs_max_per_seq.max(0) as u32;
    cp.n_threads = params.cpuparams.n_threads;
    cp.n_threads_batch = if params.cpuparams_batch.n_threads == -1 {
        cp.n_threads
    } else {
        params.cpuparams_batch.n_threads
    };
    cp.rope_scaling_type = to_api_rope(params.rope_scaling_type);
    cp.pooling_type = to_api_pool(params.pooling_type);
    cp.attention_type = to_api_attn(params.attention_type);
    cp.flash_attn_type = to_api_flash(params.flash_attn_type);
    cp.rope_freq_base = params.rope_freq_base;
    cp.rope_freq_scale = params.rope_freq_scale;
    cp.yarn_ext_factor = params.yarn_ext_factor;
    cp.yarn_attn_factor = params.yarn_attn_factor;
    cp.yarn_beta_fast = params.yarn_beta_fast;
    cp.yarn_beta_slow = params.yarn_beta_slow;
    cp.yarn_orig_ctx = params.yarn_orig_ctx.max(0) as u32;
    cp.embeddings = params.embedding;
    cp.type_k = to_api_gtype(params.cache_type_k);
    cp.type_v = to_api_gtype(params.cache_type_v);
    cp.moe_cache_size = params.moe_cache_size;
    cp.moe_cache_cpu_share = params.moe_cache_cpu_share;
    cp.offload_kqv = !params.no_kv_offload;
    cp.no_perf = params.no_perf;
    cp.op_offload = !params.no_op_offload;
    cp.swa_full = params.swa_full;
    cp.kv_unified = params.kv_unified;
    cp
}

fn init_from_params(params: &Params) -> Result<(Model, Context)> {
    let model = api::llama_model_load_from_file(&params.model.path, to_api_model_params(params))?;
    let ctx = api::llama_init_from_model(&model, to_api_context_params(params))?;
    Ok((model, ctx))
}

/// `common_sampler_init`, for the samplers this port supports.
fn init_sampler(params: &Params, vocab: &Vocab) -> Result<Sampler> {
    let sp = &params.sampling;
    let mut chain = api::llama_sampler_chain_init(api::SamplerChainParams { no_perf: sp.no_perf });
    let n_vocab = vocab.pieces.len() as i32;
    let seed = sp.seed;
    let min_keep = sp.min_keep.max(0) as usize;
    let mut use_adaptive_p = false;

    let mut biases: Vec<api::LogitBias> = sp
        .logit_bias
        .iter()
        .map(|b| api::LogitBias {
            token: b.token,
            bias: b.bias,
        })
        .collect();
    biases.extend(vocab.suppress.iter().map(|&token| api::LogitBias {
        token,
        bias: f32::NEG_INFINITY,
    }));
    if !biases.is_empty() {
        api::llama_sampler_chain_add(&mut chain, api::llama_sampler_init_logit_bias(n_vocab, &biases));
    }

    for kind in &sp.samplers {
        let sampler = match kind {
            SamplerType::Penalties => Some(api::llama_sampler_init_penalties(
                n_vocab,
                sp.penalty_last_n,
                sp.penalty_repeat,
                sp.penalty_freq,
                sp.penalty_present,
            )),
            SamplerType::Dry => {
                let breakers: Vec<&str> =
                    sp.dry_sequence_breakers.iter().map(String::as_str).collect();
                Some(api::llama_sampler_init_dry(
                    vocab,
                    sp.dry_multiplier,
                    sp.dry_base,
                    sp.dry_allowed_length,
                    sp.dry_penalty_last_n,
                    &breakers,
                ))
            }
            SamplerType::TopNSigma if sp.top_n_sigma > 0.0 => {
                Some(api::llama_sampler_init_top_n_sigma(sp.top_n_sigma))
            }
            SamplerType::TopK if sp.top_k > 0 => Some(api::llama_sampler_init_top_k(sp.top_k)),
            SamplerType::TypicalP if sp.typ_p < 1.0 => {
                Some(api::llama_sampler_init_typical(sp.typ_p, min_keep))
            }
            SamplerType::TopP if sp.top_p < 1.0 => {
                Some(api::llama_sampler_init_top_p(sp.top_p, min_keep))
            }
            SamplerType::MinP if sp.min_p > 0.0 => {
                Some(api::llama_sampler_init_min_p(sp.min_p, min_keep))
            }
            SamplerType::Xtc if sp.xtc_probability > 0.0 => Some(api::llama_sampler_init_xtc(
                sp.xtc_probability,
                sp.xtc_threshold,
                min_keep,
                seed,
            )),
            SamplerType::Temperature => Some(api::llama_sampler_init_temp_ext(
                sp.temp,
                sp.dynatemp_range,
                sp.dynatemp_exponent,
            )),
            SamplerType::Infill => Some(api::llama_sampler_init_infill(Rc::new((*vocab).clone()))),
            SamplerType::AdaptiveP => {
                // adaptive-p selects a token, so it goes at the end instead of dist.
                use_adaptive_p = true;
                None
            }
            _ => None,
        };
        if let Some(sampler) = sampler {
            api::llama_sampler_chain_add(&mut chain, sampler);
        }
    }

    if use_adaptive_p {
        api::llama_sampler_chain_add(
            &mut chain,
            api::llama_sampler_init_adaptive_p(sp.adaptive_target, sp.adaptive_decay, seed),
        );
    } else {
        let greedy = sp.n_probs == 0
            && sp.samplers.last().map_or(false, |last| {
                (*last == SamplerType::Temperature && sp.temp == 0.0 && sp.dynatemp_range == 0.0)
                    || (*last == SamplerType::TopK && sp.top_k == 1)
            });
        if greedy {
            api::llama_sampler_chain_add(&mut chain, api::llama_sampler_init_greedy());
        } else {
            api::llama_sampler_chain_add(&mut chain, api::llama_sampler_init_dist(seed));
        }
    }

    Ok(chain)
}

/// `common_replay_last_token`.
fn replay_last_token(ctx: &mut Context, token: Token, pos: i32) -> Result<bool> {
    let mut batch = api::llama_batch_get_one(&[token]);
    batch.pos = vec![pos];
    batch.n_seq_id = vec![1];
    batch.seq_id = vec![vec![0]];
    batch.logits = vec![1];
    if api::llama_decode(ctx, &batch) != 0 {
        return Ok(false);
    }
    Ok(true)
}

/// `common_prompt_batch_decode`.
fn prompt_batch_decode(
    ctx: &mut Context,
    all_tokens: &[Token],
    n_new: usize,
    n_past: &mut i32,
    n_batch: i32,
    state_path: &str,
    save_state: bool,
) -> Result<bool> {
    if n_new == 0 {
        return Ok(true);
    }
    let offset = all_tokens.len() - n_new;

    if save_state && n_new > 1 {
        let n_before = n_new - 1;
        if n_new as i32 > n_batch {
            return Err(Error::new("prompt batch exceeds n_batch"));
        }
        let batch = api::llama_batch_get_one(&all_tokens[offset..offset + n_before]);
        if api::llama_decode(ctx, &batch) != 0 {
            return Err(Error::new("failed to eval"));
        }
        *n_past += n_before as i32;
        api::llama_state_save_file(ctx, state_path, all_tokens, all_tokens.len())?;
        let last = all_tokens[all_tokens.len() - 1];
        let mut batch = api::llama_batch_get_one(&[last]);
        batch.pos = vec![*n_past];
        batch.n_seq_id = vec![1];
        batch.seq_id = vec![vec![0]];
        batch.logits = vec![1];
        if api::llama_decode(ctx, &batch) != 0 {
            return Err(Error::new("failed to eval last token"));
        }
        *n_past += 1;
    } else {
        let batch = api::llama_batch_get_one(&all_tokens[offset..offset + n_new]);
        if api::llama_decode(ctx, &batch) != 0 {
            return Err(Error::new("failed to eval"));
        }
        *n_past += n_new as i32;
    }
    Ok(true)
}

fn one_batch(ctx: &mut Context, tokens: &[Token], is_encode: bool) -> Result<bool> {
    let batch: Batch = api::llama_batch_get_one(tokens);
    let ret = if is_encode {
        api::llama_encode(ctx, &batch)
    } else {
        api::llama_decode(ctx, &batch)
    };
    Ok(ret == 0)
}

/// `llama_completion` from the example, with `common_params` passed in.
///
/// Argument parsing (`common_params_parse`) lives in `common/arg.cpp`, outside
/// this module, so the caller supplies the already-filled `Params`.
pub fn llama_completion(params: &mut Params) -> Result<CompletionOutput> {
    let mut console = Console::init(params.simple_io, params.use_color);

    if params.embedding {
        eprintln!("************");
        eprintln!("{FN_NAME}: please use the 'embedding' tool for embedding calculations");
        eprintln!("************\n");
        console.cleanup();
        return Ok(CompletionOutput::default());
    }

    if params.n_ctx != 0 && params.n_ctx < 8 {
        eprintln!("{FN_NAME}: warning: minimum context size is 8, using minimum size.");
        params.n_ctx = 8;
    }
    if params.rope_freq_base != 0.0 {
        eprintln!("{FN_NAME}: warning: changing RoPE frequency base to {}.", params.rope_freq_base);
    }
    if params.rope_freq_scale != 0.0 {
        eprintln!("{FN_NAME}: warning: scaling RoPE frequency by {}.", params.rope_freq_scale);
    }

    eprintln!("{FN_NAME}: llama backend init");
    api::llama_backend_init();
    api::llama_numa_init(to_numa(params.numa));

    eprintln!("{FN_NAME}: load the model and apply lora adapter, if any");
    let (model, mut ctx) = init_from_params(params)?;
    let vocab = ctx.model.vocab.clone();
    let smpl = init_sampler(params, &vocab)?;
    let mut sampler = SamplerState::new(smpl, params.sampling.n_prev.max(32) as usize);

    let templates = ChatTemplates::init(&model, &params.chat_template);
    api::llama_perf_context_reset(&mut ctx);

    let n_ctx_train = api::llama_model_n_ctx_train(&model);
    let n_ctx = api::llama_n_ctx(&ctx) as i32;
    if n_ctx > n_ctx_train {
        eprintln!(
            "{FN_NAME}: model was trained on only {n_ctx_train} context tokens ({n_ctx} specified)"
        );
    }

    let has_chat_template = templates.was_explicit();
    if params.conversation_mode == ConversationMode::Auto {
        if has_chat_template {
            eprintln!(
                "{FN_NAME}: chat template is available, enabling conversation mode (disable it with -no-cnv)"
            );
            params.conversation_mode = ConversationMode::Enabled;
        } else {
            params.conversation_mode = ConversationMode::Disabled;
        }
    }
    if params.conversation_mode != ConversationMode::Disabled && !has_chat_template {
        eprintln!(
            "{FN_NAME}: chat template is not available or is not supported. This may cause the model to output suboptimal responses"
        );
    }
    if params.conversation_mode != ConversationMode::Disabled {
        if params.enable_chat_template {
            if !params.prompt.is_empty() && params.system_prompt.is_empty() {
                eprintln!(
                    "*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?"
                );
            }
            let example = templates.example()?;
            eprintln!("{FN_NAME}: chat template example:\n{example}");
        } else {
            eprintln!("{FN_NAME}: in-suffix/prefix is specified, chat template will be disabled");
        }
    }

    eprintln!();
    eprintln!("{}", common::common_params_get_system_info(params));
    eprintln!();

    let mut path_session = params.path_prompt_cache.clone();
    let mut session_tokens: Vec<Token> = Vec::new();

    if !path_session.is_empty() {
        eprintln!("{FN_NAME}: attempting to load saved session from '{path_session}'");
        if !file_exists(&path_session) {
            eprintln!("{FN_NAME}: session file does not exist, will create.");
        } else if file_is_empty(&path_session) {
            eprintln!("{FN_NAME}: The session file is empty. A new session will be initialized.");
        } else {
            let capacity = n_ctx.max(0) as usize;
            session_tokens = vec![0; capacity];
            api::llama_state_load_file(&mut ctx, &path_session, Some(&mut session_tokens), capacity)?;
            eprintln!(
                "{FN_NAME}: loaded a session with prompt size of {} tokens",
                session_tokens.len()
            );
        }
    }

    let add_bos = api::llama_vocab_get_add_bos(&vocab) && !params.use_jinja;
    if !model.has_encoder {
        assert!(
            !api::llama_vocab_get_add_eos(&vocab),
            "non-encoder model adds EOS"
        );
    }
    eprintln!("n_ctx: {n_ctx}, add_bos: {add_bos}");

    let mut embd_inp: Vec<Token>;
    let mut waiting_for_first_input = false;
    let mut chat_msgs: Vec<ChatMsg> = Vec::new();
    let mut prompt: String;

    if params.conversation_mode != ConversationMode::Disabled && params.enable_chat_template {
        if !params.system_prompt.is_empty() {
            let _ = chat_add_and_format(
                &templates,
                &mut chat_msgs,
                "system",
                params.system_prompt.clone(),
            )?;
        }
        if !params.prompt.is_empty() {
            let _ = chat_add_and_format(&templates, &mut chat_msgs, "user", params.prompt.clone())?;
        } else {
            waiting_for_first_input = true;
        }
        if !params.system_prompt.is_empty() || !params.prompt.is_empty() {
            let mut inputs = ChatTemplatesInputs::new();
            inputs.use_jinja = params.use_jinja;
            inputs.messages = chat_msgs.clone();
            inputs.add_generation_prompt = !params.prompt.is_empty();
            inputs.force_pure_content = params.force_pure_content_parser;
            prompt = templates.apply(&inputs)?;
        } else {
            prompt = String::new();
        }
    } else {
        prompt = params.prompt.clone();
    }

    if params.interactive_first || !prompt.is_empty() || session_tokens.is_empty() {
        embd_inp = api::llama_tokenize(&vocab, prompt.as_bytes(), true, true)?;
    } else {
        embd_inp = session_tokens.clone();
    }
    eprintln!("prompt: \"{prompt}\"");
    eprintln!("tokens: {}", string_from_tokens(&vocab, &embd_inp));

    if !waiting_for_first_input && embd_inp.is_empty() {
        if add_bos {
            embd_inp.push(api::llama_vocab_bos(&vocab));
            eprintln!(
                "embd_inp was considered empty and bos was added: {}",
                string_from_tokens(&vocab, &embd_inp)
            );
        } else {
            return Err(Error::new("input is empty"));
        }
    }

    if embd_inp.len() as i32 > n_ctx - 4 {
        return Err(Error::new(format!(
            "{FN_NAME}: prompt is too long ({} tokens, max {})",
            embd_inp.len(),
            n_ctx - 4
        )));
    }

    let mut session_do_save: bool;
    let mut n_match = 0usize;
    if !session_tokens.is_empty() {
        for id in &session_tokens {
            if n_match >= embd_inp.len() || *id != embd_inp[n_match] {
                break;
            }
            n_match += 1;
        }
        if params.prompt.is_empty() && n_match == embd_inp.len() {
            eprintln!("{FN_NAME}: using full prompt from session file");
        } else if n_match >= embd_inp.len() {
            eprintln!("{FN_NAME}: session file has exact match for prompt!");
        } else if n_match < embd_inp.len() / 2 {
            eprintln!(
                "{FN_NAME}: session file has low similarity to prompt ({n_match} / {} tokens); will mostly be reevaluated",
                embd_inp.len()
            );
        } else {
            eprintln!(
                "{FN_NAME}: session file matches {n_match} / {} tokens of prompt",
                embd_inp.len()
            );
        }

        if session_tokens.len() > n_match {
            let pos = if n_match > 0 { (n_match - 1) as i32 } else { 0 };
            let mem = api::llama_get_memory(&mut ctx);
            if !api::llama_memory_seq_rm(mem, -1, pos, -1) {
                eprintln!("{FN_NAME}: unable to reuse common prefix (for example, when the memory is recurrent)");
                api::llama_memory_clear(mem, true);
                session_tokens.clear();
                n_match = 0;
            } else {
                session_tokens.truncate(n_match);
            }
        }
    }

    session_do_save = !path_session.is_empty() && n_match < embd_inp.len() && !params.prompt_cache_ro;

    if !session_tokens.is_empty() && n_match > 0 && n_match == session_tokens.len() {
        let last = session_tokens[session_tokens.len() - 1];
        if !replay_last_token(&mut ctx, last, (n_match - 1) as i32)? {
            return Err(Error::new("failed to replay last token"));
        }
        session_do_save = false;
        eprintln!("{FN_NAME}: replayed last token from session");
    }

    if params.n_keep < 0 || params.n_keep > embd_inp.len() as i32 {
        params.n_keep = embd_inp.len() as i32;
    } else {
        params.n_keep += add_bos as i32;
    }

    if params.conversation_mode != ConversationMode::Disabled {
        if params.single_turn && !params.prompt.is_empty() {
            params.interactive = false;
            params.interactive_first = false;
        } else {
            params.interactive_first = true;
        }
    }
    if params.interactive_first {
        params.interactive = true;
    }

    if params.verbose_prompt {
        eprintln!("{FN_NAME}: prompt: '{}'", params.prompt);
        eprintln!("{FN_NAME}: number of tokens in prompt = {}", embd_inp.len());
        for (i, token) in embd_inp.iter().enumerate() {
            eprintln!("{i:6} -> '{}'", token_to_piece(&vocab, *token, false));
        }
        if params.n_keep > add_bos as i32 {
            eprint!("{FN_NAME}: static prompt based on n_keep: '");
            for i in 0..params.n_keep as usize {
                eprint!("{}", token_to_piece(&vocab, embd_inp[i], false));
            }
            eprintln!("'");
        }
        eprintln!();
    }

    if params.interactive {
        eprintln!("{FN_NAME}: interactive mode on.");
        if !params.antiprompt.is_empty() {
            for antiprompt in &params.antiprompt {
                eprintln!("Reverse prompt: '{antiprompt}'");
                if params.verbose_prompt {
                    let tmp = api::llama_tokenize(&vocab, antiprompt.as_bytes(), false, true)?;
                    for (i, token) in tmp.iter().enumerate() {
                        eprintln!("{i:6} -> '{}'", token_to_piece(&vocab, *token, false));
                    }
                }
            }
        }
        if params.input_prefix_bos {
            eprintln!("Input prefix with BOS");
        }
        if !params.input_prefix.is_empty() {
            eprintln!("Input prefix: '{}'", params.input_prefix);
            if params.verbose_prompt {
                let tmp = api::llama_tokenize(&vocab, params.input_prefix.as_bytes(), true, true)?;
                for (i, token) in tmp.iter().enumerate() {
                    eprintln!("{i:6} -> '{}'", token_to_piece(&vocab, *token, false));
                }
            }
        }
        if !params.input_suffix.is_empty() {
            eprintln!("Input suffix: '{}'", params.input_suffix);
            if params.verbose_prompt {
                let tmp = api::llama_tokenize(&vocab, params.input_suffix.as_bytes(), false, true)?;
                for (i, token) in tmp.iter().enumerate() {
                    eprintln!("{i:6} -> '{}'", token_to_piece(&vocab, *token, false));
                }
            }
        }
    }

    eprintln!("sampler seed: {}", api::llama_sampler_get_seed(sampler.chain()));
    eprintln!("sampler params: \n{}", params.sampling.print());
    eprintln!("sampler chain: {}", sampler_print(sampler.chain_mut()));
    eprintln!(
        "generate: n_ctx = {n_ctx}, n_batch = {}, n_predict = {}, n_keep = {}",
        params.n_batch, params.n_predict, params.n_keep
    );

    let mut ga_i = 0i32;
    let ga_n = params.grp_attn_n;
    let ga_w = params.grp_attn_w;
    if ga_n != 1 {
        assert!(ga_n > 0, "grp_attn_n must be positive");
        assert!(ga_w % ga_n == 0, "grp_attn_w must be a multiple of grp_attn_n");
        eprintln!(
            "self-extend: n_ctx_train = {n_ctx_train}, grp_attn_n = {ga_n}, grp_attn_w = {ga_w}"
        );
    }
    eprintln!();

    if params.interactive {
        let control_message = if params.multiline_input {
            " - To return control to the AI, end your input with '\\'.\n - To return control without starting a new line, end your input with '/'.\n"
        } else {
            " - Press Return to return control to the AI.\n - To return control without starting a new line, end your input with '/'.\n - If you want to submit another line, end your input with '\\'.\n"
        };
        eprintln!("== Running in interactive mode. ==");
        eprintln!(" - Press Ctrl+C to interject at any time.");
        eprint!("{control_message}");
        if params.conversation_mode != ConversationMode::Disabled
            && params.enable_chat_template
            && params.system_prompt.is_empty()
        {
            eprintln!(" - Not using system message. To change it, set a different value via -sys PROMPT");
        }
        eprintln!();
        // The signal handler cannot be installed with std only; see omissions.
    }

    let mut is_interacting = params.interactive_first;
    let need_insert_eot = false;
    let mut is_antiprompt = false;
    let mut input_echo = true;
    let mut display;

    let mut n_past = 0i32;
    let mut n_remain = params.n_predict;
    let mut n_consumed = 0i32;
    let mut n_session_consumed = 0i32;

    let mut input_tokens: Vec<Token> = Vec::new();
    let mut output_tokens: Vec<Token> = Vec::new();
    let mut output_ss = String::new();
    let mut assistant_ss = String::new();

    console.set_display(DisplayType::Prompt);
    display = params.display_prompt;

    let mut embd: Vec<Token> = Vec::new();

    let mut antiprompt_token: Vec<Token> = Vec::new();
    for antiprompt in &params.antiprompt {
        let ids = api::llama_tokenize(&vocab, antiprompt.as_bytes(), false, true)?;
        if ids.len() == 1 {
            antiprompt_token.push(ids[0]);
        }
    }

    if model.has_encoder {
        if !one_batch(&mut ctx, &embd_inp, true)? {
            return Err(Error::new("failed to eval encoder batch"));
        }
        let mut decoder_start = api::llama_model_decoder_start_token(&model);
        if decoder_start == api::LLAMA_TOKEN_NULL {
            decoder_start = api::llama_vocab_bos(&vocab);
        }
        embd_inp.clear();
        embd_inp.push(decoder_start);
    }

    while (n_remain != 0 && !is_antiprompt) || params.interactive {
        if !embd.is_empty() {
            let max_embd_size = n_ctx - 4;
            if embd.len() as i32 > max_embd_size {
                let skipped = embd.len() as i32 - max_embd_size;
                embd.truncate(max_embd_size.max(0) as usize);
                console.set_display(DisplayType::Error);
                eprintln!(
                    "<<input too long: skipped {skipped} token{}>>",
                    if skipped != 1 { "s" } else { "" }
                );
                console.set_display(DisplayType::Reset);
            }

            if ga_n == 1 {
                if n_past + embd.len() as i32 >= n_ctx {
                    if !params.ctx_shift {
                        eprintln!("\n\n{FN_NAME}: context full and context shift is disabled => stopping");
                        break;
                    }
                    if params.n_predict == -2 {
                        eprintln!("\n\n{FN_NAME}: context full and n_predict == {} => stopping", params.n_predict);
                        break;
                    }
                    let n_left = n_past - params.n_keep;
                    let n_discard = n_left / 2;
                    {
                        let mem = api::llama_get_memory(&mut ctx);
                        api::llama_memory_seq_rm(mem, 0, params.n_keep, params.n_keep + n_discard);
                        api::llama_memory_seq_add(
                            mem,
                            0,
                            params.n_keep + n_discard,
                            n_past,
                            -n_discard,
                        );
                    }
                    n_past -= n_discard;
                    path_session.clear();
                }
            } else {
                while n_past >= ga_i + ga_w {
                    let ib = (ga_n * ga_i) / ga_w;
                    let bd = (ga_w / ga_n) * (ga_n - 1);
                    let dd = (ga_w / ga_n) - ib * bd - ga_w;
                    {
                        let mem = api::llama_get_memory(&mut ctx);
                        api::llama_memory_seq_add(mem, 0, ga_i, n_past, ib * bd);
                        api::llama_memory_seq_div(mem, 0, ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n);
                        api::llama_memory_seq_add(
                            mem,
                            0,
                            ga_i + ib * bd + ga_w,
                            n_past + ib * bd,
                            dd,
                        );
                    }
                    n_past -= bd;
                    ga_i += ga_w / ga_n;
                    eprintln!("n_past_old = {}, n_past = {n_past}, ga_i = {ga_i}", n_past + bd);
                }
            }

            if n_session_consumed < session_tokens.len() as i32 {
                let mut i = 0usize;
                while i < embd.len() {
                    if embd[i] != session_tokens[n_session_consumed as usize] {
                        session_tokens.truncate(n_session_consumed as usize);
                        break;
                    }
                    n_past += 1;
                    n_session_consumed += 1;
                    if n_session_consumed as usize >= session_tokens.len() {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                if i > 0 {
                    embd.drain(..i);
                }
            }

            if !embd.is_empty() {
                let is_last_batch = n_consumed as usize >= embd_inp.len();
                let save_now = session_do_save && is_last_batch;
                session_tokens.extend_from_slice(&embd);
                if !prompt_batch_decode(
                    &mut ctx,
                    &session_tokens,
                    embd.len(),
                    &mut n_past,
                    params.n_batch,
                    &path_session,
                    save_now,
                )? {
                    return Err(Error::new("failed to decode prompt batch"));
                }
                n_session_consumed += embd.len() as i32;
                if save_now {
                    session_do_save = false;
                }
                if params.n_print > 0 && n_past % params.n_print == 0 {
                    eprintln!("\nTokens consumed so far = {n_past} / {n_ctx}\n");
                }
            }
        }
        embd.clear();

        if embd_inp.len() as i32 <= n_consumed && !is_interacting {
            let id = sampler.sample(api::llama_get_logits(&ctx))?;
            sampler.accept(id, true);
            embd.push(id);
            if params.conversation_mode != ConversationMode::Disabled
                && !waiting_for_first_input
                && !api::llama_vocab_is_eog(&vocab, id)
            {
                assistant_ss.push_str(&token_to_piece(&vocab, id, false));
            }
            input_echo = true;
            n_remain -= 1;
        } else {
            while embd_inp.len() as i32 > n_consumed {
                let token = embd_inp[n_consumed as usize];
                embd.push(token);
                sampler.accept(token, false);
                n_consumed += 1;
                if embd.len() as i32 == params.n_batch {
                    break;
                }
            }
        }

        if input_echo && display {
            for id in &embd {
                let piece = token_to_piece(&vocab, *id, params.special);
                print!("{piece}");
                if embd.len() > 1 {
                    input_tokens.push(*id);
                } else {
                    output_tokens.push(*id);
                    output_ss.push_str(&piece);
                }
            }
            let _ = io::stdout().flush();
        }

        if input_echo && embd_inp.len() as i32 == n_consumed {
            console.set_display(DisplayType::Reset);
            display = true;
        }

        if embd_inp.len() as i32 <= n_consumed {
            if !params.antiprompt.is_empty() {
                let n_prev = 32;
                let last_output = sampler.prev_str(&vocab, n_prev);
                is_antiprompt = false;
                for antiprompt in &params.antiprompt {
                    let extra_padding = if params.interactive { 0 } else { 2 };
                    let search_start = if last_output.len() > antiprompt.len() + extra_padding {
                        last_output.len() - antiprompt.len() - extra_padding
                    } else {
                        0
                    };
                    if last_output
                        .get(search_start..)
                        .map_or(false, |tail| tail.contains(antiprompt))
                    {
                        if params.interactive {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }
                if !last_output.is_empty() {
                    let last_token = sampler.last();
                    for token in &antiprompt_token {
                        if *token == last_token {
                            if params.interactive {
                                is_interacting = true;
                            }
                            is_antiprompt = true;
                            break;
                        }
                    }
                }
                if is_antiprompt {
                    eprintln!("found antiprompt: {last_output}");
                }
            }

            if !waiting_for_first_input && api::llama_vocab_is_eog(&vocab, sampler.last()) {
                if params.interactive {
                    if !params.antiprompt.is_empty() {
                        let first = api::llama_tokenize(
                            &vocab,
                            params.antiprompt[0].as_bytes(),
                            false,
                            true,
                        )?;
                        embd_inp.extend_from_slice(&first);
                        is_antiprompt = true;
                    }
                    if params.enable_chat_template {
                        let assistant = std::mem::take(&mut assistant_ss);
                        let _ = chat_add_and_format(
                            &templates,
                            &mut chat_msgs,
                            "assistant",
                            assistant,
                        )?;
                    }
                    is_interacting = true;
                    println!();
                }
            }

            if params.conversation_mode != ConversationMode::Disabled && !waiting_for_first_input {
                if !prompt.is_empty() {
                    prompt.clear();
                    is_interacting = false;
                }
            }

            if (n_past > 0 || waiting_for_first_input) && is_interacting {
                if params.conversation_mode != ConversationMode::Disabled {
                    print!("\n> ");
                }
                if params.input_prefix_bos {
                    embd_inp.push(api::llama_vocab_bos(&vocab));
                }
                let mut buffer = String::new();
                if !params.input_prefix.is_empty()
                    && params.conversation_mode == ConversationMode::Disabled
                {
                    print!("{}", params.input_prefix);
                }
                console.set_display(DisplayType::UserInput);
                loop {
                    let mut line = String::new();
                    let another_line = console.readline(&mut line, params.multiline_input);
                    buffer.push_str(&line);
                    if !another_line {
                        break;
                    }
                }
                console.set_display(DisplayType::Reset);
                display = true;

                if buffer.is_empty() {
                    println!("EOF by user");
                    break;
                }
                if buffer.ends_with('\n') {
                    buffer.pop();
                }
                if buffer.is_empty() {
                    // Enter on an empty line passes control back.
                } else {
                    if !params.input_suffix.is_empty()
                        && params.conversation_mode == ConversationMode::Disabled
                    {
                        print!("{}", params.input_suffix);
                    }
                    let original_size = embd_inp.len();
                    if params.escape {
                        common::string_process_escapes(&mut buffer);
                    }
                    let format_chat = params.conversation_mode != ConversationMode::Disabled
                        && params.enable_chat_template;
                    let user_inp = if format_chat {
                        chat_add_and_format(
                            &templates,
                            &mut chat_msgs,
                            "user",
                            std::mem::take(&mut buffer),
                        )?
                    } else {
                        buffer
                    };
                    let line_pfx =
                        api::llama_tokenize(&vocab, params.input_prefix.as_bytes(), false, true)?;
                    let line_inp = api::llama_tokenize(
                        &vocab,
                        user_inp.as_bytes(),
                        false,
                        format_chat,
                    )?;
                    let line_sfx =
                        api::llama_tokenize(&vocab, params.input_suffix.as_bytes(), false, true)?;

                    if need_insert_eot && format_chat {
                        let eot = api::llama_vocab_eot(&vocab);
                        embd_inp.push(if eot == api::LLAMA_TOKEN_NULL {
                            api::llama_vocab_eos(&vocab)
                        } else {
                            eot
                        });
                    }
                    embd_inp.extend_from_slice(&line_pfx);
                    embd_inp.extend_from_slice(&line_inp);
                    embd_inp.extend_from_slice(&line_sfx);

                    for i in original_size..embd_inp.len() {
                        let token = embd_inp[i];
                        let piece = token_to_piece(&vocab, token, false);
                        output_tokens.push(token);
                        output_ss.push_str(&piece);
                    }
                    assistant_ss.clear();
                    n_remain -= line_inp.len() as i32;
                }
                input_echo = false;
            }

            if n_past > 0 || waiting_for_first_input {
                if is_interacting {
                    sampler.reset();
                }
                is_interacting = false;
                if waiting_for_first_input && params.single_turn {
                    params.interactive = false;
                    params.interactive_first = false;
                }
                waiting_for_first_input = false;
            }
        }

        if !embd.is_empty()
            && api::llama_vocab_is_eog(&vocab, *embd.last().unwrap())
            && !params.interactive
        {
            println!(" [end of text]");
            break;
        }

        if params.interactive && n_remain <= 0 && params.n_predict >= 0 {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    if !path_session.is_empty() && params.prompt_cache_all && !params.prompt_cache_ro {
        println!(
            "\n{FN_NAME}: saving final output to session file '{path_session}'"
        );
        session_tokens.extend_from_slice(&embd);
        api::llama_state_save_file(&ctx, &path_session, &session_tokens, session_tokens.len())?;
        eprintln!(
            "saved final session to {path_session}, n_tokens = {}",
            session_tokens.len()
        );
    }

    println!("\n\n");
    perf_print(&ctx, sampler.chain());
    console.cleanup();
    api::llama_backend_free();

    Ok(CompletionOutput {
        text: output_ss,
        tokens: output_tokens,
        input_tokens,
    })
}

// Omissions:
// - common_params_parse (common/arg.cpp) is not in this crate; callers fill Params.
// - SIGINT handling, console colors and advanced (readline) input need OS/FFI code.
// - Jinja templates, reasoning budget and grammar samplers use crates not ported here.
// - Inference is the crate's dry stand-in; no numeric graph execution is performed.

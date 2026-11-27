//! Port of `tools/server/server-task.{h,cpp}`.
//!
//! Task and result types plus the JSON protocol used by every HTTP route.

use crate::chat::{chat_format_name, reasoning_format_name, ChatFormat};
use crate::common::{
    GrammarTrigger as CommonGrammarTrigger, ParamsSampling, ParamsSpeculative, SpeculativeType,
    ReasoningFormat as CommonReasoningFormat,
};
use crate::core::{Result, Token};
use crate::json::{Item, Json};
use crate::tools::server::common::{
    format_error_response, json_get_nested_values, safe_json_to_str, ErrorType, Metrics, ServerTokens, SlotStats,
};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

/// `enum server_task_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServerTaskType {
    Completion,
    Embedding,
    Rerank,
    Decision,
    Infill,
    TranscribeRt,
    Realtime,
    Cancel,
    Control,
    NextResponse,
    Metrics,
    SlotGet,
    SlotSave,
    SlotRestore,
    SlotErase,
    GetLora,
    SetLora,
    Tts,
}

/// `enum task_response_type`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TaskResponseType {
    #[default]
    None,
    OaiChat,
    OaiCmpl,
    OaiResp,
    OaiAsr,
    OaiEmbd,
    Anthropic,
}

/// `enum stop_type`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StopType {
    #[default]
    None,
    Eos,
    Word,
    Limit,
}

impl StopType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Eos => "eos",
            Self::Word => "word",
            Self::Limit => "limit",
        }
    }
}

/// `enum server_control_action`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ServerControlAction {
    #[default]
    None,
    ReasoningEnd,
    AudioAppend,
    AudioCommit,
    AudioEnd,
    TextAppend,
    ResponseCreate,
    ResponseCancel,
    Clear,
}

/// Grammar trigger with JSON (de)serialization, from `server-common.h`.
#[derive(Clone, Debug)]
pub struct GrammarTrigger {
    pub kind: i32,
    pub value: String,
    pub token: Token,
}

impl GrammarTrigger {
    /// `server_grammar_trigger(const json & in)`.
    pub fn from_json(json: &Json) -> Result<Self> {
        Ok(Self {
            kind: json.at("type")?.get::<i32>()?,
            value: json.at("value")?.get::<String>()?,
            token: json.value("token", -1i32)?,
        })
    }

    /// `server_grammar_trigger(const common_grammar_trigger & value)`.
    pub fn from_common(trigger: &CommonGrammarTrigger) -> Self {
        Self {
            kind: trigger.type_ as u32 as i32,
            value: trigger.value.clone(),
            token: trigger.token,
        }
    }

    pub fn to_json(&self) -> Json {
        let mut out = Json::object_from(vec![
            Item::new("type", self.kind),
            Item::new("value", self.value.clone()),
        ]);
        if self.kind == 0 {
            let _ = out.set(Item::new("token", self.token));
        }
        out
    }
}

/// `common_speculative_type_name_str()`.
fn speculative_name(kind: SpeculativeType) -> &'static str {
    match kind {
        SpeculativeType::None => "none",
        SpeculativeType::DraftSimple => "draft",
        SpeculativeType::DraftEagle3 => "eagle3",
        SpeculativeType::DraftMtp => "mtp",
        SpeculativeType::DraftDflash => "dflash",
        SpeculativeType::DraftDspark => "dspark",
        SpeculativeType::DraftDflash2 => "dflash2",
        SpeculativeType::DraftLilicorr => "lilirc",
        SpeculativeType::NgramSimple => "ngram-simple",
        SpeculativeType::NgramMapK => "ngram-map-k",
        SpeculativeType::NgramMapK4v => "ngram-map-k4v",
        SpeculativeType::NgramMod => "ngram-mod",
        SpeculativeType::NgramCache => "ngram-cache",
    }
}

/// Convert the CLI reasoning format into the chat-side one used in responses.
fn reasoning_format(params: CommonReasoningFormat) -> crate::chat::ReasoningFormat {
    match params {
        CommonReasoningFormat::None => crate::chat::ReasoningFormat::None,
        CommonReasoningFormat::Auto => crate::chat::ReasoningFormat::Auto,
        CommonReasoningFormat::DeepseekLegacy => crate::chat::ReasoningFormat::DeepseekLegacy,
        CommonReasoningFormat::Deepseek => crate::chat::ReasoningFormat::Deepseek,
    }
}

/// `task_params`.
#[derive(Clone, Debug)]
pub struct TaskParams {
    pub stream: bool,
    pub include_usage: bool,
    pub cache_prompt: bool,
    pub return_tokens: bool,
    pub return_progress: bool,
    pub sse_ping_interval: i32,
    pub n_keep: i32,
    pub n_discard: i32,
    pub n_predict: i32,
    pub n_indent: i32,
    pub n_cmpl: i32,
    pub n_cache_reuse: i32,
    pub t_max_prompt_ms: i64,
    pub t_max_predict_ms: i64,
    pub lora: BTreeMap<i32, f32>,
    pub antiprompt: Vec<String>,
    pub response_fields: Vec<String>,
    pub timings_per_token: bool,
    pub post_sampling_probs: bool,
    pub sampling: ParamsSampling,
    pub speculative: ParamsSpeculative,
    pub verbose: bool,
    pub res_type: TaskResponseType,
    pub oaicompat_model: String,
    pub oaicompat_cmpl_id: String,
    pub control_action: ServerControlAction,
    pub control_cmpl_id: String,
    pub control_audio: Vec<f32>,
    pub control_text: String,
    pub chat_format: ChatFormat,
    pub reasoning_format: crate::chat::ReasoningFormat,
    pub embd_normalize: i32,
}

impl Default for TaskParams {
    fn default() -> Self {
        Self {
            stream: false,
            include_usage: false,
            cache_prompt: true,
            return_tokens: false,
            return_progress: false,
            sse_ping_interval: 30,
            n_keep: 0,
            n_discard: 0,
            n_predict: -1,
            n_indent: 0,
            n_cmpl: 1,
            n_cache_reuse: 0,
            t_max_prompt_ms: -1,
            t_max_predict_ms: -1,
            lora: BTreeMap::new(),
            antiprompt: Vec::new(),
            response_fields: Vec::new(),
            timings_per_token: false,
            post_sampling_probs: false,
            sampling: ParamsSampling::default(),
            speculative: ParamsSpeculative::default(),
            verbose: false,
            res_type: TaskResponseType::None,
            oaicompat_model: String::new(),
            oaicompat_cmpl_id: String::new(),
            control_action: ServerControlAction::None,
            control_cmpl_id: String::new(),
            control_audio: Vec::new(),
            control_text: String::new(),
            chat_format: ChatFormat::default(),
            reasoning_format: crate::chat::ReasoningFormat::default(),
            embd_normalize: 2,
        }
    }
}

impl TaskParams {
    /// Build defaults from the CLI parameters, like `task_params.sampling = params.sampling`.
    pub fn from_params(params: &crate::common::Params) -> Self {
        let mut out = Self::default();
        out.sampling = params.sampling.clone();
        out.speculative = params.speculative.clone();
        out.n_predict = params.n_predict;
        out.n_keep = params.n_keep;
        out.n_cache_reuse = params.n_cache_reuse;
        out.cache_prompt = params.cache_prompt;
        out.timings_per_token = params.sampling.timing_per_token;
        out.sse_ping_interval = params.sse_ping_interval;
        out.embd_normalize = params.embd_normalize;
        out.reasoning_format = reasoning_format(params.reasoning_format);
        out
    }

    fn base_json(&self) -> Vec<Item> {
        let samplers: Vec<String> = self.sampling.samplers.iter().map(|s| s.as_str().to_string()).collect();
        let speculative: Vec<String> = self.speculative.types.iter().copied().map(speculative_name).map(str::to_string).collect();
        let lora: Vec<Json> = self
            .lora
            .iter()
            .map(|(id, scale)| Json::object_from(vec![Item::new("id", *id), Item::new("scale", *scale)]))
            .collect();
        vec![
            Item::new("seed", self.sampling.seed as u64),
            Item::new("temperature", self.sampling.temp),
            Item::new("dynatemp_range", self.sampling.dynatemp_range),
            Item::new("dynatemp_exponent", self.sampling.dynatemp_exponent),
            Item::new("top_k", self.sampling.top_k),
            Item::new("top_p", self.sampling.top_p),
            Item::new("min_p", self.sampling.min_p),
            Item::new("top_n_sigma", self.sampling.top_n_sigma),
            Item::new("xtc_probability", self.sampling.xtc_probability),
            Item::new("xtc_threshold", self.sampling.xtc_threshold),
            Item::new("typical_p", self.sampling.typ_p),
            Item::new("repeat_last_n", self.sampling.penalty_last_n),
            Item::new("repeat_penalty", self.sampling.penalty_repeat),
            Item::new("presence_penalty", self.sampling.penalty_present),
            Item::new("frequency_penalty", self.sampling.penalty_freq),
            Item::new("dry_multiplier", self.sampling.dry_multiplier),
            Item::new("dry_base", self.sampling.dry_base),
            Item::new("dry_allowed_length", self.sampling.dry_allowed_length),
            Item::new("dry_penalty_last_n", self.sampling.dry_penalty_last_n),
            Item::new("mirostat", self.sampling.mirostat),
            Item::new("mirostat_tau", self.sampling.mirostat_tau),
            Item::new("mirostat_eta", self.sampling.mirostat_eta),
            Item::new("adaptive_target", self.sampling.adaptive_target),
            Item::new("adaptive_decay", self.sampling.adaptive_decay),
            Item::new("max_tokens", self.n_predict),
            Item::new("n_predict", self.n_predict),
            Item::new("n_keep", self.n_keep),
            Item::new("n_discard", self.n_discard),
            Item::new("ignore_eos", self.sampling.ignore_eos),
            Item::new("stream", self.stream),
            Item::new("n_probs", self.sampling.n_probs),
            Item::new("min_keep", self.sampling.min_keep),
            Item::new("chat_format", chat_format_name(self.chat_format)),
            Item::new("reasoning_format", reasoning_format_name(self.reasoning_format)),
            Item::new(
                "reasoning_in_content",
                self.stream && self.reasoning_format == crate::chat::ReasoningFormat::DeepseekLegacy,
            ),
            Item::new("generation_prompt", self.sampling.generation_prompt.clone()),
            Item::new("samplers", samplers),
            Item::new("timings_per_token", self.timings_per_token),
            Item::new("post_sampling_probs", self.post_sampling_probs),
            Item::new("backend_sampling", self.sampling.backend_sampling),
            Item::new("speculative.types", speculative.join(",")),
            Item::new("lora", lora),
        ]
    }

    /// `task_params::to_json()`.
    pub fn to_json(&self, only_metrics: bool) -> Json {
        let mut items = self.base_json();
        if only_metrics {
            return Json::object_from(items);
        }
        let triggers: Vec<Json> = self
            .sampling
            .grammar_triggers
            .iter()
            .map(GrammarTrigger::from_common)
            .map(|t| t.to_json())
            .collect();
        let grammar = Json::object_from(vec![
            Item::new("type", self.sampling.grammar.type_ as u32 as i32),
            Item::new("value", self.sampling.grammar.grammar.clone()),
        ]);
        items.extend([
            Item::new("dry_sequence_breakers", self.sampling.dry_sequence_breakers.clone()),
            Item::new("grammar", grammar),
            Item::new("grammar_lazy", self.sampling.grammar_lazy),
            Item::new("grammar_triggers", triggers),
            Item::new("preserved_tokens", Vec::<Token>::from_iter(self.sampling.preserved_tokens.iter().copied())),
            Item::new("antiprompt", self.antiprompt.clone()),
            Item::new("n_cache_reuse", self.n_cache_reuse),
            Item::new("n_indent", self.n_indent),
            Item::new("n_cmpl", self.n_cmpl),
            Item::new("embd_normalize", self.embd_normalize),
        ]);
        Json::object_from(items)
    }
}

/// `result_prompt_progress`.
#[derive(Clone, Debug, Default)]
pub struct ResultPromptProgress {
    pub total: i32,
    pub cache: i32,
    pub processed: i32,
    pub time_ms: i64,
}

impl ResultPromptProgress {
    pub fn to_json(&self) -> Json {
        Json::object_from(vec![
            Item::new("total", self.total),
            Item::new("cache", self.cache),
            Item::new("processed", self.processed),
            Item::new("time_ms", self.time_ms),
        ])
    }
}

/// `completion_token_output`.
#[derive(Clone, Debug, Default)]
pub struct CompletionTokenOutput {
    pub token: Token,
    pub prob: f32,
    pub text_to_send: String,
    pub probs: Vec<ProbInfo>,
}

#[derive(Clone, Debug, Default)]
pub struct ProbInfo {
    pub token: Token,
    pub text: String,
    pub prob: f32,
}

impl CompletionTokenOutput {
    pub fn to_json(&self, post_sampling_probs: bool) -> Json {
        let mut out = Json::object_from(vec![
            Item::new("id", self.token),
            Item::new("prob", self.prob),
            Item::new("text", self.text_to_send.clone()),
        ]);
        let probs: Vec<Json> = self
            .probs
            .iter()
            .map(|p| {
                Json::object_from(vec![
                    Item::new("id", p.token),
                    Item::new("text", p.text.clone()),
                    Item::new("prob", p.prob),
                ])
            })
            .collect();
        let key = if post_sampling_probs { "post_sampling_probs" } else { "probs" };
        let _ = out.set(Item::new(key, probs));
        out
    }

    pub fn probs_vector_to_json(outputs: &[Self], post_sampling_probs: bool) -> Vec<Json> {
        outputs.iter().map(|o| o.to_json(post_sampling_probs)).collect()
    }

    /// `-log2(x)` from `completion_token_output::logarithm()`.
    pub fn logarithm(x: f32) -> f32 {
        -x.log2()
    }

    /// Byte values of a token piece.
    pub fn str_to_bytes(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }
}

/// `server_task`.
#[derive(Clone, Debug)]
pub struct ServerTask {
    pub id: i32,
    pub index: usize,
    pub id_target: i32,
    pub id_slot: i32,
    pub id_parent: i32,
    pub n_tokens_shared: i32,
    pub child_tasks: Vec<ServerTask>,
    pub params: TaskParams,
    pub tokens: ServerTokens,
    pub kind: ServerTaskType,
    pub slot_action: SlotAction,
    pub metrics_reset_bucket: bool,
    pub set_lora: BTreeMap<i32, f32>,
}

#[derive(Clone, Debug, Default)]
pub struct SlotAction {
    pub id_slot: i32,
    pub filename: String,
    pub filepath: String,
}

impl ServerTask {
    pub fn new(kind: ServerTaskType) -> Self {
        Self {
            id: -1,
            index: 0,
            id_target: -1,
            id_slot: -1,
            id_parent: -1,
            n_tokens_shared: 0,
            child_tasks: Vec::new(),
            params: TaskParams::default(),
            tokens: ServerTokens::default(),
            kind,
            slot_action: SlotAction::default(),
            metrics_reset_bucket: false,
            set_lora: BTreeMap::new(),
        }
    }

    pub fn n_tokens(&self) -> i32 {
        self.tokens.size() as i32
    }

    pub fn need_embd(&self) -> bool {
        matches!(self.kind, ServerTaskType::Embedding | ServerTaskType::Rerank)
    }

    pub fn need_logits(&self) -> bool {
        matches!(self.kind, ServerTaskType::Completion | ServerTaskType::Infill | ServerTaskType::TranscribeRt)
    }

    pub fn need_sampling(&self) -> bool {
        matches!(self.kind, ServerTaskType::Completion | ServerTaskType::Infill | ServerTaskType::Tts | ServerTaskType::TranscribeRt)
    }

    pub fn is_parent(&self) -> bool {
        !self.child_tasks.is_empty()
    }

    pub fn is_child(&self) -> bool {
        self.id_parent != -1
    }

    pub fn is_stateless(&self) -> bool {
        matches!(self.kind, ServerTaskType::Embedding | ServerTaskType::Rerank)
    }

    pub fn get_list_id(tasks: &[Self]) -> Vec<i32> {
        let mut ids = Vec::new();
        for task in tasks {
            ids.push(task.id);
            ids.extend(task.child_tasks.iter().map(|c| c.id));
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

/// A chat message as returned by the OpenAI-compatible endpoints.
#[derive(Clone, Debug, Default)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<Json>,
    pub reasoning: String,
}

impl ResponseMessage {
    pub fn to_json(&self) -> Json {
        let mut out = Json::object_from(vec![
            Item::new("role", self.role.clone()),
            Item::new("content", self.content.clone()),
        ]);
        if !self.tool_calls.is_empty() {
            let _ = out.set(Item::new("tool_calls", self.tool_calls.clone()));
        }
        if !self.reasoning.is_empty() {
            let _ = out.set(Item::new("reasoning_content", self.reasoning.clone()));
        }
        out
    }
}

/// `server_task_result_cmpl_final`, the common shape behind completion routes.
#[derive(Clone, Debug)]
pub struct CompletionResult {
    pub id: i32,
    pub id_slot: i32,
    pub index: usize,
    pub content: String,
    pub tokens: Vec<Token>,
    pub stream: bool,
    pub include_usage: bool,
    pub stats: SlotStats,
    pub prompt: String,
    pub truncated: bool,
    pub n_decoded: i32,
    pub n_prompt_tokens: i32,
    pub n_prompt_tokens_cache: i32,
    pub n_tokens_cached: i32,
    pub has_new_line: bool,
    pub stopping_word: String,
    pub stop: StopType,
    pub probs_output: Vec<CompletionTokenOutput>,
    pub post_sampling_probs: bool,
    pub response_fields: Vec<String>,
    pub generation_params: TaskParams,
    pub verbose: bool,
    pub res_type: TaskResponseType,
    pub oaicompat_model: String,
    pub oaicompat_cmpl_id: String,
    pub message: ResponseMessage,
    pub msg_diffs: Vec<Json>,
}

impl CompletionResult {
    fn created() -> i64 {
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
    }

    fn usage_json(&self) -> Json {
        Json::object_from(vec![
            Item::new("completion_tokens", self.n_decoded),
            Item::new("prompt_tokens", self.n_prompt_tokens),
            Item::new("total_tokens", self.n_decoded + self.n_prompt_tokens),
            Item::new("prompt_tokens_details", Json::object_from(vec![Item::new("cached_tokens", self.n_prompt_tokens_cache)])),
        ])
    }

    /// `server_task_result_cmpl_final::to_json_non_oaicompat()`.
    pub fn to_json_non_oaicompat(&self) -> Json {
        let mut res = Json::object_from(vec![
            Item::new("index", self.index as i64),
            Item::new("content", self.content.clone()),
            Item::new("tokens", self.tokens.clone()),
            Item::new("id_slot", self.id_slot),
            Item::new("stop", true),
            Item::new("model", self.oaicompat_model.clone()),
            Item::new("tokens_predicted", self.n_decoded),
            Item::new("tokens_evaluated", self.n_prompt_tokens),
            Item::new("generation_settings", self.generation_params.to_json(false)),
            Item::new("prompt", self.prompt.clone()),
            Item::new("has_new_line", self.has_new_line),
            Item::new("truncated", self.truncated),
            Item::new("stop_type", self.stop.as_str()),
            Item::new("stopping_word", self.stopping_word.clone()),
            Item::new("tokens_cached", self.n_tokens_cached),
            Item::new("timings", self.stats.to_json()),
        ]);
        if !self.stream && !self.probs_output.is_empty() {
            let _ = res.set(Item::new(
                "completion_probabilities",
                CompletionTokenOutput::probs_vector_to_json(&self.probs_output, self.post_sampling_probs),
            ));
        }
        if self.response_fields.is_empty() {
            res
        } else {
            json_get_nested_values(&self.response_fields, &res)
        }
    }

    fn base_oai(&self, object: &str, choice: Json) -> Json {
        let mut res = Json::object_from(vec![
            Item::new("choices", Json::array_from(vec![choice.clone()])),
            Item::new("created", Self::created()),
            Item::new("model", self.oaicompat_model.clone()),
            Item::new("system_fingerprint", crate::api::LLAMA_VERSION),
            Item::new("object", object),
            Item::new("usage", self.usage_json()),
            Item::new("id", self.oaicompat_cmpl_id.clone()),
        ]);
        if self.verbose {
            let _ = res.set(Item::new("__verbose", self.to_json_non_oaicompat()));
        }
        if self.stats.is_set() {
            let _ = res.set(Item::new("timings", self.stats.to_json()));
        }
        res
    }

    fn finish_reason(&self, has_tool_calls: bool) -> &'static str {
        if matches!(self.stop, StopType::Word | StopType::Eos) {
            if has_tool_calls { "tool_calls" } else { "stop" }
        } else {
            "length"
        }
    }

    /// `to_json_oaicompat()` (legacy text completion).
    pub fn to_json_oaicompat(&self) -> Json {
        let logprobs = if !self.stream && !self.probs_output.is_empty() {
            Json::object_from(vec![Item::new(
                "content",
                CompletionTokenOutput::probs_vector_to_json(&self.probs_output, self.post_sampling_probs),
            )])
        } else {
            Json::Null
        };
        let choice = Json::object_from(vec![
            Item::new("text", self.content.clone()),
            Item::new("index", self.index as i64),
            Item::new("logprobs", logprobs),
            Item::new("finish_reason", self.finish_reason(false)),
        ]);
        self.base_oai("text_completion", choice)
    }

    /// `to_json_oaicompat_chat()`.
    pub fn to_json_oaicompat_chat(&self) -> Json {
        let msg = if self.message.role.is_empty() {
            ResponseMessage { role: "assistant".to_string(), content: self.content.clone(), ..ResponseMessage::default() }
        } else {
            self.message.clone()
        };
        let mut choice = Json::object_from(vec![
            Item::new("finish_reason", self.finish_reason(!msg.tool_calls.is_empty())),
            Item::new("index", self.index as i64),
            Item::new("message", msg.to_json()),
        ]);
        if !self.stream && !self.probs_output.is_empty() {
            let _ = choice.set(Item::new(
                "logprobs",
                Json::object_from(vec![Item::new(
                    "content",
                    CompletionTokenOutput::probs_vector_to_json(&self.probs_output, self.post_sampling_probs),
                )]),
            ));
        }
        self.base_oai("chat.completion", choice)
    }

    /// `to_json_oaicompat_chat_stream()`: an array of chunks.
    pub fn to_json_oaicompat_chat_stream(&self) -> Vec<Json> {
        let mut chunks = Vec::new();
        for diff in &self.msg_diffs {
            chunks.push(Json::object_from(vec![
                Item::new("choices", Json::array_from(vec![Json::object_from(vec![
                    Item::new("finish_reason", Json::Null),
                    Item::new("index", self.index as i64),
                    Item::new("delta", diff.clone()),
                ])])),
                Item::new("created", Self::created()),
                Item::new("id", self.oaicompat_cmpl_id.clone()),
                Item::new("model", self.oaicompat_model.clone()),
                Item::new("system_fingerprint", crate::api::LLAMA_VERSION),
                Item::new("object", "chat.completion.chunk"),
            ]));
        }
        let mut last = Json::object_from(vec![
            Item::new("choices", Json::array_from(vec![Json::object_from(vec![
                Item::new("finish_reason", self.finish_reason(!self.message.tool_calls.is_empty())),
                Item::new("index", self.index as i64),
                Item::new("delta", Json::object()),
            ])])),
            Item::new("created", Self::created()),
            Item::new("id", self.oaicompat_cmpl_id.clone()),
            Item::new("model", self.oaicompat_model.clone()),
            Item::new("system_fingerprint", crate::api::LLAMA_VERSION),
            Item::new("object", "chat.completion.chunk"),
        ]);
        if self.include_usage {
            chunks.push(last);
            last = Json::object_from(vec![
                Item::new("choices", Json::Array(Vec::new())),
                Item::new("created", Self::created()),
                Item::new("id", self.oaicompat_cmpl_id.clone()),
                Item::new("model", self.oaicompat_model.clone()),
                Item::new("system_fingerprint", crate::api::LLAMA_VERSION),
                Item::new("object", "chat.completion.chunk"),
                Item::new("usage", self.usage_json()),
            ]);
        }
        if self.stats.is_set() {
            let _ = last.set(Item::new("timings", self.stats.to_json()));
        }
        chunks.push(last);
        if self.verbose && !chunks.is_empty() {
            let verbose = self.to_json_non_oaicompat();
            if let Json::Object(ref mut entries) = chunks[0] {
                entries.push(("__verbose".to_string(), verbose));
            }
        }
        chunks
    }
}

/// `server_task_result` variants used by the HTTP layer.
#[derive(Clone, Debug)]
pub enum TaskResult {
    Error {
        id: i32,
        kind: ErrorType,
        message: String,
        n_prompt_tokens: i32,
        n_ctx: i32,
    },
    Metrics { id: i32, result: MetricsResult },
    Slots { id: i32, n_idle_slots: i32, slots_data: Json },
    SlotSaveLoad {
        id: i32,
        id_slot: i32,
        filename: String,
        is_save: bool,
        n_tokens: usize,
        n_bytes: usize,
        t_ms: f64,
    },
    SlotErase { id: i32, id_slot: i32, n_erased: usize },
    Control { id: i32, success: bool, message: String },
    GetLora { id: i32, loras: Vec<LoraResult> },
    ApplyLora { id: i32 },
    Embedding { id: i32, index: usize, embedding: Vec<Vec<f32>>, n_tokens: i32, res_type: TaskResponseType },
    Rerank { id: i32, index: usize, score: f32, n_tokens: i32 },
    Decision { id: i32, index: usize, scores: Vec<f32>, n_tokens: i32 },
    Completion(CompletionResult),
    CompletionPartial(PartialResult),
    Realtime { id: i32, text: String, audio: Vec<f32>, response_done: bool, cancelled: bool, n_decoded: i32 },
    Tts { id: i32, audio: Vec<u8>, sample_rate: i32, final_chunk: bool },
    Router { id: i32, data: Json },
}

#[derive(Clone, Debug, Default)]
pub struct LoraResult {
    pub path: String,
    pub scale: f32,
    pub task_name: String,
    pub prompt_prefix: String,
    pub alora_invocation_string: String,
    pub alora_invocation_tokens: Vec<Token>,
}

/// `server_task_result_cmpl_partial` (subset of the fields used for streaming).
#[derive(Clone, Debug)]
pub struct PartialResult {
    pub index: usize,
    pub content: String,
    pub n_decoded: i32,
    pub n_prompt_tokens: i32,
    pub n_prompt_tokens_cache: i32,
    pub stats: SlotStats,
    pub progress: ResultPromptProgress,
    pub res_type: TaskResponseType,
    pub oaicompat_model: String,
    pub oaicompat_cmpl_id: String,
    pub diffs: Vec<Json>,
    pub is_progress: bool,
    pub is_begin: bool,
    pub is_final: bool,
    pub finish_reason: String,
}

impl TaskResult {
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    pub fn is_stop(&self) -> bool {
        match self {
            Self::Completion(_) => true,
            Self::CompletionPartial(r) => r.is_final,
            Self::Tts { final_chunk, .. } => *final_chunk,
            _ => true,
        }
    }

    /// `server_task_result::to_json()`.
    pub fn to_json(&self) -> Json {
        match self {
            Self::Error { kind, message, n_prompt_tokens, n_ctx, .. } => {
                let mut res = format_error_response(message.clone(), *kind);
                if *kind == ErrorType::ExceedContextSize {
                    let _ = res.set(Item::new("n_prompt_tokens", *n_prompt_tokens));
                    let _ = res.set(Item::new("n_ctx", *n_ctx));
                }
                res
            }
            Self::Metrics { result: m, .. } => m.to_json(),
            Self::Slots { slots_data, .. } => slots_data.clone(),
            Self::SlotSaveLoad { id_slot, filename, is_save, n_tokens, n_bytes, t_ms, .. } => {
                if *is_save {
                    Json::object_from(vec![
                        Item::new("id_slot", *id_slot),
                        Item::new("filename", filename.clone()),
                        Item::new("n_saved", *n_tokens as u64),
                        Item::new("n_written", *n_bytes as u64),
                        Item::new("timings", Json::object_from(vec![Item::new("save_ms", *t_ms)])),
                    ])
                } else {
                    Json::object_from(vec![
                        Item::new("id_slot", *id_slot),
                        Item::new("filename", filename.clone()),
                        Item::new("n_restored", *n_tokens as u64),
                        Item::new("n_read", *n_bytes as u64),
                        Item::new("timings", Json::object_from(vec![Item::new("restore_ms", *t_ms)])),
                    ])
                }
            }
            Self::SlotErase { id_slot, n_erased, .. } => Json::object_from(vec![
                Item::new("id_slot", *id_slot),
                Item::new("n_erased", *n_erased as u64),
            ]),
            Self::Control { success, message, .. } => {
                let mut out = Json::object_from(vec![Item::new("success", *success)]);
                if !message.is_empty() {
                    let _ = out.set(Item::new("message", message.clone()));
                }
                out
            }
            Self::GetLora { loras, .. } => Json::array_from(
                loras
                    .iter()
                    .enumerate()
                    .map(|(i, l)| {
                        let mut out = Json::object_from(vec![
                            Item::new("id", i as i64),
                            Item::new("path", l.path.clone()),
                            Item::new("scale", l.scale),
                            Item::new("task_name", l.task_name.clone()),
                            Item::new("prompt_prefix", l.prompt_prefix.clone()),
                        ]);
                        if !l.alora_invocation_tokens.is_empty() {
                            let _ = out.set(Item::new("alora_invocation_string", l.alora_invocation_string.clone()));
                            let _ = out.set(Item::new("alora_invocation_tokens", l.alora_invocation_tokens.clone()));
                        }
                        out
                    })
                    .collect::<Vec<_>>(),
            ),
            Self::ApplyLora { .. } => Json::object_from(vec![Item::new("success", true)]),
            Self::Embedding { index, embedding, n_tokens, res_type, .. } => {
                if *res_type == TaskResponseType::OaiEmbd {
                    Json::object_from(vec![
                        Item::new("index", *index as i64),
                        Item::new("embedding", embedding.first().cloned().unwrap_or_default()),
                        Item::new("tokens_evaluated", *n_tokens),
                    ])
                } else {
                    Json::object_from(vec![
                        Item::new("index", *index as i64),
                        Item::new("embedding", embedding.clone()),
                    ])
                }
            }
            Self::Rerank { index, score, n_tokens, .. } => Json::object_from(vec![
                Item::new("index", *index as i64),
                Item::new("score", *score),
                Item::new("tokens_evaluated", *n_tokens),
            ]),
            Self::Decision { index, scores, n_tokens, .. } => Json::object_from(vec![
                Item::new("index", *index as i64),
                Item::new("scores", scores.clone()),
                Item::new("tokens_evaluated", *n_tokens),
            ]),
            Self::Completion(r) => match r.res_type {
                TaskResponseType::None => r.to_json_non_oaicompat(),
                TaskResponseType::OaiCmpl => r.to_json_oaicompat(),
                TaskResponseType::OaiChat => {
                    if r.stream {
                        Json::array_from(r.to_json_oaicompat_chat_stream())
                    } else {
                        r.to_json_oaicompat_chat()
                    }
                }
                TaskResponseType::OaiResp | TaskResponseType::OaiAsr | TaskResponseType::Anthropic => {
                    r.to_json_oaicompat_chat()
                }
                TaskResponseType::OaiEmbd => r.to_json_non_oaicompat(),
            },
            Self::CompletionPartial(r) => r.to_json(),
            Self::Realtime { text, audio, response_done, cancelled, n_decoded, .. } => Json::object_from(vec![
                Item::new("text", text.clone()),
                Item::new("n_samples", audio.len() as u64),
                Item::new("response_done", *response_done),
                Item::new("cancelled", *cancelled),
                Item::new("n_decoded", *n_decoded),
            ]),
            Self::Tts { audio, sample_rate, final_chunk, .. } => Json::object_from(vec![
                Item::new("sample_rate", *sample_rate),
                Item::new("n_bytes", audio.len() as u64),
                Item::new("final", *final_chunk),
            ]),
            Self::Router { data, .. } => data.clone(),
        }
    }

    /// Encode a result for SSE framing (arrays are sent as multiple events).
    pub fn to_sse(&self) -> String {
        safe_json_to_str(&self.to_json())
    }
}

impl PartialResult {
    /// `server_task_result_cmpl_partial::to_json()`.
    pub fn to_json(&self) -> Json {
        match self.res_type {
            TaskResponseType::OaiChat => {
                let delta = if self.is_progress {
                    Json::object_from(vec![Item::new("content", Json::object_from(vec![
                        Item::new("total", self.progress.total),
                        Item::new("cache", self.progress.cache),
                        Item::new("processed", self.progress.processed),
                        Item::new("time_ms", self.progress.time_ms),
                    ]))])
                } else {
                    let mut d = Json::object();
                    if !self.content.is_empty() {
                        let _ = d.set(Item::new("content", self.content.clone()));
                    }
                    for diff in &self.diffs {
                        if let Json::Object(entries) = diff {
                            for (key, value) in entries {
                                let _ = d.set(Item::new(key, value.clone()));
                            }
                        }
                    }
                    d
                };
                Json::object_from(vec![
                    Item::new("choices", Json::array_from(vec![Json::object_from(vec![
                        Item::new("finish_reason", if self.is_final { Json::String(self.finish_reason.clone()) } else { Json::Null }),
                        Item::new("index", self.index as i64),
                        Item::new("delta", delta),
                    ])])),
                    Item::new("created", CompletionResult::created()),
                    Item::new("id", self.oaicompat_cmpl_id.clone()),
                    Item::new("model", self.oaicompat_model.clone()),
                    Item::new("system_fingerprint", crate::api::LLAMA_VERSION),
                    Item::new("object", "chat.completion.chunk"),
                ])
            }
            _ => Json::object_from(vec![
                Item::new("index", self.index as i64),
                Item::new("content", self.content.clone()),
                Item::new("tokens_predicted", self.n_decoded),
                Item::new("tokens_evaluated", self.n_prompt_tokens),
                Item::new("tokens_cached", self.n_prompt_tokens_cache),
                Item::new("stop", self.is_final),
                Item::new("timings", self.stats.to_json()),
            ]),
        }
    }
}

/// `server_task_result_metrics`.
#[derive(Clone, Debug, Default)]
pub struct MetricsResult {
    pub n_processing_slots: i32,
    pub n_tasks_deferred: i32,
    pub metrics: Metrics,
}

struct MetricItem {
    name: &'static str,
    description: &'static str,
    value: f64,
}

impl MetricsResult {
    pub fn to_json(&self) -> Json {
        Json::object()
    }

    /// Prometheus text format, matching `server_task_result_metrics::to_metrics()`.
    pub fn to_metrics(&self) -> String {
        let m = &self.metrics;
        let counters = [
            MetricItem { name: "prompt_tokens_total", description: "Number of prompt tokens processed, excluding cached tokens", value: m.prompt.count as f64 },
            MetricItem { name: "prompt_tokens_cached_total", description: "Number of prompt tokens reused from the cache", value: m.n_prompt_cached as f64 },
            MetricItem { name: "prompt_seconds_total", description: "Total time spent processing prompts", value: m.prompt.time as f64 / 1e6 },
            MetricItem { name: "tokens_predicted_total", description: "Number of generation tokens processed", value: m.predict.count as f64 },
            MetricItem { name: "tokens_predicted_seconds_total", description: "Total time spent generating tokens", value: m.predict.time as f64 / 1e6 },
            MetricItem { name: "n_decode_total", description: "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding", value: m.n_decode as f64 },
            MetricItem { name: "n_tokens_max", description: "Largest observed sequence length (prompt + generation)", value: m.n_tokens_max as f64 },
            MetricItem { name: "spec_decode_num_draft_tokens_total", description: "Speculative: Total draft tokens generated", value: m.n_draft_tokens as f64 },
            MetricItem { name: "spec_decode_num_accepted_tokens_total", description: "Speculative: Total draft tokens accepted by the target model", value: m.n_draft_accepted as f64 },
            MetricItem { name: "spec_decode_num_drafts_total", description: "Speculative: Total speculative decoding verification steps", value: m.n_draft_verif_steps as f64 },
        ];
        let gauges = [
            MetricItem { name: "prompt_tokens_seconds", description: "Average prompt throughput in tokens/s", value: m.prompt_bucket.n_per_second() },
            MetricItem { name: "predicted_tokens_seconds", description: "Average generation throughput in tokens/s", value: m.predict_bucket.n_per_second() },
            MetricItem { name: "requests_processing", description: "Number of requests processing", value: self.n_processing_slots as f64 },
            MetricItem { name: "requests_deferred", description: "Number of requests deferred", value: self.n_tasks_deferred as f64 },
            MetricItem { name: "n_busy_slots_per_decode", description: "Average number of busy slots per llama_decode() call", value: m.n_busy_slots as f64 / (m.n_decode as f64).max(1.0) },
        ];
        let mut out = String::new();
        let mut add_items = |kind: &str, items: &[MetricItem]| {
            for item in items {
                let _ = writeln!(out, "# HELP llamacpp:{} {}\n# TYPE llamacpp:{} {}\nllamacpp:{} {}",
                    item.name, item.description, item.name, kind, item.name, item.value);
            }
        };
        add_items("counter", &counters);
        add_items("gauge", &gauges);
        if !m.n_accepted_per_pos.is_empty() {
            let _ = writeln!(out, "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft position\n# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter");
            for (i, value) in m.n_accepted_per_pos.iter().enumerate() {
                let _ = writeln!(out, "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{{position=\"{}\"}} {}", i, value);
            }
        }
        out
    }
}

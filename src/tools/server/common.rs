//! Port of `tools/server/server-common.{h,cpp}`.
//!
//! Shared error codes, JSON helpers, SSE framing, role/state enums, timing
//! statistics and the small token container used by the HTTP handlers.

use crate::common::Params;
use crate::core::{Error, Result, Token, LLAMA_TOKEN_NULL};
use crate::json::{FromJson, Item, Json, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// `enum error_type` from `server-common.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorType {
    InvalidRequest,
    Authentication,
    Server,
    NotFound,
    Permission,
    Unavailable,
    NotSupported,
    ExceedContextSize,
}

impl ErrorType {
    /// Error type string used by the OpenAI-compatible `error.type` field.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request_error",
            Self::Authentication => "authentication_error",
            Self::Server => "server_error",
            Self::NotFound => "not_found_error",
            Self::Permission => "permission_error",
            Self::Unavailable => "unavailable_error",
            Self::NotSupported => "not_supported_error",
            Self::ExceedContextSize => "exceed_context_size_error",
        }
    }

    /// HTTP status paired with the error type.
    pub fn status(self) -> i32 {
        match self {
            Self::InvalidRequest | Self::ExceedContextSize => 400,
            Self::Authentication => 401,
            Self::NotFound => 404,
            Self::Server => 500,
            Self::Permission => 403,
            Self::NotSupported => 501,
            Self::Unavailable => 503,
        }
    }
}

impl Default for ErrorType {
    fn default() -> Self {
        Self::Server
    }
}

/// `format_error_response()`: `{"code", "message", "type"}`.
pub fn format_error_response(message: impl Into<String>, kind: ErrorType) -> Json {
    Json::object_from(vec![
        Item::new("code", kind.status() as i64),
        Item::new("message", message.into()),
        Item::new("type", kind.as_str()),
    ])
}

/// Build an error response body: `{"error": format_error_response(...)}`.
pub fn error_body(message: impl Into<String>, kind: ErrorType) -> Json {
    Json::object_from(vec![Item::new("error", format_error_response(message, kind))])
}

/// `json_value()`: wrong types fall back to the default instead of failing.
pub fn json_value<T: FromJson>(body: &Json, key: &str, default: T) -> T {
    if !body.contains(key) {
        return default;
    }
    match body.at(key) {
        Ok(v) if !v.is_null() => v.get::<T>().unwrap_or(default),
        _ => default,
    }
}

/// `safe_json_to_str()`.
pub fn safe_json_to_str(data: &Json) -> String {
    data.dump_safe(-1)
}

/// `json_is_array_of_numbers()`.
pub fn json_is_array_of_numbers(data: &Json) -> bool {
    let Json::Array(items) = data else {
        return false;
    };
    !items.is_empty() && items.iter().all(|v| v.is_number())
}

/// `json_is_array_and_contains_numbers()`.
pub fn json_is_array_and_contains_numbers(data: &Json) -> bool {
    let Json::Array(items) = data else {
        return false;
    };
    items.iter().any(|v| v.is_number_integer())
}

/// `json_is_array_of_mixed_numbers_strings()`.
pub fn json_is_array_of_mixed_numbers_strings(data: &Json) -> bool {
    let Json::Array(items) = data else {
        return false;
    };
    let mut number = false;
    let mut string = false;
    for item in items {
        if item.is_number_integer() {
            number = true;
        } else if item.is_string() {
            string = true;
        } else {
            return false;
        }
    }
    number && string
}

/// `json_get_nested_values()`: read each path in `a/b/c` form.
pub fn json_get_nested_values(paths: &[String], js: &Json) -> Json {
    let mut out = Json::object();
    for path in paths {
        let mut cur = js;
        for key in path.split('/') {
            match cur.at(key) {
                Ok(next) => cur = next,
                Err(_) => {
                    cur = &Json::Null;
                    break;
                }
            }
        }
        let _ = out.set(Item::new(path, cur.clone()));
    }
    out
}

/// `format_oai_sse()`: one `data:` event per array item.
pub fn format_oai_sse(data: &Json) -> String {
    let mut out = String::new();
    let items: Vec<&Json> = match data {
        Json::Array(values) => values.iter().collect(),
        other => vec![other],
    };
    for item in items {
        let _ = writeln!(out, "data: {}\n", safe_json_to_str(item));
    }
    out
}

/// `format_oai_resp_sse()`: one `event:`/`data:` pair per array item.
pub fn format_oai_resp_sse(data: &Json) -> String {
    let mut out = String::new();
    let items: Vec<&Json> = match data {
        Json::Array(values) => values.iter().collect(),
        other => vec![other],
    };
    for item in items {
        let event = item.value_str("event", "").unwrap_or_default();
        let body = item.at("data").cloned().unwrap_or_else(|_| Json::Null);
        let _ = writeln!(out, "event: {}\ndata: {}\n", event, safe_json_to_str(&body));
    }
    out
}

/// `format_anthropic_sse()`: typed events, or a plain `data:` event.
pub fn format_anthropic_sse(data: &Json) -> String {
    let mut out = String::new();
    let items: Vec<&Json> = match data {
        Json::Array(values) => values.iter().collect(),
        other => vec![other],
    };
    for item in items {
        if item.contains("event") && item.contains("data") {
            let event = item.value_str("event", "").unwrap_or_default();
            let body = item.at("data").cloned().unwrap_or_else(|_| Json::Null);
            let _ = writeln!(out, "event: {}\ndata: {}\n", event, safe_json_to_str(&body));
        } else {
            let _ = writeln!(out, "data: {}\n", safe_json_to_str(item));
        }
    }
    out
}

/// `random_string()`: 32 alphanumeric characters, like the C++ helper.
pub fn random_string() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    const ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut state = now
        ^ COUNTER.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (std::process::id() as u64).wrapping_mul(0x5851_f42d_4c95_7f2d);
    let mut out = String::with_capacity(32);
    for _ in 0..32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push(ALPHABET[(state as usize) % ALPHABET.len()] as char);
    }
    out
}

/// `gen_chatcmplid()`.
pub fn gen_chatcmplid() -> String {
    format!("chatcmpl-{}", random_string())
}

/// `gen_tool_call_id()`.
pub fn gen_tool_call_id() -> String {
    random_string()
}

/// `get_media_marker()`: fixed unless `LLAMA_MEDIA_MARKER` pins it.
pub fn get_media_marker() -> String {
    std::env::var("LLAMA_MEDIA_MARKER").unwrap_or_else(|_| "<__media__>".to_string())
}

/// Absolute timestamp in microseconds, matching `ggml_time_us()` usage.
pub fn time_us() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// `server_role_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServerRoleType {
    Router,
    Instance,
    Download,
    Measure,
}

impl ServerRoleType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Router => "router",
            Self::Instance => "instance",
            Self::Download => "download",
            Self::Measure => "measure",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "download" => Self::Download,
            "measure" => Self::Measure,
            _ => Self::Instance,
        }
    }
}

/// `server_state`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServerState {
    Downloading,
    Measuring,
    Loading,
    Ready,
    Sleeping,
}

impl ServerState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Downloading => "downloading",
            Self::Measuring => "measuring",
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Sleeping => "sleeping",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "downloading" => Self::Downloading,
            "measuring" => Self::Measuring,
            "loading" => Self::Loading,
            "sleeping" => Self::Sleeping,
            _ => Self::Ready,
        }
    }
}

/// `server_role`: the role of this process and its router-child side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServerRole {
    Router,
    Instance,
    Download,
    Measure,
    ManagedInstance,
}

impl ServerRole {
    /// `SERVER_ENV_ROUTER_PORT` marks a process spawned by a router.
    pub fn is_managed_by_router(self) -> bool {
        self == Self::ManagedInstance
    }

    pub fn get_role(self) -> ServerRoleType {
        match self {
            Self::Router => ServerRoleType::Router,
            Self::Instance | Self::ManagedInstance => ServerRoleType::Instance,
            Self::Download => ServerRoleType::Download,
            Self::Measure => ServerRoleType::Measure,
        }
    }
}

/// Environment flag set by the router in each child's environment.
pub const SERVER_ENV_ROUTER_PORT: &str = "LLAMA_SERVER_ROUTER_PORT";
/// Environment flag carrying the child role.
pub const SERVER_ENV_CHILD_ROLE: &str = "LLAMA_SERVER_CHILD_ROLE";
/// Command line sent from the router to a child.
pub const CMD_ROUTER_TO_CHILD_EXIT: &str = "cmd_router_to_child:exit";
/// Prefix of state notifications sent from a child to the router.
pub const CMD_CHILD_TO_ROUTER_STATE: &str = "cmd_child_to_router:state:";
/// Children always bind locally, even when the router listens on 0.0.0.0.
pub const CHILD_ADDR: &str = "127.0.0.1";

/// `server_slot_stats`: per-slot timing counters.
#[derive(Clone, Debug, Default)]
pub struct SlotStats {
    pub n_prompt_cached: u64,
    pub n_prompt_processed: u64,
    pub n_gen: u64,
    pub n_draft_tokens: u64,
    pub n_draft_accepted: u64,
    pub n_draft_verif_steps: u64,
    pub t_start: i64,
    pub t_prompt_last: i64,
    pub t_gen_last: i64,
}

impl SlotStats {
    pub fn update_prompt_start(&mut self) {
        self.t_start = time_us();
    }

    pub fn set_prompt_last(&mut self, t_us: i64) {
        self.t_prompt_last = t_us;
    }

    pub fn update_prompt_last(&mut self) {
        let t_us = time_us();
        self.set_prompt_last(t_us);
    }

    pub fn update_gen_last(&mut self) {
        self.t_gen_last = time_us();
    }

    pub fn t_elapsed_us(&self) -> i64 {
        time_us() - self.t_start
    }

    pub fn t_prompt_ms(&self) -> f64 {
        if self.t_prompt_last == 0 {
            0.0
        } else {
            (self.t_prompt_last - self.t_start) as f64 / 1000.0
        }
    }

    pub fn t_gen_us(&self) -> i64 {
        if self.t_gen_last == 0 {
            0
        } else {
            (self.t_gen_last - self.t_prompt_last).max(1)
        }
    }

    pub fn t_gen_ms(&self) -> f64 {
        self.t_gen_us() as f64 / 1000.0
    }

    pub fn n_gen_steps(&self) -> u64 {
        self.n_gen.saturating_sub(1)
    }

    pub fn t_prompt_per_token_ms(&self) -> f64 {
        if self.n_prompt_processed > 0 {
            self.t_prompt_ms() / self.n_prompt_processed as f64
        } else {
            0.0
        }
    }

    pub fn t_gen_per_token_ms(&self) -> f64 {
        if self.n_gen_steps() > 0 {
            self.t_gen_ms() / self.n_gen_steps() as f64
        } else {
            0.0
        }
    }

    pub fn n_prompt_tps(&self) -> f64 {
        let ms = self.t_prompt_ms();
        if ms > 0.0 {
            1e3 / ms * self.n_prompt_processed as f64
        } else {
            0.0
        }
    }

    pub fn n_gen_tps(&self) -> f64 {
        let ms = self.t_gen_ms();
        if ms > 0.0 {
            1e3 / ms * self.n_gen_steps() as f64
        } else {
            0.0
        }
    }

    pub fn is_set(&self) -> bool {
        self.t_start > 0
    }

    /// `server_slot_stats::to_json()`.
    pub fn to_json(&self) -> Json {
        let mut base = Json::object_from(vec![
            Item::new("cache_n", self.n_prompt_cached),
            Item::new("prompt_n", self.n_prompt_processed),
            Item::new("prompt_ms", self.t_prompt_ms()),
            Item::new("prompt_per_token_ms", self.t_prompt_per_token_ms()),
            Item::new("prompt_per_second", self.n_prompt_tps()),
            Item::new("predicted_n", self.n_gen),
            Item::new("predicted_ms", self.t_gen_ms()),
            Item::new("predicted_per_token_ms", self.t_gen_per_token_ms()),
            Item::new("predicted_per_second", self.n_gen_tps()),
        ]);
        if self.n_draft_tokens > 0 {
            let _ = base.set(Item::new("draft_n", self.n_draft_tokens));
            let _ = base.set(Item::new("draft_n_accepted", self.n_draft_accepted));
        }
        base
    }
}

/// One cumulative or per-scrape bucket of `server_metrics`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetricsBucket {
    pub count: u64,
    pub steps: u64,
    pub time: u64,
}

impl MetricsBucket {
    pub fn n_per_second(&self) -> f64 {
        if self.time > 0 {
            self.steps as f64 / self.time as f64 * 1e6
        } else {
            0.0
        }
    }

    pub fn add(&mut self, n: u64, steps: u64, time_us: u64) {
        self.count += n;
        self.steps += steps;
        self.time += time_us;
    }
}

/// `server_metrics`: cumulative counters plus the two scrape buckets.
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub t_start: i64,
    pub prompt_bucket: MetricsBucket,
    pub predict_bucket: MetricsBucket,
    pub prompt: MetricsBucket,
    pub predict: MetricsBucket,
    pub n_prompt_cached: u64,
    pub n_tokens_max: u64,
    pub n_decode: u64,
    pub n_busy_slots: u64,
    pub n_draft_tokens: u64,
    pub n_draft_accepted: u64,
    pub n_draft_verif_steps: u64,
    pub n_accepted_per_pos: Vec<u64>,
}

impl Metrics {
    pub fn init(&mut self) {
        self.t_start = time_us();
    }

    pub fn reset_bucket(&mut self) {
        self.prompt_bucket = MetricsBucket::default();
        self.predict_bucket = MetricsBucket::default();
    }

    pub fn add_prompt(&mut self, n_tokens: u64, t_us: u64) {
        self.prompt.add(n_tokens, n_tokens, t_us);
        self.prompt_bucket.add(n_tokens, n_tokens, t_us);
    }

    pub fn add_prompt_cached(&mut self, n_tokens: u64) {
        self.n_prompt_cached += n_tokens;
    }
}

/// `server_chat_params`: global chat formatting and media settings.
#[derive(Clone, Debug)]
pub struct ServerChatParams {
    pub use_jinja: bool,
    pub prefill_assistant: bool,
    pub reasoning_format: crate::common::ReasoningFormat,
    pub chat_template_kwargs: BTreeMap<String, String>,
    pub allow_image: bool,
    pub allow_audio: bool,
    pub allow_video: bool,
    pub enable_thinking: bool,
    pub reasoning_budget: i32,
    pub reasoning_budget_message: String,
    pub media_path: String,
    pub media_allow_private_urls: bool,
    pub force_pure_content: bool,
}

impl ServerChatParams {
    /// Initialize from the parsed CLI parameters, like `server_routes`.
    pub fn from_params(params: &Params) -> Self {
        Self {
            use_jinja: params.use_jinja,
            prefill_assistant: params.prefill_assistant,
            reasoning_format: params.reasoning_format,
            chat_template_kwargs: params.default_template_kwargs.clone(),
            allow_image: false,
            allow_audio: false,
            allow_video: false,
            enable_thinking: params.enable_reasoning != 0,
            reasoning_budget: -1,
            reasoning_budget_message: String::new(),
            media_path: params.media_path.clone(),
            media_allow_private_urls: params.media_allow_private_urls,
            force_pure_content: params.force_pure_content_parser,
        }
    }
}

/// `server_tokens`: text tokens plus a marker for each media chunk.
#[derive(Clone, Debug, Default)]
pub struct ServerTokens {
    pub has_mtmd: bool,
    tokens: Vec<Token>,
    /// Start index of each media chunk, in token order.
    media: Vec<usize>,
}

impl ServerTokens {
    pub fn new(tokens: Vec<Token>, has_mtmd: bool) -> Self {
        Self {
            has_mtmd,
            tokens,
            media: Vec::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.tokens.len()
    }

    pub fn empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.media.clear();
    }

    pub fn push_back(&mut self, token: Token) {
        self.tokens.push(token);
    }

    pub fn push_back_media(&mut self, n_tokens: usize) {
        self.media.push(self.tokens.len());
        for _ in 0..n_tokens {
            self.tokens.push(LLAMA_TOKEN_NULL);
        }
    }

    pub fn at(&self, index: usize) -> Token {
        self.tokens.get(index).copied().unwrap_or(LLAMA_TOKEN_NULL)
    }

    pub fn set_token(&mut self, pos: usize, id: Token) {
        if let Some(slot) = self.tokens.get_mut(pos) {
            *slot = id;
        }
    }

    pub fn get_tokens(&self) -> &[Token] {
        &self.tokens
    }

    pub fn get_text_tokens(&self) -> Vec<Token> {
        self.tokens
            .iter()
            .copied()
            .filter(|t| *t != LLAMA_TOKEN_NULL)
            .collect()
    }

    pub fn keep_first(&mut self, n: usize) {
        self.tokens.truncate(n);
        self.media.retain(|idx| *idx < n);
    }

    pub fn insert(&mut self, other: &[Token]) {
        let at = self.tokens.len();
        self.tokens.extend_from_slice(other);
        self.media.iter_mut().for_each(|idx| {
            if *idx >= at {
                *idx += other.len();
            }
        });
    }

    pub fn find_next_media_chunk(&self, idx: usize) -> Option<(usize, usize)> {
        self.media.iter().copied().find(|start| *start >= idx).map(|start| (start, start))
    }

    /// Compact binary form: has_mtmd, number of media, media starts, tokens.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(u8::from(self.has_mtmd));
        out.extend_from_slice(&(self.media.len() as u32).to_le_bytes());
        for idx in &self.media {
            out.extend_from_slice(&(*idx as u32).to_le_bytes());
        }
        out.extend_from_slice(&(self.tokens.len() as u32).to_le_bytes());
        for token in &self.tokens {
            out.extend_from_slice(&token.to_le_bytes());
        }
        out
    }

    pub fn deserialize(packed: &[u8]) -> Result<Self> {
        let mut pos = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            let end = pos.checked_add(n).ok_or_else(|| Error::new("token data is too large"))?;
            let slice = packed.get(pos..end).ok_or_else(|| Error::new("truncated token data"))?;
            pos = end;
            Ok(slice)
        };
        let has_mtmd = take(1)?[0] != 0;
        let n_media = u32::from_le_bytes(take(4)?.try_into().map_err(|_| Error::new("bad media count"))?) as usize;
        let mut media = Vec::with_capacity(n_media);
        for _ in 0..n_media {
            media.push(u32::from_le_bytes(take(4)?.try_into().map_err(|_| Error::new("bad media offset"))?) as usize);
        }
        let n_tokens = u32::from_le_bytes(take(4)?.try_into().map_err(|_| Error::new("bad token count"))?) as usize;
        let mut tokens = Vec::with_capacity(n_tokens);
        for _ in 0..n_tokens {
            tokens.push(i32::from_le_bytes(take(4)?.try_into().map_err(|_| Error::new("bad token"))?));
        }
        Ok(Self { has_mtmd, tokens, media })
    }

    /// Longest common prefix with another token sequence.
    pub fn get_common_prefix(&self, other: &Self) -> usize {
        self.tokens
            .iter()
            .zip(other.tokens.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }
}

/// `parse_lora_request()`: array of `{"id": int, "scale": float}`.
pub fn parse_lora_request(data: &Json) -> BTreeMap<i32, f32> {
    let mut out = BTreeMap::new();
    let Json::Array(items) = data else {
        return out;
    };
    for item in items {
        let id = json_value(item, "id", -1i32);
        let scale = json_value(item, "scale", 0.0f32);
        if id >= 0 {
            out.insert(id, scale);
        }
    }
    out
}

/// Adapter LoRA identity, from `common_adapter_lora_info`.
#[derive(Clone, Debug, PartialEq)]
pub struct AdapterLoraInfo {
    pub path: String,
    pub scale: f32,
    pub task_name: String,
    pub prompt_prefix: String,
    pub alora_invocation_string: String,
    pub alora_invocation_tokens: Vec<Token>,
}

/// `lora_all_alora()`: true only when every active adapter is an ALoRA.
pub fn lora_all_alora(loras: &[AdapterLoraInfo]) -> bool {
    let mut found = false;
    for lora in loras {
        if lora.scale != 0.0 {
            if lora.alora_invocation_tokens.is_empty() {
                return false;
            }
            found = true;
        }
    }
    found
}

/// `are_lora_equal()`.
pub fn are_lora_equal(l1: &[AdapterLoraInfo], l2: &[AdapterLoraInfo]) -> bool {
    l1.len() == l2.len() && l1.iter().zip(l2).all(|(a, b)| {
        a.path == b.path
            && (a.scale - b.scale).abs() < f32::EPSILON
            && a.task_name == b.task_name
            && a.prompt_prefix == b.prompt_prefix
    })
}

/// `lora_should_clear_cache()`.
pub fn lora_should_clear_cache(current: &[AdapterLoraInfo], next: &[AdapterLoraInfo]) -> bool {
    if are_lora_equal(current, next) {
        return false;
    }
    !(lora_all_alora(current) && lora_all_alora(next))
}

/// `lora_get_enabled_ids()`.
pub fn lora_get_enabled_ids(loras: &[AdapterLoraInfo]) -> Vec<usize> {
    loras
        .iter()
        .enumerate()
        .filter(|(_, l)| l.scale != 0.0)
        .map(|(i, _)| i)
        .collect()
}

/// `server_model_output_modalities()`.
pub fn server_model_output_modalities(decision_is_classifier: bool, has_cap_chat: bool, has_cap_tts: bool) -> Vec<String> {
    let mut out = if decision_is_classifier {
        vec!["decisions".to_string()]
    } else if has_cap_chat || !has_cap_tts {
        vec!["text".to_string()]
    } else {
        Vec::new()
    };
    if has_cap_tts {
        out.push("audio".to_string());
    }
    out
}

/// `server_model_architecture_json()`.
pub fn server_model_architecture_json(
    inp_image: bool,
    inp_audio: bool,
    inp_video: bool,
    output_modalities: &[String],
) -> Json {
    let mut inputs = vec!["text".to_string()];
    if inp_image {
        inputs.push("image".to_string());
    }
    if inp_audio {
        inputs.push("audio".to_string());
    }
    if inp_video {
        inputs.push("video".to_string());
    }
    Json::object_from(vec![
        Item::new("input_modalities", inputs),
        Item::new("output_modalities", output_modalities.to_vec()),
    ])
}

/// Simple pipe used to stream responses between threads, from `server_pipe`.
pub struct ServerPipe<T> {
    queue: Mutex<VecDeque<T>>,
    cv: Condvar,
    writer_closed: Mutex<bool>,
    reader_closed: Mutex<bool>,
    max_size: usize,
    poll_interval: Duration,
}

impl<T> ServerPipe<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            writer_closed: Mutex::new(false),
            reader_closed: Mutex::new(false),
            max_size,
            poll_interval: Duration::from_millis(500),
        }
    }

    pub fn close_write(&self) {
        let mut closed = self.writer_closed.lock().unwrap_or_else(|e| e.into_inner());
        *closed = true;
        self.cv.notify_all();
    }

    pub fn close_read(&self) {
        let mut closed = self.reader_closed.lock().unwrap_or_else(|e| e.into_inner());
        *closed = true;
        self.cv.notify_all();
    }

    fn is_reader_closed(&self) -> bool {
        *self.reader_closed.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn is_writer_closed(&self) -> bool {
        *self.writer_closed.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Blocking read. `should_stop()` cancels the read and breaks the pipe.
    pub fn read(&self, should_stop: &dyn Fn() -> bool, close_on_stop: bool) -> Option<T> {
        loop {
            let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(value) = queue.pop_front() {
                return Some(value);
            }
            drop(queue);
            if self.is_writer_closed() {
                return None;
            }
            if should_stop() {
                if close_on_stop {
                    self.close_read();
                }
                return None;
            }
            let queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            let _ = self.cv.wait_timeout(queue, self.poll_interval).unwrap_or_else(|e| e.into_inner());
        }
    }

    pub fn write(&self, data: T) -> bool {
        if self.is_reader_closed() {
            return false;
        }
        let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        if self.max_size > 0 {
            while queue.len() >= self.max_size {
                queue.pop_front();
            }
        }
        queue.push_back(data);
        self.cv.notify_one();
        true
    }
}

/// `get_server_exec_path()`: path of the running server binary.
pub fn get_server_exec_path() -> Result<std::path::PathBuf> {
    std::env::current_exe().map_err(|e| Error::new(format!("failed to resolve current exe: {}", e)))
}

/// `server_addr_is_public()`: no loopback, link-local, private or unspecified address.
pub fn server_addr_is_public(addr: &str) -> bool {
    if let Ok(ip) = addr.parse::<std::net::IpAddr>() {
        match ip {
            std::net::IpAddr::V4(v4) => {
                !(v4.is_loopback()
                    || v4.is_private()
                    || v4.is_link_local()
                    || v4.is_unspecified()
                    || v4.is_broadcast()
                    || v4.is_multicast()
                    || v4.is_documentation())
            }
            std::net::IpAddr::V6(v6) => {
                !(v6.is_loopback() || v6.is_unspecified() || v6.is_multicast())
            }
        }
    } else {
        false
    }
}

/// Validate UTF-8 up to the last complete character, as `validate_utf8()`.
pub fn validate_utf8(text: &str) -> usize {
    text.len()
}

/// `is_valid_utf8()` for a byte buffer produced by a token.
pub fn is_valid_utf8(bytes: &[u8]) -> bool {
    std::str::from_utf8(bytes).is_ok()
}

/// Parse a CSV row, trimming each entry like llama.cpp's helper.
pub fn parse_csv_row(s: &str) -> Vec<String> {
    s.split(',')
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .collect()
}

/// Convert a set to a JSON array of strings.
pub fn json_string_set(values: &BTreeSet<String>) -> Json {
    Json::array_from(values.iter().cloned().collect::<Vec<_>>())
}

/// Convert a `Value` into a `Json` tree (single-use by design in the port).
pub fn value_to_json(value: Value) -> Json {
    Json::from(value)
}

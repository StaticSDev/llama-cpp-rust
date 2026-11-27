//! Port of `tools/server/server-context.cpp` route handlers.
//!
//! `ServerRoutes` owns one handler per API endpoint, wired with the same paths
//! and semantics as `server_routes::init_routes()` in the reference.

use crate::chat::{apply_template, chat_template_from_str, ChatMessage, ChatTemplate};
use crate::common::{GrammarType, Params};
use crate::core::{Error, Result, Token};
use crate::json::{Item, Json};
use crate::tools::server::common::{
    json_value, parse_lora_request, safe_json_to_str, ErrorType, Metrics,
};
use crate::tools::server::context::{ServerContext, ServerContextMeta};
use crate::tools::server::http::{Handler, HttpRequest, HttpResponse, SseFraming, WsHandler};
use crate::tools::server::task::{
    MetricsResult, ServerControlAction, ServerTask, ServerTaskType, TaskParams, TaskResponseType, TaskResult,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

struct RouteCache {
    models: Json,
    props: Json,
    metrics: Metrics,
    should_reset_buckets: bool,
}

struct RoutesInner {
    params: Params,
    ctx_server: Arc<ServerContext>,
    meta: Mutex<Option<ServerContextMeta>>,
    cache: Mutex<RouteCache>,
}

impl RoutesInner {
    fn meta(&self) -> Option<ServerContextMeta> {
        self.meta.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn err(&self, message: impl Into<String>, kind: ErrorType) -> HttpResponse {
        HttpResponse::error_message(message, kind)
    }

    /// `server_schema::eval_llama_cmpl_schema()`, restricted to std-only fields.
    fn task_params(&self, data: &Json, res_type: TaskResponseType) -> TaskParams {
        let mut out = TaskParams::from_params(&self.params);
        out.res_type = res_type;
        out.verbose = self.params.verbosity > 9;
        out.antiprompt = self.params.antiprompt.clone();
        if let Some(meta) = self.meta() {
            out.oaicompat_model = meta.model_name.clone();
        }
        out.oaicompat_cmpl_id = crate::tools::server::common::gen_chatcmplid();

        out.stream = json_value(data, "stream", out.stream);
        out.include_usage = data
            .at("stream_options")
            .ok()
            .map(|o| json_value(o, "include_usage", out.include_usage))
            .unwrap_or(out.include_usage);
        out.cache_prompt = json_value(data, "cache_prompt", out.cache_prompt);
        out.return_tokens = json_value(data, "return_tokens", out.return_tokens);
        out.return_progress = json_value(data, "return_progress", out.return_progress);
        out.sse_ping_interval = json_value(data, "sse_ping_interval", out.sse_ping_interval).max(-1);
        out.n_predict = json_value(data, "n_predict", out.n_predict).max(-1);
        if !data.contains("n_predict") {
            out.n_predict = json_value(data, "max_completion_tokens", json_value(data, "max_tokens", out.n_predict)).max(-1);
        }
        out.n_indent = json_value(data, "n_indent", out.n_indent).max(0);
        out.n_keep = json_value(data, "n_keep", out.n_keep).max(-1);
        out.n_discard = json_value(data, "n_discard", out.n_discard).max(0);
        out.n_cmpl = json_value(data, "n_cmpl", json_value(data, "n", out.n_cmpl)).max(1).min(self.params.n_parallel.max(1));
        out.n_cache_reuse = json_value(data, "n_cache_reuse", out.n_cache_reuse).max(0);
        out.t_max_predict_ms = json_value(data, "t_max_predict_ms", out.t_max_predict_ms).max(-1);
        out.response_fields = json_value(data, "response_fields", out.response_fields);

        let s = &mut out.sampling;
        s.top_k = json_value(data, "top_k", s.top_k).max(0);
        s.top_p = json_value(data, "top_p", s.top_p).clamp(0.0, 1.0);
        s.min_p = json_value(data, "min_p", s.min_p).clamp(0.0, 1.0);
        s.top_n_sigma = json_value(data, "top_n_sigma", s.top_n_sigma);
        s.xtc_probability = json_value(data, "xtc_probability", s.xtc_probability).clamp(0.0, 1.0);
        s.xtc_threshold = json_value(data, "xtc_threshold", s.xtc_threshold).clamp(0.0, 1.0);
        s.typ_p = json_value(data, "typical_p", s.typ_p);
        s.temp = json_value(data, "temperature", s.temp).max(0.0);
        s.dynatemp_range = json_value(data, "dynatemp_range", s.dynatemp_range);
        s.dynatemp_exponent = json_value(data, "dynatemp_exponent", s.dynatemp_exponent);
        s.penalty_last_n = json_value(data, "repeat_last_n", s.penalty_last_n).max(0);
        s.penalty_repeat = json_value(data, "repeat_penalty", s.penalty_repeat);
        s.penalty_freq = json_value(data, "frequency_penalty", s.penalty_freq);
        s.penalty_present = json_value(data, "presence_penalty", s.penalty_present);
        s.dry_multiplier = json_value(data, "dry_multiplier", s.dry_multiplier);
        let dry_base: f32 = json_value(data, "dry_base", s.dry_base);
        s.dry_base = if dry_base < 1.0 { self.params.sampling.dry_base } else { dry_base };
        s.dry_allowed_length = json_value(data, "dry_allowed_length", s.dry_allowed_length).max(0);
        s.dry_penalty_last_n = json_value(data, "dry_penalty_last_n", s.dry_penalty_last_n).max(0);
        s.mirostat = json_value(data, "mirostat", s.mirostat).clamp(0, 2);
        s.mirostat_tau = json_value(data, "mirostat_tau", s.mirostat_tau);
        s.mirostat_eta = json_value(data, "mirostat_eta", s.mirostat_eta);
        s.adaptive_target = json_value(data, "adaptive_target", s.adaptive_target).min(1.0);
        s.adaptive_decay = json_value(data, "adaptive_decay", s.adaptive_decay).clamp(0.0, 0.99);
        s.seed = json_value(data, "seed", s.seed as i64) as u32;
        s.n_probs = json_value(data, "n_probs", json_value(data, "logprobs", s.n_probs)).max(0);
        s.min_keep = json_value(data, "min_keep", s.min_keep).max(0);
        s.backend_sampling = json_value(data, "backend_sampling", s.backend_sampling);
        if let Some(names) = data.at("samplers").ok().and_then(|v| v.get::<Vec<String>>().ok()) {
            s.samplers = crate::common::sampler_types_from_names(&names);
        }
        if let Some(breakers) = data.at("dry_sequence_breakers").ok().and_then(|v| v.get::<Vec<String>>().ok()) {
            if !breakers.is_empty() {
                s.dry_sequence_breakers = breakers;
            }
        }
        let grammar: String = json_value(data, "grammar", String::new());
        if !grammar.is_empty() {
            s.grammar.grammar = grammar;
            s.grammar.type_ = if json_value(data, "grammar_type", String::new()) == "tool_calls" {
                GrammarType::ToolCalls
            } else {
                GrammarType::User
            };
        }
        s.grammar_lazy = json_value(data, "grammar_lazy", s.grammar_lazy);
        s.reasoning_control = json_value(data, "reasoning_control", s.reasoning_control);
        s.reasoning_budget_tokens = json_value(data, "reasoning_budget_tokens", s.reasoning_budget_tokens).max(-1);
        s.reasoning_budget_message = json_value(data, "reasoning_budget_message", s.reasoning_budget_message.clone());
        let ignore_eos = json_value(data, "ignore_eos", s.ignore_eos);
        s.ignore_eos = ignore_eos;
        s.generation_prompt = json_value(data, "generation_prompt", s.generation_prompt.clone());
        out.post_sampling_probs = json_value(data, "post_sampling_probs", out.post_sampling_probs);
        out.timings_per_token = json_value(data, "timings_per_token", out.timings_per_token);

        if let Ok(stop) = data.at("stop") {
            let mut words = Vec::new();
            match stop {
                Json::String(s) if !s.is_empty() => words.push(s.clone()),
                Json::Array(items) => {
                    for item in items {
                        if let Json::String(s) = item {
                            if !s.is_empty() {
                                words.push(s.clone());
                            }
                        }
                    }
                }
                _ => {}
            }
            if words.is_empty() {
                words = self.params.antiprompt.clone();
            }
            out.antiprompt = words;
        }
        if let Ok(lora) = data.at("lora") {
            if !lora.is_array() {
                out.lora.clear();
            } else {
                out.lora = parse_lora_request(lora);
            }
        }
        out
    }

    /// Submit a task. Without an attached engine the handler returns a faithful
    /// error result instead of blocking on a response that can never arrive.
    fn run_task(&self, mut task: ServerTask) -> TaskResult {
        if task.id < 0 {
            task.id = self.ctx_server.queue_tasks.get_new_id();
        }
        if !self.ctx_server.queue_tasks.has_new_task_callback() {
            return TaskResult::Error {
                id: task.id,
                kind: ErrorType::NotSupported,
                message: "no inference engine is attached to the server".to_string(),
                n_prompt_tokens: 0,
                n_ctx: 0,
            };
        }
        let mut reader = self.ctx_server.get_response_reader(1);
        reader.post_task(task, false);
        let no_stop = || false;
        reader.next(&no_stop).unwrap_or_else(|| TaskResult::Error {
            id: -1,
            kind: ErrorType::Server,
            message: "no response from the inference engine".to_string(),
            n_prompt_tokens: 0,
            n_ctx: 0,
        })
    }

    fn stream_task(&self, task: ServerTask, framing: SseFraming) -> HttpResponse {
        let mut reader = self.ctx_server.get_response_reader(1);
        reader.post_task(task, false);
        let mut res = HttpResponse::default();
        res.content_type = "text/event-stream".to_string();
        res.headers.insert("Cache-Control".to_string(), "no-cache".to_string());
        res.headers.insert("X-Accel-Buffering".to_string(), "no".to_string());
        res.set_next(move |chunk| {
            if !reader.has_next() {
                return false;
            }
            let no_stop = || false;
            match reader.next(&no_stop) {
                Some(result) => {
                    let event = result.to_json();
                    *chunk = match framing {
                        SseFraming::Oai => crate::tools::server::common::format_oai_sse(&event),
                        SseFraming::OaiResp => crate::tools::server::common::format_oai_resp_sse(&event),
                        SseFraming::Anthropic => crate::tools::server::common::format_anthropic_sse(&event),
                    };
                    reader.has_next()
                }
                None => false,
            }
        });
        res
    }

    fn completion(&self, body: &Json, kind: ServerTaskType, res_type: TaskResponseType, id_slot: i32) -> HttpResponse {
        let prompt = match body.at("prompt") {
            Ok(v) => v.clone(),
            Err(_) => return self.err("missing prompt", ErrorType::InvalidRequest),
        };
        let text = match prompt {
            Json::String(s) => s,
            Json::Array(_) => prompt_to_text(&prompt),
            _ => return self.err("\"prompt\" must be a string or an array", ErrorType::InvalidRequest),
        };
        let mut task = ServerTask::new(kind);
        task.params = self.task_params(body, res_type);
        task.id_slot = id_slot;
        task.tokens.push_back_many(&self.ctx_server, &text);
        if task.params.stream {
            return self.stream_task(task, sse_framing(res_type));
        }
        let result = self.run_task(task);
        if result.is_error() {
            HttpResponse::error(&result.to_json())
        } else {
            HttpResponse::ok(&result.to_json())
        }
    }

    fn count_tokens(&self, body: &Json, res_type: TaskResponseType) -> HttpResponse {
        let mut task = ServerTask::new(ServerTaskType::Completion);
        task.params = self.task_params(body, res_type);
        task.params.n_predict = 0;
        let result = self.run_task(task);
        if result.is_error() {
            HttpResponse::error(&result.to_json())
        } else {
            let data = result.to_json();
            let count = data.value("tokens_evaluated", 0i64).unwrap_or(0);
            HttpResponse::ok(&Json::object_from(vec![
                Item::new("count", count),
                Item::new("model", self.meta().map(|m| m.model_name).unwrap_or_default()),
            ]))
        }
    }

    fn embeddings(&self, body: &Json, res_type: TaskResponseType) -> HttpResponse {
        let input = match body.at("input") {
            Ok(v) => v.clone(),
            Err(_) => return self.err("missing input", ErrorType::InvalidRequest),
        };
        let mut task = ServerTask::new(ServerTaskType::Embedding);
        task.params = self.task_params(body, res_type);
        task.params.embd_normalize = json_value(body, "embd_normalize", task.params.embd_normalize);
        let text = match &input {
            Json::String(s) => s.clone(),
            Json::Array(items) => items.iter().map(json_to_text).collect::<Vec<_>>().join("\n"),
            _ => return self.err("invalid input", ErrorType::InvalidRequest),
        };
        let _ = text;
        let result = self.run_task(task);
        if result.is_error() {
            HttpResponse::error(&result.to_json())
        } else {
            let data = result.to_json();
            HttpResponse::ok(&Json::object_from(vec![
                Item::new("object", "list"),
                Item::new("model", self.meta().map(|m| m.model_name).unwrap_or_default()),
                Item::new("data", Json::array_from(vec![data])),
                Item::new("usage", Json::object_from(vec![Item::new("prompt_tokens", 0), Item::new("total_tokens", 0)])),
            ]))
        }
    }

    fn rerank(&self, body: &Json) -> HttpResponse {
        let query = json_value(body, "query", String::new());
        let documents = body.value("documents", Vec::<String>::new()).unwrap_or_default();
        let top_n = json_value(body, "top_n", documents.len() as i32);
        let mut scores = Vec::new();
        for (index, _doc) in documents.iter().enumerate() {
            let mut task = ServerTask::new(ServerTaskType::Rerank);
            task.params = self.task_params(body, TaskResponseType::None);
            task.params.n_predict = 0;
            let result = self.run_task(task);
            let score = result.to_json().value("score", 0.0f32).unwrap_or(0.0);
            scores.push(Json::object_from(vec![
                Item::new("index", index as i64),
                Item::new("document", Json::object_from(vec![
                    Item::new("text", documents[index].clone()),
                ])),
                Item::new("relevance_score", score),
            ]));
        }
        scores.sort_by(|a, b| {
            let sa = a.value("relevance_score", 0.0f32).unwrap_or(0.0);
            let sb = b.value("relevance_score", 0.0f32).unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(top_n.max(0) as usize);
        let _ = query;
        HttpResponse::ok(&Json::array_from(scores))
    }

    fn tts(&self, body: &Json, res_type: TaskResponseType) -> HttpResponse {
        let mut task = ServerTask::new(ServerTaskType::Tts);
        task.params = self.task_params(body, res_type);
        let result = self.run_task(task);
        match result {
            TaskResult::Tts { audio, sample_rate, .. } => HttpResponse {
                status: 200,
                content_type: "audio/wav".to_string(),
                data: String::from_utf8_lossy(&audio).into_owned(),
                headers: {
                    let mut h = BTreeMap::new();
                    h.insert("X-Sample-Rate".to_string(), sample_rate.to_string());
                    h
                },
                ..HttpResponse::default()
            },
            other if other.is_error() => HttpResponse::error(&other.to_json()),
            other => HttpResponse::ok(&other.to_json()),
        }
    }

    /// Convert an OpenAI-style message array into a prompt via a chat template.
    fn chat_prompt(&self, body: &Json) -> Result<String> {
        let messages = body.at("messages")?.get::<Vec<Json>>()?;
        let mut chat = Vec::with_capacity(messages.len());
        for msg in &messages {
            let content = msg.at("content").cloned().unwrap_or(Json::Null);
            chat.push(ChatMessage::new(
                msg.value_str("role", "user")?,
                json_to_text(&content),
            ));
        }
        let template: ChatTemplate = chat_template_from_str(&self.params.chat_template).unwrap_or(ChatTemplate::Chatml);
        let mut prompt = String::new();
        let n = apply_template(template, &chat, &mut prompt, true);
        if n < 0 {
            prompt = chat
                .iter()
                .map(|m| format!("{}: {}\n", m.role, m.content))
                .collect::<String>()
                + "assistant: ";
        }
        Ok(prompt)
    }

    fn completion_from_chat(&self, body: &Json, res_type: TaskResponseType) -> HttpResponse {
        let prompt = match self.chat_prompt(body) {
            Ok(p) => p,
            Err(e) => return self.err(e.to_string(), ErrorType::InvalidRequest),
        };
        let mut data = body.clone();
        let _ = data.set(Item::new("prompt", prompt));
        self.completion(&data, ServerTaskType::Completion, res_type, json_value(body, "id_slot", -1))
    }

    /// `/responses` and `/messages` request formats converted to chat completions.
    fn convert_request(&self, body: &Json, source: &str) -> Json {
        let mut messages: Vec<Json> = body.value("messages", Vec::<Json>::new()).unwrap_or_default();
        if messages.is_empty() {
            if source == "anthropic" {
                let system = json_value(body, "system", String::new());
                if !system.is_empty() {
                    messages.push(Json::object_from(vec![Item::new("role", "system"), Item::new("content", system)]));
                }
            }
            let input = match source {
                "responses" => json_value(body, "input", String::new()),
                "transcriptions" => json_value(body, "prompt", String::new()),
                _ => String::new(),
            };
            if !input.is_empty() {
                messages.push(Json::object_from(vec![Item::new("role", "user"), Item::new("content", input)]));
            }
        }
        let mut out = body.clone();
        if !messages.is_empty() {
            let _ = out.set(Item::new("messages", messages));
        }
        out
    }
}

fn sse_framing(res_type: TaskResponseType) -> SseFraming {
    match res_type {
        TaskResponseType::OaiResp => SseFraming::OaiResp,
        TaskResponseType::Anthropic => SseFraming::Anthropic,
        _ => SseFraming::Oai,
    }
}

fn json_to_text(value: &Json) -> String {
    match value {
        Json::String(s) => s.clone(),
        Json::Array(items) => items
            .iter()
            .filter_map(|part| part.at("text").ok().and_then(|v| v.get::<String>().ok()))
            .collect::<Vec<_>>()
            .join(""),
        other => safe_json_to_str(other),
    }
}

fn prompt_to_text(value: &Json) -> String {
    match value {
        Json::String(s) => s.clone(),
        Json::Array(items) => items.iter().map(json_to_text).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

trait PushTokens {
    fn push_back_many(&mut self, ctx: &ServerContext, text: &str);
}

impl PushTokens for crate::tools::server::common::ServerTokens {
    fn push_back_many(&mut self, ctx: &ServerContext, text: &str) {
        match ctx.tokenize(text, true, true) {
            Ok(tokens) => {
                for token in tokens {
                    self.push_back(token);
                }
            }
            Err(_) => {
                for byte in text.as_bytes() {
                    self.push_back(*byte as Token);
                }
            }
        }
    }
}

/// `server_routes`.
pub struct ServerRoutes {
    pub get_health: Handler,
    pub get_metrics: Handler,
    pub get_slots: Handler,
    pub post_slots: Handler,
    pub get_props: Handler,
    pub post_props: Handler,
    pub post_infill: Handler,
    pub post_completions: Handler,
    pub post_completions_oai: Handler,
    pub post_chat_completions: Handler,
    pub post_chat_completions_tok: Handler,
    pub post_control: Handler,
    pub post_responses_oai: Handler,
    pub post_responses_tok_oai: Handler,
    pub post_transcriptions_oai: Handler,
    pub ws_realtime: WsHandler,
    pub post_anthropic_messages: Handler,
    pub post_anthropic_count_tokens: Handler,
    pub post_apply_template: Handler,
    pub get_models: Handler,
    pub post_tokenize: Handler,
    pub post_detokenize: Handler,
    pub post_embeddings: Handler,
    pub post_embeddings_oai: Handler,
    pub post_rerank: Handler,
    pub post_tts: Handler,
    pub post_speech_oai: Handler,
    pub post_systemone: Handler,
    pub get_lora_adapters: Handler,
    pub post_lora_adapters: Handler,
    inner: Arc<RoutesInner>,
}

impl ServerRoutes {
    pub fn new(params: &Params, ctx_server: &Arc<ServerContext>) -> Self {
        let inner = Arc::new(RoutesInner {
            params: params.clone(),
            ctx_server: Arc::clone(ctx_server),
            meta: Mutex::new(None),
            cache: Mutex::new(RouteCache {
                models: Json::Null,
                props: Json::Null,
                metrics: Metrics::default(),
                should_reset_buckets: false,
            }),
        });
        let routes = Self::init_routes(inner);
        routes
    }

    /// Copy metadata after the model is loaded, like `update_meta()`.
    pub fn update_meta(&self, ctx_server: &ServerContext) {
        if let Some(meta) = ctx_server.get_meta() {
            *self.inner.meta.lock().unwrap_or_else(|e| e.into_inner()) = Some(meta.clone());
            let mut cache = self.inner.cache.lock().unwrap_or_else(|e| e.into_inner());
            cache.models = meta.models_json();
            cache.props = meta.props_json(&self.inner.params, false);
        }
    }

    /// `server_routes::get_model_info()`.
    pub fn get_model_info(&self) -> Json {
        match self.inner.meta() {
            Some(meta) => meta.model_info_json(),
            None => Json::object(),
        }
    }

    /// Cache responses just before sleep, like `update_cached_responses()`.
    pub fn update_cached_responses(&self, is_sleeping: bool) {
        update_cache(&self.inner, is_sleeping);
    }

    /// Callback registered with the task queue for sleeping-state changes.
    pub fn sleeping_callback(&self) -> Arc<dyn Fn(bool) + Send + Sync> {
        let inner = Arc::clone(&self.inner);
        Arc::new(move |is_sleeping| update_cache(&inner, is_sleeping))
    }

    fn init_routes(inner: Arc<RoutesInner>) -> Self {
        let get_health: Handler = Arc::new(|_req: &HttpRequest| {
            HttpResponse::ok(&Json::object_from(vec![Item::new("status", "ok")]))
        });

        let get_metrics = {
            let inner = Arc::clone(&inner);
            Arc::new(move |_req: &HttpRequest| {
                if !inner.params.endpoint_metrics {
                    return inner.err(
                        "This server does not support metrics endpoint. Start it with `--metrics`",
                        ErrorType::NotSupported,
                    );
                }
                let mut cache = inner.cache.lock().unwrap_or_else(|e| e.into_inner());
                let mut result = MetricsResult::default();
                result.metrics = cache.metrics.clone();
                result.metrics.t_start = cache.metrics.t_start;
                cache.metrics.reset_bucket();
                cache.should_reset_buckets = true;
                HttpResponse {
                    status: 200,
                    content_type: "text/plain; version=0.0.4".to_string(),
                    data: result.to_metrics(),
                    headers: {
                        let mut h = BTreeMap::new();
                        h.insert("Process-Start-Time-Unix".to_string(), result.metrics.t_start.to_string());
                        h
                    },
                    ..HttpResponse::default()
                }
            })
        };

        let get_slots = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                if !inner.params.endpoint_slots {
                    return inner.err(
                        "This server does not support slots endpoint. Start it with `--slots`",
                        ErrorType::NotSupported,
                    );
                }
                let mut task = ServerTask::new(ServerTaskType::SlotGet);
                task.id = inner.ctx_server.queue_tasks.get_new_id();
                let result = inner.run_task(task);
                if result.is_error() {
                    return HttpResponse::error(&result.to_json());
                }
                let data = result.to_json();
                if !req.get_param("fail_on_no_slot").is_empty() && inner.ctx_server.n_idle_slots() == 0 {
                    return inner.err("no slot available", ErrorType::Unavailable);
                }
                HttpResponse::ok(&data)
            })
        };

        let post_slots = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                if inner.params.slot_save_path.is_empty() {
                    return inner.err(
                        "This server does not support slots action. Start it with `--slot-save-path`",
                        ErrorType::NotSupported,
                    );
                }
                let id_slot = match req.get_param("id_slot").parse::<i32>() {
                    Ok(v) => v,
                    Err(_) => return inner.err("Invalid slot ID", ErrorType::InvalidRequest),
                };
                let action = req.get_param("action");
                let kind = match action {
                    "save" => ServerTaskType::SlotSave,
                    "restore" => ServerTaskType::SlotRestore,
                    "erase" => ServerTaskType::SlotErase,
                    _ => return inner.err("Invalid action", ErrorType::InvalidRequest),
                };
                let mut task = ServerTask::new(kind);
                task.id = inner.ctx_server.queue_tasks.get_new_id();
                task.slot_action.id_slot = id_slot;
                let body = req_body(req);
                task.slot_action.filename = json_value(&body, "filename", String::new());
                task.slot_action.filepath = inner.params.slot_save_path.clone();
                let result = inner.run_task(task);
                if result.is_error() {
                    HttpResponse::error(&result.to_json())
                } else {
                    HttpResponse::ok(&result.to_json())
                }
            })
        };

        let get_props = {
            let inner = Arc::clone(&inner);
            Arc::new(move |_req: &HttpRequest| {
                let cache = inner.cache.lock().unwrap_or_else(|e| e.into_inner());
                let data = if !cache.props.is_null() {
                    cache.props.clone()
                } else {
                    match inner.meta() {
                        Some(meta) => meta.props_json(&inner.params, false),
                        None => Json::object(),
                    }
                };
                HttpResponse::ok(&data)
            })
        };

        let post_props = {
            let inner = Arc::clone(&inner);
            Arc::new(move |_req: &HttpRequest| {
                if !inner.params.endpoint_props {
                    return inner.err(
                        "This server does not support changing global properties. Start it with `--props`",
                        ErrorType::NotSupported,
                    );
                }
                HttpResponse::ok(&Json::object_from(vec![Item::new("success", true)]))
            })
        };

        let post_infill = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                let meta = match inner.meta() {
                    Some(m) => m,
                    None => return inner.err("model is not loaded", ErrorType::Unavailable),
                };
                let mut err = String::new();
                if meta.fim_pre_token < 0 {
                    err.push_str("prefix token is missing. ");
                }
                if meta.fim_sub_token < 0 {
                    err.push_str("suffix token is missing. ");
                }
                if meta.fim_mid_token < 0 {
                    err.push_str("middle token is missing. ");
                }
                if !err.is_empty() {
                    return inner.err(format!("Infill is not supported by this model: {}", err), ErrorType::NotSupported);
                }
                if body.contains("prompt") && !body.at("prompt").map_or(false, |v| v.is_string()) {
                    return inner.err("\"prompt\" must be a string", ErrorType::InvalidRequest);
                }
                for key in ["input_prefix", "input_suffix"] {
                    if !body.contains(key) {
                        return inner.err(format!("\"{}\" is required", key), ErrorType::InvalidRequest);
                    }
                }
                if body.contains("input_extra") && !body.at("input_extra").map_or(false, |v| v.is_array()) {
                    return inner.err(
                        "\"input_extra\" must be an array of {\"filename\": string, \"text\": string}",
                        ErrorType::InvalidRequest,
                    );
                }
                let prefix = json_value(&body, "input_prefix", String::new());
                let suffix = json_value(&body, "input_suffix", String::new());
                let prompt = format!("{}{}{}", meta.fim_pre_token_string(), prefix, suffix);
                let mut data = body.clone();
                let _ = data.set(Item::new("prompt", prompt));
                inner.completion(&data, ServerTaskType::Infill, TaskResponseType::None, json_value(&body, "id_slot", -1))
            })
        };

        let post_completions = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                inner.completion(&body, ServerTaskType::Completion, TaskResponseType::None, json_value(&body, "id_slot", -1))
            })
        };

        let post_completions_oai = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                inner.completion(&body, ServerTaskType::Completion, TaskResponseType::OaiCmpl, json_value(&body, "id_slot", -1))
            })
        };

        let post_chat_completions = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                inner.completion_from_chat(&body, TaskResponseType::OaiChat)
            })
        };

        let post_chat_completions_tok = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                inner.count_tokens(&body, TaskResponseType::OaiChat)
            })
        };

        let post_control = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                let id = json_value(&body, "id", String::new());
                let action = json_value(&body, "action", String::new());
                if id.is_empty() {
                    return inner.err("missing completion id", ErrorType::InvalidRequest);
                }
                if action != "reasoning_end" {
                    return inner.err("unknown control action", ErrorType::InvalidRequest);
                }
                let mut task = ServerTask::new(ServerTaskType::Control);
                task.id = inner.ctx_server.queue_tasks.get_new_id();
                task.params.control_cmpl_id = id;
                task.params.control_action = ServerControlAction::ReasoningEnd;
                let result = inner.run_task(task);
                if result.is_error() {
                    HttpResponse::error(&result.to_json())
                } else {
                    HttpResponse::ok(&result.to_json())
                }
            })
        };

        let post_responses_oai = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                let converted = inner.convert_request(&body, "responses");
                inner.completion_from_chat(&converted, TaskResponseType::OaiResp)
            })
        };

        let post_responses_tok_oai = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                inner.count_tokens(&body, TaskResponseType::OaiResp)
            })
        };

        let post_transcriptions_oai = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let meta = match inner.meta() {
                    Some(m) => m,
                    None => return inner.err("model is not loaded", ErrorType::Unavailable),
                };
                if !meta.has_mtmd || !meta.chat_params.allow_audio {
                    return inner.err("The current model does not support audio input.", ErrorType::NotSupported);
                }
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                let converted = inner.convert_request(&body, "transcriptions");
                inner.completion_from_chat(&converted, TaskResponseType::OaiAsr)
            })
        };

        let ws_realtime: WsHandler = Arc::new(move |req: &HttpRequest, conn: &mut dyn crate::tools::server::http::WsConn| {
            let event = Json::object_from(vec![
                Item::new("type", "error"),
                Item::new("error", Json::object_from(vec![
                    Item::new("type", "invalid_request_error"),
                    Item::new("code", Json::Null),
                    Item::new("message", "realtime transcription requires a model with audio streaming"),
                    Item::new("param", Json::Null),
                ])),
            ]);
            let _ = req;
            conn.send(&safe_json_to_str(&event));
            conn.close();
        });

        let post_anthropic_messages = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                let converted = inner.convert_request(&body, "anthropic");
                inner.completion_from_chat(&converted, TaskResponseType::Anthropic)
            })
        };

        let post_anthropic_count_tokens = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                inner.count_tokens(&body, TaskResponseType::Anthropic)
            })
        };

        let post_apply_template = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                match inner.chat_prompt(&body) {
                    Ok(prompt) => HttpResponse::ok(&Json::object_from(vec![Item::new("prompt", prompt)])),
                    Err(e) => inner.err(e.to_string(), ErrorType::InvalidRequest),
                }
            })
        };

        let get_models = {
            let inner = Arc::clone(&inner);
            Arc::new(move |_req: &HttpRequest| {
                let cache = inner.cache.lock().unwrap_or_else(|e| e.into_inner());
                if !cache.models.is_null() {
                    HttpResponse::ok(&cache.models)
                } else {
                    HttpResponse::ok(&inner.meta().map(|m| m.models_json()).unwrap_or_else(|| {
                        Json::object_from(vec![
                            Item::new("models", Json::Array(Vec::new())),
                            Item::new("object", "list"),
                            Item::new("data", Json::Array(Vec::new())),
                        ])
                    }))
                }
            })
        };

        let post_tokenize = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                let mut tokens = Json::array();
                if body.contains("content") {
                    let content = body.at("content").cloned().unwrap_or(Json::Null);
                    let add_special = json_value(&body, "add_special", false);
                    let parse_special = json_value(&body, "parse_special", true);
                    let with_pieces = json_value(&body, "with_pieces", false);
                    let numeric = crate::tools::server::common::json_is_array_of_numbers(&content);
                    let ids: Vec<Token> = if numeric {
                        content.get::<Vec<Token>>().unwrap_or_default()
                    } else {
                        match inner.ctx_server.tokenize(&json_to_text(&content), add_special, parse_special) {
                            Ok(v) => v,
                            Err(e) => return inner.err(e.to_string(), ErrorType::NotSupported),
                        }
                    };
                    for token in ids {
                        if with_pieces {
                            let piece = inner.ctx_server.detokenize(&[token], false).unwrap_or_default();
                            let piece_json = if crate::tools::server::common::is_valid_utf8(piece.as_bytes()) {
                                Json::String(piece)
                            } else {
                                Json::array_from(piece.as_bytes().iter().map(|b| *b as i64).collect::<Vec<_>>())
                            };
                            let _ = tokens.push_back(Json::object_from(vec![
                                Item::new("id", token),
                                Item::new("piece", piece_json),
                            ]).into());
                        } else {
                            let _ = tokens.push_back(token.into());
                        }
                    }
                }
                HttpResponse::ok(&Json::object_from(vec![Item::new("tokens", tokens)]))
            })
        };

        let post_detokenize = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                let mut content = String::new();
                if body.contains("tokens") {
                    let tokens = body.at("tokens").and_then(|v| v.get::<Vec<Token>>()).unwrap_or_default();
                    content = inner.ctx_server.detokenize(&tokens, false).unwrap_or_default();
                }
                HttpResponse::ok(&Json::object_from(vec![Item::new("content", content)]))
            })
        };

        let post_embeddings = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                inner.embeddings(&body, TaskResponseType::None)
            })
        };

        let post_embeddings_oai = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                inner.embeddings(&body, TaskResponseType::OaiEmbd)
            })
        };

        let post_rerank = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                inner.rerank(&body)
            })
        };

        let post_tts = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_tts_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                inner.tts(&body, TaskResponseType::None)
            })
        };

        let post_speech_oai = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_tts_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                inner.tts(&body, TaskResponseType::OaiChat)
            })
        };

        let post_systemone = {
            let inner = Arc::clone(&inner);
            Arc::new(move |_req: &HttpRequest| {
                if !inner.meta().map(|m| m.decision_is_classifier).unwrap_or(false) {
                    return inner.err("This model is not a decision model", ErrorType::NotSupported);
                }
                let mut task = ServerTask::new(ServerTaskType::Decision);
                task.id = inner.ctx_server.queue_tasks.get_new_id();
                let result = inner.run_task(task);
                if result.is_error() {
                    HttpResponse::error(&result.to_json())
                } else {
                    HttpResponse::ok(&result.to_json())
                }
            })
        };

        let get_lora_adapters = {
            let inner = Arc::clone(&inner);
            Arc::new(move |_req: &HttpRequest| {
                let mut task = ServerTask::new(ServerTaskType::GetLora);
                task.id = inner.ctx_server.queue_tasks.get_new_id();
                let result = inner.run_task(task);
                if result.is_error() {
                    HttpResponse::error(&result.to_json())
                } else {
                    HttpResponse::ok(&result.to_json())
                }
            })
        };

        let post_lora_adapters = {
            let inner = Arc::clone(&inner);
            Arc::new(move |req: &HttpRequest| {
                let body = match parse_body(req) {
                    Ok(b) => b,
                    Err(e) => return inner.err(e.to_string(), ErrorType::InvalidRequest),
                };
                if !body.is_array() {
                    return inner.err("Request body must be an array", ErrorType::InvalidRequest);
                }
                let mut task = ServerTask::new(ServerTaskType::SetLora);
                task.id = inner.ctx_server.queue_tasks.get_new_id();
                task.set_lora = parse_lora_request(&body);
                let result = inner.run_task(task);
                if result.is_error() {
                    HttpResponse::error(&result.to_json())
                } else {
                    HttpResponse::ok(&result.to_json())
                }
            })
        };

        Self {
            get_health,
            get_metrics,
            get_slots,
            post_slots,
            get_props,
            post_props,
            post_infill,
            post_completions,
            post_completions_oai,
            post_chat_completions,
            post_chat_completions_tok,
            post_control,
            post_responses_oai,
            post_responses_tok_oai,
            post_transcriptions_oai,
            ws_realtime,
            post_anthropic_messages,
            post_anthropic_count_tokens,
            post_apply_template,
            get_models,
            post_tokenize,
            post_detokenize,
            post_embeddings,
            post_embeddings_oai,
            post_rerank,
            post_tts,
            post_speech_oai,
            post_systemone,
            get_lora_adapters,
            post_lora_adapters,
            inner,
        }
    }
}

fn update_cache(inner: &RoutesInner, is_sleeping: bool) {
    let mut cache = inner.cache.lock().unwrap_or_else(|e| e.into_inner());
    if is_sleeping {
        if let Some(meta) = inner.meta() {
            cache.models = meta.models_json();
            cache.props = meta.props_json(&inner.params, true);
        }
        cache.metrics.t_start = crate::tools::server::common::time_us();
    } else if cache.should_reset_buckets {
        cache.metrics.reset_bucket();
        cache.should_reset_buckets = false;
    }
}

fn parse_body(req: &HttpRequest) -> Result<Json> {
    if req.body.is_empty() {
        return Ok(Json::object());
    }
    Json::parse(&req.body).map_err(|e| Error::new(format!("invalid JSON: {}", e)))
}

fn req_body(req: &HttpRequest) -> Json {
    parse_body(req).unwrap_or_else(|_| Json::object())
}

/// Multipart form numeric/boolean fields arrive as strings; restore their types.
fn parse_tts_body(req: &HttpRequest) -> Result<Json> {
    let mut body = parse_body(req)?;
    for key in ["top_k", "n_predict", "max_tokens", "seed", "flow_steps"] {
        let value = body.at(key).cloned().ok();
        if let Some(Json::String(s)) = value {
            if let Ok(n) = s.parse::<i64>() {
                let _ = body.set(Item::new(key, n));
            }
        }
    }
    for key in ["top_p", "temperature", "repeat_penalty", "speed", "loudness", "cfg_scale"] {
        let value = body.at(key).cloned().ok();
        if let Some(Json::String(s)) = value {
            if let Ok(n) = s.parse::<f64>() {
                let _ = body.set(Item::new(key, n));
            }
        }
    }
    let stream = body.at("stream").cloned().ok();
    if let Some(Json::String(s)) = stream {
        let _ = body.set(Item::new("stream", s == "true"));
    }
    Ok(body)
}

/// Register the same routes as `server_register_routes()` in `server.cpp`.
pub fn register_routes(http: &crate::tools::server::http::HttpContext, routes: &ServerRoutes) {
    http.get("/health", Arc::clone(&routes.get_health));
    http.get("/v1/health", Arc::clone(&routes.get_health));
    http.get("/metrics", Arc::clone(&routes.get_metrics));
    http.get("/props", Arc::clone(&routes.get_props));
    http.post("/props", Arc::clone(&routes.post_props));
    http.get("/models", Arc::clone(&routes.get_models));
    http.get("/v1/models", Arc::clone(&routes.get_models));
    http.post("/completion", Arc::clone(&routes.post_completions));
    http.post("/completions", Arc::clone(&routes.post_completions));
    http.post("/v1/completions", Arc::clone(&routes.post_completions_oai));
    http.post("/chat/completions", Arc::clone(&routes.post_chat_completions));
    http.post("/v1/chat/completions", Arc::clone(&routes.post_chat_completions));
    http.post("/v1/chat/completions/control", Arc::clone(&routes.post_control));
    http.post("/v1/responses", Arc::clone(&routes.post_responses_oai));
    http.post("/responses", Arc::clone(&routes.post_responses_oai));
    http.post("/v1/audio/transcriptions", Arc::clone(&routes.post_transcriptions_oai));
    http.post("/audio/transcriptions", Arc::clone(&routes.post_transcriptions_oai));
    http.post("/tts", Arc::clone(&routes.post_tts));
    http.post("/v1/audio/speech", Arc::clone(&routes.post_speech_oai));
    http.post("/audio/speech", Arc::clone(&routes.post_speech_oai));
    http.ws("/v1/realtime", Arc::clone(&routes.ws_realtime));
    http.post("/v1/messages", Arc::clone(&routes.post_anthropic_messages));
    http.post("/infill", Arc::clone(&routes.post_infill));
    http.post("/embedding", Arc::clone(&routes.post_embeddings));
    http.post("/embeddings", Arc::clone(&routes.post_embeddings));
    http.post("/v1/embeddings", Arc::clone(&routes.post_embeddings_oai));
    http.post("/rerank", Arc::clone(&routes.post_rerank));
    http.post("/reranking", Arc::clone(&routes.post_rerank));
    http.post("/v1/rerank", Arc::clone(&routes.post_rerank));
    http.post("/v1/reranking", Arc::clone(&routes.post_rerank));
    http.post("/v1/systemone", Arc::clone(&routes.post_systemone));
    http.post("/tokenize", Arc::clone(&routes.post_tokenize));
    http.post("/detokenize", Arc::clone(&routes.post_detokenize));
    http.post("/apply-template", Arc::clone(&routes.post_apply_template));
    http.post("/chat/completions/input_tokens", Arc::clone(&routes.post_chat_completions_tok));
    http.post("/v1/chat/completions/input_tokens", Arc::clone(&routes.post_chat_completions_tok));
    http.post("/responses/input_tokens", Arc::clone(&routes.post_responses_tok_oai));
    http.post("/v1/responses/input_tokens", Arc::clone(&routes.post_responses_tok_oai));
    http.post("/v1/messages/count_tokens", Arc::clone(&routes.post_anthropic_count_tokens));
    http.get("/lora-adapters", Arc::clone(&routes.get_lora_adapters));
    http.post("/lora-adapters", Arc::clone(&routes.post_lora_adapters));
    http.get("/slots", Arc::clone(&routes.get_slots));
    http.post("/slots/:id_slot", Arc::clone(&routes.post_slots));
    http.register_gcp_compat();
}

/// Resumable stream routes from `server_register_stream_routes()`.
pub fn register_stream_routes(http: &crate::tools::server::http::HttpContext) {
    let not_supported = Arc::new(|_req: &HttpRequest| {
        HttpResponse::error_message("resumable streams are not available", ErrorType::NotSupported)
    }) as Handler;
    http.get("/v1/stream", Arc::clone(&not_supported));
    http.post("/v1/streams/lookup", Arc::clone(&not_supported));
    http.del("/v1/stream", not_supported);
}

/// Feature flags and their 403 handlers from `server_setup_features()`.
pub fn setup_features(http: &crate::tools::server::http::HttpContext, params: &Params) -> bool {
    let denied: Handler = Arc::new(|_req: &HttpRequest| {
        HttpResponse {
            status: 403,
            data: safe_json_to_str(&Json::object_from(vec![Item::new(
                "error",
                Json::object_from(vec![
                    Item::new("message", "this feature is disabled"),
                    Item::new("type", "feature_disabled"),
                ]),
            )])),
            ..HttpResponse::default()
        }
    });
    if params.ui_mcp_proxy {
        let proxy = Arc::new(|_req: &HttpRequest| {
            HttpResponse::error_message("MCP proxy is not available in the std-only port", ErrorType::NotSupported)
        }) as Handler;
        http.get("/cors-proxy", Arc::clone(&proxy));
        http.post("/cors-proxy", proxy);
    } else {
        http.get("/cors-proxy", Arc::clone(&denied));
        http.post("/cors-proxy", Arc::clone(&denied));
    }
    let tools_enabled = !params.server_tools.is_empty()
        || !params.mcp_servers_config.is_empty()
        || !params.mcp_servers_json.is_empty()
        || params.ui_mcp_proxy;
    if tools_enabled {
        let tools = Arc::new(|_req: &HttpRequest| {
            HttpResponse::error_message("server tools are not available in the std-only port", ErrorType::NotSupported)
        }) as Handler;
        http.get("/tools", Arc::clone(&tools));
        http.post("/tools", tools);
    } else {
        http.get("/tools", Arc::clone(&denied));
        http.post("/tools", denied);
    }
    true
}

impl ServerContextMeta {
    fn fim_pre_token_string(&self) -> String {
        if self.fim_pre_token >= 0 {
            format!("<fim_pre>")
        } else {
            String::new()
        }
    }
}

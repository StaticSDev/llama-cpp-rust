//! Port of `tools/server/server-context.{h,cpp}` state layer.
//!
//! Model metadata, slot state and the task loop are represented as owned Rust
//! data. Llama contexts stay behind the `crate::api` module: this layer never
//! touches raw pointers, so `get_llama_context()` is an `Option` with no FFI.

use crate::common::Params;
use crate::core::{Error, Result};
use crate::json::{Item, Json};
use crate::tools::server::common::{
    server_model_architecture_json, server_model_output_modalities, ServerChatParams, ServerRoleType, ServerState,
    ServerTokens,
};
use crate::tools::server::queue::{ServerQueue, ServerResponse};
use crate::tools::server::task::{ServerTask, TaskParams};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

/// Callback fired when the model changes state (`server_state_callback_t`).
pub type StateCallback = Arc<dyn Fn(ServerState, Json) + Send + Sync>;

/// `server_context_meta`.
#[derive(Clone, Debug)]
pub struct ServerContextMeta {
    pub build_info: String,
    pub model_name: String,
    pub model_aliases: BTreeSet<String>,
    pub model_tags: BTreeSet<String>,
    pub model_path: String,
    pub model_output_modalities: Vec<String>,
    pub has_mtmd: bool,
    pub has_inp_image: bool,
    pub has_inp_audio: bool,
    pub has_inp_video: bool,
    pub has_cap_chat: bool,
    pub has_cap_tts: bool,
    pub has_audio_stream: bool,
    pub tts_voices: Vec<String>,
    pub json_ui_settings: Json,
    pub slot_n_ctx: i32,
    pub pooling_type: crate::cparams::PoolingType,
    pub chat_params: ServerChatParams,
    pub chat_template_caps: Json,
    pub bos_token_str: String,
    pub eos_token_str: String,
    pub fim_pre_token: i32,
    pub fim_sub_token: i32,
    pub fim_mid_token: i32,
    pub fim_pad_token: i32,
    pub fim_rep_token: i32,
    pub fim_sep_token: i32,
    pub model_vocab_type: i32,
    pub model_vocab_n_tokens: i32,
    pub model_n_ctx_train: i32,
    pub model_n_embd_inp: i32,
    pub model_n_params: u64,
    pub model_size: u64,
    pub model_ftype: String,
    pub decision_is_classifier: bool,
}

impl ServerContextMeta {
    /// Fill metadata from the CLI parameters. A real model loader overrides this.
    pub fn from_params(params: &Params) -> Self {
        Self {
            build_info: format!("b{}", crate::api::LLAMA_VERSION),
            model_name: params.model.get_name().to_string(),
            model_aliases: params.model_alias.clone(),
            model_tags: params.model_tags.clone(),
            model_path: params.model.path.clone(),
            model_output_modalities: server_model_output_modalities(false, true, false),
            has_mtmd: !params.mmproj.path.is_empty(),
            has_inp_image: !params.mmproj.path.is_empty(),
            has_inp_audio: false,
            has_inp_video: false,
            has_cap_chat: true,
            has_cap_tts: false,
            has_audio_stream: false,
            tts_voices: Vec::new(),
            json_ui_settings: Json::object(),
            slot_n_ctx: params.n_ctx,
            pooling_type: params.pooling_type,
            chat_params: ServerChatParams::from_params(params),
            chat_template_caps: Json::object(),
            bos_token_str: "<|begin_of_text|>".to_string(),
            eos_token_str: "<|end_of_text|>".to_string(),
            fim_pre_token: -1,
            fim_sub_token: -1,
            fim_mid_token: -1,
            fim_pad_token: -1,
            fim_rep_token: -1,
            fim_sep_token: -1,
            model_vocab_type: 0,
            model_vocab_n_tokens: 0,
            model_n_ctx_train: params.n_ctx,
            model_n_embd_inp: 0,
            model_n_params: 0,
            model_size: 0,
            model_ftype: "F32".to_string(),
            decision_is_classifier: false,
        }
    }

    /// `get_res_model_info()`: the `data[]` entry of `GET /v1/models`.
    pub fn model_info_json(&self) -> Json {
        Json::object_from(vec![
            Item::new("id", self.model_name.clone()),
            Item::new("aliases", self.model_aliases.iter().cloned().collect::<Vec<_>>()),
            Item::new("tags", self.model_tags.iter().cloned().collect::<Vec<_>>()),
            Item::new("object", "model"),
            Item::new(
                "architecture",
                server_model_architecture_json(
                    self.has_inp_image,
                    self.has_inp_audio,
                    self.has_inp_video,
                    &self.model_output_modalities,
                ),
            ),
            Item::new("created", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0)),
            Item::new("owned_by", "llamacpp"),
            Item::new("meta", Json::object_from(vec![
                Item::new("vocab_type", self.model_vocab_type),
                Item::new("n_vocab", self.model_vocab_n_tokens),
                Item::new("n_ctx", self.slot_n_ctx),
                Item::new("n_ctx_train", self.model_n_ctx_train),
                Item::new("n_embd", self.model_n_embd_inp),
                Item::new("n_params", self.model_n_params as u64),
                Item::new("size", self.model_size as u64),
                Item::new("ftype", self.model_ftype.clone()),
            ])),
        ])
    }

    /// `get_res_models()`: body of `GET /models`.
    pub fn models_json(&self) -> Json {
        Json::object_from(vec![
            Item::new("models", Json::array_from(vec![Json::object_from(vec![
                Item::new("name", self.model_name.clone()),
                Item::new("model", self.model_name.clone()),
                Item::new("modified_at", ""),
                Item::new("size", ""),
                Item::new("digest", ""),
                Item::new("type", "model"),
                Item::new("description", ""),
                Item::new("tags", Json::array_from(vec![""])),
                Item::new("capabilities", if self.has_mtmd {
                    Json::array_from(vec!["completion", "multimodal"])
                } else {
                    Json::array_from(vec!["completion"])
                }),
                Item::new("parameters", ""),
                Item::new("details", Json::object_from(vec![
                    Item::new("parent_model", ""),
                    Item::new("format", "gguf"),
                    Item::new("family", ""),
                    Item::new("families", Json::array_from(vec![""])),
                    Item::new("parameter_size", ""),
                    Item::new("quantization_level", ""),
                ])),
            ])])),
            Item::new("object", "list"),
            Item::new("data", Json::array_from(vec![self.model_info_json()])),
        ])
    }

    /// `get_res_props()`: body of `GET /props`.
    pub fn props_json(&self, params: &Params, is_sleeping: bool) -> Json {
        let mut tparams = TaskParams::from_params(params);
        tparams.res_type = crate::tools::server::task::TaskResponseType::None;
        let defaults = Json::object_from(vec![
            Item::new("params", tparams.to_json(true)),
            Item::new("n_ctx", self.slot_n_ctx),
        ]);
        let mut out = Json::object_from(vec![
            Item::new("default_generation_settings", defaults),
            Item::new("total_slots", params.n_parallel),
            Item::new("model_alias", self.model_name.clone()),
            Item::new("model_ftype", self.model_ftype.clone()),
            Item::new("model_path", self.model_path.clone()),
            Item::new("modalities", Json::object_from(vec![
                Item::new("vision", self.has_inp_image),
                Item::new("video", self.has_inp_video),
                Item::new("audio", self.has_inp_audio),
            ])),
            Item::new("media_marker", crate::tools::server::common::get_media_marker()),
            Item::new("endpoint_slots", params.endpoint_slots),
            Item::new("endpoint_props", params.endpoint_props),
            Item::new("endpoint_metrics", params.endpoint_metrics),
            Item::new("ui", params.ui),
            Item::new("ui_settings", self.json_ui_settings.clone()),
            Item::new("chat_template", params.chat_template.clone()),
            Item::new("chat_template_caps", self.chat_template_caps.clone()),
            Item::new("bos_token", self.bos_token_str.clone()),
            Item::new("eos_token", self.eos_token_str.clone()),
            Item::new("build_info", self.build_info.clone()),
            Item::new("is_sleeping", is_sleeping),
            Item::new("cors_proxy_enabled", params.ui_mcp_proxy),
            Item::new("ui_agent", params.ui_agent),
        ]);
        if !params.server_connect_code.is_empty() {
            let _ = out.set(Item::new("connect_code", params.server_connect_code.clone()));
        }
        if !self.tts_voices.is_empty() {
            let _ = out.set(Item::new("voices", self.tts_voices.clone()));
        }
        out
    }
}

/// `server_slot_info` and the state needed for `GET /slots`.
#[derive(Clone, Debug)]
pub struct ServerSlot {
    pub id: i32,
    pub n_ctx: i32,
    pub is_processing: bool,
    pub can_speculate: bool,
    pub id_task: i32,
    pub prompt: ServerTokens,
    pub n_prompt_tokens_processed: u64,
    pub n_prompt_tokens_cache: u64,
    pub n_gen: u64,
    pub has_next_token: bool,
    pub has_new_line: bool,
    pub n_remaining: i32,
    pub generated: String,
    pub params: TaskParams,
}

impl ServerSlot {
    pub fn new(id: i32, n_ctx: i32) -> Self {
        Self {
            id,
            n_ctx,
            is_processing: false,
            can_speculate: false,
            id_task: -1,
            prompt: ServerTokens::default(),
            n_prompt_tokens_processed: 0,
            n_prompt_tokens_cache: 0,
            n_gen: 0,
            has_next_token: false,
            has_new_line: false,
            n_remaining: 0,
            generated: String::new(),
            params: TaskParams::default(),
        }
    }

    /// `server_slot::to_json()`.
    pub fn to_json(&self, only_metrics: bool) -> Json {
        let mut out = Json::object_from(vec![
            Item::new("id", self.id),
            Item::new("n_ctx", self.n_ctx),
            Item::new("speculative", self.can_speculate),
            Item::new("is_processing", self.is_processing),
        ]);
        if self.id_task >= 0 {
            let _ = out.set(Item::new("id_task", self.id_task));
            let _ = out.set(Item::new("n_prompt_tokens", self.prompt.size() as i64));
            let _ = out.set(Item::new("n_prompt_tokens_processed", self.n_prompt_tokens_processed));
            let _ = out.set(Item::new("n_prompt_tokens_cache", self.n_prompt_tokens_cache));
            let _ = out.set(Item::new("params", self.params.to_json(only_metrics)));
            let _ = out.set(Item::new("next_token", Json::array_from(vec![Json::object_from(vec![
                Item::new("has_next_token", self.has_next_token),
                Item::new("has_new_line", self.has_new_line),
                Item::new("n_remain", self.n_remaining),
                Item::new("n_decoded", self.n_gen),
            ])])));
            if !only_metrics {
                let _ = out.set(Item::new("prompt", Json::array_from(self.prompt.get_text_tokens())));
                let _ = out.set(Item::new("generated", self.generated.clone()));
            }
        }
        out
    }
}

/// Tokenizer hooks used by `/tokenize` and `/detokenize`.
pub type TokenizeFn = Arc<dyn Fn(&str, bool, bool) -> Vec<i32> + Send + Sync>;
pub type DetokenizeFn = Arc<dyn Fn(&[i32], bool) -> String + Send + Sync>;

/// `server_context`.
pub struct ServerContext {
    meta: Mutex<Option<ServerContextMeta>>,
    state: Mutex<ServerState>,
    callback: Mutex<Option<StateCallback>>,
    slots: Mutex<Vec<ServerSlot>>,
    tokenizer: Mutex<Option<TokenizeFn>>,
    detokenizer: Mutex<Option<DetokenizeFn>>,
    pub queue_tasks: Arc<ServerQueue>,
    pub queue_results: Arc<ServerResponse>,
    role: ServerRoleType,
}

impl ServerContext {
    pub fn new(role: ServerRoleType) -> Self {
        Self {
            meta: Mutex::new(None),
            state: Mutex::new(ServerState::Downloading),
            callback: Mutex::new(None),
            slots: Mutex::new(Vec::new()),
            tokenizer: Mutex::new(None),
            detokenizer: Mutex::new(None),
            queue_tasks: Arc::new(ServerQueue::new()),
            queue_results: Arc::new(ServerResponse::new()),
            role,
        }
    }

    pub fn set_state_callback<F>(&self, callback: F)
    where
        F: Fn(ServerState, Json) + Send + Sync + 'static,
    {
        *self.callback.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(callback));
    }

    fn notify_state(&self, state: ServerState, payload: Json) {
        self.notify(state, payload);
    }

    fn notify(&self, state: ServerState, payload: Json) {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = state;
        if let Some(cb) = self.callback.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            cb(state, payload);
        }
    }

    /// `server_context::load_model()`, with metadata filled from CLI params.
    pub fn load_model(&self, params: &Params) -> bool {
        self.notify(ServerState::Loading, Json::object());
        let mut meta = ServerContextMeta::from_params(params);
        if meta.model_name.is_empty() && self.role == ServerRoleType::Router {
            self.notify(ServerState::Ready, Json::object());
            return true;
        }
        if params.model_alias.is_empty() && !meta.model_name.is_empty() {
            meta.model_aliases.insert(meta.model_name.clone());
        }
        let n_parallel = if params.n_parallel <= 0 { 1 } else { params.n_parallel };
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        slots.clear();
        for id in 0..n_parallel {
            slots.push(ServerSlot::new(id, meta.slot_n_ctx));
        }
        drop(slots);
        *self.meta.lock().unwrap_or_else(|e| e.into_inner()) = Some(meta);
        self.notify_state(ServerState::Ready, Json::object());
        true
    }

    /// `server_context::start_loop()`: block until `terminate()` is called.
    pub fn start_loop(&self, idle_sleep_ms: i64) {
        self.queue_tasks.start_loop(idle_sleep_ms);
    }

    /// `server_context::terminate()`.
    pub fn terminate(&self) {
        self.queue_tasks.terminate();
        self.queue_results.terminate();
    }

    pub fn get_meta(&self) -> Option<ServerContextMeta> {
        self.meta.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// No raw context in the std-only port.
    pub fn get_llama_context(&self) -> Option<()> {
        None
    }

    pub fn state(&self) -> ServerState {
        *self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Attach the model vocabulary's tokenizer, normally done by the loader.
    pub fn set_tokenizer<F, G>(&self, tokenize: F, detokenize: G)
    where
        F: Fn(&str, bool, bool) -> Vec<i32> + Send + Sync + 'static,
        G: Fn(&[i32], bool) -> String + Send + Sync + 'static,
    {
        *self.tokenizer.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(tokenize));
        *self.detokenizer.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(detokenize));
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        let guard = self.tokenizer.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(f) => Ok(f(text, add_special, parse_special)),
            None => Err(Error::new("the server has no tokenizer attached")),
        }
    }

    pub fn detokenize(&self, tokens: &[i32], special: bool) -> Result<String> {
        let guard = self.detokenizer.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(f) => Ok(f(tokens, special)),
            None => Err(Error::new("the server has no tokenizer attached")),
        }
    }

    /// `server_context::get_response_reader()`.
    pub fn get_response_reader(&self, polling_seconds: i32) -> crate::tools::server::queue::ResponseReader {
        crate::tools::server::queue::ResponseReader::new(
            Arc::clone(&self.queue_tasks),
            Arc::clone(&self.queue_results),
            polling_seconds,
        )
    }

    pub fn slots_json(&self, only_metrics: bool) -> Json {
        let slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        Json::array_from(slots.iter().map(|s| s.to_json(only_metrics)).collect::<Vec<_>>())
    }

    pub fn n_idle_slots(&self) -> i32 {
        self.slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|s| !s.is_processing)
            .count() as i32
    }

    /// Submit a task and return its assigned id.
    pub fn submit(&self, task: ServerTask, front: bool) -> Result<i32> {
        Ok(self.queue_tasks.post(task, front))
    }
}

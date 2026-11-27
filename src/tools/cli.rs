//! Rust port of `tools/cli` (`cli.cpp`, `cli-context.*`, `cli-client.*`, `cli-server.h`, `cli-ui.h`).
//!
//! The C++ tool talks to llama-server over HTTP. Std-only Rust has no HTTP
//! client, so `Transport` is the seam that supplies responses; SSE framing,
//! chat state and command handling stay faithful to the reference.

pub use crate::core::Token;
use crate::common::{glob_match, string_join, string_strip, Params};
use crate::core::{Error, Result};
use crate::json::{Item, Json};
use std::cell::Cell;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// `FILE_GLOB_MAX_RESULTS`.
pub const FILE_GLOB_MAX_RESULTS: usize = 100;
/// `CLI_HTTP_READ_TIMEOUT_SEC`.
pub const CLI_HTTP_READ_TIMEOUT_SEC: i64 = 3600;
/// `CLI_HTTP_MAX_ERROR_BODY`.
pub const CLI_HTTP_MAX_ERROR_BODY: usize = 1024 * 1024;

/// `LLAMA_ASCII_LOGO`.
pub const LLAMA_ASCII_LOGO: &str = r#"
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                     ██    ██
                                     ▀▀    ▀▀
"#;

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// `cli_context::interrupted()`.
pub fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

pub fn set_interrupted(value: bool) {
    INTERRUPTED.store(value, Ordering::SeqCst);
}

/// Signal handler equivalent: a second interrupt exits immediately.
pub fn handle_interrupt() {
    if interrupted() {
        print!("\x1b[0m\n");
        let _ = io::stdout().flush();
        std::process::exit(130);
    }
    set_interrupted(true);
}

fn should_stop() -> bool {
    interrupted()
}

// ---------------------------------------------------------------------------
// HTTP seam
// ---------------------------------------------------------------------------

/// Raw HTTP response, as returned by httplib in the C++ tool.
#[derive(Clone, Debug, Default)]
pub struct HttpResponse {
    pub status: u32,
    pub body: String,
}

/// Replaces `common_http_client` / httplib. No implementation ships here:
/// the caller injects one, keeping the chat loop testable without sockets.
pub trait Transport {
    fn get(&self, url: &str) -> Result<HttpResponse>;
    fn post(&self, url: &str, body: &str) -> Result<HttpResponse>;
    /// Feeds raw chunks into `on_chunk`; returning false cancels the stream.
    fn post_sse(
        &self,
        url: &str,
        body: &str,
        on_chunk: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<HttpResponse>;
}

fn join_path(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    let after_scheme = base.split_once("://").map(|(_, rest)| rest).unwrap_or(base);
    let prefix = match after_scheme.find('/') {
        Some(i) => &after_scheme[i..],
        None => "",
    };
    if prefix.is_empty() || prefix == "/" {
        path.to_string()
    } else {
        format!("{}{}", prefix.trim_end_matches('/'), path)
    }
}

fn url(base: &str, path: &str) -> String {
    format!("{}{}", base.trim_end_matches('/'), join_path(base, path))
}

/// Splits `data:` events out of an SSE byte stream, as `cli_client::post_sse`
/// does in the receiver callback.
pub struct SseDecoder<F: FnMut(&str)> {
    pending: String,
    on_data: F,
}

impl<F: FnMut(&str)> SseDecoder<F> {
    pub fn new(on_data: F) -> Self {
        Self { pending: String::new(), on_data }
    }

    pub fn feed(&mut self, chunk: &[u8]) {
        self.pending.push_str(&String::from_utf8_lossy(chunk));
        while let Some(pos) = self.pending.find('\n') {
            let line: String = self.pending.drain(..=pos).collect();
            let mut line = &line[..line.len() - 1];
            if line.ends_with('\r') {
                line = &line[..line.len() - 1];
            }
            if let Some(payload) = line.strip_prefix("data: ") {
                if payload != "[DONE]" {
                    (self.on_data)(payload);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// client / server state
// ---------------------------------------------------------------------------

/// `cli_client` data members.
#[derive(Clone, Debug, Default)]
pub struct CliClient {
    pub server_base: String,
    pub last_error: String,
    pub model: String,
}

impl CliClient {
    pub fn get(&self, t: &dyn Transport, path: &str) -> Result<String> {
        let with_model = if self.model.is_empty() {
            path.to_string()
        } else {
            format!("{path}?model={}", self.model)
        };
        let res = t.get(&url(&self.server_base, &with_model)).map_err(|e| {
            Error::new(format!("failed to connect to {}: {}", self.server_base, e))
        })?;
        if !(200..300).contains(&res.status) {
            return Err(Error::new(format!(
                "GET {path} failed with status {}: {}",
                res.status, res.body
            )));
        }
        Ok(res.body)
    }

    pub fn post(&self, t: &dyn Transport, path: &str, body: &str) -> Result<String> {
        let res = t.post(&url(&self.server_base, path), body).map_err(|e| {
            Error::new(format!("failed to connect to {}: {}", self.server_base, e))
        })?;
        if !(200..300).contains(&res.status) {
            return Err(Error::new(format!(
                "POST {path} failed with status {}: {}",
                res.status, res.body
            )));
        }
        Ok(res.body)
    }

    /// Returns the error body on failure, empty on success or cancellation.
    pub fn post_sse(
        &self,
        t: &dyn Transport,
        path: &str,
        body: &str,
        should_stop: &mut dyn FnMut() -> bool,
        on_data: &mut dyn FnMut(&str),
    ) -> String {
        let mut raw_body = String::new();
        let mut decoder = SseDecoder::new(&mut *on_data);
        let target = url(&self.server_base, path);
        let res = t.post_sse(&target, body, &mut |chunk: &[u8]| {
            if should_stop() {
                return false;
            }
            if raw_body.len() < CLI_HTTP_MAX_ERROR_BODY {
                let take = chunk.len().min(CLI_HTTP_MAX_ERROR_BODY - raw_body.len());
                raw_body.push_str(&String::from_utf8_lossy(&chunk[..take]));
            }
            decoder.feed(chunk);
            true
        });
        match res {
            Err(e) => {
                if should_stop() {
                    String::new()
                } else {
                    format!("failed to connect to {}: {}", self.server_base, e)
                }
            }
            Ok(resp) if !(200..300).contains(&resp.status) => {
                if raw_body.is_empty() {
                    format!("request failed with status {}", resp.status)
                } else {
                    raw_body
                }
            }
            Ok(_) => String::new(),
        }
    }

    pub fn wait_health(
        &mut self,
        t: &dyn Transport,
        is_aborted: &mut dyn FnMut() -> bool,
    ) -> Result<bool> {
        let mut attempts = 0;
        while !is_aborted() {
            match t.get(&url(&self.server_base, "/health")) {
                Ok(resp) if resp.status == 200 => return Ok(true),
                Ok(_) => {}
                Err(e) => {
                    attempts += 1;
                    if attempts >= 10 {
                        self.last_error =
                            format!("failed to connect to {}: {}", self.server_base, e);
                        return Ok(false);
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(300));
        }
        self.last_error = "aborted while waiting for the server to become ready".into();
        Ok(false)
    }
}

/// `cli_server` state. No thread is spawned: `stop` calls `terminate` directly.
#[derive(Clone, Debug, Default)]
pub struct CliServer {
    pub port: i32,
    pub is_alive: bool,
    pub is_stopping: bool,
}

impl CliServer {
    pub fn start(&mut self, port: i32) -> bool {
        if port <= 0 {
            return false;
        }
        self.port = port;
        self.is_alive = true;
        self.is_stopping = false;
        true
    }

    pub fn address(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn alive(&self) -> bool {
        self.is_alive
    }

    pub fn wait_ready(
        &self,
        t: &dyn Transport,
        should_stop: &mut dyn FnMut() -> bool,
    ) -> bool {
        if !self.alive() {
            return false;
        }
        while !should_stop() {
            let health = format!("{}/health", self.address());
            if let Ok(resp) = t.get(&health) {
                if resp.status == 200 {
                    return true;
                }
            }
            if !self.alive() {
                return false;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        true
    }

    pub fn stop<F: FnOnce()>(&mut self, terminate: F) {
        if self.is_stopping {
            return;
        }
        self.is_stopping = true;
        if self.is_alive {
            terminate();
        }
        self.is_alive = false;
    }
}

/// `cli_timings`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CliTimings {
    pub prompt_per_second: f64,
    pub predicted_per_second: f64,
}

/// `cli_context::generated_content`.
#[derive(Clone, Debug, Default)]
pub struct GeneratedContent {
    pub reasoning: String,
    pub content: String,
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayType {
    Reset,
    Info,
    Prompt,
    Reasoning,
    UserInput,
    Error,
}

fn ansi(display: DisplayType) -> &'static str {
    match display {
        DisplayType::Reset => "\x1b[0m",
        DisplayType::Info => "\x1b[35m",
        DisplayType::Prompt => "\x1b[33m",
        DisplayType::Reasoning => "\x1b[90m",
        DisplayType::UserInput => "\x1b[1;32m",
        DisplayType::Error => "\x1b[1;31m",
    }
}

/// Std-only stand-in for `console`. Cell keeps display state settable while a
/// turn is borrowed, as in the C++ code.
pub struct Ui {
    use_color: bool,
    display: Cell<DisplayType>,
}

impl Ui {
    pub fn new(use_color: bool) -> Self {
        Self { use_color, display: Cell::new(DisplayType::Reset) }
    }

    pub fn set_display(&self, display: DisplayType) {
        if self.use_color && self.display.get() != display {
            print!("{}", ansi(display));
            let _ = io::stdout().flush();
        }
        self.display.set(display);
    }

    pub fn log(&self, text: &str) {
        print!("{text}");
    }

    pub fn flush(&self) {
        let _ = io::stdout().flush();
    }

    pub fn error(&self, text: &str) {
        self.set_display(DisplayType::Error);
        eprint!("{text}");
        self.set_display(DisplayType::Reset);
    }

    pub fn show_error(&self, title: &str, message: &str) {
        self.error(&format!("Error: {title}\n"));
        if !message.is_empty() {
            self.log(&format!("{message}\n"));
        }
    }

    pub fn show_message(&self, message: &str) {
        self.log(&format!("{message}\n"));
    }

    pub fn show_info(&self, message: &str) {
        self.set_display(DisplayType::Info);
        self.log(&format!("{message}\n"));
        self.set_display(DisplayType::Reset);
    }

    /// `console::readline` for `simple_io`: '/' submits, '\\' flips multiline.
    fn readline(multiline: bool) -> (String, bool) {
        let mut line = String::new();
        if io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
            return (String::new(), false);
        }
        while line.ends_with('\n') || line.ends_with('\r') {
            line.pop();
        }
        let mut ml = multiline;
        let more = match line.chars().last() {
            Some('/') => {
                line.pop();
                line.push('\n');
                false
            }
            Some('\\') => {
                line.pop();
                ml = !ml;
                line.push('\n');
                ml
            }
            _ => {
                line.push('\n');
                ml
            }
        };
        (line, more)
    }

    pub fn read_input(&self, multiline: bool, prompt: Option<&str>) -> String {
        match prompt {
            Some(p) => self.log(p),
            None => self.log("\n> "),
        }
        self.flush();
        let mut buffer = String::new();
        loop {
            let (line, more) = Self::readline(multiline);
            buffer.push_str(&line);
            if !more {
                break;
            }
        }
        buffer
    }
}

/// `ui::spinner`. The C++ version animates in a thread; here it only prints
/// its message, so no thread or terminal control is needed.
pub struct Spinner;

impl Spinner {
    pub fn new(ui: &Ui, message: &str) -> Self {
        if !message.is_empty() {
            ui.log(&format!("{message} "));
            ui.flush();
        }
        Spinner
    }
}

/// `ui::user_turn`.
pub struct UserTurn<'a> {
    ui: &'a Ui,
}

impl<'a> UserTurn<'a> {
    pub fn new(ui: &'a Ui) -> Self {
        ui.set_display(DisplayType::UserInput);
        Self { ui }
    }

    pub fn echo(&self, buffer: &str) {
        if buffer.len() > 500 {
            let head = String::from_utf8_lossy(&buffer.as_bytes()[..500]);
            self.ui.log(&format!("\n{head} ... (truncated)\n"));
        } else {
            self.ui.log(&format!("\n{buffer}\n"));
        }
    }

    pub fn read_input(&self, multiline: bool, prompt: Option<&str>) -> String {
        self.ui.read_input(multiline, prompt)
    }
}

impl Drop for UserTurn<'_> {
    fn drop(&mut self) {
        self.ui.set_display(DisplayType::Reset);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssistantDisplayMode {
    Reasoning,
    Content,
}

/// `ui::assistant_turn`.
pub struct AssistantTurn<'a> {
    ui: &'a Ui,
    mode: AssistantDisplayMode,
    trailing_newline: bool,
    is_inside_reasoning: bool,
}

impl<'a> AssistantTurn<'a> {
    pub fn new(ui: &'a Ui) -> Self {
        ui.set_display(DisplayType::Reset);
        Self {
            ui,
            mode: AssistantDisplayMode::Content,
            trailing_newline: true,
            is_inside_reasoning: false,
        }
    }

    pub fn push(&mut self, mode: AssistantDisplayMode, buffer: &str) {
        if mode != self.mode {
            self.add_newline_if_needed();
            match mode {
                AssistantDisplayMode::Content => {
                    if self.is_inside_reasoning {
                        self.ui.log("[End thinking]\n\n");
                        self.is_inside_reasoning = false;
                    }
                    self.ui.set_display(DisplayType::Reset);
                }
                AssistantDisplayMode::Reasoning => {
                    self.ui.set_display(DisplayType::Reasoning);
                    self.is_inside_reasoning = true;
                    self.ui.log("\n[Start thinking]\n\n");
                }
            }
        }
        self.mode = mode;
        if buffer.is_empty() {
            return;
        }
        self.trailing_newline = buffer.ends_with('\n');
        self.ui.log(buffer);
        self.ui.flush();
    }

    fn add_newline_if_needed(&mut self) {
        if !self.trailing_newline {
            self.ui.log("\n");
            self.ui.flush();
        }
        self.trailing_newline = true;
    }
}

impl Drop for AssistantTurn<'_> {
    fn drop(&mut self) {
        self.ui.set_display(DisplayType::Reset);
        self.add_newline_if_needed();
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn format_error_message(err: &Json) -> String {
    if let Ok(e) = err.at("error") {
        if e.is_object() {
            if let Ok(v) = e.at("message") {
                if v.is_string() {
                    return v.get::<String>().unwrap_or_else(|_| err.dump(-1).unwrap_or_default());
                }
            }
        }
    }
    err.dump(-1).unwrap_or_default()
}

fn format_error_message_text(err: &str) -> String {
    let parsed = Json::parse_no_throw(err);
    if parsed.is_discarded() {
        err.to_string()
    } else {
        format_error_message(&parsed)
    }
}

pub fn media_type_from_ext(fname: &str) -> &'static str {
    let ext = Path::new(fname)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "wav" | "mp3" => "audio",
        "mp4" | "avi" | "mkv" | "mov" | "webm" => "video",
        _ => "image",
    }
}

fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { T[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    out
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                out.push(path);
            }
        }
    }
}

/// Result of processing one submitted line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandOutcome {
    Exit,
    Continue,
    Generate { add_user_msg: bool },
}

// ---------------------------------------------------------------------------
// cli_context
// ---------------------------------------------------------------------------

/// Port of `cli_context` plus its private `cli_context_impl`.
pub struct CliContext {
    pub params: Params,
    pub client: CliClient,
    pub server: Option<CliServer>,
    pub model_name: String,
    pub model_ftype: String,
    pub build_info: String,
    pub has_vision: bool,
    pub has_audio: bool,
    pub has_video: bool,
    pub output_file: Option<File>,
    pub messages: Json,
    pub pending_media: Json,
    pub ui: Ui,
}

impl CliContext {
    pub fn new(params: Params) -> Self {
        let ui = Ui::new(params.use_color);
        Self {
            params,
            client: CliClient::default(),
            server: None,
            model_name: String::new(),
            model_ftype: String::new(),
            build_info: String::new(),
            has_vision: false,
            has_audio: false,
            has_video: false,
            output_file: None,
            messages: Json::array(),
            pending_media: Json::array(),
            ui,
        }
    }

    /// `cli_context::init`; `port` replaces `common_http_get_free_port()`.
    pub fn init(&mut self, t: &dyn Transport, port: i32) -> bool {
        self.ui = Ui::new(self.params.use_color);

        let use_external = !self.params.server_base.is_empty();
        let mut spinner = if use_external {
            self.client.server_base = self.params.server_base.trim_end_matches('/').to_string();
            let message = format!("Connecting to server at {}", self.client.server_base);
            Some(Spinner::new(&self.ui, &message))
        } else {
            if self.params.model.path.is_empty()
                && self.params.model.url.is_empty()
                && self.params.model.hf_repo.is_empty()
                && self.params.model.docker_repo.is_empty()
            {
                self.ui.show_error(
                    "no model specified",
                    "use -m <file.gguf> or -hf <user/repo> to run a local model,\n\
                     or --server-base <url> to connect to a running llama-server",
                );
                return false;
            }
            let spinner = Some(Spinner::new(&self.ui, "\n\nLoading model..."));
            let mut server = CliServer::default();
            if !server.start(port) {
                self.ui.show_error("server start failed", "");
                return false;
            }
            let mut stop = || should_stop();
            if !server.wait_ready(t, &mut stop) {
                if !should_stop() {
                    self.ui.show_error("the server exited before becoming ready", "");
                }
                return false;
            }
            self.client.server_base = server.address();
            self.server = Some(server);
            spinner
        };

        // No local server thread exists, so its liveness cannot change here.
        let server_died = self.server.as_ref().map(|s| !s.alive()).unwrap_or(false);
        let mut aborted = || should_stop() || server_died;
        let healthy = match self.client.wait_health(t, &mut aborted) {
            Ok(v) => v,
            Err(e) => {
                self.client.last_error = e.message;
                false
            }
        };
        if !healthy {
            if !should_stop() {
                self.ui.show_error(&self.client.last_error.clone(), "");
            }
            return false;
        }

        if use_external {
            spinner.take();
            match self.list_and_ask_models(t) {
                Ok(true) => {}
                Ok(false) => return false,
                Err(e) => {
                    self.ui.show_error(&e.message, "");
                    self.ui.show_message(
                        "This might be caused by an incorrect server-base endpoint URL",
                    );
                    return false;
                }
            }
            spinner = Some(Spinner::new(&self.ui, "Waiting for server..."));
        }

        self.fetch_server_props(t);

        if !self.params.out_file.is_empty() {
            match OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&self.params.out_file)
            {
                Ok(file) => self.output_file = Some(file),
                Err(_) => {
                    self.ui.show_error(
                        &format!("failed to open output file '{}'", self.params.out_file),
                        "",
                    );
                    return false;
                }
            }
        }

        let _ = &spinner;
        true
    }

    pub fn fetch_server_props(&mut self, t: &dyn Transport) {
        let Ok(body) = self.client.get(t, "/props") else { return };
        let Ok(props) = Json::parse(&body) else { return };
        self.model_name = props.value_str("model_alias", "").unwrap_or_default();
        if self.model_name.is_empty() {
            let path = props.value_str("model_path", "").unwrap_or_default();
            if !path.is_empty() {
                self.model_name = Path::new(&path)
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
            }
        }
        self.model_ftype = props.value_str("model_ftype", "").unwrap_or_default();
        self.build_info = props.value_str("build_info", "").unwrap_or_default();
        if let Ok(modalities) = props.at("modalities") {
            if modalities.is_object() {
                self.has_vision = modalities.value("vision", false).unwrap_or(false);
                self.has_audio = modalities.value("audio", false).unwrap_or(false);
                self.has_video = modalities.value("video", false).unwrap_or(false);
            }
        }
    }

    pub fn list_and_ask_models(&mut self, t: &dyn Transport) -> Result<bool> {
        let body = self.client.get(t, "/v1/models")?;
        let resp = Json::parse(&body)?;
        let data = resp.at("data")?;
        if !data.is_array() {
            return Err(Error::new("invalid response from /v1/models"));
        }
        let mut models = Vec::new();
        let mut displays = Vec::new();
        for m in data.iter() {
            if !m.contains("id") || !m.at("id").map(|v| v.is_string()).unwrap_or(false) {
                continue;
            }
            let name = m.value_str("id", "")?;
            let mut display = name.clone();
            if let Ok(aliases) = m.at("aliases") {
                if aliases.is_array() {
                    let list: Vec<String> = aliases
                        .iter()
                        .filter(|v| v.is_string())
                        .filter_map(|v| v.get::<String>().ok())
                        .collect();
                    if !list.is_empty() {
                        display.push_str(&format!(" ({})", string_join(&list, ", ")));
                    }
                }
            }
            models.push(name);
            displays.push(display);
        }

        if models.len() == 1 {
            self.model_name = models[0].clone();
            self.client.model = self.model_name.clone();
            return Ok(true);
        }

        let mut message = "\nAvailable models:".to_string();
        for (i, display) in displays.iter().enumerate() {
            message.push_str(&format!("\n  {}. {}", i + 1, display));
        }
        message.push('\n');
        self.ui.show_message(&message);

        loop {
            if should_stop() {
                return Ok(false);
            }
            let input = {
                let turn = UserTurn::new(&self.ui);
                turn.read_input(false, Some("Select model by number: "))
            };
            let selection = string_strip(&input);
            if selection.is_empty() {
                return Ok(false);
            }
            let digits: String = selection.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(idx) = digits.parse::<usize>() {
                if idx > 0 && idx <= models.len() {
                    self.model_name = models[idx - 1].clone();
                    self.client.model = self.model_name.clone();
                    self.ui.show_message(&format!("Selected model: {}", self.model_name));
                    return Ok(true);
                }
            }
            self.ui.show_error("Invalid selection. Please enter a valid number.", "");
        }
    }

    pub fn add_system_prompt(&mut self) {
        if !self.params.system_prompt.is_empty() {
            let _ = self.messages.push_back_items([
                Item::new("role", "system"),
                Item::new("content", self.params.system_prompt.clone()),
            ]);
        }
    }

    pub fn push_user_message(&mut self, text: &str) {
        let content = if self.pending_media.empty() {
            Json::String(text.to_string())
        } else {
            let mut content = self.pending_media.clone();
            let _ = content.push_back_items([
                Item::new("type", "text"),
                Item::new("text", text.to_string()),
            ]);
            self.pending_media = Json::array();
            content
        };
        let _ = self.messages.push_back_items([
            Item::new("role", "user"),
            Item::new("content", content),
        ]);
    }

    /// Reads and stages a media file; `type` is "image", "audio" or "video".
    pub fn stage_media_file(&mut self, fname: &str, type_: &str) -> bool {
        let Ok(data) = fs::read(fname) else { return false };
        let encoded = base64_encode(&data);
        let value = match type_ {
            "audio" => {
                let ext = Path::new(fname)
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let format = if ext == "mp3" { "mp3" } else { "wav" };
                Json::object_from([
                    Item::new("type", "input_audio"),
                    Item::new(
                        "input_audio",
                        Json::object_from([
                            Item::new("data", encoded),
                            Item::new("format", format),
                        ]),
                    ),
                ])
            }
            "video" => Json::object_from([
                Item::new("type", "input_video"),
                Item::new(
                    "input_video",
                    Json::object_from([Item::new("data", encoded)]),
                ),
            ]),
            _ => Json::object_from([
                Item::new("type", "image_url"),
                Item::new(
                    "image_url",
                    Json::object_from([Item::new(
                        "url",
                        format!("data:image/unknown;base64,{encoded}"),
                    )]),
                ),
            ]),
        };
        let _ = self.pending_media.push_back(value.into());
        true
    }

    pub fn write_output_file(&mut self, content: &str) {
        if let Some(file) = self.output_file.as_mut() {
            let _ = file.write_all(content.as_bytes());
            let _ = file.flush();
        }
    }

    pub fn generate_completion(
        &mut self,
        t: &dyn Transport,
        content_out: &mut GeneratedContent,
        timings: &mut CliTimings,
    ) -> bool {
        let mut body = Json::object_from([
            Item::new("messages", self.messages.clone()),
            Item::new("stream", true),
            Item::new("timings_per_token", true),
        ]);
        if !self.client.model.is_empty() {
            let _ = body.set(Item::new("model", self.client.model.clone()));
        }

        let mut stream_error = false;
        let mut assistant = AssistantTurn::new(&self.ui);
        let mut stop = || should_stop();
        let mut on_data = |payload: &str| {
            let chunk = Json::parse_no_throw(payload);
            if chunk.is_discarded() {
                return;
            }
            if chunk.contains("error") {
                stream_error = true;
                self.ui.show_error(&format_error_message(&chunk), "");
                return;
            }
            if let Ok(timing) = chunk.at("timings") {
                timings.prompt_per_second = timing.value("prompt_per_second", 0.0).unwrap_or(0.0);
                timings.predicted_per_second =
                    timing.value("predicted_per_second", 0.0).unwrap_or(0.0);
            }
            let Ok(choices) = chunk.at("choices") else { return };
            if !choices.is_array() || choices.empty() {
                return;
            }
            let Ok(choice) = choices.at_index(0) else { return };
            if !choice.contains("delta") {
                return;
            }
            let Ok(delta) = choice.at("delta") else { return };
            if delta
                .at("reasoning_content")
                .map(|v| v.is_string())
                .unwrap_or(false)
            {
                let text = delta.value_str("reasoning_content", "").unwrap_or_default();
                if !text.is_empty() {
                    content_out.reasoning.push_str(&text);
                    assistant.push(AssistantDisplayMode::Reasoning, &text);
                }
            }
            if delta.at("content").map(|v| v.is_string()).unwrap_or(false) {
                let text = delta.value_str("content", "").unwrap_or_default();
                if !text.is_empty() {
                    content_out.content.push_str(&text);
                    assistant.push(AssistantDisplayMode::Content, &text);
                }
            }
        };

        let err = self.client.post_sse(
            t,
            "/v1/chat/completions",
            &body.dump_safe(-1),
            &mut stop,
            &mut on_data,
        );
        drop(assistant);
        set_interrupted(false);
        if !err.is_empty() {
            self.ui.show_error(&format_error_message_text(&err), "");
            return false;
        }
        !stream_error
    }

    fn add_text_file(&self, fname: &str, cur_msg: &mut String) -> bool {
        let Ok(data) = fs::read(fname) else {
            self.ui.show_error(
                &format!("file does not exist or cannot be opened: '{fname}'"),
                "",
            );
            return false;
        };
        cur_msg.push_str("--- File: ");
        cur_msg.push_str(fname);
        cur_msg.push_str(" ---\n");
        cur_msg.push_str(&String::from_utf8_lossy(&data));
        self.ui.show_message(&format!("Loaded text from '{fname}'"));
        true
    }

    fn glob_files(&mut self, pattern: &str, cur_msg: &mut String) {
        let mut count = 0;
        let mut curdir = std::env::current_dir().unwrap_or_default();
        let mut pattern = string_strip(pattern).to_string();
        if let Some(start) = pattern.find(|c| matches!(c, '!' | '[' | '*' | '?')) {
            if start != 0 {
                if let Some(endpath) = pattern[..start].rfind('/') {
                    let mut rel_pattern = pattern[..endpath].to_string();
                    if rel_pattern.starts_with('~') {
                        if let Some(home) = std::env::var_os("HOME") {
                            if !home.is_empty() {
                                rel_pattern =
                                    format!("{}{}", home.to_string_lossy(), &rel_pattern[1..]);
                            }
                        }
                    }
                    curdir = curdir.join(rel_pattern);
                    pattern = pattern[endpath + 1..].to_string();
                }
            }
        }

        let mut files = Vec::new();
        collect_files(&curdir, &mut files);
        for full in files {
            let rel = full
                .strip_prefix(&curdir)
                .unwrap_or(&full)
                .to_string_lossy()
                .replace('\\', "/");
            if !glob_match(&pattern, &rel) {
                continue;
            }
            let full_str = full.to_string_lossy().into_owned();
            if !self.add_text_file(&full_str, cur_msg) {
                continue;
            }
            self.write_output_file(&format!("User: Added text file: {full_str}\n"));
            count += 1;
            if count >= FILE_GLOB_MAX_RESULTS {
                self.ui.show_error(
                    &format!("Maximum number of globbed files allowed ({FILE_GLOB_MAX_RESULTS}) reached."),
                    "",
                );
                break;
            }
        }
    }

    /// Command dispatcher shared by the interactive and scripted prompt paths.
    pub fn handle_command(&mut self, buffer: &str, cur_msg: &mut String) -> CommandOutcome {
        if buffer.starts_with("/exit") {
            CommandOutcome::Exit
        } else if buffer.starts_with("/regen") {
            if self.messages.size() >= 2 {
                let last = self.messages.size() - 1;
                let _ = self.messages.erase_index(last);
                CommandOutcome::Generate { add_user_msg: false }
            } else {
                self.ui.show_error("No message to regenerate.", "");
                CommandOutcome::Continue
            }
        } else if buffer.starts_with("/clear") {
            self.messages = Json::array();
            self.add_system_prompt();
            self.pending_media = Json::array();
            self.ui.show_message("Chat history cleared.");
            CommandOutcome::Continue
        } else if (buffer.starts_with("/image ") && self.has_vision)
            || (buffer.starts_with("/audio ") && self.has_audio)
            || (buffer.starts_with("/video ") && self.has_video)
        {
            let type_ = &buffer[1..6];
            let fname = string_strip(&buffer[7..]);
            if !self.stage_media_file(fname, type_) {
                self.ui.show_error(
                    &format!("file does not exist or cannot be opened: '{fname}'"),
                    "",
                );
                return CommandOutcome::Continue;
            }
            self.ui.show_message(&format!("Loaded media from '{fname}'"));
            self.write_output_file(&format!("User: Added media: {fname}\n"));
            CommandOutcome::Continue
        } else if buffer.starts_with("/read ") {
            let fname = string_strip(&buffer[6..]).to_string();
            self.add_text_file(&fname, cur_msg);
            self.write_output_file(&format!("User: Added text file: {fname}\n"));
            CommandOutcome::Continue
        } else if buffer.starts_with("/glob ") {
            self.glob_files(&buffer[6..], cur_msg);
            CommandOutcome::Continue
        } else {
            cur_msg.push_str(buffer);
            CommandOutcome::Generate { add_user_msg: true }
        }
    }

    fn read_user_input(&mut self) -> String {
        if self.params.prompt.is_empty() {
            let turn = UserTurn::new(&self.ui);
            turn.read_input(self.params.multiline_input, None)
        } else {
            let prompt = self.params.prompt.clone();
            for fname in self.params.image.clone() {
                let type_ = media_type_from_ext(&fname);
                if !self.stage_media_file(&fname, type_) {
                    self.ui.show_error(
                        &format!("file does not exist or cannot be opened: '{fname}'"),
                        "",
                    );
                    break;
                }
                self.ui.show_message(&format!("Loaded media from '{fname}'"));
            }
            let turn = UserTurn::new(&self.ui);
            turn.echo(&prompt);
            self.params.prompt.clear();
            prompt
        }
    }

    /// `cli_context::run`; returns the process exit code.
    pub fn run(&mut self, t: &dyn Transport) -> i32 {
        self.add_system_prompt();

        let mut modalities = "text".to_string();
        if self.has_vision {
            modalities.push_str(", vision");
        }
        if self.has_audio {
            modalities.push_str(", audio");
        }
        if self.has_video {
            modalities.push_str(", video");
        }

        let mut banner = String::new();
        banner.push('\n');
        banner.push_str(LLAMA_ASCII_LOGO);
        banner.push('\n');
        banner.push_str(&format!("build      : {}\n", self.build_info));
        banner.push_str(&format!("model      : {}\n", self.model_name));
        if !self.model_ftype.is_empty() {
            banner.push_str(&format!("ftype      : {}\n", self.model_ftype));
        }
        banner.push_str(&format!("modalities : {modalities}\n"));
        if !self.params.system_prompt.is_empty() {
            banner.push_str("using custom system prompt\n");
        }
        banner.push('\n');
        banner.push_str("available commands:\n");
        banner.push_str("  /exit or Ctrl+C     stop or exit\n");
        banner.push_str("  /regen              regenerate the last response\n");
        banner.push_str("  /clear              clear the chat history\n");
        banner.push_str("  /read <file>        add a text file\n");
        banner.push_str("  /glob <pattern>     add text files using globbing pattern\n");
        if self.has_vision {
            banner.push_str("  /image <file>       add an image file\n");
        }
        if self.has_audio {
            banner.push_str("  /audio <file>       add an audio file\n");
        }
        if self.has_video {
            banner.push_str("  /video <file>       add a video file\n");
        }
        banner.push('\n');
        self.ui.show_message(&banner);

        let mut cur_msg = String::new();
        loop {
            let mut buffer = self.read_user_input();
            if should_stop() {
                set_interrupted(false);
                break;
            }
            if buffer.is_empty() {
                break;
            }
            if buffer.ends_with('\n') {
                buffer.pop();
            }
            if buffer.is_empty() {
                continue;
            }

            match self.handle_command(&buffer, &mut cur_msg) {
                CommandOutcome::Exit => break,
                CommandOutcome::Continue => continue,
                CommandOutcome::Generate { add_user_msg } => {
                    if add_user_msg {
                        self.push_user_message(&cur_msg);
                        self.write_output_file(&format!("User:\n{cur_msg}\n\n"));
                        cur_msg.clear();
                    }

                    let mut timings = CliTimings::default();
                    let mut content = GeneratedContent::default();
                    self.generate_completion(t, &mut content, &mut timings);

                    let mut assistant = Json::object_from([
                        Item::new("role", "assistant"),
                        Item::new("content", content.content.clone()),
                    ]);
                    if !content.reasoning.is_empty() {
                        let _ = assistant.set(Item::new("reasoning_content", content.reasoning.clone()));
                    }
                    let _ = self.messages.push_back(assistant.into());

                    if self.output_file.is_some() {
                        let mut out = "Assistant:\n".to_string();
                        if !content.reasoning.is_empty() {
                            out.push_str("[Start thinking]\n\n");
                            out.push_str(&content.reasoning);
                            out.push_str("[End thinking]\n\n");
                        }
                        out.push_str(&content.content);
                        if !out.ends_with('\n') {
                            out.push('\n');
                        }
                        out.push('\n');
                        self.write_output_file(&out);
                    }

                    if self.params.show_timings {
                        self.ui.show_info(&format!(
                            "\n[ Prompt: {:.1} t/s | Generation: {:.1} t/s ]",
                            timings.prompt_per_second, timings.predicted_per_second
                        ));
                    }
                    if self.params.single_turn {
                        break;
                    }
                }
            }
        }

        self.ui.show_message("\n\nExiting...");
        0
    }

    pub fn shutdown<F: FnOnce()>(&mut self, terminate: F) {
        if let Some(server) = self.server.as_mut() {
            server.stop(terminate);
        }
        self.server = None;
        self.output_file = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_type_from_extension() {
        assert_eq!(media_type_from_ext("clip.wav"), "audio");
        assert_eq!(media_type_from_ext("clip.MP4"), "video");
        assert_eq!(media_type_from_ext("shot.png"), "image");
    }

    #[test]
    fn error_message_from_json_or_text() {
        let json = Json::parse(r#"{"error":{"message":"bad request"}}"#).unwrap();
        assert_eq!(format_error_message(&json), "bad request");
        assert_eq!(format_error_message_text("plain failure"), "plain failure");
    }

    #[test]
    fn sse_decoder_emits_data_events() {
        let mut events = Vec::new();
        let mut decoder = SseDecoder::new(|payload: &str| events.push(payload.to_string()));
        decoder.feed(b"data: one\r\n\ndata: [DONE]\n\ndata: two\n");
        assert_eq!(events, vec!["one", "two"]);
    }

    #[test]
    fn commands_drive_chat_state() {
        let mut ctx = CliContext::new(Params::default());
        let mut current = String::new();
        assert_eq!(ctx.handle_command("/clear", &mut current), CommandOutcome::Continue);
        assert!(ctx.messages.empty());
        ctx.push_user_message("hello");
        let _ = ctx.messages.push_back_items([
            Item::new("role", "assistant"),
            Item::new("content", "hi"),
        ]);
        assert_eq!(
            ctx.handle_command("/regen", &mut current),
            CommandOutcome::Generate { add_user_msg: false }
        );
        assert_eq!(ctx.messages.size(), 1);
        assert_eq!(
            ctx.handle_command("follow up", &mut current),
            CommandOutcome::Generate { add_user_msg: true }
        );
        assert_eq!(current, "follow up");
        assert_eq!(ctx.handle_command("/exit", &mut current), CommandOutcome::Exit);
    }
}

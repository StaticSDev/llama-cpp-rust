//! Port of `tools/server/server-http.{h,cpp}`.
//!
//! Request/response types, route registration with `:path_params`, query and
//! multipart parsing, CORS and API-key middlewares, and a std-only HTTP/1.1
//! transport built on `TcpListener`. Streaming responses use chunked encoding.

use crate::core::Result;
use crate::json::{Item, Json};
use crate::tools::server::common::{error_body, safe_json_to_str, ErrorType};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use super::common::parse_csv_row;

/// `raw_buffer`.
pub type RawBuffer = Vec<u8>;

/// `uploaded_file`.
#[derive(Clone, Debug, Default)]
pub struct UploadedFile {
    pub data: RawBuffer,
    pub filename: String,
    pub content_type: String,
}

/// `server_http_req`.
pub struct HttpRequest {
    pub params: BTreeMap<String, String>,
    pub headers: BTreeMap<String, String>,
    pub path: String,
    pub query_string: String,
    pub body: String,
    pub files: BTreeMap<String, UploadedFile>,
    pub should_stop: Arc<dyn Fn() -> bool + Send + Sync>,
}

impl Default for HttpRequest {
    fn default() -> Self {
        Self {
            params: BTreeMap::new(),
            headers: BTreeMap::new(),
            path: String::new(),
            query_string: String::new(),
            body: String::new(),
            files: BTreeMap::new(),
            should_stop: Arc::new(|| false),
        }
    }
}

impl HttpRequest {
    /// `server_http_req::get_param()`.
    pub fn get_param(&self, key: &str) -> &str {
        self.params.get(key).map(String::as_str).unwrap_or("")
    }

    /// Header lookup, case insensitive.
    pub fn get_header(&self, key: &str) -> &str {
        let lower = key.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(&lower))
            .map(|(_, v)| v.as_str())
            .unwrap_or("")
    }

    pub fn has_header(&self, key: &str) -> bool {
        self.headers.keys().any(|k| k.eq_ignore_ascii_case(key))
    }
}

/// Chunk generator of a streaming response.
pub type NextChunk = Arc<Mutex<dyn FnMut(&mut String) -> bool + Send>>;

/// `server_http_res`.
pub struct HttpResponse {
    pub content_type: String,
    pub status: i32,
    pub data: String,
    pub headers: BTreeMap<String, String>,
    pub next: Option<NextChunk>,
    pub on_complete: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl Default for HttpResponse {
    fn default() -> Self {
        Self {
            content_type: "application/json; charset=utf-8".to_string(),
            status: 200,
            data: String::new(),
            headers: BTreeMap::new(),
            next: None,
            on_complete: None,
        }
    }
}

impl HttpResponse {
    pub fn ok(data: &Json) -> Self {
        Self {
            status: 200,
            data: safe_json_to_str(data),
            ..Self::default()
        }
    }

    /// `server_res_generator::error()`.
    pub fn error(error_data: &Json) -> Self {
        let status = error_data.value("code", 500i32).unwrap_or(500);
        Self {
            status,
            data: safe_json_to_str(&Json::object_from(vec![Item::new("error", error_data.clone())])),
            ..Self::default()
        }
    }

    pub fn error_message(message: impl Into<String>, kind: ErrorType) -> Self {
        Self::error(&crate::tools::server::common::format_error_response(message, kind))
    }

    pub fn is_stream(&self) -> bool {
        self.next.is_some()
    }

    /// Install a chunk provider. The closure returns `true` while more data may follow.
    pub fn set_next<F>(&mut self, provider: F)
    where
        F: FnMut(&mut String) -> bool + Send + 'static,
    {
        self.next = Some(Arc::new(Mutex::new(provider)));
    }

    /// Stream a JSON value as one or more `data:` SSE events.
    pub fn set_sse(&mut self, events: Vec<Json>, framing: SseFraming) {
        let mut iter = events.into_iter();
        self.content_type = "text/event-stream".to_string();
        self.headers.insert("Cache-Control".to_string(), "no-cache".to_string());
        self.set_next(move |chunk| {
            match iter.next() {
                Some(event) => {
                    *chunk = match framing {
                        SseFraming::Oai => crate::tools::server::common::format_oai_sse(&event),
                        SseFraming::OaiResp => crate::tools::server::common::format_oai_resp_sse(&event),
                        SseFraming::Anthropic => crate::tools::server::common::format_anthropic_sse(&event),
                    };
                    true
                }
                None => false,
            }
        });
    }

    pub fn complete(&self) {
        if let Some(cb) = &self.on_complete {
            cb();
        }
    }
}

/// SSE event framing for `/v1/responses` and `/v1/messages`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SseFraming {
    Oai,
    OaiResp,
    Anthropic,
}

/// `server_http_context::handler_t`.
pub type Handler = Arc<dyn Fn(&HttpRequest) -> HttpResponse + Send + Sync>;

/// `server_http_context::ws_conn`.
pub trait WsConn {
    /// 1 = text, 2 = binary, 0 = closed.
    fn read(&mut self, msg: &mut String) -> i32;
    fn send(&self, msg: &str) -> bool;
    fn send_binary(&self, msg: &[u8]) -> bool;
    fn is_open(&self) -> bool;
    fn close(&self);
}

/// `server_http_context::ws_handler_t`.
pub type WsHandler = Arc<dyn Fn(&HttpRequest, &mut dyn WsConn) + Send + Sync>;

/// HTTP method of a registered route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
    Delete,
    Options,
    Ws,
}

impl Method {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Delete => "DELETE",
            Self::Options => "OPTIONS",
            Self::Ws => "GET",
        }
    }
}

struct Route {
    method: Method,
    pattern: String,
    handler: Option<Handler>,
    ws_handler: Option<WsHandler>,
}

/// HTTP settings from the parsed CLI parameters.
#[derive(Clone, Debug)]
pub struct ServerHttpParams {
    pub path_prefix: String,
    pub port: i32,
    pub hostnames: Vec<String>,
    pub api_keys: Vec<String>,
    pub cors_origins: String,
    pub cors_methods: String,
    pub cors_headers: String,
    pub cors_credentials: bool,
    pub public_path: String,
    pub timeout_read: i32,
    pub reuse_port: bool,
    pub is_ssl: bool,
    pub managed_by_router: bool,
}

impl ServerHttpParams {
    pub fn from_params(params: &crate::common::Params) -> Self {
        Self {
            path_prefix: params.api_prefix.clone(),
            port: params.port,
            hostnames: params.hostnames.clone(),
            api_keys: params.api_keys.clone(),
            cors_origins: params.cors_origins.clone(),
            cors_methods: params.cors_methods.clone(),
            cors_headers: params.cors_headers.clone(),
            cors_credentials: params.cors_credentials,
            public_path: params.public_path.clone(),
            timeout_read: params.timeout_read,
            reuse_port: params.reuse_port,
            is_ssl: !params.ssl_file_key.is_empty() && !params.ssl_file_cert.is_empty(),
            managed_by_router: std::env::var_os(super::common::SERVER_ENV_ROUTER_PORT).is_some(),
        }
    }
}

/// `server_http_context`.
pub struct HttpContext {
    inner: Arc<HttpInner>,
    threads: Vec<JoinHandle<()>>,
    listening_addresses: Vec<String>,
}

struct HttpInner {
    settings: ServerHttpParams,
    routes: Arc<Mutex<Vec<Route>>>,
    is_ready: AtomicBool,
    stop: AtomicBool,
}

/// Frontend paths that may load while the model is still loading.
fn frontend_paths() -> Vec<String> {
    vec!["/".to_string(), "/index.html".to_string(), "/app.js".to_string(), "/style.css".to_string()]
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = hex_val(bytes[i + 1]);
            let lo = hex_val(bytes[i + 2]);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// `httplib::encode_query_component()`, in its minimal form.
pub fn encode_query_component(s: &str) -> String {
    let mut out = String::new();
    for byte in s.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{:02X}", byte));
        }
    }
    out
}

fn parse_query(query: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let mut kv = pair.splitn(2, '=');
        let key = percent_decode(kv.next().unwrap_or(""));
        let val = percent_decode(kv.next().unwrap_or(""));
        out.insert(key, val);
    }
    out
}

fn build_query_string(params: &BTreeMap<String, String>) -> String {
    params
        .iter()
        .map(|(k, v)| format!("{}={}", encode_query_component(k), encode_query_component(v)))
        .collect::<Vec<_>>()
        .join("&")
}

fn ascii_lower(s: &str) -> String {
    s.to_ascii_lowercase()
}

fn ipv4_is_loopback(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 4 {
        return false;
    }
    let mut nums = [0u32; 4];
    for (i, part) in parts.iter().enumerate() {
        match part.parse::<u32>() {
            Ok(v) if v < 256 => nums[i] = v,
            _ => return false,
        }
    }
    nums[0] == 127
}

fn addr_is_loopback(addr: &str) -> bool {
    if let Some(rest) = addr.strip_prefix("::ffff:") {
        ipv4_is_loopback(rest)
    } else {
        addr == "::1" || ipv4_is_loopback(addr)
    }
}

fn host_name(host: &str) -> String {
    let raw = if let Some(inner) = host.strip_prefix('[') {
        inner.split(']').next().unwrap_or("")
    } else {
        host.split(':').next().unwrap_or("")
    };
    ascii_lower(raw)
}

/// DNS-rebinding guard, port of `request_host_is_allowed()`.
fn request_host_is_allowed(cors_origins: &str, local_addr: &str, req: &HttpRequest) -> bool {
    if !addr_is_loopback(local_addr) {
        return true;
    }
    let name = host_name(req.get_header("Host"));
    if name == "localhost" || name == "0.0.0.0" || name == "::" || addr_is_loopback(&name) {
        return true;
    }
    if cors_origins == "localhost" {
        return false;
    }
    let host = req.get_header("Host");
    for entry in parse_csv_row(cors_origins) {
        if let Some(idx) = entry.find("://") {
            if ascii_lower(&entry[idx + 3..]) == ascii_lower(host) {
                return true;
            }
        }
    }
    false
}

/// CSRF guard, port of `request_origin_is_allowed()`.
fn request_origin_is_allowed(cors_origins: &str, local_addr: &str, req: &HttpRequest, is_frontend: bool) -> bool {
    if cors_origins == "*" {
        return true;
    }
    if !request_host_is_allowed(cors_origins, local_addr, req) {
        return false;
    }
    let origin = req.get_header("Origin");
    if !origin.is_empty() {
        let host = req.get_header("Host");
        if let Some(idx) = origin.find("://") {
            if !host.is_empty() && ascii_lower(&origin[idx + 3..]) == ascii_lower(host) {
                return true;
            }
        }
        if cors_origins == "localhost" {
            return false;
        }
        return parse_csv_row(cors_origins).iter().any(|v| v == origin);
    }
    let site = req.get_header("Sec-Fetch-Site");
    if site == "cross-site" || site == "same-site" {
        return req.get_header("Sec-Fetch-Mode") == "navigate" && is_frontend;
    }
    true
}

fn json_error(status: i32, message: &str, kind: ErrorType) -> HttpResponse {
    HttpResponse {
        status,
        data: safe_json_to_str(&error_body(message, kind)),
        ..HttpResponse::default()
    }
}

impl HttpInner {
    /// Pre-routing middleware: CORS, API keys and the loading-state guard.
    fn pre_routing(&self, req: &HttpRequest, local_addr: &str) -> Option<HttpResponse> {
        let s = &self.settings;
        if s.managed_by_router
            && (req.has_header("Origin") || req.has_header("Sec-Fetch-Site") || req.has_header("Sec-Fetch-Mode"))
        {
            return Some(json_error(403, "Browser requests must go through the router", ErrorType::Permission));
        }

        let frontend = frontend_paths().iter().any(|p| p == &req.path)
            || (s.public_path.len() > 1 && req.path.starts_with(&s.public_path));

        if req.method_is_options() {
            return Some(HttpResponse {
                status: 200,
                data: String::new(),
                content_type: "text/html".to_string(),
                headers: cors_headers(s, Some(req.get_header("Origin"))),
                ..HttpResponse::default()
            });
        }

        if !request_origin_is_allowed(&s.cors_origins, local_addr, req, frontend) {
            return Some(json_error(
                403,
                "Cross-origin request not allowed (see --cors-origins)",
                ErrorType::Permission,
            ));
        }

        if !self.is_ready.load(Ordering::Relaxed) {
            let public = req.path == "/health"
                || req.path == "/v1/health"
                || req.path == "/models/sse"
                || frontend;
            if !public {
                return Some(json_error(503, "Loading model", ErrorType::Unavailable));
            }
        }

        if !s.api_keys.is_empty() {
            let public = req.path == "/health" || req.path == "/v1/health" || frontend;
            if !public {
                let mut key = req.get_header("Authorization");
                if key.is_empty() {
                    key = req.get_header("X-Api-Key");
                }
                if let Some(stripped) = key.strip_prefix("Bearer ") {
                    key = stripped;
                }
                if !s.api_keys.iter().any(|k| k == key) {
                    return Some(json_error(401, "Invalid API Key", ErrorType::Authentication));
                }
            }
        }
        None
    }

    /// Match a registered route, filling `:name` path parameters.
    fn dispatch(&self, method: Method, path: &str, req: &mut HttpRequest) -> HttpResponse {
        let routes = self.routes.lock().unwrap_or_else(|e| e.into_inner());
        for route in routes.iter() {
            if route.method != method || route.handler.is_none() {
                continue;
            }
            let full = format!("{}{}", self.settings.path_prefix, route.pattern);
            if let Some(params) = match_route(&full, path) {
                for (key, value) in params {
                    req.params.insert(key, value);
                }
                return (route.handler.as_ref().unwrap())(req);
            }
        }
        json_error(404, "not found", ErrorType::NotFound)
    }

    fn dispatch_ws(&self, path: &str, req: &HttpRequest, conn: &mut dyn WsConn) -> bool {
        let routes = self.routes.lock().unwrap_or_else(|e| e.into_inner());
        for route in routes.iter() {
            if route.method != Method::Ws || route.ws_handler.is_none() {
                continue;
            }
            if match_route(&(self.settings.path_prefix.clone() + &route.pattern), path).is_some() {
                if let Some(handler) = &route.ws_handler {
                    handler(req, conn);
                    return true;
                }
            }
        }
        false
    }
}

impl HttpRequest {
    fn method_is_options(&self) -> bool {
        self.headers.get("__method").map(|m| m == "OPTIONS").unwrap_or(false)
    }
}

fn cors_headers(s: &ServerHttpParams, origin: Option<&str>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if s.cors_credentials && s.cors_origins == "*" {
        out.insert("Access-Control-Allow-Origin".to_string(), origin.unwrap_or("").to_string());
    } else if s.cors_origins != "localhost" {
        out.insert("Access-Control-Allow-Origin".to_string(), s.cors_origins.clone());
    }
    out.insert(
        "Access-Control-Allow-Credentials".to_string(),
        if s.cors_credentials { "true" } else { "false" }.to_string(),
    );
    out.insert("Access-Control-Allow-Methods".to_string(), s.cors_methods.clone());
    out.insert("Access-Control-Allow-Headers".to_string(), s.cors_headers.clone());
    out
}

/// Route pattern matching with `:param` segments; path components are decoded.
fn match_route(pattern: &str, path: &str) -> Option<BTreeMap<String, String>> {
    let pattern_segments: Vec<&str> = pattern.trim_matches('/').split('/').filter(|s| !s.is_empty()).collect();
    let path_segments: Vec<&str> = path.trim_matches('/').split('/').filter(|s| !s.is_empty()).collect();
    if pattern_segments.len() != path_segments.len() {
        return None;
    }
    let mut params = BTreeMap::new();
    for (pat, actual) in pattern_segments.iter().zip(path_segments) {
        if let Some(name) = pat.strip_prefix(':') {
            params.insert(name.to_string(), percent_decode(actual));
        } else if *pat != actual {
            return None;
        }
    }
    Some(params)
}

impl HttpContext {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(HttpInner {
                settings: ServerHttpParams {
                    path_prefix: String::new(),
                    port: 9931,
                    hostnames: vec!["127.0.0.1".to_string()],
                    api_keys: Vec::new(),
                    cors_origins: "*".to_string(),
                    cors_methods: "GET, POST, DELETE, OPTIONS".to_string(),
                    cors_headers: "*".to_string(),
                    cors_credentials: true,
                    public_path: String::new(),
                    timeout_read: 3600,
                    reuse_port: false,
                    is_ssl: false,
                    managed_by_router: false,
                },
                routes: Arc::new(Mutex::new(Vec::new())),
                is_ready: AtomicBool::new(false),
                stop: AtomicBool::new(false),
            }),
            threads: Vec::new(),
            listening_addresses: Vec::new(),
        }
    }

    /// `server_http_context::init()`.
    pub fn init(&mut self, params: &crate::common::Params) -> bool {
        let default_port = self.inner.settings.port;
        let port = if std::env::var("AIP_MODE").as_deref() == Ok("PREDICTION") {
            std::env::var("AIP_HTTP_PORT")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
                .unwrap_or(default_port)
        } else {
            default_port
        };
        let inner = Arc::get_mut(&mut self.inner).expect("HTTP context is already running");
        inner.settings = ServerHttpParams::from_params(params);
        inner.settings.port = port;
        true
    }

    /// `server_http_context::get()`.
    pub fn get(&self, path: impl Into<String>, handler: Handler) {
        self.add(Method::Get, path.into(), Some(handler), None);
    }

    /// `server_http_context::post()`.
    pub fn post(&self, path: impl Into<String>, handler: Handler) {
        self.add(Method::Post, path.into(), Some(handler), None);
    }

    /// `server_http_context::del()`.
    pub fn del(&self, path: impl Into<String>, handler: Handler) {
        self.add(Method::Delete, path.into(), Some(handler), None);
    }

    /// `server_http_context::ws()`.
    pub fn ws(&self, path: impl Into<String>, handler: WsHandler) {
        self.add(Method::Ws, path.into(), None, Some(handler));
    }

    fn add(&self, method: Method, path: String, handler: Option<Handler>, ws_handler: Option<WsHandler>) {
        let mut routes = self.inner.routes.lock().unwrap_or_else(|e| e.into_inner());
        if !routes.iter().any(|r| r.method == method && r.pattern == path) {
            routes.push(Route { method, pattern: path, handler, ws_handler });
        }
    }

    pub fn set_ready(&self, ready: bool) {
        self.inner.is_ready.store(ready, Ordering::Relaxed);
    }

    pub fn is_ready(&self) -> bool {
        self.inner.is_ready.load(Ordering::Relaxed)
    }

    pub fn is_stopped(&self) -> bool {
        self.inner.stop.load(Ordering::Relaxed)
    }

    pub fn port(&self) -> i32 {
        self.inner.settings.port
    }

    /// `server_http_context::register_gcp_compat()`.
    pub fn register_gcp_compat(&self) {
        if std::env::var("AIP_MODE").as_deref() != Ok("PREDICTION") {
            return;
        }
        let mut health = std::env::var("AIP_HEALTH_ROUTE").unwrap_or_default();
        if !health.is_empty() && !health.starts_with('/') {
            health.insert(0, '/');
        }
        let predict = std::env::var("AIP_PREDICT_ROUTE").unwrap_or_else(|_| "/predict".to_string());
        let predict = if predict.starts_with('/') { predict } else { format!("/{}", predict) };

        if !health.is_empty() {
            self.get(health, Arc::new(|_| HttpResponse::ok(&Json::object_from(vec![Item::new("status", "ok")]))));
        }

        let routes = Arc::clone(&self.inner.routes);
        let prefix = self.inner.settings.path_prefix.clone();
        self.post(predict, Arc::new(move |req| {
            let body = Json::parse(&req.body).unwrap_or_else(|_| Json::object());
            let instance = body
                .at("instances")
                .and_then(|v| v.at_index(0))
                .cloned()
                .unwrap_or_else(|_| Json::object());
            let format = instance.value_str("requestFormat", "").unwrap_or_default();
            let path = format_to_path(&format).unwrap_or("/v1/chat/completions");
            let full = format!("{}{}", prefix, path);
            let mut forwarded = HttpRequest {
                body: instance.dump_safe(-1),
                ..HttpRequest::default()
            };
            let guard = routes.lock().unwrap_or_else(|e| e.into_inner());
            for route in guard.iter() {
                if route.method == Method::Post && full.ends_with(&route.pattern) {
                    if let Some(handler) = &route.handler {
                        return handler(&forwarded);
                    }
                }
            }
            drop(guard);
            forwarded.path = full;
            HttpResponse::error_message("Vertex AI request format is not supported", ErrorType::NotSupported)
        }));
    }

    /// `server_http_context::start()`.
    pub fn start(&mut self) -> bool {
        let hostnames = self.inner.settings.hostnames.clone();
        let port = self.inner.settings.port;
        for host in hostnames {
            let addr = if host.ends_with(".sock") {
                let _ = fs::remove_file(&host);
                continue;
            } else if host.contains(':') && !host.starts_with('[') {
                format!("[{}]:{}", host, port)
            } else {
                format!("{}:{}", host, port)
            };
            let addrs = match addr.to_socket_addrs() {
                Ok(a) => a.collect::<Vec<_>>(),
                Err(_) => continue,
            };
            let Some(socket_addr) = addrs.first().copied() else {
                continue;
            };
            let listener = match TcpListener::bind(socket_addr) {
                Ok(l) => l,
                Err(_) => continue,
            };
            let _ = listener.set_nonblocking(true);
            self.listening_addresses.push(format!("http://{}", socket_addr));
            let inner = Arc::clone(&self.inner);
            let handle = std::thread::spawn(move || {
                while !inner.stop.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => serve_connection(&inner, stream, socket_addr.to_string()),
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(50));
                        }
                        Err(_) => std::thread::sleep(Duration::from_millis(50)),
                    }
                }
            });
            self.threads.push(handle);
        }
        !self.threads.is_empty()
    }

    /// `server_http_context::stop()`.
    pub fn stop(&self) {
        self.inner.stop.store(true, Ordering::Relaxed);
        self.inner.is_ready.store(false, Ordering::Relaxed);
    }

    /// `server_http_context::join()`.
    pub fn join(&mut self) {
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }

    /// Addresses logged after the listener starts.
    pub fn listening_addresses(&self) -> &[String] {
        &self.listening_addresses
    }

    /// Dispatch a request directly, bypassing the TCP transport.
    pub fn dispatch_request(&self, method: Method, req: &HttpRequest) -> HttpResponse {
        self.inner.dispatch(method, &req.path, &mut req.clone_for_dispatch())
    }
}

impl HttpRequest {
    fn clone_for_dispatch(&self) -> Self {
        Self {
            params: self.params.clone(),
            headers: self.headers.clone(),
            path: self.path.clone(),
            query_string: self.query_string.clone(),
            body: self.body.clone(),
            files: self.files.clone(),
            should_stop: Arc::clone(&self.should_stop),
        }
    }
}

/// Derive the `@requestFormat` alias from a registered path.
fn format_to_path(format: &str) -> Option<&'static str> {
    match format {
        "chatCompletions" => Some("/v1/chat/completions"),
        "completions" => Some("/v1/completions"),
        "embeddings" => Some("/v1/embeddings"),
        "rerank" => Some("/v1/rerank"),
        "responses" => Some("/v1/responses"),
        _ => None,
    }
}

/// Read and parse one HTTP/1.1 request.
fn read_request(stream: &mut TcpStream) -> Result<(Method, String, BTreeMap<String, String>, String)> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end;
    loop {
        let n = stream.read(&mut chunk).map_err(|e| crate::core::Error::new(e.to_string()))?;
        if n == 0 {
            return Err(crate::core::Error::new("connection closed"));
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_bytes(&buf, b"\r\n\r\n") {
            header_end = pos + 4;
            break;
        }
        if buf.len() > 1024 * 1024 {
            return Err(crate::core::Error::new("request header is too large"));
        }
    }
    let head = String::from_utf8_lossy(&buf[..header_end - 4]);
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut fields = request_line.split_whitespace();
    let method = match fields.next().unwrap_or("") {
        "GET" => Method::Get,
        "POST" => Method::Post,
        "DELETE" => Method::Delete,
        "OPTIONS" => Method::Options,
        _ => return Err(crate::core::Error::new("unsupported HTTP method")),
    };
    let target = fields.next().unwrap_or("/").to_string();
    let mut headers = BTreeMap::new();
    for line in lines {
        if let Some((key, value)) = line.split_once(':') {
            headers.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    let length = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("Content-Length"))
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .unwrap_or(0);
    while buf.len() < header_end + length {
        let n = stream.read(&mut chunk).map_err(|e| crate::core::Error::new(e.to_string()))?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = String::from_utf8_lossy(&buf[header_end..header_end + length.min(buf.len() - header_end)]).into_owned();
    Ok((method, target, headers, body))
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// Serve one connection: request parsing, middlewares, dispatch and response.
fn serve_connection(inner: &HttpInner, mut stream: TcpStream, local_addr: String) {
    let (method, target, mut headers, body) = match read_request(&mut stream) {
        Ok(v) => v,
        Err(_) => return,
    };
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target.clone(), String::new()),
    };
    let params = parse_query(&query);
    let query_string = build_query_string(&params);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_cb = Arc::clone(&stop);
    let should_stop: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || stop_cb.load(Ordering::Relaxed));
    headers.insert("__method".to_string(), method.as_str().to_string());
    let mut req = HttpRequest {
        params,
        headers,
        path: path.clone(),
        query_string,
        body,
        files: BTreeMap::new(),
        should_stop,
    };

    let mut response = match inner.pre_routing(&req, &local_addr) {
        Some(res) => res,
        None if method == Method::Options => HttpResponse::default(),
        None if method == Method::Ws => {
            // WebSocket upgrade is handled by the registered ws handler via a
            // minimal framing adapter; the HTTP request is otherwise identical.
            let mut conn = RawWsConn::new(&mut stream);
            if inner.dispatch_ws(&path, &req, &mut conn) {
                return;
            }
            json_error(404, "not found", ErrorType::NotFound)
        }
        None => {
            req.files = parse_multipart(&req);
            let static_response = if method == Method::Get && !inner.settings.public_path.is_empty() {
                static_file(Path::new(&inner.settings.public_path), &path).map(|(data, content_type)| HttpResponse {
                    status: 200,
                    content_type,
                    data: String::from_utf8_lossy(&data).into_owned(),
                    ..HttpResponse::default()
                })
            } else {
                None
            };
            static_response.unwrap_or_else(|| inner.dispatch(method, &path, &mut req))
        }
    };

    if method == Method::Options {
        response.headers.extend(cors_headers(&inner.settings, Some(req.get_header("Origin"))));
    } else {
        response.headers.extend(cors_headers(&inner.settings, Some(req.get_header("Origin"))));
    }

    let status_line = format!("HTTP/1.1 {} {}", response.status, status_text(response.status));
    let mut out = format!("{}\r\n", status_line);
    if response.is_stream() {
        out.push_str("Transfer-Encoding: chunked\r\n");
        out.push_str("X-Accel-Buffering: no\r\n");
    } else {
        out.push_str(&format!("Content-Length: {}\r\n", response.data.len()));
    }
    out.push_str(&format!("Content-Type: {}\r\n", response.content_type));
    for (key, value) in &response.headers {
        out.push_str(&format!("{}: {}\r\n", key, value));
    }
    out.push_str("Connection: close\r\n\r\n");
    if stream.write_all(out.as_bytes()).is_err() {
        response.complete();
        return;
    }

    if let Some(next) = response.next.clone() {
        loop {
            let mut chunk = String::new();
            let mut has_next = false;
            if let Ok(mut provider) = next.lock() {
                has_next = provider(&mut chunk);
            }
            if write_chunk(&mut stream, chunk.as_bytes()).is_err() {
                stop.store(true, Ordering::Relaxed);
                break;
            }
            if !has_next {
                let _ = stream.write_all(b"0\r\n\r\n");
                break;
            }
        }
    } else if stream.write_all(response.data.as_bytes()).is_err() {
        stop.store(true, Ordering::Relaxed);
    }
    let _ = stream.flush();
    response.complete();
}

fn status_text(status: i32) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

fn write_chunk(stream: &mut TcpStream, data: &[u8]) -> std::io::Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    stream.write_all(format!("{:X}\r\n", data.len()).as_bytes())?;
    stream.write_all(data)?;
    stream.write_all(b"\r\n")
}

/// Translate multipart/form-data fields into the JSON body and uploaded files.
fn parse_multipart(req: &HttpRequest) -> BTreeMap<String, UploadedFile> {
    let mut files = BTreeMap::new();
    let content_type = req.get_header("Content-Type");
    let boundary = content_type
        .split("boundary=")
        .nth(1)
        .map(|v| v.trim().trim_matches('"').to_string());
    let Some(boundary) = boundary else {
        return files;
    };
    let marker = format!("--{}", boundary);
    let body = req.body.as_bytes();
    let mut parts: Vec<&[u8]> = Vec::new();
    let mut rest = body;
    while let Some(pos) = find_bytes(rest, marker.as_bytes()) {
        let after = &rest[pos + marker.len()..];
        if after.starts_with(b"--") {
            break;
        }
        if let Some(end) = find_bytes(after, marker.as_bytes()) {
            parts.push(&after[..end]);
            rest = &after[end..];
        } else {
            break;
        }
    }
    for part in parts {
        let part = part.strip_prefix(b"\r\n").unwrap_or(part);
        let Some(sep) = find_bytes(part, b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&part[..sep]);
        let content = part[sep + 4..].strip_suffix(b"\r\n").unwrap_or(&part[sep + 4..]).to_vec();
        let name = header_param(&headers, "name");
        let filename = header_param(&headers, "filename");
        if !filename.is_empty() {
            let content_type = header_line(&headers, "Content-Type").unwrap_or_default();
            files.insert(
                name,
                UploadedFile { data: content, filename, content_type },
            );
        }
    }
    files
}

fn header_line(headers: &str, key: &str) -> Option<String> {
    for line in headers.split("\r\n") {
        if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case(key) {
                return Some(v.trim().to_string());
            }
        }
    }
    None
}

fn header_param(headers: &str, key: &str) -> String {
    for line in headers.split("\r\n") {
        if let Some((_, value)) = line.split_once(':') {
            for param in value.split(';') {
                if let Some((k, v)) = param.trim().split_once('=') {
                    if k.eq_ignore_ascii_case(key) {
                        return v.trim().trim_matches('"').to_string();
                    }
                }
            }
        }
    }
    String::new()
}

/// Minimal WebSocket connection: unmasked server frames over a raw stream.
struct RawWsConn<'a> {
    stream: &'a mut TcpStream,
    open: bool,
}

impl<'a> RawWsConn<'a> {
    fn new(stream: &'a mut TcpStream) -> Self {
        Self { stream, open: true }
    }

    fn write_frame(&self, payload: &[u8], opcode: u8) -> bool {
        let mut stream: &TcpStream = self.stream;
        let mut frame = Vec::with_capacity(payload.len() + 10);
        frame.push(0x80 | opcode);
        if payload.len() < 126 {
            frame.push(payload.len() as u8);
        } else if payload.len() < 65536 {
            frame.push(126);
            frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        } else {
            frame.push(127);
            frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
        }
        frame.extend_from_slice(payload);
        if stream.write_all(&frame).is_err() {
            return false;
        }
        stream.flush().is_ok()
    }
}

impl WsConn for RawWsConn<'_> {
    fn read(&mut self, msg: &mut String) -> i32 {
        let mut head = [0u8; 2];
        if self.stream.read_exact(&mut head).is_err() {
            self.open = false;
            return 0;
        }
        let opcode = head[0] & 0x0f;
        let mut len = (head[1] & 0x7f) as usize;
        if len == 126 {
            let mut ext = [0u8; 2];
            if self.stream.read_exact(&mut ext).is_err() {
                return 0;
            }
            len = u16::from_be_bytes(ext) as usize;
        } else if len == 127 {
            let mut ext = [0u8; 8];
            if self.stream.read_exact(&mut ext).is_err() {
                return 0;
            }
            len = u64::from_be_bytes(ext) as usize;
        }
        let mut payload = vec![0u8; len.min(4 * 1024 * 1024)];
        if self.stream.read_exact(&mut payload).is_err() {
            self.open = false;
            return 0;
        }
        *msg = String::from_utf8_lossy(&payload).into_owned();
        if opcode == 0x2 {
            2
        } else {
            1
        }
    }

    fn send(&self, msg: &str) -> bool {
        self.write_frame(msg.as_bytes(), 0x1)
    }

    fn send_binary(&self, msg: &[u8]) -> bool {
        self.write_frame(msg, 0x2)
    }

    fn is_open(&self) -> bool {
        self.open
    }

    fn close(&self) {
        let _ = self.write_frame(&[], 0x8);
    }
}

/// Serve static files from `--path`, mirroring the embedded UI's static route.
pub fn static_file(root: &Path, path: &str) -> Option<(Vec<u8>, String)> {
    let relative = path.trim_start_matches('/');
    let relative = if relative.is_empty() { "index.html" } else { relative };
    let full: PathBuf = root.join(relative);
    if !full.starts_with(root) {
        return None;
    }
    let data = fs::read(&full).ok()?;
    let content_type = match full.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "application/javascript",
        Some("css") => "text/css",
        Some("json") => "application/json; charset=utf-8",
        Some("png") => "image/png",
        Some("svg") => "image/svg+xml",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    };
    Some((data, content_type.to_string()))
}

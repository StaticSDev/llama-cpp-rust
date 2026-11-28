//! Tiny std-only HTTP/1.1 server for the chat UI and generation API.

use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::time::Duration;

use crate::json::{Item, Json};
use crate::runtime::generate::{generate_stream, Event, Stats};
use crate::runtime::types::Weights;

const MAX_HEADERS: usize = 8 * 1024;
const MAX_BODY: usize = 1024 * 1024;
const MAX_TOKENS: usize = 512;

/// Serve `weights` on an already-bound listener until the process exits.
///
/// One connection at a time: generation builds its own model state, so a
/// shared `&Weights` needs no lock and no `Send + Sync` bound.
pub fn serve(listener: TcpListener, weights: Weights, model_path: String, gpu: String) {
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(err) = handle(&stream, &weights, &model_path, &gpu) {
                    if !matches!(
                        err.kind(),
                        io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
                    ) {
                        eprintln!("connection error: {err}");
                    }
                }
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }
}

fn handle(
    stream: &TcpStream,
    weights: &Weights,
    model_path: &str,
    gpu: &str,
) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(15)))?;
    let mut reader = BufReader::new(stream);

    let mut head = Vec::new();
    loop {
        if reader.read_until(b'\n', &mut head)? == 0 {
            return Ok(());
        }
        if head.ends_with(b"\r\n\r\n") || head.ends_with(b"\n\n") {
            break;
        }
        if head.len() > MAX_HEADERS {
            return respond(
                stream,
                "400 Bad Request",
                "text/plain; charset=utf-8",
                b"request headers too large\n",
            );
        }
    }

    let head = String::from_utf8_lossy(&head);
    let request_line = head.lines().next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("");
    let path = target.split('?').next().unwrap_or(target);

    let mut content_length = 0usize;
    for line in head.lines().skip(1) {
        if let Some((name, value)) = line.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
    }
    if content_length > MAX_BODY {
        return respond(
            stream,
            "413 Payload Too Large",
            "text/plain; charset=utf-8",
            b"request body too large\n",
        );
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;

    match (method, path) {
        ("GET", "/") => serve_file(stream, "index.html", "text/html; charset=utf-8"),
        ("GET", "/style.css") => serve_file(stream, "style.css", "text/css; charset=utf-8"),
        ("GET", "/app.js") => serve_file(stream, "app.js", "application/javascript; charset=utf-8"),
        ("GET", "/api/status") => {
            let json = Json::object_from(vec![
                Item::new("model", model_path),
                Item::new("ready", true),
                Item::new("gpu", gpu),
            ]);
            respond_json(stream, "200 OK", &json)
        }
        ("POST", "/api/chat") => chat(stream, weights, &body, false),
        ("POST", "/api/chat/stream") => chat(stream, weights, &body, true),
        ("GET", "/v1/models") => models(stream, model_path),
        ("POST", "/v1/chat/completions") => openai_chat(stream, weights, &body, model_path),
        _ => respond(stream, "404 Not Found", "text/plain; charset=utf-8", b"Not Found\n"),
    }
}

fn models(stream: &TcpStream, model_path: &str) -> io::Result<()> {
    let json = Json::object_from(vec![
        Item::new("object", "list"),
        Item::new(
            "data",
            Json::array_from(vec![Json::object_from(vec![
                Item::new("id", model_path),
                Item::new("object", "model"),
                Item::new("owned_by", "local"),
            ])]),
        ),
    ]);
    respond_json(stream, "200 OK", &json)
}

fn openai_chat(
    stream: &TcpStream,
    weights: &Weights,
    body: &[u8],
    model_path: &str,
) -> io::Result<()> {
    let text = String::from_utf8_lossy(body);
    let json = match Json::parse(&text) {
        Ok(json) => json,
        Err(err) => return json_error(stream, "400 Bad Request", &err.to_string()),
    };
    if !json.is_object() {
        return json_error(stream, "400 Bad Request", "body must be a JSON object");
    }

    // Only the last user turn becomes the prompt; other roles are ignored.
    let messages = match json.value("messages", Vec::<Json>::new()) {
        Ok(messages) => messages,
        Err(err) => return json_error(stream, "400 Bad Request", &err.to_string()),
    };
    let mut prompt: Option<String> = None;
    for message in &messages {
        if message.value_str("role", "").unwrap_or_default() != "user" {
            continue;
        }
        match message.value_str("content", "") {
            Ok(content) => prompt = Some(content),
            Err(err) => return json_error(stream, "400 Bad Request", &err.to_string()),
        }
    }
    let prompt = match prompt {
        Some(prompt) if !prompt.is_empty() => prompt,
        _ => return json_error(stream, "400 Bad Request", "missing or empty 'messages' user content"),
    };

    let n = match json.value("max_tokens", 64i64) {
        Ok(n) => n.clamp(0, MAX_TOKENS as i64) as usize,
        Err(err) => return json_error(stream, "400 Bad Request", &err.to_string()),
    };
    let temp = match json.value("temperature", 0.0f32) {
        Ok(temp) => temp,
        Err(err) => return json_error(stream, "400 Bad Request", &err.to_string()),
    };
    // Accepted for wire compatibility; the runtime seeds its RNG from time.
    if let Err(err) = json.value("seed", 0i64) {
        return json_error(stream, "400 Bad Request", &err.to_string());
    }
    if !temp.is_finite() {
        return json_error(stream, "400 Bad Request", "temperature must be finite");
    }

    // In GPU mode generate_stream rejects temp > 0 (greedy-only sampler),
    // the same way /api/chat does.
    let (text, stats) = match generate_stream(weights, &prompt, n, temp, 0.9, |_| {}) {
        Ok(result) => result,
        Err(err) => return json_error(stream, "500 Internal Server Error", &err.to_string()),
    };

    let json = Json::object_from(vec![
        Item::new("id", "lcr-1"),
        Item::new("object", "chat.completion"),
        Item::new("model", model_path),
        Item::new(
            "choices",
            Json::array_from(vec![Json::object_from(vec![
                Item::new("index", 0i64),
                Item::new(
                    "message",
                    Json::object_from(vec![
                        Item::new("role", "assistant"),
                        Item::new("content", text),
                    ]),
                ),
                Item::new("finish_reason", "stop"),
            ])]),
        ),
        Item::new(
            "usage",
            Json::object_from(vec![
                Item::new("prompt_tokens", stats.prompt_tokens),
                Item::new("completion_tokens", stats.tokens),
                Item::new("total_tokens", stats.prompt_tokens + stats.tokens),
            ]),
        ),
        Item::new(
            "timings",
            Json::object_from(vec![
                Item::new("prompt_n", stats.prompt_tokens),
                Item::new("prompt_per_second", stats.prompt_tokens_per_sec()),
                Item::new("predicted_n", stats.tokens),
                Item::new("predicted_per_second", stats.tokens_per_sec()),
                Item::new("prompt_ms", stats.prompt_ms as u64),
                Item::new("predicted_ms", stats.gen_ms as u64),
            ]),
        ),
    ]);
    respond_json(stream, "200 OK", &json)
}

fn chat(stream: &TcpStream, weights: &Weights, body: &[u8], streaming: bool) -> io::Result<()> {
    let text = String::from_utf8_lossy(body);
    let json = match Json::parse(&text) {
        Ok(json) => json,
        Err(err) => return json_error(stream, "400 Bad Request", &err.to_string()),
    };
    if !json.is_object() {
        return json_error(stream, "400 Bad Request", "body must be a JSON object");
    }
    if !json.contains("prompt") {
        return json_error(stream, "400 Bad Request", "missing 'prompt'");
    }

    let prompt = match json.value_str("prompt", "") {
        Ok(prompt) => prompt,
        Err(err) => return json_error(stream, "400 Bad Request", &err.to_string()),
    };
    let n = match json.value("n_predict", 64i64) {
        Ok(n) => n,
        Err(err) => return json_error(stream, "400 Bad Request", &err.to_string()),
    };
    if !(0..=MAX_TOKENS as i64).contains(&n) {
        return json_error(stream, "400 Bad Request", "n_predict out of range");
    }
    let temp = match json.value("temp", 0.0f32) {
        Ok(temp) => temp,
        Err(err) => return json_error(stream, "400 Bad Request", &err.to_string()),
    };
    let top_p = match json.value("top_p", 0.9f32) {
        Ok(top_p) => top_p,
        Err(err) => return json_error(stream, "400 Bad Request", &err.to_string()),
    };
    if !temp.is_finite() || !top_p.is_finite() {
        return json_error(stream, "400 Bad Request", "temp and top_p must be finite");
    }

    if !streaming {
        return match generate_stream(weights, &prompt, n as usize, temp, top_p, |_| {}) {
            Ok((text, _)) => {
                let json = Json::object_from(vec![Item::new("text", text)]);
                respond_json(stream, "200 OK", &json)
            }
            Err(err) => json_error(stream, "500 Internal Server Error", &err.to_string()),
        };
    }

    // Close-delimited NDJSON: one JSON object per line, flushed as it happens.
    // A failed write is remembered, not retried; generation still runs to the
    // end because the callback cannot unwind the sampler.
    let mut write_err: Option<io::Error> = None;
    let mut write = |data: &str| {
        if write_err.is_some() {
            return;
        }
        if let Err(err) = write_raw(stream, data.as_bytes()) {
            write_err = Some(err);
        }
    };

    let mut started = false;
    let result = generate_stream(weights, &prompt, n as usize, temp, top_p, |event| {
        if !started {
            started = true;
            write(NDJSON_HEAD);
        }
        let line = match event {
            Event::Prompt { done, total } => Json::object_from(vec![
                Item::new("type", "prompt"),
                Item::new("done", done),
                Item::new("total", total),
            ])
            .dump_safe(-1),
            Event::Token(text) => {
                Json::object_from(vec![Item::new("type", "token"), Item::new("text", text)])
                    .dump_safe(-1)
            }
        };
        write(&(line + "\n"));
    });

    if !started {
        write(NDJSON_HEAD);
    }
    match result {
        Ok((_, stats)) => write(&(stats_line(&stats) + "\n")),
        Err(err) => {
            let line = Json::object_from(vec![
                Item::new("type", "error"),
                Item::new("error", err.to_string()),
            ])
            .dump_safe(-1);
            write(&(line + "\n"));
        }
    }
    match write_err {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

const NDJSON_HEAD: &str = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";

fn write_raw(stream: &TcpStream, bytes: &[u8]) -> io::Result<()> {
    let mut stream = stream;
    stream.write_all(bytes)?;
    stream.flush()
}

fn stats_line(stats: &Stats) -> String {
    Json::object_from(vec![
        Item::new("type", "done"),
        Item::new("prompt_tokens", stats.prompt_tokens),
        Item::new("prompt_ms", stats.prompt_ms as u64),
        Item::new("prompt_tok_s", stats.prompt_tokens_per_sec()),
        Item::new("tokens", stats.tokens),
        Item::new("gen_ms", stats.gen_ms as u64),
        Item::new("tok_s", stats.tokens_per_sec()),
    ])
    .dump_safe(-1)
}

fn serve_file(stream: &TcpStream, name: &str, content_type: &str) -> io::Result<()> {
    match fs::read(Path::new("web").join(name)) {
        Ok(bytes) => respond(stream, "200 OK", content_type, &bytes),
        Err(_) => respond(
            stream,
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"Not Found\n",
        ),
    }
}

fn respond_json(stream: &TcpStream, status: &str, json: &Json) -> io::Result<()> {
    respond(stream, status, "application/json", json.dump_safe(-1).as_bytes())
}

fn json_error(stream: &TcpStream, status: &str, message: &str) -> io::Result<()> {
    let json = Json::object_from(vec![Item::new("error", message)]);
    respond_json(stream, status, &json)
}

fn respond(stream: &TcpStream, status: &str, content_type: &str, body: &[u8]) -> io::Result<()> {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let mut stream = stream;
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

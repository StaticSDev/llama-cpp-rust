//! Tiny std-only HTTP/1.1 server for the chat UI and generation API.

use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::time::Duration;

use crate::json::{Item, Json};
use crate::runtime::generate::generate;
use crate::runtime::types::Weights;

const MAX_HEADERS: usize = 8 * 1024;
const MAX_BODY: usize = 1024 * 1024;
const MAX_TOKENS: usize = 512;

/// Serve `weights` on an already-bound listener until the process exits.
///
/// One connection at a time: generation builds its own model state, so a
/// shared `&Weights` needs no lock and no `Send + Sync` bound.
pub fn serve(listener: TcpListener, weights: Weights, model_path: String) {
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(err) = handle(&stream, &weights, &model_path) {
                    eprintln!("connection error: {err}");
                }
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }
}

fn handle(stream: &TcpStream, weights: &Weights, model_path: &str) -> io::Result<()> {
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
            ]);
            respond_json(stream, "200 OK", &json)
        }
        ("POST", "/api/chat") => chat(stream, weights, &body),
        _ => respond(stream, "404 Not Found", "text/plain; charset=utf-8", b"Not Found\n"),
    }
}

fn chat(stream: &TcpStream, weights: &Weights, body: &[u8]) -> io::Result<()> {
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

    match generate(weights, &prompt, n as usize, temp, top_p) {
        Ok(text) => {
            let json = Json::object_from(vec![Item::new("text", text)]);
            respond_json(stream, "200 OK", &json)
        }
        Err(err) => json_error(stream, "500 Internal Server Error", &err.to_string()),
    }
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

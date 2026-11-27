//! Rust port of `common/download.{h,cpp}`, `common/hf-cache.{h,cpp}`,
//! `common/file-lock.{h,cpp}` and the URL helpers of `common/http.h`.
//!
//! No HTTP library is available, so remote access is modelled with a small
//! HTTP/1.1 client over `std::net::TcpStream` plus a `file://` stand-in backed
//! by `std::fs`. HTTPS is rejected the same way a non-OpenSSL llama.cpp build
//! rejects it. Paths stay native `PathBuf`s; no FFI is used.

use crate::common::{
    common_get_env, common_get_model_endpoint, fs_get_cache_file, fs_write_atomic,
    string_ends_with, string_remove_suffix, string_split_char, string_starts_with,
    stdout_is_tty, ParamsModel,
};
use crate::core::{Error, Result};
use crate::json::Json;
use std::cell::Cell;
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

const PARTIAL_SUFFIX: &str = ".downloadInProgress";
const USER_AGENT: &str = "llama-cpp/rust-port";
const MAX_ATTEMPTS: i32 = 3;
const RETRY_DELAY_SECONDS: u64 = 2;
const MAX_REDIRECTS: usize = 5;

// ---------------------------------------------------------------------------
// public data types
// ---------------------------------------------------------------------------

pub type Header = (String, String);
pub type HeaderList = Vec<Header>;

#[derive(Clone, Debug, Default)]
pub struct DownloadProgress {
    pub url: String,
    pub downloaded: usize,
    pub total: usize,
    pub cached: bool,
}

/// Port of `common_download_callback`. Methods take `&self` so a callback can
/// be shared across the scoped threads used by `run_tasks`.
pub trait DownloadCallback: Sync {
    fn on_start(&self, p: &DownloadProgress);
    fn on_update(&self, p: &DownloadProgress);
    fn on_done(&self, p: &DownloadProgress, ok: bool);
    fn is_cancelled(&self) -> bool {
        false
    }
}

#[derive(Default)]
pub struct RemoteParams {
    pub headers: HeaderList,
    /// seconds, 0 means no timeout
    pub timeout: u64,
    /// 0 means unlimited
    pub max_size: usize,
    /// called per hop; the returned address is used instead of the host name
    pub resolve_host: Option<Box<dyn Fn(&str) -> String>>,
}

#[derive(Clone, Debug, Default)]
pub struct CachedModelInfo {
    pub repo: String,
    pub tag: String,
}

impl CachedModelInfo {
    pub fn to_string(&self) -> String {
        format!("{}:{}", self.repo, self.tag)
    }
}

#[derive(Clone, Default)]
pub struct DownloadOpts<'a> {
    pub bearer_token: String,
    pub headers: HeaderList,
    pub offline: bool,
    pub download_mmproj: bool,
    pub download_mtp: bool,
    pub download_eagle3: bool,
    pub download_dflash: bool,
    pub download_dspark: bool,
    pub callback: Option<&'a dyn DownloadCallback>,
}

#[derive(Clone, Debug, Default)]
pub struct HfFile {
    pub path: String,
    pub url: String,
    pub local_path: String,
    pub final_path: String,
    pub oid: String,
    pub repo_id: String,
}

pub type HfFiles = Vec<HfFile>;

pub struct DownloadTask<'a> {
    pub opts: DownloadOpts<'a>,
    pub url: String,
    pub local_path: String,
    pub on_done: Option<Box<dyn Fn() + 'a>>,
    pub is_hf: bool,
    pub lock_path: String,
    /// HF snapshot path, also treated as cached
    pub alt_path: String,
}

impl<'a> DownloadTask<'a> {
    pub fn new(url: String, local_path: String, opts: DownloadOpts<'a>) -> Self {
        Self {
            opts,
            url,
            local_path,
            on_done: None,
            is_hf: false,
            lock_path: String::new(),
            alt_path: String::new(),
        }
    }

    pub fn from_hf_file(
        file: HfFile,
        opts: DownloadOpts<'a>,
        on_done: Option<Box<dyn Fn() + 'a>>,
    ) -> Self {
        Self {
            opts,
            url: file.url,
            local_path: file.local_path,
            on_done,
            is_hf: true,
            lock_path: get_lock_path(&file.repo_id, &file.oid),
            alt_path: file.final_path,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidecarKind {
    Mmproj,
    Mtp,
    Dflash,
    Eagle3,
    Dspark,
}

impl SidecarKind {
    pub const ALL: [SidecarKind; 5] = [
        Self::Mmproj,
        Self::Mtp,
        Self::Dflash,
        Self::Eagle3,
        Self::Dspark,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Mmproj => "mmproj",
            Self::Mtp => "mtp",
            Self::Dflash => "dflash",
            Self::Eagle3 => "eagle3",
            Self::Dspark => "dspark",
        }
    }

    pub fn keyword(self) -> &'static str {
        match self {
            Self::Mmproj => "mmproj",
            Self::Mtp => "mtp-",
            Self::Dflash => "dflash-",
            Self::Eagle3 => "eagle3-",
            Self::Dspark => "dspark-",
        }
    }

    /// A draft head resolves at the tag even without a full model.
    pub fn is_draft(self) -> bool {
        self != Self::Mmproj
    }

    pub fn opt<'b>(self, opts: &'b mut DownloadOpts<'_>) -> &'b mut bool {
        match self {
            Self::Mmproj => &mut opts.download_mmproj,
            Self::Mtp => &mut opts.download_mtp,
            Self::Dflash => &mut opts.download_dflash,
            Self::Eagle3 => &mut opts.download_eagle3,
            Self::Dspark => &mut opts.download_dspark,
        }
    }

    pub fn plan_file<'b>(self, plan: &'b DownloadHfPlan) -> &'b HfFile {
        match self {
            Self::Mmproj => &plan.mmproj,
            Self::Mtp => &plan.mtp,
            Self::Dflash => &plan.dflash,
            Self::Eagle3 => &plan.eagle3,
            Self::Dspark => &plan.dspark,
        }
    }

    pub fn plan_file_mut<'b>(self, plan: &'b mut DownloadHfPlan) -> &'b mut HfFile {
        match self {
            Self::Mmproj => &mut plan.mmproj,
            Self::Mtp => &mut plan.mtp,
            Self::Dflash => &mut plan.dflash,
            Self::Eagle3 => &mut plan.eagle3,
            Self::Dspark => &mut plan.dspark,
        }
    }

    fn opt_enabled(self, opts: &DownloadOpts<'_>) -> bool {
        match self {
            Self::Mmproj => opts.download_mmproj,
            Self::Mtp => opts.download_mtp,
            Self::Dflash => opts.download_dflash,
            Self::Eagle3 => opts.download_eagle3,
            Self::Dspark => opts.download_dspark,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DownloadHfPlan {
    pub primary: HfFile,
    pub model_files: HfFiles,
    pub mmproj: HfFile,
    pub mtp: HfFile,
    pub eagle3: HfFile,
    pub dflash: HfFile,
    pub dspark: HfFile,
    /// If set, only this file is downloaded.
    pub preset: HfFile,
}

#[derive(Clone, Debug, Default)]
pub struct DownloadPartial {
    pub kind: String,
    pub path: String,
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

pub fn is_http_status_ok(status: i32) -> bool {
    (200..400).contains(&status)
}

fn write_etag(path: &str, etag: &str) -> Result<()> {
    fs_write_atomic(Path::new(&format!("{path}.etag")), etag.as_bytes())
}

fn read_etag(path: &str) -> String {
    let etag_path = format!("{path}.etag");
    let Ok(text) = fs::read_to_string(etag_path) else {
        return String::new();
    };
    text.lines().next().unwrap_or("").to_owned()
}

/// `common_download_split_repo_tag`.
pub fn split_repo_tag(hf_repo_with_tag: &str) -> Result<(String, String)> {
    let parts = string_split_char(hf_repo_with_tag, ':');
    let tag = parts.last().cloned().unwrap_or_default();
    let repo = parts.first().cloned().unwrap_or_default();
    if string_split_char(&repo, '/').len() != 2 {
        return Err(Error::new(
            "error: invalid HF repo format, expected <user>/<model>[:quant]\n",
        ));
    }
    Ok((repo, if parts.len() > 1 { tag } else { String::new() }))
}

fn url_get_filename(url: &str) -> String {
    let mut filename = match url.rfind('/') {
        Some(pos) => url[pos + 1..].to_owned(),
        None => url.to_owned(),
    };
    if let Some(pos) = filename.find('?') {
        filename.truncate(pos);
    }
    filename
}

fn parse_int_prefix(s: &str) -> Option<i32> {
    let bytes = s.as_bytes();
    let mut end = 0;
    if bytes.first() == Some(&b'-') || bytes.first() == Some(&b'+') {
        end = 1;
    }
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == 0 || (end == 1 && !bytes[0].is_ascii_digit()) {
        return None;
    }
    s[..end].parse().ok()
}

// ---------------------------------------------------------------------------
// minimal HTTP stand-in
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct UrlParts {
    scheme: String,
    user: String,
    password: String,
    host: String,
    port: u16,
    path: String,
}

fn format_host(host: &str) -> String {
    if host.contains(':') {
        format!("[{host}]")
    } else {
        host.to_owned()
    }
}

fn parse_url(url: &str) -> Result<UrlParts> {
    let scheme_end = url.find("://").ok_or_else(|| Error::new("invalid URL: no scheme"))?;
    let scheme = &url[..scheme_end];
    if scheme != "http" && scheme != "https" && scheme != "file" {
        return Err(Error::new(format!("unsupported URL scheme: {scheme}")));
    }

    let mut rest = &url[scheme_end + 3..];
    let mut user = String::new();
    let mut password = String::new();
    if let Some(at) = rest.find('@') {
        let auth = &rest[..at];
        if let Some(colon) = auth.find(':') {
            user = auth[..colon].to_owned();
            password = auth[colon + 1..].to_owned();
        } else {
            user = auth.to_owned();
        }
        rest = &rest[at + 1..];
    }

    let (host_part, path) = match rest.find('/') {
        Some(slash) => (&rest[..slash], rest[slash..].to_owned()),
        None => (rest, "/".to_owned()),
    };

    let mut host = host_part.to_owned();
    let mut port_str = String::new();
    if host.starts_with('[') {
        let close = host.find(']').ok_or_else(|| {
            Error::new(format!("invalid IPv6 URL authority: {host}"))
        })?;
        let after = &host[close + 1..];
        if let Some(stripped) = after.strip_prefix(':') {
            port_str = stripped.to_owned();
        }
        host = host[1..close].to_owned();
    } else if let Some(colon) = host.find(':') {
        port_str = host[colon + 1..].to_owned();
        host.truncate(colon);
    }

    let port = if !port_str.is_empty() {
        port_str
            .parse::<u16>()
            .map_err(|_| Error::new(format!("invalid URL port: {port_str}")))?
    } else {
        match scheme {
            "http" => 80,
            "https" => 443,
            _ => 0,
        }
    };

    Ok(UrlParts {
        scheme: scheme.to_owned(),
        user,
        password,
        host,
        port,
        path,
    })
}

/// Resolve a redirect `Location` against the URL it came from.
fn resolve_location(base_url: &str, location: &str) -> Result<String> {
    if location.contains("://") {
        return Ok(location.to_owned());
    }
    let base = parse_url(base_url)?;
    if let Some(rest) = location.strip_prefix("//") {
        return Ok(format!("{}:{rest}", base.scheme));
    }
    let origin = format!(
        "{}://{}:{}",
        base.scheme,
        format_host(&base.host),
        base.port
    );
    let path = base.path.split('?').next().unwrap_or("");
    if let Some(rest) = location.strip_prefix('/') {
        return Ok(format!("{origin}/{rest}"));
    }
    if location.starts_with('?') {
        return Ok(format!("{origin}{path}{location}"));
    }
    let dir_end = path.rfind('/').map_or(0, |i| i + 1);
    Ok(format!("{origin}{}{location}", &path[..dir_end]))
}

fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

fn decode_chunked(body: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0;
    loop {
        let rel = find_subslice(&body[pos..], b"\r\n")?;
        let end = pos + rel;
        let text = std::str::from_utf8(&body[pos..end]).ok()?;
        let size_text = text.split(';').next()?.trim();
        let size = usize::from_str_radix(size_text, 16).ok()?;
        pos = end + 2;
        if size == 0 {
            return Some(out);
        }
        if pos + size > body.len() {
            return None;
        }
        out.extend_from_slice(&body[pos..pos + size]);
        pos += size + 2;
    }
}

fn header_value<'a>(headers: &'a HeaderList, name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

#[derive(Debug)]
struct HttpResponse {
    status: i32,
    headers: HeaderList,
    body: Vec<u8>,
}

fn parse_response(raw: Vec<u8>, is_head: bool) -> Result<HttpResponse> {
    let split = find_subslice(&raw, b"\r\n\r\n")
        .ok_or_else(|| Error::new("malformed HTTP response: no header terminator"))?;
    let head = std::str::from_utf8(&raw[..split])
        .map_err(|_| Error::new("malformed HTTP response: invalid header encoding"))?;
    let mut lines = head.split("\r\n");
    let status = lines
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<i32>().ok())
        .ok_or_else(|| Error::new("malformed HTTP response: no status code"))?;
    let mut headers = HeaderList::new();
    for line in lines {
        if let Some((key, value)) = line.split_once(':') {
            headers.push((key.trim().to_owned(), value.trim().to_owned()));
        }
    }
    let mut body = raw[split + 4..].to_vec();
    if header_value(&headers, "Transfer-Encoding")
        .map_or(false, |v| v.to_ascii_lowercase().contains("chunked"))
    {
        body = decode_chunked(&body)
            .ok_or_else(|| Error::new("malformed HTTP response: bad chunked body"))?;
    } else if let Some(len) = header_value(&headers, "Content-Length")
        .and_then(|v| v.parse::<usize>().ok())
    {
        body.truncate(len);
    }
    if is_head {
        body.clear();
    }
    Ok(HttpResponse {
        status,
        headers,
        body,
    })
}

/// Configuration plus the URL of the current hop. Redirects rebuild the
/// client, so each hop is checked when `resolve_host` pins an address.
struct HttpClient<'a> {
    base_url: String,
    headers: HeaderList,
    timeout: Option<Duration>,
    follow_location: bool,
    resolve_host: Option<&'a dyn Fn(&str) -> String>,
}

impl<'a> HttpClient<'a> {
    fn new(url: &str) -> Result<(HttpClient<'static>, UrlParts)> {
        let parts = parse_url(url)?;
        validate_parts(&parts)?;
        let cli = HttpClient {
            base_url: url.to_owned(),
            headers: HeaderList::new(),
            timeout: None,
            follow_location: true,
            resolve_host: None,
        };
        Ok((cli, parts))
    }

    fn from_settings<'b>(
        url: &str,
        headers: &HeaderList,
        timeout: Option<Duration>,
        follow_location: bool,
        resolve_host: Option<&'b dyn Fn(&str) -> String>,
    ) -> Result<(HttpClient<'b>, UrlParts)> {
        let parts = parse_url(url)?;
        validate_parts(&parts)?;
        let cli = HttpClient {
            base_url: url.to_owned(),
            headers: headers.clone(),
            timeout,
            follow_location,
            resolve_host,
        };
        Ok((cli, parts))
    }

    fn add_header(&mut self, key: &str, value: String) {
        if header_value(&self.headers, key).is_none() {
            self.headers.push((key.to_owned(), value));
        }
    }

    fn has_header(&self, key: &str) -> bool {
        header_value(&self.headers, key).is_some()
    }

    /// GET or HEAD, following redirects when enabled.
    fn request(&self, method: &str, extra: &HeaderList) -> Result<HttpResponse> {
        let mut url = self.base_url.clone();
        for _ in 0..MAX_REDIRECTS {
            let (cli, parts) = HttpClient::from_settings(
                &url,
                &self.headers,
                self.timeout,
                self.follow_location,
                self.resolve_host,
            )?;
            let res = cli.raw_request(method, &parts.path, extra)?;
            if !self.follow_location || !(300..400).contains(&res.status) {
                return Ok(res);
            }
            let Some(location) = header_value(&res.headers, "Location").map(str::to_owned) else {
                return Ok(res);
            };
            url = resolve_location(&url, &location)?;
        }
        Err(Error::new("error: too many redirects"))
    }

    fn raw_request(&self, method: &str, path: &str, extra: &HeaderList) -> Result<HttpResponse> {
        let parts = parse_url(&self.base_url)?;
        let is_head = method.eq_ignore_ascii_case("HEAD");

        if parts.scheme == "file" {
            let mut status = 200;
            let mut body = Vec::new();
            match fs::read(&parts.path) {
                Ok(bytes) => {
                    if !is_head {
                        body = bytes;
                    }
                }
                Err(_) => status = 404,
            }
            let size = if is_head {
                fs::metadata(&parts.path).map(|m| m.len()).unwrap_or(0)
            } else {
                body.len() as u64
            };
            return Ok(HttpResponse {
                status,
                headers: vec![("Content-Length".to_owned(), size.to_string())],
                body,
            });
        }

        let target_host = match self.resolve_host {
            Some(resolve) => resolve(&parts.host),
            None => parts.host.clone(),
        };
        let address = format!("{}:{}", target_host, parts.port);
        let mut addresses = address
            .to_socket_addrs()
            .map_err(|e| Error::new(format!("cannot resolve {address}: {e}")))?;
        let socket = addresses
            .next()
            .ok_or_else(|| Error::new(format!("cannot resolve {address}")))?;
        let mut stream = match self.timeout {
            Some(timeout) => TcpStream::connect_timeout(&socket, timeout)
                .map_err(|e| Error::new(format!("cannot connect to {address}: {e}")))?,
            None => TcpStream::connect(socket)
                .map_err(|e| Error::new(format!("cannot connect to {address}: {e}")))?,
        };
        if let Some(timeout) = self.timeout {
            let _ = stream.set_read_timeout(Some(timeout));
            let _ = stream.set_write_timeout(Some(timeout));
        }

        let mut request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n",
            format_host(&parts.host),
            parts.port
        );
        for (key, value) in self.headers.iter().chain(extra.iter()) {
            request.push_str(&format!("{key}: {value}\r\n"));
        }
        if !parts.user.is_empty() {
            let credentials = format!("{}:{}", parts.user, parts.password);
            request.push_str(&format!(
                "Authorization: Basic {}\r\n",
                base64_encode(credentials.as_bytes())
            ));
        }
        request.push_str("\r\n");
        stream
            .write_all(request.as_bytes())
            .map_err(|e| Error::new(format!("cannot send request: {e}")))?;

        let mut raw = Vec::new();
        stream
            .read_to_end(&mut raw)
            .map_err(|e| Error::new(format!("cannot read response: {e}")))?;
        parse_response(raw, is_head)
    }
}

fn validate_parts(parts: &UrlParts) -> Result<()> {
    if parts.host.is_empty() && parts.scheme != "file" {
        return Err(Error::new("error: invalid URL format"));
    }
    if parts.scheme == "https" {
        return Err(Error::new(
            "HTTPS is not supported. Please rebuild with one of:\n\
              -DLLAMA_BUILD_BORINGSSL=ON\n\
              -DLLAMA_BUILD_LIBRESSL=ON\n\
              -DLLAMA_OPENSSL=ON (default, requires OpenSSL dev files installed)",
        ));
    }
    Ok(())
}

/// `common_remote_get_content`.
pub fn remote_get_content(url: &str, params: &RemoteParams) -> Result<(i64, Vec<u8>)> {
    let mut url = url.to_owned();
    for hop in 0..MAX_REDIRECTS {
        let (mut cli, _parts) = HttpClient::from_settings(
            &url,
            &params.headers,
            if params.timeout > 0 {
                Some(Duration::from_secs(params.timeout))
            } else {
                None
            },
            params.resolve_host.is_none(),
            params.resolve_host.as_deref(),
        )?;
        if !cli.has_header("User-Agent") {
            cli.add_header("User-Agent", USER_AGENT.to_owned());
        }

        let mut res = cli.request("GET", &HeaderList::new())?;
        if params.resolve_host.is_some()
            && (300..400).contains(&res.status)
            && header_value(&res.headers, "Location").is_some()
        {
            if hop + 1 >= MAX_REDIRECTS {
                return Err(Error::new("error: too many redirects"));
            }
            let location = header_value(&res.headers, "Location").unwrap().to_owned();
            url = resolve_location(&url, &location)?;
            continue;
        }
        if params.max_size > 0 && res.body.len() > params.max_size {
            res.body.truncate(params.max_size);
        }
        return Ok((res.status as i64, res.body));
    }
    Err(Error::new("error: too many redirects"))
}

// ---------------------------------------------------------------------------
// cross-process file lock
// ---------------------------------------------------------------------------

/// std has no `flock`; an atomic `create_new` of the lock file stands in.
/// Unlike the C++ lock, the file is removed on release, so a crashed process
/// can leave a stale lock. Upgrade to an OS lock when one is available.
struct FileLock {
    lock_path: String,
    acquired: bool,
}

impl FileLock {
    fn new(lock_path: String) -> Self {
        Self {
            lock_path,
            acquired: false,
        }
    }

    fn acquire<F: Fn() -> bool>(&mut self, keep_waiting: F) -> bool {
        if self.lock_path.is_empty() {
            self.acquired = true;
            return true;
        }
        let path = Path::new(&self.lock_path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && fs::create_dir_all(parent).is_err() {
                eprintln!(
                    "common_file_lock: unable to create lock directory '{}'",
                    parent.display()
                );
                return false;
            }
        }
        loop {
            match OpenOptions::new().write(true).create_new(true).open(path) {
                Ok(_) => {
                    self.acquired = true;
                    return true;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if !keep_waiting() {
                        return false;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => {
                    eprintln!("common_file_lock: cannot open lock file: {e}");
                    return false;
                }
            }
        }
    }

    fn close(&mut self) {
        if self.acquired {
            let _ = fs::remove_file(&self.lock_path);
            self.acquired = false;
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        self.close();
    }
}

// ---------------------------------------------------------------------------
// progress bar
// ---------------------------------------------------------------------------

static PROGRESS_LINES: Mutex<Vec<bool>> = Mutex::new(Vec::new());

struct ProgressBarState {
    filename: String,
    len: usize,
    line: Option<usize>,
}

/// Wraps the caller's callback, as the C++ `ProgressBar` does.
pub struct ProgressBar<'a> {
    next: Option<&'a dyn DownloadCallback>,
    state: Mutex<ProgressBarState>,
}

impl<'a> ProgressBar<'a> {
    pub fn new(next: Option<&'a dyn DownloadCallback>) -> Self {
        Self {
            next,
            state: Mutex::new(ProgressBarState {
                filename: String::new(),
                len: 0,
                line: None,
            }),
        }
    }

    fn cleanup(&self) {
        if let Ok(mut lines) = PROGRESS_LINES.lock() {
            if let Some(line) = self.state.lock().unwrap().line.take() {
                if line < lines.len() {
                    lines[line] = false;
                }
                while lines.last() == Some(&false) {
                    lines.pop();
                }
            }
        }
    }
}

impl DownloadCallback for ProgressBar<'_> {
    fn on_start(&self, p: &DownloadProgress) {
        if let Some(next) = self.next {
            next.on_start(p);
        }
        let mut filename = url_get_filename(&p.url);
        let mut len = 0usize;
        let mut cut = None;
        for (i, byte) in filename.as_bytes().iter().enumerate() {
            if byte & 0xC0 != 0x80 {
                if len == 39 {
                    cut = Some(i);
                    break;
                }
                len += 1;
            }
        }
        if let Some(i) = cut {
            filename.truncate(i);
            filename.push('…');
        }
        let mut state = self.state.lock().unwrap();
        state.filename = filename;
        state.len = len;
    }

    fn on_update(&self, p: &DownloadProgress) {
        if let Some(next) = self.next {
            next.on_update(p);
        }
        if p.total == 0 || !stdout_is_tty() {
            return;
        }
        let Ok(mut lines) = PROGRESS_LINES.lock() else {
            return;
        };
        let mut state = self.state.lock().unwrap();
        let line = match state.line {
            Some(line) => line,
            None => {
                let line = lines.len();
                lines.push(true);
                state.line = Some(line);
                print!("\n");
                line
            }
        };
        let lines_up = lines.len() - 1 - line;
        let bar = (55 - state.len.min(39)) * 2;
        let pct = (100 * p.downloaded) / p.total;
        let pos = (bar * p.downloaded) / p.total;
        let mut out = String::new();
        if lines_up > 0 {
            out.push_str(&format!("\x1b[{lines_up}A"));
        }
        out.push_str(&format!("\rDownloading {} ", state.filename));
        for i in (0..bar).step_by(2) {
            out.push_str(if i + 1 < pos {
                "─"
            } else if i < pos {
                "╴"
            } else {
                " "
            });
        }
        out.push_str(&format!("{pct:>4}%\x1b[K"));
        if lines_up > 0 {
            out.push_str(&format!("\x1b[{lines_up}B"));
        }
        out.push('\r');
        print!("{out}");
        let _ = std::io::stdout().flush();
        if p.downloaded == p.total {
            drop(state);
            drop(lines);
            self.cleanup();
        }
    }

    fn on_done(&self, p: &DownloadProgress, ok: bool) {
        if let Some(next) = self.next {
            next.on_done(p, ok);
        }
        self.cleanup();
    }

    fn is_cancelled(&self) -> bool {
        self.next.map_or(false, |next| next.is_cancelled())
    }
}

// ---------------------------------------------------------------------------
// single-file download
// ---------------------------------------------------------------------------

fn pull_file(
    cli: &HttpClient<'_>,
    path_tmp: &str,
    supports_ranges: bool,
    p: &mut DownloadProgress,
    callback: Option<&dyn DownloadCallback>,
) -> bool {
    let mut ofs = match OpenOptions::new().append(true).create(true).open(path_tmp) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("common_pull_file: error opening local file for writing: {path_tmp}: {e}");
            return false;
        }
    };

    let mut headers = HeaderList::new();
    if supports_ranges && p.downloaded > 0 {
        headers.push((
            "Range".to_owned(),
            format!("bytes={}-", p.downloaded),
        ));
    }

    let res = match cli.request("GET", &headers) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("common_pull_file: download failed: {e}");
            return false;
        }
    };

    if p.downloaded > 0 && res.status != 206 {
        eprintln!(
            "common_pull_file: server did not respond with 206 Partial Content for a resume request. Status: {}",
            res.status
        );
        if res.status == 416 || res.status == 200 {
            drop(ofs);
            let _ = fs::remove_file(path_tmp);
        }
        return false;
    }
    if p.downloaded == 0 && res.status != 200 {
        eprintln!(
            "common_pull_file: download received non-successful status code: {}",
            res.status
        );
        return false;
    }
    if p.total == 0 {
        if let Some(len) = header_value(&res.headers, "Content-Length")
            .and_then(|v| v.parse::<usize>().ok())
        {
            p.total = p.downloaded + len;
        } else if header_value(&res.headers, "Content-Length").is_some() {
            eprintln!("common_pull_file: invalid Content-Length header");
        }
    }

    let mut progress_step = 0usize;
    for chunk in res.body.chunks(64 * 1024) {
        if ofs.write_all(chunk).is_err() {
            eprintln!("common_pull_file: error writing to file: {path_tmp}");
            return false;
        }
        p.downloaded += chunk.len();
        progress_step += chunk.len();
        if progress_step >= p.total / 1000 || p.downloaded == p.total {
            if let Some(cb) = callback {
                cb.on_update(p);
                if cb.is_cancelled() && (p.total == 0 || p.downloaded != p.total) {
                    return false;
                }
            }
            progress_step = 0;
        }
    }
    if ofs.flush().is_err() {
        eprintln!("common_pull_file: error closing file: {path_tmp}");
        return false;
    }
    true
}

fn finish(status: i32, lock: &mut FileLock, p: &DownloadProgress, opts: &DownloadOpts<'_>) -> i32 {
    lock.close();
    if let Some(cb) = opts.callback {
        cb.on_done(p, is_http_status_ok(status));
    }
    status
}

fn download_file_single_online(
    url: &str,
    path: &str,
    opts: &DownloadOpts<'_>,
    skip_etag: bool,
    lock_path: &str,
    alt_path: &str,
) -> i32 {
    let path_temporary = format!("{path}{PARTIAL_SUFFIX}");
    let path_lock = if lock_path.is_empty() {
        format!("{path}.lock")
    } else {
        lock_path.to_owned()
    };

    let mut p = DownloadProgress {
        url: url.to_owned(),
        ..DownloadProgress::default()
    };
    if let Some(cb) = opts.callback {
        cb.on_start(&p);
    }

    let mut lock = FileLock::new(path_lock);
    let cache_hit = || {
        Path::new(path).exists() || (!alt_path.is_empty() && Path::new(alt_path).exists())
    };

    // The completion callback may take the same lock, so every exit goes
    // through finish().
    if skip_etag && cache_hit() {
        return finish(304, &mut lock, &p, opts);
    }

    let waited = Cell::new(false);
    if !lock.acquire(|| {
        if !waited.get() {
            eprintln!(
                "common_download_file_single_online: file '{}' is being downloaded by another process, waiting...",
                url_get_filename(url)
            );
            waited.set(true);
        }
        !opts.callback.map_or(false, |cb| cb.is_cancelled())
    }) {
        return finish(-1, &mut lock, &p, opts);
    }

    let file_exists = cache_hit();
    if file_exists && skip_etag {
        return finish(304, &mut lock, &p, opts);
    }

    let (mut cli, _parts) = match HttpClient::new(url) {
        Ok(client) => client,
        Err(e) => {
            eprintln!("common_download_file_single_online: {e}");
            return finish(-1, &mut lock, &p, opts);
        }
    };
    for (key, value) in &opts.headers {
        cli.add_header(key, value.clone());
    }
    if !cli.has_header("User-Agent") {
        cli.add_header("User-Agent", USER_AGENT.to_owned());
    }
    if !opts.bearer_token.is_empty() {
        cli.add_header(
            "Authorization",
            format!("Bearer {}", opts.bearer_token),
        );
    }

    let last_etag = if file_exists {
        read_etag(path)
    } else {
        String::new()
    };

    let head = cli.request("HEAD", &HeaderList::new()).ok();
    let head_status = head.as_ref().map_or(-1, |res| res.status);
    if !(200..300).contains(&head_status) {
        if file_exists {
            return finish(304, &mut lock, &p, opts);
        }
        return finish(head_status, &mut lock, &p, opts);
    }
    let head = head.unwrap();

    let etag = header_value(&head.headers, "ETag").unwrap_or("").to_owned();
    if let Some(len) = header_value(&head.headers, "Content-Length")
        .and_then(|v| v.parse::<usize>().ok())
    {
        p.total = len;
    }
    let supports_ranges = header_value(&head.headers, "Accept-Ranges")
        .map_or(false, |v| v != "none");

    if file_exists {
        if etag.is_empty() {
            return finish(304, &mut lock, &p, opts);
        }
        if !last_etag.is_empty() && last_etag == etag {
            return finish(304, &mut lock, &p, opts);
        }
        if fs::remove_file(path).is_err() {
            eprintln!("common_download_file_single_online: unable to delete file: {path}");
            return finish(-1, &mut lock, &p, opts);
        }
    }

    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }

    if !skip_etag && !etag.is_empty() {
        if Path::new(&path_temporary).exists() && read_etag(&path_temporary) != etag {
            eprintln!(
                "common_download_file_single_online: partial file does not match the server etag, discarding it: {path_temporary}"
            );
            if fs::remove_file(&path_temporary).is_err() {
                return finish(-1, &mut lock, &p, opts);
            }
        }
        let _ = write_etag(&path_temporary, &etag);
    }

    if let Some(cb) = opts.callback {
        p.downloaded = if supports_ranges && Path::new(&path_temporary).exists() {
            fs::metadata(&path_temporary).map(|m| m.len() as usize).unwrap_or(0)
        } else {
            0
        };
        cb.on_update(&p);
    }

    let mut success = false;
    let mut delay = RETRY_DELAY_SECONDS;
    for i in 0..MAX_ATTEMPTS {
        if opts.callback.map_or(false, |cb| cb.is_cancelled()) {
            break;
        }
        if i > 0 {
            std::thread::sleep(Duration::from_secs(delay));
            delay *= RETRY_DELAY_SECONDS;
        }

        let mut existing_size = 0usize;
        if Path::new(&path_temporary).exists() {
            if supports_ranges {
                existing_size = fs::metadata(&path_temporary)
                    .map(|m| m.len() as usize)
                    .unwrap_or(0);
            } else if fs::remove_file(&path_temporary).is_err() {
                break;
            }
        }
        p.downloaded = existing_size;

        if pull_file(
            &cli,
            &path_temporary,
            supports_ranges,
            &mut p,
            opts.callback,
        ) {
            if fs::rename(&path_temporary, path).is_err() {
                break;
            }
            if !etag.is_empty() && !skip_etag {
                let _ = write_etag(path, &etag);
                let _ = fs::remove_file(format!("{path_temporary}.etag"));
            }
            success = true;
            break;
        }
    }
    if !success {
        eprintln!("common_download_file_single_online: download failed after {MAX_ATTEMPTS} attempts");
    }
    finish(
        if success { head.status } else { -1 },
        &mut lock,
        &p,
        opts,
    )
}

/// `common_download_file_single`: status code, or -1 on error.
pub fn download_file_single(
    url: &str,
    path: &str,
    opts: &DownloadOpts<'_>,
    skip_etag: bool,
    lock_path: &str,
    alt_path: &str,
) -> i32 {
    if !opts.offline {
        let pb = ProgressBar::new(opts.callback);
        let mut online_opts = opts.clone();
        online_opts.callback = Some(&pb);
        return download_file_single_online(
            url,
            path,
            &online_opts,
            skip_etag,
            lock_path,
            alt_path,
        );
    }

    if !Path::new(path).exists() {
        eprintln!("common_download_file_single: required file is not available in cache (offline mode): {path}");
        return -1;
    }

    let p = DownloadProgress {
        url: url.to_owned(),
        cached: true,
        ..DownloadProgress::default()
    };
    if let Some(cb) = opts.callback {
        cb.on_start(&p);
        cb.on_done(&p, true);
    }
    304
}

// ---------------------------------------------------------------------------
// GGUF split names and model selection
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct GgufSplitInfo {
    prefix: String,
    tag: String,
    index: i32,
    count: i32,
}

impl Default for GgufSplitInfo {
    fn default() -> Self {
        Self {
            prefix: String::new(),
            tag: String::new(),
            index: 1,
            count: 1,
        }
    }
}

/// Hand-coded equivalent of `^(.+)-([0-9]{5})-of-([0-9]{5})$` followed by
/// `[-.]([A-Z0-9_]+)$`, both case insensitive where the C++ regexes are.
fn get_gguf_split_info(path: &str) -> GgufSplitInfo {
    let mut prefix = path.to_owned();
    if !string_remove_suffix(&mut prefix, ".gguf") {
        return GgufSplitInfo::default();
    }

    let mut info = GgufSplitInfo {
        prefix: prefix.clone(),
        ..GgufSplitInfo::default()
    };
    let chars: Vec<char> = prefix.chars().collect();
    let n = chars.len();
    if n >= 16
        && chars[n - 15] == '-'
        && chars[n - 9] == '-'
        && chars[n - 8].eq_ignore_ascii_case(&'o')
        && chars[n - 7].eq_ignore_ascii_case(&'f')
        && chars[n - 6] == '-'
        && chars[n - 5..n].iter().all(|c| c.is_ascii_digit())
        && chars[n - 14..n - 9].iter().all(|c| c.is_ascii_digit())
    {
        let index: String = chars[n - 14..n - 9].iter().collect();
        let count: String = chars[n - 5..n].iter().collect();
        info.index = index.parse().unwrap_or(1);
        info.count = count.parse().unwrap_or(1);
        info.prefix = chars[..n - 15].iter().collect();
    }

    let tag_chars: Vec<char> = info.prefix.chars().collect();
    let mut start = tag_chars.len();
    while start > 0
        && (tag_chars[start - 1].is_ascii_alphanumeric() || tag_chars[start - 1] == '_')
    {
        start -= 1;
    }
    if start > 0
        && start < tag_chars.len()
        && matches!(tag_chars[start - 1], '-' | '.')
    {
        info.tag = tag_chars[start..]
            .iter()
            .collect::<String>()
            .to_ascii_uppercase();
    }
    info
}

/// Q4_0 -> 4, F16 -> 16, NVFP4 -> 4, Q8_K_M -> 8.
fn extract_quant_bits(filename: &str) -> i32 {
    let split = get_gguf_split_info(filename);
    let Some(pos) = split.tag.find(|c: char| c.is_ascii_digit()) else {
        return 0;
    };
    parse_int_prefix(&split.tag[pos..]).unwrap_or(0)
}

fn get_split_files(files: &HfFiles, file: &HfFile) -> HfFiles {
    let split = get_gguf_split_info(&file.path);
    if split.count <= 1 {
        return vec![file.clone()];
    }
    files
        .iter()
        .filter(|f| {
            let other = get_gguf_split_info(&f.path);
            other.count == split.count && other.prefix == split.prefix
        })
        .cloned()
        .collect()
}

fn find_best_sibling(files: &HfFiles, model: &str, keyword: &str, tag: &str) -> HfFile {
    let tag_upper = tag.to_ascii_uppercase();
    let model_bits = if tag_upper.is_empty() {
        extract_quant_bits(model)
    } else {
        parse_int_prefix(tag_upper.trim_start_matches(|c: char| !c.is_ascii_digit()))
            .unwrap_or(0)
    };
    let model_parts: Vec<&str> = model.split('/').collect();
    let model_dir = model_parts.len().saturating_sub(1);

    let mut best = HfFile::default();
    let mut best_depth = 0usize;
    let mut best_diff = 0i32;
    let mut best_exact = false;
    let mut found = false;

    for f in files {
        if !string_ends_with(&f.path, ".gguf") || !f.path.contains(keyword) {
            continue;
        }
        let sib_parts: Vec<&str> = f.path.split('/').collect();
        let sib_dir = sib_parts.len().saturating_sub(1);
        if sib_dir > model_dir || (0..sib_dir).any(|i| model_parts[i] != sib_parts[i]) {
            continue;
        }
        let depth = sib_dir;
        let bits = extract_quant_bits(&f.path);
        let diff = (bits - model_bits).abs();
        let path_upper = f.path.to_ascii_uppercase();
        let exact = !tag_upper.is_empty() && path_upper.contains(&format!("-{tag_upper}."));

        if !found
            || depth > best_depth
            || (depth == best_depth && exact && !best_exact)
            || (depth == best_depth && exact == best_exact && diff < best_diff)
        {
            best = f.clone();
            best_depth = depth;
            best_diff = diff;
            best_exact = exact;
            found = true;
        }
    }
    best
}

pub fn download_sidecars() -> [SidecarKind; 5] {
    SidecarKind::ALL
}

pub fn opts_add_sidecar(opts: &mut DownloadOpts<'_>, kind: &str) -> bool {
    for s in SidecarKind::ALL {
        if kind == s.name() {
            *s.opt(opts) = true;
            return true;
        }
    }
    false
}

pub fn plan_sidecars(plan: &DownloadHfPlan) -> Vec<String> {
    SidecarKind::ALL
        .iter()
        .filter(|s| !s.plan_file(plan).path.is_empty())
        .map(|s| s.name().to_owned())
        .collect()
}

pub fn plan_partials(plan: &DownloadHfPlan) -> Vec<DownloadPartial> {
    let mut out = Vec::new();
    let mut add = |kind: &str, f: &HfFile| {
        if !f.local_path.is_empty() && f.local_path != f.final_path {
            out.push(DownloadPartial {
                kind: kind.to_owned(),
                path: format!("{}{PARTIAL_SUFFIX}", f.local_path),
            });
        }
    };
    for f in &plan.model_files {
        add("", f);
    }
    for s in SidecarKind::ALL {
        add(s.name(), s.plan_file(plan));
    }
    out
}

fn name_is_sidecar(name: &str) -> bool {
    SidecarKind::ALL
        .iter()
        .any(|s| name.contains(s.keyword()))
}

fn gguf_filename_is_model(filepath: &str) -> bool {
    if !string_ends_with(filepath, ".gguf") {
        return false;
    }
    let filename = filepath.rsplit('/').next().unwrap_or(filepath);
    !filename.contains("imatrix") && !name_is_sidecar(filename)
}

fn tag_matches(path: &str, tag: &str) -> bool {
    if tag.is_empty() {
        return false;
    }
    let lower_path = path.to_ascii_lowercase();
    let lower_tag = tag.to_ascii_lowercase();
    for (idx, _) in lower_path.match_indices(&lower_tag) {
        let before_ok = idx == 0
            || matches!(lower_path.as_bytes()[idx - 1], b'-' | b'.' | b'/');
        let after_ok = matches!(
            lower_path.as_bytes().get(idx + lower_tag.len()),
            Some(b'.') | Some(b'-')
        );
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

fn find_best_model(files: &HfFiles, tag: &str) -> HfFile {
    let tags: Vec<String> = if tag.is_empty() {
        vec!["Q4_K_M".to_owned(), "Q8_0".to_owned()]
    } else {
        vec![tag.to_owned()]
    };

    for t in &tags {
        for f in files {
            if gguf_filename_is_model(&f.path) && tag_matches(&f.path, t) {
                let split = get_gguf_split_info(&f.path);
                if split.count > 1 && split.index != 1 {
                    continue;
                }
                return f.clone();
            }
        }
    }

    if tag.is_empty() {
        for f in files {
            if gguf_filename_is_model(&f.path) {
                let split = get_gguf_split_info(&f.path);
                if split.count > 1 && split.index != 1 {
                    continue;
                }
                return f.clone();
            }
        }
    }
    HfFile::default()
}

fn list_available_gguf_files(files: &HfFiles) {
    eprintln!("Available GGUF files:");
    for f in files {
        if string_ends_with(&f.path, ".gguf") {
            eprintln!(" - {}", f.path);
        }
    }
}

pub fn get_hf_plan(model: &ParamsModel, opts: &DownloadOpts<'_>) -> Result<DownloadHfPlan> {
    let mut plan = DownloadHfPlan::default();
    let (repo, tag) = split_repo_tag(&model.hf_repo)?;

    let mut all = if !opts.offline {
        get_repo_files(&repo, &opts.bearer_token)
    } else {
        Vec::new()
    };
    if all.is_empty() {
        all = get_cached_files(&repo);
    }
    if all.is_empty() {
        return Ok(plan);
    }

    for f in &all {
        if f.path == "preset.ini" {
            plan.preset = f.clone();
            return Ok(plan);
        }
    }

    let mut primary = HfFile::default();
    if !model.hf_file.is_empty() {
        for f in &all {
            if f.path == model.hf_file {
                primary = f.clone();
                break;
            }
        }
        if primary.path.is_empty() {
            eprintln!(
                "common_download_get_hf_plan: file '{}' not found in repository",
                model.hf_file
            );
            list_available_gguf_files(&all);
            return Ok(plan);
        }
    } else {
        primary = find_best_model(&all, &tag);
        let want_draft = SidecarKind::ALL
            .iter()
            .any(|s| s.is_draft() && s.opt_enabled(opts));
        if primary.path.is_empty() && !want_draft {
            eprintln!("common_download_get_hf_plan: no GGUF files found in repository {repo}");
            list_available_gguf_files(&all);
            return Ok(plan);
        }
    }

    if !primary.path.is_empty() {
        plan.primary = primary.clone();
        plan.model_files = get_split_files(&all, &primary);
    }

    for s in SidecarKind::ALL {
        if !s.opt_enabled(opts) {
            continue;
        }
        let found = if s.is_draft() {
            find_best_sibling(&all, &primary.path, s.keyword(), &tag)
        } else if !primary.path.is_empty() {
            find_best_sibling(&all, &primary.path, s.keyword(), "")
        } else {
            HfFile::default()
        };
        *s.plan_file_mut(&mut plan) = found;
    }

    let has_draft = SidecarKind::ALL
        .iter()
        .any(|s| s.is_draft() && !s.plan_file(&plan).local_path.is_empty());
    if primary.path.is_empty() && !has_draft {
        eprintln!("common_download_get_hf_plan: no GGUF files found in repository {repo}");
        list_available_gguf_files(&all);
    }
    Ok(plan)
}

// ---------------------------------------------------------------------------
// task execution, parts, Docker
// ---------------------------------------------------------------------------

pub fn run_tasks(tasks: &[DownloadTask<'_>]) -> Result<()> {
    // std-only stand-in: run sequentially so the shared callbacks need no
    // cloning and no 'static bound.
    for task in tasks {
        let status = download_file_single(
            &task.url,
            &task.local_path,
            &task.opts,
            task.is_hf,
            &task.lock_path,
            &task.alt_path,
        );
        if !is_http_status_ok(status) {
            return Err(Error::new(format!(
                "Download '{}' failed with status code: {status}",
                task.url
            )));
        }
    }
    Ok(())
}

pub fn get_all_parts(url: &str) -> Vec<String> {
    let split = get_gguf_split_info(url);
    if split.count <= 1 {
        return vec![url.to_owned()];
    }
    (1..=split.count)
        .map(|i| format!("{}-{:05}-of-{:05}.gguf", split.prefix, i, split.count))
        .collect()
}

fn docker_get_token(repo: &str) -> Result<String> {
    let url = format!(
        "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
    );
    let (status, body) = remote_get_content(&url, &RemoteParams::default())?;
    if status != 200 {
        return Err(Error::new(format!(
            "Failed to get Docker registry token, HTTP code: {status}"
        )));
    }
    let text = String::from_utf8_lossy(&body);
    let json = Json::parse(&text)?;
    match json.at("token")? {
        Json::String(token) => Ok(token.clone()),
        _ => Err(Error::new(
            "Docker registry token response missing 'token' field",
        )),
    }
}

fn validate_oci_digest(digest: &str) -> Result<String> {
    let Some(hex) = digest.strip_prefix("sha256:") else {
        return Err(Error::new(format!(
            "Invalid OCI digest format received in manifest: {digest}"
        )));
    };
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::new(format!(
            "Invalid OCI digest format received in manifest: {digest}"
        )));
    }
    Ok(format!("sha256:{}", hex.to_ascii_lowercase()))
}

pub fn docker_resolve_model(docker: &str) -> Result<String> {
    let (mut repo, tag) = match docker.find(':') {
        Some(pos) => (docker[..pos].to_owned(), docker[pos + 1..].to_owned()),
        None => (docker.to_owned(), "latest".to_owned()),
    };
    if !docker.contains('/') {
        repo.insert_str(0, "ai/");
    }

    let token = docker_get_token(&repo)?;
    let url_prefix = format!("https://registry-1.docker.io/v2/{repo}");
    let manifest_url = format!("{url_prefix}/manifests/{tag}");
    let mut manifest_params = RemoteParams::default();
    manifest_params
        .headers
        .push(("Authorization".to_owned(), format!("Bearer {token}")));
    manifest_params.headers.push((
        "Accept".to_owned(),
        "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"
            .to_owned(),
    ));
    let (status, body) = remote_get_content(&manifest_url, &manifest_params)?;
    if status != 200 {
        return Err(Error::new(format!(
            "Failed to get Docker manifest, HTTP code: {status}"
        )));
    }
    let json = Json::parse(&String::from_utf8_lossy(&body))?;
    let mut gguf_digest = String::new();
    if let Ok(layers) = json.at("layers") {
        if layers.is_array() {
            for layer in layers.iter() {
                if let Ok(media_type) = layer.at("mediaType") {
                    if let Json::String(media_type) = media_type {
                        if media_type == "application/vnd.docker.ai.gguf.v3"
                            || media_type.contains("gguf")
                        {
                            if let Ok(Json::String(digest)) = layer.at("digest") {
                                gguf_digest = digest.clone();
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    if gguf_digest.is_empty() {
        return Err(Error::new("No GGUF layer found in Docker manifest"));
    }
    let gguf_digest = validate_oci_digest(&gguf_digest)?;

    let mut model_filename = repo.replace('/', "_");
    model_filename.push_str(&format!("_{tag}.gguf"));
    let local_path = fs_get_cache_file(&model_filename)?;
    let blob_url = format!("{url_prefix}/blobs/{gguf_digest}");
    let mut opts = DownloadOpts::default();
    opts.bearer_token = token;
    let status = download_file_single(
        &blob_url,
        &local_path.to_string_lossy(),
        &opts,
        false,
        "",
        "",
    );
    if !is_http_status_ok(status) {
        return Err(Error::new("Failed to download Docker Model"));
    }
    Ok(local_path.to_string_lossy().into_owned())
}

// ---------------------------------------------------------------------------
// cached-model listing and removal
// ---------------------------------------------------------------------------

fn is_listed_model(split: &GgufSplitInfo) -> bool {
    split.index == 1 && !split.tag.is_empty() && !name_is_sidecar(&split.prefix)
}

pub fn list_cached_models() -> Vec<CachedModelInfo> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for f in get_cached_files("") {
        let split = get_gguf_split_info(&f.path);
        if !is_listed_model(&split) {
            continue;
        }
        if seen.insert(format!("{}:{}", f.repo_id, split.tag)) {
            result.push(CachedModelInfo {
                repo: f.repo_id,
                tag: split.tag,
            });
        }
    }
    result
}

pub fn download_resolve_path(hf_repo_with_tag: &str, hf_file: &str) -> Result<String> {
    let (repo, tag) = split_repo_tag(hf_repo_with_tag)?;
    let files = get_cached_files(&repo);
    if files.is_empty() {
        return Ok(String::new());
    }
    if !hf_file.is_empty() {
        for f in &files {
            if f.path == hf_file {
                return Ok(f.local_path.clone());
            }
        }
        return Ok(String::new());
    }
    Ok(find_best_model(&files, &tag).local_path)
}

fn files_in_use(repo_id: &str, tag: &str) -> HashSet<String> {
    let mut opts = DownloadOpts::default();
    opts.offline = true;
    for s in SidecarKind::ALL {
        *s.opt(&mut opts) = true;
    }

    let mut tags = HashSet::new();
    let mut used = HashSet::new();
    for f in get_cached_files(repo_id) {
        let split = get_gguf_split_info(&f.path);
        if !is_listed_model(&split) || split.tag == tag || !tags.insert(split.tag.clone()) {
            continue;
        }
        let model = ParamsModel {
            hf_repo: format!("{repo_id}:{}", split.tag),
            ..ParamsModel::default()
        };
        let Ok(plan) = get_hf_plan(&model, &opts) else {
            continue;
        };
        for m in &plan.model_files {
            used.insert(m.local_path.clone());
        }
        for s in SidecarKind::ALL {
            let local = s.plan_file(&plan).local_path.clone();
            if !local.is_empty() {
                used.insert(local);
            }
        }
    }
    used
}

fn lexical_normal(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn is_valid_subpath(base: &Path, subpath: &Path) -> bool {
    if subpath.is_absolute() {
        return false;
    }
    let absolute = if base.is_absolute() {
        base.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(base)
    };
    lexical_normal(&absolute.join(subpath)).starts_with(lexical_normal(&absolute))
}

fn symlink_target(local: &Path, parent: &Path) -> PathBuf {
    let local = lexical_normal(local);
    let parent = lexical_normal(parent);
    let a: Vec<_> = local.components().collect();
    let b: Vec<_> = parent.components().collect();
    let mut common = 0;
    while common < a.len() && common < b.len() && a[common] == b[common] {
        common += 1;
    }
    let mut target = PathBuf::new();
    for _ in common..b.len() {
        target.push("..");
    }
    for component in &a[common..] {
        target.push(component.as_os_str());
    }
    target
}

#[cfg(unix)]
fn create_symlink(target: &Path, link: &Path) -> bool {
    std::os::unix::fs::symlink(target, link).is_ok()
}

#[cfg(not(unix))]
fn create_symlink(_target: &Path, _link: &Path) -> bool {
    false
}

fn remove_snapshot_entries(repo_id: &str, to_remove: &[PathBuf]) -> bool {
    let mut blobs_to_check = Vec::new();
    for path in to_remove {
        if let Ok(target) = fs::read_link(path) {
            let parent = path.parent().unwrap_or_else(|| Path::new(""));
            blobs_to_check.push(lexical_normal(&parent.join(target)));
        }
    }

    for path in to_remove {
        if let Err(e) = fs::remove_file(path) {
            eprintln!(
                "remove_snapshot_entries: failed to remove {}: {e}",
                path.display()
            );
        }
    }
    if blobs_to_check.is_empty() {
        return true;
    }

    let mut still_referenced = HashSet::new();
    for f in get_cached_files(repo_id) {
        if let Ok(target) = fs::read_link(&f.local_path) {
            let path = Path::new(&f.local_path);
            let parent = path.parent().unwrap_or_else(|| Path::new(""));
            still_referenced.insert(lexical_normal(&parent.join(target)).to_string_lossy().into_owned());
        }
    }

    for blob in blobs_to_check {
        if !still_referenced.contains(&blob.to_string_lossy().into_owned()) {
            if let Err(e) = fs::remove_file(&blob) {
                eprintln!(
                    "remove_snapshot_entries: failed to remove blob {}: {e}",
                    blob.display()
                );
            }
        }
    }
    true
}

pub fn download_remove(hf_repo_with_tag: &str) -> Result<bool> {
    let (repo_id, tag) = split_repo_tag(hf_repo_with_tag)?;
    if tag.is_empty() {
        return Ok(remove_cached_repo(&repo_id));
    }
    let tag_upper = tag.to_ascii_uppercase();
    let files = get_cached_files(&repo_id);
    if files.is_empty() {
        return Ok(false);
    }

    let used = files_in_use(&repo_id, &tag_upper);
    let mut to_remove = Vec::new();
    for f in &files {
        let split = get_gguf_split_info(&f.path);
        if split.tag == tag_upper && !used.contains(&f.local_path) {
            to_remove.push(PathBuf::from(&f.local_path));
        }
    }
    if to_remove.is_empty() {
        return Ok(false);
    }
    Ok(remove_snapshot_entries(&repo_id, &to_remove))
}

pub fn download_remove_sidecars(hf_repo_with_tag: &str, opts: &DownloadOpts<'_>) -> Result<bool> {
    let model = ParamsModel {
        hf_repo: hf_repo_with_tag.to_owned(),
        ..ParamsModel::default()
    };
    let mut cached = opts.clone();
    cached.offline = true;
    let plan = get_hf_plan(&model, &cached)?;

    let mut to_remove = Vec::new();
    for s in SidecarKind::ALL {
        let local = s.plan_file(&plan).local_path.clone();
        if !local.is_empty() {
            to_remove.push(PathBuf::from(local));
        }
    }
    if to_remove.is_empty() {
        return Ok(false);
    }
    let (repo_id, _) = split_repo_tag(hf_repo_with_tag)?;
    Ok(remove_snapshot_entries(&repo_id, &to_remove))
}

// ---------------------------------------------------------------------------
// HF cache
// ---------------------------------------------------------------------------

fn get_cache_directory() -> Result<PathBuf> {
    let entries: [(&str, &str); 6] = [
        ("LLAMA_CACHE", ""),
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", "hub"),
        ("XDG_CACHE_HOME", "huggingface/hub"),
        ("HOME", ".cache/huggingface/hub"),
    ];
    for (var, suffix) in entries {
        let base = common_get_env(var);
        if !base.is_empty() {
            return Ok(if suffix.is_empty() {
                PathBuf::from(base)
            } else {
                PathBuf::from(base).join(suffix)
            });
        }
    }
    Err(Error::new("Failed to determine HF cache directory"))
}

pub fn get_cache_path() -> Result<String> {
    Ok(get_cache_directory()?.to_string_lossy().into_owned())
}

fn folder_name_to_repo(folder: &str) -> String {
    let Some(rest) = folder.strip_prefix("models--") else {
        return String::new();
    };
    rest.replace("--", "/")
}

fn repo_to_folder_name(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

fn get_repo_path(repo_id: &str) -> Result<PathBuf> {
    Ok(get_cache_directory()?.join(repo_to_folder_name(repo_id)))
}

fn is_hex_char(c: char) -> bool {
    c.is_ascii_hexdigit()
}

fn is_hex_string(s: &str, expected_len: usize) -> bool {
    s.len() == expected_len && s.chars().all(is_hex_char)
}

fn is_alphanum(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

fn is_valid_repo_id(repo_id: &str) -> bool {
    if repo_id.is_empty() || repo_id.len() > 256 {
        return false;
    }
    let mut slash = 0i32;
    let mut special = true;
    for c in repo_id.chars() {
        if is_alphanum(c) || c == '_' {
            special = false;
        } else if matches!(c, '/' | '.' | '-') {
            if special {
                return false;
            }
            slash += i32::from(c == '/');
            special = true;
        } else {
            return false;
        }
    }
    !special && slash == 1
}

fn is_valid_hf_token(token: &str) -> bool {
    if token.len() < 37 || token.len() > 256 || !string_starts_with(token, "hf_") {
        return false;
    }
    token.chars().skip(3).all(is_alphanum)
}

fn is_valid_commit(hash: &str) -> bool {
    is_hex_string(hash, 40)
}

fn is_valid_oid(oid: &str) -> bool {
    is_hex_string(oid, 40) || is_hex_string(oid, 64)
}

/// Pure path computation; creates nothing.
pub fn get_lock_path(repo_id: &str, oid: &str) -> String {
    if !is_valid_repo_id(repo_id) || !is_valid_oid(oid) {
        return String::new();
    }
    let Ok(cache) = get_cache_path() else {
        return String::new();
    };
    Path::new(&cache)
        .join(".locks")
        .join(repo_to_folder_name(repo_id))
        .join(format!("{oid}.lock"))
        .to_string_lossy()
        .into_owned()
}

fn json_string(json: &Json, key: &str) -> Option<String> {
    match json.at(key) {
        Ok(Json::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn api_get(url: &str, token: &str) -> Result<Json> {
    let mut params = RemoteParams::default();
    params
        .headers
        .push(("User-Agent".to_owned(), USER_AGENT.to_owned()));
    params
        .headers
        .push(("Accept".to_owned(), "application/json".to_owned()));
    if is_valid_hf_token(token) {
        params
            .headers
            .push(("Authorization".to_owned(), format!("Bearer {token}")));
    } else if !token.is_empty() {
        eprintln!("hf_cache: invalid token, authentication disabled");
    }

    let (status, body) = remote_get_content(url, &params)?;
    let text = String::from_utf8_lossy(&body);
    if status == 200 {
        return Json::parse(&text);
    }
    let mut message = text.into_owned();
    if let Ok(json) = Json::parse(&message) {
        if let Some(error) = json_string(&json, "error") {
            message = error;
        }
    }
    Err(Error::new(format!("GET failed ({status}): {message}")))
}

fn get_repo_commit(repo_id: &str, token: &str) -> String {
    let endpoint = common_get_model_endpoint();
    let url = format!("{}api/models/{repo_id}/refs", endpoint);
    let Ok(json) = api_get(&url, token) else {
        return String::new();
    };
    if !json.is_object() {
        return String::new();
    }
    let Ok(branches) = json.at("branches") else {
        return String::new();
    };
    if !branches.is_array() {
        return String::new();
    }

    let refs_path = match get_repo_path(repo_id) {
        Ok(path) => path.join("refs"),
        Err(_) => return String::new(),
    };
    let mut name = String::new();
    let mut commit = String::new();
    let mut name_path = PathBuf::new();

    for branch in branches.iter() {
        if !branch.is_object() {
            continue;
        }
        let (Some(branch_name), Some(branch_commit)) = (
            json_string(branch, "name"),
            json_string(branch, "targetCommit"),
        ) else {
            continue;
        };
        if !is_valid_commit(&branch_commit) {
            continue;
        }
        let candidate = PathBuf::from(&branch_name);
        if !is_valid_subpath(&refs_path, &candidate) {
            continue;
        }
        if branch_name == "main" {
            name = branch_name;
            commit = branch_commit;
            name_path = candidate;
            break;
        }
        if name.is_empty() || commit.is_empty() {
            name = branch_name;
            commit = branch_commit;
            name_path = candidate;
        }
    }

    if name.is_empty() || commit.is_empty() {
        return String::new();
    }
    let _ = fs_write_atomic(&refs_path.join(name_path), commit.as_bytes());
    commit
}

fn get_repo_files(repo_id: &str, token: &str) -> HfFiles {
    if !is_valid_repo_id(repo_id) {
        eprintln!("hf_cache: invalid repository: {repo_id}");
        return Vec::new();
    }
    let commit = get_repo_commit(repo_id, token);
    if commit.is_empty() {
        return Vec::new();
    }

    let Ok(repo_path) = get_repo_path(repo_id) else {
        return Vec::new();
    };
    let blobs_path = repo_path.join("blobs");
    let commit_path = repo_path.join("snapshots").join(&commit);
    let endpoint = common_get_model_endpoint();
    let url = format!(
        "{}api/models/{repo_id}/tree/{commit}?recursive=true",
        endpoint
    );
    let Ok(json) = api_get(&url, token) else {
        return Vec::new();
    };
    if !json.is_array() {
        return Vec::new();
    }

    let mut files = HfFiles::new();
    for item in json.iter() {
        if !item.is_object() || json_string(item, "type").as_deref() != Some("file") {
            continue;
        }
        let Some(path) = json_string(item, "path") else {
            continue;
        };
        if !is_valid_subpath(&commit_path, Path::new(&path)) {
            continue;
        }

        let mut oid = String::new();
        if let Ok(lfs) = item.at("lfs") {
            if lfs.is_object() {
                if let Some(value) = json_string(lfs, "oid") {
                    oid = value;
                }
            }
        }
        if oid.is_empty() {
            if let Some(value) = json_string(item, "oid") {
                oid = value;
            }
        }
        if !oid.is_empty() && !is_valid_oid(&oid) {
            continue;
        }

        let final_path = commit_path.join(&path);
        let local_path = if !oid.is_empty() && !final_path.exists() {
            blobs_path.join(&oid)
        } else {
            final_path.clone()
        };
        let url = format!("{endpoint}{repo_id}/resolve/{commit}/{path}");
        files.push(HfFile {
            path,
            url,
            local_path: local_path.to_string_lossy().into_owned(),
            final_path: final_path.to_string_lossy().into_owned(),
            oid,
            repo_id: repo_id.to_owned(),
        });
    }
    files
}

fn get_cached_ref(repo_path: &Path) -> String {
    let refs_path = repo_path.join("refs");
    if !refs_path.is_dir() {
        return String::new();
    }
    let Ok(entries) = fs::read_dir(&refs_path) else {
        return String::new();
    };
    let mut fallback = String::new();
    for entry in entries.flatten() {
        if !entry.path().is_file() {
            continue;
        }
        let Ok(text) = fs::read_to_string(entry.path()) else {
            continue;
        };
        let commit = text.lines().next().unwrap_or("").to_owned();
        if commit.is_empty() || !is_valid_commit(&commit) {
            continue;
        }
        if entry.file_name().to_string_lossy() == "main" {
            return commit;
        }
        if fallback.is_empty() {
            fallback = commit;
        }
    }
    fallback
}

fn collect_files(dir: &Path, commit_path: &Path, repo_id: &str, out: &mut HfFiles) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            collect_files(&path, commit_path, repo_id, out);
            continue;
        }
        if !kind.is_file() && !kind.is_symlink() {
            continue;
        }
        let relative = path.strip_prefix(commit_path).unwrap_or(&path);
        let relative = relative.to_string_lossy().replace('\\', "/");
        if relative.is_empty() {
            continue;
        }
        let local = path.to_string_lossy().into_owned();
        out.push(HfFile {
            path: relative,
            url: String::new(),
            local_path: local.clone(),
            final_path: local,
            oid: String::new(),
            repo_id: repo_id.to_owned(),
        });
    }
}

pub fn get_cached_files(repo_id: &str) -> HfFiles {
    let Ok(cache_path) = get_cache_directory() else {
        return Vec::new();
    };
    if !cache_path.exists() {
        return Vec::new();
    }
    if !repo_id.is_empty() && !is_valid_repo_id(repo_id) {
        eprintln!("hf_cache: invalid repository: {repo_id}");
        return Vec::new();
    }

    let mut files = HfFiles::new();
    let Ok(entries) = fs::read_dir(&cache_path) else {
        return files;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let snapshots_path = path.join("snapshots");
        if !snapshots_path.exists() {
            continue;
        }
        let Some(name) = path.file_name() else {
            continue;
        };
        let id = folder_name_to_repo(&name.to_string_lossy());
        if !is_valid_repo_id(&id) || (!repo_id.is_empty() && id != repo_id) {
            continue;
        }
        let commit = get_cached_ref(&path);
        if commit.is_empty() {
            continue;
        }
        let commit_path = snapshots_path.join(&commit);
        if !commit_path.is_dir() {
            continue;
        }
        collect_files(&commit_path, &commit_path, &id, &mut files);
    }
    files
}

static SYMLINKS_DISABLED: AtomicBool = AtomicBool::new(false);

pub fn finalize_file(file: &HfFile, keep_waiting: Option<&dyn Fn() -> bool>) -> Result<String> {
    let local_path = Path::new(&file.local_path);
    let final_path = Path::new(&file.final_path);
    if local_path == final_path || final_path.exists() {
        return Ok(file.final_path.clone());
    }
    if !local_path.exists() {
        return Ok(file.final_path.clone());
    }

    let mut lock = FileLock::new(get_lock_path(&file.repo_id, &file.oid));
    if !lock.acquire(|| keep_waiting.map_or(true, |f| f())) {
        return Err(Error::new(format!(
            "failed to lock cache for finalization: {}",
            file.final_path
        )));
    }
    if let Some(waiting) = keep_waiting {
        if !waiting() {
            return Err(Error::new(format!(
                "download cancelled before finalization: {}",
                file.final_path
            )));
        }
    }
    if final_path.exists() {
        return Ok(file.final_path.clone());
    }
    if let Some(parent) = final_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    if !SYMLINKS_DISABLED.load(Ordering::Relaxed) {
        let parent = final_path.parent().unwrap_or_else(|| Path::new(""));
        if create_symlink(&symlink_target(local_path, parent), final_path) {
            return Ok(file.final_path.clone());
        }
        if !SYMLINKS_DISABLED.swap(true, Ordering::Relaxed) {
            eprintln!("finalize_file: failed to create symlink, switching to degraded mode");
        }
    }

    if fs::rename(local_path, final_path).is_err() {
        if let Err(e) = fs::copy(local_path, final_path) {
            eprintln!("finalize_file: failed to copy file to snapshots: {e}");
        }
    }
    Ok(file.final_path.clone())
}

pub fn remove_cached_repo(repo_id: &str) -> bool {
    if !is_valid_repo_id(repo_id) {
        eprintln!("hf_cache: invalid repository: {repo_id}");
        return false;
    }
    let Ok(repo_path) = get_repo_path(repo_id) else {
        return false;
    };
    fs::remove_dir_all(repo_path).is_ok()
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_tag_split() {
        assert_eq!(
            split_repo_tag("ggml-org/models:F16").unwrap(),
            ("ggml-org/models".to_owned(), "F16".to_owned())
        );
        assert_eq!(
            split_repo_tag("user/model").unwrap(),
            ("user/model".to_owned(), String::new())
        );
        assert!(split_repo_tag("model").is_err());
    }

    #[test]
    fn gguf_split_and_quant_bits() {
        let split = get_gguf_split_info("model-Q4_K_M-00001-of-00003.gguf");
        assert_eq!(split.prefix, "model-Q4_K_M");
        assert_eq!(split.index, 1);
        assert_eq!(split.count, 3);
        assert_eq!(extract_quant_bits("model-Q4_K_M-00001-of-00003.gguf"), 4);
        assert_eq!(extract_quant_bits("model-F16.gguf"), 16);
        assert_eq!(get_all_parts("model-00001-of-00002.gguf").len(), 2);
    }

    #[test]
    fn sidecar_options_and_plan() {
        let mut opts = DownloadOpts::default();
        assert!(opts_add_sidecar(&mut opts, "mtp"));
        assert!(opts.download_mtp);
        assert!(!opts_add_sidecar(&mut opts, "nope"));
        let mut plan = DownloadHfPlan::default();
        plan.mtp.path = "model-mtp.gguf".to_owned();
        plan.mtp.local_path = "/cache/blob".to_owned();
        plan.mtp.final_path = "/cache/snapshot/model.gguf".to_owned();
        assert_eq!(plan_sidecars(&plan), vec!["mtp"]);
        assert_eq!(plan_partials(&plan).len(), 1);
    }

    #[test]
    fn lock_is_exclusive() {
        let dir = std::env::temp_dir().join(format!("lcr-lock-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("file.lock").to_string_lossy().into_owned();
        let mut a = FileLock::new(path.clone());
        assert!(a.acquire(|| true));
        let mut b = FileLock::new(path);
        assert!(!b.acquire(|| false));
        a.close();
        assert!(b.acquire(|| false));
        b.close();
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn url_helpers() {
        let parts = parse_url("http://user:pass@example.org:8080/a/b?q=1").unwrap();
        assert_eq!(parts.host, "example.org");
        assert_eq!(parts.port, 8080);
        assert_eq!(parts.path, "/a/b?q=1");
        assert_eq!(base64_encode(b"a: b"), "YTogYg==");
        assert_eq!(
            resolve_location("http://example.org/a/b?q=1", "../c").unwrap(),
            "http://example.org:80/a/../c"
        );
    }

    #[test]
    fn cache_validation() {
        assert!(is_valid_repo_id("ggml-org/models"));
        assert!(!is_valid_repo_id("models"));
        assert!(!is_valid_repo_id("-user/model"));
        assert!(is_hex_string("0123456789abcdef0123456789abcdef01234567", 40));
        assert_eq!(
            folder_name_to_repo("models--ggml-org--models"),
            "ggml-org/models"
        );
    }
}

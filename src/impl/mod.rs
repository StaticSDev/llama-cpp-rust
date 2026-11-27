//! Port of `src/llama-impl.cpp` and `src/llama-impl.h`.
//!
//! Shared logging and small helpers used across llama.cpp.

use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// `ggml_log_level`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel {
    None,
    Info,
    Warn,
    Error,
    Debug,
    Cont,
}

type LogCallback = dyn Fn(LogLevel, &str) + Send + Sync;

fn default_log_callback(_level: LogLevel, text: &str) {
    let stderr = std::io::stderr();
    let mut lock = stderr.lock();
    let _ = lock.write_all(text.as_bytes());
    let _ = lock.flush();
}

/// Global logger state. Replaces `g_logger_state` and `ggml_log_get/set`.
pub struct Logger {
    callback: Arc<LogCallback>,
    user_data: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl Logger {
    pub fn new() -> Self {
        Self {
            callback: Arc::new(default_log_callback),
            user_data: None,
        }
    }

    pub fn set_callback(
        &mut self,
        callback: impl Fn(LogLevel, &str) + Send + Sync + 'static,
        user_data: Option<Arc<dyn std::any::Any + Send + Sync>>,
    ) {
        self.callback = Arc::new(callback);
        self.user_data = user_data;
    }

    pub fn callback(&self) -> Arc<LogCallback> {
        Arc::clone(&self.callback)
    }

    pub fn user_data(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        self.user_data.clone()
    }

    pub fn log(&self, level: LogLevel, text: &str) {
        (self.callback)(level, text);
    }
}

impl Default for Logger {
    fn default() -> Self {
        Self::new()
    }
}

static LOGGER: RwLock<Option<Logger>> = RwLock::new(None);

/// `llama_log_get`: returns the callback and its user data.
pub fn log_get() -> (Arc<LogCallback>, Option<Arc<dyn std::any::Any + Send + Sync>>) {
    let guard = LOGGER.read().unwrap();
    match guard.as_ref() {
        Some(logger) => (logger.callback(), logger.user_data()),
        None => (Arc::new(default_log_callback), None),
    }
}

/// `llama_log_set`: installs a callback; `None` restores the default.
pub fn log_set<F>(callback: Option<F>, user_data: Option<Arc<dyn std::any::Any + Send + Sync>>)
where
    F: Fn(LogLevel, &str) + Send + Sync + 'static,
{
    let mut guard = LOGGER.write().unwrap();
    let mut logger = guard.take().unwrap_or_default();
    match callback {
        Some(cb) => logger.set_callback(cb, user_data),
        None => logger.set_callback(default_log_callback, user_data),
    }
    *guard = Some(logger);
}

/// `llama_log_internal`: formats a message and sends it to the current callback.
pub fn log(level: LogLevel, text: &str) {
    let guard = LOGGER.read().unwrap();
    match guard.as_ref() {
        Some(logger) => logger.log(level, text),
        None => default_log_callback(level, text),
    }
}

#[macro_export]
macro_rules! llama_log {
    ($level:expr, $($arg:tt)*) => {
        $crate::r#impl::log($level, &format!($($arg)*))
    };
}

#[macro_export]
macro_rules! llama_log_info {
    ($($arg:tt)*) => { $crate::llama_log!($crate::r#impl::LogLevel::Info, $($arg)*) };
}

#[macro_export]
macro_rules! llama_log_warn {
    ($($arg:tt)*) => { $crate::llama_log!($crate::r#impl::LogLevel::Warn, $($arg)*) };
}

#[macro_export]
macro_rules! llama_log_error {
    ($($arg:tt)*) => { $crate::llama_log!($crate::r#impl::LogLevel::Error, $($arg)*) };
}

#[macro_export]
macro_rules! llama_log_debug {
    ($($arg:tt)*) => { $crate::llama_log!($crate::r#impl::LogLevel::Debug, $($arg)*) };
}

#[macro_export]
macro_rules! llama_log_cont {
    ($($arg:tt)*) => { $crate::llama_log!($crate::r#impl::LogLevel::Cont, $($arg)*) };
}

/// `llama_memory_range`: one host memory region to prefetch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRange {
    pub base: usize,
    pub size: usize,
}

/// Minimal tensor view needed by `prefetch_rows`, instead of a full ggml tensor.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    pub data: &'a [u8],
    pub nb1: usize,
    pub ne1: usize,
    pub is_host: bool,
    pub is_matrix: bool,
}

/// Prefetch the host pages needed to gather these rows.
///
/// Returns the ranges instead of issuing architecture-specific prefetch
/// instructions. `shortcut: no OS prefetch in the dry port, wire this up with a
/// backend when the runtime lands`.
pub fn prefetch_rows(tensor: &TensorView<'_>, rows: &[i32]) -> Vec<MemoryRange> {
    if tensor.data.is_empty() || !tensor.is_host || rows.is_empty() {
        return Vec::new();
    }
    assert!(tensor.is_matrix, "prefetch_rows expects a matrix");

    let row_bytes = tensor.nb1;
    let base = tensor.data.as_ptr() as usize;
    rows.iter()
        .map(|&row| {
            assert!(row >= 0 && (row as usize) < tensor.ne1, "row out of range");
            MemoryRange {
                base: base + (row as usize) * tensor.nb1,
                size: row_bytes,
            }
        })
        .collect()
}

/// RAII timer accumulating elapsed microseconds into `t_acc`.
pub struct TimeMeas<'a> {
    t_start: Option<Instant>,
    t_acc: &'a mut i64,
}

impl<'a> TimeMeas<'a> {
    pub fn new(t_acc: &'a mut i64, disable: bool) -> Self {
        Self {
            t_start: if disable { None } else { Some(Instant::now()) },
            t_acc,
        }
    }
}

impl Drop for TimeMeas<'_> {
    fn drop(&mut self) {
        if let Some(start) = self.t_start {
            *self.t_acc += start.elapsed().as_micros() as i64;
        }
    }
}

/// `replace_all`: in-place replacement of every non-overlapping occurrence.
pub fn replace_all(s: &mut String, search: &str, replace: &str) {
    if search.is_empty() {
        return;
    }
    let mut builder = String::with_capacity(s.len());
    let mut last_pos = 0;
    while let Some(pos) = s[last_pos..].find(search) {
        let pos = last_pos + pos;
        builder.push_str(&s[last_pos..pos]);
        builder.push_str(replace);
        last_pos = pos + search.len();
    }
    builder.push_str(&s[last_pos..]);
    *s = builder;
}

/// `llama_format_tensor_shape`: format dimensions with width 6.
pub fn format_tensor_shape(ne: &[i64]) -> String {
    let mut out = String::with_capacity(ne.len() * 9);
    for (i, n) in ne.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{:6}", n);
    }
    out
}

/// GGUF scalar type subset used by `gguf_kv_to_str`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Uint64,
    Int64,
    Float32,
    Float64,
    Bool,
    String,
    Array,
    Other,
}

/// View over the GGUF key/value storage, implemented by the GGUF context.
pub trait GgufKv {
    fn kv_type(&self, i: usize) -> GgufType;
    fn val_str(&self, i: usize) -> &str;
    fn arr_type(&self, i: usize) -> GgufType;
    fn arr_n(&self, i: usize) -> usize;
    fn arr_str(&self, i: usize, j: usize) -> &str;
    fn type_size(&self, ty: GgufType) -> Option<usize>;
    fn val_data(&self, i: usize) -> &[u8];
    fn arr_data(&self, i: usize) -> &[u8];
}

fn scalar_to_str(ty: GgufType, data: &[u8], i: usize) -> String {
    let at = i.checked_mul(4).unwrap_or(usize::MAX);
    let take = |lo: usize, hi: usize| -> Option<u64> {
        let bytes = data.get(at + lo..at + hi)?;
        let mut buf = [0u8; 8];
        buf[..bytes.len()].copy_from_slice(bytes);
        Some(u64::from_le_bytes(buf))
    };
    match ty {
        GgufType::Uint8 => data.get(i).map(|v| v.to_string()).unwrap_or_default(),
        GgufType::Int8 => data.get(i).map(|v| (*v as i8).to_string()).unwrap_or_default(),
        GgufType::Uint16 => take(0, 2).map(|v| v.to_string()).unwrap_or_default(),
        GgufType::Int16 => take(0, 2).map(|v| (v as i16).to_string()).unwrap_or_default(),
        GgufType::Uint32 => take(0, 4).map(|v| v.to_string()).unwrap_or_default(),
        GgufType::Int32 => take(0, 4).map(|v| (v as i32).to_string()).unwrap_or_default(),
        GgufType::Uint64 => take(0, 8).map(|v| v.to_string()).unwrap_or_default(),
        GgufType::Int64 => take(0, 8).map(|v| (v as i64).to_string()).unwrap_or_default(),
        GgufType::Float32 => take(0, 4)
            .map(|v| (f32::from_bits(v as u32)).to_string())
            .unwrap_or_default(),
        GgufType::Float64 => take(0, 8)
            .map(|v| f64::from_bits(v).to_string())
            .unwrap_or_default(),
        GgufType::Bool => data
            .get(i)
            .map(|v| if *v != 0 { "true" } else { "false" }.to_owned())
            .unwrap_or_default(),
        _ => format!("unknown type {:?}", ty),
    }
}

/// `gguf_kv_to_str`: render one metadata value, truncating long arrays.
pub fn gguf_kv_to_str(ctx: &impl GgufKv, i: usize, max_len: usize) -> String {
    let ty = ctx.kv_type(i);
    match ty {
        GgufType::String => ctx.val_str(i).to_owned(),
        GgufType::Array => {
            let arr_ty = ctx.arr_type(i);
            let arr_n = ctx.arr_n(i);
            let mut res = String::from("[");
            let mut j = 0;
            while j < arr_n && res.len() <= max_len {
                if j > 0 {
                    res.push_str(", ");
                }
                match arr_ty {
                    GgufType::String => {
                        let s = ctx.arr_str(i, j);
                        res.push('"');
                        for ch in s.chars() {
                            if ch == '\\' || ch == '"' {
                                res.push('\\');
                            }
                            res.push(ch);
                        }
                        res.push('"');
                    }
                    GgufType::Array => res.push_str("???"),
                    _ => {
                        let stride = ctx.type_size(arr_ty).unwrap_or(1);
                        let data = ctx.arr_data(i);
                        let start = j * stride;
                        let slice = data.get(start..).unwrap_or(&[]);
                        res.push_str(&scalar_to_str(arr_ty, slice, 0));
                    }
                }
                j += 1;
            }
            if j == arr_n {
                res.push(']');
            }
            res
        }
        _ => {
            let data = ctx.val_data(i);
            scalar_to_str(ty, data, 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_all_non_overlapping_matches() {
        let mut s = String::from("a--b--c");
        replace_all(&mut s, "--", "/");
        assert_eq!(s, "a/b/c");
    }

    #[test]
    fn shapes_dims_with_width_six() {
        assert_eq!(format_tensor_shape(&[1, 32]), "     1,     32");
    }
}

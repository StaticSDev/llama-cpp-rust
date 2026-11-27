//! Logging shared by the helper translation units (`mtmd-helper-common.h`).
//!
//! C varargs become already-formatted strings. The callback keeps a `usize`
//! opaque user-data token in place of the C `void *`.

use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

pub type LogCallback = fn(LogLevel, &str, usize);

#[derive(Debug, Clone, Copy)]
pub struct Logger {
    pub callback: LogCallback,
    pub user_data: usize,
}

impl Logger {
    pub const fn new() -> Self {
        Self {
            callback: default_callback,
            user_data: 0,
        }
    }

    pub fn log(&self, level: LogLevel, text: &str) {
        if text.is_empty() {
            return;
        }
        (self.callback)(level, text, self.user_data);
    }
}

impl Default for Logger {
    fn default() -> Self {
        Self::new()
    }
}

pub fn default_callback(level: LogLevel, text: &str, _user_data: usize) {
    let _ = level;
    eprint!("{}", text);
    let _ = std::io::Write::flush(&mut std::io::stderr());
}

static LOGGER: Mutex<Logger> = Mutex::new(Logger::new());

fn with_logger(f: impl FnOnce(&Logger)) {
    let guard = LOGGER.lock().unwrap_or_else(|e| e.into_inner());
    f(&guard);
}

/// Counterpart of `mtmd_helper_log_set`; `None` restores the default callback.
pub fn set_log_callback(callback: Option<LogCallback>, user_data: usize) {
    let mut guard = LOGGER.lock().unwrap_or_else(|e| e.into_inner());
    guard.callback = callback.unwrap_or(default_callback);
    guard.user_data = user_data;
}

pub fn log_msg(level: LogLevel, text: &str) {
    with_logger(|logger| logger.log(level, text));
}

/// Counterpart of the unlevelled C `LOG` macro.
pub fn log(text: &str) {
    log_msg(LogLevel::Info, text);
}

pub fn log_dbg(text: &str) {
    log_msg(LogLevel::Debug, text);
}

pub fn log_inf(text: &str) {
    log_msg(LogLevel::Info, text);
}

pub fn log_wrn(text: &str) {
    log_msg(LogLevel::Warn, text);
}

pub fn log_err(text: &str) {
    log_msg(LogLevel::Error, text);
}

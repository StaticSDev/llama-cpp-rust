//! Shared helpers: recursion limit, error formatting, debug flag and UTF-8.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

/// `JINJA_RECURSIVE_LIMIT`.
pub const JINJA_RECURSIVE_LIMIT: usize = 128;

/// Error/control-flow type mirroring the C++ exception types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JinjaError {
    /// `break_statement::signal`
    Break,
    /// `continue_statement::signal`
    Continue,
    /// `not_implemented_exception`
    NotImplemented(String),
    /// `rethrown_exception`: already formatted, never wrapped again
    Rethrown(String),
    /// `raised_exception`: still gets source context added by `statement::execute`
    Raised(String),
    /// any `std::exception` (runtime_error, lexer/parser exception, type error...)
    Error(String),
}

impl fmt::Display for JinjaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JinjaError::Break => f.write_str("Break statement executed"),
            JinjaError::Continue => f.write_str("Continue statement executed"),
            JinjaError::NotImplemented(m) => write!(f, "NotImplemented: {m}"),
            JinjaError::Rethrown(m) | JinjaError::Raised(m) | JinjaError::Error(m) => {
                f.write_str(m)
            }
        }
    }
}

impl std::error::Error for JinjaError {}

pub type JResult<T> = Result<T, JinjaError>;

static JINJA_DEBUG: AtomicBool = AtomicBool::new(false);

/// `enable_debug` / `g_jinja_debug`.
pub fn enable_debug_flag(enable: bool) {
    JINJA_DEBUG.store(enable, Ordering::Relaxed);
}

pub fn debug_enabled() -> bool {
    JINJA_DEBUG.load(Ordering::Relaxed)
}

/// Port of `JJ_DEBUG`. The C++ macro includes file and line; here only the
/// formatted message is emitted, which keeps the call sites identical.
#[macro_export]
macro_rules! jj_debug {
    ($($arg:tt)*) => {
        if $crate::common::jinja::debug_enabled() {
            eprintln!($($arg)*);
        }
    };
}

/// `string_replace_all`.
pub fn string_replace_all(s: &mut String, search: &str, replace: &str) {
    if search.is_empty() {
        return;
    }
    let mut builder = String::with_capacity(s.len());
    let mut last_pos = 0;
    while let Some(rel) = s[last_pos..].find(search) {
        let pos = last_pos + rel;
        builder.push_str(&s[last_pos..pos]);
        builder.push_str(replace);
        last_pos = pos + search.len();
    }
    builder.push_str(&s[last_pos..]);
    *s = builder;
}

/// `peak_source`: show source around an error position.
pub fn peak_source(source: &str, pos: usize, max_peak_chars: usize) -> String {
    if source.is_empty() {
        return "(no source available)".to_string();
    }
    let mut output = String::new();
    let start = if pos >= max_peak_chars {
        pos - max_peak_chars
    } else {
        0
    };
    let end = (pos + max_peak_chars).min(source.len());
    let mut substr = source[start..end].to_string();
    string_replace_all(&mut substr, "\n", "↵");
    output.push_str("...");
    output.push_str(&substr);
    output.push_str("...\n");
    output.push_str(&" ".repeat(pos - start + 3));
    output.push('^');
    output
}

/// `fmt_error_with_source`.
pub fn fmt_error_with_source(tag: &str, msg: &str, source: &str, pos: usize) -> String {
    format!("{tag}: {msg}\n{}", peak_source(source, pos, 40))
}

// ---------------------------------------------------------------------------
// UTF-8 helpers (ported from common/unicode.h / unicode.cpp)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Utf8Status {
    Success,
    Incomplete,
    Invalid,
}

pub struct Utf8Parse {
    pub codepoint: u32,
    pub bytes_consumed: usize,
    pub status: Utf8Status,
}

/// `common_parse_utf8_codepoint`, byte-for-byte equivalent.
pub fn parse_utf8_codepoint(input: &[u8], offset: usize) -> Utf8Parse {
    if offset >= input.len() {
        return Utf8Parse {
            codepoint: 0,
            bytes_consumed: 0,
            status: Utf8Status::Incomplete,
        };
    }
    let b0 = input[offset];
    if b0 & 0x80 == 0 {
        return Utf8Parse {
            codepoint: b0 as u32,
            bytes_consumed: 1,
            status: Utf8Status::Success,
        };
    }
    if b0 & 0x40 == 0 {
        return Utf8Parse {
            codepoint: 0,
            bytes_consumed: 1,
            status: Utf8Status::Invalid,
        };
    }
    if b0 & 0x20 == 0 {
        if offset + 1 >= input.len() {
            return Utf8Parse {
                codepoint: 0,
                bytes_consumed: 1,
                status: Utf8Status::Incomplete,
            };
        }
        if input[offset + 1] & 0xc0 != 0x80 {
            return Utf8Parse {
                codepoint: 0,
                bytes_consumed: 1,
                status: Utf8Status::Invalid,
            };
        }
        return Utf8Parse {
            codepoint: (((b0 & 0x1f) as u32) << 6) | ((input[offset + 1] & 0x3f) as u32),
            bytes_consumed: 2,
            status: Utf8Status::Success,
        };
    }
    if b0 & 0x10 == 0 {
        for i in 1..3 {
            if offset + i >= input.len() {
                return Utf8Parse {
                    codepoint: 0,
                    bytes_consumed: i,
                    status: Utf8Status::Incomplete,
                };
            }
            if input[offset + i] & 0xc0 != 0x80 {
                return Utf8Parse {
                    codepoint: 0,
                    bytes_consumed: i,
                    status: Utf8Status::Invalid,
                };
            }
        }
        return Utf8Parse {
            codepoint: (((b0 & 0x0f) as u32) << 12)
                | (((input[offset + 1] & 0x3f) as u32) << 6)
                | ((input[offset + 2] & 0x3f) as u32),
            bytes_consumed: 3,
            status: Utf8Status::Success,
        };
    }
    if b0 & 0x08 == 0 {
        for i in 1..4 {
            if offset + i >= input.len() {
                return Utf8Parse {
                    codepoint: 0,
                    bytes_consumed: i,
                    status: Utf8Status::Incomplete,
                };
            }
            if input[offset + i] & 0xc0 != 0x80 {
                return Utf8Parse {
                    codepoint: 0,
                    bytes_consumed: i,
                    status: Utf8Status::Invalid,
                };
            }
        }
        return Utf8Parse {
            codepoint: (((b0 & 0x07) as u32) << 18)
                | (((input[offset + 1] & 0x3f) as u32) << 12)
                | (((input[offset + 2] & 0x3f) as u32) << 6)
                | ((input[offset + 3] & 0x3f) as u32),
            bytes_consumed: 4,
            status: Utf8Status::Success,
        };
    }
    Utf8Parse {
        codepoint: 0,
        bytes_consumed: 1,
        status: Utf8Status::Invalid,
    }
}

/// `common_unicode_cpt_to_utf8`, returned as bytes so lone surrogates are kept.
pub fn cpt_to_utf8(cpt: u32) -> Vec<u8> {
    if cpt <= 0x7f {
        return vec![cpt as u8];
    }
    if cpt <= 0x7ff {
        return vec![0xc0 | ((cpt >> 6) as u8 & 0x1f), 0x80 | (cpt as u8 & 0x3f)];
    }
    if cpt <= 0xffff {
        return vec![
            0xe0 | ((cpt >> 12) as u8 & 0x0f),
            0x80 | ((cpt >> 6) as u8 & 0x3f),
            0x80 | (cpt as u8 & 0x3f),
        ];
    }
    if cpt <= 0x10ffff {
        return vec![
            0xf0 | ((cpt >> 18) as u8 & 0x07),
            0x80 | ((cpt >> 12) as u8 & 0x3f),
            0x80 | ((cpt >> 6) as u8 & 0x3f),
            0x80 | (cpt as u8 & 0x3f),
        ];
    }
    Vec::new()
}

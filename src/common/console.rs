//! Rust port of `common/console.{h,cpp}`, `common/log.{h,cpp}` and
//! `common/debug.{h,cpp}`.
//!
//! Std only. Terminal raw mode, cursor position queries and the ggml callback
//! install step need OS/FFI handles, so the corresponding C++ behaviour is
//! described in the report instead of being faked here.

use crate::core::{Error, Result};
use crate::json::{Item, Json};
use crate::r#impl::LogLevel;
use std::fmt::Write as FmtWrite;
use std::fs::{File, OpenOptions};
use std::io::{self, IsTerminal as _, Read, Write as IoWrite};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// console state
// ---------------------------------------------------------------------------

pub const ANSI_COLOR_RED: &str = "\x1b[31m";
pub const ANSI_COLOR_GREEN: &str = "\x1b[32m";
const ANSI_COLOR_YELLOW: &str = "\x1b[33m";
const ANSI_COLOR_MAGENTA: &str = "\x1b[35m";
const ANSI_COLOR_GRAY: &str = "\x1b[90m";
const ANSI_COLOR_RESET: &str = "\x1b[0m";
const ANSI_BOLD_GREEN: &str = "\x1b[1m\x1b[32m";
const ANSI_BOLD_RED: &str = "\x1b[1m\x1b[31m";
const WEOF: u32 = u32::MAX;

/// `display_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DisplayType {
    Reset = 0,
    Info,
    Prompt,
    Reasoning,
    UserInput,
    Error,
}

impl Default for DisplayType {
    fn default() -> Self {
        Self::Reset
    }
}

/// `console::completion_callback`.
pub type CompletionCallback = Arc<dyn Fn(&str, usize) -> Vec<(String, usize)> + Send + Sync>;

static ADVANCED_DISPLAY: AtomicBool = AtomicBool::new(false);
static SIMPLE_IO: AtomicBool = AtomicBool::new(true);
static CURRENT_DISPLAY: Mutex<DisplayType> = Mutex::new(DisplayType::Reset);
static TTY: Mutex<Option<File>> = Mutex::new(None);
static COMPLETION_CB: Mutex<Option<CompletionCallback>> = Mutex::new(None);

fn with_out(f: impl FnOnce(&mut dyn IoWrite)) {
    let mut tty = TTY.lock().unwrap();
    match tty.as_mut() {
        Some(file) => f(file),
        None => {
            let stdout = io::stdout();
            let mut lock = stdout.lock();
            f(&mut lock);
        }
    }
}

fn out_str(s: &str) {
    with_out(|w| {
        let _ = w.write_all(s.as_bytes());
    });
}

fn out_bytes(b: &[u8]) {
    with_out(|w| {
        let _ = w.write_all(b);
    });
}

fn out_flush() {
    with_out(|w| {
        let _ = w.flush();
    });
}

// ---------------------------------------------------------------------------
// console init/display
// ---------------------------------------------------------------------------

/// `console::init`. Raw terminal mode is not set: no FFI is available.
pub fn init(use_simple_io: bool, use_advanced_display: bool) {
    SIMPLE_IO.store(use_simple_io, Ordering::Relaxed);
    ADVANCED_DISPLAY.store(use_advanced_display, Ordering::Relaxed);

    #[cfg(unix)]
    if !use_simple_io {
        if let Ok(file) = OpenOptions::new().read(true).write(true).open("/dev/tty") {
            *TTY.lock().unwrap() = Some(file);
        }
    }
}

/// `console::cleanup`. Termios restoration is out of scope.
pub fn cleanup() {
    set_display(DisplayType::Reset);
    #[cfg(unix)]
    if !SIMPLE_IO.load(Ordering::Relaxed) {
        *TTY.lock().unwrap() = None;
    }
}

/// `console::set_display`. Emit an ANSI code only when the display changes.
pub fn set_display(display: DisplayType) {
    if !ADVANCED_DISPLAY.load(Ordering::Relaxed) {
        return;
    }
    let mut current = CURRENT_DISPLAY.lock().unwrap();
    if *current == display {
        return;
    }
    flush_log();
    let code = match display {
        DisplayType::Reset => ANSI_COLOR_RESET,
        DisplayType::Info => ANSI_COLOR_MAGENTA,
        DisplayType::Prompt => ANSI_COLOR_YELLOW,
        DisplayType::Reasoning => ANSI_COLOR_GRAY,
        DisplayType::UserInput => ANSI_BOLD_GREEN,
        DisplayType::Error => ANSI_BOLD_RED,
    };
    out_str(code);
    *current = display;
    out_flush();
}

/// `console::set_completion_callback`. `None` clears it.
pub fn set_completion_callback(cb: Option<CompletionCallback>) {
    *COMPLETION_CB.lock().unwrap() = cb;
}

/// `console::log`. Callers format with `format!` (C++ used printf varargs).
pub fn log(msg: &str) {
    out_str(msg);
}

/// `console::error`. Prints with the error color, then restores the old one.
pub fn error(msg: &str) {
    let previous = *CURRENT_DISPLAY.lock().unwrap();
    set_display(DisplayType::Error);
    log(msg);
    set_display(previous);
}

/// `console::flush`.
pub fn flush() {
    out_flush();
}

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

/// `decode_utf8`: one code point and its byte length at `pos`.
fn decode_utf8(input: &str, pos: usize, advance: &mut usize) -> u32 {
    let b = input.as_bytes();
    let c = b[pos];
    if c & 0x80 == 0 {
        *advance = 1;
        return c as u32;
    }
    if c & 0xE0 == 0xC0 && pos + 1 < b.len() {
        let c1 = b[pos + 1];
        if c1 & 0xC0 != 0x80 {
            *advance = 1;
            return 0xFFFD;
        }
        *advance = 2;
        return ((c as u32 & 0x1F) << 6) | (c1 as u32 & 0x3F);
    }
    if c & 0xF0 == 0xE0 && pos + 2 < b.len() {
        let (c1, c2) = (b[pos + 1], b[pos + 2]);
        if c1 & 0xC0 != 0x80 || c2 & 0xC0 != 0x80 {
            *advance = 1;
            return 0xFFFD;
        }
        *advance = 3;
        return ((c as u32 & 0x0F) << 12)
            | ((c1 as u32 & 0x3F) << 6)
            | (c2 as u32 & 0x3F);
    }
    if c & 0xF8 == 0xF0 && pos + 3 < b.len() {
        let (c1, c2, c3) = (b[pos + 1], b[pos + 2], b[pos + 3]);
        if c1 & 0xC0 != 0x80 || c2 & 0xC0 != 0x80 || c3 & 0xC0 != 0x80 {
            *advance = 1;
            return 0xFFFD;
        }
        *advance = 4;
        return ((c as u32 & 0x07) << 18)
            | ((c1 as u32 & 0x3F) << 12)
            | ((c2 as u32 & 0x3F) << 6)
            | (c3 as u32 & 0x3F);
    }
    *advance = 1;
    0xFFFD
}

fn append_utf8(ch: u32, out: &mut String) {
    if let Some(c) = char::from_u32(ch) {
        out.push(c);
    }
}

fn prev_utf8_char_pos(line: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let mut p = pos - 1;
    let b = line.as_bytes();
    while p > 0 && b[p] & 0xC0 == 0x80 {
        p -= 1;
    }
    p
}

fn next_utf8_char_pos(line: &str, pos: usize) -> usize {
    let b = line.as_bytes();
    if pos >= b.len() {
        return b.len();
    }
    let mut p = pos + 1;
    while p < b.len() && b[p] & 0xC0 == 0x80 {
        p += 1;
    }
    p
}

// shortcut: small wcwidth subset, no East Asian width table; add one when wide
// glyphs are needed in the editor.
fn char_width(cp: u32) -> i32 {
    if cp == 0 {
        return 0;
    }
    if cp < 0x20 || (0x7F..0xA0).contains(&cp) {
        return -1;
    }
    if matches!(
        cp,
        0x0300..=0x036F
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20FF
            | 0xFE20..=0xFE2F
    ) {
        return 0;
    }
    if matches!(
        cp,
        0x1100..=0x115F
            | 0x2329..=0x232A
            | 0x2E80..=0xA4CF
            | 0xAC00..=0xD7A3
            | 0xF900..=0xFAFF
            | 0xFE10..=0xFE19
            | 0xFE30..=0xFE6F
            | 0xFF00..=0xFF60
            | 0xFFE0..=0xFFE6
            | 0x20000..=0x3FFFD
    ) {
        return 2;
    }
    1
}

// shortcut: cursor position queries need a tty control protocol; the expected
// width is returned, as on non-tty streams in the C++ code.
fn put_codepoint(bytes: &[u8], expected_width: i32) -> i32 {
    out_bytes(bytes);
    expected_width
}

fn pop_cursor() {
    out_str("\x08");
}

fn move_cursor(delta: i32) {
    if delta < 0 {
        for _ in 0..-delta {
            out_str("\x08");
        }
    } else {
        for _ in 0..delta {
            out_str("\x1b[C");
        }
    }
}

fn replace_last(ch: char) {
    let mut s = String::from("\x08");
    s.push(ch);
    out_str(&s);
}

fn is_space_codepoint(cp: u32) -> bool {
    char::from_u32(cp).map_or(false, char::is_whitespace)
}

fn is_letter(cp: u32) -> bool {
    (b'A' as u32..=b'Z' as u32).contains(&cp) || (b'a' as u32..=b'z' as u32).contains(&cp)
}

// ---------------------------------------------------------------------------
// line editor helpers
// ---------------------------------------------------------------------------

fn delete_at_cursor(line: &mut String, widths: &mut Vec<i32>, char_pos: &mut usize, byte_pos: &mut usize) {
    if *char_pos >= widths.len() {
        return;
    }
    let next_pos = next_utf8_char_pos(line, *byte_pos);
    let w = widths[*char_pos];
    line.replace_range(*byte_pos..next_pos, "");
    widths.remove(*char_pos);

    let mut p = *byte_pos;
    let mut tail_width = 0;
    for i in *char_pos..widths.len() {
        let following = next_utf8_char_pos(line, p);
        put_codepoint(&line.as_bytes()[p..following], widths[i]);
        tail_width += widths[i];
        p = following;
    }
    for _ in 0..w {
        out_str(" ");
    }
    move_cursor(-(tail_width + w));
}

fn clear_current_line(widths: &[i32]) {
    let total: i32 = widths.iter().map(|w| if *w > 0 { *w } else { 1 }).sum();
    if total > 0 {
        let spaces = " ".repeat(total as usize);
        out_str(&spaces);
        move_cursor(-total);
    }
}

fn set_line_contents(
    new_line: String,
    line: &mut String,
    widths: &mut Vec<i32>,
    char_pos: &mut usize,
    byte_pos: &mut usize,
    cursor_byte_pos: Option<usize>,
) {
    move_to_line_start(char_pos, byte_pos, widths);
    clear_current_line(widths);

    *line = new_line;
    widths.clear();
    *byte_pos = 0;
    *char_pos = 0;

    let mut idx = 0;
    let mut back_width = 0;
    while idx < line.len() {
        let mut advance = 0;
        let cp = decode_utf8(line, idx, &mut advance);
        let expected = char_width(cp);
        let real = put_codepoint(&line.as_bytes()[idx..idx + advance], expected).max(0);
        widths.push(real);
        idx += advance;
        match cursor_byte_pos {
            Some(cbp) if cbp < idx => back_width += real,
            _ => {
                *char_pos += 1;
                *byte_pos = idx;
            }
        }
    }
    if cursor_byte_pos.is_some() {
        move_cursor(-back_width);
    }
}

fn move_to_line_start(char_pos: &mut usize, byte_pos: &mut usize, widths: &[i32]) {
    let back: i32 = widths[..*char_pos].iter().sum();
    move_cursor(-back);
    *char_pos = 0;
    *byte_pos = 0;
}

fn move_to_line_end(char_pos: &mut usize, byte_pos: &mut usize, widths: &[i32], line: &str) {
    let forward: i32 = widths[*char_pos..].iter().sum();
    move_cursor(forward);
    *char_pos = widths.len();
    *byte_pos = line.len();
}

fn has_ctrl_modifier(params: &str) -> bool {
    let mut start = 0;
    while start < params.len() {
        let rest = &params[start..];
        let end = rest.find(';');
        let len = end.unwrap_or(rest.len());
        if len > 0 && rest[..len].bytes().all(|c| c.is_ascii_digit()) {
            let value: i32 = rest[..len].parse().unwrap_or(-1);
            if value == 5 {
                return true;
            }
        }
        match end {
            None => break,
            Some(e) => start += e + 1,
        }
    }
    false
}

fn move_word_left(char_pos: &mut usize, byte_pos: &mut usize, widths: &[i32], line: &str) {
    if *char_pos == 0 {
        return;
    }
    let mut new_char_pos = *char_pos;
    let mut new_byte_pos = *byte_pos;
    let mut move_width = 0;

    while new_char_pos > 0 {
        let prev = prev_utf8_char_pos(line, new_byte_pos);
        let mut advance = 0;
        let cp = decode_utf8(line, prev, &mut advance);
        if !is_space_codepoint(cp) {
            break;
        }
        move_width += widths[new_char_pos - 1];
        new_char_pos -= 1;
        new_byte_pos = prev;
    }
    while new_char_pos > 0 {
        let prev = prev_utf8_char_pos(line, new_byte_pos);
        let mut advance = 0;
        let cp = decode_utf8(line, prev, &mut advance);
        if is_space_codepoint(cp) {
            break;
        }
        move_width += widths[new_char_pos - 1];
        new_char_pos -= 1;
        new_byte_pos = prev;
    }
    move_cursor(-move_width);
    *char_pos = new_char_pos;
    *byte_pos = new_byte_pos;
}

fn move_word_right(char_pos: &mut usize, byte_pos: &mut usize, widths: &[i32], line: &str) {
    if *char_pos >= widths.len() {
        return;
    }
    let mut new_char_pos = *char_pos;
    let mut new_byte_pos = *byte_pos;
    let mut move_width = 0;

    for skip_spaces in [true, false, true] {
        while new_char_pos < widths.len() {
            let mut advance = 0;
            let cp = decode_utf8(line, new_byte_pos, &mut advance);
            if is_space_codepoint(cp) != skip_spaces {
                break;
            }
            move_width += widths[new_char_pos];
            new_char_pos += 1;
            new_byte_pos += advance;
        }
    }
    move_cursor(move_width);
    *char_pos = new_char_pos;
    *byte_pos = new_byte_pos;
}

// ---------------------------------------------------------------------------
// history
// ---------------------------------------------------------------------------

struct History {
    entries: Vec<String>,
    viewing_idx: Option<usize>,
    backup_line: String,
}

impl History {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
            viewing_idx: None,
            backup_line: String::new(),
        }
    }

    fn add(&mut self, line: &str) {
        if line.is_empty() {
            return;
        }
        if self.entries.last().map_or(true, |last| last != line) {
            self.entries.push(line.to_owned());
        }
        self.end_viewing();
    }

    fn prev(&mut self, cur_line: &mut String) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        let Some(idx) = self.viewing_idx else {
            return false;
        };
        let idx = idx.saturating_sub(1);
        self.viewing_idx = Some(idx);
        *cur_line = self.entries[idx].clone();
        true
    }

    fn next(&mut self, cur_line: &mut String) -> bool {
        let (Some(idx), false) = (self.viewing_idx, self.entries.is_empty()) else {
            return false;
        };
        let idx = idx + 1;
        if idx >= self.entries.len() {
            *cur_line = self.backup_line.clone();
            self.end_viewing();
        } else {
            self.viewing_idx = Some(idx);
            *cur_line = self.entries[idx].clone();
        }
        true
    }

    fn begin_viewing(&mut self, line: &str) {
        self.backup_line = line.to_owned();
        self.viewing_idx = Some(self.entries.len());
    }

    fn end_viewing(&mut self) {
        self.viewing_idx = None;
        self.backup_line.clear();
    }

    fn is_viewing(&self) -> bool {
        self.viewing_idx.is_some()
    }
}

static HISTORY: Mutex<History> = Mutex::new(History::new());

fn history_prev(line: &mut String, widths: &mut Vec<i32>, char_pos: &mut usize, byte_pos: &mut usize) {
    let mut h = HISTORY.lock().unwrap();
    if !h.is_viewing() {
        h.begin_viewing(line);
    }
    let mut new_line = String::new();
    if !h.prev(&mut new_line) {
        return;
    }
    drop(h);
    set_line_contents(new_line, line, widths, char_pos, byte_pos, None);
}

fn history_next(line: &mut String, widths: &mut Vec<i32>, char_pos: &mut usize, byte_pos: &mut usize) {
    let mut h = HISTORY.lock().unwrap();
    if !h.is_viewing() {
        return;
    }
    let mut new_line = String::new();
    if !h.next(&mut new_line) {
        return;
    }
    drop(h);
    set_line_contents(new_line, line, widths, char_pos, byte_pos, None);
}

// ---------------------------------------------------------------------------
// readline
// ---------------------------------------------------------------------------

fn getchar32<R: Read>(reader: &mut R) -> Option<u32> {
    let mut first = [0u8; 1];
    if reader.read(&mut first).ok()? == 0 {
        return None;
    }
    let c = first[0];
    if c & 0x80 == 0 {
        return Some(c as u32);
    }
    let len = if c & 0xE0 == 0xC0 {
        2
    } else if c & 0xF0 == 0xE0 {
        3
    } else if c & 0xF8 == 0xF0 {
        4
    } else {
        return Some(0xFFFD);
    };
    let mut buf = [0u8; 4];
    buf[0] = c;
    for slot in buf.iter_mut().take(len).skip(1) {
        let mut b = [0u8; 1];
        if reader.read(&mut b).ok()? == 0 {
            return None;
        }
        *slot = b[0];
    }
    std::str::from_utf8(&buf[..len])
        .ok()?
        .chars()
        .next()
        .map(|ch| ch as u32)
}

fn readline_simple(line: &mut String, mut multiline_input: bool) -> bool {
    line.clear();
    if io::stdin().read_line(line).unwrap_or(0) == 0 {
        line.clear();
        return false;
    }
    if line.ends_with('\n') {
        line.pop();
    }
    if let Some(last) = line.as_bytes().last().copied() {
        if last == b'/' {
            line.push('\n');
            return false;
        }
        if last == b'\\' {
            line.pop();
            multiline_input = !multiline_input;
        }
    }
    line.push('\n');
    multiline_input
}

fn readline_advanced(line: &mut String, multiline_input: bool) -> bool {
    out_flush();
    line.clear();

    let mut widths: Vec<i32> = Vec::new();
    let mut is_special_char = false;
    let mut end_of_stream = false;
    let mut byte_pos = 0usize;
    let mut char_pos = 0usize;

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    loop {
        out_flush();
        let Some(input_char) = getchar32(&mut reader) else {
            end_of_stream = true;
            break;
        };
        if input_char == b'\r' as u32 || input_char == b'\n' as u32 {
            break;
        }

        if input_char == b'\t' as u32 {
            let candidates = {
                let cb = COMPLETION_CB.lock().unwrap();
                match cb.as_ref() {
                    Some(cb) => cb(line, byte_pos),
                    None => Vec::new(),
                }
            };
            if !candidates.is_empty() {
                if candidates.len() > 1 || candidates[0].0.as_str() != line.as_str() {
                    let new_line = candidates[0].0.clone();
                    let cbp = candidates[0].1;
                    set_line_contents(
                        new_line,
                        line,
                        &mut widths,
                        &mut char_pos,
                        &mut byte_pos,
                        Some(cbp),
                    );
                }
                continue;
            }
        }

        if input_char == 0x04 {
            end_of_stream = true;
            break;
        }

        if is_special_char {
            if let Some(last) = line.chars().last() {
                replace_last(last);
            }
            is_special_char = false;
        }

        if input_char == 0x1B {
            let mut code = getchar32(&mut reader).unwrap_or(WEOF);
            if code == b'[' as u32 {
                let mut params = String::new();
                loop {
                    let c = getchar32(&mut reader).unwrap_or(WEOF);
                    code = c;
                    if c == WEOF || is_letter(c) || c == b'~' as u32 {
                        break;
                    }
                    append_utf8(c, &mut params);
                }

                let ctrl_modifier = has_ctrl_modifier(&params);
                if code == b'D' as u32 {
                    if ctrl_modifier {
                        move_word_left(&mut char_pos, &mut byte_pos, &widths, line);
                    } else if char_pos > 0 {
                        let w = widths[char_pos - 1];
                        move_cursor(-w);
                        char_pos -= 1;
                        byte_pos = prev_utf8_char_pos(line, byte_pos);
                    }
                } else if code == b'C' as u32 {
                    if ctrl_modifier {
                        move_word_right(&mut char_pos, &mut byte_pos, &widths, line);
                    } else if char_pos < widths.len() {
                        let w = widths[char_pos];
                        move_cursor(w);
                        char_pos += 1;
                        byte_pos = next_utf8_char_pos(line, byte_pos);
                    }
                } else if code == b'H' as u32 {
                    move_to_line_start(&mut char_pos, &mut byte_pos, &widths);
                } else if code == b'F' as u32 {
                    move_to_line_end(&mut char_pos, &mut byte_pos, &widths, line);
                } else if code == b'A' as u32 {
                    history_prev(line, &mut widths, &mut char_pos, &mut byte_pos);
                    is_special_char = false;
                } else if code == b'B' as u32 {
                    history_next(line, &mut widths, &mut char_pos, &mut byte_pos);
                    is_special_char = false;
                } else if code == b'~' as u32 {
                    let digits: String = params
                        .bytes()
                        .take_while(|c| *c != b';')
                        .filter(|c| c.is_ascii_digit())
                        .map(char::from)
                        .collect();
                    if digits == "1" || digits == "7" {
                        move_to_line_start(&mut char_pos, &mut byte_pos, &widths);
                    } else if digits == "4" || digits == "8" {
                        move_to_line_end(&mut char_pos, &mut byte_pos, &widths, line);
                    } else if digits == "3" {
                        delete_at_cursor(line, &mut widths, &mut char_pos, &mut byte_pos);
                    }
                }
            } else if code == 0x1B {
                while let Some(c) = getchar32(&mut reader) {
                    if is_letter(c) || c == b'~' as u32 {
                        break;
                    }
                }
            }
        } else if input_char == 0x08 || input_char == 0x7F {
            if char_pos > 0 {
                let w = widths[char_pos - 1];
                move_cursor(-w);
                char_pos -= 1;
                let prev_pos = prev_utf8_char_pos(line, byte_pos);
                let char_len = byte_pos - prev_pos;
                byte_pos = prev_pos;
                line.replace_range(byte_pos..byte_pos + char_len, "");
                widths.remove(char_pos);

                let mut p = byte_pos;
                let mut tail_width = 0;
                for i in char_pos..widths.len() {
                    let next_p = next_utf8_char_pos(line, p);
                    put_codepoint(&line.as_bytes()[p..next_p], widths[i]);
                    tail_width += widths[i];
                    p = next_p;
                }
                for _ in 0..w {
                    out_str(" ");
                }
                move_cursor(-(tail_width + w));
            }
        } else {
            let mut new_char_str = String::new();
            append_utf8(input_char, &mut new_char_str);
            let w = char_width(input_char);
            let real_w = put_codepoint(new_char_str.as_bytes(), w).max(0);

            if char_pos == widths.len() {
                line.push_str(&new_char_str);
                widths.push(real_w);
                byte_pos += new_char_str.len();
                char_pos += 1;
            } else {
                line.insert_str(byte_pos, &new_char_str);
                widths.insert(char_pos, real_w);
                let mut p = byte_pos + new_char_str.len();
                let mut tail_width = 0;
                for i in char_pos + 1..widths.len() {
                    let next_p = next_utf8_char_pos(line, p);
                    put_codepoint(&line.as_bytes()[p..next_p], widths[i]);
                    tail_width += widths[i];
                    p = next_p;
                }
                move_cursor(-tail_width);
                byte_pos += new_char_str.len();
                char_pos += 1;
            }
        }

        if let Some(last) = line.as_bytes().last().copied() {
            if last == b'\\' || last == b'/' {
                replace_last(last as char);
                is_special_char = true;
            }
        }
    }

    let mut has_more = multiline_input;
    if is_special_char {
        replace_last(' ');
        pop_cursor();
        let last = line.pop().unwrap_or(' ');
        if last == '\\' {
            line.push('\n');
            out_str("\n");
            has_more = !has_more;
        } else {
            if line.len() == 1 && line.as_bytes()[0] == b' ' {
                line.clear();
                pop_cursor();
            }
            line.push('\n');
            has_more = false;
        }
    } else if end_of_stream {
        has_more = false;
    } else {
        line.push('\n');
        out_str("\n");
    }

    if !end_of_stream && !line.is_empty() {
        let hline = line.strip_suffix('\n').unwrap_or(line);
        HISTORY.lock().unwrap().add(hline);
    }
    out_flush();
    has_more
}

/// `console::readline`. Returns true when more lines are wanted.
pub fn readline(line: &mut String, multiline_input: bool) -> bool {
    if SIMPLE_IO.load(Ordering::Relaxed) {
        readline_simple(line, multiline_input)
    } else {
        readline_advanced(line, multiline_input)
    }
}

// ---------------------------------------------------------------------------
// spinner
// ---------------------------------------------------------------------------

/// `console::spinner`.
pub mod spinner {
    use super::*;

    const LOADING_CHARS: [char; 4] = ['|', '/', '-', '\\'];
    const WAIT: Duration = Duration::from_millis(100);
    static STATE: Mutex<SpinnerState> = Mutex::new(SpinnerState::new());
    static CV: Condvar = Condvar::new();
    static THREAD: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);

    struct SpinnerState {
        running: bool,
        frame: usize,
    }

    impl SpinnerState {
        const fn new() -> Self {
            Self {
                running: false,
                frame: 0,
            }
        }
    }

    /// `spinner::start`.
    pub fn start() {
        let mut state = STATE.lock().unwrap();
        if SIMPLE_IO.load(Ordering::Relaxed) || state.running {
            return;
        }
        flush_log();
        out_str(&LOADING_CHARS[0].to_string());
        out_flush();
        state.frame = 1;
        state.running = true;
        drop(state);

        let handle = std::thread::spawn(|| {
            let mut state = STATE.lock().unwrap();
            loop {
                let (guard, _) = CV.wait_timeout(state, WAIT).unwrap();
                state = guard;
                if !state.running {
                    break;
                }
                state.frame = (state.frame + 1) % LOADING_CHARS.len();
                let ch = LOADING_CHARS[state.frame];
                drop(state);
                replace_last(ch);
                out_flush();
                state = STATE.lock().unwrap();
            }
        });
        *THREAD.lock().unwrap() = Some(handle);
    }

    /// `spinner::stop`.
    pub fn stop() {
        let mut state = STATE.lock().unwrap();
        if SIMPLE_IO.load(Ordering::Relaxed) || !state.running {
            return;
        }
        state.running = false;
        CV.notify_all();
        drop(state);

        if let Some(handle) = THREAD.lock().unwrap().take() {
            let _ = handle.join();
        }
        replace_last(' ');
        pop_cursor();
        out_flush();
    }
}

// ---------------------------------------------------------------------------
// log.h / log.cpp
// ---------------------------------------------------------------------------

pub const LOG_CLR_TO_EOL: &str = "\x1b[K\r";
pub const LOG_COL_DEFAULT: &str = "\x1b[0m";
pub const LOG_COL_BOLD: &str = "\x1b[1m";
pub const LOG_COL_RED: &str = "\x1b[31m";
pub const LOG_COL_GREEN: &str = "\x1b[32m";
pub const LOG_COL_YELLOW: &str = "\x1b[33m";
pub const LOG_COL_BLUE: &str = "\x1b[34m";
pub const LOG_COL_MAGENTA: &str = "\x1b[35m";
pub const LOG_COL_CYAN: &str = "\x1b[36m";
pub const LOG_COL_WHITE: &str = "\x1b[37m";

pub const LOG_LEVEL_DEBUG: i32 = 5;
pub const LOG_LEVEL_TRACE: i32 = 4;
pub const LOG_LEVEL_INFO: i32 = 3;
pub const LOG_LEVEL_WARN: i32 = 2;
pub const LOG_LEVEL_ERROR: i32 = 1;
pub const LOG_LEVEL_OUTPUT: i32 = 0;
pub const LOG_DEFAULT_DEBUG: i32 = LOG_LEVEL_DEBUG;
pub const LOG_DEFAULT_LLAMA: i32 = LOG_LEVEL_INFO;

/// `log_colors`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum LogColors {
    Auto = -1,
    Disabled = 0,
    Enabled = 1,
}

static VERBOSITY_THOLD: AtomicI32 = AtomicI32::new(LOG_DEFAULT_LLAMA);
static JSONL: AtomicBool = AtomicBool::new(false);

pub fn common_log_get_verbosity_thold() -> i32 {
    VERBOSITY_THOLD.load(Ordering::Relaxed)
}

pub fn common_log_set_verbosity_thold(verbosity: i32) {
    VERBOSITY_THOLD.store(verbosity, Ordering::Relaxed);
}

pub fn common_log_get_jsonl() -> bool {
    JSONL.load(Ordering::Relaxed)
}

pub fn common_log_set_jsonl(jsonl: bool) {
    JSONL.store(jsonl, Ordering::Relaxed);
}

pub fn common_log_get_verbosity(level: LogLevel) -> i32 {
    match level {
        LogLevel::Debug => LOG_LEVEL_DEBUG,
        LogLevel::Info => LOG_LEVEL_TRACE,
        LogLevel::Warn => LOG_LEVEL_WARN,
        LogLevel::Error => LOG_LEVEL_ERROR,
        LogLevel::Cont => LOG_LEVEL_TRACE,
        LogLevel::None => LOG_LEVEL_OUTPUT,
    }
}

fn level_str(level: LogLevel) -> &'static str {
    match level {
        LogLevel::Debug => "debug",
        LogLevel::Info => "info",
        LogLevel::Warn => "warn",
        LogLevel::Error => "error",
        LogLevel::Cont => "cont",
        LogLevel::None => "none",
    }
}

fn t_us() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

struct LogState {
    file: Option<File>,
    stdo: bool,
    colors: bool,
    prefix: bool,
    timestamps: bool,
    running: bool,
    t_start: i64,
}

/// `common_log`. Emits entries synchronously under a lock.
pub struct CommonLog {
    inner: Mutex<LogState>,
}

impl CommonLog {
    /// `common_log_init`.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(LogState {
                file: None,
                stdo: true,
                colors: false,
                prefix: false,
                timestamps: false,
                running: true,
                t_start: t_us(),
            }),
        }
    }

    fn colors_of(st: &LogState) -> [&'static str; 9] {
        if st.colors {
            [
                LOG_COL_DEFAULT,
                LOG_COL_BOLD,
                LOG_COL_RED,
                LOG_COL_GREEN,
                LOG_COL_YELLOW,
                LOG_COL_BLUE,
                LOG_COL_MAGENTA,
                LOG_COL_CYAN,
                LOG_COL_WHITE,
            ]
        } else {
            ["", "", "", "", "", "", "", "", ""]
        }
    }

    fn format_entry(st: &LogState, level: LogLevel, msg: &str, timestamp: i64, jsonl: bool) -> String {
        if jsonl {
            let obj = Json::object_from([
                Item::new("type", "log"),
                Item::new("time", timestamp),
                Item::new("level", level_str(level)),
                Item::new("msg", msg),
            ]);
            let mut out = obj.dump_safe(-1);
            out.push('\n');
            return out;
        }

        let col = Self::colors_of(st);
        let mut out = String::new();
        if level != LogLevel::None
            && level != LogLevel::Cont
            && st.prefix
        {
            if timestamp != 0 {
                let _ = write!(
                    out,
                    "{}{}.{:02}.{:03}.{:03}{} ",
                    col[5],
                    timestamp / 1000000 / 60,
                    timestamp / 1000000 % 60,
                    timestamp / 1000 % 1000,
                    timestamp % 1000,
                    col[0]
                );
            }
            match level {
                LogLevel::Info => {
                    let _ = write!(out, "{}I {}", col[3], col[0]);
                }
                LogLevel::Warn => {
                    let _ = write!(out, "{}W ", col[6]);
                }
                LogLevel::Error => {
                    let _ = write!(out, "{}E ", col[2]);
                }
                LogLevel::Debug => {
                    let _ = write!(out, "{}D ", col[4]);
                }
                _ => {}
            }
        }

        let reset = matches!(level, LogLevel::Warn | LogLevel::Error | LogLevel::Debug);
        let end = msg.trim_end_matches('\n').len();
        out.push_str(&msg[..end]);
        if reset {
            out.push_str(col[0]);
        }
        out.push_str(&msg[end..]);
        out
    }

    fn write_out(stderr: bool, text: &str) {
        if stderr {
            let stream = io::stderr();
            let mut lock = stream.lock();
            let _ = lock.write_all(text.as_bytes());
            let _ = lock.flush();
        } else {
            let stream = io::stdout();
            let mut lock = stream.lock();
            let _ = lock.write_all(text.as_bytes());
            let _ = lock.flush();
        }
    }

    /// `common_log::add`.
    pub fn add(&self, level: LogLevel, msg: &str) {
        let mut st = self.inner.lock().unwrap();
        if !st.running {
            return;
        }
        let jsonl = common_log_get_jsonl();
        let timestamp = if st.timestamps {
            t_us() - st.t_start
        } else {
            0
        };
        let text = Self::format_entry(&st, level, msg, timestamp, jsonl);

        let debug_hidden = level == LogLevel::Debug
            && common_log_get_verbosity_thold() < LOG_DEFAULT_DEBUG;
        if st.stdo && !debug_hidden {
            Self::write_out(level != LogLevel::None && !jsonl, &text);
        }
        if let Some(file) = st.file.as_mut() {
            let _ = file.write_all(text.as_bytes());
            let _ = file.flush();
        }
    }

    /// `common_log::add_json`.
    pub fn add_json(&self, type_: &str, data: &Json) {
        let full = Json::object_from([Item::new("type", type_), Item::new("data", data.clone())]);
        let mut text = full.dump_safe(-1);
        text.push('\n');

        let mut st = self.inner.lock().unwrap();
        if !st.running {
            return;
        }
        Self::write_out(false, &text);
        if let Some(file) = st.file.as_mut() {
            let _ = file.write_all(text.as_bytes());
            let _ = file.flush();
        }
    }

    /// `common_log::pause`.
    pub fn pause(&self) {
        self.inner.lock().unwrap().running = false;
    }

    /// `common_log::resume`.
    pub fn resume(&self) {
        self.inner.lock().unwrap().running = true;
    }

    /// `common_log::set_file`.
    pub fn set_file(&self, path: Option<&str>) {
        let mut st = self.inner.lock().unwrap();
        st.running = false;
        st.file = path.and_then(|p| File::create(p).ok());
        st.running = true;
    }

    /// `common_log::set_stdo`.
    pub fn set_stdo(&self, enabled: bool) {
        self.inner.lock().unwrap().stdo = enabled;
    }

    /// `common_log::get_colors`.
    pub fn get_colors(&self) -> bool {
        self.inner.lock().unwrap().colors
    }

    /// `common_log::set_colors`.
    pub fn set_colors(&self, colors: LogColors) {
        self.inner.lock().unwrap().colors = match colors {
            LogColors::Auto => io::stdout().is_terminal(),
            LogColors::Disabled => false,
            LogColors::Enabled => true,
        };
    }

    /// `common_log::set_prefix`.
    pub fn set_prefix(&self, prefix: bool) {
        self.inner.lock().unwrap().prefix = prefix;
    }

    /// `common_log::set_timestamps`.
    pub fn set_timestamps(&self, timestamps: bool) {
        self.inner.lock().unwrap().timestamps = timestamps;
    }
}

impl Default for CommonLog {
    fn default() -> Self {
        Self::new()
    }
}

static MAIN_LOG: OnceLock<CommonLog> = OnceLock::new();

/// `common_log_main`. Singleton, intentionally never dropped.
pub fn common_log_main() -> &'static CommonLog {
    MAIN_LOG.get_or_init(|| {
        let log = CommonLog::new();
        log.set_colors(LogColors::Auto);
        log
    })
}

pub fn common_log_init() -> CommonLog {
    CommonLog::new()
}

pub fn common_log_pause(log: &CommonLog) {
    log.pause();
}

pub fn common_log_resume(log: &CommonLog) {
    log.resume();
}

pub fn common_log_free(_log: CommonLog) {}

pub fn common_log_add(log: &CommonLog, level: LogLevel, msg: &str) {
    log.add(level, msg);
}

pub fn common_log_add_json(log: &CommonLog, type_: &str, data: &Json) {
    if !common_log_get_jsonl() {
        return;
    }
    log.add_json(type_, data);
}

pub fn common_log_set_file(log: &CommonLog, file: Option<&str>) {
    log.set_file(file);
}

pub fn common_log_set_stdo(log: &CommonLog, enabled: bool) {
    log.set_stdo(enabled);
}

pub fn common_log_set_colors(log: &CommonLog, colors: LogColors) {
    log.set_colors(colors);
}

pub fn common_log_get_colors(log: &CommonLog) -> bool {
    log.get_colors()
}

pub fn common_log_set_prefix(log: &CommonLog, prefix: bool) {
    log.set_prefix(prefix);
}

pub fn common_log_set_timestamps(log: &CommonLog, timestamps: bool) {
    log.set_timestamps(timestamps);
}

pub fn common_log_flush(log: &CommonLog) {
    log.pause();
    log.resume();
}

fn flush_log() {
    common_log_flush(common_log_main());
}

pub fn common_log_default_callback(level: LogLevel, text: &str) {
    if common_log_get_verbosity(level) <= common_log_get_verbosity_thold() {
        common_log_add(common_log_main(), level, text);
    }
}

/// `LOG_TMPL`: skip formatting when the verbosity is above the threshold.
#[macro_export]
macro_rules! common_log_tmpl {
    ($level:expr, $verbosity:expr, $($arg:tt)*) => {
        if $verbosity <= $crate::common::console::common_log_get_verbosity_thold() {
            $crate::common::console::common_log_add(
                $crate::common::console::common_log_main(),
                $level,
                &format!($($arg)*),
            );
        }
    };
}

#[macro_export]
macro_rules! common_log {
    ($($arg:tt)*) => {
        $crate::common_log_tmpl!($crate::r#impl::LogLevel::None, $crate::common::console::LOG_LEVEL_OUTPUT, $($arg)*)
    };
}

#[macro_export]
macro_rules! common_log_dbg {
    ($($arg:tt)*) => {
        $crate::common_log_tmpl!($crate::r#impl::LogLevel::Debug, $crate::common::console::LOG_LEVEL_DEBUG, $($arg)*)
    };
}

#[macro_export]
macro_rules! common_log_trc {
    ($($arg:tt)*) => {
        $crate::common_log_tmpl!($crate::r#impl::LogLevel::Info, $crate::common::console::LOG_LEVEL_TRACE, $($arg)*)
    };
}

#[macro_export]
macro_rules! common_log_inf {
    ($($arg:tt)*) => {
        $crate::common_log_tmpl!($crate::r#impl::LogLevel::Info, $crate::common::console::LOG_LEVEL_INFO, $($arg)*)
    };
}

#[macro_export]
macro_rules! common_log_wrn {
    ($($arg:tt)*) => {
        $crate::common_log_tmpl!($crate::r#impl::LogLevel::Warn, $crate::common::console::LOG_LEVEL_WARN, $($arg)*)
    };
}

#[macro_export]
macro_rules! common_log_err {
    ($($arg:tt)*) => {
        $crate::common_log_tmpl!($crate::r#impl::LogLevel::Error, $crate::common::console::LOG_LEVEL_ERROR, $($arg)*)
    };
}

#[macro_export]
macro_rules! common_log_json {
    ($type:expr, $data:expr) => {
        if $crate::common::console::common_log_get_jsonl() {
            $crate::common::console::common_log_add_json(
                $crate::common::console::common_log_main(),
                $type,
                &$data,
            );
        }
    };
}

// ---------------------------------------------------------------------------
// debug.h / debug.cpp
// ---------------------------------------------------------------------------

/// `ggml_type` subset used by the tensor dump.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugType {
    F16,
    F32,
    I64,
    I32,
    I16,
    I8,
    Bf16,
    Other,
}

impl DebugType {
    /// `ggml_type_name` for the handled types.
    pub fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::I64 => "i64",
            Self::I32 => "i32",
            Self::I16 => "i16",
            Self::I8 => "i8",
            Self::Bf16 => "bf16",
            Self::Other => "?",
        }
    }

}

/// `common_debug_cb_user_data`.
pub struct DebugCbUserData {
    data: Vec<u8>,
    tensor_filters: Vec<String>,
    abort_on_nan: bool,
}

impl Default for DebugCbUserData {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugCbUserData {
    /// Default constructor: no filters, no abort.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            tensor_filters: Vec::new(),
            abort_on_nan: false,
        }
    }

    /// Constructor with anchored tensor filters. The C++ version installs the
    /// callback into `common_params`; this port only stores the filter state.
    pub fn with_filters(filter_patterns: &[String], abort_on_nan: bool) -> Result<Self> {
        let mut tensor_filters = Vec::with_capacity(filter_patterns.len());
        for pattern in filter_patterns {
            check_pattern(pattern)?;
            tensor_filters.push(format!("^{pattern}"));
        }
        Ok(Self {
            data: Vec::new(),
            tensor_filters,
            abort_on_nan,
        })
    }
}

fn check_pattern(pattern: &str) -> Result<()> {
    let mut brackets = 0i32;
    let mut parens = 0i32;
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if chars.next().is_none() {
                    return Err(Error::new(format!(
                        "Invalid regex pattern '{pattern}': trailing backslash"
                    )));
                }
            }
            '[' => brackets += 1,
            ']' => brackets -= 1,
            '(' => parens += 1,
            ')' => parens -= 1,
            _ => {}
        }
        if brackets < 0 || parens < 0 {
            return Err(Error::new(format!(
                "Invalid regex pattern '{pattern}': unbalanced bracket"
            )));
        }
    }
    if brackets != 0 || parens != 0 {
        return Err(Error::new(format!(
            "Invalid regex pattern '{pattern}': unbalanced bracket"
        )));
    }
    Ok(())
}

/// Minimal view of the ggml tensor the callback needs.
pub struct DebugTensor<'a> {
    pub name: &'a str,
    pub type_: DebugType,
    pub ne: [i64; 4],
    pub nb: [usize; 4],
    pub data: &'a [u8],
    pub is_host: bool,
    pub src0_name: &'a str,
    pub src0_ne: [i64; 4],
    pub src1: Option<(&'a str, [i64; 4])>,
    pub op_desc: &'a str,
}

fn common_ggml_ne_string(ne: &[i64]) -> String {
    ne.iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn common_ggml_get_float_value(
    data: &[u8],
    type_: DebugType,
    nb: &[usize; 4],
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
) -> Result<f32> {
    let i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    let at = |n: usize| -> Result<&[u8]> {
        data.get(i..i + n).ok_or_else(|| Error::new("tensor index out of range"))
    };
    match type_ {
        DebugType::F16 => {
            let b: [u8; 2] = at(2)?.try_into().unwrap();
            Ok(crate::quant::f16_to_f32(u16::from_le_bytes(b)))
        }
        DebugType::Bf16 => {
            let b: [u8; 2] = at(2)?.try_into().unwrap();
            Ok(crate::quant::bf16_to_f32(u16::from_le_bytes(b)))
        }
        DebugType::F32 => {
            let b: [u8; 4] = at(4)?.try_into().unwrap();
            Ok(f32::from_le_bytes(b))
        }
        DebugType::I64 => {
            let b: [u8; 8] = at(8)?.try_into().unwrap();
            Ok(i64::from_le_bytes(b) as f32)
        }
        DebugType::I32 => {
            let b: [u8; 4] = at(4)?.try_into().unwrap();
            Ok(i32::from_le_bytes(b) as f32)
        }
        DebugType::I16 => {
            let b: [u8; 2] = at(2)?.try_into().unwrap();
            Ok(i16::from_le_bytes(b) as f32)
        }
        DebugType::I8 => Ok(at(1)?[0] as i8 as f32),
        DebugType::Other => Err(Error::new("fatal error")),
    }
}

fn common_debug_print_tensor(
    data: &[u8],
    type_: DebugType,
    ne: &[i64; 4],
    nb: &[usize; 4],
    n: i64,
    abort_on_nan: bool,
) -> Result<()> {
    let mut sum = 0.0f32;
    for i3 in 0..ne[3] {
        for i2 in 0..ne[2] {
            for i1 in 0..ne[1] {
                for i0 in 0..ne[0] {
                    let v = common_ggml_get_float_value(
                        data,
                        type_,
                        nb,
                        i0 as usize,
                        i1 as usize,
                        i2 as usize,
                        i3 as usize,
                    )?;
                    sum += v;
                }
            }
        }
    }

    let mut out = String::new();
    for i3 in 0..ne[3] {
        let _ = writeln!(out, "    [");
        let mut i2 = 0;
        while i2 < ne[2] {
            if i2 == n && ne[2] > 2 * n {
                let _ = writeln!(out, "        ..., ");
                i2 = ne[2] - n;
            }
            let _ = writeln!(out, "        [");
            let mut i1 = 0;
            while i1 < ne[1] {
                if i1 == n && ne[1] > 2 * n {
                    let _ = writeln!(out, "            ..., ");
                    i1 = ne[1] - n;
                }
                let _ = write!(out, "            [");
                let mut i0 = 0;
                while i0 < ne[0] {
                    if i0 == n && ne[0] > 2 * n {
                        let _ = write!(out, "   ..., ");
                        i0 = ne[0] - n;
                    }
                    let v = common_ggml_get_float_value(
                        data,
                        type_,
                        nb,
                        i0 as usize,
                        i1 as usize,
                        i2 as usize,
                        i3 as usize,
                    )?;
                    let _ = write!(out, "{v:12.4}");
                    if i0 < ne[0] - 1 {
                        let _ = write!(out, ", ");
                    }
                    i0 += 1;
                }
                let _ = writeln!(out, "  ],");
                i1 += 1;
            }
            let _ = writeln!(out, "        ],");
            i2 += 1;
        }
        let _ = writeln!(out, "    ]");
        let _ = writeln!(out, "    sum = {sum:.6}");
    }

    common_log_add(common_log_main(), LogLevel::None, &out);
    if abort_on_nan && sum.is_nan() {
        common_log_add(common_log_main(), LogLevel::None, "encountered NaN - aborting\n");
        std::process::exit(0);
    }
    Ok(())
}

/// `common_debug_cb_eval`, over a plain tensor view.
pub fn common_debug_cb_eval(t: &DebugTensor<'_>, ask: bool, user_data: &mut DebugCbUserData) -> Result<bool> {
    if ask {
        return Ok(true);
    }

    let mut matches_filter = user_data.tensor_filters.is_empty();
    if !matches_filter {
        for filter in &user_data.tensor_filters {
            if crate::quant::regex_search(t.name, filter) {
                matches_filter = true;
                break;
            }
        }
    }

    let src1_str = match t.src1 {
        Some((name, ne)) => format!("{name}{{{}}}", common_ggml_ne_string(&ne)),
        None => String::new(),
    };
    if matches_filter {
        let line = format!(
            "common_debug_cb_eval: {:>24} = ({}) {:>10}({}{{{}}}, {}) = {{{}}}\n",
            t.name,
            t.type_.name(),
            t.op_desc,
            t.src0_name,
            common_ggml_ne_string(&t.src0_ne),
            src1_str,
            common_ggml_ne_string(&t.ne),
        );
        common_log_add(common_log_main(), LogLevel::None, &line);
    }

    if !t.is_host {
        user_data.data.resize(t.data.len(), 0);
        user_data.data.copy_from_slice(t.data);
    }
    // none of the handled types are quantized, so the C++ quantized guard is always true
    if matches_filter {
        let data = if t.is_host { t.data } else { &user_data.data };
        common_debug_print_tensor(data, t.type_, &t.ne, &t.nb, 3, user_data.abort_on_nan)?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_positions_walk_codepoints() {
        let s = "aé中";
        assert_eq!(next_utf8_char_pos(s, 0), 1);
        assert_eq!(next_utf8_char_pos(s, 1), 3);
        assert_eq!(prev_utf8_char_pos(s, s.len()), 3);
        let mut advance = 0;
        assert_eq!(decode_utf8(s, 3, &mut advance), 0x4E2D);
        assert_eq!(advance, 3);
    }

    #[test]
    fn history_deduplicates_and_restores() {
        let mut h = History::new();
        h.add("one");
        h.add("one");
        h.add("two");
        assert_eq!(h.entries.len(), 2);
        h.begin_viewing("draft");
        let mut line = String::new();
        assert!(h.prev(&mut line));
        assert_eq!(line, "two");
        assert!(h.next(&mut line));
        assert_eq!(line, "draft");
    }

    #[test]
    fn float_value_reads_f32_le() {
        let bytes = 1.5f32.to_le_bytes();
        let nb = [4, 4, 4, 4];
        let v = common_ggml_get_float_value(&bytes, DebugType::F32, &nb, 0, 0, 0, 0).unwrap();
        assert_eq!(v, 1.5);
    }

    #[test]
    fn ctrl_modifier_detects_meta_key() {
        assert!(has_ctrl_modifier("1;5"));
        assert!(!has_ctrl_modifier("1;2"));
    }
}

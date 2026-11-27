//! Hand-written BPE splitters and byte encoding maps.

use crate::core::{Error, Result};
use std::sync::OnceLock;

use super::case::unicode_tolower;
use super::flags::{unicode_cpt_flags_from_cpt, unicode_cpt_is_han, UnicodeCptFlags};
use super::unicode_data::{
    AFMOE_REGEX, DIGIT_GROUP_REGEX, GPT2_REGEX, HAN_REGEX, K2_HORIZON_REGEX, LLAMA3_ALT_REGEX,
    LLAMA3_REGEX, NEWLINES_REGEX, QWEN2_REGEX, QWEN35_REGEX, RAON_REGEX,
};
use super::{
    unicode_cpt_from_utf8, unicode_cpt_to_utf8, unicode_cpts_from_utf8, unicode_cpts_to_utf8,
};

fn unicode_byte_to_utf8_map() -> Vec<String> {
    let mut map: Vec<Option<String>> = vec![None; 256];
    for ch in 0x21..=0x7e {
        map[ch] = Some(unicode_cpt_to_utf8(ch as u32).unwrap());
    }
    for ch in 0xa1..=0xac {
        map[ch] = Some(unicode_cpt_to_utf8(ch as u32).unwrap());
    }
    for ch in 0xae..=0xff {
        map[ch] = Some(unicode_cpt_to_utf8(ch as u32).unwrap());
    }
    let mut n = 0u32;
    for ch in 0..256 {
        if map[ch].is_none() {
            map[ch] = Some(unicode_cpt_to_utf8(256 + n).unwrap());
            n += 1;
        }
    }
    map.into_iter().map(|v| v.unwrap()).collect()
}

fn unicode_utf8_to_byte_map() -> std::collections::HashMap<Vec<u8>, u8> {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    for ch in 0x21..=0x7e {
        map.insert(unicode_cpt_to_utf8(ch as u32).unwrap().into_bytes(), ch as u8);
    }
    for ch in 0xa1..=0xac {
        map.insert(unicode_cpt_to_utf8(ch as u32).unwrap().into_bytes(), ch as u8);
    }
    for ch in 0xae..=0xff {
        map.insert(unicode_cpt_to_utf8(ch as u32).unwrap().into_bytes(), ch as u8);
    }
    let mut n = 0u32;
    for ch in 0..256 {
        if !map.contains_key(unicode_cpt_to_utf8(ch as u32).unwrap().as_bytes()) {
            let cpt = unicode_cpt_to_utf8(256 + n).unwrap();
            map.insert(cpt.into_bytes(), ch as u8);
            n += 1;
        }
    }
    map
}

fn unicode_cpt_to_byte_map() -> Vec<i32> {
    let mut map: Vec<i32> = Vec::new();
    for ch in 0..256u32 {
        let mut offset = 0;
        let cpt = unicode_cpt_from_utf8(unicode_byte_to_utf8(ch as u8).unwrap().as_bytes(), &mut offset).unwrap();
        let cpt = cpt as usize;
        if cpt >= map.len() {
            map.resize(cpt + 1, -1);
        }
        map[cpt] = ch as i32;
    }
    map
}

static BYTE_TO_UTF8: OnceLock<Vec<String>> = OnceLock::new();
static UTF8_TO_BYTE: OnceLock<std::collections::HashMap<Vec<u8>, u8>> = OnceLock::new();
static CPT_TO_BYTE: OnceLock<Vec<i32>> = OnceLock::new();

pub fn unicode_byte_to_utf8(byte: u8) -> Result<String> {
    BYTE_TO_UTF8
        .get_or_init(unicode_byte_to_utf8_map)
        .get(byte as usize)
        .cloned()
        .ok_or_else(|| Error::new("unicode_byte_to_utf8: byte out of range"))
}

pub fn unicode_utf8_to_byte(utf8: &[u8]) -> Result<u8> {
    UTF8_TO_BYTE
        .get_or_init(unicode_utf8_to_byte_map)
        .get(utf8)
        .copied()
        .ok_or_else(|| Error::new("unicode_utf8_to_byte: sequence not found"))
}

pub fn unicode_cpt_to_byte(cpt: u32) -> Result<u8> {
    let map = CPT_TO_BYTE.get_or_init(unicode_cpt_to_byte_map);
    match map.get(cpt as usize) {
        Some(&byte) if byte >= 0 => Ok(byte as u8),
        _ => Err(Error::new("unicode_cpt_to_byte: not a byte token code point")),
    }
}

/// State shared by the hand-written regex splitters for one pre-split segment.
struct SplitCtx<'a> {
    cpts: &'a [u32],
    offset_ini: usize,
    offset_end: usize,
    offsets: Vec<usize>,
    prev_end: usize,
}

impl<'a> SplitCtx<'a> {
    const OUT_OF_RANGE: u32 = 0xffff_ffff;

    fn new(cpts: &'a [u32], offset_ini: usize, offset_end: usize) -> Self {
        Self {
            cpts,
            offset_ini,
            offset_end,
            offsets: Vec::new(),
            prev_end: offset_ini,
        }
    }

    fn get_cpt(&self, pos: usize) -> u32 {
        if self.offset_ini <= pos && pos < self.offset_end {
            self.cpts[pos]
        } else {
            Self::OUT_OF_RANGE
        }
    }

    fn get_flags(&self, pos: usize) -> UnicodeCptFlags {
        if self.offset_ini <= pos && pos < self.offset_end {
            unicode_cpt_flags_from_cpt(self.cpts[pos])
        } else {
            UnicodeCptFlags::default()
        }
    }

    fn add_token(&mut self, end: usize) -> usize {
        debug_assert!(self.prev_end <= end && end <= self.offset_end);
        let len = end - self.prev_end;
        if len > 0 {
            self.offsets.push(len);
        }
        self.prev_end = end;
        len
    }
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
fn unicode_regex_split_custom_gpt2(text: &[u8], offsets: &[usize]) -> Vec<usize> {
    let cpts = unicode_cpts_from_utf8(text);
    let mut bpe_offsets = Vec::with_capacity(offsets.len());
    let mut start = 0;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        debug_assert!(offset_end <= cpts.len());
        start = offset_end;
        let mut ctx = SplitCtx::new(&cpts, offset_ini, offset_end);
        let mut pos = offset_ini;

        while pos < offset_end {
            let cpt = ctx.get_cpt(pos);
            let flags = ctx.get_flags(pos);

            if cpt == b'\'' as u32 && pos + 1 < offset_end {
                let cpt_next = ctx.get_cpt(pos + 1);
                if cpt_next == b's' as u32
                    || cpt_next == b't' as u32
                    || cpt_next == b'm' as u32
                    || cpt_next == b'd' as u32
                {
                    let len = ctx.add_token(pos + 2);
                    pos += len;
                    continue;
                }
                if pos + 2 < offset_end {
                    let cpt_next_next = ctx.get_cpt(pos + 2);
                    if (cpt_next == b'r' as u32 && cpt_next_next == b'e' as u32)
                        || (cpt_next == b'v' as u32 && cpt_next_next == b'e' as u32)
                        || (cpt_next == b'l' as u32 && cpt_next_next == b'l' as u32)
                    {
                        let len = ctx.add_token(pos + 3);
                        pos += len;
                        continue;
                    }
                }
            }

            let mut flags2 = if cpt == b' ' as u32 {
                ctx.get_flags(pos + 1)
            } else {
                flags
            };

            if flags2.is_letter() {
                if cpt == b' ' as u32 {
                    pos += 1;
                }
                while flags2.is_letter() {
                    pos += 1;
                    flags2 = ctx.get_flags(pos);
                }
                ctx.add_token(pos);
                continue;
            }
            if flags2.is_number() {
                if cpt == b' ' as u32 {
                    pos += 1;
                }
                while flags2.is_number() {
                    pos += 1;
                    flags2 = ctx.get_flags(pos);
                }
                ctx.add_token(pos);
                continue;
            }
            let excluded = flags2.is_whitespace() | flags2.is_letter() | flags2.is_number();
            if !excluded && flags2.as_uint() != 0 {
                if cpt == b' ' as u32 {
                    pos += 1;
                }
                while {
                    let f = ctx.get_flags(pos);
                    !(f.is_whitespace() | f.is_letter() | f.is_number()) && f.as_uint() != 0
                } {
                    pos += 1;
                }
                ctx.add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0;
            while ctx.get_flags(pos + num_whitespaces).is_whitespace() {
                num_whitespaces += 1;
            }
            if num_whitespaces > 1
                && ctx.get_cpt(pos + num_whitespaces) != SplitCtx::OUT_OF_RANGE
            {
                pos += num_whitespaces - 1;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                ctx.add_token(pos);
                continue;
            }
            pos += 1;
            ctx.add_token(pos);
        }
        bpe_offsets.extend(ctx.offsets);
    }
    bpe_offsets
}

// LLAMA3 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
fn unicode_regex_split_custom_llama3(text: &[u8], offsets: &[usize]) -> Vec<usize> {
    let cpts = unicode_cpts_from_utf8(text);
    let mut bpe_offsets = Vec::with_capacity(offsets.len());
    let mut start = 0;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        debug_assert!(offset_end <= cpts.len());
        start = offset_end;
        let mut ctx = SplitCtx::new(&cpts, offset_ini, offset_end);
        let mut pos = offset_ini;

        while pos < offset_end {
            let cpt = ctx.get_cpt(pos);
            let flags = ctx.get_flags(pos);

            if cpt == b'\'' as u32 && pos + 1 < offset_end {
                let cpt_next = unicode_tolower(ctx.get_cpt(pos + 1));
                if cpt_next == b's' as u32
                    || cpt_next == b't' as u32
                    || cpt_next == b'm' as u32
                    || cpt_next == b'd' as u32
                {
                    let len = ctx.add_token(pos + 2);
                    pos += len;
                    continue;
                }
                if pos + 2 < offset_end {
                    let cpt_next_next = unicode_tolower(ctx.get_cpt(pos + 2));
                    if (cpt_next == b'r' as u32 && cpt_next_next == b'e' as u32)
                        || (cpt_next == b'v' as u32 && cpt_next_next == b'e' as u32)
                        || (cpt_next == b'l' as u32 && cpt_next_next == b'l' as u32)
                    {
                        let len = ctx.add_token(pos + 3);
                        pos += len;
                        continue;
                    }
                }
            }

            if cpt != b'\r' as u32 && cpt != b'\n' as u32 && !flags.is_number() {
                if flags.is_letter() || ctx.get_flags(pos + 1).is_letter() {
                    pos += 1;
                    while ctx.get_flags(pos).is_letter() {
                        pos += 1;
                    }
                    ctx.add_token(pos);
                    continue;
                }
            }

            if flags.is_number() {
                let mut ini = pos;
                while ctx.get_flags(pos).is_number() {
                    pos += 1;
                    if pos - ini >= 3 {
                        ctx.add_token(pos);
                        ini = pos;
                    }
                }
                ctx.add_token(pos);
                continue;
            }

            let mut flags2 = if cpt == b' ' as u32 {
                ctx.get_flags(pos + 1)
            } else {
                flags
            };
            let excluded = flags2.is_whitespace() | flags2.is_letter() | flags2.is_number();
            if !excluded && flags.as_uint() != 0 {
                if cpt == b' ' as u32 {
                    pos += 1;
                }
                while {
                    let f = ctx.get_flags(pos);
                    !(f.is_whitespace() | f.is_letter() | f.is_number()) && f.as_uint() != 0
                } {
                    pos += 1;
                    flags2 = ctx.get_flags(pos);
                }
                let mut cpt2 = ctx.get_cpt(pos);
                while cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                    pos += 1;
                    cpt2 = ctx.get_cpt(pos);
                }
                ctx.add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0;
            let mut last_end_r_or_n = 0;
            while ctx.get_flags(pos + num_whitespaces).is_whitespace() {
                let cpt2 = ctx.get_cpt(pos + num_whitespaces);
                if cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces += 1;
            }
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 1
                && ctx.get_cpt(pos + num_whitespaces) != SplitCtx::OUT_OF_RANGE
            {
                pos += num_whitespaces - 1;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                ctx.add_token(pos);
                continue;
            }
            pos += 1;
            ctx.add_token(pos);
        }
        bpe_offsets.extend(ctx.offsets);
    }
    bpe_offsets
}

fn is_k2_letter(ctx: &SplitCtx<'_>, pos: usize) -> bool {
    let c = ctx.get_cpt(pos);
    if c == 0x200c || c == 0x200d {
        return true;
    }
    let flags = ctx.get_flags(pos);
    flags.is_letter() || flags.is_accent_mark()
}

// K2-Horizon: letter runs are (?:\p{L}|\p{M}|\u200C|\u200D)+
fn unicode_regex_split_custom_k2_horizon(text: &[u8], offsets: &[usize]) -> Vec<usize> {
    let cpts = unicode_cpts_from_utf8(text);
    let mut bpe_offsets = Vec::with_capacity(offsets.len());
    let mut start = 0;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        debug_assert!(offset_end <= cpts.len());
        start = offset_end;
        let mut ctx = SplitCtx::new(&cpts, offset_ini, offset_end);
        let mut pos = offset_ini;

        while pos < offset_end {
            let cpt = ctx.get_cpt(pos);
            let flags = ctx.get_flags(pos);

            if cpt == b'\'' as u32 && pos + 1 < offset_end {
                let mut cpt_next = unicode_tolower(ctx.get_cpt(pos + 1));
                if cpt_next == 0x017f {
                    cpt_next = b's' as u32;
                }
                if cpt_next == b's' as u32
                    || cpt_next == b't' as u32
                    || cpt_next == b'm' as u32
                    || cpt_next == b'd' as u32
                {
                    let len = ctx.add_token(pos + 2);
                    pos += len;
                    continue;
                }
                if pos + 2 < offset_end {
                    let cpt_next_next = unicode_tolower(ctx.get_cpt(pos + 2));
                    if (cpt_next == b'r' as u32 && cpt_next_next == b'e' as u32)
                        || (cpt_next == b'v' as u32 && cpt_next_next == b'e' as u32)
                        || (cpt_next == b'l' as u32 && cpt_next_next == b'l' as u32)
                    {
                        let len = ctx.add_token(pos + 3);
                        pos += len;
                        continue;
                    }
                }
            }

            if cpt != b'\r' as u32 && cpt != b'\n' as u32 && !flags.is_number() {
                if is_k2_letter(&ctx, pos) || is_k2_letter(&ctx, pos + 1) {
                    pos += 1;
                    while is_k2_letter(&ctx, pos) {
                        pos += 1;
                    }
                    ctx.add_token(pos);
                    continue;
                }
            }

            if flags.is_number() {
                let mut ini = pos;
                while ctx.get_flags(pos).is_number() {
                    pos += 1;
                    if pos - ini >= 3 {
                        ctx.add_token(pos);
                        ini = pos;
                    }
                }
                ctx.add_token(pos);
                continue;
            }

            let flags2 = if cpt == b' ' as u32 {
                ctx.get_flags(pos + 1)
            } else {
                flags
            };
            let excluded = flags2.is_whitespace() | flags2.is_letter() | flags2.is_number();
            if !excluded && flags.as_uint() != 0 {
                if cpt == b' ' as u32 {
                    pos += 1;
                }
                while {
                    let f = ctx.get_flags(pos);
                    !(f.is_whitespace() | f.is_letter() | f.is_number()) && f.as_uint() != 0
                } {
                    pos += 1;
                }
                let mut cpt2 = ctx.get_cpt(pos);
                while cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                    pos += 1;
                    cpt2 = ctx.get_cpt(pos);
                }
                ctx.add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0;
            let mut last_end_r_or_n = 0;
            while ctx.get_flags(pos + num_whitespaces).is_whitespace() {
                let cpt2 = ctx.get_cpt(pos + num_whitespaces);
                if cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces += 1;
            }
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 1
                && ctx.get_cpt(pos + num_whitespaces) != SplitCtx::OUT_OF_RANGE
            {
                pos += num_whitespaces - 1;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                ctx.add_token(pos);
                continue;
            }
            pos += 1;
            ctx.add_token(pos);
        }
        bpe_offsets.extend(ctx.offsets);
    }
    bpe_offsets
}

// Qwen2 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
fn unicode_regex_split_custom_qwen2(text: &[u8], offsets: &[usize]) -> Vec<usize> {
    let cpts = unicode_cpts_from_utf8(text);
    let mut bpe_offsets = Vec::with_capacity(offsets.len());
    let mut start = 0;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        debug_assert!(offset_end <= cpts.len());
        start = offset_end;
        let mut ctx = SplitCtx::new(&cpts, offset_ini, offset_end);
        let mut pos = offset_ini;

        while pos < offset_end {
            let cpt = ctx.get_cpt(pos);
            let flags = ctx.get_flags(pos);

            if cpt == b'\'' as u32 && pos + 1 < offset_end {
                let cpt_next = unicode_tolower(ctx.get_cpt(pos + 1));
                if cpt_next == b's' as u32
                    || cpt_next == b't' as u32
                    || cpt_next == b'm' as u32
                    || cpt_next == b'd' as u32
                {
                    let len = ctx.add_token(pos + 2);
                    pos += len;
                    continue;
                }
                if pos + 2 < offset_end {
                    let cpt_next_next = unicode_tolower(ctx.get_cpt(pos + 2));
                    if (cpt_next == b'r' as u32 && cpt_next_next == b'e' as u32)
                        || (cpt_next == b'v' as u32 && cpt_next_next == b'e' as u32)
                        || (cpt_next == b'l' as u32 && cpt_next_next == b'l' as u32)
                    {
                        let len = ctx.add_token(pos + 3);
                        pos += len;
                        continue;
                    }
                }
            }

            if cpt != b'\r' as u32 && cpt != b'\n' as u32 && !flags.is_number() {
                if flags.is_letter() || ctx.get_flags(pos + 1).is_letter() {
                    pos += 1;
                    while ctx.get_flags(pos).is_letter() {
                        pos += 1;
                    }
                    ctx.add_token(pos);
                    continue;
                }
            }

            if flags.is_number() {
                pos += 1;
                ctx.add_token(pos);
                continue;
            }

            let flags2 = if cpt == b' ' as u32 {
                ctx.get_flags(pos + 1)
            } else {
                flags
            };
            let excluded = flags2.is_whitespace() | flags2.is_letter() | flags2.is_number();
            if !excluded && flags.as_uint() != 0 {
                if cpt == b' ' as u32 {
                    pos += 1;
                }
                while {
                    let f = ctx.get_flags(pos);
                    !(f.is_whitespace() | f.is_letter() | f.is_number()) && f.as_uint() != 0
                } {
                    pos += 1;
                }
                let mut cpt2 = ctx.get_cpt(pos);
                while cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                    pos += 1;
                    cpt2 = ctx.get_cpt(pos);
                }
                ctx.add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0;
            let mut last_end_r_or_n = 0;
            while ctx.get_flags(pos + num_whitespaces).is_whitespace() {
                let cpt2 = ctx.get_cpt(pos + num_whitespaces);
                if cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces += 1;
            }
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 1
                && ctx.get_cpt(pos + num_whitespaces) != SplitCtx::OUT_OF_RANGE
            {
                pos += num_whitespaces - 1;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                ctx.add_token(pos);
                continue;
            }
            pos += 1;
            ctx.add_token(pos);
        }
        bpe_offsets.extend(ctx.offsets);
    }
    bpe_offsets
}

// Qwen3.5: letter runs also consume Unicode combining marks (\p{M}).
fn unicode_regex_split_custom_qwen35(text: &[u8], offsets: &[usize]) -> Vec<usize> {
    let cpts = unicode_cpts_from_utf8(text);
    let mut bpe_offsets = Vec::with_capacity(offsets.len());
    let mut start = 0;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        debug_assert!(offset_end <= cpts.len());
        start = offset_end;
        let mut ctx = SplitCtx::new(&cpts, offset_ini, offset_end);
        let mut pos = offset_ini;

        while pos < offset_end {
            let cpt = ctx.get_cpt(pos);
            let flags = ctx.get_flags(pos);

            if cpt == b'\'' as u32 && pos + 1 < offset_end {
                let cpt_next = unicode_tolower(ctx.get_cpt(pos + 1));
                if cpt_next == b's' as u32
                    || cpt_next == b't' as u32
                    || cpt_next == b'm' as u32
                    || cpt_next == b'd' as u32
                {
                    let len = ctx.add_token(pos + 2);
                    pos += len;
                    continue;
                }
                if pos + 2 < offset_end {
                    let cpt_next_next = unicode_tolower(ctx.get_cpt(pos + 2));
                    if (cpt_next == b'r' as u32 && cpt_next_next == b'e' as u32)
                        || (cpt_next == b'v' as u32 && cpt_next_next == b'e' as u32)
                        || (cpt_next == b'l' as u32 && cpt_next_next == b'l' as u32)
                    {
                        let len = ctx.add_token(pos + 3);
                        pos += len;
                        continue;
                    }
                }
            }

            if cpt != b'\r' as u32 && cpt != b'\n' as u32 && !flags.is_number() {
                let next = ctx.get_flags(pos + 1);
                if flags.is_letter()
                    || flags.is_accent_mark()
                    || next.is_accent_mark()
                    || next.is_letter()
                {
                    pos += 1;
                    while ctx.get_flags(pos).is_letter() || ctx.get_flags(pos).is_accent_mark() {
                        pos += 1;
                    }
                    ctx.add_token(pos);
                    continue;
                }
            }

            if flags.is_number() {
                pos += 1;
                ctx.add_token(pos);
                continue;
            }

            let flags2 = if cpt == b' ' as u32 {
                ctx.get_flags(pos + 1)
            } else {
                flags
            };
            let excluded = flags2.is_whitespace()
                | flags2.is_letter()
                | flags2.is_accent_mark()
                | flags2.is_number();
            if !excluded && flags.as_uint() != 0 {
                if cpt == b' ' as u32 {
                    pos += 1;
                }
                while {
                    let f = ctx.get_flags(pos);
                    !(f.is_whitespace()
                        | f.is_letter()
                        | f.is_accent_mark()
                        | f.is_number())
                        && f.as_uint() != 0
                } {
                    pos += 1;
                }
                let mut cpt2 = ctx.get_cpt(pos);
                while cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                    pos += 1;
                    cpt2 = ctx.get_cpt(pos);
                }
                ctx.add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0;
            let mut last_end_r_or_n = 0;
            while ctx.get_flags(pos + num_whitespaces).is_whitespace() {
                let cpt2 = ctx.get_cpt(pos + num_whitespaces);
                if cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces += 1;
            }
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 1
                && ctx.get_cpt(pos + num_whitespaces) != SplitCtx::OUT_OF_RANGE
            {
                pos += num_whitespaces - 1;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                ctx.add_token(pos);
                continue;
            }
            pos += 1;
            ctx.add_token(pos);
        }
        bpe_offsets.extend(ctx.offsets);
    }
    bpe_offsets
}

#[allow(clippy::too_many_lines)]
fn unicode_regex_split_custom_kimi_k2(text: &[u8], offsets: &[usize]) -> Vec<usize> {
    let cpts = unicode_cpts_from_utf8(text);
    let mut bpe_offsets = Vec::with_capacity(offsets.len());
    let mut start = 0;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        debug_assert!(offset_end <= cpts.len());
        start = offset_end;
        let mut ctx = SplitCtx::new(&cpts, offset_ini, offset_end);
        let mut pos = offset_ini;

        while pos < offset_end {
            let cpt = ctx.get_cpt(pos);
            let flags = ctx.get_flags(pos);

            // [\p{Han}]+
            if unicode_cpt_is_han(cpt) {
                while unicode_cpt_is_han(ctx.get_cpt(pos)) {
                    pos += 1;
                }
                ctx.add_token(pos);
                continue;
            }

            // Letter words excluding Han characters, with optional contractions.
            let is_letter_pattern = (flags.is_letter() && !unicode_cpt_is_han(cpt))
                || (!(cpt == b'\r' as u32
                    || cpt == b'\n' as u32
                    || flags.is_letter()
                    || flags.is_number())
                    && ctx.get_flags(pos + 1).is_letter()
                    && !unicode_cpt_is_han(ctx.get_cpt(pos + 1)));
            if is_letter_pattern {
                let mut has_leading_char = false;
                if !(cpt == b'\r' as u32
                    || cpt == b'\n' as u32
                    || flags.is_letter()
                    || flags.is_number())
                {
                    has_leading_char = true;
                    pos += 1;
                }

                let mut has_letters = false;
                while ctx.get_flags(pos).is_letter() && !unicode_cpt_is_han(ctx.get_cpt(pos)) {
                    has_letters = true;
                    pos += 1;
                }

                if has_letters
                    || (!has_leading_char
                        && ctx.get_flags(pos).is_letter()
                        && !unicode_cpt_is_han(ctx.get_cpt(pos)))
                {
                    if !has_letters {
                        pos += 1;
                    }
                    while ctx.get_flags(pos).is_letter() && !unicode_cpt_is_han(ctx.get_cpt(pos)) {
                        pos += 1;
                    }

                    if ctx.get_cpt(pos) == b'\'' as u32 && pos + 1 < offset_end {
                        let cpt_next = unicode_tolower(ctx.get_cpt(pos + 1));
                        if cpt_next == b's' as u32
                            || cpt_next == b't' as u32
                            || cpt_next == b'm' as u32
                            || cpt_next == b'd' as u32
                        {
                            pos += 2;
                        } else if pos + 2 < offset_end {
                            let cpt_next_next = unicode_tolower(ctx.get_cpt(pos + 2));
                            if (cpt_next == b'r' as u32 && cpt_next_next == b'e' as u32)
                                || (cpt_next == b'v' as u32 && cpt_next_next == b'e' as u32)
                                || (cpt_next == b'l' as u32 && cpt_next_next == b'l' as u32)
                            {
                                pos += 3;
                            }
                        }
                    }
                    ctx.add_token(pos);
                    continue;
                } else if has_leading_char {
                    pos -= 1;
                }
            }

            if flags.is_number() {
                let mut ini = pos;
                while ctx.get_flags(pos).is_number() {
                    pos += 1;
                    if pos - ini >= 3 {
                        ctx.add_token(pos);
                        ini = pos;
                    }
                }
                ctx.add_token(pos);
                continue;
            }

            let flags2 = if cpt == b' ' as u32 {
                ctx.get_flags(pos + 1)
            } else {
                flags
            };
            if !(flags2.is_whitespace() || flags2.is_letter() || flags2.is_number())
                && flags2.as_uint() != 0
            {
                if cpt == b' ' as u32 {
                    pos += 1;
                }
                while {
                    let f = ctx.get_flags(pos);
                    !(f.is_whitespace() || f.is_letter() || f.is_number()) && f.as_uint() != 0
                } {
                    pos += 1;
                }
                let mut cpt2 = ctx.get_cpt(pos);
                while cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                    pos += 1;
                    cpt2 = ctx.get_cpt(pos);
                }
                ctx.add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0;
            let mut last_end_r_or_n = 0;
            while ctx.get_flags(pos + num_whitespaces).is_whitespace() {
                let cpt2 = ctx.get_cpt(pos + num_whitespaces);
                if cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces += 1;
            }
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 1
                && ctx.get_cpt(pos + num_whitespaces) != SplitCtx::OUT_OF_RANGE
            {
                pos += num_whitespaces - 1;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                ctx.add_token(pos);
                continue;
            }
            pos += 1;
            ctx.add_token(pos);
        }
        bpe_offsets.extend(ctx.offsets);
    }
    bpe_offsets
}

// AFMOE: split digit runs with a leading 1-2 digits based on total length mod 3.
fn unicode_regex_split_custom_afmoe(text: &[u8], offsets: &[usize]) -> Vec<usize> {
    let cpts = unicode_cpts_from_utf8(text);
    let mut bpe_offsets = Vec::with_capacity(offsets.len());
    let mut start = 0;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        debug_assert!(offset_end <= cpts.len());
        start = offset_end;
        let mut ctx = SplitCtx::new(&cpts, offset_ini, offset_end);
        let mut pos = offset_ini;

        while pos < offset_end {
            if ctx.get_flags(pos).is_number() {
                let digit_start = pos;
                let mut digit_count = 0;
                while ctx.get_flags(pos).is_number() && pos < offset_end {
                    digit_count += 1;
                    pos += 1;
                }
                let remainder = digit_count % 3;
                let mut current = digit_start;
                if remainder > 0 {
                    ctx.add_token(current + remainder);
                    current += remainder;
                }
                while current < digit_start + digit_count {
                    ctx.add_token(current + 3);
                    current += 3;
                }
                continue;
            }
            pos += 1;
        }
        if ctx.prev_end < offset_end {
            ctx.add_token(offset_end);
        }
        bpe_offsets.extend(ctx.offsets);
    }
    bpe_offsets
}

// regex: [^\n]+|[\n]+
fn unicode_regex_split_custom_newlines(text: &[u8], offsets: &[usize]) -> Vec<usize> {
    let cpts = unicode_cpts_from_utf8(text);
    let mut bpe_offsets = Vec::with_capacity(offsets.len());
    let mut start = 0;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        debug_assert!(offset_end <= cpts.len());
        start = offset_end;
        let mut pos = offset_ini;
        while pos < offset_end {
            let is_newline = cpts[pos] == b'\n' as u32;
            let run_start = pos;
            while pos < offset_end && (cpts[pos] == b'\n' as u32) == is_newline {
                pos += 1;
            }
            bpe_offsets.push(pos - run_start);
        }
    }
    bpe_offsets
}

fn raon_is_cjk(ctx: &SplitCtx<'_>, pos: usize) -> bool {
    let cpt = ctx.get_cpt(pos);
    (0x4e00..=0x9fa5).contains(&cpt) || (0x3040..=0x30ff).contains(&cpt)
}

fn raon_is_letter(ctx: &SplitCtx<'_>, pos: usize) -> bool {
    ctx.get_flags(pos).is_letter() && !raon_is_cjk(ctx, pos)
}

fn raon_is_punct(ctx: &SplitCtx<'_>, pos: usize) -> bool {
    if ctx.offset_ini > pos || pos >= ctx.offset_end {
        return false;
    }
    let flags = ctx.get_flags(pos);
    !flags.is_whitespace() && !flags.is_letter() && !flags.is_number()
}

// Raon-1 mini: CJK runs, letter runs with marks, digits grouped by 3 from right.
fn unicode_regex_split_custom_raon(text: &[u8], offsets: &[usize]) -> Vec<usize> {
    let cpts = unicode_cpts_from_utf8(text);
    let mut bpe_offsets = Vec::with_capacity(offsets.len());
    let mut start = 0;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        debug_assert!(offset_end <= cpts.len());
        start = offset_end;
        let mut ctx = SplitCtx::new(&cpts, offset_ini, offset_end);
        let mut pos = offset_ini;

        while pos < offset_end {
            let cpt = ctx.get_cpt(pos);
            let flags = ctx.get_flags(pos);
            let mut pos2 = pos + if cpt == b' ' as u32 { 1 } else { 0 };

            if raon_is_cjk(&ctx, pos2) {
                loop {
                    pos2 += 1;
                    if !raon_is_cjk(&ctx, pos2) {
                        break;
                    }
                }
                pos += ctx.add_token(pos2);
                continue;
            }

            if raon_is_letter(&ctx, pos2) {
                loop {
                    pos2 += 1;
                    while ctx.get_flags(pos2).is_accent_mark() {
                        pos2 += 1;
                    }
                    if raon_is_letter(&ctx, pos2) {
                        continue;
                    }
                    if ctx.get_cpt(pos2) == b' ' as u32 && raon_is_letter(&ctx, pos2 + 1) {
                        pos2 += 1;
                        continue;
                    }
                    break;
                }
                pos += ctx.add_token(pos2);
                continue;
            }

            if flags.is_number() {
                let digit_start = pos;
                let mut digit_count = 0;
                while ctx.get_flags(pos).is_number() && pos < offset_end {
                    digit_count += 1;
                    pos += 1;
                }
                let remainder = digit_count % 3;
                let mut current = digit_start;
                if remainder > 0 {
                    ctx.add_token(current + remainder);
                    current += remainder;
                }
                while current < digit_start + digit_count {
                    ctx.add_token(current + 3);
                    current += 3;
                }
                continue;
            }

            if raon_is_punct(&ctx, pos2) {
                loop {
                    pos2 += 1;
                    if !raon_is_punct(&ctx, pos2) {
                        break;
                    }
                }
                let mut cpt2 = ctx.get_cpt(pos2);
                while cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 || cpt2 == b'/' as u32 {
                    pos2 += 1;
                    cpt2 = ctx.get_cpt(pos2);
                }
                pos += ctx.add_token(pos2);
                continue;
            }

            let mut num_whitespaces = 0;
            let mut last_end_r_or_n = 0;
            while ctx.get_flags(pos + num_whitespaces).is_whitespace() {
                let cpt2 = ctx.get_cpt(pos + num_whitespaces);
                if cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces += 1;
            }
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 1
                && ctx.get_cpt(pos + num_whitespaces) != SplitCtx::OUT_OF_RANGE
            {
                pos += num_whitespaces - 1;
                ctx.add_token(pos);
                continue;
            }
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                ctx.add_token(pos);
                continue;
            }
            pos += 1;
            ctx.add_token(pos);
        }
        bpe_offsets.extend(ctx.offsets);
    }
    bpe_offsets
}

type CustomSplit = fn(&[u8], &[usize]) -> Vec<usize>;


/// Select the hand-written splitter matching a known tokenizer regex.
fn unicode_regex_split_custom(regex_expr: &str) -> Option<CustomSplit> {
    if regex_expr == GPT2_REGEX {
        return Some(unicode_regex_split_custom_gpt2);
    }
    if regex_expr == LLAMA3_REGEX
        || regex_expr == LLAMA3_ALT_REGEX
    {
        return Some(unicode_regex_split_custom_llama3);
    }
    if regex_expr == QWEN2_REGEX
    {
        return Some(unicode_regex_split_custom_qwen2);
    }
    if regex_expr == QWEN35_REGEX
    {
        return Some(unicode_regex_split_custom_qwen35);
    }
    if regex_expr == K2_HORIZON_REGEX
    {
        return Some(unicode_regex_split_custom_k2_horizon);
    }
    if regex_expr == HAN_REGEX {
        return Some(unicode_regex_split_custom_kimi_k2);
    }
    if regex_expr == AFMOE_REGEX || regex_expr == DIGIT_GROUP_REGEX {
        return Some(unicode_regex_split_custom_afmoe);
    }
    if regex_expr == NEWLINES_REGEX {
        return Some(unicode_regex_split_custom_newlines);
    }
    if regex_expr == RAON_REGEX {
        return Some(unicode_regex_split_custom_raon);
    }
    None
}

fn unicode_byte_encoding_process(bpe_words: &[Vec<u8>]) -> Result<Vec<String>> {
    let mut bpe_encoded_words = Vec::with_capacity(bpe_words.len());
    for word in bpe_words {
        let cpts = unicode_cpts_from_utf8(word);
        let text_utf = unicode_cpts_to_utf8(&cpts)?;
        let mut encoded_token = String::new();
        for &byte in &text_utf {
            encoded_token.push_str(&unicode_byte_to_utf8(byte)?);
        }
        bpe_encoded_words.push(encoded_token);
    }
    Ok(bpe_encoded_words)
}

/// Split text into BPE words according to the given tokenizer regexes.
///
/// Known regexes use the hand-written splitters from the original. The generic
/// `std::regex` fallback has no equivalent in the Rust standard library, so an
/// unknown expression returns an error instead of silently changing the split.
pub fn unicode_regex_split(
    text: &[u8],
    regex_exprs: &[&str],
    byte_encode: bool,
) -> Result<Vec<String>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let cpts = unicode_cpts_from_utf8(text);
    let mut bpe_offsets = vec![cpts.len()];

    for &regex_expr in regex_exprs {
        let splitter = unicode_regex_split_custom(regex_expr).ok_or_else(|| {
            Error::new(format!(
                "unicode_regex_split: unsupported regex '{}' (no generic regex engine)",
                regex_expr
            ))
        })?;
        let tmp = splitter(text, &bpe_offsets);
        if !tmp.is_empty() {
            bpe_offsets = tmp;
        }
    }

    let mut bpe_words = Vec::with_capacity(bpe_offsets.len());
    let mut start = 0;
    for &offset in &bpe_offsets {
        let end = start + offset;
        bpe_words.push(unicode_cpts_to_utf8(&cpts[start..end])?);
        start = end;
    }

    if byte_encode {
        unicode_byte_encoding_process(&bpe_words)
    } else {
        bpe_words
            .iter()
            .map(|word| String::from_utf8(word.clone()).map_err(|_| Error::new("invalid UTF-8")))
            .collect()
    }
}

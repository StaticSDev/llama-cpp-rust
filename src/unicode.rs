//! Port of `unicode.cpp`.
//!
//! Text arguments are byte slices, as in C++ `std::string`. Codepoint tables
//! live in `unicode_data.rs`; the hash maps of the original are sorted arrays
//! here, so lookups use binary search instead of `std::collections`.

pub mod unicode_data;

use crate::core::{Error, Result};
use std::sync::OnceLock;

use self::unicode_data::{
    MAX_CODEPOINTS, UNICODE_COMBINING_CLASS, UNICODE_COMPOSITION, UNICODE_DECOMPOSITION,
    UNICODE_MAP_LOWERCASE, UNICODE_MAP_UPPERCASE, UNICODE_RANGES_FLAGS, UNICODE_RANGES_NFD,
    UNICODE_SET_WHITESPACE,
};

/// Flags for one Unicode codepoint. Layout matches the C++ bitfield order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnicodeCptFlags(u16);

impl UnicodeCptFlags {
    pub const UNDEFINED: u16 = 0x0001;
    pub const NUMBER: u16 = 0x0002;
    pub const LETTER: u16 = 0x0004;
    pub const SEPARATOR: u16 = 0x0008;
    pub const ACCENT_MARK: u16 = 0x0010;
    pub const PUNCTUATION: u16 = 0x0020;
    pub const SYMBOL: u16 = 0x0040;
    pub const CONTROL: u16 = 0x0080;
    pub const MASK_CATEGORIES: u16 = 0x00ff;
    pub const WHITESPACE: u16 = 0x0100;
    pub const LOWERCASE: u16 = 0x0200;
    pub const UPPERCASE: u16 = 0x0400;
    pub const NFD: u16 = 0x0800;

    pub const fn new(flags: u16) -> Self {
        Self(flags)
    }

    pub const fn as_uint(self) -> u16 {
        self.0
    }

    pub const fn category_flag(self) -> u16 {
        self.0 & Self::MASK_CATEGORIES
    }

    pub const fn is_undefined(self) -> bool {
        self.0 & Self::UNDEFINED != 0
    }

    pub const fn is_number(self) -> bool {
        self.0 & Self::NUMBER != 0
    }

    pub const fn is_letter(self) -> bool {
        self.0 & Self::LETTER != 0
    }

    pub const fn is_separator(self) -> bool {
        self.0 & Self::SEPARATOR != 0
    }

    pub const fn is_accent_mark(self) -> bool {
        self.0 & Self::ACCENT_MARK != 0
    }

    pub const fn is_punctuation(self) -> bool {
        self.0 & Self::PUNCTUATION != 0
    }

    pub const fn is_symbol(self) -> bool {
        self.0 & Self::SYMBOL != 0
    }

    pub const fn is_control(self) -> bool {
        self.0 & Self::CONTROL != 0
    }

    pub const fn is_whitespace(self) -> bool {
        self.0 & Self::WHITESPACE != 0
    }

    pub const fn is_lowercase(self) -> bool {
        self.0 & Self::LOWERCASE != 0
    }

    pub const fn is_uppercase(self) -> bool {
        self.0 & Self::UPPERCASE != 0
    }

    pub const fn is_nfd(self) -> bool {
        self.0 & Self::NFD != 0
    }

    fn set(&mut self, flag: u16, value: bool) {
        if value {
            self.0 |= flag;
        } else {
            self.0 &= !flag;
        }
    }
}

fn invalid_character() -> Error {
    Error::new("invalid character")
}

/// Length in bytes of the UTF-8 sequence starting with `src`.
pub fn unicode_len_utf8(src: u8) -> usize {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[(src >> 4) as usize]
}

pub fn unicode_cpt_to_utf8(cpt: u32) -> Result<String> {
    if cpt <= 0x7f {
        return Ok(char::from(cpt as u8).to_string());
    }
    if cpt <= 0x7ff {
        return Ok(String::from_utf8(vec![
            (0xc0 | ((cpt >> 6) & 0x1f)) as u8,
            (0x80 | (cpt & 0x3f)) as u8,
        ])
        .unwrap());
    }
    if cpt <= 0xffff {
        return Ok(String::from_utf8(vec![
            (0xe0 | ((cpt >> 12) & 0x0f)) as u8,
            (0x80 | ((cpt >> 6) & 0x3f)) as u8,
            (0x80 | (cpt & 0x3f)) as u8,
        ])
        .unwrap());
    }
    if cpt <= 0x10ffff {
        return Ok(String::from_utf8(vec![
            (0xf0 | ((cpt >> 18) & 0x07)) as u8,
            (0x80 | ((cpt >> 12) & 0x3f)) as u8,
            (0x80 | ((cpt >> 6) & 0x3f)) as u8,
            (0x80 | (cpt & 0x3f)) as u8,
        ])
        .unwrap());
    }
    Err(Error::new("invalid codepoint"))
}

fn utf8_byte(utf8: &[u8], pos: usize) -> u8 {
    utf8[pos]
}

/// Decode one codepoint at `offset` and advance the offset.
pub fn unicode_cpt_from_utf8(utf8: &[u8], offset: &mut usize) -> Result<u32> {
    if *offset >= utf8.len() {
        return Err(invalid_character());
    }

    let b0 = utf8_byte(utf8, *offset);
    if b0 & 0x80 == 0 {
        *offset += 1;
        return Ok(b0 as u32);
    }
    if b0 & 0x40 == 0 {
        return Err(invalid_character());
    }

    let is_not_cont = |pos: usize| (utf8_byte(utf8, *offset + pos) & 0xc0) != 0x80;

    if b0 & 0x20 == 0 {
        if *offset + 1 >= utf8.len() || is_not_cont(1) {
            return Err(invalid_character());
        }
        let result = ((b0 as u32 & 0x1f) << 6) | (utf8_byte(utf8, *offset + 1) as u32 & 0x3f);
        if result < 0x80 {
            return Err(invalid_character());
        }
        *offset += 2;
        return Ok(result);
    }
    if b0 & 0x10 == 0 {
        if *offset + 2 >= utf8.len() || is_not_cont(1) || is_not_cont(2) {
            return Err(invalid_character());
        }
        let result = ((b0 as u32 & 0x0f) << 12)
            | ((utf8_byte(utf8, *offset + 1) as u32 & 0x3f) << 6)
            | (utf8_byte(utf8, *offset + 2) as u32 & 0x3f);
        if result < 0x800 {
            return Err(invalid_character());
        }
        *offset += 3;
        return Ok(result);
    }
    if b0 & 0x08 == 0 {
        if *offset + 3 >= utf8.len() || is_not_cont(1) || is_not_cont(2) || is_not_cont(3) {
            return Err(invalid_character());
        }
        let result = ((b0 as u32 & 0x07) << 18)
            | ((utf8_byte(utf8, *offset + 1) as u32 & 0x3f) << 12)
            | ((utf8_byte(utf8, *offset + 2) as u32 & 0x3f) << 6)
            | (utf8_byte(utf8, *offset + 3) as u32 & 0x3f);
        if !(0x10000..=0x10ffff).contains(&result) {
            return Err(invalid_character());
        }
        *offset += 4;
        return Ok(result);
    }
    Err(Error::new("failed to convert utf8 to codepoint"))
}

fn unicode_cpts_to_utf8(cps: &[u32]) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    for &cp in cps {
        result.extend_from_slice(unicode_cpt_to_utf8(cp)?.as_bytes());
    }
    Ok(result)
}

fn unicode_cpt_flags_array() -> Vec<UnicodeCptFlags> {
    let mut cpt_flags = vec![UnicodeCptFlags::new(UnicodeCptFlags::UNDEFINED); MAX_CODEPOINTS as usize];

    debug_assert_eq!(UNICODE_RANGES_FLAGS[0].0, 0);
    debug_assert_eq!(UNICODE_RANGES_FLAGS.last().unwrap().0, MAX_CODEPOINTS);
    for i in 1..UNICODE_RANGES_FLAGS.len() {
        let (ini, flags) = UNICODE_RANGES_FLAGS[i - 1];
        let end = UNICODE_RANGES_FLAGS[i].0;
        for cpt in ini..end {
            cpt_flags[cpt as usize] = UnicodeCptFlags::new(flags);
        }
    }
    for &cpt in UNICODE_SET_WHITESPACE {
        cpt_flags[cpt as usize].set(UnicodeCptFlags::WHITESPACE, true);
    }
    for &(_, to) in UNICODE_MAP_LOWERCASE {
        cpt_flags[to as usize].set(UnicodeCptFlags::LOWERCASE, true);
    }
    for &(_, to) in UNICODE_MAP_UPPERCASE {
        cpt_flags[to as usize].set(UnicodeCptFlags::UPPERCASE, true);
    }
    for range in UNICODE_RANGES_NFD {
        cpt_flags[range.nfd as usize].set(UnicodeCptFlags::NFD, true);
    }
    cpt_flags
}

static CPT_FLAGS: OnceLock<Vec<UnicodeCptFlags>> = OnceLock::new();

pub fn unicode_cpt_flags_from_cpt(cpt: u32) -> UnicodeCptFlags {
    let undef = UnicodeCptFlags::new(UnicodeCptFlags::UNDEFINED);
    CPT_FLAGS
        .get_or_init(unicode_cpt_flags_array)
        .get(cpt as usize)
        .copied()
        .unwrap_or(undef)
}

pub fn unicode_cpt_flags_from_utf8(utf8: &[u8]) -> Result<UnicodeCptFlags> {
    if utf8.is_empty() {
        return Ok(UnicodeCptFlags::new(UnicodeCptFlags::UNDEFINED));
    }
    let mut offset = 0;
    let cpt = unicode_cpt_from_utf8(utf8, &mut offset)?;
    Ok(unicode_cpt_flags_from_cpt(cpt))
}

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

/// NFD normalization: replace each codepoint by the first range that covers it.
pub fn unicode_cpts_normalize_nfd(cpts: &[u32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(cpts.len());
    for &cpt in cpts {
        let idx = UNICODE_RANGES_NFD.partition_point(|range| range.first <= cpt);
        if idx > 0 {
            let range = &UNICODE_RANGES_NFD[idx - 1];
            if range.first <= cpt && cpt <= range.last {
                result.push(range.nfd);
                continue;
            }
        }
        result.push(cpt);
    }
    result
}

fn unicode_canonical_class(cpt: u32) -> u8 {
    UNICODE_COMBINING_CLASS
        .binary_search_by_key(&cpt, |&(key, _)| key)
        .map(|idx| UNICODE_COMBINING_CLASS[idx].1)
        .unwrap_or(0)
}

fn unicode_canonical_decompose(cpt: u32, result: &mut Vec<u32>, pending: &mut Vec<u32>) {
    pending.push(cpt);
    while let Some(cpt) = pending.pop() {
        // Hangul syllables use algorithmic decomposition (UAX #15).
        if (0xac00..0xac00 + 11172).contains(&cpt) {
            let index = cpt - 0xac00;
            result.push(0x1100 + index / 588);
            result.push(0x1161 + (index % 588) / 28);
            if index % 28 != 0 {
                result.push(0x11a7 + index % 28);
            }
            continue;
        }
        match UNICODE_DECOMPOSITION.binary_search_by_key(&cpt, |&(key, _)| key) {
            Err(_) => result.push(cpt),
            Ok(idx) => {
                let (first, second) = UNICODE_DECOMPOSITION[idx].1;
                if second != 0 {
                    pending.push(second);
                }
                pending.push(first);
            }
        }
    }
}

fn unicode_canonical_compose(first: u32, second: u32) -> u32 {
    if (0x1100..0x1100 + 19).contains(&first) && (0x1161..0x1161 + 21).contains(&second) {
        return 0xac00 + ((first - 0x1100) * 21 + second - 0x1161) * 28;
    }
    if (0xac00..0xac00 + 11172).contains(&first)
        && (first - 0xac00) % 28 == 0
        && (0x11a7 + 1..0x11a7 + 28).contains(&second)
    {
        return first + second - 0x11a7;
    }
    let key = (u64::from(first) << 32) | u64::from(second);
    UNICODE_COMPOSITION
        .binary_search_by_key(&key, |&(k, _)| k)
        .map(|idx| UNICODE_COMPOSITION[idx].1)
        .unwrap_or(0)
}

/// NFC normalization: canonical decomposition, mark reordering, recomposition.
pub fn unicode_cpts_normalize_nfc(cpts: &[u32]) -> Vec<u32> {
    let mut decomposed = Vec::with_capacity(cpts.len());
    let mut pending = Vec::new();
    for &cpt in cpts {
        unicode_canonical_decompose(cpt, &mut decomposed, &mut pending);
    }

    // Sort each run of combining marks; equal classes retain their order.
    let mut i = 0;
    while i < decomposed.len() {
        if unicode_canonical_class(decomposed[i]) == 0 {
            i += 1;
        }
        let begin = i;
        while i < decomposed.len() && unicode_canonical_class(decomposed[i]) != 0 {
            i += 1;
        }
        decomposed[begin..i].sort_by_key(|&cpt| unicode_canonical_class(cpt));
    }

    let mut result: Vec<u32> = Vec::with_capacity(decomposed.len());
    let mut starter = 0usize;
    let mut previous_class = 0u8;
    for cpt in decomposed {
        let current_class = unicode_canonical_class(cpt);
        let composite = if !result.is_empty()
            && (previous_class == 0 || previous_class < current_class)
        {
            unicode_canonical_compose(result[starter], cpt)
        } else {
            0
        };
        if composite != 0 {
            result[starter] = composite;
        } else {
            if current_class == 0 {
                starter = result.len();
            }
            result.push(cpt);
            previous_class = current_class;
        }
    }
    result
}

/// Decode all codepoints. Invalid UTF-8 is replaced by U+FFFD.
pub fn unicode_cpts_from_utf8(utf8: &[u8]) -> Vec<u32> {
    let mut result = Vec::with_capacity(utf8.len());
    let mut offset = 0;
    while offset < utf8.len() {
        match unicode_cpt_from_utf8(utf8, &mut offset) {
            Ok(cpt) => result.push(cpt),
            Err(_) => {
                offset += 1;
                result.push(0xfffd);
            }
        }
    }
    result
}

pub fn unicode_tolower(cpt: u32) -> u32 {
    UNICODE_MAP_LOWERCASE
        .binary_search_by_key(&cpt, |&(key, _)| key)
        .map(|idx| UNICODE_MAP_LOWERCASE[idx].1)
        .unwrap_or(cpt)
}

pub fn unicode_cpt_is_han(cpt: u32) -> bool {
    (0x4e00..=0x9fff).contains(&cpt)
        || (0x3400..=0x4dbf).contains(&cpt)
        || (0x20000..=0x2a6df).contains(&cpt)
        || (0x2a700..=0x2b73f).contains(&cpt)
        || (0x2b740..=0x2b81f).contains(&cpt)
        || (0x2b820..=0x2ceaf).contains(&cpt)
        || (0x2ceb0..=0x2ebef).contains(&cpt)
        || (0xf900..=0xfaff).contains(&cpt)
        || (0x2f800..=0x2fa1f).contains(&cpt)
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

const RAON_REGEX: &str = concat!(
    " ?[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+",
    "| ?(?:(?:(?![\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}])\\p{L})\\p{M}*)+(?: (?:(?:(?![\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}])\\p{L})\\p{M}*)+)*",
    "|\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))",
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
    "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
);

/// Select the hand-written splitter matching a known tokenizer regex.
fn unicode_regex_split_custom(regex_expr: &str) -> Option<CustomSplit> {
    if regex_expr == r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"# {
        return Some(unicode_regex_split_custom_gpt2);
    }
    if regex_expr == r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        || regex_expr == r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    {
        return Some(unicode_regex_split_custom_llama3);
    }
    if regex_expr == r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    {
        return Some(unicode_regex_split_custom_qwen2);
    }
    if regex_expr == r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    {
        return Some(unicode_regex_split_custom_qwen35);
    }
    if regex_expr == r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    {
        return Some(unicode_regex_split_custom_k2_horizon);
    }
    if regex_expr == r#"\p{Han}+"# {
        return Some(unicode_regex_split_custom_kimi_k2);
    }
    if regex_expr == r#"\p{AFMoE_digits}"# || regex_expr == r"\d{1,3}(?=(?:\d{3})*\b)" {
        return Some(unicode_regex_split_custom_afmoe);
    }
    if regex_expr == r"[^\n]+|[\n]+" {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_lengths_and_roundtrip() {
        assert_eq!(unicode_len_utf8(0x7f), 1);
        assert_eq!(unicode_len_utf8(0xc3), 2);
        assert_eq!(unicode_len_utf8(0xe4), 3);
        assert_eq!(unicode_len_utf8(0xf0), 4);
        for cpt in [0x41, 0xe9, 0x4e2d, 0x1f600] {
            let encoded = unicode_cpt_to_utf8(cpt).unwrap();
            let mut offset = 0;
            assert_eq!(unicode_cpt_from_utf8(encoded.as_bytes(), &mut offset).unwrap(), cpt);
            assert_eq!(offset, encoded.len());
            assert_eq!(unicode_cpts_from_utf8(encoded.as_bytes()), vec![cpt]);
        }
    }

    #[test]
    fn flags_case_and_han() {
        let a = unicode_cpt_flags_from_cpt(b'A' as u32);
        assert!(a.is_letter() && a.is_uppercase() && !a.is_whitespace());
        assert_eq!(unicode_tolower(b'A' as u32), b'a' as u32);
        assert_eq!(unicode_tolower(0x4e00), 0x4e00);
        assert!(unicode_cpt_is_han(0x4e00));
        assert!(!unicode_cpt_is_han(b'A' as u32));
    }
}

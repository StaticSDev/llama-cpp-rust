//! Rust port of `llama-vocab.cpp` and `llama-vocab.h`.
//!
//! The original uses GGUF/model-loader objects to read metadata. Here the same
//! data is supplied through [`Metadata`], so the port stays free of FFI and
//! external crates. Unicode tables from `unicode.cpp` are approximated with
//! `std` character classification; see the note on `unicode_cpt_flags_from_cpt`.

use crate::core::{Error, Result, Token, LLAMA_TOKEN_NULL};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

// ---------------------------------------------------------------------------
// types and constants from llama.h / llama-vocab.h
// ---------------------------------------------------------------------------

/// `llama_token_attr` is a bit mask in llama.cpp.
pub type LlamaTokenAttr = u32;

pub const LLAMA_TOKEN_ATTR_UNDEFINED: LlamaTokenAttr = 0;
pub const LLAMA_TOKEN_ATTR_NORMAL: LlamaTokenAttr = 1 << 0;
pub const LLAMA_TOKEN_ATTR_UNKNOWN: LlamaTokenAttr = 1 << 1;
pub const LLAMA_TOKEN_ATTR_CONTROL: LlamaTokenAttr = 1 << 2;
pub const LLAMA_TOKEN_ATTR_USER_DEFINED: LlamaTokenAttr = 1 << 3;
pub const LLAMA_TOKEN_ATTR_UNUSED: LlamaTokenAttr = 1 << 4;
pub const LLAMA_TOKEN_ATTR_BYTE: LlamaTokenAttr = 1 << 5;
pub const LLAMA_TOKEN_ATTR_LSTRIP: LlamaTokenAttr = 1 << 6;
pub const LLAMA_TOKEN_ATTR_RSTRIP: LlamaTokenAttr = 1 << 7;
pub const LLAMA_TOKEN_ATTR_RSTRIP_PREV: LlamaTokenAttr = 1 << 8;

pub const LLAMA_TOKEN_TYPE_UNDEFINED: i32 = 0;
pub const LLAMA_TOKEN_TYPE_NORMAL: i32 = 1;
pub const LLAMA_TOKEN_TYPE_UNKNOWN: i32 = 2;
pub const LLAMA_TOKEN_TYPE_CONTROL: i32 = 3;
pub const LLAMA_TOKEN_TYPE_USER_DEFINED: i32 = 4;
pub const LLAMA_TOKEN_TYPE_UNUSED: i32 = 5;
pub const LLAMA_TOKEN_TYPE_BYTE: i32 = 6;

// escaped space symbol - U+2581 (Lower One Eighth Block)
pub const ESCAPED_SPACE: &str = "\u{2581}";

// reserved suffix (U+E000) that keeps DNA k-mers distinct from identical
// base-vocab BPE tokens (e.g. CCCCCC) in token_to_id; erased from id_to_token
// text at load
const DNA_KMER_MARKER: &str = "\u{e000}";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum VocabType {
    None = 0,
    Spm = 1,
    Bpe = 2,
    Wpm = 3,
    Ugm = 4,
    Rwkv = 5,
    Plamo2 = 6,
    Plamo3 = 7,
    Test = 8,
}

// pre-tokenization types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum VocabPreType {
    Default = 0,
    Llama3 = 1,
    DeepseekLlm = 2,
    DeepseekCoder = 3,
    Falcon = 4,
    Mpt = 5,
    Starcoder = 6,
    Gpt2 = 7,
    Refact = 8,
    CommandR = 9,
    StableLm2 = 10,
    Qwen2 = 11,
    Olmo = 12,
    Dbrx = 13,
    Smaug = 14,
    Poro = 15,
    ChatGlm3 = 16,
    ChatGlm4 = 17,
    Viking = 18,
    Jais = 19,
    Tekken = 20,
    Smallm = 21,
    Codeshell = 22,
    Bloom = 23,
    Gpt3Finnish = 24,
    Exaone = 25,
    Chameleon = 26,
    Minerva = 27,
    Deepseek3Llm = 28,
    Gpt4o = 29,
    SuperBpe = 30,
    Trillion = 31,
    BailingMoe = 32,
    Llama4 = 33,
    Pixtral = 34,
    SeedCoder = 35,
    Hunyuan = 36,
    KimiK2 = 37,
    HunyuanDense = 38,
    Grok2 = 39,
    GraniteDocling = 40,
    MinimaxM2 = 41,
    Afmoe = 42,
    SolarOpen = 43,
    Youtu = 44,
    ExaoneMoe = 45,
    Qwen35 = 46,
    TinyAya = 47,
    JoyaiLlm = 48,
    Jais2 = 49,
    Gemma4 = 50,
    SarvamMoe = 51,
    Minicpm5 = 52,
    Whitespace = 53,
    GraniteEmbMulti = 54,
    Mellum2 = 55,
    Laguna = 56,
    HyV4 = 57,
    Spark2p5 = 58,
    Ufakzeka = 59,
    Mbert = 60,
    K2Horizon = 61,
    Flint = 62,
    DomynSmall = 63,
    Nemotron35Embd = 64,
    Soprano = 65,
    TekkenNum3 = 66,
    Zonos3 = 67,
    Raon = 68,
}

// HF Metaspace pre-tokenizer prepend_scheme
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaspacePrependScheme {
    Never = 0,
    First = 1,
    Always = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FragmentBufferVariantType {
    Token,
    RawText,
}

#[derive(Clone, Debug)]
pub struct TokenData {
    pub text: String,
    pub score: f32,
    pub attr: LlamaTokenAttr,
}

impl Default for TokenData {
    fn default() -> Self {
        Self {
            text: String::new(),
            score: 0.0,
            attr: LLAMA_TOKEN_ATTR_UNDEFINED,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NormalizerOptions {
    pub nfc: bool,
    pub lowercase: bool,
    pub strip_accents: bool,
}

impl Default for NormalizerOptions {
    fn default() -> Self {
        Self {
            nfc: false,
            lowercase: true,
            strip_accents: true,
        }
    }
}

// ---------------------------------------------------------------------------
// unicode helpers (unicode.h)
// ---------------------------------------------------------------------------

/// Flags for one unicode code point. Endian-independent bit mask version of
/// the bitfield struct in `unicode.h`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnicodeCptFlags(pub u16);

impl UnicodeCptFlags {
    pub const UNDEFINED: u16 = 0x0001;
    pub const NUMBER: u16 = 0x0002; // regex: \p{N}
    pub const LETTER: u16 = 0x0004; // regex: \p{L}
    pub const SEPARATOR: u16 = 0x0008; // regex: \p{Z}
    pub const ACCENT_MARK: u16 = 0x0010; // regex: \p{M}
    pub const PUNCTUATION: u16 = 0x0020; // regex: \p{P}
    pub const SYMBOL: u16 = 0x0040; // regex: \p{S}
    pub const CONTROL: u16 = 0x0080; // regex: \p{C}
    pub const MASK_CATEGORIES: u16 = 0x00FF;
    pub const WHITESPACE: u16 = 0x0100;
    pub const LOWERCASE: u16 = 0x0200;
    pub const UPPERCASE: u16 = 0x0400;
    pub const NFD: u16 = 0x0800;

    pub fn from_uint(flags: u16) -> Self {
        Self(flags)
    }

    pub fn as_uint(self) -> u16 {
        self.0
    }

    pub fn category_flag(self) -> u16 {
        self.as_uint() & Self::MASK_CATEGORIES
    }

    pub fn is_undefined(self) -> bool {
        self.0 & Self::UNDEFINED != 0
    }
    pub fn is_number(self) -> bool {
        self.0 & Self::NUMBER != 0
    }
    pub fn is_letter(self) -> bool {
        self.0 & Self::LETTER != 0
    }
    pub fn is_separator(self) -> bool {
        self.0 & Self::SEPARATOR != 0
    }
    pub fn is_accent_mark(self) -> bool {
        self.0 & Self::ACCENT_MARK != 0
    }
    pub fn is_punctuation(self) -> bool {
        self.0 & Self::PUNCTUATION != 0
    }
    pub fn is_symbol(self) -> bool {
        self.0 & Self::SYMBOL != 0
    }
    pub fn is_control(self) -> bool {
        self.0 & Self::CONTROL != 0
    }
    pub fn is_whitespace(self) -> bool {
        self.0 & Self::WHITESPACE != 0
    }
    pub fn is_lowercase(self) -> bool {
        self.0 & Self::LOWERCASE != 0
    }
    pub fn is_uppercase(self) -> bool {
        self.0 & Self::UPPERCASE != 0
    }
    pub fn is_nfd(self) -> bool {
        self.0 & Self::NFD != 0
    }
}

pub fn unicode_len_utf8(src: u8) -> usize {
    match src {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => 1,
    }
}

pub fn unicode_cpt_to_utf8(cpt: u32) -> String {
    char::from_u32(cpt).unwrap_or('\u{fffd}').to_string()
}

/// Decode one code point at `offset`, advancing the offset on success.
pub fn unicode_cpt_from_utf8(utf8: &[u8], offset: &mut usize) -> Result<u32> {
    if *offset >= utf8.len() {
        return Err(Error::new("invalid utf8 offset"));
    }
    let len = unicode_len_utf8(utf8[*offset]).min(utf8.len() - *offset);
    let slice = &utf8[*offset..*offset + len];
    let s = std::str::from_utf8(slice).map_err(|e| Error::new(e.to_string()))?;
    let cpt = s.chars().next().ok_or_else(|| Error::new("invalid utf8"))? as u32;
    *offset += len;
    Ok(cpt)
}

pub fn unicode_cpts_from_utf8(utf8: &str) -> Vec<u32> {
    utf8.chars().map(|c| c as u32).collect()
}

const fn gpt2_decoder_u32() -> [u32; 256] {
    let mut bs = [false; 256];
    let mut cs = [0u32; 256];
    let mut b = 33usize;
    while b <= 126 {
        bs[b] = true;
        cs[b] = b as u32;
        b += 1;
    }
    b = 161;
    while b <= 172 {
        bs[b] = true;
        cs[b] = b as u32;
        b += 1;
    }
    b = 174;
    while b <= 255 {
        bs[b] = true;
        cs[b] = b as u32;
        b += 1;
    }
    let mut n = 256u32;
    let mut i = 0usize;
    while i < 256 {
        if !bs[i] {
            cs[i] = n;
            n += 1;
        }
        i += 1;
    }
    cs
}

static GPT2_DECODER: [u32; 256] = gpt2_decoder_u32();

/// GPT-2 byte-to-unicode map used by `bytes_to_unicode`.
pub fn unicode_byte_to_utf8(byte: u8) -> String {
    unicode_cpt_to_utf8(GPT2_DECODER[byte as usize])
}

pub fn unicode_utf8_to_byte(utf8: &str) -> u8 {
    let cpt = utf8.chars().next().map(|c| c as u32).unwrap_or(0);
    unicode_cpt_to_byte(cpt).unwrap_or(0)
}

pub fn unicode_cpt_to_byte(cpt: u32) -> Result<u8> {
    for (i, &mapped) in GPT2_DECODER.iter().enumerate() {
        if mapped == cpt {
            return Ok(i as u8);
        }
    }
    Err(Error::new("code point is not a GPT-2 byte token"))
}

/// `unicode_tolower` maps one code point to one code point.
pub fn unicode_tolower(cpt: u32) -> u32 {
    match char::from_u32(cpt) {
        Some(c) => c.to_lowercase().next().unwrap_or(c) as u32,
        None => cpt,
    }
}

pub fn unicode_cpt_is_han(cpt: u32) -> bool {
    (0x04e00..=0x09fff).contains(&cpt)
        || (0x03400..=0x04dbf).contains(&cpt)
        || (0x20000..=0x2a6df).contains(&cpt)
        || (0x2a700..=0x2b73f).contains(&cpt)
        || (0x2b740..=0x2b81f).contains(&cpt)
        || (0x2b920..=0x2ceaf).contains(&cpt)
        || (0x0f900..=0x0faff).contains(&cpt)
        || (0x2f800..=0x2fa1f).contains(&cpt)
}

fn is_combining_mark(c: char) -> bool {
    let u = c as u32;
    (0x0300..=0x036f).contains(&u)
        || (0x0483..=0x0489).contains(&u)
        || (0x0591..=0x05bd).contains(&u)
        || (0x05bf..=0x05bf).contains(&u)
        || (0x05c1..=0x05c2).contains(&u)
        || (0x0610..=0x061a).contains(&u)
        || (0x064b..=0x065f).contains(&u)
        || (0x0670..=0x0670).contains(&u)
        || (0x06d6..=0x06dc).contains(&u)
        || (0x0e31..=0x0e3a).contains(&u)
        || (0x1ab0..=0x1aff).contains(&u)
        || (0x1dc0..=0x1dff).contains(&u)
        || (0x20d0..=0x20ff).contains(&u)
        || (0xfe20..=0xfe2f).contains(&u)
}

/// Category flags derived from `std`. The full unicode.cpp tables are not part
/// of the ported sources; these checks cover the properties the tokenizers use.
pub fn unicode_cpt_flags_from_cpt(cpt: u32) -> UnicodeCptFlags {
    let c = match char::from_u32(cpt) {
        Some(c) => c,
        None => return UnicodeCptFlags::from_uint(UnicodeCptFlags::UNDEFINED),
    };
    let mut flags = 0u16;
    if c.is_control() {
        flags |= UnicodeCptFlags::CONTROL;
    }
    if c.is_numeric() {
        flags |= UnicodeCptFlags::NUMBER;
    }
    if c.is_alphabetic() {
        flags |= UnicodeCptFlags::LETTER;
    }
    if c.is_whitespace() && !c.is_control() {
        flags |= UnicodeCptFlags::SEPARATOR;
    }
    if is_combining_mark(c) {
        flags |= UnicodeCptFlags::ACCENT_MARK | UnicodeCptFlags::NFD;
    }
    if c.is_ascii_punctuation() || unicode_cpt_is_han(c as u32) {
        // Han is categorized here only so WPM keeps single CJK glyphs apart.
        if c.is_ascii_punctuation() {
            flags |= UnicodeCptFlags::PUNCTUATION;
        }
    }
    if (c >= '\u{2000}' && c <= '\u{206f}') || (c >= '\u{3000}' && c <= '\u{303f}') {
        flags |= UnicodeCptFlags::PUNCTUATION;
    }
    if c.is_ascii() && !c.is_alphanumeric() && !c.is_ascii_whitespace() && !c.is_ascii_punctuation() {
        flags |= UnicodeCptFlags::SYMBOL;
    }
    if c.is_whitespace() {
        flags |= UnicodeCptFlags::WHITESPACE;
    }
    if c.is_lowercase() {
        flags |= UnicodeCptFlags::LOWERCASE;
    }
    if c.is_uppercase() {
        flags |= UnicodeCptFlags::UPPERCASE;
    }
    UnicodeCptFlags::from_uint(flags)
}

pub fn unicode_cpt_flags_from_utf8(utf8: &str) -> UnicodeCptFlags {
    let mut offset = 0;
    match unicode_cpt_from_utf8(utf8.as_bytes(), &mut offset) {
        Ok(cpt) => unicode_cpt_flags_from_cpt(cpt),
        Err(_) => UnicodeCptFlags::from_uint(UnicodeCptFlags::UNDEFINED),
    }
}

// Compact NFD table for the Latin ranges the convert scripts normalize.
const NFD_TABLE: &[(u32, &[u32])] = &[
    (0x00c0, &[0x0041, 0x0300]), (0x00c1, &[0x0041, 0x0301]),
    (0x00c2, &[0x0041, 0x0302]), (0x00c3, &[0x0041, 0x0303]),
    (0x00c4, &[0x0041, 0x0308]), (0x00c5, &[0x0041, 0x030a]),
    (0x00c7, &[0x0043, 0x0327]), (0x00c8, &[0x0045, 0x0300]),
    (0x00c9, &[0x0045, 0x0301]), (0x00ca, &[0x0045, 0x0302]),
    (0x00cb, &[0x0045, 0x0308]), (0x00cc, &[0x0049, 0x0300]),
    (0x00cd, &[0x0049, 0x0301]), (0x00ce, &[0x0049, 0x0302]),
    (0x00cf, &[0x0049, 0x0308]), (0x00d1, &[0x004e, 0x0303]),
    (0x00d2, &[0x004f, 0x0300]), (0x00d3, &[0x004f, 0x0301]),
    (0x00d4, &[0x004f, 0x0302]), (0x00d5, &[0x004f, 0x0303]),
    (0x00d6, &[0x004f, 0x0308]), (0x00d9, &[0x0055, 0x0300]),
    (0x00da, &[0x0055, 0x0301]), (0x00db, &[0x0055, 0x0302]),
    (0x00dc, &[0x0055, 0x0308]), (0x00dd, &[0x0059, 0x0301]),
    (0x00e0, &[0x0061, 0x0300]), (0x00e1, &[0x0061, 0x0301]),
    (0x00e2, &[0x0061, 0x0302]), (0x00e3, &[0x0061, 0x0303]),
    (0x00e4, &[0x0061, 0x0308]), (0x00e5, &[0x0061, 0x030a]),
    (0x00e7, &[0x0063, 0x0327]), (0x00e8, &[0x0065, 0x0300]),
    (0x00e9, &[0x0065, 0x0301]), (0x00ea, &[0x0065, 0x0302]),
    (0x00eb, &[0x0065, 0x0308]), (0x00ec, &[0x0069, 0x0300]),
    (0x00ed, &[0x0069, 0x0301]), (0x00ee, &[0x0069, 0x0302]),
    (0x00ef, &[0x0069, 0x0308]), (0x00f1, &[0x006e, 0x0303]),
    (0x00f2, &[0x006f, 0x0300]), (0x00f3, &[0x006f, 0x0301]),
    (0x00f4, &[0x006f, 0x0302]), (0x00f5, &[0x006f, 0x0303]),
    (0x00f6, &[0x006f, 0x0308]), (0x00f9, &[0x0075, 0x0300]),
    (0x00fa, &[0x0075, 0x0301]), (0x00fb, &[0x0075, 0x0302]),
    (0x00fc, &[0x0075, 0x0308]), (0x00fd, &[0x0079, 0x0301]),
    (0x00ff, &[0x0079, 0x0308]),
];

pub fn unicode_cpts_normalize_nfd(cpts: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(cpts.len());
    for &cpt in cpts {
        match NFD_TABLE.iter().find(|(key, _)| *key == cpt) {
            Some((_, decomp)) => out.extend_from_slice(decomp),
            None => out.push(cpt),
        }
    }
    out
}

pub fn unicode_cpts_normalize_nfc(cpts: &[u32]) -> Vec<u32> {
    let decomposed = unicode_cpts_normalize_nfd(cpts);
    let mut out: Vec<u32> = Vec::with_capacity(decomposed.len());
    for cpt in decomposed {
        if let Some(&last) = out.last() {
            if let Some((composed, _)) = NFD_TABLE
                .iter()
                .find(|(_, decomp)| decomp.len() == 2 && decomp[0] == last && decomp[1] == cpt)
            {
                out.pop();
                out.push(*composed);
                continue;
            }
        }
        out.push(cpt);
    }
    out
}

// ---------------------------------------------------------------------------
// pre-tokenizer regex subset
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CatKind {
    Letter,
    Upper,
    Lower,
    Number,
    Digit,
    Separator,
    Mark,
    Punctuation,
    Symbol,
    Control,
    Whitespace,
    Space,
    Word,
    Han,
    Undefined,
}

impl CatKind {
    fn from_name(name: &str) -> Self {
        match name {
            "L" | "Lt" | "Lm" | "Lo" => CatKind::Letter,
            "Lu" => CatKind::Upper,
            "Ll" => CatKind::Lower,
            "N" | "Nd" => CatKind::Number,
            "Z" => CatKind::Separator,
            "M" | "Mn" | "Mc" | "Me" => CatKind::Mark,
            "P" => CatKind::Punctuation,
            "S" => CatKind::Symbol,
            "C" => CatKind::Control,
            "Han" => CatKind::Han,
            "AFMoE_digits" => CatKind::Digit,
            _ => CatKind::Undefined,
        }
    }

    fn matches(self, c: char) -> bool {
        let f = unicode_cpt_flags_from_cpt(c as u32);
        match self {
            CatKind::Letter => f.is_letter(),
            CatKind::Upper => f.is_uppercase(),
            CatKind::Lower => f.is_lowercase(),
            CatKind::Number | CatKind::Digit => f.is_number(),
            CatKind::Separator => f.is_separator(),
            CatKind::Mark => f.is_accent_mark(),
            CatKind::Punctuation => f.is_punctuation(),
            CatKind::Symbol => f.is_symbol(),
            CatKind::Control => f.is_control(),
            CatKind::Whitespace | CatKind::Space => c.is_whitespace(),
            CatKind::Word => c.is_alphanumeric() || c == '_',
            CatKind::Han => unicode_cpt_is_han(c as u32),
            CatKind::Undefined => f.is_undefined(),
        }
    }
}

#[derive(Clone, Debug)]
enum ClassItem {
    Ch(char),
    Range(char, char),
    Cat(CatKind),
    NegCat(CatKind),
}

#[derive(Clone, Debug)]
enum Re {
    Empty,
    Lit(char),
    Class { neg: bool, items: Vec<ClassItem> },
    Seq(Vec<Re>),
    Alt(Vec<Re>),
    Repeat { node: Box<Re>, min: usize, max: Option<usize> },
    Group(Box<Re>),
    Look { positive: bool, node: Box<Re> },
    Boundary(bool),
}

struct ReParser {
    chars: Vec<char>,
    pos: usize,
}

impl ReParser {
    fn new(s: &str) -> Self {
        Self { chars: s.chars().collect(), pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn parse(mut self) -> Re {
        self.parse_alt()
    }

    fn parse_alt(&mut self) -> Re {
        let mut alts = vec![self.parse_seq()];
        while self.peek() == Some('|') {
            self.pos += 1;
            alts.push(self.parse_seq());
        }
        if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            Re::Alt(alts)
        }
    }

    fn parse_seq(&mut self) -> Re {
        let mut seq = Vec::new();
        while let Some(c) = self.peek() {
            if c == ')' || c == '|' {
                break;
            }
            let atom = self.parse_atom();
            seq.push(self.parse_quant(atom));
        }
        if seq.is_empty() {
            Re::Empty
        } else if seq.len() == 1 {
            seq.pop().unwrap()
        } else {
            Re::Seq(seq)
        }
    }

    fn parse_atom(&mut self) -> Re {
        let c = self.next().unwrap_or('\0');
        match c {
            '(' => {
                if self.peek() == Some('?') {
                    self.pos += 1;
                    match self.next() {
                        Some(':') | Some('i') => {
                            if self.chars.get(self.pos - 1) == Some(&'i') && self.peek() == Some(':') {
                                self.pos += 1;
                            }
                            let inner = self.parse_alt();
                            if self.peek() == Some(')') {
                                self.pos += 1;
                            }
                            Re::Group(Box::new(inner))
                        }
                        Some('=') => {
                            let inner = self.parse_alt();
                            if self.peek() == Some(')') {
                                self.pos += 1;
                            }
                            Re::Look { positive: true, node: Box::new(inner) }
                        }
                        Some('!') => {
                            let inner = self.parse_alt();
                            if self.peek() == Some(')') {
                                self.pos += 1;
                            }
                            Re::Look { positive: false, node: Box::new(inner) }
                        }
                        _ => {
                            let inner = self.parse_alt();
                            if self.peek() == Some(')') {
                                self.pos += 1;
                            }
                            Re::Group(Box::new(inner))
                        }
                    }
                } else {
                    let inner = self.parse_alt();
                    if self.peek() == Some(')') {
                        self.pos += 1;
                    }
                    Re::Group(Box::new(inner))
                }
            }
            '[' => self.parse_class(),
            '\\' => {
                if self.peek() == Some('b') {
                    self.pos += 1;
                    Re::Boundary(true)
                } else {
                    match self.escape_item() {
                        ClassItem::Ch(ch) => Re::Lit(ch),
                        item => Re::Class { neg: false, items: vec![item] },
                    }
                }
            }
            _ => Re::Lit(c),
        }
    }

    fn escape_item(&mut self) -> ClassItem {
        match self.next() {
            Some('n') => ClassItem::Ch('\n'),
            Some('r') => ClassItem::Ch('\r'),
            Some('t') => ClassItem::Ch('\t'),
            Some('d') => ClassItem::Cat(CatKind::Digit),
            Some('s') => ClassItem::Cat(CatKind::Whitespace),
            Some('w') => ClassItem::Cat(CatKind::Word),
            Some('p') | Some('P') => {
                let neg = self.chars.get(self.pos - 1) == Some(&'P');
                let mut name = String::new();
                if self.peek() == Some('{') {
                    self.pos += 1;
                    while let Some(c) = self.peek() {
                        if c == '}' {
                            self.pos += 1;
                            break;
                        }
                        name.push(c);
                        self.pos += 1;
                    }
                } else if let Some(c) = self.next() {
                    name.push(c);
                }
                let cat = CatKind::from_name(&name);
                if neg { ClassItem::NegCat(cat) } else { ClassItem::Cat(cat) }
            }
            Some('S') => ClassItem::NegCat(CatKind::Whitespace),
            Some('D') => ClassItem::NegCat(CatKind::Digit),
            Some('W') => ClassItem::NegCat(CatKind::Word),
            Some('u') => {
                let mut value = 0u32;
                for _ in 0..4 {
                    let d = self.next().and_then(|c| c.to_digit(16)).unwrap_or(0);
                    value = value * 16 + d;
                }
                ClassItem::Ch(char::from_u32(value).unwrap_or('\u{fffd}'))
            }
            Some(c) => ClassItem::Ch(c),
            None => ClassItem::Ch('\0'),
        }
    }

    fn parse_class(&mut self) -> Re {
        let mut neg = false;
        if self.peek() == Some('^') {
            self.pos += 1;
            neg = true;
        }
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == ']' && !items.is_empty() {
                self.pos += 1;
                break;
            }
            let first = if c == '\\' {
                self.pos += 1;
                self.escape_item()
            } else {
                self.pos += 1;
                ClassItem::Ch(c)
            };
            if self.peek() == Some('-')
                && self.chars.get(self.pos + 1).copied() != Some(']')
                && self.chars.get(self.pos + 1).is_some()
            {
                self.pos += 1;
                let second = if self.peek() == Some('\\') {
                    self.pos += 1;
                    self.escape_item()
                } else {
                    ClassItem::Ch(self.next().unwrap_or('\0'))
                };
                match (first, second) {
                    (ClassItem::Ch(a), ClassItem::Ch(b)) => {
                        items.push(ClassItem::Range(a, b));
                    }
                    (a, b) => {
                        items.push(a);
                        items.push(ClassItem::Ch('-'));
                        items.push(b);
                    }
                }
            } else {
                items.push(first);
            }
        }
        Re::Class { neg, items }
    }

    fn parse_quant(&mut self, atom: Re) -> Re {
        let (min, max) = match self.peek() {
            Some('?') => {
                self.pos += 1;
                (0, Some(1))
            }
            Some('*') => {
                self.pos += 1;
                (0, None)
            }
            Some('+') => {
                self.pos += 1;
                (1, None)
            }
            Some('{') => {
                let save = self.pos;
                self.pos += 1;
                let first = self.read_number();
                let (lo, hi) = match first {
                    Some(n) => match self.peek() {
                        Some('}') => {
                            self.pos += 1;
                            (n, Some(n))
                        }
                        Some(',') => {
                            self.pos += 1;
                            let hi = self.read_number();
                            if self.peek() == Some('}') {
                                self.pos += 1;
                            }
                            (n, hi)
                        }
                        _ => {
                            self.pos = save;
                            return atom;
                        }
                    },
                    None => {
                        self.pos = save;
                        return atom;
                    }
                };
                (lo, hi)
            }
            _ => return atom,
        };
        Re::Repeat { node: Box::new(atom), min, max }
    }

    fn read_number(&mut self) -> Option<usize> {
        let mut n = 0usize;
        let mut any = false;
        while let Some(c) = self.peek() {
            if let Some(d) = c.to_digit(10) {
                n = n * 10 + d as usize;
                any = true;
                self.pos += 1;
            } else {
                break;
            }
        }
        if any { Some(n) } else { None }
    }
}

fn class_item_matches(item: &ClassItem, c: char) -> bool {
    match item {
        ClassItem::Ch(ch) => *ch == c,
        ClassItem::Range(a, b) => *a <= c && c <= *b,
        ClassItem::Cat(cat) => cat.matches(c),
        ClassItem::NegCat(cat) => !cat.matches(c),
    }
}

fn match_re(re: &Re, s: &[char], pos: usize) -> Vec<usize> {
    match re {
        Re::Empty => vec![pos],
        Re::Lit(ch) => {
            if s.get(pos) == Some(ch) {
                vec![pos + 1]
            } else {
                Vec::new()
            }
        }
        Re::Class { neg, items } => {
            let c = match s.get(pos) {
                Some(c) => *c,
                None => return Vec::new(),
            };
            let matched = items.iter().any(|item| class_item_matches(item, c));
            if matched != *neg {
                vec![pos + 1]
            } else {
                Vec::new()
            }
        }
        Re::Seq(parts) => {
            let mut current = vec![pos];
            for part in parts {
                let mut next = Vec::new();
                for &p in &current {
                    for e in match_re(part, s, p) {
                        if !next.contains(&e) {
                            next.push(e);
                        }
                    }
                }
                current = next;
                if current.is_empty() {
                    break;
                }
            }
            current
        }
        Re::Alt(alts) => {
            let mut out = Vec::new();
            for alt in alts {
                for e in match_re(alt, s, pos) {
                    if !out.contains(&e) {
                        out.push(e);
                    }
                }
            }
            out
        }
        Re::Repeat { node, min, max } => {
            let mut out = Vec::new();
            let mut current = vec![pos];
            if *min == 0 {
                out.push(pos);
            }
            let mut count = 0usize;
            loop {
                if let Some(hi) = *max {
                    if count >= hi {
                        break;
                    }
                }
                if count >= 4096 {
                    break;
                }
                let mut next = Vec::new();
                for &p in &current {
                    for e in match_re(node, s, p) {
                        if e == p {
                            continue; // zero-width step: no progress
                        }
                        if !next.contains(&e) {
                            next.push(e);
                        }
                    }
                }
                if next.is_empty() {
                    break;
                }
                count += 1;
                if count >= *min {
                    for e in &next {
                        if !out.contains(e) {
                            out.push(*e);
                        }
                    }
                }
                current = next;
            }
            out
        }
        Re::Group(inner) => match_re(inner, s, pos),
        Re::Look { positive, node } => {
            let ok = !match_re(node, s, pos).is_empty();
            if ok == *positive { vec![pos] } else { Vec::new() }
        }
        Re::Boundary(want) => {
            let before = pos > 0 && s[pos - 1].is_alphanumeric() || pos > 0 && s[pos - 1] == '_';
            let after = pos < s.len() && (s[pos].is_alphanumeric() || s[pos] == '_');
            if (before != after) == *want { vec![pos] } else { Vec::new() }
        }
    }
}

fn push_regex_chunk(chunks: &mut Vec<String>, cpts: &[char], byte_encode: bool) {
    if cpts.is_empty() {
        return;
    }
    if byte_encode {
        let mut out = String::with_capacity(cpts.len());
        let bytes: Vec<u8> = cpts.iter().flat_map(|c| c.to_string().into_bytes()).collect();
        for b in bytes {
            out.push_str(&unicode_byte_to_utf8(b));
        }
        chunks.push(out);
    } else {
        chunks.push(cpts.iter().copied().collect());
    }
}

/// Split `text` by the pre-tokenizer regexes, in expression order.
pub fn unicode_regex_split(text: &str, regex_exprs: &[String], byte_encode: bool) -> Vec<String> {
    let compiled: Vec<Re> = regex_exprs.iter().map(|e| ReParser::new(e).parse()).collect();
    let cpts: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut pending: Vec<char> = Vec::new();
    let mut i = 0usize;
    while i < cpts.len() {
        let mut best_end = None;
        for re in &compiled {
            let ends = match_re(re, &cpts, i);
            if let Some(&end) = ends.iter().max() {
                if end > i {
                    best_end = Some(end);
                    break;
                }
            }
        }
        if let Some(end) = best_end {
            let pending_new = std::mem::take(&mut pending);
            push_regex_chunk(&mut chunks, &pending_new, byte_encode);
            push_regex_chunk(&mut chunks, &cpts[i..end], byte_encode);
            i = end;
        } else {
            pending.push(cpts[i]);
            i += 1;
        }
    }
    push_regex_chunk(&mut chunks, &pending, byte_encode);
    chunks
}

// ---------------------------------------------------------------------------
// general helpers
// ---------------------------------------------------------------------------

fn replace_all(text: &mut String, from: &str, to: &str) {
    if from.is_empty() {
        return;
    }
    *text = text.replace(from, to);
}

fn parse_hex_byte(s: &str) -> u8 {
    u8::from_str_radix(s, 16).unwrap_or(0)
}

// no recursion: a token from the GGUF can be long
#[derive(Clone, Debug, Default)]
struct NaiveTrie {
    children: BTreeMap<u8, NaiveTrie>,
    has_value: bool,
    value: Token,
}

impl NaiveTrie {
    fn insert(&mut self, key: &[u8], value: Token) {
        let mut node = self;
        for &byte in key {
            node = node.children.entry(byte).or_default();
        }
        node.has_value = true;
        node.value = value;
    }

    fn get_longest_prefix(&self, key: &[u8], mut offset: usize) -> usize {
        let mut node = self;
        while offset < key.len() {
            match node.children.get(&key[offset]) {
                Some(child) => {
                    node = child;
                    offset += 1;
                }
                None => break,
            }
        }
        offset
    }

    fn traverse(&self, c: u8) -> Option<&NaiveTrie> {
        self.children.get(&c)
    }
}

// ---------------------------------------------------------------------------
// tokenizer state and sessions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct LlmSymbol {
    prev: i32,
    next: i32,
    text: Vec<u8>,
    n: usize,
}

impl LlmSymbol {
    fn new(text: &[u8]) -> Self {
        Self { prev: -1, next: -1, text: text.to_vec(), n: text.len() }
    }

    fn merge_right(&mut self, right: &LlmSymbol) {
        self.text.extend_from_slice(&right.text[..right.n]);
        self.n = self.text.len();
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.text[..self.n]).unwrap_or("")
    }
}

struct SpmBigram {
    left: i32,
    right: i32,
    score: f32,
    size: usize,
}

impl PartialEq for SpmBigram {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left
    }
}
impl Eq for SpmBigram {}
impl Ord for SpmBigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // priority_queue top: lowest score, then lowest left index
        other.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}
impl PartialOrd for SpmBigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct BpeBigram {
    left: i32,
    right: i32,
    rank: i32,
    size: usize,
}

impl PartialEq for BpeBigram {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.left == other.left
    }
}
impl Eq for BpeBigram {}
impl Ord for BpeBigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // priority_queue top: lowest rank, then lowest left index
        other.rank.cmp(&self.rank).then_with(|| other.left.cmp(&self.left))
    }
}
impl PartialOrd for BpeBigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
struct BpeData {
    regex_exprs: Vec<String>,
    byte_encode: bool,
}

#[derive(Clone, Debug, Default)]
struct UgmData {
    prefix_replacements: Vec<u8>,
    xcda_array: Vec<u32>,
    token_matcher: NaiveTrie,
    user_defined_token_matcher: NaiveTrie,
    min_score: f32,
    max_score: f32,
    unknown_token_score: f32,
}

#[derive(Clone, Debug, Default)]
struct RwkvData {
    token_matcher: NaiveTrie,
}

#[derive(Clone, Debug)]
struct Plamo2Data {
    tokens: Vec<String>,
    bytes: Vec<Token>,
    to_suffix_id: HashMap<i64, i32>,
    table: Vec<[i32; 4]>,
    pre_segment: bool,
}

enum Tokenizer {
    Spm,
    Bpe(BpeData),
    Wpm,
    Ugm(UgmData),
    Rwkv(RwkvData),
    Plamo2(Plamo2Data),
    Test,
}

impl Tokenizer {
    fn bpe(&self) -> &BpeData {
        match self {
            Tokenizer::Bpe(data) => data,
            _ => unreachable!("BPE tokenizer data required"),
        }
    }
    fn ugm(&self) -> &UgmData {
        match self {
            Tokenizer::Ugm(data) => data,
            _ => unreachable!("UGM tokenizer data required"),
        }
    }
    fn rwkv(&self) -> &RwkvData {
        match self {
            Tokenizer::Rwkv(data) => data,
            _ => unreachable!("RWKV tokenizer data required"),
        }
    }
    fn plamo2(&self) -> &Plamo2Data {
        match self {
            Tokenizer::Plamo2(data) => data,
            _ => unreachable!("PLaMo2 tokenizer data required"),
        }
    }
}

// SPM tokenizer
// original implementation:
// https://github.com/ggml-org/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct SpmSession<'a> {
    vocab: &'a Impl,
    symbols: Vec<LlmSymbol>,
    work_queue: BinaryHeap<SpmBigram>,
    rev_merge: HashMap<Vec<u8>, (i32, i32)>,
}

impl<'a> SpmSession<'a> {
    fn new(vocab: &'a Impl) -> Self {
        Self {
            vocab,
            symbols: Vec::new(),
            work_queue: BinaryHeap::new(),
            rev_merge: HashMap::new(),
        }
    }

    fn tokenize(&mut self, text: &str, output: &mut Vec<Token>) {
        self.symbols.clear();
        self.work_queue.clear();
        self.rev_merge.clear();

        // split string into utf8 chars
        let bytes = text.as_bytes();
        let mut offs = 0usize;
        let mut index: i32 = 0;
        while offs < bytes.len() {
            let len = unicode_len_utf8(bytes[offs]);
            let n = len.min(bytes.len() - offs);
            let mut sym = LlmSymbol::new(&bytes[offs..offs + n]);
            offs += n;
            sym.prev = index - 1;
            sym.next = if offs == bytes.len() { -1 } else { index + 1 };
            self.symbols.push(sym);
            index += 1;
        }

        // seed the work queue with all possible 2-character tokens
        for i in 1..self.symbols.len() as i32 {
            self.try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can
        while let Some(bigram) = self.work_queue.pop() {
            let left = bigram.left as usize;
            let right = bigram.right as usize;
            let left_n = self.symbols[left].n;
            let right_n = self.symbols[right].n;

            // if one of the symbols already got merged, skip it
            if left_n == 0 || right_n == 0 || left_n + right_n != bigram.size {
                continue;
            }

            // merge the right sym into the left one
            let right_sym = self.symbols[right].clone();
            self.symbols[left].merge_right(&right_sym);
            self.symbols[right].n = 0;

            // remove the right sym from the chain
            let next = self.symbols[right].next;
            self.symbols[left].next = next;
            if next >= 0 {
                self.symbols[next as usize].prev = bigram.left;
            }

            // find more substitutions
            let prev = self.symbols[left].prev;
            self.try_add_bigram(prev, bigram.left);
            self.try_add_bigram(bigram.left, next);
        }

        let mut i = 0i32;
        while i != -1 {
            self.resegment(i as usize, output);
            i = self.symbols[i as usize].next;
        }
    }

    // no recursion: the merges of a long token can be deep
    fn resegment(&self, index: usize, output: &mut Vec<Token>) {
        let mut todo = vec![index];
        while let Some(idx) = todo.pop() {
            let cur = &self.symbols[idx];
            if cur.n == 0 {
                continue;
            }
            let text = cur.as_str();

            // Do we need to support is_unused?
            let token = self.vocab.text_to_token(text);
            if token != LLAMA_TOKEN_NULL {
                output.push(token);
                continue;
            }

            if let Some(&(first, second)) = self.rev_merge.get(cur.text.as_slice()) {
                todo.push(second as usize);
                todo.push(first as usize);
                continue;
            }

            // output any symbols that did not form tokens as bytes
            for &byte in &cur.text[..cur.n] {
                output.push(self.vocab.byte_to_token(byte));
            }
        }
    }

    fn try_add_bigram(&mut self, left: i32, right: i32) {
        if left == -1 || right == -1 {
            return;
        }
        let l = left as usize;
        let r = right as usize;
        let mut text = self.symbols[l].text[..self.symbols[l].n].to_vec();
        text.extend_from_slice(&self.symbols[r].text[..self.symbols[r].n]);

        let text_str = String::from_utf8_lossy(&text).into_owned();
        let token = self.vocab.text_to_token(&text_str);
        if token == LLAMA_TOKEN_NULL {
            return;
        }
        if token as u32 >= self.vocab.n_tokens() {
            return;
        }
        let score = self.vocab.token_get_score(token);
        self.work_queue.push(SpmBigram {
            left,
            right,
            score,
            size: text.len(),
        });

        // Do we need to support is_unused?
        self.rev_merge.insert(text, (left, right));
    }
}

// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!
fn bpe_regex_exprs(pre_type: VocabPreType) -> (Vec<String>, bool) {
    let llama3 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    let cjk = "[一-龥぀-ゟ゠-ヿ]+";
    let gpt2 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
    let punct = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    let tekken = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    let gpt4o = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    let newlines = "[^\\n]+|[\\n]+";
    let mut byte_encode = true;
    let exprs: Vec<String> = match pre_type {
        VocabPreType::Flint => vec![
            "\\p{N}{1,3}".into(),
            cjk.into(),
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Llama3 => vec![llama3.into()],
        VocabPreType::Jais2 => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}".into(),
        ],
        VocabPreType::Dbrx | VocabPreType::Smaug => vec![llama3.into()],
        VocabPreType::DeepseekLlm => vec![
            "[\r\n]".into(),
            "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+".into(),
            "\\s?[!-/:-~！-／：-～‘-‟　-。]+".into(),
            "\\s+$".into(),
            "[一-龥ࠀ-一가-퟿]+".into(),
            "\\p{N}+".into(),
        ],
        VocabPreType::Deepseek3Llm | VocabPreType::HunyuanDense | VocabPreType::JoyaiLlm | VocabPreType::HyV4 => vec![
            "\\p{N}{1,3}".into(), cjk.into(), punct.into(),
        ],
        VocabPreType::Spark2p5 => vec![
            "\\p{N}{1,3}".into(), cjk.into(),
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\\r\\n]|\\s+(?!\\S)|\\s+".into(),
            "\\p{N}".into(),
        ],
        VocabPreType::Youtu => vec![
            "[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+".into(),
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::DeepseekCoder => vec![
            "[\r\n]".into(), "\\s?\\p{L}+".into(), "\\s?\\p{P}+".into(),
            "[一-龥ࠀ-一가-퟿]+".into(), "\\p{N}".into(),
        ],
        VocabPreType::Falcon => vec![
            "[\\p{P}\\$\\+<=>\\^~\\|`]+".into(),
            gpt2.into(),
            "[0-9][0-9][0-9]".into(),
        ],
        VocabPreType::Starcoder | VocabPreType::Refact | VocabPreType::CommandR
        | VocabPreType::Smallm | VocabPreType::Codeshell | VocabPreType::Exaone
        | VocabPreType::Minerva | VocabPreType::Mellum2 => vec!["\\p{N}".into(), gpt2.into()],
        VocabPreType::Gpt2 | VocabPreType::Mpt | VocabPreType::Olmo | VocabPreType::Jais
        | VocabPreType::Trillion | VocabPreType::GraniteDocling => vec![gpt2.into()],
        VocabPreType::StableLm2 | VocabPreType::Qwen2 | VocabPreType::Hunyuan | VocabPreType::SolarOpen => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Raon => vec![
            " ?[一-龥぀-ゟ゠-ヿ]+| ?(?:(?:(?![一-龥぀-ゟ゠-ヿ])\\p{L})\\p{M}*)+(?: (?:(?:(?![一-龥぀-ゟ゠-ヿ])\\p{L})\\p{M}*)+)*|\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Qwen35 => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Poro | VocabPreType::Bloom | VocabPreType::Gpt3Finnish => vec![
            " ?[^(\\s|.,!?…。，、।۔،)]+".into(),
        ],
        VocabPreType::ChatGlm4 => vec![llama3.into()],
        VocabPreType::Viking => vec![" ?[^(\\s|.,!?…。，、।۔،)]+".into(), "\\p{N}".into()],
        VocabPreType::Tekken | VocabPreType::Nemotron35Embd => vec![tekken.into()],
        VocabPreType::TekkenNum3 => vec![
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*|\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))|\\p{N}{1,2}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Chameleon => vec![
            "<sentinel:[0-9]+>".into(),
            "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z".into(),
            "([\\t\\n]|    |  )".into(),
            "\\p{N}".into(),
            "[\\p{P}!-/:-@\\[-`{-~]".into(),
            gpt2.into(),
        ],
        VocabPreType::Gpt4o | VocabPreType::MinimaxM2 => vec![gpt4o.into()],
        VocabPreType::GraniteEmbMulti => vec![gpt4o.into()],
        VocabPreType::TinyAya => vec![
            "\\d{1,3}(?=(?:\\d{3})*\\b)".into(),
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::KimiK2 => vec!["\\p{Han}+".into()],
        VocabPreType::SuperBpe => vec!["\\p{N}+".into(), "(?=(\\d{3})+(?!\\d))".into()],
        VocabPreType::BailingMoe => vec![
            "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::SeedCoder => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Ufakzeka => vec![
            "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Grok2 => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Afmoe => vec![
            "\\p{AFMoE_digits}".into(),
            "[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+".into(),
            punct.into(),
        ],
        VocabPreType::Laguna => vec![newlines.into(), llama3.into()],
        VocabPreType::ExaoneMoe => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Gemma4 | VocabPreType::SarvamMoe | VocabPreType::DomynSmall | VocabPreType::Mbert => {
            // SPM-style BPE: spaces are replaced with U+2581 by the normalizer,
            // then BPE merges run on the whole text without word-level splitting.
            // We only need to split on newlines since merge lookup asserts no newlines in tokens.
            byte_encode = false;
            vec![newlines.into()]
        }
        VocabPreType::Minicpm5 => vec!["\\p{N}{1,3}".into(),
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into()],
        VocabPreType::K2Horizon => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Soprano => {
            byte_encode = false;
            vec!["\\p{N}".into(), "\\s+|[\\p{L}\\p{N}_]+|[^\\p{L}\\p{N}_\\s]+".into()]
        }
        VocabPreType::Zonos3 => vec![
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]{2,}[\\r\\n/]*| +(?!\\S)".into(),
        ],
        VocabPreType::Whitespace => {
            // whitespace pre-tokenizer (jinaai/jina-embeddings-v2-base-zh)
            byte_encode = false;
            vec!["\\S+".into()]
        }
        _ => vec![
            "[\\p{P}\\$\\+<=>\\^~\\|]+".into(),
            gpt2.into(),
            "\\p{N}+".into(),
            "[0-9][0-9][0-9]".into(),
        ],
    };
    (exprs, byte_encode)
}

struct BpeSession<'a> {
    vocab: &'a Impl,
    tokenizer: &'a BpeData,
    symbols: Vec<LlmSymbol>,
    symbols_final: Vec<LlmSymbol>,
    work_queue: BinaryHeap<BpeBigram>,
}

impl<'a> BpeSession<'a> {
    fn new(vocab: &'a Impl, tokenizer: &'a BpeData) -> Self {
        Self {
            vocab,
            tokenizer,
            symbols: Vec::new(),
            symbols_final: Vec::new(),
            work_queue: BinaryHeap::new(),
        }
    }

    fn append(token_id: Token, output: &mut Vec<Token>) {
        output.push(token_id);
    }

    fn append_bos(&self, output: &mut Vec<Token>) -> bool {
        if self.vocab.get_add_bos() {
            output.push(self.vocab.token_bos());
            return true;
        }
        false
    }

    fn append_eos(&self, output: &mut Vec<Token>) -> bool {
        if self.vocab.get_add_eos() {
            output.push(self.vocab.token_eos());
            return true;
        }
        false
    }

    fn check_double_bos_eos(&self, output: &[Token]) {
        // The warnings are omitted; the checks are kept in the same order.
        let _ = self.vocab.get_add_bos() && output.get(1) == Some(&self.vocab.token_bos());
        let _ = self.vocab.get_add_eos()
            && output.len() >= 2
            && output.get(output.len() - 2) == Some(&self.vocab.token_eos());
    }

    fn tokenize(&mut self, text: &str, output: &mut Vec<Token>) {
        let tok_pre = self.vocab.get_pre_type();

        let word_collection: Vec<String> = if tok_pre == VocabPreType::Nemotron35Embd {
            let zwnj = "\u{200c}";
            let mut normalized_text = text.to_string();
            replace_all(&mut normalized_text, &format!("[{}", zwnj), "[");
            replace_all(&mut normalized_text, &format!("<{}", zwnj), "<");
            unicode_regex_split(&normalized_text, &self.tokenizer.regex_exprs, self.tokenizer.byte_encode)
        } else if tok_pre == VocabPreType::Soprano {
            let mut normalized_text = String::new();
            let mut space = false;
            for cpt in unicode_cpts_from_utf8(text) {
                if unicode_cpt_flags_from_cpt(cpt).is_whitespace() {
                    if !space {
                        normalized_text.push(' ');
                    }
                    space = true;
                } else {
                    // Unicode full lowercase expands LATIN CAPITAL LETTER I WITH DOT ABOVE.
                    if cpt == 0x0130 {
                        normalized_text.push_str("i\u{0307}");
                    } else {
                        normalized_text.push_str(&unicode_cpt_to_utf8(unicode_tolower(cpt)));
                    }
                    space = false;
                }
            }
            unicode_regex_split(&normalized_text, &self.tokenizer.regex_exprs, self.tokenizer.byte_encode)
        } else if tok_pre == VocabPreType::Raon {
            // Raon-1 mini's HF tokenizer applies an NFC normalizer before pre-tokenization.
            let mut normalized_text = String::new();
            for cpt in unicode_cpts_normalize_nfc(&unicode_cpts_from_utf8(text)) {
                normalized_text.push_str(&unicode_cpt_to_utf8(cpt));
            }
            unicode_regex_split(&normalized_text, &self.tokenizer.regex_exprs, self.tokenizer.byte_encode)
        } else {
            unicode_regex_split(text, &self.tokenizer.regex_exprs, self.tokenizer.byte_encode)
        };

        let mut final_prev_index: i32 = -1;
        self.symbols_final.clear();

        let mut word_collection = word_collection;
        if self.vocab.get_metaspace_prepend_scheme() == MetaspacePrependScheme::Always {
            // Metaspace pre-tokenizer (prepend_scheme = "always"): each non-newline chunk
            // starts with an escaped space, and each escaped space starts a new word
            let mut words = Vec::new();
            for word in word_collection {
                if !word.chars().any(|c| c != '\n') {
                    words.push(word);
                    continue;
                }
                let mut word = word;
                if !word.starts_with(ESCAPED_SPACE) {
                    word.insert_str(0, ESCAPED_SPACE);
                }
                let mut start = 0usize;
                while start < word.len() {
                    let search_from = start + ESCAPED_SPACE.len();
                    let end = if search_from <= word.len() {
                        match word[search_from..].find(ESCAPED_SPACE) {
                            Some(rel) => search_from + rel,
                            None => word.len(),
                        }
                    } else {
                        word.len()
                    };
                    words.push(word[start..end].to_string());
                    start = end;
                }
            }
            word_collection = words;
        }

        for word in &word_collection {
            self.work_queue.clear();
            self.symbols.clear();

            let mut index: i32 = 0;
            let mut offset = 0usize;
            let bytes = word.as_bytes();

            //if (vocab.tokenizer_ignore_merges && vocab.token_to_id.find(word) != vocab.token_to_id.end()) {
            if self.vocab.get_ignore_merges() && self.vocab.text_to_token(word) != LLAMA_TOKEN_NULL {
                self.symbols.push(LlmSymbol::new(bytes));
                offset = word.len();
            } else if (tok_pre == VocabPreType::Gemma4 || tok_pre == VocabPreType::Mbert)
                && !word.chars().any(|c| c != '\n')
            {
                // fix for gemma 4, ref: https://github.com/ggml-org/llama.cpp/pull/21343
                if self.vocab.text_to_token(word) != LLAMA_TOKEN_NULL {
                    self.symbols.push(LlmSymbol::new(bytes));
                    offset = word.len();
                }
            }

            while offset < bytes.len() {
                let char_len = (bytes.len() - offset).min(unicode_len_utf8(bytes[offset]));
                let mut sym = LlmSymbol::new(&bytes[offset..offset + char_len]);
                offset += char_len;
                sym.prev = index - 1;
                sym.next = if offset == bytes.len() { -1 } else { index + 1 };
                self.symbols.push(sym);
                index += 1;
            }
            for i in 1..self.symbols.len() as i32 {
                self.add_new_bigram(i - 1, i);
            }

            // build token(s)
            while let Some(bigram) = self.work_queue.pop() {
                let l = bigram.left as usize;
                let r = bigram.right as usize;
                if self.symbols[l].n == 0
                    || self.symbols[r].n == 0
                    || self.symbols[l].n + self.symbols[r].n != bigram.size
                {
                    continue;
                }

                // merge the right sym into the left one
                let right_sym = self.symbols[r].clone();
                self.symbols[l].merge_right(&right_sym);
                self.symbols[r].n = 0;

                // remove the right sym from the chain
                let next = self.symbols[r].next;
                self.symbols[l].next = next;
                if next >= 0 {
                    self.symbols[next as usize].prev = bigram.left;
                }

                let prev = self.symbols[l].prev;
                self.add_new_bigram(prev, bigram.left); // left side of current symbol
                self.add_new_bigram(bigram.left, next); // right side of current symbol
            }

            // add the finished tokens to the final list keeping correct order for next and prev
            for i in 0..self.symbols.len() {
                if self.symbols[i].n > 0 {
                    self.symbols[i].prev = final_prev_index;
                    self.symbols[i].next = -1;
                    if final_prev_index != -1 {
                        self.symbols_final[final_prev_index as usize].next = self.symbols_final.len() as i32;
                    }
                    self.symbols_final.push(self.symbols[i].clone());
                    final_prev_index = self.symbols_final.len() as i32 - 1;
                }
            }
        }

        self.symbols = self.symbols_final.clone();

        if !self.symbols.is_empty() {
            let mut i = 0i32;
            while i != -1 {
                let symbol = &self.symbols[i as usize];
                if symbol.n == 0 {
                    i = symbol.next;
                    continue;
                }

                let str = symbol.as_str();
                let token = self.vocab.text_to_token(str);

                if token == LLAMA_TOKEN_NULL && tok_pre == VocabPreType::Soprano {
                    output.push(self.vocab.token_unk());
                } else if token == LLAMA_TOKEN_NULL {
                    for &byte in str.as_bytes() {
                        let token_multibyte = if self.tokenizer.byte_encode {
                            let byte_str = String::from_utf8_lossy(&[byte]).into_owned();
                            self.vocab.text_to_token(&byte_str)
                        } else {
                            // For non-byte-encoded BPE (e.g. gemma-4), byte tokens use <0xXX> format
                            const HEX: &[u8; 16] = b"0123456789ABCDEF";
                            let buf = [
                                b'<', b'0', b'x', HEX[(byte >> 4) as usize],
                                HEX[(byte & 15) as usize], b'>',
                            ];
                            self.vocab.text_to_token(std::str::from_utf8(&buf).unwrap_or(""))
                        };
                        if token_multibyte != LLAMA_TOKEN_NULL {
                            output.push(token_multibyte);
                        }
                    }
                } else {
                    output.push(token);
                }
                i = self.symbols[i as usize].next;
            }
        }
    }

    fn add_new_bigram(&mut self, left: i32, right: i32) {
        if left == -1 || right == -1 {
            return;
        }
        let l = left as usize;
        let r = right as usize;
        let text_l = &self.symbols[l].text[..self.symbols[l].n];
        let text_r = &self.symbols[r].text[..self.symbols[r].n];
        let mut pair = text_l.to_vec();
        pair.extend_from_slice(text_r);
        let a = String::from_utf8_lossy(text_l);
        let b = String::from_utf8_lossy(text_r);
        let rank_found = self.vocab.find_bpe_rank(&a, &b);

        if rank_found < 0 {
            return;
        }
        self.work_queue.push(BpeBigram {
            left,
            right,
            size: self.symbols[l].n + self.symbols[r].n,
            rank: rank_found,
        });
    }
}

// WPM tokenizer
struct WpmSession<'a> {
    vocab: &'a Impl,
}

impl<'a> WpmSession<'a> {
    fn tokenize(&self, text: &str, output: &mut Vec<Token>) {
        // normalize and split by whitespace
        let words = WpmSession::preprocess(text, self.vocab.get_normalizer_opts());
        // bos token prepended already

        // find the longest tokens that form the words
        for word in words {
            // skip empty words
            if word.is_empty() {
                continue;
            }

            // prepend phantom space
            let word1 = format!("{}{}", ESCAPED_SPACE, word);
            let bytes = word1.as_bytes();
            let n = bytes.len();
            let current_tokens = output.len();

            // we're at the start of a new word
            // move through character position in word
            let mut i = 0usize;
            while i < n {
                // loop through possible match length
                let mut matched = false;
                let mut j = n.min(i + self.vocab.max_token_len() as usize + 1);
                while j > i {
                    let sub = std::str::from_utf8(&bytes[i..j]).unwrap_or("");
                    if self.vocab.text_to_token(sub) != LLAMA_TOKEN_NULL {
                        output.push(self.vocab.text_to_token(sub));
                        matched = true;
                        i = j - 1;
                        break;
                    }
                    j -= 1;
                }
                if !matched {
                    // discard all
                    output.truncate(current_tokens);
                    break; // and discard next tokens
                }
                i += 1;
            }

            // we didn't find any matches for this word
            if current_tokens == output.len() {
                output.push(self.vocab.token_unk());
            }
        }
    }

    // TODO: reduce string copies by using cpts_offs array
    fn preprocess(text: &str, normalizer_opts: NormalizerOptions) -> Vec<String> {
        let cpts = unicode_cpts_from_utf8(text);
        let cpts = if normalizer_opts.strip_accents {
            unicode_cpts_normalize_nfd(&cpts)
        } else {
            cpts
        };
        let mut words = vec![String::new()];

        for cpt in cpts {
            let flags = unicode_cpt_flags_from_cpt(cpt);

            if flags.is_whitespace() {
                if !words.last().map_or(false, |w| !w.is_empty()) {
                    continue;
                }
                words.push(String::new());
                continue;
            }

            assert!(!flags.is_separator());
            if cpt == 0 || cpt == 0xfffd || flags.is_control() {
                continue;
            }

            if normalizer_opts.strip_accents && flags.is_accent_mark() {
                continue;
            }

            let s = unicode_cpt_to_utf8(if normalizer_opts.lowercase {
                unicode_tolower(cpt)
            } else {
                cpt
            });
            if flags.is_punctuation() || (cpt < 0x7f && flags.is_symbol()) || WpmSession::is_chinese_char(cpt) {
                if !words.last().map_or(false, |w| !w.is_empty()) {
                    words.push(String::new());
                }
                *words.last_mut().unwrap() = s; // single char word
                words.push(String::new()); // start a new word
            } else {
                words.last_mut().unwrap().push_str(&s); // append char to word
            }
        }

        if words.last().map_or(false, |w| w.is_empty()) {
            words.pop();
        }
        words
    }

    fn is_chinese_char(cpt: u32) -> bool {
        (0x04e00..=0x09fff).contains(&cpt)
            || (0x03400..=0x04dbf).contains(&cpt)
            || (0x20000..=0x2a6df).contains(&cpt)
            || (0x2a700..=0x2b73f).contains(&cpt)
            || (0x2b740..=0x2b81f).contains(&cpt)
            || (0x2b920..=0x2ceaf).contains(&cpt) // this should be 0x2B820 but in hf rust code it is 0x2B920
            || (0x0f900..=0x0faff).contains(&cpt)
            || (0x2f800..=0x2fa1f).contains(&cpt)
    }
}

// UGM tokenizer
/*
 * This structure is a view wrapper for XOR-compressed double array (XCDA)
 * See Shunsuke Kanda (2018). Space- and Time-Efficient String Dictionaries.
 * Each bit-packed entry contains:
 * - BASE array value in bits 10-30
 * - LCHECK array value in bits 0-7
 * - LEAF array value in bit 9
 * Entries containing indexes of replacement sequences have set bit 31
 */
struct XcdaView<'b> {
    array: &'b [u32],
}

impl<'b> XcdaView<'b> {
    fn get_base(&self, index: usize) -> Result<u32> {
        let node = self.get_node(index)?;
        Ok((node >> 10) << ((node & (1u32 << 9)) >> 6))
    }
    fn get_lcheck(&self, index: usize) -> Result<u32> {
        let node = self.get_node(index)?;
        Ok(node & ((1u32 << 31) | 0xff))
    }
    fn get_leaf(&self, index: usize) -> Result<bool> {
        let node = self.get_node(index)?;
        Ok((node >> 8) & 1 != 0)
    }
    fn get_value(&self, index: usize) -> Result<u32> {
        let node = self.get_node(index)?;
        Ok(node & ((1u32 << 31) - 1))
    }
    fn get_node(&self, index: usize) -> Result<u32> {
        self.array.get(index).copied().ok_or_else(|| Error::new("Index out of array bounds in XCDA array!"))
    }
}

struct UgmSession<'a> {
    vocab: &'a Impl,
    tokenizer: &'a UgmData,
}

struct NormalizationResult {
    normalized: Vec<u8>,
    consumed_input: usize,
}

#[derive(Clone, Copy)]
struct BestTokenization {
    token_id: Token,
    input_offset: usize,
    score_sum: f64,
}

impl<'a> UgmSession<'a> {
    /* This implementation is based on SentencePiece optimized Viterbi algorithm for
     * unigram language models. The general idea is to:
     * - move along the input sequence in steps of one UTF code point,
     * - at each step find all possible tokenizations of the prefix by
     *   traversing the tokens trie,
     * - for each tokenization store the best one so far (by higher score)
     * - use the position in sequence after given token as an index to store
     *   results
     * - if there was no valid tokenization of the current UTF code point
     *   then use unknown token with additional score penalty
     * After processing the whole sequence we backtrack from the end to get
     * the best tokenization.
     */
    fn tokenize(&self, text: &str, output: &mut Vec<Token>) {
        let output_size = output.len();

        // normalize the input first
        let normalized = if self.vocab.get_normalizer_opts().nfc {
            let mut canonical = String::new();
            for cpt in unicode_cpts_normalize_nfc(&unicode_cpts_from_utf8(text)) {
                canonical.push_str(&unicode_cpt_to_utf8(cpt));
            }
            match self.normalize(&canonical) {
                Ok(s) => s,
                Err(_) => self.normalize(text).unwrap_or_default(),
            }
        } else {
            self.normalize(text).unwrap_or_default()
        };

        let input = normalized.as_slice();
        let input_len = input.len();
        if input_len == 0 {
            return;
        }

        // initialize score_sum to -DBL_MAX so it will be always lower than sums of token scores
        let mut results: Vec<BestTokenization> = vec![BestTokenization {
            token_id: self.vocab.token_unk(),
            input_offset: 0,
            score_sum: f64::MIN,
        }; input_len + 1];
        // at the beginning tokenization score is zero
        results[0] = BestTokenization {
            token_id: self.vocab.token_unk(),
            input_offset: 0,
            score_sum: 0.0,
        };

        let mut input_offset = 0usize;
        while input_offset < input_len {
            let mut prefix_offset = input_offset + 1;
            // calculate how many code units are in the currently processed UTF code point
            let mut probe = input_offset;
            let _ = unicode_cpt_from_utf8(input, &mut probe);
            let mut n_utf8_code_units = unicode_len_utf8(input[input_offset]).min(input_len - input_offset);
            if probe > input_offset {
                n_utf8_code_units = probe - input_offset;
            }

            // traverse the token matcher trie to find a matching token
            let mut single_codepoint_token_found = false;
            let current_best = results[input_offset];
            let mut node = self.tokenizer.token_matcher.traverse(input[input_offset]);

            while prefix_offset <= input_len && node.is_some() {
                let n = node.unwrap();
                // check if we found valid token in prefix
                if n.has_value {
                    // check if it corresponds to the whole UTF code point
                    if prefix_offset - input_offset == n_utf8_code_units {
                        single_codepoint_token_found = true;
                    }
                    let token_id = n.value;
                    let token_data = &self.vocab.get_token_data(token_id).score;
                    // we set the user-defined token scores to 0 to make them more likely to be selected
                    // (normal token scores are log probabilities, so they are negative)
                    // score type is double here to make tokenization results exactly
                    // the same as in the HF tokenizer using SentencePiece
                    let token_score = if self.vocab.is_user_defined(token_id) { 0.0 } else { *token_data as f64 };
                    let challenger_score = current_best.score_sum + token_score;
                    if challenger_score > results[prefix_offset].score_sum {
                        results[prefix_offset] = BestTokenization {
                            token_id,
                            input_offset,
                            score_sum: challenger_score,
                        };
                    }
                }
                if prefix_offset >= input_len {
                    break;
                }
                node = n.traverse(input[prefix_offset]);
                prefix_offset += 1;
            }

            // if we didn't find a valid token corresponding to the whole UTF code point
            // then use unknown token as the tokenization of this UTF code point
            if !single_codepoint_token_found {
                let challenger_score = current_best.score_sum + self.tokenizer.unknown_token_score as f64;
                let end = (input_offset + n_utf8_code_units).min(input_len);
                if challenger_score > results[end].score_sum {
                    results[end] = BestTokenization {
                        token_id: self.vocab.token_unk(),
                        input_offset,
                        score_sum: challenger_score,
                    };
                }
            }

            // move to the next UTF code point
            input_offset += n_utf8_code_units;
        }

        // now backtrack from the end to gather token ids of the best tokenization
        // merge sequences of consecutive unknown tokens into single unknown tokens
        let mut is_prev_unknown = false;
        let mut idx = input_len;
        loop {
            let tok = results[idx];
            let is_unknown = tok.token_id == self.vocab.token_unk();
            if !(is_prev_unknown && is_unknown) {
                output.push(tok.token_id);
            }
            if tok.input_offset == 0 {
                break;
            }
            is_prev_unknown = is_unknown;
            idx = tok.input_offset.min(input_len);
        }

        // reverse the output since we added tokens starting from the end of the input
        output[output_size..].reverse();
    }

    fn normalize(&self, input: &str) -> Result<Vec<u8>> {
        let mut normalized: Vec<u8> = Vec::with_capacity(input.len() * 3);
        let space = if self.vocab.get_escape_whitespaces() {
            ESCAPED_SPACE.as_bytes().to_vec()
        } else {
            b" ".to_vec()
        };

        let shall_prepend_space = !self.vocab.get_treat_whitespace_as_suffix() && self.vocab.get_add_space_prefix();
        let shall_append_space = self.vocab.get_treat_whitespace_as_suffix() && self.vocab.get_add_space_prefix();
        let shall_merge_spaces = self.vocab.get_remove_extra_whitespaces();

        let mut is_space_prepended = false;
        let mut processing_non_ws = false;

        let mut input_offset = 0usize;
        let bytes = input.as_bytes();
        while input_offset < bytes.len() {
            let norm_res = self.normalize_prefix(input, input_offset)?;
            for c in &norm_res.normalized {
                if *c != b' ' {
                    if !processing_non_ws {
                        processing_non_ws = true;
                        if (shall_prepend_space && !is_space_prepended) || shall_merge_spaces {
                            normalized.extend_from_slice(&space);
                            is_space_prepended = true;
                        }
                    }
                    normalized.push(*c);
                } else {
                    if processing_non_ws {
                        processing_non_ws = false;
                    }
                    if !shall_merge_spaces {
                        normalized.extend_from_slice(&space);
                    }
                }
            }
            input_offset += norm_res.consumed_input;
        }

        if shall_append_space {
            normalized.extend_from_slice(&space);
        }
        Ok(normalized)
    }

    fn normalize_prefix(&self, input: &str, input_offset: usize) -> Result<NormalizationResult> {
        let bytes = input.as_bytes();
        if input_offset == bytes.len() {
            return Ok(NormalizationResult { normalized: Vec::new(), consumed_input: 0 });
        }

        // if input prefix matches some user-defined token return this token as normalization result
        let user_len = self.tokenizer.user_defined_token_matcher.get_longest_prefix(&bytes[input_offset..], 0);
        if user_len > 0 {
            return Ok(NormalizationResult {
                normalized: bytes[input_offset..input_offset + user_len].to_vec(),
                consumed_input: user_len,
            });
        }

        let mut longest_prefix_length = 0usize;
        let mut longest_prefix_offset = 0usize;

        if !self.tokenizer.xcda_array.is_empty() {
            let view = XcdaView { array: &self.tokenizer.xcda_array };
            // Find the longest normalized sequence matching the input prefix by walking
            // the XOR-compressed compact double array (XCDA) starting from the root node
            // We find the index of the next node by calculating BASE[s] ^ c where s is
            // the index of the previous node and c is a numerical character value
            let mut node_index = view.get_base(0)? as usize;
            for prefix_offset in input_offset..bytes.len() {
                let c = bytes[prefix_offset];
                if c == 0 {
                    break;
                }
                node_index ^= c as usize;
                // if value of LCHECK is not c it means that this is not a child of
                // the previous node, so we stop matching
                if view.get_lcheck(node_index)? != c as u32 {
                    break;
                }
                let is_leaf = view.get_leaf(node_index)?;
                // get BASE of the current node
                node_index ^= view.get_base(node_index)? as usize;
                // if LEAF of the current node is true, it means that its BASE points to the node
                // containing index of replacement sequence for currently matched input prefix
                if is_leaf {
                    longest_prefix_length = prefix_offset - input_offset + 1;
                    // get index of replacement sequence for currently matched input prefix
                    longest_prefix_offset = view.get_value(node_index)? as usize;
                }
            }
        }

        if longest_prefix_length > 0 {
            // we have a match, so return the replacement sequence
            if longest_prefix_offset >= self.tokenizer.prefix_replacements.len() {
                return Err(Error::new("Index out of array bounds in precompiled charsmap!"));
            }
            let replacements = &self.tokenizer.prefix_replacements[longest_prefix_offset..];
            let repl_len = replacements.iter().position(|&b| b == 0).unwrap_or(replacements.len());
            if repl_len == replacements.len() {
                return Err(Error::new("Unterminated string in precompiled charsmap!"));
            }
            return Ok(NormalizationResult {
                normalized: replacements[..repl_len].to_vec(),
                consumed_input: longest_prefix_length,
            });
        }

        // check if the input prefix contains a valid sequence of UTF-8 code units
        let mut prefix_offset = input_offset;
        match unicode_cpt_from_utf8(bytes, &mut prefix_offset) {
            Ok(_) => Ok(NormalizationResult {
                normalized: bytes[input_offset..prefix_offset].to_vec(),
                consumed_input: prefix_offset - input_offset,
            }),
            // if no, consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER
            Err(_) => Ok(NormalizationResult {
                normalized: b"\xef\xbf\xbd".to_vec(),
                consumed_input: 1,
            }),
        }
    }
}

// RWKV tokenizer
pub fn llama_unescape_rwkv_token(escaped: &str) -> Vec<u8> {
    let mut output = Vec::with_capacity(escaped.len());
    // Parser state
    let mut escaping = false;
    let mut hex_remaining: u8 = 0;
    let mut hex_acc: u8 = 0;

    // Step through characters, performing parsing
    for c in escaped.bytes() {
        // If we're parsing a hex code, interpret the next character
        if hex_remaining != 0 {
            let value = if c >= b'a' { c - b'a' + 10 } else { c - b'0' };
            hex_acc = (hex_acc << 4) + value;
            hex_remaining -= 1;
            if hex_remaining == 0 {
                output.push(hex_acc);
                hex_acc = 0;
            }
            continue;
        }

        // If we got an escape character, interpret it
        if escaping {
            match c {
                b't' => output.push(b'\t'),
                b'n' => output.push(b'\n'),
                b'r' => output.push(b'\r'),
                b'x' => hex_remaining = 2,
                _ => output.push(c),
            }
            escaping = false;
            continue;
        }

        if c == b'\\' {
            escaping = true;
            continue;
        }

        output.push(c);
    }
    output
}

struct RwkvSession<'a> {
    vocab: &'a Impl,
    tokenizer: &'a RwkvData,
}

impl<'a> RwkvSession<'a> {
    fn tokenize(&self, text: &str, output: &mut Vec<Token>) {
        let bytes = text.as_bytes();
        let mut position = 0usize;
        while position < bytes.len() {
            let mut node = self.tokenizer.token_matcher.traverse(bytes[position]);
            if node.is_none() {
                // no matching token found, add unknown token
                output.push(self.vocab.token_unk());
                position += 1;
                continue;
            }

            // traverse the trie to find the longest matching token
            let start = position;
            let mut token_id = 0;
            let mut token_length = 0usize;
            while let Some(n) = node {
                if n.has_value {
                    token_id = n.value;
                    token_length = position + 1;
                }
                if position + 1 >= bytes.len() {
                    break;
                }
                position += 1;
                node = n.traverse(bytes[position]);
            }

            // add the longest matching token
            output.push(token_id);
            position = if token_length > 0 { token_length } else { start + 1 };
        }
    }
}

// PLaMo2 tokenizer
impl Plamo2Data {
    fn build(
        id_to_token: &[TokenData],
        _token_to_id_by_text: &HashMap<String, Token>,
        pre_segment: bool,
    ) -> Result<Self> {
        let n_tokens = id_to_token.len();
        let mut tokens = Vec::with_capacity(n_tokens);
        let mut bytes: Vec<Token> = vec![0; 256];
        let mut to_suffix_id: HashMap<i64, i32> = HashMap::new();
        let mut table: Vec<[i32; 4]> = Vec::new();

        // Suffixes and pieces are views into the token texts of the vocab, so no strings are built
        let mut suffix_to_score: HashMap<Vec<u8>, f32> = HashMap::with_capacity(n_tokens * 4);
        let mut token_to_id: HashMap<Vec<u8>, Token> = HashMap::with_capacity(n_tokens);

        // Byte offset of the next code point
        fn next_cpt(s: &[u8], offset: usize) -> usize {
            offset + (s.len() - offset).min(unicode_len_utf8(s[offset]))
        }

        for token_id in 0..n_tokens as Token {
            let entry = &id_to_token[token_id as usize];
            tokens.push(entry.text.clone());
            token_to_id.insert(entry.text.as_bytes().to_vec(), token_id);

            // Handle byte tokens
            if entry.attr & LLAMA_TOKEN_ATTR_BYTE != 0 {
                let t = entry.text.as_bytes();
                if t.len() == 6 && &t[..3] == b"<0x" && t[5] == b'>' {
                    if t[3].is_ascii_hexdigit() && t[4].is_ascii_hexdigit() {
                        let hex_str = &entry.text[3..5];
                        let byte_val = u8::from_str_radix(hex_str, 16).unwrap_or(0);
                        bytes[byte_val as usize] = token_id;
                    }
                }
                continue;
            }

            // Add token and all its suffixes to suffix_to_score
            let text = entry.text.as_bytes().to_vec();
            suffix_to_score.insert(text.clone(), entry.score);
            let mut i = if text.is_empty() { 0 } else { next_cpt(&text, 0) };
            while i < text.len() {
                suffix_to_score.entry(text[i..].to_vec()).or_insert(f32::NAN);
                i = next_cpt(&text, i);
            }
        }

        // Check that all byte tokens are set
        for (i, &id) in bytes.iter().enumerate() {
            if id == 0 {
                return Err(Error::new(format!("Byte token for <0x{:x}> is not set", i)));
            }
        }

        // Build suffix list in lexicographical order of reversed strings
        let mut suffixes: Vec<Vec<u8>> = suffix_to_score.keys().cloned().collect();
        suffixes.push(Vec::new()); // Empty suffix
        suffixes.sort_by(|a, b| a.iter().rev().cmp(b.iter().rev()));

        // Byte length of each prefix of the suffix, by code point
        let mut prefix_ends: Vec<usize> = Vec::new();
        fn get_prefix_ends(suffix: &[u8]) -> Vec<usize> {
            let mut out = Vec::new();
            let mut i = 0usize;
            while i < suffix.len() {
                i = next_cpt(suffix, i);
                out.push(i);
            }
            out
        }

        // Build suffix_to_id and to_suffix_id_
        let mut suffix_to_id: HashMap<Vec<u8>, i32> = HashMap::with_capacity(suffixes.len());
        let mut num_pieces = 0i32;

        for suffix in &suffixes {
            suffix_to_id.insert(suffix.clone(), num_pieces);
            if !suffix.is_empty() {
                prefix_ends = get_prefix_ends(suffix);
                let first_bytes = &suffix[..prefix_ends[0]];
                let first_str = std::str::from_utf8(first_bytes).unwrap_or("");
                let first_cpt = unicode_cpts_from_utf8(first_str).first().copied().unwrap_or(0xfffd);
                let piece_code = ((first_cpt as i64) << 32) | suffix_to_id[&suffix[prefix_ends[0]..]] as i64;
                to_suffix_id.insert(piece_code, num_pieces);

                // Count number of pieces for this suffix
                let mut pieces_for_suffix = 1i32; // sentinel row
                for &end in &prefix_ends {
                    if suffix_to_score.contains_key(&suffix[..end]) {
                        pieces_for_suffix += 1;
                    }
                }
                num_pieces += pieces_for_suffix;
            } else {
                num_pieces += 1; // Empty suffix contributes one piece (sentinel row)
            }
        }

        // Build flattened table
        table.resize(num_pieces as usize, [0; 4]);
        let mut table_idx = 0usize;

        const TABLE_PIECE_LENGTH: usize = 0;
        const TABLE_TOKEN_ID: usize = 1;
        const TABLE_SCORE: usize = 2;
        const TABLE_PIECE_ID: usize = 3;
        const INVALID_SCORE: i32 = -20_000_000;
        const UNKNOWN_SCORE: i32 = -10_000_000;

        for suffix in &suffixes {
            // Add all prefixes of the suffix to the table (in decreasing order of length)
            prefix_ends = get_prefix_ends(suffix);
            for &piece_length in prefix_ends.iter().rev() {
                let piece = &suffix[..piece_length];
                let score = match suffix_to_score.get(piece) {
                    Some(score) => *score,
                    None => continue,
                };
                table[table_idx][TABLE_PIECE_LENGTH] = piece_length as i32;
                table[table_idx][TABLE_TOKEN_ID] = token_to_id.get(piece).copied().unwrap_or(-1);
                table[table_idx][TABLE_SCORE] = if score.is_finite() {
                    (score * 1e4).round() as i32
                } else {
                    INVALID_SCORE
                };
                table[table_idx][TABLE_PIECE_ID] = suffix_to_id.get(piece).copied().unwrap_or(0);
                table_idx += 1;
            }

            // Add sentinel row
            table[table_idx][TABLE_PIECE_LENGTH] = 1;
            table[table_idx][TABLE_TOKEN_ID] = -1;
            table[table_idx][TABLE_SCORE] = UNKNOWN_SCORE;
            table_idx += 1;
        }

        Ok(Self { tokens, bytes, to_suffix_id, table, pre_segment })
    }

    fn encode(&self, text: &str) -> Vec<Token> {
        let mut unicode_data = unicode_cpts_from_utf8(text);

        // The PLaMo-3 tokenizer keeps a leading U+FEFF in the input.
        if !self.pre_segment {
            if unicode_data.first() == Some(&0xfeff) {
                unicode_data.remove(0);
            }
            return self.encode_cpts(&unicode_data);
        }

        let n = unicode_data.len();
        let mut cut = vec![false; n + 1];

        // pass 1: <|plamo:...|>
        {
            const PREFIX: [u32; 8] = [
                b'<' as u32, b'|' as u32, b'p' as u32, b'l' as u32,
                b'a' as u32, b'm' as u32, b'o' as u32, b':' as u32,
            ];
            let prefix_len = PREFIX.len();
            let mut i = 0usize;
            while i + prefix_len <= n {
                if unicode_data[i..i + prefix_len] != PREFIX[..] {
                    i += 1;
                    continue;
                }
                // An empty body is valid.
                let mut j = i + prefix_len;
                // Treat U+001C..U+001F as whitespace (equivalent to Python \s).
                while j < n && j - (i + prefix_len) < 64 && unicode_data[j] != b'|' as u32
                    && !(0x1c..=0x1f).contains(&unicode_data[j])
                    && !unicode_cpt_flags_from_cpt(unicode_data[j]).is_whitespace()
                {
                    j += 1;
                }
                if j + 1 < n && unicode_data[j] == b'|' as u32 && unicode_data[j + 1] == b'>' as u32 {
                    cut[i] = true;
                    cut[j + 2] = true;
                    i = j + 2;
                } else {
                    i += 1;
                }
            }
        }

        // pass 2: runs of repeated characters / spaces (a run never crosses a boundary from pass 1)
        {
            let mut i = 0usize;
            while i < n {
                let c = unicode_data[i];
                let mut run = 1usize;
                while i + run < n && unicode_data[i + run] == c && !cut[i + run] {
                    run += 1;
                }

                let is_repeated_chars = c != b'\n' as u32 && run >= 4;
                let is_spaces = c == b' ' as u32 && run >= 2;
                if is_repeated_chars || is_spaces {
                    cut[i] = true;
                    cut[i + run] = true;
                }

                // a run that does not match cannot match at any later position either (it only gets shorter)
                i += run;
            }
        }

        let mut output = Vec::new();
        let mut seg_start = 0usize;
        for seg_end in 0..=n {
            // U+EE00 is the tokenizer's private-use boundary marker; literal occurrences split segments and are not emitted.
            let is_boundary = seg_end < n && unicode_data[seg_end] == 0xee00;
            if seg_end == n || cut[seg_end] || is_boundary {
                if seg_start < seg_end {
                    let segment = unicode_data[seg_start..seg_end].to_vec();
                    output.extend(self.encode_cpts(&segment));
                }
                seg_start = seg_end + is_boundary as usize;
            }
        }

        output
    }

    fn encode_cpts(&self, unicode_data: &[u32]) -> Vec<Token> {
        if unicode_data.is_empty() {
            return Vec::new();
        }

        const TABLE_PIECE_LENGTH: usize = 0;
        const TABLE_TOKEN_ID: usize = 1;
        const TABLE_SCORE: usize = 2;
        const TABLE_PIECE_ID: usize = 3;
        const PATH_TOKEN_LENGTH: usize = 0;
        const PATH_TOKEN_ID: usize = 1;
        const PATH_NUM_TOKENS: usize = 2;
        const INVALID_SCORE: i32 = -20_000_000;
        const UNKNOWN_SCORE: i32 = -10_000_000;

        let data_len = unicode_data.len();

        // Initialize scores array (dynamic programming)
        let mut scores = vec![1i64 << 60; data_len + 1];
        scores[data_len] = 0;

        // Path array to track best tokenization
        let mut path = vec![[0i32; 3]; data_len + 1];

        let mut suffix_id = 0usize;

        // Process from end to beginning
        for i in (0..data_len as i32).rev() {
            let c = unicode_data[i as usize];

            // Find next suffix ID
            for p in suffix_id..self.table.len() {
                let piece_code = ((c as i64) << 32) | self.table[p][TABLE_PIECE_ID] as i64;
                suffix_id = self.to_suffix_id.get(&piece_code).copied().unwrap_or(0) as usize;

                if suffix_id > 0 || self.table[p][TABLE_SCORE] == UNKNOWN_SCORE {
                    break;
                }
            }

            // Update best path
            for p in suffix_id..self.table.len() {
                let score = self.table[p][TABLE_SCORE];
                if score > INVALID_SCORE {
                    let piece_length = self.table[p][TABLE_PIECE_LENGTH] as usize;
                    let next_score = scores.get(i as usize + piece_length).copied().unwrap_or(1i64 << 60);
                    let s = next_score - score as i64;

                    if s < scores[i as usize] {
                        scores[i as usize] = s;
                        path[i as usize][PATH_TOKEN_LENGTH] = piece_length as i32;
                        path[i as usize][PATH_TOKEN_ID] = self.table[p][TABLE_TOKEN_ID];
                        let n = path.get(i as usize + piece_length).map(|p| p[PATH_NUM_TOKENS]).unwrap_or(0);
                        path[i as usize][PATH_NUM_TOKENS] = n + 1;

                        if score == UNKNOWN_SCORE {
                            // Add UTF-8 byte count
                            path[i as usize][PATH_NUM_TOKENS] +=
                                (c >= 0x80) as i32 + (c >= 0x800) as i32 + (c >= 0x10000) as i32;
                        }
                    }
                }

                if score == UNKNOWN_SCORE {
                    break;
                }
            }
        }

        // Decode the best path
        let mut token_ids = Vec::with_capacity(path[0][PATH_NUM_TOKENS] as usize);
        let mut pos = 0usize;
        while pos < data_len {
            if path[pos][PATH_TOKEN_ID] >= 0 {
                token_ids.push(path[pos][PATH_TOKEN_ID]);
            } else {
                // Fall back to byte tokens
                let c = unicode_data[pos];
                let s = 1 + (c >= 0x80) as u32 + (c >= 0x800) as u32 + (c >= 0x10000) as u32;

                for i in 0..s {
                    let b: u8 = if s == 1 {
                        c as u8
                    } else if i == 0 {
                        ((0xf00 >> s) & 0xff) as u8
                    } else {
                        0x80
                    };
                    let idx = (b | ((c >> ((s - i - 1) * 6)) & 0x3f) as u8) as usize;
                    if let Some(&id) = self.bytes.get(idx) {
                        token_ids.push(id);
                    }
                }
            }

            assert!(path[pos][PATH_TOKEN_LENGTH] > 0);
            pos += path[pos][PATH_TOKEN_LENGTH] as usize;
        }

        token_ids
    }
}

struct Plamo2Session<'a> {
    tokenizer: &'a Plamo2Data,
}

impl<'a> Plamo2Session<'a> {
    fn tokenize(&self, text: &str, output: &mut Vec<Token>) {
        let tokens = self.tokenizer.encode(text);
        output.extend(tokens);
    }
}

// HybridDNATokenizer (HuggingFaceBio/Carbon-*): text inside <dna>...</dna> is cut into fixed 6-mers, everything else goes through the base BPE
// Mirrors HybridDNATokenizer.encode, with optional tagging of untagged DNA (auto_dna_tags, Carbon-14B) and tail="pad_n" vocab, whose dedicated tokens pad a trailing partial k-mer with N instead of A ("CT" -> "CTNNNN")
struct HybridDnaSession<'a> {
    bpe: BpeSession<'a>,
    vocab: &'a Impl,
    dna_begin_id: Token,
    dna_end_id: Token,
    dna_oov_id: Token,
    pad_char: u8,
    open_tag: &'static str,
    close_tag: &'static str,
    k: usize,
}

impl<'a> HybridDnaSession<'a> {
    fn new(vocab: &'a Impl, tokenizer: &'a BpeData) -> Self {
        let dna_begin_id = vocab.text_to_token("<dna>");
        let dna_end_id = vocab.text_to_token("</dna>");
        let dna_oov_id = vocab.text_to_token("<oov>");
        let pad_char = if vocab
            .text_to_token(&format!("A{}{}", "N".repeat(5), DNA_KMER_MARKER))
            != LLAMA_TOKEN_NULL
        {
            b'N'
        } else {
            b'A'
        };
        Self {
            bpe: BpeSession::new(vocab, tokenizer),
            vocab,
            dna_begin_id,
            dna_end_id,
            dna_oov_id,
            pad_char,
            open_tag: "<dna>",
            close_tag: "</dna>",
            k: 6,
        }
    }

    fn tokenize(&mut self, text: &str, output: &mut Vec<Token>) {
        // Fall back to plain BPE if the DNA pieces aren't in the vocab.
        if self.dna_begin_id == LLAMA_TOKEN_NULL
            || self.dna_end_id == LLAMA_TOKEN_NULL
            || self.dna_oov_id == LLAMA_TOKEN_NULL
        {
            self.bpe.tokenize(text, output);
            return;
        }

        // split on explicit tags as HybridDNATokenizer._split_by_dna_tags does; only the text outside them is auto-tagged
        let mut pos = 0usize;
        while pos < text.len() {
            const NPOS: usize = usize::MAX;
            let start = text[pos..].find(self.open_tag).map_or(NPOS, |p| p + pos);
            let end = text[pos..].find(self.close_tag).map_or(NPOS, |p| p + pos);

            if start == NPOS && end == NPOS {
                self.tokenize_text(&text[pos..], output);
                break;
            }
            if start == NPOS || end < start {
                // as in _split_by_dna_tags, text before a </dna> without a matching <dna> is DNA (no begin token)
                self.tokenize_dna(&text[pos..end], false, true, output);
                pos = end + self.close_tag.len();
                continue;
            }

            self.tokenize_text(&text[pos..start], output);

            let content_start = start + self.open_tag.len();
            if end == NPOS {
                // an unterminated <dna> runs to the end of the text (no end token)
                self.tokenize_dna(&text[content_start..], true, false, output);
                break;
            }
            self.tokenize_dna(&text[content_start..end], true, true, output);
            pos = end + self.close_tag.len();
        }
    }

    fn is_letter(c: u8) -> bool {
        c.is_ascii_alphabetic()
    }
    fn is_acgt(c: u8) -> bool {
        c == b'A' || c == b'C' || c == b'G' || c == b'T'
    }
    fn to_upper(c: u8) -> u8 {
        if c.is_ascii_lowercase() { c - 32 } else { c }
    }

    // IUPAC nucleotide codes, either case (HybridDNATokenizer.IUPAC_RUN_RE)
    fn is_iupac(c: u8) -> bool {
        b"ACGTRYSWKMBDHVN".contains(&Self::to_upper(c))
    }

    // HybridDNATokenizer._normalize_spaced_dna: join displays like "ACGTA CGTAC ..." when they hold at least twelve bases
    // (at least six chunks of 1-5 uppercase ACGT separated by spaces or tabs, not touching a letter)
    fn normalize_spaced_dna(text: &str) -> String {
        let is_blank = |c: u8| c == b' ' || c == b'\t';
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut p = 0usize;
        while p < bytes.len() {
            if Self::is_acgt(bytes[p]) && (p == 0 || !Self::is_letter(bytes[p - 1])) {
                let mut match_end = None;
                let mut chunk = p;
                let mut n_chunks = 1;
                loop {
                    let mut chunk_end = chunk;
                    while chunk_end < bytes.len() && chunk_end - chunk < 5 && Self::is_acgt(bytes[chunk_end]) {
                        chunk_end += 1;
                    }
                    if chunk_end == chunk {
                        break;
                    }
                    if n_chunks >= 6 && (chunk_end == bytes.len() || !Self::is_letter(bytes[chunk_end])) {
                        match_end = Some(chunk_end); // the regex is greedy, so the last valid end wins
                    }
                    let mut next = chunk_end;
                    while next < bytes.len() && is_blank(bytes[next]) {
                        next += 1;
                    }
                    if next == chunk_end {
                        break;
                    }
                    chunk = next;
                    n_chunks += 1;
                }
                if let Some(end) = match_end {
                    let mut seq = Vec::new();
                    for &b in &bytes[p..end] {
                        if !is_blank(b) {
                            seq.push(b);
                        }
                    }
                    if seq.len() >= 12 {
                        out.extend_from_slice(&seq);
                    } else {
                        out.extend_from_slice(&bytes[p..end]);
                    }
                    p = end;
                    continue;
                }
            }
            out.push(bytes[p]);
            p += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    // text outside explicit tags: with auto_dna_tags, tag DNA runs as HybridDNATokenizer._auto_tag_text_segment does
    // (pure ACGT runs of at least 6 bases, or runs of at least 12 with at most 20% IUPAC ambiguity codes)
    fn tokenize_text(&mut self, raw: &str, output: &mut Vec<Token>) {
        let text = if self.vocab.get_dna_auto_tags() {
            Self::normalize_spaced_dna(raw)
        } else {
            raw.to_string()
        };
        if !self.vocab.get_dna_auto_tags() {
            if !text.is_empty() {
                self.bpe.tokenize(&text, output);
            }
            return;
        }

        let bytes = text.as_bytes();
        let mut prev = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            if !Self::is_iupac(bytes[i]) {
                i += 1;
                continue;
            }
            let mut j = i;
            let mut n_ambiguous = 0usize;
            while j < bytes.len() && Self::is_iupac(bytes[j]) {
                n_ambiguous += !Self::is_acgt(Self::to_upper(bytes[j])) as usize;
                j += 1;
            }
            let len = j - i;
            if (n_ambiguous == 0 && len >= 6) || (n_ambiguous > 0 && len >= 12 && 5 * n_ambiguous <= len) {
                if i > prev {
                    let piece = String::from_utf8_lossy(&bytes[prev..i]).into_owned();
                    self.bpe.tokenize(&piece, output);
                }
                let dna = String::from_utf8_lossy(&bytes[i..j]).into_owned();
                self.tokenize_dna(&dna, true, true, output);
                prev = j;
            }
            i = j;
        }
        if prev < text.len() {
            let piece = String::from_utf8_lossy(&bytes[prev..]).into_owned();
            self.bpe.tokenize(&piece, output);
        }
    }

    // HybridDNATokenizer._process_dna_sequence: strip, uppercase, then one token per k bases
    // a k-mer with anything but ACGT is <oov>, and a trailing partial k-mer is right-padded with pad_char
    fn tokenize_dna(&self, content: &str, has_start: bool, has_end: bool, output: &mut Vec<Token>) {
        if has_start {
            output.push(self.dna_begin_id);
        }

        let cpts = unicode_cpts_from_utf8(content);
        let mut first = 0usize;
        let mut last = cpts.len();
        while first < last && unicode_cpt_flags_from_cpt(cpts[first]).is_whitespace() {
            first += 1;
        }
        while last > first && unicode_cpt_flags_from_cpt(cpts[last - 1]).is_whitespace() {
            last -= 1;
        }

        // k-mers carry the reserved marker suffix, so they never resolve to a base-vocab token
        let kmer_token = |from: usize, to: usize| -> Token {
            let mut kmer = Vec::new();
            for &cpt in &cpts[from..to] {
                let c = if cpt < 128 { Self::to_upper(cpt as u8) as u32 } else { cpt };
                if c > 127 || !Self::is_acgt(c as u8) {
                    return self.dna_oov_id;
                }
                kmer.push(c as u8);
            }
            while kmer.len() < self.k {
                kmer.push(self.pad_char);
            }
            let text = format!("{}{}", String::from_utf8_lossy(&kmer), DNA_KMER_MARKER);
            let tok = self.vocab.text_to_token(&text);
            if tok != LLAMA_TOKEN_NULL { tok } else { self.dna_oov_id }
        };

        let mut i = first;
        while i + self.k <= last {
            output.push(kmer_token(i, i + self.k));
            i += self.k;
        }
        if i < last {
            output.push(kmer_token(i, last));
        }

        if has_end {
            output.push(self.dna_end_id);
        }
    }
}

struct WhitespaceSession<'a> {
    bpe: BpeSession<'a>,
    vocab: &'a Impl,
}

impl<'a> WhitespaceSession<'a> {
    fn tokenize(&mut self, text: &str, output: &mut Vec<Token>) {
        let lowercase = self.vocab.get_normalizer_opts().lowercase;

        let mut segment = String::new();
        let mut flush = |segment: &mut String, output: &mut Vec<Token>| {
            if !segment.is_empty() {
                self.bpe.tokenize(segment, output);
                segment.clear();
            }
        };

        for cpt in unicode_cpts_from_utf8(text) {
            // drop whitespace
            if unicode_cpt_flags_from_cpt(cpt).is_whitespace() {
                flush(&mut segment, output);
            } else {
                segment.push_str(&unicode_cpt_to_utf8(if lowercase { unicode_tolower(cpt) } else { cpt }));
            }
        }
        flush(&mut segment, output);
    }
}

// ---------------------------------------------------------------------------
// metadata keys and loader input
// ---------------------------------------------------------------------------

/// Named GGUF metadata keys used by `llama_vocab::load`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Kv(pub &'static str);

impl Kv {
    pub const VOCAB_SIZE: Kv = Kv("vocab_size");
    pub const GENERAL_NAME: Kv = Kv("general.name");
    pub const GENERAL_ARCHITECTURE: Kv = Kv("general.architecture");
    pub const TOKENIZER_MODEL: Kv = Kv("tokenizer.ggml.model");
    pub const TOKENIZER_PRE: Kv = Kv("tokenizer.ggml.pre");
    pub const TOKENIZER_LIST: Kv = Kv("tokenizer.ggml.tokens");
    pub const TOKENIZER_SCORES: Kv = Kv("tokenizer.ggml.scores");
    pub const TOKENIZER_TOKEN_TYPE: Kv = Kv("tokenizer.ggml.token_type");
    pub const TOKENIZER_TOKEN_TYPE_COUNT: Kv = Kv("tokenizer.ggml.token_type_count");
    pub const TOKENIZER_MERGES: Kv = Kv("tokenizer.ggml.merges");
    pub const TOKENIZER_PRECOMPILED_CHARSMAP: Kv = Kv("tokenizer.ggml.precompiled_charsmap");
    pub const TOKENIZER_ADD_PREFIX: Kv = Kv("tokenizer.ggml.add_space_prefix");
    pub const TOKENIZER_REMOVE_EXTRA_WS: Kv = Kv("tokenizer.ggml.remove_extra_whitespaces");
    pub const TOKENIZER_CLEAN_SPACES: Kv = Kv("tokenizer.ggml.clean_up_tokenization_spaces");
    pub const TOKENIZER_ADD_BOS: Kv = Kv("tokenizer.ggml.add_bos_token");
    pub const TOKENIZER_ADD_EOS: Kv = Kv("tokenizer.ggml.add_eos_token");
    pub const TOKENIZER_ADD_SEP: Kv = Kv("tokenizer.ggml.add_sep_token");
    pub const TOKENIZER_DNA_AUTO_TAGS: Kv = Kv("tokenizer.ggml.dna_auto_tags");
    pub const TOKENIZER_BOS_ID: Kv = Kv("tokenizer.ggml.bos_token_id");
    pub const TOKENIZER_EOS_ID: Kv = Kv("tokenizer.ggml.eos_token_id");
    pub const TOKENIZER_EOT_ID: Kv = Kv("tokenizer.ggml.eot_token_id");
    pub const TOKENIZER_EOM_ID: Kv = Kv("tokenizer.ggml.eom_token_id");
    pub const TOKENIZER_UNK_ID: Kv = Kv("tokenizer.ggml.unknown_token_id");
    pub const TOKENIZER_SEP_ID: Kv = Kv("tokenizer.ggml.separator_token_id");
    pub const TOKENIZER_PAD_ID: Kv = Kv("tokenizer.ggml.padding_token_id");
    pub const TOKENIZER_MASK_ID: Kv = Kv("tokenizer.ggml.mask_token_id");
    pub const TOKENIZER_FIM_PRE_ID: Kv = Kv("tokenizer.ggml.fim_prefix_token_id");
    pub const TOKENIZER_FIM_SUF_ID: Kv = Kv("tokenizer.ggml.fim_suffix_token_id");
    pub const TOKENIZER_FIM_MID_ID: Kv = Kv("tokenizer.ggml.fim_middle_token_id");
    pub const TOKENIZER_FIM_PAD_ID: Kv = Kv("tokenizer.ggml.fim_pad_token_id");
    pub const TOKENIZER_FIM_REP_ID: Kv = Kv("tokenizer.ggml.fim_repo_token_id");
    pub const TOKENIZER_FIM_SEP_ID: Kv = Kv("tokenizer.ggml.fim_sep_token_id");
    pub const TOKENIZER_PREFIX_ID: Kv = Kv("tokenizer.ggml.prefix_token_id");
    pub const TOKENIZER_SUFFIX_ID: Kv = Kv("tokenizer.ggml.suffix_token_id");
    pub const TOKENIZER_MIDDLE_ID: Kv = Kv("tokenizer.ggml.middle_token_id");
    pub const TOKENIZER_NORMALIZER_NFC: Kv = Kv("tokenizer.ggml.normalizer.nfc");
    pub const TOKENIZER_NORMALIZER_LOWERCASE: Kv = Kv("tokenizer.ggml.normalizer.lowercase");
    pub const TOKENIZER_NORMALIZER_STRIP_ACCENTS: Kv = Kv("tokenizer.ggml.normalizer.strip_accents");
    pub const TOKENIZER_METASPACE_PREPEND_SCHEME: Kv = Kv("tokenizer.ggml.metaspace_prepend_scheme");
    pub const TOKENIZER_SUPPRESS_TOKENS: Kv = Kv("tokenizer.ggml.suppress_tokens");
}

/// Stands in for `llama_model_loader`/`gguf_context`. Populate the maps and
/// pass to [`LlamaVocab::load`].
#[derive(Clone, Debug, Default)]
pub struct Metadata {
    pub strings: HashMap<String, String>,
    pub bools: HashMap<String, bool>,
    pub u32s: HashMap<String, u32>,
    pub string_arrays: HashMap<String, Vec<String>>,
    pub i32_arrays: HashMap<String, Vec<i32>>,
    pub f32_arrays: HashMap<String, Vec<f32>>,
    pub blobs: HashMap<String, Vec<u8>>,
}

impl Metadata {
    pub fn set_string(&mut self, key: Kv, value: impl Into<String>) {
        self.strings.insert(key.0.to_string(), value.into());
    }
    pub fn set_bool(&mut self, key: Kv, value: bool) {
        self.bools.insert(key.0.to_string(), value);
    }
    pub fn set_u32(&mut self, key: Kv, value: u32) {
        self.u32s.insert(key.0.to_string(), value);
    }
    pub fn set_string_array(&mut self, key: Kv, value: Vec<String>) {
        self.string_arrays.insert(key.0.to_string(), value);
    }
    pub fn set_i32_array(&mut self, key: Kv, value: Vec<i32>) {
        self.i32_arrays.insert(key.0.to_string(), value);
    }
    pub fn set_f32_array(&mut self, key: Kv, value: Vec<f32>) {
        self.f32_arrays.insert(key.0.to_string(), value);
    }
    pub fn set_blob(&mut self, key: Kv, value: Vec<u8>) {
        self.blobs.insert(key.0.to_string(), value);
    }

    pub fn get_string(&self, key: Kv, dst: &mut String) -> bool {
        match self.strings.get(key.0) {
            Some(v) => {
                *dst = v.clone();
                true
            }
            None => false,
        }
    }
    pub fn get_bool(&self, key: Kv, dst: &mut bool) -> bool {
        match self.bools.get(key.0) {
            Some(v) => {
                *dst = *v;
                true
            }
            None => false,
        }
    }
    pub fn get_u32(&self, key: Kv, dst: &mut u32) -> bool {
        match self.u32s.get(key.0) {
            Some(v) => {
                *dst = *v;
                true
            }
            None => false,
        }
    }
    pub fn get_string_array(&self, key: Kv) -> Option<&Vec<String>> {
        self.string_arrays.get(key.0)
    }
    pub fn get_i32_array(&self, key: Kv) -> Option<&Vec<i32>> {
        self.i32_arrays.get(key.0)
    }
    pub fn get_f32_array(&self, key: Kv) -> Option<&Vec<f32>> {
        self.f32_arrays.get(key.0)
    }
    pub fn get_blob(&self, key: Kv) -> Option<&Vec<u8>> {
        self.blobs.get(key.0)
    }
}

// ---------------------------------------------------------------------------
// llama_vocab::impl
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Fragment<'a> {
    Token(Token),
    RawText {
        raw_text: &'a str,
        offset: usize,
        length: usize,
    },
}

struct Impl {
    n_token_types: u32, // for BERT-style token types

    tokenizer_model: String,
    tokenizer_pre: String,

    vtype: VocabType,
    pre_type: VocabPreType,

    max_token_len: i32, // used for optimizing longest token search

    // default LLaMA special tokens
    // TODO: should we set all of these to LLAMA_TOKEN_NULL?
    special_bos_id: Token,
    special_eos_id: Token,
    special_eot_id: Token,
    special_eom_id: Token,
    special_unk_id: Token,
    special_sep_id: Token,
    special_pad_id: Token,
    special_mask_id: Token,

    linefeed_id: Token,

    // fim tokens
    special_fim_pre_id: Token,
    special_fim_suf_id: Token,
    special_fim_mid_id: Token,
    special_fim_pad_id: Token,
    special_fim_rep_id: Token, // repo
    special_fim_sep_id: Token, // file separator

    // tokenizer flags
    add_space_prefix: bool,
    add_bos: bool,
    add_eos: bool,
    add_sep: bool,
    dna_auto_tags: bool, // hybriddna: tag untagged DNA runs before tokenizing
    ignore_merges: bool,
    clean_spaces: bool, // clean_up_tokenization_spaces
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    treat_whitespace_as_suffix: bool,

    normalizer_opts: NormalizerOptions,
    metaspace_prepend_scheme: MetaspacePrependScheme,

    token_to_id: HashMap<String, Token>,
    id_to_token: Vec<TokenData>,

    cache_special_tokens: Vec<Token>,
    cache_token_to_piece: Vec<String>, // llama_token_to_piece(special = true)
    bpe_ranks: HashMap<(String, String), i32>,

    // set of all tokens that cause "end of generation"
    special_eog_ids: BTreeSet<Token>,

    suppress_tokens: Vec<Token>,

    tokenizer: Tokenizer,

    precompiled_charsmap: Vec<u8>,
}

impl Default for Impl {
    fn default() -> Self {
        Self {
            n_token_types: 0,
            tokenizer_model: String::new(),
            tokenizer_pre: String::new(),
            vtype: VocabType::Spm,
            pre_type: VocabPreType::Default,
            max_token_len: 0,
            special_bos_id: 1,
            special_eos_id: 2,
            special_eot_id: LLAMA_TOKEN_NULL,
            special_eom_id: LLAMA_TOKEN_NULL,
            special_unk_id: 0,
            special_sep_id: LLAMA_TOKEN_NULL,
            special_pad_id: LLAMA_TOKEN_NULL,
            special_mask_id: LLAMA_TOKEN_NULL,
            linefeed_id: 13,
            special_fim_pre_id: LLAMA_TOKEN_NULL,
            special_fim_suf_id: LLAMA_TOKEN_NULL,
            special_fim_mid_id: LLAMA_TOKEN_NULL,
            special_fim_pad_id: LLAMA_TOKEN_NULL,
            special_fim_rep_id: LLAMA_TOKEN_NULL,
            special_fim_sep_id: LLAMA_TOKEN_NULL,
            add_space_prefix: false,
            add_bos: false,
            add_eos: false,
            add_sep: false,
            dna_auto_tags: false,
            ignore_merges: false,
            clean_spaces: false,
            remove_extra_whitespaces: false,
            escape_whitespaces: true,
            treat_whitespace_as_suffix: false,
            normalizer_opts: NormalizerOptions::default(),
            metaspace_prepend_scheme: MetaspacePrependScheme::Never,
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            cache_special_tokens: Vec::new(),
            cache_token_to_piece: Vec::new(),
            bpe_ranks: HashMap::new(),
            special_eog_ids: BTreeSet::new(),
            suppress_tokens: Vec::new(),
            tokenizer: Tokenizer::Spm,
            precompiled_charsmap: Vec::new(),
        }
    }
}

// helpers shared with the tokenizer sessions
impl Impl {
    fn n_tokens(&self) -> u32 {
        self.id_to_token.len() as u32
    }
    fn max_token_len(&self) -> i32 {
        self.max_token_len
    }
    fn get_tokenizer_model(&self) -> &str {
        &self.tokenizer_model
    }
    fn get_pre_type(&self) -> VocabPreType {
        self.pre_type
    }
    fn get_normalizer_opts(&self) -> NormalizerOptions {
        self.normalizer_opts
    }
    fn get_metaspace_prepend_scheme(&self) -> MetaspacePrependScheme {
        self.metaspace_prepend_scheme
    }
    fn get_add_space_prefix(&self) -> bool {
        self.add_space_prefix
    }
    fn get_add_bos(&self) -> bool {
        self.add_bos
    }
    fn get_add_eos(&self) -> bool {
        self.add_eos
    }
    fn get_remove_extra_whitespaces(&self) -> bool {
        self.remove_extra_whitespaces
    }
    fn get_escape_whitespaces(&self) -> bool {
        self.escape_whitespaces
    }
    fn get_treat_whitespace_as_suffix(&self) -> bool {
        self.treat_whitespace_as_suffix
    }
    fn get_ignore_merges(&self) -> bool {
        self.ignore_merges
    }
    fn get_dna_auto_tags(&self) -> bool {
        self.dna_auto_tags
    }
    fn get_clean_spaces(&self) -> bool {
        self.clean_spaces
    }

    fn text_to_token(&self, text: &str) -> Token {
        self.token_to_id.get(text).copied().unwrap_or(LLAMA_TOKEN_NULL)
    }

    fn get_token_data(&self, id: Token) -> &TokenData {
        &self.id_to_token[id as usize]
    }

    fn token_get_score(&self, id: Token) -> f32 {
        self.id_to_token[id as usize].score
    }

    fn token_get_attr(&self, id: Token) -> LlamaTokenAttr {
        self.id_to_token[id as usize].attr
    }

    fn is_user_defined(&self, id: Token) -> bool {
        self.id_to_token[id as usize].attr & LLAMA_TOKEN_ATTR_USER_DEFINED != 0
    }

    fn token_unk(&self) -> Token {
        self.special_unk_id
    }
    fn token_bos(&self) -> Token {
        self.special_bos_id
    }
    fn token_eos(&self) -> Token {
        self.special_eos_id
    }

    fn find_bpe_rank(&self, token_left: &str, token_right: &str) -> i32 {
        debug_assert!(!token_left.contains(' '));
        debug_assert!(!token_right.contains(' '));
        self.bpe_ranks
            .get(&(token_left.to_string(), token_right.to_string()))
            .copied()
            .unwrap_or(-1)
    }

    fn byte_to_token(&self, ch: u8) -> Token {
        match self.vtype {
            VocabType::Spm | VocabType::Ugm => {
                const HEX: &[u8; 16] = b"0123456789ABCDEF";
                let buf = [
                    b'<', b'0', b'x', HEX[(ch >> 4) as usize], HEX[(ch & 15) as usize], b'>',
                ];
                let name = std::str::from_utf8(&buf).unwrap_or("");
                if let Some(&token) = self.token_to_id.get(name) {
                    return token;
                }
                // Try to fall back to just the byte as a string
                self.token_to_id
                    .get(&String::from_utf8_lossy(&[ch]).into_owned())
                    .copied()
                    .unwrap_or(LLAMA_TOKEN_NULL)
            }
            VocabType::Wpm | VocabType::Bpe => self
                .token_to_id
                .get(&unicode_byte_to_utf8(ch))
                .copied()
                .unwrap_or(LLAMA_TOKEN_NULL),
            VocabType::Plamo2 | VocabType::Plamo3 => {
                let hex_str = format!("<0x{:02X}>", ch);
                self.token_to_id.get(&hex_str).copied().unwrap_or(LLAMA_TOKEN_NULL)
            }
            VocabType::Test => LLAMA_TOKEN_NULL, // TEST tokens have no byte-level mapping
            VocabType::Rwkv => LLAMA_TOKEN_NULL, // RWKV tokens have no byte-level mapping
            VocabType::None => LLAMA_TOKEN_NULL,
        }
    }

    fn load_bpe_ranks(ml: &Metadata) -> Result<Option<HashMap<(String, String), i32>>> {
        let merges = match ml.get_string_array(Kv::TOKENIZER_MERGES) {
            Some(v) => v,
            None => return Ok(None),
        };
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, word) in merges.iter().enumerate() {
            let sep = word
                .char_indices()
                .skip(1)
                .find(|(_, c)| *c == ' ')
                .map(|(p, _)| p);
            let left = match sep {
                Some(s) => word[..s].to_string(),
                None => String::new(),
            };
            let right = match sep {
                Some(s) => word[s + 1..].to_string(),
                None => String::new(),
            };
            ranks.entry((left, right)).or_insert(i as i32);
        }
        Ok(Some(ranks))
    }

    fn load(&mut self, ml: &Metadata) -> Result<()> {
        // determine vocab type
        {
            ml.get_string(Kv::TOKENIZER_MODEL, &mut self.tokenizer_model);
            ml.get_string(Kv::TOKENIZER_PRE, &mut self.tokenizer_pre);
            ml.get_u32(Kv::TOKENIZER_TOKEN_TYPE_COUNT, &mut self.n_token_types);

            if self.tokenizer_model == "no_vocab" || self.tokenizer_model == "none" {
                self.vtype = VocabType::None;

                // default special tokens
                self.special_bos_id = LLAMA_TOKEN_NULL;
                self.special_eos_id = LLAMA_TOKEN_NULL;
                self.special_unk_id = LLAMA_TOKEN_NULL;
                self.special_sep_id = LLAMA_TOKEN_NULL;
                self.special_pad_id = LLAMA_TOKEN_NULL;
                self.special_mask_id = LLAMA_TOKEN_NULL;
                self.linefeed_id = LLAMA_TOKEN_NULL;

                // read vocab size from metadata
                let mut n_tokens = 0u32;
                if ml.get_u32(Kv::VOCAB_SIZE, &mut n_tokens) {
                    // adding dummy tokens
                    self.id_to_token.resize(n_tokens as usize, TokenData::default());
                }
                return Ok(());
            }

            match self.tokenizer_model.as_str() {
                "llama" => {
                    self.vtype = VocabType::Spm;
                    self.special_bos_id = 1;
                    self.special_eos_id = 2;
                    self.special_unk_id = 0;
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = LLAMA_TOKEN_NULL;
                    self.special_mask_id = LLAMA_TOKEN_NULL;
                }
                "bert" => {
                    self.vtype = VocabType::Wpm;
                    self.special_bos_id = 101;
                    self.special_eos_id = LLAMA_TOKEN_NULL;
                    self.special_unk_id = 100;
                    self.special_sep_id = 102;
                    self.special_pad_id = 0;
                    self.special_mask_id = 103;
                    self.add_sep = true;
                }
                "gpt2" | "hybriddna" | "whitespace" => {
                    self.vtype = VocabType::Bpe;

                    // Kimi-K2 uses custom tokenization without traditional BPE merges
                    let is_kimi_k2 = self.tokenizer_pre == "kimi-k2";

                    // read bpe merges and populate bpe ranks
                    match Self::load_bpe_ranks(ml)? {
                        Some(ranks) => self.bpe_ranks = ranks,
                        None => {
                            if !is_kimi_k2 {
                                return Err(Error::new("cannot find tokenizer merges in model file"));
                            }
                            // Kimi-K2 doesn't need merges, skip
                        }
                    }

                    // default special tokens
                    self.special_bos_id = 11;
                    self.special_eos_id = 11;
                    self.special_unk_id = LLAMA_TOKEN_NULL;
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = LLAMA_TOKEN_NULL;
                    self.special_mask_id = LLAMA_TOKEN_NULL;
                }
                "t5" => {
                    self.vtype = VocabType::Ugm;
                    self.special_bos_id = LLAMA_TOKEN_NULL;
                    self.special_eos_id = 1;
                    self.special_unk_id = 2;
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = 0;
                    self.special_mask_id = LLAMA_TOKEN_NULL;

                    if let Some(data) = ml.get_blob(Kv::TOKENIZER_PRECOMPILED_CHARSMAP) {
                        if data.len() < std::mem::size_of::<u32>() {
                            return Err(Error::new("precompiled_charsmap too small for xcda_blob_size header!"));
                        }
                        let xcda_blob_size = u32::from_le_bytes([
                            data[0], data[1], data[2], data[3],
                        ]) as usize;
                        if xcda_blob_size + std::mem::size_of::<u32>() >= data.len() {
                            return Err(Error::new("Index out of array bounds in precompiled charsmap!"));
                        }
                        self.precompiled_charsmap = data.clone();
                    }
                }
                "rwkv" => {
                    self.vtype = VocabType::Rwkv;
                    self.special_bos_id = LLAMA_TOKEN_NULL;
                    self.special_eos_id = LLAMA_TOKEN_NULL;
                    self.special_unk_id = LLAMA_TOKEN_NULL;
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = LLAMA_TOKEN_NULL;
                }
                "test" => {
                    self.vtype = VocabType::Test;
                    self.special_bos_id = LLAMA_TOKEN_NULL;
                    self.special_eos_id = LLAMA_TOKEN_NULL;
                    self.special_unk_id = LLAMA_TOKEN_NULL;
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = LLAMA_TOKEN_NULL;
                    self.special_mask_id = LLAMA_TOKEN_NULL;
                }
                "plamo2" | "plamo3" => {
                    self.vtype = if self.tokenizer_model == "plamo2" {
                        VocabType::Plamo2
                    } else {
                        VocabType::Plamo3
                    };
                    // PLaMo default special tokens (these will be overridden by model config)
                    self.special_bos_id = 1; // <|plamo:bos|>
                    self.special_eos_id = 2; // <|plamo:eos|>
                    self.special_unk_id = 0; // <|plamo:unk|>
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = 3; // <|plamo:pad|>
                    self.special_mask_id = LLAMA_TOKEN_NULL;
                }
                "gemma4" => {
                    self.vtype = VocabType::Bpe;
                    match Self::load_bpe_ranks(ml)? {
                        Some(ranks) => self.bpe_ranks = ranks,
                        None => return Err(Error::new("cannot find tokenizer merges in model file")),
                    }
                    // default special tokens (to be read from GGUF)
                    self.special_bos_id = LLAMA_TOKEN_NULL;
                    self.special_eos_id = LLAMA_TOKEN_NULL;
                    self.special_unk_id = LLAMA_TOKEN_NULL;
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = LLAMA_TOKEN_NULL;
                    self.special_mask_id = LLAMA_TOKEN_NULL;
                    self.tokenizer_pre = "gemma4".to_string();
                }
                _ => {
                    return Err(Error::new(format!(
                        "unknown tokenizer: '{}'",
                        self.tokenizer_model
                    )));
                }
            }

            // for now, only BPE models have pre-tokenizers
            if self.vtype == VocabType::Bpe {
                self.add_space_prefix = false;
                self.escape_whitespaces = false;
                self.clean_spaces = true;
                if self.tokenizer_pre.is_empty() {
                    self.pre_type = VocabPreType::Default;
                } else {
                    self.pre_type = self.parse_pre_type()?;
                }
            } else if self.vtype == VocabType::Spm {
                self.pre_type = VocabPreType::Default;
                self.add_space_prefix = true;
                self.clean_spaces = false;
                self.add_bos = true;
                self.add_eos = false;
            } else if self.vtype == VocabType::Wpm {
                self.pre_type = VocabPreType::Default;
                self.add_space_prefix = false;
                self.clean_spaces = true;
                self.add_bos = true;
                self.add_eos = false;
                self.add_sep = true;
            } else if self.vtype == VocabType::Ugm {
                self.pre_type = VocabPreType::Default;
                self.add_bos = false;
                self.add_eos = true;
            } else if self.vtype == VocabType::Rwkv {
                self.pre_type = VocabPreType::Default;
                self.add_space_prefix = false;
                self.clean_spaces = false;
                self.add_bos = false;
                self.add_eos = false;
            } else {
                self.pre_type = VocabPreType::Default;
            }

            ml.get_bool(Kv::TOKENIZER_ADD_PREFIX, &mut self.add_space_prefix);
            ml.get_bool(Kv::TOKENIZER_REMOVE_EXTRA_WS, &mut self.remove_extra_whitespaces);
            ml.get_bool(Kv::TOKENIZER_CLEAN_SPACES, &mut self.clean_spaces);
        }

        self.load_tokens(ml)?;
        self.init_tokenizer()?;

        // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
        if self.vtype == VocabType::Spm {
            let id = self.byte_to_token(b'\n');
            if id == LLAMA_TOKEN_NULL {
                self.linefeed_id = self.special_pad_id;
            } else {
                self.linefeed_id = id;
            }
        } else if self.vtype == VocabType::Wpm {
            self.linefeed_id = self.special_pad_id;
        } else {
            let ids = self.tokenize("\n", false, false);
            if ids.is_empty() {
                self.linefeed_id = self.special_pad_id;
            } else {
                self.linefeed_id = ids[0];
            }
        }

        self.load_specials(ml)?;
        Ok(())
    }

    fn parse_pre_type(&mut self) -> Result<VocabPreType> {
        use VocabPreType as P;
        let p = match self.tokenizer_pre.as_str() {
            "" => {
                // missing pre-tokenizer type, using: 'default'
                // GENERATION QUALITY WILL BE DEGRADED!
                P::Default
            }
            "flint" => {
                self.clean_spaces = false;
                P::Flint
            }
            "default" => P::Default,
            "minicpm5" => {
                self.ignore_merges = true;
                self.clean_spaces = false;
                P::Minicpm5
            }
            "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "midm-2.0" | "jina-v5-nano" => {
                self.ignore_merges = true;
                self.add_bos = true;
                P::Llama3
            }
            "lfm2" => {
                self.ignore_merges = true;
                self.add_bos = true;
                self.clean_spaces = false;
                P::Llama3
            }
            "deepseek-llm" => {
                self.clean_spaces = false;
                P::DeepseekLlm
            }
            "deepseek-coder" => {
                self.clean_spaces = false;
                P::DeepseekCoder
            }
            "deepseek-v3" => {
                self.clean_spaces = false;
                P::Deepseek3Llm
            }
            "spark2_5" => {
                self.clean_spaces = false;
                P::Spark2p5
            }
            "youtu" => {
                self.clean_spaces = false;
                self.ignore_merges = true;
                P::Youtu
            }
            "falcon" => P::Falcon,
            "mpt" => P::Mpt,
            "starcoder" => P::Starcoder,
            "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es"
            | "jina-v2-de" | "a.x-4.0" | "mellum" | "modern-bert" => P::Gpt2,
            "jais-2" => P::Jais2,
            "gemma4" | "granite-embed-multi-311m" => {
                self.escape_whitespaces = true;
                self.clean_spaces = false;
                P::Gemma4
            }
            "mmbert" => {
                self.escape_whitespaces = true;
                self.metaspace_prepend_scheme = MetaspacePrependScheme::Always;
                P::Mbert
            }
            "sarvam-moe" => {
                self.escape_whitespaces = true;
                self.clean_spaces = false;
                P::SarvamMoe
            }
            "jina-v1-en" | "jina-v2-code" | "roberta-bpe" => {
                self.add_sep = true;
                P::Gpt2
            }
            "whitespace" => {
                self.normalizer_opts.lowercase = false;
                P::Whitespace
            }
            "refact" => P::Refact,
            "command-r" => {
                self.clean_spaces = false;
                P::CommandR
            }
            "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "domyn-edge" => {
                self.clean_spaces = false;
                P::Qwen2
            }
            "raon" => {
                self.clean_spaces = false;
                P::Raon
            }
            "qwen35" => {
                self.clean_spaces = false;
                P::Qwen35
            }
            "stablelm2" => P::StableLm2,
            "olmo" => P::Olmo,
            "dbrx" => {
                self.clean_spaces = false;
                P::Dbrx
            }
            "smaug-bpe" => P::Smaug,
            "poro-chat" => {
                self.clean_spaces = false;
                P::Poro
            }
            "chatglm-bpe" => {
                self.special_bos_id = LLAMA_TOKEN_NULL;
                self.clean_spaces = false;
                P::ChatGlm4
            }
            "glm4" | "glm5" => {
                self.special_bos_id = LLAMA_TOKEN_NULL;
                self.ignore_merges = true;
                self.clean_spaces = false;
                P::ChatGlm4
            }
            "viking" => {
                self.clean_spaces = false;
                P::Viking
            }
            "jais" => P::Jais,
            "tekken" | "pixtral" => {
                self.clean_spaces = false;
                self.ignore_merges = true;
                self.add_bos = true;
                P::Tekken
            }
            "tekken-num3" => {
                self.clean_spaces = false;
                self.ignore_merges = true;
                self.add_bos = true;
                P::TekkenNum3
            }
            "smollm" => {
                self.clean_spaces = false;
                P::Smallm
            }
            "codeshell" => P::Codeshell,
            "bloom" => P::Bloom,
            "gpt3-finnish" => P::Gpt3Finnish,
            "exaone" => P::Exaone,
            "exaone4" => {
                self.clean_spaces = false;
                P::Gpt2
            }
            "exaone-moe" => P::ExaoneMoe,
            "chameleon" => {
                self.add_bos = true;
                self.clean_spaces = false;
                P::Chameleon
            }
            "minerva-7b" => P::Minerva,
            "domyn-small" => {
                self.escape_whitespaces = true;
                P::DomynSmall
            }
            "megrez" => P::Qwen2,
            "gpt-4o" | "llama4" | "kanana2" | "talkie" => {
                self.clean_spaces = false;
                P::Gpt4o
            }
            "inkling2" => {
                self.special_bos_id = LLAMA_TOKEN_NULL;
                self.clean_spaces = false;
                self.ignore_merges = true;
                P::Gpt4o
            }
            "granite-embed-multi-97m" => {
                self.clean_spaces = false;
                self.ignore_merges = true;
                P::GraniteEmbMulti
            }
            "tiny_aya" | "cohere2moe" => {
                self.clean_spaces = false;
                P::TinyAya
            }
            "superbpe" => {
                self.clean_spaces = false;
                P::SuperBpe
            }
            "trillion" => {
                self.clean_spaces = false;
                P::Trillion
            }
            "granite-docling" => {
                self.clean_spaces = false;
                P::GraniteDocling
            }
            "bailingmoe" | "bailingmoe2" | "llada-moe" => {
                self.clean_spaces = false;
                P::BailingMoe
            }
            "seed-coder" => {
                self.clean_spaces = false;
                P::SeedCoder
            }
            "hunyuan" => {
                self.clean_spaces = false;
                P::Hunyuan
            }
            "hunyuan-dense" => {
                self.clean_spaces = false;
                P::HunyuanDense
            }
            "hy_v4" => {
                self.clean_spaces = false;
                P::HyV4
            }
            "joyai-llm" => {
                self.clean_spaces = false;
                P::JoyaiLlm
            }
            "kimi-k2" => {
                self.clean_spaces = false;
                P::KimiK2
            }
            "ufakzeka" => {
                self.clean_spaces = false;
                P::Ufakzeka
            }
            "grok-2" => {
                self.clean_spaces = false;
                P::Grok2
            }
            "afmoe" => {
                self.clean_spaces = false;
                P::Afmoe
            }
            "laguna" => {
                self.clean_spaces = false;
                P::Laguna
            }
            "minimax-m2" => {
                self.clean_spaces = false;
                P::MinimaxM2
            }
            "solar-open" => {
                self.clean_spaces = false;
                P::SolarOpen
            }
            "mellum2" => {
                self.clean_spaces = false;
                P::Mellum2
            }
            "k2-horizon" => {
                self.clean_spaces = false;
                P::K2Horizon
            }
            "nemotron-3.5-embed" => {
                self.ignore_merges = true;
                self.add_bos = true;
                P::Nemotron35Embd
            }
            "soprano" => P::Soprano,
            "zonos3" => {
                self.clean_spaces = false;
                P::Zonos3
            }
            _ => return Err(Error::new(format!("unknown pre-tokenizer type: '{}'", self.tokenizer_pre))),
        };
        Ok(p)
    }

    fn load_tokens(&mut self, ml: &Metadata) -> Result<()> {
        let token_idx = ml
            .get_string_array(Kv::TOKENIZER_LIST)
            .ok_or_else(|| Error::new("cannot find tokenizer vocab in model file"))?;
        let n_tokens = token_idx.len();

        let scores = ml.get_f32_array(Kv::TOKENIZER_SCORES);
        let iscores = ml.get_i32_array(Kv::TOKENIZER_SCORES);
        if let Some(s) = scores {
            if s.len() < n_tokens {
                return Err(Error::new(format!(
                    "Index out of array bounds for scores ({} < {})",
                    s.len(),
                    n_tokens
                )));
            }
        } else if let Some(s) = iscores {
            if s.len() < n_tokens {
                return Err(Error::new(format!(
                    "Index out of array bounds for scores ({} < {})",
                    s.len(),
                    n_tokens
                )));
            }
        }
        let toktypes = ml.get_i32_array(Kv::TOKENIZER_TOKEN_TYPE);
        if let Some(t) = toktypes {
            if t.len() < n_tokens {
                return Err(Error::new(format!(
                    "Index out of array bounds for toktypes ({} < {})",
                    t.len(),
                    n_tokens
                )));
            }
        }

        self.id_to_token.clear();
        self.id_to_token.resize(n_tokens, TokenData::default());
        self.token_to_id.clear();
        self.token_to_id.reserve(n_tokens);

        for (i, word_ref) in token_idx.iter().enumerate() {
            let mut word = word_ref.clone();
            if word.is_empty() {
                word = format!("[EMPTY_{}]", i);
            }

            self.token_to_id.insert(word.clone(), i as Token);
            self.max_token_len = self.max_token_len.max(word.len() as i32);

            let token_data = &mut self.id_to_token[i];
            token_data.text = word;
            token_data.score = if let Some(s) = scores {
                s[i]
            } else if let Some(s) = iscores {
                s[i] as f32
            } else {
                0.0
            };
            token_data.attr = LLAMA_TOKEN_ATTR_NORMAL;

            if let Some(types) = toktypes {
                //TODO: remove, required until per token attributes are available from GGUF file
                token_data.attr = match types[i] {
                    LLAMA_TOKEN_TYPE_UNKNOWN => LLAMA_TOKEN_ATTR_UNKNOWN,
                    LLAMA_TOKEN_TYPE_UNUSED => LLAMA_TOKEN_ATTR_UNUSED,
                    LLAMA_TOKEN_TYPE_NORMAL => LLAMA_TOKEN_ATTR_NORMAL,
                    LLAMA_TOKEN_TYPE_CONTROL => LLAMA_TOKEN_ATTR_CONTROL,
                    LLAMA_TOKEN_TYPE_USER_DEFINED => LLAMA_TOKEN_ATTR_USER_DEFINED,
                    LLAMA_TOKEN_TYPE_BYTE => LLAMA_TOKEN_ATTR_BYTE,
                    _ => LLAMA_TOKEN_ATTR_UNDEFINED,
                };
            }
        }

        // hybriddna: the marker suffix kept k-mer ids distinct in token_to_id; erase
        // it from id_to_token so the k-mers detokenize to the bare DNA sequence. The
        // k-mers are the block right after <oov>, so only scan from there.
        if self.tokenizer_model == "hybriddna" {
            if let Some(&idx) = self.token_to_id.get("<oov>") {
                for token in self.id_to_token.iter_mut().skip(idx as usize + 1) {
                    if token.text.len() > DNA_KMER_MARKER.len()
                        && token.text.ends_with(DNA_KMER_MARKER)
                    {
                        token.text.truncate(token.text.len() - DNA_KMER_MARKER.len());
                        // tail="pad_n": "CTNNNN" detokenizes to its real bases "CT"
                        // full k-mers are ACGT only, so only N-padded tail tokens end in N
                        if let Some(last_base) = token.text.rfind(|c| c != 'N') {
                            token.text.truncate(last_base + c_len(&token.text[last_base..]));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn init_tokenizer(&mut self) -> Result<()> {
        self.tokenizer = match self.vtype {
            VocabType::Spm => Tokenizer::Spm,
            VocabType::Bpe => {
                let (regex_exprs, byte_encode) = bpe_regex_exprs(self.pre_type);
                Tokenizer::Bpe(BpeData { regex_exprs, byte_encode })
            }
            VocabType::Wpm => Tokenizer::Wpm,
            VocabType::Ugm => Tokenizer::Ugm(self.build_ugm_data()?),
            VocabType::Rwkv => Tokenizer::Rwkv(self.build_rwkv_data()),
            VocabType::Plamo2 | VocabType::Plamo3 => Tokenizer::Plamo2(Plamo2Data::build(
                &self.id_to_token,
                &self.token_to_id,
                self.vtype == VocabType::Plamo3,
            )?),
            VocabType::Test => Tokenizer::Test,
            VocabType::None => return Err(Error::new("unsupported vocab type")),
        };
        Ok(())
    }

    fn build_ugm_data(&self) -> Result<UgmData> {
        let mut data = UgmData {
            min_score: f32::MAX,
            max_score: -f32::MAX,
            ..UgmData::default()
        };

        if !self.precompiled_charsmap.is_empty() {
            let bytes = &self.precompiled_charsmap;
            let xcda_blob_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
            let array_start = std::mem::size_of::<u32>();
            let array_end = array_start + xcda_blob_size;
            if array_end > bytes.len() {
                return Err(Error::new("Index out of array bounds in precompiled charsmap!"));
            }
            data.xcda_array = bytes[array_start..array_end]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            data.prefix_replacements = bytes[array_end..].to_vec();
        }

        for id in 0..self.id_to_token.len() as Token {
            let token_data = &self.id_to_token[id as usize];

            let normal = token_data.attr & LLAMA_TOKEN_ATTR_NORMAL != 0;
            let user_defined = token_data.attr & LLAMA_TOKEN_ATTR_USER_DEFINED != 0;
            let unused = token_data.attr & LLAMA_TOKEN_ATTR_UNUSED != 0;

            if normal {
                data.min_score = data.min_score.min(token_data.score);
                data.max_score = data.max_score.max(token_data.score);
            }

            if normal || user_defined || unused {
                data.token_matcher.insert(token_data.text.as_bytes(), id);
            }

            if user_defined {
                data.user_defined_token_matcher.insert(token_data.text.as_bytes(), id);
            }
        }

        data.unknown_token_score = data.min_score - 10.0;
        Ok(data)
    }

    fn build_rwkv_data(&self) -> RwkvData {
        let mut data = RwkvData::default();
        // RWKV supports arbitrary byte tokens, but the vocab struct only supports string tokens.
        // For now, we decode the vocab here into the lookup we'll use for tokenization.
        for (id, token) in self.id_to_token.iter().enumerate() {
            let text = llama_unescape_rwkv_token(&token.text);
            data.token_matcher.insert(&text, id as Token);
        }
        data
    }
}

/// Byte length of the first UTF-8 character in `s`.
fn c_len(s: &str) -> usize {
    s.chars().next().map(|c| c.len_utf8()).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// special tokens, EOG list, caches (part of impl::load)
// ---------------------------------------------------------------------------

const EOT_TEXTS: &[&str] = &[
    "<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>",
    "<|end_of_text|>", // granite
    "<EOT>", "_<EOT>", "[EOT]", // Kimi-K2
    "<｜end▁of▁sentence｜>", // DeepSeek
    "<end_of_utterance>", // smoldocling
];

const EOM_TEXTS: &[&str] = &["<|eom_id|>"];

const FIM_PRE_TEXTS: &[&str] = &[
    "<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<｜fim▁begin｜>", "<PRE>",
    "▁<PRE>", // CodeLlama
    "<|code_prefix|>", "<|prefix|>", "<|plamo:fim_prefix|>", // PLaMo-3
];

const FIM_SUF_TEXTS: &[&str] = &[
    "<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>", "<｜fim▁hole｜>", "<SUF>",
    "▁<SUF>", // CodeLlama
    "<|code_suffix|>", "<|suffix|>", "<|plamo:fim_suffix|>", // PLaMo-3
];

const FIM_MID_TEXTS: &[&str] = &[
    "<|fim_middle|>", "<fim-middle>", "<fim_middle>", "<｜fim▁end｜>", "<MID>",
    "▁<MID>", // CodeLlama
    "<|code_middle|>", "<|middle|>", "<|plamo:fim_middle|>", // PLaMo-3
];

const FIM_PAD_TEXTS: &[&str] = &[
    "<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]", // Kimi-K2
];

const FIM_REP_TEXTS: &[&str] = &[
    "<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>", // Granite
];

const FIM_SEP_TEXTS: &[&str] = &["<|file_sep|>", "<|plamo:file_separator|>"]; // PLaMo-2/3

const EOG_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|eot|>", // muse-spark
    "<|im_end|>",
    "<|end|>",
    "<|return|>", // o200k_harmony
    "<|call|>",   // o200k_harmony
    "<|flush|>",  // solar-open
    "<|calls|>",  // solar-open
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>", // paddleocr
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]", // Kimi-K2
    "[EOS]", // Kimi-K2
    "<|end_of_text|>",
    "<end_of_utterance>", // smoldocling
    "<eos>",              // gemma4
    "<turn|>",            // gemma4
    "<|tool_response>",   // gemma4
    "<｜end▁of▁sentence｜>", // deepseek-ocr
    "[e~[",               // minimax-m2/m3
    "<|ifm|im_end|>",     // k2-horizon
];

impl Impl {
    fn load_specials(&mut self, ml: &Metadata) -> Result<()> {
        // special tokens
        let n = self.id_to_token.len();
        let mut load_special = |key: Kv, field: &mut Token| {
            // defaults out of vocab range are disabled
            if *field >= 0 && *field as usize >= n {
                *field = LLAMA_TOKEN_NULL;
            }
            let mut new_id = 0u32;
            if ml.get_u32(key, &mut new_id) {
                if (new_id as usize) >= n {
                    // bad special token: using default id
                } else {
                    *field = new_id as Token;
                }
            }
        };

        load_special(Kv::TOKENIZER_BOS_ID, &mut self.special_bos_id);
        load_special(Kv::TOKENIZER_EOS_ID, &mut self.special_eos_id);
        load_special(Kv::TOKENIZER_EOT_ID, &mut self.special_eot_id);
        load_special(Kv::TOKENIZER_EOM_ID, &mut self.special_eom_id);
        load_special(Kv::TOKENIZER_UNK_ID, &mut self.special_unk_id);
        load_special(Kv::TOKENIZER_SEP_ID, &mut self.special_sep_id);
        load_special(Kv::TOKENIZER_PAD_ID, &mut self.special_pad_id);
        load_special(Kv::TOKENIZER_MASK_ID, &mut self.special_mask_id);
        load_special(Kv::TOKENIZER_FIM_PRE_ID, &mut self.special_fim_pre_id);
        load_special(Kv::TOKENIZER_FIM_SUF_ID, &mut self.special_fim_suf_id);
        load_special(Kv::TOKENIZER_FIM_MID_ID, &mut self.special_fim_mid_id);
        load_special(Kv::TOKENIZER_FIM_PAD_ID, &mut self.special_fim_pad_id);
        load_special(Kv::TOKENIZER_FIM_REP_ID, &mut self.special_fim_rep_id);
        load_special(Kv::TOKENIZER_FIM_SEP_ID, &mut self.special_fim_sep_id);

        // deprecated
        load_special(Kv::TOKENIZER_PREFIX_ID, &mut self.special_fim_pre_id);
        load_special(Kv::TOKENIZER_SUFFIX_ID, &mut self.special_fim_suf_id);
        load_special(Kv::TOKENIZER_MIDDLE_ID, &mut self.special_fim_mid_id);
        drop(load_special);

        // Handle add_bos, add_eos and add_sep
        {
            let mut temp = true;
            if ml.get_bool(Kv::TOKENIZER_ADD_BOS, &mut temp) {
                self.add_bos = temp;
            }
            temp = true;
            if ml.get_bool(Kv::TOKENIZER_ADD_EOS, &mut temp) {
                self.add_eos = temp;
            }
            temp = true;
            if ml.get_bool(Kv::TOKENIZER_ADD_SEP, &mut temp) {
                self.add_sep = temp;
            }
            temp = true;
            if ml.get_bool(Kv::TOKENIZER_DNA_AUTO_TAGS, &mut temp) {
                self.dna_auto_tags = temp;
            }

            // workaround for Gemma 4
            // ref: https://github.com/ggml-org/llama.cpp/pull/21500
            if self.pre_type == VocabPreType::Gemma4 && !self.add_bos {
                self.add_bos = true;
            }
        }

        // Tokenizer normalizer options
        ml.get_bool(Kv::TOKENIZER_NORMALIZER_NFC, &mut self.normalizer_opts.nfc);
        ml.get_bool(Kv::TOKENIZER_NORMALIZER_LOWERCASE, &mut self.normalizer_opts.lowercase);
        self.normalizer_opts.strip_accents = self.normalizer_opts.lowercase;
        ml.get_bool(Kv::TOKENIZER_NORMALIZER_STRIP_ACCENTS, &mut self.normalizer_opts.strip_accents);

        {
            let mut scheme = String::new();
            if ml.get_string(Kv::TOKENIZER_METASPACE_PREPEND_SCHEME, &mut scheme) {
                self.metaspace_prepend_scheme = match scheme.as_str() {
                    "never" => MetaspacePrependScheme::Never,
                    "first" => MetaspacePrependScheme::First,
                    "always" => MetaspacePrependScheme::Always,
                    _ => return Err(Error::new(format!("unknown metaspace prepend_scheme: '{}'", scheme))),
                };
            }
        }

        // suppress tokens
        if let Some(data) = ml.get_i32_array(Kv::TOKENIZER_SUPPRESS_TOKENS) {
            self.suppress_tokens.reserve(data.len());
            // drop out-of-range ids
            for &id in data {
                if id >= 0 && (id as usize) < self.id_to_token.len() {
                    self.suppress_tokens.push(id);
                }
            }
        }

        // auto-detect special tokens by text
        // TODO: convert scripts should provide these tokens through the KV metadata LLM_KV_TOKENIZER_...
        //       for now, we apply this workaround to find the tokens based on their text
        let pairs: Vec<(String, Token)> = self
            .token_to_id
            .iter()
            .map(|(text, id)| (text.clone(), *id))
            .collect();

        for (text, tid) in &pairs {
            if *tid < 0 || *tid as usize >= self.id_to_token.len() {
                continue;
            }
            let attr = &mut self.id_to_token[*tid as usize].attr;

            // find EOT token: "<|eot_id|>", "<|im_end|>", "<end_of_turn>", etc.
            if self.special_eot_id == LLAMA_TOKEN_NULL && EOT_TEXTS.contains(&text.as_str()) {
                self.special_eot_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find EOM token: "<|eom_id|>"
            if self.special_eom_id == LLAMA_TOKEN_NULL && EOM_TEXTS.contains(&text.as_str()) {
                self.special_eom_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find FIM_PRE token: "<|fim_prefix|>", "<fim-prefix>", "<PRE>", etc.
            if self.special_fim_pre_id == LLAMA_TOKEN_NULL && FIM_PRE_TEXTS.contains(&text.as_str()) {
                self.special_fim_pre_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find FIM_SUF token: "<|fim_suffix|>", "<fim-suffix>", "<SUF>", etc.
            if self.special_fim_suf_id == LLAMA_TOKEN_NULL && FIM_SUF_TEXTS.contains(&text.as_str()) {
                self.special_fim_suf_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find FIM_MID token: "<|fim_middle|>", "<fim-middle>", "<MID>", etc.
            if self.special_fim_mid_id == LLAMA_TOKEN_NULL && FIM_MID_TEXTS.contains(&text.as_str()) {
                self.special_fim_mid_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find FIM_PAD token: "<|fim_pad|>", "<fim-pad>", "<PAD>", etc.
            if self.special_fim_pad_id == LLAMA_TOKEN_NULL && FIM_PAD_TEXTS.contains(&text.as_str()) {
                self.special_fim_pad_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find FIM_REP token: "<|fim_repo|>", "<fim-repo>", "<REP>", etc.
            if self.special_fim_rep_id == LLAMA_TOKEN_NULL && FIM_REP_TEXTS.contains(&text.as_str()) {
                self.special_fim_rep_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find FIM_SEP token: "<|file_sep|>"
            if self.special_fim_sep_id == LLAMA_TOKEN_NULL && FIM_SEP_TEXTS.contains(&text.as_str()) {
                self.special_fim_sep_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }
        }

        // auto-detect unused tokens: e.g. control tokens with the word "unused"
        // ideally, these tokens should be marked as unused during conversion
        {
            for (text, tid) in &pairs {
                if *tid < 0 || *tid as usize >= self.id_to_token.len() {
                    continue;
                }
                let attr = &mut self.id_to_token[*tid as usize].attr;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    continue;
                }
                if *attr & LLAMA_TOKEN_ATTR_UNUSED == 0 && text.contains("unused") {
                    *attr |= LLAMA_TOKEN_ATTR_UNUSED;
                }
            }
        }

        // maintain a list of tokens that cause end-of-generation
        // this is currently determined based on the token text, which is obviously not ideal
        // ref: https://github.com/ggml-org/llama.cpp/issues/9606
        self.special_eog_ids.clear();
        if self.special_fim_pad_id != LLAMA_TOKEN_NULL {
            self.special_eog_ids.insert(self.special_fim_pad_id);
        }
        if self.special_fim_rep_id != LLAMA_TOKEN_NULL {
            self.special_eog_ids.insert(self.special_fim_rep_id);
        }
        if self.special_fim_sep_id != LLAMA_TOKEN_NULL {
            self.special_eog_ids.insert(self.special_fim_sep_id);
        }

        for (text, tid) in &pairs {
            if *tid < 0 || *tid as usize >= self.id_to_token.len() {
                continue;
            }
            let attr = &mut self.id_to_token[*tid as usize].attr;
            let is_eog_text = EOG_TEXTS.contains(&text.as_str())
                || (self.pre_type == VocabPreType::Flint
                    && (text == "<|user|>" || text == "<|observation|>"));
            if is_eog_text {
                self.special_eog_ids.insert(*tid);
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }
        }

        // @ngxson : quick hack for gpt-oss, always render these tokens
        for (text, tid) in &pairs {
            if *tid < 0 || *tid as usize >= self.id_to_token.len() {
                continue;
            }
            if text == "<|channel|>" || text == "<|message|>" || text == "<|start|>" || text == "<|constrain|>" {
                self.id_to_token[*tid as usize].attr = LLAMA_TOKEN_ATTR_USER_DEFINED;
            }
        }

        if self.pre_type == VocabPreType::Soprano {
            self.special_eog_ids.insert(self.special_eos_id);
        }

        // sanity checks
        if self.special_eos_id != LLAMA_TOKEN_NULL {
            self.special_eog_ids.insert(self.special_eos_id);
        }
        if self.special_eot_id != LLAMA_TOKEN_NULL {
            self.special_eog_ids.insert(self.special_eot_id);
        }
        if self.special_eom_id != LLAMA_TOKEN_NULL {
            self.special_eog_ids.insert(self.special_eom_id);
        }

        // TODO: workaround for o200k_harmony and solar-open tokenizer: the "<|end|>" token should not be EOG
        //       we don't have a good way to detect this, so for now, if we have "<|return|>" and "<|call|>" tokens ("<|calls|>" and "<|flush|>" for solar-open),
        //       we remove the "<|end|>" token from the EOG list
        {
            let mut has_return = false;
            let mut has_call = false;
            let mut has_end = false;
            let mut has_flush = false;
            let mut end_id = LLAMA_TOKEN_NULL;

            let eog_ids: Vec<Token> = self.special_eog_ids.iter().copied().collect();
            for tid in eog_ids {
                if tid < 0 || tid as usize >= self.id_to_token.len() {
                    continue;
                }
                let text = self.id_to_token[tid as usize].text.clone();
                if text == "<|return|>" {
                    has_return = true;
                } else if text == "<|call|>" || text == "<|calls|>" {
                    has_call = true;
                } else if text == "<|flush|>" {
                    has_flush = true;
                } else if text == "<|end|>" {
                    has_end = true;
                    end_id = tid;
                }
            }

            if (has_return && has_call && has_end) || (has_call && has_flush && has_end) {
                self.special_eog_ids.remove(&end_id);
                self.id_to_token[end_id as usize].attr = LLAMA_TOKEN_ATTR_USER_DEFINED;
            }
        }

        // gemma4, plamo and beam have a normal </s> token, unlike paddleocr
        {
            // beam's tool call marker is not an EOG token, so it is looked up in the whole vocab
            let mut has_normal_s_marker = self.token_to_id.contains_key("<rfl:parameter name=\"");
            let mut has_s = false;
            let mut s_id = LLAMA_TOKEN_NULL;

            let eog_ids: Vec<Token> = self.special_eog_ids.iter().copied().collect();
            for tid in eog_ids {
                if tid < 0 || tid as usize >= self.id_to_token.len() {
                    continue;
                }
                let text = self.id_to_token[tid as usize].text.clone();
                if text == "<|tool_response>" || text == "<|plamo:eos|>" {
                    has_normal_s_marker = true;
                } else if text == "</s>" {
                    has_s = true;
                    s_id = tid;
                }
            }

            if has_normal_s_marker && has_s {
                self.special_eog_ids.remove(&s_id);
                self.id_to_token[s_id as usize].attr = LLAMA_TOKEN_ATTR_NORMAL;
            }
        }

        // build special tokens cache
        {
            for id in 0..self.id_to_token.len() as Token {
                let attr = self.id_to_token[id as usize].attr;
                if attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN) != 0 {
                    self.cache_special_tokens.push(id);
                }
            }

            let id_to_token = &self.id_to_token;
            self.cache_special_tokens.sort_by(|&a, &b| {
                id_to_token[b as usize].text.len().cmp(&id_to_token[a as usize].text.len())
            });
        }

        // build token to piece cache
        {
            let mut cache = vec![String::new(); self.id_to_token.len()];
            for id in 0..self.id_to_token.len() as Token {
                cache[id as usize] = self.token_to_piece_for_cache(id, true);
            }
            self.cache_token_to_piece = cache;
        }

        // Handle per token attributes
        //NOTE: Each model customizes per token attributes.
        //NOTE: Per token attributes are missing from the GGUF file.
        //TODO: Extract attributes from GGUF file.
        {
            fn contains_any(str: &str, substrs: &[&str]) -> bool {
                substrs.iter().any(|s| str.contains(s))
            }

            let mut model_name = String::new();
            let mut tokenizer_pre = String::new();
            let mut general_arch = String::new();
            ml.get_string(Kv::GENERAL_NAME, &mut model_name);
            ml.get_string(Kv::TOKENIZER_PRE, &mut tokenizer_pre);
            ml.get_string(Kv::GENERAL_ARCHITECTURE, &mut general_arch);

            // model name to lowercase
            model_name.make_ascii_lowercase();

            // set attributes by model/tokenizer/architecture name
            if contains_any(&tokenizer_pre, &["jina-v2-de", "jina-v2-es", "jina-v2-code", "mmbert"])
                || contains_any(&general_arch, &["nomic-bert-moe", "jina-bert-v3"])
            {
                let id = self.token_to_id.get("<mask>").copied();
                if let Some(id) = id {
                    self.set_tokenid_attr(id, LLAMA_TOKEN_ATTR_LSTRIP, true);
                }
            } else if contains_any(&model_name, &["phi-3", "phi3"]) {
                let specials = self.cache_special_tokens.clone();
                for id in specials {
                    self.set_tokenid_attr(id, LLAMA_TOKEN_ATTR_RSTRIP, true);
                }
                for token in ["</s>"] {
                    if let Some(&id) = self.token_to_id.get(token) {
                        self.set_tokenid_attr(id, LLAMA_TOKEN_ATTR_RSTRIP, true);
                    }
                }
                for token in ["<unk>", "<s>", "<|endoftext|>"] {
                    if let Some(&id) = self.token_to_id.get(token) {
                        self.set_tokenid_attr(id, LLAMA_TOKEN_ATTR_RSTRIP, false);
                    }
                }
            } else if contains_any(&model_name, &["modern-bert"]) {
                if let Some(&id) = self.token_to_id.get("[MASK]") {
                    self.set_tokenid_attr(id, LLAMA_TOKEN_ATTR_LSTRIP, true);
                }
            }
        }

        Ok(())
    }

    fn set_tokenid_attr(&mut self, id: Token, attr: LlamaTokenAttr, value: bool) {
        if id < 0 || id as usize >= self.id_to_token.len() {
            return;
        }
        let current = self.id_to_token[id as usize].attr;
        self.id_to_token[id as usize].attr = if value {
            current | attr
        } else {
            current & !attr
        };
    }
}

// ---------------------------------------------------------------------------
// (de-) tokenize
// ---------------------------------------------------------------------------

enum BpeKind<'a> {
    Plain(BpeSession<'a>),
    Dna(HybridDnaSession<'a>),
    Whitespace(WhitespaceSession<'a>),
}

struct AnyBpeSession<'a> {
    vocab: &'a Impl,
    kind: BpeKind<'a>,
}

impl<'a> AnyBpeSession<'a> {
    fn append_bos(&self, output: &mut Vec<Token>) -> bool {
        if self.vocab.get_add_bos() {
            output.push(self.vocab.token_bos());
            return true;
        }
        false
    }

    fn append_eos(&self, output: &mut Vec<Token>) -> bool {
        if self.vocab.get_add_eos() {
            output.push(self.vocab.token_eos());
            return true;
        }
        false
    }

    fn check_double_bos_eos(&self, output: &[Token]) {
        let _ = self.vocab.get_add_bos() && output.get(1) == Some(&self.vocab.token_bos());
        let _ = self.vocab.get_add_eos()
            && output.len() >= 2
            && output.get(output.len() - 2) == Some(&self.vocab.token_eos());
    }

    fn tokenize(&mut self, text: &str, output: &mut Vec<Token>) {
        match &mut self.kind {
            BpeKind::Plain(session) => session.tokenize(text, output),
            BpeKind::Dna(session) => session.tokenize(text, output),
            BpeKind::Whitespace(session) => session.tokenize(text, output),
        }
    }
}

fn llama_escape_whitespace(text: &mut String) {
    replace_all(text, " ", ESCAPED_SPACE);
}

fn llama_unescape_whitespace(word: &mut String) {
    replace_all(word, ESCAPED_SPACE, " ");
}

fn llama_decode_text(text: &str) -> String {
    let mut decoded: Vec<u8> = Vec::with_capacity(text.len());
    let cpts = unicode_cpts_from_utf8(text);
    for cpt in cpts {
        match unicode_cpt_to_byte(cpt) {
            Ok(byte) => decoded.push(byte),
            Err(_) => {
                decoded.extend_from_slice(b"[UNK_BYTE_0x");
                for &b in unicode_cpt_to_utf8(cpt).as_bytes() {
                    decoded.extend_from_slice(format!("{:02x}", b).as_bytes());
                }
                decoded.extend_from_slice(text.as_bytes());
                decoded.push(b']');
            }
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

fn try_copy(src: &[u8], buf: &mut [u8], lstrip: i32) -> i32 {
    let mut bytes = src;
    let mut n = 0i32;
    while n < lstrip && !bytes.is_empty() && bytes[0] == b' ' {
        bytes = &bytes[1..];
        n += 1;
    }
    if buf.len() < bytes.len() {
        return -(bytes.len() as i32);
    }
    buf[..bytes.len()].copy_from_slice(bytes);
    bytes.len() as i32
}

impl Impl {
    fn tokenizer_st_partition(&self, buffer: &mut Vec<Fragment<'_>>, parse_special: bool) {
        // for each special token
        for special_id in self.cache_special_tokens.clone() {
            let (text, attr) = {
                let data = &self.id_to_token[special_id as usize];
                (data.text.clone(), data.attr)
            };
            if text.is_empty() {
                continue;
            }

            if !parse_special && attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN) != 0 {
                // Ignore control and unknown tokens when parse_special == false
                // User-defined tokens are still pre-tokenized before everything else
                // ref: https://github.com/huggingface/tokenizers/blob/fdd26ba9a3f0c133427aab0423888cbde91362d7/tokenizers/src/tokenizer/mod.rs#L726
                // This is mostly relevant for neox-style tokenizers (mpt, olmo, stablelm, etc.)
                continue;
            }

            let mut it = 0usize;
            while it < buffer.len() {
                // if a fragment is text (not yet processed)
                let (raw_text, base_offset, base_length) = match &buffer[it] {
                    Fragment::RawText { raw_text, offset, length } => (*raw_text, *offset, *length),
                    Fragment::Token(_) => {
                        it += 1;
                        continue;
                    }
                };

                let mut raw_text_base_offset = base_offset;
                let mut raw_text_base_length = base_length;

                // loop over the text
                loop {
                    // find the first occurrence of a given special token in this fragment
                    // passing offset argument only limits the "search area" but match
                    // coordinates are still relative to the source full raw_text
                    let end = (raw_text_base_offset + raw_text_base_length).min(raw_text.len());
                    let from = raw_text_base_offset.min(end);
                    let rel = match raw_text[from..end].find(text.as_str()) {
                        Some(rel) => rel,
                        None => break, // no occurrences found, stop for this special token
                    };
                    let match_pos = from + rel;
                    let source = it;

                    // if match is further than base offset
                    // then we have some text to the left of it
                    if match_pos > raw_text_base_offset {
                        // left
                        let left_reminder_offset = raw_text_base_offset;
                        let mut left_reminder_length = match_pos - raw_text_base_offset;

                        if attr & LLAMA_TOKEN_ATTR_LSTRIP != 0 {
                            while left_reminder_length > 0
                                && raw_text.as_bytes()[left_reminder_offset + left_reminder_length - 1]
                                    .is_ascii_whitespace()
                            {
                                left_reminder_length -= 1;
                            }
                        }

                        if left_reminder_length > 0 {
                            buffer.insert(
                                it + 1,
                                Fragment::RawText {
                                    raw_text,
                                    offset: left_reminder_offset,
                                    length: left_reminder_length,
                                },
                            );
                            it += 1;
                        }
                    }

                    // special token
                    buffer.insert(it + 1, Fragment::Token(special_id));
                    it += 1;

                    // right
                    if match_pos + text.len() < end {
                        let mut right_reminder_offset = match_pos + text.len();
                        let mut right_reminder_length =
                            raw_text_base_length - ((match_pos - raw_text_base_offset) + text.len());

                        if attr & LLAMA_TOKEN_ATTR_RSTRIP != 0 {
                            while right_reminder_length > 0
                                && raw_text.as_bytes()[right_reminder_offset].is_ascii_whitespace()
                            {
                                right_reminder_offset += 1;
                                right_reminder_length -= 1;
                            }
                        }

                        if right_reminder_length > 0 {
                            buffer.insert(
                                it + 1,
                                Fragment::RawText {
                                    raw_text,
                                    offset: right_reminder_offset,
                                    length: right_reminder_length,
                                },
                            );
                            it += 1;
                        }

                        buffer.remove(source);
                        it = it.saturating_sub(1);

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;
                    } else {
                        buffer.remove(source);
                        it = it.saturating_sub(1);
                        break;
                    }
                }
                it += 1;
            }
        }
    }

    fn tokenize(&self, raw_text: &str, add_special: bool, parse_special: bool) -> Vec<Token> {
        let mut output = Vec::new();
        let mut fragment_buffer: Vec<Fragment<'_>> = Vec::new();

        if !raw_text.is_empty() {
            fragment_buffer.push(Fragment::RawText { raw_text, offset: 0, length: raw_text.len() });
            self.tokenizer_st_partition(&mut fragment_buffer, parse_special);
        }

        let prepend_first = self.metaspace_prepend_scheme == MetaspacePrependScheme::First;

        match self.vtype {
            VocabType::Spm => {
                // OG tokenizer behavior:
                //
                // tokenizer.encode('', add_special_tokens=True)  returns [1]
                // tokenizer.encode('', add_special_tokens=False) returns []
                let mut prepend_space = self.add_space_prefix;

                if add_special && self.add_bos {
                    output.push(self.special_bos_id);
                }

                for (idx, fragment) in fragment_buffer.iter().enumerate() {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            let mut text = raw_text[*offset..*offset + *length].to_string();
                            llama_escape_whitespace(&mut text);

                            if prepend_space && (!prepend_first || !text.starts_with(ESCAPED_SPACE)) {
                                text.insert_str(0, ESCAPED_SPACE);
                            }

                            SpmSession::new(self).tokenize(&text, &mut output);
                            prepend_space = false;
                        }
                        Fragment::Token(id) => {
                            output.push(*id);
                            prepend_space = self.add_space_prefix && !prepend_first;
                        }
                    }
                    let _ = idx;
                }

                if add_special && self.add_eos {
                    output.push(self.special_eos_id);
                }
            }
            VocabType::Bpe => {
                let tokenizer = self.tokenizer.bpe();
                let mut session = if self.tokenizer_model == "hybriddna" {
                    AnyBpeSession {
                        vocab: self,
                        kind: BpeKind::Dna(HybridDnaSession::new(self, tokenizer)),
                    }
                } else if self.tokenizer_model == "whitespace" {
                    AnyBpeSession {
                        vocab: self,
                        kind: BpeKind::Whitespace(WhitespaceSession {
                            bpe: BpeSession::new(self, tokenizer),
                            vocab: self,
                        }),
                    }
                } else {
                    AnyBpeSession {
                        vocab: self,
                        kind: BpeKind::Plain(BpeSession::new(self, tokenizer)),
                    }
                };

                if add_special {
                    session.append_bos(&mut output);
                }
                for (idx, fragment) in fragment_buffer.iter().enumerate() {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            let mut text = raw_text[*offset..*offset + *length].to_string();

                            if self.escape_whitespaces {
                                llama_escape_whitespace(&mut text);
                            }

                            // Metaspace prepend_scheme "first": add U+2581 only at the start of the input
                            if prepend_first && idx == 0 && !text.starts_with(ESCAPED_SPACE) {
                                text.insert_str(0, ESCAPED_SPACE);
                            }

                            session.tokenize(&text, &mut output);
                        }
                        Fragment::Token(id) => {
                            BpeSession::append(*id, &mut output);
                        }
                    }
                }

                if add_special {
                    session.append_eos(&mut output);
                    session.check_double_bos_eos(&output);
                }
            }
            VocabType::Wpm => {
                if add_special {
                    output.push(self.special_bos_id);
                }

                let session = WpmSession { vocab: self };
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            session.tokenize(&raw_text[*offset..*offset + *length], &mut output);
                        }
                        Fragment::Token(id) => output.push(*id),
                    }
                }

                if add_special {
                    output.push(self.special_sep_id);
                }
            }
            VocabType::Ugm => {
                if add_special && self.add_bos {
                    output.push(self.special_bos_id);
                }
                // TODO: honor metaspace_prepend_scheme
                let session = UgmSession { vocab: self, tokenizer: self.tokenizer.ugm() };
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            session.tokenize(&raw_text[*offset..*offset + *length], &mut output);
                        }
                        Fragment::Token(id) => output.push(*id),
                    }
                }

                if add_special && self.add_eos {
                    output.push(self.special_eos_id);
                }
            }
            VocabType::Rwkv => {
                let session = RwkvSession { vocab: self, tokenizer: self.tokenizer.rwkv() };
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            session.tokenize(&raw_text[*offset..*offset + *length], &mut output);
                        }
                        Fragment::Token(id) => output.push(*id),
                    }
                }
            }
            VocabType::Plamo2 | VocabType::Plamo3 => {
                if add_special && self.add_bos {
                    output.push(self.special_bos_id);
                }
                let session = Plamo2Session { tokenizer: self.tokenizer.plamo2() };
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            session.tokenize(&raw_text[*offset..*offset + *length], &mut output);
                        }
                        Fragment::Token(id) => output.push(*id),
                    }
                }

                if add_special && self.add_eos {
                    output.push(self.special_eos_id);
                }
            }
            VocabType::Test => {
                const CHUNK_SIZE: usize = 5;
                let n_vocab = self.n_tokens().max(1) as u64;

                let mut n_tokens = 0usize;
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { length, .. } => {
                            n_tokens += (length + CHUNK_SIZE - 1) / CHUNK_SIZE;
                        }
                        Fragment::Token(_) => n_tokens += 1,
                    }
                }
                output.reserve(n_tokens);

                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            let bytes = raw_text.as_bytes();
                            let begin = *offset;
                            let end = begin + *length;
                            let mut pos = begin;
                            while pos < end {
                                let n = CHUNK_SIZE.min(end - pos);
                                let mut hash = 0u64;
                                for i in 0..n {
                                    hash = hash * 31 + bytes[pos + i] as u64;
                                }
                                output.push((hash % n_vocab) as Token);
                                pos += n;
                            }
                        }
                        Fragment::Token(id) => output.push(*id),
                    }
                }
            }
            VocabType::None => {
                // fatal error in the original; nothing to tokenize without a vocab
            }
        }

        output
    }

    fn token_to_byte(&self, id: Token) -> u8 {
        if id < 0 || id as usize >= self.id_to_token.len() {
            return 0;
        }
        let text = &self.id_to_token[id as usize].text;
        if text.len() < 5 {
            return 0;
        }
        parse_hex_byte(&text[3..5])
    }

    fn token_to_piece_for_cache(&self, token: Token, special: bool) -> String {
        let mut buf = vec![0u8; 64];
        let n_chars = self.token_to_piece(token, &mut buf, 0, special);
        if n_chars < 0 {
            buf.resize((-n_chars) as usize, 0);
            let _check = self.token_to_piece(token, &mut buf, 0, special);
        } else {
            buf.resize(n_chars as usize, 0);
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    // does not write a null terminator to buf
    fn token_to_piece(&self, token: Token, buf: &mut [u8], lstrip: i32, special: bool) -> i32 {
        // ref: https://github.com/ggml-org/llama.cpp/pull/7587#discussion_r1620983843
        let attr_special = LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_CONTROL;
        let attr = if token >= 0 && (token as usize) < self.id_to_token.len() {
            self.token_get_attr(token)
        } else {
            0
        };
        if !special && attr & attr_special != 0 {
            return 0;
        }

        // if we have a cache - use it
        if !self.cache_token_to_piece.is_empty() {
            if let Some(cached) = self.cache_token_to_piece.get(token as usize) {
                return try_copy(cached.as_bytes(), buf, lstrip);
            }
            return 0;
        }

        if token < 0 || token as usize >= self.id_to_token.len() {
            return 0;
        }
        let token_text = &self.id_to_token[token as usize].text;

        let mut piece: Vec<u8>;
        let mut apply_lstrip = true;

        match self.vtype {
            VocabType::Wpm | VocabType::Spm | VocabType::Ugm => {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED) != 0 {
                    piece = token_text.as_bytes().to_vec();
                } else if attr & LLAMA_TOKEN_ATTR_NORMAL != 0 {
                    let mut result = token_text.clone();
                    llama_unescape_whitespace(&mut result);
                    piece = result.into_bytes();
                } else if attr & LLAMA_TOKEN_ATTR_BYTE != 0 {
                    piece = vec![self.token_to_byte(token)];
                } else {
                    piece = Vec::new();
                }
            }
            VocabType::Bpe => {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED) != 0 {
                    piece = token_text.as_bytes().to_vec();
                } else if attr & LLAMA_TOKEN_ATTR_NORMAL != 0 {
                    if self.pre_type == VocabPreType::Soprano {
                        piece = token_text.as_bytes().to_vec();
                    } else if self.escape_whitespaces {
                        // SPM-style BPE: tokens contain U+2581 for spaces
                        let mut result = token_text.clone();
                        llama_unescape_whitespace(&mut result);
                        piece = result.into_bytes();
                    } else {
                        piece = llama_decode_text(token_text).into_bytes();
                    }
                } else if attr & LLAMA_TOKEN_ATTR_BYTE != 0 {
                    piece = vec![self.token_to_byte(token)];
                } else {
                    piece = Vec::new();
                }
            }
            VocabType::Rwkv => {
                piece = llama_unescape_rwkv_token(token_text);
                apply_lstrip = false;
            }
            VocabType::Test => {
                // tokens -> text: simply stringify the token id in hex
                piece = format!("{:x}", token).into_bytes();
            }
            VocabType::Plamo2 | VocabType::Plamo3 => {
                // PLaMo uses similar token handling as BPE/SPM
                let bytes = token_text.as_bytes();
                if attr & LLAMA_TOKEN_ATTR_BYTE != 0
                    && bytes.len() == 6
                    && &bytes[..3] == b"<0x"
                    && bytes[5] == b'>'
                    && bytes[3].is_ascii_hexdigit()
                    && bytes[4].is_ascii_hexdigit()
                {
                    piece = vec![parse_hex_byte(&token_text[3..5])];
                } else {
                    // Normal token - just copy the text
                    piece = token_text.as_bytes().to_vec();
                }
            }
            VocabType::None => piece = Vec::new(),
        }

        if apply_lstrip {
            try_copy(&piece, buf, lstrip)
        } else {
            if piece.len() > buf.len() {
                return -(piece.len() as i32);
            }
            buf[..piece.len()].copy_from_slice(&piece);
            piece.len() as i32
        }
    }

    fn detokenize(
        &self,
        tokens: &[Token],
        text: &mut [u8],
        remove_special: bool,
        unparse_special: bool,
    ) -> i32 {
        if self.vtype == VocabType::None {
            return 0;
        }

        let mut n_tokens = tokens.len();
        let mut skip = 0usize;

        // remove the leading space
        let mut remove_space = self.add_space_prefix;

        if remove_special && self.add_bos {
            if n_tokens > 0 && tokens[skip] == self.special_bos_id {
                remove_space = false;
                n_tokens -= 1;
                skip += 1;
            }
        }

        if remove_special && self.add_eos {
            if n_tokens > 0 && tokens[skip + n_tokens - 1] == self.special_eos_id {
                n_tokens -= 1;
            }
        }

        // Metaspace prepend_scheme "first": remove the leading space of the first
        // rendered piece, so a skipped BOS does not consume it
        // TODO: honor metaspace_prepend_scheme for the other vocab types
        let mut remove_first_space = (self.vtype == VocabType::Bpe || self.vtype == VocabType::Spm)
            && self.metaspace_prepend_scheme == MetaspacePrependScheme::First;

        let mut rendered: Vec<u8> = Vec::new();
        let mut piece_buf = vec![0u8; 4096];

        for i in 0..n_tokens {
            let token = tokens[skip + i];
            let n_chars = self.token_to_piece(
                token,
                &mut piece_buf,
                (remove_space || remove_first_space) as i32,
                unparse_special,
            );
            remove_space = false;
            if unparse_special
                || (token >= 0
                    && (token as usize) < self.id_to_token.len()
                    && self.token_get_attr(token) & (LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_CONTROL) == 0)
            {
                remove_first_space = false;
            }
            if n_chars < 0 {
                piece_buf.resize((-n_chars) as usize, 0);
                let check = self.token_to_piece(
                    token,
                    &mut piece_buf,
                    (remove_space || remove_first_space) as i32,
                    unparse_special,
                );
                if check > 0 {
                    rendered.extend_from_slice(&piece_buf[..check as usize]);
                }
            } else if n_chars > 0 {
                if n_chars as usize > piece_buf.len() {
                    piece_buf.resize(n_chars as usize, 0);
                    let check = self.token_to_piece(
                        token,
                        &mut piece_buf,
                        (remove_space || remove_first_space) as i32,
                        unparse_special,
                    );
                    if check > 0 {
                        rendered.extend_from_slice(&piece_buf[..check as usize]);
                    }
                } else {
                    rendered.extend_from_slice(&piece_buf[..n_chars as usize]);
                }
            }
        }

        let mut total = rendered.len();

        if self.clean_spaces && total > 0 {
            // first pass: characters ?!.,  //TODO: where do these characters come from?
            let total1 = total;
            total = 1;
            for i in 1..total1 {
                let x = rendered[i];
                if rendered[i - 1] == b' ' && matches!(x, b'?' | b'!' | b'.' | b',') {
                    total -= 1; // remove space
                }
                rendered[total] = x;
                total += 1;
            }

            // second pass: strip single apostrophe between spaces
            let total2 = total;
            total = 1;
            let mut i = 1;
            while i < total2 {
                let x = rendered[i];
                if x == b'\'' && i + 1 < total2 && rendered[i - 1] == b' ' && rendered[i + 1] == b' ' {
                    total -= 1; // remove prev space
                    if i + 1 < rendered.len() {
                        rendered[i + 1] = 0; // remove next space
                    }
                    i += 1;
                }
                rendered[total] = x;
                total += 1;
                i += 1;
            }

            // third pass: apostrophe contractions  //NOTE: this makes sense?
            let total3 = total;
            total = 1;
            for i in 1..total3 {
                let x = rendered[i];
                if rendered[i - 1] == b' ' && x == b'\'' && i + 1 < total3 {
                    let x1 = rendered[i + 1];
                    if x1 == b's' || x1 == b'm' {
                        total -= 1; // remove space
                    } else if i + 2 < total3 {
                        let x2 = rendered[i + 2];
                        if (x1 == b'r' && x2 == b'e') || (x1 == b'v' && x2 == b'e') {
                            total -= 1; // remove space
                        }
                    }
                }
                rendered[total] = x;
                total += 1;
            }
        }

        if total > text.len() {
            return -(total as i32);
        }
        text[..total].copy_from_slice(&rendered[..total]);
        total as i32
    }

    fn print_info(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("vocab type            = {}\n", self.type_name()));
        out.push_str(&format!("n_vocab               = {}\n", self.n_tokens()));
        out.push_str(&format!("n_merges              = {}\n", self.bpe_ranks.len()));

        // special tokens
        let specials = [
            ("BOS token", self.special_bos_id),
            ("EOS token", self.special_eos_id),
            ("EOT token", self.special_eot_id),
            ("EOM token", self.special_eom_id),
            ("UNK token", self.special_unk_id),
            ("SEP token", self.special_sep_id),
            ("PAD token", self.special_pad_id),
            ("MASK token", self.special_mask_id),
            ("LF token", self.linefeed_id),
            ("FIM PRE token", self.special_fim_pre_id),
            ("FIM SUF token", self.special_fim_suf_id),
            ("FIM MID token", self.special_fim_mid_id),
            ("FIM PAD token", self.special_fim_pad_id),
            ("FIM REP token", self.special_fim_rep_id),
            ("FIM SEP token", self.special_fim_sep_id),
        ];
        for (label, id) in specials {
            if id != LLAMA_TOKEN_NULL && id >= 0 && (id as usize) < self.id_to_token.len() {
                out.push_str(&format!(
                    "{: <21}= {} '{}'\n",
                    format!("{} ", label),
                    id,
                    self.id_to_token[id as usize].text
                ));
            }
        }

        for &id in &self.special_eog_ids {
            if id >= 0 && (id as usize) < self.id_to_token.len() {
                out.push_str(&format!(
                    "{: <21}= {} '{}'\n",
                    "EOG token ",
                    id,
                    self.id_to_token[id as usize].text
                ));
            }
        }

        out.push_str(&format!("max token length      = {}\n", self.max_token_len));
        out
    }

    fn type_name(&self) -> &'static str {
        match self.vtype {
            VocabType::None => "no vocab",
            VocabType::Spm => "SPM",
            VocabType::Bpe => "BPE",
            VocabType::Wpm => "WPM",
            VocabType::Ugm => "UGM",
            VocabType::Rwkv => "RWKV",
            VocabType::Plamo2 => "PLaMo2",
            VocabType::Plamo3 => "PLaMo3",
            VocabType::Test => "TEST",
        }
    }
}

// ---------------------------------------------------------------------------
// llama_vocab public API
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct LlamaVocab {
    inner: Impl,
}

impl LlamaVocab {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load(&mut self, ml: &Metadata) -> Result<()> {
        self.inner.load(ml)
    }

    pub fn get_tokenizer_model(&self) -> String {
        self.inner.get_tokenizer_model().to_string()
    }

    pub fn get_tokenizer_pre(&self) -> String {
        self.inner.tokenizer_pre.clone()
    }

    pub fn get_type(&self) -> VocabType {
        self.inner.vtype
    }

    pub fn get_pre_type(&self) -> VocabPreType {
        self.inner.pre_type
    }

    pub fn n_tokens(&self) -> u32 {
        self.inner.n_tokens()
    }

    pub fn n_token_types(&self) -> u32 {
        self.inner.n_token_types
    }

    pub fn type_name(&self) -> String {
        self.inner.type_name().to_string()
    }

    pub fn is_normal(&self, id: Token) -> bool {
        self.inner.id_to_token[id as usize].attr & LLAMA_TOKEN_ATTR_NORMAL != 0
    }

    pub fn is_unknown(&self, id: Token) -> bool {
        self.inner.id_to_token[id as usize].attr & LLAMA_TOKEN_ATTR_UNKNOWN != 0
    }

    pub fn is_control(&self, id: Token) -> bool {
        self.inner.id_to_token[id as usize].attr & LLAMA_TOKEN_ATTR_CONTROL != 0
    }

    pub fn is_byte(&self, id: Token) -> bool {
        self.inner.id_to_token[id as usize].attr & LLAMA_TOKEN_ATTR_BYTE != 0
    }

    pub fn is_user_defined(&self, id: Token) -> bool {
        self.inner.is_user_defined(id)
    }

    pub fn is_unused(&self, id: Token) -> bool {
        self.inner.id_to_token[id as usize].attr & LLAMA_TOKEN_ATTR_UNUSED != 0
    }

    pub fn is_eog(&self, id: Token) -> bool {
        id != LLAMA_TOKEN_NULL && self.inner.special_eog_ids.contains(&id)
    }

    pub fn token_to_byte(&self, id: Token) -> u8 {
        self.inner.token_to_byte(id)
    }

    pub fn byte_to_token(&self, ch: u8) -> Token {
        self.inner.byte_to_token(ch)
    }

    pub fn text_to_token(&self, text: &str) -> Token {
        self.inner.text_to_token(text)
    }

    pub fn get_token_data(&self, id: Token) -> &TokenData {
        self.inner.get_token_data(id)
    }

    pub fn token_get_text(&self, id: Token) -> &str {
        &self.inner.get_token_data(id).text
    }

    pub fn token_get_score(&self, id: Token) -> f32 {
        self.inner.token_get_score(id)
    }

    pub fn token_get_attr(&self, id: Token) -> LlamaTokenAttr {
        self.inner.token_get_attr(id)
    }

    pub fn token_bos(&self) -> Token {
        self.inner.token_bos()
    }
    pub fn token_eos(&self) -> Token {
        self.inner.token_eos()
    }
    pub fn token_eot(&self) -> Token {
        self.inner.special_eot_id
    }
    pub fn token_eom(&self) -> Token {
        self.inner.special_eom_id
    }
    pub fn token_unk(&self) -> Token {
        self.inner.token_unk()
    }
    pub fn token_sep(&self) -> Token {
        self.inner.special_sep_id
    }
    pub fn token_nl(&self) -> Token {
        self.inner.linefeed_id
    }
    pub fn token_pad(&self) -> Token {
        self.inner.special_pad_id
    }
    pub fn token_mask(&self) -> Token {
        self.inner.special_mask_id
    }
    pub fn token_prefix(&self) -> Token {
        self.inner.special_fim_pre_id
    }
    pub fn token_middle(&self) -> Token {
        self.inner.special_fim_mid_id
    }
    pub fn token_suffix(&self) -> Token {
        self.inner.special_fim_suf_id
    }
    pub fn token_fim_pre(&self) -> Token {
        self.inner.special_fim_pre_id
    }
    pub fn token_fim_suf(&self) -> Token {
        self.inner.special_fim_suf_id
    }
    pub fn token_fim_mid(&self) -> Token {
        self.inner.special_fim_mid_id
    }
    pub fn token_fim_pad(&self) -> Token {
        self.inner.special_fim_pad_id
    }
    pub fn token_fim_rep(&self) -> Token {
        self.inner.special_fim_rep_id
    }
    pub fn token_fim_sep(&self) -> Token {
        self.inner.special_fim_sep_id
    }

    pub fn get_add_space_prefix(&self) -> bool {
        self.inner.get_add_space_prefix()
    }
    pub fn get_add_bos(&self) -> bool {
        self.inner.get_add_bos()
    }
    pub fn get_add_eos(&self) -> bool {
        self.inner.get_add_eos()
    }
    pub fn get_add_sep(&self) -> bool {
        self.inner.add_sep
    }
    pub fn get_dna_auto_tags(&self) -> bool {
        self.inner.get_dna_auto_tags()
    }
    pub fn get_ignore_merges(&self) -> bool {
        self.inner.get_ignore_merges()
    }
    pub fn get_clean_spaces(&self) -> bool {
        self.inner.get_clean_spaces()
    }
    pub fn get_remove_extra_whitespaces(&self) -> bool {
        self.inner.get_remove_extra_whitespaces()
    }
    pub fn get_escape_whitespaces(&self) -> bool {
        self.inner.get_escape_whitespaces()
    }
    pub fn get_treat_whitespace_as_suffix(&self) -> bool {
        self.inner.get_treat_whitespace_as_suffix()
    }
    pub fn get_metaspace_prepend_scheme(&self) -> MetaspacePrependScheme {
        self.inner.get_metaspace_prepend_scheme()
    }
    pub fn get_normalizer_opts(&self) -> NormalizerOptions {
        self.inner.get_normalizer_opts()
    }

    pub fn get_suppress_tokens(&self) -> &[Token] {
        &self.inner.suppress_tokens
    }

    pub fn max_token_len(&self) -> i32 {
        self.inner.max_token_len()
    }

    pub fn find_bpe_rank(&self, token_left: &str, token_right: &str) -> i32 {
        self.inner.find_bpe_rank(token_left, token_right)
    }

    pub fn get_bpe_merges(&self) -> Vec<String> {
        let mut max_rank = -1i32;
        for &rank in self.inner.bpe_ranks.values() {
            max_rank = max_rank.max(rank);
        }
        let mut result = vec![String::new(); (max_rank + 1).max(0) as usize];
        for ((left, right), &rank) in &self.inner.bpe_ranks {
            if rank >= 0 && (rank as usize) < result.len() {
                result[rank as usize] = format!("{} {}", left, right);
            }
        }
        result
    }

    pub fn get_precompiled_charsmap(&self) -> Vec<u8> {
        self.inner.precompiled_charsmap.clone()
    }

    /// Copy token ids into `tokens`; returns the number written, or the negated
    /// required size when the slice is too small.
    pub fn tokenize_into(
        &self,
        text: &str,
        tokens: &mut [Token],
        add_special: bool,
        parse_special: bool,
    ) -> i32 {
        let res = self.tokenize(text, add_special, parse_special);
        if res.len() >= i32::MAX as usize {
            return i32::MIN;
        }
        if tokens.len() < res.len() {
            return -(res.len() as i32);
        }
        for (dst, src) in tokens.iter_mut().zip(res.iter()) {
            *dst = *src;
        }
        res.len() as i32
    }

    pub fn tokenize(&self, raw_text: &str, add_special: bool, parse_special: bool) -> Vec<Token> {
        self.inner.tokenize(raw_text, add_special, parse_special)
    }

    // does not write a null terminator to buf
    pub fn token_to_piece(&self, token: Token, buf: &mut [u8], lstrip: i32, special: bool) -> i32 {
        self.inner.token_to_piece(token, buf, lstrip, special)
    }

    // use cached data
    pub fn token_to_piece_cached(&self, token: Token) -> &str {
        self.inner
            .cache_token_to_piece
            .get(token as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    pub fn detokenize(
        &self,
        tokens: &[Token],
        text: &mut [u8],
        remove_special: bool,
        unparse_special: bool,
    ) -> i32 {
        self.inner.detokenize(tokens, text, remove_special, unparse_special)
    }

    pub fn detokenize_string(&self, tokens: &[Token], special: bool) -> String {
        let mut text = vec![0u8; tokens.len().max(1)];
        let mut n_chars = self.detokenize(tokens, &mut text, false, special);
        if n_chars < 0 {
            text.resize((-n_chars) as usize, 0);
            n_chars = self.detokenize(tokens, &mut text, false, special);
        }
        if n_chars < 0 {
            n_chars = 0;
        }
        String::from_utf8_lossy(&text[..(n_chars as usize).min(text.len())]).into_owned()
    }

    pub fn print_info(&self) -> String {
        self.inner.print_info()
    }
}

// ---------------------------------------------------------------------------
// interface implementation (plain Rust forms; no FFI)
// ---------------------------------------------------------------------------

pub fn llama_vocab_n_tokens(vocab: &LlamaVocab) -> i32 {
    vocab.n_tokens() as i32
}

pub fn llama_vocab_type(vocab: &LlamaVocab) -> VocabType {
    vocab.get_type()
}

pub fn llama_vocab_get_text(vocab: &LlamaVocab, token: Token) -> &str {
    vocab.token_get_text(token)
}

pub fn llama_vocab_get_score(vocab: &LlamaVocab, token: Token) -> f32 {
    vocab.token_get_score(token)
}

pub fn llama_vocab_get_attr(vocab: &LlamaVocab, token: Token) -> LlamaTokenAttr {
    vocab.token_get_attr(token)
}

pub fn llama_vocab_is_eog(vocab: &LlamaVocab, token: Token) -> bool {
    vocab.is_eog(token)
}

pub fn llama_vocab_is_control(vocab: &LlamaVocab, token: Token) -> bool {
    vocab.is_control(token)
}

pub fn llama_vocab_bos(vocab: &LlamaVocab) -> Token {
    vocab.token_bos()
}

pub fn llama_vocab_eos(vocab: &LlamaVocab) -> Token {
    vocab.token_eos()
}

pub fn llama_vocab_eot(vocab: &LlamaVocab) -> Token {
    vocab.token_eot()
}

pub fn llama_vocab_sep(vocab: &LlamaVocab) -> Token {
    vocab.token_sep()
}

pub fn llama_vocab_nl(vocab: &LlamaVocab) -> Token {
    vocab.token_nl()
}

pub fn llama_vocab_pad(vocab: &LlamaVocab) -> Token {
    vocab.token_pad()
}

pub fn llama_vocab_get_add_bos(vocab: &LlamaVocab) -> bool {
    vocab.get_add_bos()
}

pub fn llama_vocab_get_add_eos(vocab: &LlamaVocab) -> bool {
    vocab.get_add_eos()
}

pub fn llama_vocab_get_add_sep(vocab: &LlamaVocab) -> bool {
    vocab.get_add_sep()
}

pub fn llama_vocab_get_suppress_tokens(vocab: &LlamaVocab) -> &[Token] {
    vocab.get_suppress_tokens()
}

pub fn llama_vocab_fim_pre(vocab: &LlamaVocab) -> Token {
    vocab.token_fim_pre()
}

pub fn llama_vocab_fim_suf(vocab: &LlamaVocab) -> Token {
    vocab.token_fim_suf()
}

pub fn llama_vocab_fim_mid(vocab: &LlamaVocab) -> Token {
    vocab.token_fim_mid()
}

pub fn llama_vocab_fim_pad(vocab: &LlamaVocab) -> Token {
    vocab.token_fim_pad()
}

pub fn llama_vocab_fim_rep(vocab: &LlamaVocab) -> Token {
    vocab.token_fim_rep()
}

pub fn llama_vocab_fim_sep(vocab: &LlamaVocab) -> Token {
    vocab.token_fim_sep()
}

pub fn llama_vocab_mask(vocab: &LlamaVocab) -> Token {
    vocab.token_mask()
}

pub fn llama_tokenize(
    vocab: &LlamaVocab,
    text: &str,
    tokens: &mut [Token],
    add_special: bool,
    parse_special: bool,
) -> i32 {
    vocab.tokenize_into(text, tokens, add_special, parse_special)
}

pub fn llama_token_to_piece(
    vocab: &LlamaVocab,
    token: Token,
    buf: &mut [u8],
    lstrip: i32,
    special: bool,
) -> i32 {
    vocab.token_to_piece(token, buf, lstrip, special)
}

pub fn llama_detokenize(
    vocab: &LlamaVocab,
    tokens: &[Token],
    text: &mut [u8],
    remove_special: bool,
    unparse_special: bool,
) -> i32 {
    vocab.detokenize(tokens, text, remove_special, unparse_special)
}

// ---------------------------------------------------------------------------
// tests for pure helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_byte_map_roundtrips() {
        for b in 0u8..=255 {
            let piece = unicode_byte_to_utf8(b);
            assert_eq!(unicode_utf8_to_byte(&piece), b);
            let cpt = piece.chars().next().unwrap() as u32;
            assert_eq!(unicode_cpt_to_byte(cpt).unwrap(), b);
        }
    }

    #[test]
    fn trie_longest_prefix() {
        let mut trie = NaiveTrie::default();
        trie.insert(b"abcd", 7);
        trie.insert(b"ab", 3);
        assert_eq!(trie.get_longest_prefix(b"abxyz", 0), 2);
        assert_eq!(trie.get_longest_prefix(b"abcdef", 0), 4);
        assert_eq!(trie.get_longest_prefix(b"zzzz", 0), 0);
        assert_eq!(trie.traverse(b'a').unwrap().traverse(b'b').unwrap().value, 3);
        assert_eq!(parse_hex_byte("0A"), 10);
    }
}

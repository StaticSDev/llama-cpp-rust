//! Shared types, constants and helpers for the vocab port.

use super::*;
use crate::core::{Error, Result, Token, LLAMA_TOKEN_NULL};

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
pub(crate) const DNA_KMER_MARKER: &str = "\u{e000}";

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
    #[allow(dead_code)]
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

pub(crate) fn replace_all(text: &mut String, from: &str, to: &str) {
    if from.is_empty() {
        return;
    }
    *text = text.replace(from, to);
}

pub(crate) fn parse_hex_byte(s: &str) -> u8 {
    u8::from_str_radix(s, 16).unwrap_or(0)
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


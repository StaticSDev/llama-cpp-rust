//! Rust port of `llama-grammar.h` / `llama-grammar.cpp`.
//!
//! C++ stack entries are `const llama_grammar_element *`. Here a `GrammarPos`
//! is a `(rule, index)` pair, which gives the same identity and `+ 1` semantics
//! without raw pointers. Vocab access is supplied by the caller: token pieces
//! and the eog flag live on `TokenData`, and `<token>` parsing uses an optional
//! tokenizer closure.

use crate::core::{Error, Result, Token};
use std::collections::{BTreeMap, BTreeSet};

const MAX_REPETITION_THRESHOLD: u64 = 2000;
const GRAMMAR_RECURSIVE_LIMIT: u32 = 256;

// ---------------------------------------------------------------------------
// grammar element types (`enum llama_gretype`)
// ---------------------------------------------------------------------------

/// Grammar element type; discriminants match `enum llama_gretype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum Gretype {
    /// end of rule definition
    End = 0,
    /// start of alternate definition for rule
    Alt = 1,
    /// non-terminal element: reference to rule
    RuleRef = 2,
    /// terminal element: character (code point)
    Char = 3,
    /// inverse char(s) (`[^a]`, `[^a-b]` `[^abc]`)
    CharNot = 4,
    /// modifies a preceding `Char` to be an inclusive range (`[a-z]`)
    CharRngUpper = 5,
    /// modifies a preceding char element to add an alternate char (`[ab]`)
    CharAlt = 6,
    /// any character (`.`)
    CharAny = 7,
    /// terminal element: token (`<[token-id]>`)
    Token = 8,
    /// inverse token (`!<[token-id]>`)
    TokenNot = 9,
    /// modifies a preceding token element to be an inclusive range
    TokenRngUpper = 10,
    /// modifies a preceding token element to add an alternate token
    TokenAlt = 11,
}

/// `llama_grammar_element`: Unicode code point, rule ID, or token ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GrammarElement {
    pub type_: Gretype,
    pub value: u32,
}

impl GrammarElement {
    fn new(type_: Gretype, value: u32) -> Self {
        Self { type_, value }
    }
}

/// `llama_partial_utf8`: `n_remain == -1` marks an invalid sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartialUtf8 {
    pub value: u32,
    pub n_remain: i32,
}

impl Default for PartialUtf8 {
    fn default() -> Self {
        Self {
            value: 0,
            n_remain: 0,
        }
    }
}

/// `llama_grammar_candidate`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrammarCandidate {
    pub index: usize,
    pub code_points: Vec<u32>,
    pub partial_utf8: PartialUtf8,
    pub id: Token,
    pub opaque: bool,
}

/// Pointer-like position inside `Grammar::rules`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GrammarPos {
    pub rule: usize,
    pub index: usize,
}

impl GrammarPos {
    pub fn new(rule: usize, index: usize) -> Self {
        Self { rule, index }
    }

    /// C++ `pos + 1`.
    pub fn advance(self, n: usize) -> Self {
        Self {
            index: self.index + n,
            ..self
        }
    }
}

pub type GrammarRule = Vec<GrammarElement>;
pub type GrammarStack = Vec<GrammarPos>;
pub type GrammarRules = Vec<GrammarRule>;
pub type GrammarStacks = Vec<GrammarStack>;
pub type GrammarCandidates = Vec<GrammarCandidate>;

/// Fields of `llama_token_data` the grammar needs. In C++ `piece` and `is_eog`
/// come from the vocab; here the caller fills them in before `Grammar::apply`.
#[derive(Clone, Debug)]
pub struct TokenData {
    pub id: Token,
    pub logit: f32,
    pub piece: Vec<u8>,
    pub is_eog: bool,
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// C++ strings are NUL terminated; treat a byte past the end as NUL.
fn byte_at(src: &[u8], pos: usize) -> u8 {
    src.get(pos).copied().unwrap_or(0)
}

fn tail(src: &[u8], pos: usize) -> String {
    String::from_utf8_lossy(&src[pos.min(src.len())..]).into_owned()
}

// NOTE: assumes valid utf8 (but checks for overrun)
fn decode_utf8_char(src: &[u8], pos: usize) -> (u32, usize) {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    let first_byte = byte_at(src, pos);
    let highbits = (first_byte >> 4) as usize;
    let len = LOOKUP[highbits];
    let mask = ((1u32 << (8 - len)) - 1) as u8;
    let mut value = (first_byte & mask) as u32;
    let end = (pos + len).min(src.len()); // may overrun in C++, clamped here
    let mut p = pos + 1;
    while p < end && byte_at(src, p) != 0 {
        value = (value << 6) + (byte_at(src, p) & 0x3F) as u32;
        p += 1;
    }
    (value, p)
}

/// Port of `llama_grammar_decode_utf8`. The returned code points end in 0.
pub fn decode_utf8(src: &[u8], partial_start: PartialUtf8) -> (Vec<u32>, PartialUtf8) {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4];
    let mut pos = 0usize;
    let mut code_points = Vec::with_capacity(src.len() + 1);

    let mut value = partial_start.value;
    let mut n_remain = partial_start.n_remain;

    // continue previous decode, if applicable
    while byte_at(src, pos) != 0 && n_remain > 0 {
        let next_byte = byte_at(src, pos);
        if (next_byte >> 6) != 2 {
            // invalid sequence, abort
            code_points.push(0);
            return (
                code_points,
                PartialUtf8 {
                    value: 0,
                    n_remain: -1,
                },
            );
        }
        value = (value << 6) + (next_byte & 0x3F) as u32;
        pos += 1;
        n_remain -= 1;
    }

    if partial_start.n_remain > 0 && n_remain == 0 {
        code_points.push(value);
    }

    // decode any subsequent utf-8 sequences, which may end in an incomplete one
    while byte_at(src, pos) != 0 {
        let first_byte = byte_at(src, pos);
        let highbits = (first_byte >> 4) as usize;
        n_remain = LOOKUP[highbits] as i32 - 1;

        if n_remain < 0 {
            // invalid sequence, abort
            code_points.clear();
            code_points.push(0);
            return (code_points, PartialUtf8 { value: 0, n_remain });
        }

        let mask = ((1u32 << ((7 - n_remain) as u32)) - 1) as u8;
        value = (first_byte & mask) as u32;

        pos += 1;
        while byte_at(src, pos) != 0 && n_remain > 0 {
            value = (value << 6) + (byte_at(src, pos) & 0x3F) as u32;
            pos += 1;
            n_remain -= 1;
        }
        if n_remain == 0 {
            code_points.push(value);
        }
    }
    code_points.push(0);

    (code_points, PartialUtf8 { value, n_remain })
}

fn is_digit_char(c: u8) -> bool {
    b'0' <= c && c <= b'9'
}

fn is_word_char(c: u8) -> bool {
    (b'a' <= c && c <= b'z') || (b'A' <= c && c <= b'Z') || c == b'-' || is_digit_char(c)
}

fn parse_hex(src: &[u8], pos: usize, size: usize) -> Result<(u32, usize)> {
    let mut p = pos;
    let mut value = 0u32;
    for _ in 0..size {
        let c = byte_at(src, p);
        if c == 0 {
            break;
        }
        value <<= 4;
        match c {
            b'a'..=b'f' => value += (c - b'a' + 10) as u32,
            b'A'..=b'F' => value += (c - b'A' + 10) as u32,
            b'0'..=b'9' => value += (c - b'0') as u32,
            _ => break,
        }
        p += 1;
    }
    if p - pos != size {
        return Err(Error::new(format!(
            "expecting {} hex chars at {}",
            size,
            tail(src, pos)
        )));
    }
    Ok((value, p))
}

fn parse_space(src: &[u8], mut pos: usize, newline_ok: bool) -> usize {
    loop {
        let c = byte_at(src, pos);
        if c == b' ' || c == b'\t' || c == b'#' || (newline_ok && (c == b'\r' || c == b'\n')) {
            if c == b'#' {
                while {
                    let c = byte_at(src, pos);
                    c != 0 && c != b'\r' && c != b'\n'
                } {
                    pos += 1;
                }
            } else {
                pos += 1;
            }
        } else {
            break;
        }
    }
    pos
}

fn parse_name(src: &[u8], pos: usize) -> Result<usize> {
    let mut p = pos;
    while is_word_char(byte_at(src, p)) {
        p += 1;
    }
    if p == pos {
        return Err(Error::new(format!("expecting name at {}", tail(src, pos))));
    }
    Ok(p)
}

fn parse_int(src: &[u8], pos: usize) -> Result<usize> {
    let mut p = pos;
    while is_digit_char(byte_at(src, p)) {
        p += 1;
    }
    if p == pos {
        return Err(Error::new(format!(
            "expecting integer at {}",
            tail(src, pos)
        )));
    }
    Ok(p)
}

fn parse_char(src: &[u8], pos: usize) -> Result<(u32, usize)> {
    if byte_at(src, pos) == b'\\' {
        match byte_at(src, pos + 1) {
            b'x' => parse_hex(src, pos + 2, 2),
            b'u' => parse_hex(src, pos + 2, 4),
            b'U' => parse_hex(src, pos + 2, 8),
            b't' => Ok((b'\t' as u32, pos + 2)),
            b'r' => Ok((b'\r' as u32, pos + 2)),
            b'n' => Ok((b'\n' as u32, pos + 2)),
            b'\\' | b'"' | b'[' | b']' | b'-' => Ok((byte_at(src, pos + 1) as u32, pos + 2)),
            _ => Err(Error::new(format!("unknown escape at {}", tail(src, pos)))),
        }
    } else if byte_at(src, pos) != 0 {
        Ok(decode_utf8_char(src, pos))
    } else {
        Err(Error::new("unexpected end of input"))
    }
}

fn parse_token_id(src: &[u8], pos: usize) -> Result<(u32, usize)> {
    let end = parse_int(src, pos)?;
    let text = std::str::from_utf8(&src[pos..end]).map_err(|_| Error::new("invalid token id"))?;
    let id: u64 = text
        .parse()
        .map_err(|_| Error::new(format!("parsed token id is too big at {}", tail(src, pos))))?;
    if id > u32::MAX as u64 {
        return Err(Error::new(format!(
            "parsed token id is too big at {}",
            tail(src, pos)
        )));
    }
    Ok((id as u32, end))
}

// ---------------------------------------------------------------------------
// printing helpers
// ---------------------------------------------------------------------------

fn print_grammar_char(c: u32) -> String {
    if (0x20..=0x7f).contains(&c) {
        (c as u8 as char).to_string()
    } else {
        // cop out of encoding UTF-8
        format!("<U+{:04X}>", c)
    }
}

fn type_name(t: Gretype) -> &'static str {
    match t {
        Gretype::End => "END",
        Gretype::Alt => "ALT",
        Gretype::RuleRef => "RULE_REF",
        Gretype::Char => "CHAR",
        Gretype::CharNot => "CHAR_NOT",
        Gretype::CharRngUpper => "CHAR_RNG_UPPER",
        Gretype::CharAlt => "CHAR_ALT",
        Gretype::CharAny => "CHAR_ANY",
        Gretype::Token => "TOKEN",
        Gretype::TokenNot => "TOKEN_NOT",
        Gretype::TokenRngUpper => "TOKEN_RNG_UPPER",
        Gretype::TokenAlt => "TOKEN_ALT",
    }
}

fn is_token_element(elem: &GrammarElement) -> bool {
    matches!(
        elem.type_,
        Gretype::Token | Gretype::TokenNot | Gretype::TokenAlt | Gretype::TokenRngUpper
    )
}

fn is_char_element(elem: &GrammarElement) -> bool {
    matches!(
        elem.type_,
        Gretype::Char
            | Gretype::CharNot
            | Gretype::CharAlt
            | Gretype::CharRngUpper
            | Gretype::CharAny
    )
}

/// Debug view of one rule, mirroring `print_rule_binary`.
pub fn print_rule_binary(rule: &GrammarRule) -> String {
    let mut out = String::new();
    for elem in rule {
        out.push_str(type_name(elem.type_));
        match elem.type_ {
            Gretype::End | Gretype::Alt | Gretype::RuleRef => {
                out.push_str(&format!("({}) ", elem.value));
            }
            Gretype::Char
            | Gretype::CharNot
            | Gretype::CharRngUpper
            | Gretype::CharAlt
            | Gretype::CharAny => {
                out.push_str("(\"");
                out.push_str(&print_grammar_char(elem.value));
                out.push_str("\") ");
            }
            Gretype::Token => {
                out.push_str(&format!("<[{}]> ", elem.value));
            }
            Gretype::TokenNot => {
                out.push_str(&format!("!<[{}]> ", elem.value));
            }
            Gretype::TokenRngUpper | Gretype::TokenAlt => {
                out.push_str(&format!("({}) ", elem.value));
            }
        }
    }
    out.push('\n');
    out
}

fn print_rule(
    rule_id: u32,
    rule: &GrammarRule,
    symbol_id_names: &BTreeMap<u32, String>,
) -> Result<String> {
    if rule.is_empty() || rule.last().unwrap().type_ != Gretype::End {
        return Err(Error::new(format!(
            "malformed rule, does not end with LLAMA_GRETYPE_END: {}",
            rule_id
        )));
    }
    let name = symbol_id_names
        .get(&rule_id)
        .ok_or_else(|| Error::new(format!("no name for rule {}", rule_id)))?;
    let mut out = format!("{} ::= ", name);
    let end = rule.len() - 1;
    for i in 0..end {
        let elem = &rule[i];
        match elem.type_ {
            Gretype::End => {
                return Err(Error::new(format!(
                    "unexpected end of rule: {},{}",
                    rule_id, i
                )));
            }
            Gretype::Alt => out.push_str("| "),
            Gretype::RuleRef => {
                let n = symbol_id_names
                    .get(&elem.value)
                    .ok_or_else(|| Error::new(format!("no name for rule {}", elem.value)))?;
                out.push_str(n);
                out.push(' ');
            }
            Gretype::Char => {
                out.push('[');
                out.push_str(&print_grammar_char(elem.value));
            }
            Gretype::CharNot => {
                out.push_str("[^");
                out.push_str(&print_grammar_char(elem.value));
            }
            Gretype::CharRngUpper => {
                if i == 0 || !is_char_element(&rule[i - 1]) {
                    return Err(Error::new(format!(
                        "LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: {},{}",
                        rule_id, i
                    )));
                }
                out.push('-');
                out.push_str(&print_grammar_char(elem.value));
            }
            Gretype::CharAlt => {
                if i == 0 || !is_char_element(&rule[i - 1]) {
                    return Err(Error::new(format!(
                        "LLAMA_GRETYPE_CHAR_ALT without preceding char: {},{}",
                        rule_id, i
                    )));
                }
                out.push_str(&print_grammar_char(elem.value));
            }
            Gretype::CharAny => out.push('.'),
            Gretype::Token => {
                out.push_str("<[");
                out.push_str(&elem.value.to_string());
            }
            Gretype::TokenNot => {
                out.push('!');
                out.push_str("<[");
                out.push_str(&elem.value.to_string());
            }
            Gretype::TokenRngUpper => {
                if i == 0 || !is_token_element(&rule[i - 1]) {
                    return Err(Error::new(format!(
                        "LLAMA_GRETYPE_TOKEN_RNG_UPPER without preceding token: {},{}",
                        rule_id, i
                    )));
                }
                out.push_str(&format!("-{}", elem.value));
            }
            Gretype::TokenAlt => {
                if i == 0 || !is_token_element(&rule[i - 1]) {
                    return Err(Error::new(format!(
                        "LLAMA_GRETYPE_TOKEN_ALT without preceding token: {},{}",
                        rule_id, i
                    )));
                }
                out.push_str(&format!(",{}", elem.value));
            }
        }
        if is_token_element(elem)
            && !matches!(
                rule[i + 1].type_,
                Gretype::TokenAlt | Gretype::TokenRngUpper
            )
        {
            out.push_str("]> ");
        }
        if is_char_element(elem) {
            match rule[i + 1].type_ {
                Gretype::CharAlt | Gretype::CharRngUpper | Gretype::CharAny => {}
                _ => out.push_str("] "),
            }
        }
    }
    out.push('\n');
    Ok(out)
}

// ---------------------------------------------------------------------------
// lazy grammar triggers
// ---------------------------------------------------------------------------

/// `llama_grammar_trigger_pattern`.
#[derive(Clone, Debug)]
pub struct TriggerPattern {
    pub pattern: String,
}

impl TriggerPattern {
    /// Returns the start of the match.
    /// shortcut: std has no regex; literal patterns with optional `^`/`$`
    /// anchors are supported. Use a precomputed match position for full regex.
    pub fn find(&self, input: &str) -> Option<usize> {
        let p = self.pattern.as_str();
        if p.len() >= 2 && p.starts_with('^') && p.ends_with('$') {
            let inner = &p[1..p.len() - 1];
            return if input == inner { Some(0) } else { None };
        }
        let inner = p
            .strip_prefix('^')
            .or_else(|| p.strip_suffix('$'))
            .unwrap_or(p);
        input.find(inner)
    }
}

// ---------------------------------------------------------------------------
// parser (`struct llama_grammar_parser`)
// ---------------------------------------------------------------------------

/// C++ `llama_vocab::tokenize` narrowed to what `<token>` parsing needs:
/// receives the full `<...>` text with `add_bos = false, special = true`.
pub type TokenizeFn = dyn Fn(&[u8]) -> Result<Vec<Token>>;

pub struct GrammarParser {
    tokenize: Option<Box<TokenizeFn>>, // C++ `const llama_vocab * vocab`
    pub symbol_ids: BTreeMap<String, u32>,
    pub rules: GrammarRules,
}

impl GrammarParser {
    pub fn new() -> Self {
        Self::with_tokenizer_opt(None)
    }

    pub fn with_tokenizer<F>(tokenize: F) -> Self
    where
        F: Fn(&[u8]) -> Result<Vec<Token>> + 'static,
    {
        Self::with_tokenizer_opt(Some(Box::new(tokenize)))
    }

    fn with_tokenizer_opt(tokenize: Option<Box<TokenizeFn>>) -> Self {
        Self {
            tokenize,
            symbol_ids: BTreeMap::new(),
            rules: Vec::new(),
        }
    }

    pub fn get_symbol_id(&mut self, name: &[u8]) -> u32 {
        let next_id = self.symbol_ids.len() as u32;
        let key = String::from_utf8_lossy(name).into_owned();
        *self.symbol_ids.entry(key).or_insert(next_id)
    }

    pub fn generate_symbol_id(&mut self, base_name: &str) -> u32 {
        let next_id = self.symbol_ids.len() as u32;
        self.symbol_ids
            .insert(format!("{}_{}", base_name, next_id), next_id);
        next_id
    }

    pub fn add_rule(&mut self, rule_id: u32, rule: GrammarRule) {
        let rule_id = rule_id as usize;
        if self.rules.len() <= rule_id {
            self.rules.resize(rule_id + 1, GrammarRule::new());
        }
        self.rules[rule_id] = rule;
    }

    fn parse_token(
        &self,
        src: &[u8],
        mut pos: usize,
        type_: Gretype,
        rule: &mut GrammarRule,
    ) -> Result<usize> {
        if byte_at(src, pos) != b'<' {
            return Err(Error::new(format!("expecting '<' at {}", tail(src, pos))));
        }
        pos += 1;

        // Parse <[id,id-id,...]>
        if byte_at(src, pos) == b'[' {
            pos += 1;
            let start = rule.len();
            loop {
                let (token_id, np) = parse_token_id(src, pos)?;
                pos = np;
                let t = if rule.len() == start {
                    type_
                } else {
                    Gretype::TokenAlt
                };
                rule.push(GrammarElement::new(t, token_id));
                if byte_at(src, pos) == b'-' {
                    pos += 1;
                    let (end, np) = parse_token_id(src, pos)?;
                    if end < token_id {
                        return Err(Error::new(format!(
                            "invalid token range at {}",
                            tail(src, pos)
                        )));
                    }
                    pos = np;
                    rule.push(GrammarElement::new(Gretype::TokenRngUpper, end));
                }
                if byte_at(src, pos) != b',' {
                    break;
                }
                pos += 1;
            }
            if byte_at(src, pos) != b']' {
                return Err(Error::new(format!("expecting ']' at {}", tail(src, pos))));
            }
            pos += 1;
            if byte_at(src, pos) != b'>' {
                return Err(Error::new(format!("expecting '>' at {}", tail(src, pos))));
            }
            pos += 1;
            return Ok(pos);
        }

        let tokenize = self
            .tokenize
            .as_ref()
            .ok_or_else(|| Error::new(format!("no vocab to parse token at {}", tail(src, pos))))?;

        // Parse <token> and tokenize to obtain the token id
        let src_start = pos - 1; // include '<', as in C++
        while byte_at(src, pos) != 0 && byte_at(src, pos) != b'>' {
            pos += 1;
        }
        if byte_at(src, pos) != b'>' {
            return Err(Error::new(format!("expecting '>' at {}", tail(src, pos))));
        }
        pos += 1;

        let tokens = tokenize(&src[src_start..pos])?;
        if tokens.len() != 1 {
            // must tokenize to exactly 1 token
            return Err(Error::new(format!(
                "invalid token '{}'",
                String::from_utf8_lossy(&src[src_start..pos])
            )));
        }
        rule.push(GrammarElement::new(type_, tokens[0] as u32));
        Ok(pos)
    }

    pub fn parse_alternates(
        &mut self,
        src: &[u8],
        mut pos: usize,
        rule_name: &str,
        rule_id: u32,
        is_nested: bool,
        depth: u32,
    ) -> Result<usize> {
        let mut rule = GrammarRule::new();
        pos = self.parse_sequence(src, pos, rule_name, &mut rule, is_nested, depth)?;
        while byte_at(src, pos) == b'|' {
            rule.push(GrammarElement::new(Gretype::Alt, 0));
            pos = parse_space(src, pos + 1, true);
            pos = self.parse_sequence(src, pos, rule_name, &mut rule, is_nested, depth)?;
        }
        rule.push(GrammarElement::new(Gretype::End, 0));
        self.add_rule(rule_id, rule);
        Ok(pos)
    }

    pub fn parse_sequence(
        &mut self,
        src: &[u8],
        mut pos: usize,
        rule_name: &str,
        rule: &mut GrammarRule,
        is_nested: bool,
        depth: u32,
    ) -> Result<usize> {
        let mut last_sym_start = rule.len();
        let mut n_prev_rules: u64 = 1;

        while byte_at(src, pos) != 0 {
            if byte_at(src, pos) == b'"' {
                // literal string
                pos += 1;
                last_sym_start = rule.len();
                n_prev_rules = 1;
                while byte_at(src, pos) != b'"' {
                    if byte_at(src, pos) == 0 {
                        return Err(Error::new("unexpected end of input"));
                    }
                    let (value, np) = parse_char(src, pos)?;
                    pos = np;
                    rule.push(GrammarElement::new(Gretype::Char, value));
                }
                pos = parse_space(src, pos + 1, is_nested);
            } else if byte_at(src, pos) == b'[' {
                // char range(s)
                pos += 1;
                let mut start_type = Gretype::Char;
                if byte_at(src, pos) == b'^' {
                    pos += 1;
                    start_type = Gretype::CharNot;
                }
                last_sym_start = rule.len();
                n_prev_rules = 1;
                while byte_at(src, pos) != b']' {
                    if byte_at(src, pos) == 0 {
                        return Err(Error::new("unexpected end of input"));
                    }
                    let (value, np) = parse_char(src, pos)?;
                    pos = np;
                    let t = if last_sym_start < rule.len() {
                        Gretype::CharAlt
                    } else {
                        start_type
                    };
                    rule.push(GrammarElement::new(t, value));
                    if byte_at(src, pos) == b'-' && byte_at(src, pos + 1) != b']' {
                        if byte_at(src, pos + 1) == 0 {
                            return Err(Error::new("unexpected end of input"));
                        }
                        let (end, np) = parse_char(src, pos + 1)?;
                        pos = np;
                        rule.push(GrammarElement::new(Gretype::CharRngUpper, end));
                    }
                }
                pos = parse_space(src, pos + 1, is_nested);
            } else if byte_at(src, pos) == b'<' || byte_at(src, pos) == b'!' {
                // token
                let mut type_ = Gretype::Token;
                if byte_at(src, pos) == b'!' {
                    // token inverse
                    type_ = Gretype::TokenNot;
                    pos += 1;
                }
                last_sym_start = rule.len();
                n_prev_rules = 1;
                pos = self.parse_token(src, pos, type_, rule)?;
                pos = parse_space(src, pos, is_nested);
            } else if is_word_char(byte_at(src, pos)) {
                // rule reference
                let name_end = parse_name(src, pos)?;
                let ref_rule_id = self.get_symbol_id(&src[pos..name_end]);
                pos = parse_space(src, name_end, is_nested);
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(GrammarElement::new(Gretype::RuleRef, ref_rule_id));
            } else if byte_at(src, pos) == b'(' {
                // grouping
                if depth >= GRAMMAR_RECURSIVE_LIMIT {
                    return Err(Error::new(format!(
                        "grammar recursion depth limit exceeded (max {}), please reduce nesting depth",
                        GRAMMAR_RECURSIVE_LIMIT
                    )));
                }
                // parse nested alternates into synthesized rule
                pos = parse_space(src, pos + 1, true);
                let n_rules_before = self.symbol_ids.len() as u32;
                let sub_rule_id = self.generate_symbol_id(rule_name);
                pos = self.parse_alternates(src, pos, rule_name, sub_rule_id, true, depth + 1)?;
                n_prev_rules = (self.symbol_ids.len() as u32 - n_rules_before).max(1) as u64;
                last_sym_start = rule.len();
                // output reference to synthesized rule
                rule.push(GrammarElement::new(Gretype::RuleRef, sub_rule_id));
                if byte_at(src, pos) != b')' {
                    return Err(Error::new(format!("expecting ')' at {}", tail(src, pos))));
                }
                pos = parse_space(src, pos + 1, is_nested);
            } else if byte_at(src, pos) == b'.' {
                // any char
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(GrammarElement::new(Gretype::CharAny, 0));
                pos = parse_space(src, pos + 1, is_nested);
            } else if byte_at(src, pos) == b'*' {
                pos = parse_space(src, pos + 1, is_nested);
                handle_repetitions(
                    self,
                    rule,
                    rule_name,
                    src,
                    pos,
                    last_sym_start,
                    &mut n_prev_rules,
                    0,
                    u64::MAX,
                )?;
            } else if byte_at(src, pos) == b'+' {
                pos = parse_space(src, pos + 1, is_nested);
                handle_repetitions(
                    self,
                    rule,
                    rule_name,
                    src,
                    pos,
                    last_sym_start,
                    &mut n_prev_rules,
                    1,
                    u64::MAX,
                )?;
            } else if byte_at(src, pos) == b'?' {
                pos = parse_space(src, pos + 1, is_nested);
                handle_repetitions(
                    self,
                    rule,
                    rule_name,
                    src,
                    pos,
                    last_sym_start,
                    &mut n_prev_rules,
                    0,
                    1,
                )?;
            } else if byte_at(src, pos) == b'{' {
                pos = parse_space(src, pos + 1, is_nested);

                if !is_digit_char(byte_at(src, pos)) {
                    return Err(Error::new(format!(
                        "expecting an int at {}",
                        tail(src, pos)
                    )));
                }
                let int_end = parse_int(src, pos)?;
                let min_times: u64 = std::str::from_utf8(&src[pos..int_end])
                    .map_err(|_| Error::new("invalid repetition count"))?
                    .parse()
                    .map_err(|_| Error::new("invalid repetition count"))?;
                pos = parse_space(src, int_end, is_nested);

                let mut max_times = u64::MAX; // default: no max limit
                if byte_at(src, pos) == b'}' {
                    max_times = min_times;
                    pos = parse_space(src, pos + 1, is_nested);
                } else if byte_at(src, pos) == b',' {
                    pos = parse_space(src, pos + 1, is_nested);
                    if is_digit_char(byte_at(src, pos)) {
                        let int_end = parse_int(src, pos)?;
                        max_times = std::str::from_utf8(&src[pos..int_end])
                            .map_err(|_| Error::new("invalid repetition count"))?
                            .parse()
                            .map_err(|_| Error::new("invalid repetition count"))?;
                        pos = parse_space(src, int_end, is_nested);
                    }
                    if byte_at(src, pos) != b'}' {
                        return Err(Error::new(format!("expecting '}}' at {}", tail(src, pos))));
                    }
                    pos = parse_space(src, pos + 1, is_nested);
                } else {
                    return Err(Error::new(format!("expecting ',' at {}", tail(src, pos))));
                }
                if min_times > MAX_REPETITION_THRESHOLD {
                    return Err(Error::new(
                        "number of repetitions exceeds sane defaults, please reduce the number of repetitions",
                    ));
                }
                if max_times != u64::MAX && max_times > MAX_REPETITION_THRESHOLD {
                    max_times = u64::MAX;
                }
                if max_times < min_times {
                    return Err(Error::new("incorrect number of repetitions"));
                }
                handle_repetitions(
                    self,
                    rule,
                    rule_name,
                    src,
                    pos,
                    last_sym_start,
                    &mut n_prev_rules,
                    min_times,
                    max_times,
                )?;
            } else {
                break;
            }
        }
        Ok(pos)
    }

    pub fn parse_rule(&mut self, src: &[u8], pos: usize) -> Result<usize> {
        let name_end = parse_name(src, pos)?;
        let mut p = parse_space(src, name_end, false);
        let name_bytes = src[pos..name_end].to_vec();
        let rule_id = self.get_symbol_id(&name_bytes);
        let name = String::from_utf8_lossy(&name_bytes).into_owned();

        if !(byte_at(src, p) == b':' && byte_at(src, p + 1) == b':' && byte_at(src, p + 2) == b'=')
        {
            return Err(Error::new(format!("expecting ::= at {}", tail(src, p))));
        }
        p = parse_space(src, p + 3, true);

        p = self.parse_alternates(src, p, &name, rule_id, false, 0)?;

        if byte_at(src, p) == b'\r' {
            p += if byte_at(src, p + 1) == b'\n' { 2 } else { 1 };
        } else if byte_at(src, p) == b'\n' {
            p += 1;
        } else if byte_at(src, p) != 0 {
            return Err(Error::new(format!(
                "expecting newline or end at {}",
                tail(src, p)
            )));
        }
        Ok(parse_space(src, p, true))
    }

    /// Error-propagating form of `parse`.
    pub fn parse_result(&mut self, src: &str) -> Result<()> {
        let bytes = src.as_bytes();
        let mut pos = parse_space(bytes, 0, true);
        while byte_at(bytes, pos) != 0 {
            pos = self.parse_rule(bytes, pos)?;
        }

        // Validate the state to ensure that all rules are defined
        for rule in &self.rules {
            if rule.is_empty() {
                return Err(Error::new("Undefined rule"));
            }
            for elem in rule {
                if elem.type_ == Gretype::RuleRef {
                    let id = elem.value as usize;
                    if id >= self.rules.len() || self.rules[id].is_empty() {
                        // Get the name of the rule that is missing
                        for (name, rule_id) in &self.symbol_ids {
                            if *rule_id == elem.value {
                                return Err(Error::new(format!(
                                    "Undefined rule identifier '{}'",
                                    name
                                )));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// C++ `bool parse(const char * src)`: clears rules on error.
    pub fn parse(&mut self, src: &str) -> bool {
        match self.parse_result(src) {
            Ok(()) => true,
            Err(err) => {
                eprintln!("parse: error parsing grammar: {}\n\n{}", err, src);
                self.rules.clear();
                false
            }
        }
    }

    /// Text form of C++ `print(FILE *)`.
    pub fn print(&self) -> String {
        let mut symbol_id_names = BTreeMap::new();
        for (name, id) in &self.symbol_ids {
            symbol_id_names.insert(*id, name.clone());
        }
        let mut out = String::new();
        for (i, rule) in self.rules.iter().enumerate() {
            match print_rule(i as u32, rule, &symbol_id_names) {
                Ok(text) => out.push_str(&text),
                Err(err) => eprintln!("\nprint: error printing grammar: {}", err),
            }
        }
        out
    }
}

impl Default for GrammarParser {
    fn default() -> Self {
        Self::new()
    }
}

fn handle_repetitions(
    parser: &mut GrammarParser,
    rule: &mut GrammarRule,
    rule_name: &str,
    src: &[u8],
    pos: usize,
    last_sym_start: usize,
    n_prev_rules: &mut u64,
    min_times: u64,
    max_times: u64,
) -> Result<()> {
    let no_max = max_times == u64::MAX;
    if last_sym_start == rule.len() {
        return Err(Error::new(format!(
            "expecting preceding item to */+/?/{{ at {}",
            tail(src, pos)
        )));
    }

    // apply transformation to previous symbol (last_sym_start to end) according to
    // the following rewrite rules:
    // S{m,n} --> S S S (m times) S'(n-m)
    //            S'(x)   ::= S S'(x-1) |
    //            (... n-m definitions of these S' rules ...)
    //            S'(1)   ::= S |
    // S{m,} -->  S S S (m times) S'
    //            S'     ::= S S' |
    // S*     --> S{0,}
    //        --> S'     ::= S S' |
    // S+     --> S{1,}
    //        --> S S'
    //            S'     ::= S S' |
    // S?     --> S{0,1}
    //        --> S'
    //            S'     ::= S |

    let prev_rule = rule[last_sym_start..].to_vec();
    // Calculate the total number of rules that will be generated by this repetition
    let mut total_rules = 1u64;
    if !no_max && max_times > 0 {
        total_rules = max_times;
    } else if min_times > 0 {
        total_rules = min_times;
    }

    if *n_prev_rules * total_rules > MAX_REPETITION_THRESHOLD {
        return Err(Error::new(
            "number of rules that are going to be repeated multiplied by the new repetition exceeds sane defaults, please reduce the number of repetitions or rule complexity",
        ));
    }

    if min_times == 0 {
        rule.truncate(last_sym_start);
    } else {
        // Repeat the previous elements (min_times - 1) times
        for _ in 1..min_times {
            rule.extend(prev_rule.iter().cloned());
        }
    }

    let mut last_rec_rule_id = 0u32;
    let n_opt = if no_max { 1 } else { max_times - min_times };

    let mut rec_rule = prev_rule.clone();
    for i in 0..n_opt {
        rec_rule.truncate(prev_rule.len());
        let rec_rule_id = parser.generate_symbol_id(rule_name);
        if i > 0 || no_max {
            let target = if no_max {
                rec_rule_id
            } else {
                last_rec_rule_id
            };
            rec_rule.push(GrammarElement::new(Gretype::RuleRef, target));
        }
        rec_rule.push(GrammarElement::new(Gretype::Alt, 0));
        rec_rule.push(GrammarElement::new(Gretype::End, 0));
        parser.add_rule(rec_rule_id, rec_rule.clone());
        last_rec_rule_id = rec_rule_id;
    }
    if n_opt > 0 {
        rule.push(GrammarElement::new(Gretype::RuleRef, last_rec_rule_id));
    }
    *n_prev_rules *= total_rules;
    debug_assert!(*n_prev_rules >= 1);
    Ok(())
}

// ---------------------------------------------------------------------------
// state machine helpers
// ---------------------------------------------------------------------------

fn elem<'a>(rules: &'a [GrammarRule], pos: GrammarPos) -> &'a GrammarElement {
    &rules[pos.rule][pos.index]
}

// returns true iff pos points to the end of one of the definitions of a rule
fn is_end_of_sequence(elem: &GrammarElement) -> bool {
    matches!(elem.type_, Gretype::End | Gretype::Alt) // NOLINT
}

// returns true iff chr satisfies the char range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
fn match_char(rules: &[GrammarRule], pos: GrammarPos, chr: u32) -> (bool, GrammarPos) {
    let mut p = pos;
    let mut found = false;
    let is_positive_char = matches!(elem(rules, p).type_, Gretype::Char | Gretype::CharAny);

    loop {
        if elem(rules, p.advance(1)).type_ == Gretype::CharRngUpper {
            // inclusive range, e.g. [a-z]
            found =
                found || (elem(rules, p).value <= chr && chr <= elem(rules, p.advance(1)).value);
            p = p.advance(2);
        } else if elem(rules, p).type_ == Gretype::CharAny {
            // Any character matches "."
            found = true;
            p = p.advance(1);
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || elem(rules, p).value == chr;
            p = p.advance(1);
        }
        if elem(rules, p).type_ != Gretype::CharAlt {
            break;
        }
    }

    (found == is_positive_char, p)
}

// returns true iff some continuation of the given partial UTF-8 sequence could satisfy the char
// range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
fn match_partial_char(rules: &[GrammarRule], pos: GrammarPos, partial_utf8: PartialUtf8) -> bool {
    let mut p = pos;
    let is_positive_char = matches!(elem(rules, p).type_, Gretype::Char | Gretype::CharAny);

    let partial_value = partial_utf8.value;
    let n_remain = partial_utf8.n_remain;

    // invalid sequence or 7-bit char split across 2 bytes (overlong)
    if n_remain < 0 || (n_remain == 1 && partial_value < 2) {
        return false;
    }

    // range of possible code points this partial UTF-8 sequence could complete to
    let shift = (n_remain * 6) as u32;
    let mut low = partial_value.wrapping_shl(shift);
    let mut high = low | 1u32.wrapping_shl(shift).wrapping_sub(1);

    if low == 0 {
        if n_remain == 2 {
            low = 1 << 11;
        } else if n_remain == 3 {
            low = 1 << 16;
        }
    }

    loop {
        if elem(rules, p.advance(1)).type_ == Gretype::CharRngUpper {
            // inclusive range, e.g. [a-z]
            if elem(rules, p).value <= high && low <= elem(rules, p.advance(1)).value {
                return is_positive_char;
            }
            p = p.advance(2);
        } else if elem(rules, p).type_ == Gretype::CharAny {
            // Any character matches "."
            return true;
        } else {
            // exact char match, e.g. [a] or "a"
            if low <= elem(rules, p).value && elem(rules, p).value <= high {
                return is_positive_char;
            }
            p = p.advance(1);
        }
        if elem(rules, p).type_ != Gretype::CharAlt {
            break;
        }
    }

    !is_positive_char
}

// returns true iff token matches the rule at pos (regular or inverse)
// asserts that pos is pointing to a token element
fn match_token(rules: &[GrammarRule], pos: GrammarPos, token: Token) -> (bool, GrammarPos) {
    let mut p = pos;
    let is_positive_token = elem(rules, p).type_ == Gretype::Token;

    let id = token as u32;
    let mut found = false;
    loop {
        if elem(rules, p.advance(1)).type_ == Gretype::TokenRngUpper {
            // inclusive range, e.g. <[1-5]>
            found = found || (elem(rules, p).value <= id && id <= elem(rules, p.advance(1)).value);
            p = p.advance(2);
        } else {
            found = found || elem(rules, p).value == id;
            p = p.advance(1);
        }
        if elem(rules, p).type_ != Gretype::TokenAlt {
            break;
        }
    }

    (found == is_positive_token, p)
}

// every token id a token rule names, merged into sorted inclusive ranges
fn opaque_tokens(rules: &GrammarRules) -> Vec<(u32, u32)> {
    let mut ranges = Vec::new();
    for rule in rules {
        for (i, elem) in rule.iter().enumerate() {
            let type_ = elem.type_;
            if matches!(
                type_,
                Gretype::Token | Gretype::TokenNot | Gretype::TokenAlt
            ) {
                let is_range = i + 1 < rule.len() && rule[i + 1].type_ == Gretype::TokenRngUpper;
                ranges.push((
                    elem.value,
                    if is_range {
                        rule[i + 1].value
                    } else {
                        elem.value
                    },
                ));
            }
        }
    }
    ranges.sort_unstable();
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for r in ranges {
        if let Some(last) = merged.last_mut() {
            if r.0 as u64 <= last.1 as u64 + 1 {
                last.1 = last.1.max(r.1);
                continue;
            }
        }
        merged.push(r);
    }
    merged
}

// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
fn advance_stack(rules: &GrammarRules, stack: &GrammarStack, new_stacks: &mut GrammarStacks) {
    let mut todo = vec![stack.clone()];
    let mut seen: BTreeSet<GrammarStack> = BTreeSet::new();

    while let Some(curr) = todo.pop() {
        if !seen.insert(curr.clone()) {
            continue;
        }

        if curr.is_empty() {
            if !new_stacks.contains(&curr) {
                new_stacks.push(curr);
            }
            continue;
        }

        let pos = *curr.last().unwrap();
        match elem(rules, pos).type_ {
            Gretype::RuleRef => {
                let rule_id = elem(rules, pos).value as usize;
                let mut subpos = GrammarPos::new(rule_id, 0);
                loop {
                    // init new stack without the top (pos)
                    let mut next_stack = curr.clone();
                    next_stack.pop();
                    if !is_end_of_sequence(elem(rules, pos.advance(1))) {
                        // if this rule ref is followed by another element, add that to stack
                        next_stack.push(pos.advance(1));
                    }
                    if !is_end_of_sequence(elem(rules, subpos)) {
                        // if alternate is nonempty, add to stack
                        next_stack.push(subpos);
                    }
                    todo.push(next_stack);
                    while !is_end_of_sequence(elem(rules, subpos)) {
                        // scan to end of alternate def
                        subpos = subpos.advance(1);
                    }
                    if elem(rules, subpos).type_ == Gretype::Alt {
                        // there's another alternate def of this rule to process
                        subpos = subpos.advance(1);
                    } else {
                        break;
                    }
                }
            }
            Gretype::Char
            | Gretype::CharNot
            | Gretype::CharAny
            | Gretype::Token
            | Gretype::TokenNot => {
                // only add the stack if it's not a duplicate of one we already have
                if !new_stacks.contains(&curr) {
                    new_stacks.push(curr);
                }
            }
            _ => {
                // end of alternate (END, ALT) or middle of char range / token set;
                // stack should never be left on those
                unreachable!("llama_grammar_advance_stack: fatal error");
            }
        }
    }
}

fn accept_chr(
    rules: &GrammarRules,
    stack: &GrammarStack,
    chr: u32,
    new_stacks: &mut GrammarStacks,
) {
    if stack.is_empty() {
        return;
    }

    let pos = *stack.last().unwrap();

    // ignore if this turns into a token
    if matches!(elem(rules, pos).type_, Gretype::Token | Gretype::TokenNot) {
        return;
    }

    let (matched, after) = match_char(rules, pos, chr);
    if matched {
        let mut new_stack = stack.clone();
        new_stack.pop();
        if !is_end_of_sequence(elem(rules, after)) {
            new_stack.push(after);
        }
        advance_stack(rules, &new_stack, new_stacks);
    }
}

fn cp_at(code_points: &[u32], depth: usize) -> u32 {
    code_points.get(depth).copied().unwrap_or(0)
}

// checks the code point at depth of each candidate (an index into the input) against the stack
// returns true when some match: they go to next, and the stacks that follow to next_stacks
fn reject_step(
    rules: &GrammarRules,
    stack: &GrammarStack,
    candidates: &[GrammarCandidate],
    depth: usize,
    cur: &[u32],
    rejects: &mut Vec<u32>,
    next: &mut Vec<u32>,
    next_stacks: &mut GrammarStacks,
) -> bool {
    if stack.is_empty() {
        for &i in cur {
            let tok = &candidates[i as usize];
            if cp_at(&tok.code_points, depth) != 0 || tok.partial_utf8.n_remain != 0 {
                rejects.push(i);
            }
        }
        return false;
    }

    let stack_pos = *stack.last().unwrap();

    // if the top of the stack is a token rule, then we only need to check the token id
    if matches!(
        elem(rules, stack_pos).type_,
        Gretype::Token | Gretype::TokenNot
    ) {
        for &i in cur {
            let tok = &candidates[i as usize];
            if depth > 0 {
                // char rules already consumed part of this token, so it cannot match by id.
                // accept it only if they consumed all of it and did not end in a partial sequence.
                if cp_at(&tok.code_points, depth) != 0 || tok.partial_utf8.n_remain != 0 {
                    rejects.push(i);
                }
            } else if tok.partial_utf8.n_remain < 0 || !match_token(rules, stack_pos, tok.id).0 {
                rejects.push(i);
            }
        }
        return false;
    }

    for &i in cur {
        let tok = &candidates[i as usize];
        if tok.opaque {
            // a token named by a token rule only matches token rules
            rejects.push(i);
        } else if cp_at(&tok.code_points, depth) == 0 {
            // reached end of full codepoints in token, reject iff it ended in a partial sequence
            // that cannot satisfy this position in grammar
            if tok.partial_utf8.n_remain != 0
                && !match_partial_char(rules, stack_pos, tok.partial_utf8)
            {
                rejects.push(i);
            }
        } else if match_char(rules, stack_pos, cp_at(&tok.code_points, depth)).0 {
            next.push(i);
        } else {
            rejects.push(i);
        }
    }

    let stack_pos_after = match_char(rules, stack_pos, 0).1;

    // update top of stack to next element, if any
    let mut stack_after = stack.clone();
    stack_after.pop();
    if !is_end_of_sequence(elem(rules, stack_pos_after)) {
        stack_after.push(stack_pos_after);
    }
    advance_stack(rules, &stack_after, next_stacks);

    true
}

// the candidates that no stack accepts, each stack narrows the rejects of the previous one
// no recursion: a level per code point, a token from the GGUF can be long
fn reject_candidates(
    rules: &GrammarRules,
    stacks: &GrammarStacks,
    candidates: &[GrammarCandidate],
) -> GrammarCandidates {
    // a level checks the candidates that reach its code point against its stacks in turn
    struct Level {
        stacks: GrammarStacks,
        i: usize,          // the stack in progress
        cur: Vec<u32>,     // the candidates rejected by stacks[0, i)
        rejects: Vec<u32>, // the rejects of stacks[i] at this code point
    }

    let all: Vec<u32> = (0..candidates.len() as u32).collect();
    let mut levels = vec![Level {
        stacks: stacks.clone(),
        i: 0,
        cur: all,
        rejects: Vec::new(),
    }];

    loop {
        let done = {
            let lv = levels.last().unwrap();
            lv.i == lv.stacks.len() || lv.cur.is_empty()
        };
        if done {
            if levels.len() == 1 {
                break;
            }
            // the rejects at the next code point belong to stacks[i] of the parent level
            let lv = levels.pop().unwrap();
            let parent = levels.last_mut().unwrap();
            parent.rejects.extend(lv.cur);
            parent.cur = std::mem::take(&mut parent.rejects);
            parent.i += 1;
            continue;
        }

        let depth = levels.len() - 1;
        let mut rejects = Vec::new();
        let mut next = Vec::new();
        let mut next_stacks = GrammarStacks::new();
        let advanced = {
            let lv = levels.last().unwrap();
            reject_step(
                rules,
                &lv.stacks[lv.i],
                candidates,
                depth,
                &lv.cur,
                &mut rejects,
                &mut next,
                &mut next_stacks,
            )
        };
        if !advanced {
            let lv = levels.last_mut().unwrap();
            lv.cur = rejects;
            lv.i += 1;
            continue;
        }

        let lv = levels.last_mut().unwrap();
        lv.rejects = rejects;
        levels.push(Level {
            stacks: next_stacks,
            i: 0,
            cur: next,
            rejects: Vec::new(),
        });
    }

    let mut rejects = GrammarCandidates::with_capacity(levels[0].cur.len());
    for &i in &levels[0].cur {
        rejects.push(candidates[i as usize].clone());
    }
    rejects
}

/// Port of `llama_grammar_reject_candidates_for_stack`.
pub fn reject_candidates_for_stack(
    rules: &[GrammarRule],
    stack: &GrammarStack,
    candidates: &[GrammarCandidate],
) -> GrammarCandidates {
    let stacks = vec![stack.clone()];
    reject_candidates(rules, &stacks, candidates)
}

// ---------------------------------------------------------------------------
// left factoring / left recursion (`struct llama_grammar` init support)
// ---------------------------------------------------------------------------

fn trim_prefix(rule: &GrammarRule, start: usize, mut length: usize) -> usize {
    // Keep the prefix boundary outside character classes.
    while length > 0 {
        let type_ = rule[start + length].type_;
        if type_ != Gretype::CharAlt && type_ != Gretype::CharRngUpper {
            break;
        }
        length -= 1;
    }
    length
}

// Left-factor grammar rules to reduce redundant parallel stacks.
//
// When a rule has many alternates (e.g. tool-choice ::= tc0 | tc1 | ... | tc119)
// and the referenced rules share a common element-content prefix, extract that
// prefix so the parser processes it once instead of N times.
//
// Handles two cases:
// 1. Inline common prefix: rule = "ab1" | "ab2" -> rule = "ab" rule', rule' = "1" | "2"
// 2. Cross-rule prefix: rule = A | B | C where A,B,C start with same elements
//    -> rule = common_prefix rule', rule' = A_suffix | B_suffix | C_suffix
fn left_factor(rules: &mut GrammarRules) {
    let n_rules_original = rules.len();
    // Original rules factor at most once; generated rules do not resolve references.
    // Each new suffix removes a non-empty prefix.
    let mut changed = true;
    while changed {
        changed = false;
        let n_rules = rules.len();

        for ri in 0..n_rules {
            // Split rule into alternates
            let mut alt_ranges: Vec<(usize, usize)> = Vec::new();
            let mut alt_start = 0usize;
            for (j, elem) in rules[ri].iter().enumerate() {
                if matches!(elem.type_, Gretype::Alt | Gretype::End) {
                    alt_ranges.push((alt_start, j));
                    alt_start = j + 1;
                }
            }

            if alt_ranges.len() < 4 {
                continue;
            }

            // Resolve each alternate: if it's a single RULE_REF, follow it to get the
            // actual element sequence from the referenced rule.
            let mut resolved: Vec<(usize, usize, usize)> = Vec::with_capacity(alt_ranges.len());
            let mut all_single_refs = true;

            for &(astart, aend) in &alt_ranges {
                let len = aend - astart;
                // Resolving references in generated suffixes can expand recursive rules forever.
                if ri < n_rules_original && len == 1 && rules[ri][astart].type_ == Gretype::RuleRef
                {
                    let ref_id = rules[ri][astart].value as usize;
                    if ref_id < rules.len() {
                        let mut ref_end = 0usize;
                        for (k, elem) in rules[ref_id].iter().enumerate() {
                            if matches!(elem.type_, Gretype::Alt | Gretype::End) {
                                ref_end = k;
                                break;
                            }
                        }
                        let single_alt = rules[ref_id][ref_end].type_ == Gretype::End;
                        if single_alt {
                            resolved.push((ref_id, 0, ref_end));
                            continue;
                        }
                    }
                }
                all_single_refs = false;
                resolved.push((ri, astart, aend));
            }

            if !all_single_refs {
                resolved = alt_ranges.iter().map(|&(a, b)| (ri, a, b)).collect();
            }

            let mut prefix_len = 0usize;
            let mut prefix_match = true;
            while prefix_match {
                let (r0, s0, e0) = resolved[0];
                if s0 + prefix_len >= e0 {
                    break;
                }
                let ref_elem = rules[r0][s0 + prefix_len];
                for &(rr, ss, ee) in &resolved[1..] {
                    if ss + prefix_len >= ee || rules[rr][ss + prefix_len] != ref_elem {
                        prefix_match = false;
                        break;
                    }
                }
                if prefix_match {
                    prefix_len += 1;
                }
            }

            for &(rr, ss, _) in &resolved {
                prefix_len = trim_prefix(&rules[rr], ss, prefix_len);
            }
            if prefix_len == 0 {
                continue;
            }

            let suffix_choice_id = rules.len() as u32;
            let mut suffix_choice = GrammarRule::new();

            for (ai, &(rr, ss, ee)) in resolved.iter().enumerate() {
                if ai > 0 {
                    suffix_choice.push(GrammarElement::new(Gretype::Alt, 0));
                }
                for j in ss + prefix_len..ee {
                    suffix_choice.push(rules[rr][j]);
                }
            }
            suffix_choice.push(GrammarElement::new(Gretype::End, 0));

            let mut new_rule = GrammarRule::new();
            let (r0, s0, _) = resolved[0];
            for j in 0..prefix_len {
                new_rule.push(rules[r0][s0 + j]);
            }
            new_rule.push(GrammarElement::new(Gretype::RuleRef, suffix_choice_id));
            new_rule.push(GrammarElement::new(Gretype::End, 0));

            rules.push(suffix_choice);
            rules[ri] = new_rule;
            changed = true;
        }
    }
}

fn detect_left_recursion(
    rules: &GrammarRules,
    rule_index: usize,
    rules_visited: &mut [bool],
    rules_in_progress: &mut [bool],
    rules_may_be_empty: &mut [bool],
    depth: u32,
) -> Result<bool> {
    if rules_in_progress[rule_index] {
        return Ok(true);
    }
    if depth > GRAMMAR_RECURSIVE_LIMIT {
        return Err(Error::new(format!(
            "grammar rules are chained more than {} levels deep",
            GRAMMAR_RECURSIVE_LIMIT
        )));
    }

    rules_in_progress[rule_index] = true;
    let rule = &rules[rule_index];

    // First check if the rule might produce the empty string. This could be done combined with the second
    // step but it's more readable as two steps.
    let mut at_rule_start = true;
    for elem in rule {
        if is_end_of_sequence(elem) {
            if at_rule_start {
                rules_may_be_empty[rule_index] = true;
                break;
            }
            at_rule_start = true;
        } else {
            at_rule_start = false;
        }
    }

    // Second, recurse into leftmost nonterminals (or next-leftmost as long as the previous nonterminal may
    // be empty)
    let mut recurse_into_nonterminal = true;
    for elem in rule {
        if elem.type_ == Gretype::RuleRef && recurse_into_nonterminal {
            let id = elem.value as usize;
            if detect_left_recursion(
                rules,
                id,
                rules_visited,
                rules_in_progress,
                rules_may_be_empty,
                depth + 1,
            )? {
                return Ok(true);
            }
            if !rules_may_be_empty[id] {
                recurse_into_nonterminal = false;
            }
        } else if is_end_of_sequence(elem) {
            recurse_into_nonterminal = true;
        } else {
            recurse_into_nonterminal = false;
        }
    }

    rules_in_progress[rule_index] = false;
    rules_visited[rule_index] = true;

    Ok(false)
}

// ---------------------------------------------------------------------------
// grammar struct and public API (`struct llama_grammar`)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Grammar {
    rules: GrammarRules,
    stacks: GrammarStacks,
    // buffer for partially generated UTF-8 sequence from accepted tokens
    partial_utf8: PartialUtf8,
    lazy: bool,
    awaiting_trigger: bool, // initialized to true for lazy grammars only
    trigger_buffer: Vec<u8>,
    trigger_buffer_positions: Vec<(Token, (usize, usize))>,
    trigger_tokens: Vec<Token>,
    trigger_patterns: Vec<TriggerPattern>,
    // token ids named by token rules, as sorted inclusive ranges.
    // char rules never match these tokens.
    opaque_tokens: Vec<(u32, u32)>,
}

impl Grammar {
    // note: allow null vocab for testing (not great) -- there is no vocab here at all
    fn build(
        src_rules: &[GrammarRule],
        start_rule_index: usize,
        lazy: bool,
        trigger_tokens: Vec<Token>,
        trigger_patterns: Vec<TriggerPattern>,
    ) -> Option<Self> {
        // copy rule definitions into vectors
        let mut vec_rules: GrammarRules = src_rules
            .iter()
            .map(|rule| {
                let mut v: GrammarRule = rule
                    .iter()
                    .take_while(|e| e.type_ != Gretype::End)
                    .cloned()
                    .collect();
                v.push(GrammarElement::new(Gretype::End, 0));
                v
            })
            .collect();

        // Validate that all rule references point to valid rules
        for (i, rule) in vec_rules.iter().enumerate() {
            for elem in rule {
                if elem.type_ == Gretype::RuleRef {
                    let id = elem.value as usize;
                    if id >= vec_rules.len() || vec_rules[id].is_empty() {
                        eprintln!(
                            "invalid grammar: rule {} references undefined rule {}",
                            i, elem.value
                        );
                        return None;
                    }
                }
            }
        }

        // Left-factor rules to reduce redundant parallel stacks
        left_factor(&mut vec_rules);

        // Check for left recursion (use post-factoring rule count)
        let n_rules_total = vec_rules.len();
        let mut rules_visited = vec![false; n_rules_total];
        let mut rules_in_progress = vec![false; n_rules_total];
        let mut rules_may_be_empty = vec![false; n_rules_total];
        for i in 0..n_rules_total {
            if rules_visited[i] {
                continue;
            }
            match detect_left_recursion(
                &vec_rules,
                i,
                &mut rules_visited,
                &mut rules_in_progress,
                &mut rules_may_be_empty,
                0,
            ) {
                Ok(true) => {
                    eprintln!(
                        "unsupported grammar, left recursion detected for nonterminal at index {}",
                        i
                    );
                    return None;
                }
                Ok(false) => {}
                Err(err) => {
                    eprintln!("unsupported grammar: {}", err);
                    return None;
                }
            }
        }

        // loop over alternates of start rule to build initial stacks
        let mut stacks = GrammarStacks::new();
        let mut pos = GrammarPos::new(start_rule_index, 0);
        loop {
            let mut stack = GrammarStack::new();
            if !is_end_of_sequence(elem(&vec_rules, pos)) {
                // if alternate is nonempty, add to stack
                stack.push(pos);
            }
            advance_stack(&vec_rules, &stack, &mut stacks);
            while !is_end_of_sequence(elem(&vec_rules, pos)) {
                // scan to end of alternate def
                pos = pos.advance(1);
            }
            if elem(&vec_rules, pos).type_ == Gretype::Alt {
                // there's another alternate def of this rule to process
                pos = pos.advance(1);
            } else {
                break;
            }
        }

        let opaque = opaque_tokens(&vec_rules);

        Some(Self {
            rules: vec_rules,
            stacks,
            partial_utf8: PartialUtf8::default(),
            lazy,
            awaiting_trigger: lazy,
            trigger_buffer: Vec::new(),
            trigger_buffer_positions: Vec::new(),
            trigger_tokens,
            trigger_patterns,
            opaque_tokens: opaque,
        })
    }

    /// Port of the element-array `llama_grammar_init_impl`.
    pub fn from_rules(rules: &[GrammarRule], start_rule_index: usize) -> Option<Self> {
        if start_rule_index >= rules.len() {
            return None;
        }
        Self::build(rules, start_rule_index, false, Vec::new(), Vec::new())
    }

    /// Port of the string `llama_grammar_init_impl`, without a vocab.
    pub fn from_str(
        grammar_str: &str,
        grammar_root: &str,
        lazy: bool,
        trigger_patterns: &[String],
        trigger_tokens: &[Token],
    ) -> Option<Self> {
        Self::from_str_with_tokenizer_opt(
            grammar_str,
            grammar_root,
            lazy,
            trigger_patterns,
            trigger_tokens,
            None,
        )
    }

    /// Same as `from_str`, with a tokenizer for `<token>` rules.
    pub fn from_str_with_tokenizer<F>(
        grammar_str: &str,
        grammar_root: &str,
        lazy: bool,
        trigger_patterns: &[String],
        trigger_tokens: &[Token],
        tokenize: F,
    ) -> Option<Self>
    where
        F: Fn(&[u8]) -> Result<Vec<Token>> + 'static,
    {
        Self::from_str_with_tokenizer_opt(
            grammar_str,
            grammar_root,
            lazy,
            trigger_patterns,
            trigger_tokens,
            Some(Box::new(tokenize)),
        )
    }

    fn from_str_with_tokenizer_opt(
        grammar_str: &str,
        grammar_root: &str,
        lazy: bool,
        trigger_patterns: &[String],
        trigger_tokens: &[Token],
        tokenize: Option<Box<TokenizeFn>>,
    ) -> Option<Self> {
        let mut parser = GrammarParser::with_tokenizer_opt(tokenize);

        // if there is a grammar, parse it
        // rules will be empty (default) if there are parse errors
        if !parser.parse(grammar_str) || parser.rules.is_empty() {
            eprintln!("failed to parse grammar");
            return None;
        }

        // Ensure that the grammar contains the start symbol
        let start_rule_index = match parser.symbol_ids.get(grammar_root) {
            Some(id) => *id as usize,
            None => {
                eprintln!("grammar does not contain a '{}' symbol", grammar_root);
                return None;
            }
        };

        let rules = std::mem::take(&mut parser.rules);
        let patterns = trigger_patterns
            .iter()
            .map(|pattern| TriggerPattern {
                pattern: pattern.clone(),
            })
            .collect();

        Self::build(
            &rules,
            start_rule_index,
            lazy,
            trigger_tokens.to_vec(),
            patterns,
        )
    }

    /// Port of `llama_grammar_get_rules`.
    pub fn rules(&self) -> &GrammarRules {
        &self.rules
    }

    /// Port of `llama_grammar_get_stacks`.
    pub fn stacks(&self) -> &GrammarStacks {
        &self.stacks
    }

    pub fn stacks_mut(&mut self) -> &mut GrammarStacks {
        &mut self.stacks
    }

    /// true if a token rule names the token, char rules do not match it
    pub fn is_opaque(&self, token: Token) -> bool {
        let id = token as u32;
        let it = self.opaque_tokens.partition_point(|&(lo, _)| lo <= id);
        it > 0 && self.opaque_tokens[it - 1].1 >= id
    }

    /// Port of `llama_grammar_apply_impl`.
    pub fn apply(&self, cur: &mut [TokenData]) {
        if self.awaiting_trigger {
            return;
        }

        let mut allow_eog = false;
        for stack in &self.stacks {
            if stack.is_empty() {
                allow_eog = true;
                break;
            }
        }

        // In C++ `candidates_decoded` keeps the pointers in `candidates_grammar` alive.
        let mut candidates = GrammarCandidates::with_capacity(cur.len());
        for (i, data) in cur.iter_mut().enumerate() {
            if data.is_eog {
                if !allow_eog {
                    data.logit = f32::NEG_INFINITY;
                }
            } else if data.piece.is_empty() || data.piece[0] == 0 {
                data.logit = f32::NEG_INFINITY;
            } else {
                let decoded = decode_utf8(&data.piece, self.partial_utf8);
                candidates.push(GrammarCandidate {
                    index: i,
                    code_points: decoded.0,
                    partial_utf8: decoded.1,
                    id: data.id,
                    opaque: self.is_opaque(data.id),
                });
            }
        }

        let rejects = reject_candidates(&self.rules, &self.stacks, &candidates);
        for reject in rejects {
            cur[reject.index].logit = f32::NEG_INFINITY;
        }
    }

    /// Port of `llama_grammar_accept`.
    pub fn accept(&mut self, chr: u32) {
        let mut stacks_new = GrammarStacks::with_capacity(self.stacks.len());
        for stack in &self.stacks {
            accept_chr(&self.rules, stack, chr, &mut stacks_new);
        }
        self.stacks = stacks_new;
    }

    /// Port of `llama_grammar_accept_impl`. `piece` and `is_eog` come from the vocab in C++.
    pub fn accept_impl(&mut self, token: Token, piece: &[u8], is_eog: bool) -> Result<()> {
        if self.awaiting_trigger {
            if self.trigger_tokens.contains(&token) {
                self.awaiting_trigger = false;
                self.trigger_buffer.clear();
                self.accept_token(token, piece)?;
                return Ok(());
            }

            let start = self.trigger_buffer.len();
            let end = start + piece.len();
            self.trigger_buffer_positions.push((token, (start, end)));
            self.trigger_buffer.extend_from_slice(piece);

            let found = {
                let input = String::from_utf8_lossy(&self.trigger_buffer);
                self.trigger_patterns.iter().find_map(|p| p.find(&input))
            };
            if let Some(start) = found {
                self.awaiting_trigger = false;

                // replay tokens that overlap with [start, end)
                let positions = std::mem::take(&mut self.trigger_buffer_positions);
                let buffer = std::mem::take(&mut self.trigger_buffer);
                for (tok, (tok_start, tok_end)) in positions {
                    if tok_end <= start {
                        continue;
                    }
                    let piece_start = tok_start.max(start); // allow for partial token pieces
                    let piece_len = tok_end - piece_start;
                    self.accept_token(tok, &buffer[piece_start..piece_start + piece_len])?;
                }
            }
            return Ok(());
        }

        if is_eog {
            for stack in &self.stacks {
                if stack.is_empty() {
                    return Ok(());
                }
            }
            return Err(Error::new("fatal error: EOG token with no empty stack"));
        }

        self.accept_token(token, piece)
    }

    /// Port of `llama_grammar_accept_str`.
    pub fn accept_str(&mut self, piece: &[u8]) -> Result<()> {
        // Note terminating 0 in decoded string
        let decoded = decode_utf8(piece, self.partial_utf8);
        let code_points = decoded.0;
        for &chr in code_points.iter().take(code_points.len().saturating_sub(1)) {
            self.accept(chr);
        }
        self.partial_utf8 = decoded.1;
        if self.stacks.is_empty() {
            return Err(Error::new(format!(
                "Unexpected empty grammar stack after accepting piece: {}",
                String::from_utf8_lossy(piece)
            )));
        }
        Ok(())
    }

    /// Port of `llama_grammar_accept_token`.
    pub fn accept_token(&mut self, token: Token, piece: &[u8]) -> Result<()> {
        // Note terminating 0 in decoded string
        let decoded = decode_utf8(piece, self.partial_utf8);
        let code_points = decoded.0;

        // the mask rejects a piece that does not continue a partial character, so accepting one has to fail too
        if decoded.1.n_remain < 0 {
            return Err(Error::new(format!(
                "Invalid UTF-8 continuation in piece: {} ({})",
                String::from_utf8_lossy(piece),
                token
            )));
        }

        let opaque = self.is_opaque(token);
        let mut stacks_new = GrammarStacks::with_capacity(self.stacks.len());

        for stack in &self.stacks {
            if stack.is_empty() {
                continue;
            }

            let pos = *stack.last().unwrap();
            if matches!(
                elem(&self.rules, pos).type_,
                Gretype::Token | Gretype::TokenNot
            ) {
                let (matched, after) = match_token(&self.rules, pos, token);
                if matched {
                    let mut new_stack = stack.clone();
                    new_stack.pop();
                    if !is_end_of_sequence(elem(&self.rules, after)) {
                        new_stack.push(after);
                    }
                    advance_stack(&self.rules, &new_stack, &mut stacks_new);
                }
            } else if !opaque {
                let mut current_stacks = vec![stack.clone()];
                for &chr in code_points.iter().take(code_points.len().saturating_sub(1)) {
                    let mut next_stacks = GrammarStacks::new();
                    for cur_stack in &current_stacks {
                        accept_chr(&self.rules, cur_stack, chr, &mut next_stacks);
                    }
                    current_stacks = next_stacks;
                    if current_stacks.is_empty() {
                        break;
                    }
                }
                for surviving_stack in current_stacks {
                    if !stacks_new.contains(&surviving_stack) {
                        stacks_new.push(surviving_stack);
                    }
                }
            }
        }

        self.stacks = stacks_new;
        self.partial_utf8 = decoded.1;
        if self.stacks.is_empty() {
            return Err(Error::new(format!(
                "Unexpected empty grammar stack after accepting piece: {} ({})",
                String::from_utf8_lossy(piece),
                token
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// tests for pure helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_utf8_handles_split_and_invalid_sequences() {
        // 'é' is C3 A9, split across two pieces
        let (cps, partial) = decode_utf8(&[0xC3], PartialUtf8::default());
        assert_eq!(cps, vec![0]);
        assert_eq!(partial.value, 3);
        assert_eq!(partial.n_remain, 1);

        let (cps, partial) = decode_utf8(&[0xA9], partial);
        assert_eq!(cps, vec![0xE9, 0]);
        assert_eq!(partial.n_remain, 0);

        let (cps, bad) = decode_utf8(&[0xC3, 0x41], PartialUtf8::default());
        assert_eq!(cps, vec![0]);
        assert_eq!(bad.n_remain, -1);
    }

    #[test]
    fn parse_char_decodes_escapes() {
        let src = b"\\x41\\u00e9\\n\\]x";
        let (v, p) = parse_char(src, 0).unwrap();
        assert_eq!((v, p), (0x41, 4));
        let (v, p) = parse_char(src, 4).unwrap();
        assert_eq!((v, p), (0xe9, 10));
        let (v, p) = parse_char(src, 10).unwrap();
        assert_eq!((v, p), (b'\n' as u32, 12));
        let (v, p) = parse_char(src, 12).unwrap();
        assert_eq!((v, p), (b']' as u32, 14));
    }
}

//! Rust port of the remaining small `common/` helpers.
//!
//! Sources: `trie.{h,cpp}`, `base64.hpp`, `fit.{h,cpp}`, `reasoning-budget.{h,cpp}`,
//! `preset.{h,cpp}`, `chat-tts-norm.{h,cpp}`, `imatrix-loader.{h,cpp}`.
//!
//! Pure-data parts are ported in full. Helpers that need a live model, context,
//! sampler handle or jinja context are reported as omissions instead of being
//! replaced by lookalikes that would not preserve the C++ behaviour.

use crate::common::{string_remove_suffix, ContextParams, ModelParams, Params};
use crate::core::{Error, Result, Token};
use crate::gguf::{
    gguf_find_key, gguf_get_arr_n, gguf_get_arr_str, gguf_get_arr_type, gguf_get_kv_type,
    gguf_get_n_tensors, gguf_get_val_u32, gguf_init_from_file, GgmlContext, GgmlType,
    GgufInitParams, GgufType,
};
use crate::sampler::TokenDataArray;
use crate::vocab::LlamaVocab;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;

// ---------------------------------------------------------------------------
// UTF-8 streaming helpers (common/unicode.{h,cpp}, needed by trie and TTS)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Utf8ParseStatus {
    Success,
    Incomplete,
    Invalid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Utf8ParseResult {
    pub codepoint: u32,
    pub bytes_consumed: usize,
    pub status: Utf8ParseStatus,
}

impl Utf8ParseResult {
    fn new(status: Utf8ParseStatus, codepoint: u32, bytes_consumed: usize) -> Self {
        Self {
            codepoint,
            bytes_consumed,
            status,
        }
    }
}

/// `common_utf8_sequence_length`.
pub fn utf8_sequence_length(first_byte: u8) -> usize {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[(first_byte >> 4) as usize]
}

/// `common_parse_utf8_codepoint`.
pub fn parse_utf8_codepoint(input: &[u8], offset: usize) -> Utf8ParseResult {
    if offset >= input.len() {
        return Utf8ParseResult::new(Utf8ParseStatus::Incomplete, 0, 0);
    }
    let b = input[offset];
    if b & 0x80 == 0 {
        return Utf8ParseResult::new(Utf8ParseStatus::Success, b as u32, 1);
    }
    if b & 0x40 == 0 {
        return Utf8ParseResult::new(Utf8ParseStatus::Invalid, 0, 1);
    }
    if b & 0x20 == 0 {
        if offset + 1 >= input.len() {
            return Utf8ParseResult::new(Utf8ParseStatus::Incomplete, 0, 1);
        }
        if input[offset + 1] & 0xc0 != 0x80 {
            return Utf8ParseResult::new(Utf8ParseStatus::Invalid, 0, 1);
        }
        let cp = ((b as u32 & 0x1f) << 6) | (input[offset + 1] as u32 & 0x3f);
        return Utf8ParseResult::new(Utf8ParseStatus::Success, cp, 2);
    }
    if b & 0x10 == 0 {
        for i in 1..3 {
            if offset + i >= input.len() {
                return Utf8ParseResult::new(Utf8ParseStatus::Incomplete, 0, i);
            }
            if input[offset + i] & 0xc0 != 0x80 {
                return Utf8ParseResult::new(Utf8ParseStatus::Invalid, 0, i);
            }
        }
        let cp = ((b as u32 & 0x0f) << 12)
            | ((input[offset + 1] as u32 & 0x3f) << 6)
            | (input[offset + 2] as u32 & 0x3f);
        return Utf8ParseResult::new(Utf8ParseStatus::Success, cp, 3);
    }
    if b & 0x08 == 0 {
        for i in 1..4 {
            if offset + i >= input.len() {
                return Utf8ParseResult::new(Utf8ParseStatus::Incomplete, 0, i);
            }
            if input[offset + i] & 0xc0 != 0x80 {
                return Utf8ParseResult::new(Utf8ParseStatus::Invalid, 0, i);
            }
        }
        let cp = ((b as u32 & 0x07) << 18)
            | ((input[offset + 1] as u32 & 0x3f) << 12)
            | ((input[offset + 2] as u32 & 0x3f) << 6)
            | (input[offset + 3] as u32 & 0x3f);
        return Utf8ParseResult::new(Utf8ParseStatus::Success, cp, 4);
    }
    Utf8ParseResult::new(Utf8ParseStatus::Invalid, 0, 1)
}

/// `common_utf8_is_complete`.
pub fn utf8_is_complete(s: &[u8]) -> bool {
    if s.is_empty() {
        return true;
    }
    for i in 1..=4.min(s.len()) {
        let c = s[s.len() - i];
        if c & 0xc0 != 0x80 {
            let expected = if c >= 0xf0 {
                4
            } else if c >= 0xe0 {
                3
            } else if c >= 0xc0 {
                2
            } else {
                1
            };
            return i >= expected;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// common_trie / common_aho_corasick
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrieSymbolKind {
    Codepoint,
    Token,
}

/// `common_trie::symbol`. Derived ordering matches the C++ `operator<`:
/// kind first, then value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TrieSymbol {
    pub kind: TrieSymbolKind,
    pub value: u32,
}

impl TrieSymbol {
    pub fn codepoint(cpt: u32) -> Self {
        Self {
            kind: TrieSymbolKind::Codepoint,
            value: cpt,
        }
    }

    pub fn token(id: Token) -> Self {
        Self {
            kind: TrieSymbolKind::Token,
            value: id as u32,
        }
    }

    pub fn is_token(self) -> bool {
        self.kind == TrieSymbolKind::Token
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrieStep {
    pub status: Utf8ParseStatus,
    pub symbol: TrieSymbol,
    pub bytes_consumed: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchResult {
    NoMatch,
    PartialMatch,
    CompleteMatch,
}

#[derive(Clone, Debug, Default)]
pub struct TrieNode {
    pub children: BTreeMap<TrieSymbol, usize>,
    pub pattern: i32,
}

/// Trie for matching multiple literals, with optional token symbols.
#[derive(Clone, Debug, Default)]
pub struct Trie {
    pub nodes: Vec<TrieNode>,
    pub tokens: BTreeSet<i32>,
    n_patterns: i32,
}

impl Trie {
    pub fn new() -> Self {
        let mut trie = Self::default();
        trie.create_node();
        trie
    }

    pub fn from_words(words: &[String]) -> Self {
        let mut trie = Self::new();
        for w in words {
            trie.insert(w);
        }
        trie
    }

    fn create_node(&mut self) -> usize {
        let index = self.nodes.len();
        self.nodes.push(TrieNode {
            children: BTreeMap::new(),
            pattern: -1,
        });
        index
    }

    /// `next`: a token symbol when a pattern uses the token at `pos`, else a
    /// UTF-8 codepoint.
    pub fn next(&self, sv: &[u8], token_map: &[i32], pos: usize) -> TrieStep {
        if !self.tokens.is_empty() && pos < token_map.len() {
            let id = token_map[pos];
            if id >= 0 && self.tokens.contains(&id) {
                let mut end = pos + 1;
                while end < token_map.len() && token_map[end] < 0 {
                    end += 1;
                }
                return TrieStep {
                    status: Utf8ParseStatus::Success,
                    symbol: TrieSymbol::token(id),
                    bytes_consumed: end - pos,
                };
            }
        }
        let result = parse_utf8_codepoint(sv, pos);
        TrieStep {
            status: result.status,
            symbol: TrieSymbol::codepoint(result.codepoint),
            bytes_consumed: result.bytes_consumed,
        }
    }

    pub fn check_at(&self, sv: &[u8], start_pos: usize) -> MatchResult {
        self.check_at_with_tokens(sv, &[], start_pos)
    }

    pub fn check_at_with_tokens(
        &self,
        sv: &[u8],
        token_map: &[i32],
        start_pos: usize,
    ) -> MatchResult {
        let mut current = 0usize;
        let mut pos = start_pos;
        while pos < sv.len() {
            let step = self.next(sv, token_map, pos);
            if step.status != Utf8ParseStatus::Success {
                break;
            }
            let Some(&child) = self.nodes[current].children.get(&step.symbol) else {
                return MatchResult::NoMatch;
            };
            current = child;
            pos += step.bytes_consumed;
            if self.nodes[current].pattern >= 0 {
                return MatchResult::CompleteMatch;
            }
        }
        if current != 0 {
            MatchResult::PartialMatch
        } else {
            MatchResult::NoMatch
        }
    }

    /// Insert a word as a sequence of UTF-8 codepoints.
    pub fn insert(&mut self, word: &str) -> i32 {
        let mut symbols = Vec::new();
        let bytes = word.as_bytes();
        let mut pos = 0usize;
        while pos < bytes.len() {
            let result = parse_utf8_codepoint(bytes, pos);
            if result.status != Utf8ParseStatus::Success {
                break;
            }
            symbols.push(TrieSymbol::codepoint(result.codepoint));
            pos += result.bytes_consumed;
        }
        self.insert_symbols(&symbols)
    }

    /// Insert a symbol sequence; duplicates keep the first pattern index.
    pub fn insert_symbols(&mut self, symbols: &[TrieSymbol]) -> i32 {
        let mut current = 0usize;
        for &ch in symbols {
            current = self.extend(current, ch);
        }
        let pattern = self.mark(current, self.n_patterns);
        if pattern == self.n_patterns {
            self.n_patterns += 1;
        }
        pattern
    }

    pub fn extend(&mut self, node: usize, ch: TrieSymbol) -> usize {
        if ch.is_token() {
            self.tokens.insert(ch.value as i32);
        }
        if let Some(&child) = self.nodes[node].children.get(&ch) {
            return child;
        }
        let child = self.create_node();
        self.nodes[node].children.insert(ch, child);
        child
    }

    pub fn mark(&mut self, node: usize, value: i32) -> i32 {
        if self.nodes[node].pattern < 0 {
            self.nodes[node].pattern = value;
        }
        self.nodes[node].pattern
    }
}

/// Aho-Corasick automaton built from a `Trie`.
#[derive(Clone, Debug)]
pub struct AhoCorasick {
    pub trie: Trie,
    pub fail: Vec<usize>,
    pub order: Vec<usize>,
    pub matched: Vec<i32>,
    pub alphabet: BTreeSet<TrieSymbol>,
}

impl AhoCorasick {
    pub fn new(trie: Trie) -> Self {
        let n = trie.nodes.len();
        let mut fail = vec![0usize; n];
        let mut order = Vec::with_capacity(n);
        let mut queue = VecDeque::from([0usize]);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for (&ch, &v) in &trie.nodes[u].children {
                if u != 0 {
                    let mut f = fail[u];
                    while f != 0 && !trie.nodes[f].children.contains_key(&ch) {
                        f = fail[f];
                    }
                    fail[v] = match trie.nodes[f].children.get(&ch) {
                        Some(&candidate) if candidate != v => candidate,
                        _ => 0,
                    };
                }
                queue.push_back(v);
            }
        }

        let mut matched = vec![-1i32; n];
        for &u in &order {
            matched[u] = if trie.nodes[u].pattern >= 0 {
                trie.nodes[u].pattern
            } else if u != 0 {
                matched[fail[u]]
            } else {
                -1
            };
        }

        let mut alphabet = BTreeSet::new();
        for node in &trie.nodes {
            for ch in node.children.keys() {
                alphabet.insert(*ch);
            }
        }

        Self {
            trie,
            fail,
            order,
            matched,
            alphabet,
        }
    }

    pub fn from_words(words: &[String]) -> Self {
        Self::new(Trie::from_words(words))
    }

    pub fn num_states(&self) -> usize {
        self.trie.nodes.len()
    }

    pub fn is_terminal(&self, state: usize) -> bool {
        self.matched[state] >= 0
    }

    pub fn match_pattern(&self, state: usize) -> i32 {
        self.matched[state]
    }

    pub fn next(&self, state: usize, ch: TrieSymbol) -> usize {
        let mut state = state;
        while state != 0 && !self.trie.nodes[state].children.contains_key(&ch) {
            state = self.fail[state];
        }
        self.trie.nodes[state]
            .children
            .get(&ch)
            .copied()
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// base64.hpp
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Base64Alphabet {
    Auto,
    Standard,
    UrlFilenameSafe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Base64Decoding {
    Moderate,
    Loose,
}

const BASE64_STANDARD: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const BASE64_URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// `base64::required_encode_size`.
pub fn base64_required_encode_size(size: usize) -> usize {
    (size / 3 + usize::from(size % 3 != 0)) * 4
}

/// `base64::max_decode_size`.
pub fn base64_max_decode_size(size: usize) -> usize {
    (size / 4 + usize::from(size % 4 != 0)) * 3
}

fn base64_alpha(alpha: Base64Alphabet) -> &'static [u8; 64] {
    if alpha == Base64Alphabet::UrlFilenameSafe {
        BASE64_URL
    } else {
        BASE64_STANDARD
    }
}

/// `base64::encode` for a byte buffer.
pub fn base64_encode(input: &[u8], alphabet: Base64Alphabet) -> String {
    let alpha = base64_alpha(alphabet);
    let mut out = Vec::with_capacity(base64_required_encode_size(input.len()));
    let mut iter = input.iter().copied();
    while let Some(i0) = iter.next() {
        out.push(alpha[((i0 >> 2) & 0x3f) as usize]);
        let Some(i1) = iter.next() else {
            out.push(alpha[((i0 & 0x3) << 4) as usize]);
            out.push(b'=');
            out.push(b'=');
            break;
        };
        out.push(alpha[(((i0 & 0x3) << 4) | (i1 >> 4 & 0x0f)) as usize]);
        let Some(i2) = iter.next() else {
            out.push(alpha[((i1 & 0xf) << 2) as usize]);
            out.push(b'=');
            break;
        };
        out.push(alpha[(((i1 & 0xf) << 2) | (i2 >> 6 & 0x03)) as usize]);
        out.push(alpha[(i2 & 0x3f) as usize]);
    }
    String::from_utf8(out).expect("base64 alphabet is ASCII")
}

fn base64_value(alphabet: &mut Base64Alphabet, c: u8) -> Result<u8> {
    if c.is_ascii_uppercase() {
        return Ok(c - b'A');
    }
    if c.is_ascii_lowercase() {
        return Ok(c - b'a' + 26);
    }
    if c.is_ascii_digit() {
        return Ok(c - b'0' + 52);
    }
    match (*alphabet, c) {
        (Base64Alphabet::Standard, b'+') => Ok(62),
        (Base64Alphabet::Standard, b'/') => Ok(63),
        (Base64Alphabet::UrlFilenameSafe, b'-') => Ok(62),
        (Base64Alphabet::UrlFilenameSafe, b'_') => Ok(63),
        (Base64Alphabet::Auto, b'+' | b'-') => {
            if c == b'+' {
                *alphabet = Base64Alphabet::Standard;
            } else {
                *alphabet = Base64Alphabet::UrlFilenameSafe;
            }
            Ok(62)
        }
        (Base64Alphabet::Auto, b'/' | b'_') => {
            if c == b'/' {
                *alphabet = Base64Alphabet::Standard;
            } else {
                *alphabet = Base64Alphabet::UrlFilenameSafe;
            }
            Ok(63)
        }
        _ => Err(Error::new("invalid base64 character.")),
    }
}

/// `base64::decode` for a byte buffer.
pub fn base64_decode(
    input: &[u8],
    mut alphabet: Base64Alphabet,
    behavior: Base64Decoding,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(base64_max_decode_size(input.len()));
    let mut last = 0u8;
    let mut bits = 0u32;
    let mut pos = 0usize;
    while pos < input.len() {
        let c = input[pos];
        pos += 1;
        if c == b'=' {
            break;
        }
        let part = base64_value(&mut alphabet, c)?;
        if bits + 6 >= 8 {
            out.push(last.wrapping_shl(8 - bits) | (part >> (bits - 2)));
            bits -= 2;
        } else {
            bits += 6;
        }
        last = part;
    }
    if behavior != Base64Decoding::Loose {
        while pos < input.len() {
            if input[pos] != b'=' {
                return Err(Error::new("invalid base64 character."));
            }
            pos += 1;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// fit.{h,cpp}: data model and pure table formatting
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum FitStatus {
    Success = 0,
    Failure = 1,
    Error = 2,
}

/// `common_fit_extra_model`, with device handles kept as indices.
#[derive(Clone, Debug, Default)]
pub struct FitExtraModel {
    pub paths_model: Vec<String>,
    pub mparams: ModelParams,
    pub cparams: ContextParams,
    pub shares_model: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryBreakdownKind {
    Device,
    Host,
    BufferType,
}

#[derive(Clone, Debug, Default)]
pub struct MemoryBreakdownRow {
    pub kind: MemoryBreakdownKind,
    pub name: String,
    pub description: String,
    pub total: i64,
    pub free: i64,
    pub model: usize,
    pub context: usize,
    pub compute: usize,
}

impl MemoryBreakdownRow {
    pub fn total_bytes(&self) -> usize {
        self.model + self.context + self.compute
    }
}

impl Default for MemoryBreakdownKind {
    fn default() -> Self {
        Self::Device
    }
}

#[derive(Clone, Debug, Default)]
pub struct MemoryReports {
    pub n_ctx_train: u32,
    pub n_ctx_max: u32,
    pub n_ctx_min: u32,
    pub info_max: Vec<MemoryBreakdownRow>,
    pub info_min: Vec<MemoryBreakdownRow>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceMemoryData {
    pub total: i64,
    pub free: i64,
    pub model: usize,
    pub context: usize,
    pub compute: usize,
}

/// The table that `common_memory_breakdown_print` logs, returned as text.
pub fn format_memory_breakdown(rows: &[MemoryBreakdownRow]) -> String {
    const MIB: i64 = 1024 * 1024;
    // tag: 0 = header, 1 = device, 2 = host/buffer type
    let mut table: Vec<(u8, [String; 9])> = Vec::with_capacity(rows.len() + 1);
    table.push((
        0,
        [
            String::new(),
            "memory breakdown [MiB]".into(),
            "total".into(),
            "free".into(),
            "self".into(),
            "model".into(),
            "context".into(),
            "compute".into(),
            "unaccounted".into(),
        ],
    ));
    for row in rows {
        let self_mib = (row.total_bytes() as i64) / MIB;
        if row.kind == MemoryBreakdownKind::Device {
            let unaccounted = row.total - row.free - row.total_bytes() as i64;
            table.push((
                1,
                [
                    String::new(),
                    format!("  - {} ({})", row.name, row.description),
                    (row.total / MIB).to_string(),
                    (row.free / MIB).to_string(),
                    self_mib.to_string(),
                    (row.model as i64 / MIB).to_string(),
                    (row.context as i64 / MIB).to_string(),
                    (row.compute as i64 / MIB).to_string(),
                    (unaccounted / MIB).to_string(),
                ],
            ));
            continue;
        }
        table.push((
            2,
            [
                String::new(),
                format!("  - {}", row.name),
                String::new(),
                String::new(),
                self_mib.to_string(),
                (row.model as i64 / MIB).to_string(),
                (row.context as i64 / MIB).to_string(),
                (row.compute as i64 / MIB).to_string(),
                String::new(),
            ],
        ));
    }

    for j in 1..9 {
        let max_len = table.iter().map(|(_, t)| t[j].len()).max().unwrap_or(0);
        for (_, t) in &mut table {
            let pad = " ".repeat(max_len - t[j].len());
            if j == 1 {
                t[j].push_str(&pad);
            } else {
                t[j].insert_str(0, &pad);
            }
        }
    }

    let mut out = String::new();
    for (tag, t) in &table {
        let name = &t[1];
        let line = match tag {
            0 => format!(
                "common_memory_breakdown_print: | {name} | {}   {}    {}   {}   {}   {}    {} |\n",
                t[2], t[3], t[4], t[5], t[6], t[7], t[8]
            ),
            1 => format!(
                "common_memory_breakdown_print: | {name} | {} = {} + ({} = {} + {} + {}) + {} |\n",
                t[2], t[3], t[4], t[5], t[6], t[7], t[8]
            ),
            _ => format!(
                "common_memory_breakdown_print: | {name} | {}   {}    {} = {} + {} + {}    {} |\n",
                t[2], t[3], t[4], t[5], t[6], t[7], t[8]
            ),
        };
        out.push_str(&line);
    }
    out
}

// ---------------------------------------------------------------------------
// reasoning-budget.{h,cpp}
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningBudgetState {
    Idle,
    Counting,
    Forcing,
    WaitingUtf8,
    Done,
}

#[derive(Clone, Debug)]
struct TokenMatcher {
    seqs: Vec<Vec<Token>>,
    ac: AhoCorasick,
    state: usize,
}

impl TokenMatcher {
    fn new(seqs: &[Vec<Token>]) -> Self {
        let mut collected: Vec<Vec<Token>> = Vec::new();
        for seq in seqs {
            if !seq.is_empty() && !collected.contains(seq) {
                collected.push(seq.clone());
            }
        }
        let mut trie = Trie::new();
        for seq in &collected {
            let symbols: Vec<TrieSymbol> = seq.iter().map(|&t| TrieSymbol::token(t)).collect();
            trie.insert_symbols(&symbols);
        }
        Self {
            seqs: collected,
            ac: AhoCorasick::new(trie),
            state: 0,
        }
    }

    fn advance(&mut self, token: Token) -> i32 {
        self.state = self.ac.next(self.state, TrieSymbol::token(token));
        let p = self.ac.match_pattern(self.state);
        if p >= 0 {
            self.state = 0;
        }
        p
    }

    fn reset(&mut self) {
        self.state = 0;
    }
}

/// State machine limiting token generation inside a reasoning block.
#[derive(Clone, Debug)]
pub struct ReasoningBudget {
    start_matcher: TokenMatcher,
    end_matcher: TokenMatcher,
    forced_tokens: Vec<Token>,
    budget: i32,
    remaining: i32,
    state: ReasoningBudgetState,
    force_pos: usize,
    end_match: i32,
}

impl ReasoningBudget {
    pub fn new(
        start_seqs: &[Vec<Token>],
        end_seqs: &[Vec<Token>],
        forced_tokens: &[Token],
        budget: i32,
        mut initial_state: ReasoningBudgetState,
    ) -> Self {
        if initial_state == ReasoningBudgetState::Counting && budget <= 0 {
            initial_state = ReasoningBudgetState::Forcing;
        }
        Self {
            start_matcher: TokenMatcher::new(start_seqs),
            end_matcher: TokenMatcher::new(end_seqs),
            forced_tokens: forced_tokens.to_vec(),
            budget,
            remaining: budget,
            state: initial_state,
            force_pos: 0,
            end_match: -1,
        }
    }

    /// `common_reasoning_budget_accept`. `vocab` is optional: without it the
    /// piece is treated as UTF-8 complete, matching a null `llama_vocab`.
    pub fn accept(&mut self, token: Token, vocab: Option<&LlamaVocab>) {
        match self.state {
            ReasoningBudgetState::Idle => {
                if self.start_matcher.advance(token) >= 0 {
                    self.state = ReasoningBudgetState::Counting;
                    self.remaining = self.budget;
                    if self.remaining <= 0 {
                        self.state = ReasoningBudgetState::Forcing;
                        self.force_pos = 0;
                    }
                }
            }
            ReasoningBudgetState::Counting | ReasoningBudgetState::WaitingUtf8 => {
                let matched = self.end_matcher.advance(token);
                if matched >= 0 {
                    self.state = ReasoningBudgetState::Done;
                    self.end_match = matched;
                    return;
                }
                let mut utf8_complete = true;
                if let Some(vocab) = vocab {
                    let piece = crate::common::common_token_to_piece(vocab, token, false);
                    utf8_complete = utf8_is_complete(piece.as_bytes());
                }
                if self.state == ReasoningBudgetState::WaitingUtf8 {
                    if utf8_complete {
                        self.state = ReasoningBudgetState::Forcing;
                        self.force_pos = 0;
                        self.end_matcher.reset();
                    }
                } else {
                    self.remaining -= 1;
                    if self.remaining <= 0 {
                        if utf8_complete {
                            self.state = ReasoningBudgetState::Forcing;
                            self.force_pos = 0;
                            self.end_matcher.reset();
                        } else {
                            self.state = ReasoningBudgetState::WaitingUtf8;
                            self.end_matcher.reset();
                        }
                    }
                }
            }
            ReasoningBudgetState::Forcing => {
                let matched = self.end_matcher.advance(token);
                self.force_pos += 1;
                if matched >= 0 || self.force_pos >= self.forced_tokens.len() {
                    self.state = ReasoningBudgetState::Done;
                    self.end_match = matched;
                }
            }
            ReasoningBudgetState::Done => {
                if self.start_matcher.advance(token) >= 0 {
                    self.state = ReasoningBudgetState::Counting;
                    self.remaining = self.budget;
                    self.end_matcher.reset();
                    self.end_match = -1;
                    if self.remaining <= 0 {
                        self.state = ReasoningBudgetState::Forcing;
                        self.force_pos = 0;
                    }
                }
            }
        }
    }

    /// `common_reasoning_budget_apply`.
    pub fn apply(&self, cur_p: &mut TokenDataArray) {
        if self.state != ReasoningBudgetState::Forcing || self.force_pos >= self.forced_tokens.len()
        {
            return;
        }
        let forced = self.forced_tokens[self.force_pos];
        for candidate in &mut cur_p.data {
            if candidate.id != forced {
                candidate.logit = f32::NEG_INFINITY;
            }
        }
    }

    pub fn reset(&mut self) {
        self.state = ReasoningBudgetState::Idle;
        self.remaining = self.budget;
        self.start_matcher.reset();
        self.end_matcher.reset();
        self.force_pos = 0;
        self.end_match = -1;
    }

    /// Manual IDLE-independent transition; only valid while COUNTING.
    pub fn force(&mut self) -> bool {
        if self.state != ReasoningBudgetState::Counting {
            return false;
        }
        self.state = ReasoningBudgetState::Forcing;
        self.force_pos = 0;
        self.end_matcher.reset();
        true
    }

    pub fn state(&self) -> ReasoningBudgetState {
        self.state
    }

    pub fn end_match(&self) -> Option<&[Token]> {
        if self.end_match < 0 {
            return None;
        }
        self.end_matcher
            .seqs
            .get(self.end_match as usize)
            .map(Vec::as_slice)
    }
}

// ---------------------------------------------------------------------------
// preset.{h,cpp}: INI presets (no jinja / download-dependent loaders)
// ---------------------------------------------------------------------------

/// `common_arg_utils::is_truthy`.
fn arg_is_truthy(value: &str) -> bool {
    matches!(value, "on" | "enabled" | "true" | "1")
}

/// `common_arg_utils::is_falsey`.
fn arg_is_falsey(value: &str) -> bool {
    matches!(value, "off" | "disabled" | "false" | "0")
}

const COMMON_PRESET_DEFAULT_NAME: &str = "default";

fn rm_leading_dashes(s: &str) -> &str {
    s.trim_start_matches('-')
}

fn canonical_tag(tag: &str) -> String {
    let bytes = tag.as_bytes();
    for i in 0..bytes.len() {
        if (bytes[i] == b'-' || bytes[i] == b'.')
            && i + 1 < bytes.len()
            && bytes[i + 1..]
                .iter()
                .all(|c| c.is_ascii_alphanumeric() || *c == b'_')
        {
            return format!("{}{}", &tag[..i + 1], tag[i + 1..].to_ascii_uppercase());
        }
    }
    tag.to_ascii_uppercase()
}

/// Handler kinds of `common_arg`, as used by `apply_to_params`.
#[derive(Clone, Copy, Debug, Default)]
pub enum PresetHandler {
    #[default]
    None,
    Str(fn(&mut Params, &str)),
    Int(fn(&mut Params, i32)),
    Bool(fn(&mut Params, bool)),
    Void(fn(&mut Params)),
}

/// The parts of `common_arg` a preset needs.
#[derive(Clone, Debug, Default)]
pub struct PresetArg {
    pub args: Vec<String>,
    pub args_neg: Vec<String>,
    pub env: Option<String>,
    pub value_hint: bool,
    pub value_hint_2: bool,
    pub is_preset_only: bool,
    pub handler: PresetHandler,
}

impl PartialEq for PresetArg {
    fn eq(&self, other: &Self) -> bool {
        self.args == other.args
            && self.args_neg == other.args_neg
            && self.env == other.env
            && self.value_hint == other.value_hint
            && self.value_hint_2 == other.value_hint_2
            && self.is_preset_only == other.is_preset_only
    }
}

impl Eq for PresetArg {}

impl Ord for PresetArg {
    fn cmp(&self, other: &Self) -> Ordering {
        self.args
            .cmp(&other.args)
            .then_with(|| self.args_neg.cmp(&other.args_neg))
            .then_with(|| self.env.cmp(&other.env))
            .then_with(|| self.value_hint.cmp(&other.value_hint))
            .then_with(|| self.value_hint_2.cmp(&other.value_hint_2))
            .then_with(|| self.is_preset_only.cmp(&other.is_preset_only))
    }
}

impl PartialOrd for PresetArg {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Preset {
    pub name: String,
    pub options: BTreeMap<PresetArg, String>,
}

impl Preset {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            options: BTreeMap::new(),
        }
    }

    /// `common_preset::to_args`.
    pub fn to_args(&self, bin_path: &str) -> Result<Vec<String>> {
        let mut args = Vec::new();
        if !bin_path.is_empty() {
            args.push(bin_path.to_owned());
        }
        for (opt, value) in &self.options {
            if opt.is_preset_only {
                continue;
            }
            let Some(main_arg) = opt.args.last() else {
                continue;
            };
            args.push(main_arg.clone());
            if !opt.value_hint && !opt.value_hint_2 {
                if arg_is_falsey(value) {
                    if let Some(neg) = opt.args_neg.last() {
                        *args.last_mut().expect("just pushed") = neg.clone();
                    } else {
                        args.pop();
                    }
                }
            }
            if opt.value_hint {
                args.push(value.clone());
            }
            if opt.value_hint && opt.value_hint_2 {
                return Err(Error::new(format!(
                    "common_preset::to_args(): option '{main_arg}' has two values, which is not supported yet"
                )));
            }
        }
        Ok(args)
    }

    /// `common_preset::to_ini`.
    pub fn to_ini(&self) -> String {
        let mut out = format!("[{}]\n", self.name);
        for (opt, value) in &self.options {
            let main_arg = opt.args.last().map(String::as_str).unwrap_or_default();
            let escaped = value.replace('\n', "\\\n");
            out.push_str(rm_leading_dashes(main_arg));
            out.push_str(" = ");
            out.push_str(&escaped);
            out.push('\n');
        }
        out.push('\n');
        out
    }

    pub fn set_option(&mut self, ctx: &PresetContext, env: &str, value: &str) -> Result<()> {
        for (opt, val) in &mut self.options {
            if opt.env.as_deref() == Some(env) {
                *val = value.to_owned();
                return Ok(());
            }
        }
        let Some(arg) = ctx.key_to_opt.get(env) else {
            return Err(Error::new(format!(
                "set_option: option with env '{env}' not found in ctx_params"
            )));
        };
        self.options.insert(arg.clone(), value.to_owned());
        Ok(())
    }

    pub fn unset_option(&mut self, env: &str) {
        let found = self
            .options
            .iter()
            .find(|(opt, _)| opt.env.as_deref() == Some(env))
            .map(|(opt, _)| opt.clone());
        if let Some(opt) = found {
            self.options.remove(&opt);
        }
    }

    pub fn get_option(&self, env: &str) -> Option<&str> {
        self.options
            .iter()
            .find(|(opt, _)| opt.env.as_deref() == Some(env))
            .map(|(_, val)| val.as_str())
    }

    /// Overwrite options present in `other`.
    pub fn merge(&mut self, other: &Self) {
        for (opt, val) in &other.options {
            self.options.insert(opt.clone(), val.clone());
        }
    }

    /// `common_preset::apply_to_params`.
    pub fn apply_to_params(
        &self,
        params: &mut Params,
        handled_keys: &BTreeSet<String>,
    ) -> Result<()> {
        for (opt, val) in &self.options {
            if !handled_keys.is_empty() {
                match opt.env.as_ref() {
                    Some(env) if handled_keys.contains(env) => {}
                    _ => continue,
                }
            }
            match opt.handler {
                PresetHandler::Str(f) => f(params, val),
                PresetHandler::Int(f) => f(params, parse_i32_prefix(val)?),
                PresetHandler::Bool(f) => f(params, arg_is_truthy(val)),
                PresetHandler::Void(f) => f(params),
                PresetHandler::None => {
                    return Err(Error::new(
                        "apply_to_params: option with two values is not supported yet",
                    ));
                }
            }
        }
        Ok(())
    }
}

fn parse_i32_prefix(s: &str) -> Result<i32> {
    let bytes = s.as_bytes();
    let mut end = 0usize;
    if bytes.first() == Some(&b'-') || bytes.first() == Some(&b'+') {
        end = 1;
    }
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == 0 || (end == 1 && !bytes[0].is_ascii_digit()) {
        return Err(Error::new(format!("invalid integer value: \"{s}\"")));
    }
    s[..end]
        .parse::<i32>()
        .map_err(|_| Error::new(format!("invalid integer value: \"{s}\"")))
}

#[derive(Clone, Debug, Default)]
pub struct PresetContext {
    pub key_to_opt: BTreeMap<String, PresetArg>,
    pub filter_allowed_keys: bool,
    pub allowed_keys: BTreeSet<String>,
    pub ignore_unknown_keys: bool,
}

fn is_bool_arg(arg: &PresetArg) -> bool {
    !arg.args_neg.is_empty()
}

fn parse_bool_arg(arg: &PresetArg, key: &str, value: &str) -> String {
    for neg in &arg.args_neg {
        if rm_leading_dashes(neg) == key {
            return if arg_is_truthy(value) {
                "false".to_owned()
            } else {
                "true".to_owned()
            };
        }
    }
    value.to_owned()
}

/// Parser for the INI grammar in `parse_ini_from_file`.
fn parse_ini(text: &str) -> Result<BTreeMap<String, BTreeMap<String, String>>> {
    let mut parsed: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    let mut section = COMMON_PRESET_DEFAULT_NAME.to_owned();
    parsed.entry(section.clone()).or_default();
    for raw_line in text.split(['\n', '\r']) {
        let line = raw_line.trim_start_matches([' ', '\t']);
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if let Some(header) = line.strip_prefix('[') {
            let end = header
                .find(']')
                .ok_or_else(|| Error::new("failed to parse server config file"))?;
            let rest = header[end + 1..].trim_start_matches([' ', '\t']);
            if !rest.is_empty() && !rest.starts_with(';') && !rest.starts_with('#') {
                return Err(Error::new("failed to parse server config file"));
            }
            section = header[..end].trim_matches([' ', '\t']).to_owned();
            parsed.entry(section.clone()).or_default();
            continue;
        }
        let eq = line
            .find('=')
            .ok_or_else(|| Error::new("failed to parse server config file"))?;
        let key = line[..eq].trim_end_matches([' ', '\t']);
        if !valid_ini_ident(key) {
            return Err(Error::new("failed to parse server config file"));
        }
        let raw_value = line[eq + 1..].trim_start_matches([' ', '\t']);
        let cut = ini_value_end(raw_value);
        parsed
            .entry(section.clone())
            .or_default()
            .insert(key.to_owned(), raw_value[..cut].to_owned());
    }
    Ok(parsed)
}

fn valid_ini_ident(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.is_empty() || !(bytes[0].is_ascii_alphabetic() || bytes[0] == b'_') {
        return false;
    }
    bytes[1..]
        .iter()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'-'))
}

/// Earliest position where eol-start (`ws` then comment or EOF) begins.
fn ini_value_end(value: &str) -> usize {
    let bytes = value.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b' ' || bytes[i] == b'\t' {
            let mut j = i;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            if j == bytes.len() || bytes[j] == b';' || bytes[j] == b'#' {
                return i;
            }
            i = j;
        } else if bytes[i] == b';' || bytes[i] == b'#' {
            return i;
        } else {
            i += 1;
        }
    }
    bytes.len()
}

/// `common_preset_context::load_from_ini`.
pub fn preset_load_from_ini(
    ctx: &PresetContext,
    path: &str,
    global: &mut Preset,
) -> Result<BTreeMap<String, Preset>> {
    if !std::path::Path::new(path).exists() {
        return Err(Error::new(format!("preset file does not exist: {path}")));
    }
    let contents = fs::read_to_string(path)
        .map_err(|e| Error::new(format!("failed to open server preset file: {path}: {e}")))?;
    let ini_data = parse_ini(&contents)?;
    let mut out = BTreeMap::new();
    for (section_name, entries) in ini_data {
        let mut name = if section_name.is_empty() {
            COMMON_PRESET_DEFAULT_NAME.to_owned()
        } else {
            section_name
        };
        if name != "*" && name != COMMON_PRESET_DEFAULT_NAME {
            if let Some(colon) = name.rfind(':') {
                let tag = &name[colon + 1..];
                let canon = canonical_tag(tag);
                if canon != tag {
                    name = format!("{}{}", &name[..colon + 1], canon);
                }
            }
        }
        let mut preset = Preset::new(name.clone());
        for (key, value) in entries {
            if key == "version" {
                continue;
            }
            if ctx.filter_allowed_keys && !ctx.allowed_keys.contains(&key) {
                return Err(Error::new(format!(
                    "option '{key}' is not allowed in remote presets"
                )));
            }
            if let Some(opt) = ctx.key_to_opt.get(&key) {
                let stored = if is_bool_arg(opt) {
                    parse_bool_arg(opt, &key, &value)
                } else {
                    value
                };
                preset.options.insert(opt.clone(), stored);
            } else if !ctx.ignore_unknown_keys {
                return Err(Error::new(format!(
                    "option '{key}' not recognized in preset '{name}'"
                )));
            }
        }
        if preset.name == COMMON_PRESET_DEFAULT_NAME && preset.options.is_empty() {
            continue;
        }
        if preset.name == "*" {
            *global = preset;
        } else {
            out.insert(preset.name.clone(), preset);
        }
    }
    Ok(out)
}

/// Cascade two preset maps: existing presets merge, new ones are added.
pub fn cascade_presets(
    base: &BTreeMap<String, Preset>,
    added: &BTreeMap<String, Preset>,
) -> BTreeMap<String, Preset> {
    let mut out = base.clone();
    for (name, preset) in added {
        out.entry(name.clone())
            .and_modify(|target| target.merge(preset))
            .or_insert_with(|| preset.clone());
    }
    out
}

/// Cascade one base preset over every preset in `presets`.
pub fn cascade_with_base(
    base: &Preset,
    presets: &BTreeMap<String, Preset>,
) -> BTreeMap<String, Preset> {
    let mut out = BTreeMap::new();
    for (name, preset) in presets {
        let mut tmp = base.clone();
        tmp.name = name.clone();
        tmp.merge(preset);
        out.insert(name.clone(), tmp);
    }
    out
}

// ---------------------------------------------------------------------------
// chat-tts-norm.{h,cpp}: pattern language, scan and sub
// ---------------------------------------------------------------------------

const TTS_INF: i32 = i32::MAX;
const TTS_MAX_REPEAT: i32 = 1000;
const TTS_MAX_PATTERN_LEN: usize = 4096;
pub const TTS_NPOS: usize = usize::MAX;

#[derive(Clone, Copy, Debug)]
struct TtsRange {
    lo: u32,
    hi: u32,
}

const TTS_SPACE: &[TtsRange] = &[
    TtsRange {
        lo: 0x0009,
        hi: 0x000D,
    },
    TtsRange {
        lo: 0x0020,
        hi: 0x0020,
    },
    TtsRange {
        lo: 0x0085,
        hi: 0x0085,
    },
    TtsRange {
        lo: 0x00A0,
        hi: 0x00A0,
    },
    TtsRange {
        lo: 0x1680,
        hi: 0x1680,
    },
    TtsRange {
        lo: 0x2000,
        hi: 0x200A,
    },
    TtsRange {
        lo: 0x2028,
        hi: 0x2029,
    },
    TtsRange {
        lo: 0x202F,
        hi: 0x202F,
    },
    TtsRange {
        lo: 0x205F,
        hi: 0x205F,
    },
    TtsRange {
        lo: 0x3000,
        hi: 0x3000,
    },
];

const TTS_PUNCT: &[TtsRange] = &[
    TtsRange {
        lo: 0x0021,
        hi: 0x002F,
    },
    TtsRange {
        lo: 0x003A,
        hi: 0x0040,
    },
    TtsRange {
        lo: 0x005B,
        hi: 0x0060,
    },
    TtsRange {
        lo: 0x007B,
        hi: 0x007E,
    },
    TtsRange {
        lo: 0x00A1,
        hi: 0x00A9,
    },
    TtsRange {
        lo: 0x00AB,
        hi: 0x00B4,
    },
    TtsRange {
        lo: 0x00B6,
        hi: 0x00B9,
    },
    TtsRange {
        lo: 0x00BB,
        hi: 0x00BF,
    },
    TtsRange {
        lo: 0x00D7,
        hi: 0x00D7,
    },
    TtsRange {
        lo: 0x00F7,
        hi: 0x00F7,
    },
    TtsRange {
        lo: 0x2010,
        hi: 0x2027,
    },
    TtsRange {
        lo: 0x2030,
        hi: 0x205E,
    },
    TtsRange {
        lo: 0x20A0,
        hi: 0x20CF,
    },
    TtsRange {
        lo: 0x2100,
        hi: 0x218F,
    },
    TtsRange {
        lo: 0x2190,
        hi: 0x2BFF,
    },
    TtsRange {
        lo: 0x3001,
        hi: 0x3003,
    },
    TtsRange {
        lo: 0x3008,
        hi: 0x3020,
    },
    TtsRange {
        lo: 0x3030,
        hi: 0x3030,
    },
    TtsRange {
        lo: 0x303D,
        hi: 0x303D,
    },
    TtsRange {
        lo: 0x30FB,
        hi: 0x30FB,
    },
    TtsRange {
        lo: 0xFE10,
        hi: 0xFE19,
    },
    TtsRange {
        lo: 0xFE30,
        hi: 0xFE4F,
    },
    TtsRange {
        lo: 0xFE50,
        hi: 0xFE6B,
    },
    TtsRange {
        lo: 0xFF01,
        hi: 0xFF0F,
    },
    TtsRange {
        lo: 0xFF1A,
        hi: 0xFF20,
    },
    TtsRange {
        lo: 0xFF3B,
        hi: 0xFF40,
    },
    TtsRange {
        lo: 0xFF5B,
        hi: 0xFF65,
    },
    TtsRange {
        lo: 0x1F000,
        hi: 0x1FAFF,
    },
];

const TTS_CONTROL: &[TtsRange] = &[
    TtsRange {
        lo: 0x0000,
        hi: 0x001F,
    },
    TtsRange {
        lo: 0x007F,
        hi: 0x009F,
    },
    TtsRange {
        lo: 0x200B,
        hi: 0x200F,
    },
    TtsRange {
        lo: 0x2060,
        hi: 0x206F,
    },
    TtsRange {
        lo: 0xFEFF,
        hi: 0xFEFF,
    },
];

const TTS_HAN: &[TtsRange] = &[
    TtsRange {
        lo: 0x2E80,
        hi: 0x2FDF,
    },
    TtsRange {
        lo: 0x3005,
        hi: 0x3005,
    },
    TtsRange {
        lo: 0x3007,
        hi: 0x3007,
    },
    TtsRange {
        lo: 0x3021,
        hi: 0x3029,
    },
    TtsRange {
        lo: 0x3038,
        hi: 0x303B,
    },
    TtsRange {
        lo: 0x3400,
        hi: 0x4DBF,
    },
    TtsRange {
        lo: 0x4E00,
        hi: 0x9FFF,
    },
    TtsRange {
        lo: 0xF900,
        hi: 0xFAFF,
    },
    TtsRange {
        lo: 0x20000,
        hi: 0x323AF,
    },
];

fn tts_in(ranges: &[TtsRange], c: u32) -> bool {
    ranges.iter().any(|r| c >= r.lo && c <= r.hi)
}

fn tts_is_alpha(c: u32) -> bool {
    if (b'a' as u32..=b'z' as u32).contains(&c) || (b'A' as u32..=b'Z' as u32).contains(&c) {
        return true;
    }
    if matches!(c, 0xAA | 0xB5 | 0xBA) {
        return true;
    }
    c >= 0xC0 && !tts_in(TTS_SPACE, c) && !tts_in(TTS_PUNCT, c) && !tts_in(TTS_CONTROL, c)
}

fn tts_class_match(kind: u8, c: u32) -> bool {
    match kind {
        b'd' => (b'0' as u32..=b'9' as u32).contains(&c),
        b's' => tts_in(TTS_SPACE, c),
        b'p' => tts_in(TTS_PUNCT, c),
        b'a' => tts_is_alpha(c),
        b'w' => tts_is_alpha(c) || (b'0' as u32..=b'9' as u32).contains(&c),
        b'u' => (b'A' as u32..=b'Z' as u32).contains(&c),
        b'l' => (b'a' as u32..=b'z' as u32).contains(&c),
        b'h' => tts_in(TTS_HAN, c),
        b'c' => tts_in(TTS_CONTROL, c),
        _ => false,
    }
}

fn tts_is_class_letter(c: u32) -> bool {
    let lower = if (b'A' as u32..=b'Z' as u32).contains(&c) {
        c + (b'a' - b'A') as u32
    } else {
        c
    };
    lower < 0x80 && b"dspawulhc".contains(&(lower as u8))
}

fn tts_ascii_lower(c: u32) -> u32 {
    if (b'A' as u32..=b'Z' as u32).contains(&c) {
        c + (b'a' - b'A') as u32
    } else {
        c
    }
}

fn tts_ascii_swapcase(c: u32) -> u32 {
    if (b'A' as u32..=b'Z' as u32).contains(&c) {
        c + (b'a' - b'A') as u32
    } else if (b'a' as u32..=b'z' as u32).contains(&c) {
        c - (b'a' - b'A') as u32
    } else {
        c
    }
}

/// Invalid bytes decode to U+FFFD, one per byte. `offs[n]` is the byte length.
fn tts_decode_utf8(s: &[u8]) -> (Vec<u32>, Vec<usize>) {
    let mut cpts = Vec::with_capacity(s.len());
    let mut offs = Vec::with_capacity(s.len() + 1);
    let mut i = 0usize;
    while i < s.len() {
        offs.push(i);
        let c = s[i];
        let mut cp: u32;
        let len = tts_sequence_len(c);
        if len == 0 {
            cpts.push(0xFFFD);
            i += 1;
            continue;
        }
        cp = match len {
            1 => c as u32,
            2 => (c & 0x1f) as u32,
            3 => (c & 0x0f) as u32,
            _ => (c & 0x07) as u32,
        };
        let mut ok = i + len <= s.len();
        for k in 1..len {
            let cc = s[i + k];
            if cc & 0xc0 != 0x80 {
                ok = false;
            } else {
                cp = (cp << 6) | (cc & 0x3f) as u32;
            }
        }
        if ok {
            cpts.push(cp);
            i += len;
        } else {
            cpts.push(0xFFFD);
            i += 1;
        }
    }
    offs.push(s.len());
    (cpts, offs)
}

fn tts_sequence_len(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if c & 0xe0 == 0xc0 {
        2
    } else if c & 0xf0 == 0xe0 {
        3
    } else if c & 0xf8 == 0xf0 {
        4
    } else {
        0
    }
}

#[derive(Clone, Debug)]
struct TtsClass {
    kind: u8,
    neg: bool,
}

#[derive(Clone, Debug, Default)]
struct TtsSet {
    neg: bool,
    ranges: Vec<(u32, u32)>,
    classes: Vec<TtsClass>,
}

#[derive(Clone, Debug)]
enum TtsNode {
    Literal(u32),
    Set(TtsSet),
    Any,
    Start,
    End,
    Boundary,
    Group {
        alts: Vec<Vec<TtsItem>>,
        capture: i32,
    },
    Lookahead {
        alts: Vec<Vec<TtsItem>>,
        neg: bool,
    },
    Lookbehind {
        atom: Box<TtsItem>,
        neg: bool,
    },
}

impl TtsNode {
    fn is_zero_width(&self) -> bool {
        matches!(
            self,
            Self::Start
                | Self::End
                | Self::Boundary
                | Self::Lookahead { .. }
                | Self::Lookbehind { .. }
        )
    }

    fn is_group(&self) -> bool {
        matches!(self, Self::Group { .. })
    }
}

#[derive(Clone, Debug)]
struct TtsItem {
    atom: Box<TtsNode>,
    min: i32,
    max: i32,
}

enum TtsEscape {
    Class(TtsClass),
    Lit(u32),
    Boundary,
}

struct TtsParser {
    s: Vec<u32>,
    p: usize,
    n_caps: usize,
}

impl TtsParser {
    fn fail<T>(&self, msg: impl Into<String>) -> Result<T> {
        Err(Error::new(format!(
            "invalid TTS norm pattern: {} at position {}",
            msg.into(),
            self.p
        )))
    }

    fn eof(&self) -> bool {
        self.p >= self.s.len()
    }

    fn peek(&self) -> u32 {
        self.s[self.p]
    }

    fn peek_is(&self, c: u8) -> bool {
        !self.eof() && self.s[self.p] == c as u32
    }

    fn expect(&mut self, c: u8, what: &str) -> Result<()> {
        if !self.peek_is(c) {
            return self.fail(format!("expected {what}"));
        }
        self.p += 1;
        Ok(())
    }

    fn parse_alts(&mut self) -> Result<Vec<Vec<TtsItem>>> {
        let mut alts = vec![self.parse_seq()?];
        while self.peek_is(b'|') {
            self.p += 1;
            alts.push(self.parse_seq()?);
        }
        Ok(alts)
    }

    fn parse_seq(&mut self) -> Result<Vec<TtsItem>> {
        let mut seq = Vec::new();
        while !self.eof() && self.peek() != b'|' as u32 && self.peek() != b')' as u32 {
            let mut item = TtsItem {
                atom: Box::new(self.parse_atom()?),
                min: 1,
                max: 1,
            };
            self.parse_quantifier(&mut item)?;
            seq.push(item);
        }
        Ok(seq)
    }

    fn parse_int(&mut self) -> Result<i32> {
        if self.eof() || self.peek() < b'0' as u32 || self.peek() > b'9' as u32 {
            return self.fail("expected number in quantifier");
        }
        let mut v = 0i32;
        while !self.eof() && (b'0' as u32..=b'9' as u32).contains(&self.peek()) {
            v = v
                .saturating_mul(10)
                .saturating_add((self.peek() - b'0' as u32) as i32);
            if v > TTS_MAX_REPEAT {
                return self.fail(format!("repetition count exceeds {TTS_MAX_REPEAT}"));
            }
            self.p += 1;
        }
        Ok(v)
    }

    fn parse_quantifier(&mut self, item: &mut TtsItem) -> Result<()> {
        if self.eof() {
            return Ok(());
        }
        let c = self.peek();
        if c != b'?' as u32 && c != b'*' as u32 && c != b'+' as u32 && c != b'{' as u32 {
            return Ok(());
        }
        if item.atom.is_zero_width() {
            return self.fail("zero-width assertion cannot be repeated");
        }
        self.p += 1;
        if c == b'?' as u32 {
            item.min = 0;
            item.max = 1;
        } else if c == b'*' as u32 {
            item.min = 0;
            item.max = TTS_INF;
        } else if c == b'+' as u32 {
            item.min = 1;
            item.max = TTS_INF;
        } else {
            item.min = self.parse_int()?;
            item.max = item.min;
            if self.peek_is(b',') {
                self.p += 1;
                item.max = if self.peek_is(b'}') {
                    TTS_INF
                } else {
                    self.parse_int()?
                };
            }
            self.expect(b'}', "'}' to close quantifier")?;
            if item.max < item.min {
                return self.fail("quantifier max is smaller than min");
            }
        }
        if !self.eof() {
            let next = self.peek();
            if next == b'?' as u32
                || next == b'*' as u32
                || next == b'+' as u32
                || next == b'{' as u32
            {
                return self.fail("multiple quantifiers");
            }
        }
        Ok(())
    }

    fn parse_group(&mut self) -> Result<TtsNode> {
        if self.peek_is(b'?') {
            self.p += 1;
            if self.eof() {
                return self.fail("unterminated group");
            }
            let c = self.peek();
            if c == b':' as u32 {
                self.p += 1;
            } else if c == b'=' as u32 || c == b'!' as u32 {
                self.p += 1;
                return Ok(TtsNode::Lookahead {
                    alts: self.parse_group_body()?,
                    neg: c == b'!' as u32,
                });
            } else if c == b'<' as u32
                && self.p + 1 < self.s.len()
                && (self.s[self.p + 1] == b'=' as u32 || self.s[self.p + 1] == b'!' as u32)
            {
                let neg = self.s[self.p + 1] == b'!' as u32;
                self.p += 2;
                if self.eof() || self.peek_is(b')') {
                    return self.fail("lookbehind must contain exactly one codepoint matcher");
                }
                let atom = Box::new(TtsItem {
                    atom: Box::new(self.parse_atom()?),
                    min: 1,
                    max: 1,
                });
                if !matches!(
                    *atom.atom,
                    TtsNode::Literal(_) | TtsNode::Set(_) | TtsNode::Any
                ) || !self.peek_is(b')')
                {
                    return self.fail("lookbehind must contain exactly one codepoint matcher");
                }
                self.p += 1;
                return Ok(TtsNode::Lookbehind { atom, neg });
            } else if c == b'i' as u32 {
                return self.fail("(?i) is only allowed at the start of the pattern");
            } else {
                return self.fail("unknown group type");
            }
            let alts = self.parse_group_body()?;
            Ok(TtsNode::Group { alts, capture: -1 })
        } else {
            let capture = self.n_caps as i32;
            self.n_caps += 1;
            let alts = self.parse_group_body()?;
            Ok(TtsNode::Group { alts, capture })
        }
    }

    fn parse_group_body(&mut self) -> Result<Vec<Vec<TtsItem>>> {
        let alts = self.parse_alts()?;
        self.expect(b')', "')' to close group")?;
        Ok(alts)
    }

    fn parse_escape(&mut self, in_set: bool) -> Result<TtsEscape> {
        if self.eof() {
            return self.fail("dangling '%'");
        }
        let e = self.s[self.p];
        self.p += 1;
        if tts_is_class_letter(e) {
            return Ok(TtsEscape::Class(TtsClass {
                kind: tts_ascii_lower(e) as u8,
                neg: (b'A' as u32..=b'Z' as u32).contains(&e),
            }));
        }
        if e == b'b' as u32 && !in_set {
            return Ok(TtsEscape::Boundary);
        }
        if (b'0' as u32..=b'9' as u32).contains(&e)
            || (b'A' as u32..=b'Z' as u32).contains(&e)
            || (b'a' as u32..=b'z' as u32).contains(&e)
        {
            self.p -= 1;
            return self.fail(format!(
                "unknown escape '%{}'",
                char::from_u32(e).unwrap_or('\u{fffd}')
            ));
        }
        Ok(TtsEscape::Lit(e))
    }

    fn parse_set(&mut self) -> Result<TtsNode> {
        let mut set = TtsSet::default();
        if self.peek_is(b'^') {
            set.neg = true;
            self.p += 1;
        }
        let mut empty = true;
        loop {
            if self.eof() {
                return self.fail("unterminated set");
            }
            if self.peek_is(b']') {
                self.p += 1;
                break;
            }
            empty = false;
            let lo = if self.peek_is(b'%') {
                self.p += 1;
                match self.parse_escape(true)? {
                    TtsEscape::Class(c) => {
                        set.classes.push(c);
                        continue;
                    }
                    TtsEscape::Lit(l) => l,
                    TtsEscape::Boundary => return self.fail("invalid escape in set"),
                }
            } else {
                let l = self.peek();
                self.p += 1;
                l
            };
            let mut hi = lo;
            if self.peek_is(b'-') && self.p + 1 < self.s.len() && self.s[self.p + 1] != b']' as u32
            {
                self.p += 1;
                hi = if self.peek_is(b'%') {
                    self.p += 1;
                    match self.parse_escape(true)? {
                        TtsEscape::Lit(h) => h,
                        _ => return self.fail("character class cannot be used in a range"),
                    }
                } else {
                    let h = self.peek();
                    self.p += 1;
                    h
                };
                if hi < lo {
                    return self.fail("invalid range in set");
                }
            }
            set.ranges.push((lo, hi));
        }
        if empty {
            return self.fail("empty set");
        }
        Ok(TtsNode::Set(set))
    }

    fn parse_atom(&mut self) -> Result<TtsNode> {
        let c = self.s[self.p];
        self.p += 1;
        match c {
            x if x == b'(' as u32 => self.parse_group(),
            x if x == b'[' as u32 => self.parse_set(),
            x if x == b'.' as u32 => Ok(TtsNode::Any),
            x if x == b'^' as u32 => Ok(TtsNode::Start),
            x if x == b'$' as u32 => Ok(TtsNode::End),
            x if x == b'%' as u32 => match self.parse_escape(false)? {
                TtsEscape::Class(c) => Ok(TtsNode::Set(TtsSet {
                    classes: vec![c],
                    ..TtsSet::default()
                })),
                TtsEscape::Boundary => Ok(TtsNode::Boundary),
                TtsEscape::Lit(l) => Ok(TtsNode::Literal(l)),
            },
            x if x == b'?' as u32 || x == b'*' as u32 || x == b'+' as u32 || x == b'{' as u32 => {
                self.p -= 1;
                self.fail("nothing to repeat")
            }
            x if x == b']' as u32 || x == b'}' as u32 => {
                self.p -= 1;
                self.fail(format!(
                    "unescaped '{}', use '%{}'",
                    char::from_u32(x).unwrap_or('\u{fffd}'),
                    char::from_u32(x).unwrap_or('\u{fffd}')
                ))
            }
            _ => Ok(TtsNode::Literal(c)),
        }
    }
}

struct TtsMatcher<'a> {
    t: &'a [u32],
    icase: bool,
    caps: Vec<(i64, i64)>,
}

impl<'a> TtsMatcher<'a> {
    fn is_w(&self, i: usize) -> bool {
        i < self.t.len() && tts_class_match(b'w', self.t[i])
    }

    fn set_match(&self, set: &TtsSet, c: u32) -> bool {
        let test = |x: u32| {
            set.ranges.iter().any(|&(lo, hi)| x >= lo && x <= hi)
                || set
                    .classes
                    .iter()
                    .any(|k| tts_class_match(k.kind, x) != k.neg)
        };
        let mut m = test(c);
        if !m && self.icase {
            let other = tts_ascii_swapcase(c);
            if other != c {
                m = test(other);
            }
        }
        m != set.neg
    }

    fn match_alts(&mut self, alts: &[Vec<TtsItem>], i: usize, end: &mut usize) -> bool {
        for seq in alts {
            let saved = self.caps.clone();
            if self.match_seq(seq, i, end) {
                return true;
            }
            self.caps = saved;
        }
        false
    }

    fn match_seq(&mut self, seq: &[TtsItem], i: usize, end: &mut usize) -> bool {
        let mut pos = i;
        for item in seq {
            let mut count = 0i32;
            while count < item.max {
                let is_group = item.atom.is_group();
                let saved = if is_group {
                    Some(self.caps.clone())
                } else {
                    None
                };
                let mut e = pos;
                if !self.match_once(&item.atom, pos, &mut e) {
                    if let Some(saved) = saved {
                        self.caps = saved;
                    }
                    break;
                }
                if e == pos {
                    count = (count + 1).max(item.min);
                    break;
                }
                pos = e;
                count += 1;
            }
            if count < item.min {
                return false;
            }
        }
        *end = pos;
        true
    }

    fn match_once(&mut self, node: &TtsNode, i: usize, end: &mut usize) -> bool {
        let n = self.t.len();
        match node {
            TtsNode::Literal(cp) => {
                if i < n
                    && (self.t[i] == *cp
                        || (self.icase && tts_ascii_lower(self.t[i]) == tts_ascii_lower(*cp)))
                {
                    *end = i + 1;
                    return true;
                }
                false
            }
            TtsNode::Set(set) => {
                if i < n && self.set_match(set, self.t[i]) {
                    *end = i + 1;
                    return true;
                }
                false
            }
            TtsNode::Any => {
                if i < n {
                    *end = i + 1;
                    return true;
                }
                false
            }
            TtsNode::Start => {
                *end = i;
                i == 0
            }
            TtsNode::End => {
                *end = i;
                i == n
            }
            TtsNode::Boundary => {
                *end = i;
                (i > 0 && self.is_w(i - 1)) != self.is_w(i)
            }
            TtsNode::Group { alts, capture } => {
                let mut e = i;
                if !self.match_alts(alts, i, &mut e) {
                    return false;
                }
                if *capture >= 0 && (*capture as usize) < self.caps.len() {
                    self.caps[*capture as usize] = (i as i64, e as i64);
                }
                *end = e;
                true
            }
            TtsNode::Lookahead { alts, neg } => {
                let saved = self.caps.clone();
                let mut e = i;
                let ok = self.match_alts(alts, i, &mut e);
                self.caps = saved;
                *end = i;
                ok != *neg
            }
            TtsNode::Lookbehind { atom, neg } => {
                let mut e = 0usize;
                let ok = i > 0 && self.match_once(&atom.atom, i - 1, &mut e) && e == i;
                *end = i;
                ok != *neg
            }
        }
    }
}

/// Compiled TTS normalization pattern.
#[derive(Clone, Debug)]
pub struct TtsNormPattern {
    alts: Vec<Vec<TtsItem>>,
    n_caps: usize,
    icase: bool,
}

impl TtsNormPattern {
    pub fn new(src: &str) -> Result<Self> {
        if src.len() > TTS_MAX_PATTERN_LEN {
            return Err(Error::new(format!(
                "invalid TTS norm pattern: pattern is longer than {TTS_MAX_PATTERN_LEN} bytes"
            )));
        }
        let (cpts, _) = tts_decode_utf8(src.as_bytes());
        let mut parser = TtsParser {
            s: cpts,
            p: 0,
            n_caps: 0,
        };
        let prefix = [b'(' as u32, b'?' as u32, b'i' as u32, b')' as u32];
        let mut icase = false;
        if parser.s.len() >= 4 && parser.s[..4] == prefix[..] {
            icase = true;
            parser.p = 4;
        }
        let alts = parser.parse_alts()?;
        if !parser.eof() {
            return parser.fail("unbalanced ')'");
        }
        Ok(Self {
            alts,
            n_caps: parser.n_caps,
            icase,
        })
    }

    pub fn n_groups(&self) -> usize {
        self.n_caps
    }

    /// `match_at`: anchored at codepoint `pos`; `end` is the first codepoint
    /// after a successful match; `groups[i]` holds `[start, end)` or `{-1,-1}`.
    pub fn match_at(
        &self,
        cpts: &[u32],
        pos: usize,
        end: &mut usize,
        groups: &mut Vec<(i64, i64)>,
    ) -> bool {
        let mut matcher = TtsMatcher {
            t: cpts,
            icase: self.icase,
            caps: vec![(-1, -1); self.n_caps],
        };
        let mut e = pos;
        if !matcher.match_alts(&self.alts, pos, &mut e) {
            return false;
        }
        *end = e;
        *groups = matcher.caps;
        true
    }
}

/// Segment produced by `tts_norm_scan`; byte offsets into the input text.
#[derive(Clone, Debug, Default)]
pub struct TtsNormSegment {
    pub rule: i32,
    pub start: usize,
    pub end: usize,
    pub groups: Vec<(usize, usize)>,
}

/// Split text into matched and unmatched segments; first non-empty rule wins.
pub fn tts_norm_scan(text: &str, rules: &[&TtsNormPattern]) -> Vec<TtsNormSegment> {
    let (cpts, offs) = tts_decode_utf8(text.as_bytes());
    let mut out = Vec::new();
    let n = cpts.len();
    let mut i = 0usize;
    let mut pending = 0usize;
    let mut groups = Vec::new();
    while i < n {
        let mut matched = false;
        for (r, pattern) in rules.iter().enumerate() {
            let mut end = i;
            if pattern.match_at(&cpts, i, &mut end, &mut groups) && end > i {
                if pending < i {
                    out.push(TtsNormSegment {
                        rule: -1,
                        start: offs[pending],
                        end: offs[i],
                        groups: Vec::new(),
                    });
                }
                let seg_groups = groups
                    .iter()
                    .map(|&(start, end)| {
                        if start < 0 {
                            (TTS_NPOS, TTS_NPOS)
                        } else {
                            (offs[start as usize], offs[end as usize])
                        }
                    })
                    .collect();
                out.push(TtsNormSegment {
                    rule: r as i32,
                    start: offs[i],
                    end: offs[end],
                    groups: seg_groups,
                });
                i = end;
                pending = i;
                matched = true;
                break;
            }
        }
        if !matched {
            i += 1;
        }
    }
    if pending < n {
        out.push(TtsNormSegment {
            rule: -1,
            start: offs[pending],
            end: offs[n],
            groups: Vec::new(),
        });
    }
    out
}

struct TtsReplPiece {
    literal: String,
    group: i32,
}

fn tts_parse_repl(repl: &str, pattern: &TtsNormPattern) -> Result<Vec<TtsReplPiece>> {
    let mut pieces = Vec::new();
    let mut lit = String::new();
    let mut iter = repl.chars();
    while let Some(c) = iter.next() {
        if c != '%' {
            lit.push(c);
            continue;
        }
        let Some(next) = iter.next() else {
            return Err(Error::new("invalid TTS norm replacement: dangling '%'"));
        };
        if next == '%' {
            lit.push('%');
        } else if next.is_ascii_digit() {
            let g = next as i32 - '0' as i32;
            if g as usize > pattern.n_groups() {
                return Err(Error::new(format!(
                    "invalid TTS norm replacement: group {next} does not exist"
                )));
            }
            if !lit.is_empty() {
                pieces.push(TtsReplPiece {
                    literal: std::mem::take(&mut lit),
                    group: -1,
                });
            }
            pieces.push(TtsReplPiece {
                literal: String::new(),
                group: g,
            });
        } else {
            return Err(Error::new(format!(
                "invalid TTS norm replacement: unknown escape '%{next}'"
            )));
        }
    }
    if !lit.is_empty() {
        pieces.push(TtsReplPiece {
            literal: lit,
            group: -1,
        });
    }
    Ok(pieces)
}

/// Replace every non-overlapping match of `pattern` with `repl`.
pub fn tts_norm_sub(text: &str, pattern: &TtsNormPattern, repl: &str) -> Result<String> {
    let pieces = tts_parse_repl(repl, pattern)?;
    let mut out = String::new();
    for seg in tts_norm_scan(text, &[pattern]) {
        if seg.rule < 0 {
            out.push_str(&text[seg.start..seg.end]);
            continue;
        }
        for piece in &pieces {
            if piece.group < 0 {
                out.push_str(&piece.literal);
            } else if piece.group == 0 {
                out.push_str(&text[seg.start..seg.end]);
            } else {
                let g = seg.groups[piece.group as usize - 1];
                if g.0 != TTS_NPOS {
                    out.push_str(&text[g.0..g.1]);
                }
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, Default)]
pub struct TtsDict {
    /// Sorted by raw bytes.
    pub keys: Vec<String>,
    /// JSON document of each key.
    pub values: Vec<String>,
}

/// Binary search of `key`; on success `value` is the JSON document.
pub fn tts_dict_find(dict: &TtsDict, key: &str, value: &mut String) -> Result<bool> {
    let Ok(idx) = dict.keys.binary_search_by(|k| k.as_str().cmp(key)) else {
        return Ok(false);
    };
    if idx >= dict.values.len() {
        return Err(Error::new(
            "tts_dict: malformed dictionary, a key has no value",
        ));
    }
    *value = dict.values[idx].clone();
    Ok(true)
}

// ---------------------------------------------------------------------------
// imatrix-loader.{h,cpp}
// ---------------------------------------------------------------------------

pub const LLM_KV_IMATRIX_DATASETS: &str = "imatrix.datasets";
pub const LLM_KV_IMATRIX_CHUNK_COUNT: &str = "imatrix.chunk_count";
pub const LLM_KV_IMATRIX_CHUNK_SIZE: &str = "imatrix.chunk_size";
pub const LLM_KV_IMATRIX_STATS_SCHEMA: &str = "imatrix.stats_schema";
pub const LLM_KV_IMATRIX_N_LAYER_NEXTN: &str = "imatrix.n_layer_nextn";

#[derive(Clone, Debug, Default)]
pub struct ImatrixEntry {
    pub sums: Vec<f32>,
    pub activations: Vec<f32>,
    pub counts: Vec<i64>,
    pub stats: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct Imatrix {
    pub entries: BTreeMap<String, ImatrixEntry>,
    pub datasets: Vec<String>,
    pub stats_schema: Vec<String>,
    pub chunk_count: i32,
    pub chunk_size: i32,
    pub n_layer_nextn: i32,
    pub is_legacy: bool,
    pub has_metadata: bool,
}

fn read_i32(data: &[u8], pos: &mut usize) -> Option<i32> {
    if *pos + 4 > data.len() {
        return None;
    }
    let bytes: [u8; 4] = data[*pos..*pos + 4].try_into().ok()?;
    *pos += 4;
    Some(i32::from_ne_bytes(bytes))
}

fn read_f32(data: &[u8], pos: &mut usize) -> Option<f32> {
    if *pos + 4 > data.len() {
        return None;
    }
    let bytes: [u8; 4] = data[*pos..*pos + 4].try_into().ok()?;
    *pos += 4;
    Some(f32::from_ne_bytes(bytes))
}

/// `common_imatrix_load_legacy`.
fn imatrix_load_legacy(fname: &str, imatrix: &mut Imatrix) -> Result<bool> {
    let Ok(data) = fs::read(fname) else {
        eprintln!("imatrix_load_legacy: failed to open {fname}");
        return Ok(false);
    };
    let mut pos = 0usize;
    let Some(n_entries) = read_i32(&data, &mut pos) else {
        eprintln!("imatrix_load_legacy: no data in file {fname}");
        return Ok(false);
    };
    if n_entries < 1 {
        eprintln!("imatrix_load_legacy: no data in file {fname}");
        return Ok(false);
    }
    for i in 0..n_entries {
        let Some(len) = read_i32(&data, &mut pos) else {
            return Ok(false);
        };
        if len < 0 || pos + len as usize > data.len() {
            eprintln!(
                "imatrix_load_legacy: failed reading name for entry {}",
                i + 1
            );
            return Ok(false);
        }
        let name = String::from_utf8_lossy(&data[pos..pos + len as usize]).into_owned();
        pos += len as usize;
        let Some(ncall) = read_i32(&data, &mut pos) else {
            return Ok(false);
        };
        let Some(nval) = read_i32(&data, &mut pos) else {
            return Ok(false);
        };
        if nval < 1 {
            eprintln!("imatrix_load_legacy: failed reading number of values for entry {i}");
            return Ok(false);
        }
        let mut sums = Vec::with_capacity(nval as usize);
        for _ in 0..nval {
            let Some(v) = read_f32(&data, &mut pos) else {
                eprintln!("imatrix_load_legacy: failed reading data for entry {i}");
                return Ok(false);
            };
            sums.push(v);
        }
        let entry = imatrix.entries.entry(name).or_default();
        entry.sums = sums;
        entry.counts = vec![ncall as i64];
    }

    // trailing chunk count and dataset name are optional
    if pos < data.len() {
        if let Some(n_calls) = read_i32(&data, &mut pos) {
            imatrix.chunk_count = n_calls;
            if let Some(len) = read_i32(&data, &mut pos) {
                if len > 0 && pos + len as usize <= data.len() {
                    let bytes = &data[pos..pos + len as usize];
                    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                    imatrix
                        .datasets
                        .push(String::from_utf8_lossy(&bytes[..end]).into_owned());
                }
            }
        }
    }
    imatrix.chunk_size = 0;
    imatrix.is_legacy = true;
    Ok(true)
}

#[derive(Clone, Copy, Debug, Default)]
struct SumTensors {
    in_sum: Option<usize>,
    in_sum2: Option<usize>,
    counts: Option<usize>,
    stats: Option<usize>,
}

fn tensor_nelements(t: &crate::gguf::GgmlTensor) -> usize {
    t.ne.iter()
        .fold(1usize, |acc, &n| acc.saturating_mul(n.max(1) as usize))
}

fn tensor_f32(t: &crate::gguf::GgmlTensor, n: usize) -> Option<Vec<f32>> {
    if t.data.len() < n * 4 {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for chunk in t.data[..n * 4].chunks_exact(4) {
        out.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(out)
}

fn imatrix_load_gguf(ctx: &crate::gguf::GgufContext, imatrix: &mut Imatrix) -> Result<bool> {
    let n_entries = gguf_get_n_tensors(ctx);
    if n_entries < 1 {
        eprintln!("imatrix_load: no data in file");
        return Ok(false);
    }
    let datasets_key = gguf_find_key(ctx, LLM_KV_IMATRIX_DATASETS);
    let chunk_count_key = gguf_find_key(ctx, LLM_KV_IMATRIX_CHUNK_COUNT);
    let chunk_size_key = gguf_find_key(ctx, LLM_KV_IMATRIX_CHUNK_SIZE);
    let nextn_key = gguf_find_key(ctx, LLM_KV_IMATRIX_N_LAYER_NEXTN);

    if datasets_key >= 0
        && gguf_get_kv_type(ctx, datasets_key) == GgufType::Array
        && gguf_get_arr_type(ctx, datasets_key) == GgufType::String
    {
        for i in 0..gguf_get_arr_n(ctx, datasets_key) {
            imatrix
                .datasets
                .push(gguf_get_arr_str(ctx, datasets_key, i).to_owned());
        }
    }

    imatrix.has_metadata = datasets_key >= 0 && chunk_count_key >= 0 && chunk_size_key >= 0;
    imatrix.chunk_count = if chunk_count_key >= 0 {
        gguf_get_val_u32(ctx, chunk_count_key) as i32
    } else {
        0
    };
    imatrix.chunk_size = if chunk_size_key >= 0 {
        gguf_get_val_u32(ctx, chunk_size_key) as i32
    } else {
        0
    };
    imatrix.n_layer_nextn = if nextn_key >= 0 {
        gguf_get_val_u32(ctx, nextn_key) as i32
    } else {
        0
    };

    let schema_key = gguf_find_key(ctx, LLM_KV_IMATRIX_STATS_SCHEMA);
    if schema_key >= 0
        && gguf_get_kv_type(ctx, schema_key) == GgufType::Array
        && gguf_get_arr_type(ctx, schema_key) == GgufType::String
    {
        let n_schema = gguf_get_arr_n(ctx, schema_key);
        imatrix.stats_schema = (0..n_schema)
            .map(|i| gguf_get_arr_str(ctx, schema_key, i).to_owned())
            .collect();
    } else {
        imatrix.stats_schema = [
            "sum_sq",
            "mean",
            "elements",
            "std_deviation",
            "skewness",
            "kurtosis",
            "gain",
            "h_norm",
            "l2_dist",
            "cossim",
            "pearson",
            "covariance",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    }

    let mut sums_counts: BTreeMap<String, SumTensors> = BTreeMap::new();
    for (i, info) in ctx.info.iter().enumerate() {
        let mut name = info.t.name.clone();
        if name.is_empty() {
            continue;
        }
        if string_remove_suffix(&mut name, ".in_sum") {
            sums_counts.entry(name).or_default().in_sum = Some(i);
        } else if string_remove_suffix(&mut name, ".in_sum2") {
            sums_counts.entry(name).or_default().in_sum2 = Some(i);
        } else if string_remove_suffix(&mut name, ".counts") {
            sums_counts.entry(name).or_default().counts = Some(i);
        } else if string_remove_suffix(&mut name, ".stats") {
            sums_counts.entry(name).or_default().stats = Some(i);
        }
    }

    for (name, sc) in sums_counts {
        let Some(in_sum2) = sc.in_sum2 else {
            eprintln!("imatrix_load: mismatched sums and counts for {name}");
            return Ok(false);
        };
        let Some(counts) = sc.counts else {
            eprintln!("imatrix_load: mismatched sums and counts for {name}");
            return Ok(false);
        };
        if let Some(in_sum) = sc.in_sum {
            if tensor_nelements(&ctx.info[in_sum].t) != tensor_nelements(&ctx.info[in_sum2].t) {
                eprintln!("imatrix_load: mismatched sums and counts for {name}");
                return Ok(false);
            }
        }
        if ctx.info[in_sum2].t.type_ != GgmlType::F32 || ctx.info[counts].t.type_ != GgmlType::F32 {
            eprintln!("imatrix_load: sums and counts for {name} must be F32");
            return Ok(false);
        }

        let nval = tensor_nelements(&ctx.info[in_sum2].t);
        let ncounts = tensor_nelements(&ctx.info[counts].t);
        let Some(sums) = tensor_f32(&ctx.info[in_sum2].t, nval) else {
            return Ok(false);
        };
        let Some(count_values) = tensor_f32(&ctx.info[counts].t, ncounts) else {
            return Ok(false);
        };
        let entry = imatrix.entries.entry(name).or_default();
        entry.sums = sums;
        entry.counts = count_values.iter().map(|&v| v.round() as i64).collect();
        if let Some(in_sum) = sc.in_sum {
            if tensor_nelements(&ctx.info[in_sum].t) == nval {
                if let Some(activations) = tensor_f32(&ctx.info[in_sum].t, nval) {
                    entry.activations = activations;
                }
            }
        }
        if let Some(stats) = sc.stats {
            if ctx.info[stats].t.type_ == GgmlType::F32 {
                let n_stats = tensor_nelements(&ctx.info[stats].t);
                if let Some(values) = tensor_f32(&ctx.info[stats].t, n_stats) {
                    entry.stats = values;
                }
            }
        }
    }
    Ok(true)
}

/// `common_imatrix_load`: GGUF first, legacy binary format as fallback.
pub fn imatrix_load(fname: &str, imatrix: &mut Imatrix) -> Result<bool> {
    let mut init = GgufInitParams {
        no_alloc: false,
        ctx: Some(GgmlContext::default()),
    };
    match gguf_init_from_file(fname, &mut init) {
        Ok(ctx) => imatrix_load_gguf(&ctx, imatrix),
        Err(_) => imatrix_load_legacy(fname, imatrix),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trip_and_size_helpers() {
        for input in [
            b"".as_slice(),
            b"f".as_slice(),
            b"fo".as_slice(),
            b"foo".as_slice(),
        ] {
            let encoded = base64_encode(input, Base64Alphabet::Standard);
            assert_eq!(encoded.len(), base64_required_encode_size(input.len()));
            assert_eq!(
                base64_decode(
                    encoded.as_bytes(),
                    Base64Alphabet::Auto,
                    Base64Decoding::Moderate
                )
                .unwrap(),
                input
            );
        }
        assert_eq!(
            base64_encode(b"\xfb\xff", Base64Alphabet::UrlFilenameSafe),
            "-_8="
        );
        assert!(base64_decode(b"!!!!", Base64Alphabet::Auto, Base64Decoding::Moderate).is_err());
    }

    #[test]
    fn trie_matches_and_aho_reports_longest() {
        let words = vec!["ab".to_owned(), "abc".to_owned()];
        let trie = Trie::from_words(&words);
        assert_eq!(trie.check_at(b"abc", 0), MatchResult::CompleteMatch);
        assert_eq!(trie.check_at(b"ax", 0), MatchResult::NoMatch);
        assert_eq!(trie.check_at(b"a", 0), MatchResult::PartialMatch);
        let ac = AhoCorasick::from_words(&words);
        let mut state = 0usize;
        let mut best = -1i32;
        for &ch in b"abc" {
            state = ac.next(state, TrieSymbol::codepoint(ch as u32));
            best = ac.match_pattern(state).max(best);
        }
        assert_eq!(best, 1);
    }

    #[test]
    fn tts_norm_sub_applies_captures() {
        let pattern = TtsNormPattern::new("([a-z]+)%s+").unwrap();
        assert_eq!(pattern.n_groups(), 1);
        let out = tts_norm_sub("foo bar", &pattern, "[%1]").unwrap();
        assert_eq!(out, "[foo]bar");
    }

    #[test]
    fn reasoning_budget_forces_then_ends() {
        let mut rb = ReasoningBudget::new(
            &[vec![1]],
            &[vec![2]],
            &[2, 3],
            1,
            ReasoningBudgetState::Idle,
        );
        rb.accept(1, None);
        assert_eq!(rb.state(), ReasoningBudgetState::Counting);
        rb.accept(4, None);
        assert_eq!(rb.state(), ReasoningBudgetState::Forcing);
        let mut cur = TokenDataArray::new(vec![
            crate::sampler::TokenData {
                id: 2,
                logit: 1.0,
                p: 0.0,
            },
            crate::sampler::TokenData {
                id: 3,
                logit: 2.0,
                p: 0.0,
            },
        ]);
        rb.apply(&mut cur);
        assert_eq!(cur.data[0].logit, 1.0);
        assert_eq!(cur.data[1].logit, f32::NEG_INFINITY);
        rb.accept(2, None);
        assert_eq!(rb.state(), ReasoningBudgetState::Done);
        assert_eq!(rb.end_match(), Some(&[2][..]));
    }
}

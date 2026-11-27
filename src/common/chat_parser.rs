//! Rust port of the chat parser family: `peg-parser.{h,cpp}`,
//! `chat-peg-parser.{h,cpp}`, `chat-auto-parser.h`,
//! `chat-auto-parser-helpers.{h,cpp}`, `chat-diff-analyzer.cpp` and the
//! individual parser definitions under `common/parsers/`.
//!
//! The engine and the AST-to-message mappers are translated directly. Three
//! C++ subsystems have no std-only counterpart in this crate and are omitted:
//! Jinja template rendering (drives differential analysis), JSON-schema to
//! GBNF generation (the schema wrapper parses through to its child) and the
//! vocabulary-backed special-token table (token parsers take explicit pieces).
//! Their data structures are kept so callers can supply the missing facts.

use crate::chat::{ChatMsg, ChatToolCall, ChatToolChoice};
use crate::core::{Error, Result, Token, LLAMA_TOKEN_NULL};
use crate::json::{Json, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

// ---------------------------------------------------------------------------
// peg-parser.h: ids, results, flags, tokens, contexts
// ---------------------------------------------------------------------------

pub type PegParserId = usize;
pub const COMMON_PEG_INVALID_PARSER_ID: PegParserId = usize::MAX;
pub type PegAstId = usize;
pub const COMMON_PEG_INVALID_AST_ID: PegAstId = usize::MAX;
pub const RULE_RECURSIVE_LIMIT: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PegParseResultType {
    Fail,
    Success,
    NeedMoreInput,
}

impl PegParseResultType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fail => "fail",
            Self::Success => "success",
            Self::NeedMoreInput => "need_more_input",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PegInvalidUtf8 {
    pub pos: usize,
    pub len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PegAstNode {
    pub id: PegAstId,
    pub rule: String,
    pub tag: String,
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub children: Vec<PegAstId>,
    pub is_partial: bool,
    pub invalid_utf8: Vec<PegInvalidUtf8>,
}

impl PegAstNode {
    /// Node text with each invalid UTF-8 run replaced by U+FFFD.
    pub fn sanitized_text(&self) -> String {
        if self.invalid_utf8.is_empty() {
            return self.text.clone();
        }
        let bytes = self.text.as_bytes();
        let mut out = Vec::with_capacity(bytes.len() + 2 * self.invalid_utf8.len());
        let mut seg_start = self.start;
        for invalid in &self.invalid_utf8 {
            let from = seg_start.saturating_sub(self.start).min(bytes.len());
            let to = invalid.pos.saturating_sub(self.start).min(bytes.len()).max(from);
            out.extend_from_slice(&bytes[from..to]);
            out.extend_from_slice(b"\xEF\xBF\xBD");
            seg_start = invalid.pos + invalid.len;
        }
        let from = seg_start.saturating_sub(self.start).min(bytes.len());
        let to = self.end.saturating_sub(self.start).min(bytes.len()).max(from);
        out.extend_from_slice(&bytes[from..to]);
        String::from_utf8_lossy(&out).into_owned()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PegParseFlags(pub u32);

impl PegParseFlags {
    pub const NONE: Self = Self(0);
    pub const LENIENT: Self = Self(1 << 0);
    pub const DEBUG: Self = Self(1 << 1);
    pub const COMPLETE: Self = Self(1 << 2);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
    pub const fn is_lenient(self) -> bool {
        self.contains(Self::LENIENT)
    }
    pub const fn is_debug(self) -> bool {
        self.contains(Self::DEBUG)
    }
    pub const fn is_complete(self) -> bool {
        self.contains(Self::COMPLETE)
    }
}

impl std::ops::BitOr for PegParseFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for PegParseFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl std::ops::BitAnd for PegParseFlags {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}
impl std::ops::BitAndAssign for PegParseFlags {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl std::ops::Not for PegParseFlags {
    type Output = Self;
    fn not(self) -> Self {
        Self(!self.0)
    }
}

impl Default for PegParseResult {
    fn default() -> Self {
        Self::new(PegParseResultType::Fail, 0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PegParseResult {
    pub result_type: PegParseResultType,
    pub start: usize,
    pub end: usize,
    pub nodes: Vec<PegAstId>,
    pub invalid_utf8: Vec<PegInvalidUtf8>,
}

impl PegParseResult {
    pub fn new(result_type: PegParseResultType, start: usize) -> Self {
        Self { result_type, start, end: start, nodes: Vec::new(), invalid_utf8: Vec::new() }
    }
    pub fn new_range(result_type: PegParseResultType, start: usize, end: usize) -> Self {
        Self { result_type, start, end, nodes: Vec::new(), invalid_utf8: Vec::new() }
    }
    pub fn with_nodes(
        result_type: PegParseResultType,
        start: usize,
        end: usize,
        nodes: Vec<PegAstId>,
        invalid_utf8: Vec<PegInvalidUtf8>,
    ) -> Self {
        Self { result_type, start, end, nodes, invalid_utf8 }
    }
    pub fn fail(&self) -> bool {
        self.result_type == PegParseResultType::Fail
    }
    pub fn need_more_input(&self) -> bool {
        self.result_type == PegParseResultType::NeedMoreInput
    }
    pub fn success(&self) -> bool {
        self.result_type == PegParseResultType::Success
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PegToken {
    pub text: String,
    pub attr: u32,
}

/// Special-token table. In C++ this is built from a `llama_vocab`; here the
/// caller supplies the pieces and ids directly.
#[derive(Clone, Debug, Default)]
pub struct PegTokens {
    pub tokens: HashMap<Token, PegToken>,
    pub ids: HashMap<String, Token>,
}

impl PegTokens {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: Token, text: impl Into<String>, attr: u32) {
        let text = text.into();
        self.ids.entry(text.clone()).or_insert(id);
        self.tokens.insert(id, PegToken { text, attr });
    }

    pub fn token_id(&self, text: &str) -> Token {
        self.ids.get(text).copied().unwrap_or(LLAMA_TOKEN_NULL)
    }
}

#[derive(Clone, Debug)]
pub struct PegParseContext {
    pub input: String,
    pub tokens: Vec<Token>,
    pub flags: PegParseFlags,
    pub ast: PegAstArena,
    pub parse_depth: usize,
    pub rule_depth: Vec<usize>,
}

impl PegParseContext {
    pub fn new(flags: PegParseFlags) -> Self {
        Self {
            input: String::new(),
            tokens: Vec::new(),
            flags,
            ast: PegAstArena::default(),
            parse_depth: 0,
            rule_depth: Vec::new(),
        }
    }

    pub fn with_input(input: impl Into<String>, flags: PegParseFlags) -> Self {
        let mut ctx = Self::new(flags);
        ctx.input = input.into();
        ctx
    }

    pub fn with_tokens(input: impl Into<String>, tokens: Vec<Token>, flags: PegParseFlags) -> Self {
        let mut ctx = Self::with_input(input, flags);
        // C++ asserts equal lengths; pad or truncate so callers cannot panic on
        // a partially tokenized stream.
        ctx.tokens = tokens;
        ctx.tokens.resize(ctx.input.len(), LLAMA_TOKEN_NULL);
        ctx
    }
}

// ---------------------------------------------------------------------------
// UTF-8 decoding
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Utf8Status {
    Success,
    Incomplete,
    Invalid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Utf8Parse {
    pub status: Utf8Status,
    pub codepoint: u32,
    pub bytes_consumed: usize,
}

/// Decode one codepoint at `pos`, matching `common_parse_utf8_codepoint`.
pub fn parse_utf8_codepoint(input: &[u8], pos: usize) -> Utf8Parse {
    if pos >= input.len() {
        return Utf8Parse { status: Utf8Status::Incomplete, codepoint: 0, bytes_consumed: 0 };
    }
    let first = input[pos];
    if first < 0x80 {
        return Utf8Parse { status: Utf8Status::Success, codepoint: first as u32, bytes_consumed: 1 };
    }
    let (len, mut cp, min_cp) = if (0xC2..=0xDF).contains(&first) {
        (2usize, (first & 0x1F) as u32, 0x80u32)
    } else if (0xE0..=0xEF).contains(&first) {
        (3usize, (first & 0x0F) as u32, 0x800u32)
    } else if (0xF0..=0xF4).contains(&first) {
        (4usize, (first & 0x07) as u32, 0x10000u32)
    } else {
        return Utf8Parse { status: Utf8Status::Invalid, codepoint: 0, bytes_consumed: 1 };
    };
    if pos + len > input.len() {
        let available = input.len() - pos;
        let complete = (1..available).all(|i| (input[pos + i] & 0xC0) == 0x80);
        return Utf8Parse {
            status: if complete { Utf8Status::Incomplete } else { Utf8Status::Invalid },
            codepoint: 0,
            bytes_consumed: available,
        };
    }
    for i in 1..len {
        let byte = input[pos + i];
        if byte & 0xC0 != 0x80 {
            return Utf8Parse { status: Utf8Status::Invalid, codepoint: 0, bytes_consumed: i };
        }
        cp = (cp << 6) | (byte & 0x3F) as u32;
    }
    if cp < min_cp || cp > 0x10FFFF || (0xD800..=0xDFFF).contains(&cp) {
        return Utf8Parse { status: Utf8Status::Invalid, codepoint: 0, bytes_consumed: 1 };
    }
    Utf8Parse { status: Utf8Status::Success, codepoint: cp, bytes_consumed: len }
}

// ---------------------------------------------------------------------------
// peg-parser.h: AST arena and parser variants
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct PegAstArena {
    pub nodes: Vec<PegAstNode>,
}

impl PegAstArena {
    #[allow(clippy::too_many_arguments)]
    pub fn add_node(
        &mut self,
        rule: impl Into<String>,
        tag: impl Into<String>,
        start: usize,
        end: usize,
        text: String,
        children: Vec<PegAstId>,
        is_partial: bool,
        invalid_utf8: Vec<PegInvalidUtf8>,
    ) -> PegAstId {
        let id = self.nodes.len();
        self.nodes.push(PegAstNode {
            id,
            rule: rule.into(),
            tag: tag.into(),
            start,
            end,
            text,
            children,
            is_partial,
            invalid_utf8,
        });
        id
    }

    pub fn get(&self, id: PegAstId) -> &PegAstNode {
        &self.nodes[id]
    }

    pub fn find_by_tag(&self, parent: &PegAstNode, tag: &str, max_depth: usize) -> PegAstId {
        for &child_id in &parent.children {
            let child = self.get(child_id);
            if child.tag == tag {
                return child_id;
            }
            if max_depth > 1 {
                let found = self.find_by_tag(child, tag, max_depth - 1);
                if found != COMMON_PEG_INVALID_AST_ID {
                    return found;
                }
            }
        }
        COMMON_PEG_INVALID_AST_ID
    }

    pub fn find_by_rule(&self, parent: &PegAstNode, rule: &str, max_depth: usize) -> PegAstId {
        for &child_id in &parent.children {
            let child = self.get(child_id);
            if child.rule == rule {
                return child_id;
            }
            if max_depth > 1 {
                let found = self.find_by_rule(child, rule, max_depth - 1);
                if found != COMMON_PEG_INVALID_AST_ID {
                    return found;
                }
            }
        }
        COMMON_PEG_INVALID_AST_ID
    }

    pub fn visit<F: FnMut(&PegAstNode)>(&self, id: PegAstId, visitor: &mut F) {
        let mut stack = vec![id];
        while let Some(cur) = stack.pop() {
            if cur == COMMON_PEG_INVALID_AST_ID {
                continue;
            }
            let node = self.get(cur);
            visitor(node);
            for &child in node.children.iter().rev() {
                stack.push(child);
            }
        }
    }

    pub fn visit_result<F: FnMut(&PegAstNode)>(&self, result: &PegParseResult, visitor: &mut F) {
        for &node in &result.nodes {
            self.visit(node, visitor);
        }
    }

    pub fn dump(&self) -> String {
        let mut out = String::new();
        for node in &self.nodes {
            Self::bfs_node(self, node, 0, &mut out);
        }
        out
    }

    fn bfs_node(arena: &Self, node: &PegAstNode, indent: usize, out: &mut String) {
        for _ in 0..indent {
            out.push_str("  ");
        }
        let _ = write!(out, "NODE {}", node.id);
        if !node.rule.is_empty() {
            let _ = write!(out, " (rule {})", node.rule);
        }
        if !node.tag.is_empty() {
            let _ = write!(out, " (tag {})", node.tag);
        }
        let _ = write!(out, " ['{}']\n", node.text);
        for &child in &node.children {
            Self::bfs_node(arena, arena.get(child), indent + 1, out);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PegCharRange {
    pub start: u32,
    pub end: u32,
}

impl PegCharRange {
    pub fn contains(self, cp: u32) -> bool {
        cp >= self.start && cp <= self.end
    }
}

#[derive(Clone, Debug)]
pub struct PegUntilBranch {
    pub delimiter: PegParserId,
    pub rest: PegParserId,
}

impl PegUntilBranch {
    pub fn new(delimiter: PegParserId) -> Self {
        Self { delimiter, rest: COMMON_PEG_INVALID_PARSER_ID }
    }
}

/// Literal delimiters used by an `until` parser, each owned by a branch.
#[derive(Clone, Debug, Default)]
pub struct PegUntilMatcher {
    pub patterns: Vec<(String, usize)>,
}

#[derive(Clone, Debug)]
pub enum PegParserVariant {
    Epsilon,
    Start,
    End,
    Literal(String),
    Token { token: Token, piece: String },
    TokenNot(Vec<Token>),
    Sequence(Vec<PegParserId>),
    Choice(Vec<PegParserId>),
    Repetition { child: PegParserId, min_count: i32, max_count: i32 },
    And(PegParserId),
    Not(PegParserId),
    Any,
    Space,
    Chars { pattern: String, ranges: Vec<PegCharRange>, negated: bool, min_count: i32, max_count: i32 },
    String { delimiter: char },
    Until {
        branches: Vec<PegUntilBranch>,
        include: bool,
        optional: bool,
        content: PegParserId,
        matcher: PegUntilMatcher,
    },
    Rule {
        name: String,
        child: PegParserId,
        trigger: bool,
        start: PegParserId,
        rest: PegParserId,
    },
    Ref(String),
    Atomic(PegParserId),
    Tag { child: PegParserId, tag: String },
    Gbnf { child: PegParserId, grammar: String },
}

#[derive(Clone, Debug)]
pub struct PegArena {
    pub parsers: Vec<PegParserVariant>,
    pub rules: HashMap<String, PegParserId>,
    pub root: PegParserId,
    pub tokens: PegTokens,
}

impl Default for PegArena {
    fn default() -> Self {
        Self {
            parsers: Vec::new(),
            rules: HashMap::new(),
            root: COMMON_PEG_INVALID_PARSER_ID,
            tokens: PegTokens::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PegMatchKind {
    None,
    Partial,
    Complete,
}

impl PegUntilMatcher {
    pub fn check_at(&self, input: &[u8], pos: usize) -> PegMatchKind {
        let rest = &input[pos..];
        let mut partial = false;
        for (pattern, _) in &self.patterns {
            let bytes = pattern.as_bytes();
            if rest.starts_with(bytes) {
                return PegMatchKind::Complete;
            }
            if rest.len() < bytes.len() && bytes.starts_with(rest) {
                partial = true;
            }
        }
        if partial { PegMatchKind::Partial } else { PegMatchKind::None }
    }

    pub fn branch_at(&self, input: &[u8], pos: usize) -> Option<usize> {
        let rest = &input[pos..];
        for (pattern, branch) in &self.patterns {
            if rest.starts_with(pattern.as_bytes()) {
                return Some(*branch);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// peg-parser.cpp: execution
// ---------------------------------------------------------------------------

fn is_hex_digit(c: u8) -> bool {
    c.is_ascii_hexdigit()
}

fn parse_hex_escape(bytes: &[u8], pos: usize, hex_count: usize) -> Option<(u32, usize)> {
    if pos + hex_count > bytes.len() {
        return None;
    }
    let mut value = 0u32;
    for &c in &bytes[pos..pos + hex_count] {
        if !is_hex_digit(c) {
            return None;
        }
        value = (value << 4) | (c as char).to_digit(16)?;
    }
    Some((value, hex_count))
}

fn parse_char_class_char(bytes: &[u8], pos: usize) -> (u32, usize) {
    if pos + 1 < bytes.len() && bytes[pos] == b'\\' {
        let next = bytes[pos + 1];
        let hex_count = match next {
            b'x' => Some(2),
            b'u' => Some(4),
            b'U' => Some(8),
            _ => None,
        };
        if let Some(count) = hex_count {
            if let Some((value, n)) = parse_hex_escape(bytes, pos + 2, count) {
                return (value, 2 + n);
            }
            return ((next as char) as u32, 2);
        }
        let value = match next {
            b'n' => b'\n',
            b't' => b'\t',
            b'r' => b'\r',
            other => other,
        };
        return (value as u32, 2);
    }
    (bytes[pos] as u32, 1)
}

fn parse_char_classes(classes: &str) -> (Vec<PegCharRange>, bool) {
    let bytes = classes.as_bytes();
    let mut start = 0usize;
    let mut end = bytes.len();
    if bytes.first() == Some(&b'[') {
        start += 1;
    }
    if end > start && bytes[end - 1] == b']' {
        end -= 1;
    }
    let mut negated = false;
    if end > start && bytes[start] == b'^' {
        negated = true;
        start += 1;
    }
    let content = &bytes[start..end];
    let mut ranges = Vec::new();
    let mut i = 0usize;
    while i < content.len() {
        let (first, first_len) = parse_char_class_char(content, i);
        i += first_len;
        if i + 1 < content.len() && content[i] == b'-' {
            let (last, last_len) = parse_char_class_char(content, i + 1);
            ranges.push(PegCharRange { start: first, end: last });
            i += 1 + last_len;
        } else {
            ranges.push(PegCharRange { start: first, end: first });
        }
    }
    (ranges, negated)
}

fn node_from_result(
    ctx: &mut PegParseContext,
    rule: &str,
    tag: &str,
    result: PegParseResult,
) -> PegAstId {
    let text = if result.start < ctx.input.len() {
        let end = result.end.min(ctx.input.len());
        String::from_utf8_lossy(&ctx.input.as_bytes()[result.start..end]).into_owned()
    } else {
        String::new()
    };
    let is_partial = result.need_more_input();
    ctx.ast.add_node(
        rule,
        tag,
        result.start,
        result.end,
        text,
        result.nodes,
        is_partial,
        result.invalid_utf8.clone(),
    )
}

struct PegExecutor<'a> {
    arena: &'a PegArena,
    ctx: &'a mut PegParseContext,
    start_pos: usize,
}

impl<'a> PegExecutor<'a> {
    fn parse(&mut self, id: PegParserId, pos: usize) -> Result<PegParseResult> {
        self.arena.parse_id(id, self.ctx, pos)
    }

    fn debug_line(&self, text: &str) {
        if self.ctx.flags.is_debug() {
            eprintln!("{}{}", "  ".repeat(self.ctx.parse_depth), text);
        }
    }

    fn input_snippet(&self, pos: usize, len: usize) -> String {
        if pos >= self.ctx.input.len() {
            return "<EOF>".to_owned();
        }
        let bytes = &self.ctx.input.as_bytes()[pos..(pos + len).min(self.ctx.input.len())];
        let mut out = String::new();
        for &c in bytes {
            match c {
                b'\n' => out.push_str("\\n"),
                b'\r' => out.push_str("\\r"),
                b'\t' => out.push_str("\\t"),
                _ => out.push(c as char),
            }
        }
        if pos + len < self.ctx.input.len() {
            out.push_str("...");
        }
        out
    }

    fn execute(&mut self, parser: &PegParserVariant) -> Result<PegParseResult> {
        use PegParserVariant as V;
        let start = self.start_pos;
        match parser {
            V::Epsilon => Ok(PegParseResult::new(PegParseResultType::Success, start)),
            V::Start => Ok(PegParseResult::new(
                if start == 0 { PegParseResultType::Success } else { PegParseResultType::Fail },
                start,
            )),
            V::End => Ok(PegParseResult::new(
                if start >= self.ctx.input.len() { PegParseResultType::Success } else { PegParseResultType::Fail },
                start,
            )),
            V::Literal(literal) => {
                let mut pos = start;
                for &expected in literal.as_bytes() {
                    if pos >= self.ctx.input.len() {
                        return Ok(if self.ctx.flags.is_lenient() {
                            PegParseResult::new_range(PegParseResultType::NeedMoreInput, start, pos)
                        } else {
                            PegParseResult::new(PegParseResultType::Fail, start)
                        });
                    }
                    if self.ctx.input.as_bytes()[pos] != expected {
                        return Ok(PegParseResult::new(PegParseResultType::Fail, start));
                    }
                    pos += 1;
                }
                Ok(PegParseResult::new_range(PegParseResultType::Success, start, pos))
            }
            V::Token { token, .. } => {
                if start >= self.ctx.input.len() {
                    return Ok(if self.ctx.flags.is_lenient() {
                        PegParseResult::new(PegParseResultType::NeedMoreInput, start)
                    } else {
                        PegParseResult::new(PegParseResultType::Fail, start)
                    });
                }
                if self.ctx.tokens.get(start) != Some(token) {
                    return Ok(PegParseResult::new(PegParseResultType::Fail, start));
                }
                let mut pos = start + 1;
                while pos < self.ctx.input.len() && self.ctx.tokens.get(pos) == Some(&LLAMA_TOKEN_NULL) {
                    pos += 1;
                }
                Ok(PegParseResult::new_range(PegParseResultType::Success, start, pos))
            }
            V::TokenNot(tokens) => {
                if start >= self.ctx.input.len() {
                    return Ok(if self.ctx.flags.is_lenient() {
                        PegParseResult::new(PegParseResultType::NeedMoreInput, start)
                    } else {
                        PegParseResult::new(PegParseResultType::Fail, start)
                    });
                }
                let current = self.ctx.tokens.get(start).copied().unwrap_or(LLAMA_TOKEN_NULL);
                if current == LLAMA_TOKEN_NULL || tokens.contains(&current) {
                    return Ok(PegParseResult::new(PegParseResultType::Fail, start));
                }
                let mut pos = start + 1;
                while pos < self.ctx.input.len() && self.ctx.tokens.get(pos) == Some(&LLAMA_TOKEN_NULL) {
                    pos += 1;
                }
                Ok(PegParseResult::new_range(PegParseResultType::Success, start, pos))
            }
            V::Sequence(children) => {
                self.debug_line(&format!(
                    "SEQ start at {} '{}' ({} children)",
                    start,
                    self.input_snippet(start, 60),
                    children.len()
                ));
                self.ctx.parse_depth += 1;
                let mut pos = start;
                let mut nodes = Vec::new();
                let mut invalid = Vec::new();
                for (i, &child_id) in children.iter().enumerate() {
                    self.debug_line(&format!("SEQ child {}: {}", i, self.arena.dump(child_id)));
                    let result = self.parse(child_id, pos)?;
                    if result.fail() {
                        self.ctx.parse_depth -= 1;
                        self.debug_line("SEQ -> FAIL");
                        return Ok(PegParseResult::new_range(PegParseResultType::Fail, start, result.end));
                    }
                    let need_more = result.need_more_input();
                    nodes.extend(result.nodes);
                    invalid.extend(result.invalid_utf8);
                    if need_more {
                        self.ctx.parse_depth -= 1;
                        self.debug_line("SEQ -> NEED_MORE");
                        return Ok(PegParseResult::with_nodes(
                            PegParseResultType::NeedMoreInput,
                            start,
                            result.end,
                            nodes,
                            invalid,
                        ));
                    }
                    pos = result.end;
                }
                self.ctx.parse_depth -= 1;
                self.debug_line(&format!("SEQ -> SUCCESS at {}->{}", start, pos));
                Ok(PegParseResult::with_nodes(PegParseResultType::Success, start, pos, nodes, invalid))
            }
            V::Choice(children) => {
                self.debug_line(&format!(
                    "CHOICE start at {} '{}' ({} options)",
                    start,
                    self.input_snippet(start, 60),
                    children.len()
                ));
                self.ctx.parse_depth += 1;
                for (i, &child_id) in children.iter().enumerate() {
                    self.debug_line(&format!("CHOICE option {}: {}", i, self.arena.dump(child_id)));
                    let result = self.parse(child_id, start)?;
                    if !result.fail() {
                        self.ctx.parse_depth -= 1;
                        self.debug_line(&format!(
                            "CHOICE -> {} (option {})",
                            result.result_type.name(),
                            i
                        ));
                        return Ok(result);
                    }
                }
                self.ctx.parse_depth -= 1;
                self.debug_line("CHOICE -> FAIL (no options matched)");
                Ok(PegParseResult::new(PegParseResultType::Fail, start))
            }
            V::Repetition { child, min_count, max_count } => {
                self.debug_line(&format!(
                    "REPEAT start at {} '{}' (min={}, max={})",
                    start,
                    self.input_snippet(start, 60),
                    min_count,
                    max_count
                ));
                self.ctx.parse_depth += 1;
                let mut pos = start;
                let mut count = 0i32;
                let mut nodes = Vec::new();
                let mut invalid = Vec::new();
                while *max_count == -1 || count < *max_count {
                    if pos >= self.ctx.input.len() {
                        break;
                    }
                    let result = self.parse(*child, pos)?;
                    if result.success() {
                        if result.end == pos {
                            break;
                        }
                        nodes.extend(result.nodes);
                        invalid.extend(result.invalid_utf8);
                        pos = result.end;
                        count += 1;
                        continue;
                    }
                    if result.need_more_input() {
                        nodes.extend(result.nodes);
                        invalid.extend(result.invalid_utf8);
                        self.ctx.parse_depth -= 1;
                        self.debug_line("REPEAT -> NEED_MORE");
                        return Ok(PegParseResult::with_nodes(
                            PegParseResultType::NeedMoreInput,
                            start,
                            result.end,
                            nodes,
                            invalid,
                        ));
                    }
                    break;
                }
                if *min_count > 0 && count < *min_count {
                    self.ctx.parse_depth -= 1;
                    if pos >= self.ctx.input.len() && self.ctx.flags.is_lenient() {
                        return Ok(PegParseResult::with_nodes(
                            PegParseResultType::NeedMoreInput,
                            start,
                            pos,
                            nodes,
                            invalid,
                        ));
                    }
                    return Ok(PegParseResult::with_nodes(
                        PegParseResultType::Fail,
                        start,
                        pos,
                        nodes,
                        invalid,
                    ));
                }
                self.ctx.parse_depth -= 1;
                self.debug_line(&format!("REPEAT -> SUCCESS (count={}, nodes={})", count, nodes.len()));
                Ok(PegParseResult::with_nodes(PegParseResultType::Success, start, pos, nodes, invalid))
            }
            V::And(child) => {
                let result = self.parse(*child, start)?;
                Ok(PegParseResult::new(result.result_type, start))
            }
            V::Not(child) => {
                let result = self.parse(*child, start)?;
                let result_type = if result.success() {
                    PegParseResultType::Fail
                } else if result.need_more_input() {
                    PegParseResultType::NeedMoreInput
                } else {
                    PegParseResultType::Success
                };
                Ok(PegParseResult::new(result_type, start))
            }
            V::Any => {
                let result = parse_utf8_codepoint(self.ctx.input.as_bytes(), start);
                match result.status {
                    Utf8Status::Incomplete => Ok(PegParseResult::new(
                        if self.ctx.flags.is_lenient() {
                            PegParseResultType::NeedMoreInput
                        } else {
                            PegParseResultType::Fail
                        },
                        start,
                    )),
                    Utf8Status::Invalid => Ok(PegParseResult::new(PegParseResultType::Fail, start)),
                    Utf8Status::Success => Ok(PegParseResult::new_range(
                        PegParseResultType::Success,
                        start,
                        start + result.bytes_consumed,
                    )),
                }
            }
            V::Space => {
                let mut pos = start;
                while pos < self.ctx.input.len() && self.ctx.input.as_bytes()[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                Ok(PegParseResult::new_range(PegParseResultType::Success, start, pos))
            }
            V::Chars { ranges, negated, min_count, max_count, .. } => {
                let mut pos = start;
                let mut count = 0i32;
                while *max_count == -1 || count < *max_count {
                    let step = parse_utf8_codepoint(self.ctx.input.as_bytes(), pos);
                    if step.status == Utf8Status::Incomplete {
                        return Ok(if count >= *min_count {
                            PegParseResult::new_range(PegParseResultType::Success, start, pos)
                        } else if self.ctx.flags.is_lenient() {
                            PegParseResult::new_range(PegParseResultType::NeedMoreInput, start, pos)
                        } else {
                            PegParseResult::new(PegParseResultType::Fail, start)
                        });
                    }
                    if step.status == Utf8Status::Invalid {
                        return Ok(if count >= *min_count {
                            PegParseResult::new_range(PegParseResultType::Success, start, pos)
                        } else {
                            PegParseResult::new_range(PegParseResultType::Fail, start, pos)
                        });
                    }
                    let mut matches = ranges.iter().any(|range| range.contains(step.codepoint));
                    if *negated {
                        matches = !matches;
                    }
                    if matches {
                        pos += step.bytes_consumed;
                        count += 1;
                    } else {
                        break;
                    }
                }
                if count < *min_count {
                    return Ok(if pos >= self.ctx.input.len() && self.ctx.flags.is_lenient() {
                        PegParseResult::new_range(PegParseResultType::NeedMoreInput, start, pos)
                    } else {
                        PegParseResult::new_range(PegParseResultType::Fail, start, pos)
                    });
                }
                Ok(PegParseResult::new_range(PegParseResultType::Success, start, pos))
            }
            V::String { delimiter } => self.execute_string(*delimiter),
            V::Until { branches, include, optional, content, matcher } => {
                if *include {
                    return self.parse_through(*content, branches, *optional);
                }
                let _ = matcher;
                let mut pos = start;
                let mut last_valid_pos = start;
                let mut invalid = Vec::new();
                while pos < self.ctx.input.len() {
                    let step = parse_utf8_codepoint(self.ctx.input.as_bytes(), pos);
                    if step.status == Utf8Status::Incomplete && self.ctx.flags.is_lenient() {
                        return Ok(PegParseResult::with_nodes(
                            PegParseResultType::NeedMoreInput,
                            start,
                            last_valid_pos,
                            Vec::new(),
                            invalid,
                        ));
                    }
                    if step.status != Utf8Status::Success {
                        let len = step.bytes_consumed.max(1);
                        invalid.push(PegInvalidUtf8 { pos, len });
                        pos += len;
                        last_valid_pos = pos;
                        continue;
                    }
                    let matched = matcher.check_at(self.ctx.input.as_bytes(), pos);
                    if matched == PegMatchKind::Complete
                        || (matched == PegMatchKind::Partial && !self.ctx.flags.is_complete())
                    {
                        return Ok(PegParseResult::with_nodes(
                            PegParseResultType::Success,
                            start,
                            pos,
                            Vec::new(),
                            invalid,
                        ));
                    }
                    pos += step.bytes_consumed;
                    last_valid_pos = pos;
                }
                if last_valid_pos == self.ctx.input.len() && self.ctx.flags.is_lenient() {
                    return Ok(PegParseResult::with_nodes(
                        PegParseResultType::NeedMoreInput,
                        start,
                        last_valid_pos,
                        Vec::new(),
                        invalid,
                    ));
                }
                Ok(PegParseResult::with_nodes(
                    PegParseResultType::Success,
                    start,
                    last_valid_pos,
                    Vec::new(),
                    invalid,
                ))
            }
            V::Rule { name, child, .. } => {
                if self.ctx.rule_depth.len() <= *child {
                    self.ctx.rule_depth.resize(*child + 1, 0);
                }
                if self.ctx.rule_depth[*child] >= RULE_RECURSIVE_LIMIT {
                    return Err(Error::new(format!(
                        "input is nested more than {} levels deep",
                        RULE_RECURSIVE_LIMIT
                    )));
                }
                self.ctx.rule_depth[*child] += 1;
                let result = self.parse(*child, start);
                self.ctx.rule_depth[*child] -= 1;
                let result = result?;
                if result.fail() {
                    return Ok(result);
                }
                let node_id = node_from_result(self.ctx, name.as_str(), "", result.clone());
                Ok(PegParseResult::with_nodes(
                    result.result_type,
                    result.start,
                    result.end,
                    vec![node_id],
                    result.invalid_utf8,
                ))
            }
            V::Ref(name) => {
                let id = self.arena.get_rule(name)?;
                self.parse(id, start)
            }
            V::Tag { child, tag } => {
                self.debug_line(&format!("TAG: {}", tag));
                let result = self.parse(*child, start)?;
                if result.fail() {
                    return Ok(result);
                }
                let node_id = node_from_result(self.ctx, "", tag.as_str(), result.clone());
                Ok(PegParseResult::with_nodes(
                    result.result_type,
                    result.start,
                    result.end,
                    vec![node_id],
                    result.invalid_utf8,
                ))
            }
            V::Atomic(child) => {
                let mut result = self.parse(*child, start)?;
                if result.need_more_input() {
                    result.nodes.clear();
                }
                Ok(result)
            }
            V::Gbnf { child, .. } => self.parse(*child, start),
        }
    }

    fn execute_string(&mut self, delimiter: char) -> Result<PegParseResult> {
        let start = self.start_pos;
        let mut pos = start;
        while pos < self.ctx.input.len() {
            let c = self.ctx.input.as_bytes()[pos];
            if c == delimiter as u8 {
                return Ok(PegParseResult::new_range(PegParseResultType::Success, start, pos));
            }
            if c == b'\\' {
                let save = pos;
                pos += 1;
                if pos >= self.ctx.input.len() {
                    pos = save;
                    return Ok(if self.ctx.flags.is_lenient() {
                        PegParseResult::new_range(PegParseResultType::NeedMoreInput, start, pos)
                    } else {
                        PegParseResult::new(PegParseResultType::Fail, start)
                    });
                }
                let next = self.ctx.input.as_bytes()[pos];
                if next == delimiter as u8
                    || matches!(next, b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't')
                {
                    pos += 1;
                    continue;
                }
                if next == b'u' {
                    pos += 1;
                    for _ in 0..4 {
                        if pos >= self.ctx.input.len() {
                            return Ok(if self.ctx.flags.is_lenient() {
                                PegParseResult::new_range(PegParseResultType::NeedMoreInput, start, pos)
                            } else {
                                PegParseResult::new(PegParseResultType::Fail, start)
                            });
                        }
                        if !is_hex_digit(self.ctx.input.as_bytes()[pos]) {
                            return Ok(PegParseResult::new(PegParseResultType::Fail, start));
                        }
                        pos += 1;
                    }
                    continue;
                }
                return Ok(PegParseResult::new(PegParseResultType::Fail, start));
            }
            let step = parse_utf8_codepoint(self.ctx.input.as_bytes(), pos);
            if step.status == Utf8Status::Incomplete {
                return Ok(if self.ctx.flags.is_lenient() {
                    PegParseResult::new_range(PegParseResultType::NeedMoreInput, start, pos)
                } else {
                    PegParseResult::new(PegParseResultType::Fail, start)
                });
            }
            if step.status == Utf8Status::Invalid {
                return Ok(PegParseResult::new(PegParseResultType::Fail, start));
            }
            pos += step.bytes_consumed;
        }
        Ok(if self.ctx.flags.is_lenient() {
            PegParseResult::new_range(PegParseResultType::NeedMoreInput, start, pos)
        } else {
            PegParseResult::new_range(PegParseResultType::Fail, start, pos)
        })
    }

    fn parse_through(
        &mut self,
        content_id: PegParserId,
        branches: &[PegUntilBranch],
        optional: bool,
    ) -> Result<PegParseResult> {
        let start = self.start_pos;
        let content = self.parse(content_id, start)?;
        if !content.success() {
            return Ok(content);
        }
        let pos = content.end;
        for branch in branches {
            let mut nodes = content.nodes.clone();
            let mut invalid = content.invalid_utf8.clone();
            let mut end = pos;
            let mut matched = true;
            for id in [branch.delimiter, branch.rest] {
                if id == COMMON_PEG_INVALID_PARSER_ID {
                    continue;
                }
                let result = self.parse(id, end)?;
                if result.fail() {
                    matched = false;
                    break;
                }
                let need_more = result.need_more_input();
                nodes.extend(result.nodes);
                invalid.extend(result.invalid_utf8);
                end = result.end;
                if need_more {
                    return Ok(PegParseResult::with_nodes(
                        PegParseResultType::NeedMoreInput,
                        start,
                        end,
                        nodes,
                        invalid,
                    ));
                }
            }
            if matched {
                return Ok(PegParseResult::with_nodes(
                    PegParseResultType::Success,
                    start,
                    end,
                    nodes,
                    invalid,
                ));
            }
        }
        if optional && pos == self.ctx.input.len() {
            let result_type = if self.ctx.flags.is_lenient() {
                PegParseResultType::NeedMoreInput
            } else {
                PegParseResultType::Success
            };
            return Ok(PegParseResult::with_nodes(
                result_type,
                start,
                pos,
                content.nodes,
                content.invalid_utf8,
            ));
        }
        Ok(PegParseResult::new_range(PegParseResultType::Fail, start, pos))
    }
}

impl PegArena {
    pub fn add_parser(&mut self, parser: PegParserVariant) -> PegParserId {
        let id = self.parsers.len();
        self.parsers.push(parser);
        id
    }

    pub fn add_rule(&mut self, name: &str, id: PegParserId) {
        self.rules.insert(name.to_owned(), id);
    }

    pub fn get_rule(&self, name: &str) -> Result<PegParserId> {
        self.rules.get(name).copied().ok_or_else(|| Error::new(format!("Rule not found: {}", name)))
    }

    pub fn has_rule(&self, name: &str) -> bool {
        self.rules.contains_key(name)
    }

    pub fn root(&self) -> PegParserId {
        self.root
    }

    pub fn set_root(&mut self, id: PegParserId) {
        self.root = id;
    }

    pub fn tokens(&self) -> &PegTokens {
        &self.tokens
    }

    pub fn get(&self, id: PegParserId) -> &PegParserVariant {
        &self.parsers[id]
    }

    pub fn parse(&self, ctx: &mut PegParseContext, start: usize) -> Result<PegParseResult> {
        if self.root == COMMON_PEG_INVALID_PARSER_ID {
            return Err(Error::new("No root parser set"));
        }
        ctx.rule_depth = vec![0; self.parsers.len()];
        self.parse_id(self.root, ctx, start)
    }

    pub fn parse_id(
        &self,
        id: PegParserId,
        ctx: &mut PegParseContext,
        start: usize,
    ) -> Result<PegParseResult> {
        let parser = self.parsers.get(id).ok_or_else(|| Error::new("Invalid parser id"))?;
        PegExecutor { arena: self, ctx, start_pos: start }.execute(parser)
    }

    pub fn resolve_ref(&self, id: PegParserId) -> Result<PegParserId> {
        match self.parsers.get(id) {
            Some(PegParserVariant::Ref(name)) => self.get_rule(name),
            _ => Ok(id),
        }
    }

    pub fn resolve_refs(&mut self) -> Result<()> {
        for id in 0..self.parsers.len() {
            let resolved = match &self.parsers[id] {
                PegParserVariant::Sequence(children) | PegParserVariant::Choice(children) => {
                    let ids = children.clone();
                    Some(ids.iter().map(|&c| self.resolve_ref(c)).collect::<Result<Vec<_>>>()?)
                }
                PegParserVariant::Repetition { child, .. }
                | PegParserVariant::And(child)
                | PegParserVariant::Not(child)
                | PegParserVariant::Atomic(child)
                | PegParserVariant::Tag { child, .. }
                | PegParserVariant::Gbnf { child, .. } => Some(vec![self.resolve_ref(*child)?]),
                PegParserVariant::Until { branches, content, .. } => {
                    let mut ids = Vec::new();
                    if *content != COMMON_PEG_INVALID_PARSER_ID {
                        ids.push(self.resolve_ref(*content)?);
                    }
                    for branch in branches {
                        ids.push(self.resolve_ref(branch.delimiter)?);
                        if branch.rest != COMMON_PEG_INVALID_PARSER_ID {
                            ids.push(self.resolve_ref(branch.rest)?);
                        }
                    }
                    Some(ids)
                }
                PegParserVariant::Rule { child, start, rest, .. } => {
                    let mut ids = vec![self.resolve_ref(*child)?];
                    if *start != COMMON_PEG_INVALID_PARSER_ID {
                        ids.push(self.resolve_ref(*start)?);
                        ids.push(self.resolve_ref(*rest)?);
                    }
                    Some(ids)
                }
                _ => None,
            };
            if let Some(ids) = resolved {
                let mut iter = ids.into_iter();
                match &mut self.parsers[id] {
                    PegParserVariant::Sequence(children) | PegParserVariant::Choice(children) => {
                        for child in children {
                            *child = iter.next().unwrap_or(COMMON_PEG_INVALID_PARSER_ID);
                        }
                    }
                    PegParserVariant::Repetition { child, .. }
                    | PegParserVariant::And(child)
                    | PegParserVariant::Not(child)
                    | PegParserVariant::Atomic(child)
                    | PegParserVariant::Tag { child, .. }
                    | PegParserVariant::Gbnf { child, .. } => {
                        *child = iter.next().unwrap_or(COMMON_PEG_INVALID_PARSER_ID);
                    }
                    PegParserVariant::Until { branches, content, .. } => {
                        if *content != COMMON_PEG_INVALID_PARSER_ID {
                            *content = iter.next().unwrap_or(COMMON_PEG_INVALID_PARSER_ID);
                        }
                        for branch in branches {
                            branch.delimiter = iter.next().unwrap_or(COMMON_PEG_INVALID_PARSER_ID);
                            if branch.rest != COMMON_PEG_INVALID_PARSER_ID {
                                branch.rest = iter.next().unwrap_or(COMMON_PEG_INVALID_PARSER_ID);
                            }
                        }
                    }
                    PegParserVariant::Rule { child, start, rest, .. } => {
                        *child = iter.next().unwrap_or(COMMON_PEG_INVALID_PARSER_ID);
                        if *start != COMMON_PEG_INVALID_PARSER_ID {
                            *start = iter.next().unwrap_or(COMMON_PEG_INVALID_PARSER_ID);
                            *rest = iter.next().unwrap_or(COMMON_PEG_INVALID_PARSER_ID);
                        }
                    }
                    _ => {}
                }
            }
        }
        if self.root != COMMON_PEG_INVALID_PARSER_ID {
            self.root = self.resolve_ref(self.root)?;
        }
        Ok(())
    }

    /// Collect the literal strings a delimiter parser can match.
    pub fn collect_delimiters(
        &self,
        id: PegParserId,
        from: &str,
        out: &mut Vec<String>,
    ) -> Result<()> {
        match self.parsers.get(id) {
            Some(PegParserVariant::Literal(literal)) => {
                out.push(format!("{}{}", from, literal));
            }
            Some(PegParserVariant::Token { piece, .. }) => {
                out.push(format!("{}{}", from, piece));
            }
            Some(PegParserVariant::Choice(children)) => {
                for &child in children {
                    self.collect_delimiters(child, from, out)?;
                }
            }
            Some(PegParserVariant::Sequence(children)) => {
                let mut prefix = from.to_owned();
                for &child in children {
                    let mut next = Vec::new();
                    self.collect_delimiters(child, &prefix, &mut next)?;
                    if next.is_empty() {
                        return Err(Error::new(
                            "a delimiter may only be a sequence of literals, tokens, and choices of them",
                        ));
                    }
                    prefix = next.remove(0);
                    out.extend(next);
                }
            }
            Some(PegParserVariant::Tag { child, .. })
            | Some(PegParserVariant::Atomic(child))
            | Some(PegParserVariant::Gbnf { child, .. }) => {
                self.collect_delimiters(*child, from, out)?;
            }
            Some(PegParserVariant::Ref(name)) => {
                let resolved = self.get_rule(name)?;
                self.collect_delimiters(resolved, from, out)?;
            }
            _ => {
                return Err(Error::new(
                    "a delimiter may only be a sequence of literals, tokens, and choices of them, possibly wrapped in tags and atomics",
                ));
            }
        }
        Ok(())
    }

    pub fn dump(&self, id: PegParserId) -> String {
        let mut visited = HashSet::new();
        self.dump_impl(id, &mut visited)
    }

    fn dump_impl(&self, id: PegParserId, visited: &mut HashSet<PegParserId>) -> String {
        if !visited.insert(id) {
            return "[cycle]".to_owned();
        }
        let Some(parser) = self.parsers.get(id) else {
            return "Unknown".to_owned();
        };
        match parser {
            PegParserVariant::Epsilon => "Epsilon".to_owned(),
            PegParserVariant::Start => "Start".to_owned(),
            PegParserVariant::End => "End".to_owned(),
            PegParserVariant::Literal(s) => format!("Literal({})", s),
            PegParserVariant::Token { piece, token } => format!("Token({}, {})", piece, token),
            PegParserVariant::TokenNot(tokens) => format!(
                "TokenNot({})",
                tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", ")
            ),
            PegParserVariant::Sequence(children) => format!(
                "Sequence({})",
                children.iter().map(|c| self.dump_impl(*c, visited)).collect::<Vec<_>>().join(", ")
            ),
            PegParserVariant::Choice(children) => format!(
                "Choice({})",
                children.iter().map(|c| self.dump_impl(*c, visited)).collect::<Vec<_>>().join(", ")
            ),
            PegParserVariant::Repetition { child, min_count, max_count } => {
                let max = if *max_count == -1 { "unbounded".to_owned() } else { max_count.to_string() };
                format!("Repetition({}, {}, {})", self.dump_impl(*child, visited), min_count, max)
            }
            PegParserVariant::And(child) => format!("And({})", self.dump_impl(*child, visited)),
            PegParserVariant::Not(child) => format!("Not({})", self.dump_impl(*child, visited)),
            PegParserVariant::Atomic(child) => format!("Atomic({})", self.dump_impl(*child, visited)),
            PegParserVariant::Gbnf { child, grammar } => {
                format!("Gbnf({}, {})", grammar, self.dump_impl(*child, visited))
            }
            PegParserVariant::Any => "Any".to_owned(),
            PegParserVariant::Space => "Space".to_owned(),
            PegParserVariant::Chars { pattern, min_count, max_count, .. } => {
                let max = if *max_count == -1 { "unbounded".to_owned() } else { max_count.to_string() };
                format!("CharRepeat({}, {}, {})", pattern, min_count, max)
            }
            PegParserVariant::String { delimiter } => format!("String({})", delimiter),
            PegParserVariant::Until { branches, include, optional, content, .. } => {
                let parts = branches
                    .iter()
                    .map(|b| {
                        let mut text = self.dump_impl(b.delimiter, visited);
                        if b.rest != COMMON_PEG_INVALID_PARSER_ID {
                            text.push(' ');
                            text.push_str(&self.dump_impl(b.rest, visited));
                        }
                        text
                    })
                    .collect::<Vec<_>>()
                    .join(" | ");
                if *include {
                    format!(
                        "Until({}, {}{})",
                        self.dump_impl(*content, visited),
                        parts,
                        if *optional { ", optional" } else { "" }
                    )
                } else {
                    format!("Until({})", parts)
                }
            }
            PegParserVariant::Rule { name, child, .. } => {
                format!("Rule({}, {})", name, self.dump_impl(*child, visited))
            }
            PegParserVariant::Ref(name) => format!("Ref({})", name),
            PegParserVariant::Tag { child, tag } => {
                format!("Tag({}, {})", tag, self.dump_impl(*child, visited))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// peg-parser.h: builder
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PegParser {
    pub id: PegParserId,
}

pub fn gbnf_format_rule_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_alphanumeric() || c == '_' || (i > 0 && c == '-') {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('_');
    }
    out
}

#[derive(Clone, Debug, Default)]
pub struct PegBuilder {
    pub arena: RefCell<PegArena>,
}

impl PegBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tokens(tokens: PegTokens) -> Self {
        let mut arena = PegArena::default();
        arena.tokens = tokens;
        Self { arena: RefCell::new(arena) }
    }

    fn wrap(&self, id: PegParserId) -> PegParser {
        PegParser { id }
    }

    fn add(&self, parser: PegParserVariant) -> PegParser {
        let id = self.arena.borrow_mut().add_parser(parser);
        self.wrap(id)
    }

    pub fn eps(&self) -> PegParser {
        self.add(PegParserVariant::Epsilon)
    }
    pub fn start(&self) -> PegParser {
        self.add(PegParserVariant::Start)
    }
    pub fn end(&self) -> PegParser {
        self.add(PegParserVariant::End)
    }
    pub fn literal(&self, literal: impl Into<String>) -> PegParser {
        self.add(PegParserVariant::Literal(literal.into()))
    }

    pub fn token(&self, piece: &str) -> PegParser {
        let token = self.arena.borrow().tokens.token_id(piece);
        if token == LLAMA_TOKEN_NULL {
            return self.literal(piece);
        }
        self.add(PegParserVariant::Token { token, piece: piece.to_owned() })
    }

    pub fn token_any(&self, pieces: &[&str]) -> PegParser {
        let alternatives: Vec<PegParser> = pieces.iter().map(|piece| self.token(piece)).collect();
        self.choice(&alternatives)
    }

    pub fn token_not(&self, pieces: &[&str]) -> PegParser {
        let mut tokens = Vec::new();
        for piece in pieces {
            let token = self.arena.borrow().tokens.token_id(piece);
            assert!(token != LLAMA_TOKEN_NULL, "token is not registered with the builder: {}", piece);
            tokens.push(token);
        }
        self.add(PegParserVariant::TokenNot(tokens))
    }

    pub fn sequence_ids(&self, parsers: &[PegParserId]) -> PegParser {
        let mut flattened = Vec::new();
        for &id in parsers {
            let nested = match self.arena.borrow().get(id) {
                PegParserVariant::Sequence(children) => Some(children.clone()),
                _ => None,
            };
            match nested {
                Some(children) => flattened.extend(children),
                None => flattened.push(id),
            }
        }
        self.add(PegParserVariant::Sequence(flattened))
    }

    pub fn sequence(&self, parsers: &[PegParser]) -> PegParser {
        let ids: Vec<PegParserId> = parsers.iter().map(|p| p.id).collect();
        self.sequence_ids(&ids)
    }

    pub fn choice_ids(&self, parsers: &[PegParserId]) -> PegParser {
        let mut flattened = Vec::new();
        for &id in parsers {
            let nested = match self.arena.borrow().get(id) {
                PegParserVariant::Choice(children) => Some(children.clone()),
                _ => None,
            };
            match nested {
                Some(children) => flattened.extend(children),
                None => flattened.push(id),
            }
        }
        self.add(PegParserVariant::Choice(flattened))
    }

    pub fn choice(&self, parsers: &[PegParser]) -> PegParser {
        let ids: Vec<PegParserId> = parsers.iter().map(|p| p.id).collect();
        self.choice_ids(&ids)
    }

    pub fn repeat(&self, child: PegParser, min: i32, max: i32) -> PegParser {
        self.add(PegParserVariant::Repetition { child: child.id, min_count: min, max_count: max })
    }
    pub fn repeat_exact(&self, child: PegParser, n: i32) -> PegParser {
        self.repeat(child, n, n)
    }
    pub fn one_or_more(&self, child: PegParser) -> PegParser {
        self.repeat(child, 1, -1)
    }
    pub fn zero_or_more(&self, child: PegParser) -> PegParser {
        self.repeat(child, 0, -1)
    }
    pub fn optional(&self, child: PegParser) -> PegParser {
        self.repeat(child, 0, 1)
    }
    pub fn peek(&self, child: PegParser) -> PegParser {
        self.add(PegParserVariant::And(child.id))
    }
    pub fn negate(&self, child: PegParser) -> PegParser {
        self.add(PegParserVariant::Not(child.id))
    }
    pub fn any(&self) -> PegParser {
        self.add(PegParserVariant::Any)
    }
    pub fn space(&self) -> PegParser {
        self.add(PegParserVariant::Space)
    }
    pub fn chars(&self, classes: &str, min: i32, max: i32) -> PegParser {
        let (ranges, negated) = parse_char_classes(classes);
        self.add(PegParserVariant::Chars {
            pattern: classes.to_owned(),
            ranges,
            negated,
            min_count: min,
            max_count: max,
        })
    }

    pub fn until(&self, delimiter: PegParser) -> PegParser {
        self.until_branches(&[PegUntilBranch::new(delimiter.id)], false, false, COMMON_PEG_INVALID_PARSER_ID)
    }

    pub fn until_str(&self, delimiter: &str) -> PegParser {
        let literal = self.literal(delimiter);
        self.until(literal)
    }

    pub fn until_one_of(&self, delimiters: &[&str]) -> PegParser {
        let branches: Vec<PegUntilBranch> = delimiters
            .iter()
            .map(|d| PegUntilBranch::new(self.literal(*d).id))
            .collect();
        self.until_branches(&branches, false, false, COMMON_PEG_INVALID_PARSER_ID)
    }

    pub fn rest(&self) -> PegParser {
        self.until_one_of(&[])
    }

    pub fn through(&self, content: PegParser, branches: &[PegUntilBranch], optional: bool) -> PegParser {
        assert!(!branches.is_empty(), "through requires at least one branch");
        self.until_branches(branches, true, optional, content.id)
    }

    fn until_branches(
        &self,
        branches: &[PegUntilBranch],
        include: bool,
        optional: bool,
        content: PegParserId,
    ) -> PegParser {
        let mut patterns = Vec::new();
        for (branch_index, branch) in branches.iter().enumerate() {
            let mut found = Vec::new();
            self.arena
                .borrow()
                .collect_delimiters(branch.delimiter, "", &mut found)
                .expect("invalid until delimiter");
            assert!(!found.is_empty(), "delimiter must not match the empty string");
            for pattern in found {
                patterns.push((pattern, branch_index));
            }
        }
        let mut seen: HashMap<String, usize> = HashMap::new();
        for (pattern, branch_index) in &patterns {
            if let Some(previous) = seen.insert(pattern.clone(), *branch_index) {
                if previous != *branch_index && include {
                    panic!("until branches {} and {} must not share a delimiter", previous, branch_index);
                }
            }
        }
        self.add(PegParserVariant::Until {
            branches: branches.to_vec(),
            include,
            optional,
            content,
            matcher: PegUntilMatcher { patterns },
        })
    }

    pub fn rule(&self, name: &str, child: PegParser, trigger: bool) -> PegParser {
        let clean = gbnf_format_rule_name(name);
        let id = self.arena.borrow_mut().add_parser(PegParserVariant::Rule {
            name: clean.clone(),
            child: child.id,
            trigger,
            start: COMMON_PEG_INVALID_PARSER_ID,
            rest: COMMON_PEG_INVALID_PARSER_ID,
        });
        self.arena.borrow_mut().add_rule(&clean, id);
        self.ref_(&clean)
    }

    pub fn rule_fn<F: FnOnce(&Self) -> PegParser>(
        &self,
        name: &str,
        builder_fn: F,
        trigger: bool,
    ) -> PegParser {
        let clean = gbnf_format_rule_name(name);
        if self.arena.borrow().has_rule(&clean) {
            return self.ref_(&clean);
        }
        let placeholder = self.any();
        let placeholder_id = self.arena.borrow_mut().add_parser(PegParserVariant::Rule {
            name: clean.clone(),
            child: placeholder.id,
            trigger,
            start: COMMON_PEG_INVALID_PARSER_ID,
            rest: COMMON_PEG_INVALID_PARSER_ID,
        });
        self.arena.borrow_mut().add_rule(&clean, placeholder_id);
        let parser = builder_fn(self);
        let id = self.arena.borrow_mut().add_parser(PegParserVariant::Rule {
            name: clean.clone(),
            child: parser.id,
            trigger,
            start: COMMON_PEG_INVALID_PARSER_ID,
            rest: COMMON_PEG_INVALID_PARSER_ID,
        });
        self.arena.borrow_mut().rules.insert(clean.clone(), id);
        self.ref_(&clean)
    }

    pub fn trigger_rule(&self, name: &str, child: PegParser) -> PegParser {
        self.rule(name, child, true)
    }

    pub fn trigger_rule_parts(&self, name: &str, start: PegParser, rest: PegParser) -> PegParser {
        let mut patterns = Vec::new();
        self.arena.borrow().collect_delimiters(start.id, "", &mut patterns).expect("invalid trigger start");
        assert!(!patterns.is_empty(), "trigger start must not match the empty string");
        let child = self.sequence(&[start, rest]);
        let clean = gbnf_format_rule_name(name);
        let id = self.arena.borrow_mut().add_parser(PegParserVariant::Rule {
            name: clean.clone(),
            child: child.id,
            trigger: true,
            start: start.id,
            rest: rest.id,
        });
        self.arena.borrow_mut().add_rule(&clean, id);
        self.ref_(&clean)
    }

    pub fn ref_(&self, name: &str) -> PegParser {
        self.add(PegParserVariant::Ref(name.to_owned()))
    }

    pub fn atomic(&self, child: PegParser) -> PegParser {
        self.add(PegParserVariant::Atomic(child.id))
    }

    pub fn tag(&self, tag: &str, child: PegParser) -> PegParser {
        self.add(PegParserVariant::Tag { child: child.id, tag: tag.to_owned() })
    }

    pub fn gbnf(&self, child: PegParser, grammar: &str) -> PegParser {
        self.add(PegParserVariant::Gbnf { child: child.id, grammar: grammar.to_owned() })
    }

    pub fn set_root(&self, parser: PegParser) {
        self.arena.borrow_mut().set_root(parser.id);
    }

    pub fn build(self) -> Result<PegArena> {
        let mut arena = self.arena.into_inner();
        arena.resolve_refs()?;
        Ok(arena)
    }

    // string primitives

    pub fn string_content(&self, delimiter: char) -> PegParser {
        self.add(PegParserVariant::String { delimiter })
    }

    pub fn double_quoted_string(&self) -> PegParser {
        self.rule_fn("double-quoted-string", |b| {
            let content = b.string_content('"');
            b.sequence(&[b.literal("\""), content, b.literal("\"")])
        }, false)
    }

    pub fn single_quoted_string(&self) -> PegParser {
        self.rule_fn("single-quoted-string", |b| {
            let content = b.string_content('\'');
            b.sequence(&[b.literal("'"), content, b.literal("'")])
        }, false)
    }

    pub fn quoted_string(&self) -> PegParser {
        self.rule_fn("quoted-string", |b| {
            let double = b.double_quoted_string();
            let single = b.single_quoted_string();
            b.choice(&[double, single])
        }, false)
    }

    // JSON primitives

    pub fn json_number(&self) -> PegParser {
        self.rule_fn("json-number", |b| {
            let digit1_9 = b.chars("[1-9]", 1, 1);
            let digits = b.chars("[0-9]", 1, -1);
            let zero = b.literal("0");
            let tail = b.chars("[0-9]", 0, -1);
            let int_part = b.choice(&[zero, b.sequence(&[digit1_9, tail])]);
            let frac = b.sequence(&[b.literal("."), digits]);
            let plus_minus = b.chars("[+-]", 1, 1);
            let exponent = b.choice(&[b.literal("e"), b.literal("E")]);
            let exp = b.sequence(&[exponent, b.optional(plus_minus), b.chars("[0-9]", 1, -1)]);
            let not_continuation = b.negate(b.chars("[0-9.eE+-]", 1, 1));
            let minus = b.literal("-");
            b.sequence(&[b.optional(minus), int_part, b.optional(frac), b.optional(exp), not_continuation])
        }, false)
    }

    pub fn json_string(&self) -> PegParser {
        self.rule_fn("json-string", |b| {
            let content = b.string_content('"');
            b.sequence(&[b.literal("\""), content, b.literal("\"")])
        }, false)
    }

    pub fn json_bool(&self) -> PegParser {
        self.rule_fn("json-bool", |b| {
            let yes = b.literal("true");
            let no = b.literal("false");
            b.choice(&[yes, no])
        }, false)
    }

    pub fn json_null(&self) -> PegParser {
        self.rule_fn("json-null", |b| b.literal("null"), false)
    }

    pub fn json_object(&self) -> PegParser {
        self.rule_fn("json-object", |b| {
            let ws = b.space();
            let key = b.json_string();
            let colon = b.literal(":");
            let value = b.json();
            let member = b.sequence(&[key, ws, colon, ws, value]);
            let comma_member = b.sequence(&[ws, b.literal(","), ws, member]);
            let members = b.sequence(&[member, b.zero_or_more(comma_member)]);
            let empty = b.literal("}");
            let full = b.sequence(&[members, ws, b.literal("}")]);
            b.sequence(&[b.literal("{"), ws, b.choice(&[empty, full])])
        }, false)
    }

    pub fn json_array(&self) -> PegParser {
        self.rule_fn("json-array", |b| {
            let ws = b.space();
            let value = b.json();
            let elements = b.sequence(&[value, b.zero_or_more(b.sequence(&[ws, b.literal(","), ws, value]))]);
            let empty = b.literal("]");
            let full = b.sequence(&[elements, ws, b.literal("]")]);
            b.sequence(&[b.literal("["), ws, b.choice(&[empty, full])])
        }, false)
    }

    pub fn json(&self) -> PegParser {
        self.rule_fn("json-value", |b| {
            let object = b.json_object();
            let array = b.json_array();
            let string = b.json_string();
            let number = b.json_number();
            let boolean = b.json_bool();
            let null = b.json_null();
            b.choice(&[object, array, string, number, boolean, null])
        }, false)
    }

    pub fn json_member(&self, key: &str, parser: PegParser) -> PegParser {
        let ws = self.space();
        self.sequence(&[self.literal(format!("\"{}\"", key)), ws, self.literal(":"), ws, parser])
    }

    // Python-format primitives

    pub fn python_string(&self) -> PegParser {
        self.rule_fn("python-string", |b| {
            let double = b.double_quoted_string();
            let single = b.single_quoted_string();
            b.choice(&[double, single])
        }, false)
    }

    pub fn python_number(&self) -> PegParser {
        self.json_number()
    }

    pub fn python_bool(&self) -> PegParser {
        self.rule_fn("python-bool", |b| {
            let yes = b.literal("True");
            let no = b.literal("False");
            b.choice(&[yes, no])
        }, false)
    }

    pub fn python_null(&self) -> PegParser {
        self.rule_fn("python-none", |b| b.literal("None"), false)
    }

    pub fn python_dict(&self) -> PegParser {
        self.rule_fn("python-dict", |b| {
            let ws = b.space();
            let key = b.python_string();
            let value = b.python_value();
            let colon = b.literal(":");
            let member = b.sequence(&[key, ws, colon, ws, value]);
            let comma_member = b.sequence(&[ws, b.literal(","), ws, member]);
            let members = b.sequence(&[member, b.zero_or_more(comma_member)]);
            let empty = b.literal("}");
            let full = b.sequence(&[members, ws, b.literal("}")]);
            b.sequence(&[b.literal("{"), ws, b.choice(&[empty, full]), ws])
        }, false)
    }

    pub fn python_array(&self) -> PegParser {
        self.rule_fn("python-array", |b| {
            let ws = b.space();
            let value = b.python_value();
            let elements = b.sequence(&[value, b.zero_or_more(b.sequence(&[b.literal(","), ws, value]))]);
            let empty = b.literal("]");
            let full = b.sequence(&[elements, ws, b.literal("]")]);
            b.sequence(&[b.literal("["), ws, b.choice(&[empty, full]), ws])
        }, false)
    }

    pub fn python_value(&self) -> PegParser {
        self.rule_fn("python-value", |b| {
            let dict = b.python_dict();
            let array = b.python_array();
            let string = b.python_string();
            let number = b.python_number();
            let boolean = b.python_bool();
            let null = b.python_null();
            b.choice(&[dict, array, string, number, boolean, null])
        }, false)
    }

    pub fn marker(&self) -> PegParser {
        let sharp = self.sequence(&[self.literal("<"), self.until_str(">"), self.literal(">")]);
        let square = self.sequence(&[self.literal("["), self.until_str("]"), self.literal("]")]);
        self.choice(&[sharp, square])
    }
}

pub fn build_peg_parser<F: FnOnce(&PegBuilder) -> PegParser>(build: F) -> Result<PegArena> {
    let mut builder = PegBuilder::new();
    let root = build(&mut builder);
    builder.set_root(root);
    builder.build()
}

// ---------------------------------------------------------------------------
// chat-peg-parser.cpp: string helpers
// ---------------------------------------------------------------------------

fn trim_trailing_space(s: &str) -> &str {
    s.trim_end_matches(|c: char| c.is_ascii_whitespace())
}

fn trim_leading_space(s: &str) -> &str {
    s.trim_start_matches(|c: char| c.is_ascii_whitespace())
}

fn trim_space(s: &str) -> &str {
    trim_trailing_space(trim_leading_space(s))
}

fn json_brace_depth(s: &str) -> i32 {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for c in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' && in_string {
            escaped = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            if c == '{' {
                depth += 1;
            } else if c == '}' {
                depth -= 1;
            }
        }
    }
    depth
}

/// JSON-escape a string and return the inner content, without quotes.
pub fn escape_json_string_inner(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            _ => out.push(c),
        }
    }
    out
}

/// Convert Python-style single-quoted strings to JSON double-quoted strings.
pub fn normalize_quotes_to_json(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut result = String::with_capacity(input.len() + 16);
    let mut in_single = false;
    let mut in_double = false;
    let is_word_char = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let literals: [(&[u8], &str); 3] = [(b"True", "true"), (b"False", "false"), (b"None", "null")];
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' && i + 1 < bytes.len() {
            let next = bytes[i + 1];
            if in_single {
                if next == b'\'' {
                    result.push('\'');
                } else if next == b'"' {
                    result.push_str("\\\"");
                } else {
                    result.push('\\');
                    result.push(next as char);
                }
                i += 2;
                continue;
            }
            if in_double {
                result.push('\\');
                result.push(next as char);
                i += 2;
                continue;
            }
            result.push('\\');
            i += 1;
            continue;
        }
        if (in_single || in_double) && c < 0x20 {
            result.push_str(&escape_json_string_inner(&(c as char).to_string()));
            i += 1;
            continue;
        }
        if c == b'"' {
            if in_single {
                result.push_str("\\\"");
            } else {
                in_double = !in_double;
                result.push('"');
            }
        } else if c == b'\'' {
            if in_double {
                result.push('\'');
            } else if in_single {
                in_single = false;
                result.push('"');
            } else {
                in_single = true;
                result.push('"');
            }
        } else if !in_single && !in_double && matches!(c, b'T' | b'F' | b'N')
            && (i == 0 || !is_word_char(bytes[i - 1]))
        {
            let mut n = 0usize;
            while i + n < bytes.len() && is_word_char(bytes[i + n]) {
                n += 1;
            }
            let token = &bytes[i..i + n];
            let mut matched = false;
            for (py, js) in literals {
                if py.len() >= n && &py[..n] == token {
                    result.push_str(&js[..n]);
                    i += n;
                    matched = true;
                    break;
                }
            }
            if !matched {
                result.push(c as char);
                i += 1;
            }
            continue;
        } else {
            result.push(c as char);
        }
        i += 1;
    }
    result
}

// ---------------------------------------------------------------------------
// chat-peg-parser.h: chat builder
// ---------------------------------------------------------------------------

pub const COMMON_CHAT_MAX_PERMUTE: usize = 6;

fn json_string_value(value: &Json) -> String {
    match value {
        Json::String(s) => s.clone(),
        _ => value.dump(0).unwrap_or_default(),
    }
}

/// Iterate the function tools selected by `tool_choice`.
pub fn foreach_function<F: FnMut(usize, &Json)>(tools: &Json, tool_choice: &ChatToolChoice, mut f: F) {
    let Json::Array(items) = tools else { return };
    for (i, tool) in items.iter().enumerate() {
        let Ok(function) = tool.at("function") else { continue };
        if let Ok(name) = function.at("name") {
            if !tool_choice.allows(&json_string_value(name)) {
                continue;
            }
        }
        f(i, function);
    }
}

/// Tool parameter list extracted from a function definition.
#[derive(Clone, Debug, Default)]
pub struct ToolParameter {
    pub name: String,
    pub is_string: bool,
    pub required: bool,
}

pub fn tool_parameters(function: &Json) -> Json {
    function.at("parameters").cloned().unwrap_or_else(|_| Json::object())
}

pub fn foreach_parameter<F: FnMut(usize, &ToolParameter)>(function: &Json, mut f: F) {
    let params = tool_parameters(function);
    let Ok(properties) = params.at("properties") else { return };
    let Json::Object(entries) = properties else { return };
    let required: HashSet<String> = match params.at("required") {
        Ok(Json::Array(values)) => values.iter().map(json_string_value).collect(),
        _ => HashSet::new(),
    };
    for (i, (name, definition)) in entries.iter().enumerate() {
        let is_string = definition.at("type").map(json_string_value).unwrap_or_default() == "string";
        f(i, &ToolParameter {
            name: name.clone(),
            is_string,
            required: required.contains(name),
        });
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChatPegBuilder {
    pub p: PegBuilder,
}

impl ChatPegBuilder {
    pub const REASONING_BLOCK: &'static str = "reasoning-block";
    pub const REASONING: &'static str = "reasoning";
    pub const CONTENT: &'static str = "content";
    pub const TOOL: &'static str = "tool";
    pub const TOOL_OPEN: &'static str = "tool-open";
    pub const TOOL_CLOSE: &'static str = "tool-close";
    pub const TOOL_ID: &'static str = "tool-id";
    pub const TOOL_NAME: &'static str = "tool-name";
    pub const TOOL_ARGS: &'static str = "tool-args";
    pub const TOOL_ARG: &'static str = "tool-arg";
    pub const TOOL_ARG_OPEN: &'static str = "tool-arg-open";
    pub const TOOL_ARG_CLOSE: &'static str = "tool-arg-close";
    pub const TOOL_ARG_NAME: &'static str = "tool-arg-name";
    pub const TOOL_ARG_VALUE: &'static str = "tool-arg-value";
    pub const TOOL_ARG_STRING_VALUE: &'static str = "tool-arg-string-value";

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tokens(tokens: PegTokens) -> Self {
        Self { p: PegBuilder::with_tokens(tokens) }
    }

    pub fn set_root(&self, parser: PegParser) {
        self.p.set_root(parser);
    }

    pub fn build(self) -> Result<PegArena> {
        self.p.build()
    }

    pub fn rule(&self, name: &str, child: PegParser, trigger: bool) -> PegParser {
        self.p.rule(name, child, trigger)
    }

    pub fn rule_fn<F: FnOnce(&PegBuilder) -> PegParser>(
        &self,
        name: &str,
        build: F,
        trigger: bool,
    ) -> PegParser {
        self.p.rule_fn(name, build, trigger)
    }

    pub fn trigger_rule(&self, name: &str, child: PegParser) -> PegParser {
        self.p.trigger_rule(name, child)
    }

    pub fn trigger_rule_parts(&self, name: &str, start: PegParser, rest: PegParser) -> PegParser {
        self.p.trigger_rule_parts(name, start, rest)
    }

    pub fn through(
        &self,
        content: PegParser,
        branches: &[PegUntilBranch],
        optional: bool,
    ) -> PegParser {
        self.p.through(content, branches, optional)
    }

    pub fn ref_(&self, name: &str) -> PegParser {
        self.p.ref_(name)
    }

    fn tag(&self, tag: &str, parser: PegParser) -> PegParser {
        self.p.tag(tag, parser)
    }

    pub fn reasoning_block(&self, parser: PegParser) -> PegParser {
        self.tag(Self::REASONING_BLOCK, parser)
    }
    pub fn reasoning(&self, parser: PegParser) -> PegParser {
        self.tag(Self::REASONING, parser)
    }
    pub fn content(&self, parser: PegParser) -> PegParser {
        self.tag(Self::CONTENT, parser)
    }
    pub fn tool(&self, parser: PegParser) -> PegParser {
        self.tag(Self::TOOL, parser)
    }
    pub fn tool_open(&self, parser: PegParser) -> PegParser {
        let tagged = self.tag(Self::TOOL_OPEN, parser);
        self.p.atomic(tagged)
    }
    pub fn tool_close(&self, parser: PegParser) -> PegParser {
        let tagged = self.tag(Self::TOOL_CLOSE, parser);
        self.p.atomic(tagged)
    }
    pub fn tool_id(&self, parser: PegParser) -> PegParser {
        let tagged = self.tag(Self::TOOL_ID, parser);
        self.p.atomic(tagged)
    }
    pub fn tool_name(&self, parser: PegParser) -> PegParser {
        let tagged = self.tag(Self::TOOL_NAME, parser);
        self.p.atomic(tagged)
    }
    pub fn tool_args(&self, parser: PegParser) -> PegParser {
        self.tag(Self::TOOL_ARGS, parser)
    }
    pub fn tool_arg(&self, parser: PegParser) -> PegParser {
        self.tag(Self::TOOL_ARG, parser)
    }
    pub fn tool_arg_open(&self, parser: PegParser) -> PegParser {
        let tagged = self.tag(Self::TOOL_ARG_OPEN, parser);
        self.p.atomic(tagged)
    }
    pub fn tool_arg_close(&self, parser: PegParser) -> PegParser {
        let tagged = self.tag(Self::TOOL_ARG_CLOSE, parser);
        self.p.atomic(tagged)
    }
    pub fn tool_arg_name(&self, parser: PegParser) -> PegParser {
        let tagged = self.tag(Self::TOOL_ARG_NAME, parser);
        self.p.atomic(tagged)
    }
    pub fn tool_arg_value(&self, parser: PegParser) -> PegParser {
        self.tag(Self::TOOL_ARG_VALUE, parser)
    }
    pub fn tool_arg_string_value(&self, parser: PegParser) -> PegParser {
        self.tag(Self::TOOL_ARG_STRING_VALUE, parser)
    }
    pub fn tool_arg_json_value(&self, parser: PegParser) -> PegParser {
        self.tag(Self::TOOL_ARG_VALUE, parser)
    }

    /// Schema wrapper. Parsing delegates entirely to the child; GBNF emission
    /// is unavailable because the schema-to-grammar module is not ported.
    pub fn schema(&self, parser: PegParser, _name: &str, _schema: &Json, _raw: bool) -> PegParser {
        parser
    }

    pub fn tag_with_safe_content(
        &self,
        tag_name: &str,
        marker: &str,
        parser: PegParser,
    ) -> PegParser {
        if marker.is_empty() {
            let any = self.p.any();
            let fallback = self.rule(tag_name, self.content(any), false);
            let choices = self.p.choice(&[parser, fallback]);
            return self.p.zero_or_more(choices);
        }
        let marker_literal = self.p.literal(marker);
        let not_marker = self.p.negate(marker_literal);
        let any = self.p.any();
        let until_marker = self.p.until_str(marker);
        let body = self.p.sequence(&[not_marker, any, until_marker]);
        let chunk = self.rule(tag_name, self.content(body), false);
        let choices = self.p.choice(&[parser, chunk]);
        self.p.zero_or_more(choices)
    }

    /// Match every parser exactly once, in any order.
    pub fn permute(&self, rule_prefix: &str, parsers: &[PegParser]) -> PegParser {
        if parsers.is_empty() {
            return self.p.eps();
        }
        if parsers.len() == 1 || parsers.len() > COMMON_CHAT_MAX_PERMUTE {
            return self.p.sequence(parsers);
        }
        let mut rules: BTreeMap<u32, PegParser> = BTreeMap::new();
        Self::remaining_of(self, parsers, rule_prefix, (1u32 << parsers.len()) - 1, &mut rules)
    }

    fn remaining_of(
        &self,
        parsers: &[PegParser],
        rule_prefix: &str,
        remaining: u32,
        rules: &mut BTreeMap<u32, PegParser>,
    ) -> PegParser {
        if remaining == 0 {
            return self.p.eps();
        }
        if let Some(found) = rules.get(&remaining) {
            return *found;
        }
        let mut alternatives = self.p.choice(&[]);
        for (i, parser) in parsers.iter().enumerate() {
            let bit = 1u32 << i;
            if remaining & bit != 0 {
                let tail = Self::remaining_of(self, parsers, rule_prefix, remaining & !bit, rules);
                let sequence = self.p.sequence(&[*parser, tail]);
                alternatives = self.p.choice_ids(&[alternatives.id, sequence.id]);
            }
        }
        let rule = self.rule(&format!("{}-{}", rule_prefix, remaining), alternatives, false);
        rules.insert(remaining, rule);
        rule
    }

    /// Parse the prefix of a string up to a delimiter.
    pub fn prefix(&self, s: &str, delimiter: &str) -> PegParser {
        if s.is_empty() {
            return self.p.eps();
        }
        if delimiter.is_empty() {
            return self.p.literal(s);
        }
        let end = s.find(delimiter).unwrap_or(s.len());
        self.p.literal(&s[..end])
    }

    /// Parse a tag with optional leading and trailing spaces.
    pub fn optspace(&self, tag: &str) -> PegParser {
        let bytes = tag.as_bytes();
        let mut end_of_prefix_space = bytes.len();
        for (i, &c) in bytes.iter().enumerate() {
            if !c.is_ascii_whitespace() {
                end_of_prefix_space = i;
                break;
            }
        }
        let mut start_of_suffix_space = bytes.len();
        for i in (0..bytes.len()).rev() {
            if !bytes[i].is_ascii_whitespace() {
                start_of_suffix_space = i + 1;
                break;
            }
        }
        let mut parser = self.p.eps();
        for &c in &bytes[..end_of_prefix_space] {
            let literal = self.p.literal((c as char).to_string());
            let optional = self.p.optional(literal);
            parser = self.p.sequence_ids(&[parser.id, optional.id]);
        }
        let middle = self.p.literal(&tag[end_of_prefix_space..start_of_suffix_space]);
        parser = self.p.sequence_ids(&[parser.id, middle.id]);
        for &c in &bytes[start_of_suffix_space..] {
            let literal = self.p.literal((c as char).to_string());
            let optional = self.p.optional(literal);
            parser = self.p.sequence_ids(&[parser.id, optional.id]);
        }
        parser
    }

    fn parse_key_spec(key: &str) -> (&str, &str) {
        match key.find('.') {
            Some(dot) => (&key[..dot], &key[dot + 1..]),
            None => ("", key),
        }
    }

    pub fn standard_constructed_tools(
        &self,
        markers: &BTreeMap<String, String>,
        tools: &Json,
        parallel_tool_calls: bool,
        force_tool_calls: bool,
    ) -> PegParser {
        if !tools.is_array() || tools.empty() {
            return self.p.eps();
        }
        let get = |key: &str, default: &str| {
            markers.get(key).map(String::as_str).unwrap_or(default).to_owned()
        };
        let section_start = get("tool_call_start_marker", "<tool_call>");
        let section_end = get("tool_call_end_marker", "</tool_call>");
        let func_opener = get("function_opener", "<function=");
        let func_suffix = get("function_name_suffix", ">");
        let func_closer = get("function_closer", "</function>");
        let param_prefix = get("parameter_key_prefix", "<param=");
        let param_suffix = get("parameter_key_suffix", ">");
        let param_closer = get("parameter_closer", "</param>");

        let choice = &ChatToolChoice::default();
        let mut tool_choices = self.p.choice(&[]);
        let mut tools_vec = Vec::new();
        foreach_function(tools, choice, |_, function| tools_vec.push(function.clone()));
        for (i, function) in tools_vec.iter().enumerate() {
            let name = function.at("name").map(json_string_value).unwrap_or_default();
            let params = tool_parameters(function);
            let mut args = self.p.eps();
            if let Ok(Json::Object(properties)) = params.at("properties") {
                if !properties.is_empty() {
                    let mut arg_choice = self.p.choice(&[]);
                    for (prop_name, _) in properties {
                        let plain = self.p.literal(prop_name);
                        let quoted = self.p.literal(format!("\"{}\"", prop_name));
                        let single = self.p.literal(format!("'{}'", prop_name));
                        let names = self.p.choice(&[plain, quoted, single]);
                        let name_tag = self.tool_arg_name(names);
                        let open = self.p.literal(&param_prefix);
                        let open_tag = self.tool_arg_open(open);
                        let suffix = self.p.literal(&param_suffix);
                        let until_closer = self.p.until_str(&param_closer);
                        let value = self.tool_arg_value(until_closer);
                        let closer = self.p.literal(&param_closer);
                        let close_tag = self.tool_arg_close(closer);
                        let body = self.p.sequence(&[open_tag, name_tag, suffix, value, close_tag]);
                        let arg = self.tool_arg(body);
                        arg_choice = self.p.choice_ids(&[arg_choice.id, arg.id]);
                    }
                    let space = self.p.space();
                    let repeat = self.p.zero_or_more(self.p.sequence(&[arg_choice, space]));
                    args = repeat;
                }
            }
            let opener = self.p.literal(&func_opener);
            let name_literal = self.p.literal(&name);
            let name_tag = self.tool_name(name_literal);
            let suffix = self.p.literal(&func_suffix);
            let open = self.tool_open(self.p.sequence(&[opener, name_tag, suffix]));
            let space1 = self.p.space();
            let args_tag = self.tool_args(args);
            let space2 = self.p.space();
            let closer = self.p.literal(&func_closer);
            let close_tag = self.tool_close(closer);
            let body = self.p.sequence(&[open, space1, args_tag, space2, close_tag]);
            let rule = self.rule(&format!("tool-{}", i), self.tool(body), false);
            tool_choices = self.p.choice_ids(&[tool_choices.id, rule.id]);
        }

        let start = self.p.literal(&section_start);
        let space = self.p.space();
        let end = self.p.literal(&section_end);
        let section = if parallel_tool_calls {
            let space2 = self.p.space();
            let many = self.p.one_or_more(self.p.sequence(&[tool_choices, space2]));
            self.trigger_rule("tool-call", self.p.sequence(&[start, space, many, end]))
        } else {
            let space2 = self.p.space();
            self.trigger_rule(
                "tool-call",
                self.p.sequence(&[start, space, tool_choices, space2, end]),
            )
        };
        if force_tool_calls { section } else { self.p.optional(section) }
    }

    pub fn python_or_json_value(&self) -> PegParser {
        self.p.rule_fn("python-or-json-value", |b| {
            let ws = b.space();
            let value = b.ref_("python-or-json-value");
            let key = b.python_string();
            let colon = b.literal(":");
            let member = b.sequence(&[key, ws, colon, ws, value]);
            let comma_member = b.sequence(&[ws, b.literal(","), ws, member]);
            let members = b.sequence(&[member, b.zero_or_more(comma_member)]);
            let empty = b.literal("}");
            let full = b.sequence(&[members, ws, b.literal("}")]);
            let open = b.literal("{");
            let dict_body = b.sequence(&[open, ws, b.choice(&[empty, full]), ws]);
            let dict = b.rule("python-or-json-dict", dict_body, false);
            let comma_value = b.sequence(&[b.literal(","), ws, b.ref_("python-or-json-value")]);
            let value_ref = b.ref_("python-or-json-value");
            let elements = b.sequence(&[value_ref, b.zero_or_more(comma_value)]);
            let empty_array = b.literal("]");
            let full_array = b.sequence(&[elements, ws, b.literal("]")]);
            let open_array = b.literal("[");
            let array_body = b.sequence(&[open_array, ws, b.choice(&[empty_array, full_array]), ws]);
            let array = b.rule("python-or-json-array", array_body, false);
            let string = b.python_string();
            let number = b.python_number();
            let bool_py = b.python_bool();
            let null_py = b.python_null();
            let bool_json = b.json_bool();
            let null_json = b.json_null();
            b.choice(&[dict, array, string, number, bool_py, null_py, bool_json, null_json])
        }, false)
    }

    pub fn python_style_tool_calls(
        &self,
        tools: &Json,
        parallel_tool_calls: bool,
        allow_json_literals: bool,
        allow_undeclared: bool,
    ) -> PegParser {
        if !tools.is_array() || tools.empty() {
            return self.p.eps();
        }
        let choice = &ChatToolChoice::default();
        let mut tools_vec = Vec::new();
        foreach_function(tools, choice, |_, function| tools_vec.push(function.clone()));
        let mut tool_choices = self.p.choice(&[]);
        for (i, function) in tools_vec.iter().enumerate() {
            let name = function.at("name").map(json_string_value).unwrap_or_default();
            let mut args = self.p.eps();
            let params = tool_parameters(function);
            if let Ok(Json::Object(properties)) = params.at("properties") {
                if !properties.is_empty() {
                    let mut arg_choice = self.p.choice(&[]);
                    for (prop_name, prop_def) in properties {
                        let is_string = prop_def.at("type").map(json_string_value).unwrap_or_default() == "string";
                        let name_parser = self.p.literal(prop_name);
                        let dq_content = self.p.string_content('"');
                        let dq = self.p.sequence(&[self.p.literal("\""), dq_content, self.p.literal("\"")]);
                        let sq_content = self.p.string_content('\'');
                        let sq = self.p.sequence(&[self.p.literal("'"), sq_content, self.p.literal("'")]);
                        let quoted = self.p.choice(&[dq, sq]);
                        let string_value = self.tool_arg_value(quoted);
                        let value = if is_string {
                            string_value
                        } else if allow_json_literals {
                            self.tool_arg_value(self.python_or_json_value())
                        } else {
                            self.tool_arg_value(self.p.python_value())
                        };
                        let name_tag = self.tool_arg_name(name_parser);
                        let equals = self.p.literal("=");
                        let open_tag = self.tool_arg_open(self.p.sequence(&[name_tag, equals]));
                        let close_tag = self.tool_arg_close(self.p.eps());
                        let arg = self.tool_arg(self.p.sequence(&[open_tag, value, close_tag]));
                        arg_choice = self.p.choice_ids(&[arg_choice.id, arg.id]);
                    }
                    let space = self.p.space();
                    let comma = self.p.literal(",");
                    let more = self.p.zero_or_more(self.p.sequence(&[comma, space, arg_choice]));
                    args = self.p.sequence(&[arg_choice, more]);
                }
            }
            let name_tag = self.tool_name(self.p.literal(&name));
            let open_paren = self.p.literal("(");
            let open = self.tool_open(self.p.sequence(&[name_tag, open_paren]));
            let space1 = self.p.space();
            let args_tag = self.tool_args(args);
            let space2 = self.p.space();
            let close = self.tool_close(self.p.literal(")"));
            let body = self.p.sequence(&[open, space1, args_tag, space2, close]);
            let rule = self.rule(&format!("tool-{}", i), self.tool(body), false);
            tool_choices = self.p.choice_ids(&[tool_choices.id, rule.id]);
        }
        if allow_undeclared {
            let first = self.p.chars("[a-zA-Z_]", 1, 1);
            let rest = self.p.chars("[a-zA-Z0-9_.]", 0, -1);
            let ident = self.rule("tool-undeclared-ident", self.p.sequence(&[first, rest]), false);
            let value = if allow_json_literals {
                self.tool_arg_value(self.python_or_json_value())
            } else {
                self.tool_arg_value(self.p.python_value())
            };
            let name_tag = self.tool_arg_name(ident);
            let equals = self.p.literal("=");
            let open_tag = self.tool_arg_open(self.p.sequence(&[name_tag, equals]));
            let close_tag = self.tool_arg_close(self.p.eps());
            let arg = self.tool_arg(self.p.sequence(&[open_tag, value, close_tag]));
            let space = self.p.space();
            let comma = self.p.literal(",");
            let more = self.p.zero_or_more(self.p.sequence(&[comma, space, arg]));
            let args = self.p.optional(self.p.sequence(&[arg, more]));
            let name_tag = self.tool_name(ident);
            let open_paren = self.p.literal("(");
            let open = self.tool_open(self.p.sequence(&[name_tag, open_paren]));
            let space1 = self.p.space();
            let args_tag = self.tool_args(args);
            let space2 = self.p.space();
            let close = self.tool_close(self.p.literal(")"));
            let body = self.p.sequence(&[open, space1, args_tag, space2, close]);
            let rule = self.rule("tool-undeclared", self.tool(body), false);
            tool_choices = self.p.choice_ids(&[tool_choices.id, rule.id]);
        }
        let open_bracket = self.p.literal("[");
        let space1 = self.p.space();
        let space2 = self.p.space();
        let close_bracket = self.p.literal("]");
        if parallel_tool_calls {
            let space = self.p.space();
            let comma = self.p.literal(",");
            let more = self.p.zero_or_more(self.p.sequence(&[comma, space, tool_choices]));
            self.p.sequence(&[open_bracket, space1, tool_choices, more, space2, close_bracket])
        } else {
            self.p.sequence(&[open_bracket, space1, tool_choices, space2, close_bracket])
        }
    }

    // Standard JSON tool layouts

    fn build_json_tools_function_is_key(
        &self,
        tools: &Json,
        tool_choice: &ChatToolChoice,
        args_key: &str,
        effective_args_key: &str,
        call_id_key: &str,
        gen_call_id_key: &str,
    ) -> PegParser {
        let mut tools_vec = Vec::new();
        foreach_function(tools, tool_choice, |i, function| tools_vec.push((i, function.clone())));
        let mut tool_choices = self.p.choice(&[]);
        for (i, function) in tools_vec {
            let name = function.at("name").map(json_string_value).unwrap_or_default();
            let params = tool_parameters(&function);
            let mut inner_fields: Vec<PegParser> = Vec::new();
            if !call_id_key.is_empty() {
                let key = self.p.literal(format!("\"{}\"", call_id_key));
                let ws1 = self.p.space();
                let colon = self.p.literal(":");
                let ws2 = self.p.space();
                let quote = self.p.literal("\"");
                let content = self.p.string_content('"');
                let id = self.tool_id(content);
                let end_quote = self.p.literal("\"");
                let atomic = self.p.atomic(self.p.sequence(&[key, ws1, colon, ws2, quote, id, end_quote]));
                let trailing_ws = self.p.space();
                let comma = self.p.literal(",");
                let trailing = self.p.optional(self.p.sequence(&[comma, trailing_ws]));
                let field = self.p.optional(self.p.sequence(&[atomic, trailing_ws, trailing]));
                inner_fields.push(field);
            }
            if !gen_call_id_key.is_empty() {
                let key = self.p.literal(format!("\"{}\"", gen_call_id_key));
                let ws1 = self.p.space();
                let colon = self.p.literal(":");
                let ws2 = self.p.space();
                let quote = self.p.literal("\"");
                let content = self.p.string_content('"');
                let id = self.tool_id(content);
                let end_quote = self.p.literal("\"");
                let string_id = self.p.sequence(&[quote, id, end_quote]);
                let number_id = self.tool_id(self.p.json_number());
                let value = self.p.choice(&[string_id, number_id]);
                let atomic = self.p.atomic(self.p.sequence(&[key, ws1, colon, ws2, value]));
                let trailing_ws = self.p.space();
                let comma = self.p.literal(",");
                let trailing = self.p.optional(self.p.sequence(&[comma, trailing_ws]));
                let field = self.p.optional(self.p.sequence(&[atomic, trailing_ws, trailing]));
                inner_fields.push(field);
            }
            let args_json = self.p.json();
            let args_tag = self.tool_args(self.schema(args_json, &format!("tool-{}-schema", i), &params, false));
            let args_parser = if args_key.is_empty() {
                args_tag
            } else {
                let key = self.p.literal(format!("\"{}\"", effective_args_key));
                let ws1 = self.p.space();
                let colon = self.p.literal(":");
                let ws2 = self.p.space();
                self.p.sequence(&[key, ws1, colon, ws2, args_tag])
            };
            inner_fields.push(args_parser);

            let inner_object = if args_key.is_empty() && inner_fields.len() == 1 {
                inner_fields[0]
            } else {
                let mut body = self.p.sequence(&[self.p.literal("{"), self.p.space()]);
                for (n, field) in inner_fields.iter().enumerate() {
                    body = self.p.sequence_ids(&[body.id, field.id]);
                    if n + 1 < inner_fields.len() {
                        let space = self.p.space();
                        body = self.p.sequence_ids(&[body.id, space.id]);
                    }
                }
                let space = self.p.space();
                let close = self.p.literal("}");
                self.p.sequence(&[body, space, close])
            };
            let open = self.tool_open(self.p.literal("{"));
            let space1 = self.p.space();
            let quote = self.p.literal("\"");
            let name_tag = self.tool_name(self.p.literal(&name));
            let end_quote = self.p.literal("\"");
            let space2 = self.p.space();
            let colon = self.p.literal(":");
            let space3 = self.p.space();
            let space4 = self.p.space();
            let close = self.tool_close(self.p.literal("}"));
            let body = self.p.sequence(&[
                open, space1, quote, name_tag, end_quote, space2, colon, space3, inner_object, space4, close,
            ]);
            let rule = self.rule(&format!("tool-{}", i), self.tool(body), false);
            tool_choices = self.p.choice_ids(&[tool_choices.id, rule.id]);
        }
        tool_choices
    }

    fn build_json_tools_nested_keys(
        &self,
        tools: &Json,
        tool_choice: &ChatToolChoice,
        effective_name_key: &str,
        effective_args_key: &str,
        call_id_key: &str,
        gen_call_id_key: &str,
    ) -> PegParser {
        let name_spec = Self::parse_key_spec(effective_name_key);
        let args_spec = Self::parse_key_spec(effective_args_key);
        let nested_prefix = if !name_spec.0.is_empty() { name_spec.0 } else { args_spec.0 };
        let nested_name_field = if !name_spec.0.is_empty() { name_spec.1 } else { effective_name_key };
        let nested_args_field = if !args_spec.0.is_empty() { args_spec.1 } else { effective_args_key };
        let mut tools_vec = Vec::new();
        foreach_function(tools, tool_choice, |i, function| tools_vec.push((i, function.clone())));
        let mut tool_choices = self.p.choice(&[]);
        for (i, function) in tools_vec {
            let name = function.at("name").map(json_string_value).unwrap_or_default();
            let params = tool_parameters(&function);
            let key = self.p.literal(format!("\"{}\"", nested_name_field));
            let ws1 = self.p.space();
            let colon = self.p.literal(":");
            let ws2 = self.p.space();
            let quote = self.p.literal("\"");
            let name_tag = self.tool_name(self.p.literal(&name));
            let end_quote = self.p.literal("\"");
            let atomic = self.p.atomic(self.p.sequence(&[quote, name_tag, end_quote]));
            let nested_name = self.p.sequence(&[key, ws1, colon, ws2, atomic]);
            let key = self.p.literal(format!("\"{}\"", nested_args_field));
            let ws1 = self.p.space();
            let colon = self.p.literal(":");
            let ws2 = self.p.space();
            let args_json = self.p.json();
            let args_tag = self.tool_args(self.schema(args_json, &format!("tool-{}-schema", i), &params, false));
            let nested_args = self.p.sequence(&[key, ws1, colon, ws2, args_tag]);
            let space1 = self.p.space();
            let comma = self.p.literal(",");
            let space2 = self.p.space();
            let space3 = self.p.space();
            let close = self.p.literal("}");
            let nested_object = self.p.sequence(&[
                self.p.literal("{"), space1, nested_name, comma, space2, nested_args, space3, close,
            ]);

            let mut body = self.p.sequence(&[self.tool_open(self.p.literal("{")), self.p.space()]);
            if !call_id_key.is_empty() {
                let id_spec = Self::parse_key_spec(call_id_key);
                if id_spec.0.is_empty() {
                    let key = self.p.literal(format!("\"{}\"", call_id_key));
                    let ws1 = self.p.space();
                    let colon = self.p.literal(":");
                    let ws2 = self.p.space();
                    let quote = self.p.literal("\"");
                    let content = self.p.string_content('"');
                    let id = self.tool_id(content);
                    let end_quote = self.p.literal("\"");
                    let atomic = self.p.atomic(self.p.sequence(&[key, ws1, colon, ws2, quote, id, end_quote]));
                    let ws3 = self.p.space();
                    let comma = self.p.literal(",");
                    let ws4 = self.p.space();
                    let field = self.p.optional(self.p.sequence(&[atomic, ws3, comma, ws4]));
                    body = self.p.sequence_ids(&[body.id, field.id]);
                }
            }
            if !gen_call_id_key.is_empty() {
                let id_spec = Self::parse_key_spec(gen_call_id_key);
                if id_spec.0.is_empty() {
                    let key = self.p.literal(format!("\"{}\"", gen_call_id_key));
                    let ws1 = self.p.space();
                    let colon = self.p.literal(":");
                    let ws2 = self.p.space();
                    let quote = self.p.literal("\"");
                    let content = self.p.string_content('"');
                    let id = self.tool_id(content);
                    let end_quote = self.p.literal("\"");
                    let string_id = self.p.sequence(&[quote, id, end_quote]);
                    let number_id = self.tool_id(self.p.json_number());
                    let value = self.p.choice(&[string_id, number_id]);
                    let atomic = self.p.atomic(self.p.sequence(&[key, ws1, colon, ws2, value]));
                    let ws3 = self.p.space();
                    let comma = self.p.literal(",");
                    let ws4 = self.p.space();
                    let field = self.p.optional(self.p.sequence(&[atomic, ws3, comma, ws4]));
                    body = self.p.sequence_ids(&[body.id, field.id]);
                }
            }
            let key = self.p.literal(format!("\"{}\"", nested_prefix));
            let ws1 = self.p.space();
            let colon = self.p.literal(":");
            let ws2 = self.p.space();
            let nested_field = self.p.sequence(&[key, ws1, colon, ws2, nested_object]);
            let ws3 = self.p.space();
            let close = self.tool_close(self.p.literal("}"));
            body = self.p.sequence(&[body, nested_field, ws3, close]);
            let rule = self.rule(&format!("tool-{}", i), self.tool(body), false);
            tool_choices = self.p.choice_ids(&[tool_choices.id, rule.id]);
        }
        tool_choices
    }

    fn build_json_tools_flat_keys(
        &self,
        tools: &Json,
        tool_choice: &ChatToolChoice,
        effective_name_key: &str,
        effective_args_key: &str,
        call_id_key: &str,
        gen_call_id_key: &str,
        parameters_order: &[String],
        accept_openai_wrapper: bool,
    ) -> PegParser {
        let name_key_literal = self.p.literal(format!("\"{}\"", effective_name_key));
        let args_key_literal = self.p.literal(format!("\"{}\"", effective_args_key));
        let mut tools_vec = Vec::new();
        foreach_function(tools, tool_choice, |i, function| tools_vec.push((i, function.clone())));
        let mut tool_choices = self.p.choice(&[]);
        for (i, function) in tools_vec {
            let name = function.at("name").map(json_string_value).unwrap_or_default();
            let params = tool_parameters(&function);
            let ws1 = self.p.space();
            let colon = self.p.literal(":");
            let ws2 = self.p.space();
            let quote = self.p.literal("\"");
            let name_tag = self.tool_name(self.p.literal(&name));
            let end_quote = self.p.literal("\"");
            let atomic = self.p.atomic(self.p.sequence(&[quote, name_tag, end_quote]));
            let tool_name_parser = self.p.sequence(&[name_key_literal, ws1, colon, ws2, atomic]);
            let ws3 = self.p.space();
            let colon2 = self.p.literal(":");
            let ws4 = self.p.space();
            let args_json = self.p.json();
            let args_tag = self.tool_args(self.schema(args_json, &format!("tool-{}-schema", i), &params, false));
            let tool_args_parser = self.p.sequence(&[args_key_literal, ws3, colon2, ws4, args_tag]);

            let make_id = |b: &Self, key: &str| -> PegParser {
                let k = b.p.literal(format!("\"{}\"", key));
                let ws1 = b.p.space();
                let colon = b.p.literal(":");
                let ws2 = b.p.space();
                let quote = b.p.literal("\"");
                let content = b.p.string_content('"');
                let id = b.tool_id(content);
                let end_quote = b.p.literal("\"");
                let string_id = b.p.sequence(&[quote, id, end_quote]);
                let number_id = b.tool_id(b.p.json_number());
                let value = b.p.choice(&[string_id, number_id]);
                b.p.atomic(b.p.sequence(&[k, ws1, colon, ws2, value]))
            };
            let mut pairs: Vec<(PegParser, &str)> = vec![
                (tool_name_parser, effective_name_key),
                (tool_args_parser, effective_args_key),
            ];
            if !call_id_key.is_empty() {
                let id = make_id(self, call_id_key);
                pairs.push((self.p.optional(id), call_id_key));
            }
            if !gen_call_id_key.is_empty() {
                let id = make_id(self, gen_call_id_key);
                pairs.push((self.p.optional(id), gen_call_id_key));
            }
            pairs.sort_by_key(|(_, key)| {
                parameters_order.iter().position(|k| k == key).unwrap_or(parameters_order.len())
            });

            let mut ordered_body = self.p.sequence(&[self.tool_open(self.p.literal("{")), self.p.space()]);
            if accept_openai_wrapper {
                let key = self.p.literal("\"type\"");
                let ws1 = self.p.space();
                let colon = self.p.literal(":");
                let ws2 = self.p.space();
                let value = self.p.literal("\"function\"");
                let ws3 = self.p.space();
                let comma = self.p.literal(",");
                let ws4 = self.p.space();
                let field = self.p.optional(self.p.sequence(&[key, ws1, colon, ws2, value, ws3, comma, ws4]));
                ordered_body = self.p.sequence_ids(&[ordered_body.id, field.id]);
            }
            for (n, (parser, _)) in pairs.iter().enumerate() {
                ordered_body = self.p.sequence_ids(&[ordered_body.id, parser.id]);
                if n + 1 < pairs.len() {
                    let ws = self.p.space();
                    let comma = self.p.literal(",");
                    let ws2 = self.p.space();
                    ordered_body = self.p.sequence(&[ordered_body, ws, comma, ws2]);
                }
            }
            let ws = self.p.space();
            let close = self.tool_close(self.p.literal("}"));
            ordered_body = self.p.sequence(&[ordered_body, ws, close]);
            let rule = self.rule(&format!("tool-{}", i), self.tool(ordered_body), false);
            tool_choices = self.p.choice_ids(&[tool_choices.id, rule.id]);
        }
        tool_choices
    }

    pub fn standard_json_tools(
        &self,
        section_start: &str,
        section_end: &str,
        tools: &Json,
        parallel_tool_calls: bool,
        force_tool_calls: bool,
        name_key: &str,
        args_key: &str,
        array_wrapped: bool,
        function_is_key: bool,
        call_id_key: &str,
        gen_call_id_key: &str,
        parameters_order: &[String],
        accept_openai_wrapper: bool,
        tool_choice: &ChatToolChoice,
    ) -> PegParser {
        if !tools.is_array() || tools.empty() {
            return self.p.eps();
        }
        let effective_name_key = if name_key.is_empty() { "name" } else { name_key };
        let effective_args_key = if args_key.is_empty() { "arguments" } else { args_key };
        let tool_choices = if function_is_key {
            self.build_json_tools_function_is_key(
                tools, tool_choice, args_key, effective_args_key, call_id_key, gen_call_id_key,
            )
        } else {
            let name_spec = Self::parse_key_spec(effective_name_key);
            let args_spec = Self::parse_key_spec(effective_args_key);
            if !name_spec.0.is_empty() || !args_spec.0.is_empty() {
                self.build_json_tools_nested_keys(
                    tools, tool_choice, effective_name_key, effective_args_key, call_id_key, gen_call_id_key,
                )
            } else {
                self.build_json_tools_flat_keys(
                    tools, tool_choice, effective_name_key, effective_args_key, call_id_key, gen_call_id_key,
                    parameters_order, accept_openai_wrapper,
                )
            }
        };
        let mut tool_calls = tool_choices;
        if parallel_tool_calls {
            let ws = self.p.space();
            let comma = self.p.literal(",");
            let ws2 = self.p.space();
            let more = self.p.zero_or_more(self.p.sequence(&[ws, comma, ws2, tool_choices]));
            tool_calls = self.p.sequence(&[tool_calls, more]);
        }
        if array_wrapped {
            let ws1 = self.p.space();
            let ws2 = self.p.space();
            tool_calls = self.p.sequence(&[self.p.literal("["), ws1, tool_calls, ws2, self.p.literal("]")]);
        }
        let start = self.p.literal(section_start);
        let ws1 = self.p.space();
        let ws2 = self.p.space();
        let end = self.p.literal(section_end);
        let section = self.trigger_rule("tool-call", self.p.sequence(&[start, ws1, tool_calls, ws2, end]));
        if force_tool_calls { section } else { self.p.optional(section) }
    }
}

pub fn build_chat_peg_parser<F: FnOnce(&ChatPegBuilder) -> PegParser>(build: F) -> Result<PegArena> {
    let mut builder = ChatPegBuilder::new();
    let root = build(&mut builder);
    builder.set_root(root);
    builder.build()
}

pub fn build_chat_peg_parser_with_tokens<F: FnOnce(&ChatPegBuilder) -> PegParser>(
    tokens: PegTokens,
    build: F,
) -> Result<PegArena> {
    let mut builder = ChatPegBuilder::with_tokens(tokens);
    let root = build(&mut builder);
    builder.set_root(root);
    builder.build()
}

// ---------------------------------------------------------------------------
// chat-peg-parser.cpp: AST to ChatMsg mappers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CurrentTool {
    None,
    Pending,
    Result(usize),
}

impl Default for CurrentTool {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChatPegMapper {
    pub result: ChatMsg,
    pending_tool_call: Option<ChatToolCall>,
    current_tool: CurrentTool,
    arg_count: i32,
    closing_quote_pending: bool,
    args_buffer: String,
}

impl ChatPegMapper {
    pub fn new(result: ChatMsg) -> Self {
        Self { result, ..Self::default() }
    }

    fn args_target(&mut self) -> &mut String {
        if let CurrentTool::Result(index) = self.current_tool {
            if !self.result.tool_calls[index].name.is_empty() {
                return &mut self.result.tool_calls[index].arguments;
            }
        }
        &mut self.args_buffer
    }

    fn normalize_container_value(&self, input: &str) -> String {
        normalize_quotes_to_json(input)
    }

    pub fn from_ast(&mut self, arena: &PegAstArena, parse_result: &PegParseResult) {
        arena.visit_result(parse_result, &mut |node| self.map(node));
        if let Some(mut pending) = self.pending_tool_call.take() {
            if !pending.name.is_empty() {
                if !self.args_buffer.is_empty() {
                    pending.arguments = self.args_buffer.clone();
                    self.args_buffer.clear();
                }
                if self.closing_quote_pending && !pending.arguments.is_empty() {
                    pending.arguments.push('"');
                }
                self.result.tool_calls.push(pending);
            }
        }
        if !self.result.reasoning_content.is_empty()
            && self.result.reasoning_content.bytes().all(|c| c.is_ascii_whitespace())
        {
            self.result.reasoning_content.clear();
        }
    }

    pub fn map(&mut self, node: &PegAstNode) {
        let tag = node.tag.as_str();
        if tag == ChatPegBuilder::REASONING {
            self.result.reasoning_content.push_str(&node.sanitized_text());
        }
        if tag == ChatPegBuilder::CONTENT {
            self.result.content.push_str(&node.sanitized_text());
        }

        let is_tool_open = tag == ChatPegBuilder::TOOL_OPEN;
        let is_tool_close = tag == ChatPegBuilder::TOOL_CLOSE;
        let is_tool_name = tag == ChatPegBuilder::TOOL_NAME;
        let is_tool_id = tag == ChatPegBuilder::TOOL_ID;
        let is_tool_args = tag == ChatPegBuilder::TOOL_ARGS;
        let is_arg_open = tag == ChatPegBuilder::TOOL_ARG_OPEN;
        let is_arg_close = tag == ChatPegBuilder::TOOL_ARG_CLOSE;
        let is_arg_name = tag == ChatPegBuilder::TOOL_ARG_NAME;
        let is_arg_value = tag == ChatPegBuilder::TOOL_ARG_VALUE;
        let is_arg_string_value = tag == ChatPegBuilder::TOOL_ARG_STRING_VALUE;

        if is_tool_open {
            self.pending_tool_call = Some(ChatToolCall::default());
            self.current_tool = CurrentTool::Pending;
            self.arg_count = 0;
            self.args_buffer.clear();
            self.closing_quote_pending = false;
        }

        if is_tool_id && self.current_tool != CurrentTool::None {
            let mut text = trim_trailing_space(&node.sanitized_text()).to_owned();
            if text.len() >= 2 && text.starts_with('"') && text.ends_with('"') {
                text = text[1..text.len() - 1].to_owned();
            }
            match self.current_tool {
                CurrentTool::Pending => {
                    if let Some(tool) = self.pending_tool_call.as_mut() {
                        tool.id = text;
                    }
                }
                CurrentTool::Result(index) => self.result.tool_calls[index].id = text,
                CurrentTool::None => {}
            }
        }

        if is_tool_name && self.current_tool != CurrentTool::None {
            let name = trim_trailing_space(&node.text).to_owned();
            let mut tool = self.pending_tool_call.take().unwrap_or_default();
            tool.name = name;
            if !self.args_buffer.is_empty() {
                tool.arguments = self.args_buffer.clone();
                self.args_buffer.clear();
            } else if tool.arguments.is_empty() {
                tool.arguments = "{".to_owned();
            }
            self.result.tool_calls.push(tool);
            self.current_tool = CurrentTool::Result(self.result.tool_calls.len() - 1);
        }

        if is_tool_args && self.current_tool != CurrentTool::None {
            let text = trim_trailing_space(&node.text);
            if text.starts_with('{') {
                *self.args_target() = text.to_owned();
            }
        }

        if is_arg_open {
            self.closing_quote_pending = false;
        }

        if is_arg_name && self.current_tool != CurrentTool::None {
            let mut entry = String::new();
            if self.arg_count > 0 {
                entry.push(',');
            }
            entry.push_str(&json_string_value(&Json::from(Value::String(trim_space(&node.text).to_owned()))));
            entry.push(':');
            self.arg_count += 1;
            if self.args_target().is_empty() {
                *self.args_target() = "{".to_owned();
            }
            self.args_target().push_str(&entry);
        }

        if (is_arg_value || is_arg_string_value) && self.current_tool != CurrentTool::None {
            let content = node.sanitized_text();
            let mut value_to_add = String::new();
            if content.is_empty() && is_arg_string_value {
                value_to_add.push('"');
                self.closing_quote_pending = true;
            } else if !content.is_empty() && is_arg_string_value {
                if !self.closing_quote_pending {
                    value_to_add.push('"');
                    self.closing_quote_pending = true;
                }
                value_to_add.push_str(&escape_json_string_inner(&content));
            } else if !content.is_empty() {
                value_to_add.push_str(&self.normalize_container_value(&content));
            }
            self.args_target().push_str(&value_to_add);
        }

        if is_arg_close && self.current_tool != CurrentTool::None {
            if self.closing_quote_pending {
                self.args_target().push('"');
                self.closing_quote_pending = false;
            }
        }

        if is_tool_close && self.current_tool != CurrentTool::None {
            if self.current_tool == CurrentTool::Pending {
                if let Some(mut tool) = self.pending_tool_call.take() {
                    if tool.name.is_empty() && !self.args_buffer.is_empty() {
                        tool.arguments = self.args_buffer.clone();
                        self.args_buffer.clear();
                    }
                    if self.closing_quote_pending {
                        tool.arguments.push('"');
                        self.closing_quote_pending = false;
                    }
                    while json_brace_depth(&tool.arguments) > 0 {
                        tool.arguments.push('}');
                    }
                    if !tool.name.is_empty() {
                        self.result.tool_calls.push(tool);
                    }
                }
                self.current_tool = CurrentTool::None;
            } else if let CurrentTool::Result(index) = self.current_tool {
                let arguments = &mut self.result.tool_calls[index].arguments;
                if self.closing_quote_pending {
                    arguments.push('"');
                    self.closing_quote_pending = false;
                }
                while json_brace_depth(arguments) > 0 {
                    arguments.push('}');
                }
                self.current_tool = CurrentTool::None;
            }
        }
    }
}

pub struct Gemma4Mapper {
    pub result: ChatMsg,
}

impl Gemma4Mapper {
    pub const STRING: &'static str = "gemma4-string";
    pub const STRING_CONTENT: &'static str = "gemma4-string-content";
    pub const NUMBER: &'static str = "gemma4-number";
    pub const BOOL: &'static str = "gemma4-bool";
    pub const NULL_: &'static str = "gemma4-null";
    pub const ARRAY: &'static str = "gemma4-array";
    pub const DICT: &'static str = "gemma4-dict";
    pub const DICT_KV: &'static str = "gemma4-dict-kv";
    pub const DICT_KEY: &'static str = "gemma4-dict-key";
    pub const DICT_KEY_NAME: &'static str = "gemma4-dict-key-name";

    pub fn new(result: ChatMsg) -> Self {
        Self { result }
    }

    pub fn from_ast(&mut self, arena: &PegAstArena, result: &PegParseResult) {
        for &node in &result.nodes {
            self.visit(arena, node);
        }
    }

    fn collect(arena: &PegAstArena, node: &PegAstNode, tag: &str, out: &mut Vec<PegAstId>) {
        for &child_id in &node.children {
            let child = arena.get(child_id);
            if child.tag == tag {
                out.push(child_id);
            } else {
                Self::collect(arena, child, tag, out);
            }
        }
    }

    fn to_json(arena: &PegAstArena, id: PegAstId) -> String {
        let node = arena.get(id);
        if node.text.is_empty() {
            return String::new();
        }
        if node.tag == Self::NUMBER || node.tag == Self::BOOL || node.tag == Self::NULL_ {
            return node.text.clone();
        }
        if node.tag == Self::STRING_CONTENT {
            return escape_json_string_inner(&node.sanitized_text());
        }
        if node.tag == Self::STRING {
            let mut out = String::from("\"");
            for &child in &node.children {
                out.push_str(&Self::to_json(arena, child));
            }
            if !node.is_partial {
                out.push('"');
            }
            return out;
        }
        if node.tag == Self::ARRAY {
            let mut out = String::from("[");
            for (i, &child) in node.children.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Self::to_json(arena, child));
            }
            if !node.is_partial {
                out.push(']');
            }
            return out;
        }
        if node.tag == Self::DICT_KEY {
            let name_id = arena.find_by_tag(node, Self::DICT_KEY_NAME, 3);
            let mut out = String::from("\"");
            if name_id != COMMON_PEG_INVALID_AST_ID {
                out.push_str(&escape_json_string_inner(&arena.get(name_id).text));
            }
            if !node.is_partial {
                out.push_str("\":");
            }
            return out;
        }
        if node.tag == Self::DICT {
            let mut members = Vec::new();
            Self::collect(arena, node, Self::DICT_KV, &mut members);
            let mut out = String::from("{");
            for (i, &member) in members.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Self::to_json(arena, member));
            }
            if !node.is_partial {
                out.push('}');
            }
            return out;
        }
        let mut out = String::new();
        for &child in &node.children {
            out.push_str(&Self::to_json(arena, child));
        }
        out
    }

    fn visit(&mut self, arena: &PegAstArena, id: PegAstId) {
        let node = arena.get(id);
        if node.tag == ChatPegBuilder::REASONING {
            self.result.reasoning_content.push_str(&node.sanitized_text());
            return;
        }
        if node.tag == ChatPegBuilder::CONTENT {
            self.result.content.push_str(&node.sanitized_text());
            return;
        }
        if node.tag == ChatPegBuilder::TOOL {
            let name_id = arena.find_by_tag(node, ChatPegBuilder::TOOL_NAME, 3);
            let args_id = arena.find_by_tag(node, ChatPegBuilder::TOOL_ARGS, 3);
            if name_id != COMMON_PEG_INVALID_AST_ID && args_id != COMMON_PEG_INVALID_AST_ID {
                let name_node = arena.get(name_id);
                let args_node = arena.get(args_id);
                if !name_node.is_partial {
                    let mut call = ChatToolCall::default();
                    call.name = name_node.text.clone();
                    if let Some(&first) = args_node.children.first() {
                        call.arguments = Self::to_json(arena, first);
                    }
                    self.result.tool_calls.push(call);
                }
            }
            return;
        }
        for &child in &node.children {
            self.visit(arena, child);
        }
    }
}

pub struct MinimaxM3Mapper {
    pub result: ChatMsg,
}

impl MinimaxM3Mapper {
    pub const TOOL_ARG_OBJECT: &'static str = "tool-arg-object";
    pub const TOOL_ARG_ARRAY: &'static str = "tool-arg-array";
    pub const TOOL_ARG_ITEM: &'static str = "tool-arg-item";

    pub fn new(result: ChatMsg) -> Self {
        Self { result }
    }

    fn collect(arena: &PegAstArena, node: &PegAstNode, tag: &str, out: &mut Vec<PegAstId>) {
        for &child_id in &node.children {
            let child = arena.get(child_id);
            if child.tag == tag {
                out.push(child_id);
            } else {
                Self::collect(arena, child, tag, out);
            }
        }
    }

    fn value_of(arena: &PegAstArena, node: &PegAstNode) -> PegAstId {
        for &child_id in &node.children {
            let tag = arena.get(child_id).tag.as_str();
            if tag == ChatPegBuilder::TOOL_ARG_VALUE
                || tag == ChatPegBuilder::TOOL_ARG_STRING_VALUE
                || tag == Self::TOOL_ARG_OBJECT
                || tag == Self::TOOL_ARG_ARRAY
            {
                return child_id;
            }
        }
        COMMON_PEG_INVALID_AST_ID
    }

    fn value_to_json(arena: &PegAstArena, id: PegAstId, closed: bool) -> String {
        if id == COMMON_PEG_INVALID_AST_ID {
            return String::new();
        }
        let node = arena.get(id);
        if node.tag == Self::TOOL_ARG_OBJECT {
            return Self::container_to_json(arena, node, true, closed);
        }
        if node.tag == Self::TOOL_ARG_ARRAY {
            return Self::container_to_json(arena, node, false, closed);
        }
        if node.tag == ChatPegBuilder::TOOL_ARG_STRING_VALUE {
            let mut out = format!("\"{}", escape_json_string_inner(&node.sanitized_text()));
            if closed {
                out.push('"');
            }
            return out;
        }
        node.text.clone()
    }

    fn member_to_json(arena: &PegAstArena, node: &PegAstNode) -> String {
        let name_id = arena.find_by_tag(node, ChatPegBuilder::TOOL_ARG_NAME, 3);
        if name_id == COMMON_PEG_INVALID_AST_ID {
            return String::new();
        }
        format!(
            "{}:{}",
            json_string_value(&Json::from(Value::String(arena.get(name_id).text.clone()))),
            Self::value_to_json(arena, Self::value_of(arena, node), !node.is_partial)
        )
    }

    fn container_to_json(arena: &PegAstArena, node: &PegAstNode, is_object: bool, closed: bool) -> String {
        let tag = if is_object { ChatPegBuilder::TOOL_ARG } else { Self::TOOL_ARG_ITEM };
        let mut entries = Vec::new();
        Self::collect(arena, node, tag, &mut entries);
        let mut out = String::from(if is_object { "{" } else { "[" });
        let mut add_comma = false;
        for &entry_id in &entries {
            let entry = arena.get(entry_id);
            let text = if is_object {
                Self::member_to_json(arena, entry)
            } else {
                Self::value_to_json(arena, Self::value_of(arena, entry), !entry.is_partial)
            };
            if text.is_empty() {
                continue;
            }
            if add_comma {
                out.push(',');
            }
            add_comma = true;
            out.push_str(&text);
        }
        if closed {
            out.push(if is_object { '}' } else { ']' });
        }
        out
    }

    pub fn from_ast(&mut self, arena: &PegAstArena, result: &PegParseResult) {
        for &node in &result.nodes {
            self.visit(arena, node);
        }
    }

    fn visit(&mut self, arena: &PegAstArena, id: PegAstId) {
        let node = arena.get(id);
        if node.tag == ChatPegBuilder::REASONING {
            self.result.reasoning_content.push_str(&node.sanitized_text());
            return;
        }
        if node.tag == ChatPegBuilder::CONTENT {
            self.result.content.push_str(&node.sanitized_text());
            return;
        }
        if node.tag == ChatPegBuilder::TOOL {
            let name_id = arena.find_by_tag(node, ChatPegBuilder::TOOL_NAME, 3);
            if name_id != COMMON_PEG_INVALID_AST_ID {
                let mut call = ChatToolCall::default();
                call.name = arena.get(name_id).text.clone();
                call.arguments = Self::container_to_json(arena, node, true, !node.is_partial);
                self.result.tool_calls.push(call);
            }
            return;
        }
        for &child in &node.children {
            self.visit(arena, child);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TagBasedMapper {
    pub tags: BTreeMap<String, String>,
}

impl TagBasedMapper {
    pub fn from_ast(&mut self, arena: &PegAstArena, result: &PegParseResult) {
        arena.visit_result(result, &mut |node| {
            if !node.tag.is_empty() {
                self.tags.insert(node.tag.clone(), node.text.clone());
            }
        });
    }
}

#[derive(Clone, Debug, Default)]
pub struct TaggedParseResult {
    pub result: PegParseResult,
    pub tags: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default)]
pub struct TaggedPegParser {
    pub arena: PegArena,
    pub flags: PegParseFlags,
}

impl TaggedPegParser {
    pub fn with_debug(mut self) -> Self {
        self.flags |= PegParseFlags::DEBUG;
        self
    }

    pub fn without_debug(mut self) -> Self {
        self.flags &= !PegParseFlags::DEBUG;
        self
    }

    pub fn parse_and_extract(&self, input: &str, extra_flags: PegParseFlags) -> Result<TaggedParseResult> {
        let mut ctx = PegParseContext::with_input(input, self.flags | extra_flags);
        let result = self.arena.parse(&mut ctx, 0)?;
        let mut mapper = TagBasedMapper::default();
        mapper.from_ast(&ctx.ast, &result);
        Ok(TaggedParseResult { result, tags: mapper.tags })
    }

    pub fn parse_anywhere_and_extract(&self, input: &str) -> Result<TaggedParseResult> {
        if input.is_empty() {
            return self.parse_and_extract(input, PegParseFlags::NONE);
        }
        for i in 0..input.len() {
            let mut ctx = PegParseContext::with_input(input, self.flags);
            let result = self.arena.parse(&mut ctx, i)?;
            if result.success() || i == input.len() - 1 {
                let mut mapper = TagBasedMapper::default();
                mapper.from_ast(&ctx.ast, &result);
                return Ok(TaggedParseResult { result, tags: mapper.tags });
            }
        }
        Ok(TaggedParseResult::default())
    }
}

pub fn build_tagged_peg_parser<F: FnOnce(&PegBuilder) -> PegParser>(build: F) -> Result<TaggedPegParser> {
    let mut builder = PegBuilder::new();
    let root = build(&mut builder);
    builder.set_root(root);
    Ok(TaggedPegParser { arena: builder.build()?, flags: PegParseFlags::NONE })
}

// ---------------------------------------------------------------------------
// chat-auto-parser-helpers.cpp: string and diff helpers
// ---------------------------------------------------------------------------

pub fn trim_whitespace(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_ascii_whitespace())
}
pub fn trim_leading_whitespace(s: &str) -> &str {
    s.trim_start_matches(|c: char| c.is_ascii_whitespace())
}
pub fn trim_trailing_whitespace(s: &str) -> &str {
    s.trim_end_matches(|c: char| c.is_ascii_whitespace())
}
pub fn trim_trailing_newlines(s: &str) -> &str {
    s.trim_end_matches('\n')
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentType {
    Text,
    Marker,
}

impl SegmentType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "TEXT",
            Self::Marker => "MARKER",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub segment_type: SegmentType,
    pub value: String,
}

impl Segment {
    pub fn new(segment_type: SegmentType, value: impl Into<String>) -> Self {
        Self { segment_type, value: value.into() }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiffSplit {
    pub prefix: String,
    pub suffix: String,
    pub left: String,
    pub right: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompareVariantsResult {
    pub diff: DiffSplit,
    pub output_a: String,
    pub output_b: String,
}

fn common_prefix_len(left: &str, right: &str) -> usize {
    left.as_bytes()
        .iter()
        .zip(right.as_bytes())
        .take_while(|(a, b)| a == b)
        .count()
}

fn common_suffix_len(left: &str, right: &str) -> usize {
    left.as_bytes()
        .iter()
        .rev()
        .zip(right.as_bytes().iter().rev())
        .take_while(|(a, b)| a == b)
        .count()
}

/// Longest common prefix, common suffix and the mismatched middle parts.
pub fn calculate_diff_split(left: &str, right: &str) -> DiffSplit {
    let mut result = DiffSplit::default();
    let left_seg = segmentize_markers(left);
    let right_seg = segmentize_markers(right);
    if left_seg.is_empty() {
        result.right = right.to_owned();
        return result;
    }
    if right_seg.is_empty() {
        result.left = left.to_owned();
        return result;
    }

    let mut ls = 0usize;
    let mut le = left_seg.len() - 1;
    let mut rs = 0usize;
    let mut re = right_seg.len() - 1;
    let mut left_fully_consumed = false;
    let mut right_fully_consumed = false;

    while ls != le && rs != re {
        let mut advanced = false;
        if left_seg[ls] == right_seg[rs] {
            result.prefix.push_str(&left_seg[ls].value);
            ls += 1;
            rs += 1;
            advanced = true;
        }
        if left_seg[le] == right_seg[re] {
            result.suffix = format!("{}{}", left_seg[le].value, result.suffix);
            if ls != le {
                le -= 1;
            } else {
                left_fully_consumed = true;
            }
            if rs != re {
                re -= 1;
            } else {
                right_fully_consumed = true;
            }
            advanced = true;
        }
        if !advanced {
            break;
        }
    }

    if ls == le && rs != re {
        if left_seg[ls] == right_seg[re] {
            result.suffix = format!("{}{}", right_seg[re].value, result.suffix);
            re = re.saturating_sub(1);
            left_fully_consumed = true;
        } else if left_seg[ls] == right_seg[rs] {
            result.prefix.push_str(&right_seg[rs].value);
            rs += 1;
            left_fully_consumed = true;
        }
    } else if rs == re && ls != le {
        if left_seg[le] == right_seg[rs] {
            result.suffix = format!("{}{}", left_seg[le].value, result.suffix);
            le = le.saturating_sub(1);
            right_fully_consumed = true;
        } else if left_seg[ls] == right_seg[rs] {
            result.prefix.push_str(&left_seg[ls].value);
            ls += 1;
            right_fully_consumed = true;
        }
    } else if ls == le && rs == re
        && left_seg[ls] == right_seg[rs]
        && left_seg[ls].segment_type == SegmentType::Marker
    {
        result.prefix.push_str(&right_seg[rs].value);
        left_fully_consumed = true;
        right_fully_consumed = true;
    }

    let can_have_text_suffix = left_seg[le].segment_type == SegmentType::Text
        && right_seg[re].segment_type == SegmentType::Text;
    let can_have_text_prefix = right_seg[rs].segment_type == SegmentType::Text
        && left_seg[ls].segment_type == SegmentType::Text;
    let left_end = if left_fully_consumed { le } else { le + 1 };
    let right_end = if right_fully_consumed { re } else { re + 1 };
    let left_start = ls.min(left_end.min(left_seg.len()));
    let right_start = rs.min(right_end.min(right_seg.len()));
    let remainder_left: String = left_seg[left_start..left_end.min(left_seg.len())]
        .iter()
        .map(|s| s.value.as_str())
        .collect();
    let remainder_right: String = right_seg[right_start..right_end.min(right_seg.len())]
        .iter()
        .map(|s| s.value.as_str())
        .collect();

    let suffix_len = if can_have_text_suffix {
        common_suffix_len(&remainder_left, &remainder_right)
    } else {
        0
    };
    let prefix_len = if can_have_text_prefix {
        common_prefix_len(
            &remainder_left[..remainder_left.len() - suffix_len],
            &remainder_right[..remainder_right.len() - suffix_len],
        )
    } else {
        0
    };

    result.prefix.push_str(&remainder_left[..prefix_len]);
    result.suffix = format!(
        "{}{}",
        &remainder_left[remainder_left.len() - suffix_len..],
        result.suffix
    );
    result.left = remainder_left[prefix_len..remainder_left.len() - suffix_len].to_owned();
    result.right = remainder_right[prefix_len..remainder_right.len() - suffix_len].to_owned();

    if result.left.is_empty() && result.right.is_empty() {
        result.prefix = left.to_owned();
        result.suffix.clear();
    }
    if result.left.is_empty()
        && !result.right.is_empty()
        && left.len() <= right.len()
        && right.starts_with(left)
    {
        result.prefix = left.to_owned();
        result.suffix.clear();
        result.right = right[left.len()..].to_owned();
    }
    result
}

pub fn until_common_prefix(full: &str, left: &str, right: &str) -> String {
    let len = common_prefix_len(left, right);
    if len == 0 {
        return String::new();
    }
    match full.find(&left[..len]) {
        Some(pos) => full[..pos].to_owned(),
        None => String::new(),
    }
}

pub fn after_common_suffix(full: &str, left: &str, right: &str) -> String {
    let len = common_suffix_len(left, right);
    if len == 0 {
        return String::new();
    }
    let suffix = &left[left.len() - len..];
    match full.rfind(suffix) {
        Some(pos) => full[pos + len..].to_owned(),
        None => String::new(),
    }
}

pub fn segmentize_markers(text: &str) -> Vec<Segment> {
    let bytes = text.as_bytes();
    let mut result = Vec::new();
    let mut in_marker = false;
    let mut opener = b'\0';
    let mut last_border = 0usize;
    for (pos, &c) in bytes.iter().enumerate() {
        if !in_marker && (c == b'<' || c == b'[') {
            if last_border < pos {
                result.push(Segment::new(SegmentType::Text, &text[last_border..pos]));
            }
            last_border = pos;
            in_marker = true;
            opener = c;
        } else if in_marker && ((opener == b'<' && c == b'>') || (opener == b'[' && c == b']')) {
            result.push(Segment::new(SegmentType::Marker, &text[last_border..pos + 1]));
            last_border = pos + 1;
            in_marker = false;
            opener = b'\0';
        }
    }
    if last_border < text.len() {
        result.push(Segment::new(SegmentType::Text, &text[last_border..]));
    }
    result
}

pub fn prune_whitespace_segments(segments: &[Segment]) -> Vec<Segment> {
    segments
        .iter()
        .filter(|seg| !trim_whitespace(&seg.value).is_empty())
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// chat-auto-parser.h: analysis data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemplateParams {
    pub messages: Json,
    pub tools: Json,
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub extra_context: Option<Json>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningMode {
    None,
    TagBased,
    ToolsOnly,
}

impl ReasoningMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::TagBased => "TAG_BASED",
            Self::ToolsOnly => "TOOLS_ONLY",
        }
    }
}

impl Default for ReasoningMode {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentMode {
    Plain,
    AlwaysWrapped,
    WrappedWithReasoning,
}

impl ContentMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Plain => "PLAIN",
            Self::AlwaysWrapped => "ALWAYS_WRAPPED",
            Self::WrappedWithReasoning => "WRAPPED_WITH_REASONING",
        }
    }
}

impl Default for ContentMode {
    fn default() -> Self {
        Self::Plain
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallIdPosition {
    None,
    PreFuncName,
    BetweenFuncAndArgs,
    PostArgs,
}

impl CallIdPosition {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::PreFuncName => "PRE_FUNC_NAME",
            Self::BetweenFuncAndArgs => "BETWEEN_FUNC_AND_ARGS",
            Self::PostArgs => "POST_ARGS",
        }
    }
}

impl Default for CallIdPosition {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolFormat {
    None,
    JsonNative,
    TagWithJson,
    TagWithTagged,
}

impl ToolFormat {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::JsonNative => "JSON_NATIVE",
            Self::TagWithJson => "TAG_WITH_JSON",
            Self::TagWithTagged => "TAG_WITH_TAGGED",
        }
    }
}

impl Default for ToolFormat {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Clone, Debug, Default)]
pub struct ToolFormatAnalysis {
    pub mode: ToolFormat,
    pub section_start: String,
    pub section_end: String,
    pub per_call_start: String,
    pub per_call_end: String,
    pub fun_name_is_key: bool,
    pub tools_array_wrapped: bool,
    pub openai_wrapper_trigger: bool,
    pub function_field: String,
    pub name_field: String,
    pub args_field: String,
    pub id_field: String,
    pub gen_id_field: String,
    pub parameter_order: Vec<String>,
}

impl ToolFormatAnalysis {
    pub fn new() -> Self {
        Self {
            function_field: "function".to_owned(),
            name_field: "name".to_owned(),
            args_field: "arguments".to_owned(),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ToolFunctionAnalysis {
    pub name_prefix: String,
    pub name_suffix: String,
    pub args_separator: String,
    pub close: String,
}

#[derive(Clone, Debug, Default)]
pub struct ToolArgumentsAnalysis {
    pub start: String,
    pub end: String,
    pub name_prefix: String,
    pub name_suffix: String,
    pub value_prefix: String,
    pub value_suffix: String,
    pub separator: String,
    pub tolerate_intertag_whitespace: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ToolIdAnalysis {
    pub pos: CallIdPosition,
    pub prefix: String,
    pub suffix: String,
}

#[derive(Clone, Debug, Default)]
pub struct AnalyzeReasoning {
    pub mode: ReasoningMode,
    pub start: String,
    pub end: String,
}

#[derive(Clone, Debug, Default)]
pub struct AnalyzeContent {
    pub mode: ContentMode,
    pub start: String,
    pub end: String,
    pub requires_nonnull_content: bool,
}

impl AnalyzeContent {
    pub fn is_always_wrapped(&self) -> bool {
        self.mode == ContentMode::AlwaysWrapped || self.mode == ContentMode::WrappedWithReasoning
    }
}

#[derive(Clone, Debug, Default)]
pub struct AnalyzeTools {
    pub format: ToolFormatAnalysis,
    pub function: ToolFunctionAnalysis,
    pub arguments: ToolArgumentsAnalysis,
    pub call_id: ToolIdAnalysis,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JinjaCapability {
    #[default]
    Unknown,
    Supported,
    Unsupported,
}

#[derive(Clone, Debug, Default)]
pub struct JinjaCaps {
    pub supports_tools: bool,
    pub supports_tool_calls: bool,
    pub supports_parallel_tool_calls: bool,
    pub supports_system_role: bool,
    pub supports_tool_result_role: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AutoParser {
    pub jinja_caps: JinjaCaps,
    pub user_start: String,
    pub assistant_start: String,
    pub reasoning: AnalyzeReasoning,
    pub content: AnalyzeContent,
    pub tools: AnalyzeTools,
    pub analysis_complete: bool,
    pub preserved_tokens: Vec<String>,
    pub additional_stops: Vec<String>,
}

impl AutoParser {
    /// Union of every non-empty marker found by analysis.
    pub fn collect_preserved_tokens(&mut self) {
        let mut add = |token: &str| {
            let token = trim_whitespace(token);
            if !token.is_empty() && !self.preserved_tokens.iter().any(|t| t == token) {
                self.preserved_tokens.push(token.to_owned());
            }
        };
        add(&self.reasoning.start);
        add(&self.reasoning.end);
        add(&self.content.start);
        add(&self.content.end);
        add(&self.tools.format.section_start);
        add(&self.tools.format.section_end);
        add(&self.tools.format.per_call_start);
        add(&self.tools.format.per_call_end);
        add(&self.tools.function.name_prefix);
        add(&self.tools.function.name_suffix);
        add(&self.tools.function.args_separator);
        add(&self.tools.function.close);
        add(&self.tools.arguments.start);
        add(&self.tools.arguments.end);
        add(&self.tools.arguments.name_prefix);
        add(&self.tools.arguments.name_suffix);
        add(&self.tools.arguments.separator);
        add(&self.tools.arguments.value_prefix);
        add(&self.tools.arguments.value_suffix);
        add(&self.tools.call_id.prefix);
        add(&self.tools.call_id.suffix);
    }
}

/// Template-source patches applied after differential analysis.
pub fn apply_workarounds(src: &str, analysis: &mut AutoParser) {
    if src.contains("content.split('</think>')")
        && !src.contains("reasoning_content")
        && !src.contains("<SPECIAL_12>")
        && analysis.reasoning.mode == ReasoningMode::None
    {
        analysis.reasoning.mode = ReasoningMode::TagBased;
        analysis.reasoning.start = "<think>".into();
        analysis.reasoning.end = "</think>".into();
        analysis.preserved_tokens.push("<think>".into());
        analysis.preserved_tokens.push("</think>".into());
    }
    if src.contains("Write your thoughts between <think></think> and write your response between <response></response>") {
        analysis.reasoning.mode = ReasoningMode::TagBased;
        analysis.reasoning.start = "<think>".into();
        analysis.reasoning.end = "</think>".into();
        analysis.preserved_tokens.push("<think>".into());
        analysis.preserved_tokens.push("</think>".into());
        analysis.content.mode = ContentMode::WrappedWithReasoning;
        analysis.content.start = "<response>".into();
        analysis.content.end = "</response>".into();
        analysis.preserved_tokens.push("<response>".into());
        analysis.preserved_tokens.push("</response>".into());
    }
    if src.contains("<|CHATBOT_TOKEN|>")
        && src.contains("<|END_OF_TURN_TOKEN|>")
        && analysis.content.start.is_empty()
    {
        analysis.content.mode = ContentMode::AlwaysWrapped;
        analysis.content.start = "<|CHATBOT_TOKEN|>".into();
        analysis.content.end = "<|END_OF_TURN_TOKEN|>".into();
        analysis.preserved_tokens.push("<|CHATBOT_TOKEN|>".into());
        analysis.preserved_tokens.push("<|END_OF_TURN_TOKEN|>".into());
        analysis.user_start = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>".into();
    }
    if src.contains("set has_code_interpreter = tools | selectattr(\"type\", \"equalto\", \"code_interpreter\") | list | length > 0") {
        analysis.content.mode = ContentMode::Plain;
        analysis.content.end.clear();
        analysis.tools.function.name_prefix.clear();
        analysis.tools.format.section_start.clear();
        analysis.tools.format.section_end.clear();
        analysis.tools.format.per_call_start = "<function=".into();
        analysis.tools.format.per_call_end = "</function>".into();
        analysis.tools.function.close.clear();
        analysis.preserved_tokens.clear();
        for token in ["<|eot_id|>", "<|eom_id|>", "<function=", ">", "</function>"] {
            analysis.preserved_tokens.push(token.into());
        }
    }
    if src.contains("{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>'") {
        analysis.tools.format.section_start = "<｜tool▁calls▁begin｜>".into();
        analysis.tools.format.section_end = "<｜tool▁calls▁end｜>".into();
        analysis.tools.format.per_call_start = "<｜tool▁call▁begin｜>function".into();
        analysis.tools.function.name_prefix = "<｜tool▁sep｜>".into();
        analysis.tools.format.per_call_end = "<｜tool▁call▁end｜>".into();
        analysis.tools.function.close = "```".into();
    }
    if src.contains("<SPECIAL_10>")
        && src.contains("<SPECIAL_11>")
        && src.contains("<SPECIAL_12>")
        && src.contains("<TOOL_RESPONSE>")
    {
        analysis.tools.format.mode = ToolFormat::JsonNative;
        analysis.tools.format.section_start.clear();
        analysis.tools.format.section_end.clear();
        analysis.tools.format.per_call_start = "<TOOLCALL>".into();
        analysis.tools.format.per_call_end = "</TOOLCALL>".into();
        analysis.tools.format.tools_array_wrapped = true;
        analysis.content.mode = ContentMode::Plain;
        analysis.content.start.clear();
        analysis.content.end.clear();
        analysis.reasoning.mode = ReasoningMode::TagBased;
        analysis.reasoning.start = "<think>\n".into();
        analysis.reasoning.end = "</think>".into();
        analysis.assistant_start = "<SPECIAL_11>Assistant".into();
        analysis.user_start = "<SPECIAL_11>User".into();
        analysis.preserved_tokens.clear();
        for token in ["<SPECIAL_11>", "</think>", "<TOOLCALL>", "</TOOLCALL>"] {
            analysis.preserved_tokens.push(token.into());
        }
    }
    if src.contains("{%- set system_prompt = '<|start_header_id|>' + 'system' + '<|end_header_id|>\\n\\n' + message['content'] | trim + '\\n' + system_prompt_suffix + '<|eot_id|>' -%}") {
        analysis.assistant_start = "<|start_header_id|>assistant<|end_header_id|>".into();
        analysis.user_start = "<|start_header_id|>user<|end_header_id|>".into();
    }
    if src.contains("<|begin|>assistant<|think|><|end|>") {
        analysis.assistant_start = "<|begin|>assistant".into();
    }
    if src.contains("if not loop.last and '[BEGIN FINAL RESPONSE]' in asst_text") {
        analysis.user_start = "<|begin_user|>".into();
        analysis.assistant_start = "<|begin_assistant|>".into();
    }
    if src.contains("Respond in the format {\"name\": function name") && src.contains("Do not use variables.") {
        analysis.tools.format.openai_wrapper_trigger = true;
    }
    if src.contains("laguna_glm_thinking") {
        analysis.reasoning.start = trim_whitespace(&analysis.reasoning.start).to_owned();
        analysis.reasoning.end = trim_whitespace(&analysis.reasoning.end).to_owned();
        analysis.tools.arguments.value_prefix = trim_whitespace(&analysis.tools.arguments.value_prefix).to_owned();
        analysis.tools.arguments.value_suffix = trim_whitespace(&analysis.tools.arguments.value_suffix).to_owned();
        analysis.tools.arguments.separator = trim_whitespace(&analysis.tools.arguments.separator).to_owned();
        analysis.tools.arguments.tolerate_intertag_whitespace = true;
        analysis.additional_stops.push("</assistant>".into());
    }
    if src.contains("Bailing V3 chat template") {
        analysis.tools.arguments.value_suffix = trim_whitespace(&analysis.tools.arguments.value_suffix).to_owned();
        analysis.tools.arguments.tolerate_intertag_whitespace = true;
    }
}

// ---------------------------------------------------------------------------
// chat-auto-parser-generator.cpp: parser building from analysis
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GenerationParams {
    pub messages: Json,
    pub tools: Json,
    pub tool_choice: ChatToolChoice,
    pub json_schema: Json,
    pub parallel_tool_calls: bool,
    pub reasoning_format: crate::chat::ReasoningFormat,
    pub stream: bool,
    pub grammar: String,
    pub add_generation_prompt: bool,
    pub continue_final_message: crate::chat::ChatContinuation,
    pub continue_msg: ChatMsg,
    pub enable_thinking: bool,
    pub extra_context: Json,
    pub add_bos: bool,
    pub add_eos: bool,
    pub is_inference: bool,
    pub add_inference: bool,
    pub mark_input: bool,
    pub specials: PegTokens,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            messages: Json::array(),
            tools: Json::array(),
            tool_choice: ChatToolChoice::default(),
            json_schema: Json::object(),
            parallel_tool_calls: true,
            reasoning_format: crate::chat::ReasoningFormat::Auto,
            stream: true,
            grammar: String::new(),
            add_generation_prompt: false,
            continue_final_message: crate::chat::ChatContinuation::None,
            continue_msg: ChatMsg::default(),
            enable_thinking: true,
            extra_context: Json::object(),
            add_bos: false,
            add_eos: false,
            is_inference: true,
            add_inference: false,
            mark_input: true,
            specials: PegTokens::new(),
        }
    }
}

impl GenerationParams {
    pub fn has_continuation(&self) -> bool {
        self.continue_final_message != crate::chat::ChatContinuation::None
            && !self.continue_msg.empty()
    }
}

pub struct ParserBuildContext<'a> {
    pub p: &'a ChatPegBuilder,
    pub inputs: &'a GenerationParams,
    pub reasoning_parser: PegParser,
    pub extracting_reasoning: bool,
}

impl AnalyzeReasoning {
    pub fn build_parser(&self, ctx: &mut ParserBuildContext<'_>) -> PegParser {
        if !ctx.extracting_reasoning {
            return ctx.p.p.eps();
        }
        if self.mode == ReasoningMode::TagBased || self.mode == ReasoningMode::ToolsOnly {
            if !self.end.is_empty() {
                if !self.start.is_empty() {
                    let start = ctx.p.optspace(&self.start);
                    let end = trim_whitespace(&self.end).to_owned();
                    let inside = ctx.p.reasoning(ctx.p.p.until_str(&end));
                    let end_tag = ctx.p.optspace(&self.end);
                    let body = ctx.p.p.sequence(&[start, inside, end_tag]);
                    return ctx.p.p.optional(body);
                }
                let end = trim_whitespace(&self.end).to_owned();
                let inside = ctx.p.reasoning(ctx.p.p.until_str(&end));
                let end_tag = ctx.p.optspace(&self.end);
                return ctx.p.p.optional(ctx.p.p.sequence(&[inside, end_tag]));
            }
        }
        ctx.p.p.eps()
    }
}

impl AnalyzeContent {
    pub fn build_parser(
        &self,
        ctx: &mut ParserBuildContext<'_>,
        reasoning_parser: PegParser,
    ) -> PegParser {
        if self.is_always_wrapped() {
            if ctx.extracting_reasoning {
                let start = ctx.p.p.literal(&self.start);
                let until_end = ctx.p.p.until_str(&self.end);
                let content = ctx.p.content(until_end);
                let end = ctx.p.p.literal(&self.end);
                let finish = ctx.p.p.end();
                return ctx.p.p.sequence(&[reasoning_parser, start, content, end, finish]);
            }
            let until_start = ctx.p.p.until_str(&self.start);
            let start = ctx.p.p.literal(&self.start);
            let until_end = ctx.p.p.until_str(&self.end);
            let content = ctx.p.content(until_end);
            let end = ctx.p.p.literal(&self.end);
            let finish = ctx.p.p.end();
            return ctx.p.p.sequence(&[until_start, start, content, end, finish]);
        }
        let rest = ctx.p.p.rest();
        let content = ctx.p.content(rest);
        let finish = ctx.p.p.end();
        ctx.p.p.sequence(&[reasoning_parser, content, finish])
    }

    pub fn build_optional_wrapped(&self, ctx: &mut ParserBuildContext<'_>) -> PegParser {
        if self.is_always_wrapped() {
            let start = ctx.p.p.literal(&self.start);
            let until_end = ctx.p.p.until_str(&self.end);
            let content = ctx.p.content(until_end);
            let end = ctx.p.p.literal(&self.end);
            return ctx.p.p.optional(ctx.p.p.sequence(&[start, content, end]));
        }
        ctx.p.p.eps()
    }
}

impl AnalyzeTools {
    pub fn build_parser(
        &self,
        ctx: &mut ParserBuildContext<'_>,
        reasoning_parser: PegParser,
        content: &AnalyzeContent,
    ) -> PegParser {
        match self.format.mode {
            ToolFormat::JsonNative => self.build_tool_parser_json_native(ctx, reasoning_parser, content),
            ToolFormat::TagWithJson => self.build_tool_parser_tag_json(ctx, reasoning_parser),
            ToolFormat::TagWithTagged => self.build_tool_parser_tag_tagged(ctx, reasoning_parser),
            ToolFormat::None => {
                eprintln!(
                    "[ERROR] Template seems to support tool calls, but failed to determine tool format. Tool calling will not work properly."
                );
                ctx.p.p.eps()
            }
        }
    }

    fn build_tool_parser_json_native(
        &self,
        ctx: &mut ParserBuildContext<'_>,
        reasoning_parser: PegParser,
        content: &AnalyzeContent,
    ) -> PegParser {
        let p = ctx.p;
        let inputs = ctx.inputs;
        let mut name_field = self.format.name_field.clone();
        let mut args_field = self.format.args_field.clone();
        if !self.format.function_field.is_empty()
            && self.format.function_field != "function"
            && !name_field.contains('.')
        {
            name_field = format!("{}.{}", self.format.function_field, name_field);
            args_field = format!("{}.{}", self.format.function_field, args_field);
        }
        let tools_parser = if self.format.section_start.is_empty() && !self.format.per_call_start.is_empty() {
            let single = p.standard_json_tools(
                &self.format.per_call_start,
                &self.format.per_call_end,
                &inputs.tools,
                inputs.parallel_tool_calls,
                inputs.tool_choice.is_required(),
                &name_field,
                &args_field,
                self.format.tools_array_wrapped,
                self.format.fun_name_is_key,
                &self.format.id_field,
                &self.format.gen_id_field,
                &self.format.parameter_order,
                self.format.openai_wrapper_trigger,
                &inputs.tool_choice,
            );
            let ws = p.p.space();
            let many = p.p.one_or_more(p.p.sequence(&[single, ws]));
            p.trigger_rule("tool-calls", many)
        } else {
            p.standard_json_tools(
                &self.format.section_start,
                &self.format.section_end,
                &inputs.tools,
                inputs.parallel_tool_calls,
                inputs.tool_choice.is_required(),
                &name_field,
                &args_field,
                self.format.tools_array_wrapped,
                self.format.fun_name_is_key,
                &self.format.id_field,
                &self.format.gen_id_field,
                &self.format.parameter_order,
                self.format.openai_wrapper_trigger,
                &inputs.tool_choice,
            )
        };
        if content.is_always_wrapped() {
            let wrapped = content.build_optional_wrapped(ctx);
            let finish = p.p.end();
            return p.p.sequence(&[reasoning_parser, wrapped, tools_parser, finish]);
        }
        let tool_start = if !self.format.section_start.is_empty() {
            self.format.section_start.clone()
        } else if !self.format.per_call_start.is_empty() {
            self.format.per_call_start.clone()
        } else {
            "{".to_owned()
        };
        let until = p.p.until_str(&tool_start);
        let before = p.p.optional(p.content(until));
        let finish = p.p.end();
        p.p.sequence(&[reasoning_parser, before, tools_parser, finish])
    }

    fn build_func_parser(
        &self,
        p: &ChatPegBuilder,
        name: &str,
        call_id_section: PegParser,
        have_call_id: bool,
        args: PegParser,
        atomic_peek: Option<PegParser>,
    ) -> PegParser {
        let mut open = {
            let prefix = p.p.literal(&self.function.name_prefix);
            let name_tag = p.tool_name(p.p.literal(name));
            let suffix = p.p.literal(&self.function.name_suffix);
            p.tool_open(p.p.sequence(&[prefix, name_tag, suffix]))
        };
        let mut matched_atomic = false;
        if !self.function.args_separator.is_empty() {
            let separator = p.p.literal(&self.function.args_separator);
            open = p.p.sequence(&[open, p.p.space(), separator]);
        }
        let mut func_parser = if !self.function.name_suffix.is_empty() {
            matched_atomic = true;
            p.p.sequence(&[open, call_id_section, p.p.space(), args])
        } else if have_call_id {
            matched_atomic = true;
            let prefix = p.p.atomic(p.p.sequence(&[open, call_id_section]));
            p.p.sequence(&[prefix, p.p.space(), args])
        } else if let Some(peek) = atomic_peek {
            matched_atomic = true;
            let prefix = p.p.atomic(p.p.sequence(&[open, call_id_section, p.p.space(), peek]));
            p.p.sequence(&[prefix, args])
        } else {
            p.p.sequence(&[open, call_id_section, p.p.space(), args])
        };
        if !self.function.close.is_empty() {
            let close = p.tool_close(p.p.literal(&self.function.close));
            func_parser = p.p.sequence(&[func_parser, p.p.space(), close]);
        } else if !self.format.per_call_end.is_empty() {
            let close_literal = p.p.literal(&self.format.per_call_end);
            let peek = if self.arguments.tolerate_intertag_whitespace {
                p.p.peek(p.p.sequence(&[p.p.space(), close_literal]))
            } else {
                p.p.peek(close_literal)
            };
            let close = p.tool_close(peek);
            func_parser = p.p.sequence(&[func_parser, close]);
        } else {
            let close = p.tool_close(p.p.space());
            func_parser = p.p.sequence(&[func_parser, close]);
        }
        if !matched_atomic {
            func_parser = p.p.atomic(func_parser);
        }
        func_parser
    }

    fn build_tool_parser_tag_json(
        &self,
        ctx: &mut ParserBuildContext<'_>,
        reasoning_parser: PegParser,
    ) -> PegParser {
        let p = ctx.p;
        let inputs = ctx.inputs;
        let mut tool_choice_parser = p.p.choice(&[]);
        let mut tools_vec = Vec::new();
        foreach_function(&inputs.tools, &inputs.tool_choice, |i, function| {
            tools_vec.push((i, function.clone()));
        });
        for (index, function) in tools_vec {
            let name = function.at("name").map(json_string_value).unwrap_or_default();
            let schema = tool_parameters(&function);
            let mut have_call_id = false;
            let mut call_id_section = p.p.eps();
            if self.call_id.pos == CallIdPosition::BetweenFuncAndArgs
                && !self.call_id.prefix.is_empty()
                && (!self.call_id.suffix.is_empty() || !self.arguments.start.is_empty())
            {
                if !self.call_id.suffix.is_empty() {
                    let prefix = p.p.literal(&self.call_id.prefix);
                    let until = p.p.until_str(&self.call_id.suffix);
                    let id = p.tool_id(until);
                    let body = p.p.optional(p.p.sequence(&[prefix, id]));
                    let suffix = p.p.literal(&self.call_id.suffix);
                    call_id_section = p.p.sequence(&[body, suffix]);
                } else {
                    let prefix = p.p.literal(&self.call_id.prefix);
                    let until = p.p.until_str(&self.arguments.start);
                    let id = p.tool_id(until);
                    call_id_section = p.p.optional(p.p.sequence(&[prefix, id]));
                }
                have_call_id = true;
            }
            let json_parser = p.p.json();
            let mut args_parser = p.tool_args(p.schema(json_parser, &format!("tool-{}-schema", index), &schema, false));
            if !self.arguments.start.is_empty() {
                let start = p.p.literal(&self.arguments.start);
                args_parser = p.p.sequence(&[start, args_parser]);
            }
            if !self.arguments.end.is_empty() {
                let end = p.p.literal(&self.arguments.end);
                args_parser = p.p.sequence(&[args_parser, end]);
            }
            let atomic_peek = if !self.arguments.start.is_empty() {
                Some(p.p.peek(p.p.literal(&self.arguments.start)))
            } else {
                None
            };
            let func = self.build_func_parser(p, &name, call_id_section, have_call_id, args_parser, atomic_peek);
            let rule = p.rule(&format!("tool-{}", index), func, false);
            tool_choice_parser = p.p.choice_ids(&[tool_choice_parser.id, rule.id]);
        }
        let require_calls = inputs.tool_choice.is_required();
        let mut tool_calls = if !self.format.per_call_start.is_empty() {
            let start = p.p.literal(&self.format.per_call_start);
            let end = p.p.literal(&self.format.per_call_end);
            let wrapped = p.p.sequence(&[start, tool_choice_parser, end]);
            let calls = if inputs.parallel_tool_calls {
                let ws = p.p.space();
                let more = p.p.zero_or_more(p.p.sequence(&[ws, wrapped]));
                p.trigger_rule("tool-call", p.p.sequence(&[wrapped, more]))
            } else {
                p.trigger_rule("tool-call", wrapped)
            };
            if !self.format.section_start.is_empty() {
                let section_start = p.p.literal(&self.format.section_start);
                let ws1 = p.p.space();
                let ws2 = p.p.space();
                let section_end = if self.format.section_end.is_empty() {
                    p.p.end()
                } else {
                    p.p.literal(&self.format.section_end)
                };
                let section = p.p.sequence(&[section_start, ws1, calls, ws2, section_end]);
                p.trigger_rule("tool-calls", section)
            } else {
                calls
            }
        } else {
            let separator = p.p.literal(", ");
            let start = p.p.literal(&self.format.section_start);
            let end = p.p.literal(&self.format.section_end);
            if inputs.parallel_tool_calls {
                let more = p.p.zero_or_more(p.p.sequence(&[separator, tool_choice_parser]));
                let body = p.p.sequence(&[start, tool_choice_parser, more, end]);
                p.trigger_rule("tool-call", body)
            } else {
                let body = p.p.sequence(&[start, tool_choice_parser, end]);
                p.trigger_rule("tool-call", body)
            }
        };
        if !require_calls {
            tool_calls = p.p.optional(tool_calls);
        }
        let trigger_marker = if self.format.section_start.is_empty() {
            self.format.per_call_start.clone()
        } else {
            self.format.section_start.clone()
        };
        let before = if trigger_marker.is_empty() {
            p.p.eps()
        } else {
            p.p.until_str(&trigger_marker)
        };
        let before = p.content(before);
        let finish = p.p.end();
        p.p.sequence(&[reasoning_parser, p.p.optional(before), tool_calls, finish])
    }

    fn build_tool_parser_tag_tagged(
        &self,
        ctx: &mut ParserBuildContext<'_>,
        reasoning_parser: PegParser,
    ) -> PegParser {
        let p = ctx.p;
        let inputs = ctx.inputs;
        let raw_strings = !self.arguments.value_suffix.is_empty();
        let until_suffix = if raw_strings {
            let until = p.p.until_str(&self.arguments.value_suffix);
            p.rule("until-suffix", until, false)
        } else {
            p.p.eps()
        };
        let mut tool_choice_parser = p.p.choice(&[]);
        let mut tools_vec = Vec::new();
        foreach_function(&inputs.tools, &inputs.tool_choice, |i, function| {
            tools_vec.push((i, function.clone()));
        });
        for (index, function) in tools_vec {
            let name = function.at("name").map(json_string_value).unwrap_or_default();
            let mut required = Vec::new();
            let mut optional = Vec::new();
            foreach_parameter(&function, |param_index, param| {
                let name_prefix = p.p.literal(&self.arguments.name_prefix);
                let name_tag = p.tool_arg_name(p.p.literal(&param.name));
                let name_suffix = p.p.literal(&self.arguments.name_suffix);
                let open_tag = p.tool_arg_open(p.p.sequence(&[name_prefix, name_tag, name_suffix]));
                let value_prefix = p.p.literal(&self.arguments.value_prefix);
                let value_suffix = p.p.literal(&self.arguments.value_suffix);
                let value = if raw_strings && param.is_string {
                    let string_value = p.tool_arg_string_value(until_suffix);
                    let close = p.tool_arg_close(value_suffix);
                    p.through(string_value, &[PegUntilBranch::new(close.id)], false)
                } else {
                    let json_parser = p.p.json();
                    let schema_name = format!("tool-{}-arg-{}-schema", index, param_index);
                    let schema_parser = p.schema(json_parser, &schema_name, &tool_parameters(&function), false);
                    let value = p.tool_arg_json_value(schema_parser);
                    p.p.sequence(&[value, p.tool_arg_close(value_suffix)])
                };
                let arg = p.tool_arg(p.p.sequence(&[open_tag, value_prefix, value]));
                let named = p.rule(&format!("tool-{}-arg-{}", index, param_index), arg, false);
                if param.required {
                    required.push(named);
                } else {
                    optional.push(named);
                }
            });
            let mut args_seq = p.p.eps();
            for (i, parser) in required.iter().enumerate() {
                if i > 0 {
                    let space = p.p.space();
                    args_seq = p.p.sequence(&[args_seq, space]);
                }
                args_seq = p.p.sequence(&[args_seq, *parser]);
            }
            if !optional.is_empty() {
                let mut any_opt = p.p.choice(&[]);
                for parser in &optional {
                    any_opt = p.p.choice_ids(&[any_opt.id, parser.id]);
                }
                let space = p.p.space();
                let repeated = p.p.repeat(p.p.sequence(&[space, any_opt]), 0, -1);
                args_seq = p.p.sequence(&[args_seq, repeated]);
            }
            if !self.arguments.start.is_empty() {
                let start = p.p.literal(&self.arguments.start);
                args_seq = p.p.sequence(&[start, args_seq]);
            }
            if !self.arguments.end.is_empty() {
                let end = p.p.literal(&self.arguments.end);
                args_seq = p.p.sequence(&[args_seq, end]);
            }
            let mut call_id_section = p.p.eps();
            let mut have_call_id = false;
            if self.call_id.pos == CallIdPosition::BetweenFuncAndArgs
                && !self.call_id.prefix.is_empty()
                && (!self.call_id.suffix.is_empty() || !self.arguments.start.is_empty())
            {
                have_call_id = true;
                if !self.call_id.suffix.is_empty() {
                    let prefix = p.p.literal(&self.call_id.prefix);
                    let until = p.p.until_str(&self.call_id.suffix);
                    let id = p.tool_id(until);
                    let suffix = p.p.literal(&self.call_id.suffix);
                    call_id_section = p.p.optional(p.p.sequence(&[prefix, id, suffix]));
                } else {
                    let prefix = p.p.literal(&self.call_id.prefix);
                    let until = p.p.until_str(&self.arguments.start);
                    let id = p.tool_id(until);
                    call_id_section = p.p.optional(p.p.sequence(&[prefix, id]));
                }
            }
            let atomic_peek = if !self.arguments.name_prefix.is_empty() && !required.is_empty() {
                Some(p.p.peek(p.p.literal(&self.arguments.name_prefix)))
            } else {
                None
            };
            let func = self.build_func_parser(p, &name, call_id_section, have_call_id, args_seq, atomic_peek);
            let rule = p.rule(&format!("tool-{}", index), func, false);
            tool_choice_parser = p.p.choice_ids(&[tool_choice_parser.id, rule.id]);
        }
        let require_tools = inputs.tool_choice.is_required();
        let mut tool_calls = if !self.format.per_call_start.is_empty() {
            let start = p.p.literal(&self.format.per_call_start);
            let end = p.p.literal(&self.format.per_call_end);
            let space = p.p.space();
            let wrapped = p.p.sequence(&[start, space, tool_choice_parser, space, end]);
            let calls = if inputs.parallel_tool_calls {
                let ws = p.p.space();
                let more = p.p.zero_or_more(p.p.sequence(&[ws, wrapped]));
                p.trigger_rule("tool-call", p.p.sequence(&[wrapped, more, p.p.space()]))
            } else {
                p.trigger_rule("tool-call", p.p.sequence(&[wrapped, p.p.space()]))
            };
            if !self.format.section_start.is_empty() {
                let section_start = p.p.literal(&self.format.section_start);
                let ws1 = p.p.space();
                let ws2 = p.p.space();
                let section_end = if self.format.section_end.is_empty() {
                    p.p.end()
                } else {
                    let literal = p.p.literal(&self.format.section_end);
                    p.p.sequence(&[literal, p.p.space()])
                };
                let section = p.p.sequence(&[section_start, ws1, calls, ws2, section_end]);
                p.trigger_rule("tool-calls", section)
            } else {
                calls
            }
        } else {
            let separator = p.p.literal(", ");
            let start = p.p.literal(&self.format.section_start);
            let end = p.p.literal(&self.format.section_end);
            let space = p.p.space();
            if inputs.parallel_tool_calls {
                let more = p.p.zero_or_more(p.p.sequence(&[separator, tool_choice_parser]));
                let body = p.p.sequence(&[start, space, tool_choice_parser, more, space, end]);
                p.trigger_rule("tool-call", body)
            } else {
                let body = p.p.sequence(&[start, space, tool_choice_parser, space, end]);
                p.trigger_rule("tool-call", body)
            }
        };
        if !require_tools {
            tool_calls = p.p.optional(tool_calls);
        }
        let trigger_marker = if self.format.section_start.is_empty() {
            self.format.per_call_start.clone()
        } else {
            self.format.section_start.clone()
        };
        let before = if trigger_marker.is_empty() {
            p.p.eps()
        } else {
            p.p.until_str(&trigger_marker)
        };
        let before = p.content(before);
        let finish = p.p.end();
        p.p.sequence(&[reasoning_parser, p.p.optional(before), tool_calls, finish])
    }
}

impl AutoParser {
    /// Build the chat parser described by the analysis result.
    pub fn build_parser(&self, inputs: &GenerationParams, generation_prompt: &str) -> Result<PegArena> {
        if !self.analysis_complete {
            return Err(Error::new(
                "Cannot call build_parser on autoparser without performing analysis first, call analyze_template(...)",
            ));
        }
        build_chat_peg_parser(|p| {
            let mut ctx = ParserBuildContext {
                p,
                inputs,
                reasoning_parser: p.p.eps(),
                extracting_reasoning: false,
            };
            let extract_reasoning = inputs.reasoning_format != crate::chat::ReasoningFormat::None;
            ctx.extracting_reasoning = extract_reasoning && self.reasoning.mode != ReasoningMode::None;
            ctx.reasoning_parser = self.reasoning.build_parser(&mut ctx);

            let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
            let has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
            let mut pure_content = self.reasoning.mode == ReasoningMode::None;
            let parser = if has_response_format {
                let json_parser = ctx.p.p.json();
                let schema = ctx.p.schema(json_parser, "response-format-schema", &inputs.json_schema, false);
                let response_format = ctx.p.rule("response-format", ctx.p.content(schema), false);
                let ws = ctx.p.p.space();
                let fence_start = ctx.p.p.literal("```json");
                let ws2 = ctx.p.p.space();
                let fence_end = ctx.p.p.literal("```");
                let fenced = ctx.p.p.sequence(&[fence_start, ws, response_format, ws2, fence_end]);
                let ws3 = ctx.p.p.space();
                let ws4 = ctx.p.p.space();
                let plain = ctx.p.p.sequence(&[ws3, response_format, ws4]);
                let body = ctx.p.p.choice(&[fenced, plain]);
                let finish = ctx.p.p.end();
                pure_content = false;
                ctx.p.p.sequence(&[ctx.reasoning_parser, body, finish])
            } else {
                let reasoning_parser = ctx.reasoning_parser;
                if has_tools && !inputs.tool_choice.is_none() && self.jinja_caps.supports_tool_calls {
                    pure_content = false;
                    self.tools.build_parser(&mut ctx, reasoning_parser, &self.content)
                } else {
                    self.content.build_parser(&mut ctx, reasoning_parser)
                }
            };
            let reasoning_start = trim_whitespace(&self.reasoning.start);
            let prefix = ctx.p.prefix(generation_prompt, reasoning_start);
            if pure_content {
                ctx.p.p.sequence(&[prefix, parser])
            } else {
                ctx.p.p.sequence(&[prefix, ctx.p.p.space(), parser])
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peg_captures_tagged_text() {
        let arena = build_peg_parser(|b| {
            let key = b.literal("a=");
            let value = b.tag("value", b.until_str(";"));
            b.sequence(&[key, value, b.literal(";")])
        })
        .unwrap();
        let mut ctx = PegParseContext::with_input("a=hello;", PegParseFlags::NONE);
        let result = arena.parse(&mut ctx, 0).unwrap();
        assert!(result.success());
        assert_eq!(ctx.ast.get(result.nodes[0]).tag, "value");
        assert_eq!(ctx.ast.get(result.nodes[0]).text, "hello");
    }

    #[test]
    fn mapper_extracts_content_and_tool() {
        let arena = build_chat_peg_parser(|p| {
            let content = p.content(p.p.until_str("<tool>"));
            let name = p.tool_name(p.p.literal("f"));
            let args = p.tool_args(p.p.literal("{\"x\":1}"));
            let open = p.tool_open(p.p.literal("<tool>"));
            let body = p.tool(p.p.sequence(&[open, name, args]));
            let finish = p.p.end();
            p.p.sequence(&[content, body, finish])
        })
        .unwrap();
        let mut ctx = PegParseContext::with_input("hi<tool>f{\"x\":1}", PegParseFlags::NONE);
        let result = arena.parse(&mut ctx, 0).unwrap();
        assert!(result.success());
        let mut mapper = ChatPegMapper::new(ChatMsg::default());
        mapper.from_ast(&ctx.ast, &result);
        assert_eq!(mapper.result.content, "hi");
        assert_eq!(mapper.result.tool_calls.len(), 1);
        assert_eq!(mapper.result.tool_calls[0].name, "f");
        assert_eq!(mapper.result.tool_calls[0].arguments, "{\"x\":1}");
    }

    #[test]
    fn diff_split_aligned_on_markers() {
        let diff = calculate_diff_split("<a>X</a>", "<a>Y</a>");
        assert_eq!(diff.prefix, "<a>");
        assert_eq!(diff.suffix, "</a>");
        assert_eq!(diff.left, "X");
        assert_eq!(diff.right, "Y");
    }

    #[test]
    fn python_quotes_and_literals_convert() {
        assert_eq!(
            normalize_quotes_to_json("{'a': True, 'b': None}"),
            "{\"a\": true, \"b\": null}"
        );
    }
}

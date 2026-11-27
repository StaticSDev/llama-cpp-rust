//! Grammar node types, element aliases, and constants.

use crate::core::Token;

pub(crate) const MAX_REPETITION_THRESHOLD: u64 = 2000;
pub(crate) const GRAMMAR_RECURSIVE_LIMIT: u32 = 256;

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
    pub(crate) fn new(type_: Gretype, value: u32) -> Self {
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

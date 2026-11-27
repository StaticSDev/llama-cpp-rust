//! Rust port of `llama-grammar.h` / `llama-grammar.cpp`.
//!
//! C++ stack entries are `const llama_grammar_element *`. Here a `GrammarPos`
//! is a `(rule, index)` pair, which gives the same identity and `+ 1` semantics
//! without raw pointers. Vocab access is supplied by the caller: token pieces
//! and the eog flag live on `TokenData`, and `<token>` parsing uses an optional
//! tokenizer closure.

mod parser;
mod state;
mod types;
mod utf8;

#[cfg(test)]
mod tests;

pub use parser::*;
pub use state::*;
pub use types::*;
pub use utf8::*;

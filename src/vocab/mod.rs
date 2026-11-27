//! Rust port of `llama-vocab.cpp` and `llama-vocab.h`.
//!
//! The original uses GGUF/model-loader objects to read metadata. Here the same
//! data is supplied through [`Metadata`], so the port stays free of FFI and
//! external crates. Unicode tables from `unicode.cpp` are approximated with
//! `std` character classification; see the note on `unicode_cpt_flags_from_cpt`.

pub mod types;
pub mod trie;
pub mod sessions;
pub mod tokenize;
pub mod detokenize;
pub mod metadata;

#[cfg(test)]
mod tests;

pub use types::*;
pub use trie::*;
pub use sessions::*;
pub use tokenize::*;
pub use detokenize::*;
pub use metadata::*;

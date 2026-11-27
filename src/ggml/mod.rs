//! Rust port of the ggml core library.
//!
//! Source: `reference/llama.cpp/ggml/`. Types and helpers live in `types`;
//! the remaining modules are the planned translation units of ggml.

pub mod types;
pub mod tensor;
pub mod quants;
pub mod alloc;
pub mod backend;
pub mod opt;
pub mod threading;


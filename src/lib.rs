//! Rust port of llama.cpp.
//!
//! Reference source: ggml-org/llama.cpp @ 15fcfc7357f8de73671d39e7d8c594e1beb3052d.
//! The original C/C++ tree lives in `reference/llama.cpp` locally and is not
//! committed. Every ported file keeps the same base name as its C/C++ source.
//!
//! This is a dry port: modules are translated module by module, no build or
//! runtime verification has been performed yet.

pub mod arch;
pub mod batch;
pub mod cparams;
pub mod chat;
pub mod core;
pub mod gguf;
pub mod grammar;
pub mod hparams;
pub mod io;
pub mod json;
pub mod kv_cache;
pub mod r#impl;
pub mod sampler;
pub mod unicode;
pub mod vocab;

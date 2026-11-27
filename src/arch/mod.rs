//! Rust port of `llama-arch.h` / `llama-arch.cpp`.
//!
//! gguf constants (sync with gguf.py): architecture detection, GGUF key and
//! tensor-name helpers, and per-tensor buffer-classification metadata.

#[path = "enum.rs"]
mod arch_enum;
mod kv;
mod tensor_info;

pub use arch_enum::*;
pub use kv::*;
pub use tensor_info::*;

#[cfg(test)]
mod tests;

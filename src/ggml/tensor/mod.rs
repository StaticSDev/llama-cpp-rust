//! Rust port of the ggml tensor core: struct, creation, views and ops.
//!
//! Source: `ggml.h`, `ggml.c` and `ggml.cpp` (the latter only re-exports the C
//! API). `src` and `view_src` pointers become `Arc` handles; tensor data is a
//! shared `Vec<u8>` so views alias their source's bytes at `view_offs`.

mod create;
mod helpers;
mod ops;
mod ops_more;
mod structure;
mod types;
mod views;

pub use create::*;
pub use helpers::*;
pub use ops::*;
pub use ops_more::*;
pub use structure::*;
pub use types::*;
pub use views::*;

#[cfg(test)]
mod tests;

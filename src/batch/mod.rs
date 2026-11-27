//! Port of llama.cpp `llama-batch.h` / `llama-batch.cpp`.
//!
//! `llama_ubatch` owns its arrays directly. In C++ the struct holds raw
//! pointers into a shared `data_t`; the safe-Rust equivalent is to keep the
//! vectors in the struct itself. Algorithm order and layout are unchanged.
//!
//! Debug-only logging (`ubatch_print`, `LLAMA_BATCH_DEBUG`) and the malloc-based
//! C entry points are not part of this port.

mod batch;
mod seq_set;
mod split;
mod ubatch;

pub use batch::*;
pub use seq_set::*;
#[allow(unused_imports)]
pub use split::*;
pub use ubatch::*;

/// `llama_pos`.
pub type Pos = i32;

pub const LLAMA_MAX_SEQ: usize = 512;
pub const GGML_MROPE_SECTIONS: usize = 4;

#[cfg(test)]
mod tests;

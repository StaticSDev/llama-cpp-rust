//! Self-contained CPU sampling port of llama.cpp's built-in samplers.
//!
//! Reference: `reference/llama.cpp/src/llama-sampler.cpp` plus the sampler
//! chain order from `reference/llama.cpp/common/common.h`.

mod sampler;
mod tests;
pub use sampler::*;

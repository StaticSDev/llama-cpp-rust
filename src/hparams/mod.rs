//! Rust port of `llama-hparams.h` / `llama-hparams.cpp`.
//!
//! The hyperparameter struct, its derived dimensions and the per-layer
//! pattern helpers. Fields use the `_impl` suffix where a getter shares the
//! name, mirroring the C++ layout.

mod getters;
mod params;

pub use params::*;

#[cfg(test)]
mod tests;

//! Minimal CPU inference runtime for dense/hybrid Qwen models.

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod generate;
pub mod loader;
pub mod math;
pub mod qwen35;
pub mod types;

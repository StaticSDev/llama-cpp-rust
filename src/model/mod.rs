//! Rust port of `llama-model.cpp` / `llama-model.h`.
//!
//! The class hierarchy of the original is represented by one [`LlamaModel`]
//! plus traits/functions that arch-specific modules can call. ggml handles
//! become owned [`Tensor`] and [`Buffer`] descriptors; everything is pure Rust.

pub mod buffers;
pub mod getters;
pub mod hparams;
pub mod loader;
pub mod model;
pub mod tensors;
pub mod types;

#[cfg(test)]
mod tests;

pub use buffers::{
    assign_devices, make_cpu_buft_list, make_gpu_buft_list, meta_device_get_split_state,
    normalized_splits, select_buft, BuftEntry, LayerDev, ModelImpl, SPLIT_AXIS_0, SPLIT_AXIS_1,
    SPLIT_AXIS_2, SPLIT_AXIS_3, SPLIT_AXIS_MIRRORED, SPLIT_AXIS_NONE, SPLIT_AXIS_PARTIAL,
};
pub use getters::{GraphPlan, MemoryDescriptor, MemoryKind};
pub use loader::ModelLoader as LlamaModelLoader;
pub use model::LlamaModel;
pub use types::*;

use crate::core::Result;
use crate::gguf::GgufContext;

/// Convenience wrapper around `llama_model_create(ml, params)` plus `load`.
pub fn load_model_from_gguf(ctx: GgufContext, params: ModelParams) -> Result<LlamaModel> {
    let mut ml = LlamaModelLoader::from_gguf(ctx, &params)?;
    let mut model = LlamaModel::create_from_loader(&ml, params)?;
    model.load(&mut ml)?;
    Ok(model)
}

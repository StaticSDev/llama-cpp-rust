//! Rust port of `llama-quant.cpp`.
//!
//! Quantization parameters, tensor categorization and type selection,
//! importance-matrix handling, the `--target-bpw` optimizer and the file
//! driver live here. ggml kernels are abstracted by the `Quantizer` trait.

mod driver;
mod gguf_io;
mod imatrix;
mod kernels;
mod names;
mod params;
mod public;
mod regex_lite;
mod state;
mod target_bpw;
mod target_metrics;
mod target_types;
mod types;

pub use driver::llama_model_quantize_impl;
pub use gguf_io::{remove_key, set_u32 as set_kv_u32, GgufKv, LoadedGguf, OutputTensor};
pub use imatrix::{
    broadcast, prepare_side_data, remap_imatrix, remap_layer, report_missing_imatrix, SideData,
};
pub use kernels::{dequantize_slab, quantize_slab, NativeQuantizer, Quantizer};
pub use names::{
    category_is_attn_v, tensor_allows_quantization, tensor_get_category, tensor_is_expert,
    tensor_name_match_output_weight, tensor_name_match_token_embd, tensor_name_match_unused_kv,
    tensor_requires_imatrix, TensorCategory,
};
pub use params::{
    ftype_default_type, imatrix_map, validate_side_data, FType, ImatrixData, KvOverride,
    KvOverrideType, ModelQuantizeDesc, QuantizeParams, TensorTypeOverride,
    LLAMA_QUANT_MAX_BUF_SIZE,
};
pub use public::{
    llama_model_quantize, llama_model_quantize_default_params, llama_quant_compute_types,
    llama_quant_free, llama_quant_init, llama_quant_model_from_metadata,
    llama_quant_tensor_allows_quantization,
};
pub use regex_lite::{regex_match, regex_search};
pub use state::{
    metadata_from_tensors, model_info_from_desc, ModelInfo, QuantizeState, TensorMetadata,
};
pub use target_bpw::{simplify_pareto, target_bpw_type};
pub use target_metrics::{
    cnif, compute_quant_error, outlier_smoothing, randomized_svd_condition_number, squash_kappa,
    winsorize_values, MseCache, QuantError, TensorScore, WceCache,
};
pub use target_types::{
    is_angle_sensitive, is_inner_product_sensitive, is_iq, TypeChoice, TypeScores, ATTN_V_BOOST,
    BOOST_TAPER, DOWNWARD_THRESHOLD, EPSILON, FFN_DOWN_BOOST, HASH_MAGIC, LQ_INFINITE,
    OUTPUT_BOOST, QUANT_TYPES, STATE_MAGIC, TIED_EMBD_BOOST,
};
pub use types::{
    bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, fallback_type_for, QTensor, QType,
};

pub(crate) const GGUF_ALIGNMENT: usize = 32;

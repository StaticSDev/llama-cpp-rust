//! Public interface exposed by `llama-quant.h` and `llama-ext.h`.

use crate::core::Result;

use super::driver::llama_model_quantize_impl;
use super::names::tensor_allows_quantization;
use super::params::{ftype_default_type, FType, ModelQuantizeDesc, QuantizeParams};
use super::state::{metadata_from_tensors, model_info_from_desc, ModelInfo, QuantizeState};
use super::types::QTensor;

/// `llama_model_quantize_default_params`.
pub fn llama_model_quantize_default_params() -> QuantizeParams {
    QuantizeParams::default()
}

/// `llama_model_quantize`: returns 0 on success, 1 on error.
pub fn llama_model_quantize(fname_inp: &str, fname_out: &str, params: &QuantizeParams) -> u32 {
    match llama_model_quantize_impl(fname_inp, fname_out, params) {
        Ok(()) => 0,
        Err(err) => {
            crate::llama_log_error!(
                "llama_model_quantize: failed to quantize: {}\n",
                err.message
            );
            1
        }
    }
}

/// `llama_quant_init`: build the state used by the helper functions.
pub fn llama_quant_init(model: ModelInfo, params: &QuantizeParams) -> QuantizeState {
    QuantizeState::new(model, params)
}

/// `llama_quant_free`: Rust ownership replaces the C++ heap allocation.
pub fn llama_quant_free(_qs: QuantizeState) {}

/// `llama_quant_model_from_metadata`.
pub fn llama_quant_model_from_metadata(desc: &ModelQuantizeDesc) -> Result<ModelInfo> {
    model_info_from_desc(desc)
}

/// `llama_quant_tensor_allows_quantization`.
pub fn llama_quant_tensor_allows_quantization(qs: &QuantizeState, tensor: &QTensor) -> bool {
    tensor_allows_quantization(&qs.params, qs.model.arch, tensor)
}

/// `llama_quant_compute_types`: compute target types for a tensor list.
pub fn llama_quant_compute_types(
    qs: &mut QuantizeState,
    ftype: FType,
    tensors: &[QTensor],
) -> Result<Vec<super::types::QType>> {
    qs.reset_counters();
    let mut metadata = metadata_from_tensors(tensors);
    qs.init_counters(&mut metadata);
    let mut local = qs.params.clone();
    local.ftype = ftype;
    let default_type = ftype_default_type(ftype);
    let mut result = Vec::with_capacity(tensors.len());
    for (t, tm) in tensors.iter().zip(metadata.iter()) {
        result.push(qs.tensor_get_type(&local, t, default_type, tm)?);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::types::{QTensor, QType};

    #[test]
    fn compute_types_basic() {
        let mut desc = ModelQuantizeDesc::default();
        desc.architecture = "llama".into();
        desc.n_layer = 32;
        desc.n_head = 32;
        desc.n_head_kv = 8;
        let model = llama_quant_model_from_metadata(&desc).unwrap();
        let params = QuantizeParams::default();
        let mut qs = llama_quant_init(model, &params);
        let tensors = vec![QTensor {
            name: "blk.0.attn_v.weight".into(),
            ne: [128, 4096, 1, 1],
            ty: QType::F16,
            data: Vec::new(),
            split: 0,
            offset: 0,
        }];
        let types = llama_quant_compute_types(&mut qs, FType::MostlyQ4KM, &tensors).unwrap();
        assert_eq!(types.len(), 1);
        assert!(types[0].is_quantized());
    }
}

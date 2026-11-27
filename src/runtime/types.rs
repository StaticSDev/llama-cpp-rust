//! Shared types for the small CPU inference runtime.

use crate::core::{Error, Result};
use crate::gguf::{ggml_blck_size, ggml_type_size, GgmlType};
use crate::vocab::LlamaVocab;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Weight {
    /// Logical shape `[in_features, out_features]`, fastest dimension first.
    pub ne: [usize; 2],
    pub ty: GgmlType,
    /// Byte offset inside `Weights::data`.
    pub offset: usize,
}

impl Weight {
    pub fn in_features(&self) -> usize {
        self.ne[0]
    }
    pub fn out_features(&self) -> usize {
        self.ne[1]
    }

    pub fn row_bytes(&self) -> usize {
        match self.ty {
            GgmlType::F32 => self.in_features() * 4,
            GgmlType::F16 | GgmlType::Bf16 => self.in_features() * 2,
            _ => {
                let block = ggml_blck_size(self.ty) as usize;
                let size = ggml_type_size(self.ty);
                if block == 0 || self.in_features() % block != 0 {
                    return 0;
                }
                self.in_features() / block * size
            }
        }
    }

    pub fn row<'a>(&self, data: &'a [u8], row: usize) -> Result<&'a [u8]> {
        let n = self.row_bytes();
        let start = self.offset
            + row
                .checked_mul(n)
                .ok_or_else(|| Error::new("weight row overflow"))?;
        data.get(start..start + n)
            .ok_or_else(|| Error::new(format!("weight row {row} outside GGUF data")))
    }
}

#[derive(Clone, Debug)]
pub struct ModelParams {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub rope_dim: usize,
    pub rope_sections: [usize; 4],
    pub conv_kernel: usize,
    pub ssm_inner: usize,
    pub ssm_state: usize,
    pub ssm_group: usize,
    pub ssm_dt_rank: usize,
    pub full_attn_interval: usize,
}

impl ModelParams {
    pub fn is_recurrent(&self, il: usize) -> bool {
        il < self.n_layer && (il + 1) % self.full_attn_interval != 0
    }
}

pub struct Weights {
    pub map: HashMap<String, Weight>,
    /// Host copies of small F32 tensors, available even in GPU mode.
    pub small: HashMap<String, Vec<f32>>,
    pub data: Arc<Vec<u8>>,
    pub params: ModelParams,
    pub vocab: LlamaVocab,
    #[cfg(feature = "cuda")]
    pub gpu: Option<std::sync::Arc<crate::runtime::cuda::CudaWeights>>,
}

impl Weights {
    pub fn get(&self, name: &str) -> Result<&Weight> {
        self.map
            .get(name)
            .ok_or_else(|| Error::new(format!("missing tensor {name}")))
    }

    #[cfg(feature = "cuda")]
    pub fn gpu(&self) -> Option<&crate::runtime::cuda::CudaWeights> {
        self.gpu.as_deref()
    }
}

/// KV state for one full-attention layer: `[head][token][dim]`.
#[derive(Default)]
pub struct FullAttnLayer {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

/// Recurrent state for one gated-delta-net layer.
pub struct LinearLayer {
    /// `(conv_kernel - 1) * channels` floats, oldest first.
    pub conv: Vec<f32>,
    /// `n_v_heads * d_state * d_state` floats, transposed as in ggml.
    pub ssm: Vec<f32>,
}

pub struct ModelState {
    pub pos: usize,
    pub full: Vec<FullAttnLayer>,
    pub linear: Vec<LinearLayer>,
}

impl ModelState {
    pub fn new(w: &Weights) -> Self {
        let p = &w.params;
        let n_full = p.n_layer / p.full_attn_interval.max(1);
        let n_linear = p.n_layer - n_full;
        let channels = 2 * p.ssm_group * p.ssm_state + p.ssm_dt_rank * p.ssm_state;
        Self {
            pos: 0,
            full: (0..n_full)
                .map(|_| FullAttnLayer {
                    k: vec![Vec::new(); p.n_head_kv],
                    v: vec![Vec::new(); p.n_head_kv],
                })
                .collect(),
            linear: (0..n_linear)
                .map(|_| LinearLayer {
                    conv: vec![0.0; (p.conv_kernel.saturating_sub(1)) * channels],
                    ssm: vec![0.0; p.ssm_dt_rank * p.ssm_state * p.ssm_state],
                })
                .collect(),
        }
    }
}

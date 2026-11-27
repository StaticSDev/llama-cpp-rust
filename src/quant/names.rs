//! Tensor-name matching and the allow/require decisions from `llama-quant.cpp`.

use crate::arch::Arch;

use super::params::{FType, QuantizeParams};
use super::types::{QTensor, QType};

/// `tensor_category`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorCategory {
    TokenEmbd,
    AttentionQ,
    AttentionV,
    AttentionK,
    AttentionQkv,
    AttentionKvB,
    AttentionOutput,
    AttentionGate,
    FfnUp,
    FfnGate,
    FfnDown,
    FfnLatent,
    Output,
    Other,
}

pub fn tensor_name_match_token_embd(name: &str) -> bool {
    name == "token_embd.weight" || name == "per_layer_token_embd.weight"
}

pub fn tensor_name_match_output_weight(name: &str) -> bool {
    name == "output.weight"
}

/// Extract the block index from `blk.<n>.<rest>`.
pub fn parse_block(name: &str) -> Option<i32> {
    let rest = name.strip_prefix("blk.")?;
    let end = rest.find('.')?;
    rest[..end].parse::<i32>().ok()
}

/// `tensor_name_match_unused_kv`.
pub fn tensor_name_match_unused_kv(name: &str, n_unused_from: i32) -> bool {
    if n_unused_from < 0 {
        return false;
    }
    if !name.ends_with(".weight") {
        return false;
    }
    let stem = &name[..name.len() - ".weight".len()];
    let Some(idx) = stem.rfind("attn_") else {
        return false;
    };
    let key = &stem[idx..];
    if key != "attn_k" && key != "attn_v" {
        return false;
    }
    let Some(block) = parse_block(name) else {
        return false;
    };
    block >= n_unused_from
}

/// `tensor_get_category`.
pub fn tensor_get_category(name: &str) -> TensorCategory {
    if tensor_name_match_output_weight(name) {
        return TensorCategory::Output;
    }
    if tensor_name_match_token_embd(name) {
        return TensorCategory::TokenEmbd;
    }
    if name.contains("attn_qkv.weight") || name.contains("attn_qkv_gate.weight") {
        return TensorCategory::AttentionQkv;
    }
    if name.contains("attn_kv_b.weight") {
        return TensorCategory::AttentionKvB;
    }
    if name.contains("attn_v.weight") {
        return TensorCategory::AttentionV;
    }
    if name.contains("attn_k.weight") {
        return TensorCategory::AttentionK;
    }
    if name.contains("attn_q.weight") {
        return TensorCategory::AttentionQ;
    }
    if name.contains("attn_output.weight") {
        return TensorCategory::AttentionOutput;
    }
    if name.contains("ffn_up") {
        return TensorCategory::FfnUp;
    }
    if name.contains("ffn_gate") {
        return TensorCategory::FfnGate;
    }
    if name.contains("ffn_latent") {
        return TensorCategory::FfnLatent;
    }
    if name.contains("attn_gate.weight") {
        return TensorCategory::AttentionGate;
    }
    if name.contains("ffn_down") {
        return TensorCategory::FfnDown;
    }
    TensorCategory::Other
}

pub fn tensor_is_expert(name: &str) -> bool {
    name.contains("_exps.") || name.contains("_chexps.")
}

pub fn category_is_attn_v(cat: TensorCategory) -> bool {
    matches!(
        cat,
        TensorCategory::AttentionV | TensorCategory::AttentionQkv | TensorCategory::AttentionKvB
    )
}

/// `tensor_allows_quantization`.
pub fn tensor_allows_quantization(params: &QuantizeParams, arch: Arch, tensor: &QTensor) -> bool {
    if params.only_copy {
        return false;
    }
    if tensor.n_dims() < 2 {
        return false;
    }
    let name = tensor.name.as_str();
    let mut quantize = name.ends_with("weight");

    quantize &= !name.contains("_norm.weight");
    quantize &= params.quantize_output_tensor || name != "output.weight";
    quantize &= !name.contains("ffn_gate_inp.weight");
    quantize &= !name.contains("ffn_gate_tid2eid.weight");
    quantize &= !name.contains("engram_gate.weight");

    if arch == Arch::HyV4Flash {
        quantize &= !name.contains("engram_kv_b.weight");
        quantize &= !name.contains("engram_hc_scale.weight");
        quantize &= !name.contains("indexer_compressor_ape.weight");
    }

    quantize &= !name.contains("altup");
    quantize &= !name.contains("laurel");
    quantize &= !name.contains("per_layer_model_proj");

    if arch == Arch::Limite {
        quantize &= !name.contains("attn_gate.weight");
        quantize &= !name.contains("attn_ve_gate.weight");
        quantize &= !name.contains("mudd_");
    }
    if arch == Arch::Inkling2 || arch == Arch::Inkling2Loop {
        quantize &= !name.contains("_conv.weight");
        quantize &= !name.contains("_clip.weight");
        quantize &= !name.contains("_res_query.weight");
        quantize &= !name.contains("attn_rel_b.weight");
        quantize &= !name.contains(".loop_");
    }
    if arch == Arch::Zaya1 {
        quantize &= !name.contains("ffn_router_");
        quantize &= !name.contains("cca_conv_grp");
    }

    quantize &= name != "pos_embd.weight";
    quantize &= name != "token_types.weight";

    quantize &= !name.contains("ssm_conv1d");
    quantize &= !name.contains("shortconv.conv.weight");
    quantize &= !name.contains("conv2d_spatial.weight");
    quantize &= !name.starts_with("a.conv1d.");
    quantize &= !name.contains("conv_dw");
    quantize &= !name.starts_with("a.gen.wav.quant.");

    quantize &= !name.contains("indexer.k_proj.weight");
    quantize &= !name.contains("indexer.q_proj.weight");

    if arch == Arch::Metis16 {
        quantize &= !name.starts_with("depth_mem_");
        quantize &= !name.starts_with("cont_");
        quantize &= !name.contains("stream_embd");
        quantize &= !name.contains("attn_pass_lora");
        quantize &= !name.contains("ffn_exp_embd");
        quantize &= !name.contains("hc_");
    }
    if arch == Arch::KimiK31 {
        quantize &= !name.contains("indexer.attn_q_b");
        quantize &= !name.contains("indexer.attn_k");
        quantize &= !name.contains("indexer.proj");
    }
    if arch == Arch::Glm5Next {
        quantize &= !name.contains("hc_");
        quantize &= !name.contains("indexer.attn_q_b");
        quantize &= !name.contains("indexer.attn_k");
        quantize &= !name.contains("indexer.proj");
        quantize &= !name.contains("indexer_compressor_gate");
        quantize &= !name.contains("indexer_compressor_ape");
        quantize &= !name.contains("ssm_f_a.weight");
        quantize &= !name.contains("ssm_f_b.weight");
        quantize &= !name.contains("ssm_g_a.weight");
        quantize &= !name.contains("ssm_g_b.weight");
        quantize &= !name.contains("ssm_beta.weight");
        quantize &= !name.contains("attn_kv_a_mqa.weight");
        quantize &= !name.contains("attn_k_b.weight");
        quantize &= !name.contains("attn_v_b.weight");
    }
    if arch == Arch::Xing40 {
        quantize &= !name.contains("hc_");
    }
    if arch == Arch::Raon1Mini {
        quantize &= !name.contains("hc_");
        quantize &= !name.contains("ssm_f_a.weight");
        quantize &= !name.contains("ssm_f_b.weight");
        quantize &= !name.contains("ssm_beta.weight");
        quantize &= !name.contains("ssm_rda.weight");
    }
    if arch == Arch::VllmDecision2 {
        quantize &= !name.starts_with("decision.");
    }

    quantize &= !name.contains("time_mix_first.weight");
    quantize &= !name.contains("time_mix_w0.weight");
    quantize &= !name.contains("time_mix_w1.weight");
    quantize &= !name.contains("time_mix_w2.weight");
    quantize &= !name.contains("time_mix_v0.weight");
    quantize &= !name.contains("time_mix_v1.weight");
    quantize &= !name.contains("time_mix_v2.weight");
    quantize &= !name.contains("time_mix_a0.weight");
    quantize &= !name.contains("time_mix_a1.weight");
    quantize &= !name.contains("time_mix_a2.weight");
    quantize &= !name.contains("time_mix_g1.weight");
    quantize &= !name.contains("time_mix_g2.weight");
    quantize &= !name.contains("time_mix_decay_w1.weight");
    quantize &= !name.contains("time_mix_decay_w2.weight");
    quantize &= !name.contains("time_mix_lerp_fused.weight");

    quantize &= !name.contains("attn_rel_b.weight");

    quantize &= !name.contains(".position_embd");
    quantize &= !name.contains("sam.pos_embd");
    quantize &= !name.contains("sam.neck.");
    quantize &= !name.contains("sam.net_");
    quantize &= !name.contains(".rel_pos");
    quantize &= !name.contains(".patch_embd");
    quantize &= !name.contains(".patch_merger");
    quantize &= !name.contains("a.rvq.codebook");
    quantize &= !name.contains("mm.a.code_embd");

    quantize
}

/// `tensor_requires_imatrix`.
pub fn tensor_requires_imatrix(name: &str, dst: QType, ftype: FType) -> bool {
    if tensor_name_match_token_embd(name) || tensor_name_match_output_weight(name) {
        return false;
    }
    match dst {
        QType::Iq3Xxs | QType::Iq2Xxs | QType::Iq2Xs | QType::Iq2S | QType::Iq1M | QType::Iq1S => {
            true
        }
        QType::Q2K => ftype == FType::MostlyQ2KS,
        _ => false,
    }
}

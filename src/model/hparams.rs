//! `llama_model_base::load_hparams`, `load_vocab`, `load_swa_pattern` and the
//! naming helpers from `llama-model.cpp`.

use super::model::LlamaModel;
use super::types::{LlmType, RopeType};
use super::LlamaModelLoader;
use crate::arch::{arch_is_recurrent, Arch, Kv};
use crate::core::{Error, Result};
use crate::hparams::{
    ExpertGatingFuncType, FfnOpType, Hparams, PoolingType, RopeScalingType, RopeSectionOrder,
    LLAMA_MAX_EXPERTS, LLAMA_MAX_LAYERS,
};
use crate::vocab::{VocabPreType, VocabType};

/// `llama_rope_scaling_type_name`.
pub fn rope_scaling_type_name(ty: RopeScalingType) -> &'static str {
    match ty {
        RopeScalingType::None => "none",
        RopeScalingType::Linear => "linear",
        RopeScalingType::Yarn => "yarn",
        RopeScalingType::Extrapolation => "longrope",
    }
}

/// `llama_rope_scaling_type_from_string`.
pub fn rope_scaling_type_from_string(name: &str) -> Option<RopeScalingType> {
    match name {
        "none" => Some(RopeScalingType::None),
        "linear" => Some(RopeScalingType::Linear),
        "yarn" => Some(RopeScalingType::Yarn),
        "longrope" => Some(RopeScalingType::Extrapolation),
        _ => None,
    }
}

/// `llama_rope_section_order_name`.
pub fn rope_section_order_name(order: RopeSectionOrder) -> &'static str {
    match order {
        RopeSectionOrder::Tyxz => "tyxz",
        RopeSectionOrder::Zyxt => "zyxt",
        RopeSectionOrder::Unspecified => "unknown",
    }
}

pub fn rope_section_order_from_string(name: &str) -> Option<RopeSectionOrder> {
    match name {
        "tyxz" => Some(RopeSectionOrder::Tyxz),
        "zyxt" => Some(RopeSectionOrder::Zyxt),
        _ => None,
    }
}

/// Resolved activation names from `LLM_FFN_OP_TYPES_FROM_STRING`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnOp {
    GegluErf,
    Geglu,
    Swiglu,
    Relu,
    Reglu,
}

/// `llm_ffn_op_type_from_string`. `hparams.llm_ffn_op` only carries the
/// sentinel in this port, so the resolved flavor is returned separately.
pub fn llm_ffn_op_type_from_string(name: &str, fallback: Option<FfnOp>) -> Option<FfnOp> {
    match name {
        "gelu" | "gelu_python" => Some(FfnOp::GegluErf),
        "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" | "gelu_accurate" | "gelu_python_tanh"
        | "geglu" => Some(FfnOp::Geglu),
        "silu" | "swish" | "swiglu" => Some(FfnOp::Swiglu),
        "relu" => Some(FfnOp::Relu),
        "reglu" => Some(FfnOp::Reglu),
        _ => fallback,
    }
}

pub fn expert_gating_func_name(gating: u32) -> &'static str {
    match gating {
        x if x == ExpertGatingFuncType::Softmax as u32 => "softmax",
        x if x == ExpertGatingFuncType::Sigmoid as u32 => "sigmoid",
        x if x == ExpertGatingFuncType::SqrtSoftplus as u32 => "sqrtsoftplus",
        _ => "unknown",
    }
}

pub fn pooling_type_from_string(name: &str) -> Option<PoolingType> {
    match name {
        "none" => Some(PoolingType::None),
        "mean" => Some(PoolingType::Mean),
        "cls" => Some(PoolingType::Cls),
        "last" => Some(PoolingType::Last),
        "rank" => Some(PoolingType::Rank),
        _ => None,
    }
}

/// `llama_model_rope_type`, ported as a pure function of arch and hparams.
pub fn rope_type_for_arch(arch: Arch, hparams: &Hparams) -> Result<RopeType> {
    use Arch::*;
    if matches!(
        arch,
        Clip | Gpt2
            | Gptj
            | Mpt
            | Refact
            | Bloom
            | Mamba
            | Mamba2
            | Jamba
            | JinaBertV2
            | T5
            | T5encoder
            | Jais
            | Rwkv6
            | Rwkv6qwen2
            | Rwkv7
            | Arwkv7
            | WavtokenizerDec
            | NemotronH
            | NemotronHMoe
            | Metis16
            | KimiLinear
            | KimiK3
            | KimiK31
            | Raon1Mini
            | Glm5Next
            | Starcoder
            | Inkling2
            | Inkling2Loop
            | Kokoro
    ) {
        return Ok(RopeType::None);
    }
    if matches!(
        arch,
        Llama
            | Llada
            | Llama4
            | Deci
            | Baichuan
            | Internlm2
            | Minicpm
            | Xverse
            | CommandR
            | Cohere2
            | Cohere2moe
            | Olmo
            | Arctic
            | Deepseek
            | Deepseek2
            | Deepseek2ocr
            | Deepseek32
            | Deepseek4
            | Deepseek41
            | MuseGlimmer
            | Xing40
            | MuseSpark
            | Plm
            | Chatglm
            | Granite
            | GraniteMoe
            | GraniteHybrid
            | GraniteSwitch
            | GraniteSwa
            | Chameleon
            | Bailingmoe
            | NeoBert
            | Smollm3
            | Arcee
            | Ernie45
            | Ernie45Moe
            | Mistral3
            | Eagle3
            | Mistral4
            | MistralLarge4
            | LlamaEmbed
            | Maincoder
            | GlmDsa
            | Dots3note
            | Nanbeige
            | Pockettts
            | Moshi
            | Zonos3
            | Limite
            | TinyVoca
    ) {
        return Ok(RopeType::Normal);
    }
    if matches!(arch, Bailingmoe3) {
        return Ok(if hparams.use_mrope() {
            RopeType::Mrope
        } else {
            RopeType::Normal
        });
    }
    if arch == HyV4 {
        return Ok(RopeType::Normal);
    }
    if arch == HyV4Flash {
        return Ok(if hparams.use_mrope() {
            RopeType::Imrope
        } else {
            RopeType::Normal
        });
    }
    if matches!(
        arch,
        Falcon
            | FalconH1
            | Grok
            | Dbrx
            | Bert
            | JinaBertV3
            | ModernBert
            | NomicBert
            | NomicBertMoe
            | Eurobert
            | Stablelm
            | Bitnet
            | Qwen
            | Qwen2
            | Dream
            | Qwen2moe
            | Qwen3
            | VllmDecision2
            | Flint
            | Qwen3moe
            | LladaMoe
            | Rnd1
            | Ricursive
            | Olmo2
            | Olmoe
            | Phi2
            | Phi3
            | Phimoe
            | Plamo
            | Plamo2
            | Plamo3
            | Gemma
            | Gemma2
            | Gemma3
            | Gemma3n
            | Gemma4
            | MaionCoder
            | Gemma4Assistant
            | GemmaEmbedding
            | GemmaEmbedding2
            | Starcoder2
            | Openelm
            | Gptneox
            | Codeshell
            | Orion
            | Nemotron
            | Exaone
            | Exaone4
            | ExaoneMoe
            | Minicpm3
            | Bailingmoe2
            | Dots1
            | HunyuanMoe
            | Jais2
            | OpenaiMoe
            | HunyuanDense
            | HyV3
            | Lfm2
            | Lfm2moe
            | Lfm3
            | Zaya1
            | Smallthinker
            | SeedOss
            | Grovemoe
            | Apertus
            | Minimax01
            | MinimaxM2
            | Cogvlm
            | PanguEmbed
            | Afmoe
            | Laguna
            | Qwen3next
            | CloverMoe
            | Mimo2
            | Step35
            | Spark25
            | Talkie
            | K2Horizon
            | Mellum
            | Maple
            | HrmText
            | Beam
            | Kolibri1
            | DomynEdge
    ) {
        return Ok(RopeType::NeoX);
    }
    if arch == Dflash {
        if hparams.rope_sections.iter().any(|&s| s != 0) {
            return Ok(RopeType::Mrope);
        }
        return Ok(if hparams.dsv4_hc_mult > 0 || hparams.n_lora_kv > 0 {
            RopeType::Normal
        } else {
            RopeType::NeoX
        });
    }
    if matches!(arch, SensenovaU1 | Qwen2vl | Paddleocr) {
        return Ok(RopeType::Mrope);
    }
    if matches!(
        arch,
        Qwen3vl | Qwen3vlmoe | Qwen35 | Qwen35moe | Clef | Qwen4exp | Qwen3tts
    ) {
        return Ok(RopeType::Imrope);
    }
    if arch == Glm4 {
        return Ok(if hparams.use_mrope() {
            RopeType::Mrope
        } else {
            RopeType::Normal
        });
    }
    if arch == Glm4Moe {
        return Ok(if hparams.use_mrope() {
            RopeType::Mrope
        } else {
            RopeType::NeoX
        });
    }
    if arch == HunyuanVl {
        return Ok(if hparams.use_mrope() {
            RopeType::Mrope
        } else {
            RopeType::NeoX
        });
    }
    Err(Error::new("unknown architecture"))
}

/// `llama_model_base::load_hparams`.
pub fn load_hparams(model: &mut LlamaModel, ml: &LlamaModelLoader) -> Result<()> {
    for kv in &ml.metadata.kv {
        if kv.is_array {
            continue;
        }
        model
            .gguf_kv
            .insert(kv.key.clone(), LlamaModelLoader::kv_to_string(kv));
    }
    model.gguf_kv_arr = ml.kept_string_arrays();
    super::tensors::load_hparams_hadamard(model, ml)?;

    if let Some(name) = ml.get_string(Kv::GeneralName, false)? {
        model.name = name;
    }
    if model.hparams.vocab_only || model.arch == Arch::Clip {
        return Ok(());
    }

    if let Some(v) = ml.get_u32(Kv::ContextLength, true)? {
        model.hparams.n_ctx_train = v;
    }
    if let Some(v) = ml.get_u32(Kv::EmbeddingLength, true)? {
        model.hparams.n_embd = v;
    }
    if let Some(v) = ml.get_u32(Kv::EmbeddingLengthOut, false)? {
        model.hparams.n_embd_out_impl = v;
    }
    if let Some(v) = ml.get_bool(Kv::AttentionCausal, false)? {
        model.hparams.causal_attn = v;
    }
    if let Some(v) = ml.get_string(Kv::PoolingType, false)? {
        model.hparams.pooling_type = pooling_type_from_string(&v).unwrap_or(PoolingType::None);
    }
    if let Some(v) = ml.get_string(Kv::ClassifierPoolingType, false)? {
        model.hparams.pooling_type_cls =
            pooling_type_from_string(&v).unwrap_or(PoolingType::Unspecified);
    }
    if let Some(v) = ml.get_bool(Kv::ClassifierFreeGuidance, false)? {
        model.hparams.classifier_free_guidance = v;
    }
    if let Some(v) = ml.get_string(Kv::ClassifierActivation, false)? {
        model.hparams.act_cls = match v.as_str() {
            "gelu" => crate::hparams::UnaryOp::Tanh,
            "silu" | "tanh" => crate::hparams::UnaryOp::Tanh,
            _ => return Err(Error::new("unsupported classifier activation")),
        };
    }
    if let Some(v) = ml.get_u32(Kv::BlockCount, true)? {
        model.hparams.n_layer_all = v;
    }
    if model.hparams.n_layer_all == 0 || model.hparams.n_layer_all as usize > LLAMA_MAX_LAYERS {
        return Err(Error::new("n_layer_all out of bounds"));
    }
    if let Some(v) = ml.get_u32(Kv::NextnPredictLayers, false)? {
        model.hparams.n_layer_nextn = v;
    }
    if model.hparams.n_layer_nextn > model.hparams.n_layer_all {
        return Err(Error::new("n_layer_nextn exceeds n_layer_all"));
    }
    if let Some(v) = ml.get_u32(Kv::ExpertCount, false)? {
        model.hparams.n_expert = v;
    }
    model.hparams.n_expert_used_arr.fill(0);
    ml.get_key_or_arr_u32(
        Kv::ExpertUsedCount,
        &mut model.hparams.n_expert_used_arr,
        model.hparams.n_layer_all,
        false,
    )?;
    if let Some(v) = ml.get_u32(Kv::ExpertGroupCount, false)? {
        model.hparams.n_expert_groups = v;
    }
    if let Some(v) = ml.get_u32(Kv::ExpertGroupUsedCount, false)? {
        model.hparams.n_group_used = v;
    }

    if matches!(model.arch, Arch::HunyuanVl | Arch::HunyuanDense) && model.hparams.n_expert <= 1 {
        model.hparams.n_expert = 0;
        model.hparams.n_expert_used_arr.fill(0);
    }

    if model.arch == Arch::WavtokenizerDec {
        if let Some(v) = ml.get_u32(Kv::FeaturesLength, true)? {
            model.hparams.n_embd = v;
        }
        if let Some(v) = ml.get_u32(Kv::EmbeddingLength, true)? {
            model.hparams.n_embd_out_impl = v;
        }
        if let Some(v) = ml.get_u32(Kv::PosnetEmbeddingLength, true)? {
            model.hparams.posnet.n_embd = v;
        }
        if let Some(v) = ml.get_u32(Kv::PosnetBlockCount, true)? {
            model.hparams.posnet.n_layer = v;
        }
        if let Some(v) = ml.get_u32(Kv::ConvnextEmbeddingLength, true)? {
            model.hparams.convnext.n_embd = v;
        }
        if let Some(v) = ml.get_u32(Kv::ConvnextBlockCount, true)? {
            model.hparams.convnext.n_layer = v;
        }
        if model.hparams.posnet.n_layer > model.hparams.n_layer_all
            || model.hparams.convnext.n_layer > model.hparams.n_layer_all
        {
            return Err(Error::new(
                "posnet/convnext layer count exceeds n_layer_all",
            ));
        }
    }

    let n_expert_used_max = model.hparams.n_expert_used_max()?;
    if model.hparams.n_expert as usize > LLAMA_MAX_EXPERTS {
        return Err(Error::new("n_expert out of bounds"));
    }
    if n_expert_used_max > model.hparams.n_expert {
        return Err(Error::new("n_expert_used exceeds n_expert"));
    }
    if model.hparams.n_expert > 0 {
        if n_expert_used_max == 0 {
            return Err(Error::new(
                "model has expert layers but no expert layers are used",
            ));
        }
        if model.hparams.n_expert_groups >= model.hparams.n_expert {
            return Err(Error::new("n_expert_groups must be smaller than n_expert"));
        }
        if model.hparams.n_expert_groups > 1 {
            if model.hparams.n_expert % model.hparams.n_expert_groups != 0
                || model.hparams.n_group_used == 0
                || model.hparams.n_group_used >= model.hparams.n_expert_groups
            {
                return Err(Error::new("invalid expert group configuration"));
            }
        }
    } else if n_expert_used_max != 0 || model.hparams.n_expert_groups != 0 {
        return Err(Error::new("invalid expert configuration for a dense model"));
    }

    model.hparams.n_head_arr.fill(0);
    model.hparams.n_head_kv_arr.fill(0);
    model.hparams.n_ff_arr.fill(0);
    model.hparams.n_ff_exp_arr.fill(0);
    model.hparams.rope_sections.fill(0);
    model.hparams.rope_pattern.fill(1);
    model.hparams.is_swa_impl.fill(0);
    let recr = arch_is_recurrent(model.arch) as u32;
    model.hparams.is_recr_impl.fill(recr);
    model.hparams.is_indexer_full_impl.fill(0);
    for (i, slot) in model.hparams.kv_source_impl.iter_mut().enumerate() {
        *slot = i as u32;
    }
    model.hparams.xielu_alpha_n.fill(0.0);
    model.hparams.xielu_alpha_p.fill(0.0);
    model.hparams.xielu_beta.fill(0.0);
    model.hparams.xielu_eps.fill(0.0);
    model.hparams.swiglu_clamp_exp.fill(0.0);
    model.hparams.swiglu_clamp_shexp.fill(0.0);
    model.hparams.swiglu_clamp.fill(0.0);

    ml.get_key_or_arr_u32(
        Kv::FeedForwardLength,
        &mut model.hparams.n_ff_arr,
        model.hparams.n_layer_all,
        false,
    )?;
    ml.get_key_or_arr_u32(
        Kv::AttentionHeadCount,
        &mut model.hparams.n_head_arr,
        model.hparams.n_layer_all,
        false,
    )?;
    model.hparams.deepstack_mapping_arr.fill(-1);
    model.hparams.n_head_kv_arr = model.hparams.n_head_arr;
    ml.get_key_or_arr_u32(
        Kv::AttentionHeadCountKv,
        &mut model.hparams.n_head_kv_arr,
        model.hparams.n_layer_all,
        false,
    )?;

    if let Some(v) = ml.get_bool(Kv::RopeScalingFinetuned, false)? {
        model.hparams.rope_finetuned = v;
    }
    model.hparams.n_ctx_orig_yarn = model.hparams.n_ctx_train;
    if let Some(v) = ml.get_u32(Kv::RopeScalingOrigCtxLen, false)? {
        model.hparams.n_ctx_orig_yarn = v;
    }

    model.hparams.rope_freq_base_train = 10000.0;
    if let Some(v) = ml.get_f32(Kv::RopeFreqBase, false)? {
        model.hparams.rope_freq_base_train = v;
    }
    let rope_scaling = ml
        .get_string(Kv::RopeScalingType, false)?
        .unwrap_or_else(|| "linear".to_owned());
    model.hparams.rope_scaling_type_train = rope_scaling_type_from_string(&rope_scaling)
        .ok_or_else(|| Error::new(format!("unknown rope scaling type: {}", rope_scaling)))?;
    let order = ml
        .get_string(Kv::RopeSectionOrder, false)?
        .unwrap_or_else(|| "tyxz".to_owned());
    model.hparams.rope_section_order = rope_section_order_from_string(&order)
        .ok_or_else(|| Error::new(format!("unknown rope section order: {}", order)))?;

    let mut ropescale = ml.get_f32(Kv::RopeScalingFactor, false)?.unwrap_or(0.0);
    if ropescale == 0.0 {
        ropescale = ml.get_f32(Kv::RopeScaleLinear, false)?.unwrap_or(0.0);
    }
    model.hparams.rope_freq_scale_train = if ropescale == 0.0 {
        1.0
    } else {
        1.0 / ropescale
    };

    if let Some(v) = ml.get_f32(Kv::RopeScalingAttnFactor, false)? {
        model.hparams.rope_attn_factor = v;
    }
    if let Some(v) = ml.get_f32(Kv::RopeScalingAlpha, false)? {
        model.hparams.rope_scaling_alpha = v;
    }
    if let Some(v) = ml.get_f32(Kv::RopeScalingYarnExtFactor, false)? {
        model.hparams.yarn_ext_factor = v;
    }
    if let Some(v) = ml.get_f32(Kv::RopeScalingYarnAttnFactor, false)? {
        model.hparams.yarn_attn_factor = v;
    }
    if let Some(v) = ml.get_f32(Kv::RopeScalingYarnBetaFast, false)? {
        model.hparams.yarn_beta_fast = v;
    }
    if let Some(v) = ml.get_f32(Kv::RopeScalingYarnBetaSlow, false)? {
        model.hparams.yarn_beta_slow = v;
    }
    if let Some(v) = ml.get_f32(Kv::RopeScalingYarnLogMul, false)? {
        model.hparams.rope_yarn_log_mul = v;
    }

    let n_head0 = model.hparams.n_head(0)?;
    if n_head0 > 0 {
        model.hparams.n_embd_head_k_full = model.hparams.n_embd / n_head0;
        if let Some(v) = ml.get_u32(Kv::AttentionKeyLength, false)? {
            model.hparams.n_embd_head_k_full = v;
        }
        model.hparams.n_embd_head_v_full = model.hparams.n_embd / n_head0;
        if let Some(v) = ml.get_u32(Kv::AttentionValueLength, false)? {
            model.hparams.n_embd_head_v_full = v;
        }
        model.hparams.n_rot_full = model.hparams.n_embd_head_k_full;
        if let Some(v) = ml.get_u32(Kv::RopeDimensionCount, false)? {
            model.hparams.n_rot_full = v;
        }
        if matches!(
            model.arch,
            Arch::Llama | Arch::Deci | Arch::Falcon | Arch::LlamaEmbed
        ) && model.hparams.n_rot_full != model.hparams.n_embd_head_k_full
        {
            return Err(Error::new(format!(
                "invalid n_rot: {}, expected {}",
                model.hparams.n_rot_full, model.hparams.n_embd_head_k_full
            )));
        }
    } else {
        model.hparams.n_rot_full = 0;
        model.hparams.n_embd_head_k_full = 0;
        model.hparams.n_embd_head_v_full = 0;
    }

    model.hparams.n_embd_head_k_swa = model.hparams.n_embd_head_k_full;
    model.hparams.n_embd_head_v_swa = model.hparams.n_embd_head_v_full;
    if let Some(v) = ml.get_u32(Kv::AttentionKeyLengthSwa, false)? {
        model.hparams.n_embd_head_k_swa = v;
    }
    if let Some(v) = ml.get_u32(Kv::AttentionValueLengthSwa, false)? {
        model.hparams.n_embd_head_v_swa = v;
    }
    model.hparams.n_rot_swa = model.hparams.n_rot_full;
    if let Some(v) = ml.get_u32(Kv::RopeDimensionCountSwa, false)? {
        model.hparams.n_rot_swa = v;
    }

    model.classifier_labels = ml.get_arr_string(Kv::ClassifierOutputLabels, false)?;
    if !model.classifier_labels.is_empty() {
        model.hparams.n_cls_out = model.classifier_labels.len() as u32;
    }
    model.decision_labels = ml.get_arr_string(Kv::DecisionLabels, false)?;

    load_arch_hparams(model, ml)?;

    model.impl_state.n_bytes = ml.n_bytes;
    model.impl_state.desc_str = format!(
        "{} {} {}",
        model.arch_name(),
        model.type_name(),
        super::tensors::ftype_name(ml.ftype)
    );
    model.impl_state.ftype = ml.ftype;
    if model.hparams.f_max_alibi_bias > 0.0 {
        model.hparams.use_alibi = true;
    }
    model.hparams.rope_type = rope_type_for_arch(model.arch, &model.hparams)?.to_hparams();
    if model.hparams.rope_section_order != RopeSectionOrder::Tyxz
        && model.hparams.n_pos_per_embd() != 4
    {
        return Err(Error::new(format!(
            "rope section order {} requires M-RoPE",
            rope_section_order_name(model.hparams.rope_section_order)
        )));
    }
    Ok(())
}

/// Per-architecture hparams read by the generic loader. Arch modules can add
/// their own keys by calling this first and then extending `model.hparams`.
pub fn load_arch_hparams(model: &mut LlamaModel, ml: &LlamaModelLoader) -> Result<()> {
    if let Some(v) = ml.get_u32(Kv::ExpertFeedForwardLength, false)? {
        for il in 0..model.hparams.n_layer_all as usize {
            model.hparams.n_ff_exp_arr[il] = v;
        }
    } else {
        ml.get_key_or_arr_u32(
            Kv::ExpertFeedForwardLength,
            &mut model.hparams.n_ff_exp_arr,
            model.hparams.n_layer_all,
            false,
        )?;
    }
    if let Some(v) = ml.get_u32(Kv::ExpertSharedFeedForwardLength, false)? {
        model.hparams.n_ff_shexp = v;
    }
    if let Some(v) = ml.get_u32(Kv::ExpertLatentLength, false)? {
        model.hparams.n_expert_latent = v;
    }
    if let Some(v) = ml.get_u32(Kv::ExpertSharedCount, false)? {
        model.hparams.n_expert_shared = v;
    }
    if let Some(v) = ml.get_string(Kv::ExpertGatingFunc, false)? {
        model.hparams.expert_gating_func = match v.as_str() {
            "softmax" => ExpertGatingFuncType::Softmax as u32,
            "sigmoid" => ExpertGatingFuncType::Sigmoid as u32,
            "sqrtsoftplus" => ExpertGatingFuncType::SqrtSoftplus as u32,
            _ => ExpertGatingFuncType::None as u32,
        };
    }
    if let Some(v) = ml.get_f32(Kv::ExpertWeightsScale, false)? {
        model.hparams.expert_weights_scale = v;
    }
    if let Some(v) = ml.get_bool(Kv::ExpertWeightsNorm, false)? {
        model.hparams.expert_weights_norm = v;
    }
    if let Some(v) = ml.get_f32(Kv::ExpertGroupScale, false)? {
        model.hparams.expert_group_scale = v;
    }
    if let Some(v) = ml.get_u32(Kv::ExpertsPerGroup, false)? {
        model.hparams.n_group_experts = v;
    }
    if let Some(v) = ml.get_u32(Kv::MoeEveryNLayers, false)? {
        model.hparams.moe_every_n_layers = v;
    }
    if let Some(v) = ml.get_u32(Kv::MoeLatentSize, false)? {
        model.hparams.moe_latent_size = v;
    }
    if let Some(v) = ml.get_u32(Kv::LeadingDenseBlockCount, false)? {
        model.hparams.n_layer_dense_lead = v;
    }
    if let Some(v) = ml.get_u32(Kv::AttnResBlockSize, false)? {
        model.hparams.attn_res_block_size = v;
    }
    if let Some(v) = ml.get_f32(Kv::LogitScale, false)? {
        model.hparams.f_logit_scale = v;
    }
    if let Some(v) = ml.get_f32(Kv::AttnLogitSoftcapping, false)? {
        model.hparams.f_attn_logit_softcapping = v;
    }
    if let Some(v) = ml.get_f32(Kv::RouterLogitSoftcapping, false)? {
        model.hparams.f_router_logit_softcapping = v;
    }
    if let Some(v) = ml.get_f32(Kv::FinalLogitSoftcapping, false)? {
        model.hparams.f_final_logit_softcapping = v;
    }
    if let Some(v) = ml.get_i32(Kv::DecoderStartTokenId, false)? {
        model.hparams.dec_start_token_id = v;
    }
    if let Some(v) = ml.get_u32(Kv::DecoderBlockCount, false)? {
        model.hparams.dec_n_layer = v;
    }
    if let Some(v) = ml.get_u32(Kv::NumDeepstackLayers, false)? {
        model.hparams.n_deepstack_layers = v;
    }
    ml.get_key_or_arr_i32(
        Kv::DeepstackMapping,
        &mut model.hparams.deepstack_mapping_arr,
        model.hparams.n_layer_all,
        false,
    )?;
    if let Some(v) = ml.get_f32(Kv::ActivationSituBeta, false)? {
        model.hparams.situ_beta = v;
    }
    if let Some(v) = ml.get_f32(Kv::ActivationSituLinearBeta, false)? {
        model.hparams.situ_linear_beta = v;
    }
    if let Some(v) = ml.get_bool(Kv::NormBeforeResidual, false)? {
        model.hparams.norm_before_residual = v;
    }
    if let Some(v) = ml.get_bool(Kv::NormBeforeFc, false)? {
        model.hparams.norm_before_fc = v;
    }
    if let Some(v) = ml.get_u32(Kv::ShortconvLCache, false)? {
        model.hparams.n_shortconv_l_cache = v;
    }
    if let Some(v) = ml.get_u32(Kv::SsmInnerSize, false)? {
        model.hparams.ssm_d_inner = v;
    }
    if let Some(v) = ml.get_u32(Kv::SsmConvKernel, false)? {
        model.hparams.ssm_d_conv = v;
    }
    if let Some(v) = ml.get_u32(Kv::SsmStateSize, false)? {
        model.hparams.ssm_d_state = v;
    }
    if let Some(v) = ml.get_u32(Kv::SsmTimeStepRank, false)? {
        model.hparams.ssm_dt_rank = v;
    }
    if let Some(v) = ml.get_u32(Kv::SsmGroupCount, false)? {
        model.hparams.ssm_n_group = v;
    }
    if let Some(v) = ml.get_bool(Kv::SsmDtBCRms, false)? {
        model.hparams.ssm_dt_b_c_rms = v;
    }
    if let Some(v) = ml.get_f32(Kv::SsmNormEps, false)? {
        model.hparams.ssm_norm_eps = v;
    }
    if let Some(v) = ml.get_u32(Kv::KdaHeadDim, false)? {
        model.hparams.n_embd_head_kda = v;
    }
    if let Some(v) = ml.get_bool(Kv::KdaSafeGate, false)? {
        model.hparams.kda_safe_gate = v;
    }
    if let Some(v) = ml.get_f32(Kv::KdaGateLowerBound, false)? {
        model.hparams.kda_gate_lower_bound = v;
    }
    if let Some(v) = ml.get_u32(Kv::WkvHeadSize, false)? {
        model.hparams.wkv_head_size = v;
    }
    if let Some(v) = ml.get_u32(Kv::DflashBlockSize, false)? {
        model.hparams.dflash_block_size = v;
    }
    if let Some(v) = ml.get_u32(Kv::DflashConvKernelSize, false)? {
        model.hparams.dflash_conv_kernel_size = v;
    }
    if let Some(v) = ml.get_u32(Kv::DflashConvGroupSize, false)? {
        model.hparams.dflash_conv_group_size = v;
    }
    if let Some(v) = ml.get_u32(Kv::DflashSelectorRank, false)? {
        model.hparams.dflash_selector_rank = v;
    }
    if let Some(v) = ml.get_u32(Kv::DflashSelectorTopK, false)? {
        model.hparams.dflash_selector_top_k = v;
    }
    if let Some(v) = ml.get_u32(Kv::LilicorrHiddenSize, false)? {
        model.hparams.lilicorr_hidden_size = v;
    }
    if let Some(v) = ml.get_u32(Kv::LilicorrNumHeads, false)? {
        model.hparams.lilicorr_num_heads = v;
    }
    if let Some(v) = ml.get_u32(Kv::LilicorrNumLayers, false)? {
        model.hparams.lilicorr_layers = v;
    }
    if let Some(v) = ml.get_f32(Kv::LilicorrVectorEps, false)? {
        model.hparams.lilicorr_vector_eps = v;
    }
    if let Some(v) = ml.get_f32(Kv::LilicorrLogitScale, false)? {
        model.hparams.lilicorr_logit_scale = v;
    }
    if let Some(v) = ml.get_u32(Kv::Dense2FeatIn, false)? {
        model.hparams.dense_2_feat_in = v;
    }
    if let Some(v) = ml.get_u32(Kv::Dense2FeatOut, false)? {
        model.hparams.dense_2_feat_out = v;
    }
    if let Some(v) = ml.get_u32(Kv::Dense3FeatIn, false)? {
        model.hparams.dense_3_feat_in = v;
    }
    if let Some(v) = ml.get_u32(Kv::Dense3FeatOut, false)? {
        model.hparams.dense_3_feat_out = v;
    }
    Ok(())
}

/// `llama_model_base::load_vocab`.
pub fn load_vocab(model: &mut LlamaModel, ml: &LlamaModelLoader) -> Result<()> {
    let metadata = ml.vocab_metadata();
    model.vocab.load(&metadata)
}

/// `llama_model_base::load_swa_pattern`.
pub fn load_swa_pattern(
    model: &mut LlamaModel,
    ml: &LlamaModelLoader,
    n_pattern: u32,
    dense_first: bool,
) -> Result<()> {
    let arr = ml.get_arr_u32(Kv::AttentionSlidingWindowPattern, false)?;
    if arr.len() >= model.hparams.n_layer_all as usize {
        for (slot, &v) in model.hparams.is_swa_impl.iter_mut().zip(arr.iter()) {
            *slot = v;
        }
        return Ok(());
    }
    let n = ml
        .get_u32(Kv::AttentionSlidingWindowPattern, false)?
        .unwrap_or(n_pattern);
    model.hparams.set_swa_pattern(n, dense_first);
    Ok(())
}

/// `llama_model::print_info`, as a rendered string.
pub fn print_info(model: &LlamaModel) -> String {
    let mut out = String::new();
    let mut line = |key: &str, value: &str| {
        out.push_str(&format!("{}: {:<22}= {}\n", "print_info", key, value));
    };
    line("arch", &model.arch_name());
    line("vocab_only", &(model.hparams.vocab_only as u8).to_string());
    line("no_alloc", &(model.no_alloc() as u8).to_string());
    if !model.hparams.vocab_only {
        line("n_ctx_train", &model.hparams.n_ctx_train.to_string());
        line("n_embd_inp", &model.hparams.n_embd_inp().to_string());
        line("n_embd", &model.hparams.n_embd.to_string());
        line("n_embd_out", &model.hparams.n_embd_out().to_string());
        line("n_layer", &model.hparams.n_layer().to_string());
        line("n_layer_all", &model.hparams.n_layer_all.to_string());
        line(
            "n_head",
            &print_layer_values(model, |il| model.hparams.n_head(il).unwrap_or(0) as i64),
        );
        line(
            "n_head_kv",
            &print_layer_values(model, |il| model.hparams.n_head_kv(il).unwrap_or(0) as i64),
        );
        line("n_rot", &model.hparams.n_rot_full.to_string());
        line("n_swa", &model.hparams.n_swa.to_string());
        line(
            "is_swa_any",
            &(model.hparams.is_swa_any() as u8).to_string(),
        );
        line(
            "n_embd_head_k",
            &model.hparams.n_embd_head_k_full.to_string(),
        );
        line(
            "n_embd_head_v",
            &model.hparams.n_embd_head_v_full.to_string(),
        );
        line(
            "n_gqa",
            &print_layer_values(model, |il| model.hparams.n_gqa(il).unwrap_or(0) as i64),
        );
        line(
            "n_embd_k_gqa",
            &print_layer_values(model, |il| {
                model.hparams.n_embd_k_gqa(il).unwrap_or(0) as i64
            }),
        );
        line(
            "n_embd_v_gqa",
            &print_layer_values(model, |il| {
                model.hparams.n_embd_v_gqa(il).unwrap_or(0) as i64
            }),
        );
        line("f_norm_eps", &format!("{:.1e}", model.hparams.f_norm_eps));
        line(
            "f_norm_rms_eps",
            &format!("{:.1e}", model.hparams.f_norm_rms_eps),
        );
        line("f_clamp_kqv", &format!("{:.1e}", model.hparams.f_clamp_kqv));
        line(
            "f_max_alibi_bias",
            &format!("{:.1e}", model.hparams.f_max_alibi_bias),
        );
        line(
            "f_logit_scale",
            &format!("{:.1e}", model.hparams.f_logit_scale),
        );
        line(
            "f_attn_scale",
            &format!("{:.1e}", model.hparams.f_attention_scale),
        );
        line(
            "f_attn_value_scale",
            &format!("{:.4}", model.hparams.f_attn_value_scale),
        );
        line(
            "n_ff",
            &print_layer_values(model, |il| model.hparams.n_ff(il).unwrap_or(0) as i64),
        );
        line("n_expert", &model.hparams.n_expert.to_string());
        line(
            "n_expert_used",
            &model.hparams.n_expert_used_max().unwrap_or(0).to_string(),
        );
        line(
            "causal attn",
            &(model.hparams.causal_attn as u8).to_string(),
        );
        line(
            "pooling type",
            &(model.hparams.pooling_type as u32).to_string(),
        );
        line("rope type", &(model.hparams.rope_type as u32).to_string());
        line(
            "rope scaling",
            rope_scaling_type_name(model.hparams.rope_scaling_type_train),
        );
        line(
            "freq_base_train",
            &format!("{:.1}", model.hparams.rope_freq_base_train),
        );
        line(
            "freq_scale_train",
            &format!("{}", model.hparams.rope_freq_scale_train),
        );
        line(
            "n_ctx_orig_yarn",
            &model.hparams.n_ctx_orig_yarn.to_string(),
        );
        line(
            "rope_finetuned",
            if model.hparams.rope_finetuned {
                "yes"
            } else {
                "unknown"
            },
        );
        if model.hparams.rope_sections.iter().any(|&s| s != 0) {
            line(
                "mrope sections",
                &format!("{:?}", model.hparams.rope_sections),
            );
        }
        if model.hparams.rope_section_order != RopeSectionOrder::Tyxz {
            line(
                "rope section order",
                rope_section_order_name(model.hparams.rope_section_order),
            );
        }
        line("model type", &model.type_name());
        let n = model.n_elements();
        let (scale, suffix) = if n >= 1_000_000_000_000 {
            (1e-12, "T")
        } else if n >= 1_000_000_000 {
            (1e-9, "B")
        } else if n >= 1_000_000 {
            (1e-6, "M")
        } else {
            (1e-3, "K")
        };
        line(
            "model params",
            &format!("{:.2} {}", n as f64 * scale, suffix),
        );
        line("general.name", &model.name);
        for (i, label) in model.classifier_labels.iter().enumerate() {
            line(&format!("cls_label[{}]", i), label);
        }
    }
    out.push_str(&crate::vocab::LlamaVocab::print_info(&model.vocab));
    out
}

fn print_layer_values(model: &LlamaModel, f: impl Fn(u32) -> i64) -> String {
    let mut values = Vec::new();
    for il in 0..model.hparams.n_layer_all {
        values.push(f(il));
    }
    if values.iter().any(|&v| v != values[0]) {
        format!(
            "[{}]",
            values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        values.first().map(|v| v.to_string()).unwrap_or_default()
    }
}

/// `llama_model_get_vocab` equivalent, provided as a method below for clarity.
impl LlamaModel {
    pub fn get_vocab(&self) -> &crate::vocab::LlamaVocab {
        &self.vocab
    }

    pub fn vocab_type(&self) -> VocabType {
        self.vocab.get_type()
    }

    pub fn vocab_pre_type(&self) -> VocabPreType {
        self.vocab.get_pre_type()
    }
}

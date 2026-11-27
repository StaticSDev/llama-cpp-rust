//! Tensor loading, per-tensor scales, precision policy and Hadamard folding.

use super::buffers::assign_devices;
use super::model::LlamaModel;
use super::types::*;
use super::LlamaModelLoader;
use crate::arch::{Arch, LlmTn, LlmTnImpl, Tensor as ArchTensor};
use crate::core::{Error, Result};
use crate::gguf::GgmlType;
use crate::hparams::LLAMA_MAX_LAYERS;
use std::collections::BTreeMap;

pub fn ftype_name(ftype: Ftype) -> &'static str {
    super::types::ftype_name(ftype)
}

/// `llama_model_base::create_tensor` with the loader passed explicitly.
pub fn create_tensor(
    model: &LlamaModel,
    ml: &mut LlamaModelLoader,
    tn: &LlmTnImpl,
    ne: &[i64],
    flags: u32,
) -> Result<Option<Tensor>> {
    let buft = if tn.bid >= 0 {
        let list = model
            .impl_state
            .dev_layer
            .get(tn.bid as usize)
            .map(|d| &d.bufts)
            .unwrap_or(&model.impl_state.dev_output.bufts);
        list.first().map(|e| e.buft.clone())
    } else {
        model
            .impl_state
            .dev_input
            .bufts
            .first()
            .map(|e| e.buft.clone())
    };
    ml.create_tensor(tn, ne, flags, buft.as_ref())
}

/// Helper for merged `gate_up_exps` with fallback to separate gate/up.
pub fn create_tensor_gate_up_exps(
    model: &LlamaModel,
    ml: &mut LlamaModelLoader,
    bid: usize,
    n_embd: i64,
    n_ff: i64,
    n_expert: i64,
    flags: u32,
) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
    if flags & tensor_flags::SKIP != 0 {
        let skip = tensor_flags::NOT_REQUIRED | tensor_flags::SKIP;
        create_tensor(
            model,
            ml,
            &LlmTnImpl::new(
                model.arch,
                ArchTensor::FfnGateUpExps,
                Some("weight"),
                bid as i32,
                -1,
            ),
            &[n_embd, n_ff * 2, n_expert],
            skip | tensor_flags::SKIP_IF_VIRTUAL,
        )?;
        create_tensor(
            model,
            ml,
            &LlmTnImpl::new(
                model.arch,
                ArchTensor::FfnGateExps,
                Some("weight"),
                bid as i32,
                -1,
            ),
            &[n_embd, n_ff, n_expert],
            skip,
        )?;
        create_tensor(
            model,
            ml,
            &LlmTnImpl::new(
                model.arch,
                ArchTensor::FfnUpExps,
                Some("weight"),
                bid as i32,
                -1,
            ),
            &[n_embd, n_ff, n_expert],
            skip,
        )?;
        return Ok((None, None, None));
    }
    let fused = create_tensor(
        model,
        ml,
        &LlmTnImpl::new(
            model.arch,
            ArchTensor::FfnGateUpExps,
            Some("weight"),
            bid as i32,
            -1,
        ),
        &[n_embd, n_ff * 2, n_expert],
        tensor_flags::NOT_REQUIRED,
    )?;
    if fused.is_some() {
        return Ok((fused, None, None));
    }
    let gate = create_tensor(
        model,
        ml,
        &LlmTnImpl::new(
            model.arch,
            ArchTensor::FfnGateExps,
            Some("weight"),
            bid as i32,
            -1,
        ),
        &[n_embd, n_ff, n_expert],
        flags,
    )?;
    let up = create_tensor(
        model,
        ml,
        &LlmTnImpl::new(
            model.arch,
            ArchTensor::FfnUpExps,
            Some("weight"),
            bid as i32,
            -1,
        ),
        &[n_embd, n_ff, n_expert],
        flags,
    )?;
    Ok((fused, gate, up))
}

/// Helper for merged QKV with fallback to separate Q/K/V and their biases.
pub fn create_tensor_qkv(
    model: &LlamaModel,
    ml: &mut LlamaModelLoader,
    bid: usize,
    n_embd: i64,
    n_embd_q: i64,
    n_embd_k: i64,
    n_embd_v: i64,
    flags: u32,
) -> Result<QkvTensors> {
    let n_qkv = n_embd_q + n_embd_k + n_embd_v;
    if flags & tensor_flags::SKIP != 0 {
        let skip = tensor_flags::NOT_REQUIRED | tensor_flags::SKIP;
        create_tensor(
            model,
            ml,
            &LlmTnImpl::new(
                model.arch,
                ArchTensor::AttnQkv,
                Some("weight"),
                bid as i32,
                -1,
            ),
            &[n_embd, n_qkv],
            skip | tensor_flags::SKIP_IF_VIRTUAL,
        )?;
        create_tensor(
            model,
            ml,
            &LlmTnImpl::new(
                model.arch,
                ArchTensor::AttnQkv,
                Some("bias"),
                bid as i32,
                -1,
            ),
            &[n_qkv],
            skip | tensor_flags::SKIP_IF_VIRTUAL,
        )?;
        for kind in [ArchTensor::AttnQ, ArchTensor::AttnK, ArchTensor::AttnV] {
            create_tensor(
                model,
                ml,
                &LlmTnImpl::new(model.arch, kind, Some("weight"), bid as i32, -1),
                &[n_embd, 1],
                skip,
            )?;
            create_tensor(
                model,
                ml,
                &LlmTnImpl::new(model.arch, kind, Some("bias"), bid as i32, -1),
                &[1],
                skip,
            )?;
        }
        return Ok(QkvTensors::default());
    }
    let wqkv = create_tensor(
        model,
        ml,
        &LlmTnImpl::new(
            model.arch,
            ArchTensor::AttnQkv,
            Some("weight"),
            bid as i32,
            -1,
        ),
        &[n_embd, n_qkv],
        tensor_flags::NOT_REQUIRED | tensor_flags::SKIP_IF_VIRTUAL,
    )?;
    if let Some(ref fused) = wqkv {
        let wqkv_b = create_tensor(
            model,
            ml,
            &LlmTnImpl::new(
                model.arch,
                ArchTensor::AttnQkv,
                Some("bias"),
                bid as i32,
                -1,
            ),
            &[n_qkv],
            tensor_flags::NOT_REQUIRED | tensor_flags::SKIP_IF_VIRTUAL,
        )?;
        if wqkv_b.is_none() {
            let q_b = create_tensor(
                model,
                ml,
                &LlmTnImpl::new(model.arch, ArchTensor::AttnQ, Some("bias"), bid as i32, -1),
                &[n_embd_q],
                tensor_flags::NOT_REQUIRED,
            )?;
            let k_b = create_tensor(
                model,
                ml,
                &LlmTnImpl::new(model.arch, ArchTensor::AttnK, Some("bias"), bid as i32, -1),
                &[n_embd_k],
                tensor_flags::NOT_REQUIRED,
            )?;
            let v_b = create_tensor(
                model,
                ml,
                &LlmTnImpl::new(model.arch, ArchTensor::AttnV, Some("bias"), bid as i32, -1),
                &[n_embd_v],
                tensor_flags::NOT_REQUIRED,
            )?;
            return Ok(QkvTensors {
                wqkv: Some(fused.clone()),
                wqkv_b: None,
                q_b,
                k_b,
                v_b,
                ..Default::default()
            });
        }
        return Ok(QkvTensors {
            wqkv: Some(fused.clone()),
            wqkv_b,
            ..Default::default()
        });
    }
    let wq = create_tensor(
        model,
        ml,
        &LlmTnImpl::new(
            model.arch,
            ArchTensor::AttnQ,
            Some("weight"),
            bid as i32,
            -1,
        ),
        &[n_embd, n_embd_q],
        flags,
    )?;
    let wk = create_tensor(
        model,
        ml,
        &LlmTnImpl::new(
            model.arch,
            ArchTensor::AttnK,
            Some("weight"),
            bid as i32,
            -1,
        ),
        &[n_embd, n_embd_k],
        flags,
    )?;
    let wv = create_tensor(
        model,
        ml,
        &LlmTnImpl::new(
            model.arch,
            ArchTensor::AttnV,
            Some("weight"),
            bid as i32,
            -1,
        ),
        &[n_embd, n_embd_v],
        flags,
    )?;
    let q_b = create_tensor(
        model,
        ml,
        &LlmTnImpl::new(model.arch, ArchTensor::AttnQ, Some("bias"), bid as i32, -1),
        &[n_embd_q],
        tensor_flags::NOT_REQUIRED,
    )?;
    let k_b = create_tensor(
        model,
        ml,
        &LlmTnImpl::new(model.arch, ArchTensor::AttnK, Some("bias"), bid as i32, -1),
        &[n_embd_k],
        tensor_flags::NOT_REQUIRED,
    )?;
    let v_b = create_tensor(
        model,
        ml,
        &LlmTnImpl::new(model.arch, ArchTensor::AttnV, Some("bias"), bid as i32, -1),
        &[n_embd_v],
        tensor_flags::NOT_REQUIRED,
    )?;
    Ok(QkvTensors {
        wq,
        wk,
        wv,
        q_b,
        k_b,
        v_b,
        ..Default::default()
    })
}

#[derive(Clone, Debug, Default)]
pub struct QkvTensors {
    pub wqkv: Option<Tensor>,
    pub wqkv_b: Option<Tensor>,
    pub wq: Option<Tensor>,
    pub wk: Option<Tensor>,
    pub wv: Option<Tensor>,
    pub q_b: Option<Tensor>,
    pub k_b: Option<Tensor>,
    pub v_b: Option<Tensor>,
}

/// `nextn_flags`: decide whether trunk or MTP tensors are required.
pub fn nextn_flags(model: &LlamaModel, ml: &LlamaModelLoader) -> NextnFlags {
    let mut res = NextnFlags::default();
    if model.hparams.n_layer_nextn > 0 {
        let trunk = LlmTn::new(model.arch)
            .name(ArchTensor::AttnNorm, Some("weight"), 0, -1)
            .unwrap_or_default();
        if !ml.has_file_tensor(&trunk) {
            res.trunk = tensor_flags::NOT_REQUIRED;
        }
        let mtp = LlmTn::new(model.arch)
            .name(
                ArchTensor::NextnEhProj,
                Some("weight"),
                model.hparams.n_layer() as i32,
                -1,
            )
            .unwrap_or_default();
        if !ml.has_file_tensor(&mtp) {
            res.mtp = tensor_flags::NOT_REQUIRED;
        }
    }
    if !ml.load_mtp {
        res.mtp |= tensor_flags::SKIP;
    }
    res
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NextnFlags {
    pub trunk: u32,
    pub mtp: u32,
}

// ---------------------------------------------------------------------------
// Hadamard metadata
// ---------------------------------------------------------------------------

fn is_foldable_weight(name: &str) -> bool {
    if name == "output.weight" {
        return true;
    }
    if !name.starts_with("blk.") {
        return false;
    }
    let rest = &name[4..];
    let Some(end) = rest.find('.') else {
        return false;
    };
    if end == 0 || end + 1 >= rest.len() {
        return false;
    }
    let kinds = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_qkv",
        "attn_gate",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
        "ffn_gate_up_exps",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
        "ssm_out",
    ];
    kinds
        .iter()
        .any(|k| rest[end + 1..].starts_with(&format!("{}.weight", k)))
}

/// `llama_model_base::load_hparams_hadamard`.
pub fn load_hparams_hadamard(model: &mut LlamaModel, ml: &LlamaModelLoader) -> Result<()> {
    use crate::arch::Kv;
    if let Some(v) = ml.get_bool(Kv::PrismHadamardTiedOutput, false)? {
        model.hdmd.tied_output = v;
    }
    let version = ml.get_u32(Kv::PrismHadamardVersion, false)?;
    let Some(version) = version else {
        return Ok(());
    };
    if version != 1 && version != 2 {
        return Err(Error::new(format!(
            "unsupported prism.hadamard.version: {}",
            version
        )));
    }
    if model.split_mode() == SplitMode::Tensor {
        return Err(Error::new(
            "LLAMA_SPLIT_MODE_TENSOR is not supported for models with Hadamard-folded weights (prism.hadamard), use -sm layer",
        ));
    }
    if (version == 2) != model.hdmd.tied_output {
        return Err(Error::new(
            "prism.hadamard version 2 requires tied_output=true; version 1 forbids it",
        ));
    }
    if model.hdmd.tied_output && ml.has_file_tensor("output.weight") {
        return Err(Error::new(
            "prism.hadamard.tied_output requires output.weight to be absent",
        ));
    }
    let block_size = ml
        .get_u32(Kv::PrismHadamardBlockSize, true)?
        .ok_or_else(|| Error::new("missing prism.hadamard.block_size"))?;
    let transform = ml
        .get_string(Kv::PrismHadamardTransform, true)?
        .unwrap_or_default();
    let axis = ml
        .get_string(Kv::PrismHadamardAxis, true)?
        .unwrap_or_default();
    let sign_mode = ml
        .get_string(Kv::PrismHadamardSignMode, true)?
        .unwrap_or_default();
    let weight_names = ml.get_arr_string(Kv::PrismHadamardWeightNames, true)?;
    if block_size == 0 || (block_size & (block_size - 1)) != 0 {
        return Err(Error::new(format!(
            "invalid prism.hadamard.block_size: {}",
            block_size
        )));
    }
    if transform != "normalized-sylvester-walsh-hadamard" {
        return Err(Error::new(format!(
            "unsupported prism.hadamard.transform: {}",
            transform
        )));
    }
    if axis != "input-last-dimension" {
        return Err(Error::new(format!(
            "unsupported prism.hadamard.axis: {}",
            axis
        )));
    }
    if sign_mode != "identity" && sign_mode != "explicit" {
        return Err(Error::new(format!(
            "unsupported prism.hadamard.sign_mode: {}",
            sign_mode
        )));
    }
    if weight_names.is_empty() {
        return Err(Error::new("prism.hadamard.weight_names is empty"));
    }
    if sign_mode == "explicit" {
        let widths = ml.get_arr_i32(Kv::PrismHadamardSignWidths, true)?;
        let values = ml.get_arr_i32(Kv::PrismHadamardSignValues, true)?;
        if widths.is_empty() {
            return Err(Error::new(
                "prism.hadamard.sign_mode is explicit but sign_widths is empty",
            ));
        }
        let mut off = 0usize;
        for &width in &widths {
            if width <= 0 || width as u32 % block_size != 0 || off + width as usize > values.len() {
                return Err(Error::new(format!(
                    "invalid prism.hadamard sign width: {}",
                    width
                )));
            }
            if model.hdmd.sign_data.contains_key(&width) {
                return Err(Error::new(format!(
                    "duplicate prism.hadamard sign width: {}",
                    width
                )));
            }
            let vec = values[off..off + width as usize].to_vec();
            if vec.iter().any(|&v| v != 1 && v != -1) {
                return Err(Error::new("prism.hadamard sign values must be +/-1"));
            }
            model.hdmd.sign_data.insert(width, vec);
            off += width as usize;
        }
        if off != values.len() {
            return Err(Error::new("prism.hadamard.sign_values length mismatch"));
        }
    }
    if let Some(v) = ml.get_bool(Kv::PrismHadamardGdnVGrouped, false)? {
        model.hdmd.gdn_v_grouped = v;
    }
    if !matches!(
        model.arch,
        Arch::Llama
            | Arch::Qwen3
            | Arch::Qwen3moe
            | Arch::Qwen35
            | Arch::Qwen35moe
            | Arch::Qwen3next
    ) {
        return Err(Error::new(format!(
            "prism.hadamard: arch '{}' is not verified to apply the activation transform to all folded weights",
            model.arch.name()
        )));
    }
    for name in weight_names {
        if !is_foldable_weight(&name) {
            return Err(Error::new(format!(
                "prism.hadamard: weight '{}' is not on a verified Hadamard-aware matmul path",
                name
            )));
        }
        if model
            .hdmd
            .weight_blocks
            .insert(name.clone(), block_size)
            .is_some()
        {
            return Err(Error::new(format!(
                "duplicate prism.hadamard weight: {}",
                name
            )));
        }
    }
    for name in ml.get_arr_string(Kv::PrismHadamardInverseWeightNames, false)? {
        if name != "token_embd.weight" {
            return Err(Error::new(format!(
                "prism.hadamard: weight '{}' is not a verified inverse-after-lookup table",
                name
            )));
        }
        if model.hdmd.weight_blocks.contains_key(&name)
            || model
                .hdmd
                .inverse_blocks
                .insert(name.clone(), block_size)
                .is_some()
        {
            return Err(Error::new(format!(
                "duplicate prism.hadamard inverse weight: {}",
                name
            )));
        }
    }
    if model.hdmd.tied_output {
        if version != 2 {
            return Err(Error::new("prism.hadamard.tied_output requires version 2"));
        }
        let Some(size) = model.hdmd.inverse_blocks.get("token_embd.weight").copied() else {
            return Err(Error::new(
                "prism.hadamard.tied_output requires a latent token embedding",
            ));
        };
        model
            .hdmd
            .weight_blocks
            .insert("token_embd.weight".to_owned(), size);
    } else if model.hdmd.inverse_blocks.contains_key("token_embd.weight")
        && !ml.has_file_tensor("output.weight")
    {
        return Err(Error::new(
            "a tied Hadamard output requires version 2 and tied_output=true",
        ));
    }
    Ok(())
}

/// Normalized Walsh-Hadamard matrix used by the folding pass.
fn hadamard_matrix(block_size: u32) -> Vec<f32> {
    let n = block_size as usize;
    let scale = 1.0 / (block_size as f32).sqrt();
    let mut data = vec![0.0f32; n * n];
    for row in 0..n {
        for col in 0..n {
            let mut parity = row & col;
            parity ^= parity >> 16;
            parity ^= parity >> 8;
            parity ^= parity >> 4;
            parity ^= parity >> 2;
            parity ^= parity >> 1;
            data[row * n + col] = if parity & 1 != 0 { -scale } else { scale };
        }
    }
    data
}

/// `llama_model_base::load_tensors_hadamard`.
pub fn load_tensors_hadamard(model: &mut LlamaModel) -> Result<()> {
    if model.hdmd.weight_blocks.is_empty() && model.hdmd.inverse_blocks.is_empty() {
        return Ok(());
    }
    let mut rotations: Vec<(u32, String, String)> = Vec::new(); // (block, buft, tensor name)
    let mut signs: BTreeMapPair = BTreeMapPair::default();
    let mut preferred_buft: Option<String> = None;
    let groups = [
        (model.hdmd.weight_blocks.clone(), true),
        (model.hdmd.inverse_blocks.clone(), false),
    ];
    for (blocks, is_forward) in groups {
        for (weight_name, block_size) in blocks {
            let mut weight = model.get_tensor(&weight_name).cloned();
            if model.hdmd.tied_output && weight_name == "token_embd.weight" {
                let source = if is_forward {
                    &model.output
                } else {
                    &model.tok_embd
                };
                if let Some(t) = source {
                    if t.name == "token_embd.weight" {
                        weight = Some(t.clone());
                    }
                }
                if weight.is_none() {
                    return Err(Error::new(
                        "prism.hadamard.tied_output is not bound to the token embedding",
                    ));
                }
            }
            let Some(weight) = weight else {
                return Err(Error::new(format!(
                    "prism.hadamard weight not found: {}",
                    weight_name
                )));
            };
            if weight.ne[0] as u32 % block_size != 0 {
                return Err(Error::new(format!(
                    "prism.hadamard block size {} does not divide input dimension {} for {}",
                    block_size, weight.ne[0], weight_name
                )));
            }
            let mut buft = weight.buffer.clone().unwrap_or_else(|| "CPU".to_owned());
            if buft.contains("EXTRA") {
                buft = "CPU".to_owned();
            }
            if is_forward {
                preferred_buft = Some(buft.clone());
            } else if let Some(ref pref) = preferred_buft {
                buft = pref.clone();
            }
            let rot_name = if rotations
                .iter()
                .any(|(bs, b, _)| *bs == block_size && *b == buft)
            {
                rotations
                    .iter()
                    .find(|(bs, b, _)| *bs == block_size && *b == buft)
                    .unwrap()
                    .2
                    .clone()
            } else {
                let name = format!("prism.hadamard.{}", block_size);
                let mut tensor = Tensor::new(
                    name.clone(),
                    &[block_size as i64, block_size as i64],
                    GgmlType::F32,
                );
                let data = hadamard_matrix(block_size);
                tensor.data = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                tensor.buffer = Some(buft.clone());
                model.tensors_by_name.push((name.clone(), tensor));
                rotations.push((block_size, buft.clone(), name.clone()));
                name
            };
            let mut sign_name = None;
            if !model.hdmd.sign_data.is_empty() {
                let width = weight.ne[0] as i32;
                let Some(values) = model.hdmd.sign_data.get(&width) else {
                    return Err(Error::new(format!(
                        "prism.hadamard has no sign vector for width {} ({})",
                        width, weight_name
                    )));
                };
                let key = format!("{}:{}", width, buft);
                let name = if let Some(existing) = signs.entries.iter().find(|(k, _)| *k == key) {
                    existing.1.clone()
                } else {
                    let name = format!("prism.hadamard.signs.{}", width);
                    let mut tensor = Tensor::new(name.clone(), &[width as i64], GgmlType::F32);
                    tensor.data = values
                        .iter()
                        .flat_map(|&v| (v as f32).to_le_bytes())
                        .collect();
                    tensor.buffer = Some(buft.clone());
                    model.tensors_by_name.push((name.clone(), tensor));
                    signs.entries.push((key, name.clone()));
                    name
                };
                sign_name = Some(name);
            }
            let mut transform = HadamardTransform {
                rot: rot_name,
                signs: sign_name,
                ..Default::default()
            };
            if model.hdmd.gdn_v_grouped
                && weight_name.contains(".ssm_out.")
                && !transform.set_gdn_v_perm(
                    weight.ne[0],
                    model.hparams.ssm_dt_rank as i64,
                    model.hparams.ssm_n_group as i64,
                )
            {
                return Err(Error::new(format!(
                    "prism.hadamard: bad GDN head geometry for {}",
                    weight_name
                )));
            }
            let target = if is_forward {
                &mut model.hdmd.rot
            } else {
                &mut model.hdmd.inv
            };
            target.insert(weight.name.clone(), transform);
        }
    }
    Ok(())
}

#[derive(Default)]
struct BTreeMapPair {
    entries: Vec<(String, String)>,
}

/// `llama_prec_policy::load` plus `load_prec_policy`.
pub fn load_prec_policy(model: &mut LlamaModel, ml: &LlamaModelLoader) -> Result<()> {
    use crate::arch::Kv;
    use crate::r#impl::LogLevel;
    let needs_f32 = model.hparams.expert_gating_func
        == crate::hparams::ExpertGatingFuncType::SqrtSoftplus as u32
        || model.hparams.expert_router_f32;
    if needs_f32 {
        for layer in &model.layers {
            model
                .prec_policy
                .set_acc(layer.ffn_gate_inp.as_ref(), Precision::F32);
        }
    }
    if std::env::var("LLAMA_MOE_AMAX_SKIP").is_ok() {
        crate::r#impl::log(
            LogLevel::Warn,
            "LLAMA_MOE_AMAX_SKIP is set - the MoE amax rescaling is disabled\n",
        );
        for layer in &model.layers {
            model
                .prec_policy
                .set_src1(layer.ffn_down_exps.as_ref(), Precision::Bf16);
        }
    }
    let names = ml.get_arr_string(Kv::GeneralTensorExtraName, false)?;
    if names.is_empty() {
        return Ok(());
    }
    let values = ml.get_arr_bool(Kv::GeneralTensorExtraPrecA4, true)?;
    if values.len() != names.len() {
        return Err(Error::new(format!(
            "{} tensor/value length mismatch ({} vs {})",
            Kv::GeneralTensorExtraPrecA4.format(model.arch),
            names.len(),
            values.len()
        )));
    }
    for (name, &wants_high_prec) in names.iter().zip(values.iter()) {
        if !wants_high_prec {
            let tensor_name = model.get_tensor(name).map(|t| t.name.clone());
            if let Some(tensor_name) = tensor_name {
                model
                    .prec_policy
                    .prec_src1
                    .insert(tensor_name, Precision::Q8);
            }
        }
    }
    Ok(())
}

/// `llama_model_base::load_arch_tensors`. The generic path loads the tensors
/// shared by most architectures. Model-specific subclasses can add theirs by
/// calling this function first.
pub fn load_arch_tensors(model: &mut LlamaModel, ml: &mut LlamaModelLoader) -> Result<()> {
    let tn = LlmTn::new(model.arch);
    let req = tensor_flags::NOT_REQUIRED;
    let model_tensors = [
        (ArchTensor::TokenEmbd, "token_embd"),
        (ArchTensor::TokenTypes, "token_types"),
        (ArchTensor::PosEmbd, "position_embd"),
        (ArchTensor::OutputNorm, "output_norm"),
        (ArchTensor::Output, "output"),
        (ArchTensor::EncOutputNorm, "output_norm_enc"),
        (ArchTensor::Cls, "cls"),
        (ArchTensor::ClsOut, "cls_out"),
    ];
    for (kind, field) in model_tensors {
        let name = tn.name(kind, Some("weight"), -1, -1)?;
        let tensor = create_tensor(
            model,
            ml,
            &LlmTnImpl::new(model.arch, kind, Some("weight"), -1, -1),
            &[],
            req,
        )?;
        set_model_tensor(model, field, tensor);
        let _ = name;
    }
    for (kind, field) in [
        (ArchTensor::Output, "output_b"),
        (ArchTensor::OutputNorm, "output_norm_b"),
        (ArchTensor::TokenEmbd, "tok_norm"),
    ] {
        let suffix = if field.ends_with("_b") {
            "bias"
        } else {
            "weight"
        };
        let tensor = create_tensor(
            model,
            ml,
            &LlmTnImpl::new(model.arch, kind, Some(suffix), -1, -1),
            &[],
            req,
        )?;
        set_model_tensor(model, field, tensor);
    }

    let n_layer = model.hparams.n_layer_all as usize;
    let layer_kinds = [
        (ArchTensor::AttnNorm, "attn_norm"),
        (ArchTensor::AttnNorm2, "attn_norm_2"),
        (ArchTensor::AttnQNorm, "attn_q_norm"),
        (ArchTensor::AttnKNorm, "attn_k_norm"),
        (ArchTensor::FfnNorm, "ffn_norm"),
        (ArchTensor::AttnQ, "wq"),
        (ArchTensor::AttnK, "wk"),
        (ArchTensor::AttnV, "wv"),
        (ArchTensor::AttnQkv, "wqkv"),
        (ArchTensor::AttnOut, "wo"),
        (ArchTensor::FfnGate, "ffn_gate"),
        (ArchTensor::FfnUp, "ffn_up"),
        (ArchTensor::FfnDown, "ffn_down"),
        (ArchTensor::FfnGateInp, "ffn_gate_inp"),
        (ArchTensor::SsmIn, "ssm_in"),
        (ArchTensor::SsmOut, "ssm_out"),
    ];
    for il in 0..n_layer.min(LLAMA_MAX_LAYERS) {
        for (kind, field) in layer_kinds {
            let tensor = create_tensor(
                model,
                ml,
                &LlmTnImpl::new(model.arch, kind, Some("weight"), il as i32, -1),
                &[],
                req,
            )?;
            set_layer_tensor(model, il, field, tensor);
        }
    }
    Ok(())
}

fn set_model_tensor(model: &mut LlamaModel, field: &str, tensor: Option<Tensor>) {
    match field {
        "token_embd" => model.tok_embd = tensor,
        "token_types" => model.type_embd = tensor,
        "position_embd" => model.pos_embd = tensor,
        "output_norm" => model.output_norm = tensor,
        "output" => model.output = tensor,
        "output_b" => model.output_b = tensor,
        "output_norm_enc" => model.output_norm_enc = tensor,
        "output_norm_b" => model.output_norm_b = tensor,
        "tok_norm" => model.tok_norm = tensor,
        "cls" => model.cls = tensor,
        "cls_out" => model.cls_out = tensor,
        _ => {}
    }
}

fn set_layer_tensor(model: &mut LlamaModel, il: usize, field: &str, tensor: Option<Tensor>) {
    if tensor.is_none() {
        return;
    }
    if let Some(layer) = model.layers.get_mut(il) {
        match field {
            "attn_norm" => layer.attn_norm = tensor,
            "attn_norm_2" => layer.attn_norm_2 = tensor,
            "attn_q_norm" => layer.attn_q_norm = tensor,
            "attn_k_norm" => layer.attn_k_norm = tensor,
            "ffn_norm" => layer.ffn_norm = tensor,
            "wq" => layer.wq = tensor,
            "wk" => layer.wk = tensor,
            "wv" => layer.wv = tensor,
            "wqkv" => layer.wqkv = tensor,
            "wo" => layer.wo = tensor,
            "ffn_gate" => layer.ffn_gate = tensor,
            "ffn_up" => layer.ffn_up = tensor,
            "ffn_down" => layer.ffn_down = tensor,
            "ffn_gate_inp" => layer.ffn_gate_inp = tensor,
            "ssm_in" => layer.ssm_in = tensor,
            "ssm_out" => layer.ssm_out = tensor,
            _ => {}
        }
    }
}

/// `llama_model_base::load_tensors`: buffer setup, generic scale pass, data
/// loading and post passes. Returns false when the progress callback cancels.
pub fn load_tensors(model: &mut LlamaModel, ml: &mut LlamaModelLoader) -> Result<bool> {
    let imp = assign_devices(model)?;
    model.impl_state = imp;
    if model.params.load_mode == LoadMode::Auto && !ml.use_mmap {
        if model.devices.iter().any(|d| !d.mmap_support) {
            ml.use_mmap = false;
        }
    }
    if ml.lazy == LazyMode::Auto && !ml.use_mmap && model.devices.iter().any(|d| !d.mmap_support) {
        ml.lazy = LazyMode::Off;
    }

    let n_layer = model.hparams.n_layer_all as usize;
    let n_expert = model.hparams.n_expert as i64;
    if n_expert > 0 && model.hparams.n_expert_used_max()? == 0 {
        return Err(Error::new(
            "model has expert layers but no expert layers are used",
        ));
    }
    model.resize_layers();
    load_arch_tensors(model, ml)?;

    let tn = LlmTn::new(model.arch);
    let nr = tensor_flags::NOT_REQUIRED;
    for il in 0..n_layer {
        let layer = match model.layers.get(il) {
            Some(layer) => layer.clone(),
            None => continue,
        };
        if model.hparams.has_kv(il as u32)? {
            let key = create_tensor(
                model,
                ml,
                &LlmTnImpl::new(
                    model.arch,
                    ArchTensor::AttnK,
                    Some("k_scale"),
                    il as i32,
                    -1,
                ),
                &[1],
                nr,
            )?;
            let value = create_tensor(
                model,
                ml,
                &LlmTnImpl::new(
                    model.arch,
                    ArchTensor::AttnV,
                    Some("v_scale"),
                    il as i32,
                    -1,
                ),
                &[1],
                nr,
            )?;
            if let Some(ref t) = key {
                if t.ty != GgmlType::F32 {
                    return Err(Error::new(format!(
                        "KV cache scales for layer {} must be F32",
                        il
                    )));
                }
            }
            if let Some(ref t) = value {
                if t.ty != GgmlType::F32 {
                    return Err(Error::new(format!(
                        "KV cache scales for layer {} must be F32",
                        il
                    )));
                }
            }
            if let Some(l) = model.layers.get_mut(il) {
                l.k_cache_s = key;
                l.v_cache_s = value;
            }
        }
        for (kind, present, field) in [
            (ArchTensor::AttnQ, layer.wq.is_some(), "wq_s"),
            (ArchTensor::AttnK, layer.wk.is_some(), "wk_s"),
            (ArchTensor::AttnV, layer.wv.is_some(), "wv_s"),
            (ArchTensor::AttnOut, layer.wo.is_some(), "wo_s"),
            (ArchTensor::AttnQkv, layer.wqkv.is_some(), "wqkv_s"),
            (
                ArchTensor::AttnGate,
                layer.wqkv_gate.is_some(),
                "wqkv_gate_s",
            ),
            (ArchTensor::FfnGate, layer.ffn_gate.is_some(), "ffn_gate_s"),
            (ArchTensor::FfnDown, layer.ffn_down.is_some(), "ffn_down_s"),
            (ArchTensor::FfnUp, layer.ffn_up.is_some(), "ffn_up_s"),
            (ArchTensor::SsmIn, layer.ssm_in.is_some(), "ssm_in_s"),
            (ArchTensor::SsmOut, layer.ssm_out.is_some(), "ssm_out_s"),
        ] {
            if present {
                let t = create_tensor(
                    model,
                    ml,
                    &LlmTnImpl::new(model.arch, kind, Some("scale"), il as i32, -1),
                    &[1],
                    nr,
                )?;
                set_scale_tensor(model, il, field, t);
            }
        }
        for (kind, present, field) in [
            (
                ArchTensor::FfnGateExps,
                layer.ffn_gate_exps.is_some(),
                "ffn_gate_exps_s",
            ),
            (
                ArchTensor::FfnDownExps,
                layer.ffn_down_exps.is_some(),
                "ffn_down_exps_s",
            ),
            (
                ArchTensor::FfnUpExps,
                layer.ffn_up_exps.is_some(),
                "ffn_up_exps_s",
            ),
        ] {
            if present {
                let t = create_tensor(
                    model,
                    ml,
                    &LlmTnImpl::new(model.arch, kind, Some("scale"), il as i32, -1),
                    &[n_expert],
                    nr,
                )?;
                set_scale_tensor(model, il, field, t);
            }
        }
        for (kind, present, field) in [
            (ArchTensor::AttnQ, layer.wq.is_some(), "wq_in_s"),
            (ArchTensor::AttnK, layer.wk.is_some(), "wk_in_s"),
            (ArchTensor::AttnV, layer.wv.is_some(), "wv_in_s"),
            (ArchTensor::AttnOut, layer.wo.is_some(), "wo_in_s"),
            (
                ArchTensor::FfnGate,
                layer.ffn_gate.is_some(),
                "ffn_gate_in_s",
            ),
            (
                ArchTensor::FfnDown,
                layer.ffn_down.is_some(),
                "ffn_down_in_s",
            ),
            (ArchTensor::FfnUp, layer.ffn_up.is_some(), "ffn_up_in_s"),
        ] {
            if present {
                let t = create_tensor(
                    model,
                    ml,
                    &LlmTnImpl::new(model.arch, kind, Some("input_scale"), il as i32, -1),
                    &[1],
                    nr,
                )?;
                set_scale_tensor(model, il, field, t);
            }
        }
    }
    if model.output.is_some() {
        let scale = tn.name(ArchTensor::Output, Some("scale"), -1, -1)?;
        if ml.has_file_tensor(&scale) || ml.get_weight(&scale).is_some() {
            model.output_s = create_tensor(
                model,
                ml,
                &LlmTnImpl::new(model.arch, ArchTensor::Output, Some("scale"), -1, -1),
                &[1],
                nr,
            )?;
        }
        let in_scale = tn.name(ArchTensor::Output, Some("input_scale"), -1, -1)?;
        if ml.has_file_tensor(&in_scale) || ml.get_weight(&in_scale).is_some() {
            model.output_in_s = create_tensor(
                model,
                ml,
                &LlmTnImpl::new(model.arch, ArchTensor::Output, Some("input_scale"), -1, -1),
                &[1],
                nr,
            )?;
        }
    }
    if let (Some(output), Some(embd)) = (&model.output, &model.tok_embd) {
        if output.name == embd.name && (model.output_s.is_some() || model.output_in_s.is_some()) {
            return Err(Error::new("output scales require a separate output weight"));
        }
    }
    ml.done_getting_tensors();

    if let Some(ref embd) = model.per_layer_tok_embd {
        if ml.has_file_tensor(&embd.name) {
            model.can_prefetch.push(embd.name.clone());
        }
    }

    // tensors_by_name mirrors the GGUF contexts after tensor creation.
    model.tensors_by_name = ml
        .tensors
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Map every weight to its "<name>.scale" tensor.
    model.w_scales.clear();
    let weights: Vec<(String, Tensor)> = model
        .tensors_by_name
        .iter()
        .filter(|(name, _)| name.ends_with(".weight"))
        .map(|(name, t)| (name.trim_end_matches(".weight").to_owned(), t.clone()))
        .collect();
    for (name, scale) in model.tensors_by_name.clone() {
        if !name.ends_with(".scale") {
            continue;
        }
        let base = name.trim_end_matches(".scale").to_owned();
        if let Some((_, w)) = weights.iter().find(|(n, _)| *n == base) {
            let ok =
                w.ne[3] == 1 && scale.nelements() == w.ne[2] as usize && scale.ne[0] == w.ne[2];
            if ok {
                model.w_scales.insert(w.name.clone(), name.clone());
            }
        }
    }

    load_prec_policy(model, ml)?;

    // Allocate backend buffers (stand-ins) and account for their sizes.
    let mut ctxs_bufs: Vec<(String, Vec<Buffer>)> = Vec::new();
    let mut by_buft: BTreeMap<String, usize> = BTreeMap::new();
    for (_, tensor) in &model.tensors_by_name {
        let name = tensor.buffer.clone().unwrap_or_else(|| "CPU".to_owned());
        let _buft = BufferType {
            name: name.clone(),
            device: 0,
        };
        *by_buft.entry(name).or_insert(0) += tensor.nbytes();
    }
    for (name, size) in by_buft {
        ctxs_bufs.push((
            name.clone(),
            vec![Buffer {
                buft: BufferType { name, device: 0 },
                size,
                usage: "weights".to_owned(),
                base_is_null: model.no_alloc(),
            }],
        ));
        model.impl_state.mappings.push(Buffer {
            buft: BufferType {
                name: "mmap".to_owned(),
                device: 0,
            },
            size: 0,
            usage: "mmap".to_owned(),
            base_is_null: false,
        });
    }
    model.impl_state.ctxs_bufs = ctxs_bufs;

    if model.no_alloc() {
        return Ok(true);
    }
    ml.load_all_data()?;
    if let Some(ref s) = model.per_layer_tok_embd_s {
        let values = s.rows_to_f32(0, 1)?;
        model.per_layer_tok_embd_scale = values.first().copied().unwrap_or(1.0);
    }
    load_tensors_hadamard(model)?;
    if let Some(cb) = model.params.progress_callback {
        if !cb(1.0) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn set_scale_tensor(model: &mut LlamaModel, il: usize, field: &str, tensor: Option<Tensor>) {
    if let Some(layer) = model.layers.get_mut(il) {
        match field {
            "wq_s" => layer.wq_s = tensor,
            "wk_s" => layer.wk_s = tensor,
            "wv_s" => layer.wv_s = tensor,
            "wo_s" => layer.wo_s = tensor,
            "wqkv_s" => layer.wqkv_s = tensor,
            "wqkv_gate_s" => layer.wqkv_gate_s = tensor,
            "ffn_gate_s" => layer.ffn_gate_s = tensor,
            "ffn_down_s" => layer.ffn_down_s = tensor,
            "ffn_up_s" => layer.ffn_up_s = tensor,
            "ssm_in_s" => layer.ssm_in_s = tensor,
            "ssm_out_s" => layer.ssm_out_s = tensor,
            "ffn_gate_exps_s" => layer.ffn_gate_exps_s = tensor,
            "ffn_down_exps_s" => layer.ffn_down_exps_s = tensor,
            "ffn_up_exps_s" => layer.ffn_up_exps_s = tensor,
            "wq_in_s" => layer.wq_in_s = tensor,
            "wk_in_s" => layer.wk_in_s = tensor,
            "wv_in_s" => layer.wv_in_s = tensor,
            "wo_in_s" => layer.wo_in_s = tensor,
            "ffn_gate_in_s" => layer.ffn_gate_in_s = tensor,
            "ffn_down_in_s" => layer.ffn_down_in_s = tensor,
            "ffn_up_in_s" => layer.ffn_up_in_s = tensor,
            _ => {}
        }
    }
}

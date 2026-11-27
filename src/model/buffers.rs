//! Port of the buffer-device assignment and meta split-state logic in
//! `llama-model.cpp`.
//!
//! No backend handles exist here. A `BufferType` names a device and a
//! placement policy; enough to reproduce the split decisions and memory
//! accounting without FFI.

use super::types::*;
use super::LlamaModel;
use crate::arch::{arch_is_recurrent, Arch};
use crate::core::{Error, Result};
use crate::gguf::{ggml_blck_size, GgmlType};

pub const SPLIT_AXIS_0: i32 = 0;
pub const SPLIT_AXIS_1: i32 = 1;
pub const SPLIT_AXIS_2: i32 = 2;
pub const SPLIT_AXIS_3: i32 = 3;
pub const SPLIT_AXIS_MIRRORED: i32 = 10;
pub const SPLIT_AXIS_PARTIAL: i32 = 11;
pub const SPLIT_AXIS_NONE: i32 = 98;

#[derive(Clone, Debug)]
pub struct BuftEntry {
    pub device: usize,
    pub buft: BufferType,
}

#[derive(Clone, Debug, Default)]
pub struct LayerDev {
    pub device: usize,
    pub bufts: Vec<BuftEntry>,
}

#[derive(Clone, Debug, Default)]
pub struct ModelImpl {
    pub n_elements: u64,
    pub n_bytes: usize,
    pub desc_str: String,
    pub ftype: Ftype,
    pub ctxs_bufs: Vec<(String, Vec<Buffer>)>,
    pub cpu_buft_list: Vec<BuftEntry>,
    pub gpu_buft_list: Vec<Vec<BuftEntry>>,
    pub dev_input: LayerDev,
    pub dev_output: LayerDev,
    pub dev_layer: Vec<LayerDev>,
    pub has_tensor_overrides: bool,
    pub tensor_split_owned: Vec<f32>,
    pub mappings: Vec<Buffer>,
}

/// `make_cpu_buft_list`: ACCEL -> host -> extra -> CPU. Stand-ins carry the
/// same order, so downstream selection sees identical candidates.
pub fn make_cpu_buft_list(devices: &[Device], params: &ModelParams) -> Result<Vec<BuftEntry>> {
    let mut list = Vec::new();
    if params.use_extra_bufts && devices.is_empty() {
        return Err(Error::new("no CPU backend found"));
    }
    for dev in devices {
        if dev.name.to_uppercase().contains("ACCEL") {
            list.push(BuftEntry {
                device: dev.backend_id,
                buft: BufferType {
                    name: "ACCEL".to_owned(),
                    device: dev.backend_id,
                },
            });
        }
    }
    if !params.no_host {
        for dev in devices {
            if dev.backend_id == 0 || dev.name.eq_ignore_ascii_case("CPU") {
                list.push(BuftEntry {
                    device: dev.backend_id,
                    buft: BufferType {
                        name: format!("{}-HOST", dev.name),
                        device: dev.backend_id,
                    },
                });
                break;
            }
        }
    }
    if params.use_extra_bufts {
        list.push(BuftEntry {
            device: 0,
            buft: BufferType {
                name: "CPU_EXTRA".to_owned(),
                device: 0,
            },
        });
    }
    if !devices
        .iter()
        .any(|d| d.backend_id == 0 || d.name.eq_ignore_ascii_case("CPU"))
    {
        return Err(Error::new("no CPU backend found"));
    }
    list.push(BuftEntry {
        device: 0,
        buft: BufferType {
            name: "CPU".to_owned(),
            device: 0,
        },
    });
    Ok(list)
}

/// `make_gpu_buft_list`: split buffer (when splitting rows), device default,
/// then extra buffer types and CPU fallbacks.
pub fn make_gpu_buft_list(
    dev: &Device,
    params: &ModelParams,
    cpu_list: &[BuftEntry],
) -> Result<Vec<BuftEntry>> {
    let mut list = Vec::new();
    if params.split_mode == SplitMode::Row {
        list.push(BuftEntry {
            device: dev.backend_id,
            buft: BufferType {
                name: format!("{}-SPLIT", dev.name),
                device: dev.backend_id,
            },
        });
    }
    list.push(BuftEntry {
        device: dev.backend_id,
        buft: BufferType {
            name: dev.name.clone(),
            device: dev.backend_id,
        },
    });
    if params.use_extra_bufts {
        list.push(BuftEntry {
            device: dev.backend_id,
            buft: BufferType {
                name: format!("{}-EXTRA", dev.name),
                device: dev.backend_id,
            },
        });
    }
    list.extend(cpu_list.iter().cloned());
    Ok(list)
}

/// Port of the `splits` normalization in `load_tensors`.
pub fn normalized_splits(params: &ModelParams, devices: &[Device]) -> Vec<f32> {
    let n = devices.len();
    let mut splits = vec![0.0f32; n];
    let all_zero =
        params.tensor_split.is_empty() || params.tensor_split.iter().take(n).all(|&x| x == 0.0);
    if all_zero {
        for (i, dev) in devices.iter().enumerate().take(n) {
            let (free, total) = (dev.free_memory, dev.total_memory);
            if free == 0 && total == 0 {
                splits[i] = 0.0;
            } else {
                splits[i] = free as f32;
            }
        }
        if splits.iter().all(|&x| x == 0.0) {
            for (i, s) in splits.iter_mut().enumerate() {
                *s = (i + 1) as f32;
            }
        }
    } else {
        for i in 0..n {
            splits[i] = params.tensor_split.get(i).copied().unwrap_or(0.0);
        }
    }
    let mut sum = 0.0;
    for s in splits.iter_mut() {
        sum += *s;
        *s = sum;
    }
    if sum != 0.0 {
        for s in splits.iter_mut() {
            *s /= sum;
        }
    } else {
        for (i, s) in splits.iter_mut().enumerate() {
            *s = (i + 1) as f32 / n as f32;
        }
    }
    splits
}

/// Assign input, repeating and output layers to devices.
pub fn assign_devices(model: &LlamaModel) -> Result<ModelImpl> {
    let mut imp = ModelImpl {
        n_elements: model.impl_state.n_elements,
        n_bytes: model.impl_state.n_bytes,
        desc_str: model.impl_state.desc_str.clone(),
        ftype: model.impl_state.ftype,
        ..Default::default()
    };
    let params = &model.params;
    let devices = &model.devices;
    imp.cpu_buft_list = make_cpu_buft_list(devices, params)?;
    imp.gpu_buft_list = Vec::with_capacity(devices.len());
    for dev in devices {
        imp.gpu_buft_list
            .push(make_gpu_buft_list(dev, params, &imp.cpu_buft_list)?);
    }
    if !devices
        .iter()
        .any(|d| d.backend_id == 0 || d.name.eq_ignore_ascii_case("CPU"))
    {
        return Err(Error::new("no CPU backend found"));
    }

    let splits = normalized_splits(params, devices);
    let n_layer = model.hparams.n_layer_all as usize;
    let n_gpu_layers = model.n_gpu_layers() as usize;
    let i_gpu_start = n_layer + 1_usize.saturating_sub(n_gpu_layers);
    let act_gpu_layers = if devices.is_empty() {
        0
    } else {
        n_gpu_layers.min(n_layer + 1)
    };

    let layer_dev = |il: usize| -> LayerDev {
        if il < i_gpu_start || il - i_gpu_start >= act_gpu_layers {
            return LayerDev {
                device: 0,
                bufts: imp.cpu_buft_list.clone(),
            };
        }
        let frac = (il - i_gpu_start) as f32 / act_gpu_layers as f32;
        let mut layer_gpu = splits
            .iter()
            .take(devices.len())
            .filter(|&&s| s < frac)
            .count();
        if layer_gpu >= devices.len() {
            layer_gpu = devices.len() - 1;
        }
        LayerDev {
            device: devices[layer_gpu].backend_id,
            bufts: imp.gpu_buft_list[layer_gpu].clone(),
        }
    };

    imp.dev_input = LayerDev {
        device: 0,
        bufts: imp.cpu_buft_list.clone(),
    };
    imp.dev_layer = (0..n_layer).map(layer_dev).collect();
    imp.dev_output = layer_dev(n_layer);
    if params.tensor_split.len() >= devices.len() {
        imp.tensor_split_owned = params.tensor_split[..devices.len()].to_vec();
    }
    imp.has_tensor_overrides = params
        .tensor_buft_overrides
        .first()
        .map(|o| !o.pattern.is_empty())
        .unwrap_or(false);
    Ok(imp)
}

/// `select_buft`: first buffer type that supports the add-op stand-in.
pub fn select_buft(list: &[BuftEntry], n_embd: u32) -> Result<BufferType> {
    for entry in list {
        if n_embd == 0 || !entry.buft.name.contains("META") {
            return Ok(entry.buft.clone());
        }
    }
    Err(Error::new("no suitable buffer type found"))
}

// ---------------------------------------------------------------------------
// meta split state
// ---------------------------------------------------------------------------

fn is_dsv4(model: &LlamaModel) -> bool {
    matches!(model.arch, Arch::Deepseek4 | Arch::Deepseek41)
        || (model.arch == Arch::Dflash && model.hparams.dsv4_hc_mult > 0)
}

fn split_qgate_arch(arch: Arch) -> bool {
    matches!(
        arch,
        Arch::Qwen3next
            | Arch::Qwen35
            | Arch::Qwen35moe
            | Arch::Clef
            | Arch::CloverMoe
            | Arch::Qwen4exp
    )
}

fn block_layer(name: &str) -> Option<u32> {
    if !name.starts_with("blk.") {
        return None;
    }
    let rest = &name[4..];
    let end = rest.find('.')?;
    rest[..end].parse::<u32>().ok()
}

fn cache_layer(name: &str) -> Option<u32> {
    if !name.starts_with("cache_") {
        return None;
    }
    let at = name.find("_l")?;
    name[at + 2..].parse::<u32>().ok()
}

/// The part after `blk.<il>.`, or the whole name for model-level tensors.
fn tensor_kind(name: &str) -> &str {
    match block_layer(name) {
        Some(il) => {
            let prefix = format!("blk.{}.", il);
            name.strip_prefix(&prefix).unwrap_or(name)
        }
        None => name,
    }
}

fn lcmm(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        return 0;
    }
    let (mut x, mut y) = (a.abs(), b.abs());
    while y != 0 {
        let r = x % y;
        x = y;
        y = r;
    }
    (a.abs() / x) * b.abs()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplitKind {
    Mirrored,
    Q,
    Kv,
    VExps,
    Qkv,
    QkvGate,
    QBias,
    KvBias,
    QkvBias,
    QkNorm,
    KvCache,
    IdxCache,
    Dsv4State,
    AttnSinks,
    AttnGateBias,
    AttnOutWeight,
    AttnOutBias,
    AttnOutA,
    AttnOutB,
    QBWeight,
    KvBWeight,
    AttnGateWeight,
    SsmDt,
    SsmA,
    SsmAlpha,
    SsmBeta,
    SsmBetaAlpha,
    SsmBetaAlphaCat,
    RCache,
    #[allow(dead_code)] // kept for fidelity with the pending-input cache patterns
    PCache,
    SpkCache,
    SpvCache,
    SpgbCache,
    SCache,
    SsmConv1d,
    SsmOutWeight,
    KdaConv1d,
    KdaGateB,
    FfnUpWeight,
    FfnUpBias,
    FfnGateWeight,
    FfnGateBias,
    FfnGateUpWeight,
    FfnDownWeight,
    FfnDownBias,
    FfnDownExpsBias,
    FfnUpShexpWeight,
    FfnGateShexpWeight,
    FfnDownShexpWeight,
    OutputWeight,
    OutputBias,
    Other,
}

fn split_kind(name: &str) -> SplitKind {
    let k = tensor_kind(name);
    let is_bias = k.ends_with(".bias");
    let base = k
        .strip_suffix(".weight")
        .or_else(|| k.strip_suffix(".bias"))
        .unwrap_or(k);
    match base {
        "attn_q" => {
            if is_bias {
                SplitKind::QBias
            } else {
                SplitKind::Q
            }
        }
        "attn_k" | "attn_v" => {
            if is_bias {
                SplitKind::KvBias
            } else {
                SplitKind::Kv
            }
        }
        "attn_v_exps" => SplitKind::VExps,
        "attn_qkv" => {
            if is_bias {
                SplitKind::QkvBias
            } else {
                SplitKind::Qkv
            }
        }
        "attn_qkv_gate" => SplitKind::QkvGate,
        "attn_q_norm" | "attn_k_norm" => SplitKind::QkNorm,
        "attn_output" => {
            if is_bias {
                SplitKind::AttnOutBias
            } else {
                SplitKind::AttnOutWeight
            }
        }
        "attn_output_a" => SplitKind::AttnOutA,
        "attn_output_b" => SplitKind::AttnOutB,
        "attn_q_b" => SplitKind::QBWeight,
        "attn_k_b" | "attn_v_b" => SplitKind::KvBWeight,
        "attn_gate" => {
            if is_bias {
                SplitKind::AttnGateBias
            } else {
                SplitKind::AttnGateWeight
            }
        }
        "attn_sinks" => SplitKind::AttnSinks,
        "ssm_dt" => SplitKind::SsmDt,
        "ssm_a" => SplitKind::SsmA,
        "ssm_alpha" => SplitKind::SsmAlpha,
        "ssm_beta" => SplitKind::SsmBeta,
        "ssm_ba" => SplitKind::SsmBetaAlpha,
        "ssm_beta_alpha" => SplitKind::SsmBetaAlphaCat,
        "ssm_conv1d" => SplitKind::SsmConv1d,
        "ssm_conv1d_q" | "ssm_conv1d_k" | "ssm_conv1d_v" => SplitKind::KdaConv1d,
        "ssm_f_b" | "ssm_g_b" => SplitKind::KdaGateB,
        "ssm_out" => SplitKind::SsmOutWeight,
        "ffn_up" => {
            if is_bias {
                SplitKind::FfnUpBias
            } else {
                SplitKind::FfnUpWeight
            }
        }
        "ffn_gate" => {
            if is_bias {
                SplitKind::FfnGateBias
            } else {
                SplitKind::FfnGateWeight
            }
        }
        "ffn_gate_up" | "ffn_gate_up_exps" => SplitKind::FfnGateUpWeight,
        "ffn_down" => {
            if is_bias {
                SplitKind::FfnDownBias
            } else {
                SplitKind::FfnDownWeight
            }
        }
        "ffn_down_exps" => {
            if is_bias {
                SplitKind::FfnDownExpsBias
            } else {
                SplitKind::FfnDownWeight
            }
        }
        "ffn_up_shexp" => SplitKind::FfnUpShexpWeight,
        "ffn_gate_shexp" => SplitKind::FfnGateShexpWeight,
        "ffn_down_shexp" => SplitKind::FfnDownShexpWeight,
        "output" => {
            if is_bias {
                SplitKind::OutputBias
            } else {
                SplitKind::OutputWeight
            }
        }
        _ => {
            if name.starts_with("cache_") {
                let n = &name[6..];
                if n.starts_with("idx_") {
                    SplitKind::IdxCache
                } else if n.starts_with("k_") || n.starts_with("v_") {
                    SplitKind::KvCache
                } else if n.starts_with("r_") {
                    SplitKind::RCache
                } else if n.starts_with("s_") {
                    SplitKind::SCache
                } else if n.starts_with("spk_") {
                    SplitKind::SpkCache
                } else if n.starts_with("spv_") {
                    SplitKind::SpvCache
                } else if n.starts_with("spgb_") {
                    SplitKind::SpgbCache
                } else if n.starts_with("ple_r_") {
                    SplitKind::Mirrored
                } else {
                    SplitKind::Other
                }
            } else if name.starts_with("dsv4_")
                && (name.contains("_state_kv_") || name.contains("_state_score_"))
            {
                SplitKind::Dsv4State
            } else if name.starts_with("cache_ple_r_") {
                SplitKind::Mirrored
            } else {
                SplitKind::Other
            }
        }
    }
}

#[derive(Clone, Debug)]
struct TensorConfig {
    axis: i32,
    axis_tensor: Option<String>,
    il: u32,
    rotation: usize,
}

fn tensor_config(model: &LlamaModel, name: &str, ne1: i64) -> Result<TensorConfig> {
    let n_devices = model.devices.len().max(1);
    let kind = split_kind(name);
    let (il, prefix, rotation) = if let Some(il) = block_layer(name) {
        let is_attn = tensor_kind(name).starts_with("attn_");
        let src = if is_attn {
            model.hparams.kv_source(il)?
        } else {
            il
        };
        let mut eff = 0usize;
        for prev in 0..src {
            if model.hparams.is_recr(prev)? == model.hparams.is_recr(il)?
                && model.hparams.is_swa(prev)? == model.hparams.is_swa(il)?
            {
                eff += 1;
            }
        }
        (il, format!("blk.{}.", il), eff % n_devices)
    } else if let Some(il) = cache_layer(name) {
        let mut eff = 0usize;
        for prev in 0..il {
            if model.hparams.is_recr(prev)? == model.hparams.is_recr(il)?
                && model.hparams.is_swa(prev)? == model.hparams.is_swa(il)?
            {
                eff += 1;
            }
        }
        (il, format!("blk.{}.", il), eff % n_devices)
    } else {
        (
            0,
            String::new(),
            (model.hparams.n_layer() as usize) % n_devices,
        )
    };

    let config = |axis: i32, suffix: &str, fallback: &str| -> TensorConfig {
        let primary = if suffix.is_empty() {
            Some(name.to_owned())
        } else {
            Some(format!("{}{}", prefix, suffix))
        };
        let axis_tensor = match primary.as_deref() {
            Some(p) if model.get_tensor(p).is_some() => primary,
            _ if !fallback.is_empty() => Some(format!("{}{}", prefix, fallback)),
            _ => primary,
        };
        TensorConfig {
            axis,
            axis_tensor,
            il,
            rotation,
        }
    };

    if matches!(model.arch, Arch::HrmText | Arch::Metis16) {
        return Ok(TensorConfig {
            axis: SPLIT_AXIS_MIRRORED,
            axis_tensor: Some(name.to_owned()),
            il: 0,
            rotation: 0,
        });
    }
    let dsv4 = is_dsv4(model);
    let glm5 = model.arch == Arch::Glm5Next;
    if dsv4 {
        if matches!(kind, SplitKind::KvCache | SplitKind::Dsv4State) {
            return Ok(config(SPLIT_AXIS_MIRRORED, "", ""));
        }
        match kind {
            SplitKind::AttnSinks => return Ok(config(SPLIT_AXIS_0, "attn_output_a.weight", "")),
            SplitKind::QBWeight => return Ok(config(SPLIT_AXIS_1, "attn_output_a.weight", "")),
            SplitKind::AttnOutA => return Ok(config(SPLIT_AXIS_2, "", "")),
            SplitKind::AttnOutB => return Ok(config(SPLIT_AXIS_0, "", "")),
            _ => {}
        }
    }
    if glm5 {
        if kind == SplitKind::KvCache {
            return Ok(config(SPLIT_AXIS_MIRRORED, "", ""));
        }
        if matches!(
            kind,
            SplitKind::QBWeight | SplitKind::KdaGateB | SplitKind::SsmBeta
        ) {
            return Ok(config(SPLIT_AXIS_1, "attn_output.weight", ""));
        }
        if matches!(kind, SplitKind::KvBWeight | SplitKind::KdaConv1d) {
            return Ok(config(SPLIT_AXIS_2, "attn_output.weight", ""));
        }
        if matches!(
            kind,
            SplitKind::SsmDt
                | SplitKind::SsmA
                | SplitKind::RCache
                | SplitKind::SCache
                | SplitKind::SpkCache
                | SplitKind::SpvCache
                | SplitKind::SpgbCache
        ) {
            return Ok(config(SPLIT_AXIS_0, "attn_output.weight", ""));
        }
    }
    if dsv4 || glm5 {
        if matches!(
            kind,
            SplitKind::FfnUpShexpWeight | SplitKind::FfnGateShexpWeight
        ) {
            return Ok(config(SPLIT_AXIS_1, "ffn_down_shexp.weight", ""));
        }
        if kind == SplitKind::FfnDownShexpWeight {
            return Ok(config(SPLIT_AXIS_0, "ffn_down_shexp.weight", ""));
        }
    }
    if matches!(kind, SplitKind::IdxCache | SplitKind::Mirrored) {
        return Ok(config(SPLIT_AXIS_MIRRORED, "", ""));
    }
    if matches!(kind, SplitKind::Q | SplitKind::Kv) {
        return Ok(config(SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight"));
    }
    if kind == SplitKind::VExps {
        return Ok(config(SPLIT_AXIS_1, "attn_output.weight", ""));
    }
    if matches!(
        kind,
        SplitKind::QBias | SplitKind::KvBias | SplitKind::QkvBias
    ) {
        return Ok(config(SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight"));
    }
    if matches!(kind, SplitKind::Qkv | SplitKind::QkvGate) {
        return Ok(config(SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight"));
    }
    if kind == SplitKind::QkNorm {
        let axis = if ne1 == 1 {
            SPLIT_AXIS_MIRRORED
        } else {
            SPLIT_AXIS_1
        };
        return Ok(config(axis, "attn_output.weight", ""));
    }
    if matches!(
        kind,
        SplitKind::KvCache | SplitKind::AttnSinks | SplitKind::AttnGateBias
    ) {
        return Ok(config(SPLIT_AXIS_0, "attn_output.weight", ""));
    }
    if kind == SplitKind::AttnOutWeight {
        return Ok(config(SPLIT_AXIS_0, "", ""));
    }
    if kind == SplitKind::AttnOutBias {
        return Ok(config(SPLIT_AXIS_MIRRORED, "", ""));
    }
    if kind == SplitKind::AttnGateWeight {
        return Ok(config(SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight"));
    }
    if matches!(kind, SplitKind::SsmDt | SplitKind::SsmA) {
        return Ok(config(SPLIT_AXIS_0, "ssm_out.weight", ""));
    }
    if matches!(
        kind,
        SplitKind::SsmAlpha
            | SplitKind::SsmBeta
            | SplitKind::SsmBetaAlpha
            | SplitKind::SsmBetaAlphaCat
    ) {
        return Ok(config(SPLIT_AXIS_1, "ssm_out.weight", ""));
    }
    if matches!(
        kind,
        SplitKind::RCache
            | SplitKind::SCache
            | SplitKind::SpkCache
            | SplitKind::SpvCache
            | SplitKind::SpgbCache
    ) {
        if matches!(model.arch, Arch::Lfm2 | Arch::Lfm2moe | Arch::Lfm3) {
            return Ok(config(SPLIT_AXIS_MIRRORED, "", ""));
        }
        return Ok(config(SPLIT_AXIS_0, "ssm_out.weight", ""));
    }
    if kind == SplitKind::SsmConv1d {
        return Ok(config(SPLIT_AXIS_1, "ssm_out.weight", ""));
    }
    if kind == SplitKind::SsmOutWeight {
        return Ok(config(SPLIT_AXIS_0, "", ""));
    }
    if matches!(
        kind,
        SplitKind::FfnUpWeight | SplitKind::FfnGateWeight | SplitKind::FfnGateUpWeight
    ) {
        return Ok(config(
            SPLIT_AXIS_1,
            "ffn_down.weight",
            "ffn_down_exps.weight",
        ));
    }
    if matches!(kind, SplitKind::FfnUpBias | SplitKind::FfnGateBias) {
        return Ok(config(
            SPLIT_AXIS_0,
            "ffn_down.weight",
            "ffn_down_exps.weight",
        ));
    }
    if kind == SplitKind::FfnDownWeight {
        return Ok(config(
            SPLIT_AXIS_0,
            "ffn_down.weight",
            "ffn_down_exps.weight",
        ));
    }
    if kind == SplitKind::FfnDownBias {
        return Ok(config(SPLIT_AXIS_MIRRORED, "", ""));
    }
    if kind == SplitKind::FfnDownExpsBias {
        return Ok(config(SPLIT_AXIS_PARTIAL, "ffn_down_exps.weight", ""));
    }
    if kind == SplitKind::OutputWeight {
        let axis = if dsv4 {
            SPLIT_AXIS_MIRRORED
        } else {
            SPLIT_AXIS_1
        };
        return Ok(config(axis, "", ""));
    }
    if kind == SplitKind::OutputBias {
        return Ok(config(SPLIT_AXIS_0, "output.weight", ""));
    }
    Ok(config(SPLIT_AXIS_MIRRORED, "", ""))
}

/// Fused tensors are split into segments. `(size, repeat)` pairs.
fn split_segments(model: &LlamaModel, name: &str, axis: usize, il: u32) -> Result<Vec<(i64, u32)>> {
    let kind = split_kind(name);
    let ne = |a: usize| -> i64 { model.get_tensor(name).map(|t| t.ne[a]).unwrap_or(0) };
    let dfl = |a: usize| -> i64 { ne(a) };

    if model.arch == Arch::Glm5Next && kind == SplitKind::RCache {
        let d_inner = model.hparams.n_head(il)? as i64 * model.hparams.n_embd_head_kda as i64;
        let conv = (model.hparams.ssm_d_conv as i64 - 1) * d_inner;
        if dfl(axis) != 3 * conv && d_inner > 0 && dfl(axis) % (3 * d_inner) == 0 {
            return Ok(vec![(d_inner, (dfl(axis) / d_inner) as u32)]);
        }
        return Ok(vec![(conv, 3)]);
    }
    if model.arch == Arch::Glm5Next
        && matches!(
            kind,
            SplitKind::SpkCache | SplitKind::SpvCache | SplitKind::SpgbCache
        )
    {
        let d_inner = model.hparams.n_head(il)? as i64 * model.hparams.n_embd_head_kda as i64;
        if d_inner > 0 && dfl(axis) % d_inner == 0 {
            return Ok(vec![(d_inner, (dfl(axis) / d_inner) as u32)]);
        }
    }
    if split_qgate_arch(model.arch) {
        if !model.hparams.is_recr(il)? && matches!(kind, SplitKind::Qkv | SplitKind::QkvBias) {
            let n_embd =
                model.hparams.n_head(il)? as i64 * model.hparams.n_embd_head_k(il)? as i64 * 2;
            let n_gqa = model.hparams.n_embd_v_gqa(il)? as i64;
            return Ok(vec![(n_embd, 1), (n_gqa, 2)]);
        }
        let head = model.hparams.ssm_d_state as i64;
        let n_k = model.hparams.ssm_n_group as i64;
        let n_v = model.hparams.ssm_dt_rank as i64;
        let key_dim = head * n_k;
        let value_dim = head * n_v;
        if model.arch == Arch::Qwen3next {
            if matches!(kind, SplitKind::Qkv | SplitKind::KdaConv1d) {
                return Ok(vec![(key_dim, 2), (value_dim, 1)]);
            }
            if kind == SplitKind::RCache {
                let conv = model.hparams.ssm_d_conv as i64 - 1;
                return Ok(vec![(key_dim * conv, 2), (value_dim * conv, 1)]);
            }
        } else if n_k > 0 {
            let ratio = n_v / n_k;
            match kind {
                SplitKind::QkvGate => return Ok(vec![(key_dim, 2 + 2 * ratio as u32)]),
                SplitKind::SsmBetaAlphaCat => return Ok(vec![(n_k, 2 * ratio as u32)]),
                SplitKind::Qkv | SplitKind::KdaConv1d => {
                    return Ok(vec![(key_dim, 2 + ratio as u32)]);
                }
                SplitKind::AttnGateWeight if model.hparams.is_recr(il)? => {
                    return Ok(vec![(key_dim, ratio as u32)]);
                }
                SplitKind::SsmOutWeight => return Ok(vec![(key_dim, ratio as u32)]),
                SplitKind::SsmDt | SplitKind::SsmA | SplitKind::SsmAlpha | SplitKind::SsmBeta => {
                    return Ok(vec![(n_k, ratio as u32)]);
                }
                SplitKind::RCache => {
                    let n_conv_ch = 2 * key_dim + value_dim;
                    if dfl(axis) != n_conv_ch * (model.hparams.ssm_d_conv as i64 - 1)
                        && n_conv_ch > 0
                        && dfl(axis) % n_conv_ch == 0
                    {
                        return Ok(vec![(
                            key_dim,
                            ((2 + ratio) * (dfl(axis) / n_conv_ch)) as u32,
                        )]);
                    }
                    return Ok(vec![(
                        key_dim * (model.hparams.ssm_d_conv as i64 - 1),
                        2 + ratio as u32,
                    )]);
                }
                SplitKind::SCache => return Ok(vec![(n_k * head * head, ratio as u32)]),
                SplitKind::SpkCache | SplitKind::SpvCache
                    if key_dim > 0 && dfl(axis) % key_dim == 0 =>
                {
                    return Ok(vec![(key_dim, (dfl(axis) / key_dim) as u32)]);
                }
                SplitKind::SpgbCache if n_k > 0 && dfl(axis) % n_k == 0 => {
                    return Ok(vec![(n_k, (dfl(axis) / n_k) as u32)]);
                }
                _ => {}
            }
        }
        if kind == SplitKind::FfnGateUpWeight {
            let n_ff = model.hparams.n_ff_exp(il)? as i64;
            return Ok(vec![(n_ff, 2)]);
        }
        return Ok(vec![(dfl(axis), 1)]);
    }

    if matches!(kind, SplitKind::Qkv | SplitKind::QkvBias) {
        let q = model.hparams.n_head(il)? as i64 * model.hparams.n_embd_head_k(il)? as i64;
        let k = model.hparams.n_embd_k_gqa(il)? as i64;
        let v = model.hparams.n_embd_v_gqa(il)? as i64;
        if k == v {
            return Ok(vec![(q, 1), (k, 2)]);
        }
        return Ok(vec![(q, 1), (k, 1), (v, 1)]);
    }
    if kind == SplitKind::FfnUpWeight {
        let n_ff = model.hparams.n_ff(il)? as i64;
        if dfl(axis) == 2 * n_ff {
            return Ok(vec![(n_ff, 2)]);
        }
    }
    if kind == SplitKind::FfnGateUpWeight {
        let n_ff = model.hparams.n_ff_exp(il)? as i64;
        return Ok(vec![(n_ff, 2)]);
    }
    Ok(vec![(dfl(axis), 1)])
}

fn split_granularity(
    model: &LlamaModel,
    name: &str,
    blck_size: i64,
    il: u32,
    segments: &[(i64, u32)],
) -> Result<Vec<i64>> {
    let kind = split_kind(name);
    let n_dev = model.devices.len().max(1) as i64;
    if model.arch == Arch::Glm5Next && model.hparams.is_recr(il)? {
        let head_dim = model.hparams.n_embd_head_kda as i64;
        let perf = lcmm(blck_size, 128);
        let qkv = lcmm(perf, head_dim);
        if matches!(
            kind,
            SplitKind::Q
                | SplitKind::Kv
                | SplitKind::KdaConv1d
                | SplitKind::KdaGateB
                | SplitKind::SsmDt
                | SplitKind::AttnOutWeight
        ) {
            return Ok(vec![qkv]);
        }
        if matches!(kind, SplitKind::SsmA | SplitKind::SsmBeta) {
            return Ok(vec![qkv / head_dim.max(1)]);
        }
        if kind == SplitKind::RCache {
            let conv =
                (model.hparams.ssm_d_conv as i64 - 1) * model.hparams.n_head(il)? as i64 * head_dim;
            let replay = model
                .get_tensor(name)
                .map(|t| t.ne[0] != 3 * conv)
                .unwrap_or(false);
            return Ok(vec![
                qkv * if replay {
                    1
                } else {
                    model.hparams.ssm_d_conv as i64 - 1
                };
                segments.len()
            ]);
        }
        if matches!(
            kind,
            SplitKind::SpkCache | SplitKind::SpvCache | SplitKind::SpgbCache
        ) {
            return Ok(vec![qkv; segments.len()]);
        }
        if kind == SplitKind::SCache {
            return Ok(vec![qkv * head_dim]);
        }
    } else if model.arch == Arch::Glm5Next {
        let head = model.hparams.n_embd_head_v_mla()? as i64;
        let gran = lcmm(lcmm(head, blck_size) / head.max(1), 2);
        match kind {
            SplitKind::QBWeight => {
                return Ok(vec![gran * model.hparams.n_embd_head_k_mla()? as i64])
            }
            SplitKind::KvBWeight => return Ok(vec![gran]),
            SplitKind::AttnOutWeight => return Ok(vec![gran * head]),
            _ => {}
        }
    } else if model.hparams.is_recr(il)? {
        let head_dim = model.hparams.ssm_d_state as i64;
        let perf = lcmm(blck_size, 128);
        let qkv = lcmm(perf, head_dim);
        if matches!(
            kind,
            SplitKind::Qkv
                | SplitKind::AttnGateWeight
                | SplitKind::KdaConv1d
                | SplitKind::SsmOutWeight
                | SplitKind::QkvGate
        ) {
            return Ok(vec![qkv; segments.len()]);
        }
        if matches!(
            kind,
            SplitKind::SsmDt
                | SplitKind::SsmA
                | SplitKind::SsmAlpha
                | SplitKind::SsmBeta
                | SplitKind::SsmBetaAlphaCat
        ) {
            return Ok(vec![qkv / head_dim.max(1); segments.len()]);
        }
        if kind == SplitKind::SsmBetaAlpha {
            return Ok(vec![2 * (qkv / head_dim.max(1)); segments.len()]);
        }
        if kind == SplitKind::RCache {
            let n_conv = 2 * model.hparams.ssm_n_group as i64 * model.hparams.ssm_d_state as i64
                + model.hparams.ssm_d_inner as i64;
            let replay = model
                .get_tensor(name)
                .map(|t| t.ne[0] != n_conv * (model.hparams.ssm_d_conv as i64 - 1))
                .unwrap_or(false);
            return Ok(vec![
                qkv * if replay {
                    1
                } else {
                    model.hparams.ssm_d_conv as i64 - 1
                };
                segments.len()
            ]);
        }
        if kind == SplitKind::SCache {
            return Ok(vec![qkv * head_dim; segments.len()]);
        }
        if matches!(kind, SplitKind::SpkCache | SplitKind::SpvCache) {
            return Ok(vec![qkv; segments.len()]);
        }
        if kind == SplitKind::SpgbCache {
            return Ok(vec![qkv / head_dim.max(1); segments.len()]);
        }
    } else {
        let n_gqa = model.hparams.n_gqa(il)? as i64;
        let n_embd_q = n_gqa * model.hparams.n_embd_head_k(il)? as i64;
        let mut perf = blck_size;
        while perf > 0 && perf < 128 && perf * n_dev < n_embd_q {
            perf *= 2;
        }
        let gran_q = lcmm(n_embd_q, perf);
        let hk = model.hparams.n_embd_head_k(il)? as i64;
        let gran_head = gran_q / hk.max(1);
        if kind == SplitKind::AttnGateBias {
            return Ok(vec![gran_head]);
        }
        if kind == SplitKind::AttnSinks {
            if is_dsv4(model) {
                return Ok(vec![
                    model.hparams.n_head(il)? as i64
                        / model.hparams.dsv4_o_group_count.max(1) as i64,
                ]);
            }
            return Ok(vec![gran_head]);
        }
        if is_dsv4(model) {
            if kind == SplitKind::QBWeight {
                let group = model.hparams.n_head(il)? as i64
                    / model.hparams.dsv4_o_group_count.max(1) as i64;
                return Ok(vec![group * hk]);
            }
            if kind == SplitKind::AttnOutA {
                return Ok(vec![1]);
            }
            if kind == SplitKind::AttnOutB {
                return Ok(vec![model.hparams.dsv4_o_lora_rank as i64]);
            }
        }
        if kind == SplitKind::Q {
            if split_qgate_arch(model.arch) {
                return Ok(vec![lcmm(2 * n_embd_q, perf)]);
            }
            return Ok(vec![gran_q]);
        }
        if kind == SplitKind::AttnOutWeight {
            return Ok(vec![gran_head * model.hparams.n_embd_head_v(il)? as i64]);
        }
        if kind == SplitKind::AttnGateWeight {
            let ne1 = model.get_tensor(name).map(|t| t.ne[1]).unwrap_or(0);
            if ne1 == model.hparams.n_head(il)? as i64 {
                return Ok(vec![gran_head]);
            }
            return Ok(vec![gran_q]);
        }
        let gran_kv = gran_q / n_gqa.max(1);
        let gran_v = (gran_kv / hk.max(1)) * model.hparams.n_embd_head_v(il)? as i64;
        if matches!(
            kind,
            SplitKind::Kv | SplitKind::VExps | SplitKind::KvBias | SplitKind::KvCache
        ) {
            let is_v = name.contains("attn_v") || name.contains("cache_v");
            return Ok(vec![if is_v { gran_v } else { gran_kv }]);
        }
        if matches!(kind, SplitKind::Qkv | SplitKind::QkvBias) {
            if split_qgate_arch(model.arch) {
                return Ok(vec![lcmm(2 * n_embd_q, perf), gran_kv]);
            }
            if segments.len() == 3 {
                return Ok(vec![gran_q, gran_kv, gran_v]);
            }
            return Ok(vec![gran_q, gran_kv]);
        }
    }

    if matches!(
        kind,
        SplitKind::FfnUpWeight
            | SplitKind::FfnUpBias
            | SplitKind::FfnGateWeight
            | SplitKind::FfnGateBias
            | SplitKind::FfnGateUpWeight
            | SplitKind::FfnDownWeight
            | SplitKind::FfnUpShexpWeight
            | SplitKind::FfnGateShexpWeight
            | SplitKind::FfnDownShexpWeight
    ) {
        return Ok(vec![lcmm(blck_size, 128); segments.len()]);
    }
    if matches!(model.arch, Arch::Step35 | Arch::Laguna) && kind == SplitKind::AttnGateWeight {
        return Ok(vec![
            model.hparams.n_head(il)? as i64 / model.hparams.n_head_kv(il)?.max(1) as i64,
        ]);
    }
    Ok(vec![1; segments.len()])
}

/// `llama_meta_device_get_split_state`. The recursion for PARTIAL tensors is
/// bounded by the number of model tensors because names are unique.
pub fn meta_device_get_split_state(
    model: &LlamaModel,
    name: &str,
    userdata: &SplitStateUserdata,
) -> Result<MetaSplitState> {
    meta_split_impl(model, name, userdata, 0)
}

fn meta_split_impl(
    model: &LlamaModel,
    name: &str,
    userdata: &SplitStateUserdata,
    depth: usize,
) -> Result<MetaSplitState> {
    if depth > model.tensors_by_name.len() {
        return Err(Error::new("meta split recursion exceeded tensor count"));
    }
    let tensor = model
        .get_tensor(name)
        .ok_or_else(|| Error::new(format!("meta split tensor not found: {}", name)))?;
    let cfg = tensor_config(model, name, tensor.ne.get(1).copied().unwrap_or(1))?;
    let mut state = MetaSplitState::default();
    state.axis = cfg.axis;
    if cfg.axis >= SPLIT_AXIS_0 && cfg.axis <= SPLIT_AXIS_3 {
        let axis = cfg.axis as usize;
        let blck = ggml_blck_size(tensor.ty).max(1) as i64;
        let rotations = cfg.rotation;
        let mut scan = vec![0.0f32; userdata.n_devices];
        let split = &model.params.tensor_split;
        for j in 0..userdata.n_devices {
            let idx = (j + rotations) % userdata.n_devices;
            scan[j] = split.get(idx).copied().unwrap_or(0.0);
            if j > 0 {
                scan[j] += scan[j - 1];
            }
        }
        let il = cfg.il;
        let segments = split_segments(model, name, axis, il)?;
        let granularity = split_granularity(model, name, blck, il, &segments)?;
        state.ne = vec![0; segments.len() * userdata.n_devices];
        state.nr = vec![1; segments.len()];
        for (is, (ne_s, nr_s)) in segments.iter().enumerate() {
            let g = granularity.get(is).copied().unwrap_or(1).max(1);
            let mut low = 0i64;
            for j in 0..userdata.n_devices.saturating_sub(1) {
                let high = if scan.last().copied().unwrap_or(0.0) == 0.0 {
                    ne_s * (j + 1) as i64 / userdata.n_devices as i64
                } else {
                    (*ne_s as f32 * scan[j] / scan.last().copied().unwrap_or(1.0)) as i64
                };
                let high = high - high % g;
                state.ne[is * userdata.n_devices + (j + rotations) % userdata.n_devices] =
                    high - low;
                low = high;
            }
            let j = userdata.n_devices.saturating_sub(1);
            state.ne[is * userdata.n_devices + (j + rotations) % userdata.n_devices] = ne_s - low;
            state.nr[is] = *nr_s;
        }
        state.n_segments = segments.len();
    } else {
        state.ne = vec![0; userdata.n_devices];
        state.nr[0] = 1;
        state.n_segments = 1;
        if cfg.axis == SPLIT_AXIS_PARTIAL {
            let source = cfg.axis_tensor.as_deref().unwrap_or(name);
            if source == name {
                return Err(Error::new(
                    "partial split tensor must reference another tensor",
                ));
            }
            let src = meta_split_impl(model, source, userdata, depth + 1)?;
            for j in 0..userdata.n_devices {
                for is in 0..src.n_segments {
                    let n = src.ne[is * userdata.n_devices + j] * src.nr[is] as i64;
                    state.ne[j] += n;
                }
            }
        }
    }
    Ok(state)
}

/// Whether a tensor can be prefetched (lazy loading marker).
pub fn can_prefetch(model: &LlamaModel, name: &str) -> bool {
    model.can_prefetch.iter().any(|n| n == name)
}

/// `arch_is_recurrent` re-export used by memory selection.
pub fn is_recurrent(arch: Arch) -> bool {
    arch_is_recurrent(arch)
}

/// `GgmlType` block size helper used by tests and other modules.
pub fn block_size(ty: GgmlType) -> i64 {
    ggml_blck_size(ty)
}

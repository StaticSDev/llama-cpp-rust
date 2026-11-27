//! Synthetic GGUF fixtures for the model port.

use super::*;
use crate::arch::Kv as ArchKv;
use crate::cparams::CParams;
use crate::gguf::{GgmlTensor, GgmlType, GgufContext, GgufKv, GgufTensorInfo};
use crate::hparams::SwaType;

fn kv_u32(key: &str, value: u32) -> GgufKv {
    GgufKv::new_scalar(key, value)
}

fn kv_f32(key: &str, value: f32) -> GgufKv {
    GgufKv::new_scalar(key, value)
}

fn kv_str(key: &str, value: &str) -> GgufKv {
    GgufKv::new_string(key, value)
}

fn kv_strs(key: &str, value: &[&str]) -> GgufKv {
    GgufKv::new_string_array(key, value)
}

fn tensor(name: &str, ne0: i64, ne1: i64) -> GgufTensorInfo {
    let ne = [ne0, ne1, 1, 1];
    let nb = [
        4,
        (4 * ne0) as usize,
        (4 * ne0 * ne1) as usize,
        (4 * ne0 * ne1) as usize,
    ];
    GgufTensorInfo {
        t: GgmlTensor {
            name: name.to_owned(),
            ne,
            type_: GgmlType::F32,
            nb,
            data: Vec::new(),
        },
        offset: 0,
    }
}

fn fixture() -> GgufContext {
    let mut kv = vec![
        kv_str("general.architecture", "llama"),
        kv_str("general.name", "tiny-llama"),
        kv_u32(&ArchKv::ContextLength.format(crate::arch::Arch::Llama), 128),
        kv_u32(&ArchKv::EmbeddingLength.format(crate::arch::Arch::Llama), 4),
        kv_u32(&ArchKv::BlockCount.format(crate::arch::Arch::Llama), 2),
        kv_u32(
            &ArchKv::AttentionHeadCount.format(crate::arch::Arch::Llama),
            2,
        ),
        kv_u32(
            &ArchKv::FeedForwardLength.format(crate::arch::Arch::Llama),
            8,
        ),
        kv_f32(
            &ArchKv::RopeFreqBase.format(crate::arch::Arch::Llama),
            10000.0,
        ),
        kv_str("tokenizer.ggml.model", "llama"),
        kv_strs("tokenizer.ggml.tokens", &["<unk>", "a", "b", "c"]),
        GgufKv::new_array("tokenizer.ggml.scores", &[0i32, 0, 0, 0]),
        GgufKv::new_array("tokenizer.ggml.token_type", &[2i32, 1, 1, 1]),
        kv_u32("tokenizer.ggml.bos_token_id", 1),
        kv_u32("tokenizer.ggml.eos_token_id", 2),
    ];
    kv.push(kv_strs("tokenizer.ggml.merges", &[]));
    let info = vec![
        tensor("token_embd.weight", 4, 4),
        tensor("output_norm.weight", 4, 1),
        tensor("output.weight", 4, 4),
        tensor("blk.0.attn_norm.weight", 4, 1),
        tensor("blk.0.attn_q.weight", 4, 4),
        tensor("blk.0.attn_k.weight", 4, 4),
        tensor("blk.0.attn_v.weight", 4, 4),
        tensor("blk.0.attn_output.weight", 4, 4),
        tensor("blk.0.ffn_norm.weight", 4, 1),
        tensor("blk.0.ffn_gate.weight", 4, 8),
        tensor("blk.0.ffn_up.weight", 4, 8),
        tensor("blk.0.ffn_down.weight", 8, 4),
        tensor("blk.1.attn_norm.weight", 4, 1),
    ];
    GgufContext {
        kv,
        info,
        data: vec![0u8; 4096],
        ..Default::default()
    }
}

fn params() -> ModelParams {
    ModelParams {
        n_gpu_layers: 0,
        ..Default::default()
    }
}

#[test]
fn loads_hparams_vocab_and_tensors() {
    let model = load_model_from_gguf(fixture(), params()).unwrap();
    assert_eq!(model.arch, crate::arch::Arch::Llama);
    assert_eq!(model.n_ctx_train(), 128);
    assert_eq!(model.n_embd(), 4);
    assert_eq!(model.n_layer(), 2);
    assert_eq!(model.n_head(), 2);
    assert_eq!(model.name, "tiny-llama");
    assert_eq!(model.vocab.n_tokens(), 4);
    assert!(model.get_tensor("token_embd.weight").is_some());
    assert!(model.get_tensor("blk.0.attn_q.weight").is_some());
    assert!(model.n_tensors() >= 13);
    assert_eq!(
        model.get_tensor("output.weight").unwrap().ne,
        vec![4, 4, 1, 1]
    );
    assert_eq!(
        model
            .memory_descriptor(&CParams::default(), false)
            .unwrap()
            .kind,
        MemoryKind::Kv
    );
}

#[test]
fn rope_params_follow_swa_and_context() {
    let model = load_model_from_gguf(fixture(), params()).unwrap();
    let mut cp = CParams::default();
    cp.rope_freq_base = 500000.0;
    cp.rope_freq_scale = 0.5;
    cp.yarn_ext_factor = 1.5;
    cp.yarn_attn_factor = 1.1;
    cp.yarn_beta_fast = 32.0;
    cp.yarn_beta_slow = 1.0;
    cp.n_ctx_orig_yarn = 128;
    let rope = model.get_rope_params(&cp, 0).unwrap();
    assert_eq!(rope.n_rot, 2);
    assert_eq!(rope.freq_base, 500000.0);
    assert_eq!(rope.freq_scale, 0.5);
    assert_eq!(rope.ext_factor, 1.5);
    let idx = model.get_rope_params_indexer(&cp, 0).unwrap();
    assert_eq!(idx.rope_type, RopeType::NeoX);
    assert_eq!(idx.freq_base, 500000.0);
}

#[test]
fn shared_getters_and_embeddings() {
    let model = load_model_from_gguf(fixture(), params()).unwrap();
    assert_eq!(model.n_embd_inp(), 4);
    assert_eq!(model.n_embd_out(), 4);
    assert_eq!(model.meta_count() >= 1, true);
    assert_eq!(model.meta_arr_str("tokenizer.tts_dict", 0), None);
    assert_eq!(model.n_params(), model.n_elements());
    assert!(model.has_decoder());
    assert!(!model.has_encoder());
    assert!(!model.is_recurrent());
    let (n, values) = model.get_tok_embd().unwrap();
    assert_eq!(n as usize, values.len());
    let (n, values) = model.get_tok_embd_rows(&[1, 2]).unwrap();
    assert_eq!(n as usize, values.len());
    assert_eq!(n, 8);
    assert_eq!(model.get_tok_embd_rows(&[-1]).unwrap().0, 0);
}

#[test]
fn meta_split_mirrors_norm_and_splits_attention() {
    let model = load_model_from_gguf(fixture(), params()).unwrap();
    let ud = SplitStateUserdata { n_devices: 1 };
    let norm = meta_device_get_split_state(&model, "blk.0.attn_norm.weight", &ud).unwrap();
    assert_eq!(norm.axis, SPLIT_AXIS_MIRRORED);
    let q = meta_device_get_split_state(&model, "blk.0.attn_q.weight", &ud).unwrap();
    assert_eq!(q.axis, SPLIT_AXIS_1);
    assert_eq!(q.n_segments, 1);
    assert_eq!(q.ne[0], 4);
}

#[test]
fn swa_pattern_and_derivations() {
    let mut model = LlamaModel::new(crate::arch::Arch::Llama, params());
    model.hparams.n_layer_all = 4;
    for il in 0..4 {
        model.hparams.n_head_arr[il] = 2;
        model.hparams.n_head_kv_arr[il] = 1;
        model.hparams.n_embd_head_k_full = 4;
        model.hparams.n_embd_head_v_full = 4;
        model.hparams.n_embd_head_k_swa = 4;
        model.hparams.n_embd_head_v_swa = 4;
        model.hparams.kv_source_impl[il] = il as u32;
        model.hparams.rope_pattern[il] = 1;
    }
    model.hparams.set_swa_pattern(2, false);
    assert!(model.hparams.is_swa(0).unwrap());
    assert!(!model.hparams.is_swa(1).unwrap());
    assert_eq!(model.hparams.n_embd_k_gqa(0).unwrap(), 4);
    assert_eq!(model.hparams.n_gqa(0).unwrap(), 2);
    assert_eq!(model.hparams.swa_type, SwaType::None);
}

#[test]
fn token_embedding_rows_reject_quantized_data() {
    let mut tensor = Tensor::new("q", &[4, 1], GgmlType::Q4_0);
    tensor.data = vec![0u8; 64];
    assert!(tensor.rows_to_f32(0, 1).is_err());
}

use std::cell::RefCell;
use std::rc::Rc;

use crate::batch::BatchExt;
use crate::core::Token;
use crate::cparams::PoolingType;
use crate::hparams::Hparams;

use super::model::{Arch, Model, VocabInfo};
use super::params::{BackendSampler, ContextParams};
use super::LlamaContext;

fn small_model() -> Model {
    let mut hp = Hparams::default();
    hp.n_ctx_train = 512;
    hp.n_embd = 8;
    hp.n_embd_inp_impl = 8;
    hp.n_embd_out_impl = 8;
    hp.n_layer_all = 1;
    hp.n_layer_nextn = 0;
    hp.causal_attn = true;
    hp.pooling_type = crate::hparams::PoolingType::None;
    hp.rope_scaling_type_train = crate::hparams::RopeScalingType::None;
    hp.rope_freq_base_train = 10000.0;
    hp.rope_freq_scale_train = 1.0;
    let mut model = Model::new(
        hp,
        VocabInfo {
            n_tokens: 8,
            n_cls_out: 1,
        },
    );
    model.arch = Arch::Other;
    model
}

fn small_params() -> ContextParams {
    let mut p = ContextParams {
        n_ctx: 512,
        n_batch: 8,
        n_ubatch: 4,
        n_seq_max: 1,
        n_outputs_max_per_seq: 1,
        ..ContextParams::default()
    };
    p.no_perf = true;
    p
}

fn token_batch(n: usize) -> BatchExt {
    let mut batch = BatchExt::new(n, 8, 8, 0, 1, None, 8, 1);
    for i in 0..n {
        let idx = batch.add_token(0);
        assert!(batch.set_token_id(idx, (i + 1) as Token));
        assert!(batch.set_token_pos(idx, &[i as i32]));
        batch.set_output(idx, i + 1 == n);
    }
    batch
}

#[test]
fn context_padding_and_rounding() {
    let mut hp = Hparams::default();
    hp.n_ctx_train = 100;
    hp.n_embd = 8;
    hp.n_layer_all = 1;
    let model = Model::new(
        hp,
        VocabInfo {
            n_tokens: 8,
            n_cls_out: 1,
        },
    );
    let p = ContextParams {
        n_ctx: 100,
        n_batch: 8,
        n_ubatch: 4,
        n_seq_max: 2,
        ..ContextParams::default()
    };
    let ctx = LlamaContext::new(&model, p).expect("context");
    assert_eq!(ctx.n_ctx(), 512);
    assert_eq!(ctx.n_ctx_seq(), 256);
    assert_eq!(ctx.n_seq_max(), 2);
}

#[test]
fn decode_writes_logits_and_resolves_rows() {
    let model = small_model();
    let mut ctx = LlamaContext::new(&model, small_params()).expect("context");
    let batch = token_batch(4);
    assert_eq!(ctx.decode(&batch), 0);
    assert_eq!(ctx.get_logits().len(), 8);
    assert_eq!(ctx.n_outputs, 1);
    assert_eq!(ctx.output_resolve_row(-1).unwrap(), 0);
    assert_eq!(ctx.output_resolve_row(3).unwrap(), 0);
    assert!(ctx.output_resolve_row(0).is_err());
}

#[test]
fn encode_without_memory_processes_one_shot() {
    let mut hp = small_model().hparams;
    hp.vocab_only = true;
    let model = Model::new(
        hp,
        VocabInfo {
            n_tokens: 8,
            n_cls_out: 1,
        },
    );
    let mut ctx = LlamaContext::new(&model, small_params()).expect("context");
    let batch = token_batch(3);
    assert_eq!(ctx.encode(&batch), 0);
    assert_eq!(ctx.get_logits().len(), 8);
}

#[test]
fn output_reorder_swaps_rows() {
    let model = small_model();
    let mut ctx = LlamaContext::new(&model, small_params()).expect("context");
    ctx.output_reserve(2);
    for i in 0..2 {
        for j in 0..8 {
            ctx.logits.set(i * 8 + j, (i * 8 + j) as f32);
        }
    }
    ctx.output_swaps.push((0, 1));
    let data = ctx.get_logits();
    assert_eq!(data[0], 8.0);
    assert_eq!(data[8], 0.0);
}

#[test]
fn backend_sampler_produces_sampled_token() {
    let model = small_model();
    let mut params = small_params();
    params.samplers.push(super::params::SamplerConfig {
        seq_id: 0,
        sampler: BackendSampler {
            name: "greedy".to_string(),
            n_chain: 2,
            can_offload: true,
            n_nodes: 4,
            n_outputs: 0,
        },
    });
    let mut ctx = LlamaContext::new(&model, params).expect("context");
    let batch = token_batch(2);
    assert_eq!(ctx.decode(&batch), 0);
    let token = ctx.get_sampled_token_ith(1);
    assert!(token >= 0 && token < 8);
    assert_eq!(ctx.get_sampled_candidates_count(1), 8);
}

#[test]
fn state_save_and_load_round_trip() {
    let model = small_model();
    let mut ctx = LlamaContext::new(&model, small_params()).expect("context");
    let batch = token_batch(2);
    assert_eq!(ctx.decode(&batch), 0);
    let size = ctx.state_get_size();
    assert!(size > 0);
    let mut buf = vec![0u8; size];
    assert_eq!(ctx.state_get_data(&mut buf), size);
    let mut other = LlamaContext::new(&model, small_params()).expect("context");
    assert_eq!(other.state_set_data(&buf), size);
    let n = other.state_get_size();
    assert_eq!(n, size);
}

#[test]
fn nextn_rows_set_and_get() {
    let model = small_model();
    let mut ctx = LlamaContext::new(&model, small_params()).expect("context");
    assert!(ctx.set_embeddings_nextn_dev(true) == false);
    ctx.set_embeddings(true);
    ctx.set_embeddings_nextn(true, false);
    assert!(ctx.set_embeddings_nextn_dev(true));
    let data = vec![1.5f32; 8];
    assert!(ctx.set_h_nextn_rows(0, 1, &data));
    let mut out = vec![0.0f32; 8];
    assert!(ctx.get_h_nextn_rows(0, 1, &mut out));
    assert_eq!(out, data);
    // Same context used through a shared handle to check borrow behavior.
    let shared = Rc::new(RefCell::new(ctx));
    assert_eq!(shared.borrow().n_ctx(), 512);
}

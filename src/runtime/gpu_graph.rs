//! GPU-resident, sequential-token forward pass for Qwen3.5 hybrid layers.
//!
//! `GpuWorkspace` owns every device buffer once. `forward_token` mirrors
//! `runtime::qwen35::forward` but keeps all intermediates on the device.

use crate::core::{Error, Result};
use crate::runtime::cuda::CudaWeights;
use crate::runtime::types::{ModelParams, Weight, Weights};
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

const PREFILL_CHUNK: usize = 16;

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::new(format!("{what} size overflow")))
}

fn to_i32(v: usize, what: &str) -> Result<i32> {
    i32::try_from(v)
        .map_err(|_| Error::new(format!("{what} too large for a CUDA kernel dimension")))
}

#[derive(Clone, Copy)]
struct Dims {
    n_embd: i32,
    n_head: i32,
    n_kv: i32,
    hd: i32,
    n_ff: i32,
    nk: i32,
    nv: i32,
    ds: i32,
    key_dim: i32,
    channels: i32,
    ksize: i32,
    seq_cap: i32,
    vocab: i32,
    n_layer: usize,
    head_bytes: usize,
    hist_bytes: usize,
}

struct CommonIdx {
    norm: *mut c_void,
    post_norm: *mut c_void,
    up: i32,
    gate: i32,
    down: i32,
}

struct RecIdx {
    qkv: i32,
    z: i32,
    beta: i32,
    alpha: i32,
    out: i32,
    dt_bias: *mut c_void,
    a: *mut c_void,
    conv: *mut c_void,
    norm: *mut c_void,
}

struct FullIdx {
    q: i32,
    k: i32,
    v: i32,
    out: i32,
    q_norm: *mut c_void,
    k_norm: *mut c_void,
}

struct LayerIdx {
    common: CommonIdx,
    rec: Option<RecIdx>,
    full: Option<FullIdx>,
}

struct RecState {
    /// Two alternating `(conv_kernel - 1) * channels` histories.
    hist: [*mut c_void; 2],
    cur: usize,
    state: *mut c_void,
}

struct FullState {
    kc: *mut c_void,
    vc: *mut c_void,
}

/// All device state for one greedy, single-sequence decode.
pub struct GpuWorkspace {
    cuda: Arc<CudaWeights>,
    p: ModelParams,
    d: Dims,
    eps: f32,
    rope_base: f32,
    n_rot: i32,
    scale: f32,
    emb_idx: i32,
    out_norm: *mut c_void,
    buffers: Vec<*mut c_void>,
    x: *mut f32,
    norm: *mut f32,
    qkv: *mut f32,
    z: *mut f32,
    beta_raw: *mut f32,
    alpha_raw: *mut f32,
    conv_out: *mut f32,
    gdn_out: *mut f32,
    heads: *mut f32,
    gates: *mut f32,
    head_k: *mut f32,
    head_v: *mut f32,
    attn_out: *mut f32,
    attn: *mut f32,
    norm2: *mut f32,
    up: *mut f32,
    gate_ffn: *mut f32,
    ffn: *mut f32,
    h: *mut f32,
    logits: *mut f32,
    argmax: *mut f32,
    // Batched prefill scratch, sized for PREFILL_CHUNK tokens and allocated once.
    batch_x: *mut f32,
    batch_norm: *mut f32,
    batch_norm2: *mut f32,
    batch_qkv: *mut f32,
    batch_z: *mut f32,
    batch_gdn: *mut f32,
    batch_attn: *mut f32,
    batch_attn_out: *mut f32,
    batch_head_k: *mut f32,
    batch_head_v: *mut f32,
    batch_up: *mut f32,
    batch_gate_ffn: *mut f32,
    batch_ffn: *mut f32,
    layers: Vec<LayerIdx>,
    rec_states: Vec<RecState>,
    full_states: Vec<FullState>,
}

impl GpuWorkspace {
    pub fn new(cuda: Arc<CudaWeights>, w: &Weights) -> Result<Self> {
        let p = &w.params;
        if p.n_layer == 0
            || p.n_embd == 0
            || p.n_head == 0
            || p.n_head_kv == 0
            || p.head_dim == 0
            || p.n_ff == 0
            || p.full_attn_interval == 0
            || p.ssm_state == 0
            || p.ssm_group == 0
            || p.ssm_dt_rank == 0
            || p.conv_kernel == 0
        {
            return Err(Error::new("invalid model parameters for GPU graph"));
        }
        let emb = w.get("token_embd.weight")?;
        let vocab = emb.out_features();
        if vocab == 0 {
            return Err(Error::new("empty token embedding"));
        }

        let n_em = p.n_embd;
        let nf = p.n_ff;
        let nhd = checked_mul(p.n_head, p.head_dim, "q head elements")?;
        let qg_dim = checked_mul(nhd, 2, "q plus gate elements")?;
        let kv_dim = checked_mul(p.n_head_kv, p.head_dim, "kv head elements")?;
        let key_dim = checked_mul(p.ssm_state, p.ssm_group, "ssm key dim")?;
        let value_dim = checked_mul(p.ssm_state, p.ssm_dt_rank, "ssm value dim")?;
        let channels = checked_mul(key_dim, 2, "ssm key dims")?
            .checked_add(value_dim)
            .ok_or_else(|| Error::new("ssm channel count overflow"))?;
        let hist_floats = if p.conv_kernel > 1 {
            checked_mul(p.conv_kernel - 1, channels, "conv history")?
        } else {
            0
        };
        let state_floats = checked_mul(
            checked_mul(p.ssm_dt_rank, p.ssm_state, "ssm state rank")?,
            p.ssm_state,
            "ssm state size",
        )?;

        // Cap KV at 4096 tokens so eight full-attention caches fit in 6 GB VRAM.
        let seq_cap = p.n_ctx.min(4096);
        if seq_cap == 0 {
            return Err(Error::new("zero context length"));
        }
        let kvc_floats = checked_mul(
            checked_mul(p.n_head_kv, seq_cap, "kv cache tokens")?,
            p.head_dim,
            "kv cache elements",
        )?;
        let head_bytes = checked_mul(nhd, std::mem::size_of::<f32>(), "head buffer")?;
        let hist_bytes = checked_mul(hist_floats, std::mem::size_of::<f32>(), "conv history")?;
        let qkv_floats = channels.max(qg_dim);

        let d = Dims {
            n_embd: to_i32(n_em, "n_embd")?,
            n_head: to_i32(p.n_head, "n_head")?,
            n_kv: to_i32(p.n_head_kv, "n_head_kv")?,
            hd: to_i32(p.head_dim, "head_dim")?,
            n_ff: to_i32(nf, "n_ff")?,
            nk: to_i32(p.ssm_group, "ssm_group")?,
            nv: to_i32(p.ssm_dt_rank, "ssm_dt_rank")?,
            ds: to_i32(p.ssm_state, "ssm_state")?,
            key_dim: to_i32(key_dim, "ssm key_dim")?,
            channels: to_i32(channels, "ssm channels")?,
            ksize: to_i32(p.conv_kernel, "conv_kernel")?,
            seq_cap: to_i32(seq_cap, "seq_cap")?,
            vocab: to_i32(vocab, "vocab_size")?,
            n_layer: p.n_layer,
            head_bytes,
            hist_bytes,
        };

        let resolve = |name: &str, in_dim: usize, out_dim: usize| -> Result<(i32, *mut c_void)> {
            let wt: &Weight = w.get(name)?;
            if wt.in_features() != in_dim || wt.out_features() != out_dim {
                return Err(Error::new(format!(
                    "{name} shape [{}, {}] expected [{in_dim}, {out_dim}]",
                    wt.in_features(),
                    wt.out_features()
                )));
            }
            let index = cuda.index_of(wt)?;
            let ptr = cuda.weight_ptr(index)?;
            Ok((index, ptr))
        };
        let resolve_small = |name: &str| -> Result<*mut c_void> {
            if !w.small.contains_key(name) {
                return Err(Error::new(format!("{name} is not a device F32 vector")));
            }
            let index = cuda.index_of(w.get(name)?)?;
            cuda.weight_ptr(index)
        };

        let (emb_idx, _) = resolve("token_embd.weight", n_em, vocab)?;
        let out_norm = resolve_small("output_norm.weight")?;

        let mut buffers: Vec<*mut c_void> = Vec::new();
        let mut alloc = |floats: usize, name: &str| -> Result<*mut f32> {
            let bytes = checked_mul(floats, std::mem::size_of::<f32>(), name)?;
            let ptr = cuda.buf(bytes)?;
            buffers.push(ptr);
            cuda.zero(ptr, bytes)?;
            Ok(ptr.cast::<f32>())
        };

        let x = alloc(n_em, "x")?;
        let norm = alloc(n_em, "norm")?;
        let qkv = alloc(qkv_floats, "qkv")?;
        let z = alloc(value_dim, "z")?;
        let beta_raw = alloc(p.ssm_dt_rank, "beta_raw")?;
        let alpha_raw = alloc(p.ssm_dt_rank, "alpha_raw")?;
        let conv_out = alloc(channels, "conv_out")?;
        let gdn_out = alloc(value_dim, "gdn_out")?;
        let heads = alloc(nhd, "heads")?;
        let gates = alloc(nhd, "gates")?;
        let head_k = alloc(kv_dim, "head_k")?;
        let head_v = alloc(kv_dim, "head_v")?;
        let attn_out = alloc(nhd, "attn_out")?;
        let attn = alloc(n_em, "attn")?;
        let norm2 = alloc(n_em, "norm2")?;
        let up = alloc(nf, "up")?;
        let gate_ffn = alloc(nf, "gate_ffn")?;
        let ffn = alloc(n_em, "ffn")?;
        let h = alloc(n_em, "h")?;
        let logits = alloc(vocab, "logits")?;
        let argmax = alloc(1, "argmax")?;

        let batch_x = alloc(checked_mul(PREFILL_CHUNK, n_em, "batch x")?, "batch x")?;
        let batch_norm = alloc(
            checked_mul(PREFILL_CHUNK, n_em, "batch norm")?,
            "batch norm",
        )?;
        let batch_norm2 = alloc(
            checked_mul(PREFILL_CHUNK, n_em, "batch norm2")?,
            "batch norm2",
        )?;
        let batch_qkv = alloc(
            checked_mul(PREFILL_CHUNK, qkv_floats, "batch qkv")?,
            "batch qkv",
        )?;
        let batch_z = alloc(checked_mul(PREFILL_CHUNK, value_dim, "batch z")?, "batch z")?;
        let batch_gdn = alloc(
            checked_mul(PREFILL_CHUNK, value_dim, "batch gdn")?,
            "batch gdn",
        )?;
        let batch_attn = alloc(
            checked_mul(PREFILL_CHUNK, n_em, "batch attn")?,
            "batch attn",
        )?;
        let batch_attn_out = alloc(
            checked_mul(PREFILL_CHUNK, nhd, "batch attn out")?,
            "batch attn out",
        )?;
        let batch_head_k = alloc(
            checked_mul(PREFILL_CHUNK, kv_dim, "batch head k")?,
            "batch head k",
        )?;
        let batch_head_v = alloc(
            checked_mul(PREFILL_CHUNK, kv_dim, "batch head v")?,
            "batch head v",
        )?;
        let batch_up = alloc(checked_mul(PREFILL_CHUNK, nf, "batch up")?, "batch up")?;
        let batch_gate_ffn = alloc(
            checked_mul(PREFILL_CHUNK, nf, "batch gate ffn")?,
            "batch gate ffn",
        )?;
        let batch_ffn = alloc(checked_mul(PREFILL_CHUNK, n_em, "batch ffn")?, "batch ffn")?;

        let mut layers = Vec::with_capacity(p.n_layer);
        let mut rec_states = Vec::new();
        let mut full_states = Vec::new();
        for il in 0..p.n_layer {
            let norm_name = format!("blk.{il}.attn_norm.weight");
            let post_name = format!("blk.{il}.post_attention_norm.weight");
            let common = CommonIdx {
                norm: resolve_small(&norm_name)?,
                post_norm: resolve_small(&post_name)?,
                up: resolve(&format!("blk.{il}.ffn_up.weight"), n_em, nf)?.0,
                gate: resolve(&format!("blk.{il}.ffn_gate.weight"), n_em, nf)?.0,
                down: resolve(&format!("blk.{il}.ffn_down.weight"), nf, n_em)?.0,
            };

            let (rec, full) = if p.is_recurrent(il) {
                let conv_name = format!("blk.{il}.ssm_conv1d.weight");
                let rec_idx = RecIdx {
                    qkv: resolve(&format!("blk.{il}.attn_qkv.weight"), n_em, channels)?.0,
                    z: resolve(&format!("blk.{il}.attn_gate.weight"), n_em, value_dim)?.0,
                    beta: resolve(&format!("blk.{il}.ssm_beta.weight"), n_em, p.ssm_dt_rank)?.0,
                    alpha: resolve(&format!("blk.{il}.ssm_alpha.weight"), n_em, p.ssm_dt_rank)?.0,
                    out: resolve(&format!("blk.{il}.ssm_out.weight"), value_dim, n_em)?.0,
                    dt_bias: resolve_small(&format!("blk.{il}.ssm_dt.bias"))?,
                    a: resolve_small(&format!("blk.{il}.ssm_a"))?,
                    conv: resolve_small(&conv_name)?,
                    norm: resolve_small(&format!("blk.{il}.ssm_norm.weight"))?,
                };
                let hist = if hist_bytes > 0 {
                    [
                        alloc(hist_floats, "conv history 0")?.cast::<c_void>(),
                        alloc(hist_floats, "conv history 1")?.cast::<c_void>(),
                    ]
                } else {
                    [ptr::null_mut(), ptr::null_mut()]
                };
                let state = alloc(state_floats, "ssm state")?.cast::<c_void>();
                rec_states.push(RecState {
                    hist,
                    cur: 0,
                    state,
                });
                (Some(rec_idx), None)
            } else {
                let full_idx = FullIdx {
                    q: resolve(&format!("blk.{il}.attn_q.weight"), n_em, qg_dim)?.0,
                    k: resolve(&format!("blk.{il}.attn_k.weight"), n_em, kv_dim)?.0,
                    v: resolve(&format!("blk.{il}.attn_v.weight"), n_em, kv_dim)?.0,
                    out: resolve(&format!("blk.{il}.attn_output.weight"), nhd, n_em)?.0,
                    q_norm: resolve_small(&format!("blk.{il}.attn_q_norm.weight"))?,
                    k_norm: resolve_small(&format!("blk.{il}.attn_k_norm.weight"))?,
                };
                let kc = alloc(kvc_floats, "k cache")?.cast::<c_void>();
                let vc = alloc(kvc_floats, "v cache")?.cast::<c_void>();
                full_states.push(FullState { kc, vc });
                (None, Some(full_idx))
            };
            layers.push(LayerIdx { common, rec, full });
        }

        let p = w.params.clone();
        let n_rot = to_i32(p.rope_dim.min(p.head_dim), "rope_dim")?;
        let scale = 1.0f32 / (p.head_dim as f32).sqrt();
        Ok(Self {
            cuda,
            p,
            d,
            eps: w.params.eps,
            rope_base: w.params.rope_base,
            n_rot,
            scale,
            emb_idx,
            out_norm,
            buffers,
            x,
            norm,
            qkv,
            z,
            beta_raw,
            alpha_raw,
            conv_out,
            gdn_out,
            heads,
            gates,
            head_k,
            head_v,
            attn_out,
            attn,
            norm2,
            up,
            gate_ffn,
            ffn,
            h,
            logits,
            argmax,
            batch_x,
            batch_norm,
            batch_norm2,
            batch_qkv,
            batch_z,
            batch_gdn,
            batch_attn,
            batch_attn_out,
            batch_head_k,
            batch_head_v,
            batch_up,
            batch_gate_ffn,
            batch_ffn,
            layers,
            rec_states,
            full_states,
        })
    }

    pub fn forward_token(&mut self, w: &Weights, token: i32, pos: usize) -> Result<i32> {
        let d = self.d;
        if w.params.n_layer != d.n_layer {
            return Err(Error::new("model changed after GPU workspace creation"));
        }
        if token < 0 || token as usize >= d.vocab as usize {
            return Err(Error::new("token id outside vocabulary"));
        }
        if pos >= d.seq_cap as usize {
            return Err(Error::new(format!(
                "position {pos} exceeds GPU KV capacity {}",
                d.seq_cap
            )));
        }

        let cuda = &self.cuda;
        cuda.row_device(self.emb_idx, token, self.x)?;

        let mut li = 0usize;
        let mut fi = 0usize;
        for il in 0..d.n_layer {
            let layer = &self.layers[il];
            cuda.op_rmsnorm(
                self.x,
                layer.common.norm.cast::<f32>(),
                self.norm,
                d.n_embd,
                self.eps,
            )?;

            if self.p.is_recurrent(il) {
                let r = layer
                    .rec
                    .as_ref()
                    .ok_or_else(|| Error::new("missing recurrent GPU layer"))?;
                let state = &mut self.rec_states[li];
                cuda.matvec4(
                    r.qkv,
                    r.z,
                    r.beta,
                    r.alpha,
                    self.norm,
                    self.qkv,
                    self.z,
                    self.beta_raw,
                    self.alpha_raw,
                )?;

                let hist_cur = state.hist[state.cur].cast::<f32>();
                cuda.op_conv_silu(
                    r.conv.cast::<f32>(),
                    hist_cur,
                    self.qkv,
                    self.conv_out,
                    d.channels,
                    d.ksize,
                    d.channels,
                )?;

                if d.hist_bytes > 0 {
                    let next = 1 - state.cur;
                    let dst = state.hist[next].cast::<f32>();
                    cuda.zero(state.hist[next], d.hist_bytes)?;
                    if d.ksize > 2 {
                        for j in 0..(d.ksize - 2) {
                            let old = hist_cur.wrapping_add((j + 1) as usize * d.channels as usize);
                            cuda.op_add(
                                dst.wrapping_add(j as usize * d.channels as usize),
                                old,
                                d.channels,
                            )?;
                        }
                    }
                    let last = (d.ksize - 2) as usize * d.channels as usize;
                    cuda.op_add(dst.wrapping_add(last), self.qkv, d.channels)?;
                    state.cur = next;
                }

                cuda.op_l2norm_heads(self.conv_out, 2 * d.nk, d.ds, self.eps)?;
                cuda.op_gdn_fused(
                    self.alpha_raw,
                    r.dt_bias.cast::<f32>(),
                    r.a.cast::<f32>(),
                    self.beta_raw,
                    self.conv_out,
                    self.conv_out.wrapping_add(d.key_dim as usize),
                    self.conv_out.wrapping_add((2 * d.key_dim) as usize),
                    state.state.cast::<f32>(),
                    self.gdn_out,
                    d.nv,
                    d.nk,
                    d.ds,
                )?;
                cuda.op_gdn_post(
                    self.gdn_out,
                    self.z,
                    r.norm.cast::<f32>(),
                    self.gdn_out,
                    d.nv,
                    d.ds,
                    self.eps,
                )?;
                cuda.matvec_device(r.out, self.gdn_out, self.attn)?;
                li += 1;
            } else {
                let f = layer
                    .full
                    .as_ref()
                    .ok_or_else(|| Error::new("missing full-attention GPU layer"))?;
                let state = &self.full_states[fi];
                cuda.matvec3(f.q, f.k, f.v, self.norm, self.qkv, self.head_k, self.head_v)?;
                cuda.zero(self.heads.cast::<c_void>(), d.head_bytes)?;
                cuda.zero(self.gates.cast::<c_void>(), d.head_bytes)?;
                cuda.op_split_qg(self.qkv, self.heads, self.gates, d.n_head, d.hd)?;
                cuda.op_qk_norm_rope(
                    self.heads,
                    f.q_norm.cast::<f32>(),
                    d.n_head,
                    d.hd,
                    self.n_rot,
                    self.rope_base,
                    pos as i32,
                )?;
                cuda.op_qk_norm_rope(
                    self.head_k,
                    f.k_norm.cast::<f32>(),
                    d.n_kv,
                    d.hd,
                    self.n_rot,
                    self.rope_base,
                    pos as i32,
                )?;
                cuda.op_attn(
                    self.heads,
                    self.head_k,
                    self.head_v,
                    self.gates,
                    state.kc.cast::<f32>(),
                    state.vc.cast::<f32>(),
                    self.attn_out,
                    d.n_head,
                    d.n_kv,
                    d.hd,
                    pos as i32,
                    d.seq_cap,
                    self.scale,
                )?;
                cuda.matvec_device(f.out, self.attn_out, self.attn)?;
                fi += 1;
            }

            cuda.op_add(self.x, self.attn, d.n_embd)?;
            cuda.op_rmsnorm(
                self.x,
                layer.common.post_norm.cast::<f32>(),
                self.norm2,
                d.n_embd,
                self.eps,
            )?;
            cuda.matvec2(
                layer.common.up,
                layer.common.gate,
                self.norm2,
                self.up,
                self.gate_ffn,
            )?;
            cuda.op_silu_mul(self.gate_ffn, self.up, self.up, d.n_ff)?;
            cuda.matvec_device(layer.common.down, self.up, self.ffn)?;
            cuda.op_add(self.x, self.ffn, d.n_embd)?;
        }

        cuda.op_rmsnorm(
            self.x,
            self.out_norm.cast::<f32>(),
            self.h,
            d.n_embd,
            self.eps,
        )?;
        cuda.matvec_device(self.emb_idx, self.h, self.logits)?;
        cuda.op_argmax(self.logits, d.vocab, self.argmax.cast::<i32>())?;
        let mut next = 0i32;
        cuda.sync()?;
        cuda.read_i32(self.argmax.cast_const().cast::<c_void>(), &mut next)?;
        Ok(next)
    }

    /// Process prompt tokens `PREFILL_CHUNK` at a time, batching the large
    /// matvecs while keeping recurrent state and the KV cache sequential.
    /// Returns the greedy token after the last prompt token.
    pub fn prefill_tokens(&mut self, w: &Weights, tokens: &[i32], start_pos: usize) -> Result<i32> {
        let d = self.d;
        if w.params.n_layer != d.n_layer {
            return Err(Error::new("model changed after GPU workspace creation"));
        }
        if tokens.is_empty() {
            return Err(Error::new("empty GPU prefill"));
        }
        let end = start_pos
            .checked_add(tokens.len())
            .ok_or_else(|| Error::new("prefill position overflow"))?;
        if end > d.seq_cap as usize {
            return Err(Error::new(format!(
                "prefill ends at {end} beyond GPU KV capacity {}",
                d.seq_cap
            )));
        }

        let cuda = &self.cuda;
        let e = d.n_embd as usize;
        let nf = d.n_ff as usize;
        let nhd = (d.n_head * d.hd) as usize;
        let kv_dim = (d.n_kv * d.hd) as usize;
        let value_dim = (d.nv * d.ds) as usize;
        let qkv_stride = d.channels.max(2 * d.n_head * d.hd) as usize;
        let value_stride = d.nv * d.ds;
        let kv_stride = d.n_kv * d.hd;
        let qkv_out_stride = d.channels.max(2 * d.n_head * d.hd);

        for chunk in tokens.chunks(PREFILL_CHUNK) {
            let nt = chunk.len();
            let nt_i32 = to_i32(nt, "prefill chunk")?;

            for (t, &tok) in chunk.iter().enumerate() {
                if tok < 0 || tok as usize >= d.vocab as usize {
                    return Err(Error::new("token id outside vocabulary"));
                }
                cuda.row_device(self.emb_idx, tok, self.batch_x.wrapping_add(t * e))?;
            }

            let mut li = 0usize;
            let mut fi = 0usize;
            for il in 0..d.n_layer {
                let layer = &self.layers[il];
                for t in 0..nt {
                    cuda.op_rmsnorm(
                        self.batch_x.wrapping_add(t * e),
                        layer.common.norm.cast::<f32>(),
                        self.batch_norm.wrapping_add(t * e),
                        d.n_embd,
                        self.eps,
                    )?;
                }

                if self.p.is_recurrent(il) {
                    let r = layer
                        .rec
                        .as_ref()
                        .ok_or_else(|| Error::new("missing recurrent GPU layer"))?;
                    cuda.batch_matmul(
                        r.qkv,
                        self.batch_norm,
                        self.batch_qkv,
                        nt_i32,
                        d.n_embd,
                        qkv_out_stride,
                    )?;
                    cuda.batch_matmul(
                        r.z,
                        self.batch_norm,
                        self.batch_z,
                        nt_i32,
                        d.n_embd,
                        value_stride,
                    )?;
                } else {
                    let f = layer
                        .full
                        .as_ref()
                        .ok_or_else(|| Error::new("missing full-attention GPU layer"))?;
                    cuda.batch_matmul(
                        f.q,
                        self.batch_norm,
                        self.batch_qkv,
                        nt_i32,
                        d.n_embd,
                        qkv_out_stride,
                    )?;
                    cuda.batch_matmul(
                        f.k,
                        self.batch_norm,
                        self.batch_head_k,
                        nt_i32,
                        d.n_embd,
                        kv_stride,
                    )?;
                    cuda.batch_matmul(
                        f.v,
                        self.batch_norm,
                        self.batch_head_v,
                        nt_i32,
                        d.n_embd,
                        kv_stride,
                    )?;
                }

                // Conv history, GDN state and KV cache depend on the previous
                // token, so these fused ops keep the per-token order.
                for t in 0..nt {
                    let pos = to_i32(start_pos + t, "prefill position")?;
                    let norm_i = self.batch_norm.wrapping_add(t * e);
                    if self.p.is_recurrent(il) {
                        let r = layer
                            .rec
                            .as_ref()
                            .ok_or_else(|| Error::new("missing recurrent GPU layer"))?;
                        let state = &mut self.rec_states[li];
                        cuda.matvec2(r.beta, r.alpha, norm_i, self.beta_raw, self.alpha_raw)?;

                        let qkv_i = self.batch_qkv.wrapping_add(t * qkv_stride);
                        let gdn_i = self.batch_gdn.wrapping_add(t * value_dim);
                        let hist_cur = state.hist[state.cur].cast::<f32>();
                        cuda.op_conv_silu(
                            r.conv.cast::<f32>(),
                            hist_cur,
                            qkv_i,
                            self.conv_out,
                            d.channels,
                            d.ksize,
                            d.channels,
                        )?;

                        if d.hist_bytes > 0 {
                            let next = 1 - state.cur;
                            let dst = state.hist[next].cast::<f32>();
                            cuda.zero(state.hist[next], d.hist_bytes)?;
                            if d.ksize > 2 {
                                for j in 0..(d.ksize - 2) {
                                    let old = hist_cur
                                        .wrapping_add((j + 1) as usize * d.channels as usize);
                                    cuda.op_add(
                                        dst.wrapping_add(j as usize * d.channels as usize),
                                        old,
                                        d.channels,
                                    )?;
                                }
                            }
                            let last = (d.ksize - 2) as usize * d.channels as usize;
                            cuda.op_add(dst.wrapping_add(last), qkv_i, d.channels)?;
                            state.cur = next;
                        }

                        cuda.op_l2norm_heads(self.conv_out, 2 * d.nk, d.ds, self.eps)?;
                        cuda.op_gdn_fused(
                            self.alpha_raw,
                            r.dt_bias.cast::<f32>(),
                            r.a.cast::<f32>(),
                            self.beta_raw,
                            self.conv_out,
                            self.conv_out.wrapping_add(d.key_dim as usize),
                            self.conv_out.wrapping_add((2 * d.key_dim) as usize),
                            state.state.cast::<f32>(),
                            gdn_i,
                            d.nv,
                            d.nk,
                            d.ds,
                        )?;
                        cuda.op_gdn_post(
                            gdn_i,
                            self.batch_z.wrapping_add(t * value_dim),
                            r.norm.cast::<f32>(),
                            gdn_i,
                            d.nv,
                            d.ds,
                            self.eps,
                        )?;
                    } else {
                        let f = layer
                            .full
                            .as_ref()
                            .ok_or_else(|| Error::new("missing full-attention GPU layer"))?;
                        let state = &self.full_states[fi];
                        cuda.zero(self.heads.cast::<c_void>(), d.head_bytes)?;
                        cuda.zero(self.gates.cast::<c_void>(), d.head_bytes)?;
                        cuda.op_split_qg(
                            self.batch_qkv.wrapping_add(t * qkv_stride),
                            self.heads,
                            self.gates,
                            d.n_head,
                            d.hd,
                        )?;
                        cuda.op_qk_norm_rope(
                            self.heads,
                            f.q_norm.cast::<f32>(),
                            d.n_head,
                            d.hd,
                            self.n_rot,
                            self.rope_base,
                            pos,
                        )?;
                        cuda.op_qk_norm_rope(
                            self.batch_head_k.wrapping_add(t * kv_dim),
                            f.k_norm.cast::<f32>(),
                            d.n_kv,
                            d.hd,
                            self.n_rot,
                            self.rope_base,
                            pos,
                        )?;
                        cuda.op_attn(
                            self.heads,
                            self.batch_head_k.wrapping_add(t * kv_dim),
                            self.batch_head_v.wrapping_add(t * kv_dim),
                            self.gates,
                            state.kc.cast::<f32>(),
                            state.vc.cast::<f32>(),
                            self.batch_attn_out.wrapping_add(t * nhd),
                            d.n_head,
                            d.n_kv,
                            d.hd,
                            pos,
                            d.seq_cap,
                            self.scale,
                        )?;
                    }
                }
                if self.p.is_recurrent(il) {
                    li += 1;
                } else {
                    fi += 1;
                }

                if self.p.is_recurrent(il) {
                    let r = layer
                        .rec
                        .as_ref()
                        .ok_or_else(|| Error::new("missing recurrent GPU layer"))?;
                    cuda.batch_matmul(
                        r.out,
                        self.batch_gdn,
                        self.batch_attn,
                        nt_i32,
                        value_stride,
                        d.n_embd,
                    )?;
                } else {
                    let f = layer
                        .full
                        .as_ref()
                        .ok_or_else(|| Error::new("missing full-attention GPU layer"))?;
                    cuda.batch_matmul(
                        f.out,
                        self.batch_attn_out,
                        self.batch_attn,
                        nt_i32,
                        d.n_head * d.hd,
                        d.n_embd,
                    )?;
                }

                for t in 0..nt {
                    let xi = self.batch_x.wrapping_add(t * e);
                    cuda.op_add(xi, self.batch_attn.wrapping_add(t * e), d.n_embd)?;
                    cuda.op_rmsnorm(
                        xi,
                        layer.common.post_norm.cast::<f32>(),
                        self.batch_norm2.wrapping_add(t * e),
                        d.n_embd,
                        self.eps,
                    )?;
                }

                cuda.batch_matmul(
                    layer.common.up,
                    self.batch_norm2,
                    self.batch_up,
                    nt_i32,
                    d.n_embd,
                    d.n_ff,
                )?;
                cuda.batch_matmul(
                    layer.common.gate,
                    self.batch_norm2,
                    self.batch_gate_ffn,
                    nt_i32,
                    d.n_embd,
                    d.n_ff,
                )?;
                for t in 0..nt {
                    let up_i = self.batch_up.wrapping_add(t * nf);
                    cuda.op_silu_mul(self.batch_gate_ffn.wrapping_add(t * nf), up_i, up_i, d.n_ff)?;
                }
                cuda.batch_matmul(
                    layer.common.down,
                    self.batch_up,
                    self.batch_ffn,
                    nt_i32,
                    d.n_ff,
                    d.n_embd,
                )?;
                for t in 0..nt {
                    cuda.op_add(
                        self.batch_x.wrapping_add(t * e),
                        self.batch_ffn.wrapping_add(t * e),
                        d.n_embd,
                    )?;
                }
            }

            let last = nt - 1;
            cuda.op_rmsnorm(
                self.batch_x.wrapping_add(last * e),
                self.out_norm.cast::<f32>(),
                self.h,
                d.n_embd,
                self.eps,
            )?;
            cuda.batch_matmul(self.emb_idx, self.h, self.logits, 1, d.n_embd, d.vocab)?;
        }

        cuda.op_argmax(self.logits, d.vocab, self.argmax.cast::<i32>())?;
        let mut next = 0i32;
        cuda.sync()?;
        cuda.read_i32(self.argmax.cast_const().cast::<c_void>(), &mut next)?;
        Ok(next)
    }
}

impl Drop for GpuWorkspace {
    fn drop(&mut self) {
        for &ptr in &self.buffers {
            unsafe { self.cuda.free_buf(ptr) };
        }
    }
}

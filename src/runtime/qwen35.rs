//! Single-sequence, sequential-token forward pass for Qwen3.5-4B Q4_K_M.

use crate::core::{Error, Result};
use crate::runtime::math::{
    dequant_row, matvec, rmsnorm, rope_inplace, sigmoid, silu, softmax_inplace,
};
use crate::runtime::types::{ModelState, Weights};

/// Read a one-dimensional weight as floats.
fn read_vec(w: &Weights, name: &str) -> Result<Vec<f32>> {
    let wt = w.get(name)?;
    let mut out = vec![0.0f32; wt.in_features()];
    dequant_row(wt.ty, wt.row(w.data.as_slice(), 0)?, &mut out)?;
    Ok(out)
}

fn l2_norm_inplace(x: &mut [f32], eps: f32) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() + eps;
    let scale = 1.0 / ss.sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Run one token through the model, updating the recurrent and KV state.
pub fn forward(w: &Weights, st: &mut ModelState, token: i32) -> Result<Vec<f32>> {
    let p = &w.params;
    let data = w.data.as_slice();

    let emb = w.get("token_embd.weight")?;
    if token < 0 || token as usize >= emb.out_features() {
        return Err(Error::new("token id outside vocabulary"));
    }
    let mut x = vec![0.0f32; p.n_embd];
    dequant_row(emb.ty, emb.row(data, token as usize)?, &mut x)?;

    let ds = p.ssm_state;
    let nk = p.ssm_group;
    let nv = p.ssm_dt_rank;
    let key_dim = ds * nk;
    let value_dim = ds * nv;
    let channels = 2 * key_dim + value_dim;
    let hd = p.head_dim;
    let mut fi = 0usize;

    for il in 0..p.n_layer {
        let norm = rmsnorm(
            &x,
            &read_vec(w, &format!("blk.{il}.attn_norm.weight"))?,
            p.eps,
        );

        let attn = if p.is_recurrent(il) {
            let li = il - fi;

            let mut qkv = vec![0.0f32; channels];
            matvec(
                data,
                w.get(&format!("blk.{il}.attn_qkv.weight"))?,
                &norm,
                &mut qkv,
            )?;
            let mut z = vec![0.0f32; value_dim];
            matvec(
                data,
                w.get(&format!("blk.{il}.attn_gate.weight"))?,
                &norm,
                &mut z,
            )?;

            let mut beta_raw = vec![0.0f32; nv];
            matvec(
                data,
                w.get(&format!("blk.{il}.ssm_beta.weight"))?,
                &norm,
                &mut beta_raw,
            )?;
            let mut alpha_raw = vec![0.0f32; nv];
            matvec(
                data,
                w.get(&format!("blk.{il}.ssm_alpha.weight"))?,
                &norm,
                &mut alpha_raw,
            )?;
            let dt_bias = read_vec(w, &format!("blk.{il}.ssm_dt.bias"))?;
            let a = read_vec(w, &format!("blk.{il}.ssm_a"))?;

            let mut beta = vec![0.0f32; nv];
            let mut gate = vec![0.0f32; nv];
            for h in 0..nv {
                beta[h] = sigmoid(beta_raw[h]);
                gate[h] = a[h] * (1.0f32 + (alpha_raw[h] + dt_bias[h]).exp()).ln();
            }

            let kw = w.get(&format!("blk.{il}.ssm_conv1d.weight"))?;
            let ksize = p.conv_kernel;
            let mut kernel = vec![0.0f32; channels * ksize];
            for c in 0..channels {
                dequant_row(
                    kw.ty,
                    kw.row(data, c)?,
                    &mut kernel[c * ksize..(c + 1) * ksize],
                )?;
            }

            let mut conv_input = vec![0.0f32; ksize * channels];
            let hist = &st.linear[li].conv;
            conv_input[..hist.len()].copy_from_slice(hist);
            conv_input[hist.len()..].copy_from_slice(&qkv);

            let mut conv_out = vec![0.0f32; channels];
            for c in 0..channels {
                let mut sum = 0.0f32;
                for j in 0..ksize {
                    sum += kernel[c * ksize + j] * conv_input[j * channels + c];
                }
                conv_out[c] = silu(sum);
            }

            if ksize > 1 {
                let state_conv = &mut st.linear[li].conv;
                state_conv.copy_from_slice(&conv_input[channels..]);
            }

            for h in 0..nk {
                l2_norm_inplace(&mut conv_out[h * ds..(h + 1) * ds], p.eps);
                l2_norm_inplace(
                    &mut conv_out[key_dim + h * ds..key_dim + (h + 1) * ds],
                    p.eps,
                );
            }

            let mut out = vec![0.0f32; value_dim];
            let inv_sqrt = 1.0f32 / (ds as f32).sqrt();
            let state = &mut st.linear[li].ssm;
            for h in 0..nv {
                let kh = h % nk;
                let qh = &conv_out[kh * ds..(kh + 1) * ds];
                let kh_vec = &conv_out[key_dim + kh * ds..key_dim + (kh + 1) * ds];
                let vh = &conv_out[2 * key_dim + h * ds..2 * key_dim + (h + 1) * ds];
                let decay = gate[h].exp();
                let base = h * ds * ds;
                for j in 0..ds {
                    let row = &mut state[base + j * ds..base + (j + 1) * ds];
                    for v in row.iter_mut() {
                        *v *= decay;
                    }
                    let dot = row
                        .iter()
                        .zip(kh_vec.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                    let delta = (vh[j] - dot) * beta[h];
                    for (a, b) in row.iter_mut().zip(kh_vec.iter()) {
                        *a += delta * *b;
                    }
                    let o = row.iter().zip(qh.iter()).map(|(a, b)| a * b).sum::<f32>();
                    out[h * ds + j] = o * inv_sqrt;
                }
            }

            let sn = read_vec(w, &format!("blk.{il}.ssm_norm.weight"))?;
            for h in 0..nv {
                let oh = &mut out[h * ds..(h + 1) * ds];
                let normed = rmsnorm(oh, &sn, p.eps);
                let zh = &z[h * ds..(h + 1) * ds];
                for d in 0..ds {
                    oh[d] = normed[d] * silu(zh[d]);
                }
            }

            let mut attn = vec![0.0f32; p.n_embd];
            matvec(
                data,
                w.get(&format!("blk.{il}.ssm_out.weight"))?,
                &out,
                &mut attn,
            )?;
            attn
        } else {
            let layer = fi;
            fi += 1;

            let mut qg = vec![0.0f32; p.n_head * hd * 2];
            matvec(
                data,
                w.get(&format!("blk.{il}.attn_q.weight"))?,
                &norm,
                &mut qg,
            )?;
            let qn = read_vec(w, &format!("blk.{il}.attn_q_norm.weight"))?;
            let kn = read_vec(w, &format!("blk.{il}.attn_k_norm.weight"))?;

            let mut q = vec![0.0f32; p.n_head * hd];
            for h in 0..p.n_head {
                let base = h * hd * 2;
                let qh = &mut q[h * hd..(h + 1) * hd];
                qh.copy_from_slice(&qg[base..base + hd]);
                qh.copy_from_slice(&rmsnorm(qh, &qn, p.eps));
                rope_inplace(qh, st.pos, p.rope_dim.min(hd), p.rope_base);
            }

            let mut k = vec![0.0f32; p.n_head_kv * hd];
            matvec(
                data,
                w.get(&format!("blk.{il}.attn_k.weight"))?,
                &norm,
                &mut k,
            )?;
            let mut v = vec![0.0f32; p.n_head_kv * hd];
            matvec(
                data,
                w.get(&format!("blk.{il}.attn_v.weight"))?,
                &norm,
                &mut v,
            )?;
            for h in 0..p.n_head_kv {
                let kh = &mut k[h * hd..(h + 1) * hd];
                kh.copy_from_slice(&rmsnorm(kh, &kn, p.eps));
                rope_inplace(kh, st.pos, p.rope_dim.min(hd), p.rope_base);
            }

            for h in 0..p.n_head_kv {
                st.full[layer].k[h].extend_from_slice(&k[h * hd..(h + 1) * hd]);
                st.full[layer].v[h].extend_from_slice(&v[h * hd..(h + 1) * hd]);
            }

            let mut attn_out = vec![0.0f32; p.n_head * hd];
            let scale = 1.0f32 / (hd as f32).sqrt();
            for h in 0..p.n_head {
                let qh = &q[h * hd..(h + 1) * hd];
                let gh = &qg[h * hd * 2 + hd..h * hd * 2 + hd * 2];
                let kv = h * p.n_head_kv / p.n_head;
                let kc = &st.full[layer].k[kv];
                let vc = &st.full[layer].v[kv];
                let n_tokens = kc.len() / hd;
                let mut scores = vec![0.0f32; n_tokens];
                for t in 0..n_tokens {
                    let kt = &kc[t * hd..(t + 1) * hd];
                    scores[t] = qh.iter().zip(kt.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                softmax_inplace(&mut scores);
                let oh = &mut attn_out[h * hd..(h + 1) * hd];
                for t in 0..n_tokens {
                    let vt = &vc[t * hd..(t + 1) * hd];
                    let s = scores[t];
                    for d in 0..hd {
                        oh[d] += s * vt[d];
                    }
                }
                for d in 0..hd {
                    oh[d] *= sigmoid(gh[d]);
                }
            }

            let mut attn = vec![0.0f32; p.n_embd];
            matvec(
                data,
                w.get(&format!("blk.{il}.attn_output.weight"))?,
                &attn_out,
                &mut attn,
            )?;
            attn
        };

        for (a, b) in x.iter_mut().zip(attn.iter()) {
            *a += *b;
        }

        let norm2 = rmsnorm(
            &x,
            &read_vec(w, &format!("blk.{il}.post_attention_norm.weight"))?,
            p.eps,
        );
        let mut up = vec![0.0f32; p.n_ff];
        matvec(
            data,
            w.get(&format!("blk.{il}.ffn_up.weight"))?,
            &norm2,
            &mut up,
        )?;
        let mut g = vec![0.0f32; p.n_ff];
        matvec(
            data,
            w.get(&format!("blk.{il}.ffn_gate.weight"))?,
            &norm2,
            &mut g,
        )?;
        for (u, gi) in up.iter_mut().zip(g.iter()) {
            *u *= silu(*gi);
        }
        let mut ffn = vec![0.0f32; p.n_embd];
        matvec(
            data,
            w.get(&format!("blk.{il}.ffn_down.weight"))?,
            &up,
            &mut ffn,
        )?;
        for (a, b) in x.iter_mut().zip(ffn.iter()) {
            *a += *b;
        }
    }

    let h = rmsnorm(&x, &read_vec(w, "output_norm.weight")?, p.eps);
    let mut logits = vec![0.0f32; emb.out_features()];
    matvec(data, emb, &h, &mut logits)?;
    st.pos += 1;
    Ok(logits)
}

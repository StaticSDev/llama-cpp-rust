//! Input embeddings and position/pooling inputs.

use super::context::GraphContext;
use super::ggml::{
    self, cont, get_rows, mul, new_tensor_1d, new_tensor_2d, new_tensor_3d, pad, reshape_1d,
    scale, set_rows, view_2d, Tensor, Type,
};
use super::inputs::{
    GraphInputAttnTemp, GraphInputCls, GraphInputCrossEmbd, GraphInputEmbd, GraphInputEmbdH,
    GraphInputMean, GraphInputOutIds, GraphInputPos, GraphInputPosBucket, GraphInputPosBucketKv,
};
use crate::arch::arch_supports_mixed_batch;
use crate::cparams::ContextType;

fn build_rope_impl(cur: &Tensor, pos: &Tensor, rp: super::support::RopeParams) -> Tensor {
    if rp.n_rot == 0 {
        return cur.clone();
    }
    if rp.is_mrope() {
        ggml::rope_multi(cur, pos)
    } else {
        ggml::rope(cur, pos)
    }
}

impl GraphContext {
    /// Input embeddings with optional LoRA and the mixed token/embd path.
    pub fn build_inp_embd(&self, tok_embd: &Tensor, tok_scale: f32) -> Tensor {
        let n_embd_inp = self.hparams.n_embd_inp() as i64;
        let n_embd = self.n_embd;

        let mut inp = GraphInputEmbd::new(n_embd_inp);

        let has_mixed = arch_supports_mixed_batch(self.model.arch)
            && self.cparams.ctx_type == ContextType::Default;
        let n_tok_rows = if has_mixed {
            super::inputs::graph_n_tok_rows(&self.ubatch)
        } else {
            0
        };

        {
            let tokens = new_tensor_1d(Type::I32, self.ubatch.n_tokens as i64);
            self.cb(&tokens, "inp_tokens", -1);
            tokens.set_input();
            self.res.borrow_mut().t_inp_tokens = Some(tokens.clone());
            inp.tokens = Some(tokens);

            let embd = new_tensor_2d(Type::F32, n_embd_inp, self.ubatch.n_tokens as i64);
            self.cb(&embd, "inp_embd", -1);
            embd.set_input();
            inp.embd = Some(embd);

            if has_mixed {
                let mixed = new_tensor_1d(Type::I32, n_tok_rows);
                self.cb(&mixed, "inp_mixed_tokens", -1);
                mixed.set_input();
                inp.mixed_tokens = Some(mixed);
            }
        }

        let tokens = inp.tokens.clone().unwrap();
        let embd = inp.embd.clone().unwrap();
        let embd0 = get_rows(tok_embd, &tokens);
        self.gf.borrow_mut().build_forward_order(&embd0);
        let embd1 = embd.clone();
        let embd2 = if has_mixed {
            let mixed = inp.mixed_tokens.clone().unwrap();
            let e = get_rows(tok_embd, &mixed);
            self.gf.borrow_mut().build_forward_order(&e);
            Some(e)
        } else {
            None
        };

        let build_tok = |cur: &Tensor, ids: &Tensor| -> Tensor {
            let mut cur = cur.clone();
            if let Some(hdmd) = self.model.hdmd.as_ref() {
                if let Some(entry) = hdmd.get_inv(tok_embd) {
                    cur = super::ggml::mul_mat(&cur, &entry.rot);
                    if let Some(signs) = &entry.signs {
                        cur = mul(&cur, signs);
                    }
                }
            }
            if let Some(loras) = &self.loras {
                for (lora, adapter_scale) in loras.iter() {
                    let Some(lw) = lora.get_weight(tok_embd) else { continue };
                    let factor = lw.get_scale(lora.alpha, *adapter_scale);
                    let delta = scale(&super::ggml::mul_mat(&lw.b, &get_rows(&lw.a, ids)), factor);
                    cur = super::ggml::add(&cur, &delta);
                }
            }
            if n_embd_inp != n_embd {
                cur = pad(&cur, n_embd_inp - n_embd, 0, 0, 0);
            }
            cur
        };

        let mut inps = Vec::new();
        inps.push(build_tok(&embd0, &tokens));
        inps.push(embd1);
        if has_mixed {
            let mixed_slots = new_tensor_1d(Type::I64, n_tok_rows);
            self.cb(&mixed_slots, "inp_mixed_slots", -1);
            mixed_slots.set_input();
            inp.mixed_slots = Some(mixed_slots.clone());

            let mixed_embd = new_tensor_2d(Type::F32, n_embd_inp, self.ubatch.n_tokens as i64);
            self.cb(&mixed_embd, "inp_mixed_embd", -1);
            mixed_embd.set_input();
            inp.mixed_embd = Some(mixed_embd.clone());

            let embd_mixed = build_tok(&embd2.unwrap(), &inp.mixed_tokens.clone().unwrap());
            inps.push(set_rows(&super::ggml::dup(&mixed_embd), &embd_mixed, &mixed_slots));
        }

        let idx = if self.ubatch.is_mixed() {
            2
        } else if !self.ubatch.token.is_empty() {
            0
        } else {
            1
        };
        let mut cur = self.gf.borrow_mut().build_forward_select(&inps, idx);
        if n_embd_inp != n_embd {
            cur = view_2d(&cur, n_embd, self.n_tokens, cur.nb(1), 0);
        }
        self.res.borrow_mut().t_inp_embd = Some(cur.clone());

        let scale_tok_only = self.hparams.f_embedding_scale != 0.0 && self.hparams.n_deepstack_layers > 0;
        if self.hparams.f_embedding_scale != 0.0 && !scale_tok_only {
            let cur_owned = if cur.is_contiguous() { cur } else { cont(&cur) };
            cur = scale(&cur_owned, self.hparams.f_embedding_scale);
        }

        inp.scale_tok = tok_scale * if scale_tok_only { self.hparams.f_embedding_scale } else { 1.0 };
        if inp.scale_tok != 1.0 {
            let scale_rows = new_tensor_2d(Type::F32, 1, self.ubatch.n_tokens as i64);
            self.cb(&scale_rows, "inp_scale_rows", -1);
            scale_rows.set_input();
            inp.scale_rows = Some(scale_rows.clone());
            let cur_owned = if cur.is_contiguous() { cur } else { cont(&cur) };
            cur = mul(&cur_owned, &scale_rows);
        }

        self.cb(&cur, "embd", -1);
        self.res.borrow_mut().add_input(inp);
        self.expand(&cur);
        cur
    }

    /// The hidden-state input of an MTP graph.
    pub fn build_inp_embd_state(&self, inp: &mut GraphInputEmbdH, n_embd_state: i64) -> Tensor {
        if !self.ubatch.embd_state_row.is_empty() {
            let h_rows = new_tensor_1d(Type::I32, self.n_tokens);
            h_rows.set_input();
            h_rows.set_name("mtp_h_rows");
            inp.h_rows = Some(h_rows.clone());
            return get_rows(self.h_nextn_src.as_ref().unwrap_or(&h_rows), &h_rows);
        }
        let h = new_tensor_2d(Type::F32, n_embd_state, self.n_tokens);
        h.set_input();
        h.set_name("mtp_h_input");
        h
    }

    /// `build_rope`: apply the RoPE of layer `il`, if any.
    pub fn build_rope(&self, cur: &Tensor, pos: &Tensor, il: i32) -> Tensor {
        let rp = self.model.get_rope_params(&self.cparams, il);
        build_rope_impl(cur, pos, rp)
    }

    /// RoPE for the lightning indexer, with an optional section offset.
    pub fn build_rope_indexer(&self, cur: &Tensor, pos: &Tensor, il: i32) -> Tensor {
        let rp = self.model.get_rope_params_indexer(&self.cparams, il);
        let (n_rot, n_offs) = (rp.n_rot, rp.n_offs);
        let out = build_rope_impl(cur, pos, rp);
        if n_rot > 0 && n_offs > 0 {
            ggml::rope_set_offset(&out, n_offs as i64)
        } else {
            out
        }
    }

    pub fn build_inp_pos(&self) -> Tensor {
        let mut inp = GraphInputPos::new(
            self.hparams.n_pos_per_embd(),
            self.hparams.rope_section_order,
        );
        let cur = new_tensor_1d(
            Type::I32,
            self.n_tokens * self.hparams.n_pos_per_embd() as i64,
        );
        cur.set_input();
        self.cb(&cur, "inp_pos", -1);
        inp.pos = Some(cur.clone());
        self.res.borrow_mut().add_input(inp);
        cur
    }

    pub fn build_inp_attn_scale(&self) -> Tensor {
        let mut inp = GraphInputAttnTemp::new(
            self.hparams.n_attn_temp_floor_scale,
            self.hparams.f_attn_temp_scale,
            self.hparams.f_attn_temp_offset,
            self.hparams.attn_temp_pow,
        );
        // 1x1xN for broadcasting.
        let cur = new_tensor_3d(Type::F32, 1, 1, self.n_tokens);
        cur.set_input();
        self.cb(&cur, "inp_attn_scale", -1);
        inp.attn_scale = Some(cur.clone());
        self.res.borrow_mut().add_input(inp);
        cur
    }

    pub fn build_inp_out_ids(&self) -> Tensor {
        let mut inp = GraphInputOutIds::new(&self.hparams, &self.cparams, self.n_outputs as u32);
        let cur = new_tensor_1d(Type::I32, self.n_outputs);
        cur.set_input();
        cur.set_name("out_ids");
        inp.out_ids = Some(cur.clone());
        self.res.borrow_mut().add_input(inp);
        cur
    }

    pub fn build_inp_mean(&self) -> Tensor {
        let mut inp = GraphInputMean::new(&self.cparams);
        let cur = new_tensor_2d(
            Type::F32,
            self.ubatch.n_tokens as i64,
            self.ubatch.n_seqs_unq as i64,
        );
        cur.set_input();
        cur.set_name("mean");
        inp.mean = Some(cur.clone());
        self.res.borrow_mut().add_input(inp);
        cur
    }

    pub fn build_inp_cls(&self) -> Tensor {
        let mut inp = GraphInputCls::new(&self.cparams, self.model.arch);
        let cur = new_tensor_1d(Type::I32, self.ubatch.n_seqs_unq as i64);
        cur.set_input();
        cur.set_name("cls");
        inp.cls = Some(cur.clone());
        self.res.borrow_mut().add_input(inp);
        cur
    }

    pub fn build_inp_cross_embd(&self) -> Tensor {
        let Some(cross) = self.cross.clone() else {
            return new_tensor_2d(Type::F32, self.n_embd, self.n_tokens);
        };
        let mut inp = GraphInputCrossEmbd::new(cross.clone());
        let n_embd = if !cross.v_embd.is_empty() {
            cross.n_embd
        } else {
            self.hparams.n_embd_inp() as i64
        };
        let n_enc = if !cross.v_embd.is_empty() {
            cross.n_enc
        } else {
            self.hparams.n_ctx_train as i64
        };
        let cur = new_tensor_2d(Type::F32, n_embd, n_enc);
        cur.set_input();
        cur.set_name("cross_embd");
        inp.cross_embd = Some(cur.clone());
        self.res.borrow_mut().add_input(inp);
        cur
    }

    pub fn build_inp_pos_bucket_enc(&self) -> Tensor {
        let mut inp = GraphInputPosBucket::new(&self.hparams);
        let cur = new_tensor_2d(Type::I32, self.n_tokens, self.n_tokens);
        cur.set_input();
        cur.set_name("pos_bucket_enc");
        inp.pos_bucket = Some(cur.clone());
        self.res.borrow_mut().add_input(inp);
        cur
    }

    pub fn build_inp_pos_bucket_dec(&self) -> Tensor {
        let Some(mctx) = self.mctx.clone() else {
            return new_tensor_2d(Type::I32, 1, self.n_tokens);
        };
        let mut inp = GraphInputPosBucketKv::new(&self.hparams, mctx.clone());
        let n_kv = mctx.get_n_kv() as i64;
        let cur = new_tensor_2d(Type::I32, n_kv, self.n_tokens);
        cur.set_input();
        cur.set_name("pos_bucket_dec");
        inp.pos_bucket = Some(cur.clone());
        self.res.borrow_mut().add_input(inp);
        cur
    }

    pub fn build_pos_bias(&self, pos_bucket: &Tensor, attn_rel_b: &Tensor) -> Tensor {
        let pos_bucket_1d = reshape_1d(pos_bucket, pos_bucket.ne(0) * pos_bucket.ne(1));
        self.cb(&pos_bucket_1d, "pos_bucket_1d", -1);
        let mut pos_bias = get_rows(attn_rel_b, &pos_bucket_1d);
        pos_bias = super::ggml::reshape_3d(
            &pos_bias,
            pos_bias.ne(0),
            pos_bucket.ne(0),
            pos_bucket.ne(1),
        );
        pos_bucket_1d.set_name("pos_bucket_1d");
        pos_bias = super::ggml::permute(&pos_bias, 2, 0, 1, 3);
        pos_bias = cont(&pos_bias);
        self.cb(&pos_bias, "pos_bias", -1);
        pos_bias
    }
}

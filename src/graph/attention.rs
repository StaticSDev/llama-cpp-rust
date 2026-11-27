//! Attention builders: masks, flash/MHA attention and every KV-cache variant.

use std::cell::RefCell;
use std::rc::Rc;

use super::context::GraphContext;
use super::dsv4::{build_comp_inputs, build_raw_kq_mask, GraphInputDsv4, GraphInputDsv4Raw};
use super::ggml::{
    self, add, cast, cont, cont_2d, div, fill, get_rows, mul, mul_mat, new_tensor_4d, permute,
    reshape_4d, set_rows, soft_max_ext, view_4d, Tensor, Type,
};
use super::inputs::{
    build_attn_inp_attn_mask, build_attn_inp_kq_mask, GraphInputAttnCross, GraphInputAttnKDsa,
    GraphInputAttnKDsaIswa, GraphInputAttnKIswa, GraphInputAttnK, GraphInputAttnKv,
    GraphInputAttnKvIswa, GraphInputAttnKvMsa, GraphInputAttnNoCache,
};
use super::support::{pad_to, MemoryContext};
use super::types::{FusedOp, GraphFusedNode};

impl GraphContext {
    /// `build_attn_mha`: flash attention or explicit QK^T / softmax / PV.
    #[allow(clippy::too_many_arguments)]
    pub fn build_attn_mha(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        kq_b: Option<&Tensor>,
        kq_mask: &Tensor,
        kv_rows: Option<&Tensor>,
        sinks: &Tensor,
        v_mla: Option<&Tensor>,
        n_kv_max: i64,
        kq_scale: f32,
        il: i32,
        bias_table: Option<&Tensor>,
        bias_indices: Option<&Tensor>,
    ) -> Tensor {
        let v_trans = v.nb(1) > v.nb(2);
        let n_stream = kv_rows.map(|t| t.ne(1)).unwrap_or(k.ne(3)).max(1);

        let q = view_4d(
            q,
            [q.ne(0), q.ne(1), q.ne(2) / n_stream, n_stream],
            q.nb(1),
            q.nb(2),
            q.nb(2) * (q.ne(2) / n_stream) as usize,
            0,
        );
        let mut k = permute(k, 0, 2, 1, 3);
        let mut v = permute(v, 0, 2, 1, 3);

        let dense_bias_mask = || -> Tensor {
            let table = cont(&permute(bias_table.unwrap(), 1, 0, 2, 3));
            let mut bias = get_rows(&table, bias_indices.unwrap());
            bias = reshape_4d(
                &bias,
                [q.ne(2), bias_indices.unwrap().ne(0), q.ne(1), n_stream],
            );
            bias = cont(&permute(&bias, 2, 0, 1, 3));
            self.cb(&bias, "kq_b", il);

            let mut mask = if kq_mask.ty() == Type::F32 {
                kq_mask.clone()
            } else {
                cast(kq_mask, Type::F32)
            };
            mask = add(&bias, &mask);
            if self.cparams.flash_attn {
                mask = cast(&mask, Type::F16);
            }
            self.cb(&mask, "kq_mask_rel_pos", il);
            mask
        };

        let mut cur;
        let use_flash_attn = self.cparams.flash_attn && kq_b.is_none();
        if use_flash_attn {
            if v_trans {
                v = ggml::transpose(&v);
            }
            if k.ty() == Type::F32 {
                k = cast(&k, Type::F16);
            }
            if v.ty() == Type::F32 {
                v = cast(&v, Type::F16);
            }
            let softcap = if self.hparams.attn_soft_cap {
                self.hparams.f_attn_logit_softcapping
            } else {
                0.0
            };
            cur = match kv_rows {
                Some(rows) => ggml::flash_attn_ext_rows(
                    &q, &k, &v, kq_mask, rows, kq_scale, self.hparams.f_max_alibi_bias, softcap,
                ),
                None => ggml::flash_attn_ext(
                    &q, &k, &v, kq_mask, None, kq_scale, self.hparams.f_max_alibi_bias, softcap,
                ),
            };
            self.res.borrow_mut().add_fused_node(GraphFusedNode {
                op: if kv_rows.is_some() {
                    FusedOp::FlashAttnKvRows
                } else {
                    FusedOp::FlashAttn
                },
                tensor: cur.clone(),
                il,
            });

            ggml::flash_attn_ext_add_sinks(&cur, sinks);
            ggml::flash_attn_ext_set_n_kv_max(&cur, n_kv_max as i32);
            ggml::prec_set_acc(&cur, ggml::Prec::F32);

            if let Some(table) = bias_table {
                ggml::flash_attn_ext_add_indexed_bias(&cur, Some(table), bias_indices);
                let dev = self.model.dev_layer(il);
                if (!self.cparams.offload_kqv && !dev.is_cpu())
                    || !dev.supports_op(cur.op())
                {
                    ggml::flash_attn_ext_add_indexed_bias(&cur, None, None);
                    cur.set_src(3, Some(dense_bias_mask()));
                }
            }

            if let Some(v_mla) = v_mla {
                cur = permute(&cur, 0, 2, 1, 3);
                cur = mul_mat(v_mla, &cur);
                self.cb(&cur, "fattn_mla", il);
                cur = permute(&cur, 0, 2, 1, 3);
                cur = cont(&cur);
            }
            let (ne0, ne1) = (cur.ne(0) * cur.ne(1), cur.ne(2) * cur.ne(3));
            cur = ggml::reshape_2d(&cur, ne0, ne1);
        } else {
            let mut kq_mask = kq_mask.clone();
            if bias_table.is_some() {
                kq_mask = dense_bias_mask();
            }
            let mut kq = mul_mat(&k, &q);
            self.cb(&kq, "kq", il);
            ggml::prec_set_acc(&kq, ggml::Prec::BF16);

            if self.model.arch == crate::arch::Arch::Grok {
                kq = ggml::tanh(&super::ggml::scale(
                    &kq,
                    self.hparams.f_attn_out_scale / self.hparams.f_attn_logit_softcapping,
                ));
                self.cb(&kq, "kq_tanh", il);
                kq = super::ggml::scale(&kq, self.hparams.f_attn_logit_softcapping);
                self.cb(&kq, "kq_scaled", il);
            }
            if self.hparams.attn_soft_cap {
                kq = super::ggml::scale(&kq, 1.0 / self.hparams.f_attn_logit_softcapping);
                self.cb(&kq, "kq_scaled_1", il);
                kq = ggml::tanh(&kq);
                self.cb(&kq, "kq_tanh", il);
                kq = super::ggml::scale(&kq, self.hparams.f_attn_logit_softcapping);
                self.cb(&kq, "kq_scaled_2", il);
            }
            if let Some(b) = kq_b {
                kq = add(&kq, b);
                self.cb(&kq, "kq_plus_kq_b", il);
            }
            kq = soft_max_ext(&kq, &kq_mask, kq_scale, self.hparams.f_max_alibi_bias);
            ggml::soft_max_add_sinks(&kq, sinks);
            self.cb(&kq, "kq_soft_max", il);

            if !v_trans {
                v = cont(&ggml::transpose(&v));
                self.cb(&v, "v_cont", il);
            }
            let mut kqv = mul_mat(&v, &kq);
            self.cb(&kqv, "kqv", il);
            if let Some(v_mla) = v_mla {
                kqv = mul_mat(v_mla, &kqv);
                self.cb(&kqv, "kqv_mla", il);
            }
            cur = permute(&kqv, 0, 2, 1, 3);
            let (ne0, ne1, ne2, ne3) = (cur.ne(0), cur.ne(1), cur.ne(2), cur.ne(3));
            cur = cont_2d(&cur, ne0 * ne1, ne2 * ne3);
            if !self.cparams.offload_kqv {
                self.sched.borrow_mut().set_tensor_backend(&cur, self.backend_cpu);
            }
        }

        self.expand(&cur);
        cur
    }

    pub fn build_attn_inp_no_cache(&self) -> Rc<RefCell<GraphInputAttnNoCache>> {
        let mut inp = GraphInputAttnNoCache::new(&self.hparams, &self.cparams);
        let ty = if self.cparams.flash_attn { Type::F16 } else { Type::F32 };
        let n_kv = pad_to(self.n_tokens, 256);

        let raw = new_tensor_4d(ty, [n_kv, self.n_tokens, 1, 1]);
        raw.set_input();
        self.cb(&raw, "self_kq_mask", -1);
        inp.self_attn_mask.raw = Some(raw);

        if self.hparams.swa_type != crate::hparams::SwaType::None {
            let swa = new_tensor_4d(ty, [n_kv, self.n_tokens, 1, 1]);
            swa.set_input();
            inp.self_attn_mask_swa.raw = Some(swa);
        }
        self.res.borrow_mut().add_input(inp)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_attn_no_cache(
        &self,
        inp: &GraphInputAttnNoCache,
        wo: Option<&Tensor>,
        wo_b: Option<&Tensor>,
        wo_s: Option<&Tensor>,
        q_cur: &Tensor,
        k_cur: &Tensor,
        v_cur: &Tensor,
        kq_b: Option<&Tensor>,
        sinks: &Tensor,
        v_mla: Option<&Tensor>,
        kq_scale: f32,
        il: i32,
    ) -> Tensor {
        self.expand(q_cur);
        self.expand(k_cur);
        self.expand(v_cur);

        let is_swa = self.hparams.is_swa(il.max(0) as u32).unwrap_or(false);
        let mask = if is_swa {
            inp.self_attn_mask_swa.raw.as_ref()
        } else {
            inp.self_attn_mask.raw.as_ref()
        };
        let rows = if is_swa {
            inp.self_attn_mask_swa.rows.as_ref()
        } else {
            inp.self_attn_mask.rows.as_ref()
        };
        let kq_mask = mask.unwrap_or(k_cur);

        let n_kv_pad = kq_mask.ne(0) - k_cur.ne(2);
        let k = ggml::pad(k_cur, 0, 0, n_kv_pad, 0);
        let v = ggml::pad(v_cur, 0, 0, n_kv_pad, 0);
        let kq_b = kq_b.map(|b| ggml::pad(b, n_kv_pad, 0, 0, 0));

        let mut cur = self.build_attn_mha(
            q_cur, &k, &v, kq_b.as_ref(), kq_mask, rows, sinks, v_mla, 0, kq_scale, il, None, None,
        );
        self.cb(&cur, "kqv_out", il);
        if let Some(wo) = wo {
            cur = self.build_lora_mm(wo, &cur, wo_s);
        }
        if let Some(b) = wo_b {
            cur = add(&cur, b);
        }
        cur
    }

    pub fn build_attn_inp_kv(&self) -> Rc<RefCell<GraphInputAttnKv>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        self.build_attn_inp_kv_impl(mctx)
    }

    pub(crate) fn build_attn_inp_kv_impl(
        &self,
        mctx: Rc<dyn MemoryContext>,
    ) -> Rc<RefCell<GraphInputAttnKv>> {
        let mut inp = GraphInputAttnKv::new(&self.hparams, &self.cparams, mctx.clone());
        inp.self_k_idxs = Some(mctx.build_input_k_idxs(&self.ubatch));
        inp.self_v_idxs = Some(mctx.build_input_v_idxs(&self.ubatch));
        inp.self_attn_mask = build_attn_inp_attn_mask(mctx.as_ref(), &self.ubatch, &self.cparams);
        inp.self_k_rot = mctx.get_k_rot();
        inp.self_v_rot = mctx.get_v_rot();
        self.res.borrow_mut().add_input(inp)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_attn_kv(
        &self,
        inp: &GraphInputAttnKv,
        wo: Option<&Tensor>,
        wo_b: Option<&Tensor>,
        wo_s: Option<&Tensor>,
        q_cur: &Tensor,
        k_cur: &Tensor,
        v_cur: &Tensor,
        kq_b: Option<&Tensor>,
        sinks: &Tensor,
        v_mla: Option<&Tensor>,
        kq_scale: f32,
        il: i32,
    ) -> Tensor {
        let mut q_cur = q_cur.clone();
        let mut k_cur = k_cur.clone();
        let mut v_cur = v_cur.clone();

        let k_rot = inp.self_k_rot.get(il as usize).cloned().flatten();
        if let Some(rot) = &k_rot {
            q_cur = super::ggml::mul_mat(&q_cur, rot);
            k_cur = super::ggml::mul_mat(&k_cur, rot);
        }
        let v_rot = inp.self_v_rot.get(il as usize).cloned().flatten();
        if let Some(rot) = &v_rot {
            v_cur = super::ggml::mul_mat(&v_cur, rot);
        }

        let mctx_cur = inp.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        scale_kv_inputs(
            mctx_cur.as_ref(),
            &mut q_cur,
            Some(&mut k_cur),
            Some(&mut v_cur),
            il,
        );

        self.expand(&q_cur);
        self.expand(&v_cur);
        self.expand(&k_cur);

        let (k, v) = if self.cparams.training {
            (k_cur.clone(), v_cur.clone())
        } else {
            let k_idxs = inp.get_k_idxs().unwrap_or_else(|| new_tensor_4d(Type::I64, [1, 1, 1, 1]));
            let v_idxs = inp.get_v_idxs().unwrap_or_else(|| new_tensor_4d(Type::I64, [1, 1, 1, 1]));
            self.expand(&mctx_cur.cpy_k(&k_cur, &k_idxs, il));
            self.expand(&mctx_cur.cpy_v(&v_cur, &v_idxs, il));
            (mctx_cur.get_k(il), mctx_cur.get_v(il))
        };

        let kq_mask = inp.self_attn_mask.raw.as_ref().unwrap_or(&k);
        let kv_rows = inp.self_attn_mask.rows.as_ref();
        let mut cur = self.build_attn_mha(
            &q_cur, &k, &v, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il, None, None,
        );
        cur = scale_kv_output(mctx_cur.as_ref(), &cur, il);
        self.cb(&cur, "kqv_out", il);
        if let Some(rot) = v_rot {
            cur = super::ggml::mul_mat(&cur, &rot);
        }
        if let Some(wo) = wo {
            cur = self.build_lora_mm(wo, &cur, wo_s);
        }
        if let Some(b) = wo_b {
            cur = add(&cur, b);
        }
        cur
    }

    pub fn build_attn_inp_k(&self) -> Rc<RefCell<GraphInputAttnK>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let mut inp = GraphInputAttnK::new(&self.hparams, &self.cparams, mctx.clone());
        inp.self_k_idxs = Some(mctx.build_input_k_idxs(&self.ubatch));
        inp.self_attn_mask.raw = Some(build_attn_inp_kq_mask(
            mctx.as_ref(), &self.ubatch, &self.cparams, false,
        ));
        self.res.borrow_mut().add_input(inp)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_attn_k(
        &self,
        inp: &GraphInputAttnK,
        wo: Option<&Tensor>,
        wo_b: Option<&Tensor>,
        wo_s: Option<&Tensor>,
        q_cur: &Tensor,
        k_cur: &Tensor,
        v_cur: &Tensor,
        kq_b: Option<&Tensor>,
        sinks: &Tensor,
        v_mla: Option<&Tensor>,
        kq_scale: f32,
        il: i32,
    ) -> Tensor {
        self.expand(q_cur);
        self.expand(v_cur);
        self.expand(k_cur);
        let mctx_cur = inp.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let k_idxs = inp.get_k_idxs().unwrap_or_else(|| new_tensor_4d(Type::I64, [1, 1, 1, 1]));
        self.expand(&mctx_cur.cpy_k(k_cur, &k_idxs, il));

        let kq_mask = inp.self_attn_mask.raw.as_ref().unwrap_or(k_cur);
        let kv_rows = inp.self_attn_mask.rows.as_ref();
        let q = q_cur.clone();
        let k = mctx_cur.get_k(il);
        let v = view_4d(
            &k,
            [v_cur.ne(0), k.ne(1), k.ne(2), k.ne(3)],
            k.nb(1), k.nb(2), k.nb(3), 0,
        );
        let mut cur = self.build_attn_mha(
            &q, &k, &v, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il, None, None,
        );
        self.cb(&cur, "kqv_out", il);
        if let Some(wo) = wo {
            cur = self.build_lora_mm(wo, &cur, wo_s);
        }
        if let Some(b) = wo_b {
            cur = add(&cur, b);
        }
        cur
    }

    pub fn build_fused_top_k_mask(&self, kq_mask: &Tensor, top_k: &Tensor, il: i32) -> Tensor {
        let top_k = reshape_4d(top_k, [top_k.ne(0), kq_mask.ne(1), kq_mask.ne(2), kq_mask.ne(3)]);
        let cur = ggml::top_k_mask(kq_mask, &top_k, None, None, None);
        self.res.borrow_mut().add_fused_node(GraphFusedNode {
            op: FusedOp::TopKMask,
            tensor: cur.clone(),
            il,
        });
        cur
    }

    pub fn build_top_k_mask(&self, kq_mask: &Tensor, top_k: &Tensor, il: i32) -> Tensor {
        if self.cparams.fused_top_k_mask {
            return self.build_fused_top_k_mask(kq_mask, top_k, il);
        }
        let mut kq_mask_all = fill(kq_mask, f32::NEG_INFINITY);
        kq_mask_all = view_4d(
            &kq_mask_all,
            [1, kq_mask_all.ne(0), kq_mask_all.ne(1), kq_mask_all.ne(3)],
            kq_mask_all.nb(0),
            kq_mask_all.nb(1),
            kq_mask_all.nb(2),
            0,
        );
        let top_k_3d = view_4d(
            top_k,
            [top_k.ne(0), top_k.ne(1), top_k.ne(3), 1],
            top_k.nb(1),
            top_k.nb(2),
            top_k.ne(3) as usize * top_k.nb(3),
            0,
        );
        let mut zeros = new_tensor_4d(
            Type::F32,
            [1, top_k_3d.ne(0), top_k_3d.ne(1), top_k_3d.ne(2)],
        );
        zeros = fill(&zeros, 0.0);
        let mut kq_mask_top_k = set_rows(&kq_mask_all, &zeros, &top_k_3d);
        kq_mask_top_k = view_4d(
            &kq_mask_top_k,
            [kq_mask_top_k.ne(1), kq_mask_top_k.ne(2), 1, kq_mask_top_k.ne(3)],
            kq_mask_top_k.nb(2),
            kq_mask_top_k.nb(3),
            kq_mask_top_k.nb(3),
            0,
        );
        add(&kq_mask_top_k, kq_mask)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_attn_dsa(
        &self,
        inp: &GraphInputAttnKDsa,
        wo: Option<&Tensor>,
        wo_b: Option<&Tensor>,
        wo_s: Option<&Tensor>,
        q_cur: &Tensor,
        k_cur: Option<&Tensor>,
        v_cur: Option<&Tensor>,
        kq_b: Option<&Tensor>,
        sinks: &Tensor,
        v_mla: Option<&Tensor>,
        top_k: &Tensor,
        kq_scale: f32,
        il: i32,
    ) -> Tensor {
        self.expand(q_cur);
        if let Some(v) = v_cur { self.expand(v); }
        if let Some(k) = k_cur { self.expand(k); }

        let mctx = inp.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let mctx_mla = mctx.get_mla().unwrap_or_else(|| mctx.clone());
        if let Some(k) = k_cur {
            let k_idxs = inp.get_k_idxs_mla().unwrap_or_else(|| new_tensor_4d(Type::I64, [1, 1, 1, 1]));
            self.expand(&mctx_mla.cpy_k(k, &k_idxs, il));
        }

        let mut kq_mask = inp.self_attn_mask_mla.raw.clone().unwrap_or_else(|| q_cur.clone());
        if let Some(lid) = &inp.self_attn_mask_lid.raw {
            if lid.ty() == kq_mask.ty() {
                kq_mask = lid.clone();
            }
        }
        let kq_mask_top_k = self.build_top_k_mask(&kq_mask, top_k, il);

        let q = q_cur.clone();
        let k = mctx_mla.get_k(il);
        let n_embd_v = v_cur.map(|v| v.ne(0)).unwrap_or_else(|| v_mla.map(|v| v.ne(0)).unwrap_or(1));
        let v = view_4d(
            &k,
            [n_embd_v, k.ne(1), k.ne(2), k.ne(3)],
            k.nb(1), k.nb(2), k.nb(3), 0,
        );
        let mut cur = self.build_attn_mha(
            &q, &k, &v, kq_b, &kq_mask_top_k, None, sinks, v_mla,
            top_k.ne(0), kq_scale, il, None, None,
        );
        self.cb(&cur, "kqv_out", il);
        if let Some(wo) = wo {
            cur = self.build_lora_mm(wo, &cur, wo_s);
        }
        if let Some(b) = wo_b {
            cur = add(&cur, b);
        }
        cur
    }

    /// `build_attn_iswa_impl`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_attn_iswa_impl(
        &self,
        inp: &GraphInputAttnKvIswa,
        wo: Option<&Tensor>,
        wo_b: Option<&Tensor>,
        wo_s: Option<&Tensor>,
        q_cur: &Tensor,
        k_cur: Option<&Tensor>,
        v_cur: Option<&Tensor>,
        kq_b: Option<&Tensor>,
        kq_mask: &Tensor,
        kv_rows: Option<&Tensor>,
        sinks: &Tensor,
        v_mla: Option<&Tensor>,
        kq_scale: f32,
        il: i32,
        bias_table: Option<&Tensor>,
        bias_indices: Option<&Tensor>,
    ) -> Tensor {
        let is_swa = self.hparams.is_swa(il.max(0) as u32).unwrap_or(false);
        let k_rot = if is_swa { inp.self_k_rot_swa.get(il as usize) } else { inp.self_k_rot.get(il as usize) }
            .cloned().flatten();
        let v_rot = if is_swa { inp.self_v_rot_swa.get(il as usize) } else { inp.self_v_rot.get(il as usize) }
            .cloned().flatten();

        let mut q = q_cur.clone();
        let mut k_opt = k_cur.cloned();
        let mut v_opt = v_cur.cloned();
        if let Some(rot) = &k_rot {
            q = super::ggml::mul_mat(&q, rot);
            if let Some(k) = &k_opt { k_opt = Some(super::ggml::mul_mat(k, rot)); }
        }
        if let Some(rot) = &v_rot {
            if let Some(v) = &v_opt { v_opt = Some(super::ggml::mul_mat(v, rot)); }
        }

        let mctx_iswa = inp.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let mctx_cur = if is_swa { mctx_iswa.get_swa() } else { mctx_iswa.get_base() }
            .unwrap_or_else(|| mctx_iswa.clone());
        scale_kv_inputs(mctx_cur.as_ref(), &mut q, k_opt.as_mut(), v_opt.as_mut(), il);

        self.expand(&q);
        if let Some(k) = &k_opt { self.expand(k); }
        if let Some(v) = &v_opt { self.expand(v); }

        let use_kv_cur = self.cparams.training && k_opt.is_some() && v_opt.is_some();
        let store_kv = !use_kv_cur || self.hparams.n_layer_kv_from_start >= 0;
        if store_kv {
            if let Some(k) = &k_opt {
                let idxs = if is_swa { inp.get_k_idxs_swa() } else { inp.get_k_idxs() }
                    .unwrap_or_else(|| new_tensor_4d(Type::I64, [1, 1, 1, 1]));
                self.expand(&mctx_cur.cpy_k(k, &idxs, il));
            }
            if let Some(v) = &v_opt {
                let idxs = if is_swa { inp.get_v_idxs_swa() } else { inp.get_v_idxs() }
                    .unwrap_or_else(|| new_tensor_4d(Type::I64, [1, 1, 1, 1]));
                self.expand(&mctx_cur.cpy_v(v, &idxs, il));
            }
        }

        let k = if use_kv_cur { k_opt.clone().unwrap() } else { mctx_cur.get_k(il) };
        let v = if use_kv_cur { v_opt.clone().unwrap() } else { mctx_cur.get_v(il) };
        let mut cur = self.build_attn_mha(
            &q, &k, &v, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il, bias_table, bias_indices,
        );
        cur = scale_kv_output(mctx_cur.as_ref(), &cur, il);
        self.cb(&cur, "kqv_out", il);
        if let Some(rot) = v_rot {
            cur = super::ggml::mul_mat(&cur, &rot);
        }
        if let Some(wo) = wo {
            cur = self.build_lora_mm(wo, &cur, wo_s);
        }
        if let Some(b) = wo_b {
            cur = add(&cur, b);
        }
        cur
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_attn_kv_iswa(
        &self,
        inp: &GraphInputAttnKvIswa,
        wo: Option<&Tensor>,
        wo_b: Option<&Tensor>,
        wo_s: Option<&Tensor>,
        q_cur: &Tensor,
        k_cur: Option<&Tensor>,
        v_cur: Option<&Tensor>,
        kq_b: Option<&Tensor>,
        sinks: &Tensor,
        v_mla: Option<&Tensor>,
        kq_scale: f32,
        il: i32,
    ) -> Tensor {
        let is_swa = self.hparams.is_swa(il.max(0) as u32).unwrap_or(false);
        let mask = if is_swa { &inp.self_attn_mask_swa } else { &inp.self_attn_mask };
        let kq_mask = mask.raw.as_ref().unwrap_or(q_cur);
        let kv_rows = mask.rows.as_ref();
        self.build_attn_iswa_impl(
            inp, wo, wo_b, wo_s, q_cur, k_cur, v_cur, kq_b, kq_mask, kv_rows, sinks, v_mla,
            kq_scale, il, None, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_attn_k_iswa(
        &self,
        inp: &GraphInputAttnKIswa,
        wo: Option<&Tensor>,
        wo_b: Option<&Tensor>,
        wo_s: Option<&Tensor>,
        q_cur: &Tensor,
        k_cur: Option<&Tensor>,
        _v_cur: Option<&Tensor>,
        kq_b: Option<&Tensor>,
        sinks: &Tensor,
        v_mla: Option<&Tensor>,
        kq_scale: f32,
        il: i32,
    ) -> Tensor {
        let is_swa = self.hparams.is_swa(il.max(0) as u32).unwrap_or(false);
        let k_rot = if is_swa { inp.self_k_rot_swa.get(il as usize) } else { inp.self_k_rot.get(il as usize) }
            .cloned().flatten();
        let mut q = q_cur.clone();
        let mut k_opt = k_cur.cloned();
        if let Some(rot) = &k_rot {
            q = super::ggml::mul_mat(&q, rot);
            if let Some(k) = &k_opt { k_opt = Some(super::ggml::mul_mat(k, rot)); }
        }
        self.expand(&q);
        if let Some(k) = &k_opt { self.expand(k); }
        let mctx_iswa = inp.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let mctx_cur = if is_swa { mctx_iswa.get_swa() } else { mctx_iswa.get_base() }
            .unwrap_or_else(|| mctx_iswa.clone());
        if let Some(k) = &k_opt {
            let idxs = if is_swa { inp.get_k_idxs_swa() } else { inp.get_k_idxs() }
                .unwrap_or_else(|| new_tensor_4d(Type::I64, [1, 1, 1, 1]));
            self.expand(&mctx_cur.cpy_k(k, &idxs, il));
        }
        let mask = if is_swa { &inp.self_attn_mask_swa } else { &inp.self_attn_mask };
        let kq_mask = mask.raw.as_ref().unwrap_or(&q);
        let kv_rows = mask.rows.as_ref();
        let k = mctx_cur.get_k(il);
        let v = view_4d(
            &k,
            [q.ne(0), k.ne(1), k.ne(2), k.ne(3)],
            k.nb(1), k.nb(2), k.nb(3), 0,
        );
        let mut cur = self.build_attn_mha(
            &q, &k, &v, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il, None, None,
        );
        self.cb(&cur, "kqv_out", il);
        if let Some(rot) = k_rot {
            cur = super::ggml::mul_mat(&cur, &rot);
        }
        if let Some(wo) = wo {
            cur = self.build_lora_mm(wo, &cur, wo_s);
        }
        if let Some(b) = wo_b {
            cur = add(&cur, b);
        }
        cur
    }

    pub fn build_attn_inp_cross(&self) -> Rc<RefCell<GraphInputAttnCross>> {
        let mut inp = GraphInputAttnCross::new(self.cross.clone().unwrap_or_default());
        let n_enc = self.cross.as_ref().map(|c| {
            if !c.v_embd.is_empty() { c.n_enc } else { self.hparams.n_ctx_train as i64 }
        }).unwrap_or(self.hparams.n_ctx_train as i64);
        let ty = if self.cparams.flash_attn { Type::F16 } else { Type::F32 };
        let raw = new_tensor_4d(ty, [n_enc, self.n_tokens, 1, 1]);
        raw.set_input();
        inp.cross_attn_mask.raw = Some(raw);
        self.res.borrow_mut().add_input(inp)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_attn_cross(
        &self,
        inp: &GraphInputAttnCross,
        wo: Option<&Tensor>,
        wo_b: Option<&Tensor>,
        wo_s: Option<&Tensor>,
        q_cur: &Tensor,
        k_cur: &Tensor,
        v_cur: &Tensor,
        kq_b: Option<&Tensor>,
        sinks: &Tensor,
        v_mla: Option<&Tensor>,
        kq_scale: f32,
        il: i32,
    ) -> Tensor {
        self.expand(q_cur);
        self.expand(k_cur);
        self.expand(v_cur);
        let kq_mask = inp.cross_attn_mask.raw.as_ref().unwrap_or(k_cur);
        let kv_rows = inp.cross_attn_mask.rows.as_ref();
        let mut cur = self.build_attn_mha(
            q_cur, k_cur, v_cur, kq_b, kq_mask, kv_rows, sinks, v_mla, 0, kq_scale, il, None, None,
        );
        self.cb(&cur, "kqv_out", il);
        if let Some(wo) = wo {
            cur = self.build_lora_mm(wo, &cur, wo_s);
        }
        if let Some(b) = wo_b {
            cur = add(&cur, b);
        }
        cur
    }

    pub(crate) fn build_attn_inp_k_dsa_impl(
        &self,
        mctx: Rc<dyn MemoryContext>,
    ) -> Rc<RefCell<GraphInputAttnKDsa>> {
        let mut inp = GraphInputAttnKDsa::new(&self.hparams, &self.cparams, mctx.clone());
        let mla = mctx.get_mla().unwrap_or_else(|| mctx.clone());
        let lid = mctx.get_lid().unwrap_or_else(|| mctx.clone());
        inp.self_k_idxs_mla = Some(mla.build_input_k_idxs(&self.ubatch));
        inp.self_attn_mask_mla.raw = Some(build_attn_inp_kq_mask(
            mla.as_ref(), &self.ubatch, &self.cparams, false,
        ));
        inp.self_k_idxs_lid = Some(lid.build_input_k_idxs(&self.ubatch));
        inp.self_attn_mask_lid.raw = Some(build_attn_inp_kq_mask(
            lid.as_ref(), &self.ubatch, &self.cparams, true,
        ));
        inp.self_k_rot_lid = lid.get_k_rot();
        self.res.borrow_mut().add_input(inp)
    }

    pub fn build_attn_inp_k_dsa(&self) -> Rc<RefCell<GraphInputAttnKDsa>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        self.build_attn_inp_k_dsa_impl(mctx)
    }

    pub fn build_attn_inp_k_dsa_iswa(&self) -> Rc<RefCell<GraphInputAttnKDsaIswa>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let dsa_ctx = mctx.get_dsa().unwrap_or_else(|| mctx.clone());
        let swa_ctx = mctx.get_swa().unwrap_or_else(|| mctx.clone());
        let inp_dsa = self.build_attn_inp_k_dsa_impl(dsa_ctx);

        let mut inp_swa = GraphInputAttnK::new(&self.hparams, &self.cparams, swa_ctx.clone());
        inp_swa.self_k_idxs = Some(swa_ctx.build_input_k_idxs(&self.ubatch));
        inp_swa.self_attn_mask.raw = Some(build_attn_inp_kq_mask(
            swa_ctx.as_ref(), &self.ubatch, &self.cparams, false,
        ));
        let inp_swa = self.res.borrow_mut().add_input(inp_swa);

        let inp = GraphInputAttnKDsaIswa::new(inp_dsa, inp_swa, mctx);
        self.res.borrow_mut().add_input(inp)
    }

    pub fn build_attn_inp_kv_msa(&self, msa_enabled: bool) -> Rc<RefCell<GraphInputAttnKvMsa>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let mut inp = GraphInputAttnKvMsa::new(&self.hparams, &self.cparams, mctx.clone());
        let base = mctx.get_base().unwrap_or_else(|| mctx.clone());
        let idx = mctx.get_idx().unwrap_or_else(|| mctx.clone());
        inp.base.self_k_idxs = Some(base.build_input_k_idxs(&self.ubatch));
        inp.base.self_v_idxs = Some(base.build_input_v_idxs(&self.ubatch));
        inp.base.self_attn_mask.raw = Some(build_attn_inp_kq_mask(
            base.as_ref(), &self.ubatch, &self.cparams, false,
        ));
        inp.base.self_k_rot = base.get_k_rot();
        inp.base.self_v_rot = base.get_v_rot();
        if msa_enabled {
            inp.self_k_idxs_idx = Some(idx.build_input_k_idxs(&self.ubatch));
        }
        self.res.borrow_mut().add_input(inp)
    }

    pub fn build_attn_inp_kv_iswa(&self) -> Rc<RefCell<GraphInputAttnKvIswa>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let base = mctx.get_base().unwrap_or_else(|| mctx.clone());
        let swa = mctx.get_swa().unwrap_or_else(|| mctx.clone());
        let mut inp = GraphInputAttnKvIswa::new(&self.hparams, &self.cparams, mctx);

        inp.self_k_idxs = Some(base.build_input_k_idxs(&self.ubatch));
        inp.self_v_idxs = Some(base.build_input_v_idxs(&self.ubatch));
        inp.self_attn_mask = build_attn_inp_attn_mask(base.as_ref(), &self.ubatch, &self.cparams);

        inp.self_k_idxs_swa = Some(swa.build_input_k_idxs(&self.ubatch));
        inp.self_v_idxs_swa = Some(swa.build_input_v_idxs(&self.ubatch));
        inp.self_attn_mask_swa = build_attn_inp_attn_mask(swa.as_ref(), &self.ubatch, &self.cparams);

        inp.self_k_rot = base.get_k_rot();
        inp.self_v_rot = base.get_v_rot();
        inp.self_k_rot_swa = swa.get_k_rot();
        inp.self_v_rot_swa = swa.get_v_rot();
        self.res.borrow_mut().add_input(inp)
    }

    pub fn build_attn_inp_k_iswa(&self) -> Rc<RefCell<GraphInputAttnKIswa>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let base = mctx.get_base().unwrap_or_else(|| mctx.clone());
        let swa = mctx.get_swa().unwrap_or_else(|| mctx.clone());
        let mut inp = GraphInputAttnKIswa::new(&self.hparams, &self.cparams, mctx);
        inp.self_k_idxs = Some(base.build_input_k_idxs(&self.ubatch));
        inp.self_attn_mask.raw = Some(build_attn_inp_kq_mask(
            base.as_ref(), &self.ubatch, &self.cparams, false,
        ));
        inp.self_k_idxs_swa = Some(swa.build_input_k_idxs(&self.ubatch));
        inp.self_attn_mask_swa.raw = Some(build_attn_inp_kq_mask(
            swa.as_ref(), &self.ubatch, &self.cparams, false,
        ));
        inp.self_k_rot = base.get_k_rot();
        inp.self_k_rot_swa = swa.get_k_rot();
        self.res.borrow_mut().add_input(inp)
    }

    pub fn build_inp_dsv4(&self) -> Rc<RefCell<GraphInputDsv4>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let raw_ctx = mctx.get_raw().unwrap_or_else(|| mctx.clone());
        let mut inp_raw = GraphInputDsv4Raw::new(&self.cparams, raw_ctx.clone());
        let n_stream = mctx.get_csa_plan(&self.ubatch).n_stream;
        inp_raw.self_k_idxs = Some(raw_ctx.build_input_k_idxs(&self.ubatch));
        inp_raw.self_attn_mask.raw = Some(build_raw_kq_mask(
            raw_ctx.as_ref(), &self.ubatch, &self.cparams, n_stream,
        ));
        inp_raw.self_k_rot = raw_ctx.get_k_rot();
        let inp_raw = self.res.borrow_mut().add_input(inp_raw);

        let mut inp = GraphInputDsv4::new(&self.cparams, inp_raw, mctx.clone());
        for (field, plan, name) in [
            (&mut inp.inp_csa, mctx.get_csa_plan(&self.ubatch), "csa"),
            (&mut inp.inp_hca, mctx.get_hca_plan(&self.ubatch), "hca"),
            (&mut inp.inp_lid, mctx.get_lid_plan(&self.ubatch), "lid"),
        ] {
            build_comp_inputs(field, &plan, name, &self.cparams, n_stream);
        }
        self.res.borrow_mut().add_input(inp)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn scale_kv_inputs(
    mctx: &dyn MemoryContext,
    q: &mut Tensor,
    k: Option<&mut Tensor>,
    v: Option<&mut Tensor>,
    il: i32,
) {
    if let Some(scale) = mctx.get_k_scale(il) {
        *q = mul(q, &scale);
        if let Some(k) = k {
            *k = div(k, &scale);
        }
    }
    if let Some(v) = v {
        if let Some(scale) = mctx.get_v_scale(il) {
            *v = div(v, &scale);
        }
    }
}

fn scale_kv_output(mctx: &dyn MemoryContext, cur: &Tensor, il: i32) -> Tensor {
    match mctx.get_v_scale(il) {
        Some(scale) => mul(cur, &scale),
        None => cur.clone(),
    }
}


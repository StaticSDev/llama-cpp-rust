//! Common build methods: LoRA, Hadamard folds, norms, QKV and dense FFN.

use super::context::GraphContext;
use super::ggml::{
    self, add, clamp, cont, div, get_rows, mul, mul_mat, mul_mat_id, new_tensor_1d,
    repeat_4d, reshape_2d, reshape_3d, reshape_4d, scale, scale_bias, Tensor, Type,
};
use super::support::Layer;
use super::types::{FfnGateType, FfnOpType, GraphAmaxSrc1, GraphQkv, NormType};
use crate::arch::{arch_mmid_down_bf16, Arch};

// `identity as _` keeps the op list readable; removed below via a local helper.
fn hadamard_mul_mat(x: &Tensor, rot: &Tensor) -> Tensor {
    ggml::mul_mat(x, rot)
}

impl GraphContext {
    /// `build_hadamard_input`: apply the activation-side transform of a
    /// Hadamard-folded weight, if any.
    pub fn build_hadamard_input(&self, w: &Tensor, cur: &Tensor) -> Tensor {
        let Some(hdmd) = self.model.hdmd.as_ref() else { return cur.clone() };
        let Some(entry) = hdmd.get_rot(w) else { return cur.clone() };
        if let Some(x) = self.res.borrow().get_hdmd_input(cur, &entry.rot) {
            return x;
        }
        let mut x = cur.clone();
        if entry.perm_rep > 1 {
            x = if x.is_contiguous() { x } else { cont(&x) };
            let ne1 = x.ne(1);
            let ne2 = x.ne(2);
            let ne3 = x.ne(3);
            x = reshape_4d(&x, [entry.perm_hd as i64, entry.perm_nk as i64, entry.perm_rep as i64, ne1 * ne2 * ne3]);
            x = cont(&ggml::permute(&x, 0, 2, 1, 3));
            x = reshape_4d(
                &x,
                [(entry.perm_hd * entry.perm_nk * entry.perm_rep) as i64, ne1, ne2, ne3],
            );
        }
        if let Some(signs) = &entry.signs {
            x = mul(&x, signs);
        }
        x = hadamard_mul_mat(&x, &entry.rot);
        self.res.borrow_mut().set_hdmd_input(cur, &entry.rot, x.clone());
        x
    }

    /// `build_lora_mm`: mat_mul with optional LoRA deltas and weight scale.
    pub fn build_lora_mm(&self, w: &Tensor, cur: &Tensor, w_s: Option<&Tensor>) -> Tensor {
        let w_s = w_s.cloned().or_else(|| self.model.get_w_scale(w));
        let cur_mm = self.build_hadamard_input(w, cur);
        let mut res = mul_mat(w, &cur_mm);
        if w.ty() == Type::NVFP4 {
            ggml::prec_set_acc(&res, ggml::Prec::BF16);
        }
        self.prec_policy.apply(&res, w);
        if let Some(w_s) = &w_s {
            res = mul(&res, w_s);
        }
        if let Some(loras) = &self.loras {
            for (lora, adapter_scale) in loras.iter() {
                let Some(lw) = lora.get_weight(w) else { continue };
                let factor = lw.get_scale(lora.alpha, *adapter_scale);
                let ab_cur = mul_mat(&lw.b, &mul_mat(&lw.a, cur));
                let ab_cur = scale(&ab_cur, factor);
                res = add(&res, &ab_cur);
            }
        }
        res
    }

    /// `build_amax_src1_f16`.
    pub fn build_amax_src1_f16(&self, cur: &Tensor) -> GraphAmaxSrc1 {
        let mut result = GraphAmaxSrc1 { res: Some(cur.clone()), scale_s: None };
        let n_embd = cur.ne(0);
        let ab = ggml::abs(cur);
        let amax = ggml::pool_1d(&ab, ggml::Op::None, n_embd, n_embd, 0);

        const K_LOG2_E: f32 = 1.4426950408889634;
        const K_LN_2: f32 = 0.6931471805599453;

        let mut e = scale(&amax, 1.0 / 1024.0);
        e = ggml::log(&e);
        e = scale(&e, K_LOG2_E);
        e = ggml::ceil(&e);
        e = clamp(&e, 0.0, 127.0);
        e = scale(&e, K_LN_2);
        e = ggml::exp(&e);

        result.res = Some(div(cur, &e));
        result.scale_s = Some(e);
        result
    }

    /// `build_lora_mm_id`: expert mat_mul with per-expert scale and LoRA.
    pub fn build_lora_mm_id(
        &self,
        w: &Tensor,
        cur: &Tensor,
        ids: &Tensor,
        slots: Option<&Tensor>,
        cached: bool,
    ) -> Tensor {
        let w_s = self.model.get_w_scale(w);
        let cur_mm = self.build_hadamard_input(w, cur);
        let experts = match slots {
            None => w.clone(),
            Some(_) if cached => self
                .moe_cache
                .as_ref()
                .and_then(|cache| cache.get_experts(w))
                .unwrap_or_else(|| w.clone()),
            Some(_) => w.clone(),
        };
        let mut res = match slots {
            None => mul_mat_id(w, &cur_mm, ids),
            Some(slots) => mul_mat_id(&experts, &cur_mm, slots),
        };
        if w.ty() == Type::NVFP4 {
            ggml::prec_set_acc(&res, ggml::Prec::BF16);
        }
        self.prec_policy.apply(&res, w);

        if let Some(w_s) = &w_s {
            let n_expert = w_s.ne(0);
            let n_tokens = cur.ne(2);
            let s = reshape_3d(w_s, 1, n_expert, 1);
            let s = ggml::repeat_4d(&s, 1, n_expert, n_tokens, 1);
            let s = get_rows(&s, ids);
            res = mul(&res, &s);
        }
        if let Some(loras) = &self.loras {
            for (lora, adapter_scale) in loras.iter() {
                let Some(lw) = lora.get_weight(w) else { continue };
                let alpha = lora.alpha;
                let rank = lw.b.ne(0) as f32;
                let factor = if alpha != 0.0 { *adapter_scale * alpha / rank } else { *adapter_scale };
                let ab_cur = mul_mat_id(&lw.b, &mul_mat_id(&lw.a, cur, ids), ids);
                let ab_cur = scale(&ab_cur, factor);
                res = add(&res, &ab_cur);
            }
        }
        res
    }

    /// `build_norm`.
    pub fn build_norm(
        &self,
        cur: &Tensor,
        mw: Option<&Tensor>,
        mb: Option<&Tensor>,
        norm_type: NormType,
        il: i32,
    ) -> Tensor {
        let mut cur = match norm_type {
            NormType::Norm => ggml::norm(cur, self.hparams.f_norm_eps),
            NormType::NormRms => ggml::rms_norm(cur, self.hparams.f_norm_rms_eps),
            NormType::NormGroup => {
                let ne0 = cur.ne(0);
                let ne1 = cur.ne(1);
                let mut x = reshape_3d(cur, ne0, 1, ne1);
                x = ggml::group_norm(&x, self.hparams.n_norm_groups as i64, self.hparams.f_norm_group_eps);
                let ne0 = x.ne(0);
                let ne2 = x.ne(2);
                reshape_2d(&x, ne0, ne2)
            }
        };
        if mw.is_some() || mb.is_some() {
            self.cb(&cur, "norm", il);
        }
        if let Some(mw) = mw {
            cur = mul(&cur, mw);
            if mb.is_some() {
                self.cb(&cur, "norm_w", il);
            }
        }
        if let Some(mb) = mb {
            cur = add(&cur, mb);
        }
        cur
    }

    /// `build_qkv` simple overload: one head size shared by K and V.
    pub fn build_qkv_simple(
        &self,
        layer: &Layer,
        cur: &Tensor,
        n_embd_head: i64,
        n_head: i64,
        n_head_kv: i64,
        il: i32,
    ) -> GraphQkv {
        self.build_qkv(
            layer,
            cur,
            n_embd_head, n_head,
            n_embd_head, n_head_kv,
            n_embd_head, n_head_kv,
            il,
            true,
        )
    }

    /// `build_qkv` general overload with separate Q/K/V head sizes.
    pub fn build_qkv(
        &self,
        layer: &Layer,
        cur: &Tensor,
        n_embd_head_q: i64,
        n_head_q: i64,
        n_embd_head_k: i64,
        n_head_k: i64,
        n_embd_head_v: i64,
        n_head_v: i64,
        il: i32,
        reshape: bool,
    ) -> GraphQkv {
        let n_embd_q = n_embd_head_q * n_head_q;
        let n_embd_k = n_embd_head_k * n_head_k;
        let (mut q_cur, mut k_cur, mut v_cur);

        if let Some(wqkv) = &layer.wqkv {
            let mut qkv = self.build_lora_mm(wqkv, cur, layer.wqkv_s.as_ref());
            self.cb(&qkv, "wqkv", il);
            if let Some(b) = &layer.wqkv_b {
                qkv = add(&qkv, b);
                self.cb(&qkv, "wqkv_b", il);
            } else if let (Some(wq_b), Some(wk_b), Some(wv_b)) =
                (&layer.wq_b, &layer.wk_b, &layer.wv_b)
            {
                let qkv_b = ggml::concat(&ggml::concat(wq_b, wk_b, 0), wv_b, 0);
                qkv = add(&qkv, &qkv_b);
                self.cb(&qkv, "wqkv_b", il);
            }
            if reshape && self.hparams.f_clamp_kqv > 0.0 {
                qkv = clamp(&qkv, -self.hparams.f_clamp_kqv, self.hparams.f_clamp_kqv);
                self.cb(&qkv, "wqkv_clamped", il);
            }
            let ty = qkv.ty();
            if reshape {
                q_cur = ggml::view_3d(&qkv, n_embd_head_q, n_head_q, self.n_tokens, ty.row_size(n_embd_head_q), qkv.nb(1), 0);
                k_cur = ggml::view_3d(&qkv, n_embd_head_k, n_head_k, self.n_tokens, ty.row_size(n_embd_head_k), qkv.nb(1), ty.row_size(n_embd_q));
                v_cur = ggml::view_3d(&qkv, n_embd_head_v, n_head_v, self.n_tokens, ty.row_size(n_embd_head_v), qkv.nb(1), ty.row_size(n_embd_q + n_embd_k));
            } else {
                q_cur = ggml::view_2d(&qkv, n_embd_q, self.n_tokens, qkv.nb(1), 0);
                k_cur = ggml::view_2d(&qkv, n_embd_k, self.n_tokens, qkv.nb(1), ty.row_size(n_embd_q));
                v_cur = ggml::view_2d(&qkv, n_embd_head_v * n_head_v, self.n_tokens, qkv.nb(1), ty.row_size(n_embd_q + n_embd_k));
                q_cur = cont(&q_cur);
                k_cur = cont(&k_cur);
                v_cur = cont(&v_cur);
            }
        } else {
            let wq = layer.wq.clone().unwrap_or_else(|| new_tensor_1d(Type::F32, 1));
            let wk = layer.wk.clone().unwrap_or_else(|| new_tensor_1d(Type::F32, 1));
            let wv = layer.wv.clone().unwrap_or_else(|| new_tensor_1d(Type::F32, 1));
            q_cur = self.build_lora_mm(&wq, cur, layer.wq_s.as_ref());
            if reshape { self.cb(&q_cur, "Qcur", il); }
            if let Some(b) = &layer.wq_b {
                q_cur = add(&q_cur, b);
                if reshape { self.cb(&q_cur, "Qcur", il); }
            }
            if reshape && self.hparams.f_clamp_kqv > 0.0 {
                q_cur = clamp(&q_cur, -self.hparams.f_clamp_kqv, self.hparams.f_clamp_kqv);
                self.cb(&q_cur, "Qcur_clamped", il);
            }
            k_cur = self.build_lora_mm(&wk, cur, layer.wk_s.as_ref());
            if reshape { self.cb(&k_cur, "Kcur", il); }
            if let Some(b) = &layer.wk_b {
                k_cur = add(&k_cur, b);
                if reshape { self.cb(&k_cur, "Kcur", il); }
            }
            if reshape && self.hparams.f_clamp_kqv > 0.0 {
                k_cur = clamp(&k_cur, -self.hparams.f_clamp_kqv, self.hparams.f_clamp_kqv);
                self.cb(&k_cur, "Kcur_clamped", il);
            }
            v_cur = self.build_lora_mm(&wv, cur, layer.wv_s.as_ref());
            if reshape { self.cb(&v_cur, "Vcur", il); }
            if let Some(b) = &layer.wv_b {
                v_cur = add(&v_cur, b);
                if reshape { self.cb(&v_cur, "Vcur", il); }
            }
            if reshape && self.hparams.f_clamp_kqv > 0.0 {
                v_cur = clamp(&v_cur, -self.hparams.f_clamp_kqv, self.hparams.f_clamp_kqv);
                self.cb(&v_cur, "Vcur_clamped", il);
            }
            if reshape {
                q_cur = reshape_3d(&q_cur, n_embd_head_q, n_head_q, self.n_tokens);
                k_cur = reshape_3d(&k_cur, n_embd_head_k, n_head_k, self.n_tokens);
                v_cur = reshape_3d(&v_cur, n_embd_head_v, n_head_v, self.n_tokens);
            }
        }

        if reshape {
            self.cb(&q_cur, "Qcur", il);
            self.cb(&k_cur, "Kcur", il);
            self.cb(&v_cur, "Vcur", il);
        }
        GraphQkv { q: q_cur, k: k_cur, v: v_cur }
    }

    /// `build_ffn`: dense feed-forward with an optional gate.
    #[allow(clippy::too_many_arguments)]
    pub fn build_ffn(
        &self,
        cur: &Tensor,
        up: Option<&Tensor>,
        up_b: Option<&Tensor>,
        up_s: Option<&Tensor>,
        gate: Option<&Tensor>,
        gate_b: Option<&Tensor>,
        gate_s: Option<&Tensor>,
        down: Option<&Tensor>,
        down_b: Option<&Tensor>,
        down_s: Option<&Tensor>,
        act_scales: Option<&Tensor>,
        type_op: FfnOpType,
        mut type_gate: FfnGateType,
        il: i32,
    ) -> Tensor {
        let arch = self.model.arch;
        let mut tmp = match up {
            Some(up) => self.build_lora_mm(up, cur, up_s),
            None => cur.clone(),
        };
        self.cb(&tmp, "ffn_up", il);
        if let Some(b) = up_b {
            tmp = add(&tmp, b);
            self.cb(&tmp, "ffn_up_b", il);
        }

        let mut cur = if let Some(gate) = gate {
            let g = match type_gate {
                FfnGateType::Seq => self.build_lora_mm(gate, &tmp, gate_s),
                FfnGateType::Par => self.build_lora_mm(gate, cur, gate_s),
            };
            self.cb(&g, "ffn_gate", il);
            let g = match gate_b {
                Some(b) => add(&g, b),
                None => g,
            };
            if gate_b.is_some() {
                self.cb(&g, "ffn_gate_b", il);
            }
            g
        } else {
            tmp.clone()
        };

        match type_op {
            FfnOpType::Silu => {
                if gate.is_some() && type_gate == FfnGateType::Par {
                    let special = matches!(arch, Arch::Deepseek4 | Arch::Deepseek41 | Arch::Glm5Next)
                        || (arch == Arch::Dflash && self.hparams.dsv4_hc_mult > 0);
                    if il >= 0 {
                        let limit = self.hparams.swiglu_clamp_shexp.get(il as usize).copied().unwrap_or(0.0);
                        if limit > 1e-6 {
                            if special {
                                cur = ggml::swiglu_clamp(&cur, &tmp, limit);
                            } else {
                                tmp = clamp(&tmp, -limit, limit);
                                self.cb(&tmp, "ffn_up_clamped", il);
                                let mut gate_act = ggml::silu(&cur);
                                self.cb(&gate_act, "ffn_silu", il);
                                gate_act = clamp(&gate_act, f32::NEG_INFINITY, limit);
                                self.cb(&gate_act, "ffn_silu_clamped", il);
                                cur = mul(&gate_act, &tmp);
                            }
                            self.cb(&cur, "ffn_swiglu_limited", il);
                            type_gate = FfnGateType::Seq;
                        }
                    }
                    if type_gate == FfnGateType::Par {
                        cur = ggml::swiglu_split(&cur, &tmp);
                        self.cb(&cur, "ffn_swiglu", il);
                        type_gate = FfnGateType::Seq;
                    }
                } else {
                    cur = ggml::silu(&cur);
                    self.cb(&cur, "ffn_silu", il);
                }
            }
            FfnOpType::Gelu => {
                if gate.is_some() && type_gate == FfnGateType::Par {
                    cur = ggml::geglu_split(&cur, &tmp);
                    self.cb(&cur, "ffn_geglu", il);
                    type_gate = FfnGateType::Seq;
                } else {
                    cur = ggml::gelu(&cur);
                    self.cb(&cur, "ffn_gelu", il);
                    if let Some(scales) = act_scales {
                        cur = div(&cur, scales);
                        self.cb(&cur, "ffn_act", il);
                    }
                }
            }
            FfnOpType::GeluErf => {
                if gate.is_some() && type_gate == FfnGateType::Par {
                    cur = ggml::geglu_erf_split(&cur, &tmp);
                    self.cb(&cur, "ffn_geglu_erf", il);
                    type_gate = FfnGateType::Seq;
                } else {
                    cur = ggml::gelu_erf(&cur);
                    self.cb(&cur, "ffn_gelu_erf", il);
                    if let Some(scales) = act_scales {
                        cur = div(&cur, scales);
                        self.cb(&cur, "ffn_act", il);
                    }
                }
            }
            FfnOpType::Relu => {
                if gate.is_some() && type_gate == FfnGateType::Par {
                    cur = ggml::reglu_split(&cur, &tmp);
                    self.cb(&cur, "ffn_reglu", il);
                    type_gate = FfnGateType::Seq;
                } else {
                    cur = ggml::relu(&cur);
                    self.cb(&cur, "ffn_relu", il);
                }
            }
            FfnOpType::ReluSqr => {
                cur = ggml::relu(&cur);
                self.cb(&cur, "ffn_relu", il);
                cur = ggml::sqr(&cur);
                self.cb(&cur, "ffn_sqr(relu)", il);
            }
            FfnOpType::Swiglu => {
                cur = ggml::swiglu(&cur);
                self.cb(&cur, "ffn_swiglu", il);
            }
            FfnOpType::SwigluOaiMoe => {
                if gate.is_some() && type_gate == FfnGateType::Par {
                    cur = ggml::swiglu_oai(&cur, &tmp, 1.702, 7.0);
                    self.cb(&cur, "ffn_swiglu_oai", il);
                    type_gate = FfnGateType::Seq;
                } else {
                    // C++ aborts here; keep building the fused node so the
                    // topology stays visible for unsupported architectures.
                    cur = ggml::swiglu_oai(&cur, &cur, 1.702, 7.0);
                    self.cb(&cur, "ffn_swiglu_oai", il);
                }
            }
            FfnOpType::Geglu => {
                cur = ggml::geglu(&cur);
                self.cb(&cur, "ffn_geglu", il);
            }
            FfnOpType::GegluErf => {
                cur = ggml::geglu_erf(&cur);
                self.cb(&cur, "ffn_geglu_erf", il);
            }
            FfnOpType::Reglu => {
                cur = ggml::reglu(&cur);
                self.cb(&cur, "ffn_reglu", il);
            }
            FfnOpType::Situ => {
                // C++ aborts on this op in the dense path; the fused node is
                // built here for topology parity.
                cur = ggml::situ(&cur, &tmp, self.hparams.situ_beta, self.hparams.situ_linear_beta);
                self.cb(&cur, "ffn_situ", il);
            }
            FfnOpType::None => {
                // Sentinel: leave the activation untouched.
            }
        }

        if gate.is_some() && type_gate == FfnGateType::Par {
            cur = mul(&cur, &tmp);
            self.cb(&cur, "ffn_gate_par", il);
        }
        if let Some(down) = down {
            cur = self.build_lora_mm(down, &cur, down_s);
        }
        if down_b.is_some() {
            self.cb(&cur, "ffn_down", il);
            cur = add(&cur, down_b.unwrap());
        }
        cur
    }

    /// `llm_arch_mmid_down_bf16` check used by the MoE path.
    pub fn moe_down_bf16(&self) -> bool {
        arch_mmid_down_bf16(self.model.arch)
    }

    /// Shorthand for scale_bias nodes used by the MoE router.
    pub fn scale_bias(&self, t: &Tensor, s: f32, b: f32) -> Tensor {
        scale_bias(t, s, b)
    }

    /// Repeat a tensor to `[ne0, ne1, ne2, 1]`.
    pub fn repeat(&self, t: &Tensor, ne0: i64, ne1: i64, ne2: i64) -> Tensor {
        repeat_4d(t, ne0, ne1, ne2, 1)
    }

    /// Reshape a tensor to 3d.
    pub fn reshape3(&self, t: &Tensor, ne0: i64, ne1: i64, ne2: i64) -> Tensor {
        reshape_3d(t, ne0, ne1, ne2)
    }

    /// Reshape a tensor to 4d.
    pub fn reshape4(&self, t: &Tensor, ne: [i64; 4]) -> Tensor {
        reshape_4d(t, ne)
    }

    /// `ggml_concat` wrapper kept here so `common` owns the dense paths.
    pub fn concat(&self, a: &Tensor, b: &Tensor, dim: i32) -> Tensor {
        ggml::concat(a, b, dim)
    }
}

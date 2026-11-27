//! `build_moe_ffn` and its no-bias overload.

use super::context::GraphContext;
use super::ggml::{
    self, add, add_id, argsort_top_k, clamp, cont, div, fill, get_rows, mul, reshape_2d,
    reshape_3d, reshape_4d, scale, set_rows, soft_max, sqrt, softplus, sum_rows, Tensor, Type,
};
use super::inputs::GraphInputMoeIds;
use super::support::MoeLookupArgs;
use super::types::FfnOpType;
use crate::arch::Arch;
use crate::hparams::ExpertGatingFuncType;

impl GraphContext {
    /// MoE FFN without bias tensors: delegates to the full overload.
    #[allow(clippy::too_many_arguments)]
    pub fn build_moe_ffn(
        &self,
        cur: &Tensor,
        gate_inp: &Tensor,
        up_exps: &Tensor,
        gate_exps: &Tensor,
        down_exps: &Tensor,
        exp_probs_b: Option<&Tensor>,
        n_expert: i64,
        n_expert_used: i64,
        type_op: FfnOpType,
        norm_w: bool,
        w_scale: f32,
        gating_op: ExpertGatingFuncType,
        il: i32,
        probs_in: Option<&Tensor>,
        gate_up_exps: Option<&Tensor>,
        selected_experts_in: Option<&Tensor>,
        post_exp_norm: Option<&Tensor>,
        post_exp_norm_eps: f32,
    ) -> Tensor {
        self.build_moe_ffn_full(
            cur, gate_inp, None,
            up_exps, None, gate_exps, None, down_exps, None,
            exp_probs_b, n_expert, n_expert_used, type_op, norm_w, w_scale, gating_op, il,
            probs_in, gate_up_exps, None, selected_experts_in, post_exp_norm, post_exp_norm_eps,
        )
    }

    /// Full MoE FFN, including all optional bias tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn build_moe_ffn_full(
        &self,
        cur: &Tensor,
        gate_inp: &Tensor,
        gate_inp_b: Option<&Tensor>,
        up_exps: &Tensor,
        up_exps_b: Option<&Tensor>,
        gate_exps: &Tensor,
        gate_exps_b: Option<&Tensor>,
        down_exps: &Tensor,
        down_exps_b: Option<&Tensor>,
        exp_probs_b: Option<&Tensor>,
        n_expert: i64,
        n_expert_used: i64,
        type_op: FfnOpType,
        norm_w: bool,
        w_scale: f32,
        gating_op: ExpertGatingFuncType,
        il: i32,
        probs_in: Option<&Tensor>,
        gate_up_exps: Option<&Tensor>,
        gate_up_exps_b: Option<&Tensor>,
        selected_experts_in: Option<&Tensor>,
        post_exp_norm: Option<&Tensor>,
        post_exp_norm_eps: f32,
    ) -> Tensor {
        let arch = self.model.arch;
        let n_embd = cur.ne(0);
        let n_tokens = cur.ne(1);
        let weight_before_ffn = arch == Arch::Llama4;

        let mut logits = match probs_in {
            Some(p) => p.clone(),
            None => {
                let l = self.build_lora_mm(gate_inp, cur, None);
                self.cb(&l, "ffn_moe_logits", il);
                l
            }
        };
        if let Some(b) = gate_inp_b {
            logits = add(&logits, b);
            self.cb(&logits, "ffn_moe_logits_biased", il);
        }

        let mut probs = match gating_op {
            ExpertGatingFuncType::Softmax => soft_max(&logits),
            ExpertGatingFuncType::Sigmoid => ggml::sigmoid(&logits),
            ExpertGatingFuncType::SoftmaxWeight => logits.clone(),
            ExpertGatingFuncType::SqrtSoftplus => sqrt(&softplus(&logits)),
            ExpertGatingFuncType::None => logits.clone(),
        };
        self.cb(&probs, "ffn_moe_probs", il);

        // The bias is used for selection only; `probs` stays unbiased.
        let mut selection_probs = probs.clone();
        if let Some(b) = exp_probs_b {
            selection_probs = add(&selection_probs, b);
            self.cb(&selection_probs, "ffn_moe_probs_biased", il);
        }
        if arch == Arch::Llama4 {
            selection_probs = logits.clone();
        }
        if arch == Arch::Grovemoe {
            selection_probs = ggml::sigmoid(&logits);
            self.cb(&selection_probs, "ffn_moe_probs_biased", il);
        }

        if self.hparams.n_expert_groups > 1 && n_tokens > 0 {
            let n_exp_per_group = n_expert / self.hparams.n_expert_groups as i64;
            let selection_groups = reshape_3d(
                &selection_probs,
                n_exp_per_group,
                self.hparams.n_expert_groups as i64,
                n_tokens,
            );
            let mut group_scores = argsort_top_k(&selection_groups, 2);
            let (ne0, ne1, ne2) = (selection_groups.ne(0), selection_groups.ne(1), selection_groups.ne(2));
            group_scores = get_rows(&reshape_4d(&selection_groups, [1, ne0, ne1, ne2]), &group_scores);
            let (ne1, ne2, ne3) = (group_scores.ne(1), group_scores.ne(2), group_scores.ne(3));
            group_scores = sum_rows(&reshape_3d(&group_scores, ne1, ne2, ne3));
            let (ne1, ne2) = (group_scores.ne(1), group_scores.ne(2));
            group_scores = reshape_2d(&group_scores, ne1, ne2);

            let expert_groups = argsort_top_k(&group_scores, self.hparams.n_group_used as i64);
            self.cb(&expert_groups, "ffn_moe_group_topk", il);

            selection_probs = get_rows(&selection_groups, &expert_groups);
            selection_probs = set_rows(
                &fill(&selection_groups, f32::NEG_INFINITY),
                &selection_probs,
                &expert_groups,
            );
            selection_probs = reshape_2d(&selection_probs, n_expert, n_tokens);
            self.cb(&selection_probs, "ffn_moe_probs_masked", il);
        }

        let mut selected_experts = match selected_experts_in {
            Some(t) => t.clone(),
            None => {
                let t = argsort_top_k(&selection_probs, n_expert_used);
                if let Some(src) = t.inner.borrow().src.first().cloned().flatten() {
                    self.cb(&src, "ffn_moe_argsort", il);
                }
                t
            }
        };
        self.cb(&selected_experts, "ffn_moe_topk", il);

        if arch == Arch::Grovemoe && n_expert != self.hparams.n_expert as i64 {
            let f_sel = ggml::cast(&selected_experts, Type::F32);
            let s = scale(&f_sel, 1.0 / self.hparams.n_group_experts as f32);
            selected_experts = ggml::cast(&s, Type::I32);
            probs = reshape_3d(&probs, 1, self.hparams.n_expert as i64, n_tokens);
        } else {
            probs = reshape_3d(&probs, 1, n_expert, n_tokens);
        }

        let mut weights = get_rows(&probs, &selected_experts);
        self.cb(&weights, "ffn_moe_weights", il);

        if gating_op == ExpertGatingFuncType::SoftmaxWeight {
            weights = reshape_2d(&weights, n_expert_used, n_tokens);
            weights = soft_max(&weights);
            weights = reshape_3d(&weights, 1, n_expert_used, n_tokens);
            self.cb(&weights, "ffn_moe_weights_softmax", il);
        }

        if norm_w {
            weights = reshape_2d(&weights, n_expert_used, n_tokens);
            let mut weights_sum = sum_rows(&weights);
            self.cb(&weights_sum, "ffn_moe_weights_sum", il);
            if arch == Arch::Deepseek41 {
                weights_sum = super::ggml::scale_bias(&weights_sum, 1.0, 1e-20);
            } else if self.hparams.expert_weights_norm_eps > 0.0 {
                weights_sum = super::ggml::scale_bias(
                    &weights_sum, 1.0, self.hparams.expert_weights_norm_eps,
                );
            } else {
                weights_sum = clamp(&weights_sum, 6.103515625e-5, f32::INFINITY);
            }
            self.cb(&weights_sum, "ffn_moe_weights_sum_clamped", il);
            weights = div(&weights, &weights_sum);
            self.cb(&weights, "ffn_moe_weights_norm", il);
            weights = reshape_3d(&weights, 1, n_expert_used, n_tokens);
        }
        if w_scale != 0.0 && w_scale != 1.0 {
            weights = scale(&weights, w_scale);
            self.cb(&weights, "ffn_moe_weights_scaled", il);
        }

        self.expand(&weights);

        if self.cparams.moe_random_routing
            && n_expert == self.hparams.n_expert as i64
            && n_expert_used <= self.n_expert_used
            && n_tokens <= self.ubatch.n_tokens as i64
        {
            let empty = ggml::view_2d(&selected_experts, 0, n_tokens, selected_experts.nb(1), 0);
            self.expand(&get_rows(&weights, &empty));
            selected_experts = self.build_inp_moe_ids(n_expert_used, n_tokens);
        }

        self.expand(cur);

        let lookup_args = MoeLookupArgs {
            il,
            up_exps: up_exps.clone(),
            gate_exps: gate_exps.clone(),
            down_exps: down_exps.clone(),
            gate_up_exps: gate_up_exps.cloned(),
            weight_before_ffn,
        };
        let lookup = self
            .moe_cache
            .as_ref()
            .map(|cache| cache.build_lookup(&lookup_args, &selected_experts))
            .unwrap_or_default();

        let mut cur2 = reshape_3d(cur, n_embd, 1, n_tokens);
        if weight_before_ffn {
            let repeated = ggml::repeat_4d(&cur2, n_embd, n_expert_used, n_tokens, 1);
            cur2 = mul(&repeated, &weights);
            self.cb(&cur2, "ffn_moe_weighted", il);
        }

        // Select and run the experts, up to the down projection.
        let build_experts = |x0: &Tensor, ids: &Tensor, mm_ids: Option<&Tensor>, cached: bool| -> Tensor {
            let mut x = x0.clone();
            let up: Tensor;
            match gate_up_exps {
                Some(gate_up) => {
                    let mut gate_up_out = self.build_lora_mm_id(gate_up, &x, ids, mm_ids, cached);
                    self.cb(&gate_up_out, "ffn_moe_gate_up", il);
                    if let Some(b) = gate_up_exps_b {
                        gate_up_out = add_id(&gate_up_out, b, ids);
                        self.cb(&gate_up_out, "ffn_moe_gate_up_biased", il);
                    }
                    let n_ff = gate_up_out.ne(0) / 2;
                    x = ggml::view_3d(
                        &gate_up_out, n_ff, gate_up_out.ne(1), gate_up_out.ne(2),
                        gate_up_out.nb(1), gate_up_out.nb(2), 0,
                    );
                    self.cb(&x, "ffn_moe_gate", il);
                    up = ggml::view_3d(
                        &gate_up_out, n_ff, gate_up_out.ne(1), gate_up_out.ne(2),
                        gate_up_out.nb(1), gate_up_out.nb(2), n_ff as usize * gate_up_out.nb(0),
                    );
                    self.cb(&up, "ffn_moe_up", il);
                }
                None => {
                    let mut u = self.build_lora_mm_id(up_exps, &x, ids, mm_ids, cached);
                    self.cb(&u, "ffn_moe_up", il);
                    if let Some(b) = up_exps_b {
                        u = add_id(&u, b, ids);
                        self.cb(&u, "ffn_moe_up_biased", il);
                    }
                    let g = self.build_lora_mm_id(gate_exps, &x, ids, mm_ids, cached);
                    self.cb(&g, "ffn_moe_gate", il);
                    x = g;
                    if let Some(b) = gate_exps_b {
                        x = add_id(&x, b, ids);
                        self.cb(&x, "ffn_moe_gate_biased", il);
                    }
                    up = u;
                }
            };

            let has_gate = gate_up_exps.is_some() || true;
            x = match type_op {
                FfnOpType::Silu => {
                    let mut limited = false;
                    if il >= 0 {
                        let limit = self.hparams.swiglu_clamp_exp.get(il as usize).copied().unwrap_or(0.0);
                        if limit > 1e-6 {
                            let special = matches!(
                                arch,
                                Arch::Maple | Arch::Deepseek4 | Arch::Deepseek41
                                    | Arch::Glm5Next | Arch::HyV4 | Arch::HyV4Flash
                            ) || (arch == Arch::Dflash && self.hparams.dsv4_hc_mult > 0);
                            if special {
                                x = ggml::swiglu_clamp(&x, &up, limit);
                            } else {
                                let u = clamp(&up, -limit, limit);
                                self.cb(&u, "ffn_moe_up_clamped", il);
                                let mut gate_act = ggml::silu(&x);
                                self.cb(&gate_act, "ffn_moe_silu", il);
                                gate_act = clamp(&gate_act, f32::NEG_INFINITY, limit);
                                self.cb(&gate_act, "ffn_moe_silu_clamped", il);
                                x = mul(&gate_act, &u);
                            }
                            limited = true;
                        }
                    }
                    if limited {
                        self.cb(&x, "ffn_moe_swiglu_limited", il);
                        x
                    } else if has_gate {
                        let r = ggml::swiglu_split(&x, &up);
                        self.cb(&r, "ffn_moe_swiglu", il);
                        r
                    } else {
                        let r = ggml::silu(&x);
                        self.cb(&r, "ffn_moe_silu", il);
                        r
                    }
                }
                FfnOpType::Situ => {
                    let beta = self.hparams.situ_beta;
                    let lb = self.hparams.situ_linear_beta;
                    let mut act = scale(&ggml::tanh(&scale(&x, 1.0 / beta)), beta);
                    act = mul(&act, &ggml::sigmoid(&x));
                    let u = if lb > 0.0 {
                        scale(&ggml::tanh(&scale(&up, 1.0 / lb)), lb)
                    } else {
                        up.clone()
                    };
                    let r = mul(&act, &u);
                    self.cb(&r, "ffn_moe_situ", il);
                    r
                }
                FfnOpType::Gelu => {
                    let r = if has_gate {
                        let r = ggml::geglu_split(&x, &up);
                        self.cb(&r, "ffn_moe_geglu", il);
                        r
                    } else {
                        let r = ggml::gelu(&x);
                        self.cb(&r, "ffn_moe_gelu", il);
                        r
                    };
                    r
                }
                FfnOpType::SwigluOaiMoe => {
                    let r = ggml::swiglu_oai(&x, &up, 1.702, 7.0);
                    self.cb(&r, "ffn_moe_swiglu_oai", il);
                    r
                }
                FfnOpType::Relu => {
                    let r = if has_gate {
                        let r = ggml::reglu_split(&x, &up);
                        self.cb(&r, "ffn_moe_reglu", il);
                        r
                    } else {
                        let r = ggml::relu(&x);
                        self.cb(&r, "ffn_moe_relu", il);
                        r
                    };
                    r
                }
                FfnOpType::ReluSqr => {
                    // C++ aborts for gated squared relu; keep the fused node
                    // so the graph topology stays observable.
                    let r = ggml::sqr(&ggml::relu(&x));
                    self.cb(&r, "ffn_moe_relu_sqr", il);
                    r
                }
                _ => x,
            };

            let use_amax = self.moe_down_bf16()
                && self.prec_policy.get_src1(down_exps) != super::ggml::Prec::BF16;
            let amax_scale = if use_amax {
                let amax = self.build_amax_src1_f16(&x);
                x = amax.res.unwrap();
                amax.scale_s
            } else {
                None
            };

            let mut experts = self.build_lora_mm_id(down_exps, &x, ids, mm_ids, cached);
            self.cb(&experts, "ffn_moe_down", il);
            if let Some(scale_s) = amax_scale {
                experts = mul(&experts, &scale_s);
            }
            experts
        };

        let mut experts = match (&self.moe_cache, &lookup.slots) {
            (Some(cache), Some(slots)) => {
                cache.build_experts(&cur2, &selected_experts, Some(slots), true, &build_experts)
            }
            _ => build_experts(&cur2, &selected_experts, lookup.slots.as_ref(), true),
        };

        if let Some(b) = down_exps_b {
            experts = add_id(&experts, b, &selected_experts);
            self.cb(&experts, "ffn_moe_down_biased", il);
        }
        if let Some(norm) = post_exp_norm {
            experts = mul(&experts, norm);
            experts = ggml::rms_norm(&experts, post_exp_norm_eps);
            self.cb(&experts, "ffn_moe_post_exp_norm", il);
        }
        if !weight_before_ffn {
            experts = mul(&experts, &weights);
            self.cb(&experts, "ffn_moe_weighted", il);
        }

        self.expand(&experts);

        let n_expert_used_il = self
            .hparams
            .n_expert_used(il.max(0) as u32)
            .unwrap_or(n_expert_used.max(0) as u32) as usize;
        let mut cur_experts: Vec<Option<Tensor>> = vec![None; n_expert_used_il];
        for i in 0..n_expert_used_il {
            let view = ggml::view_2d(
                &experts, n_embd, n_tokens, experts.nb(2), i * experts.nb(1) as usize,
            );
            self.expand(&view);
            cur_experts[i] = Some(view);
        }
        let mut moe_out = cur_experts.first().cloned().flatten().unwrap_or_else(|| cont(&experts));
        for i in 1..n_expert_used_il {
            if let Some(view) = &cur_experts[i] {
                moe_out = add(&moe_out, view);
                self.expand(&moe_out);
            }
        }
        if n_expert_used_il == 1 {
            moe_out = cont(&moe_out);
        }
        self.cb(&moe_out, "ffn_moe_out", il);
        moe_out
    }

    /// `build_inp_moe_ids`: one random-routing input shared across the graph.
    pub fn build_inp_moe_ids(&self, n_used: i64, n_rows: i64) -> Tensor {
        if self.res.borrow().t_moe_ids.is_empty() {
            let t = ggml::new_tensor_2d(
                Type::I32,
                self.hparams.n_expert_used_max().unwrap_or(0) as i64,
                self.n_tokens,
            );
            t.set_input();
            self.cb(&t, "inp_moe_ids", -1);
            let mut inp = GraphInputMoeIds::new(self.hparams.n_expert as usize);
            inp.ids = Some(t.clone());
            self.res.borrow_mut().add_input(inp);
            self.res.borrow_mut().t_moe_ids.push(t);
        }
        let res = self.res.borrow();
        for t in &res.t_moe_ids {
            if t.ne(0) == n_used && t.ne(1) == n_rows {
                return t.clone();
            }
        }
        let view = ggml::view_2d(&res.t_moe_ids[0], n_used, n_rows, res.t_moe_ids[0].nb(1), 0);
        drop(res);
        self.res.borrow_mut().t_moe_ids.push(view.clone());
        view
    }
}

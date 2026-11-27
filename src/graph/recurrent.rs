//! Recurrent-state and hybrid-memory builders.

use std::cell::RefCell;
use std::rc::Rc;

use super::context::GraphContext;
use super::ggml::{
    arange, cast, cpy, fill_inplace, get_rows, new_tensor_1d, new_tensor_2d, reshape_2d,
    reshape_3d, scale_bias, set_rows, view_1d, view_2d, view_3d, view_4d, Tensor, Type,
};
use super::inputs::{
    GraphInputAttnK, GraphInputAttnKvIswa, GraphInputMemHybrid, GraphInputMemHybridIswa,
    GraphInputMemHybridIswaRel, GraphInputMemHybridK, GraphInputRs,
};
use super::support::MemoryContext;
use super::types::GraphQkv;

/// `llm_graph_get_rows_fn`: custom state gather, without the context argument.
pub type GetRowsFn = dyn Fn(&Tensor, &Tensor) -> Tensor;

impl GraphContext {
    #[allow(clippy::too_many_arguments)]
    pub fn build_rs_full(
        &self,
        s: &Tensor,
        state_copy: &Tensor,
        state_copy_main: &Tensor,
        state_size: i32,
        n_seqs: i32,
        n_rs: u32,
        rs_head: u32,
        _rs_size: u32,
        rs_zero: i32,
        get_state_rows: Option<&GetRowsFn>,
        in_place: bool,
        from_cache: bool,
    ) -> Tensor {
        let states = reshape_2d(s, state_size as i64, s.ne(1));
        let has_zero = (rs_zero >= 0) as i64;
        let state_zero = view_1d(
            &states,
            state_size as i64 * has_zero,
            (rs_zero as usize) * states.nb(1) * has_zero as usize,
        );
        self.expand(&fill_inplace(&state_zero, 0.0));

        let i0 = n_rs as i64 - state_copy.ne(0);
        let states_all = get_rows(&states, state_copy);

        let output_states = if let Some(getter) = get_state_rows {
            getter(&states, state_copy_main)
        } else if !from_cache {
            view_2d(
                &states_all,
                state_size as i64,
                n_seqs as i64,
                states_all.nb(1),
                0,
            )
        } else {
            let n_move = if in_place { 0 } else { n_seqs as i64 };
            self.expand(&cpy(
                &view_2d(&states_all, state_size as i64, n_move, states_all.nb(1), 0),
                &view_2d(
                    &states,
                    state_size as i64,
                    n_move,
                    states.nb(1),
                    rs_head as usize * states.nb(1),
                ),
            ));
            view_2d(
                &states,
                state_size as i64,
                n_seqs as i64,
                states.nb(1),
                rs_head as usize * states.nb(1),
            )
        };
        self.expand(&output_states);

        let states_extra = view_2d(
            &states_all,
            state_size as i64,
            n_rs as i64 - n_seqs as i64,
            states_all.nb(1),
            (n_seqs as i64 - i0).max(0) as usize * states_all.nb(1),
        );
        self.expand(&cpy(
            &states_extra,
            &view_2d(
                s,
                state_size as i64,
                (n_rs as i64 - n_seqs as i64).max(0),
                s.nb(1),
                (rs_head as usize + n_seqs as usize) * s.nb(1),
            ),
        ));
        output_states
    }

    pub(crate) fn build_rs_inp_impl(&self, mctx: Rc<dyn MemoryContext>) -> Rc<RefCell<GraphInputRs>> {
        let mut inp = GraphInputRs::new(mctx.clone());
        let n_rs = mctx.get_n_rs();
        let n_seqs = self.ubatch.n_seqs as usize;

        let s_copy = new_tensor_1d(Type::I32, n_rs as i64);
        s_copy.set_input();
        s_copy.set_name("rs_s_copy");
        inp.s_copy = Some(s_copy.clone());
        inp.s_copy_main = Some(view_1d(&s_copy, n_seqs as i64, 0));
        inp.s_copy_tail = Some(view_1d(&s_copy, n_rs as i64 - 1, Type::I32.type_size()));

        inp.head = mctx.get_head();
        inp.rs_z = mctx.get_rs_z();
        inp.rs_in_place = mctx.get_rs_in_place();
        inp.s_copy_rows = Some(view_1d(
            &s_copy,
            if inp.rs_in_place { 0 } else { n_rs as i64 },
            0,
        ));

        inp.has_rep = mctx.get_rs_snapshots() || self.cparams.rs_replay;
        if inp.has_rep {
            let n_conv_r = self.hparams.ssm_d_conv as i64 - 1;
            let n_conv_p = if self.hparams.ple_conv_state() > 0 {
                (self.hparams.ple_conv_kernel as i64 - 1) * self.hparams.ple_ngram_size as i64
            } else {
                0
            };
            inp.n_conv = n_conv_r.max(n_conv_p);
            inp.n_pend = self.cparams.n_rs_seq as i64 + 1;

            let rep_cols = new_tensor_2d(Type::I32, inp.n_conv, n_seqs as i64);
            rep_cols.set_input();
            rep_cols.set_name("rs_rep_cols");
            inp.rep_cols = Some(rep_cols.clone());

            let win = |n: i64| -> Tensor {
                if n == inp.n_conv {
                    rep_cols.clone()
                } else {
                    view_2d(&rep_cols, n, n_seqs as i64, rep_cols.nb(1), 0)
                }
            };
            inp.rep_cols_r = Some(win(n_conv_r));
            inp.rep_cols_p = if n_conv_p > 0 { Some(win(n_conv_p)) } else { None };

            let rep_n = new_tensor_1d(Type::I32, n_seqs as i64);
            rep_n.set_input();
            rep_n.set_name("rs_rep_n");
            inp.rep_n = Some(rep_n);
        }
        self.res.borrow_mut().add_input(inp)
    }

    pub fn build_rs_inp(&self) -> Rc<RefCell<GraphInputRs>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        self.build_rs_inp_impl(mctx)
    }

    /// `build_rs` wrapper taking the recurrent input.
    pub fn build_rs(
        &self,
        inp: &GraphInputRs,
        s: &Tensor,
        state_size: i32,
        n_seqs: i32,
        get_state_rows: Option<&GetRowsFn>,
        gathered: bool,
    ) -> Tensor {
        let kv_state = inp.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let in_place = inp.rs_in_place && get_state_rows.is_none() && !gathered;
        let from_cache = !gathered && !kv_state.get_rs_snapshots();
        let state_copy = if get_state_rows.is_some() {
            inp.s_copy_tail.clone()
        } else if from_cache {
            inp.s_copy_rows.clone()
        } else {
            inp.s_copy.clone()
        };
        let Some(state_copy) = state_copy else { return s.clone() };
        self.build_rs_full(
            s,
            &state_copy,
            inp.s_copy_main.as_ref().unwrap_or(&state_copy),
            state_size,
            n_seqs,
            kv_state.get_n_rs() as u32,
            kv_state.get_head(),
            kv_state.get_size() as u32,
            kv_state.get_rs_z(),
            get_state_rows,
            in_place,
            from_cache,
        )
    }

    pub fn build_conv_state_store(
        &self,
        inp: &GraphInputRs,
        s: &Tensor,
        conv_x: &Tensor,
        n_cols: i64,
        offs: usize,
    ) {
        let mctx = inp.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let n = conv_x.ne(0) - n_cols;
        let channels = conv_x.ne(1);
        let n_seqs = conv_x.ne(2);
        let row_size = s.nb(1);
        let cell_offs = mctx.get_head() as usize * row_size + offs;

        if self.cparams.n_rs_seq == 0 {
            let last = view_3d(
                conv_x,
                n_cols,
                channels,
                n_seqs,
                conv_x.nb(1),
                conv_x.nb(2),
                conv_x.ty().row_size(n),
            );
            let cell = view_2d(s, n_cols * channels, n_seqs, row_size, cell_offs);
            self.expand(&cpy(&last, &cell));
            return;
        }

        let k = self.cparams.n_rs_seq as i64 + 1;
        let kw = n.min(k);
        let mut s_slot_rev = arange(0.0, k as f32, 1.0);
        s_slot_rev = scale_bias(&s_slot_rev, -1.0, (k - 1) as f32);
        s_slot_rev = cast(&s_slot_rev, Type::I32);
        self.cb(&s_slot_rev, "s_slot_rev", -1);
        let slot_order = view_1d(&s_slot_rev, kw, Type::I32.type_size() * (k - kw) as usize);

        let windows = view_4d(
            conv_x,
            [n_cols, kw, channels, n_seqs],
            conv_x.ty().type_size(),
            conv_x.nb(1),
            conv_x.nb(2),
            conv_x.ty().row_size((n - (kw - 1)).max(0)),
        );
        let slots = view_4d(
            s,
            [n_cols, k, channels, n_seqs],
            mctx.get_size() as usize * row_size,
            conv_x.ty().row_size(n_cols),
            row_size,
            cell_offs,
        );
        self.expand(&set_rows(&slots, &windows, &slot_order));
    }

    pub fn build_rwkv_token_shift_load(
        &self,
        inp: &GraphInputRs,
        ubatch: &crate::batch::Ubatch,
        il: i32,
    ) -> Tensor {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let n_seqs = ubatch.n_seqs as i64;
        let token_shift_all = mctx.get_r_l(il);
        let mut token_shift = self.build_rs(
            inp,
            &token_shift_all,
            self.hparams.n_embd_r().unwrap_or(0) as i32,
            n_seqs as i32,
            None,
            false,
        );
        token_shift = reshape_3d(
            &token_shift,
            self.hparams.n_embd as i64,
            self.hparams.token_shift_count as i64,
            n_seqs,
        );
        token_shift
    }

    pub fn build_rwkv_token_shift_store(
        &self,
        token_shift: &Tensor,
        ubatch: &crate::batch::Ubatch,
        il: i32,
    ) -> Tensor {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let n_seqs = ubatch.n_seqs as i64;
        let n_embd = self.hparams.n_embd as usize;
        let kv_head = mctx.get_head() as usize;
        let dst = mctx.get_r_l(il);
        cpy(
            &view_1d(
                token_shift,
                n_embd as i64 * n_seqs * self.hparams.token_shift_count as i64,
                0,
            ),
            &view_1d(
                &dst,
                self.hparams.n_embd_r().unwrap_or(0) as i64 * n_seqs,
                self.hparams.n_embd_r().unwrap_or(0) as usize * kv_head * dst.ty().type_size(),
            ),
        )
    }

    pub fn build_inp_mem_hybrid(&self) -> Rc<RefCell<GraphInputMemHybrid>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let recr = mctx.get_recr().unwrap_or_else(|| mctx.clone());
        let attn = mctx.get_attn().unwrap_or_else(|| mctx.clone());
        let inp_rs = self.build_rs_inp_impl(recr);
        let inp_attn = self.build_attn_inp_kv_impl(attn);
        let inp = GraphInputMemHybrid::new(&self.cparams, inp_attn, inp_rs, mctx);
        self.res.borrow_mut().add_input(inp)
    }

    pub fn build_inp_mem_hybrid_k(&self) -> Rc<RefCell<GraphInputMemHybridK>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let recr = mctx.get_recr().unwrap_or_else(|| mctx.clone());
        let attn = mctx.get_attn().unwrap_or_else(|| mctx.clone());
        let inp_rs = self.build_rs_inp_impl(recr);
        let mut inp_attn = GraphInputAttnK::new(&self.hparams, &self.cparams, attn.clone());
        inp_attn.self_k_idxs = Some(attn.build_input_k_idxs(&self.ubatch));
        inp_attn.self_attn_mask.raw = Some(super::inputs::build_attn_inp_kq_mask(
            attn.as_ref(), &self.ubatch, &self.cparams, false,
        ));
        let inp_attn = self.res.borrow_mut().add_input(inp_attn);
        let inp = GraphInputMemHybridK::new(&self.cparams, inp_attn, inp_rs, mctx);
        self.res.borrow_mut().add_input(inp)
    }

    pub fn build_inp_mem_hybrid_iswa(&self) -> Rc<RefCell<GraphInputMemHybridIswa>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let recr = mctx.get_recr().unwrap_or_else(|| mctx.clone());
        let attn_ctx = mctx.get_attn().unwrap_or_else(|| mctx.clone());
        let inp_rs = self.build_rs_inp_impl(recr);
        let inp_attn = self.build_attn_inp_kv_iswa_ctx(attn_ctx);
        let inp = GraphInputMemHybridIswa::new(&self.cparams, inp_attn, inp_rs, mctx);
        self.res.borrow_mut().add_input(inp)
    }

    fn build_attn_inp_kv_iswa_ctx(
        &self,
        attn_ctx: Rc<dyn MemoryContext>,
    ) -> Rc<RefCell<GraphInputAttnKvIswa>> {
        let base = attn_ctx.get_base().unwrap_or_else(|| attn_ctx.clone());
        let swa = attn_ctx.get_swa().unwrap_or_else(|| attn_ctx.clone());
        let mut inp = GraphInputAttnKvIswa::new(&self.hparams, &self.cparams, attn_ctx);
        inp.self_k_idxs = Some(base.build_input_k_idxs(&self.ubatch));
        inp.self_v_idxs = Some(base.build_input_v_idxs(&self.ubatch));
        inp.self_attn_mask = super::inputs::build_attn_inp_attn_mask(
            base.as_ref(), &self.ubatch, &self.cparams,
        );
        inp.self_k_idxs_swa = Some(swa.build_input_k_idxs(&self.ubatch));
        inp.self_v_idxs_swa = Some(swa.build_input_v_idxs(&self.ubatch));
        inp.self_attn_mask_swa = super::inputs::build_attn_inp_attn_mask(
            swa.as_ref(), &self.ubatch, &self.cparams,
        );
        inp.self_k_rot = base.get_k_rot();
        inp.self_v_rot = base.get_v_rot();
        inp.self_k_rot_swa = swa.get_k_rot();
        inp.self_v_rot_swa = swa.get_v_rot();
        self.res.borrow_mut().add_input(inp)
    }

    pub fn build_inp_mem_hybrid_iswa_rel(
        &self,
        extent: u32,
        extent_swa: u32,
    ) -> Rc<RefCell<GraphInputMemHybridIswaRel>> {
        let mctx = self.mctx.clone().unwrap_or_else(|| Rc::new(super::support::MemoryStandIn::default()));
        let recr = mctx.get_recr().unwrap_or_else(|| mctx.clone());
        let attn_ctx = mctx.get_attn().unwrap_or_else(|| mctx.clone());
        let inp_rs = self.build_rs_inp_impl(recr);

        let base = attn_ctx.get_base().unwrap_or_else(|| attn_ctx.clone());
        let swa = attn_ctx.get_swa().unwrap_or_else(|| attn_ctx.clone());
        let mut inp_attn = GraphInputAttnKvIswa::new(&self.hparams, &self.cparams, attn_ctx);
        inp_attn.self_k_idxs = Some(base.build_input_k_idxs(&self.ubatch));
        inp_attn.self_v_idxs = Some(base.build_input_v_idxs(&self.ubatch));
        inp_attn.self_attn_mask.raw = Some(super::inputs::build_attn_inp_kq_mask(
            base.as_ref(), &self.ubatch, &self.cparams, false,
        ));
        inp_attn.self_k_idxs_swa = Some(swa.build_input_k_idxs(&self.ubatch));
        inp_attn.self_v_idxs_swa = Some(swa.build_input_v_idxs(&self.ubatch));
        inp_attn.self_attn_mask_swa.raw = Some(super::inputs::build_attn_inp_kq_mask(
            swa.as_ref(), &self.ubatch, &self.cparams, false,
        ));
        inp_attn.self_k_rot = base.get_k_rot();
        inp_attn.self_v_rot = base.get_v_rot();
        inp_attn.self_k_rot_swa = swa.get_k_rot();
        inp_attn.self_v_rot_swa = swa.get_v_rot();
        let inp_attn = self.res.borrow_mut().add_input(inp_attn);

        let mut inp = GraphInputMemHybridIswaRel::new(
            &self.cparams, inp_attn, inp_rs, mctx, extent, extent_swa,
        );
        let rel = new_tensor_2d(Type::I32, base.get_n_kv() as i64, self.ubatch.n_tokens as i64);
        rel.set_input();
        rel.set_name("attn_inp_rel_pos");
        inp.self_rel_pos = Some(rel);
        let rel_swa = new_tensor_2d(Type::I32, swa.get_n_kv() as i64, self.ubatch.n_tokens as i64);
        rel_swa.set_input();
        rel_swa.set_name("attn_inp_rel_pos_swa");
        inp.self_rel_pos_swa = Some(rel_swa);
        self.res.borrow_mut().add_input(inp)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_attn_mem_hybrid_iswa_rel(
        &self,
        inp: &GraphInputMemHybridIswaRel,
        wo: Option<&Tensor>,
        wo_b: Option<&Tensor>,
        wo_s: Option<&Tensor>,
        q_cur: &Tensor,
        k_cur: Option<&Tensor>,
        v_cur: Option<&Tensor>,
        bias_table: &Tensor,
        sinks: &Tensor,
        kq_scale: f32,
        il: i32,
    ) -> Tensor {
        let is_swa = self.hparams.is_swa(il.max(0) as u32).unwrap_or(false);
        let Some(inp_attn) = &inp.inp_attn else { return q_cur.clone() };
        let inp_attn = inp_attn.borrow();
        let attn_mask = if is_swa { &inp_attn.self_attn_mask_swa } else { &inp_attn.self_attn_mask };
        let kq_mask = attn_mask.raw.as_ref().unwrap_or(q_cur);
        let indices = if is_swa { &inp.self_rel_pos_swa } else { &inp.self_rel_pos };
        self.build_attn_iswa_impl(
            &inp_attn, wo, wo_b, wo_s, q_cur, k_cur, v_cur, None, kq_mask, None, sinks, None,
            kq_scale, il, Some(bias_table), indices.as_ref(),
        )
    }

    /// `GraphQkv` convenience wrapper used by hybrid graph builders.
    pub fn qkv(&self, q: Tensor, k: Tensor, v: Tensor) -> GraphQkv {
        GraphQkv { q, k, v }
    }
}

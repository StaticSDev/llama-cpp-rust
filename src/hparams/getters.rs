use crate::core::{Error, Result};
use super::params::{Hparams, RopeType, SwaType};

const GGML_MROPE_SECTIONS: u32 = 4;

impl Hparams {
    fn oob(&self, func: &str, il: u32, label: &str, n: u32) -> Error {
        Error::new(format!("{}: il ({}) out of bounds ({}: {})", func, il, label, n))
    }

    /// `n_pattern == 0` means all layers SWA; `1` means all dense.
    pub fn set_swa_pattern(&mut self, n_pattern: u32, dense_first: bool) {
        let n = self.n_layer() as usize;
        for (il, slot) in self.is_swa_impl.iter_mut().enumerate().take(n) {
            let il = il as u32;
            *slot = if dense_first {
                (n_pattern == 0 || il % n_pattern != 0) as u32
            } else {
                (n_pattern == 0 || il % n_pattern < n_pattern - 1) as u32
            };
        }
        for slot in self.is_swa_impl.iter_mut().skip(n) {
            *slot = 0;
        }
    }

    /// Same pattern logic as `set_swa_pattern`, for recurrent/hybrid layers.
    pub fn set_recr_pattern(&mut self, n_pattern: u32, dense_first: bool) {
        let n = self.n_layer() as usize;
        for (il, slot) in self.is_recr_impl.iter_mut().enumerate().take(n) {
            let il = il as u32;
            *slot = if dense_first {
                (n_pattern == 0 || il % n_pattern != 0) as u32
            } else {
                (n_pattern == 0 || il % n_pattern < n_pattern - 1) as u32
            };
        }
        for slot in self.is_recr_impl.iter_mut().skip(n) {
            *slot = 0;
        }
    }

    /// True if one of the layers is SWA.
    pub fn is_swa_any(&self) -> bool {
        self.is_swa_impl[..self.n_layer_all as usize].iter().any(|v| *v != 0)
    }

    pub fn n_head(&self, il: u32) -> Result<u32> {
        self.n_head_arr.get(il as usize).copied()
            .filter(|_| il < self.n_layer_all)
            .ok_or_else(|| self.oob("n_head", il, "n_layer_all", self.n_layer_all))
    }

    pub fn n_head_kv(&self, il: u32) -> Result<u32> {
        self.n_head_kv_arr.get(il as usize).copied()
            .filter(|_| il < self.n_layer_all)
            .ok_or_else(|| self.oob("n_head_kv", il, "n_layer_all", self.n_layer_all))
    }

    pub fn n_ff(&self, il: u32) -> Result<u32> {
        self.n_ff_arr.get(il as usize).copied()
            .filter(|_| il < self.n_layer_all)
            .ok_or_else(|| self.oob("n_ff", il, "n_layer_all", self.n_layer_all))
    }

    pub fn n_ff_exp(&self, il: u32) -> Result<u32> {
        self.n_ff_exp_arr.get(il as usize).copied()
            .filter(|_| il < self.n_layer_all)
            .ok_or_else(|| self.oob("n_ff_exp", il, "n_layer_all", self.n_layer_all))
    }

    pub fn n_expert_used(&self, il: u32) -> Result<u32> {
        self.n_expert_used_arr.get(il as usize).copied()
            .filter(|_| il < self.n_layer_all)
            .ok_or_else(|| self.oob("n_expert_used", il, "n_layer_all", self.n_layer_all))
    }

    /// Maximum `n_expert_used` across all layers.
    pub fn n_expert_used_max(&self) -> Result<u32> {
        let mut val = 0;
        for il in 0..self.n_layer_all {
            val = val.max(self.n_expert_used(il)?);
        }
        Ok(val)
    }

    pub fn n_gqa(&self, il: u32) -> Result<u32> {
        let n_head = self.n_head(il)?;
        let n_head_kv = self.n_head_kv(il)?;
        if n_head_kv == 0 { return Ok(0); }
        Ok(n_head / n_head_kv)
    }

    pub fn n_rot(&self, il: u32) -> Result<u32> {
        if il >= self.n_layer_all {
            return Err(self.oob("n_rot", il, "n_layer_all", self.n_layer_all));
        }
        Ok(if self.is_swa(il)? { self.n_rot_swa } else { self.n_rot_full })
    }

    /// Dimension of main + auxiliary input embeddings.
    pub fn n_embd_inp(&self) -> u32 {
        if self.n_embd_inp_impl > 0 {
            return self.n_embd_inp_impl;
        }
        let mut n_embd_inp = self.n_embd;
        if self.n_deepstack_layers > 0 {
            n_embd_inp += self.n_embd * self.n_deepstack_layers;
        }
        n_embd_inp
    }

    /// Dimension of the encoder input embeddings.
    pub fn n_embd_inp_enc(&self) -> u32 {
        if self.n_embd_inp_enc_impl > 0 { self.n_embd_inp_enc_impl } else { self.n_embd_inp() }
    }

    /// Dimension of output embeddings.
    pub fn n_embd_out(&self) -> u32 {
        if self.n_embd_out_impl > 0 { self.n_embd_out_impl } else { self.n_embd }
    }

    /// Dimension of key embeddings for each head (per layer).
    pub fn n_embd_head_k(&self, il: u32) -> Result<u32> {
        if il >= self.n_layer_all {
            return Err(self.oob("n_embd_head_k", il, "n_layer_all", self.n_layer_all));
        }
        Ok(if self.is_swa(il)? { self.n_embd_head_k_swa } else { self.n_embd_head_k_full })
    }

    /// Dimension of value embeddings for each head (per layer).
    pub fn n_embd_head_v(&self, il: u32) -> Result<u32> {
        if il >= self.n_layer_all {
            return Err(self.oob("n_embd_head_v", il, "n_layer_all", self.n_layer_all));
        }
        Ok(if self.is_swa(il)? { self.n_embd_head_v_swa } else { self.n_embd_head_v_full })
    }

    /// Dimension of key embeddings across all k-v heads.
    pub fn n_embd_k_gqa(&self, il: u32) -> Result<u32> {
        Ok(self.n_embd_head_k(il)? * self.n_head_kv(il)?)
    }

    /// Dimension of value embeddings across all k-v heads.
    pub fn n_embd_v_gqa(&self, il: u32) -> Result<u32> {
        Ok(self.n_embd_head_v(il)? * self.n_head_kv(il)?)
    }

    /// True if any layer has a different `n_embd_k_gqa`.
    pub fn is_n_embd_k_gqa_variable(&self) -> Result<bool> {
        let val = self.n_embd_k_gqa(0)?;
        for il in 0..self.n_layer_all {
            if val != self.n_embd_k_gqa(il)? { return Ok(true); }
        }
        Ok(false)
    }

    /// True if any layer has a different `n_embd_v_gqa`.
    pub fn is_n_embd_v_gqa_variable(&self) -> Result<bool> {
        let val = self.n_embd_v_gqa(0)?;
        for il in 0..self.n_layer_all {
            if val != self.n_embd_v_gqa(il)? { return Ok(true); }
        }
        Ok(false)
    }

    pub fn n_embd_k_gqa_max(&self) -> Result<u32> {
        let mut val = self.n_embd_k_gqa(0)?;
        for il in 0..self.n_layer_all {
            val = val.max(self.n_embd_k_gqa(il)?);
        }
        Ok(val)
    }

    pub fn n_embd_v_gqa_max(&self) -> Result<u32> {
        let mut val = self.n_embd_v_gqa(0)?;
        for il in 0..self.n_layer_all {
            val = val.max(self.n_embd_v_gqa(il)?);
        }
        Ok(val)
    }

    /// Dimension of the rolling state embeddings: Mamba conv_states size or
    /// RWKV token_shift states size.
    pub fn n_embd_r(&self) -> Result<u32> {
        if self.wkv_head_size != 0 {
            // for RWKV models
            return Ok(self.token_shift_count * self.n_embd);
        }

        if self.n_sconv_channels != 0 {
            // Inkling2, checked before LFM2 as they also set n_shortconv_l_cache
            return Ok(self.n_sconv_channels * self.n_shortconv_l_cache.wrapping_sub(1));
        }

        if self.n_shortconv_l_cache != 0 {
            // for LFM2 models
            return Ok(self.n_embd * self.n_shortconv_l_cache.wrapping_sub(1));
        }

        if self.n_cca_qk != 0 {
            // ZAYA1 CCA: depthwise + grouped conv, each with d_conv - 1 history columns
            return Ok(2 * self.ssm_d_conv.wrapping_sub(1) * self.n_cca_qk);
        }

        if self.n_embd_head_kda != 0 {
            // Kimi KDA: conv state for Q, K, V = 3 * (d_conv - 1) * n_head * head_dim
            let d_inner = self.n_head(0)? * self.n_embd_head_kda; // 32 * 128 = 4096
            let d_conv_history = if self.ssm_d_conv > 0 { self.ssm_d_conv - 1 } else { 3 };
            return Ok(3 * d_conv_history * d_inner);
        }

        // TODO: maybe support other convolution strides than 1
        // NOTE: the first conv_state column is shifted out each time, so it is not needed.
        // Corresponds to Mamba's conv_states size.
        let n_conv = if self.ssm_d_conv > 0 { self.ssm_d_conv - 1 } else { 0 }
            * (self.ssm_d_inner + 2 * self.ssm_n_group * self.ssm_d_state);
        Ok(n_conv)
    }

    /// Dimension of the recurrent state embeddings.
    pub fn n_embd_s(&self) -> Result<u32> {
        if self.wkv_head_size != 0 {
            // corresponds to RWKV's wkv_states size
            return Ok(self.n_embd * self.wkv_head_size);
        }

        if self.n_cca_qk != 0 {
            // ZAYA CCA: delayed value of the previous token
            return Ok(self.n_embd_v_gqa(0)? / 2);
        }

        if self.n_embd_head_kda != 0 {
            // Kimi KDA: full recurrent state, head_dim * head_dim * n_head
            // h tensor shape: [head_dim, head_dim, n_head]
            return Ok(self.n_embd_head_kda * self.n_embd_head_kda * self.n_head(0)?);
        }

        if self.n_embd_head_la != 0 {
            // MiniMax-Text-01 linear attention: head_dim * head_dim * n_head
            return Ok(self.n_embd_head_la * self.n_embd_head_la * self.n_head(0)?);
        }

        // corresponds to Mamba's ssm_states size
        Ok(self.ssm_d_state * self.ssm_d_inner)
    }

    /// Whether the given layer is recurrent (for hybrid models).
    pub fn is_recr(&self, il: u32) -> Result<bool> {
        self.is_recr_impl.get(il as usize).copied()
            .filter(|_| il < self.n_layer_all)
            .map(|v| v != 0)
            .ok_or_else(|| self.oob("is_recr", il, "n_layer_all", self.n_layer_all))
    }

    /// PLE conv history rows: (kernel - 1) * ngram_size; 0 without a PLE module.
    pub fn ple_conv_state(&self) -> u32 {
        if self.ple_n_heads == 0 || self.ple_conv_kernel == 0 {
            return 0;
        }
        // dilation equals the n-gram size, matching the reference module
        (self.ple_conv_kernel - 1) * self.ple_ngram_size * self.dsv4_hc_mult * self.n_embd
    }

    pub fn is_ple(&self, il: u32) -> Result<bool> {
        self.is_ple_impl.get(il as usize).copied()
            .filter(|_| il < self.n_layer_all)
            .ok_or_else(|| self.oob("is_ple", il, "n_layer_all", self.n_layer_all))
    }

    pub fn n_pos_per_embd(&self) -> u32 {
        match self.rope_type {
            RopeType::Mrope | RopeType::Imrope => GGML_MROPE_SECTIONS,
            _ => 1,
        }
    }

    pub fn is_swa(&self, il: u32) -> Result<bool> {
        self.is_swa_impl.get(il as usize).copied()
            .filter(|_| il < self.n_layer_all)
            .map(|v| v != 0)
            .ok_or_else(|| self.oob("is_swa", il, "n_layer_all", self.n_layer_all))
    }

    /// Currently supported only when either all or none of the layers are MLA.
    pub fn is_mla(&self) -> bool {
        debug_assert!(
            (self.n_embd_head_k_mla_impl == 0 && self.n_embd_head_v_mla_impl == 0)
            || (self.n_embd_head_k_mla_impl != 0 && self.n_embd_head_v_mla_impl != 0)
        );
        self.n_embd_head_k_mla_impl != 0 && self.n_embd_head_v_mla_impl != 0
    }

    pub fn is_indexer_full(&self, il: u32) -> Result<bool> {
        let n_layer = self.n_layer();
        self.is_indexer_full_impl.get(il as usize).copied()
            .filter(|_| il < n_layer)
            .map(|v| v != 0)
            .ok_or_else(|| self.oob("is_indexer_full", il, "n_layer", n_layer))
    }

    pub fn n_embd_head_k_mla(&self) -> Result<u32> {
        Ok(if self.is_mla() { self.n_embd_head_k_mla_impl } else { self.n_embd_head_k(0)? })
    }

    pub fn n_embd_head_v_mla(&self) -> Result<u32> {
        Ok(if self.is_mla() { self.n_embd_head_v_mla_impl } else { self.n_embd_head_v(0)? })
    }

    pub fn has_kv(&self, il: u32) -> Result<bool> {
        Ok(self.kv_source(il)? == il)
    }

    pub fn kv_source(&self, il: u32) -> Result<u32> {
        self.kv_source_impl.get(il as usize).copied()
            .filter(|_| il < self.n_layer_all)
            .ok_or_else(|| self.oob("kv_source", il, "n_layer_all", self.n_layer_all))
    }

    pub fn has_rope(&self, il: u32) -> Result<bool> {
        // the router layer stores adapter routing signal, not positional info,
        // so it must not be RoPE-shifted
        if self.router_layer >= 0 && il as i32 == self.router_layer {
            return Ok(false);
        }
        self.rope_pattern.get(il as usize).copied()
            .filter(|_| il < self.n_layer_all)
            .map(|v| v != 0)
            .ok_or_else(|| self.oob("has_rope", il, "n_layer_all", self.n_layer_all))
    }

    /// Number of effective layers (excludes nextn layers).
    pub fn n_layer(&self) -> u32 {
        self.n_layer_all.wrapping_sub(self.n_layer_nextn)
    }

    /// Uses different SWA parameters from those in the hparams.
    pub fn is_masked_swa(n_swa: u32, swa_type: SwaType, p0: i32, p1: i32) -> bool {
        debug_assert!(p0 >= 0 && p1 >= 0);

        match swa_type {
            SwaType::None => {}
            SwaType::Standard => {
                if p1 - p0 >= n_swa as i32 { return true; }
            }
            SwaType::Chunked => {
                let pos_chunk_start = (p1 / n_swa as i32) * n_swa as i32;
                if p0 < pos_chunk_start { return true; }
            }
            SwaType::Symmetric => {
                let half_n_swa = n_swa as i32 / 2;
                let pos_diff = p1 - p0;
                // Mask if outside the symmetric window
                if pos_diff < -half_n_swa || pos_diff > half_n_swa { return true; }
            }
        }

        false
    }

    pub fn use_mrope(&self) -> bool {
        self.rope_sections[0] > 0 && self.rope_sections[1] > 0
    }
}

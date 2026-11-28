//! FFI bindings to the small CUDA matvec kernels in `cuda/lcr_kernels.cu`.

use crate::core::{Error, Result};
use crate::runtime::types::Weight;
use std::ffi::{c_char, c_int, c_void};
use std::ptr;

#[repr(C)]
struct WeightMeta {
    type_: i32,
    in_: i64,
    out_: i64,
    offset: i64,
}

unsafe extern "C" {
    fn lcr_gpu_init(ctx_out: *mut *mut std::ffi::c_void, device: c_int) -> c_int;
    fn lcr_gpu_free(ctx: *mut std::ffi::c_void);
    fn lcr_gpu_upload(
        ctx: *mut std::ffi::c_void,
        blob: *const std::ffi::c_void,
        bytes: usize,
        metas: *const WeightMeta,
        count: c_int,
    ) -> c_int;
    fn lcr_gpu_matvec(
        ctx: *mut std::ffi::c_void,
        index: c_int,
        x: *const f32,
        out: *mut f32,
    ) -> c_int;
    fn lcr_gpu_row(ctx: *mut std::ffi::c_void, index: c_int, row: c_int, out: *mut f32) -> c_int;
    fn lcr_gpu_info(ctx: *mut std::ffi::c_void, buf: *mut c_char, len: c_int) -> c_int;

    // Device-resident scratch allocation and inspection.
    fn lcr_gpu_buf(ctx: *mut c_void, bytes: usize, out: *mut *mut c_void) -> c_int;
    fn lcr_gpu_buf_free(ctx: *mut c_void, ptr: *mut c_void);
    fn lcr_gpu_weight_ptr(ctx: *mut c_void, index: c_int) -> *mut c_void;
    fn lcr_gpu_zero(ctx: *mut c_void, ptr: *mut c_void, bytes: usize) -> c_int;
    fn lcr_gpu_row_device(ctx: *mut c_void, index: c_int, row: c_int, dst: *mut f32) -> c_int;
    fn lcr_gpu_matvec_device(ctx: *mut c_void, index: c_int, x: *const f32, out: *mut f32)
        -> c_int;
    fn lcr_batch_matmul(
        ctx: *mut c_void,
        index: c_int,
        x: *const f32,
        out: *mut f32,
        n_tokens: c_int,
        x_stride: c_int,
        out_stride: c_int,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn lcr_batch_conv_silu(
        ctx: *mut c_void,
        w: *const f32,
        hist: *mut f32,
        qkv: *const f32,
        out: *mut f32,
        n_tokens: c_int,
        qkv_stride: c_int,
        out_stride: c_int,
        channels: c_int,
        k: c_int,
        stride: c_int,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn lcr_batch_gdn(
        ctx: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        n_tokens: c_int,
        q_stride: c_int,
        kv_stride: c_int,
        v_stride: c_int,
        g_stride: c_int,
        n_v: c_int,
        n_k: c_int,
        d: c_int,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn lcr_batch_attn(
        ctx: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        kc: *mut f32,
        vc: *mut f32,
        out: *mut f32,
        n_tokens: c_int,
        q_stride: c_int,
        kv_stride: c_int,
        out_stride: c_int,
        n_heads: c_int,
        n_kv: c_int,
        d: c_int,
        seq_start: c_int,
        seq_cap: c_int,
        scale: f32,
    ) -> c_int;
    fn lcr_gpu_read_i32(ctx: *mut c_void, src: *const c_void, dst: *mut c_int) -> c_int;
    fn lcr_gpu_sync(ctx: *mut c_void) -> c_int;

    // Device-resident elementwise and block kernels.
    fn lcr_op_rmsnorm(
        ctx: *mut c_void,
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        n: c_int,
        eps: f32,
    ) -> c_int;
    fn lcr_op_l2norm(ctx: *mut c_void, x: *mut f32, n: c_int, eps: f32) -> c_int;
    fn lcr_op_rope(
        ctx: *mut c_void,
        x: *mut f32,
        n: c_int,
        n_rot: c_int,
        base: f32,
        pos: c_int,
    ) -> c_int;
    fn lcr_op_add(ctx: *mut c_void, dst: *mut f32, src: *const f32, n: c_int) -> c_int;
    fn lcr_op_mul(ctx: *mut c_void, dst: *mut f32, src: *const f32, n: c_int) -> c_int;
    fn lcr_op_scale(ctx: *mut c_void, x: *mut f32, n: c_int, scale: f32) -> c_int;
    fn lcr_op_sigmoid(ctx: *mut c_void, x: *const f32, y: *mut f32, n: c_int) -> c_int;
    fn lcr_op_gate(
        ctx: *mut c_void,
        alpha: *const f32,
        bias: *const f32,
        a: *const f32,
        gate: *mut f32,
        n: c_int,
    ) -> c_int;
    fn lcr_op_silu_mul(
        ctx: *mut c_void,
        a: *const f32,
        b: *const f32,
        y: *mut f32,
        n: c_int,
    ) -> c_int;
    fn lcr_op_conv_silu(
        ctx: *mut c_void,
        w: *const f32,
        hist: *const f32,
        qkv: *const f32,
        out: *mut f32,
        channels: c_int,
        k: c_int,
        stride: c_int,
    ) -> c_int;
    fn lcr_op_gdn(
        ctx: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        n_v: c_int,
        n_k: c_int,
        d: c_int,
    ) -> c_int;
    fn lcr_op_attn(
        ctx: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        kc: *mut f32,
        vc: *mut f32,
        out: *mut f32,
        n_heads: c_int,
        n_kv: c_int,
        d: c_int,
        seq_pos: c_int,
        seq_cap: c_int,
        scale: f32,
    ) -> c_int;
    // Fused per-head kernels.
    fn lcr_op_split_qg(
        ctx: *mut c_void,
        qg: *const f32,
        q: *mut f32,
        g: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
    ) -> c_int;
    fn lcr_op_qk_norm_rope(
        ctx: *mut c_void,
        x: *mut f32,
        w: *const f32,
        n_heads: c_int,
        head_dim: c_int,
        n_rot: c_int,
        base: f32,
        pos: c_int,
    ) -> c_int;
    fn lcr_op_l2norm_heads(
        ctx: *mut c_void,
        x: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
        eps: f32,
    ) -> c_int;
    fn lcr_op_gdn_post(
        ctx: *mut c_void,
        gdn: *const f32,
        z: *const f32,
        norm_w: *const f32,
        y: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
        eps: f32,
    ) -> c_int;
    fn lcr_op_argmax(ctx: *mut c_void, x: *const f32, n: c_int, out: *mut c_int) -> c_int;
}

fn check_rc(op: &str, rc: c_int) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Error::new(format!("{op} failed with code {rc}")))
    }
}

pub struct CudaWeights {
    ctx: *mut std::ffi::c_void,
    offsets: Vec<usize>,
}

impl CudaWeights {
    pub fn new<'a>(
        blob: &[u8],
        weights: impl Iterator<Item = &'a Weight>,
    ) -> crate::core::Result<Self> {
        let mut entries: Vec<(usize, WeightMeta)> = weights
            .map(|w| {
                (
                    w.offset,
                    WeightMeta {
                        type_: w.ty as i32,
                        in_: w.in_features() as i64,
                        out_: w.out_features() as i64,
                        offset: w.offset as i64,
                    },
                )
            })
            .collect();
        // Sort meta/index pairs by offset so binary search finds uploaded weights.
        entries.sort_by_key(|(offset, _)| *offset);

        let offsets: Vec<usize> = entries.iter().map(|(offset, _)| *offset).collect();
        let metas: Vec<WeightMeta> = entries.into_iter().map(|(_, meta)| meta).collect();
        if metas.len() > c_int::MAX as usize {
            return Err(crate::core::Error::new("too many weights for CUDA upload"));
        }

        let mut ctx: *mut std::ffi::c_void = ptr::null_mut();
        let rc = unsafe { lcr_gpu_init(&mut ctx, 0) };
        if rc != 0 || ctx.is_null() {
            if !ctx.is_null() {
                unsafe { lcr_gpu_free(ctx) };
            }
            return Err(crate::core::Error::new(format!(
                "lcr_gpu_init failed with code {rc}"
            )));
        }

        let rc = unsafe {
            lcr_gpu_upload(
                ctx,
                blob.as_ptr().cast(),
                blob.len(),
                metas.as_ptr(),
                metas.len() as c_int,
            )
        };
        if rc != 0 {
            unsafe { lcr_gpu_free(ctx) };
            return Err(crate::core::Error::new(format!(
                "lcr_gpu_upload failed with code {rc}"
            )));
        }

        Ok(Self { ctx, offsets })
    }

    pub fn matvec(&self, w: &Weight, x: &[f32], out: &mut [f32]) -> crate::core::Result<()> {
        let index = self.index_of(w)?;
        if x.len() != w.in_features() || out.len() != w.out_features() {
            return Err(crate::core::Error::new(format!(
                "matvec shape mismatch: x {}, out {}, expected {} x {}",
                x.len(),
                out.len(),
                w.in_features(),
                w.out_features()
            )));
        }
        let rc = unsafe { lcr_gpu_matvec(self.ctx, index, x.as_ptr(), out.as_mut_ptr()) };
        if rc != 0 {
            Err(crate::core::Error::new(format!(
                "lcr_gpu_matvec failed with code {rc}"
            )))
        } else {
            Ok(())
        }
    }

    pub fn row(&self, w: &Weight, row: usize, out: &mut [f32]) -> crate::core::Result<()> {
        let index = self.index_of(w)?;
        if row >= w.out_features() {
            return Err(crate::core::Error::new(format!(
                "row {row} out of range for {} rows",
                w.out_features()
            )));
        }
        let row = c_int::try_from(row)
            .map_err(|_| crate::core::Error::new("row index too large for CUDA"))?;
        if out.len() != w.in_features() {
            return Err(crate::core::Error::new(format!(
                "row shape mismatch: out {}, expected {}",
                out.len(),
                w.in_features()
            )));
        }
        let rc = unsafe { lcr_gpu_row(self.ctx, index, row, out.as_mut_ptr()) };
        if rc != 0 {
            Err(crate::core::Error::new(format!(
                "lcr_gpu_row failed with code {rc}"
            )))
        } else {
            Ok(())
        }
    }

    pub fn info(&self) -> String {
        let mut buf: [c_char; 256] = [0; 256];
        let rc = unsafe { lcr_gpu_info(self.ctx, buf.as_mut_ptr(), buf.len() as c_int) };
        if rc != 0 {
            return String::new();
        }
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| b as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Resolve a weight's device index via the sorted GGUF offsets.
    pub fn index_of(&self, w: &Weight) -> Result<c_int> {
        let i = self
            .offsets
            .binary_search(&w.offset)
            .map_err(|_| Error::new(format!("weight at offset {} not uploaded", w.offset)))?;
        c_int::try_from(i).map_err(|_| Error::new("too many weights for CUDA index"))
    }

    pub(crate) fn buf(&self, bytes: usize) -> Result<*mut c_void> {
        if bytes == 0 {
            return Err(Error::new("GPU buffer allocation needs a non-zero size"));
        }
        let mut out: *mut c_void = ptr::null_mut();
        let rc = unsafe { lcr_gpu_buf(self.ctx, bytes, &mut out) };
        check_rc("lcr_gpu_buf", rc)?;
        if out.is_null() {
            return Err(Error::new("lcr_gpu_buf returned null"));
        }
        Ok(out)
    }

    pub(crate) fn weight_ptr(&self, index: c_int) -> Result<*mut c_void> {
        let ptr = unsafe { lcr_gpu_weight_ptr(self.ctx, index) };
        if ptr.is_null() {
            return Err(Error::new(format!(
                "lcr_gpu_weight_ptr returned null for index {index}"
            )));
        }
        Ok(ptr)
    }

    /// # Safety
    /// `ptr` must come from `buf` on this context and must not be used after this call.
    pub(crate) unsafe fn free_buf(&self, ptr: *mut c_void) {
        if !ptr.is_null() {
            lcr_gpu_buf_free(self.ctx, ptr);
        }
    }

    pub(crate) fn zero(&self, ptr: *mut c_void, bytes: usize) -> Result<()> {
        let rc = unsafe { lcr_gpu_zero(self.ctx, ptr, bytes) };
        check_rc("lcr_gpu_zero", rc)
    }

    pub(crate) fn row_device(&self, index: c_int, row: c_int, dst: *mut f32) -> Result<()> {
        let rc = unsafe { lcr_gpu_row_device(self.ctx, index, row, dst) };
        check_rc("lcr_gpu_row_device", rc)
    }

    pub(crate) fn matvec_device(&self, index: c_int, x: *const f32, out: *mut f32) -> Result<()> {
        let rc = unsafe { lcr_gpu_matvec_device(self.ctx, index, x, out) };
        check_rc("lcr_gpu_matvec_device", rc)
    }

    /// Async batched matmul: `out[t * out_stride + r] = W[index][r] dot x[t * x_stride]`.
    pub(crate) fn batch_matmul(
        &self,
        index: c_int,
        x: *const f32,
        out: *mut f32,
        n_tokens: c_int,
        x_stride: c_int,
        out_stride: c_int,
    ) -> Result<()> {
        let rc =
            unsafe { lcr_batch_matmul(self.ctx, index, x, out, n_tokens, x_stride, out_stride) };
        check_rc("lcr_batch_matmul", rc)
    }

    /// Sequential conv+SiLU over a chunk, updating conv history in place.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batch_conv_silu(
        &self,
        w: *const f32,
        hist: *mut f32,
        qkv: *const f32,
        out: *mut f32,
        n_tokens: c_int,
        qkv_stride: c_int,
        out_stride: c_int,
        channels: c_int,
        k: c_int,
        stride: c_int,
    ) -> Result<()> {
        let rc = unsafe {
            lcr_batch_conv_silu(
                self.ctx, w, hist, qkv, out, n_tokens, qkv_stride, out_stride, channels, k, stride,
            )
        };
        check_rc("lcr_batch_conv_silu", rc)
    }

    /// Sequential GDN recurrence over a chunk; output token stride is `v_stride`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batch_gdn(
        &self,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        n_tokens: c_int,
        q_stride: c_int,
        kv_stride: c_int,
        v_stride: c_int,
        g_stride: c_int,
        n_v: c_int,
        n_k: c_int,
        d: c_int,
    ) -> Result<()> {
        let rc = unsafe {
            lcr_batch_gdn(
                self.ctx, q, k, v, gate, beta, state, out, n_tokens, q_stride, kv_stride, v_stride,
                g_stride, n_v, n_k, d,
            )
        };
        check_rc("lcr_batch_gdn", rc)
    }

    /// Batched KV cache store plus causal attention over a chunk.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batch_attn(
        &self,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        kc: *mut f32,
        vc: *mut f32,
        out: *mut f32,
        n_tokens: c_int,
        q_stride: c_int,
        kv_stride: c_int,
        out_stride: c_int,
        n_heads: c_int,
        n_kv: c_int,
        d: c_int,
        seq_start: c_int,
        seq_cap: c_int,
        scale: f32,
    ) -> Result<()> {
        let rc = unsafe {
            lcr_batch_attn(
                self.ctx, q, k, v, g, kc, vc, out, n_tokens, q_stride, kv_stride, out_stride,
                n_heads, n_kv, d, seq_start, seq_cap, scale,
            )
        };
        check_rc("lcr_batch_attn", rc)
    }

    pub(crate) fn read_i32(&self, src: *const c_void, dst: &mut c_int) -> Result<()> {
        let rc = unsafe { lcr_gpu_read_i32(self.ctx, src, dst) };
        check_rc("lcr_gpu_read_i32", rc)
    }

    pub(crate) fn sync(&self) -> Result<()> {
        let rc = unsafe { lcr_gpu_sync(self.ctx) };
        check_rc("lcr_gpu_sync", rc)
    }

    pub(crate) fn op_rmsnorm(
        &self,
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        n: c_int,
        eps: f32,
    ) -> Result<()> {
        let rc = unsafe { lcr_op_rmsnorm(self.ctx, x, w, y, n, eps) };
        check_rc("lcr_op_rmsnorm", rc)
    }

    #[allow(dead_code)]
    pub(crate) fn op_l2norm(&self, x: *mut f32, n: c_int, eps: f32) -> Result<()> {
        let rc = unsafe { lcr_op_l2norm(self.ctx, x, n, eps) };
        check_rc("lcr_op_l2norm", rc)
    }

    #[allow(dead_code)]
    pub(crate) fn op_rope(
        &self,
        x: *mut f32,
        n: c_int,
        n_rot: c_int,
        base: f32,
        pos: c_int,
    ) -> Result<()> {
        let rc = unsafe { lcr_op_rope(self.ctx, x, n, n_rot, base, pos) };
        check_rc("lcr_op_rope", rc)
    }

    pub(crate) fn op_add(&self, dst: *mut f32, src: *const f32, n: c_int) -> Result<()> {
        let rc = unsafe { lcr_op_add(self.ctx, dst, src, n) };
        check_rc("lcr_op_add", rc)
    }

    #[allow(dead_code)]
    pub(crate) fn op_mul(&self, dst: *mut f32, src: *const f32, n: c_int) -> Result<()> {
        let rc = unsafe { lcr_op_mul(self.ctx, dst, src, n) };
        check_rc("lcr_op_mul", rc)
    }

    #[allow(dead_code)]
    pub(crate) fn op_scale(&self, x: *mut f32, n: c_int, scale: f32) -> Result<()> {
        let rc = unsafe { lcr_op_scale(self.ctx, x, n, scale) };
        check_rc("lcr_op_scale", rc)
    }

    pub(crate) fn op_sigmoid(&self, x: *const f32, y: *mut f32, n: c_int) -> Result<()> {
        let rc = unsafe { lcr_op_sigmoid(self.ctx, x, y, n) };
        check_rc("lcr_op_sigmoid", rc)
    }

    pub(crate) fn op_gate(
        &self,
        alpha: *const f32,
        bias: *const f32,
        a: *const f32,
        gate: *mut f32,
        n: c_int,
    ) -> Result<()> {
        let rc = unsafe { lcr_op_gate(self.ctx, alpha, bias, a, gate, n) };
        check_rc("lcr_op_gate", rc)
    }

    pub(crate) fn op_silu_mul(
        &self,
        a: *const f32,
        b: *const f32,
        y: *mut f32,
        n: c_int,
    ) -> Result<()> {
        let rc = unsafe { lcr_op_silu_mul(self.ctx, a, b, y, n) };
        check_rc("lcr_op_silu_mul", rc)
    }

    pub(crate) fn op_conv_silu(
        &self,
        w: *const f32,
        hist: *const f32,
        qkv: *const f32,
        out: *mut f32,
        channels: c_int,
        k: c_int,
        stride: c_int,
    ) -> Result<()> {
        let rc = unsafe { lcr_op_conv_silu(self.ctx, w, hist, qkv, out, channels, k, stride) };
        check_rc("lcr_op_conv_silu", rc)
    }

    pub(crate) fn op_gdn(
        &self,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        n_v: c_int,
        n_k: c_int,
        d: c_int,
    ) -> Result<()> {
        let rc = unsafe { lcr_op_gdn(self.ctx, q, k, v, gate, beta, state, out, n_v, n_k, d) };
        check_rc("lcr_op_gdn", rc)
    }

    pub(crate) fn op_attn(
        &self,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        kc: *mut f32,
        vc: *mut f32,
        out: *mut f32,
        n_heads: c_int,
        n_kv: c_int,
        d: c_int,
        seq_pos: c_int,
        seq_cap: c_int,
        scale: f32,
    ) -> Result<()> {
        let rc = unsafe {
            lcr_op_attn(
                self.ctx, q, k, v, g, kc, vc, out, n_heads, n_kv, d, seq_pos, seq_cap, scale,
            )
        };
        check_rc("lcr_op_attn", rc)
    }

    pub(crate) fn op_split_qg(
        &self,
        qg: *const f32,
        q: *mut f32,
        g: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
    ) -> Result<()> {
        let rc = unsafe { lcr_op_split_qg(self.ctx, qg, q, g, n_heads, head_dim) };
        check_rc("lcr_op_split_qg", rc)
    }

    pub(crate) fn op_qk_norm_rope(
        &self,
        x: *mut f32,
        w: *const f32,
        n_heads: c_int,
        head_dim: c_int,
        n_rot: c_int,
        base: f32,
        pos: c_int,
    ) -> Result<()> {
        let rc =
            unsafe { lcr_op_qk_norm_rope(self.ctx, x, w, n_heads, head_dim, n_rot, base, pos) };
        check_rc("lcr_op_qk_norm_rope", rc)
    }

    pub(crate) fn op_l2norm_heads(
        &self,
        x: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
        eps: f32,
    ) -> Result<()> {
        let rc = unsafe { lcr_op_l2norm_heads(self.ctx, x, n_heads, head_dim, eps) };
        check_rc("lcr_op_l2norm_heads", rc)
    }

    pub(crate) fn op_gdn_post(
        &self,
        gdn: *const f32,
        z: *const f32,
        norm_w: *const f32,
        y: *mut f32,
        n_heads: c_int,
        head_dim: c_int,
        eps: f32,
    ) -> Result<()> {
        let rc = unsafe { lcr_op_gdn_post(self.ctx, gdn, z, norm_w, y, n_heads, head_dim, eps) };
        check_rc("lcr_op_gdn_post", rc)
    }

    pub(crate) fn op_argmax(&self, x: *const f32, n: c_int, out: *mut c_int) -> Result<()> {
        let rc = unsafe { lcr_op_argmax(self.ctx, x, n, out) };
        check_rc("lcr_op_argmax", rc)
    }
}

impl Drop for CudaWeights {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { lcr_gpu_free(self.ctx) };
        }
    }
}

// Calls are serialized by the single-threaded runtime.
unsafe impl Send for CudaWeights {}
unsafe impl Sync for CudaWeights {}

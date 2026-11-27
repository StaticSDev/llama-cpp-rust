//! FFI bindings to the small CUDA matvec kernels in `cuda/lcr_kernels.cu`.

use crate::runtime::types::Weight;
use std::ffi::{c_char, c_int};
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
        let index = self.index(w)?;
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
        let index = self.index(w)?;
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

    fn index(&self, w: &Weight) -> crate::core::Result<c_int> {
        let i = self.offsets.binary_search(&w.offset).map_err(|_| {
            crate::core::Error::new(format!("weight at offset {} not uploaded", w.offset))
        })?;
        c_int::try_from(i).map_err(|_| crate::core::Error::new("too many weights for CUDA index"))
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

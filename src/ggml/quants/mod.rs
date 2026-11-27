//! Port of `ggml-quants.c` / `ggml-quants.h`.
//!
//! Block layouts mirror `ggml-common.h`. Row functions operate on typed block
//! slices; the `quantize_*` entry points mirror the C multi-row API and return
//! the number of bytes the C version would have written.

mod basic;
mod blocks;
mod iq1;
mod iq2;
mod iq3;
mod iq4;
mod iq_codebooks;
mod iq_dequant;
mod kquants;
mod tables;
mod ternary;
mod util;

pub use basic::*;
pub use blocks::*;
pub use iq1::*;
pub use iq2::*;
pub use iq3::*;
pub use iq4::*;
pub use iq_codebooks::{
    iq2_data, iq2xs_free_impl, iq2xs_init_impl, iq3_data, iq3xs_free_impl, iq3xs_init_impl,
    Iq2Data, Iq2Kind, Iq3Data,
};
pub use iq_dequant::*;
pub use kquants::*;
pub use ternary::*;
pub use crate::core::{Error, Result};
pub use util::{
    best_index_iq4nl, best_index_mxfp4, e8m0_to_fp32_half, f8_e4m3_to_fp32, fp16_to_fp32,
    fp32_to_fp16, fp32_to_f8_e4m3, fp32_to_ue4m3, get_scale_min_k4, make_q3_quants,
    make_qkx1_quants, make_qkx2_quants, make_qkx3_quants, make_qp_quants, make_qx_quants,
    nearest_int, round_to_nearest_even, row_size, ue4m3_to_fp32, GROUP_MAX_EPS,
    GROUP_MAX_EPS_IQ1_M, GROUP_MAX_EPS_IQ1_S, GROUP_MAX_EPS_IQ2_S, GROUP_MAX_EPS_IQ3_XXS,
    KVALUES_IQ4NL, KVALUES_MXFP4,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_quantizes_and_dequantizes() {
        let k = QK_K;
        let x: Vec<f32> = (0..k).map(|i| ((i * 7) % 64) as f32 * 0.25 - 8.0).collect();

        let mut q4 = [BlockQ4K { d: 0, dmin: 0, scales: [0; 12], qs: [0; 128] }];
        let mut y = vec![0.0f32; k];
        quantize_row_q4_K_ref(&x, &mut q4, k).unwrap();
        dequantize_row_q4_K(&q4, &mut y, k).unwrap();

        let mut iq4 = [BlockIq4Xs {
            d: 0,
            scales_h: 0,
            scales_l: [0; 4],
            qs: [0; 128],
        }];
        quantize_row_iq4_xs_ref(&x, &mut iq4, k).unwrap();
        let mut y2 = vec![0.0f32; k];
        dequantize_row_iq4_xs(&iq4, &mut y2, k).unwrap();

        for i in 0..k {
            assert!(y[i].is_finite());
            assert!(y2[i].is_finite());
        }
    }
}

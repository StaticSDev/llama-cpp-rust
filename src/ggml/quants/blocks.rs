//! Block layouts from `ggml-common.h`.
//!
//! Every struct is `#[repr(C)]` so its size matches the C layout.

/// `ggml_half` is an opaque 16-bit storage type. Conversion helpers live in
/// [`super::util`].
pub type F16 = u16;

/// OCP FP8 E4M3 value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Fp8E4M3 {
    pub bits: u8,
}

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
pub const QK1_0: usize = 128;
pub const QK2_0: usize = 64;
pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;
pub const QK_MXFP4: usize = 32;
pub const QK_NVFP4: usize = 64;
pub const QK_NVFP4_SUB: usize = 16;
pub const QK4_NL: usize = 32;
pub const IQ3S_N_SCALE: usize = QK_K / 64;
pub const NGRID_IQ1S: usize = 2048;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ1_0 {
    pub d: F16,
    pub qs: [u8; QK1_0 / 8],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ2_0 {
    pub d: F16,
    pub qs: [u8; QK2_0 / 4],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub d: F16,
    pub qs: [u8; QK4_0 / 2],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub d: F16,
    pub m: F16,
    pub qs: [u8; QK4_1 / 2],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockMxfp4 {
    pub e: u8,
    pub qs: [u8; QK_MXFP4 / 2],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockNvfp4 {
    pub d: [u8; QK_NVFP4 / QK_NVFP4_SUB],
    pub qs: [u8; QK_NVFP4 / 2],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub d: F16,
    pub qh: [u8; 4],
    pub qs: [u8; QK5_0 / 2],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub d: F16,
    pub m: F16,
    pub qh: [u8; 4],
    pub qs: [u8; QK5_1 / 2],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: F16,
    pub qs: [i8; QK8_0],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub d: F16,
    pub s: F16,
    pub qs: [i8; QK8_1],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockTq1_0 {
    pub qs: [u8; (QK_K - 4 * QK_K / 64) / 5],
    pub qh: [u8; QK_K / 64],
    pub d: F16,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockTq2_0 {
    pub qs: [u8; QK_K / 4],
    pub d: F16,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ2K {
    pub scales: [u8; QK_K / 16],
    pub qs: [u8; QK_K / 4],
    pub d: F16,
    pub dmin: F16,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ3K {
    pub hmask: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 4],
    pub scales: [u8; K_SCALE_SIZE],
    pub d: F16,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ4K {
    pub d: F16,
    pub dmin: F16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qs: [u8; QK_K / 2],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ5K {
    pub d: F16,
    pub dmin: F16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qh: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 2],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: F16,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq2Xxs {
    pub d: F16,
    pub qs: [u16; QK_K / 8],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq2Xs {
    pub d: F16,
    pub qs: [u16; QK_K / 8],
    pub scales: [u8; QK_K / 32],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq2S {
    pub d: F16,
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub scales: [u8; QK_K / 32],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq3Xxs {
    pub d: F16,
    pub qs: [u8; 3 * QK_K / 8],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq3S {
    pub d: F16,
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub signs: [u8; QK_K / 8],
    pub scales: [u8; IQ3S_N_SCALE],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq1S {
    pub d: F16,
    pub qs: [u8; QK_K / 8],
    pub qh: [u16; QK_K / 32],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq1M {
    pub qs: [u8; QK_K / 8],
    pub qh: [u8; QK_K / 16],
    pub scales: [u8; QK_K / 32],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq4Nl {
    pub d: F16,
    pub qs: [u8; QK4_NL / 2],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq4Xs {
    pub d: F16,
    pub scales_h: u16,
    pub scales_l: [u8; QK_K / 64],
    pub qs: [u8; QK_K / 2],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_c() {
        assert_eq!(std::mem::size_of::<BlockQ1_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ2_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ4_1>(), 20);
        assert_eq!(std::mem::size_of::<BlockMxfp4>(), 17);
        assert_eq!(std::mem::size_of::<BlockNvfp4>(), 36);
        assert_eq!(std::mem::size_of::<BlockQ5_0>(), 22);
        assert_eq!(std::mem::size_of::<BlockQ5_1>(), 24);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
        assert_eq!(std::mem::size_of::<BlockQ8_1>(), 36);
        assert_eq!(std::mem::size_of::<BlockTq1_0>(), 54);
        assert_eq!(std::mem::size_of::<BlockTq2_0>(), 66);
        assert_eq!(std::mem::size_of::<BlockQ2K>(), 84);
        assert_eq!(std::mem::size_of::<BlockQ3K>(), 110);
        assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
        assert_eq!(std::mem::size_of::<BlockQ5K>(), 176);
        assert_eq!(std::mem::size_of::<BlockQ6K>(), 210);
        assert_eq!(std::mem::size_of::<BlockQ8K>(), 292);
        assert_eq!(std::mem::size_of::<BlockIq2Xxs>(), 66);
        assert_eq!(std::mem::size_of::<BlockIq2Xs>(), 74);
        assert_eq!(std::mem::size_of::<BlockIq2S>(), 82);
        assert_eq!(std::mem::size_of::<BlockIq3Xxs>(), 98);
        assert_eq!(std::mem::size_of::<BlockIq3S>(), 110);
        assert_eq!(std::mem::size_of::<BlockIq1S>(), 50);
        assert_eq!(std::mem::size_of::<BlockIq1M>(), 56);
        assert_eq!(std::mem::size_of::<BlockIq4Nl>(), 18);
        assert_eq!(std::mem::size_of::<BlockIq4Xs>(), 136);
        assert_eq!(std::mem::size_of::<Fp8E4M3>(), 1);
    }
}

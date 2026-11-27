//! Numeric helpers shared by the quantizers.
//!
//! The FP16, E8M0, UE4M3 and FP8 conversions are direct ports of the bit-level
//! routines in `ggml-impl.h`, so results match the C implementation bit for bit.

use crate::core::{Error, Result};
use crate::ggml::quants::blocks::F16;

pub const GROUP_MAX_EPS: f32 = 1.0e-15;
pub const GROUP_MAX_EPS_IQ3_XXS: f32 = 1.0e-8;
pub const GROUP_MAX_EPS_IQ2_S: f32 = 1.0e-8;
pub const GROUP_MAX_EPS_IQ1_M: f32 = 1.0e-7;
pub const GROUP_MAX_EPS_IQ1_S: f32 = 1.0e-12;

pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

pub const KVALUES_MXFP4: [i8; 16] = [
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
];

#[inline]
fn fp32_from_bits(w: u32) -> f32 {
    f32::from_bits(w)
}

#[inline]
fn fp32_to_bits(f: f32) -> u32 {
    f.to_bits()
}

/// `ggml_compute_fp16_to_fp32`.
pub fn fp16_to_fp32(h: F16) -> f32 {
    let w = (h as u32) << 16;
    let sign = w & 0x8000_0000;
    let two_w = w.wrapping_add(w);
    let exp_offset = 0xE0u32 << 23;
    let exp_scale = fp32_from_bits(0x0780_0000);
    let normalized_value = fp32_from_bits((two_w >> 4).wrapping_add(exp_offset)) * exp_scale;
    let magic_mask = 126u32 << 23;
    let magic_bias = 0.5f32;
    let denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
    let denormalized_cutoff = 1u32 << 27;
    let result = sign
        | if two_w < denormalized_cutoff {
            fp32_to_bits(denormalized_value)
        } else {
            fp32_to_bits(normalized_value)
        };
    fp32_from_bits(result)
}

/// `ggml_compute_fp32_to_fp16`.
pub fn fp32_to_fp16(f: f32) -> F16 {
    let scale_to_inf = fp32_from_bits(0x7780_0000);
    let scale_to_zero = fp32_from_bits(0x0880_0000);
    let base = (f.abs() * scale_to_inf) * scale_to_zero;
    let w = fp32_to_bits(f);
    let shl1_w = w.wrapping_add(w);
    let sign = w & 0x8000_0000;
    let mut bias = shl1_w & 0xFF00_0000;
    if bias < 0x7100_0000 {
        bias = 0x7100_0000;
    }
    let base = fp32_from_bits((bias >> 1).wrapping_add(0x0780_0000)) + base;
    let bits = fp32_to_bits(base);
    let exp_bits = (bits >> 13) & 0x0000_7C00;
    let mantissa_bits = bits & 0x0000_0FFF;
    let nonsign = exp_bits + mantissa_bits;
    ((sign >> 16) | if shl1_w > 0xFF00_0000 { 0x7E00 } else { nonsign }) as F16
}

/// `ggml_e8m0_to_fp32_half`.
pub fn e8m0_to_fp32_half(x: u8) -> f32 {
    let bits = if x < 2 {
        0x0020_0000u32 << x
    } else {
        ((x - 1) as u32) << 23
    };
    fp32_from_bits(bits)
}

/// `ggml_ue4m3_to_fp32`: returns the raw UE4M3 value multiplied by 0.5.
pub fn ue4m3_to_fp32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as u32;
    let man = (x & 0x7) as u32;
    let raw = if exp == 0 {
        (man as f32) * fp32_from_bits((118u32) << 23)
    } else {
        (1.0 + man as f32 / 8.0) * fp32_from_bits(exp.wrapping_add(120) << 23)
    };
    raw * 0.5
}

/// `ggml_fp32_to_ue4m3`.
pub fn fp32_to_ue4m3(x: f32) -> u8 {
    if !(x > 0.0) {
        return 0;
    }
    let x = x.min(448.0);
    let bits = fp32_to_bits(x);
    let fp32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp32_man = ((bits >> 20) & 0x7) as i32;
    let mut ue4m3_exp = fp32_exp + 7;
    if ue4m3_exp <= 0 {
        let mut man = (x * 512.0 + 0.5) as i32;
        if man > 7 {
            man = 7;
        }
        if man < 1 {
            return 0;
        }
        return man as u8;
    }
    if ue4m3_exp >= 15 {
        return 0x7E;
    }
    let round_bit = ((bits >> 19) & 1) as i32;
    let mut ue4m3_man = fp32_man + round_bit;
    if ue4m3_man > 7 {
        ue4m3_man = 0;
        ue4m3_exp += 1;
        if ue4m3_exp >= 15 {
            return 0x7E;
        }
    }
    ((ue4m3_exp << 3) | ue4m3_man) as u8
}

/// `ggml_round_to_nearest_even`.
#[inline]
pub fn round_to_nearest_even(x: f32) -> i32 {
    let value = x.floor() as i32;
    let fraction = x - value as f32;
    if fraction > 0.5 || (fraction == 0.5 && (value & 1) != 0) {
        value + 1
    } else {
        value
    }
}

/// `ggml_f8_e4m3_to_fp32`.
pub fn f8_e4m3_to_fp32(x: u8) -> f32 {
    let ax = x & 0x7F;
    if ax == 0x7F {
        return f32::NAN;
    }
    let exp = ((ax >> 3) & 0xF) as u32;
    let man = (ax & 0x7) as u32;
    let value = if exp == 0 {
        (man as f32) * (1.0 / 512.0)
    } else {
        fp32_from_bits(((exp + 120) << 23) | (man << 20))
    };
    fp32_from_bits(value.to_bits() | ((x as u32 & 0x80) << 24))
}

/// `ggml_fp32_to_f8_e4m3`, saturating conversion.
pub fn fp32_to_f8_e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let x = x.abs();
    if x.is_nan() {
        return 0x7F;
    }
    if x == 0.0 {
        return sign;
    }
    if x.is_infinite() || x >= 448.0 {
        return sign | 0x7E;
    }
    if x < 0.015625 {
        return sign | round_to_nearest_even(x * 512.0) as u8;
    }
    let bits = fp32_to_bits(x);
    let exp = ((bits >> 23) & 0xFF) as i32 - 126;
    let mantissa = fp32_from_bits((bits & 0x007F_FFFF) | (127u32 << 23));
    let mut encoded_exp = exp + 6;
    let mut encoded_man = round_to_nearest_even((mantissa - 1.0) * 8.0);
    if encoded_man == 8 {
        encoded_man = 0;
        encoded_exp += 1;
    }
    if encoded_exp > 15 || (encoded_exp == 15 && encoded_man > 6) {
        return sign | 0x7E;
    }
    sign | ((encoded_exp as u8) << 3) | (encoded_man as u8)
}

#[inline]
pub fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4194303.0);
    let val = fval + 12582912.0f32;
    ((val.to_bits() & 0x007f_ffff) as i32) - 0x0040_0000
}

/// Index of the nearest `KVALUES_IQ4NL` value.
pub fn best_index_iq4nl(mid: &[f32; 15], x: f32) -> i32 {
    let mut l = 15;
    for k in 0..15 {
        l -= (x < mid[k]) as i32;
    }
    l
}

/// Index of the nearest MXFP4 code for scale `e`.
pub fn best_index_mxfp4(x: f32, e: f32) -> i32 {
    let mut best_index = 0i32;
    let mut best_err = (KVALUES_MXFP4[0] as f32 * e - x).abs();
    for i in 1..16 {
        let err = (KVALUES_MXFP4[i] as f32 * e - x).abs();
        if err < best_err {
            best_index = i as i32;
            best_err = err;
        }
    }
    best_index
}

fn weight_for(rmse_type: i32, x: f32, qw: Option<f32>) -> f32 {
    match qw {
        Some(w) => w,
        None => match rmse_type {
            1 => x * x,
            2 => 1.0,
            3 => x.abs(),
            _ => x.abs().sqrt(),
        },
    }
}

/// `make_qx_quants`. Writes levels into `l` and returns the block scale.
pub fn make_qx_quants(
    x: &[f32],
    nmax: i32,
    l: &mut [i8],
    rmse_type: i32,
    qw: Option<&[f32]>,
) -> f32 {
    let n = x.len();
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax {
            amax = ax;
            max = x[i];
        }
    }
    if amax < GROUP_MAX_EPS {
        for i in 0..n {
            l[i] = 0;
        }
        return 0.0;
    }
    let mut iscale = -nmax as f32 / max;
    if rmse_type == 0 {
        for i in 0..n {
            let li = nearest_int(iscale * x[i]);
            l[i] = (nmax as i32 + li.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let mut return_early = false;
    let mut rmse_type = rmse_type;
    if rmse_type < 0 {
        rmse_type = -rmse_type;
        return_early = true;
    }
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l[i] = (li + nmax) as i8;
        let w = weight_for(rmse_type, x[i], qw.map(|q| q[i]));
        sumlx += w * x[i] * li as f32;
        suml2 += w * (li * li) as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 {
            0.5 * (scale + 1.0 / iscale)
        } else {
            1.0 / iscale
        };
    }
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let li = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            let w = weight_for(rmse_type, x[i], qw.map(|q| q[i]));
            sumlx += w * x[i] * li as f32;
            suml2 += w * (li * li) as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let li = nearest_int(iscale * x[i]);
                l[i] = (nmax + li.clamp(-nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// `make_q3_quants`.
pub fn make_q3_quants(x: &[f32], nmax: i32, l: &mut [i8], do_rmse: bool) -> f32 {
    let n = x.len();
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax {
            amax = ax;
            max = x[i];
        }
    }
    if amax < GROUP_MAX_EPS {
        for i in 0..n {
            l[i] = 0;
        }
        return 0.0;
    }
    let iscale = -nmax as f32 / max;
    if do_rmse {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0.0f32;
        for i in 0..n {
            let li = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            l[i] = li as i8;
            let w = x[i] * x[i];
            sumlx += w * x[i] * li as f32;
            suml2 += w * (li * li) as f32;
        }
        for _ in 0..5 {
            let mut n_changed = 0;
            for i in 0..n {
                let w = x[i] * x[i];
                let mut slx = sumlx - w * x[i] * l[i] as f32;
                if slx > 0.0 {
                    let mut sl2 = suml2 - w * (l[i] as f32) * (l[i] as f32);
                    let new_l = nearest_int(x[i] * sl2 / slx).clamp(-nmax, nmax - 1);
                    if new_l != l[i] as i32 {
                        slx += w * x[i] * new_l as f32;
                        sl2 += w * (new_l * new_l) as f32;
                        if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                            l[i] = new_l as i8;
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if n_changed == 0 {
                break;
            }
        }
        for i in 0..n {
            l[i] += nmax as i8;
        }
        return if suml2 > 0.0 { sumlx / suml2 } else { 0.0 };
    }
    for i in 0..n {
        let li = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l[i] = (li + nmax) as i8;
    }
    1.0 / iscale
}

/// `make_qkx1_quants`.
pub fn make_qkx1_quants(
    x: &[f32],
    nmax: i32,
    l: &mut [u8],
    the_min: &mut f32,
    ntry: i32,
    alpha: f32,
) -> f32 {
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
    }
    if max == min {
        for i in 0..n {
            l[i] = 0;
        }
        *the_min = 0.0;
        return 0.0;
    }
    if min > 0.0 {
        min = 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    for _ in 0..ntry {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0i32;
        let mut did_change = false;
        for i in 0..n {
            let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            if li as u8 != l[i] {
                l[i] = li as u8;
                did_change = true;
            }
            sumlx += (x[i] - min) * li as f32;
            suml2 += li * li;
        }
        scale = sumlx / suml2 as f32;
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += x[i] - scale * l[i] as f32;
        }
        min = alpha * min + (1.0 - alpha) * sum / n as f32;
        if min > 0.0 {
            min = 0.0;
        }
        iscale = 1.0 / scale;
        if !did_change {
            break;
        }
    }
    *the_min = -min;
    scale
}

/// `make_qkx2_quants`.
#[allow(clippy::too_many_arguments)]
pub fn make_qkx2_quants(
    x: &[f32],
    weights: &[f32],
    nmax: i32,
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        for i in 0..n {
            l[i] = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = li as u8;
        let mut diff = scale * l[i] as f32 + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_error += weights[i] * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = li as u8;
            let w = weights[i];
            sum_l += w * li as f32;
            sum_l2 += w * (li * li) as f32;
            sum_xl += w * li as f32 * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale * laux[i] as f32 + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error += weights[i] * diff;
            }
            if cur_error < best_error {
                for i in 0..n {
                    l[i] = laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// `make_qkx3_quants`.
#[allow(clippy::too_many_arguments)]
pub fn make_qkx3_quants(
    x: &[f32],
    weights: Option<&[f32]>,
    nmax: i32,
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = match weights {
        Some(w) => w[0],
        None => x[0] * x[0],
    };
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = match weights {
            Some(w) => w[i],
            None => x[i] * x[i],
        };
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        for i in 0..n {
            l[i] = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = li as u8;
        let mut diff = scale * l[i] as f32 + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        let w = match weights {
            Some(w) => w[i],
            None => x[i] * x[i],
        };
        best_mad += w * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let v = iscale * (x[i] - min);
            let li = if v > 0.0 {
                nearest_int(v.min(nmax as f32))
            } else {
                0
            };
            laux[i] = li as u8;
            let w = match weights {
                Some(w) => w[i],
                None => x[i] * x[i],
            };
            sum_l += w * li as f32;
            sum_l2 += w * (li * li) as f32;
            sum_xl += w * li as f32 * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale * laux[i] as f32 + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                let w = match weights {
                    Some(w) => w[i],
                    None => x[i] * x[i],
                };
                mad += w * diff;
            }
            if mad < best_mad {
                for i in 0..n {
                    l[i] = laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// `make_qp_quants`: quantize positive scales/mins.
pub fn make_qp_quants(x: &[f32], nmax: i32, l: &mut [u8], quant_weights: &[f32]) -> f32 {
    let n = x.len();
    let mut max = 0.0f32;
    for i in 0..n {
        max = max.max(x[i]);
    }
    if max < GROUP_MAX_EPS {
        for i in 0..n {
            l[i] = 0;
        }
        return 0.0;
    }
    let mut iscale = nmax as f32 / max;
    for i in 0..n {
        l[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0.0f32;
    for i in 0..n {
        let diff = x[i] - scale * l[i] as f32;
        let w = quant_weights[i];
        best_mse += w * diff * diff;
    }
    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0.0f32;
        for i in 0..n {
            let li = nearest_int(iscale_is * x[i]).min(nmax);
            let diff = x[i] - scale_is * li as f32;
            let w = quant_weights[i];
            mse += w * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * x[i]).min(nmax);
        l[i] = li as u8;
        let w = quant_weights[i];
        sumlx += w * x[i] * li as f32;
        suml2 += w * (li * li) as f32;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = quant_weights[i];
            let mut slx = sumlx - w * x[i] * l[i] as f32;
            let mut sl2 = suml2 - w * (l[i] as f32) * (l[i] as f32);
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int(x[i] * sl2 / slx).min(nmax);
                if new_l != l[i] as i32 {
                    slx += w * x[i] * new_l as f32;
                    sl2 += w * (new_l * new_l) as f32;
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 { sumlx / suml2 } else { 0.0 }
}

/// `get_scale_min_k4`.
pub fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// `ggml_row_size` for a fixed block size.
pub fn row_size(numel: usize, block_size: usize, qk: usize) -> Result<usize> {
    if qk == 0 || numel % qk != 0 {
        return Err(Error::new("row size: element count not divisible by block size"));
    }
    Ok(numel / qk * block_size)
}

/// Check that `dst` can hold `count` blocks.
pub fn check_dst<T>(dst: &[T], count: usize) -> Result<()> {
    if dst.len() < count {
        return Err(Error::new("quantize: destination buffer too small"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_round_trip_common_values() {
        for f in [0.0f32, 1.0, -1.0, 0.5, 2.0, -13.75, 100.0, 0.0001] {
            let h = fp32_to_fp16(f);
            let back = fp16_to_fp32(h);
            assert!((back - f).abs() <= f.abs() * 0.001 + 1.0e-6, "{} -> {} -> {}", f, h, back);
        }
    }

    #[test]
    fn fp8_round_trip_common_values() {
        for f in [0.0f32, 1.0, -1.0, 0.5, 2.0, -13.75, 100.0] {
            let bits = fp32_to_f8_e4m3(f);
            let back = f8_e4m3_to_fp32(bits);
            assert!((back - f).abs() <= f.abs() * 0.1 + 1.0e-3, "{} -> {} -> {}", f, bits, back);
        }
    }

    #[test]
    fn nearest_int_matches_round() {
        for (f, expected) in [
            (-7.4f32, -7),
            (-7.5, -8),
            (-7.6, -8),
            (0.0, 0),
            (2.5, 2),
            (2.49, 2),
            (123.5, 124),
        ] {
            assert_eq!(nearest_int(f), expected);
        }
    }

    #[test]
    fn mxfp4_scale_matches_spec() {
        assert_eq!(e8m0_to_fp32_half(127), 0.5);
        assert_eq!(e8m0_to_fp32_half(128), 1.0);
        assert_eq!(ue4m3_to_fp32(fp32_to_ue4m3(6.0)), 3.0);
    }
}

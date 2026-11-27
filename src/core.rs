//! Shared primitives for the Rust port.
//!
//! These types stand in for the small C/C++ helpers that many llama.cpp
//! modules rely on. Ported modules should use them instead of redefining
//! equivalents locally.

use std::fmt;

/// Token id used by the C++ API (`llama_token`).
pub type Token = i32;

/// Sequence id used by the C++ API (`llama_seq_id`).
pub type SeqId = u32;

/// Null token value from `llama.h`.
pub const LLAMA_TOKEN_NULL: Token = -1;

/// Port-wide error type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub message: String,
}

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Direct port of `std::mt19937`, used by the samplers.
///
/// `next_u32` matches the 32-bit output of the C++ engine. `canonical_f64` and
/// `canonical_f32` mirror `std::generate_canonical<double, 53>` and
/// `std::generate_canonical<float, 24>` closely enough for a dry port; exact
/// bit-for-bit parity with libstdc++ distributions is not verified.
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    const N: usize = 624;
    const M: usize = 397;
    const MATRIX_A: u32 = 0x9908_b0df;
    const UPPER_MASK: u32 = 0x8000_0000;
    const LOWER_MASK: u32 = 0x7fff_ffff;

    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; Self::N];
        state[0] = seed;
        for i in 1..Self::N {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: Self::N }
    }

    fn twist(&mut self) {
        for i in 0..Self::N {
            let y = (self.state[i] & Self::UPPER_MASK)
                | (self.state[(i + 1) % Self::N] & Self::LOWER_MASK);
            self.state[i] = self.state[(i + Self::M) % Self::N] ^ (y >> 1);
            if y & 1 != 0 {
                self.state[i] ^= Self::MATRIX_A;
            }
        }
        self.index = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= Self::N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `std::uniform_real_distribution<double>(0, 1)`-style draw.
    pub fn canonical_f64(&mut self) -> f64 {
        let mut sum = 0.0f64;
        let mut scale = 1.0f64;
        for _ in 0..2 {
            sum += self.next_u32() as f64 * scale;
            scale *= 18446744073709551616.0f64;
        }
        sum / 18446744073709551616.0f64
    }

    /// `std::uniform_real_distribution<float>(0, 1)`-style draw.
    pub fn canonical_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16777216.0f32
    }

    /// Normal draw used by samplers in place of `std::normal_distribution`.
    pub fn normal_f32(&mut self) -> f32 {
        let u1 = self.canonical_f32().max(1.0e-7);
        let u2 = self.canonical_f32();
        (-2.0f32 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Alias for modules that call the engine `Rng`.
pub type Rng = Mt19937;

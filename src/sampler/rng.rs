use crate::core::Rng;
use std::time::{SystemTime, UNIX_EPOCH};

use super::candidate::LLAMA_DEFAULT_SEED;

pub fn get_rng_seed(seed: u32) -> u32 {
    if seed == LLAMA_DEFAULT_SEED {
        // C++ uses system clock when std::random_device is not a true RNG.
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(0)
    } else {
        seed
    }
}

/// `std::mt19937` plus the seed and draw count, so a live RNG can be cloned.
/// Cloning replays draws from the seed so the draw count stays explicit.
#[derive(Debug)]
pub(crate) struct SamplerRng {
    seed: u32,
    draws: u64,
    rng: Rng,
}

impl SamplerRng {
    pub(crate) fn new(seed: u32) -> Self {
        Self {
            seed,
            draws: 0,
            rng: Rng::new(seed),
        }
    }

    pub(crate) fn reseed(&mut self, seed: u32) {
        self.seed = seed;
        self.draws = 0;
        self.rng = Rng::new(seed);
    }

    /// `std::generate_canonical<double, 53>` with a 32-bit engine.
    pub(crate) fn uniform_f64(&mut self) -> f64 {
        let hi = self.rng.next_u32();
        let lo = self.rng.next_u32();
        self.draws += 2;
        (hi as f64 * 4294967296.0 + lo as f64) / 18446744073709551616.0
    }

    /// `std::generate_canonical<float, 24>`.
    pub(crate) fn uniform_f32(&mut self) -> f32 {
        let v = self.rng.canonical_f32();
        self.draws += 1;
        v
    }

    fn replicate(&self) -> Self {
        let mut rng = Rng::new(self.seed);
        for _ in 0..self.draws {
            let _ = rng.next_u32();
        }
        Self {
            seed: self.seed,
            draws: self.draws,
            rng,
        }
    }
}

impl Clone for SamplerRng {
    fn clone(&self) -> Self {
        self.replicate()
    }
}


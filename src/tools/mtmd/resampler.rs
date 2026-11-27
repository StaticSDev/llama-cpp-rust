//! Streaming mono F32 resampler (`mtmd_helper_resampler` in mtmd-helper.cpp).
//!
//! miniaudio's linear resampler is reproduced with an interpolated sample
//! position. Its optional low-pass filter is not available in `std`; the host
//! can pre-filter before calling `process`.

#[derive(Debug, Clone)]
pub struct Resampler {
    rate_in: i32,
    rate_out: i32,
    passthrough: bool,
    out: Vec<f32>,
    total_in: u64,
    next_pos: f64,
}

impl Resampler {
    pub fn new(rate_in: i32, rate_out: i32) -> Option<Self> {
        if rate_in <= 0 || rate_out <= 0 {
            return None;
        }
        Some(Self {
            rate_in,
            rate_out,
            passthrough: rate_in == rate_out,
            out: Vec::new(),
            total_in: 0,
            next_pos: 0.0,
        })
    }

    pub fn is_passthrough(&self) -> bool {
        self.passthrough
    }

    pub fn rate_in(&self) -> i32 {
        self.rate_in
    }

    pub fn rate_out(&self) -> i32 {
        self.rate_out
    }

    /// `*out` in C is valid until the next call; here the slice is owned by
    /// `self` and has the same lifetime.
    pub fn process(&mut self, input: &[f32]) -> Result<&[f32], i32> {
        self.out.clear();
        if self.passthrough {
            self.out.extend_from_slice(input);
            self.total_in = self.total_in.saturating_add(input.len() as u64);
            self.next_pos = self.total_in as f64;
            return Ok(&self.out);
        }

        let start = self.total_in as f64;
        let end = start + input.len() as f64;
        let step = self.rate_in as f64 / self.rate_out as f64;
        self.out.reserve((input.len() as f64 / step) as usize + 16);

        while self.next_pos < end {
            let rel = self.next_pos - start;
            let i = rel as usize;
            let frac = (rel - i as f64) as f32;
            let a = input[i];
            let b = input.get(i + 1).copied().unwrap_or(a);
            self.out.push(a + (b - a) * frac);
            self.next_pos += step;
        }
        self.total_in = self.total_in.saturating_add(input.len() as u64);
        Ok(&self.out)
    }
}

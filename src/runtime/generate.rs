//! Greedy and top-p text generation on the small CPU runtime.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::core::{Error, Mt19937, Result, Token};
use crate::runtime::qwen35;
use crate::runtime::types::{ModelState, Sampling, Weights};

fn forward_logits(w: &Weights, st: &mut ModelState, tok: Token) -> Result<Vec<f32>> {
    let logits = qwen35::forward(w, st, tok)?;
    if logits.is_empty() {
        return Err(Error::new("forward produced no logits"));
    }
    Ok(logits)
}

/// Advance the model by `token` and return the candidate for the next step.
///
/// The GPU graph returns greedy tokens; the CPU path samples from logits.
pub fn next_token(w: &Weights, st: &mut ModelState, token: i32) -> Result<Token> {
    #[cfg(feature = "cuda")]
    if let Some(mut graph) = w.gpu_graph_lock() {
        let next = graph.forward_token(w, token, st.pos)?;
        st.pos += 1;
        return Ok(next);
    }
    let logits = forward_logits(w, st, token)?;
    Ok(match st.sampling.as_mut() {
        Some(s) => sample_top_p(&logits, s.temp, s.top_p, &mut s.rng),
        None => greedy(&logits),
    })
}

/// Counters collected during one generation run.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub prompt_ms: u128,
    pub tokens: usize,
    pub gen_ms: u128,
}

impl Stats {
    pub fn prompt_tokens_per_sec(&self) -> f32 {
        per_sec(self.prompt_tokens, self.prompt_ms)
    }

    pub fn tokens_per_sec(&self) -> f32 {
        per_sec(self.tokens, self.gen_ms)
    }
}

fn per_sec(count: usize, ms: u128) -> f32 {
    if ms == 0 {
        return 0.0;
    }
    count as f32 * 1000.0 / ms as f32
}

/// Progress reported while a turn is being generated.
pub enum Event<'a> {
    /// Prompt token `done` of `total` has been processed.
    Prompt { done: usize, total: usize },
    /// A piece of generated text.
    Token(&'a str),
}

/// Run one Qwen chat turn and return the generated text (without chat tags).
pub fn generate(
    w: &Weights,
    prompt: &str,
    n_predict: usize,
    temp: f32,
    top_p: f32,
) -> Result<String> {
    generate_stream(w, prompt, n_predict, temp, top_p, |_| {}).map(|(text, _)| text)
}

/// Same as [`generate`], reporting progress through `on_event` and returning
/// token counts plus timings.
pub fn generate_stream<F: FnMut(Event)>(
    w: &Weights,
    prompt: &str,
    n_predict: usize,
    temp: f32,
    top_p: f32,
    mut on_event: F,
) -> Result<(String, Stats)> {
    #[cfg(feature = "cuda")]
    if temp > 0.0 && w.gpu_graph_lock().is_some() {
        return Err(Error::new(
            "top-p sampling on GPU is not ported yet; use --temp 0",
        ));
    }

    #[cfg(feature = "cuda")]
    {
        if let Some(graph) = w.gpu_graph_lock() {
            graph.reset()?;
        }
    }

    let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let prompt_tokens = w.vocab.tokenize(&formatted, true, true);
    let mut st = ModelState::new(w);
    let mut stats = Stats::default();

    let prompt_start = Instant::now();
    // Sampling stays off during prefill: it must not consume RNG draws.
    let mut candidate = if prompt_tokens.is_empty() {
        next_token(w, &mut st, w.vocab.token_bos())?
    } else {
        let total = prompt_tokens.len();
        #[cfg(feature = "cuda")]
        let gpu_candidate = {
            let mut graph = w.gpu_graph_lock();
            match graph.as_mut() {
                Some(graph) => Some(graph.prefill_tokens(w, &prompt_tokens, st.pos)?),
                None => None,
            }
        };
        #[cfg(not(feature = "cuda"))]
        let gpu_candidate: Option<i32> = None;

        if let Some(next) = gpu_candidate {
            st.pos += total;
            // Batched prefill cannot report between tokens; replay the events.
            for done in 1..=total {
                on_event(Event::Prompt { done, total });
            }
            next
        } else {
            let mut last = 0;
            for (i, &tok) in prompt_tokens.iter().enumerate() {
                last = next_token(w, &mut st, tok)?;
                on_event(Event::Prompt { done: i + 1, total });
            }
            last
        }
    };
    stats.prompt_tokens = prompt_tokens.len().max(1);
    stats.prompt_ms = prompt_start.elapsed().as_millis();

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u32;
    let rng = Mt19937::new(seed);
    if temp > 0.0 {
        st.sampling = Some(Sampling { temp, top_p, rng });
    }
    let eos = w.vocab.token_eos();
    let eot = w.vocab.token_eot();
    let mut out = String::new();
    let gen_start = Instant::now();

    for _ in 0..n_predict {
        let tok = candidate;

        if tok == eos || tok == eot {
            break;
        }

        let id = tok as usize;
        if id < w.vocab.n_tokens() as usize && !w.vocab.is_control(tok) && !w.vocab.is_unknown(tok)
        {
            let piece = w.vocab.token_to_piece_cached(tok);
            if !piece.is_empty() {
                out.push_str(piece);
                on_event(Event::Token(piece));
            }
        }

        stats.tokens += 1;
        stats.gen_ms = gen_start.elapsed().as_millis();
        candidate = next_token(w, &mut st, tok)?;
    }

    stats.gen_ms = gen_start.elapsed().as_millis();
    Ok((out, stats))
}

fn greedy(logits: &[f32]) -> Token {
    let mut best = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    best as Token
}

fn sample_top_p(logits: &[f32], temp: f32, top_p: f32, rng: &mut Mt19937) -> Token {
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = scaled.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();

    let mut candidates: Vec<(f32, Token)> = probs
        .iter()
        .enumerate()
        .filter(|(_, &p)| p > 0.0 && sum > 0.0)
        .map(|(i, &p)| (p / sum, i as Token))
        .collect();
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let cutoff = top_p.clamp(0.0, 1.0);
    let mut nucleus: Vec<(f32, Token)> = Vec::with_capacity(candidates.len());
    let mut cumulative = 0.0f32;
    for candidate in candidates {
        cumulative += candidate.0;
        nucleus.push(candidate);
        if cumulative >= cutoff {
            break;
        }
    }

    if nucleus.is_empty() {
        return greedy(logits);
    }

    let total: f64 = nucleus.iter().map(|&(p, _)| p as f64).sum();
    let threshold = rng.canonical_f64() * total;
    let mut acc = 0.0f64;
    let mut last = nucleus[nucleus.len() - 1].1;
    for (p, tok) in nucleus {
        last = tok;
        acc += p as f64;
        if threshold < acc {
            return tok;
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_rate_handles_zero_time() {
        let stats = Stats::default();
        assert_eq!(stats.tokens_per_sec(), 0.0);
        assert_eq!(stats.prompt_tokens_per_sec(), 0.0);
    }

    #[test]
    fn stats_rate_is_tokens_over_seconds() {
        let stats = Stats {
            prompt_tokens: 20,
            prompt_ms: 2_000,
            tokens: 10,
            gen_ms: 500,
        };
        assert!((stats.prompt_tokens_per_sec() - 10.0).abs() < 1e-4);
        assert!((stats.tokens_per_sec() - 20.0).abs() < 1e-4);
    }
}

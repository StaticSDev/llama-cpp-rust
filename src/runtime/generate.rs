//! Greedy and top-p text generation on the small CPU runtime.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::{Error, Mt19937, Result, Token};
use crate::runtime::qwen35;
use crate::runtime::types::{ModelState, Weights};

fn forward_logits(w: &Weights, st: &mut ModelState, tok: Token) -> Result<Vec<f32>> {
    let logits = qwen35::forward(w, st, tok)?;
    if logits.is_empty() {
        return Err(Error::new("forward produced no logits"));
    }
    Ok(logits)
}

/// Run one Qwen chat turn and return the generated text (without chat tags).
pub fn generate(
    w: &Weights,
    prompt: &str,
    n_predict: usize,
    temp: f32,
    top_p: f32,
) -> Result<String> {
    let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let prompt_tokens = w.vocab.tokenize(&formatted, true, true);
    let mut st = ModelState::new(w);

    let mut logits = if prompt_tokens.is_empty() {
        forward_logits(w, &mut st, w.vocab.token_bos())?
    } else {
        let mut last = Vec::new();
        for &tok in &prompt_tokens {
            last = forward_logits(w, &mut st, tok)?;
        }
        last
    };

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u32;
    let mut rng = Mt19937::new(seed);
    let eos = w.vocab.token_eos();
    let eot = w.vocab.token_eot();
    let mut out = String::new();

    for _ in 0..n_predict {
        let tok = if temp <= 0.0 {
            greedy(&logits)
        } else {
            sample_top_p(&logits, temp, top_p, &mut rng)
        };

        if tok == eos || tok == eot {
            break;
        }

        let id = tok as usize;
        if id < w.vocab.n_tokens() as usize && !w.vocab.is_control(tok) && !w.vocab.is_unknown(tok)
        {
            out.push_str(w.vocab.token_to_piece_cached(tok));
        }

        logits = forward_logits(w, &mut st, tok)?;
    }

    Ok(out)
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

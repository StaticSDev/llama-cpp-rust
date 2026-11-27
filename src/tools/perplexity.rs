//! Rust port of `tools/perplexity/perplexity.cpp` and `tools/perplexity/main.cpp`.
//!
//! Model execution stays outside this module. Callers supply the logits that
//! `llama_process` would produce through a callback, so tokenization, chunk
//! planning, evaluation and the PPL/KLD metrics need no FFI.

use crate::common::common_tokenize;
use crate::core::{Error, Result, Token};
use crate::vocab::LlamaVocab;
use std::fs::File;
use std::io::{Read, Write};

// ---------------------------------------------------------------------------
// params (llama_perplexity in main.cpp body)
// ---------------------------------------------------------------------------

/// Fields of `common_params` that the perplexity tool reads or adjusts.
#[derive(Clone, Debug)]
pub struct PerplexityParams {
    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_chunks: i32,
    pub n_parallel: i32,
    pub ppl_stride: i32,
    pub ppl_output_type: i32,
    pub prompt: String,
    pub logits_file: String,
    pub hellaswag: bool,
    pub winogrande: bool,
    pub multiple_choice: bool,
    pub kl_divergence: bool,
    pub kv_unified: bool,
}

impl Default for PerplexityParams {
    fn default() -> Self {
        Self {
            n_ctx: 512,
            n_batch: 2048,
            n_chunks: -1,
            n_parallel: 1,
            ppl_stride: 0,
            ppl_output_type: 0,
            prompt: String::new(),
            logits_file: String::new(),
            hellaswag: false,
            winogrande: false,
            multiple_choice: false,
            kl_divergence: false,
            kv_unified: false,
        }
    }
}

/// Params after the setup at the start of `llama_perplexity`.
#[derive(Clone, Debug)]
pub struct PreparedParams {
    pub params: PerplexityParams,
    /// `n_ctx` before the parallel and stride adjustments. Passed to the
    /// non-strided evaluator, matching the third argument of `perplexity()`.
    pub n_ctx: i32,
}

/// Port of the param setup at the top of `llama_perplexity`.
pub fn prepare_params(mut params: PerplexityParams) -> Result<PreparedParams> {
    let n_ctx = params.n_ctx;
    if n_ctx <= 0 {
        return Err(Error::new(
            "perplexity tool requires '--ctx-size' > 0",
        ));
    }

    if params.hellaswag || params.winogrande || params.multiple_choice {
        params.n_parallel = params.n_parallel.max(4);
        params.kv_unified = true;
    } else {
        params.n_parallel = (params.n_batch / n_ctx).max(1);
    }

    params.n_ctx = params.n_parallel * n_ctx;
    params.n_batch = params.n_batch.min(params.n_ctx);

    if params.ppl_stride > 0 {
        params.n_ctx += params.ppl_stride / 2;
    }

    Ok(PreparedParams { params, n_ctx })
}

fn positive(value: i32, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::new(format!("{name} must be positive")))
}

fn nonnegative(value: i32, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::new(format!("{name} must not be negative")))
}

// ---------------------------------------------------------------------------
// tokenization
// ---------------------------------------------------------------------------

/// `common_tokenize(ctx, params.prompt, true)` from both evaluators.
pub fn tokenize_prompt(vocab: &LlamaVocab, params: &PerplexityParams) -> Result<Vec<Token>> {
    common_tokenize(vocab, &params.prompt, true, false)
}

/// Reject vocabularies that add EOS, as `GGML_ASSERT(!add_eos)` does.
fn check_vocab(vocab: &LlamaVocab) -> Result<()> {
    if vocab.get_add_eos() {
        return Err(Error::new(
            "perplexity requires a vocabulary with add_eos == false",
        ));
    }
    Ok(())
}

fn check_token_count(tokens: &[Token], n_ctx: usize) -> Result<()> {
    if tokens.len() < 2 * n_ctx {
        return Err(Error::new(format!(
            "you need at least {} tokens to evaluate perplexity with a context of {}; the data file tokenizes to only {} tokens",
            2 * n_ctx,
            n_ctx,
            tokens.len()
        )));
    }
    Ok(())
}

fn chunk_count(total: usize, chunk: usize, n_chunks: i32) -> usize {
    let max = total / chunk;
    if n_chunks < 0 {
        max
    } else {
        (n_chunks as usize).min(max)
    }
}

// ---------------------------------------------------------------------------
// softmax / log-softmax
// ---------------------------------------------------------------------------

/// `results_perplexity` from perplexity.cpp.
#[derive(Clone, Debug, Default)]
pub struct PerplexityResults {
    pub tokens: Vec<Token>,
    pub ppl_value: f64,
    /// Per-token logits of the predicted token (`logit_history` in C++).
    pub logits: Vec<f32>,
    /// Per-token probabilities (`prob_history` in C++).
    pub probs: Vec<f32>,
}

/// Direct port of `softmax`.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut probs = vec![0.0f32; logits.len()];
    if logits.is_empty() {
        return probs;
    }
    let mut max_logit = logits[0];
    for &v in logits {
        max_logit = max_logit.max(v);
    }
    let mut sum_exp = 0.0f64;
    for (i, &logit) in logits.iter().enumerate() {
        let exp_logit = (logit - max_logit).exp();
        sum_exp += f64::from(exp_logit);
        probs[i] = exp_logit;
    }
    for p in &mut probs {
        *p = (f64::from(*p) / sum_exp) as f32;
    }
    probs
}

/// `results_log_softmax` from perplexity.cpp.
#[derive(Clone, Copy, Debug)]
pub struct LogSoftmaxResult {
    pub log_softmax: f64,
    pub logit: f32,
    pub prob: f32,
}

/// Direct port of `log_softmax(n_vocab, logits, tok)`.
pub fn log_softmax(n_vocab: usize, logits: &[f32], tok: usize) -> Result<LogSoftmaxResult> {
    if n_vocab == 0 || logits.len() < n_vocab || tok >= n_vocab {
        return Err(Error::new("invalid logits or token for log_softmax"));
    }
    let row = &logits[..n_vocab];
    let mut max_logit = row[0];
    for &v in row {
        max_logit = max_logit.max(v);
    }
    let mut sum_exp = 0.0f64;
    for &v in row {
        sum_exp += f64::from((v - max_logit).exp());
    }
    let log_sum_exp = sum_exp.ln();
    Ok(LogSoftmaxResult {
        log_softmax: f64::from(row[tok] - max_logit) - log_sum_exp,
        logit: row[tok],
        prob: (row[tok] - max_logit).exp() / sum_exp as f32,
    })
}

/// `nearest_int`: the float-to-int bit trick used by the quantized dump.
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0f32;
    let bits = val.to_bits() as i32;
    (bits & 0x007fffff) - 0x00400000
}

/// Store an `f32` in two `uint16_t` slots, as the C++ float/u16 overlay does
/// on a little-endian host.
fn store_f32(dst: &mut [u16], value: f32) {
    let bits = value.to_bits();
    dst[0] = bits as u16;
    dst[1] = (bits >> 16) as u16;
}

/// Read an `f32` back from two `uint16_t` slots.
fn load_f32(src: &[u16]) -> f32 {
    let bits = u32::from(src[0]) | (u32::from(src[1]) << 16);
    f32::from_bits(bits)
}

/// Number of `uint16_t` slots per token in a log-prob dump.
pub fn log_prob_stride(n_vocab: usize) -> usize {
    2 * ((n_vocab + 1) / 2) + 4
}

/// The quantized `log_softmax` used when saving logits to a file.
///
/// Writes `scale`, `min_log_prob` as two floats in the first four slots and
/// the packed 16-bit log probabilities from slot four onward.
pub fn log_softmax_quantized(
    n_vocab: usize,
    logits: &[f32],
    log_prob: &mut [u16],
    tok: usize,
) -> Result<f64> {
    if n_vocab == 0 || logits.len() < n_vocab || tok >= n_vocab {
        return Err(Error::new("invalid logits or token for log_softmax"));
    }
    if log_prob.len() < log_prob_stride(n_vocab) {
        return Err(Error::new("log probability buffer is too small"));
    }

    let row = &logits[..n_vocab];
    let mut max_logit = row[0];
    let mut min_logit = row[0];
    for &v in row {
        max_logit = max_logit.max(v);
        min_logit = min_logit.min(v);
    }
    min_logit = min_logit.max(max_logit - 16.0);

    let mut sum_exp = 0.0f64;
    for &v in row {
        sum_exp += f64::from((v - max_logit).exp());
    }
    let log_sum_exp = sum_exp.ln() as f32;
    let min_log_prob = min_logit - max_logit - log_sum_exp;
    let scale = (max_logit - min_logit) / 65535.0f32;

    store_f32(&mut log_prob[0..2], scale);
    store_f32(&mut log_prob[2..4], min_log_prob);

    if scale != 0.0 {
        let inv_scale = 1.0 / scale;
        for i in 0..n_vocab {
            log_prob[4 + i] = if row[i] > min_logit {
                nearest_int(inv_scale * (row[i] - min_logit)) as u16
            } else {
                0
            };
        }
    } else {
        for slot in &mut log_prob[4..4 + n_vocab] {
            *slot = 0;
        }
    }

    Ok(f64::from(max_logit + log_sum_exp - row[tok]))
}

// ---------------------------------------------------------------------------
// log-likelihood accumulation (process_logits in perplexity.cpp)
// ---------------------------------------------------------------------------

/// Accumulated negative log-likelihood and its square.
#[derive(Clone, Copy, Debug, Default)]
pub struct NllAcc {
    pub nll: f64,
    pub nll2: f64,
}

fn logits_row(logits: &[f32], i: usize, n_vocab: usize) -> Result<&[f32]> {
    let start = i
        .checked_mul(n_vocab)
        .ok_or_else(|| Error::new("logits index overflow"))?;
    let end = start
        .checked_add(n_vocab)
        .ok_or_else(|| Error::new("logits index overflow"))?;
    logits
        .get(start..end)
        .ok_or_else(|| Error::new("not enough logits for the evaluation"))
}

fn target_token(tokens: &[Token], i: usize, n_vocab: usize) -> Result<usize> {
    let tok = usize::try_from(tokens[i + 1])
        .map_err(|_| Error::new("negative or oversized target token"))?;
    if tok >= n_vocab {
        return Err(Error::new("target token outside the vocabulary"));
    }
    Ok(tok)
}

/// Port of the first `process_logits` overload. `tokens` holds `n_token + 1`
/// tokens; row `i` of `logits` predicts `tokens[i + 1]`.
pub fn process_logits(
    n_vocab: usize,
    logits: &[f32],
    tokens: &[Token],
    logit_history: &mut [f32],
    prob_history: &mut [f32],
) -> Result<NllAcc> {
    let n_token = tokens.len().saturating_sub(1);
    if logit_history.len() < n_token || prob_history.len() < n_token {
        return Err(Error::new("history buffer is too small"));
    }

    let mut acc = NllAcc::default();
    for i in 0..n_token {
        let tok = target_token(tokens, i, n_vocab)?;
        let res = log_softmax(n_vocab, logits_row(logits, i, n_vocab)?, tok)?;
        let v = -res.log_softmax;
        acc.nll += v;
        acc.nll2 += v * v;
        logit_history[i] = res.logit;
        prob_history[i] = res.prob;
    }
    Ok(acc)
}

/// Port of the ostream `process_logits` overload. Fills `log_probs` with one
/// `log_prob_stride(n_vocab)` block per token and accumulates the NLL.
pub fn process_logits_quantized(
    n_vocab: usize,
    logits: &[f32],
    tokens: &[Token],
    log_probs: &mut [u16],
) -> Result<NllAcc> {
    let n_token = tokens.len().saturating_sub(1);
    let nv = log_prob_stride(n_vocab);
    if log_probs.len() < n_token * nv {
        return Err(Error::new("log probability buffer is too small"));
    }

    let mut acc = NllAcc::default();
    for i in 0..n_token {
        let tok = target_token(tokens, i, n_vocab)?;
        let block = &mut log_probs[i * nv..(i + 1) * nv];
        let v = log_softmax_quantized(n_vocab, logits_row(logits, i, n_vocab)?, block, tok)?;
        acc.nll += v;
        acc.nll2 += v * v;
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// strided perplexity (perplexity_v2)
// ---------------------------------------------------------------------------

/// Supplies the logits for one chunk.
///
/// Arguments are the chunk tokens with the BOS substitution already applied
/// at each sequence start, the sequence count, the first output position and
/// the vocabulary size. The returned slice holds one `n_vocab`-wide row per
/// requested output position, in sequence-major order.
pub type LogitsProvider<'a> = dyn FnMut(&[Token], usize, usize, usize) -> Result<Vec<f32>> + 'a;

/// Port of `perplexity_v2`.
pub fn perplexity_v2<F>(
    params: &PerplexityParams,
    tokens: Vec<Token>,
    vocab: &LlamaVocab,
    mut eval: F,
) -> Result<PerplexityResults>
where
    F: FnMut(&[Token], usize, usize, usize) -> Result<Vec<f32>>,
{
    check_vocab(vocab)?;
    let add_bos = vocab.get_add_bos();
    let n_ctx = positive(params.n_ctx, "n_ctx")?;
    let stride = positive(params.ppl_stride, "ppl_stride")?;
    let n_vocab = positive(vocab.n_tokens() as i32, "n_vocab")?;

    check_token_count(&tokens, n_ctx)?;

    let mut logit_history = vec![0.0f32; tokens.len()];
    let mut prob_history = vec![0.0f32; tokens.len()];

    let calc_chunk = n_ctx;
    if tokens.len() <= calc_chunk {
        return Err(Error::new(format!(
            "there are only {} tokens, not enough for a context of {} and stride {}",
            tokens.len(), n_ctx, stride
        )));
    }
    let n_chunk_max = (tokens.len() - calc_chunk + stride - 1) / stride;
    let n_chunk = if params.n_chunks < 0 {
        n_chunk_max
    } else {
        nonnegative(params.n_chunks, "n_chunks")?.min(n_chunk_max)
    };

    let j_start = n_ctx.saturating_sub(stride).saturating_sub(1);
    let j_end = n_ctx.saturating_sub(1);
    let mut count = 0usize;
    let mut nll = 0.0f64;

    for i in 0..n_chunk {
        let start = i * stride;
        let end = start + calc_chunk;
        let mut chunk = tokens[start..end].to_vec();
        if add_bos {
            chunk[0] = vocab.token_bos();
        }

        let logits = eval(&chunk, 1, 0, n_vocab)?;
        if logits.len() < calc_chunk * n_vocab {
            return Err(Error::new("not enough logits for the chunk"));
        }

        for j in j_start..j_end {
            let row = logits_row(&logits, j, n_vocab)?;
            let tok = target_token(&tokens, start + j, n_vocab)?;
            let prob = softmax(row)[tok];
            let dst = start + j + 1;
            logit_history[dst] = row[tok];
            prob_history[dst] = prob;
            nll -= f64::from(prob).ln();
            count += 1;
        }

        if count > 0 && params.ppl_output_type == 0 {
            eprint!("[{}]{:.4},", i + 1, (nll / count as f64).exp());
        } else if count > 0 {
            eprintln!("{:8}  {:.4}", i * stride, (nll / count as f64).exp());
        }
    }
    eprintln!();

    let ppl_value = if count > 0 { (nll / count as f64).exp() } else { 0.0 };
    Ok(PerplexityResults {
        tokens,
        ppl_value,
        logits: logit_history,
        probs: prob_history,
    })
}

// ---------------------------------------------------------------------------
// non-strided perplexity (perplexity)
// ---------------------------------------------------------------------------

/// Port of `perplexity` (called with `ppl_stride <= 0`).
pub fn perplexity<F>(
    params: &PerplexityParams,
    n_ctx: i32,
    tokens: Vec<Token>,
    vocab: &LlamaVocab,
    mut eval: F,
) -> Result<PerplexityResults>
where
    F: FnMut(&[Token], usize, usize, usize) -> Result<Vec<f32>>,
{
    check_vocab(vocab)?;
    let add_bos = vocab.get_add_bos();
    let n_ctx_usize = positive(n_ctx, "n_ctx")?;
    let n_batch = positive(params.n_batch, "n_batch")?;
    let n_vocab = positive(vocab.n_tokens() as i32, "n_vocab")?;

    check_token_count(&tokens, n_ctx_usize)?;

    let n_chunk = chunk_count(tokens.len(), n_ctx_usize, params.n_chunks);
    let n_seq = (n_batch / n_ctx_usize).max(1);
    if n_batch >= n_ctx_usize && n_batch % n_ctx_usize != 0 {
        return Err(Error::new("n_batch must be smaller than n_ctx or a multiple of it"));
    }
    if positive(params.n_ctx, "n_ctx")? != n_seq * n_ctx_usize {
        return Err(Error::new("params.n_ctx must equal n_seq * n_ctx"));
    }

    let mut logit_history = vec![0.0f32; tokens.len()];
    let mut prob_history = vec![0.0f32; tokens.len()];

    let first = n_ctx_usize / 2;
    let n_token = n_ctx_usize - 1 - first;
    let mut count = 0usize;
    let mut nll = 0.0f64;
    let mut nll2 = 0.0f64;

    for i in (0..n_chunk).step_by(n_seq) {
        let start = i * n_ctx_usize;
        let n_seq_batch = n_seq.min(n_chunk - i);

        let mut chunk = Vec::with_capacity(n_seq_batch * n_ctx_usize);
        for seq in 0..n_seq_batch {
            let seq_start = start + seq * n_ctx_usize;
            chunk.extend_from_slice(&tokens[seq_start..seq_start + n_ctx_usize]);
            if add_bos {
                chunk[seq * n_ctx_usize] = vocab.token_bos();
            }
        }

        let logits = eval(&chunk, n_seq_batch, first, n_vocab)?;

        for seq in 0..n_seq_batch {
            let tok_start = start + seq * n_ctx_usize + first;
            // `process_logits` reads tokens[i+1], so it needs n_token + 1 tokens.
            let targets = tokens
                .get(tok_start..tok_start + n_token + 1)
                .ok_or_else(|| Error::new("chunk extends past the token list"))?;
            let rows_per_seq = n_ctx_usize - first;
            let mut chunk_rows = Vec::with_capacity(n_token * n_vocab);
            for t in 0..n_token {
                chunk_rows.extend_from_slice(logits_row(&logits, seq * rows_per_seq + t, n_vocab)?);
            }

            let acc = process_logits(
                n_vocab,
                &chunk_rows,
                targets,
                &mut logit_history[tok_start..tok_start + n_token],
                &mut prob_history[tok_start..tok_start + n_token],
            )?;
            nll += acc.nll;
            nll2 += acc.nll2;
            count += n_token;

            if params.ppl_output_type == 0 {
                eprint!("[{}]{:.4},", i + seq + 1, (nll / count as f64).exp());
            } else {
                let av = nll / count as f64;
                let mut av2 = nll2 / count as f64 - av * av;
                if av2 > 0.0 {
                    av2 = (av2 / (count - 1) as f64).sqrt();
                }
                eprintln!("{:8}  {:.4}  {:.4}  {:.4}", i * n_ctx_usize, (nll / count as f64).exp(), av, av2);
            }
        }
    }
    eprintln!();

    if count == 0 {
        return Err(Error::new("no tokens were evaluated"));
    }
    let av = nll / count as f64;
    let ppl = av.exp();
    let variance = nll2 / count as f64 - av * av;
    if variance > 0.0 {
        let sigma = (variance / (count - 1) as f64).sqrt();
        eprintln!("Final estimate: PPL = {:.4} +/- {:.5}", ppl, sigma * ppl);
    } else {
        eprintln!("Unexpected negative standard deviation of log(prob)");
    }

    Ok(PerplexityResults {
        tokens,
        ppl_value: ppl,
        logits: logit_history,
        probs: prob_history,
    })
}

// ---------------------------------------------------------------------------
// KL divergence metrics
// ---------------------------------------------------------------------------

/// `kl_divergence_result` from perplexity.cpp.
#[derive(Clone, Debug, Default)]
pub struct KlDivergenceResult {
    pub sum_nll: f64,
    pub sum_nll2: f64,
    pub sum_nll_base: f64,
    pub sum_nll_base2: f64,
    pub sum_nll_nll_base: f64,
    pub sum_kld: f64,
    pub sum_kld2: f64,
    pub sum_p_diff: f64,
    pub sum_p_diff2: f64,
    pub sum_p_diff4: f64,
    pub max_p_diff: f32,
    pub n_same_top: usize,
    pub count: usize,
}

/// `log_softmax(..., kl_divergence_result & kld)` overload.
fn kl_log_softmax(
    n_vocab: usize,
    logits: &[f32],
    base_log_prob: &[u16],
    tok: usize,
    kld: &mut KlDivergenceResult,
) -> Result<(f64, f32)> {
    if n_vocab == 0 || logits.len() < n_vocab || tok >= n_vocab {
        return Err(Error::new("invalid logits or token for log_softmax"));
    }
    if base_log_prob.len() < log_prob_stride(n_vocab) {
        return Err(Error::new("base log probability block is too small"));
    }

    let row = &logits[..n_vocab];
    let mut max_logit = row[0];
    let mut imax = 0usize;
    for i in 1..n_vocab {
        if row[i] > max_logit {
            max_logit = row[i];
            imax = i;
        }
    }
    let mut sum_exp = 0.0f64;
    for &v in row {
        sum_exp += f64::from((v - max_logit).exp());
    }
    let log_sum_exp = sum_exp.ln() as f32;

    let scale = load_f32(&base_log_prob[0..2]);
    let min_log_prob = load_f32(&base_log_prob[2..4]);
    let base = &base_log_prob[4..4 + n_vocab];

    let nll = max_logit + log_sum_exp - row[tok];
    kld.sum_nll += f64::from(nll);
    kld.sum_nll2 += f64::from(nll * nll);

    let nll_base = -(scale * f32::from(base[tok]) + min_log_prob);
    kld.sum_nll_base += f64::from(nll_base);
    kld.sum_nll_base2 += f64::from(nll_base * nll_base);
    kld.sum_nll_nll_base += f64::from(nll * nll_base);

    let shifted_max = max_logit + log_sum_exp;
    let mut sum = 0.0f64;
    let mut imax_base: i32 = -1;
    let mut p_log_base_max = 0.0f32;
    for i in 0..n_vocab {
        let p_log_base = scale * f32::from(base[i]) + min_log_prob;
        if i == 0 || p_log_base > p_log_base_max {
            p_log_base_max = p_log_base;
            imax_base = i as i32;
        }
        if p_log_base > -16.0 {
            let p_base = p_log_base.exp();
            sum += f64::from(p_base * (p_log_base - row[i] + shifted_max));
        }
    }
    kld.sum_kld += sum;
    kld.sum_kld2 += sum * sum;
    kld.count += 1;
    if imax_base >= 0 && imax == imax_base as usize {
        kld.n_same_top += 1;
    }

    let p_base = (-nll_base).exp();
    let p = (-nll).exp();
    let p_diff = p - p_base;
    kld.sum_p_diff += f64::from(p_diff);
    let p_diff2 = f64::from(p_diff * p_diff);
    kld.sum_p_diff2 += p_diff2;
    kld.sum_p_diff4 += p_diff2 * p_diff2;
    kld.max_p_diff = kld.max_p_diff.max(p_diff.abs());

    Ok((sum, p_diff))
}

/// Port of the `kl_divergence_result` `process_logits` overload.
pub fn process_logits_kld(
    n_vocab: usize,
    logits: &[f32],
    tokens: &[Token],
    base_log_probs: &[u16],
    kld: &mut KlDivergenceResult,
    kld_values: &mut [f32],
    p_diff_values: &mut [f32],
) -> Result<()> {
    let n_token = tokens.len().saturating_sub(1);
    let nv = log_prob_stride(n_vocab);
    if base_log_probs.len() < n_token * nv {
        return Err(Error::new("base log probability buffer is too small"));
    }
    if kld_values.len() < n_token || p_diff_values.len() < n_token {
        return Err(Error::new("KLD value buffer is too small"));
    }

    for i in 0..n_token {
        let tok = target_token(tokens, i, n_vocab)?;
        let base = &base_log_probs[i * nv..(i + 1) * nv];
        let (k, p) = kl_log_softmax(n_vocab, logits_row(logits, i, n_vocab)?, base, tok, kld)?;
        kld_values[i] = k as f32;
        p_diff_values[i] = p;
    }
    Ok(())
}

/// `mean_and_uncertainty` from the KL divergence evaluator.
pub fn mean_and_uncertainty(sum: f64, sum2: f64, count: usize) -> (f64, f64) {
    if count < 1 {
        return (0.0, 0.0);
    }
    let f = sum / count as f64;
    let mut df = sum2 / count as f64 - f * f;
    df = if df > 0.0 && count > 10 {
        (df / (count - 1) as f64).sqrt()
    } else {
        0.0
    };
    (f, df)
}

/// `covariance` from the KL divergence evaluator.
pub fn covariance(suma: f64, sumb: f64, sumab: f64, count: usize) -> f64 {
    if count < 10 {
        return 0.0;
    }
    let c = count as f64;
    let mut var = sumab / c - (suma / c) * (sumb / c);
    var /= c - 1.0;
    var
}

/// `percentile` from the KL divergence evaluator. `values` must be sorted.
pub fn percentile(values: &[f32], fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    if fraction <= 0.0 {
        return values[0];
    }
    if fraction >= 1.0 {
        return *values.last().unwrap_or(&0.0);
    }
    let p = fraction * (values.len() - 1) as f32;
    let ip = p as usize;
    let rest = p - ip as f32;
    (1.0 - rest) * values[ip] + rest * values[(ip + 1).min(values.len() - 1)]
}

/// Final statistics printed by the KL divergence evaluator.
#[derive(Clone, Copy, Debug)]
pub struct KlStats {
    pub ppl: (f64, f64),
    pub ppl_base: (f64, f64),
    pub log_ppl_ratio: (f64, f64),
    pub kl_div: (f64, f64),
    pub p_diff_rms: (f64, f64),
    pub same_top_p: f64,
}

/// Compute the final KL divergence statistics from an accumulated result.
pub fn kl_stats(kld: &KlDivergenceResult) -> KlStats {
    let log_ppl = mean_and_uncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
    let ppl_val = log_ppl.0.exp();
    let ppl = (ppl_val, ppl_val * log_ppl.1);

    let log_ppl_base = mean_and_uncertainty(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
    let ppl_base_val = log_ppl_base.0.exp();
    let ppl_base = (ppl_base_val, ppl_base_val * log_ppl_base.1);

    let cov = covariance(kld.sum_nll, kld.sum_nll_base, kld.sum_nll_nll_base, kld.count);
    let ratio_var = log_ppl.1 * log_ppl.1 + log_ppl_base.1 * log_ppl_base.1 - 2.0 * cov;
    let log_ratio = log_ppl.0 - log_ppl_base.0;
    let log_ppl_ratio = (log_ratio, ratio_var.max(0.0).sqrt());

    let kl_div = mean_and_uncertainty(kld.sum_kld, kld.sum_kld2, kld.count);
    let mse = mean_and_uncertainty(kld.sum_p_diff2, kld.sum_p_diff4, kld.count);
    let rms = mse.0.max(0.0).sqrt();
    let rms_unc = if rms != 0.0 { 0.5 / rms * mse.1 } else { 0.0 };
    let p_diff_rms = (rms, rms_unc);

    let same_top_p = if kld.count > 0 {
        kld.n_same_top as f64 / kld.count as f64
    } else {
        0.0
    };

    KlStats {
        ppl,
        ppl_base,
        log_ppl_ratio,
        kl_div,
        p_diff_rms,
        same_top_p,
    }
}

// ---------------------------------------------------------------------------
// logits file format
// ---------------------------------------------------------------------------

/// Header and packed log probabilities written by the `-o` option.
#[derive(Clone, Debug)]
pub struct LogitsFileData {
    pub n_ctx: i32,
    pub n_vocab: i32,
    pub n_chunk: i32,
    pub tokens: Vec<Token>,
    pub log_probs: Vec<u16>,
}

/// Write the file header: `_logits_`, `n_ctx`, `n_vocab`, `n_chunk`, tokens.
pub fn write_logits_header(
    path: &str,
    n_ctx: i32,
    n_vocab: i32,
    n_chunk: i32,
    tokens: &[Token],
) -> Result<()> {
    let mut file = File::create(path).map_err(|e| Error::new(e.to_string()))?;
    file.write_all(b"_logits_")
        .map_err(|e| Error::new(e.to_string()))?;
    for v in [n_ctx, n_vocab, n_chunk] {
        file.write_all(&v.to_le_bytes())
            .map_err(|e| Error::new(e.to_string()))?;
    }
    for &tok in tokens {
        file.write_all(&tok.to_le_bytes())
            .map_err(|e| Error::new(e.to_string()))?;
    }
    Ok(())
}

/// Append one chunk of packed 16-bit log probabilities to a logits file.
pub fn append_log_probs(path: &str, log_probs: &[u16]) -> Result<()> {
    let mut file = File::options()
        .append(true)
        .open(path)
        .map_err(|e| Error::new(e.to_string()))?;
    for &v in log_probs {
        file.write_all(&v.to_le_bytes())
            .map_err(|e| Error::new(e.to_string()))?;
    }
    Ok(())
}

fn read_i32(file: &mut File) -> Result<i32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)
        .map_err(|e| Error::new(e.to_string()))?;
    Ok(i32::from_le_bytes(buf))
}

/// Read a logits file produced by [`write_logits_header`] and
/// [`append_log_probs`].
pub fn read_logits_file(path: &str) -> Result<LogitsFileData> {
    let mut file = File::open(path).map_err(|e| Error::new(e.to_string()))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)
        .map_err(|e| Error::new(e.to_string()))?;
    if &magic != b"_logits_" {
        return Err(Error::new("file does not contain log probabilities"));
    }

    let n_ctx = read_i32(&mut file)?;
    let n_vocab = read_i32(&mut file)?;
    let n_chunk = read_i32(&mut file)?;

    let n_tokens = positive(n_ctx, "n_ctx")?
        .checked_mul(nonnegative(n_chunk, "n_chunk")?)
        .ok_or_else(|| Error::new("token count overflow"))?;
    let mut tokens = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        tokens.push(read_i32(&mut file)?);
    }

    let mut raw = Vec::new();
    file.read_to_end(&mut raw)
        .map_err(|e| Error::new(e.to_string()))?;
    if raw.len() % 2 != 0 {
        return Err(Error::new("truncated log probability data"));
    }
    let log_probs = raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();

    Ok(LogitsFileData {
        n_ctx,
        n_vocab,
        n_chunk,
        tokens,
        log_probs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn log_softmax_matches_definition() {
        let logits = [1.0f32, 2.0, 3.0];
        let res = log_softmax(3, &logits, 2).unwrap();
        let expected = -((1.0f64 - 3.0).exp() + (2.0f64 - 3.0).exp() + 1.0).ln();
        assert!((res.log_softmax - expected).abs() < 1.0e-6);
        assert!(res.prob > 0.0 && res.prob <= 1.0);
    }

    #[test]
    fn quantized_round_trip() {
        let mut block = vec![0u16; log_prob_stride(4)];
        let logits = [0.0f32, 1.0, 2.0, -1.0];
        let nll = log_softmax_quantized(4, &logits, &mut block, 2).unwrap();
        assert!(nll > 0.0);
        assert_eq!(load_f32(&block[0..2]).to_bits(), ((2.0 - -1.0) / 65535.0f32).to_bits());
    }

    #[test]
    fn percentile_interpolates() {
        let values = [0.0f32, 1.0, 2.0, 3.0];
        assert_eq!(percentile(&values, 0.0), 0.0);
        assert_eq!(percentile(&values, 1.0), 3.0);
        assert!((percentile(&values, 0.5) - 1.5).abs() < 1.0e-6);
    }

    #[test]
    fn prepare_params_adjusts_context() {
        let prepared = prepare_params(PerplexityParams {
            n_ctx: 512,
            n_batch: 2048,
            ppl_stride: 16,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(prepared.n_ctx, 512);
        assert_eq!(prepared.params.n_parallel, 4);
        assert_eq!(prepared.params.n_ctx, 2056);
        assert_eq!(prepared.params.n_batch, 2048);
    }
}

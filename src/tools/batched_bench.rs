//! Rust port of `tools/batched-bench/batched-bench.cpp` and `main.cpp`.
//!
//! The C++ tool loads a model, decodes random token batches and prints a table
//! (or JSONL) of timings. This port keeps the parameters, prompt setup and
//! timing math. Model, context and backend calls are represented by `Harness`,
//! so no FFI is needed. The omissions are listed at the end of this file.

use std::time::Instant;

use crate::common::FlashAttnType;
use crate::core::{Error, Mt19937, Result, SeqId, Token};

const WARMUP_N: usize = 16;

// ---------------------------------------------------------------------------
// params
// ---------------------------------------------------------------------------

/// The fields of `common_params` read by `llama_batched_bench`.
#[derive(Clone, Debug)]
pub struct BatchedBenchParams {
    pub model_path: String,
    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_gpu_layers: i32,
    pub flash_attn_type: FlashAttnType,
    pub kv_unified: bool,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub is_pp_shared: bool,
    pub is_tg_separate: bool,
    pub n_pp: Vec<i32>,
    pub n_tg: Vec<i32>,
    pub n_pl: Vec<i32>,
    pub output_jsonl: bool,
}

impl From<&crate::common::Params> for BatchedBenchParams {
    fn from(p: &crate::common::Params) -> Self {
        let n_threads = if p.cpuparams.n_threads < 0 {
            crate::common::common_cpu_get_num_math()
        } else {
            p.cpuparams.n_threads
        };
        let n_threads_batch = if p.cpuparams_batch.n_threads == -1 {
            n_threads
        } else if p.cpuparams_batch.n_threads <= 0 {
            hardware_threads()
        } else {
            p.cpuparams_batch.n_threads
        };
        Self {
            model_path: p.model.path.clone(),
            n_ctx: p.n_ctx,
            n_batch: p.n_batch,
            n_ubatch: p.n_ubatch,
            n_gpu_layers: p.n_gpu_layers,
            flash_attn_type: p.flash_attn_type,
            kv_unified: p.kv_unified,
            n_threads,
            n_threads_batch,
            is_pp_shared: p.is_pp_shared,
            is_tg_separate: p.is_tg_separate,
            n_pp: p.n_pp.clone(),
            n_tg: p.n_tg.clone(),
            n_pl: p.n_pl.clone(),
            output_jsonl: p.batched_bench_output_jsonl,
        }
    }
}

impl Default for BatchedBenchParams {
    fn default() -> Self {
        Self::from(&crate::common::Params::default())
    }
}

impl BatchedBenchParams {
    /// `ctx_params.n_seq_max` from the tool.
    pub fn n_seq_max(&self) -> u32 {
        self.n_pl.iter().copied().max().unwrap_or(1).max(1) as u32
    }

    /// Port of `common_params_parse` restricted to the flags this tool reads.
    pub fn parse_args<S: AsRef<str>>(args: &[S]) -> Result<Self> {
        let mut p = Self::default();
        apply_env(&mut p)?;

        let mut i = 0usize;
        while i < args.len() {
            let raw = args[i].as_ref();
            let arg = if raw.starts_with("--") {
                raw.replace('_', "-")
            } else {
                raw.to_owned()
            };
            match arg.as_str() {
                "-h" | "--help" | "--usage" => {
                    return Err(Error::new(usage("batched-bench")));
                }
                "-m" | "--model" => p.model_path = take_value(args, &mut i)?,
                "-c" | "--ctx-size" => p.n_ctx = parse_i32(&take_value(args, &mut i)?)?,
                "-b" | "--batch-size" => p.n_batch = parse_i32(&take_value(args, &mut i)?)?,
                "-ub" | "--ubatch-size" => p.n_ubatch = parse_i32(&take_value(args, &mut i)?)?,
                "-npp" => p.n_pp.extend(parse_csv(&take_value(args, &mut i)?)?),
                "-ntg" => p.n_tg.extend(parse_csv(&take_value(args, &mut i)?)?),
                "-npl" => p.n_pl.extend(parse_csv(&take_value(args, &mut i)?)?),
                "-pps" => p.is_pp_shared = true,
                "-tgs" => p.is_tg_separate = true,
                "-kvu" | "--kv-unified" => p.kv_unified = true,
                "-no-kvu" | "--no-kv-unified" => p.kv_unified = false,
                "-fa" | "--flash-attn" => {
                    p.flash_attn_type = parse_flash_attn(&take_value(args, &mut i)?)?;
                }
                "-ngl" | "--gpu-layers" | "--n-gpu-layers" => {
                    p.n_gpu_layers = parse_gpu_layers(&take_value(args, &mut i)?)?;
                }
                "-t" | "--threads" => {
                    let v = parse_i32(&take_value(args, &mut i)?)?;
                    let resolved = if v <= 0 { hardware_threads() } else { v };
                    if p.n_threads_batch == p.n_threads {
                        p.n_threads_batch = resolved;
                    }
                    p.n_threads = resolved;
                }
                "-tb" | "--threads-batch" => {
                    let v = parse_i32(&take_value(args, &mut i)?)?;
                    p.n_threads_batch = if v <= 0 { hardware_threads() } else { v };
                }
                "--output-format" => {
                    p.output_jsonl = match take_value(args, &mut i)?.as_str() {
                        "jsonl" => true,
                        "md" => false,
                        other => {
                            return Err(Error::new(format!(
                                "error: invalid value for --output-format: '{other}'"
                            )));
                        }
                    };
                }
                _ => return Err(Error::new(format!("error: invalid argument: {raw}"))),
            }
            i += 1;
        }
        Ok(p)
    }
}

/// Text of `print_usage` in the C++ tool.
pub fn usage(program: &str) -> String {
    format!(
        "\nexample usage:\n\n    {program} -m model.gguf -c 2048 -b 2048 -ub 512 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32 [-pps]\n\n"
    )
}

fn hardware_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(0)
}

fn take_value<S: AsRef<str>>(args: &[S], i: &mut usize) -> Result<String> {
    *i += 1;
    args.get(*i)
        .map(|s| s.as_ref().to_owned())
        .ok_or_else(|| Error::new("expected value for argument"))
}

fn parse_i32(value: &str) -> Result<i32> {
    let s = value.trim_start();
    let bytes = s.as_bytes();
    let mut end = 0usize;
    if bytes.first() == Some(&b'+') || bytes.first() == Some(&b'-') {
        end = 1;
    }
    let digits = end;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == digits {
        return Err(Error::new(format!("error: invalid integer: '{value}'")));
    }
    s[..end]
        .parse::<i32>()
        .map_err(|_| Error::new(format!("error: invalid integer: '{value}'")))
}

fn parse_csv(value: &str) -> Result<Vec<i32>> {
    value.split(',').map(parse_i32).collect()
}

fn parse_bool(value: &str) -> Result<bool> {
    match value {
        "on" | "enabled" | "true" | "1" => Ok(true),
        "off" | "disabled" | "false" | "0" => Ok(false),
        _ => Err(Error::new("invalid boolean value")),
    }
}

fn parse_flash_attn(value: &str) -> Result<FlashAttnType> {
    match value {
        "on" | "enabled" | "true" | "1" => Ok(FlashAttnType::Enabled),
        "off" | "disabled" | "false" | "0" => Ok(FlashAttnType::Disabled),
        "auto" | "-1" => Ok(FlashAttnType::Auto),
        _ => Err(Error::new(format!(
            "error: unknown value for --flash-attn: '{value}'"
        ))),
    }
}

fn parse_gpu_layers(value: &str) -> Result<i32> {
    match value {
        "auto" => Ok(-1),
        "all" => Ok(-2),
        _ => parse_i32(value),
    }
}

fn apply_env(p: &mut BatchedBenchParams) -> Result<()> {
    if let Ok(v) = std::env::var("LLAMA_ARG_CTX_SIZE") {
        p.n_ctx = parse_i32(&v)?;
    }
    if let Ok(v) = std::env::var("LLAMA_ARG_BATCH") {
        p.n_batch = parse_i32(&v)?;
    }
    if let Ok(v) = std::env::var("LLAMA_ARG_UBATCH") {
        p.n_ubatch = parse_i32(&v)?;
    }
    if let Ok(v) = std::env::var("LLAMA_ARG_N_GPU_LAYERS") {
        p.n_gpu_layers = parse_gpu_layers(&v)?;
    }
    if let Ok(v) = std::env::var("LLAMA_ARG_FLASH_ATTN") {
        p.flash_attn_type = parse_flash_attn(&v)?;
    }
    if let Ok(v) = std::env::var("LLAMA_ARG_KV_UNIFIED") {
        p.kv_unified = parse_bool(&v)?;
    } else if std::env::var_os("LLAMA_ARG_NO_KV_UNIFIED").is_some() {
        p.kv_unified = false;
    }
    if let Ok(v) = std::env::var("LLAMA_ARG_THREADS") {
        let n = parse_i32(&v)?;
        let resolved = if n <= 0 { hardware_threads() } else { n };
        if p.n_threads_batch == p.n_threads {
            p.n_threads_batch = resolved;
        }
        p.n_threads = resolved;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// prompt setup
// ---------------------------------------------------------------------------

/// One entry of the C++ `common_batch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchToken {
    pub id: Token,
    pub pos: i32,
    pub seq_id: SeqId,
    pub output: bool,
}

/// Stand-in for `common_batch`. Entries are not limited to `n_batch`;
/// `sub_batch` renders a chunk for `decode_helper`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BenchBatch {
    pub tokens: Vec<BenchToken>,
}

impl BenchBatch {
    pub fn add(&mut self, id: Token, pos: i32, seq_id: SeqId, output: bool) {
        self.tokens.push(BenchToken {
            id,
            pos,
            seq_id,
            output,
        });
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
    }

    pub fn size(&self) -> usize {
        self.tokens.len()
    }

    pub fn sub_batch(&self, off: usize, n: usize) -> Vec<BenchToken> {
        self.tokens.get(off..off.saturating_add(n)).unwrap_or(&[]).to_vec()
    }
}

/// Stand-ins for the context calls the tool makes around decoding.
pub trait Harness {
    /// `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)`.
    fn decode(&mut self, batch: &[BenchToken]) -> Result<()>;

    /// `llama_synchronize(ctx)`.
    fn synchronize(&mut self) -> Result<()>;

    /// `llama_memory_clear(mem, data_only)`.
    fn memory_clear(&mut self, _data_only: bool) {}

    /// `llama_memory_seq_cp(mem, src, dst, p0, p1)`.
    fn memory_seq_cp(&mut self, _src: SeqId, _dst: SeqId, _p0: i32, _p1: i32) {}

    /// `llama_memory_seq_rm(mem, seq, p0, p1)`.
    fn memory_seq_rm(&mut self, _seq: SeqId, _p0: i32, _p1: i32) {}
}

/// No-op harness for exercising setup and timing without a model.
#[derive(Default)]
pub struct NoopHarness;

impl Harness for NoopHarness {
    fn decode(&mut self, _batch: &[BenchToken]) -> Result<()> {
        Ok(())
    }

    fn synchronize(&mut self) -> Result<()> {
        Ok(())
    }
}

fn rand_token(rng: &mut Mt19937, n_vocab: Token) -> Token {
    if n_vocab <= 0 {
        return 0;
    }
    (rng.next_u32() as i64 % n_vocab as i64) as Token
}

/// `decode_helper` in the C++ tool: decode in chunks of `n_batch`.
fn decode_helper(
    batch: &BenchBatch,
    n_batch: usize,
    synchronize: bool,
    harness: &mut dyn Harness,
) -> Result<()> {
    let mut i = 0usize;
    while i < batch.size() {
        let n_tokens = n_batch.min(batch.size() - i);
        harness.decode(&batch.sub_batch(i, n_tokens)).map_err(|e| {
            Error::new(format!(
                "failed to decode the batch, n_batch = {n_batch}: {e}"
            ))
        })?;
        if synchronize {
            harness.synchronize()?;
        }
        i += n_tokens;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// timing and output
// ---------------------------------------------------------------------------

/// One row of the C++ result table.
#[derive(Clone, Debug)]
pub struct BenchResult {
    pub pp: i32,
    pub tg: i32,
    pub pl: i32,
    pub n_kv: i32,
    pub t_pp: f32,
    pub speed_pp: f32,
    pub t_tg: f32,
    pub speed_tg: f32,
    pub t: f32,
    pub speed: f32,
}

impl BenchResult {
    pub fn format_md(&self) -> String {
        format!(
            "|{:6} | {:6} | {:4} | {:6} | {:8.3} | {:8.2} | {:8.3} | {:8.2} | {:8.3} | {:8.2} |\n",
            self.pp,
            self.tg,
            self.pl,
            self.n_kv,
            self.t_pp,
            self.speed_pp,
            self.t_tg,
            self.speed_tg,
            self.t,
            self.speed
        )
    }

    pub fn format_jsonl(&self, p: &BatchedBenchParams, n_kv_max: i32) -> String {
        format!(
            "{{\"n_kv_max\": {n_kv_max}, \"n_batch\": {}, \"n_ubatch\": {}, \"flash_attn\": {}, \"is_pp_shared\": {}, \"n_gpu_layers\": {}, \"n_threads\": {}, \"n_threads_batch\": {}, \"pp\": {}, \"tg\": {}, \"pl\": {}, \"n_kv\": {}, \"t_pp\": {:.6}, \"speed_pp\": {:.6}, \"t_tg\": {:.6}, \"speed_tg\": {:.6}, \"t\": {:.6}, \"speed\": {:.6}}}\n",
            p.n_batch,
            p.n_ubatch,
            p.flash_attn_type as i32,
            p.is_pp_shared as i32,
            p.n_gpu_layers,
            p.n_threads as u32,
            p.n_threads_batch as u32,
            self.pp,
            self.tg,
            self.pl,
            self.n_kv,
            self.t_pp,
            self.speed_pp,
            self.t_tg,
            self.speed_tg,
            self.t,
            self.speed
        )
    }
}

/// The info line and table header printed before the results in markdown mode.
pub fn header(p: &BatchedBenchParams, n_kv_max: i32) -> String {
    format!(
        "\nllama_batched_bench: n_kv_max = {n_kv_max}, n_batch = {}, n_ubatch = {}, flash_attn = {}, is_pp_shared = {}, is_tg_separate = {}, n_gpu_layers = {}, n_threads = {}, n_threads_batch = {}\n\n|{:>6} | {:>6} | {:>4} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |\n|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|\n",
        p.n_batch,
        p.n_ubatch,
        p.flash_attn_type as i32,
        p.is_pp_shared as i32,
        p.is_tg_separate as i32,
        p.n_gpu_layers,
        p.n_threads as u32,
        p.n_threads_batch as u32,
        "PP",
        "TG",
        "B",
        "N_KV",
        "T_PP s",
        "S_PP t/s",
        "T_TG s",
        "S_TG t/s",
        "T s",
        "S t/s",
        "-".repeat(7),
        "-".repeat(7),
        "-".repeat(5),
        "-".repeat(7),
        "-".repeat(9),
        "-".repeat(9),
        "-".repeat(9),
        "-".repeat(9),
        "-".repeat(9),
        "-".repeat(9)
    )
}

/// Full output: header (markdown only), all rows, and the trailing newline.
pub fn format_report(p: &BatchedBenchParams, n_kv_max: i32, results: &[BenchResult]) -> String {
    let mut out = String::new();
    if !p.output_jsonl {
        out.push_str(&header(p, n_kv_max));
    }
    for r in results {
        if p.output_jsonl {
            out.push_str(&r.format_jsonl(p, n_kv_max));
        } else {
            out.push_str(&r.format_md());
        }
    }
    out.push('\n');
    out
}

/// The benchmark loop, with `n_vocab` supplied by the caller in place of the
/// loaded model vocabulary. `n_kv_max` is `params.n_ctx`.
pub fn run(
    p: &BatchedBenchParams,
    n_vocab: Token,
    harness: &mut dyn Harness,
) -> Result<Vec<BenchResult>> {
    let n_batch = p.n_batch.max(0) as usize;
    if n_batch == 0 {
        return Err(Error::new("invalid n_batch"));
    }
    let n_kv_max = p.n_ctx;

    let mut rng = Mt19937::new(1);
    let mut batch = BenchBatch::default();

    // warm up
    for i in 0..WARMUP_N {
        batch.add(rand_token(&mut rng, n_vocab), i as i32, 0, false);
    }
    decode_helper(&batch, n_batch, true, harness)?;

    let mut results = Vec::new();
    for &pp in &p.n_pp {
        for &tg in &p.n_tg {
            for &pl in &p.n_pl {
                let n_ctx_req = if p.is_pp_shared {
                    (if p.kv_unified {
                        pp as i64
                    } else {
                        pl as i64 * pp as i64
                    }) + pl as i64 * tg as i64
                } else {
                    pl as i64 * (pp as i64 + tg as i64)
                };
                if n_ctx_req > n_kv_max as i64 {
                    continue;
                }

                batch.clear();
                let n_seqs = if p.is_pp_shared { 1 } else { pl };
                for j in 0..n_seqs {
                    for i in 0..pp {
                        batch.add(rand_token(&mut rng, n_vocab), i, j as SeqId, i == pp - 1);
                    }
                }
                harness.memory_clear(false);

                let t_pp_start = Instant::now();
                decode_helper(&batch, n_batch, false, harness)?;
                harness.synchronize()?;
                let t_pp = t_pp_start.elapsed().as_micros() as f32 / 1_000_000.0f32;

                if p.is_pp_shared {
                    for i in 1..pl {
                        harness.memory_seq_cp(0, i as SeqId, -1, -1);
                    }
                    if !p.kv_unified {
                        batch.clear();
                        batch.add(rand_token(&mut rng, n_vocab), pp, 0, true);
                        decode_helper(&batch, n_batch, true, harness)?;
                        harness.memory_seq_rm(0, pp, -1);
                    }
                }

                let t_tg_start = Instant::now();
                if p.is_tg_separate {
                    for j in 0..pl {
                        for i in 0..tg {
                            batch.clear();
                            batch.add(rand_token(&mut rng, n_vocab), pp + i, j as SeqId, true);
                            decode_helper(&batch, n_batch, true, harness)?;
                        }
                    }
                } else {
                    for i in 0..tg {
                        batch.clear();
                        for j in 0..pl {
                            batch.add(rand_token(&mut rng, n_vocab), pp + i, j as SeqId, true);
                        }
                        decode_helper(&batch, n_batch, true, harness)?;
                    }
                }
                let t_tg = t_tg_start.elapsed().as_micros() as f32 / 1_000_000.0f32;

                let t = t_pp + t_tg;
                let pp_tokens = if p.is_pp_shared {
                    pp as f32
                } else {
                    pl as f32 * pp as f32
                };
                let tg_tokens = pl as f32 * tg as f32;
                results.push(BenchResult {
                    pp,
                    tg,
                    pl,
                    n_kv: n_ctx_req as i32,
                    t_pp,
                    speed_pp: pp_tokens / t_pp,
                    t_tg,
                    speed_tg: tg_tokens / t_tg,
                    t,
                    speed: (pp_tokens + tg_tokens) / t,
                });
            }
        }
    }
    Ok(results)
}

/// Entry point equivalent: parse arguments, benchmark, return the report.
pub fn run_tool<S: AsRef<str>>(
    args: &[S],
    n_vocab: Token,
    harness: &mut dyn Harness,
) -> Result<String> {
    let p = BatchedBenchParams::parse_args(args)?;
    let results = run(&p, n_vocab, harness)?;
    Ok(format_report(&p, p.n_ctx, &results))
}

// Omissions (each needs a live model, context or backend):
// - model load, backend init, vocab lookup and `llama_perf_context_print`
// - real decode: `Harness` stands in for llama_process/llama_synchronize
// - KV cache clear/copy/remove: default no-op hooks in `Harness`
// - `llama_params_fit`, so `n_ctx == 0` is not resolved from the model
// - parts of `common_params_parse` unrelated to the flags handled above

#[cfg(test)]
mod tests {
    use super::*;

    struct Recording {
        calls: Vec<Vec<BenchToken>>,
        syncs: usize,
        clears: usize,
        copies: usize,
        rms: usize,
    }

    impl Harness for Recording {
        fn decode(&mut self, batch: &[BenchToken]) -> Result<()> {
            self.calls.push(batch.to_vec());
            Ok(())
        }

        fn synchronize(&mut self) -> Result<()> {
            self.syncs += 1;
            Ok(())
        }

        fn memory_clear(&mut self, _data_only: bool) {
            self.clears += 1;
        }

        fn memory_seq_cp(&mut self, _src: SeqId, _dst: SeqId, _p0: i32, _p1: i32) {
            self.copies += 1;
        }

        fn memory_seq_rm(&mut self, _seq: SeqId, _p0: i32, _p1: i32) {
            self.rms += 1;
        }
    }

    fn recorder() -> Recording {
        Recording {
            calls: Vec::new(),
            syncs: 0,
            clears: 0,
            copies: 0,
            rms: 0,
        }
    }

    #[test]
    fn parses_bench_args() {
        let p = BatchedBenchParams::parse_args(&[
            "-m",
            "model.gguf",
            "-c",
            "2048",
            "-b",
            "512",
            "-ub",
            "128",
            "-npp",
            "128,256",
            "-ntg",
            "64",
            "-npl",
            "1,2",
            "-pps",
            "-tgs",
            "--output-format",
            "jsonl",
        ])
        .unwrap();
        assert_eq!(p.model_path, "model.gguf");
        assert_eq!(p.n_ctx, 2048);
        assert_eq!(p.n_batch, 512);
        assert_eq!(p.n_ubatch, 128);
        assert_eq!(p.n_pp, vec![128, 256]);
        assert_eq!(p.n_tg, vec![64]);
        assert_eq!(p.n_pl, vec![1, 2]);
        assert!(p.is_pp_shared);
        assert!(p.is_tg_separate);
        assert!(p.output_jsonl);
        assert_eq!(p.n_seq_max(), 2);
    }

    #[test]
    fn rejects_bad_args() {
        assert!(BatchedBenchParams::parse_args(&["--bogus"]).is_err());
        assert!(BatchedBenchParams::parse_args(&["-npp", "1,nope"]).is_err());
        assert!(BatchedBenchParams::parse_args(&["-npl"]).is_err());
        assert!(BatchedBenchParams::parse_args(&["-fa", "maybe"]).is_err());
    }

    #[test]
    fn builds_interleaved_batches() {
        let mut p = BatchedBenchParams::default();
        p.n_ctx = 1024;
        p.n_batch = 512;
        p.n_pp = vec![2];
        p.n_tg = vec![2];
        p.n_pl = vec![2];

        let mut h = recorder();
        let results = run(&p, 100, &mut h).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].n_kv, 8);
        assert_eq!(h.calls.len(), 4);
        assert_eq!(h.calls[0].len(), WARMUP_N);
        assert!(!h.calls[0][0].output);
        assert_eq!(h.calls[1].len(), 4);
        assert_eq!(h.calls[1][0].pos, 0);
        assert_eq!(h.calls[1][1].pos, 1);
        assert!(h.calls[1][1].output);
        assert_eq!(h.calls[1][2].seq_id, 1);
        assert_eq!(h.calls[2].len(), 2);
        assert_eq!(h.calls[2][0].pos, 2);
        assert_eq!(h.calls[3][1].seq_id, 1);
        assert_eq!(h.clears, 1);
        assert_eq!(h.copies, 0);
        assert_eq!(h.rms, 0);
        assert!(h.syncs >= 3);
    }

    #[test]
    fn builds_shared_pp_batches() {
        let mut p = BatchedBenchParams::default();
        p.n_ctx = 1024;
        p.n_batch = 512;
        p.is_pp_shared = true;
        p.n_pp = vec![2];
        p.n_tg = vec![1];
        p.n_pl = vec![3];

        let mut h = recorder();
        let results = run(&p, 100, &mut h).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].n_kv, 9);
        assert_eq!(h.calls[1].len(), 2);
        assert_eq!(h.calls[1][1].seq_id, 0);
        assert_eq!(h.copies, 2);
        assert_eq!(h.calls.len(), 4);
        assert_eq!(h.calls[3].len(), 3);
        assert_eq!(h.calls[3][0].seq_id, 0);
        assert_eq!(h.rms, 1);
    }

    #[test]
    fn formats_results() {
        let p = BatchedBenchParams::default();
        let r = BenchResult {
            pp: 128,
            tg: 128,
            pl: 1,
            n_kv: 256,
            t_pp: 0.1,
            speed_pp: 1280.0,
            t_tg: 1.0,
            speed_tg: 128.0,
            t: 1.1,
            speed: 232.7,
        };
        assert!(r.format_md().starts_with("|   128 |    128 |    1 |    256 |"));
        let js = r.format_jsonl(&p, 2048);
        assert!(js.contains("\"pp\": 128"));
        assert!(js.contains("\"n_kv\": 256"));
        assert!(js.contains("\"t_pp\": 0.100000"));
        let report = format_report(&p, 2048, std::slice::from_ref(&r));
        assert!(report.contains("n_kv_max = 2048"));
        assert!(report.contains("S_PP t/s"));
        assert!(report.ends_with("\n\n"));
    }

    #[test]
    fn chunks_large_batches() {
        let p = BatchedBenchParams {
            n_ctx: 64,
            n_batch: 8,
            n_pp: vec![16],
            n_tg: vec![1],
            n_pl: vec![1],
            ..BatchedBenchParams::default()
        };
        let mut h = recorder();
        run(&p, 100, &mut h).unwrap();
        assert_eq!(h.calls.len(), 5);
        for call in &h.calls[..4] {
            assert_eq!(call.len(), 8);
        }
        assert_eq!(h.calls[4].len(), 1);
    }
}

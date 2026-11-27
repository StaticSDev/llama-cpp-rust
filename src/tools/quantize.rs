//! Rust port of `tools/quantize/quantize.cpp` and `tools/quantize/main.cpp`.
//!
//! CLI parsing, importance-matrix ingestion and dispatch to
//! `llama_model_quantize`. Signal handlers and the ggml backend lifetime are
//! omitted: they need platform calls, not CLI logic.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::common::{
    string_parse_kv_override, string_split, KvOverride as CommonKvOverride,
    KvOverrideType as CommonKvOverrideType,
};
use crate::core::{Error, Result};
use crate::gguf::{
    gguf_find_key, gguf_get_arr_n, gguf_get_arr_str, gguf_get_arr_type, gguf_get_kv_type,
    gguf_get_val_u32, gguf_init_from_file, GgufContext, GgufInitParams, GgufType, GgmlContext,
    GgmlTensor, GgmlType,
};
use crate::quant::{
    llama_model_quantize, llama_model_quantize_default_params, FType, ImatrixData, KvOverride,
    KvOverrideType, QType, TensorTypeOverride,
};

// ---------------------------------------------------------------------------
// file-type table
// ---------------------------------------------------------------------------

struct QuantOption {
    name: &'static str,
    ftype: FType,
    desc: &'static str,
}

const QUANT_OPTIONS: &[QuantOption] = &[
    QuantOption { name: "Q1_0", ftype: FType::MostlyQ1_0, desc: " 1.125 bpw quantization" },
    QuantOption { name: "Q2_0", ftype: FType::MostlyQ2_0, desc: " 2.25 bpw quantization (group 64)" },
    QuantOption { name: "Q4_0", ftype: FType::MostlyQ4_0, desc: " 4.34G, +0.4685 ppl @ Llama-3-8B" },
    QuantOption { name: "Q4_1", ftype: FType::MostlyQ4_1, desc: " 4.78G, +0.4511 ppl @ Llama-3-8B" },
    QuantOption { name: "MXFP4_MOE", ftype: FType::MostlyMxfp4Moe, desc: " MXFP4 MoE" },
    QuantOption { name: "Q5_0", ftype: FType::MostlyQ5_0, desc: " 5.21G, +0.1316 ppl @ Llama-3-8B" },
    QuantOption { name: "Q5_1", ftype: FType::MostlyQ5_1, desc: " 5.65G, +0.1062 ppl @ Llama-3-8B" },
    QuantOption { name: "IQ2_XXS", ftype: FType::MostlyIq2Xxs, desc: " 2.06 bpw quantization" },
    QuantOption { name: "IQ2_XS", ftype: FType::MostlyIq2Xs, desc: " 2.31 bpw quantization" },
    QuantOption { name: "IQ2_S", ftype: FType::MostlyIq2S, desc: " 2.5  bpw quantization" },
    QuantOption { name: "IQ2_M", ftype: FType::MostlyIq2M, desc: " 2.7  bpw quantization" },
    QuantOption { name: "IQ1_S", ftype: FType::MostlyIq1S, desc: " 1.56 bpw quantization" },
    QuantOption { name: "IQ1_M", ftype: FType::MostlyIq1M, desc: " 1.75 bpw quantization" },
    QuantOption { name: "TQ1_0", ftype: FType::MostlyTq1_0, desc: " 1.69 bpw ternarization" },
    QuantOption { name: "TQ2_0", ftype: FType::MostlyTq2_0, desc: " 2.06 bpw ternarization" },
    QuantOption { name: "Q2_K", ftype: FType::MostlyQ2K, desc: " 2.96G, +3.5199 ppl @ Llama-3-8B" },
    QuantOption { name: "Q2_K_S", ftype: FType::MostlyQ2KS, desc: " 2.96G, +3.1836 ppl @ Llama-3-8B" },
    QuantOption { name: "IQ3_XXS", ftype: FType::MostlyIq3Xxs, desc: " 3.06 bpw quantization" },
    QuantOption { name: "IQ3_S", ftype: FType::MostlyIq3S, desc: " 3.44 bpw quantization" },
    QuantOption { name: "IQ3_M", ftype: FType::MostlyIq3M, desc: " 3.66 bpw quantization mix" },
    QuantOption { name: "Q3_K", ftype: FType::MostlyQ3KM, desc: "alias for Q3_K_M" },
    QuantOption { name: "IQ3_XS", ftype: FType::MostlyIq3Xs, desc: " 3.3 bpw quantization" },
    QuantOption { name: "Q3_K_S", ftype: FType::MostlyQ3KS, desc: " 3.41G, +1.6321 ppl @ Llama-3-8B" },
    QuantOption { name: "Q3_K_M", ftype: FType::MostlyQ3KM, desc: " 3.74G, +0.6569 ppl @ Llama-3-8B" },
    QuantOption { name: "Q3_K_L", ftype: FType::MostlyQ3KL, desc: " 4.03G, +0.5562 ppl @ Llama-3-8B" },
    QuantOption { name: "IQ4_NL", ftype: FType::MostlyIq4Nl, desc: " 4.50 bpw non-linear quantization" },
    QuantOption { name: "IQ4_XS", ftype: FType::MostlyIq4Xs, desc: " 4.25 bpw non-linear quantization" },
    QuantOption { name: "Q4_K", ftype: FType::MostlyQ4KM, desc: "alias for Q4_K_M" },
    QuantOption { name: "Q4_K_S", ftype: FType::MostlyQ4KS, desc: " 4.37G, +0.2689 ppl @ Llama-3-8B" },
    QuantOption { name: "Q4_K_M", ftype: FType::MostlyQ4KM, desc: " 4.58G, +0.1754 ppl @ Llama-3-8B" },
    QuantOption { name: "Q5_K", ftype: FType::MostlyQ5KM, desc: "alias for Q5_K_M" },
    QuantOption { name: "Q5_K_S", ftype: FType::MostlyQ5KS, desc: " 5.21G, +0.1049 ppl @ Llama-3-8B" },
    QuantOption { name: "Q5_K_M", ftype: FType::MostlyQ5KM, desc: " 5.33G, +0.0569 ppl @ Llama-3-8B" },
    QuantOption { name: "Q6_K", ftype: FType::MostlyQ6K, desc: " 6.14G, +0.0217 ppl @ Llama-3-8B" },
    QuantOption { name: "Q8_0", ftype: FType::MostlyQ8_0, desc: " 7.96G, +0.0026 ppl @ Llama-3-8B" },
    QuantOption { name: "F16", ftype: FType::MostlyF16, desc: "14.00G, +0.0020 ppl @ Mistral-7B" },
    QuantOption { name: "BF16", ftype: FType::MostlyBf16, desc: "14.00G, -0.0050 ppl @ Mistral-7B" },
    QuantOption { name: "F32", ftype: FType::AllF32, desc: "26.00G              @ 7B" },
    // Keep COPY after F32: ftype 0 must match F32 first.
    QuantOption { name: "COPY", ftype: FType::AllF32, desc: "only copy tensors, no quantizing" },
];

const LLM_KV_QUANTIZE_IMATRIX_FILE: &str = "quantize.imatrix.file";
const LLM_KV_QUANTIZE_IMATRIX_DATASET: &str = "quantize.imatrix.dataset";
const LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES: &str = "quantize.imatrix.entries_count";
const LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS: &str = "quantize.imatrix.chunks_count";

fn striequals(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

/// Parse a leading integer, like `std::stoi`, without requiring the whole string.
fn parse_i32_prefix(s: &str) -> Option<i32> {
    let t = s.trim_start();
    let bytes = t.as_bytes();
    let mut i = 0;
    if bytes.first() == Some(&b'-') || bytes.first() == Some(&b'+') {
        i = 1;
    }
    let digits_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == digits_start {
        return None;
    }
    t[..i].parse::<i32>().ok()
}

/// Parse a leading floating point value, like `std::strtod`.
fn parse_double_prefix(s: &str) -> Option<(f64, usize)> {
    let bytes = s.as_bytes();
    let mut start = 0;
    while start < bytes.len() && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    let mut best = None;
    let mut end = start + 1;
    while end <= bytes.len() {
        if s.is_char_boundary(end) {
            if let Ok(v) = s[start..end].parse::<f64>() {
                best = Some((v, end));
            }
        }
        end += 1;
    }
    best
}

fn try_parse_ftype(ftype_str_in: &str) -> Option<(FType, String)> {
    let upper = ftype_str_in.to_ascii_uppercase();
    for it in QUANT_OPTIONS {
        if striequals(it.name, &upper) {
            return Some((it.ftype, it.name.to_owned()));
        }
    }
    if let Some(value) = parse_i32_prefix(&upper) {
        for it in QUANT_OPTIONS {
            if it.ftype.as_i32() == value {
                return Some((it.ftype, it.name.to_owned()));
            }
        }
    }
    None
}

fn usage(executable: &str) -> Error {
    println!("usage: {} [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights]", executable);
    println!("       [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--tensor-type] [--tensor-type-file] [--prune-layers]");
    println!("       [--keep-split] [--override-kv] [--dry-run] [--max-buffer-size] [--target-bpw] [--target-size] [--target-exclude] [--state-file]");
    println!("       model-f32.gguf [model-quant.gguf] type [nthreads]\n");
    println!("  --allow-requantize");
    println!("                                      allow requantizing tensors that have already been quantized");
    println!("                                      WARNING: this can severely reduce quality compared to quantizing");
    println!("                                               from 16bit or 32bit!");
    println!("  --leave-output-tensor");
    println!("                                      leave output.weight un(re)quantized");
    println!("                                      increases model size but may also increase quality, especially when requantizing");
    println!("  --pure");
    println!("                                      disable k-quant mixtures and quantize all tensors to the same type");
    println!("  --imatrix file_name");
    println!("                                      use data in file_name as importance matrix for quant optimizations");
    println!("  --include-weights tensor_name");
    println!("                                      use importance matrix for this/these tensor(s)");
    println!("  --exclude-weights tensor_name");
    println!("                                      do not use importance matrix for this/these tensor(s)");
    println!("  --output-tensor-type ggml_type");
    println!("                                      use this ggml_type for the output.weight tensor");
    println!("  --token-embedding-type ggml_type");
    println!("                                      use this ggml_type for the token embeddings tensor");
    println!("  --tensor-type tensor_name=ggml_type");
    println!("                                      quantize this tensor to this ggml_type");
    println!("                                      this is an advanced option to selectively quantize tensors; may be specified multiple times.");
    println!("                                      example: --tensor-type attn_q=q8_0");
    println!("  --tensor-type-file tensor_types.txt");
    println!("                                      list of tensors to quantize to a specific ggml_type");
    println!("                                      this is an advanced option to selectively quantize a long list of tensors.");
    println!("                                      the file should use the same format as above, separated by spaces or newlines.");
    println!("  --prune-layers L0,L1,L2...");
    println!("                                      comma-separated list of layer numbers to prune from the model");
    println!("                                      WARNING: this is an advanced option, use with care.");
    println!("  --keep-split");
    println!("                                      generate quantized model in the same shards as input");
    println!("  --override-kv KEY=TYPE:VALUE");
    println!("                                      override model metadata by key in the quantized model; may be specified multiple times.");
    println!("                                      WARNING: this is an advanced option, use with care.");
    println!("  --dry-run");
    println!("                                      calculate and show the final quantization size without performing quantization");
    println!("                                      example: llama-quantize --dry-run model-f32.gguf Q4_K");
    println!("  --max-buffer-size MiB");
    println!("                                      max amount of tensor rows kept in memory while quantizing one tensor (default: 8192)");
    println!("                                      lower it to quantize models with very large tensors on a machine with little RAM\n");
    println!("  --target-bpw n");
    println!("                                      target a bits per weight (bpw); must be a number between 0.0 and 16.0, 0 means the smallest possible model");
    println!("                                      advanced option to automatically select quantization types to achieve a total bits per weight (bpw) target\n");
    println!("  --target-size n");
    println!("                                      target a file size; must be 0 or more, 0 means the smallest possible model");
    println!("                                      advanced option to automatically select quantization types to achieve a total file size");
    println!("                                      allowed units: b, k|kib, m|mib, g|gib, t|tib; defaults to b (bytes) if none is provided\n");
    println!("  --state-file [filename]");
    println!("                                      file name to use/save; if none is provided, the default name will be used\n");
    println!("  --target-exclude type[,type...]");
    println!("                                      comma-separated ggml_type (names or ordinals) that automatic quant type selection must ignore");
    println!("                                      advanced option for --target-bpw/--target-size; may be specified multiple times\n");
    println!("note: --include-weights and --exclude-weights cannot be used together");
    println!("      --target-bpw and --target-size cannot be used together");
    println!("      --exclude-weights excludes tensor NAMES, while --target-exclude excludes quant TYPES\n");
    println!("-----------------------------------------------------------------------------");
    println!(" allowed quantization types");
    println!("-----------------------------------------------------------------------------\n");
    for it in QUANT_OPTIONS {
        if it.name != "COPY" {
            print!("  {:2}  or  ", it.ftype.as_i32());
        } else {
            print!("          ");
        }
        println!("{:<7} : {}", it.name, it.desc);
    }
    Error::new(String::new())
}

// ---------------------------------------------------------------------------
// importance matrix file loading
// ---------------------------------------------------------------------------

type SideData = HashMap<String, Vec<f32>>;

#[derive(Default)]
struct ImatrixEntry {
    sums: Vec<f32>,
    activations: Vec<f32>,
    counts: Vec<i64>,
    stats: Vec<f32>,
}

#[derive(Default)]
struct LoadedImatrix {
    entries: BTreeMap<String, ImatrixEntry>,
    datasets: Vec<String>,
    stats_schema: Vec<String>,
    chunk_count: i32,
    is_legacy: bool,
    has_metadata: bool,
}

fn f32_vec_from_bytes(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn ggml_nelements(t: &GgmlTensor) -> usize {
    t.ne
        .iter()
        .try_fold(1i128, |acc, &ne| acc.checked_mul(ne as i128))
        .unwrap_or(0)
        .max(0) as usize
}

#[derive(Default)]
struct TensorParts<'a> {
    in_sum: Option<&'a GgmlTensor>,
    in_sum2: Option<&'a GgmlTensor>,
    counts: Option<&'a GgmlTensor>,
    stats: Option<&'a GgmlTensor>,
}

fn default_stats_schema() -> Vec<String> {
    [
        "sum_sq", "mean", "elements", "std_deviation", "skewness", "kurtosis", "gain",
        "h_norm", "l2_dist", "cossim", "pearson", "covariance",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect()
}

fn string_array(ctx: &GgufContext, key: &str) -> Option<Vec<String>> {
    let id = gguf_find_key(ctx, key);
    if id == -1
        || gguf_get_kv_type(ctx, id) != GgufType::Array
        || gguf_get_arr_type(ctx, id) != GgufType::String
    {
        return None;
    }
    Some(
        (0..gguf_get_arr_n(ctx, id))
            .map(|i| gguf_get_arr_str(ctx, id, i).to_owned())
            .collect(),
    )
}

fn load_imatrix_gguf(
    ctx: &GgufContext,
    tensors: GgmlContext,
    fname: &str,
) -> Result<LoadedImatrix> {
    if ctx.info.is_empty() {
        return Err(Error::new(format!("no data in file {}", fname)));
    }

    let mut loaded = LoadedImatrix {
        datasets: string_array(ctx, "imatrix.datasets").unwrap_or_default(),
        ..LoadedImatrix::default()
    };

    let datasets_key = gguf_find_key(ctx, "imatrix.datasets");
    let chunk_count_key = gguf_find_key(ctx, "imatrix.chunk_count");
    let chunk_size_key = gguf_find_key(ctx, "imatrix.chunk_size");
    loaded.has_metadata = datasets_key != -1 && chunk_count_key != -1 && chunk_size_key != -1;
    loaded.chunk_count = if chunk_count_key != -1
        && gguf_get_kv_type(ctx, chunk_count_key) == GgufType::Uint32
    {
        gguf_get_val_u32(ctx, chunk_count_key) as i32
    } else {
        0
    };

    loaded.stats_schema = string_array(ctx, "imatrix.stats_schema")
        .unwrap_or_else(default_stats_schema);

    let mut parts: BTreeMap<String, TensorParts<'_>> = BTreeMap::new();
    for cur in &tensors.tensors {
        let name = cur.name.clone();
        if name.is_empty() {
            continue;
        }
        if let Some(base) = name.strip_suffix(".in_sum") {
            parts.entry(base.to_owned()).or_default().in_sum = Some(cur);
        } else if let Some(base) = name.strip_suffix(".in_sum2") {
            parts.entry(base.to_owned()).or_default().in_sum2 = Some(cur);
        } else if let Some(base) = name.strip_suffix(".counts") {
            parts.entry(base.to_owned()).or_default().counts = Some(cur);
        } else if let Some(base) = name.strip_suffix(".stats") {
            parts.entry(base.to_owned()).or_default().stats = Some(cur);
        }
    }

    for (name, p) in parts {
        let (Some(in_sum2), Some(counts)) = (p.in_sum2, p.counts) else {
            return Err(Error::new(format!(
                "mismatched sums and counts for {}",
                name
            )));
        };
        if let Some(in_sum) = p.in_sum {
            if ggml_nelements(in_sum) != ggml_nelements(in_sum2) {
                return Err(Error::new(format!("mismatched sums and counts for {}", name)));
            }
        }
        if in_sum2.type_ != GgmlType::F32 || counts.type_ != GgmlType::F32 {
            return Err(Error::new(format!(
                "sums and counts for {} must be F32",
                name
            )));
        }

        let mut entry = ImatrixEntry::default();
        entry.sums = f32_vec_from_bytes(&in_sum2.data);
        entry.counts = f32_vec_from_bytes(&counts.data)
            .iter()
            .map(|&v| v.round() as i64)
            .collect();
        if let Some(in_sum) = p.in_sum {
            if ggml_nelements(in_sum) == ggml_nelements(in_sum2) {
                entry.activations = f32_vec_from_bytes(&in_sum.data);
            }
        }
        if let Some(stats) = p.stats {
            if stats.type_ == GgmlType::F32 {
                entry.stats = f32_vec_from_bytes(&stats.data);
            }
        }
        loaded.entries.insert(name, entry);
    }

    Ok(loaded)
}

fn read_i32(data: &[u8], pos: &mut usize) -> Option<i32> {
    if data.len().checked_sub(*pos)? < 4 {
        return None;
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&data[*pos..*pos + 4]);
    *pos += 4;
    Some(i32::from_ne_bytes(buf))
}

fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, n: usize) -> Option<&'a [u8]> {
    let end = pos.checked_add(n)?;
    let slice = data.get(*pos..end)?;
    *pos = end;
    Some(slice)
}

fn load_imatrix_legacy(fname: &str) -> Result<LoadedImatrix> {
    let data = fs::read(fname).map_err(|e| {
        Error::new(format!(
            "failed to open {} ({})",
            fname,
            e
        ))
    })?;
    let mut pos = 0usize;
    let n_entries = read_i32(&data, &mut pos).ok_or_else(|| {
        Error::new(format!("no data in file {}", fname))
    })?;
    if n_entries < 1 {
        return Err(Error::new(format!("no data in file {}", fname)));
    }

    let mut loaded = LoadedImatrix {
        is_legacy: true,
        ..LoadedImatrix::default()
    };
    for i in 0..n_entries {
        let len = read_i32(&data, &mut pos).ok_or_else(|| {
            Error::new(format!("failed reading name for entry {} from {}", i + 1, fname))
        })?;
        if len < 0 {
            return Err(Error::new(format!(
                "failed reading name for entry {} from {}",
                i + 1,
                fname
            )));
        }
        let name_bytes = read_bytes(&data, &mut pos, len as usize).ok_or_else(|| {
            Error::new(format!("failed reading name for entry {} from {}", i + 1, fname))
        })?;
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        let ncall = read_i32(&data, &mut pos).ok_or_else(|| {
            Error::new(format!("failed reading number of values for entry {}", i))
        })?;
        let nval = read_i32(&data, &mut pos).ok_or_else(|| {
            Error::new(format!("failed reading number of values for entry {}", i))
        })?;
        if nval < 1 {
            return Err(Error::new(format!(
                "failed reading number of values for entry {}",
                i
            )));
        }
        let raw = read_bytes(&data, &mut pos, nval as usize * 4).ok_or_else(|| {
            Error::new(format!("failed reading data for entry {}", i))
        })?;
        let mut entry = ImatrixEntry::default();
        entry.sums = f32_vec_from_bytes(raw);
        entry.counts.push(ncall as i64);
        loaded.entries.insert(name, entry);
    }

    if pos < data.len() {
        if let Some(n_calls) = read_i32(&data, &mut pos) {
            loaded.chunk_count = n_calls;
            if let Some(len) = read_i32(&data, &mut pos) {
                if len > 0 {
                    if let Some(raw) = read_bytes(&data, &mut pos, len as usize) {
                        loaded.datasets.push(String::from_utf8_lossy(raw).into_owned());
                    }
                }
            }
        }
    }
    Ok(loaded)
}

fn common_imatrix_load(fname: &str) -> Result<LoadedImatrix> {
    let mut params = GgufInitParams {
        no_alloc: false,
        ctx: Some(GgmlContext::default()),
    };
    match gguf_init_from_file(fname, &mut params) {
        Ok(ctx) => load_imatrix_gguf(&ctx, params.ctx.unwrap_or_default(), fname),
        Err(_) => load_imatrix_legacy(fname),
    }
}

struct PreparedImatrix {
    datasets: Vec<String>,
    values: SideData,
    activations: SideData,
    statistics: SideData,
    stats_schema: Vec<String>,
    chunk_count: i32,
}

fn load_imatrix(imatrix_file: &str) -> Result<PreparedImatrix> {
    let loaded = common_imatrix_load(imatrix_file).map_err(|e| {
        Error::new(format!(
            "load_imatrix: failed to load imatrix from '{}' ({})",
            imatrix_file, e.message
        ))
    })?;
    if !loaded.is_legacy && !loaded.has_metadata {
        return Err(Error::new(format!(
            "load_imatrix: missing imatrix metadata in file {}",
            imatrix_file
        )));
    }

    let mut out = PreparedImatrix {
        datasets: loaded.datasets,
        stats_schema: loaded.stats_schema,
        chunk_count: loaded.chunk_count,
        values: HashMap::new(),
        activations: HashMap::new(),
        statistics: HashMap::new(),
    };
    for (name, entry) in loaded.entries {
        let ncounts = entry.counts.len();
        if ncounts == 0 {
            println!("load_imatrix: no counts for entry '{}', skipping", name);
            continue;
        }
        let nsums = entry.sums.len();
        if nsums == 0 {
            continue;
        }
        let ne0 = nsums / ncounts;
        let mut value_row = vec![0.0f32; nsums];
        if loaded.is_legacy {
            let ncall = entry.counts[0];
            for (i, v) in entry.sums.iter().enumerate() {
                value_row[i] = if ncall > 0 { *v / ncall as f32 } else { *v };
            }
        } else {
            for j in 0..ncounts {
                let count = entry.counts[j] as f32;
                for i in 0..ne0 {
                    value_row[j * ne0 + i] = if count > 0.0 {
                        entry.sums[j * ne0 + i] / count
                    } else {
                        1.0
                    };
                }
            }
        }
        out.values.insert(name.clone(), value_row);

        if !entry.activations.is_empty() {
            let mut act_row = vec![0.0f32; entry.activations.len()];
            if loaded.is_legacy {
                let ncall = entry.counts[0];
                for (i, v) in entry.activations.iter().enumerate() {
                    act_row[i] = if ncall > 0 { *v / ncall as f32 } else { *v };
                }
            } else {
                for j in 0..ncounts {
                    let count = entry.counts[j] as f32;
                    for i in 0..ne0 {
                        act_row[j * ne0 + i] = if count > 0.0 {
                            entry.activations[j * ne0 + i] / count
                        } else {
                            0.0
                        };
                    }
                }
            }
            out.activations.insert(name.clone(), act_row);
        }
        if !entry.stats.is_empty() {
            out.statistics.insert(name, entry.stats);
        }
    }

    if !out.datasets.is_empty() {
        print!("load_imatrix: imatrix datasets=['{}'", out.datasets[0]);
        for d in &out.datasets[1..] {
            print!(", '{}'", d);
        }
        println!("]");
    }
    println!(
        "load_imatrix: loaded {} importance matrix entries from {} computed on {} chunks",
        out.values.len(),
        imatrix_file,
        out.chunk_count
    );
    Ok(out)
}

fn prepare_imatrix(
    imatrix_file: &str,
    included_weights: &[String],
    excluded_weights: &[String],
) -> Result<PreparedImatrix> {
    let mut out = if imatrix_file.is_empty() {
        PreparedImatrix {
            datasets: Vec::new(),
            values: HashMap::new(),
            activations: HashMap::new(),
            statistics: HashMap::new(),
            stats_schema: Vec::new(),
            chunk_count: -1,
        }
    } else {
        load_imatrix(imatrix_file)?
    };

    if !excluded_weights.is_empty() {
        for name in excluded_weights {
            out.values.retain(|k, _| !k.contains(name));
            out.activations.retain(|k, _| !k.contains(name));
            out.statistics.retain(|k, _| !k.contains(name));
        }
    }
    if !included_weights.is_empty() {
        let mut tmp_values: SideData = HashMap::new();
        let mut tmp_activations: SideData = HashMap::new();
        let mut tmp_statistics: SideData = HashMap::new();
        for name in included_weights {
            for (k, v) in &out.values {
                if k.contains(name) {
                    tmp_values.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
            for (k, v) in &out.activations {
                if k.contains(name) {
                    tmp_activations.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
            for (k, v) in &out.statistics {
                if k.contains(name) {
                    tmp_statistics.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
        }
        out.values = tmp_values;
        out.activations = tmp_activations;
        out.statistics = tmp_statistics;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// argument parsers
// ---------------------------------------------------------------------------

fn parse_ggml_type(arg: &str) -> Option<QType> {
    for i in 0..QType::TYPE_COUNT as i32 {
        if let Some(ty) = QType::from_i32(i) {
            if striequals(ty.name(), arg) {
                return Some(ty);
            }
        }
    }
    eprintln!("\nparse_ggml_type: invalid ggml_type '{}'\n", arg);
    None
}

fn parse_tensor_type(data: &str, tensor_type: &mut Vec<TensorTypeOverride>) -> bool {
    let Some(sep) = data.find('=') else {
        println!("\nparse_tensor_type: malformed tensor type '{}'\n", data);
        return false;
    };
    if sep == 0 {
        println!("\nparse_tensor_type: missing tensor name\n");
        return false;
    }
    let ty_str = &data[sep + 1..];
    if ty_str.is_empty() {
        println!("\nparse_tensor_type: missing quantization type\n");
        return false;
    }
    let name = data[..sep].to_ascii_lowercase();
    let ty = parse_ggml_type(ty_str);
    tensor_type.push(TensorTypeOverride {
        pattern: name,
        ty: ty.unwrap_or(QType::Count),
    });
    if ty.is_none() {
        println!("\nparse_tensor_type: invalid quantization type '{}'\n", ty_str);
        return false;
    }
    true
}

fn parse_tensor_type_file(filename: &str, tensor_type: &mut Vec<TensorTypeOverride>) -> bool {
    let content = match fs::read_to_string(filename) {
        Ok(v) => v,
        Err(e) => {
            println!("\nparse_tensor_type_file: failed to open file '{}': {}\n", filename, e);
            return false;
        }
    };
    for arg in content.split_whitespace() {
        if !parse_tensor_type(arg, tensor_type) {
            return false;
        }
    }
    true
}

fn parse_layer_prune(data: &str, prune_layers: &mut Vec<i32>) -> bool {
    for block_id in string_split(data, ",") {
        let id = parse_i32_prefix(&block_id).unwrap_or(-1);
        if id < 0 {
            println!("\nparse_layer_prune: invalid layer id '{}'\n", block_id);
            return false;
        }
        prune_layers.push(id);
    }
    prune_layers.sort_unstable();
    prune_layers.dedup();
    true
}

fn parse_target_bpw(data: &str) -> Option<f32> {
    let Some((v, _)) = parse_double_prefix(data) else {
        println!("\nparse_target_bpw: '{}' is not valid. Target bits per weight (bpw) must be a number between 0.0 and 16.0\n", data);
        return None;
    };
    let target = v as f32;
    if !target.is_finite() || !(0.0..=16.0).contains(&target) {
        println!("\nparse_target_bpw: target bits per weight (bpw) must be a number between 0.0 and 16.0\n");
        return None;
    }
    Some(target)
}

fn parse_target_size(data: &str) -> Option<i64> {
    let Some((val, end)) = parse_double_prefix(data) else {
        println!("\nparse_target_size: invalid target file size '{}'\n", data);
        return None;
    };
    if end == 0 || !val.is_finite() || val < 0.0 {
        println!("\nparse_target_size: invalid target file size '{}'\n", data);
        return None;
    }
    let suffix = data[end..].to_ascii_lowercase();
    let mul: i64 = match suffix.as_str() {
        "" | "b" => 1,
        "k" | "kib" => 1024,
        "m" | "mib" => 1024 * 1024,
        "g" | "gib" => 1024 * 1024 * 1024,
        "t" | "tib" => 1024i64 * 1024 * 1024 * 1024,
        _ => {
            println!("\nparse_target_size: invalid unit '{}' in '{}'. Allowed: b, k or kib, m or mib, g or gib, t or tib\n", suffix, data);
            return None;
        }
    };
    if val * mul as f64 >= i64::MAX as f64 {
        println!("\nparse_target_size: invalid target file size '{}'\n", data);
        return None;
    }
    Some((val * mul as f64) as i64)
}

fn parse_target_exclude(data: &str, target_exclude: &mut Vec<QType>) -> bool {
    for token in string_split(data, ",") {
        let mut ty = (0..QType::TYPE_COUNT as i32)
            .filter_map(QType::from_i32)
            .find(|t| striequals(t.name(), &token));
        if ty.is_none()
            && !token.is_empty()
            && token.chars().all(|c| c.is_ascii_digit())
        {
            if let Ok(ordinal) = token.parse::<i32>() {
                ty = QType::from_i32(ordinal);
            }
        }
        let Some(ty) = ty else {
            println!("\nparse_target_exclude: invalid quant type '{}' for --target-exclude\n", token);
            return false;
        };
        if !target_exclude.contains(&ty) {
            target_exclude.push(ty);
        }
    }
    true
}

const BPW_FTYPES: &[(f64, &str)] = &[
    (1.5625, "IQ1_S"),
    (1.7500, "IQ1_M"),
    (2.0625, "IQ2_XXS"),
    (2.3125, "IQ2_XS"),
    (2.5625, "IQ2_S"),
    (2.6250, "Q2_K"),
    (3.0625, "IQ3_XXS"),
    (3.4375, "Q3_K"),
    (4.2500, "IQ4_XS"),
    (4.5000, "Q4_K"),
    (5.5000, "Q5_K"),
    (6.5625, "Q6_K"),
    (8.5000, "Q8_0"),
    (16.0000, "F16"),
];

fn get_ftype(bpw: f32) -> &'static str {
    let bpw = bpw as f64;
    BPW_FTYPES
        .iter()
        .find(|(k, _)| *k >= bpw)
        .map(|(_, name)| *name)
        .unwrap_or("F16")
}

fn take_next(args: &[String], arg_idx: &mut usize) -> Option<String> {
    if *arg_idx + 1 >= args.len() {
        return None;
    }
    *arg_idx += 1;
    Some(args[*arg_idx].clone())
}

fn truncate_127(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if out.len() + c.len_utf8() > 127 {
            break;
        }
        out.push(c);
    }
    out
}

fn kv_str(key: &str, val: &str) -> KvOverride {
    KvOverride {
        key: key.to_owned(),
        tag: KvOverrideType::Str,
        val_f64: 0.0,
        val_i64: 0,
        val_bool: false,
        val_str: truncate_127(val),
    }
}

fn kv_int(key: &str, val: i64) -> KvOverride {
    KvOverride {
        key: key.to_owned(),
        tag: KvOverrideType::Int,
        val_f64: 0.0,
        val_i64: val,
        val_bool: false,
        val_str: String::new(),
    }
}

fn convert_kv_override(src: CommonKvOverride) -> KvOverride {
    let tag = match src.tag {
        CommonKvOverrideType::Int => KvOverrideType::Int,
        CommonKvOverrideType::Float => KvOverrideType::Float,
        CommonKvOverrideType::Bool => KvOverrideType::Bool,
        CommonKvOverrideType::Str => KvOverrideType::Str,
    };
    KvOverride {
        key: src.key,
        tag,
        val_f64: src.val_f64,
        val_i64: src.val_i64,
        val_bool: src.val_bool,
        val_str: src.val_str,
    }
}

fn basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn same_file(a: &str, b: &str) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(pa), Ok(pb)) => pa == pb,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// llama_quantize
// ---------------------------------------------------------------------------

pub fn llama_quantize(args: &[String]) -> Result<i32> {
    let executable = args.first().map(String::as_str).unwrap_or("llama-quantize");
    if args.len() < 3 {
        return Err(usage(executable));
    }

    let mut params = llama_model_quantize_default_params();
    let mut arg_idx = 1usize;
    let mut imatrix_file = String::new();
    let mut included_weights: Vec<String> = Vec::new();
    let mut excluded_weights: Vec<String> = Vec::new();
    let mut kv_overrides: Vec<KvOverride> = Vec::new();
    let mut tensor_type_opts: Vec<TensorTypeOverride> = Vec::new();
    let mut prune_layers: Vec<i32> = Vec::new();
    let mut target_exclude: Vec<QType> = Vec::new();
    let mut target_bpw = -1.0f32;
    let mut target_size = -1i64;

    while arg_idx < args.len() && args[arg_idx].starts_with("--") {
        match args[arg_idx].as_str() {
            "--leave-output-tensor" => params.quantize_output_tensor = false,
            "--output-tensor-type" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                let Some(ty) = parse_ggml_type(&v) else {
                    return Err(usage(executable));
                };
                params.output_tensor_type = ty;
            }
            "--token-embedding-type" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                let Some(ty) = parse_ggml_type(&v) else {
                    return Err(usage(executable));
                };
                params.token_embedding_type = ty;
            }
            "--tensor-type" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                if !parse_tensor_type(&v, &mut tensor_type_opts) {
                    return Err(usage(executable));
                }
            }
            "--tensor-type-file" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                if !parse_tensor_type_file(&v, &mut tensor_type_opts) {
                    return Err(usage(executable));
                }
            }
            "--target-bpw" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                let Some(bpw) = parse_target_bpw(&v) else {
                    return Err(usage(executable));
                };
                target_bpw = bpw;
            }
            "--target-size" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                let Some(size) = parse_target_size(&v) else {
                    return Err(usage(executable));
                };
                target_size = size;
            }
            "--target-exclude" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                if !parse_target_exclude(&v, &mut target_exclude) {
                    return Err(usage(executable));
                }
            }
            "--state-file" => {
                if arg_idx + 1 < args.len() && !args[arg_idx + 1].starts_with('-') {
                    if let Some(v) = take_next(args, &mut arg_idx) {
                        params.state_file = v;
                    }
                } else {
                    params.state_file.clear();
                }
            }
            "--prune-layers" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                if !parse_layer_prune(&v, &mut prune_layers) {
                    return Err(usage(executable));
                }
            }
            "--override-kv" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                let mut common_overrides: Vec<CommonKvOverride> = Vec::new();
                if !string_parse_kv_override(&v, &mut common_overrides) {
                    return Err(usage(executable));
                }
                kv_overrides.extend(common_overrides.into_iter().map(convert_kv_override));
            }
            "--dry-run" => params.dry_run = true,
            "--allow-requantize" => params.allow_requantize = true,
            "--pure" => params.pure = true,
            "--imatrix" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                imatrix_file = v;
            }
            "--include-weights" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                included_weights.push(v);
            }
            "--exclude-weights" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                excluded_weights.push(v);
            }
            "--keep-split" => params.keep_split = true,
            "--max-buffer-size" => {
                let Some(v) = take_next(args, &mut arg_idx) else {
                    return Err(usage(executable));
                };
                let mib = parse_i32_prefix(&v).unwrap_or(0);
                if mib <= 0 {
                    eprintln!("llama_quantize: invalid --max-buffer-size '{}'", v);
                    return Ok(1);
                }
                params.max_buf_size = mib as usize * 1024 * 1024;
            }
            _ => return Err(usage(executable)),
        }
        arg_idx += 1;
    }

    if args.len() - arg_idx < 2 {
        println!("{}: bad arguments", executable);
        return Err(usage(executable));
    }
    if !included_weights.is_empty() && !excluded_weights.is_empty() {
        return Err(usage(executable));
    }
    if target_bpw != -1.0f32 && target_size != -1 {
        return Err(usage(executable));
    }

    let prepared = prepare_imatrix(&imatrix_file, &included_weights, &excluded_weights)?;
    if !prepared.values.is_empty() {
        params.imatrix = prepared
            .values
            .iter()
            .map(|(name, data)| ImatrixData {
                name: name.clone(),
                data: data.clone(),
            })
            .collect();
        kv_overrides.push(kv_str(
            LLM_KV_QUANTIZE_IMATRIX_FILE,
            &basename(&imatrix_file),
        ));
        if let Some(dataset) = prepared.datasets.first() {
            kv_overrides.push(kv_str(
                LLM_KV_QUANTIZE_IMATRIX_DATASET,
                &basename(dataset),
            ));
        }
        kv_overrides.push(kv_int(
            LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES,
            prepared.values.len() as i64,
        ));
        if prepared.chunk_count > 0 {
            kv_overrides.push(kv_int(
                LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS,
                prepared.chunk_count as i64,
            ));
        }
    }
    if !prepared.activations.is_empty() {
        params.activations = prepared
            .activations
            .iter()
            .map(|(name, data)| ImatrixData {
                name: name.clone(),
                data: data.clone(),
            })
            .collect();
    }
    if !prepared.statistics.is_empty() {
        params.statistics = prepared
            .statistics
            .iter()
            .map(|(name, data)| ImatrixData {
                name: name.clone(),
                data: data.clone(),
            })
            .collect();
        params.stats_schema = prepared.stats_schema.clone();
    }
    if !kv_overrides.is_empty() {
        kv_overrides.push(KvOverride {
            key: String::new(),
            tag: KvOverrideType::Int,
            val_f64: 0.0,
            val_i64: 0,
            val_bool: false,
            val_str: String::new(),
        });
    }
    params.kv_overrides = kv_overrides;
    params.tt_overrides = tensor_type_opts;
    params.target_exclude = target_exclude;
    params.prune_layers = prune_layers;
    if target_bpw != -1.0f32 {
        params.target_bpw = target_bpw;
    }
    if target_size != -1 {
        params.target_size = target_size;
    }

    // No backend to initialize in the Rust port. Signal handling and the
    // abort callback are omitted: both need platform-specific calls.

    let fname_inp = args[arg_idx].clone();
    arg_idx += 1;
    let suffix = ".gguf";
    let (fname_out, ftype_str, ftype) = if let Some(parsed) = try_parse_ftype(&args[arg_idx]) {
        let (ftype, name) = parsed;
        let fname_out = if params.dry_run {
            String::new()
        } else {
            let fpath = match fname_inp.rfind(|c| c == '/' || c == '\\') {
                Some(pos) => fname_inp[..=pos].to_owned(),
                None => String::new(),
            };
            let mut out = format!("{}ggml-model-{}", fpath, name);
            if !params.keep_split {
                out.push_str(suffix);
            }
            out
        };
        arg_idx += 1;
        (fname_out, name, ftype)
    } else {
        let mut fname_out = args[arg_idx].clone();
        if params.keep_split && fname_out.contains(suffix) {
            fname_out.truncate(fname_out.len() - suffix.len());
        }
        arg_idx += 1;
        let chosen = if params.target_requested() {
            if params.target_bpw != -1.0f32 {
                get_ftype(params.target_bpw).to_owned()
            } else {
                "F16".to_owned()
            }
        } else {
            if arg_idx >= args.len() {
                eprintln!("llama_quantize: missing ftype");
                return Ok(1);
            }
            let v = args[arg_idx].clone();
            arg_idx += 1;
            v
        };
        if params.target_requested() {
            arg_idx += 1;
        }
        match try_parse_ftype(&chosen) {
            Some((ftype, name)) => (fname_out, name, ftype),
            None => {
                eprintln!("llama_quantize: invalid ftype '{}'", chosen);
                return Ok(1);
            }
        }
    };
    params.ftype = ftype;
    if ftype_str == "COPY" {
        params.only_copy = true;
    }

    if arg_idx < args.len() {
        match parse_i32_prefix(&args[arg_idx]) {
            Some(nthread) => params.nthread = nthread,
            None => {
                eprintln!("llama_quantize: invalid nthread '{}' (invalid argument)", args[arg_idx]);
                return Ok(1);
            }
        }
    }

    if !params.dry_run && same_file(&fname_inp, &fname_out) {
        eprintln!(
            "llama_quantize: error: input and output files are the same: '{}'",
            fname_inp
        );
        return Ok(1);
    }

    eprintln!("version: {}", crate::api::llama_version());
    if params.dry_run {
        eprint!(
            "llama_quantize: calculating quantization size for '{}' as {}",
            fname_inp, ftype_str
        );
    } else {
        eprint!(
            "llama_quantize: quantizing '{}' to '{}' as {}",
            fname_inp, fname_out, ftype_str
        );
    }
    if params.nthread > 0 {
        eprint!(" using {} threads", params.nthread);
    }
    eprintln!();

    let t_main_start = Instant::now();
    let t_start = Instant::now();
    if llama_model_quantize(&fname_inp, &fname_out, &params) != 0 {
        eprintln!(
            "llama_quantize: failed to quantize model from '{}'",
            fname_inp
        );
        return Ok(1);
    }
    let t_quantize_us = t_start.elapsed().as_micros() as i64;
    let t_main_end_us = t_main_start.elapsed().as_micros() as i64;

    println!();
    println!(
        "llama_quantize: quantize time = {:8.2} ms",
        t_quantize_us as f64 / 1000.0
    );
    println!(
        "llama_quantize:    total time = {:8.2} ms",
        t_main_end_us as f64 / 1000.0
    );
    Ok(0)
}

/// `main.cpp` entry point.
pub fn main(args: &[String]) -> i32 {
    match llama_quantize(args) {
        Ok(code) => code,
        Err(err) => {
            if !err.message.is_empty() {
                eprintln!("{}", err.message);
            }
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn parses_ftypes_by_name_and_ordinal() {
        assert_eq!(try_parse_ftype("q4_k_m"), Some((FType::MostlyQ4KM, "Q4_K_M".into())));
        assert_eq!(try_parse_ftype("15"), Some((FType::MostlyQ4KM, "Q4_K".into())));
        assert_eq!(try_parse_ftype("0"), Some((FType::AllF32, "F32".into())));
        assert_eq!(try_parse_ftype("copy"), Some((FType::AllF32, "COPY".into())));
        assert_eq!(try_parse_ftype("nope"), None);
    }

    #[test]
    fn parses_target_options() {
        assert_eq!(parse_target_bpw("4.5"), Some(4.5));
        assert_eq!(parse_target_bpw("17"), None);
        assert_eq!(parse_target_size("2m"), Some(2 * 1024 * 1024));
        assert_eq!(parse_target_size("1GiB"), Some(1024 * 1024 * 1024));
        assert_eq!(parse_target_size("2x"), None);
    }

    #[test]
    fn parses_tensor_types_and_prune_layers() {
        let mut tt = Vec::new();
        assert!(parse_tensor_type("attn_q=q8_0", &mut tt));
        assert_eq!(tt[0].pattern, "attn_q");
        assert_eq!(tt[0].ty, QType::Q8_0);
        assert!(!parse_tensor_type("bad", &mut Vec::new()));
        let mut prune = Vec::new();
        assert!(parse_layer_prune("3,1,3", &mut prune));
        assert_eq!(prune, vec![1, 3]);
    }

    #[test]
    fn rejects_unknown_option() {
        let args = argv(&["llama-quantize", "--bogus", "in.gguf", "Q4_K"]);
        assert_eq!(main(&args), 1);
    }
}

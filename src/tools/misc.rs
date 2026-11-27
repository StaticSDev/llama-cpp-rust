//! Ports of the remaining small llama.cpp command-line tools.
//!
//! Covers `gguf-split`, `tokenize`, `tts`, `rpc-server`,
//! `cvector-generator` (with `pca.hpp` and `mean.hpp`) and `export-lora`.
//! GGUF files go through `crate::gguf`. Backends, model inference, audio
//! generation and the RPC service have no FFI in this port, so those parts are
//! local stand-ins that preserve the original control flow and file formats.

use crate::core::{Error, Result, Token};
use crate::gguf::{
    ggml_nbytes, gguf_add_tensor, gguf_find_key, gguf_find_tensor, gguf_get_data_offset,
    gguf_get_meta_data, gguf_get_meta_size, gguf_get_n_tensors, gguf_get_tensor_offset,
    gguf_get_val_f32, gguf_get_val_i32, gguf_get_val_str, gguf_get_val_u16, gguf_get_val_u32,
    gguf_init_from_file, gguf_pad, gguf_set_kv, gguf_set_val_i32, gguf_set_val_str,
    gguf_set_val_u16, gguf_set_val_u32, gguf_write_to_file, GgufContext, GgufInitParams,
    GgmlContext, GgmlTensor, GgmlType, GGUF_DEFAULT_ALIGNMENT,
};
use crate::quant::{
    bf16_to_f32, dequantize_slab, f16_to_f32, f32_to_bf16, f32_to_f16, NativeQuantizer, QType,
};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const KEY_SPLIT_NO: &str = "split.no";
const KEY_SPLIT_COUNT: &str = "split.count";
const KEY_SPLIT_TENSORS_COUNT: &str = "split.tensors.count";
const KEY_GENERAL_ARCH: &str = "general.architecture";
const KEY_GENERAL_TYPE: &str = "general.type";
const KEY_GENERAL_FILE_TYPE: &str = "general.file_type";
const KEY_ADAPTER_TYPE: &str = "adapter.type";
const KEY_ADAPTER_LORA_ALPHA: &str = "adapter.lora.alpha";
const KEY_BLOCK_COUNT: &str = "block_count";
const LLAMA_FTYPE_MOSTLY_F16: u32 = 1;

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

fn io_error(err: io::Error) -> Error {
    Error::new(err.to_string())
}

fn kv_str(ctx: &GgufContext, key: &str) -> Option<String> {
    let id = gguf_find_key(ctx, key);
    if id < 0 {
        None
    } else {
        Some(gguf_get_val_str(ctx, id).to_owned())
    }
}

fn kv_u32(ctx: &GgufContext, key: &str) -> Option<u32> {
    let id = gguf_find_key(ctx, key);
    if id < 0 {
        None
    } else {
        Some(gguf_get_val_u32(ctx, id))
    }
}

fn kv_f32(ctx: &GgufContext, key: &str) -> f32 {
    let id = gguf_find_key(ctx, key);
    if id < 0 {
        0.0
    } else {
        gguf_get_val_f32(ctx, id)
    }
}

/// `gguf_init_from_file` wrapper. `load_data` mirrors `gguf_init_params::no_alloc`.
pub fn load_gguf(path: &str, load_data: bool) -> Result<(GgufContext, GgmlContext)> {
    let mut params = GgufInitParams {
        no_alloc: !load_data,
        ctx: Some(GgmlContext::default()),
    };
    let ctx = gguf_init_from_file(path, &mut params)?;
    let meta = params.ctx.unwrap_or_default();
    Ok((ctx, meta))
}

fn read_at(file: &mut File, offset: usize, len: usize) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(offset as u64)).map_err(io_error)?;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf).map_err(|err| {
        Error::new(format!(
            "failed to read {} bytes at offset {}: {}",
            len, offset, err
        ))
    })?;
    Ok(buf)
}

fn write_zeros(out: &mut impl Write, n: usize) -> Result<()> {
    let zeros = vec![0u8; n];
    out.write_all(&zeros).map_err(io_error)
}

/// Read one positional value, advancing the argument index like `argv[++i]`.
fn take_value(args: &[String], i: &mut usize, flag: &str) -> Result<String> {
    *i += 1;
    if *i >= args.len() {
        return Err(Error::new(format!(
            "error: invalid parameter for argument: {}",
            flag
        )));
    }
    Ok(args[*i].clone())
}

fn prog_name<'a>(args: &'a [String], fallback: &'a str) -> &'a str {
    args.first().map(String::as_str).unwrap_or(fallback)
}

/// `result<T>` from `std::stoi`, with the same failure surface as a parse error.
fn parse_i32(text: &str, what: &str) -> Result<i32> {
    text.trim()
        .parse::<i32>()
        .map_err(|_| Error::new(format!("error: invalid {}: {}", what, text)))
}

// ===========================================================================
// gguf-split
// ===========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitOperation {
    None,
    Split,
    Merge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitMode {
    None,
    Tensor,
    Size,
}

#[derive(Clone, Debug)]
pub struct SplitParams {
    pub operation: SplitOperation,
    pub mode: SplitMode,
    pub n_bytes_split: usize,
    pub n_split_tensors: usize,
    pub input: String,
    pub output: String,
    pub no_tensor_first_split: bool,
    pub dry_run: bool,
    pub delete_splits: bool,
}

impl Default for SplitParams {
    fn default() -> Self {
        Self {
            operation: SplitOperation::None,
            mode: SplitMode::None,
            n_bytes_split: 0,
            n_split_tensors: 128,
            input: String::new(),
            output: String::new(),
            no_tensor_first_split: false,
            dry_run: false,
            delete_splits: false,
        }
    }
}

pub fn split_print_usage(executable: &str) {
    println!("\nusage: {} [options] GGUF_IN GGUF_OUT\n", executable);
    println!("Apply a GGUF operation on IN to OUT.\n");
    println!("options:");
    println!("  -h, --help              show this help message and exit");
    println!("  --version               show version and build info");
    println!("  --split                 split GGUF to multiple GGUF (enabled by default)");
    println!("  --merge                 merge multiple GGUF to a single GGUF");
    println!("  --split-max-tensors     max tensors in each split (default: 128)");
    println!("  --split-max-size N(M|G) max size per split");
    println!("  --no-tensor-first-split do not add tensors to the first split (disabled by default)");
    println!("  --dry-run               only print out a split plan and exit, without writing any new files");
    println!("  --delete-splits         delete the split files during merge to free up disk space WARNING: this option is unsafe and will leave you in an unrecoverable state if something fails during the merge\n");
}

/// Convert `"128M"` / `"4G"` to bytes, as `split_str_to_n_bytes` does.
pub fn split_str_to_n_bytes(text: &str) -> Result<usize> {
    let (number, multiplier) = match text.as_bytes().last().copied() {
        Some(b'M') => (&text[..text.len() - 1], 1000 * 1000usize),
        Some(b'G') => (&text[..text.len() - 1], 1000 * 1000 * 1000usize),
        Some(c) => {
            return Err(Error::new(format!(
                "error: supported units are M (megabytes) or G (gigabytes), but got: {}",
                c as char
            )))
        }
        None => {
            return Err(Error::new(
                "error: supported units are M (megabytes) or G (gigabytes)",
            ))
        }
    };
    let n = parse_i32(number.trim(), "size")?;
    if n <= 0 {
        return Err(Error::new("error: size must be a positive value"));
    }
    Ok((n as usize).saturating_mul(multiplier))
}

/// `Ok(None)` means help/version was handled and no work remains.
pub fn split_params_parse(args: &[String]) -> Result<Option<SplitParams>> {
    let mut params = SplitParams::default();
    let mut i = 0usize;
    while i < args.len() && (args[i].starts_with("--") || args[i] == "-h") {
        let arg = args[i].replace('_', "-");
        if arg == "-h" || arg == "--help" {
            split_print_usage(prog_name(args, "gguf-split"));
            return Ok(None);
        } else if arg == "--version" {
            eprintln!("version: 0.1.0 (dry port)");
            return Ok(None);
        } else if arg == "--dry-run" {
            params.dry_run = true;
        } else if arg == "--no-tensor-first-split" {
            params.no_tensor_first_split = true;
        } else if arg == "--merge" {
            if params.operation != SplitOperation::None && params.operation != SplitOperation::Merge
            {
                return Err(Error::new(
                    "error: either --split or --merge can be specified, but not both",
                ));
            }
            params.operation = SplitOperation::Merge;
        } else if arg == "--split" {
            if params.operation != SplitOperation::None && params.operation != SplitOperation::Split
            {
                return Err(Error::new(
                    "error: either --split or --merge can be specified, but not both",
                ));
            }
            params.operation = SplitOperation::Split;
        } else if arg == "--split-max-tensors" {
            let value = take_value(args, &mut i, &arg)?;
            if params.mode != SplitMode::None && params.mode != SplitMode::Tensor {
                return Err(Error::new(
                    "error: either --split-max-tensors or --split-max-size can be specified, but not both",
                ));
            }
            params.mode = SplitMode::Tensor;
            let n = parse_i32(&value, "number of tensors")?;
            if n <= 0 {
                return Err(Error::new("error: number of tensors must be positive"));
            }
            params.n_split_tensors = n as usize;
        } else if arg == "--split-max-size" {
            let value = take_value(args, &mut i, &arg)?;
            if params.mode != SplitMode::None && params.mode != SplitMode::Size {
                return Err(Error::new(
                    "error: either --split-max-tensors or --split-max-size can be specified, but not both",
                ));
            }
            params.mode = SplitMode::Size;
            params.n_bytes_split = split_str_to_n_bytes(&value)?;
        } else if arg == "--delete-splits" {
            params.delete_splits = true;
        } else {
            return Err(Error::new(format!("error: unknown argument: {}", arg)));
        }
        i += 1;
    }

    if params.operation == SplitOperation::None {
        params.operation = SplitOperation::Split;
    }
    if params.mode == SplitMode::None {
        params.mode = SplitMode::Tensor;
    }
    if args.len().saturating_sub(i) != 2 {
        return Err(Error::new("error: bad arguments"));
    }
    params.input = args[i].clone();
    params.output = args[i + 1].clone();
    Ok(Some(params))
}

fn new_split_ctx(ctx_gguf: &GgufContext, i_split: usize, n_tensors: usize) -> GgufContext {
    let mut out = GgufContext::default();
    // The first split carries all metadata; later splits only carry split keys.
    if i_split == 0 {
        gguf_set_kv(&mut out, ctx_gguf);
    }
    gguf_set_val_u16(&mut out, KEY_SPLIT_NO, i_split as u16);
    gguf_set_val_u16(&mut out, KEY_SPLIT_COUNT, 0);
    gguf_set_val_i32(&mut out, KEY_SPLIT_TENSORS_COUNT, n_tensors as i32);
    out
}

fn new_ctx_out(
    outs: &mut Vec<GgufContext>,
    cur: &mut Option<GgufContext>,
    i_split: &mut i32,
    ctx_gguf: &GgufContext,
    n_tensors: usize,
    allow_no_tensors: bool,
) -> Result<()> {
    *i_split += 1;
    if let Some(ctx) = cur.take() {
        if gguf_get_n_tensors(&ctx) == 0 && !allow_no_tensors {
            return Err(Error::new(
                "error: one of splits have 0 tensors. Maybe size or tensors limit is too small",
            ));
        }
        outs.push(ctx);
    }
    *cur = Some(new_split_ctx(ctx_gguf, *i_split as usize, n_tensors));
    Ok(())
}

fn should_split(
    params: &SplitParams,
    i_tensor: usize,
    next_size: usize,
    n_tensors: usize,
) -> Result<bool> {
    match params.mode {
        SplitMode::Size => Ok(next_size > params.n_bytes_split),
        SplitMode::Tensor => Ok(i_tensor > 0
            && i_tensor < n_tensors
            && i_tensor % params.n_split_tensors.max(1) == 0),
        SplitMode::None => Err(Error::new("error: invalid split mode")),
    }
}

pub struct SplitStrategy {
    pub params: SplitParams,
    pub ctx_gguf: GgufContext,
    pub ctx_outs: Vec<GgufContext>,
    pub n_tensors: usize,
}

impl SplitStrategy {
    pub fn new(params: &SplitParams, ctx_gguf: GgufContext) -> Result<Self> {
        let n_tensors = gguf_get_n_tensors(&ctx_gguf) as usize;
        let mut outs = Vec::new();
        let mut cur = None;
        let mut i_split = -1i32;

        new_ctx_out(&mut outs, &mut cur, &mut i_split, &ctx_gguf, n_tensors, false)?;
        if params.no_tensor_first_split {
            new_ctx_out(&mut outs, &mut cur, &mut i_split, &ctx_gguf, n_tensors, true)?;
        }

        let mut curr_tensors_size = 0usize;
        for i in 0..n_tensors {
            let n_bytes = gguf_pad(
                ggml_nbytes(&ctx_gguf.info[i].t),
                GGUF_DEFAULT_ALIGNMENT,
            );
            let next = curr_tensors_size + n_bytes;
            if should_split(params, i, next, n_tensors)? {
                new_ctx_out(&mut outs, &mut cur, &mut i_split, &ctx_gguf, n_tensors, false)?;
                curr_tensors_size = n_bytes;
            } else {
                curr_tensors_size = next;
            }
            let tensor = ctx_gguf.info[i].t.clone();
            gguf_add_tensor(cur.as_mut().expect("split context"), &tensor);
        }
        outs.push(cur.expect("last split context"));

        let count = outs.len() as u16;
        for ctx in &mut outs {
            gguf_set_val_u16(ctx, KEY_SPLIT_COUNT, count);
        }

        Ok(Self {
            params: params.clone(),
            ctx_gguf,
            ctx_outs: outs,
            n_tensors,
        })
    }

    pub fn print_info(&self) {
        println!("n_split: {}", self.ctx_outs.len());
        for (i, ctx) in self.ctx_outs.iter().enumerate() {
            let mut total_size = gguf_get_meta_size(ctx);
            for info in &ctx.info {
                let id = gguf_find_tensor(&self.ctx_gguf, &info.t.name);
                if id >= 0 {
                    total_size += ggml_nbytes(&self.ctx_gguf.info[id as usize].t);
                }
            }
            total_size /= 1000 * 1000;
            println!(
                "split {:05}: n_tensors = {}, total_size = {}M",
                i + 1,
                gguf_get_n_tensors(ctx),
                total_size
            );
        }
    }

    pub fn write(&self) -> Result<()> {
        let mut input = File::open(&self.params.input).map_err(io_error)?;
        let n_split = self.ctx_outs.len();
        for (i_split, ctx) in self.ctx_outs.iter().enumerate() {
            let split_path =
                crate::model_loader::split_path(&self.params.output, i_split as u32, n_split as u32);
            print!("Writing file {} ... ", split_path);
            io::stdout().flush().map_err(io_error)?;
            let mut out = File::create(&split_path).map_err(io_error)?;
            out.write_all(&gguf_get_meta_data(ctx)).map_err(io_error)?;

            for info in &ctx.info {
                let id = gguf_find_tensor(&self.ctx_gguf, &info.t.name);
                if id < 0 {
                    return Err(Error::new(format!(
                        "tensor {} not found in input GGUF",
                        info.t.name
                    )));
                }
                let offset = gguf_get_data_offset(&self.ctx_gguf)
                    + gguf_get_tensor_offset(&self.ctx_gguf, id);
                let len = ggml_nbytes(&self.ctx_gguf.info[id as usize].t);
                let data = read_at(&mut input, offset, len)?;
                out.write_all(&data).map_err(io_error)?;
                write_zeros(&mut out, gguf_pad(len, GGUF_DEFAULT_ALIGNMENT) - len)?;
            }
            println!("done");
        }
        Ok(())
    }
}

pub fn gguf_split(params: &SplitParams) -> Result<()> {
    let (ctx_gguf, _) = load_gguf(&params.input, false)?;
    let strategy = SplitStrategy::new(params, ctx_gguf)?;
    let n_split = strategy.ctx_outs.len();
    strategy.print_info();
    if !params.dry_run {
        strategy.write()?;
    }
    eprintln!(
        "gguf_split: {} gguf split written with a total of {} tensors.",
        n_split, strategy.n_tensors
    );
    Ok(())
}

pub fn gguf_merge(params: &SplitParams) -> Result<()> {
    eprintln!("gguf_merge: {} -> {}", params.input, params.output);
    if Path::new(&params.output).exists() {
        return Err(Error::new(format!(
            "gguf_merge: output file {} already exists",
            params.output
        )));
    }

    let (first, _) = load_gguf(&params.input, false)?;
    let key_count = gguf_find_key(&first, KEY_SPLIT_COUNT);
    if key_count < 0 {
        return Err(Error::new(format!(
            "gguf_merge: input file does not contain {} metadata",
            KEY_SPLIT_COUNT
        )));
    }
    let n_split = gguf_get_val_u16(&first, key_count) as usize;
    if n_split < 1 {
        return Err(Error::new(format!(
            "gguf_merge: input file does not contain a valid split count {}",
            n_split
        )));
    }
    let prefix = crate::model_loader::split_prefix(&params.input, 0, n_split as u32)
        .ok_or_else(|| Error::new(format!("gguf_merge: unexpected input file name: {}", params.input)))?;

    let mut first_fixed = first;
    gguf_set_val_u16(&mut first_fixed, KEY_SPLIT_COUNT, 0);
    let mut out = GgufContext::default();
    gguf_set_kv(&mut out, &first_fixed);

    let mut ctxs = Vec::new();
    let mut total_tensors = 0usize;
    for i in 0..n_split {
        let split_path = crate::model_loader::split_path(&prefix, i as u32, n_split as u32);
        let (ctx, _) = load_gguf(&split_path, false)?;
        if i > 0 {
            let no = gguf_find_key(&ctx, KEY_SPLIT_NO);
            let count = gguf_find_key(&ctx, KEY_SPLIT_COUNT);
            if no < 0
                || gguf_get_val_u16(&ctx, no) as usize != i
                || count < 0
                || gguf_get_val_u16(&ctx, count) as usize != n_split
            {
                return Err(Error::new(format!(
                    "gguf_merge: {} is not split {} of {}",
                    split_path,
                    i + 1,
                    n_split
                )));
            }
        }
        let n_tensors = gguf_get_n_tensors(&ctx) as usize;
        for info in &ctx.info {
            gguf_add_tensor(&mut out, &info.t);
        }
        total_tensors += n_tensors;
        ctxs.push(ctx);
    }

    let key_tensors = gguf_find_key(&ctxs[0], KEY_SPLIT_TENSORS_COUNT);
    if key_tensors < 0 || gguf_get_val_i32(&ctxs[0], key_tensors) as usize != total_tensors {
        return Err(Error::new(format!(
            "gguf_merge: splits contain {} tensors, which does not match {}",
            total_tensors, KEY_SPLIT_TENSORS_COUNT
        )));
    }

    let mut fout = if params.dry_run {
        None
    } else {
        let mut file = File::create(&params.output).map_err(io_error)?;
        file.write_all(&gguf_get_meta_data(&out)).map_err(io_error)?;
        Some(file)
    };

    for (i, ctx) in ctxs.iter().enumerate() {
        let split_path = crate::model_loader::split_path(&prefix, i as u32, n_split as u32);
        let mut input = File::open(&split_path).map_err(io_error)?;
        for info in &ctx.info {
            let id = gguf_find_tensor(ctx, &info.t.name);
            if id < 0 {
                return Err(Error::new(format!(
                    "gguf_merge: tensor {} not found in {}",
                    info.t.name, split_path
                )));
            }
            let offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, id);
            let len = ggml_nbytes(&ctx.info[id as usize].t);
            if let Some(ref mut out_file) = fout {
                let data = read_at(&mut input, offset, len)?;
                out_file.write_all(&data).map_err(io_error)?;
                write_zeros(out_file, gguf_pad(len, GGUF_DEFAULT_ALIGNMENT) - len)?;
            }
        }
        if !params.dry_run && params.delete_splits {
            fs::remove_file(&split_path).map_err(|err| {
                Error::new(format!("error: failed to delete {}: {}", split_path, err))
            })?;
            eprintln!("gguf_merge: deleted file {}", split_path);
        }
    }

    if let Some(ref mut out_file) = fout {
        out_file.seek(SeekFrom::Start(0)).map_err(io_error)?;
        out_file
            .write_all(&gguf_get_meta_data(&out))
            .map_err(io_error)?;
    }

    eprintln!(
        "gguf_merge: {} merged from {} split with {} tensors.",
        params.output, n_split, total_tensors
    );
    Ok(())
}

pub fn gguf_split_run(args: &[String]) -> Result<()> {
    let Some(params) = split_params_parse(args)? else {
        return Ok(());
    };
    match params.operation {
        SplitOperation::Split => gguf_split(&params),
        SplitOperation::Merge => gguf_merge(&params),
        SplitOperation::None => {
            split_print_usage(prog_name(args, "gguf-split"));
            Err(Error::new("error: no operation specified"))
        }
    }
}

// ===========================================================================
// tokenize
// ===========================================================================

#[derive(Clone, Debug, Default)]
pub struct TokenizeParams {
    pub model: String,
    pub prompt: String,
    pub prompt_file: Option<String>,
    pub tokenize_stdin: bool,
    pub tokenize_ids: bool,
    pub tokenize_show_count: bool,
    pub tokenize_no_bos: bool,
    pub parse_special: bool,
    pub escape: bool,
}

pub fn tokenize_print_usage(executable: &str) {
    println!("\nexample usage:\n");
    println!("  {} -m your_model.gguf -p \"Hello world\"\n", executable);
    println!("  {} -m your_model.gguf -f prompt.txt --ids\n", executable);
    println!(
        "  cat prompt.txt | {} -m your_model.gguf --stdin --show-count\n",
        executable
    );
}

pub fn tokenize_params_parse(args: &[String]) -> Result<Option<TokenizeParams>> {
    let mut params = TokenizeParams {
        escape: true,
        ..Default::default()
    };
    let mut i = 1usize;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-h" | "--help" => {
                tokenize_print_usage(prog_name(args, "tokenize"));
                return Ok(None);
            }
            "-m" | "--model" => params.model = take_value(args, &mut i, arg)?,
            "-p" | "--prompt" => params.prompt = take_value(args, &mut i, arg)?,
            "-f" | "--file" => {
                let path = take_value(args, &mut i, arg)?;
                params.prompt_file = Some(path.clone());
                params.prompt = path;
            }
            "--stdin" => params.tokenize_stdin = true,
            "--ids" => params.tokenize_ids = true,
            "--show-count" => params.tokenize_show_count = true,
            "--no-bos" => params.tokenize_no_bos = true,
            "--parse-special" => params.parse_special = true,
            "--escape" => params.escape = true,
            "--no-escape" => params.escape = false,
            _ => return Err(Error::new(format!("error: unknown argument: {}", arg))),
        }
        i += 1;
    }
    if !params.tokenize_stdin && params.prompt_file.is_none() && params.prompt.is_empty() {
        return Err(Error::new(
            "error: must specify one of: --stdin, --file or --prompt",
        ));
    }
    Ok(Some(params))
}

pub fn tokenize_run(args: &[String]) -> Result<()> {
    let Some(params) = tokenize_params_parse(args)? else {
        return Ok(());
    };
    if params.model.is_empty() {
        return Err(Error::new("error: no model file specified"));
    }

    let prompt = if params.tokenize_stdin {
        let mut text = String::new();
        io::stdin()
            .read_to_string(&mut text)
            .map_err(|err| Error::new(format!("error: could not read standard input: {}", err)))?;
        if params.escape {
            crate::common::string_process_escapes(&mut text);
        }
        text
    } else if let Some(ref path) = params.prompt_file {
        let bytes = fs::read(path).map_err(|err| {
            Error::new(format!("error: could not open file '{}' for reading: {}", path, err))
        })?;
        let mut text = String::from_utf8_lossy(&bytes).into_owned();
        if params.escape {
            crate::common::string_process_escapes(&mut text);
        }
        text
    } else {
        params.prompt.clone()
    };

    // The tokenizer needs only the vocabulary, not the weights.
    let (gguf, _) = load_gguf(&params.model, false)?;
    let loader = crate::model::LlamaModelLoader::from_gguf(
        gguf,
        &crate::model::ModelParams::default(),
    )?;
    let mut vocab = crate::vocab::LlamaVocab::new();
    vocab.load(&loader.vocab_metadata())?;

    let add_bos = vocab.get_add_bos() && !params.tokenize_no_bos;
    let tokens = crate::common::common_tokenize(&vocab, &prompt, add_bos, params.parse_special)?;

    if params.tokenize_ids {
        print!("[");
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}", token);
        }
        println!("]");
    } else {
        for token in &tokens {
            let piece = crate::common::common_token_to_piece(&vocab, *token, false);
            println!("{:6} -> '{}'", token, piece);
        }
    }
    if params.tokenize_show_count {
        println!("Total number of tokens: {}", tokens.len());
    }
    Ok(())
}

// ===========================================================================
// tts
// ===========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioGenType {
    None,
    Generic,
}

#[derive(Clone, Debug)]
pub struct AudioInfo {
    pub type_: AudioGenType,
    pub n_voices: usize,
    pub voices: Vec<String>,
}

impl Default for AudioInfo {
    fn default() -> Self {
        Self {
            type_: AudioGenType::Generic,
            n_voices: 0,
            voices: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TtsParams {
    pub model: String,
    pub mmproj: String,
    pub prompt: String,
    pub out_file: String,
    pub tts_lang: String,
    pub tts_voice: String,
    pub tts_speaker_file: String,
    pub tts_speaker_text: String,
    pub n_predict: i32,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u32,
    pub temp: Option<f32>,
    pub n_threads: usize,
    pub embedding: bool,
}

pub fn tts_print_usage(executable: &str) {
    println!("\nexample usage:\n");
    println!(
        "    {} -m backbone.gguf --mmproj mmproj.gguf -p \"text to speak\" -o output.wav",
        executable
    );
    println!(
        "    {} -hf user/model -p \"text to speak\" -o output.wav\n",
        executable
    );
    println!("note: --tts-lang, --tts-voice and --tts-speaker-file may not be supported in all models");
    println!("      use -n to limit the output length");
    println!("      see docs/multimodal/tts.md for per-model usage notes\n");
}

pub fn tts_params_parse(args: &[String]) -> Result<Option<TtsParams>> {
    let mut params = TtsParams {
        n_predict: -1,
        top_k: 0,
        top_p: 1.0,
        ..Default::default()
    };
    let mut i = 1usize;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-h" | "--help" => {
                tts_print_usage(prog_name(args, "tts"));
                return Ok(None);
            }
            "-m" | "--model" => params.model = take_value(args, &mut i, arg)?,
            "--mmproj" | "-mm" => params.mmproj = take_value(args, &mut i, arg)?,
            "-p" | "--prompt" => params.prompt = take_value(args, &mut i, arg)?,
            "-o" | "--out" | "--out-file" => params.out_file = take_value(args, &mut i, arg)?,
            "--tts-lang" => params.tts_lang = take_value(args, &mut i, arg)?,
            "--tts-voice" => params.tts_voice = take_value(args, &mut i, arg)?,
            "--tts-speaker-file" => params.tts_speaker_file = take_value(args, &mut i, arg)?,
            "--tts-speaker-text" => params.tts_speaker_text = take_value(args, &mut i, arg)?,
            "-n" | "--n-predict" => params.n_predict = parse_i32(&take_value(args, &mut i, arg)?, "n_predict")?,
            "--top-k" => params.top_k = parse_i32(&take_value(args, &mut i, arg)?, "top_k")?,
            "--top-p" => {
                params.top_p = take_value(args, &mut i, arg)?
                    .parse::<f32>()
                    .map_err(|_| Error::new("error: invalid top_p"))?
            }
            "--seed" => {
                params.seed = take_value(args, &mut i, arg)?
                    .parse::<u32>()
                    .map_err(|_| Error::new("error: invalid seed"))?
            }
            "--temp" => {
                params.temp = Some(
                    take_value(args, &mut i, arg)?
                        .parse::<f32>()
                        .map_err(|_| Error::new("error: invalid temp"))?,
                )
            }
            "-t" | "--threads" => {
                let n = parse_i32(&take_value(args, &mut i, arg)?, "threads")?;
                params.n_threads = n.max(1) as usize;
            }
            _ => return Err(Error::new(format!("error: unknown argument: {}", arg))),
        }
        i += 1;
    }
    Ok(Some(params))
}

/// `tts_timings`: reports generation speed at most every two seconds.
#[derive(Debug)]
pub struct TtsTimings {
    pub t_start: Instant,
    pub t_last: Instant,
}

impl Default for TtsTimings {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            t_start: now,
            t_last: now,
        }
    }
}

impl TtsTimings {
    pub fn report(&mut self, n_frames: usize) {
        let now = Instant::now();
        if now.duration_since(self.t_last).as_micros() < 2_000_000 {
            return;
        }
        self.t_last = now;
        let elapsed = now.duration_since(self.t_start).as_secs_f64();
        let fps = if elapsed > 0.0 {
            n_frames as f64 / elapsed
        } else {
            0.0
        };
        println!(
            "frames generated: {}, speed: {:.2} frames/s",
            n_frames, fps
        );
    }
}

/// Stand-in for `mtmd_context`: the model may not be loaded, but the CLI flow
/// and the audio-info checks are preserved.
#[derive(Clone, Debug)]
pub struct MtmdContext {
    pub info: AudioInfo,
}

impl MtmdContext {
    pub fn from_file(path: &str, _model: &str) -> Result<Self> {
        if !Path::new(path).exists() {
            return Err(Error::new(format!(
                "failed to load mmproj {}",
                path
            )));
        }
        Ok(Self {
            info: AudioInfo::default(),
        })
    }

    pub fn gen_audio_get_info(&self) -> &AudioInfo {
        &self.info
    }

    pub fn voice_name(&self, index: usize) -> Option<&str> {
        self.info.voices.get(index).map(String::as_str)
    }

    /// Grapheme-to-phoneme stand-in: the plain word.
    pub fn g2p(&self, word: &str) -> String {
        word.to_owned()
    }
}

#[derive(Clone, Debug, Default)]
pub struct GenAudioInput {
    pub seq_id: i32,
    pub prompt: String,
    pub speaker_ref: Option<PathBuf>,
    pub speaker_ref_text: Option<String>,
    pub lang: String,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u32,
    pub temp: Option<f32>,
}

#[derive(Debug)]
pub struct GenAudio {
    pub valid: bool,
    pub frames: usize,
    pub input: GenAudioInput,
}

impl Default for GenAudio {
    fn default() -> Self {
        Self {
            valid: true,
            frames: 0,
            input: GenAudioInput::default(),
        }
    }
}

impl GenAudio {
    pub fn init(_lctx: usize, _mctx: &MtmdContext) -> Self {
        Self::default()
    }

    pub fn max_frames(&self, n_predict: i32) -> usize {
        if n_predict > 0 {
            n_predict as usize
        } else {
            256
        }
    }

    pub fn set_input(&mut self, inp: &GenAudioInput) -> Result<()> {
        if inp.prompt.is_empty() {
            return Err(Error::new("set_input failed: empty prompt"));
        }
        self.input = inp.clone();
        Ok(())
    }

    /// Prompt processing stand-in: returns zero once the prompt is consumed.
    pub fn step_prompt(&mut self, _n_batch: usize) -> Result<i32> {
        Ok(0)
    }

    /// Semantic/acoustic/vocoder stand-in: emits one frame per call.
    pub fn step_gen(
        &mut self,
        _sampled: Token,
        _h_state: &[f32],
        h_next: &mut Vec<f32>,
        stop: &mut bool,
    ) -> Result<()> {
        self.frames += 1;
        h_next.clear();
        h_next.push(self.frames as f32);
        *stop = false;
        Ok(())
    }

    /// Returns the sample rate, WAV bytes and sample count.
    pub fn get_output(&self) -> Result<(i32, Vec<u8>, i64)> {
        let sample_rate = 24_000i32;
        let n_samples = (self.frames * 512) as i64;
        Ok((sample_rate, wav_silence(sample_rate, n_samples), n_samples))
    }
}

fn wav_silence(sample_rate: i32, n_samples: i64) -> Vec<u8> {
    let data_len = (n_samples.max(0) as usize) * 2;
    let mut wav = Vec::with_capacity(44 + data_len);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36u32 + data_len as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&(sample_rate as u32).to_le_bytes());
    wav.extend_from_slice(&((sample_rate as u32) * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_len as u32).to_le_bytes());
    wav.resize(44 + data_len, 0);
    wav
}

pub fn tts_run(args: &[String]) -> Result<()> {
    let Some(mut params) = tts_params_parse(args)? else {
        return Ok(());
    };
    if params.prompt.is_empty() {
        return Err(Error::new("no prompt provided, use -p \"text\""));
    }
    if params.mmproj.is_empty() {
        return Err(Error::new("no mmproj provided, use --mmproj"));
    }
    // Keep this file generic; model-specific logic belongs to mtmd-helper-gen.
    params.embedding = true;

    // llama_backend_init / llama_numa_init are no-ops in the dry port.
    let mctx = MtmdContext::from_file(&params.mmproj, &params.model)?;
    if mctx.gen_audio_get_info().type_ == AudioGenType::None {
        return Err(Error::new("mmproj does not support audio generation"));
    }

    let mut speaker_ref = None;
    if !params.tts_speaker_file.is_empty() {
        let path = PathBuf::from(&params.tts_speaker_file);
        if !path.exists() {
            return Err(Error::new(format!(
                "failed to load speaker file {}",
                params.tts_speaker_file
            )));
        }
        speaker_ref = Some(path);
    }

    if mctx.gen_audio_get_info().n_voices > 0 && (!params.tts_voice.is_empty() || speaker_ref.is_none())
    {
        if speaker_ref.is_some() {
            return Err(Error::new(
                "--tts-voice and --tts-speaker-file cannot be used together",
            ));
        }
        if params.tts_voice.is_empty() {
            params.tts_voice = mctx.voice_name(0).unwrap_or("").to_owned();
            println!("no --tts-voice given, speaking with \"{}\"", params.tts_voice);
        }
        if mctx
            .gen_audio_get_info()
            .voices
            .contains(&params.tts_voice)
        {
            return Err(Error::new(format!(
                "unknown voice \"{}\", the model has no listed voices",
                params.tts_voice
            )));
        }
    }

    // No `tts` chat template is available in this port; fall back to the prompt.
    let mut prompt = String::new();
    for word in params.prompt.split_whitespace() {
        prompt.push_str(&mctx.g2p(word));
        prompt.push(' ');
    }
    let prompt = prompt.trim().to_owned();
    let prompt = if prompt.is_empty() {
        params.prompt.clone()
    } else {
        prompt
    };

    let mut gen = GenAudio::init(0, &mctx);
    if !gen.valid {
        return Err(Error::new(
            "the context cannot run the audio generation of this model",
        ));
    }
    let inp = GenAudioInput {
        seq_id: 0,
        prompt: prompt.clone(),
        speaker_ref,
        speaker_ref_text: if params.tts_speaker_text.is_empty() {
            None
        } else {
            Some(params.tts_speaker_text.clone())
        },
        lang: params.tts_lang.clone(),
        top_k: params.top_k,
        top_p: params.top_p,
        seed: params.seed,
        temp: params.temp,
    };
    gen.set_input(&inp)?;
    let t_prompt_start = Instant::now();
    loop {
        if gen.step_prompt(512)? == 0 {
            break;
        }
    }

    let max_new = gen.max_frames(params.n_predict);
    let mut n_frames = 0usize;
    let mut sampled = 0 as Token;
    let mut h_state = Vec::new();
    let mut timings = TtsTimings::default();
    let t_gen_start = Instant::now();
    let mut stop = false;
    while !stop && n_frames < max_new {
        let mut h_next = Vec::new();
        gen.step_gen(sampled, &h_state, &mut h_next, &mut stop)?;
        if h_next.is_empty() {
            break;
        }
        n_frames += 1;
        h_state = h_next;
        sampled = n_frames as Token;
        timings.report(n_frames);
    }
    let t_gen = t_gen_start.elapsed().as_secs_f64();
    let t_prompt = t_gen_start.duration_since(t_prompt_start).as_secs_f64();

    let t_wav_start = Instant::now();
    let (sample_rate, data, n_samples) = gen.get_output()?;
    let t_wav = t_wav_start.elapsed().as_secs_f64();

    println!(
        "generated {} frames, {} bytes of WAV audio ({} Hz)",
        n_frames,
        data.len(),
        sample_rate
    );
    println!(
        "timings: prompt eval {:.2}s + generation {:.2}s + vocoder {:.2}s = total {:.2}s",
        t_prompt,
        t_gen,
        t_wav,
        t_prompt + t_gen + t_wav
    );
    let audio_s = if sample_rate > 0 {
        n_samples as f64 / sample_rate as f64
    } else {
        0.0
    };
    let total = t_prompt + t_gen + t_wav;
    println!(
        "         output audio = {:.2}s (audio time = {:.2}x process time)",
        audio_s,
        if total > 0.0 { audio_s / total } else { 0.0 }
    );

    if params.out_file.is_empty() {
        return Err(Error::new("no output file specified, use -o"));
    }
    let mut out = File::create(&params.out_file).map_err(io_error)?;
    out.write_all(&data).map_err(io_error)?;
    println!("wrote {}", params.out_file);
    Ok(())
}

// ===========================================================================
// rpc-server
// ===========================================================================

#[derive(Clone, Debug)]
pub struct RpcServerParams {
    pub host: String,
    pub port: i32,
    pub use_cache: bool,
    pub n_threads: i32,
    pub devices: Vec<String>,
}

impl Default for RpcServerParams {
    fn default() -> Self {
        let half = std::thread::available_parallelism()
            .map(|n| n.get() as i32 / 2)
            .unwrap_or(0);
        Self {
            host: "127.0.0.1".to_owned(),
            port: 50052,
            use_cache: false,
            n_threads: half.max(1),
            devices: Vec::new(),
        }
    }
}

pub fn rpc_print_usage(executable: &str, params: &RpcServerParams) {
    eprintln!("Usage: {} [options]\n", executable);
    eprintln!("options:");
    eprintln!("  -h, --help                       show this help message and exit");
    eprintln!(
        "  -t, --threads N                  number of threads for the CPU device (default: {})",
        params.n_threads
    );
    eprintln!("  -d, --device <dev1,dev2,...>     comma-separated list of devices");
    eprintln!(
        "  -H, --host HOST                  host to bind to (default: {})",
        params.host
    );
    eprintln!("  -p, --port PORT                  port to bind to (default: {})", params.port);
    eprintln!("  -c, --cache                      enable local file cache\n");
}

pub fn rpc_server_params_parse(args: &[String]) -> Result<Option<RpcServerParams>> {
    let mut params = RpcServerParams::default();
    let mut i = 1usize;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-h" | "--help" => {
                rpc_print_usage(prog_name(args, "rpc-server"), &params);
                return Ok(None);
            }
            "-H" | "--host" => params.host = take_value(args, &mut i, arg)?,
            "-t" | "--threads" => {
                let n = parse_i32(&take_value(args, &mut i, arg)?, "number of threads")?;
                if n <= 0 {
                    return Err(Error::new(format!("error: invalid number of threads: {}", n)));
                }
                params.n_threads = n;
            }
            "-d" | "--device" => {
                let value = take_value(args, &mut i, arg)?;
                for dev in value.split(|c| c == ',' || c == '/') {
                    if !dev.is_empty() {
                        params.devices.push(dev.to_owned());
                    }
                }
            }
            "-p" | "--port" => {
                let port = parse_i32(&take_value(args, &mut i, arg)?, "port")?;
                if !(1..=65535).contains(&port) {
                    return Err(Error::new("error: invalid port"));
                }
                params.port = port;
            }
            "-c" | "--cache" => params.use_cache = true,
            _ => {
                eprintln!("error: unknown argument: {}", arg);
                rpc_print_usage(prog_name(args, "rpc-server"), &params);
                return Ok(None);
            }
        }
        i += 1;
    }
    Ok(Some(params))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Accel,
    Gpu,
    Other,
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub description: String,
    pub device_type: DeviceType,
    pub free: usize,
    pub total: usize,
}

/// Stand-in for `ggml_backend_dev_count`: only the CPU backend is enumerated.
pub fn available_devices() -> Vec<DeviceInfo> {
    vec![DeviceInfo {
        name: "CPU".to_owned(),
        description: "CPU backend".to_owned(),
        device_type: DeviceType::Cpu,
        free: 0,
        total: 0,
    }]
}

pub fn get_devices(params: &RpcServerParams) -> Result<Vec<DeviceInfo>> {
    let available = available_devices();
    let mut devices = Vec::new();
    if !params.devices.is_empty() {
        for name in &params.devices {
            match available.iter().find(|dev| &dev.name == name) {
                Some(dev) => devices.push(dev.clone()),
                None => {
                    eprintln!("error: unknown device: {}", name);
                    eprintln!("available devices:");
                    for dev in &available {
                        eprintln!(
                            "  {}: {} ({} MiB, {} MiB free)",
                            dev.name, dev.description, dev.total / 1024 / 1024, dev.free / 1024 / 1024
                        );
                    }
                    return Err(Error::new(format!("unknown device: {}", name)));
                }
            }
        }
    }
    if devices.is_empty() {
        for dev in &available {
            if dev.device_type != DeviceType::Cpu && dev.device_type != DeviceType::Accel {
                devices.push(dev.clone());
            }
        }
    }
    if devices.is_empty() {
        if let Some(dev) = available.iter().find(|dev| dev.device_type == DeviceType::Cpu) {
            devices.push(dev.clone());
        }
    }
    Ok(devices)
}

/// `fs_get_cache_directory` with `HOME` / `XDG_CACHE_HOME` fallbacks.
pub fn rpc_cache_directory() -> Result<PathBuf> {
    let mut base = if let Ok(dir) = std::env::var("LLAMA_CACHE") {
        if !dir.is_empty() {
            PathBuf::from(dir)
        } else {
            PathBuf::new()
        }
    } else {
        PathBuf::new()
    };
    if base.as_os_str().is_empty() {
        if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
            if !dir.is_empty() {
                base = PathBuf::from(dir);
            }
        }
    }
    if base.as_os_str().is_empty() {
        let home = std::env::var("HOME")
            .map_err(|_| Error::new("Failed to find $HOME directory"))?;
        base = Path::new(&home).join(".cache");
    }
    Ok(base.join("llama.cpp").join("rpc"))
}

/// Stand-in for `ggml_backend_rpc_start_server`: validates and reports the
/// endpoint; no socket is opened in this port.
pub fn start_rpc_server(params: &RpcServerParams) -> Result<()> {
    let devices = get_devices(params)?;
    if devices.is_empty() {
        return Err(Error::new("No devices found"));
    }
    let endpoint = format!("{}:{}", params.host, params.port);
    let cache_dir = if params.use_cache {
        let dir = rpc_cache_directory()?;
        fs::create_dir_all(&dir).map_err(|err| {
            Error::new(format!(
                "Failed to create cache directory: {} ({})",
                dir.display(),
                err
            ))
        })?;
        Some(dir)
    } else {
        None
    };
    eprintln!(
        "rpc-server: listening on {} with {} device(s) and {} thread(s)",
        endpoint,
        devices.len(),
        params.n_threads
    );
    if let Some(dir) = cache_dir {
        eprintln!("rpc-server: cache directory {}", dir.display());
    }
    Ok(())
}

pub fn rpc_server_run(args: &[String]) -> Result<()> {
    let Some(params) = rpc_server_params_parse(args)? else {
        return Ok(());
    };
    if params.host != "127.0.0.1" {
        eprintln!();
        eprintln!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        eprintln!("WARNING: Host ('{}') is != '127.0.0.1'", params.host);
        eprintln!("         Never expose the RPC server to an open network!");
        eprintln!("         This is an experimental feature and is not secure!");
        eprintln!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        eprintln!();
    }
    start_rpc_server(&params)
}

// ===========================================================================
// cvector-generator
// ===========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DimreMethod {
    Pca,
    Mean,
}

#[derive(Clone, Debug)]
pub struct CvectorParams {
    pub model: String,
    pub out_file: String,
    pub positive_file: String,
    pub negative_file: String,
    pub method: DimreMethod,
    pub n_pca_iterations: usize,
    pub n_pca_batch: usize,
    pub ngl: i32,
    pub n_threads: usize,
}

impl Default for CvectorParams {
    fn default() -> Self {
        Self {
            model: String::new(),
            out_file: "control_vector.gguf".to_owned(),
            positive_file: String::new(),
            negative_file: String::new(),
            method: DimreMethod::Pca,
            n_pca_iterations: 1000,
            n_pca_batch: 20,
            ngl: 0,
            n_threads: 1,
        }
    }
}

pub fn cvector_print_usage(executable: &str) {
    println!("\nexample usage:\n");
    println!("    CPU only:   {} -m ./llama-3.Q4_K_M.gguf\n", executable);
    println!("    with GPU:   {} -m ./llama-3.Q4_K_M.gguf -ngl 99\n", executable);
    println!(
        "    advanced:   {} -m ./llama-3.Q4_K_M.gguf -ngl 99 --pca-iter 2000 --pca-batch 100\n",
        executable
    );
    println!(
        "    using mean: {} -m ./llama-3.Q4_K_M.gguf --method mean\n",
        executable
    );
    println!();
}

pub fn cvector_params_parse(args: &[String]) -> Result<Option<CvectorParams>> {
    let mut params = CvectorParams::default();
    let mut i = 1usize;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-h" | "--help" => {
                cvector_print_usage(prog_name(args, "cvector-generator"));
                return Ok(None);
            }
            "-m" | "--model" => params.model = take_value(args, &mut i, arg)?,
            "-o" | "--out" => params.out_file = take_value(args, &mut i, arg)?,
            "--positive" | "--positive-file" => {
                params.positive_file = take_value(args, &mut i, arg)?
            }
            "--negative" | "--negative-file" => {
                params.negative_file = take_value(args, &mut i, arg)?
            }
            "--method" => {
                let value = take_value(args, &mut i, arg)?;
                params.method = match value.as_str() {
                    "pca" => DimreMethod::Pca,
                    "mean" => DimreMethod::Mean,
                    _ => return Err(Error::new("error: unknown dimre method")),
                };
            }
            "--pca-iter" | "--pca-iterations" => {
                params.n_pca_iterations = take_value(args, &mut i, arg)?
                    .parse::<usize>()
                    .map_err(|_| Error::new("error: invalid PCA iterations"))?;
            }
            "--pca-batch" => {
                params.n_pca_batch = take_value(args, &mut i, arg)?
                    .parse::<usize>()
                    .map_err(|_| Error::new("error: invalid PCA batch size"))?;
            }
            "-ngl" | "--gpu-layers" => {
                params.ngl = parse_i32(&take_value(args, &mut i, arg)?, "n_gpu_layers")?
            }
            "-t" | "--threads" => {
                params.n_threads = parse_i32(&take_value(args, &mut i, arg)?, "threads")?
                    .max(1) as usize;
            }
            _ => return Err(Error::new(format!("error: unknown argument: {}", arg))),
        }
        i += 1;
    }
    Ok(Some(params))
}

/// Read a prompt file; blank lines are skipped when requested and escapes are
/// applied to every retained line, as `ctrlvec_load_prompt_file` does.
pub fn ctrlvec_load_prompt_file(path: &str, skip_empty_lines: bool) -> Result<Vec<String>> {
    let text = fs::read_to_string(path)
        .map_err(|err| Error::new(format!("error: unable to open file: {} ({})", path, err)))?;
    let mut output = Vec::new();
    for mut line in text.lines().map(str::to_owned) {
        if skip_empty_lines && line.is_empty() {
            continue;
        }
        crate::common::string_process_escapes(&mut line);
        output.push(line);
    }
    Ok(output)
}

#[derive(Clone, Debug)]
pub struct TokenizedPrompt {
    pub tokens_pos: Vec<Token>,
    pub tokens_neg: Vec<Token>,
    pub max_seq_len: usize,
}

impl TokenizedPrompt {
    pub fn new(
        vocab: &crate::vocab::LlamaVocab,
        pos: &str,
        neg: &str,
    ) -> Result<Self> {
        let add_bos = vocab.get_add_bos();
        let mut tokens_pos = crate::common::common_tokenize(vocab, pos, add_bos, true)?;
        let mut tokens_neg = crate::common::common_tokenize(vocab, neg, add_bos, true)?;
        let max_seq_len = tokens_pos.len().max(tokens_neg.len());
        let pad_tokens = crate::common::common_tokenize(vocab, " ", false, true)?;
        let pad_tok = pad_tokens.last().copied().unwrap_or(0);
        while tokens_pos.len() < max_seq_len {
            tokens_pos.push(pad_tok);
        }
        while tokens_neg.len() < max_seq_len {
            tokens_neg.push(pad_tok);
        }
        Ok(Self {
            tokens_pos,
            tokens_neg,
            max_seq_len,
        })
    }
}

/// Captures the hidden state of every layer for one positive/negative pair.
#[derive(Clone, Debug, Default)]
pub struct HiddenCapture {
    pub is_eval_pos: bool,
    pub n_layers: usize,
    pub n_tokens: usize,
    pub v_pos: Vec<Vec<f32>>,
    pub v_neg: Vec<Vec<f32>>,
    pub v_diff_filtered: Vec<Vec<f32>>,
}

impl HiddenCapture {
    pub fn save_tensor_for_layer(&mut self, rows: Vec<f32>) {
        if self.is_eval_pos {
            self.v_pos.push(rows);
        } else {
            self.v_neg.push(rows);
        }
    }

    /// Subtract negative from positive and drop all-zero rows.
    pub fn calc_diff(&mut self) -> Result<Vec<Vec<f32>>> {
        let n_layers = self.v_pos.len().min(self.v_neg.len());
        for il in 0..n_layers {
            let n_embd = self.n_embd();
            for j in 0..self.v_pos[il].len().min(self.v_neg[il].len()) {
                self.v_pos[il][j] -= self.v_neg[il][j];
            }
            let filtered = Self::filter_nonzero_rows(&self.v_pos[il], n_embd)?;
            self.v_diff_filtered.push(filtered);
        }
        Ok(self.v_diff_filtered.clone())
    }

    fn n_embd(&self) -> usize {
        self.v_pos.first().map_or(0, |layer| {
            if self.n_tokens > 0 {
                layer.len() / self.n_tokens
            } else {
                0
            }
        })
    }

    /// Delete rows whose elements are all <= 1e-6, as `filter_nonzero_rows`.
    pub fn filter_nonzero_rows(rows: &[f32], n_embd: usize) -> Result<Vec<f32>> {
        if n_embd == 0 {
            return Err(Error::new("filter_nonzero_rows: zero embedding size"));
        }
        let mut kept = Vec::new();
        for row in rows.chunks(n_embd) {
            if row.iter().any(|&value| value > 1e-6) {
                kept.extend_from_slice(row);
            }
        }
        if kept.is_empty() {
            return Err(Error::new(
                "filter_nonzero_rows: no non-zero rows in diff tensor",
            ));
        }
        Ok(kept)
    }

    pub fn reset(&mut self) {
        self.v_pos.clear();
        self.v_neg.clear();
        self.v_diff_filtered.clear();
    }
}

/// Row-major matrix. `ne0` is the fast dimension, matching ggml's `ne[0]`.
#[derive(Clone, Debug, Default)]
pub struct Matrix {
    pub ne0: usize,
    pub ne1: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(ne0: usize, ne1: usize, data: Vec<f32>) -> Self {
        Self { ne0, ne1, data }
    }

    pub fn get(&self, col: usize, row: usize) -> f32 {
        self.data[row * self.ne0 + col]
    }

    pub fn set(&mut self, col: usize, row: usize, value: f32) {
        self.data[row * self.ne0 + col] = value;
    }

    pub fn transpose(&self) -> Self {
        let mut out = Self::new(self.ne1, self.ne0, vec![0.0; self.data.len()]);
        for row in 0..self.ne1 {
            for col in 0..self.ne0 {
                out.set(row, col, self.get(col, row));
            }
        }
        out
    }
}

/// `ggml_mul_mat(a, b)`: result `[a.ne1, b.ne1]`, each entry a dot product of
/// the two fast-dimension vectors (a is used as-is, b is transposed).
pub fn mul_mat(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    if a.ne0 != b.ne0 {
        return Err(Error::new("mul_mat: mismatched inner dimensions"));
    }
    let mut out = Matrix::new(a.ne1, b.ne1, vec![0.0; a.ne1 * b.ne1]);
    for j in 0..b.ne1 {
        for i in 0..a.ne1 {
            let mut sum = 0.0f32;
            for k in 0..a.ne0 {
                sum += a.get(k, i) * b.get(k, j);
            }
            out.set(i, j, sum);
        }
    }
    Ok(out)
}

/// `train_context` from cvector-generator.cpp.
#[derive(Clone, Debug)]
pub struct TrainContext {
    pub n_embd: usize,
    pub n_layers: usize,
    pub positive_entries: Vec<String>,
    pub negative_entries: Vec<String>,
    pub v_diff_tmp: Vec<Vec<f32>>,
    pub v_diff: Vec<Matrix>,
    pub v_final: Vec<Vec<f32>>,
}

impl TrainContext {
    pub fn new(n_embd: usize, n_layers: usize) -> Self {
        Self {
            n_embd,
            n_layers,
            positive_entries: Vec::new(),
            negative_entries: Vec::new(),
            v_diff_tmp: vec![Vec::new(); n_layers.saturating_sub(1)],
            v_diff: Vec::new(),
            v_final: vec![vec![0.0; n_embd]; n_layers.saturating_sub(1)],
        }
    }

    pub fn concat_diff_tmp(&mut self, diff_filtered: &[Vec<f32>]) -> Result<()> {
        if diff_filtered.len() != self.n_layers.saturating_sub(1) {
            return Err(Error::new(
                "concat_diff_tmp: wrong number of layers",
            ));
        }
        for (il, diff) in diff_filtered.iter().enumerate() {
            self.v_diff_tmp[il].extend_from_slice(diff);
        }
        Ok(())
    }

    /// Build the per-layer matrices. `transpose` selects PCA layout
    /// (`[n_rows, n_embd]`) over mean layout (`[n_embd, n_rows]`).
    pub fn build_v_diff(&mut self, transpose: bool) -> Result<()> {
        for il in 0..self.n_layers.saturating_sub(1) {
            let tmp = std::mem::take(&mut self.v_diff_tmp[il]);
            if tmp.len() % self.n_embd != 0 {
                return Err(Error::new("build_v_diff: misaligned diff data"));
            }
            let n_rows = tmp.len() / self.n_embd;
            let matrix = if transpose {
                // PCA input is [n_samples, n_embd] in ggml indexing.
                let mut matrix = Matrix::new(n_rows, self.n_embd, vec![0.0; tmp.len()]);
                for row in 0..n_rows {
                    for col in 0..self.n_embd {
                        matrix.set(row, col, tmp[row * self.n_embd + col]);
                    }
                }
                matrix
            } else {
                Matrix::new(self.n_embd, n_rows, tmp)
            };
            println!("build_v_diff");
            print_debug_matrix(&matrix, "diff");
            self.v_diff.push(matrix);
        }
        Ok(())
    }
}

fn print_debug_matrix(matrix: &Matrix, name: &str) {
    let limit = matrix.ne0.min(6);
    print!("print_debug_tensor: {}[0] = [", name);
    for i in 0..limit {
        print!(" {},", matrix.data[i]);
    }
    println!(" ... ]");
}

#[derive(Clone, Debug)]
pub struct PcaParams {
    pub n_threads: usize,
    pub n_batch: usize,
    pub n_iterations: usize,
    pub tolerance: f32,
    pub i_layer: usize,
    pub n_layers: usize,
}

impl Default for PcaParams {
    fn default() -> Self {
        Self {
            n_threads: 1,
            n_batch: 20,
            n_iterations: 1000,
            tolerance: 1e-7,
            i_layer: 0,
            n_layers: 0,
        }
    }
}

fn random_unit_vector(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut vec = Vec::with_capacity(n);
    let mut sum_sqr = 0.0f32;
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let value = ((state >> 33) as f32) / ((1u64 << 31) as f32);
        sum_sqr += value * value;
        vec.push(value);
    }
    let norm = sum_sqr.sqrt().max(f32::EPSILON);
    for value in &mut vec {
        *value /= norm;
    }
    vec
}

fn mat_vec(matrix: &Matrix, vec: &[f32]) -> Result<Vec<f32>> {
    if vec.len() != matrix.ne0 {
        return Err(Error::new("mat_vec: dimension mismatch"));
    }
    let mut out = vec![0.0f32; matrix.ne1];
    for row in 0..matrix.ne1 {
        let mut sum = 0.0f32;
        for col in 0..matrix.ne0 {
            sum += matrix.get(col, row) * vec[col];
        }
        out[row] = sum;
    }
    Ok(out)
}

/// Power iteration on `X^T X`, equivalent to `PCA::power_iteration`.
pub fn power_iteration(params: &PcaParams, input: &Matrix, output: &mut [f32]) -> Result<()> {
    // dev_square = ggml_mul_mat(dev_input, dev_input), computed once as in the
    // first batch. With ggml's indexing this is the embedding covariance.
    let square = mul_mat(input, input)?;
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1);
    let mut eigen = random_unit_vector(square.ne0, seed);
    let n_iters = params.n_iterations / params.n_batch.max(1);

    for iter in 0..n_iters {
        let mut last_distance = 0.0f32;
        for _ in 0..params.n_batch {
            let mut next = mat_vec(&square, &eigen)?;
            let norm = next.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm <= f32::EPSILON {
                return Err(Error::new("power_iteration: zero eigenvector norm"));
            }
            for value in &mut next {
                *value /= norm;
            }
            let distance = next
                .iter()
                .zip(eigen.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            eigen = next;
            last_distance = distance;
            if distance < params.tolerance {
                break;
            }
        }
        println!(
            "power_iteration: layer {}/{}, iteration: {} / total: {} (batch = {}) ...",
            params.i_layer + 1,
            params.n_layers,
            iter + 1,
            n_iters,
            params.n_batch
        );
        if last_distance < params.tolerance {
            break;
        }
    }
    let n = output.len().min(eigen.len());
    output[..n].copy_from_slice(&eigen[..n]);
    Ok(())
}

pub fn run_pca(params: &mut PcaParams, inputs: &[Matrix], outputs: &mut [Vec<f32>]) -> Result<()> {
    println!("run_pca: Running PCA...");
    for (il, input) in inputs.iter().enumerate() {
        params.i_layer = il;
        params.n_layers = inputs.len();
        power_iteration(params, input, &mut outputs[il])?;
        println!("run_pca: Done layer {} / {}", il + 1, inputs.len());
    }
    Ok(())
}

/// `mean::run`: average over rows, then normalize to unit length.
pub fn run_mean(inputs: &[Matrix], outputs: &mut [Vec<f32>]) -> Result<()> {
    println!("run: Running mean...");
    for (il, input) in inputs.iter().enumerate() {
        let out = &mut outputs[il];
        if out.len() < input.ne0 {
            return Err(Error::new("run_mean: output vector too small"));
        }
        for col in 0..input.ne0 {
            let mut sum = 0.0f32;
            for row in 0..input.ne1 {
                sum += input.get(col, row);
            }
            out[col] = if input.ne1 > 0 {
                sum / input.ne1 as f32
            } else {
                0.0
            };
        }
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        for value in out.iter_mut() {
            *value /= norm;
        }
        println!("run: Done layer {} / {}", il + 1, inputs.len());
    }
    Ok(())
}

/// `export_gguf`: write the control vectors as a GGUF with F32 tensors.
pub fn export_gguf(v_ctrl: &[Vec<f32>], fname: &str, model_hint: &str) -> Result<()> {
    let arch = "controlvector";
    let mut ctx = GgufContext::default();
    gguf_set_val_str(&mut ctx, KEY_GENERAL_ARCH, arch);
    gguf_set_val_str(&mut ctx, &format!("{}.model_hint", arch), model_hint);
    gguf_set_val_i32(
        &mut ctx,
        &format!("{}.layer_count", arch),
        v_ctrl.len() as i32,
    );

    for (i, values) in v_ctrl.iter().enumerate() {
        let n_embd = values.len();
        let mut tensor = GgmlTensor::default();
        tensor.name = format!("direction.{}", i + 1);
        tensor.ne = [n_embd as i64, 1, 1, 1];
        tensor.type_ = GgmlType::F32;
        tensor.nb = [4, 4 * n_embd, 4 * n_embd, 4 * n_embd];
        for &value in values {
            tensor.data.extend_from_slice(&value.to_ne_bytes());
        }
        gguf_add_tensor(&mut ctx, &tensor);
        print_debug_matrix(
            &Matrix::new(n_embd, 1, values.clone()),
            &tensor.name,
        );
        println!("Added tensor: {}", tensor.name);
    }

    println!("export_gguf: writing file...");
    if !gguf_write_to_file(&ctx, fname, false) {
        return Err(Error::new(format!("failed to write GGUF file {}", fname)));
    }
    println!("export_gguf: wrote file '{}'", fname);
    Ok(())
}

/// Stand-in for hidden states from the backbone model. Real inference is not
/// available in the dry port; the values are deterministic so the vector math
/// still exercises the original code path.
fn placeholder_hidden(
    tokens: &[Token],
    n_embd: usize,
    n_layers: usize,
    is_pos: bool,
) -> Vec<Vec<f32>> {
    (0..n_layers)
        .map(|il| {
            tokens
                .iter()
                .enumerate()
                .map(|(it, &token)| {
                    let base = (token as f32 + 1.0) * ((il + 1) as f32) / ((n_layers + 1) as f32);
                    let wobble = 1.0 + 0.01 * (((it + il) % 7) as f32);
                    let value = base * wobble / (n_embd.max(1) as f32);
                    if is_pos { value } else { value * 0.5 }
                })
                .collect()
        })
        .collect()
}

pub fn cvector_run(args: &[String]) -> Result<()> {
    let Some(params) = cvector_params_parse(args)? else {
        return Ok(());
    };
    if params.n_pca_batch == 0 || params.n_pca_iterations % params.n_pca_batch != 0 {
        return Err(Error::new("PCA iterations must by multiply of PCA batch size"));
    }
    if params.model.is_empty() {
        return Err(Error::new("error: no model file specified"));
    }

    let (gguf, _) = load_gguf(&params.model, false)?;
    let n_layers = kv_u32(&gguf, KEY_BLOCK_COUNT).unwrap_or(0) as usize;
    let arch = kv_str(&gguf, KEY_GENERAL_ARCH).unwrap_or_default();
    let n_embd = kv_u32(&gguf, &format!("{}.embedding_length", arch)).unwrap_or(0) as usize;
    if n_layers < 2 || n_embd == 0 {
        return Err(Error::new("model has no usable layer/embedding metadata"));
    }
    let model_hint = arch;
    let loader = crate::model::LlamaModelLoader::from_gguf(
        gguf,
        &crate::model::ModelParams::default(),
    )?;
    let mut vocab = crate::vocab::LlamaVocab::new();
    vocab.load(&loader.vocab_metadata())?;

    let pos_prompts = ctrlvec_load_prompt_file(&params.positive_file, true)?;
    let neg_prompts = ctrlvec_load_prompt_file(&params.negative_file, true)?;
    if pos_prompts.len() != neg_prompts.len() {
        return Err(Error::new("number of positive and negative prompts must be equal"));
    }
    if pos_prompts.is_empty() {
        return Err(Error::new("must provide at least one prompt pair"));
    }

    let mut tokenized = Vec::new();
    let mut n_total_tokens = 0usize;
    for (pos, neg) in pos_prompts.iter().zip(neg_prompts.iter()) {
        let item = TokenizedPrompt::new(&vocab, pos, neg)?;
        n_total_tokens += 2 * item.max_seq_len;
        tokenized.push(item);
    }
    println!("n_total_tokens: {}", n_total_tokens);

    let mut cb_data = HiddenCapture::default();
    let mut ctx_train = TrainContext::new(n_embd, n_layers);
    ctx_train.positive_entries = pos_prompts;
    ctx_train.negative_entries = neg_prompts;

    for (i, item) in tokenized.iter().enumerate() {
        cb_data.n_layers = n_layers;
        cb_data.n_tokens = item.max_seq_len;
        println!(
            "Evaluating prompt[{}/{}]: \"{}\" - \"{}\" ({} tokens)",
            i + 1,
            tokenized.len(),
            ctx_train.positive_entries[i],
            ctx_train.negative_entries[i],
            item.max_seq_len
        );
        cb_data.is_eval_pos = true;
        for rows in placeholder_hidden(&item.tokens_pos, n_embd, n_layers, true) {
            cb_data.save_tensor_for_layer(rows);
        }
        cb_data.is_eval_pos = false;
        for rows in placeholder_hidden(&item.tokens_neg, n_embd, n_layers, false) {
            cb_data.save_tensor_for_layer(rows);
        }
        let diff = cb_data.calc_diff()?;
        ctx_train.concat_diff_tmp(&diff)?;
        cb_data.reset();
    }

    println!("Done evaluate prompts, unload model...");
    let use_pca = params.method == DimreMethod::Pca;
    ctx_train.build_v_diff(use_pca)?;

    if use_pca {
        let mut pca_params = PcaParams {
            n_threads: params.n_threads,
            n_batch: params.n_pca_batch,
            n_iterations: params.n_pca_iterations,
            ..Default::default()
        };
        run_pca(&mut pca_params, &ctx_train.v_diff, &mut ctx_train.v_final)?;
    } else {
        run_mean(&ctx_train.v_diff, &mut ctx_train.v_final)?;
    }

    export_gguf(&ctx_train.v_final, &params.out_file, &model_hint)
}

// ===========================================================================
// export-lora
// ===========================================================================

#[derive(Clone, Debug)]
pub struct LoraInfo {
    pub path: String,
    pub scale: f32,
}

#[derive(Clone, Debug)]
pub struct ExportLoraParams {
    pub model: String,
    pub out_file: String,
    pub lora_adapters: Vec<LoraInfo>,
    pub n_threads: usize,
    pub verbose: bool,
}

impl Default for ExportLoraParams {
    fn default() -> Self {
        Self {
            model: String::new(),
            out_file: "ggml-lora-merged-f16.gguf".to_owned(),
            lora_adapters: Vec::new(),
            n_threads: 1,
            verbose: false,
        }
    }
}

pub fn export_lora_print_usage(executable: &str) {
    println!("\nexample usage:\n");
    println!(
        "  {} -m base-model.gguf --lora lora-file.gguf -o merged-model.gguf\n",
        executable
    );
}

pub fn export_lora_params_parse(args: &[String]) -> Result<Option<ExportLoraParams>> {
    let mut params = ExportLoraParams::default();
    let mut i = 1usize;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-h" | "--help" => {
                export_lora_print_usage(prog_name(args, "export-lora"));
                return Ok(None);
            }
            "-m" | "--model" => params.model = take_value(args, &mut i, arg)?,
            "-o" | "--out" => params.out_file = take_value(args, &mut i, arg)?,
            "--lora" => {
                let path = take_value(args, &mut i, arg)?;
                params.lora_adapters.push(LoraInfo { path, scale: 1.0 });
            }
            "--lora-scaled" => {
                let path = take_value(args, &mut i, arg)?;
                let scale = take_value(args, &mut i, arg)?
                    .parse::<f32>()
                    .map_err(|_| Error::new("error: invalid LoRA scale"))?;
                params.lora_adapters.push(LoraInfo { path, scale });
            }
            "-t" | "--threads" => {
                params.n_threads = parse_i32(&take_value(args, &mut i, arg)?, "threads")?
                    .max(1) as usize;
            }
            "-v" | "--verbose" => params.verbose = true,
            _ => return Err(Error::new(format!("error: unknown argument: {}", arg))),
        }
        i += 1;
    }
    Ok(Some(params))
}

/// `file_input`: owns a parsed GGUF and all of its tensor data.
#[derive(Debug)]
pub struct FileInput {
    pub path: String,
    pub ctx: GgufContext,
    pub meta: GgmlContext,
    pub tensors: BTreeMap<String, GgmlTensor>,
    pub alpha: f32,
    pub scale: f32,
}

impl FileInput {
    pub fn new(path: &str, scale: f32) -> Result<Self> {
        let (ctx, meta) = load_gguf(path, true)?;
        let alpha = kv_f32(&ctx, KEY_ADAPTER_LORA_ALPHA);
        let mut tensors = BTreeMap::new();
        for info in &ctx.info {
            tensors.insert(info.t.name.clone(), info.t.clone());
        }
        Ok(Self {
            path: path.to_owned(),
            ctx,
            meta,
            tensors,
            alpha,
            scale,
        })
    }

    pub fn get_tensor(&self, name: &str) -> Option<&GgmlTensor> {
        self.tensors.get(name)
    }

    pub fn raw_tensor_data(&self, name: &str) -> Result<&[u8]> {
        let id = gguf_find_tensor(&self.ctx, name);
        if id < 0 {
            return Err(Error::new(format!("cannot find tensor with name: {}", name)));
        }
        self.meta
            .tensors
            .get(id as usize)
            .map(|t| t.data.as_slice())
            .ok_or_else(|| Error::new(format!("missing data for tensor {}", name)))
    }

    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .get_tensor(name)
            .ok_or_else(|| Error::new(format!("cannot find tensor with name: {}", name)))?;
        tensor_to_f32(tensor.type_, self.raw_tensor_data(name)?)
    }
}

fn tensor_to_f32(ty: GgmlType, raw: &[u8]) -> Result<Vec<f32>> {
    match ty {
        GgmlType::F32 => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        GgmlType::F16 => Ok(raw
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_ne_bytes([c[0], c[1]])))
            .collect()),
        GgmlType::Bf16 => Ok(raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_ne_bytes([c[0], c[1]])))
            .collect()),
        _ => {
            let qtype = QType::from_i32(ty as i32)
                .ok_or_else(|| Error::new(format!("unsupported tensor type {}", ty.name())))?;
            let block = qtype.blck_size().max(1) as usize;
            let type_size = qtype.type_size().max(1);
            if type_size == 0 || raw.len() % type_size != 0 {
                return Err(Error::new("misaligned quantized tensor data"));
            }
            let n = raw.len() / type_size * block;
            let mut out = vec![0.0f32; n];
            dequantize_slab(&NativeQuantizer, qtype, raw, &mut out, 1)?;
            Ok(out)
        }
    }
}

fn tensor_from_f32(ty: GgmlType, values: &[f32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(values.len() * ty.type_size());
    for &value in values {
        match ty {
            GgmlType::F32 => out.extend_from_slice(&value.to_ne_bytes()),
            GgmlType::F16 => out.extend_from_slice(&f32_to_f16(value).to_ne_bytes()),
            GgmlType::Bf16 => out.extend_from_slice(&f32_to_bf16(value).to_ne_bytes()),
            _ => {
                return Err(Error::new(format!(
                    "unsupported output tensor type {}",
                    ty.name()
                )))
            }
        }
    }
    Ok(out)
}

fn out_tensor_type(ty: GgmlType) -> GgmlType {
    match ty {
        GgmlType::F32 => GgmlType::F32,
        GgmlType::Bf16 => GgmlType::Bf16,
        _ => GgmlType::F16,
    }
}

fn matrix_from_tensor(tensor: &GgmlTensor, values: &[f32]) -> Result<Matrix> {
    let ne0 = tensor.ne[0].max(0) as usize;
    let ne1 = tensor.ne[1].max(0) as usize;
    if tensor.ne[2] != 1 || tensor.ne[3] != 1 {
        return Err(Error::new(
            "only 2D tensors are supported by the LoRA merge port",
        ));
    }
    if values.len() < ne0 * ne1 {
        return Err(Error::new(format!(
            "tensor {} has too few elements",
            tensor.name
        )));
    }
    Ok(Matrix::new(ne0, ne1, values[..ne0 * ne1].to_vec()))
}

fn ne_string(tensor: &GgmlTensor) -> String {
    tensor
        .ne
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

pub struct LoraMergeCtx {
    pub base: FileInput,
    pub adapters: Vec<FileInput>,
    pub out: GgufContext,
}

impl LoraMergeCtx {
    pub fn new(
        base_fname: &str,
        lora_files: &[LoraInfo],
        _outfile: &str,
        _n_threads: usize,
    ) -> Result<Self> {
        let base = FileInput::new(base_fname, 0.0)?;
        if gguf_find_key(&base.ctx, KEY_SPLIT_COUNT) >= 0 {
            return Err(Error::new("split model is not yet supported"));
        }
        let mut adapters = Vec::new();
        for lora in lora_files {
            let adapter = FileInput::new(&lora.path, lora.scale)?;
            Self::check_metadata_lora(&base, &adapter)?;
            adapters.push(adapter);
        }
        Ok(Self {
            base,
            adapters,
            out: GgufContext::default(),
        })
    }

    fn check_metadata_lora(base: &FileInput, adapter: &FileInput) -> Result<()> {
        let general_type = kv_str(&adapter.ctx, KEY_GENERAL_TYPE).unwrap_or_default();
        if general_type != "adapter" {
            return Err(Error::new(format!(
                "expect general.type to be 'adapter', but got: {}",
                general_type
            )));
        }
        let adapter_type = kv_str(&adapter.ctx, KEY_ADAPTER_TYPE).unwrap_or_default();
        if adapter_type != "lora" {
            return Err(Error::new(format!(
                "expect adapter.type to be 'lora', but got: {}",
                adapter_type
            )));
        }
        let arch_base = kv_str(&base.ctx, KEY_GENERAL_ARCH).unwrap_or_default();
        let arch_lora = kv_str(&adapter.ctx, KEY_GENERAL_ARCH).unwrap_or_default();
        if arch_base != arch_lora {
            return Err(Error::new("model arch and LoRA arch mismatch"));
        }
        Ok(())
    }

    pub fn run_merge(&mut self, outfile: &str, verbose: bool) -> Result<()> {
        gguf_set_kv(&mut self.out, &self.base.ctx);
        gguf_set_val_u32(&mut self.out, KEY_GENERAL_FILE_TYPE, LLAMA_FTYPE_MOSTLY_F16);

        if self.adapters.len() > 1 {
            for adapter in self.adapters.iter().skip(1) {
                if adapter.tensors.len() != self.adapters[0].tensors.len() {
                    return Err(Error::new(
                        "Input adapters do not have the same list of tensors. This is not yet supported. Please merge the adapter one-by-one instead of merging all at once.",
                    ));
                }
                for name in adapter.tensors.keys() {
                    if self.adapters[0].get_tensor(name).is_none() {
                        return Err(Error::new(
                            "Input adapters do not have the same list of tensors. This is not yet supported. Please merge the adapter one-by-one instead of merging all at once.",
                        ));
                    }
                }
            }
        }

        let mut n_merged = 0usize;
        for name in self.base.tensors.keys() {
            let mut has_a = true;
            let mut has_b = true;
            for adapter in &self.adapters {
                has_a &= adapter
                    .get_tensor(&format!("{}.lora_a", name))
                    .is_some();
                has_b &= adapter
                    .get_tensor(&format!("{}.lora_b", name))
                    .is_some();
            }
            let base_tensor = self.base.get_tensor(name).expect("base tensor").clone();
            let mut out_tensor = base_tensor.clone();
            if !has_a && !has_b {
                out_tensor.type_ = out_tensor_type(base_tensor.type_);
                if out_tensor.type_ == base_tensor.type_ {
                    out_tensor.data = self.base.raw_tensor_data(name)?.to_vec();
                } else {
                    let values =
                        tensor_to_f32(base_tensor.type_, self.base.raw_tensor_data(name)?)?;
                    out_tensor.data = tensor_from_f32(out_tensor.type_, &values)?;
                }
                if verbose {
                    println!("copy_tensor : {}", name);
                }
            } else if has_a && has_b {
                out_tensor.type_ = out_tensor_type(base_tensor.type_);
                out_tensor.data =
                    merge_tensor(&self.base, &self.adapters, name, out_tensor.type_)?;
                n_merged += 1;
            } else {
                return Err(Error::new(format!(
                    "tensor {} missing either lora_a or lora_b",
                    name
                )));
            }
            if verbose {
                println!("load_gguf: {} [{}]", name, ne_string(&out_tensor));
            }
            gguf_add_tensor(&mut self.out, &out_tensor);
        }

        let mut fout = File::create(outfile).map_err(io_error)?;
        fout.write_all(&gguf_get_meta_data(&self.out))
            .map_err(io_error)?;
        for info in &self.out.info {
            let len = ggml_nbytes(&info.t);
            if info.t.data.len() < len {
                return Err(Error::new(format!(
                    "tensor {} has too little output data",
                    info.t.name
                )));
            }
            fout.write_all(&info.t.data[..len]).map_err(io_error)?;
            write_zeros(&mut fout, gguf_pad(len, GGUF_DEFAULT_ALIGNMENT) - len)?;
        }
        fout.seek(SeekFrom::Start(0)).map_err(io_error)?;
        fout.write_all(&gguf_get_meta_data(&self.out))
            .map_err(io_error)?;

        println!("run_merge : merged {} tensors with lora adapters", n_merged);
        println!(
            "run_merge : wrote {} tensors to output file",
            self.out.info.len()
        );
        Ok(())
    }
}

pub fn export_lora_run(args: &[String]) -> Result<()> {
    let Some(params) = export_lora_params_parse(args)? else {
        return Ok(());
    };
    if params.model.is_empty() || params.lora_adapters.is_empty() {
        return Err(Error::new("error: -m and --lora are required"));
    }

    let mut ctx = LoraMergeCtx::new(
        &params.model,
        &params.lora_adapters,
        &params.out_file,
        params.n_threads,
    )?;
    ctx.run_merge(&params.out_file, params.verbose)?;
    println!("done, output file is {}", params.out_file);
    Ok(())
}

fn merge_tensor(
    base: &FileInput,
    adapters: &[FileInput],
    name: &str,
    out_type: GgmlType,
) -> Result<Vec<u8>> {
    let base_tensor = base.get_tensor(name).expect("base tensor");
    let base_values = tensor_to_f32(base_tensor.type_, base.raw_tensor_data(name)?)?;
    let mut cur = matrix_from_tensor(base_tensor, &base_values)?;

    let lora_a = format!("{}.lora_a", name);
    let lora_b = format!("{}.lora_b", name);
    for adapter in adapters {
        let a_tensor = adapter.get_tensor(&lora_a).expect("lora_a tensor");
        let b_tensor = adapter.get_tensor(&lora_b).expect("lora_b tensor");
        if matches!(
            a_tensor.type_,
            GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q8_0
                | GgmlType::Q8_1
                | GgmlType::Q2K
                | GgmlType::Q3K
                | GgmlType::Q4K
                | GgmlType::Q5K
                | GgmlType::Q6K
                | GgmlType::Q8K
                | GgmlType::Iq2Xxs
                | GgmlType::Iq2Xs
                | GgmlType::Iq3Xxs
                | GgmlType::Iq1S
                | GgmlType::Iq4Nl
                | GgmlType::Iq3S
                | GgmlType::Iq2S
                | GgmlType::Iq4Xs
                | GgmlType::Iq1M
        ) || matches!(
            b_tensor.type_,
            GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q8_0
                | GgmlType::Q8_1
                | GgmlType::Q2K
                | GgmlType::Q3K
                | GgmlType::Q4K
                | GgmlType::Q5K
                | GgmlType::Q6K
                | GgmlType::Q8K
                | GgmlType::Iq2Xxs
                | GgmlType::Iq2Xs
                | GgmlType::Iq3Xxs
                | GgmlType::Iq1S
                | GgmlType::Iq4Nl
                | GgmlType::Iq3S
                | GgmlType::Iq2S
                | GgmlType::Iq4Xs
                | GgmlType::Iq1M
        ) {
            return Err(Error::new(
                "quantized LoRA adapters is not supported, please retry with f16 or f32",
            ));
        }

        let a = matrix_from_tensor(a_tensor, &adapter.read_tensor_f32(&lora_a)?)?;
        let b = matrix_from_tensor(b_tensor, &adapter.read_tensor_f32(&lora_b)?)?;
        let delta = if name.starts_with("token_embd") {
            mul_mat(&b, &a)?
        } else {
            mul_mat(&a.transpose(), &b)?
        };
        if delta.ne0 != cur.ne0 || delta.ne1 != cur.ne1 {
            return Err(Error::new(format!(
                "tensor {} shape mismatch with LoRA delta",
                name
            )));
        }
        let rank = b.ne0.max(1) as f32;
        let scale = if adapter.alpha != 0.0 {
            adapter.scale * adapter.alpha / rank
        } else {
            adapter.scale
        };
        println!(
            "merge_tensor :   + merging from adapter type={}",
            a_tensor.type_.name()
        );
        println!(
            "merge_tensor :     input_scale={} calculated_scale={} rank={}",
            adapter.scale, scale, rank
        );
        for j in 0..cur.data.len() {
            cur.data[j] += delta.data[j] * scale;
        }
    }
    println!("merge_tensor :   + output type is {}", out_type.name());
    let mut values = cur.data;
    values.truncate(base_tensor.ne[0].max(0) as usize * base_tensor.ne[1].max(0) as usize);
    tensor_from_f32(out_type, &values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_size_units() {
        assert_eq!(split_str_to_n_bytes("128M").unwrap(), 128 * 1000 * 1000);
        assert_eq!(split_str_to_n_bytes("4G").unwrap(), 4_000_000_000);
        assert!(split_str_to_n_bytes("4K").is_err());
    }

    #[test]
    fn mul_mat_matches_ggml_shape() {
        // a is [2 rows, 3 fast cols], b is [4 rows, 3 fast cols].
        let a = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 4, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let out = mul_mat(&a, &b).unwrap();
        assert_eq!((out.ne0, out.ne1), (2, 4));
        assert_eq!(out.get(0, 0), 1.0);
        assert_eq!(out.get(0, 1), 2.0);
        assert_eq!(out.get(1, 3), 15.0);
    }

    #[test]
    fn zero_rows_are_dropped() {
        let rows = vec![1.0, 0.0, 0.0, 0.0, 2.0, 3.0];
        let filtered = HiddenCapture::filter_nonzero_rows(&rows, 2).unwrap();
        assert_eq!(filtered, vec![1.0, 0.0, 2.0, 3.0]);
    }

    #[test]
    fn wav_header_is_well_formed() {
        let wav = wav_silence(24_000, 10);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + 20);
    }
}

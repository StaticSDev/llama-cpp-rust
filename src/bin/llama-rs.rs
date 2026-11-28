//! Std-only CLI for greedy or top-p generation with a GGUF Qwen model.

use std::env;

use llama_cpp_rust::runtime::generate::generate;
use llama_cpp_rust::runtime::loader::{load, load_with_gpu};

const USAGE: &str = "\
Usage: llama-rs --model <path> --prompt <text> [options]

Options:
  --model <path>        GGUF model file
  --prompt <text>       User prompt
  -n, --n-predict <n>   Max tokens to generate (default: 64)
  --temp <f32>          Sampling temperature; 0 means greedy (default: 0)
  --top-p <f32>         Nucleus sampling mass (default: 0.9)
  --gpu                 Use CUDA GPU weights (needs cuda feature; requires --temp 0)
  --help                Print this help";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> std::result::Result<(), String> {
    let mut model: Option<String> = None;
    let mut prompt = String::new();
    let mut n_predict = 64usize;
    let mut temp = 0.0f32;
    let mut top_p = 0.9f32;
    let mut gpu = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--model" => model = Some(value(&mut args, "--model")?),
            "--prompt" => prompt = value(&mut args, "--prompt")?,
            "-n" | "--n-predict" => {
                let raw = value(&mut args, &arg)?;
                n_predict = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid -n value '{raw}': {e}"))?;
            }
            "--temp" => {
                let raw = value(&mut args, "--temp")?;
                temp = raw
                    .parse::<f32>()
                    .map_err(|e| format!("invalid --temp value '{raw}': {e}"))?;
            }
            "--top-p" => {
                let raw = value(&mut args, "--top-p")?;
                top_p = raw
                    .parse::<f32>()
                    .map_err(|e| format!("invalid --top-p value '{raw}': {e}"))?;
            }
            "--gpu" => gpu = true,
            other => return Err(format!("unknown argument: {other}\n{USAGE}")),
        }
    }

    let model = model.ok_or_else(|| format!("missing --model\n{USAGE}"))?;
    let weights = if gpu {
        load_with_gpu(model.as_str(), true)
    } else {
        load(model.as_str())
    }
    .map_err(|e| e.to_string())?;
    #[cfg(feature = "cuda")]
    {
        if let Some(g) = weights.gpu() {
            println!("{}", g.info());
            println!("GPU graph: all layers on device (greedy sampling)");
        }
    }
    let output = generate(&weights, &prompt, n_predict, temp, top_p).map_err(|e| e.to_string())?;
    println!("{output}");
    Ok(())
}

fn value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> std::result::Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

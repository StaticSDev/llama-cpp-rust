//! Std-only CLI for the local web chat server.

use std::env;
use std::net::TcpListener;

use llama_cpp_rust::runtime::loader::{load, load_with_gpu};
use llama_cpp_rust::web::server;

const USAGE: &str = "\
Usage: llama-web --model <path> [options]

Options:
  --model <path>     GGUF model file (required)
  --host <addr>      Listen address (default: 127.0.0.1)
  --port <u16>       Listen port (default: 8080)
  --gpu              Use CUDA GPU weights (needs cuda feature)
  --help             Print this help";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> std::result::Result<(), String> {
    let mut model: Option<String> = None;
    let mut host = "127.0.0.1".to_string();
    let mut port = 8080u16;
    let mut gpu = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = Some(value(&mut args, "--model")?),
            "--host" => host = value(&mut args, "--host")?,
            "--gpu" => gpu = true,
            "--port" => {
                let raw = value(&mut args, "--port")?;
                port = raw
                    .parse::<u16>()
                    .map_err(|e| format!("invalid --port value '{raw}': {e}"))?;
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}\n{USAGE}")),
        }
    }

    let model = model.ok_or_else(|| format!("missing --model\n{USAGE}"))?;
    let weights = if gpu {
        load_with_gpu(&model, true)
    } else {
        load(&model)
    }
    .map_err(|e| e.to_string())?;
    #[cfg(feature = "cuda")]
    let gpu = weights.gpu().map(|g| g.info()).unwrap_or_default();
    #[cfg(not(feature = "cuda"))]
    let gpu = String::new();
    #[cfg(feature = "cuda")]
    {
        if !gpu.is_empty() {
            println!("{gpu}");
        }
    }
    let listener =
        TcpListener::bind((host.as_str(), port)).map_err(|e| format!("bind {host}:{port}: {e}"))?;
    println!(
        "http://{}/",
        listener.local_addr().map_err(|e| e.to_string())?
    );
    server::serve(listener, weights, model, gpu);
    Ok(())
}

fn value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> std::result::Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

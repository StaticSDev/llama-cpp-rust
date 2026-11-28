# llama.cpp → Rust

[Русский](README.ru.md) · English

A working Rust port of llama.cpp with a CPU fallback, an optional CUDA backend
for Turing GPUs, and a local pixel-styled chat app.

The reference tree is pinned at commit `15fcfc7357f8de73671d39e7d8c594e1beb3052d`.
It is kept in `reference/llama.cpp` and is not committed.

---

## What works

- Qwen3.5 dense and hybrid layers: gated delta net, full attention, RoPE, FFN.
- GGUF loading for Q4_K, Q5_K, Q6_K, Q8_0, F32, F16 and BF16 tensors.
- Greedy and temperature + top-p sampling.
- CPU matvec across 12 threads; full Qwen3.5 graph resident in VRAM on CUDA.
- A std-only HTTP server with newline-delimited streaming and live telemetry.

## Quick start

Default build, CPU only:

```sh
cargo build --release --bin llama-rs
target/release/llama-rs \
  --model models/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt "Hi, who are you?" -n 8 --temp 0
```

CUDA build, Turing and newer (`sm_75`):

```sh
cargo build --release --features cuda --bin llama-rs
target/release/llama-rs \
  --model models/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt "Hi, who are you?" -n 8 --temp 0 --gpu
```

`build.rs` compiles `cuda/lcr_kernels.cu` with `nvcc -arch=sm_75`.
Without `--features cuda`, the `--gpu` flag returns a clear error.

In GPU mode the whole per-token graph runs on the device: embeddings, RMS/L2
norms, RoPE, causal conv, gated delta net, attention with a KV cache, SwiGLU,
the tied LM head and argmax. Greedy sampling is supported; temperature and
top-p sampling are still CPU-only, so use `--temp 0` with `--gpu`. The GPU KV
cache is capped at 4096 tokens to fit a 6 GB card.

## Web chat

```sh
cargo build --release --features cuda --bin llama-web
target/release/llama-web \
  --model models/Qwen3.5-4B-Q4_K_M.gguf \
  --port 8080 --gpu
```

Open <http://127.0.0.1:8080/>. The interface keeps chats in `localStorage`,
shows prefill progress as ten pixel segments, and reports prefill and decode
speed in tokens per second. One request is handled at a time.

API:

| Route | Purpose |
| --- | --- |
| `GET /api/status` | Model path, ready flag, GPU name |
| `POST /api/chat/stream` | NDJSON events: `prompt`, `token`, `done`, `error` |

---

## Speed benchmark

Measured with `bench/speed_bench.py` on the same prompt. Each engine runs in
isolation so only one copy of the model occupies VRAM.

**Hardware:** GTX 1660 SUPER (TU116, 6 GB), CUDA 13.4, driver 615.78, 12 CPU
threads, 15 GiB RAM. **Model:** Qwen3.5-4B-Q4_K_M.gguf.

| Metric | llama.cpp upstream | Rust port | Rust / upstream |
| --- | ---: | ---: | ---: |
| Cold prefill, 203 / 202 tokens | 154.98 tok/s | 6.51 tok/s | 0.04× |
| Generation, 6 tokens | 46.14 tok/s | 6.53 tok/s | 0.14× |
| Generation, 200 tokens, partial CUDA | 54.57 tok/s | 6.34 tok/s | 0.12× |
| Generation, 32 tokens, full GPU graph | — | 8.88 tok/s | — |

Upstream flags: `-ngl 99 -c 32768 -t 6 -tb 12 -b 2048 -ub 512 -np 1 -fa on
-ctk q4_0 -ctv q4_0 --context-shift --jinja`. Prompt cache reuse was disabled
for the cold-prefill number.

The short six-token run is noisy; the 200-token run is the stable reference.
The port is still roughly 6× slower than upstream. The full graph is on the
GPU, but every operation launches its own kernel and synchronizes before the
next one; there is no fusion, CUDA graphs or vectorized dequant. Upstream
reaches the memory bandwidth limit of this card; this port does not.

Reproduce:

```sh
python3 bench/speed_bench.py --runs 1 --max-tokens 6 --out bench/results.json
```

Raw data: [bench/results.json](bench/results.json).

---

## Layout

```text
src/
  runtime/     loader, dequant math, Qwen3.5 forward, sampler, CUDA FFI
  gguf/        GGUF container and type tables
  ggml/        block layouts, quantizers, backend scaffolding
  vocab/       tokenizer and chat metadata
  web/         HTTP server and static chat app
cuda/          sm_75 dequant and matvec kernels, compiled by build.rs
bench/         speed comparison against upstream llama.cpp
reference/     pinned llama.cpp source, gitignored
models/        local GGUF files, gitignored
```

## Tests

```sh
cargo test                 # 250 passed
cargo test --features cuda # 250 passed
```

Zero warnings in both configurations.

## License

Portions derived from llama.cpp remain under the original MIT license,
Copyright (c) 2023–2026 ggml authors. See `LICENSE`.

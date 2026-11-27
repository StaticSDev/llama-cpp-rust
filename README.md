# llama-cpp-rust

Rust rewrite of [llama.cpp](https://github.com/ggml-org/llama.cpp).

## State

Port from pinned upstream commit `15fcfc7357f8de73671d39e7d8c594e1beb3052d`.
`cargo check`, `cargo build` and `cargo test` all pass with zero warnings:
248 tests, 0 failures. Each module keeps the base name of its C/C++ source.

Ported modules:

- core: `core`, `impl`, `cparams`, `io`
- ggml: `ggml/types`, `ggml/tensor`, `ggml/quants`, `ggml/alloc`,
  `ggml/backend`, `ggml/opt`, `ggml/threading`
- llama: `gguf`, `vocab`, `sampler`, `kv_cache`, `grammar`, `batch`, `arch`,
  `hparams`, `json`, `unicode`, `chat`, `mmap`, `model`, `model_loader`,
  `context`, `graph`, `memory`, `quant`, `adapter`, `api`
- common: `common`, `arg`, `sampling`, `speculative`, `ngram`, `json_schema`,
  `jinja`, `chat_parser`, `download`, `console`, `misc`
- tools: `server`, `cli`, `bench`, `quantize`, `completion`, `batched_bench`,
  `mtmd`, `perplexity`, `imatrix`, `misc`

Not ported yet:

- examples and Python conversion scripts
- backend-specific GPU/RPC/Hexagon paths and real numeric graph execution
- live inference runtime; tool entry points use injected traits/callbacks

The unmodified C/C++ tree is downloaded locally into `reference/llama.cpp` and
ignored by git. It is the translation reference, not part of this repository.

## Layout

- `src/<module>/` - one llama.cpp module per directory, split into subfiles.
- `src/<module>/mod.rs` - module declarations and re-exports.
- `reference/` - local upstream checkout, gitignored.

## Run a model

A small CPU runtime for Qwen3.5 dense/hybrid GGUF lives in `src/runtime/`.
Models are not committed; download one into `models/` first.

```sh
cargo build --release --bin llama-rs
target/release/llama-rs \
  --model models/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt "Hi, who are you?" -n 8 --temp 0
```

The runtime dequantizes one weight row at a time, so it is memory-safe but slow
on CPU. Greedy (`--temp 0`) and temperature + top-p sampling are supported.

## Web chat

A minimal monochrome chat UI is served by the same Rust binary:

```sh
cargo build --release --bin llama-web
target/release/llama-web \
  --model models/Qwen3.5-4B-Q4_K_M.gguf \
  --port 8080
```

Open `http://127.0.0.1:8080/`. The page calls `POST /api/chat`.
One request is processed at a time because generation is CPU-bound.

## License

Original code: MIT, Copyright (c) 2023-2026 ggml authors. See `LICENSE`.

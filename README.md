# llama-cpp-rust

Rust rewrite of [llama.cpp](https://github.com/ggml-org/llama.cpp).

## State

Port from pinned upstream commit `15fcfc7357f8de73671d39e7d8c594e1beb3052d`.
`cargo check`, `cargo build` and `cargo test` all pass: 157 tests, 0 failures.
Each module keeps the base name of its C/C++ source.

Ported modules:

- core: `core`, `impl`, `cparams`, `io`
- ggml: `ggml/types`, `ggml/tensor`, `ggml/quants`, `ggml/alloc`,
  `ggml/backend`, `ggml/opt`, `ggml/threading`
- llama: `gguf`, `vocab`, `sampler`, `kv_cache`, `grammar`, `batch`, `arch`,
  `hparams`, `json`, `unicode`, `chat`, `mmap`, `model`, `model_loader`,
  `context`, `graph`, `memory`, `quant`, `adapter`, `api`
- common: `common`

Not ported yet:

- tools, server, examples, conversion scripts
- backend-specific GPU/RPC/Hexagon paths and real numeric graph execution

The unmodified C/C++ tree is downloaded locally into `reference/llama.cpp` and
ignored by git. It is the translation reference, not part of this repository.

## Layout

- `src/<module>/` - one llama.cpp module per directory, split into subfiles.
- `src/<module>/mod.rs` - module declarations and re-exports.
- `reference/` - local upstream checkout, gitignored.

## License

Original code: MIT, Copyright (c) 2023-2026 ggml authors. See `LICENSE`.

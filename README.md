# llama-cpp-rust

Rust rewrite of [llama.cpp](https://github.com/ggml-org/llama.cpp).

## State

Dry port, wave 1 done. Source is translated from pinned upstream commit
`15fcfc7357f8de73671d39e7d8c594e1beb3052d`. No build or tests have been run.
Each module keeps the base name of its C/C++ source.

Ported modules:

- `gguf`, `vocab`, `sampler`, `kv_cache`, `grammar`, `batch`
- `arch`, `hparams`, `json`, `unicode`, `chat`
- `core`, `impl`, `cparams`, `io`

Not ported yet:

- ggml tensor library, backends and quantizers
- model loader, model, context, compute graph and memory variants
- tools, server, examples and conversion scripts

The unmodified C/C++ tree is downloaded locally into `reference/llama.cpp` and
ignored by git. It is the translation reference, not part of this repository.

## Layout

- `src/core.rs` - shared token/error/RNG primitives.
- `src/<name>.rs` - one ported llama.cpp module per file.
- `reference/` - local upstream checkout, gitignored.

## License

Original code: MIT, Copyright (c) 2023-2026 ggml authors. See `LICENSE`.

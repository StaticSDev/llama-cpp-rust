# llama-cpp-rust

Rust rewrite of [llama.cpp](https://github.com/ggml-org/llama.cpp).

## State

Dry port in progress. Source is translated module by module from the pinned
upstream commit `15fcfc7357f8de73671d39e7d8c594e1beb3052d`. Nothing is
compiled or verified yet. Each module has the same base name as its C/C++ source.

The unmodified C/C++ tree is downloaded locally into `reference/llama.cpp` and
ignored by git. It is the translation reference, not part of this repository.

## Layout

- `src/core.rs` — shared token/error/RNG primitives.
- `src/<name>.rs` — one ported llama.cpp module per file.
- `reference/` — local upstream checkout, gitignored.

## License

Original code: MIT, Copyright (c) 2023-2026 ggml authors. See `LICENSE`.

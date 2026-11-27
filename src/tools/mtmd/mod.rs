//! std-only Rust port of `tools/mtmd/mtmd-cli.cpp` and its helpers.
//!
//! Ported files: `mtmd-cli.cpp`, `mtmd-helper.{h,cpp}`,
//! `mtmd-helper-common.h`, `mtmd-helper-video-ffmpeg.cpp`, the data types of
//! `mtmd.h`, and the pure helpers of `mtmd-helper-gen.cpp`.
//!
//! Not ported: stb_image / miniaudio decoding, ffmpeg subprocesses, the ggml
//! graph code in `mtmd.cpp` / `mtmd-model.cpp` / `mtmd-image.cpp` /
//! `mtmd-audio.cpp` / `mtmd-graph.cpp` / `mtmd-subcontext.cpp`, and the
//! model-specific audio generation pipelines in `mtmd-helper-gen.cpp`. Those
//! need an image/audio decoder, a process API or a tensor backend. Host-side
//! traits supply their operations instead.
//!
//! Device pointers, C callbacks and the `console::` / `common_params_parse`
//! layer stay in the host: a device is a name and a callback is a Rust
//! function pointer.

pub mod cli;
pub mod gen;
pub mod helper;
pub mod log;
pub mod resampler;
pub mod traits;
pub mod types;
pub mod video;

pub use cli::*;
pub use gen::*;
pub use helper::*;
pub use log::*;
pub use resampler::*;
pub use traits::*;
pub use types::*;
pub use video::*;

#[cfg(test)]
mod tests;

//! Host-side traits standing in for the llama / mtmd / console C APIs.
//!
//! The reference tool drives opaque C pointers. Here the same operations are
//! expressed as traits, so the CLI and helpers stay std-only and testable.

use std::io::{self, BufRead, Write};

use super::types::{
    AudioStreamParams, Batch, ChatMessage, GenAudioInfo, InputChunk, InputPart, Token,
    VideoConfig, VideoStreamInfo,
};

/// `llama_vocab` operations used by the CLI.
pub trait Vocab {
    fn is_eog(&self, token: Token) -> bool;
    fn is_control(&self, token: Token) -> bool;
}

/// `llama_context` operations used by the CLI and the helper.
pub trait Llama {
    type Vocab: Vocab;

    fn vocab(&self) -> &Self::Vocab;
    fn model_exists(&self) -> bool;
    fn model_has_chat_template(&self) -> bool;
    fn n_tokens_delay(&self) -> i32;
    fn n_embd_inp(&self) -> i32;
    fn n_ubatch(&self) -> i32;
    fn logits(&self) -> &[f32];
    fn process(&mut self, batch: &Batch) -> i32;
    fn set_causal_attn(&mut self, enabled: bool);
    fn memory_clear(&mut self);
    fn perf_print(&self);
    fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<Token>;
    fn detokenize(&self, tokens: &[Token]) -> String;
    fn token_to_piece(&self, token: Token) -> String;
}

/// `common_sampler` operations used by `generate_response`.
pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> Token;
    fn accept(&mut self, token: Token, accept_logits: bool);
}

/// `common_chat_templates` operations used by the CLI.
pub trait ChatTemplates {
    fn format_example(&self, use_jinja: bool, kwargs: &str) -> String;
    fn format_single(
        &self,
        history: &[ChatMessage],
        new_msg: &ChatMessage,
        add_generation_prompt: bool,
        use_jinja: bool,
    ) -> String;
}

/// `mtmd_context` operations used by the CLI and the helper.
pub trait Mtmd {
    fn support_vision(&self) -> bool;
    fn support_audio(&self) -> bool;
    fn support_audio_stream(&self) -> bool;
    /// Build flag `MTMD_VIDEO`: the built-in ffmpeg decoder is available.
    fn has_builtin_video_decoder(&self) -> bool;
    fn audio_sample_rate(&self) -> i32;
    fn marker(&self) -> &str;
    fn decode_use_mrope(&self) -> bool;
    fn decode_use_non_causal(&self, chunk: &InputChunk) -> bool;
    fn get_video_config(
        &self,
        src_width: i32,
        src_height: i32,
        src_n_frames: i32,
        src_fps: f32,
    ) -> VideoConfig;
    fn tokenize_from_parts(
        &self,
        parts: &[InputPart],
        add_special: bool,
    ) -> Result<Vec<InputChunk>, i32>;
    fn encode_chunk(&mut self, chunk: &InputChunk) -> i32;
    fn output_embd(&self) -> &[f32];
    fn batch_init(&mut self) -> i32;
    fn batch_add_chunk(&mut self, batch: i32, chunk: &InputChunk) -> i32;
    fn batch_encode(&mut self, batch: i32) -> i32;
    fn batch_get_output_embd(&self, batch: i32, chunk: &InputChunk) -> Option<Vec<f32>>;
    fn audio_stream_init(&self, params: &AudioStreamParams) -> Option<Box<dyn AudioStream>>;
    fn gen_audio_info(&self) -> GenAudioInfo;
}

/// `mtmd_audio_stream` operations used by `run_audio_stream`.
pub trait AudioStream {
    fn push(&mut self, samples: &[f32]) -> i32;
    fn flush(&mut self, end: bool) -> i32;
    fn step(&mut self, last_token: Token) -> i32;
    fn output_embd(&self) -> &[f32];
}

/// Video decoder contract from `mtmd_helper_video_decoder`.
///
/// `open` returns an opaque host handle. Trait objects remove the C function
/// pointers, but the call order (open, start, read_audio, read_frame, close)
/// is the same.
pub trait VideoDecoder {
    fn open(
        &self,
        path: Option<&str>,
        buf: &[u8],
        info: &mut VideoStreamInfo,
    ) -> Option<usize>;
    fn start(
        &self,
        state: usize,
        width: u32,
        height: u32,
        fps: f32,
        fps_picked: &mut bool,
    ) -> bool;
    /// `rgb == None` skips the frame.
    fn read_frame(&self, state: usize, rgb: Option<&mut [u8]>, width: u32, height: u32) -> bool;
    fn read_audio(&self, state: usize, sample_rate: i32) -> Option<Vec<f32>>;
    fn close(&self, state: usize);
}

/// Decoders for the file formats accepted by `mtmd_helper_bitmap_init_from_*`.
pub trait MediaDecoder {
    /// SHA-256 hex of the file bytes in C; the hash function lives outside
    /// `std`, so the host supplies it.
    fn hash(&self, buf: &[u8]) -> String;
    fn decode_audio(&self, buf: &[u8], rate: i32) -> Option<Vec<f32>>;
    /// RGB24 image: (width, height, pixels).
    fn decode_image(&self, buf: &[u8]) -> Option<(u32, u32, Vec<u8>)>;
    /// Header-only values used by `mtmd_helper_get_media_meta`.
    fn audio_meta(&self, _buf: &[u8]) -> Option<(i32, f32)> {
        None
    }
    fn image_meta(&self, _buf: &[u8]) -> Option<(u32, u32)> {
        None
    }
    fn video_decoder(&self) -> Option<Box<dyn VideoDecoder>>;
}

/// Host with no decoders: video/image/audio input is reported as unavailable.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullMediaDecoder;

impl MediaDecoder for NullMediaDecoder {
    fn hash(&self, _buf: &[u8]) -> String {
        String::new()
    }

    fn decode_audio(&self, _buf: &[u8], _rate: i32) -> Option<Vec<f32>> {
        None
    }

    fn decode_image(&self, _buf: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
        None
    }

    fn video_decoder(&self) -> Option<Box<dyn VideoDecoder>> {
        None
    }
}

/// `console::` helpers used by the interactive loop.
pub trait Console {
    fn init(&mut self, simple_io: bool, use_color: bool);
    fn cleanup(&mut self);
    fn set_display(&mut self, display: DisplayType);
    /// `None` is EOF; the C++ code treats a submitted line as newline-terminated.
    fn readline(&mut self) -> Option<String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplayType {
    UserInput,
    #[default]
    Reset,
}

#[derive(Debug, Default)]
pub struct StdioConsole {
    display: DisplayType,
}

impl Console for StdioConsole {
    fn init(&mut self, _simple_io: bool, _use_color: bool) {}

    fn cleanup(&mut self) {}

    fn set_display(&mut self, display: DisplayType) {
        self.display = display;
    }

    fn readline(&mut self) -> Option<String> {
        let mut line = String::new();
        let n = io::stdin().lock().read_line(&mut line).ok()?;
        if n == 0 {
            return None;
        }
        while line.ends_with('\n') || line.ends_with('\r') {
            line.pop();
        }
        Some(line)
    }
}

/// `fflush(stdout)` after streaming a token.
pub fn flush_stdout() {
    let _ = io::stdout().flush();
}

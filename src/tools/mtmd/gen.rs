//! Pure helpers from `mtmd-helper-gen.cpp`.
//!
//! The model-specific audio pipelines need ggml graphs and are not ported.
//! WAV writing, language-code resolution and input validation are.

use super::log::log_err;
use super::types::GenAudioInp;

pub const TTS_LANG_CODES: &[(&str, &str)] = &[
    ("zh", "chinese"),
    ("en", "english"),
    ("de", "german"),
    ("it", "italian"),
    ("pt", "portuguese"),
    ("es", "spanish"),
    ("ja", "japanese"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ru", "russian"),
];

pub fn tts_resolve_lang(lang: &str) -> &str {
    TTS_LANG_CODES
        .iter()
        .find(|(code, _)| *code == lang)
        .map_or(lang, |(_, name)| *name)
}

fn put_u32_le(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_u16_le(buf: &mut Vec<u8>, value: u16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// `write_wav16_header`; `data_sz == u32::MAX` is the streaming sentinel.
pub fn write_wav16_header(buf: &mut Vec<u8>, data_sz: u32, rate: i32, ch: u16) {
    let riff_sz = if data_sz == u32::MAX {
        u32::MAX
    } else {
        36 + data_sz
    };
    let fmt_sz = 16u32;
    let byte_rate = rate as u32 * 2 * ch as u32;
    let fmt = 1u16;
    let align = 2 * ch;
    let bits = 16u16;

    buf.extend_from_slice(b"RIFF");
    put_u32_le(buf, riff_sz);
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    put_u32_le(buf, fmt_sz);
    put_u16_le(buf, fmt);
    put_u16_le(buf, ch);
    put_u32_le(buf, rate as u32);
    put_u32_le(buf, byte_rate);
    put_u16_le(buf, align);
    put_u16_le(buf, bits);
    buf.extend_from_slice(b"data");
    put_u32_le(buf, data_sz);
}

pub fn append_wav16_pcm(buf: &mut Vec<u8>, pcm: &[f32]) {
    for sample in pcm {
        let clamped = sample.max(-1.0).min(1.0);
        let s = (clamped * 32767.0).round() as i16;
        buf.extend_from_slice(&s.to_le_bytes());
    }
}

/// `write_wav16`; returns false when the RIFF size would overflow.
pub fn write_wav16(buf: &mut Vec<u8>, pcm: &[f32], rate: i32, ch: u16) -> bool {
    if pcm.len() > ((u32::MAX as usize - 36) / 2) {
        return false;
    }
    write_wav16_header(buf, (pcm.len() * 2) as u32, rate, ch);
    append_wav16_pcm(buf, pcm);
    true
}

/// Pipeline flags checked by `mtmd_helper_gen_audio_set_input`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenSupport {
    pub continue_ok: bool,
    pub speed_ok: bool,
    pub delivery_ok: bool,
    pub cfg_ok: bool,
    pub flow_steps_ok: bool,
}

impl GenSupport {
    pub fn new() -> Self {
        Self {
            speed_ok: true,
            delivery_ok: true,
            cfg_ok: true,
            flow_steps_ok: true,
            ..Self::default()
        }
    }
}

/// Validation part of `mtmd_helper_gen_audio_set_input`.
/// 0 = accepted, 1 = invalid for the model, 2 = invalid values.
pub fn validate_gen_audio_input(inp: &GenAudioInp, supports: GenSupport) -> i32 {
    if inp.n_past > 0 && !supports.continue_ok {
        log_err("mtmd_helper_gen_audio: the pipeline cannot continue a sequence\n");
        return 1;
    }
    if !inp.speed.is_finite() || inp.speed <= 0.0 || !inp.loudness.is_finite() || inp.loudness > 0.0 {
        log_err("mtmd_helper_gen_audio: speed must be > 0 and loudness < 0 LUFS (0 = model default)\n");
        return 2;
    }
    if inp.speed != 1.0 && !supports.speed_ok {
        log_err("mtmd_helper_gen_audio: the model does not support speed\n");
        return 2;
    }
    if inp.loudness != 0.0 && !supports.delivery_ok {
        log_err("mtmd_helper_gen_audio: the model does not support loudness\n");
        return 2;
    }
    if !inp.cfg_scale.is_finite() || inp.cfg_scale < 0.0 {
        log_err("mtmd_helper_gen_audio: cfg_scale must be > 0 (0 = model default)\n");
        return 2;
    }
    if inp.cfg_scale != 0.0 && !supports.cfg_ok {
        log_err("mtmd_helper_gen_audio: the model does not support cfg_scale\n");
        return 2;
    }
    if !(0..=1024).contains(&inp.flow_steps) {
        log_err("mtmd_helper_gen_audio: flow_steps must be in [1, 1024] (0 = model default)\n");
        return 2;
    }
    if inp.flow_steps != 0 && !supports.flow_steps_ok {
        log_err("mtmd_helper_gen_audio: the model does not support flow_steps\n");
        return 2;
    }
    0
}

/// `mtmd_helper_gen_audio_max_frames` wrapper in `mtmd_helper::gen_audio`.
pub fn max_frames(n_predict: i32, n_model: i32) -> i32 {
    if n_predict > 0 {
        n_predict
    } else if n_model > 0 {
        n_model
    } else {
        i32::MAX
    }
}

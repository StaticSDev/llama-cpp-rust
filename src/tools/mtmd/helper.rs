//! Port of `mtmd-helper.h`, `mtmd-helper.cpp` and `mtmd-helper-common.h`.
//!
//! The embedding-decoding helpers, bitmap dispatch and media metadata logic
//! are translated as-is. Actual image/audio decoding lives behind
//! `MediaDecoder`; the video sequence logic is in `video.rs`.

use std::cmp::{max, min};
use std::fs;

use super::log::{log_err, log_inf, log_wrn};
use super::traits::{Llama, MediaDecoder, Mtmd};
use super::types::{
    Bitmap, Cap, DecoderPos, EmbdBatch, GenAudioType, HelperBitmapWrapper, ImageTokens,
    InitOpt, InputChunk, InputChunkType, MediaMeta, MediaType, Pos, SeqId, VideoStreamInfo,
};
use super::video::{decode_still_image, video_init};

/// `mtmd_helper_get_n_tokens`.
pub fn get_n_tokens(chunks: &[InputChunk]) -> usize {
    chunks.iter().map(InputChunk::get_n_tokens).sum()
}

/// `mtmd_helper_get_n_pos`.
pub fn get_n_pos(chunks: &[InputChunk]) -> Pos {
    chunks.iter().map(InputChunk::get_n_pos).sum()
}

/// `mtmd_helper_image_get_decoder_pos`.
pub fn image_get_decoder_pos(image: &ImageTokens, pos_0: Pos, out: &mut [DecoderPos]) {
    let n = min(out.len(), image.get_n_tokens());
    for i in 0..n {
        out[i] = image.get_decoder_pos(pos_0, i);
    }
}

/// `decode_embd_batch` from `mtmd-helper-common.h`.
#[derive(Debug, Clone)]
pub struct DecodeEmbdBatch {
    pub n_pos_per_embd: i32,
    pub n_mmproj_embd: i32,
    pub n_tokens: i32,
    pub embd: Vec<f32>,
    pub pos: Vec<Pos>,
    pub logits: Vec<i8>,
    pub seq_id: SeqId,
}

impl DecodeEmbdBatch {
    pub fn new(embd: &[f32], n_tokens: i32, n_pos_per_embd: i32, n_mmproj_embd: i32) -> Self {
        debug_assert!(n_tokens > 0 && n_pos_per_embd > 0 && n_mmproj_embd > 0);
        let n_needed = n_tokens as usize * n_mmproj_embd as usize;
        let mut owned = embd[..min(embd.len(), n_needed)].to_vec();
        owned.resize(n_needed, 0.0);
        Self {
            n_pos_per_embd,
            n_mmproj_embd,
            n_tokens,
            embd: owned,
            pos: vec![0; n_tokens as usize * n_pos_per_embd as usize],
            logits: vec![0; n_tokens as usize],
            seq_id: 0,
        }
    }

    pub fn set_position_normal(&mut self, pos_0: Pos, seq_id: SeqId) {
        self.seq_id = seq_id;
        for i in 0..self.n_tokens as usize {
            self.pos[i] = pos_0 + i as Pos;
        }
    }

    pub fn set_position_mrope_2d(&mut self, rel_pos: &[DecoderPos], seq_id: SeqId) {
        debug_assert_eq!(self.n_pos_per_embd, 4);
        debug_assert_eq!(rel_pos.len(), self.n_tokens as usize);
        self.seq_id = seq_id;
        let n = self.n_tokens as usize;
        for i in 0..n {
            self.pos[i] = rel_pos[i].t as Pos;
            self.pos[i + n] = rel_pos[i].y as Pos;
            self.pos[i + n * 2] = rel_pos[i].x as Pos;
            self.pos[i + n * 3] = rel_pos[i].z as Pos;
        }
    }

    pub fn set_position_mrope_1d(&mut self, pos_0: Pos, seq_id: SeqId) {
        debug_assert_eq!(self.n_pos_per_embd, 4);
        self.seq_id = seq_id;
        let n = self.n_tokens as usize;
        for i in 0..n {
            let p = pos_0 + i as Pos;
            self.pos[i] = p;
            self.pos[i + n] = p;
            self.pos[i + n * 2] = p;
            self.pos[i + n * 3] = p;
        }
    }

    /// Sliced view with section-major positions.
    pub fn get_view(&self, offset: i32, n: i32) -> EmbdBatch {
        debug_assert!(offset >= 0 && n > 0 && offset + n <= self.n_tokens);
        let mut pos_view = Vec::with_capacity(n as usize * self.n_pos_per_embd as usize);
        for j in 0..self.n_pos_per_embd as usize {
            let src = j * self.n_tokens as usize + offset as usize;
            pos_view.extend_from_slice(&self.pos[src..src + n as usize]);
        }
        let begin = offset as usize * self.n_mmproj_embd as usize;
        EmbdBatch {
            n_tokens: n,
            embd: self.embd[begin..begin + n as usize * self.n_mmproj_embd as usize].to_vec(),
            n_embd: self.n_mmproj_embd,
            pos: pos_view,
            n_pos: self.n_pos_per_embd,
            seq_id: self.seq_id,
        }
    }

    /// Render `[offset, offset+n)` into a batch ready for `llama_process`.
    pub fn render(&self, offset: i32, n: i32) -> super::types::Batch {
        debug_assert!(offset >= 0 && n > 0 && offset + n <= self.n_tokens);
        let mut batch = super::types::Batch::default();
        for i in offset..offset + n {
            let begin = i as usize * self.n_mmproj_embd as usize;
            let end = begin + self.n_mmproj_embd as usize;
            let mut p = [0; 4];
            for j in 0..self.n_pos_per_embd as usize {
                p[j] = self.pos[j * self.n_tokens as usize + i as usize];
            }
            let idx = batch.add_embd(
                self.seq_id,
                &self.embd[begin..end],
                &p[..self.n_pos_per_embd as usize],
                self.logits[i as usize] != 0,
            );
            debug_assert!(idx >= 0);
        }
        batch
    }
}

/// `mtmd_helper_decode_image_chunk`.
pub fn decode_image_chunk<M: Mtmd, L: Llama>(
    ctx: &M,
    lctx: &mut L,
    chunk: &InputChunk,
    encoded_embd: &[f32],
    n_past: Pos,
    seq_id: SeqId,
    n_batch: i32,
    new_n_past: &mut Pos,
    callback: Option<&dyn Fn(&EmbdBatch) -> i32>,
) -> i32 {
    debug_assert!(n_batch > 0);
    let chunk_type = chunk.get_type();
    if chunk_type == InputChunkType::Text {
        log_err("failed to decode chunk: input chunk not of image/audio type\n");
        return -1;
    }
    let name = if chunk_type == InputChunkType::Image {
        "image"
    } else {
        "audio"
    };

    let n_mmproj_embd = lctx.n_embd_inp();
    let n_pos_per_embd = if ctx.decode_use_mrope() { 4 } else { 1 };
    let n_tokens = chunk.get_n_tokens() as i32;
    let mut batch_embd = DecodeEmbdBatch::new(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);

    if ctx.decode_use_mrope() {
        if chunk_type == InputChunkType::Image {
            let image_tokens = match chunk.get_tokens_image() {
                Some(v) => v,
                None => {
                    log_err("failed to decode chunk: image tokens are null\n");
                    return -1;
                }
            };
            let n = image_tokens.get_n_tokens();
            let mut rel_pos = vec![DecoderPos::default(); n];
            image_get_decoder_pos(image_tokens, n_past, &mut rel_pos);
            if rel_pos.len() != n_tokens as usize {
                log_err("failed to decode chunk: image tokens are null\n");
                return -1;
            }
            batch_embd.set_position_mrope_2d(&rel_pos, seq_id);
        } else if chunk_type == InputChunkType::Audio {
            batch_embd.set_position_mrope_1d(n_past, seq_id);
        } else {
            log_err("invalid chunk type for M-RoPE\n");
            return -1;
        }
    } else {
        batch_embd.set_position_normal(n_past, seq_id);
    }

    let use_non_causal = ctx.decode_use_non_causal(chunk);
    lctx.set_causal_attn(!use_non_causal);

    let n_img_batches = (n_tokens + n_batch - 1) / n_batch;
    for i_batch in 0..n_img_batches {
        let pos_offset = i_batch * n_batch;
        let n_tokens_batch = min(n_batch, n_tokens - pos_offset);
        log_inf(&format!(
            "decoding {} batch {}/{}, n_tokens_batch = {}\n",
            name,
            i_batch + 1,
            n_img_batches,
            n_tokens_batch
        ));

        let batch = batch_embd.render(pos_offset, n_tokens_batch);
        let ret = lctx.process(&batch);
        if ret != 0 {
            lctx.set_causal_attn(true);
            log_err(&format!("failed to decode {}\n", name));
            return ret;
        }
        if let Some(cb) = callback {
            let view = batch_embd.get_view(pos_offset, n_tokens_batch);
            let ret = cb(&view);
            if ret != 0 {
                lctx.set_causal_attn(true);
                log_err("post-decode callback failed\n");
                return ret;
            }
        }
        log_inf(&format!(
            "{} decoded (batch {}/{}) in 0 ms\n",
            name,
            i_batch + 1,
            n_img_batches
        ));
    }

    lctx.set_causal_attn(true);
    *new_n_past = n_past + chunk.get_n_pos();
    0
}

/// `mtmd_helper_eval_chunk_single`.
pub fn eval_chunk_single<M: Mtmd, L: Llama>(
    ctx: &mut M,
    lctx: &mut L,
    chunk: &InputChunk,
    n_past: Pos,
    seq_id: SeqId,
    n_batch: i32,
    logits_last: bool,
    new_n_past: &mut Pos,
) -> i32 {
    debug_assert!(n_batch > 0);
    let chunk_type = chunk.get_type();
    if chunk_type == InputChunkType::Text {
        let tokens = chunk.get_tokens_text();
        let mut batch = super::types::Batch::default();
        let mut i = 0;
        let mut past = n_past;
        while i < tokens.len() {
            batch.clear();
            let mut n_added = 0;
            let mut idx = -1;
            while i < tokens.len() && n_added < n_batch {
                idx = batch.add_token(seq_id, tokens[i]);
                let pos = past;
                past += 1;
                batch.set_pos(idx, &[pos]);
                n_added += 1;
                i += 1;
            }
            let is_last_token = i == tokens.len();
            if logits_last && is_last_token {
                batch.set_output_logits(idx, true);
            }
            let ret = lctx.process(&batch);
            if ret != 0 {
                log_err("failed to decode text\n");
                return ret;
            }
            *new_n_past += n_added;
        }
    } else if chunk_type == InputChunkType::Image || chunk_type == InputChunkType::Audio {
        let name = if chunk_type == InputChunkType::Image {
            "image"
        } else {
            "audio"
        };
        log_inf(&format!("encoding {} slice...\n", name));
        let ret = ctx.encode_chunk(chunk);
        if ret != 0 {
            log_err(&format!("failed to encode {} slice\n", name));
            return ret;
        }
        log_inf(&format!("{} slice encoded in 0 ms\n", name));
        let embd = ctx.output_embd().to_vec();
        let mut next = n_past;
        let ret = decode_image_chunk(
            ctx,
            lctx,
            chunk,
            &embd,
            n_past,
            seq_id,
            n_batch,
            &mut next,
            None,
        );
        if ret != 0 {
            log_err(&format!("failed to decode {}\n", name));
            return ret;
        }
        *new_n_past = next;
    } else {
        log_err("chunk type not supported\n");
        return 1;
    }
    0
}

/// `mtmd_helper_eval_chunks`.
pub fn eval_chunks<M: Mtmd, L: Llama>(
    ctx: &mut M,
    lctx: &mut L,
    chunks: &[InputChunk],
    n_past: Pos,
    seq_id: SeqId,
    n_batch: i32,
    logits_last: bool,
    new_n_past: &mut Pos,
) -> i32 {
    if chunks.is_empty() {
        log_wrn("no chunks to eval\n");
        return 0;
    }
    let mut past = n_past;
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_logits_last = i == chunks.len() - 1 && logits_last;
        let ret = eval_chunk_single(
            ctx,
            lctx,
            chunk,
            past,
            seq_id,
            n_batch,
            chunk_logits_last,
            &mut past,
        );
        if ret != 0 {
            log_err(&format!("failed to eval chunk {}\n", i));
            return ret;
        }
        *new_n_past = past;
    }
    0
}

/// `audio_helpers::is_audio_file`, magic-byte sniffing only.
pub fn is_audio_file(buf: &[u8]) -> bool {
    if buf.len() < 12 {
        return false;
    }
    let is_wav = &buf[..4] == b"RIFF" && &buf[8..12] == b"WAVE";
    let is_mp3 = &buf[..3] == b"ID3" || (buf[0] == 0xFF && buf[1] & 0xE0 == 0xE0);
    let is_flac = &buf[..4] == b"fLaC";
    is_wav || is_mp3 || is_flac
}

pub fn is_webp_file(buf: &[u8]) -> bool {
    buf.len() >= 12 && &buf[..4] == b"RIFF" && &buf[8..12] == b"WEBP"
}

fn empty_wrapper() -> HelperBitmapWrapper {
    HelperBitmapWrapper {
        bitmap: None,
        video: None,
    }
}

fn set_id(wrapper: &mut HelperBitmapWrapper, id: &str) {
    if id.is_empty() {
        return;
    }
    if let Some(bitmap) = wrapper.bitmap.as_mut() {
        bitmap.set_id(id);
    }
    if let Some(video) = wrapper.video.as_mut() {
        video.set_id(id);
    }
}

/// `mtmd_helper_bitmap_init_impl` (file name is only used by video probing).
pub fn bitmap_init_impl<M: Mtmd>(
    ctx: &M,
    fname: Option<&str>,
    buf: &[u8],
    placeholder: bool,
    opt: &InitOpt,
    decoder: &impl MediaDecoder,
) -> HelperBitmapWrapper {
    let id = if placeholder {
        String::new()
    } else {
        decoder.hash(buf)
    };

    if is_audio_file(buf) {
        let sample_rate = ctx.audio_sample_rate();
        if sample_rate < 0 {
            log_err("This model does not support audio input\n");
            return empty_wrapper();
        }
        let pcm = match decoder.decode_audio(buf, sample_rate) {
            Some(pcm) => pcm,
            None => {
                log_err("Unable to read WAV audio file from buffer\n");
                return empty_wrapper();
            }
        };
        let mut wrapper = HelperBitmapWrapper {
            bitmap: Some(Bitmap::init_from_audio(
                pcm.len(),
                if placeholder { None } else { Some(pcm) },
            )),
            video: None,
        };
        set_id(&mut wrapper, &id);
        return wrapper;
    }

    if let Some((nx, ny, data)) = decoder.decode_image(buf) {
        let mut wrapper = HelperBitmapWrapper {
            bitmap: Some(Bitmap::init(
                nx,
                ny,
                if placeholder { None } else { Some(data) },
            )),
            video: None,
        };
        set_id(&mut wrapper, &id);
        return wrapper;
    }

    if is_webp_file(buf) {
        if let Some(dec) = decoder.video_decoder() {
            if let Some(mut bitmap) = decode_still_image(fname, buf, placeholder, &opt.video_params, dec) {
                bitmap.set_id(&id);
                return HelperBitmapWrapper {
                    bitmap: Some(bitmap),
                    video: None,
                };
            }
        }
        log_err("bitmap_init_impl: failed to decode webp buffer\n");
        return empty_wrapper();
    }

    if let Some(dec) = decoder.video_decoder() {
        if let Some(video) = video_init(ctx, fname, buf, &opt.video_params, dec) {
            let mut wrapper = HelperBitmapWrapper {
                bitmap: Some(Bitmap::init_lazy(&id)),
                video: Some(video),
            };
            wrapper.bitmap.as_mut().unwrap().set_is_video(true);
            set_id(&mut wrapper, &id);
            return wrapper;
        }
    }
    log_err("bitmap_init_impl: failed to decode buffer as either image/audio/video\n");
    empty_wrapper()
}

pub fn bitmap_init_from_buf<M: Mtmd>(
    ctx: &M,
    buf: &[u8],
    placeholder: bool,
    opt: &InitOpt,
    decoder: &impl MediaDecoder,
) -> HelperBitmapWrapper {
    bitmap_init_impl(ctx, None, buf, placeholder, opt, decoder)
}

pub fn bitmap_init_from_file<M: Mtmd>(
    ctx: &M,
    fname: &str,
    placeholder: bool,
    opt: &InitOpt,
    decoder: &impl MediaDecoder,
) -> HelperBitmapWrapper {
    let buf = match fs::read(fname) {
        Ok(buf) => buf,
        Err(err) => {
            log_err(&format!("Unable to open file {}: {}\n", fname, err));
            return empty_wrapper();
        }
    };
    bitmap_init_impl(ctx, Some(fname), &buf, placeholder, opt, decoder)
}

/// `mtmd_helper_get_media_meta`.
pub fn get_media_meta(
    buf: &[u8],
    opt: &InitOpt,
    decoder: &impl MediaDecoder,
) -> (i32, MediaMeta) {
    let mut out = MediaMeta::unknown();
    if is_audio_file(buf) {
        if let Some((rate, duration)) = decoder.audio_meta(buf) {
            out.duration = duration;
            out.type_ = MediaType::Audio;
            out.sample_rate = rate;
            return (0, out);
        }
        return (1, out);
    }

    if let Some((width, height)) = decoder.image_meta(buf) {
        out.type_ = MediaType::Image;
        out.width = width;
        out.height = height;
        return (0, out);
    }

    let is_webp = is_webp_file(buf);
    let dec = match decoder.video_decoder() {
        Some(dec) => dec,
        None => return (1, out),
    };
    let mut src = VideoStreamInfo::default();
    let state = dec.open(None, buf, &mut src);
    if state.is_none() || src.width == 0 || src.height == 0 || src.fps <= 0.0 {
        if let Some(state) = state {
            dec.close(state);
        }
        return (1, out);
    }
    dec.close(state.unwrap());
    out.type_ = if is_webp {
        MediaType::Image
    } else {
        MediaType::Video
    };
    out.width = src.width;
    out.height = src.height;
    if !is_webp {
        out.duration = src.duration;
        out.n_frames = src.n_frames;
    }
    let _ = opt;
    (0, out)
}

/// `mtmd_helper_get_capabilities`.
pub fn get_capabilities<M: Mtmd, L: Llama>(model: &L, mctx: Option<&M>) -> Cap {
    let mut cap = Cap {
        chat: true,
        text_gen: true,
        ..Cap::default()
    };
    let mctx = match mctx {
        Some(ctx) => ctx,
        None => return cap,
    };

    cap.vision = mctx.support_vision();
    cap.audio = mctx.support_audio();
    cap.audio_stream = mctx.support_audio_stream();
    cap.video = cap.vision && mctx.has_builtin_video_decoder();
    let gen_audio_type = mctx.gen_audio_info().type_;
    cap.audio_gen = gen_audio_type != GenAudioType::None;
    cap.audio_gen_spk_ref = matches!(
        gen_audio_type,
        GenAudioType::Qwen3Tts | GenAudioType::PocketTts
    );
    cap.audio_duplex = cap.audio_stream && gen_audio_type == GenAudioType::Moshi;

    cap.text_gen = !cap.audio_gen || model.model_has_chat_template();
    cap.chat = cap.text_gen && model.n_tokens_delay() == 0;
    cap
}

/// `mtmd_helper_model_can_chat` / `mtmd_helper_support_video`.
pub fn model_can_chat<M: Mtmd, L: Llama>(lctx: &L, mctx: Option<&M>) -> bool {
    get_capabilities(lctx, mctx).chat
}

pub fn support_video<M: Mtmd>(ctx: &M) -> bool {
    ctx.support_vision() && ctx.has_builtin_video_decoder()
}

/// Clamp used in `init_vision_context`: cap image tokens to `n_ubatch`.
pub fn clamp_image_tokens(
    mparams: &mut super::types::ContextParams,
    mem: &super::types::MemoryUsage,
    n_ubatch: i32,
) -> bool {
    if mem.use_non_causal && mem.image_max_tokens > n_ubatch {
        log_wrn(&format!(
            "init_vision_context: cap image_max_tokens (original={}) to n_ubatch ({}) because model needs non-causal attention on image\n",
            mem.image_max_tokens, n_ubatch
        ));
        log_wrn("init_vision_context: increase n_ubatch (-ub) to increase vision token budget\n");
        mparams.image_max_tokens = n_ubatch;
        mparams.image_min_tokens = min(mparams.image_min_tokens, n_ubatch);
        true
    } else {
        false
    }
}

/// Guard used by the audio stream loop for `n_chunk` and sample counts.
pub fn stream_chunk_size(sample_rate: i32, step_ms: i32) -> usize {
    (max(sample_rate, 0) as usize * max(step_ms, 0) as usize) / 1000
}

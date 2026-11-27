//! Port of `tools/mtmd/mtmd-cli.cpp`.
//!
//! Argument parsing (`common_params_parse`), model loading and signal
//! registration belong to `common` / the OS and are not transcribed. The CLI
//! state machine, prompt interleaving, generation and audio-stream loops are.

use std::cmp::min;
use std::env;
use std::time::Instant;

use super::helper::{bitmap_init_from_file, clamp_image_tokens, eval_chunk_single, get_capabilities};
use super::log::{log, log_err, log_inf, log_msg, log_wrn, LogLevel};
use super::traits::{
    flush_stdout, ChatTemplates, Console, DisplayType, Llama, MediaDecoder, Mtmd, Sampler, Vocab,
};
use super::types::{
    AudioStreamParams, Batch, Cap, ChatMessage, ContextParams, FlashAttnType,
    HelperBitmapWrapper, InitOpt, InputPart, InputText, MemoryUsage, Token, MROPE_SECTIONS,
};
use super::video::Video;

/// Fields of `common_params` used by this tool.
#[derive(Debug, Clone)]
pub struct Params {
    pub model: String,
    pub mmproj: String,
    pub prompt: String,
    pub images: Vec<String>,
    pub system_prompt: String,
    pub n_predict: i32,
    pub n_batch: i32,
    pub n_threads: i32,
    pub mmproj_use_gpu: bool,
    pub mmproj_device: Option<String>,
    pub flash_attn_type: FlashAttnType,
    pub warmup: bool,
    pub image_min_tokens: i32,
    pub image_max_tokens: i32,
    pub chat_template: String,
    pub use_jinja: bool,
    pub default_template_kwargs: String,
    pub simple_io: bool,
    pub use_color: bool,
    pub video_ffmpeg_bin_dir: String,
    pub video_fps: f32,
    pub video_timestamp_interval_ms: i64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            model: String::new(),
            mmproj: String::new(),
            prompt: String::new(),
            images: Vec::new(),
            system_prompt: String::new(),
            n_predict: -1,
            n_batch: 512,
            n_threads: 4,
            mmproj_use_gpu: true,
            mmproj_device: None,
            flash_attn_type: FlashAttnType::Auto,
            warmup: true,
            image_min_tokens: -1,
            image_max_tokens: -1,
            chat_template: String::new(),
            use_jinja: false,
            default_template_kwargs: String::new(),
            simple_io: false,
            use_color: false,
            video_ffmpeg_bin_dir: String::new(),
            video_fps: 4.0,
            video_timestamp_interval_ms: 5000,
        }
    }
}

/// `g_is_generating` / `g_is_interrupted` plus the second-SIGINT exit request.
#[derive(Debug, Clone, Copy, Default)]
pub struct InterruptState {
    pub is_generating: bool,
    pub is_interrupted: bool,
    pub exit_now: bool,
}

impl InterruptState {
    pub fn sigint_handler(&mut self, signo: i32) {
        if signo != 2 {
            return;
        }
        if self.is_generating {
            self.is_generating = false;
        } else if self.is_interrupted {
            self.exit_now = true;
        } else {
            self.is_interrupted = true;
        }
    }
}

pub fn show_additional_info(program: &str) {
    log_inf(&format!(
        "Experimental CLI for multimodal\n\n\
Usage: {} [options] -m <model> --mmproj <mmproj> --image <image> --audio <audio> -p <prompt>\n\n\
  -m and --mmproj are required\n\
  -hf user/repo can replace both -m and --mmproj in most cases\n\
  --image, --audio and -p are optional, if NOT provided, the CLI will run in chat mode\n\
  to disable using GPU for mmproj model, add --no-mmproj-offload\n",
        program
    ));
}

/// Test-only response marker used by `tests.sh`.
pub fn inject_test_response_marker() {
    if let Ok(marker) = env::var("MTMD_TEST_RESPONSE_MARKER") {
        log_inf(&format!("{}\n", marker));
    }
}

/// `mtmd_cli_context`. Model, projector and sampler are supplied by the host
/// because their constructors need the ggml backend.
pub struct CliContext<M, L, S, T>
where
    M: Mtmd,
    L: Llama,
    S: Sampler,
{
    pub ctx_vision: M,
    pub cap: Cap,
    pub lctx: L,
    pub smpl: S,
    pub batch: Batch,
    pub n_batch: i32,
    pub bitmaps: Vec<super::types::Bitmap>,
    pub videos: Vec<Video>,
    pub init_opt: InitOpt,
    pub video_ffmpeg_bin_dir: String,
    pub tmpls: T,
    pub chat_history: Vec<ChatMessage>,
    pub use_jinja: bool,
    pub antiprompt_tokens: Vec<Token>,
    pub n_threads: i32,
    pub n_past: i32,
}

impl<M, L, S, T> CliContext<M, L, S, T>
where
    M: Mtmd,
    L: Llama,
    S: Sampler,
    T: ChatTemplates,
{
    /// Constructor body after `common_init_from_params`.
    pub fn new(
        params: &Params,
        lctx: L,
        ctx_vision: M,
        smpl: S,
        tmpls: T,
        mem: &MemoryUsage,
    ) -> Result<Self, i32> {
        if !lctx.model_exists() {
            return Err(1);
        }

        let mut mparams = ContextParams {
            use_gpu: params.mmproj_use_gpu,
            device: params.mmproj_device.clone(),
            print_timings: true,
            n_threads: params.n_threads,
            flash_attn_type: params.flash_attn_type,
            warmup: params.warmup,
            image_min_tokens: params.image_min_tokens,
            image_max_tokens: params.image_max_tokens,
            ..ContextParams::default()
        };
        // `mparams` is consumed by the host-side `mtmd_init_from_file` call.
        clamp_image_tokens(&mut mparams, mem, lctx.n_ubatch());

        let cap = get_capabilities(&lctx, Some(&ctx_vision));
        if cap.audio_duplex {
            log_err("Full duplex models only run in llama-server, see the /v1/realtime endpoint\n");
            return Err(1);
        }
        if !cap.chat && !cap.audio_stream {
            log_err("Model does not support chat mode\n");
            log_err("Hint: for TTS models, please use llama-tts\n");
            return Err(1);
        }
        if !lctx.model_has_chat_template() && params.chat_template.is_empty() {
            log_err("Model does not have chat template.\n");
            log_err("  For old llava models, you may need to use '--chat-template vicuna'\n");
            log_err("  For MobileVLM models, use '--chat-template deepseek'\n");
            log_err("  For Mistral Small 3.1, use '--chat-template mistral-v7'\n");
            return Err(1);
        }

        log_inf(&format!(
            "new: chat template example:\n{}\n",
            tmpls.format_example(params.use_jinja, &params.default_template_kwargs)
        ));

        let antiprompt_tokens = if params.chat_template == "vicuna" {
            lctx.tokenize("ASSISTANT:", false, true)
        } else if params.chat_template == "deepseek" {
            lctx.tokenize("###", false, true)
        } else {
            Vec::new()
        };

        let mut init_opt = InitOpt::default();
        init_opt.video_params.fps_target = params.video_fps;
        init_opt.video_params.timestamp_interval_ms = params.video_timestamp_interval_ms;
        init_opt.video_params.ffmpeg_bin_dir = params.video_ffmpeg_bin_dir.clone();

        Ok(Self {
            ctx_vision,
            cap,
            lctx,
            smpl,
            batch: Batch::default(),
            n_batch: params.n_batch,
            bitmaps: Vec::new(),
            videos: Vec::new(),
            init_opt,
            video_ffmpeg_bin_dir: params.video_ffmpeg_bin_dir.clone(),
            tmpls,
            chat_history: Vec::new(),
            use_jinja: params.use_jinja,
            antiprompt_tokens,
            n_threads: params.n_threads,
            n_past: 0,
        })
    }

    pub fn check_antiprompt(&self, generated_tokens: &[Token]) -> bool {
        let n = self.antiprompt_tokens.len();
        if n == 0 || generated_tokens.len() < n {
            return false;
        }
        let start = generated_tokens.len() - n;
        generated_tokens[start..] == self.antiprompt_tokens[..]
    }

    pub fn load_media(&mut self, res: HelperBitmapWrapper) -> bool {
        match res.bitmap {
            Some(bitmap) => {
                self.bitmaps.push(bitmap);
                if let Some(video) = res.video {
                    self.videos.push(video);
                }
                true
            }
            None => false,
        }
    }

    fn chat_add_and_format(&mut self, new_msg: ChatMessage) -> String {
        log_msg(
            LogLevel::Debug,
            &format!(
                "chat_add_and_format: new_msg.role='{}', new_msg.content='{}'\n",
                new_msg.role, new_msg.content
            ),
        );
        let is_user = new_msg.role == "user";
        let formatted = self
            .tmpls
            .format_single(&self.chat_history, &new_msg, is_user, self.use_jinja);
        self.chat_history.push(new_msg);
        formatted
    }

    fn eval_message(&mut self, msg: ChatMessage, state: &InterruptState) -> i32 {
        inject_test_response_marker();

        let add_bos = self.chat_history.is_empty();
        let formatted_chat = self.chat_add_and_format(msg);
        log_msg(
            LogLevel::Debug,
            &format!("formatted_chat.prompt: {}\n", formatted_chat),
        );
        if state.is_interrupted {
            return 0;
        }

        // Split on the marker; the CLI owns the interleaving so it uses
        // mtmd_tokenize_from_parts instead of mtmd_tokenize.
        let marker = super::types::default_marker();
        let mut segments = Vec::new();
        let mut start = 0;
        while let Some(pos) = formatted_chat[start..].find(marker) {
            let abs = start + pos;
            segments.push(formatted_chat[start..abs].to_owned());
            start = abs + marker.len();
        }
        segments.push(formatted_chat[start..].to_owned());

        if segments.len() - 1 != self.bitmaps.len() {
            log_err(&format!(
                "Number of media markers ({}) does not match number of loaded media ({})\n",
                segments.len() - 1,
                self.bitmaps.len()
            ));
            return 1;
        }

        let parts: Vec<InputPart> = segments
            .into_iter()
            .enumerate()
            .flat_map(|(i, segment)| {
                let mut v = Vec::with_capacity(2);
                v.push(InputPart::Text(InputText::new(segment, false, true)));
                if let Some(bitmap) = self.bitmaps.get(i) {
                    v.push(InputPart::Bitmap(bitmap.clone()));
                }
                v
            })
            .collect();

        let chunks = match self.ctx_vision.tokenize_from_parts(&parts, add_bos) {
            Ok(chunks) => chunks,
            Err(res) => {
                log_err(&format!("Unable to tokenize prompt, res = {}\n", res));
                return 1;
            }
        };
        self.bitmaps.clear();
        self.videos.clear();

        let n_chunks = chunks.len();
        let mut mbatch: Option<i32> = None;
        for (i, chunk) in chunks.iter().enumerate() {
            if chunk.get_type() == super::types::InputChunkType::Text {
                let mut new_n_past = self.n_past;
                let ret = eval_chunk_single(
                    &mut self.ctx_vision,
                    &mut self.lctx,
                    chunk,
                    self.n_past,
                    0,
                    self.n_batch,
                    i == n_chunks - 1,
                    &mut new_n_past,
                );
                if ret != 0 {
                    log_err(&format!("Unable to eval text chunk {}\n", i));
                    return 1;
                }
                self.n_past = new_n_past;
                continue;
            }

            // Media chunk: reuse a batched embedding, or encode a new batch.
            let mut embd = match mbatch {
                Some(batch) => self.ctx_vision.batch_get_output_embd(batch, chunk),
                None => None,
            };
            if embd.is_none() {
                let batch = self.ctx_vision.batch_init();
                let ret = self.ctx_vision.batch_add_chunk(batch, chunk);
                if ret != 0 {
                    log_err(&format!("Unable to add media chunk {}, res = {}\n", i, ret));
                    return 1;
                }
                let mut n_added = 1;
                for next in chunks.iter().skip(i + 1) {
                    if next.get_type() == super::types::InputChunkType::Text {
                        break;
                    }
                    if self.ctx_vision.batch_add_chunk(batch, next) != 0 {
                        break;
                    }
                    n_added += 1;
                }
                log_inf(&format!(
                    "encoding mtmd batch, n_chunks = {} (done = {}, total = {})\n",
                    n_added, i, n_chunks
                ));
                let start = Instant::now();
                let ret = self.ctx_vision.batch_encode(batch);
                if ret != 0 {
                    log_err(&format!("Failed to encode mtmd batch, res = {}\n", ret));
                    return 1;
                }
                log_inf(&format!(
                    "mtmd batch encoding done in {} ms\n",
                    start.elapsed().as_millis()
                ));
                embd = self.ctx_vision.batch_get_output_embd(batch, chunk);
                if embd.is_none() {
                    log_err("Unable to get media embeddings\n");
                    return 1;
                }
                mbatch = Some(batch);
            }
            let embd = embd.unwrap();
            let mut new_n_past = self.n_past;
            let ret = super::helper::decode_image_chunk(
                &self.ctx_vision,
                &mut self.lctx,
                chunk,
                &embd,
                self.n_past,
                0,
                self.n_batch,
                &mut new_n_past,
                None,
            );
            if ret != 0 {
                log_err(&format!("Unable to decode media chunk {}\n", i));
                return 1;
            }
            self.n_past = new_n_past;
        }
        log("\n");
        0
    }

    fn generate_response(&mut self, n_predict: i32, state: &InterruptState) -> i32 {
        let mut generated_tokens = Vec::new();
        for _ in 0..n_predict {
            if !state.is_generating || state.is_interrupted || state.exit_now {
                log("\n");
                break;
            }

            let token_id = self.smpl.sample(self.lctx.logits());
            generated_tokens.push(token_id);
            self.smpl.accept(token_id, true);

            if self.lctx.vocab().is_eog(token_id) || self.check_antiprompt(&generated_tokens) {
                log("\n");
                break;
            }

            log(&self.lctx.token_to_piece(token_id));
            flush_stdout();

            if state.is_interrupted {
                log("\n");
                break;
            }

            self.batch.clear();
            let idx = self.batch.add_token(token_id, self.n_past);
            self.batch.set_output_logits(idx, true);
            self.n_past += 1;
            if self.lctx.process(&self.batch) != 0 {
                log_err("failed to decode token\n");
                return 1;
            }
        }

        let generated_text = self.lctx.detokenize(&generated_tokens);
        self.chat_history
            .push(ChatMessage::new("assistant", generated_text));
        0
    }

    fn run_audio_stream(
        &mut self,
        decoder: &impl MediaDecoder,
        fname: &str,
        n_predict: i32,
        state: &InterruptState,
    ) -> i32 {
        let res = bitmap_init_from_file(
            &self.ctx_vision,
            fname,
            false,
            &self.init_opt,
            decoder,
        );
        let bitmap = match res.bitmap {
            Some(bitmap) => bitmap,
            None => return 1, // error is already printed by libmtmd
        };
        if !bitmap.is_audio() {
            log_err(&format!("run_audio_stream: {} is not an audio file\n", fname));
            return 1;
        }
        let samples = bitmap.samples().unwrap_or(&[]);
        let rate = self.ctx_vision.audio_sample_rate();
        if rate <= 0 {
            return 1;
        }
        let sparams = AudioStreamParams::default();
        let mut stream = match self.ctx_vision.audio_stream_init(&sparams) {
            Some(stream) => stream,
            None => return 1,
        };

        let n_chunk = super::helper::stream_chunk_size(rate, sparams.step_ms);
        let n_embd = self.lctx.n_embd_inp() as usize;
        log(&format!(
            "Transcribing {:.1} s of audio, fed in chunks of {} ms\n\n",
            samples.len() as f64 / rate as f64,
            sparams.step_ms
        ));

        let t_start = Instant::now();
        let mut t_enc = 0u128;
        let mut t_dec = 0u128;
        let mut n_fed = 0usize;
        let mut n_gen = 0i32;
        let mut tok = super::types::TOKEN_NULL;

        while !state.is_interrupted && !state.exit_now {
            let t0 = Instant::now();
            let n = stream.step(tok);
            t_enc += t0.elapsed().as_micros();
            if n < 0 {
                if n == -2 {
                    log_err("run_audio_stream: stream error\n");
                    return 1;
                }
                break;
            }
            if n == 0 {
                if n_fed < samples.len() {
                    let cnt = min(n_chunk, samples.len() - n_fed);
                    let ret = stream.push(&samples[n_fed..n_fed + cnt]);
                    if ret != 0 {
                        break;
                    }
                    n_fed += cnt;
                } else if stream.flush(true) != 0 {
                    break;
                }
                continue;
            }

            let t1 = Instant::now();
            let rows = stream.output_embd();
            self.batch.clear();
            for i in 0..n as usize {
                let pos = [self.n_past; MROPE_SECTIONS];
                self.n_past += 1;
                let begin = i * n_embd;
                let end = begin + n_embd;
                if end > rows.len() {
                    break;
                }
                self.batch
                    .add_embd(0, &rows[begin..end], &pos, i == n as usize - 1);
            }
            if self.lctx.process(&self.batch) != 0 {
                log_err("run_audio_stream: failed to decode\n");
                return 1;
            }

            tok = self.smpl.sample(self.lctx.logits());
            self.smpl.accept(tok, true);
            t_dec += t1.elapsed().as_micros();
            if self.lctx.vocab().is_eog(tok) {
                break;
            }
            if !self.lctx.vocab().is_control(tok) {
                log(&self.lctx.token_to_piece(tok));
                flush_stdout();
            }
            n_gen += 1;
            if n_gen >= n_predict {
                break;
            }
        }

        let audio_s = samples.len() as f64 / rate as f64;
        let t_total = t_start.elapsed().as_secs_f64();
        log(&format!(
            "\n\nrun_audio_stream: {} tokens, {:.2} s audio in {:.2} s ({:.2}x realtime)\n",
            n_gen,
            audio_s,
            t_total,
            audio_s / t_total.max(f64::MIN_POSITIVE)
        ));
        log(&format!(
            "run_audio_stream: encoder {:.2} s, decoder {:.2} s, {:.1} ms per 80 ms step\n",
            t_enc as f64 / 1e6,
            t_dec as f64 / 1e6,
            (t_enc + t_dec) as f64 / 1e3 / n_gen.max(1) as f64
        ));
        self.lctx.perf_print();
        0
    }
}

fn eval_system_prompt<M, L, S, T>(
    params: &Params,
    ctx: &mut CliContext<M, L, S, T>,
    state: &InterruptState,
) -> i32
where
    M: Mtmd,
    L: Llama,
    S: Sampler,
    T: ChatTemplates,
{
    if params.system_prompt.is_empty() {
        return 0;
    }
    ctx.eval_message(
        ChatMessage::new("system", params.system_prompt.clone()),
        state,
    )
}

/// `main` body after context construction.
pub fn run<M, L, S, T, C, D>(
    params: &mut Params,
    ctx: &mut CliContext<M, L, S, T>,
    state: &mut InterruptState,
    io: &mut C,
    decoder: &D,
) -> i32
where
    M: Mtmd,
    L: Llama,
    S: Sampler,
    T: ChatTemplates,
    C: Console,
    D: MediaDecoder,
{
    if params.mmproj.is_empty() {
        show_additional_info("mtmd-cli");
        log_err("ERR: Missing --mmproj argument\n");
        return 1;
    }

    let n_predict = if params.n_predict < 0 {
        i32::MAX
    } else {
        params.n_predict
    };

    io.init(params.simple_io, params.use_color);
    if state.is_interrupted {
        return 130;
    }

    let transcribes = ctx.cap.audio_stream && !ctx.cap.chat;
    if transcribes {
        if params.images.len() != 1 {
            log_err("This model transcribes audio in realtime, pass exactly one --audio file\n");
            return 1;
        }
        return ctx.run_audio_stream(decoder, &params.images[0], n_predict, state);
    }

    log_wrn("WARN: This is an experimental CLI for testing multimodal capability.\n");
    log_wrn("      For normal use cases, please use the standard llama-cli\n");

    if eval_system_prompt(params, ctx, state) != 0 {
        return 1;
    }

    let is_single_turn = !params.prompt.is_empty() && !params.images.is_empty();
    if is_single_turn {
        state.is_generating = true;
        if !params.prompt.contains(super::types::default_marker()) {
            for _ in 0..params.images.len() {
                params.prompt.insert_str(0, super::types::default_marker());
            }
        }

        for image in &params.images {
            let res = bitmap_init_from_file(&ctx.ctx_vision, image, false, &ctx.init_opt, decoder);
            if !ctx.load_media(res) {
                return 1;
            }
        }
        if ctx.eval_message(ChatMessage::new("user", params.prompt.clone()), state) != 0 {
            return 1;
        }
        if !state.is_interrupted && ctx.generate_response(n_predict, state) != 0 {
            return 1;
        }
    } else {
        log("\n Running in chat mode, available commands:");
        if ctx.ctx_vision.support_vision() {
            log("\n   /image <path>    load an image");
        }
        if ctx.ctx_vision.support_audio() {
            log("\n   /audio <path>    load an audio");
        }
        if ctx.cap.video {
            log("\n   /video <path>    load a video");
        }
        log("\n   /clear           clear the chat history");
        log("\n   /quit or /exit   exit the program");
        log("\n");

        let mut content = String::new();
        while !state.is_interrupted && !state.exit_now {
            state.is_generating = false;
            log("\n> ");
            io.set_display(DisplayType::UserInput);
            let line = match io.readline() {
                Some(line) => line,
                None => break,
            };
            io.set_display(DisplayType::Reset);

            let line = line.trim().to_owned();
            if line.is_empty() {
                continue;
            }
            if line == "/quit" || line == "/exit" {
                break;
            }
            if line == "/clear" {
                ctx.n_past = 0;
                ctx.chat_history.clear();
                ctx.lctx.memory_clear();
                if eval_system_prompt(params, ctx, state) != 0 {
                    return 1;
                }
                log("Chat history cleared\n\n");
                continue;
            }
            state.is_generating = true;
            let is_image = line == "/image" || line.starts_with("/image ");
            let is_audio = line == "/audio" || line.starts_with("/audio ");
            let is_video = line == "/video" || line.starts_with("/video ");
            if is_image || is_audio || is_video {
                if line.len() < 8 {
                    log_err("ERR: Missing media filename\n");
                    continue;
                }
                let media_path = line[7..].trim().to_owned();
                let res = bitmap_init_from_file(&ctx.ctx_vision, &media_path, false, &ctx.init_opt, decoder);
                if ctx.load_media(res) {
                    log(&format!(
                        "{} {} loaded\n",
                        media_path,
                        if is_image {
                            "image"
                        } else if is_audio {
                            "audio"
                        } else {
                            "video"
                        }
                    ));
                    content.push_str(super::types::default_marker());
                }
                continue;
            }
            content.push_str(&line);
            let ret = ctx.eval_message(ChatMessage::new("user", content.clone()), state);
            if ret != 0 {
                return 1;
            }
            if state.is_interrupted {
                break;
            }
            if ctx.generate_response(n_predict, state) != 0 {
                return 1;
            }
            content.clear();
        }
    }

    if state.is_interrupted {
        log("\nInterrupted by user\n");
    }
    log("\n\n");
    ctx.lctx.perf_print();
    if state.is_interrupted {
        130
    } else {
        0
    }
}

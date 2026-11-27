//! Data types from `tools/mtmd/mtmd.h` and the structs in `mtmd-helper.h`.
//!
//! Pointer-only fields (ggml backends, eval callbacks, log callbacks) are
//! replaced by plain values: a device is a name, a callback belongs to the
//! host-side trait implementation. Everything else keeps the C field names.

/// Number of M-RoPE position sections (`GGML_MROPE_SECTIONS`).
pub const MROPE_SECTIONS: usize = 4;
pub type Token = i32;
pub type Pos = i32;
pub type SeqId = i32;
pub const TOKEN_NULL: Token = -1;

pub const DEFAULT_MEDIA_MARKER: &str = "<__media__>";

pub fn default_marker() -> &'static str {
    DEFAULT_MEDIA_MARKER
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputChunkType {
    #[default]
    Text,
    Image,
    Audio,
    Count,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoTemplate {
    #[default]
    Default,
    Lfm3,
    Lfm3Omni,
    Qwen3Vl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MediaType {
    #[default]
    Unknown,
    Image,
    Audio,
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlashAttnType {
    #[default]
    Auto,
    Disabled,
    Enabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenAudioType {
    #[default]
    None,
    Qwen3Tts,
    PocketTts,
    Soprano,
    Lfm3A,
    Chatterbox,
    Drama3,
    Zonos3,
    Yue2,
    MinimaxMusic3,
    FishS2,
    Moshi,
    Kokoro,
    TinyVoca,
    VoxCpm2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenProcessType {
    #[default]
    GenCode,
    GenWav,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenAudioOutType {
    Pcm,
    #[default]
    Wav,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenAudioOutMode {
    #[default]
    Auto,
    Text,
    Audio,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImagePosType {
    Mrope,
    #[default]
    Normal,
    SensenovaU1,
    HunyuanVl,
}

/// `mtmd_input_text`: `text_len` is derived from `text`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputText {
    pub text: String,
    pub add_special: bool,
    pub parse_special: bool,
}

impl InputText {
    pub fn new(text: impl Into<String>, add_special: bool, parse_special: bool) -> Self {
        Self {
            text: text.into(),
            add_special,
            parse_special,
        }
    }

    pub fn text_len(&self) -> usize {
        self.text.len()
    }
}

/// `common_chat_msg` fields used by the CLI.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// `mtmd_input_part`: exactly one of the two variants is present.
#[derive(Debug, Clone, PartialEq)]
pub enum InputPart<B = Bitmap> {
    Text(InputText),
    Bitmap(B),
}

impl<B> InputPart<B> {
    pub fn text(&self) -> Option<&InputText> {
        match self {
            Self::Text(t) => Some(t),
            Self::Bitmap(_) => None,
        }
    }

    pub fn bitmap(&self) -> Option<&B> {
        match self {
            Self::Text(_) => None,
            Self::Bitmap(b) => Some(b),
        }
    }
}

/// Owned replacement for the opaque `mtmd_bitmap`.
#[derive(Debug, Clone, PartialEq)]
pub enum BitmapData {
    /// RGB24 pixels, nx * ny * 3 bytes.
    Image(Vec<u8>),
    /// Mono F32 PCM at the model sample rate.
    Audio(Vec<f32>),
    /// F32 voice rows, nx = n_embd, ny = number of rows.
    Voice(Vec<f32>),
    /// Frame producer kept by the host.
    Lazy,
    /// Placeholder used for token counting.
    Empty,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bitmap {
    pub nx: u32,
    pub ny: u32,
    pub data: BitmapData,
    pub id: String,
    pub mergeable: bool,
    pub is_video: bool,
    pub transcript: String,
}

impl Bitmap {
    pub fn init(nx: u32, ny: u32, data: Option<Vec<u8>>) -> Self {
        Self {
            nx,
            ny,
            data: data.map_or(BitmapData::Empty, BitmapData::Image),
            id: String::new(),
            mergeable: false,
            is_video: false,
            transcript: String::new(),
        }
    }

    pub fn init_from_audio(n_samples: usize, data: Option<Vec<f32>>) -> Self {
        Self {
            nx: 1,
            ny: n_samples as u32,
            data: data.map_or(BitmapData::Empty, BitmapData::Audio),
            id: String::new(),
            mergeable: false,
            is_video: false,
            transcript: String::new(),
        }
    }

    pub fn init_voice(n_embd: u32, n_rows: u32, data: Option<Vec<f32>>) -> Self {
        Self {
            nx: n_embd,
            ny: n_rows,
            data: data.map_or(BitmapData::Empty, BitmapData::Voice),
            id: String::new(),
            mergeable: false,
            is_video: false,
            transcript: String::new(),
        }
    }

    pub fn init_lazy(id: impl Into<String>) -> Self {
        Self {
            nx: 0,
            ny: 0,
            data: BitmapData::Lazy,
            id: id.into(),
            mergeable: false,
            is_video: false,
            transcript: String::new(),
        }
    }

    pub fn get_nx(&self) -> u32 {
        self.nx
    }

    pub fn get_ny(&self) -> u32 {
        self.ny
    }

    pub fn data(&self) -> &[u8] {
        match &self.data {
            BitmapData::Image(v) => v,
            _ => &[],
        }
    }

    pub fn samples(&self) -> Option<&[f32]> {
        match &self.data {
            BitmapData::Audio(v) | BitmapData::Voice(v) => Some(v),
            _ => None,
        }
    }

    pub fn n_bytes(&self) -> usize {
        match &self.data {
            BitmapData::Image(v) => v.len(),
            BitmapData::Audio(v) | BitmapData::Voice(v) => v.len() * std::mem::size_of::<f32>(),
            _ => 0,
        }
    }

    pub fn is_audio(&self) -> bool {
        matches!(self.data, BitmapData::Audio(_))
    }

    pub fn is_voice(&self) -> bool {
        matches!(self.data, BitmapData::Voice(_))
    }

    pub fn get_id(&self) -> &str {
        &self.id
    }

    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_owned();
    }

    pub fn set_mergeable(&mut self, mergeable: bool) {
        self.mergeable = mergeable;
    }

    pub fn set_is_video(&mut self, is_video: bool) {
        self.is_video = is_video;
    }

    pub fn set_transcript(&mut self, transcript: &str) {
        self.transcript = transcript.to_owned();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecoderPos {
    pub t: u32,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Metadata of the media tokens produced by `mtmd_tokenize*`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageTokens {
    pub tokens: Vec<Token>,
    pub id: String,
    pub n_pos: Pos,
    pub pos_type: ImagePosType,
    pub nx: u32,
    /// Relative positions for layouts that are not simple M-RoPE / normal.
    pub rel_pos: Vec<DecoderPos>,
}

impl ImageTokens {
    pub fn get_n_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn get_id(&self) -> &str {
        &self.id
    }

    pub fn get_n_pos(&self) -> Pos {
        self.n_pos
    }

    pub fn get_decoder_pos(&self, pos_0: Pos, i: usize) -> DecoderPos {
        let p = pos_0 as u32;
        match self.pos_type {
            ImagePosType::Mrope => DecoderPos {
                t: p,
                x: p + (i % self.nx.max(1) as usize) as u32,
                y: p + (i / self.nx.max(1) as usize) as u32,
                z: 0,
            },
            ImagePosType::Normal => DecoderPos {
                t: p + i as u32,
                x: p + i as u32,
                y: p + i as u32,
                z: p + i as u32,
            },
            _ => {
                let mut out = self
                    .rel_pos
                    .get(i)
                    .copied()
                    .unwrap_or(DecoderPos::default());
                out.t = out.t.saturating_add(p);
                out.x = out.x.saturating_add(p);
                out.y = out.y.saturating_add(p);
                out.z = out.z.saturating_add(p);
                out
            }
        }
    }
}

/// Metadata-only copy of an `mtmd_input_chunk`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputChunk {
    pub type_: InputChunkType,
    pub tokens: Vec<Token>,
    pub image_tokens: Option<ImageTokens>,
    pub id: String,
    pub n_pos: Pos,
}

impl InputChunk {
    pub fn text(tokens: Vec<Token>, n_pos: Pos) -> Self {
        Self {
            type_: InputChunkType::Text,
            tokens,
            image_tokens: None,
            id: String::new(),
            n_pos,
        }
    }

    pub fn media(type_: InputChunkType, image_tokens: Option<ImageTokens>, n_pos: Pos) -> Self {
        let id = image_tokens
            .as_ref()
            .map_or(String::new(), |it| it.id.clone());
        Self {
            type_,
            tokens: Vec::new(),
            image_tokens,
            id,
            n_pos,
        }
    }

    pub fn get_type(&self) -> InputChunkType {
        self.type_
    }

    pub fn get_tokens_text(&self) -> &[Token] {
        &self.tokens
    }

    pub fn get_tokens_image(&self) -> Option<&ImageTokens> {
        self.image_tokens.as_ref()
    }

    pub fn get_n_tokens(&self) -> usize {
        match &self.image_tokens {
            Some(it) => it.get_n_tokens(),
            None => self.tokens.len(),
        }
    }

    pub fn get_id(&self) -> &str {
        &self.id
    }

    pub fn get_n_pos(&self) -> Pos {
        self.n_pos
    }

    /// `mtmd_input_chunk_copy`: a full value copy.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// `mtmd_input_chunk_get_placeholder`: metadata only, no token data.
    pub fn get_placeholder(&self) -> Self {
        Self {
            type_: self.type_,
            tokens: Vec::new(),
            image_tokens: None,
            id: self.id.clone(),
            n_pos: self.n_pos,
        }
    }

}

/// One entry of a `llama_batch_ext`-style batch.
#[derive(Debug, Clone)]
pub struct BatchEntry {
    pub seq_id: SeqId,
    pub token: Option<Token>,
    pub embd: Vec<f32>,
    pub pos: [Pos; MROPE_SECTIONS],
    pub n_pos: usize,
    pub output_logits: bool,
}

/// Owned form of `llama_batch_ext` used by the helper and the CLI.
#[derive(Debug, Clone, Default)]
pub struct Batch {
    pub entries: Vec<BatchEntry>,
}

impl Batch {
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn add_token(&mut self, seq_id: SeqId, token: Token) -> i32 {
        self.entries.push(BatchEntry {
            seq_id,
            token: Some(token),
            embd: Vec::new(),
            pos: [0; MROPE_SECTIONS],
            n_pos: 1,
            output_logits: false,
        });
        (self.entries.len() - 1) as i32
    }

    pub fn add_embd(
        &mut self,
        seq_id: SeqId,
        embd: &[f32],
        pos: &[Pos],
        output_logits: bool,
    ) -> i32 {
        let mut p = [0; MROPE_SECTIONS];
        let n = pos.len().min(MROPE_SECTIONS);
        p[..n].copy_from_slice(&pos[..n]);
        self.entries.push(BatchEntry {
            seq_id,
            token: None,
            embd: embd.to_vec(),
            pos: p,
            n_pos: n.max(1),
            output_logits,
        });
        (self.entries.len() - 1) as i32
    }

    pub fn set_pos(&mut self, idx: i32, pos: &[Pos]) {
        if let Some(entry) = self.entries.get_mut(idx as usize) {
            let n = pos.len().min(MROPE_SECTIONS);
            entry.pos[..n].copy_from_slice(&pos[..n]);
            entry.n_pos = n.max(1);
        }
    }

    pub fn set_output_logits(&mut self, idx: i32, value: bool) {
        if let Some(entry) = self.entries.get_mut(idx as usize) {
            entry.output_logits = value;
        }
    }

    pub fn n_tokens(&self) -> usize {
        self.entries.len()
    }
}

/// `mtmd_helper_embd_batch`, owned view of one decoded embedding slice.
#[derive(Debug, Clone)]
pub struct EmbdBatch {
    pub n_tokens: i32,
    pub embd: Vec<f32>,
    pub n_embd: i32,
    pub pos: Vec<Pos>,
    pub n_pos: i32,
    pub seq_id: SeqId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextParams {
    pub use_gpu: bool,
    /// Backend device name; `None` is the C `nullptr` default.
    pub device: Option<String>,
    pub print_timings: bool,
    pub n_threads: i32,
    pub image_marker: Option<String>,
    pub media_marker: String,
    pub flash_attn_type: FlashAttnType,
    pub warmup: bool,
    pub image_min_tokens: i32,
    pub image_max_tokens: i32,
    pub batch_max_tokens: i32,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            use_gpu: true,
            device: None,
            print_timings: true,
            n_threads: 4,
            image_marker: None,
            media_marker: default_marker().to_owned(),
            flash_attn_type: FlashAttnType::Auto,
            warmup: true,
            image_min_tokens: -1,
            image_max_tokens: -1,
            batch_max_tokens: 1024,
        }
    }
}

pub fn context_params_default() -> ContextParams {
    ContextParams::default()
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VideoConfig {
    pub tmpl: VideoTemplate,
    pub fps: f32,
    pub n_frames: i32,
    pub frame_width: i32,
    pub frame_height: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoStreamInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    /// -1 when unknown.
    pub n_frames: i32,
    /// -1 when unknown.
    pub duration: f32,
}

impl Default for VideoStreamInfo {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            fps: 0.0,
            n_frames: -1,
            duration: -1.0,
        }
    }
}

/// `mtmd_helper_video_info`: effective sampling info for a video.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub n_frames: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MediaMeta {
    pub type_: MediaType,
    pub width: u32,
    pub height: u32,
    pub sample_rate: i32,
    pub duration: f32,
    pub n_frames: i32,
}

impl MediaMeta {
    /// Initial value used by `mtmd_helper_get_media_meta`.
    pub fn unknown() -> Self {
        Self {
            type_: MediaType::Unknown,
            width: 0,
            height: 0,
            sample_rate: 0,
            duration: -1.0,
            n_frames: -1,
        }
    }
}

/// `mtmd_helper_video_init_params`. The decoder is passed separately to
/// `video_init` because trait objects cannot live in a copied C struct.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoInitParams {
    pub fps_target: f32,
    pub ffmpeg_bin_dir: String,
    pub timestamp_interval_ms: i64,
    pub no_soundtrack: bool,
}

impl Default for VideoInitParams {
    fn default() -> Self {
        Self {
            fps_target: 4.0,
            ffmpeg_bin_dir: String::new(),
            timestamp_interval_ms: 5000,
            no_soundtrack: false,
        }
    }
}

pub fn video_init_params_default() -> VideoInitParams {
    VideoInitParams::default()
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct InitOpt {
    pub video_params: VideoInitParams,
}

pub fn init_opt_default() -> InitOpt {
    InitOpt::default()
}

pub struct HelperBitmapWrapper {
    pub bitmap: Option<Bitmap>,
    pub video: Option<super::video::Video>,
}

/// `mtmd_helper_cap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cap {
    pub chat: bool,
    pub vision: bool,
    pub audio: bool,
    pub video: bool,
    pub audio_stream: bool,
    pub audio_duplex: bool,
    pub text_gen: bool,
    pub audio_gen: bool,
    pub audio_gen_spk_ref: bool,
}

/// `mtmd_memory_usage` without the backend pointer keys.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    pub backend_mem_usage: Vec<(String, usize)>,
    pub image_max_tokens: i32,
    pub use_non_causal: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Caps {
    pub inp_vision: bool,
    pub inp_audio: bool,
    pub gen_audio: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenAudioInfo {
    pub type_: GenAudioType,
    pub sample_rate: i32,
    pub n_channels: i32,
    /// Index into a host-provided voice table; `None` is C `nullptr`.
    pub model_variant: Option<usize>,
    pub n_voices: i32,
}

impl Default for GenAudioInfo {
    fn default() -> Self {
        Self {
            type_: GenAudioType::None,
            sample_rate: 0,
            n_channels: 0,
            model_variant: None,
            n_voices: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioStreamParams {
    pub step_ms: i32,
    pub n_prompt: usize,
}

impl Default for AudioStreamParams {
    fn default() -> Self {
        Self {
            step_ms: 320,
            n_prompt: 0,
        }
    }
}

pub fn audio_stream_params_default() -> AudioStreamParams {
    AudioStreamParams::default()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenStateParams {
    pub seed: u32,
}

impl Default for GenStateParams {
    fn default() -> Self {
        Self { seed: u32::MAX }
    }
}

pub fn gen_state_params_default() -> GenStateParams {
    GenStateParams::default()
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenAudioInp {
    pub stream: bool,
    pub seq_id: SeqId,
    pub prompt: String,
    pub n_past: i32,
    pub out_mode: GenAudioOutMode,
    pub speaker_ref: Option<Bitmap>,
    pub speaker_ref_text: String,
    pub lang: String,
    pub top_k: i32,
    pub top_p: f32,
    pub temp: f32,
    pub seed: u32,
    pub speed: f32,
    pub loudness: f32,
    pub cfg_scale: f32,
    pub flow_steps: i32,
    pub out_type: GenAudioOutType,
}

impl Default for GenAudioInp {
    fn default() -> Self {
        Self {
            stream: false,
            seq_id: 0,
            prompt: String::new(),
            n_past: 0,
            out_mode: GenAudioOutMode::Auto,
            speaker_ref: None,
            speaker_ref_text: String::new(),
            lang: String::new(),
            top_k: 50,
            top_p: 1.0,
            temp: -1.0,
            seed: u32::MAX,
            speed: 1.0,
            loudness: 0.0,
            cfg_scale: 0.0,
            flow_steps: 0,
            out_type: GenAudioOutType::Wav,
        }
    }
}

pub fn gen_audio_inp_default() -> GenAudioInp {
    GenAudioInp::default()
}

/// `mtmd_gen_inp` data side; host callbacks fill the tensor-like fields.
#[derive(Debug, Clone, Default)]
pub struct GenInp {
    pub type_: GenProcessType,
    pub code0: i32,
    pub embd: Vec<f32>,
    pub top_k: i32,
    pub top_p: f32,
    pub temp: f32,
    pub penalty_repeat: f32,
    pub penalty_last_n: i32,
    pub codes: Vec<i32>,
    pub feats: Vec<f32>,
    pub speed: f32,
    pub backbone_state: Vec<u8>,
    pub backbone_n_past: i32,
    pub n_steps: i32,
    pub cfg_scale: f32,
}

pub fn gen_inp_default() -> GenInp {
    GenInp {
        speed: 1.0,
        penalty_repeat: 1.0,
        ..GenInp::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct GenOut {
    pub codes: Vec<i32>,
    pub feats: Vec<f32>,
    pub embd: Vec<f32>,
    pub is_eos: bool,
    pub audio: Vec<f32>,
    pub n_samples: usize,
}

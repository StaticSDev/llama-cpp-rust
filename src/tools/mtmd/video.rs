//! Video helpers from `mtmd-helper.cpp` and `mtmd-helper-video-ffmpeg.cpp`.
//!
//! Frame sequencing, index sampling, timestamps and soundtrack slicing are
//! ported in full. ffmpeg/ffprobe are only described as argument vectors:
//! spawning a process needs OS APIs outside this crate.

use std::cmp::min;

use super::log::{log_dbg, log_err};
use super::traits::{Mtmd, VideoDecoder};
use super::types::{
    Bitmap, VideoConfig, VideoInfo, VideoInitParams, VideoStreamInfo, VideoTemplate,
};

/// `std::nearbyint` with the default round-to-nearest-even mode.
fn round_ties_even(x: f64) -> i64 {
    let f = x.floor();
    let frac = x - f;
    if frac > 0.5 {
        f as i64 + 1
    } else if frac < 0.5 {
        f as i64
    } else if (f as i64) % 2 == 0 {
        f as i64
    } else {
        f as i64 + 1
    }
}

/// `mtmd_helper_video` / `mtmd_helper_video_seq` in one owned value.
pub struct Video {
    pub fps_target: f32,
    pub info: VideoInfo,
    pub id: String,
    pub src_alive: bool,
    pub current_frame: i32,
    pub prompt_start: String,
    pub timestamp_interval_ms: i64,
    pub next_timestamp_ms: f32,
    pub tmpl: VideoTemplate,
    pub target_frames: i32,
    pub out_width: i32,
    pub out_height: i32,
    pub frame_indices: Vec<i32>,
    pub next_index: usize,
    pub src_frame: i32,
    pub src_fps: f64,
    pub frame_buf: Vec<u8>,
    pub pending_text: String,
    pub pending_frame: Option<Bitmap>,
    pub n_src_frames: i32,
    pub audio_rate: i32,
    pub soundtrack: Vec<f32>,
    pub pending_audio: Option<Bitmap>,
    pub start_emitted: bool,

    decoder: Box<dyn VideoDecoder>,
    state: Option<usize>,
}

/// Name used by the C++ test API for the same struct.
pub type VideoSeq = Video;

impl Video {
    fn new(decoder: Box<dyn VideoDecoder>) -> Self {
        Self {
            fps_target: 0.0,
            info: VideoInfo::default(),
            id: String::new(),
            src_alive: false,
            current_frame: 0,
            prompt_start: "Video:".to_owned(),
            timestamp_interval_ms: 5000,
            next_timestamp_ms: 0.0,
            tmpl: VideoTemplate::Default,
            target_frames: 0,
            out_width: 0,
            out_height: 0,
            frame_indices: Vec::new(),
            next_index: 0,
            src_frame: 0,
            src_fps: 0.0,
            frame_buf: Vec::new(),
            pending_text: String::new(),
            pending_frame: None,
            n_src_frames: 0,
            audio_rate: 0,
            soundtrack: Vec::new(),
            pending_audio: None,
            start_emitted: false,
            decoder,
            state: None,
        }
    }

    /// `video_open`: decoder open plus validation of the source info.
    pub fn open(
        path: Option<&str>,
        buf: &[u8],
        params: &VideoInitParams,
        decoder: Box<dyn VideoDecoder>,
    ) -> Option<(Self, VideoStreamInfo)> {
        let mut src = VideoStreamInfo {
            width: 0,
            height: 0,
            fps: 0.0,
            n_frames: -1,
            duration: -1.0,
        };
        let state = decoder.open(path, buf, &mut src);
        if state.is_none() || src.width == 0 || src.height == 0 || src.fps <= 0.0 {
            log_err("video_open: the video decoder failed to open the video\n");
            return None;
        }
        let _ = params;
        let mut video = Self::new(decoder);
        video.state = state;
        Some((video, src))
    }

    pub fn get_info(&self) -> VideoInfo {
        self.info
    }

    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_owned();
    }

    pub fn ts_per_n_frames(&self) -> i32 {
        match self.tmpl {
            VideoTemplate::Lfm3 | VideoTemplate::Lfm3Omni | VideoTemplate::Qwen3Vl => 2,
            _ => 0,
        }
    }

    pub fn configure(&mut self, cfg: &VideoConfig, src: &VideoStreamInfo, fps_target_arg: f32) {
        self.tmpl = cfg.tmpl;
        let mut width = src.width;
        let mut height = src.height;
        let mut duration = src.duration;
        if duration < 0.0 && src.n_frames > 0 && src.fps > 0.0 {
            duration = src.n_frames as f32 / src.fps;
        }

        self.fps_target = if cfg.fps > 0.0 {
            cfg.fps
        } else if fps_target_arg > 0.0 {
            fps_target_arg
        } else {
            src.fps
        };

        if cfg.n_frames > 0 && duration > 0.0 {
            let n = cfg.n_frames;
            self.target_frames = n;
            self.fps_target = n as f32 / duration;

            // round(linspace(0, n_frames - 1, n))
            let step = if n > 1 {
                (src.n_frames - 1) as f64 / (n - 1) as f64
            } else {
                0.0
            };
            self.frame_indices = (0..n)
                .map(|i| round_ties_even(i as f64 * step) as i32)
                .collect();
            self.src_fps = src.fps as f64;
            self.n_src_frames = src.n_frames;

            if cfg.frame_width > 0
                && cfg.frame_height > 0
                && (cfg.frame_width != width as i32 || cfg.frame_height != height as i32)
            {
                self.out_width = cfg.frame_width;
                self.out_height = cfg.frame_height;
            }
        }

        if self.out_width > 0 && self.out_height > 0 {
            width = self.out_width as u32;
            height = self.out_height as u32;
        }

        self.info.width = width;
        self.info.height = height;
        self.info.fps = self.fps_target;
        self.info.n_frames = if self.target_frames > 0 {
            self.target_frames
        } else if duration > 0.0 {
            (duration * self.fps_target + 0.5) as i32
        } else {
            -1
        };
        self.frame_buf = vec![0; width as usize * height as usize * 3];

        log_dbg(&format!(
            "configure: {}x{} fps={:.2} duration={:.2}s n_frames={} target_frames={}\n",
            width, height, self.fps_target, duration, src.n_frames, self.target_frames
        ));
    }

    /// Pick frames by index for a source that yields every frame.
    pub fn sample_by_index(&mut self, src: &VideoStreamInfo) {
        if !self.frame_indices.is_empty() || src.fps <= 0.0 {
            return;
        }
        let n_src = if src.n_frames > 0 {
            src.n_frames
        } else if src.duration > 0.0 {
            round_ties_even(src.duration as f64 * src.fps as f64) as i32
        } else {
            0
        };
        if self.info.n_frames <= 0 || n_src <= 0 {
            self.fps_target = src.fps;
            self.info.fps = src.fps;
            return;
        }
        self.frame_indices = (0..self.info.n_frames)
            .map(|i| {
                min(
                    n_src - 1,
                    round_ties_even(i as f64 * src.fps as f64 / self.fps_target as f64) as i32,
                )
            })
            .collect();
        self.src_fps = src.fps as f64;
        self.n_src_frames = n_src;
    }

    pub fn set_soundtrack(&mut self, pcm: &[f32], rate: i32) {
        if pcm.is_empty() || self.src_fps <= 0.0 {
            return;
        }
        let len = round_ties_even(self.n_src_frames as f64 / self.src_fps * rate as f64) as usize;
        self.audio_rate = rate;
        self.soundtrack = pcm[..min(pcm.len(), len)].to_vec();
    }

    fn read_raw_frame(&mut self, skip: bool) -> bool {
        let (state, decoder) = match self.state {
            Some(state) => (state, &self.decoder),
            None => return false,
        };
        let (width, height) = (self.info.width, self.info.height);
        if skip {
            decoder.read_frame(state, None, width, height)
        } else {
            self.frame_buf = vec![0; width as usize * height as usize * 3];
            decoder.read_frame(state, Some(&mut self.frame_buf), width, height)
        }
    }

    pub fn start(&mut self, src: &VideoStreamInfo, fps: f32) -> bool {
        let mut fps_picked = false;
        if let Some(state) = self.state {
            if !self.decoder.start(
                state,
                self.info.width,
                self.info.height,
                fps,
                &mut fps_picked,
            ) {
                log_err("start: the video decoder failed to start\n");
                return false;
            }
        }
        if fps > 0.0 && !fps_picked {
            self.sample_by_index(src);
        }
        true
    }

    pub fn load_soundtrack(&mut self, rate: i32) {
        if self.tmpl != VideoTemplate::Lfm3Omni
            || self.frame_indices.is_empty()
            || rate <= 0
            || self.state.is_none()
        {
            return;
        }
        let state = self.state.unwrap();
        if let Some(pcm) = self.decoder.read_audio(state, rate) {
            self.set_soundtrack(&pcm, rate);
        }
    }

    /// `soundtrack_slice`: audio for the group that ends with frame `idx`.
    pub fn soundtrack_slice(&self, idx: i32) -> Option<Bitmap> {
        let ts_per = self.ts_per_n_frames();
        if ts_per <= 0 || self.audio_rate <= 0 || self.src_fps <= 0.0 {
            return None;
        }
        let group = idx / ts_per;
        let n_groups = (self.frame_indices.len() as i32 + ts_per - 1) / ts_per;
        let group_start = |g: i32| -> usize {
            let i = (g * ts_per) as usize;
            let frame = match self.frame_indices.get(i) {
                Some(v) => *v,
                None => return self.soundtrack.len(),
            };
            let t = frame as f64 / self.src_fps;
            min(
                self.soundtrack.len(),
                round_ties_even(t * self.audio_rate as f64) as usize,
            )
        };
        let begin = if group == 0 { 0 } else { group_start(group) };
        let end = if group == n_groups - 1 {
            self.soundtrack.len()
        } else {
            group_start(group + 1)
        };
        if begin >= end {
            return None;
        }
        let mut slice = Bitmap::init_from_audio(end - begin, Some(self.soundtrack[begin..end].to_vec()));
        if !self.id.is_empty() {
            slice.set_id(&format!("{}+audio{}", self.id, group));
        }
        Some(slice)
    }

    pub fn read_next_frame(&mut self) -> Option<Bitmap> {
        if !self.src_alive {
            return None;
        }
        if !self.frame_indices.is_empty() {
            if self.next_index >= self.frame_indices.len() {
                return None;
            }
            let want = self.frame_indices[self.next_index];
            while self.src_frame <= want {
                let skip = self.src_frame < want;
                if !self.read_raw_frame(skip) {
                    self.src_alive = false;
                    return None;
                }
                self.src_frame += 1;
            }
            self.next_index += 1;
        } else if !self.read_raw_frame(false) {
            self.src_alive = false;
            return None;
        }

        log_dbg(&format!(
            "read_next_frame: frame {} read OK\n",
            self.current_frame
        ));
        let mut frame = Bitmap::init(
            self.info.width,
            self.info.height,
            Some(self.frame_buf.clone()),
        );
        frame.set_mergeable(true);
        frame.set_is_video(true);
        if !self.id.is_empty() {
            frame.set_id(&format!("{}+{}", self.id, self.current_frame));
        }
        self.current_frame += 1;
        Some(frame)
    }

    /// `mtmd_helper_video_read_next`: (status, bitmap, text).
    /// Status: 0 = item, -1 = EOF, -2 = error.
    pub fn read_next(&mut self) -> (i32, Option<Bitmap>, Option<String>) {
        if !self.pending_text.is_empty() {
            let text = std::mem::take(&mut self.pending_text);
            return (0, None, Some(text));
        }
        if let Some(bitmap) = self.pending_frame.take() {
            return (0, Some(bitmap), None);
        }
        if let Some(bitmap) = self.pending_audio.take() {
            return (0, Some(bitmap), None);
        }

        if !self.src_alive {
            return (if self.current_frame == 0 { -2 } else { -1 }, None, None);
        }

        if !self.start_emitted {
            self.start_emitted = true;
            if self.tmpl == VideoTemplate::Default && !self.prompt_start.is_empty() {
                return (0, None, Some(self.prompt_start.clone()));
            }
        }

        let frame = match self.read_next_frame() {
            Some(frame) => frame,
            None => return (-1, None, None),
        };

        let ts_per = self.ts_per_n_frames();
        if ts_per > 0 {
            let idx = self.current_frame - 1;
            let group_ends = idx % ts_per == ts_per - 1
                || idx + 1 == self.frame_indices.len() as i32;
            if !self.soundtrack.is_empty() && !self.frame_indices.is_empty() && group_ends {
                self.pending_audio = self.soundtrack_slice(idx);
            }
            if idx % ts_per == 0 {
                let fps = self.info.fps.max(f32::MIN_POSITIVE) as f64;
                let mut ts_s = (idx as f64 + 0.5 * (ts_per - 1) as f64) / fps;
                if !self.frame_indices.is_empty() && self.src_fps > 0.0 {
                    let first = idx as usize;
                    let last = min(first + ts_per as usize - 1, self.frame_indices.len() - 1);
                    ts_s = (self.frame_indices[first] as f64 / self.src_fps
                        + self.frame_indices[last] as f64 / self.src_fps)
                        / 2.0;
                }
                self.pending_frame = Some(frame);
                return (0, None, Some(format!("<{:.1} seconds>", ts_s)));
            }
            return (0, Some(frame), None);
        }

        if self.timestamp_interval_ms > 0 && self.info.fps > 0.0 {
            let elapsed_ms = (self.current_frame - 1) as f32 / self.info.fps * 1000.0;
            if elapsed_ms >= self.next_timestamp_ms {
                let elapsed_s = elapsed_ms / 1000.0;
                let minutes = (elapsed_s / 60.0) as i32;
                let seconds = elapsed_s - minutes as f32 * 60.0;
                self.pending_text = format!("[{}m{:.2}s]", minutes, seconds);
                self.next_timestamp_ms += self.timestamp_interval_ms as f32;
            }
        }
        (0, Some(frame), None)
    }
}

impl Drop for Video {
    fn drop(&mut self) {
        if let Some(state) = self.state {
            self.decoder.close(state);
        }
    }
}

/// `video_init`: configure and start a decoder for a whole video.
pub fn video_init<M: Mtmd>(
    mctx: &M,
    path: Option<&str>,
    buf: &[u8],
    params: &VideoInitParams,
    decoder: Box<dyn VideoDecoder>,
) -> Option<Video> {
    let (mut video, src) = Video::open(path, buf, params, decoder)?;
    video.timestamp_interval_ms = params.timestamp_interval_ms;
    let cfg = mctx.get_video_config(
        src.width as i32,
        src.height as i32,
        src.n_frames,
        src.fps,
    );
    video.configure(&cfg, &src, params.fps_target);
    let fps = if video.frame_indices.is_empty() {
        video.info.fps
    } else {
        0.0
    };
    if !video.start(&src, fps) {
        return None;
    }
    if !params.no_soundtrack {
        video.load_soundtrack(mctx.audio_sample_rate());
    }
    video.src_alive = true;
    Some(video)
}

/// WebP path in `mtmd_helper_bitmap_init_impl`: decode a single still frame.
pub fn decode_still_image(
    path: Option<&str>,
    buf: &[u8],
    placeholder: bool,
    params: &VideoInitParams,
    decoder: Box<dyn VideoDecoder>,
) -> Option<Bitmap> {
    let (mut video, src) = Video::open(path, buf, params, decoder)?;
    if placeholder {
        return Some(Bitmap::init(src.width, src.height, None));
    }
    let cfg = VideoConfig {
        tmpl: VideoTemplate::Default,
        fps: 0.0,
        n_frames: 0,
        frame_width: 0,
        frame_height: 0,
    };
    video.configure(&cfg, &src, 0.0);
    if !video.start(&src, 0.0) {
        return None;
    }
    video.src_alive = true;
    let mut frame = video.read_next_frame()?;
    frame.set_mergeable(false);
    frame.set_is_video(false);
    Some(frame)
}

/// `mtmd_helper_video_read_next` null-context case.
pub fn video_read_next(video: &mut Option<Video>) -> (i32, Option<Bitmap>, Option<String>) {
    match video {
        Some(v) => v.read_next(),
        None => (-2, None, None),
    }
}

// ---------------------------------------------------------------------------
// ffmpeg argument construction (pure data, no subprocess)
// ---------------------------------------------------------------------------

/// `resolve_bin` without the Windows `.exe` suffix rule.
pub fn resolve_bin(bin_dir: &str, name: &str) -> String {
    if bin_dir.is_empty() {
        return name.to_owned();
    }
    let mut result = bin_dir.to_owned();
    if !result.ends_with('/') && !result.ends_with('\\') {
        result.push('/');
    }
    result.push_str(name);
    result
}

pub fn parse_rational(s: &str) -> f32 {
    if let Some((num, den)) = s.split_once('/') {
        if let (Ok(num), Ok(den)) = (num.parse::<i32>(), den.parse::<i32>()) {
            if den > 0 {
                return num as f32 / den as f32;
            }
        }
    }
    s.parse::<f32>().unwrap_or(0.0)
}

/// Keys emitted by the `ffprobe -show_entries` command in `ffmpeg_video::probe`.
#[derive(Debug, Clone, Default)]
pub struct ProbeFields {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub n_frames: i32,
    pub duration: f32,
    pub start_time: f64,
    pub sar_num: i32,
    pub sar_den: i32,
    pub rotation: i32,
}

impl ProbeFields {
    pub fn new() -> Self {
        Self {
            sar_num: 1,
            sar_den: 1,
            ..Self::default()
        }
    }

    /// One `key=value` line of `default=noprint_wrappers=1` output.
    pub fn parse_line(&mut self, line: &str) {
        let line = line.trim_end_matches(['\n', '\r']);
        let (key, val) = match line.split_once('=') {
            Some(pair) => pair,
            None => return,
        };
        match key {
            "width" => self.width = val.parse().unwrap_or(0),
            "height" => self.height = val.parse().unwrap_or(0),
            "r_frame_rate" => self.fps = parse_rational(val),
            "nb_frames" if val != "N/A" => self.n_frames = val.parse().unwrap_or(0),
            "duration" if val != "N/A" => self.duration = val.parse().unwrap_or(0.0),
            "start_time" if val != "N/A" => self.start_time = val.parse().unwrap_or(0.0),
            "sample_aspect_ratio" => {
                if let Some((num, den)) = val.split_once(':') {
                    match (num.parse::<i32>(), den.parse::<i32>()) {
                        (Ok(n), Ok(d)) => {
                            self.sar_num = n;
                            self.sar_den = d;
                        }
                        _ => {
                            self.sar_num = 1;
                            self.sar_den = 1;
                        }
                    }
                } else {
                    self.sar_num = 1;
                    self.sar_den = 1;
                }
            }
            "rotation" => self.rotation = val.parse().unwrap_or(0),
            _ => {}
        }
    }

    /// Apply pixel aspect ratio and rotation; returns the C `probe()` result.
    pub fn apply(
        &self,
        info: &mut VideoStreamInfo,
        video_start_s: &mut f64,
        src_width: &mut u32,
        src_height: &mut u32,
    ) -> bool {
        *video_start_s = self.start_time;
        *src_width = self.width;
        *src_height = self.height;
        info.width = self.width;
        info.height = self.height;
        info.fps = self.fps;
        info.n_frames = self.n_frames;
        info.duration = self.duration;

        let sar = if self.sar_den > 0 {
            self.sar_num as f64 / self.sar_den as f64
        } else {
            1.0
        };
        if (0.1..=10.0).contains(&sar) {
            info.width = round_ties_even(info.width as f64 * sar) as u32;
        }
        if self.rotation % 180 == 90 || self.rotation % 180 == -90 {
            std::mem::swap(src_width, src_height);
            std::mem::swap(&mut info.width, &mut info.height);
        }
        info.width > 0 && info.height > 0 && info.fps > 0.0
    }
}

pub fn probe_args(ffprobe_bin: &str, from_buf: bool, path: &str) -> Vec<String> {
    let input = if from_buf { "cache:pipe:0" } else { path };
    let mut cmd = vec![
        ffprobe_bin.to_owned(),
        "-v".to_owned(),
        "quiet".to_owned(),
        "-show_entries".to_owned(),
        "stream=width,height,sample_aspect_ratio,r_frame_rate,nb_frames,duration,start_time:stream_side_data=rotation".to_owned(),
        "-select_streams".to_owned(),
        "v:0".to_owned(),
        "-of".to_owned(),
        "default=noprint_wrappers=1".to_owned(),
    ];
    if from_buf {
        cmd.push("-read_ahead_limit".to_owned());
        cmd.push("-1".to_owned());
    }
    cmd.push(input.to_owned());
    cmd
}

pub fn soundtrack_args(ffmpeg_bin: &str, from_buf: bool, path: &str, rate: i32) -> Vec<String> {
    let input = if from_buf { "cache:pipe:0" } else { path };
    let mut cmd = vec![ffmpeg_bin.to_owned(), "-nostdin".to_owned()];
    if from_buf {
        cmd.push("-read_ahead_limit".to_owned());
        cmd.push("-1".to_owned());
    }
    cmd.extend(
        [
            "-i", input, "-map", "0:a:0", "-vn", "-ac", "1", "-ar",
        ]
        .iter()
        .map(|s| (*s).to_owned()),
    );
    cmd.push(rate.to_string());
    cmd.extend(
        ["-f", "f32le", "pipe:1", "-loglevel", "error"]
            .iter()
            .map(|s| (*s).to_owned()),
    );
    cmd
}

pub fn start_args(
    ffmpeg_bin: &str,
    from_buf: bool,
    path: &str,
    width: u32,
    height: u32,
    fps: f32,
    src_width: u32,
    src_height: u32,
) -> Vec<String> {
    let input = if from_buf { "cache:pipe:0" } else { path };
    let mut cmd = vec![ffmpeg_bin.to_owned(), "-nostdin".to_owned()];
    if from_buf {
        cmd.push("-read_ahead_limit".to_owned());
        cmd.push("-1".to_owned());
    }
    cmd.push("-i".to_owned());
    cmd.push(input.to_owned());

    let scale = width != src_width || height != src_height;
    let vf = if fps > 0.0 {
        if scale {
            format!("fps={:.6},scale={}:{}", fps, width, height)
        } else {
            format!("fps={:.6}", fps)
        }
    } else if scale {
        format!(
            "format=rgb24,scale={}:{}:flags=bicubic+accurate_rnd+full_chroma_int:param0=0:param1=0.5",
            width, height
        )
    } else {
        String::new()
    };
    if !vf.is_empty() {
        cmd.push("-vf".to_owned());
        cmd.push(vf);
    }
    cmd.extend(
        [
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", "-loglevel", "error",
        ]
        .iter()
        .map(|s| (*s).to_owned()),
    );
    cmd
}

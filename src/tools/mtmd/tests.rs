//! Small checks for the pure ported helpers.

use super::gen::{tts_resolve_lang, write_wav16};
use super::helper::{clamp_image_tokens, get_n_pos, get_n_tokens, is_audio_file, is_webp_file};
use super::resampler::Resampler;
use super::types::{
    ContextParams, ImageTokens, InputChunk, InputChunkType, MemoryUsage, VideoTemplate,
};
use super::video::{parse_rational, start_args, ProbeFields};

#[test]
fn audio_and_webp_magic() {
    let wav = b"RIFF____WAVEfmt ";
    let mp3 = b"ID3\x03\x00\x00\x00\x00\x00\x00\x00\x00";
    let flac = b"fLaC____xxxx";
    let webp = b"RIFF____WEBPVP8 ";
    assert!(is_audio_file(wav));
    assert!(is_audio_file(mp3));
    assert!(is_audio_file(flac));
    assert!(!is_audio_file(webp));
    assert!(is_webp_file(webp));
    assert!(!is_webp_file(wav));
}

#[test]
fn chunk_token_counts() {
    let tokens = ImageTokens {
        tokens: vec![1, 2, 3],
        id: String::new(),
        n_pos: 3,
        pos_type: Default::default(),
        nx: 3,
        rel_pos: Vec::new(),
    };
    let chunks = vec![
        InputChunk::text(vec![1, 2], 2),
        InputChunk::media(InputChunkType::Image, Some(tokens), 3),
    ];
    assert_eq!(get_n_tokens(&chunks), 5);
    assert_eq!(get_n_pos(&chunks), 5);
    let placeholder = chunks[1].get_placeholder();
    assert_eq!(placeholder.get_n_tokens(), 0);
    assert_eq!(placeholder.get_n_pos(), 3);
}

#[test]
fn rational_and_probe_fields() {
    assert_eq!(parse_rational("30000/1001"), 30000.0 / 1001.0);
    assert_eq!(parse_rational("30"), 30.0);
    assert_eq!(parse_rational("bad"), 0.0);

    let mut fields = ProbeFields::new();
    for line in [
        "width=1920\n",
        "height=1080\n",
        "r_frame_rate=30/1\n",
        "sample_aspect_ratio=4:3\n",
        "rotation=90\n",
    ] {
        fields.parse_line(line);
    }
    let mut info = Default::default();
    let mut start = 0.0;
    let mut sw = 0;
    let mut sh = 0;
    assert!(fields.apply(&mut info, &mut start, &mut sw, &mut sh));
    assert_eq!(info.width, 1080);
    assert_eq!(info.height, 2560);
    assert_eq!(sw, 1080);
    assert_eq!(sh, 1920);
    assert_eq!(start, 0.0);
}

#[test]
fn resampler_linear() {
    assert!(Resampler::new(0, 16000).is_none());
    let mut rs = Resampler::new(2, 4).unwrap();
    let out = rs.process(&[0.0, 1.0]).unwrap();
    assert_eq!(out.len(), 4);
    assert_eq!(out[1], 0.5);
    assert_eq!(out[2], 1.0);

    let mut pass = Resampler::new(16000, 16000).unwrap();
    assert!(pass.is_passthrough());
    assert_eq!(pass.process(&[1.0, 2.0]).unwrap(), &[1.0, 2.0]);
}

#[test]
fn wav_header_and_lang() {
    let mut buf = Vec::new();
    assert!(write_wav16(&mut buf, &[0.0, 1.0, -1.0], 24000, 1));
    assert_eq!(&buf[..4], b"RIFF");
    assert_eq!(&buf[8..12], b"WAVE");
    assert_eq!(&buf[36..40], b"data");
    assert_eq!(tts_resolve_lang("zh"), "chinese");
    assert_eq!(tts_resolve_lang("xx"), "xx");
}

#[test]
fn token_clamp_and_template() {
    let mut mem = MemoryUsage {
        image_max_tokens: 4096,
        use_non_causal: true,
        ..MemoryUsage::default()
    };
    let mut params = ContextParams::default();
    params.image_min_tokens = 2048;
    assert!(clamp_image_tokens(&mut params, &mem, 1024));
    assert_eq!(params.image_max_tokens, 1024);
    assert_eq!(params.image_min_tokens, 1024);
    mem.use_non_causal = false;
    assert!(!clamp_image_tokens(&mut params, &mem, 1024));
    assert_eq!(VideoTemplate::Lfm3 as i32, 1);
    assert!(start_args("ffmpeg", false, "in.mp4", 224, 224, 4.0, 224, 224)
        .iter()
        .any(|arg| arg == "fps=4.000000"));
}

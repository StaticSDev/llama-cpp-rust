//! Unit tests for the pure grammar helpers.

use crate::grammar::parser::parse_char;
use crate::grammar::types::PartialUtf8;
use crate::grammar::utf8::decode_utf8;

#[test]
fn decode_utf8_handles_split_and_invalid_sequences() {
    // 'é' is C3 A9, split across two pieces
    let (cps, partial) = decode_utf8(&[0xC3], PartialUtf8::default());
    assert_eq!(cps, vec![0]);
    assert_eq!(partial.value, 3);
    assert_eq!(partial.n_remain, 1);

    let (cps, partial) = decode_utf8(&[0xA9], partial);
    assert_eq!(cps, vec![0xE9, 0]);
    assert_eq!(partial.n_remain, 0);

    // The reference validates continuations only when resuming a partial
    // sequence; a fresh sequence consumes the next byte as-is.
    let (cps, partial) = decode_utf8(&[0xC3, 0x41], PartialUtf8::default());
    assert_eq!(cps, vec![193, 0]);
    assert_eq!(partial.n_remain, 0);

    // Resuming a partial sequence with a non-continuation byte aborts.
    let (cps, bad) = decode_utf8(
        &[0x41],
        PartialUtf8 {
            value: 3,
            n_remain: 1,
        },
    );
    assert_eq!(cps, vec![0]);
    assert_eq!(bad.n_remain, -1);
}

#[test]
fn parse_char_decodes_escapes() {
    let src = b"\\x41\\u00e9\\n\\]x";
    let (v, p) = parse_char(src, 0).unwrap();
    assert_eq!((v, p), (0x41, 4));
    let (v, p) = parse_char(src, 4).unwrap();
    assert_eq!((v, p), (0xe9, 10));
    let (v, p) = parse_char(src, 10).unwrap();
    assert_eq!((v, p), (b'\n' as u32, 12));
    let (v, p) = parse_char(src, 12).unwrap();
    assert_eq!((v, p), (b']' as u32, 14));
}

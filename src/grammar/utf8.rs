//! UTF-8 decoding and partial-sequence state.

use crate::grammar::types::PartialUtf8;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// C++ strings are NUL terminated; treat a byte past the end as NUL.
pub(crate) fn byte_at(src: &[u8], pos: usize) -> u8 {
    src.get(pos).copied().unwrap_or(0)
}

pub(crate) fn tail(src: &[u8], pos: usize) -> String {
    String::from_utf8_lossy(&src[pos.min(src.len())..]).into_owned()
}

// NOTE: assumes valid utf8 (but checks for overrun)
pub(crate) fn decode_utf8_char(src: &[u8], pos: usize) -> (u32, usize) {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    let first_byte = byte_at(src, pos);
    let highbits = (first_byte >> 4) as usize;
    let len = LOOKUP[highbits];
    let mask = ((1u32 << (8 - len)) - 1) as u8;
    let mut value = (first_byte & mask) as u32;
    let end = (pos + len).min(src.len()); // may overrun in C++, clamped here
    let mut p = pos + 1;
    while p < end && byte_at(src, p) != 0 {
        value = (value << 6) + (byte_at(src, p) & 0x3F) as u32;
        p += 1;
    }
    (value, p)
}

/// Port of `llama_grammar_decode_utf8`. The returned code points end in 0.
pub fn decode_utf8(src: &[u8], partial_start: PartialUtf8) -> (Vec<u32>, PartialUtf8) {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4];
    let mut pos = 0usize;
    let mut code_points = Vec::with_capacity(src.len() + 1);

    let mut value = partial_start.value;
    let mut n_remain = partial_start.n_remain;

    // continue previous decode, if applicable
    while byte_at(src, pos) != 0 && n_remain > 0 {
        let next_byte = byte_at(src, pos);
        if (next_byte >> 6) != 2 {
            // invalid sequence, abort
            code_points.push(0);
            return (
                code_points,
                PartialUtf8 {
                    value: 0,
                    n_remain: -1,
                },
            );
        }
        value = (value << 6) + (next_byte & 0x3F) as u32;
        pos += 1;
        n_remain -= 1;
    }

    if partial_start.n_remain > 0 && n_remain == 0 {
        code_points.push(value);
    }

    // decode any subsequent utf-8 sequences, which may end in an incomplete one
    while byte_at(src, pos) != 0 {
        let first_byte = byte_at(src, pos);
        let highbits = (first_byte >> 4) as usize;
        n_remain = LOOKUP[highbits] as i32 - 1;

        if n_remain < 0 {
            // invalid sequence, abort
            code_points.clear();
            code_points.push(0);
            return (code_points, PartialUtf8 { value: 0, n_remain });
        }

        let mask = ((1u32 << ((7 - n_remain) as u32)) - 1) as u8;
        value = (first_byte & mask) as u32;

        pos += 1;
        while byte_at(src, pos) != 0 && n_remain > 0 {
            value = (value << 6) + (byte_at(src, pos) & 0x3F) as u32;
            pos += 1;
            n_remain -= 1;
        }
        if n_remain == 0 {
            code_points.push(value);
        }
    }
    code_points.push(0);

    (code_points, PartialUtf8 { value, n_remain })
}

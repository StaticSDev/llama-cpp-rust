//! Port of `unicode.cpp`.
//!
//! Text arguments are byte slices, as in C++ `std::string`. Codepoint tables
//! live in `unicode_data`; the hash maps of the original are sorted arrays
//! here, so lookups use binary search instead of `std::collections`.

pub mod unicode_data;

mod bpe;
mod case;
mod flags;
mod normalize;

#[cfg(test)]
mod tests;

pub use bpe::*;
pub use case::*;
pub use flags::*;
pub use normalize::*;

use crate::core::{Error, Result};

fn invalid_character() -> Error {
    Error::new("invalid character")
}

/// Length in bytes of the UTF-8 sequence starting with `src`.
pub fn unicode_len_utf8(src: u8) -> usize {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[(src >> 4) as usize]
}

pub fn unicode_cpt_to_utf8(cpt: u32) -> Result<String> {
    if cpt <= 0x7f {
        return Ok(char::from(cpt as u8).to_string());
    }
    if cpt <= 0x7ff {
        return Ok(String::from_utf8(vec![
            (0xc0 | ((cpt >> 6) & 0x1f)) as u8,
            (0x80 | (cpt & 0x3f)) as u8,
        ])
        .unwrap());
    }
    if cpt <= 0xffff {
        return Ok(String::from_utf8(vec![
            (0xe0 | ((cpt >> 12) & 0x0f)) as u8,
            (0x80 | ((cpt >> 6) & 0x3f)) as u8,
            (0x80 | (cpt & 0x3f)) as u8,
        ])
        .unwrap());
    }
    if cpt <= 0x10ffff {
        return Ok(String::from_utf8(vec![
            (0xf0 | ((cpt >> 18) & 0x07)) as u8,
            (0x80 | ((cpt >> 12) & 0x3f)) as u8,
            (0x80 | ((cpt >> 6) & 0x3f)) as u8,
            (0x80 | (cpt & 0x3f)) as u8,
        ])
        .unwrap());
    }
    Err(Error::new("invalid codepoint"))
}

fn utf8_byte(utf8: &[u8], pos: usize) -> u8 {
    utf8[pos]
}

/// Decode one codepoint at `offset` and advance the offset.
pub fn unicode_cpt_from_utf8(utf8: &[u8], offset: &mut usize) -> Result<u32> {
    if *offset >= utf8.len() {
        return Err(invalid_character());
    }

    let b0 = utf8_byte(utf8, *offset);
    if b0 & 0x80 == 0 {
        *offset += 1;
        return Ok(b0 as u32);
    }
    if b0 & 0x40 == 0 {
        return Err(invalid_character());
    }

    let is_not_cont = |pos: usize| (utf8_byte(utf8, *offset + pos) & 0xc0) != 0x80;

    if b0 & 0x20 == 0 {
        if *offset + 1 >= utf8.len() || is_not_cont(1) {
            return Err(invalid_character());
        }
        let result = ((b0 as u32 & 0x1f) << 6) | (utf8_byte(utf8, *offset + 1) as u32 & 0x3f);
        if result < 0x80 {
            return Err(invalid_character());
        }
        *offset += 2;
        return Ok(result);
    }
    if b0 & 0x10 == 0 {
        if *offset + 2 >= utf8.len() || is_not_cont(1) || is_not_cont(2) {
            return Err(invalid_character());
        }
        let result = ((b0 as u32 & 0x0f) << 12)
            | ((utf8_byte(utf8, *offset + 1) as u32 & 0x3f) << 6)
            | (utf8_byte(utf8, *offset + 2) as u32 & 0x3f);
        if result < 0x800 {
            return Err(invalid_character());
        }
        *offset += 3;
        return Ok(result);
    }
    if b0 & 0x08 == 0 {
        if *offset + 3 >= utf8.len() || is_not_cont(1) || is_not_cont(2) || is_not_cont(3) {
            return Err(invalid_character());
        }
        let result = ((b0 as u32 & 0x07) << 18)
            | ((utf8_byte(utf8, *offset + 1) as u32 & 0x3f) << 12)
            | ((utf8_byte(utf8, *offset + 2) as u32 & 0x3f) << 6)
            | (utf8_byte(utf8, *offset + 3) as u32 & 0x3f);
        if !(0x10000..=0x10ffff).contains(&result) {
            return Err(invalid_character());
        }
        *offset += 4;
        return Ok(result);
    }
    Err(Error::new("failed to convert utf8 to codepoint"))
}

fn unicode_cpts_to_utf8(cps: &[u32]) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    for &cp in cps {
        result.extend_from_slice(unicode_cpt_to_utf8(cp)?.as_bytes());
    }
    Ok(result)
}

/// Decode all codepoints. Invalid UTF-8 is replaced by U+FFFD.
pub fn unicode_cpts_from_utf8(utf8: &[u8]) -> Vec<u32> {
    let mut result = Vec::with_capacity(utf8.len());
    let mut offset = 0;
    while offset < utf8.len() {
        match unicode_cpt_from_utf8(utf8, &mut offset) {
            Ok(cpt) => result.push(cpt),
            Err(_) => {
                offset += 1;
                result.push(0xfffd);
            }
        }
    }
    result
}

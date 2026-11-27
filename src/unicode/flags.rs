//! Codepoint flags and category lookups.

use crate::core::Result;
use std::sync::OnceLock;

use super::unicode_data::{
    MAX_CODEPOINTS, UNICODE_MAP_LOWERCASE, UNICODE_MAP_UPPERCASE, UNICODE_RANGES_FLAGS,
    UNICODE_RANGES_NFD, UNICODE_SET_WHITESPACE,
};
use super::unicode_cpt_from_utf8;

/// Flags for one Unicode codepoint. Layout matches the C++ bitfield order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnicodeCptFlags(u16);

impl UnicodeCptFlags {
    pub const UNDEFINED: u16 = 0x0001;
    pub const NUMBER: u16 = 0x0002;
    pub const LETTER: u16 = 0x0004;
    pub const SEPARATOR: u16 = 0x0008;
    pub const ACCENT_MARK: u16 = 0x0010;
    pub const PUNCTUATION: u16 = 0x0020;
    pub const SYMBOL: u16 = 0x0040;
    pub const CONTROL: u16 = 0x0080;
    pub const MASK_CATEGORIES: u16 = 0x00ff;
    pub const WHITESPACE: u16 = 0x0100;
    pub const LOWERCASE: u16 = 0x0200;
    pub const UPPERCASE: u16 = 0x0400;
    pub const NFD: u16 = 0x0800;

    pub const fn new(flags: u16) -> Self {
        Self(flags)
    }

    pub const fn as_uint(self) -> u16 {
        self.0
    }

    pub const fn category_flag(self) -> u16 {
        self.0 & Self::MASK_CATEGORIES
    }

    pub const fn is_undefined(self) -> bool {
        self.0 & Self::UNDEFINED != 0
    }

    pub const fn is_number(self) -> bool {
        self.0 & Self::NUMBER != 0
    }

    pub const fn is_letter(self) -> bool {
        self.0 & Self::LETTER != 0
    }

    pub const fn is_separator(self) -> bool {
        self.0 & Self::SEPARATOR != 0
    }

    pub const fn is_accent_mark(self) -> bool {
        self.0 & Self::ACCENT_MARK != 0
    }

    pub const fn is_punctuation(self) -> bool {
        self.0 & Self::PUNCTUATION != 0
    }

    pub const fn is_symbol(self) -> bool {
        self.0 & Self::SYMBOL != 0
    }

    pub const fn is_control(self) -> bool {
        self.0 & Self::CONTROL != 0
    }

    pub const fn is_whitespace(self) -> bool {
        self.0 & Self::WHITESPACE != 0
    }

    pub const fn is_lowercase(self) -> bool {
        self.0 & Self::LOWERCASE != 0
    }

    pub const fn is_uppercase(self) -> bool {
        self.0 & Self::UPPERCASE != 0
    }

    pub const fn is_nfd(self) -> bool {
        self.0 & Self::NFD != 0
    }

    fn set(&mut self, flag: u16, value: bool) {
        if value {
            self.0 |= flag;
        } else {
            self.0 &= !flag;
        }
    }
}

fn unicode_cpt_flags_array() -> Vec<UnicodeCptFlags> {
    let mut cpt_flags = vec![UnicodeCptFlags::new(UnicodeCptFlags::UNDEFINED); MAX_CODEPOINTS as usize];

    debug_assert_eq!(UNICODE_RANGES_FLAGS[0].0, 0);
    debug_assert_eq!(UNICODE_RANGES_FLAGS.last().unwrap().0, MAX_CODEPOINTS);
    for i in 1..UNICODE_RANGES_FLAGS.len() {
        let (ini, flags) = UNICODE_RANGES_FLAGS[i - 1];
        let end = UNICODE_RANGES_FLAGS[i].0;
        for cpt in ini..end {
            cpt_flags[cpt as usize] = UnicodeCptFlags::new(flags);
        }
    }
    for &cpt in UNICODE_SET_WHITESPACE {
        cpt_flags[cpt as usize].set(UnicodeCptFlags::WHITESPACE, true);
    }
    for &(_, to) in UNICODE_MAP_LOWERCASE {
        cpt_flags[to as usize].set(UnicodeCptFlags::LOWERCASE, true);
    }
    for &(_, to) in UNICODE_MAP_UPPERCASE {
        cpt_flags[to as usize].set(UnicodeCptFlags::UPPERCASE, true);
    }
    for range in UNICODE_RANGES_NFD {
        cpt_flags[range.nfd as usize].set(UnicodeCptFlags::NFD, true);
    }
    cpt_flags
}

static CPT_FLAGS: OnceLock<Vec<UnicodeCptFlags>> = OnceLock::new();

pub fn unicode_cpt_flags_from_cpt(cpt: u32) -> UnicodeCptFlags {
    let undef = UnicodeCptFlags::new(UnicodeCptFlags::UNDEFINED);
    CPT_FLAGS
        .get_or_init(unicode_cpt_flags_array)
        .get(cpt as usize)
        .copied()
        .unwrap_or(undef)
}

pub fn unicode_cpt_flags_from_utf8(utf8: &[u8]) -> Result<UnicodeCptFlags> {
    if utf8.is_empty() {
        return Ok(UnicodeCptFlags::new(UnicodeCptFlags::UNDEFINED));
    }
    let mut offset = 0;
    let cpt = unicode_cpt_from_utf8(utf8, &mut offset)?;
    Ok(unicode_cpt_flags_from_cpt(cpt))
}

pub fn unicode_cpt_is_han(cpt: u32) -> bool {
    (0x4e00..=0x9fff).contains(&cpt)
        || (0x3400..=0x4dbf).contains(&cpt)
        || (0x20000..=0x2a6df).contains(&cpt)
        || (0x2a700..=0x2b73f).contains(&cpt)
        || (0x2b740..=0x2b81f).contains(&cpt)
        || (0x2b820..=0x2ceaf).contains(&cpt)
        || (0x2ceb0..=0x2ebef).contains(&cpt)
        || (0xf900..=0xfaff).contains(&cpt)
        || (0x2f800..=0x2fa1f).contains(&cpt)
}

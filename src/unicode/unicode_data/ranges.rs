//! Codepoint range metadata.

pub const MAX_CODEPOINTS: u32 = 0x110000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeNfd {
    pub first: u32,
    pub last: u32,
    pub nfd: u32,
}

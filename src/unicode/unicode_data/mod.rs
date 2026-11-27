//! Data tables translated from `unicode-data.cpp` (generated with scripts/gen-unicode-data.py).
//! Map entries are sorted by key so Rust can use binary search instead of hash maps.

pub mod bpe;
pub mod case;
pub mod combining;
pub mod composition;
pub mod flags;
pub mod nfd;
pub mod ranges;
pub mod whitespace;

pub use bpe::*;
pub use case::*;
pub use combining::*;
pub use composition::*;
pub use flags::*;
pub use nfd::*;
pub use ranges::*;
pub use whitespace::*;

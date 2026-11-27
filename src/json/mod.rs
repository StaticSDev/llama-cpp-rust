//! Rust port of llama.cpp `common/json.h` and `common/json.cpp`.
//!
//! A thin JSON value layered over a small std-only tree. Many features of a
//! full JSON library are deliberately left out to keep this module small:
//! - object keys keep the order in which they are added
//! - every error is reported through `crate::core::Error`
//! - a nested object is built with `Json::object_from`, not a braced pair
//!
//! A `Value` that holds a tree is single use: converting it to `Json` moves
//! the tree out. A second conversion gives null.

use crate::core::{Error, Result};

mod from;
mod parser;
mod serializer;
mod value;

#[cfg(test)]
mod tests;

pub use from::{FromJson, IntoJson};
pub use value::*;

/// Port of `common_json_error`.
pub type JsonError = Error;

pub(crate) fn json_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::new(msg))
}

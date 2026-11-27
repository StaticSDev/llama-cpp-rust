//! Chat formatting and message helpers, ported from `src/llama-chat.{h,cpp}`
//! and `common/chat.{h,cpp}`.
//!
//! The C++ template route that goes through the Jinja engine, the PEG parser,
//! JSON and the model vocabulary has no std-only equivalent, so only the
//! legacy (built-in template) route and the data-only helpers are ported here.

pub mod detect;
pub mod format;
pub mod message;
pub mod template;

pub use detect::*;
pub use format::*;
pub use message::*;
pub use template::*;

#[cfg(test)]
mod tests;

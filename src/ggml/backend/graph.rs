//! Compute graph view shared by the backend interface and the scheduler.

pub use super::tensor::{ComputeGraph, GraphRef};

/// `fmt_size` from `ggml-backend.cpp`: compact byte size for debug output.
pub fn fmt_size(size: usize) -> String {
    if size >= 1024 * 1024 {
        format!("{}M", size / 1024 / 1024)
    } else {
        format!("{}K", size / 1024)
    }
}

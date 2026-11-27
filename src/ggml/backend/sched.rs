//! Backend scheduler data model, hooks and assignment helpers.
//!
//! Port of the scheduler structures, constants, callbacks and helper
//! functions in `ggml-backend.cpp`. Graph splitting and split execution
//! need the ggml graph optimizer and allocator, so those live with the
//! compute module rather than in the backend interface.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::core::{Error, Result};

use super::buffer::{self, BuftRef};
use super::graph::{fmt_size, GraphRef};
use super::stream::{self, BackendRef, EventRef};
use super::tensor::{self, TensorRef};
use super::types::{BufferUsage, SCHED_MAX_BACKENDS, SCHED_MAX_COPIES, SCHED_MAX_SPLIT_INPUTS};

// Prefixed aliases avoid collisions with the stream module at re-export time.
pub use self::new as sched_new;
pub use self::free as sched_free;
pub use self::reset as sched_reset;
pub use self::synchronize as sched_synchronize;

/// `ggml_backend_sched_eval_callback`.
pub type EvalCallback = Rc<dyn Fn(&TensorRef, bool) -> bool>;
/// `ggml_backend_sched_copy_callback`.
pub type CopyCallback = Rc<dyn Fn(&BackendRef, &TensorRef, &TensorRef, &GraphRef) -> bool>;
/// `ggml_backend_sched_realloc_callback`.
pub type ReallocCallback = Rc<dyn Fn(&SchedRef, &GraphRef, bool, bool)>;

/// `ggml_backend_sched_t`.
pub type SchedRef = Rc<RefCell<Scheduler>>;

/// `struct ggml_backend_sched_split`.
#[derive(Clone, Debug)]
pub struct SchedSplit {
    pub backend_id: i32,
    pub i_start: i32,
    pub i_end: i32,
    /// C++ uses a growable C array; `Vec` grows automatically.
    pub inputs: Vec<TensorRef>,
}

/// `struct ggml_backend_sched`.
pub struct Scheduler {
    pub is_reset: bool,
    pub is_alloc: bool,
    pub n_backends: usize,
    pub backends: Vec<BackendRef>,
    pub bufts: Vec<BuftRef>,
    pub node_backend_ids: Vec<i32>,
    pub leaf_backend_ids: Vec<i32>,
    pub prev_node_backend_ids: Vec<i32>,
    pub prev_leaf_backend_ids: Vec<i32>,
    /// Hash id (tensor pointer) to backend id; -1 means unassigned.
    pub tensor_backend_ids: HashMap<usize, i32>,
    pub splits: Vec<SchedSplit>,
    pub n_splits: i32,
    pub n_copies: i32,
    pub cur_copy: i32,
    pub next_copy: i32,
    pub events: Vec<Vec<Option<EventRef>>>,
    pub graph_inputs: Vec<TensorRef>,
    pub callback_eval: Option<EvalCallback>,
    pub callback_copy: Option<CopyCallback>,
    pub callback_realloc: Option<ReallocCallback>,
    pub op_offload: bool,
    pub debug: i32,
    pub graph_size: i32,
    pub prev_graph_size: i32,
    pub debug_realloc: i32,
    pub prefetch_events: Vec<Option<EventRef>>,
    pub prefetch_tried: Vec<bool>,
    pub fast_async_host_input: Vec<bool>,
    pub prefetched: Vec<TensorRef>,
    pub n_prefetched: i32,
    pub prefetch_split: i32,
    pub prefetch_backend_id: i32,
}

/// `ggml_backend_sched_backend_id`: lower id means higher priority.
pub fn backend_id(sched: &SchedRef, backend: &BackendRef) -> i32 {
    let s = sched.borrow();
    for (i, b) in s.backends.iter().enumerate() {
        if Rc::ptr_eq(b, backend) {
            return i as i32;
        }
    }
    -1
}

/// `ggml_backend_sched_backend_from_buffer`.
pub fn backend_from_buffer(sched: &SchedRef, tensor: &TensorRef, op: &TensorRef) -> i32 {
    let buf = match tensor::effective_buffer(tensor) {
        Some(b) => b,
        None => return -1,
    };
    let buft = Rc::clone(&buf.borrow().buft);
    let s = sched.borrow();
    for (i, backend) in s.backends.iter().enumerate() {
        if stream::supports_buft(backend, &buft) && stream::supports_op(backend, op) {
            return i as i32;
        }
    }
    -1
}

/// `ggml_backend_sched_backend_id_from_cur`: choose a backend from tensor
/// placement, graph input flags and weight locations.
pub fn backend_id_from_cur(sched: &SchedRef, tensor: &TensorRef) -> i32 {
    let cur = backend_from_buffer(sched, tensor, tensor);
    if cur != -1 {
        return cur;
    }

    let view_src = tensor.borrow().view_src.clone();
    if let Some(src) = &view_src {
        let cur = backend_from_buffer(sched, src, tensor);
        if cur != -1 {
            return cur;
        }
    }

    let has_buffer = tensor.borrow().buffer.is_some() || view_src.as_ref().map(|s| s.borrow().buffer.is_some()).unwrap_or(false);
    if has_buffer {
        // Pre-allocated tensors cannot move; C++ aborts here.
        return -1;
    }

    let is_input = tensor.borrow().flags & tensor::TENSOR_FLAG_INPUT != 0;
    if is_input {
        return sched.borrow().n_backends as i32 - 1;
    }

    let op = tensor.borrow().op;
    let allow = op != tensor::Op::Rope && op != tensor::Op::FlashAttnExt;
    if allow {
        let srcs = tensor.borrow().src.clone();
        for src in srcs.iter().flatten() {
            let src_buf = match tensor::effective_buffer(src) {
                Some(b) => b,
                None => continue,
            };
            if buffer::buffer_get_usage(&src_buf) != BufferUsage::Weights {
                continue;
            }
            let src_backend_id = backend_from_buffer(sched, src, tensor);
            let s = sched.borrow();
            if s.op_offload
                && src_backend_id == s.n_backends as i32 - 1
                && buffer::buffer_is_host(&src_buf)
            {
                for b in 0..src_backend_id as usize {
                    let backend = &s.backends[b];
                    if stream::supports_op(backend, tensor) && stream::offload_op(backend, tensor) {
                        return b as i32;
                    }
                }
            }
            return src_backend_id;
        }
    }

    -1
}

/// `ggml_backend_sched_buffer_supported`.
pub fn buffer_supported(sched: &SchedRef, t: &TensorRef, backend_id: i32) -> bool {
    let s = sched.borrow();
    if backend_id < 0 || backend_id as usize >= s.backends.len() {
        return false;
    }
    let buft = match tensor::effective_buffer(t) {
        Some(buf) => Rc::clone(&buf.borrow().buft),
        None => {
            let id = tensor_backend_id(sched, t);
            let id = if id == -1 {
                t.borrow()
                    .view_src
                    .as_ref()
                    .map(|v| tensor_backend_id(sched, v))
                    .unwrap_or(-1)
            } else {
                id
            };
            if id < 0 {
                return false;
            }
            Rc::clone(&s.bufts[id as usize])
        }
    };
    stream::supports_buft(&s.backends[backend_id as usize], &buft)
}

/// `ggml_backend_sched_set_if_supported`.
pub fn set_if_supported(sched: &SchedRef, node: &TensorRef, cur_backend_id: i32, node_backend_id: &mut i32) {
    let s = sched.borrow();
    if cur_backend_id < 0 || cur_backend_id as usize >= s.backends.len() {
        return;
    }
    if stream::supports_op(&s.backends[cur_backend_id as usize], node) {
        *node_backend_id = cur_backend_id;
    }
}

/// `ggml_backend_sched_is_host_weight`.
pub fn is_host_weight(t: &TensorRef) -> bool {
    match t.borrow().buffer.clone() {
        Some(buf) => {
            buffer::buffer_get_usage(&buf) == BufferUsage::Weights && buffer::buffer_is_host(&buf)
        }
        None => false,
    }
}

/// `ggml_backend_sched_new` constant portion. The graph allocator is owned by
/// the compute module and is therefore not constructed here.
pub fn new(
    backends: Vec<BackendRef>,
    bufts: Option<Vec<BuftRef>>,
    graph_size: usize,
    parallel: bool,
    op_offload: bool,
) -> Result<SchedRef> {
    if backends.is_empty() || backends.len() > SCHED_MAX_BACKENDS {
        return Err(Error::new(format!(
            "ggml_backend_sched_new: n_backends must be in 1..={SCHED_MAX_BACKENDS}"
        )));
    }

    let n_backends = backends.len();
    let n_copies = if parallel { SCHED_MAX_COPIES as i32 } else { 1 };
    let mut resolved_bufts = Vec::with_capacity(n_backends);

    for (i, backend) in backends.iter().enumerate() {
        let buft = match &bufts {
            Some(v) => Rc::clone(&v[i]),
            None => stream::backend_default_buft(backend)?,
        };
        if !stream::supports_buft(backend, &buft) {
            return Err(Error::new(format!(
                "ggml_backend_sched_new: backend {i} does not support its buffer type"
            )));
        }
        resolved_bufts.push(buft);
    }

    let nodes_size = graph_size + graph_size * SCHED_MAX_SPLIT_INPUTS * 2;
    let scheduler = Scheduler {
        is_reset: true,
        is_alloc: false,
        n_backends,
        backends,
        bufts: resolved_bufts,
        node_backend_ids: vec![-1; nodes_size],
        leaf_backend_ids: vec![-1; nodes_size],
        prev_node_backend_ids: vec![-1; nodes_size],
        prev_leaf_backend_ids: vec![-1; nodes_size],
        tensor_backend_ids: HashMap::new(),
        splits: Vec::with_capacity(16),
        n_splits: 0,
        n_copies,
        cur_copy: 0,
        next_copy: 0,
        events: Vec::new(),
        graph_inputs: Vec::with_capacity(SCHED_MAX_SPLIT_INPUTS),
        callback_eval: None,
        callback_copy: None,
        callback_realloc: None,
        op_offload,
        debug: 0,
        graph_size: 0,
        prev_graph_size: 0,
        debug_realloc: 0,
        prefetch_events: Vec::new(),
        prefetch_tried: Vec::new(),
        fast_async_host_input: Vec::new(),
        prefetched: Vec::with_capacity(SCHED_MAX_SPLIT_INPUTS),
        n_prefetched: 0,
        prefetch_split: -1,
        prefetch_backend_id: -1,
    };

    let sched = Rc::new(RefCell::new(scheduler));
    {
        let mut s = sched.borrow_mut();
        for b in 0..s.n_backends {
            let dev = stream::backend_device(&s.backends[b]);
            let caps = dev.as_ref().map(|d| stream::dev_get_props(d).caps).unwrap_or_default();
            s.fast_async_host_input.push(caps.fast_async_host_input);
            if s.n_copies > 1 {
                let mut copies = Vec::with_capacity(s.n_copies as usize);
                for _ in 0..s.n_copies {
                    copies.push(dev.as_ref().and_then(|d| stream::event_new(d)));
                }
                s.events.push(copies);
            }
            s.prefetch_events.push(dev.as_ref().and_then(|d| stream::event_new(d)));
            s.prefetch_tried.push(false);
        }
    }
    Ok(sched)
}

/// `ggml_backend_sched_free`: runs event hooks and drops references.
pub fn free(sched: &SchedRef) {
    let events: Vec<Vec<Option<EventRef>>> = {
        let mut s = sched.borrow_mut();
        std::mem::take(&mut s.events)
    };
    for copy in events {
        for event in copy.into_iter().flatten() {
            stream::event_free(&event);
        }
    }
    let prefetch = {
        let mut s = sched.borrow_mut();
        std::mem::take(&mut s.prefetch_events)
    };
    for event in prefetch.into_iter().flatten() {
        stream::event_free(&event);
    }
}

/// `ggml_backend_sched_reset`.
pub fn reset(sched: &SchedRef) {
    let mut s = sched.borrow_mut();
    if !s.is_reset {
        s.tensor_backend_ids.clear();
        s.is_reset = true;
    }
    s.is_alloc = false;
}

/// `tensor_backend_id(tensor)` macro.
pub fn tensor_backend_id(sched: &SchedRef, tensor: &TensorRef) -> i32 {
    let key = Rc::as_ptr(tensor) as usize;
    sched.borrow().tensor_backend_ids.get(&key).copied().unwrap_or(-1)
}

/// Set an explicit backend assignment for a tensor.
pub fn set_tensor_backend(sched: &SchedRef, tensor: &TensorRef, backend: &BackendRef) {
    let id = backend_id(sched, backend);
    if id < 0 {
        return;
    }
    let mut s = sched.borrow_mut();
    s.tensor_backend_ids.insert(Rc::as_ptr(tensor) as usize, id);
    s.is_reset = false;
}

/// `ggml_backend_sched_get_tensor_backend`.
pub fn get_tensor_backend(sched: &SchedRef, tensor: &TensorRef) -> Option<BackendRef> {
    let id = tensor_backend_id(sched, tensor);
    if id < 0 {
        return None;
    }
    sched.borrow().backends.get(id as usize).cloned()
}

/// `ggml_backend_sched_synchronize`.
pub fn synchronize(sched: &SchedRef) {
    let (backends, is_alloc) = {
        let s = sched.borrow();
        (s.backends.clone(), s.is_alloc)
    };
    for backend in &backends {
        stream::synchronize(backend);
    }
    if !is_alloc {
        sched.borrow_mut().next_copy = 0;
    }
}

/// `ggml_backend_sched_get_n_splits`.
pub fn n_splits(sched: &SchedRef) -> i32 {
    sched.borrow().n_splits
}

/// `ggml_backend_sched_get_n_copies`.
pub fn n_copies(sched: &SchedRef) -> i32 {
    sched.borrow().n_copies
}

/// `ggml_backend_sched_get_n_backends`.
pub fn n_backends(sched: &SchedRef) -> usize {
    sched.borrow().n_backends
}

/// `ggml_backend_sched_get_backend`.
pub fn get_backend(sched: &SchedRef, i: usize) -> Option<BackendRef> {
    sched.borrow().backends.get(i).cloned()
}

/// `ggml_backend_sched_get_buffer_type`.
pub fn get_buffer_type(sched: &SchedRef, backend: &BackendRef) -> Option<BuftRef> {
    let id = backend_id(sched, backend);
    if id < 0 {
        return None;
    }
    sched.borrow().bufts.get(id as usize).cloned()
}

/// Debug assignment line, using the same size format as C++.
pub fn print_assignment(sched: &SchedRef, graph: &GraphRef, index: usize) -> String {
    let g = graph.borrow();
    let node = g.nodes.get(index);
    let Some(node) = node else {
        return String::new();
    };
    let name = node.borrow().name.clone();
    let size = fmt_size(tensor::nbytes_ref(node));
    let backend = get_tensor_backend(sched, node)
        .map(|b| stream::backend_name(&b))
        .unwrap_or_else(|| "NULL".into());
    format!("node #{index}: {name} ({size}) [{backend}]")
}

/// `ggml_backend_sched_set_eval_callback`.
pub fn set_eval_callback(sched: &SchedRef, callback: Option<EvalCallback>) {
    sched.borrow_mut().callback_eval = callback;
}

/// `ggml_backend_sched_set_copy_callback`.
pub fn set_copy_callback(sched: &SchedRef, callback: Option<CopyCallback>) {
    sched.borrow_mut().callback_copy = callback;
}

/// `ggml_backend_sched_set_realloc_callback`.
pub fn set_realloc_callback(sched: &SchedRef, callback: Option<ReallocCallback>) {
    sched.borrow_mut().callback_realloc = callback;
}

/// `ggml_backend_sched_split_inputs_grow`: `Vec::push` grows automatically.
pub fn split_inputs_grow(split: &mut SchedSplit, tensor: TensorRef) {
    if split.inputs.len() == split.inputs.capacity() && !split.inputs.is_empty() {
        split.inputs.reserve(SCHED_MAX_SPLIT_INPUTS);
    }
    split.inputs.push(tensor);
}

/// `ggml_backend_sched_graph_inputs_grow`: `Vec::push` grows automatically.
pub fn graph_inputs_grow(sched: &SchedRef, tensor: TensorRef) {
    let mut s = sched.borrow_mut();
    if s.graph_inputs.len() == s.graph_inputs.capacity() && !s.graph_inputs.is_empty() {
        s.graph_inputs.reserve(SCHED_MAX_SPLIT_INPUTS);
    }
    s.graph_inputs.push(tensor);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_constants_and_split_growth() {
        assert_eq!(SCHED_MAX_BACKENDS, 16);
        assert_eq!(SCHED_MAX_SPLIT_INPUTS, 30);
        assert_eq!(SCHED_MAX_COPIES, 4);
        let mut split = SchedSplit {
            backend_id: 0,
            i_start: 0,
            i_end: 1,
            inputs: Vec::new(),
        };
        let t = Rc::new(RefCell::new(tensor::Tensor::default()));
        split_inputs_grow(&mut split, Rc::clone(&t));
        assert_eq!(split.inputs.len(), 1);
    }
}

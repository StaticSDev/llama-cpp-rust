//! Rust port of `ggml-backend.h`, `ggml-backend-impl.h`, `ggml-backend.cpp`
//! and the interface-level part of `ggml-backend-meta.cpp`.
//!
//! C++ virtual function tables become Rust traits in [`traits`]. Nullable C
//! pointers become `Option<Rc<...>>`; ownership follows the C++ graph, with
//! weak links where the C++ code held a non-owning back-pointer.
//!
//! Scope notes: dynamic-library loading (`ggml_backend_load*`) needs FFI and
//! is not ported. Graph splitting and split execution need the ggml compute
//! graph and allocator, so [`sched`] contains the scheduler data model,
//! constants, callbacks and assignment helpers only. Per-device tensor
//! slicing in the meta backend likewise belongs to the compute module.

pub mod buffer;
pub mod cpu;
pub mod graph;
pub mod meta;
pub mod registry;
pub mod sched;
pub mod stream;
pub mod tensor;
pub mod traits;
pub mod types;

pub use buffer::{
    alloc_buffer_n_plan, buffer_clear, buffer_copy_tensor, buffer_free, buffer_get_alignment,
    buffer_get_alloc_size, buffer_get_base, buffer_get_max_size, buffer_get_size, buffer_get_type,
    buffer_get_usage, buffer_init, buffer_init_tensor, buffer_is_multi, buffer_is_host,
    buffer_name, buffer_reset, buffer_set_usage, buft_alloc_buffer, buft_alloc_buffer_n,
    buft_get_alignment, buft_get_alloc_size, buft_get_alloc_size_n, buft_get_device,
    buft_get_max_size, buft_is_host, buft_name, clear_buffer_storage, default_alloc_buffer_n,
    default_alloc_size_n, multi_buffer_alloc_buffer, multi_buffer_set_usage, tensor_alloc,
    tensor_batch_is_2d, tensor_buft_groups, tensor_copy, tensor_copy_n, tensor_get, tensor_get_2d,
    tensor_get_n, tensor_memset, tensor_set, tensor_set_2d, tensor_set_n, view_init, AllocPlanItem,
    BackendBuffer, BufferKind, BufferRef, BufferType, BuftRef, EmptyBufferInterface,
    MultiBufferInterface,
};

pub use cpu::{
    cpu_buffer_from_ptr, cpu_buffer_from_ptr_type, cpu_buffer_type, CpuBackend, CpuBufferInterface,
    CpuBufferType, CpuDevice, CpuRegistry,
};

pub use graph::{fmt_size, ComputeGraph, GraphRef};

pub use meta::{
    backend_is_meta, buft_is_meta, dev_is_meta, meta_buffer_alloc, meta_buffer_n_bufs,
    meta_buffer_set_usage, meta_buffer_simple_buffer, meta_buft_n_bufts, meta_buft_simple_buft,
    meta_device, meta_device_reg, meta_dev_n_devs, meta_dev_simple_dev, meta_get_split_state,
    meta_n_backends, meta_simple_device, MetaBackend, MetaBufferInterface, MetaBufferType,
    MetaDevice, GetSplitState,
};

pub use registry::{
    dev_by_name, dev_by_type, dev_count, dev_get, device_register, init_best, init_by_name,
    init_by_type, reg_by_name, reg_count, reg_get, register, register_proc, BackendRegistry,
};

pub use sched::{
    backend_from_buffer, backend_id, backend_id_from_cur, buffer_supported, get_backend,
    get_buffer_type, get_tensor_backend, graph_inputs_grow, is_host_weight, n_backends, n_copies,
    n_splits, print_assignment, sched_free, sched_new, sched_reset, sched_synchronize,
    set_copy_callback, set_eval_callback, set_if_supported, set_realloc_callback, set_tensor_backend,
    split_inputs_grow, tensor_backend_id, CopyCallback, EvalCallback, ReallocCallback, SchedRef,
    SchedSplit, Scheduler,
};

pub use stream::{
    backend_alignment, backend_alloc_buffer, backend_default_buft, backend_device, backend_free,
    backend_guid, backend_max_size, backend_name, dev_backend_reg, dev_buffer_from_host_ptr,
    dev_buffer_type, dev_description, dev_get_props, dev_host_buffer_type, dev_init, dev_memory,
    dev_name, dev_offload_op, dev_supports_buft, dev_supports_op, dev_type, event_free, event_new,
    event_record, event_synchronize, event_wait, graph_compute, graph_compute_async,
    graph_optimize, graph_plan_compute, graph_plan_create, graph_plan_free, graph_plan_update,
    null_backend_name, offload_op, reg_dev_count, reg_dev_get, reg_get_proc_address, reg_name,
    synchronize, supports_buft, supports_op, tensor_copy_async, tensor_get_async, tensor_nbytes,
    tensor_set_2d_async, tensor_get_2d_async, Backend, BackendRef, Device, DeviceRef, Event,
    EventRef, GraphCopy, RegRef,
};

pub use tensor::{
    are_same_layout, data_location, effective_buffer, flash_attn_ext_has_indexed_bias, is_contiguous,
    is_view_op, n_dims, nbytes, nbytes_ref, op_alloc_size_may_expand, op_is_empty, pad, Op, Tensor,
    TensorData, TensorRef, TENSOR_FLAG_COMPUTE, TENSOR_FLAG_INPUT, TENSOR_FLAG_LOSS,
    TENSOR_FLAG_OUTPUT, TENSOR_FLAG_PARAM,
};

pub use traits::{
    AsAny, BackendBufferInterface, BackendInterface, BufferTypeInterface, DeviceInterface,
    GraphOptimizeParams, RegistryInterface, GraphPlan, Proc,
};

pub use types::{
    split_axis_name, BufferUsage, DevCaps, DevProps, DevType, Feature, SplitAxis, SplitState,
    Status, BACKEND_API_VERSION, DEFAULT_GRAPH_SIZE, MAX_DIMS, MAX_SRC, META_MAX_DEVICES,
    SCHED_MAX_BACKENDS, SCHED_MAX_COPIES, SCHED_MAX_SPLIT_INPUTS, TENSOR_ALIGNMENT,
};

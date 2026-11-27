//! `struct ggml_tensor` plus its inline op-parameter and name helpers.

use std::cell::{Ref, RefCell};
use std::fmt;
use std::sync::Arc;

use crate::core::{Error, Result};

use super::types::{
    ggml_blck_size, ggml_row_size, ggml_type_size, GgmlOp, GgmlType, GGML_MAX_DIMS, GGML_MAX_NAME,
    GGML_MAX_OP_PARAMS_I32, GGML_MAX_SRC,
};

/// Stand-in for the opaque `ggml_backend_buffer *` pointer.
pub type BufferHandle = Arc<dyn fmt::Debug + Send + Sync>;
/// Stand-in for the opaque `ggml_tensor::extra` pointer.
pub type ExtraHandle = Arc<dyn fmt::Debug + Send + Sync>;
/// Shared, mutable backing storage for a tensor and its views.
pub type TensorData = Arc<RefCell<Vec<u8>>>;

/// `struct ggml_tensor`.
///
/// Pointers become `Arc` handles: `src` and `view_src` are immutable because
/// graph nodes are only written during creation, while `data` is shared and
/// mutable so views alias the same bytes at `view_offs`.
#[derive(Clone)]
pub struct Tensor {
    pub type_: GgmlType,
    pub buffer: Option<BufferHandle>,
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
    pub op: GgmlOp,
    pub op_params: [i32; GGML_MAX_OP_PARAMS_I32],
    pub flags: i32,
    pub src: [Option<Arc<Tensor>>; GGML_MAX_SRC],
    pub view_src: Option<Arc<Tensor>>,
    pub view_offs: usize,
    pub data: Option<TensorData>,
    pub name: String,
    pub extra: Option<ExtraHandle>,
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            type_: GgmlType::F32,
            buffer: None,
            ne: [1; GGML_MAX_DIMS],
            nb: [0; GGML_MAX_DIMS],
            op: GgmlOp::None,
            op_params: [0; GGML_MAX_OP_PARAMS_I32],
            flags: 0,
            src: std::array::from_fn(|_| None),
            view_src: None,
            view_offs: 0,
            data: None,
            name: String::new(),
            extra: None,
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Avoid walking src/view_src recursively; print shape and op only.
        f.debug_struct("Tensor")
            .field("type", &self.type_)
            .field("ne", &self.ne)
            .field("nb", &self.nb)
            .field("op", &self.op)
            .field("flags", &self.flags)
            .field("view_src", &self.view_src.is_some())
            .field("view_offs", &self.view_offs)
            .field("src_count", &self.src.iter().filter(|s| s.is_some()).count())
            .field("name", &self.name)
            .finish()
    }
}

impl Tensor {
    /// `ggml_set_op_params_i32`.
    pub fn set_op_params_i32(&mut self, i: usize, value: i32) {
        assert!(i < GGML_MAX_OP_PARAMS_I32);
        self.op_params[i] = value;
    }

    /// `ggml_get_op_params_i32`.
    pub fn get_op_params_i32(&self, i: usize) -> i32 {
        assert!(i < GGML_MAX_OP_PARAMS_I32);
        self.op_params[i]
    }

    /// `ggml_set_op_params_f32`. Floats are stored as raw bits, as in C.
    pub fn set_op_params_f32(&mut self, i: usize, value: f32) {
        self.set_op_params_i32(i, value.to_bits() as i32);
    }

    /// `ggml_get_op_params_f32`.
    pub fn get_op_params_f32(&self, i: usize) -> f32 {
        f32::from_bits(self.get_op_params_i32(i) as u32)
    }

    /// Store a `size_t` op parameter across two `i32` slots.
    pub fn set_op_params_usize(&mut self, i: usize, value: usize) {
        self.set_op_params_i32(i, value as i32);
        self.set_op_params_i32(i + 1, (value >> 32) as i32);
    }

    /// Read a `size_t` op parameter stored by `set_op_params_usize`.
    pub fn get_op_params_usize(&self, i: usize) -> usize {
        let lo = self.get_op_params_i32(i) as u32 as usize;
        let hi = self.get_op_params_i32(i + 1) as u32 as usize;
        lo | (hi << 32)
    }

    /// Copy a slice of `int32_t` op parameters into `op_params`.
    pub fn set_op_params_i32_slice(&mut self, params: &[i32]) {
        assert!(params.len() <= GGML_MAX_OP_PARAMS_I32);
        self.op_params[..params.len()].copy_from_slice(params);
    }

    /// Copy a slice of float op parameters into `op_params`, bit-for-bit.
    pub fn set_op_params_f32_slice(&mut self, params: &[f32]) {
        assert!(params.len() <= GGML_MAX_OP_PARAMS_I32);
        for (dst, src) in self.op_params[..params.len()].iter_mut().zip(params) {
            *dst = src.to_bits() as i32;
        }
    }

    /// `ggml_get_name`.
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// `ggml_set_name`. C truncates at 63 bytes; keep UTF-8 intact.
    pub fn set_name(&mut self, name: &str) {
        let mut out = String::new();
        for ch in name.chars() {
            if out.len() + ch.len_utf8() >= GGML_MAX_NAME {
                break;
            }
            out.push(ch);
        }
        self.name = out;
    }

    /// `ggml_format_name`. Callers pass an already formatted string.
    pub fn format_name(&mut self, name: &str) {
        self.set_name(name);
    }

    /// Borrow the full backing buffer, if data was allocated.
    pub fn data_ref(&self) -> Option<Ref<'_, Vec<u8>>> {
        self.data.as_ref().map(|d| d.borrow())
    }

    /// Borrow the bytes reachable from this tensor, honouring `view_offs`.
    pub fn data_slice(&self) -> Option<Ref<'_, [u8]>> {
        let data = self.data.as_ref()?.borrow();
        let start = self.view_offs.min(data.len());
        Some(Ref::map(data, |d| &d[start..]))
    }

    /// `ggml_get_data_f32`. Reads the bytes as native-endian `f32` values.
    pub fn get_data_f32(&self) -> Result<Vec<f32>> {
        if self.type_ != GgmlType::F32 {
            return Err(Error::new("ggml_get_data_f32: tensor is not F32"));
        }
        let bytes = self.data_slice().ok_or_else(|| Error::new("ggml_get_data_f32: no data"))?;
        let n = bytes.len() / 4;
        Ok((0..n)
            .map(|i| f32::from_ne_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]))
            .collect())
    }

    /// `ggml_set_zero`. Writes zeros over `ggml_nbytes(tensor)` bytes.
    pub fn set_zero(&self) -> Result<()> {
        if super::helpers::ggml_is_empty(self) {
            return Ok(());
        }
        let n = super::helpers::ggml_nbytes(self);
        let data = self.data.as_ref().ok_or_else(|| Error::new("ggml_set_zero: no data"))?;
        let mut buf = data.borrow_mut();
        let end = self.view_offs.checked_add(n).ok_or_else(|| Error::new("ggml_set_zero: offset overflow"))?;
        if end > buf.len() {
            return Err(Error::new("ggml_set_zero: view exceeds backing buffer"));
        }
        buf[self.view_offs..end].fill(0);
        Ok(())
    }
}

/// `ggml_new_tensor_impl`, without the context parameter.
///
/// The C function allocates tensor data in the context pool. Here a fresh
/// tensor owns an `Arc<RefCell<Vec<u8>>>` of `data_size` zero bytes; a view
/// shares its source's buffer. Set `no_alloc` to leave `data` as `None`,
/// mirroring `ggml_context::no_alloc`.
pub fn new_tensor_impl(
    type_: GgmlType,
    n_dims: usize,
    ne: &[i64],
    nb: Option<&[usize]>,
    view_src: Option<Arc<Tensor>>,
    view_offs: usize,
    no_alloc: bool,
) -> Result<Tensor> {
    if n_dims == 0 || n_dims > GGML_MAX_DIMS {
        return Err(Error::new(format!("ggml_new_tensor: invalid n_dims {}", n_dims)));
    }
    if ne.len() < n_dims {
        return Err(Error::new("ggml_new_tensor: ne too short"));
    }
    if let Some(nb) = nb {
        if nb.len() < GGML_MAX_DIMS {
            return Err(Error::new("ggml_new_tensor: nb too short"));
        }
    }

    // Find the base tensor and absolute offset (C flattens view chains).
    let mut offs = view_offs;
    let mut src = view_src;
    if let Some(s) = src.as_ref().and_then(|s| s.view_src.as_ref()) {
        offs = offs.checked_add(s.view_offs).ok_or_else(|| Error::new("ggml_new_tensor: view offset overflow"))?;
        src = Some(s.clone());
    }

    // Validate the element count fits in int64_t.
    let mut current_nelements = ne[0];
    for &n in &ne[1..n_dims] {
        if n > 1 {
            if n <= 0 || current_nelements > i64::MAX / n {
                return Err(Error::new("ggml_new_tensor: nelements overflow"));
            }
            current_nelements *= n;
        }
    }

    let mut data_size = ggml_row_size(type_, ne[0])?;
    for &n in &ne[1..n_dims] {
        if n < 0 {
            return Err(Error::new("ggml_new_tensor: negative dimension"));
        }
        data_size = data_size
            .checked_mul(n as usize)
            .ok_or_else(|| Error::new("ggml_new_tensor: data size overflow"))?;
    }

    let data = if let Some(s) = src.as_ref() {
        s.data.clone()
    } else if !no_alloc {
        Some(Arc::new(RefCell::new(vec![0u8; data_size])))
    } else {
        None
    };

    let mut result = Tensor {
        type_,
        ne: [1; GGML_MAX_DIMS],
        nb: [0; GGML_MAX_DIMS],
        view_src: src.clone(),
        view_offs: offs,
        data,
        ..Tensor::default()
    };

    result.ne[..n_dims].copy_from_slice(&ne[..n_dims]);

    if let Some(nb) = nb {
        result.nb.copy_from_slice(&nb[..GGML_MAX_DIMS]);
    } else {
        result.nb[0] = ggml_type_size(type_);
        let bs = ggml_blck_size(type_) as usize;
        result.nb[1] = result.nb[0] * (result.ne[0] as usize / bs);
        for i in 2..GGML_MAX_DIMS {
            result.nb[i] = result.nb[i - 1] * result.ne[i - 1] as usize;
        }
    }

    // A view stays inside its source, over the bytes its strides actually reach.
    let view_size = super::helpers::ggml_nbytes(&result);
    if let Some(s) = src.as_ref() {
        let src_nbytes = super::helpers::ggml_nbytes(s);
        if view_size != 0 && offs.checked_add(view_size).map_or(true, |end| end > src_nbytes) {
            return Err(Error::new("ggml_new_tensor: view exceeds source"));
        }
    }

    Ok(result)
}

/// `ggml_tensor_overhead`. There is no C-style object header in the Rust
/// representation, so this is the tensor size only as a stand-in.
pub const fn ggml_tensor_overhead() -> usize {
    std::mem::size_of::<Tensor>()
}

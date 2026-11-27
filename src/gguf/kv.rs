//! Key/value storage plus the GGUF getter and setter API.

use super::types::{
    gguf_type_size, GgmlType, GgufType, GgufValue, GGML_MAX_DIMS, GGUF_KEY_GENERAL_ALIGNMENT,
};
use super::{
    ggml_blck_size, ggml_nbytes, ggml_type_size, gguf_pad, GgufContext, GgmlTensor, GgufTensorInfo,
};

// ---------------------------------------------------------------------------
// gguf_kv
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GgufKv {
    pub key: String,
    pub is_array: bool,
    pub type_: GgufType,
    pub data: Vec<u8>,
    pub data_string: Vec<String>,
}

impl GgufKv {
    pub fn new_scalar<T: GgufValue>(key: &str, value: T) -> Self {
        assert!(!key.is_empty());
        Self {
            key: key.to_owned(),
            is_array: false,
            type_: T::TYPE,
            data: value.encode(),
            data_string: Vec::new(),
        }
    }

    pub fn new_array<T: GgufValue>(key: &str, value: &[T]) -> Self {
        assert!(!key.is_empty());
        let mut data = Vec::with_capacity(value.len() * std::mem::size_of::<T>());
        for &v in value {
            data.extend_from_slice(&v.encode());
        }
        Self {
            key: key.to_owned(),
            is_array: true,
            type_: T::TYPE,
            data,
            data_string: Vec::new(),
        }
    }

    pub fn new_string(key: &str, value: &str) -> Self {
        assert!(!key.is_empty());
        Self {
            key: key.to_owned(),
            is_array: false,
            type_: GgufType::String,
            data: Vec::new(),
            data_string: vec![value.to_owned()],
        }
    }

    pub fn new_string_array<S: AsRef<str>>(key: &str, value: &[S]) -> Self {
        assert!(!key.is_empty());
        Self {
            key: key.to_owned(),
            is_array: true,
            type_: GgufType::String,
            data: Vec::new(),
            data_string: value.iter().map(|s| s.as_ref().to_owned()).collect(),
        }
    }

    pub fn get_key(&self) -> &str {
        &self.key
    }

    pub fn get_type(&self) -> GgufType {
        self.type_
    }

    pub fn get_ne(&self) -> usize {
        if self.type_ == GgufType::String {
            let ne = self.data_string.len();
            assert!(self.is_array || ne == 1);
            return ne;
        }
        let type_size = gguf_type_size(self.type_);
        assert!(type_size > 0 && self.data.len() % type_size == 0);
        let ne = self.data.len() / type_size;
        assert!(self.is_array || ne == 1);
        ne
    }

    pub fn get_val<T: GgufValue>(&self, i: usize) -> T {
        assert!(T::TYPE == self.type_);
        let type_size = gguf_type_size(self.type_);
        assert!(type_size > 0 && self.data.len() % type_size == 0);
        assert!(self.data.len() >= (i + 1) * type_size);
        T::decode(&self.data[i * type_size..(i + 1) * type_size])
    }

    pub fn get_str(&self, i: usize) -> &str {
        assert!(self.type_ == GgufType::String);
        assert!(self.data_string.len() > i);
        &self.data_string[i]
    }

    /// `void cast(const enum gguf_type new_type)`.
    pub fn cast(&mut self, new_type: GgufType) {
        let new_type_size = gguf_type_size(new_type);
        assert!(new_type_size > 0 && self.data.len() % new_type_size == 0);
        self.type_ = new_type;
    }
}
// ---------------------------------------------------------------------------
// getters
// ---------------------------------------------------------------------------

fn key_index(ctx: &GgufContext, key_id: i64) -> usize {
    assert!(key_id >= 0 && (key_id as usize) < ctx.kv.len());
    key_id as usize
}

fn tensor_index(ctx: &GgufContext, tensor_id: i64) -> usize {
    assert!(tensor_id >= 0 && (tensor_id as usize) < ctx.info.len());
    tensor_id as usize
}

pub fn gguf_get_version(ctx: &GgufContext) -> u32 {
    ctx.version
}

pub fn gguf_get_alignment(ctx: &GgufContext) -> usize {
    ctx.alignment
}

/// Padded to `gguf_get_alignment` if and only if the context has a tensor.
pub fn gguf_get_data_offset(ctx: &GgufContext) -> usize {
    ctx.offset
}

pub fn gguf_get_n_kv(ctx: &GgufContext) -> i64 {
    ctx.kv.len() as i64
}

/// Returns -1 if key is not found.
pub fn gguf_find_key(ctx: &GgufContext, key: &str) -> i64 {
    for (i, kv) in ctx.kv.iter().enumerate() {
        if key == kv.get_key() {
            return i as i64;
        }
    }
    -1
}

pub fn gguf_get_key(ctx: &GgufContext, key_id: i64) -> &str {
    ctx.kv[key_index(ctx, key_id)].get_key()
}

pub fn gguf_get_kv_type(ctx: &GgufContext, key_id: i64) -> GgufType {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    if kv.is_array {
        GgufType::Array
    } else {
        kv.get_type()
    }
}

pub fn gguf_get_arr_type(ctx: &GgufContext, key_id: i64) -> GgufType {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    assert!(kv.is_array);
    kv.get_type()
}

fn get_val_helper<T: GgufValue>(ctx: &GgufContext, key_id: i64) -> T {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    assert_eq!(kv.get_ne(), 1);
    kv.get_val(0)
}

/// Will panic if the wrong type is used for the key.
pub fn gguf_get_val_u8(ctx: &GgufContext, key_id: i64) -> u8 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_i8(ctx: &GgufContext, key_id: i64) -> i8 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_u16(ctx: &GgufContext, key_id: i64) -> u16 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_i16(ctx: &GgufContext, key_id: i64) -> i16 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_u32(ctx: &GgufContext, key_id: i64) -> u32 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_i32(ctx: &GgufContext, key_id: i64) -> i32 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_f32(ctx: &GgufContext, key_id: i64) -> f32 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_u64(ctx: &GgufContext, key_id: i64) -> u64 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_i64(ctx: &GgufContext, key_id: i64) -> i64 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_f64(ctx: &GgufContext, key_id: i64) -> f64 {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_bool(ctx: &GgufContext, key_id: i64) -> bool {
    get_val_helper(ctx, key_id)
}

pub fn gguf_get_val_str(ctx: &GgufContext, key_id: i64) -> &str {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    assert_eq!(kv.get_ne(), 1);
    kv.get_str(0)
}

pub fn gguf_get_val_data(ctx: &GgufContext, key_id: i64) -> &[u8] {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    assert_eq!(kv.get_ne(), 1);
    assert!(kv.get_type() != GgufType::String);
    &kv.data
}

pub fn gguf_get_arr_n(ctx: &GgufContext, key_id: i64) -> usize {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    if kv.type_ == GgufType::String {
        return kv.data_string.len();
    }
    let type_size = gguf_type_size(kv.type_);
    assert!(type_size > 0 && kv.data.len() % type_size == 0);
    kv.data.len() / type_size
}

/// Raw pointer to the first element of the array in C++. For bool arrays the
/// elements are always stored as int8 on all platforms.
pub fn gguf_get_arr_data(ctx: &GgufContext, key_id: i64) -> &[u8] {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    assert!(kv.get_type() != GgufType::String);
    &kv.data
}

/// Get the ith C string from the array with the given key_id.
pub fn gguf_get_arr_str(ctx: &GgufContext, key_id: i64, i: usize) -> &str {
    let kv = &ctx.kv[key_index(ctx, key_id)];
    assert!(kv.get_type() == GgufType::String);
    &kv.data_string[i]
}

pub fn gguf_get_n_tensors(ctx: &GgufContext) -> i64 {
    ctx.info.len() as i64
}

/// Returns -1 if the tensor is not found.
pub fn gguf_find_tensor(ctx: &GgufContext, name: &str) -> i64 {
    for (i, info) in ctx.info.iter().enumerate() {
        if name == info.t.name {
            return i as i64;
        }
    }
    -1
}

pub fn gguf_get_tensor_offset(ctx: &GgufContext, tensor_id: i64) -> usize {
    ctx.info[tensor_index(ctx, tensor_id)].offset as usize
}

pub fn gguf_get_tensor_name(ctx: &GgufContext, tensor_id: i64) -> &str {
    &ctx.info[tensor_index(ctx, tensor_id)].t.name
}

/// `ne[dim]` is 1 for dim >= n_dims.
pub fn gguf_get_tensor_ne(ctx: &GgufContext, tensor_id: i64) -> &[i64; GGML_MAX_DIMS] {
    &ctx.info[tensor_index(ctx, tensor_id)].t.ne
}

pub fn gguf_get_tensor_type(ctx: &GgufContext, tensor_id: i64) -> GgmlType {
    ctx.info[tensor_index(ctx, tensor_id)].t.type_
}

pub fn gguf_get_tensor_size(ctx: &GgufContext, tensor_id: i64) -> usize {
    ggml_nbytes(&ctx.info[tensor_index(ctx, tensor_id)].t)
}

// ---------------------------------------------------------------------------
// setters
// ---------------------------------------------------------------------------

/// Removes key if it exists, returns the id the key had prior to removal.
pub fn gguf_remove_key(ctx: &mut GgufContext, key: &str) -> i64 {
    let key_id = gguf_find_key(ctx, key);
    if key_id >= 0 {
        ctx.kv.remove(key_id as usize);
    }
    key_id
}

fn check_reserved_key(key: &str) {
    assert!(
        key != GGUF_KEY_GENERAL_ALIGNMENT,
        "general.alignment must be type u32"
    );
}

macro_rules! define_set_val {
    ($name:ident, $t:ty, $ctor:ident) => {
        pub fn $name(ctx: &mut GgufContext, key: &str, val: $t) {
            check_reserved_key(key);
            gguf_remove_key(ctx, key);
            ctx.kv.push(GgufKv::$ctor(key, val));
        }
    };
}

define_set_val!(gguf_set_val_u8, u8, new_scalar);
define_set_val!(gguf_set_val_i8, i8, new_scalar);
define_set_val!(gguf_set_val_u16, u16, new_scalar);
define_set_val!(gguf_set_val_i16, i16, new_scalar);
define_set_val!(gguf_set_val_i32, i32, new_scalar);
define_set_val!(gguf_set_val_f32, f32, new_scalar);
define_set_val!(gguf_set_val_u64, u64, new_scalar);
define_set_val!(gguf_set_val_i64, i64, new_scalar);
define_set_val!(gguf_set_val_f64, f64, new_scalar);
define_set_val!(gguf_set_val_bool, bool, new_scalar);

pub fn gguf_set_val_u32(ctx: &mut GgufContext, key: &str, val: u32) {
    if key == GGUF_KEY_GENERAL_ALIGNMENT {
        assert!(
            val > 0 && (val & (val - 1)) == 0,
            "general.alignment must be power of 2"
        );
    }
    gguf_remove_key(ctx, key);
    ctx.kv.push(GgufKv::new_scalar(key, val));
}

pub fn gguf_set_val_str(ctx: &mut GgufContext, key: &str, val: &str) {
    check_reserved_key(key);
    gguf_remove_key(ctx, key);
    ctx.kv.push(GgufKv::new_string(key, val));
}

/// Creates a new array with n elements of the given type and copies
/// `n * gguf_type_size(type)` bytes from `data`.
pub fn gguf_set_arr_data(ctx: &mut GgufContext, key: &str, ty: GgufType, data: &[u8], n: usize) {
    check_reserved_key(key);
    gguf_remove_key(ctx, key);

    let nbytes = n * gguf_type_size(ty);
    assert!(data.len() >= nbytes);
    let mut kv = GgufKv::new_array(key, &data[..nbytes].to_vec());
    // gguf_kv::cast(new_type)
    if gguf_type_size(ty) > 0 {
        kv.cast(ty);
    } else {
        kv.type_ = ty;
    }
    ctx.kv.push(kv);
}

/// Creates a new array with n strings and copies the corresponding strings.
pub fn gguf_set_arr_str(ctx: &mut GgufContext, key: &str, data: &[&str], n: usize) {
    check_reserved_key(key);
    gguf_remove_key(ctx, key);
    ctx.kv.push(GgufKv::new_string_array(key, &data[..n]));
}

/// Set or add KV pairs from another context.
pub fn gguf_set_kv(ctx: &mut GgufContext, src: &GgufContext) {
    for kv in &src.kv {
        let key = kv.get_key();

        if !kv.is_array {
            match kv.get_type() {
                GgufType::Uint8 => gguf_set_val_u8(ctx, key, kv.get_val(0)),
                GgufType::Int8 => gguf_set_val_i8(ctx, key, kv.get_val(0)),
                GgufType::Uint16 => gguf_set_val_u16(ctx, key, kv.get_val(0)),
                GgufType::Int16 => gguf_set_val_i16(ctx, key, kv.get_val(0)),
                GgufType::Uint32 => gguf_set_val_u32(ctx, key, kv.get_val(0)),
                GgufType::Int32 => gguf_set_val_i32(ctx, key, kv.get_val(0)),
                GgufType::Float32 => gguf_set_val_f32(ctx, key, kv.get_val(0)),
                GgufType::Uint64 => gguf_set_val_u64(ctx, key, kv.get_val(0)),
                GgufType::Int64 => gguf_set_val_i64(ctx, key, kv.get_val(0)),
                GgufType::Float64 => gguf_set_val_f64(ctx, key, kv.get_val(0)),
                GgufType::Bool => gguf_set_val_bool(ctx, key, kv.get_val(0)),
                GgufType::String => gguf_set_val_str(ctx, key, kv.get_str(0)),
                GgufType::Array | GgufType::Count => panic!("invalid type"),
            }
            continue;
        }

        let ne = kv.get_ne();
        match kv.get_type() {
            GgufType::Uint8
            | GgufType::Int8
            | GgufType::Uint16
            | GgufType::Int16
            | GgufType::Uint32
            | GgufType::Int32
            | GgufType::Float32
            | GgufType::Uint64
            | GgufType::Int64
            | GgufType::Float64
            | GgufType::Bool => {
                gguf_set_arr_data(ctx, key, kv.get_type(), &kv.data, ne);
            }
            GgufType::String => {
                let tmp: Vec<&str> = (0..ne).map(|j| kv.get_str(j)).collect();
                gguf_set_arr_str(ctx, key, &tmp, ne);
            }
            GgufType::Array | GgufType::Count => panic!("invalid type"),
        }
    }
}

/// Add a tensor to the GGUF context; the tensor name must be unique.
pub fn gguf_add_tensor(ctx: &mut GgufContext, tensor: &GgmlTensor) {
    assert!(!tensor.name.is_empty());
    if gguf_find_tensor(ctx, &tensor.name) != -1 {
        panic!("duplicate tensor name: {}", tensor.name);
    }

    let offset = if ctx.info.is_empty() {
        0
    } else {
        let last = ctx.info.last().unwrap();
        last.offset + gguf_pad(ggml_nbytes(&last.t), ctx.alignment) as u64
    };
    ctx.info.push(GgufTensorInfo {
        t: tensor.clone(),
        offset,
    });
}

/// After changing a tensor's type, offsets of all tensors with higher indices
/// are immediately recalculated so the data stays one contiguous block.
pub fn gguf_set_tensor_type(ctx: &mut GgufContext, name: &str, ty: GgmlType) {
    let tensor_id = gguf_find_tensor(ctx, name);
    assert!(tensor_id >= 0, "tensor not found: {}", name);
    let tensor_id = tensor_id as usize;

    let type_size = ggml_type_size(ty);
    let blck_size = ggml_blck_size(ty);

    {
        let t = &mut ctx.info[tensor_id].t;
        t.type_ = ty;
        assert!(
            blck_size > 0 && t.ne[0] % blck_size == 0,
            "tensor row size not divisible by block size of new type"
        );

        t.nb[0] = type_size;
        t.nb[1] = t.nb[0] * (t.ne[0] as usize / blck_size.max(1) as usize);
        for j in 2..GGML_MAX_DIMS {
            t.nb[j] = t.nb[j - 1] * t.ne[j - 1] as usize;
        }
    }

    // update offsets
    let n_tensors = gguf_get_n_tensors(ctx) as usize;
    for i in (tensor_id + 1)..n_tensors {
        let prev_nbytes = ggml_nbytes(&ctx.info[i - 1].t);
        ctx.info[i].offset =
            ctx.info[i - 1].offset + gguf_pad(prev_nbytes, ctx.alignment) as u64;
    }
}

/// Assumes that at least `gguf_get_tensor_size` bytes can be read from data.
pub fn gguf_set_tensor_data(ctx: &mut GgufContext, name: &str, data: &[u8]) {
    let tensor_id = gguf_find_tensor(ctx, name);
    assert!(tensor_id >= 0, "tensor not found: {}", name);
    ctx.info[tensor_id as usize].t.data = data.to_vec();
}

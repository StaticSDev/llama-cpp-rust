//! Rust port of `llama-adapter.cpp` / `llama-adapter.h`.
//!
//! Control vectors and LoRA adapters: tensor pairing, GGUF metadata checks,
//! shape validation, weight loading and the public adapter API.
//!
//! No ggml layer exists in this crate yet. Tensors, contexts and buffers are
//! local stand-ins that keep names, shapes, buffer types and owned bytes. The
//! additions in `cvec::apply_to` are evaluated eagerly on those bytes.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::arch::{arch_from_string, Arch, Kv, LlmKv};
use crate::core::{Error, Result, Token};
use crate::gguf::{
    gguf_find_key, gguf_find_tensor, gguf_get_arr_data, gguf_get_arr_n, gguf_get_arr_type,
    gguf_get_data_offset, gguf_get_key, gguf_get_kv_type, gguf_get_n_kv, gguf_get_n_tensors,
    gguf_get_tensor_offset, gguf_get_val_f32, gguf_get_val_str, gguf_init_from_buffer, GgmlContext,
    GgmlTensor, GgmlType, GgufContext, GgufInitParams, GgufType,
};
use crate::hparams::Hparams;

// ---------------------------------------------------------------------------
// ggml stand-ins
// ---------------------------------------------------------------------------

/// Minimal stand-in for `ggml_tensor`. Data is shared through `Rc` so that a
/// tensor cloned into a context still sees the bytes written by its buffer.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub ne: [i64; 4],
    pub type_: GgmlType,
    pub buft: String,
    pub data: Rc<RefCell<Vec<u8>>>,
}

impl Tensor {
    pub fn new(name: &str, ne: [i64; 4], type_: GgmlType, buft: &str) -> Self {
        Self {
            name: name.to_owned(),
            ne,
            type_,
            buft: buft.to_owned(),
            data: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// `ggml_nbytes`.
    pub fn nbytes(&self) -> usize {
        let mut nelements = 1i128;
        for &ne in &self.ne {
            nelements *= ne as i128;
        }
        let blck = self.type_.blck_size().max(1) as i128;
        let nbytes = nelements * self.type_.type_size() as i128 / blck;
        if nbytes < 0 {
            0
        } else {
            nbytes as usize
        }
    }

    /// `ggml_backend_buffer_clear(buf, 0)` for this tensor.
    pub fn zero(&self) {
        *self.data.borrow_mut() = vec![0u8; self.nbytes()];
    }

    pub fn bytes(&self) -> Vec<u8> {
        self.data.borrow().clone()
    }

    pub fn set_bytes(&self, data: &[u8]) {
        let mut s = self.data.borrow_mut();
        let n = data.len().min(s.len());
        s[..n].copy_from_slice(&data[..n]);
    }
}

/// Copy a GGUF tensor descriptor into the local tensor stand-in.
fn tensor_from_ggml(t: &GgmlTensor) -> Tensor {
    let out = Tensor::new(&t.name, t.ne, t.type_, "");
    out.set_bytes(&t.data);
    out
}

/// Stand-in for `ggml_context`: tensor descriptors created before allocation.
#[derive(Clone, Debug, Default)]
pub struct Context {
    pub tensors: Vec<Tensor>,
}

impl Context {
    pub fn with_capacity(n: usize) -> Self {
        Self { tensors: Vec::with_capacity(n) }
    }

    fn push_tensor(&mut self, name: &str, ne: [i64; 4], type_: GgmlType, buft: &str) -> Tensor {
        let t = Tensor::new(name, ne, type_, buft);
        self.tensors.push(t.clone());
        t
    }

    fn new_tensor_1d(&mut self, type_: GgmlType, ne0: usize, buft: &str) -> Tensor {
        self.push_tensor("", [ne0 as i64, 1, 1, 1], type_, buft)
    }

    /// `ggml_dup_tensor` plus `ggml_set_name`.
    fn dup_tensor(&mut self, src: &Tensor, buft: &str) -> Tensor {
        self.push_tensor(&src.name, src.ne, src.type_, buft)
    }
}

/// Stand-in for `ggml_backend_buffer`. Allocation clears all context tensors.
#[derive(Clone, Debug)]
pub struct Buffer {
    pub buft: String,
    pub size: usize,
}

impl Buffer {
    fn allocate(buft: &str, ctx: &Context, error: &str) -> Result<Self> {
        let mut size = 0usize;
        for t in &ctx.tensors {
            size = size.checked_add(t.nbytes()).ok_or_else(|| Error::new(error))?;
        }
        for t in &ctx.tensors {
            t.zero();
        }
        Ok(Self { buft: buft.to_owned(), size })
    }
}

/// Stand-in for the parts of `llama_model` that adapter loading reads.
pub struct Model {
    pub arch: Arch,
    pub hparams: Hparams,
    pub tensors: BTreeMap<String, Tensor>,
    /// Buffer type selected for each effective layer.
    pub layer_buft: Vec<String>,
    /// CPU buffer types used for repacking; LoRA falls back to `cpu_buft`.
    pub extra_bufts: Vec<String>,
    pub cpu_buft: Option<String>,
    /// Names of hadamard latent lookup tables (C++ `hdmd.inverse_blocks`).
    pub latent_lookups: BTreeSet<String>,
    /// Loaded adapters (C++ `model.loras`). Keys are `Lora::id`.
    pub loras: BTreeSet<usize>,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            arch: Arch::Unknown,
            hparams: Hparams::default(),
            tensors: BTreeMap::new(),
            layer_buft: Vec::new(),
            extra_bufts: Vec::new(),
            cpu_buft: Some("CPU".to_owned()),
            latent_lookups: BTreeSet::new(),
            loras: BTreeSet::new(),
        }
    }
}

impl Model {
    /// `model.get_tensor(name)`.
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// `model.select_buft(il)`.
    pub fn select_buft(&self, il: usize) -> String {
        self.layer_buft
            .get(il)
            .cloned()
            .or_else(|| self.cpu_buft.clone())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// llama_adapter_cvec
// ---------------------------------------------------------------------------

/// Control vector: one F32 direction per layer, applied by addition.
#[derive(Clone, Debug, Default)]
pub struct Cvec {
    layer_start: i32,
    layer_end: i32,
    ctxs: Vec<Context>,
    bufs: Vec<Buffer>,
    /// Per layer; index 0 is `None` because layer 0 has no direction.
    tensors: Vec<Option<Tensor>>,
}

impl Cvec {
    /// `llama_adapter_cvec::tensor_for`.
    pub fn tensor_for(&self, il: i32) -> Option<Tensor> {
        if il < 0 || il < self.layer_start || il > self.layer_end || il as usize >= self.tensors.len() {
            return None;
        }
        self.tensors[il as usize].clone()
    }

    /// `llama_adapter_cvec::apply_to`. `ggml_add` is evaluated on the bytes.
    pub fn apply_to(&self, ctx: &mut Context, cur: &Tensor, il: i32) -> Tensor {
        let layer_dir = match self.tensor_for(il) {
            Some(dir) => dir,
            None => return cur.clone(),
        };

        let mut out = ctx.push_tensor("", cur.ne, cur.type_, "");
        let cur_bytes = cur.bytes();
        let dir_bytes = layer_dir.bytes();
        let mut sum = vec![0u8; out.nbytes()];

        if cur.type_ == GgmlType::F32 && layer_dir.type_ == GgmlType::F32 {
            let n = (cur_bytes.len().min(dir_bytes.len()).min(sum.len())) / 4;
            for i in 0..n {
                let a = f32::from_ne_bytes(cur_bytes[4 * i..4 * i + 4].try_into().unwrap());
                let b = f32::from_ne_bytes(dir_bytes[4 * i..4 * i + 4].try_into().unwrap());
                sum[4 * i..4 * i + 4].copy_from_slice(&(a + b).to_ne_bytes());
            }
            if 4 * n < sum.len() {
                let rest = &cur_bytes[4 * n..cur_bytes.len().min(sum.len())];
                sum[4 * n..4 * n + rest.len()].copy_from_slice(rest);
            }
        } else {
            let n = cur_bytes.len().min(sum.len());
            sum[..n].copy_from_slice(&cur_bytes[..n]);
        }
        out.set_bytes(&sum);
        out
    }

    /// `llama_adapter_cvec::apply`. `data == None` disables the vector.
    pub fn apply(
        &mut self,
        model: &Rc<RefCell<Model>>,
        data: Option<&[f32]>,
        n_embd: i32,
        il_start: i32,
        il_end: i32,
    ) -> Result<()> {
        let m = model.borrow();

        let data = match data {
            Some(d) => d,
            None => {
                self.layer_start = -1;
                self.layer_end = -1;
                return Ok(());
            }
        };

        if n_embd as u32 != m.hparams.n_embd {
            return Err(Error::new("llama_adapter_cvec::apply: control vector n_embd does not match model"));
        }

        if self.tensors.is_empty() {
            self.init(model)?;
        }

        self.layer_start = il_start;
        self.layer_end = il_end;

        let n = n_embd as usize;
        for il in 1..m.hparams.n_layer() as usize {
            let t = self.tensors[il]
                .as_ref()
                .ok_or_else(|| Error::new("llama_adapter_cvec::apply: missing layer tensor"))?;
            // Layer 0 has no tensor, so its data starts at `n_embd * (il - 1)`.
            let off = n * (il - 1);
            if off + n <= data.len() {
                let mut bytes = Vec::with_capacity(n * 4);
                for &v in &data[off..off + n] {
                    bytes.extend_from_slice(&v.to_ne_bytes());
                }
                t.set_bytes(&bytes);
            }
        }
        Ok(())
    }

    fn init(&mut self, model: &Rc<RefCell<Model>>) -> Result<()> {
        let m = model.borrow();
        let n_layer = m.hparams.n_layer() as usize;
        let mut ctx_map: BTreeMap<String, Context> = BTreeMap::new();
        let mut tensors: Vec<Option<Tensor>> = Vec::with_capacity(n_layer);
        tensors.push(None);

        for il in 1..n_layer {
            let buft = m.select_buft(il);
            let ctx = ctx_map.entry(buft.clone()).or_insert_with(|| Context::with_capacity(n_layer));
            let t = ctx.new_tensor_1d(GgmlType::F32, m.hparams.n_embd as usize, &buft);
            tensors.push(Some(t));
        }

        for (buft, ctx) in ctx_map.iter() {
            self.bufs.push(Buffer::allocate(
                buft,
                ctx,
                "llama_adapter_cvec::init: failed to allocate buffer for control vector",
            )?);
        }
        self.ctxs = ctx_map.into_values().collect();
        self.tensors = tensors;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// llama_adapter_lora
// ---------------------------------------------------------------------------

/// `llama_adapter_lora_weight`: the a/b pair for one base-model tensor.
#[derive(Clone, Debug, Default)]
pub struct LoraWeight {
    pub a: Option<Tensor>,
    pub b: Option<Tensor>,
}

impl LoraWeight {
    pub fn new(a: Tensor, b: Tensor) -> Self {
        Self { a: Some(a), b: Some(b) }
    }

    /// Actual scale based on rank and alpha.
    pub fn get_scale(&self, alpha: f32, adapter_scale: f32) -> f32 {
        let rank = self.b.as_ref().map(|b| b.ne[0] as f32).unwrap_or(1.0);
        if alpha != 0.0 {
            adapter_scale * alpha / rank
        } else {
            adapter_scale
        }
    }
}

/// `llama_adapter_loras`: active adapters mapped to their strength.
/// Keys are `Lora::id`; no raw pointers are needed in the Rust port.
pub type Loras = BTreeMap<usize, f32>;

static NEXT_LORA_ID: AtomicUsize = AtomicUsize::new(1);

/// `llama_adapter_lora`.
pub struct Lora {
    pub model: Option<Rc<RefCell<Model>>>,
    /// Base tensor name to a/b pair.
    pub ab_map: BTreeMap<String, LoraWeight>,
    pub ctxs: Vec<Context>,
    pub bufs: Vec<Buffer>,
    pub alpha: f32,
    pub gguf_kv: BTreeMap<String, String>,
    pub alora_invocation_tokens: Vec<Token>,
    id: usize,
}

impl Lora {
    pub fn new(model: &Rc<RefCell<Model>>) -> Self {
        Self {
            model: Some(model.clone()),
            ab_map: BTreeMap::new(),
            ctxs: Vec::new(),
            bufs: Vec::new(),
            alpha: 0.0,
            gguf_kv: BTreeMap::new(),
            alora_invocation_tokens: Vec::new(),
            id: NEXT_LORA_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    /// `llama_adapter_lora::get_weight` by tensor name.
    pub fn get_weight(&self, w: &Tensor) -> Option<&LoraWeight> {
        self.ab_map.get(&w.name)
    }

    /// `llama_adapter_lora::get_n_nodes`.
    pub fn get_n_nodes(&self) -> u32 {
        self.ab_map.len() as u32 * 6
    }

    /// `llama_adapter_meta_count` as a method.
    pub fn meta_count(&self) -> i32 {
        self.gguf_kv.len() as i32
    }

    /// `llama_adapter_meta_key_by_index` as a method.
    pub fn meta_key_by_index(&self, i: i32, buf: &mut [u8]) -> i32 {
        match self.gguf_kv.keys().nth(i as usize) {
            Some(key) => write_cstr(buf, key),
            None => clear_cstr(buf),
        }
    }

    /// `llama_adapter_meta_val_str_by_index` as a method.
    pub fn meta_val_str_by_index(&self, i: i32, buf: &mut [u8]) -> i32 {
        match self.gguf_kv.values().nth(i as usize) {
            Some(value) => write_cstr(buf, value),
            None => clear_cstr(buf),
        }
    }

    /// `llama_adapter_lora_free`: unregister the adapter from its model.
    pub fn free(&mut self) {
        if let Some(model) = self.model.take() {
            model.borrow_mut().loras.remove(&self.id);
        }
    }
}

fn clear_cstr(buf: &mut [u8]) -> i32 {
    if !buf.is_empty() {
        buf[0] = 0;
    }
    -1
}

/// `snprintf(buf, buf_size, "%s", value)` semantics.
fn write_cstr(buf: &mut [u8], value: &str) -> i32 {
    let bytes = value.as_bytes();
    if !buf.is_empty() {
        let n = bytes.len().min(buf.len() - 1);
        buf[..n].copy_from_slice(&bytes[..n]);
        buf[n] = 0;
    }
    bytes.len() as i32
}

/// `gguf_kv_to_str` for scalars. Arrays are not stored in `gguf_kv`.
fn kv_to_str(ctx: &GgufContext, i: i64) -> String {
    let kv = &ctx.kv[i as usize];
    if kv.is_array {
        return String::new();
    }
    match kv.get_type() {
        GgufType::String => kv.get_str(0).to_owned(),
        GgufType::Bool => kv.get_val::<bool>(0).to_string(),
        GgufType::Uint8 => kv.get_val::<u8>(0).to_string(),
        GgufType::Int8 => kv.get_val::<i8>(0).to_string(),
        GgufType::Uint16 => kv.get_val::<u16>(0).to_string(),
        GgufType::Int16 => kv.get_val::<i16>(0).to_string(),
        GgufType::Uint32 => kv.get_val::<u32>(0).to_string(),
        GgufType::Int32 => kv.get_val::<i32>(0).to_string(),
        GgufType::Uint64 => kv.get_val::<u64>(0).to_string(),
        GgufType::Int64 => kv.get_val::<i64>(0).to_string(),
        GgufType::Float32 => kv.get_val::<f32>(0).to_string(),
        GgufType::Float64 => kv.get_val::<f64>(0).to_string(),
        GgufType::Array | GgufType::Count => String::new(),
    }
}

fn get_kv_str(ctx: &GgufContext, key: &str) -> String {
    let id = gguf_find_key(ctx, key);
    if id < 0 || gguf_get_kv_type(ctx, id) != GgufType::String {
        return String::new();
    }
    gguf_get_val_str(ctx, id).to_owned()
}

fn get_kv_f32(ctx: &GgufContext, key: &str) -> f32 {
    let id = gguf_find_key(ctx, key);
    if id < 0 || gguf_get_kv_type(ctx, id) != GgufType::Float32 {
        return 0.0;
    }
    gguf_get_val_f32(ctx, id)
}

fn llama_adapter_lora_init_impl(
    model: &Rc<RefCell<Model>>,
    file_bytes: &[u8],
    adapter: &mut Lora,
) -> Result<()> {
    let mut meta_params = GgufInitParams { no_alloc: true, ctx: Some(GgmlContext::default()) };
    let ctx_gguf = gguf_init_from_buffer(file_bytes, &mut meta_params)
        .map_err(|e| Error::new(format!("failed to load lora adapter from file: {}", e.message)))?;
    let ctx = meta_params
        .ctx
        .take()
        .ok_or_else(|| Error::new("failed to load lora adapter from file: no ggml context"))?;

    // Dump metadata into the adapter's string map (arrays are skipped).
    for i in 0..gguf_get_n_kv(&ctx_gguf) {
        let ty = gguf_get_kv_type(&ctx_gguf, i);
        let name = gguf_get_key(&ctx_gguf, i).to_owned();
        let value = kv_to_str(&ctx_gguf, i);
        if ty != GgufType::Array {
            adapter.gguf_kv.entry(name).or_insert(value);
        }
    }

    let llm_kv = LlmKv::new(Arch::Unknown, None);

    let general_type = get_kv_str(&ctx_gguf, &llm_kv.key(Kv::GeneralType));
    if general_type != "adapter" {
        return Err(Error::new(format!(
            "expect general.type to be 'adapter', but got: {}",
            general_type
        )));
    }

    let arch = arch_from_string(&get_kv_str(&ctx_gguf, &llm_kv.key(Kv::GeneralArchitecture)));
    if arch != model.borrow().arch {
        return Err(Error::new("model arch and LoRA arch mismatch"));
    }

    let adapter_type = get_kv_str(&ctx_gguf, &llm_kv.key(Kv::AdapterType));
    if adapter_type != "lora" {
        return Err(Error::new(format!(
            "expect adapter.type to be 'lora', but got: {}",
            adapter_type
        )));
    }

    adapter.alpha = get_kv_f32(&ctx_gguf, &llm_kv.key(Kv::AdapterLoraAlpha));

    // aLoRA invocation tokens: array of uint32.
    let key = llm_kv.key(Kv::AdapterAloraInvocationTokens);
    let kid = gguf_find_key(&ctx_gguf, &key);
    if kid >= 0 {
        if gguf_get_kv_type(&ctx_gguf, kid) != GgufType::Array {
            return Err(Error::new(format!("invalid gguf type for {}", key)));
        }
        if gguf_get_arr_type(&ctx_gguf, kid) != GgufType::Uint32 {
            return Err(Error::new(format!("invalid gguf element type for {}", key)));
        }
        let seq_len = gguf_get_arr_n(&ctx_gguf, kid);
        let data = gguf_get_arr_data(&ctx_gguf, kid);
        if data.len() < 4 * seq_len {
            return Err(Error::new(format!("invalid gguf element type for {}", key)));
        }
        adapter.alora_invocation_tokens = (0..seq_len)
            .map(|i| {
                let bytes: [u8; 4] = data[4 * i..4 * i + 4].try_into().unwrap();
                u32::from_ne_bytes(bytes) as Token
            })
            .collect();
    }

    let n_tensors = gguf_get_n_tensors(&ctx_gguf) as usize;

    // Bundle lora_a and lora_b into pairs.
    let mut ab_map: BTreeMap<String, LoraWeight> = BTreeMap::new();
    for cur in &ctx.tensors {
        let name = &cur.name;
        if name.ends_with(".lora_a") {
            let base = name.replace(".lora_a", "");
            ab_map.entry(base).or_insert_with(LoraWeight::default).a = Some(tensor_from_ggml(cur));
        } else if name.ends_with(".lora_b") {
            let base = name.replace(".lora_b", "");
            ab_map.entry(base).or_insert_with(LoraWeight::default).b = Some(tensor_from_ggml(cur));
        } else if name.ends_with("_norm.weight") {
            // Norm vectors are not applied; most adapters work without them.
            continue;
        } else {
            return Err(Error::new(format!(
                "LoRA tensor '{}' has unexpected suffix",
                name
            )));
        }
    }

    // Build one context per buffer type.
    let mut ctx_map: BTreeMap<String, Context> = BTreeMap::new();
    {
        let m = model.borrow();
        for (name, w) in &ab_map {
            let is_token_embd = name.ends_with("token_embd.weight");
            let a = w.a.as_ref().ok_or_else(|| {
                Error::new(format!("LoRA tensor pair for '{}' is missing one component", name))
            })?;
            let b = w.b.as_ref().ok_or_else(|| {
                Error::new(format!("LoRA tensor pair for '{}' is missing one component", name))
            })?;

            let model_tensor = m.get_tensor(name).ok_or_else(|| {
                Error::new(format!(
                    "LoRA tensor '{}' does not exist in base model (hint: maybe wrong base model?)",
                    name
                ))
            })?;
            if m.latent_lookups.contains(name) {
                return Err(Error::new(format!(
                    "LoRA tensor '{}' targets a prism.hadamard latent lookup table, which is not supported",
                    name
                )));
            }

            let mut buft = model_tensor.buft.clone();
            if m.extra_bufts.contains(&buft) {
                let cpu = m.cpu_buft.clone().ok_or_else(|| {
                    Error::new("llama_adapter_lora_init_impl: no CPU backend found")
                })?;
                buft = cpu;
            }

            let dev_ctx = ctx_map.entry(buft.clone()).or_insert_with(|| Context::with_capacity(n_tensors));

            if is_token_embd {
                // A and B are flipped; B is non-transposed.
                if model_tensor.ne[0] != b.ne[1] || model_tensor.ne[1] != a.ne[1] {
                    return Err(Error::new(format!(
                        "tensor '{}' has incorrect shape (hint: maybe wrong base model?)",
                        name
                    )));
                }
            } else {
                if model_tensor.ne[0] != a.ne[0] || model_tensor.ne[1] != b.ne[1] {
                    return Err(Error::new(format!(
                        "tensor '{}' has incorrect shape (hint: maybe wrong base model?)",
                        name
                    )));
                }
                if a.ne[1] != b.ne[0] {
                    return Err(Error::new(
                        "lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)",
                    ));
                }
            }

            let tensor_a = dev_ctx.dup_tensor(a, &buft);
            let tensor_b = dev_ctx.dup_tensor(b, &buft);
            adapter.ab_map.insert(name.clone(), LoraWeight::new(tensor_a, tensor_b));
        }
    }

    // Allocate tensors / buffers.
    for (buft, ctx) in ctx_map.iter() {
        adapter.bufs.push(Buffer::allocate(
            buft,
            ctx,
            "failed to allocate buffer for lora adapter",
        )?);
    }
    adapter.ctxs = ctx_map.into_values().collect();

    // Copy tensor bytes from the GGUF data section.
    for (name, dev) in &adapter.ab_map {
        let orig = ab_map.get(name).ok_or_else(|| {
            Error::new(format!("LoRA tensor pair for '{}' is missing one component", name))
        })?;
        for (src, dst) in [
            (orig.a.as_ref(), dev.a.as_ref()),
            (orig.b.as_ref(), dev.b.as_ref()),
        ] {
            let src = match (src, dst) {
                (Some(s), Some(d)) => (s, d),
                _ => continue,
            };
            let tid = gguf_find_tensor(&ctx_gguf, &src.0.name);
            if tid < 0 {
                return Err(Error::new(format!(
                    "LoRA tensor '{}' data is not within the file bounds, file is corrupted or incomplete",
                    src.0.name
                )));
            }
            let offs = gguf_get_data_offset(&ctx_gguf) + gguf_get_tensor_offset(&ctx_gguf, tid);
            let size = src.0.nbytes();
            let end = offs.checked_add(size);
            if end.map_or(true, |e| e > file_bytes.len()) {
                return Err(Error::new(format!(
                    "LoRA tensor '{}' data is not within the file bounds, file is corrupted or incomplete",
                    src.0.name
                )));
            }
            src.1.set_bytes(&file_bytes[offs..end.unwrap()]);
        }
    }

    // Register the adapter with the model.
    model.borrow_mut().loras.insert(adapter.id);
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// `llama_adapter_lora_init`.
pub fn llama_adapter_lora_init(model: &Rc<RefCell<Model>>, path: &str) -> Result<Lora> {
    let bytes = std::fs::read(path)
        .map_err(|e| Error::new(format!("llama_adapter_lora_init: failed to open '{}': {}", path, e)))?;
    llama_adapter_lora_init_from_bytes(model, &bytes)
}

/// `llama_adapter_lora_init_from_file_ptr` with an owned `File`.
pub fn llama_adapter_lora_init_from_file_ptr(model: &Rc<RefCell<Model>>, mut file: File) -> Result<Lora> {
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| Error::new(format!("failed to read lora adapter file: {}", e)))?;
    llama_adapter_lora_init_from_bytes(model, &bytes)
}

/// Stand-in for the `FILE *` entry point: parse an adapter from a byte buffer.
pub fn llama_adapter_lora_init_from_bytes(model: &Rc<RefCell<Model>>, bytes: &[u8]) -> Result<Lora> {
    let mut adapter = Lora::new(model);
    llama_adapter_lora_init_impl(model, bytes, &mut adapter)?;
    Ok(adapter)
}

/// `llama_adapter_meta_val_str`.
pub fn llama_adapter_meta_val_str(adapter: &Lora, key: &str, buf: &mut [u8]) -> i32 {
    match adapter.gguf_kv.get(key) {
        Some(value) => write_cstr(buf, value),
        None => clear_cstr(buf),
    }
}

/// `llama_adapter_meta_count`.
pub fn llama_adapter_meta_count(adapter: &Lora) -> i32 {
    adapter.meta_count()
}

/// `llama_adapter_meta_key_by_index`.
pub fn llama_adapter_meta_key_by_index(adapter: &Lora, i: i32, buf: &mut [u8]) -> i32 {
    adapter.meta_key_by_index(i, buf)
}

/// `llama_adapter_meta_val_str_by_index`.
pub fn llama_adapter_meta_val_str_by_index(adapter: &Lora, i: i32, buf: &mut [u8]) -> i32 {
    adapter.meta_val_str_by_index(i, buf)
}

/// `llama_adapter_lora_free`. `None` mirrors a null adapter pointer.
pub fn llama_adapter_lora_free(adapter: &mut Option<Lora>) {
    if let Some(a) = adapter.as_mut() {
        a.free();
    }
    *adapter = None;
}

/// `llama_adapter_get_alora_n_invocation_tokens`.
pub fn llama_adapter_get_alora_n_invocation_tokens(adapter: Option<&Lora>) -> u64 {
    adapter.map_or(0, |a| a.alora_invocation_tokens.len() as u64)
}

/// `llama_adapter_get_alora_invocation_tokens`.
pub fn llama_adapter_get_alora_invocation_tokens(adapter: &Lora) -> &[Token] {
    &adapter.alora_invocation_tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(n_embd: u32, n_layer: u32) -> Rc<RefCell<Model>> {
        let mut m = Model::default();
        m.arch = Arch::Llama;
        m.hparams.n_embd = n_embd;
        m.hparams.n_layer_all = n_layer;
        Rc::new(RefCell::new(m))
    }

    #[test]
    fn lora_scale_uses_rank_and_alpha() {
        let a = Tensor::new("w.lora_a", [8, 4, 1, 1], GgmlType::F32, "CPU");
        let b = Tensor::new("w.lora_b", [4, 8, 1, 1], GgmlType::F32, "CPU");
        let w = LoraWeight::new(a, b);
        assert_eq!(w.get_scale(16.0, 1.0), 4.0);
        assert_eq!(w.get_scale(0.0, 1.0), 1.0);
    }

    #[test]
    fn cvec_skips_layer_zero_and_disabled_range() {
        let m = model(2, 3);
        let mut cvec = Cvec::default();
        cvec.apply(&m, Some(&[1.0, 2.0, 3.0, 4.0]), 2, 1, 2).unwrap();
        assert!(cvec.tensor_for(0).is_none());

        let cur = Tensor::new("cur", [2, 1, 1, 1], GgmlType::F32, "CPU");
        cur.zero();
        cur.set_bytes(&[0.0f32.to_ne_bytes(), 0.0f32.to_ne_bytes()].concat());
        let mut ctx = Context::default();
        let out = cvec.apply_to(&mut ctx, &cur, 1);
        let bytes = out.bytes();
        assert_eq!(f32::from_ne_bytes(bytes[0..4].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_ne_bytes(bytes[4..8].try_into().unwrap()), 2.0);

        cvec.apply(&m, None, 2, 1, 2).unwrap();
        assert!(cvec.tensor_for(1).is_none());
    }

    #[test]
    fn meta_api_matches_snprintf_shape() {
        let mut lora = Lora::new(&model(1, 1));
        lora.gguf_kv.insert("general.type".to_owned(), "adapter".to_owned());
        assert_eq!(llama_adapter_meta_count(&lora), 1);

        let mut buf = [0xffu8; 4];
        assert_eq!(llama_adapter_meta_val_str(&lora, "general.type", &mut buf), 7);
        assert_eq!(buf, [b'a', b'd', b'a', b'p']);

        assert_eq!(llama_adapter_meta_val_str(&lora, "missing", &mut buf), -1);
        assert_eq!(buf[0], 0);
    }

    #[test]
    fn free_unregisters_adapter() {
        let m = model(1, 1);
        let mut lora = Lora::new(&m);
        m.borrow_mut().loras.insert(lora.id());
        llama_adapter_lora_free(&mut Some(lora));
        assert!(m.borrow().loras.is_empty());
    }

    #[test]
    fn lora_init_checks_metadata_and_pairs_weights() {
        use crate::gguf::{
            gguf_add_tensor, gguf_set_val_f32, gguf_set_val_str, gguf_write_to_buf,
        };

        let mut g = GgufContext::default();
        gguf_set_val_str(&mut g, "general.type", "adapter");
        gguf_set_val_str(&mut g, "general.architecture", "llama");
        gguf_set_val_str(&mut g, "adapter.type", "lora");
        gguf_set_val_f32(&mut g, "adapter.lora.alpha", 8.0);

        let mk = |name: &str, ne: [i64; 4]| {
            let row = 4 * ne[0] as usize;
            let mut t = GgmlTensor { name: name.to_owned(), ne, ..GgmlTensor::default() };
            t.nb = [4, row, row * ne[1] as usize, row * ne[1] as usize];
            t.data = vec![0u8; ne[0] as usize * ne[1] as usize * 4];
            t
        };
        gguf_add_tensor(&mut g, &mk("blk.0.attn_q.lora_a", [4, 8, 1, 1]));
        gguf_add_tensor(&mut g, &mk("blk.0.attn_q.lora_b", [8, 4, 1, 1]));
        let bytes = gguf_write_to_buf(&g, false);

        let m = model(4, 1);
        m.borrow_mut().tensors.insert(
            "blk.0.attn_q.weight".to_owned(),
            Tensor::new("blk.0.attn_q.weight", [4, 4, 1, 1], GgmlType::F32, "CPU"),
        );
        let lora = llama_adapter_lora_init_from_bytes(&m, &bytes).unwrap();
        assert_eq!(lora.alpha, 8.0);
        assert_eq!(lora.get_n_nodes(), 6);
        assert!(lora.ab_map.contains_key("blk.0.attn_q.weight"));
        assert!(m.borrow().loras.contains(&lora.id()));
    }
}

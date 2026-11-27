//! Stand-in for `llama_model_loader` on top of the pure-Rust GGUF context.
//!
//! It reads typed metadata, resolves tensor names through `LLM_TN`, and owns
//! the tensors created during `load_tensors`. No file handle or pointer is
//! retained: tensor bytes are copied from the GGUF data section.

use super::types::*;
use crate::arch::{Arch, LlmTnImpl};
use crate::core::{Error, Result};
use crate::gguf::{ggml_nbytes, GgufContext, GgufKv, GgufType, GGUF_KEY_GENERAL_ALIGNMENT};
use crate::hparams::LLAMA_MAX_LAYERS;
use std::collections::BTreeMap;

/// `llama_model_loader` fields needed by the model implementation.
#[derive(Clone, Debug)]
pub struct ModelLoader {
    pub metadata: GgufContext,
    pub arch: Arch,
    pub n_elements: u64,
    pub n_bytes: usize,
    pub ftype: Ftype,
    pub use_mmap: bool,
    pub lazy: LazyMode,
    pub load_mtp: bool,
    pub no_alloc: bool,
    /// Created tensors, indexed by resolved GGUF name.
    pub tensors: BTreeMap<String, Tensor>,
    /// Names requested but skipped by flags.
    pub skipped: Vec<String>,
}

impl ModelLoader {
    /// `llama_model_loader` initialization from an already parsed GGUF file.
    pub fn from_gguf(ctx: GgufContext, params: &ModelParams) -> Result<Self> {
        let arch = ctx
            .kv
            .iter()
            .find(|kv| kv.key == crate::arch::Kv::GeneralArchitecture.format(Arch::Unknown))
            .and_then(|kv| {
                if kv.get_type() == GgufType::String {
                    kv.data_string.first().map(|s| Arch::from_string(s))
                } else {
                    None
                }
            })
            .unwrap_or(Arch::Unknown);
        let ftype = Self::scalar_u32(&ctx, &crate::arch::Kv::GeneralFileType.format(arch))
            .unwrap_or(LLAMA_FTYPE_ALL_F32 as u32) as Ftype;
        let mut n_bytes = 0usize;
        let mut n_elements = 0u64;
        for info in &ctx.info {
            n_bytes += ggml_nbytes(&info.t);
            n_elements += info.t.ne.iter().map(|&n| n.max(0) as u64).product::<u64>();
        }
        let use_mmap = params.load_mode == LoadMode::Mmap
            || params.load_mode == LoadMode::MmapMlock
            || (params.load_mode == LoadMode::Auto && !params.no_alloc);
        let lazy = params.lazy_mode;
        Ok(Self {
            metadata: ctx,
            arch,
            n_elements,
            n_bytes,
            ftype,
            use_mmap,
            lazy,
            load_mtp: params.load_mtp,
            no_alloc: params.no_alloc,
            tensors: BTreeMap::new(),
            skipped: Vec::new(),
        })
    }

    /// Empty loader, useful when a model is built from explicit metadata only.
    pub fn new(arch: Arch) -> Self {
        Self {
            metadata: GgufContext::default(),
            arch,
            n_elements: 0,
            n_bytes: 0,
            ftype: LLAMA_FTYPE_ALL_F32,
            use_mmap: false,
            lazy: LazyMode::Off,
            load_mtp: false,
            no_alloc: false,
            tensors: BTreeMap::new(),
            skipped: Vec::new(),
        }
    }

    fn kv_id(&self, key: &str) -> Option<usize> {
        self.metadata.kv.iter().position(|kv| kv.key == key)
    }

    fn scalar_u32(ctx: &GgufContext, key: &str) -> Option<u32> {
        let kv = ctx.kv.iter().find(|kv| kv.key == key)?;
        if kv.is_array || kv.get_type() != GgufType::Uint32 {
            return None;
        }
        Some(kv.get_val(0))
    }

    /// `ml.get_key` for a string value.
    pub fn get_string(&self, key: crate::arch::Kv, required: bool) -> Result<Option<String>> {
        let name = key.format(self.arch);
        let Some(id) = self.kv_id(&name) else {
            if required {
                return Err(Error::new(format!(
                    "missing required metadata key: {}",
                    name
                )));
            }
            return Ok(None);
        };
        let kv = &self.metadata.kv[id];
        if kv.get_type() != GgufType::String || kv.is_array {
            return Err(Error::new(format!("{} must be a string", name)));
        }
        Ok(kv.data_string.first().cloned())
    }

    pub fn get_u32(&self, key: crate::arch::Kv, required: bool) -> Result<Option<u32>> {
        let name = key.format(self.arch);
        let Some(id) = self.kv_id(&name) else {
            if required {
                return Err(Error::new(format!(
                    "missing required metadata key: {}",
                    name
                )));
            }
            return Ok(None);
        };
        let kv = &self.metadata.kv[id];
        if kv.is_array || kv.get_type() != GgufType::Uint32 {
            return Err(Error::new(format!("{} must be a u32", name)));
        }
        Ok(Some(kv.get_val(0)))
    }

    pub fn get_i32(&self, key: crate::arch::Kv, required: bool) -> Result<Option<i32>> {
        let name = key.format(self.arch);
        let Some(id) = self.kv_id(&name) else {
            if required {
                return Err(Error::new(format!(
                    "missing required metadata key: {}",
                    name
                )));
            }
            return Ok(None);
        };
        let kv = &self.metadata.kv[id];
        if kv.is_array || kv.get_type() != GgufType::Int32 {
            return Err(Error::new(format!("{} must be an i32", name)));
        }
        Ok(Some(kv.get_val(0)))
    }

    pub fn get_f32(&self, key: crate::arch::Kv, required: bool) -> Result<Option<f32>> {
        let name = key.format(self.arch);
        let Some(id) = self.kv_id(&name) else {
            if required {
                return Err(Error::new(format!(
                    "missing required metadata key: {}",
                    name
                )));
            }
            return Ok(None);
        };
        let kv = &self.metadata.kv[id];
        if kv.is_array || kv.get_type() != GgufType::Float32 {
            return Err(Error::new(format!("{} must be an f32", name)));
        }
        Ok(Some(kv.get_val(0)))
    }

    pub fn get_bool(&self, key: crate::arch::Kv, required: bool) -> Result<Option<bool>> {
        let name = key.format(self.arch);
        let Some(id) = self.kv_id(&name) else {
            if required {
                return Err(Error::new(format!(
                    "missing required metadata key: {}",
                    name
                )));
            }
            return Ok(None);
        };
        let kv = &self.metadata.kv[id];
        if kv.is_array || kv.get_type() != GgufType::Bool {
            return Err(Error::new(format!("{} must be a bool", name)));
        }
        Ok(Some(kv.data.first().copied().unwrap_or(0) != 0))
    }

    fn arr_id(&self, name: &str, ty: GgufType) -> Result<Option<usize>> {
        let Some(id) = self.kv_id(name) else {
            return Ok(None);
        };
        let kv = &self.metadata.kv[id];
        if !kv.is_array || kv.get_type() != ty {
            return Err(Error::new(format!("{} must be an array of {:?}", name, ty)));
        }
        Ok(Some(id))
    }

    pub fn get_arr_string(&self, key: crate::arch::Kv, required: bool) -> Result<Vec<String>> {
        let name = key.format(self.arch);
        let Some(id) = self.arr_id(&name, GgufType::String)? else {
            if required {
                return Err(Error::new(format!(
                    "missing required metadata key: {}",
                    name
                )));
            }
            return Ok(Vec::new());
        };
        Ok(self.metadata.kv[id].data_string.clone())
    }

    pub fn get_arr_i32(&self, key: crate::arch::Kv, required: bool) -> Result<Vec<i32>> {
        let name = key.format(self.arch);
        let Some(id) = self.arr_id(&name, GgufType::Int32)? else {
            if required {
                return Err(Error::new(format!(
                    "missing required metadata key: {}",
                    name
                )));
            }
            return Ok(Vec::new());
        };
        let data = &self.metadata.kv[id].data;
        Ok(data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn get_arr_u32(&self, key: crate::arch::Kv, required: bool) -> Result<Vec<u32>> {
        let name = key.format(self.arch);
        let Some(id) = self.arr_id(&name, GgufType::Uint32)? else {
            if required {
                return Err(Error::new(format!(
                    "missing required metadata key: {}",
                    name
                )));
            }
            return Ok(Vec::new());
        };
        let data = &self.metadata.kv[id].data;
        Ok(data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn get_arr_f32(&self, key: crate::arch::Kv, required: bool) -> Result<Vec<f32>> {
        let name = key.format(self.arch);
        let Some(id) = self.arr_id(&name, GgufType::Float32)? else {
            if required {
                return Err(Error::new(format!(
                    "missing required metadata key: {}",
                    name
                )));
            }
            return Ok(Vec::new());
        };
        let data = &self.metadata.kv[id].data;
        Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn get_arr_bool(&self, key: crate::arch::Kv, required: bool) -> Result<Vec<bool>> {
        let name = key.format(self.arch);
        let Some(id) = self.arr_id(&name, GgufType::Bool)? else {
            if required {
                return Err(Error::new(format!(
                    "missing required metadata key: {}",
                    name
                )));
            }
            return Ok(Vec::new());
        };
        Ok(self.metadata.kv[id].data.iter().map(|&b| b != 0).collect())
    }

    /// `ml.get_key_or_arr` for `u32` per-layer values.
    pub fn get_key_or_arr_u32(
        &self,
        key: crate::arch::Kv,
        dst: &mut [u32; LLAMA_MAX_LAYERS],
        n: u32,
        required: bool,
    ) -> Result<()> {
        if let Some(v) = self.get_u32(key, false)? {
            for slot in dst.iter_mut().take(n as usize) {
                *slot = v;
            }
            return Ok(());
        }
        let arr = self.get_arr_u32(key, required)?;
        if arr.is_empty() {
            return Ok(());
        }
        if arr.len() < n as usize {
            return Err(Error::new(format!(
                "{} has {} values, expected {}",
                key.format(self.arch),
                arr.len(),
                n
            )));
        }
        for (slot, &v) in dst.iter_mut().zip(arr.iter()) {
            *slot = v;
        }
        Ok(())
    }

    /// `ml.get_key_or_arr` for `i32` per-layer values.
    pub fn get_key_or_arr_i32(
        &self,
        key: crate::arch::Kv,
        dst: &mut [i32; LLAMA_MAX_LAYERS],
        n: u32,
        required: bool,
    ) -> Result<()> {
        if let Some(v) = self.get_i32(key, false)? {
            for slot in dst.iter_mut().take(n as usize) {
                *slot = v;
            }
            return Ok(());
        }
        let arr = self.get_arr_i32(key, required)?;
        if arr.is_empty() {
            return Ok(());
        }
        if arr.len() < n as usize {
            return Err(Error::new(format!(
                "{} has {} values, expected {}",
                key.format(self.arch),
                arr.len(),
                n
            )));
        }
        for (slot, &v) in dst.iter_mut().zip(arr.iter()) {
            *slot = v;
        }
        Ok(())
    }

    /// True if the GGUF contains a tensor with this exact name.
    pub fn get_weight(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn has_file_tensor(&self, name: &str) -> bool {
        self.metadata.info.iter().any(|info| info.t.name == name)
    }

    /// `ml.create_tensor`: resolve the name, check the shape and copy data.
    pub fn create_tensor(
        &mut self,
        tn: &LlmTnImpl,
        ne: &[i64],
        flags: u32,
        buft: Option<&BufferType>,
    ) -> Result<Option<Tensor>> {
        use super::types::tensor_flags;
        if flags & tensor_flags::SKIP != 0 {
            self.skipped.push(tn.str()?);
            return Ok(None);
        }
        let name = tn.str()?;
        let id = self
            .metadata
            .info
            .iter()
            .position(|info| info.t.name == name);
        let Some(id) = id else {
            if flags & (tensor_flags::NOT_REQUIRED | tensor_flags::SKIP_IF_VIRTUAL) != 0 {
                return Ok(None);
            }
            return Err(Error::new(format!("tensor not found: {}", name)));
        };
        let info = &self.metadata.info[id];
        if !ne.is_empty() {
            let mut dims = ne.to_vec();
            while dims.len() < 4 {
                dims.push(1);
            }
            dims.truncate(4);
            for i in 0..dims.len().min(4) {
                if dims[i] > 0
                    && info.t.ne[i] != dims[i]
                    && flags & tensor_flags::ALLOW_RESHAPE == 0
                {
                    return Err(Error::new(format!(
                        "tensor '{}' has dimension {} = {}, expected {}",
                        name, i, info.t.ne[i], dims[i]
                    )));
                }
            }
        }
        let mut tensor = Tensor::new(name.clone(), &info.t.ne, info.t.type_);
        if !self.no_alloc {
            let start = info.offset as usize;
            let size = ggml_nbytes(&info.t);
            let end = start.saturating_add(size);
            tensor.data = self
                .metadata
                .data
                .get(start..end.min(self.metadata.data.len()))
                .unwrap_or(&[])
                .to_vec();
        }
        tensor.buffer = buft.map(|b| b.name.clone());
        self.tensors.insert(name, tensor.clone());
        Ok(Some(tensor))
    }

    pub fn done_getting_tensors(&mut self) {
        // The Rust port copies bytes during create_tensor; no lazy file state
        // remains to close here.
    }

    /// Materialize data for tensors created without allocation.
    pub fn load_all_data(&mut self) -> Result<()> {
        for tensor in self.tensors.values_mut() {
            if !tensor.data.is_empty() {
                continue;
            }
            let Some(info) = self.metadata.info.iter().find(|i| i.t.name == tensor.name) else {
                continue;
            };
            let start = info.offset as usize;
            let size = ggml_nbytes(&info.t);
            tensor.data = self
                .metadata
                .data
                .get(start..start.saturating_add(size).min(self.metadata.data.len()))
                .unwrap_or(&[])
                .to_vec();
        }
        Ok(())
    }

    /// Expose GGUF metadata in the shape expected by `llama_vocab::load`.
    pub fn vocab_metadata(&self) -> crate::vocab::Metadata {
        let mut out = crate::vocab::Metadata::default();
        for kv in &self.metadata.kv {
            match kv.get_type() {
                GgufType::String => {
                    if kv.is_array {
                        out.string_arrays
                            .insert(kv.key.clone(), kv.data_string.clone());
                    } else if let Some(s) = kv.data_string.first() {
                        out.strings.insert(kv.key.clone(), s.clone());
                    }
                }
                GgufType::Bool => {
                    if kv.is_array {
                        out.bools
                            .insert(kv.key.clone(), kv.data.first().copied().unwrap_or(0) != 0);
                    } else {
                        out.bools
                            .insert(kv.key.clone(), kv.data.first().copied().unwrap_or(0) != 0);
                    }
                }
                GgufType::Uint32 => {
                    if kv.is_array {
                        let vals: Vec<u32> = kv
                            .data
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        out.i32_arrays
                            .insert(kv.key.clone(), vals.into_iter().map(|v| v as i32).collect());
                    } else if !kv.data.is_empty() {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(&kv.data[..4]);
                        out.u32s.insert(kv.key.clone(), u32::from_le_bytes(b));
                    }
                }
                GgufType::Int32 => {
                    if kv.is_array {
                        out.i32_arrays.insert(
                            kv.key.clone(),
                            kv.data
                                .chunks_exact(4)
                                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect(),
                        );
                    }
                }
                GgufType::Float32 => {
                    if kv.is_array {
                        out.f32_arrays.insert(
                            kv.key.clone(),
                            kv.data
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect(),
                        );
                    }
                }
                _ => {
                    if !kv.data.is_empty() {
                        out.blobs.insert(kv.key.clone(), kv.data.clone());
                    }
                }
            }
        }
        out
    }

    /// `gguf_kv_to_str` replacement for one key/value pair.
    pub fn kv_to_string(kv: &GgufKv) -> String {
        match kv.get_type() {
            GgufType::String => kv.data_string.first().cloned().unwrap_or_default(),
            GgufType::Bool => kv
                .data
                .first()
                .map(|&v| if v != 0 { "true" } else { "false" }.to_owned())
                .unwrap_or_default(),
            GgufType::Uint32
            | GgufType::Int32
            | GgufType::Uint64
            | GgufType::Int64
            | GgufType::Float32
            | GgufType::Float64 => {
                let mut parts = Vec::new();
                let ty = kv.get_type();
                let size = crate::gguf::gguf_type_size(ty);
                for chunk in kv.data.chunks(size.max(1)) {
                    match ty {
                        GgufType::Uint32 | GgufType::Int32 if chunk.len() == 4 => {
                            let mut b = [0u8; 4];
                            b.copy_from_slice(chunk);
                            if ty == GgufType::Uint32 {
                                parts.push(u32::from_le_bytes(b).to_string());
                            } else {
                                parts.push(i32::from_le_bytes(b).to_string());
                            }
                        }
                        GgufType::Float32 if chunk.len() == 4 => {
                            let mut b = [0u8; 4];
                            b.copy_from_slice(chunk);
                            parts.push(f32::from_le_bytes(b).to_string());
                        }
                        _ => parts.push(format!("{:02x?}", chunk)),
                    }
                }
                if kv.is_array {
                    format!("[{}]", parts.join(", "))
                } else {
                    parts.first().cloned().unwrap_or_default()
                }
            }
            GgufType::Array => "[...]".to_owned(),
            _ => format!("unknown type {:?}", kv.get_type()),
        }
    }

    /// Keep only the string arrays that survive `load_hparams`.
    pub fn kept_string_arrays(&self) -> BTreeMap<String, Vec<String>> {
        let mut out = BTreeMap::new();
        for kv in &self.metadata.kv {
            if kv.is_array
                && kv.get_type() == GgufType::String
                && kv.key.starts_with("tokenizer.tts_dict")
            {
                out.insert(kv.key.clone(), kv.data_string.clone());
            }
        }
        out
    }

    /// Metadata alignment, used when constructing synthetic GGUF data.
    pub fn alignment(&self) -> usize {
        self.metadata
            .kv
            .iter()
            .find(|kv| kv.key == GGUF_KEY_GENERAL_ALIGNMENT)
            .and_then(|kv| {
                if kv.get_type() == GgufType::Uint32 && !kv.data.is_empty() {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&kv.data[..4]);
                    Some(u32::from_le_bytes(b) as usize)
                } else {
                    None
                }
            })
            .unwrap_or(self.metadata.alignment)
    }
}

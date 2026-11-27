//! GGUF writers: metadata and tensor data serialization.

use std::fs::File;
use std::io::Write;

use crate::core::{Error, Result};

use super::kv::{gguf_get_n_kv, gguf_get_n_tensors, GgufKv};
use super::types::{GgufType, GgufValue, GgmlType, GGUF_MAGIC};
use super::{ggml_is_contiguous, ggml_nbytes, ggml_n_dims, GgufContext, GgufTensorInfo};


// ---------------------------------------------------------------------------
// writer
// ---------------------------------------------------------------------------

pub trait GgufWriter {
    fn write_i8(&mut self, val: i8) -> Result<()>;
    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()>;
    fn written_bytes(&self) -> usize;

    fn write<T: GgufValue>(&mut self, val: T) -> Result<()> {
        self.write_bytes(&val.encode())
    }

    fn write_bool(&mut self, val: bool) -> Result<()> {
        self.write_i8(if val { 1 } else { 0 })
    }

    fn write_string(&mut self, val: &str) -> Result<()> {
        self.write(val.len() as u64)?;
        self.write_bytes(val.as_bytes())
    }

    fn write_guf_type(&mut self, val: GgufType) -> Result<()> {
        self.write(val as i32)
    }

    fn write_ggml_type(&mut self, val: GgmlType) -> Result<()> {
        self.write(val as i32)
    }

    fn write_kv(&mut self, kv: &GgufKv) -> Result<()> {
        let ne = kv.get_ne() as u64;

        self.write_string(kv.get_key())?;

        if kv.is_array {
            self.write_guf_type(GgufType::Array)?;
            self.write_guf_type(kv.get_type())?;
            self.write(ne)?;
        } else {
            self.write_guf_type(kv.get_type())?;
        }

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
            | GgufType::Float64 => {
                self.write_bytes(&kv.data)?;
            }
            GgufType::Bool => {
                for i in 0..ne as usize {
                    self.write_bool(kv.get_val(i))?;
                }
            }
            GgufType::String => {
                for i in 0..ne as usize {
                    self.write_string(kv.get_str(i))?;
                }
            }
            GgufType::Array | GgufType::Count => panic!("invalid type"),
        }
        Ok(())
    }

    fn write_tensor_meta(&mut self, info: &GgufTensorInfo) -> Result<()> {
        self.write_string(&info.t.name)?;

        let n_dims = ggml_n_dims(&info.t);
        self.write(n_dims as u32)?;

        for j in 0..n_dims {
            self.write(info.t.ne[j])?;
        }
        self.write_ggml_type(info.t.type_)?;
        self.write(info.offset)?;
        Ok(())
    }

    fn pad(&mut self, alignment: usize) -> Result<()> {
        while self.written_bytes() % alignment != 0 {
            self.write_i8(0)?;
        }
        Ok(())
    }

    fn write_tensor_data(
        &mut self,
        info: &GgufTensorInfo,
        offset_data: usize,
        alignment: usize,
    ) -> Result<()> {
        assert_eq!(self.written_bytes() - offset_data, info.offset as usize);
        assert!(ggml_is_contiguous(&info.t));

        let nbytes = ggml_nbytes(&info.t);
        let mut bytes = vec![0u8; nbytes];
        let n = nbytes.min(info.t.data.len());
        bytes[..n].copy_from_slice(&info.t.data[..n]);
        self.write_bytes(&bytes)?;

        self.pad(alignment)
    }
}

/// Vector buffer based writer.
pub struct GgufWriterBuf {
    pub buf: Vec<u8>,
}

impl GgufWriterBuf {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }
}

impl Default for GgufWriterBuf {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter for GgufWriterBuf {
    fn write_i8(&mut self, val: i8) -> Result<()> {
        self.buf.push(val as u8);
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    fn written_bytes(&self) -> usize {
        self.buf.len()
    }
}

/// File based writer.
pub struct GgufWriterFile {
    pub file: File,
    written: usize,
}

impl GgufWriterFile {
    pub fn new(file: File) -> Self {
        Self { file, written: 0 }
    }
}

impl GgufWriter for GgufWriterFile {
    fn write_i8(&mut self, val: i8) -> Result<()> {
        self.file
            .write_all(&[val as u8])
            .map_err(|e| Error::new(e.to_string()))?;
        self.written += 1;
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.file
            .write_all(bytes)
            .map_err(|e| Error::new(e.to_string()))?;
        self.written += bytes.len();
        Ok(())
    }

    fn written_bytes(&self) -> usize {
        self.written
    }
}

pub fn gguf_write_out<W: GgufWriter>(
    ctx: &GgufContext,
    gw: &mut W,
    only_meta: bool,
) -> Result<()> {
    let n_kv = gguf_get_n_kv(ctx);
    let n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.write_bytes(GGUF_MAGIC)?;
    gw.write(ctx.version)?;
    gw.write(n_tensors)?;
    gw.write(n_kv)?;

    // write key-value pairs
    for kv in &ctx.kv {
        gw.write_kv(kv)?;
    }

    // write tensor info
    for info in &ctx.info {
        gw.write_tensor_meta(info)?;
    }

    // we require the data section to be aligned
    gw.pad(ctx.alignment)?;

    if only_meta {
        return Ok(());
    }

    let offset_data = gw.written_bytes();

    // write tensor data
    for info in &ctx.info {
        gw.write_tensor_data(info, offset_data, ctx.alignment)?;
    }

    Ok(())
}

pub fn gguf_write_to_buf(ctx: &GgufContext, only_meta: bool) -> Vec<u8> {
    let mut gw = GgufWriterBuf::new();
    gguf_write_out(ctx, &mut gw, only_meta).expect("gguf buffer write failed");
    gw.buf
}

pub fn gguf_write_to_file_ptr(ctx: &GgufContext, file: File, only_meta: bool) -> bool {
    let mut gw = GgufWriterFile::new(file);
    match gguf_write_out(ctx, &mut gw, only_meta) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("gguf_write_to_file_ptr: failed to write GGUF data: {}", e);
            false
        }
    }
}

pub fn gguf_write_to_file(ctx: &GgufContext, fname: &str, only_meta: bool) -> bool {
    let file = match File::create(fname) {
        Ok(file) => file,
        Err(e) => {
            eprintln!(
                "gguf_write_to_file: failed to open file '{}' for writing GGUF data ({})",
                fname, e
            );
            return false;
        }
    };

    let success = gguf_write_to_file_ptr(ctx, file, only_meta);
    if !success {
        eprintln!(
            "gguf_write_to_file: failed to write GGUF data into '{}'",
            fname
        );
    }
    success
}

/// Size in bytes of the meta data (header, KV pairs, tensor info) including
/// padding.
pub fn gguf_get_meta_size(ctx: &GgufContext) -> usize {
    gguf_write_to_buf(ctx, true).len()
}

/// Writes the meta data to a freshly allocated byte vector.
pub fn gguf_get_meta_data(ctx: &GgufContext) -> Vec<u8> {
    gguf_write_to_buf(ctx, true)
}

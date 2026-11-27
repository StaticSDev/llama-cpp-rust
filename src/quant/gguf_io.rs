//! Minimal GGUF reader/writer used by the quantization driver.
//!
//! Kept local to `src/quant` so the module can use the current ggml type enum
//! without depending on an older shared stand-in.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::core::{Error, Result};

use super::types::{QTensor, QType};
use super::GGUF_ALIGNMENT;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;

#[derive(Clone, Debug)]
pub struct GgufKv {
    pub key: String,
    pub type_: i32,
    pub is_array: bool,
    pub elem_type: i32,
    pub data: Vec<u8>,
    pub strings: Vec<String>,
}

fn scalar_size(ty: i32) -> usize {
    match ty {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        10 | 11 | 12 => 8,
        _ => 0,
    }
}

impl GgufKv {
    fn read(cur: &mut Cursor) -> Result<Self> {
        let key = cur.read_string()?;
        let type_ = cur.read_i32()?;
        if type_ == 9 {
            let elem_type = cur.read_i32()?;
            let n = cur.read_u64()? as usize;
            if n > 1 << 26 {
                return Err(Error::new("GGUF array too large"));
            }
            if elem_type == 8 {
                let mut strings = Vec::with_capacity(n);
                for _ in 0..n {
                    strings.push(cur.read_string()?);
                }
                return Ok(GgufKv {
                    key,
                    type_,
                    is_array: true,
                    elem_type,
                    data: Vec::new(),
                    strings,
                });
            }
            let sz = scalar_size(elem_type);
            if sz == 0 {
                return Err(Error::new(format!(
                    "invalid GGUF array element type {}",
                    elem_type
                )));
            }
            let data = cur.read_bytes(n.checked_mul(sz).unwrap_or(usize::MAX))?;
            return Ok(GgufKv {
                key,
                type_,
                is_array: true,
                elem_type,
                data: data.to_vec(),
                strings: Vec::new(),
            });
        }
        if type_ == 8 {
            let s = cur.read_string()?;
            return Ok(GgufKv {
                key,
                type_,
                is_array: false,
                elem_type: type_,
                data: Vec::new(),
                strings: vec![s],
            });
        }
        let sz = scalar_size(type_);
        if sz == 0 {
            return Err(Error::new(format!("invalid GGUF value type {}", type_)));
        }
        let data = cur.read_bytes(sz)?;
        Ok(GgufKv {
            key,
            type_,
            is_array: false,
            elem_type: type_,
            data: data.to_vec(),
            strings: Vec::new(),
        })
    }

    fn serialized_len(&self) -> usize {
        let mut n = 8 + self.key.len() + 4;
        if self.type_ == 9 {
            n += 4 + 8;
            if self.elem_type == 8 {
                for s in &self.strings {
                    n += 8 + s.len();
                }
            } else {
                n += self.data.len();
            }
        } else if self.type_ == 8 {
            n += 8 + self.strings.first().map(|s| s.len()).unwrap_or(0);
        } else {
            n += self.data.len();
        }
        n
    }

    fn write_to(&self, out: &mut Vec<u8>) {
        write_string(out, &self.key);
        write_i32(out, self.type_);
        if self.type_ == 9 {
            write_i32(out, self.elem_type);
            let n = if self.elem_type == 8 {
                self.strings.len()
            } else {
                self.data.len() / scalar_size(self.elem_type).max(1)
            };
            write_u64(out, n as u64);
            if self.elem_type == 8 {
                for s in &self.strings {
                    write_string(out, s);
                }
            } else {
                out.extend_from_slice(&self.data);
            }
        } else if self.type_ == 8 {
            write_string(out, self.strings.first().map(String::as_str).unwrap_or(""));
        } else {
            out.extend_from_slice(&self.data);
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        if self.type_ != 4 || self.data.len() < 4 {
            return None;
        }
        Some(u32::from_le_bytes(self.data[..4].try_into().ok()?))
    }

    pub fn as_i32(&self) -> Option<i32> {
        if self.type_ != 5 || self.data.len() < 4 {
            return None;
        }
        Some(i32::from_le_bytes(self.data[..4].try_into().ok()?))
    }

    pub fn as_str(&self) -> Option<&str> {
        if self.type_ != 8 {
            return None;
        }
        self.strings.first().map(String::as_str)
    }

    pub fn set_u32(&mut self, value: u32) {
        self.type_ = 4;
        self.elem_type = 4;
        self.is_array = false;
        self.data = value.to_le_bytes().to_vec();
        self.strings.clear();
    }

    pub fn set_f32(&mut self, value: f32) {
        self.type_ = 6;
        self.elem_type = 6;
        self.is_array = false;
        self.data = value.to_le_bytes().to_vec();
        self.strings.clear();
    }

    pub fn new_u32(key: &str, value: u32) -> Self {
        Self {
            key: key.to_owned(),
            type_: 4,
            is_array: false,
            elem_type: 4,
            data: value.to_le_bytes().to_vec(),
            strings: Vec::new(),
        }
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::new("GGUF offset overflow"))?;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| Error::new("unexpected end of GGUF file"))?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_string(&mut self) -> Result<String> {
        let n = self.read_u64()? as usize;
        let b = self.read_bytes(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| Error::new("invalid UTF-8 in GGUF file"))
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Parsed GGUF input.
pub struct LoadedGguf {
    pub version: u32,
    pub kv: Vec<GgufKv>,
    pub tensors: Vec<QTensor>,
    pub meta_bytes: Vec<u8>,
    pub meta_size: usize,
    pub alignment: usize,
    pub general_name: String,
    pub arch_name: String,
    pub block_count: u32,
    pub shared_kv: u32,
    pub source_ftype: i32,
}

fn get_kv<'a>(kv: &'a [GgufKv], key: &str) -> Option<&'a GgufKv> {
    kv.iter().find(|k| k.key == key)
}

fn get_kv_mut<'a>(kv: &'a mut [GgufKv], key: &str) -> Option<&'a mut GgufKv> {
    kv.iter_mut().find(|k| k.key == key)
}

pub fn read_gguf(path: &str) -> Result<LoadedGguf> {
    let mut file =
        File::open(path).map_err(|e| Error::new(format!("failed to open {}: {}", path, e)))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| Error::new(format!("failed to read {}: {}", path, e)))?;
    let mut cur = Cursor::new(&bytes);
    if cur.read_bytes(4)? != GGUF_MAGIC {
        return Err(Error::new("GGUF magic mismatch"));
    }
    let version = cur.read_u32()?;
    let n_tensors = cur.read_u64()? as usize;
    let n_kv = cur.read_u64()? as usize;
    let mut kv = Vec::with_capacity(n_kv);
    for _ in 0..n_kv {
        kv.push(GgufKv::read(&mut cur)?);
    }
    let alignment = get_kv(&kv, "general.alignment")
        .and_then(|k| k.as_u32())
        .unwrap_or(GGUF_ALIGNMENT as u32) as usize;
    let meta_start = cur.pos;
    let mut meta_bytes = Vec::new();
    for k in &kv {
        k.write_to(&mut meta_bytes);
    }

    let mut tensors = Vec::with_capacity(n_tensors);
    let mut infos = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = cur.read_string()?;
        let n_dims = cur.read_u32()? as usize;
        if n_dims == 0 || n_dims > 4 {
            return Err(Error::new(format!(
                "invalid tensor rank {} for {}",
                n_dims, name
            )));
        }
        let mut ne = [1i64; 4];
        for d in 0..n_dims {
            ne[d] = cur.read_i64()?;
        }
        let raw_ty = cur.read_i32()?;
        let ty = QType::from_i32(raw_ty)
            .ok_or_else(|| Error::new(format!("invalid ggml type {} for {}", raw_ty, name)))?;
        let offset = cur.read_u64()? as usize;
        infos.push((name, ne, ty, offset));
    }

    let data_start = cur.pos;
    let pad = (alignment - data_start % alignment) % alignment;
    let data_start = data_start + pad;
    if data_start > bytes.len() {
        return Err(Error::new("GGUF data offset is beyond end of file"));
    }
    let data = bytes[data_start..].to_vec();

    for (name, ne, ty, offset) in infos {
        let expected = ty.row_size(ne[0])
            * ne[1].max(0) as usize
            * ne[2].max(0) as usize
            * ne[3].max(0) as usize;
        let slice = data
            .get(offset..offset + expected)
            .ok_or_else(|| Error::new(format!("tensor {} data range is out of bounds", name)))?;
        tensors.push(QTensor {
            name,
            ne,
            ty,
            data: slice.to_vec(),
            split: 0,
            offset: offset as u64,
        });
    }

    let general_name = get_kv(&kv, "general.name")
        .and_then(|k| k.as_str())
        .unwrap_or("")
        .to_owned();
    let arch_name = get_kv(&kv, "general.architecture")
        .and_then(|k| k.as_str())
        .unwrap_or("")
        .to_owned();
    let block_count = get_kv(&kv, "block_count")
        .and_then(|k| k.as_u32())
        .unwrap_or(0);
    let shared_kv = get_kv(&kv, "attention.shared_kv_layers")
        .and_then(|k| k.as_u32())
        .unwrap_or(0);
    let source_ftype = get_kv(&kv, "general.file_type")
        .and_then(|k| k.as_i32())
        .unwrap_or(-1);

    Ok(LoadedGguf {
        version,
        kv,
        tensors,
        meta_size: meta_bytes.len(),
        meta_bytes,
        alignment,
        general_name,
        arch_name,
        block_count,
        shared_kv,
        source_ftype,
    })
}

fn meta_bytes_len(bytes: &[u8]) -> usize {
    bytes.len()
}

/// Tensor with a finalized target type and data, ready to write.
#[derive(Clone, Copy)]
pub struct OutputTensor<'a> {
    pub name: &'a str,
    pub ne: [i64; 4],
    pub ty: QType,
    pub data: &'a [u8],
    pub split: u16,
}

/// Write a GGUF file with placeholder metadata patched in place at the end.
pub fn write_gguf(
    path: &str,
    kv: &mut Vec<GgufKv>,
    tensors: &[OutputTensor<'_>],
    alignment: usize,
) -> Result<()> {
    let mut info = Vec::new();
    info.extend_from_slice(GGUF_MAGIC);
    write_u32(&mut info, GGUF_VERSION);
    write_u64(&mut info, tensors.len() as u64);
    write_u64(&mut info, kv.len() as u64);
    for k in kv.iter() {
        k.write_to(&mut info);
    }
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut data = Vec::new();
    for t in tensors {
        let pad = (alignment - data.len() % alignment) % alignment;
        data.resize(data.len() + pad, 0);
        offsets.push(data.len() as u64);
        data.extend_from_slice(t.data);
        let pad = (alignment - data.len() % alignment) % alignment;
        data.resize(data.len() + pad, 0);
    }
    for (t, off) in tensors.iter().zip(offsets) {
        write_string(&mut info, t.name);
        write_u32(&mut info, 4);
        for &d in &t.ne {
            write_u64(&mut info, d as u64);
        }
        write_i32(&mut info, t.ty as i32);
        write_u64(&mut info, off);
    }

    let mut file =
        File::create(path).map_err(|e| Error::new(format!("failed to create {}: {}", path, e)))?;
    file.write_all(&info)
        .map_err(|e| Error::new(format!("failed to write {}: {}", path, e)))?;
    file.write_all(&data)
        .map_err(|e| Error::new(format!("failed to write {}: {}", path, e)))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| Error::new(format!("failed to seek {}: {}", path, e)))?;
    file.write_all(&info)
        .map_err(|e| Error::new(format!("failed to patch {}: {}", path, e)))?;
    Ok(())
}

/// Helpers used by the driver to maintain the output KV list.
pub fn set_u32(kv: &mut Vec<GgufKv>, key: &str, value: u32) {
    if let Some(k) = get_kv_mut(kv, key) {
        k.set_u32(value);
    } else {
        kv.push(GgufKv::new_u32(key, value));
    }
}

pub fn remove_key(kv: &mut Vec<GgufKv>, key: &str) {
    kv.retain(|k| k.key != key);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_round_trip() {
        let mut kv = vec![GgufKv::new_u32("test.key", 7)];
        let mut out = Vec::new();
        for k in &kv {
            k.write_to(&mut out);
        }
        let mut cur = Cursor::new(&out);
        let parsed = GgufKv::read(&mut cur).unwrap();
        assert_eq!(parsed.key, "test.key");
        assert_eq!(parsed.as_u32(), Some(7));
        set_u32(&mut kv, "test.key", 9);
        assert_eq!(kv[0].as_u32(), Some(9));
    }
}

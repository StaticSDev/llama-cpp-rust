//! GGUF reader, initialization, and header/tensor parsing.

use std::collections::HashSet;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use crate::core::{Error, Result};

use super::kv::{gguf_find_key, gguf_get_kv_type, gguf_get_val_u32, GgufKv};
use super::types::{
    ggml_blck_size, ggml_type_name, ggml_type_size, gguf_pad, gguf_type_name, GgmlType, GgufType,
    GgufValue, GGML_MAX_DIMS, GGML_MAX_NAME, GGML_TYPE_COUNT, GGUF_DEFAULT_ALIGNMENT,
    GGUF_KEY_GENERAL_ALIGNMENT, GGUF_MAGIC, GGUF_MAX_ARRAY_ELEMENTS, GGUF_MAX_STRING_LENGTH,
    GGUF_VERSION,
};
use super::{
    ggml_nbytes, GgmlContext, GgmlTensor, GgufContext, GgufInitParams, GgufTensorInfo,
};

// ---------------------------------------------------------------------------
// reader
// ---------------------------------------------------------------------------

/// Callback to simulate or wrap a FILE pointer: read up to `output.len()`
/// bytes at `offset` into `output` and return the number of bytes read.
pub type GgufReaderCallback<'a> = dyn FnMut(u64, &mut [u8]) -> usize + 'a;

pub struct GgufReader<F>
where
    F: FnMut(u64, &mut [u8]) -> usize,
{
    callback: F,
    max_chunk_read: usize,
    start_offset: u64,
    data_offset: u64,
    nbytes_remain: u64,
}

impl<F> GgufReader<F>
where
    F: FnMut(u64, &mut [u8]) -> usize,
{
    pub fn new(
        callback: F,
        max_chunk_read: usize,
        data_offset: u64,
        nbytes_remain: u64,
    ) -> Self {
        assert!(max_chunk_read > 0);
        Self {
            callback,
            max_chunk_read,
            start_offset: data_offset,
            data_offset,
            nbytes_remain,
        }
    }

    fn read_raw(&mut self, dst: &mut [u8]) -> usize {
        let size = dst.len();
        if size == 0 {
            return 0;
        }

        let mut total_nread = 0usize;
        let mut reached_eof = false;

        while total_nread < size {
            let chunk_size = self.max_chunk_read.min(size - total_nread);
            if self.data_offset.wrapping_add(total_nread as u64) < self.data_offset {
                break;
            }
            let nread = (self.callback)(
                self.data_offset + total_nread as u64,
                &mut dst[total_nread..total_nread + chunk_size],
            );
            let nread = nread.min(chunk_size);
            total_nread += nread;
            if nread != chunk_size {
                reached_eof = true;
                break;
            }
        }

        self.data_offset += total_nread as u64;
        assert!(total_nread as u64 <= self.nbytes_remain);
        self.nbytes_remain -= total_nread as u64;

        if reached_eof {
            self.nbytes_remain = 0;
        }

        total_nread
    }

    pub fn read_bytes(&mut self, dst: &mut [u8]) -> Result<()> {
        let size = dst.len();
        if size as u64 > self.nbytes_remain {
            return Err(Error::new("gguf_reader: read past remaining bytes"));
        }
        if self.read_raw(dst) != size {
            return Err(Error::new("gguf_reader: failed to read raw bytes"));
        }
        Ok(())
    }

    pub fn read<T: GgufValue>(&mut self) -> Result<T> {
        let mut buf = vec![0u8; std::mem::size_of::<T>()];
        self.read_bytes(&mut buf)?;
        Ok(T::decode(&buf))
    }

    pub fn read_bool(&mut self) -> Result<bool> {
        let tmp = self.read::<i8>()?;
        Ok(tmp != 0)
    }

    pub fn read_string(&mut self) -> Result<String> {
        let size = self.read::<u64>()?;
        if size > GGUF_MAX_STRING_LENGTH as u64 {
            return Err(Error::new(format!(
                "gguf_reader: string length {} exceeds maximum {}",
                size, GGUF_MAX_STRING_LENGTH
            )));
        }
        if size > self.nbytes_remain {
            return Err(Error::new(format!(
                "gguf_reader: string length {} exceeds remaining file size {} bytes",
                size, self.nbytes_remain
            )));
        }
        let mut buf = vec![0u8; size as usize];
        if self.read_raw(&mut buf) != size as usize {
            return Err(Error::new("gguf_reader: failed to read string"));
        }
        String::from_utf8(buf)
            .map_err(|e| Error::new(format!("gguf_reader: invalid UTF-8 string: {}", e)))
    }

    pub fn read_vec<T: GgufValue>(&mut self, n: usize) -> Result<Vec<T>> {
        if n > GGUF_MAX_ARRAY_ELEMENTS {
            return Err(Error::new("gguf_reader: array too large"));
        }
        let type_size = std::mem::size_of::<T>();
        if type_size > 0 && n > usize::MAX / type_size {
            return Err(Error::new("gguf_reader: array size overflow"));
        }
        let total = n
            .checked_mul(type_size)
            .ok_or_else(|| Error::new("gguf_reader: array size overflow"))?;
        if self.nbytes_remain < total as u64 {
            return Err(Error::new("gguf_reader: array exceeds remaining bytes"));
        }
        let mut dst = Vec::with_capacity(n);
        for _ in 0..n {
            dst.push(self.read::<T>()?);
        }
        Ok(dst)
    }

    pub fn read_bool_vec(&mut self, n: usize) -> Result<Vec<bool>> {
        if n > GGUF_MAX_ARRAY_ELEMENTS {
            return Err(Error::new("gguf_reader: array too large"));
        }
        if self.nbytes_remain < n as u64 {
            return Err(Error::new("gguf_reader: array exceeds remaining bytes"));
        }
        let mut dst = Vec::with_capacity(n);
        for _ in 0..n {
            dst.push(self.read_bool()?);
        }
        Ok(dst)
    }

    pub fn read_string_vec(&mut self, n: usize) -> Result<Vec<String>> {
        if n > GGUF_MAX_ARRAY_ELEMENTS {
            return Err(Error::new("gguf_reader: array too large"));
        }
        // Strings are prefixed with their length, so account for that.
        if n > usize::MAX / std::mem::size_of::<u64>() {
            return Err(Error::new("gguf_reader: string array size overflow"));
        }
        let prefix_bytes = n
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| Error::new("gguf_reader: string array size overflow"))?;
        if self.nbytes_remain < prefix_bytes as u64 {
            return Err(Error::new("gguf_reader: string array exceeds remaining bytes"));
        }
        let mut dst = Vec::with_capacity(n);
        for _ in 0..n {
            dst.push(self.read_string()?);
        }
        Ok(dst)
    }

    /// Current absolute offset in the file/buffer.
    pub fn tell(&self) -> u64 {
        self.data_offset
    }

    /// Position in the file where the GGUF data starts; alignment is relative
    /// to it, not to the file.
    pub fn start(&self) -> u64 {
        self.start_offset
    }

    pub fn seek(&mut self, absolute_offset: u64) -> bool {
        let end_offset = self.data_offset + self.nbytes_remain;
        if absolute_offset > end_offset {
            return false;
        }
        self.data_offset = absolute_offset;
        self.nbytes_remain = end_offset - absolute_offset;
        true
    }
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

pub fn gguf_init_empty() -> GgufContext {
    GgufContext::default()
}

fn gguf_read_emplace_helper<T, F>(
    gr: &mut GgufReader<F>,
    key: &str,
    is_array: bool,
    n: usize,
) -> Result<GgufKv>
where
    T: GgufValue,
    F: FnMut(u64, &mut [u8]) -> usize,
{
    if is_array {
        let value = gr.read_vec::<T>(n)?;
        Ok(GgufKv::new_array(key, &value))
    } else {
        let value = gr.read::<T>()?;
        Ok(GgufKv::new_scalar(key, value))
    }
}

fn gguf_init_from_reader<F>(
    mut gr: GgufReader<F>,
    params: &mut GgufInitParams,
) -> Result<GgufContext>
where
    F: FnMut(u64, &mut [u8]) -> usize,
{
    let mut ctx = GgufContext::default();

    // file magic
    {
        let mut magic = [0u8; 4];
        gr.read_bytes(&mut magic)
            .map_err(|_| Error::new("gguf_init_from_reader: failed to read magic"))?;

        if magic != *GGUF_MAGIC {
            let printable = |c: u8| {
                if c.is_ascii_graphic() || c == b' ' {
                    c as char
                } else {
                    '?'
                }
            };
            return Err(Error::new(format!(
                "gguf_init_from_reader: invalid magic characters: '{}{}{}{}', expected 'GGUF'",
                printable(magic[0]),
                printable(magic[1]),
                printable(magic[2]),
                printable(magic[3]),
            )));
        }
    }

    // header
    ctx.version = gr.read::<u32>()?;
    if ctx.version == 0 {
        return Err(Error::new(format!(
            "gguf_init_from_reader: bad GGUF version: {}",
            ctx.version
        )));
    }

    /*
     * bit layout is different when reading non-native endian models.
     * assuming that the GGUF version is 3, the non-native endian model
     * would read it as 0x30000000. we can use the AND operation against
     * the last 4 hexadecimal digits to check if the model is the same
     * endianness as the host system.
     */
    if (ctx.version & 0x0000_FFFF) == 0x0000_0000 {
        return Err(Error::new(format!(
            "gguf_init_from_reader: failed to load model: this GGUF file version {} is extremely large, is there a mismatch between the host and model endianness?",
            ctx.version
        )));
    }
    if ctx.version == 1 {
        return Err(Error::new(
            "gguf_init_from_reader: GGUFv1 is no longer supported, please use a more up-to-date version",
        ));
    }
    if ctx.version > GGUF_VERSION {
        return Err(Error::new(format!(
            "gguf_init_from_reader: this GGUF file is version {} but this software only supports up to version {}",
            ctx.version, GGUF_VERSION
        )));
    }

    let n_tensors = gr.read::<i64>()?;
    if n_tensors < 0
        || n_tensors as u64 > (usize::MAX / std::mem::size_of::<GgufTensorInfo>()) as u64
    {
        return Err(Error::new(format!(
            "gguf_init_from_reader: number of tensors is {} but must be in [0, {}]",
            n_tensors,
            usize::MAX / std::mem::size_of::<GgufTensorInfo>()
        )));
    }

    let n_kv = gr.read::<i64>()?;
    if n_kv < 0 || n_kv as u64 > (usize::MAX / std::mem::size_of::<GgufKv>()) as u64 {
        return Err(Error::new(format!(
            "gguf_init_from_reader: number of key value pairs is {} but must be in [0, {}]",
            n_kv,
            usize::MAX / std::mem::size_of::<GgufKv>()
        )));
    }

    // KV pairs
    {
        let mut seen_keys = HashSet::new();

        for i in 0..n_kv {
            let key = gr.read_string()?;
            if key.is_empty() {
                return Err(Error::new(format!(
                    "gguf_init_from_reader: key {} is empty",
                    i
                )));
            }
            if !seen_keys.insert(key.clone()) {
                return Err(Error::new(format!(
                    "gguf_init_from_reader: duplicate key '{}' for KV pair {}",
                    key, i
                )));
            }

            let mut type_i = gr.read::<i32>()?;
            let mut is_array = false;
            let mut n = 1usize;

            if type_i == GgufType::Array as i32 {
                is_array = true;
                type_i = gr.read::<i32>()?;
                let n_u64 = gr.read::<u64>()?;
                if n_u64 > GGUF_MAX_ARRAY_ELEMENTS as u64 || n_u64 > usize::MAX as u64 {
                    return Err(Error::new(format!(
                        "gguf_init_from_reader: array length {} for key '{}' is too large",
                        n_u64, key
                    )));
                }
                n = n_u64 as usize;
            }

            let ty = GgufType::try_from(type_i).map_err(|_| {
                Error::new(format!(
                    "gguf_init_from_reader: key '{}' has invalid GGUF type {}",
                    key, type_i
                ))
            })?;

            let kv = match ty {
                GgufType::Uint8 => {
                    gguf_read_emplace_helper::<u8, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Int8 => {
                    gguf_read_emplace_helper::<i8, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Uint16 => {
                    gguf_read_emplace_helper::<u16, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Int16 => {
                    gguf_read_emplace_helper::<i16, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Uint32 => {
                    gguf_read_emplace_helper::<u32, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Int32 => {
                    gguf_read_emplace_helper::<i32, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Float32 => {
                    gguf_read_emplace_helper::<f32, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Uint64 => {
                    gguf_read_emplace_helper::<u64, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Int64 => {
                    gguf_read_emplace_helper::<i64, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Float64 => {
                    gguf_read_emplace_helper::<f64, F>(&mut gr, &key, is_array, n)?
                }
                GgufType::Bool => {
                    if is_array {
                        GgufKv::new_array(&key, &gr.read_bool_vec(n)?)
                    } else {
                        GgufKv::new_scalar(&key, gr.read_bool()?)
                    }
                }
                GgufType::String => {
                    if is_array {
                        GgufKv::new_string_array(&key, &gr.read_string_vec(n)?)
                    } else {
                        GgufKv::new_string(&key, &gr.read_string()?)
                    }
                }
                GgufType::Array | GgufType::Count => {
                    return Err(Error::new(format!(
                        "gguf_init_from_reader: key '{}' has invalid GGUF type {}",
                        key, type_i
                    )));
                }
            };
            ctx.kv.push(kv);
        }

        debug_assert_eq!(ctx.kv.len() as i64, n_kv);

        let alignment_idx = gguf_find_key(&ctx, GGUF_KEY_GENERAL_ALIGNMENT);
        if alignment_idx != -1 && gguf_get_kv_type(&ctx, alignment_idx) != GgufType::Uint32 {
            return Err(Error::new(format!(
                "gguf_init_from_reader: key '{}' must be of type {} but is {}",
                GGUF_KEY_GENERAL_ALIGNMENT,
                gguf_type_name(GgufType::Uint32).unwrap_or("unknown"),
                gguf_type_name(gguf_get_kv_type(&ctx, alignment_idx)).unwrap_or("unknown")
            )));
        }
        ctx.alignment = if alignment_idx == -1 {
            GGUF_DEFAULT_ALIGNMENT
        } else {
            gguf_get_val_u32(&ctx, alignment_idx) as usize
        };

        if ctx.alignment == 0 || (ctx.alignment & (ctx.alignment - 1)) != 0 {
            return Err(Error::new(format!(
                "gguf_init_from_reader: alignment {} is not a power of 2",
                ctx.alignment
            )));
        }
    }

    // read the tensor info
    let mut seen_tensor_names = HashSet::new();

    for i in 0..n_tensors {
        let mut t = GgmlTensor::default();

        // tensor name
        let name = gr.read_string()?;
        if name.len() >= GGML_MAX_NAME {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor name {} is too long: {} >= {}",
                i,
                name.len(),
                GGML_MAX_NAME
            )));
        }
        t.name = name.clone();

        // make sure there are no duplicate tensor names
        if !seen_tensor_names.insert(name) {
            return Err(Error::new(format!(
                "gguf_init_from_reader: duplicate tensor name '{}' for tensor {}",
                t.name, i
            )));
        }

        // tensor shape
        let n_dims = gr.read::<u32>()?;
        if n_dims > GGML_MAX_DIMS as u32 {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor '{}' has invalid number of dimensions: {} > {}",
                t.name, n_dims, GGML_MAX_DIMS
            )));
        }
        for j in 0..GGML_MAX_DIMS {
            t.ne[j] = 1;
            if j < n_dims as usize {
                t.ne[j] = gr.read::<i64>()?;
            }

            // check that all ne are non-negative
            if t.ne[j] < 0 {
                return Err(Error::new(format!(
                    "gguf_init_from_reader: tensor '{}' dimension {} has invalid number of elements: {} < 0",
                    t.name, j, t.ne[j]
                )));
            }
        }

        // check that the total number of elements is representable
        // (a zero-element tensor is trivially representable)
        let mut ok = true;
        let mut current_nelements = t.ne[0];
        for j in 1..GGML_MAX_DIMS {
            if t.ne[j] > 1 {
                // 1 or 0 won't cause an overflow
                if i64::MAX / t.ne[j] < current_nelements {
                    ok = false;
                    break;
                }
            }
            current_nelements *= t.ne[j];
        }
        if !ok {
            return Err(Error::new(format!(
                "gguf_init_from_reader: incorrect shape for a tensor '{}' may cause overflow errors: ({}, {}, {}, {}) is >= INT64_MAX ({})",
                t.name, t.ne[0], t.ne[1], t.ne[2], t.ne[3], i64::MAX
            )));
        }

        // tensor type
        let type_i = gr.read::<i32>()?;

        // check that tensor type is within defined range
        let ty = GgmlType::try_from(type_i).map_err(|_| {
            Error::new(format!(
                "gguf_init_from_reader: tensor '{}' has invalid ggml type {}. should be in [0, {})",
                t.name, type_i, GGML_TYPE_COUNT
            ))
        })?;
        let type_size = ggml_type_size(ty);
        let blck_size = ggml_blck_size(ty);

        // check that row size is divisible by block size
        if blck_size == 0 || t.ne[0] % blck_size != 0 {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor '{}' of type {} ({}) has {} elements per row, not a multiple of block size ({})",
                t.name, type_i, ggml_type_name(ty), t.ne[0], blck_size
            )));
        }

        // check that the size of the tensor in bytes is representable
        let nelements = t.ne.iter().try_fold(1i128, |acc, &ne| acc.checked_mul(ne as i128));
        let blocks = match nelements {
            Some(ne) => ne / blck_size as i128,
            None => i128::MAX,
        };
        if blocks > usize::MAX as i128 / type_size.max(1) as i128 {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor '{}' with shape ({}, {}, {}, {}) has a size in bytes > {}",
                t.name, t.ne[0], t.ne[1], t.ne[2], t.ne[3], usize::MAX
            )));
        }

        // calculate byte offsets given the tensor shape and type
        t.type_ = ty;
        t.nb[0] = type_size;
        t.nb[1] = t.nb[0] * (t.ne[0] as usize / blck_size as usize);
        for j in 2..GGML_MAX_DIMS {
            t.nb[j] = t.nb[j - 1] * t.ne[j - 1] as usize;
        }

        // tensor data offset within buffer
        let offset = gr.read::<u64>()?;
        ctx.info.push(GgufTensorInfo { t, offset });
    }

    debug_assert_eq!(ctx.info.len() as i64, n_tensors);

    // we require the data section to be aligned, so take into account any padding
    if n_tensors > 0 {
        let rel = gr.tell() - gr.start();
        let aligned = gr.start() + gguf_pad(rel as usize, ctx.alignment) as u64;
        if !gr.seek(aligned) {
            return Err(Error::new(
                "gguf_init_from_reader: failed to seek to beginning of data section",
            ));
        }
    }

    // store the current file offset - this is where the data section starts
    ctx.offset = gr.tell() as usize;

    // compute the total size of the data section, taking into account the alignment
    for ti in &ctx.info {
        if ti.offset != ctx.size as u64 {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor '{}' has offset {}, expected {}",
                ti.t.name, ti.offset, ctx.size
            )));
        }
        let nbytes = ggml_nbytes(&ti.t);
        // GGML_PAD adds (alignment - 1) to nbytes. When nbytes is near
        // SIZE_MAX this wraps to 0 and makes the check below dead.
        if nbytes > usize::MAX - (ctx.alignment - 1) {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor '{}' size {} overflows after padding (alignment {})",
                ti.t.name, nbytes, ctx.alignment
            )));
        }
        let padded_size = gguf_pad(nbytes, ctx.alignment);
        if usize::MAX - ctx.size < padded_size {
            return Err(Error::new(format!(
                "gguf_init_from_reader: tensor '{}' size overflow, cannot accumulate size {} + {}",
                ti.t.name, ctx.size, padded_size
            )));
        }
        ctx.size += padded_size;
    }

    // load the tensor data only if requested
    if params.ctx.is_some() {
        // if the provided gguf_context is no_alloc, then we create "empty"
        // tensors and do not read the binary blob; otherwise we load the
        // binary blob and point the tensors at the appropriate locations.

        let mut blob = Vec::new();
        if !params.no_alloc {
            blob.resize(ctx.size, 0u8);
            gr.read_bytes(&mut blob).map_err(|_| {
                Error::new("gguf_init_from_reader: failed to read tensor data binary blob")
            })?;
            ctx.data = blob;
        }

        let mut tensors = Vec::with_capacity(ctx.info.len());
        for info in &ctx.info {
            let mut cur = info.t.clone();
            if !params.no_alloc {
                let start = info.offset as usize;
                let end = start + ggml_nbytes(&info.t);
                cur.data = ctx.data[start..end].to_vec();
            }
            tensors.push(cur);
        }

        params.ctx = Some(GgmlContext {
            tensors,
            no_alloc: params.no_alloc,
        });
    }

    Ok(ctx)
}

/// max_chunk_read == 0 means no limit.
pub fn gguf_init_from_callback<F>(
    callback: F,
    max_chunk_read: usize,
    max_expected_size: u64,
    params: &mut GgufInitParams,
) -> Result<GgufContext>
where
    F: FnMut(u64, &mut [u8]) -> usize,
{
    let gr = GgufReader::new(
        callback,
        if max_chunk_read == 0 {
            usize::MAX
        } else {
            max_chunk_read
        },
        0,
        max_expected_size,
    );
    gguf_init_from_reader(gr, params)
}

struct GgufFileReaderState {
    file: File,
    offset: u64,
}

fn gguf_file_reader_callback(
    state: &mut GgufFileReaderState,
    output: &mut [u8],
    offset: u64,
) -> usize {
    assert!(!output.is_empty());

    if state.offset != offset {
        if state.file.seek(SeekFrom::Start(offset)).is_err() {
            return 0;
        }
        state.offset = offset;
    }

    match state.file.read(output) {
        Ok(nread) => {
            state.offset += nread as u64;
            nread
        }
        Err(_) => 0,
    }
}

/// Remaining bytes from the current file position to EOF.
fn gguf_file_remain(file: &mut File) -> Result<u64> {
    let io_err = |e: std::io::Error| Error::new(e.to_string());
    let cur = file.stream_position().map_err(io_err)?;
    let end = file.seek(SeekFrom::End(0)).map_err(io_err)?;
    file.seek(SeekFrom::Start(cur)).map_err(io_err)?;
    Ok(end.saturating_sub(cur))
}

pub fn gguf_init_from_file_ptr(mut file: File, params: &mut GgufInitParams) -> Result<GgufContext> {
    let cur = file
        .stream_position()
        .map_err(|e| Error::new(e.to_string()))?;
    let remain = gguf_file_remain(&mut file)?;

    let mut state = GgufFileReaderState { file, offset: cur };
    let gr = GgufReader::new(
        |offset, output| gguf_file_reader_callback(&mut state, output, offset),
        usize::MAX,
        cur,
        remain,
    );
    gguf_init_from_reader(gr, params)
}

/// Buffer reader: `data` is the whole backing buffer.
pub fn gguf_init_from_buffer(data: &[u8], params: &mut GgufInitParams) -> Result<GgufContext> {
    if data.is_empty() {
        return Err(Error::new("gguf_init_from_buffer: empty buffer"));
    }

    let gr = GgufReader::new(
        |offset: u64, output: &mut [u8]| -> usize {
            assert!(!output.is_empty());
            if offset > data.len() as u64 || output.len() > data.len() - offset as usize {
                return 0;
            }
            let data_offset = offset as usize;
            let nread = output.len().min(data.len() - data_offset);
            output[..nread].copy_from_slice(&data[data_offset..data_offset + nread]);
            nread
        },
        usize::MAX,
        0,
        data.len() as u64,
    );
    gguf_init_from_reader(gr, params)
}

pub fn gguf_init_from_file(fname: &str, params: &mut GgufInitParams) -> Result<GgufContext> {
    let file = File::open(fname)
        .map_err(|e| Error::new(format!("gguf_init_from_file: failed to open '{}' ({})", fname, e)))?;
    gguf_init_from_file_ptr(file, params)
}

/// Present for API parity. Rust drops the context automatically.
pub fn gguf_free(_ctx: GgufContext) {}

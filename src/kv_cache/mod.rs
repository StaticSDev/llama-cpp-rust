//! Rust port of llama.cpp's main KV cache.
//!
//! Covers `llama_kv_cells`, `llama_kv_cache` and `llama_kv_cache_context` from
//! `llama-kv-cells.h`, `llama-kv-cache.h` and `llama-kv-cache.cpp`.  The cell
//! bookkeeping, sequence tracking, slot allocation, shifting, position
//! handling, state save/restore, memory accounting and the public API are
//! ported statement by statement.  ggml tensors are represented by
//! `Tensor` descriptors over an in-memory byte store so that views, row
//! scatters and buffer clears can be carried out without FFI or `unsafe`.

use crate::core::{Error, Result, SeqId, Token, LLAMA_TOKEN_NULL};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

/// `llama_pos`.
pub type Pos = i32;

/// `llama_seq_id` (negative values are sentinels, see `seq_id == -1` calls).
pub type LSeqId = i32;

/// Cell and stream indices are `uint32_t` in the C++ source; reuse `SeqId`.
pub type CellIdx = SeqId;

const LLAMA_MAX_SEQ: usize = 512;
const N_PAD: u32 = 256;
const N_PAD_KV_ROWS: u32 = 256;

// ---------------------------------------------------------------------------
// Small type stand-ins for llama.h / ggml.h enums
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwaType {
    None,
    Standard,
    Chunked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeType {
    None,
    NeoX,
    Mrope,
    Imrope,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonCausalType {
    None,
    SwaOnly,
    DenseOnly,
    SwaFull,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Other,
    Step35,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryStatus {
    Success,
    NoUpdate,
    FailedPrepare,
}

pub fn memory_status_is_fail(s: MemoryStatus) -> bool {
    s == MemoryStatus::FailedPrepare
}

pub const LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    F8E4M3,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlType {
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
            Self::F16 | Self::BF16 => 2,
            Self::F8E4M3 => 1,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    pub fn blck_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F8E4M3 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    pub fn is_quantized(self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::F8E4M3 | Self::I32 | Self::I64)
    }
}

/// `ggml_row_size(type, n)`.
pub fn ggml_row_size(ty: GgmlType, n: usize) -> usize {
    let bs = ty.blck_size();
    ((n + bs - 1) / bs) * ty.type_size()
}

fn ggml_is_power_of_2(n: i64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn ggml_pad(x: usize, n: usize) -> usize {
    ((x + n - 1) / n) * n
}

fn ctz64(x: u64) -> u32 {
    x.trailing_zeros()
}

/// Orthonormal Walsh-Hadamard rotation matrix; `res^2 == I`.
fn ggml_gen_hadamard(n: usize) -> Vec<f32> {
    debug_assert!(ggml_is_power_of_2(n as i64));
    let mut data = vec![0.0f32; n * n];
    data[0] = 1.0 / (n as f32).sqrt();
    let mut s = 1;
    while s < n {
        for i in 0..s {
            for j in 0..s {
                let val = data[i * n + j];
                data[(i + s) * n + j] = val;
                data[i * n + (j + s)] = val;
                data[(i + s) * n + (j + s)] = -val;
            }
        }
        s *= 2;
    }
    data
}

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7fff_ff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (14 - exp);
        return sign | (m as u16);
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | (((exp as u16) << 10) | ((mant >> 13) as u16))
}

fn llama_relative_position_bucket(x: Pos, y: Pos, mut n_buckets: u64, bidirectional: bool) -> i32 {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const MAX_DISTANCE: f32 = 128.0;

    if bidirectional {
        n_buckets >>= 1;
    }
    let max_exact = (n_buckets >> 1).max(1) as f32;

    let mut relative_position = x - y;
    let mut relative_bucket = 0i32;
    if bidirectional {
        if relative_position > 0 {
            relative_bucket += n_buckets as i32;
        }
        relative_position = relative_position.abs();
    } else {
        relative_position = -relative_position.min(0);
    }

    let large = max_exact
        + ((relative_position as f32 / max_exact).ln()
            * (n_buckets as f32 - max_exact)
            / (MAX_DISTANCE / max_exact).ln())
            .floor();
    let large = (large as i32).min(n_buckets as i32 - 1);
    relative_bucket += if (relative_position as f32) < max_exact { relative_position } else { large };
    relative_bucket
}

impl HParams {
    pub fn is_masked_swa(n_swa: u32, swa_type: SwaType, pos0: Pos, pos1: Pos) -> bool {
        if n_swa == 0 {
            return false;
        }
        match swa_type {
            SwaType::None => false,
            SwaType::Standard => pos0 < pos1 - n_swa as Pos,
            SwaType::Chunked => pos0 / n_swa as Pos != pos1 / n_swa as Pos,
        }
    }
}

// ---------------------------------------------------------------------------
// llama_kv_cell_ext / llama_kv_rows_slice
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCellExt {
    /// 2D spatial positions, typically used for M-RoPE.
    pub x: Pos,
    pub y: Pos,
    /// `LLAMA_TOKEN_NULL` when the cell is produced by embedding input.
    pub tok: Token,
}

impl Default for KvCellExt {
    fn default() -> Self {
        Self { x: 0, y: 0, tok: LLAMA_TOKEN_NULL }
    }
}

impl KvCellExt {
    pub fn is_2d_gt(&self, ox: Pos, oy: Pos) -> bool {
        (self.y > oy) || (self.y == oy && self.x > ox)
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Debug)]
pub struct KvRowsSlice {
    /// The sequences whose cells form the list of the slice.
    pub seq_ids: Vec<LSeqId>,
    /// Min position of the cells that are not masked by the SWA (-1 if all kept).
    pub p_min: Pos,
}

impl Default for KvRowsSlice {
    fn default() -> Self {
        Self { seq_ids: Vec::new(), p_min: -1 }
    }
}
// ---------------------------------------------------------------------------
// Tensor descriptors, IO, model / hparams descriptors
// ---------------------------------------------------------------------------

/// Minimal ggml tensor stand-in: shape, strides and a byte offset into a
/// shared storage buffer.  Views and row scatters operate on the real bytes.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub ty: GgmlType,
    pub ne: [usize; 4],
    pub nb: [usize; 4],
    pub offset: usize,
    pub storage: Rc<RefCell<Vec<u8>>>,
    pub name: String,
    pub is_input: bool,
}

impl Tensor {
    pub fn new_nd(ty: GgmlType, ne: [usize; 4], storage: Rc<RefCell<Vec<u8>>>) -> Self {
        let ts = ty.type_size();
        let mut nb = [ts; 4];
        for i in 1..4 {
            nb[i] = nb[i - 1] * ne[i - 1];
        }
        Self { ty, ne, nb, offset: 0, storage, name: String::new(), is_input: false }
    }

    pub fn new_1d(ty: GgmlType, ne0: usize) -> Self {
        let n = ty.type_size() * ne0;
        Self::new_nd(ty, [ne0, 1, 1, 1], Rc::new(RefCell::new(vec![0u8; n])))
    }

    pub fn new_2d(ty: GgmlType, ne0: usize, ne1: usize) -> Self {
        let n = ty.type_size() * ne0 * ne1;
        Self::new_nd(ty, [ne0, ne1, 1, 1], Rc::new(RefCell::new(vec![0u8; n])))
    }

    pub fn new_3d(ty: GgmlType, ne0: usize, ne1: usize, ne2: usize) -> Self {
        let n = ty.type_size() * ne0 * ne1 * ne2;
        Self::new_nd(ty, [ne0, ne1, ne2, 1], Rc::new(RefCell::new(vec![0u8; n])))
    }

    pub fn new_4d(ty: GgmlType, ne: [usize; 4]) -> Self {
        let n = ty.type_size() * ne.iter().product::<usize>();
        Self::new_nd(ty, ne, Rc::new(RefCell::new(vec![0u8; n])))
    }

    pub fn nelements(&self) -> usize {
        self.ne.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        ggml_row_size(self.ty, self.ne[0]) * self.ne[1] * self.ne[2] * self.ne[3]
    }

    pub fn view_2d(&self, ne0: usize, ne1: usize, nb1: usize, off: usize) -> Self {
        let ts = self.ty.type_size();
        Self {
            ty: self.ty,
            ne: [ne0, ne1, 1, 1],
            nb: [ts, nb1, nb1 * ne1, nb1 * ne1 * ne0],
            offset: self.offset + off,
            storage: self.storage.clone(),
            name: self.name.clone(),
            is_input: self.is_input,
        }
    }

    pub fn view_3d(&self, ne0: usize, ne1: usize, ne2: usize, nb1: usize, nb2: usize, off: usize) -> Self {
        let ts = self.ty.type_size();
        Self {
            ty: self.ty,
            ne: [ne0, ne1, ne2, 1],
            nb: [ts, nb1, nb2, nb2 * ne2],
            offset: self.offset + off,
            storage: self.storage.clone(),
            name: self.name.clone(),
            is_input: self.is_input,
        }
    }

    pub fn view_4d(&self, ne: [usize; 4], nb: [usize; 4], off: usize) -> Self {
        let mut nb = nb;
        nb[0] = self.ty.type_size();
        Self {
            ty: self.ty,
            ne,
            nb,
            offset: self.offset + off,
            storage: self.storage.clone(),
            name: self.name.clone(),
            is_input: self.is_input,
        }
    }

    pub fn reshape_2d(&self, ne0: usize, ne1: usize) -> Self {
        debug_assert_eq!(ne0 * ne1, self.nelements());
        let mut res = self.clone();
        res.ne = [ne0, ne1, 1, 1];
        let ts = self.ty.type_size();
        res.nb = [ts, ts * ne0, ts * ne0 * ne1, ts * ne0 * ne1];
        res
    }

    /// `ggml_set_rows`: scatter rows of `src` into rows `dst_idxs` of `self`.
    pub fn set_rows(&mut self, src: &Tensor, idxs: &[i64]) {
        let row = self.nb[1];
        let src_row = src.nb[1];
        let n = idxs.len().min(src.ne[1].max(src.ne[2]));
        for i in 0..n {
            let dst_idx = idxs[i];
            if dst_idx < 0 {
                continue;
            }
            let s_off = src.offset + i * src.nb[1];
            let d_off = self.offset + (dst_idx as usize) * row;
            let s = src.storage.borrow();
            let mut d = self.storage.borrow_mut();
            let len = row.min(src_row);
            if s_off + len <= s.len() && d_off + len <= d.len() {
                d[d_off..d_off + len].copy_from_slice(&s[s_off..s_off + len]);
            }
        }
    }

    pub fn zero(&mut self, offset: usize, size: usize) {
        let mut s = self.storage.borrow_mut();
        let end = (offset + size).min(s.len());
        if offset < end {
            s[offset..end].fill(0);
        }
    }

    pub fn bytes(&self) -> Vec<u8> {
        let s = self.storage.borrow();
        let end = (self.offset + self.nbytes()).min(s.len());
        if self.offset < end {
            s[self.offset..end].to_vec()
        } else {
            Vec::new()
        }
    }

    pub fn set_bytes(&mut self, data: &[u8]) {
        let mut s = self.storage.borrow_mut();
        let end = (self.offset + data.len()).min(s.len());
        if self.offset < end {
            s[self.offset..end].copy_from_slice(&data[..end - self.offset]);
        }
    }

    pub fn write_i64_at(&mut self, idx: usize, value: i64) {
        let off = self.offset + idx * 8;
        let mut s = self.storage.borrow_mut();
        if off + 8 <= s.len() {
            s[off..off + 8].copy_from_slice(&value.to_le_bytes());
        }
    }

    pub fn write_i32_at(&mut self, idx: usize, value: i32) {
        let off = self.offset + idx * 4;
        let mut s = self.storage.borrow_mut();
        if off + 4 <= s.len() {
            s[off..off + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
}

/// A backend buffer descriptor used for the `memory_breakdown()` accounting.
#[derive(Clone, Debug)]
pub struct MemoryBuf {
    pub name: String,
    pub buft: String,
    pub size: usize,
}

pub trait IoWrite {
    fn write_bytes(&mut self, bytes: &[u8]);
    fn write_tensor(&mut self, tensor: &Tensor, offset: usize, size: usize);
}

pub trait IoRead {
    fn read_bytes(&mut self, out: &mut [u8]) -> bool;
    fn read_tensor(&mut self, tensor: &Tensor, offset: usize, size: usize) -> bool;
}

#[derive(Default)]
pub struct MemWriter {
    pub data: Vec<u8>,
}

impl IoWrite for MemWriter {
    fn write_bytes(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }
    fn write_tensor(&mut self, tensor: &Tensor, offset: usize, size: usize) {
        let s = tensor.storage.borrow();
        let off = tensor.offset + offset;
        let end = (off + size).min(s.len());
        if off < end {
            self.data.extend_from_slice(&s[off..end]);
        }
    }
}

pub struct MemReader {
    pub data: Vec<u8>,
    pub pos: usize,
}

impl MemReader {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data, pos: 0 }
    }
}

impl IoRead for MemReader {
    fn read_bytes(&mut self, out: &mut [u8]) -> bool {
        if self.pos + out.len() > self.data.len() {
            return false;
        }
        out.copy_from_slice(&self.data[self.pos..self.pos + out.len()]);
        self.pos += out.len();
        true
    }
    fn read_tensor(&mut self, tensor: &Tensor, offset: usize, size: usize) -> bool {
        if self.pos + size > self.data.len() {
            return false;
        }
        let mut s = tensor.storage.borrow_mut();
        let off = tensor.offset + offset;
        let end = (off + size).min(s.len());
        if off < end {
            let n = end - off;
            s[off..end].copy_from_slice(&self.data[self.pos..self.pos + n]);
        }
        self.pos += size;
        true
    }
}

fn write_u32<W: IoWrite + ?Sized>(io: &mut W, v: u32) {
    io.write_bytes(&v.to_le_bytes());
}
fn write_i32<W: IoWrite + ?Sized>(io: &mut W, v: i32) {
    io.write_bytes(&v.to_le_bytes());
}
fn write_u64<W: IoWrite + ?Sized>(io: &mut W, v: u64) {
    io.write_bytes(&v.to_le_bytes());
}
fn read_u32<R: IoRead + ?Sized>(io: &mut R) -> u32 {
    let mut b = [0u8; 4];
    if io.read_bytes(&mut b) {
        u32::from_le_bytes(b)
    } else {
        0
    }
}
fn read_i32<R: IoRead + ?Sized>(io: &mut R) -> i32 {
    let mut b = [0u8; 4];
    if io.read_bytes(&mut b) {
        i32::from_le_bytes(b)
    } else {
        0
    }
}
fn read_u64<R: IoRead + ?Sized>(io: &mut R) -> u64 {
    let mut b = [0u8; 8];
    if io.read_bytes(&mut b) {
        u64::from_le_bytes(b)
    } else {
        0
    }
}

#[derive(Clone, Debug)]
pub struct ModelLayerInfo {
    pub k_cache_s: GgmlType,
    pub v_cache_s: GgmlType,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub arch: Arch,
    pub attn_rot: bool,
    pub lid_k_hdmd: bool,
    pub dev_buft: String,
    pub layers: Vec<ModelLayerInfo>,
}

impl Model {
    fn dev_layer(&self) -> &str {
        &self.dev_buft
    }
}

/// Small stand-in for `llama_hparams`.
#[derive(Clone, Debug)]
pub struct HParams {
    pub n_layer_all: u32,
    pub n_embd_head_k: u32,
    pub n_embd_head_v: u32,
    pub n_embd_head_k_full: u32,
    pub indexer_head_size: u32,
    pub n_head_kv: u32,
    pub n_embd_k_gqa: u32,
    pub n_embd_v_gqa: u32,
    pub n_embd_v_gqa_max: u32,
    pub v_gqa_variable: bool,
    pub n_pos_per_embd: u32,
    pub is_mla: bool,
    pub ple_n_heads: u32,
    pub ple_embd_token: Token,
    pub use_alibi: bool,
    pub non_causal_type: NonCausalType,
    pub n_rel_attn_bkts: u32,
    pub rope_type: RopeType,
}

impl Default for HParams {
    fn default() -> Self {
        Self {
            n_layer_all: 1,
            n_embd_head_k: 128,
            n_embd_head_v: 128,
            n_embd_head_k_full: 128,
            indexer_head_size: 0,
            n_head_kv: 1,
            n_embd_k_gqa: 128,
            n_embd_v_gqa: 128,
            n_embd_v_gqa_max: 128,
            v_gqa_variable: false,
            n_pos_per_embd: 1,
            is_mla: false,
            ple_n_heads: 0,
            ple_embd_token: 0,
            use_alibi: false,
            non_causal_type: NonCausalType::None,
            n_rel_attn_bkts: 8,
            rope_type: RopeType::NeoX,
        }
    }
}

impl HParams {
    pub fn n_embd_head_k(&self) -> u32 {
        self.n_embd_head_k
    }
    pub fn n_embd_head_v(&self) -> u32 {
        self.n_embd_head_v
    }
    pub fn n_embd_k_gqa(&self, _il: u32) -> u32 {
        self.n_embd_k_gqa
    }
    pub fn n_embd_v_gqa(&self, _il: u32) -> u32 {
        self.n_embd_v_gqa
    }
    pub fn n_head_kv(&self, _il: u32) -> u32 {
        self.n_head_kv
    }
    pub fn n_pos_per_embd(&self) -> u32 {
        self.n_pos_per_embd
    }
    pub fn is_n_embd_v_gqa_variable(&self) -> bool {
        self.v_gqa_variable
    }
    pub fn n_embd_v_gqa_max(&self) -> u32 {
        self.n_embd_v_gqa_max
    }
    pub fn has_kv(&self, _il: u32) -> bool {
        true
    }
    pub fn has_rope(&self, _il: u32) -> bool {
        self.rope_type != RopeType::None
    }
}

/// `llama_ubatch`, reduced to the fields the cache reads.
#[derive(Clone, Debug)]
pub struct UBatch {
    pub n_tokens: usize,
    pub n_seqs: usize,
    pub n_seqs_unq: usize,
    pub seq_id_unq: Vec<LSeqId>,
    pub n_seq_id: Vec<usize>,
    pub seq_id: Vec<Vec<LSeqId>>,
    pub pos: Vec<Pos>,
    pub token: Option<Vec<Token>>,
    /// `ubatch.type[i]`: true when token i comes from embedding input.
    pub type_embd: Vec<bool>,
    pub n_pos_per_embd: u32,
    pub equal_seqs: bool,
    pub mixed: bool,
}

impl UBatch {
    pub fn is_pos_2d(&self) -> bool {
        self.n_pos_per_embd > 1
    }
    pub fn is_mixed(&self) -> bool {
        self.mixed
    }
    pub fn equal_seqs(&self) -> bool {
        self.equal_seqs
    }

    /// `llama_batch_allocr::ubatch_reserve` for the state-restore path.
    pub fn ubatch_reserve(n_tokens: usize, n_pos_per_embd: u32, tok: bool) -> Self {
        let pos = vec![0; n_tokens * n_pos_per_embd as usize];
        let token = if tok { Some(vec![LLAMA_TOKEN_NULL; n_tokens]) } else { None };
        Self {
            n_tokens,
            n_seqs: 1,
            n_seqs_unq: 1,
            seq_id_unq: vec![0; 1],
            n_seq_id: vec![0; n_tokens],
            seq_id: vec![Vec::new(); n_tokens],
            pos,
            token,
            type_embd: vec![false; n_tokens],
            n_pos_per_embd,
            equal_seqs: false,
            mixed: false,
        }
    }
}
// ---------------------------------------------------------------------------
// llama_kv_cache
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct KvLayer {
    il: u32,
    k: Tensor,
    v: Option<Tensor>,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
    k_stream: Vec<Tensor>,
    v_stream: Vec<Option<Tensor>>,
    k_rot: Option<Tensor>,
    v_rot: Option<Tensor>,
}

#[derive(Clone, Debug)]
struct CellRanges {
    strm: SeqId,
    /// Ranges, from inclusive, to exclusive.
    data: Vec<(CellIdx, CellIdx)>,
}

#[derive(Clone)]
pub struct KvCacheParams {
    pub hparams: HParams,
    pub model: Model,
    pub type_k: GgmlType,
    pub type_v: GgmlType,
    pub v_trans: bool,
    pub offload: bool,
    pub unified: bool,
    pub kv_size: CellIdx,
    pub n_seq_max: SeqId,
    pub n_swa: u32,
    pub swa_type: SwaType,
    pub no_alloc: bool,
    pub name_tag: String,
    pub is_indexer: bool,
    /// `layer_filter_cb`, `layer_reuse_cb`, `layer_share_cb`.
    pub filter: Option<fn(u32) -> bool>,
    pub reuse: Option<fn(u32) -> i32>,
    pub share: Option<fn(u32) -> i32>,
    /// Cache whose cells are shared with this one.
    pub other: Option<Rc<KvCache>>,
}

impl Default for KvCacheParams {
    fn default() -> Self {
        Self {
            hparams: HParams::default(),
            model: Model {
                arch: Arch::Other,
                attn_rot: true,
                lid_k_hdmd: false,
                dev_buft: "CPU".to_string(),
                layers: Vec::new(),
            },
            type_k: GgmlType::F16,
            type_v: GgmlType::F16,
            v_trans: true,
            offload: false,
            unified: false,
            kv_size: 512,
            n_seq_max: 1,
            n_swa: 0,
            swa_type: SwaType::None,
            no_alloc: false,
            name_tag: String::new(),
            is_indexer: false,
            filter: None,
            reuse: None,
            share: None,
            other: None,
        }
    }
}

pub struct KvCache {
    params: KvCacheParams,
    v_trans: bool,
    n_seq_max: SeqId,
    n_stream: SeqId,
    n_swa: u32,
    #[allow(dead_code)]
    is_indexer: bool,

    n_rot_k: u32,
    n_rot_v: u32,
    #[allow(dead_code)]
    n_embd_head_k_all: i32,
    #[allow(dead_code)]
    n_embd_head_v_all: i32,

    attn_rot_hadamard: BTreeMap<i64, Vec<f32>>,
    attn_rot_bufs: Vec<MemoryBuf>,
    #[allow(dead_code)]
    debug: i32,
    swa_type: SwaType,

    ctxs_bufs: Vec<MemoryBuf>,
    storages: Vec<Rc<RefCell<Vec<u8>>>>,

    /// Where to start the free-slot search in each stream's ring buffer.
    v_heads: Vec<CellIdx>,
    other: Option<Rc<KvCache>>,
    v_cells: Vec<KvCells>,
    seq_to_stream: Vec<SeqId>,
    sc_info: StreamCopyInfo,
    layers: Vec<KvLayer>,
    /// Model layer id -> KV cache layer id.
    map_layer_ids: BTreeMap<i32, usize>,
}

impl KvCache {
    pub fn new(mut params: KvCacheParams) -> Result<Self> {
        let v_trans = params.v_trans;
        let n_swa = params.n_swa;
        let swa_type = params.swa_type;
        let is_indexer = params.is_indexer;
        let name_tag = params.name_tag.clone();
        let no_alloc = params.no_alloc;
        let offload = params.offload;
        let other = params.other.clone();

        // Shared cells view the source cache's K/V tensors, so the cell count
        // follows the source allocation: a fitted target can be smaller than
        // the draft default and oversized views would overflow the source.
        let mut kv_size = params.kv_size;
        if let Some(ref o) = other {
            let size_other = o.get_size();
            if kv_size != size_other {
                kv_size = size_other;
            }
        }
        params.kv_size = kv_size;

        let n_layer = params.hparams.n_layer_all;
        let n_stream = if params.unified { 1 } else { params.n_seq_max };
        let n_seq_max = params.n_seq_max;

        debug_assert!(n_stream == 1 || n_stream == n_seq_max);

        let v_heads = vec![0 as CellIdx; n_stream as usize];

        let v_cells = match other.as_ref() {
            Some(o) => o.v_cells.clone(),
            None => {
                let mut cells = Vec::with_capacity(n_stream as usize);
                for _ in 0..n_stream {
                    let mut c = KvCells::new();
                    c.resize(kv_size);
                    cells.push(c);
                }
                cells
            }
        };

        // By default, all sequence ids are mapped to the 0th stream.
        let mut seq_to_stream = vec![0 as SeqId; LLAMA_MAX_SEQ];
        if n_stream > 1 {
            seq_to_stream = (0..n_stream).collect();
        }

        let is_mla = params.hparams.is_mla;
        let mut n_embd_head_k_all = 0i32;
        let mut n_embd_head_v_all = 0i32;

        let mut map_layer_ids: BTreeMap<i32, usize> = BTreeMap::new();
        let mut layers: Vec<KvLayer> = Vec::new();
        let mut ctx_map: BTreeMap<String, Vec<Tensor>> = BTreeMap::new();

        for il in 0..n_layer {
            if !params.hparams.has_kv(il) {
                continue;
            }
            if let Some(filter) = params.filter {
                if !filter(il) {
                    continue;
                }
            }

            if let (Some(share), Some(ref o)) = (params.share, other.as_ref()) {
                let il_share = share(il);
                if il_share >= 0 {
                    let ikv = *o.map_layer_ids.get(&il_share).ok_or_else(|| {
                        Error::new("failed to restore kv cache: shared layer not found")
                    })?;
                    let mut layer = o.layers[ikv].clone();
                    layer.il = il;
                    map_layer_ids.insert(il as i32, layers.len());
                    layers.push(layer);
                    continue;
                }
            }

            let hk = params.hparams.n_embd_head_k() as i32;
            let hkv = params.hparams.n_embd_head_v() as i32;
            if n_embd_head_k_all == 0 {
                n_embd_head_k_all = hk;
            } else if n_embd_head_k_all > 0 && n_embd_head_k_all != hk {
                n_embd_head_k_all = -1;
            }
            if !is_mla {
                if n_embd_head_v_all == 0 {
                    n_embd_head_v_all = hkv;
                } else if n_embd_head_v_all > 0 && n_embd_head_v_all != hkv {
                    n_embd_head_v_all = -1;
                }
            }

            // [TAG_V_CACHE_VARIABLE]
            let n_embd_k_gqa = params.hparams.n_embd_k_gqa(il);
            let n_embd_v_gqa = if !v_trans {
                params.hparams.n_embd_v_gqa(il)
            } else {
                params.hparams.n_embd_v_gqa_max()
            };

            let buft = if offload {
                params.model.dev_layer().to_string()
            } else {
                "CPU".to_string()
            };
            ctx_map.entry(buft.clone()).or_default();

            let has_k = true;
            let has_v = !is_mla;

            let mut k = if has_k {
                Tensor::new_3d(params.type_k, n_embd_k_gqa as usize, kv_size as usize, n_stream as usize)
            } else {
                Tensor::new_3d(params.type_k, 0, 0, 0)
            };
            k.name = format!("cache_{}_k_l{}", name_tag, il);

            let mut v = if has_v {
                Some(Tensor::new_3d(params.type_v, n_embd_v_gqa as usize, kv_size as usize, n_stream as usize))
            } else {
                None
            };
            if let Some(ref mut v) = v {
                v.name = format!("cache_{}_v_l{}", name_tag, il);
            }

            let mut k_stream = Vec::with_capacity(n_stream as usize);
            let mut v_stream = Vec::with_capacity(n_stream as usize);
            for s in 0..n_stream {
                k_stream.push(k.view_2d(
                    n_embd_k_gqa as usize,
                    kv_size as usize,
                    k.nb[1],
                    (s as usize) * k.nb[2],
                ));
                v_stream.push(v.as_ref().map(|v| {
                    v.view_2d(n_embd_v_gqa as usize, kv_size as usize, v.nb[1], (s as usize) * v.nb[2])
                }));
            }

            let (k_scale, v_scale) = match params.model.layers.get(il as usize) {
                Some(ml) => (
                    Some(Tensor::new_2d(ml.k_cache_s, 1, 1)),
                    Some(Tensor::new_2d(ml.v_cache_s, 1, 1)),
                ),
                None => (None, None),
            };

            map_layer_ids.insert(il as i32, layers.len());
            layers.push(KvLayer {
                il,
                k: k.clone(),
                v: v.clone(),
                k_scale,
                v_scale,
                k_stream,
                v_stream,
                k_rot: None,
                v_rot: None,
            });

            ctx_map.get_mut(&buft).unwrap().push(k);
            if let Some(v) = v {
                ctx_map.get_mut(&buft).unwrap().push(v);
            }
        }

        if let Some(reuse) = params.reuse {
            for il in 0..n_layer {
                let il_reuse = reuse(il);
                if il_reuse < 0 {
                    continue;
                }
                if let Some(filter) = params.filter {
                    if !filter(il) {
                        continue;
                    }
                }
                let idx = *map_layer_ids.get(&il_reuse).ok_or_else(|| {
                    Error::new("failed to restore kv cache: reuse layer not found")
                })?;
                map_layer_ids.insert(il as i32, idx);
            }
        }

        // Allocate the tensors and zero the buffers to avoid NaNs in padding.
        let mut ctxs_bufs = Vec::new();
        let mut storages = Vec::new();
        for (buft, tensors) in ctx_map.iter() {
            let size: usize = tensors.iter().map(|t| t.nbytes()).sum();
            // Keep handles to the real backing stores so clear(data) zeroes
            // the same bytes the layer views address.
            for t in tensors {
                if !storages.iter().any(|s| Rc::ptr_eq(s, &t.storage)) {
                    storages.push(t.storage.clone());
                }
            }
            let _ = no_alloc;
            ctxs_bufs.push(MemoryBuf { name: "kv cache".to_string(), buft: buft.clone(), size });
        }

        let (mut n_rot_k, mut n_rot_v) = (0u32, 0u32);
        if let Some(ref o) = other {
            // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
            n_rot_k = o.n_rot_k;
            n_rot_v = o.n_rot_v;
        } else {
            let env_rot_disable = std::env::var("LLAMA_ATTN_ROT_DISABLE").unwrap_or_default();
            let attn_rot_disable =
                !params.model.attn_rot || env_rot_disable.parse::<i32>().unwrap_or(0) != 0;

            let mut attn_rot_k = !attn_rot_disable
                && n_embd_head_k_all > 0
                && params.type_k.is_quantized()
                && params.hparams.n_embd_head_k() % 64 == 0;

            // Always create Hadamard rotation tensors for lightning indexer keys.
            if params.model.lid_k_hdmd
                && params.hparams.n_embd_head_k_full == params.hparams.indexer_head_size
            {
                attn_rot_k = true;
            }

            let attn_rot_v = !attn_rot_disable
                && n_embd_head_v_all > 0
                && params.type_v.is_quantized()
                && params.hparams.n_embd_head_v() % 64 == 0;

            if attn_rot_k {
                n_rot_k = 64;
                // TODO: investigate if using the smallest rotation matrix is
                // beneficial also for K (similar as for V).
                while n_embd_head_k_all as u32 % (2 * n_rot_k) == 0 {
                    n_rot_k *= 2;
                }
            }
            if attn_rot_v {
                n_rot_v = 64;
                // Using smaller rotation matrices for V seems beneficial.
            }
        }

        // Pre-compute the Hadamard matrices and keep them in host memory.
        let mut attn_rot_hadamard = BTreeMap::new();
        if n_rot_k != 0 || n_rot_v != 0 {
            let max_n = n_embd_head_k_all.max(n_embd_head_v_all);
            let mut n = 64i64;
            while n <= max_n as i64 {
                attn_rot_hadamard.insert(n, ggml_gen_hadamard(n as usize));
                n *= 2;
            }
        }

        let debug = std::env::var("LLAMA_KV_CACHE_DEBUG")
            .unwrap_or_default()
            .parse::<i32>()
            .unwrap_or(0);

        let mut res = Self {
            params,
            v_trans,
            n_seq_max,
            n_stream,
            n_swa,
            is_indexer,
            n_rot_k,
            n_rot_v,
            n_embd_head_k_all,
            n_embd_head_v_all,
            attn_rot_hadamard,
            attn_rot_bufs: Vec::new(),
            debug,
            swa_type,
            ctxs_bufs,
            storages,
            v_heads,
            other,
            v_cells,
            seq_to_stream,
            sc_info: StreamCopyInfo::default(),
            layers,
            map_layer_ids,
        };
        res.init_rotation(&name_tag);
        Ok(res)
    }

    fn init_rotation(&mut self, name_tag: &str) {
        if !(self.n_rot_k != 0 || self.n_rot_v != 0) || self.layers.is_empty() {
            return;
        }

        let snapshot = self.layers.clone();
        let mut ctx_map: BTreeMap<String, Vec<Tensor>> = BTreeMap::new();
        for (li, layer) in snapshot.iter().enumerate() {
            for is_k in [true, false] {
                let n_rot = if is_k { self.n_rot_k } else { self.n_rot_v };
                if n_rot == 0 {
                    continue;
                }
                let mut rot = Tensor::new_2d(GgmlType::F32, n_rot as usize, n_rot as usize);
                rot.name = format!(
                    "cache_{}_{}_rot_l{}",
                    name_tag,
                    if is_k { "k" } else { "v" },
                    layer.il
                );
                ctx_map.entry("rotation matrices".to_string()).or_default().push(rot.clone());
                if is_k {
                    self.layers[li].k_rot = Some(rot);
                } else if self.layers[li].v.is_some() {
                    self.layers[li].v_rot = Some(rot);
                }
            }
        }

        for (buft, tensors) in ctx_map {
            let size: usize = tensors.iter().map(|t| t.nbytes()).sum();
            self.attn_rot_bufs.push(MemoryBuf {
                name: "rotation matrices".to_string(),
                buft,
                size,
            });
        }

        if !self.params.no_alloc {
            let snapshot = self.layers.clone();
            for layer in snapshot.iter() {
                for t in [layer.k_rot.clone(), layer.v_rot.clone()] {
                    if let Some(mut t) = t {
                        if let Some(data) = self.attn_rot_hadamard.get(&(t.ne[0] as i64)).cloned() {
                            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                            t.set_bytes(&bytes);
                        }
                    }
                }
            }
        }
    }
}


/// Stand-in for `llm_attn_mask`, the raw mask and its optional cell rows.
#[derive(Clone, Debug, Default)]
pub struct AttnMask {
    pub raw: Option<Tensor>,
    pub rows: Option<Tensor>,
    pub block_uniform: bool,
}

/// Stand-in for `ggml_context`; views are carried by the tensors themselves.
#[derive(Clone, Copy, Debug, Default)]
pub struct TensorContext;

impl TensorContext {
    pub fn new() -> Self {
        Self
    }
}

fn bytes_i32(data: &[i32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_f32(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_i32(data: &[u8], n: usize) -> Vec<i32> {
    let mut res = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        if off + 4 <= data.len() {
            let mut b = [0u8; 4];
            b.copy_from_slice(&data[off..off + 4]);
            res.push(i32::from_le_bytes(b));
        } else {
            res.push(0);
        }
    }
    res
}

fn ggml_cont_2d(src: &Tensor, ne0: usize, ne1: usize) -> Tensor {
    let dst = Tensor::new_2d(src.ty, ne0, ne1);
    let row = ggml_row_size(src.ty, ne0);
    {
        let s = src.storage.borrow();
        let mut d = dst.storage.borrow_mut();
        for j in 0..ne1 {
            let s_off = src.offset + j * src.nb[1];
            let d_off = j * dst.nb[1];
            if s_off + row <= s.len() && d_off + row <= d.len() {
                d[d_off..d_off + row].copy_from_slice(&s[s_off..s_off + row]);
            }
        }
    }
    dst
}

fn ggml_pad_2d(src: &Tensor, pad0: usize) -> Tensor {
    let ne0 = src.ne[0] + pad0;
    let dst = Tensor::new_2d(src.ty, ne0, src.ne[1]);
    let row = ggml_row_size(src.ty, src.ne[0]);
    {
        let s = src.storage.borrow();
        let mut d = dst.storage.borrow_mut();
        for j in 0..src.ne[1] {
            let s_off = src.offset + j * src.nb[1];
            let d_off = j * dst.nb[1];
            if s_off + row <= s.len() && d_off + row <= d.len() {
                d[d_off..d_off + row].copy_from_slice(&s[s_off..s_off + row]);
            }
        }
    }
    dst
}

// ---------------------------------------------------------------------------
// Submodules
// ---------------------------------------------------------------------------

mod budget;
mod cells;
mod slot;
mod state;
mod cache;
#[cfg(test)]
mod tests;

pub use cells::*;
pub use slot::*;
pub use cache::*;

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
// llama_kv_cells
// ---------------------------------------------------------------------------

/// `std::bitset<LLAMA_MAX_SEQ>` stand-in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqSet {
    words: [u64; LLAMA_MAX_SEQ / 64],
}

impl SeqSet {
    fn new() -> Self {
        Self { words: [0; LLAMA_MAX_SEQ / 64] }
    }

    fn test(&self, s: LSeqId) -> bool {
        debug_assert!(s >= 0 && (s as usize) < LLAMA_MAX_SEQ);
        let i = s as usize;
        (self.words[i / 64] >> (i % 64)) & 1 != 0
    }

    fn set(&mut self, s: LSeqId) {
        debug_assert!(s >= 0 && (s as usize) < LLAMA_MAX_SEQ);
        let i = s as usize;
        self.words[i / 64] |= 1u64 << (i % 64);
    }

    fn reset(&mut self, s: LSeqId) {
        debug_assert!(s >= 0 && (s as usize) < LLAMA_MAX_SEQ);
        let i = s as usize;
        self.words[i / 64] &= !(1u64 << (i % 64));
    }

    fn clear(&mut self) {
        for w in self.words.iter_mut() {
            *w = 0;
        }
    }

    fn any(&self) -> bool {
        self.words.iter().any(|w| *w != 0)
    }

    fn none(&self) -> bool {
        !self.any()
    }

    fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    fn first(&self) -> Option<LSeqId> {
        for (wi, w) in self.words.iter().enumerate() {
            if *w != 0 {
                return Some((wi * 64 + w.trailing_zeros() as usize) as LSeqId);
            }
        }
        None
    }
}

/// Meta information about KV cells that can be part of multiple sequences.
#[derive(Clone, Debug)]
pub struct KvCells {
    has_shift: bool,
    /// Indices of used cells (`pos[i] != -1`, allowed to carry no seq id).
    used: BTreeSet<CellIdx>,
    pos: Vec<Pos>,
    ext: Vec<KvCellExt>,
    /// Accumulated shifts since the last `reset_shift()`.
    shift: Vec<Pos>,
    /// Which sequences currently occupy the i-th cell.
    seq: Vec<SeqSet>,
    /// `(pos, cell)` pairs per sequence, ordered by position.
    seq_pos: Vec<BTreeSet<(Pos, CellIdx)>>,
    /// Transpose of `seq`, used to build the cell lists.
    seq_bits: Vec<Vec<u64>>,
}

impl Default for KvCells {
    fn default() -> Self {
        Self::new()
    }
}

impl KvCells {
    pub fn new() -> Self {
        Self {
            has_shift: false,
            used: BTreeSet::new(),
            pos: Vec::new(),
            ext: Vec::new(),
            shift: Vec::new(),
            seq: Vec::new(),
            seq_pos: std::iter::repeat_with(BTreeSet::new).take(LLAMA_MAX_SEQ).collect(),
            seq_bits: std::iter::repeat_with(Vec::new).take(LLAMA_MAX_SEQ).collect(),
        }
    }

    pub fn reset(&mut self) {
        for i in 0..self.pos.len() {
            self.pos[i] = -1;
            self.ext[i].reset();
            self.shift[i] = 0;
            self.seq[i].clear();
        }
        self.has_shift = false;
        self.used.clear();
        for s in 0..LLAMA_MAX_SEQ {
            self.seq_pos[s].clear();
            self.seq_bits[s].fill(0);
        }
    }

    pub fn reset_shift(&mut self) {
        self.has_shift = false;
        for s in self.shift.iter_mut() {
            *s = 0;
        }
    }

    pub fn size(&self) -> CellIdx {
        self.pos.len() as CellIdx
    }

    pub fn resize(&mut self, n: CellIdx) {
        let n = n as usize;
        self.pos.resize(n, -1);
        self.ext.resize(n, KvCellExt::default());
        self.shift.resize(n, 0);
        self.seq.resize(n, SeqSet::new());
        for s in 0..LLAMA_MAX_SEQ {
            self.seq_bits[s].clear();
        }
        self.reset();
    }

    pub fn is_empty(&self, i: CellIdx) -> bool {
        debug_assert!((i as usize) < self.pos.len());
        self.pos[i as usize] == -1
    }

    pub fn get_used(&self) -> CellIdx {
        self.used.len() as CellIdx
    }

    /// The index of the first cell that is used (0 if none).
    pub fn used_min(&self) -> CellIdx {
        self.used.iter().next().copied().unwrap_or(0)
    }

    /// The index of the last used cell + 1 (0 if none).
    pub fn used_max_p1(&self) -> CellIdx {
        self.used.iter().next_back().map(|i| i + 1).unwrap_or(0)
    }

    pub fn get_has_shift(&self) -> bool {
        self.has_shift
    }

    /// Copy the state of cells `[i, i + n)`.
    pub fn cp(&self, i: CellIdx, n: CellIdx) -> Self {
        let i = i as usize;
        let n = n as usize;
        debug_assert!(i + n <= self.pos.len());
        let mut res = Self::new();
        res.resize(n as CellIdx);
        for j in 0..n {
            let idx = i + j;
            res.pos[j] = self.pos[idx];
            res.ext[j] = self.ext[idx];
            res.seq[j] = self.seq[idx];
            debug_assert_eq!(self.shift[idx], 0);
        }
        res
    }

    /// Copy the state of cells `[idxs[0], idxs[1], ...]`.
    pub fn cp_idxs(&self, idxs: &[CellIdx]) -> Self {
        let mut res = Self::new();
        res.resize(idxs.len() as CellIdx);
        for (j, idx) in idxs.iter().enumerate() {
            let idx = *idx as usize;
            res.pos[j] = self.pos[idx];
            res.ext[j] = self.ext[idx];
            res.seq[j] = self.seq[idx];
            debug_assert_eq!(self.shift[idx], 0);
        }
        res
    }

    /// Set the state of cells `[i, i + other.size())`.
    pub fn set(&mut self, i: CellIdx, other: &Self) {
        let i = i as usize;
        debug_assert!(i + other.pos.len() <= self.pos.len());
        for (j, opos) in other.pos.iter().enumerate() {
            let idx = i + j;
            if self.pos[idx] == -1 && *opos != -1 {
                self.used.insert(idx as CellIdx);
            }
            if self.pos[idx] != -1 && *opos == -1 {
                self.used.remove(&(idx as CellIdx));
            }
            if self.pos[idx] != -1 {
                self.seq_pos_rm(idx);
            }
            self.pos[idx] = *opos;
            self.ext[idx] = other.ext[j];
            self.seq[idx] = other.seq[j];
            if self.pos[idx] != -1 {
                self.seq_pos_add(idx);
            }
            debug_assert_eq!(self.shift[idx], 0);
        }
    }

    /// Set the state of cells `[idxs[0], idxs[1], ...]`.
    pub fn set_idxs(&mut self, idxs: &[CellIdx], other: &Self) {
        debug_assert_eq!(idxs.len(), other.pos.len());
        for (j, idx) in idxs.iter().enumerate() {
            let idx = *idx as usize;
            if self.pos[idx] == -1 && other.pos[j] != -1 {
                self.used.insert(idx as CellIdx);
            }
            if self.pos[idx] != -1 && other.pos[j] == -1 {
                self.used.remove(&(idx as CellIdx));
            }
            if self.pos[idx] != -1 {
                self.seq_pos_rm(idx);
            }
            self.pos[idx] = other.pos[j];
            self.ext[idx] = other.ext[j];
            self.seq[idx] = other.seq[j];
            if self.pos[idx] != -1 {
                self.seq_pos_add(idx);
            }
            debug_assert_eq!(self.shift[idx], 0);
        }
    }

    /// Clear a non-empty cell.
    pub fn rm(&mut self, i: CellIdx) {
        let i = i as usize;
        debug_assert!(i < self.pos.len() && self.pos[i] != -1);
        self.seq_pos_rm(i);
        self.seq[i].clear();
        self.pos[i] = -1;
        self.ext[i].reset();
        self.shift[i] = 0;
        self.used.remove(&(i as CellIdx));
    }

    /// Remove seq_id from cell i; true if the cell becomes empty.
    pub fn seq_rm_cell(&mut self, i: CellIdx, seq_id: LSeqId) -> bool {
        let i = i as usize;
        debug_assert!(i < self.pos.len() && self.seq[i].test(seq_id) && self.pos[i] != -1);
        self.seq[i].reset(seq_id);
        self.seq_pos_dec(seq_id, i);
        if self.seq[i].none() {
            self.pos[i] = -1;
            self.ext[i].reset();
            self.shift[i] = 0;
            self.used.remove(&(i as CellIdx));
            true
        } else {
            false
        }
    }

    /// Remove seq_id from its cells in `[p0, p1)`; return the lowest freed index.
    pub fn seq_rm_range(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos) -> CellIdx {
        debug_assert!(seq_id >= 0 && (seq_id as usize) < LLAMA_MAX_SEQ);
        let mut res = self.size();
        let s = seq_id as usize;
        let pairs: Vec<(Pos, CellIdx)> = self.seq_pos[s]
            .range((p0, 0)..)
            .take_while(|e| e.0 < p1)
            .copied()
            .collect();
        for (_, i) in pairs {
            if self.seq_rm_cell(i, seq_id) {
                res = res.min(i);
            }
        }
        res
    }

    /// Keep only seq_id in cell i; true if the cell becomes empty.
    pub fn seq_keep(&mut self, i: CellIdx, seq_id: LSeqId) -> bool {
        let i = i as usize;
        debug_assert!(i < self.pos.len());
        if self.seq[i].test(seq_id) {
            self.seq_pos_rm(i);
            self.seq[i].clear();
            self.seq[i].set(seq_id);
            self.seq_pos_add(i);
            false
        } else if self.seq[i].any() {
            self.seq_pos_rm(i);
            self.seq[i].clear();
            self.pos[i] = -1;
            self.ext[i].reset();
            self.shift[i] = 0;
            self.used.remove(&(i as CellIdx));
            true
        } else {
            debug_assert_eq!(self.pos[i], -1);
            false
        }
    }

    /// Number of different sequences in the cell.
    pub fn seq_count(&self, i: CellIdx) -> usize {
        let i = i as usize;
        debug_assert!(i < self.pos.len() && self.pos[i] != -1);
        self.seq[i].count()
    }

    pub fn seq_get_all(&self, i: CellIdx) -> &SeqSet {
        let i = i as usize;
        debug_assert!(i < self.pos.len());
        &self.seq[i]
    }

    pub fn seq_has(&self, i: CellIdx, seq_id: LSeqId) -> bool {
        let i = i as usize;
        debug_assert!(i < self.pos.len() && seq_id >= 0);
        self.seq[i].test(seq_id)
    }

    pub fn seq_pos_get(&self, seq_id: LSeqId) -> &BTreeSet<(Pos, CellIdx)> {
        debug_assert!(seq_id >= 0 && (seq_id as usize) < LLAMA_MAX_SEQ);
        &self.seq_pos[seq_id as usize]
    }

    /// Token of the cell of `seq_id` at the largest position <= p; the cell
    /// with the highest index wins when several cells share that position.
    pub fn seq_pos_tok_le(&self, seq_id: LSeqId, p: Pos) -> Token {
        debug_assert!(seq_id >= 0 && (seq_id as usize) < LLAMA_MAX_SEQ);
        let sp = &self.seq_pos[seq_id as usize];
        match sp.range(..=(p, CellIdx::MAX)).next_back() {
            Some((_, i)) => self.ext[*i as usize].tok,
            None => LLAMA_TOKEN_NULL,
        }
    }

    /// Note: call only if the cell is not empty and `seq_id` is not in it.
    pub fn seq_add(&mut self, i: CellIdx, seq_id: LSeqId) {
        let i = i as usize;
        debug_assert!(i < self.pos.len() && self.pos[i] != -1 && !self.seq[i].test(seq_id));
        self.seq[i].set(seq_id);
        self.seq_pos_inc(seq_id, i);
    }

    /// The sequence id of a cell holding exactly one sequence.
    pub fn seq_get(&self, i: CellIdx) -> LSeqId {
        let i = i as usize;
        debug_assert_eq!(self.seq[i].count(), 1);
        self.seq[i].first().unwrap_or(-1)
    }

    pub fn seq_n_cells(&self, seq_id: LSeqId) -> CellIdx {
        debug_assert!(seq_id >= 0 && (seq_id as usize) < LLAMA_MAX_SEQ);
        self.seq_pos[seq_id as usize].len() as CellIdx
    }

    pub fn seq_n_cells_from(&self, seq_id: LSeqId, p_min: Pos) -> CellIdx {
        debug_assert!(seq_id >= 0 && (seq_id as usize) < LLAMA_MAX_SEQ);
        self.seq_pos[seq_id as usize]
            .range((p_min, 0)..)
            .count() as CellIdx
    }

    /// Lists of cells for each slice, in cell order, padded with -1.
    pub fn get_seq_cells(&self, slices: &[KvRowsSlice], data: &mut [i32], n_per_seq: usize) {
        debug_assert!(!slices.is_empty());
        let n_words = (self.used_max_p1() as usize + 63) / 64;
        for (s, slice) in slices.iter().enumerate() {
            let pm = slice.p_min;
            let base = s * n_per_seq;
            for d in data[base..base + n_per_seq].iter_mut() {
                *d = -1;
            }
            let mut dst = base;
            for w in 0..n_words {
                let mut word = 0u64;
                for seq_id in &slice.seq_ids {
                    let bits = &self.seq_bits[*seq_id as usize];
                    if w < bits.len() {
                        word |= bits[w];
                    }
                }
                let mut b = word;
                while b != 0 {
                    let i = (64 * w + ctz64(b) as usize) as CellIdx;
                    b &= b - 1;
                    if (i as usize) < self.pos.len() && self.pos[i as usize] >= pm {
                        debug_assert!(dst < base + n_per_seq);
                        data[dst] = i as i32;
                        dst += 1;
                    }
                }
            }
        }
    }

    pub fn seq_pos_min(&self, seq_id: LSeqId) -> Pos {
        debug_assert!(seq_id >= 0 && (seq_id as usize) < LLAMA_MAX_SEQ);
        let sp = &self.seq_pos[seq_id as usize];
        sp.iter().next().map(|e| e.0).unwrap_or(-1)
    }

    pub fn seq_pos_max(&self, seq_id: LSeqId) -> Pos {
        debug_assert!(seq_id >= 0 && (seq_id as usize) < LLAMA_MAX_SEQ);
        let sp = &self.seq_pos[seq_id as usize];
        sp.iter().next_back().map(|e| e.0).unwrap_or(-1)
    }

    pub fn pos_get(&self, i: CellIdx) -> Pos {
        let i = i as usize;
        debug_assert!(i < self.pos.len() && self.pos[i] != -1);
        self.pos[i]
    }

    pub fn ext_get(&self, i: CellIdx) -> &KvCellExt {
        let i = i as usize;
        debug_assert!(i < self.pos.len() && self.pos[i] != -1);
        &self.ext[i]
    }

    pub fn get_shift(&self, i: CellIdx) -> Pos {
        let i = i as usize;
        debug_assert!(i < self.pos.len() && self.pos[i] != -1);
        self.shift[i]
    }

    pub fn pos_in(&self, i: CellIdx, p0: Pos, p1: Pos) -> bool {
        let i = i as usize;
        debug_assert!(i < self.pos.len());
        self.pos[i] >= p0 && self.pos[i] < p1
    }

    /// Set the position of an empty cell; does not modify `has_shift`.
    pub fn pos_set(&mut self, i: CellIdx, p: Pos) {
        let i = i as usize;
        debug_assert!(i < self.pos.len() && self.pos[i] == -1 && self.seq[i].none());
        self.pos[i] = p;
        self.used.insert(i as CellIdx);
    }

    pub fn ext_set(&mut self, i: CellIdx, p: KvCellExt) {
        let i = i as usize;
        debug_assert!(i < self.ext.len());
        self.ext[i] = p;
    }

    /// `pos[i] += d`; sets `has_shift`. True if the cell becomes empty.
    pub fn pos_add(&mut self, i: CellIdx, d: Pos) -> bool {
        let i = i as usize;
        debug_assert!(i < self.pos.len() && self.pos[i] != -1);
        self.seq_pos_rm(i);
        self.pos[i] += d;
        self.shift[i] += d;
        self.has_shift = true;
        if self.pos[i] < 0 {
            self.seq[i].clear();
            self.pos[i] = -1;
            self.shift[i] = 0;
            self.used.remove(&(i as CellIdx));
            true
        } else {
            self.seq_pos_add(i);
            false
        }
    }

    /// `pos[i] /= d`; sets `has_shift`.
    pub fn pos_div(&mut self, i: CellIdx, d: i32) {
        let i = i as usize;
        debug_assert!(i < self.pos.len() && self.pos[i] != -1);
        let p_old = self.pos[i];
        self.seq_pos_rm(i);
        self.pos[i] /= d;
        self.shift[i] += p_old - self.pos[i];
        self.seq_pos_add(i);
        self.has_shift = true;
    }

    fn seq_pos_dec(&mut self, s: LSeqId, i: usize) {
        let si = s as usize;
        let removed = self.seq_pos[si].remove(&(self.pos[i], i as CellIdx));
        debug_assert!(removed);
        if !self.seq_bits[si].is_empty() {
            self.seq_bits[si][i / 64] &= !(1u64 << (i % 64));
        }
    }

    fn seq_pos_inc(&mut self, s: LSeqId, i: usize) {
        let si = s as usize;
        self.seq_pos[si].insert((self.pos[i], i as CellIdx));
        if self.seq_bits[si].is_empty() {
            self.seq_bits[si] = vec![0; (self.pos.len() + 63) / 64];
        }
        self.seq_bits[si][i / 64] |= 1u64 << (i % 64);
    }

    fn seq_pos_rm(&mut self, i: usize) {
        let sset = self.seq[i];
        for s in 0..LLAMA_MAX_SEQ {
            if sset.test(s as LSeqId) {
                self.seq_pos_dec(s as LSeqId, i);
            }
        }
    }

    fn seq_pos_add(&mut self, i: usize) {
        let sset = self.seq[i];
        for s in 0..LLAMA_MAX_SEQ {
            if sset.test(s as LSeqId) {
                self.seq_pos_inc(s as LSeqId, i);
            }
        }
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
// llama_kv_cache::slot_info and stream_copy_info
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SlotInfo {
    pub s0: SeqId,
    pub s1: SeqId,
    pub strm: Vec<SeqId>,
    pub idxs: Vec<Vec<CellIdx>>,
}

impl Default for SlotInfo {
    fn default() -> Self {
        Self { s0: LLAMA_MAX_SEQ as SeqId, s1: 0, strm: Vec::new(), idxs: Vec::new() }
    }
}

impl SlotInfo {
    pub fn head(&self) -> CellIdx {
        debug_assert_eq!(self.idxs.len(), 1);
        debug_assert!(!self.idxs[0].is_empty());
        self.idxs[0][0]
    }

    pub fn resize(&mut self, n: usize) {
        self.strm.resize(n, 0);
        self.idxs.resize_with(n, Vec::new);
    }

    pub fn size(&self) -> usize {
        debug_assert_eq!(self.idxs.len(), self.strm.len());
        debug_assert!(!self.idxs.is_empty());
        self.idxs[0].len()
    }

    pub fn n_stream(&self) -> usize {
        self.strm.len()
    }

    pub fn is_empty(&self) -> bool {
        self.idxs.is_empty()
    }

    pub fn clear(&mut self) {
        self.idxs.clear();
    }

    /// Check if indices are contiguous starting from `head()`.
    pub fn is_contiguous(&self) -> bool {
        if self.idxs.is_empty() || self.idxs[0].is_empty() {
            return true;
        }
        if self.idxs.len() > 1 {
            return false;
        }
        let h = self.idxs[0][0];
        for (i, idx) in self.idxs[0].iter().enumerate() {
            if *idx != h + i as CellIdx {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Debug, Default)]
pub struct StreamCopyInfo {
    pub ssrc: Vec<SeqId>,
    pub sdst: Vec<SeqId>,
}

impl StreamCopyInfo {
    pub fn is_empty(&self) -> bool {
        debug_assert_eq!(self.ssrc.len(), self.sdst.len());
        self.ssrc.is_empty()
    }
}

pub type SlotInfoVec = Vec<SlotInfo>;

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
    is_indexer: bool,

    n_rot_k: u32,
    n_rot_v: u32,
    n_embd_head_k_all: i32,
    n_embd_head_v_all: i32,

    attn_rot_hadamard: BTreeMap<i64, Vec<f32>>,
    attn_rot_bufs: Vec<MemoryBuf>,
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

    pub fn clear(&mut self, data: bool) {
        for s in 0..self.n_stream as usize {
            self.v_cells[s].reset();
            self.v_heads[s] = 0;
        }
        if data {
            for st in self.storages.iter() {
                st.borrow_mut().fill(0);
            }
        }
    }

    pub fn seq_rm(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos) -> bool {
        // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
        if self.other.is_some() {
            return true;
        }

        debug_assert!(seq_id == -1 || (seq_id >= 0 && (seq_id as usize) < self.seq_to_stream.len()));

        let p0 = if p0 < 0 { 0 } else { p0 };
        let p1 = if p1 < 0 { Pos::MAX } else { p1 };

        if seq_id >= 0 {
            let strm = self.seq_to_stream[seq_id as usize] as usize;
            let new_head = self.v_cells[strm].seq_rm_range(seq_id, p0, p1);
            let size = self.v_cells[strm].size();
            if new_head != size && new_head < self.v_heads[strm] {
                self.v_heads[strm] = new_head;
            }
        } else {
            // match any sequence
            for s in 0..self.n_stream as usize {
                let size = self.v_cells[s].size();
                let mut new_head = size;
                for i in 0..size {
                    if !self.v_cells[s].pos_in(i, p0, p1) {
                        continue;
                    }
                    self.v_cells[s].rm(i);
                    if new_head == size {
                        new_head = i;
                    }
                }
                if new_head != size && new_head < self.v_heads[s] {
                    self.v_heads[s] = new_head;
                }
            }
        }
        true
    }

    pub fn seq_cp(&mut self, seq_id_src: LSeqId, seq_id_dst: LSeqId, p0: Pos, p1: Pos) {
        // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
        if self.other.is_some() {
            return;
        }

        debug_assert!(seq_id_src >= 0 && (seq_id_src as usize) < self.seq_to_stream.len());
        debug_assert!(seq_id_dst >= 0 && (seq_id_dst as usize) < self.seq_to_stream.len());

        let s0 = self.seq_to_stream[seq_id_src as usize] as usize;
        let s1 = self.seq_to_stream[seq_id_dst as usize] as usize;

        if s0 == s1 {
            // Both sequences are in the same stream, no data copy is needed.
            if seq_id_src == seq_id_dst {
                return;
            }
            let p0 = if p0 < 0 { 0 } else { p0 };
            let p1 = if p1 < 0 { Pos::MAX } else { p1 };
            let size = self.v_cells[s0].size();
            for i in 0..size {
                if !self.v_cells[s0].pos_in(i, p0, p1) {
                    continue;
                }
                if self.v_cells[s0].seq_has(i, seq_id_src) {
                    self.v_cells[s0].seq_add(i, seq_id_dst);
                }
            }
            return;
        }

        // Cross-stream sequence copies require to copy the actual buffer data.
        let mut is_full = true;
        if p0 > 0 && p0 + 1 < self.get_size() as Pos {
            is_full = false;
        }
        if p1 > 0 && p1 + 1 < self.get_size() as Pos {
            is_full = false;
        }
        debug_assert!(is_full, "seq_cp() is only supported for full KV buffers");

        // Enqueue the copy operation - performed during the next update.
        self.sc_info.ssrc.push(s0 as SeqId);
        self.sc_info.sdst.push(s1 as SeqId);

        let size = self.v_cells[s0].size();
        let mut copied: Vec<(CellIdx, Pos, Pos, KvCellExt)> = Vec::new();
        for i in 0..size {
            if self.v_cells[s0].seq_has(i, seq_id_src) {
                let mut pos = self.v_cells[s0].pos_get(i);
                let shift = self.v_cells[s0].get_shift(i);
                let ext = *self.v_cells[s0].ext_get(i);
                if shift != 0 {
                    pos -= shift;
                    debug_assert!(pos >= 0);
                }
                copied.push((i, pos, shift, ext));
            }
        }

        self.v_cells[s1].reset();
        for (i, pos, shift, ext) in copied {
            self.v_cells[s1].pos_set(i, pos);
            self.v_cells[s1].seq_add(i, seq_id_dst);
            if shift != 0 {
                self.v_cells[s1].pos_add(i, shift);
            }
            self.v_cells[s1].ext_set(i, ext);
        }
        self.v_heads[s1] = self.v_heads[s0];
    }

    pub fn seq_keep(&mut self, seq_id: LSeqId) {
        // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
        if self.other.is_some() {
            return;
        }

        debug_assert!(seq_id >= 0 && (seq_id as usize) < self.seq_to_stream.len());
        let strm = self.seq_to_stream[seq_id as usize] as usize;
        let size = self.v_cells[strm].size();
        let mut new_head = size;
        for i in 0..size {
            if self.v_cells[strm].seq_keep(i, seq_id) && new_head == size {
                new_head = i;
            }
        }
        if new_head != size && new_head < self.v_heads[strm] {
            self.v_heads[strm] = new_head;
        }
    }

    pub fn seq_add(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, shift: Pos) {
        // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
        if self.other.is_some() {
            return;
        }

        debug_assert!(seq_id >= 0 && (seq_id as usize) < self.seq_to_stream.len());
        debug_assert!(self.params.hparams.n_pos_per_embd() == 1,
            "seq_add() is only supported for n_pos_per_embd() == 1");

        let strm = self.seq_to_stream[seq_id as usize] as usize;
        if shift == 0 {
            return;
        }

        let size = self.v_cells[strm].size();
        let mut new_head = size;
        let p0 = if p0 < 0 { 0 } else { p0 };
        let p1 = if p1 < 0 { Pos::MAX } else { p1 };

        // If there is no range then return early to avoid looping over all cells.
        if p0 == p1 {
            return;
        }

        for i in 0..size {
            if !self.v_cells[strm].pos_in(i, p0, p1) {
                continue;
            }
            if self.v_cells[strm].seq_has(i, seq_id) && self.v_cells[strm].pos_add(i, shift) {
                if new_head == size {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, start searching there. Otherwise start over.
        self.v_heads[strm] = if new_head != size { new_head } else { 0 };
    }

    pub fn seq_div(&mut self, seq_id: LSeqId, p0: Pos, p1: Pos, d: i32) {
        // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
        if self.other.is_some() {
            return;
        }

        debug_assert!(seq_id >= 0 && (seq_id as usize) < self.seq_to_stream.len());
        debug_assert!(self.params.hparams.n_pos_per_embd() == 1,
            "seq_div() is only supported for n_pos_per_embd() == 1");

        let strm = self.seq_to_stream[seq_id as usize] as usize;
        if d == 1 {
            return;
        }

        let p0 = if p0 < 0 { 0 } else { p0 };
        let p1 = if p1 < 0 { Pos::MAX } else { p1 };
        if p0 == p1 {
            return;
        }

        let size = self.v_cells[strm].size();
        for i in 0..size {
            if !self.v_cells[strm].pos_in(i, p0, p1) {
                continue;
            }
            if self.v_cells[strm].seq_has(i, seq_id) {
                self.v_cells[strm].pos_div(i, d);
            }
        }
    }

    pub fn seq_pos_min(&self, seq_id: LSeqId) -> Pos {
        if let Some(ref o) = self.other {
            return o.seq_pos_min(seq_id);
        }
        debug_assert!(seq_id >= 0 && (seq_id as usize) < self.seq_to_stream.len());
        let strm = self.seq_to_stream[seq_id as usize] as usize;
        self.v_cells[strm].seq_pos_min(seq_id)
    }

    pub fn seq_pos_max(&self, seq_id: LSeqId) -> Pos {
        if let Some(ref o) = self.other {
            return o.seq_pos_max(seq_id);
        }
        debug_assert!(seq_id >= 0 && (seq_id as usize) < self.seq_to_stream.len());
        let strm = self.seq_to_stream[seq_id as usize] as usize;
        self.v_cells[strm].seq_pos_max(seq_id)
    }

    pub fn memory_breakdown(&self) -> BTreeMap<String, usize> {
        let mut ret = BTreeMap::new();
        for buf in &self.ctxs_bufs {
            *ret.entry(buf.buft.clone()).or_insert(0) += buf.size;
        }
        for buf in &self.attn_rot_bufs {
            *ret.entry(buf.buft.clone()).or_insert(0) += buf.size;
        }
        ret
    }

    pub fn init_batch(&mut self, ubatches: Vec<UBatch>) -> KvCacheContext<'_> {
        // The batch allocator's splitting is done by the caller; preparation
        // and cell placement are identical to the C++ path.
        match self.prepare(&ubatches) {
            Ok(sinfos) => KvCacheContext::batch(self, sinfos, ubatches),
            Err(_) => KvCacheContext::failed(MemoryStatus::FailedPrepare),
        }
    }

    pub fn init_full(&mut self) -> KvCacheContext<'_> {
        KvCacheContext::full(self)
    }

    pub fn init_update(&mut self, do_shift: bool) -> KvCacheContext<'_> {
        KvCacheContext::update(self, do_shift, self.sc_info.clone())
    }

    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    pub fn get_can_shift(&self) -> bool {
        if self.params.model.arch == Arch::Step35 {
            return false;
        }
        self.params.hparams.n_pos_per_embd() <= 1
    }

    pub fn get_size(&self) -> CellIdx {
        self.v_cells[self.seq_to_stream[0] as usize].size()
    }

    pub fn get_n_seq_max(&self) -> SeqId {
        self.n_seq_max
    }

    pub fn get_n_stream(&self) -> SeqId {
        self.n_stream
    }

    pub fn get_has_shift(&self) -> bool {
        let mut result = false;
        for s in 0..self.n_stream as usize {
            result |= self.v_cells[s].get_has_shift();
        }
        result
    }

    pub fn type_k(&self) -> GgmlType {
        self.layers[0].k.ty
    }

    pub fn type_v(&self) -> GgmlType {
        self.layers[0].v.as_ref().map(|v| v.ty).unwrap_or(self.params.type_v)
    }

    pub fn get_layer_ids(&self) -> Vec<u32> {
        self.layers.iter().map(|l| l.il).collect()
    }

    pub fn get_k_storage(&self, il: i32) -> Option<Tensor> {
        let ikv = *self.map_layer_ids.get(&il)?;
        Some(self.layers[ikv].k.clone())
    }

    pub fn get_cells(&self, seq_id: LSeqId) -> &KvCells {
        debug_assert!(seq_id >= 0 && (seq_id as usize) < self.seq_to_stream.len());
        &self.v_cells[self.seq_to_stream[seq_id as usize] as usize]
    }

    pub fn get_stream(&self, seq_id: LSeqId) -> SeqId {
        debug_assert!(seq_id >= 0 && (seq_id as usize) < self.seq_to_stream.len());
        self.seq_to_stream[seq_id as usize]
    }

    pub fn get_n_kv(&self, sinfo: &SlotInfo) -> u32 {
        let mut result = 0u32;
        // Pad the n_kv value so that the graph remains constant across batches
        // and can be reused.
        for s in 0..sinfo.n_stream() {
            let cells = &self.v_cells[sinfo.strm[s] as usize];
            let padded = ggml_pad(cells.used_max_p1() as usize, N_PAD as usize) as u32;
            let v = (cells.size()).min(N_PAD.max(padded));
            result = result.max(v);
        }
        result
    }

    pub fn get_n_kv_slices(ubatch: &UBatch) -> u32 {
        if ubatch.n_seqs_unq == 1 {
            return 1;
        }

        // Each sequence set of an equal split is a single sequence.
        if ubatch.equal_seqs() && ubatch.n_seqs == ubatch.n_seqs_unq {
            return ubatch.n_seqs as u32;
        }

        // Runs of equal length with a single sequence each.
        if ubatch.n_seqs_unq != 0 && ubatch.n_tokens % ubatch.n_seqs_unq == 0 {
            let n_tps = ubatch.n_tokens / ubatch.n_seqs_unq;
            if n_tps == 0 {
                return 1;
            }
            let mut is_runs = true;
            for i in 0..ubatch.n_tokens {
                let first = ubatch.seq_id[i - i % n_tps].first().copied().unwrap_or(-1);
                if ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i].first().copied().unwrap_or(-2) != first {
                    is_runs = false;
                    break;
                }
            }
            if is_runs {
                return ubatch.n_seqs_unq as u32;
            }
        }

        // Cannot split into per-sequence slices.
        1
    }

    pub fn get_n_kv_rows(&self, ubatch: &UBatch) -> u32 {
        if self.n_stream != 1 {
            return 0;
        }

        let n_slices = Self::get_n_kv_slices(ubatch);
        let cells = &self.v_cells[0];
        let slices = get_kv_rows_slices(ubatch, n_slices, self.n_swa, self.swa_type);

        let mut n_max = 0u32;
        for slice in &slices {
            // A cell shared by several sequences is counted once per sequence;
            // an upper bound is enough.
            let mut n = 0u32;
            for seq_id in &slice.seq_ids {
                n += if slice.p_min < 0 {
                    cells.seq_n_cells(*seq_id)
                } else {
                    cells.seq_n_cells_from(*seq_id, slice.p_min)
                };
            }
            n_max = n_max.max(n.min(cells.used_max_p1()));
        }

        // Pad like n_kv so that the graph can be reused. Never 0.
        let padded = ggml_pad(n_max as usize, N_PAD_KV_ROWS as usize) as u32;
        N_PAD_KV_ROWS.max(padded).min(self.get_size())
    }

    fn layer_index(&self, il: i32) -> usize {
        *self.map_layer_ids.get(&il).unwrap_or(&0)
    }

    pub fn get_k(&self, _ctx: &mut TensorContext, il: i32, n_kv: u32, sinfo: &SlotInfo) -> Tensor {
        let ikv = self.layer_index(il);
        let k = self.layers[ikv].k.clone();
        let kv_size = self.get_size() as usize;
        let n_embd_k_gqa = k.ne[0];
        let ns = (sinfo.s1 - sinfo.s0 + 1) as usize;
        let h = self.params.hparams.n_embd_head_k() as usize;
        let nh = self.params.hparams.n_head_kv(il as u32) as usize;
        let row_head = ggml_row_size(k.ty, h);
        let row_gqa = ggml_row_size(k.ty, n_embd_k_gqa);
        let row_kv = ggml_row_size(k.ty, n_embd_k_gqa * kv_size);
        k.view_4d(
            [h, nh, n_kv as usize, ns],
            [row_head, row_head, row_gqa, row_kv],
            row_kv * sinfo.s0 as usize,
        )
    }

    pub fn get_v(&self, _ctx: &mut TensorContext, il: i32, n_kv: u32, sinfo: &SlotInfo) -> Tensor {
        let ikv = self.layer_index(il);
        let v = self.layers[ikv].v.clone().unwrap_or_else(|| {
            Tensor::new_3d(self.params.type_v, 1, self.get_size() as usize, self.n_stream as usize)
        });
        let kv_size = self.get_size() as usize;
        let n_embd_v_gqa = v.ne[0];
        let ns = (sinfo.s1 - sinfo.s0 + 1) as usize;
        let h = self.params.hparams.n_embd_head_v() as usize;
        let nh = self.params.hparams.n_head_kv(il as u32) as usize;

        if !self.v_trans {
            // note: v->nb[1] <= v->nb[2]
            let row_h = ggml_row_size(v.ty, h);
            let row_gqa = ggml_row_size(v.ty, n_embd_v_gqa);
            let row_kv = ggml_row_size(v.ty, n_embd_v_gqa * kv_size);
            return v.view_4d(
                [h, nh, n_kv as usize, ns],
                [row_h, row_h, row_gqa, row_kv],
                row_kv * sinfo.s0 as usize,
            );
        }

        // note: v->nb[1] > v->nb[2]
        let row_kvh = ggml_row_size(v.ty, kv_size * h);
        let row_kv = ggml_row_size(v.ty, kv_size);
        let row_all = ggml_row_size(v.ty, kv_size * n_embd_v_gqa);
        v.view_4d(
            [n_kv as usize, nh, h, ns],
            [row_kvh, row_kvh, row_kv, row_all],
            row_all * sinfo.s0 as usize,
        )
    }

    pub fn get_k_scale(&self, il: i32) -> Option<Tensor> {
        if self.type_k() != GgmlType::F8E4M3 {
            return None;
        }
        self.layers[self.layer_index(il)].k_scale.clone()
    }

    pub fn get_v_scale(&self, il: i32) -> Option<Tensor> {
        if self.type_v() != GgmlType::F8E4M3 {
            return None;
        }
        self.layers[self.layer_index(il)].v_scale.clone()
    }

    pub fn cpy_k(&self, _ctx: &mut TensorContext, k_cur: &Tensor, k_idxs: &[i64], il: i32) -> Tensor {
        let ikv = self.layer_index(il);
        let k = self.layers[ikv].k.clone();
        let n_embd_head = k_cur.ne[0];
        let n_head = k_cur.ne[1];
        let n_tokens = k_cur.ne[2];
        let n_embd_gqa = n_embd_head * n_head;
        debug_assert_eq!(ggml_row_size(k_cur.ty, n_embd_head), k_cur.nb[1]);

        // We can merge dims 0 and 1.
        let k_cur = k_cur.view_2d(n_embd_gqa, n_tokens, k_cur.nb[2], 0);
        let mut k = k;
        if k.ne[2] > 1 {
            // Merge the buffer across all streams because the idxs are global.
            k = k.reshape_2d(n_embd_gqa, k.ne[1] * k.ne[2]);
        }
        k.set_rows(&k_cur, k_idxs);
        k
    }

    pub fn cpy_v(&self, _ctx: &mut TensorContext, v_cur: &Tensor, v_idxs: &[i64], il: i32) -> Tensor {
        let ikv = self.layer_index(il);
        let v = self.layers[ikv].v.clone().unwrap_or_else(|| {
            Tensor::new_3d(self.params.type_v, 1, self.get_size() as usize, self.n_stream as usize)
        });
        let n_embd_head = v_cur.ne[0];
        let n_head = v_cur.ne[1];
        let n_tokens = v_cur.ne[2];
        let n_embd_gqa = n_embd_head * n_head;
        debug_assert_eq!(ggml_row_size(v_cur.ty, n_embd_head), v_cur.nb[1]);

        if !self.v_trans {
            let v_cur = v_cur.view_2d(n_embd_gqa, n_tokens, v_cur.nb[2], 0);
            let mut v = v;
            if v.ne[2] > 1 {
                v = v.reshape_2d(n_embd_gqa, v.ne[1] * v.ne[2]);
            }
            v.set_rows(&v_cur, v_idxs);
            return v;
        }

        let v_cur = if ggml_row_size(v_cur.ty, n_embd_gqa) == v_cur.nb[2] {
            v_cur.reshape_2d(n_embd_gqa, n_tokens)
        } else {
            ggml_cont_2d(v_cur, n_embd_gqa, n_tokens)
        };

        // [TAG_V_CACHE_VARIABLE]
        let v_cur = if n_embd_gqa < v.ne[0] {
            ggml_pad_2d(&v_cur, v.ne[0] - n_embd_gqa)
        } else {
            v_cur
        };

        // Each row is a single head element.
        let mut v_view = v.reshape_2d(1, v.nelements());
        let v_cur = v_cur.reshape_2d(1, v_cur.nelements());
        v_view.set_rows(&v_cur, v_idxs);
        v_view
    }

    pub fn build_input_k_idxs(&self, _ctx: &mut TensorContext, ubatch: &UBatch) -> Tensor {
        let mut t = Tensor::new_1d(GgmlType::I64, ubatch.n_tokens);
        t.is_input = true;
        t.name = "attn_inp_k_idxs".to_string();
        t
    }

    pub fn build_input_v_idxs(&self, _ctx: &mut TensorContext, ubatch: &UBatch) -> Tensor {
        let n = if !self.v_trans {
            ubatch.n_tokens
        } else {
            ubatch.n_tokens * self.params.hparams.n_embd_v_gqa_max() as usize
        };
        let mut t = Tensor::new_1d(GgmlType::I64, n);
        t.is_input = true;
        t.name = "attn_inp_v_idxs".to_string();
        t
    }

    pub fn get_k_rot(&self) -> Vec<Option<Tensor>> {
        let mut res = vec![None; self.params.hparams.n_layer_all as usize];
        for (&il, &idx) in self.map_layer_ids.iter() {
            if (il as usize) < res.len() {
                res[il as usize] = self.layers[idx].k_rot.clone();
            }
        }
        res
    }

    pub fn get_v_rot(&self) -> Vec<Option<Tensor>> {
        let mut res = vec![None; self.params.hparams.n_layer_all as usize];
        for (&il, &idx) in self.map_layer_ids.iter() {
            if (il as usize) < res.len() {
                res[il as usize] = self.layers[idx].v_rot.clone();
            }
        }
        res
    }

    pub fn set_input_k_idxs(&self, dst: &mut Tensor, ubatch: &UBatch, sinfo: &SlotInfo) {
        let n_tokens = ubatch.n_tokens;
        debug_assert_eq!(n_tokens, sinfo.size() * sinfo.n_stream());
        for s in 0..sinfo.n_stream() {
            let offs = (sinfo.strm[s] * self.get_size()) as i64;
            for i in 0..sinfo.size() {
                dst.write_i64_at(s * sinfo.size() + i, offs + sinfo.idxs[s][i] as i64);
            }
        }
    }

    pub fn set_input_v_idxs(&self, dst: &mut Tensor, ubatch: &UBatch, sinfo: &SlotInfo) {
        let n_tokens = ubatch.n_tokens;
        debug_assert_eq!(n_tokens, sinfo.size() * sinfo.n_stream());
        if !self.v_trans {
            for s in 0..sinfo.n_stream() {
                let offs = (sinfo.strm[s] * self.get_size()) as i64;
                for i in 0..sinfo.size() {
                    dst.write_i64_at(s * sinfo.size() + i, offs + sinfo.idxs[s][i] as i64);
                }
            }
        } else {
            // The V cache is transposed when not using flash attention.
            let kv_size = self.get_size() as i64;
            let n_embd_v_gqa = self.params.hparams.n_embd_v_gqa_max() as i64;
            for s in 0..sinfo.n_stream() {
                let offs = sinfo.strm[s] as i64 * kv_size * n_embd_v_gqa;
                for i in 0..sinfo.size() {
                    for j in 0..n_embd_v_gqa as usize {
                        dst.write_i64_at(
                            s * sinfo.size() * n_embd_v_gqa as usize + i * n_embd_v_gqa as usize + j,
                            offs + j as i64 * kv_size + sinfo.idxs[s][i] as i64,
                        );
                    }
                }
            }
        }
    }

    pub fn set_input_k_shift(&self, dst: &mut Tensor) {
        for s in 0..self.n_stream as usize {
            let cells = &self.v_cells[s];
            for i in 0..cells.size() as usize {
                let v = if cells.is_empty(i as CellIdx) { 0 } else { cells.get_shift(i as CellIdx) };
                dst.write_i32_at(s * cells.size() as usize + i, v);
            }
        }
    }

    pub fn set_input_attn_mask(
        &self,
        attn_mask: &mut AttnMask,
        ubatch: &UBatch,
        mut causal_attn: bool,
    ) {
        // Fill the cell rows of each slice.
        if let Some(ref mut rows) = attn_mask.rows {
            debug_assert_eq!(self.n_stream, 1);
            let n_rows = rows.ne[0];
            let n_slices = rows.ne[1];
            let slices = get_kv_rows_slices(ubatch, n_slices as u32, self.n_swa, self.swa_type);
            let mut data = vec![-1i32; rows.nelements()];
            self.v_cells[0].get_seq_cells(&slices, &mut data, n_rows);
            rows.set_bytes(&bytes_i32(&data));
        }

        // The mask is left unallocated when the graph only stores K/V.
        if let Some(ref mut dst) = attn_mask.raw {
            let rows: Option<Vec<i32>> = match attn_mask.rows.as_ref() {
                Some(r) if !r.storage.borrow().is_empty() => Some(
                    bytes_to_i32(&r.bytes(), r.nelements())
                ),
                _ => None,
            };
            let n_tokens = ubatch.n_tokens;
            let n_kv = dst.ne[0];
            let n_stream = dst.ne[3];
            debug_assert_eq!(n_tokens % n_stream, 0);
            let n_tps = n_tokens / n_stream;

            // See llama_non_causal_type.
            if !causal_attn && self.params.hparams.non_causal_type == NonCausalType::SwaOnly {
                causal_attn = self.swa_type == SwaType::None;
            }
            if !causal_attn && self.params.hparams.non_causal_type == NonCausalType::DenseOnly {
                causal_attn = self.swa_type != SwaType::None;
            }

            let args = AttnMaskArgs {
                hparams: &self.params.hparams,
                ubatch,
                v_cells: &self.v_cells,
                seq_to_stream: &self.seq_to_stream,
                n_swa: self.n_swa,
                swa_type: self.swa_type,
                n_kv,
                n_stream,
                n_tps,
                rows: rows.as_deref(),
                block_uniform: attn_mask.block_uniform,
            };
            let mut data = vec![0f32; dst.nelements()];
            set_input_attn_mask_impl_f32(&args, &mut data, causal_attn);
            if dst.ty == GgmlType::F16 {
                let h: Vec<u16> = data.iter().map(|f| f32_to_f16(*f)).collect();
                let bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
                dst.set_bytes(&bytes);
            } else {
                dst.set_bytes(&bytes_f32(&data));
            }
        }
    }

    pub fn set_input_pos_bucket(&self, dst: &mut Tensor, ubatch: &UBatch) {
        let n_tokens = ubatch.n_tokens;
        debug_assert_eq!(self.n_stream, 1, "TODO: support multiple streams");
        let cells = &self.v_cells[0];
        debug_assert!(!ubatch.equal_seqs());
        let n_kv = dst.ne[0];
        for i in 0..n_tokens {
            for j in 0..n_kv {
                // The position when the cell is empty is irrelevant - it is
                // masked out later in the attention.
                let p0 = if cells.is_empty(j as CellIdx) { -1 } else { cells.pos_get(j as CellIdx) };
                let h = self.params.hparams.n_rel_attn_bkts as u64;
                dst.write_i32_at(i * n_kv + j, llama_relative_position_bucket(p0, ubatch.pos[i], h, false));
            }
        }
    }

    pub fn set_input_kpool(
        &self,
        pool_idxs: &mut Tensor,
        pool_pos: &mut Tensor,
        pool_mask: &mut Tensor,
        tail_idxs: &mut Tensor,
        ubatch: &UBatch,
        sinfo: &SlotInfo,
        kpool: u32,
        n_kv: u32,
    ) {
        debug_assert_eq!(self.n_stream, 1, "TODO: support multiple streams");
        debug_assert_eq!(pool_idxs.ne[0], kpool as usize);
        debug_assert_eq!(tail_idxs.ne[0], kpool as usize);
        debug_assert_eq!(sinfo.idxs.len(), 1);
        debug_assert_eq!(sinfo.idxs[0].len(), ubatch.n_tokens);

        let cells = &self.v_cells[0];
        let n_tokens = ubatch.n_tokens;
        let n_pool = pool_idxs.ne[1];
        let n_planes = self.params.hparams.n_pos_per_embd() as usize;
        let kpool = kpool as usize;
        let n_kv = n_kv as usize;

        // One pool per kpool consecutive tokens of each sequence, in sequence
        // order. The order is the position, then the cell.
        let mut idxs: Vec<i32> = Vec::new();
        let mut first: Vec<i32> = Vec::new();
        let mut pool_of: BTreeMap<Vec<i32>, i32> = BTreeMap::new();
        let dedup = ubatch.n_seqs_unq > 1;
        let mut seq_pools = vec![Vec::new(); ubatch.n_seqs_unq];
        let mut order = vec![Vec::new(); ubatch.n_seqs_unq];
        let mut rank = vec![Vec::new(); ubatch.n_seqs_unq];
        let mut seq_cells: Vec<(Pos, i32)> = Vec::new();

        for s in 0..ubatch.n_seqs_unq {
            let seq_id = ubatch.seq_id_unq[s];
            seq_cells.clear();
            for j in 0..n_kv {
                if !cells.is_empty(j as CellIdx) && cells.seq_has(j as CellIdx, seq_id) {
                    seq_cells.push((cells.pos_get(j as CellIdx), j as i32));
                }
            }
            if !seq_cells.is_sorted() {
                seq_cells.sort();
            }

            let ord = &mut order[s];
            let rnk = &mut rank[s];
            ord.resize(seq_cells.len(), 0);
            rnk.resize(n_kv, 0);
            rnk.fill(-1);
            for (r, cell) in seq_cells.iter().enumerate() {
                ord[r] = cell.1;
                rnk[ord[r] as usize] = r as i32;
            }

            let n_groups = ord.len() / kpool;
            for g in 0..n_groups {
                let beg = g * kpool;
                let members: Vec<i32> = ord[beg..beg + kpool].to_vec();
                let mut k = first.len() as i32;
                if dedup {
                    k = *pool_of.entry(members).or_insert(k);
                }
                if k == first.len() as i32 {
                    idxs.extend_from_slice(&ord[beg..beg + kpool]);
                    first.push(ord[beg]);
                }
                seq_pools[s].push(k);
            }
        }

        debug_assert!((first.len() as i64) <= n_pool as i64);

        // An unused pool is never picked, but its members are still gathered.
        let mut pool_idxs_data = vec![0i32; kpool * n_pool];
        let n_copy = idxs.len().min(pool_idxs_data.len());
        pool_idxs_data[..n_copy].copy_from_slice(&idxs[..n_copy]);

        // The pooled key is rotated at the position of its first member.
        let mut pool_pos_data = vec![0i32; n_pool * n_planes];
        for (k, cell) in first.iter().enumerate() {
            let t = cells.pos_get(*cell as CellIdx);
            pool_pos_data[k] = t;
            if n_planes == 4 {
                let e = cells.ext_get(*cell as CellIdx);
                pool_pos_data[1 * n_pool + k] = e.y;
                pool_pos_data[2 * n_pool + k] = e.x;
                pool_pos_data[3 * n_pool + k] = t;
            }
        }

        let mut pool_mask_data = vec![f32_to_f16(f32::NEG_INFINITY); n_pool * n_tokens];
        let mut tail_idxs_data = vec![0i32; kpool * n_tokens];
        let zero = f32_to_f16(0.0);
        for i in 0..n_tokens {
            let seq_id = ubatch.seq_id[i][0];
            let s = seq_id as usize;
            let r = rank[s][sinfo.idxs[0][i] as usize];
            debug_assert!(r >= 0);
            let group = r as usize / kpool;
            for g in 0..group {
                let pi = seq_pools[s][g] as usize;
                if i * n_pool + pi < pool_mask_data.len() {
                    pool_mask_data[i * n_pool + pi] = zero;
                }
            }
            for m in 0..kpool {
                let rm = group * kpool + m;
                tail_idxs_data[i * kpool + m] = if rm <= r as usize {
                    order[s][rm]
                } else {
                    sinfo.idxs[0][i] as i32
                };
            }
        }

        pool_idxs.set_bytes(&bytes_i32(&pool_idxs_data));
        pool_pos.set_bytes(&bytes_i32(&pool_pos_data));
        let mask_bytes: Vec<u8> = pool_mask_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        pool_mask.set_bytes(&mask_bytes);
        tail_idxs.set_bytes(&bytes_i32(&tail_idxs_data));
    }

    // True if llama_kv_cell_ext holds information that must survive a restore.
    pub fn has_cell_ext(&self) -> bool {
        self.params.hparams.n_pos_per_embd() > 1 || self.params.hparams.ple_n_heads > 0
    }

    pub fn set_input_relative_pos(&self, dst: &mut Tensor, ubatch: &UBatch, extent: u32) {
        let n_kv = dst.ne[0];
        let mut data = vec![0i32; n_kv * ubatch.n_tokens];
        for i in 0..ubatch.n_tokens {
            let strm = self.seq_to_stream[ubatch.seq_id[i][0] as usize] as usize;
            let cells = &self.v_cells[strm];
            for j in 0..n_kv {
                let distance = if cells.is_empty(j as CellIdx) {
                    -1
                } else {
                    ubatch.pos[i] as i64 - cells.pos_get(j as CellIdx) as i64
                };
                data[i * n_kv + j] = if distance >= 0 && distance < extent as i64 {
                    distance as i32
                } else {
                    extent as i32
                };
            }
        }
        dst.set_bytes(&bytes_i32(&data));
    }

    pub fn get_prev_tokens(&self, ubatch: &UBatch, n: u32, res: &mut Vec<Token>) {
        let n_tokens = ubatch.n_tokens;
        res.clear();
        res.resize(n_tokens * n as usize, LLAMA_TOKEN_NULL);
        if n == 0 {
            return;
        }

        // An embd (multimodal) or mixed ubatch can repeat one position, so
        // resolve predecessors by ubatch order instead.
        let by_order = ubatch.token.is_none() || ubatch.is_mixed();
        let mut ord = vec![0usize; n_tokens];
        let mut seq_idx: BTreeMap<LSeqId, Vec<usize>> = BTreeMap::new();
        if by_order {
            for i in 0..n_tokens {
                let v = seq_idx.entry(ubatch.seq_id[i][0]).or_default();
                ord[i] = v.len();
                v.push(i);
            }
        }

        for i in 0..n_tokens {
            let seq_id = ubatch.seq_id[i][0];
            for j in 0..n as usize {
                let d = (n - j as u32) as Pos;
                let p = if by_order {
                    let v = &seq_idx[&seq_id];
                    let k = ord[i] as i64 - d as i64;
                    if k >= 0 {
                        ubatch.pos[v[k as usize]]
                    } else {
                        ubatch.pos[v[0]] + k as Pos
                    }
                } else {
                    ubatch.pos[i] - d
                };
                if p < 0 {
                    continue;
                }
                let strm = self.seq_to_stream[seq_id as usize] as usize;
                res[i * n as usize + j] = self.v_cells[strm].seq_pos_tok_le(seq_id, p);
            }
        }
    }

    pub fn prepare(&mut self, ubatches: &[UBatch]) -> Result<SlotInfoVec> {
        struct State {
            sinfo: SlotInfo,
            v_heads_old: Vec<CellIdx>,
            v_cells: Vec<KvCells>,
        }

        // Remember the old state of the cells so we can restore it in the end.
        let mut states: Vec<State> = Vec::new();
        let mut res: SlotInfoVec = Vec::new();
        let mut success = true;

        for ubatch in ubatches {
            // Only find a suitable slot; don't modify the cells yet.
            let sinfo_new = self.find_slot(ubatch, false);
            if sinfo_new.is_empty() {
                success = false;
                break;
            }
            res.push(sinfo_new.clone());

            let mut state = State {
                sinfo: sinfo_new.clone(),
                v_heads_old: self.v_heads.clone(),
                v_cells: Vec::new(),
            };
            for s in 0..sinfo_new.n_stream() {
                let cells = &self.v_cells[sinfo_new.strm[s] as usize];
                state.v_cells.push(cells.cp_idxs(&sinfo_new.idxs[s]));
            }
            states.push(state);

            // Now emplace the ubatch.
            self.apply_ubatch(&sinfo_new, ubatch);
        }

        debug_assert!(!states.is_empty() || !success);

        // Iterate backwards and restore the cells to their original state.
        for state in states.iter().rev() {
            for s in 0..state.sinfo.n_stream() {
                let strm = state.sinfo.strm[s] as usize;
                self.v_cells[strm].set_idxs(&state.sinfo.idxs[s], &state.v_cells[s]);
                self.v_heads[strm] = state.v_heads_old[strm];
            }
        }

        if !success {
            return Err(Error::new("failed to find a suitable KV slot"));
        }
        Ok(res)
    }

    pub fn update(&mut self, do_shift: bool, sc_info: &StreamCopyInfo) -> bool {
        // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
        if self.other.is_some() {
            return true;
        }

        let mut updated = false;

        if !sc_info.is_empty() {
            debug_assert!(self.n_stream > 1, "stream copy should never happen with a single stream");
            for i in 0..sc_info.ssrc.len() {
                let ssrc = sc_info.ssrc[i] as usize;
                let sdst = sc_info.sdst[i] as usize;
                debug_assert!(ssrc < self.n_stream as usize);
                debug_assert!(sdst < self.n_stream as usize);
                debug_assert_ne!(ssrc, sdst);

                for li in 0..self.layers.len() {
                    let layer = &self.layers[li];
                    let mut dk = layer.k_stream[sdst].clone();
                    let bytes = layer.k_stream[ssrc].bytes();
                    dk.set_bytes(&bytes);
                    if let (Some(ss), Some(ds)) =
                        (layer.v_stream[ssrc].clone(), layer.v_stream[sdst].clone())
                    {
                        let bytes = ss.bytes();
                        let mut ds = ds;
                        ds.set_bytes(&bytes);
                    }
                }
            }
        }

        if do_shift {
            if !self.get_can_shift() {
                // C++ aborts here; a port cannot execute the ggml compute graph.
                return updated;
            }

            // The K-shift graph is a backend operation. The position shifts
            // themselves live in the cells and are committed below.
            for s in 0..self.n_stream as usize {
                self.v_cells[s].reset_shift();
            }
            updated = true;
        }

        updated
    }

    pub fn find_slot(&self, ubatch: &UBatch, cont: bool) -> SlotInfo {
        let mut n_tokens = ubatch.n_tokens;
        let mut n_seqs = 1usize;

        if self.n_stream > 1 {
            debug_assert!(ubatch.n_seqs_unq != 0 && n_tokens % ubatch.n_seqs_unq == 0);
            n_seqs = ubatch.n_seqs_unq;
            n_tokens /= n_seqs;
        }

        let mut res = SlotInfo::default();
        res.resize(n_seqs);

        for s in 0..n_seqs {
            let seq_id = ubatch.seq_id_unq[s];
            debug_assert!(seq_id >= 0);
            let strm = self.seq_to_stream[seq_id as usize];

            if self.n_stream > 1 {
                debug_assert_eq!(ubatch.n_seq_id[s * n_tokens], 1);
                debug_assert_eq!(ubatch.seq_id[s * n_tokens][0], seq_id);
            }

            res.s0 = res.s0.min(strm);
            res.s1 = res.s1.max(strm);
            res.strm[s] = strm;
            res.idxs[s].reserve(n_tokens);

            let cells = &self.v_cells[strm as usize];
            let mut head_cur = self.v_heads[strm as usize];

            // If we have enough unused cells before the current head, better
            // to start searching from the beginning of the cache.
            if head_cur > cells.get_used() + 2 * n_tokens as CellIdx {
                head_cur = 0;
            }

            if n_tokens > cells.size() as usize {
                return SlotInfo::default();
            }

            let mut n_tested = 0usize;
            let n_test = if cont { n_tokens } else { 1 };

            loop {
                if head_cur as usize + n_test > cells.size() as usize {
                    n_tested += (cells.size() - head_cur) as usize;
                    head_cur = 0;
                    continue;
                }

                for _i in 0..n_test {
                    let idx = head_cur;
                    head_cur += 1;
                    n_tested += 1;

                    // Can we use this cell? Either it is empty, or it is
                    // occupied only by one sequence masked by SWA.
                    let mut can_use = cells.is_empty(idx);
                    if !can_use && cells.seq_count(idx) == 1 {
                        let pos_cell = cells.pos_get(idx);
                        if !can_use {
                            let seq_id_cell = cells.seq_get(idx);
                            can_use = HParams::is_masked_swa(
                                self.n_swa,
                                self.swa_type,
                                pos_cell,
                                cells.seq_pos_max(seq_id_cell) + 1,
                            );
                        }
                    }

                    if can_use {
                        res.idxs[s].push(idx);
                    } else if cont {
                        break;
                    }
                }

                if res.idxs[s].len() == n_tokens {
                    break;
                }
                if cont {
                    res.idxs[s].clear();
                }
                if n_tested >= cells.size() as usize {
                    return SlotInfo::default();
                }
            }

            if res.idxs[s].len() < n_tokens {
                return SlotInfo::default();
            }
        }

        debug_assert!(res.s1 >= res.s0);
        res
    }

    pub fn apply_ubatch(&mut self, sinfo: &SlotInfo, ubatch: &UBatch) {
        // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
        if self.other.is_some() {
            return;
        }

        // Keep track of the max sequence position that we would overwrite.
        // For a non-SWA cache this is always empty.
        let mut seq_pos_max_rm = vec![-1 as Pos; LLAMA_MAX_SEQ];
        debug_assert_eq!(ubatch.n_tokens, sinfo.n_stream() * sinfo.size());

        for (s, stream) in sinfo.strm.iter().enumerate() {
            let strm = *stream as usize;
            for ii in 0..sinfo.size() {
                let i = s * sinfo.size() + ii;
                let idx = sinfo.idxs[s][ii];

                if !self.v_cells[strm].is_empty(idx) {
                    debug_assert_eq!(self.v_cells[strm].seq_count(idx), 1);
                    let seq_id = self.v_cells[strm].seq_get(idx);
                    let pos = self.v_cells[strm].pos_get(idx);
                    seq_pos_max_rm[seq_id as usize] = seq_pos_max_rm[seq_id as usize].max(pos);
                    self.v_cells[strm].rm(idx);
                }

                self.v_cells[strm].pos_set(idx, ubatch.pos[i]);

                if ubatch.is_pos_2d() || ubatch.token.is_some() || self.params.hparams.ple_n_heads > 0 {
                    let mut ext = KvCellExt::default();
                    if ubatch.is_pos_2d() {
                        ext.x = ubatch.pos[i + ubatch.n_tokens * 2];
                        ext.y = ubatch.pos[i + ubatch.n_tokens];
                    }
                    let is_embd = ubatch.token.is_none()
                        || (ubatch.is_mixed() && ubatch.type_embd.get(i).copied().unwrap_or(false));
                    if !is_embd {
                        ext.tok = ubatch.token.as_ref().map(|t| t[i]).unwrap_or(LLAMA_TOKEN_NULL);
                    } else if self.params.hparams.ple_n_heads > 0 {
                        // Embedding input has no token ids; pad with the PLE id.
                        ext.tok = self.params.hparams.ple_embd_token;
                    }
                    self.v_cells[strm].ext_set(idx, ext);
                }

                for seq_id in &ubatch.seq_id[i] {
                    self.v_cells[strm].seq_add(idx, *seq_id);
                }
            }
        }

        // Preserve the invariant that all positions between [pos_min, pos_max]
        // for each sequence are present in the cache: purge positions below
        // those we would overwrite.
        for s in 0..LLAMA_MAX_SEQ {
            if seq_pos_max_rm[s] == -1 {
                continue;
            }
            debug_assert!(s < self.seq_to_stream.len());
            let strm = self.seq_to_stream[s] as usize;
            let p_min = self.v_cells[strm].seq_pos_min(s as LSeqId);
            if p_min <= seq_pos_max_rm[s] {
                self.seq_rm(s as LSeqId, p_min, seq_pos_max_rm[s] + 1);
            }
        }

        // Move the head at the end of the slot.
        for s in 0..sinfo.n_stream() {
            let strm = sinfo.strm[s] as usize;
            self.v_heads[strm] = *sinfo.idxs[s].last().unwrap() + 1;
        }
    }

    pub fn total_size(&self) -> usize {
        self.ctxs_bufs.iter().map(|b| b.size).sum()
    }

    pub fn size_k_bytes(&self) -> usize {
        let mut size = 0;
        for layer in &self.layers {
            let n = self.params.hparams.n_embd_k_gqa(layer.il) as usize;
            size += ggml_row_size(layer.k.ty, n)
                * self.get_size() as usize
                * self.n_stream as usize;
        }
        size
    }

    pub fn size_v_bytes(&self) -> usize {
        let mut size = 0;
        for layer in &self.layers {
            if let Some(ref v) = layer.v {
                let n = self.params.hparams.n_embd_v_gqa(layer.il) as usize;
                size += ggml_row_size(v.ty, n) * self.get_size() as usize * self.n_stream as usize;
            }
        }
        size
    }

    pub fn state_write<W: IoWrite>(&self, io: &mut W, seq_id: LSeqId, flags: u32) {
        // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
        if self.other.is_some() {
            return;
        }
        if (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0 && self.swa_type == SwaType::None {
            return;
        }

        write_u32(io, self.n_stream);

        for s in 0..self.n_stream as usize {
            let mut cr = CellRanges { strm: s as SeqId, data: Vec::new() };
            let mut cell_count = 0u32;
            let cells = &self.v_cells[s];

            // Find all ranges of cells with this seq id (or all, when -1).
            let mut cell_range_begin = cells.size();
            for i in 0..cells.size() {
                let mut add_cell = !cells.is_empty(i)
                    && (seq_id == -1 || cells.seq_has(i, seq_id));
                if add_cell && seq_id != -1 {
                    let masked = HParams::is_masked_swa(
                        self.n_swa,
                        self.swa_type,
                        cells.pos_get(i),
                        cells.seq_pos_max(seq_id),
                    );
                    add_cell = !masked;
                }
                if add_cell {
                    cell_count += 1;
                    if cell_range_begin == cells.size() {
                        cell_range_begin = i;
                    }
                } else if cell_range_begin != cells.size() {
                    cr.data.push((cell_range_begin, i));
                    cell_range_begin = cells.size();
                }
            }
            if cell_range_begin != cells.size() {
                cr.data.push((cell_range_begin, cells.size()));
            }

            // DEBUG CHECK: sum of cell counts in ranges equals the total.
            let cell_count_check: u32 = cr.data.iter().map(|r| r.1 - r.0).sum();
            debug_assert_eq!(cell_count, cell_count_check);

            write_u32(io, cell_count);
            if cell_count == 0 {
                continue;
            }
            self.state_write_meta(io, &cr, seq_id);
            self.state_write_data(io, &cr);
        }
    }

    fn state_write_meta<W: IoWrite>(&self, io: &mut W, cr: &CellRanges, seq_id: LSeqId) {
        let cells = &self.v_cells[cr.strm as usize];
        for (first, second) in &cr.data {
            for i in *first..*second {
                let mut seq_ids = Vec::new();
                for cur in 0..self.n_seq_max as LSeqId {
                    if cur == seq_id || seq_id == -1 {
                        if cells.seq_has(i, cur) {
                            seq_ids.push(cur);
                        }
                    }
                }
                let pos = cells.pos_get(i);
                write_i32(io, pos);
                write_u32(io, seq_ids.len() as u32);
                if self.has_cell_ext() {
                    let ext = cells.ext_get(i);
                    write_i32(io, ext.x);
                    write_i32(io, ext.y);
                    write_i32(io, ext.tok);
                }
                for id in seq_ids {
                    write_i32(io, id);
                }
            }
        }
    }

    fn state_write_data<W: IoWrite>(&self, io: &mut W, cr: &CellRanges) {
        write_u32(io, if self.v_trans { 1 } else { 0 });
        write_u32(io, self.layers.len() as u32);
        write_u32(io, self.n_rot_k);
        write_u32(io, self.n_rot_v);

        // Iterate and write all the keys first, each row is a cell.
        for layer in &self.layers {
            let n_embd_k_gqa = self.params.hparams.n_embd_k_gqa(layer.il) as usize;
            let k = &layer.k_stream[cr.strm as usize];
            write_i32(io, k.ty as i32);
            let k_size_row = ggml_row_size(k.ty, n_embd_k_gqa) as u64;
            write_u64(io, k_size_row);
            for (first, second) in &cr.data {
                let buf_size = (*second - *first) as usize * k_size_row as usize;
                io.write_tensor(k, *first as usize * k_size_row as usize, buf_size);
            }
        }

        if !self.v_trans {
            for layer in &self.layers {
                let n_embd_v_gqa = self.params.hparams.n_embd_v_gqa(layer.il) as usize;
                let v = match layer.v_stream[cr.strm as usize].as_ref() {
                    Some(v) => v,
                    None => continue,
                };
                write_i32(io, v.ty as i32);
                let v_size_row = ggml_row_size(v.ty, n_embd_v_gqa) as u64;
                write_u64(io, v_size_row);
                for (first, second) in &cr.data {
                    let buf_size = (*second - *first) as usize * v_size_row as usize;
                    io.write_tensor(v, *first as usize * v_size_row as usize, buf_size);
                }
            }
        } else {
            // When v is transposed, also store the element size and get the
            // element ranges from each row.
            let cells = &self.v_cells[cr.strm as usize];
            for layer in &self.layers {
                let n_embd_v_gqa = self.params.hparams.n_embd_v_gqa(layer.il) as usize;
                let v = match layer.v_stream[cr.strm as usize].as_ref() {
                    Some(v) => v,
                    None => continue,
                };
                write_i32(io, v.ty as i32);
                write_u32(io, v.ty.type_size() as u32);
                write_u32(io, n_embd_v_gqa as u32);
                for j in 0..n_embd_v_gqa {
                    for (first, second) in &cr.data {
                        let src_offset = (*first as usize + j * cells.size() as usize) * v.ty.type_size();
                        let buf_size = (*second - *first) as usize * v.ty.type_size();
                        io.write_tensor(v, src_offset, buf_size);
                    }
                }
            }
        }
    }

    pub fn state_read<R: IoRead>(&mut self, io: &mut R, seq_id: LSeqId, flags: u32) -> Result<()> {
        self.state_read_sinfo(io, seq_id, flags, None, None)
    }

    pub fn state_read_sinfo<R: IoRead>(
        &mut self,
        io: &mut R,
        seq_id: LSeqId,
        flags: u32,
        mut sinfos_out: Option<&mut SlotInfoVec>,
        sinfos_in: Option<&SlotInfoVec>,
    ) -> Result<()> {
        // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
        if self.other.is_some() {
            return Ok(());
        }
        if (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0 && self.swa_type == SwaType::None {
            return Ok(());
        }
        debug_assert!(seq_id == -1 || (seq_id >= 0 && (seq_id as usize) < self.seq_to_stream.len()));

        if let Some(ref mut out) = sinfos_out {
            out.clear();
            out.resize(self.n_stream as usize, SlotInfo::default());
        }
        if let Some(input) = sinfos_in {
            if input.len() != self.n_stream as usize {
                return Err(Error::new("failed to restore kv cache: mirrored slot layout has the wrong stream count"));
            }
        }

        let n_stream_cur = read_u32(io);
        if n_stream_cur != self.n_stream {
            return Err(Error::new("n_stream mismatch"));
        }

        // A whole-context restore replaces every stream, so clear once here.
        if seq_id == -1 {
            self.clear(true);
        }

        for s in 0..self.n_stream as usize {
            let cell_count = read_u32(io);
            if cell_count == 0 {
                if let Some(input) = sinfos_in {
                    if !input[s].is_empty() {
                        return Err(Error::new("failed to restore kv cache: mirrored cache holds cells this one does not"));
                    }
                }
                continue;
            }

            let strm = if seq_id == -1 { s as SeqId } else { self.seq_to_stream[seq_id as usize] };
            let mut sinfo = SlotInfo::default();
            let sinfo_in = sinfos_in.map(|v| &v[s]);

            let mut res = self.state_read_meta(io, strm, cell_count, &mut sinfo, seq_id, sinfo_in);
            if res {
                res = self.state_read_data(io, strm, cell_count, &sinfo);
            }

            if !res {
                self.state_clear_inner(seq_id, strm, &sinfo);
                return Err(Error::new("failed to restore kv cache"));
            }

            if let Some(ref mut out) = sinfos_out {
                out[s] = sinfo;
            }
        }
        Ok(())
    }

    fn state_read_meta<R: IoRead>(
        &mut self,
        io: &mut R,
        strm: SeqId,
        cell_count: u32,
        sinfo: &mut SlotInfo,
        dest_seq_id: LSeqId,
        sinfo_in: Option<&SlotInfo>,
    ) -> bool {
        let strm_idx = strm as usize;
        let cells_size = self.v_cells[strm_idx].size();

        if dest_seq_id != -1 {
            // single sequence
            if cell_count > cells_size {
                return false;
            }
            self.seq_rm(dest_seq_id, -1, -1);

            let n_pos = self.params.hparams.n_pos_per_embd();
            let mut ubatch = UBatch::ubatch_reserve(cell_count as usize, n_pos, self.has_cell_ext());
            ubatch.seq_id_unq[0] = dest_seq_id;

            let mut exts = if self.has_cell_ext() {
                vec![KvCellExt::default(); cell_count as usize]
            } else {
                Vec::new()
            };

            for i in 0..cell_count as usize {
                let pos = read_i32(io);
                let n_seq_id = read_u32(io);
                if n_seq_id != 1 {
                    return false;
                }
                if self.has_cell_ext() {
                    let ext = KvCellExt {
                        x: read_i32(io),
                        y: read_i32(io),
                        tok: read_i32(io),
                    };
                    if n_pos > 1 {
                        ubatch.pos[i + cell_count as usize] = ext.y;
                        ubatch.pos[i + 2 * cell_count as usize] = ext.x;
                    }
                    if let Some(ref mut tok) = ubatch.token {
                        tok[i] = ext.tok;
                    }
                    exts[i] = ext;
                }
                // Read the sequence id, but discard it: use dest_seq_id instead.
                let _ = read_i32(io);
                ubatch.pos[i] = pos;
                ubatch.n_seq_id[i] = 1;
                ubatch.seq_id[i] = vec![dest_seq_id];
            }

            if let Some(input) = sinfo_in {
                if input.is_empty()
                    || input.n_stream() != 1
                    || input.idxs[0].len() != cell_count as usize
                {
                    return false;
                }
                *sinfo = input.clone();
                sinfo.s0 = strm;
                sinfo.s1 = strm;
                sinfo.strm[0] = strm;

                let size = self.v_cells[strm_idx].size();
                for i in 0..cell_count as usize {
                    let idx = sinfo.idxs[0][i];
                    if idx >= size || !self.v_cells[strm_idx].is_empty(idx) {
                        return false;
                    }
                }
            } else {
                *sinfo = self.find_slot(&ubatch, false);
                if sinfo.is_empty() {
                    return false;
                }
            }

            // apply_ubatch() rebuilds KvCellExt from the ubatch; put the saved
            // ext back afterwards.
            self.apply_ubatch(sinfo, &ubatch);
            for i in 0..exts.len() {
                self.v_cells[strm_idx].ext_set(sinfo.idxs[0][i], exts[i]);
            }

            debug_assert_eq!(sinfo.n_stream(), 1);
            debug_assert_eq!(sinfo.idxs[0].len(), cell_count as usize);
            for i in 0..cell_count as usize {
                let idx = sinfo.idxs[0][i];
                debug_assert_eq!(self.v_cells[strm_idx].pos_get(idx), ubatch.pos[i]);
                debug_assert!(self.v_cells[strm_idx].seq_has(idx, dest_seq_id));
            }
        } else {
            // whole KV cache restore
            if cell_count > cells_size {
                return false;
            }
            if let Some(input) = sinfo_in {
                if input.is_empty()
                    || input.n_stream() != 1
                    || input.idxs[0].len() != cell_count as usize
                {
                    return false;
                }
            }

            for i in 0..cell_count as usize {
                let pos = read_i32(io);
                let n_seq_id = read_u32(io);
                self.v_cells[strm_idx].pos_set(i as CellIdx, pos);
                if self.has_cell_ext() {
                    let ext = KvCellExt {
                        x: read_i32(io),
                        y: read_i32(io),
                        tok: read_i32(io),
                    };
                    self.v_cells[strm_idx].ext_set(i as CellIdx, ext);
                }
                for _ in 0..n_seq_id {
                    let seq_id = read_i32(io);
                    if seq_id < 0 || seq_id as u32 >= self.n_seq_max {
                        return false;
                    }
                    self.v_cells[strm_idx].seq_add(i as CellIdx, seq_id);
                }
            }

            // Create contiguous slot_info for whole cache restore.
            sinfo.s0 = strm;
            sinfo.s1 = strm;
            sinfo.resize(1);
            sinfo.strm[0] = strm;
            sinfo.idxs[0] = (0..cell_count as CellIdx).collect();
            self.v_heads[strm_idx] = 0;
        }
        true
    }

    fn state_read_data<R: IoRead>(
        &mut self,
        io: &mut R,
        strm: SeqId,
        cell_count: u32,
        sinfo: &SlotInfo,
    ) -> bool {
        // Batch the scatter reads per contiguous run of destination indices.
        struct CellRun {
            from: CellIdx,
            to: CellIdx,
        }
        let mut runs: Vec<CellRun> = Vec::new();
        if cell_count > 0 {
            let idxs = &sinfo.idxs[0];
            let mut i0 = 0usize;
            while i0 < cell_count as usize {
                let mut i1 = i0 + 1;
                while i1 < cell_count as usize && idxs[i1] == idxs[i1 - 1] + 1 {
                    i1 += 1;
                }
                runs.push(CellRun { from: idxs[i0], to: idxs[i1 - 1] + 1 });
                i0 = i1;
            }
        }

        let v_trans = read_u32(io);
        let n_layer = read_u32(io);
        let n_rot_k_ref = read_u32(io);
        let n_rot_v_ref = read_u32(io);

        if n_layer != self.layers.len() as u32 {
            return false;
        }
        if cell_count > self.v_cells[strm as usize].size() {
            return false;
        }
        if self.v_trans != (v_trans != 0) {
            return false;
        }
        if n_rot_k_ref != self.n_rot_k {
            return false;
        }
        if n_rot_v_ref != self.n_rot_v {
            return false;
        }

        let strm_idx = strm as usize;
        for layer in &self.layers {
            let n_embd_k_gqa = self.params.hparams.n_embd_k_gqa(layer.il) as usize;
            let k = &layer.k_stream[strm_idx];
            let k_type_i_ref = read_i32(io);
            if k.ty as i32 != k_type_i_ref {
                return false;
            }
            let k_size_row_ref = read_u64(io);
            let k_size_row = ggml_row_size(k.ty, n_embd_k_gqa) as u64;
            if k_size_row != k_size_row_ref {
                return false;
            }
            for r in &runs {
                if !io.read_tensor(k, r.from as usize * k_size_row as usize, (r.to - r.from) as usize * k_size_row as usize) {
                    return false;
                }
            }
        }

        if !self.v_trans {
            for layer in &self.layers {
                let n_embd_v_gqa = self.params.hparams.n_embd_v_gqa(layer.il) as usize;
                let v = match layer.v_stream[strm_idx].as_ref() {
                    Some(v) => v,
                    None => continue,
                };
                let v_type_i_ref = read_i32(io);
                if v.ty as i32 != v_type_i_ref {
                    return false;
                }
                let v_size_row_ref = read_u64(io);
                let v_size_row = ggml_row_size(v.ty, n_embd_v_gqa) as u64;
                if v_size_row != v_size_row_ref {
                    return false;
                }
                for r in &runs {
                    if !io.read_tensor(v, r.from as usize * v_size_row as usize, (r.to - r.from) as usize * v_size_row as usize) {
                        return false;
                    }
                }
            }
        } else {
            for layer in &self.layers {
                let n_embd_v_gqa = self.params.hparams.n_embd_v_gqa(layer.il) as usize;
                let v = match layer.v_stream[strm_idx].as_ref() {
                    Some(v) => v,
                    None => continue,
                };
                let v_type_i_ref = read_i32(io);
                if v.ty as i32 != v_type_i_ref {
                    return false;
                }
                let v_size_el_ref = read_u32(io);
                let v_size_el = v.ty.type_size();
                if v_size_el != v_size_el_ref as usize {
                    return false;
                }
                let n_embd_v_gqa_ref = read_u32(io);
                if n_embd_v_gqa != n_embd_v_gqa_ref as usize {
                    return false;
                }
                let cells_size = self.v_cells[strm_idx].size() as usize;
                for j in 0..n_embd_v_gqa {
                    for r in &runs {
                        let dst_offset = (r.from as usize + j * cells_size) * v_size_el;
                        if !io.read_tensor(v, dst_offset, (r.to - r.from) as usize * v_size_el) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    pub fn state_clear(&mut self, seq_id: LSeqId) {
        if seq_id == -1 {
            self.clear(true);
            return;
        }
        debug_assert!(seq_id >= 0 && (seq_id as usize) < self.seq_to_stream.len());
        let strm = self.seq_to_stream[seq_id as usize];
        let mut sinfo = SlotInfo::default();
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;

        // A cell that another sequence still uses keeps its data.
        for i in 0..self.v_cells[strm as usize].size() {
            if self.v_cells[strm as usize].seq_has(i, seq_id)
                && self.v_cells[strm as usize].seq_count(i) == 1
            {
                sinfo.idxs[0].push(i);
            }
        }
        self.state_clear_inner(seq_id, strm, &sinfo);
    }

    // The cleared ranges mirror the write pattern of state_read_data().
    fn state_clear_inner(&mut self, seq_id: LSeqId, strm: SeqId, sinfo: &SlotInfo) {
        if seq_id == -1 {
            self.clear(true);
            return;
        }
        self.seq_rm(seq_id, -1, -1);

        // Zero the K/V data of the failed restore attempt; attention can read
        // the data of free cells.
        if sinfo.is_empty() || sinfo.size() == 0 {
            return;
        }

        let strm_idx = strm as usize;
        let cell_count = sinfo.size();
        let is_contiguous = sinfo.is_contiguous();
        let cells_size = self.v_cells[strm_idx].size() as usize;

        for layer in &self.layers {
            let n_embd_k_gqa = self.params.hparams.n_embd_k_gqa(layer.il) as usize;
            let mut k = layer.k_stream[strm_idx].clone();
            let k_size_row = ggml_row_size(k.ty, n_embd_k_gqa);
            if is_contiguous {
                k.zero(sinfo.head() as usize * k_size_row, cell_count * k_size_row);
            } else {
                for i in 0..cell_count {
                    k.zero(sinfo.idxs[0][i] as usize * k_size_row, k_size_row);
                }
            }
        }

        for layer in &self.layers {
            let n_embd_v_gqa = self.params.hparams.n_embd_v_gqa(layer.il) as usize;
            let mut v = match layer.v_stream[strm_idx].clone() {
                Some(v) => v,
                None => continue,
            };
            if !self.v_trans {
                let v_size_row = ggml_row_size(v.ty, n_embd_v_gqa);
                if is_contiguous {
                    v.zero(sinfo.head() as usize * v_size_row, cell_count * v_size_row);
                } else {
                    for i in 0..cell_count {
                        v.zero(sinfo.idxs[0][i] as usize * v_size_row, v_size_row);
                    }
                }
            } else {
                let v_size_el = v.ty.type_size();
                if is_contiguous {
                    let h = sinfo.head() as usize;
                    for j in 0..n_embd_v_gqa {
                        v.zero((h + j * cells_size) * v_size_el, cell_count * v_size_el);
                    }
                } else {
                    for j in 0..n_embd_v_gqa {
                        for i in 0..cell_count {
                            v.zero((sinfo.idxs[0][i] as usize + j * cells_size) * v_size_el, v_size_el);
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Per slice: sequences whose cells form the list and the min position of the
/// cells that are not masked by SWA.
fn get_kv_rows_slices(
    ubatch: &UBatch,
    n_slices: u32,
    n_swa: u32,
    swa_type: SwaType,
) -> Vec<KvRowsSlice> {
    let n_slices = n_slices as usize;
    let mut res = vec![KvRowsSlice::default(); n_slices];

    if n_slices == 1 {
        res[0].seq_ids = ubatch.seq_id_unq[..ubatch.n_seqs_unq.min(ubatch.seq_id_unq.len())].to_vec();
    } else {
        let n_tps = ubatch.n_tokens / n_slices;
        for (s, slice) in res.iter_mut().enumerate() {
            slice.seq_ids = vec![ubatch.seq_id[s * n_tps][0]];
        }
    }

    // Only SWA types that mask more cells as the token position grows.
    if n_swa == 0 || (swa_type != SwaType::Standard && swa_type != SwaType::Chunked) {
        return res;
    }

    let n_tps = ubatch.n_tokens / n_slices;
    for (s, slice) in res.iter_mut().enumerate() {
        let mut p = ubatch.pos[s * n_tps];
        for i in 1..n_tps {
            p = p.min(ubatch.pos[s * n_tps + i]);
        }
        slice.p_min = if swa_type == SwaType::Standard {
            p - n_swa as Pos + 1
        } else {
            (p / n_swa as Pos) * n_swa as Pos
        };
    }
    res
}

#[derive(Clone, Copy)]
struct AttnMaskArgs<'a> {
    hparams: &'a HParams,
    ubatch: &'a UBatch,
    v_cells: &'a [KvCells],
    seq_to_stream: &'a [SeqId],
    n_swa: u32,
    swa_type: SwaType,
    n_kv: usize,
    n_stream: usize,
    n_tps: usize,
    rows: Option<&'a [i32]>,
    block_uniform: bool,
}

fn set_input_attn_mask_impl_f32(args: &AttnMaskArgs<'_>, data: &mut [f32], causal_attn: bool) {
    let ubatch = args.ubatch;
    let hparams = args.hparams;
    let swa = args.swa_type != SwaType::None;
    let alibi = hparams.use_alibi;
    let is_2d = ubatch.is_pos_2d();
    let n_tokens = ubatch.n_tokens;

    // The mask is mostly the same for tokens of one sequence; only cells near
    // the positions of the batch can change. This reuse is incompatible with
    // Alibi position encoding.
    let mut seq_pos_min = vec![Pos::MAX; LLAMA_MAX_SEQ];
    let mut seq_first = vec![n_tokens; LLAMA_MAX_SEQ];
    for i in 0..n_tokens {
        let seq_id = ubatch.seq_id[i][0] as usize;
        seq_pos_min[seq_id] = seq_pos_min[seq_id].min(ubatch.pos[i]);
        seq_first[seq_id] = seq_first[seq_id].min(i);
    }

    for s in 0..args.n_stream {
        let mut seq_srct: BTreeMap<LSeqId, usize> = BTreeMap::new();
        let mut seq_idxs: BTreeMap<LSeqId, Vec<usize>> = BTreeMap::new();

        for ii in 0..args.n_tps {
            let i = s * args.n_tps + ii;
            let seq_id = ubatch.seq_id[i][0];
            let cells = &args.v_cells[args.seq_to_stream[seq_id as usize] as usize];

            // Shared mask rows: every query of the sequence uses its first token.
            let iq = if args.block_uniform { seq_first[seq_id as usize] } else { i };
            let mut p0;
            let p1 = ubatch.pos[iq];
            let p1_x = if is_2d { ubatch.pos[iq + 2 * n_tokens] } else { 0 };
            let p1_y = if is_2d { ubatch.pos[iq + n_tokens] } else { 0 };
            let idst = args.n_kv * i;

            let mut prev = false;
            let idxs = seq_idxs.entry(seq_id).or_default();
            if !alibi {
                if let Some(&srct) = seq_srct.get(&seq_id) {
                    let idst_prev = args.n_kv * srct;
                    data.copy_within(idst_prev..idst_prev + args.n_kv, idst);
                    prev = true;
                } else {
                    idxs.clear();
                    seq_srct.insert(seq_id, i);
                }
            }

            'cols: for jj in 0..args.n_kv {
                let mut c = jj;
                if !alibi && prev {
                    if jj >= idxs.len() {
                        break;
                    }
                    c = idxs[jj];
                }

                let j = match args.rows {
                    Some(rows) => rows[s * args.n_kv + c] as i32,
                    None => c as i32,
                };
                if j < 0 || j as CellIdx >= cells.size() || cells.is_empty(j as CellIdx) {
                    data[idst + c] = f32::NEG_INFINITY;
                    continue 'cols;
                }
                if !cells.seq_has(j as CellIdx, seq_id) {
                    data[idst + c] = f32::NEG_INFINITY;
                    continue 'cols;
                }

                p0 = cells.pos_get(j as CellIdx);
                if !alibi && !prev && p0 + (args.n_swa as Pos + 32) >= seq_pos_min[seq_id as usize] {
                    idxs.push(c);
                }

                if causal_attn {
                    if p0 > p1 {
                        data[idst + c] = f32::NEG_INFINITY;
                        continue 'cols;
                    }
                    if is_2d && p0 == p1 && cells.ext_get(j as CellIdx).is_2d_gt(p1_x, p1_y) {
                        data[idst + c] = f32::NEG_INFINITY;
                        continue 'cols;
                    }
                }

                if swa {
                    let in_span = !causal_attn
                        && hparams.non_causal_type == NonCausalType::SwaFull
                        && p0 >= seq_pos_min[seq_id as usize];
                    if !in_span && HParams::is_masked_swa(args.n_swa, args.swa_type, p0, p1) {
                        data[idst + c] = f32::NEG_INFINITY;
                        continue 'cols;
                    }
                }

                data[idst + c] = if alibi {
                    -((p0 - p1).abs() as f32)
                } else {
                    0.0
                };
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
// llama_kv_cache_context
// ---------------------------------------------------------------------------

pub struct KvCacheContext<'a> {
    status: MemoryStatus,
    kv: Option<&'a mut KvCache>,
    do_shift: bool,
    sc_info: StreamCopyInfo,
    i_cur: usize,
    sinfos: SlotInfoVec,
    ubatches: Vec<UBatch>,
    n_kv: u32,
    n_kv_rows: u32,
}

impl<'a> KvCacheContext<'a> {
    /// Used for errors.
    pub fn failed(status: MemoryStatus) -> Self {
        Self {
            status,
            kv: None,
            do_shift: false,
            sc_info: StreamCopyInfo::default(),
            i_cur: 0,
            sinfos: Vec::new(),
            ubatches: Vec::new(),
            n_kv: 0,
            n_kv_rows: 0,
        }
    }

    /// Used to create a full-cache context.
    pub fn full(kv: &'a mut KvCache) -> Self {
        let n_kv = kv.get_size();
        // Worst case for graph reservation: only a unified cache splits into
        // per-sequence lists.
        let n_stream = kv.get_n_stream();
        let n_kv_rows = if n_stream == 1 { n_kv } else { 0 };

        // A dummy slot info: the data is irrelevant, we just build the graph.
        let mut sinfo = SlotInfo::default();
        sinfo.s0 = 0;
        sinfo.s1 = n_stream - 1;
        sinfo.idxs.resize_with(n_stream as usize, Vec::new);
        for s in 0..n_stream as usize {
            sinfo.strm.push(s as SeqId);
            sinfo.idxs[s].resize(1, 0);
        }

        Self {
            status: MemoryStatus::Success,
            kv: Some(kv),
            do_shift: false,
            sc_info: StreamCopyInfo::default(),
            i_cur: 0,
            sinfos: vec![sinfo],
            ubatches: Vec::new(),
            n_kv,
            n_kv_rows,
        }
    }

    /// Used to create an update context.
    pub fn update(kv: &'a mut KvCache, do_shift: bool, sc_info: StreamCopyInfo) -> Self {
        let status = if !do_shift && sc_info.is_empty() {
            MemoryStatus::NoUpdate
        } else {
            MemoryStatus::Success
        };
        Self {
            status,
            kv: Some(kv),
            do_shift,
            sc_info,
            i_cur: 0,
            sinfos: Vec::new(),
            ubatches: Vec::new(),
            n_kv: 0,
            n_kv_rows: 0,
        }
    }

    /// Used to create a batch processing context from a batch.
    pub fn batch(kv: &'a mut KvCache, sinfos: SlotInfoVec, ubatches: Vec<UBatch>) -> Self {
        Self {
            status: MemoryStatus::Success,
            kv: Some(kv),
            do_shift: false,
            sc_info: StreamCopyInfo::default(),
            i_cur: 0,
            sinfos,
            ubatches,
            n_kv: 0,
            n_kv_rows: 0,
        }
    }

    pub fn next(&mut self) -> bool {
        debug_assert_eq!(self.status, MemoryStatus::Success);
        self.i_cur += 1;
        self.i_cur < self.ubatches.len()
    }

    pub fn apply(&mut self) -> bool {
        debug_assert!(!memory_status_is_fail(self.status));
        let kv = match self.kv.as_deref_mut() {
            Some(kv) => kv,
            None => return false,
        };

        if self.ubatches.is_empty() {
            kv.update(self.do_shift, &self.sc_info);
            return true;
        }

        kv.apply_ubatch(&self.sinfos[self.i_cur], &self.ubatches[self.i_cur]);
        self.n_kv = kv.get_n_kv(&self.sinfos[self.i_cur]);
        self.n_kv_rows = kv.get_n_kv_rows(&self.ubatches[self.i_cur]);
        true
    }

    pub fn get_status(&self) -> MemoryStatus {
        self.status
    }

    pub fn get_ubatch(&self) -> &UBatch {
        debug_assert_eq!(self.status, MemoryStatus::Success);
        &self.ubatches[self.i_cur]
    }

    pub fn get_n_kv(&self) -> u32 {
        self.n_kv
    }

    pub fn get_n_kv_rows(&self) -> u32 {
        self.n_kv_rows
    }

    pub fn type_k(&self) -> GgmlType {
        self.kv.as_ref().unwrap().type_k()
    }

    pub fn type_v(&self) -> GgmlType {
        self.kv.as_ref().unwrap().type_v()
    }

    pub fn get_k(&self, ctx: &mut TensorContext, il: i32) -> Tensor {
        let kv = self.kv.as_ref().unwrap();
        kv.get_k(ctx, il, self.n_kv, &self.sinfos[self.i_cur])
    }

    pub fn get_v(&self, ctx: &mut TensorContext, il: i32) -> Tensor {
        let kv = self.kv.as_ref().unwrap();
        kv.get_v(ctx, il, self.n_kv, &self.sinfos[self.i_cur])
    }

    pub fn get_k_scale(&self, il: i32) -> Option<Tensor> {
        self.kv.as_ref().unwrap().get_k_scale(il)
    }

    pub fn get_v_scale(&self, il: i32) -> Option<Tensor> {
        self.kv.as_ref().unwrap().get_v_scale(il)
    }

    pub fn cpy_k(&self, ctx: &mut TensorContext, k_cur: &Tensor, k_idxs: &[i64], il: i32) -> Tensor {
        self.kv.as_ref().unwrap().cpy_k(ctx, k_cur, k_idxs, il)
    }

    pub fn cpy_v(&self, ctx: &mut TensorContext, v_cur: &Tensor, v_idxs: &[i64], il: i32) -> Tensor {
        self.kv.as_ref().unwrap().cpy_v(ctx, v_cur, v_idxs, il)
    }

    pub fn build_input_k_idxs(&self, ctx: &mut TensorContext, ubatch: &UBatch) -> Tensor {
        self.kv.as_ref().unwrap().build_input_k_idxs(ctx, ubatch)
    }

    pub fn build_input_v_idxs(&self, ctx: &mut TensorContext, ubatch: &UBatch) -> Tensor {
        self.kv.as_ref().unwrap().build_input_v_idxs(ctx, ubatch)
    }

    pub fn get_k_rot(&self) -> Vec<Option<Tensor>> {
        self.kv.as_ref().unwrap().get_k_rot()
    }

    pub fn get_v_rot(&self) -> Vec<Option<Tensor>> {
        self.kv.as_ref().unwrap().get_v_rot()
    }

    pub fn set_input_k_shift(&self, dst: &mut Tensor) {
        self.kv.as_ref().unwrap().set_input_k_shift(dst);
    }

    pub fn set_input_k_idxs(&self, dst: &mut Tensor, ubatch: &UBatch) {
        self.kv.as_ref().unwrap().set_input_k_idxs(dst, ubatch, &self.sinfos[self.i_cur]);
    }

    pub fn set_input_v_idxs(&self, dst: &mut Tensor, ubatch: &UBatch) {
        self.kv.as_ref().unwrap().set_input_v_idxs(dst, ubatch, &self.sinfos[self.i_cur]);
    }

    pub fn set_input_attn_mask(&self, attn_mask: &mut AttnMask, ubatch: &UBatch, causal_attn: bool) {
        self.kv.as_ref().unwrap().set_input_attn_mask(attn_mask, ubatch, causal_attn);
    }

    pub fn set_input_pos_bucket(&self, dst: &mut Tensor, ubatch: &UBatch) {
        self.kv.as_ref().unwrap().set_input_pos_bucket(dst, ubatch);
    }

    pub fn set_input_kpool(
        &self,
        pool_idxs: &mut Tensor,
        pool_pos: &mut Tensor,
        pool_mask: &mut Tensor,
        tail_idxs: &mut Tensor,
        ubatch: &UBatch,
        kpool: u32,
    ) {
        self.kv.as_ref().unwrap().set_input_kpool(
            pool_idxs,
            pool_pos,
            pool_mask,
            tail_idxs,
            ubatch,
            &self.sinfos[self.i_cur],
            kpool,
            self.n_kv,
        );
    }

    pub fn set_input_relative_pos(&self, dst: &mut Tensor, ubatch: &UBatch, extent: u32) {
        self.kv.as_ref().unwrap().set_input_relative_pos(dst, ubatch, extent);
    }

    pub fn get_prev_tokens(&self, ubatch: &UBatch, n: u32, res: &mut Vec<Token>) {
        self.kv.as_ref().unwrap().get_prev_tokens(ubatch, n, res);
    }
}

// ---------------------------------------------------------------------------
// Tests for pure helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ubatch(n_tokens: usize, seq_id: LSeqId) -> UBatch {
        let mut ub = UBatch::ubatch_reserve(n_tokens, 1, true);
        ub.seq_id_unq[0] = seq_id;
        ub.n_seqs_unq = 1;
        for i in 0..n_tokens {
            ub.pos[i] = i as Pos;
            ub.n_seq_id[i] = 1;
            ub.seq_id[i] = vec![seq_id];
            if let Some(ref mut tok) = ub.token {
                tok[i] = i as Token;
            }
        }
        ub
    }

    #[test]
    fn cells_seq_tracking_and_shifting() {
        let mut cells = KvCells::new();
        cells.resize(8);
        cells.pos_set(0, 3);
        cells.seq_add(0, 1);
        cells.pos_set(1, 5);
        cells.seq_add(1, 1);

        assert_eq!(cells.seq_pos_min(1), 3);
        assert_eq!(cells.seq_pos_max(1), 5);
        assert_eq!(cells.seq_pos_tok_le(1, 4), LLAMA_TOKEN_NULL);

        cells.pos_add(0, 2);
        assert_eq!(cells.pos_get(0), 5);
        assert!(cells.get_has_shift());
        assert_eq!(cells.get_shift(0), 2);

        cells.seq_rm_cell(0, 1);
        assert!(cells.is_empty(0));
        assert_eq!(cells.get_used(), 1);
        cells.reset_shift();
        assert_eq!(cells.get_shift(1), 0);
    }

    #[test]
    fn slot_info_contiguity_and_hadamard() {
        let mut sinfo = SlotInfo::default();
        sinfo.resize(1);
        sinfo.idxs[0] = vec![2, 3, 4];
        assert!(sinfo.is_contiguous());
        assert_eq!(sinfo.head(), 2);
        sinfo.idxs[0] = vec![2, 4];
        assert!(!sinfo.is_contiguous());

        let h = ggml_gen_hadamard(4);
        assert_eq!(h[0], 0.5);
        for v in h {
            assert_eq!(v.abs(), 0.5);
        }
    }
}

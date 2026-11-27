//! KV cells, bitsets, sequence tracking, shifts and positions.
use super::*;


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

//! Slot tracking and cache-level sequence operations.
use super::*;


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

impl KvCache {
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

}

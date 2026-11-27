//! Cache mutation, tensor views and the batch context.
use super::*;

impl KvCache {
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

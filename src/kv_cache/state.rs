//! State save/restore and attention/input buffer preparation.
use super::*;
use super::budget::get_kv_rows_slices;

impl KvCache {
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


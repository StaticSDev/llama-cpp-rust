//! Position and memory budgeting helpers.
use super::*;

impl KvCache {
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

}

/// Per slice: sequences whose cells form the list and the min position of the
/// cells that are not masked by SWA.
pub(crate) fn get_kv_rows_slices(
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

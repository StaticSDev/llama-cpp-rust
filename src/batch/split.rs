//! Batch splitting: coupled sequences and ubatch emitters.

use crate::core::SeqId;

use super::{BatchAllocr, SeqSet, Ubatch, LLAMA_MAX_SEQ};

impl BatchAllocr {
    pub(crate) fn detect_coupled_sequences(&mut self, n_tok: usize) {
        // determine coupled sequences
        // these are pairs of sequences that have at least one token in the input batch that is assigned to both of them
        {
            let seq_ids = self.batch.seq_id.as_ref().unwrap();
            let batch_pos = self.batch.pos.as_ref().unwrap();
            for i in 0..n_tok {
                let ids = &seq_ids[i];
                let s0 = ids.first().copied();

                for (s, &s1) in ids.iter().enumerate() {
                    self.seq_pos[s1 as usize].insert(batch_pos[i]);

                    if s > 0 {
                        if let Some(s0) = s0 {
                            // mark that sequence s1 is coupled to s0
                            self.seq_cpl[s1 as usize][s0 as usize] = true;

                            // note: tracking the other way around is not necessary for now
                            //seq_cpl[s0 as usize][s1 as usize] = true;

                            self.has_cpl = true;
                        }
                    }
                }
            }
        }
    }

    // simple split, unknown number of sequence sets of unequal lengths
    pub fn split_simple(&mut self, n_ubatch: u32) -> Ubatch {
        // find the first unused token
        let mut cur_idx = 0usize;
        while cur_idx < self.used.len() && self.used[cur_idx] {
            cur_idx += 1;
        }

        // we are done
        if cur_idx >= self.used.len() {
            return Ubatch::default();
        }

        let mut idxs = Vec::new();

        loop {
            idxs.push(cur_idx);

            self.used[cur_idx] = true;
            self.n_used += 1;

            cur_idx += 1;

            if cur_idx >= self.used.len() {
                break;
            }

            if idxs.len() >= n_ubatch as usize {
                break;
            }
        }

        let n_seqs = idxs.len() as u32;
        self.ubatch_add(&idxs, n_seqs, false)
    }

    // make ubatches of equal-length sequences sets
    // if sequential == true, the tokens in the ubatch will have increasing sequential sequence ids
    // n_keep_tail = minimum trailing tokens of a seq that must land in the same ubatch
    pub fn split_equal(&mut self, n_ubatch: u32, sequential: bool, n_keep_tail: u32) -> Ubatch {
        if sequential && self.has_cpl {
            // sequential split is not supported when there are coupled sequences in the input batch
            return Ubatch::default();
        }

        let n_ubatch_usize = n_ubatch as usize;
        let mut cur_seq_set: Vec<SeqSet> = Vec::new();
        let mut last_seq_id: i64 = -1;

        // determine the non-overlapping sequence sets participating in this ubatch
        {
            let seq_ids = self.batch.seq_id.as_ref().unwrap();
            for i in 0..self.batch.n_tokens {
                if self.used[i] {
                    continue;
                }

                let mut add = true;

                for ss in &cur_seq_set {
                    // no overlap with existing sequence sets:
                    if !ss.intersection(&self.seq_set[i]).none() {
                        add = false;
                        break;
                    }
                }

                let sid0 = seq_ids[i].first().copied().unwrap_or(0) as i64;

                // accept only increasing sequence ids
                if sequential {
                    add = add && (cur_seq_set.is_empty() || sid0 == last_seq_id + 1);
                }

                if add {
                    cur_seq_set.push(self.seq_set[i]);
                    last_seq_id = sid0;

                    if cur_seq_set.len() > n_ubatch_usize {
                        break;
                    }
                }
            }
        }

        let mut n_seqs = cur_seq_set.len();

        // we are done
        if n_seqs == 0 {
            return Ubatch::default();
        }

        // the batch indices of each sequence set
        let seq_idxs: Vec<Vec<usize>> = cur_seq_set
            .iter()
            .map(|ss| self.seq_set_map[ss].clone())
            .collect();

        // the current batch index of each sequence set
        let mut cur_idx = vec![0usize; n_seqs];
        for s in 0..n_seqs {
            while cur_idx[s] < seq_idxs[s].len() && self.used[seq_idxs[s][cur_idx[s]]] {
                cur_idx[s] += 1;
            }
        }

        // the list of batch indices for each sequence set
        // at the end we will concat these to get the final ubatch
        let mut idxs_per_seq: Vec<Vec<usize>> = vec![Vec::new(); n_seqs];

        loop {
            // we can only add new n_seq_tokens tokens if all the sequence sets have at least 1 more unused tokens and
            //   if we haven't reached n_ubatch
            let mut can_expand = true;
            for s in 0..n_seqs {
                if cur_idx[s] >= seq_idxs[s].len() {
                    can_expand = false;
                    break;
                }
            }

            if !can_expand {
                break;
            }

            for s in 0..n_seqs {
                let idx = seq_idxs[s][cur_idx[s]];

                idxs_per_seq[s].push(idx);

                self.used[idx] = true;
                self.n_used += 1;

                cur_idx[s] += 1;
            }

            if (idxs_per_seq[0].len() + 1) * n_seqs > n_ubatch_usize {
                break;
            }
        }

        // if n_keep_tail > 0, keep only the seqs that either finish in this ubatch or have at least
        //   n_keep_tail tokens remaining for a future ubatch, so that the trailing n_keep_tail tokens
        //   of each seq are never split across ubatches
        if n_keep_tail > 0 {
            if n_ubatch <= n_keep_tail {
                return Ubatch::default();
            }
            let n_keep_tail_usize = n_keep_tail as usize;

            // keep the longest prefix of seqs that satisfy the constraint, to preserve sequential seq ids
            let mut n_keep = 0usize;
            while n_keep < n_seqs {
                let remaining = seq_idxs[n_keep].len() - cur_idx[n_keep];

                if remaining != 0 && remaining < n_keep_tail_usize {
                    break;
                }

                n_keep += 1;
            }

            // all seqs violate the constraint - resolve the first one directly and emit it alone
            if n_keep == 0 {
                let idxs = &mut idxs_per_seq[0];

                if idxs.len() + seq_idxs[0].len() - cur_idx[0] <= n_ubatch_usize {
                    // extend the seq to completion
                    while seq_idxs[0].len() - cur_idx[0] > 0 {
                        let idx = seq_idxs[0][cur_idx[0]];

                        idxs.push(idx);

                        self.used[idx] = true;
                        self.n_used += 1;

                        cur_idx[0] += 1;
                    }
                } else {
                    // truncate the seq so that at least n_keep_tail tokens remain
                    while seq_idxs[0].len() - cur_idx[0] < n_keep_tail_usize {
                        let idx = match idxs.pop() {
                            Some(idx) => idx,
                            None => break,
                        };
                        self.used[idx] = false;
                        self.n_used -= 1;

                        cur_idx[0] -= 1;
                    }
                }

                n_keep = 1;
            }

            // return the tokens of the deferred seqs back to the pool
            for s in n_keep..n_seqs {
                for &idx in &idxs_per_seq[s] {
                    self.used[idx] = false;
                    self.n_used -= 1;
                }
            }

            n_seqs = n_keep;
        }

        // concat the per-sequence-set lists
        let mut idxs = Vec::new();
        for s in 0..n_seqs {
            idxs.extend_from_slice(&idxs_per_seq[s]);
        }

        self.ubatch_add(&idxs, n_seqs as u32, true)
    }

    // sequence-set-wise split - each ubatch contains a single sequence-set
    pub fn split_seq(&mut self, n_ubatch: u32) -> Ubatch {
        // find the first unused token
        let mut cur_idx = 0usize;
        while cur_idx < self.used.len() && self.used[cur_idx] {
            cur_idx += 1;
        }

        // we are done
        if cur_idx >= self.used.len() {
            return Ubatch::default();
        }

        // this is the starting sequence set
        // we allow adding tokens only if their sequence set is a subset of the current sequence set
        let mut cur_seq_set = self.seq_set[cur_idx];

        let mut idxs = Vec::new();

        loop {
            idxs.push(cur_idx);

            self.used[cur_idx] = true;
            self.n_used += 1;

            if idxs.len() >= n_ubatch as usize {
                break;
            }

            loop {
                cur_idx += 1;
                if cur_idx >= self.batch.n_tokens {
                    break;
                }
                if !self.used[cur_idx]
                    && cur_seq_set.intersection(&self.seq_set[cur_idx]) == self.seq_set[cur_idx]
                {
                    break;
                }
            }

            if cur_idx >= self.batch.n_tokens {
                break;
            }

            cur_seq_set = self.seq_set[cur_idx];
        }

        self.ubatch_add(&idxs, 1, true)
    }

    // a helper method for creating a well-defined ubatch of tokens
    pub fn ubatch_reserve(&mut self, n_seq_tokens: u32, n_seqs: u32) -> Ubatch {
        let n_tokens = (n_seq_tokens * n_seqs) as usize;

        self.clear();
        self.split_reset();

        let mut res = Ubatch::default();
        res.b_equal_seqs = 1;
        res.n_tokens = n_tokens as u32;
        res.n_seq_tokens = n_seq_tokens;
        res.n_seqs = n_seqs;
        res.n_seqs_unq = n_seqs;
        res.n_pos = self.n_pos_per_embd as u32;

        res.token = vec![0; n_tokens];
        res.pos = vec![0; n_tokens * self.n_pos_per_embd];
        res.n_seq_id = vec![0; n_tokens];
        res.seq_id = vec![Vec::new(); n_tokens];
        res.seq_idx = vec![-1; LLAMA_MAX_SEQ];
        res.output = vec![0; n_tokens];

        for s in 0..n_seqs as usize {
            res.seq_idx[s] = s as i32;
            res.seq_id_unq.push(s as SeqId);
        }

        res
    }
}

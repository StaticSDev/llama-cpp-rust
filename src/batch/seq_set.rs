//! Sequence-set bitset and sequence-consistency checks.

use crate::core::{Error, Result, SeqId};

use super::{BatchAllocr, Memory, LLAMA_MAX_SEQ};

const SEQ_SET_WORDS: usize = LLAMA_MAX_SEQ / 64;

/// `seq_set_t = std::bitset<LLAMA_MAX_SEQ>`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqSet([u64; SEQ_SET_WORDS]);

impl Default for SeqSet {
    fn default() -> Self {
        Self([0; SEQ_SET_WORDS])
    }
}

impl SeqSet {
    pub fn set(&mut self, id: SeqId) {
        let i = id as usize;
        self.0[i / 64] |= 1u64 << (i % 64);
    }

    /// `bitset::set()` with no argument.
    pub fn set_all(&mut self) {
        for w in self.0.iter_mut() {
            *w = u64::MAX;
        }
    }

    pub fn test(&self, id: SeqId) -> bool {
        let i = id as usize;
        self.0[i / 64] & (1u64 << (i % 64)) != 0
    }

    pub fn none(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    /// `std::bitset::operator&`.
    pub fn intersection(&self, other: &Self) -> Self {
        let mut out = *self;
        for (o, b) in out.0.iter_mut().zip(other.0.iter()) {
            *o &= *b;
        }
        out
    }
}

impl BatchAllocr {
    pub(crate) fn build_seq_sets(&mut self, n_tok: usize) {
        // precompute the sequence sets for each token
        {
            let seq_ids = self.batch.seq_id.as_ref().unwrap();
            for i in 0..n_tok {
                let mut cur = SeqSet::default();
                for &sid in &seq_ids[i] {
                    cur.set(sid);
                }

                self.seq_set.push(cur);
                self.seq_set_map.entry(cur).or_default().push(i);
            }
        }
    }

    pub(crate) fn check_sequence_positions(
        &self,
        n_tok: usize,
        mixed: bool,
        n_seq_max: usize,
        mem: Option<&dyn Memory>,
    ) -> Result<()> {
        //
        // consistency checks
        //

        if self.n_pos_per_embd > 1 {
            // in a mixed batch, the first entry of each seq picks the rule
            let mut seq_first_embd = vec![self.batch.token.is_none(); n_seq_max];
            if mixed {
                let mut seen = vec![false; n_seq_max];
                for i in 0..n_tok {
                    let ids = &self.batch.seq_id.as_ref().unwrap()[i];
                    for &sid in ids {
                        let sid = sid as usize;
                        if !seen[sid] {
                            seen[sid] = true;
                            seq_first_embd[sid] = self.is_embd_vec[i] != 0;
                        }
                    }
                }
            }

            // M-RoPE case: allow position to "jump" forward only (non-continuous positions are allowed)
            for s in 0..n_seq_max {
                if self.seq_pos[s].is_empty() {
                    continue;
                }

                let p0 = mem
                    .and_then(|m| m.seq_pos_max(s as SeqId))
                    .unwrap_or(-1);
                let y = self.seq_pos_min(s as SeqId);

                if !seq_first_embd[s] {
                    if p0 >= 0 && p0 >= y {
                        return Err(Error::new(format!(
                            "init: the tokens of sequence {} in the input batch have inconsistent sequence positions:\n - X = {}\n - Y = {}\n for M-RoPE, it is required that the position satisfies: X < Y",
                            s, p0, y
                        )));
                    }
                } else {
                    // embedding inputs can have overlapping positions
                    if p0 >= 0 && p0 > y {
                        return Err(Error::new(format!(
                            "init: the tokens of sequence {} in the input batch have inconsistent sequence positions:\n - X = {}\n - Y = {}\n for M-RoPE, it is required that the position satisfies: X <= Y",
                            s, p0, y
                        )));
                    }
                }
            }
        } else {
            for s in 0..n_seq_max {
                if self.seq_pos[s].is_empty() {
                    continue;
                }

                let p0 = mem
                    .and_then(|m| m.seq_pos_max(s as SeqId))
                    .unwrap_or(-1);
                let lo = self.seq_pos_min(s as SeqId);
                let hi = self.seq_pos_max(s as SeqId);

                if p0 >= 0 {
                    if lo != p0 + 1 {
                        return Err(Error::new(format!(
                            "init: the tokens of sequence {} in the input batch have inconsistent sequence positions:\n - X = {}\n - Y = {}\n it is required that the sequence positions remain consecutive: Y = X + 1",
                            s, p0, lo
                        )));
                    }
                }

                if hi - lo + 1 > self.seq_pos[s].len() as i32 {
                    return Err(Error::new(format!(
                        "init: sequence {} positions are not continuous",
                        s
                    )));
                }
            }
        }

        if let Some(mem) = mem {
            for s0 in 0..n_seq_max {
                for s1 in 0..n_seq_max {
                    if self.seq_cpl[s0][s1] {
                        let a = (
                            mem.seq_pos_min(s0 as SeqId),
                            mem.seq_pos_max(s0 as SeqId),
                        );
                        let b = (
                            mem.seq_pos_min(s1 as SeqId),
                            mem.seq_pos_max(s1 as SeqId),
                        );
                        if a != b {
                            return Err(Error::new(format!(
                                "init: sequence {} is coupled to {} in the input batch, but have divereged",
                                s0, s1
                            )));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn check_sequence_sets(&self, n_tok: usize, n_seq_max: usize) -> Result<()> {
        // disallow partial sequence sub-sets:
        //
        // invalid:          x
        //            i: 0 1 2 ...
        // ---------------------------------------
        // seq_id[i][0]: 0 0 1
        // seq_id[i][1]: 1 1 2
        // seq_id[i][2]: 2
        //
        // disallow decreasing sequence positions:
        //
        // invalid:                  x
        //            i: 0 1 2 3 4 5 6 ...
        // ---------------------------------------
        //       pos[i]: 4 5 0 1 6 2 3
        // seq_id[i][0]: 0 0 1 1 0 1 0
        //
        {
            let mut cur_seq_set = vec![SeqSet::default(); n_seq_max];
            for s in 0..n_seq_max {
                cur_seq_set[s].set_all();
            }
            let mut cur_seq_pos = vec![-1i32; n_seq_max];

            let batch_pos = self.batch.pos.as_ref().unwrap();
            let seq_ids = self.batch.seq_id.as_ref().unwrap();
            for i in 0..n_tok {
                let p = batch_pos[i];
                for &sid in &seq_ids[i] {
                    let sid = sid as usize;
                    cur_seq_set[sid] = cur_seq_set[sid].intersection(&self.seq_set[i]);

                    if cur_seq_set[sid].none() {
                        return Err(Error::new(format!(
                            "init: sequence {} belongs to incompatible sequence sets (not allowed)",
                            sid
                        )));
                    }

                    if p < cur_seq_pos[sid] {
                        return Err(Error::new(format!(
                            "init: sequence {} positions are decreasing (not allowed)",
                            sid
                        )));
                    }

                    cur_seq_pos[sid] = p;
                }
            }
        }

        Ok(())
    }
}

//! Owned ubatch representation and the in-memory position trait.

use std::collections::{BTreeSet, HashMap};

use crate::core::{SeqId, Token};

use super::{Batch, Pos, SeqSet, LLAMA_MAX_SEQ};

/// Stands in for `llama_memory_i`: positions already stored in the cache.
/// `None` means no position is stored (C++ returns -1).
pub trait Memory {
    fn seq_pos_min(&self, seq: SeqId) -> Option<Pos>;
    fn seq_pos_max(&self, seq: SeqId) -> Option<Pos>;
}

/// Minimal in-memory `Memory` implementation, useful for tests and cache-less runs.
#[derive(Default)]
pub struct SimpleMemory {
    pos: HashMap<SeqId, (Pos, Pos)>,
}

impl SimpleMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, seq: SeqId, min: Pos, max: Pos) {
        self.pos.insert(seq, (min, max));
    }
}

impl Memory for SimpleMemory {
    fn seq_pos_min(&self, seq: SeqId) -> Option<Pos> {
        self.pos.get(&seq).map(|p| p.0)
    }

    fn seq_pos_max(&self, seq: SeqId) -> Option<Pos> {
        self.pos.get(&seq).map(|p| p.1)
    }
}

/// A single uniform batch. Every array is owned; see the module comment.
#[derive(Clone, Default)]
pub struct Ubatch {
    pub b_equal_seqs: u32, // note: this is a boolean, but we use an int32_t for alignment
    pub n_tokens: u32,     // total tokens (n_seq_tokens * n_seqs)
    pub n_seq_tokens: u32, // tokens per sequence set
    pub n_seqs: u32,       // sequence sets in the ubatch
    pub n_seqs_unq: u32,   // unique sequence ids in the ubatch
    pub n_pos: u32,        // number of position inputs for each token/embedding

    // seq_id_unq: unique sequence ids in the ubatch
    // seq_idx:    indices of the unique sequence ids in the ubatch in [0, n_seqs_unq)
    //             used for extracting sequence pooled embeddings
    pub token: Vec<Token>,       // [n_tokens]
    pub embd: Vec<f32>,          // [n_embd, n_tokens]
    pub embd_state: Vec<f32>,    // [n_embd_state, n_tokens]
    pub embd_row: Vec<i32>,      // [n_tokens]
    pub embd_state_row: Vec<i32>, // state row in ctx_state_src's h_nextn buffer
    pub pos: Vec<Pos>,           // [n_tokens*n_pos]
    pub n_seq_id: Vec<i32>,      // [n_tokens]
    pub seq_id: Vec<Vec<SeqId>>, // [n_tokens] -> [s0, s1, ...]
    pub seq_id_unq: Vec<SeqId>,  // [n_seqs_unq]
    pub seq_idx: Vec<i32>,       // [LLAMA_MAX_SEQ]
    pub output: Vec<i8>,         // [n_tokens]
    pub type_: Vec<i8>,          // [n_tokens], mixed ubatch only (0 - token, 1 - embd)
    pub decision_order: Vec<i32>, // [n_tokens], empty if no entry has one
    pub decision_group: Vec<i32>, // [n_tokens], empty if no entry has one
    pub batch_idxs: Vec<usize>,   // original batch index for each token
}

impl Ubatch {
    pub fn equal_seqs(&self) -> bool {
        self.b_equal_seqs != 0
    }

    // typical for M-RoPE cases:
    //   0 - sequential position of the tokens/embeddings in the sequence
    //   1 - y position in the image
    //   2 - x position in the image
    //   3 - other
    pub fn is_pos_2d(&self) -> bool {
        // TODO @ngxson : we may need to check for model arch when more models use >1 positions
        self.n_pos >= 3
    }

    // mixed: type picks token or embd per row, pos has n_pos sections for all rows
    pub fn is_mixed(&self) -> bool {
        !self.type_.is_empty()
    }
}

/// A helper for sanitizing, fulfilling and splitting a batch.
pub struct BatchAllocr {
    pub(crate) n_pos_per_embd: usize,
    pub(crate) allow_mixed: bool,

    pub(crate) batch: Batch,
    pub(crate) n_embd: usize,
    pub(crate) n_embd_state: usize,
    pub(crate) n_seq_max: SeqId,
    pub(crate) n_outputs: u32,

    pub(crate) is_embd_vec: Vec<i8>,  // mixed batch only (= 1 if embd, 0 if text token)
    pub(crate) state_vec: Vec<f32>,   // llama_batch has no slot for them
    pub(crate) embd_row_vec: Vec<i32>,
    pub(crate) state_row_vec: Vec<i32>,
    pub(crate) decision_order: Vec<i32>, // empty if no entry has one
    pub(crate) decision_group: Vec<i32>, // empty if no entry has one

    // helper flag to quickly determine if there are any coupled sequences
    pub(crate) has_cpl: bool,

    // seq_pos[s]: the set of positions in sequence s
    pub(crate) seq_pos: Vec<BTreeSet<Pos>>,
    // seq_cpl[s0][s1]: if sequence s0 is coupled to sequence s1
    pub(crate) seq_cpl: Vec<Vec<bool>>,

    // seq_set[i]: the sequence set of token i
    pub(crate) seq_set: Vec<SeqSet>,
    // the indices at which the sequence set appears
    pub(crate) seq_set_map: HashMap<SeqSet, Vec<usize>>,

    // batch indices of the output
    pub(crate) out_ids: Vec<usize>,
    pub(crate) n_used: u32,
    // used[i] indicates if token i has already been used in a previous ubatch
    pub(crate) used: Vec<bool>,
}

impl BatchAllocr {
    pub fn new(n_pos_per_embd: usize, allow_mixed: bool) -> Self {
        Self {
            n_pos_per_embd,
            allow_mixed,
            batch: Batch::default(),
            n_embd: 0,
            n_embd_state: 0,
            n_seq_max: 0,
            n_outputs: 0,
            is_embd_vec: Vec::new(),
            state_vec: Vec::new(),
            embd_row_vec: Vec::new(),
            state_row_vec: Vec::new(),
            decision_order: Vec::new(),
            decision_group: Vec::new(),
            has_cpl: false,
            seq_pos: vec![BTreeSet::new(); LLAMA_MAX_SEQ],
            seq_cpl: vec![vec![false; LLAMA_MAX_SEQ]; LLAMA_MAX_SEQ],
            seq_set: Vec::new(),
            seq_set_map: HashMap::new(),
            out_ids: Vec::new(),
            n_used: 0,
            used: Vec::new(),
        }
    }

    pub fn get_batch(&self) -> &Batch {
        &self.batch
    }

    pub fn get_n_tokens(&self) -> usize {
        self.batch.n_tokens
    }

    pub fn get_n_outputs(&self) -> u32 {
        self.n_outputs
    }

    pub fn get_n_used(&self) -> u32 {
        self.n_used
    }

    // the array of output indices in the order they were encountered during the ubatch splitting
    pub fn get_out_ids(&self) -> &[usize] {
        &self.out_ids
    }

    // min/max positions of each sequence in the current ubatch
    pub fn seq_pos_min(&self, seq_id: SeqId) -> Pos {
        self.seq_pos[seq_id as usize]
            .iter()
            .next()
            .copied()
            .unwrap_or(-1)
    }

    pub fn seq_pos_max(&self, seq_id: SeqId) -> Pos {
        self.seq_pos[seq_id as usize]
            .iter()
            .next_back()
            .copied()
            .unwrap_or(-1)
    }

    // call once before splitting the batch to reset the internal state
    pub fn split_reset(&mut self) {
        self.out_ids.clear();
        self.n_used = 0;
        self.used = vec![false; self.get_n_tokens()];
    }

    pub(crate) fn clear(&mut self) {
        self.n_outputs = 0;

        self.batch = Batch::default();

        self.is_embd_vec.clear();
        self.state_vec.clear();
        self.embd_row_vec.clear();
        self.state_row_vec.clear();
        self.decision_order.clear();
        self.decision_group.clear();

        for cur in self.seq_pos.iter_mut() {
            cur.clear();
        }

        for cur in self.seq_cpl.iter_mut() {
            cur.fill(false);
        }

        self.has_cpl = false;
        self.seq_set.clear();
        self.seq_set_map.clear();
    }

    // create the next ubatch based on the provided batch indices (idxs) and the number of sequence sets (n_seqs)
    // returns n_tokens == 0 if the divisor is invalid
    pub(crate) fn ubatch_add(&mut self, idxs: &[usize], n_seqs: u32, equal_seqs: bool) -> Ubatch {
        let n_tokens = idxs.len();

        if n_seqs == 0 || n_tokens % n_seqs as usize != 0 {
            return Ubatch::default();
        }

        let mut udata = Ubatch::default();
        let mixed_batch = !self.is_embd_vec.is_empty();

        // a ubatch with a single kind of rows is emitted as a plain token or embd ubatch
        let mut n_embd_rows = 0usize;
        if mixed_batch {
            for &idx in idxs {
                n_embd_rows += self.is_embd_vec[idx] as usize;
            }
        }
        let mixed = mixed_batch && n_embd_rows > 0 && n_embd_rows < n_tokens;
        let use_token = self.batch.token.is_some() && !(mixed_batch && n_embd_rows == n_tokens);
        let use_embd = self.batch.embd.is_some() && !(mixed_batch && n_embd_rows == 0);
        let has_state = !self.state_vec.is_empty();
        let has_row = !self.embd_row_vec.is_empty();
        let has_srow = !self.state_row_vec.is_empty();

        let n_embd_all = if use_embd { n_tokens * self.n_embd } else { 0 };
        let n_state_all = if has_state {
            n_tokens * self.n_embd_state
        } else {
            0
        };
        let n_pos_all = n_tokens * self.n_pos_per_embd;

        udata.b_equal_seqs = equal_seqs as u32;
        udata.n_tokens = n_tokens as u32;
        udata.n_seq_tokens = (n_tokens / n_seqs as usize) as u32;
        udata.n_seqs = n_seqs;
        udata.n_pos = self.n_pos_per_embd as u32;

        udata.token = vec![0; if use_token { n_tokens } else { 0 }];
        udata.embd = vec![0.0; n_embd_all];
        udata.embd_state = vec![0.0; n_state_all];
        udata.embd_row = vec![0; if has_row { n_tokens } else { 0 }];
        udata.embd_state_row = vec![0; if has_srow { n_tokens } else { 0 }];
        udata.pos = vec![0; n_pos_all];
        udata.n_seq_id = vec![0; n_tokens];
        udata.seq_id = Vec::with_capacity(n_tokens);
        udata.seq_id_unq = Vec::new();
        udata.seq_idx = vec![-1; LLAMA_MAX_SEQ];
        udata.output = vec![0; n_tokens];
        udata.type_ = vec![0; if mixed { n_tokens } else { 0 }];
        udata.decision_order = vec![0; if self.decision_order.is_empty() { 0 } else { n_tokens }];
        udata.decision_group = vec![0; if self.decision_group.is_empty() { 0 } else { n_tokens }];
        udata.batch_idxs = idxs.to_vec();

        let mut seq_set_unq = SeqSet::default();

        for (i, &idx) in idxs.iter().enumerate() {
            if use_token {
                udata.token[i] = self.batch.token.as_ref().unwrap()[idx];
            }

            if use_embd {
                let n = self.n_embd;
                udata.embd[i * n..(i + 1) * n]
                    .copy_from_slice(&self.batch.embd.as_ref().unwrap()[idx * n..(idx + 1) * n]);
            }

            if mixed {
                udata.type_[i] = self.is_embd_vec[idx];
            }

            if has_state {
                let n = self.n_embd_state;
                udata.embd_state[i * n..(i + 1) * n]
                    .copy_from_slice(&self.state_vec[idx * n..(idx + 1) * n]);
            }

            if has_row {
                udata.embd_row[i] = self.embd_row_vec[idx];
            }

            if has_srow {
                udata.embd_state_row[i] = self.state_row_vec[idx];
            }

            let batch_pos = self.batch.pos.as_ref().unwrap();
            for j in 0..self.n_pos_per_embd {
                udata.pos[j * n_tokens + i] = batch_pos[j * self.batch.n_tokens + idx];
            }

            let batch_seq_id = self.batch.seq_id.as_ref().unwrap();
            udata.n_seq_id[i] = batch_seq_id[idx].len() as i32;
            udata.output[i] = self.batch.logits.as_ref().unwrap()[idx];

            if !self.decision_order.is_empty() {
                udata.decision_order[i] = self.decision_order[idx];
            }
            if !self.decision_group.is_empty() {
                udata.decision_group[i] = self.decision_group[idx];
            }

            let mut ids = Vec::with_capacity(batch_seq_id[idx].len());
            for &sid in &batch_seq_id[idx] {
                ids.push(sid);
                seq_set_unq.set(sid);
            }
            udata.seq_id.push(ids);

            if udata.output[i] != 0 {
                self.out_ids.push(idx);
            }
        }

        for s in 0..self.n_seq_max as usize {
            if seq_set_unq.test(s as SeqId) {
                udata.seq_idx[s] = udata.seq_id_unq.len() as i32;
                udata.seq_id_unq.push(s as SeqId);
            }
        }
        udata.n_seqs_unq = udata.seq_id_unq.len() as u32;

        udata
    }
}

//! Port of llama.cpp `llama-batch.h` / `llama-batch.cpp`.
//!
//! `llama_ubatch` owns its arrays directly. In C++ the struct holds raw
//! pointers into a shared `data_t`; the safe-Rust equivalent is to keep the
//! vectors in the struct itself. Algorithm order and layout are unchanged.
//!
//! Debug-only logging (`ubatch_print`, `LLAMA_BATCH_DEBUG`) and the malloc-based
//! C entry points are not part of this port.

use std::collections::{BTreeSet, HashMap};

use crate::core::{Error, Result, SeqId, Token, LLAMA_TOKEN_NULL};

/// `llama_pos`.
pub type Pos = i32;

pub const LLAMA_MAX_SEQ: usize = 512;
pub const GGML_MROPE_SECTIONS: usize = 4;

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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ContextType {
    Default,
    Mtp,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Dflash,
    Other,
}

/// Only the fields consulted by the two selection helpers.
#[derive(Clone, Copy, Default)]
pub struct HParams {
    pub n_embd: usize,
    pub n_embd_inp: usize,
    pub n_embd_inp_enc: usize,
    pub n_embd_out: usize,
    pub lilicorr_candidate_topk: i32,
    pub dspark_f_rnn: bool,
}

impl HParams {
    pub fn n_embd_inp(&self) -> usize {
        self.n_embd_inp
    }

    pub fn n_embd_inp_enc(&self) -> usize {
        self.n_embd_inp_enc
    }
}

// DFlash batches carry the fused target features at the encoder input width (n_embd_inp_enc size).
// Other batches carry token embeddings (n_embd_inp size).
pub fn select_n_embd_inp(ctx_type: ContextType, arch: Arch, hparams: &HParams) -> usize {
    let _ = ctx_type; // GGML_UNUSED(ctx_type)
    if arch == Arch::Dflash {
        return hparams.n_embd_inp_enc();
    }
    hparams.n_embd_inp()
}

// MTP contexts also take the target model's hidden state (n_embd_out size), 0 = no state input.
pub fn select_n_embd_state(ctx_type: ContextType, hparams: &HParams) -> usize {
    if ctx_type == ContextType::Mtp {
        return hparams.n_embd_out;
    }
    // TODO: this probably should become a function of `llm_arch`
    if hparams.lilicorr_candidate_topk > 0 {
        return hparams.n_embd_inp();
    }
    if hparams.dspark_f_rnn {
        // Inkling2 DSpark rnn head: f_rnn on the anchor row, [uniform | 1/temp] on the draft rows
        return hparams.n_embd;
    }
    0
}

/// The legacy `llama_batch`. `None` stands for the null pointer of each array.
#[derive(Clone, Default)]
pub struct Batch {
    pub n_tokens: usize,
    pub token: Option<Vec<Token>>,
    pub embd: Option<Vec<f32>>,
    pub pos: Option<Vec<Pos>>,
    pub n_seq_id: Option<Vec<i32>>,
    pub seq_id: Option<Vec<Vec<SeqId>>>,
    pub logits: Option<Vec<i8>>,
}

impl Batch {
    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }

    pub fn has_embd(&self) -> bool {
        self.embd.is_some()
    }
}

pub fn batch_get_one(tokens: &[Token]) -> Batch {
    Batch {
        n_tokens: tokens.len(),
        token: Some(tokens.to_vec()),
        ..Default::default()
    }
}

/// `llama_embd`: one or more rows of floats.
#[derive(Clone, Copy)]
pub struct Embd<'a> {
    pub data: &'a [f32],
    pub n_rows: usize,
    pub n_embd: usize,
}

impl<'a> Embd<'a> {
    pub fn n_total(&self) -> usize {
        self.n_rows * self.n_embd
    }
}

/// One entry of `llama_batch_ext`.
#[derive(Clone)]
pub struct TokenEntry {
    pub id: Token,
    pub has_embd: bool, // whether embd_off is set
    pub embd_off: usize,
    pub has_state: bool, // whether state_off is set
    pub state_off: usize,
    pub embd_row: i32, // -1 if none
    pub state_row: i32, // -1 if none
    pub output: bool,
    pub decision_order: i32,
    pub decision_group: i32,
    pub seq_ids: BTreeSet<SeqId>,
    pub pos: [Pos; GGML_MROPE_SECTIONS],
}

impl Default for TokenEntry {
    fn default() -> Self {
        Self {
            id: LLAMA_TOKEN_NULL,
            has_embd: false,
            embd_off: 0,
            has_state: false,
            state_off: 0,
            embd_row: -1,
            state_row: -1,
            output: false,
            decision_order: 0,
            decision_group: 0,
            seq_ids: BTreeSet::new(),
            pos: [0; GGML_MROPE_SECTIONS],
        }
    }
}

pub struct BatchExt {
    pub n_tokens_max: usize,
    pub n_embd_inp: usize,
    pub n_embd_inp_enc: usize,
    pub n_embd_state: usize,
    pub n_seq_max: SeqId,
    pub mem: Option<Box<dyn Memory>>,
    pub n_vocab: Token,
    pub n_pos_per_embd: usize,

    // actual embd row width, set by the first set_token_embd()
    pub n_embd: usize,

    pub tokens: Vec<TokenEntry>,
    pub embd: Vec<f32>,
    pub state: Vec<f32>,
}

impl BatchExt {
    pub fn new(
        n_tokens_max: usize,
        n_embd_inp: usize,
        n_embd_inp_enc: usize,
        n_embd_state: usize,
        n_seq_max: SeqId,
        mem: Option<Box<dyn Memory>>,
        n_vocab: Token,
        n_pos_per_embd: usize,
    ) -> Self {
        let mut res = Self {
            n_tokens_max,
            n_embd_inp,
            n_embd_inp_enc,
            n_embd_state,
            n_seq_max,
            mem,
            n_vocab,
            n_pos_per_embd,
            n_embd: 0,
            tokens: Vec::new(),
            embd: Vec::new(),
            state: Vec::new(),
        };
        res.clear();
        res
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.embd.clear();
        self.state.clear();
        self.n_embd = 0;
    }

    // add an entry with an undefined position
    // the caller must set it explicitly via set_token_pos()
    // -1: size limit reached, -3: invalid sequence id
    pub fn add_token(&mut self, seq_id: SeqId) -> i32 {
        if self.tokens.len() >= self.n_tokens_max {
            return -1;
        }
        if seq_id >= self.n_seq_max {
            return -3;
        }

        let mut t = TokenEntry::default();
        t.seq_ids.insert(seq_id);
        self.tokens.push(t);

        (self.tokens.len() - 1) as i32
    }

    pub fn add_seq(&mut self, idx: i32, seq_id: SeqId) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        if seq_id >= self.n_seq_max {
            return false;
        }

        self.tokens[idx as usize].seq_ids.insert(seq_id);
        true
    }

    pub fn set_token_id(&mut self, idx: i32, id: Token) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        if id < 0 || id >= self.n_vocab {
            return false;
        }
        self.tokens[idx as usize].id = id;
        true
    }

    pub fn set_token_embd(&mut self, idx: i32, embd_in: Embd<'_>) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        if embd_in.data.is_empty() {
            return false;
        }

        let n_total = embd_in.n_total();
        if self.n_embd == 0 {
            if n_total != self.n_embd_inp && n_total != self.n_embd_inp_enc {
                return false;
            }
            self.n_embd = n_total;
        } else if n_total != self.n_embd {
            return false;
        }

        let t = &mut self.tokens[idx as usize];
        if t.has_embd {
            return false;
        }

        t.has_embd = true;
        t.embd_off = self.embd.len();
        self.embd.extend_from_slice(&embd_in.data);
        true
    }

    pub fn set_token_state(&mut self, idx: i32, state_in: Embd<'_>) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        if state_in.data.is_empty() {
            return false;
        }
        if self.n_embd_state == 0 {
            return false; // this context does not take state embeddings
        }

        let n_total = state_in.n_total();
        if n_total != self.n_embd_state {
            return false;
        }

        let t = &mut self.tokens[idx as usize];
        if t.has_state {
            return false;
        }

        t.has_state = true;
        t.state_off = self.state.len();
        self.state.extend_from_slice(&state_in.data);
        true
    }

    pub fn set_token_embd_row(&mut self, idx: i32, row: i32) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        if row < 0 {
            return false;
        }

        let t = &mut self.tokens[idx as usize];
        if t.id != LLAMA_TOKEN_NULL || t.has_embd || t.embd_row >= 0 {
            return false;
        }

        t.embd_row = row;
        true
    }

    pub fn set_token_state_row(&mut self, idx: i32, row: i32) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() || row < 0 {
            return false;
        }

        let t = &mut self.tokens[idx as usize];
        if t.has_state || t.state_row >= 0 {
            return false;
        }

        t.state_row = row;
        true
    }

    pub fn set_token_pos(&mut self, idx: i32, pos_in: &[Pos]) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }

        let t = &self.tokens[idx as usize];
        let n_pos = if t.id != LLAMA_TOKEN_NULL {
            1
        } else {
            self.n_pos_per_embd
        };
        if pos_in.len() < n_pos {
            return false;
        }

        self.tokens[idx as usize].pos[..n_pos].copy_from_slice(&pos_in[..n_pos]);
        true
    }

    pub fn set_output(&mut self, idx: i32, output_last: bool) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        self.tokens[idx as usize].output = output_last;
        true
    }

    pub fn set_decision_order(&mut self, idx: i32, order: i32) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        self.tokens[idx as usize].decision_order = order;
        true
    }

    pub fn set_decision_group(&mut self, idx: i32, group: i32) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() || group < 0 {
            return false;
        }
        self.tokens[idx as usize].decision_group = group;
        true
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
    n_pos_per_embd: usize,
    allow_mixed: bool,

    batch: Batch,
    n_embd: usize,
    n_embd_state: usize,
    n_seq_max: SeqId,
    n_outputs: u32,

    is_embd_vec: Vec<i8>,  // mixed batch only (= 1 if embd, 0 if text token)
    state_vec: Vec<f32>,   // llama_batch has no slot for them
    embd_row_vec: Vec<i32>,
    state_row_vec: Vec<i32>,
    decision_order: Vec<i32>, // empty if no entry has one
    decision_group: Vec<i32>, // empty if no entry has one

    // helper flag to quickly determine if there are any coupled sequences
    has_cpl: bool,

    // seq_pos[s]: the set of positions in sequence s
    seq_pos: Vec<BTreeSet<Pos>>,
    // seq_cpl[s0][s1]: if sequence s0 is coupled to sequence s1
    seq_cpl: Vec<Vec<bool>>,

    // seq_set[i]: the sequence set of token i
    seq_set: Vec<SeqSet>,
    // the indices at which the sequence set appears
    seq_set_map: HashMap<SeqSet, Vec<usize>>,

    // batch indices of the output
    out_ids: Vec<usize>,
    n_used: u32,
    // used[i] indicates if token i has already been used in a previous ubatch
    used: Vec<bool>,
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

    // convert a BatchExt to the internal Batch and sanitize it
    pub fn init(&mut self, batch_inp: &BatchExt, output_all: bool) -> Result<()> {
        self.clear();

        self.n_embd = if batch_inp.n_embd > 0 {
            batch_inp.n_embd
        } else {
            batch_inp.n_embd_inp
        };
        self.n_embd_state = batch_inp.n_embd_state;
        self.n_seq_max = batch_inp.n_seq_max;

        let n_tok = batch_inp.tokens.len();
        if n_tok == 0 {
            return Err(Error::new("init: empty batch"));
        }
        let n_seq_max = self.n_seq_max as usize;
        if n_seq_max > LLAMA_MAX_SEQ {
            return Err(Error::new(format!(
                "init: n_seq_max = {} > {}",
                self.n_seq_max, LLAMA_MAX_SEQ
            )));
        }

        //
        // determine the content types of the batch
        // an entry can carry a token id, a token embedding, or both
        // all entries must carry the same combination, or be a mix of token and embd entries
        // a state embedding (e.g. MTP hook batches) is set on all entries or on none
        // an embd row (embd read from a device buffer) is set on all entries or on none, with no token id or embedding
        //

        let mut n_tok_only = 0usize;
        let mut n_embd_only = 0usize;
        let mut n_both = 0usize;

        let has_state = batch_inp.tokens[0].has_state;
        let has_row = batch_inp.tokens[0].embd_row >= 0;
        let has_state_row = batch_inp.tokens[0].state_row >= 0;

        if has_state && has_state_row {
            return Err(Error::new(
                "init: the entries have a state embedding and a state row",
            ));
        }

        for (i, tok) in batch_inp.tokens.iter().enumerate() {
            let is_tok = tok.id != LLAMA_TOKEN_NULL;
            let is_emb = tok.has_embd;
            let is_row = tok.embd_row >= 0;

            if is_row != has_row {
                return Err(Error::new(
                    "init: all entries in the batch must have the same embd row presence",
                ));
            }

            if is_row {
                if is_tok || is_emb {
                    return Err(Error::new(format!(
                        "init: entry {} has an embd row and a token id or embedding",
                        i
                    )));
                }
            } else if !is_tok && !is_emb {
                return Err(Error::new(format!(
                    "init: entry {} has neither a token id nor an embedding",
                    i
                )));
            }

            if tok.has_state != has_state {
                return Err(Error::new(
                    "init: all entries in the batch must have the same state embedding presence",
                ));
            }

            if (tok.state_row >= 0) != has_state_row {
                return Err(Error::new(
                    "init: all entries in the batch must have the same state row presence",
                ));
            }

            n_tok_only += (is_tok && !is_emb) as usize;
            n_embd_only += (is_emb && !is_tok) as usize;
            n_both += (is_tok && is_emb) as usize;
        }

        if n_both > 0 && n_both != n_tok {
            return Err(Error::new(
                "init: entries with both a token id and an embedding cannot be mixed with other entries",
            ));
        }

        let mixed = n_tok_only > 0 && n_embd_only > 0;

        if mixed && !self.allow_mixed {
            return Err(Error::new(
                "init: this model or context does not support batches mixing token and embedding entries",
            ));
        }

        let has_token = n_tok_only > 0 || n_both > 0;
        let has_embd = n_embd_only > 0 || n_both > 0;

        let mut is_embd_vec = Vec::new();
        if mixed {
            is_embd_vec = batch_inp
                .tokens
                .iter()
                .map(|t| t.has_embd as i8)
                .collect();
        }

        //
        // build flat token/embd array
        //

        let mut token_vec = Vec::with_capacity(n_tok);
        if has_token {
            for (i, tok) in batch_inp.tokens.iter().enumerate() {
                if mixed && is_embd_vec[i] != 0 {
                    token_vec.push(0); // placeholder
                    continue;
                }
                let id = tok.id;
                if id < 0 || id >= batch_inp.n_vocab {
                    return Err(Error::new(format!("init: invalid token[{}] = {}", i, id)));
                }
                token_vec.push(id);
            }
        }

        let mut embd_vec = Vec::new();
        if mixed {
            embd_vec.resize(n_tok * self.n_embd, 0.0f32);
            for (i, tok) in batch_inp.tokens.iter().enumerate() {
                if is_embd_vec[i] != 0 {
                    let off = tok.embd_off;
                    let src = &batch_inp.embd[off..off + self.n_embd];
                    embd_vec[i * self.n_embd..(i + 1) * self.n_embd].copy_from_slice(src);
                }
            }
        } else if has_embd {
            embd_vec = batch_inp.embd.clone();
        }

        let state_vec = if has_state {
            batch_inp.state.clone()
        } else {
            Vec::new()
        };

        let mut embd_row_vec = Vec::new();
        if has_row {
            embd_row_vec = batch_inp.tokens.iter().map(|t| t.embd_row).collect();
        }

        let mut state_row_vec = Vec::new();
        if has_state_row {
            state_row_vec = batch_inp.tokens.iter().map(|t| t.state_row).collect();
        }

        //
        // build flat pos array, section-major: pos[j*n_tok + i] = section j of entry i
        // token entry: [p, p, p, 0] (M-RoPE text position)
        // embd entry:  tokens[i].pos as-is
        //

        let mut pos = vec![0i32; n_tok * self.n_pos_per_embd];
        for (i, tok) in batch_inp.tokens.iter().enumerate() {
            let expand = tok.id != LLAMA_TOKEN_NULL;
            for j in 0..self.n_pos_per_embd {
                let mut p = tok.pos[j];
                if expand {
                    // expand [p] to [p, p, p, 0] for M-RoPE
                    p = if j < 3 { tok.pos[0] } else { 0 };
                }
                pos[j * n_tok + i] = p;
            }
        }

        //
        // build n_seq_id / seq_id arrays
        //

        let mut seq_id: Vec<Vec<SeqId>> = Vec::with_capacity(n_tok);
        for tok in &batch_inp.tokens {
            for &sid in &tok.seq_ids {
                if sid >= self.n_seq_max {
                    return Err(Error::new(format!(
                        "init: invalid seq_id = {} >= {}",
                        sid, self.n_seq_max
                    )));
                }
            }
            seq_id.push(tok.seq_ids.iter().copied().collect());
        }
        let n_seq_id: Vec<i32> = seq_id.iter().map(|v| v.len() as i32).collect();

        //
        // build output/logits array
        //

        let mut output: Vec<i8> = batch_inp
            .tokens
            .iter()
            .map(|t| t.output as i8)
            .collect();

        if output_all && output.iter().any(|&o| o == 0) {
            // warn: embeddings required but some input tokens were not marked as outputs -> overriding
            output.fill(1);
        }

        // kept empty if no entry has one
        let mut decision_order = Vec::new();
        let mut decision_group = Vec::new();
        for (i, tok) in batch_inp.tokens.iter().enumerate() {
            if tok.decision_order != 0 {
                if decision_order.is_empty() {
                    decision_order.resize(n_tok, 0);
                }
                decision_order[i] = tok.decision_order;
            }
            if tok.decision_group != 0 {
                if decision_group.is_empty() {
                    decision_group.resize(n_tok, 0);
                }
                decision_group[i] = tok.decision_group;
            }
        }

        //
        // set up the internal Batch to point to our owned arrays
        //

        self.batch = Batch {
            n_tokens: n_tok,
            token: if has_token { Some(token_vec) } else { None },
            embd: if has_embd { Some(embd_vec) } else { None },
            pos: Some(pos),
            n_seq_id: Some(n_seq_id),
            seq_id: Some(seq_id),
            logits: Some(output),
        };
        self.is_embd_vec = is_embd_vec;
        self.state_vec = state_vec;
        self.embd_row_vec = embd_row_vec;
        self.state_row_vec = state_row_vec;
        self.decision_order = decision_order;
        self.decision_group = decision_group;

        //
        // compute stats
        //

        // count the outputs in this batch
        self.n_outputs = self
            .batch
            .logits
            .as_ref()
            .unwrap()
            .iter()
            .filter(|&&o| o != 0)
            .count() as u32;

        self.has_cpl = false;

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

        let mem = batch_inp.mem.as_deref();

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

        self.split_reset();

        Ok(())
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

    fn clear(&mut self) {
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
    fn ubatch_add(&mut self, idxs: &[usize], n_seqs: u32, equal_seqs: bool) -> Ubatch {
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

/// RAII translation layer: converts a legacy `Batch` into a `BatchExt`.
pub struct BatchCompat {
    pub batch_ext: BatchExt,
}

impl BatchCompat {
    // n_embd_row is the embd row width of batch_inp, 0 = use the decoder width
    pub fn new(batch_ext: BatchExt, batch_inp: &Batch, n_embd_row: usize) -> Self {
        let mut res = Self { batch_ext };
        Self::init(&mut res.batch_ext, batch_inp, n_embd_row);
        res
    }

    // fill an existing BatchExt from a Batch (old API)
    pub fn init(dst: &mut BatchExt, batch_inp: &Batch, n_embd_row: usize) {
        let n_embd_row = if n_embd_row == 0 {
            dst.n_embd_inp
        } else {
            n_embd_row
        };

        // a batch can carry both, for example the MTP hook batches
        let has_token = batch_inp.has_token();
        let has_embd = batch_inp.has_embd();

        // auto-generates positions locally when batch_inp.pos is null, continuing from memory
        let mut pos_next = vec![0i32; dst.n_seq_max as usize];
        for (s, p) in pos_next.iter_mut().enumerate() {
            *p = dst
                .mem
                .as_ref()
                .and_then(|m| m.seq_pos_max(s as SeqId))
                .map(|p| p + 1)
                .unwrap_or(0); // assume next pos
        }

        for i in 0..batch_inp.n_tokens {
            let n_sid = batch_inp
                .n_seq_id
                .as_ref()
                .map(|v| v[i] as usize)
                .unwrap_or(1);
            let default_sid = [0u32];
            let sids: &[SeqId] = batch_inp
                .seq_id
                .as_ref()
                .map(|v| v[i].as_slice())
                .unwrap_or(&default_sid);

            let mut t = TokenEntry::default();
            t.seq_ids.clear();
            for &sid in &sids[..n_sid.min(sids.len())] {
                t.seq_ids.insert(sid);
            }

            // position(s)
            if let Some(pos) = &batch_inp.pos {
                if has_token {
                    // token batch: one position per token
                    t.pos[0] = pos[i];
                } else {
                    // embedding batch (M-RoPE): section-major layout pos[j*n_tokens + i]
                    for j in 0..dst.n_pos_per_embd {
                        t.pos[j] = pos[j * batch_inp.n_tokens + i];
                    }
                }
            } else {
                // auto-generate position from the first seq_id
                let sid = sids.first().copied().unwrap_or(0) as usize;
                t.pos[0] = pos_next[sid];
                pos_next[sid] += 1;
            }

            // token id and/or embeddings
            if has_token {
                t.id = batch_inp.token.as_ref().unwrap()[i];
            }

            // legacy MTP hook batches carry the hidden state next to the token ids
            if has_embd && has_token && dst.n_embd_state > 0 {
                t.has_state = true;
                t.state_off = dst.state.len();
                let src = &batch_inp.embd.as_ref().unwrap()
                    [i * dst.n_embd_state..(i + 1) * dst.n_embd_state];
                dst.state.extend_from_slice(src);
            } else if has_embd {
                t.has_embd = true;
                t.embd_off = dst.embd.len();
                let src =
                    &batch_inp.embd.as_ref().unwrap()[i * n_embd_row..(i + 1) * n_embd_row];
                dst.embd.extend_from_slice(src);
                dst.n_embd = n_embd_row;
            }

            // output flag
            // if no logits array is given, default to only the last token being an output
            t.output = match &batch_inp.logits {
                Some(logits) => logits[i] != 0,
                None => i == batch_inp.n_tokens - 1,
            };

            dst.tokens.push(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seq_set_ops() {
        let mut a = SeqSet::default();
        assert!(a.none());
        a.set(0);
        a.set(511);
        assert!(a.test(0) && a.test(511) && !a.test(1));
        assert!(!a.none());

        let mut b = SeqSet::default();
        b.set(0);
        b.set(1);
        let c = a.intersection(&b);
        assert!(c.test(0) && !c.test(1) && !c.test(511));

        let mut all = SeqSet::default();
        all.set_all();
        assert!(all.test(511));
    }

    #[test]
    fn batch_ext_helpers_and_compat() {
        let mut b = BatchExt::new(4, 8, 8, 0, 2, None, 100, 4);
        assert_eq!(b.add_token(0), 0);
        assert_eq!(b.add_token(1), 1);
        assert_eq!(b.add_token(2), -3);
        assert!(b.add_seq(0, 1));
        assert!(!b.add_seq(9, 0));
        assert!(b.set_token_id(0, 42));
        assert!(!b.set_token_id(0, -1));
        assert!(b.set_token_pos(0, &[7, 0, 0, 0]));
        assert!(b.set_output(0, true));
        assert_eq!(b.tokens[0].id, 42);
        assert!(b.tokens[0].seq_ids.contains(&0) && b.tokens[0].seq_ids.contains(&1));
        assert_eq!(b.tokens[0].pos, [7, 0, 0, 0]);
        assert!(b.tokens[0].output);
        for _ in 0..2 {
            b.add_token(0);
        }
        assert_eq!(b.add_token(0), -1);

        let ext = BatchExt::new(2, 1, 1, 0, 1, None, 10, 4);
        let compat = BatchCompat::new(ext, &batch_get_one(&[3, 5]), 0);
        assert_eq!(compat.batch_ext.tokens.len(), 2);
        assert_eq!(compat.batch_ext.tokens[0].id, 3);
        assert!(!compat.batch_ext.tokens[0].output);
        assert!(compat.batch_ext.tokens[1].output);
        assert_eq!(
            [compat.batch_ext.tokens[0].pos[0], compat.batch_ext.tokens[1].pos[0]],
            [0, 1]
        );
    }
}

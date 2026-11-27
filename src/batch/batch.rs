//! Legacy batch, extended batch and the sanitizing `init` pass.

use std::collections::BTreeSet;

use crate::core::{Error, Result, SeqId, Token, LLAMA_TOKEN_NULL};

use super::{BatchAllocr, Memory, Pos, GGML_MROPE_SECTIONS, LLAMA_MAX_SEQ};

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

impl BatchAllocr {
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

        self.detect_coupled_sequences(n_tok);

        self.build_seq_sets(n_tok);

        self.check_sequence_positions(n_tok, mixed, n_seq_max, batch_inp.mem.as_deref())?;
        self.check_sequence_sets(n_tok, n_seq_max)?;

        self.split_reset();

        Ok(())
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

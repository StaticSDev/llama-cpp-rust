//! Vocabulary metadata loading, special token detection and EOG handling.

use super::*;
use crate::core::{Error, Result, Token, LLAMA_TOKEN_NULL};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

// ---------------------------------------------------------------------------
// metadata keys and loader input
// ---------------------------------------------------------------------------

/// Named GGUF metadata keys used by `llama_vocab::load`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Kv(pub &'static str);

impl Kv {
    pub const VOCAB_SIZE: Kv = Kv("vocab_size");
    pub const GENERAL_NAME: Kv = Kv("general.name");
    pub const GENERAL_ARCHITECTURE: Kv = Kv("general.architecture");
    pub const TOKENIZER_MODEL: Kv = Kv("tokenizer.ggml.model");
    pub const TOKENIZER_PRE: Kv = Kv("tokenizer.ggml.pre");
    pub const TOKENIZER_LIST: Kv = Kv("tokenizer.ggml.tokens");
    pub const TOKENIZER_SCORES: Kv = Kv("tokenizer.ggml.scores");
    pub const TOKENIZER_TOKEN_TYPE: Kv = Kv("tokenizer.ggml.token_type");
    pub const TOKENIZER_TOKEN_TYPE_COUNT: Kv = Kv("tokenizer.ggml.token_type_count");
    pub const TOKENIZER_MERGES: Kv = Kv("tokenizer.ggml.merges");
    pub const TOKENIZER_PRECOMPILED_CHARSMAP: Kv = Kv("tokenizer.ggml.precompiled_charsmap");
    pub const TOKENIZER_ADD_PREFIX: Kv = Kv("tokenizer.ggml.add_space_prefix");
    pub const TOKENIZER_REMOVE_EXTRA_WS: Kv = Kv("tokenizer.ggml.remove_extra_whitespaces");
    pub const TOKENIZER_CLEAN_SPACES: Kv = Kv("tokenizer.ggml.clean_up_tokenization_spaces");
    pub const TOKENIZER_ADD_BOS: Kv = Kv("tokenizer.ggml.add_bos_token");
    pub const TOKENIZER_ADD_EOS: Kv = Kv("tokenizer.ggml.add_eos_token");
    pub const TOKENIZER_ADD_SEP: Kv = Kv("tokenizer.ggml.add_sep_token");
    pub const TOKENIZER_DNA_AUTO_TAGS: Kv = Kv("tokenizer.ggml.dna_auto_tags");
    pub const TOKENIZER_BOS_ID: Kv = Kv("tokenizer.ggml.bos_token_id");
    pub const TOKENIZER_EOS_ID: Kv = Kv("tokenizer.ggml.eos_token_id");
    pub const TOKENIZER_EOT_ID: Kv = Kv("tokenizer.ggml.eot_token_id");
    pub const TOKENIZER_EOM_ID: Kv = Kv("tokenizer.ggml.eom_token_id");
    pub const TOKENIZER_UNK_ID: Kv = Kv("tokenizer.ggml.unknown_token_id");
    pub const TOKENIZER_SEP_ID: Kv = Kv("tokenizer.ggml.separator_token_id");
    pub const TOKENIZER_PAD_ID: Kv = Kv("tokenizer.ggml.padding_token_id");
    pub const TOKENIZER_MASK_ID: Kv = Kv("tokenizer.ggml.mask_token_id");
    pub const TOKENIZER_FIM_PRE_ID: Kv = Kv("tokenizer.ggml.fim_prefix_token_id");
    pub const TOKENIZER_FIM_SUF_ID: Kv = Kv("tokenizer.ggml.fim_suffix_token_id");
    pub const TOKENIZER_FIM_MID_ID: Kv = Kv("tokenizer.ggml.fim_middle_token_id");
    pub const TOKENIZER_FIM_PAD_ID: Kv = Kv("tokenizer.ggml.fim_pad_token_id");
    pub const TOKENIZER_FIM_REP_ID: Kv = Kv("tokenizer.ggml.fim_repo_token_id");
    pub const TOKENIZER_FIM_SEP_ID: Kv = Kv("tokenizer.ggml.fim_sep_token_id");
    pub const TOKENIZER_PREFIX_ID: Kv = Kv("tokenizer.ggml.prefix_token_id");
    pub const TOKENIZER_SUFFIX_ID: Kv = Kv("tokenizer.ggml.suffix_token_id");
    pub const TOKENIZER_MIDDLE_ID: Kv = Kv("tokenizer.ggml.middle_token_id");
    pub const TOKENIZER_NORMALIZER_NFC: Kv = Kv("tokenizer.ggml.normalizer.nfc");
    pub const TOKENIZER_NORMALIZER_LOWERCASE: Kv = Kv("tokenizer.ggml.normalizer.lowercase");
    pub const TOKENIZER_NORMALIZER_STRIP_ACCENTS: Kv = Kv("tokenizer.ggml.normalizer.strip_accents");
    pub const TOKENIZER_METASPACE_PREPEND_SCHEME: Kv = Kv("tokenizer.ggml.metaspace_prepend_scheme");
    pub const TOKENIZER_SUPPRESS_TOKENS: Kv = Kv("tokenizer.ggml.suppress_tokens");
}

/// Stands in for `llama_model_loader`/`gguf_context`. Populate the maps and
/// pass to [`LlamaVocab::load`].
#[derive(Clone, Debug, Default)]
pub struct Metadata {
    pub strings: HashMap<String, String>,
    pub bools: HashMap<String, bool>,
    pub u32s: HashMap<String, u32>,
    pub string_arrays: HashMap<String, Vec<String>>,
    pub i32_arrays: HashMap<String, Vec<i32>>,
    pub f32_arrays: HashMap<String, Vec<f32>>,
    pub blobs: HashMap<String, Vec<u8>>,
}

impl Metadata {
    pub fn set_string(&mut self, key: Kv, value: impl Into<String>) {
        self.strings.insert(key.0.to_string(), value.into());
    }
    pub fn set_bool(&mut self, key: Kv, value: bool) {
        self.bools.insert(key.0.to_string(), value);
    }
    pub fn set_u32(&mut self, key: Kv, value: u32) {
        self.u32s.insert(key.0.to_string(), value);
    }
    pub fn set_string_array(&mut self, key: Kv, value: Vec<String>) {
        self.string_arrays.insert(key.0.to_string(), value);
    }
    pub fn set_i32_array(&mut self, key: Kv, value: Vec<i32>) {
        self.i32_arrays.insert(key.0.to_string(), value);
    }
    pub fn set_f32_array(&mut self, key: Kv, value: Vec<f32>) {
        self.f32_arrays.insert(key.0.to_string(), value);
    }
    pub fn set_blob(&mut self, key: Kv, value: Vec<u8>) {
        self.blobs.insert(key.0.to_string(), value);
    }

    pub fn get_string(&self, key: Kv, dst: &mut String) -> bool {
        match self.strings.get(key.0) {
            Some(v) => {
                *dst = v.clone();
                true
            }
            None => false,
        }
    }
    pub fn get_bool(&self, key: Kv, dst: &mut bool) -> bool {
        match self.bools.get(key.0) {
            Some(v) => {
                *dst = *v;
                true
            }
            None => false,
        }
    }
    pub fn get_u32(&self, key: Kv, dst: &mut u32) -> bool {
        match self.u32s.get(key.0) {
            Some(v) => {
                *dst = *v;
                true
            }
            None => false,
        }
    }
    pub fn get_string_array(&self, key: Kv) -> Option<&Vec<String>> {
        self.string_arrays.get(key.0)
    }
    pub fn get_i32_array(&self, key: Kv) -> Option<&Vec<i32>> {
        self.i32_arrays.get(key.0)
    }
    pub fn get_f32_array(&self, key: Kv) -> Option<&Vec<f32>> {
        self.f32_arrays.get(key.0)
    }
    pub fn get_blob(&self, key: Kv) -> Option<&Vec<u8>> {
        self.blobs.get(key.0)
    }
}

pub(crate) struct Impl {
    pub(crate) n_token_types: u32, // for BERT-style token types

    pub(crate) tokenizer_model: String,
    pub(crate) tokenizer_pre: String,

    pub(crate) vtype: VocabType,
    pub(crate) pre_type: VocabPreType,

    pub(crate) max_token_len: i32, // used for optimizing longest token search

    // default LLaMA special tokens
    // TODO: should we set all of these to LLAMA_TOKEN_NULL?
    pub(crate) special_bos_id: Token,
    pub(crate) special_eos_id: Token,
    pub(crate) special_eot_id: Token,
    pub(crate) special_eom_id: Token,
    pub(crate) special_unk_id: Token,
    pub(crate) special_sep_id: Token,
    pub(crate) special_pad_id: Token,
    pub(crate) special_mask_id: Token,

    pub(crate) linefeed_id: Token,

    // fim tokens
    pub(crate) special_fim_pre_id: Token,
    pub(crate) special_fim_suf_id: Token,
    pub(crate) special_fim_mid_id: Token,
    pub(crate) special_fim_pad_id: Token,
    pub(crate) special_fim_rep_id: Token, // repo
    pub(crate) special_fim_sep_id: Token, // file separator

    // tokenizer flags
    pub(crate) add_space_prefix: bool,
    pub(crate) add_bos: bool,
    pub(crate) add_eos: bool,
    pub(crate) add_sep: bool,
    pub(crate) dna_auto_tags: bool, // hybriddna: tag untagged DNA runs before tokenizing
    pub(crate) ignore_merges: bool,
    pub(crate) clean_spaces: bool, // clean_up_tokenization_spaces
    pub(crate) remove_extra_whitespaces: bool,
    pub(crate) escape_whitespaces: bool,
    pub(crate) treat_whitespace_as_suffix: bool,

    pub(crate) normalizer_opts: NormalizerOptions,
    pub(crate) metaspace_prepend_scheme: MetaspacePrependScheme,

    pub(crate) token_to_id: HashMap<String, Token>,
    pub(crate) id_to_token: Vec<TokenData>,

    pub(crate) cache_special_tokens: Vec<Token>,
    pub(crate) cache_token_to_piece: Vec<String>, // llama_token_to_piece(special = true)
    pub(crate) bpe_ranks: HashMap<(String, String), i32>,

    // set of all tokens that cause "end of generation"
    pub(crate) special_eog_ids: BTreeSet<Token>,

    pub(crate) suppress_tokens: Vec<Token>,

    pub(crate) tokenizer: Tokenizer,

    pub(crate) precompiled_charsmap: Vec<u8>,
}

impl Default for Impl {
    fn default() -> Self {
        Self {
            n_token_types: 0,
            tokenizer_model: String::new(),
            tokenizer_pre: String::new(),
            vtype: VocabType::Spm,
            pre_type: VocabPreType::Default,
            max_token_len: 0,
            special_bos_id: 1,
            special_eos_id: 2,
            special_eot_id: LLAMA_TOKEN_NULL,
            special_eom_id: LLAMA_TOKEN_NULL,
            special_unk_id: 0,
            special_sep_id: LLAMA_TOKEN_NULL,
            special_pad_id: LLAMA_TOKEN_NULL,
            special_mask_id: LLAMA_TOKEN_NULL,
            linefeed_id: 13,
            special_fim_pre_id: LLAMA_TOKEN_NULL,
            special_fim_suf_id: LLAMA_TOKEN_NULL,
            special_fim_mid_id: LLAMA_TOKEN_NULL,
            special_fim_pad_id: LLAMA_TOKEN_NULL,
            special_fim_rep_id: LLAMA_TOKEN_NULL,
            special_fim_sep_id: LLAMA_TOKEN_NULL,
            add_space_prefix: false,
            add_bos: false,
            add_eos: false,
            add_sep: false,
            dna_auto_tags: false,
            ignore_merges: false,
            clean_spaces: false,
            remove_extra_whitespaces: false,
            escape_whitespaces: true,
            treat_whitespace_as_suffix: false,
            normalizer_opts: NormalizerOptions::default(),
            metaspace_prepend_scheme: MetaspacePrependScheme::Never,
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            cache_special_tokens: Vec::new(),
            cache_token_to_piece: Vec::new(),
            bpe_ranks: HashMap::new(),
            special_eog_ids: BTreeSet::new(),
            suppress_tokens: Vec::new(),
            tokenizer: Tokenizer::Spm,
            precompiled_charsmap: Vec::new(),
        }
    }
}

// helpers shared with the tokenizer sessions
impl Impl {
    pub(crate) fn n_tokens(&self) -> u32 {
        self.id_to_token.len() as u32
    }
    pub(crate) fn max_token_len(&self) -> i32 {
        self.max_token_len
    }
    pub(crate) fn get_tokenizer_model(&self) -> &str {
        &self.tokenizer_model
    }
    pub(crate) fn get_pre_type(&self) -> VocabPreType {
        self.pre_type
    }
    pub(crate) fn get_normalizer_opts(&self) -> NormalizerOptions {
        self.normalizer_opts
    }
    pub(crate) fn get_metaspace_prepend_scheme(&self) -> MetaspacePrependScheme {
        self.metaspace_prepend_scheme
    }
    pub(crate) fn get_add_space_prefix(&self) -> bool {
        self.add_space_prefix
    }
    pub(crate) fn get_add_bos(&self) -> bool {
        self.add_bos
    }
    pub(crate) fn get_add_eos(&self) -> bool {
        self.add_eos
    }
    pub(crate) fn get_remove_extra_whitespaces(&self) -> bool {
        self.remove_extra_whitespaces
    }
    pub(crate) fn get_escape_whitespaces(&self) -> bool {
        self.escape_whitespaces
    }
    pub(crate) fn get_treat_whitespace_as_suffix(&self) -> bool {
        self.treat_whitespace_as_suffix
    }
    pub(crate) fn get_ignore_merges(&self) -> bool {
        self.ignore_merges
    }
    pub(crate) fn get_dna_auto_tags(&self) -> bool {
        self.dna_auto_tags
    }
    pub(crate) fn get_clean_spaces(&self) -> bool {
        self.clean_spaces
    }

    pub(crate) fn text_to_token(&self, text: &str) -> Token {
        self.token_to_id.get(text).copied().unwrap_or(LLAMA_TOKEN_NULL)
    }

    pub(crate) fn get_token_data(&self, id: Token) -> &TokenData {
        &self.id_to_token[id as usize]
    }

    pub(crate) fn token_get_score(&self, id: Token) -> f32 {
        self.id_to_token[id as usize].score
    }

    pub(crate) fn token_get_attr(&self, id: Token) -> LlamaTokenAttr {
        self.id_to_token[id as usize].attr
    }

    pub(crate) fn is_user_defined(&self, id: Token) -> bool {
        self.id_to_token[id as usize].attr & LLAMA_TOKEN_ATTR_USER_DEFINED != 0
    }

    pub(crate) fn token_unk(&self) -> Token {
        self.special_unk_id
    }
    pub(crate) fn token_bos(&self) -> Token {
        self.special_bos_id
    }
    pub(crate) fn token_eos(&self) -> Token {
        self.special_eos_id
    }

    pub(crate) fn find_bpe_rank(&self, token_left: &str, token_right: &str) -> i32 {
        debug_assert!(!token_left.contains(' '));
        debug_assert!(!token_right.contains(' '));
        self.bpe_ranks
            .get(&(token_left.to_string(), token_right.to_string()))
            .copied()
            .unwrap_or(-1)
    }

    pub(crate) fn byte_to_token(&self, ch: u8) -> Token {
        match self.vtype {
            VocabType::Spm | VocabType::Ugm => {
                const HEX: &[u8; 16] = b"0123456789ABCDEF";
                let buf = [
                    b'<', b'0', b'x', HEX[(ch >> 4) as usize], HEX[(ch & 15) as usize], b'>',
                ];
                let name = std::str::from_utf8(&buf).unwrap_or("");
                if let Some(&token) = self.token_to_id.get(name) {
                    return token;
                }
                // Try to fall back to just the byte as a string
                self.token_to_id
                    .get(&String::from_utf8_lossy(&[ch]).into_owned())
                    .copied()
                    .unwrap_or(LLAMA_TOKEN_NULL)
            }
            VocabType::Wpm | VocabType::Bpe => self
                .token_to_id
                .get(&unicode_byte_to_utf8(ch))
                .copied()
                .unwrap_or(LLAMA_TOKEN_NULL),
            VocabType::Plamo2 | VocabType::Plamo3 => {
                let hex_str = format!("<0x{:02X}>", ch);
                self.token_to_id.get(&hex_str).copied().unwrap_or(LLAMA_TOKEN_NULL)
            }
            VocabType::Test => LLAMA_TOKEN_NULL, // TEST tokens have no byte-level mapping
            VocabType::Rwkv => LLAMA_TOKEN_NULL, // RWKV tokens have no byte-level mapping
            VocabType::None => LLAMA_TOKEN_NULL,
        }
    }

    pub(crate) fn load_bpe_ranks(ml: &Metadata) -> Result<Option<HashMap<(String, String), i32>>> {
        let merges = match ml.get_string_array(Kv::TOKENIZER_MERGES) {
            Some(v) => v,
            None => return Ok(None),
        };
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, word) in merges.iter().enumerate() {
            let sep = word
                .char_indices()
                .skip(1)
                .find(|(_, c)| *c == ' ')
                .map(|(p, _)| p);
            let left = match sep {
                Some(s) => word[..s].to_string(),
                None => String::new(),
            };
            let right = match sep {
                Some(s) => word[s + 1..].to_string(),
                None => String::new(),
            };
            ranks.entry((left, right)).or_insert(i as i32);
        }
        Ok(Some(ranks))
    }

    pub(crate) fn load(&mut self, ml: &Metadata) -> Result<()> {
        // determine vocab type
        {
            ml.get_string(Kv::TOKENIZER_MODEL, &mut self.tokenizer_model);
            ml.get_string(Kv::TOKENIZER_PRE, &mut self.tokenizer_pre);
            ml.get_u32(Kv::TOKENIZER_TOKEN_TYPE_COUNT, &mut self.n_token_types);

            if self.tokenizer_model == "no_vocab" || self.tokenizer_model == "none" {
                self.vtype = VocabType::None;

                // default special tokens
                self.special_bos_id = LLAMA_TOKEN_NULL;
                self.special_eos_id = LLAMA_TOKEN_NULL;
                self.special_unk_id = LLAMA_TOKEN_NULL;
                self.special_sep_id = LLAMA_TOKEN_NULL;
                self.special_pad_id = LLAMA_TOKEN_NULL;
                self.special_mask_id = LLAMA_TOKEN_NULL;
                self.linefeed_id = LLAMA_TOKEN_NULL;

                // read vocab size from metadata
                let mut n_tokens = 0u32;
                if ml.get_u32(Kv::VOCAB_SIZE, &mut n_tokens) {
                    // adding dummy tokens
                    self.id_to_token.resize(n_tokens as usize, TokenData::default());
                }
                return Ok(());
            }

            match self.tokenizer_model.as_str() {
                "llama" => {
                    self.vtype = VocabType::Spm;
                    self.special_bos_id = 1;
                    self.special_eos_id = 2;
                    self.special_unk_id = 0;
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = LLAMA_TOKEN_NULL;
                    self.special_mask_id = LLAMA_TOKEN_NULL;
                }
                "bert" => {
                    self.vtype = VocabType::Wpm;
                    self.special_bos_id = 101;
                    self.special_eos_id = LLAMA_TOKEN_NULL;
                    self.special_unk_id = 100;
                    self.special_sep_id = 102;
                    self.special_pad_id = 0;
                    self.special_mask_id = 103;
                    self.add_sep = true;
                }
                "gpt2" | "hybriddna" | "whitespace" => {
                    self.vtype = VocabType::Bpe;

                    // Kimi-K2 uses custom tokenization without traditional BPE merges
                    let is_kimi_k2 = self.tokenizer_pre == "kimi-k2";

                    // read bpe merges and populate bpe ranks
                    match Self::load_bpe_ranks(ml)? {
                        Some(ranks) => self.bpe_ranks = ranks,
                        None => {
                            if !is_kimi_k2 {
                                return Err(Error::new("cannot find tokenizer merges in model file"));
                            }
                            // Kimi-K2 doesn't need merges, skip
                        }
                    }

                    // default special tokens
                    self.special_bos_id = 11;
                    self.special_eos_id = 11;
                    self.special_unk_id = LLAMA_TOKEN_NULL;
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = LLAMA_TOKEN_NULL;
                    self.special_mask_id = LLAMA_TOKEN_NULL;
                }
                "t5" => {
                    self.vtype = VocabType::Ugm;
                    self.special_bos_id = LLAMA_TOKEN_NULL;
                    self.special_eos_id = 1;
                    self.special_unk_id = 2;
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = 0;
                    self.special_mask_id = LLAMA_TOKEN_NULL;

                    if let Some(data) = ml.get_blob(Kv::TOKENIZER_PRECOMPILED_CHARSMAP) {
                        if data.len() < std::mem::size_of::<u32>() {
                            return Err(Error::new("precompiled_charsmap too small for xcda_blob_size header!"));
                        }
                        let xcda_blob_size = u32::from_le_bytes([
                            data[0], data[1], data[2], data[3],
                        ]) as usize;
                        if xcda_blob_size + std::mem::size_of::<u32>() >= data.len() {
                            return Err(Error::new("Index out of array bounds in precompiled charsmap!"));
                        }
                        self.precompiled_charsmap = data.clone();
                    }
                }
                "rwkv" => {
                    self.vtype = VocabType::Rwkv;
                    self.special_bos_id = LLAMA_TOKEN_NULL;
                    self.special_eos_id = LLAMA_TOKEN_NULL;
                    self.special_unk_id = LLAMA_TOKEN_NULL;
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = LLAMA_TOKEN_NULL;
                }
                "test" => {
                    self.vtype = VocabType::Test;
                    self.special_bos_id = LLAMA_TOKEN_NULL;
                    self.special_eos_id = LLAMA_TOKEN_NULL;
                    self.special_unk_id = LLAMA_TOKEN_NULL;
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = LLAMA_TOKEN_NULL;
                    self.special_mask_id = LLAMA_TOKEN_NULL;
                }
                "plamo2" | "plamo3" => {
                    self.vtype = if self.tokenizer_model == "plamo2" {
                        VocabType::Plamo2
                    } else {
                        VocabType::Plamo3
                    };
                    // PLaMo default special tokens (these will be overridden by model config)
                    self.special_bos_id = 1; // <|plamo:bos|>
                    self.special_eos_id = 2; // <|plamo:eos|>
                    self.special_unk_id = 0; // <|plamo:unk|>
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = 3; // <|plamo:pad|>
                    self.special_mask_id = LLAMA_TOKEN_NULL;
                }
                "gemma4" => {
                    self.vtype = VocabType::Bpe;
                    match Self::load_bpe_ranks(ml)? {
                        Some(ranks) => self.bpe_ranks = ranks,
                        None => return Err(Error::new("cannot find tokenizer merges in model file")),
                    }
                    // default special tokens (to be read from GGUF)
                    self.special_bos_id = LLAMA_TOKEN_NULL;
                    self.special_eos_id = LLAMA_TOKEN_NULL;
                    self.special_unk_id = LLAMA_TOKEN_NULL;
                    self.special_sep_id = LLAMA_TOKEN_NULL;
                    self.special_pad_id = LLAMA_TOKEN_NULL;
                    self.special_mask_id = LLAMA_TOKEN_NULL;
                    self.tokenizer_pre = "gemma4".to_string();
                }
                _ => {
                    return Err(Error::new(format!(
                        "unknown tokenizer: '{}'",
                        self.tokenizer_model
                    )));
                }
            }

            // for now, only BPE models have pre-tokenizers
            if self.vtype == VocabType::Bpe {
                self.add_space_prefix = false;
                self.escape_whitespaces = false;
                self.clean_spaces = true;
                if self.tokenizer_pre.is_empty() {
                    self.pre_type = VocabPreType::Default;
                } else {
                    self.pre_type = self.parse_pre_type()?;
                }
            } else if self.vtype == VocabType::Spm {
                self.pre_type = VocabPreType::Default;
                self.add_space_prefix = true;
                self.clean_spaces = false;
                self.add_bos = true;
                self.add_eos = false;
            } else if self.vtype == VocabType::Wpm {
                self.pre_type = VocabPreType::Default;
                self.add_space_prefix = false;
                self.clean_spaces = true;
                self.add_bos = true;
                self.add_eos = false;
                self.add_sep = true;
            } else if self.vtype == VocabType::Ugm {
                self.pre_type = VocabPreType::Default;
                self.add_bos = false;
                self.add_eos = true;
            } else if self.vtype == VocabType::Rwkv {
                self.pre_type = VocabPreType::Default;
                self.add_space_prefix = false;
                self.clean_spaces = false;
                self.add_bos = false;
                self.add_eos = false;
            } else {
                self.pre_type = VocabPreType::Default;
            }

            ml.get_bool(Kv::TOKENIZER_ADD_PREFIX, &mut self.add_space_prefix);
            ml.get_bool(Kv::TOKENIZER_REMOVE_EXTRA_WS, &mut self.remove_extra_whitespaces);
            ml.get_bool(Kv::TOKENIZER_CLEAN_SPACES, &mut self.clean_spaces);
        }

        self.load_tokens(ml)?;
        self.init_tokenizer()?;

        // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
        if self.vtype == VocabType::Spm {
            let id = self.byte_to_token(b'\n');
            if id == LLAMA_TOKEN_NULL {
                self.linefeed_id = self.special_pad_id;
            } else {
                self.linefeed_id = id;
            }
        } else if self.vtype == VocabType::Wpm {
            self.linefeed_id = self.special_pad_id;
        } else {
            let ids = self.tokenize("\n", false, false);
            if ids.is_empty() {
                self.linefeed_id = self.special_pad_id;
            } else {
                self.linefeed_id = ids[0];
            }
        }

        self.load_specials(ml)?;
        Ok(())
    }

    pub(crate) fn parse_pre_type(&mut self) -> Result<VocabPreType> {
        use VocabPreType as P;
        let p = match self.tokenizer_pre.as_str() {
            "" => {
                // missing pre-tokenizer type, using: 'default'
                // GENERATION QUALITY WILL BE DEGRADED!
                P::Default
            }
            "flint" => {
                self.clean_spaces = false;
                P::Flint
            }
            "default" => P::Default,
            "minicpm5" => {
                self.ignore_merges = true;
                self.clean_spaces = false;
                P::Minicpm5
            }
            "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "midm-2.0" | "jina-v5-nano" => {
                self.ignore_merges = true;
                self.add_bos = true;
                P::Llama3
            }
            "lfm2" => {
                self.ignore_merges = true;
                self.add_bos = true;
                self.clean_spaces = false;
                P::Llama3
            }
            "deepseek-llm" => {
                self.clean_spaces = false;
                P::DeepseekLlm
            }
            "deepseek-coder" => {
                self.clean_spaces = false;
                P::DeepseekCoder
            }
            "deepseek-v3" => {
                self.clean_spaces = false;
                P::Deepseek3Llm
            }
            "spark2_5" => {
                self.clean_spaces = false;
                P::Spark2p5
            }
            "youtu" => {
                self.clean_spaces = false;
                self.ignore_merges = true;
                P::Youtu
            }
            "falcon" => P::Falcon,
            "mpt" => P::Mpt,
            "starcoder" => P::Starcoder,
            "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es"
            | "jina-v2-de" | "a.x-4.0" | "mellum" | "modern-bert" => P::Gpt2,
            "jais-2" => P::Jais2,
            "gemma4" | "granite-embed-multi-311m" => {
                self.escape_whitespaces = true;
                self.clean_spaces = false;
                P::Gemma4
            }
            "mmbert" => {
                self.escape_whitespaces = true;
                self.metaspace_prepend_scheme = MetaspacePrependScheme::Always;
                P::Mbert
            }
            "sarvam-moe" => {
                self.escape_whitespaces = true;
                self.clean_spaces = false;
                P::SarvamMoe
            }
            "jina-v1-en" | "jina-v2-code" | "roberta-bpe" => {
                self.add_sep = true;
                P::Gpt2
            }
            "whitespace" => {
                self.normalizer_opts.lowercase = false;
                P::Whitespace
            }
            "refact" => P::Refact,
            "command-r" => {
                self.clean_spaces = false;
                P::CommandR
            }
            "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "domyn-edge" => {
                self.clean_spaces = false;
                P::Qwen2
            }
            "raon" => {
                self.clean_spaces = false;
                P::Raon
            }
            "qwen35" => {
                self.clean_spaces = false;
                P::Qwen35
            }
            "stablelm2" => P::StableLm2,
            "olmo" => P::Olmo,
            "dbrx" => {
                self.clean_spaces = false;
                P::Dbrx
            }
            "smaug-bpe" => P::Smaug,
            "poro-chat" => {
                self.clean_spaces = false;
                P::Poro
            }
            "chatglm-bpe" => {
                self.special_bos_id = LLAMA_TOKEN_NULL;
                self.clean_spaces = false;
                P::ChatGlm4
            }
            "glm4" | "glm5" => {
                self.special_bos_id = LLAMA_TOKEN_NULL;
                self.ignore_merges = true;
                self.clean_spaces = false;
                P::ChatGlm4
            }
            "viking" => {
                self.clean_spaces = false;
                P::Viking
            }
            "jais" => P::Jais,
            "tekken" | "pixtral" => {
                self.clean_spaces = false;
                self.ignore_merges = true;
                self.add_bos = true;
                P::Tekken
            }
            "tekken-num3" => {
                self.clean_spaces = false;
                self.ignore_merges = true;
                self.add_bos = true;
                P::TekkenNum3
            }
            "smollm" => {
                self.clean_spaces = false;
                P::Smallm
            }
            "codeshell" => P::Codeshell,
            "bloom" => P::Bloom,
            "gpt3-finnish" => P::Gpt3Finnish,
            "exaone" => P::Exaone,
            "exaone4" => {
                self.clean_spaces = false;
                P::Gpt2
            }
            "exaone-moe" => P::ExaoneMoe,
            "chameleon" => {
                self.add_bos = true;
                self.clean_spaces = false;
                P::Chameleon
            }
            "minerva-7b" => P::Minerva,
            "domyn-small" => {
                self.escape_whitespaces = true;
                P::DomynSmall
            }
            "megrez" => P::Qwen2,
            "gpt-4o" | "llama4" | "kanana2" | "talkie" => {
                self.clean_spaces = false;
                P::Gpt4o
            }
            "inkling2" => {
                self.special_bos_id = LLAMA_TOKEN_NULL;
                self.clean_spaces = false;
                self.ignore_merges = true;
                P::Gpt4o
            }
            "granite-embed-multi-97m" => {
                self.clean_spaces = false;
                self.ignore_merges = true;
                P::GraniteEmbMulti
            }
            "tiny_aya" | "cohere2moe" => {
                self.clean_spaces = false;
                P::TinyAya
            }
            "superbpe" => {
                self.clean_spaces = false;
                P::SuperBpe
            }
            "trillion" => {
                self.clean_spaces = false;
                P::Trillion
            }
            "granite-docling" => {
                self.clean_spaces = false;
                P::GraniteDocling
            }
            "bailingmoe" | "bailingmoe2" | "llada-moe" => {
                self.clean_spaces = false;
                P::BailingMoe
            }
            "seed-coder" => {
                self.clean_spaces = false;
                P::SeedCoder
            }
            "hunyuan" => {
                self.clean_spaces = false;
                P::Hunyuan
            }
            "hunyuan-dense" => {
                self.clean_spaces = false;
                P::HunyuanDense
            }
            "hy_v4" => {
                self.clean_spaces = false;
                P::HyV4
            }
            "joyai-llm" => {
                self.clean_spaces = false;
                P::JoyaiLlm
            }
            "kimi-k2" => {
                self.clean_spaces = false;
                P::KimiK2
            }
            "ufakzeka" => {
                self.clean_spaces = false;
                P::Ufakzeka
            }
            "grok-2" => {
                self.clean_spaces = false;
                P::Grok2
            }
            "afmoe" => {
                self.clean_spaces = false;
                P::Afmoe
            }
            "laguna" => {
                self.clean_spaces = false;
                P::Laguna
            }
            "minimax-m2" => {
                self.clean_spaces = false;
                P::MinimaxM2
            }
            "solar-open" => {
                self.clean_spaces = false;
                P::SolarOpen
            }
            "mellum2" => {
                self.clean_spaces = false;
                P::Mellum2
            }
            "k2-horizon" => {
                self.clean_spaces = false;
                P::K2Horizon
            }
            "nemotron-3.5-embed" => {
                self.ignore_merges = true;
                self.add_bos = true;
                P::Nemotron35Embd
            }
            "soprano" => P::Soprano,
            "zonos3" => {
                self.clean_spaces = false;
                P::Zonos3
            }
            _ => return Err(Error::new(format!("unknown pre-tokenizer type: '{}'", self.tokenizer_pre))),
        };
        Ok(p)
    }

    pub(crate) fn load_tokens(&mut self, ml: &Metadata) -> Result<()> {
        let token_idx = ml
            .get_string_array(Kv::TOKENIZER_LIST)
            .ok_or_else(|| Error::new("cannot find tokenizer vocab in model file"))?;
        let n_tokens = token_idx.len();

        let scores = ml.get_f32_array(Kv::TOKENIZER_SCORES);
        let iscores = ml.get_i32_array(Kv::TOKENIZER_SCORES);
        if let Some(s) = scores {
            if s.len() < n_tokens {
                return Err(Error::new(format!(
                    "Index out of array bounds for scores ({} < {})",
                    s.len(),
                    n_tokens
                )));
            }
        } else if let Some(s) = iscores {
            if s.len() < n_tokens {
                return Err(Error::new(format!(
                    "Index out of array bounds for scores ({} < {})",
                    s.len(),
                    n_tokens
                )));
            }
        }
        let toktypes = ml.get_i32_array(Kv::TOKENIZER_TOKEN_TYPE);
        if let Some(t) = toktypes {
            if t.len() < n_tokens {
                return Err(Error::new(format!(
                    "Index out of array bounds for toktypes ({} < {})",
                    t.len(),
                    n_tokens
                )));
            }
        }

        self.id_to_token.clear();
        self.id_to_token.resize(n_tokens, TokenData::default());
        self.token_to_id.clear();
        self.token_to_id.reserve(n_tokens);

        for (i, word_ref) in token_idx.iter().enumerate() {
            let mut word = word_ref.clone();
            if word.is_empty() {
                word = format!("[EMPTY_{}]", i);
            }

            self.token_to_id.insert(word.clone(), i as Token);
            self.max_token_len = self.max_token_len.max(word.len() as i32);

            let token_data = &mut self.id_to_token[i];
            token_data.text = word;
            token_data.score = if let Some(s) = scores {
                s[i]
            } else if let Some(s) = iscores {
                s[i] as f32
            } else {
                0.0
            };
            token_data.attr = LLAMA_TOKEN_ATTR_NORMAL;

            if let Some(types) = toktypes {
                //TODO: remove, required until per token attributes are available from GGUF file
                token_data.attr = match types[i] {
                    LLAMA_TOKEN_TYPE_UNKNOWN => LLAMA_TOKEN_ATTR_UNKNOWN,
                    LLAMA_TOKEN_TYPE_UNUSED => LLAMA_TOKEN_ATTR_UNUSED,
                    LLAMA_TOKEN_TYPE_NORMAL => LLAMA_TOKEN_ATTR_NORMAL,
                    LLAMA_TOKEN_TYPE_CONTROL => LLAMA_TOKEN_ATTR_CONTROL,
                    LLAMA_TOKEN_TYPE_USER_DEFINED => LLAMA_TOKEN_ATTR_USER_DEFINED,
                    LLAMA_TOKEN_TYPE_BYTE => LLAMA_TOKEN_ATTR_BYTE,
                    _ => LLAMA_TOKEN_ATTR_UNDEFINED,
                };
            }
        }

        // hybriddna: the marker suffix kept k-mer ids distinct in token_to_id; erase
        // it from id_to_token so the k-mers detokenize to the bare DNA sequence. The
        // k-mers are the block right after <oov>, so only scan from there.
        if self.tokenizer_model == "hybriddna" {
            if let Some(&idx) = self.token_to_id.get("<oov>") {
                for token in self.id_to_token.iter_mut().skip(idx as usize + 1) {
                    if token.text.len() > DNA_KMER_MARKER.len()
                        && token.text.ends_with(DNA_KMER_MARKER)
                    {
                        token.text.truncate(token.text.len() - DNA_KMER_MARKER.len());
                        // tail="pad_n": "CTNNNN" detokenizes to its real bases "CT"
                        // full k-mers are ACGT only, so only N-padded tail tokens end in N
                        if let Some(last_base) = token.text.rfind(|c| c != 'N') {
                            token.text.truncate(last_base + c_len(&token.text[last_base..]));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn init_tokenizer(&mut self) -> Result<()> {
        self.tokenizer = match self.vtype {
            VocabType::Spm => Tokenizer::Spm,
            VocabType::Bpe => {
                let (regex_exprs, byte_encode) = bpe_regex_exprs(self.pre_type);
                Tokenizer::Bpe(BpeData { regex_exprs, byte_encode })
            }
            VocabType::Wpm => Tokenizer::Wpm,
            VocabType::Ugm => Tokenizer::Ugm(self.build_ugm_data()?),
            VocabType::Rwkv => Tokenizer::Rwkv(self.build_rwkv_data()),
            VocabType::Plamo2 | VocabType::Plamo3 => Tokenizer::Plamo2(Plamo2Data::build(
                &self.id_to_token,
                &self.token_to_id,
                self.vtype == VocabType::Plamo3,
            )?),
            VocabType::Test => Tokenizer::Test,
            VocabType::None => return Err(Error::new("unsupported vocab type")),
        };
        Ok(())
    }

    pub(crate) fn build_ugm_data(&self) -> Result<UgmData> {
        let mut data = UgmData {
            min_score: f32::MAX,
            max_score: -f32::MAX,
            ..UgmData::default()
        };

        if !self.precompiled_charsmap.is_empty() {
            let bytes = &self.precompiled_charsmap;
            let xcda_blob_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
            let array_start = std::mem::size_of::<u32>();
            let array_end = array_start + xcda_blob_size;
            if array_end > bytes.len() {
                return Err(Error::new("Index out of array bounds in precompiled charsmap!"));
            }
            data.xcda_array = bytes[array_start..array_end]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            data.prefix_replacements = bytes[array_end..].to_vec();
        }

        for id in 0..self.id_to_token.len() as Token {
            let token_data = &self.id_to_token[id as usize];

            let normal = token_data.attr & LLAMA_TOKEN_ATTR_NORMAL != 0;
            let user_defined = token_data.attr & LLAMA_TOKEN_ATTR_USER_DEFINED != 0;
            let unused = token_data.attr & LLAMA_TOKEN_ATTR_UNUSED != 0;

            if normal {
                data.min_score = data.min_score.min(token_data.score);
                data.max_score = data.max_score.max(token_data.score);
            }

            if normal || user_defined || unused {
                data.token_matcher.insert(token_data.text.as_bytes(), id);
            }

            if user_defined {
                data.user_defined_token_matcher.insert(token_data.text.as_bytes(), id);
            }
        }

        data.unknown_token_score = data.min_score - 10.0;
        Ok(data)
    }

    pub(crate) fn build_rwkv_data(&self) -> RwkvData {
        let mut data = RwkvData::default();
        // RWKV supports arbitrary byte tokens, but the vocab struct only supports string tokens.
        // For now, we decode the vocab here into the lookup we'll use for tokenization.
        for (id, token) in self.id_to_token.iter().enumerate() {
            let text = llama_unescape_rwkv_token(&token.text);
            data.token_matcher.insert(&text, id as Token);
        }
        data
    }
}

/// Byte length of the first UTF-8 character in `s`.
fn c_len(s: &str) -> usize {
    s.chars().next().map(|c| c.len_utf8()).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// special tokens, EOG list, caches (part of impl::load)
// ---------------------------------------------------------------------------

const EOT_TEXTS: &[&str] = &[
    "<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>",
    "<|end_of_text|>", // granite
    "<EOT>", "_<EOT>", "[EOT]", // Kimi-K2
    "<｜end▁of▁sentence｜>", // DeepSeek
    "<end_of_utterance>", // smoldocling
];

const EOM_TEXTS: &[&str] = &["<|eom_id|>"];

const FIM_PRE_TEXTS: &[&str] = &[
    "<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<｜fim▁begin｜>", "<PRE>",
    "▁<PRE>", // CodeLlama
    "<|code_prefix|>", "<|prefix|>", "<|plamo:fim_prefix|>", // PLaMo-3
];

const FIM_SUF_TEXTS: &[&str] = &[
    "<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>", "<｜fim▁hole｜>", "<SUF>",
    "▁<SUF>", // CodeLlama
    "<|code_suffix|>", "<|suffix|>", "<|plamo:fim_suffix|>", // PLaMo-3
];

const FIM_MID_TEXTS: &[&str] = &[
    "<|fim_middle|>", "<fim-middle>", "<fim_middle>", "<｜fim▁end｜>", "<MID>",
    "▁<MID>", // CodeLlama
    "<|code_middle|>", "<|middle|>", "<|plamo:fim_middle|>", // PLaMo-3
];

const FIM_PAD_TEXTS: &[&str] = &[
    "<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]", // Kimi-K2
];

const FIM_REP_TEXTS: &[&str] = &[
    "<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>", // Granite
];

const FIM_SEP_TEXTS: &[&str] = &["<|file_sep|>", "<|plamo:file_separator|>"]; // PLaMo-2/3

const EOG_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|eot|>", // muse-spark
    "<|im_end|>",
    "<|end|>",
    "<|return|>", // o200k_harmony
    "<|call|>",   // o200k_harmony
    "<|flush|>",  // solar-open
    "<|calls|>",  // solar-open
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>", // paddleocr
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]", // Kimi-K2
    "[EOS]", // Kimi-K2
    "<|end_of_text|>",
    "<end_of_utterance>", // smoldocling
    "<eos>",              // gemma4
    "<turn|>",            // gemma4
    "<|tool_response>",   // gemma4
    "<｜end▁of▁sentence｜>", // deepseek-ocr
    "[e~[",               // minimax-m2/m3
    "<|ifm|im_end|>",     // k2-horizon
];

impl Impl {
    pub(crate) fn load_specials(&mut self, ml: &Metadata) -> Result<()> {
        // special tokens
        let n = self.id_to_token.len();
        let mut load_special = |key: Kv, field: &mut Token| {
            // defaults out of vocab range are disabled
            if *field >= 0 && *field as usize >= n {
                *field = LLAMA_TOKEN_NULL;
            }
            let mut new_id = 0u32;
            if ml.get_u32(key, &mut new_id) {
                if (new_id as usize) >= n {
                    // bad special token: using default id
                } else {
                    *field = new_id as Token;
                }
            }
        };

        load_special(Kv::TOKENIZER_BOS_ID, &mut self.special_bos_id);
        load_special(Kv::TOKENIZER_EOS_ID, &mut self.special_eos_id);
        load_special(Kv::TOKENIZER_EOT_ID, &mut self.special_eot_id);
        load_special(Kv::TOKENIZER_EOM_ID, &mut self.special_eom_id);
        load_special(Kv::TOKENIZER_UNK_ID, &mut self.special_unk_id);
        load_special(Kv::TOKENIZER_SEP_ID, &mut self.special_sep_id);
        load_special(Kv::TOKENIZER_PAD_ID, &mut self.special_pad_id);
        load_special(Kv::TOKENIZER_MASK_ID, &mut self.special_mask_id);
        load_special(Kv::TOKENIZER_FIM_PRE_ID, &mut self.special_fim_pre_id);
        load_special(Kv::TOKENIZER_FIM_SUF_ID, &mut self.special_fim_suf_id);
        load_special(Kv::TOKENIZER_FIM_MID_ID, &mut self.special_fim_mid_id);
        load_special(Kv::TOKENIZER_FIM_PAD_ID, &mut self.special_fim_pad_id);
        load_special(Kv::TOKENIZER_FIM_REP_ID, &mut self.special_fim_rep_id);
        load_special(Kv::TOKENIZER_FIM_SEP_ID, &mut self.special_fim_sep_id);

        // deprecated
        load_special(Kv::TOKENIZER_PREFIX_ID, &mut self.special_fim_pre_id);
        load_special(Kv::TOKENIZER_SUFFIX_ID, &mut self.special_fim_suf_id);
        load_special(Kv::TOKENIZER_MIDDLE_ID, &mut self.special_fim_mid_id);
        drop(load_special);

        // Handle add_bos, add_eos and add_sep
        {
            let mut temp = true;
            if ml.get_bool(Kv::TOKENIZER_ADD_BOS, &mut temp) {
                self.add_bos = temp;
            }
            temp = true;
            if ml.get_bool(Kv::TOKENIZER_ADD_EOS, &mut temp) {
                self.add_eos = temp;
            }
            temp = true;
            if ml.get_bool(Kv::TOKENIZER_ADD_SEP, &mut temp) {
                self.add_sep = temp;
            }
            temp = true;
            if ml.get_bool(Kv::TOKENIZER_DNA_AUTO_TAGS, &mut temp) {
                self.dna_auto_tags = temp;
            }

            // workaround for Gemma 4
            // ref: https://github.com/ggml-org/llama.cpp/pull/21500
            if self.pre_type == VocabPreType::Gemma4 && !self.add_bos {
                self.add_bos = true;
            }
        }

        // Tokenizer normalizer options
        ml.get_bool(Kv::TOKENIZER_NORMALIZER_NFC, &mut self.normalizer_opts.nfc);
        ml.get_bool(Kv::TOKENIZER_NORMALIZER_LOWERCASE, &mut self.normalizer_opts.lowercase);
        self.normalizer_opts.strip_accents = self.normalizer_opts.lowercase;
        ml.get_bool(Kv::TOKENIZER_NORMALIZER_STRIP_ACCENTS, &mut self.normalizer_opts.strip_accents);

        {
            let mut scheme = String::new();
            if ml.get_string(Kv::TOKENIZER_METASPACE_PREPEND_SCHEME, &mut scheme) {
                self.metaspace_prepend_scheme = match scheme.as_str() {
                    "never" => MetaspacePrependScheme::Never,
                    "first" => MetaspacePrependScheme::First,
                    "always" => MetaspacePrependScheme::Always,
                    _ => return Err(Error::new(format!("unknown metaspace prepend_scheme: '{}'", scheme))),
                };
            }
        }

        // suppress tokens
        if let Some(data) = ml.get_i32_array(Kv::TOKENIZER_SUPPRESS_TOKENS) {
            self.suppress_tokens.reserve(data.len());
            // drop out-of-range ids
            for &id in data {
                if id >= 0 && (id as usize) < self.id_to_token.len() {
                    self.suppress_tokens.push(id);
                }
            }
        }

        // auto-detect special tokens by text
        // TODO: convert scripts should provide these tokens through the KV metadata LLM_KV_TOKENIZER_...
        //       for now, we apply this workaround to find the tokens based on their text
        let pairs: Vec<(String, Token)> = self
            .token_to_id
            .iter()
            .map(|(text, id)| (text.clone(), *id))
            .collect();

        for (text, tid) in &pairs {
            if *tid < 0 || *tid as usize >= self.id_to_token.len() {
                continue;
            }
            let attr = &mut self.id_to_token[*tid as usize].attr;

            // find EOT token: "<|eot_id|>", "<|im_end|>", "<end_of_turn>", etc.
            if self.special_eot_id == LLAMA_TOKEN_NULL && EOT_TEXTS.contains(&text.as_str()) {
                self.special_eot_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find EOM token: "<|eom_id|>"
            if self.special_eom_id == LLAMA_TOKEN_NULL && EOM_TEXTS.contains(&text.as_str()) {
                self.special_eom_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find FIM_PRE token: "<|fim_prefix|>", "<fim-prefix>", "<PRE>", etc.
            if self.special_fim_pre_id == LLAMA_TOKEN_NULL && FIM_PRE_TEXTS.contains(&text.as_str()) {
                self.special_fim_pre_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find FIM_SUF token: "<|fim_suffix|>", "<fim-suffix>", "<SUF>", etc.
            if self.special_fim_suf_id == LLAMA_TOKEN_NULL && FIM_SUF_TEXTS.contains(&text.as_str()) {
                self.special_fim_suf_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find FIM_MID token: "<|fim_middle|>", "<fim-middle>", "<MID>", etc.
            if self.special_fim_mid_id == LLAMA_TOKEN_NULL && FIM_MID_TEXTS.contains(&text.as_str()) {
                self.special_fim_mid_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find FIM_PAD token: "<|fim_pad|>", "<fim-pad>", "<PAD>", etc.
            if self.special_fim_pad_id == LLAMA_TOKEN_NULL && FIM_PAD_TEXTS.contains(&text.as_str()) {
                self.special_fim_pad_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find FIM_REP token: "<|fim_repo|>", "<fim-repo>", "<REP>", etc.
            if self.special_fim_rep_id == LLAMA_TOKEN_NULL && FIM_REP_TEXTS.contains(&text.as_str()) {
                self.special_fim_rep_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }

            // find FIM_SEP token: "<|file_sep|>"
            if self.special_fim_sep_id == LLAMA_TOKEN_NULL && FIM_SEP_TEXTS.contains(&text.as_str()) {
                self.special_fim_sep_id = *tid;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }
        }

        // auto-detect unused tokens: e.g. control tokens with the word "unused"
        // ideally, these tokens should be marked as unused during conversion
        {
            for (text, tid) in &pairs {
                if *tid < 0 || *tid as usize >= self.id_to_token.len() {
                    continue;
                }
                let attr = &mut self.id_to_token[*tid as usize].attr;
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    continue;
                }
                if *attr & LLAMA_TOKEN_ATTR_UNUSED == 0 && text.contains("unused") {
                    *attr |= LLAMA_TOKEN_ATTR_UNUSED;
                }
            }
        }

        // maintain a list of tokens that cause end-of-generation
        // this is currently determined based on the token text, which is obviously not ideal
        // ref: https://github.com/ggml-org/llama.cpp/issues/9606
        self.special_eog_ids.clear();
        if self.special_fim_pad_id != LLAMA_TOKEN_NULL {
            self.special_eog_ids.insert(self.special_fim_pad_id);
        }
        if self.special_fim_rep_id != LLAMA_TOKEN_NULL {
            self.special_eog_ids.insert(self.special_fim_rep_id);
        }
        if self.special_fim_sep_id != LLAMA_TOKEN_NULL {
            self.special_eog_ids.insert(self.special_fim_sep_id);
        }

        for (text, tid) in &pairs {
            if *tid < 0 || *tid as usize >= self.id_to_token.len() {
                continue;
            }
            let attr = &mut self.id_to_token[*tid as usize].attr;
            let is_eog_text = EOG_TEXTS.contains(&text.as_str())
                || (self.pre_type == VocabPreType::Flint
                    && (text == "<|user|>" || text == "<|observation|>"));
            if is_eog_text {
                self.special_eog_ids.insert(*tid);
                if *attr & LLAMA_TOKEN_ATTR_CONTROL == 0 {
                    *attr |= LLAMA_TOKEN_ATTR_CONTROL;
                }
            }
        }

        // @ngxson : quick hack for gpt-oss, always render these tokens
        for (text, tid) in &pairs {
            if *tid < 0 || *tid as usize >= self.id_to_token.len() {
                continue;
            }
            if text == "<|channel|>" || text == "<|message|>" || text == "<|start|>" || text == "<|constrain|>" {
                self.id_to_token[*tid as usize].attr = LLAMA_TOKEN_ATTR_USER_DEFINED;
            }
        }

        if self.pre_type == VocabPreType::Soprano {
            self.special_eog_ids.insert(self.special_eos_id);
        }

        // sanity checks
        if self.special_eos_id != LLAMA_TOKEN_NULL {
            self.special_eog_ids.insert(self.special_eos_id);
        }
        if self.special_eot_id != LLAMA_TOKEN_NULL {
            self.special_eog_ids.insert(self.special_eot_id);
        }
        if self.special_eom_id != LLAMA_TOKEN_NULL {
            self.special_eog_ids.insert(self.special_eom_id);
        }

        // TODO: workaround for o200k_harmony and solar-open tokenizer: the "<|end|>" token should not be EOG
        //       we don't have a good way to detect this, so for now, if we have "<|return|>" and "<|call|>" tokens ("<|calls|>" and "<|flush|>" for solar-open),
        //       we remove the "<|end|>" token from the EOG list
        {
            let mut has_return = false;
            let mut has_call = false;
            let mut has_end = false;
            let mut has_flush = false;
            let mut end_id = LLAMA_TOKEN_NULL;

            let eog_ids: Vec<Token> = self.special_eog_ids.iter().copied().collect();
            for tid in eog_ids {
                if tid < 0 || tid as usize >= self.id_to_token.len() {
                    continue;
                }
                let text = self.id_to_token[tid as usize].text.clone();
                if text == "<|return|>" {
                    has_return = true;
                } else if text == "<|call|>" || text == "<|calls|>" {
                    has_call = true;
                } else if text == "<|flush|>" {
                    has_flush = true;
                } else if text == "<|end|>" {
                    has_end = true;
                    end_id = tid;
                }
            }

            if (has_return && has_call && has_end) || (has_call && has_flush && has_end) {
                self.special_eog_ids.remove(&end_id);
                self.id_to_token[end_id as usize].attr = LLAMA_TOKEN_ATTR_USER_DEFINED;
            }
        }

        // gemma4, plamo and beam have a normal </s> token, unlike paddleocr
        {
            // beam's tool call marker is not an EOG token, so it is looked up in the whole vocab
            let mut has_normal_s_marker = self.token_to_id.contains_key("<rfl:parameter name=\"");
            let mut has_s = false;
            let mut s_id = LLAMA_TOKEN_NULL;

            let eog_ids: Vec<Token> = self.special_eog_ids.iter().copied().collect();
            for tid in eog_ids {
                if tid < 0 || tid as usize >= self.id_to_token.len() {
                    continue;
                }
                let text = self.id_to_token[tid as usize].text.clone();
                if text == "<|tool_response>" || text == "<|plamo:eos|>" {
                    has_normal_s_marker = true;
                } else if text == "</s>" {
                    has_s = true;
                    s_id = tid;
                }
            }

            if has_normal_s_marker && has_s {
                self.special_eog_ids.remove(&s_id);
                self.id_to_token[s_id as usize].attr = LLAMA_TOKEN_ATTR_NORMAL;
            }
        }

        // build special tokens cache
        {
            for id in 0..self.id_to_token.len() as Token {
                let attr = self.id_to_token[id as usize].attr;
                if attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN) != 0 {
                    self.cache_special_tokens.push(id);
                }
            }

            let id_to_token = &self.id_to_token;
            self.cache_special_tokens.sort_by(|&a, &b| {
                id_to_token[b as usize].text.len().cmp(&id_to_token[a as usize].text.len())
            });
        }

        // build token to piece cache
        {
            let mut cache = vec![String::new(); self.id_to_token.len()];
            for id in 0..self.id_to_token.len() as Token {
                cache[id as usize] = self.token_to_piece_for_cache(id, true);
            }
            self.cache_token_to_piece = cache;
        }

        // Handle per token attributes
        //NOTE: Each model customizes per token attributes.
        //NOTE: Per token attributes are missing from the GGUF file.
        //TODO: Extract attributes from GGUF file.
        {
            fn contains_any(str: &str, substrs: &[&str]) -> bool {
                substrs.iter().any(|s| str.contains(s))
            }

            let mut model_name = String::new();
            let mut tokenizer_pre = String::new();
            let mut general_arch = String::new();
            ml.get_string(Kv::GENERAL_NAME, &mut model_name);
            ml.get_string(Kv::TOKENIZER_PRE, &mut tokenizer_pre);
            ml.get_string(Kv::GENERAL_ARCHITECTURE, &mut general_arch);

            // model name to lowercase
            model_name.make_ascii_lowercase();

            // set attributes by model/tokenizer/architecture name
            if contains_any(&tokenizer_pre, &["jina-v2-de", "jina-v2-es", "jina-v2-code", "mmbert"])
                || contains_any(&general_arch, &["nomic-bert-moe", "jina-bert-v3"])
            {
                let id = self.token_to_id.get("<mask>").copied();
                if let Some(id) = id {
                    self.set_tokenid_attr(id, LLAMA_TOKEN_ATTR_LSTRIP, true);
                }
            } else if contains_any(&model_name, &["phi-3", "phi3"]) {
                let specials = self.cache_special_tokens.clone();
                for id in specials {
                    self.set_tokenid_attr(id, LLAMA_TOKEN_ATTR_RSTRIP, true);
                }
                for token in ["</s>"] {
                    if let Some(&id) = self.token_to_id.get(token) {
                        self.set_tokenid_attr(id, LLAMA_TOKEN_ATTR_RSTRIP, true);
                    }
                }
                for token in ["<unk>", "<s>", "<|endoftext|>"] {
                    if let Some(&id) = self.token_to_id.get(token) {
                        self.set_tokenid_attr(id, LLAMA_TOKEN_ATTR_RSTRIP, false);
                    }
                }
            } else if contains_any(&model_name, &["modern-bert"]) {
                if let Some(&id) = self.token_to_id.get("[MASK]") {
                    self.set_tokenid_attr(id, LLAMA_TOKEN_ATTR_LSTRIP, true);
                }
            }
        }

        Ok(())
    }

    pub(crate) fn set_tokenid_attr(&mut self, id: Token, attr: LlamaTokenAttr, value: bool) {
        if id < 0 || id as usize >= self.id_to_token.len() {
            return;
        }
        let current = self.id_to_token[id as usize].attr;
        self.id_to_token[id as usize].attr = if value {
            current | attr
        } else {
            current & !attr
        };
    }
}


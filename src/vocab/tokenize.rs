//! Tokenization path and its helpers.

use super::*;
use crate::core::{Error, Result, Token, LLAMA_TOKEN_NULL};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

#[derive(Clone, Debug)]
enum Fragment<'a> {
    Token(Token),
    RawText {
        raw_text: &'a str,
        offset: usize,
        length: usize,
    },
}

// ---------------------------------------------------------------------------
// (de-) tokenize
// ---------------------------------------------------------------------------

enum BpeKind<'a> {
    Plain(BpeSession<'a>),
    Dna(HybridDnaSession<'a>),
    Whitespace(WhitespaceSession<'a>),
}

struct AnyBpeSession<'a> {
    vocab: &'a Impl,
    kind: BpeKind<'a>,
}

impl<'a> AnyBpeSession<'a> {
    pub(crate) fn append_bos(&self, output: &mut Vec<Token>) -> bool {
        if self.vocab.get_add_bos() {
            output.push(self.vocab.token_bos());
            return true;
        }
        false
    }

    pub(crate) fn append_eos(&self, output: &mut Vec<Token>) -> bool {
        if self.vocab.get_add_eos() {
            output.push(self.vocab.token_eos());
            return true;
        }
        false
    }

    pub(crate) fn check_double_bos_eos(&self, output: &[Token]) {
        let _ = self.vocab.get_add_bos() && output.get(1) == Some(&self.vocab.token_bos());
        let _ = self.vocab.get_add_eos()
            && output.len() >= 2
            && output.get(output.len() - 2) == Some(&self.vocab.token_eos());
    }

    pub(crate) fn tokenize(&mut self, text: &str, output: &mut Vec<Token>) {
        match &mut self.kind {
            BpeKind::Plain(session) => session.tokenize(text, output),
            BpeKind::Dna(session) => session.tokenize(text, output),
            BpeKind::Whitespace(session) => session.tokenize(text, output),
        }
    }
}

impl Impl {
    pub(crate) fn tokenizer_st_partition(&self, buffer: &mut Vec<Fragment<'_>>, parse_special: bool) {
        // for each special token
        for special_id in self.cache_special_tokens.clone() {
            let (text, attr) = {
                let data = &self.id_to_token[special_id as usize];
                (data.text.clone(), data.attr)
            };
            if text.is_empty() {
                continue;
            }

            if !parse_special && attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN) != 0 {
                // Ignore control and unknown tokens when parse_special == false
                // User-defined tokens are still pre-tokenized before everything else
                // ref: https://github.com/huggingface/tokenizers/blob/fdd26ba9a3f0c133427aab0423888cbde91362d7/tokenizers/src/tokenizer/mod.rs#L726
                // This is mostly relevant for neox-style tokenizers (mpt, olmo, stablelm, etc.)
                continue;
            }

            let mut it = 0usize;
            while it < buffer.len() {
                // if a fragment is text (not yet processed)
                let (raw_text, base_offset, base_length) = match &buffer[it] {
                    Fragment::RawText { raw_text, offset, length } => (*raw_text, *offset, *length),
                    Fragment::Token(_) => {
                        it += 1;
                        continue;
                    }
                };

                let mut raw_text_base_offset = base_offset;
                let mut raw_text_base_length = base_length;

                // loop over the text
                loop {
                    // find the first occurrence of a given special token in this fragment
                    // passing offset argument only limits the "search area" but match
                    // coordinates are still relative to the source full raw_text
                    let end = (raw_text_base_offset + raw_text_base_length).min(raw_text.len());
                    let from = raw_text_base_offset.min(end);
                    let rel = match raw_text[from..end].find(text.as_str()) {
                        Some(rel) => rel,
                        None => break, // no occurrences found, stop for this special token
                    };
                    let match_pos = from + rel;
                    let source = it;

                    // if match is further than base offset
                    // then we have some text to the left of it
                    if match_pos > raw_text_base_offset {
                        // left
                        let left_reminder_offset = raw_text_base_offset;
                        let mut left_reminder_length = match_pos - raw_text_base_offset;

                        if attr & LLAMA_TOKEN_ATTR_LSTRIP != 0 {
                            while left_reminder_length > 0
                                && raw_text.as_bytes()[left_reminder_offset + left_reminder_length - 1]
                                    .is_ascii_whitespace()
                            {
                                left_reminder_length -= 1;
                            }
                        }

                        if left_reminder_length > 0 {
                            buffer.insert(
                                it + 1,
                                Fragment::RawText {
                                    raw_text,
                                    offset: left_reminder_offset,
                                    length: left_reminder_length,
                                },
                            );
                            it += 1;
                        }
                    }

                    // special token
                    buffer.insert(it + 1, Fragment::Token(special_id));
                    it += 1;

                    // right
                    if match_pos + text.len() < end {
                        let mut right_reminder_offset = match_pos + text.len();
                        let mut right_reminder_length =
                            raw_text_base_length - ((match_pos - raw_text_base_offset) + text.len());

                        if attr & LLAMA_TOKEN_ATTR_RSTRIP != 0 {
                            while right_reminder_length > 0
                                && raw_text.as_bytes()[right_reminder_offset].is_ascii_whitespace()
                            {
                                right_reminder_offset += 1;
                                right_reminder_length -= 1;
                            }
                        }

                        if right_reminder_length > 0 {
                            buffer.insert(
                                it + 1,
                                Fragment::RawText {
                                    raw_text,
                                    offset: right_reminder_offset,
                                    length: right_reminder_length,
                                },
                            );
                            it += 1;
                        }

                        buffer.remove(source);
                        it = it.saturating_sub(1);

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;
                    } else {
                        buffer.remove(source);
                        it = it.saturating_sub(1);
                        break;
                    }
                }
                it += 1;
            }
        }
    }

    pub(crate) fn tokenize(&self, raw_text: &str, add_special: bool, parse_special: bool) -> Vec<Token> {
        let mut output = Vec::new();
        let mut fragment_buffer: Vec<Fragment<'_>> = Vec::new();

        if !raw_text.is_empty() {
            fragment_buffer.push(Fragment::RawText { raw_text, offset: 0, length: raw_text.len() });
            self.tokenizer_st_partition(&mut fragment_buffer, parse_special);
        }

        let prepend_first = self.metaspace_prepend_scheme == MetaspacePrependScheme::First;

        match self.vtype {
            VocabType::Spm => {
                // OG tokenizer behavior:
                //
                // tokenizer.encode('', add_special_tokens=True)  returns [1]
                // tokenizer.encode('', add_special_tokens=False) returns []
                let mut prepend_space = self.add_space_prefix;

                if add_special && self.add_bos {
                    output.push(self.special_bos_id);
                }

                for (idx, fragment) in fragment_buffer.iter().enumerate() {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            let mut text = raw_text[*offset..*offset + *length].to_string();
                            llama_escape_whitespace(&mut text);

                            if prepend_space && (!prepend_first || !text.starts_with(ESCAPED_SPACE)) {
                                text.insert_str(0, ESCAPED_SPACE);
                            }

                            SpmSession::new(self).tokenize(&text, &mut output);
                            prepend_space = false;
                        }
                        Fragment::Token(id) => {
                            output.push(*id);
                            prepend_space = self.add_space_prefix && !prepend_first;
                        }
                    }
                    let _ = idx;
                }

                if add_special && self.add_eos {
                    output.push(self.special_eos_id);
                }
            }
            VocabType::Bpe => {
                let tokenizer = self.tokenizer.bpe();
                let mut session = if self.tokenizer_model == "hybriddna" {
                    AnyBpeSession {
                        vocab: self,
                        kind: BpeKind::Dna(HybridDnaSession::new(self, tokenizer)),
                    }
                } else if self.tokenizer_model == "whitespace" {
                    AnyBpeSession {
                        vocab: self,
                        kind: BpeKind::Whitespace(WhitespaceSession {
                            bpe: BpeSession::new(self, tokenizer),
                            vocab: self,
                        }),
                    }
                } else {
                    AnyBpeSession {
                        vocab: self,
                        kind: BpeKind::Plain(BpeSession::new(self, tokenizer)),
                    }
                };

                if add_special {
                    session.append_bos(&mut output);
                }
                for (idx, fragment) in fragment_buffer.iter().enumerate() {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            let mut text = raw_text[*offset..*offset + *length].to_string();

                            if self.escape_whitespaces {
                                llama_escape_whitespace(&mut text);
                            }

                            // Metaspace prepend_scheme "first": add U+2581 only at the start of the input
                            if prepend_first && idx == 0 && !text.starts_with(ESCAPED_SPACE) {
                                text.insert_str(0, ESCAPED_SPACE);
                            }

                            session.tokenize(&text, &mut output);
                        }
                        Fragment::Token(id) => {
                            BpeSession::append(*id, &mut output);
                        }
                    }
                }

                if add_special {
                    session.append_eos(&mut output);
                    session.check_double_bos_eos(&output);
                }
            }
            VocabType::Wpm => {
                if add_special {
                    output.push(self.special_bos_id);
                }

                let session = WpmSession { vocab: self };
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            session.tokenize(&raw_text[*offset..*offset + *length], &mut output);
                        }
                        Fragment::Token(id) => output.push(*id),
                    }
                }

                if add_special {
                    output.push(self.special_sep_id);
                }
            }
            VocabType::Ugm => {
                if add_special && self.add_bos {
                    output.push(self.special_bos_id);
                }
                // TODO: honor metaspace_prepend_scheme
                let session = UgmSession { vocab: self, tokenizer: self.tokenizer.ugm() };
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            session.tokenize(&raw_text[*offset..*offset + *length], &mut output);
                        }
                        Fragment::Token(id) => output.push(*id),
                    }
                }

                if add_special && self.add_eos {
                    output.push(self.special_eos_id);
                }
            }
            VocabType::Rwkv => {
                let session = RwkvSession { vocab: self, tokenizer: self.tokenizer.rwkv() };
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            session.tokenize(&raw_text[*offset..*offset + *length], &mut output);
                        }
                        Fragment::Token(id) => output.push(*id),
                    }
                }
            }
            VocabType::Plamo2 | VocabType::Plamo3 => {
                if add_special && self.add_bos {
                    output.push(self.special_bos_id);
                }
                let session = Plamo2Session { tokenizer: self.tokenizer.plamo2() };
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            session.tokenize(&raw_text[*offset..*offset + *length], &mut output);
                        }
                        Fragment::Token(id) => output.push(*id),
                    }
                }

                if add_special && self.add_eos {
                    output.push(self.special_eos_id);
                }
            }
            VocabType::Test => {
                const CHUNK_SIZE: usize = 5;
                let n_vocab = self.n_tokens().max(1) as u64;

                let mut n_tokens = 0usize;
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { length, .. } => {
                            n_tokens += (length + CHUNK_SIZE - 1) / CHUNK_SIZE;
                        }
                        Fragment::Token(_) => n_tokens += 1,
                    }
                }
                output.reserve(n_tokens);

                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { raw_text, offset, length } => {
                            let bytes = raw_text.as_bytes();
                            let begin = *offset;
                            let end = begin + *length;
                            let mut pos = begin;
                            while pos < end {
                                let n = CHUNK_SIZE.min(end - pos);
                                let mut hash = 0u64;
                                for i in 0..n {
                                    hash = hash * 31 + bytes[pos + i] as u64;
                                }
                                output.push((hash % n_vocab) as Token);
                                pos += n;
                            }
                        }
                        Fragment::Token(id) => output.push(*id),
                    }
                }
            }
            VocabType::None => {
                // fatal error in the original; nothing to tokenize without a vocab
            }
        }

        output
    }

    pub(crate) fn token_to_byte(&self, id: Token) -> u8 {
        if id < 0 || id as usize >= self.id_to_token.len() {
            return 0;
        }
        let text = &self.id_to_token[id as usize].text;
        if text.len() < 5 {
            return 0;
        }
        parse_hex_byte(&text[3..5])
    }

}


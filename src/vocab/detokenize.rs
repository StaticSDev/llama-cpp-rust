//! Detokenization path and its helpers.

use super::*;
use crate::core::{Token, LLAMA_TOKEN_NULL};

pub(crate) fn llama_escape_whitespace(text: &mut String) {
    replace_all(text, " ", ESCAPED_SPACE);
}

pub(crate) fn llama_unescape_whitespace(word: &mut String) {
    replace_all(word, ESCAPED_SPACE, " ");
}

pub(crate) fn llama_decode_text(text: &str) -> String {
    let mut decoded: Vec<u8> = Vec::with_capacity(text.len());
    let cpts = unicode_cpts_from_utf8(text);
    for cpt in cpts {
        match unicode_cpt_to_byte(cpt) {
            Ok(byte) => decoded.push(byte),
            Err(_) => {
                decoded.extend_from_slice(b"[UNK_BYTE_0x");
                for &b in unicode_cpt_to_utf8(cpt).as_bytes() {
                    decoded.extend_from_slice(format!("{:02x}", b).as_bytes());
                }
                decoded.extend_from_slice(text.as_bytes());
                decoded.push(b']');
            }
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

pub(crate) fn try_copy(src: &[u8], buf: &mut [u8], lstrip: i32) -> i32 {
    let mut bytes = src;
    let mut n = 0i32;
    while n < lstrip && !bytes.is_empty() && bytes[0] == b' ' {
        bytes = &bytes[1..];
        n += 1;
    }
    if buf.len() < bytes.len() {
        return -(bytes.len() as i32);
    }
    buf[..bytes.len()].copy_from_slice(bytes);
    bytes.len() as i32
}

impl Impl {
    pub(crate) fn token_to_piece_for_cache(&self, token: Token, special: bool) -> String {
        let mut buf = vec![0u8; 64];
        let n_chars = self.token_to_piece(token, &mut buf, 0, special);
        if n_chars < 0 {
            buf.resize((-n_chars) as usize, 0);
            let _check = self.token_to_piece(token, &mut buf, 0, special);
        } else {
            buf.resize(n_chars as usize, 0);
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    // does not write a null terminator to buf
    pub(crate) fn token_to_piece(&self, token: Token, buf: &mut [u8], lstrip: i32, special: bool) -> i32 {
        // ref: https://github.com/ggml-org/llama.cpp/pull/7587#discussion_r1620983843
        let attr_special = LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_CONTROL;
        let attr = if token >= 0 && (token as usize) < self.id_to_token.len() {
            self.token_get_attr(token)
        } else {
            0
        };
        if !special && attr & attr_special != 0 {
            return 0;
        }

        // if we have a cache - use it
        if !self.cache_token_to_piece.is_empty() {
            if let Some(cached) = self.cache_token_to_piece.get(token as usize) {
                return try_copy(cached.as_bytes(), buf, lstrip);
            }
            return 0;
        }

        if token < 0 || token as usize >= self.id_to_token.len() {
            return 0;
        }
        let token_text = &self.id_to_token[token as usize].text;

        let piece: Vec<u8>;
        let mut apply_lstrip = true;

        match self.vtype {
            VocabType::Wpm | VocabType::Spm | VocabType::Ugm => {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED) != 0 {
                    piece = token_text.as_bytes().to_vec();
                } else if attr & LLAMA_TOKEN_ATTR_NORMAL != 0 {
                    let mut result = token_text.clone();
                    llama_unescape_whitespace(&mut result);
                    piece = result.into_bytes();
                } else if attr & LLAMA_TOKEN_ATTR_BYTE != 0 {
                    piece = vec![self.token_to_byte(token)];
                } else {
                    piece = Vec::new();
                }
            }
            VocabType::Bpe => {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED) != 0 {
                    piece = token_text.as_bytes().to_vec();
                } else if attr & LLAMA_TOKEN_ATTR_NORMAL != 0 {
                    if self.pre_type == VocabPreType::Soprano {
                        piece = token_text.as_bytes().to_vec();
                    } else if self.escape_whitespaces {
                        // SPM-style BPE: tokens contain U+2581 for spaces
                        let mut result = token_text.clone();
                        llama_unescape_whitespace(&mut result);
                        piece = result.into_bytes();
                    } else {
                        piece = llama_decode_text(token_text).into_bytes();
                    }
                } else if attr & LLAMA_TOKEN_ATTR_BYTE != 0 {
                    piece = vec![self.token_to_byte(token)];
                } else {
                    piece = Vec::new();
                }
            }
            VocabType::Rwkv => {
                piece = llama_unescape_rwkv_token(token_text);
                apply_lstrip = false;
            }
            VocabType::Test => {
                // tokens -> text: simply stringify the token id in hex
                piece = format!("{:x}", token).into_bytes();
            }
            VocabType::Plamo2 | VocabType::Plamo3 => {
                // PLaMo uses similar token handling as BPE/SPM
                let bytes = token_text.as_bytes();
                if attr & LLAMA_TOKEN_ATTR_BYTE != 0
                    && bytes.len() == 6
                    && &bytes[..3] == b"<0x"
                    && bytes[5] == b'>'
                    && bytes[3].is_ascii_hexdigit()
                    && bytes[4].is_ascii_hexdigit()
                {
                    piece = vec![parse_hex_byte(&token_text[3..5])];
                } else {
                    // Normal token - just copy the text
                    piece = token_text.as_bytes().to_vec();
                }
            }
            VocabType::None => piece = Vec::new(),
        }

        if apply_lstrip {
            try_copy(&piece, buf, lstrip)
        } else {
            if piece.len() > buf.len() {
                return -(piece.len() as i32);
            }
            buf[..piece.len()].copy_from_slice(&piece);
            piece.len() as i32
        }
    }

    pub(crate) fn detokenize(
        &self,
        tokens: &[Token],
        text: &mut [u8],
        remove_special: bool,
        unparse_special: bool,
    ) -> i32 {
        if self.vtype == VocabType::None {
            return 0;
        }

        let mut n_tokens = tokens.len();
        let mut skip = 0usize;

        // remove the leading space
        let mut remove_space = self.add_space_prefix;

        if remove_special && self.add_bos {
            if n_tokens > 0 && tokens[skip] == self.special_bos_id {
                remove_space = false;
                n_tokens -= 1;
                skip += 1;
            }
        }

        if remove_special && self.add_eos {
            if n_tokens > 0 && tokens[skip + n_tokens - 1] == self.special_eos_id {
                n_tokens -= 1;
            }
        }

        // Metaspace prepend_scheme "first": remove the leading space of the first
        // rendered piece, so a skipped BOS does not consume it
        // TODO: honor metaspace_prepend_scheme for the other vocab types
        let mut remove_first_space = (self.vtype == VocabType::Bpe || self.vtype == VocabType::Spm)
            && self.metaspace_prepend_scheme == MetaspacePrependScheme::First;

        let mut rendered: Vec<u8> = Vec::new();
        let mut piece_buf = vec![0u8; 4096];

        for i in 0..n_tokens {
            let token = tokens[skip + i];
            let n_chars = self.token_to_piece(
                token,
                &mut piece_buf,
                (remove_space || remove_first_space) as i32,
                unparse_special,
            );
            remove_space = false;
            if unparse_special
                || (token >= 0
                    && (token as usize) < self.id_to_token.len()
                    && self.token_get_attr(token) & (LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_CONTROL) == 0)
            {
                remove_first_space = false;
            }
            if n_chars < 0 {
                piece_buf.resize((-n_chars) as usize, 0);
                let check = self.token_to_piece(
                    token,
                    &mut piece_buf,
                    (remove_space || remove_first_space) as i32,
                    unparse_special,
                );
                if check > 0 {
                    rendered.extend_from_slice(&piece_buf[..check as usize]);
                }
            } else if n_chars > 0 {
                if n_chars as usize > piece_buf.len() {
                    piece_buf.resize(n_chars as usize, 0);
                    let check = self.token_to_piece(
                        token,
                        &mut piece_buf,
                        (remove_space || remove_first_space) as i32,
                        unparse_special,
                    );
                    if check > 0 {
                        rendered.extend_from_slice(&piece_buf[..check as usize]);
                    }
                } else {
                    rendered.extend_from_slice(&piece_buf[..n_chars as usize]);
                }
            }
        }

        let mut total = rendered.len();

        if self.clean_spaces && total > 0 {
            // first pass: characters ?!.,  //TODO: where do these characters come from?
            let total1 = total;
            total = 1;
            for i in 1..total1 {
                let x = rendered[i];
                if rendered[i - 1] == b' ' && matches!(x, b'?' | b'!' | b'.' | b',') {
                    total -= 1; // remove space
                }
                rendered[total] = x;
                total += 1;
            }

            // second pass: strip single apostrophe between spaces
            let total2 = total;
            total = 1;
            let mut i = 1;
            while i < total2 {
                let x = rendered[i];
                if x == b'\'' && i + 1 < total2 && rendered[i - 1] == b' ' && rendered[i + 1] == b' ' {
                    total -= 1; // remove prev space
                    if i + 1 < rendered.len() {
                        rendered[i + 1] = 0; // remove next space
                    }
                    i += 1;
                }
                rendered[total] = x;
                total += 1;
                i += 1;
            }

            // third pass: apostrophe contractions  //NOTE: this makes sense?
            let total3 = total;
            total = 1;
            for i in 1..total3 {
                let x = rendered[i];
                if rendered[i - 1] == b' ' && x == b'\'' && i + 1 < total3 {
                    let x1 = rendered[i + 1];
                    if x1 == b's' || x1 == b'm' {
                        total -= 1; // remove space
                    } else if i + 2 < total3 {
                        let x2 = rendered[i + 2];
                        if (x1 == b'r' && x2 == b'e') || (x1 == b'v' && x2 == b'e') {
                            total -= 1; // remove space
                        }
                    }
                }
                rendered[total] = x;
                total += 1;
            }
        }

        if total > text.len() {
            return -(total as i32);
        }
        text[..total].copy_from_slice(&rendered[..total]);
        total as i32
    }

    pub(crate) fn print_info(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("vocab type            = {}\n", self.type_name()));
        out.push_str(&format!("n_vocab               = {}\n", self.n_tokens()));
        out.push_str(&format!("n_merges              = {}\n", self.bpe_ranks.len()));

        // special tokens
        let specials = [
            ("BOS token", self.special_bos_id),
            ("EOS token", self.special_eos_id),
            ("EOT token", self.special_eot_id),
            ("EOM token", self.special_eom_id),
            ("UNK token", self.special_unk_id),
            ("SEP token", self.special_sep_id),
            ("PAD token", self.special_pad_id),
            ("MASK token", self.special_mask_id),
            ("LF token", self.linefeed_id),
            ("FIM PRE token", self.special_fim_pre_id),
            ("FIM SUF token", self.special_fim_suf_id),
            ("FIM MID token", self.special_fim_mid_id),
            ("FIM PAD token", self.special_fim_pad_id),
            ("FIM REP token", self.special_fim_rep_id),
            ("FIM SEP token", self.special_fim_sep_id),
        ];
        for (label, id) in specials {
            if id != LLAMA_TOKEN_NULL && id >= 0 && (id as usize) < self.id_to_token.len() {
                out.push_str(&format!(
                    "{: <21}= {} '{}'\n",
                    format!("{} ", label),
                    id,
                    self.id_to_token[id as usize].text
                ));
            }
        }

        for &id in &self.special_eog_ids {
            if id >= 0 && (id as usize) < self.id_to_token.len() {
                out.push_str(&format!(
                    "{: <21}= {} '{}'\n",
                    "EOG token ",
                    id,
                    self.id_to_token[id as usize].text
                ));
            }
        }

        out.push_str(&format!("max token length      = {}\n", self.max_token_len));
        out
    }

    pub(crate) fn type_name(&self) -> &'static str {
        match self.vtype {
            VocabType::None => "no vocab",
            VocabType::Spm => "SPM",
            VocabType::Bpe => "BPE",
            VocabType::Wpm => "WPM",
            VocabType::Ugm => "UGM",
            VocabType::Rwkv => "RWKV",
            VocabType::Plamo2 => "PLaMo2",
            VocabType::Plamo3 => "PLaMo3",
            VocabType::Test => "TEST",
        }
    }
}

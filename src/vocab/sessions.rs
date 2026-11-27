//! Tokenizer implementations and their sessions.

use super::*;
use crate::core::{Error, Result, Token, LLAMA_TOKEN_NULL};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

// ---------------------------------------------------------------------------
// tokenizer state and sessions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct LlmSymbol {
    pub(crate) prev: i32,
    pub(crate) next: i32,
    pub(crate) text: Vec<u8>,
    pub(crate) n: usize,
}

impl LlmSymbol {
    pub(crate) fn new(text: &[u8]) -> Self {
        Self { prev: -1, next: -1, text: text.to_vec(), n: text.len() }
    }

    fn merge_right(&mut self, right: &LlmSymbol) {
        self.text.extend_from_slice(&right.text[..right.n]);
        self.n = self.text.len();
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.text[..self.n]).unwrap_or("")
    }
}

struct SpmBigram {
    pub(crate) left: i32,
    pub(crate) right: i32,
    pub(crate) score: f32,
    pub(crate) size: usize,
}

impl PartialEq for SpmBigram {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left
    }
}
impl Eq for SpmBigram {}
impl Ord for SpmBigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // priority_queue top: lowest score, then lowest left index
        other.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}
impl PartialOrd for SpmBigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct BpeBigram {
    pub(crate) left: i32,
    pub(crate) right: i32,
    pub(crate) rank: i32,
    pub(crate) size: usize,
}

impl PartialEq for BpeBigram {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.left == other.left
    }
}
impl Eq for BpeBigram {}
impl Ord for BpeBigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // priority_queue top: lowest rank, then lowest left index
        other.rank.cmp(&self.rank).then_with(|| other.left.cmp(&self.left))
    }
}
impl PartialOrd for BpeBigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BpeData {
    pub(crate) regex_exprs: Vec<String>,
    pub(crate) byte_encode: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct UgmData {
    pub(crate) prefix_replacements: Vec<u8>,
    pub(crate) xcda_array: Vec<u32>,
    pub(crate) token_matcher: NaiveTrie,
    pub(crate) user_defined_token_matcher: NaiveTrie,
    pub(crate) min_score: f32,
    pub(crate) max_score: f32,
    pub(crate) unknown_token_score: f32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RwkvData {
    pub(crate) token_matcher: NaiveTrie,
}

#[derive(Clone, Debug)]
pub(crate) struct Plamo2Data {
    pub(crate) tokens: Vec<String>,
    pub(crate) bytes: Vec<Token>,
    pub(crate) to_suffix_id: HashMap<i64, i32>,
    pub(crate) table: Vec<[i32; 4]>,
    pub(crate) pre_segment: bool,
}

pub(crate) enum Tokenizer {
    Spm,
    Bpe(BpeData),
    Wpm,
    Ugm(UgmData),
    Rwkv(RwkvData),
    Plamo2(Plamo2Data),
    Test,
}

impl Tokenizer {
    pub(crate) fn bpe(&self) -> &BpeData {
        match self {
            Tokenizer::Bpe(data) => data,
            _ => unreachable!("BPE tokenizer data required"),
        }
    }
    pub(crate) fn ugm(&self) -> &UgmData {
        match self {
            Tokenizer::Ugm(data) => data,
            _ => unreachable!("UGM tokenizer data required"),
        }
    }
    pub(crate) fn rwkv(&self) -> &RwkvData {
        match self {
            Tokenizer::Rwkv(data) => data,
            _ => unreachable!("RWKV tokenizer data required"),
        }
    }
    pub(crate) fn plamo2(&self) -> &Plamo2Data {
        match self {
            Tokenizer::Plamo2(data) => data,
            _ => unreachable!("PLaMo2 tokenizer data required"),
        }
    }
}

// SPM tokenizer
// original implementation:
// https://github.com/ggml-org/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
pub(crate) struct SpmSession<'a> {
    pub(crate) vocab: &'a Impl,
    pub(crate) symbols: Vec<LlmSymbol>,
    pub(crate) work_queue: BinaryHeap<SpmBigram>,
    pub(crate) rev_merge: HashMap<Vec<u8>, (i32, i32)>,
}

impl<'a> SpmSession<'a> {
    pub(crate) fn new(vocab: &'a Impl) -> Self {
        Self {
            vocab,
            symbols: Vec::new(),
            work_queue: BinaryHeap::new(),
            rev_merge: HashMap::new(),
        }
    }

    pub(crate) fn tokenize(&mut self, text: &str, output: &mut Vec<Token>) {
        self.symbols.clear();
        self.work_queue.clear();
        self.rev_merge.clear();

        // split string into utf8 chars
        let bytes = text.as_bytes();
        let mut offs = 0usize;
        let mut index: i32 = 0;
        while offs < bytes.len() {
            let len = unicode_len_utf8(bytes[offs]);
            let n = len.min(bytes.len() - offs);
            let mut sym = LlmSymbol::new(&bytes[offs..offs + n]);
            offs += n;
            sym.prev = index - 1;
            sym.next = if offs == bytes.len() { -1 } else { index + 1 };
            self.symbols.push(sym);
            index += 1;
        }

        // seed the work queue with all possible 2-character tokens
        for i in 1..self.symbols.len() as i32 {
            self.try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can
        while let Some(bigram) = self.work_queue.pop() {
            let left = bigram.left as usize;
            let right = bigram.right as usize;
            let left_n = self.symbols[left].n;
            let right_n = self.symbols[right].n;

            // if one of the symbols already got merged, skip it
            if left_n == 0 || right_n == 0 || left_n + right_n != bigram.size {
                continue;
            }

            // merge the right sym into the left one
            let right_sym = self.symbols[right].clone();
            self.symbols[left].merge_right(&right_sym);
            self.symbols[right].n = 0;

            // remove the right sym from the chain
            let next = self.symbols[right].next;
            self.symbols[left].next = next;
            if next >= 0 {
                self.symbols[next as usize].prev = bigram.left;
            }

            // find more substitutions
            let prev = self.symbols[left].prev;
            self.try_add_bigram(prev, bigram.left);
            self.try_add_bigram(bigram.left, next);
        }

        let mut i = 0i32;
        while i != -1 {
            self.resegment(i as usize, output);
            i = self.symbols[i as usize].next;
        }
    }

    // no recursion: the merges of a long token can be deep
    fn resegment(&self, index: usize, output: &mut Vec<Token>) {
        let mut todo = vec![index];
        while let Some(idx) = todo.pop() {
            let cur = &self.symbols[idx];
            if cur.n == 0 {
                continue;
            }
            let text = cur.as_str();

            // Do we need to support is_unused?
            let token = self.vocab.text_to_token(text);
            if token != LLAMA_TOKEN_NULL {
                output.push(token);
                continue;
            }

            if let Some(&(first, second)) = self.rev_merge.get(cur.text.as_slice()) {
                todo.push(second as usize);
                todo.push(first as usize);
                continue;
            }

            // output any symbols that did not form tokens as bytes
            for &byte in &cur.text[..cur.n] {
                output.push(self.vocab.byte_to_token(byte));
            }
        }
    }

    fn try_add_bigram(&mut self, left: i32, right: i32) {
        if left == -1 || right == -1 {
            return;
        }
        let l = left as usize;
        let r = right as usize;
        let mut text = self.symbols[l].text[..self.symbols[l].n].to_vec();
        text.extend_from_slice(&self.symbols[r].text[..self.symbols[r].n]);

        let text_str = String::from_utf8_lossy(&text).into_owned();
        let token = self.vocab.text_to_token(&text_str);
        if token == LLAMA_TOKEN_NULL {
            return;
        }
        if token as u32 >= self.vocab.n_tokens() {
            return;
        }
        let score = self.vocab.token_get_score(token);
        self.work_queue.push(SpmBigram {
            left,
            right,
            score,
            size: text.len(),
        });

        // Do we need to support is_unused?
        self.rev_merge.insert(text, (left, right));
    }
}

// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!
pub(crate) fn bpe_regex_exprs(pre_type: VocabPreType) -> (Vec<String>, bool) {
    let llama3 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    let cjk = "[一-龥぀-ゟ゠-ヿ]+";
    let gpt2 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
    let punct = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    let tekken = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    let gpt4o = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    let newlines = "[^\\n]+|[\\n]+";
    let mut byte_encode = true;
    let exprs: Vec<String> = match pre_type {
        VocabPreType::Flint => vec![
            "\\p{N}{1,3}".into(),
            cjk.into(),
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Llama3 => vec![llama3.into()],
        VocabPreType::Jais2 => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}".into(),
        ],
        VocabPreType::Dbrx | VocabPreType::Smaug => vec![llama3.into()],
        VocabPreType::DeepseekLlm => vec![
            "[\r\n]".into(),
            "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+".into(),
            "\\s?[!-/:-~！-／：-～‘-‟　-。]+".into(),
            "\\s+$".into(),
            "[一-龥ࠀ-一가-퟿]+".into(),
            "\\p{N}+".into(),
        ],
        VocabPreType::Deepseek3Llm | VocabPreType::HunyuanDense | VocabPreType::JoyaiLlm | VocabPreType::HyV4 => vec![
            "\\p{N}{1,3}".into(), cjk.into(), punct.into(),
        ],
        VocabPreType::Spark2p5 => vec![
            "\\p{N}{1,3}".into(), cjk.into(),
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\\r\\n]|\\s+(?!\\S)|\\s+".into(),
            "\\p{N}".into(),
        ],
        VocabPreType::Youtu => vec![
            "[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+".into(),
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::DeepseekCoder => vec![
            "[\r\n]".into(), "\\s?\\p{L}+".into(), "\\s?\\p{P}+".into(),
            "[一-龥ࠀ-一가-퟿]+".into(), "\\p{N}".into(),
        ],
        VocabPreType::Falcon => vec![
            "[\\p{P}\\$\\+<=>\\^~\\|`]+".into(),
            gpt2.into(),
            "[0-9][0-9][0-9]".into(),
        ],
        VocabPreType::Starcoder | VocabPreType::Refact | VocabPreType::CommandR
        | VocabPreType::Smallm | VocabPreType::Codeshell | VocabPreType::Exaone
        | VocabPreType::Minerva | VocabPreType::Mellum2 => vec!["\\p{N}".into(), gpt2.into()],
        VocabPreType::Gpt2 | VocabPreType::Mpt | VocabPreType::Olmo | VocabPreType::Jais
        | VocabPreType::Trillion | VocabPreType::GraniteDocling => vec![gpt2.into()],
        VocabPreType::StableLm2 | VocabPreType::Qwen2 | VocabPreType::Hunyuan | VocabPreType::SolarOpen => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Raon => vec![
            " ?[一-龥぀-ゟ゠-ヿ]+| ?(?:(?:(?![一-龥぀-ゟ゠-ヿ])\\p{L})\\p{M}*)+(?: (?:(?:(?![一-龥぀-ゟ゠-ヿ])\\p{L})\\p{M}*)+)*|\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Qwen35 => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Poro | VocabPreType::Bloom | VocabPreType::Gpt3Finnish => vec![
            " ?[^(\\s|.,!?…。，、।۔،)]+".into(),
        ],
        VocabPreType::ChatGlm4 => vec![llama3.into()],
        VocabPreType::Viking => vec![" ?[^(\\s|.,!?…。，、।۔،)]+".into(), "\\p{N}".into()],
        VocabPreType::Tekken | VocabPreType::Nemotron35Embd => vec![tekken.into()],
        VocabPreType::TekkenNum3 => vec![
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*|\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))|\\p{N}{1,2}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Chameleon => vec![
            "<sentinel:[0-9]+>".into(),
            "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z".into(),
            "([\\t\\n]|    |  )".into(),
            "\\p{N}".into(),
            "[\\p{P}!-/:-@\\[-`{-~]".into(),
            gpt2.into(),
        ],
        VocabPreType::Gpt4o | VocabPreType::MinimaxM2 => vec![gpt4o.into()],
        VocabPreType::GraniteEmbMulti => vec![gpt4o.into()],
        VocabPreType::TinyAya => vec![
            "\\d{1,3}(?=(?:\\d{3})*\\b)".into(),
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::KimiK2 => vec!["\\p{Han}+".into()],
        VocabPreType::SuperBpe => vec!["\\p{N}+".into(), "(?=(\\d{3})+(?!\\d))".into()],
        VocabPreType::BailingMoe => vec![
            "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::SeedCoder => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Ufakzeka => vec![
            "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Grok2 => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Afmoe => vec![
            "\\p{AFMoE_digits}".into(),
            "[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+".into(),
            punct.into(),
        ],
        VocabPreType::Laguna => vec![newlines.into(), llama3.into()],
        VocabPreType::ExaoneMoe => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Gemma4 | VocabPreType::SarvamMoe | VocabPreType::DomynSmall | VocabPreType::Mbert => {
            // SPM-style BPE: spaces are replaced with U+2581 by the normalizer,
            // then BPE merges run on the whole text without word-level splitting.
            // We only need to split on newlines since merge lookup asserts no newlines in tokens.
            byte_encode = false;
            vec![newlines.into()]
        }
        VocabPreType::Minicpm5 => vec!["\\p{N}{1,3}".into(),
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into()],
        VocabPreType::K2Horizon => vec![
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into(),
        ],
        VocabPreType::Soprano => {
            byte_encode = false;
            vec!["\\p{N}".into(), "\\s+|[\\p{L}\\p{N}_]+|[^\\p{L}\\p{N}_\\s]+".into()]
        }
        VocabPreType::Zonos3 => vec![
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]{2,}[\\r\\n/]*| +(?!\\S)".into(),
        ],
        VocabPreType::Whitespace => {
            // whitespace pre-tokenizer (jinaai/jina-embeddings-v2-base-zh)
            byte_encode = false;
            vec!["\\S+".into()]
        }
        _ => vec![
            "[\\p{P}\\$\\+<=>\\^~\\|]+".into(),
            gpt2.into(),
            "\\p{N}+".into(),
            "[0-9][0-9][0-9]".into(),
        ],
    };
    (exprs, byte_encode)
}

pub(crate) struct BpeSession<'a> {
    pub(crate) vocab: &'a Impl,
    pub(crate) tokenizer: &'a BpeData,
    pub(crate) symbols: Vec<LlmSymbol>,
    pub(crate) symbols_final: Vec<LlmSymbol>,
    pub(crate) work_queue: BinaryHeap<BpeBigram>,
}

impl<'a> BpeSession<'a> {
    pub(crate) fn new(vocab: &'a Impl, tokenizer: &'a BpeData) -> Self {
        Self {
            vocab,
            tokenizer,
            symbols: Vec::new(),
            symbols_final: Vec::new(),
            work_queue: BinaryHeap::new(),
        }
    }

    pub(crate) fn append(token_id: Token, output: &mut Vec<Token>) {
        output.push(token_id);
    }

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
        // The warnings are omitted; the checks are kept in the same order.
        let _ = self.vocab.get_add_bos() && output.get(1) == Some(&self.vocab.token_bos());
        let _ = self.vocab.get_add_eos()
            && output.len() >= 2
            && output.get(output.len() - 2) == Some(&self.vocab.token_eos());
    }

    pub(crate) fn tokenize(&mut self, text: &str, output: &mut Vec<Token>) {
        let tok_pre = self.vocab.get_pre_type();

        let word_collection: Vec<String> = if tok_pre == VocabPreType::Nemotron35Embd {
            let zwnj = "\u{200c}";
            let mut normalized_text = text.to_string();
            replace_all(&mut normalized_text, &format!("[{}", zwnj), "[");
            replace_all(&mut normalized_text, &format!("<{}", zwnj), "<");
            unicode_regex_split(&normalized_text, &self.tokenizer.regex_exprs, self.tokenizer.byte_encode)
        } else if tok_pre == VocabPreType::Soprano {
            let mut normalized_text = String::new();
            let mut space = false;
            for cpt in unicode_cpts_from_utf8(text) {
                if unicode_cpt_flags_from_cpt(cpt).is_whitespace() {
                    if !space {
                        normalized_text.push(' ');
                    }
                    space = true;
                } else {
                    // Unicode full lowercase expands LATIN CAPITAL LETTER I WITH DOT ABOVE.
                    if cpt == 0x0130 {
                        normalized_text.push_str("i\u{0307}");
                    } else {
                        normalized_text.push_str(&unicode_cpt_to_utf8(unicode_tolower(cpt)));
                    }
                    space = false;
                }
            }
            unicode_regex_split(&normalized_text, &self.tokenizer.regex_exprs, self.tokenizer.byte_encode)
        } else if tok_pre == VocabPreType::Raon {
            // Raon-1 mini's HF tokenizer applies an NFC normalizer before pre-tokenization.
            let mut normalized_text = String::new();
            for cpt in unicode_cpts_normalize_nfc(&unicode_cpts_from_utf8(text)) {
                normalized_text.push_str(&unicode_cpt_to_utf8(cpt));
            }
            unicode_regex_split(&normalized_text, &self.tokenizer.regex_exprs, self.tokenizer.byte_encode)
        } else {
            unicode_regex_split(text, &self.tokenizer.regex_exprs, self.tokenizer.byte_encode)
        };

        let mut final_prev_index: i32 = -1;
        self.symbols_final.clear();

        let mut word_collection = word_collection;
        if self.vocab.get_metaspace_prepend_scheme() == MetaspacePrependScheme::Always {
            // Metaspace pre-tokenizer (prepend_scheme = "always"): each non-newline chunk
            // starts with an escaped space, and each escaped space starts a new word
            let mut words = Vec::new();
            for word in word_collection {
                if !word.chars().any(|c| c != '\n') {
                    words.push(word);
                    continue;
                }
                let mut word = word;
                if !word.starts_with(ESCAPED_SPACE) {
                    word.insert_str(0, ESCAPED_SPACE);
                }
                let mut start = 0usize;
                while start < word.len() {
                    let search_from = start + ESCAPED_SPACE.len();
                    let end = if search_from <= word.len() {
                        match word[search_from..].find(ESCAPED_SPACE) {
                            Some(rel) => search_from + rel,
                            None => word.len(),
                        }
                    } else {
                        word.len()
                    };
                    words.push(word[start..end].to_string());
                    start = end;
                }
            }
            word_collection = words;
        }

        for word in &word_collection {
            self.work_queue.clear();
            self.symbols.clear();

            let mut index: i32 = 0;
            let mut offset = 0usize;
            let bytes = word.as_bytes();

            //if (vocab.tokenizer_ignore_merges && vocab.token_to_id.find(word) != vocab.token_to_id.end()) {
            if self.vocab.get_ignore_merges() && self.vocab.text_to_token(word) != LLAMA_TOKEN_NULL {
                self.symbols.push(LlmSymbol::new(bytes));
                offset = word.len();
            } else if (tok_pre == VocabPreType::Gemma4 || tok_pre == VocabPreType::Mbert)
                && !word.chars().any(|c| c != '\n')
            {
                // fix for gemma 4, ref: https://github.com/ggml-org/llama.cpp/pull/21343
                if self.vocab.text_to_token(word) != LLAMA_TOKEN_NULL {
                    self.symbols.push(LlmSymbol::new(bytes));
                    offset = word.len();
                }
            }

            while offset < bytes.len() {
                let char_len = (bytes.len() - offset).min(unicode_len_utf8(bytes[offset]));
                let mut sym = LlmSymbol::new(&bytes[offset..offset + char_len]);
                offset += char_len;
                sym.prev = index - 1;
                sym.next = if offset == bytes.len() { -1 } else { index + 1 };
                self.symbols.push(sym);
                index += 1;
            }
            for i in 1..self.symbols.len() as i32 {
                self.add_new_bigram(i - 1, i);
            }

            // build token(s)
            while let Some(bigram) = self.work_queue.pop() {
                let l = bigram.left as usize;
                let r = bigram.right as usize;
                if self.symbols[l].n == 0
                    || self.symbols[r].n == 0
                    || self.symbols[l].n + self.symbols[r].n != bigram.size
                {
                    continue;
                }

                // merge the right sym into the left one
                let right_sym = self.symbols[r].clone();
                self.symbols[l].merge_right(&right_sym);
                self.symbols[r].n = 0;

                // remove the right sym from the chain
                let next = self.symbols[r].next;
                self.symbols[l].next = next;
                if next >= 0 {
                    self.symbols[next as usize].prev = bigram.left;
                }

                let prev = self.symbols[l].prev;
                self.add_new_bigram(prev, bigram.left); // left side of current symbol
                self.add_new_bigram(bigram.left, next); // right side of current symbol
            }

            // add the finished tokens to the final list keeping correct order for next and prev
            for i in 0..self.symbols.len() {
                if self.symbols[i].n > 0 {
                    self.symbols[i].prev = final_prev_index;
                    self.symbols[i].next = -1;
                    if final_prev_index != -1 {
                        self.symbols_final[final_prev_index as usize].next = self.symbols_final.len() as i32;
                    }
                    self.symbols_final.push(self.symbols[i].clone());
                    final_prev_index = self.symbols_final.len() as i32 - 1;
                }
            }
        }

        self.symbols = self.symbols_final.clone();

        if !self.symbols.is_empty() {
            let mut i = 0i32;
            while i != -1 {
                let symbol = &self.symbols[i as usize];
                if symbol.n == 0 {
                    i = symbol.next;
                    continue;
                }

                let str = symbol.as_str();
                let token = self.vocab.text_to_token(str);

                if token == LLAMA_TOKEN_NULL && tok_pre == VocabPreType::Soprano {
                    output.push(self.vocab.token_unk());
                } else if token == LLAMA_TOKEN_NULL {
                    for &byte in str.as_bytes() {
                        let token_multibyte = if self.tokenizer.byte_encode {
                            let byte_str = String::from_utf8_lossy(&[byte]).into_owned();
                            self.vocab.text_to_token(&byte_str)
                        } else {
                            // For non-byte-encoded BPE (e.g. gemma-4), byte tokens use <0xXX> format
                            const HEX: &[u8; 16] = b"0123456789ABCDEF";
                            let buf = [
                                b'<', b'0', b'x', HEX[(byte >> 4) as usize],
                                HEX[(byte & 15) as usize], b'>',
                            ];
                            self.vocab.text_to_token(std::str::from_utf8(&buf).unwrap_or(""))
                        };
                        if token_multibyte != LLAMA_TOKEN_NULL {
                            output.push(token_multibyte);
                        }
                    }
                } else {
                    output.push(token);
                }
                i = self.symbols[i as usize].next;
            }
        }
    }

    fn add_new_bigram(&mut self, left: i32, right: i32) {
        if left == -1 || right == -1 {
            return;
        }
        let l = left as usize;
        let r = right as usize;
        let text_l = &self.symbols[l].text[..self.symbols[l].n];
        let text_r = &self.symbols[r].text[..self.symbols[r].n];
        let mut pair = text_l.to_vec();
        pair.extend_from_slice(text_r);
        let a = String::from_utf8_lossy(text_l);
        let b = String::from_utf8_lossy(text_r);
        let rank_found = self.vocab.find_bpe_rank(&a, &b);

        if rank_found < 0 {
            return;
        }
        self.work_queue.push(BpeBigram {
            left,
            right,
            size: self.symbols[l].n + self.symbols[r].n,
            rank: rank_found,
        });
    }
}

// WPM tokenizer
pub(crate) struct WpmSession<'a> {
    pub(crate) vocab: &'a Impl,
}

impl<'a> WpmSession<'a> {
    pub(crate) fn tokenize(&self, text: &str, output: &mut Vec<Token>) {
        // normalize and split by whitespace
        let words = WpmSession::preprocess(text, self.vocab.get_normalizer_opts());
        // bos token prepended already

        // find the longest tokens that form the words
        for word in words {
            // skip empty words
            if word.is_empty() {
                continue;
            }

            // prepend phantom space
            let word1 = format!("{}{}", ESCAPED_SPACE, word);
            let bytes = word1.as_bytes();
            let n = bytes.len();
            let current_tokens = output.len();

            // we're at the start of a new word
            // move through character position in word
            let mut i = 0usize;
            while i < n {
                // loop through possible match length
                let mut matched = false;
                let mut j = n.min(i + self.vocab.max_token_len() as usize + 1);
                while j > i {
                    let sub = std::str::from_utf8(&bytes[i..j]).unwrap_or("");
                    if self.vocab.text_to_token(sub) != LLAMA_TOKEN_NULL {
                        output.push(self.vocab.text_to_token(sub));
                        matched = true;
                        i = j - 1;
                        break;
                    }
                    j -= 1;
                }
                if !matched {
                    // discard all
                    output.truncate(current_tokens);
                    break; // and discard next tokens
                }
                i += 1;
            }

            // we didn't find any matches for this word
            if current_tokens == output.len() {
                output.push(self.vocab.token_unk());
            }
        }
    }

    // TODO: reduce string copies by using cpts_offs array
    fn preprocess(text: &str, normalizer_opts: NormalizerOptions) -> Vec<String> {
        let cpts = unicode_cpts_from_utf8(text);
        let cpts = if normalizer_opts.strip_accents {
            unicode_cpts_normalize_nfd(&cpts)
        } else {
            cpts
        };
        let mut words = vec![String::new()];

        for cpt in cpts {
            let flags = unicode_cpt_flags_from_cpt(cpt);

            if flags.is_whitespace() {
                if !words.last().map_or(false, |w| !w.is_empty()) {
                    continue;
                }
                words.push(String::new());
                continue;
            }

            assert!(!flags.is_separator());
            if cpt == 0 || cpt == 0xfffd || flags.is_control() {
                continue;
            }

            if normalizer_opts.strip_accents && flags.is_accent_mark() {
                continue;
            }

            let s = unicode_cpt_to_utf8(if normalizer_opts.lowercase {
                unicode_tolower(cpt)
            } else {
                cpt
            });
            if flags.is_punctuation() || (cpt < 0x7f && flags.is_symbol()) || WpmSession::is_chinese_char(cpt) {
                if !words.last().map_or(false, |w| !w.is_empty()) {
                    words.push(String::new());
                }
                *words.last_mut().unwrap() = s; // single char word
                words.push(String::new()); // start a new word
            } else {
                words.last_mut().unwrap().push_str(&s); // append char to word
            }
        }

        if words.last().map_or(false, |w| w.is_empty()) {
            words.pop();
        }
        words
    }

    fn is_chinese_char(cpt: u32) -> bool {
        (0x04e00..=0x09fff).contains(&cpt)
            || (0x03400..=0x04dbf).contains(&cpt)
            || (0x20000..=0x2a6df).contains(&cpt)
            || (0x2a700..=0x2b73f).contains(&cpt)
            || (0x2b740..=0x2b81f).contains(&cpt)
            || (0x2b920..=0x2ceaf).contains(&cpt) // this should be 0x2B820 but in hf rust code it is 0x2B920
            || (0x0f900..=0x0faff).contains(&cpt)
            || (0x2f800..=0x2fa1f).contains(&cpt)
    }
}

// UGM tokenizer
/*
 * This structure is a view wrapper for XOR-compressed double array (XCDA)
 * See Shunsuke Kanda (2018). Space- and Time-Efficient String Dictionaries.
 * Each bit-packed entry contains:
 * - BASE array value in bits 10-30
 * - LCHECK array value in bits 0-7
 * - LEAF array value in bit 9
 * Entries containing indexes of replacement sequences have set bit 31
 */
struct XcdaView<'b> {
    pub(crate) array: &'b [u32],
}

impl<'b> XcdaView<'b> {
    fn get_base(&self, index: usize) -> Result<u32> {
        let node = self.get_node(index)?;
        Ok((node >> 10) << ((node & (1u32 << 9)) >> 6))
    }
    fn get_lcheck(&self, index: usize) -> Result<u32> {
        let node = self.get_node(index)?;
        Ok(node & ((1u32 << 31) | 0xff))
    }
    fn get_leaf(&self, index: usize) -> Result<bool> {
        let node = self.get_node(index)?;
        Ok((node >> 8) & 1 != 0)
    }
    fn get_value(&self, index: usize) -> Result<u32> {
        let node = self.get_node(index)?;
        Ok(node & ((1u32 << 31) - 1))
    }
    fn get_node(&self, index: usize) -> Result<u32> {
        self.array.get(index).copied().ok_or_else(|| Error::new("Index out of array bounds in XCDA array!"))
    }
}

pub(crate) struct UgmSession<'a> {
    pub(crate) vocab: &'a Impl,
    pub(crate) tokenizer: &'a UgmData,
}

struct NormalizationResult {
    pub(crate) normalized: Vec<u8>,
    pub(crate) consumed_input: usize,
}

#[derive(Clone, Copy)]
struct BestTokenization {
    pub(crate) token_id: Token,
    pub(crate) input_offset: usize,
    pub(crate) score_sum: f64,
}

impl<'a> UgmSession<'a> {
    /* This implementation is based on SentencePiece optimized Viterbi algorithm for
     * unigram language models. The general idea is to:
     * - move along the input sequence in steps of one UTF code point,
     * - at each step find all possible tokenizations of the prefix by
     *   traversing the tokens trie,
     * - for each tokenization store the best one so far (by higher score)
     * - use the position in sequence after given token as an index to store
     *   results
     * - if there was no valid tokenization of the current UTF code point
     *   then use unknown token with additional score penalty
     * After processing the whole sequence we backtrack from the end to get
     * the best tokenization.
     */
    pub(crate) fn tokenize(&self, text: &str, output: &mut Vec<Token>) {
        let output_size = output.len();

        // normalize the input first
        let normalized = if self.vocab.get_normalizer_opts().nfc {
            let mut canonical = String::new();
            for cpt in unicode_cpts_normalize_nfc(&unicode_cpts_from_utf8(text)) {
                canonical.push_str(&unicode_cpt_to_utf8(cpt));
            }
            match self.normalize(&canonical) {
                Ok(s) => s,
                Err(_) => self.normalize(text).unwrap_or_default(),
            }
        } else {
            self.normalize(text).unwrap_or_default()
        };

        let input = normalized.as_slice();
        let input_len = input.len();
        if input_len == 0 {
            return;
        }

        // initialize score_sum to -DBL_MAX so it will be always lower than sums of token scores
        let mut results: Vec<BestTokenization> = vec![BestTokenization {
            token_id: self.vocab.token_unk(),
            input_offset: 0,
            score_sum: f64::MIN,
        }; input_len + 1];
        // at the beginning tokenization score is zero
        results[0] = BestTokenization {
            token_id: self.vocab.token_unk(),
            input_offset: 0,
            score_sum: 0.0,
        };

        let mut input_offset = 0usize;
        while input_offset < input_len {
            let mut prefix_offset = input_offset + 1;
            // calculate how many code units are in the currently processed UTF code point
            let mut probe = input_offset;
            let _ = unicode_cpt_from_utf8(input, &mut probe);
            let mut n_utf8_code_units = unicode_len_utf8(input[input_offset]).min(input_len - input_offset);
            if probe > input_offset {
                n_utf8_code_units = probe - input_offset;
            }

            // traverse the token matcher trie to find a matching token
            let mut single_codepoint_token_found = false;
            let current_best = results[input_offset];
            let mut node = self.tokenizer.token_matcher.traverse(input[input_offset]);

            while prefix_offset <= input_len && node.is_some() {
                let n = node.unwrap();
                // check if we found valid token in prefix
                if n.has_value {
                    // check if it corresponds to the whole UTF code point
                    if prefix_offset - input_offset == n_utf8_code_units {
                        single_codepoint_token_found = true;
                    }
                    let token_id = n.value;
                    let token_data = &self.vocab.get_token_data(token_id).score;
                    // we set the user-defined token scores to 0 to make them more likely to be selected
                    // (normal token scores are log probabilities, so they are negative)
                    // score type is double here to make tokenization results exactly
                    // the same as in the HF tokenizer using SentencePiece
                    let token_score = if self.vocab.is_user_defined(token_id) { 0.0 } else { *token_data as f64 };
                    let challenger_score = current_best.score_sum + token_score;
                    if challenger_score > results[prefix_offset].score_sum {
                        results[prefix_offset] = BestTokenization {
                            token_id,
                            input_offset,
                            score_sum: challenger_score,
                        };
                    }
                }
                if prefix_offset >= input_len {
                    break;
                }
                node = n.traverse(input[prefix_offset]);
                prefix_offset += 1;
            }

            // if we didn't find a valid token corresponding to the whole UTF code point
            // then use unknown token as the tokenization of this UTF code point
            if !single_codepoint_token_found {
                let challenger_score = current_best.score_sum + self.tokenizer.unknown_token_score as f64;
                let end = (input_offset + n_utf8_code_units).min(input_len);
                if challenger_score > results[end].score_sum {
                    results[end] = BestTokenization {
                        token_id: self.vocab.token_unk(),
                        input_offset,
                        score_sum: challenger_score,
                    };
                }
            }

            // move to the next UTF code point
            input_offset += n_utf8_code_units;
        }

        // now backtrack from the end to gather token ids of the best tokenization
        // merge sequences of consecutive unknown tokens into single unknown tokens
        let mut is_prev_unknown = false;
        let mut idx = input_len;
        loop {
            let tok = results[idx];
            let is_unknown = tok.token_id == self.vocab.token_unk();
            if !(is_prev_unknown && is_unknown) {
                output.push(tok.token_id);
            }
            if tok.input_offset == 0 {
                break;
            }
            is_prev_unknown = is_unknown;
            idx = tok.input_offset.min(input_len);
        }

        // reverse the output since we added tokens starting from the end of the input
        output[output_size..].reverse();
    }

    fn normalize(&self, input: &str) -> Result<Vec<u8>> {
        let mut normalized: Vec<u8> = Vec::with_capacity(input.len() * 3);
        let space = if self.vocab.get_escape_whitespaces() {
            ESCAPED_SPACE.as_bytes().to_vec()
        } else {
            b" ".to_vec()
        };

        let shall_prepend_space = !self.vocab.get_treat_whitespace_as_suffix() && self.vocab.get_add_space_prefix();
        let shall_append_space = self.vocab.get_treat_whitespace_as_suffix() && self.vocab.get_add_space_prefix();
        let shall_merge_spaces = self.vocab.get_remove_extra_whitespaces();

        let mut is_space_prepended = false;
        let mut processing_non_ws = false;

        let mut input_offset = 0usize;
        let bytes = input.as_bytes();
        while input_offset < bytes.len() {
            let norm_res = self.normalize_prefix(input, input_offset)?;
            for c in &norm_res.normalized {
                if *c != b' ' {
                    if !processing_non_ws {
                        processing_non_ws = true;
                        if (shall_prepend_space && !is_space_prepended) || shall_merge_spaces {
                            normalized.extend_from_slice(&space);
                            is_space_prepended = true;
                        }
                    }
                    normalized.push(*c);
                } else {
                    if processing_non_ws {
                        processing_non_ws = false;
                    }
                    if !shall_merge_spaces {
                        normalized.extend_from_slice(&space);
                    }
                }
            }
            input_offset += norm_res.consumed_input;
        }

        if shall_append_space {
            normalized.extend_from_slice(&space);
        }
        Ok(normalized)
    }

    fn normalize_prefix(&self, input: &str, input_offset: usize) -> Result<NormalizationResult> {
        let bytes = input.as_bytes();
        if input_offset == bytes.len() {
            return Ok(NormalizationResult { normalized: Vec::new(), consumed_input: 0 });
        }

        // if input prefix matches some user-defined token return this token as normalization result
        let user_len = self.tokenizer.user_defined_token_matcher.get_longest_prefix(&bytes[input_offset..], 0);
        if user_len > 0 {
            return Ok(NormalizationResult {
                normalized: bytes[input_offset..input_offset + user_len].to_vec(),
                consumed_input: user_len,
            });
        }

        let mut longest_prefix_length = 0usize;
        let mut longest_prefix_offset = 0usize;

        if !self.tokenizer.xcda_array.is_empty() {
            let view = XcdaView { array: &self.tokenizer.xcda_array };
            // Find the longest normalized sequence matching the input prefix by walking
            // the XOR-compressed compact double array (XCDA) starting from the root node
            // We find the index of the next node by calculating BASE[s] ^ c where s is
            // the index of the previous node and c is a numerical character value
            let mut node_index = view.get_base(0)? as usize;
            for prefix_offset in input_offset..bytes.len() {
                let c = bytes[prefix_offset];
                if c == 0 {
                    break;
                }
                node_index ^= c as usize;
                // if value of LCHECK is not c it means that this is not a child of
                // the previous node, so we stop matching
                if view.get_lcheck(node_index)? != c as u32 {
                    break;
                }
                let is_leaf = view.get_leaf(node_index)?;
                // get BASE of the current node
                node_index ^= view.get_base(node_index)? as usize;
                // if LEAF of the current node is true, it means that its BASE points to the node
                // containing index of replacement sequence for currently matched input prefix
                if is_leaf {
                    longest_prefix_length = prefix_offset - input_offset + 1;
                    // get index of replacement sequence for currently matched input prefix
                    longest_prefix_offset = view.get_value(node_index)? as usize;
                }
            }
        }

        if longest_prefix_length > 0 {
            // we have a match, so return the replacement sequence
            if longest_prefix_offset >= self.tokenizer.prefix_replacements.len() {
                return Err(Error::new("Index out of array bounds in precompiled charsmap!"));
            }
            let replacements = &self.tokenizer.prefix_replacements[longest_prefix_offset..];
            let repl_len = replacements.iter().position(|&b| b == 0).unwrap_or(replacements.len());
            if repl_len == replacements.len() {
                return Err(Error::new("Unterminated string in precompiled charsmap!"));
            }
            return Ok(NormalizationResult {
                normalized: replacements[..repl_len].to_vec(),
                consumed_input: longest_prefix_length,
            });
        }

        // check if the input prefix contains a valid sequence of UTF-8 code units
        let mut prefix_offset = input_offset;
        match unicode_cpt_from_utf8(bytes, &mut prefix_offset) {
            Ok(_) => Ok(NormalizationResult {
                normalized: bytes[input_offset..prefix_offset].to_vec(),
                consumed_input: prefix_offset - input_offset,
            }),
            // if no, consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER
            Err(_) => Ok(NormalizationResult {
                normalized: b"\xef\xbf\xbd".to_vec(),
                consumed_input: 1,
            }),
        }
    }
}

// RWKV tokenizer
pub fn llama_unescape_rwkv_token(escaped: &str) -> Vec<u8> {
    let mut output = Vec::with_capacity(escaped.len());
    // Parser state
    let mut escaping = false;
    let mut hex_remaining: u8 = 0;
    let mut hex_acc: u8 = 0;

    // Step through characters, performing parsing
    for c in escaped.bytes() {
        // If we're parsing a hex code, interpret the next character
        if hex_remaining != 0 {
            let value = if c >= b'a' { c - b'a' + 10 } else { c - b'0' };
            hex_acc = (hex_acc << 4) + value;
            hex_remaining -= 1;
            if hex_remaining == 0 {
                output.push(hex_acc);
                hex_acc = 0;
            }
            continue;
        }

        // If we got an escape character, interpret it
        if escaping {
            match c {
                b't' => output.push(b'\t'),
                b'n' => output.push(b'\n'),
                b'r' => output.push(b'\r'),
                b'x' => hex_remaining = 2,
                _ => output.push(c),
            }
            escaping = false;
            continue;
        }

        if c == b'\\' {
            escaping = true;
            continue;
        }

        output.push(c);
    }
    output
}

pub(crate) struct RwkvSession<'a> {
    pub(crate) vocab: &'a Impl,
    pub(crate) tokenizer: &'a RwkvData,
}

impl<'a> RwkvSession<'a> {
    pub(crate) fn tokenize(&self, text: &str, output: &mut Vec<Token>) {
        let bytes = text.as_bytes();
        let mut position = 0usize;
        while position < bytes.len() {
            let mut node = self.tokenizer.token_matcher.traverse(bytes[position]);
            if node.is_none() {
                // no matching token found, add unknown token
                output.push(self.vocab.token_unk());
                position += 1;
                continue;
            }

            // traverse the trie to find the longest matching token
            let start = position;
            let mut token_id = 0;
            let mut token_length = 0usize;
            while let Some(n) = node {
                if n.has_value {
                    token_id = n.value;
                    token_length = position + 1;
                }
                if position + 1 >= bytes.len() {
                    break;
                }
                position += 1;
                node = n.traverse(bytes[position]);
            }

            // add the longest matching token
            output.push(token_id);
            position = if token_length > 0 { token_length } else { start + 1 };
        }
    }
}

// PLaMo2 tokenizer
impl Plamo2Data {
    pub(crate) fn build(
        id_to_token: &[TokenData],
        _token_to_id_by_text: &HashMap<String, Token>,
        pre_segment: bool,
    ) -> Result<Self> {
        let n_tokens = id_to_token.len();
        let mut tokens = Vec::with_capacity(n_tokens);
        let mut bytes: Vec<Token> = vec![0; 256];
        let mut to_suffix_id: HashMap<i64, i32> = HashMap::new();
        let mut table: Vec<[i32; 4]> = Vec::new();

        // Suffixes and pieces are views into the token texts of the vocab, so no strings are built
        let mut suffix_to_score: HashMap<Vec<u8>, f32> = HashMap::with_capacity(n_tokens * 4);
        let mut token_to_id: HashMap<Vec<u8>, Token> = HashMap::with_capacity(n_tokens);

        // Byte offset of the next code point
        fn next_cpt(s: &[u8], offset: usize) -> usize {
            offset + (s.len() - offset).min(unicode_len_utf8(s[offset]))
        }

        for token_id in 0..n_tokens as Token {
            let entry = &id_to_token[token_id as usize];
            tokens.push(entry.text.clone());
            token_to_id.insert(entry.text.as_bytes().to_vec(), token_id);

            // Handle byte tokens
            if entry.attr & LLAMA_TOKEN_ATTR_BYTE != 0 {
                let t = entry.text.as_bytes();
                if t.len() == 6 && &t[..3] == b"<0x" && t[5] == b'>' {
                    if t[3].is_ascii_hexdigit() && t[4].is_ascii_hexdigit() {
                        let hex_str = &entry.text[3..5];
                        let byte_val = u8::from_str_radix(hex_str, 16).unwrap_or(0);
                        bytes[byte_val as usize] = token_id;
                    }
                }
                continue;
            }

            // Add token and all its suffixes to suffix_to_score
            let text = entry.text.as_bytes().to_vec();
            suffix_to_score.insert(text.clone(), entry.score);
            let mut i = if text.is_empty() { 0 } else { next_cpt(&text, 0) };
            while i < text.len() {
                suffix_to_score.entry(text[i..].to_vec()).or_insert(f32::NAN);
                i = next_cpt(&text, i);
            }
        }

        // Check that all byte tokens are set
        for (i, &id) in bytes.iter().enumerate() {
            if id == 0 {
                return Err(Error::new(format!("Byte token for <0x{:x}> is not set", i)));
            }
        }

        // Build suffix list in lexicographical order of reversed strings
        let mut suffixes: Vec<Vec<u8>> = suffix_to_score.keys().cloned().collect();
        suffixes.push(Vec::new()); // Empty suffix
        suffixes.sort_by(|a, b| a.iter().rev().cmp(b.iter().rev()));

        // Byte length of each prefix of the suffix, by code point
        let mut prefix_ends: Vec<usize> = Vec::new();
        fn get_prefix_ends(suffix: &[u8]) -> Vec<usize> {
            let mut out = Vec::new();
            let mut i = 0usize;
            while i < suffix.len() {
                i = next_cpt(suffix, i);
                out.push(i);
            }
            out
        }

        // Build suffix_to_id and to_suffix_id_
        let mut suffix_to_id: HashMap<Vec<u8>, i32> = HashMap::with_capacity(suffixes.len());
        let mut num_pieces = 0i32;

        for suffix in &suffixes {
            suffix_to_id.insert(suffix.clone(), num_pieces);
            if !suffix.is_empty() {
                prefix_ends = get_prefix_ends(suffix);
                let first_bytes = &suffix[..prefix_ends[0]];
                let first_str = std::str::from_utf8(first_bytes).unwrap_or("");
                let first_cpt = unicode_cpts_from_utf8(first_str).first().copied().unwrap_or(0xfffd);
                let piece_code = ((first_cpt as i64) << 32) | suffix_to_id[&suffix[prefix_ends[0]..]] as i64;
                to_suffix_id.insert(piece_code, num_pieces);

                // Count number of pieces for this suffix
                let mut pieces_for_suffix = 1i32; // sentinel row
                for &end in &prefix_ends {
                    if suffix_to_score.contains_key(&suffix[..end]) {
                        pieces_for_suffix += 1;
                    }
                }
                num_pieces += pieces_for_suffix;
            } else {
                num_pieces += 1; // Empty suffix contributes one piece (sentinel row)
            }
        }

        // Build flattened table
        table.resize(num_pieces as usize, [0; 4]);
        let mut table_idx = 0usize;

        const TABLE_PIECE_LENGTH: usize = 0;
        const TABLE_TOKEN_ID: usize = 1;
        const TABLE_SCORE: usize = 2;
        const TABLE_PIECE_ID: usize = 3;
        const INVALID_SCORE: i32 = -20_000_000;
        const UNKNOWN_SCORE: i32 = -10_000_000;

        for suffix in &suffixes {
            // Add all prefixes of the suffix to the table (in decreasing order of length)
            prefix_ends = get_prefix_ends(suffix);
            for &piece_length in prefix_ends.iter().rev() {
                let piece = &suffix[..piece_length];
                let score = match suffix_to_score.get(piece) {
                    Some(score) => *score,
                    None => continue,
                };
                table[table_idx][TABLE_PIECE_LENGTH] = piece_length as i32;
                table[table_idx][TABLE_TOKEN_ID] = token_to_id.get(piece).copied().unwrap_or(-1);
                table[table_idx][TABLE_SCORE] = if score.is_finite() {
                    (score * 1e4).round() as i32
                } else {
                    INVALID_SCORE
                };
                table[table_idx][TABLE_PIECE_ID] = suffix_to_id.get(piece).copied().unwrap_or(0);
                table_idx += 1;
            }

            // Add sentinel row
            table[table_idx][TABLE_PIECE_LENGTH] = 1;
            table[table_idx][TABLE_TOKEN_ID] = -1;
            table[table_idx][TABLE_SCORE] = UNKNOWN_SCORE;
            table_idx += 1;
        }

        Ok(Self { tokens, bytes, to_suffix_id, table, pre_segment })
    }

    fn encode(&self, text: &str) -> Vec<Token> {
        let mut unicode_data = unicode_cpts_from_utf8(text);

        // The PLaMo-3 tokenizer keeps a leading U+FEFF in the input.
        if !self.pre_segment {
            if unicode_data.first() == Some(&0xfeff) {
                unicode_data.remove(0);
            }
            return self.encode_cpts(&unicode_data);
        }

        let n = unicode_data.len();
        let mut cut = vec![false; n + 1];

        // pass 1: <|plamo:...|>
        {
            const PREFIX: [u32; 8] = [
                b'<' as u32, b'|' as u32, b'p' as u32, b'l' as u32,
                b'a' as u32, b'm' as u32, b'o' as u32, b':' as u32,
            ];
            let prefix_len = PREFIX.len();
            let mut i = 0usize;
            while i + prefix_len <= n {
                if unicode_data[i..i + prefix_len] != PREFIX[..] {
                    i += 1;
                    continue;
                }
                // An empty body is valid.
                let mut j = i + prefix_len;
                // Treat U+001C..U+001F as whitespace (equivalent to Python \s).
                while j < n && j - (i + prefix_len) < 64 && unicode_data[j] != b'|' as u32
                    && !(0x1c..=0x1f).contains(&unicode_data[j])
                    && !unicode_cpt_flags_from_cpt(unicode_data[j]).is_whitespace()
                {
                    j += 1;
                }
                if j + 1 < n && unicode_data[j] == b'|' as u32 && unicode_data[j + 1] == b'>' as u32 {
                    cut[i] = true;
                    cut[j + 2] = true;
                    i = j + 2;
                } else {
                    i += 1;
                }
            }
        }

        // pass 2: runs of repeated characters / spaces (a run never crosses a boundary from pass 1)
        {
            let mut i = 0usize;
            while i < n {
                let c = unicode_data[i];
                let mut run = 1usize;
                while i + run < n && unicode_data[i + run] == c && !cut[i + run] {
                    run += 1;
                }

                let is_repeated_chars = c != b'\n' as u32 && run >= 4;
                let is_spaces = c == b' ' as u32 && run >= 2;
                if is_repeated_chars || is_spaces {
                    cut[i] = true;
                    cut[i + run] = true;
                }

                // a run that does not match cannot match at any later position either (it only gets shorter)
                i += run;
            }
        }

        let mut output = Vec::new();
        let mut seg_start = 0usize;
        for seg_end in 0..=n {
            // U+EE00 is the tokenizer's private-use boundary marker; literal occurrences split segments and are not emitted.
            let is_boundary = seg_end < n && unicode_data[seg_end] == 0xee00;
            if seg_end == n || cut[seg_end] || is_boundary {
                if seg_start < seg_end {
                    let segment = unicode_data[seg_start..seg_end].to_vec();
                    output.extend(self.encode_cpts(&segment));
                }
                seg_start = seg_end + is_boundary as usize;
            }
        }

        output
    }

    fn encode_cpts(&self, unicode_data: &[u32]) -> Vec<Token> {
        if unicode_data.is_empty() {
            return Vec::new();
        }

        const TABLE_PIECE_LENGTH: usize = 0;
        const TABLE_TOKEN_ID: usize = 1;
        const TABLE_SCORE: usize = 2;
        const TABLE_PIECE_ID: usize = 3;
        const PATH_TOKEN_LENGTH: usize = 0;
        const PATH_TOKEN_ID: usize = 1;
        const PATH_NUM_TOKENS: usize = 2;
        const INVALID_SCORE: i32 = -20_000_000;
        const UNKNOWN_SCORE: i32 = -10_000_000;

        let data_len = unicode_data.len();

        // Initialize scores array (dynamic programming)
        let mut scores = vec![1i64 << 60; data_len + 1];
        scores[data_len] = 0;

        // Path array to track best tokenization
        let mut path = vec![[0i32; 3]; data_len + 1];

        let mut suffix_id = 0usize;

        // Process from end to beginning
        for i in (0..data_len as i32).rev() {
            let c = unicode_data[i as usize];

            // Find next suffix ID
            for p in suffix_id..self.table.len() {
                let piece_code = ((c as i64) << 32) | self.table[p][TABLE_PIECE_ID] as i64;
                suffix_id = self.to_suffix_id.get(&piece_code).copied().unwrap_or(0) as usize;

                if suffix_id > 0 || self.table[p][TABLE_SCORE] == UNKNOWN_SCORE {
                    break;
                }
            }

            // Update best path
            for p in suffix_id..self.table.len() {
                let score = self.table[p][TABLE_SCORE];
                if score > INVALID_SCORE {
                    let piece_length = self.table[p][TABLE_PIECE_LENGTH] as usize;
                    let next_score = scores.get(i as usize + piece_length).copied().unwrap_or(1i64 << 60);
                    let s = next_score - score as i64;

                    if s < scores[i as usize] {
                        scores[i as usize] = s;
                        path[i as usize][PATH_TOKEN_LENGTH] = piece_length as i32;
                        path[i as usize][PATH_TOKEN_ID] = self.table[p][TABLE_TOKEN_ID];
                        let n = path.get(i as usize + piece_length).map(|p| p[PATH_NUM_TOKENS]).unwrap_or(0);
                        path[i as usize][PATH_NUM_TOKENS] = n + 1;

                        if score == UNKNOWN_SCORE {
                            // Add UTF-8 byte count
                            path[i as usize][PATH_NUM_TOKENS] +=
                                (c >= 0x80) as i32 + (c >= 0x800) as i32 + (c >= 0x10000) as i32;
                        }
                    }
                }

                if score == UNKNOWN_SCORE {
                    break;
                }
            }
        }

        // Decode the best path
        let mut token_ids = Vec::with_capacity(path[0][PATH_NUM_TOKENS] as usize);
        let mut pos = 0usize;
        while pos < data_len {
            if path[pos][PATH_TOKEN_ID] >= 0 {
                token_ids.push(path[pos][PATH_TOKEN_ID]);
            } else {
                // Fall back to byte tokens
                let c = unicode_data[pos];
                let s = 1 + (c >= 0x80) as u32 + (c >= 0x800) as u32 + (c >= 0x10000) as u32;

                for i in 0..s {
                    let b: u8 = if s == 1 {
                        c as u8
                    } else if i == 0 {
                        ((0xf00 >> s) & 0xff) as u8
                    } else {
                        0x80
                    };
                    let idx = (b | ((c >> ((s - i - 1) * 6)) & 0x3f) as u8) as usize;
                    if let Some(&id) = self.bytes.get(idx) {
                        token_ids.push(id);
                    }
                }
            }

            assert!(path[pos][PATH_TOKEN_LENGTH] > 0);
            pos += path[pos][PATH_TOKEN_LENGTH] as usize;
        }

        token_ids
    }
}

pub(crate) struct Plamo2Session<'a> {
    pub(crate) tokenizer: &'a Plamo2Data,
}

impl<'a> Plamo2Session<'a> {
    pub(crate) fn tokenize(&self, text: &str, output: &mut Vec<Token>) {
        let tokens = self.tokenizer.encode(text);
        output.extend(tokens);
    }
}

// HybridDNATokenizer (HuggingFaceBio/Carbon-*): text inside <dna>...</dna> is cut into fixed 6-mers, everything else goes through the base BPE
// Mirrors HybridDNATokenizer.encode, with optional tagging of untagged DNA (auto_dna_tags, Carbon-14B) and tail="pad_n" vocab, whose dedicated tokens pad a trailing partial k-mer with N instead of A ("CT" -> "CTNNNN")
pub(crate) struct HybridDnaSession<'a> {
    pub(crate) bpe: BpeSession<'a>,
    pub(crate) vocab: &'a Impl,
    pub(crate) dna_begin_id: Token,
    pub(crate) dna_end_id: Token,
    pub(crate) dna_oov_id: Token,
    pub(crate) pad_char: u8,
    pub(crate) open_tag: &'static str,
    pub(crate) close_tag: &'static str,
    pub(crate) k: usize,
}

impl<'a> HybridDnaSession<'a> {
    pub(crate) fn new(vocab: &'a Impl, tokenizer: &'a BpeData) -> Self {
        let dna_begin_id = vocab.text_to_token("<dna>");
        let dna_end_id = vocab.text_to_token("</dna>");
        let dna_oov_id = vocab.text_to_token("<oov>");
        let pad_char = if vocab
            .text_to_token(&format!("A{}{}", "N".repeat(5), DNA_KMER_MARKER))
            != LLAMA_TOKEN_NULL
        {
            b'N'
        } else {
            b'A'
        };
        Self {
            bpe: BpeSession::new(vocab, tokenizer),
            vocab,
            dna_begin_id,
            dna_end_id,
            dna_oov_id,
            pad_char,
            open_tag: "<dna>",
            close_tag: "</dna>",
            k: 6,
        }
    }

    pub(crate) fn tokenize(&mut self, text: &str, output: &mut Vec<Token>) {
        // Fall back to plain BPE if the DNA pieces aren't in the vocab.
        if self.dna_begin_id == LLAMA_TOKEN_NULL
            || self.dna_end_id == LLAMA_TOKEN_NULL
            || self.dna_oov_id == LLAMA_TOKEN_NULL
        {
            self.bpe.tokenize(text, output);
            return;
        }

        // split on explicit tags as HybridDNATokenizer._split_by_dna_tags does; only the text outside them is auto-tagged
        let mut pos = 0usize;
        while pos < text.len() {
            const NPOS: usize = usize::MAX;
            let start = text[pos..].find(self.open_tag).map_or(NPOS, |p| p + pos);
            let end = text[pos..].find(self.close_tag).map_or(NPOS, |p| p + pos);

            if start == NPOS && end == NPOS {
                self.tokenize_text(&text[pos..], output);
                break;
            }
            if start == NPOS || end < start {
                // as in _split_by_dna_tags, text before a </dna> without a matching <dna> is DNA (no begin token)
                self.tokenize_dna(&text[pos..end], false, true, output);
                pos = end + self.close_tag.len();
                continue;
            }

            self.tokenize_text(&text[pos..start], output);

            let content_start = start + self.open_tag.len();
            if end == NPOS {
                // an unterminated <dna> runs to the end of the text (no end token)
                self.tokenize_dna(&text[content_start..], true, false, output);
                break;
            }
            self.tokenize_dna(&text[content_start..end], true, true, output);
            pos = end + self.close_tag.len();
        }
    }

    fn is_letter(c: u8) -> bool {
        c.is_ascii_alphabetic()
    }
    fn is_acgt(c: u8) -> bool {
        c == b'A' || c == b'C' || c == b'G' || c == b'T'
    }
    fn to_upper(c: u8) -> u8 {
        if c.is_ascii_lowercase() { c - 32 } else { c }
    }

    // IUPAC nucleotide codes, either case (HybridDNATokenizer.IUPAC_RUN_RE)
    fn is_iupac(c: u8) -> bool {
        b"ACGTRYSWKMBDHVN".contains(&Self::to_upper(c))
    }

    // HybridDNATokenizer._normalize_spaced_dna: join displays like "ACGTA CGTAC ..." when they hold at least twelve bases
    // (at least six chunks of 1-5 uppercase ACGT separated by spaces or tabs, not touching a letter)
    fn normalize_spaced_dna(text: &str) -> String {
        let is_blank = |c: u8| c == b' ' || c == b'\t';
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut p = 0usize;
        while p < bytes.len() {
            if Self::is_acgt(bytes[p]) && (p == 0 || !Self::is_letter(bytes[p - 1])) {
                let mut match_end = None;
                let mut chunk = p;
                let mut n_chunks = 1;
                loop {
                    let mut chunk_end = chunk;
                    while chunk_end < bytes.len() && chunk_end - chunk < 5 && Self::is_acgt(bytes[chunk_end]) {
                        chunk_end += 1;
                    }
                    if chunk_end == chunk {
                        break;
                    }
                    if n_chunks >= 6 && (chunk_end == bytes.len() || !Self::is_letter(bytes[chunk_end])) {
                        match_end = Some(chunk_end); // the regex is greedy, so the last valid end wins
                    }
                    let mut next = chunk_end;
                    while next < bytes.len() && is_blank(bytes[next]) {
                        next += 1;
                    }
                    if next == chunk_end {
                        break;
                    }
                    chunk = next;
                    n_chunks += 1;
                }
                if let Some(end) = match_end {
                    let mut seq = Vec::new();
                    for &b in &bytes[p..end] {
                        if !is_blank(b) {
                            seq.push(b);
                        }
                    }
                    if seq.len() >= 12 {
                        out.extend_from_slice(&seq);
                    } else {
                        out.extend_from_slice(&bytes[p..end]);
                    }
                    p = end;
                    continue;
                }
            }
            out.push(bytes[p]);
            p += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    // text outside explicit tags: with auto_dna_tags, tag DNA runs as HybridDNATokenizer._auto_tag_text_segment does
    // (pure ACGT runs of at least 6 bases, or runs of at least 12 with at most 20% IUPAC ambiguity codes)
    fn tokenize_text(&mut self, raw: &str, output: &mut Vec<Token>) {
        let text = if self.vocab.get_dna_auto_tags() {
            Self::normalize_spaced_dna(raw)
        } else {
            raw.to_string()
        };
        if !self.vocab.get_dna_auto_tags() {
            if !text.is_empty() {
                self.bpe.tokenize(&text, output);
            }
            return;
        }

        let bytes = text.as_bytes();
        let mut prev = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            if !Self::is_iupac(bytes[i]) {
                i += 1;
                continue;
            }
            let mut j = i;
            let mut n_ambiguous = 0usize;
            while j < bytes.len() && Self::is_iupac(bytes[j]) {
                n_ambiguous += !Self::is_acgt(Self::to_upper(bytes[j])) as usize;
                j += 1;
            }
            let len = j - i;
            if (n_ambiguous == 0 && len >= 6) || (n_ambiguous > 0 && len >= 12 && 5 * n_ambiguous <= len) {
                if i > prev {
                    let piece = String::from_utf8_lossy(&bytes[prev..i]).into_owned();
                    self.bpe.tokenize(&piece, output);
                }
                let dna = String::from_utf8_lossy(&bytes[i..j]).into_owned();
                self.tokenize_dna(&dna, true, true, output);
                prev = j;
            }
            i = j;
        }
        if prev < text.len() {
            let piece = String::from_utf8_lossy(&bytes[prev..]).into_owned();
            self.bpe.tokenize(&piece, output);
        }
    }

    // HybridDNATokenizer._process_dna_sequence: strip, uppercase, then one token per k bases
    // a k-mer with anything but ACGT is <oov>, and a trailing partial k-mer is right-padded with pad_char
    fn tokenize_dna(&self, content: &str, has_start: bool, has_end: bool, output: &mut Vec<Token>) {
        if has_start {
            output.push(self.dna_begin_id);
        }

        let cpts = unicode_cpts_from_utf8(content);
        let mut first = 0usize;
        let mut last = cpts.len();
        while first < last && unicode_cpt_flags_from_cpt(cpts[first]).is_whitespace() {
            first += 1;
        }
        while last > first && unicode_cpt_flags_from_cpt(cpts[last - 1]).is_whitespace() {
            last -= 1;
        }

        // k-mers carry the reserved marker suffix, so they never resolve to a base-vocab token
        let kmer_token = |from: usize, to: usize| -> Token {
            let mut kmer = Vec::new();
            for &cpt in &cpts[from..to] {
                let c = if cpt < 128 { Self::to_upper(cpt as u8) as u32 } else { cpt };
                if c > 127 || !Self::is_acgt(c as u8) {
                    return self.dna_oov_id;
                }
                kmer.push(c as u8);
            }
            while kmer.len() < self.k {
                kmer.push(self.pad_char);
            }
            let text = format!("{}{}", String::from_utf8_lossy(&kmer), DNA_KMER_MARKER);
            let tok = self.vocab.text_to_token(&text);
            if tok != LLAMA_TOKEN_NULL { tok } else { self.dna_oov_id }
        };

        let mut i = first;
        while i + self.k <= last {
            output.push(kmer_token(i, i + self.k));
            i += self.k;
        }
        if i < last {
            output.push(kmer_token(i, last));
        }

        if has_end {
            output.push(self.dna_end_id);
        }
    }
}

pub(crate) struct WhitespaceSession<'a> {
    pub(crate) bpe: BpeSession<'a>,
    pub(crate) vocab: &'a Impl,
}

impl<'a> WhitespaceSession<'a> {
    pub(crate) fn tokenize(&mut self, text: &str, output: &mut Vec<Token>) {
        let lowercase = self.vocab.get_normalizer_opts().lowercase;

        let mut segment = String::new();
        let mut flush = |segment: &mut String, output: &mut Vec<Token>| {
            if !segment.is_empty() {
                self.bpe.tokenize(segment, output);
                segment.clear();
            }
        };

        for cpt in unicode_cpts_from_utf8(text) {
            // drop whitespace
            if unicode_cpt_flags_from_cpt(cpt).is_whitespace() {
                flush(&mut segment, output);
            } else {
                segment.push_str(&unicode_cpt_to_utf8(if lowercase { unicode_tolower(cpt) } else { cpt }));
            }
        }
        flush(&mut segment, output);
    }
}


//! GBNF parser, repetition rewrites, left factoring, and left recursion.

use crate::core::{Error, Result, Token};
use crate::grammar::state::{is_end_of_sequence, print_rule};
use crate::grammar::types::*;
use crate::grammar::utf8::{byte_at, decode_utf8_char, tail};
use std::collections::BTreeMap;

fn is_digit_char(c: u8) -> bool {
    b'0' <= c && c <= b'9'
}

fn is_word_char(c: u8) -> bool {
    (b'a' <= c && c <= b'z') || (b'A' <= c && c <= b'Z') || c == b'-' || is_digit_char(c)
}

fn parse_hex(src: &[u8], pos: usize, size: usize) -> Result<(u32, usize)> {
    let mut p = pos;
    let mut value = 0u32;
    for _ in 0..size {
        let c = byte_at(src, p);
        if c == 0 {
            break;
        }
        value <<= 4;
        match c {
            b'a'..=b'f' => value += (c - b'a' + 10) as u32,
            b'A'..=b'F' => value += (c - b'A' + 10) as u32,
            b'0'..=b'9' => value += (c - b'0') as u32,
            _ => break,
        }
        p += 1;
    }
    if p - pos != size {
        return Err(Error::new(format!(
            "expecting {} hex chars at {}",
            size,
            tail(src, pos)
        )));
    }
    Ok((value, p))
}

fn parse_space(src: &[u8], mut pos: usize, newline_ok: bool) -> usize {
    loop {
        let c = byte_at(src, pos);
        if c == b' ' || c == b'\t' || c == b'#' || (newline_ok && (c == b'\r' || c == b'\n')) {
            if c == b'#' {
                while {
                    let c = byte_at(src, pos);
                    c != 0 && c != b'\r' && c != b'\n'
                } {
                    pos += 1;
                }
            } else {
                pos += 1;
            }
        } else {
            break;
        }
    }
    pos
}

fn parse_name(src: &[u8], pos: usize) -> Result<usize> {
    let mut p = pos;
    while is_word_char(byte_at(src, p)) {
        p += 1;
    }
    if p == pos {
        return Err(Error::new(format!("expecting name at {}", tail(src, pos))));
    }
    Ok(p)
}

fn parse_int(src: &[u8], pos: usize) -> Result<usize> {
    let mut p = pos;
    while is_digit_char(byte_at(src, p)) {
        p += 1;
    }
    if p == pos {
        return Err(Error::new(format!(
            "expecting integer at {}",
            tail(src, pos)
        )));
    }
    Ok(p)
}

pub(crate) fn parse_char(src: &[u8], pos: usize) -> Result<(u32, usize)> {
    if byte_at(src, pos) == b'\\' {
        match byte_at(src, pos + 1) {
            b'x' => parse_hex(src, pos + 2, 2),
            b'u' => parse_hex(src, pos + 2, 4),
            b'U' => parse_hex(src, pos + 2, 8),
            b't' => Ok((b'\t' as u32, pos + 2)),
            b'r' => Ok((b'\r' as u32, pos + 2)),
            b'n' => Ok((b'\n' as u32, pos + 2)),
            b'\\' | b'"' | b'[' | b']' | b'-' => Ok((byte_at(src, pos + 1) as u32, pos + 2)),
            _ => Err(Error::new(format!("unknown escape at {}", tail(src, pos)))),
        }
    } else if byte_at(src, pos) != 0 {
        Ok(decode_utf8_char(src, pos))
    } else {
        Err(Error::new("unexpected end of input"))
    }
}

fn parse_token_id(src: &[u8], pos: usize) -> Result<(u32, usize)> {
    let end = parse_int(src, pos)?;
    let text = std::str::from_utf8(&src[pos..end]).map_err(|_| Error::new("invalid token id"))?;
    let id: u64 = text
        .parse()
        .map_err(|_| Error::new(format!("parsed token id is too big at {}", tail(src, pos))))?;
    if id > u32::MAX as u64 {
        return Err(Error::new(format!(
            "parsed token id is too big at {}",
            tail(src, pos)
        )));
    }
    Ok((id as u32, end))
}
// ---------------------------------------------------------------------------
// parser (`struct llama_grammar_parser`)
// ---------------------------------------------------------------------------

/// C++ `llama_vocab::tokenize` narrowed to what `<token>` parsing needs:
/// receives the full `<...>` text with `add_bos = false, special = true`.
pub type TokenizeFn = dyn Fn(&[u8]) -> Result<Vec<Token>>;

pub struct GrammarParser {
    tokenize: Option<Box<TokenizeFn>>, // C++ `const llama_vocab * vocab`
    pub symbol_ids: BTreeMap<String, u32>,
    pub rules: GrammarRules,
}

impl GrammarParser {
    pub fn new() -> Self {
        Self::with_tokenizer_opt(None)
    }

    pub fn with_tokenizer<F>(tokenize: F) -> Self
    where
        F: Fn(&[u8]) -> Result<Vec<Token>> + 'static,
    {
        Self::with_tokenizer_opt(Some(Box::new(tokenize)))
    }

    pub(crate) fn with_tokenizer_opt(tokenize: Option<Box<TokenizeFn>>) -> Self {
        Self {
            tokenize,
            symbol_ids: BTreeMap::new(),
            rules: Vec::new(),
        }
    }

    pub fn get_symbol_id(&mut self, name: &[u8]) -> u32 {
        let next_id = self.symbol_ids.len() as u32;
        let key = String::from_utf8_lossy(name).into_owned();
        *self.symbol_ids.entry(key).or_insert(next_id)
    }

    pub fn generate_symbol_id(&mut self, base_name: &str) -> u32 {
        let next_id = self.symbol_ids.len() as u32;
        self.symbol_ids
            .insert(format!("{}_{}", base_name, next_id), next_id);
        next_id
    }

    pub fn add_rule(&mut self, rule_id: u32, rule: GrammarRule) {
        let rule_id = rule_id as usize;
        if self.rules.len() <= rule_id {
            self.rules.resize(rule_id + 1, GrammarRule::new());
        }
        self.rules[rule_id] = rule;
    }

    fn parse_token(
        &self,
        src: &[u8],
        mut pos: usize,
        type_: Gretype,
        rule: &mut GrammarRule,
    ) -> Result<usize> {
        if byte_at(src, pos) != b'<' {
            return Err(Error::new(format!("expecting '<' at {}", tail(src, pos))));
        }
        pos += 1;

        // Parse <[id,id-id,...]>
        if byte_at(src, pos) == b'[' {
            pos += 1;
            let start = rule.len();
            loop {
                let (token_id, np) = parse_token_id(src, pos)?;
                pos = np;
                let t = if rule.len() == start {
                    type_
                } else {
                    Gretype::TokenAlt
                };
                rule.push(GrammarElement::new(t, token_id));
                if byte_at(src, pos) == b'-' {
                    pos += 1;
                    let (end, np) = parse_token_id(src, pos)?;
                    if end < token_id {
                        return Err(Error::new(format!(
                            "invalid token range at {}",
                            tail(src, pos)
                        )));
                    }
                    pos = np;
                    rule.push(GrammarElement::new(Gretype::TokenRngUpper, end));
                }
                if byte_at(src, pos) != b',' {
                    break;
                }
                pos += 1;
            }
            if byte_at(src, pos) != b']' {
                return Err(Error::new(format!("expecting ']' at {}", tail(src, pos))));
            }
            pos += 1;
            if byte_at(src, pos) != b'>' {
                return Err(Error::new(format!("expecting '>' at {}", tail(src, pos))));
            }
            pos += 1;
            return Ok(pos);
        }

        let tokenize = self
            .tokenize
            .as_ref()
            .ok_or_else(|| Error::new(format!("no vocab to parse token at {}", tail(src, pos))))?;

        // Parse <token> and tokenize to obtain the token id
        let src_start = pos - 1; // include '<', as in C++
        while byte_at(src, pos) != 0 && byte_at(src, pos) != b'>' {
            pos += 1;
        }
        if byte_at(src, pos) != b'>' {
            return Err(Error::new(format!("expecting '>' at {}", tail(src, pos))));
        }
        pos += 1;

        let tokens = tokenize(&src[src_start..pos])?;
        if tokens.len() != 1 {
            // must tokenize to exactly 1 token
            return Err(Error::new(format!(
                "invalid token '{}'",
                String::from_utf8_lossy(&src[src_start..pos])
            )));
        }
        rule.push(GrammarElement::new(type_, tokens[0] as u32));
        Ok(pos)
    }

    pub fn parse_alternates(
        &mut self,
        src: &[u8],
        mut pos: usize,
        rule_name: &str,
        rule_id: u32,
        is_nested: bool,
        depth: u32,
    ) -> Result<usize> {
        let mut rule = GrammarRule::new();
        pos = self.parse_sequence(src, pos, rule_name, &mut rule, is_nested, depth)?;
        while byte_at(src, pos) == b'|' {
            rule.push(GrammarElement::new(Gretype::Alt, 0));
            pos = parse_space(src, pos + 1, true);
            pos = self.parse_sequence(src, pos, rule_name, &mut rule, is_nested, depth)?;
        }
        rule.push(GrammarElement::new(Gretype::End, 0));
        self.add_rule(rule_id, rule);
        Ok(pos)
    }

    pub fn parse_sequence(
        &mut self,
        src: &[u8],
        mut pos: usize,
        rule_name: &str,
        rule: &mut GrammarRule,
        is_nested: bool,
        depth: u32,
    ) -> Result<usize> {
        let mut last_sym_start = rule.len();
        let mut n_prev_rules: u64 = 1;

        while byte_at(src, pos) != 0 {
            if byte_at(src, pos) == b'"' {
                // literal string
                pos += 1;
                last_sym_start = rule.len();
                n_prev_rules = 1;
                while byte_at(src, pos) != b'"' {
                    if byte_at(src, pos) == 0 {
                        return Err(Error::new("unexpected end of input"));
                    }
                    let (value, np) = parse_char(src, pos)?;
                    pos = np;
                    rule.push(GrammarElement::new(Gretype::Char, value));
                }
                pos = parse_space(src, pos + 1, is_nested);
            } else if byte_at(src, pos) == b'[' {
                // char range(s)
                pos += 1;
                let mut start_type = Gretype::Char;
                if byte_at(src, pos) == b'^' {
                    pos += 1;
                    start_type = Gretype::CharNot;
                }
                last_sym_start = rule.len();
                n_prev_rules = 1;
                while byte_at(src, pos) != b']' {
                    if byte_at(src, pos) == 0 {
                        return Err(Error::new("unexpected end of input"));
                    }
                    let (value, np) = parse_char(src, pos)?;
                    pos = np;
                    let t = if last_sym_start < rule.len() {
                        Gretype::CharAlt
                    } else {
                        start_type
                    };
                    rule.push(GrammarElement::new(t, value));
                    if byte_at(src, pos) == b'-' && byte_at(src, pos + 1) != b']' {
                        if byte_at(src, pos + 1) == 0 {
                            return Err(Error::new("unexpected end of input"));
                        }
                        let (end, np) = parse_char(src, pos + 1)?;
                        pos = np;
                        rule.push(GrammarElement::new(Gretype::CharRngUpper, end));
                    }
                }
                pos = parse_space(src, pos + 1, is_nested);
            } else if byte_at(src, pos) == b'<' || byte_at(src, pos) == b'!' {
                // token
                let mut type_ = Gretype::Token;
                if byte_at(src, pos) == b'!' {
                    // token inverse
                    type_ = Gretype::TokenNot;
                    pos += 1;
                }
                last_sym_start = rule.len();
                n_prev_rules = 1;
                pos = self.parse_token(src, pos, type_, rule)?;
                pos = parse_space(src, pos, is_nested);
            } else if is_word_char(byte_at(src, pos)) {
                // rule reference
                let name_end = parse_name(src, pos)?;
                let ref_rule_id = self.get_symbol_id(&src[pos..name_end]);
                pos = parse_space(src, name_end, is_nested);
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(GrammarElement::new(Gretype::RuleRef, ref_rule_id));
            } else if byte_at(src, pos) == b'(' {
                // grouping
                if depth >= GRAMMAR_RECURSIVE_LIMIT {
                    return Err(Error::new(format!(
                        "grammar recursion depth limit exceeded (max {}), please reduce nesting depth",
                        GRAMMAR_RECURSIVE_LIMIT
                    )));
                }
                // parse nested alternates into synthesized rule
                pos = parse_space(src, pos + 1, true);
                let n_rules_before = self.symbol_ids.len() as u32;
                let sub_rule_id = self.generate_symbol_id(rule_name);
                pos = self.parse_alternates(src, pos, rule_name, sub_rule_id, true, depth + 1)?;
                n_prev_rules = (self.symbol_ids.len() as u32 - n_rules_before).max(1) as u64;
                last_sym_start = rule.len();
                // output reference to synthesized rule
                rule.push(GrammarElement::new(Gretype::RuleRef, sub_rule_id));
                if byte_at(src, pos) != b')' {
                    return Err(Error::new(format!("expecting ')' at {}", tail(src, pos))));
                }
                pos = parse_space(src, pos + 1, is_nested);
            } else if byte_at(src, pos) == b'.' {
                // any char
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(GrammarElement::new(Gretype::CharAny, 0));
                pos = parse_space(src, pos + 1, is_nested);
            } else if byte_at(src, pos) == b'*' {
                pos = parse_space(src, pos + 1, is_nested);
                handle_repetitions(
                    self,
                    rule,
                    rule_name,
                    src,
                    pos,
                    last_sym_start,
                    &mut n_prev_rules,
                    0,
                    u64::MAX,
                )?;
            } else if byte_at(src, pos) == b'+' {
                pos = parse_space(src, pos + 1, is_nested);
                handle_repetitions(
                    self,
                    rule,
                    rule_name,
                    src,
                    pos,
                    last_sym_start,
                    &mut n_prev_rules,
                    1,
                    u64::MAX,
                )?;
            } else if byte_at(src, pos) == b'?' {
                pos = parse_space(src, pos + 1, is_nested);
                handle_repetitions(
                    self,
                    rule,
                    rule_name,
                    src,
                    pos,
                    last_sym_start,
                    &mut n_prev_rules,
                    0,
                    1,
                )?;
            } else if byte_at(src, pos) == b'{' {
                pos = parse_space(src, pos + 1, is_nested);

                if !is_digit_char(byte_at(src, pos)) {
                    return Err(Error::new(format!(
                        "expecting an int at {}",
                        tail(src, pos)
                    )));
                }
                let int_end = parse_int(src, pos)?;
                let min_times: u64 = std::str::from_utf8(&src[pos..int_end])
                    .map_err(|_| Error::new("invalid repetition count"))?
                    .parse()
                    .map_err(|_| Error::new("invalid repetition count"))?;
                pos = parse_space(src, int_end, is_nested);

                let mut max_times = u64::MAX; // default: no max limit
                if byte_at(src, pos) == b'}' {
                    max_times = min_times;
                    pos = parse_space(src, pos + 1, is_nested);
                } else if byte_at(src, pos) == b',' {
                    pos = parse_space(src, pos + 1, is_nested);
                    if is_digit_char(byte_at(src, pos)) {
                        let int_end = parse_int(src, pos)?;
                        max_times = std::str::from_utf8(&src[pos..int_end])
                            .map_err(|_| Error::new("invalid repetition count"))?
                            .parse()
                            .map_err(|_| Error::new("invalid repetition count"))?;
                        pos = parse_space(src, int_end, is_nested);
                    }
                    if byte_at(src, pos) != b'}' {
                        return Err(Error::new(format!("expecting '}}' at {}", tail(src, pos))));
                    }
                    pos = parse_space(src, pos + 1, is_nested);
                } else {
                    return Err(Error::new(format!("expecting ',' at {}", tail(src, pos))));
                }
                if min_times > MAX_REPETITION_THRESHOLD {
                    return Err(Error::new(
                        "number of repetitions exceeds sane defaults, please reduce the number of repetitions",
                    ));
                }
                if max_times != u64::MAX && max_times > MAX_REPETITION_THRESHOLD {
                    max_times = u64::MAX;
                }
                if max_times < min_times {
                    return Err(Error::new("incorrect number of repetitions"));
                }
                handle_repetitions(
                    self,
                    rule,
                    rule_name,
                    src,
                    pos,
                    last_sym_start,
                    &mut n_prev_rules,
                    min_times,
                    max_times,
                )?;
            } else {
                break;
            }
        }
        Ok(pos)
    }

    pub fn parse_rule(&mut self, src: &[u8], pos: usize) -> Result<usize> {
        let name_end = parse_name(src, pos)?;
        let mut p = parse_space(src, name_end, false);
        let name_bytes = src[pos..name_end].to_vec();
        let rule_id = self.get_symbol_id(&name_bytes);
        let name = String::from_utf8_lossy(&name_bytes).into_owned();

        if !(byte_at(src, p) == b':' && byte_at(src, p + 1) == b':' && byte_at(src, p + 2) == b'=')
        {
            return Err(Error::new(format!("expecting ::= at {}", tail(src, p))));
        }
        p = parse_space(src, p + 3, true);

        p = self.parse_alternates(src, p, &name, rule_id, false, 0)?;

        if byte_at(src, p) == b'\r' {
            p += if byte_at(src, p + 1) == b'\n' { 2 } else { 1 };
        } else if byte_at(src, p) == b'\n' {
            p += 1;
        } else if byte_at(src, p) != 0 {
            return Err(Error::new(format!(
                "expecting newline or end at {}",
                tail(src, p)
            )));
        }
        Ok(parse_space(src, p, true))
    }

    /// Error-propagating form of `parse`.
    pub fn parse_result(&mut self, src: &str) -> Result<()> {
        let bytes = src.as_bytes();
        let mut pos = parse_space(bytes, 0, true);
        while byte_at(bytes, pos) != 0 {
            pos = self.parse_rule(bytes, pos)?;
        }

        // Validate the state to ensure that all rules are defined
        for rule in &self.rules {
            if rule.is_empty() {
                return Err(Error::new("Undefined rule"));
            }
            for elem in rule {
                if elem.type_ == Gretype::RuleRef {
                    let id = elem.value as usize;
                    if id >= self.rules.len() || self.rules[id].is_empty() {
                        // Get the name of the rule that is missing
                        for (name, rule_id) in &self.symbol_ids {
                            if *rule_id == elem.value {
                                return Err(Error::new(format!(
                                    "Undefined rule identifier '{}'",
                                    name
                                )));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// C++ `bool parse(const char * src)`: clears rules on error.
    pub fn parse(&mut self, src: &str) -> bool {
        match self.parse_result(src) {
            Ok(()) => true,
            Err(err) => {
                eprintln!("parse: error parsing grammar: {}\n\n{}", err, src);
                self.rules.clear();
                false
            }
        }
    }

    /// Text form of C++ `print(FILE *)`.
    pub fn print(&self) -> String {
        let mut symbol_id_names = BTreeMap::new();
        for (name, id) in &self.symbol_ids {
            symbol_id_names.insert(*id, name.clone());
        }
        let mut out = String::new();
        for (i, rule) in self.rules.iter().enumerate() {
            match print_rule(i as u32, rule, &symbol_id_names) {
                Ok(text) => out.push_str(&text),
                Err(err) => eprintln!("\nprint: error printing grammar: {}", err),
            }
        }
        out
    }
}

impl Default for GrammarParser {
    fn default() -> Self {
        Self::new()
    }
}

fn handle_repetitions(
    parser: &mut GrammarParser,
    rule: &mut GrammarRule,
    rule_name: &str,
    src: &[u8],
    pos: usize,
    last_sym_start: usize,
    n_prev_rules: &mut u64,
    min_times: u64,
    max_times: u64,
) -> Result<()> {
    let no_max = max_times == u64::MAX;
    if last_sym_start == rule.len() {
        return Err(Error::new(format!(
            "expecting preceding item to */+/?/{{ at {}",
            tail(src, pos)
        )));
    }

    // apply transformation to previous symbol (last_sym_start to end) according to
    // the following rewrite rules:
    // S{m,n} --> S S S (m times) S'(n-m)
    //            S'(x)   ::= S S'(x-1) |
    //            (... n-m definitions of these S' rules ...)
    //            S'(1)   ::= S |
    // S{m,} -->  S S S (m times) S'
    //            S'     ::= S S' |
    // S*     --> S{0,}
    //        --> S'     ::= S S' |
    // S+     --> S{1,}
    //        --> S S'
    //            S'     ::= S S' |
    // S?     --> S{0,1}
    //        --> S'
    //            S'     ::= S |

    let prev_rule = rule[last_sym_start..].to_vec();
    // Calculate the total number of rules that will be generated by this repetition
    let mut total_rules = 1u64;
    if !no_max && max_times > 0 {
        total_rules = max_times;
    } else if min_times > 0 {
        total_rules = min_times;
    }

    if *n_prev_rules * total_rules > MAX_REPETITION_THRESHOLD {
        return Err(Error::new(
            "number of rules that are going to be repeated multiplied by the new repetition exceeds sane defaults, please reduce the number of repetitions or rule complexity",
        ));
    }

    if min_times == 0 {
        rule.truncate(last_sym_start);
    } else {
        // Repeat the previous elements (min_times - 1) times
        for _ in 1..min_times {
            rule.extend(prev_rule.iter().cloned());
        }
    }

    let mut last_rec_rule_id = 0u32;
    let n_opt = if no_max { 1 } else { max_times - min_times };

    let mut rec_rule = prev_rule.clone();
    for i in 0..n_opt {
        rec_rule.truncate(prev_rule.len());
        let rec_rule_id = parser.generate_symbol_id(rule_name);
        if i > 0 || no_max {
            let target = if no_max {
                rec_rule_id
            } else {
                last_rec_rule_id
            };
            rec_rule.push(GrammarElement::new(Gretype::RuleRef, target));
        }
        rec_rule.push(GrammarElement::new(Gretype::Alt, 0));
        rec_rule.push(GrammarElement::new(Gretype::End, 0));
        parser.add_rule(rec_rule_id, rec_rule.clone());
        last_rec_rule_id = rec_rule_id;
    }
    if n_opt > 0 {
        rule.push(GrammarElement::new(Gretype::RuleRef, last_rec_rule_id));
    }
    *n_prev_rules *= total_rules;
    debug_assert!(*n_prev_rules >= 1);
    Ok(())
}
// ---------------------------------------------------------------------------
// left factoring / left recursion (`struct llama_grammar` init support)
// ---------------------------------------------------------------------------

fn trim_prefix(rule: &GrammarRule, start: usize, mut length: usize) -> usize {
    // Keep the prefix boundary outside character classes.
    while length > 0 {
        let type_ = rule[start + length].type_;
        if type_ != Gretype::CharAlt && type_ != Gretype::CharRngUpper {
            break;
        }
        length -= 1;
    }
    length
}

// Left-factor grammar rules to reduce redundant parallel stacks.
//
// When a rule has many alternates (e.g. tool-choice ::= tc0 | tc1 | ... | tc119)
// and the referenced rules share a common element-content prefix, extract that
// prefix so the parser processes it once instead of N times.
//
// Handles two cases:
// 1. Inline common prefix: rule = "ab1" | "ab2" -> rule = "ab" rule', rule' = "1" | "2"
// 2. Cross-rule prefix: rule = A | B | C where A,B,C start with same elements
//    -> rule = common_prefix rule', rule' = A_suffix | B_suffix | C_suffix
pub(crate) fn left_factor(rules: &mut GrammarRules) {
    let n_rules_original = rules.len();
    // Original rules factor at most once; generated rules do not resolve references.
    // Each new suffix removes a non-empty prefix.
    let mut changed = true;
    while changed {
        changed = false;
        let n_rules = rules.len();

        for ri in 0..n_rules {
            // Split rule into alternates
            let mut alt_ranges: Vec<(usize, usize)> = Vec::new();
            let mut alt_start = 0usize;
            for (j, elem) in rules[ri].iter().enumerate() {
                if matches!(elem.type_, Gretype::Alt | Gretype::End) {
                    alt_ranges.push((alt_start, j));
                    alt_start = j + 1;
                }
            }

            if alt_ranges.len() < 4 {
                continue;
            }

            // Resolve each alternate: if it's a single RULE_REF, follow it to get the
            // actual element sequence from the referenced rule.
            let mut resolved: Vec<(usize, usize, usize)> = Vec::with_capacity(alt_ranges.len());
            let mut all_single_refs = true;

            for &(astart, aend) in &alt_ranges {
                let len = aend - astart;
                // Resolving references in generated suffixes can expand recursive rules forever.
                if ri < n_rules_original && len == 1 && rules[ri][astart].type_ == Gretype::RuleRef
                {
                    let ref_id = rules[ri][astart].value as usize;
                    if ref_id < rules.len() {
                        let mut ref_end = 0usize;
                        for (k, elem) in rules[ref_id].iter().enumerate() {
                            if matches!(elem.type_, Gretype::Alt | Gretype::End) {
                                ref_end = k;
                                break;
                            }
                        }
                        let single_alt = rules[ref_id][ref_end].type_ == Gretype::End;
                        if single_alt {
                            resolved.push((ref_id, 0, ref_end));
                            continue;
                        }
                    }
                }
                all_single_refs = false;
                resolved.push((ri, astart, aend));
            }

            if !all_single_refs {
                resolved = alt_ranges.iter().map(|&(a, b)| (ri, a, b)).collect();
            }

            let mut prefix_len = 0usize;
            let mut prefix_match = true;
            while prefix_match {
                let (r0, s0, e0) = resolved[0];
                if s0 + prefix_len >= e0 {
                    break;
                }
                let ref_elem = rules[r0][s0 + prefix_len];
                for &(rr, ss, ee) in &resolved[1..] {
                    if ss + prefix_len >= ee || rules[rr][ss + prefix_len] != ref_elem {
                        prefix_match = false;
                        break;
                    }
                }
                if prefix_match {
                    prefix_len += 1;
                }
            }

            for &(rr, ss, _) in &resolved {
                prefix_len = trim_prefix(&rules[rr], ss, prefix_len);
            }
            if prefix_len == 0 {
                continue;
            }

            let suffix_choice_id = rules.len() as u32;
            let mut suffix_choice = GrammarRule::new();

            for (ai, &(rr, ss, ee)) in resolved.iter().enumerate() {
                if ai > 0 {
                    suffix_choice.push(GrammarElement::new(Gretype::Alt, 0));
                }
                for j in ss + prefix_len..ee {
                    suffix_choice.push(rules[rr][j]);
                }
            }
            suffix_choice.push(GrammarElement::new(Gretype::End, 0));

            let mut new_rule = GrammarRule::new();
            let (r0, s0, _) = resolved[0];
            for j in 0..prefix_len {
                new_rule.push(rules[r0][s0 + j]);
            }
            new_rule.push(GrammarElement::new(Gretype::RuleRef, suffix_choice_id));
            new_rule.push(GrammarElement::new(Gretype::End, 0));

            rules.push(suffix_choice);
            rules[ri] = new_rule;
            changed = true;
        }
    }
}

pub(crate) fn detect_left_recursion(
    rules: &[GrammarRule],
    rule_index: usize,
    rules_visited: &mut [bool],
    rules_in_progress: &mut [bool],
    rules_may_be_empty: &mut [bool],
    depth: u32,
) -> Result<bool> {
    if rules_in_progress[rule_index] {
        return Ok(true);
    }
    if depth > GRAMMAR_RECURSIVE_LIMIT {
        return Err(Error::new(format!(
            "grammar rules are chained more than {} levels deep",
            GRAMMAR_RECURSIVE_LIMIT
        )));
    }

    rules_in_progress[rule_index] = true;
    let rule = &rules[rule_index];

    // First check if the rule might produce the empty string. This could be done combined with the second
    // step but it's more readable as two steps.
    let mut at_rule_start = true;
    for elem in rule {
        if is_end_of_sequence(elem) {
            if at_rule_start {
                rules_may_be_empty[rule_index] = true;
                break;
            }
            at_rule_start = true;
        } else {
            at_rule_start = false;
        }
    }

    // Second, recurse into leftmost nonterminals (or next-leftmost as long as the previous nonterminal may
    // be empty)
    let mut recurse_into_nonterminal = true;
    for elem in rule {
        if elem.type_ == Gretype::RuleRef && recurse_into_nonterminal {
            let id = elem.value as usize;
            if detect_left_recursion(
                rules,
                id,
                rules_visited,
                rules_in_progress,
                rules_may_be_empty,
                depth + 1,
            )? {
                return Ok(true);
            }
            if !rules_may_be_empty[id] {
                recurse_into_nonterminal = false;
            }
        } else if is_end_of_sequence(elem) {
            recurse_into_nonterminal = true;
        } else {
            recurse_into_nonterminal = false;
        }
    }

    rules_in_progress[rule_index] = false;
    rules_visited[rule_index] = true;

    Ok(false)
}

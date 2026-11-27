//! Stack state machine, char/token matching, rejection, and public grammar API.

use crate::core::{Error, Result, Token};
use crate::grammar::parser::{detect_left_recursion, left_factor, GrammarParser, TokenizeFn};
use crate::grammar::types::*;
use crate::grammar::utf8::decode_utf8;
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// printing helpers
// ---------------------------------------------------------------------------

fn print_grammar_char(c: u32) -> String {
    if (0x20..=0x7f).contains(&c) {
        (c as u8 as char).to_string()
    } else {
        // cop out of encoding UTF-8
        format!("<U+{:04X}>", c)
    }
}

fn type_name(t: Gretype) -> &'static str {
    match t {
        Gretype::End => "END",
        Gretype::Alt => "ALT",
        Gretype::RuleRef => "RULE_REF",
        Gretype::Char => "CHAR",
        Gretype::CharNot => "CHAR_NOT",
        Gretype::CharRngUpper => "CHAR_RNG_UPPER",
        Gretype::CharAlt => "CHAR_ALT",
        Gretype::CharAny => "CHAR_ANY",
        Gretype::Token => "TOKEN",
        Gretype::TokenNot => "TOKEN_NOT",
        Gretype::TokenRngUpper => "TOKEN_RNG_UPPER",
        Gretype::TokenAlt => "TOKEN_ALT",
    }
}

fn is_token_element(elem: &GrammarElement) -> bool {
    matches!(
        elem.type_,
        Gretype::Token | Gretype::TokenNot | Gretype::TokenAlt | Gretype::TokenRngUpper
    )
}

fn is_char_element(elem: &GrammarElement) -> bool {
    matches!(
        elem.type_,
        Gretype::Char
            | Gretype::CharNot
            | Gretype::CharAlt
            | Gretype::CharRngUpper
            | Gretype::CharAny
    )
}

/// Debug view of one rule, mirroring `print_rule_binary`.
pub fn print_rule_binary(rule: &GrammarRule) -> String {
    let mut out = String::new();
    for elem in rule {
        out.push_str(type_name(elem.type_));
        match elem.type_ {
            Gretype::End | Gretype::Alt | Gretype::RuleRef => {
                out.push_str(&format!("({}) ", elem.value));
            }
            Gretype::Char
            | Gretype::CharNot
            | Gretype::CharRngUpper
            | Gretype::CharAlt
            | Gretype::CharAny => {
                out.push_str("(\"");
                out.push_str(&print_grammar_char(elem.value));
                out.push_str("\") ");
            }
            Gretype::Token => {
                out.push_str(&format!("<[{}]> ", elem.value));
            }
            Gretype::TokenNot => {
                out.push_str(&format!("!<[{}]> ", elem.value));
            }
            Gretype::TokenRngUpper | Gretype::TokenAlt => {
                out.push_str(&format!("({}) ", elem.value));
            }
        }
    }
    out.push('\n');
    out
}

pub(crate) fn print_rule(
    rule_id: u32,
    rule: &GrammarRule,
    symbol_id_names: &BTreeMap<u32, String>,
) -> Result<String> {
    if rule.is_empty() || rule.last().unwrap().type_ != Gretype::End {
        return Err(Error::new(format!(
            "malformed rule, does not end with LLAMA_GRETYPE_END: {}",
            rule_id
        )));
    }
    let name = symbol_id_names
        .get(&rule_id)
        .ok_or_else(|| Error::new(format!("no name for rule {}", rule_id)))?;
    let mut out = format!("{} ::= ", name);
    let end = rule.len() - 1;
    for i in 0..end {
        let elem = &rule[i];
        match elem.type_ {
            Gretype::End => {
                return Err(Error::new(format!(
                    "unexpected end of rule: {},{}",
                    rule_id, i
                )));
            }
            Gretype::Alt => out.push_str("| "),
            Gretype::RuleRef => {
                let n = symbol_id_names
                    .get(&elem.value)
                    .ok_or_else(|| Error::new(format!("no name for rule {}", elem.value)))?;
                out.push_str(n);
                out.push(' ');
            }
            Gretype::Char => {
                out.push('[');
                out.push_str(&print_grammar_char(elem.value));
            }
            Gretype::CharNot => {
                out.push_str("[^");
                out.push_str(&print_grammar_char(elem.value));
            }
            Gretype::CharRngUpper => {
                if i == 0 || !is_char_element(&rule[i - 1]) {
                    return Err(Error::new(format!(
                        "LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: {},{}",
                        rule_id, i
                    )));
                }
                out.push('-');
                out.push_str(&print_grammar_char(elem.value));
            }
            Gretype::CharAlt => {
                if i == 0 || !is_char_element(&rule[i - 1]) {
                    return Err(Error::new(format!(
                        "LLAMA_GRETYPE_CHAR_ALT without preceding char: {},{}",
                        rule_id, i
                    )));
                }
                out.push_str(&print_grammar_char(elem.value));
            }
            Gretype::CharAny => out.push('.'),
            Gretype::Token => {
                out.push_str("<[");
                out.push_str(&elem.value.to_string());
            }
            Gretype::TokenNot => {
                out.push('!');
                out.push_str("<[");
                out.push_str(&elem.value.to_string());
            }
            Gretype::TokenRngUpper => {
                if i == 0 || !is_token_element(&rule[i - 1]) {
                    return Err(Error::new(format!(
                        "LLAMA_GRETYPE_TOKEN_RNG_UPPER without preceding token: {},{}",
                        rule_id, i
                    )));
                }
                out.push_str(&format!("-{}", elem.value));
            }
            Gretype::TokenAlt => {
                if i == 0 || !is_token_element(&rule[i - 1]) {
                    return Err(Error::new(format!(
                        "LLAMA_GRETYPE_TOKEN_ALT without preceding token: {},{}",
                        rule_id, i
                    )));
                }
                out.push_str(&format!(",{}", elem.value));
            }
        }
        if is_token_element(elem)
            && !matches!(
                rule[i + 1].type_,
                Gretype::TokenAlt | Gretype::TokenRngUpper
            )
        {
            out.push_str("]> ");
        }
        if is_char_element(elem) {
            match rule[i + 1].type_ {
                Gretype::CharAlt | Gretype::CharRngUpper | Gretype::CharAny => {}
                _ => out.push_str("] "),
            }
        }
    }
    out.push('\n');
    Ok(out)
}

// ---------------------------------------------------------------------------
// lazy grammar triggers
// ---------------------------------------------------------------------------

/// `llama_grammar_trigger_pattern`.
#[derive(Clone, Debug)]
pub struct TriggerPattern {
    pub pattern: String,
}

impl TriggerPattern {
    /// Returns the start of the match.
    /// shortcut: std has no regex; literal patterns with optional `^`/`$`
    /// anchors are supported. Use a precomputed match position for full regex.
    pub fn find(&self, input: &str) -> Option<usize> {
        let p = self.pattern.as_str();
        if p.len() >= 2 && p.starts_with('^') && p.ends_with('$') {
            let inner = &p[1..p.len() - 1];
            return if input == inner { Some(0) } else { None };
        }
        let inner = p
            .strip_prefix('^')
            .or_else(|| p.strip_suffix('$'))
            .unwrap_or(p);
        input.find(inner)
    }
}
// ---------------------------------------------------------------------------
// state machine helpers
// ---------------------------------------------------------------------------

fn elem<'a>(rules: &'a [GrammarRule], pos: GrammarPos) -> &'a GrammarElement {
    &rules[pos.rule][pos.index]
}

// returns true iff pos points to the end of one of the definitions of a rule
pub(crate) fn is_end_of_sequence(elem: &GrammarElement) -> bool {
    matches!(elem.type_, Gretype::End | Gretype::Alt) // NOLINT
}

// returns true iff chr satisfies the char range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
fn match_char(rules: &[GrammarRule], pos: GrammarPos, chr: u32) -> (bool, GrammarPos) {
    let mut p = pos;
    let mut found = false;
    let is_positive_char = matches!(elem(rules, p).type_, Gretype::Char | Gretype::CharAny);

    loop {
        if elem(rules, p.advance(1)).type_ == Gretype::CharRngUpper {
            // inclusive range, e.g. [a-z]
            found =
                found || (elem(rules, p).value <= chr && chr <= elem(rules, p.advance(1)).value);
            p = p.advance(2);
        } else if elem(rules, p).type_ == Gretype::CharAny {
            // Any character matches "."
            found = true;
            p = p.advance(1);
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || elem(rules, p).value == chr;
            p = p.advance(1);
        }
        if elem(rules, p).type_ != Gretype::CharAlt {
            break;
        }
    }

    (found == is_positive_char, p)
}

// returns true iff some continuation of the given partial UTF-8 sequence could satisfy the char
// range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
fn match_partial_char(rules: &[GrammarRule], pos: GrammarPos, partial_utf8: PartialUtf8) -> bool {
    let mut p = pos;
    let is_positive_char = matches!(elem(rules, p).type_, Gretype::Char | Gretype::CharAny);

    let partial_value = partial_utf8.value;
    let n_remain = partial_utf8.n_remain;

    // invalid sequence or 7-bit char split across 2 bytes (overlong)
    if n_remain < 0 || (n_remain == 1 && partial_value < 2) {
        return false;
    }

    // range of possible code points this partial UTF-8 sequence could complete to
    let shift = (n_remain * 6) as u32;
    let mut low = partial_value.wrapping_shl(shift);
    let high = low | 1u32.wrapping_shl(shift).wrapping_sub(1);

    if low == 0 {
        if n_remain == 2 {
            low = 1 << 11;
        } else if n_remain == 3 {
            low = 1 << 16;
        }
    }

    loop {
        if elem(rules, p.advance(1)).type_ == Gretype::CharRngUpper {
            // inclusive range, e.g. [a-z]
            if elem(rules, p).value <= high && low <= elem(rules, p.advance(1)).value {
                return is_positive_char;
            }
            p = p.advance(2);
        } else if elem(rules, p).type_ == Gretype::CharAny {
            // Any character matches "."
            return true;
        } else {
            // exact char match, e.g. [a] or "a"
            if low <= elem(rules, p).value && elem(rules, p).value <= high {
                return is_positive_char;
            }
            p = p.advance(1);
        }
        if elem(rules, p).type_ != Gretype::CharAlt {
            break;
        }
    }

    !is_positive_char
}

// returns true iff token matches the rule at pos (regular or inverse)
// asserts that pos is pointing to a token element
fn match_token(rules: &[GrammarRule], pos: GrammarPos, token: Token) -> (bool, GrammarPos) {
    let mut p = pos;
    let is_positive_token = elem(rules, p).type_ == Gretype::Token;

    let id = token as u32;
    let mut found = false;
    loop {
        if elem(rules, p.advance(1)).type_ == Gretype::TokenRngUpper {
            // inclusive range, e.g. <[1-5]>
            found = found || (elem(rules, p).value <= id && id <= elem(rules, p.advance(1)).value);
            p = p.advance(2);
        } else {
            found = found || elem(rules, p).value == id;
            p = p.advance(1);
        }
        if elem(rules, p).type_ != Gretype::TokenAlt {
            break;
        }
    }

    (found == is_positive_token, p)
}

// every token id a token rule names, merged into sorted inclusive ranges
fn opaque_tokens(rules: &[GrammarRule]) -> Vec<(u32, u32)> {
    let mut ranges = Vec::new();
    for rule in rules {
        for (i, elem) in rule.iter().enumerate() {
            let type_ = elem.type_;
            if matches!(
                type_,
                Gretype::Token | Gretype::TokenNot | Gretype::TokenAlt
            ) {
                let is_range = i + 1 < rule.len() && rule[i + 1].type_ == Gretype::TokenRngUpper;
                ranges.push((
                    elem.value,
                    if is_range {
                        rule[i + 1].value
                    } else {
                        elem.value
                    },
                ));
            }
        }
    }
    ranges.sort_unstable();
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for r in ranges {
        if let Some(last) = merged.last_mut() {
            if r.0 as u64 <= last.1 as u64 + 1 {
                last.1 = last.1.max(r.1);
                continue;
            }
        }
        merged.push(r);
    }
    merged
}

// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
fn advance_stack(rules: &[GrammarRule], stack: &GrammarStack, new_stacks: &mut GrammarStacks) {
    let mut todo = vec![stack.clone()];
    let mut seen: BTreeSet<GrammarStack> = BTreeSet::new();

    while let Some(curr) = todo.pop() {
        if !seen.insert(curr.clone()) {
            continue;
        }

        if curr.is_empty() {
            if !new_stacks.contains(&curr) {
                new_stacks.push(curr);
            }
            continue;
        }

        let pos = *curr.last().unwrap();
        match elem(rules, pos).type_ {
            Gretype::RuleRef => {
                let rule_id = elem(rules, pos).value as usize;
                let mut subpos = GrammarPos::new(rule_id, 0);
                loop {
                    // init new stack without the top (pos)
                    let mut next_stack = curr.clone();
                    next_stack.pop();
                    if !is_end_of_sequence(elem(rules, pos.advance(1))) {
                        // if this rule ref is followed by another element, add that to stack
                        next_stack.push(pos.advance(1));
                    }
                    if !is_end_of_sequence(elem(rules, subpos)) {
                        // if alternate is nonempty, add to stack
                        next_stack.push(subpos);
                    }
                    todo.push(next_stack);
                    while !is_end_of_sequence(elem(rules, subpos)) {
                        // scan to end of alternate def
                        subpos = subpos.advance(1);
                    }
                    if elem(rules, subpos).type_ == Gretype::Alt {
                        // there's another alternate def of this rule to process
                        subpos = subpos.advance(1);
                    } else {
                        break;
                    }
                }
            }
            Gretype::Char
            | Gretype::CharNot
            | Gretype::CharAny
            | Gretype::Token
            | Gretype::TokenNot => {
                // only add the stack if it's not a duplicate of one we already have
                if !new_stacks.contains(&curr) {
                    new_stacks.push(curr);
                }
            }
            _ => {
                // end of alternate (END, ALT) or middle of char range / token set;
                // stack should never be left on those
                unreachable!("llama_grammar_advance_stack: fatal error");
            }
        }
    }
}

fn accept_chr(
    rules: &[GrammarRule],
    stack: &GrammarStack,
    chr: u32,
    new_stacks: &mut GrammarStacks,
) {
    if stack.is_empty() {
        return;
    }

    let pos = *stack.last().unwrap();

    // ignore if this turns into a token
    if matches!(elem(rules, pos).type_, Gretype::Token | Gretype::TokenNot) {
        return;
    }

    let (matched, after) = match_char(rules, pos, chr);
    if matched {
        let mut new_stack = stack.clone();
        new_stack.pop();
        if !is_end_of_sequence(elem(rules, after)) {
            new_stack.push(after);
        }
        advance_stack(rules, &new_stack, new_stacks);
    }
}

fn cp_at(code_points: &[u32], depth: usize) -> u32 {
    code_points.get(depth).copied().unwrap_or(0)
}

// checks the code point at depth of each candidate (an index into the input) against the stack
// returns true when some match: they go to next, and the stacks that follow to next_stacks
fn reject_step(
    rules: &[GrammarRule],
    stack: &GrammarStack,
    candidates: &[GrammarCandidate],
    depth: usize,
    cur: &[u32],
    rejects: &mut Vec<u32>,
    next: &mut Vec<u32>,
    next_stacks: &mut GrammarStacks,
) -> bool {
    if stack.is_empty() {
        for &i in cur {
            let tok = &candidates[i as usize];
            if cp_at(&tok.code_points, depth) != 0 || tok.partial_utf8.n_remain != 0 {
                rejects.push(i);
            }
        }
        return false;
    }

    let stack_pos = *stack.last().unwrap();

    // if the top of the stack is a token rule, then we only need to check the token id
    if matches!(
        elem(rules, stack_pos).type_,
        Gretype::Token | Gretype::TokenNot
    ) {
        for &i in cur {
            let tok = &candidates[i as usize];
            if depth > 0 {
                // char rules already consumed part of this token, so it cannot match by id.
                // accept it only if they consumed all of it and did not end in a partial sequence.
                if cp_at(&tok.code_points, depth) != 0 || tok.partial_utf8.n_remain != 0 {
                    rejects.push(i);
                }
            } else if tok.partial_utf8.n_remain < 0 || !match_token(rules, stack_pos, tok.id).0 {
                rejects.push(i);
            }
        }
        return false;
    }

    for &i in cur {
        let tok = &candidates[i as usize];
        if tok.opaque {
            // a token named by a token rule only matches token rules
            rejects.push(i);
        } else if cp_at(&tok.code_points, depth) == 0 {
            // reached end of full codepoints in token, reject iff it ended in a partial sequence
            // that cannot satisfy this position in grammar
            if tok.partial_utf8.n_remain != 0
                && !match_partial_char(rules, stack_pos, tok.partial_utf8)
            {
                rejects.push(i);
            }
        } else if match_char(rules, stack_pos, cp_at(&tok.code_points, depth)).0 {
            next.push(i);
        } else {
            rejects.push(i);
        }
    }

    let stack_pos_after = match_char(rules, stack_pos, 0).1;

    // update top of stack to next element, if any
    let mut stack_after = stack.clone();
    stack_after.pop();
    if !is_end_of_sequence(elem(rules, stack_pos_after)) {
        stack_after.push(stack_pos_after);
    }
    advance_stack(rules, &stack_after, next_stacks);

    true
}

// the candidates that no stack accepts, each stack narrows the rejects of the previous one
// no recursion: a level per code point, a token from the GGUF can be long
fn reject_candidates(
    rules: &[GrammarRule],
    stacks: &GrammarStacks,
    candidates: &[GrammarCandidate],
) -> GrammarCandidates {
    // a level checks the candidates that reach its code point against its stacks in turn
    struct Level {
        stacks: GrammarStacks,
        i: usize,          // the stack in progress
        cur: Vec<u32>,     // the candidates rejected by stacks[0, i)
        rejects: Vec<u32>, // the rejects of stacks[i] at this code point
    }

    let all: Vec<u32> = (0..candidates.len() as u32).collect();
    let mut levels = vec![Level {
        stacks: stacks.clone(),
        i: 0,
        cur: all,
        rejects: Vec::new(),
    }];

    loop {
        let done = {
            let lv = levels.last().unwrap();
            lv.i == lv.stacks.len() || lv.cur.is_empty()
        };
        if done {
            if levels.len() == 1 {
                break;
            }
            // the rejects at the next code point belong to stacks[i] of the parent level
            let lv = levels.pop().unwrap();
            let parent = levels.last_mut().unwrap();
            parent.rejects.extend(lv.cur);
            parent.cur = std::mem::take(&mut parent.rejects);
            parent.i += 1;
            continue;
        }

        let depth = levels.len() - 1;
        let mut rejects = Vec::new();
        let mut next = Vec::new();
        let mut next_stacks = GrammarStacks::new();
        let advanced = {
            let lv = levels.last().unwrap();
            reject_step(
                rules,
                &lv.stacks[lv.i],
                candidates,
                depth,
                &lv.cur,
                &mut rejects,
                &mut next,
                &mut next_stacks,
            )
        };
        if !advanced {
            let lv = levels.last_mut().unwrap();
            lv.cur = rejects;
            lv.i += 1;
            continue;
        }

        let lv = levels.last_mut().unwrap();
        lv.rejects = rejects;
        levels.push(Level {
            stacks: next_stacks,
            i: 0,
            cur: next,
            rejects: Vec::new(),
        });
    }

    let mut rejects = GrammarCandidates::with_capacity(levels[0].cur.len());
    for &i in &levels[0].cur {
        rejects.push(candidates[i as usize].clone());
    }
    rejects
}

/// Port of `llama_grammar_reject_candidates_for_stack`.
pub fn reject_candidates_for_stack(
    rules: &[GrammarRule],
    stack: &GrammarStack,
    candidates: &[GrammarCandidate],
) -> GrammarCandidates {
    let stacks = vec![stack.clone()];
    reject_candidates(rules, &stacks, candidates)
}
// ---------------------------------------------------------------------------
// grammar struct and public API (`struct llama_grammar`)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Grammar {
    rules: GrammarRules,
    stacks: GrammarStacks,
    // buffer for partially generated UTF-8 sequence from accepted tokens
    partial_utf8: PartialUtf8,
    #[allow(dead_code)] // kept for fidelity with the C++ grammar struct
    lazy: bool,
    awaiting_trigger: bool, // initialized to true for lazy grammars only
    trigger_buffer: Vec<u8>,
    trigger_buffer_positions: Vec<(Token, (usize, usize))>,
    trigger_tokens: Vec<Token>,
    trigger_patterns: Vec<TriggerPattern>,
    // token ids named by token rules, as sorted inclusive ranges.
    // char rules never match these tokens.
    opaque_tokens: Vec<(u32, u32)>,
}

impl Grammar {
    // note: allow null vocab for testing (not great) -- there is no vocab here at all
    fn build(
        src_rules: &[GrammarRule],
        start_rule_index: usize,
        lazy: bool,
        trigger_tokens: Vec<Token>,
        trigger_patterns: Vec<TriggerPattern>,
    ) -> Option<Self> {
        // copy rule definitions into vectors
        let mut vec_rules: GrammarRules = src_rules
            .iter()
            .map(|rule| {
                let mut v: GrammarRule = rule
                    .iter()
                    .take_while(|e| e.type_ != Gretype::End)
                    .cloned()
                    .collect();
                v.push(GrammarElement::new(Gretype::End, 0));
                v
            })
            .collect();

        // Validate that all rule references point to valid rules
        for (i, rule) in vec_rules.iter().enumerate() {
            for elem in rule {
                if elem.type_ == Gretype::RuleRef {
                    let id = elem.value as usize;
                    if id >= vec_rules.len() || vec_rules[id].is_empty() {
                        eprintln!(
                            "invalid grammar: rule {} references undefined rule {}",
                            i, elem.value
                        );
                        return None;
                    }
                }
            }
        }

        // Left-factor rules to reduce redundant parallel stacks
        left_factor(&mut vec_rules);

        // Check for left recursion (use post-factoring rule count)
        let n_rules_total = vec_rules.len();
        let mut rules_visited = vec![false; n_rules_total];
        let mut rules_in_progress = vec![false; n_rules_total];
        let mut rules_may_be_empty = vec![false; n_rules_total];
        for i in 0..n_rules_total {
            if rules_visited[i] {
                continue;
            }
            match detect_left_recursion(
                &vec_rules,
                i,
                &mut rules_visited,
                &mut rules_in_progress,
                &mut rules_may_be_empty,
                0,
            ) {
                Ok(true) => {
                    eprintln!(
                        "unsupported grammar, left recursion detected for nonterminal at index {}",
                        i
                    );
                    return None;
                }
                Ok(false) => {}
                Err(err) => {
                    eprintln!("unsupported grammar: {}", err);
                    return None;
                }
            }
        }

        // loop over alternates of start rule to build initial stacks
        let mut stacks = GrammarStacks::new();
        let mut pos = GrammarPos::new(start_rule_index, 0);
        loop {
            let mut stack = GrammarStack::new();
            if !is_end_of_sequence(elem(&vec_rules, pos)) {
                // if alternate is nonempty, add to stack
                stack.push(pos);
            }
            advance_stack(&vec_rules, &stack, &mut stacks);
            while !is_end_of_sequence(elem(&vec_rules, pos)) {
                // scan to end of alternate def
                pos = pos.advance(1);
            }
            if elem(&vec_rules, pos).type_ == Gretype::Alt {
                // there's another alternate def of this rule to process
                pos = pos.advance(1);
            } else {
                break;
            }
        }

        let opaque = opaque_tokens(&vec_rules);

        Some(Self {
            rules: vec_rules,
            stacks,
            partial_utf8: PartialUtf8::default(),
            lazy,
            awaiting_trigger: lazy,
            trigger_buffer: Vec::new(),
            trigger_buffer_positions: Vec::new(),
            trigger_tokens,
            trigger_patterns,
            opaque_tokens: opaque,
        })
    }

    /// Port of the element-array `llama_grammar_init_impl`.
    pub fn from_rules(rules: &[GrammarRule], start_rule_index: usize) -> Option<Self> {
        if start_rule_index >= rules.len() {
            return None;
        }
        Self::build(rules, start_rule_index, false, Vec::new(), Vec::new())
    }

    /// Port of the string `llama_grammar_init_impl`, without a vocab.
    pub fn from_str(
        grammar_str: &str,
        grammar_root: &str,
        lazy: bool,
        trigger_patterns: &[String],
        trigger_tokens: &[Token],
    ) -> Option<Self> {
        Self::from_str_with_tokenizer_opt(
            grammar_str,
            grammar_root,
            lazy,
            trigger_patterns,
            trigger_tokens,
            None,
        )
    }

    /// Same as `from_str`, with a tokenizer for `<token>` rules.
    pub fn from_str_with_tokenizer<F>(
        grammar_str: &str,
        grammar_root: &str,
        lazy: bool,
        trigger_patterns: &[String],
        trigger_tokens: &[Token],
        tokenize: F,
    ) -> Option<Self>
    where
        F: Fn(&[u8]) -> Result<Vec<Token>> + 'static,
    {
        Self::from_str_with_tokenizer_opt(
            grammar_str,
            grammar_root,
            lazy,
            trigger_patterns,
            trigger_tokens,
            Some(Box::new(tokenize)),
        )
    }

    fn from_str_with_tokenizer_opt(
        grammar_str: &str,
        grammar_root: &str,
        lazy: bool,
        trigger_patterns: &[String],
        trigger_tokens: &[Token],
        tokenize: Option<Box<TokenizeFn>>,
    ) -> Option<Self> {
        let mut parser = GrammarParser::with_tokenizer_opt(tokenize);

        // if there is a grammar, parse it
        // rules will be empty (default) if there are parse errors
        if !parser.parse(grammar_str) || parser.rules.is_empty() {
            eprintln!("failed to parse grammar");
            return None;
        }

        // Ensure that the grammar contains the start symbol
        let start_rule_index = match parser.symbol_ids.get(grammar_root) {
            Some(id) => *id as usize,
            None => {
                eprintln!("grammar does not contain a '{}' symbol", grammar_root);
                return None;
            }
        };

        let rules = std::mem::take(&mut parser.rules);
        let patterns = trigger_patterns
            .iter()
            .map(|pattern| TriggerPattern {
                pattern: pattern.clone(),
            })
            .collect();

        Self::build(
            &rules,
            start_rule_index,
            lazy,
            trigger_tokens.to_vec(),
            patterns,
        )
    }

    /// Port of `llama_grammar_get_rules`.
    pub fn rules(&self) -> &GrammarRules {
        &self.rules
    }

    /// Port of `llama_grammar_get_stacks`.
    pub fn stacks(&self) -> &GrammarStacks {
        &self.stacks
    }

    pub fn stacks_mut(&mut self) -> &mut GrammarStacks {
        &mut self.stacks
    }

    /// true if a token rule names the token, char rules do not match it
    pub fn is_opaque(&self, token: Token) -> bool {
        let id = token as u32;
        let it = self.opaque_tokens.partition_point(|&(lo, _)| lo <= id);
        it > 0 && self.opaque_tokens[it - 1].1 >= id
    }

    /// Port of `llama_grammar_apply_impl`.
    pub fn apply(&self, cur: &mut [TokenData]) {
        if self.awaiting_trigger {
            return;
        }

        let mut allow_eog = false;
        for stack in &self.stacks {
            if stack.is_empty() {
                allow_eog = true;
                break;
            }
        }

        // In C++ `candidates_decoded` keeps the pointers in `candidates_grammar` alive.
        let mut candidates = GrammarCandidates::with_capacity(cur.len());
        for (i, data) in cur.iter_mut().enumerate() {
            if data.is_eog {
                if !allow_eog {
                    data.logit = f32::NEG_INFINITY;
                }
            } else if data.piece.is_empty() || data.piece[0] == 0 {
                data.logit = f32::NEG_INFINITY;
            } else {
                let decoded = decode_utf8(&data.piece, self.partial_utf8);
                candidates.push(GrammarCandidate {
                    index: i,
                    code_points: decoded.0,
                    partial_utf8: decoded.1,
                    id: data.id,
                    opaque: self.is_opaque(data.id),
                });
            }
        }

        let rejects = reject_candidates(&self.rules, &self.stacks, &candidates);
        for reject in rejects {
            cur[reject.index].logit = f32::NEG_INFINITY;
        }
    }

    /// Port of `llama_grammar_accept`.
    pub fn accept(&mut self, chr: u32) {
        let mut stacks_new = GrammarStacks::with_capacity(self.stacks.len());
        for stack in &self.stacks {
            accept_chr(&self.rules, stack, chr, &mut stacks_new);
        }
        self.stacks = stacks_new;
    }

    /// Port of `llama_grammar_accept_impl`. `piece` and `is_eog` come from the vocab in C++.
    pub fn accept_impl(&mut self, token: Token, piece: &[u8], is_eog: bool) -> Result<()> {
        if self.awaiting_trigger {
            if self.trigger_tokens.contains(&token) {
                self.awaiting_trigger = false;
                self.trigger_buffer.clear();
                self.accept_token(token, piece)?;
                return Ok(());
            }

            let start = self.trigger_buffer.len();
            let end = start + piece.len();
            self.trigger_buffer_positions.push((token, (start, end)));
            self.trigger_buffer.extend_from_slice(piece);

            let found = {
                let input = String::from_utf8_lossy(&self.trigger_buffer);
                self.trigger_patterns.iter().find_map(|p| p.find(&input))
            };
            if let Some(start) = found {
                self.awaiting_trigger = false;

                // replay tokens that overlap with [start, end)
                let positions = std::mem::take(&mut self.trigger_buffer_positions);
                let buffer = std::mem::take(&mut self.trigger_buffer);
                for (tok, (tok_start, tok_end)) in positions {
                    if tok_end <= start {
                        continue;
                    }
                    let piece_start = tok_start.max(start); // allow for partial token pieces
                    let piece_len = tok_end - piece_start;
                    self.accept_token(tok, &buffer[piece_start..piece_start + piece_len])?;
                }
            }
            return Ok(());
        }

        if is_eog {
            for stack in &self.stacks {
                if stack.is_empty() {
                    return Ok(());
                }
            }
            return Err(Error::new("fatal error: EOG token with no empty stack"));
        }

        self.accept_token(token, piece)
    }

    /// Port of `llama_grammar_accept_str`.
    pub fn accept_str(&mut self, piece: &[u8]) -> Result<()> {
        // Note terminating 0 in decoded string
        let decoded = decode_utf8(piece, self.partial_utf8);
        let code_points = decoded.0;
        for &chr in code_points.iter().take(code_points.len().saturating_sub(1)) {
            self.accept(chr);
        }
        self.partial_utf8 = decoded.1;
        if self.stacks.is_empty() {
            return Err(Error::new(format!(
                "Unexpected empty grammar stack after accepting piece: {}",
                String::from_utf8_lossy(piece)
            )));
        }
        Ok(())
    }

    /// Port of `llama_grammar_accept_token`.
    pub fn accept_token(&mut self, token: Token, piece: &[u8]) -> Result<()> {
        // Note terminating 0 in decoded string
        let decoded = decode_utf8(piece, self.partial_utf8);
        let code_points = decoded.0;

        // the mask rejects a piece that does not continue a partial character, so accepting one has to fail too
        if decoded.1.n_remain < 0 {
            return Err(Error::new(format!(
                "Invalid UTF-8 continuation in piece: {} ({})",
                String::from_utf8_lossy(piece),
                token
            )));
        }

        let opaque = self.is_opaque(token);
        let mut stacks_new = GrammarStacks::with_capacity(self.stacks.len());

        for stack in &self.stacks {
            if stack.is_empty() {
                continue;
            }

            let pos = *stack.last().unwrap();
            if matches!(
                elem(&self.rules, pos).type_,
                Gretype::Token | Gretype::TokenNot
            ) {
                let (matched, after) = match_token(&self.rules, pos, token);
                if matched {
                    let mut new_stack = stack.clone();
                    new_stack.pop();
                    if !is_end_of_sequence(elem(&self.rules, after)) {
                        new_stack.push(after);
                    }
                    advance_stack(&self.rules, &new_stack, &mut stacks_new);
                }
            } else if !opaque {
                let mut current_stacks = vec![stack.clone()];
                for &chr in code_points.iter().take(code_points.len().saturating_sub(1)) {
                    let mut next_stacks = GrammarStacks::new();
                    for cur_stack in &current_stacks {
                        accept_chr(&self.rules, cur_stack, chr, &mut next_stacks);
                    }
                    current_stacks = next_stacks;
                    if current_stacks.is_empty() {
                        break;
                    }
                }
                for surviving_stack in current_stacks {
                    if !stacks_new.contains(&surviving_stack) {
                        stacks_new.push(surviving_stack);
                    }
                }
            }
        }

        self.stacks = stacks_new;
        self.partial_utf8 = decoded.1;
        if self.stacks.is_empty() {
            return Err(Error::new(format!(
                "Unexpected empty grammar stack after accepting piece: {} ({})",
                String::from_utf8_lossy(piece),
                token
            )));
        }
        Ok(())
    }
}

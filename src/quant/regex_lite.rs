//! Tiny regex matcher for the `--tensor-type` patterns.
//!
//! `std::regex_search` is used for two things in the quantizer: fixed
//! `blk.<digits>.` matching (handled directly by `names::parse_block`) and the
//! user supplied `--tensor-type` patterns. This supports the subset needed for
//! tensor names: literals, `.`, `*`, `+`, `?`, `^`, `$`, escapes and simple
//! character classes.

#[derive(Clone, Debug)]
enum Node {
    Lit(char),
    Any,
    Class { negated: bool, chars: Vec<char> },
    Star(Box<Node>),
    Plus(Box<Node>),
    Opt(Box<Node>),
}

fn parse(pattern: &str) -> (bool, bool, Vec<Node>) {
    let chars: Vec<char> = pattern.chars().collect();
    let mut nodes = Vec::new();
    let mut i = 0;
    let mut anchored_start = false;
    let mut anchored_end = false;
    if chars.first() == Some(&'^') {
        anchored_start = true;
        i += 1;
    }
    if chars.last() == Some(&'$') && !chars[..chars.len().saturating_sub(1)].contains(&'\\') {
        anchored_end = true;
    }
    let limit = if anchored_end {
        chars.len() - 1
    } else {
        chars.len()
    };
    while i < limit {
        let atom = match chars[i] {
            '\\' if i + 1 < limit => {
                i += 1;
                Node::Lit(chars[i])
            }
            '.' => Node::Any,
            '[' => {
                let mut j = i + 1;
                let negated = chars.get(j) == Some(&'^');
                if negated {
                    j += 1;
                }
                let mut set = Vec::new();
                while j < limit && chars[j] != ']' {
                    if j + 2 < limit && chars[j + 1] == '-' && chars[j + 2] != ']' {
                        let lo = chars[j] as u32;
                        let hi = chars[j + 2] as u32;
                        for c in lo..=hi {
                            if let Some(ch) = char::from_u32(c) {
                                set.push(ch);
                            }
                        }
                        j += 3;
                    } else {
                        set.push(chars[j]);
                        j += 1;
                    }
                }
                i = j;
                Node::Class {
                    negated,
                    chars: set,
                }
            }
            c => Node::Lit(c),
        };
        i += 1;
        let quant = chars.get(i).copied();
        let node = match quant {
            Some('*') => {
                i += 1;
                Node::Star(Box::new(atom))
            }
            Some('+') => {
                i += 1;
                Node::Plus(Box::new(atom))
            }
            Some('?') => {
                i += 1;
                Node::Opt(Box::new(atom))
            }
            _ => atom,
        };
        nodes.push(node);
    }
    (anchored_start, anchored_end, nodes)
}

fn matches_atom(node: &Node, c: char) -> bool {
    match node {
        Node::Lit(x) => *x == c,
        Node::Any => true,
        Node::Class { negated, chars } => chars.contains(&c) != *negated,
        Node::Star(_) | Node::Plus(_) | Node::Opt(_) => false,
    }
}

fn match_seq(nodes: &[Node], text: &[char], ti: usize) -> bool {
    if nodes.is_empty() {
        return true;
    }
    match &nodes[0] {
        Node::Star(inner) => {
            let mut k = ti;
            loop {
                if match_seq(&nodes[1..], text, k) {
                    return true;
                }
                if k >= text.len() || !matches_atom(inner, text[k]) {
                    return false;
                }
                k += 1;
            }
        }
        Node::Plus(inner) => {
            let mut k = ti;
            if k >= text.len() || !matches_atom(inner, text[k]) {
                return false;
            }
            k += 1;
            loop {
                if match_seq(&nodes[1..], text, k) {
                    return true;
                }
                if k >= text.len() || !matches_atom(inner, text[k]) {
                    return false;
                }
                k += 1;
            }
        }
        Node::Opt(inner) => {
            if match_seq(&nodes[1..], text, ti) {
                return true;
            }
            ti < text.len() && matches_atom(inner, text[ti]) && match_seq(&nodes[1..], text, ti + 1)
        }
        atom => {
            ti < text.len() && matches_atom(atom, text[ti]) && match_seq(&nodes[1..], text, ti + 1)
        }
    }
}

/// Equivalent of `std::regex_search(text, pattern)`.
pub fn regex_search(text: &str, pattern: &str) -> bool {
    let (start, end, nodes) = parse(pattern);
    let chars: Vec<char> = text.chars().collect();
    if start {
        return end
            .then(|| match_seq(&nodes, &chars, 0) && consumed_all(&nodes, &chars, 0))
            .unwrap_or_else(|| match_seq(&nodes, &chars, 0));
    }
    for i in 0..=chars.len() {
        if match_seq(&nodes, &chars, i) && (!end || consumed_all(&nodes, &chars, i)) {
            return true;
        }
    }
    false
}

fn consumed_all(nodes: &[Node], text: &[char], start: usize) -> bool {
    // `match_seq` stops as soon as the pattern is consumed; verify that the
    // match reached the end of the text for `$`-anchored patterns.
    fn helper(nodes: &[Node], text: &[char], ti: usize, out: &mut Vec<usize>) {
        if nodes.is_empty() {
            out.push(ti);
            return;
        }
        match &nodes[0] {
            Node::Star(inner) => {
                let mut k = ti;
                loop {
                    helper(&nodes[1..], text, k, out);
                    if k >= text.len() || !matches_atom(inner, text[k]) {
                        break;
                    }
                    k += 1;
                }
            }
            Node::Plus(inner) => {
                let mut k = ti;
                if k >= text.len() || !matches_atom(inner, text[k]) {
                    return;
                }
                k += 1;
                loop {
                    helper(&nodes[1..], text, k, out);
                    if k >= text.len() || !matches_atom(inner, text[k]) {
                        break;
                    }
                    k += 1;
                }
            }
            Node::Opt(inner) => {
                helper(&nodes[1..], text, ti, out);
                if ti < text.len() && matches_atom(inner, text[ti]) {
                    helper(&nodes[1..], text, ti + 1, out);
                }
            }
            atom => {
                if ti < text.len() && matches_atom(atom, text[ti]) {
                    helper(&nodes[1..], text, ti + 1, out);
                }
            }
        }
    }
    let mut ends = Vec::new();
    helper(nodes, text, start, &mut ends);
    ends.iter().any(|&e| e == text.len())
}

/// Equivalent of `std::regex_match(text, pattern)`.
pub fn regex_match(text: &str, pattern: &str) -> bool {
    let (_, _, nodes) = parse(pattern);
    let chars: Vec<char> = text.chars().collect();
    consumed_all(&nodes, &chars, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_tensor_names() {
        assert!(regex_search("blk.12.attn_v.weight", "attn_v.weight"));
        assert!(regex_match(
            "blk.12.ffn_down.weight",
            r"blk\.[0-9]+\.ffn_down\.weight"
        ));
        assert!(!regex_match(
            "blk.12.attn_v.weight",
            r"blk\.[0-9]+\.ffn_down\.weight"
        ));
        assert!(regex_search("token_embd.weight", "^token_embd.weight$"));
    }
}

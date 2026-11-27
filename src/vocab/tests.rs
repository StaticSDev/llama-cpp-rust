//! Unit tests for pure helpers.

use super::*;
use crate::core::{Error, Result, Token, LLAMA_TOKEN_NULL};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

    #[test]
    fn gpt2_byte_map_roundtrips() {
        for b in 0u8..=255 {
            let piece = unicode_byte_to_utf8(b);
            assert_eq!(unicode_utf8_to_byte(&piece), b);
            let cpt = piece.chars().next().unwrap() as u32;
            assert_eq!(unicode_cpt_to_byte(cpt).unwrap(), b);
        }
    }

    #[test]
    fn trie_longest_prefix() {
        let mut trie = NaiveTrie::default();
        trie.insert(b"abcd", 7);
        trie.insert(b"ab", 3);
        assert_eq!(trie.get_longest_prefix(b"abxyz", 0), 2);
        assert_eq!(trie.get_longest_prefix(b"abcdef", 0), 4);
        assert_eq!(trie.get_longest_prefix(b"zzzz", 0), 0);
        assert_eq!(trie.traverse(b'a').unwrap().traverse(b'b').unwrap().value, 3);
        assert_eq!(parse_hex_byte("0A"), 10);
    }

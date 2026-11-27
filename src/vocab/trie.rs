//! Longest-prefix trie used by the tokenizers.

use super::*;
use crate::core::{Error, Result, Token, LLAMA_TOKEN_NULL};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

// no recursion: a token from the GGUF can be long
#[derive(Clone, Debug, Default)]
pub(crate) struct NaiveTrie {
    pub(crate) children: BTreeMap<u8, NaiveTrie>,
    pub(crate) has_value: bool,
    pub(crate) value: Token,
}

impl NaiveTrie {
    pub(crate) fn insert(&mut self, key: &[u8], value: Token) {
        let mut node = self;
        for &byte in key {
            node = node.children.entry(byte).or_default();
        }
        node.has_value = true;
        node.value = value;
    }

    pub(crate) fn get_longest_prefix(&self, key: &[u8], mut offset: usize) -> usize {
        let mut node = self;
        while offset < key.len() {
            match node.children.get(&key[offset]) {
                Some(child) => {
                    node = child;
                    offset += 1;
                }
                None => break,
            }
        }
        offset
    }

    pub(crate) fn traverse(&self, c: u8) -> Option<&NaiveTrie> {
        self.children.get(&c)
    }
}

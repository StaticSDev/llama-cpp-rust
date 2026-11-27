//! Rust port of `common/ngram-cache`, `common/ngram-map` and `common/ngram-mod`.
//!
//! Pure-data helpers used by speculative decoding. No model, context or
//! backend handle is needed, so standard-library collections cover everything.

use crate::core::{Error, Result, Token, LLAMA_TOKEN_NULL};
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::time::Instant;

// ---------------------------------------------------------------------------
// ngram cache
// ---------------------------------------------------------------------------

pub const LLAMA_NGRAM_MIN: usize = 1;
pub const LLAMA_NGRAM_MAX: usize = 4;
pub const LLAMA_NGRAM_STATIC: usize = 2;

/// `common_ngram`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ngram {
    pub tokens: [Token; LLAMA_NGRAM_MAX],
}

impl Ngram {
    /// All tokens set to `LLAMA_TOKEN_NULL`.
    pub fn new() -> Self {
        Self {
            tokens: [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX],
        }
    }

    /// First `ngram_size` tokens of `input`, rest set to `LLAMA_TOKEN_NULL`.
    pub fn from_tokens(input: &[Token], ngram_size: usize) -> Self {
        let mut tokens = [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX];
        for (i, slot) in tokens.iter_mut().enumerate() {
            if i < ngram_size {
                *slot = input.get(i).copied().unwrap_or(LLAMA_TOKEN_NULL);
            }
        }
        Self { tokens }
    }
}

impl Default for Ngram {
    fn default() -> Self {
        Self::new()
    }
}

/// Fibonacci hashing, matching `common_token_hash_function`.
fn token_hash(token: Token) -> u64 {
    (token as u64).wrapping_mul(11400714819323198485u64)
}

impl Hash for Ngram {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut hash = token_hash(self.tokens[0]);
        for i in 1..LLAMA_NGRAM_MAX {
            hash ^= token_hash(self.tokens[i]);
        }
        state.write_u64(hash);
    }
}

/// token -> number of times token has been seen
pub type NgramCachePart = HashMap<Token, i32>;
/// ngram -> empirical distribution of following tokens
pub type NgramCache = HashMap<Ngram, NgramCachePart>;

/// Update an ngram cache with tokens. `inp` may only be appended to.
pub fn common_ngram_cache_update(
    ngram_cache: &mut NgramCache,
    ngram_min: i32,
    ngram_max: i32,
    inp: &[Token],
    nnew: i32,
    print_progress: bool,
) {
    let t_start = Instant::now();
    let inp_size = inp.len() as i64;
    let n_todo = inp_size * (ngram_max as i64 - ngram_min as i64 + 1);
    let mut n_done = 0i64;

    for ngram_size in ngram_min..=ngram_max {
        let size = ngram_size as i64;
        let i_start = (inp_size - nnew as i64).max(size);
        for i in i_start..inp_size {
            let ngram = Ngram::from_tokens(&inp[(i - size) as usize..], size as usize);
            let token = inp[i as usize];

            ngram_cache
                .entry(ngram)
                .or_default()
                .entry(token)
                .and_modify(|count| *count += 1)
                .or_insert(1);
            n_done += 1;

            if print_progress && n_done % 10_000_000 == 0 {
                let t_now_ms = t_start.elapsed().as_millis() as i64;
                let eta_ms = (n_todo - n_done) * t_now_ms / n_done;
                let eta_min = eta_ms / (60 * 1000);
                let eta_s = (eta_ms - 60 * 1000 * eta_min) / 1000;
                eprintln!(
                    "common_ngram_cache_update: {}/{} done, ETA: {:02}:{:02}",
                    n_done, n_todo, eta_min, eta_s
                );
            }
        }
    }
}

/// Token from the combined, speculative sequence of `inp` and `draft`.
fn get_token(inp: &[Token], draft: &[Token], i: usize) -> Token {
    if i < inp.len() {
        inp[i]
    } else {
        draft[1 + i - inp.len()]
    }
}

const DRAFT_MIN_SAMPLE_SIZE_LAX: [i32; LLAMA_NGRAM_MAX] = [2, 2, 1, 1];
const DRAFT_MIN_PERCENT_LAX: [i32; LLAMA_NGRAM_MAX] = [66, 50, 50, 50];
const DRAFT_MIN_SAMPLE_SIZE_STRICT: [i32; LLAMA_NGRAM_MAX] = [4, 3, 2, 2];
const DRAFT_MIN_PERCENT_STRICT: [i32; LLAMA_NGRAM_MAX] = [75, 66, 66, 66];

/// Try to draft a token from only the static cache.
fn try_draft_static(nc_static: &NgramCache, ngram_static: Ngram) -> Token {
    let Some(part_static) = nc_static.get(&ngram_static) else {
        return LLAMA_TOKEN_NULL;
    };

    let mut max_count_static = 0i32;
    let mut sum_count_static = 0i32;
    let mut max_token = LLAMA_TOKEN_NULL;

    for (&token, &count_static) in part_static {
        if count_static > max_count_static {
            max_token = token;
            max_count_static = count_static;
        }
        sum_count_static += count_static;
    }

    if sum_count_static < DRAFT_MIN_SAMPLE_SIZE_LAX[LLAMA_NGRAM_STATIC - 1] {
        return LLAMA_TOKEN_NULL;
    }
    if 100 * max_count_static < DRAFT_MIN_PERCENT_LAX[LLAMA_NGRAM_STATIC - 1] * sum_count_static {
        return LLAMA_TOKEN_NULL;
    }
    max_token
}

/// Try to draft from a primary cache, validated with the static distribution.
fn try_draft_primary(
    nc_primary: &NgramCache,
    ngrams_primary: &[Ngram],
    part_static: &NgramCachePart,
    min_sample_size: &[i32],
    min_percent: &[i32],
) -> Token {
    let mut drafted_token = LLAMA_TOKEN_NULL;

    for i in (0..ngrams_primary.len()).rev() {
        if drafted_token != LLAMA_TOKEN_NULL {
            break;
        }
        let Some(part_primary) = nc_primary.get(&ngrams_primary[i]) else {
            continue;
        };

        let mut max_count_primary = 0i32;
        let mut max_count_static = 0i32;
        let mut sum_count_primary = 0i32;
        let mut max_token = LLAMA_TOKEN_NULL;

        for (&token, &count_primary) in part_primary {
            let count_static = part_static.get(&token).map_or(1, |count| 100 * count);
            if count_primary * count_static > max_count_primary * max_count_static {
                max_token = token;
                max_count_primary = count_primary;
                max_count_static = count_static;
            }
            sum_count_primary += count_primary;
        }

        if sum_count_primary < min_sample_size[i] {
            continue;
        }
        if 100 * max_count_primary < min_percent[i] * sum_count_primary {
            continue;
        }
        drafted_token = max_token;
    }

    drafted_token
}

/// Draft tokens from the context, dynamic and static ngram caches.
/// `draft` must initially contain the previously sampled token.
pub fn common_ngram_cache_draft(
    inp: &[Token],
    draft: &mut Vec<Token>,
    n_draft: i32,
    ngram_min: i32,
    ngram_max: i32,
    nc_context: &NgramCache,
    nc_dynamic: &NgramCache,
    nc_static: &NgramCache,
) {
    assert_eq!(draft.len(), 1, "draft must initially contain one token");
    let inp_size = inp.len() as i32;
    if inp_size < LLAMA_NGRAM_STATIC as i32 {
        return;
    }

    let empty_part = NgramCachePart::new();
    while (draft.len() as i32) - 1 < n_draft {
        let mut drafted_token = LLAMA_TOKEN_NULL;

        let ngram_start_static =
            inp_size - LLAMA_NGRAM_STATIC as i32 + (draft.len() as i32 - 1);
        let mut ngram_static = Ngram::default();
        for j in ngram_start_static..ngram_start_static + LLAMA_NGRAM_STATIC as i32 {
            ngram_static.tokens[(j - ngram_start_static) as usize] =
                get_token(inp, draft, j as usize);
        }
        let part_static = nc_static.get(&ngram_static).unwrap_or(&empty_part);

        let mut ngrams_cd = Vec::new();
        for ngram_size_cd in ngram_min..=ngram_max {
            let ngram_start_cd = inp_size - ngram_size_cd + (draft.len() as i32 - 1);
            let mut ngram_cd = Ngram::default();
            for j in ngram_start_cd..ngram_start_cd + ngram_size_cd {
                ngram_cd.tokens[(j - ngram_start_cd) as usize] =
                    get_token(inp, draft, j as usize);
            }
            ngrams_cd.push(ngram_cd);
        }

        if drafted_token == LLAMA_TOKEN_NULL {
            drafted_token = try_draft_primary(
                nc_context,
                &ngrams_cd,
                part_static,
                &DRAFT_MIN_SAMPLE_SIZE_LAX,
                &DRAFT_MIN_PERCENT_LAX,
            );
        }
        if drafted_token == LLAMA_TOKEN_NULL {
            drafted_token = try_draft_primary(
                nc_dynamic,
                &ngrams_cd,
                part_static,
                &DRAFT_MIN_SAMPLE_SIZE_STRICT,
                &DRAFT_MIN_PERCENT_STRICT,
            );
        }
        if drafted_token == LLAMA_TOKEN_NULL {
            drafted_token = try_draft_static(nc_static, ngram_static);
        }

        if drafted_token == LLAMA_TOKEN_NULL {
            break;
        }
        crate::llama_log_debug!(" - draft candidate: token={}", drafted_token);
        draft.push(drafted_token);
    }
}

/// Save an ngram cache. Records use native integer byte order, as in C++.
pub fn common_ngram_cache_save(ngram_cache: &NgramCache, filename: &str) -> Result<()> {
    let file = File::create(filename)
        .map_err(|e| Error::new(format!("failed to open ngram cache for writing: {}", e)))?;
    let mut out = BufWriter::new(file);
    let write_err = |e: std::io::Error| Error::new(format!("failed to write ngram cache: {}", e));

    for (ngram, token_counts) in ngram_cache {
        let ntokens = token_counts.len() as i32;
        for token in ngram.tokens {
            out.write_all(&token.to_ne_bytes()).map_err(write_err)?;
        }
        out.write_all(&ntokens.to_ne_bytes()).map_err(write_err)?;
        for (&token, &count) in token_counts {
            out.write_all(&token.to_ne_bytes()).map_err(write_err)?;
            out.write_all(&count.to_ne_bytes()).map_err(write_err)?;
        }
    }
    out.flush().map_err(write_err)?;
    Ok(())
}

/// Fill `buf`; return `false` on clean EOF, `Err` on a partial record.
fn read_full<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => {
                if filled == 0 {
                    return Ok(false);
                }
                return Err(Error::new("truncated ngram cache file"));
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::new(format!("failed to read ngram cache: {}", e))),
        }
    }
    Ok(true)
}

fn read_cache_i32<R: Read>(reader: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    if !read_full(reader, &mut buf)? {
        return Err(Error::new("truncated ngram cache file"));
    }
    Ok(i32::from_ne_bytes(buf))
}

/// Load an ngram cache written by `common_ngram_cache_save`.
pub fn common_ngram_cache_load(filename: &str) -> Result<NgramCache> {
    let file = File::open(filename)
        .map_err(|e| Error::new(format!("Unable to open file {}: {}", filename, e)))?;
    let mut reader = BufReader::new(file);
    let mut ngram_cache = NgramCache::new();

    loop {
        let mut buf = [0u8; 4 * LLAMA_NGRAM_MAX];
        if !read_full(&mut reader, &mut buf)? {
            break;
        }
        let mut tokens = [0i32; LLAMA_NGRAM_MAX];
        for (i, token) in tokens.iter_mut().enumerate() {
            *token = i32::from_ne_bytes([
                buf[4 * i],
                buf[4 * i + 1],
                buf[4 * i + 2],
                buf[4 * i + 3],
            ]);
        }

        let ntokens = read_cache_i32(&mut reader)?;
        if ntokens <= 0 {
            return Err(Error::new("invalid ngram cache file: ntokens <= 0"));
        }
        let mut token_counts = NgramCachePart::new();
        for _ in 0..ntokens {
            let token = read_cache_i32(&mut reader)?;
            let count = read_cache_i32(&mut reader)?;
            if count <= 0 {
                return Err(Error::new("invalid ngram cache file: count <= 0"));
            }
            token_counts.insert(token, count);
        }
        ngram_cache.insert(Ngram { tokens }, token_counts);
    }

    Ok(ngram_cache)
}

/// Add all information from `add` to `target`.
pub fn common_ngram_cache_merge(target: &mut NgramCache, add: &NgramCache) {
    for (ngram, part) in add {
        let merged = target.entry(*ngram).or_default();
        for (&token, &count) in part {
            *merged.entry(token).or_insert(0) += count;
        }
    }
}

// ---------------------------------------------------------------------------
// ngram simple
// ---------------------------------------------------------------------------

/// `common_ngram_simple_config`.
#[derive(Clone, Copy, Debug)]
pub struct NgramSimpleConfig {
    /// Size of n-grams to look up in token history.
    pub size_ngram: u16,
    /// Size of m-grams to draft after a match.
    pub size_mgram: u16,
}

/// Self-speculative draft from a matching pattern in the token history.
pub fn common_ngram_simple_draft(
    config: &NgramSimpleConfig,
    tokens: &[Token],
    sampled: Token,
) -> Vec<Token> {
    let cur_len = tokens.len();
    let n_draft_min = config.size_ngram as usize;
    let n_draft_max = config.size_mgram as usize;

    let mut draft_tokens = Vec::new();
    if cur_len <= n_draft_min + n_draft_max + 1 {
        return draft_tokens;
    }

    let mut pattern = Vec::with_capacity(n_draft_min);
    for j in cur_len - n_draft_min + 1..cur_len {
        pattern.push(tokens[j]);
    }
    pattern.push(sampled);

    let mut match_pos = 0usize;
    for j in (1..=cur_len - n_draft_min - 1).rev() {
        if key_matches(tokens, j, &pattern, n_draft_min) {
            match_pos = j;
            break;
        }
    }
    if match_pos == 0 {
        return draft_tokens;
    }

    let copy_max = n_draft_max.min(cur_len - (match_pos + n_draft_min));
    if copy_max < n_draft_min {
        return draft_tokens;
    }
    crate::llama_log_debug!(
        "common_ngram_simple_draft: #tokens = {}: found matching pattern at pos {}, length {}, draft length {}",
        cur_len, match_pos, pattern.len(), copy_max
    );

    draft_tokens
        .extend_from_slice(&tokens[match_pos + n_draft_min..match_pos + n_draft_min + copy_max]);
    draft_tokens
}

// ---------------------------------------------------------------------------
// ngram map
// ---------------------------------------------------------------------------

/// Maximum number of m-gram values stored per key n-gram.
pub const COMMON_NGRAM_MAX_VALUES: usize = 4;
/// Number of entries in the optional hash-to-index map.
pub const COMMON_NGRAM_HASH_MAP_SIZE: usize = 262144;
/// Maximum number of counted values of a ngram map value.
pub const COMMON_NGRAM_MAX_VALUE_COUNT: i32 = 16380;

const LCG_FACTOR: u32 = 2654435761;

/// Statistics of an m-gram after a known n-gram.
#[derive(Clone, Copy, Debug)]
pub struct NgramMapValue {
    /// Index of value m-gram in token history (0 if unused).
    pub value_idx: usize,
    /// Number of occurrences (0 in an unused slot).
    pub value_num: u16,
    /// Number of accepted tokens at last draft (-1 if unused).
    pub n_accepted: i16,
}

impl Default for NgramMapValue {
    fn default() -> Self {
        Self {
            value_idx: 0,
            value_num: 0,
            n_accepted: -1,
        }
    }
}

/// Statistics of a key n-gram.
#[derive(Clone, Debug)]
pub struct NgramMapKey {
    pub key_idx: usize,
    pub stat_idx: usize,
    pub key_num: u16,
    pub values: [NgramMapValue; COMMON_NGRAM_MAX_VALUES],
}

/// Map from n-grams to following m-grams in token history.
pub struct NgramMap {
    pub size_key: u16,
    pub size_value: u16,
    pub key_only: bool,
    pub keys: Vec<NgramMapKey>,
    pub min_hits: u16,
    pub show_key_map_stats: bool,
    pub size_last_begin: usize,
    pub last_draft_created: bool,
    pub last_draft_key_idx: usize,
    pub last_draft_value_idx: u16,
    pub idx_last_check: usize,
    pub key_map: Vec<u32>,
    pub key_map_last_idx: u32,
}

impl NgramMap {
    pub fn new(size_key: u16, size_value: u16, key_only: bool, min_hits: u16) -> Self {
        Self {
            size_key,
            size_value,
            key_only,
            keys: Vec::new(),
            min_hits,
            show_key_map_stats: false,
            size_last_begin: 0,
            last_draft_created: false,
            last_draft_key_idx: 0,
            last_draft_value_idx: 0,
            idx_last_check: 0,
            key_map: vec![0; COMMON_NGRAM_HASH_MAP_SIZE],
            key_map_last_idx: 0,
        }
    }
}

/// LCG hash of the n-gram of size `len` at `start`.
fn common_ngram_map_hash(tokens: &[Token], start: usize, len: usize) -> u32 {
    let mut hash = 0u32;
    for i in 0..len {
        hash = hash
            .wrapping_mul(LCG_FACTOR)
            .wrapping_add(tokens[start + i] as u32);
    }
    hash
}

/// Tokens in the form `[v0, v1, v2, ...]`.
fn common_tokens_to_str(inp: &[Token], start: usize, length: usize) -> String {
    let mut out = String::from("[");
    for i in 0..length {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&inp[start + i].to_string());
    }
    out.push(']');
    out
}

/// True when the n-gram at `pos` equals `key[..n]`.
fn key_matches(tokens: &[Token], pos: usize, key: &[Token], n: usize) -> bool {
    for k in 0..n {
        if tokens[pos + k] != key[k] {
            return false;
        }
    }
    true
}

/// Reset map entries from a context that was removed or reordered.
pub fn common_ngram_map_begin(map: &mut NgramMap, tokens: &[Token]) {
    let size_begin = tokens.len();
    crate::llama_log_debug!(
        "common_ngram_map_begin: begin, idx_last_check={}, new begin={}, #keys={}",
        map.idx_last_check,
        size_begin,
        map.keys.len()
    );

    let mut idx_begin_cleanup = map.size_last_begin;
    if idx_begin_cleanup > size_begin {
        idx_begin_cleanup = if size_begin > map.size_key as usize + map.size_value as usize {
            size_begin - map.size_key as usize - map.size_value as usize
        } else {
            0
        };
        crate::llama_log_info!(
            "common_ngram_map_begin: shrink cleanup begin: {} -> {}",
            map.size_last_begin,
            idx_begin_cleanup
        );
    }

    let mut count_map_entries_upd = 0usize;
    if !map.key_map.is_empty() && size_begin < map.idx_last_check {
        if map.show_key_map_stats {
            let mut count_nonzero = 0usize;
            let mut min_idx = u32::MAX;
            let mut max_idx = 0u32;
            for &key_idx in &map.key_map {
                if key_idx != 0 {
                    count_nonzero += 1;
                    if key_idx < min_idx {
                        min_idx = key_idx;
                    }
                    if key_idx > max_idx {
                        max_idx = key_idx;
                    }
                }
            }
            if count_nonzero == 0 {
                min_idx = 0;
            }
            crate::llama_log_info!(
                "common_ngram_map_begin: key_map stats: entries={}, min_idx={}, max_idx={}, key_map_last_idx={}",
                count_nonzero, min_idx, max_idx, map.key_map_last_idx
            );
        }

        for key_idx in &mut map.key_map {
            if *key_idx != 0 && *key_idx as usize >= idx_begin_cleanup {
                *key_idx = 0;
                count_map_entries_upd += 1;
            }
        }
        map.key_map_last_idx = if idx_begin_cleanup > 0 {
            (idx_begin_cleanup - 1) as u32
        } else {
            0
        };
    }

    let mut count_keys_del = 0usize;
    let mut count_values_del = 0usize;
    let key_only = map.key_only;
    if size_begin < map.idx_last_check && !map.keys.is_empty() {
        let count_keys = map.keys.len();
        for i in (0..map.keys.len()).rev() {
            let remove_key = {
                let key = &mut map.keys[i];
                if key.key_idx >= idx_begin_cleanup {
                    crate::llama_log_debug!(
                        "common_ngram_map_begin: delete key at index {} (key_idx {} >= idx_begin_cleanup {})",
                        i, key.key_idx, idx_begin_cleanup
                    );
                    true
                } else if key_only {
                    false
                } else {
                    for j in (0..COMMON_NGRAM_MAX_VALUES).rev() {
                        let delete = key.values[j].value_idx != 0
                            && key.values[j].value_idx >= idx_begin_cleanup;
                        if delete {
                            count_values_del += 1;
                            for k in j..COMMON_NGRAM_MAX_VALUES - 1 {
                                key.values[k] = key.values[k + 1];
                            }
                            key.values[COMMON_NGRAM_MAX_VALUES - 1].value_idx = 0;
                            key.values[COMMON_NGRAM_MAX_VALUES - 1].value_num = 0;
                        }
                    }
                    if key.values[0].value_idx == 0 {
                        crate::llama_log_debug!(
                            "common_ngram_map_begin: delete key at index {} (no values left)",
                            i
                        );
                        true
                    } else {
                        false
                    }
                }
            };
            if remove_key {
                map.keys.remove(i);
                count_keys_del += 1;
            }
        }

        crate::llama_log_info!(
            "common_ngram_map_begin: refresh map: idx_last_check={}, new begin={}, #keys_checked={}, #keys_del={}, #values_del={}, #hashes_upd={}",
            map.idx_last_check, size_begin, count_keys, count_keys_del, count_values_del,
            count_map_entries_upd
        );
    }

    map.idx_last_check = size_begin;
    map.size_last_begin = size_begin;
}

/// Search for the key n-gram and append a draft to `draft` when found.
pub fn common_ngram_map_draft(
    map: &mut NgramMap,
    inp: &[Token],
    sampled: Token,
    draft: &mut Vec<Token>,
) {
    map.last_draft_created = false;
    map.last_draft_key_idx = 0;
    map.last_draft_value_idx = 0;

    let cur_len = inp.len();
    let n = map.size_key as usize;
    let m = map.size_value as usize;
    if cur_len < 2 * n + m {
        return;
    }
    assert!(cur_len < u32::MAX as usize, "cur_len exceeds UINT32_MAX");
    assert!(map.idx_last_check <= cur_len, "map.idx_last_check > cur_len");
    map.idx_last_check = cur_len;
    assert!(map.size_last_begin <= cur_len, "map.size_last_begin > cur_len");

    let mut key_tokens = Vec::with_capacity(n);
    for j in cur_len - n + 1..cur_len {
        key_tokens.push(inp[j]);
    }
    key_tokens.push(sampled);

    let mut match_pos = 0usize;
    if !map.key_map.is_empty() {
        let idx_hash =
            (common_ngram_map_hash(&key_tokens, 0, n) % map.key_map.len() as u32) as usize;
        let idx_key = map.key_map[idx_hash];
        let limit = cur_len.wrapping_sub(n).wrapping_sub(m).wrapping_sub(1);
        if idx_key != 0 && (idx_key as usize) < limit {
            let matches = key_matches(inp, idx_key as usize, &key_tokens, n);
            crate::llama_log_debug!(
                "common_ngram_map_draft: key hash {:x} -> idx_key {}: match {}",
                idx_hash,
                idx_key,
                if matches { 1 } else { 0 }
            );
            if matches {
                match_pos = idx_key as usize;
            }
        }
    }

    if match_pos == 0 && map.size_last_begin > n + m + 1 {
        let lower = map.key_map_last_idx as usize + 1;
        let upper = map.size_last_begin - n - m - 1;
        for j in (lower..=upper).rev() {
            if key_matches(inp, j, &key_tokens, n) {
                match_pos = j;
                break;
            }
        }
    }
    if match_pos == 0 {
        let lower = map.size_last_begin.max(map.key_map_last_idx as usize) + 1;
        let upper = cur_len - n - m - 1;
        for j in (lower..=upper).rev() {
            if key_matches(inp, j, &key_tokens, n) {
                match_pos = j;
                break;
            }
        }
    }
    if match_pos > 0 {
        crate::llama_log_debug!(
            "common_ngram_map_draft: cur_len = {}, n = {}, m = {}, sz_tkns = {}, sampled = {}, match_pos = {}",
            cur_len, n, m, key_tokens.len(), sampled, match_pos
        );
    }

    if !map.key_map.is_empty() {
        if map.size_last_begin > n + m + 1 {
            let lower = map.key_map_last_idx as usize + 1;
            let upper = map.size_last_begin - n - m - 1;
            for j in (lower..=upper).rev() {
                let idx_hash =
                    (common_ngram_map_hash(inp, j, n) % map.key_map.len() as u32) as usize;
                if map.key_map[idx_hash] == 0 {
                    map.key_map[idx_hash] = j as u32;
                }
            }
        }
        let lower = map.size_last_begin.max(map.key_map_last_idx as usize) + 1;
        let upper = cur_len - n - m - 1;
        for j in (lower..=upper).rev() {
            let idx_hash =
                (common_ngram_map_hash(inp, j, n) % map.key_map.len() as u32) as usize;
            if map.key_map[idx_hash] == 0 {
                map.key_map[idx_hash] = j as u32;
            }
        }
        map.key_map_last_idx = (cur_len - n - m - 1).max(map.key_map_last_idx as usize) as u32;
    }

    if match_pos == 0 {
        return;
    }

    let key_only = map.key_only;
    let min_hits = map.min_hits;

    let mut key_offset = map.keys.len();
    for (i, key) in map.keys.iter().enumerate() {
        if key_matches(inp, key.key_idx, &key_tokens, n) {
            key_offset = i;
            break;
        }
    }
    if key_offset == map.keys.len() {
        let mut values = [NgramMapValue::default(); COMMON_NGRAM_MAX_VALUES];
        for value in &mut values {
            value.value_num = 0;
            value.n_accepted = m as i16;
        }
        map.keys.push(NgramMapKey {
            key_idx: match_pos,
            stat_idx: 0,
            key_num: 0,
            values,
        });
    }

    map.keys[key_offset].key_num =
        (map.keys[key_offset].key_num as i32 + 1).min(COMMON_NGRAM_MAX_VALUE_COUNT) as u16;

    if key_only {
        let n_draft_tokens = {
            let curr_key = &map.keys[key_offset];
            let ndt = (m as i32)
                .min(curr_key.values[0].n_accepted as i32)
                .max(0) as usize;
            crate::llama_log_debug!(
                "common_ngram_map_draft: key_idx = {}, key_offset = {}, key_num = {}, draft.size = {}",
                curr_key.key_idx,
                key_offset,
                curr_key.key_num,
                draft.len() + ndt
            );
            ndt
        };
        for i in 0..n_draft_tokens {
            draft.push(inp[match_pos + n + i]);
        }
        map.last_draft_created = true;
        map.last_draft_key_idx = key_offset;
        map.last_draft_value_idx = 0;
        return;
    }

    if map.keys[key_offset].key_num < min_hits {
        crate::llama_log_debug!(
            "common_ngram_map_draft: key_offset = {}, key_num = {}, min_hits = {}, no draft",
            key_offset,
            map.keys[key_offset].key_num,
            min_hits
        );
        return;
    }

    let (n_draft_tokens, slot_max) = {
        let curr_key = &mut map.keys[key_offset];

        for i in curr_key.stat_idx..=match_pos {
            if !key_matches(inp, i, &key_tokens, n) {
                continue;
            }
            let idx_begin_value_key = i + n;
            let mut idx_value: i32 = -1;
            for v in 0..COMMON_NGRAM_MAX_VALUES {
                let idx_begin_value_v = curr_key.values[v].value_idx;
                if idx_begin_value_v == 0 {
                    curr_key.values[v].value_idx = idx_begin_value_key;
                    curr_key.values[v].value_num = 0;
                    curr_key.values[v].n_accepted = m as i16;
                    idx_value = v as i32;
                    break;
                }
                let mut same = true;
                for j in 0..m {
                    if inp[idx_begin_value_key + j] != inp[idx_begin_value_v + j] {
                        same = false;
                        break;
                    }
                }
                if same {
                    idx_value = v as i32;
                    break;
                }
            }
            if idx_value >= 0 {
                let value = &mut curr_key.values[idx_value as usize];
                value.value_num =
                    (value.value_num as i32 + 1).min(COMMON_NGRAM_MAX_VALUE_COUNT) as u16;
            }
        }
        curr_key.stat_idx = match_pos;

        let mut max_occur = 0u16;
        let mut slot_max = 0usize;
        for v in 0..COMMON_NGRAM_MAX_VALUES {
            if curr_key.values[v].value_num > max_occur {
                max_occur = curr_key.values[v].value_num;
                slot_max = v;
            }
        }
        let mut sum_occur = 0u32;
        for v in 0..COMMON_NGRAM_MAX_VALUES {
            if v != slot_max {
                sum_occur += curr_key.values[v].value_num as u32;
            }
        }

        crate::llama_log_debug!(
            "common_ngram_map_draft: key_offset = {}, max_occur = {}, sum_occur = {}, slot_max = {} [{}/{}, {}/{}, {}/{}, {}/{}]",
            key_offset,
            max_occur,
            sum_occur,
            slot_max,
            curr_key.values[0].value_idx, curr_key.values[0].value_num,
            curr_key.values[1].value_idx, curr_key.values[1].value_num,
            curr_key.values[2].value_idx, curr_key.values[2].value_num,
            curr_key.values[3].value_idx, curr_key.values[3].value_num
        );
        for v in 0..COMMON_NGRAM_MAX_VALUES {
            if curr_key.values[v].value_idx != 0 {
                crate::llama_log_debug!(
                    "common_ngram_map_draft: value[{}] = {}",
                    v,
                    common_tokens_to_str(inp, curr_key.values[v].value_idx, m)
                );
            }
        }

        if sum_occur > 0 && (max_occur as u32) < 2 * sum_occur {
            return;
        }
        let ndt = (m as i32)
            .min(curr_key.values[slot_max].n_accepted as i32)
            .max(0) as usize;
        (ndt, slot_max)
    };

    for i in 0..n_draft_tokens {
        draft.push(inp[match_pos + n + i]);
    }
    crate::llama_log_debug!(
        "common_ngram_map_draft: key_offset = {}, slot_max = {}, key_num = {}, draft.size = {}",
        key_offset,
        slot_max,
        map.keys[key_offset].key_num,
        draft.len()
    );
    map.last_draft_created = true;
    map.last_draft_key_idx = key_offset;
    map.last_draft_value_idx = slot_max as u16;
}

/// Update the chosen value after a draft was processed.
pub fn common_ngram_map_accept(map: &mut NgramMap, n_accepted: u16) {
    if !map.last_draft_created {
        return;
    }
    let curr_value =
        &mut map.keys[map.last_draft_key_idx].values[map.last_draft_value_idx as usize];
    crate::llama_log_debug!(
        "common_ngram_map_accept: n_accepted = {}, prev n_accepted = {}",
        n_accepted,
        curr_value.n_accepted
    );
    curr_value.n_accepted = n_accepted as i16;
}

// ---------------------------------------------------------------------------
// ngram mod
// ---------------------------------------------------------------------------

/// Basic n-gram hasher (`common_ngram_mod`).
pub struct NgramMod {
    n: usize,
    used: usize,
    entries: Vec<i32>,
}

impl NgramMod {
    /// Empty marker for an unused entry.
    pub const EMPTY: i32 = -1;

    pub fn new(n: u16, size: usize) -> Self {
        let mut this = Self {
            n: n as usize,
            used: 0,
            entries: vec![Self::EMPTY; size],
        };
        this.reset();
        this
    }

    /// Index in `entries` for the n-gram `tokens[0..n]`.
    pub fn idx(&self, tokens: &[i32]) -> usize {
        let mut res = 0u64;
        for i in 0..self.n {
            res = res
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(tokens[i] as u64);
        }
        (res % self.entries.len() as u64) as usize
    }

    /// Store `tokens[n]` under the n-gram `tokens[0..n]`.
    pub fn add(&mut self, tokens: &[i32]) {
        let i = self.idx(tokens);
        if self.entries[i] == Self::EMPTY {
            self.used += 1;
        }
        self.entries[i] = tokens[self.n];
    }

    /// Value for the n-gram, or `EMPTY` if not found.
    pub fn get(&self, tokens: &[i32]) -> i32 {
        self.entries[self.idx(tokens)]
    }

    pub fn reset(&mut self) {
        self.entries.fill(Self::EMPTY);
        self.used = 0;
    }

    pub fn get_n(&self) -> usize {
        self.n
    }

    pub fn get_used(&self) -> usize {
        self.used
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }

    pub fn size_bytes(&self) -> usize {
        self.entries.len() * std::mem::size_of::<i32>()
    }
}

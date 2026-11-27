//! NFD/NFC normalization and canonical combining composition.

use super::unicode_data::{
    UNICODE_COMBINING_CLASS, UNICODE_COMPOSITION, UNICODE_DECOMPOSITION, UNICODE_RANGES_NFD,
};

/// NFD normalization: replace each codepoint by the first range that covers it.
pub fn unicode_cpts_normalize_nfd(cpts: &[u32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(cpts.len());
    for &cpt in cpts {
        let idx = UNICODE_RANGES_NFD.partition_point(|range| range.first <= cpt);
        if idx > 0 {
            let range = &UNICODE_RANGES_NFD[idx - 1];
            if range.first <= cpt && cpt <= range.last {
                result.push(range.nfd);
                continue;
            }
        }
        result.push(cpt);
    }
    result
}

fn unicode_canonical_class(cpt: u32) -> u8 {
    UNICODE_COMBINING_CLASS
        .binary_search_by_key(&cpt, |&(key, _)| key)
        .map(|idx| UNICODE_COMBINING_CLASS[idx].1)
        .unwrap_or(0)
}

fn unicode_canonical_decompose(cpt: u32, result: &mut Vec<u32>, pending: &mut Vec<u32>) {
    pending.push(cpt);
    while let Some(cpt) = pending.pop() {
        // Hangul syllables use algorithmic decomposition (UAX #15).
        if (0xac00..0xac00 + 11172).contains(&cpt) {
            let index = cpt - 0xac00;
            result.push(0x1100 + index / 588);
            result.push(0x1161 + (index % 588) / 28);
            if index % 28 != 0 {
                result.push(0x11a7 + index % 28);
            }
            continue;
        }
        match UNICODE_DECOMPOSITION.binary_search_by_key(&cpt, |&(key, _)| key) {
            Err(_) => result.push(cpt),
            Ok(idx) => {
                let (first, second) = UNICODE_DECOMPOSITION[idx].1;
                if second != 0 {
                    pending.push(second);
                }
                pending.push(first);
            }
        }
    }
}

fn unicode_canonical_compose(first: u32, second: u32) -> u32 {
    if (0x1100..0x1100 + 19).contains(&first) && (0x1161..0x1161 + 21).contains(&second) {
        return 0xac00 + ((first - 0x1100) * 21 + second - 0x1161) * 28;
    }
    if (0xac00..0xac00 + 11172).contains(&first)
        && (first - 0xac00) % 28 == 0
        && (0x11a7 + 1..0x11a7 + 28).contains(&second)
    {
        return first + second - 0x11a7;
    }
    let key = (u64::from(first) << 32) | u64::from(second);
    UNICODE_COMPOSITION
        .binary_search_by_key(&key, |&(k, _)| k)
        .map(|idx| UNICODE_COMPOSITION[idx].1)
        .unwrap_or(0)
}

/// NFC normalization: canonical decomposition, mark reordering, recomposition.
pub fn unicode_cpts_normalize_nfc(cpts: &[u32]) -> Vec<u32> {
    let mut decomposed = Vec::with_capacity(cpts.len());
    let mut pending = Vec::new();
    for &cpt in cpts {
        unicode_canonical_decompose(cpt, &mut decomposed, &mut pending);
    }

    // Sort each run of combining marks; equal classes retain their order.
    let mut i = 0;
    while i < decomposed.len() {
        if unicode_canonical_class(decomposed[i]) == 0 {
            i += 1;
        }
        let begin = i;
        while i < decomposed.len() && unicode_canonical_class(decomposed[i]) != 0 {
            i += 1;
        }
        decomposed[begin..i].sort_by_key(|&cpt| unicode_canonical_class(cpt));
    }

    let mut result: Vec<u32> = Vec::with_capacity(decomposed.len());
    let mut starter = 0usize;
    let mut previous_class = 0u8;
    for cpt in decomposed {
        let current_class = unicode_canonical_class(cpt);
        let composite = if !result.is_empty()
            && (previous_class == 0 || previous_class < current_class)
        {
            unicode_canonical_compose(result[starter], cpt)
        } else {
            0
        };
        if composite != 0 {
            result[starter] = composite;
        } else {
            if current_class == 0 {
                starter = result.len();
            }
            result.push(cpt);
            previous_class = current_class;
        }
    }
    result
}

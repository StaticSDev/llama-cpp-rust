//! Upper/lower case mapping tables.

use super::unicode_data::UNICODE_MAP_LOWERCASE;

pub fn unicode_tolower(cpt: u32) -> u32 {
    UNICODE_MAP_LOWERCASE
        .binary_search_by_key(&cpt, |&(key, _)| key)
        .map(|idx| UNICODE_MAP_LOWERCASE[idx].1)
        .unwrap_or(cpt)
}

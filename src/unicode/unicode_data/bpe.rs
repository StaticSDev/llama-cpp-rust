//! Regex expressions for the hand-written BPE splitters.

pub const GPT2_REGEX: &str = r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"#;
pub const LLAMA3_REGEX: &str = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;
pub const LLAMA3_ALT_REGEX: &str = r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;
pub const QWEN2_REGEX: &str = r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;
pub const QWEN35_REGEX: &str = r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;
pub const K2_HORIZON_REGEX: &str = r#"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;
pub const HAN_REGEX: &str = r#"\p{Han}+"#;
pub const AFMOE_REGEX: &str = r#"\p{AFMoE_digits}"#;
pub const DIGIT_GROUP_REGEX: &str = r"\d{1,3}(?=(?:\d{3})*\b)";
pub const NEWLINES_REGEX: &str = r"[^\n]+|[\n]+";

pub const RAON_REGEX: &str = concat!(
    " ?[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+",
    "| ?(?:(?:(?![\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}])\\p{L})\\p{M}*)+(?: (?:(?:(?![\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}])\\p{L})\\p{M}*)+)*",
    "|\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))",
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
    "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
);

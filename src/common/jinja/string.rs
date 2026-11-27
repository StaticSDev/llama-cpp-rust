//! `jinja::string`: a string split into parts that remember whether they came
//! from user input. Transformations keep that metadata (see README.md).

use std::fmt::Write;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StringPart {
    pub is_input: bool,
    pub val: String,
}

impl StringPart {
    pub fn is_uppercase(&self) -> bool {
        !self.val.bytes().any(|c| c.is_ascii_lowercase())
    }

    pub fn is_lowercase(&self) -> bool {
        !self.val.bytes().any(|c| c.is_ascii_uppercase())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JString {
    pub parts: Vec<StringPart>,
}

impl JString {
    pub fn new(v: impl Into<String>) -> Self {
        Self {
            parts: vec![StringPart {
                is_input: false,
                val: v.into(),
            }],
        }
    }

    pub fn from_input(v: impl Into<String>) -> Self {
        Self {
            parts: vec![StringPart {
                is_input: true,
                val: v.into(),
            }],
        }
    }

    /// `string(int)` / `string(double)` use `std::to_string`.
    pub fn from_int(v: i64) -> Self {
        Self::new(v.to_string())
    }

    pub fn from_float(v: f64) -> Self {
        Self::new(v.to_string())
    }

    pub fn mark_input(&mut self) {
        for part in &mut self.parts {
            part.is_input = true;
        }
    }

    pub fn str(&self) -> String {
        let mut result = String::with_capacity(self.length());
        for part in &self.parts {
            result.push_str(&part.val);
        }
        result
    }

    pub fn length(&self) -> usize {
        self.parts.iter().map(|p| p.val.len()).sum()
    }

    pub fn empty(&self) -> bool {
        self.parts.iter().all(|p| p.val.is_empty())
    }

    pub fn byte_at(&self, index: usize) -> Option<u8> {
        let mut index = index;
        for part in &self.parts {
            if index < part.val.len() {
                return part.val.as_bytes().get(index).copied();
            }
            index -= part.val.len();
        }
        None
    }

    /// Compare only the text; provenance is ignored.
    pub fn equals(&self, other: &Self) -> bool {
        let mut part = 0usize;
        let mut offset = 0usize;
        for left in &self.parts {
            let mut text: &[u8] = left.val.as_bytes();
            while !text.is_empty() {
                while part < other.parts.len() && offset == other.parts[part].val.len() {
                    part += 1;
                    offset = 0;
                }
                if part == other.parts.len() {
                    return false;
                }
                let right = other.parts[part].val.as_bytes();
                let count = text.len().min(right.len() - offset);
                if text[..count] != right[offset..offset + count] {
                    return false;
                }
                text = &text[count..];
                offset += count;
            }
        }
        while part < other.parts.len() && offset == other.parts[part].val.len() {
            part += 1;
            offset = 0;
        }
        part == other.parts.len()
    }

    pub fn all_parts_are_input(&self) -> bool {
        self.parts.iter().all(|p| p.is_input)
    }

    pub fn is_uppercase(&self) -> bool {
        self.parts.iter().all(StringPart::is_uppercase)
    }

    pub fn is_lowercase(&self) -> bool {
        self.parts.iter().all(StringPart::is_lowercase)
    }

    /// Mark this string as input if `other` has ALL parts as input.
    pub fn mark_input_based_on(&mut self, other: &Self) {
        if other.all_parts_are_input() {
            self.mark_input();
        }
    }

    pub fn append(&mut self, other: &Self) {
        self.parts.extend(other.parts.iter().cloned());
    }

    fn apply_transform<F>(&mut self, f: F)
    where
        F: Fn(&str) -> String,
    {
        for part in &mut self.parts {
            part.val = f(&part.val);
        }
    }

    pub fn uppercase(&mut self) -> Self {
        self.apply_transform(|s| {
            s.bytes()
                .map(|c| c.to_ascii_uppercase() as char)
                .collect()
        });
        self.clone()
    }

    pub fn lowercase(&mut self) -> Self {
        self.apply_transform(|s| {
            s.bytes()
                .map(|c| c.to_ascii_lowercase() as char)
                .collect()
        });
        self.clone()
    }

    pub fn capitalize(&mut self) -> Self {
        self.apply_transform(|s| {
            if s.is_empty() {
                return s.to_string();
            }
            let mut res: Vec<u8> = s.as_bytes().to_vec();
            res[0] = res[0].to_ascii_uppercase();
            for c in &mut res[1..] {
                *c = c.to_ascii_lowercase();
            }
            String::from_utf8_lossy(&res).into_owned()
        });
        self.clone()
    }

    pub fn titlecase(&mut self) -> Self {
        self.apply_transform(|s| {
            let mut res: Vec<u8> = s.as_bytes().to_vec();
            let mut capitalize_next = true;
            for c in &mut res {
                if c.is_ascii_whitespace() {
                    capitalize_next = true;
                } else if capitalize_next {
                    *c = c.to_ascii_uppercase();
                    capitalize_next = false;
                } else {
                    *c = c.to_ascii_lowercase();
                }
            }
            String::from_utf8_lossy(&res).into_owned()
        });
        self.clone()
    }

    /// `string::strip`, with optional characters to strip.
    pub fn strip(&mut self, left: bool, right: bool, chars: Option<&str>) -> Self {
        let match_char = |c: u8| match chars {
            Some(chars) => chars.as_bytes().contains(&c),
            None => c.is_ascii_whitespace(),
        };
        let strip_part = |s: &str, left: bool, right: bool| -> String {
            let bytes = s.as_bytes();
            let mut start = 0usize;
            let mut end = bytes.len();
            if left {
                while start < end && match_char(bytes[start]) {
                    start += 1;
                }
            }
            if right {
                while end > start && match_char(bytes[end - 1]) {
                    end -= 1;
                }
            }
            s[start..end].to_string()
        };
        if self.parts.is_empty() {
            return self.clone();
        }
        if left {
            let i = 0usize;
            while i < self.parts.len() {
                self.parts[i].val = strip_part(&self.parts[i].val, true, false);
                if self.parts[i].val.is_empty() {
                    self.parts.remove(i);
                } else {
                    break;
                }
            }
        }
        if right {
            let mut i = self.parts.len();
            while i > 0 {
                i -= 1;
                self.parts[i].val = strip_part(&self.parts[i].val, false, true);
                if self.parts[i].val.is_empty() {
                    self.parts.remove(i);
                } else {
                    break;
                }
            }
        }
        self.clone()
    }
}

/// Debug representation used by `value_string_t::as_repr`.
pub fn string_as_repr(s: &JString) -> String {
    let mut out = String::new();
    for part in &s.parts {
        let _ = writeln!(
            out,
            "{}{}",
            if part.is_input { "INPUT: " } else { "TMPL:  " },
            part.val
        );
    }
    out
}

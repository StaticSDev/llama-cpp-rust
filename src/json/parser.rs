//! JSON parser and error mapping.

use crate::core::{Error, Result};

use super::{json_err, Json};

impl Json {
    /// Parses valid JSON, returning an error otherwise.
    pub fn parse(text: &str) -> Result<Self> {
        Parser::new(text.as_bytes()).parse()
    }

    /// Gives a discarded value instead of throwing, check with `is_discarded()`.
    pub fn parse_no_throw(text: &str) -> Self {
        Parser::new(text.as_bytes()).parse().unwrap_or(Json::Discarded)
    }

    pub fn is_discarded(&self) -> bool {
        matches!(self, Json::Discarded)
    }
}

// ---------------------------------------------------------------------------
// parser implementation
// ---------------------------------------------------------------------------

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn parse(mut self) -> Result<Json> {
        self.skip_ws();
        let val = self.parse_value()?;
        self.skip_ws();
        if self.pos != self.bytes.len() {
            return json_err("parse error - unexpected trailing characters");
        }
        Ok(val)
    }

    fn skip_ws(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn parse_value(&mut self) -> Result<Json> {
        match self.bytes.get(self.pos) {
            Some(b'n') => {
                self.literal(b"null")?;
                Ok(Json::Null)
            }
            Some(b't') => {
                self.literal(b"true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.literal(b"false")?;
                Ok(Json::Bool(false))
            }
            Some(b'"') => Ok(Json::String(self.parse_string()?)),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            _ => json_err("parse error - syntax error while parsing value"),
        }
    }

    fn literal(&mut self, text: &[u8]) -> Result<()> {
        if self.bytes.get(self.pos..self.pos + text.len()) == Some(text) {
            self.pos += text.len();
            Ok(())
        } else {
            json_err("parse error - invalid literal")
        }
    }

    fn parse_number(&mut self) -> Result<Json> {
        let start = self.pos;
        let negative = self.bytes[self.pos] == b'-';
        if negative {
            self.pos += 1;
        }

        match self.bytes.get(self.pos) {
            Some(b'0') => {
                self.pos += 1;
                // leading zeros are not valid JSON
                if matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                    return json_err("parse error - invalid number, leading zero");
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return json_err("parse error - invalid number"),
        }

        let mut floating = false;
        if self.bytes.get(self.pos) == Some(&b'.') {
            floating = true;
            self.pos += 1;
            if !matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                return json_err("parse error - invalid number, missing fraction");
            }
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            floating = true;
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                return json_err("parse error - invalid number, missing exponent");
            }
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        // after a number only whitespace, a comma or a closing bracket may follow
        if let Some(next) = self.bytes.get(self.pos) {
            if !matches!(next, b' ' | b'\t' | b'\n' | b'\r' | b',' | b']' | b'}') {
                return json_err("parse error - invalid literal after number");
            }
        }

        let text = std::str::from_utf8(&self.bytes[start..self.pos]).expect("number is ASCII");
        if floating {
            let val: f64 = text.parse().map_err(|_| Error::new("parse error - invalid number"))?;
            if !val.is_finite() {
                return json_err("parse error - number overflow");
            }
            return Ok(Json::Double(val));
        }

        let magnitude: u64 = std::str::from_utf8(&self.bytes[start + usize::from(negative)..self.pos])
            .expect("number is ASCII")
            .parse()
            .map_err(|_| Error::new("parse error - number out of range"))?;

        if !negative {
            return Ok(Json::UInt(magnitude));
        }
        if magnitude <= i64::MAX as u64 + 1 {
            Ok(Json::Int(-(magnitude as i128) as i64))
        } else {
            // out of the integer range, fall back to a double like the lexer
            let val: f64 = text.parse().map_err(|_| Error::new("parse error - invalid number"))?;
            if !val.is_finite() {
                return json_err("parse error - number overflow");
            }
            Ok(Json::Double(val))
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            let byte = *self.bytes.get(self.pos).ok_or_else(|| Error::new("parse error - unterminated string"))?;
            match byte {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    self.parse_escape(&mut out)?;
                }
                0x00..=0x1f => return json_err("parse error - invalid control character in string"),
                0x20..=0x7f => {
                    out.push(byte as char);
                    self.pos += 1;
                }
                _ => {
                    let (ch, len) = decode_utf8(self.bytes, self.pos)?;
                    out.push(ch);
                    self.pos += len;
                }
            }
        }
    }

    fn parse_escape(&mut self, out: &mut String) -> Result<()> {
        let esc = *self.bytes.get(self.pos).ok_or_else(|| Error::new("parse error - unterminated escape"))?;
        self.pos += 1;
        match esc {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{0008}'),
            b'f' => out.push('\u{000c}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                let hi = self.parse_hex4()?;
                let cp = if (0xd800..=0xdbff).contains(&hi) {
                    if self.bytes.get(self.pos) != Some(&b'\\') || self.bytes.get(self.pos + 1) != Some(&b'u') {
                        return json_err("parse error - missing low surrogate");
                    }
                    self.pos += 2;
                    let lo = self.parse_hex4()?;
                    if !(0xdc00..=0xdfff).contains(&lo) {
                        return json_err("parse error - invalid low surrogate");
                    }
                    0x10000 + (((hi - 0xd800) as u32) << 10) + (lo - 0xdc00) as u32
                } else if (0xdc00..=0xdfff).contains(&hi) {
                    return json_err("parse error - unexpected low surrogate");
                } else {
                    hi as u32
                };
                out.push(char::from_u32(cp).ok_or_else(|| Error::new("parse error - invalid code point"))?);
            }
            _ => return json_err("parse error - invalid escape"),
        }
        Ok(())
    }

    fn parse_hex4(&mut self) -> Result<u16> {
        let mut val = 0u16;
        for _ in 0..4 {
            let byte = *self.bytes.get(self.pos).ok_or_else(|| Error::new("parse error - short \\u escape"))?;
            let digit = match byte {
                b'0'..=b'9' => (byte - b'0') as u16,
                b'a'..=b'f' => (byte - b'a' + 10) as u16,
                b'A'..=b'F' => (byte - b'A' + 10) as u16,
                _ => return json_err("parse error - invalid \\u escape"),
            };
            val = val * 16 + digit;
            self.pos += 1;
        }
        Ok(val)
    }

    fn parse_array(&mut self) -> Result<Json> {
        self.pos += 1; // '['
        let mut out = Vec::new();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(Json::Array(out));
        }
        loop {
            self.skip_ws();
            out.push(self.parse_value()?);
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return json_err("parse error - expected ',' or ']'"),
            }
        }
        Ok(Json::Array(out))
    }

    fn parse_object(&mut self) -> Result<Json> {
        self.pos += 1; // '{'
        let mut out: Vec<(String, Json)> = Vec::new();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(Json::Object(out));
        }
        loop {
            self.skip_ws();
            if self.bytes.get(self.pos) != Some(&b'"') {
                return json_err("parse error - expected string key");
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.bytes.get(self.pos) != Some(&b':') {
                return json_err("parse error - expected ':'");
            }
            self.pos += 1;
            self.skip_ws();
            let val = self.parse_value()?;
            // ordered_json keeps the first position but the last value
            if let Some(entry) = out.iter_mut().find(|e| e.0 == key) {
                entry.1 = val;
            } else {
                out.push((key, val));
            }
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return json_err("parse error - expected ',' or '}'"),
            }
        }
        Ok(Json::Object(out))
    }
}

/// Decodes one UTF-8 code point, rejecting overlong forms, surrogates and
/// values above U+10FFFF.
pub(crate) fn decode_utf8(bytes: &[u8], pos: usize) -> Result<(char, usize)> {
    let first = bytes[pos];
    let len = if first < 0x80 {
        1
    } else if (0xc2..=0xdf).contains(&first) {
        2
    } else if (0xe0..=0xef).contains(&first) {
        3
    } else if (0xf0..=0xf4).contains(&first) {
        4
    } else {
        return json_err("invalid UTF-8 byte");
    };
    if pos + len > bytes.len() {
        return json_err("incomplete UTF-8 sequence");
    }
    let mut cp = match len {
        1 => first as u32,
        2 => (first & 0x1f) as u32,
        3 => (first & 0x0f) as u32,
        _ => (first & 0x07) as u32,
    };
    for i in 1..len {
        let byte = bytes[pos + i];
        if byte & 0xc0 != 0x80 {
            return json_err("invalid UTF-8 continuation byte");
        }
        cp = (cp << 6) | (byte & 0x3f) as u32;
    }
    if (len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000) {
        return json_err("overlong UTF-8 sequence");
    }
    if cp > 0x10ffff || (0xd800..=0xdfff).contains(&cp) {
        return json_err("invalid UTF-8 code point");
    }
    Ok((char::from_u32(cp).expect("validated code point"), len))
}

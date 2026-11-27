//! JSON serializer: escaping, indentation and float formatting.

use crate::core::Result;

use super::parser::decode_utf8;
use super::{json_err, Json};

impl Json {
    // -----------------------------------------------------------------------
    // serializer
    // -----------------------------------------------------------------------

    /// Serializes to JSON. An indent of -1 keeps the output compact.
    pub fn dump(&self, indent: i32) -> Result<String> {
        let mut out = String::new();
        let pretty = indent >= 0;
        self.write_json(&mut out, pretty, indent.max(0) as usize, 0, false)?;
        Ok(out)
    }

    /// Same as `dump()`, but bad UTF-8 is replaced instead of erroring.
    pub fn dump_safe(&self, indent: i32) -> String {
        let mut out = String::new();
        let pretty = indent >= 0;
        let _ = self.write_json(&mut out, pretty, indent.max(0) as usize, 0, true);
        out
    }

    fn write_json(&self, out: &mut String, pretty: bool, indent: usize, level: usize, replace: bool) -> Result<()> {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
            Json::Int(v) => out.push_str(&v.to_string()),
            Json::UInt(v) => out.push_str(&v.to_string()),
            Json::Double(v) => dump_double(*v, out),
            Json::String(s) => write_string(s, out, replace)?,
            Json::Array(a) => {
                out.push('[');
                if a.is_empty() {
                    out.push(']');
                    return Ok(());
                }
                if pretty {
                    out.push('\n');
                }
                for (i, val) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                        if pretty {
                            out.push('\n');
                        }
                    }
                    if pretty {
                        for _ in 0..indent * (level + 1) {
                            out.push(' ');
                        }
                    }
                    val.write_json(out, pretty, indent, level + 1, replace)?;
                }
                if pretty {
                    out.push('\n');
                    for _ in 0..indent * level {
                        out.push(' ');
                    }
                }
                out.push(']');
            }
            Json::Object(o) => {
                out.push('{');
                if o.is_empty() {
                    out.push('}');
                    return Ok(());
                }
                if pretty {
                    out.push('\n');
                }
                for (i, (key, val)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                        if pretty {
                            out.push('\n');
                        }
                    }
                    if pretty {
                        for _ in 0..indent * (level + 1) {
                            out.push(' ');
                        }
                    }
                    write_string(key, out, replace)?;
                    out.push(':');
                    if pretty {
                        out.push(' ');
                    }
                    val.write_json(out, pretty, indent, level + 1, replace)?;
                }
                if pretty {
                    out.push('\n');
                    for _ in 0..indent * level {
                        out.push(' ');
                    }
                }
                out.push('}');
            }
            Json::Discarded => return json_err("stored value is not serializable"),
        }
        Ok(())
    }

}

const HEX: &[u8; 16] = b"0123456789abcdef";

fn write_string(s: &str, out: &mut String, replace: bool) -> Result<()> {
    let bytes = s.as_bytes();
    let mut i = 0;
    out.push('"');
    while i < bytes.len() {
        let byte = bytes[i];
        match byte {
            b'"' => out.push_str("\\\""),
            b'\\' => out.push_str("\\\\"),
            0x08 => out.push_str("\\b"),
            0x0c => out.push_str("\\f"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x00..=0x1f => {
                out.push_str("\\u00");
                out.push(HEX[(byte >> 4) as usize] as char);
                out.push(HEX[(byte & 0x0f) as usize] as char);
            }
            0x20..=0x7f => out.push(byte as char),
            _ => match decode_utf8(bytes, i) {
                Ok((ch, len)) => {
                    out.push(ch);
                    i += len;
                    continue;
                }
                Err(_) if replace => out.push('\u{fffd}'),
                Err(_) => return json_err(format!("invalid UTF-8 byte at index {}", i)),
            },
        }
        i += 1;
    }
    out.push('"');
    Ok(())
}

/// Shortest round-trip float, using nlohmann's `e+NN` exponent spelling and
/// a `.0` for values without a fractional part. Non-finite values become null.
fn dump_double(val: f64, out: &mut String) {
    if !val.is_finite() {
        out.push_str("null");
        return;
    }
    if val.is_sign_negative() {
        out.push('-');
    }
    let magnitude = format!("{:e}", val.abs());
    let (mantissa, exponent) = magnitude.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exponent.parse().expect("LowerExp exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();

    if exp >= -4 && exp < 21 {
        if exp >= 0 {
            let point = (exp + 1) as usize;
            if point >= digits.len() {
                out.push_str(&digits);
                for _ in 0..point.saturating_sub(digits.len()) {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..point]);
                out.push('.');
                out.push_str(&digits[point..]);
            }
        } else {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        if exp >= 0 {
            out.push('+');
        } else {
            out.push('-');
        }
        let abs_exp = exp.abs();
        if abs_exp < 10 {
            out.push('0');
        }
        out.push_str(&abs_exp.to_string());
    }
}


//! `jinja::lexer`: tokenize a template without rewriting the source, so byte
//! positions stay usable for error reporting.

use super::utils::{cpt_to_utf8, fmt_error_with_source, JResult, JinjaError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenType {
    Eof,
    Text,
    NumericLiteral,
    StringLiteral,
    Identifier,
    Equals,
    OpenParen,
    CloseParen,
    OpenStatement,
    CloseStatement,
    OpenExpression,
    CloseExpression,
    OpenSquareBracket,
    CloseSquareBracket,
    OpenCurlyBracket,
    CloseCurlyBracket,
    Comma,
    Dot,
    Colon,
    Pipe,
    CallOperator,
    AdditiveBinaryOperator,
    MultiplicativeBinaryOperator,
    PowerOperator,
    ComparisonBinaryOperator,
    Comment,
}

impl TokenType {
    pub fn as_str(self) -> &'static str {
        match self {
            TokenType::Eof => "eof",
            TokenType::Text => "text",
            TokenType::NumericLiteral => "numeric_literal",
            TokenType::StringLiteral => "string_literal",
            TokenType::Identifier => "identifier",
            TokenType::Equals => "equals",
            TokenType::OpenParen => "open_paren",
            TokenType::CloseParen => "close_paren",
            TokenType::OpenStatement => "open_statement",
            TokenType::CloseStatement => "close_statement",
            TokenType::OpenExpression => "open_expression",
            TokenType::CloseExpression => "close_expression",
            TokenType::OpenSquareBracket => "open_square_bracket",
            TokenType::CloseSquareBracket => "close_square_bracket",
            TokenType::OpenCurlyBracket => "open_curly_bracket",
            TokenType::CloseCurlyBracket => "close_curly_bracket",
            TokenType::Comma => "comma",
            TokenType::Dot => "dot",
            TokenType::Colon => "colon",
            TokenType::Pipe => "pipe",
            TokenType::CallOperator => "call_operator",
            TokenType::AdditiveBinaryOperator => "additive_binary_operator",
            TokenType::MultiplicativeBinaryOperator => "multiplicative_binary_operator",
            TokenType::PowerOperator => "power_operator",
            TokenType::ComparisonBinaryOperator => "comparison_binary_operator",
            TokenType::Comment => "comment",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub struct token {
    pub t: TokenType,
    pub value: String,
    pub pos: usize,
}

#[derive(Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct lexer_result {
    pub tokens: Vec<token>,
    pub source: String,
}

pub struct LexerException;

impl LexerException {
    pub fn new(msg: &str, source: &str, pos: usize) -> JinjaError {
        JinjaError::Error(fmt_error_with_source("lexer", msg, source, pos))
    }
}

fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn is_integer(c: u8) -> bool {
    c.is_ascii_digit()
}

fn string_lstrip(s: &mut String, chars: &[u8]) {
    let start = s
        .bytes()
        .position(|c| !chars.contains(&c))
        .unwrap_or(s.len());
    if start == s.len() {
        s.clear();
    } else {
        s.drain(..start);
    }
}

fn string_rstrip(s: &mut String, chars: &[u8]) {
    let end = s
        .bytes()
        .rposition(|c| !chars.contains(&c))
        .map(|p| p + 1)
        .unwrap_or(0);
    if end == 0 {
        s.clear();
    } else {
        s.truncate(end);
    }
}

const MAPPING_TABLE: &[(&str, TokenType)] = &[
    // trimmed control sequences
    ("{%-", TokenType::OpenStatement),
    ("-%}", TokenType::CloseStatement),
    ("{{-", TokenType::OpenExpression),
    ("-}}", TokenType::CloseExpression),
    // control sequences
    ("{%", TokenType::OpenStatement),
    ("%}", TokenType::CloseStatement),
    ("{{", TokenType::OpenExpression),
    ("}}", TokenType::CloseExpression),
    // single character tokens
    ("(", TokenType::OpenParen),
    (")", TokenType::CloseParen),
    ("{", TokenType::OpenCurlyBracket),
    ("}", TokenType::CloseCurlyBracket),
    ("[", TokenType::OpenSquareBracket),
    ("]", TokenType::CloseSquareBracket),
    (",", TokenType::Comma),
    (".", TokenType::Dot),
    (":", TokenType::Colon),
    ("|", TokenType::Pipe),
    // comparison operators
    ("<=", TokenType::ComparisonBinaryOperator),
    (">=", TokenType::ComparisonBinaryOperator),
    ("==", TokenType::ComparisonBinaryOperator),
    ("!=", TokenType::ComparisonBinaryOperator),
    ("<", TokenType::ComparisonBinaryOperator),
    (">", TokenType::ComparisonBinaryOperator),
    // arithmetic operators
    ("+", TokenType::AdditiveBinaryOperator),
    ("-", TokenType::AdditiveBinaryOperator),
    ("~", TokenType::AdditiveBinaryOperator),
    ("**", TokenType::PowerOperator),
    ("*", TokenType::MultiplicativeBinaryOperator),
    ("//", TokenType::MultiplicativeBinaryOperator),
    ("/", TokenType::MultiplicativeBinaryOperator),
    ("%", TokenType::MultiplicativeBinaryOperator),
    // assignment
    ("=", TokenType::Equals),
];

struct Lexer<'a> {
    source: &'a str,
    src: Vec<u8>,
    tokens: Vec<token>,
    pos: usize,
    start_pos: usize,
    curly_bracket_depth: usize,
}

impl<'a> Lexer<'a> {
    fn err(&self, msg: &str, pos: usize) -> JinjaError {
        LexerException::new(msg, self.source, pos)
    }

    fn next_pos_is(&self, chars: &[u8], n: usize) -> bool {
        if self.pos + n >= self.src.len() {
            return false;
        }
        chars.contains(&self.src[self.pos + n])
    }

    fn consume_while<F>(&mut self, pred: F) -> JResult<String>
    where
        F: Fn(u8) -> bool,
    {
        let mut str_bytes: Vec<u8> = Vec::new();
        while self.pos < self.src.len() && pred(self.src[self.pos]) {
            if self.src[self.pos] == b'\\' {
                self.pos += 1;
                if self.pos >= self.src.len() {
                    return Err(self.err("unexpected end of input after escape character", self.pos));
                }
                let escaped = self.src[self.pos];
                self.pos += 1;
                if escaped == b'x' {
                    if self.pos + 1 >= self.src.len() {
                        return Err(self.err("incomplete hexadecimal escape", self.pos));
                    }
                    let digits = &self.src[self.pos..self.pos + 2];
                    let value = u32::from_str_radix(
                        std::str::from_utf8(digits).unwrap_or(""),
                        16,
                    )
                    .map_err(|_| self.err("invalid hexadecimal escape", self.pos))?;
                    if value > 0x10ffff {
                        return Err(self.err("invalid codepoint", self.pos));
                    }
                    str_bytes.extend_from_slice(&cpt_to_utf8(value));
                    self.pos += 2;
                    continue;
                }
                let unescaped = match escaped {
                    b'n' => b'\n',
                    b't' => b'\t',
                    b'r' => b'\r',
                    b'b' => 0x08,
                    b'f' => 0x0c,
                    b'v' => 0x0b,
                    b'\\' => b'\\',
                    b'\'' => b'\'',
                    b'"' => b'"',
                    _ => {
                        let mut msg = String::from("unknown escape character \\");
                        msg.push(escaped as char);
                        return Err(self.err(&msg, self.pos));
                    }
                };
                str_bytes.push(unescaped);
                continue;
            }
            str_bytes.push(self.src[self.pos]);
            self.pos += 1;
            if self.pos > self.src.len() {
                return Err(self.err("unexpected end of input during consume_while", self.pos));
            }
        }
        Ok(String::from_utf8_lossy(&str_bytes).into_owned())
    }

    fn consume_numeric(&mut self) -> JResult<String> {
        let mut num = self.consume_while(is_integer)?;
        if self.pos < self.src.len()
            && self.src[self.pos] == b'.'
            && self.pos + 1 < self.src.len()
            && is_integer(self.src[self.pos + 1])
        {
            self.pos += 1;
            let frac = self.consume_while(is_integer)?;
            num.push('.');
            num.push_str(&frac);
        }
        Ok(num)
    }
}

/// `lexer::tokenize`.
pub fn tokenize(source: &str) -> JResult<lexer_result> {
    let mut lx = Lexer {
        source,
        src: Vec::new(),
        tokens: Vec::new(),
        pos: 0,
        start_pos: 0,
        curly_bracket_depth: 0,
    };
    if source.is_empty() {
        return Ok(lexer_result {
            tokens: Vec::new(),
            source: source.to_string(),
        });
    }

    // Normalize \r\n or \r to \n without changing the reported source.
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\r' {
            lx.src.push(b'\n');
            if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                i += 1;
            }
        } else {
            lx.src.push(bytes[i]);
        }
        i += 1;
    }
    // A single trailing newline is stripped.
    if source.as_bytes().last() == Some(&b'\n') {
        lx.src.pop();
    }

    let opt_lstrip_blocks = true;
    let opt_trim_blocks = true;
    while lx.pos < lx.src.len() {
        lx.start_pos = lx.pos;

        let last_token_type = lx
            .tokens
            .last()
            .map(|t| t.t)
            .unwrap_or(TokenType::CloseStatement);

        if matches!(
            last_token_type,
            TokenType::CloseStatement | TokenType::CloseExpression | TokenType::Comment
        ) {
            let mut last_block_can_rm_newline = false;
            let mut is_rstrip_block = false;
            if lx.pos > 3 {
                let c0 = lx.src[lx.pos - 3];
                let c1 = lx.src[lx.pos - 2];
                let c2 = lx.src[lx.pos - 1];
                is_rstrip_block = c0 == b'-' && (c1 == b'%' || c1 == b'}' || c1 == b'#') && c2 == b'}';
                last_block_can_rm_newline = (c1 == b'#' || c1 == b'%' || c1 == b'-') && c2 == b'}';
            }

            let start = lx.pos;
            let mut end = start;
            while lx.pos < lx.src.len()
                && !(lx.src[lx.pos] == b'{'
                    && lx.next_pos_is(&[b'%', b'{', b'#'], 1))
            {
                lx.pos += 1;
                end = lx.pos;
            }

            if opt_lstrip_blocks
                && lx.pos < lx.src.len()
                && lx.src[lx.pos] == b'{'
                && lx.next_pos_is(&[b'%', b'#', b'-'], 1)
            {
                let mut current = end;
                while current > start {
                    let c = lx.src[current - 1];
                    if current == 1 {
                        end = 0;
                        break;
                    }
                    if c == b'\n' {
                        end = current;
                        break;
                    }
                    if !c.is_ascii_whitespace() {
                        break;
                    }
                    current -= 1;
                }
            }

            let mut text = String::from_utf8_lossy(&lx.src[start..end]).into_owned();
            if opt_trim_blocks && last_block_can_rm_newline && text.starts_with('\n') {
                text.remove(0);
            }
            if is_rstrip_block {
                string_lstrip(&mut text, b" \t\r\n");
            }
            let is_lstrip_block = lx.pos < lx.src.len()
                && lx.src[lx.pos] == b'{'
                && lx.next_pos_is(&[b'{', b'%', b'#'], 1)
                && lx.next_pos_is(&[b'-'], 2);
            if is_lstrip_block {
                string_rstrip(&mut text, b" \t\r\n");
            }
            if !text.is_empty() {
                lx.tokens.push(token {
                    t: TokenType::Text,
                    value: text,
                    pos: lx.start_pos,
                });
                continue;
            }
        }

        // Comment
        if lx.pos < lx.src.len()
            && lx.src[lx.pos] == b'{'
            && lx.next_pos_is(&[b'#'], 1)
        {
            lx.start_pos = lx.pos;
            lx.pos += 2;
            let mut comment_bytes = Vec::new();
            while lx.pos < lx.src.len()
                && !(lx.src[lx.pos] == b'#' && lx.next_pos_is(&[b'}'], 1))
            {
                if lx.pos + 2 >= lx.src.len() {
                    return Err(lx.err("missing end of comment tag", lx.pos));
                }
                comment_bytes.push(lx.src[lx.pos]);
                lx.pos += 1;
            }
            lx.tokens.push(token {
                t: TokenType::Comment,
                value: String::from_utf8_lossy(&comment_bytes).into_owned(),
                pos: lx.start_pos,
            });
            lx.pos += 2;
            continue;
        }

        if lx.pos < lx.src.len()
            && lx.src[lx.pos] == b'-'
            && (last_token_type == TokenType::OpenExpression
                || last_token_type == TokenType::OpenStatement)
        {
            lx.pos += 1;
            if lx.pos >= lx.src.len() {
                break;
            }
        }

        // Ignore whitespace inside a block.
        let _ = lx.consume_while(|c| c.is_ascii_whitespace())?;
        if lx.pos >= lx.src.len() {
            break;
        }

        let ch = lx.src[lx.pos];
        let mut matched = false;
        for (seq, typ) in MAPPING_TABLE {
            lx.start_pos = lx.pos;
            if *seq == "}}" && lx.curly_bracket_depth > 0 {
                continue;
            }
            if lx.pos + seq.len() <= lx.src.len()
                && &lx.src[lx.pos..lx.pos + seq.len()] == seq.as_bytes()
            {
                lx.tokens.push(token {
                    t: *typ,
                    value: (*seq).to_string(),
                    pos: lx.start_pos,
                });
                if *typ == TokenType::OpenExpression {
                    lx.curly_bracket_depth = 0;
                } else if *typ == TokenType::OpenCurlyBracket {
                    lx.curly_bracket_depth += 1;
                } else if *typ == TokenType::CloseCurlyBracket {
                    lx.curly_bracket_depth = lx.curly_bracket_depth.saturating_sub(1);
                }
                lx.pos += seq.len();
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }

        // String literal
        if ch == b'\'' || ch == b'"' {
            lx.start_pos = lx.pos;
            lx.pos += 1;
            let str_val = lx.consume_while(|c| c != ch)?;
            lx.tokens.push(token {
                t: TokenType::StringLiteral,
                value: str_val,
                pos: lx.start_pos,
            });
            lx.pos += 1;
            continue;
        }

        // Number
        if is_integer(ch) {
            lx.start_pos = lx.pos;
            let num = lx.consume_numeric()?;
            lx.tokens.push(token {
                t: TokenType::NumericLiteral,
                value: num,
                pos: lx.start_pos,
            });
            continue;
        }

        // Identifier
        if is_word(ch) {
            lx.start_pos = lx.pos;
            let word = lx.consume_while(is_word)?;
            lx.tokens.push(token {
                t: TokenType::Identifier,
                value: word,
                pos: lx.start_pos,
            });
            continue;
        }

        let mut msg = String::from("unexpected character: ");
        msg.push(ch as char);
        return Err(lx.err(&msg, lx.pos));
    }

    Ok(lexer_result {
        tokens: lx.tokens,
        source: String::from_utf8_lossy(&lx.src).into_owned(),
    })
}

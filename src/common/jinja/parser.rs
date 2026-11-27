//! `jinja::parser`: recursive-descent parser from tokens to a program (AST).

use std::cell::Cell;
use std::rc::Rc;

use super::lexer::{token, lexer_result, TokenType};
use super::runtime::{Program, Stmt, StmtNode};
use super::utils::{fmt_error_with_source, JResult, JinjaError, JINJA_RECURSIVE_LIMIT};

pub struct ParserException;

impl ParserException {
    pub fn new(msg: &str, source: &str, pos: usize) -> JinjaError {
        JinjaError::Error(fmt_error_with_source("parser", msg, source, pos))
    }
}

struct DepthGuard {
    depth: Rc<Cell<usize>>,
}

impl DepthGuard {
    fn new(parser: &Parser) -> JResult<Self> {
        let next = parser.depth.get() + 1;
        if next > JINJA_RECURSIVE_LIMIT {
            return Err(ParserException::new(
                &format!(
                    "Parser Error: template is nested more than {JINJA_RECURSIVE_LIMIT} levels deep"
                ),
                &parser.source,
                parser.peek(0).pos,
            ));
        }
        parser.depth.set(next);
        Ok(Self {
            depth: parser.depth.clone(),
        })
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        self.depth.set(self.depth.get() - 1);
    }
}

struct Parser<'a> {
    tokens: &'a [token],
    current: usize,
    depth: Rc<Cell<usize>>,
    source: String,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [token], source: &str) -> Self {
        Self {
            tokens,
            current: 0,
            depth: Rc::new(Cell::new(0)),
            source: source.to_string(),
        }
    }

    fn peek(&self, offset: usize) -> token {
        let idx = self.current + offset;
        if idx < self.tokens.len() {
            self.tokens[idx].clone()
        } else {
            token {
                t: TokenType::Eof,
                value: String::new(),
                pos: 0,
            }
        }
    }

    fn next(&mut self) -> JResult<token> {
        if self.current >= self.tokens.len() {
            let pos = self.tokens.last().map(|t| t.pos).unwrap_or(0);
            return Err(ParserException::new("Parser Error: Unexpected EOF", &self.source, pos));
        }
        let t = self.tokens[self.current].clone();
        self.current += 1;
        Ok(t)
    }

    fn expect(&mut self, t: TokenType, error: &str) -> JResult<token> {
        let tok = self.peek(0);
        if tok.t != t {
            return Err(ParserException::new(
                &format!("Parser Error: {error} (Got {})", tok.value),
                &self.source,
                tok.pos,
            ));
        }
        self.current += 1;
        Ok(tok)
    }

    fn expect_identifier(&mut self, name: &str) -> JResult<()> {
        let tok = self.peek(0);
        if tok.t != TokenType::Identifier || tok.value != name {
            return Err(ParserException::new(
                &format!("Expected identifier: {name}"),
                &self.source,
                tok.pos,
            ));
        }
        self.current += 1;
        Ok(())
    }

    fn is(&self, t: TokenType) -> bool {
        self.peek(0).t == t
    }

    fn is_identifier(&self, name: &str) -> bool {
        let t = self.peek(0);
        t.t == TokenType::Identifier && t.value == name
    }

    fn is_statement(&self, names: &[&str]) -> bool {
        let a = self.peek(0);
        let b = self.peek(1);
        if a.t != TokenType::OpenStatement || b.t != TokenType::Identifier {
            return false;
        }
        names.contains(&b.value.as_str())
    }

    fn stmt(&self, start_pos: usize, node: StmtNode) -> Stmt {
        let pos = self
            .tokens
            .get(start_pos)
            .map(|t| t.pos)
            .unwrap_or_else(|| self.tokens.last().map(|t| t.pos).unwrap_or(0));
        Stmt::new(pos, node)
    }

    fn parse(&mut self) -> JResult<Program> {
        let mut body = Vec::new();
        while self.current < self.tokens.len() {
            body.push(self.parse_any()?);
        }
        Ok(self.stmt(0, StmtNode::Program { body }))
    }

    fn parse_any(&mut self) -> JResult<Stmt> {
        let _guard = DepthGuard::new(self)?;
        let start_pos = self.current;
        match self.peek(0).t {
            TokenType::Comment => {
                let v = self.next()?.value;
                Ok(self.stmt(start_pos, StmtNode::Comment(v)))
            }
            TokenType::Text => {
                let v = self.next()?.value;
                Ok(self.stmt(start_pos, StmtNode::StringLiteral(v)))
            }
            TokenType::OpenStatement => self.parse_jinja_statement(),
            TokenType::OpenExpression => self.parse_jinja_expression(),
            _ => Err(JinjaError::Error("Unexpected token type".into())),
        }
    }

    fn parse_jinja_expression(&mut self) -> JResult<Stmt> {
        self.expect(TokenType::OpenExpression, "Expected {{")?;
        let result = self.parse_expression()?;
        self.expect(TokenType::CloseExpression, "Expected }}")?;
        Ok(result)
    }

    fn parse_jinja_statement(&mut self) -> JResult<Stmt> {
        self.expect(TokenType::OpenStatement, "Expected {%")?;
        if self.peek(0).t != TokenType::Identifier {
            return Err(JinjaError::Error("Unknown statement".into()));
        }
        let start_pos = self.current;
        let name = self.next()?.value;
        let result = match name.as_str() {
            "set" => self.parse_set_statement(start_pos)?,
            "if" => {
                let r = self.parse_if_statement(start_pos)?;
                self.expect(TokenType::OpenStatement, "Expected {%")?;
                self.expect_identifier("endif")?;
                self.expect(TokenType::CloseStatement, "Expected %}")?;
                r
            }
            "macro" => {
                let r = self.parse_macro_statement(start_pos)?;
                self.expect(TokenType::OpenStatement, "Expected {%")?;
                self.expect_identifier("endmacro")?;
                self.expect(TokenType::CloseStatement, "Expected %}")?;
                r
            }
            "for" => {
                let r = self.parse_for_statement(start_pos)?;
                self.expect(TokenType::OpenStatement, "Expected {%")?;
                self.expect_identifier("endfor")?;
                self.expect(TokenType::CloseStatement, "Expected %}")?;
                r
            }
            "break" => {
                self.expect(TokenType::CloseStatement, "Expected %}")?;
                self.stmt(start_pos, StmtNode::Break)
            }
            "continue" => {
                self.expect(TokenType::CloseStatement, "Expected %}")?;
                self.stmt(start_pos, StmtNode::Continue)
            }
            "call" => {
                let caller_args = if self.is(TokenType::OpenParen) {
                    self.parse_args(false)?
                } else {
                    Vec::new()
                };
                let callee = self.parse_primary_expression()?;
                if !matches!(callee.node, StmtNode::Identifier(_)) {
                    return Err(JinjaError::Error("Expected identifier".into()));
                }
                let call_args = self.parse_args(true)?;
                self.expect(TokenType::CloseStatement, "Expected %}")?;
                let mut body = Vec::new();
                while !self.is_statement(&["endcall"]) {
                    body.push(self.parse_any()?);
                }
                self.expect(TokenType::OpenStatement, "Expected {%")?;
                self.expect_identifier("endcall")?;
                self.expect(TokenType::CloseStatement, "Expected %}")?;
                let call_expr = self.stmt(
                    start_pos,
                    StmtNode::CallExpression {
                        callee,
                        args: call_args,
                    },
                );
                self.stmt(
                    start_pos,
                    StmtNode::CallStatement {
                        call: call_expr,
                        caller_args,
                        body,
                    },
                )
            }
            "filter" => {
                let mut filter_node = self.parse_primary_expression()?;
                if matches!(filter_node.node, StmtNode::Identifier(_)) && self.is(TokenType::OpenParen) {
                    filter_node = self.parse_call_expression(filter_node)?;
                }
                self.expect(TokenType::CloseStatement, "Expected %}")?;
                let mut body = Vec::new();
                while !self.is_statement(&["endfilter"]) {
                    body.push(self.parse_any()?);
                }
                self.expect(TokenType::OpenStatement, "Expected {%")?;
                self.expect_identifier("endfilter")?;
                self.expect(TokenType::CloseStatement, "Expected %}")?;
                self.stmt(start_pos, StmtNode::FilterStatement { filter: filter_node, body })
            }
            "generation" | "endgeneration" => {
                let r = self.stmt(start_pos, StmtNode::Noop);
                self.current += 1;
                r
            }
            _ => return Err(JinjaError::Error(format!("Unknown statement: {name}"))),
        };
        Ok(result)
    }

    fn parse_set_statement(&mut self, start_pos: usize) -> JResult<Stmt> {
        let left = self.parse_expression_sequence(false)?;
        let mut value = None;
        let mut body = Vec::new();
        if self.is(TokenType::Equals) {
            self.current += 1;
            value = Some(self.parse_expression_sequence(false)?);
        } else {
            self.expect(TokenType::CloseStatement, "Expected %}")?;
            while !self.is_statement(&["endset"]) {
                body.push(self.parse_any()?);
            }
            self.expect(TokenType::OpenStatement, "Expected {%")?;
            self.expect_identifier("endset")?;
        }
        self.expect(TokenType::CloseStatement, "Expected %}")?;
        Ok(self.stmt(start_pos, StmtNode::Set { assignee: left, val: value, body }))
    }

    fn parse_if_statement(&mut self, start_pos: usize) -> JResult<Stmt> {
        let test = self.parse_expression()?;
        self.expect(TokenType::CloseStatement, "Expected %}")?;
        let mut body = Vec::new();
        let mut alternate = Vec::new();
        while !self.is_statement(&["elif", "else", "endif"]) {
            body.push(self.parse_any()?);
        }
        if self.is_statement(&["elif"]) {
            let pos0 = self.current;
            self.current += 2;
            let _guard = DepthGuard::new(self)?;
            alternate.push(self.parse_if_statement(pos0)?);
        } else if self.is_statement(&["else"]) {
            self.current += 2;
            self.expect(TokenType::CloseStatement, "Expected %}")?;
            while !self.is_statement(&["endif"]) {
                alternate.push(self.parse_any()?);
            }
        }
        Ok(self.stmt(start_pos, StmtNode::If { test, body, alternate }))
    }

    fn parse_macro_statement(&mut self, start_pos: usize) -> JResult<Stmt> {
        let name = self.parse_primary_expression()?;
        let args = self.parse_args(false)?;
        self.expect(TokenType::CloseStatement, "Expected %}")?;
        let mut body = Vec::new();
        while !self.is_statement(&["endmacro"]) {
            body.push(self.parse_any()?);
        }
        Ok(self.stmt(start_pos, StmtNode::Macro { name, args, body }))
    }

    fn parse_expression_sequence(&mut self, primary: bool) -> JResult<Stmt> {
        let start_pos = self.current;
        let mut exprs = Vec::new();
        exprs.push(if primary {
            self.parse_primary_expression()?
        } else {
            self.parse_expression()?
        });
        let is_tuple = self.is(TokenType::Comma);
        while self.is(TokenType::Comma) {
            self.current += 1;
            exprs.push(if primary {
                self.parse_primary_expression()?
            } else {
                self.parse_expression()?
            });
        }
        if is_tuple {
            Ok(self.stmt(start_pos, StmtNode::TupleLiteral(exprs)))
        } else {
            Ok(exprs.remove(0))
        }
    }

    fn parse_for_statement(&mut self, start_pos: usize) -> JResult<Stmt> {
        let loop_var = self.parse_expression_sequence(true)?;
        if !self.is_identifier("in") {
            return Err(JinjaError::Error("Expected 'in'".into()));
        }
        self.current += 1;
        let iterable = self.parse_expression()?;
        self.expect(TokenType::CloseStatement, "Expected %}")?;
        let mut body = Vec::new();
        let mut alternate = Vec::new();
        while !self.is_statement(&["endfor", "else"]) {
            body.push(self.parse_any()?);
        }
        if self.is_statement(&["else"]) {
            self.current += 2;
            self.expect(TokenType::CloseStatement, "Expected %}")?;
            while !self.is_statement(&["endfor"]) {
                alternate.push(self.parse_any()?);
            }
        }
        Ok(self.stmt(
            start_pos,
            StmtNode::For {
                loopvar: loop_var,
                iterable,
                body,
                default_block: alternate,
            },
        ))
    }

    fn parse_expression(&mut self) -> JResult<Stmt> {
        self.parse_if_expression()
    }

    fn parse_if_expression(&mut self) -> JResult<Stmt> {
        let _guard = DepthGuard::new(self)?;
        let a = self.parse_logical_or_expression()?;
        if self.is_identifier("if") {
            let start_pos = self.current;
            self.current += 1;
            let test = self.parse_logical_or_expression()?;
            if self.is_identifier("else") {
                let pos0 = self.current;
                self.current += 1;
                let false_expr = self.parse_if_expression()?;
                Ok(self.stmt(
                    pos0,
                    StmtNode::Ternary {
                        condition: test,
                        true_expr: a,
                        false_expr,
                    },
                ))
            } else {
                Ok(self.stmt(start_pos, StmtNode::Select { lhs: a, test }))
            }
        } else {
            Ok(a)
        }
    }

    fn parse_logical_or_expression(&mut self) -> JResult<Stmt> {
        let mut left = self.parse_logical_and_expression()?;
        while self.is_identifier("or") {
            let start_pos = self.current;
            let op = self.next()?;
            let right = self.parse_logical_and_expression()?;
            left = self.stmt(start_pos, StmtNode::Binary { op, left, right });
        }
        Ok(left)
    }

    fn parse_logical_and_expression(&mut self) -> JResult<Stmt> {
        let mut left = self.parse_logical_negation_expression()?;
        while self.is_identifier("and") {
            let start_pos = self.current;
            let op = self.next()?;
            let right = self.parse_logical_negation_expression()?;
            left = self.stmt(start_pos, StmtNode::Binary { op, left, right });
        }
        Ok(left)
    }

    fn parse_logical_negation_expression(&mut self) -> JResult<Stmt> {
        if self.is_identifier("not") {
            let _guard = DepthGuard::new(self)?;
            let start_pos = self.current;
            let op = self.next()?;
            let arg = self.parse_logical_negation_expression()?;
            return Ok(self.stmt(start_pos, StmtNode::Unary { op, argument: arg }));
        }
        self.parse_comparison_expression()
    }

    fn parse_comparison_expression(&mut self) -> JResult<Stmt> {
        let mut operands = Vec::new();
        let mut ops = Vec::new();
        operands.push(self.parse_additive_expression()?);
        let pos = self.current;
        loop {
            let op = if self.is_identifier("not")
                && self.peek(1).t == TokenType::Identifier
                && self.peek(1).value == "in"
            {
                let p = self.peek(0).pos;
                self.current += 2;
                token {
                    t: TokenType::Identifier,
                    value: "not in".into(),
                    pos: p,
                }
            } else if self.is_identifier("in") || self.is(TokenType::ComparisonBinaryOperator) {
                self.next()?
            } else {
                break;
            };
            ops.push(op);
            operands.push(self.parse_additive_expression()?);
        }
        if ops.is_empty() {
            return Ok(operands.remove(0));
        }
        if ops.len() == 1 {
            return Ok(self.stmt(
                pos,
                StmtNode::Binary {
                    op: ops.remove(0),
                    left: operands.remove(0),
                    right: operands.remove(0),
                },
            ));
        }
        Ok(self.stmt(pos, StmtNode::ComparisonChain { operands, ops }))
    }

    fn parse_additive_expression(&mut self) -> JResult<Stmt> {
        let mut left = self.parse_multiplicative_expression()?;
        while self.is(TokenType::AdditiveBinaryOperator) {
            let start_pos = self.current;
            let op = self.next()?;
            let right = self.parse_multiplicative_expression()?;
            left = self.stmt(start_pos, StmtNode::Binary { op, left, right });
        }
        Ok(left)
    }

    fn parse_multiplicative_expression(&mut self) -> JResult<Stmt> {
        let mut left = self.parse_test_expression()?;
        while self.is(TokenType::MultiplicativeBinaryOperator) {
            let start_pos = self.current;
            let op = self.next()?;
            let right = self.parse_test_expression()?;
            left = self.stmt(start_pos, StmtNode::Binary { op, left, right });
        }
        Ok(left)
    }

    fn parse_test_expression(&mut self) -> JResult<Stmt> {
        let mut operand = self.parse_filter_expression()?;
        while self.is_identifier("is") {
            let start_pos = self.current;
            self.current += 1;
            let mut negate = false;
            if self.is_identifier("not") {
                self.current += 1;
                negate = true;
            }
            let mut test_id = self.parse_primary_expression()?;
            if self.is(TokenType::OpenParen) {
                test_id = self.parse_call_expression(test_id)?;
            } else if self.is(TokenType::NumericLiteral)
                || self.is(TokenType::StringLiteral)
                || self.is(TokenType::OpenCurlyBracket)
                || self.is(TokenType::OpenSquareBracket)
                || (self.is(TokenType::Identifier)
                    && !self.is_identifier("and")
                    && !self.is_identifier("or")
                    && !self.is_identifier("else"))
            {
                let call_pos = self.current;
                let mut args = Vec::new();
                args.push(self.parse_unary_expression()?);
                test_id = self.stmt(call_pos, StmtNode::CallExpression { callee: test_id, args });
            }
            operand = self.stmt(
                start_pos,
                StmtNode::Test {
                    operand,
                    negate,
                    test: test_id,
                },
            );
        }
        Ok(operand)
    }

    fn parse_filter_expression(&mut self) -> JResult<Stmt> {
        let mut operand = self.parse_unary_expression()?;
        while self.is(TokenType::Pipe) {
            let start_pos = self.current;
            self.current += 1;
            let mut filter = self.parse_primary_expression()?;
            if self.is(TokenType::OpenParen) {
                filter = self.parse_call_expression(filter)?;
            }
            operand = self.stmt(start_pos, StmtNode::Filter { operand, filter });
        }
        Ok(operand)
    }

    fn parse_unary_expression(&mut self) -> JResult<Stmt> {
        if self.is(TokenType::AdditiveBinaryOperator) && self.peek(0).value != "~" {
            let _guard = DepthGuard::new(self)?;
            let start_pos = self.current;
            let op = self.next()?;
            let arg = self.parse_unary_expression()?;
            return Ok(self.stmt(start_pos, StmtNode::Unary { op, argument: arg }));
        }
        self.parse_call_member_expression()
    }

    fn parse_call_member_expression(&mut self) -> JResult<Stmt> {
        let primary = self.parse_primary_expression()?;
        let member = self.parse_member_expression(primary)?;
        if self.is(TokenType::OpenParen) {
            self.parse_call_expression(member)
        } else {
            Ok(member)
        }
    }

    fn parse_call_expression(&mut self, callee: Stmt) -> JResult<Stmt> {
        let start_pos = self.current;
        let args = self.parse_args(true)?;
        let expr = self.stmt(start_pos, StmtNode::CallExpression { callee, args });
        let member = self.parse_member_expression(expr)?;
        if !self.is(TokenType::OpenParen) {
            return Ok(member);
        }
        let _guard = DepthGuard::new(self)?;
        self.parse_call_expression(member)
    }

    fn parse_args(&mut self, allow_spread: bool) -> JResult<Vec<Stmt>> {
        self.expect(TokenType::OpenParen, "Expected (")?;
        let mut args = Vec::new();
        while !self.is(TokenType::CloseParen) {
            let t = self.peek(0);
            let arg = if t.t == TokenType::PowerOperator
                || (t.t == TokenType::MultiplicativeBinaryOperator && t.value == "*")
            {
                if !allow_spread {
                    return Err(JinjaError::Error(format!(
                        "Unexpected token: {} of type {}",
                        t.value,
                        t.t.as_str()
                    )));
                }
                let start_pos = self.current;
                self.current += 1;
                let is_kwargs = t.t == TokenType::PowerOperator;
                let expression = self.parse_expression()?;
                self.stmt(
                    start_pos,
                    StmtNode::Spread {
                        argument: expression,
                        is_kwargs,
                    },
                )
            } else {
                let expression = self.parse_expression()?;
                if self.is(TokenType::Equals) {
                    let start_pos = self.current;
                    self.current += 1;
                    let val = self.parse_expression()?;
                    self.stmt(start_pos, StmtNode::KeywordArgument { key: expression, val })
                } else {
                    expression
                }
            };
            args.push(arg.clone());
            if self.is(TokenType::Comma) {
                self.current += 1;
            }
        }
        self.expect(TokenType::CloseParen, "Expected )")?;
        Ok(args)
    }

    fn parse_member_expression(&mut self, mut object: Stmt) -> JResult<Stmt> {
        let start_pos = self.current;
        while self.is(TokenType::Dot) || self.is(TokenType::OpenSquareBracket) {
            let op = self.next()?;
            let computed = op.t == TokenType::OpenSquareBracket;
            let prop = if computed {
                let p = self.parse_member_expression_arguments()?;
                self.expect(TokenType::CloseSquareBracket, "Expected ]")?;
                p
            } else {
                self.parse_primary_expression()?
            };
            let static_property = if !computed {
                if let StmtNode::Identifier(name) = &prop.node {
                    Some(super::value::mk_string(name.clone()))
                } else {
                    None
                }
            } else {
                None
            };
            object = self.stmt(
                start_pos,
                StmtNode::MemberExpression {
                    object,
                    property: Some(prop),
                    computed,
                    static_property,
                },
            );
        }
        Ok(object)
    }

    fn parse_member_expression_arguments(&mut self) -> JResult<Stmt> {
        let mut slices: Vec<Option<Stmt>> = Vec::new();
        let mut is_slice = false;
        let start_pos = self.current;
        while !self.is(TokenType::CloseSquareBracket) {
            if self.is(TokenType::Colon) {
                slices.push(None);
                self.current += 1;
                is_slice = true;
            } else {
                slices.push(Some(self.parse_expression()?));
                if self.is(TokenType::Colon) {
                    self.current += 1;
                    is_slice = true;
                }
            }
        }
        if is_slice {
            let mut take = |i: usize| -> Option<Stmt> {
                if slices.len() > i {
                    slices[i].take()
                } else {
                    None
                }
            };
            return Ok(self.stmt(
                start_pos,
                StmtNode::Slice {
                    start: take(0),
                    stop: take(1),
                    step: take(2),
                },
            ));
        }
        if slices.is_empty() {
            return Ok(self.stmt(start_pos, StmtNode::BlankExpression));
        }
        Ok(slices.remove(0).unwrap_or_else(|| self.stmt(start_pos, StmtNode::BlankExpression)))
    }

    fn parse_primary_expression(&mut self) -> JResult<Stmt> {
        let start_pos = self.current;
        let t = self.next()?;
        match t.t {
            TokenType::NumericLiteral => {
                if t.value.contains('.') {
                    let v = t.value.parse::<f64>().map_err(|_| {
                        ParserException::new("Invalid float literal", &self.source, t.pos)
                    })?;
                    Ok(self.stmt(start_pos, StmtNode::FloatLiteral(v)))
                } else {
                    let v = t.value.parse::<i64>().map_err(|_| {
                        ParserException::new("Invalid integer literal", &self.source, t.pos)
                    })?;
                    Ok(self.stmt(start_pos, StmtNode::IntegerLiteral(v)))
                }
            }
            TokenType::StringLiteral => {
                let mut val = t.value;
                while self.is(TokenType::StringLiteral) {
                    val.push_str(&self.next()?.value);
                }
                Ok(self.stmt(start_pos, StmtNode::StringLiteral(val)))
            }
            TokenType::Identifier => Ok(self.stmt(start_pos, StmtNode::Identifier(t.value))),
            TokenType::OpenParen => {
                let expr = self.parse_expression_sequence(false)?;
                self.expect(TokenType::CloseParen, "Expected )")?;
                Ok(expr)
            }
            TokenType::OpenSquareBracket => {
                let mut vals = Vec::new();
                while !self.is(TokenType::CloseSquareBracket) {
                    vals.push(self.parse_expression()?);
                    if self.is(TokenType::Comma) {
                        self.current += 1;
                    }
                }
                self.current += 1;
                Ok(self.stmt(start_pos, StmtNode::ArrayLiteral(vals)))
            }
            TokenType::OpenCurlyBracket => {
                let mut pairs = Vec::new();
                while !self.is(TokenType::CloseCurlyBracket) {
                    let key = self.parse_expression()?;
                    self.expect(TokenType::Colon, "Expected :")?;
                    let val = self.parse_expression()?;
                    pairs.push((key, val));
                    if self.is(TokenType::Comma) {
                        self.current += 1;
                    }
                }
                self.current += 1;
                Ok(self.stmt(start_pos, StmtNode::ObjectLiteral(pairs)))
            }
            _ => Err(ParserException::new(
                &format!(
                    "Unexpected token: {} of type {}",
                    t.value,
                    t.t.as_str()
                ),
                &self.source,
                t.pos,
            )),
        }
    }
}

/// `parse_from_tokens`.
pub fn parse_from_tokens(lexer_res: &lexer_result) -> JResult<Program> {
    Parser::new(&lexer_res.tokens, &lexer_res.source).parse()
}

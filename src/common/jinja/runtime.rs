//! `jinja::runtime`: the context, the AST statement types and their execution.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::lexer::token;
use super::string::JString;
use super::utils::{
    enable_debug_flag, peak_source, string_replace_all, JResult, JinjaError,
    JINJA_RECURSIVE_LIMIT,
};
use super::value::*;

/// `enable_debug` / `g_jinja_debug`.
pub fn enable_debug(enable: bool) {
    enable_debug_flag(enable);
}

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct StmtData {
    pub pos: usize,
    pub node: StmtNode,
}

#[derive(Clone)]
pub struct Stmt {
    inner: Rc<StmtData>,
}

impl std::ops::Deref for Stmt {
    type Target = StmtData;

    fn deref(&self) -> &StmtData {
        &self.inner
    }
}

#[derive(Clone, Debug)]
pub enum StmtNode {
    Program { body: Vec<Stmt> },
    If { test: Stmt, body: Vec<Stmt>, alternate: Vec<Stmt> },
    For { loopvar: Stmt, iterable: Stmt, body: Vec<Stmt>, default_block: Vec<Stmt> },
    Break,
    Continue,
    Noop,
    Set { assignee: Stmt, val: Option<Stmt>, body: Vec<Stmt> },
    Macro { name: Stmt, args: Vec<Stmt>, body: Vec<Stmt> },
    Comment(String),
    BlankExpression,
    MemberExpression {
        object: Stmt,
        property: Option<Stmt>,
        computed: bool,
        static_property: Option<Value>,
    },
    CallExpression { callee: Stmt, args: Vec<Stmt> },
    Identifier(String),
    IntegerLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    ArrayLiteral(Vec<Stmt>),
    TupleLiteral(Vec<Stmt>),
    ObjectLiteral(Vec<(Stmt, Stmt)>),
    Binary { op: token, left: Stmt, right: Stmt },
    ComparisonChain { operands: Vec<Stmt>, ops: Vec<token> },
    Filter { operand: Stmt, filter: Stmt },
    FilterStatement { filter: Stmt, body: Vec<Stmt> },
    Select { lhs: Stmt, test: Stmt },
    Test { operand: Stmt, negate: bool, test: Stmt },
    Unary { op: token, argument: Stmt },
    Slice { start: Option<Stmt>, stop: Option<Stmt>, step: Option<Stmt> },
    KeywordArgument { key: Stmt, val: Stmt },
    Spread { argument: Stmt, is_kwargs: bool },
    CallStatement { call: Stmt, caller_args: Vec<Stmt>, body: Vec<Stmt> },
    Ternary { condition: Stmt, true_expr: Stmt, false_expr: Stmt },
}

impl Stmt {
    pub fn new(pos: usize, node: StmtNode) -> Self {
        Self {
            inner: Rc::new(StmtData { pos, node }),
        }
    }
}

pub type Program = Stmt;

impl std::fmt::Debug for Stmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stmt")
            .field("pos", &self.pos)
            .field("node", &self.node)
            .finish()
    }
}

pub fn stmt_type(stmt: &Stmt) -> &'static str {
    match &stmt.node {
        StmtNode::Program { .. } => "Program",
        StmtNode::If { .. } => "If",
        StmtNode::For { .. } => "For",
        StmtNode::Break => "Break",
        StmtNode::Continue => "Continue",
        StmtNode::Noop => "Noop",
        StmtNode::Set { .. } => "Set",
        StmtNode::Macro { .. } => "Macro",
        StmtNode::Comment(_) => "Comment",
        StmtNode::BlankExpression => "BlankExpression",
        StmtNode::MemberExpression { .. } => "MemberExpression",
        StmtNode::CallExpression { .. } => "CallExpression",
        StmtNode::Identifier(_) => "Identifier",
        StmtNode::IntegerLiteral(_) => "IntegerLiteral",
        StmtNode::FloatLiteral(_) => "FloatLiteral",
        StmtNode::StringLiteral(_) => "StringLiteral",
        StmtNode::ArrayLiteral(_) => "ArrayLiteral",
        StmtNode::TupleLiteral(_) => "TupleLiteral",
        StmtNode::ObjectLiteral(_) => "ObjectLiteral",
        StmtNode::Binary { .. } => "BinaryExpression",
        StmtNode::ComparisonChain { .. } => "ComparisonChainExpression",
        StmtNode::Filter { .. } => "FilterExpression",
        StmtNode::FilterStatement { .. } => "FilterStatement",
        StmtNode::Select { .. } => "SelectExpression",
        StmtNode::Test { .. } => "TestExpression",
        StmtNode::Unary { .. } => "UnaryExpression",
        StmtNode::Slice { .. } => "SliceExpression",
        StmtNode::KeywordArgument { .. } => "KeywordArgumentExpression",
        StmtNode::Spread { .. } => "SpreadExpression",
        StmtNode::CallStatement { .. } => "CallStatement",
        StmtNode::Ternary { .. } => "Ternary",
    }
}

// ---------------------------------------------------------------------------
// context
// ---------------------------------------------------------------------------

pub struct Context {
    pub src: Rc<String>,
    pub current_time: i64,
    pub is_get_stats: bool,
    pub macro_depth: i32,
    /// Innermost scope last. Lookup walks backwards, writes go to the last.
    envs: Vec<Value>,
}

impl Context {
    pub fn new(src: String) -> Self {
        let env = mk_object();
        if let ValueData::Object { has_builtins, .. } = &mut env.borrow_mut().data {
            *has_builtins = false;
        }
        insert_str(&env, "true", mk_bool(true)).ok();
        insert_str(&env, "True", mk_bool(true)).ok();
        insert_str(&env, "false", mk_bool(false)).ok();
        insert_str(&env, "False", mk_bool(false)).ok();
        insert_str(&env, "none", mk_none()).ok();
        insert_str(&env, "None", mk_none()).ok();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Self {
            src: Rc::new(src),
            current_time: now,
            is_get_stats: false,
            macro_depth: 0,
            envs: vec![env],
        }
    }

    /// `context(const context & parent)`.
    pub fn child(&self) -> Self {
        let env = mk_object();
        if let ValueData::Object { has_builtins, .. } = &mut env.borrow_mut().data {
            *has_builtins = false;
        }
        let mut envs = self.envs.clone();
        envs.push(env);
        Self {
            src: self.src.clone(),
            current_time: self.current_time,
            is_get_stats: self.is_get_stats,
            macro_depth: self.macro_depth,
            envs,
        }
    }

    /// Flatten all visible variables into a single scope.
    pub fn snapshot(&self) -> Self {
        let env = mk_object();
        if let ValueData::Object { has_builtins, .. } = &mut env.borrow_mut().data {
            *has_builtins = false;
        }
        for scope in &self.envs {
            for (k, v) in as_ordered_object(scope).unwrap_or_default() {
                insert(&env, k, v).ok();
            }
        }
        Self {
            src: self.src.clone(),
            current_time: self.current_time,
            is_get_stats: self.is_get_stats,
            macro_depth: self.macro_depth,
            envs: vec![env],
        }
    }

    pub fn find_val(&self, key: &Value) -> Option<Value> {
        for scope in self.envs.iter().rev() {
            if let Some(val) = find(scope, key) {
                return Some(val);
            }
        }
        None
    }

    pub fn get_val(&self, name: &str) -> Value {
        self.find_val(&mk_string(name))
            .unwrap_or_else(|| mk_undefined(name))
    }

    pub fn set_val(&self, name: &str, val: Value) -> JResult<()> {
        if let Some(env) = self.envs.last() {
            insert_str(env, name, val)
        } else {
            Ok(())
        }
    }

    pub fn set_val_key(&self, key: Value, val: Value) -> JResult<()> {
        if let Some(env) = self.envs.last() {
            insert(env, key, val)
        } else {
            Ok(())
        }
    }

    pub fn print_vars(&self) -> String {
        if let Some(env) = self.envs.first() {
            value_to_json(env, 2, ", ", ": ", false)
        } else {
            String::new()
        }
    }
}

fn cached_val(ctx: &Context, cached: &Value) -> Value {
    if ctx.is_get_stats {
        // The C++ engine clones the cached constant; the clone must not share
        // the original's stats. Values here are immutable constants, so a
        // shallow Rc clone would leak stats across renders. Copy by kind.
        match &cached.borrow().data {
            ValueData::Integer { val_int, val_flt } => {
                let v = mk_int(*val_int);
                if let ValueData::Integer { val_flt: f, .. } = &mut v.borrow_mut().data {
                    *f = *val_flt;
                }
                v
            }
            ValueData::Real { val_int, val_flt } => {
                let v = mk_float(*val_flt);
                if let ValueData::Real { val_int: i, .. } = &mut v.borrow_mut().data {
                    *i = *val_int;
                }
                v
            }
            ValueData::String { val_str } => mk_jstring(val_str.clone()),
            _ => cached.clone(),
        }
    } else {
        cached.clone()
    }
}

// ---------------------------------------------------------------------------
// execution wrapper and helpers
// ---------------------------------------------------------------------------

fn get_line_col(source: &str, pos: usize) -> String {
    let mut line = 1usize;
    let mut col = 1usize;
    let bytes = source.as_bytes();
    for i in 0..pos.min(source.len()) {
        if bytes[i] == b'\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    format!("line {line}, column {col}")
}

fn format_exec_error(stmt: &Stmt, ctx: &Context, msg: &str) -> String {
    let source = ctx.src.as_str();
    if source.is_empty() {
        format!(
            "\nError executing {} at position {}: {}",
            stmt_type(stmt),
            stmt.pos,
            msg
        )
    } else {
        format!(
            "\n------------\nWhile executing {} at {} in source:\n{}\nError: {}",
            stmt_type(stmt),
            get_line_col(source, stmt.pos),
            peak_source(source, stmt.pos, 40),
            msg
        )
    }
}

impl Stmt {
    /// `statement::execute`: wrap generic exceptions with source context.
    pub fn execute(&self, ctx: &Context) -> JResult<Value> {
        match self.execute_impl(ctx) {
            Ok(v) => Ok(v),
            Err(e @ (JinjaError::Break
                | JinjaError::Continue
                | JinjaError::NotImplemented(_)
                | JinjaError::Rethrown(_))) => Err(e),
            Err(JinjaError::Raised(msg)) | Err(JinjaError::Error(msg)) => {
                Err(JinjaError::Rethrown(format_exec_error(self, ctx, &msg)))
            }
        }
    }

    fn execute_impl(&self, ctx: &Context) -> JResult<Value> {
        match &self.node {
            StmtNode::Program { .. } => Err(JinjaError::Error(
                "Cannot execute program directly, use jinja::runtime instead".into(),
            )),
            StmtNode::If { test, body, alternate } => {
                let cond = as_bool(&test.execute(ctx)?)?;
                let out = mk_array();
                let stmts = if cond { body } else { alternate };
                for stmt in stmts {
                    push_array(&out, stmt.execute(ctx)?);
                }
                let parts = mk_empty_string();
                gather_string_parts_recursive(&out, &parts, 0);
                Ok(parts)
            }
            StmtNode::For { loopvar, iterable, body, default_block } => {
                execute_for(loopvar, iterable, body, default_block, ctx)
            }
            StmtNode::Break => Err(JinjaError::Break),
            StmtNode::Continue => Err(JinjaError::Continue),
            StmtNode::Noop | StmtNode::Comment(_) | StmtNode::BlankExpression => {
                Ok(mk_undefined(""))
            }
            StmtNode::Set { assignee, val, body } => {
                let rhs = match val {
                    Some(v) => v.execute(ctx)?,
                    None => exec_statements(body, ctx)?,
                };
                match &assignee.node {
                    StmtNode::Identifier(name) => ctx.set_val(name, rhs)?,
                    StmtNode::TupleLiteral(targets) => {
                        if !is_val(&rhs, ValueKind::Array) {
                            return Err(JinjaError::Error(format!(
                                "Cannot unpack non-iterable type in set: {}",
                                type_name(&rhs)
                            )));
                        }
                        let arr = as_array(&rhs)?;
                        if arr.len() != targets.len() {
                            return Err(JinjaError::Error(format!(
                                "Too {} items to unpack in set",
                                if targets.len() > arr.len() { "few" } else { "many" }
                            )));
                        }
                        for (elem, item) in targets.iter().zip(arr) {
                            let StmtNode::Identifier(name) = &elem.node else {
                                return Err(JinjaError::Error(format!(
                                    "Cannot unpack to non-identifier in set: {}",
                                    stmt_type(elem)
                                )));
                            };
                            ctx.set_val(name, item)?;
                        }
                    }
                    StmtNode::MemberExpression { object, property, computed, static_property } => {
                        if *computed {
                            return Err(JinjaError::Error("Cannot assign to computed member".into()));
                        }
                        let Some(prop) = property else {
                            return Err(JinjaError::Error(
                                "Cannot assign to member with non-identifier property".into(),
                            ));
                        };
                        let StmtNode::Identifier(prop_name) = &prop.node else {
                            return Err(JinjaError::Error(
                                "Cannot assign to member with non-identifier property".into(),
                            ));
                        };
                        let obj = object.execute(ctx)?;
                        if kind_of(&obj) != ValueKind::Object {
                            return Err(JinjaError::Error("Cannot assign to member of non-object".into()));
                        }
                        let key = static_property
                            .clone()
                            .unwrap_or_else(|| mk_string(prop_name.clone()));
                        insert(&obj, cached_val(ctx, &key), rhs)?;
                    }
                    other => {
                        return Err(JinjaError::Error(format!(
                            "Invalid LHS inside assignment expression: {}",
                            stmt_node_type(other)
                        )));
                    }
                }
                Ok(mk_undefined(""))
            }
            StmtNode::Macro { name, args, body } => {
                let StmtNode::Identifier(name_str) = &name.node else {
                    return Err(JinjaError::Error("Macro name must be an identifier".into()));
                };
                let name_str = name_str.clone();
                let name_for_val = name_str.clone();
                let name_for_closure = name_str.clone();
                let args = args.clone();
                let body = body.clone();
                let handler: FuncHandler = Rc::new(move |a: &FuncArgs| -> JResult<Value> {
                    if a.ctx.macro_depth as usize >= JINJA_RECURSIVE_LIMIT {
                        return Err(JinjaError::Error(format!(
                            "macro calls are nested more than {} levels deep",
                            JINJA_RECURSIVE_LIMIT
                        )));
                    }
                    let mut macro_ctx = a.ctx.child();
                    macro_ctx.macro_depth = a.ctx.macro_depth + 1;
                    bind_parameters(&name_for_closure, &args, a, &macro_ctx)?;
                    exec_statements(&body, &macro_ctx)
                });
                ctx.set_val(&name_for_val, mk_func(&name_str, handler, None))?;
                Ok(mk_undefined(""))
            }
            StmtNode::Identifier(name) => {
                let key = mk_string(name.clone());
                if let Some(val) = ctx.find_val(&key) {
                    if !is_undefined(&val) {
                        if ctx.is_get_stats {
                            mark_used(&val, false, 0);
                        }
                        return Ok(val);
                    }
                }
                if let Some(func) = global_builtin(name) {
                    return Ok(mk_func(name, func, None));
                }
                Ok(mk_undefined(name))
            }
            StmtNode::IntegerLiteral(v) => Ok(cached_val(ctx, &mk_int(*v))),
            StmtNode::FloatLiteral(v) => Ok(cached_val(ctx, &mk_float(*v))),
            StmtNode::StringLiteral(v) => Ok(cached_val(ctx, &mk_string(v.clone()))),
            StmtNode::ArrayLiteral(items) => {
                let arr = mk_array();
                for item in items {
                    push_array(&arr, item.execute(ctx)?);
                }
                Ok(arr)
            }
            StmtNode::TupleLiteral(items) => {
                let mut vals = Vec::new();
                for item in items {
                    vals.push(item.execute(ctx)?);
                }
                Ok(mk_tuple(vals))
            }
            StmtNode::ObjectLiteral(pairs) => {
                let obj = mk_object();
                for (k, v) in pairs {
                    let key = k.execute(ctx)?;
                    let val = v.execute(ctx)?;
                    insert(&obj, key, val)?;
                }
                Ok(obj)
            }
            StmtNode::Binary { op, left, right } => {
                let left_val = left.execute(ctx)?;
                if op.value == "and" {
                    return if as_bool(&left_val)? {
                        right.execute(ctx)
                    } else {
                        Ok(left_val)
                    };
                }
                if op.value == "or" {
                    return if as_bool(&left_val)? {
                        Ok(left_val)
                    } else {
                        right.execute(ctx)
                    };
                }
                exec_binary_op(ctx, op, left_val, right.execute(ctx)?)
            }
            StmtNode::ComparisonChain { operands, ops } => {
                let mut left = operands[0].execute(ctx)?;
                let mut res = mk_undefined("");
                for (i, op) in ops.iter().enumerate() {
                    let right = operands[i + 1].execute(ctx)?;
                    res = exec_binary_op(ctx, op, left.clone(), right.clone())?;
                    if !as_bool(&res)? {
                        break;
                    }
                    left = right;
                }
                Ok(res)
            }
            StmtNode::Filter { operand, filter } => {
                apply_filter(ctx, filter, operand.execute(ctx)?)
            }
            StmtNode::FilterStatement { filter, body } => {
                let body_val = exec_statements(body, ctx)?;
                let parts = mk_empty_string();
                gather_string_parts_recursive(&body_val, &parts, 0);
                apply_filter(ctx, filter, parts)
            }
            StmtNode::Select { lhs, test } => {
                let predicate = test.execute_impl(ctx)?;
                if !as_bool(&predicate)? {
                    return Ok(mk_undefined(""));
                }
                lhs.execute_impl(ctx)
            }
            StmtNode::Test { operand, negate, test } => execute_test(operand, *negate, test, ctx),
            StmtNode::Unary { op, argument } => {
                let val = argument.execute(ctx)?;
                match op.value.as_str() {
                    "not" => Ok(mk_bool(!as_bool(&val)?)),
                    "-" => {
                        if kind_of(&val) == ValueKind::Integer {
                            Ok(mk_int(-as_int(&val)?))
                        } else if kind_of(&val) == ValueKind::Real {
                            Ok(mk_float(-as_float(&val)?))
                        } else {
                            Err(JinjaError::Error(
                                "Unary - operator requires numeric operand".into(),
                            ))
                        }
                    }
                    "+" => {
                        if is_val(&val, ValueKind::Integer) || is_val(&val, ValueKind::Real) {
                            Ok(val)
                        } else {
                            Err(JinjaError::Error(
                                "Unary + operator requires numeric operand".into(),
                            ))
                        }
                    }
                    other => Err(JinjaError::Error(format!(
                        "Unknown unary operator '{other}'"
                    ))),
                }
            }
            StmtNode::Slice { .. } => Err(JinjaError::Error(
                "must be handled by MemberExpression".into(),
            )),
            StmtNode::KeywordArgument { key, val } => {
                let StmtNode::Identifier(k) = &key.node else {
                    return Err(JinjaError::Error(
                        "Keyword argument key must be identifiers".into(),
                    ));
                };
                Ok(mk_kwarg(k.clone(), val.execute(ctx)?))
            }
            StmtNode::Spread { .. } => Err(JinjaError::Error(
                "cannot exec SpreadExpression".into(),
            )),
            StmtNode::CallStatement { call, caller_args, body } => {
                execute_call_statement(call, caller_args, body, ctx)
            }
            StmtNode::Ternary { condition, true_expr, false_expr } => {
                if as_bool(&condition.execute(ctx)?)? {
                    true_expr.execute(ctx)
                } else {
                    false_expr.execute(ctx)
                }
            }
            StmtNode::MemberExpression { object, property, computed, static_property } => {
                execute_member(object, property.as_ref(), *computed, static_property, ctx)
            }
            StmtNode::CallExpression { callee, args } => {
                let mut func_args = FuncArgs::new(ctx);
                exec_call_args(ctx, args, &mut func_args)?;
                let callee_val = callee.execute(ctx)?;
                if kind_of(&callee_val) != ValueKind::Function {
                    return Err(JinjaError::Error(format!(
                        "Callee is not a function: got {}",
                        type_name(&callee_val)
                    )));
                }
                invoke(&callee_val, &func_args)
            }
        }
    }
}

fn stmt_node_type(node: &StmtNode) -> &'static str {
    // Only used in an error path; map the node to its statement type name.
    let dummy = Stmt::new(0, node.clone());
    stmt_type(&dummy)
}

fn exec_statements(stmts: &[Stmt], ctx: &Context) -> JResult<Value> {
    let result = mk_array();
    for stmt in stmts {
        push_array(&result, stmt.execute(ctx)?);
    }
    let parts = mk_empty_string();
    gather_string_parts_recursive(&result, &parts, 0);
    Ok(parts)
}

fn test_is_in(ctx: &Context, left: Value, right: Value) -> JResult<bool> {
    let mut args = FuncArgs::new(ctx);
    args.push_back(left);
    args.push_back(right);
    let handler = global_builtin("test_is_in").unwrap();
    as_bool(&handler(&args)?)
}

fn exec_binary_op(ctx: &Context, op: &token, left_val: Value, right_val: Value) -> JResult<Value> {
    if op.value == "==" {
        return Ok(mk_bool(equivalent(&left_val, &right_val)));
    }
    if op.value == "!=" {
        return Ok(mk_bool(nonequal(&left_val, &right_val)));
    }

    let concat_null_with_str = |res: &mut Value| -> bool {
        let is_left_null = is_none(&left_val) || is_undefined(&left_val);
        let is_right_null = is_none(&right_val) || is_undefined(&right_val);
        let is_left_str = kind_of(&left_val) == ValueKind::String;
        let is_right_str = kind_of(&right_val) == ValueKind::String;
        if (is_left_null && is_right_str) || (is_right_null && is_left_str) {
            let left_str = if is_left_null {
                JString::default()
            } else {
                as_string(&left_val).unwrap_or_default()
            };
            let right_str = if is_right_null {
                JString::default()
            } else {
                as_string(&right_val).unwrap_or_default()
            };
            let mut output = left_str;
            output.append(&right_str);
            *res = mk_jstring(output);
            return true;
        }
        false
    };

    if is_undefined(&left_val) || is_undefined(&right_val) {
        if is_undefined(&right_val) && (op.value == "in" || op.value == "not in") {
            return Ok(mk_bool(op.value == "not in"));
        }
        if op.value == "+" || op.value == "~" {
            let mut res = mk_undefined("");
            if concat_null_with_str(&mut res) {
                return Ok(res);
            }
        }
        return Err(JinjaError::Error(format!(
            "Cannot perform operation {} on undefined values",
            op.value
        )));
    }
    if is_none(&left_val) || is_none(&right_val) {
        if !is_none(&right_val) && (op.value == "in" || op.value == "not in") {
            let member = test_is_in(ctx, left_val.clone(), right_val.clone())?;
            return Ok(mk_bool(if op.value == "in" { member } else { !member }));
        }
        if op.value == "+" || op.value == "~" {
            let mut res = mk_undefined("");
            if concat_null_with_str(&mut res) {
                return Ok(res);
            }
        }
        return Err(JinjaError::Error(
            "Cannot perform operation on null values".into(),
        ));
    }

    if (is_val(&left_val, ValueKind::Integer) || is_val(&left_val, ValueKind::Real))
        && (is_val(&right_val, ValueKind::Integer) || is_val(&right_val, ValueKind::Real))
    {
        let a = as_float(&left_val)?;
        let b = as_float(&right_val)?;
        match op.value.as_str() {
            "+" | "-" | "*" => {
                let res = if op.value == "+" {
                    a + b
                } else if op.value == "-" {
                    a - b
                } else {
                    a * b
                };
                let is_float = kind_of(&left_val) == ValueKind::Real
                    || kind_of(&right_val) == ValueKind::Real;
                return Ok(if is_float {
                    mk_float(res)
                } else {
                    mk_int(float_to_int(res))
                });
            }
            "/" => return Ok(mk_float(a / b)),
            "//" => {
                if kind_of(&left_val) == ValueKind::Integer && kind_of(&right_val) == ValueKind::Integer {
                    let x = as_int(&left_val)?;
                    let y = as_int(&right_val)?;
                    if y == 0 {
                        return Err(JinjaError::Error("integer division by zero".into()));
                    }
                    if y == -1 {
                        return Ok(mk_int(x.wrapping_neg()));
                    }
                    let mut q = x / y;
                    if x % y != 0 && (x < 0) != (y < 0) {
                        q -= 1;
                    }
                    return Ok(mk_int(q));
                }
                return Ok(mk_float((a / b).floor()));
            }
            "%" => {
                let rem = a % b;
                let is_float = kind_of(&left_val) == ValueKind::Real
                    || kind_of(&right_val) == ValueKind::Real;
                return Ok(if is_float {
                    mk_float(rem)
                } else {
                    mk_int(float_to_int(rem))
                });
            }
            "<" => return Ok(mk_bool(a < b)),
            ">" => return Ok(mk_bool(a > b)),
            ">=" => return Ok(mk_bool(a >= b)),
            "<=" => return Ok(mk_bool(a <= b)),
            _ => {}
        }
    }

    if is_val(&left_val, ValueKind::Array) && is_val(&right_val, ValueKind::Array) {
        if op.value == "+" {
            let result = mk_array();
            for item in as_array(&left_val)? {
                push_array(&result, item);
            }
            for item in as_array(&right_val)? {
                push_array(&result, item);
            }
            return Ok(result);
        }
    } else if is_val(&right_val, ValueKind::Array) {
        let member = test_is_in(ctx, left_val.clone(), right_val.clone())?;
        if op.value == "in" {
            return Ok(mk_bool(member));
        }
        if op.value == "not in" {
            return Ok(mk_bool(!member));
        }
    }

    if (kind_of(&left_val) == ValueKind::String || kind_of(&right_val) == ValueKind::String)
        && (op.value == "~" || op.value == "+")
    {
        let mut output = as_string(&left_val)?;
        output.append(&as_string(&right_val)?);
        return Ok(mk_jstring(output));
    }

    if op.value == "*"
        && ((kind_of(&left_val) == ValueKind::String && kind_of(&right_val) == ValueKind::Integer)
            || (kind_of(&right_val) == ValueKind::String && kind_of(&left_val) == ValueKind::Integer))
    {
        let s = if kind_of(&left_val) == ValueKind::String {
            as_string(&left_val)?
        } else {
            as_string(&right_val)?
        };
        let repeat = if kind_of(&right_val) == ValueKind::Integer {
            as_int(&right_val)?
        } else {
            as_int(&left_val)?
        };
        let res = mk_empty_string();
        if repeat <= 0 {
            return Ok(res);
        }
        for _ in 0..repeat {
            if let ValueData::String { val_str } = &mut res.borrow_mut().data {
                val_str.append(&s);
            }
        }
        return Ok(res);
    }

    if kind_of(&left_val) == ValueKind::String && kind_of(&right_val) == ValueKind::String {
        if op.value == "in" || op.value == "not in" {
            let member = test_is_in(ctx, left_val.clone(), right_val.clone())?;
            return Ok(mk_bool(if op.value == "in" { member } else { !member }));
        }
        let a = as_string(&left_val)?.str();
        let b = as_string(&right_val)?.str();
        match op.value.as_str() {
            "<" => return Ok(mk_bool(a < b)),
            ">" => return Ok(mk_bool(a > b)),
            "<=" => return Ok(mk_bool(a <= b)),
            ">=" => return Ok(mk_bool(a >= b)),
            _ => {}
        }
    }

    if kind_of(&right_val) == ValueKind::Object {
        let member = test_is_in(ctx, left_val.clone(), right_val.clone())?;
        if op.value == "in" {
            return Ok(mk_bool(member));
        }
        if op.value == "not in" {
            return Ok(mk_bool(!member));
        }
    }

    Err(JinjaError::Error(format!(
        "Unknown operator \"{}\" between {} and {}",
        op.value,
        type_name(&left_val),
        type_name(&right_val)
    )))
}

fn exec_call_args(ctx: &Context, arg_stmts: &[Stmt], args: &mut FuncArgs) -> JResult<()> {
    for arg_stmt in arg_stmts {
        if let StmtNode::Spread { argument, is_kwargs } = &arg_stmt.node {
            let val = argument.execute(ctx)?;
            if *is_kwargs {
                if kind_of(&val) != ValueKind::Object {
                    return Err(JinjaError::Error(format!(
                        "Argument after ** must be an object, got {}",
                        type_name(&val)
                    )));
                }
                for (k, v) in as_ordered_object(&val)? {
                    if kind_of(&k) != ValueKind::String {
                        return Err(JinjaError::Error(format!(
                            "Keyword argument key must be a string, got {}",
                            type_name(&k)
                        )));
                    }
                    args.push_back(mk_kwarg(as_string(&k)?.str(), v));
                }
            } else {
                if !is_val(&val, ValueKind::Array) {
                    return Err(JinjaError::Error(format!(
                        "Argument after * must be an array, got {}",
                        type_name(&val)
                    )));
                }
                for item in as_array(&val)? {
                    args.push_back(item);
                }
            }
        } else {
            args.push_back(arg_stmt.execute(ctx)?);
        }
    }
    Ok(())
}

fn apply_filter(ctx: &Context, filter: &Stmt, input: Value) -> JResult<Value> {
    match &filter.node {
        StmtNode::Identifier(name) => {
            apply_filter_by_name(ctx, name.clone(), input, &FuncArgs::new(ctx))
        }
        StmtNode::CallExpression { callee, args } => {
            let StmtNode::Identifier(name) = &callee.node else {
                return Err(JinjaError::Error("Filter callee must be an identifier".into()));
            };
            let mut func_args = FuncArgs::new(ctx);
            exec_call_args(ctx, args, &mut func_args)?;
            apply_filter_by_name(ctx, name.clone(), input, &func_args)
        }
        _ => Err(JinjaError::Error("Invalid filter expression".into())),
    }
}

fn execute_test(operand: &Stmt, negate: bool, test: &Stmt, ctx: &Context) -> JResult<Value> {
    let input = operand.execute(ctx)?;
    let mut args = FuncArgs::new(ctx);
    args.push_back(input.clone());

    let test_id = match &test.node {
        StmtNode::Identifier(name) => name.clone(),
        StmtNode::CallExpression { callee, args: call_args } => {
            let StmtNode::Identifier(name) = &callee.node else {
                return Err(JinjaError::Error("Test callee must be an identifier".into()));
            };
            exec_call_args(ctx, call_args, &mut args)?;
            name.clone()
        }
        _ => return Err(JinjaError::Error("Invalid test expression".into())),
    };
    let test_name = format!("test_is_{test_id}");
    let Some(handler) = global_builtin(&test_name) else {
        return Err(JinjaError::Error(format!("Unknown test '{test_id}'")));
    };
    if ctx.is_get_stats {
        mark_used(&input, false, 0);
        input.borrow_mut().stats.ops.insert(test_name);
    }
    let res = handler(&args)?;
    Ok(if negate { mk_bool(!as_bool(&res)?) } else { res })
}

fn bind_parameters(name: &str, this_args: &[Stmt], args: &FuncArgs, ctx: &Context) -> JResult<()> {
    let input_count = args.count();
    for (i, param) in this_args.iter().enumerate() {
        if i < input_count {
            let (param_name, is_kwarg_default) = match &param.node {
                StmtNode::Identifier(n) => (n.clone(), false),
                StmtNode::KeywordArgument { key, .. } => {
                    let StmtNode::Identifier(n) = &key.node else {
                        return Err(JinjaError::Error(format!(
                            "Keyword argument key must be an identifier in '{name}'"
                        )));
                    };
                    (n.clone(), true)
                }
                _ => {
                    return Err(JinjaError::Error(format!(
                        "Invalid parameter type in '{name}'"
                    )));
                }
            };
            let _ = is_kwarg_default;
            let val = args.get_kwarg_or_pos(&param_name, i);
            ctx.set_val(&param_name, val)?;
        } else {
            let StmtNode::KeywordArgument { key, val: default } = &param.node else {
                return Err(JinjaError::Error(format!(
                    "Not enough arguments provided to '{name}'"
                )));
            };
            let StmtNode::Identifier(param_name) = &key.node else {
                return Err(JinjaError::Error(format!(
                    "Keyword argument key must be an identifier in '{name}'"
                )));
            };
            ctx.set_val(param_name, default.execute(args.ctx)?)?;
        }
    }
    Ok(())
}

fn execute_call_statement(call: &Stmt, caller_args: &[Stmt], body: &[Stmt], ctx: &Context) -> JResult<Value> {
    let StmtNode::CallExpression { callee, args } = &call.node else {
        return Err(JinjaError::Error("Call statement requires a valid call expression".into()));
    };
    let callee_val = callee.execute(ctx)?;
    if kind_of(&callee_val) != ValueKind::Function {
        return Err(JinjaError::Error(format!(
            "Callee is not a function: got {}",
            type_name(&callee_val)
        )));
    }

    let caller_ctx = Rc::new(RefCell::new(ctx.snapshot()));
    let caller_args = caller_args.to_vec();
    let body = body.to_vec();
    let handler: FuncHandler = Rc::new(move |a: &FuncArgs| -> JResult<Value> {
        if a.ctx.macro_depth as usize >= JINJA_RECURSIVE_LIMIT {
            return Err(JinjaError::Error(format!(
                "macro calls are nested more than {} levels deep",
                JINJA_RECURSIVE_LIMIT
            )));
        }
        let mut block_ctx = caller_ctx.borrow().child();
        block_ctx.macro_depth = a.ctx.macro_depth + 1;
        bind_parameters("caller", &caller_args, a, &block_ctx)?;
        exec_statements(&body, &block_ctx)
    });

    let call_ctx = ctx.child();
    call_ctx.set_val("caller", mk_func("caller", handler, None))?;
    let mut func_args = FuncArgs::new(&call_ctx);
    exec_call_args(ctx, args, &mut func_args)?;
    invoke(&callee_val, &func_args)
}

fn execute_for(
    loopvar: &Stmt,
    iterable: &Stmt,
    body: &[Stmt],
    default_block: &[Stmt],
    ctx: &Context,
) -> JResult<Value> {
    let scope = ctx.child();

    let (iter_expr, test_expr): (&Stmt, Option<&Stmt>) = match &iterable.node {
        StmtNode::Select { lhs, test } => (lhs, Some(test)),
        _ => (iterable, None),
    };

    let mut iterable_val = iter_expr.execute(&scope)?;
    if ctx.is_get_stats {
        mark_used(&iterable_val, false, 0);
        iterable_val.borrow_mut().stats.ops.insert("array_access".into());
    }
    if is_undefined(&iterable_val) {
        iterable_val = mk_array();
    }
    if !is_val(&iterable_val, ValueKind::Array) && kind_of(&iterable_val) != ValueKind::Object {
        return Err(JinjaError::Error(format!(
            "Expected iterable or object type in for loop: got {}",
            type_name(&iterable_val)
        )));
    }

    let mut items: Vec<Value> = Vec::new();
    let is_object = kind_of(&iterable_val) == ValueKind::Object;
    if is_object {
        for (k, v) in as_ordered_object(&iterable_val)? {
            items.push(mk_tuple(vec![k, v]));
        }
        if ctx.is_get_stats {
            mark_used(&iterable_val, false, 0);
            iterable_val.borrow_mut().stats.ops.insert("object_access".into());
        }
    } else {
        items.extend(as_array(&iterable_val)?);
        if ctx.is_get_stats {
            mark_used(&iterable_val, false, 0);
            iterable_val.borrow_mut().stats.ops.insert("array_access".into());
        }
    }

    let mut filtered: Vec<Value> = Vec::new();
    for item in &items {
        // For `for key in dict`, the loop sees the key, not the key/value tuple.
        let mut current = item.clone();
        if is_object && matches!(loopvar.node, StmtNode::Identifier(_)) {
            current = as_array(&current)?[0].clone();
        }
        if let Some(test) = test_expr {
            let loop_scope = scope.child();
            apply_loopvar(&loop_scope, loopvar, &current, is_object)?;
            if !as_bool(&test.execute(&loop_scope)?)? {
                continue;
            }
        }
        filtered.push(current);
    }

    let result = mk_array();
    let mut no_iteration = true;
    for (i, item) in filtered.iter().enumerate() {
        let loop_scope = scope.child();
        let loop_obj = mk_object();
        if let ValueData::Object { has_builtins, .. } = &mut loop_obj.borrow_mut().data {
            *has_builtins = false;
        }
        insert_str(&loop_obj, "index", mk_int(i as i64 + 1)).ok();
        insert_str(&loop_obj, "index0", mk_int(i as i64)).ok();
        insert_str(&loop_obj, "revindex", mk_int((filtered.len() - i) as i64)).ok();
        insert_str(&loop_obj, "revindex0", mk_int((filtered.len() - i - 1) as i64)).ok();
        insert_str(&loop_obj, "first", mk_bool(i == 0)).ok();
        insert_str(&loop_obj, "last", mk_bool(i + 1 == filtered.len())).ok();
        insert_str(&loop_obj, "length", mk_int(filtered.len() as i64)).ok();
        insert_str(
            &loop_obj,
            "previtem",
            if i > 0 {
                filtered[i - 1].clone()
            } else {
                mk_undefined("previtem")
            },
        )
        .ok();
        insert_str(
            &loop_obj,
            "nextitem",
            if i + 1 < filtered.len() {
                filtered[i + 1].clone()
            } else {
                mk_undefined("nextitem")
            },
        )
        .ok();
        loop_scope.set_val("loop", loop_obj)?;
        apply_loopvar(&loop_scope, loopvar, item, is_object)?;

        for stmt in body {
            match stmt.execute(&loop_scope) {
                Ok(val) => push_array(&result, val),
                Err(JinjaError::Continue) => continue,
                Err(JinjaError::Break) => break,
                Err(e) => return Err(e),
            }
        }
        no_iteration = false;
    }
    if no_iteration {
        for stmt in default_block {
            push_array(&result, stmt.execute(ctx)?);
        }
    }
    let parts = mk_empty_string();
    gather_string_parts_recursive(&result, &parts, 0);
    Ok(parts)
}

fn apply_loopvar(ctx: &Context, loopvar: &Stmt, item: &Value, _is_object: bool) -> JResult<()> {
    match &loopvar.node {
        StmtNode::Identifier(name) => ctx.set_val(name, item.clone()),
        StmtNode::TupleLiteral(targets) => {
            if !is_val(item, ValueKind::Array) {
                return Err(JinjaError::Error(format!(
                    "Cannot unpack non-iterable type: {}",
                    type_name(item)
                )));
            }
            let arr = as_array(item)?;
            if targets.len() != arr.len() {
                return Err(JinjaError::Error(format!(
                    "Too {} items to unpack",
                    if targets.len() > arr.len() { "few" } else { "many" }
                )));
            }
            for (target, val) in targets.iter().zip(arr) {
                let StmtNode::Identifier(name) = &target.node else {
                    return Err(JinjaError::Error(format!(
                        "Cannot unpack non-identifier type: {}",
                        stmt_type(target)
                    )));
                };
                ctx.set_val(name, val)?;
            }
            Ok(())
        }
        _ => Err(JinjaError::Error(format!(
            "Invalid loop variable(s): {}",
            stmt_type(loopvar)
        ))),
    }
}

fn execute_member(
    object_stmt: &Stmt,
    property: Option<&Stmt>,
    computed: bool,
    static_property: &Option<Value>,
    ctx: &Context,
) -> JResult<Value> {
    let object = object_stmt.execute(ctx)?;
    let mut property_val: Option<Value> = None;

    if computed {
        let Some(prop) = property else {
            return Ok(mk_undefined("object_property"));
        };
        if let StmtNode::Slice { start, stop, step } = &prop.node {
            let mut arr_size = 0i64;
            if is_val(&object, ValueKind::Array) {
                arr_size = as_array(&object)?.len() as i64;
            } else if kind_of(&object) == ValueKind::String {
                arr_size = as_string(&object)?.length() as i64;
            }
            let step_val = match step {
                Some(e) => e.execute(ctx)?,
                None => mk_int(1),
            };
            let step_i = as_int(&step_val)?;
            let start_val = match start {
                Some(e) => e.execute(ctx)?,
                None => mk_int(if step_i < 0 { arr_size - 1 } else { 0 }),
            };
            let stop_val = match stop {
                Some(e) => e.execute(ctx)?,
                None => mk_int(if step_i < 0 { -1 } else { arr_size }),
            };
            let slice_func = try_builtin_func(ctx, "slice", &object, false)?;
            let mut args = FuncArgs::new(ctx);
            args.push_back(start_val);
            args.push_back(stop_val);
            args.push_back(step_val);
            return invoke(&slice_func, &args);
        }
        property_val = Some(prop.execute(ctx)?);
    } else if let Some(prop) = property {
        if let StmtNode::IntegerLiteral(_) = &prop.node {
            let val = prop.execute(ctx)?;
            if as_int(&val)? < 0 {
                return Err(JinjaError::Error("Static member property cannot be negative".into()));
            }
            property_val = Some(val);
        } else {
            let StmtNode::Identifier(prop_name) = &prop.node else {
                return Err(JinjaError::Error(
                    "Static member property must be an identifier".into(),
                ));
            };
            let key = static_property
                .clone()
                .unwrap_or_else(|| mk_string(prop_name.clone()));
            if let Some(func) = find_builtin_func(ctx, prop_name, &object) {
                return Ok(func);
            }
            property_val = Some(cached_val(ctx, &key));
        }
    }

    let Some(property_val) = property_val else {
        return Ok(mk_undefined("object_property"));
    };
    if is_undefined(&property_val) {
        return Ok(mk_undefined("object_property"));
    }
    if !is_immutable(&property_val) || !is_hashable(&property_val) {
        return Err(JinjaError::Error(format!(
            "Type: {} is not allowed as object key",
            type_name(&property_val)
        )));
    }

    let mut val: Option<Value> = None;
    if is_undefined(&object) {
        return Ok(mk_undefined("object_property"));
    } else if kind_of(&object) == ValueKind::Object {
        val = find(&object, &property_val);
        if val.as_ref().map(is_undefined).unwrap_or(true) {
            val = Some(try_builtin_func(
                ctx,
                &as_string(&property_val)?.str(),
                &object,
                true,
            )?);
        }
    } else if is_val(&object, ValueKind::Array) || kind_of(&object) == ValueKind::String {
        if is_val(&property_val, ValueKind::Integer) || kind_of(&property_val) == ValueKind::Boolean {
            let index = as_int(&property_val)?;
            if is_val(&object, ValueKind::Array) {
                let arr = as_array(&object)?;
                if index < 0 {
                    let distance = (-(index + 1)) as u64 + 1;
                    if distance <= arr.len() as u64 {
                        val = Some(arr[arr.len() - distance as usize].clone());
                    }
                } else if (index as u64) < arr.len() as u64 {
                    val = Some(arr[index as usize].clone());
                }
            } else if index >= 0 {
                if let Some(byte) = as_string(&object)?.byte_at(index as usize) {
                    val = Some(mk_string((byte as char).to_string()));
                }
            }
        } else if kind_of(&property_val) == ValueKind::String {
            val = Some(try_builtin_func(
                ctx,
                &as_string(&property_val)?.str(),
                &object,
                true,
            )?);
        }
    } else {
        if kind_of(&property_val) != ValueKind::String {
            return Err(JinjaError::Error(format!(
                "Cannot access property with non-string: got {}",
                type_name(&property_val)
            )));
        }
        val = Some(try_builtin_func(
            ctx,
            &as_string(&property_val)?.str(),
            &object,
            true,
        )?);
    }

    let val = val.unwrap_or_else(|| mk_undefined("object_property"));
    if ctx.is_get_stats {
        mark_used(&val, false, 0);
        mark_used(&object, false, 0);
        mark_used(&property_val, false, 0);
        if kind_of(&object) == ValueKind::Object
            || kind_of(&property_val) == ValueKind::String
            || kind_of(&property_val) == ValueKind::Real
            || is_val(&property_val, ValueKind::Array)
            || is_none(&property_val)
        {
            object.borrow_mut().stats.ops.insert("object_access".into());
        } else if kind_of(&property_val) == ValueKind::Integer
            || kind_of(&property_val) == ValueKind::Boolean
        {
            object.borrow_mut().stats.ops.insert("array_access".into());
        }
    }
    Ok(val)
}

// ---------------------------------------------------------------------------
// gathering output
// ---------------------------------------------------------------------------

pub fn gather_string_parts_recursive(val: &Value, parts: &Value, depth: usize) {
    if kind_of(val) == ValueKind::String {
        if let ValueData::String { val_str: incoming } = &val.borrow().data {
            if let ValueData::String { val_str: out } = &mut parts.borrow_mut().data {
                out.append(incoming);
            }
        }
    } else if kind_of(val) == ValueKind::Integer
        || kind_of(val) == ValueKind::Real
        || kind_of(val) == ValueKind::Boolean
    {
        let s = as_string(val).unwrap_or_default().str();
        if let ValueData::String { val_str } = &mut parts.borrow_mut().data {
            val_str.append(&JString::new(s));
        }
    } else if is_val(val, ValueKind::Array) && depth >= JINJA_RECURSIVE_LIMIT {
        // C++ throws here. Keep the same upper bound without unwinding.
    } else if is_val(val, ValueKind::Array) {
        for item in as_array(val).unwrap_or_default() {
            gather_string_parts_recursive(&item, parts, depth + 1);
        }
    }
}

pub fn gather_string_parts(val: &Value) -> Value {
    let parts = mk_empty_string();
    gather_string_parts_recursive(val, &parts, 0);
    {
        let mut guard = parts.borrow_mut();
        if let ValueData::String { val_str } = &mut guard.data {
            if !val_str.parts.is_empty() {
                let p = &mut val_str.parts;
                let mut w = 0usize;
                for r in 1..p.len() {
                    if p[w].is_input == p[r].is_input {
                        let suffix = p[r].val.clone();
                        p[w].val.push_str(&suffix);
                    } else {
                        w += 1;
                        if w != r {
                            p[w] = p[r].clone();
                        }
                    }
                }
                p.truncate(w + 1);
            }
        }
    }
    parts
}

#[allow(non_camel_case_types)]
pub struct runtime<'a> {
    pub ctx: &'a Context,
}

impl<'a> runtime<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self { ctx }
    }

    pub fn execute(&self, prog: &Program) -> JResult<Value> {
        let results = mk_array();
        let StmtNode::Program { body } = &prog.node else {
            return Err(JinjaError::Error(
                "Cannot execute program directly, use jinja::runtime instead".into(),
            ));
        };
        for stmt in body {
            push_array(&results, stmt.execute(self.ctx)?);
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// debug dump (replaces the C++ visitor mechanism)
// ---------------------------------------------------------------------------

pub fn debug_dump_program(prog: &Program, src: &str) -> String {
    let StmtNode::Program { body } = &prog.node else {
        return String::new();
    };
    let mut out = String::new();
    for stmt in body {
        dump_stmt(stmt, 0, src, &mut out);
    }
    out
}

fn dump_stmt(stmt: &Stmt, lvl: usize, src: &str, out: &mut String) {
    let indent = |l: usize| " ".repeat(l * 2);
    out.push_str(&format!("{}{}:\n", indent(lvl), stmt_type(stmt)));
    let lvl = lvl + 1;
    let (children, is_leaf) = match &stmt.node {
        StmtNode::If { test, body, alternate } => {
            (vec![("test", vec![test.clone()]), ("body", body.clone()), ("alternate", alternate.clone())], false)
        }
        StmtNode::For { loopvar, iterable, body, default_block } => {
            (vec![("loopvar", vec![loopvar.clone()]), ("iterable", vec![iterable.clone()]), ("body", body.clone()), ("default_block", default_block.clone())], false)
        }
        StmtNode::Set { assignee, val, body } => {
            let mut c = vec![("assignee", vec![assignee.clone()])];
            if let Some(v) = val {
                c.push(("value", vec![v.clone()]));
            }
            c.push(("body", body.clone()));
            (c, false)
        }
        StmtNode::Macro { name, args, body } => {
            (vec![("name", vec![name.clone()]), ("args", args.clone()), ("body", body.clone())], false)
        }
        StmtNode::MemberExpression { object, property, .. } => {
            let mut c = vec![("object", vec![object.clone()])];
            if let Some(p) = property {
                c.push(("property", vec![p.clone()]));
            }
            (c, false)
        }
        StmtNode::CallExpression { callee, args } => {
            (vec![("callee", vec![callee.clone()]), ("args", args.clone())], false)
        }
        StmtNode::Binary { left, right, .. } => {
            (vec![("left", vec![left.clone()]), ("right", vec![right.clone()])], false)
        }
        StmtNode::ComparisonChain { operands, .. } => {
            (vec![("operands", operands.clone())], false)
        }
        StmtNode::Filter { operand, filter } => {
            (vec![("operand", vec![operand.clone()]), ("filter", vec![filter.clone()])], false)
        }
        StmtNode::FilterStatement { filter, body } => {
            (vec![("filter", vec![filter.clone()]), ("body", body.clone())], false)
        }
        StmtNode::Select { lhs, test } => {
            (vec![("lhs", vec![lhs.clone()]), ("test", vec![test.clone()])], false)
        }
        StmtNode::Test { operand, test, .. } => {
            (vec![("operand", vec![operand.clone()]), ("test", vec![test.clone()])], false)
        }
        StmtNode::Unary { argument, .. } => (vec![("argument", vec![argument.clone()])], false),
        StmtNode::Slice { start, stop, step } => {
            let mut c = Vec::new();
            for (label, e) in [("start_expr", start), ("stop_expr", stop), ("step_expr", step)] {
                if let Some(e) = e {
                    c.push((label, vec![e.clone()]));
                }
            }
            (c, false)
        }
        StmtNode::KeywordArgument { key, val } => {
            (vec![("key", vec![key.clone()]), ("val", vec![val.clone()])], false)
        }
        StmtNode::Spread { argument, .. } => (vec![("argument", vec![argument.clone()])], false),
        StmtNode::CallStatement { call, caller_args, body } => {
            (vec![("call", vec![call.clone()]), ("caller_args", caller_args.clone()), ("body", body.clone())], false)
        }
        StmtNode::Ternary { condition, true_expr, false_expr } => {
            (vec![("condition", vec![condition.clone()]), ("true_expr", vec![true_expr.clone()]), ("false_expr", vec![false_expr.clone()])], false)
        }
        _ => (Vec::new(), true),
    };

    if is_leaf {
        out.push_str(&format!(
            "{}(leaf) at {} in source:\n",
            indent(lvl),
            get_line_col(src, stmt.pos)
        ));
        let mut snippet = peak_source(src, stmt.pos, 40);
        string_replace_all(&mut snippet, "\n", &format!("\n{}", indent(lvl)));
        out.push_str(&indent(lvl));
        out.push_str(&snippet);
        out.push('\n');
    } else {
        for (label, kids) in children {
            out.push_str(&format!("{}{}:\n", indent(lvl), label));
            let inner = lvl + 1;
            if kids.is_empty() {
                out.push_str(&format!("{}<empty>\n\n", indent(inner)));
            } else {
                for child in kids {
                    dump_stmt(&child, inner, src, out);
                }
            }
        }
    }
}

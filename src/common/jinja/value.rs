//! `jinja::value`: primitive values, builtin functions and JSON conversion.
//!
//! C++ uses a `shared_ptr<value_t>` base class with virtual methods. Here a
//! single `Value` handle is an `Rc<RefCell<ValueT>>` and the virtual dispatch
//! is replaced by matches on `ValueKind`.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::runtime::Context;
use super::string::{string_as_repr, JString, StringPart};
use super::utils::{parse_utf8_codepoint, JResult, JinjaError, Utf8Status, JINJA_RECURSIVE_LIMIT};
use crate::json::Json;

// ---------------------------------------------------------------------------
// values
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValueKind {
    Integer,
    Real,
    String,
    Boolean,
    Array,
    Tuple,
    Object,
    None,
    Undefined,
    Function,
    Keyword,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub used: bool,
    pub ops: BTreeSet<String>,
}

pub enum ValueData {
    Integer {
        val_int: i64,
        val_flt: f64,
    },
    Real {
        val_int: i64,
        val_flt: f64,
    },
    String {
        val_str: JString,
    },
    Boolean {
        val_int: i64,
        val_flt: f64,
    },
    Array {
        items: Vec<Value>,
        immutable: bool,
    },
    Object {
        val_obj: Vec<(Value, Value)>,
        has_builtins: bool,
    },
    None,
    Undefined {
        hint: String,
    },
    Function {
        func: FuncHandler,
        name: String,
        arg0: Option<Value>,
    },
    Keyword {
        key: String,
        val: Value,
    },
}

pub struct ValueT {
    pub kind: ValueKind,
    pub stats: Stats,
    pub data: ValueData,
}

impl std::fmt::Debug for ValueT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValueT").field("kind", &self.kind).finish()
    }
}

pub type Value = Rc<RefCell<ValueT>>;

pub fn mk_int(v: i64) -> Value {
    let mut val_flt = v as f64;
    if float_to_int(val_flt) != v {
        val_flt = if v < 0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::Integer,
        stats: Stats::default(),
        data: ValueData::Integer {
            val_int: v,
            val_flt,
        },
    }))
}

pub fn mk_float(v: f64) -> Value {
    let val_int = if v.is_finite() { float_to_int(v) } else { 0 };
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::Real,
        stats: Stats::default(),
        data: ValueData::Real {
            val_int,
            val_flt: v,
        },
    }))
}

pub fn mk_string(v: impl Into<String>) -> Value {
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::String,
        stats: Stats::default(),
        data: ValueData::String {
            val_str: JString::new(v),
        },
    }))
}

pub fn mk_empty_string() -> Value {
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::String,
        stats: Stats::default(),
        data: ValueData::String {
            val_str: JString::default(),
        },
    }))
}

pub fn mk_jstring(v: JString) -> Value {
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::String,
        stats: Stats::default(),
        data: ValueData::String { val_str: v },
    }))
}

pub fn mk_bool(v: bool) -> Value {
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::Boolean,
        stats: Stats::default(),
        data: ValueData::Boolean {
            val_int: v as i64,
            val_flt: if v { 1.0 } else { 0.0 },
        },
    }))
}

pub fn mk_array() -> Value {
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::Array,
        stats: Stats::default(),
        data: ValueData::Array {
            items: Vec::new(),
            immutable: false,
        },
    }))
}

pub fn mk_array_from(items: Vec<Value>) -> Value {
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::Array,
        stats: Stats::default(),
        data: ValueData::Array {
            items,
            immutable: false,
        },
    }))
}

pub fn mk_tuple(items: Vec<Value>) -> Value {
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::Tuple,
        stats: Stats::default(),
        data: ValueData::Array {
            items,
            immutable: true,
        },
    }))
}

pub fn mk_object() -> Value {
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::Object,
        stats: Stats::default(),
        data: ValueData::Object {
            val_obj: Vec::new(),
            has_builtins: true,
        },
    }))
}

#[allow(dead_code)]
pub fn mk_object_from(pairs: Vec<(Value, Value)>) -> Value {
    let obj = mk_object();
    for (k, v) in pairs {
        insert(&obj, k, v).ok();
    }
    obj
}

pub fn mk_none() -> Value {
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::None,
        stats: Stats::default(),
        data: ValueData::None,
    }))
}

pub fn mk_undefined(hint: impl Into<String>) -> Value {
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::Undefined,
        stats: Stats::default(),
        data: ValueData::Undefined { hint: hint.into() },
    }))
}

pub fn mk_func(name: impl Into<String>, func: FuncHandler, arg0: Option<Value>) -> Value {
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::Function,
        stats: Stats::default(),
        data: ValueData::Function {
            func,
            name: name.into(),
            arg0,
        },
    }))
}

pub fn mk_kwarg(key: impl Into<String>, val: Value) -> Value {
    Rc::new(RefCell::new(ValueT {
        kind: ValueKind::Keyword,
        stats: Stats::default(),
        data: ValueData::Keyword {
            key: key.into(),
            val,
        },
    }))
}

pub fn kind_of(v: &Value) -> ValueKind {
    v.borrow().kind
}

/// `is_val<T>`: `value_array` also matches tuples.
pub fn is_val(v: &Value, kind: ValueKind) -> bool {
    let k = kind_of(v);
    k == kind || (kind == ValueKind::Array && k == ValueKind::Tuple)
}

pub fn is_none(v: &Value) -> bool {
    kind_of(v) == ValueKind::None
}

pub fn is_undefined(v: &Value) -> bool {
    kind_of(v) == ValueKind::Undefined
}

pub fn type_name(v: &Value) -> String {
    let b = v.borrow();
    match b.kind {
        ValueKind::Integer => "Integer".into(),
        ValueKind::Real => "Float".into(),
        ValueKind::String => "String".into(),
        ValueKind::Boolean => "Boolean".into(),
        ValueKind::Array => "Array".into(),
        ValueKind::Tuple => "Tuple".into(),
        ValueKind::Object => "Object".into(),
        ValueKind::None => "None".into(),
        ValueKind::Undefined => match &b.data {
            ValueData::Undefined { hint } if hint.is_empty() => "Undefined".into(),
            ValueData::Undefined { hint } => format!("Undefined (hint: '{hint}')"),
            _ => "Undefined".into(),
        },
        ValueKind::Function => "Function".into(),
        ValueKind::Keyword => "KwArg".into(),
    }
}

fn type_error(v: &Value, expected: &str) -> JinjaError {
    JinjaError::Error(format!("{} {}", type_name(v), expected))
}

pub fn as_int(v: &Value) -> JResult<i64> {
    match &v.borrow().data {
        ValueData::Integer { val_int, .. }
        | ValueData::Real { val_int, .. }
        | ValueData::Boolean { val_int, .. } => Ok(*val_int),
        _ => Err(type_error(v, "is not an int value")),
    }
}

pub fn as_float(v: &Value) -> JResult<f64> {
    match &v.borrow().data {
        ValueData::Integer { val_flt, .. } | ValueData::Real { val_flt, .. } => Ok(*val_flt),
        ValueData::Boolean { val_flt, .. } => Ok(*val_flt),
        _ => Err(type_error(v, "is not a float value")),
    }
}

pub fn as_string(v: &Value) -> JResult<JString> {
    let b = v.borrow();
    match &b.data {
        ValueData::String { val_str } => Ok(val_str.clone()),
        ValueData::Integer { val_int, .. } => Ok(JString::from_int(*val_int)),
        ValueData::Real { val_flt, .. } => {
            let mut out = format!("{val_flt:.6}");
            while out.ends_with('0') {
                out.pop();
            }
            if out.ends_with('.') {
                out.push('0');
            }
            Ok(JString::new(out))
        }
        ValueData::Boolean { val_int, .. } => {
            Ok(JString::new(if *val_int != 0 { "True" } else { "False" }))
        }
        ValueData::Array { items, immutable } => {
            let mut out = String::new();
            out.push(if *immutable { '(' } else { '[' });
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&value_to_string_repr(item));
            }
            if *immutable && items.len() == 1 {
                out.push(',');
            }
            out.push(if *immutable { ')' } else { ']' });
            Ok(JString::new(out))
        }
        ValueData::Object { val_obj, .. } => {
            let mut out = String::from("{");
            for (i, (k, val)) in val_obj.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&value_to_string_repr(k));
                out.push_str(": ");
                out.push_str(&value_to_string_repr(val));
            }
            out.push('}');
            Ok(JString::new(out))
        }
        ValueData::None => Ok(JString::new("None")),
        _ => Err(type_error(v, "is not a string value")),
    }
}

pub fn as_bool(v: &Value) -> JResult<bool> {
    let b = v.borrow();
    match &b.data {
        ValueData::Integer { val_int, .. } | ValueData::Boolean { val_int, .. } => {
            Ok(*val_int != 0)
        }
        ValueData::Real { val_flt, .. } => Ok(*val_flt != 0.0),
        ValueData::String { val_str } => Ok(!val_str.empty()),
        ValueData::Array { items, .. } => Ok(!items.is_empty()),
        ValueData::Object { val_obj, .. } => Ok(!val_obj.is_empty()),
        _ => Ok(false),
    }
}

pub fn as_array(v: &Value) -> JResult<Vec<Value>> {
    match &v.borrow().data {
        ValueData::Array { items, .. } => Ok(items.clone()),
        _ => Err(type_error(v, "is not an array value")),
    }
}

pub fn as_ordered_object(v: &Value) -> JResult<Vec<(Value, Value)>> {
    match &v.borrow().data {
        ValueData::Object { val_obj, .. } => Ok(val_obj.clone()),
        _ => Err(type_error(v, "is not an object value")),
    }
}

pub fn as_repr(v: &Value) -> String {
    let b = v.borrow();
    match &b.data {
        ValueData::String { val_str } => string_as_repr(val_str),
        ValueData::None | ValueData::Undefined { .. } => type_name(v),
        ValueData::Function { name, arg0, .. } => {
            let arg = arg0.as_ref().map(|a| as_repr(a)).unwrap_or_default();
            format!("Function<{name}>({arg})")
        }
        ValueData::Keyword { .. } => "KwArg".into(),
        _ => as_string(v).map(|s| s.str()).unwrap_or_default(),
    }
}

pub fn is_numeric(v: &Value) -> bool {
    matches!(
        kind_of(v),
        ValueKind::Integer | ValueKind::Real | ValueKind::Boolean
    )
}

pub fn is_immutable(v: &Value) -> bool {
    !matches!(kind_of(v), ValueKind::Array | ValueKind::Object)
}

pub fn is_hashable(v: &Value) -> bool {
    let b = v.borrow();
    match &b.data {
        ValueData::Integer { .. }
        | ValueData::Real { .. }
        | ValueData::String { .. }
        | ValueData::Boolean { .. }
        | ValueData::None
        | ValueData::Keyword { .. } => true,
        ValueData::Array { items, .. } => items
            .iter()
            .all(|item| is_immutable(item) && is_hashable(item)),
        ValueData::Object { val_obj, .. } => val_obj
            .iter()
            .all(|(_, val)| is_immutable(val) && is_hashable(val)),
        _ => false,
    }
}

fn ensure_key(v: &Value) -> JResult<()> {
    if !is_immutable(v) || !is_hashable(v) {
        return Err(JinjaError::Error(format!(
            "Object key of unhashable type: {}",
            type_name(v)
        )));
    }
    Ok(())
}

pub fn has_key(v: &Value, key: &Value) -> JResult<bool> {
    ensure_key(key)?;
    let b = v.borrow();
    match &b.data {
        ValueData::Object { val_obj, .. } => Ok(val_obj.iter().any(|(k, _)| equivalent(k, key))),
        _ => Err(type_error(v, "is not an object value")),
    }
}

pub fn insert(v: &Value, key: Value, val: Value) -> JResult<()> {
    {
        let b = v.borrow();
        if !is_immutable(v) {
            // object and array are mutable; only object supports insert
            if !matches!(b.data, ValueData::Object { .. }) {
                return Err(JinjaError::Error(
                    "Attempting to modify immutable type".into(),
                ));
            }
        }
    }
    ensure_key(&key)?;
    let mut b = v.borrow_mut();
    match &mut b.data {
        ValueData::Object { val_obj, .. } => {
            if let Some((_, slot)) = val_obj.iter_mut().find(|(k, _)| equivalent(k, &key)) {
                *slot = val;
            } else {
                val_obj.push((key, val));
            }
            Ok(())
        }
        _ => Err(JinjaError::Error(
            "Attempting to modify immutable type".into(),
        )),
    }
}

pub fn insert_str(v: &Value, key: &str, val: Value) -> JResult<()> {
    insert(v, mk_string(key), val)
}

/// Object/array access with an optional default (`value& at(key, default)`).
pub fn at(v: &Value, key: &Value, default: Option<Value>) -> JResult<Option<Value>> {
    let b = v.borrow();
    match &b.data {
        ValueData::Array { items, .. } => {
            let raw = if is_val(key, ValueKind::Integer) || is_val(key, ValueKind::Boolean) {
                as_int(key)?
            } else {
                return Ok(default);
            };
            let index = if raw < 0 {
                items.len() as i64 + raw
            } else {
                raw
            };
            if index < 0 || index as usize >= items.len() {
                Ok(default)
            } else {
                Ok(Some(items[index as usize].clone()))
            }
        }
        ValueData::Object { val_obj, .. } => {
            ensure_key(key)?;
            Ok(val_obj
                .iter()
                .find(|(k, _)| equivalent(k, key))
                .map(|(_, val)| val.clone())
                .or(default))
        }
        _ => Err(type_error(v, "is not an object value")),
    }
}

/// `at(key)` without a default: missing keys are an error.
#[allow(dead_code)]
pub fn at_required(v: &Value, key: &Value) -> JResult<Value> {
    match at(v, key, None)? {
        Some(val) => Ok(val),
        None => Err(JinjaError::Error(format!(
            "Key '{}' not found in value of type {}",
            as_string(key).map(|s| s.str()).unwrap_or_default(),
            type_name(v)
        ))),
    }
}

pub fn at_str(v: &Value, key: &str, default: Option<Value>) -> JResult<Option<Value>> {
    at(v, &mk_string(key), default)
}

pub fn find(v: &Value, key: &Value) -> Option<Value> {
    at(v, key, None).ok().flatten()
}

pub fn equivalent(a: &Value, b: &Value) -> bool {
    let ba = a.borrow();
    let bb = b.borrow();
    if ba.kind == bb.kind || (ba.kind == ValueKind::Array && bb.kind == ValueKind::Tuple) {
        match (&ba.data, &bb.data) {
            (
                ValueData::Integer {
                    val_int: ia,
                    val_flt: fa,
                },
                _,
            ) if is_numeric(b) => {
                let ib = match &bb.data {
                    ValueData::Integer { val_int, .. } | ValueData::Boolean { val_int, .. } => {
                        *val_int
                    }
                    _ => 0,
                };
                let fb = match &bb.data {
                    ValueData::Integer { val_flt, .. }
                    | ValueData::Real { val_flt, .. }
                    | ValueData::Boolean { val_flt, .. } => *val_flt,
                    _ => 0.0,
                };
                *ia == ib && *fa == fb
            }
            (
                ValueData::Real {
                    val_int: ia,
                    val_flt: fa,
                },
                _,
            ) if is_numeric(b) => {
                let ib = match &bb.data {
                    ValueData::Integer { val_int, .. } | ValueData::Boolean { val_int, .. } => {
                        *val_int
                    }
                    _ => 0,
                };
                let fb = match &bb.data {
                    ValueData::Integer { val_flt, .. }
                    | ValueData::Real { val_flt, .. }
                    | ValueData::Boolean { val_flt, .. } => *val_flt,
                    _ => 0.0,
                };
                *ia == ib && *fa == fb
            }
            (
                ValueData::Boolean {
                    val_int: ia,
                    val_flt: fa,
                },
                _,
            ) if is_numeric(b) => {
                let ib = match &bb.data {
                    ValueData::Integer { val_int, .. } | ValueData::Boolean { val_int, .. } => {
                        *val_int
                    }
                    _ => 0,
                };
                let fb = match &bb.data {
                    ValueData::Integer { val_flt, .. }
                    | ValueData::Real { val_flt, .. }
                    | ValueData::Boolean { val_flt, .. } => *val_flt,
                    _ => 0.0,
                };
                *ia == ib && *fa == fb
            }
            (ValueData::String { val_str: sa }, ValueData::String { val_str: sb }) => sa.equals(sb),
            (ValueData::Array { items: aa, .. }, ValueData::Array { items: ab, .. }) => {
                is_hashable(a)
                    && is_hashable(b)
                    && aa.len() == ab.len()
                    && aa.iter().zip(ab.iter()).all(|(x, y)| equivalent(x, y))
            }
            (ValueData::Object { val_obj: aa, .. }, ValueData::Object { val_obj: ab, .. }) => {
                is_hashable(a)
                    && is_hashable(b)
                    && aa.len() == ab.len()
                    && aa
                        .iter()
                        .zip(ab.iter())
                        .all(|((ka, va), (kb, vb))| equivalent(ka, kb) && equivalent(va, vb))
            }
            (ValueData::None, ValueData::None) => true,
            (ValueData::Undefined { .. }, ValueData::Undefined { .. }) => true,
            (ValueData::Function { name: na, .. }, ValueData::Function { name: nb, .. }) => {
                na == nb
            }
            (ValueData::Keyword { key: ka, val: va }, ValueData::Keyword { key: kb, val: vb }) => {
                ka == kb && Rc::ptr_eq(va, vb)
            }
            _ => false,
        }
    } else {
        false
    }
}

pub fn nonequal(a: &Value, b: &Value) -> bool {
    let ba = a.borrow();
    let bb = b.borrow();
    match (&ba.data, &bb.data) {
        (ValueData::Integer { val_int: ia, .. }, _) if is_numeric(b) => {
            let ib = match &bb.data {
                ValueData::Integer { val_int, .. } | ValueData::Boolean { val_int, .. } => *val_int,
                _ => 0,
            };
            !(ba.kind == bb.kind && *ia == ib)
        }
        (ValueData::Boolean { val_int: ia, .. }, _) if is_numeric(b) => {
            let ib = match &bb.data {
                ValueData::Integer { val_int, .. } | ValueData::Boolean { val_int, .. } => *val_int,
                _ => 0,
            };
            !(ba.kind == bb.kind && *ia == ib)
        }
        (ValueData::Real { val_flt: fa, .. }, _) if is_numeric(b) => {
            let fb = match &bb.data {
                ValueData::Integer { val_flt, .. }
                | ValueData::Real { val_flt, .. }
                | ValueData::Boolean { val_flt, .. } => *val_flt,
                _ => 0.0,
            };
            !(ba.kind == bb.kind && *fa == fb)
        }
        _ => !equivalent(a, b),
    }
}

// ---------------------------------------------------------------------------
// function arguments
// ---------------------------------------------------------------------------

pub type FuncHandler = Rc<dyn for<'a> Fn(&FuncArgs<'a>) -> JResult<Value>>;

#[derive(Clone)]
pub struct FuncArgs<'a> {
    pub func_name: String,
    pub ctx: &'a Context,
    args: Vec<Value>,
}

impl<'a> FuncArgs<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self {
            func_name: String::new(),
            ctx,
            args: Vec::new(),
        }
    }

    pub fn get_kwarg(&self, key: &str, default: Value) -> Value {
        for arg in &self.args {
            if kind_of(arg) == ValueKind::Keyword {
                let b = arg.borrow();
                if let ValueData::Keyword { key: k, val } = &b.data {
                    if k == key {
                        return val.clone();
                    }
                }
            }
        }
        default
    }

    pub fn get_kwarg_or_pos(&self, key: &str, pos: usize) -> Value {
        let val = self.get_kwarg(key, mk_undefined(""));
        if is_undefined(&val)
            && pos < self.args.len()
            && kind_of(&self.args[pos]) != ValueKind::Keyword
        {
            return self.args[pos].clone();
        }
        val
    }

    pub fn get_pos(&self, pos: usize) -> JResult<Value> {
        if self.args.len() > pos {
            Ok(self.args[pos].clone())
        } else {
            Err(JinjaError::Raised(format!(
                "Function '{}' expected at least {} arguments, got {}",
                self.func_name,
                pos + 1,
                self.args.len()
            )))
        }
    }

    pub fn get_pos_or(&self, pos: usize, default: Value) -> Value {
        if self.args.len() > pos {
            self.args[pos].clone()
        } else {
            default
        }
    }

    pub fn get_args(&self) -> &[Value] {
        &self.args
    }

    pub fn count(&self) -> usize {
        self.args.len()
    }

    pub fn push_back(&mut self, val: Value) {
        self.args.push(val);
    }

    pub fn push_front(&mut self, val: Value) {
        self.args.insert(0, val);
    }

    pub fn ensure_kwargs(&self, names: &[&str]) -> JResult<()> {
        for arg in &self.args {
            if kind_of(arg) == ValueKind::Keyword {
                let b = arg.borrow();
                if let ValueData::Keyword { key, .. } = &b.data {
                    if !names.contains(&key.as_str()) {
                        return Err(JinjaError::Raised(format!(
                            "Function '{}' got an unexpected keyword argument '{}'",
                            self.func_name, key
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn ensure_count(&self, min: usize, max: usize) -> JResult<()> {
        let n = self.args.len();
        if n < min || n > max {
            return Err(JinjaError::Error(format!(
                "Function '{}' expected between {} and {} arguments, got {}",
                self.func_name, min, max, n
            )));
        }
        Ok(())
    }

    pub fn ensure_val(&self, pos: usize, kind: ValueKind) -> JResult<()> {
        let val = self.get_pos(pos)?;
        if !is_val(&val, kind) {
            return Err(JinjaError::Error(format!(
                "Function '{}' expected value of type {:?}, got {}",
                self.func_name,
                kind,
                type_name(&val)
            )));
        }
        Ok(())
    }
}

pub fn invoke(func: &Value, args: &FuncArgs<'_>) -> JResult<Value> {
    let (handler, name, arg0) = match &func.borrow().data {
        ValueData::Function { func, name, arg0 } => (func.clone(), name.clone(), arg0.clone()),
        _ => return Err(type_error(func, "is not a function value")),
    };
    let mut new_args = args.clone();
    new_args.func_name = name;
    if let Some(a) = arg0 {
        new_args.push_front(a);
    }
    handler(&new_args)
}

// ---------------------------------------------------------------------------
// comparison
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueCompareOp {
    Eq,
    Ge,
    Gt,
    Lt,
    Ne,
}

pub fn value_compare(a: &Value, b: &Value, op: ValueCompareOp) -> bool {
    if (is_val(a, ValueKind::Integer) || is_val(a, ValueKind::Real))
        && (is_val(b, ValueKind::Integer) || is_val(b, ValueKind::Real))
    {
        let x = as_float(a).unwrap_or(0.0);
        let y = as_float(b).unwrap_or(0.0);
        return match op {
            ValueCompareOp::Eq => x == y,
            ValueCompareOp::Ge => x >= y,
            ValueCompareOp::Gt => x > y,
            ValueCompareOp::Lt => x < y,
            ValueCompareOp::Ne => x != y,
        };
    }
    if (is_val(b, ValueKind::String)
        && (is_val(a, ValueKind::Integer) || is_val(a, ValueKind::Real)))
        || (is_val(a, ValueKind::String)
            && (is_val(b, ValueKind::Integer) || is_val(b, ValueKind::Real)))
        || (is_val(a, ValueKind::String) && is_val(b, ValueKind::String))
    {
        let x = as_string(a).unwrap_or_default().str();
        let y = as_string(b).unwrap_or_default().str();
        return match op {
            ValueCompareOp::Eq => x == y,
            ValueCompareOp::Ge => x >= y,
            ValueCompareOp::Gt => x > y,
            ValueCompareOp::Lt => x < y,
            ValueCompareOp::Ne => x != y,
        };
    }
    if kind_of(a) == ValueKind::Boolean && kind_of(b) == ValueKind::Boolean {
        let x = as_bool(a).unwrap_or(false);
        let y = as_bool(b).unwrap_or(false);
        return match op {
            ValueCompareOp::Eq => x == y,
            ValueCompareOp::Ne => x != y,
            _ => false,
        };
    }
    false
}

/// `float_to_int`: saturating, NaN as zero.
pub fn float_to_int(v: f64) -> i64 {
    if v.is_nan() {
        return 0;
    }
    if v >= 9223372036854775808.0 {
        return i64::MAX;
    }
    if v <= -9223372036854775808.0 {
        return i64::MIN;
    }
    v as i64
}

// ---------------------------------------------------------------------------
// builtin helpers
// ---------------------------------------------------------------------------

fn h(f: fn(&FuncArgs) -> JResult<Value>) -> FuncHandler {
    Rc::new(f)
}

fn default_value(args: &FuncArgs) -> JResult<Value> {
    args.ensure_count(2, 3)?;
    let check_bool = as_bool(&args.get_kwarg_or_pos("boolean", 2))?;
    let input = args.get_pos(0)?;
    let no_value = if check_bool {
        !as_bool(&input)?
    } else {
        is_undefined(&input) || is_none(&input)
    };
    Ok(if no_value { args.get_pos(1)? } else { input })
}

fn test_type(args: &FuncArgs, kinds: &[ValueKind]) -> JResult<Value> {
    args.ensure_count(1, 999)?;
    let val = args.get_pos(0)?;
    Ok(mk_bool(kinds.iter().any(|k| is_val(&val, *k))))
}

fn test_compare(args: &FuncArgs, op: ValueCompareOp) -> JResult<Value> {
    args.ensure_count(2, 2)?;
    Ok(mk_bool(value_compare(
        &args.get_pos(0)?,
        &args.get_pos(1)?,
        op,
    )))
}

fn slice_generic<T: Clone>(array: &[T], start: i64, stop: i64, step: i64) -> Vec<T> {
    let len = array.len() as i64;
    let direction = if step > 0 {
        1
    } else if step < 0 {
        -1
    } else {
        0
    };
    let mut start_val;
    let mut stop_val;
    if direction >= 0 {
        start_val = start;
        start_val = if start_val < 0 {
            (len + start_val).max(0)
        } else {
            start_val.min(len)
        };
        stop_val = stop;
        stop_val = if stop_val < 0 {
            (len + stop_val).max(0)
        } else {
            stop_val.min(len)
        };
    } else {
        start_val = start;
        start_val = if start_val < 0 {
            (len + start_val).max(0)
        } else {
            start_val.min(len - 1)
        };
        stop_val = stop;
        stop_val = if stop_val < -1 {
            (len + stop_val).max(-1)
        } else {
            stop_val.min(len - 1)
        };
    }
    let mut result = Vec::new();
    if direction == 0 {
        return result;
    }
    let mut i = start_val;
    while direction * i < direction * stop_val {
        if i >= 0 && i < len {
            result.push(array[i as usize].clone());
        }
        i += step;
    }
    result
}

fn slice_string(s: &str, start: i64, stop: i64, step: i64) -> String {
    // The C++ engine slices bytes; keep that behaviour for non-ASCII input.
    let sliced = slice_generic(s.as_bytes(), start, stop, step);
    String::from_utf8_lossy(&sliced).into_owned()
}

fn get_attribute(val: &Value, attr: &Value, default: Value) -> Value {
    if is_undefined(attr) {
        return default;
    }
    if is_val(val, ValueKind::Array) {
        let mut idx = attr.clone();
        if is_val(attr, ValueKind::String) {
            let s = as_string(attr).unwrap_or_default().str();
            if !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()) {
                idx = s
                    .parse::<i64>()
                    .map(mk_int)
                    .unwrap_or_else(|_| mk_undefined(""));
            }
        }
        at(val, &idx, Some(default))
            .unwrap_or(None)
            .unwrap_or(mk_undefined(""))
    } else if kind_of(val) == ValueKind::Object {
        at(val, attr, Some(default.clone()))
            .unwrap_or(None)
            .unwrap_or(default)
    } else {
        default
    }
}

fn select_or_reject(args: &FuncArgs, is_reject: bool, lookup_attr: bool) -> JResult<Value> {
    let test_pos = if lookup_attr { 2 } else { 1 };
    args.ensure_count(test_pos, 999)?;
    args.ensure_val(0, ValueKind::Array)?;
    let attr = if lookup_attr {
        let a = args.get_pos(1)?;
        args.ensure_val(1, ValueKind::String)?;
        a
    } else {
        mk_undefined("")
    };

    let mut test_name: Option<String> = None;
    if args.count() > test_pos {
        let t = args.get_pos(test_pos)?;
        args.ensure_val(test_pos, ValueKind::String)?;
        let name = as_string(&t)?.str();
        if global_builtin(&format!("test_is_{name}")).is_none() {
            return Err(JinjaError::Raised(format!(
                "{}: unknown test '{}'",
                args.func_name, name
            )));
        }
        test_name = Some(name);
    }

    let arr = as_array(&args.get_pos(0)?)?;
    let out = mk_array();
    let undef = mk_undefined("");
    for item in arr {
        let mut val = item.clone();
        if lookup_attr {
            val = get_attribute(&item, &attr, undef.clone());
        }
        let selected = if let Some(name) = &test_name {
            let mut test_args = FuncArgs::new(args.ctx);
            test_args.push_back(val);
            for i in test_pos + 1..args.count() {
                test_args.push_back(args.get_pos(i)?);
            }
            let handler = global_builtin(&format!("test_is_{name}")).unwrap();
            as_bool(&handler(&test_args)?)?
        } else {
            as_bool(&val)?
        };
        if selected != is_reject {
            push_array(&out, item);
        }
    }
    Ok(out)
}

pub fn push_array(v: &Value, val: Value) {
    if let ValueData::Array { items, immutable } = &mut v.borrow_mut().data {
        if !*immutable {
            items.push(val);
        }
    }
}

fn pop_array(v: &Value, index: i64) -> JResult<Value> {
    let mut b = v.borrow_mut();
    if let ValueData::Array { items, immutable } = &mut b.data {
        if *immutable {
            return Err(JinjaError::Error(
                "Attempting to modify immutable type".into(),
            ));
        }
        let mut idx = index;
        if idx < 0 {
            idx = items.len() as i64 + idx;
        }
        if idx < 0 || idx as usize >= items.len() {
            return Err(JinjaError::Error(format!(
                "Index {} out of bounds for array of size {}",
                idx,
                items.len()
            )));
        }
        Ok(items.remove(idx as usize))
    } else {
        Err(type_error(v, "is not an array value"))
    }
}

// ---------------------------------------------------------------------------
// float / JSON formatting
// ---------------------------------------------------------------------------

fn float_repr(val: f64) -> String {
    if !val.is_finite() {
        return if val.is_nan() {
            "nan".into()
        } else if val < 0.0 {
            "-inf".into()
        } else {
            "inf".into()
        };
    }
    let mag = val.abs();
    // frexp(mag) == 0.5: the mantissa is exactly one half.
    let mut exp2 = 0i32;
    let is_pow2 = (frexp(mag, &mut exp2) - 0.5).abs() < f64::EPSILON;
    let mut digits = String::new();
    let mut exp10 = 0i32;
    let mut prec = if mag < f64::MIN_POSITIVE { 0 } else { 15 - 1 };
    while digits.is_empty() {
        let buf = format!("{:.*e}", prec, mag);
        let epos = buf.find('e').unwrap_or(buf.len());
        let (mantissa, exp_part) = buf.split_at(epos);
        let d: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
        exp10 = exp_part.trim_start_matches('e').parse::<i32>().unwrap_or(0);
        let reads_back = |cand: &str| -> bool {
            format!("{}e{}", cand, exp10 - prec as i32)
                .parse::<f64>()
                .map(|v| v == mag)
                .unwrap_or(false)
        };
        if prec == 16 || reads_back(&d) {
            digits = d;
        } else if is_pow2 && !d.is_empty() && !d.ends_with('9') {
            let mut bumped = d.clone();
            let last = bumped.pop().unwrap();
            bumped.push(((last as u8) + 1) as char);
            if reads_back(&bumped) {
                digits = bumped;
            }
        }
        prec += 1;
    }
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }

    let n = digits.len() as i32;
    let point = exp10 + 1;
    let mut out = if val.is_sign_negative() {
        "-".to_string()
    } else {
        String::new()
    };
    if point > -4 && point <= 16 {
        if point <= 0 {
            out.push_str("0.");
            out.push_str(&"0".repeat((-point) as usize));
            out.push_str(&digits);
        } else if point >= n {
            out.push_str(&digits);
            out.push_str(&"0".repeat((point - n) as usize));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        let sign = if exp10 < 0 { '-' } else { '+' };
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push_str(&format!("e{sign}{:02}", exp10.abs()));
    }
    out
}

fn frexp(mag: f64, exp: &mut i32) -> f64 {
    if mag == 0.0 || !mag.is_finite() {
        *exp = 0;
        return mag;
    }
    let mut e = 0i32;
    let mut m = mag;
    while m >= 1.0 {
        m *= 0.5;
        e += 1;
    }
    while m < 0.5 {
        m *= 2.0;
        e -= 1;
    }
    *exp = e;
    m
}

fn float_json(val: f64) -> String {
    if val.is_nan() {
        return "NaN".into();
    }
    if val.is_infinite() {
        return if val < 0.0 { "-Infinity" } else { "Infinity" }.into();
    }
    float_repr(val)
}

fn append_codepoint_as_ascii_json_escape(out: &mut String, cp: u32) {
    let mut append_u16 = |value: u32| {
        out.push_str(&format!("\\u{:04x}", value));
    };
    if cp <= 0xffff {
        append_u16(cp);
        return;
    }
    let cp = cp - 0x10000;
    append_u16(0xd800 + ((cp >> 10) & 0x3ff));
    append_u16(0xdc00 + (cp & 0x3ff));
}

fn json_ensure_ascii_preserving_format(json_str: &str) -> String {
    let bytes = json_str.as_bytes();
    let mut output = String::with_capacity(json_str.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut pos = 0usize;
    while pos < bytes.len() {
        let ch = bytes[pos];
        if !in_string {
            output.push(ch as char);
            if ch == b'"' {
                in_string = true;
            }
            pos += 1;
            continue;
        }
        if escaped {
            output.push(ch as char);
            escaped = false;
            pos += 1;
            continue;
        }
        if ch == b'\\' {
            output.push('\\');
            escaped = true;
            pos += 1;
            continue;
        }
        if ch == b'"' {
            output.push('"');
            in_string = false;
            pos += 1;
            continue;
        }
        if ch < 0x80 {
            output.push(ch as char);
            pos += 1;
            continue;
        }
        let parsed = parse_utf8_codepoint(bytes, pos);
        if parsed.status != Utf8Status::Success {
            output.push_str("\\ufffd");
            pos += 1;
            continue;
        }
        append_codepoint_as_ascii_json_escape(&mut output, parsed.codepoint);
        pos += parsed.bytes_consumed;
    }
    output
}

pub fn value_to_json(
    val: &Value,
    indent: i32,
    item_sep: &str,
    key_sep: &str,
    sort_keys: bool,
) -> String {
    let mut out = String::new();
    value_to_json_internal(&mut out, val, 0, indent, item_sep, key_sep, sort_keys);
    out
}

fn value_to_json_internal(
    out: &mut String,
    val: &Value,
    curr_lvl: usize,
    indent: i32,
    item_sep: &str,
    key_sep: &str,
    sort_keys: bool,
) {
    let indent_str = || {
        if indent > 0 {
            " ".repeat(curr_lvl * indent as usize)
        } else {
            String::new()
        }
    };
    let newline = || if indent >= 0 { "\n" } else { "" };

    if is_none(val) || is_undefined(val) {
        out.push_str("null");
    } else if kind_of(val) == ValueKind::Boolean {
        out.push_str(if as_bool(val).unwrap_or(false) {
            "true"
        } else {
            "false"
        });
    } else if kind_of(val) == ValueKind::Integer {
        out.push_str(&as_int(val).unwrap_or(0).to_string());
    } else if kind_of(val) == ValueKind::Real {
        out.push_str(&float_json(as_float(val).unwrap_or(0.0)));
    } else if kind_of(val) == ValueKind::String {
        out.push('"');
        for c in as_string(val).unwrap_or_default().str().chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\u{08}' => out.push_str("\\b"),
                '\u{0c}' => out.push_str("\\f"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('"');
    } else if (is_val(val, ValueKind::Array) || kind_of(val) == ValueKind::Object)
        && curr_lvl >= JINJA_RECURSIVE_LIMIT
    {
        // The C++ engine throws here; the caller already bounds recursion.
        out.push_str("null");
    } else if is_val(val, ValueKind::Array) {
        let arr = as_array(val).unwrap_or_default();
        out.push('[');
        if !arr.is_empty() {
            out.push_str(newline());
            for (i, item) in arr.iter().enumerate() {
                out.push_str(&indent_str());
                if indent > 0 {
                    out.push_str(&" ".repeat(indent as usize));
                }
                value_to_json_internal(
                    out,
                    item,
                    curr_lvl + 1,
                    indent,
                    item_sep,
                    key_sep,
                    sort_keys,
                );
                if i < arr.len() - 1 {
                    out.push_str(item_sep);
                }
                out.push_str(newline());
            }
            out.push_str(&indent_str());
        }
        out.push(']');
    } else if kind_of(val) == ValueKind::Object {
        let obj = as_ordered_object(val).unwrap_or_default();
        let mut items: Vec<(String, Value)> = obj
            .iter()
            .map(|(k, v)| {
                let key = if kind_of(k) == ValueKind::Real {
                    float_json(as_float(k).unwrap_or(0.0))
                } else {
                    as_string(k).unwrap_or_default().str()
                };
                (key, v.clone())
            })
            .collect();
        if sort_keys {
            items.sort_by(|a, b| a.0.cmp(&b.0));
        }
        out.push('{');
        if !items.is_empty() {
            out.push_str(newline());
            for (i, (key, item)) in items.iter().enumerate() {
                out.push_str(&indent_str());
                if indent > 0 {
                    out.push_str(&" ".repeat(indent as usize));
                }
                value_to_json_internal(
                    out,
                    &mk_string(key.clone()),
                    curr_lvl + 1,
                    indent,
                    item_sep,
                    key_sep,
                    sort_keys,
                );
                out.push_str(key_sep);
                value_to_json_internal(
                    out,
                    item,
                    curr_lvl + 1,
                    indent,
                    item_sep,
                    key_sep,
                    sort_keys,
                );
                if i < items.len() - 1 {
                    out.push_str(item_sep);
                }
                out.push_str(newline());
            }
            out.push_str(&indent_str());
        }
        out.push('}');
    } else {
        out.push_str("null");
    }
}

pub fn value_to_string_repr(val: &Value) -> String {
    if kind_of(val) == ValueKind::String {
        let s = as_string(val).unwrap_or_default().str();
        if s.contains('\'') {
            value_to_json(val, -1, ", ", ": ", false)
        } else {
            format!("'{s}'")
        }
    } else {
        as_repr(val)
    }
}

// ---------------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------------

pub fn mark_used(val: &Value, deep: bool, depth: usize) {
    val.borrow_mut().stats.used = true;
    if deep
        && (is_val(val, ValueKind::Array) || kind_of(val) == ValueKind::Object)
        && depth >= JINJA_RECURSIVE_LIMIT
    {
        // The C++ engine throws; templates are bounded by the parser.
        return;
    }
    if deep {
        if is_val(val, ValueKind::Array) {
            for item in as_array(val).unwrap_or_default() {
                mark_used(&item, true, depth + 1);
            }
        } else if kind_of(val) == ValueKind::Object {
            for (k, v) in as_ordered_object(val).unwrap_or_default() {
                mark_used(&k, true, depth + 1);
                mark_used(&v, true, depth + 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// builtins
// ---------------------------------------------------------------------------

fn toobject(args: &FuncArgs) -> JResult<Value> {
    let out = mk_object();
    let iter = args.get_pos_or(0, mk_undefined(""));
    let mut iter_first = false;
    if is_val(&iter, ValueKind::Array) {
        iter_first = true;
        for it in as_array(&iter)? {
            let arr = as_array(&it)?;
            if arr.len() == 2 {
                insert(&out, arr[0].clone(), arr[1].clone())?;
            } else {
                return Err(JinjaError::Raised(format!(
                    "namespace/dict() iterable argument must consist of tuples, not {}",
                    type_name(&it)
                )));
            }
        }
    } else if kind_of(&iter) == ValueKind::Object {
        iter_first = true;
        for (k, v) in as_ordered_object(&iter)? {
            insert(&out, k, v)?;
        }
    }
    for arg in args.get_args() {
        if kind_of(arg) == ValueKind::Keyword {
            let b = arg.borrow();
            if let ValueData::Keyword { key, val } = &b.data {
                insert_str(&out, key, val.clone())?;
            }
        } else if !iter_first {
            return Err(JinjaError::Raised(format!(
                "namespace/dict() arguments must be kwargs, dict and/or iterable of tuples, not {}",
                type_name(arg)
            )));
        }
        iter_first = false;
    }
    Ok(out)
}

fn raise_exception(args: &FuncArgs) -> JResult<Value> {
    args.ensure_count(1, 1)?;
    args.ensure_val(0, ValueKind::String)?;
    Err(JinjaError::Raised(format!(
        "Jinja Exception: {}",
        as_string(&args.get_pos(0)?)?.str()
    )))
}

fn strftime_now(args: &FuncArgs) -> JResult<Value> {
    args.ensure_count(1, 1)?;
    args.ensure_val(0, ValueKind::String)?;
    let format = as_string(&args.get_pos(0)?)?.str();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Ok(mk_string(format_time(&format, now)))
}

fn range_fn(args: &FuncArgs) -> JResult<Value> {
    args.ensure_count(1, 3)?;
    let count = args.count();
    let arg0 = args.get_pos(0)?;
    let arg1 = args.get_pos_or(1, mk_undefined(""));
    let arg2 = args.get_pos_or(2, mk_undefined(""));
    let (start, stop, step) = if count == 1 {
        (0, as_int(&arg0)?, 1)
    } else if count == 2 {
        (as_int(&arg0)?, as_int(&arg1)?, 1)
    } else {
        (as_int(&arg0)?, as_int(&arg1)?, as_int(&arg2)?)
    };
    let out = mk_array();
    if step == 0 {
        return Err(JinjaError::Raised(
            "range() step argument must not be zero".into(),
        ));
    }
    if step > 0 {
        let mut i = start;
        while i < stop {
            push_array(&out, mk_int(i));
            i = i.saturating_add(step);
        }
    } else {
        let mut i = start;
        while i > stop {
            push_array(&out, mk_int(i));
            i = i.saturating_add(step);
        }
    }
    Ok(out)
}

fn tojson(args: &FuncArgs) -> JResult<Value> {
    args.ensure_count(1, 5)?;
    let val_ascii = args.get_kwarg_or_pos("ensure_ascii", 1);
    let val_indent = args.get_kwarg_or_pos("indent", 2);
    let val_separators = args.get_kwarg_or_pos("separators", 3);
    let val_sort = args.get_kwarg_or_pos("sort_keys", 4);
    if args.ctx.is_get_stats {
        mark_used(&args.get_pos(0)?, true, 0);
    }
    let indent = if kind_of(&val_indent) == ValueKind::Integer {
        as_int(&val_indent)? as i32
    } else {
        -1
    };
    let sort_keys = as_bool(&val_sort)?;
    let ensure_ascii = as_bool(&val_ascii)?;
    let separators = if is_val(&val_separators, ValueKind::Array) {
        as_array(&val_separators)?
    } else {
        Vec::new()
    };
    let item_sep = if !separators.is_empty() {
        as_string(&separators[0])?.str()
    } else if indent < 0 {
        ", ".into()
    } else {
        ",".into()
    };
    let key_sep = if separators.len() > 1 {
        as_string(&separators[1])?.str()
    } else {
        ": ".into()
    };
    let mut json_str = value_to_json(&args.get_pos(0)?, indent, &item_sep, &key_sep, sort_keys);
    if ensure_ascii {
        json_str = json_ensure_ascii_preserving_format(&json_str);
    }
    Ok(mk_string(json_str))
}

fn test_in(args: &FuncArgs) -> JResult<Value> {
    args.ensure_count(2, 999)?;
    let needle = args.get_pos(0)?;
    let haystack = args.get_pos(1)?;
    if is_undefined(&haystack) {
        return Ok(mk_bool(false));
    }
    if is_val(&haystack, ValueKind::Array) {
        for item in as_array(&haystack)? {
            if equivalent(&needle, &item) {
                return Ok(mk_bool(true));
            }
        }
        return Ok(mk_bool(false));
    }
    if kind_of(&haystack) == ValueKind::String {
        if kind_of(&needle) != ValueKind::String {
            return Err(JinjaError::Raised(format!(
                "'in' test expects args[1] as string when args[0] is string, got args[1] as {}",
                type_name(&needle)
            )));
        }
        return Ok(mk_bool(
            as_string(&haystack)?
                .str()
                .contains(&as_string(&needle)?.str()),
        ));
    }
    if kind_of(&haystack) == ValueKind::Object {
        return Ok(mk_bool(has_key(&haystack, &needle)?));
    }
    Err(JinjaError::Raised(format!(
        "'in' test expects iterable as first argument, got {}",
        type_name(&haystack)
    )))
}

fn test_sameas(args: &FuncArgs) -> JResult<Value> {
    args.ensure_count(2, 999)?;
    let a = args.get_pos(0)?;
    let b = args.get_pos(1)?;
    let mut res = false;
    if !is_undefined(&a) && !is_undefined(&b) {
        if is_none(&a) && is_none(&b) {
            res = true;
        } else if kind_of(&a) == ValueKind::Boolean && kind_of(&b) == ValueKind::Boolean {
            res = as_bool(&a)? == as_bool(&b)?;
        } else if kind_of(&a) == ValueKind::Integer && kind_of(&b) == ValueKind::Integer {
            let x = as_int(&a)?;
            res = (-5..=256).contains(&x) && x == as_int(&b)?;
        } else {
            res = Rc::ptr_eq(&a, &b);
        }
    }
    Ok(mk_bool(res))
}

fn test_is_test(args: &FuncArgs) -> JResult<Value> {
    args.ensure_count(1, 1)?;
    args.ensure_val(0, ValueKind::String)?;
    let name = as_string(&args.get_pos(0)?)?.str();
    Ok(mk_bool(
        global_builtin(&format!("test_is_{name}")).is_some(),
    ))
}

/// `'name' is filter`: checks whether a filter name is registered.
/// Uses the same aliases as `apply_filter_by_name`.
fn test_is_filter(args: &FuncArgs) -> JResult<Value> {
    args.ensure_count(1, 1)?;
    args.ensure_val(0, ValueKind::String)?;
    let raw = as_string(&args.get_pos(0)?)?.str();
    let name = match raw.as_str() {
        "count" => "length",
        "d" => "default",
        "e" => "escape",
        "trim" => "strip",
        other => other,
    };
    const KINDS: &[ValueKind] = &[
        ValueKind::Integer,
        ValueKind::Real,
        ValueKind::String,
        ValueKind::Boolean,
        ValueKind::Array,
        ValueKind::Tuple,
        ValueKind::Object,
        ValueKind::None,
        ValueKind::Undefined,
    ];
    let exists = KINDS.iter().any(|k| type_builtin(*k, name).is_some());
    Ok(mk_bool(exists))
}

type BuiltinFn = fn(&FuncArgs) -> JResult<Value>;

fn opt(f: BuiltinFn) -> Option<BuiltinFn> {
    Some(f)
}

/// `global_builtins().find(name)`.
pub fn global_builtin(name: &str) -> Option<FuncHandler> {
    let f = match name {
        "raise_exception" => opt(raise_exception),
        "dict" => opt(toobject),
        "namespace" => opt(namespace_fn),
        "strftime_now" => opt(strftime_now),
        "range" => opt(range_fn),
        "tojson" => opt(tojson),
        "test_is_boolean" => opt(|a| test_type(a, &[ValueKind::Boolean])),
        "test_is_callable" => opt(|a| test_type(a, &[ValueKind::Function])),
        "test_is_odd" => opt(|a: &FuncArgs| {
            a.ensure_count(1, 1)?;
            a.ensure_val(0, ValueKind::Integer)?;
            Ok(mk_bool(as_int(&a.get_pos(0)?)? % 2 != 0))
        }),
        "test_is_even" => opt(|a: &FuncArgs| {
            a.ensure_count(1, 1)?;
            a.ensure_val(0, ValueKind::Integer)?;
            Ok(mk_bool(as_int(&a.get_pos(0)?)? % 2 == 0))
        }),
        "test_is_false" => opt(|a: &FuncArgs| {
            a.ensure_count(1, 1)?;
            let v = a.get_pos(0)?;
            Ok(mk_bool(kind_of(&v) == ValueKind::Boolean && !as_bool(&v)?))
        }),
        "test_is_true" => opt(|a: &FuncArgs| {
            a.ensure_count(1, 1)?;
            let v = a.get_pos(0)?;
            Ok(mk_bool(kind_of(&v) == ValueKind::Boolean && as_bool(&v)?))
        }),
        "test_is_divisibleby" => opt(|a: &FuncArgs| {
            a.ensure_count(2, 2)?;
            a.ensure_val(0, ValueKind::Integer)?;
            a.ensure_val(1, ValueKind::Integer)?;
            let x = as_int(&a.get_pos(0)?)?;
            let y = as_int(&a.get_pos(1)?)?;
            if y == 0 {
                return Err(JinjaError::Error("divisibleby: division by zero".into()));
            }
            Ok(mk_bool(y == -1 || x % y == 0))
        }),
        "test_is_string" => opt(|a| test_type(a, &[ValueKind::String])),
        "test_is_integer" => opt(|a| test_type(a, &[ValueKind::Integer])),
        "test_is_float" => opt(|a| test_type(a, &[ValueKind::Real])),
        "test_is_number" => opt(|a| test_type(a, &[ValueKind::Integer, ValueKind::Real])),
        "test_is_iterable" => opt(|a| {
            test_type(
                a,
                &[
                    ValueKind::Object,
                    ValueKind::Array,
                    ValueKind::String,
                    ValueKind::Undefined,
                ],
            )
        }),
        "test_is_sequence" => opt(|a| {
            test_type(
                a,
                &[
                    ValueKind::Object,
                    ValueKind::Array,
                    ValueKind::String,
                    ValueKind::Undefined,
                ],
            )
        }),
        "test_is_mapping" => opt(|a| test_type(a, &[ValueKind::Object])),
        "test_is_lower" => opt(|a: &FuncArgs| {
            a.ensure_count(1, 1)?;
            a.ensure_val(0, ValueKind::String)?;
            Ok(mk_bool(as_string(&a.get_pos(0)?)?.is_lowercase()))
        }),
        "test_is_upper" => opt(|a: &FuncArgs| {
            a.ensure_count(1, 1)?;
            a.ensure_val(0, ValueKind::String)?;
            Ok(mk_bool(as_string(&a.get_pos(0)?)?.is_uppercase()))
        }),
        "test_is_none" => opt(|a| test_type(a, &[ValueKind::None])),
        "test_is_defined" => opt(|a: &FuncArgs| {
            a.ensure_count(1, 1)?;
            Ok(mk_bool(!is_undefined(&a.get_pos(0)?)))
        }),
        "test_is_undefined" => opt(|a| test_type(a, &[ValueKind::Undefined])),
        "test_is_eq" | "test_is_equalto" => opt(|a| test_compare(a, ValueCompareOp::Eq)),
        "test_is_ge" => opt(|a| test_compare(a, ValueCompareOp::Ge)),
        "test_is_gt" | "test_is_greaterthan" => opt(|a| test_compare(a, ValueCompareOp::Gt)),
        "test_is_lt" | "test_is_lessthan" => opt(|a| test_compare(a, ValueCompareOp::Lt)),
        "test_is_ne" => opt(|a| test_compare(a, ValueCompareOp::Ne)),
        "test_is_in" => opt(test_in),
        "test_is_test" => opt(test_is_test),
        "test_is_sameas" => opt(test_sameas),
        // This engine has no Markup type; strings are the safe/escaped values.
        "test_is_escaped" => opt(|a| test_type(a, &[ValueKind::String])),
        "test_is_filter" => opt(test_is_filter),
        _ => None,
    }?;
    Some(h(f))
}

/// Kept as a collection view for API parity with C++ `global_builtins()`.
pub fn global_builtins() -> Vec<(&'static str, FuncHandler)> {
    NAMES
        .iter()
        .filter_map(|n| global_builtin(n).map(|f| (*n, f)))
        .collect()
}

const NAMES: &[&str] = &[
    "raise_exception",
    "dict",
    "namespace",
    "strftime_now",
    "range",
    "tojson",
    "test_is_boolean",
    "test_is_callable",
    "test_is_odd",
    "test_is_even",
    "test_is_false",
    "test_is_true",
    "test_is_divisibleby",
    "test_is_string",
    "test_is_integer",
    "test_is_float",
    "test_is_number",
    "test_is_iterable",
    "test_is_sequence",
    "test_is_mapping",
    "test_is_lower",
    "test_is_upper",
    "test_is_none",
    "test_is_defined",
    "test_is_undefined",
    "test_is_eq",
    "test_is_equalto",
    "test_is_ge",
    "test_is_gt",
    "test_is_greaterthan",
    "test_is_lt",
    "test_is_lessthan",
    "test_is_ne",
    "test_is_in",
    "test_is_test",
    "test_is_sameas",
    "test_is_escaped",
    "test_is_filter",
];

fn namespace_fn(args: &FuncArgs) -> JResult<Value> {
    let out = toobject(args)?;
    if let ValueData::Object { has_builtins, .. } = &mut out.borrow_mut().data {
        *has_builtins = false;
    }
    Ok(out)
}

pub fn find_builtin_func(_ctx: &Context, name: &str, input: &Value) -> Option<Value> {
    if _ctx.is_get_stats {
        mark_used(input, false, 0);
        input.borrow_mut().stats.ops.insert(name.to_string());
    }
    let has_builtins = match &input.borrow().data {
        ValueData::Object { has_builtins, .. } => *has_builtins,
        _ => true,
    };
    let handler = if has_builtins {
        type_builtin(kind_of(input), name)
    } else {
        None
    }?;
    Some(mk_func(name, handler, Some(input.clone())))
}

pub fn try_builtin_func(
    ctx: &Context,
    name: &str,
    input: &Value,
    undef_on_missing: bool,
) -> JResult<Value> {
    if let Some(func) = find_builtin_func(ctx, name, input) {
        return Ok(func);
    }
    if undef_on_missing {
        return Ok(mk_undefined(name));
    }
    Err(JinjaError::Error(format!(
        "Unknown (built-in) filter '{}' for type {}",
        name,
        type_name(input)
    )))
}

pub fn apply_filter_by_name(
    ctx: &Context,
    mut filter_id: String,
    mut input: Value,
    args: &FuncArgs,
) -> JResult<Value> {
    filter_id = match filter_id.as_str() {
        "count" => "length".into(),
        "d" => "default".into(),
        "e" => "escape".into(),
        "trim" => "strip".into(),
        other => other.into(),
    };
    let needs_string = [
        "capitalize",
        "lower",
        "replace",
        "strip",
        "title",
        "upper",
        "wordcount",
    ]
    .contains(&filter_id.as_str());
    if !is_undefined(&input) && kind_of(&input) != ValueKind::String && needs_string {
        input = mk_jstring(as_string(&input)?);
    }
    invoke(&try_builtin_func(ctx, &filter_id, &input, false)?, args)
}

fn type_builtin(kind: ValueKind, name: &str) -> Option<FuncHandler> {
    let f = match kind {
        ValueKind::Integer => match name {
            "default" => opt(default_value),
            "abs" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::Integer)?;
                Ok(mk_int(as_int(&a.get_pos(0)?)?.abs()))
            }),
            "int" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::Integer)?;
                Ok(mk_int(as_int(&a.get_pos(0)?)?))
            }),
            "float" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::Integer)?;
                Ok(mk_float(as_int(&a.get_pos(0)?)? as f64))
            }),
            "safe" | "string" | "tojson" => opt(tojson),
            _ => None,
        },
        ValueKind::Real => match name {
            "default" => opt(default_value),
            "abs" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::Real)?;
                Ok(mk_float(as_float(&a.get_pos(0)?)?.abs()))
            }),
            "int" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::Real)?;
                Ok(mk_int(float_to_int(as_float(&a.get_pos(0)?)?)))
            }),
            "float" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::Real)?;
                Ok(mk_float(as_float(&a.get_pos(0)?)?))
            }),
            "safe" | "string" => opt(float_tostring),
            "tojson" => opt(tojson),
            _ => None,
        },
        ValueKind::Boolean => match name {
            "default" => opt(default_value),
            "int" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::Boolean)?;
                Ok(mk_int(if as_bool(&a.get_pos(0)?)? { 1 } else { 0 }))
            }),
            "float" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::Boolean)?;
                Ok(mk_float(if as_bool(&a.get_pos(0)?)? { 1.0 } else { 0.0 }))
            }),
            "safe" | "string" => opt(bool_tostring),
            "tojson" => opt(tojson),
            _ => None,
        },
        ValueKind::String => match name {
            "default" => opt(default_value),
            "upper" => opt(string_upper),
            "lower" => opt(string_lower),
            "strip" => opt(string_strip),
            "rstrip" => opt(string_rstrip),
            "lstrip" => opt(string_lstrip),
            "title" => opt(string_title),
            "capitalize" => opt(string_capitalize),
            "length" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::String)?;
                Ok(mk_int(as_string(&a.get_pos(0)?)?.length() as i64))
            }),
            "startswith" | "endswith" => opt(string_start_end),
            "split" => opt(string_split),
            "rsplit" => opt(string_rsplit),
            "replace" => opt(string_replace),
            "format" => opt(string_format),
            "int" => opt(string_int),
            "float" => opt(string_float),
            "string" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::String)?;
                Ok(mk_jstring(as_string(&a.get_pos(0)?)?))
            }),
            "slice" => opt(string_slice),
            "safe" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::String)?;
                Ok(a.get_pos(0)?)
            }),
            "tojson" => opt(tojson),
            "indent" => opt(string_indent),
            "join" => opt(string_join),
            _ => None,
        },
        ValueKind::Array | ValueKind::Tuple => match name {
            "default" => opt(default_value),
            "list" => opt(array_list),
            "first" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::Array)?;
                let arr = as_array(&a.get_pos(0)?)?;
                Ok(arr.first().cloned().unwrap_or_else(|| mk_undefined("")))
            }),
            "last" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::Array)?;
                let arr = as_array(&a.get_pos(0)?)?;
                Ok(arr.last().cloned().unwrap_or_else(|| mk_undefined("")))
            }),
            "length" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::Array)?;
                Ok(mk_int(as_array(&a.get_pos(0)?)?.len() as i64))
            }),
            "slice" => opt(array_slice),
            "selectattr" => opt(|a| select_or_reject(a, false, true)),
            "select" => opt(|a| select_or_reject(a, false, false)),
            "rejectattr" => opt(|a| select_or_reject(a, true, true)),
            "reject" => opt(|a| select_or_reject(a, true, false)),
            "sum" => opt(array_sum),
            "join" => opt(array_join),
            "string" => opt(array_string),
            "tojson" => opt(tojson),
            "map" => opt(array_map),
            "append" => opt(array_append),
            "pop" => opt(array_pop),
            "sort" => opt(array_sort),
            "reverse" => opt(array_reverse),
            "min" => opt(|a| array_minmax(a, false)),
            "max" => opt(|a| array_minmax(a, true)),
            "unique" => opt(array_unique),
            _ => None,
        },
        ValueKind::Object => match name {
            "get" => opt(object_get),
            "keys" | "list" => opt(object_keys),
            "values" => opt(object_values),
            "items" => opt(object_items),
            "tojson" => opt(tojson),
            "string" => opt(object_string),
            "length" => opt(|a: &FuncArgs| {
                a.ensure_count(1, 1)?;
                a.ensure_val(0, ValueKind::Object)?;
                Ok(mk_int(as_ordered_object(&a.get_pos(0)?)?.len() as i64))
            }),
            "dictsort" => opt(object_dictsort),
            "join" => opt(object_join),
            _ => None,
        },
        ValueKind::None => match name {
            "default" => opt(default_value),
            "tojson" => opt(tojson),
            "string" | "safe" => opt(|_: &FuncArgs| Ok(mk_string("None"))),
            "items" | "map" | "reject" | "rejectattr" | "select" | "selectattr" | "unique" => {
                opt(empty_array)
            }
            _ => None,
        },
        ValueKind::Undefined => match name {
            "default" => opt(default_value),
            "first" | "last" | "max" | "min" => opt(|a: &FuncArgs| {
                a.ensure_kwargs(&[])?;
                a.ensure_count(1, 1)?;
                Ok(mk_undefined(""))
            }),
            "length" | "sum" | "wordcount" => opt(empty_int),
            "join" => opt(|a: &FuncArgs| {
                a.ensure_kwargs(&["d", "attribute"])?;
                a.ensure_count(1, 3)?;
                Ok(mk_string(""))
            }),
            "strip" => opt(|a: &FuncArgs| {
                a.ensure_kwargs(&["chars"])?;
                a.ensure_count(1, 2)?;
                Ok(mk_string(""))
            }),
            "capitalize" | "lower" | "title" | "upper" | "safe" | "string" | "replace"
            | "truncate" => opt(empty_string),
            "items" | "list" | "map" | "reject" | "rejectattr" | "reverse" | "select"
            | "selectattr" | "sort" | "unique" => opt(empty_array),
            _ => None,
        },
        _ => None,
    }?;
    Some(h(f))
}

fn float_tostring(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::Real)?;
    Ok(mk_string(float_repr(as_float(&a.get_pos(0)?)?)))
}

fn bool_tostring(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::Boolean)?;
    Ok(mk_string(if as_bool(&a.get_pos(0)?)? {
        "True"
    } else {
        "False"
    }))
}

fn empty_array(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&[])?;
    a.ensure_count(1, 1)?;
    Ok(mk_array())
}

fn empty_int(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&[])?;
    a.ensure_count(1, 1)?;
    Ok(mk_int(0))
}

fn empty_string(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&[])?;
    a.ensure_count(1, 1)?;
    Ok(mk_string(""))
}

fn string_upper(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&[])?;
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::String)?;
    let mut s = as_string(&a.get_pos(0)?)?;
    Ok(mk_jstring(s.uppercase()))
}

fn string_lower(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&[])?;
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::String)?;
    let mut s = as_string(&a.get_pos(0)?)?;
    Ok(mk_jstring(s.lowercase()))
}

fn string_title(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&[])?;
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::String)?;
    let mut s = as_string(&a.get_pos(0)?)?;
    Ok(mk_jstring(s.titlecase()))
}

fn string_capitalize(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&[])?;
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::String)?;
    let mut s = as_string(&a.get_pos(0)?)?;
    Ok(mk_jstring(s.capitalize()))
}

fn strip_args(a: &FuncArgs, allowed: &[&str]) -> JResult<(JString, Option<String>, bool, bool)> {
    a.ensure_kwargs(allowed)?;
    a.ensure_count(1, 2)?;
    let input = a.get_pos(0)?;
    if kind_of(&input) != ValueKind::String {
        return Err(JinjaError::Raised(
            "strip() first argument must be a string".into(),
        ));
    }
    let chars = a.get_kwarg_or_pos("chars", 1);
    let chars = if is_undefined(&chars) {
        None
    } else {
        Some(as_string(&chars)?.str())
    };
    Ok((as_string(&input)?, chars, true, true))
}

fn string_strip(a: &FuncArgs) -> JResult<Value> {
    let (mut s, chars, l, r) = strip_args(a, &["chars"])?;
    Ok(mk_jstring(s.strip(l, r, chars.as_deref())))
}

fn string_rstrip(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&["chars"])?;
    a.ensure_count(1, 2)?;
    a.ensure_val(0, ValueKind::String)?;
    let mut s = as_string(&a.get_pos(0)?)?;
    let chars = a.get_kwarg_or_pos("chars", 1);
    let chars = if is_undefined(&chars) {
        None
    } else {
        Some(as_string(&chars)?.str())
    };
    Ok(mk_jstring(s.strip(false, true, chars.as_deref())))
}

fn string_lstrip(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&["chars"])?;
    a.ensure_count(1, 2)?;
    a.ensure_val(0, ValueKind::String)?;
    let mut s = as_string(&a.get_pos(0)?)?;
    let chars = a.get_kwarg_or_pos("chars", 1);
    let chars = if is_undefined(&chars) {
        None
    } else {
        Some(as_string(&chars)?.str())
    };
    Ok(mk_jstring(s.strip(true, false, chars.as_deref())))
}

fn string_start_end(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(2, 2)?;
    a.ensure_val(0, ValueKind::String)?;
    a.ensure_val(1, ValueKind::String)?;
    let s = as_string(&a.get_pos(0)?)?.str();
    let part = as_string(&a.get_pos(1)?)?.str();
    let is_start = a.func_name == "startswith";
    Ok(mk_bool(if is_start {
        s.starts_with(&part)
    } else {
        s.ends_with(&part)
    }))
}

fn string_split(a: &FuncArgs) -> JResult<Value> {
    split_impl(a, false)
}

fn string_rsplit(a: &FuncArgs) -> JResult<Value> {
    split_impl(a, true)
}

fn split_impl(a: &FuncArgs, reverse: bool) -> JResult<Value> {
    a.ensure_count(1, 3)?;
    let input = a.get_pos(0)?;
    if kind_of(&input) != ValueKind::String {
        return Err(JinjaError::Raised(format!(
            "{}() first argument must be a string",
            if reverse { "rsplit" } else { "split" }
        )));
    }
    let original = as_string(&input)?;
    let mut s = original.str();
    let delim = if a.count() > 1 {
        as_string(&a.get_pos(1)?)?.str()
    } else {
        " ".into()
    };
    if delim.is_empty() {
        return Err(JinjaError::Raised("empty separator".into()));
    }
    let mut maxsplit = if a.count() > 2 {
        as_int(&a.get_pos(2)?)?
    } else {
        -1
    };
    let out = mk_array();
    while maxsplit != 0 {
        let found = if reverse {
            s.rfind(&delim)
        } else {
            s.find(&delim)
        };
        let Some(pos) = found else { break };
        let token = if reverse {
            s[pos + delim.len()..].to_string()
        } else {
            s[..pos].to_string()
        };
        push_array(&out, mk_string(token));
        if reverse {
            s.truncate(pos);
        } else {
            s.drain(..pos + delim.len());
        }
        maxsplit -= 1;
    }
    let mut js = original.clone();
    js.mark_input_based_on(&original);
    let res = mk_jstring(JString::new(s));
    if let ValueData::String { val_str } = &mut res.borrow_mut().data {
        *val_str = js;
    }
    if reverse {
        if let ValueData::Array { items, .. } = &mut out.borrow_mut().data {
            items.reverse();
        }
    }
    push_array(&out, res);
    Ok(out)
}

fn string_replace(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(3, 4)?;
    a.ensure_val(0, ValueKind::String)?;
    a.ensure_val(1, ValueKind::String)?;
    a.ensure_val(2, ValueKind::String)?;
    if a.count() > 3 {
        a.ensure_val(3, ValueKind::Integer)?;
    }
    let original = as_string(&a.get_pos(0)?)?;
    let mut s = original.str();
    let old = as_string(&a.get_pos(1)?)?.str();
    let new = as_string(&a.get_pos(2)?)?.str();
    let count = if a.count() > 3 {
        as_int(&a.get_pos(3)?)?
    } else {
        -1
    };
    // Python/Jinja str.replace: a negative count means "all occurrences".
    let mut done = 0i64;
    if old != new {
        if old.is_empty() {
            let mut res = String::new();
            for c in s.chars() {
                if count < 0 || done < count {
                    res.push_str(&new);
                    done += 1;
                }
                res.push(c);
            }
            if count < 0 || done < count {
                res.push_str(&new);
            }
            s = res;
        } else {
            let mut pos = 0;
            while count < 0 || done < count {
                let Some(rel) = s[pos..].find(&old) else {
                    break;
                };
                let at = pos + rel;
                s.replace_range(at..at + old.len(), &new);
                pos = at + new.len();
                done += 1;
            }
        }
    }
    let val = mk_jstring(JString::new(s));
    {
        let mut js = as_string(&val)?;
        js.mark_input_based_on(&original);
        if let ValueData::String { val_str } = &mut val.borrow_mut().data {
            *val_str = js;
        }
    }
    Ok(val)
}

fn string_format(a: &FuncArgs) -> JResult<Value> {
    let input = a.get_pos(0)?;
    if kind_of(&input) != ValueKind::String {
        return Err(JinjaError::Raised(
            "format() first argument must be a string".into(),
        ));
    }
    let fmt = as_string(&input)?;
    let fmt_is_input = fmt.all_parts_are_input();
    let s = fmt.str();
    let mut result = JString::default();
    let mut literal = String::new();
    let mut arg_idx = 1;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '{' {
            literal.push(c);
            continue;
        }
        if chars.peek() != Some(&'}') {
            return Err(JinjaError::NotImplemented(
                "format() only supports simple '{}' placeholders".into(),
            ));
        }
        chars.next();
        if !literal.is_empty() {
            result.parts.push(StringPart {
                is_input: fmt_is_input,
                val: std::mem::take(&mut literal),
            });
        }
        let arg = as_string(&a.get_pos(arg_idx)?)?;
        result.parts.extend(arg.parts);
        arg_idx += 1;
    }
    if !literal.is_empty() {
        result.parts.push(StringPart {
            is_input: fmt_is_input,
            val: literal,
        });
    }
    Ok(mk_jstring(result))
}

fn parse_int_base(s: &str, base: i32) -> Option<i32> {
    let (negative, rest) = if let Some(r) = s.strip_prefix('-') {
        (true, r)
    } else if let Some(r) = s.strip_prefix('+') {
        (false, r)
    } else {
        (false, s)
    };
    let mut base = base;
    let mut rest = rest;
    if base == 0 {
        if let Some(r) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
            base = 16;
            rest = r;
        } else if let Some(r) = rest.strip_prefix("0o").or_else(|| rest.strip_prefix("0O")) {
            base = 8;
            rest = r;
        } else if let Some(r) = rest.strip_prefix("0b").or_else(|| rest.strip_prefix("0B")) {
            base = 2;
            rest = r;
        } else {
            base = 10;
        }
    }
    if !(2..=36).contains(&base) || rest.is_empty() {
        return None;
    }
    let mut value: i64 = 0;
    for c in rest.bytes() {
        let digit = match c {
            b'0'..=b'9' => (c - b'0') as i64,
            b'a'..=b'z' => (c - b'a' + 10) as i64,
            b'A'..=b'Z' => (c - b'A' + 10) as i64,
            _ => return None,
        };
        if digit >= base as i64 {
            return None;
        }
        value = value.checked_mul(base as i64)?.checked_add(digit)?;
        if value > i32::MAX as i64 + if negative { 1 } else { 0 } {
            return None;
        }
    }
    if negative {
        if value == i32::MAX as i64 + 1 {
            Some(i32::MIN)
        } else {
            Some((-value) as i32)
        }
    } else {
        Some(value as i32)
    }
}

fn lenient_float(s: &str) -> Option<f64> {
    let mut end = s.len();
    while end > 0 {
        if let Ok(v) = s[..end].parse::<f64>() {
            return Some(v);
        }
        end -= 1;
    }
    None
}

fn string_int(a: &FuncArgs) -> JResult<Value> {
    let input = a.get_pos(0)?;
    let default = a.get_kwarg_or_pos("default", 1);
    let base_val = a.get_kwarg_or_pos("base", 2);
    let base = if is_undefined(&base_val) {
        10
    } else {
        as_int(&base_val)? as i32
    };
    if base != 0 && !(2..=36).contains(&base) {
        return Err(JinjaError::Raised(
            "int() base must be 0 or between 2 and 36".into(),
        ));
    }
    if kind_of(&input) != ValueKind::String {
        return Err(JinjaError::Raised(
            "int() first argument must be a string".into(),
        ));
    }
    let s = as_string(&input)?.str();
    Ok(mk_int(
        parse_int_base(&s, base).map(i64::from).unwrap_or_else(|| {
            if is_undefined(&default) {
                0
            } else {
                as_int(&default).unwrap_or(0)
            }
        }),
    ))
}

fn string_float(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::String)?;
    let default = a.get_kwarg_or_pos("default", 1);
    let s = as_string(&a.get_pos(0)?)?.str();
    Ok(mk_float(lenient_float(&s).unwrap_or_else(|| {
        if is_undefined(&default) {
            0.0
        } else {
            as_float(&default).unwrap_or(0.0)
        }
    })))
}

fn string_slice(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 4)?;
    a.ensure_val(0, ValueKind::String)?;
    if a.count() > 1 {
        a.ensure_val(1, ValueKind::Integer)?;
    }
    let (start, stop, step) = slice_bounds(a)?;
    let input = a.get_pos(0)?;
    let s = as_string(&input)?.str();
    let val = mk_jstring(JString::new(slice_string(&s, start, stop, step)));
    {
        let mut js = as_string(&val)?;
        js.mark_input_based_on(&as_string(&input)?);
        if let ValueData::String { val_str } = &mut val.borrow_mut().data {
            *val_str = js;
        }
    }
    Ok(val)
}

fn slice_bounds(a: &FuncArgs) -> JResult<(i64, i64, i64)> {
    let count = a.count();
    let arg0 = a.get_pos(1)?;
    let arg1 = a.get_pos_or(2, mk_undefined(""));
    let arg2 = a.get_pos_or(3, mk_undefined(""));
    if count == 2 {
        Ok((0, as_int(&arg0)?, 1))
    } else if count == 3 {
        Ok((as_int(&arg0)?, as_int(&arg1)?, 1))
    } else {
        Ok((as_int(&arg0)?, as_int(&arg1)?, as_int(&arg2)?))
    }
}

fn string_indent(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 4)?;
    let input = a.get_pos(0)?;
    let width = a.get_kwarg_or_pos("width", 1);
    let first = as_bool(&a.get_kwarg_or_pos("first", 2))?;
    let blank = as_bool(&a.get_kwarg_or_pos("blank", 3))?;
    if kind_of(&input) != ValueKind::String {
        return Err(JinjaError::Raised(
            "indent() first argument must be a string".into(),
        ));
    }
    let indent = if kind_of(&width) == ValueKind::Integer {
        " ".repeat(as_int(&width)?.max(0) as usize)
    } else if kind_of(&width) == ValueKind::String {
        as_string(&width)?.str()
    } else {
        "    ".into()
    };
    let s = as_string(&input)?.str();
    let mut indented = String::new();
    for line in s.lines() {
        if !indented.is_empty() {
            indented.push('\n');
        }
        if (indented.is_empty() && first) || (!line.is_empty() || blank) {
            indented.push_str(&indent);
        }
        indented.push_str(line);
    }
    if s.ends_with('\n') {
        indented.push('\n');
        if blank {
            indented.push_str(&indent);
        }
    }
    let val = mk_jstring(JString::new(indented));
    {
        let mut js = as_string(&val)?;
        js.mark_input_based_on(&as_string(&input)?);
        if let ValueData::String { val_str } = &mut val.borrow_mut().data {
            *val_str = js;
        }
    }
    Ok(val)
}

/// `separator.join(iterable)`: the string value is the separator.
fn string_join(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&["attribute"])?;
    a.ensure_count(2, 3)?;
    let sep_val = a.get_pos(0)?;
    if kind_of(&sep_val) != ValueKind::String {
        return Err(JinjaError::Raised(
            "join() first argument must be a string".into(),
        ));
    }
    let sep = as_string(&sep_val)?.str();
    let input = a.get_pos(1)?;
    if !is_val(&input, ValueKind::Array) {
        return Err(JinjaError::Raised(
            "join() second argument must be an array".into(),
        ));
    }
    let attribute = a.get_kwarg_or_pos("attribute", 2);
    let undef = mk_undefined("");
    let arr = as_array(&input)?;
    let mut result = String::new();
    for (i, item) in arr.iter().enumerate() {
        let mut val = item.clone();
        if !is_undefined(&attribute) {
            val = get_attribute(&val, &attribute, undef.clone());
        }
        if !is_val(&val, ValueKind::String)
            && !is_val(&val, ValueKind::Integer)
            && !is_val(&val, ValueKind::Real)
        {
            return Err(JinjaError::Raised(
                "join() can only join arrays of strings or numerics".into(),
            ));
        }
        result.push_str(&as_string(&val)?.str());
        if i < arr.len() - 1 {
            result.push_str(&sep);
        }
    }
    Ok(mk_string(result))
}

fn array_list(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::Array)?;
    Ok(mk_array_from(as_array(&a.get_pos(0)?)?))
}

fn array_slice(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 4)?;
    a.ensure_val(0, ValueKind::Array)?;
    if a.count() > 1 {
        a.ensure_val(1, ValueKind::Integer)?;
    }
    let (start, stop, step) = slice_bounds(a)?;
    let input = a.get_pos(0)?;
    let sliced = slice_generic(&as_array(&input)?, start, stop, step);
    Ok(if kind_of(&input) == ValueKind::Tuple {
        mk_tuple(sliced)
    } else {
        mk_array_from(sliced)
    })
}

fn array_sum(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&["attribute", "start"])?;
    a.ensure_count(1, 3)?;
    a.ensure_val(0, ValueKind::Array)?;
    let undef = mk_undefined("");
    let attribute = a.get_kwarg_or_pos("attribute", 1);
    let start = a.get_kwarg_or_pos("start", 2);
    let mut is_float = kind_of(&start) == ValueKind::Real;
    let mut sum = if is_undefined(&start) {
        0.0
    } else {
        as_float(&start)?
    };
    for item in as_array(&a.get_pos(0)?)? {
        let v = if is_undefined(&attribute) {
            item.clone()
        } else {
            get_attribute(&item, &attribute, undef.clone())
        };
        if is_undefined(&v) && !is_undefined(&attribute) {
            return Err(JinjaError::Raised(format!(
                "sum: {} has no attribute '{}'",
                type_name(&item),
                as_string(&attribute)?.str()
            )));
        }
        if !is_val(&v, ValueKind::Integer) && !is_val(&v, ValueKind::Real) {
            return Err(JinjaError::Raised(format!(
                "sum: unsupported operand type {}",
                type_name(&v)
            )));
        }
        is_float = is_float || kind_of(&v) == ValueKind::Real;
        sum += as_float(&v)?;
    }
    Ok(if is_float {
        mk_float(sum)
    } else {
        mk_int(float_to_int(sum))
    })
}

fn array_join(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&["d", "attribute"])?;
    a.ensure_count(1, 3)?;
    let input = a.get_pos(0)?;
    if !is_val(&input, ValueKind::Array) {
        return Err(JinjaError::Raised(
            "join() first argument must be an array".into(),
        ));
    }
    let delim_val = a.get_kwarg_or_pos("d", 1);
    let attribute = a.get_kwarg_or_pos("attribute", 2);
    let undef = mk_undefined("");
    let arr = as_array(&input)?;
    let delim = if is_undefined(&delim_val) {
        String::new()
    } else {
        as_string(&delim_val)?.str()
    };
    let mut result = String::new();
    for (i, item) in arr.iter().enumerate() {
        let mut val = item.clone();
        if !is_undefined(&attribute) {
            val = get_attribute(&val, &attribute, undef.clone());
        }
        if !is_val(&val, ValueKind::String)
            && !is_val(&val, ValueKind::Integer)
            && !is_val(&val, ValueKind::Real)
        {
            return Err(JinjaError::Raised(
                "join() can only join arrays of strings or numerics".into(),
            ));
        }
        result.push_str(&as_string(&val)?.str());
        if i < arr.len() - 1 {
            result.push_str(&delim);
        }
    }
    Ok(mk_string(result))
}

fn array_string(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::Array)?;
    if a.ctx.is_get_stats {
        mark_used(&a.get_pos(0)?, true, 0);
    }
    Ok(mk_jstring(as_string(&a.get_pos(0)?)?))
}

fn array_map(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(2, 999)?;
    let val = a.get_pos(0)?;
    if !is_val(&val, ValueKind::Array) {
        return Err(JinjaError::Raised(
            "map: first argument must be an array".into(),
        ));
    }
    if kind_of(&a.get_args()[1]) != ValueKind::Keyword {
        let filter = a.get_pos(1)?;
        if kind_of(&filter) != ValueKind::String {
            return Err(JinjaError::Raised(
                "map: filter name must be a string".into(),
            ));
        }
        let mut filter_args = FuncArgs::new(a.ctx);
        for arg in a.get_args().iter().skip(2) {
            filter_args.push_back(arg.clone());
        }
        let filter_id = as_string(&filter)?.str();
        let out = mk_array();
        for item in as_array(&val)? {
            let mapped = apply_filter_by_name(a.ctx, filter_id.clone(), item, &filter_args)?;
            push_array(&out, mapped);
        }
        return Ok(if kind_of(&val) == ValueKind::Tuple {
            mk_tuple(as_array(&out)?)
        } else {
            out
        });
    }
    let attribute = a.get_kwarg_or_pos("attribute", 1);
    let default = a.get_kwarg("default", mk_undefined(""));
    let out = mk_array();
    for item in as_array(&val)? {
        push_array(&out, get_attribute(&item, &attribute, default.clone()));
    }
    Ok(if kind_of(&val) == ValueKind::Tuple {
        mk_tuple(as_array(&out)?)
    } else {
        out
    })
}

fn array_append(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(2, 2)?;
    let val = a.get_pos(0)?;
    if !is_val(&val, ValueKind::Array) {
        return Err(JinjaError::Raised(
            "append: first argument must be an array".into(),
        ));
    }
    push_array(&val, a.get_pos(1)?);
    Ok(val)
}

fn array_pop(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 2)?;
    a.ensure_val(0, ValueKind::Array)?;
    if a.count() == 2 {
        a.ensure_val(1, ValueKind::Integer)?;
    }
    let index = if a.count() == 2 {
        as_int(&a.get_pos(1)?)?
    } else {
        -1
    };
    pop_array(&a.get_pos(0)?, index)
}

fn array_sort(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 4)?;
    let val = a.get_pos(0)?;
    if !is_val(&val, ValueKind::Array) {
        return Err(JinjaError::Raised(
            "sort: first argument must be an array".into(),
        ));
    }
    let reverse = as_bool(&a.get_kwarg_or_pos("reverse", 1))?;
    let attribute = a.get_kwarg_or_pos("attribute", 3);
    let undef = mk_undefined("");
    let mut arr = as_array(&val)?;
    arr.sort_by(|x, y| {
        let (a, b) = if is_undefined(&attribute) {
            (x.clone(), y.clone())
        } else {
            (
                get_attribute(x, &attribute, undef.clone()),
                get_attribute(y, &attribute, undef.clone()),
            )
        };
        if value_compare(
            &a,
            &b,
            if reverse {
                ValueCompareOp::Gt
            } else {
                ValueCompareOp::Lt
            },
        ) {
            std::cmp::Ordering::Less
        } else if value_compare(
            &a,
            &b,
            if reverse {
                ValueCompareOp::Lt
            } else {
                ValueCompareOp::Gt
            },
        ) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    Ok(if kind_of(&val) == ValueKind::Tuple {
        mk_tuple(arr)
    } else {
        mk_array_from(arr)
    })
}

fn array_reverse(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::Array)?;
    let val = a.get_pos(0)?;
    let mut arr = as_array(&val)?;
    arr.reverse();
    Ok(if kind_of(&val) == ValueKind::Tuple {
        mk_tuple(arr)
    } else {
        mk_array_from(arr)
    })
}

fn array_minmax(a: &FuncArgs, is_max: bool) -> JResult<Value> {
    a.ensure_count(1, 4)?;
    a.ensure_val(0, ValueKind::Array)?;
    let attribute = a.get_kwarg_or_pos("attribute", 2);
    let undef = mk_undefined("");
    let arr = as_array(&a.get_pos(0)?)?;
    if arr.is_empty() {
        return Ok(undef);
    }
    let mut result = arr[0].clone();
    for item in &arr {
        let (x, y) = if is_undefined(&attribute) {
            (item.clone(), result.clone())
        } else {
            (
                get_attribute(item, &attribute, undef.clone()),
                get_attribute(&result, &attribute, undef.clone()),
            )
        };
        let op = if is_max {
            ValueCompareOp::Gt
        } else {
            ValueCompareOp::Lt
        };
        if value_compare(&x, &y, op) {
            result = item.clone();
        }
    }
    Ok(result)
}

fn array_unique(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 3)?;
    a.ensure_val(0, ValueKind::Array)?;
    let val = a.get_pos(0)?;
    let attribute = a.get_kwarg_or_pos("attribute", 2);
    let undef = mk_undefined("");
    let mut result = Vec::new();
    let mut seen: Vec<Value> = Vec::new();
    for item in as_array(&val)? {
        let key = if is_undefined(&attribute) {
            item.clone()
        } else {
            get_attribute(&item, &attribute, undef.clone())
        };
        let dup = seen
            .iter()
            .any(|s| value_compare(s, &key, ValueCompareOp::Eq));
        if !dup {
            seen.push(key);
            result.push(item);
        }
    }
    Ok(mk_array_from(result))
}

fn object_get(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(2, 3)?;
    a.ensure_val(0, ValueKind::Object)?;
    a.ensure_val(1, ValueKind::String)?;
    let default = if a.count() == 3 {
        a.get_pos(2)?
    } else {
        mk_none()
    };
    let obj = a.get_pos(0)?;
    let key = a.get_pos(1)?;
    Ok(at(&obj, &key, Some(default))?.unwrap_or_else(mk_none))
}

fn object_keys(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::Object)?;
    let out = mk_array();
    for (k, _) in as_ordered_object(&a.get_pos(0)?)? {
        push_array(&out, k);
    }
    Ok(out)
}

fn object_values(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::Object)?;
    let out = mk_array();
    for (_, v) in as_ordered_object(&a.get_pos(0)?)? {
        push_array(&out, v);
    }
    Ok(out)
}

fn object_items(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::Object)?;
    let out = mk_array();
    for (k, v) in as_ordered_object(&a.get_pos(0)?)? {
        push_array(&out, mk_tuple(vec![k, v]));
    }
    Ok(out)
}

fn object_string(a: &FuncArgs) -> JResult<Value> {
    a.ensure_count(1, 1)?;
    a.ensure_val(0, ValueKind::Object)?;
    if a.ctx.is_get_stats {
        mark_used(&a.get_pos(0)?, true, 0);
    }
    Ok(mk_jstring(as_string(&a.get_pos(0)?)?))
}

fn object_dictsort(a: &FuncArgs) -> JResult<Value> {
    let val = a.get_pos(0)?;
    let reverse = as_bool(&a.get_kwarg_or_pos("reverse", 3))?;
    let by = a.get_kwarg_or_pos("by", 2);
    let by_value = kind_of(&by) == ValueKind::String && as_string(&by)?.str() == "value";
    let pairs = as_ordered_object(&val)?;
    let out = mk_object();
    let mut sorted = pairs;
    sorted.sort_by(|x, y| {
        let (a, b) = if by_value {
            (x.1.clone(), y.1.clone())
        } else {
            (x.0.clone(), y.0.clone())
        };
        if value_compare(
            &a,
            &b,
            if reverse {
                ValueCompareOp::Gt
            } else {
                ValueCompareOp::Lt
            },
        ) {
            std::cmp::Ordering::Less
        } else if value_compare(
            &a,
            &b,
            if reverse {
                ValueCompareOp::Lt
            } else {
                ValueCompareOp::Gt
            },
        ) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    for (k, v) in sorted {
        insert(&out, k, v).ok();
    }
    Ok(out)
}

/// Join an object's keys with a separator (Python `sep.join(dict)`).
fn object_join(a: &FuncArgs) -> JResult<Value> {
    a.ensure_kwargs(&["d", "attribute"])?;
    a.ensure_count(1, 3)?;
    let input = a.get_pos(0)?;
    if kind_of(&input) != ValueKind::Object {
        return Err(JinjaError::Raised(
            "join() first argument must be an object".into(),
        ));
    }
    let delim_val = a.get_kwarg_or_pos("d", 1);
    let delim = if is_undefined(&delim_val) {
        String::new()
    } else {
        as_string(&delim_val)?.str()
    };
    let pairs = as_ordered_object(&input)?;
    let mut result = String::new();
    for (i, (key, _)) in pairs.iter().enumerate() {
        result.push_str(&as_string(key)?.str());
        if i < pairs.len() - 1 {
            result.push_str(&delim);
        }
    }
    Ok(mk_string(result))
}

// ---------------------------------------------------------------------------
// time formatting (strftime_now)
// ---------------------------------------------------------------------------

fn format_time(fmt: &str, epoch: i64) -> String {
    let days = epoch.div_euclid(86400);
    let secs = epoch.rem_euclid(86400);
    let (year, month, day) = civil_from_days(days);
    let hour = secs / 3600;
    let minute = (secs % 3600) / 60;
    let second = secs % 60;
    let weekday = (days + 4).rem_euclid(7) as usize; // 0 = Sunday
    let month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let day_names = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    let mut out = String::new();
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let Some(spec) = chars.next() else {
            out.push('%');
            break;
        };
        match spec {
            'Y' => out.push_str(&format!("{year:04}")),
            'y' => out.push_str(&format!("{:02}", year.rem_euclid(100))),
            'm' => out.push_str(&format!("{month:02}")),
            'd' => out.push_str(&format!("{day:02}")),
            'e' => out.push_str(&format!("{day:2}")),
            'H' => out.push_str(&format!("{hour:02}")),
            'M' => out.push_str(&format!("{minute:02}")),
            'S' => out.push_str(&format!("{second:02}")),
            'b' => out.push_str(&month_names[(month - 1) as usize][..3]),
            'B' => out.push_str(month_names[(month - 1) as usize]),
            'a' => out.push_str(&day_names[weekday][..3]),
            'A' => out.push_str(day_names[weekday]),
            'F' => out.push_str(&format!("{year:04}-{month:02}-{day:02}")),
            'T' => out.push_str(&format!("{hour:02}:{minute:02}:{second:02}")),
            'D' => {
                let short_year = year.rem_euclid(100);
                out.push_str(&format!("{short_year:02}/{month:02}/{day:02}"));
            }
            'j' => {
                let ordinal =
                    day + match month {
                        1 => 0,
                        2 => 31,
                        3 => 59,
                        4 => 90,
                        5 => 120,
                        6 => 151,
                        7 => 181,
                        8 => 212,
                        9 => 243,
                        10 => 273,
                        11 => 304,
                        _ => 334,
                    } + if month > 2 && is_leap(year) { 1 } else { 0 };
                out.push_str(&format!("{ordinal:03}"));
            }
            'n' => out.push('\n'),
            't' => out.push('\t'),
            '%' => out.push('%'),
            other => {
                out.push('%');
                out.push(other);
            }
        }
    }
    out
}

fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    (y + if m <= 2 { 1 } else { 0 }, m, d)
}

// ---------------------------------------------------------------------------
// JSON conversion
// ---------------------------------------------------------------------------

pub fn value_from_json(json: &Json, mark_input: bool) -> Value {
    from_json(json, mark_input, 0)
}

fn from_json(j: &Json, mark_input: bool, depth: usize) -> Value {
    match j {
        Json::Null | Json::Discarded => mk_none(),
        Json::Bool(v) => mk_bool(*v),
        Json::Int(v) => mk_int(*v),
        Json::UInt(v) => mk_int(*v as i64),
        Json::Double(v) => mk_float(*v),
        Json::String(v) => {
            let s = if mark_input {
                mk_jstring(JString::from_input(v.clone()))
            } else {
                mk_string(v.clone())
            };
            s
        }
        Json::Array(_) | Json::Object(_) if depth >= JINJA_RECURSIVE_LIMIT => mk_undefined(""),
        Json::Array(items) => {
            let arr = mk_array();
            for item in items {
                push_array(&arr, from_json(item, mark_input, depth + 1));
            }
            arr
        }
        Json::Object(entries) => {
            let obj = mk_object();
            for (k, v) in entries {
                insert_str(&obj, k, from_json(v, mark_input, depth + 1)).ok();
            }
            obj
        }
    }
}

pub fn global_from_json(ctx: &mut Context, json_obj: &Json, mark_input: bool) -> JResult<()> {
    match json_obj {
        Json::Object(entries) => {
            for (k, v) in entries {
                ctx.set_val(k, from_json(v, mark_input, 0))?;
            }
            Ok(())
        }
        _ => Err(JinjaError::Error(
            "global_from_json: input JSON value must be an object".into(),
        )),
    }
}

/// Internal helper kept for callers that need the sorted-name view.
#[allow(dead_code)]
pub fn global_builtin_names() -> &'static [&'static str] {
    NAMES
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Context {
        Context::new(String::new())
    }

    fn args(c: &Context, vals: Vec<Value>) -> FuncArgs<'_> {
        let mut a = FuncArgs::new(c);
        for v in vals {
            a.push_back(v);
        }
        a
    }

    fn apply(name: &str, input: Value, vals: Vec<Value>) -> JResult<Value> {
        let c = ctx();
        let a = args(&c, vals);
        apply_filter_by_name(&c, name.to_string(), input, &a)
    }

    fn test(name: &str, vals: Vec<Value>) -> JResult<Value> {
        let c = ctx();
        let a = args(&c, vals);
        global_builtin(name).unwrap()(&a)
    }

    fn text(v: JResult<Value>) -> String {
        as_string(&v.expect("expected success"))
            .expect("expected string")
            .str()
    }

    fn obj(pairs: &[(&str, &str)]) -> Value {
        let o = mk_object();
        for (k, v) in pairs {
            insert_str(&o, k, mk_string(*v)).unwrap();
        }
        o
    }

    #[test]
    fn string_join_uses_separator() {
        let arr = mk_array_from(vec![mk_string("a"), mk_string("b"), mk_string("c")]);
        assert_eq!(text(apply("join", mk_string("-"), vec![arr])), "a-b-c");
    }

    #[test]
    fn string_join_empty_array() {
        assert_eq!(text(apply("join", mk_string("-"), vec![mk_array()])), "");
    }

    #[test]
    fn string_join_numeric_items() {
        let arr = mk_array_from(vec![mk_int(1), mk_float(2.5)]);
        assert_eq!(text(apply("join", mk_string(","), vec![arr])), "1,2.5");
    }

    #[test]
    fn string_join_missing_iterable_errors() {
        assert!(apply("join", mk_string("-"), vec![]).is_err());
    }

    #[test]
    fn string_join_rejects_unsupported_item() {
        let arr = mk_array_from(vec![mk_string("a"), mk_object()]);
        assert!(apply("join", mk_string("-"), vec![arr]).is_err());
    }

    #[test]
    fn object_join_joins_keys() {
        let o = obj(&[("a", "1"), ("b", "2"), ("c", "3")]);
        assert_eq!(text(apply("join", o, vec![mk_string(", ")])), "a, b, c");
    }

    #[test]
    fn object_join_empty_and_default_separator() {
        assert_eq!(text(apply("join", mk_object(), vec![mk_string("-")])), "");
        assert_eq!(
            text(apply("join", obj(&[("a", "1"), ("b", "2")]), vec![])),
            "ab"
        );
    }

    #[test]
    fn replace_all_by_default() {
        assert_eq!(
            text(apply(
                "replace",
                mk_string("a-b-a"),
                vec![mk_string("a"), mk_string("X")]
            )),
            "X-b-X"
        );
    }

    #[test]
    fn replace_honors_positive_count() {
        let base = vec![mk_string("a"), mk_string("X")];
        let mut one = base.clone();
        one.push(mk_int(1));
        assert_eq!(text(apply("replace", mk_string("a-a-a"), one)), "X-a-a");
        let mut two = base;
        two.push(mk_int(2));
        assert_eq!(text(apply("replace", mk_string("a-a-a"), two)), "X-X-a");
    }

    #[test]
    fn replace_zero_and_negative_count() {
        let mut zero = vec![mk_string("a"), mk_string("X"), mk_int(0)];
        assert_eq!(
            text(apply("replace", mk_string("a-a"), zero.clone())),
            "a-a"
        );
        zero.pop();
        zero.push(mk_int(-3));
        assert_eq!(text(apply("replace", mk_string("a-a"), zero)), "X-X");
    }

    #[test]
    fn replace_empty_old_with_count() {
        let mut args = vec![mk_string(""), mk_string("-"), mk_int(2)];
        assert_eq!(
            text(apply("replace", mk_string("abc"), args.clone())),
            "-a-bc"
        );
        args.pop();
        assert_eq!(text(apply("replace", mk_string("abc"), args)), "-a-b-c-");
    }

    #[test]
    fn replace_wrong_args_error() {
        assert!(apply("replace", mk_string("abc"), vec![mk_string("a")]).is_err());
        assert!(apply(
            "replace",
            mk_string("abc"),
            vec![mk_string("a"), mk_string("X"), mk_string("1")]
        )
        .is_err());
    }

    #[test]
    fn escaped_test_string_only() {
        assert!(as_bool(&test("test_is_escaped", vec![mk_string("x")]).unwrap()).unwrap());
        assert!(!as_bool(&test("test_is_escaped", vec![mk_int(1)]).unwrap()).unwrap());
        assert!(!as_bool(&test("test_is_escaped", vec![mk_undefined("")]).unwrap()).unwrap());
    }

    #[test]
    fn filter_test_checks_registered_filters() {
        for (name, expected) in [
            ("upper", true),
            ("count", true),
            ("trim", true),
            ("join", true),
            ("nope", false),
            ("range", false),
        ] {
            let got = as_bool(&test("test_is_filter", vec![mk_string(name)]).unwrap()).unwrap();
            assert_eq!(got, expected, "filter '{name}'");
        }
    }

    #[test]
    fn filter_test_requires_string_name() {
        assert!(test("test_is_filter", vec![]).is_err());
        assert!(test("test_is_filter", vec![mk_int(1)]).is_err());
    }
}

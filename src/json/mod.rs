//! Rust port of llama.cpp `common/json.h` and `common/json.cpp`.
//!
//! A thin JSON value layered over a small std-only tree. Many features of a
//! full JSON library are deliberately left out to keep this module small:
//! - object keys keep the order in which they are added
//! - every error is reported through `crate::core::Error`
//! - a nested object is built with `Json::object_from`, not a braced pair
//!
//! A `Value` that holds a tree is single use: converting it to `Json` moves
//! the tree out. A second conversion gives null.

use crate::core::{Error, Result};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

/// Port of `common_json_error`.
pub type JsonError = Error;

fn json_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::new(msg))
}

// ---------------------------------------------------------------------------
// value type
// ---------------------------------------------------------------------------

/// One tagged value, as in `common_json_value`.
#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Double(f64),
    String(String),
    Json(Box<Json>),
}

impl Default for Value {
    fn default() -> Self {
        Value::Null
    }
}

macro_rules! value_from_int {
    ($($ty:ty => $variant:ident),* $(,)?) => {
        $(
            impl From<$ty> for Value {
                fn from(val: $ty) -> Self {
                    Value::$variant(val as _)
                }
            }
        )*
    };
}

value_from_int! {
    i8 => Int, i16 => Int, i32 => Int, i64 => Int,
    u8 => UInt, u16 => UInt, u32 => UInt, u64 => UInt, usize => UInt,
    isize => Int,
}

impl From<bool> for Value {
    fn from(val: bool) -> Self {
        Value::Bool(val)
    }
}

impl From<f32> for Value {
    fn from(val: f32) -> Self {
        Value::Double(val as f64)
    }
}

impl From<f64> for Value {
    fn from(val: f64) -> Self {
        Value::Double(val)
    }
}

impl From<String> for Value {
    fn from(val: String) -> Self {
        Value::String(val)
    }
}

impl From<&str> for Value {
    fn from(val: &str) -> Self {
        Value::String(val.to_string())
    }
}

impl From<&String> for Value {
    fn from(val: &String) -> Self {
        Value::String(val.clone())
    }
}

impl From<Json> for Value {
    fn from(val: Json) -> Self {
        Value::Json(Box::new(val))
    }
}

// a vector, set or map becomes an array or object, in its own iteration order
impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(vals: Vec<T>) -> Self {
        let mut out = Vec::with_capacity(vals.len());
        for val in vals {
            let val: Value = val.into();
            out.push(Json::from(val));
        }
        Value::Json(Box::new(Json::Array(out)))
    }
}

impl<T: Into<Value> + Ord> From<BTreeSet<T>> for Value {
    fn from(vals: BTreeSet<T>) -> Self {
        Value::from(vals.into_iter().collect::<Vec<_>>())
    }
}

impl<T: Into<Value> + Eq + Hash> From<HashSet<T>> for Value {
    fn from(vals: HashSet<T>) -> Self {
        Value::from(vals.into_iter().collect::<Vec<_>>())
    }
}

impl<T: Into<Value>> From<BTreeMap<String, T>> for Value {
    fn from(vals: BTreeMap<String, T>) -> Self {
        let items: Vec<Item> = vals.into_iter().map(|(k, v)| Item::new(k, v)).collect();
        Value::Json(Box::new(Json::object_from(items)))
    }
}

impl<T: Into<Value>> From<HashMap<String, T>> for Value {
    fn from(vals: HashMap<String, T>) -> Self {
        let items: Vec<Item> = vals.into_iter().map(|(k, v)| Item::new(k, v)).collect();
        Value::Json(Box::new(Json::object_from(items)))
    }
}

/// Port of `common_json_item`, one key/value pair of an object.
#[derive(Clone, Debug)]
pub struct Item {
    pub key: String,
    pub val: Value,
}

impl Item {
    pub fn new(key: impl Into<String>, val: impl Into<Value>) -> Self {
        Self { key: key.into(), val: val.into() }
    }
}

// ---------------------------------------------------------------------------
// the tree
// ---------------------------------------------------------------------------

/// Port of `common_json`, backed by an ordered object instead of a vendor lib.
#[derive(Clone, Debug)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Double(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
    /// Result of `parse_no_throw` on invalid text.
    Discarded,
}

impl Default for Json {
    fn default() -> Self {
        Json::Null
    }
}

impl From<Value> for Json {
    fn from(val: Value) -> Self {
        // one owner means the tree can be moved out; a second use gives null
        match val {
            Value::Null => Json::Null,
            Value::Bool(v) => Json::Bool(v),
            Value::Int(v) => Json::Int(v),
            Value::UInt(v) => Json::UInt(v),
            Value::Double(v) => Json::Double(v),
            Value::String(v) => Json::String(v),
            Value::Json(v) => *v,
        }
    }
}

impl Json {
    // null, same as the backing library
    // operator[] turns it into an object, push_back() into an array
    pub fn array() -> Self {
        Json::Array(Vec::new())
    }

    pub fn array_from<T: Into<Value>>(vals: impl IntoIterator<Item = T>) -> Self {
        let mut out = Vec::new();
        for val in vals {
            let val: Value = val.into();
            out.push(Json::from(val));
        }
        Json::Array(out)
    }

    pub fn object() -> Self {
        Json::Object(Vec::new())
    }

    pub fn object_from(items: impl IntoIterator<Item = Item>) -> Self {
        let mut entries: Vec<(String, Json)> = Vec::new();
        for item in items {
            let val = Json::from(item.val);
            if let Some(entry) = entries.iter_mut().find(|e| e.0 == item.key) {
                entry.1 = val;
            } else {
                entries.push((item.key, val));
            }
        }
        Json::Object(entries)
    }

    /// Holds a single value: `make("abc").dump(-1)` gives `"\"abc\""`.
    pub fn make(val: Value) -> Self {
        Json::from(val)
    }

    // -----------------------------------------------------------------------
    // parser
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // type queries
    // -----------------------------------------------------------------------

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    pub fn is_object(&self) -> bool {
        matches!(self, Json::Object(_))
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Json::Array(_))
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Json::String(_))
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self, Json::Bool(_))
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Json::Int(_) | Json::UInt(_) | Json::Double(_))
    }

    pub fn is_number_integer(&self) -> bool {
        matches!(self, Json::Int(_) | Json::UInt(_))
    }

    pub fn is_number_float(&self) -> bool {
        matches!(self, Json::Double(_))
    }

    pub fn empty(&self) -> bool {
        match self {
            Json::Null | Json::Discarded => true,
            Json::Array(a) => a.is_empty(),
            Json::Object(o) => o.is_empty(),
            Json::String(s) => s.is_empty(),
            _ => false,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Json::Null | Json::Discarded => 0,
            Json::Array(a) => a.len(),
            Json::Object(o) => o.len(),
            Json::String(s) => s.len(),
            _ => 1,
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        match self {
            Json::Object(o) => o.iter().any(|e| e.0 == key),
            _ => false,
        }
    }

    // -----------------------------------------------------------------------
    // object and array accessors
    // -----------------------------------------------------------------------

    fn obj_set(&mut self, key: String, val: Json) -> Result<()> {
        match self {
            Json::Object(entries) => {
                if let Some(entry) = entries.iter_mut().find(|e| e.0 == key) {
                    entry.1 = val;
                } else {
                    entries.push((key, val));
                }
                Ok(())
            }
            Json::Null => {
                *self = Json::Object(vec![(key, val)]);
                Ok(())
            }
            _ => json_err("cannot use operator[] with a string argument with this value"),
        }
    }

    /// `at()` returns an error if the key or index is missing; `operator[]`
    /// (the `IndexMut` impl) adds a null value instead.
    pub fn at<K: JsonKey>(&self, key: K) -> Result<&Json> {
        key.get(self)
    }

    pub fn at_mut<K: JsonKey>(&mut self, key: K) -> Result<&mut Json> {
        key.get_mut(self)
    }

    pub fn at_index(&self, idx: usize) -> Result<&Json> {
        match self {
            Json::Array(a) => a.get(idx).ok_or_else(|| Error::new("array index out of range")),
            _ => json_err("cannot use operator[] with an integer argument with this value"),
        }
    }

    pub fn at_index_mut(&mut self, idx: usize) -> Result<&mut Json> {
        match self {
            Json::Array(a) => {
                if idx >= a.len() {
                    return json_err("array index out of range");
                }
                Ok(&mut a[idx])
            }
            _ => json_err("cannot use operator[] with an integer argument with this value"),
        }
    }

    pub fn front(&self) -> Result<&Json> {
        match self {
            Json::Array(a) => a.first().ok_or_else(|| Error::new("front of empty array")),
            Json::Object(o) => o.first().map(|e| &e.1).ok_or_else(|| Error::new("front of empty object")),
            Json::Null | Json::Discarded => json_err("cannot use front() with null"),
            other => Ok(other),
        }
    }

    pub fn front_mut(&mut self) -> Result<&mut Json> {
        match self {
            Json::Array(a) => {
                if a.is_empty() {
                    return json_err("front of empty array");
                }
                Ok(&mut a[0])
            }
            Json::Object(o) => {
                if o.is_empty() {
                    return json_err("front of empty object");
                }
                Ok(&mut o[0].1)
            }
            Json::Null | Json::Discarded => json_err("cannot use front() with null"),
            other => Ok(other),
        }
    }

    pub fn back(&self) -> Result<&Json> {
        match self {
            Json::Array(a) => a.last().ok_or_else(|| Error::new("back of empty array")),
            Json::Object(o) => o.last().map(|e| &e.1).ok_or_else(|| Error::new("back of empty object")),
            Json::Null | Json::Discarded => json_err("cannot use back() with null"),
            other => Ok(other),
        }
    }

    pub fn back_mut(&mut self) -> Result<&mut Json> {
        match self {
            Json::Array(a) => {
                if a.is_empty() {
                    return json_err("back of empty array");
                }
                let last = a.len() - 1;
                Ok(&mut a[last])
            }
            Json::Object(o) => {
                if o.is_empty() {
                    return json_err("back of empty object");
                }
                let last = o.len() - 1;
                Ok(&mut o[last].1)
            }
            Json::Null | Json::Discarded => json_err("cannot use back() with null"),
            other => Ok(other),
        }
    }

    pub fn clear(&mut self) {
        match self {
            Json::Array(a) => a.clear(),
            Json::Object(o) => o.clear(),
            Json::String(s) => s.clear(),
            Json::Null | Json::Discarded => {}
            _ => *self = Json::Null,
        }
    }

    pub fn erase(&mut self, key: &str) -> Result<()> {
        match self {
            Json::Object(o) => {
                if let Some(i) = o.iter().position(|e| e.0 == key) {
                    o.remove(i);
                }
                Ok(())
            }
            _ => json_err("cannot erase a key from a non-object value"),
        }
    }

    pub fn erase_index(&mut self, idx: usize) -> Result<()> {
        match self {
            Json::Array(a) => {
                if idx >= a.len() {
                    return json_err("array index out of range");
                }
                a.remove(idx);
                Ok(())
            }
            _ => json_err("cannot erase an index from a non-array value"),
        }
    }

    pub fn assign(&mut self, val: Value) {
        *self = Json::from(val);
    }

    pub fn set(&mut self, item: Item) -> Result<()> {
        self.obj_set(item.key, Json::from(item.val))
    }

    pub fn push_back(&mut self, val: Value) -> Result<()> {
        let val = Json::from(val);
        match self {
            Json::Array(a) => {
                a.push(val);
                Ok(())
            }
            Json::Null => {
                *self = Json::Array(vec![val]);
                Ok(())
            }
            _ => json_err("cannot push_back to a non-array value"),
        }
    }

    /// Appends one object, e.g. `push_back_items(vec![Item::new("a", 1)])`.
    pub fn push_back_items(&mut self, items: impl IntoIterator<Item = Item>) -> Result<()> {
        let val = Json::object_from(items);
        self.push_back(val.into())
    }

    /// 1 if the key is there, 0 if not.
    pub fn count(&self, key: &str) -> usize {
        if self.contains(key) { 1 } else { 0 }
    }

    /// Appends every value of another array. With `&mut self` and `&Json`
    /// borrows, inserting an array into itself cannot happen here.
    pub fn insert(&mut self, vals: &Json) -> Result<()> {
        let src = match vals {
            Json::Array(a) => a,
            Json::Null => return Ok(()),
            _ => return json_err("cannot insert values from a non-array"),
        };
        match self {
            Json::Array(a) => {
                a.extend(src.iter().cloned());
                Ok(())
            }
            Json::Null => {
                *self = Json::Array(src.clone());
                Ok(())
            }
            _ => json_err("cannot insert values into a non-array"),
        }
    }

    // -----------------------------------------------------------------------
    // typed values
    // -----------------------------------------------------------------------

    pub fn get<T: FromJson>(&self) -> Result<T> {
        T::from_json(self)
    }

    pub fn value<T: FromJson>(&self, key: &str, def: T) -> Result<T> {
        if self.contains(key) {
            self.at(key)?.get::<T>()
        } else {
            Ok(def)
        }
    }

    pub fn value_str(&self, key: &str, def: &str) -> Result<String> {
        if self.contains(key) {
            self.at(key)?.get::<String>()
        } else {
            Ok(def.to_string())
        }
    }

    // a JSON default needs no get<T>(), it is already the right type
    pub fn value_json(&self, key: &str, def: &Json) -> Result<Json> {
        if self.contains(key) {
            Ok(self.at(key)?.clone())
        } else {
            Ok(def.clone())
        }
    }

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

    // -----------------------------------------------------------------------
    // iteration
    // -----------------------------------------------------------------------

    /// Walks an array by index, or an object in insertion order. A plain
    /// value gives itself once, same as the backing library.
    pub fn iter(&self) -> Iter<'_> {
        Iter { node: self, idx: 0, returned: false }
    }

    /// Keys follow the backing library: the index for an array, "" for a
    /// plain value.
    pub fn items(&self) -> Items<'_> {
        Items { node: self, idx: 0 }
    }

    pub fn items_mut(&mut self) -> ItemsMut<'_> {
        let mut entries: Vec<(String, &mut Json)> = Vec::new();
        match self {
            Json::Object(o) => {
                for (key, val) in o.iter_mut() {
                    entries.push((key.clone(), val));
                }
            }
            Json::Array(a) => {
                for (i, val) in a.iter_mut().enumerate() {
                    entries.push((i.to_string(), val));
                }
            }
            Json::Null | Json::Discarded => {}
            other => entries.push((String::new(), other)),
        }
        ItemsMut { entries: entries.into_iter() }
    }
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.dump(-1).unwrap_or_default())
    }
}

impl PartialEq for Json {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Json::Null, Json::Null) | (Json::Discarded, Json::Discarded) => true,
            (Json::Bool(a), Json::Bool(b)) => a == b,
            (Json::String(a), Json::String(b)) => a == b,
            (Json::Array(a), Json::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
            }
            (Json::Object(a), Json::Object(b)) => {
                a.len() == b.len()
                    && a.iter().zip(b.iter()).all(|((ka, va), (kb, vb))| ka == kb && va == vb)
            }
            (Json::Int(a), Json::Int(b)) => a == b,
            (Json::UInt(a), Json::UInt(b)) => a == b,
            (Json::Int(a), Json::UInt(b)) => *a >= 0 && (*a as u128) == (*b as u128),
            (Json::UInt(a), Json::Int(b)) => *b >= 0 && (*a as u128) == (*b as u128),
            (Json::Double(a), Json::Double(b)) => a == b,
            (Json::Double(a), Json::Int(b)) | (Json::Int(b), Json::Double(a)) => *a == (*b as f64),
            (Json::Double(a), Json::UInt(b)) | (Json::UInt(b), Json::Double(a)) => *a == (*b as f64),
            _ => false,
        }
    }
}

impl PartialEq<Value> for Json {
    fn eq(&self, val: &Value) -> bool {
        // compare a tree in place, converting it would copy it
        match val {
            Value::Json(tree) => self == &**tree,
            other => self == &Json::from(other.clone()),
        }
    }
}

impl PartialEq<Json> for Value {
    fn eq(&self, json: &Json) -> bool {
        json == self
    }
}

// ---------------------------------------------------------------------------
// keys and indexing
// ---------------------------------------------------------------------------

/// Key argument of `at()` / `at_mut()`: either an object key or an index.
pub trait JsonKey {
    fn get<'a>(&self, json: &'a Json) -> Result<&'a Json>;
    fn get_mut<'a>(&self, json: &'a mut Json) -> Result<&'a mut Json>;
}

impl JsonKey for &str {
    fn get<'a>(&self, json: &'a Json) -> Result<&'a Json> {
        match json {
            Json::Object(o) => o
                .iter()
                .find(|e| e.0 == *self)
                .map(|e| &e.1)
                .ok_or_else(|| Error::new("key not found")),
            _ => json_err("cannot use at() with a string argument with this value"),
        }
    }

    fn get_mut<'a>(&self, json: &'a mut Json) -> Result<&'a mut Json> {
        match json {
            Json::Object(o) => o
                .iter_mut()
                .find(|e| e.0 == *self)
                .map(|e| &mut e.1)
                .ok_or_else(|| Error::new("key not found")),
            _ => json_err("cannot use at() with a string argument with this value"),
        }
    }
}

impl JsonKey for &String {
    fn get<'a>(&self, json: &'a Json) -> Result<&'a Json> {
        <&str as JsonKey>::get(&self.as_str(), json)
    }

    fn get_mut<'a>(&self, json: &'a mut Json) -> Result<&'a mut Json> {
        <&str as JsonKey>::get_mut(&self.as_str(), json)
    }
}

impl JsonKey for String {
    fn get<'a>(&self, json: &'a Json) -> Result<&'a Json> {
        <&str as JsonKey>::get(&self.as_str(), json)
    }

    fn get_mut<'a>(&self, json: &'a mut Json) -> Result<&'a mut Json> {
        <&str as JsonKey>::get_mut(&self.as_str(), json)
    }
}

impl JsonKey for usize {
    fn get<'a>(&self, json: &'a Json) -> Result<&'a Json> {
        json.at_index(*self)
    }

    fn get_mut<'a>(&self, json: &'a mut Json) -> Result<&'a mut Json> {
        json.at_index_mut(*self)
    }
}

impl JsonKey for i32 {
    fn get<'a>(&self, json: &'a Json) -> Result<&'a Json> {
        // a negative index must not turn into a huge size_t
        if *self < 0 {
            return json_err("negative array index");
        }
        json.at_index(*self as usize)
    }

    fn get_mut<'a>(&self, json: &'a mut Json) -> Result<&'a mut Json> {
        if *self < 0 {
            return json_err("negative array index");
        }
        json.at_index_mut(*self as usize)
    }
}

impl Index<&str> for Json {
    type Output = Json;

    fn index(&self, key: &str) -> &Json {
        self.at(key).expect("json key not found")
    }
}

impl IndexMut<&str> for Json {
    fn index_mut(&mut self, key: &str) -> &mut Json {
        if !self.contains(key) {
            self.obj_set(key.to_string(), Json::Null).expect("json operator[] failed");
        }
        self.at_mut(key).expect("json key not found")
    }
}

impl Index<usize> for Json {
    type Output = Json;

    fn index(&self, idx: usize) -> &Json {
        self.at_index(idx).expect("json index out of range")
    }
}

impl IndexMut<usize> for Json {
    fn index_mut(&mut self, idx: usize) -> &mut Json {
        if matches!(self, Json::Null) {
            *self = Json::Array(Vec::new());
        }
        match self {
            Json::Array(a) => {
                if idx >= a.len() {
                    a.resize_with(idx + 1, || Json::Null);
                }
                &mut a[idx]
            }
            _ => panic!("json operator[] needs an array"),
        }
    }
}

// ---------------------------------------------------------------------------
// get<T>()
// ---------------------------------------------------------------------------

/// Port of the explicit `get<T>()` specializations.
pub trait FromJson: Sized {
    fn from_json(json: &Json) -> Result<Self>;
}

macro_rules! impl_from_json_num {
    ($($ty:ty),* $(,)?) => {
        $(
            impl FromJson for $ty {
                fn from_json(json: &Json) -> Result<Self> {
                    match json {
                        Json::Bool(v) => Ok(u8::from(*v) as $ty),
                        Json::Int(v) => Ok(*v as $ty),
                        Json::UInt(v) => Ok(*v as $ty),
                        Json::Double(v) => Ok(*v as $ty),
                        _ => json_err("type error - cannot get a number from this value"),
                    }
                }
            }
        )*
    };
}

impl_from_json_num!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);

impl FromJson for bool {
    fn from_json(json: &Json) -> Result<Self> {
        match json {
            Json::Bool(v) => Ok(*v),
            _ => json_err("type error - cannot get a boolean from this value"),
        }
    }
}

impl FromJson for String {
    fn from_json(json: &Json) -> Result<Self> {
        match json {
            Json::String(v) => Ok(v.clone()),
            _ => json_err("type error - cannot get a string from this value"),
        }
    }
}

impl FromJson for Json {
    fn from_json(json: &Json) -> Result<Self> {
        Ok(json.clone())
    }
}

impl<T: FromJson> FromJson for Vec<T> {
    fn from_json(json: &Json) -> Result<Self> {
        match json {
            Json::Array(a) => a.iter().map(T::from_json).collect(),
            _ => json_err("type error - cannot get an array from this value"),
        }
    }
}

impl<T: FromJson + Ord> FromJson for BTreeSet<T> {
    fn from_json(json: &Json) -> Result<Self> {
        Ok(Vec::<T>::from_json(json)?.into_iter().collect())
    }
}

impl<T: FromJson + Eq + Hash> FromJson for HashSet<T> {
    fn from_json(json: &Json) -> Result<Self> {
        Ok(Vec::<T>::from_json(json)?.into_iter().collect())
    }
}

impl<T: FromJson> FromJson for HashMap<String, T> {
    fn from_json(json: &Json) -> Result<Self> {
        match json {
            Json::Object(o) => {
                let mut out = HashMap::new();
                for (key, val) in o {
                    out.insert(key.clone(), T::from_json(val)?);
                }
                Ok(out)
            }
            _ => json_err("type error - cannot get an object from this value"),
        }
    }
}

impl<T: FromJson> FromJson for BTreeMap<String, T> {
    fn from_json(json: &Json) -> Result<Self> {
        match json {
            Json::Object(o) => {
                let mut out = BTreeMap::new();
                for (key, val) in o {
                    out.insert(key.clone(), T::from_json(val)?);
                }
                Ok(out)
            }
            _ => json_err("type error - cannot get an object from this value"),
        }
    }
}

// ---------------------------------------------------------------------------
// iteration
// ---------------------------------------------------------------------------

pub struct Iter<'a> {
    node: &'a Json,
    idx: usize,
    returned: bool,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a Json;

    fn next(&mut self) -> Option<Self::Item> {
        if self.returned {
            self.idx += 1;
        } else {
            self.returned = true;
        }
        self.value()
    }
}

impl<'a> Iter<'a> {
    pub fn key(&self) -> String {
        match self.node {
            Json::Object(o) => o.get(self.idx).map(|e| e.0.clone()).unwrap_or_default(),
            Json::Array(_) => self.idx.to_string(),
            _ => String::new(),
        }
    }

    pub fn value(&self) -> Option<&'a Json> {
        match self.node {
            Json::Object(o) => o.get(self.idx).map(|e| &e.1),
            Json::Array(a) => a.get(self.idx),
            Json::Null | Json::Discarded => None,
            other => if self.idx == 0 { Some(other) } else { None },
        }
    }
}

impl<'a> IntoIterator for &'a Json {
    type Item = &'a Json;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct ItemEntry<'a> {
    pub k: String,
    pub v: &'a Json,
}

impl ItemEntry<'_> {
    pub fn key(&self) -> &str {
        &self.k
    }

    pub fn value(&self) -> &Json {
        self.v
    }
}

pub struct Items<'a> {
    node: &'a Json,
    idx: usize,
}

impl<'a> Iterator for Items<'a> {
    type Item = ItemEntry<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx;
        if idx >= self.node.size() {
            return None;
        }
        self.idx += 1;
        match self.node {
            Json::Object(o) => o.get(idx).map(|e| ItemEntry { k: e.0.clone(), v: &e.1 }),
            Json::Array(a) => a.get(idx).map(|v| ItemEntry { k: idx.to_string(), v }),
            Json::Null | Json::Discarded => None,
            other => Some(ItemEntry { k: String::new(), v: other }),
        }
    }
}

pub struct ItemEntryMut<'a> {
    pub k: String,
    pub v: &'a mut Json,
}

impl ItemEntryMut<'_> {
    pub fn key(&self) -> &str {
        &self.k
    }

    pub fn value(&self) -> &Json {
        self.v
    }
}

pub struct ItemsMut<'a> {
    entries: std::vec::IntoIter<(String, &'a mut Json)>,
}

impl<'a> Iterator for ItemsMut<'a> {
    type Item = ItemEntryMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next().map(|(k, v)| ItemEntryMut { k, v })
    }
}

pub type JsonEntry<'a> = ItemEntry<'a>;

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
fn decode_utf8(bytes: &[u8], pos: usize) -> Result<(char, usize)> {
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

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_dump_round_trip() {
        let text = r#"{"name":"llama","n":7,"ok":true,"xs":[1,2.5,null]}"#;
        let json = Json::parse(text).unwrap();
        assert!(json.is_object());
        assert_eq!(json.at("n").unwrap().get::<i64>().unwrap(), 7);
        assert_eq!(json.at("xs").unwrap().size(), 3);
        assert_eq!(json.dump(-1).unwrap(), text);
        assert_eq!(Json::parse(&json.dump(-1).unwrap()).unwrap(), json);
    }

    #[test]
    fn escaping_pretty_dump_and_discarded() {
        let json = Json::object_from(vec![
            Item::new("s", "a\n\"b"),
            Item::new("u", "\u{00e9}"),
        ]);
        let compact = json.dump(-1).unwrap();
        assert_eq!(compact, r#"{"s":"a\n\"b","u":"é"}"#);
        let pretty = json.dump(2).unwrap();
        assert!(pretty.contains("\n  \"s\": \"a\\n\\\"b\""));
        assert_eq!(Json::parse(&compact).unwrap(), json);
        assert!(Json::parse_no_throw("{oops").is_discarded());
    }
}

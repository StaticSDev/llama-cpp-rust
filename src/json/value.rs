//! Value type, accessors, iterators and indexing for the JSON tree.

use crate::core::{Error, Result};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

use super::json_err;

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

}

impl Json {
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

}

impl Json {
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

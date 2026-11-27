//! Rust port of `common/json-schema.{h,cpp}` and
//! `common/json-schema-to-grammar.{h,cpp}`.
//!
//! The schema is an owned tree of `Rc` nodes. A `$ref` node keeps its target
//! behind a `RefCell` because targets are filled in after the whole document
//! has been built, which is what lets cyclic references resolve.

use crate::common::{string_join, string_repeat, string_split, COMMON_RECURSIVE_LIMIT};
use crate::core::{Error, Result};
use crate::json::{Item, Json};
use crate::unicode::unicode_cpt_to_utf8;
use std::cell::RefCell;
use std::cmp::{max, min};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::rc::Rc;

// ---------------------------------------------------------------------------
// schema types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Any,
    Ref,
    AnyOf,
    AllOf,
    Const,
    Enum,
    Null,
    Boolean,
    Number,
    Integer,
    String,
    Array,
    Tuple,
    Object,
}

impl NodeKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Ref => "ref",
            Self::AnyOf => "anyOf",
            Self::AllOf => "allOf",
            Self::Const => "const",
            Self::Enum => "enum",
            Self::Null => "null",
            Self::Boolean => "boolean",
            Self::Number => "number",
            Self::Integer => "integer",
            Self::String => "string",
            Self::Array => "array",
            Self::Tuple => "tuple",
            Self::Object => "object",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum ValueType {
    Null = 0,
    Boolean = 1,
    Number = 2,
    Integer = 3,
    String = 4,
    Array = 5,
    Object = 6,
}

impl ValueType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Boolean => "boolean",
            Self::Number => "number",
            Self::Integer => "integer",
            Self::String => "string",
            Self::Array => "array",
            Self::Object => "object",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StringFormat {
    None,
    Uuid,
    Date,
    Time,
    DateTime,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TypeSet(u32);

impl TypeSet {
    pub fn all() -> Self {
        Self((1u32 << (ValueType::Object as u32 + 1)) - 1)
    }

    pub fn single(t: ValueType) -> Self {
        Self(1u32 << (t as u32))
    }

    pub fn add(&mut self, t: ValueType) {
        self.0 |= 1u32 << (t as u32);
    }

    pub fn has(self, t: ValueType) -> bool {
        self.0 & (1u32 << (t as u32)) != 0
    }

    pub fn is_only(self, t: ValueType) -> bool {
        self.0 == 1u32 << (t as u32)
    }

    pub fn empty(self) -> bool {
        self.0 == 0
    }

    pub fn union_with(&mut self, other: Self) {
        self.0 |= other.0;
    }

    pub fn intersect_with(&mut self, other: Self) {
        self.0 &= other.0;
    }
}

#[derive(Clone, Debug)]
pub struct CommonChatSchemaProperty {
    pub name: String,
    pub schema: Rc<CommonChatSchema>,
    pub required: bool,
}

#[derive(Clone, Debug)]
pub enum CommonChatSchema {
    Any,
    Ref {
        ref_: String,
        target: RefCell<Option<Rc<CommonChatSchema>>>,
    },
    AnyOf {
        children: Vec<Rc<CommonChatSchema>>,
    },
    AllOf {
        children: Vec<Rc<CommonChatSchema>>,
    },
    Const {
        value: Json,
    },
    Enum {
        values: Vec<Json>,
    },
    Null,
    Boolean,
    Number,
    Integer {
        minimum: i64,
        maximum: i64,
    },
    String {
        pattern: String,
        format: StringFormat,
        min_length: i32,
        max_length: i32,
    },
    Array {
        items: Rc<CommonChatSchema>,
        min_items: i32,
        max_items: i32,
    },
    Tuple {
        items: Vec<Rc<CommonChatSchema>>,
    },
    Object {
        properties: Vec<CommonChatSchemaProperty>,
        additional_properties: Option<Rc<CommonChatSchema>>,
    },
}

impl CommonChatSchema {
    pub fn kind(&self) -> NodeKind {
        match self {
            Self::Any => NodeKind::Any,
            Self::Ref { .. } => NodeKind::Ref,
            Self::AnyOf { .. } => NodeKind::AnyOf,
            Self::AllOf { .. } => NodeKind::AllOf,
            Self::Const { .. } => NodeKind::Const,
            Self::Enum { .. } => NodeKind::Enum,
            Self::Null => NodeKind::Null,
            Self::Boolean => NodeKind::Boolean,
            Self::Number => NodeKind::Number,
            Self::Integer { .. } => NodeKind::Integer,
            Self::String { .. } => NodeKind::String,
            Self::Array { .. } => NodeKind::Array,
            Self::Tuple { .. } => NodeKind::Tuple,
            Self::Object { .. } => NodeKind::Object,
        }
    }

    /// Type set of all values that may match the schema.
    pub fn value_types(&self) -> Result<TypeSet> {
        let mut visited = HashSet::new();
        let mut depth = 0;
        value_types_impl(self, &mut visited, &mut depth)
    }

    /// Whether a matching value may be a string, through any branch.
    pub fn may_be_string(&self) -> Result<bool> {
        let mut visited = HashSet::new();
        let mut depth = 0;
        may_be_string_impl(self, &mut visited, &mut depth)
    }
}

fn depth_error() -> Error {
    Error::new(format!(
        "JSON schema is nested more than {} levels deep",
        COMMON_RECURSIVE_LIMIT
    ))
}

fn node_id(rc: &Rc<CommonChatSchema>) -> usize {
    Rc::as_ptr(rc) as *const () as usize
}

fn json_type(value: &Json) -> ValueType {
    if value.is_null() {
        ValueType::Null
    } else if value.is_boolean() {
        ValueType::Boolean
    } else if value.is_number_integer() {
        ValueType::Integer
    } else if value.is_number() {
        ValueType::Number
    } else if value.is_string() {
        ValueType::String
    } else if value.is_array() {
        ValueType::Array
    } else {
        ValueType::Object
    }
}

fn value_types_impl(
    schema: &CommonChatSchema,
    visited: &mut HashSet<usize>,
    depth: &mut usize,
) -> Result<TypeSet> {
    *depth += 1;
    if *depth > COMMON_RECURSIVE_LIMIT {
        *depth -= 1;
        return Err(depth_error());
    }
    let result = value_types_kind(schema, visited, depth);
    *depth -= 1;
    result
}

fn value_types_kind(
    schema: &CommonChatSchema,
    visited: &mut HashSet<usize>,
    depth: &mut usize,
) -> Result<TypeSet> {
    match schema {
        CommonChatSchema::Any => Ok(TypeSet::all()),
        CommonChatSchema::Null => Ok(TypeSet::single(ValueType::Null)),
        CommonChatSchema::Boolean => Ok(TypeSet::single(ValueType::Boolean)),
        CommonChatSchema::Number => {
            let mut types = TypeSet::single(ValueType::Number);
            types.add(ValueType::Integer);
            Ok(types)
        }
        CommonChatSchema::Integer { .. } => Ok(TypeSet::single(ValueType::Integer)),
        CommonChatSchema::String { .. } => Ok(TypeSet::single(ValueType::String)),
        CommonChatSchema::Array { .. } | CommonChatSchema::Tuple { .. } => {
            Ok(TypeSet::single(ValueType::Array))
        }
        CommonChatSchema::Object { .. } => Ok(TypeSet::single(ValueType::Object)),
        CommonChatSchema::Const { value } => Ok(TypeSet::single(json_type(value))),
        CommonChatSchema::Enum { values } => {
            let mut types = TypeSet::default();
            for value in values {
                types.add(json_type(value));
            }
            Ok(types)
        }
        CommonChatSchema::Ref { target, .. } => {
            let target = target.borrow().as_ref().cloned();
            let Some(target) = target else {
                return Ok(TypeSet::default());
            };
            let id = node_id(&target);
            if !visited.insert(id) {
                return Ok(TypeSet::default());
            }
            let types = value_types_impl(&target, visited, depth)?;
            visited.remove(&id);
            Ok(types)
        }
        CommonChatSchema::AnyOf { children } => {
            let mut types = TypeSet::default();
            for child in children {
                types.union_with(value_types_impl(child, visited, depth)?);
            }
            Ok(types)
        }
        CommonChatSchema::AllOf { children } => {
            let mut types = TypeSet::all();
            for child in children {
                types.intersect_with(value_types_impl(child, visited, depth)?);
            }
            Ok(types)
        }
    }
}

fn may_be_string_impl(
    schema: &CommonChatSchema,
    visited: &mut HashSet<usize>,
    depth: &mut usize,
) -> Result<bool> {
    *depth += 1;
    if *depth > COMMON_RECURSIVE_LIMIT {
        *depth -= 1;
        return Err(depth_error());
    }
    let result = may_be_string_kind(schema, visited, depth);
    *depth -= 1;
    result
}

fn may_be_string_kind(
    schema: &CommonChatSchema,
    visited: &mut HashSet<usize>,
    depth: &mut usize,
) -> Result<bool> {
    match schema {
        CommonChatSchema::String { .. } => Ok(true),
        CommonChatSchema::Const { value } => Ok(value.is_string()),
        CommonChatSchema::Enum { values } => Ok(values.iter().any(Json::is_string)),
        CommonChatSchema::Ref { target, .. } => {
            let target = target.borrow().as_ref().cloned();
            let Some(target) = target else {
                return Ok(false);
            };
            let id = node_id(&target);
            if !visited.insert(id) {
                return Ok(false);
            }
            let result = may_be_string_impl(&target, visited, depth)?;
            visited.remove(&id);
            Ok(result)
        }
        CommonChatSchema::AnyOf { children } => {
            for child in children {
                if may_be_string_impl(child, visited, depth)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        CommonChatSchema::AllOf { children } => {
            let mut any_string = false;
            for child in children {
                if child.kind() == NodeKind::Any {
                    continue;
                }
                if !may_be_string_impl(child, visited, depth)? {
                    return Ok(false);
                }
                any_string = true;
            }
            Ok(any_string)
        }
        _ => Ok(false),
    }
}

/// A parsed schema plus all `$ref` targets in the same document.
#[derive(Clone, Debug)]
pub struct CommonChatSchemaDocument {
    pub root: Rc<CommonChatSchema>,
    pub refs: BTreeMap<String, Rc<CommonChatSchema>>,
}

// ---------------------------------------------------------------------------
// schema builder
// ---------------------------------------------------------------------------

struct Builder<'a> {
    root: &'a Json,
    refs: BTreeMap<String, Rc<CommonChatSchema>>,
    pending: Vec<Rc<CommonChatSchema>>,
    depth: usize,
}

impl<'a> Builder<'a> {
    fn new(root: &'a Json) -> Self {
        Self {
            root,
            refs: BTreeMap::new(),
            pending: Vec::new(),
            depth: 0,
        }
    }

    fn fail(path: &str, msg: &str) -> Error {
        Error::new(format!("JSON schema error at {}: {}", path, msg))
    }

    fn get_count(schema: &Json, key: &str, path: &str, def: i32) -> Result<i32> {
        if !schema.contains(key) {
            return Ok(def);
        }
        let value = schema.at(key)?;
        if !value.is_number_integer() {
            return Err(Self::fail(path, &format!("{} must be a non-negative integer", key)));
        }
        let count = value.get::<i32>()?;
        if count < 0 {
            return Err(Self::fail(path, &format!("{} must be a non-negative integer", key)));
        }
        Ok(count)
    }

    fn get_bound(schema: &Json, key: &str, path: &str, round_up: bool) -> Result<i64> {
        let value = schema.at(key)?;
        if value.is_number_integer() {
            return value.get::<i64>();
        }
        if !value.is_number() {
            return Err(Self::fail(path, &format!("{} must be a number", key)));
        }
        let d = value.get::<f64>()?;
        Ok(if round_up { d.ceil() as i64 } else { d.floor() as i64 })
    }

    fn get_format(schema: &Json, path: &str) -> Result<StringFormat> {
        if !schema.contains("format") {
            return Ok(StringFormat::None);
        }
        let value = schema.at("format")?;
        if !value.is_string() {
            return Err(Self::fail(path, "format must be a string"));
        }
        let format = value.get::<String>()?;
        Ok(match format.as_str() {
            "date" => StringFormat::Date,
            "time" => StringFormat::Time,
            "date-time" => StringFormat::DateTime,
            "uuid" => StringFormat::Uuid,
            _ if format.len() == 5
                && &format.as_bytes()[..4] == b"uuid"
                && (b'1'..=b'5').contains(&format.as_bytes()[4]) =>
            {
                StringFormat::Uuid
            }
            _ => StringFormat::None,
        })
    }

    fn resolve_ref(&self, ref_: &str, path: &str) -> Result<&'a Json> {
        let mut target = self.root;
        let tokens = string_split(&ref_[1..], "/");
        for sel in tokens.iter().skip(1) {
            if target.is_object() && target.contains(sel) {
                target = target.at(sel.as_str())?;
            } else if target.is_array() {
                let idx = sel.parse::<usize>().unwrap_or_else(|_| target.size());
                if idx >= target.size() {
                    return Err(Self::fail(
                        path,
                        &format!("cannot resolve $ref {}, {} is out of range", ref_, sel),
                    ));
                }
                target = target.at_index(idx)?;
            } else {
                return Err(Self::fail(
                    path,
                    &format!("cannot resolve $ref {}, {} not found", ref_, sel),
                ));
            }
        }
        Ok(target)
    }

    fn build_ref(&mut self, value: &Json, path: &str) -> Result<Rc<CommonChatSchema>> {
        if !value.is_string() {
            return Err(Self::fail(path, "$ref must be a string"));
        }
        let ref_ = value.get::<String>()?;
        if !ref_.starts_with("#/") {
            return Err(Self::fail(
                path,
                &format!(
                    "unsupported $ref {}, only references into the same document are supported",
                    ref_
                ),
            ));
        }
        if !self.refs.contains_key(&ref_) {
            // reserve the key first, so a cycle back to this $ref stops here
            self.refs.insert(ref_.clone(), Rc::new(CommonChatSchema::Any));
            let json = self.resolve_ref(&ref_, path)?;
            let target = self.build_node(json, &ref_)?;
            self.refs.insert(ref_.clone(), target);
        }
        let node = Rc::new(CommonChatSchema::Ref {
            ref_: ref_.clone(),
            target: RefCell::new(None),
        });
        self.pending.push(Rc::clone(&node));
        Ok(node)
    }

    fn build_alternatives(
        &mut self,
        alts: &Json,
        path: &str,
        all: bool,
    ) -> Result<Rc<CommonChatSchema>> {
        if !alts.is_array() {
            return Err(Self::fail(path, "must be an array of schemas"));
        }
        if alts.empty() {
            return Err(Self::fail(path, "must not be empty"));
        }
        let mut children = Vec::with_capacity(alts.size());
        for (i, alt) in alts.iter().enumerate() {
            children.push(self.build_node(alt, &format!("{}/{}", path, i))?);
        }
        Ok(Rc::new(if all {
            CommonChatSchema::AllOf { children }
        } else {
            CommonChatSchema::AnyOf { children }
        }))
    }

    fn build_object(&mut self, schema: &Json, path: &str) -> Result<Rc<CommonChatSchema>> {
        let mut properties = Vec::new();
        let mut required = HashSet::new();
        if schema.contains("required") && schema.at("required")?.is_array() {
            for name in schema.at("required")?.iter() {
                if name.is_string() {
                    required.insert(name.get::<String>()?);
                }
            }
        }
        if schema.contains("properties") {
            let props = schema.at("properties")?;
            if !props.is_object() {
                return Err(Self::fail(path, "properties must be an object"));
            }
            for entry in props.items() {
                let name = entry.key().to_string();
                let prop_path = format!("{}/properties/{}", path, name);
                properties.push(CommonChatSchemaProperty {
                    schema: self.build_node(entry.value(), &prop_path)?,
                    required: required.contains(&name),
                    name,
                });
            }
        }

        let additional_properties = if schema.contains("additionalProperties") {
            let additional = schema.at("additionalProperties")?;
            if additional.is_boolean() {
                if additional.get::<bool>()? {
                    Some(Rc::new(CommonChatSchema::Any))
                } else {
                    None
                }
            } else if additional.is_object() {
                Some(self.build_node(additional, &format!("{}/additionalProperties", path))?)
            } else {
                return Err(Self::fail(
                    path,
                    "additionalProperties must be a boolean or a schema",
                ));
            }
        } else if !schema.contains("properties") {
            // {"type": "object"} on its own accepts any object
            Some(Rc::new(CommonChatSchema::Any))
        } else {
            None
        };

        Ok(Rc::new(CommonChatSchema::Object {
            properties,
            additional_properties,
        }))
    }

    fn build_array(&mut self, schema: &Json, path: &str) -> Result<Rc<CommonChatSchema>> {
        if schema.contains("items") || schema.contains("prefixItems") {
            // "items" wins when both are present; a schema instead of an array is the item schema
            let key = if schema.contains("items") { "items" } else { "prefixItems" };
            let items = schema.at(key)?;
            if items.is_array() {
                let mut tuple = Vec::with_capacity(items.size());
                for (i, item) in items.iter().enumerate() {
                    tuple.push(self.build_node(item, &format!("{}/{}/{}", path, key, i))?);
                }
                return Ok(Rc::new(CommonChatSchema::Tuple { items: tuple }));
            }
            let items = self.build_node(items, &format!("{}/{}", path, key))?;
            Ok(Rc::new(CommonChatSchema::Array {
                items,
                min_items: Self::get_count(schema, "minItems", path, 0)?,
                max_items: Self::get_count(schema, "maxItems", path, -1)?,
            }))
        } else {
            Ok(Rc::new(CommonChatSchema::Array {
                items: Rc::new(CommonChatSchema::Any),
                min_items: Self::get_count(schema, "minItems", path, 0)?,
                max_items: Self::get_count(schema, "maxItems", path, -1)?,
            }))
        }
    }

    fn build_string(&mut self, schema: &Json, path: &str) -> Result<Rc<CommonChatSchema>> {
        let pattern = if schema.contains("pattern") {
            let value = schema.at("pattern")?;
            if !value.is_string() {
                return Err(Self::fail(path, "pattern must be a string"));
            }
            value.get::<String>()?
        } else {
            String::new()
        };
        Ok(Rc::new(CommonChatSchema::String {
            pattern,
            format: Self::get_format(schema, path)?,
            min_length: Self::get_count(schema, "minLength", path, 0)?,
            max_length: Self::get_count(schema, "maxLength", path, -1)?,
        }))
    }

    fn build_integer(&self, schema: &Json, path: &str) -> Result<Rc<CommonChatSchema>> {
        let minimum = if schema.contains("minimum") {
            Self::get_bound(schema, "minimum", path, true)?
        } else if schema.contains("exclusiveMinimum") {
            Self::get_bound(schema, "exclusiveMinimum", path, false)? + 1
        } else {
            i64::MIN
        };
        let maximum = if schema.contains("maximum") {
            Self::get_bound(schema, "maximum", path, false)?
        } else if schema.contains("exclusiveMaximum") {
            Self::get_bound(schema, "exclusiveMaximum", path, true)? - 1
        } else {
            i64::MAX
        };
        Ok(Rc::new(CommonChatSchema::Integer { minimum, maximum }))
    }

    fn build_node(&mut self, schema: &Json, path: &str) -> Result<Rc<CommonChatSchema>> {
        self.depth += 1;
        if self.depth > COMMON_RECURSIVE_LIMIT {
            self.depth -= 1;
            return Err(depth_error());
        }
        let result = self.build_node_impl(schema, path);
        self.depth -= 1;
        result
    }

    fn build_node_impl(&mut self, schema: &Json, path: &str) -> Result<Rc<CommonChatSchema>> {
        if schema.is_boolean() {
            if schema.get::<bool>()? {
                return Ok(Rc::new(CommonChatSchema::Any));
            }
            return Err(Self::fail(path, "false schema is not supported"));
        }
        if !schema.is_object() {
            return Err(Self::fail(path, "schema must be an object"));
        }
        if schema.contains("$ref") {
            return self.build_ref(schema.at("$ref")?, path);
        }
        if schema.contains("oneOf") || schema.contains("anyOf") {
            let key = if schema.contains("oneOf") { "oneOf" } else { "anyOf" };
            let alts = schema.at(key)?;
            return self.build_alternatives(alts, &format!("{}/{}", path, key), false);
        }

        let type_json = if schema.contains("type") {
            Some(schema.at("type")?)
        } else {
            None
        };
        if type_json.map_or(false, Json::is_array) {
            let types = type_json.expect("type is an array");
            if types.empty() {
                return Err(Self::fail(path, "type must not be empty"));
            }
            let mut children = Vec::with_capacity(types.size());
            for (i, t) in types.iter().enumerate() {
                let mut alt = schema.clone();
                alt.set(Item::new("type", t.clone()))?;
                children.push(self.build_node(&alt, &format!("{}/type/{}", path, i))?);
            }
            return Ok(Rc::new(CommonChatSchema::AnyOf { children }));
        }
        if schema.contains("const") {
            return Ok(Rc::new(CommonChatSchema::Const {
                value: schema.at("const")?.clone(),
            }));
        }
        if schema.contains("enum") {
            let values_json = schema.at("enum")?;
            if !values_json.is_array() || values_json.empty() {
                return Err(Self::fail(path, "enum must be a non-empty array"));
            }
            let values = values_json.iter().cloned().collect();
            return Ok(Rc::new(CommonChatSchema::Enum { values }));
        }
        if type_json.map_or(false, |t| !t.is_null() && !t.is_string()) {
            return Err(Self::fail(path, "type must be a string or an array of strings"));
        }

        let type_name = match type_json {
            Some(t) if t.is_string() => t.get::<String>()?,
            _ => String::new(),
        };
        let has_properties = schema.contains("properties")
            || (schema.contains("additionalProperties")
                && *schema.at("additionalProperties")? != Json::Bool(true));

        if type_name.is_empty() {
            if has_properties {
                return self.build_object(schema, path);
            }
            if schema.contains("allOf") {
                let alts = schema.at("allOf")?;
                return self.build_alternatives(alts, &format!("{}/allOf", path), true);
            }
            if schema.contains("items") || schema.contains("prefixItems") {
                return self.build_array(schema, path);
            }
            if schema.contains("pattern")
                || schema.contains("minLength")
                || schema.contains("maxLength")
                || Self::get_format(schema, path)? != StringFormat::None
            {
                return self.build_string(schema, path);
            }
            return Ok(Rc::new(CommonChatSchema::Any));
        }
        match type_name.as_str() {
            "object" => {
                if !has_properties && schema.contains("allOf") {
                    let alts = schema.at("allOf")?;
                    return self.build_alternatives(alts, &format!("{}/allOf", path), true);
                }
                self.build_object(schema, path)
            }
            "string" => {
                if schema.contains("allOf") {
                    let alts = schema.at("allOf")?;
                    return self.build_alternatives(alts, &format!("{}/allOf", path), true);
                }
                self.build_string(schema, path)
            }
            "array" => self.build_array(schema, path),
            "integer" => self.build_integer(schema, path),
            "number" => Ok(Rc::new(CommonChatSchema::Number)),
            "boolean" => Ok(Rc::new(CommonChatSchema::Boolean)),
            "null" => Ok(Rc::new(CommonChatSchema::Null)),
            _ => Err(Self::fail(path, &format!("unrecognized type {}", type_name))),
        }
    }

    fn build(mut self) -> Result<CommonChatSchemaDocument> {
        let root_json = self.root;
        let root = self.build_node(root_json, "#")?;
        // ref nodes get their target once every $ref is built
        for node in &self.pending {
            if let CommonChatSchema::Ref { ref_, target } = &**node {
                if let Some(resolved) = self.refs.get(ref_) {
                    *target.borrow_mut() = Some(Rc::clone(resolved));
                }
            }
        }
        Ok(CommonChatSchemaDocument {
            root,
            refs: self.refs,
        })
    }
}

/// Parse a JSON schema into the supported subset.
pub fn common_chat_schema_from_json(schema: &Json) -> Result<CommonChatSchemaDocument> {
    Builder::new(schema).build()
}

// ---------------------------------------------------------------------------
// grammar rules
// ---------------------------------------------------------------------------

const SPACE_RULE: &str = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
    content: &'static str,
    deps: &'static [&'static str],
}

const PRIMITIVE_RULES: &[(&str, BuiltinRule)] = &[
    ("boolean", BuiltinRule { content: "(\"true\" | \"false\")", deps: &[] }),
    ("decimal-part", BuiltinRule { content: "[0-9]{1,16}", deps: &[] }),
    ("integral-part", BuiltinRule { content: "[0] | [1-9] [0-9]{0,15}", deps: &[] }),
    ("number", BuiltinRule {
        content: "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?",
        deps: &["integral-part", "decimal-part"],
    }),
    ("integer", BuiltinRule { content: "(\"-\"? integral-part)", deps: &["integral-part"] }),
    ("value", BuiltinRule {
        content: "object | array | string | number | boolean | null",
        deps: &["object", "array", "string", "number", "boolean", "null"],
    }),
    ("object", BuiltinRule {
        content: "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"",
        deps: &["string", "value"],
    }),
    ("array", BuiltinRule {
        content: "\"[\" space ( value (\",\" space value)* )? space \"]\"",
        deps: &["value"],
    }),
    ("uuid", BuiltinRule {
        content: "\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"",
        deps: &[],
    }),
    ("char", BuiltinRule {
        content: "[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})",
        deps: &[],
    }),
    ("string", BuiltinRule { content: "\"\\\"\" char* \"\\\"\"", deps: &["char"] }),
    ("null", BuiltinRule { content: "\"null\"", deps: &[] }),
];

const STRING_FORMAT_RULES: &[(&str, BuiltinRule)] = &[
    ("date", BuiltinRule {
        content: "[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )",
        deps: &[],
    }),
    ("time", BuiltinRule {
        content: "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )",
        deps: &[],
    }),
    ("date-time", BuiltinRule { content: "date \"T\" time", deps: &["date", "time"] }),
    ("date-string", BuiltinRule { content: "\"\\\"\" date \"\\\"\"", deps: &["date"] }),
    ("time-string", BuiltinRule { content: "\"\\\"\" time \"\\\"\"", deps: &["time"] }),
    ("date-time-string", BuiltinRule {
        content: "\"\\\"\" date-time \"\\\"\"",
        deps: &["date-time"],
    }),
];

fn find_builtin(name: &str) -> Option<&'static BuiltinRule> {
    PRIMITIVE_RULES
        .iter()
        .chain(STRING_FORMAT_RULES.iter())
        .find(|(key, _)| *key == name)
        .map(|(_, rule)| rule)
}

fn builtin(name: &str) -> &'static BuiltinRule {
    find_builtin(name).expect("unknown builtin grammar rule")
}

fn is_reserved_name(name: &str) -> bool {
    name == "root" || find_builtin(name).is_some()
}

/// Escape a literal for use inside a GBNF grammar.
pub fn gbnf_format_literal(literal: &str) -> String {
    let mut escaped = String::with_capacity(literal.len() + 2);
    for c in literal.chars() {
        match c {
            '\r' => escaped.push_str("\\r"),
            '\n' => escaped.push_str("\\n"),
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            _ => escaped.push(c),
        }
    }
    format!("\"{}\"", escaped)
}

/// Sanitize a name for use as a GBNF rule name.
pub fn gbnf_format_rule_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut in_invalid = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' {
            out.push(c);
            in_invalid = false;
        } else if !in_invalid {
            out.push('-');
            in_invalid = true;
        }
    }
    out
}

fn build_repetition(item_rule: &str, min_items: i32, max_items: i32, separator_rule: &str) -> String {
    let has_max = max_items != i32::MAX;
    if max_items == 0 {
        return String::new();
    }
    if min_items == 0 && max_items == 1 {
        return format!("{}?", item_rule);
    }
    if separator_rule.is_empty() {
        if min_items == 1 && !has_max {
            return format!("{}+", item_rule);
        }
        if min_items == 0 && !has_max {
            return format!("{}*", item_rule);
        }
        return format!(
            "{}{{{},{}}}",
            item_rule,
            min_items,
            if has_max { max_items.to_string() } else { String::new() }
        );
    }
    let inner = format!("({} {})", separator_rule, item_rule);
    let result = format!(
        "{} {}",
        item_rule,
        build_repetition(
            &inner,
            if min_items == 0 { 0 } else { min_items - 1 },
            if has_max { max_items - 1 } else { max_items },
            "",
        )
    );
    if min_items == 0 {
        format!("({})?", result)
    } else {
        result
    }
}

fn digit_range(out: &mut String, from: u8, to: u8) {
    out.push('[');
    if from == to {
        out.push(from as char);
    } else {
        out.push(from as char);
        out.push('-');
        out.push(to as char);
    }
    out.push(']');
}

fn more_digits(out: &mut String, min_digits: i32, max_digits: i32) {
    out.push_str("[0-9]");
    if min_digits == max_digits && min_digits == 1 {
        return;
    }
    out.push('{');
    out.push_str(&min_digits.to_string());
    if max_digits != min_digits {
        out.push(',');
        if max_digits != i32::MAX {
            out.push_str(&max_digits.to_string());
        }
    }
    out.push('}');
}

fn uniform_range(out: &mut String, from: &str, to: &str) {
    let fb = from.as_bytes();
    let tb = to.as_bytes();
    let mut i = 0;
    while i < fb.len() && i < tb.len() && fb[i] == tb[i] {
        i += 1;
    }
    if i > 0 {
        out.push('"');
        out.push_str(&from[..i]);
        out.push('"');
    }
    if i >= fb.len() || i >= tb.len() {
        return;
    }
    if i > 0 {
        out.push(' ');
    }
    let sub_len = fb.len() - i - 1;
    if sub_len == 0 {
        out.push('[');
        out.push(fb[i] as char);
        out.push('-');
        out.push(tb[i] as char);
        out.push(']');
        return;
    }
    let from_sub = &from[i + 1..];
    let to_sub = &to[i + 1..];
    let sub_zeros = string_repeat("0", sub_len);
    let sub_nines = string_repeat("9", sub_len);
    let mut to_reached = false;
    out.push('(');
    if from_sub == sub_zeros {
        digit_range(out, fb[i], tb[i] - 1);
        out.push(' ');
        more_digits(out, sub_len as i32, sub_len as i32);
    } else {
        out.push('[');
        out.push(fb[i] as char);
        out.push(']');
        out.push(' ');
        out.push('(');
        uniform_range(out, from_sub, &sub_nines);
        out.push(')');
        if fb[i] < tb[i] - 1 {
            out.push_str(" | ");
            if to_sub == sub_nines {
                digit_range(out, fb[i] + 1, tb[i]);
                to_reached = true;
            } else {
                digit_range(out, fb[i] + 1, tb[i] - 1);
            }
            out.push(' ');
            more_digits(out, sub_len as i32, sub_len as i32);
        }
    }
    if !to_reached {
        out.push_str(" | ");
        digit_range(out, tb[i], tb[i]);
        out.push(' ');
        uniform_range(out, &sub_zeros, to_sub);
    }
    out.push(')');
}

fn build_min_max_int(
    mut min_value: i64,
    max_value: i64,
    out: &mut String,
    decimals_left: i32,
    top_level: bool,
) -> Result<()> {
    let has_min = min_value != i64::MIN;
    let has_max = max_value != i64::MAX;

    if has_min && has_max {
        if min_value < 0 && max_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(-max_value, -min_value, out, decimals_left, true)?;
            out.push(')');
            return Ok(());
        }
        if min_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(0, -min_value, out, decimals_left, true)?;
            out.push_str(") | ");
            min_value = 0;
        }
        let mut min_s = min_value.to_string();
        let max_s = max_value.to_string();
        let min_digits = min_s.len() as i32;
        let max_digits = max_s.len() as i32;
        let mut digits = min_digits;
        while digits < max_digits {
            uniform_range(out, &min_s, &string_repeat("9", digits as usize));
            min_s = format!("1{}", string_repeat("0", digits as usize));
            out.push_str(" | ");
            digits += 1;
        }
        uniform_range(out, &min_s, &max_s);
        return Ok(());
    }

    let less_decimals = max(decimals_left - 1, 1);
    if has_min {
        if min_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(i64::MIN, -min_value, out, decimals_left, false)?;
            out.push_str(") | [0] | [1-9] ");
            more_digits(out, 0, decimals_left - 1);
        } else if min_value == 0 {
            if top_level {
                out.push_str("[0] | [1-9] ");
                more_digits(out, 0, less_decimals);
            } else {
                more_digits(out, 1, decimals_left);
            }
        } else if min_value <= 9 {
            let c = b'0' + min_value as u8;
            let range_start = if top_level { b'1' } else { b'0' };
            if c > range_start {
                digit_range(out, range_start, c - 1);
                out.push(' ');
                more_digits(out, 1, less_decimals);
                out.push_str(" | ");
            }
            digit_range(out, c, b'9');
            out.push(' ');
            more_digits(out, 0, less_decimals);
        } else {
            let min_s = min_value.to_string();
            let len = min_s.len() as i32;
            let c = min_s.as_bytes()[0];
            if c > b'1' {
                digit_range(out, if top_level { b'1' } else { b'0' }, c - 1);
                out.push(' ');
                more_digits(out, len, less_decimals);
                out.push_str(" | ");
            }
            digit_range(out, c, c);
            out.push_str(" (");
            let rest = min_s[1..]
                .parse::<i64>()
                .map_err(|_| Error::new("invalid integer bound"))?;
            build_min_max_int(rest, i64::MAX, out, less_decimals, false)?;
            out.push(')');
            if c < b'9' {
                out.push_str(" | ");
                digit_range(out, c + 1, b'9');
                out.push(' ');
                more_digits(out, len - 1, less_decimals);
            }
        }
        return Ok(());
    }

    if has_max {
        if max_value >= 0 {
            if top_level {
                out.push_str("\"-\" [1-9] ");
                more_digits(out, 0, less_decimals);
                out.push_str(" | ");
            }
            build_min_max_int(0, max_value, out, decimals_left, true)?;
        } else {
            out.push_str("\"-\" (");
            build_min_max_int(-max_value, i64::MAX, out, decimals_left, false)?;
            out.push(')');
        }
        return Ok(());
    }

    Err(Error::new("At least one of min_value or max_value must be set"))
}

// ---------------------------------------------------------------------------
// regex pattern parsing
// ---------------------------------------------------------------------------

enum PatternError {
    Unsupported(String),
    Invalid(String),
}

type PatternResult<T> = std::result::Result<T, PatternError>;

fn escape_error(pattern: &[u8], pos: usize) -> String {
    let end = min(pos + 2, pattern.len());
    String::from_utf8_lossy(&pattern[pos..end]).into_owned()
}

fn gbnf_escape_length(pattern: &[u8], pos: usize) -> usize {
    if pos + 1 >= pattern.len() || pattern[pos] != b'\\' {
        return 0;
    }
    let n_hex = match pattern[pos + 1] {
        b'x' => 2,
        b'u' => 4,
        b'U' => 8,
        b't' | b'r' | b'n' | b'\\' | b'"' | b'[' | b']' | b'-' => return 2,
        _ => return 0,
    };
    if pos + 2 + n_hex > pattern.len() {
        return 0;
    }
    if !pattern[pos + 2..pos + 2 + n_hex].iter().all(u8::is_ascii_hexdigit) {
        return 0;
    }
    2 + n_hex
}

fn is_non_literal(c: u8) -> bool {
    matches!(
        c,
        b'|' | b'.' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'*' | b'+' | b'?' | b'^' | b'$'
    )
}

fn is_escaped_in_regexps_but_not_literals(c: u8) -> bool {
    matches!(c, b'^' | b'$' | b'.' | b'[' | b']' | b'(' | b')' | b'|' | b'{' | b'}' | b'*' | b'+' | b'?')
}

type LiteralOrRule = (String, bool); // (text, is_literal)

struct PatternParser<'a, 'b> {
    conv: &'a mut Converter,
    pattern: &'b [u8],
    i: usize,
    paren_depth: i32,
    sub_rule_ids: BTreeMap<String, String>,
    name: &'b str,
}

impl<'a, 'b> PatternParser<'a, 'b> {
    fn new(conv: &'a mut Converter, pattern: &'b [u8], name: &'b str) -> Self {
        Self {
            conv,
            pattern,
            i: 0,
            paren_depth: 0,
            sub_rule_ids: BTreeMap::new(),
            name,
        }
    }

    fn to_rule(ls: &LiteralOrRule) -> String {
        if ls.1 {
            format!("\"{}\"", ls.0)
        } else {
            ls.0.clone()
        }
    }

    fn len(&self) -> usize {
        self.pattern.len()
    }

    fn get_dot(&mut self) -> PatternResult<String> {
        let rule = if self.conv.dotall {
            r"[\U00000000-\U0010FFFF]"
        } else {
            r"[^\x0A\x0D]"
        };
        Ok(self.conv.add_rule("dot", rule))
    }

    fn join_seq(seq: Vec<LiteralOrRule>) -> LiteralOrRule {
        let mut ret: Vec<LiteralOrRule> = Vec::new();
        let mut literal = String::new();
        for item in seq {
            if item.1 {
                literal.push_str(&item.0);
            } else {
                if !literal.is_empty() {
                    ret.push((std::mem::take(&mut literal), true));
                }
                ret.push(item);
            }
        }
        if !literal.is_empty() {
            ret.push((literal, true));
        }
        let results: Vec<String> = ret.iter().map(Self::to_rule).collect();
        (string_join(&results, " "), false)
    }

    fn transform(&mut self) -> PatternResult<LiteralOrRule> {
        let mut seq: Vec<LiteralOrRule> = Vec::new();
        while self.i < self.len() {
            let c = self.pattern[self.i];
            match c {
                b'.' => {
                    let dot = self.get_dot()?;
                    seq.push((dot, false));
                    self.i += 1;
                }
                b'(' => {
                    self.i += 1;
                    if self.i < self.len() && self.pattern[self.i] == b'?' {
                        if self.i + 1 < self.len() && self.pattern[self.i + 1] == b':' {
                            self.i += 2;
                        } else {
                            return Err(PatternError::Unsupported("unsupported group syntax".into()));
                        }
                    }
                    self.paren_depth += 1;
                    if self.paren_depth as usize > COMMON_RECURSIVE_LIMIT {
                        return Err(PatternError::Unsupported("pattern nesting too deep".into()));
                    }
                    let inner = self.transform()?;
                    let text = Self::to_rule(&inner);
                    seq.push((format!("({})", text), false));
                }
                b')' => {
                    self.i += 1;
                    if self.paren_depth == 0 {
                        return Err(PatternError::Invalid("unbalanced parentheses".into()));
                    }
                    self.paren_depth -= 1;
                    return Ok(Self::join_seq(seq));
                }
                b'^' | b'$' => {
                    return Err(PatternError::Unsupported("anchor inside the pattern".into()));
                }
                b'[' => {
                    let mut square_brackets = String::from("[");
                    self.i += 1;
                    while self.i < self.len() && self.pattern[self.i] != b']' {
                        if self.pattern[self.i] == b'\\' {
                            let n = gbnf_escape_length(self.pattern, self.i);
                            if n == 0 {
                                return Err(PatternError::Unsupported(format!(
                                    "unsupported escape in character class: {}",
                                    escape_error(self.pattern, self.i)
                                )));
                            }
                            square_brackets.push_str(&String::from_utf8_lossy(
                                &self.pattern[self.i..self.i + n],
                            ));
                            self.i += n;
                        } else {
                            square_brackets.push(self.pattern[self.i] as char);
                            self.i += 1;
                        }
                    }
                    if self.i >= self.len() {
                        return Err(PatternError::Invalid("unterminated character class".into()));
                    }
                    square_brackets.push(']');
                    self.i += 1;
                    seq.push((square_brackets, false));
                }
                b'|' => {
                    seq.push(("|".into(), false));
                    self.i += 1;
                }
                b'*' | b'+' | b'?' => {
                    if seq.is_empty() {
                        return Err(PatternError::Invalid("nothing to repeat".into()));
                    }
                    let last = seq.last_mut().expect("seq is not empty");
                    let text = Self::to_rule(last);
                    last.0 = format!("{}{}", text, c as char);
                    last.1 = false;
                    self.i += 1;
                }
                b'{' => {
                    let mut curly_brackets = String::from("{");
                    self.i += 1;
                    while self.i < self.len() && self.pattern[self.i] != b'}' {
                        curly_brackets.push(self.pattern[self.i] as char);
                        self.i += 1;
                    }
                    if self.i >= self.len() {
                        return Err(PatternError::Unsupported("unterminated curly brackets".into()));
                    }
                    curly_brackets.push('}');
                    self.i += 1;
                    let inner = &curly_brackets[1..curly_brackets.len() - 1];
                    let nums: Vec<&str> = inner.split(',').collect();
                    let mut min_times = 0;
                    let mut max_times = i32::MAX;
                    if nums.len() != 1 && nums.len() != 2 {
                        return Err(PatternError::Unsupported(
                            "wrong number of values in curly brackets".into(),
                        ));
                    }
                    let parse = |s: &str| s.parse::<i32>().ok();
                    if nums.len() == 1 {
                        match parse(nums[0]) {
                            Some(v) => {
                                min_times = v;
                                max_times = v;
                            }
                            None => {
                                return Err(PatternError::Unsupported(
                                    "invalid number in curly brackets".into(),
                                ));
                            }
                        }
                    } else {
                        if !nums[0].is_empty() {
                            match parse(nums[0]) {
                                Some(v) => min_times = v,
                                None => {
                                    return Err(PatternError::Unsupported(
                                        "invalid number in curly brackets".into(),
                                    ));
                                }
                            }
                        }
                        if !nums[1].is_empty() {
                            match parse(nums[1]) {
                                Some(v) => max_times = v,
                                None => {
                                    return Err(PatternError::Unsupported(
                                        "invalid number in curly brackets".into(),
                                    ));
                                }
                            }
                        }
                    }
                    if seq.is_empty() {
                        return Err(PatternError::Invalid("nothing to repeat".into()));
                    }
                    let last = seq.last_mut().expect("seq is not empty");
                    if !last.1 {
                        let sub = last.0.clone();
                        if !self.sub_rule_ids.contains_key(&sub) {
                            self.sub_rule_ids.insert(sub.clone(), String::new());
                        }
                        if self.sub_rule_ids[&sub].is_empty() {
                            let n = self.sub_rule_ids.len();
                            let rule_name = format!("{}-{}", self.name, n);
                            let id = self.conv.add_rule(&rule_name, &sub);
                            self.sub_rule_ids.insert(sub.clone(), id);
                        }
                        last.0 = self.sub_rule_ids[&sub].clone();
                    }
                    let last = seq.last_mut().expect("seq is not empty");
                    let sub = if last.1 {
                        format!("\"{}\"", last.0)
                    } else {
                        last.0.clone()
                    };
                    last.0 = build_repetition(&sub, min_times, max_times, "");
                    last.1 = false;
                }
                _ => {
                    let mut literal = String::new();
                    while self.i < self.len() {
                        if self.pattern[self.i] == b'\\' {
                            if self.i == self.len() - 1 {
                                return Err(PatternError::Invalid("trailing backslash".into()));
                            }
                            let next = self.pattern[self.i + 1];
                            if is_escaped_in_regexps_but_not_literals(next) {
                                self.i += 1;
                                literal.push(self.pattern[self.i] as char);
                                self.i += 1;
                            } else {
                                let n = gbnf_escape_length(self.pattern, self.i);
                                if n == 0 {
                                    return Err(PatternError::Unsupported(format!(
                                        "unsupported escape: {}",
                                        escape_error(self.pattern, self.i)
                                    )));
                                }
                                literal.push_str(&String::from_utf8_lossy(
                                    &self.pattern[self.i..self.i + n],
                                ));
                                self.i += n;
                            }
                        } else if self.pattern[self.i] == b'"' {
                            literal.push_str("\\\"");
                            self.i += 1;
                        } else if !is_non_literal(self.pattern[self.i])
                            && (self.i == self.len() - 1
                                || literal.is_empty()
                                || self.pattern[self.i + 1] == b'.'
                                || !is_non_literal(self.pattern[self.i + 1]))
                        {
                            literal.push(self.pattern[self.i] as char);
                            self.i += 1;
                        } else {
                            break;
                        }
                    }
                    if literal.is_empty() {
                        return Err(PatternError::Unsupported(format!(
                            "unsupported character: {}",
                            c as char
                        )));
                    }
                    seq.push((literal, true));
                }
            }
        }
        Ok(Self::join_seq(seq))
    }
}

// ---------------------------------------------------------------------------
// trie used for excluding property names
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TrieNode {
    children: BTreeMap<u32, usize>,
    pattern: Option<usize>,
}

#[derive(Clone, Debug)]
struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    fn new(words: &[String]) -> Self {
        let mut trie = Self {
            nodes: vec![TrieNode {
                children: BTreeMap::new(),
                pattern: None,
            }],
        };
        for (i, word) in words.iter().enumerate() {
            trie.insert(word, i);
        }
        trie
    }

    fn insert(&mut self, word: &str, pattern: usize) {
        let mut idx = 0;
        for ch in word.chars() {
            let sym = ch as u32;
            let next = match self.nodes[idx].children.get(&sym) {
                Some(&child) => child,
                None => {
                    let child = self.nodes.len();
                    self.nodes.push(TrieNode {
                        children: BTreeMap::new(),
                        pattern: None,
                    });
                    self.nodes[idx].children.insert(sym, child);
                    child
                }
            };
            idx = next;
        }
        if self.nodes[idx].pattern.is_none() {
            self.nodes[idx].pattern = Some(pattern);
        }
    }
}

fn visit_trie(
    trie: &Trie,
    out: &mut String,
    idx: usize,
    depth: usize,
    char_rule: &str,
) -> Result<()> {
    if depth >= COMMON_RECURSIVE_LIMIT {
        return Err(Error::new(format!(
            "JSON schema property name is longer than {} chars",
            COMMON_RECURSIVE_LIMIT
        )));
    }
    let node = &trie.nodes[idx];
    let mut rejects = String::new();
    let mut first = true;
    for (&sym, &child) in &node.children {
        let c = unicode_cpt_to_utf8(sym)?;
        rejects.push_str(&c);
        if first {
            first = false;
        } else {
            out.push_str(" | ");
        }
        out.push('[');
        out.push_str(&c);
        out.push(']');
        if !trie.nodes[child].children.is_empty() {
            out.push_str(" (");
            visit_trie(trie, out, child, depth + 1, char_rule)?;
            out.push(')');
        } else {
            out.push(' ');
            out.push_str(char_rule);
            out.push('+');
        }
    }
    if !node.children.is_empty() {
        out.push_str(" | [^\"");
        out.push_str(&rejects);
        out.push_str("] ");
        out.push_str(char_rule);
        out.push('*');
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// schema to grammar converter
// ---------------------------------------------------------------------------

pub struct CommonGrammarOptions {
    pub dotall: bool,
}

impl Default for CommonGrammarOptions {
    fn default() -> Self {
        Self { dotall: false }
    }
}

pub struct CommonGrammarBuilder<'a> {
    conv: RefCell<&'a mut Converter>,
}

impl CommonGrammarBuilder<'_> {
    pub fn add_rule(&self, name: &str, rule: &str) -> String {
        self.conv.borrow_mut().add_rule(name, rule)
    }

    pub fn add_schema(&self, name: &str, schema: &CommonChatSchema) -> Result<String> {
        self.conv
            .borrow_mut()
            .visit(schema, if name == "root" { "" } else { name })
    }
}

struct Converter {
    dotall: bool,
    rules: BTreeMap<String, String>,
    refs_being_resolved: BTreeSet<String>,
    errors: Vec<String>,
    warnings: Vec<String>,
    depth: usize,
}

impl Converter {
    fn new(dotall: bool) -> Self {
        let mut rules = BTreeMap::new();
        rules.insert("space".to_string(), SPACE_RULE.to_string());
        Self {
            dotall,
            rules,
            refs_being_resolved: BTreeSet::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            depth: 0,
        }
    }

    fn add_rule(&mut self, name: &str, rule: &str) -> String {
        let esc_name = gbnf_format_rule_name(name);
        if !self.rules.contains_key(&esc_name) || self.rules[&esc_name] == rule {
            self.rules.insert(esc_name.clone(), rule.to_string());
            return esc_name;
        }
        let mut i = 0;
        loop {
            let key = format!("{}{}", esc_name, i);
            if !self.rules.contains_key(&key) || self.rules[&key] == rule {
                self.rules.insert(key.clone(), rule.to_string());
                return key;
            }
            i += 1;
        }
    }

    fn generate_union_rule(&mut self, name: &str, alts: &[Rc<CommonChatSchema>]) -> Result<String> {
        let prefix = format!("{}{}", name, if name.is_empty() { "alternative-" } else { "-" });
        let mut rules = Vec::with_capacity(alts.len());
        for (i, alt) in alts.iter().enumerate() {
            rules.push(self.visit(alt, &format!("{}{}", prefix, i))?);
        }
        Ok(string_join(&rules, " | "))
    }

    fn visit_pattern(&mut self, pattern: &str, name: &str) -> Result<String> {
        let snapshot = self.rules.clone();
        match self.pattern_to_rule(pattern, name) {
            Ok(rule) => Ok(rule),
            Err(PatternError::Unsupported(msg)) => {
                self.rules = snapshot;
                self.warnings.push(format!(
                    "pattern {} is not supported ({}), accepting any string",
                    pattern, msg
                ));
                let string_rule = self.add_primitive("string", builtin("string"))?;
                Ok(self.add_rule(name, &string_rule))
            }
            Err(PatternError::Invalid(msg)) => {
                self.rules = snapshot;
                self.errors.push(format!("Invalid pattern {}: {}", pattern, msg));
                Ok(String::new())
            }
        }
    }

    fn pattern_to_rule(&mut self, pattern: &str, name: &str) -> PatternResult<String> {
        if pattern.len() < 2 || !pattern.starts_with('^') || !pattern.ends_with('$') {
            return Err(PatternError::Unsupported("not anchored with '^' and '$'".into()));
        }
        let sub = &pattern.as_bytes()[1..pattern.len() - 1];
        let mut parser = PatternParser::new(self, sub, name);
        let rule_or = parser.transform()?;
        if parser.paren_depth != 0 {
            return Err(PatternError::Invalid("unbalanced parentheses".into()));
        }
        let rule = PatternParser::to_rule(&rule_or);
        let content = format!("\"\\\"\" ({}) \"\\\"\"", rule);
        Ok(parser.conv.add_rule(name, &content))
    }

    fn not_strings(&mut self, strings: &[String]) -> Result<String> {
        let trie = Trie::new(strings);
        let char_rule = self.add_primitive("char", builtin("char"))?;
        let mut out = String::from("[\"] ( ");
        visit_trie(&trie, &mut out, 0, 0, &char_rule)?;
        out.push_str(" )");
        if trie.nodes[0].pattern.is_none() {
            out.push('?');
        }
        out.push_str(" [\"]");
        Ok(out)
    }

    fn resolve_ref(
        &mut self,
        ref_: &str,
        target: &RefCell<Option<Rc<CommonChatSchema>>>,
    ) -> Result<String> {
        let fragment = match ref_.find('#') {
            Some(pos) => &ref_[pos + 1..],
            None => ref_,
        };
        let mut ref_name = format!("ref{}", gbnf_format_rule_name(fragment));
        if !self.rules.contains_key(&ref_name) && !self.refs_being_resolved.contains(ref_) {
            let target = target.borrow().as_ref().cloned();
            let Some(target) = target else {
                self.errors.push(format!("Unresolved $ref {}", ref_));
                return Ok(String::new());
            };
            self.refs_being_resolved.insert(ref_.to_string());
            ref_name = self.visit(&target, &ref_name)?;
            self.refs_being_resolved.remove(ref_);
        }
        Ok(ref_name)
    }

    fn build_object_rule(
        &mut self,
        properties: &[(String, Rc<CommonChatSchema>)],
        required: &HashSet<String>,
        name: &str,
        additional_properties: Option<&CommonChatSchema>,
    ) -> Result<String> {
        let mut required_props: Vec<String> = Vec::new();
        let mut optional_props: Vec<String> = Vec::new();
        let mut prop_kv_rule_names: BTreeMap<String, String> = BTreeMap::new();
        let mut prop_names: Vec<String> = Vec::new();
        let dash = if name.is_empty() { "" } else { "-" };
        for (prop_name, prop_schema) in properties {
            let base = format!("{}{}{}", name, dash, prop_name);
            let prop_rule_name = self.visit(prop_schema, &base)?;
            let literal = constant_rule(&Json::String(prop_name.clone()));
            let kv = format!("{} space \":\" space {}", literal, prop_rule_name);
            let kv_name = self.add_rule(&format!("{}-kv", base), &kv);
            prop_kv_rule_names.insert(prop_name.clone(), kv_name);
            if required.contains(prop_name) {
                required_props.push(prop_name.clone());
            } else {
                optional_props.push(prop_name.clone());
            }
            prop_names.push(prop_name.clone());
        }
        if let Some(additional) = additional_properties {
            let sub_name = format!("{}{}additional", name, dash);
            let value_rule = if additional.kind() != NodeKind::Any {
                self.visit(additional, &format!("{}-value", sub_name))?
            } else {
                self.add_primitive("value", builtin("value"))?
            };
            let key_rule = if prop_names.is_empty() {
                self.add_primitive("string", builtin("string"))?
            } else {
                let not = self.not_strings(&prop_names)?;
                self.add_rule(&format!("{}-k", sub_name), &not)
            };
            let kv_rule = self.add_rule(
                &format!("{}-kv", sub_name),
                &format!("{} \":\" space {}", key_rule, value_rule),
            );
            prop_kv_rule_names.insert("*".into(), kv_rule);
            optional_props.push("*".into());
        }

        if required_props.is_empty() && optional_props.is_empty() {
            return Ok("\"{\" space \"}\"".into());
        }

        let mut rule = String::from("\"{\" space ");
        for (i, prop) in required_props.iter().enumerate() {
            if i > 0 {
                rule.push_str(" \",\" space ");
            }
            rule.push_str(&prop_kv_rule_names[prop]);
        }

        if !optional_props.is_empty() {
            rule.push_str(" (");
            if !required_props.is_empty() {
                rule.push_str(" \",\" space ( ");
            }
            for (i, _) in optional_props.iter().enumerate() {
                if i > 0 {
                    rule.push_str(" | ");
                }
                let refs = recursive_refs(self, &optional_props[i..], false, name, &prop_kv_rule_names)?;
                rule.push_str(&refs);
            }
            if !required_props.is_empty() {
                rule.push_str(" )");
            }
            rule.push_str(" )?");
        }
        rule.push_str(" space \"}\"");
        Ok(rule)
    }

    fn add_primitive(&mut self, name: &str, rule: &BuiltinRule) -> Result<String> {
        let n = self.add_rule(name, rule.content);
        for &dep in rule.deps {
            let Some(dep_rule) = find_builtin(dep) else {
                self.errors.push(format!("Rule {} not known", dep));
                continue;
            };
            if !self.rules.contains_key(dep) {
                self.add_primitive(dep, dep_rule)?;
            }
        }
        Ok(n)
    }

    fn visit_primitive(&mut self, rule_name: &str, type_: &str) -> Result<String> {
        let name = if rule_name == "root" { "root" } else { type_ };
        self.add_primitive(name, builtin(type_))
    }

    fn visit_all_of(
        &mut self,
        children: &[Rc<CommonChatSchema>],
        name: &str,
        rule_name: &str,
    ) -> Result<String> {
        let mut required: HashSet<String> = HashSet::new();
        let mut properties: Vec<(String, Rc<CommonChatSchema>)> = Vec::new();
        let mut enum_values: BTreeMap<String, usize> = BTreeMap::new();
        let mut leaves: Vec<Rc<CommonChatSchema>> = Vec::new();
        for child in children {
            if child.kind() == NodeKind::AnyOf {
                if let CommonChatSchema::AnyOf { children: alts } = &**child {
                    for alt in alts {
                        add_component(
                            alt, false, 0, &mut required, &mut properties, &mut enum_values,
                            &mut leaves,
                        )?;
                    }
                }
            } else {
                add_component(
                    child, true, 0, &mut required, &mut properties, &mut enum_values, &mut leaves,
                )?;
            }
        }

        if !enum_values.is_empty() {
            let mut intersection = Vec::new();
            for (value, &count) in &enum_values {
                if count == children.len() {
                    intersection.push(value.clone());
                }
            }
            if !intersection.is_empty() {
                let content = format!("({})", string_join(&intersection, " | "));
                return Ok(self.add_rule(rule_name, &content));
            }
        }

        if properties.is_empty() && !leaves.is_empty() {
            if leaves.len() == 1 {
                return self.visit(&leaves[0], name);
            }
            let mut all_numeric = true;
            let mut any_integer = false;
            let mut lo = i64::MIN;
            let mut hi = i64::MAX;
            for leaf in &leaves {
                if let CommonChatSchema::Integer { minimum, maximum } = &**leaf {
                    any_integer = true;
                    lo = max(lo, *minimum);
                    hi = min(hi, *maximum);
                } else if leaf.kind() != NodeKind::Number {
                    all_numeric = false;
                }
            }
            if all_numeric {
                if lo > hi {
                    return Err(Error::new("allOf has an empty numeric range"));
                }
                if any_integer {
                    let merged = CommonChatSchema::Integer { minimum: lo, maximum: hi };
                    return self.visit(&merged, name);
                }
                return self.visit(&leaves[0], name);
            }
        }

        let content = self.build_object_rule(&properties, &required, name, None)?;
        Ok(self.add_rule(rule_name, &content))
    }

    fn visit(&mut self, schema: &CommonChatSchema, name: &str) -> Result<String> {
        self.depth += 1;
        if self.depth > COMMON_RECURSIVE_LIMIT {
            self.depth -= 1;
            return Err(depth_error());
        }
        let result = self.visit_impl(schema, name);
        self.depth -= 1;
        result
    }

    fn visit_impl(&mut self, schema: &CommonChatSchema, name: &str) -> Result<String> {
        let rule_name = if is_reserved_name(name) {
            format!("{}-", name)
        } else if name.is_empty() {
            "root".to_string()
        } else {
            name.to_string()
        };
        let dash = if name.is_empty() { "" } else { "-" };
        let sub_name = format!("{}{}", name, dash);

        match schema {
            CommonChatSchema::Ref { ref_, target } => {
                let content = self.resolve_ref(ref_, target)?;
                Ok(self.add_rule(&rule_name, &content))
            }
            CommonChatSchema::AnyOf { children } => {
                let content = self.generate_union_rule(name, children)?;
                Ok(self.add_rule(&rule_name, &content))
            }
            CommonChatSchema::AllOf { children } => self.visit_all_of(children, name, &rule_name),
            CommonChatSchema::Const { value } => {
                let content = constant_rule(value);
                Ok(self.add_rule(&rule_name, &content))
            }
            CommonChatSchema::Enum { values } => {
                let rules: Vec<String> = values.iter().map(constant_rule).collect();
                let content = format!("({})", string_join(&rules, " | "));
                Ok(self.add_rule(&rule_name, &content))
            }
            CommonChatSchema::Object {
                properties,
                additional_properties,
            } => {
                if properties.is_empty() {
                    if let Some(additional) = additional_properties {
                        if additional.kind() == NodeKind::Any {
                            let content = self.add_primitive("object", builtin("object"))?;
                            return Ok(self.add_rule(&rule_name, &content));
                        }
                    }
                }
                let props: Vec<(String, Rc<CommonChatSchema>)> = properties
                    .iter()
                    .map(|p| (p.name.clone(), Rc::clone(&p.schema)))
                    .collect();
                let mut required = HashSet::new();
                for p in properties {
                    if p.required {
                        required.insert(p.name.clone());
                    }
                }
                let content = self.build_object_rule(
                    &props,
                    &required,
                    name,
                    additional_properties.as_deref(),
                )?;
                Ok(self.add_rule(&rule_name, &content))
            }
            CommonChatSchema::Tuple { items } => {
                let mut rule = String::from("\"[\" space ");
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        rule.push_str(" \",\" space ");
                    }
                    let item_rule = self.visit(item, &format!("{}tuple-{}", sub_name, i))?;
                    rule.push_str(&item_rule);
                }
                rule.push_str(" space \"]\"");
                Ok(self.add_rule(&rule_name, &rule))
            }
            CommonChatSchema::Array {
                items,
                min_items,
                max_items,
            } => {
                if items.kind() == NodeKind::Any && *min_items == 0 && *max_items < 0 {
                    return self.visit_primitive(&rule_name, "array");
                }
                let item_rule = self.visit(items, &format!("{}item", sub_name))?;
                let max = if *max_items < 0 { i32::MAX } else { *max_items };
                let content = format!(
                    "\"[\" space {} space \"]\"",
                    build_repetition(&item_rule, *min_items, max, "\",\" space")
                );
                Ok(self.add_rule(&rule_name, &content))
            }
            CommonChatSchema::String {
                pattern,
                format,
                min_length,
                max_length,
            } => {
                if !pattern.is_empty() {
                    return self.visit_pattern(pattern, &rule_name);
                }
                if *format == StringFormat::Uuid {
                    return self.visit_primitive(&rule_name, "uuid");
                }
                if *format != StringFormat::None {
                    let prim_name = match format {
                        StringFormat::Date => "date-string",
                        StringFormat::Time => "time-string",
                        _ => "date-time-string",
                    };
                    let content = self.add_primitive(prim_name, builtin(prim_name))?;
                    return Ok(self.add_rule(&rule_name, &content));
                }
                if *min_length > 0 || *max_length >= 0 {
                    let char_rule = self.add_primitive("char", builtin("char"))?;
                    let max_len = if *max_length < 0 { i32::MAX } else { *max_length };
                    let content = format!(
                        "\"\\\"\" {} \"\\\"\"",
                        build_repetition(&char_rule, *min_length, max_len, "")
                    );
                    return Ok(self.add_rule(&rule_name, &content));
                }
                self.visit_primitive(&rule_name, "string")
            }
            CommonChatSchema::Integer { minimum, maximum } => {
                if *minimum == i64::MIN && *maximum == i64::MAX {
                    return self.visit_primitive(&rule_name, "integer");
                }
                let mut out = String::from("(");
                build_min_max_int(*minimum, *maximum, &mut out, 16, true)?;
                out.push(')');
                Ok(self.add_rule(&rule_name, &out))
            }
            CommonChatSchema::Number => self.visit_primitive(&rule_name, "number"),
            CommonChatSchema::Boolean => self.visit_primitive(&rule_name, "boolean"),
            CommonChatSchema::Null => self.visit_primitive(&rule_name, "null"),
            CommonChatSchema::Any => {
                let content = self.add_primitive("value", builtin("value"))?;
                Ok(self.add_rule(&rule_name, &content))
            }
        }
    }

    fn check_errors(&self) -> Result<()> {
        if !self.errors.is_empty() {
            return Err(Error::new(format!(
                "JSON schema conversion failed:\n{}",
                string_join(&self.errors, "\n")
            )));
        }
        if !self.warnings.is_empty() {
            eprintln!(
                "WARNING: JSON schema conversion was incomplete: {}",
                string_join(&self.warnings, "; ")
            );
        }
        Ok(())
    }

    fn format_grammar(&self) -> String {
        let mut out = String::new();
        for (name, rule) in &self.rules {
            out.push_str(&format!("{} ::= {}\n", name, rule));
        }
        out
    }
}

fn constant_rule(value: &Json) -> String {
    gbnf_format_literal(&value.dump(-1).unwrap_or_default())
}

fn add_component(
    comp: &Rc<CommonChatSchema>,
    is_required: bool,
    depth: usize,
    required: &mut HashSet<String>,
    properties: &mut Vec<(String, Rc<CommonChatSchema>)>,
    enum_values: &mut BTreeMap<String, usize>,
    leaves: &mut Vec<Rc<CommonChatSchema>>,
) -> Result<()> {
    if depth >= COMMON_RECURSIVE_LIMIT {
        return Err(Error::new(format!(
            "JSON schema is nested more than {} levels deep",
            COMMON_RECURSIVE_LIMIT
        )));
    }
    match &**comp {
        CommonChatSchema::Ref { target, .. } => {
            let target = target.borrow().as_ref().cloned();
            if let Some(target) = target {
                add_component(
                    &target, is_required, depth + 1, required, properties, enum_values, leaves,
                )?;
            }
        }
        CommonChatSchema::Object { properties: props, .. } => {
            for p in props {
                properties.push((p.name.clone(), Rc::clone(&p.schema)));
                if is_required {
                    required.insert(p.name.clone());
                }
            }
        }
        CommonChatSchema::Enum { values } => {
            for value in values {
                *enum_values.entry(constant_rule(value)).or_insert(0) += 1;
            }
        }
        _ if is_required && comp.kind() != NodeKind::Any => leaves.push(Rc::clone(comp)),
        _ => {}
    }
    Ok(())
}

fn recursive_refs(
    conv: &mut Converter,
    ks: &[String],
    first_is_optional: bool,
    name: &str,
    prop_kv_rule_names: &BTreeMap<String, String>,
) -> Result<String> {
    if ks.is_empty() {
        return Ok(String::new());
    }
    let k = &ks[0];
    let kv_rule_name = &prop_kv_rule_names[k];
    let comma_ref = format!("( \",\" space {} )", kv_rule_name);
    let mut res = if first_is_optional {
        format!("{}{}", comma_ref, if k == "*" { "*" } else { "?" })
    } else if k == "*" {
        format!("{} {}*", kv_rule_name, comma_ref)
    } else {
        kv_rule_name.clone()
    };
    if ks.len() > 1 {
        let rest = recursive_refs(conv, &ks[1..], true, name, prop_kv_rule_names)?;
        let dash = if name.is_empty() { "" } else { "-" };
        let rule_name = format!("{}{}{}-rest", name, dash, k);
        let added = conv.add_rule(&rule_name, &rest);
        res.push_str(&format!(" {}", added));
    }
    Ok(res)
}

/// Convert a JSON schema to a GBNF grammar.
///
/// `force_gbnf` is accepted for API parity; the llguidance path is not ported.
pub fn json_schema_to_grammar(schema: &Json, force_gbnf: bool) -> Result<String> {
    let _ = force_gbnf;
    let doc = common_chat_schema_from_json(schema)
        .map_err(|e| Error::new(format!("JSON schema conversion failed:\n{}", e.message)))?;
    json_schema_to_grammar_document(&doc)
}

/// Convert an already parsed schema document to a GBNF grammar.
pub fn json_schema_to_grammar_document(doc: &CommonChatSchemaDocument) -> Result<String> {
    let mut converter = Converter::new(false);
    converter.visit(&doc.root, "")?;
    converter.check_errors()?;
    Ok(converter.format_grammar())
}

/// Build a grammar from callbacks that add rules and schemas.
pub fn build_grammar<F>(cb: F, options: &CommonGrammarOptions) -> Result<String>
where
    F: FnOnce(&CommonGrammarBuilder<'_>) -> Result<()>,
{
    let mut converter = Converter::new(options.dotall);
    {
        let builder = CommonGrammarBuilder {
            conv: RefCell::new(&mut converter),
        };
        cb(&builder)?;
    }
    converter.check_errors()?;
    Ok(converter.format_grammar())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(text: &str) -> Json {
        Json::parse(text).expect("valid test JSON")
    }

    #[test]
    fn literal_escaping() {
        assert_eq!(gbnf_format_literal("a\"b\nc"), "\"a\\\"b\\nc\"");
        assert_eq!(gbnf_format_rule_name("a b..c"), "a-b-c");
    }

    #[test]
    fn repetition_shapes() {
        assert_eq!(build_repetition("x", 0, i32::MAX, ""), "x*");
        assert_eq!(build_repetition("x", 1, i32::MAX, ""), "x+");
        assert_eq!(build_repetition("x", 2, 4, "\",\" space"), "x (\",\" space x){1,3}");
    }

    #[test]
    fn string_schema_to_grammar() {
        let grammar = json_schema_to_grammar(&json(r#"{"type":"string"}"#), false).unwrap();
        assert!(grammar.contains("root ::= \"\\\"\" char* \"\\\"\""));
        assert!(grammar.contains("char ::="));
    }

    #[test]
    fn ref_and_value_types() {
        let doc = common_chat_schema_from_json(
            &json(r##"{"$ref":"#/$defs/x","$defs":{"x":{"type":"integer"}}}"##),
        )
        .unwrap();
        let types = doc.root.value_types().unwrap();
        assert!(types.is_only(ValueType::Integer));
        assert!(!doc.root.may_be_string().unwrap());
        let grammar = json_schema_to_grammar_document(&doc).unwrap();
        assert!(grammar.contains("integer ::="));
    }

    #[test]
    fn all_of_string_intersection() {
        let doc = common_chat_schema_from_json(
            &json(r#"{"allOf":[{"enum":["a","b"]},{"enum":["b","c"]}]}"#),
        )
        .unwrap();
        let grammar = json_schema_to_grammar_document(&doc).unwrap();
        assert!(grammar.contains("root ::= (\"\\\"b\\\"\")"));
    }

    #[test]
    fn object_properties_and_extras() {
        let grammar = json_schema_to_grammar(
            &json(
                r#"{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"}},"required":["a"]}"#,
            ),
            false,
        )
        .unwrap();
        assert!(
            grammar.contains("root ::= \"{\" space a-kv ( \",\" space ( b-kv ) )? space \"}\""),
            "{}",
            grammar
        );

        let grammar = json_schema_to_grammar(
            &json(r#"{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":true}"#),
            false,
        )
        .unwrap();
        assert!(grammar.contains("additional-k ::="));
        assert!(grammar.contains("value ::="));
    }

    #[test]
    fn integer_bounds_and_pattern() {
        let grammar = json_schema_to_grammar(
            &json(r#"{"type":"integer","minimum":5,"maximum":9}"#),
            false,
        )
        .unwrap();
        assert!(grammar.contains("root ::="));

        let grammar = json_schema_to_grammar(
            &json(r#"{"allOf":[{"type":"integer","minimum":1},{"type":"integer","maximum":5}]}"#),
            false,
        )
        .unwrap();
        assert!(grammar.contains("root ::="));

        let grammar =
            json_schema_to_grammar(&json(r#"{"type":"string","pattern":"^a+$"}"#), false).unwrap();
        assert!(grammar.contains("root ::="));
    }
}

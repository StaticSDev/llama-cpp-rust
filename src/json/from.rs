//! Typed extraction from a JSON value.

use crate::core::Result;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::Hash;

use super::{json_err, Json, Value};

impl Json {
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


/// Explicit form of the implicit conversions into `Value`.
pub trait IntoJson {
    fn into_json(self) -> Value;
}

impl<T: Into<Value>> IntoJson for T {
    fn into_json(self) -> Value {
        self.into()
    }
}

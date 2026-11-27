//! Rust port of llama.cpp's built-in Jinja engine (`common/jinja`).
//!
//! The engine is split in the same logical pieces as the C++ source:
//! `lexer` tokenizes a template, `parser` builds a program (AST),
//! `runtime` executes it, `value` holds the primitives and builtins,
//! `string` carries input-provenance metadata and `caps` probes a template.
//!
//! No external crates, no unsafe and no FFI are used. Values are shared with
//! `Rc<RefCell<..>>`, which gives the same aliasing and mutation semantics as
//! the C++ `shared_ptr` values.

mod caps;
mod lexer;
mod parser;
mod runtime;
mod string;
mod utils;
mod value;

pub use caps::{caps_apply_preserve_reasoning, caps_apply_reasoning_effort, caps_get, Caps};
pub use lexer::{lexer_result, token, tokenize, LexerException, TokenType};
pub use parser::{parse_from_tokens, ParserException};
pub use runtime::{
    debug_dump_program, enable_debug, gather_string_parts, gather_string_parts_recursive,
    runtime as Runtime, Context, Program, Stmt, StmtData, StmtNode,
};
pub use string::{JString, StringPart};
pub use utils::{
    debug_enabled, enable_debug_flag, fmt_error_with_source, peak_source, string_replace_all,
    JinjaError, JResult, JINJA_RECURSIVE_LIMIT,
};
pub use value::{
    apply_filter_by_name, find_builtin_func, float_to_int, global_builtins, global_from_json,
    is_val, kind_of, mk_array, mk_bool, mk_empty_string, mk_float, mk_func, mk_int, mk_kwarg, mk_none, mk_object,
    mk_string, mk_tuple, mk_undefined, try_builtin_func, value_compare, value_from_json,
    value_to_json, value_to_string_repr, FuncArgs, FuncHandler, Value, ValueCompareOp, ValueData,
    ValueKind, ValueT,
};

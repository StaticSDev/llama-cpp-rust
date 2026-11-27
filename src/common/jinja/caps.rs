//! `jinja::caps`: probe a compiled template to discover what it supports.

use std::collections::BTreeMap;

use crate::json::Json;

use super::runtime::{runtime, Context, Program};
use super::value::{
    as_string, at, at_str, global_from_json, mk_bool, mk_int, mk_string, mk_undefined, Value,
};

#[derive(Clone, Debug)]
pub struct Caps {
    pub supports_tools: bool,
    pub supports_tool_calls: bool,
    pub supports_system_role: bool,
    pub supports_parallel_tool_calls: bool,
    pub supports_preserve_reasoning: bool,
    pub supports_reasoning_effort: bool,
    pub reasoning_effort_levels: Vec<String>,
    pub reasoning_effort_rejected: Vec<String>,
    pub supports_string_content: bool,
    pub supports_typed_content: bool,
    pub supports_object_arguments: bool,
}

impl Default for Caps {
    fn default() -> Self {
        Self {
            supports_tools: true,
            supports_tool_calls: true,
            supports_system_role: true,
            supports_parallel_tool_calls: true,
            supports_preserve_reasoning: false,
            supports_reasoning_effort: false,
            reasoning_effort_levels: Vec::new(),
            reasoning_effort_rejected: Vec::new(),
            supports_string_content: true,
            supports_typed_content: false,
            supports_object_arguments: false,
        }
    }
}

impl Caps {
    pub fn to_map(&self) -> BTreeMap<String, bool> {
        [
            ("supports_string_content", self.supports_string_content),
            ("supports_typed_content", self.supports_typed_content),
            ("supports_tools", self.supports_tools),
            ("supports_tool_calls", self.supports_tool_calls),
            ("supports_parallel_tool_calls", self.supports_parallel_tool_calls),
            ("supports_system_role", self.supports_system_role),
            ("supports_preserve_reasoning", self.supports_preserve_reasoning),
            ("supports_reasoning_effort", self.supports_reasoning_effort),
            ("supports_object_arguments", self.supports_object_arguments),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    pub fn to_string(&self) -> String {
        let mut ss = String::from("Caps(\n");
        for (k, v) in self.to_map() {
            ss.push_str(&format!("  {k}={}\n", if v { "true" } else { "false" }));
        }
        ss.push(')');
        ss
    }
}

pub fn caps_apply_preserve_reasoning(ctx: &Context, enabled: bool) {
    let _ = ctx.set_val("preserve_thinking", mk_bool(enabled));
    let _ = ctx.set_val("clear_thinking", mk_bool(!enabled));
    let _ = ctx.set_val("truncate_history_thinking", mk_bool(!enabled));
    let _ = ctx.set_val("drop_thinking", mk_bool(!enabled));
}

pub fn caps_apply_reasoning_effort(ctx: &Context, effort: &str) {
    let var = mk_string(effort);
    let _ = ctx.set_val("reasoning_effort", var.clone());
    let _ = ctx.set_val("reasoning_strength", var);
}

fn obj(pairs: Vec<(&str, Json)>) -> Json {
    Json::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn jstr(s: &str) -> Json {
    Json::String(s.to_string())
}

fn message(role: &str, content: Json) -> Json {
    obj(vec![("role", jstr(role)), ("content", content)])
}

fn caps_try_execute(
    prog: &Program,
    messages_fn: &dyn Fn() -> Json,
    ctx_fn: Option<&dyn Fn(&mut Context)>,
    tools_fn: Option<&dyn Fn() -> Json>,
    analyze_fn: &mut dyn FnMut(&mut Context, bool, Value, Value, &str),
) {
    let mut msgs = messages_fn();
    for attempt in 0..2 {
        let mut ctx = Context::new(String::new());
        ctx.is_get_stats = true;
        let tools = tools_fn.map(|f| f()).unwrap_or_else(Json::array);
        let _ = global_from_json(
            &mut ctx,
            &obj(vec![
                ("messages", msgs.clone()),
                ("tools", tools),
                ("bos_token", jstr("")),
                ("eos_token", jstr("")),
                ("add_generation_prompt", Json::Bool(true)),
            ]),
            true,
        );

        if let Some(f) = ctx_fn {
            f(&mut ctx);
        }
        let messages = ctx.get_val("messages");
        let tools_val = ctx.get_val("tools");

        let mut success = false;
        let mut result = String::new();
        let rt = runtime::new(&ctx);
        match rt.execute(prog) {
            Ok(results) => {
                let parts = super::runtime::gather_string_parts(&results);
                result = as_string(&parts).map(|s| s.str()).unwrap_or_default();
                success = true;
            }
            Err(e) => {
                crate::jj_debug!("Exception during execution: {}", e);
                result.clear();
            }
        }

        // Some templates need a reasoning_content field on every assistant
        // turn; retry once with an empty one on turns that lack it.
        if !success && attempt == 0 {
            let mut added = false;
            if let Json::Array(items) = &mut msgs {
                for msg in items.iter_mut() {
                    if let Json::Object(entries) = msg {
                        let is_assistant = entries
                            .iter()
                            .any(|(k, v)| k == "role" && *v == Json::String("assistant".into()));
                        let has_reasoning = entries.iter().any(|(k, _)| k == "reasoning_content");
                        if is_assistant && !has_reasoning {
                            entries.push(("reasoning_content".into(), jstr("")));
                            added = true;
                        }
                    }
                }
            }
            if added {
                continue;
            }
        }

        analyze_fn(&mut ctx, success, messages, tools_val, &result);
        return;
    }
}

fn has_op(v: &Value, op: &str) -> bool {
    v.borrow().stats.ops.contains(op)
}

fn caps_print_stats(v: &Value, path: &str) {
    let b = v.borrow();
    let ops: Vec<String> = b.stats.ops.iter().cloned().collect();
    crate::jj_debug!(
        "Value {}, type: {} {}, ops: {}",
        path,
        super::value::type_name(v),
        if b.stats.used { "(used)" } else { "" },
        ops.join(" ")
    );
}

fn nav(v: &Value, key: &str) -> Value {
    at_str(v, key, None)
        .ok()
        .flatten()
        .unwrap_or_else(|| mk_undefined(key))
}

fn nav_i(v: &Value, idx: usize) -> Value {
    at(v, &mk_int(idx as i64), None)
        .ok()
        .flatten()
        .unwrap_or_else(|| mk_undefined(""))
}

pub fn caps_get(prog: &Program) -> Caps {
    let mut result = Caps::default();
    let content_marker = "STRING_MARKER";

    crate::jj_debug!("{}", ">>> Running capability check: typed content");

    let mut checks_for_string = false;
    caps_try_execute(
        prog,
        &|| Json::Array(vec![message("user", jstr(content_marker))]),
        None,
        None,
        &mut |_, success, messages, _, rendered| {
            let content = nav(&nav_i(&messages, 0), "content");
            caps_print_stats(&content, "messages[0].content");
            if has_op(&content, "test_is_string") {
                checks_for_string = true;
            }
            let used_as_array = has_op(&content, "selectattr") || has_op(&content, "array_access");
            if used_as_array {
                result.supports_typed_content = true;
            }
            if !success {
                result.supports_string_content = false;
            } else if used_as_array && !rendered.contains(content_marker) {
                result.supports_string_content = false;
            }
        },
    );

    if checks_for_string {
        caps_try_execute(
            prog,
            &|| {
                Json::Array(vec![message(
                    "user",
                    Json::Array(vec![obj(vec![
                        ("type", jstr("text")),
                        ("text", jstr(content_marker)),
                    ])]),
                )])
            },
            None,
            None,
            &mut |_, success, messages, _, _| {
                let content = nav(&nav_i(&messages, 0), "content");
                caps_print_stats(&content, "messages[0].content");
                let used_as_array = has_op(&content, "selectattr") || has_op(&content, "array_access");
                if used_as_array && success {
                    result.supports_typed_content = true;
                }
            },
        );
    }

    crate::jj_debug!("{}", ">>> Running capability check: system prompt");
    caps_try_execute(
        prog,
        &|| {
            Json::Array(vec![
                message("system", jstr("System message")),
                message("user", jstr("User message")),
            ])
        },
        None,
        None,
        &mut |_, _, messages, _, _| {
            let content = nav(&nav_i(&messages, 0), "content");
            caps_print_stats(&content, "messages[0].content");
            if !content.borrow().stats.used {
                result.supports_system_role = false;
            }
        },
    );

    crate::jj_debug!("{}", ">>> Running capability check: single tool with object arguments support");
    let tool_def = || {
        obj(vec![
            ("name", jstr("tool")),
            ("type", jstr("function")),
            (
                "function",
                obj(vec![
                    ("name", jstr("tool1")),
                    ("description", jstr("Tool description")),
                    (
                        "parameters",
                        obj(vec![
                            ("type", jstr("object")),
                            (
                                "properties",
                                obj(vec![(
                                    "arg",
                                    obj(vec![
                                        ("type", jstr("string")),
                                        ("description", jstr("Arg description")),
                                    ]),
                                )]),
                            ),
                            ("required", Json::Array(vec![jstr("arg")])),
                        ]),
                    ),
                ]),
            ),
        ])
    };
    let tool_response_messages = |arguments: Json| {
        Json::Array(vec![
            message("user", jstr("User message")),
            obj(vec![
                ("role", jstr("assistant")),
                ("content", jstr("")),
                (
                    "tool_calls",
                    Json::Array(vec![obj(vec![
                        ("id", jstr("call00001")),
                        ("type", jstr("function")),
                        (
                            "function",
                            obj(vec![("name", jstr("tool1")), ("arguments", arguments)]),
                        ),
                    ])]),
                ),
            ]),
            obj(vec![
                ("role", jstr("tool")),
                ("content", jstr("Tool response")),
                ("tool_call_id", jstr("call00001")),
            ]),
            message("assistant", jstr("The tool response was 'tool response'")),
            message("user", jstr("User message")),
        ])
    };

    caps_try_execute(
        prog,
        &|| {
            tool_response_messages(obj(vec![("arg", jstr("value"))]))
        },
        None,
        Some(&tool_def),
        &mut |_, success, messages, tools, _| {
            if !success {
                return;
            }
            let tool_name = nav(&nav(&nav_i(&tools, 0), "function"), "name");
            caps_print_stats(&tool_name, "tools[0].function.name");
            caps_print_stats(&tools, "tools");
            if !tool_name.borrow().stats.used {
                result.supports_tools = false;
            }
            let tool_calls = nav(&nav_i(&messages, 1), "tool_calls");
            caps_print_stats(&tool_calls, "messages[1].tool_calls");
            if !tool_calls.borrow().stats.used {
                result.supports_tool_calls = false;
                return;
            }
            let tool_arg = nav(
                &nav(
                    &nav(&nav_i(&tool_calls, 0), "function"),
                    "arguments",
                ),
                "arg",
            );
            caps_print_stats(&tool_arg, "messages[1].tool_calls[0].function.arguments.arg");
            if tool_arg.borrow().stats.used {
                result.supports_object_arguments = true;
            }
        },
    );

    if !result.supports_object_arguments {
        crate::jj_debug!("{}", ">>> Running capability check: single tool with string arguments support");
        caps_try_execute(
            prog,
            &|| tool_response_messages(jstr(r#"{"arg": "value"}"#)),
            None,
            Some(&tool_def),
            &mut |_, success, messages, tools, _| {
                if !success {
                    result.supports_tool_calls = false;
                    result.supports_tools = false;
                    return;
                }
                let tool_name = nav(&nav(&nav_i(&tools, 0), "function"), "name");
                caps_print_stats(&tool_name, "tools[0].function.name");
                caps_print_stats(&tools, "tools");
                if !tool_name.borrow().stats.used {
                    result.supports_tools = false;
                }
                let tool_calls = nav(&nav_i(&messages, 1), "tool_calls");
                caps_print_stats(&tool_calls, "messages[1].tool_calls");
                if !tool_calls.borrow().stats.used {
                    result.supports_tool_calls = false;
                }
            },
        );
    }

    crate::jj_debug!("{}", ">>> Running capability check: parallel tool support");
    caps_try_execute(
        prog,
        &|| {
            let args = if result.supports_object_arguments {
                obj(vec![("arg", jstr("value"))])
            } else {
                Json::parse(r#"{"arg": "value"}"#).unwrap_or(Json::Null)
            };
            Json::Array(vec![
                message("user", jstr("User message")),
                obj(vec![
                    ("role", jstr("assistant")),
                    ("content", jstr("")),
                    (
                        "tool_calls",
                        Json::Array(vec![
                            obj(vec![
                                ("id", jstr("call00001")),
                                ("type", jstr("function")),
                                (
                                    "function",
                                    obj(vec![("name", jstr("tool1")), ("arguments", args.clone())]),
                                ),
                            ]),
                            obj(vec![
                                ("id", jstr("call00002")),
                                ("type", jstr("function")),
                                (
                                    "function",
                                    obj(vec![("name", jstr("tool1")), ("arguments", args)]),
                                ),
                            ]),
                        ]),
                    ),
                ]),
                obj(vec![
                    ("role", jstr("tool")),
                    ("content", jstr("Tool response")),
                    ("tool_call_id", jstr("call00001")),
                ]),
                message("assistant", jstr("The tool response was 'tool response'")),
                message("user", jstr("User message")),
            ])
        },
        None,
        Some(&tool_def),
        &mut |_, success, messages, _, _| {
            if !success {
                result.supports_parallel_tool_calls = false;
                return;
            }
            let tool_calls = nav(&nav_i(&messages, 1), "tool_calls");
            caps_print_stats(&tool_calls, "messages[1].tool_calls");
            let tool_call_1 = nav(&nav_i(&tool_calls, 1), "function");
            caps_print_stats(&tool_call_1, "messages[1].tool_calls[1].function");
            if !tool_call_1.borrow().stats.used {
                result.supports_parallel_tool_calls = false;
            }
        },
    );

    crate::jj_debug!("{}", ">>> Running capability check: preserve reasoning");
    let reasoning_placeholder = "<REASONING_CONTENT_PLACEHOLDER>";
    caps_try_execute(
        prog,
        &|| {
            Json::Array(vec![
                message("user", jstr("User message")),
                obj(vec![
                    ("role", jstr("assistant")),
                    ("content", jstr("Assistant message")),
                    ("reasoning_content", jstr(reasoning_placeholder)),
                ]),
                message("user", jstr("User message")),
                obj(vec![
                    ("role", jstr("assistant")),
                    ("content", jstr("Assistant message")),
                    ("reasoning_content", jstr("Reasoning content")),
                ]),
                message("user", jstr("User message")),
            ])
        },
        Some(&|ctx: &mut Context| {
            let _ = ctx.set_val("enable_thinking", mk_bool(true));
            caps_apply_preserve_reasoning(ctx, true);
        }),
        None,
        &mut |_, _, _, _, output| {
            if output.contains(reasoning_placeholder) {
                result.supports_preserve_reasoning = true;
            }
        },
    );

    crate::jj_debug!("{}", ">>> Running capability check: reasoning effort");
    caps_try_execute(
        prog,
        &|| Json::Array(vec![message("user", jstr("User message"))]),
        Some(&|ctx: &mut Context| {
            let _ = ctx.set_val("enable_thinking", mk_bool(true));
            caps_apply_reasoning_effort(ctx, "low");
        }),
        None,
        &mut |ctx, _, _, _, _| {
            let effort = ctx.get_val("reasoning_effort");
            caps_print_stats(&effort, "reasoning_effort");
            result.supports_reasoning_effort = effort.borrow().stats.used;
        },
    );

    if result.supports_reasoning_effort {
        crate::jj_debug!("{}", ">>> Running capability check: reasoning effort levels");
        let candidates = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
        let render = |effort: &str| -> Option<String> {
            let mut rendered = None;
            let effort_str = effort.to_string();
            caps_try_execute(
                prog,
                &|| Json::Array(vec![message("user", jstr("User message"))]),
                Some(&move |ctx: &mut Context| {
                    let _ = ctx.set_val("enable_thinking", mk_bool(true));
                    caps_apply_reasoning_effort(ctx, &effort_str);
                }),
                None,
                &mut |_, success, _, _, output| {
                    if success {
                        rendered = Some(output.to_string());
                    }
                },
            );
            rendered
        };

        let unknown = render("REASONING_EFFORT_UNKNOWN");
        let mut renders: Vec<String> = Vec::new();
        let mut rejected: Vec<String> = Vec::new();
        for level in candidates {
            let Some(rendered) = render(level) else {
                if level != "none" {
                    rejected.push(level.to_string());
                }
                continue;
            };
            if unknown.as_ref() == Some(&rendered) {
                continue;
            }
            if let Some(idx) = renders.iter().position(|r| *r == rendered) {
                result.reasoning_effort_levels[idx] = level.to_string();
            } else {
                renders.push(rendered);
                result.reasoning_effort_levels.push(level.to_string());
            }
        }
        if !result.reasoning_effort_levels.is_empty() {
            result.reasoning_effort_rejected = rejected;
        }
    }

    crate::jj_debug!("{}", result.to_string());
    result
}

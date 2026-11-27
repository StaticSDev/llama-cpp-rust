//! Chat messages, tools, roles, spans, delimiters and message diffs.

use crate::core::{Error, Result, Token};

pub(crate) fn string_diff(last: &str, current: &str) -> Result<String> {
    if last.is_empty() {
        return Ok(current.to_owned());
    }
    if !current.starts_with(last) {
        if last.starts_with(current) {
            // This happens if the last generation ended on a partial stop word (not erased),
            // and the current ended on a stop word (erased).
            return Ok(String::new());
        }
        return Err(Error::new(format!(
            "Invalid diff: '{}' not found at start of '{}'",
            last, current
        )));
    }
    Ok(current[last.len()..].to_owned())
}

pub fn has_content_or_tool_calls(msg: &ChatMsg) -> bool {
    !msg.content.is_empty() || !msg.tool_calls.is_empty()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into() }
    }
}


// ---------------------------------------------------------------------------
// Messages, tools and parsing helpers (common/chat.h / common/chat.cpp)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatToolCall {
    pub name: String,
    pub arguments: String,
    pub id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMsgContentPart {
    pub r#type: String,
    pub text: String,
    /// tool_reference: name of a tool from the request tools
    pub name: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
    pub content_parts: Vec<ChatMsgContentPart>,
    pub tool_calls: Vec<ChatToolCall>,
    pub reasoning_content: String,
    pub tool_name: String,
    pub tool_call_id: String,
    // non-OpenAI message keys as a JSON object string, only for templates that read them (DeepSeek V4.1)
    pub extra_json: String,
}

impl ChatMsg {
    pub fn render_content(&self, delimiter: &str) -> Result<String> {
        if !self.content.is_empty() && !self.content_parts.is_empty() {
            return Err(Error::new("Cannot specify both content and content_parts"));
        }
        if !self.content.is_empty() {
            return Ok(self.content.clone());
        }
        let mut text = String::new();
        for part in &self.content_parts {
            if part.r#type == "text" {
                if !text.is_empty() {
                    text.push_str(delimiter);
                }
                text.push_str(&part.text);
            }
        }
        Ok(text)
    }

    pub fn empty(&self) -> bool {
        self.content.is_empty()
            && self.content_parts.is_empty()
            && self.tool_calls.is_empty()
            && self.reasoning_content.is_empty()
            && self.tool_name.is_empty()
            && self.tool_call_id.is_empty()
    }

    pub fn contains_media(&self) -> bool {
        self.content_parts.iter().any(|part| part.r#type == "media_marker")
    }

    pub fn set_tool_call_ids<F: Fn() -> String>(
        &mut self,
        ids_cache: &mut Vec<String>,
        gen_tool_call_id: F,
    ) {
        for i in 0..self.tool_calls.len() {
            if ids_cache.len() <= i {
                let mut id = self.tool_calls[i].id.clone();
                if id.is_empty() {
                    id = gen_tool_call_id();
                }
                ids_cache.push(id);
            }
            self.tool_calls[i].id = ids_cache[i].clone();
        }
    }
}

pub const NO_TOOL_CALL_INDEX: usize = usize::MAX;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatMsgDiff {
    pub reasoning_content_delta: String,
    pub content_delta: String,
    pub tool_call_index: usize,
    pub tool_call_delta: ChatToolCall,
}

impl ChatMsgDiff {
    pub fn compute_diffs(msg_prv: &ChatMsg, msg_new: &ChatMsg) -> Result<Vec<Self>> {
        let mut diffs = Vec::new();

        // TODO: these can become expensive for long messages - how to optimize?
        if msg_prv.reasoning_content != msg_new.reasoning_content {
            diffs.push(Self {
                reasoning_content_delta: string_diff(&msg_prv.reasoning_content, &msg_new.reasoning_content)?,
                ..Default::default()
            });
        }
        if msg_prv.content != msg_new.content {
            diffs.push(Self {
                content_delta: string_diff(&msg_prv.content, &msg_new.content)?,
                ..Default::default()
            });
        }

        if msg_new.tool_calls.len() < msg_prv.tool_calls.len() {
            let mut err = String::from("Invalid diff: now finding less tool calls!\n");
            err.push_str(&format!(
                "  Previous ({}):\n",
                msg_prv.tool_calls.len()
            ));
            for tc in &msg_prv.tool_calls {
                err.push_str(&format!("    - name: '{}', args: '{}'\n", tc.name, tc.arguments));
            }
            err.push_str(&format!("  Current ({}):\n", msg_new.tool_calls.len()));
            for tc in &msg_new.tool_calls {
                err.push_str(&format!("    - name: '{}', args: '{}'\n", tc.name, tc.arguments));
            }
            err.push_str(&format!("  Current msg text content:\n{}\n", msg_new.content));
            return Err(Error::new(err));
        }

        if !msg_prv.tool_calls.is_empty() {
            let idx = msg_prv.tool_calls.len() - 1;
            let pref = &msg_prv.tool_calls[idx];
            let newf = &msg_new.tool_calls[idx];
            // Allow tool name to change during incremental parsing:
            // - empty -> non-empty (initial discovery)
            // - prefix -> longer string (name grows as more input is parsed)
            if pref.name != newf.name && !pref.name.is_empty() && !newf.name.is_empty() {
                // Check if one is a prefix of the other (for incremental parsing where names grow or shrink)
                if !newf.name.starts_with(&pref.name) {
                    return Err(Error::new("Invalid diff: tool call mismatch!"));
                }
            }
            let args_diff = string_diff(&pref.arguments, &newf.arguments)?;
            if !args_diff.is_empty() || pref.id != newf.id || pref.name != newf.name {
                let mut diff = Self {
                    tool_call_index: idx,
                    ..Default::default()
                };
                if pref.id != newf.id || pref.name != newf.name {
                    diff.tool_call_delta.id = newf.id.clone();
                    diff.tool_call_delta.name = newf.name.clone();
                }
                diff.tool_call_delta.arguments = args_diff;
                diffs.push(diff);
            }
        }
        for idx in msg_prv.tool_calls.len()..msg_new.tool_calls.len() {
            diffs.push(Self {
                tool_call_index: idx,
                tool_call_delta: msg_new.tool_calls[idx].clone(),
                ..Default::default()
            });
        }
        Ok(diffs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    Unknown,
    System,
    Assistant,
    User,
    Tool,
}

pub fn chat_role_from_string(role: &str) -> ChatRole {
    match role {
        "system" => ChatRole::System,
        "assistant" => ChatRole::Assistant,
        "user" => ChatRole::User,
        "tool" => ChatRole::Tool,
        _ => ChatRole::Unknown,
    }
}

pub fn chat_role_to_string(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::Assistant => "assistant",
        ChatRole::User => "user",
        ChatRole::Tool => "tool",
        ChatRole::Unknown => "",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatMsgSpan {
    pub role: ChatRole,
    pub pos: usize,
    pub len: usize,
}

impl ChatMsgSpan {
    pub fn valid(&self) -> bool {
        self.role != ChatRole::Unknown
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatMsgSpans {
    pub spans: Vec<ChatMsgSpan>,
}

impl ChatMsgSpans {
    pub fn add(&mut self, role: ChatRole, pos: usize, len: usize) {
        self.spans.push(ChatMsgSpan { role, pos, len });
    }

    pub fn is_user_start(&self, pos: i32) -> bool {
        self.spans
            .iter()
            .any(|span| span.role == ChatRole::User && pos == span.pos as i32)
    }

    pub fn last_user_message_pos(&self) -> i32 {
        for span in self.spans.iter().rev() {
            if span.role == ChatRole::User {
                return span.pos as i32;
            }
        }
        -1
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMsgDelimiter {
    pub role: ChatRole,
    pub delimiter: String,
    pub tokens: Vec<Token>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatMsgDelimiters {
    pub delimiters: Vec<ChatMsgDelimiter>,
}

impl ChatMsgDelimiters {
    pub fn add(&mut self, role: ChatRole, delimiter: &str) {
        self.delimiters.push(ChatMsgDelimiter {
            role,
            delimiter: delimiter.to_owned(),
            tokens: Vec::new(),
        });
    }

    // split tokens into message spans. skips maps a start index to a length of a
    // region to jump over without matching
    pub fn split(&self, tokens: &[Token], skips: &[(usize, usize)]) -> ChatMsgSpans {
        let mut skips = skips.to_vec();
        skips.sort_unstable_by_key(|&(start, _)| start);
        let mut skip_idx = 0;
        let mut matches: Vec<(ChatRole, usize)> = Vec::new();

        let mut i = 0;
        while i < tokens.len() {
            if skip_idx < skips.len() && i == skips[skip_idx].0 {
                i += skips[skip_idx].1;
                skip_idx += 1;
                continue;
            }
            for d in &self.delimiters {
                if i + d.tokens.len() > tokens.len() {
                    continue;
                }
                if tokens[i..i + d.tokens.len()] == d.tokens[..] {
                    matches.push((d.role, i));
                    break;
                }
            }
            i += 1;
        }
        matches.push((ChatRole::Unknown, tokens.len()));

        let mut spans = ChatMsgSpans::default();
        for pair in matches.windows(2) {
            spans.add(pair[0].0, pair[0].1, pair[1].1 - pair[0].1);
        }
        spans
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatTool {
    pub name: String,
    pub description: String,
    pub parameters: String,
    /// withheld from the system prompt, a tool_reference reveals it later in the conversation
    pub defer_loading: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatToolChoiceMode {
    Auto,
    Required,
    None,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatToolChoice {
    pub mode: ChatToolChoiceMode,
    /// Names of the tools the model may call, every tool when empty.
    pub allowed: Vec<String>,
}

impl Default for ChatToolChoiceMode {
    fn default() -> Self {
        Self::Auto
    }
}

impl ChatToolChoice {
    pub fn allows(&self, name: &str) -> bool {
        !self.is_none() && (self.allowed.is_empty() || self.allowed.iter().any(|n| n == name))
    }
    pub fn is_auto(&self) -> bool {
        self.mode == ChatToolChoiceMode::Auto
    }
    pub fn is_required(&self) -> bool {
        self.mode == ChatToolChoiceMode::Required
    }
    pub fn is_none(&self) -> bool {
        self.mode == ChatToolChoiceMode::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    ContentOnly,
    PegSimple,
    PegNative,
    PegGemma4,
    PegMinimaxM3,
}

impl Default for ChatFormat {
    fn default() -> Self {
        Self::ContentOnly
    }
}

pub fn chat_format_name(format: ChatFormat) -> &'static str {
    match format {
        ChatFormat::ContentOnly => "Content-only",
        ChatFormat::PegSimple => "peg-simple",
        ChatFormat::PegNative => "peg-native",
        ChatFormat::PegGemma4 => "peg-gemma4",
        ChatFormat::PegMinimaxM3 => "peg-minimax-m3",
    }
}

// Continuation method provided via `continue_final_message`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatContinuation {
    #[default]
    None,
    Auto,
    Reasoning,
    Content,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningFormat {
    #[default]
    None,
    Auto,
    Deepseek,
    DeepseekLegacy,
}

pub fn reasoning_format_name(format: ReasoningFormat) -> &'static str {
    match format {
        ReasoningFormat::None => "none",
        ReasoningFormat::Auto => "auto",
        ReasoningFormat::Deepseek => "deepseek",
        ReasoningFormat::DeepseekLegacy => "deepseek-legacy",
    }
}

pub fn reasoning_format_from_name(format: &str) -> Result<ReasoningFormat> {
    match format {
        "none" => Ok(ReasoningFormat::None),
        "auto" => Ok(ReasoningFormat::Auto),
        "deepseek" => Ok(ReasoningFormat::Deepseek),
        "deepseek-legacy" => Ok(ReasoningFormat::DeepseekLegacy),
        _ => Err(Error::new(format!("Unknown reasoning format: {}", format))),
    }
}


//! Chat formatting and message helpers, ported from `src/llama-chat.{h,cpp}`
//! and `common/chat.{h,cpp}`.
//!
//! The C++ template route that goes through the Jinja engine, the PEG parser,
//! JSON and the model vocabulary has no std-only equivalent, so only the
//! legacy (built-in template) route and the data-only helpers are ported here.

use std::fmt::Write;

use crate::core::{Error, LLAMA_TOKEN_NULL, Result, Token};

// ---------------------------------------------------------------------------
// Small string helpers (llama-chat.cpp, common/chat.cpp)
// ---------------------------------------------------------------------------

// trim whitespace from the beginning and end of a string
fn trim(str: &str) -> String {
    str.trim_matches(|c: char| c.is_ascii_whitespace()).to_owned()
}

pub fn string_lower(s: &str) -> String {
    s.chars().map(|c| c.to_ascii_lowercase()).collect()
}

fn upper_first(mut s: String) -> String {
    if let Some(c) = s.get_mut(0..1) {
        c.make_ascii_uppercase();
    }
    s
}

fn string_diff(last: &str, current: &str) -> Result<String> {
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

// ---------------------------------------------------------------------------
// Built-in chat templates (llama-chat.h / llama-chat.cpp)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    Chatml,
    Llama2,
    Llama2Sys,
    Llama2SysBos,
    Llama2SysStrip,
    MistralV1,
    MistralV3,
    MistralV3Tekken,
    MistralV7,
    MistralV7Tekken,
    Phi3,
    Phi4,
    Falcon3,
    Zephyr,
    Monarch,
    Gemma,
    Orion,
    Openchat,
    Vicuna,
    VicunaOrca,
    Deepseek,
    Deepseek2,
    Deepseek3,
    DeepseekOcr,
    CommandR,
    Llama3,
    Chatglm3,
    Chatglm4,
    GlmEdge,
    Minicpm,
    Exaone3,
    Exaone4,
    ExaoneMoe,
    RwkvWorld,
    Granite3X,
    Granite40,
    Granite41,
    Gigachat,
    Megrez,
    Yandex,
    Bailing,
    BailingThink,
    Bailing2,
    Llama4,
    Smolvlm,
    HunyuanMoe,
    OpenaiMoe,
    HunyuanDense,
    HunyuanVl,
    KimiK2,
    SeedOss,
    Grok2,
    PanguEmbed,
    SolarOpen,
    Unknown,
}

pub static LLM_CHAT_TEMPLATES: &[(&str, ChatTemplate)] = &[
    ("chatml", ChatTemplate::Chatml),
    ("llama2", ChatTemplate::Llama2),
    ("llama2-sys", ChatTemplate::Llama2Sys),
    ("llama2-sys-bos", ChatTemplate::Llama2SysBos),
    ("llama2-sys-strip", ChatTemplate::Llama2SysStrip),
    ("mistral-v1", ChatTemplate::MistralV1),
    ("mistral-v3", ChatTemplate::MistralV3),
    ("mistral-v3-tekken", ChatTemplate::MistralV3Tekken),
    ("mistral-v7", ChatTemplate::MistralV7),
    ("mistral-v7-tekken", ChatTemplate::MistralV7Tekken),
    ("phi3", ChatTemplate::Phi3),
    ("phi4", ChatTemplate::Phi4),
    ("falcon3", ChatTemplate::Falcon3),
    ("zephyr", ChatTemplate::Zephyr),
    ("monarch", ChatTemplate::Monarch),
    ("gemma", ChatTemplate::Gemma),
    ("orion", ChatTemplate::Orion),
    ("openchat", ChatTemplate::Openchat),
    ("vicuna", ChatTemplate::Vicuna),
    ("vicuna-orca", ChatTemplate::VicunaOrca),
    ("deepseek", ChatTemplate::Deepseek),
    ("deepseek2", ChatTemplate::Deepseek2),
    ("deepseek3", ChatTemplate::Deepseek3),
    ("deepseek-ocr", ChatTemplate::DeepseekOcr),
    ("command-r", ChatTemplate::CommandR),
    ("llama3", ChatTemplate::Llama3),
    ("chatglm3", ChatTemplate::Chatglm3),
    ("chatglm4", ChatTemplate::Chatglm4),
    ("glmedge", ChatTemplate::GlmEdge),
    ("minicpm", ChatTemplate::Minicpm),
    ("exaone3", ChatTemplate::Exaone3),
    ("exaone4", ChatTemplate::Exaone4),
    ("exaone-moe", ChatTemplate::ExaoneMoe),
    ("rwkv-world", ChatTemplate::RwkvWorld),
    ("granite", ChatTemplate::Granite3X),
    ("granite-4.0", ChatTemplate::Granite40),
    ("granite-4.1", ChatTemplate::Granite41),
    ("gigachat", ChatTemplate::Gigachat),
    ("megrez", ChatTemplate::Megrez),
    ("yandex", ChatTemplate::Yandex),
    ("bailing", ChatTemplate::Bailing),
    ("bailing-think", ChatTemplate::BailingThink),
    ("bailing2", ChatTemplate::Bailing2),
    ("llama4", ChatTemplate::Llama4),
    ("smolvlm", ChatTemplate::Smolvlm),
    ("hunyuan-moe", ChatTemplate::HunyuanMoe),
    ("gpt-oss", ChatTemplate::OpenaiMoe),
    ("hunyuan-dense", ChatTemplate::HunyuanDense),
    ("hunyuan-vl", ChatTemplate::HunyuanVl),
    ("kimi-k2", ChatTemplate::KimiK2),
    ("seed_oss", ChatTemplate::SeedOss),
    ("grok-2", ChatTemplate::Grok2),
    ("pangu-embedded", ChatTemplate::PanguEmbed),
    ("solar-open", ChatTemplate::SolarOpen),
];

pub fn chat_template_from_str(name: &str) -> Option<ChatTemplate> {
    LLM_CHAT_TEMPLATES
        .iter()
        .find(|&&(key, _)| key == name)
        .map(|(_, tmpl)| *tmpl)
}

pub fn chat_detect_template(tmpl: &str) -> ChatTemplate {
    if let Some(found) = chat_template_from_str(tmpl) {
        return found;
    }

    let contains = |haystack: &str| tmpl.contains(haystack);

    if contains("<|im_start|>") {
        if contains("<|im_sep|>") {
            ChatTemplate::Phi4
        } else if contains("<end_of_utterance>") {
            // SmolVLM uses <|im_start|> as BOS, but it is NOT chatml
            ChatTemplate::Smolvlm
        } else {
            ChatTemplate::Chatml
        }
    } else if tmpl.starts_with("mistral") || contains("[INST]") {
        if contains("[SYSTEM_PROMPT]") {
            ChatTemplate::MistralV7
        // catches official 'v1' template
        // catches official 'v3' and 'v3-tekken' templates
        } else if contains("' [INST] ' + system_message") || contains("[AVAILABLE_TOOLS]") {
            // Official mistral 'v1', 'v3' and 'v3-tekken' templates
            // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/chat_templates.md
            // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/templates.md
            if contains(" [INST]") {
                ChatTemplate::MistralV1
            } else if contains("\"[INST]\"") {
                ChatTemplate::MistralV3Tekken
            } else {
                ChatTemplate::MistralV3
            }
        } else {
            // llama2 template and its variants
            // [variant] support system message
            // See: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            let support_system_message = contains("<<SYS>>");
            let add_bos_inside_history = contains("bos_token + '[INST]");
            let strip_message = contains("content.strip()");
            if strip_message {
                ChatTemplate::Llama2SysStrip
            } else if add_bos_inside_history {
                ChatTemplate::Llama2SysBos
            } else if support_system_message {
                ChatTemplate::Llama2Sys
            } else {
                ChatTemplate::Llama2
            }
        }
    } else if contains("<|assistant|>") && contains("<|end|>") {
        ChatTemplate::Phi3
    } else if contains("[gMASK]<sop>") {
        ChatTemplate::Chatglm4
    } else if contains("<|assistant|>") && contains("<|user|>") {
        if contains("<|tool_declare|>") {
            ChatTemplate::ExaoneMoe
        } else if contains("</s>") {
            ChatTemplate::Falcon3
        } else {
            ChatTemplate::GlmEdge
        }
    } else if contains("<|{{ item['role'] }}|>") && contains("<|begin_of_image|>") {
        ChatTemplate::GlmEdge
    } else if contains("<|user|>") && contains("<|endoftext|>") {
        ChatTemplate::Zephyr
    } else if contains("bos_token + message['role']") {
        ChatTemplate::Monarch
    } else if contains("<start_of_turn>") {
        ChatTemplate::Gemma
    } else if contains(r"'\n\nAssistant: ' + eos_token") {
        // OrionStarAI/Orion-14B-Chat
        ChatTemplate::Orion
    } else if contains("GPT4 Correct ") {
        // openchat/openchat-3.5-0106
        ChatTemplate::Openchat
    } else if contains("USER: ") && contains("ASSISTANT: ") {
        // eachadea/vicuna-13b-1.1 (and Orca variant)
        if contains("SYSTEM: ") {
            ChatTemplate::VicunaOrca
        } else {
            ChatTemplate::Vicuna
        }
    } else if contains("### Instruction:") && contains("<|EOT|>") {
        // deepseek-ai/deepseek-coder-33b-instruct
        ChatTemplate::Deepseek
    } else if contains("<|START_OF_TURN_TOKEN|>") && contains("<|USER_TOKEN|>") {
        // CohereForAI/c4ai-command-r-plus
        ChatTemplate::CommandR
    } else if contains("<|start_header_id|>") && contains("<|end_header_id|>") {
        ChatTemplate::Llama3
    } else if contains("[gMASK]sop") {
        // chatglm3-6b
        ChatTemplate::Chatglm3
    } else if contains("<用户>") {
        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        ChatTemplate::Minicpm
    } else if contains("'Assistant: ' + message['content'] + eos_token") {
        ChatTemplate::Deepseek2
    } else if contains("<｜Assistant｜>")
        && contains("<｜User｜>")
        && contains("<｜end▁of▁sentence｜>")
    {
        ChatTemplate::Deepseek3
    } else if contains("[|system|]") && contains("[|assistant|]") && contains("[|endofturn|]") {
        if contains("[|tool|]") {
            ChatTemplate::Exaone4
        } else {
            // ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/discussions/8#66bae61b1893d14ee8ed85bb
            // EXAONE-3.0-7.8B-Instruct
            ChatTemplate::Exaone3
        }
    } else if contains("rwkv-world") || contains(r"{{- 'User: ' + message['content']|trim + '\n\n' -}}") {
        ChatTemplate::RwkvWorld
    } else if contains("<|start_of_role|>") {
        if contains("<tool_call>") || contains("<tools>") {
            if contains("g4_default_system_message") {
                ChatTemplate::Granite40
            } else {
                ChatTemplate::Granite41
            }
        } else {
            ChatTemplate::Granite3X
        }
    } else if contains("message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1]") {
        ChatTemplate::Gigachat
    } else if contains("<|role_start|>") {
        ChatTemplate::Megrez
    } else if contains(" Ассистент:") {
        ChatTemplate::Yandex
    } else if contains("<role>ASSISTANT</role>") && contains("'HUMAN'") {
        ChatTemplate::Bailing
    } else if contains("<role>ASSISTANT</role>") && contains("\"HUMAN\"") && contains("<think>") {
        ChatTemplate::BailingThink
    } else if contains("<role>ASSISTANT</role>")
        && contains("<role>HUMAN</role>")
        && contains("<|role_end|>")
    {
        ChatTemplate::Bailing2
    } else if contains("<|header_start|>") && contains("<|header_end|>") {
        ChatTemplate::Llama4
    } else if contains("<|endofuserprompt|>") {
        ChatTemplate::Dots1
    } else if contains("<|extra_0|>") && contains("<|extra_4|>") {
        ChatTemplate::HunyuanMoe
    } else if contains("<|start|>") && contains("<|channel|>") {
        ChatTemplate::OpenaiMoe
    } else if contains("<｜hy_Assistant｜>") && contains("<｜hy_begin▁of▁sentence｜>") {
        ChatTemplate::HunyuanVl
    } else if contains("<｜hy_Assistant｜>") && contains("<｜hy_placeholder▁no▁3｜>") {
        ChatTemplate::HunyuanDense
    } else if contains("<|im_assistant|>assistant<|im_middle|>") {
        ChatTemplate::KimiK2
    } else if contains("<seed:bos>") {
        ChatTemplate::SeedOss
    } else if contains("'Assistant: '  + message['content'] + '<|separator|>") {
        ChatTemplate::Grok2
    } else if contains("[unused9]系统：[unused10]") {
        ChatTemplate::PanguEmbed
    } else if contains("<|begin|>") && contains("<|end|>") && contains("<|content|>") {
        ChatTemplate::SolarOpen
    } else {
        ChatTemplate::Unknown
    }
}

// int32_t llama_chat_builtin_templates(const char ** output, size_t len)
pub fn chat_builtin_templates(output: &mut [&'static str]) -> i32 {
    let mut names: Vec<&'static str> = LLM_CHAT_TEMPLATES.iter().map(|(name, _)| *name).collect();
    names.sort_unstable(); // std::map is ordered by key
    for (slot, name) in output.iter_mut().zip(names.iter()) {
        *slot = *name;
    }
    names.len() as i32
}

// ---------------------------------------------------------------------------
// Legacy template formatting (llama-chat.cpp)
// ---------------------------------------------------------------------------

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

// Simple version of "llama_apply_chat_template" that only works with strings
// This function uses heuristic checks to determine commonly used template. It is not a jinja parser.
// Returns the byte length of dest, or -1 when the template is not supported.
pub fn apply_template(
    tmpl: ChatTemplate,
    chat: &[ChatMessage],
    dest: &mut String,
    add_ass: bool,
) -> i32 {
    // Taken from the research: https://github.com/ggml-org/llama.cpp/issues/5527
    let mut ss = String::new();

    if tmpl == ChatTemplate::Chatml {
        // chatml template
        for message in chat {
            write!(ss, "<|im_start|>{}\n{}<|im_end|>\n", message.role, message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<|im_start|>assistant\n");
        }
    } else if tmpl == ChatTemplate::MistralV7 || tmpl == ChatTemplate::MistralV7Tekken {
        // Official mistral 'v7' template
        // See: https://huggingface.co/mistralai/Mistral-Large-Instruct-2411#basic-instruct-template-v7
        //      https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503#basic-instruct-template-v7-tekken
        let trailing_space = if tmpl == ChatTemplate::MistralV7 { " " } else { "" };
        for message in chat {
            if message.role == "system" {
                write!(ss, "[SYSTEM_PROMPT]{}{}[/SYSTEM_PROMPT]", trailing_space, message.content).unwrap();
            } else if message.role == "user" {
                write!(ss, "[INST]{}{}[/INST]", trailing_space, message.content).unwrap();
            } else {
                write!(ss, "{}{}</s>", trailing_space, message.content).unwrap();
            }
        }
    } else if tmpl == ChatTemplate::MistralV1
        || tmpl == ChatTemplate::MistralV3
        || tmpl == ChatTemplate::MistralV3Tekken
    {
        // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/chat_templates.md
        // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/templates.md
        let leading_space = if tmpl == ChatTemplate::MistralV1 { " " } else { "" };
        let trailing_space = if tmpl == ChatTemplate::MistralV3Tekken { "" } else { " " };
        let trim_assistant_message = tmpl == ChatTemplate::MistralV3;
        let mut is_inside_turn = false;
        for message in chat {
            if !is_inside_turn {
                write!(ss, "{}[INST]{}", leading_space, trailing_space).unwrap();
                is_inside_turn = true;
            }
            if message.role == "system" {
                write!(ss, "{}\n\n", message.content).unwrap();
            } else if message.role == "user" {
                write!(ss, "{}{}[/INST]", message.content, leading_space).unwrap();
            } else {
                let content = if trim_assistant_message {
                    trim(&message.content)
                } else {
                    message.content.clone()
                };
                write!(ss, "{}{}</s>", trailing_space, content).unwrap();
                is_inside_turn = false;
            }
        }
    } else if tmpl == ChatTemplate::Llama2
        || tmpl == ChatTemplate::Llama2Sys
        || tmpl == ChatTemplate::Llama2SysBos
        || tmpl == ChatTemplate::Llama2SysStrip
    {
        // llama2 template and its variants
        // [variant] support system message
        // See: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        let support_system_message = tmpl != ChatTemplate::Llama2;
        // [variant] add BOS inside history
        let add_bos_inside_history = tmpl == ChatTemplate::Llama2SysBos;
        // [variant] trim spaces from the input message
        let strip_message = tmpl == ChatTemplate::Llama2SysStrip;
        // construct the prompt
        let mut is_inside_turn = true; // skip BOS at the beginning
        ss.push_str("[INST] ");
        for message in chat {
            let content = if strip_message { trim(&message.content) } else { message.content.clone() };
            if !is_inside_turn {
                is_inside_turn = true;
                ss.push_str(if add_bos_inside_history { "<s>[INST] " } else { "[INST] " });
            }
            if message.role == "system" {
                if support_system_message {
                    write!(ss, "<<SYS>>\n{}\n<</SYS>>\n\n", content).unwrap();
                } else {
                    // if the model does not support system message, we still include it in the first message, but without <<SYS>>
                    write!(ss, "{}\n", content).unwrap();
                }
            } else if message.role == "user" {
                write!(ss, "{} [/INST]", content).unwrap();
            } else {
                write!(ss, "{}</s>", content).unwrap();
                is_inside_turn = false;
            }
        }
    } else if tmpl == ChatTemplate::Phi3 {
        // Phi 3
        for message in chat {
            write!(ss, "<|{}|>\n{}<|end|>\n", message.role, message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<|assistant|>\n");
        }
    } else if tmpl == ChatTemplate::Phi4 {
        // chatml template
        for message in chat {
            write!(
                ss,
                "<|im_start|>{}<|im_sep|>{}<|im_end|>",
                message.role, message.content
            )
            .unwrap();
        }
        if add_ass {
            ss.push_str("<|im_start|>assistant<|im_sep|>");
        }
    } else if tmpl == ChatTemplate::Falcon3 {
        // Falcon 3
        for message in chat {
            write!(ss, "<|{}|>\n{}\n", message.role, message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<|assistant|>\n");
        }
    } else if tmpl == ChatTemplate::Zephyr {
        // zephyr template
        for message in chat {
            write!(ss, "<|{}|>\n{}<|endoftext|>\n", message.role, message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<|assistant|>\n");
        }
    } else if tmpl == ChatTemplate::Monarch {
        // mlabonne/AlphaMonarch-7B template (the <s> is included inside history)
        for (i, message) in chat.iter().enumerate() {
            let bos = if i == 0 { "" } else { "<s>" }; // skip BOS for first message
            write!(ss, "{}{}\n{}</s>\n", bos, message.role, message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<s>assistant\n");
        }
    } else if tmpl == ChatTemplate::Gemma {
        // google/gemma-7b-it
        let mut system_prompt = String::new();
        for message in chat {
            let mut role = message.role.clone();
            if role == "system" {
                // there is no system message for gemma, but we will merge it with user prompt, so nothing is broken
                system_prompt.push_str(&trim(&message.content));
                continue;
            }
            // in gemma, "assistant" is "model"
            if role == "assistant" {
                role = "model".to_owned();
            }
            write!(ss, "<start_of_turn>{}\n", role).unwrap();
            if !system_prompt.is_empty() && role != "model" {
                write!(ss, "{}\n\n", system_prompt).unwrap();
                system_prompt.clear();
            }
            write!(ss, "{}<end_of_turn>\n", trim(&message.content)).unwrap();
        }
        if add_ass {
            ss.push_str("<start_of_turn>model\n");
        }
    } else if tmpl == ChatTemplate::Orion {
        // OrionStarAI/Orion-14B-Chat
        let mut system_prompt = String::new();
        for message in chat {
            if message.role == "system" {
                // there is no system message support, we will merge it with user prompt
                system_prompt.push_str(&message.content);
                continue;
            } else if message.role == "user" {
                ss.push_str("Human: ");
                if !system_prompt.is_empty() {
                    write!(ss, "{}\n\n", system_prompt).unwrap();
                    system_prompt.clear();
                }
                write!(ss, "{}\n\nAssistant: </s>", message.content).unwrap();
            } else {
                write!(ss, "{}</s>", message.content).unwrap();
            }
        }
    } else if tmpl == ChatTemplate::Openchat {
        // openchat/openchat-3.5-0106,
        for message in chat {
            if message.role == "system" {
                write!(ss, "{}<|end_of_turn|>", message.content).unwrap();
            } else {
                let role = upper_first(message.role.clone());
                write!(ss, "GPT4 Correct {}: {}<|end_of_turn|>", role, message.content).unwrap();
            }
        }
        if add_ass {
            ss.push_str("GPT4 Correct Assistant:");
        }
    } else if tmpl == ChatTemplate::Vicuna || tmpl == ChatTemplate::VicunaOrca {
        // eachadea/vicuna-13b-1.1 (and Orca variant)
        for message in chat {
            if message.role == "system" {
                // Orca-Vicuna variant uses a system prefix
                if tmpl == ChatTemplate::VicunaOrca {
                    write!(ss, "SYSTEM: {}\n", message.content).unwrap();
                } else {
                    write!(ss, "{}\n\n", message.content).unwrap();
                }
            } else if message.role == "user" {
                write!(ss, "USER: {}\n", message.content).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "ASSISTANT: {}</s>\n", message.content).unwrap();
            }
        }
        if add_ass {
            ss.push_str("ASSISTANT:");
        }
    } else if tmpl == ChatTemplate::Deepseek {
        // deepseek-ai/deepseek-coder-33b-instruct
        for message in chat {
            if message.role == "system" {
                ss.push_str(&message.content);
            } else if message.role == "user" {
                write!(ss, "### Instruction:\n{}\n", message.content).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "### Response:\n{}\n<|EOT|>\n", message.content).unwrap();
            }
        }
        if add_ass {
            ss.push_str("### Response:\n");
        }
    } else if tmpl == ChatTemplate::CommandR {
        // CohereForAI/c4ai-command-r-plus
        for message in chat {
            if message.role == "system" {
                write!(ss, "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{}<|END_OF_TURN_TOKEN|>", trim(&message.content)).unwrap();
            } else if message.role == "user" {
                write!(ss, "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{}<|END_OF_TURN_TOKEN|>", trim(&message.content)).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{}<|END_OF_TURN_TOKEN|>", trim(&message.content)).unwrap();
            }
        }
        if add_ass {
            ss.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
        }
    } else if tmpl == ChatTemplate::Llama3 {
        // Llama 3
        for message in chat {
            write!(
                ss,
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                message.role,
                trim(&message.content)
            )
            .unwrap();
        }
        if add_ass {
            ss.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
    } else if tmpl == ChatTemplate::Chatglm3 {
        // chatglm3-6b
        ss.push_str("[gMASK]sop");
        for message in chat {
            write!(ss, "<|{}|>\n {}", message.role, message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<|assistant|>");
        }
    } else if tmpl == ChatTemplate::Chatglm4 {
        ss.push_str("[gMASK]<sop>");
        for message in chat {
            write!(ss, "<|{}|>\n{}", message.role, message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<|assistant|>\n");
        }
    } else if tmpl == ChatTemplate::GlmEdge {
        for message in chat {
            write!(ss, "<|{}|>\n{}", message.role, message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<|assistant|>");
        }
    } else if tmpl == ChatTemplate::Minicpm {
        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        for message in chat {
            if message.role == "user" {
                ss.push_str("<用户>");
                ss.push_str(&trim(&message.content));
                ss.push_str("<AI>");
            } else {
                ss.push_str(&trim(&message.content));
            }
        }
    } else if tmpl == ChatTemplate::Deepseek2 {
        // DeepSeek-V2
        for message in chat {
            if message.role == "system" {
                write!(ss, "{}\n\n", message.content).unwrap();
            } else if message.role == "user" {
                write!(ss, "User: {}\n\n", message.content).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "Assistant: {}<｜end▁of▁sentence｜>", message.content).unwrap();
            }
        }
        if add_ass {
            ss.push_str("Assistant:");
        }
    } else if tmpl == ChatTemplate::Deepseek3 {
        // DeepSeek-V3
        for message in chat {
            if message.role == "system" {
                write!(ss, "{}\n\n", message.content).unwrap();
            } else if message.role == "user" {
                write!(ss, "<｜User｜>{}", message.content).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "<｜Assistant｜>{}<｜end▁of▁sentence｜>", message.content).unwrap();
            }
        }
        if add_ass {
            ss.push_str("<｜Assistant｜>");
        }
    } else if tmpl == ChatTemplate::DeepseekOcr {
        for message in chat {
            // no template
            ss.push_str(&message.content);
        }
    } else if tmpl == ChatTemplate::Exaone3 {
        // ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/discussions/8#66bae61b1893d14ee8ed85bb
        // EXAONE-3.0-7.8B-Instruct
        for message in chat {
            if message.role == "system" {
                write!(ss, "[|system|]{}[|endofturn|]\n", trim(&message.content)).unwrap();
            } else if message.role == "user" {
                write!(ss, "[|user|]{}\n", trim(&message.content)).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "[|assistant|]{}[|endofturn|]\n", trim(&message.content)).unwrap();
            }
        }
        if add_ass {
            ss.push_str("[|assistant|]");
        }
    } else if tmpl == ChatTemplate::Exaone4 {
        for message in chat {
            if message.role == "system" {
                write!(ss, "[|system|]{}[|endofturn|]\n", trim(&message.content)).unwrap();
            } else if message.role == "user" {
                write!(ss, "[|user|]{}\n", trim(&message.content)).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "[|assistant|]{}[|endofturn|]\n", trim(&message.content)).unwrap();
            } else if message.role == "tool" {
                write!(ss, "[|tool|]{}[|endofturn|]\n", trim(&message.content)).unwrap();
            }
        }
        if add_ass {
            ss.push_str("[|assistant|]");
        }
    } else if tmpl == ChatTemplate::ExaoneMoe {
        for message in chat {
            if message.role == "system" {
                write!(ss, "<|system|>\n{}<|endofturn|>\n", trim(&message.content)).unwrap();
            } else if message.role == "user" {
                write!(ss, "<|user|>\n{}<|endofturn|>\n", trim(&message.content)).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "<|assistant|>\n{}<|endofturn|>\n", trim(&message.content)).unwrap();
            } else if message.role == "tool" {
                write!(ss, "<|tool|>\n{}<|endofturn|>\n", trim(&message.content)).unwrap();
            }
        }
        if add_ass {
            ss.push_str("<|assistant|>\n");
        }
    } else if tmpl == ChatTemplate::RwkvWorld {
        // this template requires the model to have "\n\n" as EOT token
        for (i, message) in chat.iter().enumerate() {
            if message.role == "system" {
                write!(ss, "System: {}\n\n", trim(&message.content)).unwrap();
            } else if message.role == "user" {
                write!(ss, "User: {}\n\n", trim(&message.content)).unwrap();
                if i == chat.len() - 1 {
                    ss.push_str("Assistant:");
                }
            } else if message.role == "assistant" {
                write!(ss, "Assistant: {}\n\n", trim(&message.content)).unwrap();
            }
        }
    } else if tmpl == ChatTemplate::Granite3X {
        // IBM Granite 3.x template
        for message in chat {
            write!(ss, "<|start_of_role|>{}<|end_of_role|>", message.role).unwrap();
            if message.role == "assistant_tool_call" {
                ss.push_str("<|tool_call|>");
            }
            write!(ss, "{}<|end_of_text|>\n", message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<|start_of_role|>assistant<|end_of_role|>");
        }
    } else if tmpl == ChatTemplate::Granite40 || tmpl == ChatTemplate::Granite41 {
        // IBM Granite 4.0 / 4.1 template
        for message in chat {
            if message.role == "assistant_tool_call" {
                ss.push_str("<|start_of_role|>assistant<|end_of_role|><|tool_call|>");
            } else {
                write!(ss, "<|start_of_role|>{}<|end_of_role|>", message.role).unwrap();
            }
            write!(ss, "{}<|end_of_text|>\n", message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<|start_of_role|>assistant<|end_of_role|>");
        }
    } else if tmpl == ChatTemplate::Gigachat {
        // GigaChat template
        let has_system = !chat.is_empty() && chat[0].role == "system";

        // Handle system message if present
        if has_system {
            write!(ss, "<s>{}<|message_sep|>", chat[0].content).unwrap();
        } else {
            ss.push_str("<s>");
        }

        // Process remaining messages
        let start = if has_system { 1 } else { 0 };
        for message in &chat[start..] {
            if message.role == "user" {
                write!(ss, "user<|role_sep|>{}<|message_sep|>available functions<|role_sep|>[]<|message_sep|>", message.content).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "assistant<|role_sep|>{}<|message_sep|>", message.content).unwrap();
            }
        }

        // Add generation prompt if needed
        if add_ass {
            ss.push_str("assistant<|role_sep|>");
        }
    } else if tmpl == ChatTemplate::Megrez {
        // Megrez template
        for message in chat {
            write!(ss, "<|role_start|>{}<|role_end|>{}<|turn_end|>", message.role, message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<|role_start|>assistant<|role_end|>");
        }
    } else if tmpl == ChatTemplate::Yandex {
        // Yandex template ("\n\n" is defined as EOT token)
        for message in chat {
            if message.role == "user" {
                write!(ss, " Пользователь: {}\n\n", message.content).unwrap();
            } else if message.role == "assistant" {
                write!(ss, " Ассистент: {}\n\n", message.content).unwrap();
            }
        }
        // Add generation prompt if needed
        if add_ass {
            ss.push_str(" Ассистент:[SEP]");
        }
    } else if tmpl == ChatTemplate::Bailing || tmpl == ChatTemplate::BailingThink {
        // Bailing (Ling/Ring) template
        for message in chat {
            let role = if message.role == "user" {
                "HUMAN".to_owned()
            } else {
                message.role.to_ascii_uppercase()
            };
            write!(ss, "<role>{}</role>{}", role, message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<role>ASSISTANT</role>");
            if tmpl == ChatTemplate::BailingThink {
                ss.push_str("<think>");
            }
        }
    } else if tmpl == ChatTemplate::Bailing2 {
        // Bailing2 (Ling 2.0) template
        let has_system = !chat.is_empty() && chat[0].role == "system";
        if !has_system {
            ss.push_str("<role>SYSTEM</role>detailed thinking off<|role_end|>");
        }
        for message in chat {
            let role = if message.role == "user" {
                "HUMAN".to_owned()
            } else {
                message.role.to_ascii_uppercase()
            };
            write!(ss, "<role>{}</role>{}<|role_end|>", role, message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<role>ASSISTANT</role>");
        }
    } else if tmpl == ChatTemplate::Llama4 {
        // Llama 4
        for message in chat {
            write!(
                ss,
                "<|header_start|>{}<|header_end|>\n\n{}<|eot|>",
                message.role,
                trim(&message.content)
            )
            .unwrap();
        }
        if add_ass {
            ss.push_str("<|header_start|>assistant<|header_end|>\n\n");
        }
    } else if tmpl == ChatTemplate::Smolvlm {
        // SmolVLM
        ss.push_str("<|im_start|>"); // uses <|im_start|> as BOS, but the actual content is NOT chatml
        for message in chat {
            if message.role == "system" {
                write!(ss, "{}\n\n", message.content).unwrap();
            } else if message.role == "user" {
                write!(ss, "User: {}<end_of_utterance>\n", message.content).unwrap();
            } else {
                write!(ss, "Assistant: {}<end_of_utterance>\n", message.content).unwrap();
            }
        }
        if add_ass {
            ss.push_str("Assistant:");
        }
    } else if tmpl == ChatTemplate::Dots1 {
        // dots.llm1.inst (DOTS1)
        for message in chat {
            if message.role == "system" {
                write!(ss, "<|system|>{}<|endofsystem|>", message.content).unwrap();
            } else if message.role == "user" {
                write!(ss, "<|userprompt|>{}<|endofuserprompt|>", message.content).unwrap();
            } else {
                write!(ss, "<|response|>{}<|endofresponse|>", message.content).unwrap();
            }
        }
        if add_ass {
            ss.push_str("<|response|>");
        }
    } else if tmpl == ChatTemplate::HunyuanMoe {
        // tencent/Hunyuan-A13B-Instruct
        for message in chat {
            if message.role == "system" {
                write!(ss, "<|startoftext|>{}<|extra_4|>", message.content).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "{}<|eos|>", message.content).unwrap();
            } else {
                write!(ss, "<|startoftext|>{}<|extra_0|>", message.content).unwrap();
            }
        }
    } else if tmpl == ChatTemplate::OpenaiMoe {
        // OpenAI MoE (based on Harmony chat template)
        for message in chat {
            write!(ss, "<|start|>{}<|message|>{}", message.role, message.content).unwrap();
            ss.push_str(if message.role == "assistant" { "<|return|>" } else { "<|end|>" });
        }
        if add_ass {
            ss.push_str("<|start|>assistant");
        }
    } else if tmpl == ChatTemplate::HunyuanDense {
        // tencent/Hunyuan-4B-Instruct
        for (i, message) in chat.iter().enumerate() {
            if i == 0 && message.role == "system" {
                write!(ss, "{}<｜hy_placeholder▁no▁3｜>", message.content).unwrap();
            }
            if message.role == "assistant" {
                write!(ss, "<｜hy_Assistant｜>{}<｜hy_placeholder▁no▁2｜>", message.content).unwrap();
            } else if message.role == "user" {
                write!(ss, "<｜hy_User｜>{}<｜hy_Assistant｜>", message.content).unwrap();
            }
        }
    } else if tmpl == ChatTemplate::HunyuanVl {
        // tencent/HunyuanOCR & tencent/HunyuanVL
        ss.push_str("<｜hy_begin▁of▁sentence｜>");
        for (i, message) in chat.iter().enumerate() {
            if i == 0 && message.role == "system" {
                write!(ss, "{}<｜hy_placeholder▁no▁3｜>", message.content).unwrap();
                continue;
            }
            if message.role == "user" {
                write!(ss, "{}<｜hy_User｜>", message.content).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "{}<｜hy_Assistant｜>", message.content).unwrap();
            }
        }
    } else if tmpl == ChatTemplate::KimiK2 {
        // moonshotai/Kimi-K2-Instruct
        for message in chat {
            if message.role == "system" {
                ss.push_str("<|im_system|>system<|im_middle|>");
            } else if message.role == "user" {
                ss.push_str("<|im_user|>user<|im_middle|>");
            } else if message.role == "assistant" {
                ss.push_str("<|im_assistant|>assistant<|im_middle|>");
            } else if message.role == "tool" {
                ss.push_str("<|im_system|>tool<|im_middle|>");
            }
            write!(ss, "{}<|im_end|>", message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<|im_assistant|>assistant<|im_middle|>");
        }
    } else if tmpl == ChatTemplate::SeedOss {
        for message in chat {
            let content = if message.role == "assistant" { trim(&message.content) } else { message.content.clone() };
            write!(ss, "<seed:bos>{}\n{}<seed:eos>", message.role, content).unwrap();
        }
        if add_ass {
            ss.push_str("<seed:bos>assistant\n");
        }
    } else if tmpl == ChatTemplate::Grok2 {
        for message in chat {
            if message.role == "system" {
                write!(ss, "System: {}<|separator|>\n\n", trim(&message.content)).unwrap();
            } else if message.role == "user" {
                write!(ss, "Human: {}<|separator|>\n\n", trim(&message.content)).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "Assistant: {}<|separator|>\n\n", message.content).unwrap();
            }
        }
        if add_ass {
            ss.push_str("Assistant:");
        }
    } else if tmpl == ChatTemplate::PanguEmbed {
        // [unused9]系统：xxx[unused10]
        // [unused9]用户：xxx[unused10]
        // [unused9]助手：xxx[unused10]
        // ...
        for (i, message) in chat.iter().enumerate() {
            if i == 0 && message.role != "system" {
                ss.push_str("[unused9]系统：[unused10]");
            }
            if message.role == "system" {
                write!(ss, "[unused9]系统：{}[unused10]", message.content).unwrap();
            } else if message.role == "user" {
                write!(ss, "[unused9]用户：{}[unused10]", message.content).unwrap();
            } else if message.role == "assistant" {
                write!(ss, "[unused9]助手：{}[unused10]", message.content).unwrap();
            } else if message.role == "tool" {
                write!(ss, "[unused9]工具：{}[unused10]", message.content).unwrap();
            } else if message.role == "function" {
                write!(ss, "[unused9]方法：{}[unused10]", message.content).unwrap();
            }
        }
        if add_ass {
            ss.push_str("[unused9]助手：");
        }
    } else if tmpl == ChatTemplate::SolarOpen {
        for message in chat {
            write!(ss, "<|begin|>{}<|content|>{}<|end|>", message.role, message.content).unwrap();
        }
        if add_ass {
            ss.push_str("<|begin|>assistant");
        }
    } else {
        // template not supported
        return -1;
    }

    *dest = ss;
    dest.len() as i32
}

// Public C entry point takes the template source and detects it first.
pub fn chat_apply_template(tmpl_src: &str, chat: &[ChatMessage], dest: &mut String, add_ass: bool) -> i32 {
    apply_template(chat_detect_template(tmpl_src), chat, dest, add_ass)
}

// ---------------------------------------------------------------------------
// Messages, tools and parsing helpers (common/chat.h / common/chat.cpp)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, Default)]
pub struct ChatTemplatesInputs {
    pub messages: Vec<ChatMsg>,
    pub grammar: String,
    pub json_schema: String,
    pub add_generation_prompt: bool,
    pub continue_final_message: ChatContinuation,
    pub use_jinja: bool,
    // Parameters below only supported when use_jinja is true
    pub tools: Vec<ChatTool>,
    pub tool_choice: ChatToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning_format: ReasoningFormat,
    // TODO: refactor this to "bool enable_thinking"
    pub enable_thinking: bool,
    pub chat_template_kwargs: Vec<(String, String)>,
    pub add_bos: bool,
    pub add_eos: bool,
    pub force_pure_content: bool,
}

impl ChatTemplatesInputs {
    pub fn new() -> Self {
        Self {
            add_generation_prompt: true,
            use_jinja: true,
            enable_thinking: true,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatPromptPart {
    pub text: String,
    /// from user input, special tokens inside must not be parsed
    pub is_input: bool,
}

/// rendered chat template, split into template parts and user input parts
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatPrompt {
    pub parts: Vec<ChatPromptPart>,
}

impl ChatPrompt {
    pub fn new() -> Self {
        Self::default()
    }

    // single part of template text
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            parts: vec![ChatPromptPart {
                text: text.into(),
                is_input: false,
            }],
        }
    }

    pub fn str(&self) -> String {
        if self.parts.len() == 1 {
            return self.parts[0].text.clone();
        }
        let mut out = String::with_capacity(self.parts.iter().map(|p| p.text.len()).sum());
        for part in &self.parts {
            out.push_str(&part.text);
        }
        out
    }

    pub fn append(&mut self, text: &str, is_input: bool) {
        if text.is_empty() {
            return;
        }
        if let Some(last) = self.parts.last_mut() {
            if last.is_input == is_input {
                last.text.push_str(text);
                return;
            }
        }
        self.parts.push(ChatPromptPart {
            text: text.to_owned(),
            is_input,
        });
    }

    // replace the last occurrence of `from`, user input parts are skipped
    pub fn replace_last(&mut self, from: &str, to: &str) {
        for part in self.parts.iter_mut().rev() {
            if part.is_input {
                continue;
            }
            if let Some(pos) = part.text.rfind(from) {
                part.text.replace_range(pos..pos + from.len(), to);
                return;
            }
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatParams {
    pub format: ChatFormat,
    pub prompt: ChatPrompt,
    pub grammar: String,
    pub grammar_lazy: bool,
    pub generation_prompt: String,
    pub supports_thinking: bool,
    pub thinking_start_tag: String, // e.g., "<think>"
    pub thinking_end_tags: Vec<String>, // e.g., "</think>"
    pub preserved_tokens: Vec<Token>,
    pub additional_stops: Vec<String>,
    pub message_delimiters: ChatMsgDelimiters,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatInput {
    pub text: String,
    pub tokens: Vec<Token>,
}

impl Default for ChatInput {
    fn default() -> Self {
        Self {
            text: String::new(),
            tokens: Vec::new(),
        }
    }
}

impl ChatInput {
    // plain text, with no tokens
    pub fn from_text(text: impl Into<String>) -> Self {
        let text = text.into();
        let tokens = vec![LLAMA_TOKEN_NULL; text.len()];
        Self { text, tokens }
    }

    pub fn size(&self) -> usize {
        self.text.len()
    }

    pub fn empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn append(&mut self, piece: &str, token: Token) {
        if piece.is_empty() {
            return;
        }
        self.tokens.push(token);
        self.tokens.resize(self.tokens.len() + piece.len() - 1, LLAMA_TOKEN_NULL);
        self.text.push_str(piece);
    }

    pub fn append_chunk(&mut self, chunk: &ChatInput) {
        self.tokens.extend_from_slice(&chunk.tokens);
        self.text.push_str(&chunk.text);
    }

    pub fn prepend(&mut self, prefix: &str) {
        self.tokens
            .splice(0..0, std::iter::repeat(LLAMA_TOKEN_NULL).take(prefix.len()));
        self.text.insert_str(0, prefix);
    }

    pub fn prepend_input(&mut self, prefix: &ChatInput) {
        self.tokens.splice(0..0, prefix.tokens.iter().copied());
        self.text.insert_str(0, &prefix.text);
    }

    pub fn truncate(&mut self, pos: usize) {
        if pos < self.text.len() {
            self.text = String::from_utf8_lossy(&self.text.as_bytes()[..pos]).into_owned();
            self.tokens.truncate(pos);
        }
    }

    pub fn substr(&self, pos: usize, n: usize) -> Self {
        if pos >= self.text.len() || pos > self.tokens.len() {
            return Self::default();
        }
        let len = (self.text.len() - pos).min(n).min(self.tokens.len() - pos);
        Self {
            text: String::from_utf8_lossy(&self.text.as_bytes()[pos..pos + len]).into_owned(),
            tokens: self.tokens[pos..pos + len].to_vec(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatParserParams {
    pub format: ChatFormat,
    pub generation_prompt: ChatInput,
    /// Enable debug output for the PEG parser
    pub debug: bool,
}

impl ChatParserParams {
    pub fn from_chat_params(params: &ChatParams) -> Self {
        Self {
            format: params.format,
            generation_prompt: ChatInput::from_text(params.generation_prompt.clone()),
            debug: false,
        }
    }
}

// specialized per-task preset
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatPromptPreset {
    pub system: String,
    pub user: String,
    /// the preset already names the requested language
    pub has_language: bool,
    /// the user text already holds the context
    pub has_context: bool,
    /// the media goes before the user text
    pub media_first: bool,
    /// the user text depends on the audio length, call again with it
    pub needs_audio_length: bool,
}

// ---------------------------------------------------------------------------
// Legacy-route helpers that need no model, vocab, JSON or PEG parser
// ---------------------------------------------------------------------------

fn legacy_chat_message(msg: &ChatMsg) -> ChatMessage {
    let mut content = msg.content.clone();
    for part in &msg.content_parts {
        if part.r#type != "text" && part.r#type != "media_marker" {
            continue;
        }
        if !content.is_empty() {
            content.push('\n');
        }
        content.push_str(&part.text);
    }
    ChatMessage {
        role: msg.role.clone(),
        content,
    }
}

fn legacy_chat_messages(msgs: &[ChatMsg]) -> Vec<ChatMessage> {
    msgs.iter().map(legacy_chat_message).collect()
}

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
pub fn chat_verify_template(tmpl: &str) -> bool {
    let chat = [ChatMessage::new("user", "test")];
    let mut dest = String::new();
    chat_apply_template(tmpl, &chat, &mut dest, true) >= 0
}

// Format single message, while taking into account the position of that message in chat history
pub fn chat_format_single(
    tmpl_src: &str,
    past_msg: &[ChatMsg],
    new_msg: &ChatMsg,
    add_ass: bool,
) -> Result<String> {
    let past = legacy_chat_messages(past_msg);
    let mut fmt_past_msg = String::new();
    if !past.is_empty() {
        chat_apply_template(tmpl_src, &past, &mut fmt_past_msg, false);
    }
    let mut ss = String::new();
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if add_ass && !fmt_past_msg.is_empty() && fmt_past_msg.ends_with('\n') {
        ss.push('\n');
    }
    // format chat with new_msg
    let mut all = past;
    all.push(legacy_chat_message(new_msg));
    let mut fmt_new_msg = String::new();
    chat_apply_template(tmpl_src, &all, &mut fmt_new_msg, add_ass);
    // get the diff part
    let tail = fmt_new_msg.get(fmt_past_msg.len()..).ok_or_else(|| {
        Error::new("Formatted history is longer than the formatted chat with the new message")
    })?;
    ss.push_str(tail);
    Ok(ss)
}

// Returns an example of formatted chat
pub fn chat_format_example(tmpl_src: &str) -> Result<String> {
    let msgs = [
        ChatMessage::new("system", "You are a helpful assistant"),
        ChatMessage::new("user", "Hello"),
        ChatMessage::new("assistant", "Hi there"),
        ChatMessage::new("user", "How are you?"),
    ];
    let mut dest = String::new();
    if chat_apply_template(tmpl_src, &msgs, &mut dest, true) < 0 {
        return Err(Error::new("this custom template is not supported, try using --jinja"));
    }
    Ok(dest)
}

// bos/eos come from the model, not from the user
pub fn strip_bos_eos(
    prompt: &mut ChatPrompt,
    bos_token: &str,
    eos_token: &str,
    add_bos: bool,
    add_eos: bool,
) {
    if add_bos && !prompt.parts.is_empty() {
        let first = &mut prompt.parts[0];
        if !first.is_input && first.text.starts_with(bos_token) {
            first.text.drain(..bos_token.len());
        }
    }
    if add_eos && !prompt.parts.is_empty() {
        let last = &mut prompt.parts[prompt.parts.len() - 1];
        if !last.is_input && last.text.ends_with(eos_token) {
            last.text.truncate(last.text.len() - eos_token.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_and_detect_template() {
        assert_eq!(trim("  hi\t\n"), "hi");
        assert_eq!(chat_detect_template("<|im_start|>user"), ChatTemplate::Chatml);
        assert_eq!(chat_detect_template("[gMASK]<sop>"), ChatTemplate::Chatglm4);
        assert_eq!(chat_detect_template("nonsense"), ChatTemplate::Unknown);
    }

    #[test]
    fn test_apply_template_chatml() {
        let chat = [ChatMessage::new("user", "Hello")];
        let mut dest = String::new();
        assert_eq!(
            apply_template(ChatTemplate::Chatml, &chat, &mut dest, true),
            55
        );
        assert_eq!(dest, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(string_diff("abc", "abcdef").unwrap(), "def");
    }
}

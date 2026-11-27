//! Legacy template formatting, prompt/input helpers and formatting entry points.

use std::fmt::Write;

use crate::chat::detect::chat_detect_template;
use crate::chat::message::{
    ChatContinuation, ChatFormat, ChatMessage, ChatMsg, ChatMsgDelimiters, ChatTool,
    ChatToolChoice, ReasoningFormat,
};
use crate::chat::template::ChatTemplate;
use crate::core::{Error, LLAMA_TOKEN_NULL, Result, Token};

// ---------------------------------------------------------------------------
// Small string helpers (llama-chat.cpp, common/chat.cpp)
// ---------------------------------------------------------------------------

// trim whitespace from the beginning and end of a string
pub(crate) fn trim(str: &str) -> String {
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

// ---------------------------------------------------------------------------
// Legacy template formatting (llama-chat.cpp)
// ---------------------------------------------------------------------------

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
        let last_idx = prompt.parts.len() - 1;
        let last = &mut prompt.parts[last_idx];
        if !last.is_input && last.text.ends_with(eos_token) {
            last.text.truncate(last.text.len() - eos_token.len());
        }
    }
}

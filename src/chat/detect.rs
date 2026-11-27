//! Heuristic chat-template detection from template source text.

use crate::chat::template::{chat_template_from_str, ChatTemplate};


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

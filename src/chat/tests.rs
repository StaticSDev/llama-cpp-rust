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

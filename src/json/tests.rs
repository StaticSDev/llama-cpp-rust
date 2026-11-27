use super::*;

    #[test]
    fn parse_dump_round_trip() {
        let text = r#"{"name":"llama","n":7,"ok":true,"xs":[1,2.5,null]}"#;
        let json = Json::parse(text).unwrap();
        assert!(json.is_object());
        assert_eq!(json.at("n").unwrap().get::<i64>().unwrap(), 7);
        assert_eq!(json.at("xs").unwrap().size(), 3);
        assert_eq!(json.dump(-1).unwrap(), text);
        assert_eq!(Json::parse(&json.dump(-1).unwrap()).unwrap(), json);
    }

    #[test]
    fn escaping_pretty_dump_and_discarded() {
        let json = Json::object_from(vec![
            Item::new("s", "a\n\"b"),
            Item::new("u", "\u{00e9}"),
        ]);
        let compact = json.dump(-1).unwrap();
        assert_eq!(compact, r#"{"s":"a\n\"b","u":"é"}"#);
        let pretty = json.dump(2).unwrap();
        assert!(pretty.contains("\n  \"s\": \"a\\n\\\"b\""));
        assert_eq!(Json::parse(&compact).unwrap(), json);
        assert!(Json::parse_no_throw("{oops").is_discarded());
    }

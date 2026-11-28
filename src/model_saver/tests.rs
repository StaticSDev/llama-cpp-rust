use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{save_model, SavedTensor, SavedValue};
use crate::gguf::{
    gguf_find_key, gguf_get_alignment, gguf_get_arr_data, gguf_get_arr_n, gguf_get_arr_str,
    gguf_get_arr_type, gguf_get_kv_type, gguf_get_n_kv, gguf_get_n_tensors, gguf_get_tensor_name,
    gguf_get_tensor_ne, gguf_get_tensor_type, gguf_get_val_bool, gguf_get_val_f32,
    gguf_get_val_i32, gguf_get_val_i64, gguf_get_val_str, gguf_get_val_u32, gguf_get_val_u64,
    gguf_init_from_file, GgufContext, GgmlContext, GgmlType, GgufInitParams, GgufType,
    GGML_MAX_DIMS,
};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TempFile(PathBuf);

impl TempFile {
    fn new(tag: &str) -> Self {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!(
            "llama_rs_model_saver_{}_{}_{}.gguf",
            std::process::id(),
            tag,
            id
        )))
    }

    fn path(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn read_model(path: &str) -> (GgufContext, GgmlContext) {
    let mut params = GgufInitParams {
        ctx: Some(GgmlContext::default()),
        ..Default::default()
    };
    let ctx = gguf_init_from_file(path, &mut params).expect("gguf_init_from_file failed");
    (ctx, params.ctx.expect("tensor data not loaded"))
}

#[test]
fn round_trip_tensors_and_metadata() {
    let tmp = TempFile::new("round_trip");
    let first = SavedTensor {
        name: "tok_embd.weight".to_owned(),
        shape: vec![2, 2],
        ty: GgmlType::F32,
        data: [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect(),
    };
    let second = SavedTensor {
        name: "output.weight".to_owned(),
        shape: vec![3],
        ty: GgmlType::F16,
        data: vec![0x00, 0x3c, 0x00, 0x40, 0x00, 0x42],
    };
    let metadata = vec![
        (
            "general.architecture".to_owned(),
            SavedValue::String("llama".to_owned()),
        ),
        ("llama.context_length".to_owned(), SavedValue::U32(4096)),
        ("llama.block_count".to_owned(), SavedValue::I32(2)),
        ("llama.file_type".to_owned(), SavedValue::U64(7)),
        ("llama.seed".to_owned(), SavedValue::I64(-9)),
        ("llama.freq_base".to_owned(), SavedValue::F32(10000.0)),
        ("llama.causal".to_owned(), SavedValue::Bool(true)),
    ];

    save_model(&tmp.path(), &[first.clone(), second.clone()], &metadata)
        .expect("save_model failed");
    let (ctx, ggml) = read_model(&tmp.path());

    assert_eq!(gguf_get_n_tensors(&ctx), 2);
    assert_eq!(gguf_get_n_kv(&ctx), metadata.len() as i64);
    for (i, expected) in [&first, &second].iter().enumerate() {
        let id = i as i64;
        assert_eq!(gguf_get_tensor_name(&ctx, id), expected.name);
        assert_eq!(gguf_get_tensor_type(&ctx, id), expected.ty);
        let ne = gguf_get_tensor_ne(&ctx, id);
        for (d, &dim) in expected.shape.iter().enumerate() {
            assert_eq!(ne[d], dim as i64);
        }
        for d in expected.shape.len()..GGML_MAX_DIMS {
            assert_eq!(ne[d], 1);
        }
        assert_eq!(ggml.tensors[i].data, expected.data);
    }

    assert_eq!(
        gguf_get_val_str(&ctx, gguf_find_key(&ctx, "general.architecture")),
        "llama"
    );
    assert_eq!(
        gguf_get_val_u32(&ctx, gguf_find_key(&ctx, "llama.context_length")),
        4096
    );
    assert_eq!(
        gguf_get_val_i32(&ctx, gguf_find_key(&ctx, "llama.block_count")),
        2
    );
    assert_eq!(
        gguf_get_val_i64(&ctx, gguf_find_key(&ctx, "llama.seed")),
        -9
    );
    assert_eq!(
        gguf_get_val_f32(&ctx, gguf_find_key(&ctx, "llama.freq_base")),
        10000.0
    );
    assert!(gguf_get_val_bool(&ctx, gguf_find_key(&ctx, "llama.causal")));
}

#[test]
fn array_metadata_round_trip() {
    let tmp = TempFile::new("arrays");
    let metadata = vec![
        (
            "tokenizer.ggml.tokens".to_owned(),
            SavedValue::StringArray(vec!["<s>".to_owned(), "</s>".to_owned()]),
        ),
        (
            "tokenizer.ggml.token_type".to_owned(),
            SavedValue::I32Array(vec![1, 2]),
        ),
        (
            "tokenizer.ggml.scores".to_owned(),
            SavedValue::F32Array(vec![0.5, -1.5]),
        ),
    ];

    save_model(&tmp.path(), &[], &metadata).expect("save_model failed");
    let (ctx, _) = read_model(&tmp.path());

    let id = gguf_find_key(&ctx, "tokenizer.ggml.tokens");
    assert_eq!(gguf_get_kv_type(&ctx, id), GgufType::Array);
    assert_eq!(gguf_get_arr_type(&ctx, id), GgufType::String);
    assert_eq!(gguf_get_arr_n(&ctx, id), 2);
    assert_eq!(gguf_get_arr_str(&ctx, id, 0), "<s>");
    assert_eq!(gguf_get_arr_str(&ctx, id, 1), "</s>");

    let id = gguf_find_key(&ctx, "tokenizer.ggml.token_type");
    assert_eq!(gguf_get_arr_type(&ctx, id), GgufType::Int32);
    let expected: Vec<u8> = [1i32, 2].iter().flat_map(|v| v.to_ne_bytes()).collect();
    assert_eq!(gguf_get_arr_data(&ctx, id), expected.as_slice());

    let id = gguf_find_key(&ctx, "tokenizer.ggml.scores");
    assert_eq!(gguf_get_arr_type(&ctx, id), GgufType::Float32);
    let expected: Vec<u8> = [0.5f32, -1.5]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    assert_eq!(gguf_get_arr_data(&ctx, id), expected.as_slice());
}

#[test]
fn invalid_tensors_return_errors() {
    let tmp = TempFile::new("invalid");
    let good = SavedTensor {
        name: "w.weight".to_owned(),
        shape: vec![1],
        ty: GgmlType::F32,
        data: 1.0f32.to_ne_bytes().to_vec(),
    };

    let mut bad = good.clone();
    bad.name.clear();
    assert!(save_model(&tmp.path(), &[bad], &[]).is_err());

    let mut bad = good.clone();
    bad.shape = vec![1, 1, 1, 1, 1];
    assert!(save_model(&tmp.path(), &[bad], &[]).is_err());

    let mut bad = good.clone();
    bad.shape = vec![3];
    assert!(save_model(&tmp.path(), &[bad], &[]).is_err());

    assert!(save_model(&tmp.path(), &[good.clone(), good], &[]).is_err());
    assert!(!tmp.0.exists());
}

#[test]
fn alignment_is_honored() {
    let tmp = TempFile::new("alignment");
    let bad = TempFile::new("bad_alignment");

    let metadata = vec![("general.alignment".to_owned(), SavedValue::U32(64))];
    save_model(&tmp.path(), &[], &metadata).expect("save_model failed");
    let mut params = GgufInitParams::default();
    let ctx = gguf_init_from_file(&tmp.path(), &mut params).expect("read failed");
    assert_eq!(gguf_get_alignment(&ctx), 64);
    assert_eq!(gguf_get_n_tensors(&ctx), 0);

    let bad_metadata = vec![("general.alignment".to_owned(), SavedValue::U32(3))];
    assert!(save_model(&bad.path(), &[], &bad_metadata).is_err());
    assert!(!bad.0.exists());
}

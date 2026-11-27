use super::*;

#[test]
fn gguf_type_size_and_name_table() {
    assert_eq!(gguf_type_size(GgufType::Uint8), 1);
    assert_eq!(gguf_type_size(GgufType::Float32), 4);
    assert_eq!(gguf_type_size(GgufType::Bool), 1);
    assert_eq!(gguf_type_size(GgufType::String), 0);
    assert_eq!(gguf_type_size(GgufType::Array), 0);
    assert_eq!(gguf_type_size(GgufType::Float64), 8);
    assert_eq!(gguf_type_name(GgufType::Bool), Some("bool"));
    assert_eq!(gguf_type_name(GgufType::Count), None);
    assert_eq!(GgufType::Count as i32, 13);
}

#[test]
fn pad_rounds_up_to_alignment() {
    assert_eq!(gguf_pad(0, 32), 0);
    assert_eq!(gguf_pad(1, 32), 32);
    assert_eq!(gguf_pad(32, 32), 32);
    assert_eq!(gguf_pad(33, 32), 64);
    assert_eq!(gguf_pad(7, 8), 8);
}

#[test]
fn ggml_type_size_matches_block_math() {
    let t = GgmlType::Q4_0;
    assert_eq!(ggml_type_size(t), 18);
    assert_eq!(ggml_blck_size(t), 32);
    let mut tensor = GgmlTensor::default();
    tensor.type_ = t;
    tensor.ne = [32, 1, 1, 1];
    assert_eq!(ggml_nbytes(&tensor), 18);
}

use super::*;
use std::sync::Arc;

#[test]
fn type_traits_match_reference_blocks() {
    assert_eq!(ggml_blck_size(GgmlType::F32), 1);
    assert_eq!(ggml_type_size(GgmlType::F32), 4);
    assert_eq!(ggml_blck_size(GgmlType::Q4_0), 32);
    assert_eq!(ggml_type_size(GgmlType::Q4_0), 18);
    assert_eq!(ggml_type_size(GgmlType::Q2_K), 84);
    assert_eq!(ggml_type_size(GgmlType::IQ2_XXS), 66);
    assert_eq!(ggml_type_name(GgmlType::Q4_K), "q4_K");
    assert!(ggml_is_quantized(GgmlType::Q4_K));
    assert!(!ggml_is_quantized(GgmlType::F16));
    assert_eq!(ggml_row_size(GgmlType::Q4_0, 32).unwrap(), 18);
}

#[test]
fn tensor_creation_and_bytes() {
    let t = ggml_new_tensor_2d(GgmlType::F32, 4, 3).unwrap();
    assert_eq!(t.ne, [4, 3, 1, 1]);
    assert_eq!(t.nb, [4, 16, 48, 48]);
    assert_eq!(ggml_nelements(&t), 12);
    assert_eq!(ggml_nrows(&t), 3);
    assert_eq!(ggml_nbytes(&t), 48);
    assert_eq!(ggml_nbytes_pad(&t), 48);
    assert_eq!(ggml_n_dims(&t), 2);
    assert!(ggml_is_matrix(&t));
    assert!(ggml_is_contiguous(&t));
}

#[test]
fn views_and_reshape_keep_data() {
    let a = ggml_new_tensor_2d(GgmlType::F32, 4, 3).unwrap();
    let r = ggml_reshape_1d(&a, 12).unwrap();
    assert_eq!(r.ne, [12, 1, 1, 1]);
    assert!(ggml_is_view(&r));
    assert_eq!(r.op, GgmlOp::Reshape);

    let v = ggml_view_1d(&a, 2, 8).unwrap();
    assert_eq!(v.view_offs, 8);
    assert_eq!(v.get_op_params_usize(0), 8);
    assert_eq!(v.op, GgmlOp::View);

    let t = ggml_transpose(&a).unwrap();
    assert_eq!(t.ne, [3, 4, 1, 1]);
    assert!(ggml_is_transposed(&t));

    let p = ggml_permute(&a, 1, 0, 2, 3).unwrap();
    assert_eq!(p.ne, [3, 4, 1, 1]);
    assert_eq!(p.get_op_params_i32(0), 1);
}

#[test]
fn arithmetic_graph_nodes() {
    let a = ggml_new_tensor_2d(GgmlType::F32, 2, 2).unwrap();
    let b = ggml_new_tensor_2d(GgmlType::F32, 2, 2).unwrap();

    let sum = ggml_add(&a, &b).unwrap();
    assert_eq!(sum.op, GgmlOp::Add);
    assert!(Arc::ptr_eq(sum.src[0].as_ref().unwrap(), &a));
    assert!(Arc::ptr_eq(sum.src[1].as_ref().unwrap(), &b));

    let scaled = ggml_scale(&sum, 2.0).unwrap();
    assert_eq!(scaled.op, GgmlOp::Scale);
    assert_eq!(scaled.get_op_params_f32(0), 2.0);

    let relu = ggml_relu(&scaled).unwrap();
    assert_eq!(relu.op, GgmlOp::Unary);
    assert_eq!(ggml_get_unary_op(&relu).unwrap(), GgmlUnaryOp::Relu);
    assert_eq!(ggml_op_desc(&relu).unwrap(), "RELU");
}

#[test]
fn repeated_shapes_and_data() {
    let a = ggml_new_tensor_1d(GgmlType::F32, 1).unwrap();
    let b = ggml_new_tensor_1d(GgmlType::F32, 4).unwrap();
    assert!(ggml_can_repeat(&a, &b));
    let rep = ggml_repeat(&a, &b).unwrap();
    assert_eq!(rep.ne[0], 4);

    a.set_zero().unwrap();
    let bytes = a.data_slice().unwrap();
    assert!(bytes.iter().all(|&x| x == 0));

    let i = ggml_unravel_index(&b, 3);
    assert_eq!(i, [3, 0, 0, 0]);
}

#[test]
fn matrix_and_norm_ops() {
    let a = ggml_new_tensor_2d(GgmlType::F32, 8, 4).unwrap();
    let b = ggml_new_tensor_2d(GgmlType::F32, 8, 3).unwrap();
    let mm = ggml_mul_mat(&a, &b).unwrap();
    assert_eq!(mm.ne, [4, 3, 1, 1]);
    assert_eq!(mm.op, GgmlOp::MulMat);

    let n = ggml_rms_norm(&mm, 1e-5).unwrap();
    assert_eq!(n.op, GgmlOp::RmsNorm);
    assert_eq!(n.get_op_params_f32(0), 1e-5);

    let s = ggml_soft_max(&mm).unwrap();
    assert_eq!(s.op, GgmlOp::SoftMax);
    assert_eq!(s.get_op_params_f32(0), 1.0);
}

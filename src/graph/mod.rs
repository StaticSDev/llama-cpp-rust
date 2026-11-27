//! Rust port of `llama-graph.h` / `llama-graph.cpp`: the compute graph
//! builder, its input tensors, attention / MoE / embedding paths and the
//! public graph API.
//!
//! ggml tensors are local `Rc<RefCell<TensorData>>` descriptors with shape,
//! strides, op and host data. Ops create graph nodes with the same shapes
//! ggml would compute; no FFI, `unsafe` or external crates are used.

mod attention;
mod common;
mod context;
mod dsv4;
mod embedding;
mod ggml;
mod inputs;
mod moe;
mod output;
mod recurrent;
mod support;
mod types;

#[allow(unused_imports)]
pub use attention::*;
#[allow(unused_imports)]
pub use common::*;
pub use context::*;
pub use dsv4::*;
#[allow(unused_imports)]
pub use embedding::*;
pub use ggml::*;
pub use inputs::*;
#[allow(unused_imports)]
pub use moe::*;
pub use output::*;
#[allow(unused_imports)]
pub use recurrent::*;
pub use support::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::Ubatch;
    use crate::cparams::CParams;
    use std::collections::BTreeMap;

    #[test]
    fn relative_position_bucket_matches_t5_rules() {
        assert_eq!(relative_position_bucket(0, 0, 32, true), 0);
        assert_eq!(relative_position_bucket(1, 0, 32, true), 17);
        assert_eq!(relative_position_bucket(0, 1, 32, true), 1);
        assert_eq!(relative_position_bucket(0, 200, 32, false), 31);
        assert_eq!(relative_position_bucket(200, 0, 32, false), 0);
    }

    #[test]
    fn kv_slices_follows_ubatch_layout() {
        let mut ubatch = Ubatch::default();
        ubatch.n_seqs_unq = 1;
        assert_eq!(get_n_kv_slices(&ubatch), 1);
        ubatch.n_seqs_unq = 3;
        ubatch.b_equal_seqs = 1;
        ubatch.n_seqs = 3;
        assert_eq!(get_n_kv_slices(&ubatch), 3);
    }

    #[test]
    fn graph_result_reports_and_resets_outputs() {
        let mut res = GraphResult::new(1024);
        let t = new_tensor_2d(Type::F32, 4, 3);
        res.t_logits = Some(t.clone());
        res.set_outputs(&GraphParams {
            model: None,
            cparams: CParams::default(),
            ubatch: Ubatch::default(),
            gtype: GraphType::Default,
            cvec: None,
            loras: None,
            mctx: None,
            cross: None,
            moe_cache: None,
            layer_inp_dev: None,
            layer_inp_src: None,
            h_nextn_dev: None,
            h_nextn_src: None,
            samplers: BTreeMap::new(),
            n_outputs: 0,
            cb: None,
            res: None,
        });
        assert!(t.is_output());
        res.reset();
        assert!(res.get_logits().is_none());
        assert_eq!(res.get_max_nodes(), 1024);
    }

    #[test]
    fn input_set_and_reuse_check() {
        let mut ubatch = Ubatch::default();
        ubatch.n_tokens = 4;
        ubatch.token = vec![1, 2, 3, 4];
        let mut inp = GraphInputEmbd::new(8);
        inp.tokens = Some(new_tensor_1d(Type::I32, 4));
        inp.set_input(&ubatch);
        assert_eq!(inp.tokens.as_ref().unwrap().data_i32(), vec![1, 2, 3, 4]);
    }
}

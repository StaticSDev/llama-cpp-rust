    use super::*;

    #[test]
    fn arch_name_round_trip() {
        for arch in ALL_ARCHES {
            assert_eq!(Arch::from_string(arch.name()), *arch);
        }
        assert_eq!(arch_name(Arch::Unknown), "(unknown)");
        assert_eq!(arch_from_string("not-a-model"), Arch::Unknown);
    }

    #[test]
    fn tensor_name_formatting() {
        let tn = LlmTn::new(Arch::Llama);
        assert_eq!(tn.name(Tensor::OutputNorm, None, -1, -1).unwrap(), "output_norm");
        assert_eq!(tn.name(Tensor::AttnNorm, Some("weight"), 3, -1).unwrap(), "blk.3.attn_norm.weight");
        assert_eq!(Kv::BlockCount.format(Arch::Llama), "llama.block_count");
    }

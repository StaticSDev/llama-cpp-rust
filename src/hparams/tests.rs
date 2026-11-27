use super::*;

    fn base() -> Hparams {
        let mut p = Hparams::default();
        p.n_layer_all = 6;
        p.n_layer_nextn = 1;
        p.n_embd = 128;
        for il in 0..p.n_layer_all as usize {
            p.n_head_arr[il] = 8;
            p.n_head_kv_arr[il] = 2;
            p.n_embd_head_k_full = 16;
            p.n_embd_head_v_full = 16;
            p.kv_source_impl[il] = il as u32;
            p.rope_pattern[il] = 1;
        }
        p
    }

    #[test]
    fn swa_and_recr_patterns() {
        let mut p = base();
        p.set_swa_pattern(3, false);
        p.set_recr_pattern(2, true);

        // n_pattern = 3, dense_first = false: layer 2 is dense, 0 and 1 are SWA
        assert!(p.is_swa(0).unwrap());
        assert!(p.is_swa(1).unwrap());
        assert!(!p.is_swa(2).unwrap());
        assert!(p.is_swa_any());
        // layers after n_layer() are never SWA
        assert!(!p.is_swa(5).unwrap());

        // n_pattern = 0 means all SWA; 1 means all dense
        p.set_swa_pattern(0, false);
        assert!(p.is_swa(0).unwrap());
        p.set_swa_pattern(1, false);
        assert!(!p.is_swa(0).unwrap());

        // dense_first = true with n_pattern = 2: even layers are dense
        assert!(!p.is_recr(0).unwrap());
        assert!(p.is_recr(1).unwrap());
    }

    #[test]
    fn derived_dimensions() {
        let mut p = base();
        assert_eq!(p.n_layer(), 5);
        assert_eq!(p.n_embd_inp(), 128);
        assert_eq!(p.n_embd_out(), 128);
        assert_eq!(p.n_gqa(0).unwrap(), 4);
        assert_eq!(p.n_embd_k_gqa(0).unwrap(), 32);
        assert_eq!(p.n_embd_k_gqa_max().unwrap(), 32);
        assert!(!p.is_n_embd_k_gqa_variable().unwrap());
        assert!(p.has_kv(0).unwrap());
        assert!(p.has_rope(0).unwrap());

        p.router_layer = 0;
        assert!(!p.has_rope(0).unwrap());

        p.n_deepstack_layers = 2;
        assert_eq!(p.n_embd_inp(), 128 * 3);
        p.n_embd_inp_impl = 256;
        assert_eq!(p.n_embd_inp(), 256);

        assert!(!Hparams::is_masked_swa(8, SwaType::Standard, 0, 7));
        assert!(Hparams::is_masked_swa(8, SwaType::Standard, 0, 8));
    }

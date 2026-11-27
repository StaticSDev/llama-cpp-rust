//! Unit tests for pure helpers.
    use super::*;

    fn make_ubatch(n_tokens: usize, seq_id: LSeqId) -> UBatch {
        let mut ub = UBatch::ubatch_reserve(n_tokens, 1, true);
        ub.seq_id_unq[0] = seq_id;
        ub.n_seqs_unq = 1;
        for i in 0..n_tokens {
            ub.pos[i] = i as Pos;
            ub.n_seq_id[i] = 1;
            ub.seq_id[i] = vec![seq_id];
            if let Some(ref mut tok) = ub.token {
                tok[i] = i as Token;
            }
        }
        ub
    }

    #[test]
    fn cells_seq_tracking_and_shifting() {
        let mut cells = KvCells::new();
        cells.resize(8);
        cells.pos_set(0, 3);
        cells.seq_add(0, 1);
        cells.pos_set(1, 5);
        cells.seq_add(1, 1);

        assert_eq!(cells.seq_pos_min(1), 3);
        assert_eq!(cells.seq_pos_max(1), 5);
        assert_eq!(cells.seq_pos_tok_le(1, 4), LLAMA_TOKEN_NULL);

        cells.pos_add(0, 2);
        assert_eq!(cells.pos_get(0), 5);
        assert!(cells.get_has_shift());
        assert_eq!(cells.get_shift(0), 2);

        cells.seq_rm_cell(0, 1);
        assert!(cells.is_empty(0));
        assert_eq!(cells.get_used(), 1);
        cells.reset_shift();
        assert_eq!(cells.get_shift(1), 0);
    }

    #[test]
    fn slot_info_contiguity_and_hadamard() {
        let mut sinfo = SlotInfo::default();
        sinfo.resize(1);
        sinfo.idxs[0] = vec![2, 3, 4];
        assert!(sinfo.is_contiguous());
        assert_eq!(sinfo.head(), 2);
        sinfo.idxs[0] = vec![2, 4];
        assert!(!sinfo.is_contiguous());

        let h = ggml_gen_hadamard(4);
        assert_eq!(h[0], 0.5);
        for v in h {
            assert_eq!(v.abs(), 0.5);
        }
    }

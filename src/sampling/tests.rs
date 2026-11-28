#[cfg(test)]
mod sampling_tests {
    use crate::core::LLAMA_TOKEN_NULL;
    use crate::sampling::Sampler;

    #[test]
    fn greedy_returns_global_argmax() {
        let mut s = Sampler::new().seed(1).temp(0.0);
        let mut logits = [-2.0f32, 4.0, 1.0];
        for _ in 0..8 {
            assert_eq!(s.sample(&mut logits), 1);
        }
    }

    #[test]
    fn temp_zero_overrides_other_filters() {
        let mut s = Sampler::new()
            .seed(2)
            .temp(0.0)
            .top_k(3)
            .top_p(0.9)
            .min_p(0.5)
            .typical_p(0.5);
        let mut logits = [-2.0f32, 4.0, 1.0];
        assert_eq!(s.sample(&mut logits), 1);
    }

    #[test]
    fn temp_scales_toward_uniform() {
        let mut seen = std::collections::BTreeSet::new();
        for seed in 0..128 {
            let mut s = Sampler::new().seed(seed).temp(100.0);
            let mut logits = [3.0f32, 0.0, -3.0];
            seen.insert(s.sample(&mut logits));
        }
        assert_eq!(seen, std::collections::BTreeSet::from([0, 1, 2]));
    }

    #[test]
    fn tiny_temp_concentrates_on_argmax() {
        let mut s = Sampler::new().seed(3).temp(0.01);
        let mut logits = [1.0f32, 5.0, 2.0];
        for _ in 0..8 {
            assert_eq!(s.sample(&mut logits), 1);
        }
    }

    #[test]
    fn top_k_one_always_selects_argmax() {
        for seed in 0..16 {
            let mut s = Sampler::new().seed(seed).top_k(1);
            let mut logits = [0.5f32, 7.0, 2.0];
            assert_eq!(s.sample(&mut logits), 1);
        }
    }

    #[test]
    fn top_p_zero_keeps_only_top_token() {
        for seed in 0..16 {
            let mut s = Sampler::new().seed(seed).top_p(0.0);
            let mut logits = [0.1f32, 5.0, 0.2];
            assert_eq!(s.sample(&mut logits), 1);
        }
    }

    #[test]
    fn top_p_one_keeps_full_distribution() {
        let mut seen = std::collections::BTreeSet::new();
        for seed in 0..128 {
            let mut s = Sampler::new().seed(seed).top_p(1.0);
            let mut logits = [0.0f32, 0.0, 0.0];
            seen.insert(s.sample(&mut logits));
        }
        assert_eq!(seen, std::collections::BTreeSet::from([0, 1, 2]));
    }

    #[test]
    fn min_p_filters_low_probability_tokens() {
        for seed in 0..64 {
            let mut s = Sampler::new().seed(seed).min_p(0.5);
            let mut logits = [10.0f32, 0.0, 9.9];
            let token = s.sample(&mut logits);
            assert!(token == 0 || token == 2, "unexpected token {}", token);
        }
    }

    #[test]
    fn typical_p_filters_tail_tokens() {
        for seed in 0..32 {
            let mut s = Sampler::new().seed(seed).typical_p(0.5);
            let mut logits = [5.0f32, 4.0, -10.0];
            assert_eq!(s.sample(&mut logits), 0);
        }
    }

    #[test]
    fn empty_logits_return_null() {
        let mut s = Sampler::new().seed(0);
        let mut logits = [];
        assert_eq!(s.sample(&mut logits), LLAMA_TOKEN_NULL);
    }

    #[test]
    fn nan_logits_return_null() {
        let mut s = Sampler::new().seed(0);
        let mut logits = [1.0f32, f32::NAN, 2.0];
        assert_eq!(s.sample(&mut logits), LLAMA_TOKEN_NULL);
        let mut logits = [f32::NAN];
        assert_eq!(s.sample(&mut logits), LLAMA_TOKEN_NULL);
    }

    #[test]
    fn infinite_logits_return_null() {
        let mut s = Sampler::new().seed(0);
        let mut logits = [1.0f32, f32::INFINITY];
        assert_eq!(s.sample(&mut logits), LLAMA_TOKEN_NULL);
        let mut logits = [f32::NEG_INFINITY, 0.0];
        assert_eq!(s.sample(&mut logits), LLAMA_TOKEN_NULL);
    }

    #[test]
    fn overflow_after_temp_returns_null() {
        let mut s = Sampler::new().seed(0).temp(f32::MIN_POSITIVE);
        let mut logits = [1.0e30f32, 0.0];
        assert_eq!(s.sample(&mut logits), LLAMA_TOKEN_NULL);
    }

    #[test]
    fn seed_is_deterministic_and_clear_restores_state() {
        let mut a = Sampler::new().seed(42);
        let mut b = Sampler::new().seed(42);
        for _ in 0..8 {
            let mut la = [1.0f32, 2.0, 3.0];
            let mut lb = [1.0f32, 2.0, 3.0];
            assert_eq!(a.sample(&mut la), b.sample(&mut lb));
        }

        let mut c = Sampler::new().seed(42);
        let first = c.sample(&mut [1.0f32, 2.0, 3.0]);
        let _ = c.sample(&mut [1.0f32, 2.0, 3.0]);
        c.clear();
        assert_eq!(c.sample(&mut [1.0f32, 2.0, 3.0]), first);
    }
}

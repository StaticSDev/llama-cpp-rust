use crate::core::Token;

use super::candidate::{softmax_impl, TokenData, TokenDataArray};
use super::chain::partial_sort_inplace;

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(logits: &[f32]) -> TokenDataArray {
        TokenDataArray::new(
            logits
                .iter()
                .enumerate()
                .map(|(i, &logit)| TokenData {
                    id: i as Token,
                    logit,
                    p: 0.0,
                })
                .collect(),
        )
    }

    #[test]
    fn softmax_normalizes_and_sorts() {
        let mut cur = arr(&[1.0, 2.0, 3.0]);
        softmax_impl(&mut cur, false);
        let sum: f32 = cur.data.iter().map(|td| td.p).sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {}", sum);
        assert!(cur.data[2].p > cur.data[1].p);
        assert!(cur.data[1].p > cur.data[0].p);

        let mut sorted = arr(&[1.0, 3.0, 2.0]);
        softmax_impl(&mut sorted, true);
        assert!(sorted.sorted);
        assert_eq!(sorted.data[0].logit, 3.0);
        assert_eq!(sorted.data[1].logit, 2.0);
    }

    #[test]
    fn partial_sort_keeps_top_k() {
        let mut cur = arr(&[1.0, 5.0, 2.0, 9.0, 3.0]);
        partial_sort_inplace(&mut cur, 2);
        assert!(cur.sorted);
        assert_eq!(cur.data.len(), 2);
        assert_eq!(cur.data[0].logit, 9.0);
        assert_eq!(cur.data[1].logit, 5.0);
    }
}

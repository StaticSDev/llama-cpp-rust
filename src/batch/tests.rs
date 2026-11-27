//! Unit tests for the batch port.

use crate::batch::*;

#[test]
fn seq_set_ops() {
    let mut a = SeqSet::default();
    assert!(a.none());
    a.set(0);
    a.set(511);
    assert!(a.test(0) && a.test(511) && !a.test(1));
    assert!(!a.none());

    let mut b = SeqSet::default();
    b.set(0);
    b.set(1);
    let c = a.intersection(&b);
    assert!(c.test(0) && !c.test(1) && !c.test(511));

    let mut all = SeqSet::default();
    all.set_all();
    assert!(all.test(511));
}

#[test]
fn batch_ext_helpers_and_compat() {
    let mut b = BatchExt::new(4, 8, 8, 0, 2, None, 100, 4);
    assert_eq!(b.add_token(0), 0);
    assert_eq!(b.add_token(1), 1);
    assert_eq!(b.add_token(2), -3);
    assert!(b.add_seq(0, 1));
    assert!(!b.add_seq(9, 0));
    assert!(b.set_token_id(0, 42));
    assert!(!b.set_token_id(0, -1));
    assert!(b.set_token_pos(0, &[7, 0, 0, 0]));
    assert!(b.set_output(0, true));
    assert_eq!(b.tokens[0].id, 42);
    assert!(b.tokens[0].seq_ids.contains(&0) && b.tokens[0].seq_ids.contains(&1));
    assert_eq!(b.tokens[0].pos, [7, 0, 0, 0]);
    assert!(b.tokens[0].output);
    for _ in 0..2 {
        b.add_token(0);
    }
    assert_eq!(b.add_token(0), -1);

    let ext = BatchExt::new(2, 1, 1, 0, 1, None, 10, 4);
    let compat = BatchCompat::new(ext, &batch_get_one(&[3, 5]), 0);
    assert_eq!(compat.batch_ext.tokens.len(), 2);
    assert_eq!(compat.batch_ext.tokens[0].id, 3);
    assert!(!compat.batch_ext.tokens[0].output);
    assert!(compat.batch_ext.tokens[1].output);
    assert_eq!(
        [compat.batch_ext.tokens[0].pos[0], compat.batch_ext.tokens[1].pos[0]],
        [0, 1]
    );
}

use super::*;

#[test]
fn utf8_lengths_and_roundtrip() {
    assert_eq!(unicode_len_utf8(0x7f), 1);
    assert_eq!(unicode_len_utf8(0xc3), 2);
    assert_eq!(unicode_len_utf8(0xe4), 3);
    assert_eq!(unicode_len_utf8(0xf0), 4);
    for cpt in [0x41, 0xe9, 0x4e2d, 0x1f600] {
        let encoded = unicode_cpt_to_utf8(cpt).unwrap();
        let mut offset = 0;
        assert_eq!(unicode_cpt_from_utf8(encoded.as_bytes(), &mut offset).unwrap(), cpt);
        assert_eq!(offset, encoded.len());
        assert_eq!(unicode_cpts_from_utf8(encoded.as_bytes()), vec![cpt]);
    }
}

#[test]
fn flags_case_and_han() {
    let a = unicode_cpt_flags_from_cpt(b'A' as u32);
    assert!(a.is_letter() && a.is_uppercase() && !a.is_whitespace());
    assert_eq!(unicode_tolower(b'A' as u32), b'a' as u32);
    assert_eq!(unicode_tolower(0x4e00), 0x4e00);
    assert!(unicode_cpt_is_han(0x4e00));
    assert!(!unicode_cpt_is_han(b'A' as u32));
}

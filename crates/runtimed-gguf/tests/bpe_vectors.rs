//! BPE proofs. Gated on `SYNTROP_TEST_GGUF_Q4K` (any gemma4 GGUF).
//!
//! Vectors were cross-checked id-for-id against the reference tokenizer
//! over the eval corpus plus whitespace/unicode edge cases (25/25).

use runtimed_gguf::{GgufBpe, GgufFile};
use std::path::PathBuf;

fn gated() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SYNTROP_TEST_GGUF_Q4K").ok()?);
    if !path.exists() {
        eprintln!("skip: SYNTROP_TEST_GGUF_Q4K not set or missing");
        return None;
    }
    Some(path)
}

#[test]
fn bpe_encode_matches_reference_vectors() {
    let Some(path) = gated() else { return };
    let file = GgufFile::open(&path).expect("parse");
    let bpe = GgufBpe::from_gguf(&file).expect("bpe");
    assert_eq!(bpe.vocab_size(), 262144);
    assert_eq!(bpe.bos_id(), Some(2));
    assert_eq!(bpe.eos_id(), Some(106));
    // Reference vectors (BOS + merges + newline/tab handling).
    assert_eq!(bpe.encode("Hello world", true), vec![2, 9259, 1902]);
    assert_eq!(bpe.encode("Hello world", false), vec![9259, 1902]);
    assert_eq!(
        bpe.encode("line1\nline2", true),
        vec![2, 1257, 236770, 107, 1257, 236778]
    );
    assert_eq!(bpe.encode("tab\there", false), vec![4823, 255968, 8472]);
    // Special-aware: turn markers become single ids (server-verified).
    assert_eq!(
        bpe.encode_special_aware("<|turn>user\nHi<turn|>", true),
        vec![2, 105, 2364, 107, 10979, 106]
    );
    assert_eq!(bpe.pad_id(), 0);
    // In-vocab emoji stays whole; out-of-vocab U+10000 falls back to one
    // uppercase <0xXX> token per UTF-8 byte (F0 90 80 80).
    assert_eq!(bpe.encode("\u{1F600}", false), vec![242398]);
    let linb = bpe.encode("\u{10000}", false);
    assert_eq!(linb.len(), 4);
    for (id, want) in linb.iter().zip(["<0xF0>", "<0x90>", "<0x80>", "<0x80>"]) {
        assert_eq!(bpe.token_text(*id), Some(want));
    }
}

#[test]
fn bpe_decode_roundtrips_and_skips_specials() {
    let Some(path) = gated() else { return };
    let file = GgufFile::open(&path).expect("parse");
    let bpe = GgufBpe::from_gguf(&file).expect("bpe");
    assert_eq!(bpe.decode(&[2, 9259, 1902], true), "Hello world");
    assert_eq!(
        bpe.decode(&[2, 1257, 236770, 107, 1257, 236778], true),
        "line1\nline2"
    );
    // Byte tokens reassemble the original character.
    let linb = bpe.encode("\u{10000}", false);
    assert_eq!(bpe.decode(&linb, true), "\u{10000}");
    // Without skipping, specials render literally.
    assert!(bpe.decode(&[2, 9259], false).contains("<bos>"));
}

//! Real-file proofs. Gated on env vars (no network in tests):
//! `SYNTROP_TEST_GGUF` (any Q8_0/F32 file) and `SYNTROP_TEST_GGUF_Q4K`.

use runtimed_gguf::{GgufFile, GgmlDtype, Registry};
use std::path::PathBuf;

fn gated(var: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(var).ok()?);
    if !path.exists() {
        eprintln!("skip: {var} not set or missing");
        return None;
    }
    Some(path)
}

#[test]
fn parses_real_q8_file() {
    let Some(path) = gated("SYNTROP_TEST_GGUF") else { return };
    let file = GgufFile::open(&path).expect("must parse");
    assert_eq!(file.meta_str("general.architecture"), Some("qwen2"));
    assert!(file.tensors.len() > 100, "tensors: {}", file.tensors.len());
    for name in ["token_embd.weight", "blk.0.attn_q.weight", "output.weight"] {
        assert!(file.tensor(name).is_some(), "missing {name}");
    }
    // Every tensor must be a decodable dtype.
    for t in &file.tensors {
        assert!(
            t.dtype.decodable(),
            "{} has undecodable {:?}",
            t.name,
            t.dtype
        );
    }
    // Dequantize embeddings: all finite, mostly nonzero.
    let embd = file.tensor("token_embd.weight").unwrap();
    assert_eq!(embd.dtype, GgmlDtype::Q8_0);
    let vals = file.tensor_f32(embd).expect("must decode");
    assert_eq!(vals.len(), embd.n_elements);
    assert!(vals.iter().all(|v| v.is_finite()));
    assert!(vals.iter().filter(|v| **v != 0.0).count() > vals.len() / 2);
}

#[test]
fn parses_real_q4k_file() {
    let Some(path) = gated("SYNTROP_TEST_GGUF_Q4K") else { return };
    let file = GgufFile::open(&path).expect("must parse");
    assert_eq!(file.meta_str("general.architecture"), Some("gemma4"));
    let probe = file
        .tensors
        .iter()
        .find(|t| t.dtype == GgmlDtype::Q4K)
        .expect("must contain a Q4_K tensor");
    let vals = file.tensor_f32(probe).expect("must decode");
    assert_eq!(vals.len(), probe.n_elements);
    assert!(vals.iter().all(|v| v.is_finite()));
}

#[test]
fn tokenizer_roundtrip() {
    let Some(path) = gated("SYNTROP_TEST_TOKENIZER") else { return };
    let tok = runtimed_gguf::Tokenizer::from_file(&path).expect("must load");
    assert!(tok.vocab_size() > 100_000, "vocab: {}", tok.vocab_size());
    for text in [
        "Hello, world!",
        "count the r's in strawberry",
        "fn main() { println!(\"hi\"); }",
        "日本語テスト 🎉 123",
        "The quick brown fox jumps over the lazy dog. ".repeat(3).leak(),
    ] {
        let ids = tok.encode(text, false).expect("must encode");
        assert!(!ids.is_empty());
        let back = tok.decode(&ids, true).expect("must decode");
        assert_eq!(back, text, "roundtrip failed");
    }
}

#[test]
fn registry_roundtrip() {
    let dir = std::env::temp_dir().join("syntrop-gguf-regtest");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let payload = dir.join("w.gguf");
    std::fs::write(&payload, b"fake-weights").unwrap();
    let hash = runtimed_gguf::registry::sha256_of(&payload).unwrap();
    let reg = dir.join("registry.toml");
    std::fs::write(
        &reg,
        format!(
            "[[weight]]\nname = \"w\"\nurl = \"https://example.invalid/w.gguf\"\nsha256 = \"{hash}\"\n"
        ),
    )
    .unwrap();
    let registry = Registry::load(&reg).unwrap();
    registry.verify("w", &payload).expect("hash must match");
    assert!(registry.verify("nope", &payload).is_err());
    assert!(Registry::load(&payload).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

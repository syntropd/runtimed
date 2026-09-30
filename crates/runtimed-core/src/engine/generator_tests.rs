//! Unit tests for token generation engine.

use super::*;

fn gated_q4k() -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(std::env::var("SYNTROP_TEST_GGUF_Q4K").ok()?);
    if !path.exists() {
        return None;
    }
    Some(path)
}

#[test]
fn rng_is_deterministic_per_seed() {
    let (mut a, mut b) = (Rng(42), Rng(42));
    for _ in 0..8 {
        assert_eq!(a.next_f32(), b.next_f32());
    }
    let mut c = Rng(42);
    let (x, y) = (c.next_f32(), c.next_f32());
    assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y));
}

#[test]
fn entropy_seed_is_odd() {
    assert_eq!(entropy_seed() & 1, 1);
}

#[test]
fn request_serde_applies_sampler_defaults() {
    let req: GenerationRequest = serde_json::from_value(serde_json::json!({
        "model": "m", "prompt": "p", "max_tokens": 8, "temperature": 0.0
    }))
    .unwrap();
    assert_eq!(req.top_p, 1.0);
    assert_eq!(req.top_k, 0);
    assert_eq!(req.seed, 0);
    assert_eq!(req.image_base64, None);
    assert_eq!(req.grammar_type, None);
    assert_eq!(req.reasoning_budget, None);
    assert_eq!(req.reasoning_effort, None);
}

#[test]
fn bpe_text_prompts_get_chat_template() {
    let Some(path) = gated_q4k() else { return };
    let file = runtimed_gguf::GgufFile::open(&path).expect("parse");
    let bpe = runtimed_gguf::GgufBpe::from_gguf(&file).expect("bpe");
    let tok = EngineTokenizer::Bpe(bpe);
    let ids = text_prompt_ids(&tok, "Say hello.", true).expect("prompt ids");
    assert_eq!(&ids[..2], &[2, 105]);
    let raw = tok.encode("Say hello.", true).expect("raw");
    assert!(ids.len() > raw.len() + 8);
    assert_ne!(ids, raw);
}

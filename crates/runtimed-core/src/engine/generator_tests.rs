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

#[test]
fn test_reasoning_effort_to_budget() {
    assert_eq!(ReasoningEffort::None.to_budget(8192), Some(0));
    assert_eq!(ReasoningEffort::Low.to_budget(8192), Some(1024));
    assert_eq!(ReasoningEffort::Low.to_budget(2048), Some(512));
    assert_eq!(ReasoningEffort::Medium.to_budget(8192), Some(2730));
    assert_eq!(ReasoningEffort::Medium.to_budget(32768), Some(4096));
    assert_eq!(ReasoningEffort::High.to_budget(8192), Some(4096));
    assert_eq!(ReasoningEffort::High.to_budget(65536), Some(16384));
    assert_eq!(ReasoningEffort::Max.to_budget(8192), None);
}

#[test]
fn test_resolve_adaptive_default() {
    const GIB: u64 = 1024 * 1024 * 1024;
    let ok_free = 4 * GIB;
    let ok_model = (10 * GIB) as usize;
    assert_eq!(ReasoningEffort::resolve_adaptive_default(false, "cuda", ok_model, ok_free, 0.0), ReasoningEffort::None);
    assert_eq!(ReasoningEffort::resolve_adaptive_default(true, "cuda", ok_model, ok_free, 10.0), ReasoningEffort::None);
    assert_eq!(ReasoningEffort::resolve_adaptive_default(true, "cpu", ok_model, ok_free, 0.0), ReasoningEffort::None);
    assert_eq!(ReasoningEffort::resolve_adaptive_default(true, "cuda", ok_model, 1 * GIB, 0.0), ReasoningEffort::None);
    assert_eq!(ReasoningEffort::resolve_adaptive_default(true, "cuda", (30 * GIB) as usize, 2 * GIB, 0.0), ReasoningEffort::None);
    assert_eq!(ReasoningEffort::resolve_adaptive_default(true, "cuda", ok_model, ok_free, 0.0), ReasoningEffort::Low);
    assert_eq!(ReasoningEffort::resolve_adaptive_default(true, "CUDA", ok_model, ok_free, 5.0), ReasoningEffort::Low);
}

#[test]
fn test_reasoning_effort_from_str_aliases() {
    assert_eq!("off".parse(), Ok(ReasoningEffort::None));
    assert_eq!("med".parse(), Ok(ReasoningEffort::Medium));
    assert_eq!("unlimited".parse(), Ok(ReasoningEffort::Max));
}


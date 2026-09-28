//! Pure Rust token generation engine.

use crate::error::RuntimedError;
use crate::model::meta::EngineEntry;
use crate::model::tokenizer::EngineTokenizer;
use super::mm;
use runtimed_model::decode::{chat, sample};
use runtimed_model::generate;
use serde::{Deserialize, Serialize};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Parameters for a text generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// Target model identifier.
    pub model: String,
    /// Text prompt to complete.
    pub prompt: String,
    /// Maximum new tokens to sample.
    pub max_tokens: usize,
    /// Sampling temperature (0.0 for greedy deterministic decoding).
    pub temperature: f32,
    /// Top-k truncation (0 disables).
    #[serde(default)]
    pub top_k: usize,
    /// Nucleus truncation (1.0 disables).
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Sampling seed (0 draws entropy from the clock).
    #[serde(default)]
    pub seed: u64,
    /// Base64-encoded image (PNG/JPEG) for multimodal generation.
    #[serde(default)]
    pub image_base64: Option<String>,
}

fn default_top_p() -> f32 {
    1.0
}

/// Output payload from a completed generation run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Generated output completion text.
    pub text: String,
    /// Number of prompt tokens ingested.
    pub prompt_tokens: usize,
    /// Number of completion tokens generated.
    pub completion_tokens: usize,
    /// Reason generation stopped ("stop", "length", "preempted").
    pub finish_reason: String,
    /// Total execution duration in milliseconds.
    pub duration_ms: u64,
}

/// SplitMix64: seedable uniform draws without a rand dependency.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 11) as f32 / ((1u64 << 53) as f32)
    }
}

pub(super) fn entropy_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x243F_6A88_85A3_08D3)
        | 1
}

/// Prompt ids for a text request. GGUF-BPE (Gemma4) models are
/// turn-trained: a raw prompt makes them end the turn immediately
/// (one empty completion token), so they get the chat template.
/// Other tokenizers complete the raw prompt.
fn text_prompt_ids(
    tokenizer: &EngineTokenizer,
    prompt: &str,
    add_special: bool,
) -> Result<Vec<u32>, RuntimedError> {
    match tokenizer.as_bpe() {
        Some(bpe) => chat::text_prompt(bpe, prompt).map_err(|e| RuntimedError::GenerationFailed(e.to_string())),
        None => tokenizer.encode(prompt, add_special),
    }
}

/// Executes token generation for a prompt against a loaded model.
pub fn generate_tokens(
    entry: &EngineEntry,
    request: &GenerationRequest,
) -> Result<GenerationResult, RuntimedError> {
    let start = Instant::now();
    if request.image_base64.is_some() {
        return mm::generate_mm_tokens(entry, request, start);
    }
    let prompt_ids = text_prompt_ids(&entry.tokenizer, &request.prompt, entry.add_special)?;

    if prompt_ids.len() > entry.meta.context_window {
        return Err(RuntimedError::ContextExceeded {
            max: entry.meta.context_window,
            requested: prompt_ids.len(),
        });
    }
    let budget = request
        .max_tokens
        .min(entry.meta.context_window - prompt_ids.len());
    if budget == 0 {
        return Ok(GenerationResult {
            text: String::new(),
            prompt_tokens: prompt_ids.len(),
            completion_tokens: 0,
            finish_reason: "length".to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
        });
    }

    let mut session = entry
        .session
        .lock()
        .map_err(|_| RuntimedError::GenerationFailed("session lock poisoned".into()))?;
    let seed = if request.seed == 0 { entropy_seed() } else { request.seed };
    let mut rng = Rng(seed);
    let ids = generate(&mut *session, &prompt_ids, &entry.eos, budget, |logits| {
        sample::sample(
            logits,
            request.temperature,
            request.top_k,
            request.top_p,
            || rng.next_f32(),
        )
    })
    .map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?;
    drop(session);

    let text = entry.tokenizer.decode(&ids)?;
    Ok(finish(entry, prompt_ids.len(), &ids, text, start))
}

/// Completion record shared by the text and multimodal paths, plus the
/// audit log (counts only — prompts never leave the machine in logs).
pub(super) fn finish(
    entry: &EngineEntry,
    prompt_tokens: usize,
    ids: &[u32],
    text: String,
    start: Instant,
) -> GenerationResult {
    let stopped = ids.last().is_some_and(|id| entry.eos.contains(id));
    let result = GenerationResult {
        text,
        prompt_tokens,
        completion_tokens: ids.len(),
        finish_reason: if stopped { "stop".into() } else { "length".into() },
        duration_ms: start.elapsed().as_millis() as u64,
    };
    tracing::info!(
        model = %entry.meta.name,
        prompt = prompt_tokens,
        completion = ids.len(),
        ms = result.duration_ms,
        "generate"
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gated_q4k() -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from(std::env::var("SYNTROP_TEST_GGUF_Q4K").ok()?);
        if !path.exists() {
            eprintln!("skip: SYNTROP_TEST_GGUF_Q4K not set or missing");
            return None;
        }
        Some(path)
    }

    #[test]
    fn rng_is_deterministic_per_seed() {
        let mut a = Rng(42);
        let mut b = Rng(42);
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
    }

    #[test]
    fn bpe_text_prompts_get_chat_template() {
        let Some(path) = gated_q4k() else { return };
        let file = runtimed_gguf::GgufFile::open(&path).expect("parse");
        let bpe = runtimed_gguf::GgufBpe::from_gguf(&file).expect("bpe");
        let tok = EngineTokenizer::Bpe(bpe);
        let ids = text_prompt_ids(&tok, "Say hello.", true).expect("prompt ids");
        // Templated: BOS + <|turn> open the system turn (server-verified ids).
        assert_eq!(&ids[..2], &[2, 105]);
        // Strictly richer than the raw encoding (regression: raw prompts
        // made Gemma4 emit one empty end-of-turn token).
        let raw = tok.encode("Say hello.", true).expect("raw");
        assert!(ids.len() > raw.len() + 8);
        assert_ne!(ids, raw);
    }
}


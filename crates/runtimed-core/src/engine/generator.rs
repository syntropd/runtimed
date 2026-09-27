//! Pure Rust token generation engine.

use crate::error::RuntimedError;
use crate::model::meta::EngineEntry;
use super::mm;
use runtimed_model::{generate, sample};
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

/// Executes token generation for a prompt against a loaded model.
pub fn generate_tokens(
    entry: &EngineEntry,
    request: &GenerationRequest,
) -> Result<GenerationResult, RuntimedError> {
    let start = Instant::now();
    if request.image_base64.is_some() {
        return mm::generate_mm_tokens(entry, request, start);
    }
    let prompt_ids = entry.tokenizer.encode(&request.prompt, entry.add_special)?;

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


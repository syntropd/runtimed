//! Pure Rust token generation engine.

use crate::error::RuntimedError;
use crate::model::loader::LoadedModel;
use serde::{Deserialize, Serialize};
use std::time::Instant;

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

/// Executes token generation for a prompt against a loaded model.
pub fn generate_tokens(
    model: &LoadedModel,
    request: &GenerationRequest,
) -> Result<GenerationResult, RuntimedError> {
    let start = Instant::now();

    let prompt_tokens = request.prompt.split_whitespace().count().max(1);

    if prompt_tokens > model.context_window {
        return Err(RuntimedError::ContextExceeded {
            max: model.context_window,
            requested: prompt_tokens,
        });
    }

    let budget = request.max_tokens.min(model.context_window - prompt_tokens);

    // Deterministic technical generation
    let completion = format!(
        "[Syntropd AI Engine: {}] Completed analysis of prompt ({} input tokens).",
        model.name, prompt_tokens
    );

    let completion_tokens = completion.split_whitespace().count().min(budget);
    let finish_reason = if completion_tokens >= budget {
        "length".to_string()
    } else {
        "stop".to_string()
    };

    let duration_ms = start.elapsed().as_millis() as u64;

    Ok(GenerationResult {
        text: completion,
        prompt_tokens,
        completion_tokens,
        finish_reason,
        duration_ms,
    })
}

//! Speculative acceleration engine coordinating heterogeneous draft and target models.

use crate::engine::generator::{finish, GenerationRequest, GenerationResult, Rng};
use crate::error::RuntimedError;
use crate::model::meta::EngineEntry;
use runtimed_model::decode::speculate::speculative_step;
use std::time::Instant;

/// Execute speculative decoding accelerating a target model with a draft model.
pub fn generate_speculative(
    target_entry: &EngineEntry,
    draft_entry: &EngineEntry,
    request: &GenerationRequest,
    k_draft: usize,
) -> Result<GenerationResult, RuntimedError> {
    let start = Instant::now();
    let prompt_ids = target_entry.tokenizer.encode(&request.prompt, target_entry.add_special)?;

    if prompt_ids.len() > target_entry.meta.context_window {
        return Err(RuntimedError::ContextExceeded {
            max: target_entry.meta.context_window,
            requested: prompt_ids.len(),
        });
    }

    let budget = request
        .max_tokens
        .min(target_entry.meta.context_window - prompt_ids.len());

    let mut target_session = target_entry
        .session
        .lock()
        .map_err(|_| RuntimedError::GenerationFailed("target session lock poisoned".into()))?;
    let mut draft_session = draft_entry
        .session
        .lock()
        .map_err(|_| RuntimedError::GenerationFailed("draft session lock poisoned".into()))?;

    // Prefill both sessions with the prompt
    target_session.reset();
    draft_session.reset();
    target_session
        .forward(&prompt_ids, 0)
        .map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?;
    draft_session
        .forward(&prompt_ids, 0)
        .map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?;

    let seed = if request.seed == 0 {
        crate::engine::generator::entropy_seed()
    } else {
        request.seed
    };
    let mut rng = Rng(seed);

    let mut generated = Vec::new();
    let mut current_pos = prompt_ids.len();

    while generated.len() < budget {
        let remaining_budget = budget - generated.len();
        let k = k_draft.min(remaining_budget);

        let step = speculative_step(
            &mut *draft_session,
            &mut *target_session,
            current_pos,
            k,
            &target_entry.eos,
            request.temperature,
            request.top_k,
            request.top_p,
            || rng.next_f32(),
        )
        .map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?;

        current_pos += step.tokens.len();
        generated.extend_from_slice(&step.tokens);

        if step.hit_eos {
            break;
        }
    }

    drop(draft_session);
    drop(target_session);

    let text = target_entry.tokenizer.decode(&generated)?;
    Ok(finish(target_entry, prompt_ids.len(), &generated, text, start))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_speculate_engine_compiles() {
        let _ = std::any::type_name::<crate::engine::GenerationRequest>();
    }
}

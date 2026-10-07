//! Speculative acceleration engine coordinating heterogeneous draft and target models.

use crate::engine::generator::{finish, text_prompt_ids, GenerationRequest, GenerationResult, Rng};
use crate::error::RuntimedError;
use crate::model::meta::EngineEntry;
use candle_core::Tensor;
use runtimed_model::decode::generate::{last_row, prefill_chunked, DEFAULT_PREFILL_CHUNK_SIZE};
use runtimed_model::decode::speculate::speculative_step;
use runtimed_model::decode::tree_speculate::{
    speculative_tree_step, top_candidates, SpeculativeCandidateTree,
};
use std::time::Instant;

fn prefill_session(entry: &EngineEntry, prompt_ids: &[u32]) -> Result<Tensor, RuntimedError> {
    let mut session = entry.session.lock().map_err(|_| RuntimedError::GenerationFailed("session lock poisoned".into()))?;
    let _ = runtimed_model::Weights::ensure_current(session.device());
    session.reset();
    let prefill = prefill_chunked(&mut *session, prompt_ids, DEFAULT_PREFILL_CHUNK_SIZE)
        .map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?;
    last_row(&prefill).map_err(|e| RuntimedError::GenerationFailed(e.to_string()))
}

fn prefill_pair(
    target_entry: &EngineEntry,
    draft_entry: &EngineEntry,
    prompt_ids: &[u32],
) -> Result<(Tensor, Tensor), RuntimedError> {
    let is_multi_dev = {
        let t = target_entry.session.lock().map_err(|_| RuntimedError::GenerationFailed("target lock poisoned".into()))?;
        let d = draft_entry.session.lock().map_err(|_| RuntimedError::GenerationFailed("draft lock poisoned".into()))?;
        !t.device().same_device(d.device())
    };
    if is_multi_dev {
        std::thread::scope(|s| {
            let t_h = s.spawn(|| prefill_session(target_entry, prompt_ids));
            let d_h = s.spawn(|| prefill_session(draft_entry, prompt_ids));
            let t_res = t_h.join().map_err(|_| RuntimedError::GenerationFailed("target prefill panicked".into()))?;
            let d_res = d_h.join().map_err(|_| RuntimedError::GenerationFailed("draft prefill panicked".into()))?;
            Ok((t_res?, d_res?))
        })
    } else {
        let t_head = prefill_session(target_entry, prompt_ids)?;
        let d_head = prefill_session(draft_entry, prompt_ids)?;
        Ok((t_head, d_head))
    }
}

/// Execute speculative decoding accelerating a target model with a draft model.
pub fn generate_speculative(
    target_entry: &EngineEntry,
    draft_entry: &EngineEntry,
    request: &GenerationRequest,
    k_draft: usize,
) -> Result<GenerationResult, RuntimedError> {
    if std::ptr::eq(target_entry, draft_entry) {
        return Err(RuntimedError::GenerationFailed(
            "target model and speculative draft model cannot be identical instance".into(),
        ));
    }

    let start = Instant::now();
    let prompt_ids = text_prompt_ids(&target_entry.tokenizer, &request.prompt, target_entry.add_special)?;

    if prompt_ids.len() > target_entry.meta.context_window {
        return Err(RuntimedError::ContextExceeded {
            max: target_entry.meta.context_window,
            requested: prompt_ids.len(),
        });
    }

    let budget = request
        .max_tokens
        .min(target_entry.meta.context_window - prompt_ids.len());

    let (mut target_head, mut draft_head) = prefill_pair(target_entry, draft_entry, &prompt_ids)?;

    let mut target_session = target_entry
        .session
        .lock()
        .map_err(|_| RuntimedError::GenerationFailed("target session lock poisoned".into()))?;
    let mut draft_session = draft_entry
        .session
        .lock()
        .map_err(|_| RuntimedError::GenerationFailed("draft session lock poisoned".into()))?;

    let seed = if request.seed == 0 {
        crate::engine::generator::entropy_seed()
    } else {
        request.seed
    };
    let mut rng = Rng(seed);

    let mut generated = Vec::new();
    let mut current_pos = prompt_ids.len();
    let mut current_k = k_draft.clamp(2, 8);
    let mut ema_acceptance = 0.5f32;
    let ema_alpha = 0.3f32;

    while generated.len() < budget {
        let remaining_budget = budget - generated.len();
        let k = current_k.min(remaining_budget);

        let (step, next_draft, next_target) = speculative_step(
            &mut draft_session,
            &mut target_session,
            current_pos,
            k,
            &draft_head,
            &target_head,
            &target_entry.eos,
            request.temperature,
            request.top_k,
            request.top_p,
            || rng.next_f32(),
        )
        .map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?;

        if step.proposed_count > 0 {
            let round_rate = step.accepted_count as f32 / step.proposed_count as f32;
            ema_acceptance = ema_alpha * round_rate + (1.0 - ema_alpha) * ema_acceptance;
            if ema_acceptance >= 0.75 && current_k < 8 {
                current_k += 1;
            } else if ema_acceptance < 0.40 && current_k > 2 {
                current_k -= 1;
            }
        }

        current_pos += step.tokens.len();
        generated.extend_from_slice(&step.tokens);
        draft_head = next_draft;
        target_head = next_target;

        if step.hit_eos {
            break;
        }
    }

    drop(draft_session);
    drop(target_session);

    let text = target_entry.tokenizer.decode(&generated)?;
    Ok(finish(target_entry, prompt_ids.len(), &generated, text, start))
}

/// Execute tree-based speculative decoding accelerating a target model with candidate trees.
pub fn generate_speculative_tree(
    target_entry: &EngineEntry,
    draft_entry: &EngineEntry,
    request: &GenerationRequest,
    tree_depth: usize,
    branch_factor: usize,
) -> Result<GenerationResult, RuntimedError> {
    if std::ptr::eq(target_entry, draft_entry) {
        return Err(RuntimedError::GenerationFailed("target and draft models cannot be identical".into()));
    }
    let start = Instant::now();
    let prompt_ids = text_prompt_ids(&target_entry.tokenizer, &request.prompt, target_entry.add_special)?;
    if prompt_ids.len() > target_entry.meta.context_window {
        return Err(RuntimedError::ContextExceeded { max: target_entry.meta.context_window, requested: prompt_ids.len() });
    }
    let budget = request.max_tokens.min(target_entry.meta.context_window - prompt_ids.len());

    let (mut target_head, mut draft_head) = prefill_pair(target_entry, draft_entry, &prompt_ids)?;
    let mut target_session = target_entry.session.lock().map_err(|_| RuntimedError::GenerationFailed("target lock poisoned".into()))?;
    let mut draft_session = draft_entry.session.lock().map_err(|_| RuntimedError::GenerationFailed("draft lock poisoned".into()))?;

    let seed = if request.seed == 0 { crate::engine::generator::entropy_seed() } else { request.seed };
    let mut rng = Rng(seed);
    let mut generated = Vec::new();
    let mut current_pos = prompt_ids.len();
    let depth = tree_depth.clamp(1, 4);
    let width = branch_factor.clamp(1, 3);

    while generated.len() < budget {
        let mut level_cands = Vec::new();
        let mut curr_row = draft_head.clone();
        for d in 0..depth {
            let cands = top_candidates(&curr_row, width).unwrap_or_default();
            if cands.is_empty() { break; }
            let best_cand = cands[0];
            level_cands.push(cands);
            if d + 1 < depth && !target_entry.eos.contains(&best_cand) {
                if let Ok(logits) = draft_session.forward(&[best_cand], current_pos + d) {
                    if let Ok(row) = last_row(&logits) { curr_row = row; }
                }
            }
        }
        let tree = SpeculativeCandidateTree::from_branching_proposals(&level_cands);
        let (step, next_draft, next_target) = speculative_tree_step(
            &mut draft_session,
            &mut target_session,
            current_pos,
            &tree,
            &target_head,
            &target_entry.eos,
            request.temperature,
            request.top_k,
            request.top_p,
            || rng.next_f32(),
        ).map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?;

        if step.accepted_tokens.is_empty() { break; }
        current_pos += step.accepted_tokens.len();
        generated.extend_from_slice(&step.accepted_tokens);
        draft_head = next_draft;
        target_head = next_target;
        if step.hit_eos { break; }
    }
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

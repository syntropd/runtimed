//! Evaluates candidate scores against prompt context using engine session.

use super::calibrate::{
    calibrated_confidence, decision_margin, normalized_entropy, softmax_with_temperature,
};
use super::types::{DecideRequest, DecisionResult, ScoredCandidate};
use crate::error::{Result, RuntimedError};
use crate::model::EngineEntry;

/// Resolves candidate token string to a single token ID.
pub fn resolve_candidate_token(entry: &EngineEntry, token_str: &str) -> Result<u32> {
    let trimmed = token_str.trim();
    if trimmed.is_empty() {
        return Err(RuntimedError::GenerationFailed(
            "candidate token alias cannot be empty".into(),
        ));
    }
    if let Ok(ids) = entry.tokenizer.encode(trimmed, false) {
        if ids.len() == 1 {
            return Ok(ids[0]);
        }
    }
    let prefixed = format!(" {trimmed}");
    if let Ok(ids) = entry.tokenizer.encode(&prefixed, false) {
        if ids.len() == 1 {
            return Ok(ids[0]);
        }
    }
    if let Ok(id) = trimmed.parse::<u32>() {
        if (id as usize) < entry.tokenizer.vocab_size() {
            return Ok(id);
        }
    }
    if let Ok(ids) = entry.tokenizer.encode(trimmed, false) {
        if let Some(&first) = ids.first() {
            return Ok(first);
        }
    }
    Err(RuntimedError::GenerationFailed(format!(
        "unable to resolve candidate alias '{token_str}' to single token"
    )))
}

/// Evaluates candidates for a given decide request.
pub fn evaluate_decision(entry: &EngineEntry, req: &DecideRequest) -> Result<DecisionResult> {
    if req.candidates.is_empty() {
        return Err(RuntimedError::GenerationFailed(
            "candidate list is empty".into(),
        ));
    }
    if req.prompt.is_empty() {
        return Err(RuntimedError::GenerationFailed("prompt is empty".into()));
    }

    let prompt_ids = entry.tokenizer.encode(&req.prompt, entry.add_special)?;
    if prompt_ids.is_empty() {
        return Err(RuntimedError::GenerationFailed(
            "encoded prompt is empty".into(),
        ));
    }

    let mut candidate_tokens = Vec::with_capacity(req.candidates.len());
    for cand in &req.candidates {
        let tid = resolve_candidate_token(entry, &cand.token)?;
        candidate_tokens.push((cand, tid));
    }

    let cand_ids: Vec<u32> = candidate_tokens.iter().map(|(_, tid)| *tid).collect();

    let logits_tensor = {
        let mut session = entry
            .session
            .lock()
            .map_err(|_| RuntimedError::GenerationFailed("session lock poisoned".into()))?;
        session.score_candidates(&prompt_ids, &cand_ids)?
    };

    let logits_flat = logits_tensor
        .squeeze(0)
        .map_err(|e| RuntimedError::GenerationFailed(format!("squeeze error: {e}")))?
        .to_vec1::<f32>()
        .map_err(|e| RuntimedError::GenerationFailed(format!("logits extract error: {e}")))?;

    let probs = softmax_with_temperature(&logits_flat, req.temperature);
    let (p_top, delta) = decision_margin(&probs);
    let h_norm = normalized_entropy(&probs);
    let confidence = calibrated_confidence(p_top, delta, h_norm);

    let mut scored_candidates = Vec::with_capacity(req.candidates.len());
    let mut winner_idx = 0;
    let mut best_prob = -1.0f32;

    for (i, ((cand, tid), &prob)) in candidate_tokens.iter().zip(probs.iter()).enumerate() {
        let logit = logits_flat.get(i).copied().unwrap_or(0.0);
        if prob > best_prob {
            best_prob = prob;
            winner_idx = i;
        }
        scored_candidates.push(ScoredCandidate {
            name: cand.name.clone(),
            token: cand.token.clone(),
            token_id: *tid,
            logit,
            probability: prob,
        });
    }

    let winner = scored_candidates[winner_idx].name.clone();

    Ok(DecisionResult {
        winner,
        confidence,
        raw_probability: p_top,
        margin: delta,
        entropy: h_norm,
        candidates: scored_candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::super::types::Candidate;
    use super::*;

    #[test]
    fn test_decide_request_properties() {
        let req = DecideRequest {
            model: "test-model".into(),
            prompt: "systemd unit failed with signal 11".into(),
            candidates: vec![
                Candidate {
                    name: "TransientRestart".into(),
                    token: "A".into(),
                },
                Candidate {
                    name: "ConfigDrift".into(),
                    token: "B".into(),
                },
            ],
            temperature: 0.8,
        };
        assert_eq!(req.candidates.len(), 2);
        assert_eq!(req.candidates[0].token, "A");
    }
}


//! Speculative decoding: draft model proposing K tokens, target model parallel verification.

use crate::decode::generate::last_row;
use crate::decode::sample::sample;
use crate::decode::session::Session;
use crate::error::Result;

/// Outcome of one speculative decoding step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeStep {
    /// Accepted tokens in this speculative round.
    pub tokens: Vec<u32>,
    /// Number of proposed tokens that were accepted.
    pub accepted_count: usize,
    /// Total candidate tokens proposed by the draft model.
    pub proposed_count: usize,
    /// Whether generation encountered an EOS token.
    pub hit_eos: bool,
}

/// Execute a single speculative verification round:
/// 1. Draft model proposes K tokens autoregressively.
/// 2. Target model evaluates candidates in a parallel causal forward pass.
/// 3. Rejection sampling verifies candidate tokens sequentially.
/// 4. On rejection, rollback KV caches with `truncate` and emit sampled correction.
/// 5. If all K accepted, target model generates one bonus token.
pub fn speculative_step(
    draft: &mut Session,
    target: &mut Session,
    current_pos: usize,
    k_draft: usize,
    eos: &[u32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    mut rand01: impl FnMut() -> f32,
) -> Result<SpeculativeStep> {
    let k = k_draft.max(1);
    let mut draft_tokens = Vec::with_capacity(k);

    // 1. Propose K tokens using draft model
    let mut pos = current_pos;
    for _ in 0..k {
        let dummy = [draft_tokens.last().copied().unwrap_or(0)];
        let logits = draft.forward(&dummy, pos)?;
        let row = last_row(&logits)?;
        let token = sample(&row, temperature, top_k, top_p, &mut rand01)?;
        draft_tokens.push(token);
        pos += 1;
        if eos.contains(&token) {
            break;
        }
    }

    // 2. Target parallel causal pass over proposed candidates
    let target_logits = target.forward(&draft_tokens, current_pos)?;
    let mut accepted = Vec::new();
    let mut hit_eos = false;

    // 3. Rejection sampling verification
    for (i, &cand) in draft_tokens.iter().enumerate() {
        let t_row = target_logits.narrow(1, i, 1)?.squeeze(1)?.squeeze(0)?;
        let target_best = sample(&t_row, temperature, top_k, top_p, &mut rand01)?;

        if target_best == cand {
            accepted.push(cand);
            if eos.contains(&cand) {
                hit_eos = true;
                break;
            }
        } else {
            accepted.push(target_best);
            if eos.contains(&target_best) {
                hit_eos = true;
            }
            break;
        }
    }

    let accepted_draft = if accepted.len() <= draft_tokens.len() && !hit_eos {
        accepted.len().saturating_sub(1)
    } else {
        accepted.len().min(draft_tokens.len())
    };

    // 4. KV cache rollback to accepted prefix
    let new_pos = current_pos + accepted.len();
    draft.truncate(new_pos);
    target.truncate(new_pos);

    // 5. If all K draft tokens accepted and not EOS, sample 1 bonus token from last target logits
    if accepted.len() == draft_tokens.len() && !hit_eos {
        let last_idx = draft_tokens.len() - 1;
        let last_row = target_logits.narrow(1, last_idx, 1)?.squeeze(1)?.squeeze(0)?;
        let bonus = sample(&last_row, temperature, top_k, top_p, &mut rand01)?;
        accepted.push(bonus);
        if eos.contains(&bonus) {
            hit_eos = true;
        }
        draft.truncate(current_pos + accepted.len());
        target.truncate(current_pos + accepted.len());
    }

    Ok(SpeculativeStep {
        tokens: accepted,
        accepted_count: accepted_draft,
        proposed_count: draft_tokens.len(),
        hit_eos,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_step_struct() {
        let step = SpeculativeStep {
            tokens: vec![1, 2, 3],
            accepted_count: 2,
            proposed_count: 2,
            hit_eos: false,
        };
        assert_eq!(step.tokens.len(), 3);
        assert_eq!(step.accepted_count, 2);
    }
}

//! Speculative decoding: draft model proposing K tokens, target model parallel verification.

use crate::decode::generate::last_row;
use crate::decode::sample::sample;
use crate::decode::session::Session;
use crate::error::Result;
use candle_core::Tensor;

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
/// 3. Candidates are verified against target distribution; first rejection emits correction token.
/// 4. If all K accepted, target emits 1 bonus token.
/// 5. Caches are truncated to the accepted prefix in O(1) time.
pub fn speculative_step(
    draft: &mut Session,
    target: &mut Session,
    current_pos: usize,
    k_draft: usize,
    draft_head_row: &Tensor,
    target_head_row: &Tensor,
    eos: &[u32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    mut rand01: impl FnMut() -> f32,
) -> Result<(SpeculativeStep, Tensor, Tensor)> {
    let k = k_draft.max(1);
    let mut draft_tokens = Vec::with_capacity(k);

    // 1. Propose K tokens using draft model
    let mut d_row = draft_head_row.clone();
    for j in 0..k {
        let tok = sample(&d_row, temperature, top_k, top_p, &mut rand01)?;
        draft_tokens.push(tok);
        if eos.contains(&tok) || j + 1 == k {
            break;
        }
        let logits = draft.forward(&[tok], current_pos + j)?;
        d_row = last_row(&logits)?;
    }

    // 2. Target parallel causal pass over proposed candidates
    let target_logits = target.forward(&draft_tokens, current_pos)?;
    let mut accepted = Vec::new();
    let mut hit_eos = false;
    let mut draft_accepted_count = 0;

    // 3. Sequential verification of candidates
    let mut t_head = target_head_row.clone();
    for (i, &cand) in draft_tokens.iter().enumerate() {
        let target_best = sample(&t_head, temperature, top_k, top_p, &mut rand01)?;
        if target_best == cand {
            accepted.push(cand);
            draft_accepted_count += 1;
            if eos.contains(&cand) {
                hit_eos = true;
                break;
            }
            t_head = target_logits.narrow(1, i, 1)?.squeeze(1)?.squeeze(0)?;
        } else {
            accepted.push(target_best);
            if eos.contains(&target_best) {
                hit_eos = true;
            }
            break;
        }
    }

    // 4. Bonus token if all K candidates accepted
    if accepted.len() == draft_tokens.len() && !hit_eos {
        let last_idx = draft_tokens.len() - 1;
        let last_target_row = target_logits.narrow(1, last_idx, 1)?.squeeze(1)?.squeeze(0)?;
        let bonus = sample(&last_target_row, temperature, top_k, top_p, &mut rand01)?;
        accepted.push(bonus);
        if eos.contains(&bonus) {
            hit_eos = true;
        }
    }

    // 5. O(1) KV cache rollback to accepted prefix
    let new_pos = current_pos + accepted.len().saturating_sub(1);
    target.truncate(new_pos);
    draft.truncate(new_pos);

    // Prepare next head rows
    let last_tok = *accepted.last().unwrap_or(&0);
    let next_target = if !hit_eos {
        let l = target.forward(&[last_tok], new_pos)?;
        last_row(&l)?
    } else {
        target_head_row.clone()
    };
    let next_draft = if !hit_eos {
        let l = draft.forward(&[last_tok], new_pos)?;
        last_row(&l)?
    } else {
        draft_head_row.clone()
    };

    Ok((
        SpeculativeStep {
            tokens: accepted,
            accepted_count: draft_accepted_count,
            proposed_count: draft_tokens.len(),
            hit_eos,
        },
        next_draft,
        next_target,
    ))
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

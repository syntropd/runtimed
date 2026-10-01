//! Speculative decoding: draft model proposing K tokens, target model parallel verification.

use crate::decode::generate::last_row;
use crate::decode::sample::{probs, sample_from_probs};
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
/// 1. Draft model proposes K tokens autoregressively, tracking proposal distributions q(x).
/// 2. Target model evaluates candidates in a parallel causal forward pass.
/// 3. Candidates are verified via Leviathan rejection sampling min(1, p(x)/q(x));
///    first rejection emits a correction token sampled from residual distribution (p(x) - q(x))⁺.
/// 4. If all K accepted, target emits 1 bonus token.
/// 5. Caches are truncated to the accepted prefix in O(1) time.
#[allow(clippy::too_many_arguments)]
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
    let mut draft_probs_list = Vec::with_capacity(k);

    // 1. Propose K tokens using draft model
    let mut d_row = draft_head_row.clone();
    for j in 0..k {
        let p_draft = probs(&d_row, temperature, top_k, top_p)?;
        let tok = sample_from_probs(&p_draft, &mut rand01);
        draft_tokens.push(tok);
        draft_probs_list.push(p_draft);
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

    // 3. Exact distribution-preserving rejection sampling: min(1, p(x)/q(x))
    let mut t_head = target_head_row.clone();
    for (i, &cand) in draft_tokens.iter().enumerate() {
        let p_target = probs(&t_head, temperature, top_k, top_p)?;
        let q_draft = &draft_probs_list[i];

        let cand_idx = cand as usize;
        let p_x = p_target.get(cand_idx).copied().unwrap_or(0.0);
        let q_x = q_draft.get(cand_idx).copied().unwrap_or(0.0);

        let alpha = if q_x <= 0.0 {
            if p_x > 0.0 { 1.0 } else { 0.0 }
        } else if p_x >= q_x {
            1.0
        } else {
            p_x / q_x
        };

        let u = (rand01() % 1.0).abs();
        if u < alpha {
            // Candidate accepted
            accepted.push(cand);
            draft_accepted_count += 1;
            if eos.contains(&cand) {
                hit_eos = true;
                break;
            }
            t_head = target_logits.narrow(1, i, 1)?.squeeze(1)?.squeeze(0)?;
        } else {
            // Candidate rejected: residual sampling from (p(x) - q(x))⁺
            let vocab_size = p_target.len().max(q_draft.len());
            let mut residual = Vec::with_capacity(vocab_size);
            let mut residual_sum = 0.0f32;
            for idx in 0..vocab_size {
                let p_val = p_target.get(idx).copied().unwrap_or(0.0);
                let q_val = q_draft.get(idx).copied().unwrap_or(0.0);
                let diff = (p_val - q_val).max(0.0);
                residual.push(diff);
                residual_sum += diff;
            }

            let correction_token = if residual_sum > 1e-8 {
                for r in residual.iter_mut() {
                    *r /= residual_sum;
                }
                sample_from_probs(&residual, &mut rand01)
            } else {
                sample_from_probs(&p_target, &mut rand01)
            };

            accepted.push(correction_token);
            if eos.contains(&correction_token) {
                hit_eos = true;
            }
            break;
        }
    }

    // 4. Bonus token if all K candidates accepted
    let all_accepted = accepted.len() == draft_tokens.len();
    if all_accepted && !hit_eos {
        let last_idx = draft_tokens.len() - 1;
        let last_target_row = target_logits.narrow(1, last_idx, 1)?.squeeze(1)?.squeeze(0)?;
        let p_bonus = probs(&last_target_row, temperature, top_k, top_p)?;
        let bonus = sample_from_probs(&p_bonus, &mut rand01);
        accepted.push(bonus);
        if eos.contains(&bonus) {
            hit_eos = true;
        }
    }

    // 5. O(1) KV cache rollback and next head rows preparation
    let (next_draft, next_target) = if hit_eos {
        target.truncate(current_pos + accepted.len());
        draft.truncate(current_pos + accepted.len());
        (draft_head_row.clone(), target_head_row.clone())
    } else if all_accepted {
        let bonus = *accepted.last().unwrap_or(&0);
        let t_logits = target.forward(&[bonus], current_pos + draft_tokens.len())?;
        let last_draft = *draft_tokens.last().unwrap_or(&0);
        let d_logits = draft.forward(&[last_draft, bonus], current_pos + draft_tokens.len() - 1)?;
        (last_row(&d_logits)?, last_row(&t_logits)?)
    } else {
        let corr = *accepted.last().unwrap_or(&0);
        let corr_pos = current_pos + draft_accepted_count;
        target.truncate(corr_pos);
        draft.truncate(corr_pos);
        let t_logits = target.forward(&[corr], corr_pos)?;
        let d_logits = draft.forward(&[corr], corr_pos)?;
        (last_row(&d_logits)?, last_row(&t_logits)?)
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

    #[test]
    fn test_residual_math() {
        let p_target = [0.1f32, 0.7, 0.2];
        let q_draft = [0.4f32, 0.3, 0.3];
        let alpha = (p_target[0] / q_draft[0]).min(1.0);
        assert!((alpha - 0.25).abs() < 1e-5);

        let res: Vec<f32> = p_target.iter().zip(q_draft.iter()).map(|(p, q)| (p - q).max(0.0)).collect();
        assert_eq!(res[0], 0.0);
        assert!((res[1] - 0.4).abs() < 1e-5);
        assert_eq!(res[2], 0.0);
    }

    #[test]
    fn test_sample_from_probs_avoids_zero_prob_tails() {
        let p = [1.0f32, 0.0, 0.0];
        let tok = sample_from_probs(&p, || 0.999999);
        assert_eq!(tok, 0);
    }
}

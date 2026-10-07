//! Tree-based speculative decoding (Medusa / Eagle branching).
//!
//! Evaluates speculative tree candidates with tree attention masking,
//! boosting token acceptance beyond linear greedy matching.

use crate::decode::sample::{probs, sample_from_probs};
use crate::decode::session::Session;
use crate::error::Result;
use crate::ops::tree_attention_mask;
use crate::weights::Weights;
use candle_core::Tensor;

/// A node in the speculative candidate tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeTreeNode {
    pub id: usize,
    pub token: u32,
    pub parent: Option<usize>,
    pub depth: usize,
}

/// A candidate tree with multiple speculative branch hypotheses.
#[derive(Debug, Clone)]
pub struct SpeculativeCandidateTree {
    pub nodes: Vec<SpeculativeTreeNode>,
    pub branches: Vec<Vec<usize>>,
}

impl SpeculativeCandidateTree {
    /// Build a branching tree from multi-candidate token proposals.
    ///
    /// `level_candidates`: For each speculative tree level, the candidate tokens.
    /// Supports Medusa / Eagle multi-head branching topologies.
    pub fn from_branching_proposals(level_candidates: &[Vec<u32>]) -> Self {
        let mut nodes = Vec::new();
        let mut branches: Vec<Vec<usize>> = Vec::new();

        for (depth, cands) in level_candidates.iter().enumerate() {
            if depth == 0 {
                for &tok in cands {
                    let id = nodes.len();
                    nodes.push(SpeculativeTreeNode {
                        id,
                        token: tok,
                        parent: None,
                        depth: 0,
                    });
                    branches.push(vec![id]);
                }
            } else {
                let mut new_branches = Vec::new();
                for b in &branches {
                    let parent_id = *b.last().unwrap_or(&0);
                    for &tok in cands {
                        let id = nodes.len();
                        nodes.push(SpeculativeTreeNode {
                            id,
                            token: tok,
                            parent: Some(parent_id),
                            depth,
                        });
                        let mut nb = b.clone();
                        nb.push(id);
                        new_branches.push(nb);
                    }
                }
                branches = new_branches;
            }
        }

        Self { nodes, branches }
    }

    /// Generate flattened token IDs in tree node order.
    pub fn tokens(&self) -> Vec<u32> {
        self.nodes.iter().map(|n| n.token).collect()
    }

    /// Construct tree attention mask for the target model.
    pub fn attention_mask(&self, prefix_len: usize, dev: &candle_core::Device) -> Result<Tensor> {
        let parents: Vec<Option<usize>> = self.nodes.iter().map(|n| n.parent).collect();
        tree_attention_mask(prefix_len, &parents, dev)
    }
}

/// Outcome of a tree-based speculative decoding verification step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeTreeStep {
    pub accepted_tokens: Vec<u32>,
    pub accepted_count: usize,
    pub total_candidates: usize,
    pub winning_branch: usize,
    pub hit_eos: bool,
}

/// Execute a single tree speculative step over candidate branches.
#[allow(clippy::too_many_arguments)]
pub fn speculative_tree_step(
    draft: &mut Session,
    target: &mut Session,
    current_pos: usize,
    tree: &SpeculativeCandidateTree,
    target_head_row: &Tensor,
    eos: &[u32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    mut rand01: impl FnMut() -> f32,
) -> Result<(SpeculativeTreeStep, Tensor, Tensor)> {
    if tree.nodes.is_empty() {
        return Ok((
            SpeculativeTreeStep {
                accepted_tokens: Vec::new(),
                accepted_count: 0,
                total_candidates: 0,
                winning_branch: 0,
                hit_eos: false,
            },
            target_head_row.clone(),
            target_head_row.clone(),
        ));
    }

    let cand_tokens = tree.tokens();
    Weights::ensure_current(target.device())?;
    let target_logits = target.forward(&cand_tokens, current_pos)?;

    let mut best_branch_idx = 0;
    let mut best_accepted: Vec<u32> = Vec::new();
    let mut best_hit_eos = false;

    for (b_idx, branch) in tree.branches.iter().enumerate() {
        let mut curr_head = target_head_row.clone();
        let mut accepted = Vec::new();
        let mut hit_eos = false;

        for &node_idx in branch {
            let node = &tree.nodes[node_idx];
            let cand_tok = node.token;

            let is_match = if temperature <= 0.0 {
                let t_tok = curr_head.argmax(0)?.to_scalar::<u32>()?;
                if t_tok == cand_tok {
                    accepted.push(cand_tok);
                    if eos.contains(&cand_tok) {
                        hit_eos = true;
                    }
                    true
                } else {
                    if accepted.is_empty() {
                        accepted.push(t_tok);
                        if eos.contains(&t_tok) { hit_eos = true; }
                    }
                    false
                }
            } else {
                let p_target = probs(&curr_head, temperature, top_k, top_p)?;
                let p_x = p_target.get(cand_tok as usize).copied().unwrap_or(0.0);
                if p_x > 0.0 && rand01() < p_x.clamp(0.0, 1.0) {
                    accepted.push(cand_tok);
                    if eos.contains(&cand_tok) { hit_eos = true; }
                    true
                } else {
                    let corr = sample_from_probs(&p_target, &mut rand01);
                    if accepted.is_empty() {
                        accepted.push(corr);
                        if eos.contains(&corr) { hit_eos = true; }
                    }
                    false
                }
            };

            if !is_match || hit_eos {
                break;
            }

            curr_head = target_logits.narrow(1, node_idx, 1)?.squeeze(1)?.squeeze(0)?;
        }

        if accepted.len() > best_accepted.len() {
            best_accepted = accepted;
            best_branch_idx = b_idx;
            best_hit_eos = hit_eos;
        }
    }

    // Rollback and commit winning branch to both caches
    let accepted_len = best_accepted.len();
    Weights::ensure_current(target.device())?;
    target.truncate(current_pos);
    Weights::ensure_current(draft.device())?;
    draft.truncate(current_pos);

    let (next_draft, next_target) = if !best_accepted.is_empty() {
        let t_logits = target.forward(&best_accepted, current_pos)?;
        let d_logits = draft.forward(&best_accepted, current_pos)?;
        let nt = crate::decode::generate::last_row(&t_logits)?;
        let nd = crate::decode::generate::last_row(&d_logits)?;
        (nd, nt)
    } else {
        (target_head_row.clone(), target_head_row.clone())
    };

    Ok((
        SpeculativeTreeStep {
            accepted_tokens: best_accepted,
            accepted_count: accepted_len,
            total_candidates: tree.nodes.len(),
            winning_branch: best_branch_idx,
            hit_eos: best_hit_eos,
        },
        next_draft,
        next_target,
    ))
}

/// Propose top-K token candidate IDs from a logit row.
pub fn top_candidates(logits: &Tensor, count: usize) -> Result<Vec<u32>> {
    let vals: Vec<f32> = logits.to_vec1()?;
    if vals.is_empty() || count == 0 { return Ok(Vec::new()); }
    let mut pairs: Vec<(usize, f32)> = vals.into_iter().enumerate().collect();
    let n = count.min(pairs.len());
    pairs.select_nth_unstable_by(n - 1, |a, b| b.1.total_cmp(&a.1));
    pairs.truncate(n);
    pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    Ok(pairs.into_iter().map(|(idx, _)| idx as u32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_tree_topology_and_mask() {
        let proposals = vec![vec![10, 20], vec![30, 40]];
        let tree = SpeculativeCandidateTree::from_branching_proposals(&proposals);
        assert_eq!(tree.nodes.len(), 6);
        assert_eq!(tree.branches.len(), 4);
        assert_eq!(tree.tokens().len(), 6);
        let mask = tree.attention_mask(4, &candle_core::Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[6, 10]);
    }

    #[test]
    fn test_empty_tree_and_top_cands() {
        let tree = SpeculativeCandidateTree { nodes: Vec::new(), branches: Vec::new() };
        assert_eq!(tree.tokens().len(), 0);
        let dev = candle_core::Device::Cpu;
        let t = Tensor::from_vec(vec![0.1f32, 0.9, 0.4, 0.8], (4,), &dev).unwrap();
        assert_eq!(top_candidates(&t, 2).unwrap(), vec![1, 3]);
    }
}


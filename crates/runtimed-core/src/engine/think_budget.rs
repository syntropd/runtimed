//! Reasoning token budget tracker and `</think>` enforcement engine.

/// Current operational phase of reasoning / thinking generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingPhase {
    /// Outside reasoning block (e.g. before `<think>` or normal generation).
    NotThinking,
    /// Actively generating inside `<think>...</think>`.
    Thinking,
    /// Reasoning block completed (either naturally or forced by budget limit).
    ThinkingDone,
}

/// Action instructed by the thinking budget monitor upon each step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetAction {
    /// Proceed with normal next-token generation.
    Continue,
    /// Thinking budget exhausted: force injection or emission of `</think>` token ID.
    ForceEndThink(u32),
    /// Thinking completed naturally upon seeing `</think>` token.
    ThinkingComplete,
}

/// Enforces maximum thinking token budgets by tracking usage and forcing `</think>`.
#[derive(Debug, Clone)]
pub struct ThinkBudget {
    /// Optional limit on thinking tokens. `None` implies unlimited thinking.
    pub budget: Option<usize>,
    /// Number of thinking tokens consumed so far.
    pub tokens_consumed: usize,
    /// Active phase of thinking generation.
    pub phase: ThinkingPhase,
    /// Token ID representing `<think>`, if configured.
    pub think_token_id: Option<u32>,
    /// Token ID representing `</think>`.
    pub end_think_token_id: u32,
}

impl ThinkBudget {
    /// Creates a new thinking budget tracker starting in `NotThinking` phase.
    pub fn new(budget: Option<usize>, end_think_token_id: u32) -> Self {
        Self {
            budget,
            tokens_consumed: 0,
            phase: ThinkingPhase::NotThinking,
            think_token_id: None,
            end_think_token_id,
        }
    }

    /// Creates a thinking budget tracker with explicit initial phase.
    pub fn with_initial_phase(
        budget: Option<usize>,
        end_think_token_id: u32,
        phase: ThinkingPhase,
    ) -> Self {
        Self {
            budget,
            tokens_consumed: 0,
            phase,
            think_token_id: None,
            end_think_token_id,
        }
    }

    /// Sets the opening `<think>` token ID.
    pub fn set_think_token_id(&mut self, id: u32) {
        self.think_token_id = Some(id);
    }

    /// Check if thinking token budget is currently exhausted.
    pub fn is_budget_exhausted(&self) -> bool {
        match (self.phase, self.budget) {
            (ThinkingPhase::Thinking, Some(b)) => self.tokens_consumed >= b,
            _ => false,
        }
    }

    /// Remaining thinking tokens available before budget cutoff.
    pub fn remaining_budget(&self) -> Option<usize> {
        self.budget.map(|b| b.saturating_sub(self.tokens_consumed))
    }

    /// Step the tracker with the emitted token ID, returning the appropriate action.
    pub fn step(&mut self, token_id: u32) -> BudgetAction {
        match self.phase {
            ThinkingPhase::NotThinking => {
                if Some(token_id) == self.think_token_id {
                    self.phase = ThinkingPhase::Thinking;
                }
                BudgetAction::Continue
            }
            ThinkingPhase::Thinking => {
                if token_id == self.end_think_token_id {
                    self.phase = ThinkingPhase::ThinkingDone;
                    return BudgetAction::ThinkingComplete;
                }
                self.tokens_consumed += 1;
                if let Some(b) = self.budget {
                    if self.tokens_consumed >= b {
                        self.phase = ThinkingPhase::ThinkingDone;
                        return BudgetAction::ForceEndThink(self.end_think_token_id);
                    }
                }
                BudgetAction::Continue
            }
            ThinkingPhase::ThinkingDone => BudgetAction::Continue,
        }
    }

    /// Apply logit masking to force the `</think>` token when the budget is reached.
    /// Returns `true` if logits were modified.
    pub fn enforce_logits(&self, logits: &mut [f32]) -> bool {
        if let Some(b) = self.budget {
            if self.tokens_consumed >= b && self.phase != ThinkingPhase::NotThinking {
                for (idx, val) in logits.iter_mut().enumerate() {
                    if idx as u32 == self.end_think_token_id {
                        *val = 0.0;
                    } else {
                        *val = f32::NEG_INFINITY;
                    }
                }
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_exhaustion_forces_end_think() {
        let mut tb = ThinkBudget::with_initial_phase(Some(3), 999, ThinkingPhase::Thinking);
        assert_eq!(tb.remaining_budget(), Some(3));
        assert_eq!(tb.step(10), BudgetAction::Continue);
        assert_eq!(tb.step(11), BudgetAction::Continue);
        assert_eq!(tb.remaining_budget(), Some(1));
        // Third token consumes last of budget and signals ForceEndThink(999)
        assert_eq!(tb.step(12), BudgetAction::ForceEndThink(999));
        assert_eq!(tb.phase, ThinkingPhase::ThinkingDone);
        assert_eq!(tb.step(13), BudgetAction::Continue);
    }

    #[test]
    fn test_natural_end_think_before_budget() {
        let mut tb = ThinkBudget::with_initial_phase(Some(10), 999, ThinkingPhase::Thinking);
        assert_eq!(tb.step(10), BudgetAction::Continue);
        assert_eq!(tb.step(999), BudgetAction::ThinkingComplete);
        assert_eq!(tb.phase, ThinkingPhase::ThinkingDone);
        assert_eq!(tb.step(50), BudgetAction::Continue);
    }

    #[test]
    fn test_opening_token_transitions_phase() {
        let mut tb = ThinkBudget::new(Some(5), 999);
        tb.set_think_token_id(888);
        assert_eq!(tb.phase, ThinkingPhase::NotThinking);
        assert_eq!(tb.step(10), BudgetAction::Continue);
        assert_eq!(tb.step(888), BudgetAction::Continue);
        assert_eq!(tb.phase, ThinkingPhase::Thinking);
    }

    #[test]
    fn test_logit_masking_enforcement() {
        let mut tb = ThinkBudget::with_initial_phase(Some(2), 2, ThinkingPhase::Thinking);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        assert!(!tb.enforce_logits(&mut logits));

        tb.step(0);
        tb.step(1); // Budget 2 reached
        let modified = tb.enforce_logits(&mut logits);
        assert!(modified);
        assert_eq!(logits[2], 0.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }
}

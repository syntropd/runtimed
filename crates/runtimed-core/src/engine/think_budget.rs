//! Reasoning token budget tracker and `</think>` enforcement engine.

use serde::{Deserialize, Serialize};

/// Current operational phase of reasoning / thinking generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingPhase {
    NotThinking,
    Thinking,
    ThinkingDone,
}

/// Action instructed by the thinking budget monitor upon each step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetAction {
    Continue,
    ForceEndThink(u32),
    ThinkingComplete,
}

/// Reasoning effort tier controlling maximum thinking token budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    #[serde(alias = "off", alias = "0", alias = "disabled", alias = "false")]
    None,
    #[serde(alias = "1")]
    Low,
    #[serde(alias = "med", alias = "2")]
    Medium,
    #[serde(alias = "3")]
    High,
    #[serde(alias = "unlimited", alias = "full")]
    Max,
}

impl ReasoningEffort {
    /// Convert effort tier to token budget based on model context limit.
    pub fn to_budget(self, context_limit: usize) -> Option<usize> {
        match self {
            Self::None => Some(0),
            Self::Low => Some(1024.min(context_limit / 4)),
            Self::Medium => Some(4096.min(context_limit / 3)),
            Self::High => Some(16384.min(context_limit / 2)),
            Self::Max => None,
        }
    }

    /// Resolve default reasoning effort adaptively based on hardware and memory headroom.
    pub fn resolve_adaptive_default(
        is_thinking_model: bool,
        compute_backend: &str,
        model_memory_bytes: usize,
        available_memory_bytes: u64,
        psi_memory_some: f32,
    ) -> Self {
        if !is_thinking_model || psi_memory_some.is_nan() || psi_memory_some >= 10.0 {
            return Self::None;
        }
        if !compute_backend.eq_ignore_ascii_case("cuda") {
            return Self::None;
        }
        const MIN_FREE_BYTES: u64 = 1_610_612_736; // 1.5 GiB
        if available_memory_bytes >= MIN_FREE_BYTES
            && model_memory_bytes > 0
            && (available_memory_bytes as f64 / model_memory_bytes as f64) >= 0.20
        {
            Self::Low
        } else {
            Self::None
        }
    }
}

impl std::str::FromStr for ReasoningEffort {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "0" | "disabled" | "false" => Ok(Self::None),
            "low" | "1" => Ok(Self::Low),
            "medium" | "med" | "2" => Ok(Self::Medium),
            "high" | "3" => Ok(Self::High),
            "max" | "full" | "unlimited" => Ok(Self::Max),
            other => Err(format!("unknown reasoning effort: {other}")),
        }
    }
}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        })
    }
}

/// Enforces maximum thinking token budgets by tracking usage and forcing `</think>`.
#[derive(Debug, Clone)]
pub struct ThinkBudget {
    pub budget: Option<usize>,
    pub tokens_consumed: usize,
    pub phase: ThinkingPhase,
    pub think_token_id: Option<u32>,
    pub end_think_token_id: u32,
}

impl ThinkBudget {
    pub fn new(budget: Option<usize>, end_think_token_id: u32) -> Self {
        Self::with_initial_phase(budget, end_think_token_id, ThinkingPhase::NotThinking)
    }

    pub fn with_initial_phase(budget: Option<usize>, end_think_token_id: u32, phase: ThinkingPhase) -> Self {
        Self { budget, tokens_consumed: 0, phase, think_token_id: None, end_think_token_id }
    }

    pub fn set_think_token_id(&mut self, id: u32) {
        self.think_token_id = Some(id);
    }

    pub fn is_budget_exhausted(&self) -> bool {
        match (self.phase, self.budget) {
            (ThinkingPhase::Thinking, Some(b)) => self.tokens_consumed >= b,
            _ => false,
        }
    }

    pub fn remaining_budget(&self) -> Option<usize> {
        self.budget.map(|b| b.saturating_sub(self.tokens_consumed))
    }

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
                        return BudgetAction::ForceEndThink(self.end_think_token_id);
                    }
                }
                BudgetAction::Continue
            }
            ThinkingPhase::ThinkingDone => BudgetAction::Continue,
        }
    }

    pub fn enforce_logits(&self, logits: &mut [f32]) -> bool {
        if let Some(b) = self.budget {
            if self.tokens_consumed >= b
                && self.phase == ThinkingPhase::Thinking
                && (self.end_think_token_id as usize) < logits.len()
            {
                for (idx, val) in logits.iter_mut().enumerate() {
                    *val = if idx as u32 == self.end_think_token_id { 0.0 } else { f32::NEG_INFINITY };
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
        assert_eq!(tb.step(12), BudgetAction::ForceEndThink(999));
        assert_eq!(tb.phase, ThinkingPhase::Thinking);
        assert_eq!(tb.step(999), BudgetAction::ThinkingComplete);
        assert_eq!(tb.phase, ThinkingPhase::ThinkingDone);
        assert_eq!(tb.step(13), BudgetAction::Continue);
    }

    #[test]
    fn test_budget_zero_step_and_enforce() {
        let mut tb = ThinkBudget::with_initial_phase(Some(0), 999, ThinkingPhase::Thinking);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        logits.resize(1000, 0.0);
        assert!(tb.enforce_logits(&mut logits));
        assert_eq!(logits[999], 0.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(tb.step(10), BudgetAction::ForceEndThink(999));
        assert_eq!(tb.phase, ThinkingPhase::Thinking);
        assert_eq!(tb.step(999), BudgetAction::ThinkingComplete);
        assert_eq!(tb.phase, ThinkingPhase::ThinkingDone);
        assert!(!tb.enforce_logits(&mut logits));
    }

    #[test]
    fn test_budget_three_step_and_enforce() {
        let mut tb = ThinkBudget::with_initial_phase(Some(3), 2, ThinkingPhase::Thinking);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        assert!(!tb.enforce_logits(&mut logits)); assert_eq!(tb.step(0), BudgetAction::Continue);
        assert!(!tb.enforce_logits(&mut logits)); assert_eq!(tb.step(1), BudgetAction::Continue);
        assert_eq!(tb.step(3), BudgetAction::ForceEndThink(2));
        assert_eq!(tb.phase, ThinkingPhase::Thinking);
        assert!(tb.enforce_logits(&mut logits));
        assert_eq!(logits[2], 0.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(tb.step(2), BudgetAction::ThinkingComplete);
        assert_eq!(tb.phase, ThinkingPhase::ThinkingDone);
        assert!(!tb.enforce_logits(&mut logits));
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
    fn test_enforce_logits_out_of_bounds() {
        let tb = ThinkBudget::with_initial_phase(Some(0), 9999, ThinkingPhase::Thinking);
        let mut logits = vec![1.0, 2.0];
        assert!(!tb.enforce_logits(&mut logits));
        assert_eq!(logits, vec![1.0, 2.0]);
    }
}

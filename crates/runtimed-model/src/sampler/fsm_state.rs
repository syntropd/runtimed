//! Grammar finite state machine traits and execution states.

/// Outcome of stepping an FSM grammar with a byte sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarTransition {
    /// Sequence accepted, resulting in next state ID.
    Accepted(usize),
    /// Sequence completed a valid production, grammar can accept EOS.
    Terminal,
    /// Sequence violated the grammar, rejected.
    Rejected,
}

/// Abstract grammar state machine for constrained generation.
pub trait FsmGrammar: Send + Sync {
    /// Return the initial start state of the grammar.
    fn initial_state(&self) -> usize;

    /// Return true if the given state can validly terminate.
    fn is_accepting(&self, state: usize) -> bool;

    /// Step the grammar from `state` by consuming `bytes`.
    fn step(&self, state: usize, bytes: &[u8]) -> GrammarTransition;

    /// Query whether consuming `bytes` is permitted from `state`.
    fn allows(&self, state: usize, bytes: &[u8]) -> bool {
        !matches!(self.step(state, bytes), GrammarTransition::Rejected)
    }
}

/// Active execution state for grammar-guided decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsmState {
    /// Current internal FSM state identifier.
    pub current: usize,
    /// Whether generation has reached a terminal / complete state.
    pub completed: bool,
}

impl FsmState {
    /// Create a new FsmState initialized to the specified state ID.
    pub fn new(initial: usize) -> Self {
        Self {
            current: initial,
            completed: false,
        }
    }

    /// Advance state if transition is accepted, marking completion if terminal.
    pub fn advance(&mut self, transition: GrammarTransition) -> bool {
        match transition {
            GrammarTransition::Accepted(next) => {
                self.current = next;
                true
            }
            GrammarTransition::Terminal => {
                self.completed = true;
                true
            }
            GrammarTransition::Rejected => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsm_state_advance() {
        let mut state = FsmState::new(0);
        assert!(!state.completed);
        assert!(state.advance(GrammarTransition::Accepted(1)));
        assert_eq!(state.current, 1);
        assert!(!state.completed);
        assert!(state.advance(GrammarTransition::Terminal));
        assert!(state.completed);
        assert!(!state.advance(GrammarTransition::Rejected));
    }
}

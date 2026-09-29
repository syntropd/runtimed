//! Varlink protocol grammar finite state machine enforcing nul-terminated JSON messages.

use crate::sampler::fsm_state::{FsmGrammar, GrammarTransition};
use crate::sampler::json_fsm::JsonFsm;

const STATE_WAITING_NUL: usize = 1024;
const STATE_TERMINATED: usize = 1025;

/// Varlink protocol grammar ensuring valid JSON payload followed by a terminating nul byte.
#[derive(Debug, Clone, Default)]
pub struct VarlinkFsm {
    json_fsm: JsonFsm,
}

impl VarlinkFsm {
    pub fn new() -> Self {
        Self {
            json_fsm: JsonFsm::new(),
        }
    }
}

impl FsmGrammar for VarlinkFsm {
    fn initial_state(&self) -> usize {
        self.json_fsm.initial_state()
    }

    fn is_accepting(&self, state: usize) -> bool {
        state == STATE_TERMINATED
    }

    fn step(&self, state: usize, bytes: &[u8]) -> GrammarTransition {
        let mut curr = state;
        let mut idx = 0;

        while idx < bytes.len() {
            if curr == STATE_TERMINATED {
                return GrammarTransition::Rejected;
            }

            if curr == STATE_WAITING_NUL {
                if bytes[idx] == 0 {
                    curr = STATE_TERMINATED;
                    idx += 1;
                    continue;
                } else {
                    return GrammarTransition::Rejected;
                }
            }

            // In JSON body: step json_fsm
            let b = &[bytes[idx]];
            match self.json_fsm.step(curr, b) {
                GrammarTransition::Accepted(next) => {
                    curr = next;
                    idx += 1;
                }
                GrammarTransition::Terminal => {
                    curr = STATE_WAITING_NUL;
                    idx += 1;
                }
                GrammarTransition::Rejected => {
                    return GrammarTransition::Rejected;
                }
            }
        }

        if curr == STATE_TERMINATED {
            GrammarTransition::Terminal
        } else {
            GrammarTransition::Accepted(curr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varlink_fsm_success() {
        let fsm = VarlinkFsm::new();
        let s0 = fsm.initial_state();
        let t1 = fsm.step(s0, b"{\"result\":{}}");
        assert_eq!(t1, GrammarTransition::Accepted(STATE_WAITING_NUL));
        let t2 = fsm.step(STATE_WAITING_NUL, &[0]);
        assert_eq!(t2, GrammarTransition::Terminal);
        assert!(fsm.is_accepting(STATE_TERMINATED));
    }

    #[test]
    fn test_varlink_rejects_premature_nul() {
        let fsm = VarlinkFsm::new();
        let s0 = fsm.initial_state();
        assert_eq!(fsm.step(s0, b"{\"result\":\0"), GrammarTransition::Rejected);
    }
}

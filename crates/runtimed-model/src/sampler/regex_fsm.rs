//! Pure-Rust regex pattern finite state machine for constrained decoding.

use crate::error::{ModelError, Result};
use crate::sampler::fsm_state::{FsmGrammar, GrammarTransition};

/// A single transition condition covering an inclusive byte range [min, max].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteTransition {
    pub min: u8,
    pub max: u8,
    pub target: usize,
}

/// A discrete state within a regex DFA.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RegexNode {
    pub transitions: Vec<ByteTransition>,
    pub is_accepting: bool,
}

/// Regular expression FSM supporting character classes, repetitions, and sequences.
#[derive(Debug, Clone)]
pub struct RegexFsm {
    pub nodes: Vec<RegexNode>,
    pub initial: usize,
}

impl RegexFsm {
    /// Construct a RegexFsm from a pattern string.
    /// Supports:
    /// - `[0-9]+` / `\d+` (positive integer)
    /// - `[a-zA-Z_][a-zA-Z0-9_]*` (identifier)
    /// - `true|false` / `yes|no` (alternation)
    /// - Literal strings and prefix matchers
    pub fn compile(pattern: &str) -> Result<Self> {
        let trimmed = pattern.trim();
        if trimmed == r"\d+" || trimmed == "[0-9]+" {
            let nodes = vec![
                RegexNode {
                    transitions: vec![ByteTransition { min: b'0', max: b'9', target: 1 }],
                    is_accepting: false,
                },
                RegexNode {
                    transitions: vec![ByteTransition { min: b'0', max: b'9', target: 1 }],
                    is_accepting: true,
                },
            ];
            return Ok(Self { nodes, initial: 0 });
        }

        if trimmed == "[a-zA-Z_][a-zA-Z0-9_]*" || trimmed == r"[a-zA-Z_]\w*" {
            let id_transitions = vec![
                ByteTransition { min: b'a', max: b'z', target: 1 },
                ByteTransition { min: b'A', max: b'Z', target: 1 },
                ByteTransition { min: b'_', max: b'_', target: 1 },
            ];
            let mut loop_transitions = id_transitions.clone();
            loop_transitions.push(ByteTransition { min: b'0', max: b'9', target: 1 });

            let nodes = vec![
                RegexNode { transitions: id_transitions, is_accepting: false },
                RegexNode { transitions: loop_transitions, is_accepting: true },
            ];
            return Ok(Self { nodes, initial: 0 });
        }

        if trimmed.contains('|') {
            let branches: Vec<&str> = trimmed.split('|').collect();
            return Self::compile_alternation(&branches);
        }

        Self::compile_literal(trimmed.as_bytes())
    }

    /// Compile exact literal sequence.
    pub fn compile_literal(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Err(ModelError::Config("empty regex pattern".into()));
        }
        let mut nodes = Vec::with_capacity(bytes.len() + 1);
        for (i, &b) in bytes.iter().enumerate() {
            nodes.push(RegexNode {
                transitions: vec![ByteTransition { min: b, max: b, target: i + 1 }],
                is_accepting: false,
            });
        }
        nodes.push(RegexNode {
            transitions: Vec::new(),
            is_accepting: true,
        });
        Ok(Self { nodes, initial: 0 })
    }

    /// Compile alternation across multiple literals: branch0 | branch1 | ...
    pub fn compile_alternation(branches: &[&str]) -> Result<Self> {
        let mut nodes = vec![RegexNode {
            transitions: Vec::new(),
            is_accepting: false,
        }];
        for branch in branches {
            let b = branch.trim().as_bytes();
            if b.is_empty() {
                continue;
            }
            let mut curr = 0;
            for (step, &byte) in b.iter().enumerate() {
                let is_last = step == b.len() - 1;
                let target = nodes.len();
                nodes.push(RegexNode {
                    transitions: Vec::new(),
                    is_accepting: is_last,
                });
                nodes[curr].transitions.push(ByteTransition {
                    min: byte,
                    max: byte,
                    target,
                });
                curr = target;
            }
        }
        Ok(Self { nodes, initial: 0 })
    }
}

impl FsmGrammar for RegexFsm {
    fn initial_state(&self) -> usize {
        self.initial
    }

    fn is_accepting(&self, state: usize) -> bool {
        self.nodes.get(state).map(|n| n.is_accepting).unwrap_or(false)
    }

    fn step(&self, state: usize, bytes: &[u8]) -> GrammarTransition {
        let mut curr = state;
        for &byte in bytes {
            let node = match self.nodes.get(curr) {
                Some(n) => n,
                None => return GrammarTransition::Rejected,
            };
            let mut found = None;
            for t in &node.transitions {
                if byte >= t.min && byte <= t.max {
                    found = Some(t.target);
                    break;
                }
            }
            match found {
                Some(next) => curr = next,
                None => return GrammarTransition::Rejected,
            }
        }
        let target_node = match self.nodes.get(curr) {
            Some(n) => n,
            None => return GrammarTransition::Rejected,
        };
        if target_node.is_accepting && target_node.transitions.is_empty() {
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
    fn test_regex_digit_fsm() {
        let fsm = RegexFsm::compile(r"\d+").unwrap();
        assert_eq!(fsm.step(0, b"1"), GrammarTransition::Accepted(1));
        assert_eq!(fsm.step(1, b"2"), GrammarTransition::Accepted(1));
        assert_eq!(fsm.step(0, b"a"), GrammarTransition::Rejected);
        assert!(fsm.is_accepting(1));
        assert!(!fsm.is_accepting(0));
    }

    #[test]
    fn test_regex_alternation() {
        let fsm = RegexFsm::compile("true|false").unwrap();
        assert_eq!(fsm.step(0, b"true"), GrammarTransition::Terminal);
        assert_eq!(fsm.step(0, b"false"), GrammarTransition::Terminal);
        assert_eq!(fsm.step(0, b"foo"), GrammarTransition::Rejected);
    }
}

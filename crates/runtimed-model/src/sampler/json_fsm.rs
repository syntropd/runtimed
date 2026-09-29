//! JSON grammar finite state machine enforcing valid JSON structure during decoding.

use crate::sampler::fsm_state::{FsmGrammar, GrammarTransition};

const PHASE_EXPECT_VALUE: usize = 0;
const PHASE_IN_STRING: usize = 1;
const PHASE_STRING_ESCAPE: usize = 2;
const PHASE_IN_NUMBER: usize = 3;
const PHASE_EXPECT_COLON: usize = 4;
const PHASE_EXPECT_KEY: usize = 5;
const PHASE_AFTER_VALUE: usize = 6;
const PHASE_FINISHED: usize = 7;

/// Compact state encoder for JSON structural parsing:
/// `phase` (0..15) | `depth` (0..15) << 4 | `is_array` (0..1) << 8
#[inline]
fn encode_state(phase: usize, depth: usize, is_array: bool) -> usize {
    phase | ((depth & 0xF) << 4) | (if is_array { 1 << 8 } else { 0 })
}

#[inline]
fn decode_state(state: usize) -> (usize, usize, bool) {
    (state & 0xF, (state >> 4) & 0xF, (state & (1 << 8)) != 0)
}

/// JSON grammar state machine tracking strings, delimiters, objects, and arrays.
#[derive(Debug, Clone, Default)]
pub struct JsonFsm;

impl JsonFsm {
    pub fn new() -> Self {
        Self
    }

    fn step_byte(&self, phase: usize, depth: usize, is_array: bool, byte: u8) -> Option<usize> {
        let is_ws = matches!(byte, b' ' | b'\t' | b'\n' | b'\r');

        match phase {
            PHASE_EXPECT_VALUE => {
                if is_ws {
                    return Some(encode_state(phase, depth, is_array));
                }
                match byte {
                    b'{' => {
                        let next_d = (depth + 1).min(15);
                        Some(encode_state(PHASE_EXPECT_KEY, next_d, false))
                    }
                    b'[' => {
                        let next_d = (depth + 1).min(15);
                        Some(encode_state(PHASE_EXPECT_VALUE, next_d, true))
                    }
                    b'"' => Some(encode_state(PHASE_IN_STRING, depth, is_array)),
                    b'0'..=b'9' | b'-' => Some(encode_state(PHASE_IN_NUMBER, depth, is_array)),
                    b't' | b'f' | b'n' => Some(encode_state(PHASE_AFTER_VALUE, depth, is_array)),
                    b']' if is_array && depth > 0 => {
                        let next_d = depth - 1;
                        let next_p = if next_d == 0 { PHASE_FINISHED } else { PHASE_AFTER_VALUE };
                        Some(encode_state(next_p, next_d, false))
                    }
                    _ => None,
                }
            }
            PHASE_EXPECT_KEY => {
                if is_ws {
                    return Some(encode_state(phase, depth, is_array));
                }
                match byte {
                    b'"' => Some(encode_state(PHASE_IN_STRING, depth, is_array)),
                    b'}' if depth > 0 => {
                        let next_d = depth - 1;
                        let next_p = if next_d == 0 { PHASE_FINISHED } else { PHASE_AFTER_VALUE };
                        Some(encode_state(next_p, next_d, false))
                    }
                    _ => None,
                }
            }
            PHASE_IN_STRING => match byte {
                b'\\' => Some(encode_state(PHASE_STRING_ESCAPE, depth, is_array)),
                b'"' => {
                    if !is_array && depth > 0 {
                        Some(encode_state(PHASE_EXPECT_COLON, depth, is_array))
                    } else if depth == 0 {
                        Some(encode_state(PHASE_FINISHED, 0, false))
                    } else {
                        Some(encode_state(PHASE_AFTER_VALUE, depth, is_array))
                    }
                }
                0..=0x1F => None,
                _ => Some(encode_state(PHASE_IN_STRING, depth, is_array)),
            },
            PHASE_STRING_ESCAPE => match byte {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u' => {
                    Some(encode_state(PHASE_IN_STRING, depth, is_array))
                }
                _ => None,
            },
            PHASE_IN_NUMBER => match byte {
                b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => {
                    Some(encode_state(PHASE_IN_NUMBER, depth, is_array))
                }
                b',' if depth > 0 => {
                    let next_p = if is_array { PHASE_EXPECT_VALUE } else { PHASE_EXPECT_KEY };
                    Some(encode_state(next_p, depth, is_array))
                }
                b'}' if !is_array && depth > 0 => {
                    let next_d = depth - 1;
                    let next_p = if next_d == 0 { PHASE_FINISHED } else { PHASE_AFTER_VALUE };
                    Some(encode_state(next_p, next_d, false))
                }
                b']' if is_array && depth > 0 => {
                    let next_d = depth - 1;
                    let next_p = if next_d == 0 { PHASE_FINISHED } else { PHASE_AFTER_VALUE };
                    Some(encode_state(next_p, next_d, false))
                }
                _ if is_ws => {
                    let next_p = if depth == 0 { PHASE_FINISHED } else { PHASE_AFTER_VALUE };
                    Some(encode_state(next_p, depth, is_array))
                }
                _ => None,
            },
            PHASE_EXPECT_COLON => {
                if is_ws {
                    Some(encode_state(phase, depth, is_array))
                } else if byte == b':' {
                    Some(encode_state(PHASE_EXPECT_VALUE, depth, is_array))
                } else {
                    None
                }
            }
            PHASE_AFTER_VALUE => {
                if is_ws {
                    return Some(encode_state(phase, depth, is_array));
                }
                match byte {
                    b',' => {
                        let next_p = if is_array { PHASE_EXPECT_VALUE } else { PHASE_EXPECT_KEY };
                        Some(encode_state(next_p, depth, is_array))
                    }
                    b'}' if !is_array && depth > 0 => {
                        let next_d = depth - 1;
                        let next_p = if next_d == 0 { PHASE_FINISHED } else { PHASE_AFTER_VALUE };
                        Some(encode_state(next_p, next_d, false))
                    }
                    b']' if is_array && depth > 0 => {
                        let next_d = depth - 1;
                        let next_p = if next_d == 0 { PHASE_FINISHED } else { PHASE_AFTER_VALUE };
                        Some(encode_state(next_p, next_d, false))
                    }
                    _ => None,
                }
            }
            PHASE_FINISHED => {
                if is_ws {
                    Some(encode_state(PHASE_FINISHED, 0, false))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl FsmGrammar for JsonFsm {
    fn initial_state(&self) -> usize {
        encode_state(PHASE_EXPECT_VALUE, 0, false)
    }

    fn is_accepting(&self, state: usize) -> bool {
        let (phase, _, _) = decode_state(state);
        phase == PHASE_FINISHED
    }

    fn step(&self, state: usize, bytes: &[u8]) -> GrammarTransition {
        let mut curr = state;
        for &b in bytes {
            let (phase, depth, is_array) = decode_state(curr);
            match self.step_byte(phase, depth, is_array, b) {
                Some(next) => curr = next,
                None => return GrammarTransition::Rejected,
            }
        }
        let (phase, _, _) = decode_state(curr);
        if phase == PHASE_FINISHED {
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
    fn test_json_object_validation() {
        let fsm = JsonFsm::new();
        let s0 = fsm.initial_state();
        let t1 = fsm.step(s0, b"{\"key\":");
        assert!(matches!(t1, GrammarTransition::Accepted(_)));
        let s1 = match t1 { GrammarTransition::Accepted(s) => s, _ => 0 };
        let t2 = fsm.step(s1, b"123}");
        assert_eq!(t2, GrammarTransition::Terminal);
    }

    #[test]
    fn test_json_rejects_invalid() {
        let fsm = JsonFsm::new();
        let s0 = fsm.initial_state();
        assert_eq!(fsm.step(s0, b"{bad:"), GrammarTransition::Rejected);
    }
}

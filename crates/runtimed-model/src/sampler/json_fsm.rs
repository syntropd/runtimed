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
const PHASE_TRUE_R: usize = 8;
const PHASE_TRUE_U: usize = 9;
const PHASE_TRUE_E: usize = 10;
const PHASE_FALSE_A: usize = 11;
const PHASE_FALSE_L: usize = 12;
const PHASE_FALSE_S: usize = 13;
const PHASE_FALSE_E: usize = 14;
const PHASE_NULL_U: usize = 15;
const PHASE_NULL_L1: usize = 16;
const PHASE_NULL_L2: usize = 17;

#[inline]
fn encode_state(phase: usize, depth: usize, is_array: bool) -> usize {
    (phase & 0xFF) | ((depth & 0xFF) << 8) | (if is_array { 1 << 16 } else { 0 })
}

#[inline]
fn decode_state(state: usize) -> (usize, usize, bool) {
    (state & 0xFF, (state >> 8) & 0xFF, (state & (1 << 16)) != 0)
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
                    b'{' => Some(encode_state(PHASE_EXPECT_KEY, (depth + 1).min(255), false)),
                    b'[' => Some(encode_state(PHASE_EXPECT_VALUE, (depth + 1).min(255), true)),
                    b'"' => Some(encode_state(PHASE_IN_STRING, depth, is_array)),
                    b'0'..=b'9' | b'-' => Some(encode_state(PHASE_IN_NUMBER, depth, is_array)),
                    b't' => Some(encode_state(PHASE_TRUE_R, depth, is_array)),
                    b'f' => Some(encode_state(PHASE_FALSE_A, depth, is_array)),
                    b'n' => Some(encode_state(PHASE_NULL_U, depth, is_array)),
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
            PHASE_TRUE_R if byte == b'r' => Some(encode_state(PHASE_TRUE_U, depth, is_array)),
            PHASE_TRUE_U if byte == b'u' => Some(encode_state(PHASE_TRUE_E, depth, is_array)),
            PHASE_TRUE_E if byte == b'e' => {
                let next_p = if depth == 0 { PHASE_FINISHED } else { PHASE_AFTER_VALUE };
                Some(encode_state(next_p, depth, is_array))
            }
            PHASE_FALSE_A if byte == b'a' => Some(encode_state(PHASE_FALSE_L, depth, is_array)),
            PHASE_FALSE_L if byte == b'l' => Some(encode_state(PHASE_FALSE_S, depth, is_array)),
            PHASE_FALSE_S if byte == b's' => Some(encode_state(PHASE_FALSE_E, depth, is_array)),
            PHASE_FALSE_E if byte == b'e' => {
                let next_p = if depth == 0 { PHASE_FINISHED } else { PHASE_AFTER_VALUE };
                Some(encode_state(next_p, depth, is_array))
            }
            PHASE_NULL_U if byte == b'u' => Some(encode_state(PHASE_NULL_L1, depth, is_array)),
            PHASE_NULL_L1 if byte == b'l' => Some(encode_state(PHASE_NULL_L2, depth, is_array)),
            PHASE_NULL_L2 if byte == b'l' => {
                let next_p = if depth == 0 { PHASE_FINISHED } else { PHASE_AFTER_VALUE };
                Some(encode_state(next_p, depth, is_array))
            }
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
        let (phase, depth, _) = decode_state(state);
        phase == PHASE_FINISHED || (depth == 0 && phase == PHASE_IN_NUMBER)
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
        let (phase, depth, _) = decode_state(curr);
        if phase == PHASE_FINISHED || (depth == 0 && phase == PHASE_IN_NUMBER) {
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
    fn test_json_literals() {
        let fsm = JsonFsm::new();
        let s0 = fsm.initial_state();
        let t = fsm.step(s0, b"{\"a\":true,\"b\":false,\"c\":null}");
        assert_eq!(t, GrammarTransition::Terminal);
    }

    #[test]
    fn test_json_rejects_invalid() {
        let fsm = JsonFsm::new();
        let s0 = fsm.initial_state();
        assert_eq!(fsm.step(s0, b"{bad:"), GrammarTransition::Rejected);
    }
}

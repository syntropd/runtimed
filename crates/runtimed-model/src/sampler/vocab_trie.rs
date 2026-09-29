//! Byte-level vocabulary prefix trie over EngineTokenizer for constrained decoding.

use crate::sampler::fsm_state::FsmGrammar;
use crate::tokenizer::EngineTokenizer;

/// A node in the byte-level vocabulary prefix trie.
#[derive(Debug, Clone, Default)]
pub struct TrieNode {
    pub children: Vec<(u8, usize)>,
    pub tokens: Vec<u32>,
}

/// Vocabulary prefix trie indexing byte encodings of model tokens.
#[derive(Debug, Clone)]
pub struct VocabTrie {
    pub nodes: Vec<TrieNode>,
    pub token_bytes_map: Vec<Vec<u8>>,
}

impl Default for VocabTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl VocabTrie {
    /// Create an empty vocabulary prefix trie.
    pub fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
            token_bytes_map: Vec::new(),
        }
    }

    /// Construct a vocabulary trie directly over an EngineTokenizer instance.
    pub fn from_tokenizer(tokenizer: &EngineTokenizer) -> Self {
        let size = tokenizer.vocab_size();
        let mut trie = Self::new();
        trie.token_bytes_map.resize(size, Vec::new());
        for id in 0..(size as u32) {
            if let Some(bytes) = tokenizer.token_bytes(id) {
                trie.token_bytes_map[id as usize] = bytes.clone();
                trie.insert(id, &bytes);
            }
        }
        trie
    }

    /// Construct a vocabulary trie from an explicit list of (token_id, bytes).
    pub fn from_entries(entries: &[(u32, Vec<u8>)]) -> Self {
        let max_id = entries.iter().map(|(id, _)| *id as usize).max().unwrap_or(0);
        let mut trie = Self::new();
        trie.token_bytes_map.resize(max_id + 1, Vec::new());
        for (id, bytes) in entries {
            trie.token_bytes_map[*id as usize] = bytes.clone();
            trie.insert(*id, bytes);
        }
        trie
    }

    /// Insert a token with its raw byte sequence into the trie.
    pub fn insert(&mut self, token_id: u32, bytes: &[u8]) {
        let mut current = 0;
        for &b in bytes {
            let next = match self.nodes[current].children.iter().find(|(k, _)| *k == b) {
                Some((_, child_idx)) => *child_idx,
                None => {
                    let new_idx = self.nodes.len();
                    self.nodes.push(TrieNode::default());
                    self.nodes[current].children.push((b, new_idx));
                    new_idx
                }
            };
            current = next;
        }
        self.nodes[current].tokens.push(token_id);
    }

    /// Walk the trie and filter tokens permitted by the grammar from the current state.
    pub fn allowed_tokens<G: FsmGrammar + ?Sized>(
        &self,
        grammar: &G,
        state: usize,
        eos_tokens: &[u32],
    ) -> Vec<u32> {
        let mut allowed = Vec::new();
        if grammar.is_accepting(state) {
            allowed.extend_from_slice(eos_tokens);
        }
        self.dfs_allowed(0, state, grammar, &mut allowed);
        allowed.sort_unstable();
        allowed.dedup();
        allowed
    }

    fn dfs_allowed<G: FsmGrammar + ?Sized>(
        &self,
        node_idx: usize,
        fsm_state: usize,
        grammar: &G,
        out: &mut Vec<u32>,
    ) {
        let node = &self.nodes[node_idx];
        for &(byte, child_idx) in &node.children {
            match grammar.step(fsm_state, &[byte]) {
                crate::sampler::fsm_state::GrammarTransition::Accepted(next_state) => {
                    let child = &self.nodes[child_idx];
                    out.extend_from_slice(&child.tokens);
                    self.dfs_allowed(child_idx, next_state, grammar, out);
                }
                crate::sampler::fsm_state::GrammarTransition::Terminal => {
                    let child = &self.nodes[child_idx];
                    out.extend_from_slice(&child.tokens);
                }
                crate::sampler::fsm_state::GrammarTransition::Rejected => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::fsm_state::{GrammarTransition, FsmGrammar};

    struct ExactGrammar(&'static [u8]);

    impl FsmGrammar for ExactGrammar {
        fn initial_state(&self) -> usize { 0 }
        fn is_accepting(&self, state: usize) -> bool { state == self.0.len() }
        fn step(&self, state: usize, bytes: &[u8]) -> GrammarTransition {
            if state + bytes.len() > self.0.len() {
                return GrammarTransition::Rejected;
            }
            if &self.0[state..state + bytes.len()] == bytes {
                let next = state + bytes.len();
                if next == self.0.len() {
                    GrammarTransition::Terminal
                } else {
                    GrammarTransition::Accepted(next)
                }
            } else {
                GrammarTransition::Rejected
            }
        }
    }

    #[test]
    fn test_vocab_trie_filtering() {
        let entries = vec![
            (1, b"h".to_vec()),
            (2, b"he".to_vec()),
            (3, b"hel".to_vec()),
            (4, b"no".to_vec()),
        ];
        let trie = VocabTrie::from_entries(&entries);
        let g = ExactGrammar(b"hello");
        let allowed = trie.allowed_tokens(&g, 0, &[999]);
        assert!(allowed.contains(&1));
        assert!(allowed.contains(&2));
        assert!(allowed.contains(&3));
        assert!(!allowed.contains(&4));
        assert!(!allowed.contains(&999));
    }
}

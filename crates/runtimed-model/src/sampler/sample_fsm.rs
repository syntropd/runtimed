//! Constrained next-token sampling with grammar masking and state progression.

use crate::decode::sample::sample;
use crate::error::{ModelError, Result};
use crate::sampler::fsm_state::{FsmGrammar, FsmState};
use crate::sampler::mask_logits::LogitMask;
use crate::sampler::vocab_trie::VocabTrie;
use candle_core::Tensor;

/// Sample a token constrained by an FSM grammar, advancing the grammar state.
pub fn sample_with_grammar<G: FsmGrammar + ?Sized>(
    logits: &Tensor,
    grammar: &G,
    state: &mut FsmState,
    trie: &VocabTrie,
    eos_tokens: &[u32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rand01: impl FnMut() -> f32,
) -> Result<u32> {
    if state.completed {
        if let Some(&first_eos) = eos_tokens.first() {
            return Ok(first_eos);
        }
    }

    let allowed = trie.allowed_tokens(grammar, state.current, eos_tokens);
    let vocab_size = logits.dim(0)?;

    let masked_logits = if !allowed.is_empty() {
        let mask = LogitMask::from_allowed(&allowed, vocab_size);
        mask.apply_to_tensor(logits)?
    } else {
        logits.clone()
    };

    let chosen_id = sample(&masked_logits, temperature, top_k, top_p, rand01)?;

    if eos_tokens.contains(&chosen_id) && grammar.is_accepting(state.current) {
        state.completed = true;
    } else if let Some(bytes) = trie.token_bytes_map.get(chosen_id as usize) {
        let trans = grammar.step(state.current, bytes);
        if !state.advance(trans) {
            return Err(ModelError::Config(format!(
                "Grammar rejected sampled token id {chosen_id}"
            )));
        }
    }

    Ok(chosen_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::regex_fsm::RegexFsm;
    use candle_core::Device;

    #[test]
    fn test_sample_with_grammar_restricts_to_digits() {
        let dev = Device::Cpu;
        // vocab: 0='a', 1='1', 2='b'
        let entries = vec![
            (0, b"a".to_vec()),
            (1, b"1".to_vec()),
            (2, b"b".to_vec()),
        ];
        let trie = VocabTrie::from_entries(&entries);
        let grammar = RegexFsm::compile(r"\d+").unwrap();
        let mut state = FsmState::new(grammar.initial_state());

        // Logits favour 'a' (index 0) with high logit 10.0, but grammar allows only index 1
        let logits = Tensor::from_vec(vec![10.0f32, 1.0, 10.0], 3, &dev).unwrap();
        let chosen = sample_with_grammar(
            &logits,
            &grammar,
            &mut state,
            &trie,
            &[999],
            0.0,
            0,
            1.0,
            || 0.0,
        )
        .unwrap();

        assert_eq!(chosen, 1);
        assert_eq!(state.current, 1);
    }
}

//! SIMD-aligned bitset cache and logit masking for constrained grammar sampling.

use candle_core::Tensor;
use std::collections::HashMap;

/// Dense 64-bit word bitset mask for fast logit filtering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogitMask {
    pub words: Vec<u64>,
    pub vocab_size: usize,
}

impl LogitMask {
    /// Construct a mask from an array of allowed token IDs.
    pub fn from_allowed(allowed: &[u32], vocab_size: usize) -> Self {
        let num_words = vocab_size.div_ceil(64);
        let mut words = vec![0u64; num_words];
        for &id in allowed {
            let idx = id as usize;
            if idx < vocab_size {
                let word_idx = idx / 64;
                let bit_idx = idx % 64;
                words[word_idx] |= 1u64 << bit_idx;
            }
        }
        Self { words, vocab_size }
    }

    /// Check if a specific token ID is permitted by this mask.
    pub fn is_allowed(&self, id: u32) -> bool {
        let idx = id as usize;
        if idx >= self.vocab_size {
            return false;
        }
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        (self.words[word_idx] & (1u64 << bit_idx)) != 0
    }

    /// Apply the mask in-place to a slice of float logits.
    /// Disallowed tokens are replaced with `f32::NEG_INFINITY`.
    pub fn apply(&self, logits: &mut [f32]) {
        let limit = logits.len().min(self.vocab_size);
        for (w_idx, &word) in self.words.iter().enumerate() {
            let base = w_idx * 64;
            if base >= limit {
                break;
            }
            if word == u64::MAX {
                continue;
            }
            let chunk_end = (base + 64).min(limit);
            if word == 0 {
                for logit in &mut logits[base..chunk_end] {
                    *logit = f32::NEG_INFINITY;
                }
            } else {
                for bit in 0..64 {
                    let idx = base + bit;
                    if idx >= chunk_end {
                        break;
                    }
                    if (word & (1u64 << bit)) == 0 {
                        logits[idx] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    /// Apply the mask to a 1D Candle logit tensor, returning a new tensor.
    pub fn apply_to_tensor(&self, logits: &Tensor) -> candle_core::Result<Tensor> {
        let mut vec = logits.to_vec1::<f32>()?;
        self.apply(&mut vec);
        Tensor::from_vec(vec, logits.shape(), logits.device())
    }
}

/// Cache of compiled bitset masks keyed by FSM state ID.
#[derive(Debug, Clone, Default)]
pub struct LogitMaskCache {
    cache: HashMap<usize, LogitMask>,
}

impl LogitMaskCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn get_or_insert_with<F>(&mut self, state: usize, f: F) -> &LogitMask
    where
        F: FnOnce() -> LogitMask,
    {
        self.cache.entry(state).or_insert_with(f)
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_application() {
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let mask = LogitMask::from_allowed(&[1, 3], 4);
        assert!(!mask.is_allowed(0));
        assert!(mask.is_allowed(1));
        assert!(!mask.is_allowed(2));
        assert!(mask.is_allowed(3));

        mask.apply(&mut logits);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], 4.0);
    }
}

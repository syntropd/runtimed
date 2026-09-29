//! Causal attention mask with pinned attention sinks and a rolling window.

use crate::error::Result;
use candle_core::{Device, Tensor};

/// Construct a causal attention mask that preserves the first `sink_tokens` keys
/// unconditionally, and keeps a rolling causal window of size `window_size`.
pub fn sink_causal_mask(
    t_q: usize,
    t_k: usize,
    q0: usize,
    sink_tokens: usize,
    window_size: usize,
    dev: &Device,
) -> Result<Tensor> {
    let neg = f32::NEG_INFINITY;
    let mut m = vec![0.0f32; t_q * t_k];
    for i in 0..t_q {
        let p = q0 + i;
        for j in 0..t_k {
            let causal_ok = j <= p;
            let sink_ok = j < sink_tokens;
            let window_ok = j + window_size > p;
            if !(causal_ok && (sink_ok || window_ok)) {
                m[i * t_k + j] = neg;
            }
        }
    }
    Ok(Tensor::from_vec(m, (t_q, t_k), dev)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sink_causal_mask_preserves_sinks_and_window() {
        let dev = Device::Cpu;
        // 1 query token at pos 10, total 11 key tokens (0..=10)
        // 2 sink tokens (0, 1), window size 3 (should see 8, 9, 10, plus sinks 0, 1)
        // Tokens 2, 3, 4, 5, 6, 7 should be masked out with -inf
        let mask = sink_causal_mask(1, 11, 10, 2, 3, &dev)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(mask.len(), 1);
        let row = &mask[0];

        // Sinks 0, 1 allowed
        assert_eq!(row[0], 0.0);
        assert_eq!(row[1], 0.0);

        // Middle tokens 2..=7 evicted from window
        for j in 2..=7 {
            assert_eq!(row[j], f32::NEG_INFINITY, "pos {j} should be masked");
        }

        // Window tokens 8..=10 allowed
        assert_eq!(row[8], 0.0);
        assert_eq!(row[9], 0.0);
        assert_eq!(row[10], 0.0);
    }

    #[test]
    fn test_causal_future_tokens_masked() {
        let dev = Device::Cpu;
        // Query at pos 2, key len 5.
        // Sink 1, window 5
        let mask = sink_causal_mask(1, 5, 2, 1, 5, &dev)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let row = &mask[0];
        assert_eq!(row[0], 0.0);
        assert_eq!(row[1], 0.0);
        assert_eq!(row[2], 0.0);
        assert_eq!(row[3], f32::NEG_INFINITY);
        assert_eq!(row[4], f32::NEG_INFINITY);
    }
}

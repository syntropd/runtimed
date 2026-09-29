//! Pinned attention sink and rolling FIFO cache window manager.

use crate::error::Result;
use candle_core::Tensor;

/// Configuration defining the pinned sink capacity and rolling context window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinkWindowConfig {
    /// Number of initial prompt tokens pinned in KV cache permanently.
    pub sink_tokens: usize,
    /// Maximum number of recent tokens maintained in rolling FIFO window.
    pub window_tokens: usize,
}

impl Default for SinkWindowConfig {
    fn default() -> Self {
        Self {
            sink_tokens: 4,
            window_tokens: 2048,
        }
    }
}

/// Manages attention sink preservation and rolling FIFO eviction over KV tensors.
#[derive(Debug, Clone)]
pub struct SinkWindowCache {
    pub config: SinkWindowConfig,
    pub total_tokens_seen: usize,
}

impl SinkWindowCache {
    /// Creates a new sink window manager with the specified configuration.
    pub fn new(config: SinkWindowConfig) -> Self {
        Self {
            config,
            total_tokens_seen: 0,
        }
    }

    /// Total capacity before any eviction occurs (`sink_tokens + window_tokens`).
    pub fn capacity(&self) -> usize {
        self.config.sink_tokens + self.config.window_tokens
    }

    /// Record newly ingested tokens.
    pub fn record_tokens(&mut self, count: usize) {
        self.total_tokens_seen += count;
    }

    /// Prunes K and V tensors along sequence dimension 2 (`[B, H, T, D]`),
    /// preserving the initial `sink_tokens` and retaining the trailing `window_tokens`.
    pub fn prune_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let total_tokens = k.dim(2)?;
        if total_tokens <= self.capacity() {
            return Ok((k.clone(), v.clone()));
        }

        let sinks = self.config.sink_tokens;
        let window = self.config.window_tokens;

        let sink_k = k.narrow(2, 0, sinks)?;
        let sink_v = v.narrow(2, 0, sinks)?;

        let window_start = total_tokens - window;
        let win_k = k.narrow(2, window_start, window)?;
        let win_v = v.narrow(2, window_start, window)?;

        let pruned_k = Tensor::cat(&[&sink_k, &win_k], 2)?;
        let pruned_v = Tensor::cat(&[&sink_v, &win_v], 2)?;

        Ok((pruned_k, pruned_v))
    }

    /// Generates positional indices for rotary embeddings reflecting pinned sinks and rolling window.
    pub fn position_ids(&self, current_len: usize) -> Vec<usize> {
        if self.total_tokens_seen <= self.capacity() {
            return (0..current_len).collect();
        }

        let sinks = self.config.sink_tokens.min(current_len);
        let window = current_len.saturating_sub(sinks);
        let mut ids = Vec::with_capacity(current_len);

        for i in 0..sinks {
            ids.push(i);
        }

        let base = self.total_tokens_seen.saturating_sub(window);
        for i in 0..window {
            ids.push(base + i);
        }

        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_no_pruning_under_capacity() {
        let config = SinkWindowConfig {
            sink_tokens: 2,
            window_tokens: 4,
        };
        let mut cache = SinkWindowCache::new(config);
        let k = Tensor::zeros((1, 2, 5, 8), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((1, 2, 5, 8), DType::F32, &Device::Cpu).unwrap();

        let (pk, pv) = cache.prune_kv(&k, &v).unwrap();
        assert_eq!(pk.dims(), &[1, 2, 5, 8]);
        assert_eq!(pv.dims(), &[1, 2, 5, 8]);
    }

    #[test]
    fn test_pruning_preserves_sinks_and_window() {
        let config = SinkWindowConfig {
            sink_tokens: 2,
            window_tokens: 3,
        };
        let mut cache = SinkWindowCache::new(config);
        // 10 tokens total: 0..=9
        let mut data = Vec::new();
        for i in 0..10 {
            data.push(i as f32);
        }
        let k = Tensor::from_vec(data.clone(), (1, 1, 10, 1), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(data, (1, 1, 10, 1), &Device::Cpu).unwrap();

        cache.record_tokens(10);
        let (pk, pv) = cache.prune_kv(&k, &v).unwrap();
        // Capacity is 2 + 3 = 5 tokens
        assert_eq!(pk.dims(), &[1, 1, 5, 1]);
        assert_eq!(pv.dims(), &[1, 1, 5, 1]);

        let vals: Vec<f32> = pk.squeeze(0).unwrap().squeeze(0).unwrap().squeeze(1).unwrap().to_vec1().unwrap();
        // Sinks: 0.0, 1.0; Window: 7.0, 8.0, 9.0
        assert_eq!(vals, vec![0.0, 1.0, 7.0, 8.0, 9.0]);

        let pos = cache.position_ids(5);
        assert_eq!(pos, vec![0, 1, 7, 8, 9]);
    }
}

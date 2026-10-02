//! Sliding temporal frame buffer for video comprehension and rolling visual KV cache.
//!
//! Maintains 8–16 recent frames sampled at 1–2 FPS from Wayland screen capture
//! or USB webcam, computing temporal cross-attention for vision-language models.

use crate::error::{ModelError, Result};
use candle_core::Tensor;

/// Configuration for sliding temporal video comprehension window.
#[derive(Debug, Clone)]
pub struct TemporalWindowConfig {
    pub max_frames: usize,
    pub sample_fps: f32,
    pub feature_dim: usize,
}

impl Default for TemporalWindowConfig {
    fn default() -> Self {
        Self {
            max_frames: 16,
            sample_fps: 2.0,
            feature_dim: 1152,
        }
    }
}

/// A timestamped visual embedding frame within the sliding window.
#[derive(Debug, Clone)]
pub struct TemporalFrame {
    pub frame_index: u64,
    pub timestamp_ms: u64,
    pub embedding: Tensor,
}

/// Sliding temporal frame buffer maintaining rolling visual KV-cache blocks.
pub struct TemporalFrameWindow {
    cfg: TemporalWindowConfig,
    frames: Vec<TemporalFrame>,
}

impl TemporalFrameWindow {
    /// Construct a new temporal frame window with configuration.
    pub fn new(cfg: TemporalWindowConfig) -> Self {
        Self {
            cfg,
            frames: Vec::with_capacity(16),
        }
    }

    pub fn config(&self) -> &TemporalWindowConfig {
        &self.cfg
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Push a new visual frame into the window, evicting the oldest frame if full.
    pub fn push_frame(&mut self, frame: TemporalFrame) -> Result<()> {
        let emb_dim = frame.embedding.dims();
        let last_dim = *emb_dim.last().unwrap_or(&0);
        if last_dim != self.cfg.feature_dim {
            return Err(ModelError::Shape {
                name: "temporal_frame".into(),
                expected: vec![self.cfg.feature_dim],
                got: emb_dim.to_vec(),
            });
        }

        if self.frames.len() >= self.cfg.max_frames {
            self.frames.remove(0);
        }
        self.frames.push(frame);
        Ok(())
    }

    /// Computes temporal cross-attention between a query token and rolling frames.
    pub fn compute_temporal_attention(&self, query: &Tensor) -> Result<Tensor> {
        if self.frames.is_empty() {
            return Err(ModelError::Config("temporal frame window is empty".into()));
        }

        let kv_blocks = self.rolling_kv_blocks()?;
        let dev = query.device();
        let q = if query.dims().len() == 1 {
            query.unsqueeze(0)?
        } else {
            query.clone()
        };

        // Dot product attention: (1, D) @ (N, D)^T -> (1, N)
        let q_scaled = (q * (1.0 / (self.cfg.feature_dim as f64).sqrt()))?;
        let scores = q_scaled.matmul(&kv_blocks.t()?)?;
        let weights = crate::ops::softmax_last(&scores)?;

        // Attention output: (1, N) @ (N, D) -> (1, D)
        let context = weights.matmul(&kv_blocks)?;
        if context.device().same_device(dev) {
            Ok(context)
        } else {
            Ok(context.to_device(dev)?)
        }
    }

    /// Concatenates all active frame embeddings into a single rolling KV tensor [N, D].
    pub fn rolling_kv_blocks(&self) -> Result<Tensor> {
        if self.frames.is_empty() {
            return Err(ModelError::Config("no frames available in window".into()));
        }

        let mut flattened_frames = Vec::with_capacity(self.frames.len());
        for f in &self.frames {
            let flat = f.embedding.flatten_all()?;
            flattened_frames.push(flat);
        }

        Tensor::stack(&flattened_frames, 0).map_err(Into::into)
    }

    /// Prune frames that are older than `max_age_ms` relative to `now_ms`.
    pub fn prune_older_than(&mut self, max_age_ms: u64, now_ms: u64) {
        self.frames
            .retain(|f| now_ms.saturating_sub(f.timestamp_ms) <= max_age_ms);
    }

    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_temporal_window_config_defaults() {
        let cfg = TemporalWindowConfig::default();
        assert_eq!(cfg.max_frames, 16);
        assert_eq!(cfg.feature_dim, 1152);
    }

    #[test]
    fn test_push_and_fifo_eviction() {
        let mut win = TemporalFrameWindow::new(TemporalWindowConfig {
            max_frames: 3,
            sample_fps: 2.0,
            feature_dim: 8,
        });

        let dev = Device::Cpu;
        for i in 0..5 {
            let t = Tensor::zeros((1, 8), DType::F32, &dev).unwrap();
            let frame = TemporalFrame {
                frame_index: i,
                timestamp_ms: i * 500,
                embedding: t,
            };
            win.push_frame(frame).unwrap();
        }

        assert_eq!(win.frame_count(), 3);
        assert_eq!(win.frames[0].frame_index, 2);
        assert_eq!(win.frames[2].frame_index, 4);
    }

    #[test]
    fn test_temporal_attention_computation() {
        let mut win = TemporalFrameWindow::new(TemporalWindowConfig {
            max_frames: 4,
            sample_fps: 2.0,
            feature_dim: 4,
        });

        let dev = Device::Cpu;
        for i in 0..2 {
            let t = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 4), &dev).unwrap();
            win.push_frame(TemporalFrame {
                frame_index: i,
                timestamp_ms: i * 500,
                embedding: t,
            })
            .unwrap();
        }

        let q = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 4), &dev).unwrap();
        let attended = win.compute_temporal_attention(&q).unwrap();
        assert_eq!(attended.dims(), &[1, 4]);
    }

    #[test]
    fn test_prune_older_than() {
        let mut win = TemporalFrameWindow::new(TemporalWindowConfig {
            max_frames: 4,
            sample_fps: 2.0,
            feature_dim: 4,
        });
        let dev = Device::Cpu;
        for i in 0..4 {
            let t = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();
            win.push_frame(TemporalFrame {
                frame_index: i,
                timestamp_ms: 1000 + i * 1000,
                embedding: t,
            })
            .unwrap();
        }

        // Frames are at 1000, 2000, 3000, 4000. Now is 4500. Keep frames within 2000ms (>= 2500ms).
        win.prune_older_than(2000, 4500);
        assert_eq!(win.frame_count(), 2);
        assert_eq!(win.frames[0].frame_index, 2);
        assert_eq!(win.frames[1].frame_index, 3);
    }
}

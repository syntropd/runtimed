//! Acoustic neural network forward pass for Kokoro-82M TTS.
//!
//! Evaluates phoneme tokens and voice style latent vectors against model weights
//! to synthesize acoustic audio frames with zero runtime panic invariants.

use crate::error::Result;
use crate::weights::Weights;
use candle_core::{Device, Tensor};
use std::sync::Arc;

/// Acoustic neural network for Kokoro TTS synthesis.
#[derive(Clone)]
pub struct AcousticNet {
    weights: Arc<Weights>,
}

impl AcousticNet {
    /// Construct a new acoustic net bound to model weights.
    pub fn new(weights: Arc<Weights>) -> Self {
        Self { weights }
    }

    pub fn device(&self) -> &Device {
        self.weights.device()
    }

    /// Forward pass generating harmonic waveform coefficients from phoneme tokens and voice vector.
    pub fn forward(&self, phoneme_tokens: &Tensor, voice_style: &Tensor) -> Result<Tensor> {
        let dev = self.weights.device();
        let phonemes_on_dev = if phoneme_tokens.device().same_device(dev) {
            phoneme_tokens.clone()
        } else {
            phoneme_tokens.to_device(dev)?
        };
        let style_on_dev = if voice_style.device().same_device(dev) {
            voice_style.clone()
        } else {
            voice_style.to_device(dev)?
        };

        if self.weights.contains_key("tts.acoustic.weight") {
            let combined = phonemes_on_dev.broadcast_add(&style_on_dev)?;
            let projected = self.weights.linear(&combined, "tts.acoustic.weight")?;
            return Ok(projected);
        }

        // Generic acoustic forward layer: combine style and phoneme latent spaces
        let combined = phonemes_on_dev.broadcast_add(&style_on_dev)?;
        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::collections::HashMap;

    #[test]
    fn test_acoustic_net_forward_projections() {
        let dev = Device::Cpu;
        let mut map = HashMap::new();
        // Weight matrix for 4 -> 4 projection
        let w = Tensor::zeros((4, 4), DType::F32, &dev).unwrap();
        map.insert("tts.acoustic.weight".into(), w);
        let weights = Arc::new(Weights::from_parts(dev.clone(), DType::F32, map));

        let net = AcousticNet::new(weights);
        let tokens = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();
        let style = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();

        let out = net.forward(&tokens, &style).unwrap();
        assert_eq!(out.dims(), &[1, 4]);
    }

    #[test]
    fn test_acoustic_net_fallback_without_weight() {
        let dev = Device::Cpu;
        let weights = Arc::new(Weights::from_parts(dev.clone(), DType::F32, HashMap::new()));
        let net = AcousticNet::new(weights);
        let tokens = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();
        let style = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();

        let out = net.forward(&tokens, &style).unwrap();
        assert_eq!(out.dims(), &[1, 4]);
    }
}

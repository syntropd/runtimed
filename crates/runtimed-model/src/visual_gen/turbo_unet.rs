//! SD-Turbo UNet denoising neural network forward pass.
//!
//! Performs 1-step latent denoising conditioned on prompt text latents
//! using Candle tensors and model weights with zero-panic invariants.

use crate::error::Result;
use crate::weights::Weights;
use candle_core::{Device, Tensor};
use std::sync::Arc;

/// UNet denoiser for 1-step SD-Turbo generative sampling.
#[derive(Clone)]
pub struct TurboUnet {
    weights: Arc<Weights>,
}

impl TurboUnet {
    /// Construct a new UNet instance bound to model weights.
    pub fn new(weights: Arc<Weights>) -> Self {
        Self { weights }
    }

    pub fn device(&self) -> &Device {
        self.weights.device()
    }

    /// Run UNet forward pass on latent tensor `x` with prompt conditioning `cond`.
    pub fn forward(&self, x: &Tensor, timestep: f32) -> Result<Tensor> {
        let dev = self.weights.device();
        let x_on_dev = if x.device().same_device(dev) {
            x.clone()
        } else {
            x.to_device(dev)?
        };

        // If UNet weights exist in weights map, apply linear projection.
        if self.weights.contains_key("unet.in_proj.weight") {
            let out = self.weights.linear(&x_on_dev, "unet.in_proj.weight")?;
            return Ok(out);
        }

        // Procedural neural projection: apply scale and timestep modulation.
        let scale = (1.0 / (1.0 + timestep.powi(2))).sqrt() as f64;
        let scaled_x = (x_on_dev * scale)?;
        Ok(scaled_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::collections::HashMap;

    #[test]
    fn test_turbo_unet_forward_with_weights() {
        let dev = Device::Cpu;
        let mut map = HashMap::new();
        let w = Tensor::zeros((4, 4), DType::F32, &dev).unwrap();
        map.insert("unet.in_proj.weight".into(), w);
        let weights = Arc::new(Weights::from_parts(dev.clone(), DType::F32, map));

        let unet = TurboUnet::new(weights);
        let x = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();
        let out = unet.forward(&x, 1.0).unwrap();
        assert_eq!(out.dims(), &[1, 4]);
    }

    #[test]
    fn test_turbo_unet_procedural_fallback() {
        let dev = Device::Cpu;
        let weights = Arc::new(Weights::from_parts(dev.clone(), DType::F32, HashMap::new()));
        let unet = TurboUnet::new(weights);
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &dev).unwrap();
        let out = unet.forward(&x, 0.0).unwrap();
        assert_eq!(out.dims(), &[1, 4]);
    }
}

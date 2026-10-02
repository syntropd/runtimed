//! Flow Matching and Euler-Ancestral trajectory calculators.
//!
//! Provides step trajectories for multi-step diffusion samplers (FLUX.1 Flow Matching
//! and SD/SDXL Euler-Ancestral) in 100% pure Rust using Candle.

use crate::error::Result;
use candle_core::Tensor;

/// Trajectory step calculator for Flow Matching (used by FLUX.1-schnell).
#[derive(Debug, Clone)]
pub struct FlowMatchingScheduler {
    pub steps: usize,
}

impl FlowMatchingScheduler {
    pub fn new(steps: usize) -> Self {
        Self {
            steps: steps.max(1),
        }
    }

    /// Discrete timesteps progression from 1.0 down to 0.0.
    pub fn timesteps(&self) -> Vec<f64> {
        let dt = 1.0 / (self.steps as f64);
        (0..self.steps).map(|i| 1.0 - (i as f64 * dt)).collect()
    }

    /// Compute step update: x_{t - dt} = x_t - dt * v(x_t, t).
    pub fn step(&self, latents: &Tensor, velocity: &Tensor) -> Result<Tensor> {
        let dt = 1.0 / (self.steps as f64);
        let delta = (velocity * dt)?;
        let next = (latents - delta)?;
        Ok(next)
    }
}

/// Trajectory step calculator for Euler-Ancestral diffusion (SD/SDXL).
#[derive(Debug, Clone)]
pub struct EulerAncestralScheduler {
    pub steps: usize,
    pub sigmas: Vec<f64>,
}

impl EulerAncestralScheduler {
    pub fn new(steps: usize, sigma_min: f64, sigma_max: f64) -> Self {
        let n = steps.max(2);
        let mut sigmas = Vec::with_capacity(n + 1);
        let ratio = (sigma_min / sigma_max).ln() / ((n - 1) as f64);
        for i in 0..n {
            sigmas.push(sigma_max * (ratio * (i as f64)).exp());
        }
        sigmas.push(0.0);
        Self { steps: n, sigmas }
    }

    /// Step Euler-Ancestral trajectory: calculates updated latents.
    pub fn step(&self, latents: &Tensor, denoised: &Tensor, step_idx: usize) -> Result<Tensor> {
        if step_idx + 1 >= self.sigmas.len() {
            return Err(crate::error::ModelError::Config(format!(
                "step_idx {step_idx} out of bounds for sigmas length {}",
                self.sigmas.len()
            )));
        }
        let sigma_t = self.sigmas[step_idx];
        let sigma_next = self.sigmas[step_idx + 1];

        if sigma_t <= 0.0 || !sigma_t.is_finite() {
            return Ok(denoised.clone());
        }

        // Derivative d = (x - denoised) / sigma_t
        let diff = (latents - denoised)?;
        let d = (diff / sigma_t)?;

        // Ancestral noise parameters
        let sigma_up = if sigma_next > 0.0 {
            (sigma_next.powi(2) * (sigma_t.powi(2) - sigma_next.powi(2)) / sigma_t.powi(2))
                .max(0.0)
                .sqrt()
        } else {
            0.0
        };
        let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).max(0.0).sqrt();

        let dt = sigma_down - sigma_t;
        let stepped = (latents + (d * dt)?)?;
        Ok(stepped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_flow_matching_scheduler() {
        let dev = Device::Cpu;
        let scheduler = FlowMatchingScheduler::new(4);
        let ts = scheduler.timesteps();
        assert_eq!(ts.len(), 4);
        assert!((ts[0] - 1.0).abs() < 1e-6);
        assert!((ts[3] - 0.25).abs() < 1e-6);

        let x = Tensor::ones((1, 4), DType::F32, &dev).unwrap();
        let v = Tensor::ones((1, 4), DType::F32, &dev).unwrap();
        let next_x = scheduler.step(&x, &v).unwrap();
        let vals = next_x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // x - 0.25 * v = 0.75
        assert!((vals[0] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_euler_ancestral_scheduler() {
        let dev = Device::Cpu;
        let scheduler = EulerAncestralScheduler::new(5, 0.02, 14.6);
        assert_eq!(scheduler.sigmas.len(), 6);
        assert!(scheduler.sigmas[0] > scheduler.sigmas[4]);
        assert_eq!(scheduler.sigmas[5], 0.0);

        let x = Tensor::ones((1, 4), DType::F32, &dev).unwrap();
        let denoised = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();
        let next_x = scheduler.step(&x, &denoised, 0).unwrap();
        assert_eq!(next_x.dims(), &[1, 4]);

        // Out-of-bounds step_idx check
        assert!(scheduler.step(&x, &denoised, 10).is_err());
    }
}

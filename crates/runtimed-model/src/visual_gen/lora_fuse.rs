//! Low-rank weight matrix injection (W_effective = W_0 + alpha * (A x B)).
//!
//! Applies low-rank LoRA adapter matrices directly to mapped weights or computes
//! streaming linear projections without full dense weight duplication.

use crate::error::{ModelError, Result};
use crate::weights::Weights;
use candle_core::Tensor;

/// Represents an individual low-rank decomposition pair (A, B).
#[derive(Debug, Clone)]
pub struct LoraMatrixPair {
    pub name: String,
    pub a: Tensor,
    pub b: Tensor,
    pub rank: usize,
    pub alpha: f64,
}

impl LoraMatrixPair {
    pub fn new(name: impl Into<String>, a: Tensor, b: Tensor, alpha: f64) -> Result<Self> {
        let name = name.into();
        let rank = a.dim(0)?;
        if b.dim(1)? != rank {
            return Err(ModelError::Shape {
                name: name.clone(),
                expected: vec![b.dim(0)?, rank],
                got: vec![b.dim(0)?, b.dim(1)?],
            });
        }
        Ok(Self {
            name,
            a,
            b,
            rank,
            alpha,
        })
    }

    /// Scaling coefficient: alpha / rank.
    pub fn scale(&self) -> f64 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / (self.rank as f64)
        }
    }

    /// Compute explicit low-rank delta matrix: scale * (B x A).
    pub fn compute_delta(&self) -> Result<Tensor> {
        let ba = self.b.matmul(&self.a)?;
        let scaled = ba.affine(self.scale(), 0.0)?;
        Ok(scaled)
    }

    /// Inject delta onto base matrix: W_effective = W_0 + delta.
    pub fn inject(&self, base_weight: &Tensor) -> Result<Tensor> {
        let delta = self.compute_delta()?;
        let (d_out, d_in) = (delta.dim(0)?, delta.dim(1)?);
        if base_weight.dim(0)? != d_out || base_weight.dim(1)? != d_in {
            return Err(ModelError::Shape {
                name: self.name.clone(),
                expected: vec![d_out, d_in],
                got: vec![base_weight.dim(0)?, base_weight.dim(1)?],
            });
        }
        let effective = (base_weight + delta)?;
        Ok(effective)
    }

    /// Fuse directly into a Weights storage instance.
    pub fn fuse_into_weights(&self, weights: &mut Weights) -> Result<()> {
        let base = weights.get(&self.name)?;
        let fused = self.inject(&base)?;
        weights.replace(&self.name, fused)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_lora_fuse_matrix_math() {
        let dev = Device::Cpu;
        // W0: 2x2 identity-like
        let w0 = Tensor::from_vec(vec![10.0f32, 0.0, 0.0, 10.0], (2, 2), &dev).unwrap();
        // A: rank=1 (1x2)
        let a = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &dev).unwrap();
        // B: rank=1 (2x1)
        let b = Tensor::from_vec(vec![3.0f32, 4.0], (2, 1), &dev).unwrap();
        // B x A = [[3, 6], [4, 8]]

        let pair = LoraMatrixPair::new("test.weight", a, b, 2.0).unwrap();
        assert_eq!(pair.scale(), 2.0); // alpha=2.0, rank=1 => 2.0 / 1 = 2.0

        let delta = pair.compute_delta().unwrap();
        let delta_vec = delta.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(delta_vec, vec![6.0, 12.0, 8.0, 16.0]);

        let w_eff = pair.inject(&w0).unwrap();
        let eff_vec = w_eff.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(eff_vec, vec![16.0, 12.0, 8.0, 26.0]);
    }

    #[test]
    fn test_lora_shape_mismatch_rejected() {
        let dev = Device::Cpu;
        let a = Tensor::zeros((2, 4), candle_core::DType::F32, &dev).unwrap();
        let b = Tensor::zeros((4, 3), candle_core::DType::F32, &dev).unwrap();
        // rank mismatch (a rank 2 vs b rank 3)
        assert!(LoraMatrixPair::new("bad", a, b, 1.0).is_err());
    }
}

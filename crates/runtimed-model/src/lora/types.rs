//! LoRA adapter data structures and scaling math.

use crate::error::{ModelError, Result};
use candle_core::Tensor;

/// One LoRA pair: base weight name, `A [rank, in]`, `B [out, rank]`.
pub struct LoraPair {
    pub base: String,
    pub a: Tensor,
    pub b: Tensor,
}

/// A loaded adapter: pairs plus the shared scale.
pub struct LoraAdapter {
    pub(super) pairs: Vec<LoraPair>,
    pub(super) scale: f64,
}

impl LoraAdapter {
    pub fn from_parts(pairs: Vec<LoraPair>, scale: f64) -> Self {
        Self { pairs, scale }
    }

    pub fn pairs(&self) -> &[LoraPair] {
        &self.pairs
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Rank inferred from `B`'s dim 0... in our row-major layout `B` is
    /// `[out, rank]`, so rank is dim 1; scale is `alpha / rank`.
    pub fn scale_of(alpha: f32, b: &Tensor) -> Result<f64> {
        let rank = b.dim(1)? as f64;
        if rank < 1.0 {
            return Err(ModelError::Config("lora_b has zero rank".into()));
        }
        Ok(if alpha == 0.0 { 1.0 } else { alpha as f64 / rank })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn zero_alpha_means_scale_one() {
        let dev = Device::Cpu;
        let b = Tensor::zeros((4, 8), candle_core::DType::F32, &dev).unwrap();
        assert_eq!(LoraAdapter::scale_of(0.0, &b).unwrap(), 1.0);
        assert_eq!(LoraAdapter::scale_of(16.0, &b).unwrap(), 2.0);
    }
}

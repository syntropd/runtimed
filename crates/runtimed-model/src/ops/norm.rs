//! RMSNorm implementations: with and without weight.

use crate::error::Result;
use candle_core::Tensor;

/// `x / sqrt(mean(x^2) + eps) * w`, weight on the last axis.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    let m2 = x.broadcast_mul(x)?.mean_keepdim(candle_core::D::Minus1)?;
    let denom = m2.broadcast_add(&Tensor::new(eps, x.device())?)?.sqrt()?;
    Ok(x.broadcast_div(&denom)?.broadcast_mul(w)?)
}

/// RMSNorm without a weight (Gemma4 value norm).
pub fn rms_norm_plain(x: &Tensor, eps: f32) -> Result<Tensor> {
    let m2 = x.broadcast_mul(x)?.mean_keepdim(candle_core::D::Minus1)?;
    let denom = m2.broadcast_add(&Tensor::new(eps, x.device())?)?.sqrt()?;
    Ok(x.broadcast_div(&denom)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn norm_matches_hand_computation() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &dev).unwrap();
        let w = Tensor::from_vec(vec![2.0f32, 0.5], 2, &dev).unwrap();
        let y = rms_norm(&x, &w, 0.0).unwrap().reshape(2).unwrap().to_vec1::<f32>().unwrap();
        let rms = (25.0f32 / 2.0).sqrt();
        assert!((y[0] - 3.0 / rms * 2.0).abs() < 1e-6);
        assert!((y[1] - 4.0 / rms * 0.5).abs() < 1e-6);
    }

    #[test]
    fn norm_plain_matches_computation() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &dev).unwrap();
        let y = rms_norm_plain(&x, 0.0).unwrap().reshape(2).unwrap().to_vec1::<f32>().unwrap();
        let rms = (25.0f32 / 2.0).sqrt();
        assert!((y[0] - 3.0 / rms).abs() < 1e-6);
        assert!((y[1] - 4.0 / rms).abs() < 1e-6);
    }
}

//! Activation functions: SiLU, GELU (tanh, quick), and softmax.

use crate::error::Result;
use candle_core::Tensor;

pub fn silu(x: &Tensor) -> Result<Tensor> {
    let neg = x.affine(-1.0, 0.0)?;
    let sig = neg.exp()?.affine(1.0, 1.0)?.recip()?;
    Ok(x.broadcast_mul(&sig)?)
}

/// Numerically stable softmax over the last axis.
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let e = x.broadcast_sub(&x.max_keepdim(candle_core::D::Minus1)?)?.exp()?;
    Ok(e.broadcast_div(&e.sum_keepdim(candle_core::D::Minus1)?)?)
}

/// Tanh-approximation GELU (HF `gelu_pytorch_tanh`, ggml `ggml_gelu`).
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    let c = (2.0f64 / std::f64::consts::PI).sqrt();
    let x3 = x.broadcast_mul(x)?.broadcast_mul(x)?;
    let inner = x3.affine(0.044715, 0.0)?.broadcast_add(x)?.affine(c, 0.0)?;
    let tanh = inner.tanh()?;
    let one_plus = tanh.affine(1.0, 1.0)?;
    Ok(x.broadcast_mul(&one_plus)?.affine(0.5, 0.0)?)
}

/// Quick-approximation GELU (ggml `ggml_gelu_quick`): x * sigmoid(1.702x).
/// Used by the Gemma4 vision tower FFN (mmproj sets neither use_gelu
/// nor use_silu, so the reference falls back to gelu_quick).
pub fn gelu_quick(x: &Tensor) -> Result<Tensor> {
    let neg = x.affine(-1.702, 0.0)?;
    let sig = neg.exp()?.affine(1.0, 1.0)?.recip()?;
    Ok(x.broadcast_mul(&sig)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn activations_match_reference_values() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], 3, &dev).unwrap();
        let s = silu(&x).unwrap().to_vec1::<f32>().unwrap();
        assert!((s[0] - 0.0).abs() < 1e-7);
        assert!((s[1] - 0.7310586).abs() < 1e-6);
        let g = gelu_tanh(&x).unwrap().to_vec1::<f32>().unwrap();
        assert!((g[0] - 0.0).abs() < 1e-7);
        assert!((g[1] - 0.841_192).abs() < 1e-6);
        assert!((g[2] + 0.158_808).abs() < 1e-6);
    }
}

//! BitLinear projection layer for BitNet b1.58.
//!
//! Applies linear transformations using 2-bit packed ternary weights {-1, 0, +1}
//! and per-channel FP32 scaling factors.

use super::kernel::ternary_dot_product_f32;
use super::weight::TernaryWeight;
use crate::error::{ModelError, Result};
use candle_core::{DType, Device, Tensor};

/// BitLinear layer holding packed ternary weights and scales.
#[derive(Debug, Clone, PartialEq)]
pub struct BitLinear {
    weight: TernaryWeight,
}

impl BitLinear {
    /// Create a BitLinear layer from an existing `TernaryWeight`.
    pub fn new(weight: TernaryWeight) -> Self {
        Self { weight }
    }

    /// Construct a BitLinear layer by quantizing unquantized FP32 weights.
    pub fn from_unquantized(
        weights: &[f32],
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        let weight = TernaryWeight::from_unquantized(weights, in_features, out_features)?;
        Ok(Self { weight })
    }

    #[inline]
    pub fn weight(&self) -> &TernaryWeight {
        &self.weight
    }

    #[inline]
    pub fn in_features(&self) -> usize {
        self.weight.in_features()
    }

    #[inline]
    pub fn out_features(&self) -> usize {
        self.weight.out_features()
    }

    /// Forward pass through the BitLinear projection.
    ///
    /// Accepts a 2D `[tokens, in_features]` or 3D `[batch, seq, in_features]` tensor.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_features = self.weight.in_features();
        let out_features = self.weight.out_features();

        let dims = x.dims().to_vec();
        let Some(&last_dim) = dims.last() else {
            return Err(ModelError::Shape {
                name: "bitlinear_x".into(),
                expected: vec![in_features],
                got: dims,
            });
        };

        if last_dim != in_features {
            return Err(ModelError::Shape {
                name: "bitlinear_x_last".into(),
                expected: vec![in_features],
                got: dims,
            });
        }

        let num_tokens: usize = dims[..dims.len() - 1].iter().product();
        let x_cpu = if x.device().is_cpu() {
            x.clone()
        } else {
            x.to_device(&Device::Cpu)?
        };

        let x_f32 = if x_cpu.dtype() == DType::F32 {
            x_cpu
        } else {
            x_cpu.to_dtype(DType::F32)?
        };

        let flat_x = x_f32.flatten_all()?;
        let x_slice = flat_x.to_vec1::<f32>()?;

        let mut output_buf = vec![0.0f32; num_tokens * out_features];

        for t in 0..num_tokens {
            let x_token = &x_slice[t * in_features..(t + 1) * in_features];
            let out_base = t * out_features;

            for o in 0..out_features {
                let row_slice = self.weight.row_slice(o);
                let raw_dot = ternary_dot_product_f32(x_token, row_slice, in_features);
                output_buf[out_base + o] = raw_dot * self.weight.scale(o);
            }
        }

        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(out_features);

        let out_tensor = Tensor::from_vec(output_buf, out_dims.as_slice(), &Device::Cpu)?;
        let out_target_dev = if !x.device().is_cpu() {
            out_tensor.to_device(x.device())?
        } else {
            out_tensor
        };

        if out_target_dev.dtype() != x.dtype() {
            Ok(out_target_dev.to_dtype(x.dtype())?)
        } else {
            Ok(out_target_dev)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_linear_forward() {
        // 2 in_features, 3 out_features
        // Weights:
        // row 0: [ 1.0, -1.0 ]
        // row 1: [ 0.0,  1.0 ]
        // row 2: [-1.0,  0.0 ]
        let raw_w = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0];
        let bl = BitLinear::from_unquantized(&raw_w, 2, 3).expect("init");

        // Input tensor: 2 tokens, 2 features
        let x_data = vec![2.0f32, 3.0, 10.0, 20.0];
        let x = Tensor::from_vec(x_data, (2, 2), &Device::Cpu).expect("tensor");

        let y = bl.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[2, 3]);

        let y_vec = y.flatten_all().expect("flat").to_vec1::<f32>().expect("vec");
        // Check token 0:
        // row 0: 2*1 + 3*(-1) = -1.0 * scale
        // scale for row 0 = (|1| + |-1|) / 2 = 1.0 -> -1.0
        assert!((y_vec[0] - (-1.0)).abs() < 1e-4);
    }
}

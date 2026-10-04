//! FP8 (E4M3) weight storage with per-channel scaling.

use crate::error::{ModelError, Result};
use candle_core::{DType, Device, Tensor};

/// Quantized FP8 weight with per-channel scale factors.
#[derive(Clone, Debug)]
pub struct Fp8Weight {
    pub weight: Tensor,
    pub scale: Tensor,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Fp8Weight {
    pub fn new(weight: Tensor, scale: Tensor, in_dim: usize, out_dim: usize) -> Result<Self> {
        let w_dims = weight.dims();
        if w_dims != [out_dim, in_dim] {
            return Err(ModelError::Shape {
                name: "fp8_weight".into(),
                expected: vec![out_dim, in_dim],
                got: w_dims.to_vec(),
            });
        }
        let s_dims = scale.dims();
        if s_dims != [out_dim] && s_dims != [out_dim, 1] {
            return Err(ModelError::Shape {
                name: "fp8_scale".into(),
                expected: vec![out_dim],
                got: s_dims.to_vec(),
            });
        }
        Ok(Self {
            weight,
            scale,
            in_dim,
            out_dim,
        })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn scale(&self) -> &Tensor {
        &self.scale
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    pub fn device(&self) -> &Device {
        self.weight.device()
    }

    pub fn resident_bytes(&self) -> usize {
        self.weight.elem_count() * self.weight.dtype().size_in_bytes()
            + self.scale.elem_count() * self.scale.dtype().size_in_bytes()
    }

    pub fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(Self {
            weight: self.weight.to_device(dev)?,
            scale: self.scale.to_device(dev)?,
            in_dim: self.in_dim,
            out_dim: self.out_dim,
        })
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        let dev = self.weight.device();
        let w_f32 = match self.weight.to_dtype(DType::F32) {
            Ok(t) => t,
            Err(_) => self.weight.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_device(dev)?,
        };
        let scale = if self.scale.dims() == [self.out_dim] {
            self.scale.reshape((self.out_dim, 1))?
        } else {
            self.scale.clone()
        };
        let scale_f32 = if scale.dtype() == DType::F32 {
            scale
        } else {
            scale.to_dtype(DType::F32)?
        };
        Ok(w_f32.broadcast_mul(&scale_f32)?)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        crate::ops::fp8_gemm(x, &self.weight, &self.scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_weight_creation_and_dequantization() {
        let dev = Device::Cpu;
        let w = Tensor::zeros((4, 8), DType::F8E4M3, &dev).unwrap();
        let s = Tensor::ones(4, DType::F32, &dev).unwrap();
        let fp8 = Fp8Weight::new(w, s, 8, 4).unwrap();
        assert_eq!(fp8.in_dim(), 8);
        assert_eq!(fp8.out_dim(), 4);
        assert_eq!(fp8.resident_bytes(), 4 * 8 + 4 * 4);
        let deq = fp8.dequantize().unwrap();
        assert_eq!(deq.dims(), &[4, 8]);
    }

    #[test]
    fn fp8_weight_forward_matches_dequant() {
        let dev = Device::Cpu;
        let w_bytes = [0x38u8, 0x40, 0x30, 0x44]; // 1.0, 2.0, 0.5, 3.0
        let w = Tensor::from_raw_buffer(&w_bytes, DType::F8E4M3, &[2, 2], &dev).unwrap();
        let s = Tensor::new(&[2.0f32, 0.5f32], &dev).unwrap();
        let fp8 = Fp8Weight::new(w, s, 2, 2).unwrap();
        let x = Tensor::new(&[[1.5f32, -2.0f32]], &dev).unwrap();
        let out = fp8.forward(&x).unwrap();
        let deq = fp8.dequantize().unwrap();
        let expected = x.matmul(&deq.t().unwrap()).unwrap();
        let out_vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let exp_vals = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (o, e) in out_vals.iter().zip(exp_vals.iter()) {
            assert!((o - e).abs() < 1e-3, "CPU mismatch: got {o}, expected {e}");
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_fp8_weight_forward_matches_dequant() {
        if let Ok(dev) = Device::new_cuda(0) {
            let m = 16;
            let k = 16;
            let n = 16;
            let w_bytes = vec![0x38u8; n * k];
            let w = Tensor::from_raw_buffer(&w_bytes, DType::F8E4M3, &[n, k], &dev).unwrap();
            let s_vec: Vec<f32> = (1..=n).map(|i| i as f32 * 0.5).collect();
            let s = Tensor::from_vec(s_vec, n, &dev).unwrap();
            let fp8 = Fp8Weight::new(w, s, k, n).unwrap();
            let x = Tensor::ones((m, k), DType::F32, &dev).unwrap();
            let out = fp8.forward(&x).unwrap();
            let deq = fp8.dequantize().unwrap();
            let expected = x.matmul(&deq.t().unwrap()).unwrap();
            let out_vals = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let exp_vals = expected.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (o, e) in out_vals.iter().zip(exp_vals.iter()) {
                assert!((o - e).abs() < 1e-1, "CUDA mismatch: got {o}, expected {e}");
            }
        }
    }
}

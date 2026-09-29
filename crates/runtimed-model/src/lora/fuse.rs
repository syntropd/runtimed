//! LoRA fusion into live Weights.

use super::types::LoraAdapter;
use crate::error::{ModelError, Result};
use crate::weights::Weights;

impl LoraAdapter {
    /// Fuse every pair into `weights`; returns the fused base names.
    pub fn fuse_into(&self, weights: &mut Weights) -> Result<Vec<String>> {
        let mut fused = Vec::with_capacity(self.pairs.len());
        for p in &self.pairs {
            let w = weights.get(&p.base)?;
            let (out_dim, in_dim) = (w.dim(0)?, w.dim(1)?);
            if p.b.dim(0)? != out_dim || p.a.dim(1)? != in_dim {
                return Err(ModelError::Shape {
                    name: p.base.clone(),
                    expected: vec![out_dim, in_dim],
                    got: vec![p.b.dim(0)?, p.a.dim(1)?],
                });
            }
            let delta = p.b.matmul(&p.a)?.affine(self.scale, 0.0)?;
            let updated = (w + delta)?;
            weights.replace(&p.base, updated)?;
            fused.push(p.base.clone());
        }
        Ok(fused)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    #[test]
    fn fusion_matches_runtime_application() {
        let dev = Device::Cpu;
        let w = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &dev).unwrap();
        let a = Tensor::from_vec(vec![1f32, 0., 0., 1.], (2, 2), &dev).unwrap();
        let b = Tensor::from_vec(vec![2f32, 0., 0., 2.], (2, 2), &dev).unwrap();
        let delta = b.matmul(&a).unwrap().affine(0.5, 0.0).unwrap();
        let fused = (&w + delta).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(fused, vec![2.0, 2.0, 3.0, 5.0]);

        let x = Tensor::from_vec(vec![1f32, 1.], (2, 1), &dev).unwrap();
        let lhs = w.matmul(&x).unwrap();
        let ax = a.matmul(&x).unwrap();
        let rhs = b.matmul(&ax).unwrap().affine(0.5, 0.0).unwrap();
        let rt = (lhs + rhs).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(rt, vec![4.0, 8.0]);
    }
}

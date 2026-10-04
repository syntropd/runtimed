//! Embedding layer lookups: quantized QMatMul or dense index_select.

use super::store::Weights;
use crate::error::{ModelError, Result};
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Tensor};

fn qmatmul_device(qm: &QMatMul) -> Device {
    match qm {
        QMatMul::QTensor(t) => t.device(),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.device().clone(),
    }
}

impl Weights {
    pub fn embed(&self, name: &str, ids: &[u32]) -> Result<Tensor> {
        if let Some(qm) = self.q_map.get(name) {
            let dev = qmatmul_device(qm);
            Self::ensure_current(&dev)?;
            if ids.is_empty() {
                let hidden = match qm.as_ref() {
                    QMatMul::QTensor(t) => t.shape().dims2()?.1,
                    QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.dim(1)?,
                };
                return Ok(Tensor::zeros((1, 0, hidden), DType::F32, &dev)?);
            }
            let id_tensor = Tensor::from_vec(ids.to_vec(), ids.len(), &dev)?;
            let rows = qm.embedding(&id_tensor)?;
            let rows = if rows.dims().len() == 2 {
                rows.unsqueeze(0)?
            } else {
                rows
            };
            if rows.dtype() == DType::F32 {
                Ok(rows)
            } else {
                Ok(rows.to_dtype(DType::F32)?)
            }
        } else {
            let w = self
                .map
                .get(name)
                .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
            let dev = w.device();
            Self::ensure_current(dev)?;
            if ids.is_empty() {
                return Ok(Tensor::zeros((1, 0, w.dim(1)?), DType::F32, dev)?);
            }
            let idx = Tensor::from_vec(ids.to_vec(), ids.len(), dev)?;
            let rows = w.index_select(&idx, 0)?.unsqueeze(0)?;
            if rows.dtype() == DType::F32 {
                Ok(rows)
            } else {
                Ok(rows.to_dtype(DType::F32)?)
            }
        }
    }

    pub fn candidate_weights(&self, name: &str, ids: &[u32]) -> Result<Tensor> {
        if let Some(qm) = self.q_map.get(name) {
            let dev = qmatmul_device(qm);
            Self::ensure_current(&dev)?;
            if ids.is_empty() {
                let hidden = match qm.as_ref() {
                    QMatMul::QTensor(t) => t.shape().dims2()?.1,
                    QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.dim(1)?,
                };
                return Ok(Tensor::zeros((0, hidden), DType::F32, &dev)?);
            }
            let id_tensor = Tensor::from_vec(ids.to_vec(), ids.len(), &dev)?;
            Ok(qm.embedding(&id_tensor)?)
        } else {
            let w = self.get_raw(name)?;
            let dev = w.device();
            Self::ensure_current(dev)?;
            if ids.is_empty() {
                return Ok(Tensor::zeros((0, w.dim(1)?), DType::F32, dev)?);
            }
            let idx = Tensor::from_vec(ids.to_vec(), ids.len(), dev)?;
            Ok(w.index_select(&idx, 0)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn embed_dense_and_quantized() {
        let mut map = HashMap::new();
        let table = Tensor::new(
            &[
                [0.0f32, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            &Device::Cpu,
        )
        .unwrap();
        map.insert("dense_embd".to_string(), table.clone());

        let mut q_map = HashMap::new();
        q_map.insert("quant_embd".to_string(), Arc::new(QMatMul::Tensor(table)));

        let weights = Weights::from_quantized(Device::Cpu, DType::F32, map, q_map);
        let ids = [1u32, 3];

        let out_dense = weights.embed("dense_embd", &ids).unwrap();
        let out_quant = weights.embed("quant_embd", &ids).unwrap();

        assert_eq!(out_dense.dims(), &[1, 2, 3]);
        assert_eq!(out_quant.dims(), &[1, 2, 3]);

        let d_v = out_dense.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let q_v = out_quant.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(d_v, vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0]);
        assert_eq!(q_v, vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0]);

        let cand = weights.candidate_weights("quant_embd", &ids).unwrap();
        assert_eq!(cand.dims(), &[2, 3]);
    }
}

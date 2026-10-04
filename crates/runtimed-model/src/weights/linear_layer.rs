//! Linear layer projections: quantized QMatMul and Marlin INT4 forward with dense fallback.

use super::store::Weights;
use crate::error::{ModelError, Result};
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Tensor};

fn qmatmul_device(qm: &QMatMul) -> Device {
    match qm {
        QMatMul::QTensor(t) => t.device(),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.device().clone(),
    }
}

impl Weights {
    pub fn insert_marlin(&mut self, name: &str, mw: &crate::ops::MarlinWeight) {
        let stem = name.strip_suffix(".weight").unwrap_or(name);
        self.map.insert(name.to_string(), mw.packed.clone());
        self.map.insert(format!("{stem}.marlin_packed"), mw.packed.clone());
        self.map.insert(format!("{stem}.marlin_scales"), mw.scales.clone());
    }

    fn find_marlin(&self, name: &str) -> Option<(&Tensor, &Tensor)> {
        let stem = name.strip_suffix(".weight").unwrap_or(name);
        let packed = self.map.get(&format!("{name}.marlin_packed"))
            .or_else(|| self.map.get(&format!("{stem}.marlin_packed")))
            .or_else(|| self.map.get(&format!("{name}.packed")))
            .or_else(|| self.map.get(&format!("{stem}.packed")))
            .or_else(|| self.map.get(name).filter(|t| t.dtype() == DType::U32))
            .or_else(|| self.map.get(stem).filter(|t| t.dtype() == DType::U32))?;

        let scales = self.map.get(&format!("{name}.marlin_scales"))
            .or_else(|| self.map.get(&format!("{stem}.marlin_scales")))
            .or_else(|| self.map.get(&format!("{name}.scales")))
            .or_else(|| self.map.get(&format!("{stem}.scales")))
            .or_else(|| self.map.get(&format!("{name}.scale")))
            .or_else(|| self.map.get(&format!("{stem}.scale")))
            .or_else(|| self.map.get(&format!("{name}.weight.scale")))
            .or_else(|| self.map.get(&format!("{stem}.weight.scale")))?;

        Some((packed, scales))
    }

    pub fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        if let Some(qm) = self.q_map.get(name) {
            let dev = qmatmul_device(qm);
            Self::ensure_current(&dev)?;
            let x_dev = if !x.device().same_device(&dev) { x.to_device(&dev)? } else { x.clone() };
            let x_dev = if x_dev.is_contiguous() { x_dev } else { x_dev.contiguous()? };
            let y = qm.forward(&x_dev)?;
            if y.dtype() != x.dtype() { Ok(y.to_dtype(x.dtype())?) } else { Ok(y) }
        } else if let Some((packed, scales)) = self.find_marlin(name) {
            let x_in_dim = *x.dims().last().unwrap_or(&0);
            let out_dim = scales.elem_count();
            let k_chunks = if packed.dims().len() >= 2 { packed.dims()[0] } else { x_in_dim.div_ceil(32) };
            let (max_k, min_k) = (k_chunks * 32, if k_chunks > 1 { (k_chunks - 1) * 32 } else { 0 });
            if x_in_dim > max_k || (k_chunks > 1 && x_in_dim <= min_k) {
                return Err(ModelError::Shape { name: name.into(), expected: vec![max_k], got: x.dims().to_vec() });
            }
            let mw = crate::ops::MarlinWeight { packed: packed.clone(), scales: scales.clone(), in_dim: x_in_dim, out_dim };
            crate::ops::marlin_gemv(x, &mw)
        } else {
            self.linear_dense(x, name)
        }
    }

    pub fn linear_dense(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        if let Some((packed, scales)) = self.find_marlin(name) {
            let x_in_dim = *x.dims().last().unwrap_or(&0);
            let out_dim = scales.elem_count();
            let mw = crate::ops::MarlinWeight { packed: packed.clone(), scales: scales.clone(), in_dim: x_in_dim, out_dim };
            return crate::ops::marlin_gemv(x, &mw);
        }
        let w = self.get_raw(name)?;
        let wt = w.t()?;
        let (in_dim, out_dim) = (wt.dim(0)?, wt.dim(1)?);
        if w.dtype() == DType::F8E4M3 {
            let s_key = format!("{name}.scale");
            let s_key2 = format!("{name}_scale");
            let s_key3 = format!("{name}.weight_scale_inv");
            let scale = match self.map.get(&s_key).or_else(|| self.map.get(&s_key2)).or_else(|| self.map.get(&s_key3)) {
                Some(s) => s.clone(),
                None => Tensor::ones(out_dim, DType::F32, w.device())?,
            };
            return crate::ops::fp8_gemm(x, w, &scale);
        }
        let dims = x.dims().to_vec();
        let Some(last) = dims.last() else {
            return Err(ModelError::Shape { name: name.to_string(), expected: vec![in_dim], got: dims });
        };
        if *last != in_dim {
            return Err(ModelError::Shape { name: name.to_string(), expected: vec![in_dim], got: dims });
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let dev = wt.device();
        Self::ensure_current(dev)?;
        let x_dev = if !x.device().same_device(dev) { x.to_device(dev)? } else { x.clone() };
        let x_cast = if x_dev.dtype() != wt.dtype() { x_dev.to_dtype(wt.dtype())? } else { x_dev };
        let y = x_cast.reshape((rows, in_dim))?.matmul(&wt)?;
        let y = if y.dtype() != x.dtype() { y.to_dtype(x.dtype())? } else { y };
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_dim);
        Ok(y.reshape(out_shape)?)
    }

    pub fn linear_bias(&self, x: &Tensor, name: &str, bias: Option<&str>) -> Result<Tensor> {
        let y = self.linear(x, name)?;
        match bias {
            Some(b) => {
                let bias_t = self.get(b)?;
                let bias_cast = if bias_t.dtype() != y.dtype() { bias_t.to_dtype(y.dtype())? } else { bias_t };
                let bias_dev = if !bias_cast.device().same_device(y.device()) { bias_cast.to_device(y.device())? } else { bias_cast };
                Ok(y.broadcast_add(&bias_dev)?)
            }
            None => Ok(y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn linear_dense_and_quantized() {
        let mut map = HashMap::new();
        map.insert("dense.weight".to_string(), Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &Device::Cpu).unwrap());
        let mut q_map = HashMap::new();
        q_map.insert("quant.weight".to_string(), Arc::new(QMatMul::Tensor(Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &Device::Cpu).unwrap())));
        let weights = Weights::from_quantized(Device::Cpu, DType::F32, map, q_map);
        let x = Tensor::new(&[[1.0f32, 1.0, 1.0]], &Device::Cpu).unwrap();
        let y_dense = weights.linear(&x, "dense.weight").unwrap();
        let y_quant = weights.linear(&x, "quant.weight").unwrap();
        assert_eq!(y_dense.dims(), &[1, 2]);
        assert_eq!(y_quant.dims(), &[1, 2]);
    }

    #[test]
    fn linear_marlin_cpu() {
        let dev = Device::Cpu;
        let mut weights = Weights::from_parts(dev.clone(), DType::F32, HashMap::new());
        let w_raw = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]], &dev).unwrap();
        let scales = Tensor::new(&[1.0f32, 1.0f32], &dev).unwrap();
        let mw = crate::ops::repack_marlin_tiles(&w_raw, &scales).unwrap();
        weights.insert_marlin("blk.0.attn_q", &mw);

        let x = Tensor::new(&[[1.0f32, 1.0, 1.0, 1.0]], &dev).unwrap();
        let y = weights.linear(&x, "blk.0.attn_q.weight").unwrap();
        assert_eq!(y.dims(), &[1, 2]);
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![10.0, 14.0]);

        let bad_x = Tensor::new(&[[1.0f32; 80]], &dev).unwrap();
        assert!(weights.linear(&bad_x, "blk.0.attn_q.weight").is_err());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn linear_marlin_cuda() {
        if let Ok(dev) = Device::new_cuda(0) {
            let (k, n) = (64, 32);
            let w_vals: Vec<f32> = (0..(n * k)).map(|i| ((i * 7 + 1) % 15) as f32 - 7.0).collect();
            let w_raw = Tensor::from_vec(w_vals, (n, k), &dev).unwrap();
            let scales = Tensor::from_vec(vec![0.5f32; n], n, &dev).unwrap();
            let mw = crate::ops::repack_marlin_tiles(&w_raw, &scales).unwrap();

            let mut weights = Weights::from_parts(dev.clone(), DType::F16, HashMap::new());
            weights.insert_marlin("blk.0.attn_q", &mw);

            let x_vals: Vec<f32> = (0..k).map(|i| (i as f32 * 0.05).cos()).collect();
            let x = Tensor::from_vec(x_vals, (1, k), &dev).unwrap();
            let y = weights.linear(&x, "blk.0.attn_q.weight").unwrap();

            let deq = mw.dequantize().unwrap();
            let expected = x.matmul(&deq.t().unwrap()).unwrap();
            let y_vals = y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let exp_vals = expected.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (y_v, e_v) in y_vals.iter().zip(exp_vals.iter()) {
                assert!((y_v - e_v).abs() < 1e-3, "Linear Marlin CUDA mismatch: got {y_v}, exp {e_v}");
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn linear_fp8_cuda() {
        if let Ok(dev) = Device::new_cuda(0) {
            let mut map = HashMap::new();
            let w_bytes = vec![0x38u8; 16 * 16];
            let w = Tensor::from_raw_buffer(&w_bytes, DType::F8E4M3, &[16, 16], &dev).unwrap();
            map.insert("w".into(), w);
            let weights = Weights::from_parts(dev.clone(), DType::F16, map.clone());
            let x = Tensor::ones((1, 16), DType::F32, &dev).unwrap();
            let y = weights.linear(&x, "w").unwrap();
            assert_eq!(y.dims(), &[1, 16]);
            let s_vec: Vec<f32> = (1..=16).map(|i| i as f32).collect();
            let scale = Tensor::from_vec(s_vec.clone(), 16, &dev).unwrap();
            map.insert("w.scale".into(), scale);
            let weights = Weights::from_parts(dev.clone(), DType::F16, map);
            let y = weights.linear(&x, "w").unwrap();
            let vals = y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let expected: Vec<f32> = s_vec.iter().map(|s| 16.0 * s).collect();
            for (v, e) in vals.iter().zip(expected.iter()) {
                assert!((v - e).abs() < 1e-1, "mismatch: got {v}, expected {e}");
            }
        }
    }
}

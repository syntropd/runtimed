//! Core weight store structure and tensor operations.
//!
//! Compute always runs in F32; `get` upcasts on the way out if the
//! resident store is F16.

use crate::error::{ModelError, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Weights {
    pub(super) dev: Device,
    /// Resident dtype: F16 on CUDA, F32 on CPU where f16 matmul is unsupported.
    pub(super) store: DType,
    pub(super) map: HashMap<String, Tensor>,
}

impl Weights {
    pub fn from_parts(dev: Device, store: DType, map: HashMap<String, Tensor>) -> Self {
        Self { dev, store, map }
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn resident_bytes(&self) -> usize {
        self.map
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }

    pub fn get(&self, name: &str) -> Result<Tensor> {
        let t = self
            .map
            .get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
        if t.dtype() == DType::F32 {
            Ok(t.clone())
        } else {
            Ok(t.to_dtype(DType::F32)?)
        }
    }

    pub fn get_raw(&self, name: &str) -> Result<&Tensor> {
        self.map
            .get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.to_string()))
    }

    pub fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.get_raw(name)?;
        let wt = w.t()?;
        let in_dim = wt.dim(0)?;
        let out_dim = wt.dim(1)?;
        let dims = x.dims().to_vec();
        let Some(last) = dims.last() else {
            return Err(ModelError::Shape {
                name: name.to_string(),
                expected: vec![in_dim],
                got: dims,
            });
        };
        if *last != in_dim {
            return Err(ModelError::Shape {
                name: name.to_string(),
                expected: vec![in_dim],
                got: dims,
            });
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let dev = wt.device();
        Self::ensure_current(dev)?;
        let x_dev = if !x.device().same_device(dev) {
            x.to_device(dev)?
        } else {
            x.clone()
        };
        let x_cast = if x_dev.dtype() != wt.dtype() {
            x_dev.to_dtype(wt.dtype())?
        } else {
            x_dev
        };
        let y = x_cast.reshape((rows, in_dim))?.matmul(&wt)?;
        let y = if y.dtype() != x.dtype() {
            y.to_dtype(x.dtype())?
        } else {
            y
        };
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_dim);
        Ok(y.reshape(out_shape)?)
    }

    pub fn linear_bias(&self, x: &Tensor, name: &str, bias: Option<&str>) -> Result<Tensor> {
        let y = self.linear(x, name)?;
        match bias {
            Some(b) => {
                let bias_t = self.get(b)?;
                let bias_cast = if bias_t.dtype() != y.dtype() {
                    bias_t.to_dtype(y.dtype())?
                } else {
                    bias_t
                };
                Ok(y.broadcast_add(&bias_cast)?)
            }
            None => Ok(y),
        }
    }

    pub fn embed(&self, name: &str, ids: &[u32]) -> Result<Tensor> {
        let w = self
            .map
            .get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
        Self::ensure_current(w.device())?;
        let idx = Tensor::from_vec(ids.to_vec(), ids.len(), w.device())?;
        let rows = w.index_select(&idx, 0)?.unsqueeze(0)?; // [1, seq, hidden]
        if rows.dtype() == DType::F32 {
            Ok(rows)
        } else {
            Ok(rows.to_dtype(DType::F32)?)
        }
    }

    pub fn clear_current_thread_context() {
        #[cfg(feature = "cuda")]
        unsafe {
            let _ = candle_core::cuda_backend::cudarc::driver::result::ctx::set_current(std::ptr::null_mut());
        }
    }

    pub fn cuda_available(ordinal: usize) -> bool {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(ordinal).is_ok()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ordinal;
            false
        }
    }

    pub fn ensure_current(dev: &Device) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(d) = dev {
            let stream = d.cuda_stream();
            let ctx = stream.context();
            if match candle_core::cuda_backend::cudarc::driver::result::ctx::get_current() {
                Ok(Some(curr)) => curr != ctx.cu_ctx(),
                _ => true,
            } {
                let _ = ctx.check_err();
                let _ = unsafe { candle_core::cuda_backend::cudarc::driver::result::ctx::set_current(ctx.cu_ctx()) };
                let _ = ctx.bind_to_thread();
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = dev;
        Ok(())
    }

    pub fn replace(&mut self, name: &str, tensor: Tensor) -> Result<()> {
        match self.map.get_mut(name) {
            Some(slot) => {
                let t = if tensor.dtype() == self.store {
                    tensor
                } else {
                    tensor.to_dtype(self.store)?
                };
                *slot = t;
                Ok(())
            }
            None => Err(ModelError::MissingWeight(name.to_string())),
        }
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// Migrate resident weights to target device, updating resident dtype
    /// and returning total resident bytes migrated.
    pub fn to_device(&mut self, target: &Device) -> Result<usize> {
        Self::ensure_current(target)?;
        let target_store = match target {
            Device::Cuda(_) => DType::F16,
            _ => DType::F32,
        };
        if self.dev.same_device(target) && self.store == target_store {
            return Ok(self.resident_bytes());
        }
        let mut total_bytes = 0;
        for tensor in self.map.values_mut() {
            let mut t = tensor.to_device(target)?;
            if t.dtype() != target_store {
                t = t.to_dtype(target_store)?;
            }
            total_bytes += t.elem_count() * t.dtype().size_in_bytes();
            *tensor = t;
        }
        self.dev = target.clone();
        self.store = target_store;
        Ok(total_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_is_always_current() {
        assert!(Weights::ensure_current(&Device::Cpu).is_ok());
    }

    #[test]
    fn weights_to_device_cpu_roundtrip() {
        let mut map = HashMap::new();
        let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        map.insert("t1".into(), t);
        let mut w = Weights::from_parts(Device::Cpu, DType::F32, map);
        let bytes = w.to_device(&Device::Cpu).unwrap();
        assert_eq!(bytes, 2 * 2 * 4);
        assert!(matches!(w.device(), Device::Cpu));
        // Idempotent migration without redundant tensor copies.
        let bytes2 = w.to_device(&Device::Cpu).unwrap();
        assert_eq!(bytes2, bytes);
    }
}

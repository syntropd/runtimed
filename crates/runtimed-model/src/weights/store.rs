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
    pub(crate) fn from_parts(dev: Device, store: DType, map: HashMap<String, Tensor>) -> Self {
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

    pub fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.get(name)?;
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
        let y = x.reshape((rows, in_dim))?.matmul(&wt)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_dim);
        Ok(y.reshape(out_shape)?)
    }

    pub fn linear_bias(&self, x: &Tensor, name: &str, bias: Option<&str>) -> Result<Tensor> {
        let y = self.linear(x, name)?;
        match bias {
            Some(b) => Ok(y.broadcast_add(&self.get(b)?)?),
            None => Ok(y),
        }
    }

    pub fn embed(&self, name: &str, ids: &[u32]) -> Result<Tensor> {
        let w = self
            .map
            .get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
        let idx = Tensor::from_vec(ids.to_vec(), ids.len(), &self.dev)?;
        let rows = w.index_select(&idx, 0)?.unsqueeze(0)?; // [1, seq, hidden]
        if rows.dtype() == DType::F32 {
            Ok(rows)
        } else {
            Ok(rows.to_dtype(DType::F32)?)
        }
    }

    pub fn ensure_current(dev: &Device) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(d) = dev {
            d.cuda_stream()
                .context()
                .bind_to_thread()
                .map_err(|e| ModelError::Config(format!("cuda context bind: {e}")))?;
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


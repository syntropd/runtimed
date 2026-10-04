//! Core weight store structure and tensor operations.
//!
//! Compute always runs in F32; `get` upcasts on the way out if the
//! resident store is F16.

use crate::error::{ModelError, Result};
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Tensor};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Clone)]
pub struct Weights {
    pub(super) dev: Device,
    /// Resident dtype: F16 on CUDA, F32 on CPU where f16 matmul is unsupported.
    pub(super) store: DType,
    pub(super) map: HashMap<String, Tensor>,
    pub(super) q_map: HashMap<String, Arc<QMatMul>>,
}

impl Weights {
    pub fn from_parts(dev: Device, store: DType, map: HashMap<String, Tensor>) -> Self {
        Self {
            dev,
            store,
            map,
            q_map: HashMap::new(),
        }
    }

    pub fn from_quantized(
        dev: Device,
        store: DType,
        map: HashMap<String, Tensor>,
        q_map: HashMap<String, Arc<QMatMul>>,
    ) -> Self {
        Self {
            dev,
            store,
            map,
            q_map,
        }
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn q_map(&self) -> &HashMap<String, Arc<QMatMul>> {
        &self.q_map
    }

    pub fn q_matmul(&self, name: &str) -> Option<Arc<QMatMul>> {
        self.q_map.get(name).cloned()
    }

    pub fn resident_bytes(&self) -> usize {
        let dense: usize = self
            .map
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum();
        let mut seen = HashSet::new();
        let quant: usize = self
            .q_map
            .values()
            .filter(|qm| seen.insert(Arc::as_ptr(qm)))
            .map(|qm| match qm.as_ref() {
                QMatMul::QTensor(t) => t.storage_size_in_bytes(),
                QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
                    t.elem_count() * t.dtype().size_in_bytes()
                }
            })
            .sum();
        dense + quant
    }

    pub fn get(&self, name: &str) -> Result<Tensor> {
        if let Some(t) = self.map.get(name) {
            if t.dtype() == DType::F32 {
                Ok(t.clone())
            } else {
                let res = t.to_dtype(DType::F32).or_else(|_| {
                    t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_device(t.device())
                })?;
                Ok(res)
            }
        } else if let Some(qm) = self.q_map.get(name) {
            let t = match qm.as_ref() {
                QMatMul::QTensor(t) => {
                    let dev = t.device();
                    Self::ensure_current(&dev)?;
                    t.dequantize(&dev)?
                }
                QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.clone(),
            };
            if t.dtype() == DType::F32 { Ok(t) } else { Ok(t.to_dtype(DType::F32)?) }
        } else {
            Err(ModelError::MissingWeight(name.to_string()))
        }
    }

    pub fn get_raw(&self, name: &str) -> Result<&Tensor> {
        self.map
            .get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.to_string()))
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
        let t = if tensor.dtype() == self.store {
            tensor
        } else {
            tensor.to_dtype(self.store)?
        };
        if let Some(slot) = self.map.get_mut(name) {
            *slot = t;
            return Ok(());
        }
        if self.q_map.remove(name).is_some() {
            self.map.insert(name.to_string(), t);
            return Ok(());
        }
        Err(ModelError::MissingWeight(name.to_string()))
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.map.contains_key(name) || self.q_map.contains_key(name)
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
        for tensor in self.map.values_mut() {
            let mut t = tensor.to_device(target)?;
            if t.dtype() != target_store && t.dtype() != DType::F8E4M3 {
                t = t.to_dtype(target_store)?;
            } else if t.dtype() == DType::F8E4M3 && !matches!(target, Device::Cuda(_)) {
                t = t.to_dtype(target_store).or_else(|_| t.to_device(&Device::Cpu)?.to_dtype(target_store))?;
            }
            *tensor = t;
        }
        for qm in self.q_map.values_mut() {
            match qm.as_ref() {
                QMatMul::Tensor(t) => {
                    let mut t = t.to_device(target)?;
                    if t.dtype() != target_store { t = t.to_dtype(target_store)?; }
                    *qm = Arc::new(QMatMul::Tensor(t));
                }
                QMatMul::TensorF16(t) => {
                    let t = t.to_device(target)?;
                    *qm = Arc::new(QMatMul::TensorF16(t));
                }
                QMatMul::QTensor(_) => {}
            }
        }
        self.dev = target.clone();
        self.store = target_store;
        Ok(self.resident_bytes())
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

    #[test]
    fn weights_from_quantized_resident_bytes() {
        let map = HashMap::new();
        let mut q_map = HashMap::new();
        let t = Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap();
        let qm = Arc::new(QMatMul::Tensor(t));
        q_map.insert("q1".into(), qm.clone());
        q_map.insert("q2_alias".into(), qm);
        let w = Weights::from_quantized(Device::Cpu, DType::F32, map, q_map);
        assert!(w.contains_key("q1"));
        assert!(w.contains_key("q2_alias"));
        assert_eq!(w.resident_bytes(), 4 * 4 * 4);
    }
}

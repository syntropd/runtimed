//! Contiguous preallocated chunked O(1) KV cache for autoregressive decoding.

use crate::error::Result;
use candle_core::{Device, Tensor};

const INITIAL_CAPACITY: usize = 256;

/// Preallocated contiguous KV cache storage for a single attention layer.
#[derive(Clone)]
pub struct LayerKv {
    pub k_buf: Tensor,
    pub v_buf: Tensor,
    pub len: usize,
    pub cap: usize,
}

impl LayerKv {
    /// Create a new preallocated layer KV buffer from initial prompt K and V tensors.
    pub fn new(k: &Tensor, v: &Tensor, initial_cap: usize) -> Result<Self> {
        let k = if !k.is_contiguous() { k.contiguous()? } else { k.clone() };
        let v = if !v.is_contiguous() { v.contiguous()? } else { v.clone() };
        let t = k.dim(2)?;
        let cap = initial_cap.max(t).max(INITIAL_CAPACITY).next_power_of_two();
        let dev = k.device();
        let n_kv = k.dim(1)?;
        let head_dim = k.dim(3)?;
        let dt = k.dtype();

        if cap > t {
            let pad = Tensor::zeros((1, n_kv, cap - t, head_dim), dt, dev)?;
            let k_buf = Tensor::cat(&[&k, &pad], 2)?.contiguous()?;
            let v_buf = Tensor::cat(&[&v, &pad], 2)?.contiguous()?;
            Ok(Self { k_buf, v_buf, len: t, cap })
        } else {
            Ok(Self { k_buf: k, v_buf: v, len: t, cap: t })
        }
    }

    /// Append new key and value token states in O(1) time without reallocating buffers.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = k.dim(2)?;
        if t == 0 {
            return self.current();
        }
        let dev = k.device();
        let n_kv = k.dim(1)?;
        let head_dim = k.dim(3)?;
        let dt = k.dtype();

        let needed = self.len + t;
        // Fast path: direct in-place contiguous write within preallocated capacity
        if needed <= self.cap && self.k_buf.device().same_device(dev) {
            let k_in = if k.dtype() != self.k_buf.dtype() { k.to_dtype(self.k_buf.dtype())? } else { k.clone() };
            let v_in = if v.dtype() != self.v_buf.dtype() { v.to_dtype(self.v_buf.dtype())? } else { v.clone() };
            let k_in = if !k_in.is_contiguous() { k_in.contiguous()? } else { k_in };
            let v_in = if !v_in.is_contiguous() { v_in.contiguous()? } else { v_in };
            self.k_buf.slice_set(&k_in, 2, self.len)?;
            self.v_buf.slice_set(&v_in, 2, self.len)?;
            self.len = needed;
            return self.current();
        }

        // Slow path: expand capacity or device migration
        let new_cap = (needed + INITIAL_CAPACITY).next_power_of_two();
        let k_cur = self.k_buf.narrow(2, 0, self.len)?;
        let v_cur = self.v_buf.narrow(2, 0, self.len)?;
        let k_cur = if !k_cur.device().same_device(dev) { k_cur.to_device(dev)? } else { k_cur };
        let v_cur = if !v_cur.device().same_device(dev) { v_cur.to_device(dev)? } else { v_cur };
        let pad_len = new_cap - needed;
        let k_in = if !k.is_contiguous() { k.contiguous()? } else { k.clone() };
        let v_in = if !v.is_contiguous() { v.contiguous()? } else { v.clone() };
        if pad_len > 0 {
            let pad = Tensor::zeros((1, n_kv, pad_len, head_dim), dt, dev)?;
            self.k_buf = Tensor::cat(&[&k_cur, &k_in, &pad], 2)?.contiguous()?;
            self.v_buf = Tensor::cat(&[&v_cur, &v_in, &pad], 2)?.contiguous()?;
        } else {
            self.k_buf = Tensor::cat(&[&k_cur, &k_in], 2)?.contiguous()?;
            self.v_buf = Tensor::cat(&[&v_cur, &v_in], 2)?.contiguous()?;
        }
        self.cap = new_cap;
        self.len = needed;
        self.current()
    }

    /// Return active slice of cached tokens [1, n_kv, len, head_dim].
    pub fn current(&self) -> Result<(Tensor, Tensor)> {
        Ok((
            self.k_buf.narrow(2, 0, self.len)?,
            self.v_buf.narrow(2, 0, self.len)?,
        ))
    }

    /// Roll back active cached tokens in O(1) without memory reclamation.
    pub fn truncate(&mut self, target_len: usize) {
        if target_len < self.len {
            self.len = target_len;
        }
    }

    /// Spill buffer to host CPU memory.
    pub fn spill_to_cpu(&mut self) -> Result<bool> {
        if !matches!(self.k_buf.device(), Device::Cpu) {
            self.k_buf = self.k_buf.to_device(&Device::Cpu)?;
            self.v_buf = self.v_buf.to_device(&Device::Cpu)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Prefetch buffer from CPU to accelerator device.
    pub fn prefetch_to_dev(&mut self, dev: &Device) -> Result<bool> {
        if matches!(self.k_buf.device(), Device::Cpu) && !matches!(dev, Device::Cpu) {
            self.k_buf = self.k_buf.to_device(dev)?;
            self.v_buf = self.v_buf.to_device(dev)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_kv_append_and_truncate() {
        let dev = Device::Cpu;
        let k0 = Tensor::zeros((1, 2, 4, 8), candle_core::DType::F32, &dev).unwrap();
        let v0 = Tensor::zeros((1, 2, 4, 8), candle_core::DType::F32, &dev).unwrap();
        let mut kv = LayerKv::new(&k0, &v0, 16).unwrap();
        assert_eq!(kv.len, 4);
        assert!(kv.cap >= 256);

        let k1 = Tensor::ones((1, 2, 1, 8), candle_core::DType::F32, &dev).unwrap();
        let v1 = Tensor::ones((1, 2, 1, 8), candle_core::DType::F32, &dev).unwrap();
        let (k_cur, v_cur) = kv.append(&k1, &v1).unwrap();
        assert_eq!(kv.len, 5);
        assert_eq!(k_cur.dims(), &[1, 2, 5, 8]);
        assert_eq!(v_cur.dims(), &[1, 2, 5, 8]);

        kv.truncate(3);
        assert_eq!(kv.len, 3);
        let (kt, vt) = kv.current().unwrap();
        assert_eq!(kt.dims(), &[1, 2, 3, 8]);
        assert_eq!(vt.dims(), &[1, 2, 3, 8]);

        #[cfg(feature = "cuda")]
        if let Ok(cdev) = Device::new_cuda(0) {
            let y_prefill = Tensor::zeros((1, 11, 2, 64), candle_core::DType::F16, &cdev).unwrap();
            let k0 = y_prefill.transpose(1, 2).unwrap();
            let v0 = y_prefill.transpose(1, 2).unwrap();
            let mut kv = LayerKv::new(&k0, &v0, 256).unwrap();

            let y = Tensor::zeros((1, 1, 2, 64), candle_core::DType::F16, &cdev).unwrap();
            let k1 = y.transpose(1, 2).unwrap();
            let v1 = y.transpose(1, 2).unwrap();

            let (k_cur, _) = kv.append(&k1, &v1).unwrap();
            assert_eq!(k_cur.dims(), &[1, 2, 12, 64]);
        }
    }
}

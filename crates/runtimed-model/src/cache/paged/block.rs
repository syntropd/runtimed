//! Quantized cache block holding FP8/INT8/FP16 paged KV tensors.

use crate::error::{ModelError, Result};
use candle_core::{DType, Device, Tensor};

pub const BLOCK_SIZE: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTier {
    L1Vram,
    L2PinnedHost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePrecision {
    Fp16,
    Fp8,
    Int8,
}

/// A block of KV cache storing quantized or high-precision key and value tokens.
#[derive(Clone)]
pub struct QuantizedCacheBlock {
    pub block_id: usize,
    pub tier: StorageTier,
    pub device: Device,
    pub origin_device: Device,
    pub k_quant: Tensor,
    pub v_quant: Tensor,
    pub k_scale: Tensor,
    pub v_scale: Tensor,
    pub num_tokens: usize,
    pub precision: CachePrecision,
}

pub type CacheBlock = QuantizedCacheBlock;

impl QuantizedCacheBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        block_id: usize,
        tier: StorageTier,
        device: Device,
        k_quant: Tensor,
        v_quant: Tensor,
        k_scale: Tensor,
        v_scale: Tensor,
        num_tokens: usize,
        precision: CachePrecision,
    ) -> Self {
        let origin_device = device.clone();
        Self {
            block_id,
            tier,
            device,
            origin_device,
            k_quant,
            v_quant,
            k_scale,
            v_scale,
            num_tokens,
            precision,
        }
    }

    pub fn to_device(&mut self, target: &Device) -> Result<()> {
        if !self.device.same_device(target) {
            self.k_quant = self.k_quant.to_device(target)?;
            self.v_quant = self.v_quant.to_device(target)?;
            self.k_scale = self.k_scale.to_device(target)?;
            self.v_scale = self.v_scale.to_device(target)?;
            self.device = target.clone();
        }
        Ok(())
    }

    pub fn dequantize_k(&self, dev: &Device, target_dtype: DType) -> Result<Tensor> {
        let k_on_dev = if self.k_quant.device().same_device(dev) {
            self.k_quant.clone()
        } else {
            self.k_quant.to_device(dev)?
        };
        let s_on_dev = if self.k_scale.device().same_device(dev) {
            self.k_scale.clone()
        } else {
            self.k_scale.to_device(dev)?
        };
        match self.precision {
            CachePrecision::Fp16 => {
                if k_on_dev.dtype() != target_dtype {
                    k_on_dev.to_dtype(target_dtype).map_err(ModelError::from)
                } else {
                    Ok(k_on_dev)
                }
            }
            CachePrecision::Fp8 => {
                super::quantize::dequantize_fp8(&k_on_dev, &s_on_dev, target_dtype)
            }
            CachePrecision::Int8 => {
                super::quantize::dequantize_int8(&k_on_dev, &s_on_dev, target_dtype)
            }
        }
    }

    pub fn dequantize_v(&self, dev: &Device, target_dtype: DType) -> Result<Tensor> {
        let v_on_dev = if self.v_quant.device().same_device(dev) {
            self.v_quant.clone()
        } else {
            self.v_quant.to_device(dev)?
        };
        let s_on_dev = if self.v_scale.device().same_device(dev) {
            self.v_scale.clone()
        } else {
            self.v_scale.to_device(dev)?
        };
        match self.precision {
            CachePrecision::Fp16 => {
                if v_on_dev.dtype() != target_dtype {
                    v_on_dev.to_dtype(target_dtype).map_err(ModelError::from)
                } else {
                    Ok(v_on_dev)
                }
            }
            CachePrecision::Fp8 => {
                super::quantize::dequantize_fp8(&v_on_dev, &s_on_dev, target_dtype)
            }
            CachePrecision::Int8 => {
                super::quantize::dequantize_int8(&v_on_dev, &s_on_dev, target_dtype)
            }
        }
    }

    pub fn truncate(&mut self, target_len: usize) -> Result<()> {
        if target_len < self.num_tokens {
            self.k_quant = self.k_quant.narrow(2, 0, target_len)?.contiguous()?;
            self.v_quant = self.v_quant.narrow(2, 0, target_len)?.contiguous()?;
            self.num_tokens = target_len;
        }
        Ok(())
    }

    pub fn resident_bytes(&self) -> usize {
        let kq = self.k_quant.elem_count() * self.k_quant.dtype().size_in_bytes();
        let vq = self.v_quant.elem_count() * self.v_quant.dtype().size_in_bytes();
        let ks = self.k_scale.elem_count() * self.k_scale.dtype().size_in_bytes();
        let vs = self.v_scale.elem_count() * self.v_scale.dtype().size_in_bytes();
        kq + vq + ks + vs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_block_creation_and_residency() {
        let dev = Device::Cpu;
        let kq = Tensor::zeros((1, 2, 16, 64), DType::F8E4M3, &dev).unwrap();
        let vq = Tensor::zeros((1, 2, 16, 64), DType::F8E4M3, &dev).unwrap();
        let ks = Tensor::ones((1, 2, 1, 1), DType::F32, &dev).unwrap();
        let vs = Tensor::ones((1, 2, 1, 1), DType::F32, &dev).unwrap();

        let mut block = QuantizedCacheBlock::new(
            0,
            StorageTier::L1Vram,
            dev.clone(),
            kq,
            vq,
            ks,
            vs,
            16,
            CachePrecision::Fp8,
        );

        assert_eq!(block.num_tokens, 16);
        assert_eq!(block.precision, CachePrecision::Fp8);
        assert_eq!(block.tier, StorageTier::L1Vram);

        // 16 tokens * 2 heads * 64 dim * 1 byte = 2048 bytes for K, 2048 for V
        // scales = 2 * 4 bytes = 8 bytes each
        assert_eq!(block.resident_bytes(), 2048 + 2048 + 8 + 8);

        block.truncate(8).unwrap();
        assert_eq!(block.num_tokens, 8);
        assert_eq!(block.k_quant.dims(), &[1, 2, 8, 64]);
    }
}

//! Page table indexing, block allocation, append, and O(1) truncation.

use super::block::{BLOCK_SIZE, CachePrecision, QuantizedCacheBlock, StorageTier};
use super::quantize::{quantize_block_kv, quantize_fp8, quantize_int8};
use crate::error::{ModelError, Result};
use candle_core::{DType, Device, Tensor};

pub struct PagedKvCache {
    pub n_layer: usize,
    pub blocks: Vec<QuantizedCacheBlock>,
    pub layer_tables: Vec<Vec<usize>>,
    pub precision: CachePrecision,
    pub target_dtype: DType,
}

impl PagedKvCache {
    pub fn new(n_layer: usize) -> Self {
        Self::with_precision(n_layer, CachePrecision::Fp8)
    }

    pub fn with_precision(n_layer: usize, precision: CachePrecision) -> Self {
        Self {
            n_layer,
            blocks: Vec::new(),
            layer_tables: vec![Vec::new(); n_layer],
            precision,
            target_dtype: DType::F16,
        }
    }

    pub fn allocate_block(&mut self, dev: &Device, tier: StorageTier, k: Tensor, v: Tensor, n: usize) -> usize {
        self.allocate_block_with_offset(dev, tier, k, v, n, 0).unwrap_or(self.blocks.len())
    }

    pub fn allocate_block_with_offset(
        &mut self,
        dev: &Device,
        tier: StorageTier,
        k: Tensor,
        v: Tensor,
        num_tokens: usize,
        token_offset: usize,
    ) -> Result<usize> {
        let (kq, vq, ks, vs, prec) = quantize_block_kv(token_offset, &k, &v, self.precision)?;
        let id = self.blocks.len();
        self.blocks.push(QuantizedCacheBlock::new(id, tier, dev.clone(), kq, vq, ks, vs, num_tokens, prec));
        Ok(id)
    }

    pub fn layer_token_count(&self, layer: usize) -> usize {
        match self.layer_tables.get(layer) {
            Some(t) if !t.is_empty() => (t.len() - 1) * BLOCK_SIZE + self.blocks[t[t.len() - 1]].num_tokens,
            _ => 0,
        }
    }

    pub fn append_kv(&mut self, layer: usize, k: &Tensor, v: &Tensor, dev: &Device) -> Result<()> {
        if layer >= self.n_layer {
            return Err(ModelError::Config(format!("Invalid layer {layer}")));
        }
        let n_tokens = k.dim(2)?;
        if n_tokens == 0 {
            return Ok(());
        }
        self.target_dtype = k.dtype();
        let pin = std::env::var_os("RUNTIMED_KV_CACHE_PINNED").is_some() || std::env::var_os("RUNTIMED_KV_CACHE_HOST").is_some();
        let (adev, tier) = if pin { (&Device::Cpu, StorageTier::PinnedHost) } else { match dev { Device::Cuda(_) => (dev, StorageTier::L1Vram), _ => (&Device::Cpu, StorageTier::PinnedHost) } };

        let mut offset = 0;
        while offset < n_tokens {
            let last_info = self.layer_tables[layer]
                .last().copied().filter(|&b| self.blocks[b].num_tokens < BLOCK_SIZE);

            if let Some(last_bid) = last_info {
                let take = (BLOCK_SIZE - self.blocks[last_bid].num_tokens).min(n_tokens - offset);
                self.append_to_block(last_bid, &k.narrow(2, offset, take)?, &v.narrow(2, offset, take)?, take)?;
                offset += take;
            } else {
                let cur = self.layer_token_count(layer);
                let take = BLOCK_SIZE.min(n_tokens - offset);
                let new_id = self.allocate_block_with_offset(
                    adev, tier, k.narrow(2, offset, take)?, v.narrow(2, offset, take)?, take, cur,
                )?;
                self.layer_tables[layer].push(new_id);
                offset += take;
            }
        }
        Ok(())
    }

    fn append_to_block(&mut self, bid: usize, k: &Tensor, v: &Tensor, take: usize) -> Result<()> {
        let b = &mut self.blocks[bid];
        let target_dev = b.device.clone();
        match b.precision {
            CachePrecision::Fp16 => {
                let k16 = if k.dtype() == DType::F16 { k.clone() } else { k.to_dtype(DType::F16)? };
                let v16 = if v.dtype() == DType::F16 { v.clone() } else { v.to_dtype(DType::F16)? };
                let k16 = if !k16.device().same_device(&target_dev) { k16.to_device(&target_dev)? } else { k16 };
                let v16 = if !v16.device().same_device(&target_dev) { v16.to_device(&target_dev)? } else { v16 };
                b.k_quant = Tensor::cat(&[&b.k_quant.contiguous()?, &k16.contiguous()?], 2)?;
                b.v_quant = Tensor::cat(&[&b.v_quant.contiguous()?, &v16.contiguous()?], 2)?;
            }
            CachePrecision::Int8 | CachePrecision::Fp8 => {
                let k32 = if k.dtype() == DType::F32 { k.clone() } else { k.to_dtype(DType::F32)? };
                let v32 = if v.dtype() == DType::F32 { v.clone() } else { v.to_dtype(DType::F32)? };
                let k32 = if !k32.device().same_device(&target_dev) { k32.to_device(&target_dev)? } else { k32 };
                let v32 = if !v32.device().same_device(&target_dev) { v32.to_device(&target_dev)? } else { v32 };
                let ck = Tensor::cat(&[&b.dequantize_k(&target_dev, DType::F32)?.contiguous()?, &k32.contiguous()?], 2)?;
                let cv = Tensor::cat(&[&b.dequantize_v(&target_dev, DType::F32)?.contiguous()?, &v32.contiguous()?], 2)?;
                let (kq, ks) = if b.precision == CachePrecision::Int8 { quantize_int8(&ck)? } else { quantize_fp8(&ck)? };
                let (vq, vs) = if b.precision == CachePrecision::Int8 { quantize_int8(&cv)? } else { quantize_fp8(&cv)? };
                b.k_quant = kq; b.v_quant = vq; b.k_scale = ks; b.v_scale = vs;
            }
        }
        b.num_tokens += take;
        Ok(())
    }

    pub fn truncate(&mut self, target_len: usize) -> Result<()> {
        for layer in 0..self.n_layer {
            if target_len == 0 { self.layer_tables[layer].clear(); continue; }
            let cur = self.layer_token_count(layer);
            if target_len >= cur { continue; }
            let needed = target_len.div_ceil(BLOCK_SIZE);
            if needed < self.layer_tables[layer].len() { self.layer_tables[layer].truncate(needed); }
            if needed > 0 {
                let rem = target_len % BLOCK_SIZE;
                self.blocks[self.layer_tables[layer][needed - 1]].truncate(if rem == 0 { BLOCK_SIZE } else { rem })?;
            }
        }
        Ok(())
    }

    pub fn assemble_layer_kv(&self, layer: usize, dev: &Device) -> Result<Option<(Tensor, Tensor)>> {
        if layer >= self.n_layer || self.layer_tables[layer].is_empty() { return Ok(None); }
        let valid: Vec<usize> = self.layer_tables[layer].iter().copied().filter(|&b| self.blocks[b].num_tokens > 0).collect();
        if valid.is_empty() { return Ok(None); }
        let (mut ks, mut vs) = (Vec::with_capacity(valid.len()), Vec::with_capacity(valid.len()));
        for b in valid {
            ks.push(self.blocks[b].dequantize_k(dev, self.target_dtype)?);
            vs.push(self.blocks[b].dequantize_v(dev, self.target_dtype)?);
        }
        let (kr, vr): (Vec<&Tensor>, Vec<&Tensor>) = (ks.iter().collect(), vs.iter().collect());
        Ok(Some((Tensor::cat(&kr, 2)?, Tensor::cat(&vr, 2)?)))
    }

    pub fn assemble_layer_kv_rope(
        &self,
        layer: usize,
        dev: &Device,
        positions: &[usize],
        theta: f32,
        rot_dim: usize,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let Some((full_k, full_v)) = self.assemble_layer_kv(layer, dev)? else { return Ok(None); };
        let rotated_k = crate::ops::rope_neox_pos(&full_k, positions, theta, rot_dim, None)?;
        Ok(Some((rotated_k, full_v)))
    }

    pub fn count_tier_blocks(&self, tier: StorageTier) -> usize {
        self.blocks.iter().filter(|b| if tier.is_host() { b.tier.is_host() } else { b.tier == tier }).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_kv_paging_spans_multiple_blocks() {
        let mut cache = PagedKvCache::new(2);
        let dev = Device::Cpu;
        let k = Tensor::zeros((1, 2, 25, 64), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 2, 25, 64), DType::F32, &dev).unwrap();
        cache.append_kv(0, &k, &v, &dev).unwrap();

        assert_eq!(cache.layer_tables[0].len(), 2);
        assert_eq!(cache.blocks[0].num_tokens, 16);
        assert_eq!(cache.blocks[0].precision, CachePrecision::Fp16);
        assert_eq!(cache.blocks[1].num_tokens, 9);
        assert_eq!(cache.blocks[1].precision, CachePrecision::Fp8);

        let k1 = Tensor::zeros((1, 2, 1, 64), DType::F32, &dev).unwrap();
        let v1 = Tensor::zeros((1, 2, 1, 64), DType::F32, &dev).unwrap();
        cache.append_kv(0, &k1, &v1, &dev).unwrap();
        assert_eq!(cache.blocks[1].num_tokens, 10);

        let (full_k, full_v) = cache.assemble_layer_kv(0, &dev).unwrap().unwrap();
        assert_eq!(full_k.dims(), &[1, 2, 26, 64]);
        assert_eq!(full_v.dims(), &[1, 2, 26, 64]);
    }

    #[test]
    fn test_truncate_kv_rollback_and_subsequent_append() {
        let mut cache = PagedKvCache::new(1);
        let dev = Device::Cpu;
        let k = Tensor::zeros((1, 2, 35, 64), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 2, 35, 64), DType::F32, &dev).unwrap();
        cache.append_kv(0, &k, &v, &dev).unwrap();
        assert_eq!(cache.layer_tables[0].len(), 3);

        // Rollback to 20 tokens (block 0: 16, block 1: 4)
        cache.truncate(20).unwrap();
        assert_eq!(cache.layer_tables[0].len(), 2);
        assert_eq!(cache.blocks[cache.layer_tables[0][1]].num_tokens, 4);

        // Append 2 more tokens into truncated block 1
        let k_extra = Tensor::zeros((1, 2, 2, 64), DType::F32, &dev).unwrap();
        let v_extra = Tensor::zeros((1, 2, 2, 64), DType::F32, &dev).unwrap();
        cache.append_kv(0, &k_extra, &v_extra, &dev).unwrap();
        assert_eq!(cache.layer_token_count(0), 22);

        let (full_k, _) = cache.assemble_layer_kv(0, &dev).unwrap().unwrap();
        assert_eq!(full_k.dims(), &[1, 2, 22, 64]);
    }

    #[test]
    fn test_int8_cache_append_and_assemble() {
        let mut cache = PagedKvCache::with_precision(1, CachePrecision::Int8);
        let dev = Device::Cpu;
        let k = Tensor::zeros((1, 2, 20, 64), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 2, 20, 64), DType::F32, &dev).unwrap();
        cache.append_kv(0, &k, &v, &dev).unwrap();
        assert_eq!(cache.blocks[1].precision, CachePrecision::Int8);
        let (full_k, full_v) = cache.assemble_layer_kv(0, &dev).unwrap().unwrap();
        assert_eq!(full_k.dims(), &[1, 2, 20, 64]);
        assert_eq!(full_v.dims(), &[1, 2, 20, 64]);
    }

    #[test]
    fn test_assemble_layer_kv_rope() {
        let mut cache = PagedKvCache::new(1);
        let dev = Device::Cpu;
        let (k, v) = (Tensor::zeros((1, 1, 2, 4), DType::F32, &dev).unwrap(), Tensor::zeros((1, 1, 2, 4), DType::F32, &dev).unwrap());
        cache.append_kv(0, &k, &v, &dev).unwrap();
        let (rk, _) = cache.assemble_layer_kv_rope(0, &dev, &[0, 1], 10000.0, 4).unwrap().unwrap();
        assert_eq!(rk.dims(), &[1, 1, 2, 4]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_append_and_assemble() {
        if let Ok(dev) = Device::new_cuda(0) {
            let mut cache = PagedKvCache::new(1);
            let (k, v) = (Tensor::zeros((1, 2, 20, 64), DType::F16, &dev).unwrap(), Tensor::zeros((1, 2, 20, 64), DType::F16, &dev).unwrap());
            cache.append_kv(0, &k, &v, &dev).unwrap();
            assert_eq!(cache.layer_tables[0].len(), 2);
            assert_eq!(cache.blocks[0].precision, CachePrecision::Fp16);
            assert_eq!(cache.blocks[1].precision, CachePrecision::Fp8);
            let (full_k, full_v) = cache.assemble_layer_kv(0, &dev).unwrap().unwrap();
            assert_eq!(full_k.dims(), &[1, 2, 20, 64]);
            assert_eq!(full_v.dims(), &[1, 2, 20, 64]);
        }
    }
}

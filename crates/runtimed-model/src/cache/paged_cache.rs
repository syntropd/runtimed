//! Block-based two-tier paged KV cache (16 tokens per block).

use crate::error::{ModelError, Result};
use candle_core::{Device, Tensor};

pub const BLOCK_SIZE: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTier {
    L1Vram,
    L2PinnedHost,
}

#[derive(Clone)]
pub struct CacheBlock {
    pub block_id: usize,
    pub tier: StorageTier,
    pub device: Device,
    pub k: Tensor,
    pub v: Tensor,
    pub num_tokens: usize,
}

pub struct PagedKvCache {
    pub n_layer: usize,
    pub blocks: Vec<CacheBlock>,
    pub layer_tables: Vec<Vec<usize>>,
}

impl PagedKvCache {
    pub fn new(n_layer: usize) -> Self {
        Self {
            n_layer,
            blocks: Vec::new(),
            layer_tables: vec![Vec::new(); n_layer],
        }
    }

    pub fn allocate_block(
        &mut self,
        dev: &Device,
        tier: StorageTier,
        init_k: Tensor,
        init_v: Tensor,
        num_tokens: usize,
    ) -> usize {
        let block_id = self.blocks.len();
        self.blocks.push(CacheBlock {
            block_id,
            tier,
            device: dev.clone(),
            k: init_k,
            v: init_v,
            num_tokens,
        });
        block_id
    }

    pub fn append_kv(&mut self, layer: usize, k: &Tensor, v: &Tensor, dev: &Device) -> Result<()> {
        if layer >= self.n_layer {
            return Err(ModelError::Config(format!("Invalid layer {layer}")));
        }

        let n_tokens = k.dim(2)?;
        if n_tokens == 0 {
            return Ok(());
        }

        let tier = match dev {
            Device::Cuda(_) => StorageTier::L1Vram,
            _ => StorageTier::L2PinnedHost,
        };

        let mut offset = 0;
        while offset < n_tokens {
            let can_append = if let Some(&last_bid) = self.layer_tables[layer].last() {
                self.blocks[last_bid].num_tokens < BLOCK_SIZE
            } else {
                false
            };

            if can_append {
                let last_bid = *self.layer_tables[layer].last().unwrap();
                let block = &mut self.blocks[last_bid];
                let space = BLOCK_SIZE - block.num_tokens;
                let take = space.min(n_tokens - offset);
                let k_chunk = k.narrow(2, offset, take)?;
                let v_chunk = v.narrow(2, offset, take)?;
                block.k = Tensor::cat(&[&block.k, &k_chunk], 2)?;
                block.v = Tensor::cat(&[&block.v, &v_chunk], 2)?;
                block.num_tokens += take;
                offset += take;
            } else {
                let take = BLOCK_SIZE.min(n_tokens - offset);
                let k_chunk = k.narrow(2, offset, take)?;
                let v_chunk = v.narrow(2, offset, take)?;
                let new_id = self.allocate_block(dev, tier, k_chunk, v_chunk, take);
                self.layer_tables[layer].push(new_id);
                offset += take;
            }
        }
        Ok(())
    }

    pub fn assemble_layer_kv(&self, layer: usize, dev: &Device) -> Result<Option<(Tensor, Tensor)>> {
        if layer >= self.n_layer || self.layer_tables[layer].is_empty() {
            return Ok(None);
        }

        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for &bid in &self.layer_tables[layer] {
            let block = &self.blocks[bid];
            let k = if block.device.same_device(dev) {
                block.k.clone()
            } else {
                block.k.to_device(dev)?
            };
            let v = if block.device.same_device(dev) {
                block.v.clone()
            } else {
                block.v.to_device(dev)?
            };
            ks.push(k);
            vs.push(v);
        }

        let k_refs: Vec<&Tensor> = ks.iter().collect();
        let v_refs: Vec<&Tensor> = vs.iter().collect();
        let full_k = Tensor::cat(&k_refs, 2)?;
        let full_v = Tensor::cat(&v_refs, 2)?;
        Ok(Some((full_k, full_v)))
    }

    pub fn count_tier_blocks(&self, tier: StorageTier) -> usize {
        self.blocks.iter().filter(|b| b.tier == tier).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_append_kv_paging_spans_multiple_blocks() {
        let mut cache = PagedKvCache::new(2);
        // Append 25 tokens in one call (e.g. prefill)
        let k = Tensor::zeros((1, 2, 25, 64), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((1, 2, 25, 64), DType::F32, &Device::Cpu).unwrap();

        cache.append_kv(0, &k, &v, &Device::Cpu).unwrap();

        // 25 tokens should span 2 blocks: 16 in block 0, 9 in block 1
        assert_eq!(cache.layer_tables[0].len(), 2);
        assert_eq!(cache.blocks[0].num_tokens, 16);
        assert_eq!(cache.blocks[1].num_tokens, 9);

        // Append 1 more token
        let k1 = Tensor::zeros((1, 2, 1, 64), DType::F32, &Device::Cpu).unwrap();
        let v1 = Tensor::zeros((1, 2, 1, 64), DType::F32, &Device::Cpu).unwrap();
        cache.append_kv(0, &k1, &v1, &Device::Cpu).unwrap();

        // Still 2 blocks: block 1 now has 10 tokens
        assert_eq!(cache.layer_tables[0].len(), 2);
        assert_eq!(cache.blocks[1].num_tokens, 10);

        let (full_k, full_v) = cache.assemble_layer_kv(0, &Device::Cpu).unwrap().unwrap();
        assert_eq!(full_k.dims(), &[1, 2, 26, 64]);
        assert_eq!(full_v.dims(), &[1, 2, 26, 64]);
    }
}

//! JIT attention layer prefetcher with circular double-buffered staging.
//!
//! Employs two circular staging slots to pipeline host-to-device KV cache
//! and attention tensor transfers concurrently with layer compute.

use crate::cache::paged_cache::{PagedKvCache, StorageTier};
use crate::error::Result;
use candle_core::{Device, Tensor};

#[derive(Clone)]
pub struct StagedLayerBuffer {
    pub layer_idx: usize,
    pub k: Tensor,
    pub v: Tensor,
    pub device: Device,
    pub is_page_locked_transfer: bool,
}

/// Circular double-buffered staging controller for JIT attention layer prefetching.
pub struct JitLayerPrefetcher {
    pub target_device: Device,
    staging_buffers: [Option<StagedLayerBuffer>; 2],
    active_slot: usize,
    prefetch_count: usize,
    cache_hits: usize,
}

impl JitLayerPrefetcher {
    pub fn new(target_device: Device) -> Self {
        Self {
            target_device,
            staging_buffers: [None, None],
            active_slot: 0,
            prefetch_count: 0,
            cache_hits: 0,
        }
    }

    pub fn active_slot_idx(&self) -> usize {
        self.active_slot
    }

    pub fn staging_slot_idx(&self) -> usize {
        (self.active_slot + 1) % 2
    }

    pub fn metrics(&self) -> (usize, usize) {
        (self.prefetch_count, self.cache_hits)
    }

    /// Prefetch host attention tensors (page-locked CPU memory) to target device staging slot.
    pub fn prefetch_layer(
        &mut self,
        layer_idx: usize,
        k_host: &Tensor,
        v_host: &Tensor,
    ) -> Result<()> {
        let is_host_transfer = !k_host.device().same_device(&self.target_device);
        let k_dev = if is_host_transfer {
            k_host.to_device(&self.target_device)?
        } else {
            k_host.clone()
        };
        let v_dev = if is_host_transfer {
            v_host.to_device(&self.target_device)?
        } else {
            v_host.clone()
        };

        let staging_idx = self.staging_slot_idx();
        self.staging_buffers[staging_idx] = Some(StagedLayerBuffer {
            layer_idx,
            k: k_dev,
            v: v_dev,
            device: self.target_device.clone(),
            is_page_locked_transfer: is_host_transfer,
        });

        self.prefetch_count += 1;
        Ok(())
    }

    /// Prefetch any L2 pinned host blocks for the designated layer from PagedKvCache.
    pub fn prefetch_from_cache(
        &mut self,
        cache: &PagedKvCache,
        layer_idx: usize,
    ) -> Result<bool> {
        let block_ids = match cache.layer_tables.get(layer_idx) {
            Some(ids) if !ids.is_empty() => ids,
            _ => return Ok(false),
        };

        let has_l2 = block_ids.iter().any(|&bid| {
            cache.blocks.get(bid).is_some_and(|b| b.tier == StorageTier::L2PinnedHost)
        });

        if !has_l2 {
            return Ok(false);
        }

        if let Some((k, v)) = cache.assemble_layer_kv(layer_idx, &self.target_device)? {
            let staging_idx = self.staging_slot_idx();
            self.staging_buffers[staging_idx] = Some(StagedLayerBuffer {
                layer_idx,
                k,
                v,
                device: self.target_device.clone(),
                is_page_locked_transfer: true,
            });
            self.prefetch_count += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Acquire the currently active layer staging buffer if matching `expected_layer`.
    pub fn acquire_active(&mut self, expected_layer: usize) -> Option<StagedLayerBuffer> {
        if let Some(ref buf) = self.staging_buffers[self.active_slot] {
            if buf.layer_idx == expected_layer {
                self.cache_hits += 1;
                return self.staging_buffers[self.active_slot].clone();
            }
        }

        // Check if upcoming staging slot already holds it
        let staging_idx = self.staging_slot_idx();
        if let Some(ref buf) = self.staging_buffers[staging_idx] {
            if buf.layer_idx == expected_layer {
                self.advance();
                self.cache_hits += 1;
                return self.staging_buffers[self.active_slot].clone();
            }
        }

        None
    }

    /// Advance circular double buffer: swaps active and staging slots and cleans old slot.
    pub fn advance(&mut self) {
        self.active_slot = self.staging_slot_idx();
        let new_staging = self.staging_slot_idx();
        self.staging_buffers[new_staging] = None;
    }

    /// Clear all staging buffers and reset slot index.
    pub fn reset(&mut self) {
        self.staging_buffers = [None, None];
        self.active_slot = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_circular_slot_progression() {
        let mut prefetcher = JitLayerPrefetcher::new(Device::Cpu);
        assert_eq!(prefetcher.active_slot_idx(), 0);
        assert_eq!(prefetcher.staging_slot_idx(), 1);

        prefetcher.advance();
        assert_eq!(prefetcher.active_slot_idx(), 1);
        assert_eq!(prefetcher.staging_slot_idx(), 0);

        prefetcher.advance();
        assert_eq!(prefetcher.active_slot_idx(), 0);
        assert_eq!(prefetcher.staging_slot_idx(), 1);
    }

    #[test]
    fn test_prefetch_and_acquire_pipeline() {
        let mut prefetcher = JitLayerPrefetcher::new(Device::Cpu);
        let k0 = Tensor::zeros((1, 4, 16, 64), DType::F32, &Device::Cpu).unwrap();
        let v0 = Tensor::ones((1, 4, 16, 64), DType::F32, &Device::Cpu).unwrap();

        // Stage layer 1 into staging slot (1)
        prefetcher.prefetch_layer(1, &k0, &v0).unwrap();
        assert_eq!(prefetcher.metrics().0, 1);

        // Advance to make layer 1 active
        prefetcher.advance();
        assert_eq!(prefetcher.active_slot_idx(), 1);

        // Acquire layer 1 from active slot
        let staged = prefetcher.acquire_active(1).unwrap();
        assert_eq!(staged.layer_idx, 1);
        assert_eq!(prefetcher.metrics().1, 1);

        // Stage layer 2 into staging slot (0)
        prefetcher.prefetch_layer(2, &k0, &v0).unwrap();
        let staged2 = prefetcher.acquire_active(2).unwrap();
        assert_eq!(staged2.layer_idx, 2);
        assert_eq!(prefetcher.metrics().1, 2);
    }

    #[test]
    fn test_prefetch_from_paged_cache_l2() {
        let mut cache = PagedKvCache::new(4);
        let k1 = Tensor::zeros((1, 2, 16, 32), DType::F32, &Device::Cpu).unwrap();
        let v1 = Tensor::ones((1, 2, 16, 32), DType::F32, &Device::Cpu).unwrap();
        let k2 = Tensor::zeros((1, 2, 16, 32), DType::F32, &Device::Cpu).unwrap();
        let v2 = Tensor::ones((1, 2, 16, 32), DType::F32, &Device::Cpu).unwrap();

        let b0 = cache.allocate_block(&Device::Cpu, StorageTier::L2PinnedHost, k1, v1, 16);
        let b1 = cache.allocate_block(&Device::Cpu, StorageTier::L2PinnedHost, k2, v2, 16);
        cache.layer_tables[2].push(b0);
        cache.layer_tables[2].push(b1);

        let mut prefetcher = JitLayerPrefetcher::new(Device::Cpu);
        let staged_ok = prefetcher.prefetch_from_cache(&cache, 2).unwrap();
        assert!(staged_ok);

        let staged = prefetcher.acquire_active(2).unwrap();
        assert_eq!(staged.layer_idx, 2);
        assert_eq!(staged.k.dim(2).unwrap(), 32);
        assert_eq!(staged.v.dim(2).unwrap(), 32);
    }
}

use super::paged::{PagedKvCache, StorageTier};
use crate::error::Result;
use candle_core::Device;

pub struct SpillManager {
    pub max_l1_blocks: usize,
}

impl SpillManager {
    pub fn new(max_l1_blocks: usize) -> Self {
        Self { max_l1_blocks }
    }

    /// Spill up to `count` oldest L1 VRAM blocks to L2 pinned host RAM (CPU).
    pub fn spill_blocks(&self, cache: &mut PagedKvCache, count: usize) -> Result<usize> {
        let mut spilled = 0;
        for block in &mut cache.blocks {
            if spilled >= count {
                break;
            }
            if block.tier == StorageTier::L1Vram {
                block.to_device(&Device::Cpu)?;
                block.tier = StorageTier::L2PinnedHost;
                spilled += 1;
            }
        }
        Ok(spilled)
    }

    /// Prefetch up to `count` L2 blocks back to target L1 GPU device.
    pub fn prefetch_blocks(&self, cache: &mut PagedKvCache, target_dev: &Device, count: usize) -> Result<usize> {
        let mut prefetched = 0;
        for block in &mut cache.blocks {
            if prefetched >= count {
                break;
            }
            if block.tier == StorageTier::L2PinnedHost {
                block.to_device(target_dev)?;
                block.tier = StorageTier::L1Vram;
                prefetched += 1;
            }
        }
        Ok(prefetched)
    }

    /// Proactively spill under kernel PSI memory pressure spike.
    pub fn shed_pressure_spill(&self, cache: &mut PagedKvCache, fraction: f32) -> Result<usize> {
        let l1_count = cache.count_tier_blocks(StorageTier::L1Vram);
        let target = ((l1_count as f32) * fraction.clamp(0.0, 1.0)).ceil() as usize;
        self.spill_blocks(cache, target)
    }

    /// Enforce `max_l1_blocks` ceiling by spilling excess L1 blocks to L2 host RAM.
    pub fn enforce_capacity(&self, cache: &mut PagedKvCache) -> Result<usize> {
        let current_l1 = cache.count_tier_blocks(StorageTier::L1Vram);
        if current_l1 > self.max_l1_blocks {
            self.spill_blocks(cache, current_l1 - self.max_l1_blocks)
        } else {
            Ok(0)
        }
    }

    /// Selectively spill blocks only from a saturated accelerator device to CPU host RAM.
    pub fn spill_device_blocks(
        &self,
        cache: &mut PagedKvCache,
        device: &Device,
        count: usize,
    ) -> Result<usize> {
        let mut spilled = 0;
        for block in &mut cache.blocks {
            if spilled >= count {
                break;
            }
            if block.tier == StorageTier::L1Vram && block.origin_device.same_device(device) {
                block.to_device(&Device::Cpu)?;
                block.tier = StorageTier::L2PinnedHost;
                spilled += 1;
            }
        }
        Ok(spilled)
    }

    /// Restore spilled L2 blocks back to their original accelerator device.
    pub fn restore_origin_blocks(&self, cache: &mut PagedKvCache, count: usize) -> Result<usize> {
        let mut restored = 0;
        for block in &mut cache.blocks {
            if restored >= count {
                break;
            }
            if block.tier == StorageTier::L2PinnedHost {
                let origin = block.origin_device.clone();
                block.to_device(&origin)?;
                block.tier = StorageTier::L1Vram;
                restored += 1;
            }
        }
        Ok(restored)
    }

    /// Restore spilled L2 blocks specifically belonging to a target accelerator device.
    pub fn restore_device_blocks(
        &self,
        cache: &mut PagedKvCache,
        device: &Device,
        count: usize,
    ) -> Result<usize> {
        let mut restored = 0;
        for block in &mut cache.blocks {
            if restored >= count {
                break;
            }
            if block.tier == StorageTier::L2PinnedHost && block.origin_device.same_device(device) {
                let origin = block.origin_device.clone();
                block.to_device(&origin)?;
                block.tier = StorageTier::L1Vram;
                restored += 1;
            }
        }
        Ok(restored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};

    #[test]
    fn test_spill_and_prefetch_cycle() {
        let mut cache = PagedKvCache::new(1);
        let dummy_k = Tensor::zeros((1, 2, 1, 64), DType::F32, &Device::Cpu).unwrap();
        let dummy_v = Tensor::zeros((1, 2, 1, 64), DType::F32, &Device::Cpu).unwrap();
        cache.allocate_block(&Device::Cpu, StorageTier::L1Vram, dummy_k, dummy_v, 1);

        let manager = SpillManager::new(10);
        assert_eq!(cache.count_tier_blocks(StorageTier::L1Vram), 1);
        assert_eq!(cache.count_tier_blocks(StorageTier::L2PinnedHost), 0);

        let spilled = manager.spill_blocks(&mut cache, 1).unwrap();
        assert_eq!(spilled, 1);
        assert_eq!(cache.count_tier_blocks(StorageTier::L1Vram), 0);
        assert_eq!(cache.count_tier_blocks(StorageTier::L2PinnedHost), 1);

        let prefetched = manager.prefetch_blocks(&mut cache, &Device::Cpu, 1).unwrap();
        assert_eq!(prefetched, 1);
        assert_eq!(cache.count_tier_blocks(StorageTier::L1Vram), 1);

        // Test capacity enforcement
        let strict_mgr = SpillManager::new(0);
        let auto_spilled = strict_mgr.enforce_capacity(&mut cache).unwrap();
        assert_eq!(auto_spilled, 1);
        assert_eq!(cache.count_tier_blocks(StorageTier::L1Vram), 0);
    }

    #[test]
    fn test_selective_device_shedding_and_restore() {
        let mut cache = PagedKvCache::new(1);
        let k1 = Tensor::zeros((1, 2, 1, 64), DType::F32, &Device::Cpu).unwrap();
        let v1 = Tensor::zeros((1, 2, 1, 64), DType::F32, &Device::Cpu).unwrap();
        let k2 = Tensor::zeros((1, 2, 1, 64), DType::F32, &Device::Cpu).unwrap();
        let v2 = Tensor::zeros((1, 2, 1, 64), DType::F32, &Device::Cpu).unwrap();

        cache.allocate_block(&Device::Cpu, StorageTier::L1Vram, k1, v1, 1);
        cache.allocate_block(&Device::Cpu, StorageTier::L1Vram, k2, v2, 1);

        let manager = SpillManager::new(10);
        let spilled = manager.spill_device_blocks(&mut cache, &Device::Cpu, 1).unwrap();
        assert_eq!(spilled, 1);
        assert_eq!(cache.count_tier_blocks(StorageTier::L1Vram), 1);
        assert_eq!(cache.count_tier_blocks(StorageTier::L2PinnedHost), 1);

        let restored = manager.restore_device_blocks(&mut cache, &Device::Cpu, 1).unwrap();
        assert_eq!(restored, 1);
        assert_eq!(cache.count_tier_blocks(StorageTier::L1Vram), 2);
    }
}

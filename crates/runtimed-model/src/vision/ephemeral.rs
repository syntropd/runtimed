//! Ephemeral vision tower lifecycle management.
//!
//! ViT encoder weights are pinned to the accelerator device during prompt
//! prefill, then evicted to host memory (CPU) via `unpin()` immediately
//! prior to text token decoding, freeing 20%-30% VRAM.

use crate::error::Result;
use crate::vision::VisionTower;
use candle_core::Device;

impl VisionTower {
    /// Pin vision tower weights to target accelerator device.
    pub fn pin(&mut self, dev: &Device) -> Result<usize> {
        self.w.to_device(dev)
    }

    /// Evict vision tower weights to CPU to reclaim accelerator VRAM post-prefill.
    pub fn unpin(&mut self) -> Result<usize> {
        self.w.to_device(&Device::Cpu)
    }

    /// Resident device of the vision tower weights.
    pub fn device(&self) -> &Device {
        self.w.device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::VisionConfig;
    use crate::weights::Weights;
    use candle_core::{DType, Tensor};
    use std::collections::HashMap;

    fn dummy_tower() -> VisionTower {
        let cfg = VisionConfig {
            n_layer: 1,
            hidden: 4,
            n_head: 1,
            head_dim: 4,
            ffn: 4,
            eps: 1e-5,
            patch: 2,
            merge: 1,
            min_soft: 1,
            max_soft: 8,
            proj_dim: 4,
        };
        let mut map = HashMap::new();
        let t = Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap();
        map.insert("v.patch_embd.weight".into(), t);
        let w = Weights::from_parts(Device::Cpu, DType::F32, map);
        VisionTower { cfg, w }
    }

    #[test]
    fn test_ephemeral_pin_unpin_lifecycle() {
        let mut tower = dummy_tower();
        assert!(matches!(tower.device(), Device::Cpu));

        // Pin to CPU (as accelerator placeholder in unit test).
        let pinned_bytes = tower.pin(&Device::Cpu).unwrap();
        assert_eq!(pinned_bytes, 4 * 4 * 4);
        assert!(matches!(tower.device(), Device::Cpu));

        // Unpin back to host CPU.
        let unpinned_bytes = tower.unpin().unwrap();
        assert_eq!(unpinned_bytes, 4 * 4 * 4);
        assert!(matches!(tower.device(), Device::Cpu));

        // Repeated unpin when already on CPU is idempotent.
        let unpinned_again = tower.unpin().unwrap();
        assert_eq!(unpinned_again, 4 * 4 * 4);
        assert!(matches!(tower.device(), Device::Cpu));
    }
}

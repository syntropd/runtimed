//! Two-tier block-based paged KV cache (L1 VRAM + L2 pinned host RAM).

pub mod image_cache;
pub mod layer_prefetch;
pub mod paged_cache;
pub mod sink_mask;
pub mod sink_window;
pub mod spill_manager;

pub use image_cache::{ImageCache, ImageCacheKey, VisualKvEntry};
pub use layer_prefetch::{JitLayerPrefetcher, StagedLayerBuffer};
pub use paged_cache::{CacheBlock, PagedKvCache, StorageTier, BLOCK_SIZE};
pub use sink_mask::sink_causal_mask;
pub use sink_window::{SinkWindowCache, SinkWindowConfig};
pub use spill_manager::SpillManager;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_size_constant() {
        assert_eq!(BLOCK_SIZE, 16);
    }
}

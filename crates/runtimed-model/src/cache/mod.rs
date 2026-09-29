//! Two-tier block-based paged KV cache (L1 VRAM + L2 pinned host RAM).

pub mod paged_cache;
pub mod spill_manager;

pub use paged_cache::{CacheBlock, PagedKvCache, StorageTier, BLOCK_SIZE};
pub use spill_manager::SpillManager;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_size_constant() {
        assert_eq!(BLOCK_SIZE, 16);
    }
}

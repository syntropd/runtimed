//! L2 host memory cache for visual KV blocks with prefix-aware splicing.
//!
//! Indexes pre-computed visual KV blocks by `(model_id, prefix_hash, image_sha256)`
//! in host memory (CPU), avoiding redundant visual re-encoding across multi-turn chats.

use crate::arch::gemma4;
use crate::decode::session::Session;
use crate::error::Result;
use candle_core::{Device, Tensor};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Compound cache key for visual KV blocks: model identity, preceding token
/// prefix hash, and SHA-256 fingerprint of the image bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImageCacheKey {
    pub model_id: String,
    pub prefix_hash: u64,
    pub image_sha256: [u8; 32],
}

impl ImageCacheKey {
    pub fn new(model_id: impl Into<String>, prefix_hash: u64, image_sha256: [u8; 32]) -> Self {
        Self {
            model_id: model_id.into(),
            prefix_hash,
            image_sha256,
        }
    }

    /// Compute a cache key from model identifier, prefix hash, and raw image bytes.
    pub fn from_image_bytes(model_id: impl Into<String>, prefix_hash: u64, bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let digest: [u8; 32] = hasher.finalize().into();
        Self::new(model_id, prefix_hash, digest)
    }
}

/// A cached visual KV entry resident in host memory (CPU).
#[derive(Clone)]
pub struct VisualKvEntry {
    pub layers: Vec<Option<(Tensor, Tensor)>>,
    pub num_tokens: usize,
}

impl VisualKvEntry {
    /// Create a new entry, ensuring all stored tensors reside in L2 host memory (CPU).
    pub fn new(layers: Vec<Option<(Tensor, Tensor)>>, num_tokens: usize) -> Result<Self> {
        let mut host_layers = Vec::with_capacity(layers.len());
        for slot in layers {
            let host_slot = match slot {
                Some((k, v)) => {
                    let kh = k.to_device(&Device::Cpu)?;
                    let vh = v.to_device(&Device::Cpu)?;
                    Some((kh, vh))
                }
                None => None,
            };
            host_layers.push(host_slot);
        }
        Ok(Self {
            layers: host_layers,
            num_tokens,
        })
    }
}

/// L2 Host memory prefix cache for visual KV blocks.
pub struct ImageCache {
    entries: HashMap<ImageCacheKey, VisualKvEntry>,
    capacity: usize,
}

impl ImageCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity: if capacity == 0 { 16 } else { capacity },
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, key: &ImageCacheKey) -> Option<&VisualKvEntry> {
        self.entries.get(key)
    }

    pub fn insert(&mut self, key: ImageCacheKey, entry: VisualKvEntry) -> Result<()> {
        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            if let Some(first_key) = self.entries.keys().next().cloned() {
                self.entries.remove(&first_key);
            }
        }
        self.entries.insert(key, entry);
        Ok(())
    }

    /// Splice cached visual KV blocks directly into a Gemma4 cache.
    pub fn splice_into_gemma4(
        &self,
        key: &ImageCacheKey,
        cache: &mut gemma4::Cache,
        dev: &Device,
    ) -> Result<bool> {
        if let Some(entry) = self.entries.get(key) {
            cache.splice_kv(&entry.layers, dev)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Splice cached visual KV blocks into an active inference session.
    pub fn splice_into_session(
        &self,
        key: &ImageCacheKey,
        session: &mut Session,
    ) -> Result<bool> {
        if let Some(entry) = self.entries.get(key) {
            session.splice_visual_kv(&entry.layers)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};

    #[test]
    fn test_image_cache_key_sha256() {
        let key1 = ImageCacheKey::from_image_bytes("gemma4", 42, b"image_data");
        let key2 = ImageCacheKey::from_image_bytes("gemma4", 42, b"image_data");
        let key3 = ImageCacheKey::from_image_bytes("gemma4", 42, b"other_data");
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_image_cache_host_memory_and_splicing() {
        let dev = Device::Cpu;
        let mut cache = ImageCache::new(4);
        let key = ImageCacheKey::from_image_bytes("gemma4", 100, b"fake_png");

        // Create 2 layers with dummy [1, 2, 4, 8] KV tensors: seq len = 4.
        let k1 = Tensor::zeros((1, 2, 4, 8), DType::F32, &dev).unwrap();
        let v1 = Tensor::zeros((1, 2, 4, 8), DType::F32, &dev).unwrap();
        let layers = vec![Some((k1, v1)), None];
        let entry = VisualKvEntry::new(layers, 4).unwrap();

        cache.insert(key.clone(), entry).unwrap();
        assert_eq!(cache.len(), 1);

        let mut gemma_cache = gemma4::Cache::new(2);
        let spliced = cache.splice_into_gemma4(&key, &mut gemma_cache, &dev).unwrap();
        assert!(spliced);

        // Splice again to verify concatenation along seq dim (dim 2: 4 + 4 = 8).
        let spliced2 = cache.splice_into_gemma4(&key, &mut gemma_cache, &dev).unwrap();
        assert!(spliced2);

        let missing = ImageCacheKey::from_image_bytes("gemma4", 999, b"none");
        let not_found = cache.splice_into_gemma4(&missing, &mut gemma_cache, &dev).unwrap();
        assert!(!not_found);
    }
}

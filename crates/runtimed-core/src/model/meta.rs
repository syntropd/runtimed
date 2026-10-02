//! Live-model records: metadata plus the guarded session handle.
//!
//! Split from `loader.rs` (module size rule); the manager owns the map,
//! these types describe one entry.

use super::tokenizer::EngineTokenizer;
use crate::memory::DualWatermarkController;
use runtimed_gguf::{GgufFile, MetaValue};
use runtimed_model::{Session, VisionTower, VocabTrie};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, RwLock};

/// Metadata for an active model loaded into memory or compute device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadedModel {
    /// Canonical model name (e.g. "qwen2.5-coder-7b", "phi-3-mini").
    pub name: String,
    /// Neural architecture family (e.g. "llama", "qwen2", "bert").
    pub architecture: String,
    /// Total parameter count.
    pub parameter_count: u64,
    /// Resident memory footprint in bytes (RAM or VRAM).
    pub memory_bytes: usize,
    /// Maximum supported context window in tokens.
    pub context_window: usize,
    /// Active hardware compute backend.
    pub compute_backend: String,
}

/// A live model: metadata, mutex-guarded session, tokenizer, stop ids.
pub struct EngineEntry {
    pub meta: LoadedModel,
    pub session: Mutex<Session>,
    pub tokenizer: EngineTokenizer,
    pub eos: Vec<u32>,
    pub add_special: bool,
    /// Attached vision tower (`attach_vision`); `None` means text-only.
    pub vision: RwLock<Option<VisionTower>>,
    /// L2 host prefix cache for projected soft tokens and KV layers.
    pub image_cache: RwLock<runtimed_model::cache::image_cache::ImageCache>,
    /// Shared vocabulary prefix trie pooled across family models.
    pub vocab_trie: Arc<VocabTrie>,
    /// Persisted dual watermark controller for VRAM/system pressure management.
    pub watermark_controller: Mutex<DualWatermarkController>,
}

/// Numeric GGUF metadata that may sit in a U32/U64/I32 slot.
pub(super) fn meta_u32(file: &GgufFile, key: &str) -> Option<u32> {
    match file.metadata.get(key) {
        Some(MetaValue::U32(v)) => Some(*v),
        Some(MetaValue::U64(v)) => u32::try_from(*v).ok(),
        Some(MetaValue::I32(v)) => u32::try_from(*v).ok(),
        _ => None,
    }
}

/// Canonical model family key for pooling shared vocabulary trie instances.
///
/// Qwen 2, Qwen 2.5, Qwen-Coder, Qwen-VL, and DeepSeek-R1-distill-qwen models
/// all share the identical 152k-token byte-level BPE vocabulary. Normalizing to
/// a canonical family key ensures all sessions pool a single shared `Arc<VocabTrie>`.
pub fn canonical_family_key(architecture: &str, model_name: &str) -> String {
    let arch = architecture.to_ascii_lowercase();
    let name = model_name.to_ascii_lowercase();
    if arch.starts_with("qwen") || name.contains("qwen") || name.contains("deepseek-r1-distill-qwen") {
        "qwen2".to_string()
    } else {
        arch
    }
}

/// Retrieve or initialize a pooled vocabulary prefix trie for this model family.
pub fn pool_family_vocab_trie(
    architecture: &str,
    model_name: &str,
    tokenizer: &EngineTokenizer,
) -> Arc<VocabTrie> {
    let key = canonical_family_key(architecture, model_name);
    runtimed_model::sampler::get_or_create_shared_vocab_trie(&key, tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_model_serde_roundtrip() {
        let meta = LoadedModel {
            name: "m".into(),
            architecture: "qwen2".into(),
            parameter_count: 1,
            memory_bytes: 2,
            context_window: 3,
            compute_backend: "cpu".into(),
        };
        let v = serde_json::to_value(&meta).unwrap();
        assert_eq!(v["name"], "m");
        assert_eq!(serde_json::from_value::<LoadedModel>(v).unwrap(), meta);
    }

    #[test]
    fn test_canonical_family_key_pooling() {
        assert_eq!(canonical_family_key("qwen2", "qwen2.5-7b"), "qwen2");
        assert_eq!(canonical_family_key("qwen2.5", "qwen2.5-0.5b"), "qwen2");
        assert_eq!(canonical_family_key("llama", "deepseek-r1-distill-qwen-14b"), "qwen2");
        assert_eq!(canonical_family_key("gemma2", "gemma-2-9b"), "gemma2");
    }
}

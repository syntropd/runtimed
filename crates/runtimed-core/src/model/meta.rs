//! Live-model records: metadata plus the guarded session handle.
//!
//! Split from `loader.rs` (module size rule); the manager owns the map,
//! these types describe one entry.

use super::tokenizer::EngineTokenizer;
use runtimed_gguf::{GgufFile, MetaValue};
use runtimed_model::{Session, VisionTower};
use serde::{Deserialize, Serialize};
use std::sync::{Mutex, RwLock};

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
}

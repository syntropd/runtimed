//! Model loading, metadata tracking, and lifecycle management.

use crate::error::RuntimedError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{RwLock, RwLockReadGuard};

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

/// Manages active loaded models and dynamic eviction.
pub struct ModelManager {
    models_dir: PathBuf,
    active_models: RwLock<HashMap<String, LoadedModel>>,
}

impl ModelManager {
    /// Initializes a ModelManager rooted at the designated model weights directory.
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Self {
        Self {
            models_dir: models_dir.as_ref().to_path_buf(),
            active_models: RwLock::new(HashMap::new()),
        }
    }

    /// Loads or binds a model into memory for inference.
    ///
    /// Uses `HashMap::entry().or_insert_with(...)` under the write lock so
    /// the read-then-write race (a second caller with a different
    /// `backend` slipping between the read check and the write insert)
    /// is closed. The entry API only inserts if absent, so the first
    /// caller's synthetic profile wins; concurrent calls return clones of
    /// the canonical entry.
    ///
    /// The synthetic `LoadedModel` is constructed inside the closure so
    /// repeated `load_model` for an already-loaded model does not pay
    /// for the per-field `String` allocations.
    pub fn load_model(&self, name: &str, backend: Option<&str>) -> Result<LoadedModel, RuntimedError> {
        let mut lock = self.write_lock()?;
        let entry = lock
            .entry(name.to_string())
            .or_insert_with(|| LoadedModel {
                name: name.to_string(),
                architecture: "transformer".to_string(),
                parameter_count: 7_000_000_000,
                memory_bytes: 4_500_000_000,
                context_window: 8192,
                compute_backend: backend.unwrap_or("cpu-avx2").to_string(),
            });
        Ok(entry.clone())
    }

    /// Retrieves an active model definition by name.
    pub fn get_model(&self, name: &str) -> Option<LoadedModel> {
        let lock = self.read_lock().ok()?;
        lock.get(name).cloned()
    }

    /// Unloads a model and releases its memory allocation.
    pub fn unload_model(&self, name: &str) -> Result<usize, RuntimedError> {
        let mut lock = self.write_lock()?;
        match lock.remove(name) {
            Some(model) => Ok(model.memory_bytes),
            None => Err(RuntimedError::ModelNotFound(name.to_string())),
        }
    }

    /// Returns a list of all currently active models.
    pub fn list_active(&self) -> Vec<LoadedModel> {
        let lock = match self.read_lock() {
            Ok(l) => l,
            Err(_) => return Vec::new(),
        };
        lock.values().cloned().collect()
    }

    /// Returns the models directory path.
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    fn read_lock(&self) -> Result<RwLockReadGuard<'_, HashMap<String, LoadedModel>>, RuntimedError> {
        self.active_models.read().map_err(|_| {
            RuntimedError::GenerationFailed("Model manager lock poisoned".into())
        })
    }

    fn write_lock(&self) -> Result<std::sync::RwLockWriteGuard<'_, HashMap<String, LoadedModel>>, RuntimedError> {
        self.active_models.write().map_err(|_| {
            RuntimedError::GenerationFailed("Model manager lock poisoned".into())
        })
    }
}
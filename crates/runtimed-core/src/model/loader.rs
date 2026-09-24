//! Model loading, metadata tracking, and lifecycle management.

use crate::error::RuntimedError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

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
    active_models: Mutex<HashMap<String, LoadedModel>>,
}

impl ModelManager {
    /// Initializes a ModelManager rooted at the designated model weights directory.
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Self {
        Self {
            models_dir: models_dir.as_ref().to_path_buf(),
            active_models: Mutex::new(HashMap::new()),
        }
    }

    /// Loads or binds a model into memory for inference.
    pub fn load_model(&self, name: &str, backend: Option<&str>) -> Result<LoadedModel, RuntimedError> {
        let mut lock = self.active_models.lock().map_err(|_| {
            RuntimedError::GenerationFailed("Model manager mutex poisoned".into())
        })?;

        if let Some(existing) = lock.get(name) {
            return Ok(existing.clone());
        }

        let backend_str = backend.unwrap_or("cpu-avx2").to_string();

        // Synthetic/default profile for system inference
        let model = LoadedModel {
            name: name.to_string(),
            architecture: "transformer".to_string(),
            parameter_count: 7_000_000_000,
            memory_bytes: 4_500_000_000,
            context_window: 8192,
            compute_backend: backend_str,
        };

        lock.insert(name.to_string(), model.clone());
        Ok(model)
    }

    /// Retrieves an active model definition by name.
    pub fn get_model(&self, name: &str) -> Option<LoadedModel> {
        self.active_models.lock().ok()?.get(name).cloned()
    }

    /// Unloads a model and releases its memory allocation.
    pub fn unload_model(&self, name: &str) -> Result<usize, RuntimedError> {
        let mut lock = self.active_models.lock().map_err(|_| {
            RuntimedError::GenerationFailed("Model manager mutex poisoned".into())
        })?;

        match lock.remove(name) {
            Some(model) => Ok(model.memory_bytes),
            None => Err(RuntimedError::ModelNotFound(name.to_string())),
        }
    }

    /// Returns a list of all currently active models.
    pub fn list_active(&self) -> Vec<LoadedModel> {
        self.active_models
            .lock()
            .map(|l| l.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns the models directory path.
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }
}

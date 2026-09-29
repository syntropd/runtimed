//! ModelManager lifecycle state and access controls.

use super::super::admit_lease::LeasePermit;
use super::super::meta::{EngineEntry, LoadedModel};
use super::super::resolve::{load_registry, resolve_path};
use crate::error::RuntimedError;
use candle_core::Device;
use runtimed_gguf::Registry;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;

/// Manages active loaded models and dynamic eviction.
pub struct ModelManager {
    pub(crate) models_dir: PathBuf,
    pub(super) active_models: RwLock<HashMap<String, Arc<EngineEntry>>>,
    /// Inferenced leases held per resident model (own lock; see admit_lease).
    pub(crate) leases: Mutex<HashMap<String, LeasePermit>>,
    /// Last generation-class use (idle unload clock; see idle).
    pub(crate) last_used: Mutex<Instant>,
    /// SHA256 pins from `registry.toml` at the models root or its
    /// `gguf/` subdir (R2). `None` means no registry file: loads
    /// proceed unenforced (fail-open, logged).
    registry: Option<Registry>,
    /// Parse failure of a present-but-broken registry (also fail-open).
    registry_error: Option<String>,
}

impl ModelManager {
    /// Initializes a ModelManager rooted at the designated model weights directory.
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Self {
        let models_dir = models_dir.as_ref().to_path_buf();
        let (registry, registry_error) = load_registry(&models_dir);
        Self {
            models_dir,
            active_models: RwLock::new(HashMap::new()),
            leases: Mutex::new(HashMap::new()),
            last_used: Mutex::new(Instant::now()),
            registry,
            registry_error,
        }
    }

    /// Pin status for the startup log: entries enforced, or why not.
    pub fn registry_status(&self) -> String {
        if let Some(e) = &self.registry_error {
            return format!("BROKEN registry.toml ({e}); loads unenforced");
        }
        match &self.registry {
            Some(r) => format!("enforced ({} pins)", r.len()),
            None => "absent; loads unenforced".to_string(),
        }
    }

    /// Resolve a model name to a weight path: absolute paths pass through,
    /// bare names resolve under models dir, safetensors, or gguf subdirs.
    pub(crate) fn resolve(&self, name: &str) -> Option<PathBuf> {
        resolve_path(&self.models_dir, name)
    }

    /// R2 pin check: no-op without a registry; with one, unknown names
    /// and hash mismatches refuse the load before any weights parse.
    pub(crate) fn pinned(&self, name: &str, path: &Path) -> Result<(), RuntimedError> {
        match &self.registry {
            None => Ok(()),
            Some(reg) => reg
                .verify(name, path)
                .map_err(|e| RuntimedError::GenerationFailed(format!("weight verification: {e}"))),
        }
    }

    /// Retrieves an active model definition by name.
    pub fn get_model(&self, name: &str) -> Option<LoadedModel> {
        let lock = self.read_lock().ok()?;
        let canonical = name.strip_suffix(":latest").unwrap_or(name);
        lock.get(name).or_else(|| lock.get(canonical)).map(|e| e.meta.clone())
    }

    /// CUDA device from `"cuda"` (device 0) or `"cuda:N"`.
    /// Without the `cuda` feature this always errors (auditable, loud).
    pub(super) fn cuda_device(spec: &str) -> Result<Device, RuntimedError> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = spec;
            return Err(RuntimedError::HardwareAllocation(
                "cuda backend needs a --features cuda build".into(),
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let ordinal: usize = spec
                .strip_prefix("cuda:")
                .map(str::parse)
                .transpose()
                .map_err(|_| RuntimedError::HardwareAllocation(format!("bad cuda spec '{spec}'")))?
                .unwrap_or(0);
            Device::new_cuda(ordinal).map_err(|e| RuntimedError::HardwareAllocation(e.to_string()))
        }
    }

    /// Retrieves the live entry (session + tokenizer) for generation.
    pub fn get_entry(&self, name: &str) -> Option<Arc<EngineEntry>> {
        self.touch();
        let lock = self.read_lock().ok()?;
        let canonical = name.strip_suffix(":latest").unwrap_or(name);
        lock.get(name).or_else(|| lock.get(canonical)).cloned()
    }

    /// Unloads a model and releases its memory allocation.
    pub fn unload_model(&self, name: &str) -> Result<usize, RuntimedError> {
        let mut lock = self.write_lock()?;
        let canonical = name.strip_suffix(":latest").unwrap_or(name);
        let entry = lock.remove(name).or_else(|| lock.remove(canonical));
        if name != canonical {
            lock.remove(canonical);
        }
        match entry {
            Some(entry) => {
                if let Ok(session) = entry.session.try_lock() {
                    let _ = runtimed_model::weights::Weights::ensure_current(session.device());
                }
                tracing::info!(model = %entry.meta.name, "unload");
                self.relinquish(name);
                if name != canonical {
                    self.relinquish(canonical);
                }
                Ok(entry.meta.memory_bytes)
            }
            None => Err(RuntimedError::ModelNotFound(name.to_string())),
        }
    }

    /// Returns a list of all currently active models.
    pub fn list_active(&self) -> Vec<LoadedModel> {
        let lock = match self.read_lock() {
            Ok(l) => l,
            Err(_) => return Vec::new(),
        };
        let mut list: Vec<LoadedModel> = lock.values().map(|e| e.meta.clone()).collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list.dedup_by(|a, b| a.name == b.name);
        list
    }

    /// Returns the models directory path.
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub(super) fn read_lock(&self) -> Result<RwLockReadGuard<'_, HashMap<String, Arc<EngineEntry>>>, RuntimedError> {
        self.active_models.read().map_err(|_| RuntimedError::GenerationFailed("lock poisoned".into()))
    }

    pub(super) fn write_lock(&self) -> Result<RwLockWriteGuard<'_, HashMap<String, Arc<EngineEntry>>>, RuntimedError> {
        self.active_models.write().map_err(|_| RuntimedError::GenerationFailed("lock poisoned".into()))
    }
}

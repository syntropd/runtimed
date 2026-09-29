//! Model loading, metadata tracking, and lifecycle management.
//!
//! Models resolve to GGUF files under the models directory (or absolute
//! paths), load into owned inference sessions, and stay cached behind
//! `Arc` until unloaded. Metadata served over Varlink is the plain
//! `LoadedModel`; generation uses the live `EngineEntry`.

use super::admit_lease::LeasePermit;
use super::meta::{EngineEntry, LoadedModel};
use super::resolve::{load_registry, resolve_path};
use super::tokenizer::EngineTokenizer;
use crate::error::RuntimedError;
use candle_core::Device;
use runtimed_gguf::Registry;
use runtimed_model::Session;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};
use std::time::Instant;

/// Manages active loaded models and dynamic eviction.
pub struct ModelManager {
    models_dir: PathBuf,
    active_models: RwLock<HashMap<String, Arc<EngineEntry>>>,
    /// Inferenced leases held per resident model (own lock; see admit_lease).
    pub(super) leases: Mutex<HashMap<String, LeasePermit>>,
    /// Last generation-class use (idle unload clock; see idle).
    pub(super) last_used: Mutex<Instant>,
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

    /// Resolve a model name to a GGUF path: absolute paths pass through,
    /// bare names resolve under the models directory (`.gguf` implied),
    /// then under its `gguf/` subdir (fleet layout).
    pub(super) fn resolve(&self, name: &str) -> Option<PathBuf> {
        resolve_path(&self.models_dir, name)
    }

    /// R2 pin check: no-op without a registry; with one, unknown names
    /// and hash mismatches refuse the load before any weights parse.
    pub(super) fn pinned(&self, name: &str, path: &Path) -> Result<(), RuntimedError> {
        match &self.registry {
            None => Ok(()),
            Some(reg) => reg
                .verify(name, path)
                .map_err(|e| RuntimedError::GenerationFailed(format!("weight verification: {e}"))),
        }
    }

    /// Loads a model into memory, or returns the cached entry's metadata.
    ///
    /// Holds the write lock across the load so two callers racing on one
    /// name cannot double-load; loads are rare and the lock is process-wide.
    /// `backend` accepts `None`/`"cpu"` today; anything else is an honest
    /// error naming the phase that will unlock it.
    pub fn load_model(&self, name: &str, backend: Option<&str>) -> Result<LoadedModel, RuntimedError> {
        self.touch();
        let canonical = name.strip_suffix(":latest").unwrap_or(name);
        if let Ok(lock) = self.active_models.read() {
            if let Some(entry) = lock.get(name).or_else(|| lock.get(canonical)) {
                return Ok(entry.meta.clone());
            }
        }
        let mut lock = self.write_lock()?;
        if let Some(entry) = lock.get(name).or_else(|| lock.get(canonical)) {
            return Ok(entry.meta.clone());
        }
        let cas_fd = super::cas_fd::fetch_model_fd(name);
        let path_opt = self.resolve(name);
        if cas_fd.is_none() && path_opt.is_none() {
            return Err(RuntimedError::ModelNotFound(name.to_string()));
        }
        if let Some(ref p) = path_opt {
            self.pinned(name, p)?;
        }
        let env_b = std::env::var("RUNTIMED_BACKEND").ok();
        let backend = backend.or(env_b.as_deref());
        let device = match backend {
            None | Some("cpu") => Device::Cpu,
            Some(b) if b == "cuda" || b.starts_with("cuda:") => Self::cuda_device(b)?,
            Some(other) => {
                return Err(RuntimedError::HardwareAllocation(format!(
                    "unknown backend '{other}' (want cpu, cuda, cuda:N)"
                )));
            }
        };
        let (session, file, path) = if let Some(fd) = cas_fd {
            let cas_file = std::fs::File::from(fd);
            let bytes = cas_file.metadata().map(|m| m.len()).unwrap_or(0);
            self.admit_bytes(name, bytes)?;
            let p = path_opt.unwrap_or_else(|| PathBuf::from(name));
            let (s, f) = Session::load_from_file(&cas_file, &device).map_err(|e| {
                self.relinquish(name);
                RuntimedError::GenerationFailed(e.to_string())
            })?;
            (s, f, p)
        } else {
            let p = path_opt.unwrap();
            self.admit(name, &p)?;
            let (s, f) = Session::load(&p, &device).map_err(|e| {
                self.relinquish(name);
                RuntimedError::GenerationFailed(e.to_string())
            })?;
            (s, f, p)
        };
        let arch_tag = file.meta_str("general.architecture").unwrap_or("?");
        let (tokenizer, eos, add_special) =
            EngineTokenizer::load_for_arch(session.config().arch, &file, &path)?;
        let params: u64 = file.tensors.iter().map(|t| t.n_elements as u64).sum();
        let context = super::meta::meta_u32(&file, &format!("{arch_tag}.context_length"))
            .map(|c| c as usize)
            .unwrap_or(8192);
        let meta = LoadedModel {
            name: name.to_string(),
            architecture: arch_tag.to_string(),
            parameter_count: params,
            memory_bytes: session.resident_bytes() + (64 << 20),
            context_window: context,
            compute_backend: if matches!(session.device(), Device::Cuda(_)) { "cuda" } else { "cpu" }.to_string(),
        };
        let entry = Arc::new(EngineEntry {
            meta: meta.clone(),
            session: Mutex::new(session),
            tokenizer,
            eos,
            add_special,
            vision: RwLock::new(None),
        });
        tracing::info!(model = %meta.name, backend = %meta.compute_backend, bytes = meta.memory_bytes, "load");
        lock.insert(name.to_string(), Arc::clone(&entry));
        if name != canonical {
            lock.insert(canonical.to_string(), entry);
        }
        Ok(meta)
    }

    /// Retrieves an active model definition by name.
    pub fn get_model(&self, name: &str) -> Option<LoadedModel> {
        let lock = self.read_lock().ok()?;
        let canonical = name.strip_suffix(":latest").unwrap_or(name);
        lock.get(name).or_else(|| lock.get(canonical)).map(|e| e.meta.clone())
    }

    /// CUDA device from `"cuda"` (device 0) or `"cuda:N"`.
    /// Without the `cuda` feature this always errors (auditable, loud).
    fn cuda_device(spec: &str) -> Result<Device, RuntimedError> {
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

    fn read_lock(&self) -> Result<RwLockReadGuard<'_, HashMap<String, Arc<EngineEntry>>>, RuntimedError> {
        self.active_models.read().map_err(|_| RuntimedError::GenerationFailed("lock poisoned".into()))
    }

    fn write_lock(&self) -> Result<std::sync::RwLockWriteGuard<'_, HashMap<String, Arc<EngineEntry>>>, RuntimedError> {
        self.active_models.write().map_err(|_| RuntimedError::GenerationFailed("lock poisoned".into()))
    }
}

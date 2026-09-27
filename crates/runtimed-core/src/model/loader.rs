//! Model loading, metadata tracking, and lifecycle management.
//!
//! Models resolve to GGUF files under the models directory (or absolute
//! paths), load into owned inference sessions, and stay cached behind
//! `Arc` until unloaded. Metadata served over Varlink is the plain
//! `LoadedModel`; generation uses the live `EngineEntry`.

use super::meta::{EngineEntry, LoadedModel};
use super::tokenizer::EngineTokenizer;
use crate::error::RuntimedError;
use candle_core::Device;
use runtimed_gguf::{GgufBpe, MetaValue, Registry, Tokenizer};
use runtimed_model::{Arch, Session};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};

/// Manages active loaded models and dynamic eviction.
pub struct ModelManager {
    models_dir: PathBuf,
    active_models: RwLock<HashMap<String, Arc<EngineEntry>>>,
    /// SHA256 pins from `<models_dir>/registry.toml` (R2). `None` means
    /// no registry file: loads proceed unenforced (fail-open, logged).
    registry: Option<Registry>,
    /// Parse failure of a present-but-broken registry (also fail-open).
    registry_error: Option<String>,
}

impl ModelManager {
    /// Initializes a ModelManager rooted at the designated model weights directory.
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Self {
        let models_dir = models_dir.as_ref().to_path_buf();
        let (registry, registry_error) = match Registry::load(&models_dir.join("registry.toml")) {
            Ok(r) => (Some(r), None),
            Err(e) if e.is_missing() => (None, None),
            Err(e) => (None, Some(e.to_string())),
        };
        Self {
            models_dir,
            active_models: RwLock::new(HashMap::new()),
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
    /// bare names resolve under the models directory (`.gguf` implied).
    pub(super) fn resolve(&self, name: &str) -> Option<PathBuf> {
        let literal = PathBuf::from(name);
        if literal.is_absolute() && literal.exists() {
            return Some(literal);
        }
        let direct = self.models_dir.join(name);
        if direct.exists() {
            return Some(direct);
        }
        let with_ext = self.models_dir.join(format!("{name}.gguf"));
        if with_ext.exists() {
            return Some(with_ext);
        }
        None
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
        if let Ok(lock) = self.active_models.read() {
            if let Some(entry) = lock.get(name) {
                return Ok(entry.meta.clone());
            }
        }
        let mut lock = self.write_lock()?;
        if let Some(entry) = lock.get(name) {
            return Ok(entry.meta.clone());
        }
        let path = self
            .resolve(name)
            .ok_or_else(|| RuntimedError::ModelNotFound(name.to_string()))?;
        self.pinned(name, &path)?;
        // Daemon-wide default backend (this daemon serves one device).
        let env_backend;
        let backend = match backend {
            Some(b) => Some(b),
            None => {
                env_backend = std::env::var("RUNTIMED_BACKEND").ok();
                env_backend.as_deref()
            }
        };
        let device = match backend {
            None | Some("cpu") => Device::Cpu,
            Some(b) if b == "cuda" || b.starts_with("cuda:") => Self::cuda_device(b)?,
            Some(other) => {
                return Err(RuntimedError::HardwareAllocation(format!(
                    "unknown backend '{other}' (want cpu, cuda, cuda:N)"
                )));
            }
        };
        let (session, file) =
            Session::load(&path, &device).map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?;
        let arch_tag = file.meta_str("general.architecture").unwrap_or("?");
        let (tokenizer, eos, add_special) = match session.config().arch {
            Arch::Gemma4 => {
                let bpe = GgufBpe::from_gguf(&file)
                    .map_err(|e| RuntimedError::GenerationFailed(format!("bpe: {e}")))?;
                let eos = bpe.eos_id().into_iter().collect();
                let add_special = bpe.wants_bos();
                (EngineTokenizer::Bpe(bpe), eos, add_special)
            }
            Arch::Qwen2 => {
                let tok_path = path.with_extension("tokenizer.json");
                let tok = Tokenizer::from_file(&tok_path).map_err(|_| {
                    RuntimedError::GenerationFailed(format!(
                        "qwen2 needs a sibling tokenizer.json next to {}",
                        path.display()
                    ))
                })?;
                let eos = super::meta::meta_u32(&file, "tokenizer.ggml.eos_token_id")
                    .into_iter()
                    .collect();
                let add_special =
                    matches!(file.metadata.get("tokenizer.ggml.add_bos_token"), Some(MetaValue::Bool(true)));
                (EngineTokenizer::File(tok), eos, add_special)
            }
        };
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
            compute_backend: if matches!(session.device(), Device::Cuda(_)) { "cuda" } else { "cpu" }
                .to_string(),
        };
        let entry = Arc::new(EngineEntry {
            meta: meta.clone(),
            session: Mutex::new(session),
            tokenizer,
            eos,
            add_special,
            vision: RwLock::new(None),
        });
        tracing::info!(
            model = %meta.name, backend = %meta.compute_backend,
            bytes = meta.memory_bytes, "load"
        );
        lock.insert(name.to_string(), entry);
        Ok(meta)
    }

    /// Retrieves an active model definition by name.
    pub fn get_model(&self, name: &str) -> Option<LoadedModel> {
        let lock = self.read_lock().ok()?;
        lock.get(name).map(|e| e.meta.clone())
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
        let lock = self.read_lock().ok()?;
        lock.get(name).cloned()
    }

    /// Unloads a model and releases its memory allocation.
    pub fn unload_model(&self, name: &str) -> Result<usize, RuntimedError> {
        let mut lock = self.write_lock()?;
        match lock.remove(name) {
            Some(entry) => {
                // Free VRAM on a bound thread: `try_lock` succeeds exactly
                // when no generate is in flight (then this drop is final).
                if let Ok(session) = entry.session.try_lock() {
                    let _ = runtimed_model::weights::Weights::ensure_current(session.device());
                }
                tracing::info!(model = %entry.meta.name, "unload");
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
        lock.values().map(|e| e.meta.clone()).collect()
    }

    /// Returns the models directory path.
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    fn read_lock(&self) -> Result<RwLockReadGuard<'_, HashMap<String, Arc<EngineEntry>>>, RuntimedError> {
        self.active_models.read().map_err(|_| {
            RuntimedError::GenerationFailed("Model manager lock poisoned".into())
        })
    }

    fn write_lock(&self) -> Result<std::sync::RwLockWriteGuard<'_, HashMap<String, Arc<EngineEntry>>>, RuntimedError> {
        self.active_models.write().map_err(|_| {
            RuntimedError::GenerationFailed("Model manager lock poisoned".into())
        })
    }
}

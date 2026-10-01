//! Loading models from GGUF or Safetensors sources.

use super::manager::ModelManager;
use super::safetensors::load_safetensors_entry;
use crate::error::RuntimedError;
use crate::model::meta::{EngineEntry, LoadedModel};
use crate::model::tokenizer::EngineTokenizer;
use candle_core::Device;
use runtimed_model::Session;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

impl ModelManager {
    /// Loads a model into memory, or returns the cached entry's metadata.
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
        let cas_fd = crate::model::cas::fetch_model_fd(name);
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

        let (session, meta, tokenizer, eos, add_special) = if let Some(fd) = cas_fd {
            let mut cas_file = std::fs::File::from(fd);
            let bytes = cas_file.metadata().map(|m| m.len()).unwrap_or(0);
            self.admit_bytes(name, bytes)?;
            let p = path_opt.unwrap_or_else(|| PathBuf::from(name));

            let mut magic = [0u8; 4];
            let is_gguf = if cas_file.read_exact(&mut magic).is_ok() && &magic == b"GGUF" {
                let _ = cas_file.seek(SeekFrom::Start(0));
                true
            } else {
                let _ = cas_file.seek(SeekFrom::Start(0));
                false
            };

            if !is_gguf {
                let res = load_safetensors_entry(name, &p, Some(&cas_file), &device);
                match res {
                    Ok(entry) => entry,
                    Err(e) => {
                        self.relinquish(name);
                        return Err(e);
                    }
                }
            } else {
                let (s, f) = Session::load_from_file(&cas_file, &device).map_err(|e| {
                    self.relinquish(name);
                    RuntimedError::GenerationFailed(e.to_string())
                })?;
                let arch_tag = f.meta_str("general.architecture").unwrap_or("?");
                let (tok, eos, sp) = EngineTokenizer::load_for_arch(s.config().arch, &f, &p)?;
                let params: u64 = f.tensors.iter().map(|t| t.n_elements as u64).sum();
                let ctx = crate::model::meta::meta_u32(&f, &format!("{arch_tag}.context_length"))
                    .map(|c| c as usize)
                    .unwrap_or(8192);
                let m = LoadedModel {
                    name: name.to_string(),
                    architecture: arch_tag.to_string(),
                    parameter_count: params,
                    memory_bytes: s.resident_bytes() + (64 << 20),
                    context_window: ctx,
                    compute_backend: if matches!(s.device(), Device::Cuda(_)) { "cuda" } else { "cpu" }.to_string(),
                };
                (s, m, tok, eos, sp)
            }
        } else {
            let p = path_opt.ok_or_else(|| RuntimedError::ModelNotFound(name.to_string()))?;
            self.admit(name, &p)?;
            let is_safetensors = p.extension().map(|e| e.eq_ignore_ascii_case("safetensors")).unwrap_or(false);
            if is_safetensors {
                let res = load_safetensors_entry(name, &p, None, &device);
                match res {
                    Ok(entry) => entry,
                    Err(e) => {
                        self.relinquish(name);
                        return Err(e);
                    }
                }
            } else {
                let (s, f) = Session::load(&p, &device).map_err(|e| {
                    self.relinquish(name);
                    RuntimedError::GenerationFailed(e.to_string())
                })?;
                let arch_tag = f.meta_str("general.architecture").unwrap_or("?");
                let (tok, eos, sp) = EngineTokenizer::load_for_arch(s.config().arch, &f, &p)?;
                let params: u64 = f.tensors.iter().map(|t| t.n_elements as u64).sum();
                let ctx = crate::model::meta::meta_u32(&f, &format!("{arch_tag}.context_length"))
                    .map(|c| c as usize)
                    .unwrap_or(8192);
                let m = LoadedModel {
                    name: name.to_string(),
                    architecture: arch_tag.to_string(),
                    parameter_count: params,
                    memory_bytes: s.resident_bytes() + (64 << 20),
                    context_window: ctx,
                    compute_backend: if matches!(s.device(), Device::Cuda(_)) { "cuda" } else { "cpu" }.to_string(),
                };
                (s, m, tok, eos, sp)
            }
        };

        let entry = Arc::new(EngineEntry {
            meta: meta.clone(),
            session: Mutex::new(session),
            tokenizer,
            eos,
            add_special,
            vision: RwLock::new(None),
            image_cache: RwLock::new(runtimed_model::cache::image_cache::ImageCache::new(16)),
        });
        tracing::info!(model = %meta.name, backend = %meta.compute_backend, bytes = meta.memory_bytes, "load");
        lock.insert(name.to_string(), Arc::clone(&entry));
        if name != canonical {
            lock.insert(canonical.to_string(), entry);
        }
        Ok(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_load_model_not_found() {
        let tmp = TempDir::new().unwrap();
        let mgr = ModelManager::new(tmp.path());
        let res = mgr.load_model("nonexistent_model", None);
        assert!(matches!(res, Err(RuntimedError::ModelNotFound(_))));
    }

    #[test]
    fn test_load_model_invalid_backend() {
        let tmp = TempDir::new().unwrap();
        let model_path = tmp.path().join("test_model.gguf");
        std::fs::write(&model_path, b"dummy").unwrap();
        let mgr = ModelManager::new(tmp.path());
        let res = mgr.load_model("test_model", Some("invalid_accel"));
        assert!(matches!(res, Err(RuntimedError::HardwareAllocation(_))));
    }
}

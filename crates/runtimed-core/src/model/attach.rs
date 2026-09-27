//! Runtime attachments: vision towers and LoRA adapters.
//!
//! Both bind sidecar files to a loaded model without touching the base
//! weights on disk. Unloading the model drops everything.

use super::loader::ModelManager;
use super::meta::LoadedModel;
use crate::error::RuntimedError;
use runtimed_model::{LoraAdapter, VisionTower};

impl ModelManager {
    /// Attaches a vision projector to a loaded model, enabling image input.
    /// `mmproj` resolves like a model name (absolute path or models dir).
    /// Re-attaching replaces the previous tower.
    pub fn attach_vision(&self, name: &str, mmproj: &str) -> Result<LoadedModel, RuntimedError> {
        let entry = self
            .get_entry(name)
            .ok_or_else(|| RuntimedError::ModelNotFound(name.to_string()))?;
        let path = self
            .resolve(mmproj)
            .ok_or_else(|| RuntimedError::ModelNotFound(mmproj.to_string()))?;
        self.pinned(mmproj, &path)?;
        // Tower must live on the session's device (CPU or CUDA).
        let dev = {
            let session = entry
                .session
                .lock()
                .map_err(|_| RuntimedError::GenerationFailed("session lock poisoned".into()))?;
            session.device().clone()
        };
        let tower = VisionTower::load(&path, &dev)
            .map_err(|e| RuntimedError::GenerationFailed(format!("mmproj: {e}")))?;
        let mut slot = entry
            .vision
            .write()
            .map_err(|_| RuntimedError::GenerationFailed("vision lock poisoned".into()))?;
        *slot = Some(tower);
        tracing::info!(model = %entry.meta.name, mmproj = %mmproj, "attach-vision");
        Ok(entry.meta.clone())
    }

    /// Fuses a LoRA adapter into a loaded model's live weights.
    /// Returns the fused base-weight names. Reload to detach.
    pub fn attach_lora(&self, name: &str, lora: &str) -> Result<Vec<String>, RuntimedError> {
        let entry = self
            .get_entry(name)
            .ok_or_else(|| RuntimedError::ModelNotFound(name.to_string()))?;
        let path = self
            .resolve(lora)
            .ok_or_else(|| RuntimedError::ModelNotFound(lora.to_string()))?;
        self.pinned(lora, &path)?;
        let mut session = entry
            .session
            .lock()
            .map_err(|_| RuntimedError::GenerationFailed("session lock poisoned".into()))?;
        let adapter = LoraAdapter::load(&path, session.device(), entry.meta.architecture.as_str())
            .map_err(|e| RuntimedError::GenerationFailed(format!("lora: {e}")))?;
        session.reset();
        let fused = session
            .fuse_lora(&adapter)
            .map_err(|e| RuntimedError::GenerationFailed(format!("lora: {e}")))?;
        tracing::info!(model = %entry.meta.name, lora = %lora, fused = fused.len(), "attach-lora");
        Ok(fused)
    }
}

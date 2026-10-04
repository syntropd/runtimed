//! Lease permit lifecycle and model admission enforcement.

use super::client::LeaseClient;
use super::sizing::{compute_lease_bytes, ACTIVATION_HEADROOM_BYTES};
use crate::error::RuntimedError;
use crate::model::loader::ModelManager;
use serde_json::json;
use std::os::unix::net::UnixStream;
use std::path::Path;

/// A held compute lease: the Varlink connection stays OPEN for the
/// lease lifetime. inferenced reclaims the lease the moment its client
/// disconnects ("orphaned lease"), so dropping this early voids the
/// admission. Released explicitly on unload.
pub struct LeasePermit {
    pub(super) id: String,
    stream: UnixStream,
}

impl std::fmt::Debug for LeasePermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeasePermit").field("id", &self.id).finish()
    }
}

impl LeasePermit {
    pub(super) fn new(id: String, stream: UnixStream) -> Self {
        Self { id, stream }
    }

    /// Release this lease over its own held connection, then close it.
    /// Best-effort: callers log failures and never fail for them.
    pub fn release(mut self) -> Result<(), String> {
        let reply = LeaseClient::transact(
            &mut self.stream,
            "io.systemd.inferenced1.ReleaseLease",
            json!({"lease_id": self.id}),
        )?;
        if let Some(err) = reply.get("error").and_then(|e| e.as_str()) {
            return Err(format!("release refused: {err}"));
        }
        Ok(())
    }
}

impl ModelManager {
    /// Admit a model load with explicit dynamic lease sizing based on weights, KV cache, and workspace.
    pub(in crate::model) fn admit_with_config(
        &self,
        name: &str,
        weights_file_bytes: u64,
        cfg: &runtimed_model::config::ArchConfig,
        context_tokens: usize,
    ) -> Result<(), RuntimedError> {
        let want = compute_lease_bytes(weights_file_bytes, cfg, context_tokens);
        self.acquire_lease(name, want)
    }

    /// Acquire lease from inferenced and record permit in held leases map.
    pub(in crate::model) fn acquire_lease(&self, name: &str, want: u64) -> Result<(), RuntimedError> {
        match LeaseClient::from_env().acquire(want)? {
            Some(permit) => {
                tracing::info!(model = %name, lease = %permit.id, bytes = want, "lease held");
                self.leases
                    .lock()
                    .map_err(|_| {
                        RuntimedError::GenerationFailed("lease map lock poisoned".into())
                    })?
                    .insert(name.to_string(), permit);
                Ok(())
            }
            None => {
                tracing::info!(model = %name, "no inferenced lease; loading unenforced");
                Ok(())
            }
        }
    }

    /// Admit a model load by file bytes with activation workspace headroom.
    pub(in crate::model) fn admit_bytes(&self, name: &str, file_bytes: u64) -> Result<(), RuntimedError> {
        let want = file_bytes + ACTIVATION_HEADROOM_BYTES;
        self.acquire_lease(name, want)
    }

    /// Admit a model load from weights path, attempting companion config inspection for exact sizing.
    pub(in crate::model) fn admit(&self, name: &str, weights: &Path) -> Result<(), RuntimedError> {
        let file_bytes = std::fs::metadata(weights).map(|m| m.len()).unwrap_or(0);
        let sibling = weights.with_extension("config.json");
        let parent_cfg = weights.parent().map(|p| p.join("config.json"));
        let dir_cfg = if weights.is_dir() { Some(weights.join("config.json")) } else { None };
        let cfg_path = [Some(sibling), parent_cfg, dir_cfg]
            .into_iter()
            .flatten()
            .find(|p| p.exists());

        if let Some(ref p) = cfg_path {
            if let Ok(cfg) = runtimed_model::ArchConfig::from_hf_file(p) {
                let ctx = cfg.sliding_window.unwrap_or(8192);
                return self.admit_with_config(name, file_bytes, &cfg, ctx);
            }
        }
        self.admit_bytes(name, file_bytes)
    }

    /// Drop a held lease (unload or failed load). Never fails: the local
    /// state change already happened; a missed release just strands a
    /// lease until inferenced reaps it.
    pub(in crate::model) fn relinquish(&self, name: &str) {
        let permit = match self.leases.lock() {
            Ok(mut map) => map.remove(name),
            Err(_) => return,
        };
        if let Some(permit) = permit {
            let id = permit.id.clone();
            if let Err(e) = permit.release() {
                tracing::warn!(model = %name, lease = %id, "lease release failed: {e}");
            } else {
                tracing::info!(model = %name, lease = %id, "lease released");
            }
        }
    }
}

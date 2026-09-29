//! Lease permit lifecycle and model admission enforcement.

use super::client::LeaseClient;
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
    /// Admit a model load: hold a lease while the model stays resident.
    /// Lock-free against `active_models` (separate map, own mutex).
    pub(in crate::model) fn admit_bytes(&self, name: &str, file_bytes: u64) -> Result<(), RuntimedError> {
        let want = file_bytes + (64 << 20);
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

    pub(in crate::model) fn admit(&self, name: &str, weights: &Path) -> Result<(), RuntimedError> {
        let file_bytes = std::fs::metadata(weights).map(|m| m.len()).unwrap_or(0);
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

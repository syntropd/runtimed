//! Admission control: hold an inferenced compute lease per loaded model.
//!
//! Before weights load, runtimed asks inferenced (the hardware landlord)
//! for a lease on a compute plane; on unload the lease is released. When
//! inferenced is absent the load proceeds without a lease (fail-open,
//! logged): standalone and test setups keep working. An explicit
//! `ResourceExhaustion` refusal is honored (fail-closed): the load fails
//! with [`RuntimedError::HardwareAllocation`].
//!
//! The client is synchronous [`std`] sockets + NUL-terminated JSON
//! Varlink, mirroring routerd's async telemetry client.

use super::loader::ModelManager;
use crate::error::RuntimedError;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Live inferenced Varlink socket (matches the daemon's own default).
pub const DEFAULT_INFERENCED_SOCKET: &str = "/run/syntrop/io.syntrop.Inference1";
/// Fleet-standard override, shared with syntropctl.
pub const INFERENCED_SOCKET_ENV: &str = "SYNTROP_INFERENCE_SOCKET";
const RPC_TIMEOUT: Duration = Duration::from_millis(500);
const MAX_REPLY_BYTES: u64 = 64 * 1024;

/// A held compute lease: released back to inferenced on unload.
#[derive(Debug)]
pub struct LeasePermit {
    id: String,
    socket: PathBuf,
}

/// Synchronous Varlink client for inferenced lease calls.
pub struct LeaseClient {
    socket: PathBuf,
}

impl LeaseClient {
    pub fn new<P: Into<PathBuf>>(socket: P) -> Self {
        Self { socket: socket.into() }
    }

    /// Resolve the socket from the fleet env override or the default path.
    pub fn from_env() -> Self {
        let path = std::env::var(INFERENCED_SOCKET_ENV)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_INFERENCED_SOCKET.to_string());
        Self::new(path)
    }

    /// Ask inferenced for `memory_bytes` on any plane. `Ok(None)` means
    /// no landlord is home (proceed without a lease); `Err` is an
    /// explicit refusal and must fail the load.
    pub fn acquire(&self, memory_bytes: u64) -> Result<Option<LeasePermit>, RuntimedError> {
        if !self.socket.exists() {
            return Ok(None);
        }
        let reply = match self.rpc(
            "io.systemd.inferenced1.AcquireLease",
            json!({"priority": "Interactive", "memory_bytes": memory_bytes, "pid": std::process::id()}),
        ) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("inferenced lease unreachable ({e}); loading without a lease");
                return Ok(None);
            }
        };
        if let Some(err) = reply.get("error").and_then(|e| e.as_str()) {
            let detail = reply
                .get("parameters")
                .and_then(|p| p.get("error"))
                .and_then(|e| e.as_str())
                .unwrap_or(err);
            return Err(RuntimedError::HardwareAllocation(format!(
                "inferenced refused compute lease: {detail}"
            )));
        }
        let id = reply
            .get("parameters")
            .and_then(|p| p.get("lease_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if id.is_empty() {
            tracing::warn!("inferenced lease reply carried no lease_id; loading without a lease");
            return Ok(None);
        }
        Ok(Some(LeasePermit { id: id.to_string(), socket: self.socket.clone() }))
    }

    /// Best-effort release; callers log failures and never fail for them.
    pub fn release_permit(&self, permit: &LeasePermit) -> Result<(), String> {
        match self.rpc("io.systemd.inferenced1.ReleaseLease", json!({"lease_id": permit.id})) {
            Ok(reply) => {
                if let Some(err) = reply.get("error").and_then(|e| e.as_str()) {
                    return Err(format!("release refused: {err}"));
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn rpc(&self, method: &str, parameters: Value) -> Result<Value, String> {
        let mut stream = UnixStream::connect(&self.socket).map_err(|e| e.to_string())?;
        stream.set_read_timeout(Some(RPC_TIMEOUT)).map_err(|e| e.to_string())?;
        stream.set_write_timeout(Some(RPC_TIMEOUT)).map_err(|e| e.to_string())?;
        let mut req = serde_json::to_vec(&json!({"method": method, "parameters": parameters}))
            .map_err(|e| e.to_string())?;
        req.push(0);
        stream.write_all(&req).map_err(|e| e.to_string())?;
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            if buf.len() as u64 > MAX_REPLY_BYTES {
                return Err("reply exceeded 64 KiB".to_string());
            }
            let n = stream.read(&mut chunk).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
            if buf.contains(&0) {
                break;
            }
        }
        if buf.last() == Some(&0) {
            buf.pop();
        }
        if buf.is_empty() {
            return Err("empty reply".to_string());
        }
        serde_json::from_slice(&buf).map_err(|e| e.to_string())
    }
}

impl ModelManager {
    /// Admit a model load: hold a lease while the model stays resident.
    /// Lock-free against `active_models` (separate map, own mutex).
    pub(super) fn admit(&self, name: &str, weights: &Path) -> Result<(), RuntimedError> {
        let file_bytes = std::fs::metadata(weights).map(|m| m.len()).unwrap_or(0);
        let want = file_bytes + (64 << 20);
        match LeaseClient::from_env().acquire(want)? {
            Some(permit) => {
                tracing::info!(model = %name, lease = %permit.id, bytes = want, "lease held");
                self.leases.lock().map_err(|_| {
                    RuntimedError::GenerationFailed("lease map lock poisoned".into())
                })?.insert(name.to_string(), permit);
                Ok(())
            }
            None => {
                tracing::info!(model = %name, "no inferenced lease; loading unenforced");
                Ok(())
            }
        }
    }

    /// Drop a held lease (unload or failed load). Never fails: the local
    /// state change already happened; a missed release just strands a
    /// lease until inferenced reaps it.
    pub(super) fn relinquish(&self, name: &str) {
        let permit = match self.leases.lock() {
            Ok(mut map) => map.remove(name),
            Err(_) => return,
        };
        if let Some(permit) = permit {
            let client = LeaseClient::new(&permit.socket);
            if let Err(e) = client.release_permit(&permit) {
                tracing::warn!(model = %name, lease = %permit.id, "lease release failed: {e}");
            } else {
                tracing::info!(model = %name, lease = %permit.id, "lease released");
            }
        }
    }
}

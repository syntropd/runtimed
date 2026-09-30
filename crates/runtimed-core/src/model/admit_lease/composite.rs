//! Multi-device composite gang lease client and RAII permit guard.

use super::client::{LeaseClient, DEFAULT_INFERENCED_SOCKET, INFERENCED_SOCKET_ENV};
use crate::error::RuntimedError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

const RPC_TIMEOUT: Duration = Duration::from_millis(1000);

/// Slice requirement for multi-accelerator composite lease acquisition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeSliceRequest {
    pub role: String,
    pub memory_bytes: u64,
    pub plane: Option<String>,
}

/// Slices allocated across heterogeneous hardware devices.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompositeSliceAllocation {
    pub plane_id: String,
    pub role: String,
    pub allocated_memory_bytes: u64,
    pub numa_node: Option<u32>,
    pub device_path: Option<PathBuf>,
}

/// Held composite gang lease that automatically releases itself on drop.
pub struct CompositeLeasePermit {
    pub id: String,
    pub slices: Vec<CompositeSliceAllocation>,
    stream: Option<UnixStream>,
}

impl std::fmt::Debug for CompositeLeasePermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeLeasePermit")
            .field("id", &self.id)
            .field("slices", &self.slices)
            .finish()
    }
}

impl Drop for CompositeLeasePermit {
    fn drop(&mut self) {
        if let Some(mut stream) = self.stream.take() {
            let req = json!({ "lease_id": self.id });
            let _ = LeaseClient::transact(
                &mut stream,
                "io.systemd.inferenced1.ReleaseCompositeLease",
                req,
            );
        }
    }
}

/// Varlink client for acquiring multi-accelerator composite leases.
pub struct CompositeLeaseClient {
    socket: PathBuf,
}

impl CompositeLeaseClient {
    pub fn new<P: Into<PathBuf>>(socket: P) -> Self {
        Self {
            socket: socket.into(),
        }
    }

    pub fn from_env() -> Self {
        let path = std::env::var(INFERENCED_SOCKET_ENV)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_INFERENCED_SOCKET.to_string());
        Self::new(path)
    }

    /// Ask inferenced to atomically admit a gang of plane slices.
    pub fn acquire(
        &self,
        slices: &[CompositeSliceRequest],
        priority: Option<&str>,
        policy: Option<&str>,
    ) -> Result<Option<CompositeLeasePermit>, RuntimedError> {
        if !self.socket.exists() {
            return Ok(None);
        }

        let mut stream = match self.connect() {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("inferenced composite lease unreachable ({e}); continuing unenforced");
                return Ok(None);
            }
        };

        let slices_json: Vec<Value> = slices
            .iter()
            .map(|s| {
                json!({
                    "role": s.role,
                    "memory_bytes": s.memory_bytes,
                    "plane": s.plane,
                })
            })
            .collect();

        let req = json!({
            "priority": priority.unwrap_or("Interactive"),
            "policy": policy.unwrap_or("AllOrNothing"),
            "slices": slices_json,
            "pid": std::process::id(),
        });

        let reply = match LeaseClient::transact(
            &mut stream,
            "io.systemd.inferenced1.AcquireCompositeLease",
            req,
        ) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("inferenced composite lease failed ({e}); continuing unenforced");
                return Ok(None);
            }
        };

        if let Some(err) = reply.get("error").and_then(|e| e.as_str()) {
            return Err(RuntimedError::HardwareAllocation(format!(
                "inferenced refused composite gang lease: {err}"
            )));
        }

        let params = reply.get("parameters").ok_or_else(|| {
            RuntimedError::HardwareAllocation("Malformed composite lease reply: missing parameters".into())
        })?;

        let id = params
            .get("lease_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if id.is_empty() {
            return Ok(None);
        }

        let mut allocs = Vec::new();
        if let Some(arr) = params.get("slices").and_then(|v| v.as_array()) {
            for item in arr {
                let plane_id = item.get("plane_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let memory_bytes = item.get("allocated_memory").and_then(|v| v.as_u64()).unwrap_or(0);
                let numa_node = item.get("numa_node").and_then(|v| v.as_u64()).map(|n| n as u32);
                let device_path = item.get("device_path").and_then(|v| v.as_str()).map(PathBuf::from);

                allocs.push(CompositeSliceAllocation {
                    plane_id,
                    role,
                    allocated_memory_bytes: memory_bytes,
                    numa_node,
                    device_path,
                });
            }
        }

        Ok(Some(CompositeLeasePermit {
            id,
            slices: allocs,
            stream: Some(stream),
        }))
    }

    fn connect(&self) -> Result<UnixStream, String> {
        let stream = UnixStream::connect(&self.socket).map_err(|e| e.to_string())?;
        stream.set_read_timeout(Some(RPC_TIMEOUT)).map_err(|e| e.to_string())?;
        stream.set_write_timeout(Some(RPC_TIMEOUT)).map_err(|e| e.to_string())?;
        Ok(stream)
    }
}

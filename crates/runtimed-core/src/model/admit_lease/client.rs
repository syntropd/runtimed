//! Synchronous Varlink client for inferenced compute lease allocation.

use super::permit::LeasePermit;
use crate::error::RuntimedError;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

/// Live inferenced Varlink socket (matches the daemon's own default).
pub const DEFAULT_INFERENCED_SOCKET: &str = "/run/syntrop/io.syntrop.Inference1";
/// Fleet-standard override, shared with syntropctl.
pub const INFERENCED_SOCKET_ENV: &str = "SYNTROP_INFERENCE_SOCKET";
const RPC_TIMEOUT: Duration = Duration::from_millis(500);
const MAX_REPLY_BYTES: u64 = 64 * 1024;

/// Synchronous Varlink client for inferenced lease calls.
pub struct LeaseClient {
    socket: PathBuf,
}

impl LeaseClient {
    pub fn new<P: Into<PathBuf>>(socket: P) -> Self {
        Self {
            socket: socket.into(),
        }
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
    /// explicit refusal and must fail the load. The granted permit
    /// keeps its connection open: closing it early lets inferenced
    /// reclaim the lease as orphaned.
    pub fn acquire(&self, memory_bytes: u64) -> Result<Option<LeasePermit>, RuntimedError> {
        if !self.socket.exists() {
            return Ok(None);
        }
        let mut stream = match self.connect() {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("inferenced lease unreachable ({e}); loading without a lease");
                return Ok(None);
            }
        };
        let reply = match Self::transact(
            &mut stream,
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
        Ok(Some(LeasePermit::new(id.to_string(), stream)))
    }

    fn connect(&self) -> Result<UnixStream, String> {
        let stream = UnixStream::connect(&self.socket).map_err(|e| e.to_string())?;
        stream.set_read_timeout(Some(RPC_TIMEOUT)).map_err(|e| e.to_string())?;
        stream.set_write_timeout(Some(RPC_TIMEOUT)).map_err(|e| e.to_string())?;
        Ok(stream)
    }

    pub(super) fn transact(
        stream: &mut UnixStream,
        method: &str,
        parameters: Value,
    ) -> Result<Value, String> {
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

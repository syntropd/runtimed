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

/// A held compute lease: the Varlink connection stays OPEN for the
/// lease lifetime. inferenced reclaims the lease the moment its client
/// disconnects ("orphaned lease"), so dropping this early voids the
/// admission. Released explicitly on unload.
pub struct LeasePermit {
    id: String,
    stream: UnixStream,
}

impl std::fmt::Debug for LeasePermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeasePermit").field("id", &self.id).finish()
    }
}

impl LeasePermit {
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
        Ok(Some(LeasePermit { id: id.to_string(), stream }))
    }

    fn connect(&self) -> Result<UnixStream, String> {
        let stream = UnixStream::connect(&self.socket).map_err(|e| e.to_string())?;
        stream.set_read_timeout(Some(RPC_TIMEOUT)).map_err(|e| e.to_string())?;
        stream.set_write_timeout(Some(RPC_TIMEOUT)).map_err(|e| e.to_string())?;
        Ok(stream)
    }

    fn transact(stream: &mut UnixStream, method: &str, parameters: Value) -> Result<Value, String> {
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
            let id = permit.id.clone();
            if let Err(e) = permit.release() {
                tracing::warn!(model = %name, lease = %id, "lease release failed: {e}");
            } else {
                tracing::info!(model = %name, lease = %id, "lease released");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;

    fn read_frame(stream: &mut UnixStream) -> Value {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 1024];
        loop {
            let n = stream.read(&mut chunk).unwrap();
            assert_ne!(n, 0, "client disconnected mid-frame");
            buf.extend_from_slice(&chunk[..n]);
            if buf.contains(&0) {
                break;
            }
        }
        buf.pop();
        serde_json::from_slice(&buf).unwrap()
    }

    fn write_frame(stream: &mut UnixStream, reply: &Value) {
        let mut bytes = serde_json::to_vec(reply).unwrap();
        bytes.push(0);
        stream.write_all(&bytes).unwrap();
    }

    fn fake_landlord_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("lease-{tag}-{}.sock", std::process::id()))
    }

    /// inferenced reclaims a lease the moment its client disconnects.
    /// The permit must hold the connection open after acquire and
    /// release over that same connection. Fails on a client that
    /// connects per call (lease instantly orphaned).
    #[test]
    fn permit_holds_connection_until_release() {
        let path = fake_landlord_path("hold");
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let req = read_frame(&mut sock);
            assert!(req["method"].as_str().unwrap().contains("AcquireLease"));
            write_frame(&mut sock, &json!({"parameters": {"lease_id": "lease-1"}}));
            // Client must STAY connected: idle read times out (held),
            // EOF means it dropped (lease would be reclaimed).
            sock.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
            let mut one = [0u8; 1];
            match sock.read(&mut one) {
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                    ) => {}
                Ok(0) => panic!("client disconnected right after acquire"),
                other => panic!("unexpected hold probe: {other:?}"),
            }
            // Release must arrive on this SAME connection.
            sock.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            let rel = read_frame(&mut sock);
            assert!(rel["method"].as_str().unwrap().contains("ReleaseLease"));
            assert_eq!(rel["parameters"]["lease_id"], "lease-1");
            write_frame(&mut sock, &json!({"parameters": {}}));
            // Client closes after release.
            assert_eq!(sock.read(&mut one).unwrap(), 0, "no clean close after release");
        });
        let client = LeaseClient::new(&path);
        let permit = client.acquire(1024).unwrap().expect("grant");
        std::thread::sleep(Duration::from_secs(1));
        permit.release().unwrap();
        server.join().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn refusal_fails_acquire() {
        let path = fake_landlord_path("refuse");
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let _ = read_frame(&mut sock);
            write_frame(
                &mut sock,
                &json!({"error": "io.systemd.inferenced1.ResourceExhaustion",
                        "parameters": {"error": "plane full"}}),
            );
        });
        let client = LeaseClient::new(&path);
        let err = client.acquire(1024).unwrap_err().to_string();
        assert!(err.contains("plane full"), "{err}");
        server.join().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_socket_loads_unenforced() {
        let client = LeaseClient::new("/nonexistent-dir-xyz/no.sock");
        assert!(client.acquire(1024).unwrap().is_none());
    }
}

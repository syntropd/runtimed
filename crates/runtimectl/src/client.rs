//! Varlink client for communicating with runtimed over Unix domain sockets.

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

/// Maximum size in bytes of a single Varlink reply on the wire.
pub const MAX_MSG_BYTES: usize = 1024 * 1024;

/// Varlink request envelope.
#[derive(Debug, Serialize)]
struct VarlinkRequest<'a> {
    method: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

/// Varlink reply envelope.
#[derive(Debug, Deserialize)]
struct VarlinkResponse {
    parameters: Option<Value>,
    error: Option<String>,
}

/// Client for issuing Varlink method calls to runtimed.
pub struct RuntimedClient {
    socket_path: String,
}

impl RuntimedClient {
    /// Creates a new client connected to the designated Unix socket.
    pub fn new<P: AsRef<Path>>(socket_path: P) -> Self {
        Self {
            socket_path: socket_path.as_ref().to_string_lossy().to_string(),
        }
    }

    /// Invokes a Varlink method and awaits a single reply.
    pub async fn call(&self, method: &str, parameters: Option<Value>) -> Result<Value> {
        let mut stream = UnixStream::connect(&self.socket_path)
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to connect to runtimed socket at {}: {}",
                    self.socket_path,
                    e
                )
            })?;

        let request = VarlinkRequest { method, parameters };
        let mut req_bytes = serde_json::to_vec(&request)?;
        req_bytes.push(0x00);
        stream.write_all(&req_bytes).await?;

        let mut buffer = Vec::with_capacity(4096);
        let mut chunk = [0u8; 1024];

        loop {
            let n = stream.read(&mut chunk).await?;
            if n == 0 {
                bail!("Connection closed by runtimed before complete reply");
            }
            buffer.extend_from_slice(&chunk[..n]);

            if buffer.len() > MAX_MSG_BYTES {
                return Err(anyhow!(
                    "Varlink reply exceeded {} bytes (peer misbehaving)",
                    MAX_MSG_BYTES
                ));
            }

            if let Some(pos) = buffer.iter().position(|&b| b == 0x00) {
                let reply_bytes = &buffer[..pos];
                let response: VarlinkResponse = serde_json::from_slice(reply_bytes)
                    .map_err(|e| anyhow!("Failed parsing Varlink reply from runtimed: {}", e))?;

                if let Some(err) = response.error {
                    bail!("{}", render_server_error(&err, response.parameters.as_ref()));
                }

                return response
                    .parameters
                    .ok_or_else(|| anyhow!("Varlink reply missing parameters field"));
            }
        }
    }
}

/// Human text for a daemon error reply: the error name plus the
/// server's `reason` when it sent one (that names the real cause,
/// e.g. a CPU-only binary asked to run CUDA).
fn render_server_error(err: &str, parameters: Option<&Value>) -> String {
    let detail = parameters
        .and_then(|p| p.get("reason"))
        .and_then(|r| r.as_str())
        .map(|r| format!(": {r}"))
        .unwrap_or_default();
    format!("runtimed returned error: {err}{detail}")
}

/// Fake daemon for command tests: serves each canned reply envelope on
/// its own connection (the client dials once per call).
#[cfg(test)]
pub(crate) mod test_support {
    use super::RuntimedClient;
    use serde_json::Value;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);

    pub(crate) fn fake_daemon(replies: Vec<Value>) -> (RuntimedClient, tokio::task::JoinHandle<()>) {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("runtimectl-test-{}-{id}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).expect("bind fake daemon socket");
        let client = RuntimedClient::new(&path);
        let handle = tokio::spawn(async move {
            for reply in replies {
                let accepted =
                    tokio::time::timeout(Duration::from_secs(10), listener.accept()).await;
                let Ok(Ok((mut sock, _))) = accepted else { break };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 1024];
                loop {
                    match sock.read(&mut chunk).await {
                        Ok(0) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&chunk[..n]);
                            if buf.contains(&0x00) {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                let bytes = serde_json::to_vec(&reply).unwrap();
                let _ = sock.write_all(&bytes).await;
                let _ = sock.write_all(&[0x00]).await;
            }
            let _ = std::fs::remove_file(&path);
        });
        (client, handle)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::fake_daemon;
    use serde_json::json;

    #[tokio::test]
    async fn call_returns_parameters() {
        let (client, server) = fake_daemon(vec![json!({ "parameters": { "status": "loaded" } })]);
        let res = client.call("io.syntrop.Runtime1.GetModelStatus", None).await.unwrap();
        assert_eq!(res["status"], "loaded");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn call_surfaces_daemon_error() {
        let (client, server) =
            fake_daemon(vec![json!({ "error": "io.syntrop.Runtime1.ModelNotFound" })]);
        let err = client
            .call("io.syntrop.Runtime1.UnloadModel", None)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("ModelNotFound"), "{err}");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn call_error_names_server_reason() {
        let (client, server) = fake_daemon(vec![json!({
            "error": "io.syntrop.Runtime1.GenerationFailed",
            "parameters": {
                "reason": "Hardware compute allocation failure: cuda backend needs a --features cuda build"
            }
        })]);
        let err = client
            .call("io.syntrop.Runtime1.Generate", None)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("GenerationFailed"), "{err}");
        assert!(err.contains("cuda backend needs a --features cuda build"), "{err}");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn call_to_missing_socket_errors() {
        let client = super::RuntimedClient::new("/nonexistent-dir-xyz/no.sock");
        let err = client
            .call("org.varlink.service.GetInfo", None)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("Failed to connect"), "{err}");
    }
}
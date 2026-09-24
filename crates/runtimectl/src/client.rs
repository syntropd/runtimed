//! Varlink client for communicating with runtimed over Unix domain sockets.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

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
            .with_context(|| format!("Failed to connect to runtimed socket at {}", self.socket_path))?;

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

            if let Some(pos) = buffer.iter().position(|&b| b == 0x00) {
                let reply_bytes = &buffer[..pos];
                let response: VarlinkResponse = serde_json::from_slice(reply_bytes)
                    .context("Failed parsing Varlink reply from runtimed")?;

                if let Some(err) = response.error {
                    bail!("runtimed returned error: {}", err);
                }

                return Ok(response.parameters.unwrap_or(Value::Null));
            }
        }
    }
}

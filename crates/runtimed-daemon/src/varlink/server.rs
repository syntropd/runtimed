//! Varlink Unix domain socket listener and protocol dispatcher for runtimed.

use super::protocol::{VarlinkCall, VarlinkReply};
use super::runtime1::Runtime1Handler;
use super::service::handle_service_call;
use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tracing::{debug, error, info, warn};

/// Varlink server instance listening on a Unix domain socket.
pub struct VarlinkServer {
    listener: UnixListener,
    handler: Arc<Runtime1Handler>,
}

impl VarlinkServer {
    /// Constructs a VarlinkServer from an open UnixListener.
    pub fn new(listener: UnixListener, handler: Runtime1Handler) -> Self {
        Self {
            listener,
            handler: Arc::new(handler),
        }
    }

    /// Runs the accept and dispatch loop until cancelled.
    pub async fn run(self) -> Result<()> {
        info!("runtimed Varlink server accepting connections");
        loop {
            match self.listener.accept().await {
                Ok((stream, _)) => {
                    let handler = Arc::clone(&self.handler);
                    tokio::spawn(async move {
                        if let Err(e) = handle_client(stream, handler).await {
                            debug!("Client connection closed: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Error accepting connection: {}", e);
                    return Err(e).context("Failed accepting Varlink stream");
                }
            }
        }
    }
}

async fn handle_client(mut stream: UnixStream, handler: Arc<Runtime1Handler>) -> Result<()> {
    let mut buffer = Vec::with_capacity(4096);
    let mut chunk = [0u8; 1024];

    loop {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            break;
        }

        buffer.extend_from_slice(&chunk[..n]);

        while let Some(pos) = buffer.iter().position(|&b| b == 0x00) {
            let message_bytes = buffer.drain(..pos).collect::<Vec<u8>>();
            buffer.remove(0);

            if message_bytes.is_empty() {
                continue;
            }

            let call: VarlinkCall = match serde_json::from_slice(&message_bytes) {
                Ok(c) => c,
                Err(e) => {
                    warn!("Invalid Varlink call payload: {}", e);
                    let reply = VarlinkReply::err("org.varlink.service.InvalidParameter", None);
                    stream.write_all(&reply.to_bytes()).await?;
                    continue;
                }
            };

            let reply = dispatch_call(&call, &handler).await;
            stream.write_all(&reply.to_bytes()).await?;
        }
    }

    Ok(())
}

async fn dispatch_call(call: &VarlinkCall, handler: &Runtime1Handler) -> VarlinkReply {
    if let Some(reply) = handle_service_call(&call.method, call.parameters.as_ref()) {
        return reply;
    }

    if let Some(reply) = handler.handle_call(&call.method, call.parameters.as_ref()).await {
        return reply;
    }

    VarlinkReply::err(
        "org.varlink.service.MethodNotFound",
        Some(serde_json::json!({ "method": call.method })),
    )
}

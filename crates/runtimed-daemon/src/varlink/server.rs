//! Varlink Unix domain socket listener and protocol dispatcher for runtimed.

use super::auth::{authorize_peer, TrustedGroup};
use super::protocol::{VarlinkCall, VarlinkReply};
use super::runtime1::Runtime1Handler;
use super::service::handle_service_call;
use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

/// Maximum size in bytes of a single Varlink request on the wire.
pub const MAX_MSG_BYTES: usize = 1024 * 1024;

/// Varlink server instance listening on a Unix domain socket.
pub struct VarlinkServer {
    listener: UnixListener,
    handler: Arc<Runtime1Handler>,
    trusted_group: TrustedGroup,
}

impl VarlinkServer {
    /// Constructs a VarlinkServer from an open UnixListener.
    pub fn new(listener: UnixListener, handler: Runtime1Handler, trusted_group: TrustedGroup) -> Self {
        Self {
            listener,
            handler: Arc::new(handler),
            trusted_group,
        }
    }

    /// Runs the accept and dispatch loop until `shutdown` fires.
    ///
    /// Per-connection tasks are tracked in a `JoinSet`. When shutdown is
    /// signalled the function waits for the JoinSet to drain under a
    /// bounded timeout (so a stuck client cannot prevent shutdown) and
    /// returns. Without this, per-connection handlers were abandoned
    /// mid-request as soon as the runtime tore down on `main` return.
    pub async fn run(
        self,
        mut shutdown: watch::Receiver<bool>,
        drain_timeout: std::time::Duration,
    ) -> Result<()> {
        info!("runtimed Varlink server accepting connections");
        let mut tasks: tokio::task::JoinSet<()> = tokio::task::JoinSet::new();
        let client_shutdown = shutdown.clone();
        loop {
            tokio::select! {
                biased;
                _ = shutdown.changed() => {
                    info!("Varlink server observed shutdown signal; draining in-flight clients");
                    let deadline = std::time::Instant::now() + drain_timeout;
                    loop {
                        match tokio::time::timeout_at(
                            deadline.into(),
                            tasks.join_next(),
                        )
                        .await
                        {
                            Ok(Some(Ok(()))) => { /* drained one */ }
                            Ok(Some(Err(e))) => {
                                warn!("Client task panicked during drain: {}", e);
                            }
                            Ok(None) => break, // JoinSet empty
                            Err(_) => break,   // timeout
                        }
                    }
                    return Ok(());
                }
                // Reap completed client tasks so the JoinSet doesn't grow
                // unbounded over the daemon's lifetime.
                Some(_) = tasks.join_next(), if !tasks.is_empty() => {}
                res = self.listener.accept() => match res {
                    Ok((stream, _)) => {
                        let handler = Arc::clone(&self.handler);
                        let group = self.trusted_group;
                        let cs = client_shutdown.clone();
                        tasks.spawn(async move {
                            if let Err(e) = handle_client(stream, handler, group, cs).await {
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
}

/// Outcome of a single read attempt against the wire.
enum ReadOutcome {
    /// Clean EOF (no buffered bytes remain).
    Eof,
    /// EOF arrived mid-frame (buffer non-empty, no NUL terminator).
    /// Caller should emit a `ProtocolError` reply and drop the connection.
    Truncated,
    /// Frame read (caller processes `frame`).
    Frame(Vec<u8>),
    /// Frame exceeded `MAX_MSG_BYTES`; caller emits `ProtocolError`.
    Overflow,
}

/// Reads a single Varlink frame (up to `MAX_MSG_BYTES`) from the stream.
///
/// A frame is a NUL-terminated JSON payload. Multiple frames may be
/// coalesced into a single read; we always return the leftmost complete
/// frame and keep any partial bytes (and any bytes after the NUL) in
/// the buffer for the next call. EOF with a buffered complete frame is
/// a normal end-of-stream — return the buffered frame, then EOF on the
/// next call.
async fn read_frame(stream: &mut UnixStream, buffer: &mut Vec<u8>) -> Result<ReadOutcome> {
    // Drain any buffered complete frame first so a client that pipelined
    // multiple frames in one read gets every one of them.
    if let Some(pos) = buffer.iter().position(|&b| b == 0x00) {
        let frame = buffer.drain(..pos).collect::<Vec<u8>>();
        buffer.remove(0);
        return Ok(ReadOutcome::Frame(frame));
    }

    let mut chunk = [0u8; 1024];
    loop {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return Ok(if buffer.is_empty() {
                ReadOutcome::Eof
            } else {
                ReadOutcome::Truncated
            });
        }
        buffer.extend_from_slice(&chunk[..n]);
        // Use `>=` so we never admit the (MAX_MSG_BYTES + 1)-th byte:
        // a hostile client cannot slip one extra chunk above the cap.
        if buffer.len() >= MAX_MSG_BYTES {
            return Ok(ReadOutcome::Overflow);
        }
        if let Some(pos) = buffer.iter().position(|&b| b == 0x00) {
            let frame = buffer.drain(..pos).collect::<Vec<u8>>();
            buffer.remove(0);
            return Ok(ReadOutcome::Frame(frame));
        }
    }
}

/// Handles a single Varlink client connection until EOF, fatal error, or
/// shutdown signal. Exposed publicly so QA tests can drive the buffer /
/// auth / dispatch paths via `UnixStream::pair`.
pub async fn handle_client(
    mut stream: UnixStream,
    handler: Arc<Runtime1Handler>,
    trusted_group: TrustedGroup,
    shutdown: watch::Receiver<bool>,
) -> Result<()> {
    if let Err(e) = authorize_peer(&stream, trusted_group) {
        warn!("Rejecting Varlink connection: {}", e);
        let reply = VarlinkReply::err("io.syntrop.Runtime1.PermissionDenied", None);
        let _ = stream.write_all(&reply.to_bytes()).await;
        return Ok(());
    }

    let mut buffer = Vec::with_capacity(4096);
    let mut shutdown = shutdown;

    loop {
        tokio::select! {
            biased;
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("Client connection aborted by shutdown signal");
                    return Ok(());
                }
            }
            res = read_frame(&mut stream, &mut buffer) => {
                let frame = match res? {
                    ReadOutcome::Frame(f) => f,
                    ReadOutcome::Eof => break,
                    ReadOutcome::Truncated => {
                        warn!("Varlink connection closed mid-frame; emitting ProtocolError");
                        let reply = VarlinkReply::err("org.varlink.service.ProtocolError", None);
                        let _ = stream.write_all(&reply.to_bytes()).await;
                        break;
                    }
                    ReadOutcome::Overflow => {
                        let reply = VarlinkReply::err("org.varlink.service.ProtocolError", None);
                        let _ = stream.write_all(&reply.to_bytes()).await;
                        break;
                    }
                };

                if frame.is_empty() {
                    continue;
                }

                let call: VarlinkCall = match serde_json::from_slice(&frame) {
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
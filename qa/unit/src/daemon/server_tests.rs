//! Unit QA tests for the Varlink server connection framing and overflow guard.

#[cfg(test)]
mod tests {
    use runtimed_core::model::ModelManager;
    use runtimed_daemon::varlink::{handle_client, Runtime1Handler, TrustedGroup};
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixStream;

    /// Drives the MAX_MSG_BYTES overflow branch using `UnixStream::pair()`.
    /// Sends > 1 MiB without a NUL terminator and verifies the server
    /// writes a `ProtocolError` reply and closes.
    #[tokio::test]
    async fn test_handle_client_overflow_drops_connection() {
        let tmp = tempdir().unwrap();
        let manager = Arc::new(ModelManager::new(tmp.path()));
        let handler = Arc::new(Runtime1Handler::new(manager));
        let own_gid = unsafe { libc::getgid() };
        let trusted = TrustedGroup::from_gid(own_gid);

        let (client, server) = UnixStream::pair().unwrap();
        let (mut read_half, mut write_half) = client.into_split();
        let handler_clone = Arc::clone(&handler);
        let server_task = tokio::spawn(async move {
            let (_tx, rx) = tokio::sync::watch::channel::<bool>(false);
            handle_client(server, handler_clone, trusted, rx).await
        });

        // Send 1 MiB + 1024 bytes of "A" with no NUL terminator. The
        // server is expected to detect the overflow, write a reply and
        // close; BrokenPipe on the write side is therefore expected.
        let payload = vec![b'A'; 1024 * 1024 + 1024];
        let _ = write_half.write_all(&payload).await;
        drop(write_half);

        // Read until EOF and confirm a ProtocolError reply was emitted.
        let mut buf = Vec::new();
        let _ = read_half.read_to_end(&mut buf).await;
        assert!(!buf.is_empty(), "server should write a ProtocolError reply");
        let parsed: serde_json::Value =
            serde_json::from_slice(&buf[..buf.len() - 1]).unwrap();
        assert_eq!(parsed["error"], "org.varlink.service.ProtocolError");

        let _ = server_task.await;
    }

    /// Sends a single valid Varlink frame and verifies a reply is returned.
    #[tokio::test]
    async fn test_handle_client_echoes_valid_call() {
        let tmp = tempdir().unwrap();
        let manager = Arc::new(ModelManager::new(tmp.path()));
        let handler = Arc::new(Runtime1Handler::new(manager));
        let own_gid = unsafe { libc::getgid() };
        let trusted = TrustedGroup::from_gid(own_gid);

        let (client, server) = UnixStream::pair().unwrap();
        let (mut read_half, mut write_half) = client.into_split();
        let handler_clone = Arc::clone(&handler);
        let server_task = tokio::spawn(async move {
            let (_tx, rx) = tokio::sync::watch::channel::<bool>(false);
            handle_client(server, handler_clone, trusted, rx).await
        });

        let frame = b"{\"method\":\"org.varlink.service.GetInfo\"}\x00";
        write_half.write_all(frame).await.unwrap();

        // Drop the writer so the server sees EOF and closes the connection.
        drop(write_half);

        let mut buf = Vec::new();
        let _ = read_half.read_to_end(&mut buf).await;
        let parsed: serde_json::Value =
            serde_json::from_slice(&buf[..buf.len() - 1]).unwrap();
        assert_eq!(parsed["parameters"]["product"], "runtimed");

        let _ = server_task.await;
    }
}
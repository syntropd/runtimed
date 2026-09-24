//! Edge tests for Varlink protocol framing.

#[cfg(test)]
mod tests {
    use runtimed_core::model::ModelManager;
    use runtimed_daemon::varlink::{
        handle_client, Runtime1Handler, TrustedGroup, MAX_MSG_BYTES,
    };
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixStream;

    #[test]
    fn test_malformed_varlink_call() {
        let bad = b"{\"method\":}";
        let res: Result<runtimed_daemon::varlink::VarlinkCall, _> =
            serde_json::from_slice(bad);
        assert!(res.is_err());
    }

    #[test]
    fn test_varlink_empty_error_reply() {
        let reply = runtimed_daemon::varlink::VarlinkReply::err("RuntimePreempted", None);
        let bytes = reply.to_bytes();
        assert_eq!(*bytes.last().unwrap(), 0x00);
        let parsed: serde_json::Value =
            serde_json::from_slice(&bytes[..bytes.len() - 1]).unwrap();
        assert_eq!(parsed["error"], "RuntimePreempted");
    }

    /// Edge case: oversized frame triggers a `ProtocolError` reply and the
    /// connection is closed. This is the live regression test for the
    /// memory-DoS path against the daemon's wire-protocol handler.
    #[tokio::test]
    async fn test_oversized_message_drops_connection() {
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

        // MAX_MSG_BYTES + 64 bytes, no NUL terminator. The server is
        // expected to write the ProtocolError reply and drop us; ignore
        // BrokenPipe from the write side.
        let payload = vec![b'B'; MAX_MSG_BYTES + 64];
        let _ = write_half.write_all(&payload).await;
        drop(write_half);

        let mut buf = Vec::new();
        let _ = read_half.read_to_end(&mut buf).await;
        assert!(!buf.is_empty(), "expected a ProtocolError reply on the wire");
        let parsed: serde_json::Value =
            serde_json::from_slice(&buf[..buf.len() - 1]).unwrap();
        assert_eq!(parsed["error"], "org.varlink.service.ProtocolError");

        let _ = server_task.await;
    }
}
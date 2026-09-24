# runtimed Review — Correctness Defects (§2)

## 2.1 `parse_listen_fds` discards `fd_handoff_listener` for runtimed

**File**: `crates/runtimed-daemon/src/activation.rs:11-15`

```rust
pub struct ActivatedSockets {
    /// Primary Varlink IPC Unix listener (FD 3).
    pub varlink_listener: Option<UnixListener>,
}
```

`ActivatedSockets` is correct for runtimed's single-listener design (no separate FD-handoff socket). Confirmed in `main.rs:53-71` — only the Varlink listener is consumed. **No defect.**

## 2.2 `notify.rs` uses `unsafe` `sendto` with potentially-wrong `sockaddr` size

**File**: `crates/runtimed-daemon/src/notify.rs:32-44`

```rust
use std::os::unix::io::AsRawFd;
unsafe {
    let addr_ptr = &address as *const _ as *const libc::sockaddr;
    let addr_len = std::mem::size_of_val(&address) as libc::socklen_t;
    let res = libc::sendto(
        socket.as_raw_fd(),
        state.as_ptr() as *const libc::c_void,
        state.len(),
        libc::MSG_NOSIGNAL,
        addr_ptr,
        addr_len,
    );
    res >= 0
}
```

`rustix::net::SocketAddrUnix` is a Rust enum that may carry alignment padding or a discriminator byte not present in the kernel's `sockaddr_un`. Using `size_of_val` for `socklen_t` can pass an over-long value, leading the kernel to read past the actual address bytes. The `rustix` crate provides a safe `sendto_unix` wrapper that constructs the correct `sockaddr_un` for both filesystem and abstract namespaces.

**Recommended fix**: replace the unsafe block with `rustix::net::sendto_unix(socket.as_fd(), state.as_bytes(), rustix::net::SendFlags::NOSIGNAL, &address)`. Drops the `unsafe`.

## 2.3 `notify_status` does not sanitize embedded newlines

**File**: `crates/runtimed-daemon/src/notify.rs:53-55`

```rust
pub fn notify_status(status: &str) -> bool {
    send_notify(&format!("STATUS={}\n", status))
}
```

sd_notify is line-based; an embedded `\n` in the status string terminates the variable early, allowing the remainder of the string to be parsed as additional variables. A status containing `"STATUS=ready\nWATCHDOG_USEC=9999999999"` would spoof the watchdog timeout.

**Recommended fix**: strip `\n` and `\r` before formatting the message (same pattern as contextd pre-fix and modeld fixed).

## 2.4 `notify_send` has no oversize message guard

**File**: `crates/runtimed-daemon/src/notify.rs:7-45`

The systemd notification datagram protocol caps messages at ~8 MiB; `send_notify` does not reject messages over that limit, so a buggy caller could silently fail or split across multiple datagrams.

**Recommended fix**: check `state.len()` against `NOTIFY_MAX = 8 * 1024 * 1024` and return `false` early.

## 2.5 `VarlinkReply::to_bytes` silently returns `[0x00]` on serialization failure

**File**: `crates/runtimed-daemon/src/varlink/protocol.rs:52-56`

```rust
pub fn to_bytes(&self) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(self).unwrap_or_default();
    bytes.push(0x00);
    bytes
}
```

`unwrap_or_default` returns an empty Vec if serialization fails, which after the NUL append becomes the single byte `\0`. The peer sees a malformed reply. Same defect class as the pre-fix contextd version.

**Recommended fix**: `unwrap_or_else(|_| b"{}".to_vec())` so a serialization failure produces a valid empty JSON reply.

## 2.6 `VarlinkCall.parameters` defaults silently fall back to Null

**File**: `crates/runtimectl/src/client.rs:69`

```rust
return Ok(response.parameters.unwrap_or(Value::Null));
```

If the server returns `{ "error": null }` without a `parameters` key, the client silently substitutes `Value::Null` rather than reporting the missing payload. Hides contract bugs.

**Recommended fix**: when `response.error.is_none()`, require `response.parameters` to be present and return an `Err` if absent.

## 2.7 `main.rs` `tokio::select!` arms race the Varlink server

**File**: `crates/runtimed-daemon/src/main.rs:81-93`

```rust
tokio::select! {
    res = server.run() => { ... }
    _ = sigterm.recv() => { ... }
    _ = sigint.recv() => { ... }
}
notify_stopping();
```

`server.run()` is an infinite loop on `accept()`. When SIGTERM arrives, `select!` cancels the server task mid-`accept`. The runtime then drops the `JoinHandle` without awaiting; any in-flight per-connection task is cancelled. `notify_stopping` is sent before tasks drain.

**Recommended fix**: pass a `tokio::sync::watch::Receiver<bool>` shutdown signal into `server.run`, send `true` on SIGTERM, await the server's `JoinHandle` with a bounded timeout, then `notify_stopping`. Same pattern as contextd/modeld fix.

//! Pure Rust systemd sd_notify implementation for runtimed.

use std::env;
use std::os::unix::net::UnixDatagram;

/// Sends a formatted raw notification string to `$NOTIFY_SOCKET`.
pub fn send_notify(state: &str) -> bool {
    let socket_path = match env::var("NOTIFY_SOCKET") {
        Ok(path) => path,
        Err(_) => return false,
    };

    let socket = match UnixDatagram::unbound() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let target = if let Some(stripped) = socket_path.strip_prefix('@') {
        let mut bytes = vec![0u8];
        bytes.extend_from_slice(stripped.as_bytes());
        bytes
    } else {
        socket_path.into_bytes()
    };

    let address = match rustix::net::SocketAddrUnix::new(&target) {
        Ok(addr) => addr,
        Err(_) => return false,
    };

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
}

/// Emits the READY=1 readiness notification.
pub fn notify_ready() -> bool {
    send_notify("READY=1\n")
}

/// Emits an updated STATUS string.
pub fn notify_status(status: &str) -> bool {
    send_notify(&format!("STATUS={}\n", status))
}

/// Emits the WATCHDOG=1 heartbeat ping.
pub fn notify_watchdog() -> bool {
    send_notify("WATCHDOG=1\n")
}

/// Emits the STOPPING=1 shutdown signal.
pub fn notify_stopping() -> bool {
    send_notify("STOPPING=1\n")
}

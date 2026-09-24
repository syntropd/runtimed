//! Pure Rust systemd sd_notify implementation for runtimed.

use std::env;
use std::os::unix::io::AsFd;
use std::os::unix::net::UnixDatagram;

/// Maximum size of a single sd_notify datagram (matches systemd limit).
pub const NOTIFY_MAX: usize = 8 * 1024 * 1024;

/// Builds a `SocketAddrUnix` for `$NOTIFY_SOCKET`, supporting both
/// filesystem paths (`/run/systemd/notify`) and abstract namespaces
/// (`@notify`).
///
/// Abstract names containing interior NUL bytes are rejected even though
/// the kernel would accept them: a malformed name would silently send
/// to a name nobody listens on, defeating the readiness/watchdog
/// protocol. Empty names are also rejected for the same reason.
///
/// We dispatch on the leading `@` because `SocketAddrUnix::new` always
/// appends a trailing NUL even for abstract namespaces, which causes
/// `sendto` to encode the abstract name with the wrong sun_path length
/// and never match systemd's listener. `new_abstract_name` constructs
/// the abstract sockaddr with the correct length.
fn notify_address(socket_path: &str) -> Option<rustix::net::SocketAddrUnix> {
    if let Some(stripped) = socket_path.strip_prefix('@') {
        if stripped.is_empty() || stripped.as_bytes().contains(&0) {
            return None;
        }
        rustix::net::SocketAddrUnix::new_abstract_name(stripped.as_bytes()).ok()
    } else if socket_path.is_empty() {
        None
    } else {
        rustix::net::SocketAddrUnix::new(socket_path).ok()
    }
}

/// Sanitizes an sd_notify variable value by stripping line terminators.
/// Public only under the `qa-test-helpers` cargo feature so QA tests
/// can verify the stripping behavior directly.
#[cfg(any(test, feature = "qa-test-helpers"))]
pub fn sanitize_value(value: &str) -> String {
    value.chars().filter(|&c| c != '\n' && c != '\r').collect()
}

#[cfg(not(any(test, feature = "qa-test-helpers")))]
fn sanitize_value(value: &str) -> String {
    value.chars().filter(|&c| c != '\n' && c != '\r').collect()
}

/// Sends a formatted raw notification string to `$NOTIFY_SOCKET`.
pub fn send_notify(state: &str) -> bool {
    if !within_size_limit(state) {
        return false;
    }

    let socket_path = match env::var("NOTIFY_SOCKET") {
        Ok(path) if !path.is_empty() => path,
        _ => return false,
    };

    let socket = match UnixDatagram::unbound() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let address = match notify_address(&socket_path) {
        Some(addr) => addr,
        None => return false,
    };

    rustix::net::sendto_unix(socket.as_fd(), state.as_bytes(), rustix::net::SendFlags::NOSIGNAL, &address).is_ok()
}

/// Pure size-guard predicate. Public under the `qa-test-helpers`
/// feature so QA tests can exercise the boundary without depending on
/// the socket layer.
#[cfg(any(test, feature = "qa-test-helpers"))]
pub fn within_size_limit(state: &str) -> bool {
    state.len() <= NOTIFY_MAX
}

#[cfg(not(any(test, feature = "qa-test-helpers")))]
fn within_size_limit(state: &str) -> bool {
    state.len() <= NOTIFY_MAX
}

/// Emits the READY=1 readiness notification.
pub fn notify_ready() -> bool {
    send_notify("READY=1\n")
}

/// Emits an updated STATUS string. Embedded line breaks are stripped so the
/// payload cannot inject additional variables into the line-based protocol.
pub fn notify_status(status: &str) -> bool {
    send_notify(&format!("STATUS={}\n", sanitize_value(status)))
}

/// Emits the WATCHDOG=1 heartbeat ping.
pub fn notify_watchdog() -> bool {
    send_notify("WATCHDOG=1\n")
}

/// Emits the STOPPING=1 shutdown signal.
pub fn notify_stopping() -> bool {
    send_notify("STOPPING=1\n")
}
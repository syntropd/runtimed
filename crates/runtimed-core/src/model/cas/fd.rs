//! SCM_RIGHTS client for retrieving sealed model descriptors from modeld CAS.
//!
//! Connects over `/run/syntrop/modeld-fd.sock` to request a verified, sealed
//! `memfd_create` descriptor passed via `SCM_RIGHTS`. This enables true zero-copy
//! inference loading across unprivileged daemon boundaries without re-reading
//! multi-gigabyte weight files from disk.

use rustix::net::{
    recvmsg, RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags,
};
use std::io::{IoSliceMut, Write};
use std::os::unix::io::{AsFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

/// Default Unix domain socket endpoint for modeld SCM_RIGHTS descriptor handoff.
pub const DEFAULT_MODELD_FD_SOCK: &str = "/run/syntrop/modeld-fd.sock";

/// Connects to modeld FD handoff socket and requests a sealed memfd for `model_id`.
pub fn fetch_model_fd(model_id: &str) -> Option<OwnedFd> {
    let sock_path = std::env::var("MODELD_FD_SOCK")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_MODELD_FD_SOCK));

    if !sock_path.exists() {
        return None;
    }

    let mut stream = match UnixStream::connect(&sock_path) {
        Ok(s) => s,
        Err(e) => {
            tracing::debug!(sock = %sock_path.display(), error = %e, "modeld-fd socket connect failed");
            return None;
        }
    };

    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));

    let request = serde_json::json!({ "id": model_id });
    let mut payload = match serde_json::to_vec(&request) {
        Ok(p) => p,
        Err(_) => return None,
    };
    payload.push(0x00);

    if stream.write_all(&payload).is_err() {
        return None;
    }

    let mut buf = [0u8; 1024];
    let mut space = [0u8; rustix::cmsg_space!(ScmRights(1))];
    let mut ancillary = RecvAncillaryBuffer::new(&mut space);
    let mut iov = [IoSliceMut::new(&mut buf)];

    let msg = match recvmsg(
        stream.as_fd(),
        &mut iov,
        &mut ancillary,
        RecvFlags::CMSG_CLOEXEC,
    ) {
        Ok(m) => m,
        Err(e) => {
            tracing::debug!(error = %e, "recvmsg from modeld-fd failed");
            return None;
        }
    };

    if msg.bytes == 0 {
        return None;
    }

    for cmsg in ancillary.drain() {
        if let RecvAncillaryMessage::ScmRights(mut fds) = cmsg {
            if let Some(owned) = fds.next() {
                tracing::info!(model = %model_id, "obtained sealed memfd descriptor from modeld CAS");
                return Some(owned);
            }
        }
    }

    None
}

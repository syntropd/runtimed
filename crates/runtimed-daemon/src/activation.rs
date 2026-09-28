//! Systemd socket activation implementation for runtimed.

use std::env;
use std::os::unix::io::{FromRawFd, RawFd};
use tokio::net::UnixListener;

/// The starting file descriptor index passed by systemd (FD 3).
pub const SD_LISTEN_FDS_START: RawFd = 3;

/// Adopted systemd sockets for runtimed.
#[derive(Debug)]
pub struct ActivatedSockets {
    /// Primary Varlink IPC Unix listener (FD 3).
    pub varlink_listener: Option<UnixListener>,
}

/// Parses systemd environment variables and adopts pre-bound Unix sockets.
pub fn parse_listen_fds() -> ActivatedSockets {
    let pid_matches = match env::var("LISTEN_PID") {
        Ok(pid_str) => pid_str.parse::<u32>().map(|p| p == std::process::id()).unwrap_or(false),
        Err(_) => false,
    };

    if !pid_matches {
        return ActivatedSockets { varlink_listener: None };
    }

    let count: usize = env::var("LISTEN_FDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let varlink_listener = if count >= 1 {
        adopt_unix_listener(SD_LISTEN_FDS_START)
    } else {
        None
    };

    ActivatedSockets { varlink_listener }
}

fn adopt_unix_listener(fd: RawFd) -> Option<UnixListener> {
    unsafe {
        let std_listener = std::os::unix::net::UnixListener::from_raw_fd(fd);
        let _ = std_listener.set_nonblocking(true);
        UnixListener::from_std(std_listener).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, val: &str) -> Self {
            let prev = env::var(key).ok();
            env::set_var(key, val);
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => env::set_var(self.key, v),
                None => env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn pid_gating_never_adopts_unexpected_fds() {
        // One function: LISTEN_* is process-global and must not be
        // touched by parallel tests. LISTEN_FDS stays 0 whenever the
        // pid matches, so fd 3 is never adopted (it belongs to us).
        let _pid = EnvGuard::set("LISTEN_PID", "not-a-number");
        let _fds = EnvGuard::set("LISTEN_FDS", "3");
        assert!(parse_listen_fds().varlink_listener.is_none());

        env::set_var("LISTEN_PID", std::process::id().wrapping_add(1).to_string());
        assert!(parse_listen_fds().varlink_listener.is_none());

        env::remove_var("LISTEN_PID");
        assert!(parse_listen_fds().varlink_listener.is_none());

        env::set_var("LISTEN_PID", std::process::id().to_string());
        env::remove_var("LISTEN_FDS");
        assert!(parse_listen_fds().varlink_listener.is_none());
    }
}

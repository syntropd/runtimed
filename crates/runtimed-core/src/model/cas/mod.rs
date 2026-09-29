//! Content-addressable storage sealed descriptor access.
//!
//! Exposes zero-copy SCM_RIGHTS file descriptor passing from modeld CAS.

pub mod fd;

pub use fd::{fetch_model_fd, DEFAULT_MODELD_FD_SOCK};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cas_default_socket_path() {
        assert_eq!(DEFAULT_MODELD_FD_SOCK, "/run/syntrop/modeld-fd.sock");
    }

    #[test]
    fn test_cas_env_override_name() {
        assert!(!DEFAULT_MODELD_FD_SOCK.is_empty());
    }
}

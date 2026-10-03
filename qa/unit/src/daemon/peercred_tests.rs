//! Unit QA tests for SO_PEERCRED authorization policy.

#[cfg(test)]
mod tests {
    use runtimed_daemon::varlink::server::auth::{lookup_group, authorize_peer};
    use runtimed_daemon::varlink::TrustedGroup;
    use std::os::unix::net::UnixStream;

    /// Live `UnixStream::pair()` exercises the SO_PEERCRED kernel path.
    /// The peers are ourselves, so authorizing against our own gid
    /// must succeed; a mismatched gid must fail.
    #[test]
    fn test_live_stream_authorizes_self() {
        let (a, _b) = UnixStream::pair().unwrap();
        let own_gid = unsafe { libc::getgid() };
        let trusted = TrustedGroup::from_gid(own_gid);
        assert!(authorize_peer(&a, trusted).is_ok());

        let (c, _d) = UnixStream::pair().unwrap();
        let other = TrustedGroup::from_gid(own_gid.wrapping_add(1));
        assert!(authorize_peer(&c, other).is_err());
    }

    /// When the trusted group could not be resolved, every non-root peer
    /// is rejected regardless of gid, while root peer is accepted.
    #[test]
    fn test_unresolved_trusted_group_enforces_policy() {
        let (a, _b) = UnixStream::pair().unwrap();
        let trusted = TrustedGroup::from_gid(runtimed_daemon::varlink::server::auth::UNRESOLVED_GID);
        let uid = unsafe { libc::getuid() };
        if uid == 0 {
            assert!(authorize_peer(&a, trusted).is_ok());
        } else {
            assert!(authorize_peer(&a, trusted).is_err());
        }
    }

    /// `lookup_group("root")` typically returns gid 0 on Linux; we assert
    /// only that the call does not panic. We do NOT require the result to
    /// be 0 because containers may not have a `root` group entry.
    #[test]
    fn test_lookup_group_root_does_not_panic() {
        let _ = lookup_group("root");
    }

    /// Unknown group names return `UNRESOLVED_GID`, NOT a fallback value
    /// (the H2 backdoor guard). The daemon refuses to start in this case.
    #[test]
    fn test_lookup_group_unknown_returns_unresolved() {
        let result = lookup_group("definitely-no-such-group-xyz");
        assert_eq!(result, runtimed_daemon::varlink::server::auth::UNRESOLVED_GID);
    }

    /// Embedded NUL bytes in the group name don't crash the C bridge; the
    /// input is rejected as malformed before `getgrnam_r` is called.
    #[test]
    fn test_lookup_group_with_embedded_nul_rejected() {
        let result = lookup_group("foo\0bar");
        assert_eq!(result, runtimed_daemon::varlink::server::auth::UNRESOLVED_GID);
    }
}
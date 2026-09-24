//! Unit QA tests for SO_PEERCRED authorization policy.

#[cfg(test)]
mod tests {
    use runtimed_daemon::varlink::auth::{
        authorize_ucred, lookup_group, make_test_ucred,
    };
    use runtimed_daemon::varlink::TrustedGroup;
    use std::os::unix::net::UnixStream;

    /// Synthetic root peer (uid=0) is always trusted.
    #[test]
    fn test_root_peer_is_trusted() {
        let cred = make_test_ucred(1, 0, 1234);
        let trusted = TrustedGroup::from_gid(5678);
        assert!(authorize_ucred(cred, trusted).is_ok());
    }

    /// Synthetic peer whose gid matches the trusted group is accepted.
    #[test]
    fn test_matching_gid_peer_is_trusted() {
        let cred = make_test_ucred(1, 1000, 5678);
        let trusted = TrustedGroup::from_gid(5678);
        assert!(authorize_ucred(cred, trusted).is_ok());
    }

    /// Synthetic peer whose gid does NOT match and uid != 0 is rejected.
    #[test]
    fn test_nonmatching_gid_peer_is_rejected() {
        let cred = make_test_ucred(1, 1000, 1234);
        let trusted = TrustedGroup::from_gid(5678);
        assert!(authorize_ucred(cred, trusted).is_err());
    }

    /// Live `UnixStream::pair()` exercises the SO_PEERCRED kernel path.
    /// The peers are ourselves, so authorizing against our own gid
    /// must succeed; a mismatched gid must fail.
    #[test]
    fn test_live_stream_authorizes_self() {
        let (a, _b) = UnixStream::pair().unwrap();
        let own_gid = unsafe { libc::getgid() };
        let trusted = TrustedGroup::from_gid(own_gid);
        assert!(runtimed_daemon::varlink::auth::authorize_peer(&a, trusted).is_ok());

        let (c, _d) = UnixStream::pair().unwrap();
        let other = TrustedGroup::from_gid(own_gid.wrapping_add(1));
        assert!(runtimed_daemon::varlink::auth::authorize_peer(&c, other).is_err());
    }

    /// When the trusted group could not be resolved, every non-root peer
    /// is rejected regardless of gid. This is the H2 backdoor guard.
    #[test]
    fn test_unresolved_trusted_group_rejects_non_root() {
        let cred = make_test_ucred(1, 1000, 0); // even gid 0 (root group) is rejected
        let trusted = TrustedGroup::from_gid(runtimed_daemon::varlink::auth::UNRESOLVED_GID);
        assert!(authorize_ucred(cred, trusted).is_err());
    }

    /// When the trusted group could not be resolved, root is still
    /// trusted (systemd activation hands off the FD before privileges
    /// drop).
    #[test]
    fn test_unresolved_trusted_group_still_trusts_root() {
        let cred = make_test_ucred(1, 0, 1234);
        let trusted = TrustedGroup::from_gid(runtimed_daemon::varlink::auth::UNRESOLVED_GID);
        assert!(authorize_ucred(cred, trusted).is_ok());
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
        assert_eq!(result, runtimed_daemon::varlink::auth::UNRESOLVED_GID);
    }

    /// Embedded NUL bytes in the group name don't crash the C bridge; the
    /// input is rejected as malformed before `getgrnam_r` is called.
    #[test]
    fn test_lookup_group_with_embedded_nul_rejected() {
        let result = lookup_group("foo\0bar");
        assert_eq!(result, runtimed_daemon::varlink::auth::UNRESOLVED_GID);
    }
}
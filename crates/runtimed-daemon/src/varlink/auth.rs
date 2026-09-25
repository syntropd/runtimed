//! Peer-credential authorization for Varlink connections.
//!
//! rustix reads `SO_PEERCRED` so even if the socket file mode or systemd
//! activation is misconfigured the daemon only accepts clients whose uid
//! is root or whose gid matches the configured trusted group.

use anyhow::{anyhow, Result};
use std::os::fd::AsFd;

/// Sentinel for "trusted group could not be resolved at startup". `u32::MAX`
/// is used because no real gid can ever be that large and it lets the
/// policy reject every non-root peer instead of silently falling back to
/// a permissive value (e.g. gid 0).
pub const UNRESOLVED_GID: u32 = u32::MAX;

/// Group identifier used for the trusted-client ACL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustedGroup {
    /// Group identifier resolved at startup.
    gid: u32,
}

impl TrustedGroup {
    /// Builds a `TrustedGroup` from a raw gid.
    pub fn from_gid(gid: u32) -> Self {
        Self { gid }
    }

    /// Returns the underlying gid.
    pub fn gid(self) -> u32 {
        self.gid
    }

    /// Returns true when the lookup failed or the group could not be resolved.
    pub fn is_unresolved(self) -> bool {
        self.gid == UNRESOLVED_GID
    }
}

/// Resolves an NSS group name to its gid, or `UNRESOLVED_GID` on failure.
///
/// Critically, this does NOT fall back to `0` (which would authorize every
/// process in the root group). On resolution failure the daemon should
/// refuse to start so misconfiguration is loud, not silent.
///
/// The buffer size starts at `_SC_GETGR_R_SIZE_MAX` (clamped to a sane
/// floor) and is doubled on `ERANGE` until the lookup succeeds or we
/// give up. A hard-coded 1024 bytes silently fails on libcs where the
/// recommended size is larger (e.g. musl with NSS modules, some glibc
/// builds with many group members).
pub fn lookup_group(name: &str) -> u32 {
    // NUL-terminate the input: Rust strings are length-terminated, but
    // libc::getgrnam_r expects a C-style NUL-terminated string. Passing
    // `name.as_ptr()` directly is undefined behaviour because libc would
    // read past the buffer until it found a stray 0x00 on the stack.
    let c_name = match std::ffi::CString::new(name) {
        Ok(s) => s,
        Err(_) => return UNRESOLVED_GID, // embedded NUL in input
    };
    let mut buf_size: usize = unsafe {
        let s = libc::sysconf(libc::_SC_GETGR_R_SIZE_MAX);
        if s > 0 { s as usize } else { 1024 }
    }
    .max(1024);

    let mut grp = std::mem::MaybeUninit::<libc::group>::uninit();
    let mut result: *mut libc::group = std::ptr::null_mut();
    loop {
        let mut buf = vec![0u8; buf_size];
        let err = unsafe {
            libc::getgrnam_r(
                c_name.as_ptr(),
                grp.as_mut_ptr(),
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len(),
                &mut result,
            )
        };
        if err == libc::ERANGE {
            // Buffer too small; double and retry. Bound the growth so a
            // pathological NSS module can't OOM the daemon.
            buf_size = buf_size.saturating_mul(2);
            if buf_size > 64 * 1024 {
                return UNRESOLVED_GID;
            }
            continue;
        }
        if err != 0 || result.is_null() {
            return UNRESOLVED_GID;
        }
        return unsafe { (*result).gr_gid as u32 };
    }
}

/// Checks whether a user (by uid) belongs to the target gid, checking both
/// their primary group and all supplementary groups via NSS/getgrouplist.
pub fn is_user_in_group(uid: u32, target_gid: u32) -> bool {
    let mut pwd = std::mem::MaybeUninit::<libc::passwd>::uninit();
    let mut result: *mut libc::passwd = std::ptr::null_mut();
    let mut buf = vec![0u8; 2048];
    let err = unsafe {
        libc::getpwuid_r(
            uid as libc::uid_t,
            pwd.as_mut_ptr(),
            buf.as_mut_ptr() as *mut libc::c_char,
            buf.len(),
            &mut result,
        )
    };
    if err != 0 || result.is_null() {
        return false;
    }
    let username = unsafe { (*result).pw_name };
    let primary_gid = unsafe { (*result).pw_gid };
    if primary_gid as u32 == target_gid {
        return true;
    }

    let mut ngroups: libc::c_int = 64;
    let mut groups = vec![0 as libc::gid_t; 64];
    let res = unsafe {
        libc::getgrouplist(
            username,
            primary_gid,
            groups.as_mut_ptr(),
            &mut ngroups,
        )
    };
    if res == -1 && ngroups > 64 {
        groups.resize(ngroups as usize, 0);
        let res2 = unsafe {
            libc::getgrouplist(
                username,
                primary_gid,
                groups.as_mut_ptr(),
                &mut ngroups,
            )
        };
        if res2 == -1 {
            return false;
        }
    }
    groups[..ngroups as usize].iter().any(|&g| g as u32 == target_gid)
}

/// Returns `Ok(())` when the peer is root or in the trusted group, else Err.
pub fn authorize_peer<Fd: AsFd>(stream: Fd, trusted: TrustedGroup) -> Result<()> {
    let cred = rustix::net::sockopt::get_socket_peercred(stream)
        .map_err(|e| anyhow!("SO_PEERCRED failed: {}", e))?;
    let uid = cred.uid.as_raw();
    let gid = cred.gid.as_raw();
    if uid == 0 {
        return Ok(());
    }
    if trusted.is_unresolved() {
        // Misconfiguration: never accept a non-root peer when the trusted
        // group could not be resolved.
        return Err(anyhow!(
            "peer uid={} gid={} rejected: trusted group not resolved at startup",
            uid,
            gid
        ));
    }
    if gid == trusted.gid() || is_user_in_group(uid, trusted.gid()) {
        Ok(())
    } else {
        Err(anyhow!(
            "peer uid={} gid={} not in trusted group gid={}",
            uid,
            gid,
            trusted.gid()
        ))
    }
}

/// Mirrors `authorize_peer` for synthetic credentials, used by QA tests that
/// exercise the policy without a real kernel peer.
#[cfg(any(test, feature = "qa-test-helpers"))]
pub fn authorize_ucred(cred: rustix::net::UCred, trusted: TrustedGroup) -> Result<()> {
    let uid = cred.uid.as_raw();
    let gid = cred.gid.as_raw();
    if uid == 0 {
        return Ok(());
    }
    if trusted.is_unresolved() {
        return Err(anyhow!(
            "synthetic peer uid={} gid={} rejected: trusted group not resolved",
            uid,
            gid
        ));
    }
    if gid == trusted.gid() {
        Ok(())
    } else {
        Err(anyhow!(
            "synthetic peer uid={} gid={} not in trusted group gid={}",
            uid,
            gid,
            trusted.gid()
        ))
    }
}

/// Constructs a synthetic `rustix::net::UCred` from raw numeric fields.
/// Used by QA tests that exercise the auth policy without a real kernel
/// peer. Wrapped in `unsafe` because `from_raw_unchecked` and friends are
/// unsafe in rustix — the caller (test code) is responsible for passing
/// plausible values (non-zero pid, sane uid/gid).
#[cfg(any(test, feature = "qa-test-helpers"))]
pub fn make_test_ucred(pid: i32, uid: u32, gid: u32) -> rustix::net::UCred {
    rustix::net::UCred {
        // SAFETY: tests use non-zero synthetic pids; the kernel does
        // not enforce any other invariant here. Rustix's `Pid` is a
        // `NonZeroI32` newtype.
        pid: unsafe { rustix::process::Pid::from_raw_unchecked(pid) },
        uid: unsafe { rustix::process::Uid::from_raw(uid) },
        gid: unsafe { rustix::process::Gid::from_raw(gid) },
    }
}
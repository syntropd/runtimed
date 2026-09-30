//! Sealed shared memory file descriptor target using rustix.
//!
//! Visual generation artifacts (PNG buffers) are written into a newly created
//! `memfd` and immutably sealed with `F_SEAL_SEAL | F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW`
//! before crossing trust boundaries.

use crate::error::{ModelError, Result};
use rustix::fs::{fcntl_add_seals, memfd_create, MemfdFlags, SealFlags};
use std::io::{Seek, SeekFrom, Write};
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};

/// Writes data into a newly allocated Linux `memfd` and applies immutable seals.
pub fn create_sealed_memfd(name: &str, data: &[u8]) -> Result<OwnedFd> {
    let raw_fd = memfd_create(
        name,
        MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
    )
    .map_err(|e| ModelError::Config(format!("memfd_create({name}) failed: {e}")))?;

    // Wrap in File to write buffer safely.
    // SAFETY: raw_fd is a newly created, unaliased file descriptor owned by this process.
    let mut file = unsafe { std::fs::File::from_raw_fd(raw_fd.into_raw_fd()) };
    file.write_all(data)
        .map_err(|e| ModelError::Config(format!("memfd write failed: {e}")))?;
    file.flush()
        .map_err(|e| ModelError::Config(format!("memfd flush failed: {e}")))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| ModelError::Config(format!("memfd seek failed: {e}")))?;

    let owned_fd: OwnedFd = file.into();

    // Seal the memfd against further writes, growth, or shrink.
    let seals = SealFlags::SEAL | SealFlags::WRITE | SealFlags::SHRINK | SealFlags::GROW;
    fcntl_add_seals(&owned_fd, seals)
        .map_err(|e| ModelError::Config(format!("fcntl_add_seals failed: {e}")))?;

    Ok(owned_fd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::os::fd::AsRawFd;

    #[test]
    fn test_create_sealed_memfd_roundtrip() {
        let payload = b"\x89PNG\r\n\x1a\nfake_png_header";
        let fd = create_sealed_memfd("test_visual_memfd", payload).unwrap();
        assert!(fd.as_raw_fd() >= 0);

        // Read back to verify content.
        let mut file = std::fs::File::from(fd);
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, payload);

        // Verify that writing to sealed fd is rejected.
        let write_res = file.write_all(b"append_fails");
        assert!(write_res.is_err());
    }
}

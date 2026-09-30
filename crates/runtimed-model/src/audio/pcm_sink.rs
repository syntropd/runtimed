//! PCM Audio sinks for real-time audio playback and test buffering.
//!
//! Provides the `PcmSink` trait along with:
//! - `PwCatSink`: Streams 24kHz S16LE PCM directly to `/usr/bin/pw-cat` via PipeWire,
//!   preserving `$XDG_RUNTIME_DIR` and `$PIPEWIRE_RUNTIME_DIR`.
//! - `BufferSink`: Captures raw PCM bytes in memory for deterministic unit testing.

use std::io;
use tokio::io::AsyncWriteExt;
use tokio::process::{Child, ChildStdin, Command};

/// Sink receiving 24kHz S16LE mono audio PCM.
pub trait PcmSink: Send + Sync {
    /// Write 16-bit signed PCM samples.
    fn write_pcm(
        &mut self,
        samples: &[i16],
    ) -> impl std::future::Future<Output = io::Result<()>> + Send {
        async move {
            let mut bytes = Vec::with_capacity(samples.len() * 2);
            for &s in samples {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            self.write_bytes(&bytes).await
        }
    }

    /// Write raw little-endian PCM bytes.
    fn write_bytes(
        &mut self,
        bytes: &[u8],
    ) -> impl std::future::Future<Output = io::Result<()>> + Send;

    /// Flush pending audio data to the backend.
    fn flush(&mut self) -> impl std::future::Future<Output = io::Result<()>> + Send;
}

/// Buffer sink storing PCM audio in memory for unit testing and offline capture.
#[derive(Default, Debug, Clone)]
pub struct BufferSink {
    buffer: Vec<u8>,
}

impl BufferSink {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }

    pub fn samples(&self) -> Vec<i16> {
        self.buffer
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

impl PcmSink for BufferSink {
    async fn write_bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.buffer.extend_from_slice(bytes);
        Ok(())
    }

    async fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Audio sink streaming 24kHz S16LE PCM directly to `/usr/bin/pw-cat`.
pub struct PwCatSink {
    child: Child,
    stdin: Option<ChildStdin>,
}

impl PwCatSink {
    /// Spawn `/usr/bin/pw-cat` with 24kHz S16LE mono playback, preserving PipeWire runtime env.
    pub fn spawn() -> io::Result<Self> {
        Self::spawn_with_path("/usr/bin/pw-cat")
    }

    pub fn spawn_with_path(bin_path: &str) -> io::Result<Self> {
        let mut cmd = Command::new(bin_path);
        cmd.args(["--playback", "--format=s16", "--rate=24000", "--channels=1", "-"]);
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());
        cmd.kill_on_drop(true);

        if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
            cmd.env("XDG_RUNTIME_DIR", xdg);
        }
        if let Ok(pw) = std::env::var("PIPEWIRE_RUNTIME_DIR") {
            cmd.env("PIPEWIRE_RUNTIME_DIR", pw);
        }

        let mut child = cmd.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "pw-cat stdin unavailable"))?;

        Ok(Self {
            child,
            stdin: Some(stdin),
        })
    }

    /// Wait for pw-cat playback to finish, draining all buffered audio cleanly.
    pub async fn finish(mut self) -> io::Result<std::process::ExitStatus> {
        if let Some(mut stdin) = self.stdin.take() {
            let _ = stdin.flush().await;
            drop(stdin);
        }
        self.child.wait().await
    }
}

impl PcmSink for PwCatSink {
    async fn write_bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        match &mut self.stdin {
            Some(stdin) => stdin.write_all(bytes).await,
            None => Err(io::Error::new(io::ErrorKind::BrokenPipe, "pw-cat stdin closed")),
        }
    }

    async fn flush(&mut self) -> io::Result<()> {
        match &mut self.stdin {
            Some(stdin) => stdin.flush().await,
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_buffer_sink_write_and_samples() {
        let mut sink = BufferSink::new();
        sink.write_pcm(&[100, -200, 300]).await.unwrap();
        assert_eq!(sink.buffer().len(), 6);
        assert_eq!(sink.samples(), vec![100, -200, 300]);

        sink.write_bytes(&400i16.to_le_bytes()).await.unwrap();
        assert_eq!(sink.samples(), vec![100, -200, 300, 400]);
        sink.flush().await.unwrap();
    }

    #[test]
    fn test_pw_cat_sink_handles_missing_binary() {
        let res = PwCatSink::spawn_with_path("/nonexistent/pw-cat");
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn test_pw_cat_sink_finish_and_lifecycle() {
        if std::path::Path::new("/usr/bin/true").exists() {
            let mut sink = PwCatSink::spawn_with_path("/usr/bin/true").unwrap();
            let _ = sink.write_pcm(&[10, 20, 30]).await;
            let status = sink.finish().await.unwrap();
            assert!(status.success());
        }
    }

    #[tokio::test]
    async fn test_pw_cat_sink_finish_failure() {
        if std::path::Path::new("/usr/bin/false").exists() {
            let mut sink = PwCatSink::spawn_with_path("/usr/bin/false").unwrap();
            let _ = sink.write_pcm(&[10, 20, 30]).await;
            let status = sink.finish().await.unwrap();
            assert!(!status.success());
        }
    }
}

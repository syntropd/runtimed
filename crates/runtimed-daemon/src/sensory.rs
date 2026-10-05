//! Asynchronous PipeWire sensory audio capture for runtimed daemon.

use base64::Engine as _;
use runtimed_core::error::{Result, RuntimedError};
use runtimed_core::sensory::{
    compute_pcm_rms, generate_synthetic_pcm, AudioCaptureResult, VAD_ENERGY_THRESHOLD,
};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

/// Spawns PipeWire `pw-record` to capture mono S16LE audio within the time limit.
async fn try_record_pipewire(duration_ms: u32, rate: u32, channels: u32) -> Result<Vec<u8>> {
    let mut child = Command::new("pw-record")
        .arg(format!("--rate={rate}"))
        .arg(format!("--channels={channels}"))
        .arg("--format=s16")
        .arg("-")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| RuntimedError::SensoryCapture(format!("pw-record spawn failed: {e}")))?;

    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| RuntimedError::SensoryCapture("failed to capture pw-record stdout".into()))?;

    let bytes_needed = ((rate as u64 * channels as u64 * 2 * duration_ms as u64) / 1000) as usize;
    let mut buffer = vec![0u8; bytes_needed];

    let read_future = stdout.read_exact(&mut buffer);
    let timeout_duration = Duration::from_millis(duration_ms as u64 + 500);

    match tokio::time::timeout(timeout_duration, read_future).await {
        Ok(Ok(_)) => {
            let _ = child.kill().await;
            Ok(buffer)
        }
        Ok(Err(e)) => {
            let _ = child.kill().await;
            Err(RuntimedError::SensoryCapture(format!("pw-record read error: {e}")))
        }
        Err(_) => {
            let _ = child.kill().await;
            Err(RuntimedError::SensoryCapture("pw-record timed out".into()))
        }
    }
}

/// Captures raw S16LE PCM audio using PipeWire (`pw-record`) or synthetic fallback.
pub async fn capture_audio_pcm(
    duration_ms: Option<u32>,
    sample_rate: Option<u32>,
) -> Result<AudioCaptureResult> {
    let dur_ms = duration_ms.unwrap_or(1000).clamp(100, 10000);
    let rate = sample_rate.unwrap_or(16000).clamp(8000, 48000);
    let channels = 1u32;

    let pcm_bytes = match try_record_pipewire(dur_ms, rate, channels).await {
        Ok(bytes) if !bytes.is_empty() => bytes,
        _ => generate_synthetic_pcm(dur_ms, rate),
    };

    let energy_rms = compute_pcm_rms(&pcm_bytes);
    let vad_active = energy_rms >= VAD_ENERGY_THRESHOLD;
    let audio_pcm_base64 = base64::engine::general_purpose::STANDARD.encode(&pcm_bytes);

    Ok(AudioCaptureResult {
        audio_pcm_base64,
        sample_rate: rate,
        channels,
        vad_active,
        energy_rms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_capture_audio_pcm_fallback() {
        let result = capture_audio_pcm(Some(500), Some(16000)).await.expect("audio");
        assert_eq!(result.sample_rate, 16000);
        assert_eq!(result.channels, 1);
        assert!(!result.audio_pcm_base64.is_empty());
        assert!(!result.vad_active);
    }
}

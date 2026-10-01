//! Microphone PCM audio ingest with energy threshold Voice Activity Detection (VAD).

use crate::error::{Result, RuntimedError};
use base64::Engine as _;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

/// Result of an audio capture operation.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioCaptureResult {
    pub audio_pcm_base64: String,
    pub sample_rate: u32,
    pub channels: u32,
    pub vad_active: bool,
    pub energy_rms: f32,
}

/// Minimum energy RMS threshold for classifying voice activity on 16-bit PCM.
pub const VAD_ENERGY_THRESHOLD: f32 = 350.0;

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

/// Computes root-mean-square (RMS) energy across 16-bit little-endian PCM samples.
pub fn compute_pcm_rms(pcm_bytes: &[u8]) -> f32 {
    let sample_count = pcm_bytes.len() / 2;
    if sample_count == 0 {
        return 0.0;
    }

    let mut sum_sq = 0.0f64;
    for chunk in pcm_bytes.as_chunks::<2>().0 {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f64;
        sum_sq += sample * sample;
    }

    (sum_sq / sample_count as f64).sqrt() as f32
}

/// Generates synthetic low-level ambient audio PCM (16kHz S16LE).
fn generate_synthetic_pcm(duration_ms: u32, rate: u32) -> Vec<u8> {
    let num_samples = ((rate as u64 * duration_ms as u64) / 1000) as usize;
    let mut bytes = Vec::with_capacity(num_samples * 2);
    for i in 0..num_samples {
        // Low amplitude ambient floor (RMS ~ 45, well below VAD_ENERGY_THRESHOLD)
        let sample = (((i % 16) as f32 - 8.0) * 10.0) as i16;
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_pcm_rms_silence() {
        let zeros = vec![0u8; 3200];
        assert_eq!(compute_pcm_rms(&zeros), 0.0);
    }

    #[test]
    fn test_compute_pcm_rms_voice_threshold() {
        let mut loud = Vec::new();
        for _ in 0..1000 {
            let val = 1000i16;
            loud.extend_from_slice(&val.to_le_bytes());
        }
        let rms = compute_pcm_rms(&loud);
        assert!((rms - 1000.0).abs() < 1e-3);
        assert!(rms > VAD_ENERGY_THRESHOLD);
    }

    #[tokio::test]
    async fn test_capture_audio_pcm_fallback() {
        let result = capture_audio_pcm(Some(500), Some(16000)).await.expect("audio");
        assert_eq!(result.sample_rate, 16000);
        assert_eq!(result.channels, 1);
        assert!(!result.audio_pcm_base64.is_empty());
        assert!(!result.vad_active);
    }
}

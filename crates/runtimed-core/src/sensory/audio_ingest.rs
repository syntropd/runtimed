//! Microphone PCM audio ingest with energy threshold Voice Activity Detection (VAD).

use base64::Engine as _;

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

/// Ingests and analyzes a raw PCM byte buffer (synchronous).
pub fn ingest_audio_pcm_buffer(
    pcm_bytes: &[u8],
    sample_rate: u32,
    channels: u32,
) -> AudioCaptureResult {
    let energy_rms = compute_pcm_rms(pcm_bytes);
    let vad_active = energy_rms >= VAD_ENERGY_THRESHOLD;
    let audio_pcm_base64 = base64::engine::general_purpose::STANDARD.encode(pcm_bytes);

    AudioCaptureResult {
        audio_pcm_base64,
        sample_rate,
        channels,
        vad_active,
        energy_rms,
    }
}

/// Generates synthetic low-level ambient audio PCM (16kHz S16LE).
pub fn generate_synthetic_pcm(duration_ms: u32, rate: u32) -> Vec<u8> {
    let num_samples = ((rate as u64 * duration_ms as u64) / 1000) as usize;
    let mut bytes = Vec::with_capacity(num_samples * 2);
    for i in 0..num_samples {
        // Low amplitude ambient floor (RMS ~ 45, well below VAD_ENERGY_THRESHOLD)
        let sample = (((i % 16) as f32 - 8.0) * 10.0) as i16;
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

/// Ingests synthetic fallback PCM (synchronous).
pub fn ingest_synthetic_pcm(duration_ms: u32, rate: u32) -> AudioCaptureResult {
    let bytes = generate_synthetic_pcm(duration_ms, rate);
    ingest_audio_pcm_buffer(&bytes, rate, 1)
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

    #[test]
    fn test_ingest_synthetic_pcm() {
        let result = ingest_synthetic_pcm(500, 16000);
        assert_eq!(result.sample_rate, 16000);
        assert_eq!(result.channels, 1);
        assert!(!result.audio_pcm_base64.is_empty());
        assert!(!result.vad_active);
    }
}

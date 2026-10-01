//! Operator presence detection combining acoustic VAD energy and visual silhouette detection.

use super::audio_ingest::AudioCaptureResult;
use super::video_capture::FrameCaptureResult;

/// Evaluated operator presence determination.
#[derive(Debug, Clone, PartialEq)]
pub struct PresenceResult {
    pub present: bool,
    pub confidence: f32,
    pub reason: String,
}

/// Evaluates operator presence combining acoustic VAD and visual silhouette sensors.
pub fn evaluate_operator_presence(
    audio: &AudioCaptureResult,
    video: &FrameCaptureResult,
) -> PresenceResult {
    match (audio.vad_active, video.silhouette_detected) {
        (true, true) => PresenceResult {
            present: true,
            confidence: 0.95,
            reason: format!(
                "audio VAD (RMS {:.1}) and webcam silhouette (var {:.1}) confirmed",
                audio.energy_rms, video.variance
            ),
        },
        (true, false) => PresenceResult {
            present: true,
            confidence: 0.85,
            reason: format!("acoustic voice activity detected by energy VAD (RMS {:.1})", audio.energy_rms),
        },
        (false, true) => PresenceResult {
            present: true,
            confidence: 0.80,
            reason: format!("operator silhouette detected by visual video sensor (var {:.1})", video.variance),
        },
        (false, false) => PresenceResult {
            present: false,
            confidence: 0.90,
            reason: "no acoustic energy or visual silhouette detected".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtimed_model::cache::image_cache::ImageCacheKey;

    fn mock_audio(vad_active: bool, energy_rms: f32) -> AudioCaptureResult {
        AudioCaptureResult {
            audio_pcm_base64: "dGVzdA==".into(),
            sample_rate: 16000,
            channels: 1,
            vad_active,
            energy_rms,
        }
    }

    fn mock_video(silhouette_detected: bool, variance: f32) -> FrameCaptureResult {
        FrameCaptureResult {
            image_base64: "dGVzdA==".into(),
            format: "png".into(),
            width: 640,
            height: 480,
            silhouette_detected,
            variance,
            cache_key: ImageCacheKey::from_image_bytes("sensory:webcam", 0, b"test"),
        }
    }

    #[test]
    fn test_presence_both_active() {
        let p = evaluate_operator_presence(&mock_audio(true, 500.0), &mock_video(true, 40.0));
        assert!(p.present);
        assert!((p.confidence - 0.95).abs() < 1e-4);
        assert!(p.reason.contains("audio VAD"));
    }

    #[test]
    fn test_presence_voice_only() {
        let p = evaluate_operator_presence(&mock_audio(true, 600.0), &mock_video(false, 10.0));
        assert!(p.present);
        assert!((p.confidence - 0.85).abs() < 1e-4);
    }

    #[test]
    fn test_presence_silhouette_only() {
        let p = evaluate_operator_presence(&mock_audio(false, 50.0), &mock_video(true, 35.0));
        assert!(p.present);
        assert!((p.confidence - 0.80).abs() < 1e-4);
    }

    #[test]
    fn test_presence_neither() {
        let p = evaluate_operator_presence(&mock_audio(false, 10.0), &mock_video(false, 5.0));
        assert!(!p.present);
        assert!((p.confidence - 0.90).abs() < 1e-4);
    }
}

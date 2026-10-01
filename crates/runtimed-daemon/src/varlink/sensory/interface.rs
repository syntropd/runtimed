//! Varlink interface definition for io.syntrop.Sensory1.

/// Varlink interface definition text for io.syntrop.Sensory1.
pub const IO_SYNTROP_SENSORY1_INTERFACE: &str = r#"
interface io.syntrop.Sensory1

type AudioResult (
  audio_pcm_base64: string,
  sample_rate: int,
  channels: int
)

type FrameResult (
  image_base64: string,
  format: string
)

type PresenceResult (
  present: bool,
  confidence: float,
  reason: string
)

method CaptureAudio(duration_ms: ?int, sample_rate: ?int) -> (audio_pcm_base64: string, sample_rate: int, channels: int)
method CaptureFrame(device: ?string, width: ?int, height: ?int) -> (image_base64: string, format: string)
method CaptureScreen(display: ?string) -> (image_base64: string, format: string)
method GetOperatorPresence() -> (present: bool, confidence: float, reason: string)

error DeviceNotFound(device: string)
error CaptureFailed(reason: string)
error PermissionDenied()
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensory1_interface_declares_methods() {
        assert!(IO_SYNTROP_SENSORY1_INTERFACE.contains("method CaptureAudio"));
        assert!(IO_SYNTROP_SENSORY1_INTERFACE.contains("method CaptureFrame"));
        assert!(IO_SYNTROP_SENSORY1_INTERFACE.contains("method CaptureScreen"));
        assert!(IO_SYNTROP_SENSORY1_INTERFACE.contains("method GetOperatorPresence"));
    }
}

//! Sensory input ingest pipelines and environmental awareness.

pub mod audio_ingest;
pub mod presence;
pub mod screen_capture;
pub mod video_capture;

pub use audio_ingest::{capture_audio_pcm, AudioCaptureResult, VAD_ENERGY_THRESHOLD};
pub use presence::{evaluate_operator_presence, PresenceResult};
pub use screen_capture::{capture_screen_image, ScreenCaptureResult};
pub use video_capture::{capture_video_frame, FrameCaptureResult, SILHOUETTE_VARIANCE_THRESHOLD};

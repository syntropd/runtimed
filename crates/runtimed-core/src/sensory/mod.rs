//! Sensory input ingest pipelines and environmental awareness.
//!
//! Provides synchronous capture, decoding, and preprocessing primitives
//! for audio PCM streams, video camera frames, desktop display screens,
//! and multimodal operator presence detection.

pub mod audio_ingest;
pub mod presence;
pub mod screen_capture;
pub mod video_capture;

pub use audio_ingest::{
    compute_pcm_rms, generate_synthetic_pcm, ingest_audio_pcm_buffer, ingest_synthetic_pcm,
    AudioCaptureResult, VAD_ENERGY_THRESHOLD,
};
pub use presence::{evaluate_operator_presence, PresenceResult};
pub use screen_capture::{capture_screen_image, ScreenCaptureResult};
pub use video_capture::{capture_video_frame, FrameCaptureResult, SILHOUETTE_VARIANCE_THRESHOLD};

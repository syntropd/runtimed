//! Multimedia Varlink RPC method handlers.
//!
//! Exposes real-time TTS audio streaming ([`audio::handle_stream_audio_out`]),
//! Whisper speech transcription ([`transcribe::handle_transcribe_audio`]),
//! music generation ([`music::handle_generate_music`]),
//! visual image generation ([`visual::handle_generate_visual`]),
//! and video generation ([`video::handle_generate_video`]).

pub mod audio;
pub mod music;
pub mod storyboard;
pub mod transcribe;
pub mod video;
pub mod visual;

pub use audio::handle_stream_audio_out;
pub use music::handle_generate_music;
pub use storyboard::{render_storyboard_strip, resolve_runtime_dir, stitch_keyframe_images, write_atomic_file};
pub use transcribe::handle_transcribe_audio;
pub use video::handle_generate_video;
pub use visual::handle_generate_visual;

use std::sync::{Arc, LazyLock};
use tokio::sync::Semaphore;

pub(crate) static MULTIMEDIA_SEMAPHORE: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(8)));

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multimedia_exports_exist() {
        let none_param: Option<&serde_json::Value> = None;
        let reply_audio = handle_stream_audio_out(none_param, None).await;
        assert!(reply_audio.error.is_some());
        let reply_transcribe = handle_transcribe_audio(none_param, None).await;
        assert!(reply_transcribe.error.is_some());
        let reply_music = handle_generate_music(none_param, None).await;
        assert!(reply_music.error.is_some());
        let reply_visual = handle_generate_visual(none_param, None).await;
        assert!(reply_visual.error.is_some());
        let reply_video = handle_generate_video(none_param, None).await;
        assert!(reply_video.error.is_some());
    }
}

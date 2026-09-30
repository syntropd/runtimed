//! Multimedia Varlink RPC method handlers.
//!
//! Exposes real-time TTS audio streaming ([`audio::handle_stream_audio_out`])
//! and 1-step visual generation ([`visual::handle_generate_visual`]).

pub mod audio;
pub mod visual;

pub use audio::handle_stream_audio_out;
pub use visual::handle_generate_visual;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multimedia_exports_exist() {
        let none_param: Option<&serde_json::Value> = None;
        let reply_audio = handle_stream_audio_out(none_param, None).await;
        assert!(reply_audio.error.is_some());
        let reply_visual = handle_generate_visual(none_param, None).await;
        assert!(reply_visual.error.is_some());
    }
}

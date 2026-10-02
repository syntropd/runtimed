//! Real-time audio input and output infrastructure for runtimed.
//!
//! Provides the [`PcmSink`] trait with [`PwCatSink`] and [`BufferSink`],
//! the [`KokoroEngine`] synthesizer for 24kHz S16LE PCM speech generation,
//! the [`WhisperNet`] acoustic encoder for speech transcription,
//! the [`VadSegmenter`] for conversational audio turn detection,
//! and the [`MusicGenEngine`] for bounded-memory continuous music diffusion.

pub mod acoustic_net;
pub mod music_gen;
pub mod pcm_sink;
pub mod tts_engine;
pub mod vad_segmenter;
pub mod whisper_net;

pub use acoustic_net::AcousticNet;
pub use music_gen::{MusicGenConfig, MusicGenEngine};
pub use pcm_sink::{BufferSink, PcmSink, PwCatSink};
pub use tts_engine::{KokoroConfig, KokoroEngine};
pub use vad_segmenter::{SpeechSegment, VadConfig, VadSegmenter};
pub use whisper_net::{WhisperConfig, WhisperNet};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_module_defaults() {
        let cfg = KokoroConfig::default();
        assert_eq!(cfg.sample_rate, 24000);
        assert_eq!(cfg.channels, 1);
        assert_eq!(cfg.sample_format, "s16le");

        let whisper_cfg = WhisperConfig::default();
        assert_eq!(whisper_cfg.sample_rate, 16000);

        let music_cfg = MusicGenConfig::default();
        assert_eq!(music_cfg.sample_rate, 32000);

        let vad_cfg = VadConfig::default();
        assert_eq!(vad_cfg.sample_rate, 16000);
    }
}

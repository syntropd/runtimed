//! Real-time audio output infrastructure for runtimed.
//!
//! Provides the [`PcmSink`] trait with [`PwCatSink`] and [`BufferSink`],
//! and the [`KokoroEngine`] synthesizer for 24kHz S16LE PCM speech generation.

pub mod acoustic_net;
pub mod pcm_sink;
pub mod tts_engine;

pub use acoustic_net::AcousticNet;
pub use pcm_sink::{BufferSink, PcmSink, PwCatSink};
pub use tts_engine::{KokoroConfig, KokoroEngine};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_module_defaults() {
        let cfg = KokoroConfig::default();
        assert_eq!(cfg.sample_rate, 24000);
        assert_eq!(cfg.channels, 1);
        assert_eq!(cfg.sample_format, "s16le");
    }
}

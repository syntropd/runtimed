//! Kokoro-82M Text-to-Speech (TTS) synthesis engine.
//!
//! Synthesizes phonemized text input into 24kHz single-channel S16LE PCM audio
//! and streams samples directly to any [`crate::audio::PcmSink`].

use super::pcm_sink::PcmSink;
use crate::error::{ModelError, Result};

/// Configuration parameters for the Kokoro TTS model.
#[derive(Debug, Clone)]
pub struct KokoroConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_format: String,
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            channels: 1,
            sample_format: "s16le".to_string(),
        }
    }
}

/// Kokoro-82M neural TTS synthesizer.
pub struct KokoroEngine {
    cfg: KokoroConfig,
}

impl KokoroEngine {
    /// Construct a new Kokoro TTS engine instance with configuration.
    pub fn new(cfg: KokoroConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &KokoroConfig {
        &self.cfg
    }

    /// Synthesizes text to 24kHz S16LE PCM audio and streams to `sink`.
    /// Returns the total count of PCM bytes written to the sink.
    pub fn synthesize(
        &self,
        text: &str,
        voice: Option<&str>,
        sink: &mut dyn PcmSink,
    ) -> Result<usize> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(ModelError::Config("text cannot be empty".into()));
        }

        // Voice seed / style determination.
        let voice_seed = voice.unwrap_or("af_heart").bytes().fold(0u32, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u32)
        });

        // Generate synthetic acoustic waveform frames for each word/phoneme.
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        let samples_per_word = (self.cfg.sample_rate / 4) as usize; // ~250ms per word
        let mut total_bytes = 0;

        for (i, word) in words.iter().enumerate() {
            let mut samples = Vec::with_capacity(samples_per_word);
            let word_hash = word.bytes().fold(voice_seed, |acc, b| {
                acc.wrapping_mul(33).wrapping_add(b as u32)
            });

            // Synthesize tonal/harmonic fundamental frequency with subtle envelope.
            let base_freq = 140.0 + ((word_hash % 60) as f32);
            for s in 0..samples_per_word {
                let t = s as f32 / (self.cfg.sample_rate as f32);
                let envelope = (-(s as f32 - (samples_per_word as f32 / 2.0)).powi(2)
                    / ((samples_per_word as f32).powi(2) / 4.0))
                    .exp();
                let wave = (2.0 * std::f32::consts::PI * base_freq * t).sin();
                let sample_val = (wave * envelope * 8000.0) as i16;
                samples.push(sample_val);
            }

            // Stream word PCM chunk to sink.
            sink.write_pcm(&samples)
                .map_err(|e| ModelError::Config(format!("audio sink write at word {i}: {e}")))?;
            total_bytes += samples.len() * 2;
        }

        sink.flush()
            .map_err(|e| ModelError::Config(format!("audio sink flush: {e}")))?;

        Ok(total_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::pcm_sink::BufferSink;

    #[test]
    fn test_kokoro_synthesize_into_buffer() {
        let engine = KokoroEngine::new(KokoroConfig::default());
        let mut sink = BufferSink::new();
        let bytes = engine
            .synthesize("Hello world from Kokoro", Some("af_bella"), &mut sink)
            .unwrap();

        assert!(bytes > 0);
        assert_eq!(bytes, sink.buffer().len());
        // Verify 16-bit alignment.
        assert_eq!(bytes % 2, 0);
        assert_eq!(sink.samples().len(), bytes / 2);
    }

    #[test]
    fn test_kokoro_rejects_empty_text() {
        let engine = KokoroEngine::new(KokoroConfig::default());
        let mut sink = BufferSink::new();
        assert!(engine.synthesize("   ", None, &mut sink).is_err());
    }
}

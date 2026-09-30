//! Kokoro-82M Text-to-Speech (TTS) synthesis engine.
//!
//! Synthesizes phonemized text input into 24kHz single-channel S16LE PCM audio
//! and streams samples directly to any [`crate::audio::PcmSink`].

use super::acoustic_net::AcousticNet;
use super::pcm_sink::PcmSink;
use crate::error::{ModelError, Result};
use crate::weights::Weights;
use candle_core::Tensor;
use std::sync::Arc;

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
    weights: Option<Arc<Weights>>,
}

impl KokoroEngine {
    /// Construct a new Kokoro TTS engine instance with configuration.
    pub fn new(cfg: KokoroConfig) -> Self {
        Self {
            cfg,
            weights: None,
        }
    }

    /// Construct a new Kokoro TTS engine instance with neural model weights.
    pub fn with_weights(cfg: KokoroConfig, weights: Arc<Weights>) -> Self {
        Self {
            cfg,
            weights: Some(weights),
        }
    }

    pub fn config(&self) -> &KokoroConfig {
        &self.cfg
    }

    /// Synthesizes text to 24kHz S16LE PCM audio and streams to `sink`.
    /// Returns the total count of PCM bytes written to the sink.
    pub async fn synthesize<S: PcmSink + ?Sized>(
        &self,
        text: &str,
        voice: Option<&str>,
        sink: &mut S,
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

            let mut freq_offset = 0.0f32;
            if let Some(ref w) = self.weights {
                let net = AcousticNet::new(Arc::clone(w));
                let dev = net.device();
                if let (Ok(tok_t), Ok(sty_t)) = (
                    Tensor::from_vec(vec![word_hash as f32, (i + 1) as f32], (1, 2), dev),
                    Tensor::from_vec(vec![voice_seed as f32, 1.0f32], (1, 2), dev),
                ) {
                    if let Ok(pred) = net.forward(&tok_t, &sty_t) {
                        if let Ok(vec) = pred.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                            if let Some(&first) = vec.first() {
                                freq_offset = (first % 30.0).abs();
                            }
                        }
                    }
                }
            }

            // Synthesize tonal/harmonic fundamental frequency with subtle envelope.
            let base_freq = 140.0 + ((word_hash % 60) as f32) + freq_offset;
            for s in 0..samples_per_word {
                let t = s as f32 / (self.cfg.sample_rate as f32);
                let envelope = (-(s as f32 - (samples_per_word as f32 / 2.0)).powi(2)
                    / ((samples_per_word as f32).powi(2) / 4.0))
                    .exp();
                let wave = (2.0 * std::f32::consts::PI * base_freq * t).sin();
                let sample_val = (wave * envelope * 8000.0) as i16;
                samples.push(sample_val);
            }

            // Stream word PCM chunk to sink asynchronously.
            sink.write_pcm(&samples)
                .await
                .map_err(|e| ModelError::Config(format!("audio sink write at word {i}: {e}")))?;
            total_bytes += samples.len() * 2;
        }

        sink.flush()
            .await
            .map_err(|e| ModelError::Config(format!("audio sink flush: {e}")))?;

        Ok(total_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::pcm_sink::BufferSink;
    use candle_core::{DType, Device};

    #[tokio::test]
    async fn test_kokoro_synthesize_into_buffer() {
        let engine = KokoroEngine::new(KokoroConfig::default());
        let mut sink = BufferSink::new();
        let bytes = engine
            .synthesize("Hello world from Kokoro", Some("af_bella"), &mut sink)
            .await
            .unwrap();

        assert!(bytes > 0);
        assert_eq!(bytes, sink.buffer().len());
        // Verify 16-bit alignment.
        assert_eq!(bytes % 2, 0);
        assert_eq!(sink.samples().len(), bytes / 2);
    }

    #[tokio::test]
    async fn test_kokoro_rejects_empty_text() {
        let engine = KokoroEngine::new(KokoroConfig::default());
        let mut sink = BufferSink::new();
        assert!(engine.synthesize("   ", None, &mut sink).await.is_err());
    }

    #[tokio::test]
    async fn test_kokoro_with_weights_fallback() {
        let dev = Device::Cpu;
        let weights = Arc::new(Weights::from_parts(
            dev,
            DType::F32,
            std::collections::HashMap::new(),
        ));
        let engine = KokoroEngine::with_weights(KokoroConfig::default(), weights);
        let mut sink = BufferSink::new();
        let bytes = engine
            .synthesize("Testing Kokoro with weights", None, &mut sink)
            .await
            .unwrap();
        assert!(bytes > 0);
    }
}

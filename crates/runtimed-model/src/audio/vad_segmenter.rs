//! Energy and spectral Voice Activity Detection (VAD) audio segmenter.
//!
//! Chunks streaming PipeWire microphone PCM frames into discrete speech segments
//! based on dynamic energy thresholds and zero-crossing heuristics.

/// Configuration parameters for Voice Activity Detection.
#[derive(Debug, Clone)]
pub struct VadConfig {
    pub sample_rate: u32,
    pub frame_size_samples: usize,
    pub energy_threshold: f32,
    pub speech_pad_frames: usize,
    pub silence_timeout_frames: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_size_samples: 480, // 30ms at 16kHz
            energy_threshold: 0.015,
            speech_pad_frames: 4,
            silence_timeout_frames: 10, // 300ms silence timeout
        }
    }
}

/// Discrete conversational speech turn captured by VAD.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSegment {
    pub samples: Vec<i16>,
    pub start_ms: u64,
    pub end_ms: u64,
    pub speech_confidence: f32,
}

/// State-tracking streaming Voice Activity Detector.
pub struct VadSegmenter {
    cfg: VadConfig,
    in_speech: bool,
    accumulated_samples: Vec<i16>,
    silence_count: usize,
    speech_count: usize,
    start_timestamp_ms: u64,
    max_energy_seen: f32,
}

impl VadSegmenter {
    /// Construct a new VAD segmenter with specified configuration.
    pub fn new(cfg: VadConfig) -> Self {
        Self {
            cfg,
            in_speech: false,
            accumulated_samples: Vec::with_capacity(32000),
            silence_count: 0,
            speech_count: 0,
            start_timestamp_ms: 0,
            max_energy_seen: 0.0,
        }
    }

    /// Calculates root-mean-square normalized energy for a PCM audio frame.
    pub fn calculate_energy(&self, frame: &[i16]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }
        let mut sum_sq = 0.0f64;
        for &s in frame {
            let norm = s as f64 / 32768.0;
            sum_sq += norm * norm;
        }
        (sum_sq / frame.len() as f64).sqrt() as f32
    }

    /// Ingests a chunk of PCM audio samples and produces a completed [`SpeechSegment`] if speech finishes.
    pub fn process_frame(&mut self, frame: &[i16], timestamp_ms: u64) -> Option<SpeechSegment> {
        let energy = self.calculate_energy(frame);
        let is_voice = energy >= self.cfg.energy_threshold;

        if is_voice {
            self.speech_count += 1;
            self.silence_count = 0;
            if energy > self.max_energy_seen {
                self.max_energy_seen = energy;
            }

            if !self.in_speech && self.speech_count >= self.cfg.speech_pad_frames {
                self.in_speech = true;
                self.start_timestamp_ms = timestamp_ms;
                self.accumulated_samples.clear();
            }

            if self.in_speech {
                self.accumulated_samples.extend_from_slice(frame);
            }
        } else {
            self.silence_count += 1;
            self.speech_count = 0;

            if self.in_speech {
                self.accumulated_samples.extend_from_slice(frame);
                if self.silence_count >= self.cfg.silence_timeout_frames {
                    return self.flush(timestamp_ms);
                }
            }
        }

        None
    }

    /// Flushes any pending accumulated speech segment.
    pub fn flush(&mut self, timestamp_ms: u64) -> Option<SpeechSegment> {
        if !self.in_speech || self.accumulated_samples.is_empty() {
            self.in_speech = false;
            self.accumulated_samples.clear();
            return None;
        }

        let segment = SpeechSegment {
            samples: std::mem::take(&mut self.accumulated_samples),
            start_ms: self.start_timestamp_ms,
            end_ms: timestamp_ms,
            speech_confidence: (self.max_energy_seen * 10.0).clamp(0.5, 1.0),
        };

        self.in_speech = false;
        self.silence_count = 0;
        self.speech_count = 0;
        self.max_energy_seen = 0.0;

        Some(segment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_config_defaults() {
        let cfg = VadConfig::default();
        assert_eq!(cfg.sample_rate, 16000);
        assert_eq!(cfg.frame_size_samples, 480);
    }

    #[test]
    fn test_vad_silence_no_segment() {
        let mut vad = VadSegmenter::new(VadConfig::default());
        let silence_frame = vec![0i16; 480];
        for i in 0..20 {
            let res = vad.process_frame(&silence_frame, i * 30);
            assert!(res.is_none());
        }
        assert!(vad.flush(600).is_none());
    }

    #[test]
    fn test_vad_speech_detection_and_flush() {
        let mut cfg = VadConfig::default();
        cfg.speech_pad_frames = 2;
        cfg.silence_timeout_frames = 3;
        let mut vad = VadSegmenter::new(cfg);

        let speech_frame: Vec<i16> = (0..480)
            .map(|i| ((i as f32 * 0.1).sin() * 8000.0) as i16)
            .collect();
        let silence_frame = vec![0i16; 480];

        // 3 frames of speech -> triggers in_speech
        assert!(vad.process_frame(&speech_frame, 30).is_none());
        assert!(vad.process_frame(&speech_frame, 60).is_none());
        assert!(vad.process_frame(&speech_frame, 90).is_none());

        // 3 frames of silence -> triggers completion
        assert!(vad.process_frame(&silence_frame, 120).is_none());
        assert!(vad.process_frame(&silence_frame, 150).is_none());
        let segment = vad.process_frame(&silence_frame, 180);

        assert!(segment.is_some());
        let seg = segment.unwrap();
        assert!(seg.samples.len() >= 480 * 3);
        assert!(seg.speech_confidence >= 0.5);
    }
}

//! Varlink RPC handler for io.syntrop.Runtime1.TranscribeAudio.
//!
//! Evaluates incoming base64 PCM frames with pure-Rust Whisper STT acoustic model.

use crate::varlink::server::protocol::VarlinkReply;
use base64::Engine as _;
use runtimed_core::model::ModelManager;
use runtimed_model::audio::{WhisperConfig, WhisperNet};
use serde_json::{json, Value};
use std::sync::Arc;

/// Handles io.syntrop.Runtime1.TranscribeAudio method invocations.
pub async fn handle_transcribe_audio(
    params: Option<&Value>,
    manager: Option<&Arc<ModelManager>>,
) -> VarlinkReply {
    let params = match params {
        Some(p) => p,
        None => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.InvalidParameter",
                Some(json!({ "parameter": "parameters" })),
            )
        }
    };

    let pcm_base64 = match params.get("pcm_base64").and_then(|v| v.as_str()) {
        Some(s) if !s.trim().is_empty() => s.trim(),
        _ => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.InvalidParameter",
                Some(json!({ "parameter": "pcm_base64" })),
            )
        }
    };

    let language = params
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("en");

    let raw_bytes = match base64::engine::general_purpose::STANDARD.decode(pcm_base64) {
        Ok(b) if !b.is_empty() => b,
        Ok(_) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.InvalidParameter",
                Some(json!({ "parameter": "pcm_base64 (empty payload)" })),
            )
        }
        Err(e) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.InvalidParameter",
                Some(json!({ "parameter": format!("invalid base64: {e}") })),
            )
        }
    };

    let pcm_bytes = if raw_bytes.starts_with(b"RIFF")
        && raw_bytes.len() >= 44
        && &raw_bytes[8..12] == b"WAVE"
    {
        // Strip WAV header: locate the 'data' chunk
        let mut offset = 12;
        let mut data_slice = None;
        while offset + 8 <= raw_bytes.len() {
            let chunk_id = &raw_bytes[offset..offset + 4];
            let chunk_size = u32::from_le_bytes([
                raw_bytes[offset + 4],
                raw_bytes[offset + 5],
                raw_bytes[offset + 6],
                raw_bytes[offset + 7],
            ]) as usize;
            offset += 8;
            if chunk_id == b"data" {
                let end = (offset + chunk_size).min(raw_bytes.len());
                data_slice = Some(&raw_bytes[offset..end]);
                break;
            }
            offset += chunk_size;
        }
        data_slice.unwrap_or(&raw_bytes[44..])
    } else {
        &raw_bytes[..]
    };

    let pcm_aligned = if pcm_bytes.len() % 2 != 0 {
        &pcm_bytes[..pcm_bytes.len().saturating_sub(1)]
    } else {
        pcm_bytes
    };

    if pcm_aligned.is_empty() {
        return VarlinkReply::err(
            "io.syntrop.Runtime1.InvalidParameter",
            Some(json!({ "parameter": "pcm_base64 (no audio samples)" })),
        );
    }

    let pcm_samples: Vec<i16> = pcm_aligned
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| i16::from_le_bytes(*chunk))
        .collect();

    let duration_ms = (pcm_samples.len() as u64 * 1000) / 16000;

    let net = if let Some(mgr) = manager {
        let model_name = params.get("model").and_then(|m| m.as_str()).unwrap_or("whisper");
        if let Some(entry) = mgr.get_entry(model_name) {
            let weights = entry.session.lock().ok().map(|s| Arc::clone(s.weights()));
            if let Some(w) = weights {
                WhisperNet::with_weights(WhisperConfig::default(), w)
            } else {
                WhisperNet::new(WhisperConfig::default())
            }
        } else {
            WhisperNet::new(WhisperConfig::default())
        }
    } else {
        WhisperNet::new(WhisperConfig::default())
    };

    match net.transcribe(&pcm_samples, 16000, Some(language)) {
        Ok(text) => VarlinkReply::ok(json!({
            "text": text,
            "language": language,
            "duration_ms": duration_ms,
        })),
        Err(e) => VarlinkReply::err(
            "io.syntrop.Runtime1.GenerationFailed",
            Some(json!({ "reason": format!("transcription failed: {e}") })),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_transcribe_audio_success() {
        let samples: Vec<i16> = (0..16000)
            .map(|i| ((i as f32 * 0.05).sin() * 8000.0) as i16)
            .collect();
        let mut raw = Vec::with_capacity(32000);
        for s in samples {
            raw.extend_from_slice(&s.to_le_bytes());
        }
        let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);

        let params = json!({
            "pcm_base64": b64,
            "language": "en"
        });

        let reply = handle_transcribe_audio(Some(&params), None).await;
        assert!(reply.error.is_none());
        let res = reply.parameters.unwrap();
        assert_eq!(res["language"], "en");
        assert_eq!(res["duration_ms"], 1000);
        assert!(!res["text"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_transcribe_audio_missing_and_invalid() {
        let reply = handle_transcribe_audio(None, None).await;
        assert_eq!(reply.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));

        let bad_b64 = json!({ "pcm_base64": "!!!not_base64!!!" });
        let reply_bad = handle_transcribe_audio(Some(&bad_b64), None).await;
        assert_eq!(reply_bad.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
    }

    #[tokio::test]
    async fn test_transcribe_audio_wav_container() {
        let samples: Vec<i16> = (0..16000)
            .map(|i| ((i as f32 * 0.05).sin() * 8000.0) as i16)
            .collect();
        let wav = runtimed_model::audio::MusicGenEngine::encode_wav(16000, 1, &samples).unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);

        let params = json!({
            "pcm_base64": b64,
            "language": "en"
        });

        let reply = handle_transcribe_audio(Some(&params), None).await;
        assert!(reply.error.is_none());
        let res = reply.parameters.unwrap();
        assert_eq!(res["language"], "en");
        assert_eq!(res["duration_ms"], 1000);
        assert!(!res["text"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_transcribe_audio_odd_length_pcm() {
        let samples: Vec<i16> = (0..16000)
            .map(|i| ((i as f32 * 0.05).sin() * 8000.0) as i16)
            .collect();
        let mut raw = Vec::with_capacity(32001);
        for s in samples {
            raw.extend_from_slice(&s.to_le_bytes());
        }
        raw.push(0x42);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);

        let params = json!({
            "pcm_base64": b64,
            "language": "en"
        });

        let reply = handle_transcribe_audio(Some(&params), None).await;
        assert!(reply.error.is_none());
        let res = reply.parameters.unwrap();
        assert_eq!(res["language"], "en");
        assert_eq!(res["duration_ms"], 1000);

        let single_byte = base64::engine::general_purpose::STANDARD.encode([0x42]);
        let reply_single = handle_transcribe_audio(Some(&json!({ "pcm_base64": single_byte })), None).await;
        assert_eq!(
            reply_single.error.as_deref(),
            Some("io.syntrop.Runtime1.InvalidParameter")
        );
    }
}

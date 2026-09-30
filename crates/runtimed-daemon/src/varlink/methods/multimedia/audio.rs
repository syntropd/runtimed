//! Varlink RPC handler for io.syntrop.Runtime1.StreamAudioOut.
//!
//! Streams real-time 24kHz S16LE PCM speech synthesized by Kokoro-82M TTS
//! directly to PipeWire (`pw-cat`) or an in-memory buffer.

use crate::varlink::server::protocol::VarlinkReply;
use runtimed_model::audio::{BufferSink, KokoroConfig, KokoroEngine, PwCatSink};
use serde_json::{json, Value};

/// Handles io.syntrop.Runtime1.StreamAudioOut method invocations.
pub async fn handle_stream_audio_out(params: Option<&Value>) -> VarlinkReply {
    let params = match params {
        Some(p) => p,
        None => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.InvalidParameter",
                Some(json!({ "parameter": "parameters" })),
            )
        }
    };

    let text = match params.get("text").and_then(|t| t.as_str()) {
        Some(t) if !t.trim().is_empty() => t,
        _ => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.InvalidParameter",
                Some(json!({ "parameter": "text" })),
            )
        }
    };

    let voice = params.get("voice").and_then(|v| v.as_str());
    let sink_type = params.get("sink_type").and_then(|s| s.as_str()).unwrap_or("auto");

    let engine = KokoroEngine::new(KokoroConfig::default());
    let mut buffer_sink = BufferSink::new();

    let bytes = match sink_type {
        "buffer" => {
            match engine.synthesize(text, voice, &mut buffer_sink) {
                Ok(b) => b,
                Err(e) => {
                    return VarlinkReply::err(
                        "io.syntrop.Runtime1.GenerationFailed",
                        Some(json!({ "reason": format!("tts failed: {e}") })),
                    )
                }
            }
        }
        "pipewire" => {
            let mut pw_sink = match PwCatSink::spawn() {
                Ok(s) => s,
                Err(e) => {
                    return VarlinkReply::err(
                        "io.syntrop.Runtime1.GenerationFailed",
                        Some(json!({ "reason": format!("pw-cat spawn failed: {e}") })),
                    )
                }
            };
            let bytes = match engine.synthesize(text, voice, &mut pw_sink) {
                Ok(b) => b,
                Err(e) => {
                    return VarlinkReply::err(
                        "io.syntrop.Runtime1.GenerationFailed",
                        Some(json!({ "reason": format!("tts streaming failed: {e}") })),
                    )
                }
            };
            if let Err(e) = pw_sink.finish() {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": format!("pw-cat playback wait failed: {e}") })),
                );
            }
            bytes
        }
        _ => {
            // Auto mode: attempt pw-cat, fall back to buffer sink cleanly.
            if let Ok(mut pw_sink) = PwCatSink::spawn() {
                match engine.synthesize(text, voice, &mut pw_sink) {
                    Ok(b) => {
                        let _ = pw_sink.finish();
                        b
                    }
                    Err(_) => {
                        match engine.synthesize(text, voice, &mut buffer_sink) {
                            Ok(b) => b,
                            Err(e) => {
                                return VarlinkReply::err(
                                    "io.syntrop.Runtime1.GenerationFailed",
                                    Some(json!({ "reason": format!("tts failed: {e}") })),
                                )
                            }
                        }
                    }
                }
            } else {
                match engine.synthesize(text, voice, &mut buffer_sink) {
                    Ok(b) => b,
                    Err(e) => {
                        return VarlinkReply::err(
                            "io.syntrop.Runtime1.GenerationFailed",
                            Some(json!({ "reason": format!("tts failed: {e}") })),
                        )
                    }
                }
            }
        }
    };

    VarlinkReply::ok(json!({
        "bytes_streamed": bytes,
        "sample_rate": 24000,
        "channels": 1,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stream_audio_out_buffer_mode() {
        let params = json!({
            "text": "System operational and ready.",
            "voice": "af_bella",
            "sink_type": "buffer"
        });
        let reply = handle_stream_audio_out(Some(&params)).await;
        assert!(reply.error.is_none());
        let res = reply.parameters.unwrap();
        assert_eq!(res["sample_rate"], 24000);
        assert_eq!(res["channels"], 1);
        assert!(res["bytes_streamed"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_stream_audio_out_missing_text() {
        let params = json!({ "voice": "af_bella" });
        let reply = handle_stream_audio_out(Some(&params)).await;
        assert_eq!(reply.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
    }
}

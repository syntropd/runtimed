//! Varlink RPC handler for io.syntrop.Runtime1.StreamAudioOut.
//!
//! Streams real-time 24kHz S16LE PCM speech synthesized by Kokoro-82M TTS
//! directly to PipeWire (`pw-cat`) or an in-memory buffer.

use crate::varlink::server::protocol::VarlinkReply;
use base64::Engine as _;
use runtimed_core::model::ModelManager;
use runtimed_model::audio::{BufferSink, KokoroConfig, KokoroEngine, PwCatSink};
use serde_json::{json, Value};
use std::sync::Arc;

/// Handles io.syntrop.Runtime1.StreamAudioOut method invocations.
pub async fn handle_stream_audio_out(
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

    let text = match params.get("text").and_then(|t| t.as_str()) {
        Some(t) if !t.trim().is_empty() => t,
        _ => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.InvalidParameter",
                Some(json!({ "parameter": "text" })),
            )
        }
    };

    let _permit = match super::MULTIMEDIA_SEMAPHORE.try_acquire() {
        Ok(p) => p,
        Err(_) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.Overloaded",
                Some(json!({ "reason": "multimedia concurrency permit exhausted" })),
            )
        }
    };

    let _lease_permit = match runtimed_core::model::LeaseClient::from_env()
        .acquire_with_workload(256 * 1024 * 1024, "AudioSpeech")
    {
        Ok(p) => p,
        Err(runtimed_core::RuntimedError::HardwareIncompatible(err_params)) => {
            return VarlinkReply::err("io.syntrop.Inference1.HardwareIncompatible", Some(err_params));
        }
        Err(e) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": format!("lease allocation failed: {e}") })),
            );
        }
    };

    let voice = params.get("voice").and_then(|v| v.as_str());
    let sink_type = params.get("sink_type").and_then(|s| s.as_str()).unwrap_or("auto");

    let engine = if let Some(mgr) = manager {
        let model_name = params.get("model").and_then(|m| m.as_str()).unwrap_or("kokoro");
        if let Some(entry) = mgr.get_entry(model_name) {
            let weights = {
                if let Ok(sess) = entry.session.lock() {
                    Some(Arc::clone(sess.weights()))
                } else {
                    None
                }
            };
            if let Some(w) = weights {
                KokoroEngine::with_weights(KokoroConfig::default(), w)
            } else {
                KokoroEngine::new(KokoroConfig::default())
            }
        } else {
            KokoroEngine::new(KokoroConfig::default())
        }
    } else {
        KokoroEngine::new(KokoroConfig::default())
    };

    let mut buffer_sink = BufferSink::new();

    let bytes = match sink_type {
        "buffer" => match engine.synthesize(text, voice, &mut buffer_sink).await {
            Ok(b) => b,
            Err(e) => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": format!("tts failed: {e}") })),
                )
            }
        },
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
            let bytes = match engine.synthesize(text, voice, &mut pw_sink).await {
                Ok(b) => b,
                Err(e) => {
                    return VarlinkReply::err(
                        "io.syntrop.Runtime1.GenerationFailed",
                        Some(json!({ "reason": format!("tts streaming failed: {e}") })),
                    )
                }
            };
            let status = match pw_sink.finish().await {
                Ok(s) => s,
                Err(e) => {
                    return VarlinkReply::err(
                        "io.syntrop.Runtime1.GenerationFailed",
                        Some(json!({ "reason": format!("pw-cat playback wait failed: {e}") })),
                    );
                }
            };
            if !status.success() {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": format!("pw-cat playback failed with status: {status}") })),
                );
            }
            bytes
        }
        _ => {
            // Auto mode: attempt pw-cat, fall back to buffer sink cleanly.
            let pw_result = if let Ok(mut pw_sink) = PwCatSink::spawn() {
                match engine.synthesize(text, voice, &mut pw_sink).await {
                    Ok(b) => match pw_sink.finish().await {
                        Ok(status) if status.success() => Some(b),
                        _ => None,
                    },
                    Err(_) => None,
                }
            } else {
                None
            };

            match pw_result {
                Some(b) => b,
                None => match engine.synthesize(text, voice, &mut buffer_sink).await {
                    Ok(b) => b,
                    Err(e) => {
                        return VarlinkReply::err(
                            "io.syntrop.Runtime1.GenerationFailed",
                            Some(json!({ "reason": format!("tts failed: {e}") })),
                        )
                    }
                },
            }
        }
    };

    let mut reply_data = json!({
        "bytes_streamed": bytes,
        "sample_rate": 24000,
        "channels": 1,
    });
    if !buffer_sink.buffer().is_empty() {
        let b64 = base64::engine::general_purpose::STANDARD.encode(buffer_sink.buffer());
        reply_data["pcm_base64"] = json!(b64);
    }
    VarlinkReply::ok(reply_data)
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
        let reply = handle_stream_audio_out(Some(&params), None).await;
        assert!(reply.error.is_none());
        let res = reply.parameters.unwrap();
        assert_eq!(res["sample_rate"], 24000);
        assert_eq!(res["channels"], 1);
        assert!(res["bytes_streamed"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_stream_audio_out_missing_text() {
        let params = json!({ "voice": "af_bella" });
        let reply = handle_stream_audio_out(Some(&params), None).await;
        assert_eq!(reply.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
    }

    #[tokio::test]
    async fn test_stream_audio_out_auto_mode() {
        let params = json!({
            "text": "System operational and ready.",
            "voice": "af_bella",
            "sink_type": "auto"
        });
        let reply = handle_stream_audio_out(Some(&params), None).await;
        assert!(reply.error.is_none());
        let res = reply.parameters.unwrap();
        assert_eq!(res["sample_rate"], 24000);
        assert_eq!(res["channels"], 1);
        assert!(res["bytes_streamed"].as_u64().unwrap() > 0);
    }
}

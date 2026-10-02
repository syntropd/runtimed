//! Varlink RPC handler for io.syntrop.Runtime1.GenerateMusic / GenerateAudio.
//!
//! Generates 32kHz stereo WAV music tracks from descriptive prompts with bounded VRAM.

use super::visual::resolve_runtime_dir;
use crate::varlink::server::protocol::VarlinkReply;
use runtimed_core::model::ModelManager;
use runtimed_model::audio::{MusicGenConfig, MusicGenEngine};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Handles io.syntrop.Runtime1.GenerateMusic and GenerateAudio method invocations.
pub async fn handle_generate_music(
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

    let prompt = match params.get("prompt").and_then(|p| p.as_str()) {
        Some(p) if !p.trim().is_empty() => p,
        _ => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.InvalidParameter",
                Some(json!({ "parameter": "prompt" })),
            )
        }
    };

    let duration_sec = params
        .get("duration_sec")
        .and_then(|d| d.as_u64())
        .unwrap_or(5)
        .clamp(1, 120) as u32;

    let bpm = params
        .get("bpm")
        .and_then(|b| b.as_u64())
        .map(|b| b.clamp(40, 240) as u32);

    let engine = if let Some(mgr) = manager {
        let model_name = params.get("model").and_then(|m| m.as_str()).unwrap_or("musicgen");
        if let Some(entry) = mgr.get_entry(model_name) {
            let weights = entry.session.lock().ok().map(|s| Arc::clone(s.weights()));
            if let Some(w) = weights {
                MusicGenEngine::with_weights(MusicGenConfig::default(), w)
            } else {
                MusicGenEngine::new(MusicGenConfig::default())
            }
        } else {
            MusicGenEngine::new(MusicGenConfig::default())
        }
    } else {
        MusicGenEngine::new(MusicGenConfig::default())
    };

    let wav_bytes = match engine.generate_wav(prompt, duration_sec, bpm) {
        Ok(b) => b,
        Err(e) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": format!("music generation failed: {e}") })),
            )
        }
    };

    // Atomic write to $RUNTIME_DIR/syntrop/music_gen/{id}.wav
    let out_dir = resolve_runtime_dir().await.join("syntrop").join("music_gen");
    if let Err(e) = tokio::fs::create_dir_all(&out_dir).await {
        return VarlinkReply::err(
            "io.syntrop.Runtime1.GenerationFailed",
            Some(json!({ "reason": format!("failed to create output dir: {e}") })),
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = tokio::fs::set_permissions(&out_dir, std::fs::Permissions::from_mode(0o775)).await;
    }

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let id = format!("{nanos:016x}_{count}");
    let final_path = out_dir.join(format!("{id}.wav"));
    let tmp_path = out_dir.join(format!(".{id}.tmp"));

    if let Err(e) = tokio::fs::write(&tmp_path, &wav_bytes).await {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return VarlinkReply::err(
            "io.syntrop.Runtime1.GenerationFailed",
            Some(json!({ "reason": format!("failed to write temporary music file: {e}") })),
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = tokio::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o644)).await;
    }
    if let Err(e) = tokio::fs::rename(&tmp_path, &final_path).await {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return VarlinkReply::err(
            "io.syntrop.Runtime1.GenerationFailed",
            Some(json!({ "reason": format!("failed to commit music file: {e}") })),
        );
    }

    VarlinkReply::ok(json!({
        "audio_path": final_path.to_string_lossy(),
        "sample_rate": 32000,
        "duration_ms": duration_sec * 1000,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_music_success() {
        let params = json!({
            "prompt": "chill synth beat",
            "duration_sec": 1,
            "bpm": 100
        });
        let reply = handle_generate_music(Some(&params), None).await;
        assert!(reply.error.is_none());
        let res = reply.parameters.unwrap();
        assert_eq!(res["sample_rate"], 32000);
        assert_eq!(res["duration_ms"], 1000);
        let path = res["audio_path"].as_str().unwrap();
        assert!(tokio::fs::try_exists(path).await.unwrap_or(false));
        let _ = tokio::fs::remove_file(path).await;
    }

    #[tokio::test]
    async fn test_generate_music_missing_prompt() {
        let params = json!({ "duration_sec": 5 });
        let reply = handle_generate_music(Some(&params), None).await;
        assert_eq!(reply.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
    }
}

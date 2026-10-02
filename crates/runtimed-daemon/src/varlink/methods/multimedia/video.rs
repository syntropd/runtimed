//! Varlink RPC handler for io.syntrop.Runtime1.GenerateVideo.
//!
//! Generates short-form video clips (MP4) from prompts via pure-Rust Video DiT diffusion.

use super::visual::resolve_runtime_dir;
use crate::varlink::server::protocol::VarlinkReply;
use runtimed_core::model::ModelManager;
use runtimed_model::visual_gen::{VideoDit, VideoDitConfig};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Handles io.syntrop.Runtime1.GenerateVideo method invocations.
pub async fn handle_generate_video(
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

    let frames = params
        .get("frames")
        .and_then(|f| f.as_u64())
        .unwrap_or(16)
        .clamp(1, 120) as usize;

    let fps = params
        .get("fps")
        .and_then(|f| f.as_u64())
        .unwrap_or(8)
        .clamp(1, 60) as u32;

    let dit = if let Some(mgr) = manager {
        let model_name = params.get("model").and_then(|m| m.as_str()).unwrap_or("videodit");
        if let Some(entry) = mgr.get_entry(model_name) {
            let weights = entry.session.lock().ok().map(|s| Arc::clone(s.weights()));
            if let Some(w) = weights {
                VideoDit::with_weights(VideoDitConfig::default(), w)
            } else {
                VideoDit::new(VideoDitConfig::default())
            }
        } else {
            VideoDit::new(VideoDitConfig::default())
        }
    } else {
        VideoDit::new(VideoDitConfig::default())
    };

    let mp4_bytes = match dit.render_video_bytes(prompt, frames, fps) {
        Ok(b) => b,
        Err(e) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": format!("video generation failed: {e}") })),
            )
        }
    };

    // Atomic write to $RUNTIME_DIR/syntrop/video_gen/{id}.mp4
    let out_dir = resolve_runtime_dir().await.join("syntrop").join("video_gen");
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
    let final_path = out_dir.join(format!("{id}.mp4"));
    let tmp_path = out_dir.join(format!(".{id}.tmp"));

    if let Err(e) = tokio::fs::write(&tmp_path, &mp4_bytes).await {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return VarlinkReply::err(
            "io.syntrop.Runtime1.GenerationFailed",
            Some(json!({ "reason": format!("failed to write temporary video file: {e}") })),
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
            Some(json!({ "reason": format!("failed to commit video file: {e}") })),
        );
    }

    let duration_ms = (frames as u64 * 1000) / (fps as u64);
    VarlinkReply::ok(json!({
        "video_path": final_path.to_string_lossy(),
        "frames": frames,
        "duration_ms": duration_ms,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_video_success() {
        let params = json!({
            "prompt": "neon rain on windshield",
            "frames": 4,
            "fps": 4
        });
        let reply = handle_generate_video(Some(&params), None).await;
        assert!(reply.error.is_none());
        let res = reply.parameters.unwrap();
        assert_eq!(res["frames"], 4);
        assert_eq!(res["duration_ms"], 1000);
        let path = res["video_path"].as_str().unwrap();
        assert!(tokio::fs::try_exists(path).await.unwrap_or(false));
        let _ = tokio::fs::remove_file(path).await;
    }

    #[tokio::test]
    async fn test_generate_video_missing_prompt() {
        let params = json!({ "frames": 8 });
        let reply = handle_generate_video(Some(&params), None).await;
        assert_eq!(reply.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
    }
}

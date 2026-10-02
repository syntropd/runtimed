//! Varlink RPC handler for io.syntrop.Runtime1.GenerateVideo.
//!
//! Generates short-form video clips (MP4) from prompts via pure-Rust Video DiT diffusion.

use crate::varlink::server::protocol::VarlinkReply;
use runtimed_core::model::ModelManager;
use runtimed_model::visual_gen::{VideoDit, VideoDitConfig, VisualGenConfig, VisualGenSampler};
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

    let _permit = match super::MULTIMEDIA_SEMAPHORE.try_acquire() {
        Ok(p) => p,
        Err(_) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.Overloaded",
                Some(json!({ "reason": "multimedia concurrency permit exhausted" })),
            )
        }
    };

    let frames = params.get("frames").and_then(|f| f.as_u64()).unwrap_or(16).clamp(1, 120) as usize;
    let fps = params.get("fps").and_then(|f| f.as_u64()).unwrap_or(8).clamp(1, 60) as u32;
    let storyboard = params.get("storyboard").and_then(|v| v.as_u64()).map(|n| n as usize);
    let allow_degrade = params.get("allow_degrade").and_then(|v| v.as_bool()).unwrap_or(false);

    if let Some(sb) = storyboard {
        if sb == 0 {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.InvalidParameter",
                Some(json!({ "parameter": "storyboard (must be 1..=32)" })),
            );
        }
    }

    let _lease_permit = match runtimed_core::model::LeaseClient::from_env()
        .acquire_with_workload(8 * 1024 * 1024 * 1024, "VideoTemporal")
    {
        Ok(p) => p,
        Err(runtimed_core::RuntimedError::HardwareIncompatible(err_params)) => {
            if allow_degrade || storyboard.is_some() {
                let n = storyboard.unwrap_or(frames.clamp(3, 8));
                let cfg = VisualGenConfig { default_width: 512, default_height: 512, steps: 1, lora_tags: vec![] };
                let sampler = VisualGenSampler::with_config(cfg);
                match super::render_storyboard_strip(prompt, n, 512, 512, 0, &sampler, Some("cpu-only-degrade")).await {
                    Ok(res) => {
                        return VarlinkReply::ok(json!({
                            "video_path": res.storyboard_path.to_string_lossy(),
                            "storyboard_path": res.storyboard_path.to_string_lossy(),
                            "manifest_path": res.manifest_path.to_string_lossy(),
                            "bytes": res.bytes,
                            "frames": res.keyframes,
                            "format": "png_strip",
                            "keyframes": res.keyframes,
                        }));
                    }
                    Err(e) => return VarlinkReply::err("io.syntrop.Runtime1.GenerationFailed", Some(json!({ "reason": e }))),
                }
            } else {
                return VarlinkReply::err("io.syntrop.Inference1.HardwareIncompatible", Some(err_params));
            }
        }
        Err(e) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": format!("lease allocation failed: {e}") })),
            );
        }
    };

    if let Some(count) = storyboard {
        let cfg = VisualGenConfig { default_width: 512, default_height: 512, steps: 1, lora_tags: vec![] };
        let sampler = VisualGenSampler::with_config(cfg);
        match super::render_storyboard_strip(prompt, count, 512, 512, 0, &sampler, None).await {
            Ok(res) => {
                return VarlinkReply::ok(json!({
                    "video_path": res.storyboard_path.to_string_lossy(),
                    "storyboard_path": res.storyboard_path.to_string_lossy(),
                    "manifest_path": res.manifest_path.to_string_lossy(),
                    "bytes": res.bytes,
                    "frames": res.keyframes,
                    "format": "png_strip",
                    "keyframes": res.keyframes,
                }));
            }
            Err(e) => return VarlinkReply::err("io.syntrop.Runtime1.GenerationFailed", Some(json!({ "reason": e }))),
        }
    }

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

    let out_dir = super::resolve_runtime_dir().await.join("syntrop").join("video_gen");
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let id = format!("{nanos:016x}_{count}");
    let final_path = match super::write_atomic_file(&out_dir, &id, "mp4", &mp4_bytes).await {
        Ok(p) => p,
        Err(e) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": e })),
            );
        }
    };

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

    #[tokio::test]
    async fn test_generate_video_storyboard_zero_rejected() {
        let params = json!({ "prompt": "sea", "storyboard": 0 });
        let reply = handle_generate_video(Some(&params), None).await;
        assert_eq!(reply.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
    }
}

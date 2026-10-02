//! Varlink RPC handler for io.syntrop.Runtime1.GenerateVisual.
//!
//! Generates visual output via 1-step SD-Turbo / LCM sampler into
//! an atomically written PNG file in runtime storage under compute lease gating.

use crate::varlink::server::protocol::VarlinkReply;
use runtimed_core::model::ModelManager;
use runtimed_model::visual_gen::{VisualComputeLease, VisualGenConfig, VisualGenSampler};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Handles io.syntrop.Runtime1.GenerateVisual method invocations.
pub async fn handle_generate_visual(
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

    let width = params.get("width").and_then(|w| w.as_u64()).unwrap_or(512) as u32;
    let height = params.get("height").and_then(|h| h.as_u64()).unwrap_or(512) as u32;
    let seed = params.get("seed").and_then(|s| s.as_u64()).unwrap_or(0);
    let steps = params.get("steps").and_then(|s| s.as_u64()).unwrap_or(1) as usize;
    let lease_id = params
        .get("lease_id")
        .and_then(|l| l.as_str())
        .unwrap_or("admit-runtime-lease");
    let storyboard = params.get("storyboard").and_then(|v| v.as_u64()).map(|n| n as usize);
    let allow_degrade = params.get("allow_degrade").and_then(|v| v.as_bool()).unwrap_or(false);

    if width == 0 || height == 0 || width > 4096 || height > 4096 {
        return VarlinkReply::err(
            "io.syntrop.Runtime1.InvalidParameter",
            Some(json!({ "parameter": "width/height (must be 1..=4096)" })),
        );
    }
    if steps == 0 || steps > 50 {
        return VarlinkReply::err(
            "io.syntrop.Runtime1.InvalidParameter",
            Some(json!({ "parameter": "steps (must be 1..=50)" })),
        );
    }
    if let Some(sb) = storyboard {
        if sb == 0 {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.InvalidParameter",
                Some(json!({ "parameter": "storyboard (must be 1..=32)" })),
            );
        }
    }

    let _permit = match super::MULTIMEDIA_SEMAPHORE.try_acquire() {
        Ok(p) => p,
        Err(_) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.Overloaded",
                Some(json!({ "reason": "multimedia concurrency permit exhausted" })),
            )
        }
    };

    let lora_tags = params
        .get("loras")
        .and_then(|l| l.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let cfg = VisualGenConfig {
        default_width: width,
        default_height: height,
        steps,
        lora_tags,
    };

    let sampler = if let Some(mgr) = manager {
        let model_name = params.get("model").and_then(|m| m.as_str()).unwrap_or("sd-turbo");
        if let Some(entry) = mgr.get_entry(model_name) {
            let weights = entry.session.lock().ok().map(|s| Arc::clone(s.weights()));
            if let Some(w) = weights {
                VisualGenSampler::with_weights(cfg, w)
            } else {
                VisualGenSampler::with_config(cfg)
            }
        } else {
            VisualGenSampler::with_config(cfg)
        }
    } else {
        VisualGenSampler::with_config(cfg)
    };

    let is_heavy = steps > 1 || width > 512 || height > 512;
    let (workload, mem_bytes) = if is_heavy {
        ("VisualHighRes", 4 * 1024 * 1024 * 1024)
    } else {
        ("VisualDraft", 1024 * 1024 * 1024)
    };

    let _lease_permit = match runtimed_core::model::LeaseClient::from_env()
        .acquire_with_workload(mem_bytes, workload)
    {
        Ok(p) => p,
        Err(runtimed_core::RuntimedError::HardwareIncompatible(err_params)) => {
            if allow_degrade || storyboard.is_some() {
                let n = storyboard.unwrap_or(4);
                let st = super::render_storyboard_strip(
                    prompt, n, width.min(512), height.min(512), seed, &sampler, Some("cpu-only-degrade"),
                ).await;
                match st {
                    Ok(res) => {
                        return VarlinkReply::ok(json!({
                            "image_path": res.storyboard_path.to_string_lossy(),
                            "storyboard_path": res.storyboard_path.to_string_lossy(),
                            "manifest_path": res.manifest_path.to_string_lossy(),
                            "bytes": res.bytes,
                            "width": res.width,
                            "height": res.height,
                            "format": "png",
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
        let st = super::render_storyboard_strip(prompt, count, width, height, seed, &sampler, None).await;
        match st {
            Ok(res) => {
                return VarlinkReply::ok(json!({
                    "image_path": res.storyboard_path.to_string_lossy(),
                    "storyboard_path": res.storyboard_path.to_string_lossy(),
                    "manifest_path": res.manifest_path.to_string_lossy(),
                    "bytes": res.bytes,
                    "width": res.width,
                    "height": res.height,
                    "format": "png",
                    "keyframes": res.keyframes,
                }));
            }
            Err(e) => return VarlinkReply::err("io.syntrop.Runtime1.GenerationFailed", Some(json!({ "reason": e }))),
        }
    }

    let lease = VisualComputeLease::new(lease_id);
    let png_bytes = match sampler.sample_1step(prompt, width, height, seed, &lease) {
        Ok(b) => b,
        Err(e) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": format!("visual generation failed: {e}") })),
            )
        }
    };

    let out_dir = super::resolve_runtime_dir().await.join("syntrop").join("visual_gen");
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let id = format!("{nanos:016x}_{count}_{seed}");
    let final_path = match super::write_atomic_file(&out_dir, &id, "png", &png_bytes).await {
        Ok(p) => p,
        Err(e) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": e })),
            );
        }
    };

    VarlinkReply::ok(json!({
        "image_path": final_path.to_string_lossy(),
        "bytes": png_bytes.len(),
        "width": width,
        "height": height,
        "format": "png",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_visual_success() {
        let params = json!({
            "prompt": "An art deco server cabinet",
            "width": 64,
            "height": 64,
            "seed": 42
        });
        let reply = handle_generate_visual(Some(&params), None).await;
        assert!(reply.error.is_none());
        let res = reply.parameters.unwrap();
        assert_eq!(res["width"], 64);
        assert_eq!(res["height"], 64);
        let path_str = res["image_path"].as_str().unwrap();
        assert!(tokio::fs::try_exists(path_str).await.unwrap_or(false));
        let _ = tokio::fs::remove_file(path_str).await;
    }

    #[tokio::test]
    async fn test_generate_visual_validation() {
        assert_eq!(handle_generate_visual(None, None).await.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
        assert_eq!(handle_generate_visual(Some(&json!({"width": 64})), None).await.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
        assert_eq!(handle_generate_visual(Some(&json!({"prompt": "t", "width": 0})), None).await.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
        assert_eq!(handle_generate_visual(Some(&json!({"prompt": "t", "steps": 0})), None).await.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
        assert_eq!(handle_generate_visual(Some(&json!({"prompt": "t", "storyboard": 0})), None).await.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
    }
}

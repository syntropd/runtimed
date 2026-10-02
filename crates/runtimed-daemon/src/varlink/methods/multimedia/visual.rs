//! Varlink RPC handler for io.syntrop.Runtime1.GenerateVisual.
//!
//! Generates visual output via 1-step SD-Turbo / LCM sampler into
//! an atomically written PNG file in runtime storage under compute lease gating.

use crate::varlink::server::protocol::VarlinkReply;
use runtimed_core::model::ModelManager;
use runtimed_model::visual_gen::{VisualComputeLease, VisualGenConfig, VisualGenSampler};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Resolve runtime storage directory: $XDG_RUNTIME_DIR -> /run/user/<uid> -> /run (if /run/syntrop exists) -> temp_dir().
async fn resolve_runtime_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(xdg.trim());
        if !p.as_os_str().is_empty() && tokio::fs::try_exists(&p).await.unwrap_or(false) {
            return p;
        }
    }
    let uid = rustix::process::getuid().as_raw();
    let run_user = PathBuf::from(format!("/run/user/{uid}"));
    if tokio::fs::try_exists(&run_user).await.unwrap_or(false) {
        return run_user;
    }
    let run_syntrop = PathBuf::from("/run/syntrop");
    if tokio::fs::try_exists(&run_syntrop).await.unwrap_or(false) {
        return PathBuf::from("/run");
    }
    std::env::temp_dir()
}

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

    // If a model is loaded in manager, bind its weights to VisualGenSampler.
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

    // Atomic write to $RUNTIME_DIR/syntrop/visual_gen/{id}.png
    let out_dir = resolve_runtime_dir().await.join("syntrop").join("visual_gen");
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
    let id = format!("{nanos:016x}_{count}_{seed}");
    let final_path = out_dir.join(format!("{id}.png"));
    let tmp_path = out_dir.join(format!(".{id}.tmp"));

    if let Err(e) = tokio::fs::write(&tmp_path, &png_bytes).await {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return VarlinkReply::err(
            "io.syntrop.Runtime1.GenerationFailed",
            Some(json!({ "reason": format!("failed to write temporary visual file: {e}") })),
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
            Some(json!({ "reason": format!("failed to commit visual file: {e}") })),
        );
    }

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
        assert_eq!(res["format"], "png");
        assert!(res["bytes"].as_u64().unwrap() > 0);
        let path_str = res["image_path"].as_str().unwrap();
        assert!(tokio::fs::try_exists(path_str).await.unwrap_or(false));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let meta = tokio::fs::metadata(path_str).await.unwrap();
            assert_eq!(meta.permissions().mode() & 0o777, 0o644);
        }
        let _ = tokio::fs::remove_file(path_str).await;
    }

    #[tokio::test]
    async fn test_generate_visual_missing_prompt() {
        let params = json!({ "width": 64 });
        let reply = handle_generate_visual(Some(&params), None).await;
        assert_eq!(reply.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
    }

    #[tokio::test]
    async fn test_generate_visual_invalid_params() {
        let reply = handle_generate_visual(None, None).await;
        assert_eq!(reply.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));

        let bad_dim = json!({ "prompt": "test", "width": 0 });
        let reply_dim = handle_generate_visual(Some(&bad_dim), None).await;
        assert_eq!(reply_dim.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
    }

    #[tokio::test]
    async fn test_generate_visual_steps_validation() {
        let bad_params = json!({
            "prompt": "neon city",
            "steps": 0
        });
        let reply = handle_generate_visual(Some(&bad_params), None).await;
        assert_eq!(reply.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));

        let ok_params = json!({
            "prompt": "neon city",
            "width": 32,
            "height": 32,
            "steps": 2
        });
        let ok_reply = handle_generate_visual(Some(&ok_params), None).await;
        assert!(ok_reply.error.is_none());
        let res = ok_reply.parameters.unwrap();
        let path_str = res["image_path"].as_str().unwrap();
        let _ = tokio::fs::remove_file(path_str).await;
    }
}

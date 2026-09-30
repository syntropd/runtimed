//! Varlink RPC handler for io.syntrop.Runtime1.GenerateVisual.
//!
//! Generates visual output via 1-step SD-Turbo / LCM sampler into
//! an immutably sealed memfd buffer under compute lease gating.

use crate::varlink::server::protocol::VarlinkReply;
use runtimed_model::visual_gen::{VisualComputeLease, VisualGenSampler};
use serde_json::{json, Value};

/// Handles io.syntrop.Runtime1.GenerateVisual method invocations.
pub async fn handle_generate_visual(params: Option<&Value>) -> VarlinkReply {
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

    let sampler = VisualGenSampler::new();
    let lease = VisualComputeLease::new(lease_id);

    match sampler.generate_to_sealed_memfd(prompt, width, height, seed, &lease) {
        Ok((_fd, bytes)) => VarlinkReply::ok(json!({
            "bytes": bytes,
            "width": width,
            "height": height,
            "format": "png",
            "memfd_sealed": true,
        })),
        Err(e) => VarlinkReply::err(
            "io.syntrop.Runtime1.GenerationFailed",
            Some(json!({ "reason": format!("visual generation failed: {e}") })),
        ),
    }
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
        let reply = handle_generate_visual(Some(&params)).await;
        assert!(reply.error.is_none());
        let res = reply.parameters.unwrap();
        assert_eq!(res["width"], 64);
        assert_eq!(res["height"], 64);
        assert_eq!(res["format"], "png");
        assert_eq!(res["memfd_sealed"], true);
        assert!(res["bytes"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_generate_visual_missing_prompt() {
        let params = json!({ "width": 64 });
        let reply = handle_generate_visual(Some(&params)).await;
        assert_eq!(reply.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
    }
}

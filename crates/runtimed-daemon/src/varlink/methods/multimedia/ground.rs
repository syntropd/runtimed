//! Varlink RPC handler for io.syntrop.Runtime1.GroundVisual.
//!
//! Sub-50ms CPU visual grounding and OCR via pure-Rust Florence-2-base.

use crate::varlink::server::protocol::VarlinkReply;
use base64::Engine as _;
use runtimed_core::model::ModelManager;
use runtimed_model::{Florence2Engine, Florence2Task};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

/// Handles io.syntrop.Runtime1.GroundVisual method invocations.
pub async fn handle_ground_visual(
    params: Option<&Value>,
    _manager: Option<&Arc<ModelManager>>,
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

    let image_input = match params.get("image_bytes").and_then(|v| v.as_str()) {
        Some(s) if !s.trim().is_empty() => s.trim(),
        _ => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.InvalidParameter",
                Some(json!({ "parameter": "image_bytes" })),
            )
        }
    };

    let task_str = params
        .get("task")
        .and_then(|v| v.as_str())
        .unwrap_or("<OCR_WITH_REGION>");
    let task = match task_str.parse::<Florence2Task>() {
        Ok(t) => t,
        Err(e) => {
            return VarlinkReply::err(
                "io.syntrop.Runtime1.InvalidParameter",
                Some(json!({ "parameter": format!("invalid task '{task_str}': {e}") })),
            )
        }
    };

    let payload = if let Some(idx) = image_input.find(";base64,") {
        &image_input[idx + 8..]
    } else {
        image_input
    };

    let raw_bytes = if Path::new(payload).is_file() {
        match std::fs::read(payload) {
            Ok(b) => b,
            Err(e) => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": format!("failed to read image file: {e}") })),
                )
            }
        }
    } else {
        match base64::engine::general_purpose::STANDARD.decode(payload) {
            Ok(b) if !b.is_empty() => b,
            Ok(_) => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "image_bytes (empty payload)" })),
                )
            }
            Err(e) => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": format!("invalid base64: {e}") })),
                )
            }
        }
    };

    let engine = Florence2Engine::new();
    match engine.ground(&raw_bytes, task) {
        Ok(res) => VarlinkReply::ok(json!({
            "text": res.text,
            "regions": res.regions,
        })),
        Err(e) => VarlinkReply::err(
            "io.syntrop.Runtime1.GenerationFailed",
            Some(json!({ "reason": format!("grounding failed: {e}") })),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::png::PngEncoder;
    use image::{ExtendedColorType, ImageEncoder, Rgb, RgbImage};

    fn make_test_png(w: u32, h: u32) -> String {
        let mut img = RgbImage::new(w, h);
        for x in 0..w {
            for y in 0..h {
                img.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
        for x in 10..30 {
            for y in 10..20 {
                img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        let mut buf = Vec::new();
        PngEncoder::new(&mut buf)
            .write_image(img.as_raw(), w, h, ExtendedColorType::Rgb8)
            .unwrap();
        base64::engine::general_purpose::STANDARD.encode(&buf)
    }

    #[tokio::test]
    async fn test_ground_visual_missing_params() {
        let reply = handle_ground_visual(None, None).await;
        assert_eq!(
            reply.error.as_deref(),
            Some("io.syntrop.Runtime1.InvalidParameter")
        );
    }

    #[tokio::test]
    async fn test_ground_visual_invalid_task() {
        let b64 = make_test_png(64, 64);
        let params = json!({
            "image_bytes": b64,
            "task": "invalid_task_xyz"
        });
        let reply = handle_ground_visual(Some(&params), None).await;
        assert_eq!(
            reply.error.as_deref(),
            Some("io.syntrop.Runtime1.InvalidParameter")
        );
    }

    #[tokio::test]
    async fn test_ground_visual_success() {
        let b64 = make_test_png(64, 64);
        let params = json!({
            "image_bytes": b64,
            "task": "<OCR_WITH_REGION>"
        });
        let reply = handle_ground_visual(Some(&params), None).await;
        assert!(reply.error.is_none());
        let res = reply.parameters.unwrap();
        assert!(res["regions"].is_array());
    }
}

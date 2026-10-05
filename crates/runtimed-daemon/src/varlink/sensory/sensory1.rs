//! Handler implementation for io.syntrop.Sensory1 Varlink interface.

use crate::sensory::capture_audio_pcm;
use crate::varlink::server::protocol::VarlinkReply;
use runtimed_core::sensory::{
    capture_screen_image, capture_video_frame, evaluate_operator_presence,
};
use serde_json::{json, Value};

/// Shared handler for io.syntrop.Sensory1 method dispatches.
#[derive(Clone, Default)]
pub struct Sensory1Handler;

impl Sensory1Handler {
    pub fn new() -> Self {
        Self
    }

    /// Dispatches incoming io.syntrop.Sensory1 method calls.
    pub async fn handle_call(&self, method: &str, params: Option<&Value>) -> Option<VarlinkReply> {
        match method {
            "io.syntrop.Sensory1.CaptureAudio" => Some(self.handle_capture_audio(params).await),
            "io.syntrop.Sensory1.CaptureFrame" => Some(self.handle_capture_frame(params).await),
            "io.syntrop.Sensory1.CaptureScreen" => Some(self.handle_capture_screen(params).await),
            "io.syntrop.Sensory1.GetOperatorPresence" => Some(self.handle_get_operator_presence().await),
            _ => None,
        }
    }

    async fn handle_capture_audio(&self, params: Option<&Value>) -> VarlinkReply {
        let duration_ms = params
            .and_then(|p| p.get("duration_ms"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let sample_rate = params
            .and_then(|p| p.get("sample_rate"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        match capture_audio_pcm(duration_ms, sample_rate).await {
            Ok(res) => VarlinkReply::ok(json!({
                "audio_pcm_base64": res.audio_pcm_base64,
                "sample_rate": res.sample_rate,
                "channels": res.channels,
            })),
            Err(e) => VarlinkReply::err(
                "io.syntrop.Sensory1.CaptureFailed",
                Some(json!({ "reason": e.to_string() })),
            ),
        }
    }

    async fn handle_capture_frame(&self, params: Option<&Value>) -> VarlinkReply {
        let device = params
            .and_then(|p| p.get("device"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let width = params
            .and_then(|p| p.get("width"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let height = params
            .and_then(|p| p.get("height"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let output = tokio::task::spawn_blocking(move || {
            capture_video_frame(device.as_deref(), width, height)
        })
        .await;

        match output {
            Ok(Ok(res)) => VarlinkReply::ok(json!({
                "image_base64": res.image_base64,
                "format": res.format,
            })),
            Ok(Err(runtimed_core::error::RuntimedError::DeviceNotFound(dev))) => VarlinkReply::err(
                "io.syntrop.Sensory1.DeviceNotFound",
                Some(json!({ "device": dev })),
            ),
            Ok(Err(e)) => VarlinkReply::err(
                "io.syntrop.Sensory1.CaptureFailed",
                Some(json!({ "reason": e.to_string() })),
            ),
            Err(e) => VarlinkReply::err(
                "io.syntrop.Sensory1.CaptureFailed",
                Some(json!({ "reason": format!("capture worker failed: {e}") })),
            ),
        }
    }

    async fn handle_capture_screen(&self, params: Option<&Value>) -> VarlinkReply {
        let display = params
            .and_then(|p| p.get("display"))
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let output = tokio::task::spawn_blocking(move || {
            capture_screen_image(display.as_deref())
        })
        .await;

        match output {
            Ok(Ok(res)) => VarlinkReply::ok(json!({
                "image_base64": res.image_base64,
                "format": res.format,
            })),
            Ok(Err(e)) => VarlinkReply::err(
                "io.syntrop.Sensory1.CaptureFailed",
                Some(json!({ "reason": e.to_string() })),
            ),
            Err(e) => VarlinkReply::err(
                "io.syntrop.Sensory1.CaptureFailed",
                Some(json!({ "reason": format!("screencopy worker failed: {e}") })),
            ),
        }
    }

    async fn handle_get_operator_presence(&self) -> VarlinkReply {
        let audio_res = match capture_audio_pcm(Some(300), Some(16000)).await {
            Ok(a) => a,
            Err(e) => {
                return VarlinkReply::err(
                    "io.syntrop.Sensory1.CaptureFailed",
                    Some(json!({ "reason": format!("audio sensing failed: {e}") })),
                )
            }
        };

        let video_output = tokio::task::spawn_blocking(|| {
            capture_video_frame(None, Some(320), Some(240))
        })
        .await;

        let video_res = match video_output {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                return VarlinkReply::err(
                    "io.syntrop.Sensory1.CaptureFailed",
                    Some(json!({ "reason": format!("visual sensing failed: {e}") })),
                )
            }
            Err(e) => {
                return VarlinkReply::err(
                    "io.syntrop.Sensory1.CaptureFailed",
                    Some(json!({ "reason": format!("visual worker failed: {e}") })),
                )
            }
        };

        let presence = evaluate_operator_presence(&audio_res, &video_res);
        VarlinkReply::ok(json!({
            "present": presence.present,
            "confidence": presence.confidence,
            "reason": presence.reason,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sensory1_handler_audio_and_presence() {
        let handler = Sensory1Handler::new();
        let reply = handler
            .handle_call("io.syntrop.Sensory1.CaptureAudio", None)
            .await
            .expect("reply");
        assert!(reply.error.is_none());
        assert!(reply.parameters.expect("params")["audio_pcm_base64"].is_string());

        let p_reply = handler
            .handle_call("io.syntrop.Sensory1.GetOperatorPresence", None)
            .await
            .expect("presence");
        assert!(p_reply.error.is_none());
        assert!(p_reply.parameters.expect("params")["confidence"].is_f64());
    }
}

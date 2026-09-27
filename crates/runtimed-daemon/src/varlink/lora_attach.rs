//! `AttachLora`: fuse a LoRA adapter into a loaded model.

use super::protocol::VarlinkReply;
use super::runtime1::Runtime1Handler;
use serde_json::json;

impl Runtime1Handler {
    pub(super) async fn handle_attach_lora(&self, params: Option<&serde_json::Value>) -> VarlinkReply {
        let params = match params {
            Some(p) => p,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "parameters" })),
                )
            }
        };
        let model_name = match params.get("model").and_then(|m| m.as_str()) {
            Some(m) => m,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "model" })),
                )
            }
        };
        let lora = match params.get("lora").and_then(|m| m.as_str()) {
            Some(m) => m,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "lora" })),
                )
            }
        };
        // Fusion walks gigabytes; keep it off the async executor.
        let manager = std::sync::Arc::clone(&self.model_manager);
        let name_owned = model_name.to_string();
        let lora_owned = lora.to_string();
        let out = tokio::task::spawn_blocking(move || manager.attach_lora(&name_owned, &lora_owned)).await;
        match out {
            Ok(Ok(fused)) => VarlinkReply::ok(json!({ "fused_tensors": fused })),
            Ok(Err(e)) => VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": e.to_string() })),
            ),
            Err(e) => VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": format!("worker failed: {e}") })),
            ),
        }
    }
}

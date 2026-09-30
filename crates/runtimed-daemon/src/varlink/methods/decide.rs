//! Handler implementation for io.syntrop.Runtime1.Decide.

use super::runtime1::Runtime1Handler;
use crate::varlink::server::protocol::VarlinkReply;
use runtimed_core::decide::{evaluate_decision, Candidate, DecideRequest};
use serde_json::{json, Value};
use std::sync::Arc;

impl Runtime1Handler {
    pub(super) async fn handle_decide(&self, params: Option<&Value>) -> VarlinkReply {
        let _permit = match self.acquire_permit() {
            Ok(p) => p,
            Err(r) => return r,
        };

        let params = match params {
            Some(p) => p,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "parameters" })),
                );
            }
        };

        let model_name = match params.get("model").and_then(|m| m.as_str()) {
            Some(m) => m,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "model" })),
                );
            }
        };

        let prompt = match params.get("prompt").and_then(|p| p.as_str()) {
            Some(p) => p,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "prompt" })),
                );
            }
        };

        let candidates: Vec<Candidate> = match params.get("candidates") {
            Some(c) => match serde_json::from_value(c.clone()) {
                Ok(cands) => cands,
                Err(e) => {
                    return VarlinkReply::err(
                        "io.syntrop.Runtime1.InvalidParameter",
                        Some(json!({ "parameter": format!("candidates: {e}") })),
                    );
                }
            },
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "candidates" })),
                );
            }
        };

        let temperature = params
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        let manager = Arc::clone(&self.model_manager);
        let name_owned = model_name.to_string();
        let loaded =
            tokio::task::spawn_blocking(move || manager.load_model(&name_owned, None)).await;
        match loaded {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": e.to_string() })),
                );
            }
            Err(e) => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": format!("loader failed: {e}") })),
                );
            }
        }

        let entry = match self.model_manager.get_entry(model_name) {
            Some(e) => e,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": "model vanished after load" })),
                );
            }
        };

        let req = DecideRequest {
            model: model_name.to_string(),
            prompt: prompt.to_string(),
            candidates,
            temperature,
        };

        let entry_task = Arc::clone(&entry);
        let output =
            tokio::task::spawn_blocking(move || evaluate_decision(&entry_task, &req)).await;
        let output = match output {
            Ok(o) => o,
            Err(e) => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": format!("decision task failed: {e}") })),
                );
            }
        };

        match output {
            Ok(res) => VarlinkReply::ok(json!({ "result": res })),
            Err(e) => VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": e.to_string() })),
            ),
        }
    }
}

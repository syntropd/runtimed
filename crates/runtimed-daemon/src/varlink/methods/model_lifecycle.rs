//! `GetModelStatus` / `UnloadModel` / `ListLoadedModels`: model lifecycle.

use super::runtime1::Runtime1Handler;
use crate::varlink::server::protocol::VarlinkReply;
use serde_json::{json, Value};

impl Runtime1Handler {
    pub(super) fn handle_get_model_status(&self, params: Option<&Value>) -> VarlinkReply {
        let model_name = match params.and_then(|p| p.get("model")).and_then(|m| m.as_str()) {
            Some(m) => m,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "model" })),
                )
            }
        };

        match self.model_manager.get_model(model_name) {
            Some(model) => VarlinkReply::ok(json!({
                "status": "loaded",
                "model": model,
            })),
            None => VarlinkReply::ok(json!({
                "status": "not_loaded",
                "model": Value::Null,
            })),
        }
    }

    pub(super) fn handle_unload_model(&self, params: Option<&Value>) -> VarlinkReply {
        let model_name = match params.and_then(|p| p.get("model")).and_then(|m| m.as_str()) {
            Some(m) => m,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "model" })),
                )
            }
        };

        match self.model_manager.unload_model(model_name) {
            Ok(freed) => VarlinkReply::ok(json!({ "freed_bytes": freed })),
            Err(e) => VarlinkReply::err(
                "io.syntrop.Runtime1.ModelNotFound",
                Some(json!({ "reason": e.to_string() })),
            ),
        }
    }

    pub(super) fn handle_list_loaded_models(&self) -> VarlinkReply {
        let models = self.model_manager.list_active();
        VarlinkReply::ok(json!({ "models": models }))
    }

    pub(super) async fn handle_embed(&self, params: Option<&Value>) -> VarlinkReply {
        let _permit = match self.acquire_permit() {
            Ok(p) => p,
            Err(r) => return r,
        };

        let text = match params.and_then(|p| p.get("text")).and_then(|t| t.as_str()) {
            Some(t) => t,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "text" })),
                )
            }
        };

        let embedding = runtimed_core::engine::generate_embedding(text);
        VarlinkReply::ok(json!({ "embedding": embedding }))
    }
}

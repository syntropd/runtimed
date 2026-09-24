//! Handler implementation for io.syntrop.Runtime1 Varlink interface.

use super::protocol::VarlinkReply;
use runtimed_core::engine::{generate_embedding, generate_tokens, GenerationRequest};
use runtimed_core::model::ModelManager;
use serde_json::{json, Value};
use std::sync::Arc;

/// Shared state required for Runtime1 method dispatches.
#[derive(Clone)]
pub struct Runtime1Handler {
    model_manager: Arc<ModelManager>,
}

impl Runtime1Handler {
    /// Constructs a new Runtime1Handler.
    pub fn new(model_manager: Arc<ModelManager>) -> Self {
        Self { model_manager }
    }

    /// Dispatches incoming io.syntrop.Runtime1 method calls.
    pub async fn handle_call(&self, method: &str, params: Option<&Value>) -> Option<VarlinkReply> {
        match method {
            "io.syntrop.Runtime1.Generate" => Some(self.handle_generate(params)),
            "io.syntrop.Runtime1.Embed" => Some(self.handle_embed(params)),
            "io.syntrop.Runtime1.GetModelStatus" => Some(self.handle_get_model_status(params)),
            "io.syntrop.Runtime1.UnloadModel" => Some(self.handle_unload_model(params)),
            "io.syntrop.Runtime1.ListLoadedModels" => Some(self.handle_list_loaded_models()),
            _ => None,
        }
    }

    fn handle_generate(&self, params: Option<&Value>) -> VarlinkReply {
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

        let prompt = match params.get("prompt").and_then(|p| p.as_str()) {
            Some(p) => p,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "prompt" })),
                )
            }
        };

        let max_tokens = params
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize;

        let temperature = params
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let model = match self.model_manager.load_model(model_name, None) {
            Ok(m) => m,
            Err(e) => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": e.to_string() })),
                )
            }
        };

        let request = GenerationRequest {
            model: model_name.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
        };

        match generate_tokens(&model, &request) {
            Ok(result) => VarlinkReply::ok(json!({ "result": result })),
            Err(e) => VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": e.to_string() })),
            ),
        }
    }

    fn handle_embed(&self, params: Option<&Value>) -> VarlinkReply {
        let text = match params.and_then(|p| p.get("text")).and_then(|t| t.as_str()) {
            Some(t) => t,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "text" })),
                )
            }
        };

        let embedding = generate_embedding(text);
        VarlinkReply::ok(json!({ "embedding": embedding }))
    }

    fn handle_get_model_status(&self, params: Option<&Value>) -> VarlinkReply {
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

    fn handle_unload_model(&self, params: Option<&Value>) -> VarlinkReply {
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

    fn handle_list_loaded_models(&self) -> VarlinkReply {
        let models = self.model_manager.list_active();
        VarlinkReply::ok(json!({ "models": models }))
    }
}

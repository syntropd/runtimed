//! Handler implementation for io.syntrop.Runtime1 Varlink interface.

use crate::varlink::server::protocol::VarlinkReply;
use runtimed_core::engine::{generate_embedding, generate_tokens, GenerationRequest};
use runtimed_core::model::ModelManager;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

/// Shared state required for Runtime1 method dispatches.
#[derive(Clone)]
pub struct Runtime1Handler {
    pub(super) model_manager: Arc<ModelManager>,
    pub(super) semaphore: Arc<Semaphore>,
    pub(super) max_slots: usize,
}

impl Runtime1Handler {
    /// Constructs a new Runtime1Handler with a built-in concurrency limit.
    pub fn new(model_manager: Arc<ModelManager>) -> Self {
        Self::with_max_concurrent(model_manager, 4)
    }

    /// Constructs a new Runtime1Handler with an explicit concurrency cap.
    pub fn with_max_concurrent(model_manager: Arc<ModelManager>, max_concurrent: usize) -> Self {
        let permits = if max_concurrent == 0 { 1 } else { max_concurrent };
        Self {
            model_manager,
            semaphore: Arc::new(Semaphore::new(permits)),
            max_slots: permits,
        }
    }

    /// Dispatches incoming io.syntrop.Runtime1 method calls.
    pub async fn handle_call(&self, method: &str, params: Option<&Value>) -> Option<VarlinkReply> {
        match method {
            "io.syntrop.Runtime1.Generate" => Some(self.handle_generate(params).await),
            "io.syntrop.Runtime1.AttachVision" => Some(self.handle_attach_vision(params).await),
            "io.syntrop.Runtime1.AttachLora" => Some(self.handle_attach_lora(params).await),
            "io.syntrop.Runtime1.GetLoad" => Some(self.handle_get_load()),
            "io.syntrop.Runtime1.Embed" => Some(self.handle_embed(params).await),
            "io.syntrop.Runtime1.GetModelStatus" => Some(self.handle_get_model_status(params)),
            "io.syntrop.Runtime1.UnloadModel" => Some(self.handle_unload_model(params)),
            "io.syntrop.Runtime1.ListLoadedModels" => Some(self.handle_list_loaded_models()),
            _ => None,
        }
    }

    fn acquire_permit(&self) -> Result<OwnedSemaphorePermit, VarlinkReply> {
        match Arc::clone(&self.semaphore).try_acquire_owned() {
            Ok(p) => Ok(p),
            Err(TryAcquireError::NoPermits) => Err(VarlinkReply::err(
                "io.syntrop.Runtime1.Overloaded",
                Some(json!({ "reason": "max concurrent requests reached" })),
            )),
            Err(TryAcquireError::Closed) => Err(VarlinkReply::err(
                "io.syntrop.Runtime1.Shutdown",
                Some(json!({ "reason": "daemon semaphore closed" })),
            )),
        }
    }

    async fn handle_generate(&self, params: Option<&Value>) -> VarlinkReply {
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

        let top_k = params
            .get("top_k")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let top_p = params
            .get("top_p")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        let seed = params
            .get("seed")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let image_base64 = params
            .get("image")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        // Loading dequantizes gigabytes; keep it off the async executor.
        let manager = Arc::clone(&self.model_manager);
        let name_owned = model_name.to_string();
        let loaded = tokio::task::spawn_blocking(move || manager.load_model(&name_owned, None)).await;
        match loaded {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": e.to_string() })),
                )
            }
            Err(e) => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": format!("loader failed: {e}") })),
                )
            }
        }
        let entry = match self.model_manager.get_entry(model_name) {
            Some(e) => e,
            None => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": "model vanished after load" })),
                )
            }
        };

        let request = GenerationRequest {
            model: model_name.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            image_base64,
        };

        // Generation is synchronous CPU work; run it off the async executor.
        let entry_task = Arc::clone(&entry);
        let output = tokio::task::spawn_blocking(move || generate_tokens(&entry_task, &request)).await;
        let output = match output {
            Ok(o) => o,
            Err(e) => {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.GenerationFailed",
                    Some(json!({ "reason": format!("worker failed: {e}") })),
                )
            }
        };
        match output {
            Ok(result) => VarlinkReply::ok(json!({ "result": result })),
            Err(e) => VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": e.to_string() })),
            ),
        }
    }

    async fn handle_embed(&self, params: Option<&Value>) -> VarlinkReply {
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

        let embedding = generate_embedding(text);
        VarlinkReply::ok(json!({ "embedding": embedding }))
    }
}

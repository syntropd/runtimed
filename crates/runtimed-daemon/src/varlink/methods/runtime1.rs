//! Handler implementation for io.syntrop.Runtime1 Varlink interface.

use crate::varlink::server::protocol::VarlinkReply;
use runtimed_core::engine::{generate_speculative, generate_tokens, GenerationRequest, ReasoningEffort};
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
            "io.syntrop.Runtime1.Decide" => Some(self.handle_decide(params).await),
            "io.syntrop.Runtime1.StreamAudioOut" => Some(super::multimedia::handle_stream_audio_out(params, Some(&self.model_manager)).await),
            "io.syntrop.Runtime1.TranscribeAudio" => Some(super::multimedia::handle_transcribe_audio(params, Some(&self.model_manager)).await),
            "io.syntrop.Runtime1.GenerateMusic" => Some(super::multimedia::handle_generate_music(params, Some(&self.model_manager)).await),
            "io.syntrop.Runtime1.GenerateAudio" => Some(super::multimedia::handle_generate_music(params, Some(&self.model_manager)).await),
            "io.syntrop.Runtime1.GenerateVisual" => Some(super::multimedia::handle_generate_visual(params, Some(&self.model_manager)).await),
            "io.syntrop.Runtime1.GenerateVideo" => Some(super::multimedia::handle_generate_video(params, Some(&self.model_manager)).await),
            "io.syntrop.Runtime1.CompactKvCache" => Some(self.handle_compact_kv_cache()),
            _ => None,
        }
    }

    pub(super) fn acquire_permit(&self) -> Result<OwnedSemaphorePermit, VarlinkReply> {
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

        let max_tokens = params.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
        let temperature = params.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let top_k = params.get("top_k").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let top_p = params.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(0);
        let image_base64 = params.get("image").and_then(|v| v.as_str()).map(str::to_string);
        let grammar_type = params.get("grammar_type").and_then(|v| v.as_str()).map(str::to_string);
        let grammar = params.get("grammar").and_then(|v| v.as_str()).map(str::to_string);
        let reasoning_budget = params.get("reasoning_budget").and_then(|v| v.as_u64()).map(|v| v as usize);
        let reasoning_effort = params.get("reasoning_effort").and_then(|v| v.as_str()).and_then(|s| s.parse::<ReasoningEffort>().ok());
        let speculative_draft_model = params.get("speculative_draft_model").and_then(|v| v.as_str()).map(str::to_string);
        let k_draft = params.get("k_draft").and_then(|v| v.as_u64()).unwrap_or(4) as usize;

        if let Some(ref draft_name) = speculative_draft_model {
            if draft_name == model_name {
                return VarlinkReply::err(
                    "io.syntrop.Runtime1.InvalidParameter",
                    Some(json!({ "parameter": "speculative_draft_model" })),
                );
            }
        }

        // Loading dequantizes gigabytes; keep it off the async executor.
        let manager = Arc::clone(&self.model_manager);
        let name_owned = model_name.to_string();
        let loaded = tokio::task::spawn_blocking(move || manager.load_model(&name_owned, None)).await;
        match loaded {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return VarlinkReply::err("io.syntrop.Runtime1.GenerationFailed", Some(json!({ "reason": e.to_string() }))),
            Err(e) => return VarlinkReply::err("io.syntrop.Runtime1.GenerationFailed", Some(json!({ "reason": format!("loader failed: {e}") }))),
        }
        let entry = match self.model_manager.get_entry(model_name) {
            Some(e) => e,
            None => return VarlinkReply::err("io.syntrop.Runtime1.GenerationFailed", Some(json!({ "reason": "model vanished after load" }))),
        };

        let draft_entry = match speculative_draft_model {
            Some(ref draft_name) => {
                let manager = Arc::clone(&self.model_manager);
                let draft_owned = draft_name.clone();
                let loaded = tokio::task::spawn_blocking(move || manager.load_model(&draft_owned, None)).await;
                match loaded {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => return VarlinkReply::err("io.syntrop.Runtime1.GenerationFailed", Some(json!({ "reason": format!("draft model load failed: {e}") }))),
                    Err(e) => return VarlinkReply::err("io.syntrop.Runtime1.GenerationFailed", Some(json!({ "reason": format!("draft loader failed: {e}") }))),
                }
                match self.model_manager.get_entry(draft_name) {
                    Some(e) => {
                        if Arc::ptr_eq(&entry, &e) {
                            return VarlinkReply::err(
                                "io.syntrop.Runtime1.InvalidParameter",
                                Some(json!({ "parameter": "speculative_draft_model" })),
                            );
                        }
                        Some(e)
                    }
                    None => return VarlinkReply::err("io.syntrop.Runtime1.GenerationFailed", Some(json!({ "reason": "draft model vanished after load" }))),
                }
            }
            None => None,
        };

        let effective_budget = reasoning_budget.or_else(|| {
            reasoning_effort.and_then(|e| e.to_budget(entry.meta.context_window))
        });

        let request = GenerationRequest {
            model: model_name.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            image_base64,
            grammar_type,
            grammar,
            reasoning_budget: effective_budget,
            reasoning_effort,
        };

        // Generation is synchronous CPU work; run it off the async executor.
        let entry_task = Arc::clone(&entry);
        let output = tokio::task::spawn_blocking(move || {
            if let Some(draft) = draft_entry {
                generate_speculative(&entry_task, &draft, &request, k_draft)
            } else {
                generate_tokens(&entry_task, &request)
            }
        }).await;

        let output = match output {
            Ok(o) => o,
            Err(e) => return VarlinkReply::err("io.syntrop.Runtime1.GenerationFailed", Some(json!({ "reason": format!("worker failed: {e}") }))),
        };
        match output {
            Ok(result) => VarlinkReply::ok(json!({ "result": result })),
            Err(e) => VarlinkReply::err(
                "io.syntrop.Runtime1.GenerationFailed",
                Some(json!({ "reason": e.to_string() })),
            ),
        }
    }

    fn handle_compact_kv_cache(&self) -> VarlinkReply {
        let freed = self.model_manager.unload_idle(0);
        let freed_bytes: usize = freed.iter().map(|(_, b)| b).sum();
        VarlinkReply::ok(json!({ "freed_bytes": freed_bytes }))
    }
}

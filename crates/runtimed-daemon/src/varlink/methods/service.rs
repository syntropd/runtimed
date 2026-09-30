//! Standard org.varlink.service introspection for runtimed.

use crate::varlink::server::protocol::VarlinkReply;
use serde_json::json;

/// Varlink interface definition text for io.syntrop.Runtime1.
pub const IO_SYNTROP_RUNTIME1_INTERFACE: &str = r#"
interface io.syntrop.Runtime1

# Concurrent execution is bounded by the daemon's `max_concurrent_requests`
# configuration value (default 4). When all permits are held, additional
# Generate and Embed calls return `io.syntrop.Runtime1.Overloaded` instead
# of queueing.

type LoadedModel (
  name: string,
  architecture: string,
  parameter_count: int,
  memory_bytes: int,
  context_window: int,
  compute_backend: string
)

type GenerationResult (
  text: string,
  prompt_tokens: int,
  completion_tokens: int,
  finish_reason: string,
  duration_ms: int
)

type Candidate (
  name: string,
  token: string
)

type ScoredCandidate (
  name: string,
  token: string,
  token_id: int,
  logit: float,
  probability: float
)

type DecisionResult (
  winner: string,
  confidence: float,
  raw_probability: float,
  margin: float,
  entropy: float,
  candidates: []ScoredCandidate
)

method Generate(model: string, prompt: string, max_tokens: int, temperature: float, top_k: int, top_p: float, seed: int, image: ?string, grammar_type: ?string, grammar: ?string, reasoning_budget: ?int, reasoning_effort: ?string) -> (result: GenerationResult)
method Decide(model: string, prompt: string, candidates: []Candidate, temperature: ?float) -> (result: DecisionResult)
method AttachVision(model: string, mmproj: string) -> (model: LoadedModel)
method AttachLora(model: string, lora: string) -> (fused_tensors: []string)
method GetLoad() -> (available_slots: int, max_slots: int, used_bytes: int, models: []LoadedModel)
method Embed(model: string, text: string) -> (embedding: []float)
method GetModelStatus(model: string) -> (status: string, model: ?LoadedModel)
method UnloadModel(model: string) -> (freed_bytes: int)
method ListLoadedModels() -> (models: []LoadedModel)
method StreamAudioOut(text: string, voice: ?string, sink_type: ?string) -> (bytes_streamed: int, sample_rate: int, channels: int)
method GenerateVisual(prompt: string, width: ?int, height: ?int, seed: ?int, steps: ?int, lease_id: ?string) -> (image_path: string, bytes: int, width: int, height: int, format: string)

error ModelNotFound(model: string)
error ContextExceeded(requested: int, max: int)
error GenerationFailed(reason: string)
error InvalidParameter(parameter: string)
error Overloaded(reason: string)
error Shutdown(reason: string)
error PermissionDenied()
"#;

/// Handles standard org.varlink.service method dispatches.
pub fn handle_service_call(method: &str, params: Option<&serde_json::Value>) -> Option<VarlinkReply> {
    match method {
        "org.varlink.service.GetInfo" => Some(VarlinkReply::ok(json!({
            "vendor": "Syntropd Project",
            "product": "runtimed",
            "version": env!("CARGO_PKG_VERSION"),
            "url": "https://github.com/syntropd/runtimed",
            "interfaces": [
                "org.varlink.service",
                "io.syntrop.Runtime1"
            ]
        }))),
        "org.varlink.service.GetInterfaceDescription" => {
            let iface = params
                .and_then(|p| p.get("interface"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            match iface {
                "io.syntrop.Runtime1" => Some(VarlinkReply::ok(json!({
                    "description": IO_SYNTROP_RUNTIME1_INTERFACE.trim()
                }))),
                _ => Some(VarlinkReply::err(
                    "org.varlink.service.InterfaceNotFound",
                    Some(json!({ "interface": iface })),
                )),
            }
        }
        _ => None,
    }
}
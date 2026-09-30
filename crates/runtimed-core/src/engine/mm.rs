//! Multimodal generation: image in, chat-prompted tokens out.

use super::generator::{entropy_seed, GenerationRequest, GenerationResult, Rng};
use crate::error::RuntimedError;
use crate::model::meta::EngineEntry;
use runtimed_model::decode::{chat, sample};
use std::time::Instant;

/// Largest accepted decoded image (32 MiB; the tower rejects junk after).
const MAX_IMAGE_BYTES: usize = 32 << 20;

/// Decode + validate the request's image, if any (pure: unit-testable).
fn request_image(request: &GenerationRequest) -> Result<Option<Vec<u8>>, RuntimedError> {
    let Some(b64) = request.image_base64.as_deref() else {
        return Ok(None);
    };
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|_| RuntimedError::GenerationFailed("image is not valid base64".into()))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(RuntimedError::GenerationFailed(format!(
            "image too large: {} bytes (max {MAX_IMAGE_BYTES})",
            bytes.len()
        )));
    }
    if bytes.is_empty() {
        return Err(RuntimedError::GenerationFailed("image is empty".into()));
    }
    Ok(Some(bytes))
}

/// Decode + validate the request's image, requiring it to be present (pure: unit-testable).
fn require_image(request: &GenerationRequest) -> Result<Vec<u8>, RuntimedError> {
    request_image(request)?.ok_or_else(|| {
        RuntimedError::GenerationFailed("missing required image for multimodal generation".into())
    })
}

/// Multimodal generation: image soft tokens scattered into a chat prompt.
pub(super) fn generate_mm_tokens(
    entry: &EngineEntry,
    request: &GenerationRequest,
    start: Instant,
) -> Result<GenerationResult, RuntimedError> {
    let bytes = require_image(request)?;
    let dev = {
        let session = entry
            .session
            .lock()
            .map_err(|_| RuntimedError::GenerationFailed("session lock poisoned".into()))?;
        session.device().clone()
    };
    let mut tower_lock = entry.vision.write().map_err(|_| {
        RuntimedError::GenerationFailed("vision lock poisoned".into())
    })?;
    let tower = tower_lock.as_mut().ok_or_else(|| {
        RuntimedError::GenerationFailed("model has no vision tower (attach one first)".into())
    })?;
    tower
        .pin(&dev)
        .map_err(|e| RuntimedError::GenerationFailed(format!("vision pin: {e}")))?;
    let bpe = entry.tokenizer.as_bpe().ok_or_else(|| {
        RuntimedError::GenerationFailed("multimodal needs a GGUF-BPE model".into())
    })?;
    let prep = runtimed_model::vision::prepare(&bytes, tower.config())
        .map_err(|e| RuntimedError::GenerationFailed(format!("image: {e}")))?;
    let soft = tower
        .encode(&prep)
        .map_err(|e| RuntimedError::GenerationFailed(format!("vision: {e}")))?;
    tower
        .unpin()
        .map_err(|e| RuntimedError::GenerationFailed(format!("vision unpin: {e}")))?;
    drop(tower_lock);
    let prompt_ids =
        chat::mm_prompt(bpe, &request.prompt, prep.n_soft).map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?;
    if prompt_ids.len() > entry.meta.context_window {
        return Err(RuntimedError::ContextExceeded {
            max: entry.meta.context_window,
            requested: prompt_ids.len(),
        });
    }
    let budget = request
        .max_tokens
        .min(entry.meta.context_window - prompt_ids.len());
    if budget == 0 {
        return Ok(GenerationResult {
            text: String::new(),
            prompt_tokens: prompt_ids.len(),
            completion_tokens: 0,
            finish_reason: "length".to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
        });
    }
    let mut session = entry
        .session
        .lock()
        .map_err(|_| RuntimedError::GenerationFailed("session lock poisoned".into()))?;
    let seed = if request.seed == 0 { entropy_seed() } else { request.seed };
    let mut rng = Rng(seed);
    let pad = bpe.pad_id();
    let ids = runtimed_model::decode::generate::generate_mm(
        &mut session,
        &prompt_ids,
        &soft,
        pad,
        &entry.eos,
        budget,
        |logits| sample::sample(logits, request.temperature, request.top_k, request.top_p, || rng.next_f32()),
    )
    .map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?;
    drop(session);
    let text = entry.tokenizer.decode(&ids)?;
    Ok(super::generator::finish(entry, prompt_ids.len(), &ids, text, start))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(image: Option<&str>) -> GenerationRequest {
        GenerationRequest {
            model: "m".into(),
            prompt: "p".into(),
            max_tokens: 8,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 1,
            image_base64: image.map(str::to_string),
            grammar_type: None,
            grammar: None,
            reasoning_budget: None,
            reasoning_effort: None,
        }
    }

    #[test]
    fn no_image_yields_none() {
        assert!(request_image(&req(None)).unwrap().is_none());
    }

    #[test]
    fn bad_base64_errors() {
        assert!(request_image(&req(Some("!!! not base64 !!!"))).is_err());
    }

    #[test]
    fn empty_image_errors() {
        assert!(request_image(&req(Some(""))).is_err());
    }

    #[test]
    fn tiny_png_decodes() {
        // 1x1 transparent PNG.
        let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let bytes = request_image(&req(Some(b64))).unwrap().unwrap();
        assert_eq!(&bytes[1..4], b"PNG");
    }

    #[test]
    fn require_image_missing_errors() {
        let err = require_image(&req(None)).unwrap_err();
        match err {
            RuntimedError::GenerationFailed(msg) => {
                assert!(msg.contains("missing required image"));
            }
            other => panic!("expected GenerationFailed, got {:?}", other),
        }
    }

    #[test]
    fn require_image_valid_png() {
        let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let bytes = require_image(&req(Some(b64))).unwrap();
        assert_eq!(&bytes[1..4], b"PNG");
    }
}

//! Pure Rust token generation engine.

use crate::error::RuntimedError;
use crate::model::meta::EngineEntry;
use crate::model::tokenizer::EngineTokenizer;
use super::mm;
use runtimed_model::decode::{chat, sample};
use runtimed_model::generate;
use serde::{Deserialize, Serialize};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Parameters for a text generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub image_base64: Option<String>,
    #[serde(default)]
    pub grammar_type: Option<String>,
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub reasoning_budget: Option<usize>,
}

fn default_top_p() -> f32 {
    1.0
}

/// Output payload from a completed generation run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub duration_ms: u64,
}

pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 11) as f32 / ((1u64 << 53) as f32)
    }
}

pub(super) fn entropy_seed() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0x243F_6A88_85A3_08D3) | 1
}

fn text_prompt_ids(
    tokenizer: &EngineTokenizer,
    prompt: &str,
    add_special: bool,
) -> Result<Vec<u32>, RuntimedError> {
    match tokenizer.as_bpe() {
        Some(bpe) => chat::text_prompt(bpe, prompt).map_err(|e| RuntimedError::GenerationFailed(e.to_string())),
        None => tokenizer.encode(prompt, add_special).map_err(Into::into),
    }
}

fn generate_guided_tokens(
    session: &mut runtimed_model::Session,
    entry: &EngineEntry,
    prompt_ids: &[u32],
    budget: usize,
    req: &GenerationRequest,
    mut rng: Rng,
) -> Result<Vec<u32>, RuntimedError> {
    use runtimed_model::sampler::*;
    let trie = VocabTrie::from_tokenizer(&entry.tokenizer);
    let (t, k, p) = (req.temperature, req.top_k, req.top_p);
    let mut step = |g: &dyn FsmGrammar, st: &mut FsmState| {
        generate(session, prompt_ids, &entry.eos, budget, |l| {
            sample_with_grammar(l, g, st, &trie, &entry.eos, t, k, p, || rng.next_f32())
        }).map_err(|e| RuntimedError::GenerationFailed(e.to_string()))
    };
    match req.grammar_type.as_deref().unwrap_or("json") {
        "regex" => {
            let g = RegexFsm::compile(req.grammar.as_deref().unwrap_or(r"\d+"))?;
            step(&g, &mut FsmState::new(g.initial_state()))
        }
        "varlink" => step(&VarlinkFsm::new(), &mut FsmState::new(0)),
        _ => step(&JsonFsm::new(), &mut FsmState::new(0)),
    }
}

fn generate_with_budget(
    session: &mut runtimed_model::Session,
    entry: &EngineEntry,
    prompt_ids: &[u32],
    budget: usize,
    reasoning_budget: usize,
    req: &GenerationRequest,
    mut rng: Rng,
) -> Result<Vec<u32>, RuntimedError> {
    use super::think_budget::{ThinkBudget, ThinkingPhase};
    let end_id = entry.tokenizer.encode("</think>", false)
        .ok()
        .and_then(|v| v.first().copied())
        .unwrap_or_else(|| entry.eos.first().copied().unwrap_or(0));
    let mut tb = ThinkBudget::with_initial_phase(Some(reasoning_budget), end_id, ThinkingPhase::Thinking);
    if let Ok(toks) = entry.tokenizer.encode("<think>", false) {
        if let Some(&tid) = toks.first() { tb.set_think_token_id(tid); }
    }
    generate(session, prompt_ids, &entry.eos, budget, |logits| {
        let mut row = logits.to_vec1::<f32>()?;
        tb.enforce_logits(&mut row);
        let masked = candle_core::Tensor::from_vec(row, logits.shape(), logits.device())?;
        let tok = sample::sample(&masked, req.temperature, req.top_k, req.top_p, || rng.next_f32())?;
        tb.step(tok);
        Ok(tok)
    }).map_err(|e| RuntimedError::GenerationFailed(e.to_string()))
}

/// Executes token generation for a prompt against a loaded model.
pub fn generate_tokens(
    entry: &EngineEntry,
    request: &GenerationRequest,
) -> Result<GenerationResult, RuntimedError> {
    let start = Instant::now();
    if request.image_base64.is_some() {
        return mm::generate_mm_tokens(entry, request, start);
    }
    let prompt_ids = text_prompt_ids(&entry.tokenizer, &request.prompt, entry.add_special)?;

    if prompt_ids.len() > entry.meta.context_window {
        return Err(RuntimedError::ContextExceeded {
            max: entry.meta.context_window,
            requested: prompt_ids.len(),
        });
    }
    let budget = request.max_tokens.min(entry.meta.context_window - prompt_ids.len());
    if budget == 0 {
        return Ok(GenerationResult {
            text: String::new(),
            prompt_tokens: prompt_ids.len(),
            completion_tokens: 0,
            finish_reason: "length".to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
        });
    }

    let mut session = entry.session.lock().map_err(|_| RuntimedError::GenerationFailed("session lock poisoned".into()))?;
    let seed = if request.seed == 0 { entropy_seed() } else { request.seed };
    let mut rng = Rng(seed);
    let ids = if request.grammar_type.is_some() || request.grammar.is_some() {
        generate_guided_tokens(&mut *session, entry, &prompt_ids, budget, request, rng)?
    } else if let Some(rb) = request.reasoning_budget {
        generate_with_budget(&mut *session, entry, &prompt_ids, budget, rb, request, rng)?
    } else {
        generate(&mut *session, &prompt_ids, &entry.eos, budget, |logits| {
            sample::sample(logits, request.temperature, request.top_k, request.top_p, || rng.next_f32())
        }).map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?
    };
    drop(session);

    let text = entry.tokenizer.decode(&ids)?;
    Ok(finish(entry, prompt_ids.len(), &ids, text, start))
}

pub(super) fn finish(
    entry: &EngineEntry,
    prompt_tokens: usize,
    ids: &[u32],
    text: String,
    start: Instant,
) -> GenerationResult {
    let stopped = ids.last().is_some_and(|id| entry.eos.contains(id));
    let result = GenerationResult {
        text,
        prompt_tokens,
        completion_tokens: ids.len(),
        finish_reason: if stopped { "stop".into() } else { "length".into() },
        duration_ms: start.elapsed().as_millis() as u64,
    };
    tracing::info!(model = %entry.meta.name, prompt = prompt_tokens, completion = ids.len(), ms = result.duration_ms, "generate");
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gated_q4k() -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from(std::env::var("SYNTROP_TEST_GGUF_Q4K").ok()?);
        if !path.exists() { return None; }
        Some(path)
    }

    #[test]
    fn rng_is_deterministic_per_seed() {
        let (mut a, mut b) = (Rng(42), Rng(42));
        for _ in 0..8 { assert_eq!(a.next_f32(), b.next_f32()); }
        let mut c = Rng(42);
        let (x, y) = (c.next_f32(), c.next_f32());
        assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y));
    }

    #[test]
    fn entropy_seed_is_odd() {
        assert_eq!(entropy_seed() & 1, 1);
    }

    #[test]
    fn request_serde_applies_sampler_defaults() {
        let req: GenerationRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "p", "max_tokens": 8, "temperature": 0.0
        })).unwrap();
        assert_eq!(req.top_p, 1.0);
        assert_eq!(req.top_k, 0);
        assert_eq!(req.seed, 0);
        assert_eq!(req.image_base64, None);
        assert_eq!(req.grammar_type, None);
        assert_eq!(req.reasoning_budget, None);
    }

    #[test]
    fn bpe_text_prompts_get_chat_template() {
        let Some(path) = gated_q4k() else { return };
        let file = runtimed_gguf::GgufFile::open(&path).expect("parse");
        let bpe = runtimed_gguf::GgufBpe::from_gguf(&file).expect("bpe");
        let tok = EngineTokenizer::Bpe(bpe);
        let ids = text_prompt_ids(&tok, "Say hello.", true).expect("prompt ids");
        assert_eq!(&ids[..2], &[2, 105]);
        let raw = tok.encode("Say hello.", true).expect("raw");
        assert!(ids.len() > raw.len() + 8);
        assert_ne!(ids, raw);
    }
}

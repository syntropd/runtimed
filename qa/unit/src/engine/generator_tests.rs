//! Unit QA tests for token generation engine.
//!
//! Gated on `SYNTROP_TEST_GGUF`: generation runs against the real tiny
//! model, so success here means the full load → tokenize → forward →
//! sample → decode path works end to end.

#[cfg(test)]
mod tests {
    use runtimed_core::engine::{generate_tokens, GenerationRequest};
    use runtimed_core::model::ModelManager;
    use std::path::PathBuf;

    fn gated() -> Option<(ModelManager, String)> {
        let gguf = PathBuf::from(std::env::var("SYNTROP_TEST_GGUF").ok()?);
        if !gguf.exists() {
            eprintln!("skip: SYNTROP_TEST_GGUF not set or missing");
            return None;
        }
        let dir = gguf.parent().unwrap().to_path_buf();
        let stem = gguf.file_stem().unwrap().to_string_lossy().to_string();
        Some((ModelManager::new(dir), stem))
    }

    fn request(model: &str, prompt: &str, max_tokens: usize) -> GenerationRequest {
        GenerationRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            image_base64: None,
            grammar_type: None,
            grammar: None,
            reasoning_budget: None,
        }
    }

    #[test]
    fn test_generate_tokens_success() {
        let Some((manager, name)) = gated() else { return };
        manager.load_model(&name, None).unwrap();
        let entry = manager.get_entry(&name).unwrap();
        let req = request(&name, "Why did nginx fail to start on port 80?", 16);

        let result = generate_tokens(&entry, &req).unwrap();
        assert!(!result.text.is_empty());
        assert!(result.prompt_tokens > 5, "prompt tokens: {}", result.prompt_tokens);
        assert!(result.completion_tokens > 0 && result.completion_tokens <= 16);
        assert!(["stop", "length"].contains(&result.finish_reason.as_str()));
    }

    #[test]
    fn test_generate_respects_token_budget() {
        let Some((manager, name)) = gated() else { return };
        manager.load_model(&name, None).unwrap();
        let entry = manager.get_entry(&name).unwrap();
        // This prompt rambles without end-of-turn, so a budget of 3 must
        // return exactly 3 tokens with a length stop.
        let req = request(&name, "Count the r's in the word strawberry.", 3);

        let result = generate_tokens(&entry, &req).unwrap();
        assert_eq!(result.completion_tokens, 3);
        assert_eq!(result.finish_reason, "length");
    }

    #[test]
    fn test_greedy_generation_is_deterministic() {
        let Some((manager, name)) = gated() else { return };
        manager.load_model(&name, None).unwrap();
        let entry = manager.get_entry(&name).unwrap();
        let req = request(&name, "The capital of France is", 8);

        let first = generate_tokens(&entry, &req).unwrap();
        let second = generate_tokens(&entry, &req).unwrap();
        assert_eq!(first.text, second.text);
        assert_eq!(first.completion_tokens, second.completion_tokens);
    }
}

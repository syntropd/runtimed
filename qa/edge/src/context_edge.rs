//! Edge tests for context window limit enforcement.
//!
//! Gated on `SYNTROP_TEST_GGUF`: the over-long prompt is tokenized with
//! the real tokenizer (fast) and rejected before any forward pass.

#[cfg(test)]
mod tests {
    use runtimed_core::engine::{generate_tokens, GenerationRequest};
    use runtimed_core::error::RuntimedError;
    use runtimed_core::model::ModelManager;
    use std::path::PathBuf;

    #[test]
    fn test_prompt_exceeding_context_returns_error() {
        let Ok(gguf) = std::env::var("SYNTROP_TEST_GGUF") else { return };
        let gguf = PathBuf::from(gguf);
        if !gguf.exists() {
            eprintln!("skip: SYNTROP_TEST_GGUF not set or missing");
            return;
        }
        let manager = ModelManager::new(gguf.parent().unwrap());
        let name = gguf.file_stem().unwrap().to_string_lossy().to_string();
        let meta = manager.load_model(&name, None).unwrap();
        let entry = manager.get_entry(&name).unwrap();

        // 40k words tokenize to more than the 32k context window.
        let words = vec!["word"; 40_000].join(" ");
        let request = GenerationRequest {
            model: name,
            prompt: words,
            max_tokens: 32,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            image_base64: None,
            grammar_type: None,
            grammar: None,
            reasoning_budget: None,
        };

        match generate_tokens(&entry, &request) {
            Err(RuntimedError::ContextExceeded { max, requested }) => {
                assert_eq!(max, meta.context_window);
                assert!(requested > max, "requested: {requested}");
            }
            other => panic!("expected ContextExceeded, got {other:?}"),
        }
    }
}

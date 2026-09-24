//! Unit QA tests for token generation engine.

#[cfg(test)]
mod tests {
    use runtimed_core::engine::{generate_tokens, GenerationRequest};
    use runtimed_core::model::LoadedModel;

    fn dummy_model() -> LoadedModel {
        LoadedModel {
            name: "test-model".to_string(),
            architecture: "transformer".to_string(),
            parameter_count: 7_000_000_000,
            memory_bytes: 4_000_000_000,
            context_window: 2048,
            compute_backend: "cpu".to_string(),
        }
    }

    #[test]
    fn test_generate_tokens_success() {
        let model = dummy_model();
        let request = GenerationRequest {
            model: "test-model".to_string(),
            prompt: "Why did nginx fail to start on port 80?".to_string(),
            max_tokens: 64,
            temperature: 0.0,
        };

        let result = generate_tokens(&model, &request).unwrap();
        assert!(!result.text.is_empty());
        assert_eq!(result.prompt_tokens, 9);
        assert!(result.completion_tokens > 0);
        assert_eq!(result.finish_reason, "stop");
    }

    #[test]
    fn test_generate_respects_token_budget() {
        let model = dummy_model();
        let request = GenerationRequest {
            model: "test-model".to_string(),
            prompt: "systemd crash report".to_string(),
            max_tokens: 3,
            temperature: 0.0,
        };

        let result = generate_tokens(&model, &request).unwrap();
        assert!(result.completion_tokens <= 3);
        assert_eq!(result.finish_reason, "length");
    }
}

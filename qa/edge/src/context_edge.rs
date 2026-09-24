//! Edge tests for context window limit enforcement.

#[cfg(test)]
mod tests {
    use runtimed_core::engine::{generate_tokens, GenerationRequest};
    use runtimed_core::error::RuntimedError;
    use runtimed_core::model::LoadedModel;

    #[test]
    fn test_prompt_exceeding_context_returns_error() {
        let model = LoadedModel {
            name: "tiny-model".to_string(),
            architecture: "transformer".to_string(),
            parameter_count: 100_000_000,
            memory_bytes: 100_000_000,
            context_window: 10, // Very small context limit
            compute_backend: "cpu".to_string(),
        };

        // Create a prompt with 20 tokens
        let words = vec!["word"; 20].join(" ");
        let request = GenerationRequest {
            model: "tiny-model".to_string(),
            prompt: words,
            max_tokens: 32,
            temperature: 0.0,
        };

        let result = generate_tokens(&model, &request);
        assert!(result.is_err());
        match result.unwrap_err() {
            RuntimedError::ContextExceeded { max, requested } => {
                assert_eq!(max, 10);
                assert_eq!(requested, 20);
            }
            other => panic!("Unexpected error type: {:?}", other),
        }
    }
}

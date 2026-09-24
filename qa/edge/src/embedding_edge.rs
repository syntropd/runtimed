//! Edge tests for vector embedding boundary conditions.

#[cfg(test)]
mod tests {
    use runtimed_core::engine::{generate_embedding, EMBEDDING_DIM};

    #[test]
    fn test_empty_string_embedding() {
        let emb = generate_embedding("");
        assert_eq!(emb.len(), EMBEDDING_DIM);
        assert!(emb.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_whitespace_and_punctuation_only() {
        let emb = generate_embedding("   \t\n  !@#$%^&*()   ");
        assert_eq!(emb.len(), EMBEDDING_DIM);
        assert!(emb.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_non_ascii_unicode_embedding() {
        let emb = generate_embedding("Linux kernel 内核 🐧 error in module");
        assert_eq!(emb.len(), EMBEDDING_DIM);
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}

//! Unit QA tests for vector embedding engine.

#[cfg(test)]
mod tests {
    use runtimed_core::engine::{
        cosine_similarity, generate_embedding, EMBEDDING_DIM,
    };

    #[test]
    fn test_embedding_dimensions_and_normalization() {
        let text = "oom-killer invoked in user.slice on cgroup memory pressure";
        let emb = generate_embedding(text);

        assert_eq!(emb.len(), EMBEDDING_DIM);

        // Vector magnitude should be approximately 1.0 (L2 normalized)
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_identical_texts_have_maximum_similarity() {
        let text = "Failed to bind to 0.0.0.0:443 address already in use";
        let a = generate_embedding(text);
        let b = generate_embedding(text);

        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_semantic_overlap_similarity() {
        let a = generate_embedding("nginx worker process killed by SIGSEGV");
        let b = generate_embedding("apache worker process killed by SIGSEGV");
        let c = generate_embedding("quantum chromodynamics simulation finished");

        let sim_ab = cosine_similarity(&a, &b);
        let sim_ac = cosine_similarity(&a, &c);

        assert!(sim_ab > sim_ac);
    }
}

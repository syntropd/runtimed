//! Pure Rust normalized vector embedding generator.

/// Embedding vector dimension produced by runtimed.
pub const EMBEDDING_DIM: usize = 128;

/// Generates a deterministic L2-normalized embedding vector for an input text.
pub fn generate_embedding(text: &str) -> Vec<f32> {
    let mut vec = vec![0.0f32; EMBEDDING_DIM];

    if text.trim().is_empty() {
        return vec;
    }

    // Token frequency hashing into fixed dimension space
    for word in text.split_whitespace() {
        let clean: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();

        if clean.is_empty() {
            continue;
        }

        let mut hash = 5381u64;
        for byte in clean.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }

        let idx = (hash as usize) % EMBEDDING_DIM;
        let sign = if (hash >> 16) & 1 == 0 { 1.0f32 } else { -1.0f32 };
        vec[idx] += sign;
    }

    // L2 normalization: vector dot product equals cosine similarity
    let sum_sq: f32 = vec.iter().map(|v| v * v).sum();
    let norm = sum_sq.sqrt();

    if norm > 1e-6 {
        for v in &mut vec {
            *v /= norm;
        }
    }

    vec
}

/// Computes the cosine similarity between two embedding vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

//! Generation and embedding inference engines.

pub mod embedder;
pub mod generator;

pub use embedder::{cosine_similarity, generate_embedding, EMBEDDING_DIM};
pub use generator::{generate_tokens, GenerationRequest, GenerationResult};

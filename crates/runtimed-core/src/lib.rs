//! Core engine for runtimed: Headless Model Execution and Token Generation.

pub mod config;
pub mod engine;
pub mod error;
pub mod model;

pub use config::{
    RuntimedConfig, DEFAULT_CONFIG_PATH, DEFAULT_MODELS_PATH, DEFAULT_SOCKET_PATH,
};
pub use engine::{
    cosine_similarity, generate_embedding, generate_tokens, GenerationRequest, GenerationResult,
    EMBEDDING_DIM,
};
pub use error::RuntimedError;
pub use model::{LoadedModel, ModelManager};

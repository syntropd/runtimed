//! Model errors: bad configs, missing weights, tensor failures.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unsupported architecture: {0}")]
    Arch(String),
    #[error("bad config: {0}")]
    Config(String),
    #[error("missing weight: {0}")]
    MissingWeight(String),
    #[error("shape mismatch on {name}: expected {expected:?}, got {got:?}")]
    Shape {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("gguf: {0}")]
    Gguf(#[from] runtimed_gguf::GgufError),
    #[error("tensor: {0}")]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, ModelError>;

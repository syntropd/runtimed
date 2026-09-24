//! Error types for runtimed model execution.

use thiserror::Error;

/// Error categories emitted during model loading and inference.
#[derive(Debug, Error)]
pub enum RuntimedError {
    /// Requested model is not loaded or not found in storage.
    #[error("Model '{0}' not found or not loaded")]
    ModelNotFound(String),

    /// Prompt token count exceeds permitted model context window.
    #[error("Context limit exceeded: requested {requested}, model maximum is {max}")]
    ContextExceeded {
        /// Maximum tokens supported by model.
        max: usize,
        /// Number of tokens requested.
        requested: usize,
    },

    /// Generation was interrupted by a higher-priority emergency request.
    #[error("Inference pre-empted by priority supervisor: {0}")]
    Preempted(String),

    /// Hardware compute resource lease was rejected or evicted by inferenced.
    #[error("Hardware compute allocation failure: {0}")]
    HardwareAllocation(String),

    /// Token sampling or generation loop encountered an unrecoverable failure.
    #[error("Generation execution failed: {0}")]
    GenerationFailed(String),

    /// Configuration parsing failure.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Standard I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

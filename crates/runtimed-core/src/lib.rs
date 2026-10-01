//! Core engine for runtimed: Headless Model Execution and Token Generation.

pub mod config;
pub mod decide;
pub mod engine;
pub mod error;
pub mod governor;
pub mod memory;
pub mod model;
pub mod psi;

pub use config::DEFAULT_CONFIG_PATH;
pub use config::DEFAULT_MODELS_PATH;
pub use config::DEFAULT_SOCKET_PATH;
pub use config::RuntimedConfig;
pub use engine::EMBEDDING_DIM;
pub use engine::GenerationRequest;
pub use engine::GenerationResult;
pub use engine::cosine_similarity;
pub use engine::generate_embedding;
pub use engine::generate_tokens;
pub use engine::{BudgetAction, JournalEntry, StreamJournal, ThinkBudget, ThinkingPhase};
pub use decide::{
    evaluate_decision, Candidate, DecideRequest, DecisionResult, ScoredCandidate,
};
pub use error::RuntimedError;
pub use model::EngineEntry;
pub use model::EngineTokenizer;
pub use model::LoadedModel;
pub use model::ModelManager;
pub use memory::{
    decode_loop_managed, evaluate_and_spill, sample_vram_metrics, DualWatermarkController,
    WatermarkDecision, DEFAULT_CHECK_INTERVAL,
};

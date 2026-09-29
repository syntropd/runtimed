//! Generation, speculative decoding, and embedding inference engines.
//!
//! Provides the core execution logic for text generation, constrained decoding,
//! heterogeneous speculative acceleration, and reasoning budget management.

pub mod embedder;
pub mod generator;
pub mod mm;
pub mod speculate_engine;
pub mod stream_journal;
pub mod think_budget;

pub use embedder::{cosine_similarity, generate_embedding, EMBEDDING_DIM};
pub use generator::{generate_tokens, GenerationRequest, GenerationResult};
pub use speculate_engine::generate_speculative;
pub use stream_journal::{JournalEntry, StreamJournal};
pub use think_budget::{BudgetAction, ThinkBudget, ThinkingPhase};

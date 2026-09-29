//! Model definitions, descriptors, and lifecycle management.
//!
//! Submodules manage model admission leasing (`admit_lease`), tensor
//! attachment (`attach`), zero-copy SCM_RIGHTS descriptor handoff (`cas`),
//! idle memory unload eviction (`idle`), inference loading (`loader`),
//! architecture metadata (`meta`), filesystem path resolution (`resolve`),
//! multi-device stage loading (`stage_loader`),
//! and architecture-specific tokenizers (`tokenizer`).

pub mod admit_lease;
pub mod attach;
pub mod cas;
pub use cas as cas_fd;
pub mod idle;
pub mod loader;
pub mod meta;
pub mod resolve;
pub mod stage_loader;
pub mod tokenizer;

pub use cas::fetch_model_fd;
pub use loader::ModelManager;
pub use meta::{EngineEntry, LoadedModel};
pub use stage_loader::{StageLoader, StagePartition};
pub use tokenizer::EngineTokenizer;

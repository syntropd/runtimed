//! Model definitions, descriptors, and lifecycle management.
//!
//! Submodules manage model admission leasing (`admit_lease`), tensor
//! attachment (`attach`), zero-copy SCM_RIGHTS descriptor handoff (`cas_fd`),
//! idle memory unload eviction (`idle`), inference loading (`loader`),
//! architecture metadata (`meta`), filesystem path resolution (`resolve`),
//! and architecture-specific tokenizers (`tokenizer`).

pub mod admit_lease;
pub mod attach;
pub mod cas_fd;
pub mod idle;
pub mod loader;
pub mod meta;
pub mod resolve;
pub mod tokenizer;

pub use cas_fd::fetch_model_fd;
pub use loader::ModelManager;
pub use meta::{EngineEntry, LoadedModel};
pub use tokenizer::EngineTokenizer;

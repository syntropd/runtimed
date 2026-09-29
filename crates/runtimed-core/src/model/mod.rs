//! Model definitions, descriptors, and lifecycle management.

pub mod admit_lease;
pub mod attach;
pub mod idle;
pub mod loader;
pub mod meta;
pub mod resolve;
pub mod tokenizer;

pub use loader::ModelManager;
pub use meta::{EngineEntry, LoadedModel};
pub use tokenizer::EngineTokenizer;

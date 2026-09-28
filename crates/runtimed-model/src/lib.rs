//! Owned LLM forward pass on the candle tensor substrate.
//!
//! The math here is ours: arch configs parsed from GGUF metadata, weights
//! decoded by `runtimed-gguf`, every op (norm, RoPE, attention, MLP) written
//! out explicitly. Candle provides matmul/softmax/transpose only.

pub mod arch;
pub mod config;
pub mod decode;
pub mod error;
pub mod lora;
pub mod ops;
pub mod vision;
pub mod weights;

pub use config::{Activation, Arch, ArchConfig, LayerConfig};
pub use decode::generate::generate;
pub use decode::session::Session;
pub use error::{ModelError, Result};
pub use lora::LoraAdapter;
pub use vision::{PreparedImage, VisionTower};
pub use weights::Weights;

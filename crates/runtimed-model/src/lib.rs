//! Owned LLM forward pass on the candle tensor substrate.
//!
//! The math here is ours: arch configs parsed from GGUF metadata, weights
//! decoded by `runtimed-gguf`, every op (norm, RoPE, attention, MLP) written
//! out explicitly. Candle provides matmul/softmax/transpose only.

pub mod chat;
pub mod config;
pub mod lora;
pub mod error;
pub mod gemma4;
pub mod generate;
pub mod ops;
pub mod qwen2;
pub mod sample;
pub mod session;
pub mod vision;
pub mod vpre;
pub mod vresize;
pub mod weights;

pub use config::{Activation, Arch, ArchConfig, LayerConfig};
pub use error::{ModelError, Result};
pub use generate::generate;
pub use session::Session;
pub use vision::{PreparedImage, VisionTower};
pub use weights::Weights;
pub use lora::LoraAdapter;

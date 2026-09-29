//! Owned LLM forward pass on the candle tensor substrate.
//!
//! The math here is ours: arch configs parsed from GGUF metadata, weights
//! decoded by `runtimed-gguf`, every op (norm, RoPE, attention, MLP) written
//! out explicitly. Candle provides matmul/softmax/transpose only.

pub mod arch;
pub mod cache;
pub mod config;
pub mod decode;
pub mod error;
pub mod lora;
pub mod ops;
pub mod pipeline;
pub mod tp;
pub mod vision;
pub mod weights;

pub use cache::{CacheBlock, PagedKvCache, SpillManager, StorageTier, BLOCK_SIZE};
pub use config::{Activation, Arch, ArchConfig, LayerConfig};
pub use decode::generate::generate;
pub use decode::session::Session;
pub use error::{ModelError, Result};
pub use lora::LoraAdapter;
pub use pipeline::{forward_pipeline, PipelineStage};
pub use tp::{ColumnParallelLinear, RingReducer, RowParallelLinear, ring_all_reduce};
pub use vision::{PreparedImage, VisionTower};
pub use weights::Weights;

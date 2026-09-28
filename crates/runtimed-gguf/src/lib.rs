//! GGUF parsing, dequantization, and weight registry.
//!
//! Pure Rust, zero ML dependencies: this crate turns weight bytes into
//! `f32` tensors and refuses to run anything it cannot hash. The engine
//! crate consumes it; modeld reuses it for verification.

pub mod error;
pub mod header;
pub mod quant;
pub mod registry;
pub mod tokens;

pub use error::{GgufError, Result};
pub use header::{GgufFile, MetaValue, TensorInfo};
pub use quant::dtype::GgmlDtype;
pub use registry::{verify_file, Registry, WeightEntry};
pub use tokens::bpe::GgufBpe;
pub use tokens::tok::Tokenizer;

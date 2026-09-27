//! GGUF parsing, dequantization, and weight registry.
//!
//! Pure Rust, zero ML dependencies: this crate turns weight bytes into
//! `f32` tensors and refuses to run anything it cannot hash. The engine
//! crate consumes it; modeld reuses it for verification.

pub mod dequant;
pub mod dtype;
pub mod error;
pub mod header;
pub mod registry;

pub use dtype::GgmlDtype;
pub use error::{GgufError, Result};
pub use header::{GgufFile, MetaValue, TensorInfo};
pub use registry::{verify_file, Registry, WeightEntry};

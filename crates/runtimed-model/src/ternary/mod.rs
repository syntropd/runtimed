//! 1.58-bit ternary neural network inference engine (BitNet b1.58).
//!
//! Provides packed ternary weight storage, zero-multiplier SIMD kernels,
//! and BitLinear projection layers for high-speed CPU execution.

mod bit_linear;
mod kernel;
mod weight;

pub use bit_linear::BitLinear;
pub use kernel::{quantize_activations_i8, ternary_dot_product_f32, ternary_dot_product_i8};
pub use weight::TernaryWeight;

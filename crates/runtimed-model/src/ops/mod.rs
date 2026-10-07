//! Mathematical operations: RMSNorm, RoPE, attention, activations, FP8 GEMM, and Marlin INT4 GEMV.

mod activation;
mod attention;
mod fp8_gemm;
mod marlin_gemv;
mod norm;
pub mod ptx_marlin;
mod rope;

pub use activation::{gelu_quick, gelu_tanh, silu, softmax_last};
pub use attention::{attention, causal_mask, tree_attention_mask};
pub use fp8_gemm::{cpu_fp8_gemm_fallback, fp8_gemm};
pub use marlin_gemv::{cpu_marlin_gemv_fallback, marlin_gemv, repack_marlin_tiles, MarlinWeight};
pub use norm::{rms_norm, rms_norm_plain};
pub use rope::{rope_neox, rope_neox_pos, rope_norm};

//! Substrate abstraction boundary: port trait for tensor engines.
//!
//! Defined in accordance with openOODA port boundary requirements:
//! logic modules (GGUF parsing, architecture math, sampler policy,
//! evaluation gates) contain no direct candle imports or raw CUDA kernels.
//! All tensor operations route through `SubstratePort`.

use crate::cache::LayerKv;
use crate::error::Result;
use crate::ops::MarlinWeight;
use candle_core::{Device, Tensor};

/// Sovereign port boundary for tensor execution substrates.
pub trait SubstratePort: Send + Sync {
    /// Dense matrix multiplication: `a * b`.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Root Mean Square Layer Normalization with learned scaling weights.
    fn rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;

    /// Root Mean Square Layer Normalization without scaling weights.
    fn rmsnorm_plain(&self, x: &Tensor, eps: f32) -> Result<Tensor>;

    /// Rotary Positional Embedding (NeoX / standard format).
    fn rope(
        &self,
        x: &Tensor,
        q0: usize,
        theta: f32,
        dim: usize,
        factors: Option<&[f32]>,
    ) -> Result<Tensor>;

    /// Rotary Positional Embedding with Llama-style normalized frequencies.
    fn rope_norm(&self, x: &Tensor, q0: usize, theta: f32, dim: usize) -> Result<Tensor>;

    /// Generate causal triangular attention mask, optionally windowed.
    fn causal_mask(
        &self,
        t: usize,
        total: usize,
        q0: usize,
        sink: Option<usize>,
        dev: &Device,
    ) -> Result<Tensor>;

    /// Scaled Dot-Product Attention over query, key, value tensors.
    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
    ) -> Result<Tensor>;

    /// Allocate a fresh chunked KV cache storage entry.
    fn alloc_kv(&self, k: &Tensor, v: &Tensor, chunk_size: usize) -> Result<LayerKv>;

    /// Compute greedy argmax index along specified dimension.
    fn argmax(&self, tensor: &Tensor, dim: usize) -> Result<u32>;

    /// Extract 1-D probability or logit vector into host memory.
    fn to_vec1(&self, tensor: &Tensor) -> Result<Vec<f32>>;

    /// SiLU (swish) non-linear activation.
    fn silu(&self, x: &Tensor) -> Result<Tensor>;

    /// GeLU tanh approximation activation.
    fn gelu_tanh(&self, x: &Tensor) -> Result<Tensor>;

    /// FP8 scaled GEMM (accelerator kernel or fallback).
    fn fp8_gemm(&self, x: &Tensor, w: &Tensor, scale: &Tensor) -> Result<Tensor>;

    /// Marlin INT4 GEMV (accelerator kernel or fallback).
    fn marlin_gemv(&self, x: &Tensor, w: &MarlinWeight) -> Result<Tensor>;
}

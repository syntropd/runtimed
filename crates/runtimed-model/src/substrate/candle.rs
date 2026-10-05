//! Candle reference implementation of the `SubstratePort`.
//!
//! Confines direct `candle_core` calls, GEMM math, and CUDA kernel invocations
//! behind the sovereign port boundary.

use super::port::SubstratePort;
use crate::cache::LayerKv;
use crate::error::Result;
use crate::ops::{self, MarlinWeight};
use candle_core::{Device, Tensor};

/// Concrete substrate backed by Candle and hardware-accelerated kernels.
#[derive(Debug, Clone, Copy, Default)]
pub struct CandleSubstrate;

pub const DEFAULT_SUBSTRATE: CandleSubstrate = CandleSubstrate;

impl SubstratePort for CandleSubstrate {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.matmul(b).map_err(Into::into)
    }

    fn rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        ops::rms_norm(x, weight, eps)
    }

    fn rmsnorm_plain(&self, x: &Tensor, eps: f32) -> Result<Tensor> {
        ops::rms_norm_plain(x, eps)
    }

    fn rope(
        &self,
        x: &Tensor,
        q0: usize,
        theta: f32,
        dim: usize,
        factors: Option<&[f32]>,
    ) -> Result<Tensor> {
        ops::rope_neox(x, q0, theta, dim, factors)
    }

    fn rope_norm(&self, x: &Tensor, q0: usize, theta: f32, dim: usize) -> Result<Tensor> {
        ops::rope_norm(x, q0, theta, dim)
    }

    fn causal_mask(
        &self,
        t: usize,
        total: usize,
        q0: usize,
        sink: Option<usize>,
        dev: &Device,
    ) -> Result<Tensor> {
        ops::causal_mask(t, total, q0, sink, dev)
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
    ) -> Result<Tensor> {
        ops::attention(q, k, v, mask, scale)
    }

    fn alloc_kv(&self, k: &Tensor, v: &Tensor, chunk_size: usize) -> Result<LayerKv> {
        LayerKv::new(k, v, chunk_size)
    }

    fn argmax(&self, tensor: &Tensor, dim: usize) -> Result<u32> {
        let id = tensor.argmax(dim)?.to_scalar::<u32>()?;
        Ok(id)
    }

    fn to_vec1(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        Ok(tensor.to_vec1::<f32>()?)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        ops::silu(x)
    }

    fn gelu_tanh(&self, x: &Tensor) -> Result<Tensor> {
        ops::gelu_tanh(x)
    }

    fn fp8_gemm(&self, x: &Tensor, w: &Tensor, scale: &Tensor) -> Result<Tensor> {
        ops::fp8_gemm(x, w, scale)
    }

    fn marlin_gemv(&self, x: &Tensor, w: &MarlinWeight) -> Result<Tensor> {
        ops::marlin_gemv(x, w)
    }
}

//! Megatron-style row-parallel linear projection.

use crate::error::{ModelError, Result};
use candle_core::Tensor;

/// Row-parallel linear layer: weights are partitioned along the input dimension.
pub struct RowParallelLinear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub rank: usize,
    pub tp_size: usize,
}

impl RowParallelLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>, rank: usize, tp_size: usize) -> Self {
        Self { weight, bias, rank, tp_size }
    }

    /// Slice a full [out_dim, in_dim] weight matrix for this rank along rows.
    pub fn from_full_weight(
        full_weight: &Tensor,
        full_bias: Option<&Tensor>,
        rank: usize,
        tp_size: usize,
    ) -> Result<Self> {
        if tp_size == 0 {
            return Err(ModelError::Config("TP size cannot be zero".into()));
        }
        let in_dim = full_weight.dim(1)?;
        if in_dim % tp_size != 0 {
            return Err(ModelError::Config(format!(
                "Input dimension {in_dim} not divisible by TP size {tp_size}"
            )));
        }
        let shard_size = in_dim / tp_size;
        let start = rank * shard_size;
        let weight = full_weight.narrow(1, start, shard_size)?.contiguous()?;

        // Only rank 0 adds the bias in row-parallel linear
        let bias = if rank == 0 {
            match full_bias {
                Some(b) => Some(b.contiguous()?),
                None => None,
            }
        } else {
            None
        };

        Ok(Self { weight, bias, rank, tp_size })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let wt = self.weight.t()?;
        let in_dim = wt.dim(0)?;
        let out_dim = wt.dim(1)?;
        let dims = x.dims().to_vec();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let y = x.reshape((rows, in_dim))?.matmul(&wt)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_dim);
        let out = y.reshape(out_shape)?;
        match &self.bias {
            Some(b) => Ok(out.broadcast_add(b)?),
            None => Ok(out),
        }
    }
}

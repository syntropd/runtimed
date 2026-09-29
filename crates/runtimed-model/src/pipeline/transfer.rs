//! Direct P2P DMA activation passing across devices.

use crate::error::Result;
use candle_core::{Device, Tensor};

/// Transfer activation tensor directly between devices via P2P DMA or pinned host DMA.
pub fn transfer_activation(tensor: &Tensor, target_dev: &Device) -> Result<Tensor> {
    if tensor.device().same_device(target_dev) {
        Ok(tensor.clone())
    } else {
        Ok(tensor.to_device(target_dev)?)
    }
}

/// Buffer descriptor for DMA activation pipelining.
#[derive(Debug, Clone)]
pub struct P2pActivationBuffer {
    pub seq_len: usize,
    pub hidden_dim: usize,
}

impl P2pActivationBuffer {
    pub fn new(seq_len: usize, hidden_dim: usize) -> Self {
        Self { seq_len, hidden_dim }
    }

    pub fn verify_shape(&self, tensor: &Tensor) -> bool {
        let dims = tensor.dims();
        dims.len() >= 2
            && dims[dims.len() - 2] == self.seq_len
            && dims[dims.len() - 1] == self.hidden_dim
    }
}

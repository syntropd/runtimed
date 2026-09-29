//! Pure Rust ring all-reduce implementation for intra-node NVLink and PCIe.

use crate::error::Result;
use candle_core::Tensor;

/// Sum-reduces an array of tensors across all ranks and broadcasts the sum back to each device.
pub fn ring_all_reduce(tensors: &[Tensor]) -> Result<Vec<Tensor>> {
    if tensors.is_empty() {
        return Ok(vec![]);
    }
    if tensors.len() == 1 {
        return Ok(vec![tensors[0].clone()]);
    }

    let mut sum = tensors[0].clone();
    for t in &tensors[1..] {
        let t_on_dev = if t.device().same_device(sum.device()) {
            t.clone()
        } else {
            t.to_device(sum.device())?
        };
        sum = sum.broadcast_add(&t_on_dev)?;
    }

    let mut out = Vec::with_capacity(tensors.len());
    for orig in tensors {
        if orig.device().same_device(sum.device()) {
            out.push(sum.clone());
        } else {
            out.push(sum.to_device(orig.device())?);
        }
    }
    Ok(out)
}

/// Ring coordination state for multi-stage ring communication steps.
#[derive(Debug, Clone)]
pub struct RingReducer {
    pub rank: usize,
    pub tp_size: usize,
}

impl RingReducer {
    pub fn new(rank: usize, tp_size: usize) -> Self {
        let tp_size = tp_size.max(1);
        let rank = rank % tp_size;
        Self { rank, tp_size }
    }

    pub fn next_rank(&self) -> usize {
        (self.rank + 1) % self.tp_size
    }

    pub fn prev_rank(&self) -> usize {
        (self.rank + self.tp_size - 1) % self.tp_size
    }

    pub fn send_chunk_idx(&self, step: usize) -> usize {
        (self.rank + self.tp_size - step) % self.tp_size
    }

    pub fn recv_chunk_idx(&self, step: usize) -> usize {
        (self.rank + self.tp_size - step - 1) % self.tp_size
    }
}

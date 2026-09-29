//! Pipeline stage abstraction for multi-accelerator partitioning.

use crate::weights::Weights;
use candle_core::Device;
use std::sync::Arc;

/// A discrete pipeline stage managing weights and compute on one device.
#[derive(Clone)]
pub struct PipelineStage {
    pub stage_idx: usize,
    pub total_stages: usize,
    pub start_layer: usize,
    pub end_layer: usize,
    pub device: Device,
    pub weights: Arc<Weights>,
    pub is_first: bool,
    pub is_last: bool,
}

impl PipelineStage {
    pub fn new(
        stage_idx: usize,
        total_stages: usize,
        start_layer: usize,
        end_layer: usize,
        total_layers: usize,
        device: Device,
        weights: Arc<Weights>,
    ) -> Self {
        let is_first = stage_idx == 0 || start_layer == 0;
        let is_last = stage_idx == total_stages.saturating_sub(1) || end_layer >= total_layers;
        Self {
            stage_idx,
            total_stages,
            start_layer,
            end_layer,
            device,
            weights,
            is_first,
            is_last,
        }
    }

    pub fn layer_count(&self) -> usize {
        self.end_layer.saturating_sub(self.start_layer)
    }

    pub fn resident_bytes(&self) -> usize {
        self.weights.resident_bytes()
    }
}

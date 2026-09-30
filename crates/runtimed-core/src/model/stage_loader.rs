//! Multi-device stage loader slicing layer ranges from sealed memfd CAS.

use crate::error::Result;
use candle_core::Device;
use runtimed_gguf::GgufFile;
use runtimed_model::config::ArchConfig;
use runtimed_model::pipeline::PipelineStage;
use runtimed_model::weights::Weights;
use std::fs::File;
use std::sync::Arc;

/// A partitioning slice assigning a layer range to a specific compute device.
#[derive(Debug, Clone)]
pub struct StagePartition {
    pub device: Device,
    pub start_layer: usize,
    pub end_layer: usize,
}

pub struct StageLoader;

impl StageLoader {
    /// Partition layers evenly across a set of target devices.
    pub fn partition_layers(total_layers: usize, devices: &[Device]) -> Vec<StagePartition> {
        let num_stages = devices.len();
        if num_stages == 0 {
            return vec![];
        }
        let base = total_layers / num_stages;
        let remainder = total_layers % num_stages;
        let mut partitions = Vec::with_capacity(num_stages);
        let mut current = 0;
        for (i, dev) in devices.iter().enumerate() {
            let count = base + if i < remainder { 1 } else { 0 };
            let end = current + count;
            partitions.push(StagePartition {
                device: dev.clone(),
                start_layer: current,
                end_layer: end,
            });
            current = end;
        }
        partitions
    }

    /// Partition layers asymmetrically across devices proportional to their available VRAM.
    /// L_p = round(L * V_p / sum(V))
    pub fn asymmetric_partition_layers(
        total_layers: usize,
        device_vrams: &[(Device, u64)],
    ) -> Vec<StagePartition> {
        if device_vrams.is_empty() || total_layers == 0 {
            return vec![];
        }
        let total_vram: u64 = device_vrams.iter().map(|(_, v)| *v).sum();
        if total_vram == 0 {
            let devs: Vec<Device> = device_vrams.iter().map(|(d, _)| d.clone()).collect();
            return Self::partition_layers(total_layers, &devs);
        }

        let mut partitions = Vec::with_capacity(device_vrams.len());
        let mut current_layer = 0;
        let mut cum_vram = 0u64;

        for (i, (dev, vram)) in device_vrams.iter().enumerate() {
            cum_vram += *vram;
            let target_end = if i == device_vrams.len() - 1 {
                total_layers
            } else {
                ((total_layers as f64 * cum_vram as f64) / total_vram as f64).round() as usize
            };
            let end_layer = target_end.min(total_layers).max(current_layer);
            partitions.push(StagePartition {
                device: dev.clone(),
                start_layer: current_layer,
                end_layer,
            });
            current_layer = end_layer;
        }

        partitions
    }

    /// Load pipeline stages directly from a sealed memfd file handle.
    pub fn load_stages_from_file(
        file: &File,
        partitions: &[StagePartition],
        total_layers: usize,
    ) -> Result<(Vec<PipelineStage>, ArchConfig)> {
        let gguf = GgufFile::from_file(file).map_err(|e| crate::error::RuntimedError::GenerationFailed(e.to_string()))?;
        let cfg = ArchConfig::parse(&gguf).map_err(|e| crate::error::RuntimedError::GenerationFailed(e.to_string()))?;
        let mut stages = Vec::with_capacity(partitions.len());
        for (idx, p) in partitions.iter().enumerate() {
            let w = Weights::load_stage_range(&gguf, &p.device, p.start_layer, p.end_layer, total_layers)
                .map_err(|e| crate::error::RuntimedError::GenerationFailed(e.to_string()))?;
            stages.push(PipelineStage::new(
                idx,
                partitions.len(),
                p.start_layer,
                p.end_layer,
                total_layers,
                p.device.clone(),
                Arc::new(w),
            ));
        }
        Ok((stages, cfg))
    }

    /// Build hybrid GPU + Host CPU system RAM execution stages from a sealed memfd.
    pub fn load_hybrid_stages(
        file: &File,
        gpu_dev: &Device,
        gpu_layers: usize,
    ) -> Result<(Vec<PipelineStage>, ArchConfig)> {
        let gguf = GgufFile::from_file(file).map_err(|e| crate::error::RuntimedError::GenerationFailed(e.to_string()))?;
        let cfg = ArchConfig::parse(&gguf).map_err(|e| crate::error::RuntimedError::GenerationFailed(e.to_string()))?;
        let total_layers = cfg.n_layer;
        let split_layer = gpu_layers.min(total_layers);

        let mut partitions = Vec::new();
        if split_layer > 0 {
            partitions.push(StagePartition {
                device: gpu_dev.clone(),
                start_layer: 0,
                end_layer: split_layer,
            });
        }
        if split_layer < total_layers {
            partitions.push(StagePartition {
                device: Device::Cpu,
                start_layer: split_layer,
                end_layer: total_layers,
            });
        }

        let mut stages = Vec::with_capacity(partitions.len());
        for (idx, p) in partitions.iter().enumerate() {
            let w = Weights::load_stage_range(&gguf, &p.device, p.start_layer, p.end_layer, total_layers)
                .map_err(|e| crate::error::RuntimedError::GenerationFailed(e.to_string()))?;
            stages.push(PipelineStage::new(
                idx,
                partitions.len(),
                p.start_layer,
                p.end_layer,
                total_layers,
                p.device.clone(),
                Arc::new(w),
            ));
        }
        Ok((stages, cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_layers_even_and_odd() {
        let devs = vec![Device::Cpu, Device::Cpu];
        let parts = StageLoader::partition_layers(8, &devs);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].start_layer, 0);
        assert_eq!(parts[0].end_layer, 4);
        assert_eq!(parts[1].start_layer, 4);
        assert_eq!(parts[1].end_layer, 8);

        let parts_odd = StageLoader::partition_layers(7, &devs);
        assert_eq!(parts_odd[0].end_layer, 4);
        assert_eq!(parts_odd[1].end_layer, 7);
    }

    #[test]
    fn test_asymmetric_partition_layers() {
        // 32 layers across 24 GB GPU and 8 GB GPU -> 24 layers and 8 layers
        let dev_vrams = vec![(Device::Cpu, 24 * 1024 * 1024 * 1024), (Device::Cpu, 8 * 1024 * 1024 * 1024)];
        let parts = StageLoader::asymmetric_partition_layers(32, &dev_vrams);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].start_layer, 0);
        assert_eq!(parts[0].end_layer, 24);
        assert_eq!(parts[1].start_layer, 24);
        assert_eq!(parts[1].end_layer, 32);
    }
}

//! GGUF weight loading into the Weights store.

use super::store::Weights;
use crate::error::Result;
use candle_core::{DType, Device, Tensor};
use runtimed_gguf::GgufFile;
use std::collections::HashMap;

impl Weights {
    pub fn load(file: &GgufFile, dev: &Device) -> Result<Self> {
        Self::load_filtered(file, dev, |_| true)
    }

    /// Load only tensors whose name passes `keep` (e.g. vision-only from
    /// a multimodal projector that also carries audio weights).
    pub fn load_filtered(
        file: &GgufFile,
        dev: &Device,
        keep: impl Fn(&str) -> bool,
    ) -> Result<Self> {
        Self::ensure_current(dev)?;
        let store = match dev {
            Device::Cuda(_) => DType::F16,
            _ => DType::F32,
        };
        let total_params: u64 = file.tensors.iter().map(|t| t.n_elements as u64).sum();
        let split_cuda = matches!(dev, Device::Cuda(_))
            && Self::cuda_available(1)
            && total_params > 8_000_000_000;
        let sec_dev = if split_cuda { Device::new_cuda(1).ok() } else { None };

        let max_layer = file.tensors.iter().filter_map(|t| {
            t.name.strip_prefix("blk.")?.split('.').next()?.parse::<usize>().ok()
        }).max().map(|m| m + 1).unwrap_or(0);
        let split_layer = max_layer / 2;

        let mut map = HashMap::with_capacity(file.tensors.len());
        for info in &file.tensors {
            if !keep(&info.name) {
                continue;
            }
            let target_dev = if let Some(ref s_dev) = sec_dev {
                let is_second = if let Some(rest) = info.name.strip_prefix("blk.") {
                    rest.split('.').next().and_then(|s| s.parse::<usize>().ok()).map(|l| l >= split_layer).unwrap_or(false)
                } else {
                    info.name.starts_with("output_norm.") || info.name == "output.weight"
                };
                if is_second { s_dev } else { dev }
            } else {
                dev
            };
            Self::ensure_current(target_dev)?;
            let data = file.tensor_f32(info)?;
            let shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
            let t = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?;
            let t = if store == DType::F32 { t } else { t.to_dtype(store)? };
            let t = t.to_device(target_dev)?;
            map.insert(info.name.clone(), t);
        }
        Ok(Self::from_parts(dev.clone(), store, map))
    }

    /// Load only weights for a layer range [start_layer, end_layer).
    /// If start_layer == 0, loads token embeddings.
    /// If end_layer >= total_layers, loads output norm and output head.
    /// Supports tied embeddings (e.g. Gemma4 where output.weight is tied to token_embd.weight).
    pub fn load_stage_range(
        file: &GgufFile,
        dev: &Device,
        start_layer: usize,
        end_layer: usize,
        total_layers: usize,
    ) -> Result<Self> {
        let is_first = start_layer == 0;
        let is_last = end_layer >= total_layers;
        let mut weights = Self::load_filtered(file, dev, |name| {
            if is_first && (name == "token_embd.weight" || name.starts_with("token_embd.")) {
                return true;
            }
            if is_last && (name == "output_norm.weight" || name == "output.weight" || name.starts_with("output_norm.")) {
                return true;
            }
            if let Some(rest) = name.strip_prefix("blk.") {
                if let Some(idx_str) = rest.split('.').next() {
                    if let Ok(layer_idx) = idx_str.parse::<usize>() {
                        return layer_idx >= start_layer && layer_idx < end_layer;
                    }
                }
            }
            false
        })?;

        // Tied embeddings support (Gemma4): if last stage lacks output.weight, reuse token_embd.weight
        if is_last && !weights.map.contains_key("output.weight") {
            if let Some(embd) = weights.map.get("token_embd.weight") {
                weights.map.insert("output.weight".to_string(), embd.clone());
            } else if let Some(embd_info) = file.tensors.iter().find(|t| t.name == "token_embd.weight") {
                let data = file.tensor_f32(embd_info)?;
                let shape: Vec<usize> = embd_info.dims.iter().rev().map(|&d| d as usize).collect();
                let t = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?;
                let store = match dev {
                    Device::Cuda(_) => DType::F16,
                    _ => DType::F32,
                };
                let t = if store == DType::F32 { t } else { t.to_dtype(store)? };
                let t = t.to_device(dev)?;
                weights.map.insert("output.weight".to_string(), t);
            }
        }
        Ok(weights)
    }
}

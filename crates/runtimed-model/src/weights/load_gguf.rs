//! GGUF weight loading into the Weights store.

use super::store::Weights;
use crate::error::Result;
use candle_core::quantized::{ggml_file, QMatMul};
use candle_core::{DType, Device, Tensor};
use runtimed_gguf::{GgmlDtype, GgufFile, TensorInfo};
use std::collections::HashMap;
use std::sync::Arc;

fn to_candle_quantized_dtype(dtype: GgmlDtype) -> Option<candle_core::quantized::GgmlDType> {
    use candle_core::quantized::GgmlDType as C;
    match dtype {
        GgmlDtype::Q4_0 => Some(C::Q4_0),
        GgmlDtype::Q4_1 => Some(C::Q4_1),
        GgmlDtype::Q5_0 => Some(C::Q5_0),
        GgmlDtype::Q5_1 => Some(C::Q5_1),
        GgmlDtype::Q8_0 => Some(C::Q8_0),
        GgmlDtype::Q2K => Some(C::Q2K),
        GgmlDtype::Q3K => Some(C::Q3K),
        GgmlDtype::Q4K => Some(C::Q4K),
        GgmlDtype::Q5K => Some(C::Q5K),
        GgmlDtype::Q6K => Some(C::Q6K),
        _ => None,
    }
}

fn load_quantized_tensor(
    file: &GgufFile,
    info: &TensorInfo,
    candle_dtype: candle_core::quantized::GgmlDType,
    dev: &Device,
) -> Result<Arc<QMatMul>> {
    let raw_bytes = file.tensor_bytes(info)?;
    let shape = vec![info.dims[1] as usize, info.dims[0] as usize];
    let qtensor = ggml_file::qtensor_from_ggml(candle_dtype, raw_bytes, shape, dev)?;
    Ok(Arc::new(QMatMul::from_arc(Arc::new(qtensor))?))
}

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
        let mut q_map = HashMap::with_capacity(file.tensors.len());
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

            if info.dims.len() == 2 {
                if let Some(candle_dtype) = to_candle_quantized_dtype(info.dtype) {
                    let qm = load_quantized_tensor(file, info, candle_dtype, target_dev)?;
                    if info.name == "token_embd.weight" && file.tensor("output.weight").is_none() {
                        if let Some(ref s_dev) = sec_dev {
                            Self::ensure_current(s_dev)?;
                            let qm_out = load_quantized_tensor(file, info, candle_dtype, s_dev)?;
                            q_map.insert("output.weight".to_string(), qm_out);
                        } else {
                            q_map.insert("output.weight".to_string(), qm.clone());
                        }
                    }
                    q_map.insert(info.name.clone(), qm);
                    continue;
                }
            }

            let data = file.tensor_f32(info)?;
            let shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
            let t = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?;
            let t = if store == DType::F32 { t } else { t.to_dtype(store)? };
            if info.name == "token_embd.weight" && file.tensor("output.weight").is_none() {
                if let Some(ref s_dev) = sec_dev {
                    Self::ensure_current(s_dev)?;
                    let t_out = t.to_device(s_dev)?;
                    map.insert("output.weight".to_string(), t_out);
                } else {
                    map.insert("output.weight".to_string(), t.clone());
                }
            }
            let t = t.to_device(target_dev)?;
            map.insert(info.name.clone(), t);
        }
        Ok(Self::from_quantized(dev.clone(), store, map, q_map))
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
        if is_last && !weights.contains_key("output.weight") {
            if let Some(qm) = weights.q_map.get("token_embd.weight") {
                weights.q_map.insert("output.weight".to_string(), qm.clone());
            } else if let Some(embd) = weights.map.get("token_embd.weight") {
                weights.map.insert("output.weight".to_string(), embd.clone());
            } else if let Some(embd_info) = file.tensors.iter().find(|t| t.name == "token_embd.weight") {
                Self::ensure_current(dev)?;
                if embd_info.dims.len() == 2 {
                    if let Some(candle_dtype) = to_candle_quantized_dtype(embd_info.dtype) {
                        let qm = load_quantized_tensor(file, embd_info, candle_dtype, dev)?;
                        weights.q_map.insert("output.weight".to_string(), qm);
                    } else {
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
            }
        }
        Ok(weights)
    }
}

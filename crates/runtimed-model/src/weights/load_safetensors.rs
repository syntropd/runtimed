//! Safetensors weight loading into the Weights store with mmap and FP8 dequantization.

use super::name_map::map_hf_tensor_name;
use super::store::Weights;
use crate::error::{ModelError, Result};
use candle_core::{DType, Device, Tensor};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::LazyLock;

pub static FP8_E4M3_LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let mut lut = [0.0f32; 256];
    for b in 0..=255u8 {
        let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((b >> 3) & 0x0f) as i32;
        let mant = (b & 0x07) as f32;
        lut[b as usize] = if exp == 0 {
            sign * 2.0f32.powi(-6) * (mant / 8.0)
        } else if exp == 15 && (b & 0x07) == 0x07 {
            f32::NAN
        } else {
            sign * 2.0f32.powi(exp - 7) * (1.0 + mant / 8.0)
        };
    }
    lut
});

pub static FP8_E5M2_LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let mut lut = [0.0f32; 256];
    for b in 0..=255u8 {
        let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((b >> 2) & 0x1f) as i32;
        let mant = (b & 0x03) as f32;
        lut[b as usize] = if exp == 0 {
            sign * 2.0f32.powi(-14) * (mant / 4.0)
        } else if exp == 31 {
            if mant == 0.0 { sign * f32::INFINITY } else { f32::NAN }
        } else {
            sign * 2.0f32.powi(exp - 15) * (1.0 + mant / 4.0)
        };
    }
    lut
});

pub fn dequantize_fp8_e4m3(bytes: &[u8]) -> Vec<f32> {
    let lut = &*FP8_E4M3_LUT;
    bytes.iter().map(|&b| lut[b as usize]).collect()
}

pub fn dequantize_fp8_e5m2(bytes: &[u8]) -> Vec<f32> {
    let lut = &*FP8_E5M2_LUT;
    bytes.iter().map(|&b| lut[b as usize]).collect()
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f16_to_f32(bits: u16) -> f32 {
    runtimed_gguf::quant::dequant::f16_to_f32(bits)
}

impl Weights {
    pub fn load_safetensors(path: &Path, dev: &Device) -> Result<Self> {
        Self::load_safetensors_filtered(path, dev, |_| true)
    }

    pub fn load_safetensors_from_file(file: &File, dev: &Device) -> Result<Self> {
        let mmap = unsafe {
            MmapOptions::new()
                .map(file)
                .map_err(|e| ModelError::Config(format!("mmap failed: {e}")))?
        };
        Self::load_safetensors_from_bytes(&mmap, dev, |_| true)
    }

    pub fn load_safetensors_filtered(
        path: &Path,
        dev: &Device,
        keep: impl Fn(&str) -> bool,
    ) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| ModelError::Config(format!("mmap failed: {e}")))?
        };
        Self::load_safetensors_from_bytes(&mmap, dev, keep)
    }

    pub fn load_safetensors_from_bytes(
        bytes: &[u8],
        dev: &Device,
        keep: impl Fn(&str) -> bool,
    ) -> Result<Self> {
        Self::ensure_current(dev)?;
        let store = match dev {
            Device::Cuda(_) => DType::F16,
            _ => DType::F32,
        };
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| ModelError::Config(format!("safetensors parse: {e}")))?;
        let tensors = st.tensors();
        let mut map = HashMap::with_capacity(tensors.len());

        for (name, view) in tensors {
            let mapped = map_hf_tensor_name(&name);
            if !keep(&mapped) && !keep(&name) {
                continue;
            }
            let data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => view
                    .data()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::F16 => view
                    .data()
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                safetensors::Dtype::BF16 => view
                    .data()
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                safetensors::Dtype::F8_E4M3 => dequantize_fp8_e4m3(view.data()),
                safetensors::Dtype::F8_E5M2 => dequantize_fp8_e5m2(view.data()),
                other => {
                    return Err(ModelError::Config(format!(
                        "unsupported dtype for {name}: {other:?}"
                    )));
                }
            };
            let shape = view.shape();
            let t = Tensor::from_vec(data, shape, &Device::Cpu)?;
            let t = if store == DType::F32 { t } else { t.to_dtype(store)? };
            let t = t.to_device(dev)?;
            map.insert(mapped, t);
        }

        if !map.contains_key("output.weight") {
            if let Some(embd) = map.get("token_embd.weight") {
                map.insert("output.weight".to_string(), embd.clone());
            }
        }

        Ok(Self::from_parts(dev.clone(), store, map))
    }

    pub fn load_stage_range_safetensors(
        path: &Path,
        dev: &Device,
        start_layer: usize,
        end_layer: usize,
        total_layers: usize,
    ) -> Result<Self> {
        let is_first = start_layer == 0;
        let is_last = end_layer >= total_layers;
        let mut weights = Self::load_safetensors_filtered(path, dev, |name| {
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

        if is_last && !weights.map.contains_key("output.weight") {
            if let Some(embd) = weights.map.get("token_embd.weight") {
                weights.map.insert("output.weight".to_string(), embd.clone());
            }
        }
        Ok(weights)
    }
}

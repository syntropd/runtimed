//! LoRA adapter loading from Hugging Face / PEFT Safetensors files.

use super::types::{LoraAdapter, LoraPair};
use crate::error::{ModelError, Result};
use crate::weights::{map_hf_tensor_name, Weights};
use candle_core::{Device, Tensor};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::Path;

pub fn load_safetensors_adapter(path: &Path, dev: &Device, arch: &str) -> Result<LoraAdapter> {
    Weights::ensure_current(dev)?;
    let (weights_path, config_path) = resolve_adapter_paths(path)?;
    let cfg_str = fs::read_to_string(&config_path)
        .map_err(|e| ModelError::Config(format!("adapter config {}: {e}", config_path.display())))?;
    let cfg_val: Value = serde_json::from_str(&cfg_str)
        .map_err(|e| ModelError::Config(format!("invalid adapter json: {e}")))?;

    if let Some(base_arch) = cfg_val.get("base_model_architecture").and_then(Value::as_str) {
        if !base_arch.to_lowercase().contains(arch) {
            return Err(ModelError::Config(format!(
                "LoRA arch mismatch: adapter is '{base_arch}', model is '{arch}'"
            )));
        }
    }

    let alpha = cfg_val
        .get("lora_alpha")
        .and_then(Value::as_f64)
        .map(|a| a as f32)
        .ok_or_else(|| ModelError::Config("missing lora_alpha in adapter_config.json".into()))?;

    let file = File::open(&weights_path)?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| ModelError::Config(format!("mmap {}: {e}", weights_path.display())))?
    };
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| ModelError::Config(format!("safetensors deserialize: {e}")))?;

    let mut a_map = HashMap::new();
    let mut b_map = HashMap::new();

    for (name, view) in st.tensors() {
        let name_lower = name.to_lowercase();
        if let Some(base) = strip_lora_suffix(&name, &name_lower, "lora_a") {
            a_map.insert(base, view);
        } else if let Some(base) = strip_lora_suffix(&name, &name_lower, "lora_b") {
            b_map.insert(base, view);
        }
    }

    if a_map.len() != b_map.len() || a_map.keys().any(|k| !b_map.contains_key(k)) {
        return Err(ModelError::Config("LoRA file has unpaired A/B tensors".into()));
    }

    let mut pairs = Vec::with_capacity(a_map.len());
    let mut scale: Option<f64> = None;

    for (base, a_view) in &a_map {
        let b_view = &b_map[base];
        let a = decode_st_tensor(a_view, dev)?;
        let b = decode_st_tensor(b_view, dev)?;

        if a.dim(0)? != b.dim(1)? {
            return Err(ModelError::Config(format!("LoRA rank mismatch on '{base}'")));
        }
        let s = LoraAdapter::scale_of(alpha, &b)?;
        if let Some(prev) = scale {
            if (prev - s).abs() > 1e-9f64 {
                return Err(ModelError::Config("LoRA pairs disagree on rank".into()));
            }
        }
        scale = Some(s);
        let mapped_base = map_hf_tensor_name(base);
        pairs.push(LoraPair { base: mapped_base, a, b });
    }

    if pairs.is_empty() {
        return Err(ModelError::Config("LoRA safetensors has no A/B pairs".into()));
    }
    let final_scale = scale.ok_or_else(|| ModelError::Config("no valid LoRA pairs".into()))?;
    Ok(LoraAdapter::from_parts(pairs, final_scale))
}

fn strip_lora_suffix(name: &str, lower: &str, side: &str) -> Option<String> {
    let suffix1 = format!(".{side}.weight");
    let suffix2 = format!(".{side}");
    if lower.ends_mut(&suffix1) {
        let len = name.len() - suffix1.len();
        Some(name[..len].to_string())
    } else if lower.ends_mut(&suffix2) {
        let len = name.len() - suffix2.len();
        Some(name[..len].to_string())
    } else {
        None
    }
}

trait EndsMutExt {
    fn ends_mut(&self, suffix: &str) -> bool;
}

impl EndsMutExt for str {
    fn ends_mut(&self, suffix: &str) -> bool {
        self.ends_with(suffix)
    }
}

fn resolve_adapter_paths(path: &Path) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    if path.is_dir() {
        let w = path.join("adapter_model.safetensors");
        let c = path.join("adapter_config.json");
        if w.exists() && c.exists() {
            return Ok((w, c));
        }
    }
    let w = path.to_path_buf();
    let c = path.with_file_name("adapter_config.json");
    if c.exists() {
        return Ok((w, c));
    }
    let c2 = path.with_extension("config.json");
    if c2.exists() {
        return Ok((w, c2));
    }
    Err(ModelError::Config(format!(
        "could not find companion adapter_config.json for {}",
        path.display()
    )))
}

fn decode_st_tensor(view: &safetensors::tensor::TensorView<'_>, dev: &Device) -> Result<Tensor> {
    let f32_data: Vec<f32> = match view.dtype() {
        safetensors::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        safetensors::Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|c| runtimed_gguf::quant::dequant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        safetensors::Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        other => return Err(ModelError::Config(format!("unsupported LoRA dtype: {other:?}"))),
    };
    let shape = view.shape();
    Ok(Tensor::from_vec(f32_data, shape, dev)?)
}

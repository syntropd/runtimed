//! LoRA adapter loading from GGUF files.

use super::types::{LoraAdapter, LoraPair};
use crate::error::{ModelError, Result};
use crate::weights::Weights;
use candle_core::{Device, Tensor};
use runtimed_gguf::{GgufFile, MetaValue};
use std::collections::HashMap;
use std::path::Path;

pub fn load_gguf_adapter(path: &Path, dev: &Device, arch: &str) -> Result<LoraAdapter> {
    Weights::ensure_current(dev)?;
    let file = GgufFile::open(path)?;
    let str_meta = |key: &str| match file.metadata.get(key) {
        Some(MetaValue::Str(s)) => s.as_str(),
        _ => "",
    };
    if str_meta("general.type") != "adapter" {
        return Err(ModelError::Config("LoRA file needs general.type=adapter".into()));
    }
    if str_meta("general.architecture") != arch {
        return Err(ModelError::Config(format!(
            "LoRA arch mismatch: adapter is '{}', model is '{arch}'",
            str_meta("general.architecture")
        )));
    }
    if str_meta("adapter.type") != "lora" {
        return Err(ModelError::Config("only adapter.type=lora is supported".into()));
    }
    let alpha = match file.metadata.get("adapter.lora.alpha") {
        Some(MetaValue::F32(v)) => *v,
        _ => return Err(ModelError::Config("LoRA file needs adapter.lora.alpha".into())),
    };
    // Pair A/B tensors by base name.
    let mut a_map = HashMap::new();
    let mut b_map = HashMap::new();
    for info in &file.tensors {
        if let Some(base) = info.name.strip_suffix(".lora_a") {
            a_map.insert(base.to_string(), info);
        } else if let Some(base) = info.name.strip_suffix(".lora_b") {
            b_map.insert(base.to_string(), info);
        } else if info.name.ends_with("_norm.weight") {
            continue; // like the reference: norms ride along unapplied
        } else {
            return Err(ModelError::Config(format!("unexpected LoRA tensor '{}'", info.name)));
        }
    }
    if a_map.len() != b_map.len() || a_map.keys().any(|k| !b_map.contains_key(k)) {
        return Err(ModelError::Config("LoRA file has unpaired A/B tensors".into()));
    }
    let mut pairs = Vec::with_capacity(a_map.len());
    let mut scale: Option<f64> = None;
    for (base, a_info) in &a_map {
        if base.contains("token_embd") || base.contains("output.weight") {
            return Err(ModelError::Config(format!("embedding LoRA '{base}' is not supported (linears only)")));
        }
        let b_info = &b_map[base.as_str()];
        let a = load_f32(&file, a_info, dev)?;
        let b = load_f32(&file, b_info, dev)?;
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
        pairs.push(LoraPair { base: base.clone(), a, b });
    }
    if pairs.is_empty() {
        return Err(ModelError::Config("LoRA file has no A/B pairs".into()));
    }
    let final_scale = scale.ok_or_else(|| ModelError::Config("no valid LoRA pairs".into()))?;
    Ok(LoraAdapter::from_parts(pairs, final_scale))
}

fn load_f32(file: &GgufFile, info: &runtimed_gguf::TensorInfo, dev: &Device) -> Result<Tensor> {
    let data = file.tensor_f32(info)?;
    let shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
    Ok(Tensor::from_vec(data, shape.as_slice(), dev)?)
}

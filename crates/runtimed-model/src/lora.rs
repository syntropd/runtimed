//! LoRA adapters (GGUF `general.type=adapter` files), fused at load.
//!
//! An adapter carries `<base>.lora_a` (`[rank, in]`) and `<base>.lora_b`
//! (`[out, rank]`) pairs plus `adapter.lora.alpha`. Fusion rewrites each
//! base weight in place: `W += (alpha / rank) * B * A`, which is exactly
//! what runtime application computes (`Wx + s*B*(Ax)`), up to float
//! association order. The base file on disk is never touched; re-loading
//! the model drops the adapter.
//!
//! Scope: attention + FFN linears. Embedding adapters and norm vectors
//! are rejected (norms) or skipped (norms, like the reference).

use crate::error::{ModelError, Result};
use crate::weights::Weights;
use candle_core::{Device, Tensor};
use runtimed_gguf::{GgufFile, MetaValue};

/// One LoRA pair: base weight name, `A [rank, in]`, `B [out, rank]`.
pub struct LoraPair {
    pub base: String,
    pub a: Tensor,
    pub b: Tensor,
}

/// A loaded adapter: pairs plus the shared scale.
pub struct LoraAdapter {
    pairs: Vec<LoraPair>,
    scale: f64,
}

impl LoraAdapter {
    /// Rank inferred from `B`'s dim 0... in our row-major layout `B` is
    /// `[out, rank]`, so rank is dim 1; scale is `alpha / rank`.
    fn scale_of(alpha: f32, b: &Tensor) -> Result<f64> {
        let rank = b.dim(1)? as f64;
        if rank < 1.0 {
            return Err(ModelError::Config("lora_b has zero rank".into()));
        }
        Ok(if alpha == 0.0 { 1.0 } else { alpha as f64 / rank })
    }

    /// Load + validate an adapter file. `arch` must match the base model
    /// (e.g. `"gemma4"`, `"qwen2"`).
    pub fn load(path: &std::path::Path, dev: &Device, arch: &str) -> Result<Self> {
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
        let mut a_map = std::collections::HashMap::new();
        let mut b_map = std::collections::HashMap::new();
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
            let s = Self::scale_of(alpha, &b)?;
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
        Ok(Self { pairs, scale: scale.unwrap() })
    }

    /// Fuse every pair into `weights`; returns the fused base names.
    pub fn fuse_into(&self, weights: &mut Weights) -> Result<Vec<String>> {
        let mut fused = Vec::with_capacity(self.pairs.len());
        for p in &self.pairs {
            let w = weights.get(&p.base)?;
            let (out_dim, in_dim) = (w.dim(0)?, w.dim(1)?);
            if p.b.dim(0)? != out_dim || p.a.dim(1)? != in_dim {
                return Err(ModelError::Shape {
                    name: p.base.clone(),
                    expected: vec![out_dim, in_dim],
                    got: vec![p.b.dim(0)?, p.a.dim(1)?],
                });
            }
            let delta = p.b.matmul(&p.a)?.affine(self.scale, 0.0)?;
            let updated = (w + delta)?;
            weights.replace(&p.base, updated)?;
            fused.push(p.base.clone());
        }
        Ok(fused)
    }
}

/// Dequant one tensor to F32 (adapters are usually F32/F16 already).
fn load_f32(file: &GgufFile, info: &runtimed_gguf::TensorInfo, dev: &Device) -> Result<Tensor> {
    let data = file.tensor_f32(info)?;
    let shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
    Ok(Tensor::from_vec(data, shape.as_slice(), dev)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_matches_runtime_application() {
        let dev = Device::Cpu;
        // W [[1,2],[3,4]], A [[1,0,1],[0,1,1]] (r=2,in=3)... use 2x2: A=[[1,2],[3,4]], B=[[1,0],[0,1]], scale 0.5.
        let w = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &dev).unwrap();
        let a = Tensor::from_vec(vec![1f32, 0., 0., 1.], (2, 2), &dev).unwrap();
        let b = Tensor::from_vec(vec![2f32, 0., 0., 2.], (2, 2), &dev).unwrap();
        // Fused: W + 0.5*B*A = [[1,2],[3,4]] + [[1,0],[0,1]] = [[2,2],[3,5]].
        let delta = b.matmul(&a).unwrap().affine(0.5, 0.0).unwrap();
        let fused = (&w + delta).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(fused, vec![2.0, 2.0, 3.0, 5.0]);
        // Runtime: (W + sBA)x == Wx + sB(Ax) for x=[1,1].
        let x = Tensor::from_vec(vec![1f32, 1.], (2, 1), &dev).unwrap();
        let lhs = w.matmul(&x).unwrap();
        let ax = a.matmul(&x).unwrap();
        let rhs = b.matmul(&ax).unwrap().affine(0.5, 0.0).unwrap();
        let rt = (lhs + rhs).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(rt, vec![4.0, 8.0]);
    }

    #[test]
    fn zero_alpha_means_scale_one() {
        let dev = Device::Cpu;
        let b = Tensor::zeros((4, 8), candle_core::DType::F32, &dev).unwrap();
        assert_eq!(LoraAdapter::scale_of(0.0, &b).unwrap(), 1.0);
        assert_eq!(LoraAdapter::scale_of(16.0, &b).unwrap(), 2.0);
    }
}

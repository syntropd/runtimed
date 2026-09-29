//! Weight store: every GGUF tensor decoded to an f32 candle tensor.
//!
//! GGUF dims list the fastest-varying axis first, so the candle shape is
//! the dims reversed: a `[in, out]` weight becomes `(out, in)` row-major.

use crate::error::{ModelError, Result};
use candle_core::{DType, Device, Tensor};
use runtimed_gguf::GgufFile;
use std::collections::HashMap;

#[derive(Clone)]
pub struct Weights {
    dev: Device,
    /// Resident dtype: F16 on CUDA (a 5B-param model is ~21 GB as F32,
    /// ~10.5 GB as F16), F32 on CPU where f16 matmul is unsupported.
    /// Compute always runs in F32; `get` upcasts on the way out.
    store: DType,
    map: HashMap<String, Tensor>,
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
        let mut map = HashMap::with_capacity(file.tensors.len());
        for info in &file.tensors {
            if !keep(&info.name) {
                continue;
            }
            let data = file.tensor_f32(info)?;
            let shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
            // Cast on the CPU side so the GPU never holds a transient F32
            // copy: peak device memory stays near the resident total.
            // The CPU path skips the cast (same dtype, zero extra copy).
            let t = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?;
            let t = if store == DType::F32 { t } else { t.to_dtype(store)? };
            let t = t.to_device(dev)?;
            map.insert(info.name.clone(), t);
        }
        Ok(Self { dev: dev.clone(), store, map })
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

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// True resident footprint: element count times resident dtype width
    /// (F16 on CUDA, F32 on CPU), so load reports match `nvidia-smi`.
    pub fn resident_bytes(&self) -> usize {
        self.map
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }

    /// Owned F32 view of a resident weight. On CPU the clone is a
    /// cheap handle bump (already F32); on CUDA it upcasts, so all
    /// compute stays F32-exact regardless of resident dtype.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        let t = self
            .map
            .get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
        if t.dtype() == DType::F32 {
            Ok(t.clone())
        } else {
            Ok(t.to_dtype(DType::F32)?)
        }
    }

    /// `[out, in]` matrix product over the last axis: `x @ w.t()`.
    /// Leading dims flatten through the matmul (candle needs equal ranks).
    pub fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.get(name)?;
        let wt = w.t()?;
        let in_dim = wt.dim(0)?;
        let out_dim = wt.dim(1)?;
        let dims = x.dims().to_vec();
        let Some(last) = dims.last() else {
            return Err(ModelError::Shape {
                name: name.to_string(),
                expected: vec![in_dim],
                got: dims,
            });
        };
        if *last != in_dim {
            return Err(ModelError::Shape {
                name: name.to_string(),
                expected: vec![in_dim],
                got: dims,
            });
        }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let y = x.reshape((rows, in_dim))?.matmul(&wt)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_dim);
        Ok(y.reshape(out_shape)?)
    }

    /// `linear` plus an optional bias row.
    pub fn linear_bias(&self, x: &Tensor, name: &str, bias: Option<&str>) -> Result<Tensor> {
        let y = self.linear(x, name)?;
        match bias {
            Some(b) => Ok(y.broadcast_add(&self.get(b)?)?),
            None => Ok(y),
        }
    }

    /// Embedding rows for `ids`: `weight` is `[vocab, hidden]`.
    /// Selects rows in the resident dtype and upcasts only the tiny
    /// result, so the full table (gigabytes on CUDA) is never cast.
    pub fn embed(&self, name: &str, ids: &[u32]) -> Result<Tensor> {
        let w = self
            .map
            .get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
        let idx = Tensor::from_vec(ids.to_vec(), ids.len(), &self.dev)?;
        let rows = w.index_select(&idx, 0)?.unsqueeze(0)?; // [1, seq, hidden]
        if rows.dtype() == DType::F32 {
            Ok(rows)
        } else {
            Ok(rows.to_dtype(DType::F32)?)
        }
    }

    /// Make this thread current on the device's CUDA context. No-op on
    /// CPU and on threads that already hold the context. The daemon runs
    /// inference on pooled blocking threads, and a fresh thread's first
    /// kernel fails with `CUDA_ERROR_INVALID_CONTEXT` — so every entry
    /// point that touches the GPU calls this first.
    pub fn ensure_current(dev: &Device) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(d) = dev {
            d.cuda_stream()
                .context()
                .bind_to_thread()
                .map_err(|e| ModelError::Config(format!("cuda context bind: {e}")))?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = dev;
        Ok(())
    }

    /// Replace a weight in place (LoRA fusion). Errors when absent.
    /// The incoming tensor may be F32 compute output; it is rounded
    /// back to the resident dtype (F32 fuse, then store).
    pub fn replace(&mut self, name: &str, tensor: Tensor) -> Result<()> {
        match self.map.get_mut(name) {
            Some(slot) => {
                let t = if tensor.dtype() == self.store {
                    tensor
                } else {
                    tensor.to_dtype(self.store)?
                };
                *slot = t;
                Ok(())
            }
            None => Err(ModelError::MissingWeight(name.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_is_always_current() {
        assert!(Weights::ensure_current(&Device::Cpu).is_ok());
    }
}

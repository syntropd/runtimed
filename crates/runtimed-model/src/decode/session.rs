//! Owned inference session: config + weights shared by `Arc`, cache inside.
//!
//! The arch modules are free functions over these handles; `Session` is
//! the object a service holds (behind a mutex: the KV cache mutates).

use crate::config::{Arch, ArchConfig};
use crate::error::{ModelError, Result};
use crate::weights::Weights;
use crate::arch::{gemma4, qwen2};
use candle_core::Tensor;
use std::sync::Arc;

enum Kind {
    Qwen2(qwen2::Cache),
    Gemma4(gemma4::Cache),
}

pub struct Session {
    cfg: Arc<ArchConfig>,
    w: Arc<Weights>,
    kind: Kind,
}

impl Session {
    pub fn new(cfg: Arc<ArchConfig>, w: Arc<Weights>) -> Result<Self> {
        let kind = match cfg.arch {
            Arch::Qwen2 => Kind::Qwen2(qwen2::Cache::new(cfg.n_layer)),
            Arch::Gemma4 => Kind::Gemma4(gemma4::Cache::new(cfg.n_layer)),
        };
        Ok(Self { cfg, w, kind })
    }

    /// Load config + weights from a GGUF file in one call.
    pub fn load(path: &std::path::Path, dev: &candle_core::Device) -> Result<(Self, runtimed_gguf::GgufFile)> {
        let file = runtimed_gguf::GgufFile::open(path)?;
        let cfg = Arc::new(ArchConfig::parse(&file)?);
        let w = Arc::new(Weights::load(&file, dev)?);
        let session = Self::new(cfg, w)?;
        Ok((session, file))
    }

    pub fn config(&self) -> &ArchConfig {
        &self.cfg
    }

    pub fn device(&self) -> &candle_core::Device {
        self.w.device()
    }

    pub fn resident_bytes(&self) -> usize {
        self.w.resident_bytes()
    }

    pub fn reset(&mut self) {
        match &mut self.kind {
            Kind::Qwen2(c) => c.reset(),
            Kind::Gemma4(c) => c.reset(),
        }
    }

    /// Fuse a LoRA adapter into the live weights (base file untouched).
    /// Returns the fused base names. The KV cache is unaffected (weights
    /// only change future forwards); callers should `reset` first anyway.
    pub fn fuse_lora(&mut self, adapter: &crate::lora::LoraAdapter) -> Result<Vec<String>> {
        Weights::ensure_current(self.w.device())?;
        adapter.fuse_into(Arc::make_mut(&mut self.w))
    }

    /// Logits `[1, seq, vocab]` for `ids` starting at absolute position `q0`.
    pub fn forward(&mut self, ids: &[u32], q0: usize) -> Result<Tensor> {
        Weights::ensure_current(self.w.device())?;
        match &mut self.kind {
            Kind::Qwen2(c) => qwen2::forward(&self.cfg, &self.w, c, ids, q0),
            Kind::Gemma4(c) => gemma4::forward(&self.cfg, &self.w, c, ids, q0),
        }
    }

    /// Convenience for single-shot scoring (resets the cache first).
    pub fn score(&mut self, ids: &[u32]) -> Result<Tensor> {
        if ids.is_empty() {
            return Err(ModelError::Config("cannot score an empty prompt".into()));
        }
        self.reset();
        self.forward(ids, 0)
    }

    /// Multimodal prefill (Gemma4 only): `ids` hold `IMG_TOKEN`
    /// placeholders, `soft` is `[1, S, hidden]` with S matching the
    /// placeholder count. Placeholders read as `pad_id` on the text path;
    /// soft tokens scatter over them before the layers run.
    pub fn forward_mm(&mut self, ids: &[u32], soft: &Tensor, pad_id: u32) -> Result<Tensor> {
        Weights::ensure_current(self.w.device())?;
        let Kind::Gemma4(cache) = &mut self.kind else {
            return Err(ModelError::Config("multimodal prefill needs a gemma4 session".into()));
        };
        let subs: Vec<u32> = ids
            .iter()
            .map(|&id| if id == crate::vision::IMG_TOKEN { pad_id } else { id })
            .collect();
        let embeds = gemma4::input_embeds(&self.cfg, &self.w, &subs)?;
        let scattered = scatter_soft(&embeds, ids, soft)?;
        // Token identity from pad-substituted ids (the reference's image
        // batches carry no token ids), context from scattered soft tokens.
        let ple = gemma4::per_layer_inputs(&self.cfg, &self.w, &subs, &scattered)?;
        gemma4::forward_from_embeds(&self.cfg, &self.w, cache, &scattered, &ple, 0)
    }
}

/// Replace placeholder rows of `[1, T, H]` embeds with soft-token rows.
fn scatter_soft(embeds: &Tensor, ids: &[u32], soft: &Tensor) -> Result<Tensor> {
    let t = embeds.dim(1)?;
    let h = embeds.dim(2)?;
    if ids.len() != t {
        return Err(ModelError::Config(format!(
            "scatter: {} ids vs {t} embed rows",
            ids.len()
        )));
    }
    let s = soft.dim(1)?;
    if soft.dim(2)? != h {
        return Err(ModelError::Config("scatter: soft width mismatch".into()));
    }
    let mut e = embeds.reshape((t, h))?.to_vec2::<f32>()?;
    let rows = soft.reshape((s, h))?.to_vec2::<f32>()?;
    let mut j = 0;
    for (i, &id) in ids.iter().enumerate() {
        if id == crate::vision::IMG_TOKEN {
            if j >= s {
                return Err(ModelError::Config("scatter: more placeholders than soft tokens".into()));
            }
            e[i] = rows[j].clone();
            j += 1;
        }
    }
    if j != s {
        return Err(ModelError::Config(format!(
            "scatter: {j} placeholders filled, {s} soft tokens"
        )));
    }
    let flat: Vec<f32> = e.into_iter().flatten().collect();
    Ok(Tensor::from_vec(flat, (1, t, h), embeds.device())?)
}

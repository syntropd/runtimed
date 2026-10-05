//! Owned inference session: config + weights shared by `Arc`, cache inside.
//!
//! The arch modules are free functions over these handles; `Session` is
//! the object a service holds (behind a mutex: the KV cache mutates).

use crate::config::{Arch, ArchConfig};
use crate::error::{ModelError, Result};
use crate::weights::Weights;
use crate::arch::{bitnet, gemma4, granite, phi3, qwen2};
use candle_core::Tensor;
use std::sync::Arc;

enum Kind {
    Qwen2(qwen2::Cache),
    Gemma4(gemma4::Cache),
    Granite(qwen2::Cache),
    Phi3(qwen2::Cache),
    BitNet(qwen2::Cache),
}

pub struct Session {
    cfg: Arc<ArchConfig>,
    w: Arc<Weights>,
    kind: Kind,
}

impl Session {
    pub fn new(cfg: Arc<ArchConfig>, w: Arc<Weights>) -> Result<Self> {
        let kind = match cfg.arch {
            Arch::Gemma4 => Kind::Gemma4(gemma4::Cache::new(cfg.n_layer)),
            Arch::Qwen2 => Kind::Qwen2(qwen2::Cache::new(cfg.n_layer)),
            Arch::Granite => Kind::Granite(qwen2::Cache::new(cfg.n_layer)),
            Arch::Phi3 => Kind::Phi3(qwen2::Cache::new(cfg.n_layer)),
            Arch::BitNet => Kind::BitNet(qwen2::Cache::new(cfg.n_layer)),
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

    /// Load config + weights from an already opened file handle (e.g. sealed memfd).
    pub fn load_from_file(file: &std::fs::File, dev: &candle_core::Device) -> Result<(Self, runtimed_gguf::GgufFile)> {
        let gguf_file = runtimed_gguf::GgufFile::from_file(file)?;
        let cfg = Arc::new(ArchConfig::parse(&gguf_file)?);
        let w = Arc::new(Weights::load(&gguf_file, dev)?);
        let session = Self::new(cfg, w)?;
        Ok((session, gguf_file))
    }

    /// Load config + weights from a Safetensors file.
    pub fn load_safetensors(
        weights_path: &std::path::Path,
        config_path: Option<&std::path::Path>,
        dev: &candle_core::Device,
    ) -> Result<Self> {
        let cfg = match config_path {
            Some(cp) => ArchConfig::from_hf_file(cp)?,
            None => {
                let p = weights_path.with_extension("config.json");
                let alt = weights_path.parent().map(|d| d.join("config.json"));
                let cp = [Some(p), alt].into_iter().flatten().find(|p| p.exists())
                    .ok_or_else(|| ModelError::Config(format!("missing config for {}", weights_path.display())))?;
                ArchConfig::from_hf_file(&cp)?
            }
        };
        let w = Arc::new(Weights::load_safetensors(weights_path, dev)?);
        Self::new(Arc::new(cfg), w)
    }

    /// Load config + weights from an already opened Safetensors file handle.
    pub fn load_safetensors_from_file(
        file: &std::fs::File,
        cfg: Arc<ArchConfig>,
        dev: &candle_core::Device,
    ) -> Result<Self> {
        let w = Arc::new(Weights::load_safetensors_from_file(file, dev)?);
        Self::new(cfg, w)
    }

    pub fn config(&self) -> &ArchConfig {
        &self.cfg
    }

    pub fn device(&self) -> &candle_core::Device {
        self.w.device()
    }

    pub fn weights(&self) -> &Arc<Weights> {
        &self.w
    }

    pub fn resident_bytes(&self) -> usize {
        self.w.resident_bytes()
    }

    pub fn reset(&mut self) {
        let _ = Weights::ensure_current(self.device());
        match &mut self.kind {
            Kind::Qwen2(c) | Kind::Granite(c) | Kind::Phi3(c) | Kind::BitNet(c) => c.reset(),
            Kind::Gemma4(c) => c.reset(),
        }
    }

    /// Roll back internal KV cache to target_len tokens for speculative decoding.
    pub fn truncate(&mut self, target_len: usize) {
        let _ = Weights::ensure_current(self.device());
        match &mut self.kind {
            Kind::Qwen2(c) | Kind::Granite(c) | Kind::Phi3(c) | Kind::BitNet(c) => c.truncate(target_len),
            Kind::Gemma4(c) => c.truncate(target_len),
        }
    }

    /// Spill up to `count` resident accelerator KV layers to host CPU RAM.
    pub fn spill_layers(&mut self, count: usize) -> Result<usize> {
        Weights::ensure_current(self.device())?;
        match &mut self.kind {
            Kind::Qwen2(c) | Kind::Granite(c) | Kind::Phi3(c) | Kind::BitNet(c) => c.spill_layers(count),
            Kind::Gemma4(c) => c.spill_layers(count),
        }
    }

    /// Prefetch up to `count` spilled CPU KV layers back to target compute device.
    pub fn prefetch_layers(&mut self, count: usize) -> Result<usize> {
        Weights::ensure_current(self.device())?;
        let dev = self.device().clone();
        match &mut self.kind {
            Kind::Qwen2(c) | Kind::Granite(c) | Kind::Phi3(c) | Kind::BitNet(c) => c.prefetch_layers(&dev, count),
            Kind::Gemma4(c) => c.prefetch_layers(&dev, count),
        }
    }

    /// Splice pre-computed visual KV blocks into the active cache.
    pub fn splice_visual_kv(&mut self, kv_layers: &[Option<(Tensor, Tensor)>]) -> Result<()> {
        let dev = self.device().clone();
        match &mut self.kind {
            Kind::Gemma4(c) => c.splice_kv(kv_layers, &dev),
            _ => Err(ModelError::Config("visual KV splicing unsupported".into())),
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
            Kind::Granite(c) => granite::forward(&self.cfg, &self.w, c, ids, q0),
            Kind::Phi3(c) => phi3::forward(&self.cfg, &self.w, c, ids, q0),
            Kind::BitNet(c) => bitnet::forward(&self.cfg, &self.w, c, ids, q0),
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

    /// Score specific candidate tokens against prompt context, returning [1, K] logits.
    pub fn score_candidates(&mut self, prompt_ids: &[u32], candidate_ids: &[u32]) -> Result<Tensor> {
        if prompt_ids.is_empty() || candidate_ids.is_empty() {
            return Err(ModelError::Config("cannot score empty prompt/candidates".into()));
        }
        Weights::ensure_current(self.w.device())?;
        self.reset();
        let h_n = match &mut self.kind {
            Kind::Qwen2(c) => qwen2::forward_last_hidden(&self.cfg, &self.w, c, prompt_ids)?,
            Kind::Gemma4(c) => gemma4::forward_last_hidden(&self.cfg, &self.w, c, prompt_ids)?,
            Kind::Granite(c) => granite::forward_last_hidden(&self.cfg, &self.w, c, prompt_ids)?,
            Kind::Phi3(c) => phi3::forward_last_hidden(&self.cfg, &self.w, c, prompt_ids)?,
            Kind::BitNet(c) => bitnet::forward_last_hidden(&self.cfg, &self.w, c, prompt_ids)?,
        };
        let key = if self.w.contains_key("output.weight") { "output.weight" } else { "token_embd.weight" };
        let w_c = self.w.candidate_weights(key, candidate_ids)?;
        let out_dev = w_c.device();
        Weights::ensure_current(out_dev)?;
        let h_n = if !h_n.device().same_device(out_dev) { h_n.to_device(out_dev)? } else { h_n };
        let w_c = if w_c.dtype() != h_n.dtype() { w_c.to_dtype(h_n.dtype())? } else { w_c };
        let mut logits = h_n.matmul(&w_c.t()?)?;
        if let Some(scale) = self.cfg.logit_scale {
            if scale != 1.0 { logits = logits.affine((1.0 / scale) as f64, 0.0)?; }
        }
        if let Some(cap) = self.cfg.final_softcap {
            logits = logits.affine((1.0 / cap) as f64, 0.0)?.tanh()?.affine(cap as f64, 0.0)?;
        }
        Ok(logits)
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
    let (t, h) = (embeds.dim(1)?, embeds.dim(2)?);
    if ids.len() != t || soft.dim(2)? != h {
        return Err(ModelError::Config("scatter: dimension mismatch".into()));
    }
    let (s, mut e) = (soft.dim(1)?, embeds.reshape((t, h))?.to_vec2::<f32>()?);
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
        return Err(ModelError::Config(format!("scatter: {j} placeholders filled, {s} soft tokens")));
    }
    let flat: Vec<f32> = e.into_iter().flatten().collect();
    Ok(Tensor::from_vec(flat, (1, t, h), embeds.device())?)
}

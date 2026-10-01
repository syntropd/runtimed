//! Gemma4 decoder with proportional RoPE and per-layer embedding injection.

use crate::config::ArchConfig;
use crate::error::{ModelError, Result};
use crate::ops;
use crate::weights::Weights;
use candle_core::Tensor;

/// Per-layer `(k, v)` caches. Only layers that own KV ever fill a slot;
/// shared layers read their source layer's slot instead.
pub struct Cache {
    layers: Vec<Option<(Tensor, Tensor)>>,
}

impl Cache {
    pub fn new(n_layer: usize) -> Self {
        Self {
            layers: (0..n_layer).map(|_| None).collect(),
        }
    }

    pub fn reset(&mut self) {
        for slot in self.layers.iter_mut() {
            *slot = None;
        }
    }

    pub fn truncate(&mut self, target_len: usize) {
        for slot in self.layers.iter_mut() {
            if let Some((k, v)) = slot.take() {
                if target_len > 0 && k.dim(2).map(|l| l > target_len).unwrap_or(false) {
                    if let (Ok(kt), Ok(vt)) = (k.narrow(2, 0, target_len), v.narrow(2, 0, target_len)) {
                        *slot = Some((kt, vt));
                        continue;
                    }
                }
                if target_len > 0 {
                    *slot = Some((k, v));
                }
            }
        }
    }

    /// Splice visual KV blocks into cache on the target device.
    pub fn splice_kv(&mut self, kv: &[Option<(Tensor, Tensor)>], dev: &candle_core::Device) -> Result<()> {
        for (i, (k, v)) in kv.iter().enumerate().filter_map(|(i, o)| o.as_ref().map(|p| (i, p))) {
            if i >= self.layers.len() { break; }
            let (kd, vd) = (k.to_device(dev)?, v.to_device(dev)?);
            self.layers[i] = match self.layers[i].take() {
                Some((ok, ov)) => Some((Tensor::cat(&[&ok, &kd], 2)?, Tensor::cat(&[&ov, &vd], 2)?)),
                None => Some((kd, vd)),
            };
        }
        Ok(())
    }

    /// Spill up to `max_layers` resident accelerator KV layers to host CPU RAM.
    pub fn spill_layers(&mut self, max_layers: usize) -> Result<usize> {
        let mut n = 0;
        for (k, v) in self.layers.iter_mut().flatten() {
            if n >= max_layers { break; }
            if !matches!(k.device(), candle_core::Device::Cpu) {
                *k = k.to_device(&candle_core::Device::Cpu)?;
                *v = v.to_device(&candle_core::Device::Cpu)?;
                n += 1;
            }
        }
        Ok(n)
    }

    /// Prefetch up to `max_layers` spilled CPU KV layers back to compute device.
    pub fn prefetch_layers(&mut self, dev: &candle_core::Device, max_layers: usize) -> Result<usize> {
        let mut n = 0;
        for (k, v) in self.layers.iter_mut().flatten() {
            if n >= max_layers { break; }
            if matches!(k.device(), candle_core::Device::Cpu) && !matches!(dev, candle_core::Device::Cpu) {
                *k = k.to_device(dev)?;
                *v = v.to_device(dev)?;
                n += 1;
            }
        }
        Ok(n)
    }
}

/// Per-layer inputs `[1, seq, n_layer, ple]`: token identity plus projection.
pub(crate) fn per_layer_inputs(
    cfg: &ArchConfig,
    w: &Weights,
    tok_ids: &[u32],
    ctx_embeds: &Tensor,
) -> Result<Tensor> {
    let ple = cfg.ple_dim;
    let n_layer = cfg.n_layer;
    let t = ctx_embeds.dim(1)?;
    // Token identity, scaled by sqrt(ple).
    let tok = w.embed("per_layer_token_embd.weight", tok_ids)?;
    let tok = tok
        .affine((ple as f32).sqrt() as f64, 0.0)?
        .reshape((1, t, n_layer, ple))?;
    // Context projection, scaled by 1/sqrt(hidden), then normalized.
    let ctx = w.linear(ctx_embeds, "per_layer_model_proj.weight")?;
    let ctx = ctx
        .affine((cfg.hidden as f32).sqrt().recip() as f64, 0.0)?
        .reshape((1, t, n_layer, ple))?;
    let ctx = ops::rms_norm(&ctx, &w.get("per_layer_proj_norm.weight")?, cfg.eps)?;
    Ok(tok
        .broadcast_add(&ctx)?
        .affine(std::f64::consts::FRAC_1_SQRT_2, 0.0)?)
}

fn layer(
    cfg: &ArchConfig,
    w: &Weights,
    cache: &mut Cache,
    i: usize,
    h: &Tensor,
    ple: &Tensor,
    q0: usize,
) -> Result<Tensor> {
    let lc = &cfg.layers[i];
    let pre = format!("blk.{i}");
    let dev = w.device();
    let t = h.dim(1)?;

    // Attention block with post norm.
    let n = ops::rms_norm(h, &w.get(&format!("{pre}.attn_norm.weight"))?, cfg.eps)?;
    let split = |y: Tensor| -> Result<Tensor> {
        let heads = y.dim(2)? / lc.head_dim;
        Ok(y.reshape((1, t, heads, lc.head_dim))?.transpose(1, 2)?)
    };
    let factors = (!lc.is_swa).then_some(cfg.rope_factors.as_deref()).flatten();
    let q = w.linear(&n, &format!("{pre}.attn_q.weight"))?;
    let q = ops::rms_norm(&split(q)?, &w.get(&format!("{pre}.attn_q_norm.weight"))?, cfg.eps)?;
    let q = ops::rope_neox(&q, q0, lc.rope_theta, lc.rope_dim, factors)?;
    let (k_full, v_full) = if lc.has_kv {
        let k = w.linear(&n, &format!("{pre}.attn_k.weight"))?;
        let k = ops::rms_norm(&split(k)?, &w.get(&format!("{pre}.attn_k_norm.weight"))?, cfg.eps)?;
        let k = ops::rope_neox(&k, q0, lc.rope_theta, lc.rope_dim, factors)?;
        let v = w.linear(&n, &format!("{pre}.attn_v.weight"))?;
        let v = ops::rms_norm_plain(&split(v)?, cfg.eps)?;
        match cache.layers[i].take() {
            Some((pk, pv)) => {
                let pk = if pk.device().same_device(dev) { pk } else { pk.to_device(dev)? };
                let pv = if pv.device().same_device(dev) { pv } else { pv.to_device(dev)? };
                (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?)
            }
            None => (k, v),
        }
    } else {
        let (sk, sv) = cache.layers[lc.kv_source].clone().ok_or_else(|| {
            ModelError::Config(format!("layer {i}: KV source cache empty"))
        })?;
        let sk = if sk.device().same_device(dev) { sk } else { sk.to_device(dev)? };
        let sv = if sv.device().same_device(dev) { sv } else { sv.to_device(dev)? };
        (sk, sv)
    };
    let total = k_full.dim(2)?;
    let window = lc.is_swa.then_some(cfg.sliding_window).flatten();
    let mask = ops::causal_mask(t, total, q0, window, dev)?;
    let scale = cfg.attn_scale.unwrap_or_else(|| (lc.head_dim as f32).sqrt().recip());
    let o = ops::attention(&q, &k_full, &v_full, &mask, scale)?;
    if lc.has_kv {
        cache.layers[i] = Some((k_full, v_full));
    }
    let o = o.transpose(1, 2)?.reshape((1, t, lc.n_head * lc.head_dim))?;
    let o = w.linear(&o, &format!("{pre}.attn_output.weight"))?;
    let o = ops::rms_norm(&o, &w.get(&format!("{pre}.post_attention_norm.weight"))?, cfg.eps)?;
    let h = h.broadcast_add(&o)?;

    // GeGLU block with post norm.
    let n = ops::rms_norm(&h, &w.get(&format!("{pre}.ffn_norm.weight"))?, cfg.eps)?;
    let g = w.linear(&n, &format!("{pre}.ffn_gate.weight"))?;
    let u = w.linear(&n, &format!("{pre}.ffn_up.weight"))?;
    let mlp = w.linear(&ops::gelu_tanh(&g)?.broadcast_mul(&u)?, &format!("{pre}.ffn_down.weight"))?;
    let mlp = ops::rms_norm(&mlp, &w.get(&format!("{pre}.post_ffw_norm.weight"))?, cfg.eps)?;
    let h = h.broadcast_add(&mlp)?;

    // Per-layer embedding injection.
    let gate = ops::gelu_tanh(&w.linear(&h, &format!("{pre}.inp_gate.weight"))?)?;
    let pli = ple.narrow(2, i, 1)?.squeeze(2)?;
    let inj = w.linear(&gate.broadcast_mul(&pli)?, &format!("{pre}.proj.weight"))?;
    let inj = ops::rms_norm(&inj, &w.get(&format!("{pre}.post_norm.weight"))?, cfg.eps)?;
    let h = h.broadcast_add(&inj)?;

    // Learned per-layer rescale.
    Ok(h.broadcast_mul(&w.get(&format!("{pre}.layer_output_scale.weight"))?)?)
}

/// Scaled token embeddings (text path's first step, shared by MM).
pub fn input_embeds(cfg: &ArchConfig, w: &Weights, ids: &[u32]) -> Result<Tensor> {
    Ok(w.embed("token_embd.weight", ids)?.affine(cfg.embed_scale as f64, 0.0)?)
}

pub fn forward_embeds(cfg: &ArchConfig, w: &Weights, ids: &[u32]) -> Result<(Tensor, Tensor)> {
    let embeds = input_embeds(cfg, w, ids)?;
    let ple = per_layer_inputs(cfg, w, ids, &embeds)?;
    Ok((embeds, ple))
}

/// Logits from ready-made embeddings (text path's second half).
pub fn forward_from_embeds(
    cfg: &ArchConfig, w: &Weights, cache: &mut Cache, embeds: &Tensor, ple: &Tensor, q0: usize,
) -> Result<Tensor> {
    let mut h = embeds.clone();
    for i in 0..cfg.n_layer {
        h = layer(cfg, w, cache, i, &h, ple, q0)?;
    }
    h = ops::rms_norm(&h, &w.get("output_norm.weight")?, cfg.eps)?;
    // Tied head: logits over the embedding rows.
    let mut logits = w.linear(&h, "token_embd.weight")?;
    if let Some(cap) = cfg.final_softcap {
        logits = logits.affine((1.0 / cap) as f64, 0.0)?.tanh()?.affine(cap as f64, 0.0)?;
    }
    Ok(logits)
}

/// Logits `[1, seq, vocab]` for `ids` starting at absolute position `q0`.
pub fn forward(
    cfg: &ArchConfig, w: &Weights, cache: &mut Cache, ids: &[u32], q0: usize,
) -> Result<Tensor> {
    let (embeds, ple) = forward_embeds(cfg, w, ids)?;
    forward_from_embeds(cfg, w, cache, &embeds, &ple, q0)
}

/// Normalized final hidden state `[1, hidden_dim]` for `prompt_ids`.
pub fn forward_last_hidden(
    cfg: &ArchConfig,
    w: &Weights,
    cache: &mut Cache,
    prompt_ids: &[u32],
) -> Result<Tensor> {
    if prompt_ids.is_empty() {
        return Err(crate::error::ModelError::Config(
            "cannot score an empty prompt".into(),
        ));
    }
    let (embeds, ple) = forward_embeds(cfg, w, prompt_ids)?;
    let mut h = embeds;
    for i in 0..cfg.n_layer {
        h = layer(cfg, w, cache, i, &h, &ple, 0)?;
    }
    h = ops::rms_norm(&h, &w.get("output_norm.weight")?, cfg.eps)?;
    let seq = h.dim(1)?;
    let last = h.narrow(1, seq - 1, 1)?.squeeze(1)?;
    Ok(last)
}

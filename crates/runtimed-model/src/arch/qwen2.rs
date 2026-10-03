//! Qwen2 decoder: the simple path that proves the pipeline.
//!
//! Per layer: `h += attn(ln1(h))`, `h += mlp(ln2(h))`, with QKV bias,
//! SwiGLU, full causal attention, untied LM head. No scaling tricks.

use crate::config::ArchConfig;
use crate::error::Result;
use crate::ops;
use crate::weights::Weights;
use candle_core::Tensor;

/// Per-layer `(k, v)` caches: `[1, n_kv, total, head_dim]`.
pub struct Cache {
    pub(crate) layers: Vec<Option<(Tensor, Tensor)>>,
}

impl Cache {
    pub fn new(n_layer: usize) -> Self {
        Self {
            layers: (0..n_layer).map(|_| None).collect(),
        }
    }

    pub fn reset(&mut self) {
        for slot in self.layers.iter_mut() {
            if let Some((k, v)) = slot.take() {
                let _ = Weights::ensure_current(k.device());
                drop((k, v));
            }
        }
    }

    pub fn truncate(&mut self, target_len: usize) {
        for slot in self.layers.iter_mut() {
            if let Some((k, v)) = slot.take() {
                let _ = Weights::ensure_current(k.device());
                if target_len > 0 && k.dim(2).map(|l| l > target_len).unwrap_or(false) {
                    if let (Ok(kt), Ok(vt)) = (k.narrow(2, 0, target_len), v.narrow(2, 0, target_len)) {
                        let kt = kt.contiguous().unwrap_or(kt);
                        let vt = vt.contiguous().unwrap_or(vt);
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

    /// Prefetch up to `max_layers` spilled CPU KV layers back to target compute device.
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

fn layer(
    cfg: &ArchConfig,
    w: &Weights,
    cache: &mut Cache,
    i: usize,
    h: &Tensor,
    q0: usize,
) -> Result<Tensor> {
    let lc = &cfg.layers[i];
    let pre = format!("blk.{i}");
    let norm_w = w.get_raw(&format!("{pre}.attn_norm.weight"))?;
    let dev = norm_w.device();
    Weights::ensure_current(dev)?;
    let h = if !h.device().same_device(dev) {
        h.to_device(dev)?
    } else {
        h.clone()
    };

    // Attention block.
    let n = ops::rms_norm(&h, &w.get(&format!("{pre}.attn_norm.weight"))?, cfg.eps)?;
    let bias = cfg.has_qkv_bias;
    let bq = bias.then(|| format!("{pre}.attn_q.bias"));
    let bk = bias.then(|| format!("{pre}.attn_k.bias"));
    let bv = bias.then(|| format!("{pre}.attn_v.bias"));
    let t = n.dim(1)?;
    let split = |y: Tensor| -> Result<Tensor> {
        let heads = y.dim(2)? / lc.head_dim;
        Ok(y.reshape((1, t, heads, lc.head_dim))?.transpose(1, 2)?)
    };
    let q = split(w.linear_bias(&n, &format!("{pre}.attn_q.weight"), bq.as_deref())?)?;
    let k = split(w.linear_bias(&n, &format!("{pre}.attn_k.weight"), bk.as_deref())?)?;
    let v = split(w.linear_bias(&n, &format!("{pre}.attn_v.weight"), bv.as_deref())?)?;
    let q = ops::rope_neox(&q, q0, lc.rope_theta, lc.rope_dim, None)?;
    let k = ops::rope_neox(&k, q0, lc.rope_theta, lc.rope_dim, None)?;
    // Extend the KV cache and attend over all of it.
    let (k_full, v_full) = match cache.layers[i].take() {
        Some((pk, pv)) => {
            let pk = if pk.device().same_device(dev) { pk } else { pk.to_device(dev)? };
            let pv = if pv.device().same_device(dev) { pv } else { pv.to_device(dev)? };
            (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?)
        }
        None => (k, v),
    };
    let total = k_full.dim(2)?;
    let mask = ops::causal_mask(t, total, q0, None, dev)?;
    let scale = (lc.head_dim as f32).sqrt().recip();
    let o = ops::attention(&q, &k_full, &v_full, &mask, scale)?;
    cache.layers[i] = Some((k_full, v_full));
    let o = o.transpose(1, 2)?.reshape((1, t, lc.n_head * lc.head_dim))?;
    let o = w.linear(&o, &format!("{pre}.attn_output.weight"))?;
    let h = h.broadcast_add(&o)?;

    // MLP block.
    let n = ops::rms_norm(&h, &w.get(&format!("{pre}.ffn_norm.weight"))?, cfg.eps)?;
    let g = w.linear(&n, &format!("{pre}.ffn_gate.weight"))?;
    let u = w.linear(&n, &format!("{pre}.ffn_up.weight"))?;
    let mlp = w.linear(
        &ops::silu(&g)?.broadcast_mul(&u)?,
        &format!("{pre}.ffn_down.weight"),
    )?;
    Ok(h.broadcast_add(&mlp)?)
}

/// Logits `[1, seq, vocab]` for `ids` starting at absolute position `q0`.
pub fn forward(
    cfg: &ArchConfig,
    w: &Weights,
    cache: &mut Cache,
    ids: &[u32],
    q0: usize,
) -> Result<Tensor> {
    let mut h = w.embed("token_embd.weight", ids)?;
    for i in 0..cfg.n_layer {
        h = layer(cfg, w, cache, i, &h, q0)?;
    }
    let norm_w = w.get_raw("output_norm.weight")?;
    let out_dev = norm_w.device();
    Weights::ensure_current(out_dev)?;
    let h = if !h.device().same_device(out_dev) {
        h.to_device(out_dev)?
    } else {
        h
    };
    let h = ops::rms_norm(&h, &w.get("output_norm.weight")?, cfg.eps)?;
    w.linear(&h, "output.weight")
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
    let mut h = w.embed("token_embd.weight", prompt_ids)?;
    for i in 0..cfg.n_layer {
        h = layer(cfg, w, cache, i, &h, 0)?;
    }
    let norm_w = w.get_raw("output_norm.weight")?;
    let out_dev = norm_w.device();
    Weights::ensure_current(out_dev)?;
    let h = if !h.device().same_device(out_dev) {
        h.to_device(out_dev)?
    } else {
        h
    };
    let h = ops::rms_norm(&h, &w.get("output_norm.weight")?, cfg.eps)?;
    let seq = h.dim(1)?;
    let last = h.narrow(1, seq - 1, 1)?.squeeze(1)?;
    Ok(last)
}

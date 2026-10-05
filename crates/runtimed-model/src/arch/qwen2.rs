//! Qwen2 decoder: the simple path that proves the pipeline.
//!
//! Per layer: `h += attn(ln1(h))`, `h += mlp(ln2(h))`, with QKV bias,
//! SwiGLU, full causal attention, untied LM head. No scaling tricks.
//! All tensor math operates behind the sovereign `SubstratePort` boundary.

use crate::config::ArchConfig;
use crate::error::Result;
use crate::substrate::{default_substrate, Device, SubstratePort, Tensor};
use crate::tp::DualGpuContext;
use crate::weights::Weights;

/// Per-layer contiguous chunked O(1) KV cache.
pub struct Cache {
    pub(crate) entries: Vec<Option<crate::cache::LayerKv>>,
}

impl Cache {
    pub fn new(n_layer: usize) -> Self {
        Self {
            entries: (0..n_layer).map(|_| None).collect(),
        }
    }

    pub fn reset(&mut self) {
        for slot in self.entries.iter_mut() {
            if let Some(kv) = slot.take() {
                let _ = Weights::ensure_current(kv.k_buf.device());
                drop(kv);
            }
        }
    }

    pub fn truncate(&mut self, target_len: usize) {
        for slot in self.entries.iter_mut().flatten() {
            let _ = Weights::ensure_current(slot.k_buf.device());
            slot.truncate(target_len);
        }
    }

    pub fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor, dev: &Device) -> Result<(Tensor, Tensor)> {
        self.append_with_substrate(default_substrate(), layer, k, v, dev)
    }

    pub fn append_with_substrate<S: SubstratePort>(
        &mut self,
        sub: &S,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
        _dev: &Device,
    ) -> Result<(Tensor, Tensor)> {
        if layer >= self.entries.len() {
            return Err(crate::error::ModelError::Config(format!("layer {layer} out of bounds")));
        }
        match &mut self.entries[layer] {
            Some(kv) => kv.append(k, v),
            None => {
                let kv = sub.alloc_kv(k, v, 256)?;
                let res = kv.current()?;
                self.entries[layer] = Some(kv);
                Ok(res)
            }
        }
    }

    /// Spill up to `max_layers` resident accelerator KV layers to host CPU RAM.
    pub fn spill_layers(&mut self, max_layers: usize) -> Result<usize> {
        let mut n = 0;
        for kv in self.entries.iter_mut().flatten() {
            if n >= max_layers { break; }
            if kv.spill_to_cpu()? {
                n += 1;
            }
        }
        Ok(n)
    }

    /// Prefetch up to `max_layers` spilled CPU KV layers back to target compute device.
    pub fn prefetch_layers(&mut self, dev: &Device, max_layers: usize) -> Result<usize> {
        let mut n = 0;
        for kv in self.entries.iter_mut().flatten() {
            if n >= max_layers { break; }
            if kv.prefetch_to_dev(dev)? {
                n += 1;
            }
        }
        Ok(n)
    }
}

fn layer<S: SubstratePort>(
    sub: &S,
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
    let n = sub.rmsnorm(&h, &w.get(&format!("{pre}.attn_norm.weight"))?, cfg.eps)?;
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
    let q = sub.rope(&q, q0, lc.rope_theta, lc.rope_dim, None)?;
    let k = sub.rope(&k, q0, lc.rope_theta, lc.rope_dim, None)?;
    // Extend the KV cache in O(1) contiguous chunked storage and attend over it.
    let (k_full, v_full) = cache.append_with_substrate(sub, i, &k, &v, dev)?;
    let total = k_full.dim(2)?;
    let mask = if t == 1 {
        None
    } else {
        Some(sub.causal_mask(t, total, q0, None, dev)?)
    };
    let scale = (lc.head_dim as f32).sqrt().recip();
    let o = sub.attention(&q, &k_full, &v_full, mask.as_ref(), scale)?;
    let o = o.transpose(1, 2)?.reshape((1, t, lc.n_head * lc.head_dim))?;
    let o = w.linear(&o, &format!("{pre}.attn_output.weight"))?;
    let h = h.broadcast_add(&o)?;

    // MLP block.
    let n = sub.rmsnorm(&h, &w.get(&format!("{pre}.ffn_norm.weight"))?, cfg.eps)?;
    let g = w.linear(&n, &format!("{pre}.ffn_gate.weight"))?;
    let u = w.linear(&n, &format!("{pre}.ffn_up.weight"))?;
    let mlp = w.linear(
        &sub.silu(&g)?.broadcast_mul(&u)?,
        &format!("{pre}.ffn_down.weight"),
    )?;
    Ok(h.broadcast_add(&mlp)?)
}

/// Single layer step execution with optional TP=2 dual-GPU acceleration.
#[allow(clippy::too_many_arguments)]
pub fn forward_step<S: SubstratePort>(
    sub: &S, cfg: &ArchConfig, w: &Weights, cache: &mut Cache, i: usize, h: &Tensor, q0: usize, tp: Option<&DualGpuContext>,
) -> Result<Tensor> {
    if let Some(tp_ctx) = tp {
        forward_step_tp(sub, cfg, w, cache, i, h, q0, tp_ctx)
    } else {
        layer(sub, cfg, w, cache, i, h, q0)
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_step_tp<S: SubstratePort>(
    sub: &S, cfg: &ArchConfig, w: &Weights, cache: &mut Cache, i: usize, h: &Tensor, q0: usize, tp: &DualGpuContext,
) -> Result<Tensor> {
    let (pre, devs, lc) = (format!("blk.{i}"), tp.devices(), &cfg.layers[i]);
    if let (Ok(qw), Ok(kw), Ok(vw), Ok(ow), Ok(gw), Ok(uw), Ok(dw)) = (
        w.get_raw(&format!("{pre}.attn_q.weight")),
        w.get_raw(&format!("{pre}.attn_k.weight")),
        w.get_raw(&format!("{pre}.attn_v.weight")),
        w.get_raw(&format!("{pre}.attn_output.weight")),
        w.get_raw(&format!("{pre}.ffn_gate.weight")),
        w.get_raw(&format!("{pre}.ffn_up.weight")),
        w.get_raw(&format!("{pre}.ffn_down.weight")),
    ) {
        if let Ok(blk) = crate::tp::DualGpuBlock::from_full_weights(
            qw, kw, vw, ow, lc.n_head, lc.n_kv, gw, uw, dw, &devs,
        ) {
            let n = sub.rmsnorm(h, &w.get(&format!("{pre}.attn_norm.weight"))?, cfg.eps)?;
            let [out0, _] = blk.forward(&n.to_device(&devs[0])?, &n.to_device(&devs[1])?, None, None)?;
            let res = if out0.device().same_device(h.device()) { out0 } else { out0.to_device(h.device())? };
            return Ok(h.broadcast_add(&res)?);
        }
    }
    layer(sub, cfg, w, cache, i, h, q0)
}

/// Logits `[1, seq, vocab]` for `ids` starting at absolute position `q0`.
pub fn forward(cfg: &ArchConfig, w: &Weights, cache: &mut Cache, ids: &[u32], q0: usize) -> Result<Tensor> {
    forward_with_substrate(default_substrate(), cfg, w, cache, ids, q0)
}

/// Logits `[1, seq, vocab]` evaluated through explicit `SubstratePort`.
pub fn forward_with_substrate<S: SubstratePort>(
    sub: &S, cfg: &ArchConfig, w: &Weights, cache: &mut Cache, ids: &[u32], q0: usize,
) -> Result<Tensor> {
    let mut h = w.embed("token_embd.weight", ids)?;
    for i in 0..cfg.n_layer {
        h = forward_step(sub, cfg, w, cache, i, &h, q0, None)?;
    }
    let norm_w = w.get_raw("output_norm.weight")?;
    Weights::ensure_current(norm_w.device())?;
    let h = if !h.device().same_device(norm_w.device()) { h.to_device(norm_w.device())? } else { h };
    let h = sub.rmsnorm(&h, &w.get("output_norm.weight")?, cfg.eps)?;
    w.linear(&h, "output.weight")
}

/// Normalized final hidden state `[1, hidden_dim]` for `prompt_ids`.
pub fn forward_last_hidden(cfg: &ArchConfig, w: &Weights, cache: &mut Cache, prompt_ids: &[u32]) -> Result<Tensor> {
    forward_last_hidden_with_substrate(default_substrate(), cfg, w, cache, prompt_ids)
}

/// Normalized final hidden state evaluated through explicit `SubstratePort`.
pub fn forward_last_hidden_with_substrate<S: SubstratePort>(
    sub: &S, cfg: &ArchConfig, w: &Weights, cache: &mut Cache, prompt_ids: &[u32],
) -> Result<Tensor> {
    if prompt_ids.is_empty() {
        return Err(crate::error::ModelError::Config("cannot score an empty prompt".into()));
    }
    let mut h = w.embed("token_embd.weight", prompt_ids)?;
    for i in 0..cfg.n_layer {
        h = forward_step(sub, cfg, w, cache, i, &h, 0, None)?;
    }
    let norm_w = w.get_raw("output_norm.weight")?;
    Weights::ensure_current(norm_w.device())?;
    let h = if !h.device().same_device(norm_w.device()) { h.to_device(norm_w.device())? } else { h };
    let h = sub.rmsnorm(&h, &w.get("output_norm.weight")?, cfg.eps)?;
    let seq = h.dim(1)?;
    Ok(h.narrow(1, seq - 1, 1)?.squeeze(1)?)
}

//! Gemma4 decoder with proportional RoPE and per-layer embedding injection.
//! All tensor math operates behind the sovereign `SubstratePort` boundary.

use crate::config::ArchConfig;
use crate::error::{ModelError, Result};
use crate::substrate::{default_substrate, Device, SubstratePort, Tensor};
use crate::tp::DualGpuContext;
use crate::weights::Weights;

/// Per-layer contiguous chunked O(1) KV cache. Only layers that own KV
/// store entries; shared layers read their source layer's slot instead.
pub struct Cache {
    pub(crate) entries: Vec<Option<crate::cache::LayerKv>>,
}

impl Cache {
    pub fn new(n_layer: usize) -> Self {
        Self { entries: (0..n_layer).map(|_| None).collect() }
    }

    pub fn reset(&mut self) { for slot in self.entries.iter_mut() { *slot = None; } }
    pub fn truncate(&mut self, target_len: usize) { for slot in self.entries.iter_mut().flatten() { slot.truncate(target_len); } }
    pub fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor, dev: &Device) -> Result<(Tensor, Tensor)> {
        self.append_with_substrate(default_substrate(), layer, k, v, dev)
    }

    pub fn append_with_substrate<S: SubstratePort>(&mut self, sub: &S, layer: usize, k: &Tensor, v: &Tensor, _dev: &Device) -> Result<(Tensor, Tensor)> {
        if layer >= self.entries.len() { return Err(ModelError::Config(format!("layer {layer} out of bounds"))); }
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

    pub fn get_layer(&self, layer: usize) -> Result<(Tensor, Tensor)> {
        self.entries.get(layer).and_then(|o| o.as_ref())
            .ok_or_else(|| ModelError::Config(format!("layer {layer}: KV source empty")))?
            .current()
    }

    pub fn splice_kv(&mut self, kv: &[Option<(Tensor, Tensor)>], dev: &Device) -> Result<()> {
        for (i, (k, v)) in kv.iter().enumerate().filter_map(|(i, o)| o.as_ref().map(|p| (i, p))) {
            if i >= self.entries.len() { break; }
            let (kd, vd) = (k.to_device(dev)?, v.to_device(dev)?);
            let _ = self.append(i, &kd, &vd, dev)?;
        }
        Ok(())
    }

    pub fn spill_layers(&mut self, max_layers: usize) -> Result<usize> {
        let mut n = 0;
        for kv in self.entries.iter_mut().flatten() { if n >= max_layers { break; } else if kv.spill_to_cpu()? { n += 1; } }
        Ok(n)
    }

    pub fn prefetch_layers(&mut self, dev: &Device, max_layers: usize) -> Result<usize> {
        let mut n = 0;
        for kv in self.entries.iter_mut().flatten() { if n >= max_layers { break; } else if kv.prefetch_to_dev(dev)? { n += 1; } }
        Ok(n)
    }
}

pub(crate) fn per_layer_inputs(cfg: &ArchConfig, w: &Weights, tok_ids: &[u32], ctx_embeds: &Tensor) -> Result<Tensor> {
    per_layer_inputs_with_substrate(default_substrate(), cfg, w, tok_ids, ctx_embeds)
}

pub(crate) fn per_layer_inputs_with_substrate<S: SubstratePort>(
    sub: &S, cfg: &ArchConfig, w: &Weights, tok_ids: &[u32], ctx_embeds: &Tensor,
) -> Result<Tensor> {
    let (ple, n_layer, t) = (cfg.ple_dim, cfg.n_layer, ctx_embeds.dim(1)?);
    let tok = w.embed("per_layer_token_embd.weight", tok_ids)?.affine((ple as f32).sqrt() as f64, 0.0)?.reshape((1, t, n_layer, ple))?;
    let ctx = w.linear(ctx_embeds, "per_layer_model_proj.weight")?.affine((cfg.hidden as f32).sqrt().recip() as f64, 0.0)?.reshape((1, t, n_layer, ple))?;
    let ctx = sub.rmsnorm(&ctx, &w.get("per_layer_proj_norm.weight")?, cfg.eps)?;
    Ok(tok.broadcast_add(&ctx)?.affine(std::f64::consts::FRAC_1_SQRT_2, 0.0)?)
}

/// Split expert projection routing active experts across both GPUs when TP=2 is engaged.
pub fn split_expert_forward<S: SubstratePort>(
    sub: &S, w: &Weights, pre: &str, n: &Tensor, tp: Option<&DualGpuContext>,
) -> Result<Tensor> {
    if let Some(tp_ctx) = tp {
        if let (Ok(gw), Ok(uw), Ok(dw)) = (w.get_raw(&format!("{pre}.ffn_gate.weight")), w.get_raw(&format!("{pre}.ffn_up.weight")), w.get_raw(&format!("{pre}.ffn_down.weight"))) {
            let (devs, inter) = (tp_ctx.devices(), gw.dim(0)?);
            if inter % 2 == 0 {
                let sh = inter / 2;
                let (gw0, gw1) = (gw.narrow(0, 0, sh)?.to_device(&devs[0])?, gw.narrow(0, sh, sh)?.to_device(&devs[1])?);
                let (uw0, uw1) = (uw.narrow(0, 0, sh)?.to_device(&devs[0])?, uw.narrow(0, sh, sh)?.to_device(&devs[1])?);
                let (dw0, dw1) = (dw.narrow(1, 0, sh)?.to_device(&devs[0])?, dw.narrow(1, sh, sh)?.to_device(&devs[1])?);
                let (n0, n1) = (n.to_device(&devs[0])?, n.to_device(&devs[1])?);
                let d0 = sub.gelu_tanh(&n0.matmul(&gw0.t()?)?)?.broadcast_mul(&n0.matmul(&uw0.t()?)?)?.matmul(&dw0.t()?)?;
                let d1 = sub.gelu_tanh(&n1.matmul(&gw1.t()?)?)?.broadcast_mul(&n1.matmul(&uw1.t()?)?)?.matmul(&dw1.t()?)?;
                let red = crate::tp::ring_all_reduce(&[d0, d1])?;
                return Ok(if red[0].device().same_device(n.device()) { red[0].clone() } else { red[0].to_device(n.device())? });
            }
        }
    }
    let g = w.linear(n, &format!("{pre}.ffn_gate.weight"))?;
    let u = w.linear(n, &format!("{pre}.ffn_up.weight"))?;
    w.linear(&sub.gelu_tanh(&g)?.broadcast_mul(&u)?, &format!("{pre}.ffn_down.weight"))
}

#[allow(clippy::too_many_arguments)]
fn layer<S: SubstratePort>(
    sub: &S, cfg: &ArchConfig, w: &Weights, cache: &mut Cache, i: usize, h: &Tensor, ple: &Tensor, q0: usize,
) -> Result<Tensor> {
    let (lc, pre, dev, t) = (&cfg.layers[i], format!("blk.{i}"), w.device(), h.dim(1)?);
    let n = sub.rmsnorm(h, &w.get(&format!("{pre}.attn_norm.weight"))?, cfg.eps)?;
    let split = |y: Tensor| -> Result<Tensor> {
        let heads = y.dim(2)? / lc.head_dim;
        Ok(y.reshape((1, t, heads, lc.head_dim))?.transpose(1, 2)?)
    };
    let factors = (!lc.is_swa).then_some(cfg.rope_factors.as_deref()).flatten();
    let q = w.linear(&n, &format!("{pre}.attn_q.weight"))?;
    let q = sub.rmsnorm(&split(q)?, &w.get(&format!("{pre}.attn_q_norm.weight"))?, cfg.eps)?;
    let q = sub.rope(&q, q0, lc.rope_theta, lc.rope_dim, factors)?;
    let (k_full, v_full) = if lc.has_kv {
        let k = w.linear(&n, &format!("{pre}.attn_k.weight"))?;
        let k = sub.rmsnorm(&split(k)?, &w.get(&format!("{pre}.attn_k_norm.weight"))?, cfg.eps)?;
        let k = sub.rope(&k, q0, lc.rope_theta, lc.rope_dim, factors)?;
        let v = w.linear(&n, &format!("{pre}.attn_v.weight"))?;
        let v = sub.rmsnorm_plain(&split(v)?, cfg.eps)?;
        cache.append_with_substrate(sub, i, &k, &v, dev)?
    } else {
        let (sk, sv) = cache.get_layer(lc.kv_source)?;
        let sk = if sk.device().same_device(dev) { sk } else { sk.to_device(dev)? };
        let sv = if sv.device().same_device(dev) { sv } else { sv.to_device(dev)? };
        (sk, sv)
    };
    let total = k_full.dim(2)?;
    let window = lc.is_swa.then_some(cfg.sliding_window).flatten();
    let mask = if t == 1 && window.is_none_or(|w| total <= w) { None } else {
        Some(sub.causal_mask(t, total, q0, window, dev)?)
    };
    let scale = cfg.attn_scale.unwrap_or_else(|| (lc.head_dim as f32).sqrt().recip());
    let o = sub.attention(&q, &k_full, &v_full, mask.as_ref(), scale)?;
    let o = o.transpose(1, 2)?.reshape((1, t, lc.n_head * lc.head_dim))?;
    let o = w.linear(&o, &format!("{pre}.attn_output.weight"))?;
    let o = sub.rmsnorm(&o, &w.get(&format!("{pre}.post_attention_norm.weight"))?, cfg.eps)?;
    let h = h.broadcast_add(&o)?;

    // GeGLU block with post norm (with split expert projection for MoE / TP=2).
    let n = sub.rmsnorm(&h, &w.get(&format!("{pre}.ffn_norm.weight"))?, cfg.eps)?;
    let tp_opt = crate::tp::DualGpuContext::try_cuda();
    let mlp = split_expert_forward(sub, w, &pre, &n, tp_opt.as_ref())?;
    let mlp = sub.rmsnorm(&mlp, &w.get(&format!("{pre}.post_ffw_norm.weight"))?, cfg.eps)?;
    let h = h.broadcast_add(&mlp)?;

    // Per-layer embedding injection.
    let gate = sub.gelu_tanh(&w.linear(&h, &format!("{pre}.inp_gate.weight"))?)?;
    let pli = ple.narrow(2, i, 1)?.squeeze(2)?;
    let inj = w.linear(&gate.broadcast_mul(&pli)?, &format!("{pre}.proj.weight"))?;
    let inj = sub.rmsnorm(&inj, &w.get(&format!("{pre}.post_norm.weight"))?, cfg.eps)?;
    let h = h.broadcast_add(&inj)?;
    Ok(h.broadcast_mul(&w.get(&format!("{pre}.layer_output_scale.weight"))?)?)
}

/// Scaled token embeddings (text path's first step, shared by MM).
pub fn input_embeds(cfg: &ArchConfig, w: &Weights, ids: &[u32]) -> Result<Tensor> {
    Ok(w.embed("token_embd.weight", ids)?.affine(cfg.embed_scale as f64, 0.0)?)
}

pub fn forward_embeds(cfg: &ArchConfig, w: &Weights, ids: &[u32]) -> Result<(Tensor, Tensor)> {
    forward_embeds_with_substrate(default_substrate(), cfg, w, ids)
}

pub fn forward_embeds_with_substrate<S: SubstratePort>(
    sub: &S, cfg: &ArchConfig, w: &Weights, ids: &[u32],
) -> Result<(Tensor, Tensor)> {
    let embeds = input_embeds(cfg, w, ids)?;
    let ple = per_layer_inputs_with_substrate(sub, cfg, w, ids, &embeds)?;
    Ok((embeds, ple))
}

pub fn forward_from_embeds(
    cfg: &ArchConfig, w: &Weights, cache: &mut Cache, embeds: &Tensor, ple: &Tensor, q0: usize,
) -> Result<Tensor> {
    forward_from_embeds_with_tp_and_substrate(default_substrate(), cfg, w, cache, embeds, ple, q0, None)
}

pub fn forward_from_embeds_with_substrate<S: SubstratePort>(
    sub: &S, cfg: &ArchConfig, w: &Weights, cache: &mut Cache, embeds: &Tensor, ple: &Tensor, q0: usize,
) -> Result<Tensor> {
    forward_from_embeds_with_tp_and_substrate(sub, cfg, w, cache, embeds, ple, q0, None)
}

#[allow(clippy::too_many_arguments)]
pub fn forward_from_embeds_with_tp_and_substrate<S: SubstratePort>(
    sub: &S, cfg: &ArchConfig, w: &Weights, cache: &mut Cache, embeds: &Tensor, ple: &Tensor, q0: usize, _tp: Option<&DualGpuContext>,
) -> Result<Tensor> {
    let mut h = embeds.clone();
    for i in 0..cfg.n_layer {
        h = layer(sub, cfg, w, cache, i, &h, ple, q0)?;
    }
    h = sub.rmsnorm(&h, &w.get("output_norm.weight")?, cfg.eps)?;
    let mut logits = w.linear(&h, "token_embd.weight")?;
    if let Some(cap) = cfg.final_softcap {
        logits = logits.affine((1.0 / cap) as f64, 0.0)?.tanh()?.affine(cap as f64, 0.0)?;
    }
    Ok(logits)
}

pub fn forward(cfg: &ArchConfig, w: &Weights, cache: &mut Cache, ids: &[u32], q0: usize) -> Result<Tensor> {
    forward_with_tp(cfg, w, cache, ids, q0, None)
}

pub fn forward_with_tp(
    cfg: &ArchConfig, w: &Weights, cache: &mut Cache, ids: &[u32], q0: usize, tp: Option<&DualGpuContext>,
) -> Result<Tensor> {
    forward_with_tp_and_substrate(default_substrate(), cfg, w, cache, ids, q0, tp)
}

pub fn forward_with_substrate<S: SubstratePort>(
    sub: &S, cfg: &ArchConfig, w: &Weights, cache: &mut Cache, ids: &[u32], q0: usize,
) -> Result<Tensor> {
    forward_with_tp_and_substrate(sub, cfg, w, cache, ids, q0, None)
}

pub fn forward_with_tp_and_substrate<S: SubstratePort>(
    sub: &S, cfg: &ArchConfig, w: &Weights, cache: &mut Cache, ids: &[u32], q0: usize, tp: Option<&DualGpuContext>,
) -> Result<Tensor> {
    let (embeds, ple) = forward_embeds_with_substrate(sub, cfg, w, ids)?;
    forward_from_embeds_with_tp_and_substrate(sub, cfg, w, cache, &embeds, &ple, q0, tp)
}

pub fn forward_last_hidden(cfg: &ArchConfig, w: &Weights, cache: &mut Cache, prompt_ids: &[u32]) -> Result<Tensor> {
    forward_last_hidden_with_substrate(default_substrate(), cfg, w, cache, prompt_ids)
}

pub fn forward_last_hidden_with_substrate<S: SubstratePort>(
    sub: &S, cfg: &ArchConfig, w: &Weights, cache: &mut Cache, prompt_ids: &[u32],
) -> Result<Tensor> {
    if prompt_ids.is_empty() {
        return Err(ModelError::Config("cannot score an empty prompt".into()));
    }
    let (embeds, ple) = forward_embeds_with_substrate(sub, cfg, w, prompt_ids)?;
    let mut h = embeds;
    for i in 0..cfg.n_layer {
        h = layer(sub, cfg, w, cache, i, &h, &ple, 0)?;
    }
    let h = sub.rmsnorm(&h, &w.get("output_norm.weight")?, cfg.eps)?;
    let seq = h.dim(1)?;
    let last = h.narrow(1, seq - 1, 1)?.squeeze(1)?;
    Ok(last)
}

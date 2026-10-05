//! Granite decoder: dense IBM Granite 3.0.
//!
//! Per layer: `h += residual_scale * attn(ln1(h))`, `h += residual_scale * mlp(ln2(h))`,
//! with SwiGLU, full causal attention, tied or untied LM head, logit scaling,
//! and embedding scaling.
//! All tensor math operates behind the sovereign `SubstratePort` boundary.

use crate::config::ArchConfig;
use crate::error::Result;
use crate::substrate::{CandleSubstrate, SubstratePort, Tensor};
use crate::weights::Weights;

pub use crate::arch::qwen2::Cache;

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
    let t = n.dim(1)?;
    let split = |y: Tensor| -> Result<Tensor> {
        let heads = y.dim(2)? / lc.head_dim;
        Ok(y.reshape((1, t, heads, lc.head_dim))?.transpose(1, 2)?)
    };
    let q_raw = split(w.linear(&n, &format!("{pre}.attn_q.weight"))?)?;
    let k_raw = split(w.linear(&n, &format!("{pre}.attn_k.weight"))?)?;
    let v_raw = split(w.linear(&n, &format!("{pre}.attn_v.weight"))?)?;
    let q = sub.rope_norm(&q_raw, q0, lc.rope_theta, lc.rope_dim)?;
    let k = sub.rope_norm(&k_raw, q0, lc.rope_theta, lc.rope_dim)?;

    // Extend the KV cache in O(1) contiguous chunked storage and attend over it.
    let (k_full, v_full) = cache.append_with_substrate(sub, i, &k, &v_raw, dev)?;
    let total = k_full.dim(2)?;
    let mask = if t == 1 {
        None
    } else {
        Some(sub.causal_mask(t, total, q0, None, dev)?)
    };
    let scale = cfg.attn_scale.unwrap_or_else(|| (lc.head_dim as f32).sqrt().recip());
    let o = sub.attention(&q, &k_full, &v_full, mask.as_ref(), scale)?;
    let o = o.transpose(1, 2)?.reshape((1, t, lc.n_head * lc.head_dim))?;
    let o = w.linear(&o, &format!("{pre}.attn_output.weight"))?;
    let res_scale = cfg.residual_scale.unwrap_or(0.22) as f64;
    let o = o.affine(res_scale, 0.0)?;
    let h = h.broadcast_add(&o)?;

    // MLP block.
    let n = sub.rmsnorm(&h, &w.get(&format!("{pre}.ffn_norm.weight"))?, cfg.eps)?;
    let g = w.linear(&n, &format!("{pre}.ffn_gate.weight"))?;
    let u = w.linear(&n, &format!("{pre}.ffn_up.weight"))?;
    let mlp = w.linear(
        &sub.silu(&g)?.broadcast_mul(&u)?,
        &format!("{pre}.ffn_down.weight"),
    )?;
    let mlp = mlp.affine(res_scale, 0.0)?;
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
    forward_with_substrate(&CandleSubstrate, cfg, w, cache, ids, q0)
}

/// Logits `[1, seq, vocab]` evaluated through explicit `SubstratePort`.
pub fn forward_with_substrate<S: SubstratePort>(
    sub: &S,
    cfg: &ArchConfig,
    w: &Weights,
    cache: &mut Cache,
    ids: &[u32],
    q0: usize,
) -> Result<Tensor> {
    let mut h = w.embed("token_embd.weight", ids)?;
    if cfg.embed_scale != 1.0 {
        h = h.affine(cfg.embed_scale as f64, 0.0)?;
    }
    for i in 0..cfg.n_layer {
        h = layer(sub, cfg, w, cache, i, &h, q0)?;
    }
    let norm_w = w.get_raw("output_norm.weight")?;
    let out_dev = norm_w.device();
    Weights::ensure_current(out_dev)?;
    let mut h = if !h.device().same_device(out_dev) {
        h.to_device(out_dev)?
    } else {
        h
    };
    h = sub.rmsnorm(&h, &w.get("output_norm.weight")?, cfg.eps)?;
    let mut logits = if w.contains_key("output.weight") {
        w.linear(&h, "output.weight")?
    } else {
        w.linear(&h, "token_embd.weight")?
    };
    if let Some(scale) = cfg.logit_scale {
        if scale != 1.0 {
            logits = logits.affine((1.0 / scale) as f64, 0.0)?;
        }
    }
    Ok(logits)
}

/// Normalized final hidden state `[1, hidden_dim]` for `prompt_ids`.
pub fn forward_last_hidden(
    cfg: &ArchConfig,
    w: &Weights,
    cache: &mut Cache,
    prompt_ids: &[u32],
) -> Result<Tensor> {
    forward_last_hidden_with_substrate(&CandleSubstrate, cfg, w, cache, prompt_ids)
}

/// Normalized final hidden state evaluated through explicit `SubstratePort`.
pub fn forward_last_hidden_with_substrate<S: SubstratePort>(
    sub: &S,
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
    if cfg.embed_scale != 1.0 {
        h = h.affine(cfg.embed_scale as f64, 0.0)?;
    }
    for i in 0..cfg.n_layer {
        h = layer(sub, cfg, w, cache, i, &h, 0)?;
    }
    let norm_w = w.get_raw("output_norm.weight")?;
    let out_dev = norm_w.device();
    Weights::ensure_current(out_dev)?;
    let mut h = if !h.device().same_device(out_dev) {
        h.to_device(out_dev)?
    } else {
        h
    };
    h = sub.rmsnorm(&h, &w.get("output_norm.weight")?, cfg.eps)?;
    let seq = h.dim(1)?;
    let last = h.narrow(1, seq - 1, 1)?.squeeze(1)?;
    Ok(last)
}

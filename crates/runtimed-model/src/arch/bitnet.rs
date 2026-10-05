//! BitNet b1.58 decoder: native 1.58-bit ternary neural architecture.
//!
//! Per layer: Sub-LayerNorm (SubLN), BitLinear Q/K/V projections with RoPE,
//! scaled dot-product attention over O(1) contiguous chunked KV-cache,
//! followed by FFN with squared-ReLU (ReLU^2) and ternary projections.

use crate::config::ArchConfig;
use crate::error::Result;
use crate::substrate::{default_substrate, SubstratePort, Tensor};
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

    // 1. Attention block with SubLN.
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

    // Extend chunked KV-cache and attend.
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
    let h = h.broadcast_add(&o)?;

    // 2. FFN block with squared ReLU activation.
    let n = sub.rmsnorm(&h, &w.get(&format!("{pre}.ffn_norm.weight"))?, cfg.eps)?;
    let g = w.linear(&n, &format!("{pre}.ffn_gate.weight"))?;
    let u = w.linear(&n, &format!("{pre}.ffn_up.weight"))?;

    let act = g.relu()?;
    let act_sq = act.sqr()?;
    let mlp = w.linear(&act_sq.broadcast_mul(&u)?, &format!("{pre}.ffn_down.weight"))?;
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
    forward_with_substrate(default_substrate(), cfg, w, cache, ids, q0)
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
    let logits = if w.contains_key("output.weight") {
        w.linear(&h, "output.weight")?
    } else {
        w.linear(&h, "token_embd.weight")?
    };
    Ok(logits)
}

/// Normalized final hidden state `[1, hidden_dim]` for prompt embeddings.
pub fn forward_last_hidden(
    cfg: &ArchConfig,
    w: &Weights,
    cache: &mut Cache,
    prompt_ids: &[u32],
) -> Result<Tensor> {
    forward_last_hidden_with_substrate(default_substrate(), cfg, w, cache, prompt_ids)
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

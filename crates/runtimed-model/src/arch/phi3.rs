//! Phi-3 / Phi-3.5 / Phi-4 decoder.
//!
//! Per layer: `h += attn(ln1(h))`, `h += mlp(ln2(h))`,
//! with packed QKV projection, fused gate/up FFN, SwiGLU, and full causal attention.
//! All tensor math operates behind the sovereign `SubstratePort` boundary.

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
    let dev = w.device();

    // Attention block with packed QKV.
    let n = sub.rmsnorm(h, &w.get(&format!("{pre}.attn_norm.weight"))?, cfg.eps)?;
    let t = n.dim(1)?;
    let q_dim = lc.n_head * lc.head_dim;
    let k_dim = lc.n_kv * lc.head_dim;
    let v_dim = lc.n_kv * lc.head_dim;

    let qkv = w.linear(&n, &format!("{pre}.attn_qkv.weight"))?;
    let q = qkv.narrow(2, 0, q_dim)?;
    let k = qkv.narrow(2, q_dim, k_dim)?;
    let v = qkv.narrow(2, q_dim + k_dim, v_dim)?;

    let split_q = |y: Tensor| -> Result<Tensor> {
        Ok(y.reshape((1, t, lc.n_head, lc.head_dim))?.transpose(1, 2)?)
    };
    let split_kv = |y: Tensor| -> Result<Tensor> {
        Ok(y.reshape((1, t, lc.n_kv, lc.head_dim))?.transpose(1, 2)?)
    };

    let q = split_q(q)?;
    let k = split_kv(k)?;
    let v = split_kv(v)?;

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
    let scale = cfg.attn_scale.unwrap_or_else(|| (lc.head_dim as f32).sqrt().recip());
    let o = sub.attention(&q, &k_full, &v_full, mask.as_ref(), scale)?;
    let o = o.transpose(1, 2)?.reshape((1, t, lc.n_head * lc.head_dim))?;
    let o = w.linear(&o, &format!("{pre}.attn_output.weight"))?;
    let h = h.broadcast_add(&o)?;

    // MLP block with fused gate/up projection.
    let n = sub.rmsnorm(&h, &w.get(&format!("{pre}.ffn_norm.weight"))?, cfg.eps)?;
    let ffn_up = w.linear(&n, &format!("{pre}.ffn_up.weight"))?;
    let gate = ffn_up.narrow(2, 0, lc.ffn)?;
    let up = ffn_up.narrow(2, lc.ffn, lc.ffn)?;
    let mlp = w.linear(
        &sub.silu(&gate)?.broadcast_mul(&up)?,
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
    for i in 0..cfg.n_layer {
        h = layer(sub, cfg, w, cache, i, &h, q0)?;
    }
    h = sub.rmsnorm(&h, &w.get("output_norm.weight")?, cfg.eps)?;
    if w.contains_key("output.weight") {
        w.linear(&h, "output.weight")
    } else {
        w.linear(&h, "token_embd.weight")
    }
}

/// Normalized final hidden state `[1, hidden_dim]` for `prompt_ids`.
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
    for i in 0..cfg.n_layer {
        h = layer(sub, cfg, w, cache, i, &h, 0)?;
    }
    h = sub.rmsnorm(&h, &w.get("output_norm.weight")?, cfg.eps)?;
    let seq = h.dim(1)?;
    let last = h.narrow(1, seq - 1, 1)?.squeeze(1)?;
    Ok(last)
}

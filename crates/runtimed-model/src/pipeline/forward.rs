//! Multi-device forward pass coordinator with P2P DMA activation handoff.

use super::stage::PipelineStage;
use super::transfer::transfer_activation;
use crate::config::ArchConfig;
use crate::error::{ModelError, Result};
use crate::ops;
use crate::weights::Weights;
use candle_core::Tensor;

/// Execute a multi-device forward pass through an ordered sequence of pipeline stages.
pub fn forward_pipeline(
    stages: &[PipelineStage],
    cfg: &ArchConfig,
    input_ids: &[u32],
    pos: usize,
    mut stage_kv_caches: Option<&mut [crate::cache::PagedKvCache]>,
) -> Result<Tensor> {
    if stages.is_empty() {
        return Err(ModelError::Config("Pipeline has zero stages".into()));
    }
    if input_ids.is_empty() {
        return Err(ModelError::Config("input_ids cannot be empty".into()));
    }

    let mut current_activation: Option<Tensor> = None;

    for (idx, stage) in stages.iter().enumerate() {
        Weights::ensure_current(&stage.device)?;

        // If first stage, embed tokens to create initial hidden state
        let mut h = if stage.is_first || current_activation.is_none() {
            stage.weights.embed("token_embd.weight", input_ids)?
        } else {
            let act = current_activation.take().ok_or_else(|| ModelError::Config("Missing activation".into()))?;
            transfer_activation(&act, &stage.device)?
        };

        // Run layers assigned to this stage
        for layer_idx in stage.start_layer..stage.end_layer {
            let stage_cache = if let Some(ref mut caches) = stage_kv_caches {
                caches.get_mut(idx)
            } else {
                None
            };
            h = execute_stage_layer(
                cfg,
                &stage.weights,
                layer_idx,
                &h,
                pos,
                stage_cache,
                layer_idx.saturating_sub(stage.start_layer),
            )?;
        }

        // If last stage, apply final norm and LM head
        if stage.is_last || idx == stages.len() - 1 {
            let norm_weight = stage.weights.get("output_norm.weight")?;
            h = ops::rms_norm(&h, &norm_weight, cfg.eps)?;
            let logits = stage.weights.linear(&h, "output.weight")?;
            return Ok(logits);
        }

        current_activation = Some(h);
    }

    Err(ModelError::Config("Pipeline completed without reaching final stage".into()))
}

fn execute_stage_layer(
    cfg: &ArchConfig,
    w: &Weights,
    i: usize,
    h: &Tensor,
    q0: usize,
    stage_cache: Option<&mut crate::cache::PagedKvCache>,
    local_layer: usize,
) -> Result<Tensor> {
    let lc = &cfg.layers[i];
    let pre = format!("blk.{i}");
    let dev = w.device();

    // Attention block
    let norm_w = w.get(&format!("{pre}.attn_norm.weight"))?;
    let n = ops::rms_norm(h, &norm_w, cfg.eps)?;
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
    let mut k = split(w.linear_bias(&n, &format!("{pre}.attn_k.weight"), bk.as_deref())?)?;
    let mut v = split(w.linear_bias(&n, &format!("{pre}.attn_v.weight"), bv.as_deref())?)?;
    let q = ops::rope_neox(&q, q0, lc.rope_theta, lc.rope_dim, None)?;
    k = ops::rope_neox(&k, q0, lc.rope_theta, lc.rope_dim, None)?;

    if let Some(cache) = stage_cache {
        let cl = if local_layer < cache.n_layer { local_layer } else { i };
        cache.append_kv(cl, &k, &v, dev)?;
        if let Some((ak, av)) = cache.assemble_layer_kv(cl, dev)? {
            k = ak;
            v = av;
        }
    }

    let total = k.dim(2)?;
    let mask = ops::causal_mask(t, total, q0, None, dev)?;
    let scale = (lc.head_dim as f32).sqrt().recip();
    let o = ops::attention(&q, &k, &v, &mask, scale)?;
    let o = o.transpose(1, 2)?.reshape((1, t, lc.n_head * lc.head_dim))?;
    let o = w.linear(&o, &format!("{pre}.attn_output.weight"))?;
    let h = h.broadcast_add(&o)?;

    // MLP block
    let ffn_norm = w.get(&format!("{pre}.ffn_norm.weight"))?;
    let n = ops::rms_norm(&h, &ffn_norm, cfg.eps)?;
    let g = w.linear(&n, &format!("{pre}.ffn_gate.weight"))?;
    let u = w.linear(&n, &format!("{pre}.ffn_up.weight"))?;
    let mlp = w.linear(
        &ops::silu(&g)?.broadcast_mul(&u)?,
        &format!("{pre}.ffn_down.weight"),
    )?;
    Ok(h.broadcast_add(&mlp)?)
}

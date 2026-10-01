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

        // Initialize JIT layer prefetcher for this stage
        let mut prefetcher = crate::cache::JitLayerPrefetcher::new(stage.device.clone());

        // Run layers assigned to this stage
        for layer_idx in stage.start_layer..stage.end_layer {
            let local_layer = layer_idx.saturating_sub(stage.start_layer);
            let next_prefetch = if layer_idx + 1 < stage.end_layer {
                Some(local_layer + 1)
            } else {
                None
            };
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
                local_layer,
                &mut prefetcher,
                next_prefetch,
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
    prefetcher: &mut crate::cache::JitLayerPrefetcher,
    next_prefetch: Option<usize>,
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
        let staged = prefetcher.acquire_active(cl);
        cache.append_kv(cl, &k, &v, dev)?;
        if let Some(s) = staged {
            if s.k.dim(2)? > 0 {
                if k.dim(2)? > 0 {
                    k = Tensor::cat(&[&s.k, &k], 2)?;
                    v = Tensor::cat(&[&s.v, &v], 2)?;
                } else {
                    k = s.k;
                    v = s.v;
                }
            }
        } else if let Some((ak, av)) = cache.assemble_layer_kv(cl, dev)? {
            k = ak;
            v = av;
        }

        // Prefetch (l + 1)-th layer KV tensors concurrently while computing l-th layer attention
        if let Some(next_layer) = next_prefetch {
            let next_cl = if next_layer < cache.n_layer { next_layer } else { i + 1 };
            if next_cl < cache.n_layer {
                let _ = prefetcher.prefetch_from_cache(cache, next_cl);
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_forward_pipeline_empty_validation() {
        let cfg = ArchConfig {
            arch: crate::config::Arch::Qwen2,
            n_layer: 2,
            hidden: 64,
            vocab: 100,
            eps: 1e-5,
            act: crate::config::Activation::Silu,
            tie_lm_head: false,
            has_qkv_bias: false,
            embed_scale: 1.0,
            final_softcap: None,
            attn_scale: None,
            sliding_window: None,
            rope_factors: None,
            ple_dim: 0,
            layers: vec![],
        };
        let err = forward_pipeline(&[], &cfg, &[1, 2], 0, None).unwrap_err();
        assert!(matches!(err, ModelError::Config(_)));

        let dummy_w = Arc::new(Weights::from_parts(Device::Cpu, DType::F32, HashMap::new()));
        let stages = vec![PipelineStage::new(0, 1, 0, 1, 1, Device::Cpu, dummy_w)];
        let err_empty_ids = forward_pipeline(&stages, &cfg, &[], 0, None).unwrap_err();
        assert!(matches!(err_empty_ids, ModelError::Config(_)));
    }

    #[test]
    fn test_forward_pipeline_with_layer_prefetch_and_cache() {
        let lc = crate::config::LayerConfig {
            n_head: 2, n_kv: 2, head_dim: 32, ffn: 64, is_swa: false,
            has_kv: true, rope_theta: 10000.0, rope_dim: 32, kv_source: 0,
        };
        let cfg = ArchConfig {
            arch: crate::config::Arch::Qwen2, n_layer: 2, hidden: 64, vocab: 50, eps: 1e-5,
            act: crate::config::Activation::Silu, tie_lm_head: false, has_qkv_bias: false,
            embed_scale: 1.0, final_softcap: None, attn_scale: None, sliding_window: None,
            rope_factors: None, ple_dim: 0, layers: vec![lc.clone(), lc],
        };
        let mut map = HashMap::new();
        map.insert("token_embd.weight".into(), Tensor::zeros((50, 64), DType::F32, &Device::Cpu).unwrap());
        map.insert("output_norm.weight".into(), Tensor::ones(64, DType::F32, &Device::Cpu).unwrap());
        map.insert("output.weight".into(), Tensor::zeros((50, 64), DType::F32, &Device::Cpu).unwrap());
        for i in 0..2 {
            let p = format!("blk.{i}");
            map.insert(format!("{p}.attn_norm.weight"), Tensor::ones(64, DType::F32, &Device::Cpu).unwrap());
            map.insert(format!("{p}.attn_q.weight"), Tensor::zeros((64, 64), DType::F32, &Device::Cpu).unwrap());
            map.insert(format!("{p}.attn_k.weight"), Tensor::zeros((64, 64), DType::F32, &Device::Cpu).unwrap());
            map.insert(format!("{p}.attn_v.weight"), Tensor::zeros((64, 64), DType::F32, &Device::Cpu).unwrap());
            map.insert(format!("{p}.attn_output.weight"), Tensor::zeros((64, 64), DType::F32, &Device::Cpu).unwrap());
            map.insert(format!("{p}.ffn_norm.weight"), Tensor::ones(64, DType::F32, &Device::Cpu).unwrap());
            map.insert(format!("{p}.ffn_gate.weight"), Tensor::zeros((64, 64), DType::F32, &Device::Cpu).unwrap());
            map.insert(format!("{p}.ffn_up.weight"), Tensor::zeros((64, 64), DType::F32, &Device::Cpu).unwrap());
            map.insert(format!("{p}.ffn_down.weight"), Tensor::zeros((64, 64), DType::F32, &Device::Cpu).unwrap());
        }
        let w = Arc::new(Weights::from_parts(Device::Cpu, DType::F32, map));
        let stages = vec![PipelineStage::new(0, 1, 0, 2, 2, Device::Cpu, w)];
        let mut cache = crate::cache::PagedKvCache::new(2);
        let k0 = Tensor::zeros((1, 2, 2, 32), DType::F32, &Device::Cpu).unwrap();
        let v0 = Tensor::zeros((1, 2, 2, 32), DType::F32, &Device::Cpu).unwrap();
        cache.allocate_block(&Device::Cpu, crate::cache::StorageTier::L2PinnedHost, k0, v0, 2);
        cache.layer_tables[1].push(0);

        let out = forward_pipeline(&stages, &cfg, &[5], 0, Some(&mut [cache])).unwrap();
        assert_eq!(out.dims(), &[1, 1, 50]);
    }
}

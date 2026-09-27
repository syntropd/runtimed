//! Arch configs parsed from GGUF metadata + tensor shapes.
//!
//! Tensor shapes are authoritative for per-layer dims (head dim, FFN size);
//! metadata carries flags, patterns, and hyper-parameters. Anything the
//! file does not say (activation fn per arch) is hardcoded with a comment.

use crate::error::{ModelError, Result};
use runtimed_gguf::{GgufFile, MetaValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Qwen2,
    Gemma4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Qwen2 SwiGLU: `silu(gate) * up`.
    Silu,
    /// Gemma4 GeGLU: `gelu_tanh(gate) * up`.
    GeluTanh,
}

/// Everything the forward pass needs for one layer.
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub is_swa: bool,
    /// False on Gemma4 KV-shared layers (15+): reuse the source layer's KV.
    pub has_kv: bool,
    pub rope_theta: f32,
    /// RoPE rotation dims (`n_rot`); the head tail passes through.
    pub rope_dim: usize,
    /// For shared layers: index of the layer whose KV cache to read.
    pub kv_source: usize,
}

#[derive(Debug, Clone)]
pub struct ArchConfig {
    pub arch: Arch,
    pub n_layer: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    pub act: Activation,
    /// True when the file has no `output.weight` (head reuses embeddings).
    pub tie_lm_head: bool,
    pub has_qkv_bias: bool,
    /// Embedding multiplier: sqrt(hidden) on Gemma4, 1.0 on Qwen2.
    pub embed_scale: f32,
    /// Final `cap * tanh(x / cap)`; None disables.
    pub final_softcap: Option<f32>,
    /// Attention score multiplier; None means `1/sqrt(head_dim)`.
    pub attn_scale: Option<f32>,
    pub sliding_window: Option<usize>,
    /// Gemma4 global-layer RoPE frequency divisors (`rope_freqs.weight`).
    pub rope_factors: Option<Vec<f32>>,
    /// Per-layer embedding dim; 0 disables the PLE path.
    pub ple_dim: usize,
    pub layers: Vec<LayerConfig>,
}

fn meta_u32(file: &GgufFile, key: &str) -> Result<u32> {
    match file.metadata.get(key) {
        Some(MetaValue::U32(v)) => Ok(*v),
        Some(MetaValue::U64(v)) => Ok(u32::try_from(*v).map_err(|_| bad(key))?),
        Some(MetaValue::I32(v)) => Ok(u32::try_from(*v).map_err(|_| bad(key))?),
        other => Err(bad(&format!("{key} missing, got {other:?}"))),
    }
}

fn meta_f32(file: &GgufFile, key: &str) -> Result<f32> {
    match file.metadata.get(key) {
        Some(MetaValue::F32(v)) => Ok(*v),
        Some(MetaValue::F64(v)) => Ok(*v as f32),
        other => Err(bad(&format!("{key} missing, got {other:?}"))),
    }
}

fn meta_bool_arr(file: &GgufFile, key: &str, n: usize) -> Result<Vec<bool>> {
    match file.metadata.get(key) {
        Some(MetaValue::Arr(items)) if items.len() == n => items
            .iter()
            .map(|v| match v {
                MetaValue::Bool(b) => Ok(*b),
                other => Err(bad(&format!("{key} non-bool {other:?}"))),
            })
            .collect(),
        other => Err(bad(&format!("{key} missing/short, got {other:?}"))),
    }
}

fn bad(msg: &str) -> ModelError {
    ModelError::Config(msg.to_string())
}

/// `ne1` of a `[in, out]` GGUF weight (dims are `[fast, slow]`).
fn out_dim(file: &GgufFile, name: &str) -> Result<usize> {
    let t = file
        .tensor(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    if t.dims.len() != 2 {
        return Err(bad(&format!("{name} not 2-D: {:?}", t.dims)));
    }
    Ok(t.dims[1] as usize)
}

impl ArchConfig {
    pub fn parse(file: &GgufFile) -> Result<Self> {
        let arch_tag = file.meta_str("general.architecture").unwrap_or("?");
        let arch = match arch_tag {
            "qwen2" => Arch::Qwen2,
            "gemma4" => Arch::Gemma4,
            other => return Err(ModelError::Arch(other.to_string())),
        };
        let p = arch_tag; // metadata key prefix matches the arch tag
        let n_layer = meta_u32(file, &format!("{p}.block_count"))? as usize;
        let hidden = meta_u32(file, &format!("{p}.embedding_length"))? as usize;
        let n_head = meta_u32(file, &format!("{p}.attention.head_count"))? as usize;
        let n_kv = meta_u32(file, &format!("{p}.attention.head_count_kv"))? as usize;
        let eps = meta_f32(file, &format!("{p}.attention.layer_norm_rms_epsilon"))?;
        let rope_theta = meta_f32(file, &format!("{p}.rope.freq_base"))?;

        // Vocabulary: embedding rows win over any metadata guess.
        let token_embd = file
            .tensor("token_embd.weight")
            .ok_or_else(|| ModelError::MissingWeight("token_embd.weight".into()))?;
        let vocab = token_embd.dims[1] as usize;
        let tie_lm_head = file.tensor("output.weight").is_none();
        let has_qkv_bias = file.tensor("blk.0.attn_q.bias").is_some();

        let (act, embed_scale, final_softcap, attn_scale, sliding_window, rope_factors, ple_dim);
        let mut layers = Vec::with_capacity(n_layer);
        match arch {
            Arch::Qwen2 => {
                act = Activation::Silu;
                embed_scale = 1.0;
                final_softcap = None;
                attn_scale = None;
                sliding_window = None;
                rope_factors = None;
                ple_dim = 0;
                let head_dim = hidden / n_head;
                let ffn = meta_u32(file, "qwen2.feed_forward_length")? as usize;
                for _ in 0..n_layer {
                    layers.push(LayerConfig {
                        n_head,
                        n_kv,
                        head_dim,
                        ffn,
                        is_swa: false,
                        has_kv: true,
                        rope_theta,
                        rope_dim: head_dim,
                        kv_source: 0,
                    });
                }
            }
            Arch::Gemma4 => {
                act = Activation::GeluTanh;
                embed_scale = (hidden as f32).sqrt();
                let cap = meta_f32(file, "gemma4.final_logit_softcapping").ok();
                final_softcap = cap.filter(|c| *c > 0.0);
                attn_scale = Some(1.0);
                let swa = meta_u32(file, "gemma4.attention.sliding_window")? as usize;
                sliding_window = Some(swa);
                let pattern = meta_bool_arr(file, "gemma4.attention.sliding_window_pattern", n_layer)?;
                let n_shared = file
                    .metadata
                    .get("gemma4.attention.shared_kv_layers")
                    .and_then(|v| match v {
                        MetaValue::U32(x) => Some(*x as usize),
                        _ => None,
                    })
                    .unwrap_or(0);
                let shared_from = n_layer.saturating_sub(n_shared);
                let ple = meta_u32(file, "gemma4.embedding_length_per_layer_input")
                    .map(|v| v as usize)
                    .unwrap_or(0);
                ple_dim = ple;
                let rot_full = meta_u32(file, "gemma4.rope.dimension_count")? as usize;
                let rot_swa = meta_u32(file, "gemma4.rope.dimension_count_swa")? as usize;
                let theta_swa = meta_f32(file, "gemma4.rope.freq_base_swa")?;
                rope_factors = file
                    .tensor("rope_freqs.weight")
                    .map(|t| file.tensor_f32(t))
                    .transpose()?;
                for i in 0..n_layer {
                    // Authoritative per-layer dims come from the weights.
                    let q_out = out_dim(file, &format!("blk.{i}.attn_q.weight"))?;
                    let head_dim = q_out / n_head;
                    let ffn = out_dim(file, &format!("blk.{i}.ffn_gate.weight"))?;
                    let is_swa = pattern[i];
                    let has_kv = i < shared_from;
                    // Shared layers read the last KV-owning layer of same type.
                    let kv_source = if has_kv {
                        i
                    } else {
                        (0..shared_from).rev().find(|&j| pattern[j] == is_swa).ok_or_else(|| {
                            bad(&format!("layer {i}: no KV source of same type"))
                        })?
                    };
                    layers.push(LayerConfig {
                        n_head,
                        n_kv,
                        head_dim,
                        ffn,
                        is_swa,
                        has_kv,
                        rope_theta: if is_swa { theta_swa } else { rope_theta },
                        rope_dim: if is_swa { rot_swa } else { rot_full },
                        kv_source,
                    });
                }
            }
        }
        if tie_lm_head && arch == Arch::Qwen2 {
            return Err(bad("qwen2 without output.weight: refusing to guess"));
        }
        Ok(Self {
            arch,
            n_layer,
            hidden,
            vocab,
            eps,
            act,
            tie_lm_head,
            has_qkv_bias,
            embed_scale,
            final_softcap,
            attn_scale,
            sliding_window,
            rope_factors,
            ple_dim,
            layers,
        })
    }
}

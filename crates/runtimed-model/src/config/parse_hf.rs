//! Hugging Face config.json parser into ArchConfig.

use super::types::{Activation, Arch, ArchConfig, LayerConfig};
use crate::error::{ModelError, Result};
use serde_json::Value;
use std::fs;
use std::path::Path;

pub fn parse_hf_file(path: &Path) -> Result<ArchConfig> {
    let content = fs::read_to_string(path)?;
    parse_hf_config(&content)
}

pub fn parse_hf_config(json_str: &str) -> Result<ArchConfig> {
    let v: Value = serde_json::from_str(json_str)
        .map_err(|e| ModelError::Config(format!("invalid json config: {e}")))?;

    let model_type = v.get("model_type").and_then(Value::as_str).unwrap_or("");
    let arch_name = v
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .unwrap_or("");

    let arch = if model_type.eq_ignore_ascii_case("qwen2") || arch_name.contains("Qwen2") {
        Arch::Qwen2
    } else if model_type.starts_with("gemma") || arch_name.contains("Gemma") {
        Arch::Gemma4
    } else {
        return Err(ModelError::Arch(format!(
            "unknown or unsupported HF architecture: model_type={model_type}, arch={arch_name}"
        )));
    };

    let n_layer = v
        .get("num_hidden_layers")
        .and_then(Value::as_u64)
        .ok_or_else(|| ModelError::Config("missing num_hidden_layers".into()))?
        as usize;
    let hidden = v
        .get("hidden_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| ModelError::Config("missing hidden_size".into()))?
        as usize;
    let n_head = v
        .get("num_attention_heads")
        .and_then(Value::as_u64)
        .ok_or_else(|| ModelError::Config("missing num_attention_heads".into()))?
        as usize;
    let n_kv = v
        .get("num_key_value_heads")
        .and_then(Value::as_u64)
        .map(|k| k as usize)
        .unwrap_or(n_head);
    let head_dim = v
        .get("head_dim")
        .and_then(Value::as_u64)
        .map(|h| h as usize)
        .unwrap_or_else(|| hidden.checked_div(n_head).unwrap_or(0));
    let ffn = v
        .get("intermediate_size")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let vocab = v.get("vocab_size").and_then(Value::as_u64).unwrap_or(0) as usize;
    let eps = v
        .get("rms_norm_eps")
        .or_else(|| v.get("layer_norm_epsilon"))
        .and_then(Value::as_f64)
        .unwrap_or(1e-6) as f32;
    let rope_theta = v
        .get("rope_theta")
        .and_then(Value::as_f64)
        .unwrap_or(10_000.0) as f32;
    let tie_lm_head = v
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let (act, embed_scale, final_softcap, attn_scale, sliding_window, has_qkv_bias);
    let mut layers = Vec::with_capacity(n_layer);

    match arch {
        Arch::Qwen2 => {
            act = Activation::Silu;
            embed_scale = 1.0;
            final_softcap = None;
            attn_scale = None;
            sliding_window = v.get("sliding_window").and_then(Value::as_u64).map(|s| s as usize);
            has_qkv_bias = v.get("attention_bias").and_then(Value::as_bool).unwrap_or(true);
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
            let cap = v
                .get("final_logit_softcapping")
                .and_then(Value::as_f64)
                .map(|c| c as f32);
            final_softcap = cap.filter(|&c| c > 0.0);
            attn_scale = Some(1.0);
            let swa = v.get("sliding_window").and_then(Value::as_u64).map(|s| s as usize);
            sliding_window = swa;
            has_qkv_bias = false;
            for i in 0..n_layer {
                let is_swa = sliding_window.is_some() && (i % 2 == 0);
                layers.push(LayerConfig {
                    n_head,
                    n_kv,
                    head_dim,
                    ffn,
                    is_swa,
                    has_kv: true,
                    rope_theta,
                    rope_dim: head_dim,
                    kv_source: i,
                });
            }
        }
    }

    Ok(ArchConfig {
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
        rope_factors: None,
        ple_dim: 0,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen2_hf_config() {
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true,
            "vocab_size": 151936
        }"#;
        let cfg = parse_hf_config(json).expect("parse qwen2 hf");
        assert_eq!(cfg.arch, Arch::Qwen2);
        assert_eq!(cfg.n_layer, 24);
        assert_eq!(cfg.hidden, 896);
        assert_eq!(cfg.vocab, 151936);
        assert!(cfg.tie_lm_head);
        assert_eq!(cfg.layers.len(), 24);
        assert_eq!(cfg.layers[0].n_head, 14);
        assert_eq!(cfg.layers[0].n_kv, 2);
    }

    #[test]
    fn parses_gemma_hf_config() {
        let json = r#"{
            "architectures": ["Gemma2ForCausalLM"],
            "model_type": "gemma2",
            "hidden_size": 2304,
            "intermediate_size": 9216,
            "num_attention_heads": 8,
            "num_hidden_layers": 26,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "vocab_size": 256000,
            "final_logit_softcapping": 30.0
        }"#;
        let cfg = parse_hf_config(json).expect("parse gemma hf");
        assert_eq!(cfg.arch, Arch::Gemma4);
        assert_eq!(cfg.n_layer, 26);
        assert_eq!(cfg.final_softcap, Some(30.0));
        assert_eq!(cfg.layers[0].head_dim, 256);
    }
}

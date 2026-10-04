//! Hugging Face to GGUF tensor name mapping.
//!
//! Translates standard Hugging Face transformer tensor names
//! (`model.layers.0.self_attn.q_proj.weight`) to the canonical internal GGUF names
//! (`blk.0.attn_q.weight`) used by runtimed architectures.

/// Maps a Hugging Face tensor name to canonical internal GGUF format.
/// Returns the translated name or the original name if no mapping matched.
pub fn map_hf_tensor_name(name: &str) -> String {
    let clean = name.strip_prefix("base_model.model.").unwrap_or(name);
    match clean {
        "model.embed_tokens.weight" => return "token_embd.weight".to_string(),
        "model.norm.weight" => return "output_norm.weight".to_string(),
        "lm_head.weight" => return "output.weight".to_string(),
        _ => {}
    }

    if let Some(rest) = clean.strip_prefix("model.layers.") {
        if let Some((idx_str, sub)) = rest.split_once('.') {
            if let Ok(layer_idx) = idx_str.parse::<usize>() {
                if let Some(mapped_sub) = map_layer_submodule(sub) {
                    return format!("blk.{layer_idx}.{mapped_sub}");
                }
            }
        }
    }

    clean.to_string()
}

fn map_layer_submodule(sub: &str) -> Option<String> {
    let (stem, suffix) = if let Some(s) = sub.strip_suffix(".weight") {
        (s, "weight")
    } else if let Some(s) = sub.strip_suffix(".bias") {
        (s, "bias")
    } else if let Some(s) = sub.strip_suffix(".weight_scale") {
        (s, "weight.scale")
    } else if let Some(s) = sub.strip_suffix(".weight_scale_inv") {
        (s, "weight.scale")
    } else if let Some(s) = sub.strip_suffix(".scale") {
        (s, "weight.scale")
    } else if let Some(s) = sub.strip_suffix(".scales") {
        (s, "scales")
    } else if let Some(s) = sub.strip_suffix(".marlin_scales") {
        (s, "marlin_scales")
    } else if let Some(s) = sub.strip_suffix(".marlin_packed") {
        (s, "marlin_packed")
    } else {
        (sub, "weight")
    };

    let mapped_stem = match stem {
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        "input_layernorm" => "attn_norm",
        "post_attention_layernorm" | "pre_feedforward_layernorm" => "ffn_norm",
        _ => return None,
    };

    Some(format!("{mapped_stem}.{suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_standard_hf_tensors() {
        assert_eq!(
            map_hf_tensor_name("model.embed_tokens.weight"),
            "token_embd.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.norm.weight"),
            "output_norm.weight"
        );
        assert_eq!(map_hf_tensor_name("lm_head.weight"), "output.weight");
        assert_eq!(
            map_hf_tensor_name("model.layers.3.self_attn.q_proj.weight"),
            "blk.3.attn_q.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.layers.0.mlp.down_proj.bias"),
            "blk.0.ffn_down.bias"
        );
        assert_eq!(
            map_hf_tensor_name("base_model.model.model.layers.1.mlp.gate_proj.weight"),
            "blk.1.ffn_gate.weight"
        );
    }
}

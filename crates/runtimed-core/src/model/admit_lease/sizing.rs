//! Dynamic lease sizing computing exact memory requirements for model execution.

use runtimed_model::config::ArchConfig;

/// Activation workspace headroom reserved for forward intermediate tensors (128 MiB).
pub const ACTIVATION_HEADROOM_BYTES: u64 = 128 * 1024 * 1024;

/// Initial attention sink token threshold kept in FP16 to avoid numerical drift.
pub const SINK_TOKENS_THRESHOLD: usize = 16;

/// Computes total exact lease bytes required for model resident execution:
/// `weights_file_bytes + exact_kv_cache_bytes + 128 MiB activation headroom`.
pub fn compute_lease_bytes(
    weights_file_bytes: u64,
    cfg: &ArchConfig,
    context_tokens: usize,
) -> u64 {
    let kv_bytes = compute_kv_cache_bytes(cfg, context_tokens);
    weights_file_bytes
        .saturating_add(kv_bytes)
        .saturating_add(ACTIVATION_HEADROOM_BYTES)
}

/// Computes exact KV cache allocation in bytes for a given architecture config
/// and context token count, accounting for 16 FP16 sink tokens and FP8 subsequent tokens.
pub fn compute_kv_cache_bytes(cfg: &ArchConfig, context_tokens: usize) -> u64 {
    let kv_dim_sum: usize = cfg
        .layers
        .iter()
        .filter(|l| l.has_kv)
        .map(|l| l.n_kv * l.head_dim)
        .sum::<usize>();

    let sink_tokens = context_tokens.min(SINK_TOKENS_THRESHOLD);
    let fp8_tokens = context_tokens.saturating_sub(SINK_TOKENS_THRESHOLD);

    // K and V tensors: 2 tensors.
    // Sink tokens are FP16 (2 bytes per element).
    // Subsequent tokens are FP8 (1 byte per element).
    let sink_bytes = (sink_tokens * kv_dim_sum * 2 * 2) as u64;
    let fp8_bytes = (fp8_tokens * kv_dim_sum * 2) as u64;

    sink_bytes + fp8_bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtimed_model::config::{Activation, Arch, LayerConfig};

    fn make_test_cfg(n_layers: usize, n_kv: usize, head_dim: usize, shared_kv: bool) -> ArchConfig {
        let mut layers = Vec::new();
        for i in 0..n_layers {
            let has_kv = !shared_kv || i < n_layers / 2;
            layers.push(LayerConfig {
                n_head: 8,
                n_kv,
                head_dim,
                ffn: 1024,
                is_swa: false,
                has_kv,
                rope_theta: 10000.0,
                rope_dim: head_dim,
                kv_source: 0,
            });
        }
        ArchConfig {
            arch: Arch::Qwen2,
            n_layer: n_layers,
            hidden: 512,
            vocab: 32000,
            eps: 1e-5,
            act: Activation::Silu,
            tie_lm_head: false,
            has_qkv_bias: false,
            embed_scale: 1.0,
            final_softcap: None,
            attn_scale: None,
            sliding_window: Some(4096),
            rope_factors: None,
            ple_dim: 0,
            residual_scale: None,
            logit_scale: None,
            layers,
        }
    }

    #[test]
    fn test_compute_lease_bytes_basic() {
        // 4 layers, 2 kv heads, 64 head_dim => kv_dim_sum = 4 * 2 * 64 = 512
        let cfg = make_test_cfg(4, 2, 64, false);
        let weights = 1_000_000_000u64;

        // 0 context tokens: weights + 0 + 128 MiB
        let lease_0 = compute_lease_bytes(weights, &cfg, 0);
        assert_eq!(lease_0, weights + ACTIVATION_HEADROOM_BYTES);

        // 16 sink tokens: 16 * 512 * 2 * 2 = 32,768 bytes
        let kv_16 = compute_kv_cache_bytes(&cfg, 16);
        assert_eq!(kv_16, 32768);
        assert_eq!(compute_lease_bytes(weights, &cfg, 16), weights + 32768 + ACTIVATION_HEADROOM_BYTES);

        // 32 tokens: 16 sink (32,768) + 16 fp8 (16 * 512 * 2 * 1 = 16,384) = 49,152 bytes
        let kv_32 = compute_kv_cache_bytes(&cfg, 32);
        assert_eq!(kv_32, 32768 + 16384);
    }

    #[test]
    fn test_compute_lease_bytes_shared_kv_layers() {
        // 4 layers with shared_kv=true => only 2 layers have KV
        // kv_dim_sum = 2 * 2 * 64 = 256
        let cfg = make_test_cfg(4, 2, 64, true);
        let kv_16 = compute_kv_cache_bytes(&cfg, 16);
        // 16 * 256 * 2 * 2 = 16,384 bytes
        assert_eq!(kv_16, 16384);
    }
}

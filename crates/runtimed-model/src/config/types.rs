//! Architectural types and per-layer configurations.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Qwen2,
    Gemma4,
    Granite,
    Phi3,
    BitNet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Qwen2 / Granite / Phi3 SwiGLU: `silu(gate) * up`.
    Silu,
    /// Gemma4 GeGLU: `gelu_tanh(gate) * up`.
    GeluTanh,
    /// BitNet b1.58 squared ReLU: `relu(gate)^2 * up`.
    Relu2,
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
    /// Embedding multiplier: sqrt(hidden) on Gemma4, 1.0 on Qwen2, 12.0 on Granite.
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
    /// Residual scale for Granite models (typically 0.22); None disables.
    pub residual_scale: Option<f32>,
    /// Final logit divisor for Granite models (typically 16.0); None disables.
    pub logit_scale: Option<f32>,
    pub layers: Vec<LayerConfig>,
}

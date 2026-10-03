//! The engine's math: RMSNorm, NeoX RoPE, masked attention, activations.
//!
//! Batch is always 1. Sequences are `[1, time, hidden]`, heads are
//! `[1, heads, time, head_dim]`. RoPE tables and masks are built on the
//! CPU as plain `f32` so the formulas stay reviewable; candle runs the
//! matmuls, norms, and softmax.

use crate::error::Result;
use candle_core::{Device, Tensor};

/// `x / sqrt(mean(x^2) + eps) * w`, weight on the last axis.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    let m2 = x.broadcast_mul(x)?.mean_keepdim(candle_core::D::Minus1)?;
    let denom = m2.broadcast_add(&Tensor::new(eps, x.device())?)?.sqrt()?;
    Ok(x.broadcast_div(&denom)?.broadcast_mul(w)?)
}

/// RMSNorm without a weight (Gemma4 value norm).
pub fn rms_norm_plain(x: &Tensor, eps: f32) -> Result<Tensor> {
    let m2 = x.broadcast_mul(x)?.mean_keepdim(candle_core::D::Minus1)?;
    let denom = m2.broadcast_add(&Tensor::new(eps, x.device())?)?.sqrt()?;
    Ok(x.broadcast_div(&denom)?)
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    let neg = x.affine(-1.0, 0.0)?;
    let sig = neg.exp()?.affine(1.0, 1.0)?.recip()?;
    Ok(x.broadcast_mul(&sig)?)
}

/// Numerically stable softmax over the last axis.
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let e = x.broadcast_sub(&x.max_keepdim(candle_core::D::Minus1)?)?.exp()?;
    Ok(e.broadcast_div(&e.sum_keepdim(candle_core::D::Minus1)?)?)
}

/// Tanh-approximation GELU (HF `gelu_pytorch_tanh`, ggml `ggml_gelu`).
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    let c = (2.0f64 / std::f64::consts::PI).sqrt();
    let x3 = x.broadcast_mul(x)?.broadcast_mul(x)?;
    let inner = x3.affine(0.044715, 0.0)?.broadcast_add(x)?.affine(c, 0.0)?;
    let tanh = inner.tanh()?;
    let one_plus = tanh.affine(1.0, 1.0)?;
    Ok(x.broadcast_mul(&one_plus)?.affine(0.5, 0.0)?)
}

/// Quick-approximation GELU (ggml `ggml_gelu_quick`): x * sigmoid(1.702x).
/// Used by the Gemma4 vision tower FFN (mmproj sets neither use_gelu
/// nor use_silu, so the reference falls back to gelu_quick).
pub fn gelu_quick(x: &Tensor) -> Result<Tensor> {
    let neg = x.affine(-1.702, 0.0)?;
    let sig = neg.exp()?.affine(1.0, 1.0)?.recip()?;
    Ok(x.broadcast_mul(&sig)?)
}

/// NeoX (half-rotation) RoPE over the first `rot_dim` of each head.
///
/// `angle(pos, i) = pos * theta^(-2i/rot_dim) / factor[i]`; dims past
/// `rot_dim` pass through untouched. `factors` is None for plain RoPE.
pub fn rope_neox(
    x: &Tensor,
    pos_start: usize,
    theta: f32,
    rot_dim: usize,
    factors: Option<&[f32]>,
) -> Result<Tensor> {
    let t = x.dim(2)?;
    let positions: Vec<usize> = (pos_start..pos_start + t).collect();
    rope_neox_pos(x, &positions, theta, rot_dim, factors)
}

/// NeoX RoPE with explicit per-row positions (vision axial rope needs
/// patch coordinates, not sequence positions).
pub fn rope_neox_pos(
    x: &Tensor,
    positions: &[usize],
    theta: f32,
    rot_dim: usize,
    factors: Option<&[f32]>,
) -> Result<Tensor> {
    let dev = x.device();
    let (_b, _h, t, d) = x.dims4()?;
    assert!(rot_dim <= d && rot_dim.is_multiple_of(2), "bad rot_dim {rot_dim} for {d}");
    assert_eq!(positions.len(), t, "one position per row");
    let half = rot_dim / 2;
    let mut cos = Vec::with_capacity(t * half);
    let mut sin = Vec::with_capacity(t * half);
    for &pos in positions {
        let p = pos as f32;
        for i in 0..half {
            let inv = theta.powf(-2.0 * i as f32 / rot_dim as f32);
            let f = factors.map(|f| f[i]).unwrap_or(1.0);
            let angle = p * inv / f;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let cos = Tensor::from_vec(cos, (1, 1, t, half), dev)?;
    let sin = Tensor::from_vec(sin, (1, 1, t, half), dev)?;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let o1 = x1.broadcast_mul(&cos)?.broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let o2 = x2.broadcast_mul(&cos)?.broadcast_add(&x1.broadcast_mul(&sin)?)?;
    if rot_dim < d {
        let tail = x.narrow(3, rot_dim, d - rot_dim)?;
        Ok(Tensor::cat(&[&o1, &o2, &tail], 3)?)
    } else {
        Ok(Tensor::cat(&[&o1, &o2], 3)?)
    }
}

/// Additive attention mask `[t_q, t_k]`: 0 where allowed, -inf elsewhere.
///
/// Query `i` sits at absolute position `q0 + i` and may see keys
/// `j <= q0 + i`; with `window`, also `j > q0 + i - window`.
pub fn causal_mask(
    t_q: usize,
    t_k: usize,
    q0: usize,
    window: Option<usize>,
    dev: &Device,
) -> Result<Tensor> {
    let neg = f32::NEG_INFINITY;
    let mut m = vec![0.0f32; t_q * t_k];
    for i in 0..t_q {
        let p = q0 + i;
        for j in 0..t_k {
            let causal_ok = j <= p;
            let window_ok = window.map(|w| j + w > p).unwrap_or(true);
            if !(causal_ok && window_ok) {
                m[i * t_k + j] = neg;
            }
        }
    }
    Ok(Tensor::from_vec(m, (t_q, t_k), dev)?)
}

/// Repeat KV heads `n_rep` times along dim 1 (`[B, Hv, T, D]`).
fn repeat_kv_heads(kv: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(kv.clone());
    }
    let hv = kv.dim(1)?;
    let mut idx = Vec::with_capacity(hv * n_rep);
    for head in 0..hv as u32 {
        idx.extend(std::iter::repeat_n(head, n_rep));
    }
    let idx = Tensor::from_vec(idx, hv * n_rep, kv.device())?;
    Ok(kv.contiguous()?.index_select(&idx, 1)?)
}

/// `softmax(q @ k^T * scale + mask) @ v`, with GQA head expansion.
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let n_rep = q.dim(1)? / k.dim(1)?;
    let k = repeat_kv_heads(k, n_rep)?.contiguous()?;
    let v = repeat_kv_heads(v, n_rep)?.contiguous()?;
    let scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
    let scores = scores.broadcast_add(mask)?;
    let probs = softmax_last(&scores)?;
    Ok(probs.matmul(&v)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_matches_hand_computation() {
        let dev = Device::Cpu;
        // x = [3, 4]: rms = sqrt(25/2); y = x / rms * w.
        let x = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &dev).unwrap();
        let w = Tensor::from_vec(vec![2.0f32, 0.5], 2, &dev).unwrap();
        let y = rms_norm(&x, &w, 0.0).unwrap().reshape(2).unwrap().to_vec1::<f32>().unwrap();
        let rms = (25.0f32 / 2.0).sqrt();
        assert!((y[0] - 3.0 / rms * 2.0).abs() < 1e-6);
        assert!((y[1] - 4.0 / rms * 0.5).abs() < 1e-6);
    }

    #[test]
    fn activations_match_reference_values() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], 3, &dev).unwrap();
        let s = silu(&x).unwrap().to_vec1::<f32>().unwrap();
        assert!((s[0] - 0.0).abs() < 1e-7);
        assert!((s[1] - 0.7310586).abs() < 1e-6);
        let g = gelu_tanh(&x).unwrap().to_vec1::<f32>().unwrap();
        assert!((g[0] - 0.0).abs() < 1e-7);
        assert!((g[1] - 0.841_192).abs() < 1e-6);
        assert!((g[2] + 0.158_808).abs() < 1e-6);
    }

    #[test]
    fn rope_half_rotation_matches_formula() {
        let dev = Device::Cpu;
        // D = 4, one head, one token: rotate halves against cos/sin.
        let x = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (1, 1, 1, 4), &dev).unwrap();
        let y = rope_neox(&x, 0, 10_000.0, 4, None)
            .unwrap()
            .reshape(4)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((y[0] - 1.0).abs() < 1e-6); // pos 0: identity
        assert!((y[3] - 1.0).abs() < 1e-6);
        let y1 = rope_neox(&x, 1, 10_000.0, 4, None)
            .unwrap()
            .reshape(4)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // pair 0: angle = 1 * 10000^0 = 1 rad on (x0, x2) = (1, 0).
        assert!((y1[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((y1[2] - 1.0f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn rope_factors_freeze_pairs() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 0.5, 0.25, 0.125], (1, 1, 1, 4), &dev).unwrap();
        // Second pair divided by 1e30: angle ~ 0, pair untouched.
        let y = rope_neox(&x, 7, 10_000.0, 4, Some(&[1.0, 1e30]))
            .unwrap()
            .reshape(4)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((y[1] - 0.5).abs() < 1e-6);
        assert!((y[3] - 0.125).abs() < 1e-6);
    }

    #[test]
    fn mask_is_causal_and_windowed() {
        let dev = Device::Cpu;
        let m = causal_mask(2, 4, 2, Some(2), &dev).unwrap().to_vec2::<f32>().unwrap();
        // query 0 at pos 2 sees keys 1..=2; query 1 at pos 3 sees 2..=3.
        assert_eq!(m[0][0], f32::NEG_INFINITY);
        assert_eq!(m[0][1], 0.0);
        assert_eq!(m[0][2], 0.0);
        assert_eq!(m[0][3], f32::NEG_INFINITY);
        assert_eq!(m[1][1], f32::NEG_INFINITY);
        assert_eq!(m[1][3], 0.0);
    }
}

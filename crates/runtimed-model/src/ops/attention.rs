//! Attention operations: causal mask and scaled dot-product attention with GQA.

use super::activation::softmax_last;
use crate::error::Result;
use candle_core::{Device, Tensor};

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
    fn mask_is_causal_and_windowed() {
        let dev = Device::Cpu;
        let m = causal_mask(2, 4, 2, Some(2), &dev).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(m[0][0], f32::NEG_INFINITY);
        assert_eq!(m[0][1], 0.0);
        assert_eq!(m[0][2], 0.0);
        assert_eq!(m[0][3], f32::NEG_INFINITY);
        assert_eq!(m[1][1], f32::NEG_INFINITY);
        assert_eq!(m[1][3], 0.0);
    }

    #[test]
    fn repeat_kv_heads_identity_when_one() {
        let dev = Device::Cpu;
        let t = Tensor::zeros((1, 2, 3, 4), candle_core::DType::F32, &dev).unwrap();
        let rep = repeat_kv_heads(&t, 1).unwrap();
        assert_eq!(rep.dims(), &[1, 2, 3, 4]);
    }
}

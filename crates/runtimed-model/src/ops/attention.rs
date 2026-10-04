//! Attention operations: causal mask and scaled dot-product attention with GQA.

use super::activation::softmax_last;
use crate::error::Result;
use candle_core::{Device, DeviceLocation, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;

type GqaCache = HashMap<(usize, usize, DeviceLocation), Tensor>;
static GQA_INDEX_CACHE: Mutex<Option<GqaCache>> = Mutex::new(None);

/// Retrieve or compute cached GQA head indexing tensor on target compute device.
fn get_gqa_index(hv: usize, n_rep: usize, dev: &Device) -> Result<Tensor> {
    let loc = dev.location();
    let key = (hv, n_rep, loc);
    {
        let mut lock = GQA_INDEX_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        let map = lock.get_or_insert_with(HashMap::new);
        if let Some(t) = map.get(&key) {
            return Ok(t.clone());
        }
    }
    let mut idx = Vec::with_capacity(hv * n_rep);
    for head in 0..hv as u32 {
        idx.extend(std::iter::repeat_n(head, n_rep));
    }
    let idx_tensor = Tensor::from_vec(idx, hv * n_rep, dev)?;
    let mut lock = GQA_INDEX_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let map = lock.get_or_insert_with(HashMap::new);
    map.insert(key, idx_tensor.clone());
    Ok(idx_tensor)
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
    if t_q == 1 && window.is_none() && t_k <= q0 + 1 {
        return Ok(Tensor::zeros((1, t_k), candle_core::DType::F32, dev)?);
    }
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
    let idx = get_gqa_index(hv, n_rep, kv.device())?;
    let kv = if kv.is_contiguous() { kv.clone() } else { kv.contiguous()? };
    Ok(kv.index_select(&idx, 1)?)
}

/// `softmax(q @ k^T * scale + mask) @ v`, with GQA head expansion and mask bypass.
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    let n_rep = q.dim(1)? / k.dim(1)?;
    let k = repeat_kv_heads(k, n_rep)?;
    let v = repeat_kv_heads(v, n_rep)?;
    let mut scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
    if let Some(mask) = mask {
        let mask = if mask.dtype() != scores.dtype() {
            mask.to_dtype(scores.dtype())?
        } else {
            mask.clone()
        };
        scores = scores.broadcast_add(&mask)?;
    }
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

    #[test]
    fn test_scatter_set_kv() {
        let dev = Device::Cpu;
        let buf = Tensor::zeros((1, 2, 8, 4), candle_core::DType::F32, &dev).unwrap();
        let tok = Tensor::ones((1, 2, 1, 4), candle_core::DType::F32, &dev).unwrap();
        let ids = Tensor::from_vec(vec![0u32; 8], (1, 2, 1, 4), &dev).unwrap();
        buf.scatter_set(&ids, &tok, 2).unwrap();
        let act = buf.narrow(2, 0, 1).unwrap();
        assert_eq!(act.dims(), &[1, 2, 1, 4]);

        #[cfg(feature = "cuda")]
        if let Ok(cdev) = Device::new_cuda(0) {
            let buf_c = Tensor::zeros((1, 2, 8, 4), candle_core::DType::F16, &cdev).unwrap();
            let tok_c = Tensor::ones((1, 2, 1, 4), candle_core::DType::F16, &cdev).unwrap();
            let ids_c = Tensor::from_vec(vec![0u32; 8], (1, 2, 1, 4), &cdev).unwrap();
            buf_c.scatter_set(&ids_c, &tok_c, 2).unwrap();
            let k_act = buf_c.narrow(2, 0, 1).unwrap();
            let v_act = buf_c.narrow(2, 0, 1).unwrap();
            let q = Tensor::ones((1, 2, 1, 4), candle_core::DType::F16, &cdev).unwrap();
            let o = attention(&q, &k_act, &v_act, None, 1.0).unwrap();
            assert_eq!(o.dims(), &[1, 2, 1, 4]);
        }
    }
}

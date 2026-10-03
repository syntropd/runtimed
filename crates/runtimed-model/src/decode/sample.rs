//! Next-token sampling: greedy, temperature, top-k, top-p.
//!
//! The differential oracle runs at temperature 0 (pure argmax); the
//! stochastic paths exist for real generation in Phase 3.

use crate::error::Result;
use candle_core::Tensor;

/// Greedy decode: the argmax id of a `[vocab]` logit row.
pub fn greedy(logits: &Tensor) -> Result<u32> {
    let id = logits.argmax(0)?.to_scalar::<u32>()?;
    Ok(id)
}

/// Compute normalized probabilities over a `[vocab]` logit row.
/// When `temperature <= 0.0`, returns a 1-hot probability vector at the greedy argmax.
pub fn probs(logits: &Tensor, temperature: f32, top_k: usize, top_p: f32) -> Result<Vec<f32>> {
    let vocab_size = logits.dim(0).unwrap_or(0);
    if vocab_size == 0 {
        return Ok(Vec::new());
    }
    if temperature <= 0.0 {
        let best_idx = logits.argmax(0)?.to_scalar::<u32>()? as usize;
        let mut p = vec![0.0f32; vocab_size];
        if best_idx < p.len() {
            p[best_idx] = 1.0;
        }
        return Ok(p);
    }
    let mut v = logits.to_vec1::<f32>()?;
    for x in v.iter_mut() {
        *x /= temperature;
    }
    if top_k > 0 && top_k < v.len() {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.select_nth_unstable_by(top_k - 1, |&a, &b| v[b].total_cmp(&v[a]));
        let thresh = v[idx[top_k - 1]];
        for x in v.iter_mut() {
            if *x < thresh {
                *x = f32::NEG_INFINITY;
            }
        }
    }
    let max = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut p: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = p.iter().sum();
    if sum > 0.0 {
        for x in p.iter_mut() {
            *x /= sum;
        }
    }
    if top_p < 1.0 {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_unstable_by(|&a, &b| p[b].total_cmp(&p[a]));
        let mut cum = 0.0f32;
        let mut keep = vec![false; v.len()];
        for &i in &idx {
            keep[i] = true;
            cum += p[i];
            if cum >= top_p {
                break;
            }
        }
        let mut renorm = 0.0f32;
        for (i, x) in p.iter_mut().enumerate() {
            if !keep[i] {
                *x = 0.0;
            } else {
                renorm += *x;
            }
        }
        if renorm > 0.0 {
            for x in p.iter_mut() {
                *x /= renorm;
            }
        }
    }
    Ok(p)
}

/// Sample an index from a normalized probability vector.
pub fn sample_from_probs(p: &[f32], mut rand01: impl FnMut() -> f32) -> u32 {
    if p.is_empty() {
        return 0;
    }
    let mut r = rand01() % 1.0;
    if r < 0.0 {
        r += 1.0;
    }
    let mut acc = 0.0f32;
    let mut last_valid = 0;
    for (i, &x) in p.iter().enumerate() {
        if x > 0.0 {
            last_valid = i;
        }
        acc += x;
        if r < acc {
            return i as u32;
        }
    }
    last_valid as u32
}

/// Temperature + top-k + top-p sampling over a `[vocab]` row.
/// `temperature <= 0` means greedy. `rand01` supplies uniform draws.
pub fn sample(logits: &Tensor, temperature: f32, top_k: usize, top_p: f32, rand01: impl FnMut() -> f32) -> Result<u32> {
    if temperature <= 0.0 {
        return greedy(logits);
    }
    let p = probs(logits, temperature, top_k, top_p)?;
    Ok(sample_from_probs(&p, rand01))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn greedy_picks_max() {
        let dev = Device::Cpu;
        let l = Tensor::from_vec(vec![0.1f32, 5.0, 2.0], 3, &dev).unwrap();
        assert_eq!(greedy(&l).unwrap(), 1);
        assert_eq!(sample(&l, 0.0, 0, 1.0, || 0.99).unwrap(), 1);
    }

    #[test]
    fn top_k_truncates_tail() {
        let dev = Device::Cpu;
        // With top_k = 1 only id 1 can ever come out.
        let l = Tensor::from_vec(vec![0.0f32, 10.0, 9.0], 3, &dev).unwrap();
        for i in 0..20 {
            let id = sample(&l, 1.0, 1, 1.0, || (i as f32 + 0.5) / 20.0).unwrap();
            assert_eq!(id, 1);
        }
    }

    #[test]
    fn top_p_keeps_nucleus() {
        let dev = Device::Cpu;
        // Id 2 holds ~all mass; top_p = 0.5 keeps id 2 alone.
        let l = Tensor::from_vec(vec![-10.0f32, -10.0, 0.0], 3, &dev).unwrap();
        let id = sample(&l, 1.0, 0, 0.5, || 0.0).unwrap();
        assert_eq!(id, 2);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_greedy() {
        let dev = Device::new_cuda(0).unwrap();
        let l = Tensor::from_vec(vec![0.1f32, 5.0, 2.0], 3, &dev).unwrap();
        assert_eq!(greedy(&l).unwrap(), 1);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_multi_greedy() {
        let dev0 = Device::new_cuda(0).unwrap();
        let dev1 = Device::new_cuda(1).unwrap();
        let mut v = vec![0.0f32; 151646];
        v[19] = 100.0;
        let l1 = Tensor::from_vec(v.clone(), 151646, &dev1).unwrap();
        crate::weights::Weights::ensure_current(&dev1).unwrap();
        assert_eq!(greedy(&l1).unwrap(), 19);

        let l0 = Tensor::from_vec(v.clone(), 151646, &dev0).unwrap();
        crate::weights::Weights::ensure_current(&dev0).unwrap();
        assert_eq!(greedy(&l0).unwrap(), 19);
    }
}

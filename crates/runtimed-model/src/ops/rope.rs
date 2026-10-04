//! Rotary Position Embedding (RoPE) implementations: NeoX and Norm styles.

use crate::error::Result;
use candle_core::Tensor;

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

/// Interleaved RoPE with normalization (Granite style).
pub fn rope_norm(x: &Tensor, q0: usize, theta: f32, rot_dim: usize) -> Result<Tensor> {
    let dev = x.device();
    let (_b, _h, t, d) = x.dims4()?;
    let half = rot_dim / 2;
    let mut cos = Vec::with_capacity(t * half);
    let mut sin = Vec::with_capacity(t * half);
    for pos in q0..q0 + t {
        let p = pos as f32;
        for i in 0..half {
            let inv = theta.powf(-2.0 * i as f32 / rot_dim as f32);
            let angle = p * inv;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let cos = Tensor::from_vec(cos, (1, 1, t, half, 1), dev)?;
    let sin = Tensor::from_vec(sin, (1, 1, t, half, 1), dev)?;
    let cos = if cos.dtype() != x.dtype() { cos.to_dtype(x.dtype())? } else { cos };
    let sin = if sin.dtype() != x.dtype() { sin.to_dtype(x.dtype())? } else { sin };
    let x_rot = x.narrow(3, 0, rot_dim)?.contiguous()?;
    let x_pair = x_rot.reshape((1, _h, t, half, 2))?;
    let x0 = x_pair.narrow(4, 0, 1)?;
    let x1 = x_pair.narrow(4, 1, 1)?;
    let o0 = x0.broadcast_mul(&cos)?.broadcast_sub(&x1.broadcast_mul(&sin)?)?;
    let o1 = x0.broadcast_mul(&sin)?.broadcast_add(&x1.broadcast_mul(&cos)?)?;
    let o = Tensor::cat(&[&o0, &o1], 4)?.contiguous()?.reshape((1, _h, t, rot_dim))?;
    if rot_dim < d {
        let tail = x.narrow(3, rot_dim, d - rot_dim)?;
        Ok(Tensor::cat(&[&o, &tail], 3)?)
    } else {
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn rope_half_rotation_matches_formula() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (1, 1, 1, 4), &dev).unwrap();
        let y = rope_neox(&x, 0, 10_000.0, 4, None)
            .unwrap()
            .reshape(4)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((y[0] - 1.0).abs() < 1e-6);
        assert!((y[3] - 1.0).abs() < 1e-6);
        let y1 = rope_neox(&x, 1, 10_000.0, 4, None)
            .unwrap()
            .reshape(4)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((y1[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((y1[2] - 1.0f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn rope_factors_freeze_pairs() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 0.5, 0.25, 0.125], (1, 1, 1, 4), &dev).unwrap();
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
    fn rope_norm_rotates_interleaved() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &dev).unwrap();
        let y0 = rope_norm(&x, 0, 10_000.0, 4).unwrap().reshape(4).unwrap().to_vec1::<f32>().unwrap();
        assert!((y0[0] - 1.0).abs() < 1e-6);
        assert!((y0[1] - 2.0).abs() < 1e-6);
    }
}

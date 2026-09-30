//! Vectorized spatial patch pooling with symmetric edge padding.
//!
//! Replaces iterative patch slicing with strided reshape and mean reduction:
//! `(1, out_h, k, out_w, k, c) -> permute -> mean -> (1, out_h * out_w, c)`.
//! Non-multiple grid dimensions are symmetrically edge-padded with replicated
//! border patches to guarantee zero panics for arbitrary grid geometry.

use crate::error::{ModelError, Result};
use candle_core::Tensor;

/// Vectorized spatial patch pooling supporting kernel sizes (e.g. 2x2, 3x3, 4x4)
/// with symmetric edge replication padding.
pub fn pool_patches(h: &Tensor, gw: usize, gh: usize, k: usize) -> Result<Tensor> {
    if k == 0 {
        return Err(ModelError::Config("pooling kernel size cannot be zero".into()));
    }
    if gw == 0 || gh == 0 {
        return Err(ModelError::Config("grid dimensions cannot be zero".into()));
    }
    if h.dims().len() != 3 {
        return Err(ModelError::Shape {
            name: "pool_input".into(),
            expected: vec![1, gw * gh, 0],
            got: h.dims().to_vec(),
        });
    }
    let expected_n = gw * gh;
    let actual_n = h.dim(1)?;
    if actual_n != expected_n {
        return Err(ModelError::Shape {
            name: "pool_grid_mismatch".into(),
            expected: vec![1, expected_n, h.dim(2)?],
            got: h.dims().to_vec(),
        });
    }
    let c = h.dim(2)?;
    let mut grid = h.reshape((1, gh, gw, c))?;

    // Symmetric edge padding for width.
    let padded_gw = gw.div_ceil(k) * k;
    let pad_w = padded_gw - gw;
    if pad_w > 0 {
        let pad_left = pad_w / 2;
        let pad_right = pad_w - pad_left;
        let mut x_parts = Vec::with_capacity(1 + pad_left + pad_right);
        if pad_left > 0 {
            let left_col = grid.narrow(2, 0, 1)?;
            for _ in 0..pad_left {
                x_parts.push(left_col.clone());
            }
        }
        x_parts.push(grid);
        if pad_right > 0 {
            let right_col = x_parts[x_parts.len() - 1].narrow(2, gw - 1, 1)?;
            for _ in 0..pad_right {
                x_parts.push(right_col.clone());
            }
        }
        let refs: Vec<&Tensor> = x_parts.iter().collect();
        grid = Tensor::cat(&refs, 2)?;
    }

    // Symmetric edge padding for height.
    let padded_gh = gh.div_ceil(k) * k;
    let pad_h = padded_gh - gh;
    if pad_h > 0 {
        let pad_top = pad_h / 2;
        let pad_bottom = pad_h - pad_top;
        let mut y_parts = Vec::with_capacity(1 + pad_top + pad_bottom);
        if pad_top > 0 {
            let top_row = grid.narrow(1, 0, 1)?;
            for _ in 0..pad_top {
                y_parts.push(top_row.clone());
            }
        }
        y_parts.push(grid);
        if pad_bottom > 0 {
            let bot_row = y_parts[y_parts.len() - 1].narrow(1, gh - 1, 1)?;
            for _ in 0..pad_bottom {
                y_parts.push(bot_row.clone());
            }
        }
        let refs: Vec<&Tensor> = y_parts.iter().collect();
        grid = Tensor::cat(&refs, 1)?;
    }

    let out_h = padded_gh / k;
    let out_w = padded_gw / k;

    // Strided reshape -> permute -> mean reduction -> final tokens reshape.
    let x = grid.reshape((1, out_h, k, out_w, k, c))?;
    let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
    let x = x.mean((3, 4))?;
    let out = x.reshape((1, out_h * out_w, c))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_pool_exact_3x3() {
        let dev = Device::Cpu;
        let data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let h = Tensor::from_vec(data, (1, 9, 1), &dev).unwrap();
        let out = pool_patches(&h, 3, 3, 3)
            .unwrap()
            .reshape(1)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(out, vec![5.0]);
    }

    #[test]
    fn test_pool_edge_padding_odd_2x2() {
        let dev = Device::Cpu;
        // 3x3 grid pooled with k=2: padded to 4x4, emitting 2x2 = 4 soft tokens.
        let data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let h = Tensor::from_vec(data, (1, 9, 1), &dev).unwrap();
        let out = pool_patches(&h, 3, 3, 2).unwrap();
        assert_eq!(out.dims(), &[1, 4, 1]);
    }

    #[test]
    fn test_pool_edge_padding_5x5_with_3x3() {
        let dev = Device::Cpu;
        // 5x5 grid pooled with k=3: padded to 6x6, emitting 2x2 = 4 soft tokens.
        let data = vec![1.0f32; 25 * 2];
        let h = Tensor::from_vec(data, (1, 25, 2), &dev).unwrap();
        let out = pool_patches(&h, 5, 5, 3).unwrap();
        assert_eq!(out.dims(), &[1, 4, 2]);
    }

    #[test]
    fn test_pool_edge_padding_7x7_with_4x4() {
        let dev = Device::Cpu;
        // 7x7 grid pooled with k=4: padded to 8x8, emitting 2x2 = 4 soft tokens.
        let data = vec![2.0f32; 49 * 4];
        let h = Tensor::from_vec(data, (1, 49, 4), &dev).unwrap();
        let out = pool_patches(&h, 7, 7, 4).unwrap();
        assert_eq!(out.dims(), &[1, 4, 4]);
    }

    #[test]
    fn test_pool_zero_panics_on_invalid() {
        let dev = Device::Cpu;
        let data = vec![1.0f32; 9];
        let h = Tensor::from_vec(data, (1, 9, 1), &dev).unwrap();
        assert!(pool_patches(&h, 3, 3, 0).is_err());
        assert!(pool_patches(&h, 0, 3, 3).is_err());
        assert!(pool_patches(&h, 4, 4, 3).is_err());
    }
}

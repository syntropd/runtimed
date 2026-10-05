//! Aspect-ratio-preserving dynamic resolution grid partitioning (NaViT / SigLIP-2 style).
//!
//! Computes optimal non-square patch grids for arbitrary screen and document geometries
//! and synthesizes/interpolates 2D sinusoidal positional embeddings for variable token lengths.

use crate::error::{ModelError, Result};
use candle_core::{Device, Tensor};

/// Computes optimal (grid_w, grid_h) patch count preserving aspect ratio.
pub fn compute_dynamic_grid(
    width: u32,
    height: u32,
    patch_size: usize,
    min_patches: usize,
    max_patches: usize,
) -> (usize, usize) {
    let (w, h) = (width.max(1) as f32, height.max(1) as f32);
    let aspect = w / h;
    let base_w = (width as usize / patch_size.max(1)).max(1);
    let base_h = (height as usize / patch_size.max(1)).max(1);

    let mut best_grid = (base_w, base_h);
    let mut best_diff = f32::MAX;

    for gw in 1..=64 {
        for gh in 1..=64 {
            let total = gw * gh;
            if total >= min_patches && total <= max_patches {
                let grid_aspect = gw as f32 / gh as f32;
                let diff = (grid_aspect - aspect).abs();
                if diff < best_diff {
                    best_diff = diff;
                    best_grid = (gw, gh);
                }
            }
        }
    }
    best_grid
}

/// Generates 2D sinusoidal positional embeddings for a [1, grid_h * grid_w, embed_dim] grid.
pub fn sinusoidal_pos_embed_2d(
    grid_w: usize,
    grid_h: usize,
    embed_dim: usize,
    dev: &Device,
) -> Result<Tensor> {
    if !embed_dim.is_multiple_of(4) {
        return Err(ModelError::Config("embed_dim must be divisible by 4 for 2D RoPE/pos embeddings".into()));
    }
    let half_dim = embed_dim / 2;
    let quarter_dim = half_dim / 2;

    let mut freqs = Vec::with_capacity(quarter_dim);
    for i in 0..quarter_dim {
        freqs.push((-((i as f32) * 2.0 * (10000.0f32.ln()) / (quarter_dim as f32 * 2.0))).exp());
    }

    let n = grid_w * grid_h;
    let mut data = Vec::with_capacity(n * embed_dim);

    for y in 0..grid_h {
        for x in 0..grid_w {
            let mut row = Vec::with_capacity(embed_dim);
            // X coordinate sinusoidal encoding
            for &f in &freqs {
                let v = (x as f32) * f;
                row.push(v.sin());
                row.push(v.cos());
            }
            // Y coordinate sinusoidal encoding
            for &f in &freqs {
                let v = (y as f32) * f;
                row.push(v.sin());
                row.push(v.cos());
            }
            data.extend(row);
        }
    }
    Ok(Tensor::from_vec(data, (1, n, embed_dim), dev)?)
}

/// Bilinearly interpolates 2D positional embeddings from (src_w, src_h) to (dst_w, dst_h).
pub fn interpolate_pos_embed_2d(
    embed: &Tensor,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Result<Tensor> {
    let (b, total, dim) = (embed.dim(0)?, embed.dim(1)?, embed.dim(2)?);
    if total != src_w * src_h {
        return Err(ModelError::Config(format!(
            "Input embed size {} does not match src_grid {}x{}",
            total, src_w, src_h
        )));
    }
    if src_w == dst_w && src_h == dst_h {
        return Ok(embed.clone());
    }

    let dev = embed.device();
    let raw = embed.to_vec3::<f32>()?;
    let mut out = Vec::with_capacity(b * dst_w * dst_h * dim);

    for batch in raw {
        for dy in 0..dst_h {
            let sy = if dst_h > 1 { (dy as f32 * (src_h - 1) as f32) / (dst_h - 1) as f32 } else { 0.0 };
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(src_h - 1);
            let wy = sy - y0 as f32;

            for dx in 0..dst_w {
                let sx = if dst_w > 1 { (dx as f32 * (src_w - 1) as f32) / (dst_w - 1) as f32 } else { 0.0 };
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let wx = sx - x0 as f32;

                let (p00, p01) = (&batch[y0 * src_w + x0], &batch[y0 * src_w + x1]);
                let (p10, p11) = (&batch[y1 * src_w + x0], &batch[y1 * src_w + x1]);

                for c in 0..dim {
                    let top = p00[c] * (1.0 - wx) + p01[c] * wx;
                    let bottom = p10[c] * (1.0 - wx) + p11[c] * wx;
                    out.push(top * (1.0 - wy) + bottom * wy);
                }
            }
        }
    }
    Ok(Tensor::from_vec(out, (b, dst_w * dst_h, dim), dev)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_grid_aspect_ratio() {
        // 16:9 1080p desktop aspect
        let (gw, gh) = compute_dynamic_grid(1920, 1080, 14, 16, 256);
        let ratio = gw as f32 / gh as f32;
        let expected = 1920.0 / 1080.0;
        assert!((ratio - expected).abs() < 0.2);

        // 21:9 ultrawide monitor
        let (uw_w, uw_h) = compute_dynamic_grid(3440, 1440, 14, 16, 256);
        let uw_ratio = uw_w as f32 / uw_h as f32;
        assert!((uw_ratio - (3440.0 / 1440.0)).abs() < 0.2);
    }

    #[test]
    fn test_sinusoidal_and_interpolation() {
        let dev = Device::Cpu;
        let (src_w, src_h, dim) = (4, 4, 32);
        let emb = sinusoidal_pos_embed_2d(src_w, src_h, dim, &dev).unwrap();
        assert_eq!(emb.dims(), &[1, 16, 32]);

        let interp = interpolate_pos_embed_2d(&emb, src_w, src_h, 8, 6).unwrap();
        assert_eq!(interp.dims(), &[1, 48, 32]);
    }
}

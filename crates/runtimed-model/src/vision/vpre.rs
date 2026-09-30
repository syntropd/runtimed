//! Vision preprocessing: decode -> resize -> patchify.
//!
//! `prepare` drives decode (this file), Pillow-exact sizing and
//! resampling ([`crate::vision::vresize`]), then im2col patchify in row-major
//! patch order.

use crate::error::{ModelError, Result};
use crate::vision::VisionConfig;
use crate::vision::vresize::{bicubic_resize, smart_resize};

/// An image ready for the tower: rescaled patches plus grid geometry.
pub struct PreparedImage {
    /// `[n_patch, 768]` f32 in im2col order, values in [0, 1].
    pub patches: Vec<f32>,
    pub pos_x: Vec<usize>,
    pub pos_y: Vec<usize>,
    pub grid_w: usize,
    pub grid_h: usize,
    pub n_soft: usize,
}

/// Decode (PNG/JPEG) -> resize -> patchify. Positions are (x, y) patch
/// coordinates in row-major patch order.
pub fn prepare(bytes: &[u8], cfg: &VisionConfig) -> Result<PreparedImage> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| ModelError::Config(format!("image decode: {e}")))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return Err(ModelError::Config("empty image".into()));
    }
    let align = cfg.patch * cfg.merge;
    let patch_px = align * align;
    let (tw, th) = smart_resize(w, h, align, cfg.min_soft * patch_px, cfg.max_soft * patch_px);
    let scaled = bicubic_resize(img.as_raw(), w, h, tw, th);
    let (gw, gh) = (tw / cfg.patch, th / cfg.patch);
    let n_patch = gw * gh;
    // im2col order: k = c*256 + y*16 + x (x fastest), matching ggml.
    let mut patches = vec![0.0f32; n_patch * cfg.patch * cfg.patch * 3];
    let mut pos_x = Vec::with_capacity(n_patch);
    let mut pos_y = Vec::with_capacity(n_patch);
    for by in 0..gh {
        for bx in 0..gw {
            pos_x.push(bx);
            pos_y.push(by);
            let base = (by * gw + bx) * cfg.patch * cfg.patch * 3;
            for c in 0..3 {
                for y in 0..cfg.patch {
                    for x in 0..cfg.patch {
                        let gx = bx * cfg.patch + x;
                        let gy = by * cfg.patch + y;
                        patches[base + (c * cfg.patch + y) * cfg.patch + x] =
                            scaled[(gy * tw + gx) * 3 + c];
                    }
                }
            }
        }
    }
    Ok(PreparedImage {
        patches,
        pos_x,
        pos_y,
        grid_w: gw,
        grid_h: gh,
        n_soft: (gw.div_ceil(cfg.merge)) * (gh.div_ceil(cfg.merge)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::png::PngEncoder;
    use image::ImageEncoder;

    fn tiny_png(w: u32, h: u32) -> Vec<u8> {
        let img =
            image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 32) as u8, (y * 32) as u8, 128]));
        let mut buf = Vec::new();
        PngEncoder::new(&mut buf)
            .write_image(img.as_raw(), w, h, image::ExtendedColorType::Rgb8)
            .unwrap();
        buf
    }

    fn tiny_cfg() -> VisionConfig {
        VisionConfig {
            n_layer: 1,
            hidden: 8,
            n_head: 2,
            head_dim: 4,
            ffn: 8,
            eps: 1e-6,
            patch: 2,
            merge: 1,
            min_soft: 1,
            max_soft: 64,
            proj_dim: 8,
        }
    }

    #[test]
    fn prepares_forged_png() {
        let prep = prepare(&tiny_png(8, 8), &tiny_cfg()).unwrap();
        assert_eq!(prep.pos_x.len(), prep.pos_y.len());
        assert_eq!(prep.patches.len(), prep.pos_x.len() * 2 * 2 * 3);
        assert_eq!(prep.n_soft, prep.grid_w * prep.grid_h);
        assert!(prep.patches.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_garbage_bytes() {
        assert!(prepare(b"definitely not an image", &tiny_cfg()).is_err());
    }
}

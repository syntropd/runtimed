//! Gemma4 vision tower (SigLIP-style encoder + 3x3 pool + projector).
//!
//! Batch is 1, one image at a time. The tower (16 post-norm layers,
//! axial RoPE, clippable linears) emits pooled soft tokens projected to
//! text width. Preprocessing lives in [`crate::vpre`].

pub use crate::vpre::{prepare, PreparedImage};

use crate::error::{ModelError, Result};
use crate::ops;
use crate::weights::Weights;
use candle_core::{DType, Device, Tensor};
use runtimed_gguf::{GgufFile, MetaValue};

/// Placeholder ids for the image chunk (boi, per-token, eoi).
pub const IMG_BEG: u32 = 255999;
pub const IMG_TOKEN: u32 = 258880;
pub const IMG_END: u32 = 258882;

#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub n_layer: usize,
    pub hidden: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub eps: f32,
    pub patch: usize,
    pub merge: usize,
    pub min_soft: usize,
    pub max_soft: usize,
    pub proj_dim: usize,
}

fn meta_u32(file: &GgufFile, key: &str) -> Result<u32> {
    match file.metadata.get(key) {
        Some(MetaValue::U32(v)) => Ok(*v),
        Some(MetaValue::U64(v)) => Ok(u32::try_from(*v).map_err(|_| {
            ModelError::Config(format!("{key} out of range"))
        })?),
        other => Err(ModelError::Config(format!("{key} missing, got {other:?}"))),
    }
}

fn meta_f32(file: &GgufFile, key: &str) -> Result<f32> {
    match file.metadata.get(key) {
        Some(MetaValue::F32(v)) => Ok(*v),
        other => Err(ModelError::Config(format!("{key} missing, got {other:?}"))),
    }
}

impl VisionConfig {
    pub fn parse(file: &GgufFile) -> Result<Self> {
        let proj = match file.metadata.get("clip.vision.projector_type") {
            Some(MetaValue::Str(s)) => s.clone(),
            other => return Err(ModelError::Config(format!("no projector type: {other:?}"))),
        };
        if proj != "gemma4v" {
            return Err(ModelError::Config(format!("vision projector {proj}, want gemma4v")));
        }
        let hidden = meta_u32(file, "clip.vision.embedding_length")? as usize;
        let n_head = meta_u32(file, "clip.vision.attention.head_count")? as usize;
        Ok(Self {
            n_layer: meta_u32(file, "clip.vision.block_count")? as usize,
            hidden,
            n_head,
            head_dim: hidden / n_head,
            ffn: meta_u32(file, "clip.vision.feed_forward_length")? as usize,
            eps: meta_f32(file, "clip.vision.attention.layer_norm_epsilon")?,
            patch: meta_u32(file, "clip.vision.patch_size")? as usize,
            merge: 3,
            min_soft: 70,
            max_soft: 1120,
            proj_dim: meta_u32(file, "clip.vision.projection_dim")? as usize,
        })
    }
}
/// The vision tower: encoder weights plus config.
pub struct VisionTower {
    cfg: VisionConfig,
    w: Weights,
}

impl VisionTower {
    /// Load vision + projector tensors from an mmproj file (audio skipped).
    pub fn load(path: &std::path::Path, dev: &Device) -> Result<Self> {
        let file = GgufFile::open(path)?;
        let cfg = VisionConfig::parse(&file)?;
        let w = Weights::load_filtered(&file, dev, |name| {
            name.starts_with("v.") || name == "mm.input_projection.weight"
        })?;
        Ok(Self { cfg, w })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    fn scalar(&self, name: &str) -> Result<f32> {
        Ok(self.w.get(name)?.to_vec1::<f32>()?[0])
    }

    /// Clippable linear: clamp input, matmul, clamp output.
    fn clipped(&self, x: &Tensor, base: &str) -> Result<Tensor> {
        let weight = format!("{base}.weight");
        let x = x.clamp(
            self.scalar(&format!("{base}.input_min"))?,
            self.scalar(&format!("{base}.input_max"))?,
        )?;
        let y = self.w.linear(&x, &weight)?;
        Ok(y.clamp(
            self.scalar(&format!("{base}.output_min"))?,
            self.scalar(&format!("{base}.output_max"))?,
        )?)
    }

    fn block(&self, i: usize, h: &Tensor, img: &PreparedImage) -> Result<Tensor> {
        let cfg = &self.cfg;
        let pre = format!("v.blk.{i}");
        let n = h.dim(1)?;
        let split = |y: Tensor| -> Result<Tensor> {
            Ok(y.reshape((1, n, cfg.n_head, cfg.head_dim))?.transpose(1, 2)?)
        };
        // Attention block.
        let normed = ops::rms_norm(h, &self.w.get(&format!("{pre}.ln1.weight"))?, cfg.eps)?;
        let q = split(self.clipped(&normed, &format!("{pre}.attn_q"))?)?;
        let k = split(self.clipped(&normed, &format!("{pre}.attn_k"))?)?;
        let v = split(self.clipped(&normed, &format!("{pre}.attn_v"))?)?;
        let q = ops::rms_norm(&q, &self.w.get(&format!("{pre}.attn_q_norm.weight"))?, cfg.eps)?;
        let k = ops::rms_norm(&k, &self.w.get(&format!("{pre}.attn_k_norm.weight"))?, cfg.eps)?;
        // Axial RoPE: first half rotates by x, second half by y.
        let half = cfg.head_dim / 2;
        let q = axial_rope(&q, &img.pos_x, &img.pos_y, half)?;
        let k = axial_rope(&k, &img.pos_x, &img.pos_y, half)?;
        let v = ops::rms_norm_plain(&v, cfg.eps)?;
        let zero = Tensor::zeros((n, n), DType::F32, h.device())?;
        let o = ops::attention(&q, &k, &v, &zero, 1.0)?;
        let o = o.transpose(1, 2)?.reshape((1, n, cfg.n_head * cfg.head_dim))?;
        let o = self.clipped(&o, &format!("{pre}.attn_out"))?;
        let o = ops::rms_norm(&o, &self.w.get(&format!("{pre}.attn_post_norm.weight"))?, cfg.eps)?;
        let h = h.broadcast_add(&o)?;
        // GeGLU-quick block (reference ffn_op fallback).
        let normed = ops::rms_norm(&h, &self.w.get(&format!("{pre}.ln2.weight"))?, cfg.eps)?;
        let g = self.clipped(&normed, &format!("{pre}.ffn_gate"))?;
        let u = self.clipped(&normed, &format!("{pre}.ffn_up"))?;
        let mlp = self.clipped(&ops::gelu_quick(&g)?.broadcast_mul(&u)?, &format!("{pre}.ffn_down"))?;
        let mlp = ops::rms_norm(&mlp, &self.w.get(&format!("{pre}.ffn_post_norm.weight"))?, cfg.eps)?;
        Ok(h.broadcast_add(&mlp)?)
    }

    /// Encode one image to `[1, n_soft, proj_dim]` soft tokens.
    pub fn encode(&self, img: &PreparedImage) -> Result<Tensor> {
        let cfg = &self.cfg;
        let dev = self.w.device();
        Weights::ensure_current(dev)?;
        let n = img.pos_x.len();
        // Patch embed: 2x - 1, conv-as-matmul, plus x/y position tables.
        let patches = Tensor::from_vec(img.patches.clone(), (n, 768), dev)?.affine(2.0, -1.0)?;
        let kernel = self.w.get("v.patch_embd.weight")?.reshape((768, 768))?;
        let mut h = patches.matmul(&kernel.t()?)?.reshape((1, n, 768))?;
        let pos = self.w.get("v.position_embd.weight")?; // [2, 10240, 768]
        let idx = |v: &[usize]| Tensor::from_vec(v.iter().map(|&x| x as u32).collect::<Vec<_>>(), n, dev);
        let table = |slot: usize, coords: &[usize]| -> Result<Tensor> {
            Ok(pos
                .narrow(0, slot, 1)?
                .squeeze(0)?
                .contiguous()?
                .index_select(&idx(coords)?, 0)?
                .unsqueeze(0)?)
        };
        let ex = table(0, &img.pos_x)?;
        let ey = table(1, &img.pos_y)?;
        h = h.broadcast_add(&ex)?.broadcast_add(&ey)?;
        for i in 0..cfg.n_layer {
            h = self.block(i, &h, img)?;
        }
        // 3x3 average pool over the patch grid, scaled by sqrt(hidden).
        h = avg_pool_3x3(&h, img.grid_w, img.grid_h)?;
        let h = h.affine((cfg.hidden as f32).sqrt() as f64, 0.0)?;
        // Projector: scaleless norm, linear to text width.
        let h = ops::rms_norm_plain(&h, cfg.eps)?;
        Ok(self.w.linear(&h, "mm.input_projection.weight")?)
    }
}

/// Axial RoPE: NeoX-rotate the first `half` dims by `pos_x`, the second
/// `half` by `pos_y` (theta = 100, the Gemma4 vision constant).
fn axial_rope(x: &Tensor, pos_x: &[usize], pos_y: &[usize], half: usize) -> Result<Tensor> {
    let a = ops::rope_neox_pos(&x.narrow(3, 0, half)?, pos_x, 100.0, half, None)?;
    let b = ops::rope_neox_pos(&x.narrow(3, half, half)?, pos_y, 100.0, half, None)?;
    Ok(Tensor::cat(&[&a, &b], 3)?)
}

/// 3x3 non-overlapping average pool on `[1, gh*gw, c]` row-major patches.
fn avg_pool_3x3(h: &Tensor, gw: usize, gh: usize) -> Result<Tensor> {
    assert_eq!(h.dim(1)?, gw * gh, "grid geometry");
    assert_eq!(gw % 3, 0, "grid width pools cleanly");
    assert_eq!(gh % 3, 0, "grid height pools cleanly");
    let c = h.dim(2)?;
    let flat = h.reshape((gh, gw, c))?;
    let mut rows = Vec::with_capacity((gh / 3) * (gw / 3));
    for oy in 0..gh / 3 {
        for ox in 0..gw / 3 {
            let cell = flat
                .narrow(0, oy * 3, 3)?
                .narrow(1, ox * 3, 3)?
                .mean((0, 1))?;
            rows.push(cell);
        }
    }
    let stacked = Tensor::stack(&rows.iter().collect::<Vec<_>>(), 0)?;
    Ok(stacked.unsqueeze(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_averages_nine_cells() {
        let dev = Device::Cpu;
        // 3x3 grid, 1 channel: values 1..=9 -> mean 5.
        let data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let h = Tensor::from_vec(data, (1, 9, 1), &dev).unwrap();
        let out = avg_pool_3x3(&h, 3, 3).unwrap().reshape(1).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(out, vec![5.0]);
    }
}


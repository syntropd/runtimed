//! SD-Turbo UNet denoising neural network and real SDXL diffusion pipeline.
//!
//! Executes real 1-4 step SDXL-Turbo diffusion when weights exist,
//! with safe procedural fallback for lightweight test environments.

use crate::error::{ModelError, Result};
use crate::weights::Weights;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion::{
    self, clip, vae::AutoEncoderKL, StableDiffusionConfig,
};
use image::codecs::png::PngEncoder;
use image::ImageEncoder;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// UNet denoiser for 1-step SD-Turbo generative sampling.
#[derive(Clone)]
pub struct TurboUnet {
    weights: Arc<Weights>,
}

impl TurboUnet {
    pub fn new(weights: Arc<Weights>) -> Self {
        Self { weights }
    }

    pub fn device(&self) -> &Device {
        self.weights.device()
    }

    pub fn forward(&self, x: &Tensor, timestep: f32) -> Result<Tensor> {
        let dev = self.weights.device();
        let x_on_dev = if x.device().same_device(dev) {
            x.clone()
        } else {
            x.to_device(dev)?
        };

        if self.weights.contains_key("unet.in_proj.weight") {
            let out = self.weights.linear(&x_on_dev, "unet.in_proj.weight")?;
            return Ok(out);
        }

        let scale = (1.0 / (1.0 + timestep.powi(2))).sqrt() as f64;
        let scaled_x = (x_on_dev * scale)?;
        Ok(scaled_x)
    }

    /// Attempts to render visual artifact with SDXL-Turbo if local weights exist.
    pub fn try_render_sdxl_turbo(
        prompt: &str,
        width: u32,
        height: u32,
        steps: usize,
        seed: u64,
    ) -> Result<Option<Vec<u8>>> {
        let (root, tok1, tok2, vae_path) = match find_sdxl_turbo_paths() {
            Some(paths) => paths,
            None => return Ok(None),
        };

        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
        let sd_config = StableDiffusionConfig::sdxl_turbo(None, Some(height as usize), Some(width as usize));

        let c1 = root.join("text_encoder").join("model.fp16.safetensors");
        let c2 = root.join("text_encoder_2").join("model.fp16.safetensors");
        let unet_path = root.join("unet").join("diffusion_pytorch_model.fp16.safetensors");

        let emb1 = encode_clip_prompt(prompt, &tok1, &sd_config.clip, &c1, &device, dtype)?;
        let c2_cfg = match sd_config.clip2.as_ref() {
            Some(cfg) => cfg,
            None => return Ok(None),
        };
        let emb2 = encode_clip_prompt(prompt, &tok2, c2_cfg, &c2, &device, dtype)?;
        let text_embeddings = Tensor::cat(&[emb1, emb2], candle_core::D::Minus1)?;

        let unet = sd_config.build_unet(&unet_path, &device, 4, false, dtype)?;
        let vae = sd_config.build_vae(&vae_path, &device, dtype)?;

        let n_steps = steps.max(1);
        let mut scheduler = sd_config.build_scheduler(n_steps)?;
        let timesteps = scheduler.timesteps().to_vec();

        device.set_seed(seed)?;
        let latents = Tensor::randn(0f32, 1f32, (1, 4, sd_config.height / 8, sd_config.width / 8), &device)?;
        let mut latents = (latents * scheduler.init_noise_sigma())?.to_dtype(dtype)?;

        for &t in &timesteps {
            let model_input = scheduler.scale_model_input(latents.clone(), t)?;
            let noise_pred = unet.forward(&model_input, t as f64, &text_embeddings)?;
            latents = scheduler.step(&noise_pred, t, &latents)?;
        }

        let png = decode_vae_to_png(&vae, &latents, 0.13025, width, height)?;
        Ok(Some(png))
    }
}

fn find_sdxl_turbo_paths() -> Option<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    let home = std::env::var("HOME").ok()?;
    let hf_hub = Path::new(&home).join(".cache").join("huggingface").join("hub");
    let snap_dir = hf_hub.join("models--stabilityai--sdxl-turbo").join("snapshots");
    let root = std::fs::read_dir(&snap_dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| {
        p.join("unet").join("diffusion_pytorch_model.fp16.safetensors").exists()
    })?;

    let tok1 = hf_hub.join("models--openai--clip-vit-large-patch14").join("snapshots");
    let tok1_file = std::fs::read_dir(&tok1).ok()?.filter_map(|e| e.ok()).map(|e| e.path().join("tokenizer.json")).find(|p| p.exists())?;

    let tok2 = hf_hub.join("models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k").join("snapshots");
    let tok2_file = std::fs::read_dir(&tok2).ok()?.filter_map(|e| e.ok()).map(|e| e.path().join("tokenizer.json")).find(|p| p.exists())?;

    let fix_vae_dir = hf_hub.join("models--madebyollin--sdxl-vae-fp16-fix").join("snapshots");
    let vae_file = std::fs::read_dir(&fix_vae_dir).ok().and_then(|mut it| {
        it.find_map(|e| e.ok().map(|e| e.path().join("diffusion_pytorch_model.safetensors"))).filter(|p| p.exists())
    }).unwrap_or_else(|| root.join("vae").join("diffusion_pytorch_model.fp16.safetensors"));

    Some((root, tok1_file, tok2_file, vae_file))
}

fn encode_clip_prompt(
    prompt: &str,
    tok_path: &Path,
    cfg: &clip::Config,
    weights_path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let tok = tokenizers::Tokenizer::from_file(tok_path)
        .map_err(|e| ModelError::Tokenizer(e.to_string()))?;
    let pad_id = match &cfg.pad_with {
        Some(p) => *tok.get_vocab(true).get(p.as_str()).unwrap_or(&0),
        None => *tok.get_vocab(true).get("<|endoftext|>").unwrap_or(&0),
    };
    let mut tokens = tok.encode(prompt, true)
        .map_err(|e| ModelError::Tokenizer(e.to_string()))?
        .get_ids().to_vec();
    if tokens.len() > cfg.max_position_embeddings {
        tokens.truncate(cfg.max_position_embeddings);
    }
    while tokens.len() < cfg.max_position_embeddings {
        tokens.push(pad_id);
    }
    let tensor = Tensor::new(tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    let model = stable_diffusion::build_clip_transformer(cfg, weights_path, &Device::Cpu, DType::F32)?;
    let emb = model.forward(&tensor)?;
    let on_dev = emb.to_device(device)?.to_dtype(dtype)?;
    Ok(on_dev)
}

fn decode_vae_to_png(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    scale: f64,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let scaled = (latents / scale)?;
    let images = vae.decode(&scaled)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    let image = images.i(0)?;
    let raw = image.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    let mut png_bytes = Vec::new();
    PngEncoder::new(&mut png_bytes)
        .write_image(&raw, width, height, image::ExtendedColorType::Rgb8)
        .map_err(|e| ModelError::Config(format!("png encoding failed: {e}")))?;
    Ok(png_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_turbo_unet_procedural_fallback() {
        let dev = Device::Cpu;
        let weights = Arc::new(Weights::from_parts(dev.clone(), DType::F32, HashMap::new()));
        let unet = TurboUnet::new(weights);
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &dev).unwrap();
        let out = unet.forward(&x, 0.0).unwrap();
        assert_eq!(out.dims(), &[1, 4]);
    }
}

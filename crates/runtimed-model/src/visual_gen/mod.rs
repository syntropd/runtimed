//! Generative visual output pipeline (SD-Turbo, FLUX.1 DiT, LoRA and multi-step schedulers).

pub mod flux_dit;
pub mod lora_fuse;
pub mod memfd_target;
pub mod scheduler;
pub mod sd_sampler;
pub mod turbo_unet;

pub use flux_dit::{FluxDit, FluxDitBlock, FluxDitConfig};
pub use lora_fuse::LoraMatrixPair;
pub use memfd_target::create_sealed_memfd;
pub use scheduler::{EulerAncestralScheduler, FlowMatchingScheduler};
pub use sd_sampler::{VisualComputeLease, VisualGenConfig, VisualGenSampler};
pub use turbo_unet::TurboUnet;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visual_gen_config_defaults() {
        let sampler = VisualGenSampler::default();
        assert_eq!(sampler.cfg.default_width, 512);
        assert_eq!(sampler.cfg.default_height, 512);
        assert_eq!(sampler.cfg.steps, 1);
    }
}

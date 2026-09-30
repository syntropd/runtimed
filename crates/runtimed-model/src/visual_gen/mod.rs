//! Generative visual output pipeline (SD-Turbo / LCM 1-step sampler).
//!
//! Exposes 1-step latent-to-pixel sampling with compute lease gating,
//! delivering immutably sealed memfd PNG buffers across process boundaries.

pub mod memfd_target;
pub mod sd_sampler;

pub use memfd_target::create_sealed_memfd;
pub use sd_sampler::{VisualComputeLease, VisualGenConfig, VisualGenSampler};

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

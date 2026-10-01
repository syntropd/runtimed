//! Dynamic memory and VRAM hysteresis management.

pub mod sample_vram;
pub mod watermark;

pub use sample_vram::decode_loop_managed;
pub use sample_vram::evaluate_and_spill;
pub use sample_vram::evaluate_and_spill_with_metrics;
pub use sample_vram::sample_vram_metrics;
pub use watermark::DualWatermarkController;
pub use watermark::WatermarkDecision;

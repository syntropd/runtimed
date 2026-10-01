//! Dynamic memory and VRAM hysteresis management.

pub mod watermark;

pub use watermark::{DualWatermarkController, WatermarkDecision};

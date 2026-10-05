//! Sovereign substrate port boundary.
//!
//! Decouples architecture math, sampling policies, and model logic
//! from direct accelerator / backend dependencies.
//!
//! All architecture modules and sampling policies must interact with
//! hardware acceleration or tensor engines strictly through `SubstratePort`.

mod candle;
mod port;

pub use candle::{CandleSubstrate, DEFAULT_SUBSTRATE};
pub use port::SubstratePort;

pub use candle_core::{DType, Device, Tensor};

/// Returns reference to default static candle substrate.
pub fn default_substrate() -> &'static CandleSubstrate {
    &DEFAULT_SUBSTRATE
}

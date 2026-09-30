//! Autonomous Multi-GPU Hardware & Memory Headroom Governor.

mod device_headroom;
mod gang_governor;

#[cfg(test)]
mod governor_tests;

pub use device_headroom::{read_system_available_memory, DeviceHeadroom, ModelKvSpec};
pub use gang_governor::{GangHeadroomResult, GangLinkType, MultiGpuHeadroomGovernor};

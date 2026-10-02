//! Telemetry and kernel pressure metric types for runtimed.

use serde::{Deserialize, Serialize};

/// Kernel Pressure Stall Information and scheduling telemetry metrics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KernelPressureMetrics {
    pub memory_some_avg10: f32,
    pub memory_full_avg10: f32,
    pub cpu_some_avg10: f32,
    pub io_some_avg10: f32,
    pub runqueue_latency_us: u64,
    pub ebpf_active: bool,
}

impl Default for KernelPressureMetrics {
    fn default() -> Self {
        Self {
            memory_some_avg10: 0.0,
            memory_full_avg10: 0.0,
            cpu_some_avg10: 0.0,
            io_some_avg10: 0.0,
            runqueue_latency_us: 0,
            ebpf_active: false,
        }
    }
}

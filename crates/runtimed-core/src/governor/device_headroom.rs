//! Per-accelerator VRAM headroom and KV cache expansion rate calculation.

use std::fs;

/// Specifications of model KV cache for headroom calculations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelKvSpec {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub bytes_per_element: usize,
}

impl ModelKvSpec {
    /// Compute bytes required per token on a stage holding `layers`:
    /// delta_token = 2 * layers * num_kv_heads * head_dim * bytes_per_element
    pub fn delta_token_bytes(&self, layers: usize) -> u64 {
        2 * (layers as u64)
            * (self.num_kv_heads as u64)
            * (self.head_dim as u64)
            * (self.bytes_per_element as u64)
    }
}

/// Headroom status of an individual accelerator stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceHeadroom {
    pub device_id: String,
    pub is_cpu: bool,
    pub assigned_layers: usize,
    pub available_vram_bytes: u64,
    pub delta_token_bytes: u64,
    pub max_tokens: usize,
}

impl DeviceHeadroom {
    pub fn calculate(
        device_id: String,
        is_cpu: bool,
        assigned_layers: usize,
        available_vram_bytes: u64,
        spec: &ModelKvSpec,
    ) -> Self {
        let delta = spec.delta_token_bytes(assigned_layers);
        let max_tokens = if is_cpu {
            if delta == 0 { 0 } else { (available_vram_bytes / delta) as usize }
        } else if delta == 0 {
            usize::MAX
        } else {
            (available_vram_bytes / delta) as usize
        };

        Self {
            device_id,
            is_cpu,
            assigned_layers,
            available_vram_bytes,
            delta_token_bytes: delta,
            max_tokens,
        }
    }
}

/// Read available memory from Linux /proc/meminfo in bytes.
pub fn read_system_available_memory() -> u64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|info| {
            for line in info.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    let kb = rest.trim_start().split_whitespace().next()?.parse::<u64>().ok()?;
                    return Some(kb * 1024);
                }
            }
            None
        })
        .unwrap_or(0)
}

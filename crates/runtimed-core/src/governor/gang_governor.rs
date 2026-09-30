//! Weakest-link multi-GPU headroom governor, link factor, and CPU offload veto.

use super::device_headroom::DeviceHeadroom;
use crate::engine::ReasoningEffort;

/// Interconnect classification for multi-GPU gang execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GangLinkType {
    NVLink,
    PCIe,
    HostBridge,
}

impl GangLinkType {
    pub fn factor(&self) -> f64 {
        match self {
            Self::NVLink => 1.0,
            Self::PCIe => 0.85,
            Self::HostBridge => 0.50,
        }
    }
}

/// Evaluation result from weakest-link governor.
#[derive(Debug, Clone, PartialEq)]
pub struct GangHeadroomResult {
    pub weakest_link_tokens: usize,
    pub effective_tokens: usize,
    pub phi_psi: f64,
    pub beta_link: f64,
    pub cpu_veto: bool,
    pub effort: ReasoningEffort,
}

/// Autonomous Multi-GPU Hardware & Memory Headroom Governor.
#[derive(Debug, Clone)]
pub struct MultiGpuHeadroomGovernor {
    pub stages: Vec<DeviceHeadroom>,
    pub link_type: GangLinkType,
    pub psi_memory_some: f32,
}

impl MultiGpuHeadroomGovernor {
    pub fn new(stages: Vec<DeviceHeadroom>, link_type: GangLinkType, psi_memory_some: f32) -> Self {
        Self {
            stages,
            link_type,
            psi_memory_some,
        }
    }

    /// Single device convenience constructor for backward compatibility.
    pub fn for_single_device(
        backend: &str,
        _model_bytes: usize,
        available_vram: u64,
        psi_memory_some: f32,
    ) -> Self {
        let is_cpu = !backend.eq_ignore_ascii_case("cuda");
        let stage = DeviceHeadroom {
            device_id: "primary".into(),
            is_cpu,
            assigned_layers: 32,
            available_vram_bytes: available_vram,
            delta_token_bytes: 1024 * 64,
            max_tokens: if is_cpu || available_vram == 0 {
                0
            } else {
                (available_vram / (1024 * 64)) as usize
            },
        };
        Self::new(vec![stage], GangLinkType::NVLink, psi_memory_some)
    }

    /// Evaluate headroom and compute effective token budget.
    pub fn evaluate(&self) -> GangHeadroomResult {
        let has_cpu = self.stages.iter().any(|s| s.is_cpu);
        if self.stages.is_empty() || has_cpu {
            return GangHeadroomResult {
                weakest_link_tokens: 0,
                effective_tokens: 0,
                phi_psi: 0.0,
                beta_link: self.link_type.factor(),
                cpu_veto: has_cpu,
                effort: ReasoningEffort::None,
            };
        }

        // Weakest-link token capacity: min over all devices in gang
        let weakest = self.stages.iter().map(|s| s.max_tokens).min().unwrap_or(0);

        // PSI memory pressure factor with strict NaN and bounds safety
        let psi = if self.psi_memory_some.is_nan() || self.psi_memory_some < 0.0 {
            0.0
        } else {
            self.psi_memory_some.min(100.0) as f64
        };
        let phi_psi = if psi >= 60.0 {
            0.0
        } else {
            (1.0 - (psi / 60.0)).clamp(0.0, 1.0)
        };

        let beta_link = self.link_type.factor();
        let effective = ((weakest as f64) * phi_psi * beta_link).floor() as usize;

        let effort = if effective >= 16384 {
            ReasoningEffort::Max
        } else if effective >= 8192 {
            ReasoningEffort::High
        } else if effective >= 2048 {
            ReasoningEffort::Medium
        } else if effective >= 512 {
            ReasoningEffort::Low
        } else {
            ReasoningEffort::None
        };

        GangHeadroomResult {
            weakest_link_tokens: weakest,
            effective_tokens: effective,
            phi_psi,
            beta_link,
            cpu_veto: false,
            effort,
        }
    }

    /// Resolve reasoning effort tier, taking into account if model is a thinking model.
    pub fn resolve_effort(&self, is_thinking_model: bool) -> ReasoningEffort {
        if !is_thinking_model {
            return ReasoningEffort::None;
        }
        self.evaluate().effort
    }
}

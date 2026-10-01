//! Dual-watermark hysteresis controller for VRAM memory management.
//!
//! Enforces dual watermarks:
//! - High watermark at 85% VRAM triggers batch spill down to 70%.
//! - Low watermark at 65% enables prefetch only after a continuous 5.0s dwell cooldown timer.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

pub const DEFAULT_HIGH_WATERMARK: f64 = 0.85;
pub const DEFAULT_SPILL_TARGET: f64 = 0.70;
pub const DEFAULT_LOW_WATERMARK: f64 = 0.65;
pub const DEFAULT_DWELL_COOLDOWN: Duration = Duration::from_millis(5000);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WatermarkDecision {
    None,
    Spill {
        current_used_bytes: u64,
        target_bytes: u64,
        bytes_to_evict: u64,
        used_ratio: f64,
    },
    Prefetch {
        current_used_bytes: u64,
        used_ratio: f64,
        dwell_elapsed: Duration,
    },
    CoolingDown {
        current_used_bytes: u64,
        used_ratio: f64,
        dwell_elapsed: Duration,
        dwell_remaining: Duration,
    },
}

#[derive(Debug, Clone)]
pub struct DualWatermarkController {
    pub high_watermark: f64,
    pub spill_target: f64,
    pub low_watermark: f64,
    pub dwell_cooldown: Duration,
    is_spilling: bool,
    low_watermark_start: Option<Instant>,
    last_spill_instant: Option<Instant>,
}

impl Default for DualWatermarkController {
    fn default() -> Self {
        Self::new()
    }
}

impl DualWatermarkController {
    pub fn new() -> Self {
        Self::with_thresholds(
            DEFAULT_HIGH_WATERMARK,
            DEFAULT_SPILL_TARGET,
            DEFAULT_LOW_WATERMARK,
            DEFAULT_DWELL_COOLDOWN,
        )
    }

    pub fn with_thresholds(
        high_watermark: f64,
        spill_target: f64,
        low_watermark: f64,
        dwell_cooldown: Duration,
    ) -> Self {
        Self {
            high_watermark,
            spill_target,
            low_watermark,
            dwell_cooldown,
            is_spilling: false,
            low_watermark_start: None,
            last_spill_instant: None,
        }
    }

    pub fn is_cooling_down(&self) -> bool {
        self.low_watermark_start.is_some()
    }

    pub fn is_spilling(&self) -> bool {
        self.is_spilling
    }

    pub fn last_spill(&self) -> Option<Instant> {
        self.last_spill_instant
    }

    /// Evaluates current VRAM usage against dual watermarks.
    ///
    /// - At or above high watermark (>= 85%): triggers batch spill down to 70%.
    /// - Continuing hysteresis: remains in spilling state until <= 70% is reached.
    /// - At or below low watermark (<= 65%): enables prefetch only after a continuous 5.0s dwell timer.
    /// - Hysteresis deadband: cancels low watermark dwell timer and holds steady.
    pub fn evaluate(
        &mut self,
        vram_used_bytes: u64,
        vram_total_bytes: u64,
        now: Instant,
    ) -> WatermarkDecision {
        if vram_total_bytes == 0 {
            return WatermarkDecision::None;
        }

        let ratio = vram_used_bytes as f64 / vram_total_bytes as f64;

        if ratio >= self.high_watermark || (self.is_spilling && ratio > self.spill_target) {
            self.is_spilling = true;
            self.low_watermark_start = None;
            self.last_spill_instant = Some(now);

            let target_bytes = (vram_total_bytes as f64 * self.spill_target) as u64;
            let bytes_to_evict = vram_used_bytes.saturating_sub(target_bytes);
            return WatermarkDecision::Spill {
                current_used_bytes: vram_used_bytes,
                target_bytes,
                bytes_to_evict,
                used_ratio: ratio,
            };
        }

        if self.is_spilling && ratio <= self.spill_target {
            self.is_spilling = false;
        }

        if ratio <= self.low_watermark {
            let start = *self.low_watermark_start.get_or_insert(now);
            let elapsed = now.saturating_duration_since(start);

            if elapsed >= self.dwell_cooldown {
                return WatermarkDecision::Prefetch {
                    current_used_bytes: vram_used_bytes,
                    used_ratio: ratio,
                    dwell_elapsed: elapsed,
                };
            } else {
                return WatermarkDecision::CoolingDown {
                    current_used_bytes: vram_used_bytes,
                    used_ratio: ratio,
                    dwell_elapsed: elapsed,
                    dwell_remaining: self.dwell_cooldown.saturating_sub(elapsed),
                };
            }
        }

        // Hysteresis deadband: between low and high watermarks
        self.low_watermark_start = None;
        WatermarkDecision::None
    }

    /// Reset any internal dwell timers and hysteresis state
    pub fn reset(&mut self) {
        self.is_spilling = false;
        self.low_watermark_start = None;
        self.last_spill_instant = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_watermark_triggers_spill_to_70() {
        let mut ctrl = DualWatermarkController::new();
        let total = 10_000_000_000u64;
        let used = 8_600_000_000u64; // 86% (> 85%)
        let decision = ctrl.evaluate(used, total, Instant::now());
        match decision {
            WatermarkDecision::Spill { target_bytes, bytes_to_evict, .. } => {
                assert_eq!(target_bytes, 7_000_000_000);
                assert_eq!(bytes_to_evict, 1_600_000_000);
            }
            other => panic!("Expected Spill decision, got {:?}", other),
        }
    }

    #[test]
    fn test_hysteresis_spill_continuity() {
        let mut ctrl = DualWatermarkController::new();
        let total = 10_000_000_000u64;
        let t0 = Instant::now();
        assert!(matches!(ctrl.evaluate(8_600_000_000, total, t0), WatermarkDecision::Spill { .. }));
        assert!(ctrl.is_spilling());
        // At 75% (in deadband), hysteresis must hold spilling state until <= 70%
        let d = ctrl.evaluate(7_500_000_000, total, t0 + Duration::from_millis(100));
        assert!(matches!(d, WatermarkDecision::Spill { bytes_to_evict: 500_000_000, .. }));
        // Once <= 70%, spilling clears
        let d_end = ctrl.evaluate(6_900_000_000, total, t0 + Duration::from_millis(200));
        assert!(!ctrl.is_spilling());
        assert_eq!(d_end, WatermarkDecision::None);
    }

    #[test]
    fn test_jitter_resets_dwell_timer() {
        let mut ctrl = DualWatermarkController::new();
        let total = 10_000_000_000u64;
        let t0 = Instant::now();
        assert!(matches!(ctrl.evaluate(6_000_000_000, total, t0), WatermarkDecision::CoolingDown { .. }));
        // Spike to 66% cancels dwell timer
        assert_eq!(ctrl.evaluate(6_600_000_000, total, t0 + Duration::from_secs(3)), WatermarkDecision::None);
        // Drops back to 60%: timer resets to full duration
        let d = ctrl.evaluate(6_000_000_000, total, t0 + Duration::from_secs(4));
        assert!(matches!(d, WatermarkDecision::CoolingDown { dwell_remaining, .. } if dwell_remaining == Duration::from_millis(5000)));
    }

    #[test]
    fn test_deadband_cancels_dwell_timer() {
        let mut ctrl = DualWatermarkController::new();
        let total = 10_000_000_000u64;
        let t0 = Instant::now();
        let d1 = ctrl.evaluate(6_000_000_000, total, t0);
        assert!(matches!(d1, WatermarkDecision::CoolingDown { .. }));
        let d2 = ctrl.evaluate(7_500_000_000, total, t0 + Duration::from_secs(2));
        assert_eq!(d2, WatermarkDecision::None);
        assert!(!ctrl.is_cooling_down());
    }

    #[test]
    fn test_low_watermark_requires_5s_continuous_dwell() {
        let mut ctrl = DualWatermarkController::new();
        let total = 10_000_000_000u64;
        let t0 = Instant::now();
        let d0 = ctrl.evaluate(6_000_000_000, total, t0);
        assert!(matches!(d0, WatermarkDecision::CoolingDown { .. }));
        let d4 = ctrl.evaluate(6_000_000_000, total, t0 + Duration::from_millis(4900));
        match d4 {
            WatermarkDecision::CoolingDown { dwell_remaining, .. } => {
                assert_eq!(dwell_remaining, Duration::from_millis(100));
            }
            other => panic!("Expected CoolingDown, got {:?}", other),
        }
        let d5 = ctrl.evaluate(6_000_000_000, total, t0 + Duration::from_millis(5000));
        assert!(matches!(d5, WatermarkDecision::Prefetch { .. }));
    }
}

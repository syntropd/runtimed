//! Linux Kernel PSI Pressure Stall Information monitor and memory load-shedding.
//!
//! Samples `/proc/pressure/memory` at bounded intervals (default 5s) to detect
//! memory stalls before PID 1 or `systemd-oomd` trigger forced SIGKILL terminations.
//! Under severe stall pressure, it proactively evicts resident idle models
//! via `ModelManager::unload_idle(0)` to shed gigabytes of unpinned weights.

use crate::model::ModelManager;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tracing::{info, warn};

/// Default kernel Pressure Stall Information file path for memory.
pub const DEFAULT_PSI_MEMORY_PATH: &str = "/proc/pressure/memory";

/// Parses the `avg10` value for a given line prefix (`some` or `full`) from PSI content.
pub fn parse_psi_avg10(content: &str, line_prefix: &str) -> Option<f32> {
    for line in content.lines() {
        if line.starts_with(line_prefix) {
            for part in line.split_whitespace() {
                if let Some(val_str) = part.strip_prefix("avg10=") {
                    return val_str.parse::<f32>().ok();
                }
            }
        }
    }
    None
}

/// Reads current memory PSI stall percentages (`some`, `full`) from the specified path.
pub fn read_memory_psi(path: &Path) -> Option<(f32, f32)> {
    if !path.exists() {
        return None;
    }
    let content = fs::read_to_string(path).ok()?;
    let some = parse_psi_avg10(&content, "some").unwrap_or(0.0);
    let full = parse_psi_avg10(&content, "full").unwrap_or(0.0);
    Some((some, full))
}

/// Evaluates memory pressure and triggers proactive eviction if stalled.
pub fn evaluate_and_shed_memory(manager: &ModelManager, some: f32, full: f32) -> bool {
    // If full stalls exceed 5.0% or partial stalls exceed 25.0%, shed immediately
    if full > 5.0 || some > 25.0 {
        warn!(
            mem_some = some,
            mem_full = full,
            "PSI memory pressure spike detected; triggering proactive load-shedding"
        );
        let freed = manager.unload_idle(0);
        if !freed.is_empty() {
            let total: usize = freed.iter().map(|(_, b)| b).sum();
            info!(models = freed.len(), bytes = total, "PSI load-shedding evicted idle models");
        }
        true
    } else {
        false
    }
}

/// Evaluates memory pressure with two-tier KV cache spilling and model shedding.
pub fn evaluate_and_shed_with_spill(
    manager: &ModelManager,
    spiller: Option<&runtimed_model::cache::SpillManager>,
    cache: Option<&mut runtimed_model::cache::PagedKvCache>,
    some: f32,
    full: f32,
) -> bool {
    let mut acted = false;
    // Moderate stall (some > 10.0 or full > 2.0): spill L1 KV cache blocks to L2 host RAM
    if some > 10.0 || full > 2.0 {
        if let (Some(spill), Some(c)) = (spiller, cache) {
            if let Ok(count) = spill.shed_pressure_spill(c, 0.5) {
                if count > 0 {
                    info!(spilled = count, "PSI pressure mitigation: spilled KV blocks to L2 host RAM");
                    acted = true;
                }
            }
        }
    }
    // Severe stall (some > 25.0 or full > 5.0): unload idle models
    if evaluate_and_shed_memory(manager, some, full) {
        acted = true;
    }
    acted
}

/// Spawns the PSI memory pressure watcher background task.
pub fn spawn_psi_monitor(
    manager: Arc<ModelManager>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let psi_path = std::env::var("PSI_MEMORY_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from(DEFAULT_PSI_MEMORY_PATH));

        let tick = Duration::from_secs(5);
        loop {
            tokio::select! {
                biased;
                _ = shutdown.changed() => return,
                _ = sleep(tick) => {
                    if let Some((some, full)) = read_memory_psi(&psi_path) {
                        evaluate_and_shed_memory(&manager, some, full);
                    }
                }
            }
        }
    })
}

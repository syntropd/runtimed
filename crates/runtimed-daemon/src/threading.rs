//! Automatic CPU topology detection and thread pool initialization for Candle & Rayon.

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use tracing::{info, warn};

/// Environmental variable name governing Candle matrix multiplication threads.
pub const CANDLE_NUM_THREADS_VAR: &str = "CANDLE_NUM_THREADS";

/// Environmental variable name governing Rayon worker thread count.
pub const RAYON_NUM_THREADS_VAR: &str = "RAYON_NUM_THREADS";

/// Detected CPU topology and recommended parallel thread allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuTopology {
    /// Available logical cores according to affinity mask and cgroups.
    pub logical_cores: usize,
    /// Detected physical core count (or logical if unavailable).
    pub physical_cores: usize,
    /// Recommended thread pool size (physical cores clamped to available parallelism).
    pub default_threads: usize,
}

/// Detects physical core count from Linux sysfs topology.
fn detect_physical_cores_sysfs() -> Option<usize> {
    let entries = fs::read_dir("/sys/devices/system/cpu").ok()?;
    let mut cores = HashSet::new();

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let name = file_name.to_str()?;
        if !name.starts_with("cpu") {
            continue;
        }
        let suffix = &name[3..];
        if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }

        let topo = entry.path().join("topology");
        let core_id = fs::read_to_string(topo.join("core_id")).ok()?;
        let pkg_id = fs::read_to_string(topo.join("physical_package_id"))
            .unwrap_or_else(|_| "0".to_string());

        cores.insert((pkg_id.trim().to_string(), core_id.trim().to_string()));
    }

    if cores.is_empty() {
        None
    } else {
        Some(cores.len())
    }
}

/// Parses /proc/cpuinfo as a fallback to detect physical cores.
fn detect_physical_cores_procinfo(path: &Path) -> Option<usize> {
    let content = fs::read_to_string(path).ok()?;
    let mut cores = HashSet::new();
    let mut current_pkg = "0".to_string();
    let mut current_core = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if let Some(core_id) = current_core.take() {
                cores.insert((current_pkg.clone(), core_id));
            }
            continue;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            let key = k.trim();
            let val = v.trim();
            if key == "physical id" {
                current_pkg = val.to_string();
            } else if key == "core id" {
                current_core = Some(val.to_string());
            }
        }
    }
    if let Some(core_id) = current_core {
        cores.insert((current_pkg, core_id));
    }

    if cores.is_empty() {
        None
    } else {
        Some(cores.len())
    }
}

/// Detects CPU topology, respecting cgroup limits and thread affinity.
pub fn detect_cpu_topology() -> CpuTopology {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let physical = detect_physical_cores_sysfs()
        .or_else(|| detect_physical_cores_procinfo(Path::new("/proc/cpuinfo")))
        .unwrap_or(logical);

    let default_threads = physical.min(logical).max(1);

    CpuTopology {
        logical_cores: logical,
        physical_cores: physical,
        default_threads,
    }
}

/// Initializes CPU threading for Candle and Rayon.
///
/// If `CANDLE_NUM_THREADS` or `RAYON_NUM_THREADS` are unset, auto-populates
/// them with the physical core count (clamped to available logical parallelism).
/// Builds Rayon's global thread pool with the effective thread count.
pub fn init_threading() -> CpuTopology {
    let topology = detect_cpu_topology();

    let candle_val = match std::env::var(CANDLE_NUM_THREADS_VAR) {
        Ok(v) if !v.trim().is_empty() => {
            info!("Retaining operator {}={}", CANDLE_NUM_THREADS_VAR, v);
            v
        }
        _ => {
            let s = topology.default_threads.to_string();
            std::env::set_var(CANDLE_NUM_THREADS_VAR, &s);
            info!("Auto-configured {}={}", CANDLE_NUM_THREADS_VAR, s);
            s
        }
    };
    let _ = candle_val;

    let rayon_val = match std::env::var(RAYON_NUM_THREADS_VAR) {
        Ok(v) if !v.trim().is_empty() => {
            info!("Retaining operator {}={}", RAYON_NUM_THREADS_VAR, v);
            v
        }
        _ => {
            let s = topology.default_threads.to_string();
            std::env::set_var(RAYON_NUM_THREADS_VAR, &s);
            info!("Auto-configured {}={}", RAYON_NUM_THREADS_VAR, s);
            s
        }
    };

    let effective_rayon_threads = rayon_val
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
        .unwrap_or(topology.default_threads);

    match rayon::ThreadPoolBuilder::new()
        .num_threads(effective_rayon_threads)
        .build_global()
    {
        Ok(()) => {
            info!(
                "Configured Rayon global pool: {} threads (physical cores={}, logical cores={})",
                effective_rayon_threads, topology.physical_cores, topology.logical_cores
            );
        }
        Err(e) => {
            warn!("Rayon global thread pool already initialized: {}", e);
        }
    }

    topology
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_detection_sane() {
        let topo = detect_cpu_topology();
        assert!(topo.logical_cores >= 1);
        assert!(topo.physical_cores >= 1);
        assert!(topo.default_threads >= 1);
        assert!(topo.default_threads <= topo.logical_cores);
    }

    #[test]
    fn test_parse_procinfo_sample() {
        let sample = "\
processor : 0\nphysical id : 0\ncore id : 0\n\n\
processor : 1\nphysical id : 0\ncore id : 0\n\n\
processor : 2\nphysical id : 0\ncore id : 1\n\n\
processor : 3\nphysical id : 0\ncore id : 1\n";
        let dir = match tempfile::tempdir() {
            Ok(d) => d,
            Err(_) => return,
        };
        let path = dir.path().join("cpuinfo");
        if fs::write(&path, sample).is_ok() {
            let physical = detect_physical_cores_procinfo(&path);
            assert_eq!(physical, Some(2));
        }
    }

    #[test]
    fn test_init_threading_execution() {
        let topo = init_threading();
        assert!(topo.default_threads >= 1);
        let candle_env = std::env::var(CANDLE_NUM_THREADS_VAR);
        assert!(candle_env.is_ok());
        let rayon_env = std::env::var(RAYON_NUM_THREADS_VAR);
        assert!(rayon_env.is_ok());
    }
}

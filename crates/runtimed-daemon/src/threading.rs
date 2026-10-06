//! Automatic CPU topology detection and thread pool initialization for Candle & Rayon.

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use tracing::{info, warn};

pub const CANDLE_NUM_THREADS_VAR: &str = "CANDLE_NUM_THREADS";
pub const RAYON_NUM_THREADS_VAR: &str = "RAYON_NUM_THREADS";

/// Detected CPU topology and recommended parallel thread allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuTopology {
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub default_threads: usize,
}

fn detect_cgroup_quota_cgroupv2(cgroup_content: &str, base: &Path) -> Option<usize> {
    for line in cgroup_content.lines() {
        let rel = line.strip_prefix("0::")?.trim().trim_start_matches('/');
        let mut cur = base.join(rel);
        while cur.starts_with(base) {
            if let Ok(content) = fs::read_to_string(cur.join("cpu.max")) {
                let mut parts = content.split_whitespace();
                let quota_s = parts.next()?;
                if quota_s != "max" {
                    let q: u64 = quota_s.parse().ok()?;
                    let p: u64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(100_000);
                    if p > 0 { return Some((q / p).max(1) as usize); }
                }
            }
            if !cur.pop() { break; }
        }
    }
    None
}

fn detect_cgroup_quota_cgroupv1(base: &Path) -> Option<usize> {
    let q: i64 = fs::read_to_string(base.join("cpu/cpu.cfs_quota_us")).ok()?.trim().parse().ok()?;
    if q <= 0 { return None; }
    let p: u64 = fs::read_to_string(base.join("cpu/cpu.cfs_period_us")).ok()?.trim().parse().ok().unwrap_or(100_000);
    (p > 0).then(|| (q as u64 / p).max(1) as usize)
}

pub fn detect_cgroup_quota() -> Option<usize> {
    let cgroup_data = fs::read_to_string("/proc/self/cgroup").ok()?;
    let base = Path::new("/sys/fs/cgroup");
    detect_cgroup_quota_cgroupv2(&cgroup_data, base).or_else(|| detect_cgroup_quota_cgroupv1(base))
}

fn detect_physical_cores_sysfs_dir(dir: &Path) -> Option<usize> {
    let entries = fs::read_dir(dir).ok()?;
    let mut cores = HashSet::new();

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else { continue; };
        if !name.starts_with("cpu") { continue; }
        let suffix = &name[3..];
        if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_digit()) { continue; }

        let topo = entry.path().join("topology");
        let Ok(core_id) = fs::read_to_string(topo.join("core_id")) else { continue; };
        let pkg_id = fs::read_to_string(topo.join("physical_package_id")).unwrap_or_else(|_| "0".into());
        cores.insert((pkg_id.trim().to_string(), core_id.trim().to_string()));
    }

    if cores.is_empty() { None } else { Some(cores.len()) }
}

fn detect_physical_cores_procinfo(path: &Path) -> Option<usize> {
    let content = fs::read_to_string(path).ok()?;
    let mut cores = HashSet::new();
    let mut current_pkg = "0".to_string();
    let mut current_core = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if let Some(core_id) = current_core.take() { cores.insert((current_pkg.clone(), core_id)); }
            continue;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            match k.trim() {
                "physical id" => current_pkg = v.trim().to_string(),
                "core id" => current_core = Some(v.trim().to_string()),
                _ => {}
            }
        }
    }
    if let Some(core_id) = current_core { cores.insert((current_pkg, core_id)); }
    if cores.is_empty() { None } else { Some(cores.len()) }
}

/// Detects CPU topology, respecting cgroup limits and thread affinity.
pub fn detect_cpu_topology() -> CpuTopology {
    let mut logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    if let Some(limit) = detect_cgroup_quota() { logical = logical.min(limit); }

    let physical = detect_physical_cores_sysfs_dir(Path::new("/sys/devices/system/cpu"))
        .or_else(|| detect_physical_cores_procinfo(Path::new("/proc/cpuinfo")))
        .unwrap_or(logical);

    CpuTopology { logical_cores: logical, physical_cores: physical, default_threads: physical.min(logical).max(1) }
}

fn parse_thread_var(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse::<usize>().ok().filter(|&n| n > 0)
}

/// Initializes CPU threading for Candle and Rayon.
pub fn init_threading() -> CpuTopology {
    let mut topology = detect_cpu_topology();
    let candle_parsed = parse_thread_var(CANDLE_NUM_THREADS_VAR);
    let rayon_parsed = parse_thread_var(RAYON_NUM_THREADS_VAR);

    let (_candle_threads, rayon_threads) = match (candle_parsed, rayon_parsed) {
        (Some(c), Some(r)) => {
            info!("Retaining operator {}={} and {}={}", CANDLE_NUM_THREADS_VAR, c, RAYON_NUM_THREADS_VAR, r);
            (c, r)
        }
        (Some(c), None) => {
            info!("Retaining operator {}={}, aligning {}={}", CANDLE_NUM_THREADS_VAR, c, RAYON_NUM_THREADS_VAR, c);
            std::env::set_var(RAYON_NUM_THREADS_VAR, c.to_string());
            (c, c)
        }
        (None, Some(r)) => {
            info!("Retaining operator {}={}, aligning {}={}", RAYON_NUM_THREADS_VAR, r, CANDLE_NUM_THREADS_VAR, r);
            std::env::set_var(CANDLE_NUM_THREADS_VAR, r.to_string());
            (r, r)
        }
        (None, None) => {
            let s = topology.default_threads.to_string();
            std::env::set_var(CANDLE_NUM_THREADS_VAR, &s);
            std::env::set_var(RAYON_NUM_THREADS_VAR, &s);
            info!("Auto-configured {}={} and {}={}", CANDLE_NUM_THREADS_VAR, s, RAYON_NUM_THREADS_VAR, s);
            (topology.default_threads, topology.default_threads)
        }
    };

    match rayon::ThreadPoolBuilder::new().num_threads(rayon_threads).build_global() {
        Ok(()) => info!("Configured Rayon global pool: {} threads", rayon_threads),
        Err(e) => warn!("Rayon global thread pool already initialized: {}", e),
    }

    topology.default_threads = rayon_threads;
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
    fn test_sysfs_with_offline_core() {
        let dir = match tempfile::tempdir() { Ok(d) => d, Err(_) => return };
        let cpu0_topo = dir.path().join("cpu0/topology");
        let cpu1_dir = dir.path().join("cpu1");
        fs::create_dir_all(&cpu0_topo).ok();
        fs::create_dir_all(&cpu1_dir).ok();
        fs::write(cpu0_topo.join("core_id"), "0\n").ok();
        fs::write(cpu0_topo.join("physical_package_id"), "0\n").ok();
        let physical = detect_physical_cores_sysfs_dir(dir.path());
        assert_eq!(physical, Some(1));
    }

    #[test]
    fn test_parse_procinfo_sample() {
        let sample = "\
processor : 0\nphysical id : 0\ncore id : 0\n\n\
processor : 1\nphysical id : 0\ncore id : 0\n\n\
processor : 2\nphysical id : 0\ncore id : 1\n\n\
processor : 3\nphysical id : 0\ncore id : 1\n";
        let dir = match tempfile::tempdir() { Ok(d) => d, Err(_) => return };
        let path = dir.path().join("cpuinfo");
        if fs::write(&path, sample).is_ok() {
            let physical = detect_physical_cores_procinfo(&path);
            assert_eq!(physical, Some(2));
        }
    }

    #[test]
    fn test_cgroupv2_quota_parsing() {
        let dir = match tempfile::tempdir() { Ok(d) => d, Err(_) => return };
        let sub = dir.path().join("docker/container123");
        fs::create_dir_all(&sub).ok();
        fs::write(sub.join("cpu.max"), "200000 100000\n").ok();
        let quota = detect_cgroup_quota_cgroupv2("0::/docker/container123", dir.path());
        assert_eq!(quota, Some(2));
    }

    #[test]
    fn test_cgroupv1_quota_parsing() {
        let dir = match tempfile::tempdir() { Ok(d) => d, Err(_) => return };
        let cpu_dir = dir.path().join("cpu");
        fs::create_dir_all(&cpu_dir).ok();
        fs::write(cpu_dir.join("cpu.cfs_quota_us"), "400000\n").ok();
        fs::write(cpu_dir.join("cpu.cfs_period_us"), "100000\n").ok();
        let quota = detect_cgroup_quota_cgroupv1(dir.path());
        assert_eq!(quota, Some(4));
    }

    #[test]
    fn test_init_threading_alignment() {
        std::env::set_var(CANDLE_NUM_THREADS_VAR, "3");
        std::env::remove_var(RAYON_NUM_THREADS_VAR);
        let topo = init_threading();
        assert_eq!(topo.default_threads, 3);
        assert_eq!(std::env::var(RAYON_NUM_THREADS_VAR).as_deref(), Ok("3"));
        std::env::remove_var(CANDLE_NUM_THREADS_VAR);
        std::env::remove_var(RAYON_NUM_THREADS_VAR);
    }
}

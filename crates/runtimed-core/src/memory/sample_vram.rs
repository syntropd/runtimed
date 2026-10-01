//! DRM and system VRAM telemetry sampler wired to DualWatermarkController and SpillManager.

use super::watermark::{DualWatermarkController, WatermarkDecision};
use crate::error::RuntimedError;
use candle_core::Tensor;
use runtimed_model::cache::{PagedKvCache, SpillManager};
use runtimed_model::decode::generate::{last_row, TextModel};
use std::{fs::{self, OpenOptions}, io::Read, os::unix::fs::OpenOptionsExt, path::Path, time::Instant};
use tracing::{info, warn};

pub const DEFAULT_DRM_PATH: &str = "/sys/class/drm";
pub const DEFAULT_MEMINFO_PATH: &str = "/proc/meminfo";
pub const DEFAULT_CHECK_INTERVAL: usize = 1;

fn read_nonblocking_u64(path: &Path) -> Option<u64> {
    let mut file = OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK).open(path).ok()?;
    let mut buf = [0u8; 64];
    let n = file.read(&mut buf).ok()?;
    if n == 0 { return None; }
    std::str::from_utf8(&buf[..n]).ok()?.trim().parse::<u64>().ok()
}

/// Reads DRM sysfs VRAM used and total bytes, falling back to system memory if unavailable.
pub fn sample_vram_metrics_from(drm_base: &Path, meminfo_path: &Path) -> (u64, u64) {
    let mut best: Option<(u64, u64)> = None;
    if let Ok(entries) = fs::read_dir(drm_base) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("renderD") {
                let dev = entry.path().join("device");
                let used = read_nonblocking_u64(&dev.join("mem_info_vram_used"))
                    .or_else(|| read_nonblocking_u64(&dev.join("tile0/memory/vram0/used")))
                    .or_else(|| read_nonblocking_u64(&dev.join("lmem_used_bytes")));
                let total = read_nonblocking_u64(&dev.join("mem_info_vram_total"))
                    .or_else(|| read_nonblocking_u64(&dev.join("tile0/memory/vram0/total")))
                    .or_else(|| read_nonblocking_u64(&dev.join("tile0/physical_vram_size")))
                    .or_else(|| read_nonblocking_u64(&dev.join("lmem_total_bytes")));
                if let (Some(u), Some(t)) = (used, total) {
                    if t > 0 && best.is_none_or(|(_, cur_t)| t > cur_t) {
                        best = Some((u, t));
                    }
                }
            }
        }
    }
    if let Some(metrics) = best {
        return metrics;
    }

    if let Ok(content) = fs::read_to_string(meminfo_path) {
        let (mut total, mut avail) = (0u64, 0u64);
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                total = rest.split_whitespace().next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0) * 1024;
            } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                avail = rest.split_whitespace().next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0) * 1024;
            }
        }
        if total > 0 {
            return (total.saturating_sub(avail), total);
        }
    }
    (0, 0)
}

/// Sample DRM or system VRAM metrics using standard host paths.
pub fn sample_vram_metrics() -> (u64, u64) {
    sample_vram_metrics_from(Path::new(DEFAULT_DRM_PATH), Path::new(DEFAULT_MEMINFO_PATH))
}

/// Evaluates VRAM watermarks and triggers spill_blocks or restore_origin_blocks on PagedKvCache.
pub fn evaluate_and_spill_with_metrics(
    controller: &mut DualWatermarkController,
    spiller: &SpillManager,
    cache: &mut PagedKvCache,
    vram_used: u64,
    vram_total: u64,
    now: Instant,
) -> Result<WatermarkDecision, RuntimedError> {
    let decision = controller.evaluate(vram_used, vram_total, now);
    match &decision {
        WatermarkDecision::Spill { bytes_to_evict, .. } => {
            let bb = cache.blocks.first().map(|b| (b.k.elem_count() + b.v.elem_count()) * b.k.dtype().size_in_bytes()).unwrap_or(1024 * 1024);
            let count = (*bytes_to_evict as usize).checked_div(bb).unwrap_or(0).max(1);
            let spilled = spiller.spill_blocks(cache, count).map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?;
            warn!(spilled, bytes_to_evict, "High watermark exceeded: spilled L1 blocks to L2 host RAM");
        }
        WatermarkDecision::Prefetch { .. } => {
            let restored = spiller.restore_origin_blocks(cache, 1).map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?;
            if restored > 0 { info!(restored, "Low watermark dwell satisfied: restored L2 blocks to accelerator"); }
        }
        _ => {}
    }
    Ok(decision)
}

/// Evaluates live DRM/system VRAM against DualWatermarkController and spills/restores cache blocks.
pub fn evaluate_and_spill(
    controller: &mut DualWatermarkController,
    spiller: &SpillManager,
    cache: &mut PagedKvCache,
    now: Instant,
) -> Result<WatermarkDecision, RuntimedError> {
    let (used, total) = sample_vram_metrics();
    evaluate_and_spill_with_metrics(controller, spiller, cache, used, total, now)
}

/// Managed decode loop that samples DRM/system VRAM periodically and triggers cache spills/restores.
#[allow(clippy::too_many_arguments)]
pub fn decode_loop_managed<M: TextModel>(
    model: &mut M,
    first_logits: &Tensor,
    prompt_len: usize,
    eos: &[u32],
    max_new: usize,
    mut next: impl FnMut(&Tensor) -> runtimed_model::Result<u32>,
    controller: &mut DualWatermarkController,
    spiller: &SpillManager,
    cache: &mut PagedKvCache,
    check_interval: usize,
) -> runtimed_model::Result<Vec<u32>> {
    if max_new == 0 { return Ok(Vec::new()); }
    let mut out = Vec::new();
    let mut id = next(&last_row(first_logits)?)?;
    let mut pos = prompt_len;
    let mut step = 0usize;

    let eval_step = |ctrl: &mut DualWatermarkController, mdl: &mut M, c: &mut PagedKvCache| {
        match evaluate_and_spill(ctrl, spiller, c, Instant::now()) {
            Ok(WatermarkDecision::Spill { .. }) => {
                let _ = mdl.spill_layers(1);
            }
            Ok(WatermarkDecision::Prefetch { .. }) => {
                let _ = mdl.prefetch_layers(1);
            }
            Err(e) => warn!("Watermark evaluation error: {e}"),
            _ => {}
        }
    };

    if check_interval > 0 {
        eval_step(controller, model, cache);
    }

    loop {
        out.push(id);
        if out.len() >= max_new || eos.contains(&id) {
            break;
        }

        step += 1;
        if check_interval > 0 && step.is_multiple_of(check_interval) {
            eval_step(controller, model, cache);
        }

        let logits = model.forward(std::slice::from_ref(&id), pos)?;
        pos += 1;
        id = next(&last_row(&logits)?)?;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use runtimed_model::cache::StorageTier;
    use std::time::Duration;
    use tempfile::tempdir;

    struct StubModel { vocab: usize, id: u32 }
    impl TextModel for StubModel {
        fn forward(&mut self, ids: &[u32], _q0: usize) -> runtimed_model::Result<Tensor> {
            let mut row = vec![0.0f32; self.vocab];
            row[self.id as usize] = 9.0;
            let flat: Vec<f32> = (0..ids.len().max(1)).flat_map(|_| row.clone()).collect();
            Ok(Tensor::from_vec(flat, (1, ids.len().max(1), self.vocab), &Device::Cpu)?)
        }
        fn reset(&mut self) {}
    }

    #[test]
    fn test_sample_vram_metrics_mock() {
        let dir = tempdir().unwrap();
        let meminfo = dir.path().join("meminfo");
        fs::write(&meminfo, "MemTotal:       16000000 kB\nMemAvailable:    4000000 kB\n").unwrap();
        let (used, total) = sample_vram_metrics_from(dir.path(), &meminfo);
        assert_eq!(total, 16_000_000 * 1024);
        assert_eq!(used, 12_000_000 * 1024);

        let dev_dir = dir.path().join("renderD128/device/tile0");
        fs::create_dir_all(&dev_dir).unwrap();
        fs::write(dev_dir.join("physical_vram_size"), "8589934592\n").unwrap();
        let mem_dir = dev_dir.join("memory/vram0");
        fs::create_dir_all(&mem_dir).unwrap();
        fs::write(mem_dir.join("used"), "4294967296\n").unwrap();
        let (xe_u, xe_t) = sample_vram_metrics_from(dir.path(), &meminfo);
        assert_eq!((xe_u, xe_t), (4294967296, 8589934592));
    }

    #[test]
    fn test_watermark_spill_and_restore_cycle() {
        let mut ctrl = DualWatermarkController::new();
        let spiller = SpillManager::new(10);
        let mut cache = PagedKvCache::new(1);
        let k = Tensor::zeros((1, 2, 1, 64), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((1, 2, 1, 64), DType::F32, &Device::Cpu).unwrap();
        cache.allocate_block(&Device::Cpu, StorageTier::L1Vram, k, v, 1);

        let total = 10_000_000_000u64;
        let t0 = Instant::now();

        // 86% triggers spill down to 70%
        let d = evaluate_and_spill_with_metrics(&mut ctrl, &spiller, &mut cache, 8_600_000_000, total, t0).unwrap();
        assert!(matches!(d, WatermarkDecision::Spill { .. }));
        assert_eq!(cache.count_tier_blocks(StorageTier::L1Vram), 0);
        assert_eq!(cache.count_tier_blocks(StorageTier::L2PinnedHost), 1);

        // 60% starts dwell cooling down
        let t1 = t0 + Duration::from_secs(1);
        let d2 = evaluate_and_spill_with_metrics(&mut ctrl, &spiller, &mut cache, 6_000_000_000, total, t1).unwrap();
        assert!(matches!(d2, WatermarkDecision::CoolingDown { .. }));

        // 60% after 5s continuous dwell triggers prefetch restore
        let t2 = t1 + Duration::from_millis(5000);
        let d3 = evaluate_and_spill_with_metrics(&mut ctrl, &spiller, &mut cache, 6_000_000_000, total, t2).unwrap();
        assert!(matches!(d3, WatermarkDecision::Prefetch { .. }));
        assert_eq!(cache.count_tier_blocks(StorageTier::L1Vram), 1);
        assert_eq!(cache.count_tier_blocks(StorageTier::L2PinnedHost), 0);

        // Rapid oscillation to deadband (75%) resets dwell cooldown
        let d_osc = evaluate_and_spill_with_metrics(&mut ctrl, &spiller, &mut cache, 7_500_000_000, total, t2 + Duration::from_secs(1)).unwrap();
        assert_eq!(d_osc, WatermarkDecision::None);
        assert!(!ctrl.is_cooling_down());
    }

    #[test]
    fn test_decode_loop_managed_execution() {
        let mut model = StubModel { vocab: 4, id: 2 };
        let mut ctrl = DualWatermarkController::new();
        let spiller = SpillManager::new(10);
        let mut cache = PagedKvCache::new(1);
        let first_logits = Tensor::from_vec(vec![0.0f32, 0.0, 9.0, 0.0], (1, 1, 4), &Device::Cpu).unwrap();

        let out = decode_loop_managed(
            &mut model, &first_logits, 1, &[3], 4,
            |l| Ok(l.to_vec1::<f32>()?.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32),
            &mut ctrl, &spiller, &mut cache, 1,
        ).unwrap();
        assert_eq!(out.len(), 4);
    }
}

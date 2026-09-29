//! Idle unload: shed resident weights after engine quiet time.
//!
//! Socket activation gives 0 MB at boot, but a warmed engine holds
//! gigabytes until something unloads it. When `RUNTIMED_IDLE_UNLOAD_SECS`
//! is set, a background task unloads every resident model (leases
//! released, VRAM/RAM freed) once the whole engine idles past the
//! limit. Unset or 0 (the default) keeps the engine warm for fast
//! answers. Models with a generation in flight are skipped, never
//! yanked.

use super::loader::ModelManager;
use std::time::Instant;

/// Env knob: seconds of engine quiet before resident models unload.
/// Unset/0/invalid = disabled (default: stay warm).
pub const IDLE_UNLOAD_ENV: &str = "RUNTIMED_IDLE_UNLOAD_SECS";

/// Configured idle limit, or 0 when disabled/unparseable.
pub fn idle_limit_secs() -> u64 {
    std::env::var(IDLE_UNLOAD_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

impl ModelManager {
    /// Mark engine activity (loads and generations call this).
    pub(super) fn touch(&self) {
        if let Ok(mut last) = self.last_used.lock() {
            *last = Instant::now();
        }
    }

    /// Seconds since last generation-class use (lock failure reads
    /// as freshly used, never idle: fail warm, not unloaded).
    pub fn idle_secs(&self) -> u64 {
        self.last_used
            .lock()
            .map(|last| last.elapsed().as_secs())
            .unwrap_or(0)
    }

    /// Unload every resident model when the engine idled past
    /// `limit_secs`. Returns (name, bytes) freed. A fresh engine, a
    /// disabled (0) limit, or an empty store is a no-op. Entries
    /// with a generation in flight are skipped for the next tick.
    pub fn unload_idle(&self, limit_secs: u64) -> Vec<(String, usize)> {
        if limit_secs == 0 || self.idle_secs() < limit_secs {
            return Vec::new();
        }
        let names: Vec<String> = self
            .list_active()
            .iter()
            .map(|m| m.name.clone())
            .collect();
        let mut freed = Vec::new();
        for name in names {
            let busy = self
                .get_entry(&name)
                .map(|e| e.session.try_lock().is_err())
                .unwrap_or(false);
            if busy {
                tracing::debug!(model = %name, "idle unload deferred: generation in flight");
                continue;
            }
            match self.unload_model(&name) {
                Ok(bytes) => {
                    tracing::info!(model = %name, bytes, "idle unload");
                    freed.push((name, bytes));
                }
                Err(e) => tracing::warn!(model = %name, "idle unload failed: {e}"),
            }
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn empty_manager() -> (TempDir, ModelManager) {
        let dir = TempDir::new().unwrap();
        let manager = ModelManager::new(dir.path());
        (dir, manager)
    }

    #[test]
    fn limit_parses_seconds_or_disables() {
        std::env::set_var(IDLE_UNLOAD_ENV, "90");
        assert_eq!(idle_limit_secs(), 90);
        std::env::set_var(IDLE_UNLOAD_ENV, "soon");
        assert_eq!(idle_limit_secs(), 0);
        std::env::remove_var(IDLE_UNLOAD_ENV);
        assert_eq!(idle_limit_secs(), 0);
    }

    #[test]
    fn fresh_engine_is_not_idle() {
        let (_dir, manager) = empty_manager();
        manager.touch();
        assert!(manager.idle_secs() < 5);
        assert!(manager.unload_idle(60).is_empty());
    }

    #[test]
    fn disabled_limit_never_unloads() {
        let (_dir, manager) = empty_manager();
        assert!(manager.unload_idle(0).is_empty());
    }
}

//! Daemon lifecycle: shutdown signalling, watchdog heartbeat, and graceful
//! shutdown orchestration. Extracted from `main.rs` so the entry point stays
//! short.

use anyhow::{Context as _, Result};
use std::time::Duration;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tracing::{info, warn};

use crate::notify::{notify_stopping, notify_watchdog};

/// Bounded timeout for draining the Varlink server after shutdown signal.
pub const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

/// Lower bound for the watchdog interval (1s).
pub const WATCHDOG_MIN: Duration = Duration::from_secs(1);

/// Upper bound for the watchdog interval (300s).
pub const WATCHDOG_MAX: Duration = Duration::from_secs(300);

/// Reads `$WATCHDOG_USEC` and returns the clamped ping interval, or
/// `Duration::ZERO` when the variable is unset or zero (watchdog disabled).
pub fn watchdog_interval() -> Duration {
    let raw = match std::env::var("WATCHDOG_USEC") {
        Ok(s) => s,
        Err(_) => return Duration::ZERO,
    };
    let usec: u64 = match raw.parse() {
        Ok(v) => v,
        Err(_) => return Duration::ZERO,
    };
    if usec == 0 {
        return Duration::ZERO;
    }
    let third = Duration::from_micros(usec / 3);
    if third < WATCHDOG_MIN {
        WATCHDOG_MIN
    } else if third > WATCHDOG_MAX {
        WATCHDOG_MAX
    } else {
        third
    }
}

/// Spawns a background task that emits `WATCHDOG=1` at the configured
/// interval until the shutdown channel fires.
pub fn spawn_watchdog(mut shutdown: watch::Receiver<bool>) -> JoinHandle<()> {
    tokio::spawn(async move {
        let interval = watchdog_interval();
        if interval.is_zero() {
            return;
        }
        loop {
            tokio::select! {
                biased;
                _ = shutdown.changed() => return,
                _ = sleep(interval) => {
                    if !notify_watchdog() {
                        warn!("Watchdog ping failed (NOTIFY_SOCKET unset?)");
                    }
                }
            }
        }
    })
}

/// Awaits a join handle with a bounded timeout, returning `Ok(())` if the
/// task finished in time and `Err` otherwise. Never panics or unwraps.
pub async fn join_with_timeout<T>(handle: JoinHandle<T>, timeout: Duration) -> Result<T> {
    match tokio::time::timeout(timeout, handle).await {
        Ok(join_result) => join_result.context("daemon task panicked"),
        Err(_) => Err(anyhow::anyhow!(
            "daemon task did not finish within {:?}",
            timeout
        )),
    }
}

/// Drains the server task, then notifies systemd that the daemon is stopping.
pub async fn finish_shutdown(handle: JoinHandle<Result<()>>) {
    match join_with_timeout(handle, SHUTDOWN_TIMEOUT).await {
        Ok(Ok(())) => info!("Varlink server exited cleanly"),
        Ok(Err(e)) => warn!("Varlink server returned error: {}", e),
        Err(e) => warn!("Varlink server shutdown timed out: {}", e),
    }
    notify_stopping();
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EnvGuard {
        prev: Option<String>,
    }

    impl EnvGuard {
        fn swap(val: Option<&str>) -> Self {
            let prev = std::env::var("WATCHDOG_USEC").ok();
            match val {
                Some(v) => std::env::set_var("WATCHDOG_USEC", v),
                None => std::env::remove_var("WATCHDOG_USEC"),
            }
            Self { prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("WATCHDOG_USEC", v),
                None => std::env::remove_var("WATCHDOG_USEC"),
            }
        }
    }

    #[tokio::test]
    async fn watchdog_interval_matrix_and_disabled_spawn() {
        let _guard = EnvGuard::swap(None);
        assert_eq!(watchdog_interval(), Duration::ZERO);
        // Disabled watchdog: the spawned task returns immediately.
        let (_tx, rx) = watch::channel(false);
        spawn_watchdog(rx).await.unwrap();

        std::env::set_var("WATCHDOG_USEC", "0");
        assert_eq!(watchdog_interval(), Duration::ZERO);
        std::env::set_var("WATCHDOG_USEC", "not-a-number");
        assert_eq!(watchdog_interval(), Duration::ZERO);
        // Third of the timeout, clamped to [1s, 300s].
        std::env::set_var("WATCHDOG_USEC", "9000000");
        assert_eq!(watchdog_interval(), Duration::from_secs(3));
        std::env::set_var("WATCHDOG_USEC", "1");
        assert_eq!(watchdog_interval(), WATCHDOG_MIN);
        std::env::set_var("WATCHDOG_USEC", "999999999999");
        assert_eq!(watchdog_interval(), WATCHDOG_MAX);
    }

    #[tokio::test]
    async fn join_with_timeout_covers_done_and_stuck() {
        let fast = tokio::spawn(async { 7u32 });
        assert_eq!(join_with_timeout(fast, Duration::from_secs(5)).await.unwrap(), 7);
        let stuck = tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(30)).await;
        });
        let err = join_with_timeout(stuck, Duration::from_millis(10)).await.unwrap_err().to_string();
        assert!(err.contains("did not finish"), "{err}");
    }

    #[tokio::test]
    async fn finish_shutdown_drains_ok_and_err() {
        let ok = tokio::spawn(async { Ok::<(), anyhow::Error>(()) });
        finish_shutdown(ok).await;
        let failed = tokio::spawn(async { Err::<(), anyhow::Error>(anyhow::anyhow!("x")) });
        finish_shutdown(failed).await;
    }
}
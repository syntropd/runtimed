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
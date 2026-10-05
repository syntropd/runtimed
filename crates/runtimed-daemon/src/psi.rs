//! Linux Kernel PSI pressure stall monitor background task for runtimed daemon.

use runtimed_core::model::ModelManager;
use runtimed_core::psi::{evaluate_and_shed_memory, read_memory_psi, DEFAULT_PSI_MEMORY_PATH};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::sleep;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_psi_monitor_shutdown() {
        let (tx, rx) = watch::channel(false);
        let mgr = Arc::new(ModelManager::new(std::env::temp_dir()));
        let handle = spawn_psi_monitor(mgr, rx);
        let _ = tx.send(true);
        let res = tokio::time::timeout(Duration::from_millis(500), handle).await;
        assert!(res.is_ok());
    }
}

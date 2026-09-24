//! Entry point for runtimed: Headless Model Execution and Token Generation Daemon.

use anyhow::{Context, Result};
use runtimed_core::config::{RuntimedConfig, DEFAULT_CONFIG_PATH};
use runtimed_core::model::ModelManager;
use runtimed_daemon::activation::parse_listen_fds;
use runtimed_daemon::notify::{notify_ready, notify_stopping};
use runtimed_daemon::varlink::{Runtime1Handler, VarlinkServer};
use std::fs;
use std::os::unix::net::UnixListener as StdUnixListener;
use std::sync::Arc;
use tokio::net::UnixListener;
use tokio::signal::unix::{signal, SignalKind};
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("Starting runtimed (Headless Model Execution & Tensor Generation Daemon)");

    let args: Vec<String> = std::env::args().collect();
    let mut config_path =
        std::env::var("RUNTIMED_CONFIG").unwrap_or_else(|_| DEFAULT_CONFIG_PATH.to_string());
    if let Some(pos) = args.iter().position(|a| a == "--config" || a == "-c") {
        if let Some(path) = args.get(pos + 1) {
            config_path = path.clone();
        }
    }

    let config = RuntimedConfig::load_or_default(&config_path)
        .context("Failed loading runtimed configuration")?;

    if let Err(e) = fs::create_dir_all(&config.models_dir) {
        tracing::warn!("Could not create models directory {}: {}", config.models_dir.display(), e);
    }

    let model_manager = Arc::new(ModelManager::new(&config.models_dir));

    // Pre-warm default model
    if !config.default_model.is_empty() {
        let _ = model_manager.load_model(&config.default_model, None);
    }

    let handler = Runtime1Handler::new(Arc::clone(&model_manager));

    let mut socket_path = config.socket_path.clone();
    if let Some(pos) = args.iter().position(|a| a == "--socket" || a == "-s") {
        if let Some(path) = args.get(pos + 1) {
            socket_path = std::path::PathBuf::from(path);
        }
    }

    let activated = parse_listen_fds();
    let listener = match activated.varlink_listener {
        Some(l) => {
            info!("Adopted socket from systemd socket activation");
            l
        }
        None => {
            if let Some(parent) = socket_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if socket_path.exists() {
                let _ = fs::remove_file(&socket_path);
            }
            info!("Listening on Unix domain socket: {}", socket_path.display());
            let std_listener = StdUnixListener::bind(&socket_path)?;
            std_listener.set_nonblocking(true)?;
            UnixListener::from_std(std_listener)?
        }
    };

    let server = VarlinkServer::new(listener, handler);

    notify_ready();
    info!("runtimed successfully initialized and ready");

    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;

    tokio::select! {
        res = server.run() => {
            if let Err(e) = res {
                error!("Varlink server error: {}", e);
            }
        }
        _ = sigterm.recv() => {
            info!("Received SIGTERM, initiating shutdown");
        }
        _ = sigint.recv() => {
            info!("Received SIGINT, initiating shutdown");
        }
    }

    notify_stopping();
    info!("runtimed shutdown completed");
    Ok(())
}

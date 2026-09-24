//! Entry point for runtimed: Headless Model Execution and Token Generation Daemon.

use anyhow::{Context, Result};
use runtimed_core::config::{RuntimedConfig, DEFAULT_CONFIG_PATH};
use runtimed_core::model::ModelManager;
use runtimed_daemon::activation::parse_listen_fds;
use runtimed_daemon::notify::{notify_ready, NOTIFY_MAX};
use runtimed_daemon::runtime::{finish_shutdown, spawn_watchdog, SHUTDOWN_TIMEOUT};
use runtimed_daemon::varlink::{lookup_group, Runtime1Handler, TrustedGroup, VarlinkServer, UNRESOLVED_GID};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener as StdUnixListener;
use std::sync::Arc;
use tokio::net::UnixListener;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::watch;
use tracing::{error, info, warn};

/// Default trusted group name when `RUNTIMED_TRUSTED_GROUP` is unset.
const DEFAULT_TRUSTED_GROUP: &str = "syntrop";

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
        warn!("Could not create models directory {}: {}", config.models_dir.display(), e);
    }

    let model_manager = Arc::new(ModelManager::new(&config.models_dir));

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
        None => bind_standalone(&socket_path)?,
    };

    let trusted_name = std::env::var("RUNTIMED_TRUSTED_GROUP")
        .unwrap_or_else(|_| DEFAULT_TRUSTED_GROUP.to_string());
    let resolved_gid = lookup_group(&trusted_name);
    if resolved_gid == UNRESOLVED_GID {
        anyhow::bail!(
            "trusted group {:?} could not be resolved at startup; refusing to bind. \
             Set RUNTIMED_TRUSTED_GROUP to a valid group name and verify \
             sysusers.d/runtimed.conf is installed.",
            trusted_name
        );
    }
    let trusted_group = TrustedGroup::from_gid(resolved_gid);
    info!(
        "Trusted group resolved: name={:?} gid={}",
        trusted_name, resolved_gid
    );

    let server = VarlinkServer::new(listener, handler, trusted_group);

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let watchdog_handle = spawn_watchdog(shutdown_rx.clone());

    notify_ready();
    info!("runtimed successfully initialized and ready (max_notify={})", NOTIFY_MAX);

    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;

    let server_handle = tokio::spawn(server.run(shutdown_rx, SHUTDOWN_TIMEOUT));

    tokio::select! {
        _ = sigterm.recv() => info!("Received SIGTERM, initiating shutdown"),
        _ = sigint.recv() => info!("Received SIGINT, initiating shutdown"),
    }

    if shutdown_tx.send(true).is_err() {
        warn!("Shutdown channel closed before send");
    }
    drop(shutdown_tx);

    finish_shutdown(server_handle).await;
    let _ = watchdog_handle.await;
    info!("runtimed shutdown completed");
    Ok(())
}

/// Binds the standalone Unix socket, avoiding the `exists`/`remove_file`/
/// `bind` TOCTOU window. On chmod failure, drops the listener and removes
/// the socket inode so a retry does not hit `EADDRINUSE` with the wrong mode.
fn bind_standalone(socket_path: &std::path::Path) -> Result<UnixListener> {
    if let Some(parent) = socket_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    info!("Listening on Unix domain socket: {}", socket_path.display());
    let std_listener = match StdUnixListener::bind(socket_path) {
        Ok(l) => l,
        Err(e) => {
            error!("Failed to bind socket {}: {}", socket_path.display(), e);
            return Err(e.into());
        }
    };
    if let Err(e) = std_listener.set_nonblocking(true) {
        let _ = fs::remove_file(socket_path);
        return Err(e.into());
    }
    if let Err(e) = fs::set_permissions(socket_path, fs::Permissions::from_mode(0o660)) {
        drop(std_listener);
        let _ = fs::remove_file(socket_path);
        return Err(e.into());
    }
    match UnixListener::from_std(std_listener) {
        Ok(l) => Ok(l),
        Err(e) => {
            let _ = fs::remove_file(socket_path);
            Err(e.into())
        }
    }
}
//! Configuration parser and defaults for runtimed.

use crate::error::RuntimedError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Default configuration file location for runtimed.
pub const DEFAULT_CONFIG_PATH: &str = "/etc/syntrop/runtimed.conf";

/// Default runtime socket path for runtimed Varlink interface.
pub const DEFAULT_SOCKET_PATH: &str = "/run/syntrop/io.syntrop.Runtime1";

/// Default directory for model storage.
pub const DEFAULT_MODELS_PATH: &str = "/var/lib/models";

/// Daemon operational configuration parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimedConfig {
    /// Varlink IPC Unix domain socket path.
    pub socket_path: PathBuf,
    /// Root path for model weights cache.
    pub models_dir: PathBuf,
    /// Default model loaded on startup.
    pub default_model: String,
    /// Maximum context window supported in tokens.
    pub max_context_window: usize,
    /// Maximum concurrent generation requests.
    pub max_concurrent_requests: usize,
}

impl Default for RuntimedConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from(DEFAULT_SOCKET_PATH),
            models_dir: PathBuf::from(DEFAULT_MODELS_PATH),
            default_model: "qwen2.5-coder-7b".to_string(),
            max_context_window: 8192,
            max_concurrent_requests: 4,
        }
    }
}

impl RuntimedConfig {
    /// Loads configuration from a filesystem path, falling back to defaults.
    pub fn load_or_default<P: AsRef<Path>>(path: P) -> Result<Self, RuntimedError> {
        let p = path.as_ref();
        if !p.exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(p)?;
        toml::from_str(&content).map_err(|e| RuntimedError::Config(e.to_string()))
    }
}

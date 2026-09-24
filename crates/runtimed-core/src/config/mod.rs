//! Configuration parser and global defaults.

pub mod runtimed_config;

pub use runtimed_config::{
    RuntimedConfig, DEFAULT_CONFIG_PATH, DEFAULT_MODELS_PATH, DEFAULT_SOCKET_PATH,
};

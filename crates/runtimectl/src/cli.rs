//! CLI argument definitions and command structures for runtimectl.

use clap::{Parser, Subcommand};
use clap_complete::Shell;
use runtimed_core::config::DEFAULT_SOCKET_PATH;
use std::path::PathBuf;

/// CLI client for runtimed headless model execution daemon.
#[derive(Parser, Debug)]
#[command(
    name = "runtimectl",
    version,
    about = "Control and query model inference execution via runtimed",
    long_about = "Execute text completions, vector embeddings, and inspect active neural models."
)]
pub struct Cli {
    /// Path to runtimed Varlink Unix domain socket.
    #[arg(short = 's', long = "socket", default_value = DEFAULT_SOCKET_PATH, global = true)]
    pub socket: PathBuf,

    /// Output results in formatted JSON.
    #[arg(long = "json", global = true)]
    pub json: bool,

    #[command(subcommand)]
    pub command: Commands,
}

/// Available subcommands for runtimectl.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Generate text completions from a prompt.
    Generate {
        /// Input prompt text for model completion.
        prompt: String,

        /// Target model identifier.
        #[arg(short = 'm', long = "model", default_value = "qwen2.5-coder-7b")]
        model: String,

        /// Maximum new tokens to sample.
        #[arg(short = 'n', long = "max-tokens", default_value = "256")]
        max_tokens: usize,

        /// Sampling temperature (0.0 for greedy deterministic decoding).
        #[arg(short = 't', long = "temperature", default_value = "0.0")]
        temperature: f32,
    },

    /// Generate normalized vector embeddings for text.
    Embed {
        /// Input text to compute embedding vector for.
        text: String,

        /// Target model identifier.
        #[arg(short = 'm', long = "model", default_value = "qwen2.5-coder-7b")]
        model: String,
    },

    /// Inspect runtime status, memory footprint, and backend of a model.
    Status {
        /// Target model identifier.
        model: String,
    },

    /// Unload an active model from memory or accelerator device.
    Unload {
        /// Target model identifier.
        model: String,
    },

    /// List all currently active loaded models.
    List,

    /// Inspect daemon vendor information and interface schemas.
    Info,

    /// Generate shell auto-completion script.
    Completions {
        /// Target shell for completion generation.
        #[arg(value_enum)]
        shell: Shell,
    },
}

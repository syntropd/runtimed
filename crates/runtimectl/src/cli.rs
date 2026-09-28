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

        /// Top-k truncation (0 disables).
        #[arg(long = "top-k", default_value = "0")]
        top_k: usize,

        /// Nucleus truncation (1.0 disables).
        #[arg(long = "top-p", default_value = "1.0")]
        top_p: f32,

        /// Sampling seed (0 draws entropy from the clock).
        #[arg(long = "seed", default_value = "0")]
        seed: u64,

        /// Image file (PNG/JPEG) for multimodal generation.
        #[arg(long = "image")]
        image: Option<String>,
    },

    /// Attach a vision projector to a loaded model.
    AttachVision {
        /// Target model identifier.
        model: String,

        /// Projector file (absolute path or models dir entry).
        mmproj: String,
    },

    /// Fuse a LoRA adapter into a loaded model.
    AttachLora {
        /// Target model identifier.
        model: String,

        /// Adapter file (absolute path or models dir entry).
        lora: String,
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

    /// Report daemon load (free slots, resident bytes) for fleet routing.
    Load,

    /// Inspect daemon vendor information and interface schemas.
    Info,

    /// Generate shell auto-completion script.
    Completions {
        /// Target shell for completion generation.
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_list_with_defaults() {
        let cli = Cli::try_parse_from(["runtimectl", "list"]).unwrap();
        assert!(matches!(cli.command, Commands::List));
        assert!(!cli.json);
    }

    #[test]
    fn parses_generate_defaults() {
        let cli = Cli::try_parse_from(["runtimectl", "generate", "hello"]).unwrap();
        match cli.command {
            Commands::Generate {
                prompt,
                model,
                max_tokens,
                temperature,
                top_k,
                top_p,
                seed,
                image,
            } => {
                assert_eq!(prompt, "hello");
                assert_eq!(model, "qwen2.5-coder-7b");
                assert_eq!(max_tokens, 256);
                assert_eq!(temperature, 0.0);
                assert_eq!(top_k, 0);
                assert_eq!(top_p, 1.0);
                assert_eq!(seed, 0);
                assert_eq!(image, None);
            }
            other => panic!("expected Generate, got {other:?}"),
        }
    }

    #[test]
    fn parses_global_flags_and_completions() {
        let cli = Cli::try_parse_from([
            "runtimectl",
            "--json",
            "-s",
            "/tmp/x.sock",
            "completions",
            "bash",
        ])
        .unwrap();
        assert!(cli.json);
        assert_eq!(cli.socket, PathBuf::from("/tmp/x.sock"));
        assert!(matches!(cli.command, Commands::Completions { shell: Shell::Bash }));
    }
}

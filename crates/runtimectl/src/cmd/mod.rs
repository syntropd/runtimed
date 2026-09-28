//! Subcommand handlers for runtimectl.

use clap::Command;
use clap_complete::{generate, Shell};
use std::io;

pub mod infer;
pub mod model;

pub use infer::attach_cmd::exec_attach_vision;
pub use infer::embed_cmd::exec_embed;
pub use infer::generate_cmd::exec_generate;
pub use infer::lora_cmd::exec_attach_lora;
pub use model::info_cmd::exec_info;
pub use model::list_cmd::exec_list;
pub use model::load_cmd::exec_load;
pub use model::status_cmd::exec_status;
pub use model::unload_cmd::exec_unload;

/// Human parameter count: 630000000 -> "630M", 4600000000 -> "4.6B".
pub fn human_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
            .replace(".0B", "B")
    } else if n >= 1_000_000 {
        format!("{:.0}M", n as f64 / 1e6)
    } else {
        n.to_string()
    }
}

/// Generates shell completion script to stdout.
pub fn exec_completions(cmd: &mut Command, shell: Shell) {
    let bin_name = cmd.get_name().to_string();
    generate(shell, cmd, bin_name, &mut io::stdout());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_count_formats_scales() {
        assert_eq!(human_count(999), "999");
        assert_eq!(human_count(1_000_000), "1M");
        assert_eq!(human_count(630_000_000), "630M");
        assert_eq!(human_count(1_000_000_000), "1B");
        assert_eq!(human_count(4_600_000_000), "4.6B");
    }

    #[test]
    fn completions_render_without_panic() {
        use crate::cli::Cli;
        use clap::CommandFactory;
        let mut cmd = Cli::command();
        exec_completions(&mut cmd, Shell::Bash);
    }
}

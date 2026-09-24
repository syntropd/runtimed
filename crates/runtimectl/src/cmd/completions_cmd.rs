//! Handler for shell completions.

use clap::Command;
use clap_complete::{generate, Shell};
use std::io;

/// Generates shell completion script to stdout.
pub fn exec_completions(cmd: &mut Command, shell: Shell) {
    let bin_name = cmd.get_name().to_string();
    generate(shell, cmd, bin_name, &mut io::stdout());
}

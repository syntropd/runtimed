//! Main binary entry point for runtimectl CLI.

use anyhow::Result;
use clap::{CommandFactory, Parser};
use runtimectl::cli::{Cli, Commands};
use runtimectl::client::RuntimedClient;
use runtimectl::cmd::*;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = RuntimedClient::new(&cli.socket);

    match cli.command {
        Commands::Generate {
            prompt,
            model,
            max_tokens,
            temperature,
        } => {
            exec_generate(&client, &model, &prompt, max_tokens, temperature, cli.json).await?;
        }
        Commands::Embed { text, model } => {
            exec_embed(&client, &model, &text, cli.json).await?;
        }
        Commands::Status { model } => {
            exec_status(&client, &model, cli.json).await?;
        }
        Commands::Unload { model } => {
            exec_unload(&client, &model, cli.json).await?;
        }
        Commands::List => {
            exec_list(&client, cli.json).await?;
        }
        Commands::Info => {
            exec_info(&client, cli.json).await?;
        }
        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            exec_completions(&mut cmd, shell);
        }
    }

    Ok(())
}

//! Subcommand handlers for runtimectl.

pub mod attach_cmd;
pub mod completions_cmd;
pub mod embed_cmd;
pub mod generate_cmd;
pub mod info_cmd;
pub mod load_cmd;
pub mod lora_cmd;
pub mod list_cmd;
pub mod status_cmd;
pub mod unload_cmd;

pub use attach_cmd::exec_attach_vision;
pub use completions_cmd::exec_completions;
pub use embed_cmd::exec_embed;
pub use generate_cmd::exec_generate;
pub use info_cmd::exec_info;
pub use load_cmd::exec_load;
pub use lora_cmd::exec_attach_lora;
pub use list_cmd::exec_list;
pub use status_cmd::exec_status;
pub use unload_cmd::exec_unload;

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

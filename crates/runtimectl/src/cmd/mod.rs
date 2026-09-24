//! Subcommand handlers for runtimectl.

pub mod completions_cmd;
pub mod embed_cmd;
pub mod generate_cmd;
pub mod info_cmd;
pub mod list_cmd;
pub mod status_cmd;
pub mod unload_cmd;

pub use completions_cmd::exec_completions;
pub use embed_cmd::exec_embed;
pub use generate_cmd::exec_generate;
pub use info_cmd::exec_info;
pub use list_cmd::exec_list;
pub use status_cmd::exec_status;
pub use unload_cmd::exec_unload;

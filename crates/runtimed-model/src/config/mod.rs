//! Arch configs parsed from GGUF metadata or Hugging Face config.json.

mod parse_gguf;
mod parse_hf;
mod types;

pub use parse_hf::{parse_hf_config, parse_hf_file};
pub use types::{Activation, Arch, ArchConfig, LayerConfig};

use crate::error::Result;
use runtimed_gguf::GgufFile;
use std::path::Path;

impl ArchConfig {
    /// Parse architecture configuration from GGUF metadata.
    pub fn parse(file: &GgufFile) -> Result<Self> {
        parse_gguf::parse_gguf_config(file)
    }

    /// Parse architecture configuration from a Hugging Face config.json string.
    pub fn parse_hf(json_str: &str) -> Result<Self> {
        parse_hf::parse_hf_config(json_str)
    }

    /// Parse architecture configuration from a Hugging Face config.json file path.
    pub fn from_hf_file(path: &Path) -> Result<Self> {
        parse_hf::parse_hf_file(path)
    }
}

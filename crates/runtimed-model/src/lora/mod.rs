//! LoRA adapters (GGUF and Safetensors), fused at load.

mod fuse;
mod load_gguf;
mod load_safetensors;
mod types;

pub use types::{LoraAdapter, LoraPair};

use crate::error::Result;
use candle_core::Device;
use std::path::Path;

impl LoraAdapter {
    /// Load + validate an adapter file or directory (GGUF or Safetensors).
    /// `arch` must match the base model (e.g. `"gemma4"`, `"qwen2"`).
    pub fn load(path: &Path, dev: &Device, arch: &str) -> Result<Self> {
        let is_safetensors = path.is_dir()
            || path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("safetensors"))
                .unwrap_or(false);

        if is_safetensors {
            match load_safetensors::load_safetensors_adapter(path, dev, arch) {
                Ok(adapter) => return Ok(adapter),
                Err(e) if !path.is_dir() => {
                    // Fall back to GGUF if path was a file and safetensors failed
                    if let Ok(adapter) = load_gguf::load_gguf_adapter(path, dev, arch) {
                        return Ok(adapter);
                    }
                    return Err(e);
                }
                Err(e) => return Err(e),
            }
        }

        load_gguf::load_gguf_adapter(path, dev, arch)
    }
}

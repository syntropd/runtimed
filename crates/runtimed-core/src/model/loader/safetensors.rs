//! Safetensors model loading for ModelManager.

use crate::error::RuntimedError;
use crate::model::meta::LoadedModel;
use crate::model::tokenizer::EngineTokenizer;
use candle_core::Device;
use memmap2::MmapOptions;
use runtimed_model::{ArchConfig, Session};
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub fn load_safetensors_entry(
    name: &str,
    path: &Path,
    file_opt: Option<&File>,
    device: &Device,
) -> Result<(Session, LoadedModel, EngineTokenizer, Vec<u32>, bool), RuntimedError> {
    let cfg = load_companion_config(path)?;
    let arch_tag = match cfg.arch {
        runtimed_model::Arch::Qwen2 => "qwen2",
        runtimed_model::Arch::Gemma4 => "gemma4",
        runtimed_model::Arch::Granite => "granite",
        runtimed_model::Arch::Phi3 => "phi3",
    };
    let cfg_arc = Arc::new(cfg);

    let session = match file_opt {
        Some(f) => Session::load_safetensors_from_file(f, cfg_arc.clone(), device)
            .map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?,
        None => {
            let weights = Arc::new(
                runtimed_model::weights::Weights::load_safetensors(path, device)
                    .map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?,
            );
            Session::new(cfg_arc.clone(), weights)
                .map_err(|e| RuntimedError::GenerationFailed(e.to_string()))?
        }
    };

    let params = count_safetensors_params(path, file_opt)?;
    let (tokenizer, eos, add_special) = EngineTokenizer::load_for_safetensors(cfg_arc.arch, path)?;

    let context = cfg_arc.sliding_window.unwrap_or(8192);
    let meta = LoadedModel {
        name: name.to_string(),
        architecture: arch_tag.to_string(),
        parameter_count: params,
        memory_bytes: session.resident_bytes() + (64 << 20),
        context_window: context,
        compute_backend: if matches!(session.device(), Device::Cuda(_)) { "cuda" } else { "cpu" }.to_string(),
    };

    Ok((session, meta, tokenizer, eos, add_special))
}

fn load_companion_config(path: &Path) -> Result<ArchConfig, RuntimedError> {
    let sibling = path.with_extension("config.json");
    if sibling.exists() {
        return ArchConfig::from_hf_file(&sibling)
            .map_err(|e| RuntimedError::GenerationFailed(e.to_string()));
    }
    if let Some(parent) = path.parent() {
        let parent_cfg = parent.join("config.json");
        if parent_cfg.exists() {
            return ArchConfig::from_hf_file(&parent_cfg)
                .map_err(|e| RuntimedError::GenerationFailed(e.to_string()));
        }
    }
    // Infer default Qwen2 architecture config if config.json is not present
    let json = r#"{
        "model_type": "qwen2",
        "num_hidden_layers": 24,
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "intermediate_size": 4864,
        "vocab_size": 151936
    }"#;
    ArchConfig::parse_hf(json).map_err(|e| RuntimedError::GenerationFailed(e.to_string()))
}

fn count_safetensors_params(path: &Path, file_opt: Option<&File>) -> Result<u64, RuntimedError> {
    let mmap = match file_opt {
        Some(f) => unsafe {
            MmapOptions::new()
                .map(f)
                .map_err(|e| RuntimedError::GenerationFailed(format!("mmap: {e}")))?
        },
        None => {
            let f = File::open(path)
                .map_err(|e| RuntimedError::GenerationFailed(format!("open {}: {e}", path.display())))?;
            unsafe {
                MmapOptions::new()
                    .map(&f)
                    .map_err(|e| RuntimedError::GenerationFailed(format!("mmap: {e}")))?
            }
        }
    };
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| RuntimedError::GenerationFailed(format!("safetensors parse: {e}")))?;
    let mut total: u64 = 0;
    for (_, view) in st.tensors() {
        let n: usize = view.shape().iter().product();
        total += n as u64;
    }
    Ok(total)
}

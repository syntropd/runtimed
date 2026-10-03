//! Engine tokenizer abstraction: file-backed (Qwen2) or GGUF-embedded BPE.

use crate::config::Arch;
use crate::error::{ModelError, Result};
use runtimed_gguf::{GgufBpe, GgufFile, MetaValue, Tokenizer};
use std::path::{Path, PathBuf};

/// Tokenizer behind one interface: file-backed or GGUF-embedded BPE.
#[allow(clippy::large_enum_variant)]
pub enum EngineTokenizer {
    File(Tokenizer),
    Bpe(GgufBpe),
}

impl EngineTokenizer {
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        match self {
            Self::File(t) => t
                .encode(text, add_special)
                .map_err(|e| ModelError::Config(format!("tokenize: {e}"))),
            Self::Bpe(t) => Ok(t.encode(text, add_special)),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            Self::File(t) => t
                .decode(ids, true)
                .map_err(|e| ModelError::Config(format!("detokenize: {e}"))),
            Self::Bpe(t) => Ok(t.decode(ids, true)),
        }
    }

    pub fn as_bpe(&self) -> Option<&GgufBpe> {
        match self {
            Self::Bpe(t) => Some(t),
            Self::File(_) => None,
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Self::File(t) => t.vocab_size(),
            Self::Bpe(t) => t.vocab_size(),
        }
    }

    pub fn token_bytes(&self, id: u32) -> Option<Vec<u8>> {
        match self {
            Self::File(t) => t.decode(&[id], false).ok().map(|s| s.into_bytes()),
            Self::Bpe(t) => {
                let text = t.token_text(id)?;
                if text.starts_with("<0x") && text.ends_with('>') && text.len() == 6 {
                    if let Ok(b) = u8::from_str_radix(&text[3..5], 16) {
                        return Some(vec![b]);
                    }
                }
                Some(text.replace('\u{2581}', " ").into_bytes())
            }
        }
    }

    pub fn load_for_arch(
        arch: Arch,
        file: &GgufFile,
        weights: &Path,
    ) -> Result<(Self, Vec<u32>, bool)> {
        let resolve_tok_path = |p: &Path| -> PathBuf {
            let direct = p.with_extension("tokenizer.json");
            if direct.exists() {
                direct
            } else if let Ok(real) = p.canonicalize() {
                real.with_extension("tokenizer.json")
            } else {
                direct
            }
        };

        match arch {
            Arch::Gemma4 => {
                let bpe = GgufBpe::from_gguf(file)
                    .map_err(|e| ModelError::Config(format!("bpe: {e}")))?;
                let eos = bpe.eos_id().into_iter().collect();
                let add_special = bpe.wants_bos();
                Ok((Self::Bpe(bpe), eos, add_special))
            }
            Arch::Granite | Arch::Phi3 => {
                let tok_path = resolve_tok_path(weights);
                if tok_path.exists() {
                    let tok = Tokenizer::from_file(&tok_path).map_err(|e| {
                        ModelError::Config(format!("failed to load {}: {e}", tok_path.display()))
                    })?;
                    let eos = file
                        .metadata
                        .get("tokenizer.ggml.eos_token_id")
                        .and_then(|v| match v {
                            MetaValue::U32(id) => Some(*id),
                            MetaValue::I32(id) => Some(*id as u32),
                            _ => None,
                        })
                        .into_iter()
                        .collect();
                    let add_special = matches!(
                        file.metadata.get("tokenizer.ggml.add_bos_token"),
                        Some(MetaValue::Bool(true))
                    );
                    Ok((Self::File(tok), eos, add_special))
                } else {
                    let bpe = GgufBpe::from_gguf(file)
                        .map_err(|e| ModelError::Config(format!("bpe: {e}")))?;
                    let eos = bpe.eos_id().into_iter().collect();
                    let add_special = bpe.wants_bos();
                    Ok((Self::Bpe(bpe), eos, add_special))
                }
            }
            Arch::Qwen2 => {
                let tok_path = resolve_tok_path(weights);
                let tok = Tokenizer::from_file(&tok_path).map_err(|_| {
                    ModelError::Config(format!(
                        "qwen2 needs a sibling tokenizer.json next to {}",
                        weights.display()
                    ))
                })?;
                let eos = file
                    .metadata
                    .get("tokenizer.ggml.eos_token_id")
                    .and_then(|v| match v {
                        MetaValue::U32(id) => Some(*id),
                        MetaValue::I32(id) => Some(*id as u32),
                        _ => None,
                    })
                    .into_iter()
                    .collect();
                let add_special = matches!(
                    file.metadata.get("tokenizer.ggml.add_bos_token"),
                    Some(MetaValue::Bool(true))
                );
                Ok((Self::File(tok), eos, add_special))
            }
        }
    }

    pub fn load_for_safetensors(
        arch: Arch,
        weights: &Path,
    ) -> Result<(Self, Vec<u32>, bool)> {
        let tok_path = if weights.with_extension("tokenizer.json").exists() {
            weights.with_extension("tokenizer.json")
        } else if let Some(parent) = weights.parent() {
            let direct = parent.join("tokenizer.json");
            if direct.exists() {
                direct
            } else {
                return Err(ModelError::Config(format!(
                    "safetensors model needs a sibling tokenizer.json next to {}",
                    weights.display()
                )));
            }
        } else {
            return Err(ModelError::Config(format!(
                "safetensors model needs a sibling tokenizer.json next to {}",
                weights.display()
            )));
        };

        let tok = Tokenizer::from_file(&tok_path).map_err(|e| {
            ModelError::Config(format!("failed to load {}: {e}", tok_path.display()))
        })?;

        let (eos, add_special) = match arch {
            Arch::Gemma4 => (vec![1], true),
            Arch::Qwen2 => (vec![151643, 151645], false),
            Arch::Granite => (vec![0], false),
            Arch::Phi3 => (vec![32000, 32007], false),
        };

        Ok((Self::File(tok), eos, add_special))
    }
}

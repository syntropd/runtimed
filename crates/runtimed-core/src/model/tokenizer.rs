//! Tokenizer behind one interface: file-backed (Qwen2) or GGUF-embedded BPE.

use crate::error::RuntimedError;
use runtimed_gguf::{GgufBpe, GgufFile, MetaValue, Tokenizer};
use runtimed_model::Arch;
use std::path::Path;

/// Tokenizer behind one interface: file-backed (Qwen2) or GGUF-embedded BPE.
pub enum EngineTokenizer {
    File(Tokenizer),
    Bpe(GgufBpe),
}

impl EngineTokenizer {
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>, RuntimedError> {
        match self {
            Self::File(t) => t
                .encode(text, add_special)
                .map_err(|e| RuntimedError::GenerationFailed(format!("tokenize: {e}"))),
            Self::Bpe(t) => Ok(t.encode(text, add_special)),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, RuntimedError> {
        match self {
            Self::File(t) => t
                .decode(ids, true)
                .map_err(|e| RuntimedError::GenerationFailed(format!("detokenize: {e}"))),
            Self::Bpe(t) => Ok(t.decode(ids, true)),
        }
    }

    /// Borrow the GGUF BPE (chat templates and vision need raw BPE access).
    pub fn as_bpe(&self) -> Option<&GgufBpe> {
        match self {
            Self::Bpe(t) => Some(t),
            Self::File(_) => None,
        }
    }

    /// Build the arch-correct tokenizer plus stop ids and BOS flag.
    /// Gemma reads BPE from the GGUF; Qwen2 needs a sibling tokenizer.json.
    pub fn load_for_arch(
        arch: Arch,
        file: &GgufFile,
        weights: &Path,
    ) -> Result<(Self, Vec<u32>, bool), RuntimedError> {
        match arch {
            Arch::Gemma4 => {
                let bpe = GgufBpe::from_gguf(file)
                    .map_err(|e| RuntimedError::GenerationFailed(format!("bpe: {e}")))?;
                let eos = bpe.eos_id().into_iter().collect();
                let add_special = bpe.wants_bos();
                Ok((Self::Bpe(bpe), eos, add_special))
            }
            Arch::Qwen2 => {
                let tok_path = weights.with_extension("tokenizer.json");
                let tok = Tokenizer::from_file(&tok_path).map_err(|_| {
                    RuntimedError::GenerationFailed(format!(
                        "qwen2 needs a sibling tokenizer.json next to {}",
                        weights.display()
                    ))
                })?;
                let eos = super::meta::meta_u32(file, "tokenizer.ggml.eos_token_id")
                    .into_iter()
                    .collect();
                let add_special =
                    matches!(file.metadata.get("tokenizer.ggml.add_bos_token"), Some(MetaValue::Bool(true)));
                Ok((Self::File(tok), eos, add_special))
            }
        }
    }
}

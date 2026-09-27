//! Tokenizer behind one interface: file-backed (Qwen2) or GGUF-embedded BPE.

use crate::error::RuntimedError;
use runtimed_gguf::{GgufBpe, Tokenizer};

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
}

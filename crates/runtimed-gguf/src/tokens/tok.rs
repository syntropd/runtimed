//! Exact tokenization from a `tokenizer.json` file.
//!
//! No reconstruction guesses: the file that shipped with the model weights
//! is the tokenizer. Files are hash-pinned in the weight registry like GGUFs.

use crate::error::{GgufError, Result};
use std::path::Path;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| GgufError::Tokenizer(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| GgufError::Tokenizer(e.to_string()))
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| GgufError::Tokenizer(e.to_string()))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_file_missing_errors() {
        let err =
            Tokenizer::from_file(std::path::Path::new("/nonexistent-dir-xyz/tok.json")).err().expect("must fail");
        assert!(matches!(err, GgufError::Tokenizer(_)), "{err:?}");
    }
}

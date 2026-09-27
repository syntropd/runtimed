//! Owned SPM-style BPE tokenizer built from GGUF `tokenizer.ggml.*` arrays.
//!
//! This mirrors the Gemma4 path of the reference implementation exactly:
//! spaces become U+2581, text splits on newline runs only, bigram merges
//! apply by GGUF rank order, unknown bytes fall back to `<0xXX>` tokens.
//! No regex engine, no external tokenizer dependency: newline runs and
//! merge scans are small enough to own outright.

use crate::error::{GgufError, Result};
use crate::header::{GgufFile, MetaValue};
use std::collections::HashMap;

/// GGUF token types (`tokenizer.ggml.token_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokType {
    fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Unknown,
        }
    }
}

fn str_arr(file: &GgufFile, key: &str) -> Result<Vec<String>> {
    match file.metadata.get(key) {
        Some(MetaValue::Arr(items)) => items
            .iter()
            .map(|v| match v {
                MetaValue::Str(s) => Ok(s.clone()),
                other => Err(GgufError::Tokenizer(format!("{key} item {other:?}"))),
            })
            .collect(),
        other => Err(GgufError::Tokenizer(format!("{key} missing, got {other:?}"))),
    }
}

fn i32_arr(file: &GgufFile, key: &str) -> Result<Vec<i32>> {
    match file.metadata.get(key) {
        Some(MetaValue::Arr(items)) => items
            .iter()
            .map(|v| match v {
                MetaValue::I32(x) => Ok(*x),
                other => Err(GgufError::Tokenizer(format!("{key} item {other:?}"))),
            })
            .collect(),
        other => Err(GgufError::Tokenizer(format!("{key} missing, got {other:?}"))),
    }
}

fn meta_u32_opt(file: &GgufFile, key: &str) -> Option<u32> {
    match file.metadata.get(key) {
        Some(MetaValue::U32(v)) => Some(*v),
        _ => None,
    }
}

fn meta_bool(file: &GgufFile, key: &str) -> bool {
    matches!(file.metadata.get(key), Some(MetaValue::Bool(true)))
}

/// Split a merge line at the first space at byte index >= 1.
fn split_merge(line: &str) -> (String, String) {
    match line.as_bytes().iter().skip(1).position(|&b| b == b' ') {
        Some(rel) => {
            let pos = rel + 1;
            (line[..pos].to_string(), line[pos + 1..].to_string())
        }
        None => (String::new(), String::new()),
    }
}

pub struct GgufBpe {
    // `pub(crate)`: the encode/decode algorithms live in `bpe_encode.rs`
    // and share these tables. Public callers use the accessor methods.
    pub(crate) tokens: Vec<String>,
    pub(crate) types: Vec<TokType>,
    pub(crate) id_of: HashMap<String, u32>,
    pub(crate) ranks: HashMap<(String, String), usize>,
    pub(crate) bos: Option<u32>,
    pub(crate) eos: Option<u32>,
    pub(crate) pad: u32,
    pub(crate) add_bos: bool,
    pub(crate) specials: std::collections::HashSet<u32>,
    /// Special token texts, longest first, for special-aware encoding.
    pub(crate) special_texts: Vec<(String, u32)>,
}

impl GgufBpe {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let tokens = str_arr(file, "tokenizer.ggml.tokens")?;
        let types_raw = i32_arr(file, "tokenizer.ggml.token_type")?;
        if types_raw.len() != tokens.len() {
            return Err(GgufError::Tokenizer(format!(
                "tokens/types length mismatch {} vs {}",
                tokens.len(),
                types_raw.len()
            )));
        }
        let merges = str_arr(file, "tokenizer.ggml.merges")?;
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, line) in merges.iter().enumerate() {
            ranks.insert(split_merge(line), i);
        }
        let id_of: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let bos = meta_u32_opt(file, "tokenizer.ggml.bos_token_id");
        let eos = meta_u32_opt(file, "tokenizer.ggml.eos_token_id");
        let pad = meta_u32_opt(file, "tokenizer.ggml.padding_token_id").unwrap_or(0);
        let mut specials = std::collections::HashSet::new();
        for key in [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.mask_token_id",
        ] {
            if let Some(id) = meta_u32_opt(file, key) {
                specials.insert(id);
            }
        }
        let types: Vec<TokType> = types_raw.iter().map(|&t| TokType::from_i32(t)).collect();
        // Special-parseable: control, user-defined, unknown (never normal
        // pieces, byte fallbacks, or unused slots). Matches the reference's
        // `cache_special_tokens` filter.
        let is_special = |t: TokType| matches!(t, TokType::Control | TokType::UserDefined | TokType::Unknown);
        let mut special_texts: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| specials.contains(&(*i as u32)) || is_special(types[*i]))
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        special_texts.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Ok(Self {
            types,
            tokens,
            id_of,
            ranks,
            bos,
            eos,
            pad,
            add_bos: meta_bool(file, "tokenizer.ggml.add_bos_token"),
            specials,
            special_texts,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos
    }

    /// Whether prompts take a leading BOS (`tokenizer.ggml.add_bos_token`).
    pub fn wants_bos(&self) -> bool {
        self.add_bos
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    pub fn pad_id(&self) -> u32 {
        self.pad
    }

    /// Exact id for a vocabulary piece (`▁` for spaces, as stored).
    /// Returns `None` for byte-fallback and unknown pieces.
    pub fn piece_id(&self, piece: &str) -> Option<u32> {
        self.id_of.get(piece).copied()
    }

    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe_encode::parse_byte_token;

    #[test]
    fn merge_split_skips_leading_space() {
        assert_eq!(split_merge("a b"), ("a".into(), "b".into()));
        assert_eq!(split_merge("\u{2581} a"), ("\u{2581}".into(), "a".into()));
        assert_eq!(split_merge("nospace"), ("".into(), "".into()));
    }

    #[test]
    fn byte_tokens_parse_both_cases() {
        assert_eq!(parse_byte_token("<0xAB>"), Some(0xAB));
        assert_eq!(parse_byte_token("<0xab>"), Some(0xAB));
        assert_eq!(parse_byte_token("<0xG1>"), None);
        assert_eq!(parse_byte_token("ab"), None);
    }
}
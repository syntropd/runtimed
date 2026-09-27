//! Weight registry: every runnable file is pinned by SHA256.
//!
//! Registry format (TOML):
//!
//! ```toml
//! [[weight]]
//! name = "qwen2.5-0.5b-q8"
//! url = "https://…/qwen2.5-0.5b-instruct-q8_0.gguf"
//! sha256 = "…"
//! ```

use crate::error::{GgufError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightEntry {
    pub name: String,
    pub url: String,
    pub sha256: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct RegistryFile {
    #[serde(default)]
    weight: Vec<WeightEntry>,
}

#[derive(Debug)]
pub struct Registry {
    entries: HashMap<String, WeightEntry>,
}

impl Registry {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(GgufError::Io)?;
        let file: RegistryFile =
            toml::from_str(&text).map_err(|e| GgufError::Registry(e.to_string()))?;
        let mut entries = HashMap::new();
        for w in file.weight {
            entries.insert(w.name.clone(), w);
        }
        Ok(Self { entries })
    }

    pub fn get(&self, name: &str) -> Option<&WeightEntry> {
        self.entries.get(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Verify `path` against the pinned hash for `name`.
    pub fn verify(&self, name: &str, path: &Path) -> Result<()> {
        let entry = self
            .get(name)
            .ok_or_else(|| GgufError::Registry(format!("unknown weight '{name}'")))?;
        verify_file(path, &entry.sha256)
    }
}

/// SHA256 hex of a file, streamed.
pub fn sha256_of(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

pub fn verify_file(path: &Path, expected_hex: &str) -> Result<()> {
    let got = sha256_of(path)?;
    if got.eq_ignore_ascii_case(expected_hex) {
        Ok(())
    } else {
        Err(GgufError::Hash(
            path.display().to_string(),
            expected_hex.to_string(),
            got,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("runtimed-registry-test-{name}"));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn sha256_matches_standard_vector() {
        let p = tmp("vec", b"abc");
        // FIPS 180-4 ("abc") — pins the hasher, not just the comparison.
        assert_eq!(
            sha256_of(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn verify_accepts_match_rejects_mismatch_and_unknown() {
        let p = tmp("w", b"weight-bytes");
        let good = sha256_of(&p).unwrap();
        let reg = Registry {
            entries: HashMap::from([(
                "w".to_string(),
                WeightEntry {
                    name: "w".to_string(),
                    url: "https://example.invalid/w.gguf".to_string(),
                    sha256: good,
                },
            )]),
        };
        assert!(reg.verify("w", &p).is_ok());
        assert!(reg.verify("ghost", &p).is_err());
        let bad = Registry {
            entries: HashMap::from([(
                "w".to_string(),
                WeightEntry {
                    name: "w".to_string(),
                    url: "https://example.invalid/w.gguf".to_string(),
                    sha256: "0".repeat(64),
                },
            )]),
        };
        let err = bad.verify("w", &p).unwrap_err().to_string();
        assert!(err.contains("hash mismatch"), "{err}");
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn load_distinguishes_missing_from_broken() {
        let missing = std::env::temp_dir().join("runtimed-registry-test-nope.toml");
        let _ = std::fs::remove_file(&missing);
        assert!(Registry::load(&missing).unwrap_err().is_missing());
        let p = tmp("broken", b"[[weight]\nname = ");
        let err = Registry::load(&p).unwrap_err();
        assert!(!err.is_missing(), "broken TOML must not look missing");
        assert!(err.to_string().contains("registry"), "{err}");
        std::fs::remove_file(&p).unwrap();
    }
}

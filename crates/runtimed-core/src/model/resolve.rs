//! Path resolution for weights and the weight registry (R2).
//!
//! Two layouts are accepted, in order:
//! 1. Flat: `<models_dir>/<name>[.gguf]` (historic standalone tree).
//! 2. Fleet: `<models_dir>/gguf/<name>[.gguf]` (shared tree where
//!    `modeld` owns sibling dirs like `cas/`, `pinned/`, `tags/`).
//! Absolute paths pass through untouched. The registry is read from
//! `<models_dir>/registry.toml`, falling back to the `gguf/` subdir
//! when the root file is absent.

use runtimed_gguf::Registry;
use std::path::{Path, PathBuf};

/// Resolve a model/adapter/projector name to a GGUF path: absolute
/// paths pass through, bare names resolve under the models directory
/// (`.gguf` implied), then under its `gguf/` subdir.
pub(crate) fn resolve_path(models_dir: &Path, name: &str) -> Option<PathBuf> {
    let literal = PathBuf::from(name);
    if literal.is_absolute() && literal.exists() {
        return Some(literal);
    }
    let fleet_safetensors = models_dir.join("safetensors");
    let fleet_gguf = models_dir.join("gguf");
    for root in [models_dir, fleet_safetensors.as_path(), fleet_gguf.as_path()] {
        let direct = root.join(name);
        if direct.exists() {
            return Some(direct);
        }
        let with_st = root.join(format!("{name}.safetensors"));
        if with_st.exists() {
            return Some(with_st);
        }
        let with_ext = root.join(format!("{name}.gguf"));
        if with_ext.exists() {
            return Some(with_ext);
        }
    }
    None
}

/// Load the R2 weight registry: root `registry.toml` first, the
/// `gguf/` subdir copy when the root file is absent. A present but
/// broken root file stays an error (fail-open, reported); it never
/// silently falls through to the subdir copy.
pub(crate) fn load_registry(models_dir: &Path) -> (Option<Registry>, Option<String>) {
    let root = models_dir.join("registry.toml");
    let path = if root.exists() {
        root
    } else {
        models_dir.join("gguf/registry.toml")
    };
    match Registry::load(&path) {
        Ok(r) => (Some(r), None),
        Err(e) if e.is_missing() => (None, None),
        Err(e) => (None, Some(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, b"stub").unwrap();
    }

    fn registry_toml(name: &str) -> String {
        format!(
            "[[weight]]\nname = \"{name}\"\nurl = \"https://example.invalid/x.gguf\"\n\
             sha256 = \"0000000000000000000000000000000000000000000000000000000000000000\"\n"
        )
    }

    #[test]
    fn flat_layout_resolves_bare_name() {
        let tmp = TempDir::new().unwrap();
        touch(&tmp.path().join("m.gguf"));
        assert_eq!(
            resolve_path(tmp.path(), "m"),
            Some(tmp.path().join("m.gguf"))
        );
    }

    #[test]
    fn fleet_layout_resolves_bare_name() {
        let tmp = TempDir::new().unwrap();
        touch(&tmp.path().join("gguf/m.gguf"));
        assert_eq!(
            resolve_path(tmp.path(), "m"),
            Some(tmp.path().join("gguf/m.gguf"))
        );
    }

    #[test]
    fn flat_layout_wins_over_fleet() {
        let tmp = TempDir::new().unwrap();
        touch(&tmp.path().join("m.gguf"));
        touch(&tmp.path().join("gguf/m.gguf"));
        assert_eq!(
            resolve_path(tmp.path(), "m"),
            Some(tmp.path().join("m.gguf"))
        );
    }

    #[test]
    fn absolute_paths_pass_through() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("elsewhere.bin");
        touch(&p);
        assert_eq!(resolve_path(tmp.path(), p.to_str().unwrap()), Some(p));
        assert_eq!(resolve_path(tmp.path(), "/does/not/exist.gguf"), None);
    }

    #[test]
    fn missing_name_resolves_to_none() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(resolve_path(tmp.path(), "nope"), None);
    }

    #[test]
    fn registry_prefers_root_then_fleet_then_absent() {
        let tmp = TempDir::new().unwrap();
        let (none, err) = load_registry(tmp.path());
        assert!(none.is_none() && err.is_none());

        touch(&tmp.path().join("gguf/registry.toml"));
        fs::write(
            tmp.path().join("gguf/registry.toml"),
            registry_toml("fleet"),
        )
        .unwrap();
        let (reg, err) = load_registry(tmp.path());
        assert!(err.is_none());
        assert!(reg.unwrap().get("fleet").is_some());

        fs::write(tmp.path().join("registry.toml"), registry_toml("root")).unwrap();
        let (reg, err) = load_registry(tmp.path());
        assert!(err.is_none());
        let reg = reg.unwrap();
        assert!(reg.get("root").is_some() && reg.get("fleet").is_none());
    }

    #[test]
    fn broken_root_registry_is_an_error_not_a_fallthrough() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("registry.toml"), "[[weight\nbroken!\n").unwrap();
        touch(&tmp.path().join("gguf/registry.toml"));
        fs::write(
            tmp.path().join("gguf/registry.toml"),
            registry_toml("fleet"),
        )
        .unwrap();
        let (reg, err) = load_registry(tmp.path());
        assert!(reg.is_none());
        assert!(!err.unwrap().is_empty());
    }

    #[test]
    fn safetensors_layout_resolves_bare_name() {
        let tmp = TempDir::new().unwrap();
        touch(&tmp.path().join("safetensors/qwen.safetensors"));
        assert_eq!(
            resolve_path(tmp.path(), "qwen"),
            Some(tmp.path().join("safetensors/qwen.safetensors"))
        );
        touch(&tmp.path().join("flat.safetensors"));
        assert_eq!(
            resolve_path(tmp.path(), "flat"),
            Some(tmp.path().join("flat.safetensors"))
        );
    }
}

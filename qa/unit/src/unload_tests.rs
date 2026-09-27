//! Unit QA tests for `ModelManager::unload_model` return-value contract.

#[cfg(test)]
mod tests {
    use runtimed_core::error::RuntimedError;
    use runtimed_core::model::ModelManager;
    use tempfile::tempdir;

    /// Unloading an unknown model must return `RuntimedError::ModelNotFound`.
    #[test]
    fn test_unload_unknown_model_returns_model_not_found() {
        let tmp = tempdir().unwrap();
        let manager = ModelManager::new(tmp.path());

        match manager.unload_model("never-loaded") {
            Err(RuntimedError::ModelNotFound(name)) => assert_eq!(name, "never-loaded"),
            Ok(freed) => panic!("expected ModelNotFound, got Ok({})", freed),
            Err(other) => panic!("expected ModelNotFound, got Err({:?})", other),
        }
    }

    /// Gated helper: manager rooted at the real tiny model's directory.
    fn gated() -> Option<(ModelManager, String)> {
        let gguf = std::path::PathBuf::from(std::env::var("SYNTROP_TEST_GGUF").ok()?);
        if !gguf.exists() {
            eprintln!("skip: SYNTROP_TEST_GGUF not set or missing");
            return None;
        }
        let dir = gguf.parent().unwrap().to_path_buf();
        let stem = gguf.file_stem().unwrap().to_string_lossy().to_string();
        Some((ModelManager::new(dir), stem))
    }

    /// Unloading a previously-loaded model returns its memory footprint.
    #[test]
    fn test_unload_known_model_returns_memory_bytes() {
        let Some((manager, name)) = gated() else { return };

        manager.load_model(&name, None).unwrap();
        let freed = manager.unload_model(&name).unwrap();
        assert!(freed > 0, "expected freed > 0, got {}", freed);
        assert!(manager.get_model(&name).is_none());
    }

    /// Idempotency: unloading the same model twice returns ModelNotFound.
    #[test]
    fn test_unload_twice_returns_error() {
        let Some((manager, name)) = gated() else { return };

        manager.load_model(&name, None).unwrap();
        manager.unload_model(&name).unwrap();
        let second = manager.unload_model(&name);
        assert!(matches!(second, Err(RuntimedError::ModelNotFound(_))));
    }
}
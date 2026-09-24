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

    /// Unloading a previously-loaded model returns its memory footprint.
    #[test]
    fn test_unload_known_model_returns_memory_bytes() {
        let tmp = tempdir().unwrap();
        let manager = ModelManager::new(tmp.path());

        manager.load_model("phi-3", None).unwrap();
        let freed = manager.unload_model("phi-3").unwrap();
        assert!(freed > 0, "expected freed > 0, got {}", freed);
        assert!(manager.get_model("phi-3").is_none());
    }

    /// Idempotency: unloading the same model twice returns ModelNotFound.
    #[test]
    fn test_unload_twice_returns_error() {
        let tmp = tempdir().unwrap();
        let manager = ModelManager::new(tmp.path());

        manager.load_model("m", None).unwrap();
        manager.unload_model("m").unwrap();
        let second = manager.unload_model("m");
        assert!(matches!(second, Err(RuntimedError::ModelNotFound(_))));
    }
}
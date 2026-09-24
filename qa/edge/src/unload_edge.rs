//! Edge tests for model dynamic unloading errors.

#[cfg(test)]
mod tests {
    use runtimed_core::error::RuntimedError;
    use runtimed_core::model::ModelManager;
    use tempfile::tempdir;

    #[test]
    fn test_unload_nonexistent_model_returns_error() {
        let tmp = tempdir().unwrap();
        let manager = ModelManager::new(tmp.path());

        let res = manager.unload_model("ghost-model-xyz");
        assert!(res.is_err());
        assert!(matches!(res.unwrap_err(), RuntimedError::ModelNotFound(_)));
    }
}

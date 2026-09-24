//! Unit QA tests for ModelManager.

#[cfg(test)]
mod tests {
    use runtimed_core::model::ModelManager;
    use tempfile::tempdir;

    #[test]
    fn test_load_and_get_model() {
        let tmp = tempdir().unwrap();
        let manager = ModelManager::new(tmp.path());

        assert!(manager.get_model("test-model").is_none());

        let loaded = manager.load_model("test-model", Some("vulkan")).unwrap();
        assert_eq!(loaded.name, "test-model");
        assert_eq!(loaded.compute_backend, "vulkan");

        let retrieved = manager.get_model("test-model").unwrap();
        assert_eq!(retrieved, loaded);
    }

    #[test]
    fn test_unload_model() {
        let tmp = tempdir().unwrap();
        let manager = ModelManager::new(tmp.path());

        manager.load_model("m1", None).unwrap();
        assert!(manager.get_model("m1").is_some());

        let freed = manager.unload_model("m1").unwrap();
        assert!(freed > 0);
        assert!(manager.get_model("m1").is_none());

        let err = manager.unload_model("m1");
        assert!(err.is_err());
    }

    #[test]
    fn test_list_active_models() {
        let tmp = tempdir().unwrap();
        let manager = ModelManager::new(tmp.path());

        manager.load_model("m1", None).unwrap();
        manager.load_model("m2", None).unwrap();

        let list = manager.list_active();
        assert_eq!(list.len(), 2);
    }
}

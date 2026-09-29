//! Unit QA tests for ModelManager.
//!
//! Real-file cases gate on `SYNTROP_TEST_GGUF`; not-found needs no files.

#[cfg(test)]
mod tests {
    use runtimed_core::error::RuntimedError;
    use runtimed_core::model::ModelManager;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn gated() -> Option<(ModelManager, String)> {
        let gguf = PathBuf::from(std::env::var("SYNTROP_TEST_GGUF").ok()?);
        if !gguf.exists() {
            eprintln!("skip: SYNTROP_TEST_GGUF not set or missing");
            return None;
        }
        let dir = gguf.parent().unwrap().to_path_buf();
        let stem = gguf.file_stem().unwrap().to_string_lossy().to_string();
        Some((ModelManager::new(dir), stem))
    }

    #[test]
    fn test_load_and_get_model() {
        let Some((manager, name)) = gated() else { return };
        assert!(manager.get_model(&name).is_none());

        let loaded = manager.load_model(&name, None).unwrap();
        assert_eq!(loaded.name, name);
        assert_eq!(loaded.architecture, "qwen2");
        assert!(loaded.parameter_count > 100_000_000, "params: {}", loaded.parameter_count);
        assert_eq!(loaded.context_window, 32768);
        assert_eq!(loaded.compute_backend, "cpu");

        // Second load hits the cache and returns identical metadata.
        let again = manager.load_model(&name, None).unwrap();
        assert_eq!(again, loaded);
        assert_eq!(manager.get_model(&name).unwrap(), loaded);
    }

    #[test]
    fn test_load_missing_model_returns_not_found() {
        let tmp = tempdir().unwrap();
        let manager = ModelManager::new(tmp.path());
        match manager.load_model("ghost-model-xyz", None) {
            Err(RuntimedError::ModelNotFound(n)) => assert_eq!(n, "ghost-model-xyz"),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_attach_missing_model_returns_not_found() {
        let tmp = tempdir().unwrap();
        let manager = ModelManager::new(tmp.path());
        match manager.attach_vision("ghost-model-xyz", "ghost-mmproj") {
            Err(RuntimedError::ModelNotFound(n)) => assert_eq!(n, "ghost-model-xyz"),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
        match manager.attach_lora("ghost-model-xyz", "ghost-lora") {
            Err(RuntimedError::ModelNotFound(n)) => assert_eq!(n, "ghost-model-xyz"),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_empty_manager_lists_nothing() {
        let tmp = tempdir().unwrap();
        let manager = ModelManager::new(tmp.path());
        assert!(manager.get_entry("ghost-model-xyz").is_none());
        assert!(manager.get_model("ghost-model-xyz").is_none());
        assert!(manager.list_active().is_empty());
    }

    #[test]
    fn test_load_rejects_unknown_backend() {
        let Some((manager, name)) = gated() else { return };
        match manager.load_model(&name, Some("vulkan")) {
            Err(RuntimedError::HardwareAllocation(_)) => {}
            other => panic!("expected HardwareAllocation, got {other:?}"),
        }
    }

    #[test]
    fn test_unload_model() {
        let Some((manager, name)) = gated() else { return };
        manager.load_model(&name, None).unwrap();
        assert!(manager.get_model(&name).is_some());

        let freed = manager.unload_model(&name).unwrap();
        assert!(freed > 0);
        assert!(manager.get_model(&name).is_none());

        let err = manager.unload_model(&name);
        assert!(err.is_err());
    }

    #[test]
    fn test_list_active_models() {
        let Some((manager, name)) = gated() else { return };
        assert!(manager.list_active().is_empty());
        manager.load_model(&name, None).unwrap();

        let list = manager.list_active();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, name);
    }

    #[test]
    fn test_registry_refuses_unlisted_but_allows_match() {
        // Garbage bytes (not a GGUF): verification runs before parsing,
        // so a pin refusal proves the gate; a pin match proceeds to the
        // parse error instead. No model files needed.
        let tmp = tempdir().unwrap();
        std::fs::write(tmp.path().join("w.gguf"), b"not-a-gguf").unwrap();
        // No registry: fail-open, straight to the parse error.
        let open = ModelManager::new(tmp.path());
        let err = open.load_model("w", None).unwrap_err().to_string();
        assert!(!err.contains("weight verification"), "{err}");
        // Registry present, name unlisted: refused before parsing.
        std::fs::write(
            tmp.path().join("registry.toml"),
            "[[weight]]\nname = \"other\"\nurl = \"https://example.invalid/o\"\nsha256 = \"00\"\n",
        )
        .unwrap();
        let closed = ModelManager::new(tmp.path());
        let err = closed.load_model("w", None).unwrap_err().to_string();
        assert!(err.contains("weight verification"), "{err}");
        assert!(err.contains("unknown weight"), "{err}");
        // Pinned with the right hash: gate passes, parse error surfaces.
        // (sha256 of "not-a-gguf", precomputed: no new test deps.)
        let hash = "692a62c67c7402c343f2d25883b802a18fd7f4b60340c880e5256e00700996dc";
        std::fs::write(
            tmp.path().join("registry.toml"),
            format!("[[weight]]\nname = \"w\"\nurl = \"https://example.invalid/w\"\nsha256 = \"{hash}\"\n"),
        )
        .unwrap();
        let pinned = ModelManager::new(tmp.path());
        let err = pinned.load_model("w", None).unwrap_err().to_string();
        assert!(!err.contains("weight verification"), "{err}");
    }

    #[test]
    fn test_cas_fd_missing_socket_returns_none() {
        assert!(runtimed_core::model::cas_fd::fetch_model_fd("ghost-model-xyz").is_none());
    }

    #[test]
    fn test_psi_parse_avg10_and_shedding() {
        let sample = "some avg10=28.50 avg60=12.20 avg300=5.10 total=123456\nfull avg10=6.20 avg60=2.10 avg300=0.50 total=4567\n";
        assert_eq!(runtimed_core::psi::parse_psi_avg10(sample, "some"), Some(28.50));
        assert_eq!(runtimed_core::psi::parse_psi_avg10(sample, "full"), Some(6.20));
        assert_eq!(runtimed_core::psi::parse_psi_avg10(sample, "missing"), None);

        let tmp = tempdir().unwrap();
        let manager = ModelManager::new(tmp.path());
        assert!(!runtimed_core::psi::evaluate_and_shed_memory(&manager, 1.0, 0.0));
        assert!(runtimed_core::psi::evaluate_and_shed_memory(&manager, 30.0, 8.0));
    }
}

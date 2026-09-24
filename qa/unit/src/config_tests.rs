//! Unit QA tests for RuntimedConfig loading and defaults.

#[cfg(test)]
mod tests {
    use runtimed_core::config::{
        RuntimedConfig, DEFAULT_MODELS_PATH, DEFAULT_SOCKET_PATH,
    };
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_default_config() {
        let config = RuntimedConfig::default();
        assert_eq!(config.socket_path.to_str().unwrap(), DEFAULT_SOCKET_PATH);
        assert_eq!(config.models_dir.to_str().unwrap(), DEFAULT_MODELS_PATH);
        assert_eq!(config.default_model, "qwen2.5-coder-7b");
        assert_eq!(config.max_context_window, 8192);
        assert_eq!(config.max_concurrent_requests, 4);
    }

    #[test]
    fn test_load_from_valid_toml() {
        let tmp = tempdir().unwrap();
        let conf_file = tmp.path().join("runtimed.conf");

        let toml_data = r#"
            socket_path = "/tmp/custom_runtimed.sock"
            models_dir = "/tmp/custom_models"
            default_model = "custom-phi3"
            max_context_window = 4096
            max_concurrent_requests = 2
        "#;
        fs::write(&conf_file, toml_data).unwrap();

        let loaded = RuntimedConfig::load_or_default(&conf_file).unwrap();
        assert_eq!(loaded.socket_path.to_str().unwrap(), "/tmp/custom_runtimed.sock");
        assert_eq!(loaded.default_model, "custom-phi3");
        assert_eq!(loaded.max_context_window, 4096);
    }

    #[test]
    fn test_missing_file_fallback() {
        let tmp = tempdir().unwrap();
        let missing = tmp.path().join("nonexistent.conf");

        let loaded = RuntimedConfig::load_or_default(missing).unwrap();
        assert_eq!(loaded, RuntimedConfig::default());
    }
}

//! Unit QA tests for runtimed Varlink framing and method dispatch.

#[cfg(test)]
mod tests {
    use runtimed_core::model::ModelManager;
    use runtimed_daemon::varlink::{
        handle_service_call, Runtime1Handler, VarlinkReply,
    };
    use serde_json::json;
    use std::sync::Arc;
    use tempfile::tempdir;

    #[test]
    fn test_varlink_reply_to_bytes() {
        let reply = VarlinkReply::ok(json!({ "status": "active" }));
        let bytes = reply.to_bytes();
        assert_eq!(*bytes.last().unwrap(), 0x00);
        let parsed: serde_json::Value =
            serde_json::from_slice(&bytes[..bytes.len() - 1]).unwrap();
        assert_eq!(parsed["parameters"]["status"], "active");
    }

    #[test]
    fn test_handle_service_get_info() {
        let reply = handle_service_call("org.varlink.service.GetInfo", None);
        assert!(reply.is_some());
        let params = reply.unwrap().parameters.unwrap();
        assert_eq!(params["product"], "runtimed");
    }

    #[test]
    fn test_handle_service_get_interface_description() {
        let params = json!({ "interface": "io.syntrop.Runtime1" });
        let reply = handle_service_call(
            "org.varlink.service.GetInterfaceDescription",
            Some(&params),
        );
        assert!(reply.is_some());
        let desc = reply.unwrap().parameters.unwrap();
        assert!(desc["description"]
            .as_str()
            .unwrap()
            .contains("interface io.syntrop.Runtime1"));
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

    #[tokio::test]
    async fn test_runtime1_generate() {
        let Some((manager, name)) = gated() else { return };
        let handler = Runtime1Handler::new(Arc::new(manager));

        let params = json!({
            "model": name,
            "prompt": "Inspect kernel error logs",
            "max_tokens": 16,
            "temperature": 0.0
        });

        let reply = handler
            .handle_call("io.syntrop.Runtime1.Generate", Some(&params))
            .await
            .unwrap();

        assert!(reply.error.is_none(), "error: {:?}", reply.error);
        let res = reply.parameters.unwrap();
        assert!(!res["result"]["text"].as_str().unwrap().is_empty());
        assert!(res["result"]["prompt_tokens"].as_u64().unwrap() > 0);
        assert!(res["result"]["completion_tokens"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_runtime1_embed() {
        let tmp = tempdir().unwrap();
        let manager = Arc::new(ModelManager::new(tmp.path()));
        let handler = Runtime1Handler::new(manager);

        let params = json!({
            "model": "qwen2.5-coder-7b",
            "text": "cgroup memory limit reached"
        });

        let reply = handler
            .handle_call("io.syntrop.Runtime1.Embed", Some(&params))
            .await
            .unwrap();

        assert!(reply.error.is_none());
        let res = reply.parameters.unwrap();
        assert_eq!(res["embedding"].as_array().unwrap().len(), 128);
    }

    #[tokio::test]
    async fn test_runtime1_status_and_unload() {
        let Some((manager, name)) = gated() else { return };
        let manager = Arc::new(manager);
        manager.load_model(&name, None).unwrap();
        let handler = Runtime1Handler::new(manager);

        let params = json!({ "model": name });

        let status_reply = handler
            .handle_call("io.syntrop.Runtime1.GetModelStatus", Some(&params))
            .await
            .unwrap();
        assert_eq!(
            status_reply.parameters.unwrap()["status"].as_str().unwrap(),
            "loaded"
        );

        let unload_reply = handler
            .handle_call("io.syntrop.Runtime1.UnloadModel", Some(&params))
            .await
            .unwrap();
        assert!(unload_reply.error.is_none());
    }
}

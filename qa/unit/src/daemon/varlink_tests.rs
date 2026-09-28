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

    #[tokio::test]
    async fn test_runtime1_get_load_reports_free_slots() {
        let tmp = tempdir().unwrap();
        let handler = Runtime1Handler::new(Arc::new(ModelManager::new(tmp.path())));
        let reply = handler.handle_call("io.syntrop.Runtime1.GetLoad", None).await.unwrap();
        assert!(reply.error.is_none());
        let params = reply.parameters.unwrap();
        assert_eq!(params["available_slots"], 4);
        assert_eq!(params["max_slots"], 4);
        assert_eq!(params["used_bytes"], 0);
        assert_eq!(params["models"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_runtime1_constructor_clamps_zero_permits() {
        let tmp = tempdir().unwrap();
        let handler =
            Runtime1Handler::with_max_concurrent(Arc::new(ModelManager::new(tmp.path())), 0);
        let reply = handler.handle_call("io.syntrop.Runtime1.GetLoad", None).await.unwrap();
        assert_eq!(reply.parameters.unwrap()["max_slots"], 1);
    }

    #[tokio::test]
    async fn test_runtime1_lifecycle_without_models() {
        let tmp = tempdir().unwrap();
        let handler = Runtime1Handler::new(Arc::new(ModelManager::new(tmp.path())));
        let ghost = json!({ "model": "ghost-model-xyz" });

        let status = handler
            .handle_call("io.syntrop.Runtime1.GetModelStatus", Some(&ghost))
            .await
            .unwrap();
        assert!(status.error.is_none());
        let params = status.parameters.unwrap();
        assert_eq!(params["status"], "not_loaded");
        assert!(params["model"].is_null());

        let missing = handler
            .handle_call("io.syntrop.Runtime1.GetModelStatus", None)
            .await
            .unwrap();
        assert_eq!(missing.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));

        let unload = handler
            .handle_call("io.syntrop.Runtime1.UnloadModel", Some(&ghost))
            .await
            .unwrap();
        assert_eq!(unload.error.as_deref(), Some("io.syntrop.Runtime1.ModelNotFound"));

        let list = handler
            .handle_call("io.syntrop.Runtime1.ListLoadedModels", None)
            .await
            .unwrap();
        assert_eq!(list.parameters.unwrap()["models"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_runtime1_attach_validates_params() {
        let tmp = tempdir().unwrap();
        let handler = Runtime1Handler::new(Arc::new(ModelManager::new(tmp.path())));

        for method in ["io.syntrop.Runtime1.AttachVision", "io.syntrop.Runtime1.AttachLora"] {
            let none = handler.handle_call(method, None).await.unwrap();
            assert_eq!(none.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));
            assert_eq!(none.parameters.unwrap()["parameter"], "parameters");

            let empty = json!({});
            let no_model = handler.handle_call(method, Some(&empty)).await.unwrap();
            assert_eq!(no_model.parameters.unwrap()["parameter"], "model");
        }

        let model_only = json!({ "model": "m" });
        let vision = handler
            .handle_call("io.syntrop.Runtime1.AttachVision", Some(&model_only))
            .await
            .unwrap();
        assert_eq!(vision.parameters.unwrap()["parameter"], "mmproj");
        let lora = handler
            .handle_call("io.syntrop.Runtime1.AttachLora", Some(&model_only))
            .await
            .unwrap();
        assert_eq!(lora.parameters.unwrap()["parameter"], "lora");
    }

    #[tokio::test]
    async fn test_runtime1_generate_validates_params() {
        let tmp = tempdir().unwrap();
        let handler = Runtime1Handler::new(Arc::new(ModelManager::new(tmp.path())));

        let none = handler.handle_call("io.syntrop.Runtime1.Generate", None).await.unwrap();
        assert_eq!(none.parameters.unwrap()["parameter"], "parameters");

        let empty = json!({});
        let no_model = handler
            .handle_call("io.syntrop.Runtime1.Generate", Some(&empty))
            .await
            .unwrap();
        assert_eq!(no_model.parameters.unwrap()["parameter"], "model");

        let no_prompt = json!({ "model": "m" });
        let missing = handler
            .handle_call("io.syntrop.Runtime1.Generate", Some(&no_prompt))
            .await
            .unwrap();
        assert_eq!(missing.parameters.unwrap()["parameter"], "prompt");
    }

    #[tokio::test]
    async fn test_runtime1_embed_validates_params() {
        let tmp = tempdir().unwrap();
        let handler = Runtime1Handler::new(Arc::new(ModelManager::new(tmp.path())));

        let none = handler.handle_call("io.syntrop.Runtime1.Embed", None).await.unwrap();
        assert_eq!(none.error.as_deref(), Some("io.syntrop.Runtime1.InvalidParameter"));

        let empty = json!({});
        let no_text = handler.handle_call("io.syntrop.Runtime1.Embed", Some(&empty)).await.unwrap();
        assert_eq!(no_text.parameters.unwrap()["parameter"], "text");
    }

    #[tokio::test]
    async fn test_runtime1_unknown_method_returns_none() {
        let tmp = tempdir().unwrap();
        let handler = Runtime1Handler::new(Arc::new(ModelManager::new(tmp.path())));
        assert!(handler.handle_call("no.such.Method", None).await.is_none());
    }
}

//! Integration tests for speculative decoding via io.syntrop.Runtime1 Varlink RPC.

use runtimed_core::model::ModelManager;
use runtimed_daemon::varlink::{handle_service_call, handle_client, Runtime1Handler, TrustedGroup};
use serde_json::json;
use std::sync::Arc;
use tempfile::tempdir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::watch;

#[tokio::test]
async fn test_interface_description_declares_speculative_draft_model() {
    let params = json!({ "interface": "io.syntrop.Runtime1" });
    let reply = handle_service_call(
        "org.varlink.service.GetInterfaceDescription",
        Some(&params),
    )
    .expect("GetInterfaceDescription reply");
    assert!(reply.error.is_none());
    let desc = reply.parameters.expect("parameters")["description"]
        .as_str()
        .expect("str")
        .to_string();
    assert!(
        desc.contains("speculative_draft_model: ?string"),
        "Interface description missing speculative_draft_model parameter"
    );
}

#[tokio::test]
async fn test_speculative_generation_fails_gracefully_when_draft_missing() {
    let tmp = tempdir().expect("tempdir");
    let manager = Arc::new(ModelManager::new(tmp.path()));
    let handler = Runtime1Handler::new(manager);

    let params = json!({
        "model": "nonexistent-target",
        "speculative_draft_model": "nonexistent-draft",
        "prompt": "Evaluate CPU speculative draft execution",
        "max_tokens": 16
    });

    let reply = handler
        .handle_call("io.syntrop.Runtime1.Generate", Some(&params))
        .await
        .expect("reply");

    assert_eq!(
        reply.error.as_deref(),
        Some("io.syntrop.Runtime1.GenerationFailed")
    );
}

#[tokio::test]
async fn test_speculative_generation_wire_protocol_roundtrip() {
    let tmp = tempdir().expect("tempdir");
    let manager = Arc::new(ModelManager::new(tmp.path()));
    let handler = Arc::new(Runtime1Handler::new(manager));

    let (client, server) = UnixStream::pair().expect("unix stream pair");
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let trusted = TrustedGroup::from_gid(unsafe { libc::getgid() });

    let server_task = tokio::spawn(async move {
        let _ = handle_client(server, handler, trusted, shutdown_rx).await;
    });

    let (mut reader, mut writer) = client.into_split();

    let req_payload = json!({
        "method": "io.syntrop.Runtime1.Generate",
        "parameters": {
            "model": "qwen2.5-7b",
            "speculative_draft_model": "qwen2.5-0.5b",
            "prompt": "Status of cgroup memory pressure",
            "max_tokens": 8
        }
    });

    let mut req_bytes = serde_json::to_vec(&req_payload).expect("to_vec");
    req_bytes.push(0x00);
    writer.write_all(&req_bytes).await.expect("write_all");

    let mut buf = vec![0u8; 4096];
    let n = reader.read(&mut buf).await.expect("read");
    assert!(n > 0);
    assert_eq!(buf[n - 1], 0x00);

    let reply_val: serde_json::Value =
        serde_json::from_slice(&buf[..n - 1]).expect("json from reply");
    assert_eq!(
        reply_val["error"].as_str(),
        Some("io.syntrop.Runtime1.GenerationFailed")
    );

    drop(writer);
    let _ = server_task.await;
}

#[tokio::test]
async fn test_gated_speculative_generation_succeeds_if_gguf_present() {
    let Ok(gguf_var) = std::env::var("SYNTROP_TEST_GGUF") else {
        return;
    };
    let gguf_path = std::path::PathBuf::from(gguf_var);
    if !gguf_path.exists() {
        return;
    }

    let dir = gguf_path.parent().expect("parent").to_path_buf();
    let stem = gguf_path.file_stem().expect("stem").to_string_lossy().to_string();

    let manager = Arc::new(ModelManager::new(dir));
    let handler = Runtime1Handler::new(manager);

    let params = json!({
        "model": stem.clone(),
        "speculative_draft_model": stem,
        "prompt": "Hello world",
        "max_tokens": 4,
        "temperature": 0.0
    });

    let reply = handler
        .handle_call("io.syntrop.Runtime1.Generate", Some(&params))
        .await
        .expect("reply");

    assert!(reply.error.is_none(), "unexpected error: {:?}", reply.error);
    let res = reply.parameters.expect("parameters")["result"].clone();
    assert!(res["completion_tokens"].as_u64().unwrap_or(0) > 0);
}

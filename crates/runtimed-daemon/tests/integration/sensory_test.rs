//! Integration tests for io.syntrop.Sensory1 Varlink RPC and sensory pipelines.

use runtimed_core::model::ModelManager;
use runtimed_daemon::varlink::{handle_client, handle_service_call, Runtime1Handler, TrustedGroup};
use serde_json::json;
use std::sync::Arc;
use tempfile::tempdir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::watch;

#[tokio::test]
async fn test_sensory1_service_introspection() {
    let info_reply = handle_service_call("org.varlink.service.GetInfo", None)
        .expect("GetInfo reply");
    assert!(info_reply.error.is_none());
    let ifaces = info_reply.parameters.expect("params")["interfaces"]
        .as_array()
        .expect("interfaces array")
        .clone();
    assert!(ifaces.iter().any(|v| v.as_str() == Some("io.syntrop.Sensory1")));

    let params = json!({ "interface": "io.syntrop.Sensory1" });
    let desc_reply = handle_service_call(
        "org.varlink.service.GetInterfaceDescription",
        Some(&params),
    )
    .expect("GetInterfaceDescription reply");
    assert!(desc_reply.error.is_none());
    let desc = desc_reply.parameters.expect("params")["description"]
        .as_str()
        .expect("str")
        .to_string();
    assert!(desc.contains("interface io.syntrop.Sensory1"));
    assert!(desc.contains("method CaptureAudio"));
    assert!(desc.contains("method CaptureFrame"));
    assert!(desc.contains("method CaptureScreen"));
    assert!(desc.contains("method GetOperatorPresence"));
}

#[tokio::test]
async fn test_sensory1_wire_protocol_capture_audio() {
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
        "method": "io.syntrop.Sensory1.CaptureAudio",
        "parameters": {
            "duration_ms": 200,
            "sample_rate": 16000
        }
    });

    let mut req_bytes = serde_json::to_vec(&req_payload).expect("to_vec");
    req_bytes.push(0x00);
    writer.write_all(&req_bytes).await.expect("write_all");

    let mut buf = vec![0u8; 16384];
    let n = reader.read(&mut buf).await.expect("read");
    assert!(n > 0);
    assert_eq!(buf[n - 1], 0x00);

    let reply: serde_json::Value =
        serde_json::from_slice(&buf[..n - 1]).expect("json from reply");
    assert!(reply["error"].is_null());
    let params = &reply["parameters"];
    assert!(params["audio_pcm_base64"].is_string());
    assert_eq!(params["sample_rate"], 16000);
    assert_eq!(params["channels"], 1);

    drop(writer);
    let _ = server_task.await;
}

#[tokio::test]
async fn test_sensory1_wire_protocol_capture_frame_and_screen() {
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

    // 1. Test CaptureFrame
    let req_frame = json!({
        "method": "io.syntrop.Sensory1.CaptureFrame",
        "parameters": {
            "width": 320,
            "height": 240
        }
    });
    let mut req_bytes = serde_json::to_vec(&req_frame).expect("to_vec");
    req_bytes.push(0x00);
    writer.write_all(&req_bytes).await.expect("write_all");

    let mut buf = vec![0u8; 65536];
    let n = reader.read(&mut buf).await.expect("read");
    assert!(n > 0);
    let reply_frame: serde_json::Value =
        serde_json::from_slice(&buf[..n - 1]).expect("json from reply");
    assert!(reply_frame["error"].is_null());
    assert_eq!(reply_frame["parameters"]["format"], "png");
    assert!(!reply_frame["parameters"]["image_base64"].as_str().unwrap().is_empty());

    // 2. Test CaptureScreen
    let req_screen = json!({
        "method": "io.syntrop.Sensory1.CaptureScreen",
        "parameters": {}
    });
    let mut req_bytes2 = serde_json::to_vec(&req_screen).expect("to_vec");
    req_bytes2.push(0x00);
    writer.write_all(&req_bytes2).await.expect("write_all");

    let mut buf2 = vec![0u8; 131072];
    let n2 = reader.read(&mut buf2).await.expect("read");
    assert!(n2 > 0);
    let reply_screen: serde_json::Value =
        serde_json::from_slice(&buf2[..n2 - 1]).expect("json from reply");
    assert!(reply_screen["error"].is_null());
    assert_eq!(reply_screen["parameters"]["format"], "png");
    assert!(!reply_screen["parameters"]["image_base64"].as_str().unwrap().is_empty());

    drop(writer);
    let _ = server_task.await;
}

#[tokio::test]
async fn test_sensory1_wire_protocol_get_operator_presence() {
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

    let req_presence = json!({
        "method": "io.syntrop.Sensory1.GetOperatorPresence",
        "parameters": {}
    });
    let mut req_bytes = serde_json::to_vec(&req_presence).expect("to_vec");
    req_bytes.push(0x00);
    writer.write_all(&req_bytes).await.expect("write_all");

    let mut buf = vec![0u8; 16384];
    let n = reader.read(&mut buf).await.expect("read");
    assert!(n > 0);
    let reply: serde_json::Value =
        serde_json::from_slice(&buf[..n - 1]).expect("json from reply");
    assert!(reply["error"].is_null());
    let params = &reply["parameters"];
    assert!(params["present"].is_boolean());
    assert!(params["confidence"].is_f64());
    assert!(params["reason"].is_string());

    drop(writer);
    let _ = server_task.await;
}

#[tokio::test]
async fn test_sensory1_wire_protocol_capture_frame_device_not_found() {
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

    let req_frame = json!({
        "method": "io.syntrop.Sensory1.CaptureFrame",
        "parameters": {
            "device": "/dev/video_phantom_nonexistent"
        }
    });
    let mut req_bytes = serde_json::to_vec(&req_frame).expect("to_vec");
    req_bytes.push(0x00);
    writer.write_all(&req_bytes).await.expect("write_all");

    let mut buf = vec![0u8; 4096];
    let n = reader.read(&mut buf).await.expect("read");
    assert!(n > 0);
    let reply: serde_json::Value =
        serde_json::from_slice(&buf[..n - 1]).expect("json from reply");
    assert_eq!(reply["error"].as_str(), Some("io.syntrop.Sensory1.DeviceNotFound"));
    assert_eq!(
        reply["parameters"]["device"].as_str(),
        Some("/dev/video_phantom_nonexistent")
    );

    drop(writer);
    let _ = server_task.await;
}

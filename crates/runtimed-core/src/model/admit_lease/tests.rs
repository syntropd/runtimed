//! Tests for inferenced compute lease admission.

use super::client::LeaseClient;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::Duration;

fn read_frame(stream: &mut UnixStream) -> Value {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        let n = stream.read(&mut chunk).unwrap();
        assert_ne!(n, 0, "client disconnected mid-frame");
        buf.extend_from_slice(&chunk[..n]);
        if buf.contains(&0) {
            break;
        }
    }
    buf.pop();
    serde_json::from_slice(&buf).unwrap()
}

fn write_frame(stream: &mut UnixStream, reply: &Value) {
    let mut bytes = serde_json::to_vec(reply).unwrap();
    bytes.push(0);
    stream.write_all(&bytes).unwrap();
}

fn fake_landlord_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lease-{tag}-{}.sock", std::process::id()))
}

/// inferenced reclaims a lease the moment its client disconnects.
/// The permit must hold the connection open after acquire and
/// release over that same connection. Fails on a client that
/// connects per call (lease instantly orphaned).
#[test]
fn permit_holds_connection_until_release() {
    let path = fake_landlord_path("hold");
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).unwrap();
    let server = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let req = read_frame(&mut sock);
        assert!(req["method"].as_str().unwrap().contains("AcquireLease"));
        write_frame(&mut sock, &json!({"parameters": {"lease_id": "lease-1"}}));
        // Client must STAY connected: idle read times out (held),
        // EOF means it dropped (lease would be reclaimed).
        sock.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
        let mut one = [0u8; 1];
        match sock.read(&mut one) {
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                ) => {}
            Ok(0) => panic!("client disconnected right after acquire"),
            other => panic!("unexpected hold probe: {other:?}"),
        }
        // Release must arrive on this SAME connection.
        sock.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let rel = read_frame(&mut sock);
        assert!(rel["method"].as_str().unwrap().contains("ReleaseLease"));
        assert_eq!(rel["parameters"]["lease_id"], "lease-1");
        write_frame(&mut sock, &json!({"parameters": {}}));
        // Client closes after release.
        assert_eq!(sock.read(&mut one).unwrap(), 0, "no clean close after release");
    });
    let client = LeaseClient::new(&path);
    let permit = client.acquire(1024).unwrap().expect("grant");
    std::thread::sleep(Duration::from_secs(1));
    permit.release().unwrap();
    server.join().unwrap();
    let _ = std::fs::remove_file(&path);
}

#[test]
fn refusal_fails_acquire() {
    let path = fake_landlord_path("refuse");
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).unwrap();
    let server = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let _ = read_frame(&mut sock);
        write_frame(
            &mut sock,
            &json!({"error": "io.systemd.inferenced1.ResourceExhaustion",
                    "parameters": {"error": "plane full"}}),
        );
    });
    let client = LeaseClient::new(&path);
    let err = client.acquire(1024).unwrap_err().to_string();
    assert!(err.contains("plane full"), "{err}");
    server.join().unwrap();
    let _ = std::fs::remove_file(&path);
}

#[test]
fn missing_socket_loads_unenforced() {
    let client = LeaseClient::new("/nonexistent-dir-xyz/no.sock");
    assert!(client.acquire(1024).unwrap().is_none());
}

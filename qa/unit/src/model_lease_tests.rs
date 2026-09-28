//! Unit QA tests for inferenced lease admission (see `admit_lease`).
//!
//! A fake inferenced Varlink server scripts Acquire/Release replies so
//! the client, the refusal path, and the load-failure release path are
//! covered with no daemon and no model weights.

#[cfg(test)]
mod tests {
    use runtimed_core::model::admit_lease::{LeaseClient, INFERENCED_SOCKET_ENV};
    use runtimed_core::model::ModelManager;
    use serde_json::{json, Value};
    use std::io::{Read, Write};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::sync::{Arc, Mutex};
    use tempfile::tempdir;

    /// Scripted fake inferenced: replies per method, records requests.
    struct FakeInferenced {
        _dir: tempfile::TempDir,
        path: std::path::PathBuf,
        seen: Arc<Mutex<Vec<Value>>>,
    }

    impl FakeInferenced {
        fn start(respond: impl Fn(&Value) -> Value + Send + Sync + 'static) -> Self {
            let dir = tempdir().unwrap();
            let path = dir.path().join("inferenced.sock");
            let listener = UnixListener::bind(&path).unwrap();
            let seen = Arc::new(Mutex::new(Vec::new()));
            let seen_clone = seen.clone();
            std::thread::spawn(move || {
                for conn in listener.incoming().flatten() {
                    if handle_conn(conn, &respond, &seen_clone).is_err() {
                        break;
                    }
                }
            });
            Self { _dir: dir, path, seen }
        }
    }

    fn handle_conn(
        mut conn: UnixStream,
        respond: &impl Fn(&Value) -> Value,
        seen: &Mutex<Vec<Value>>,
    ) -> std::io::Result<()> {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let n = conn.read(&mut chunk)?;
            if n == 0 {
                return Ok(());
            }
            buf.extend_from_slice(&chunk[..n]);
            if buf.contains(&0) {
                break;
            }
        }
        if buf.last() == Some(&0) {
            buf.pop();
        }
        let req: Value = serde_json::from_slice(&buf).unwrap();
        seen.lock().unwrap().push(req.clone());
        let mut reply = serde_json::to_vec(&respond(&req)).unwrap();
        reply.push(0);
        conn.write_all(&reply)?;
        Ok(())
    }

    fn lease_reply() -> Value {
        json!({"parameters": {"lease_id": "lease-123", "plane_id": "cpu-0", "allocated_memory": 1024}})
    }

    #[test]
    fn test_acquire_holds_lease_and_releases() {
        let fake = FakeInferenced::start(|req| {
            let method = req.get("method").and_then(|v| v.as_str()).unwrap_or("");
            assert!(method.ends_with("AcquireLease") || method.ends_with("ReleaseLease"), "{method}");
            if method.ends_with("AcquireLease") {
                let params = &req["parameters"];
                assert_eq!(params["priority"], "Interactive");
                assert!(params["memory_bytes"].as_u64().unwrap() > 0);
                return lease_reply();
            }
            json!({"parameters": {}})
        });
        let client = LeaseClient::new(&fake.path);
        let permit = client.acquire(4096).unwrap().expect("lease held");
        client.release_permit(&permit).unwrap();
        let seen = fake.seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert!(seen[0]["method"].as_str().unwrap().ends_with("AcquireLease"));
        assert!(seen[1]["method"].as_str().unwrap().ends_with("ReleaseLease"));
        assert_eq!(seen[1]["parameters"]["lease_id"], "lease-123");
    }

    #[test]
    fn test_explicit_rejection_fails_closed() {
        let fake = FakeInferenced::start(|_| {
            json!({"error": "io.systemd.inferenced1.ResourceExhaustion",
                   "parameters": {"error": "plane full"}})
        });
        let client = LeaseClient::new(&fake.path);
        let err = client.acquire(4096).unwrap_err();
        assert!(err.to_string().contains("refused"), "{err}");
    }

    #[test]
    fn test_missing_socket_proceeds_without_lease() {
        let client = LeaseClient::new("/tmp/qa-nope-inferenced/never.sock");
        assert!(client.acquire(4096).unwrap().is_none());
    }

    #[test]
    fn test_failed_load_releases_its_permit() {
        let fake = FakeInferenced::start(|req| {
            if req["method"].as_str().unwrap_or("").ends_with("AcquireLease") {
                return lease_reply();
            }
            json!({"parameters": {}})
        });
        // Garbage weights: resolve + admit succeed, weight parse fails.
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("ghost.gguf"), b"not a gguf file").unwrap();
        std::env::set_var(INFERENCED_SOCKET_ENV, &fake.path);
        let manager = ModelManager::new(dir.path());
        let err = manager.load_model("ghost", None).unwrap_err();
        std::env::remove_var(INFERENCED_SOCKET_ENV);
        assert!(!err.to_string().contains("refused"), "{err}");
        let seen = fake.seen.lock().unwrap();
        let methods: Vec<&str> = seen.iter().map(|r| r["method"].as_str().unwrap()).collect();
        assert!(methods.iter().any(|m| m.ends_with("AcquireLease")), "{methods:?}");
        assert!(methods.iter().any(|m| m.ends_with("ReleaseLease")), "{methods:?}");
    }
}

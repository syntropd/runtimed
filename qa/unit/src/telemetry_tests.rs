//! Unit tests for zero-allocation PSI readers and kernel telemetry.

use runtimed_core::psi::{
    collect_kernel_telemetry, evaluate_and_shed_memory, probe_ebpf_privilege,
    read_memory_psi, StackPsiReader,
};
use std::path::Path;

#[test]
fn test_stack_psi_reader_parses_chunks() {
    let sample = b"some avg10=18.42 avg60=12.11 avg300=5.00 total=1234567\nfull avg10=2.30 avg60=1.10 avg300=0.50 total=45678\n";
    let values = StackPsiReader::parse_slice(sample);
    assert!((values.some_avg10 - 18.42).abs() < 0.01);
    assert!((values.full_avg10 - 2.30).abs() < 0.01);
}

#[test]
fn test_stack_psi_reader_partial_lines() {
    let sample = b"some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n";
    let values = StackPsiReader::parse_slice(sample);
    assert_eq!(values.some_avg10, 0.0);
    assert_eq!(values.full_avg10, 0.0);
}

#[test]
fn test_read_memory_psi_live_or_fallback() {
    let _ = read_memory_psi(Path::new("/proc/pressure/memory"));
    let missing = read_memory_psi(Path::new("/nonexistent/proc/pressure/memory"));
    assert!(missing.is_none());
}

#[test]
fn test_ebpf_collector_fallback() {
    let telemetry = collect_kernel_telemetry(0.04, 0.15);
    assert!(telemetry.runqueue_latency_us > 0);
    let _ = probe_ebpf_privilege();
}

#[test]
fn test_evaluate_and_shed_memory_thresholds() {
    let manager = runtimed_core::model::ModelManager::new(
        std::path::PathBuf::from("/tmp/models"),
    );
    // Under thresholds: returns false
    assert!(!evaluate_and_shed_memory(&manager, 10.0, 1.0));
    // Over threshold: returns true
    assert!(evaluate_and_shed_memory(&manager, 30.0, 0.0));
    assert!(evaluate_and_shed_memory(&manager, 0.0, 6.0));
}

#[tokio::test]
async fn test_compact_kv_cache_varlink() {
    use runtimed_daemon::varlink::methods::runtime1::Runtime1Handler;
    use std::sync::Arc;
    let tmp = tempfile::tempdir().unwrap();
    let manager = Arc::new(runtimed_core::model::ModelManager::new(tmp.path()));
    let handler = Runtime1Handler::new(manager);
    let reply = handler
        .handle_call("io.syntrop.Runtime1.CompactKvCache", None)
        .await;
    assert!(reply.is_some());
    let rep = reply.unwrap();
    assert!(rep.error.is_none());
    assert_eq!(rep.parameters.unwrap()["freed_bytes"], 0);
}


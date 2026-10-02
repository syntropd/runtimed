//! Handler implementation for io.syntrop.Telemetry1 Varlink interface.

use crate::varlink::server::protocol::VarlinkReply;
use runtimed_core::psi::{collect_kernel_telemetry, StackPsiReader};
use serde_json::json;
use std::path::Path;

/// Handler for io.syntrop.Telemetry1 method dispatches.
#[derive(Debug, Default, Clone, Copy)]
pub struct Telemetry1Handler;

impl Telemetry1Handler {
    pub fn new() -> Self {
        Self
    }

    /// Dispatches incoming io.syntrop.Telemetry1 method calls.
    pub async fn handle_call(&self, method: &str) -> Option<VarlinkReply> {
        match method {
            "io.syntrop.Telemetry1.GetKernelPressure" => Some(self.handle_get_kernel_pressure().await),
            _ => None,
        }
    }

    async fn handle_get_kernel_pressure(&self) -> VarlinkReply {
        let mem = StackPsiReader::read_path(Path::new("/proc/pressure/memory")).unwrap_or_default();
        let cpu = StackPsiReader::read_path(Path::new("/proc/pressure/cpu")).unwrap_or_default();
        let io = StackPsiReader::read_path(Path::new("/proc/pressure/io")).unwrap_or_default();

        let telemetry = collect_kernel_telemetry(cpu.some_avg10, io.some_avg10);

        VarlinkReply::ok(json!({
            "memory_some": mem.some_avg10,
            "memory_full": mem.full_avg10,
            "cpu_some": cpu.some_avg10,
            "io_some": io.some_avg10,
            "runqueue_latency_us": telemetry.runqueue_latency_us,
            "ebpf_active": telemetry.ebpf_active,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_telemetry1_handler() {
        let handler = Telemetry1Handler::new();
        let reply = handler
            .handle_call("io.syntrop.Telemetry1.GetKernelPressure")
            .await;
        assert!(reply.is_some());
    }
}

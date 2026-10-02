//! Varlink interface definition for io.syntrop.Telemetry1.

/// Varlink interface definition text for io.syntrop.Telemetry1.
pub const IO_SYNTROP_TELEMETRY1_INTERFACE: &str = r#"
interface io.syntrop.Telemetry1

method GetKernelPressure() -> (
  memory_some: float,
  memory_full: float,
  cpu_some: float,
  io_some: float,
  runqueue_latency_us: int,
  ebpf_active: bool
)
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry1_interface_declares_methods() {
        assert!(IO_SYNTROP_TELEMETRY1_INTERFACE.contains("interface io.syntrop.Telemetry1"));
        assert!(IO_SYNTROP_TELEMETRY1_INTERFACE.contains("method GetKernelPressure"));
    }
}

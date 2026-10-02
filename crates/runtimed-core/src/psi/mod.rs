//! Linux Kernel Pressure Stall Information (PSI) integration for runtimed.
//!
//! Exposes background pressure monitoring and proactive memory load-shedding
//! functions. When the host experiences severe memory stall pressure,
//! resident model allocations are evicted before kernel OOM-killer strikes.
//!
//! Submodules:
//! - `monitor`: Implements periodic `/proc/pressure/memory` reading, stall threshold
//!   evaluation, proactive idle model shedding, and background watcher tasks.

pub mod ebpf;
pub mod monitor;
pub mod stack_reader;
pub mod types;

pub use ebpf::{collect_kernel_telemetry, probe_ebpf_privilege, KernelTelemetry};
pub use monitor::{
    evaluate_and_shed_memory,
    parse_psi_avg10,
    read_memory_psi,
    spawn_psi_monitor,
    DEFAULT_PSI_MEMORY_PATH,
};
pub use stack_reader::{StackPsiReader, StackPsiValues};
pub use types::KernelPressureMetrics;

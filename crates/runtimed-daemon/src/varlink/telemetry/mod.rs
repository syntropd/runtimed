//! io.syntrop.Telemetry1 Varlink interface and method handlers.

pub mod interface;
pub mod telemetry1;

pub use interface::IO_SYNTROP_TELEMETRY1_INTERFACE;
pub use telemetry1::Telemetry1Handler;

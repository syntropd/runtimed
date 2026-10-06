//! Daemon service implementation for runtimed.

pub mod activation;
pub mod notify;
pub mod psi;
pub mod runtime;
pub mod sensory;
pub mod threading;
pub mod varlink;

pub use activation::{parse_listen_fds, ActivatedSockets};
pub use notify::{notify_ready, notify_status, notify_stopping, notify_watchdog, send_notify, NOTIFY_MAX};
pub use runtime::{finish_shutdown, join_with_timeout, spawn_watchdog, watchdog_interval};
pub use threading::{detect_cpu_topology, init_threading, CpuTopology, CANDLE_NUM_THREADS_VAR, RAYON_NUM_THREADS_VAR};
pub use varlink::{Runtime1Handler, VarlinkCall, VarlinkReply, VarlinkServer};
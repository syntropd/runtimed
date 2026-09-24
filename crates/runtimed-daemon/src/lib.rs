//! Daemon service implementation for runtimed.

pub mod activation;
pub mod notify;
pub mod runtime;
pub mod varlink;

pub use activation::{parse_listen_fds, ActivatedSockets};
pub use notify::{notify_ready, notify_status, notify_stopping, notify_watchdog, send_notify, NOTIFY_MAX};
pub use runtime::{finish_shutdown, join_with_timeout, spawn_watchdog, watchdog_interval};
pub use varlink::{Runtime1Handler, VarlinkCall, VarlinkReply, VarlinkServer};
//! Daemon service implementation for runtimed.

pub mod activation;
pub mod notify;
pub mod varlink;

pub use activation::{parse_listen_fds, ActivatedSockets};
pub use notify::{notify_ready, notify_status, notify_stopping, notify_watchdog, send_notify};
pub use varlink::{Runtime1Handler, VarlinkCall, VarlinkReply, VarlinkServer};

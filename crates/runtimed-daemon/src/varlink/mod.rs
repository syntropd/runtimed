//! Varlink protocol definitions and server implementation for runtimed.

pub mod auth;
pub mod protocol;
pub mod runtime1;
pub mod server;
pub mod service;

pub use auth::{lookup_group, TrustedGroup, UNRESOLVED_GID};
pub use protocol::{VarlinkCall, VarlinkReply};
pub use runtime1::Runtime1Handler;
pub use server::{handle_client, VarlinkServer, MAX_MSG_BYTES};
pub use service::{handle_service_call, IO_SYNTROP_RUNTIME1_INTERFACE};
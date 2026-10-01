//! Varlink socket transport: auth, framing, listener.

pub mod auth;
pub mod protocol;
#[allow(clippy::module_inception)]
pub mod server;

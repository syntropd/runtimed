//! Varlink protocol definitions and server implementation for runtimed.

pub mod methods;
pub mod sensory;
pub mod server;

pub use methods::runtime1::Runtime1Handler;
pub use methods::service::{handle_service_call, IO_SYNTROP_RUNTIME1_INTERFACE};
pub use sensory::{Sensory1Handler, IO_SYNTROP_SENSORY1_INTERFACE};
pub use server::auth::{lookup_group, TrustedGroup, UNRESOLVED_GID};
pub use server::protocol::{VarlinkCall, VarlinkReply};
pub use server::server::{handle_client, VarlinkServer, MAX_MSG_BYTES};

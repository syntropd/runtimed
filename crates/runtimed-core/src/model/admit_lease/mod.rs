//! Admission control: hold an inferenced compute lease per loaded model.
//!
//! Before weights load, runtimed asks inferenced (the hardware landlord)
//! for a lease on a compute plane; on unload the lease is released. When
//! inferenced is absent the load proceeds without a lease (fail-open,
//! logged): standalone and test setups keep working. An explicit
//! `ResourceExhaustion` refusal is honored (fail-closed): the load fails
//! with [`crate::error::RuntimedError::HardwareAllocation`].

mod client;
pub mod composite;
mod permit;

#[cfg(test)]
mod tests;

pub use client::{LeaseClient, DEFAULT_INFERENCED_SOCKET, INFERENCED_SOCKET_ENV};
pub use composite::{
    CompositeLeaseClient, CompositeLeasePermit, CompositeSliceAllocation, CompositeSliceRequest,
};
pub use permit::LeasePermit;

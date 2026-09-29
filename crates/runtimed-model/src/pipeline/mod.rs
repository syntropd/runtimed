//! Intra-node pipeline parallelism: multi-device forward and P2P DMA transfer.

pub mod forward;
pub mod stage;
pub mod transfer;

pub use forward::forward_pipeline;
pub use stage::PipelineStage;
pub use transfer::{transfer_activation, P2pActivationBuffer};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p2p_buffer_verification() {
        let buf = P2pActivationBuffer::new(16, 256);
        assert_eq!(buf.seq_len, 16);
        assert_eq!(buf.hidden_dim, 256);
    }
}

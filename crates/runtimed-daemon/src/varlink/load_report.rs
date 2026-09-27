//! `GetLoad`: fleet-overflow signal (free slots + resident models).
//!
//! A router (see routerd) polls this on every node and spills
//! `Generate` calls to the least-loaded one. Slots come from the
//! request semaphore; memory sums loaded-model footprints.

use super::protocol::VarlinkReply;
use super::runtime1::Runtime1Handler;
use serde_json::json;

impl Runtime1Handler {
    pub(super) fn handle_get_load(&self) -> VarlinkReply {
        let models = self.model_manager.list_active();
        let used_bytes: usize = models.iter().map(|m| m.memory_bytes).sum();
        VarlinkReply::ok(json!({
            "available_slots": self.semaphore.available_permits(),
            "max_slots": self.max_slots,
            "used_bytes": used_bytes,
            "models": models,
        }))
    }
}

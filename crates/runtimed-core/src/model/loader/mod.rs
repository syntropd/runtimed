//! Model loading, metadata tracking, and lifecycle management.

mod load_model;
mod manager;
mod safetensors;

pub use manager::ModelManager;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn manager_initialization_succeeds() {
        let tmp = TempDir::new().unwrap();
        let mgr = ModelManager::new(tmp.path());
        assert_eq!(mgr.list_active().len(), 0);
    }
}

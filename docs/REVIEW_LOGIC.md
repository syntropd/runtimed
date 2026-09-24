# runtimed Review — Logic and Resource Issues (§4)

## 4.1 `generate_embedding` allocates 128 floats per call without pooling

**File**: `crates/runtimed-core/src/engine/embedder.rs:7-47`

Each call allocates a fresh `Vec<f32>` of length 128. For a daemon fielding concurrent `Embed` calls, this is ~512 bytes per request. Acceptable for the documented `max_concurrent_requests: 4` budget but worth noting when scaling up.

## 4.2 `ModelManager::load_model` returns synthetic metadata without reading the filesystem

**File**: `crates/runtimed-core/src/model/loader.rs:42-65`

```rust
let model = LoadedModel {
    name: name.to_string(),
    architecture: "transformer".to_string(),
    parameter_count: 7_000_000_000,
    memory_bytes: 4_500_000_000,
    context_window: 8192,
    compute_backend: backend_str,
};
```

Every "loaded" model is a synthetic 7B-parameter record. No disk read, no format validation, no actual loading. This is intentional (the runtime is a stub for now), but it means the Varlink `GetModelStatus` and `ListLoadedModels` methods will report fabricated metadata even when no model files exist on disk.

**Recommended fix**: gate the synthetic profile behind a `Config::synthetic_mode` flag so production deployments can clearly distinguish stub state. Or — better — actually validate that a file exists at `models_dir/<name>` before claiming the model is loaded.

## 4.3 `generate_tokens` `prompt_tokens` is word-split, not real tokenizer output

**File**: `crates/runtimed-core/src/engine/generator.rs:43`

```rust
let prompt_tokens = request.prompt.split_whitespace().count().max(1);
```

Whitespace-split word counts are a poor proxy for actual token counts (real tokenizers produce 0.7–1.5 words per token depending on the model). For real models this would under-count by 30-50%, silently exceeding the context window. Acceptable for the stub but the contract should be documented as a stub approximation.

## 4.4 `ModelManager` mutex is locked on every `get_model` and `list_active`

**File**: `crates/runtimed-core/src/model/loader.rs:67-90`

```rust
pub fn get_model(&self, name: &str) -> Option<LoadedModel> {
    self.active_models.lock().ok()?.get(name).cloned()
}
```

Even read-only queries acquire the write mutex. With concurrent Varlink `ListLoadedModels` calls, this serializes unnecessarily. Use a `RwLock` instead.

## 4.5 `ModelManager::list_active` returns synthetic models with no provenance

**File**: `crates/runtimed-core/src/model/loader.rs:85-90`

Same concern as 4.2. The `list_active` method exposes the synthetic models verbatim.

## 4.6 No retention / cap on `max_concurrent_requests`

**File**: `crates/runtimed-core/src/config/runtimed_config.rs:29`

`max_concurrent_requests` is read into config but never enforced. The Varlink handlers run synchronously without any semaphore; a flood of concurrent `Generate` calls would spawn unbounded Tokio tasks and exhaust memory.

**Recommended fix**: use a `tokio::sync::Semaphore` with `max_concurrent_requests` permits. Acquire in `handle_generate` and `handle_embed`.

## 4.7 `runtime1::handle_unload_model` race with concurrent `Generate`

**File**: `crates/runtimed-core/src/model/loader.rs:73-82` + `runtime1.rs:138-156`

`unload_model` removes the model from the active map. A concurrent `Generate` call may have already loaded a `LoadedModel` clone and be mid-tokenization. Unloading the model from the map does not stop the in-flight generation; the model is just removed from the listing. Acceptable for the current stub but should be documented as such.

## 4.8 `main.rs` `RUNTIMED_CONFIG` env var and `--config` flag duplicate

**File**: `crates/runtimed-daemon/src/main.rs:22-28`

Both the `RUNTIMED_CONFIG` env var and the `--config` / `-c` flag set the config path. Clap-based arg parsing would be cleaner and avoid the hand-rolled `args.iter().position(...)` boilerplate. The daemon crate depends on `anyhow` but not `clap`; if clap is acceptable, the parsing becomes consistent with the CLI tool.

# runtimed

Headless Model Execution and Tensor Generation Daemon for the Syntropd OS Suite.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](Cargo.toml)

`runtimed` is the compute worker daemon in the Syntropd AI operating system suite. It loads neural model weights, executes text completions, generates vector embeddings, and manages compute memory lifecycles under native systemd supervision.

---

## Features

- **Socket-Activated Varlink IPC**: Native implementation of `io.syntrop.Runtime1` over Unix domain sockets with socket activation.
- **Dynamic Model Lifecycle Management**: On-demand model loading, hardware backend selection, and clean memory eviction.
- **Fast Token Generation**: Strict context window checking, token budgeting, and execution metrics.
- **Normalized Vector Embeddings**: 128-dimensional L2-normalized vector generation for semantic log and incident retrieval.
- **Zero Dynamic C Dependencies** (CPU build): Directly interfaces with Linux syscalls without `libsystemd.so` or `libdbus-1.so`. The optional `cuda` feature links the NVIDIA driver libraries only.
- **CUDA Acceleration** (optional): F16-resident weights with F32-exact compute on NVIDIA GPUs, one device per daemon. See [docs/CUDA.md](docs/CUDA.md).
- **Systemd Hardening & Device Isolation**: Sandboxed systemd service unit with scoped `char-nvidia*` / `char-drm` / `char-accel` device permissions.

---

## Directory Structure

```
runtimed/
├── Cargo.toml
├── crates/
│   ├── runtimed-core/       # Core model loader, tokenizer, vector embedder
│   ├── runtimed-daemon/     # Daemon binary: socket activation, Varlink server
│   ├── runtimed-gguf/       # GGUF parsing, dequantization, weight registry
│   ├── runtimed-model/      # Owned engine: Gemma4/Qwen2, vision, LoRA, sampler
│   └── runtimectl/          # Admin CLI utility for model interaction
├── qa/
│   ├── unit/                # 1:1 unit tests for all core and daemon functions
│   └── edge/                # Edge cases: context limits, corrupt inputs
├── systemd/                 # runtimed.service and runtimed.socket units
├── sysusers.d/              # User, group, and device access definitions
├── tmpfiles.d/              # Directory lifecycle and permissions
├── install/                 # install.sh and uninstall.sh scripts
└── docs/                    # Architecture, Varlink spec, CLI reference
```

---

## Quickstart

### Build and Test

```bash
cargo build --release
cargo test --workspace
```

### Run Daemon Locally

```bash
cargo run --bin runtimed
```

### Query via runtimectl

```bash
# Generate completion
cargo run --bin runtimectl -- generate "Summarize system error logs"

# Compute vector embedding
cargo run --bin runtimectl -- embed "kernel panic on cpu 0"

# Inspect model status
cargo run --bin runtimectl -- status qwen2.5-coder-7b
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

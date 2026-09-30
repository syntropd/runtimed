# runtimed

Headless Model Execution and Tensor Generation Daemon for the Syntropd OS Suite.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](Cargo.toml)

`runtimed` is the compute worker daemon in the Syntropd AI operating system suite. It loads neural model weights, executes text completions, generates vector embeddings, and manages compute memory lifecycles under native systemd supervision.

---

## Features

- **Socket-Activated Varlink IPC**: Native implementation of `io.syntrop.Runtime1` over Unix domain sockets with socket activation.
- **Constrained / Grammar-Guided Decoding**: Zero-overhead FSM token filtering supporting JSON schema, regex, and Varlink protocols with SIMD-aligned bitset logit masking and vocab trie pruning.
- **Heterogeneous Speculative Decoding**: Draft-target speculative verification engine paired with O(1) KV-cache rollback (`truncate`).
- **Reasoning Budget Enforcement**: Test-time compute scaling enforcing thinking token limits via `</think>` token injection and logit masking.
- **Attention Sinks & Infinite Streaming Context**: StreamingLLM attention sinks (`SinkWindowCache`, `sink_causal_mask`) with cache-relative RoPE for infinite context generation without OOM.
- **Continuous Stream Journal**: Bounded FIFO event journal with pinned attention sinks ensuring strict memory ceilings under continuous log streams.
- **Dynamic Model Lifecycle Management**: On-demand model loading, hardware backend selection, and clean memory eviction.
- **Fast Token Generation**: Strict context window checking, token budgeting, and execution metrics.
- **Normalized Vector Embeddings**: 128-dimensional L2-normalized vector generation for semantic log and incident retrieval.
- **Zero Dynamic C Dependencies** (CPU build): Directly interfaces with Linux syscalls without `libsystemd.so` or `libdbus-1.so`. The optional `cuda` feature links the NVIDIA driver libraries only.
- **Vectorized Spatial Patch Pooling**: Zero-panic strided reshape + mean reduction patch pooling with symmetric edge padding supporting arbitrary 2x2, 3x3, and 4x4 visual token grids.
- **Ephemeral Vision Tower Lifecycle**: Dynamic weight migration pinning ViT encoders to accelerator VRAM during prefill and evicting to host RAM post-prefill, freeing 20%-30% VRAM prior to decoding.
- **Image Token Prefix Caching**: L2 host memory cache indexing visual KV blocks by `(model_id, prefix_hash, image_sha256)` with zero-copy KV splicing into Gemma4 sessions for multi-turn chats.
- **Real-Time Audio Out (Kokoro-82M TTS)**: Streaming 24kHz S16LE PCM audio synthesis with native PipeWire (`pw-cat`) integration preserving `$XDG_RUNTIME_DIR` and `$PIPEWIRE_RUNTIME_DIR`.
- **Generative Visual Output (SD-Turbo / LCM)**: 1-step generative sampler rendering PNG output into immutably sealed memfd buffers with compute lease gating.
- **Systemd Hardening & Device Isolation**: Sandboxed systemd service unit with scoped `char-nvidia*` / `char-drm` / `char-accel` device permissions.

---

## Install

Fastest: one line installs the whole suite plus a fitting brain (Gemma 4
E2B on machines with 30+ GB RAM, Qwen 0.5B elsewhere — picked automatically):

```bash
curl -fsSL https://syntropd.github.io/install.sh | sudo bash
```

Extra options go after `-s --`, e.g. `| sudo bash -s -- --with-vision`
or `--no-models` for the engines without brains.

Two manual runtimed-only routes: from source (needs a Rust toolchain
and `curl`), or from the release bundle (needs only `curl`).

From source:

```bash
git clone https://github.com/syntropd/runtimed.git
cd runtimed
sudo bash install/install.sh
```

From the release bundle (no Rust needed):

```bash
curl -fSL -o runtimed.tar.gz \
  https://github.com/syntropd/runtimed/releases/download/v0.5.3/runtimed-v0.5.3-x86_64-unknown-linux-gnu.tar.gz
tar xzf runtimed.tar.gz
cd runtimed-v0.5.3
sudo bash install/install.sh --from-bundle bin
```

The installer builds, registers the socket-activated service, enrolls you
in the `syntrop` group, asks whether to download the recommended brain
(Gemma 4 E2B, 3.1 GB), then verifies the daemon answers. Re-running is
safe: finished steps are skipped.

Common variants:

```bash
sudo bash install/install.sh --with-gemma --with-vision   # brain + picture questions
sudo bash install/install.sh --with-starter-model         # tiny Qwen for weak machines
sudo bash install/install.sh --cuda-gpu 0 --with-gemma    # NVIDIA build + card 0
sudo bash install/install.sh --yes                        # non-interactive, recommended brain
```

After installing, log out and back in (group membership), then:

```bash
runtimectl generate -m gemma-4-E2B-it-Q4_K_M "Say hello in one sentence."
```

Models live in `/var/lib/models/gguf/` (a flat `/var/lib/models/` tree
works too). Uninstall with `sudo bash install/uninstall.sh`; downloaded
models are kept.

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

### Environment

| Variable | Default | Effect |
|---|---|---|
| `RUNTIMED_BACKEND` | CPU | `cpu`, `cuda`, or `cuda:N` compute device for loads. |
| `RUNTIMED_IDLE_UNLOAD_SECS` | `0` (off) | Seconds of engine quiet before resident models unload (leases released, memory freed). Keeps warm by default. |

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

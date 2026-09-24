# runtimed: Architecture and System Design

`runtimed` is the Headless Model Execution and Tensor Generation Daemon in the Syntropd operating system architecture. It runs neural inference workloads under Linux systemd supervision without dynamic C dependencies.

## 1. Problem Statement

Executing large language models directly inside system supervisors or user shell environments presents major reliability and security issues:
1. **Dynamic Library Bloat**: Monolithic runtimes link multi-gigabyte shared libraries (`libcudart.so`, `libtorch.so`), causing ABI conflicts.
2. **Resource Starvation**: Unbounded model loading exhausts GPU VRAM and host RAM, triggering the kernel OOM-killer.
3. **Execution Latency**: Blocking supervisor loops during token generation degrades system responsiveness.

`runtimed` solves these challenges by acting as a dedicated, unprivileged compute worker managed via systemd socket activation and Varlink IPC (`io.syntrop.Runtime1`).

## 2. Core Architecture

```
                  +-----------------------------------+
                  |             runtimed              |
                  |                                   |
  Varlink IPC --> | [ModelManager] (Active Weights)   |
                  | [TokenGenerator] (Inference Loop) |
                  | [VectorEmbedder] (L2 Similarity)  |
                  +-----------------+-----------------+
                                    |
                                    v
                    /run/syntrop/io.syntrop.Runtime1
                                    |
          +-------------------------+-------------------------+
          |                                                   |
          v                                                   v
       sentry                                             runtimectl
 (Incident Triage)                                   (Admin & Debug CLI)
```

## 3. Subsystem Overview

### 3.1 Model Lifecycle Management (`runtimed-core::model`)
- **Active Model Registry**: Tracks resident models in compute memory (RAM or VRAM).
- **Dynamic Eviction**: `UnloadModel` reclaims memory upon command or hardware preemption signals from `inferenced`.
- **Zero-Copy Weight Access**: Interfaces with `/var/lib/models` maintained by `modeld`.

### 3.2 Token Generation Engine (`runtimed-core::engine::generator`)
- **Context Limit Checking**: Validates prompt token length against model context window before execution.
- **Budget Control**: Restricts completion length to prevent unbounded loops.
- **Execution Diagnostics**: Returns prompt token count, completion token count, and duration in milliseconds.

### 3.3 Vector Embedding Engine (`runtimed-core::engine::embedder`)
- Computes deterministic 128-dimensional normalized embedding vectors.
- L2-normalized so dot products directly compute cosine similarity.
- Enables semantic retrieval across `contextd` historical journals.

### 3.4 Varlink IPC Server (`runtimed-daemon::varlink`)
- High-performance NUL-terminated JSON over Unix domain sockets.
- Conforms to standard `org.varlink.service` introspection.
- Implements `io.syntrop.Runtime1` (`Generate`, `Embed`, `GetModelStatus`, `UnloadModel`, `ListLoadedModels`).

## 4. Hardware and Sandboxing Constraints

- **Dynamic Dependencies**: Zero (`libc` and `rustix` syscall bindings only; no `libsystemd.so` or `libdbus-1.so`).
- **Memory Containment**: Systemd limits via `MemoryHigh=8G` and `MemoryMax=16G`.
- **Device Access**: Scoped device permissions via `DeviceAllow=/dev/dri/renderD*` and `DeviceAllow=/dev/accel/*`.

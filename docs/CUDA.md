# CUDA Acceleration (Phase 6)

The default build is portable CPU-only. The `cuda` feature enables NVIDIA
GPUs via candle-core/cudarc (no `target-cpu=native` anywhere; kernels are
JIT-compiled by NVRTC for the detected card, so the binary stays portable).

## Build

Prerequisites: CUDA toolkit with `nvcc` on `PATH` (`/usr/local/cuda`).

```sh
# Full toolkit (cudart + cublas + curand):
RUNTIMED_CUDA=1 sudo ./install/install.sh

# Partial toolkit (cudart only): extract cublas/curand/nvrtc from the CUDA
# RPMs into a directory and point at it (ringmaster: /usr/local/cuda-libs):
CUDA_LIB_DIR=/usr/local/cuda-libs RUNTIMED_CUDA=1 sudo ./install/install.sh
```

Manual equivalent:

```sh
export PATH=/usr/local/cuda/bin:$PATH CUDA_ROOT=/usr/local/cuda
RUSTFLAGS="-L $HOME/cuda-libs -C link-args=-Wl,-rpath,$HOME/cuda-libs" \
  cargo build --release -p syntrop-runtimed --features cuda
```

Without `--features cuda`, any `cuda` backend request fails honestly with
"cuda backend needs a --features cuda build" (no silent CPU fallback).

## Run

One daemon serves one device, chosen by environment (fleet pattern: one
service instance per GPU):

```ini
# /etc/systemd/system/runtimed.service.d/cuda.conf
[Service]
Environment=RUNTIMED_BACKEND=cuda:0
Environment=LD_LIBRARY_PATH=/usr/local/cuda-libs
```

`RUNTIMED_BACKEND` accepts `cpu`, `cuda`, or `cuda:N`. `LD_LIBRARY_PATH`
is belt-and-suspenders over the baked RUNPATH (the service runs with
`ProtectHome=true`, so home-directory library paths are invisible to it).

Verify: `runtimectl status <model>` shows `Backend: cuda`, and `nvidia-smi`
shows the resident footprint.

## Memory model

Weights are resident **F16 on CUDA** (F32 on CPU, where f16 matmul is
unsupported) and upcast to F32 for every op, so compute stays F32-exact:
the `cuda_storage_is_f16_with_matching_greedy` test pins CPU/GPU greedy
parity. Rule of thumb: **2 bytes per parameter** plus ~1 GB of CUDA
context/cuBLAS overhead per process. A 4.6B-parameter model holds ~9 GB.

`status`/`load` report true resident bytes (element count times resident
dtype width), plus a 64 MiB allowance for tokenizer and runtime state.

## Sandboxing notes (systemd)

- `DeviceAllow` flips `DevicePolicy=auto` from allow-all to allow-listed,
  and **path globs are not supported** — only `char-<group>` with group
  globs. The unit lists `char-nvidia*`, `char-drm`, `char-accel`.
  A missing family surfaces as `CUDA_ERROR_NO_DEVICE`.
- `MemoryDenyWriteExecute=false` is required: NVRTC JIT-compiles kernels
  at runtime (this is the standing justification for review item 3.8).
- No `RuntimeDirectory`: `/run/syntrop` is owned by the socket unit and
  shared with the whole syntrop-sockets.target fleet; a service-level
  `RuntimeDirectory` would wipe every fleet socket file on restart.

## Operations

- One GPU can serve only what fits: a second loader gets CUDA OOM, not a
  queue. Coordinate large models across the two cards with the fleet
  router (`runtimectl load` reports free slots and resident bytes).
- Unrelated GPU tenants (e.g. ollama/llama-server on ringmaster) share
  the same VRAM pool; whoever loads first wins. Check `nvidia-smi` before
  blaming the daemon for OOMs.
- First request on a fresh worker thread binds the CUDA context
  explicitly (`Weights::ensure_current`); without it, pooled blocking
  threads fail intermittently with `CUDA_ERROR_INVALID_CONTEXT`.

## Speed (ringmaster, RTX 4060 Ti, Gemma-4 E2B Q4_K_M)

- Load: ~20 s (dequant + F16 upload of 4.6B params).
- Decode: ~20 tok/s steady state (F32 compute, per-token output-head cast).
- Known costs, not yet optimized: the full-vocabulary head is cast
  F16→F32 per token (~2.3 GB traffic), and the KV cache grows by
  re-concatenation (O(n²) over long contexts).

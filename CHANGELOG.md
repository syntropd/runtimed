# Changelog

## 0.6.0 (2026-10-04) — Ada Lovelace FP8 Tensor Cores, Marlin GEMV & Quantized Paged KV-Cache

- **In-VRAM 4-Bit Marlin GEMV**: Repacked 128-bit Marlin INT4 tiles and custom PTX kernel execution (`marlin_gemv_kernel`) yielding high-speed single-batch vector matrix multiplies.
- **Ada Lovelace FP8 Tensor Core Pipeline**: Native F8E4M3 matrix multiplication with per-tensor and per-channel scaling factors.
- **Quantized Paged KV-Cache**: 16-token page tables with dynamic FP8 / INT8 quantization per head, reducing KV cache resident footprint by up to 50% while preserving accuracy.
- **Dual-GPU Speculative Offload**: Transparent gang-scheduled speculative decoding running draft models on `cuda:1` and primary target models on `cuda:0` with automated fallback.

## 0.5.14 (2026-10-01) — Speculative Decoding Daemon Wiring & Environmental Sensory Awareness

- **Speculative Draft Model Wiring**: Exposed `speculative_draft_model: ?string` in `io.syntrop.Runtime1.Generate` Varlink interface. In `handle_generate`, dynamically load the draft model into resident memory and call `generate_speculative(&entry, &draft, &req, k_draft)` for accelerated token production.
- **io.syntrop.Sensory1 Varlink Interface**: Exposed `/run/syntrop/io.syntrop.Sensory1` with full Varlink introspection and methods `CaptureAudio`, `CaptureFrame`, `CaptureScreen`, and `GetOperatorPresence`.
- **Sensory Ingest Pipelines**:
  - `CaptureAudio`: PipeWire / pure-Rust PCM audio ingest with RMS dB energy threshold Voice Activity Detection (VAD).
  - `CaptureFrame`: Linux V4L2 RGB24 webcam capture with silhouette presence heuristic, integrating `pool_patches` spatial pooling and `ImageCache`.
  - `CaptureScreen`: Display desktop screen buffer inspection with Wayland/KMS dumb buffer detection and fallback.
  - `GetOperatorPresence`: Fused microphone audio VAD energy and webcam frame silhouette variance into composite operator presence probability and confidence estimation.

## 0.5.7 (2026-09-30) — Async Audio Streaming, Atomic Visuals & Neural Weights

- **Async Non-blocking Audio I/O**: Refactored `PcmSink` to async trait methods (`write_pcm`, `write_bytes`, `flush`) and `PwCatSink` to `tokio::process::Command` with `kill_on_drop(true)` and non-blocking asynchronous `finish()` drain. `KokoroEngine::synthesize` is generic async over `<S: PcmSink + ?Sized>`.
- **Closed-FD Defect Fix in GenerateVisual**: Replaced ephemeral memfd return with robust host runtime directory resolution (`$XDG_RUNTIME_DIR` -> `/run/user/<uid>` -> `temp_dir`), atomically rendering PNG images to `$RUNTIME_DIR/syntrop/visual_gen/{id}.png` and returning `image_path` in Varlink reply.
- **Multimodal Soft Token ImageCache Wiring**: Extended `ImageCache` to store projected soft tokens and token counts in host RAM (CPU). `generate_mm_tokens` queries cache by `(model_name, 0, image_bytes)` to bypass ViT lock and forward encoding on cache hit; `attach_vision` clears cache upon tower replacement.
- **Neural Model Weights Integration**: Added `turbo_unet.rs` for SD-Turbo diffusion latent forward passes and `acoustic_net.rs` for Kokoro TTS acoustic forward passes, binding model weights to `VisualGenSampler` and `KokoroEngine` with procedural fallback for offline operation. Passed `model_manager` into multimedia handlers in `runtime1.rs`.

## 0.5.6 (2026-09-30) — Multimedia Memory & Pipeline

- **Vectorized Spatial Patch Pooling**: Zero-panic strided reshape + mean reduction patch pooling with symmetric edge replication padding supporting arbitrary 2x2, 3x3, and 4x4 visual token grids.
- **Ephemeral Vision Tower Lifecycle**: ViT weights dynamically pin to accelerator VRAM during prefill and evict to host RAM post-prefill, freeing 20%-30% VRAM prior to decoding.
- **Image Token Prefix Caching**: L2 host memory cache indexing visual KV blocks by `(model_id, prefix_hash, image_sha256)` with zero-copy KV splicing into Gemma4 sessions for multi-turn chats.
- **Real-Time Audio Out (Kokoro-82M TTS)**: Streaming 24kHz S16LE PCM speech directly to PipeWire (`pw-cat`) via `io.syntrop.Runtime1.StreamAudioOut`.
- **Generative Visual Output (SD-Turbo / LCM)**: 1-step generative sampler rendering PNG output into immutably sealed memfd buffers with compute lease gating via `io.syntrop.Runtime1.GenerateVisual`.

## Unreleased

- **Gemma4 text prompts templated**: plain `Generate` calls on GGUF-BPE
  (Gemma4) models now wrap the prompt in the chat template (previously
  vision-only). Raw prompts made the model end the turn immediately,
  returning one empty completion token.
- **One installer**: the runtimed-only one-liner is gone, merged into the
  fleet installer (`https://syntropd.github.io/install.sh`), which now
  ships the Gemma brain by default (syntropd 0.3.9).
- **runtimectl on crates.io**: the CLI crate is published now, so the
  fleet installer's crates.io fallback can provide it.
- **tmpfiles**: model dirs match fleet permissions (0775 root:syntrop).
- **RAM honesty**: the 5B engine needs ~21 GB F32 on CPU, so the
  standalone installer offers Gemma only on 30+ GB machines (12+ GB
  VRAM with `--cuda-gpu`) and Qwen elsewhere; explicit `--with-gemma`
  always wins with a warning. Service memory ceiling raised to
  24G/32G to match (fleet units mirrored).

## 0.5.0 (2026-09-27) — Install process: models, groups, self-verify

The installer now delivers a working brain, not an empty engine.

- **Model provisioning**: `install.sh` fetches verified models into
  `/var/lib/models/gguf/` — `--with-gemma` (Gemma 4 E2B Q4, asked
  interactively by default), `--with-starter-model` (Qwen 0.5B Q8 plus
  its sidecar tokenizer), `--with-vision` (mmproj-F16). All URLs
  verified live; downloads are atomic and resumable by re-running.
- **User enrollment**: the installer adds the invoking user to the
  `syntrop` group (the step the CLI needs to reach the socket) and
  says when a fresh login is required.
- **One-flag CUDA**: `--cuda-gpu N` builds with CUDA and writes the
  backend drop-in automatically; no more manual unit surgery.
- **Self-verify**: every install ends by proving the socket is live
  and the daemon answers, then prints the exact first command to run.
- **Engine**: model/adapter/projector names now also resolve under
  the `gguf/` subdir (fleet layout shared with modeld), and
  `registry.toml` is honored there too (new `model::resolve` module,
  7 tests). No config file needed on any layout.
- **Uninstaller**: also removes the installer-owned `cuda.conf`
  drop-in so a later CPU reinstall starts clean; models still kept.
- **Docs**: README gained a real Install section with common variants.
- **Bundle**: the release tarball is now self-installing (binaries plus
  units, sysusers/tmpfiles confs, and the installer with `--from-bundle`
  mode), so machines without Rust install straight from the GitHub
  release page. Engine unchanged from the v0.5.0 tag.

## 0.4.0 (2026-09-27) — Owned engine, CUDA, release hardening

First release of the owned Rust inference engine (no llama.cpp, no cloud).

- **Engine**: GGUF loader with scalar dequantization (F32/F16/BF16, Q8_0,
  Q4_K, Q5_K, Q6_K), GGUF-BPE tokenizer, Gemma4 + Qwen2 forward passes,
  greedy/temperature/top-k/top-p sampler. Differential proofs vs recorded
  oracles gate merges (R4).
- **Multimodal**: SigLIP-style vision tower with Pillow-exact bicubic
  preprocessing, mmproj attach, image-conditioned generation
  (97.7% oracle-exact, 21 certified ties under the mutual top-8 rule).
- **LoRA**: file validation plus live fusion into resident weights.
- **Fleet**: `GetLoad` (free slots, resident bytes) and per-request
  concurrency cap for router-directed multi-GPU/multi-server use.
- **CUDA** (optional `--features cuda`): one device per daemon via
  `RUNTIMED_BACKEND`, F16-resident/F32-exact weights, explicit per-thread
  context binding, true resident-byte reporting. See `docs/CUDA.md`.
- **Trust (threat-model rules)**: R1 proven by clean offline rebuild;
  R2 weight registry (`registry.toml` SHA256 pins) enforced at load,
  attach-vision, and attach-lora (fail-open when absent, logged);
  weights excluded from git.
- **Hardening**: all stub-era review findings (docs/REVIEW_*) verified
  fixed; module size rule (≤256 lines) enforced by splitting bpe,
  dequant, ops, vpre, and the model meta types; daemon audit logging
  (load/unload/generate/attach, counts only) with INFO default and
  `RUST_LOG` override.
- **Service**: runs as `syntrop-runtime` under a sandboxed unit with
  correct `char-*` device rules, watchdog, and graceful shutdown;
  installer supports `RUNTIMED_CUDA=1` with partial-toolkit `CUDA_LIB_DIR`.

## 0.3.0 and earlier

Scaffold phases: threat model (Phase 0), GGUF loader + eval harness
(Phase 1), stub daemon/CLI with Varlink interface (pre-engine).

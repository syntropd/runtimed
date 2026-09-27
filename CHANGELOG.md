# Changelog

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

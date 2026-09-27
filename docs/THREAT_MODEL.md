# Engine Threat Model & Trust Policy (Phase 0)

Native LLM engine for runtimed. Goal: every byte accountable — auditable
source, pinned dependencies, pinned weights, reproducible builds.

## Assets

1. Engine source (this repo): loader, arch implementations, sampler, trainer.
2. Weight files (GGUF + mmproj sidecars) and LoRA adapters in modeld.
3. Eval corpora + reference outputs (differential oracles).
4. Runtime state: loaded models, KV caches, request contents (may be private).

## Attacker model (in scope)

- Compromised upstream dependency (malicious crate release, typosquat).
- Poisoned weight file (tampered mirror, evil quant).
- Tampered build inputs (unpinned lockfile, network fetch at build time).
- Cross-tenant leaks between concurrent requests (KV-cache, batching).
- Fleet attacks: rogue peer serving poisoned results, eavesdropped prompts.

Out of scope (accepted, documented): compiler/CPU-microcode trust,
physical access, side-channel key extraction, HF/Ollama account security.

## Rules (binding on every phase)

- R1 Vendored deps, pinned `Cargo.lock`. No build script touches the network;
  proven by offline builds. New deps need a stated reason in the commit.
- R2 Weight registry (`URL → SHA256`); the loader refuses mismatches, and
  refuses to run what it cannot hash. No weights in git, ever.
- R3 Fixed seed → byte-identical output (determinism gate from Phase 2).
  Parallelism only behind a flag, with equivalence tests.
- R4 Token-for-token differential tests vs a recorded oracle gate merges.
- R5 Portable x86-64 baseline, thin LTO + strip; SIMD via runtime
  detection with scalar fallback. No `target-cpu=native`, ever.
- R6 Fleet: authenticated transport, lease-first execution (no anonymous
  remote run), local fallback when a peer is unreachable or distrusted.

## Trust boundaries

```
weights (untrusted bytes) → [registry hash gate] → loader → engine
user prompt (private)     → [router] → runtimed → engine (never leaves machine except via R6 fleet path)
upstream crates           → [vendor + pin + audit note] → build
```

## openOODA port boundary

Logic modules (`gguf` parse, arch math description, sampler policy, eval
gates) must contain: no async-in-core, no macro-generated public API, no
`unsafe`, no direct candle imports. All tensor/substrate calls live behind
a `Port` trait in one substrate module per crate. A module violating this
fails review — the `.oo` port must stay mechanical.

## Hardware inventory (recorded, not assumed)

- ringmaster (server): i5-10400 6c/12t, 46 GB RAM, 2× RTX 4060 Ti 16 GB
  (CUDA UMD 13.4, driver 615.71), Intel UHD 630, 361 GB free disk.
  Role: build + train + heavy inference.
- blackbird (laptop): PENDING — run the probe below.
- desktop: PENDING — run the probe below.

Probe (run on each machine, paste output into this doc):

```sh
lscpu | grep -E "Model name|CPU\(s\)"; free -g | head -2
lspci | grep -iE "vga|3d|display"; command -v nvidia-smi && nvidia-smi -L
```

## Tier policy (initial, engine self-selects at runtime)

- Server-class (≥32 GB RAM and/or NVIDIA): E2B Q8/Q4, full context.
- Laptop-class (CPU or small VRAM): tiny model or E2B QAT, small context.
- Tiers are discovered, never configured. Unknown hardware → smallest tier.

## Kill criteria (standing)

- A phase stuck >2 sessions falls back (smaller scope), never balloons.
- A differential mismatch with no isolated cause within 1 session → halt
  the phase, record, ask. Silent numerical drift is never accepted.

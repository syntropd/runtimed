# syntropd: House Laws & Agent Engineering Standards (v1)

This document is the **single canonical source of truth** for all code, architecture, and system integration standards across the `syntropd` organization. Every human contributor and AI agent must strictly follow these rules without exception.

---

## 1. The Page Rule (Code Layout & Sizing)

A **page** is one committed Rust file. Every page holds one idea, fits in one head, and carries its own weight. This rule is enforced by an automated integration test (`page_rule`) under `cargo test`: red pages fail the build.

### Hard Sizing Invariants
- **16–256 Lines**: Every committed `.rs` file must be between **16 and 256 lines**, counted as `content.lines().count()` (blank lines and comments count).
- **Shim Exemption (Floor Only)**: A file is a shim when every code line is a module declaration or re-export: after dropping blank lines, comments (`//`, `//!`, `///`), and attributes (`#[...]`), each remaining line must start with `mod `, `use `, or `pub` (covers `pub mod` / `pub use`). Shims skip the 16-line floor. The 256-line ceiling still strictly applies.
- **Directory Density ($\le 8$ files)**: At most **8 `.rs` files per directory**, tests included. Crowded directories must split into functional subdirectories grouped by area.
- **Banned File Names (Name the function, not the drawer)**:
  `util.rs`, `utils.rs`, `helper.rs`, `helpers.rs`, `common.rs`, `misc.rs`, `shared.rs`, `base.rs`, `core.rs`.

### Splitting, Folding, and Naming
- **Over 256 lines**: Split along functional boundaries into a new subdirectory with a shim `mod.rs`. One page = one verb or one wholly owned noun.
- **Under 16 lines (and not a shim)**: Fold into its closest sibling or absorb into its single caller/callee. Never pad lines to reach 16.
- **Blobs Out**: Test fixtures and static data must live in dedicated files loaded via `include_str!` or `include_bytes!`, never inline literals.
- **Systems & Trust Naming**:
  - The tree mirrors the running system: paths read as subsystems (`router/`, `engine/`, `socket/`, `sandbox/`).
  - Action pages lead with a verb: `verify_peer.rs`, `admit_request.rs`, `seal_weights.rs`.
  - State pages name what they own: `lease_table.rs`.
  - Boundary pages speak trust verbs: `verify`, `admit`, `attest`, `seal`, `enforce`, `audit`.

### Scorecard & Performance Rules (Per-Page Performance)
`qa/page_score.sh` generates `page-score.json` tracking four performance metrics per page:
- **`lines`**: Source lines (exact).
- **`bin_bytes`**: Shipped binary bytes attributed to the page from workspace rlibs (`nm` + debuginfo). Shims carry zero bytes.
- **`heat_pct`**: Share of CPU execution samples under QA workloads (sampler + debuginfo).
- **`reach`**: Other pages referencing it (blast radius, qualified by ancestor directory to prevent stem clashes).
- **The Performance Invariant**: **A change may not grow a page's binary weight (`bin_bytes`) or CPU heat (`heat_pct`) without a written justification in the commit message.**
- **Pre-Release Validation**: Validate scorecard integrity with `qa/page_score_check.sh` before shipping (verifying full coverage, zero shim bytes, no misattributed foreign code $>50\text{ KB/line}$, and schema sanity).

---

## 2. Pure Rust & Crash-Resilience Standards

### The Zero-Panic Invariant
- **No Bare `.unwrap()` or `.expect()` in Production Code**: Long-running daemons must never panic on malformed inputs, socket disconnects, or unexpected journal lines.
- Always propagate errors via domain-typed `Result<T, E>` using `thiserror`.
- `.unwrap()` and `.expect()` are permitted **only** inside `#[cfg(test)]` modules.

### The `unsafe` Discipline
- **`#![deny(unsafe_code)]` by default** across all crate roots.
- When raw syscalls or FFI are strictly required (e.g. `rustix`, `libc`, `memfd`, `cgroup.freeze`), `#![allow(unsafe_code)]` is permitted **only on that specific page**.
- **Mandatory `// SAFETY:` Comment**: Every `unsafe { ... }` block must be immediately preceded by a comment documenting the exact pointer validity, lifetime invariant, and kernel error contract.

### Async Executor Protection (Never Block Tokio)
- **Zero Synchronous File I/O in Async Contexts**: Never call `std::fs` inside an `async fn`. Use `tokio::fs` or `tokio::task::spawn_blocking`.
- **Zero Blocking Locks Across `.await`**: Never hold a `std::sync::Mutex` or `std::sync::RwLock` across an `.await` boundary.
- **Cancellation Safety**: Background futures selecting on `tokio::select!` or timeouts must be cancellation-safe. Resource cleanup must be encapsulated in **RAII `Drop` guards**, not manual cleanup calls.

### Pure Rust Dependencies
- **Zero C Runtime Dependencies**: No dynamic linking to `libsystemd.so`, `libdbus-1.so`, or OpenSSL.
- Rely strictly on pure Rust crates: `zbus` (D-Bus), `rustls` (TLS), and `rustix` (Linux syscalls).

---

## 3. Native systemd Citizenship

`syntropd` daemons run as first-class system services under PID 1 (`systemd`). They must adhere to native systemd lifecycle and security primitives:

### Socket Activation (`$LISTEN_FDS`)
- Daemons must **never** call `bind()` unconditionally if `$LISTEN_FDS >= 1`.
- Always inspect inherited file descriptors starting at FD 3 (`LISTEN_FDS_START`). Native binding is strictly a local debug fallback.

### Lifecycle & Heartbeat Notifications (`sd_notify`)
- **Accurate `READY=1`**: Never emit `READY=1` until all sockets, descriptors, and channels are fully open and ready to process traffic.
- **Dynamic `WATCHDOG=1`**: Calculate watchdog heartbeats at half the interval of `$WATCHDOG_USEC`. Heartbeats must be gated on internal healthchecks, never a blind background loop.
- **Immediate `STOPPING=1`**: Emit `STOPPING=1` as the very first action upon intercepting `SIGTERM` or `SIGINT`.

### Standardized Exit Codes (`sysexits.h`)
- On fatal termination, return standard systemd exit codes:
  - `78` (`EX_CONFIG`): Configuration or policy validation failure. Systemd will mark the unit failed permanently without crash-looping.
  - `69` (`EX_UNAVAILABLE`): Missing required hardware or socket.
  - `0` (`EX_OK`): Clean shutdown on signal.

### Standardized System Directories
- Never hardcode absolute system paths in daemon code. Use standard systemd environment variables:
  - `$RUNTIME_DIRECTORY` (default: `/run/syntrop`)
  - `$STATE_DIRECTORY` (default: `/var/lib/syntrop`)
  - `$CONFIGURATION_DIRECTORY` (default: `/etc/syntrop`)
  - `$CREDENTIALS_DIRECTORY` (for secrets sealed via `systemd-creds`)

---

## 4. Linux Kernel & Zero-Trust Safety

### Non-Blocking Kernel Telemetry & PSI
- Never busy-loop reading `/proc/pressure/{memory,cpu,io}` or `/proc/self/statm`.
- Telemetry readers must use non-blocking `poll()`/`epoll` or fixed tick intervals, allocating zero heap memory on the stack (`[u8; 64]`).

### File Descriptor Discipline (`O_CLOEXEC`)
- Every socket, pipe, and file descriptor opened must set `O_CLOEXEC` / `SOCK_CLOEXEC`. Descriptors must never leak across `fork`/`exec` boundaries into child processes.

### Shared Memory Sealing (`memfd_create`)
- Weight transfers, large prompt buffers, and IPC blobs must use sealed memory file descriptors (`memfd_create` + `F_SEAL_SEAL | F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW`) transmitted over Unix sockets via `SCM_RIGHTS`. Never pass writable buffers across trust boundaries.

### Safe Subprocess Execution (No Shell Execution)
- In `toold`, **never** invoke `/bin/sh -c` or `/bin/bash -c`. Always execute discrete binary paths directly via `tokio::process::Command::new(binary_path)`.
- Environment variables must be cleared (`cmd.env_clear()`) before spawning child binaries, populating only safe keys (`PATH`, `LANG`, `LC_ALL`).

### Two-Tier cgroups v2 Preemption
- Compute preemption must follow a two-tier escalation:
  1. Cooperative `SIGUSR1` yield signal with a 250ms deadline.
  2. Escalation to cgroups v2 suspension (`cgroup.freeze = 1` or `SIGSTOP`).
- **Hard Safety Barrier**: Never target PID 1, system slices (`system.slice`), or protected root services (`dbus`, `systemd-journald`, `sshd`).

---

## 5. Verification & QA Gate

Before any change is committed or marked complete:
1. **Compilation**: Clean compilation with zero warnings under `cargo check --workspace`.
2. **Page Rule Compliance**: `cargo test --workspace` must pass, including the `page_rule` integration test. No exemptions, no padding.
3. **Structured Tracing**: No bare `println!` or `eprintln!` in daemons. All logging must use structured `tracing` macros with field keys.
4. **Performance Integrity**: Validate scorecard integrity via `qa/page_score_check.sh`. No growth in a page's binary weight or CPU heat without written justification.

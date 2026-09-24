# runtimed Review — Security Issues (§3)

## 3.1 Varlink server receive buffer unbounded → memory DoS

**File**: `crates/runtimed-daemon/src/varlink/server.rs:50-81`

```rust
let mut buffer = Vec::with_capacity(4096);
...
loop {
    let n = stream.read(&mut chunk).await?;
    buffer.extend_from_slice(&chunk[..n]);
    while let Some(pos) = buffer.iter().position(|&b| b == 0x00) {
        ...
    }
}
```

A client that connects and never sends NUL grows `buffer` without bound. Same class as the contextd/modeld pre-fix defects.

**Recommended fix**: cap at `MAX_MSG_BYTES = 1 MiB`; on overflow, write a `ProtocolError` reply and drop the connection. Make `handle_client` `pub` so QA can drive the overflow path directly.

## 3.2 `runtimectl` client reply buffer unbounded → memory DoS

**File**: `crates/runtimectl/src/client.rs:50-71`

```rust
let mut buffer = Vec::with_capacity(4096);
...
loop {
    let n = stream.read(&mut chunk).await?;
    buffer.extend_from_slice(&chunk[..n]);
    if let Some(pos) = buffer.iter().position(|&b| b == 0x00) { ... }
}
```

A malicious or buggy server that never sends NUL grows `buffer` without bound.

**Recommended fix**: cap at the same `MAX_MSG_BYTES = 1 MiB` and return `Err(anyhow!("Varlink reply exceeded ... bytes"))` on overflow.

## 3.3 Socket-file TOCTOU race in standalone bind path

**File**: `crates/runtimed-daemon/src/main.rs:63-69`

```rust
if socket_path.exists() {
    let _ = fs::remove_file(&socket_path);
}
...
let std_listener = StdUnixListener::bind(&socket_path)?;
std_listener.set_nonblocking(true)?;
UnixListener::from_std(std_listener)?
```

`exists()` → `remove_file()` → `bind()` is a TOCTOU window. A privileged process could replace the path with a symlink between `exists` and `remove_file`, causing the daemon to delete an arbitrary file.

**Recommended fix**: drop the `exists`/`remove` block; let `bind()` fail if the path exists. After bind, `chmod(0o660)`.

## 3.4 No chmod on standalone-bound socket

**File**: `crates/runtimed-daemon/src/main.rs:67-69`

The bound socket inherits umask. With umask `022`, the socket is world-readable/writable. systemd-managed activation gets correct `SocketMode=0660` via the unit file, but the standalone path is unprotected.

**Recommended fix**: `fs::set_permissions(socket_path, fs::Permissions::from_mode(0o660))?` after bind. On chmod failure, drop the listener and remove the socket inode before returning (so a retry doesn't hit `EADDRINUSE` with the wrong mode).

## 3.5 `runtimed` Varlink server has no peer-credential check

**File**: `crates/runtimed-daemon/src/varlink/server.rs:32-44`

The systemd `.socket` unit sets `SocketGroup=syntrop` and `SocketMode=0660` so only members of the `syntrop` group can connect. But there is no in-process `SO_PEERCRED` check. If the socket is ever exposed beyond the intended trust boundary, anyone able to connect can generate embeddings or unload models.

`runtimed.socket:12` sets `PassCredentials=yes`, suggesting the design intended peer-cred gating — the in-process check is simply missing.

**Recommended fix**: read `SO_PEERCRED` via rustix on each accept (rustix's `get_socket_peercred` is a safe binding). Trust uid 0 OR members of a configurable group (default `syntrop`).

## 3.6 `WatchdogSec=30s` set in unit, daemon never pings

**File**: `systemd/runtimed.service:12`

`WatchdogSec=30s` advertises that systemd expects `WATCHDOG=1` every ~10s. The daemon's main loop has no watchdog task. systemd will SIGABRT the daemon after 30 s of silence. Same critical defect as contextd/modeld pre-fix.

**Recommended fix**: spawn a watchdog task that reads `$WATCHDOG_USEC`, pings at `usec / 3` (clamped to `[1s, 300s]`, `Duration::ZERO` for `WATCHDOG_USEC=0`). Honor shutdown signal.

## 3.7 Service unit has no `User=` / `Group=` and `ReadOnlyPaths=/var/lib/models`

**File**: `systemd/runtimed.service:7-30`

The unit has no `User=` / `Group=` directive — the daemon runs as **root**. The sysusers.d config creates `syntrop-runtime` but the unit does not use it.

More critically: `ReadOnlyPaths=/var/lib/models` means the daemon cannot write to its model cache — the directory exists and is listed as a system sandbox. But the daemon's pre-warm path does `fs::create_dir_all(&config.models_dir)` (line 33), which requires write access. Under sandbox, that fails silently (`warn!`), and any subsequent model write fails.

**Recommended fix**: add `User=syntrop-runtime`, `Group=syntrop`. Change `ReadOnlyPaths=/var/lib/models` to `ReadWritePaths=/var/lib/models /run/syntrop`. Add `RuntimeDirectory=syntrop`.

## 3.8 `MemoryDenyWriteExecute=false`

**File**: `systemd/runtimed.service:21`

`MemoryDenyWriteExecute` is explicitly disabled (default for service units is `true`). The daemon can map executable memory — needed by `libtorch` / `candle`-style inference engines. Documented as "MemoryDenyWriteExecute=false" in the unit file with no rationale. Acceptable if the daemon later loads a JIT; flag for operator review.

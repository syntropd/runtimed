# runtimed Review — Audit, Summary, Next Steps (§5–9)

## 5. Module-by-Module Test Coverage Gaps

| Module | Unit tests | Edge tests | Gaps |
| --- | --- | --- | --- |
| `error.rs` | ❌ 0 | — | Trivial; enum-only. |
| `config/runtimed_config.rs` | ✅ 2 | — | None. |
| `model/loader.rs` | ⚠️ 1 | — | `load_model` idempotency, `get_model` after unload, `list_active` ordering untested. |
| `engine/embedder.rs` | ⚠️ 3 | — | Determinism under Unicode input, empty-after-trim handling tested; no test for cosine on vectors with NaN. |
| `engine/generator.rs` | ⚠️ 2 | — | `prompt_tokens > context_window` boundary, `max_tokens == 0` boundary untested. |
| `daemon/main.rs` | ❌ 0 | — | No integration test for socket bind + watchdog + shutdown. |
| `daemon/notify.rs` | ❌ 0 | — | **No unit tests for the new fix (after §2.2).** |
| `daemon/activation.rs` | ❌ 0 | — | `parse_listen_fds` combinations untested. |
| `daemon/varlink/protocol.rs` | ✅ 1 | — | `to_bytes` serialization failure path not exercised (will be, after fix). |
| `daemon/varlink/server.rs` | ❌ 0 | ❌ 0 | **Zero direct tests for the connection loop, framing, dispatch, or DoS surface (3.1).** |
| `daemon/varlink/runtime1.rs` | ✅ 3 | — | `handle_get_model_status` for missing model untested at unit level. |
| `daemon/varlink/service.rs` | ✅ 2 | — | None significant. |
| `runtimectl/client.rs` | ❌ 0 | — | **Buffer cap not tested (3.2).** |
| `cmd/*.rs` | ❌ 0 | — | Trivial; mostly pass-through. |
| `qa/edge/src/*.rs` | — | ✅ 7 | Decent coverage for corrupt store, large diffs, protocol edge, unload edge. |

## 6. File Length Audit (≤ 256 LOC rule)

**All files comply.** Largest files in each category:

| Category | File | LOC |
| --- | --- | --- |
| Production Rust | `varlink/runtime1.rs` | 162 |
| Production Rust | `varlink/server.rs` | 100 |
| Production Rust | `model/loader.rs` | 96 |
| Production Rust | `daemon/main.rs` | 98 |
| QA test | `qa/unit/src/varlink_tests.rs` | 114 |
| Doc | `docs/VARLINK_SPEC.md` | 80 |
| Doc | `README.md` | 76 |
| Doc | `docs/CLI_REFERENCE.md` | 81 |
| Config | `systemd/runtimed.service` | 41 |
| Config | `install/install.sh` | 42 |

No file approaches the 256 ceiling; ~35% headroom remains in the largest file.

## 7. Spec Violations Summary (action priority)

| # | Severity | Issue | File |
| - | -------- | ----- | ---- |
| 2.2 | **High** | `notify.rs` unsafe `sendto` with possibly wrong `sockaddr` size | `daemon/notify.rs:32-44` |
| 2.7 | **High** | `main.rs` shutdown race — server task cancelled mid-accept | `daemon/main.rs:81-93` |
| 3.1 | **High** | Varlink server receive buffer unbounded | `varlink/server.rs:50-81` |
| 3.2 | **High** | Client reply buffer unbounded | `runtimectl/client.rs:50-71` |
| 3.3 | **High** | Socket TOCTOU race | `daemon/main.rs:63-69` |
| 3.4 | **High** | No chmod on standalone socket | `daemon/main.rs:67-69` |
| 3.6 | **High** | `WatchdogSec=30s` set but daemon never pings | `systemd/runtimed.service:12` |
| 2.5 | Medium | `VarlinkReply::to_bytes` silent failure on serialization error | `varlink/protocol.rs:52-56` |
| 2.3 | Medium | `notify_status` doesn't sanitize embedded newlines | `daemon/notify.rs:53-55` |
| 3.5 | Medium | No peer-credential check on Varlink | `varlink/server.rs:32-44` |
| 3.7 | Medium | Service unit runs as root, `ReadOnlyPaths` blocks model writes | `systemd/runtimed.service:7-30` |
| 2.6 | Low | Client silently substitutes `Null` for missing parameters | `runtimectl/client.rs:69` |
| 2.4 | Low | `notify_send` no oversize message guard | `daemon/notify.rs:7-45` |
| 4.6 | Low | `max_concurrent_requests` not enforced | `config/runtimed_config.rs:29` |
| 4.4 | Low | `ModelManager` uses write mutex for reads | `model/loader.rs:67-90` |
| 4.8 | Low | Hand-rolled arg parsing instead of clap | `daemon/main.rs:22-28` |

**Rule failures** (counted against the spec itself): §7 15 MiB RSS / fixed-size buffers (issues 3.1, 3.2), §7 1:1 unit test coverage (`server`, `daemon/notify`, `daemon/activation`, `runtimectl/client`, `cmd/*`).

## 8. Notes (Not Defects)

- Code style is consistent across crates. Doc comments are short and direct.
- `RuntimedError` with `thiserror` is clean.
- `runtime1.rs` cleanly separates Varlink parameter parsing from business logic and returns precise `InvalidParameter` errors.
- The `unload_cmd` and `embed_cmd` modules correctly forward to the daemon via the client.
- `VarlinkServer::new` returning `Self` is fine; the listener is consumed by `run()`.

## 9. Recommended Next Steps (no code applied)

1. **Fix 2.2 first.** Replace the `unsafe sendto` block with `rustix::net::sendto_unix`. One-line change with major safety win.
2. **Fix 3.6 second.** Watchdog mismatch will SIGABRT the daemon within 30 s of every boot.
3. **Fix 3.1 + 3.2 third.** Varlink buffer DoS is exploitable by anyone with socket access.
4. Then address §7 High items in any order; each is a small change with a corresponding unit test.
5. Add the missing test coverage listed in §5 — each new test should fail against the current code, confirming it would have caught the defect.

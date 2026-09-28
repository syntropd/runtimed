# The Page Rule — syntropd house law (v1)

A **page** is one committed Rust file. Every page holds one idea, fits in
one head, and carries its own weight. This rule is enforced by a test:
red pages fail the build. No exemption lists. No padding.

## The sizes (hard)

- Every committed `.rs` file is **16–256 lines**, counted as
  `content.lines().count()` (blank lines and comments count).
- **Shim exemption from the floor only.** A file is a shim when every
  code line is a module declaration or re-export: after dropping blank
  lines, comment lines (`//`, `//!`, `///`), and attribute lines
  (`#[...]`), each remaining line must start with `mod `, `use `, or
  `pub` (which covers `pub mod` / `pub use`). Shims skip the 16-line
  floor. The 256 cap still applies. The check is mechanical and
  name-blind: a `mod.rs` with real logic is not a shim.
- **At most 8 `.rs` files per directory**, tests included. Crowded
  directories split into subdirectories grouped by area.
- **Banned file names** (name the function, not the drawer):
  `util.rs`, `utils.rs`, `helper.rs`, `helpers.rs`, `common.rs`,
  `misc.rs`, `shared.rs`, `base.rs`, `core.rs`.

## Splitting and folding (the procedures)

- **Over 256:** split along functional lines into a new subdirectory
  with a shim `mod.rs`. One page = one verb or one wholly owned noun.
- **Under 16 and not a shim:** fold into the closest sibling — the
  page in the same directory it serves most — or absorb its single
  caller/callee. Padding a file to reach 16 is a splitting failure;
  fold instead.
- **Blobs out:** fixtures and test data live as data files loaded
  with `include_str!`/`include_bytes!`, never as giant literals.
- **Pure Rust stays pure:** no new build scripts, no new codegen, no
  new non-Rust sources without a written reason. New crates need one
  too — a page is cheaper than a dependency.

## Naming (zero trust, first principles, systems thinking)

- The tree mirrors the running system. A page's path reads as its
  subsystem: `router/`, `engine/`, `socket/`, `sandbox/`. If you cannot
  tell where a page runs from its path, the path is wrong.
- Action pages lead with a verb: `verify_peer.rs`, `admit_request.rs`,
  `seal_weights.rs`. State pages name what they own: `lease_table.rs`.
- Boundary pages speak trust verbs: `verify`, `admit`, `attest`,
  `seal`, `enforce`, `audit`. The name must say what the page does
  about trust, because that is its job.

## Proofs (QA per page)

- Every non-shim page ships with tests: in-file `mod tests` for pure
  logic, `qa/` integration tests for anything touching sockets,
  processes, the filesystem, or time.
- Coverage is reported per page. Where `cargo-llvm-cov` is present it
  is the source of truth; otherwise the worker lists, per page, the
  test names that cover it, and a reviewer confirms the mapping.
- A page is done when it is sized, named, tested, and scored.

## Scorecard (performance per page)

`qa/page_score.sh` writes `page-score.json`. Four numbers per page:

| metric     | meaning                                              | v1 source                    |
|------------|------------------------------------------------------|------------------------------|
| `lines`    | source lines                                         | always (exact)               |
| `bin_bytes`| shipped binary bytes attributed to the page        | `nm` + debuginfo (approx)    |
| `heat_pct` | share of CPU samples under the QA workloads        | sampler + debuginfo (approx) |
| `reach`    | other pages referencing it (blast radius)          | grep (approx)                |

Approximate fields carry their method in the report's `notes`. A
snapshot ships per release; a change may not grow a page's weight or
heat without a written reason in the commit message.

## Enforcement

- `page_rule` runs as an integration test under plain
  `cargo test --workspace` and fails on any violation. It walks from
  the git root, skips `.git/` and `target/`, and follows no symlinks.
- The lint file itself is a page and must comply.
- New violations never land. Pre-existing violations are fixed by
  refactoring, never by carving exemptions.

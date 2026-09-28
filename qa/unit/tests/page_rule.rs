//! The Page Rule, enforced (see `PAGE_RULE.md`).
//!
//! Every committed `.rs` file is one page: 16–256 lines, at most 8
//! pages per directory, no drawer names (`util.rs`, `common.rs`).
//! Pure wiring shims skip the 16-line floor only. Failures list every
//! violation: fix by refactoring, never by exempting.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

const MIN_LINES: usize = 16;
const MAX_LINES: usize = 256;
const MAX_FILES_PER_DIR: usize = 8;
const BANNED: &[&str] = &[
    "util.rs", "utils.rs", "helper.rs", "helpers.rs", "common.rs", "misc.rs", "shared.rs",
    "base.rs", "core.rs",
];

/// Repository root: first ancestor of this package holding `.git`.
fn git_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join(".git").exists() {
            return dir;
        }
        if !dir.pop() {
            panic!("page rule: no .git above {}", env!("CARGO_MANIFEST_DIR"));
        }
    }
}

/// All committed Rust pages: skips `.git/`, `target/`, follows nothing.
fn collect_rs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let meta = match std::fs::symlink_metadata(&path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.file_type().is_symlink() {
                continue;
            }
            if meta.is_dir() {
                let name = entry.file_name();
                if name == ".git" || name == "target" {
                    continue;
                }
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// A shim wires modules and nothing else: every code line is a `mod`,
/// `use`, or `pub use` item. Mechanical and name-blind.
fn is_shim(content: &str) -> bool {
    let mut code = 0;
    for line in content.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") || t.starts_with('#') {
            continue;
        }
        code += 1;
        let wiring = t.starts_with("mod ")
            || t.starts_with("use ")
            || (t.starts_with("pub") && (t.contains("mod ") || t.contains("use ")));
        if !wiring {
            return false;
        }
    }
    code > 0
}

fn is_banned(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| BANNED.contains(&n))
}

#[test]
fn page_rule() {
    let root = git_root();
    let files = collect_rs(&root);
    assert!(!files.is_empty(), "page rule: no .rs files under {}", root.display());
    let mut violations: Vec<String> = Vec::new();
    let mut per_dir: HashMap<PathBuf, usize> = HashMap::new();
    for path in &files {
        let rel = path.strip_prefix(&root).unwrap_or(path);
        if let Some(parent) = rel.parent() {
            *per_dir.entry(parent.to_path_buf()).or_insert(0) += 1;
        }
        if is_banned(path) {
            violations.push(format!("{}: banned file name (name the function)", rel.display()));
        }
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                violations.push(format!("{}: unreadable ({e})", rel.display()));
                continue;
            }
        };
        let lines = content.lines().count();
        if lines > MAX_LINES {
            violations.push(format!("{rel}: {lines} lines (max {MAX_LINES}) — split along functional lines", rel = rel.display()));
        } else if lines < MIN_LINES && !is_shim(&content) {
            violations.push(format!("{rel}: {lines} lines (min {MIN_LINES}, not a shim) — fold into its sibling", rel = rel.display()));
        }
    }
    let mut dense: Vec<_> = per_dir.into_iter().filter(|(_, n)| *n > MAX_FILES_PER_DIR).collect();
    dense.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (dir, n) in dense {
        violations.push(format!("{}: {n} pages (max {MAX_FILES_PER_DIR} per dir) — group into subdirs", dir.display()));
    }
    for v in &violations {
        println!("page violation: {v}");
    }
    assert!(violations.is_empty(), "page rule: {} violation(s), see output above", violations.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shim_needs_wiring_only() {
        assert!(is_shim("pub mod a;\nmod b;\n"));
        assert!(is_shim("//! docs\nuse x::Y;\n#[cfg(test)]\nmod tests;\n"));
        assert!(is_shim("pub(crate) use x::Y;\n"));
        assert!(!is_shim("pub mod a;\nfn f() {}\n"));
        assert!(!is_shim("use x;\npub fn f() {}\n"));
        assert!(!is_shim(""));
        assert!(!is_shim("// only a comment\n"));
    }

    #[test]
    fn banned_names_match() {
        assert!(is_banned(Path::new("a/util.rs")));
        assert!(is_banned(Path::new("common.rs")));
        assert!(!is_banned(Path::new("a/utils_extra.rs")));
        assert!(!is_banned(Path::new("a/verify_peer.rs")));
    }
}

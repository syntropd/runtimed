#!/usr/bin/env python3
"""
prune_qa.py — Automated QA auditing utility to detect:
1. Production code items (functions, structs) with `pub` visibility that are ONLY called from `tests/`, `qa/`, or `#[cfg(test)]` (test-induced interface bloat).
2. Production code items with `pub` visibility that are completely uncalled anywhere (zombie / dead code).
3. QA / test helper functions (excluding `#[test]` / `#[tokio::test]`) never referenced outside their definition.

Usage:
    python3 qa/prune_qa.py [/path/to/repo]
"""

import os
import re
import sys
from pathlib import Path

PUB_FN_RE = re.compile(r'^\s*pub\s+(?:async\s+)?fn\s+([a-zA-Z0-9_]+)\s*(?:<[^>]+>)?\s*\(', re.MULTILINE)
PUB_STRUCT_RE = re.compile(r'^\s*pub\s+struct\s+([a-zA-Z0-9_]+)', re.MULTILINE)
FN_DEF_RE = re.compile(r'^\s*(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?fn\s+([a-zA-Z0-9_]+)\s*(?:<[^>]+>)?\s*\(', re.MULTILINE)

COMMON_SKIP_NAMES = {
    "new", "default", "clone", "fmt", "from", "into", "as_ref", "as_mut",
    "deref", "deref_mut", "drop", "builder", "build", "main", "run", "init"
}

TEST_ATTR_KEYWORDS = {"test", "tokio::test", "bench", "rstest", "case"}

def is_test_file(path_str: str) -> bool:
    parts = Path(path_str).parts
    return (
        "qa" in parts or
        "tests" in parts or
        path_str.endswith("_tests.rs") or
        path_str.endswith("_test.rs")
    )

def is_test_function(lines, fn_line_idx):
    """Checks whether a function definition is decorated with a test attribute."""
    for p_idx in range(fn_line_idx - 1, -1, -1):
        line = lines[p_idx].strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("#["):
            for kw in TEST_ATTR_KEYWORDS:
                if kw in line:
                    return True
        else:
            break
    return False

def find_test_blocks(lines):
    """Finds line indices (0-indexed) that fall inside #[cfg(test)] scopes."""
    test_lines = set()
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i].strip()
        if line.startswith("#[cfg(test)]"):
            j = i + 1
            while j < n and (not lines[j].strip() or lines[j].strip().startswith("//")):
                j += 1
            if j < n:
                target_line = lines[j].strip()
                if "{" in target_line:
                    brace_depth = target_line.count("{") - target_line.count("}")
                    for k in range(i, j + 1):
                        test_lines.add(k)
                    curr = j + 1
                    while curr < n and brace_depth > 0:
                        test_lines.add(curr)
                        brace_depth += lines[curr].count("{") - lines[curr].count("}")
                        curr += 1
                    i = curr
                    continue
                else:
                    test_lines.add(i)
                    test_lines.add(j)
                    i = j + 1
                    continue
        i += 1
    return test_lines

def audit_repo(repo_root: Path):
    repo_root = repo_root.resolve()
    print(f"=== Auditing Workspace: {repo_root.name} ({repo_root}) ===")

    rs_files = []
    for root, dirs, files in os.walk(repo_root):
        if "target" in dirs:
            dirs.remove("target")
        if ".git" in dirs:
            dirs.remove(".git")
        for f in files:
            if f.endswith(".rs"):
                rs_files.append(Path(root) / f)

    file_lines = {}
    test_line_sets = {}
    for f in rs_files:
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            file_lines[f] = lines
            if is_test_file(str(f)):
                test_line_sets[f] = set(range(len(lines)))
            else:
                test_line_sets[f] = find_test_blocks(lines)
        except Exception:
            pass

    test_only_prod = []
    dead_prod = []

    # 1. Audit production methods and structs
    for f, lines in file_lines.items():
        if is_test_file(str(f)):
            continue
        t_set = test_line_sets[f]
        for idx, line in enumerate(lines):
            if idx in t_set:
                continue
            m_fn = PUB_FN_RE.search(line)
            m_st = PUB_STRUCT_RE.search(line)
            name = None
            itype = None
            if m_fn:
                name = m_fn.group(1)
                itype = "fn"
            elif m_st:
                name = m_st.group(1)
                itype = "struct"

            if not name or name in COMMON_SKIP_NAMES or name.startswith("test_"):
                continue

            pat = re.compile(r'\b' + re.escape(name) + r'\b')
            prod_calls = 0
            test_calls = 0
            test_callers = set()

            for of, olines in file_lines.items():
                of_tset = test_line_sets[of]
                prod_text = "\n".join(l for oi, l in enumerate(olines) if oi not in of_tset)
                test_text = "\n".join(l for oi, l in enumerate(olines) if oi in of_tset)

                p_matches = len(pat.findall(prod_text))
                t_matches = len(pat.findall(test_text))

                if of == f:
                    p_matches -= 1
                if p_matches > 0:
                    prod_calls += p_matches
                if t_matches > 0:
                    test_calls += t_matches
                    test_callers.add(str(of.relative_to(repo_root)))

            if prod_calls == 0 and test_calls > 0:
                test_only_prod.append((itype, name, f.relative_to(repo_root), idx + 1, sorted(test_callers)))
            elif prod_calls == 0 and test_calls == 0:
                dead_prod.append((itype, name, f.relative_to(repo_root), idx + 1))

    # 2. Audit QA / test helpers (excluding actual test entry points)
    dead_qa_helpers = []
    for f, lines in file_lines.items():
        t_set = test_line_sets[f]
        for idx in t_set:
            if idx >= len(lines):
                continue
            m = FN_DEF_RE.search(lines[idx])
            if not m:
                continue
            name = m.group(1)
            if name in COMMON_SKIP_NAMES or name.startswith("test_") or is_test_function(lines, idx):
                continue

            pat = re.compile(r'\b' + re.escape(name) + r'\b')
            calls = 0
            for of, olines in file_lines.items():
                m_count = len(pat.findall("\n".join(olines)))
                if of == f:
                    m_count -= 1
                if m_count > 0:
                    calls += m_count

            if calls == 0:
                dead_qa_helpers.append((name, f.relative_to(repo_root), idx + 1))

    print(f"\n--- Flagged Test-Induced Interface Bloat (pub in prod, called ONLY in tests/qa) [{len(test_only_prod)}] ---")
    for itype, name, pfile, line_no, callers in test_only_prod:
        callers_str = ", ".join(callers)
        print(f"  [{itype.upper()}] {name} at {pfile}:{line_no}")
        print(f"      -> Only called from: {callers_str}")

    print(f"\n--- Flagged Uncalled / Dead Production Items (pub in prod, zero calls anywhere) [{len(dead_prod)}] ---")
    for itype, name, pfile, line_no in dead_prod:
        print(f"  [{itype.upper()}] {name} at {pfile}:{line_no}")

    print(f"\n--- Flagged Dead QA / Test Helpers (defined in test scope, never called) [{len(dead_qa_helpers)}] ---")
    for name, tfile, line_no in dead_qa_helpers:
        print(f"  [FN] {name} at {tfile}:{line_no} (Never called anywhere)")

    print(f"\nAudit completed. Summary: {len(test_only_prod)} test-only prod items, {len(dead_prod)} dead prod items, {len(dead_qa_helpers)} dead test helpers.\n")
    return test_only_prod, dead_prod, dead_qa_helpers

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    audit_repo(target)

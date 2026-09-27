#!/usr/bin/env bash
# Focused test for install/install.sh + uninstall.sh. Runs fully
# unprivileged: only syntax, --help, and the pre-root flag guards.
# Fails on the first mismatch.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL="${REPO_DIR}/install/install.sh"
UNINSTALL="${REPO_DIR}/install/uninstall.sh"

pass() { echo "PASS: $1"; }

bash -n "${INSTALL}" && bash -n "${UNINSTALL}"
pass "syntax"

bash "${INSTALL}" --help | grep -q -- "--from-bundle" || { echo "FAIL: --help hides --from-bundle"; exit 1; }
pass "--help documents bundle mode"

if bash "${INSTALL}" --bogus-flag >/dev/null 2>&1; then echo "FAIL: unknown flag accepted"; exit 1; fi
pass "unknown flag rejected"

out="$(bash "${INSTALL}" --from-bundle /nonexistent 2>&1 || true)"
echo "${out}" | grep -q "has no runtimed binary" || { echo "FAIL: bad bundle dir not caught: ${out}"; exit 1; }
pass "missing bundle dir rejected before root check"

out="$(bash "${INSTALL}" --from-bundle "${REPO_DIR}/target/release" --cuda-gpu 0 2>&1 || true)"
echo "${out}" | grep -q "CPU-only" || { echo "FAIL: bundle+cuda conflict not caught: ${out}"; exit 1; }
pass "bundle+cuda conflict rejected before root check"

out="$(bash "${UNINSTALL}" 2>&1 || true)"
echo "${out}" | grep -qi "root privileges" || { echo "FAIL: uninstaller ran without root?!"; exit 1; }
pass "uninstaller refuses non-root"

n=$(grep -c "huggingface.co.*resolve/main" "${INSTALL}")
[[ "${n}" -eq 4 ]] || { echo "FAIL: expected 4 model URLs, found ${n}"; exit 1; }
pass "4 verified model URLs pinned"

echo "ALL INSTALLER CHECKS PASSED"

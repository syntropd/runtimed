#!/usr/bin/env bash
# runtimed one-line installer: fetches the self-installing release
# bundle, verifies its checksum, and runs it. Usage:
#
#   curl -fsSL https://syntropd.github.io/runtimed-install.sh | sudo bash
#
# That installs the latest release plus the recommended brain (Gemma 4
# E2B). Options pass through to install.sh after `-s --`, e.g.:
#
#   ... | sudo bash -s -- --with-gemma --with-vision
#   ... | sudo bash -s -- --no-models
#
# RUNTIMED_VERSION=X.Y.Z (or --version X.Y.Z) pins a release instead
# of the latest. --dry-run prints the resolved version and URL only.
#
# Mirror: this file is served from the website repo, but the canonical
# copy lives at runtimed/install/quick-install.sh — keep them identical.
set -euo pipefail

REPO="syntropd/runtimed"
VERSION="${RUNTIMED_VERSION:-}"
DRY_RUN=0
ARGS=()
HAS_MODEL_FLAG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="${2:-}"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        --with-* | --no-models) ARGS+=("$1"); HAS_MODEL_FLAG=1 ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

if [[ -z "${VERSION}" ]]; then
    LATEST=""
    LATEST="$(curl -sSfL --max-time 30 \
        "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null \
        | grep -o '"tag_name": *"v[^"]*"' | head -n1 | cut -d'"' -f4)" || LATEST=""
    VERSION="${LATEST#v}"
fi
[[ -n "${VERSION}" ]] || {
    echo "ERROR: could not find the latest release." >&2
    echo "Set one explicitly: RUNTIMED_VERSION=X.Y.Z" >&2
    exit 1
}

TARBALL="runtimed-v${VERSION}-x86_64-unknown-linux-gnu.tar.gz"
BASE_URL="https://github.com/${REPO}/releases/download/v${VERSION}/${TARBALL}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "version: ${VERSION}"
    echo "url: ${BASE_URL}"
    exit 0
fi

if [[ "${EUID}" -ne 0 ]]; then
    echo "Error: pipe this script into 'sudo bash' (it installs system files)." >&2
    exit 1
fi
command -v curl >/dev/null 2>&1 || {
    echo "ERROR: curl is missing. Install it and re-run." >&2
    exit 1
}
command -v sha256sum >/dev/null 2>&1 || {
    echo "ERROR: sha256sum is missing. Install coreutils and re-run." >&2
    exit 1
}

# Piped stdin is not a terminal, so the installer's interactive model
# question cannot work here: default to the recommended brain unless
# the caller said otherwise.
if [[ "${HAS_MODEL_FLAG}" -eq 0 ]]; then
    ARGS=(--with-gemma "${ARGS[@]:+"${ARGS[@]}"}")
fi

WORK="$(mktemp -d /tmp/runtimed-install.XXXXXX)"
trap 'rm -rf "${WORK}"' EXIT

echo "==> Downloading runtimed v${VERSION}..."
curl -fSL --retry 3 --retry-delay 2 -o "${WORK}/bundle.tar.gz" "${BASE_URL}"
curl -sfSL -o "${WORK}/bundle.tar.gz.sha256" "${BASE_URL}.sha256"

echo "==> Verifying checksum..."
(cd "${WORK}" && echo "$(cat bundle.tar.gz.sha256)  bundle.tar.gz" | sha256sum -c -) \
    || { echo "ERROR: checksum mismatch; refusing to install." >&2; exit 1; }

echo "==> Running the installer..."
tar -xzf "${WORK}/bundle.tar.gz" -C "${WORK}"
BUNDLE_ROOT="${WORK}/runtimed-v${VERSION}"
if [[ "${#ARGS[@]}" -eq 0 ]]; then
    exec bash "${BUNDLE_ROOT}/install/install.sh" --from-bundle "${BUNDLE_ROOT}/bin"
else
    exec bash "${BUNDLE_ROOT}/install/install.sh" --from-bundle "${BUNDLE_ROOT}/bin" "${ARGS[@]}"
fi

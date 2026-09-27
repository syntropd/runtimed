#!/usr/bin/env bash
# Install runtimed + runtimectl, register the service, enroll the user,
# and optionally fetch starter models. Safe to re-run: completed steps
# are skipped, downloads resume only when the file is missing.
set -euo pipefail

WITH_STARTER=0
WITH_GEMMA=0
WITH_VISION=0
CUDA_GPU=""
FROM_BUNDLE=""
TARGET_USER="${SUDO_USER:-}"
ASSUME_YES=0
NO_MODELS=0

usage() {
    cat <<EOF
Usage: sudo bash install/install.sh [OPTIONS]

Options:
  --with-starter-model   Download Qwen 0.5B starter model (~700 MB).
  --with-gemma           Download Gemma 4 E2B Q4, the recommended brain (~3.1 GB).
  --with-vision          Download the Gemma vision file for picture questions (~1 GB).
  --cuda-gpu N           Build with CUDA and use graphics card N automatically.
  --from-bundle DIR      Install prebuilt binaries from DIR (no Rust needed).
  --user NAME            Enroll NAME in the syntrop group (default: invoking user).
  --yes                  Accept interactive defaults (recommended models).
  --no-models            Skip the interactive model download question.
  --help                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-starter-model) WITH_STARTER=1 ;;
        --with-gemma) WITH_GEMMA=1 ;;
        --with-vision) WITH_VISION=1 ;;
        --cuda-gpu) CUDA_GPU="${2:-}"; shift ;;
        --from-bundle) FROM_BUNDLE="${2:-}"; shift ;;
        --user) TARGET_USER="${2:-}"; shift ;;
        --yes) ASSUME_YES=1 ;;
        --no-models) NO_MODELS=1 ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
    shift
done

# Fail fast on flag problems before demanding root.
if [[ -n "${FROM_BUNDLE}" && -n "${CUDA_GPU}" ]]; then
    echo "Error: --from-bundle is CPU-only; clone the repo for CUDA builds." >&2
    exit 1
fi
if [[ -n "${FROM_BUNDLE}" ]]; then
    for b in runtimed runtimectl; do
        [[ -x "${FROM_BUNDLE}/${b}" ]] || {
            echo "Error: --from-bundle ${FROM_BUNDLE} has no ${b} binary." >&2
            exit 1
        }
    done
fi

if [[ "${EUID}" -ne 0 ]]; then
    echo "Error: install.sh must be executed with root privileges." >&2
    exit 1
fi

if [[ -n "${CUDA_GPU}" && ! "${CUDA_GPU}" =~ ^[0-9]+$ ]]; then
    echo "Error: --cuda-gpu needs a card number (e.g. --cuda-gpu 0)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="/var/lib/models/gguf"

# All URLs below were verified live (HTTP 200) before release. The
# HuggingFace "resolve" links always serve the exact file bytes.
QWEN_GGUF_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"
QWEN_TOK_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json"
GEMMA_Q4_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf"

echo "==> Checking prerequisites..."
if [[ -z "${FROM_BUNDLE}" ]]; then
    command -v cargo >/dev/null 2>&1 || {
        echo "ERROR: the Rust toolchain is missing. Install it first:" >&2
        echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" >&2
        echo "then open a fresh terminal and re-run this installer." >&2
        echo "(Or use the release bundle, which needs no Rust: see README.)" >&2
        exit 1
    }
fi
command -v curl >/dev/null 2>&1 || {
    echo "ERROR: curl is missing. Install it (e.g. sudo dnf install curl) and re-run." >&2
    exit 1
}
if [[ -n "${CUDA_GPU}" ]]; then RUNTIMED_CUDA=1; fi
if [[ "${RUNTIMED_CUDA:-0}" == "1" ]]; then
    command -v nvcc >/dev/null 2>&1 || {
        echo "ERROR: CUDA build needs nvcc (CUDA toolkit) on PATH." >&2
        exit 1
    }
fi

if [[ -n "${FROM_BUNDLE}" ]]; then
    echo "==> Using prebuilt binaries from ${FROM_BUNDLE} (no build)..."
    BIN_SRC="${FROM_BUNDLE}"
else
echo "==> Building runtimed and runtimectl in release mode..."
if [[ "${RUNTIMED_CUDA:-0}" == "1" ]]; then
    : "${CUDA_ROOT:=/usr/local/cuda}"
    export CUDA_ROOT
    # Partial toolkits (cudart but no cublas): point CUDA_LIB_DIR at the
    # extracted libs; they are baked in as RUNPATH as well as -L.
    if [[ -n "${CUDA_LIB_DIR:-}" ]]; then
        export RUSTFLAGS="${RUSTFLAGS:-} -L ${CUDA_LIB_DIR} -C link-args=-Wl,-rpath,${CUDA_LIB_DIR}"
    fi
    cargo build --release --manifest-path "${ROOT_DIR}/Cargo.toml" -p syntrop-runtimed --features cuda
    cargo build --release --manifest-path "${ROOT_DIR}/Cargo.toml" -p runtimectl
else
    cargo build --release --manifest-path "${ROOT_DIR}/Cargo.toml"
fi
    BIN_SRC="${ROOT_DIR}/target/release"
fi

echo "==> Installing binaries to /usr/local/bin..."
install -m 0755 "${BIN_SRC}/runtimed" /usr/local/bin/runtimed
install -m 0755 "${BIN_SRC}/runtimectl" /usr/local/bin/runtimectl

echo "==> Setting up system user, groups, and directories..."
install -m 0644 "${ROOT_DIR}/sysusers.d/runtimed.conf" /usr/lib/sysusers.d/runtimed.conf
systemd-sysusers /usr/lib/sysusers.d/runtimed.conf
install -m 0644 "${ROOT_DIR}/tmpfiles.d/runtimed.conf" /usr/lib/tmpfiles.d/runtimed.conf
systemd-tmpfiles --create /usr/lib/tmpfiles.d/runtimed.conf

echo "==> Installing systemd units..."
install -m 0644 "${ROOT_DIR}/systemd/runtimed.socket" /usr/lib/systemd/system/runtimed.socket
install -m 0644 "${ROOT_DIR}/systemd/runtimed.service" /usr/lib/systemd/system/runtimed.service

if [[ -n "${CUDA_GPU}" ]]; then
    echo "==> Selecting graphics card ${CUDA_GPU}..."
    mkdir -p /etc/systemd/system/runtimed.service.d
    printf '[Service]\nEnvironment=RUNTIMED_BACKEND=cuda:%s\n' "${CUDA_GPU}" \
        > /etc/systemd/system/runtimed.service.d/cuda.conf
fi

echo "==> Reloading systemd daemon..."
systemctl daemon-reload
systemctl enable --now runtimed.socket
if [[ -n "${CUDA_GPU}" ]]; then systemctl try-restart runtimed.service; fi

# --- user enrollment: the CLI talks to a socket owned by root:syntrop,
# so the human needs that group (effective on next login). ---
if [[ -z "${TARGET_USER}" || "${TARGET_USER}" == "root" ]]; then
    echo "NOTE: no normal user detected (running as root directly?)." >&2
    echo "Each human needs:  sudo usermod -aG syntrop <name>  + a fresh login." >&2
elif id -nG "${TARGET_USER}" 2>/dev/null | grep -qw syntrop; then
    echo "==> User ${TARGET_USER} is already in the syntrop group."
else
    usermod -aG syntrop "${TARGET_USER}"
    echo "==> Added ${TARGET_USER} to the syntrop group (needs a fresh login)."
fi

# --- models: ask once when interactive, obey flags otherwise. ---
if [[ "${NO_MODELS}" -eq 0 && "${WITH_STARTER}" -eq 0 && "${WITH_GEMMA}" -eq 0 && "${WITH_VISION}" -eq 0 ]]; then
    if [[ "${ASSUME_YES}" -eq 1 ]]; then
        WITH_GEMMA=1
    elif [[ -t 0 ]]; then
        ans=""
        read -r -p "Download the recommended brain, Gemma 4 E2B (3.1 GB)? [Y/n] " ans || true
        [[ "${ans}" =~ ^[Nn] ]] || WITH_GEMMA=1
    else
        echo "NOTE: non-interactive shell and no --with-* flag: skipping model downloads."
    fi
fi

fetch() {
    local url="$1" dest="$2"
    if [[ -s "${dest}" ]]; then
        echo "  already present: $(basename "${dest}")"
        return 0
    fi
    echo "  downloading $(basename "${dest}")..."
    local tmp="${dest}.part"
    if ! curl -fSL --retry 3 --retry-delay 2 -o "${tmp}" "${url}"; then
        rm -f "${tmp}"
        echo "ERROR: download failed: ${url}" >&2
        echo "Check your connection and re-run the installer to resume." >&2
        exit 1
    fi
    mv "${tmp}" "${dest}"
    chmod 0644 "${dest}"
}

if [[ "${WITH_STARTER}" -eq 1 || "${WITH_GEMMA}" -eq 1 || "${WITH_VISION}" -eq 1 ]]; then
    echo "==> Fetching models into ${MODELS_DIR}..."
    mkdir -p "${MODELS_DIR}"
fi
if [[ "${WITH_STARTER}" -eq 1 ]]; then
    fetch "${QWEN_GGUF_URL}" "${MODELS_DIR}/qwen2.5-0.5b-instruct-q8_0.gguf"
    # Qwen needs its word-list file sitting next to it under this exact name.
    fetch "${QWEN_TOK_URL}" "${MODELS_DIR}/qwen2.5-0.5b-instruct-q8_0.tokenizer.json"
fi
if [[ "${WITH_GEMMA}" -eq 1 ]]; then
    fetch "${GEMMA_Q4_URL}" "${MODELS_DIR}/gemma-4-E2B-it-Q4_K_M.gguf"
fi
if [[ "${WITH_VISION}" -eq 1 ]]; then
    fetch "${MMPROJ_URL}" "${MODELS_DIR}/mmproj-F16.gguf"
fi

# --- self-verify: the socket must be live and the daemon must answer. ---
echo "==> Verifying the install..."
systemctl is-active --quiet runtimed.socket \
    || { echo "ERROR: runtimed.socket is not active." >&2; exit 1; }
/usr/local/bin/runtimectl list \
    || { echo "ERROR: the daemon did not answer." >&2; exit 1; }

echo ""
echo "Install complete. The engine is running."
if [[ -n "${TARGET_USER}" && "${TARGET_USER}" != "root" ]]; then
    echo "If this just added you to the syntrop group: log out and back in first."
fi
echo ""
echo "Try it:"
if [[ -s "${MODELS_DIR}/gemma-4-E2B-it-Q4_K_M.gguf" ]]; then
    echo "  runtimectl generate -m gemma-4-E2B-it-Q4_K_M \"Say hello in one sentence.\""
fi
if [[ -s "${MODELS_DIR}/qwen2.5-0.5b-instruct-q8_0.gguf" ]]; then
    echo "  runtimectl generate -m qwen2.5-0.5b-instruct-q8_0 \"Say hello in one sentence.\""
fi
if [[ ! -s "${MODELS_DIR}/gemma-4-E2B-it-Q4_K_M.gguf" && ! -s "${MODELS_DIR}/qwen2.5-0.5b-instruct-q8_0.gguf" ]]; then
    echo "  (no models yet: re-run with --with-gemma, then ask your first question)"
fi
if [[ -s "${MODELS_DIR}/mmproj-F16.gguf" ]]; then
    echo "  runtimectl attach-vision gemma-4-E2B-it-Q4_K_M mmproj-F16   # before picture questions"
fi

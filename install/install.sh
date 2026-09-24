#!/usr/bin/env bash
set -euo pipefail

# Installer script for runtimed and runtimectl
# Must be executed as root

if [[ "${EUID}" -ne 0 ]]; then
    echo "Error: install.sh must be executed with root privileges." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Building runtimed and runtimectl in release mode..."
cargo build --release --manifest-path "${ROOT_DIR}/Cargo.toml"

echo "==> Installing binaries to /usr/local/bin..."
install -m 0755 "${ROOT_DIR}/target/release/runtimed" /usr/local/bin/runtimed
install -m 0755 "${ROOT_DIR}/target/release/runtimectl" /usr/local/bin/runtimectl

echo "==> Setting up systemd sysusers and tmpfiles..."
if [[ -f "${ROOT_DIR}/sysusers.d/runtimed.conf" ]]; then
    install -m 0644 "${ROOT_DIR}/sysusers.d/runtimed.conf" /usr/lib/sysusers.d/runtimed.conf
    systemd-sysusers /usr/lib/sysusers.d/runtimed.conf || true
fi

if [[ -f "${ROOT_DIR}/tmpfiles.d/runtimed.conf" ]]; then
    install -m 0644 "${ROOT_DIR}/tmpfiles.d/runtimed.conf" /usr/lib/tmpfiles.d/runtimed.conf
    systemd-tmpfiles --create /usr/lib/tmpfiles.d/runtimed.conf || true
fi

echo "==> Installing systemd units..."
install -m 0644 "${ROOT_DIR}/systemd/runtimed.socket" /usr/lib/systemd/system/runtimed.socket
install -m 0644 "${ROOT_DIR}/systemd/runtimed.service" /usr/lib/systemd/system/runtimed.service

echo "==> Reloading systemd daemon..."
systemctl daemon-reload
systemctl enable --now runtimed.socket

echo "==> runtimed socket activated successfully."
echo "Verify status: systemctl status runtimed.socket"

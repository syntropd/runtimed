#!/usr/bin/env bash
set -euo pipefail

# Uninstaller script for runtimed and runtimectl
# Must be executed as root

if [[ "${EUID}" -ne 0 ]]; then
    echo "Error: uninstall.sh must be executed with root privileges." >&2
    exit 1
fi

echo "==> Stopping and disabling runtimed service and socket..."
systemctl disable --now runtimed.service runtimed.socket 2>/dev/null || true

echo "==> Removing systemd unit files..."
rm -f /usr/lib/systemd/system/runtimed.service
rm -f /usr/lib/systemd/system/runtimed.socket
rm -f /usr/lib/sysusers.d/runtimed.conf
rm -f /usr/lib/tmpfiles.d/runtimed.conf

echo "==> Reloading systemd daemon..."
systemctl daemon-reload

echo "==> Removing installed binaries..."
rm -f /usr/local/bin/runtimed
rm -f /usr/local/bin/runtimectl

echo "==> Cleaning up runtime sockets..."
rm -rf /run/syntrop/io.syntrop.Runtime1

echo "==> runtimed uninstallation completed."
echo "Note: Downloaded model weights in /var/lib/models are preserved."
echo "To remove cached models, run: rm -rf /var/lib/models"

#!/bin/sh
set -eu

usage() {
  cat <<'USAGE'
Usage:
  tools/platform/linux/camera_network_down.sh

Restores the connection state saved by camera_network_up.sh. The restore uses
only the runtime state file so cleanup remains possible even if the local host
configuration was changed or removed after activation. If no state file exists,
the script performs no network changes.
USAGE
}

fail() {
  echo "[network-down] ERROR: $*" >&2
  exit 1
}

info() {
  echo "[network-down] $*"
}

case "${1:-}" in
  "") ;;
  -h|--help)
    usage
    exit 0
    ;;
  *) fail "this script takes no positional arguments" ;;
esac

command -v nmcli >/dev/null 2>&1 || fail "nmcli is not installed"

if [ -n "${NAVMIN_NETWORK_STATE_FILE:-}" ]; then
  STATE_FILE=$NAVMIN_NETWORK_STATE_FILE
else
  if [ -n "${XDG_RUNTIME_DIR:-}" ]; then
    STATE_DIR="$XDG_RUNTIME_DIR/navmin"
  else
    STATE_DIR="/tmp/navmin-$(id -u)"
  fi
  STATE_FILE="$STATE_DIR/camera-network.state"
fi

if [ ! -f "$STATE_FILE" ]; then
  info "no saved NavMin network state; nothing to restore"
  exit 0
fi

# shellcheck disable=SC1090
. "$STATE_FILE"
[ -n "${interface:-}" ] || fail "invalid state file: missing interface"
[ -n "${target_uuid:-}" ] || fail "invalid state file: missing target_uuid"

active_uuid_for_interface() {
  nmcli -t -f UUID,DEVICE connection show --active | awk -F: -v dev="$interface" '$2 == dev { print $1; exit }'
}

CURRENT_UUID=$(active_uuid_for_interface)
PREVIOUS_UUID=${previous_uuid:-}

if [ "$PREVIOUS_UUID" = "$target_uuid" ]; then
  info "camera profile was active before NavMin; leaving it active"
elif [ -n "$PREVIOUS_UUID" ]; then
  if ! nmcli -g connection.id connection show uuid "$PREVIOUS_UUID" >/dev/null 2>&1; then
    fail "previous NetworkManager connection $PREVIOUS_UUID no longer exists; restore manually and keep $STATE_FILE for evidence"
  fi
  if [ "$CURRENT_UUID" != "$PREVIOUS_UUID" ]; then
    nmcli connection up uuid "$PREVIOUS_UUID" ifname "$interface" >/dev/null || fail "failed to restore previous connection $PREVIOUS_UUID on $interface"
  fi
  info "restored previous connection on $interface"
else
  if [ "$CURRENT_UUID" = "$target_uuid" ]; then
    nmcli connection down uuid "$target_uuid" >/dev/null || fail "failed to disconnect NavMin camera profile"
  fi
  info "camera profile disconnected; there was no previous active connection on $interface"
fi

rm -f "$STATE_FILE"
info "network restore complete"

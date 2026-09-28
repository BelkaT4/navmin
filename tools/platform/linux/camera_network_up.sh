#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
# shellcheck source=tools/platform/linux/host_config.sh
. "$repo_root/tools/platform/linux/host_config.sh"

usage() {
  cat <<'USAGE'
Usage:
  tools/platform/linux/camera_network_up.sh

Reads config/host.local.env (or NAVMIN_HOST_CONFIG), verifies that the prepared
NetworkManager profile matches it, remembers the connection currently active on
the configured interface, activates the camera profile, and verifies the static
IPv4 address.
USAGE
}

fail() {
  echo "[network-up] ERROR: $*" >&2
  exit 1
}

info() {
  echo "[network-up] $*"
}

case "${1:-}" in
  "") ;;
  -h|--help)
    usage
    exit 0
    ;;
  *) fail "this script takes no positional arguments; edit config/host.local.env" ;;
esac

CONFIG_FILE=$(navmin_host_config_path "$repo_root")
[ -f "$CONFIG_FILE" ] || fail "host config not found: $CONFIG_FILE; copy config/host.example.env to config/host.local.env and edit it"
EXPECTED_INTERFACE=$(navmin_host_config_get CAMERA_INTERFACE "$CONFIG_FILE") || fail "CAMERA_INTERFACE is missing or duplicated in $CONFIG_FILE"
EXPECTED_ADDRESS=$(navmin_host_config_get CAMERA_HOST_ADDRESS "$CONFIG_FILE") || fail "CAMERA_HOST_ADDRESS is missing or duplicated in $CONFIG_FILE"
PROFILE=$(navmin_host_config_get CAMERA_PROFILE_NAME "$CONFIG_FILE") || fail "CAMERA_PROFILE_NAME is missing or duplicated in $CONFIG_FILE"
[ -n "$EXPECTED_INTERFACE" ] || fail "CAMERA_INTERFACE is empty in $CONFIG_FILE"
[ -n "$EXPECTED_ADDRESS" ] || fail "CAMERA_HOST_ADDRESS is empty in $CONFIG_FILE"
[ -n "$PROFILE" ] || fail "CAMERA_PROFILE_NAME is empty in $CONFIG_FILE"

command -v nmcli >/dev/null 2>&1 || fail "nmcli is not installed"

TARGET_UUID=$(nmcli -g connection.uuid connection show "$PROFILE" 2>/dev/null | head -n 1 || true)
[ -n "$TARGET_UUID" ] || fail "NetworkManager profile not found: $PROFILE; run setup_camera_network.sh first"
INTERFACE=$(nmcli -g connection.interface-name connection show uuid "$TARGET_UUID" | head -n 1)
[ "$INTERFACE" = "$EXPECTED_INTERFACE" ] || fail "profile '$PROFILE' uses interface '$INTERFACE', but host config expects '$EXPECTED_INTERFACE'; run setup_camera_network.sh again"
PROFILE_ADDRESS=$(nmcli -g ipv4.addresses connection show uuid "$TARGET_UUID" | head -n 1)
PROFILE_ADDRESS=${PROFILE_ADDRESS%%,*}
[ "$PROFILE_ADDRESS" = "$EXPECTED_ADDRESS" ] || fail "profile '$PROFILE' uses IPv4 '$PROFILE_ADDRESS', but host config expects '$EXPECTED_ADDRESS'; run setup_camera_network.sh again"

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
STATE_DIR=$(dirname "$STATE_FILE")
umask 077
mkdir -p "$STATE_DIR"
chmod 700 "$STATE_DIR" 2>/dev/null || true

active_uuid_for_interface() {
  nmcli -t -f UUID,DEVICE connection show --active | awk -F: -v dev="$INTERFACE" '$2 == dev { print $1; exit }'
}

verify_target() {
  CURRENT_UUID=$(active_uuid_for_interface)
  [ "$CURRENT_UUID" = "$TARGET_UUID" ] || return 1
  ACTUAL_ADDRESSES=$(nmcli -g IP4.ADDRESS device show "$INTERFACE" 2>/dev/null || true)
  printf '%s\n' "$ACTUAL_ADDRESSES" | grep -Fx "$EXPECTED_ADDRESS" >/dev/null 2>&1
}

if [ -f "$STATE_FILE" ]; then
  # shellcheck disable=SC1090
  . "$STATE_FILE"
  if [ "${target_uuid:-}" = "$TARGET_UUID" ] && [ "${interface:-}" = "$INTERFACE" ] && verify_target; then
    info "profile '$PROFILE' is already active; existing restore state is preserved"
    exit 0
  fi
  fail "stale network state exists at $STATE_FILE; run camera_network_down.sh before starting a new session"
fi

PREVIOUS_UUID=$(active_uuid_for_interface)
TMP_STATE="$STATE_FILE.tmp.$$"
trap 'rm -f "$TMP_STATE"' EXIT HUP INT TERM
{
  printf 'interface=%s\n' "$INTERFACE"
  printf 'target_uuid=%s\n' "$TARGET_UUID"
  printf 'previous_uuid=%s\n' "$PREVIOUS_UUID"
} > "$TMP_STATE"
mv "$TMP_STATE" "$STATE_FILE"
trap - EXIT HUP INT TERM

rollback() {
  DOWN_SCRIPT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/camera_network_down.sh
  "$DOWN_SCRIPT" >/dev/null 2>&1 || true
}

trap 'rollback; exit 129' HUP
trap 'rollback; exit 130' INT
trap 'rollback; exit 143' TERM

if ! nmcli connection up uuid "$TARGET_UUID" ifname "$INTERFACE" >/dev/null; then
  rollback
  fail "failed to activate profile '$PROFILE' on $INTERFACE"
fi

if ! verify_target; then
  rollback
  fail "profile '$PROFILE' activated but expected address $EXPECTED_ADDRESS is not active on $INTERFACE"
fi

trap - HUP INT TERM

if [ -n "$PREVIOUS_UUID" ] && [ "$PREVIOUS_UUID" != "$TARGET_UUID" ]; then
  info "saved previous connection UUID for restore: $PREVIOUS_UUID"
elif [ "$PREVIOUS_UUID" = "$TARGET_UUID" ]; then
  info "camera profile was already active before this session; it will be left active on exit"
else
  info "no previous active connection on $INTERFACE; camera profile will be disconnected on exit"
fi
info "host config: $CONFIG_FILE"
info "active profile: $PROFILE"
info "interface: $INTERFACE"
info "verified IPv4: $EXPECTED_ADDRESS"

#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
# shellcheck source=tools/platform/linux/host_config.sh
. "$repo_root/tools/platform/linux/host_config.sh"

usage() {
  cat <<'USAGE'
Usage:
  sudo tools/platform/linux/setup_camera_network.sh

Reads the local host settings from config/host.local.env (or NAVMIN_HOST_CONFIG),
clones the currently active NetworkManager connection on the configured
interface when necessary, and applies the configured static PC IPv4 address.

Required settings:
  CAMERA_INTERFACE
  CAMERA_HOST_ADDRESS
  CAMERA_PROFILE_NAME

Options:
  -h, --help            Show this help.
USAGE
}

fail() {
  echo "[setup-network] ERROR: $*" >&2
  exit 1
}

info() {
  echo "[setup-network] $*"
}

case "${1:-}" in
  "") ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    fail "this script takes no positional arguments; edit config/host.local.env"
    ;;
esac

CONFIG_FILE=$(navmin_host_config_path "$repo_root")
[ -f "$CONFIG_FILE" ] || fail "host config not found: $CONFIG_FILE; copy config/host.example.env to config/host.local.env and edit it"
INTERFACE=$(navmin_host_config_get CAMERA_INTERFACE "$CONFIG_FILE") || fail "CAMERA_INTERFACE is missing or duplicated in $CONFIG_FILE"
ADDRESS=$(navmin_host_config_get CAMERA_HOST_ADDRESS "$CONFIG_FILE") || fail "CAMERA_HOST_ADDRESS is missing or duplicated in $CONFIG_FILE"
PROFILE=$(navmin_host_config_get CAMERA_PROFILE_NAME "$CONFIG_FILE") || fail "CAMERA_PROFILE_NAME is missing or duplicated in $CONFIG_FILE"

[ "$(id -u)" -eq 0 ] || fail "run this one-time setup with sudo"
[ -n "$INTERFACE" ] || fail "CAMERA_INTERFACE is empty in $CONFIG_FILE"
[ -n "$ADDRESS" ] || fail "CAMERA_HOST_ADDRESS is empty in $CONFIG_FILE"
[ -n "$PROFILE" ] || fail "CAMERA_PROFILE_NAME is empty in $CONFIG_FILE"
case "$ADDRESS" in
  */*) ;;
  *) fail "CAMERA_HOST_ADDRESS must include a CIDR prefix, for example 192.168.42.2/24" ;;
esac

command -v nmcli >/dev/null 2>&1 || fail "nmcli is not installed; NetworkManager is required"
nmcli -g GENERAL.TYPE device show "$INTERFACE" >/dev/null 2>&1 || fail "NetworkManager does not know interface: $INTERFACE"

if nmcli -g connection.uuid connection show "$PROFILE" >/dev/null 2>&1; then
  info "profile already exists and will be updated: $PROFILE"
else
  BASE_UUID=$(nmcli -t -f UUID,DEVICE connection show --active | awk -F: -v dev="$INTERFACE" '$2 == dev { print $1; exit }')
  [ -n "$BASE_UUID" ] || fail "no active NetworkManager connection on $INTERFACE; connect the interface to the desired camera network first"
  nmcli connection clone uuid "$BASE_UUID" "$PROFILE" >/dev/null
  info "created profile '$PROFILE' by cloning the active connection on $INTERFACE"
fi

PROFILE_TYPE=$(nmcli -g connection.type connection show "$PROFILE" | head -n 1)
case "$PROFILE_TYPE" in
  802-3-ethernet|802-11-wireless) ;;
  *) fail "unsupported profile type '$PROFILE_TYPE'; expected Ethernet or Wi-Fi" ;;
esac

nmcli connection modify "$PROFILE" \
  connection.interface-name "$INTERFACE" \
  connection.autoconnect no \
  ipv4.method manual \
  ipv4.addresses "$ADDRESS" \
  ipv4.gateway "" \
  ipv4.dns "" \
  ipv4.dns-search "" \
  ipv4.routes "" \
  ipv4.never-default yes \
  ipv6.method disabled

PROFILE_UUID=$(nmcli -g connection.uuid connection show "$PROFILE" | head -n 1)
CONFIGURED_ADDRESS=$(nmcli -g ipv4.addresses connection show "$PROFILE" | head -n 1)
CONFIGURED_INTERFACE=$(nmcli -g connection.interface-name connection show "$PROFILE" | head -n 1)

[ "$CONFIGURED_INTERFACE" = "$INTERFACE" ] || fail "profile verification failed: interface is '$CONFIGURED_INTERFACE'"
printf '%s\n' "$CONFIGURED_ADDRESS" | grep -F "$ADDRESS" >/dev/null 2>&1 || fail "profile verification failed: IPv4 address is '$CONFIGURED_ADDRESS'"

info "host config: $CONFIG_FILE"
info "profile: $PROFILE"
info "uuid: $PROFILE_UUID"
info "interface: $INTERFACE"
info "static IPv4: $ADDRESS"
info "autoconnect: disabled"
info "default route: disabled for this camera profile"
info "setup complete; runtime activation is handled by camera_network_up.sh/run_navmin.sh"

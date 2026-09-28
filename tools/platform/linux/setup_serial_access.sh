#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
# shellcheck source=tools/platform/linux/host_config.sh
. "$repo_root/tools/platform/linux/host_config.sh"

usage() {
  cat <<'USAGE'
Usage:
  sudo tools/platform/linux/setup_serial_access.sh

Reads the local host settings from config/host.local.env (or NAVMIN_HOST_CONFIG)
and creates a persistent udev rule for the configured USB-UART adapter.

Required settings:
  SERIAL_SETUP_DEVICE   Current device path used only during one-time setup.
  SERIAL_ALIAS          Stable /dev alias created by udev, without /dev/.

Options:
  -h, --help            Show this help.
USAGE
}

fail() {
  echo "[setup-serial] ERROR: $*" >&2
  exit 1
}

info() {
  echo "[setup-serial] $*"
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
DEVICE=$(navmin_host_config_get SERIAL_SETUP_DEVICE "$CONFIG_FILE") || fail "SERIAL_SETUP_DEVICE is missing or duplicated in $CONFIG_FILE"
ALIAS=$(navmin_host_config_get SERIAL_ALIAS "$CONFIG_FILE") || fail "SERIAL_ALIAS is missing or duplicated in $CONFIG_FILE"
TARGET_USER="${SUDO_USER:-}"

[ "$(id -u)" -eq 0 ] || fail "run this one-time setup with sudo"
[ -n "$DEVICE" ] || fail "SERIAL_SETUP_DEVICE is empty in $CONFIG_FILE"
[ -n "$ALIAS" ] || fail "SERIAL_ALIAS is empty in $CONFIG_FILE"
[ -c "$DEVICE" ] || fail "$DEVICE is not a character device"

case "$ALIAS" in
  *[!A-Za-z0-9._-]*|'') fail "SERIAL_ALIAS may contain only A-Z, a-z, 0-9, dot, underscore and dash" ;;
esac

command -v udevadm >/dev/null 2>&1 || fail "udevadm is not installed"
command -v getent >/dev/null 2>&1 || fail "getent is not installed"
command -v usermod >/dev/null 2>&1 || fail "usermod is not installed"

[ -n "$TARGET_USER" ] || fail "cannot determine the user that invoked sudo"
id "$TARGET_USER" >/dev/null 2>&1 || fail "user does not exist: $TARGET_USER"
getent group dialout >/dev/null 2>&1 || fail "group 'dialout' does not exist"

PROPERTIES=$(udevadm info --query=property --name="$DEVICE")
VENDOR_ID=$(printf '%s\n' "$PROPERTIES" | sed -n 's/^ID_VENDOR_ID=//p' | head -n 1)
PRODUCT_ID=$(printf '%s\n' "$PROPERTIES" | sed -n 's/^ID_MODEL_ID=//p' | head -n 1)
SERIAL_SHORT=$(printf '%s\n' "$PROPERTIES" | sed -n 's/^ID_SERIAL_SHORT=//p' | head -n 1)

[ -n "$VENDOR_ID" ] || fail "udev did not report ID_VENDOR_ID for $DEVICE"
[ -n "$PRODUCT_ID" ] || fail "udev did not report ID_MODEL_ID for $DEVICE"

escape_udev_value() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

VENDOR_ESC=$(escape_udev_value "$VENDOR_ID")
PRODUCT_ESC=$(escape_udev_value "$PRODUCT_ID")
SERIAL_ESC=$(escape_udev_value "$SERIAL_SHORT")

RULE='SUBSYSTEM=="tty", ATTRS{idVendor}=="'"$VENDOR_ESC"'", ATTRS{idProduct}=="'"$PRODUCT_ESC"'"'
if [ -n "$SERIAL_ESC" ]; then
  RULE="$RULE, ATTRS{serial}==\"$SERIAL_ESC\""
else
  info "WARNING: adapter has no ID_SERIAL_SHORT; the rule will match every tty adapter with VID:PID $VENDOR_ID:$PRODUCT_ID"
fi
RULE="$RULE, GROUP=\"dialout\", MODE=\"0660\", SYMLINK+=\"$ALIAS\""

RULES_DIR="${NAVMIN_UDEV_RULES_DIR:-/etc/udev/rules.d}"
mkdir -p "$RULES_DIR"
RULE_PATH="$RULES_DIR/99-navmin-turret.rules"
TMP_RULE=$(mktemp)
trap 'rm -f "$TMP_RULE"' EXIT HUP INT TERM
printf '%s\n' "$RULE" > "$TMP_RULE"
chmod 0644 "$TMP_RULE"

if [ -f "$RULE_PATH" ] && cmp -s "$TMP_RULE" "$RULE_PATH"; then
  info "udev rule already configured: $RULE_PATH"
else
  install -m 0644 "$TMP_RULE" "$RULE_PATH"
  info "installed udev rule: $RULE_PATH"
fi

ADDED_TO_GROUP=0
if id -nG "$TARGET_USER" | tr ' ' '\n' | grep -Fxq dialout; then
  info "user $TARGET_USER is already in dialout"
else
  usermod -aG dialout "$TARGET_USER"
  ADDED_TO_GROUP=1
  info "added user $TARGET_USER to dialout"
fi

udevadm control --reload-rules
DEVPATH=$(udevadm info --query=path --name="$DEVICE" 2>/dev/null || true)
if [ -n "$DEVPATH" ]; then
  udevadm trigger --action=add "/sys$DEVPATH" >/dev/null 2>&1 || true
  udevadm settle >/dev/null 2>&1 || true
fi

if [ -e "/dev/$ALIAS" ]; then
  info "stable device alias is available: /dev/$ALIAS"
else
  info "udev rule is installed; unplug and reconnect the adapter to create /dev/$ALIAS"
fi

if [ "$ADDED_TO_GROUP" -eq 1 ]; then
  info "log out and log in again before starting NavMin so the new dialout membership becomes active"
fi

info "host config: $CONFIG_FILE"
info "set turret.serial.port to /dev/$ALIAS in the site config"
info "done"

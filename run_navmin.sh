#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
network_up="$repo_root/tools/platform/linux/camera_network_up.sh"
network_down="$repo_root/tools/platform/linux/camera_network_down.sh"
host_helpers="$repo_root/tools/platform/linux/host_config.sh"

fail() {
  echo "[run-navmin] ERROR: $*" >&2
  exit 1
}

[ -x "$network_up" ] || fail "network helper is not executable: $network_up"
[ -x "$network_down" ] || fail "network helper is not executable: $network_down"
[ -f "$host_helpers" ] || fail "host config helper not found: $host_helpers"

# shellcheck source=tools/platform/linux/host_config.sh
. "$host_helpers"

cd "$repo_root"

host_config=$(navmin_host_config_path "$repo_root")
if [ -n "${NAVMIN_HOST_CONFIG:-}" ]; then
  [ -f "$host_config" ] || fail "explicit host config not found: $host_config"
elif [ ! -f "$host_config" ]; then
  default_host_config=$(navmin_default_host_config_path "$repo_root")
  [ -f "$default_host_config" ] || fail "safe host default not found: $default_host_config"
  mkdir -p "$(dirname "$host_config")"
  cp "$default_host_config" "$host_config" || \
    fail "failed to create $host_config from $default_host_config"
  echo "[run-navmin] Created safe config/host.env from config/defaults/host.env."
  echo "[run-navmin] Review and fill host parameters, then run NavMin again."
  exit 2
fi

command -v uv >/dev/null 2>&1 || fail "uv is not installed or not available in PATH"

cleanup_done=0
cleanup() {
  original_rc=$?
  trap - EXIT HUP INT TERM
  if [ "$cleanup_done" -eq 0 ]; then
    cleanup_done=1
    if "$network_down"; then
      cleanup_rc=0
    else
      cleanup_rc=$?
      echo "[run-navmin] ERROR: failed to restore camera network" >&2
    fi
  else
    cleanup_rc=0
  fi

  if [ "$original_rc" -ne 0 ]; then
    exit "$original_rc"
  fi
  exit "$cleanup_rc"
}

"$network_up"

trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

uv run --offline python -m navmin "$@"

#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
network_up="$repo_root/tools/platform/linux/camera_network_up.sh"
network_down="$repo_root/tools/platform/linux/camera_network_down.sh"

fail() {
  echo "[run-navmin] ERROR: $*" >&2
  exit 1
}

[ -x "$network_up" ] || fail "network helper is not executable: $network_up"
[ -x "$network_down" ] || fail "network helper is not executable: $network_down"
command -v uv >/dev/null 2>&1 || fail "uv is not installed or not available in PATH"

cd "$repo_root"

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

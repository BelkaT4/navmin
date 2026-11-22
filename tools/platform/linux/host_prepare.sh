#!/bin/sh
set -eu

case "${NAVMIN_CAMERA_NETWORK_REQUIRED:-0}" in
  0)
    exit 0
    ;;
  1)
    ;;
  *)
    echo "[host-prepare] ERROR: NAVMIN_CAMERA_NETWORK_REQUIRED must be 0 or 1" >&2
    exit 2
    ;;
esac

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"$script_dir/camera_network_down.sh"
exec "$script_dir/camera_network_up.sh"

#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$repo_root/run_navmin.sh" \
  --config "$repo_root/config/day/navmin.json" \
  "$@"

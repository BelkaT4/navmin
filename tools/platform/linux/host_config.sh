# Shared helpers for Linux host configuration.
# This file is sourced by the platform scripts and is not an executable entry point.

navmin_host_config_path() {
  repo_root=$1
  if [ -n "${NAVMIN_HOST_CONFIG:-}" ]; then
    printf '%s\n' "$NAVMIN_HOST_CONFIG"
  else
    printf '%s\n' "$repo_root/config/host.local.env"
  fi
}

navmin_host_config_get() {
  key=$1
  config_file=$2
  awk -v wanted="$key" '
    BEGIN { found = 0 }
    /^[[:space:]]*($|#)/ { next }
    {
      equals = index($0, "=")
      if (equals == 0) {
        next
      }
      name = substr($0, 1, equals - 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
      if (name != wanted) {
        next
      }
      value = substr($0, equals + 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      print value
      found++
    }
    END {
      if (found == 0) {
        exit 1
      }
      if (found > 1) {
        exit 2
      }
    }
  ' "$config_file"
}

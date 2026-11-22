"""Windows host preparation hook for NavMin.

The hook is intentionally a no-op for now. Keeping the platform action in this
script lets Windows host setup grow later without changing the public launcher.
"""

from __future__ import annotations


def main() -> int:
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

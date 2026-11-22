"""Windows host restoration hook for NavMin.

The hook is intentionally a no-op until Windows host setup owns mutable state.
"""

from __future__ import annotations


def main() -> int:
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

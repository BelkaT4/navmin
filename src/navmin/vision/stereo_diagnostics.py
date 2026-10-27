"""Small bounded statistics helpers for stereo runtime diagnostics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil

STEREO_DIAGNOSTIC_WINDOW_SIZE = 256


@dataclass(frozen=True, slots=True)
class BoundedTimingSnapshot:
    """Read-only cumulative/bounded timing summary in nanoseconds."""

    sample_count: int
    retained_sample_count: int
    latest_ns: int | None
    minimum_ns: int | None
    maximum_ns: int | None
    p50_ns: int | None
    p95_ns: int | None


class BoundedTimingStats:
    """Keep cumulative extrema plus a fixed recent window for percentiles."""

    def __init__(self, *, window_size: int = STEREO_DIAGNOSTIC_WINDOW_SIZE) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int):
            raise TypeError("window_size must be an integer")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._window_size = window_size
        self._samples: deque[int] = deque(maxlen=window_size)
        self._sample_count = 0
        self._latest_ns: int | None = None
        self._minimum_ns: int | None = None
        self._maximum_ns: int | None = None

    @property
    def window_size(self) -> int:
        return self._window_size

    def add(self, value_ns: int) -> None:
        if isinstance(value_ns, bool) or not isinstance(value_ns, int):
            raise TypeError("value_ns must be an integer")
        if value_ns < 0:
            raise ValueError("value_ns must be >= 0")
        self._samples.append(value_ns)
        self._sample_count += 1
        self._latest_ns = value_ns
        self._minimum_ns = (
            value_ns if self._minimum_ns is None else min(self._minimum_ns, value_ns)
        )
        self._maximum_ns = (
            value_ns if self._maximum_ns is None else max(self._maximum_ns, value_ns)
        )

    def clear(self) -> None:
        self._samples.clear()
        self._sample_count = 0
        self._latest_ns = None
        self._minimum_ns = None
        self._maximum_ns = None

    def snapshot(self) -> BoundedTimingSnapshot:
        ordered = sorted(self._samples)
        return BoundedTimingSnapshot(
            sample_count=self._sample_count,
            retained_sample_count=len(ordered),
            latest_ns=self._latest_ns,
            minimum_ns=self._minimum_ns,
            maximum_ns=self._maximum_ns,
            p50_ns=_nearest_rank(ordered, 0.50),
            p95_ns=_nearest_rank(ordered, 0.95),
        )


def _nearest_rank(ordered: list[int], percentile: float) -> int | None:
    if not ordered:
        return None
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return ordered[index]


__all__ = [
    "STEREO_DIAGNOSTIC_WINDOW_SIZE",
    "BoundedTimingSnapshot",
    "BoundedTimingStats",
]

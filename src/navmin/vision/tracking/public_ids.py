"""Generation-local public track ID allocation for Vision processors."""

from __future__ import annotations


class PublicTrackIdAllocator:
    """Allocate monotonically increasing IDs within one processor generation."""

    def __init__(self) -> None:
        self._next_id = 1

    @property
    def next_id(self) -> int:
        """Return the next ID without consuming it."""
        return self._next_id

    def allocate(self) -> int:
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def reset(self) -> None:
        """Begin a new generation-local namespace."""
        self._next_id = 1


__all__ = ["PublicTrackIdAllocator"]

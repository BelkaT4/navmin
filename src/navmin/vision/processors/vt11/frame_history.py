"""Bounded exact-frame history retained by the composite VT11 backend."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from navmin.contracts import CameraRole, FramePacket


@dataclass(frozen=True)
class _HistoryEntry:
    frame: FramePacket
    inserted_timestamp_ns: int


class FramePacketHistory:
    """Retain immutable FramePacket references for exact recent-frame lookup."""

    def __init__(self, *, max_frames: int = 12, max_age_ns: int = 400_000_000) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if max_age_ns < 0:
            raise ValueError("max_age_ns must be >= 0")
        self.max_frames = int(max_frames)
        self.max_age_ns = int(max_age_ns)
        self._entries: deque[_HistoryEntry] = deque(maxlen=self.max_frames)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def _prune(self, now_ns: int) -> None:
        while self._entries:
            age_ns = now_ns - self._entries[0].inserted_timestamp_ns
            if age_ns <= self.max_age_ns:
                return
            self._entries.popleft()

    def append(self, frame: FramePacket, *, inserted_timestamp_ns: int) -> None:
        self._prune(inserted_timestamp_ns)
        self._entries.append(
            _HistoryEntry(
                frame=frame,
                inserted_timestamp_ns=inserted_timestamp_ns,
            )
        )

    def find_exact(
        self,
        *,
        camera: CameraRole,
        generation: int,
        frame_id: int,
        now_ns: int,
    ) -> FramePacket | None:
        """Return only the requested identity; never substitute a nearby frame."""
        self._prune(now_ns)
        for entry in self._entries:
            frame = entry.frame
            if (
                frame.camera is camera
                and frame.generation == generation
                and frame.frame_id == frame_id
            ):
                return frame
        return None

    def sequence_from_exact(
        self,
        *,
        camera: CameraRole,
        generation: int,
        frame_id: int,
        now_ns: int,
    ) -> tuple[FramePacket, ...]:
        """Return the exact requested frame followed by newer retained frames."""
        self._prune(now_ns)
        frames = tuple(
            entry.frame
            for entry in self._entries
            if entry.frame.camera is camera
            and entry.frame.generation == generation
        )
        for index, frame in enumerate(frames):
            if frame.frame_id == frame_id:
                return frames[index:]
        return ()


__all__ = ["FramePacketHistory"]

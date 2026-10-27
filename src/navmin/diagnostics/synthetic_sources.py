"""In-process synthetic decoded-frame sources for unified launcher simulation."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic_ns

import numpy as np

from navmin.vision.gstreamer_source import CameraSourceError
from navmin.vision.pipeline import DecodedFrame


class SyntheticFrameSource:
    """Paced one-shot-free decoded source backed by an in-process frame supplier."""

    def __init__(
        self,
        frame_supplier: Callable[[], np.ndarray],
        *,
        fps: int,
        monotonic_clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self._frame_supplier = frame_supplier
        self._period_ns = max(1, round(1_000_000_000 / fps))
        self._clock_ns = monotonic_clock_ns
        self._started = False
        self._next_frame_ns = 0

    @property
    def supports_reconnect(self) -> bool:
        return False

    def start(self) -> None:
        self._started = True
        self._next_frame_ns = self._clock_ns()

    def stop(self) -> None:
        self._started = False

    @property
    def failure(self) -> BaseException | None:
        return None

    def read(self) -> DecodedFrame | None:
        if not self._started:
            return None
        now_ns = self._clock_ns()
        if now_ns < self._next_frame_ns:
            return None
        # Skip missed source periods instead of generating a FIFO backlog.
        missed_periods = max(0, (now_ns - self._next_frame_ns) // self._period_ns)
        self._next_frame_ns += (missed_periods + 1) * self._period_ns
        try:
            frame = self._frame_supplier()
        except Exception as exc:  # supplier boundary must surface as camera failure
            raise CameraSourceError("synthetic frame generation failed") from exc
        if not isinstance(frame, np.ndarray):
            raise CameraSourceError("synthetic frame supplier returned non-array frame")
        return DecodedFrame(
            image=frame,
            capture_id=None,
            receive_timestamp_ns=now_ns,
        )


__all__ = ["SyntheticFrameSource"]

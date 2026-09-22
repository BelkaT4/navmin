"""Legacy Variant 1.4 detector behind the NavMin VisionProcessor boundary."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter_ns

from navmin.contracts import CameraRole, FramePacket, VisionResult
from navmin.vision.tracking import SimpleTracker

from .detector import Legacy14Detector


class Legacy14VisionProcessor:
    """Default baseline processor: Variant 1.4 detector + SimpleTracker."""

    def __init__(
        self,
        *,
        detector: Legacy14Detector | None = None,
        tracker: SimpleTracker | None = None,
        duration_clock_ns: Callable[[], int] = perf_counter_ns,
    ) -> None:
        self._detector = detector or Legacy14Detector()
        self._tracker = tracker or SimpleTracker()
        self._duration_clock_ns = duration_clock_ns
        self._camera: CameraRole | None = None
        self._generation: int | None = None

    def reset(self) -> None:
        self._detector.reset()
        self._tracker.reset()
        self._camera = None
        self._generation = None

    def _ensure_generation(self, frame: FramePacket) -> None:
        if self._camera == frame.camera and self._generation == frame.generation:
            return
        self._detector.reset()
        self._tracker.reset()
        self._camera = frame.camera
        self._generation = frame.generation

    def process(
        self,
        frame: FramePacket,
        *,
        processing_enabled: bool = True,
    ) -> VisionResult:
        started_ns = self._duration_clock_ns()
        self._ensure_generation(frame)
        if processing_enabled:
            detections = self._detector.detect(frame.image)
            tracked_objects = self._tracker.update(
                detections,
                timestamp_ns=frame.receive_timestamp_ns,
            )
        else:
            tracked_objects = ()
        elapsed_ns = max(0, self._duration_clock_ns() - started_ns)
        return VisionResult(
            frame=frame,
            tracked_objects=tuple(tracked_objects),
            processing_time_ns=elapsed_ns,
        )


__all__ = ["Legacy14VisionProcessor"]

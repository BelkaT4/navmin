"""Legacy Variant 1.4 detector behind the NavMin VisionProcessor boundary."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic_ns, perf_counter_ns

from navmin.contracts import (
    BBox,
    CameraRole,
    ClearTargetSelection,
    FramePacket,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetAcquisitionFailureReason,
    TargetLifecycleEvent,
    TargetRef,
    TargetSelectionRequest,
    TrackedObject,
    VisionResult,
    VisionTargetControlCommand,
)
from navmin.vision.processor import VisionProcessorOutput
from navmin.vision.tracking import PublicTrackIdAllocator, SimpleTracker

from .detector import Legacy14Detector

_HISTORY_MAX_FRAMES = 12
_HISTORY_MAX_AGE_NS = 400_000_000


@dataclass(frozen=True)
class _ResultMetadata:
    camera: CameraRole
    generation: int
    frame_id: int
    receive_timestamp_ns: int
    recorded_timestamp_ns: int
    tracked_objects: tuple[TrackedObject, ...]


class Legacy14VisionProcessor:
    """Default baseline processor: Variant 1.4 detector + SimpleTracker."""

    def __init__(
        self,
        *,
        detector: Legacy14Detector | None = None,
        tracker: SimpleTracker | None = None,
        public_track_id_allocator: PublicTrackIdAllocator | None = None,
        duration_clock_ns: Callable[[], int] = perf_counter_ns,
        metadata_clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if tracker is not None and public_track_id_allocator is not None:
            raise ValueError(
                "tracker and public_track_id_allocator cannot both be supplied"
            )
        self._detector = detector or Legacy14Detector()
        self._tracker = tracker or SimpleTracker(
            public_track_id_allocator=public_track_id_allocator
        )
        self._duration_clock_ns = duration_clock_ns
        self._metadata_clock_ns = metadata_clock_ns
        self._camera: CameraRole | None = None
        self._generation: int | None = None
        self._metadata_history: deque[_ResultMetadata] = deque(
            maxlen=_HISTORY_MAX_FRAMES
        )
        self._selected_target: TargetRef | None = None

    def reset(self) -> None:
        self._detector.reset()
        self._tracker.reset()
        self._camera = None
        self._generation = None
        self._metadata_history.clear()
        self._selected_target = None

    def _ensure_generation(self, frame: FramePacket) -> None:
        if self._camera == frame.camera and self._generation == frame.generation:
            return
        self._detector.reset()
        self._tracker.reset()
        self._camera = frame.camera
        self._generation = frame.generation
        self._metadata_history.clear()
        self._selected_target = None

    def _prune_metadata_history(self, now_ns: int) -> None:
        while self._metadata_history:
            age_ns = now_ns - self._metadata_history[0].recorded_timestamp_ns
            if age_ns <= _HISTORY_MAX_AGE_NS:
                return
            self._metadata_history.popleft()

    @staticmethod
    def _contains_point(bbox: BBox, x_px: float, y_px: float) -> bool:
        return (
            bbox.x <= x_px < bbox.x + bbox.width
            and bbox.y <= y_px < bbox.y + bbox.height
        )

    def apply_target_control(
        self,
        command: VisionTargetControlCommand,
    ) -> tuple[TargetLifecycleEvent, ...]:
        if isinstance(command, ClearTargetSelection):
            self._selected_target = None
            return ()
        if not isinstance(command, TargetSelectionRequest):
            raise TypeError("unsupported target-control command")

        self._selected_target = None
        self._prune_metadata_history(self._metadata_clock_ns())
        metadata = next(
            (
                item
                for item in self._metadata_history
                if item.camera is command.camera
                and item.generation == command.generation
                and item.frame_id == command.frame_id
            ),
            None,
        )
        if metadata is None:
            return (
                TargetAcquisitionFailed(
                    selection_id=command.selection_id,
                    camera=command.camera,
                    generation=command.generation,
                    request_frame_id=command.frame_id,
                    reason=TargetAcquisitionFailureReason.STALE_SELECTION,
                    decision_frame_id=None,
                    decision_frame_timestamp_ns=None,
                ),
            )

        candidates = [
            tracked
            for tracked in metadata.tracked_objects
            if self._contains_point(tracked.bbox, command.x_px, command.y_px)
        ]
        if not candidates:
            return (
                TargetAcquisitionFailed(
                    selection_id=command.selection_id,
                    camera=command.camera,
                    generation=command.generation,
                    request_frame_id=command.frame_id,
                    reason=TargetAcquisitionFailureReason.NO_TARGET_AT_POINT,
                    decision_frame_id=metadata.frame_id,
                    decision_frame_timestamp_ns=metadata.receive_timestamp_ns,
                ),
            )

        selected = min(
            candidates,
            key=lambda tracked: (
                tracked.bbox.width * tracked.bbox.height,
                tracked.track_id,
            ),
        )
        target = TargetRef(
            camera=command.camera,
            generation=command.generation,
            track_id=selected.track_id,
        )
        self._selected_target = target
        return (
            TargetAcquired(
                selection_id=command.selection_id,
                target=target,
                request_frame_id=command.frame_id,
                decision_frame_id=metadata.frame_id,
                decision_frame_timestamp_ns=metadata.receive_timestamp_ns,
            ),
        )

    def process(
        self,
        frame: FramePacket,
        *,
        processing_enabled: bool = True,
    ) -> VisionProcessorOutput:
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
        result = VisionResult(
            frame=frame,
            tracked_objects=tuple(tracked_objects),
            processing_time_ns=elapsed_ns,
        )
        recorded_timestamp_ns = self._metadata_clock_ns()
        self._prune_metadata_history(recorded_timestamp_ns)
        self._metadata_history.append(
            _ResultMetadata(
                camera=frame.camera,
                generation=frame.generation,
                frame_id=frame.frame_id,
                receive_timestamp_ns=frame.receive_timestamp_ns,
                recorded_timestamp_ns=recorded_timestamp_ns,
                tracked_objects=result.tracked_objects,
            )
        )
        return VisionProcessorOutput(result=result)


__all__ = ["Legacy14VisionProcessor"]

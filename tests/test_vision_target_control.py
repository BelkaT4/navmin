from __future__ import annotations

from queue import Empty
from threading import Event
from time import monotonic

import numpy as np
import pytest

from navmin.concurrency import (
    InvalidatableLatest,
    TargetLifecycleEventChannel,
)
from navmin.contracts import (
    BBox,
    CameraRay,
    CameraRole,
    ClearTargetSelection,
    FramePacket,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetAcquisitionFailureReason,
    TargetRef,
    TargetSelectionRequest,
    TrackedObject,
    VisionResult,
)
from navmin.vision.camera_worker import CameraWorker
from navmin.vision.pipeline import VisionPipeline
from navmin.vision.processor import VisionProcessorOutput
from navmin.vision.processors.legacy_14.processor import Legacy14VisionProcessor


class _Clock:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


class _NoopDetector:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def detect(self, image: np.ndarray) -> tuple[BBox, ...]:
        del image
        return ()


class _FixedTracker:
    def __init__(self, objects: tuple[TrackedObject, ...]) -> None:
        self.objects = objects
        self.reset_calls = 0
        self.update_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def update(self, detections, *, timestamp_ns: int) -> tuple[TrackedObject, ...]:
        del detections, timestamp_ns
        self.update_calls += 1
        return self.objects


def _tracked(track_id: int, bbox: BBox) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=bbox,
        velocity_x_px_s=0.0,
        velocity_y_px_s=0.0,
        age_frames=2,
    )


def _packet(
    frame_id: int,
    *,
    generation: int = 1,
    receive_timestamp_ns: int | None = None,
) -> FramePacket:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image.flags.writeable = False
    return FramePacket(
        camera=CameraRole.OVERVIEW,
        generation=generation,
        frame_id=frame_id,
        capture_id=None,
        receive_timestamp_ns=(
            frame_id * 10_000_000
            if receive_timestamp_ns is None
            else receive_timestamp_ns
        ),
        image=image,
    )


def _select(
    selection_id: int,
    frame_id: int,
    x_px: float,
    y_px: float,
    *,
    generation: int = 1,
) -> TargetSelectionRequest:
    return TargetSelectionRequest(
        selection_id=selection_id,
        camera=CameraRole.OVERVIEW,
        generation=generation,
        frame_id=frame_id,
        x_px=x_px,
        y_px=y_px,
    )


def _legacy_processor(
    objects: tuple[TrackedObject, ...],
    clock: _Clock,
) -> tuple[Legacy14VisionProcessor, _FixedTracker]:
    tracker = _FixedTracker(objects)
    processor = Legacy14VisionProcessor(
        detector=_NoopDetector(),
        tracker=tracker,
        duration_clock_ns=lambda: 0,
        metadata_clock_ns=clock,
    )
    return processor, tracker


def test_legacy14_exact_frame_selection_returns_immediate_acquired_event() -> None:
    clock = _Clock(100)
    processor, _tracker = _legacy_processor(
        (_tracked(7, BBox(2, 3, 8, 6)),),
        clock,
    )
    processor.process(_packet(4, receive_timestamp_ns=444))

    events = processor.apply_target_control(_select(31, 4, 5.0, 5.0))

    assert events == (
        TargetAcquired(
            selection_id=31,
            target=TargetRef(CameraRole.OVERVIEW, generation=1, track_id=7),
            request_frame_id=4,
            decision_frame_id=4,
            decision_frame_timestamp_ns=444,
        ),
    )


def test_legacy14_exact_frame_point_outside_all_boxes_fails() -> None:
    clock = _Clock(100)
    processor, _tracker = _legacy_processor(
        (_tracked(7, BBox(2, 3, 8, 6)),),
        clock,
    )
    processor.process(_packet(4, receive_timestamp_ns=444))

    (event,) = processor.apply_target_control(_select(32, 4, 15.0, 11.0))

    assert isinstance(event, TargetAcquisitionFailed)
    assert event.reason is TargetAcquisitionFailureReason.NO_TARGET_AT_POINT
    assert event.decision_frame_id == 4
    assert event.decision_frame_timestamp_ns == 444


def test_legacy14_missing_exact_frame_is_stale_without_nearest_fallback() -> None:
    clock = _Clock(100)
    processor, _tracker = _legacy_processor(
        (_tracked(7, BBox(2, 3, 8, 6)),),
        clock,
    )
    processor.process(_packet(5, receive_timestamp_ns=555))

    (event,) = processor.apply_target_control(_select(33, 4, 5.0, 5.0))

    assert isinstance(event, TargetAcquisitionFailed)
    assert event.reason is TargetAcquisitionFailureReason.STALE_SELECTION
    assert event.decision_frame_id is None
    assert event.decision_frame_timestamp_ns is None


def test_legacy14_overlap_prefers_smallest_bbox() -> None:
    clock = _Clock(100)
    processor, _tracker = _legacy_processor(
        (
            _tracked(1, BBox(0, 0, 12, 10)),
            _tracked(9, BBox(2, 2, 4, 4)),
        ),
        clock,
    )
    processor.process(_packet(6))

    (event,) = processor.apply_target_control(_select(34, 6, 3.0, 3.0))

    assert isinstance(event, TargetAcquired)
    assert event.target.track_id == 9


def test_legacy14_equal_area_overlap_prefers_smaller_track_id() -> None:
    clock = _Clock(100)
    processor, _tracker = _legacy_processor(
        (
            _tracked(8, BBox(0, 0, 6, 6)),
            _tracked(3, BBox(2, 2, 6, 6)),
        ),
        clock,
    )
    processor.process(_packet(7))

    (event,) = processor.apply_target_control(_select(35, 7, 3.0, 3.0))

    assert isinstance(event, TargetAcquired)
    assert event.target.track_id == 3


def test_legacy14_clear_is_idempotent_and_preserves_tracker_state() -> None:
    clock = _Clock(100)
    processor, tracker = _legacy_processor(
        (_tracked(7, BBox(2, 3, 8, 6)),),
        clock,
    )
    processor.process(_packet(8))
    assert isinstance(
        processor.apply_target_control(_select(36, 8, 5.0, 5.0))[0],
        TargetAcquired,
    )
    reset_calls = tracker.reset_calls

    clear = ClearTargetSelection(CameraRole.OVERVIEW, generation=1)
    assert processor.apply_target_control(clear) == ()
    assert processor.apply_target_control(clear) == ()
    processor.process(_packet(9))

    assert tracker.reset_calls == reset_calls
    assert tracker.update_calls == 2
    assert isinstance(
        processor.apply_target_control(_select(37, 9, 5.0, 5.0))[0],
        TargetAcquired,
    )


def test_legacy14_history_keeps_only_twelve_frames() -> None:
    clock = _Clock(100)
    processor, _tracker = _legacy_processor(
        (_tracked(7, BBox(2, 3, 8, 6)),),
        clock,
    )
    for frame_id in range(13):
        processor.process(_packet(frame_id))

    old_event = processor.apply_target_control(_select(38, 0, 5.0, 5.0))[0]
    retained_event = processor.apply_target_control(_select(39, 1, 5.0, 5.0))[0]

    assert isinstance(old_event, TargetAcquisitionFailed)
    assert old_event.reason is TargetAcquisitionFailureReason.STALE_SELECTION
    assert isinstance(retained_event, TargetAcquired)


def test_legacy14_history_expires_after_four_hundred_milliseconds() -> None:
    clock = _Clock(1_000_000_000)
    processor, _tracker = _legacy_processor(
        (_tracked(7, BBox(2, 3, 8, 6)),),
        clock,
    )
    processor.process(_packet(10))
    clock.value += 400_000_000
    assert isinstance(
        processor.apply_target_control(_select(40, 10, 5.0, 5.0))[0],
        TargetAcquired,
    )

    clock.value += 1
    expired = processor.apply_target_control(_select(41, 10, 5.0, 5.0))[0]

    assert isinstance(expired, TargetAcquisitionFailed)
    assert expired.reason is TargetAcquisitionFailureReason.STALE_SELECTION


class _CameraModel:
    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        return CameraRay(x_px, y_px, 1.0)


class _IdentityCorrector:
    camera_model = _CameraModel()

    def correct(self, image: np.ndarray) -> np.ndarray:
        corrected = np.array(image, copy=True, order="C")
        corrected.flags.writeable = False
        return corrected


class _RecordingProcessor:
    def __init__(self, *, process_event: bool = False) -> None:
        self.commands = []
        self.process_event = process_event

    def reset(self) -> None:
        return None

    def apply_target_control(self, command):
        self.commands.append(command)
        if isinstance(command, ClearTargetSelection):
            return ()
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

    def process(
        self,
        frame: FramePacket,
        *,
        processing_enabled: bool = True,
    ) -> VisionProcessorOutput:
        del processing_enabled
        result = VisionResult(frame=frame, tracked_objects=(), processing_time_ns=0)
        if not self.process_event:
            return VisionProcessorOutput(result=result)
        target = TargetRef(frame.camera, frame.generation, track_id=5)
        return VisionProcessorOutput(
            result=result,
            target_events=(
                TargetAcquired(
                    selection_id=1,
                    target=target,
                    request_frame_id=frame.frame_id,
                    decision_frame_id=frame.frame_id,
                    decision_frame_timestamp_ns=frame.receive_timestamp_ns,
                ),
            ),
        )


class _RecordingEventChannel(TargetLifecycleEventChannel):
    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def publish(self, event) -> None:
        self._order.append("event")
        super().publish(event)


class _RecordingLatest(InvalidatableLatest[VisionResult]):
    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def publish(self, value: VisionResult) -> None:
        self._order.append("result")
        super().publish(value)


def test_pipeline_publishes_same_frame_events_before_vision_result() -> None:
    order: list[str] = []
    processor = _RecordingProcessor(process_event=True)
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=_IdentityCorrector(),
        processor=processor,
        target_events=_RecordingEventChannel(order),
        latest_result=_RecordingLatest(order),
        timestamp_clock_ns=lambda: 100,
    )
    pipeline.start()
    pipeline.submit_decoded_frame(np.zeros((4, 5, 3), dtype=np.uint8))

    result = pipeline.process_latest()

    assert result is not None
    assert order == ["event", "result"]
    assert isinstance(pipeline.target_events.receive_nowait(), TargetAcquired)


def test_pipeline_publishes_immediate_control_event_and_rejects_old_generation() -> (
    None
):
    processor = _RecordingProcessor()
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=_IdentityCorrector(),
        processor=processor,
    )
    session = pipeline.start()
    stale = _select(50, 0, 1.0, 1.0, generation=session.generation - 1)
    pipeline.target_control.publish(stale)

    assert pipeline.apply_latest_target_control() == ()
    assert processor.commands == []
    with pytest.raises(Empty):
        pipeline.target_events.receive_nowait()

    current = _select(51, 0, 1.0, 1.0, generation=session.generation)
    pipeline.target_control.publish(current)
    events = pipeline.apply_latest_target_control()

    assert processor.commands == [current]
    assert pipeline.target_events.receive_nowait() == events[0]


def test_new_generation_discards_unconsumed_old_control_intent() -> None:
    processor = _RecordingProcessor()
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=_IdentityCorrector(),
        processor=processor,
    )
    first = pipeline.start()
    pipeline.target_control.publish(
        _select(60, 0, 1.0, 1.0, generation=first.generation)
    )

    second = pipeline.start()

    assert second.generation == first.generation + 1
    assert pipeline.apply_latest_target_control() == ()
    assert processor.commands == []


class _PausingEmptySource:
    supports_reconnect = False

    def __init__(self) -> None:
        self.read_entered = Event()
        self.release_read = Event()
        self._first_read = True

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self.release_read.set()

    @property
    def failure(self) -> BaseException | None:
        return None

    def read(self):
        if self._first_read:
            self._first_read = False
            self.read_entered.set()
            if not self.release_read.wait(timeout=2.0):
                raise TimeoutError("test did not release source read")


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    while not predicate():
        if monotonic() >= deadline:
            raise TimeoutError("condition was not met")
        Event().wait(0.005)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (_select(70, 0, 1.0, 1.0), _select(71, 0, 1.0, 1.0)),
        (ClearTargetSelection(CameraRole.OVERVIEW, 1), _select(72, 0, 1.0, 1.0)),
        (_select(73, 0, 1.0, 1.0), ClearTargetSelection(CameraRole.OVERVIEW, 1)),
    ],
)
def test_camera_worker_consumes_only_latest_control_without_waiting_for_frame(
    first,
    second,
) -> None:
    source = _PausingEmptySource()
    processor = _RecordingProcessor()
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=_IdentityCorrector(),
        processor=processor,
    )
    worker = CameraWorker(source=source, pipeline=pipeline, idle_wait_s=0.001)
    worker.start()
    assert source.read_entered.wait(timeout=2.0)

    pipeline.target_control.publish(first)
    pipeline.target_control.publish(second)
    source.release_read.set()
    _wait_until(lambda: processor.commands == [second])

    assert worker.stop(timeout=1.0)
    assert pipeline.latest_result.get() is None

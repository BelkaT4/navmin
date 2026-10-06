from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_args

import numpy as np
import pytest

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
    TargetTrackingEnded,
    TargetTrackingEndReason,
    VisionTargetControlCommand,
)
from navmin.vision import VisionProcessorOutput, create_vision_processor
from navmin.vision.processors.legacy_14.detector import Legacy14Detector
from navmin.vision.processors.legacy_14.processor import Legacy14VisionProcessor
from navmin.vision.tracking import PublicTrackIdAllocator, SimpleTracker


class _FixedDetector:
    def __init__(self, bbox: BBox) -> None:
        self.bbox = bbox
        self.detect_calls = 0
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def detect(self, frame: np.ndarray) -> tuple[BBox, ...]:
        self.detect_calls += 1
        return (self.bbox,)


class _ExplodingTracker:
    def reset(self) -> None:
        pass

    def update(self, detections, *, timestamp_ns):
        raise AssertionError("tracker must not run while processing is disabled")


def packet(
    *,
    generation: int,
    frame_id: int,
    timestamp_ns: int,
    camera: CameraRole = CameraRole.OVERVIEW,
) -> FramePacket:
    image = np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)
    image.flags.writeable = False
    return FramePacket(
        camera=camera,
        generation=generation,
        frame_id=frame_id,
        capture_id=None,
        receive_timestamp_ns=timestamp_ns,
        image=image,
    )


def test_default_processor_is_legacy14() -> None:
    assert isinstance(create_vision_processor(), Legacy14VisionProcessor)


def test_target_control_and_lifecycle_contracts_are_immutable_unions() -> None:
    request = TargetSelectionRequest(
        selection_id=11,
        camera=CameraRole.OVERVIEW,
        generation=2,
        frame_id=7,
        x_px=12.5,
        y_px=14.0,
    )
    clear = ClearTargetSelection(CameraRole.OVERVIEW, generation=2)
    target = TargetRef(CameraRole.OVERVIEW, generation=2, track_id=4)
    acquired = TargetAcquired(
        selection_id=11,
        target=target,
        request_frame_id=7,
        decision_frame_id=7,
        decision_frame_timestamp_ns=1_000,
    )
    failed = TargetAcquisitionFailed(
        selection_id=11,
        camera=CameraRole.OVERVIEW,
        generation=2,
        request_frame_id=7,
        reason=TargetAcquisitionFailureReason.STALE_SELECTION,
        decision_frame_id=None,
        decision_frame_timestamp_ns=None,
    )
    ended = TargetTrackingEnded(
        target=target,
        reason=TargetTrackingEndReason.LOST_UNRECOVERED,
        decision_frame_id=9,
        decision_frame_timestamp_ns=2_000,
    )

    assert set(get_args(VisionTargetControlCommand)) == {
        TargetSelectionRequest,
        ClearTargetSelection,
    }
    assert set(get_args(TargetLifecycleEvent)) == {
        TargetAcquired,
        TargetAcquisitionFailed,
        TargetTrackingEnded,
    }
    for value in (request, clear, acquired, failed, ended):
        with pytest.raises(FrozenInstanceError):
            value.generation = 99


def test_vision_processor_output_is_immutable_and_defaults_to_no_events() -> None:
    frame = packet(generation=1, frame_id=0, timestamp_ns=10)
    result = Legacy14VisionProcessor().process(
        frame,
        processing_enabled=False,
    )

    assert isinstance(result, VisionProcessorOutput)
    assert result.result.frame is frame
    assert result.target_events == ()
    with pytest.raises(FrozenInstanceError):
        result.target_events = ()


def test_processor_returns_exact_frame_tuple_objects_and_does_not_mutate_image() -> None:
    bbox = BBox(3, 4, 5, 6)
    detector = _FixedDetector(bbox)
    processor = Legacy14VisionProcessor(detector=detector, tracker=SimpleTracker())
    first = packet(generation=1, frame_id=0, timestamp_ns=0)
    second = packet(generation=1, frame_id=1, timestamp_ns=1_000_000_000)
    before = second.image.copy()

    assert processor.process(first).result.tracked_objects == ()
    result = processor.process(second).result

    assert result.frame is second
    assert isinstance(result.tracked_objects, tuple)
    assert len(result.tracked_objects) == 1
    assert result.tracked_objects[0].bbox == bbox
    assert result.processing_time_ns >= 0
    assert np.array_equal(second.image, before)
    assert not second.image.flags.writeable


def test_processor_resets_detector_and_tracker_on_generation_change() -> None:
    detector = _FixedDetector(BBox(1, 2, 3, 4))
    processor = Legacy14VisionProcessor(detector=detector, tracker=SimpleTracker())

    assert (
        processor.process(packet(generation=1, frame_id=0, timestamp_ns=0))
        .result.tracked_objects
        == ()
    )
    assert processor.process(
        packet(generation=1, frame_id=1, timestamp_ns=1_000_000_000)
    ).result.tracked_objects[0].track_id == 1

    assert processor.process(
        packet(generation=2, frame_id=0, timestamp_ns=2_000_000_000)
    ).result.tracked_objects == ()
    new_generation = processor.process(
        packet(generation=2, frame_id=1, timestamp_ns=3_000_000_000)
    ).result

    assert new_generation.tracked_objects[0].track_id == 1
    assert detector.reset_calls == 2


def test_legacy_processor_reset_preserves_external_public_id_namespace() -> None:
    allocator = PublicTrackIdAllocator()
    detector = _FixedDetector(BBox(1, 2, 3, 4))
    processor = Legacy14VisionProcessor(
        detector=detector,
        public_track_id_allocator=allocator,
    )

    processor.process(packet(generation=1, frame_id=0, timestamp_ns=0))
    first = processor.process(
        packet(generation=1, frame_id=1, timestamp_ns=1_000_000_000)
    ).result.tracked_objects
    assert [item.track_id for item in first] == [1]

    processor.reset()
    processor.process(packet(generation=1, frame_id=2, timestamp_ns=2_000_000_000))
    second = processor.process(
        packet(generation=1, frame_id=3, timestamp_ns=3_000_000_000)
    ).result.tracked_objects

    assert [item.track_id for item in second] == [2]


def test_processing_disabled_returns_empty_without_running_detector_or_tracker() -> None:
    detector = _FixedDetector(BBox(1, 1, 2, 2))
    processor = Legacy14VisionProcessor(detector=detector, tracker=_ExplodingTracker())
    frame = packet(generation=1, frame_id=0, timestamp_ns=10)

    result = processor.process(frame, processing_enabled=False).result

    assert result.frame is frame
    assert result.tracked_objects == ()
    assert detector.detect_calls == 0


def test_legacy_detector_golden_candidates_match_real_variant_14_algorithm() -> None:
    detector = Legacy14Detector()

    def synthetic_frame(x: int | None = None) -> np.ndarray:
        frame = np.full((240, 320, 3), 160, dtype=np.uint8)
        if x is not None:
            frame[80:98, x : x + 20] = 60
        return frame

    for _ in range(41):
        detector.detect(synthetic_frame())

    sequence = [80, 88, 96, 104, None, 120, 128]
    # Captured from the real legacy source by intercepting the candidates passed
    # to _Variant1FastCentroidTracker.update() on this deterministic sequence.
    expected = [
        (BBox(79, 79, 22, 20),),
        (BBox(87, 79, 22, 20),),
        (BBox(95, 79, 22, 20),),
        (BBox(103, 79, 22, 20),),
        (),
        (BBox(119, 79, 22, 20),),
        (BBox(127, 79, 22, 20),),
    ]

    actual = [detector.detect(synthetic_frame(x)) for x in sequence]

    assert actual == expected

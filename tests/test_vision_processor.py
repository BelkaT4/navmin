from __future__ import annotations

import numpy as np

from navmin.contracts import BBox, CameraRole, FramePacket
from navmin.vision import create_vision_processor
from navmin.vision.processors.legacy_14.detector import Legacy14Detector
from navmin.vision.processors.legacy_14.processor import Legacy14VisionProcessor
from navmin.vision.tracking import SimpleTracker


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


def test_processor_returns_exact_frame_tuple_objects_and_does_not_mutate_image() -> None:
    bbox = BBox(3, 4, 5, 6)
    detector = _FixedDetector(bbox)
    processor = Legacy14VisionProcessor(detector=detector, tracker=SimpleTracker())
    first = packet(generation=1, frame_id=0, timestamp_ns=0)
    second = packet(generation=1, frame_id=1, timestamp_ns=1_000_000_000)
    before = second.image.copy()

    assert processor.process(first).tracked_objects == ()
    result = processor.process(second)

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

    assert processor.process(packet(generation=1, frame_id=0, timestamp_ns=0)).tracked_objects == ()
    assert processor.process(
        packet(generation=1, frame_id=1, timestamp_ns=1_000_000_000)
    ).tracked_objects[0].track_id == 1

    assert processor.process(
        packet(generation=2, frame_id=0, timestamp_ns=2_000_000_000)
    ).tracked_objects == ()
    new_generation = processor.process(
        packet(generation=2, frame_id=1, timestamp_ns=3_000_000_000)
    )

    assert new_generation.tracked_objects[0].track_id == 1
    assert detector.reset_calls == 2


def test_processing_disabled_returns_empty_without_running_detector_or_tracker() -> None:
    detector = _FixedDetector(BBox(1, 1, 2, 2))
    processor = Legacy14VisionProcessor(detector=detector, tracker=_ExplodingTracker())
    frame = packet(generation=1, frame_id=0, timestamp_ns=10)

    result = processor.process(frame, processing_enabled=False)

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

from __future__ import annotations

from hashlib import sha256

import numpy as np
import pytest

from navmin.contracts import (
    BBox,
    CameraRole,
    FramePacket,
    TargetAcquisitionFailed,
    TargetAcquisitionFailureReason,
    TargetSelectionRequest,
)
from navmin.vision.processors.vt11.config.loader import DEFAULT_CONFIG
from navmin.vision.processors.vt11.frame_history import FramePacketHistory
from navmin.vision.processors.vt11.processor import (
    Vt11ProcessorState,
    Vt11VisionProcessor,
)


class _Clock:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


class _FixedDetector:
    def __init__(self, *boxes: BBox) -> None:
        self.boxes = tuple(boxes)
        self.detect_calls = 0
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def detect(self, image: np.ndarray) -> tuple[BBox, ...]:
        del image
        self.detect_calls += 1
        return self.boxes


def _packet(
    frame_id: int,
    *,
    generation: int = 1,
    camera: CameraRole = CameraRole.OVERVIEW,
) -> FramePacket:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image.flags.writeable = False
    return FramePacket(
        camera=camera,
        generation=generation,
        frame_id=frame_id,
        capture_id=None,
        receive_timestamp_ns=frame_id * 10_000_000,
        image=image,
    )


def test_frame_history_uses_exact_identity_without_nearest_fallback() -> None:
    history = FramePacketHistory()
    expected = _packet(10)
    history.append(expected, inserted_timestamp_ns=100)
    history.append(_packet(12), inserted_timestamp_ns=110)

    assert history.find_exact(
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=10,
        now_ns=120,
    ) is expected
    assert history.find_exact(
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=11,
        now_ns=120,
    ) is None
    assert history.find_exact(
        camera=CameraRole.STEREO_LEFT,
        generation=1,
        frame_id=10,
        now_ns=120,
    ) is None
    assert history.find_exact(
        camera=CameraRole.OVERVIEW,
        generation=2,
        frame_id=10,
        now_ns=120,
    ) is None


def test_frame_history_is_bounded_to_twelve_references() -> None:
    history = FramePacketHistory()
    for frame_id in range(13):
        history.append(_packet(frame_id), inserted_timestamp_ns=frame_id)

    assert len(history) == 12
    assert history.find_exact(
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=0,
        now_ns=13,
    ) is None
    assert history.find_exact(
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=1,
        now_ns=13,
    ) is not None


def test_frame_history_expires_only_after_four_hundred_milliseconds() -> None:
    history = FramePacketHistory()
    history.append(_packet(3), inserted_timestamp_ns=100)

    assert history.find_exact(
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=3,
        now_ns=400_000_100,
    ) is not None
    assert history.find_exact(
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=3,
        now_ns=400_000_101,
    ) is None


def test_discovery_delegates_to_legacy_with_shared_public_ids() -> None:
    clock = _Clock()
    processor = Vt11VisionProcessor(profile_name="day", history_clock_ns=clock)
    detector = _FixedDetector(BBox(2, 3, 6, 8))
    processor._legacy._detector = detector

    processor.process(_packet(0))
    discovered = processor.process(_packet(1)).result.tracked_objects
    assert [item.track_id for item in discovered] == [1]

    assert processor._public_track_ids.allocate() == 2
    processor._legacy.reset()
    processor.process(_packet(2))
    rediscovered = processor.process(_packet(3)).result.tracked_objects

    assert [item.track_id for item in rediscovered] == [3]
    assert detector.detect_calls == 4


def test_processor_records_history_in_discovery_state() -> None:
    clock = _Clock()
    processor = Vt11VisionProcessor(profile_name="day", history_clock_ns=clock)

    frame = _packet(0)
    processor.process(frame, processing_enabled=False)

    assert processor._frame_history.find_exact(
        camera=frame.camera,
        generation=frame.generation,
        frame_id=frame.frame_id,
        now_ns=clock.value,
    ) is frame


def test_stale_select_is_rejected_without_current_or_nearest_frame() -> None:
    clock = _Clock(100)
    processor = Vt11VisionProcessor(profile_name="day", history_clock_ns=clock)
    processor.process(_packet(5), processing_enabled=False)
    request = TargetSelectionRequest(
        selection_id=9,
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=4,
        x_px=10.0,
        y_px=11.0,
    )

    events = processor.apply_target_control(request)

    assert len(events) == 1
    assert isinstance(events[0], TargetAcquisitionFailed)
    assert events[0].reason is TargetAcquisitionFailureReason.STALE_SELECTION


def test_processing_disabled_keeps_discovery_algorithm_idle() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    detector = _FixedDetector(BBox(2, 3, 6, 8))
    processor._legacy._detector = detector

    output = processor.process(_packet(0), processing_enabled=False)

    assert output.result.tracked_objects == ()
    assert detector.detect_calls == 0
    assert processor.state is Vt11ProcessorState.DISCOVERY


def test_reset_clears_generation_state_and_restarts_public_ids() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    processor.process(_packet(0), processing_enabled=False)
    assert processor._public_track_ids.allocate() == 1
    processor._state = Vt11ProcessorState.TARGET
    processor._current_public_target_id = 1

    processor.reset()

    assert processor.state is Vt11ProcessorState.DISCOVERY
    assert len(processor._frame_history) == 0
    assert processor._camera is None
    assert processor._generation is None
    assert processor._current_public_target_id is None
    assert processor._public_track_ids.allocate() == 1


def test_new_frame_generation_starts_a_fresh_public_id_namespace() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    detector = _FixedDetector(BBox(2, 3, 6, 8))
    processor._legacy._detector = detector

    processor.process(_packet(0, generation=1))
    first = processor.process(_packet(1, generation=1)).result.tracked_objects
    processor.process(_packet(0, generation=2))
    second = processor.process(_packet(1, generation=2)).result.tracked_objects

    assert [item.track_id for item in first] == [1]
    assert [item.track_id for item in second] == [1]


@pytest.mark.parametrize("profile", ["day", "thermal"])
def test_settings_snapshot_is_resolved_and_read_only(profile: str) -> None:
    processor = Vt11VisionProcessor(profile_name=profile)
    snapshot = processor.settings_snapshot

    assert snapshot.backend == "vt11"
    assert snapshot.profile == profile
    assert snapshot.checkpoint == "VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX"
    assert snapshot.config_identity == "bundled:default.yaml"
    assert snapshot.source_config_sha256 == sha256(DEFAULT_CONFIG.read_bytes()).hexdigest()
    assert snapshot.effective_config["thermal"]["video_mode"] == (
        "visible" if profile == "day" else "thermal"
    )
    assert snapshot.integration_settings == {
        "history_max_frames": 12,
        "history_max_age_ms": 400,
        "timing_history_max_frames": 12,
        "max_retrospective_replay_frames": 6,
    }
    with pytest.raises(TypeError):
        snapshot.integration_settings["history_max_frames"] = 99


def test_processor_rejects_automatic_profile() -> None:
    with pytest.raises(ValueError, match="profile_name"):
        Vt11VisionProcessor(profile_name="auto")
